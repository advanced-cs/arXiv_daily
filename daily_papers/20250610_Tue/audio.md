# 音频 cs.SD;  eess.SP

- **最新发布 36 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Lightweight Joint Audio-Visual Deepfake Detection via Single-Stream Multi-Modal Learning Framework
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频-视频深度伪造检测任务，旨在解决传统方法效率低、冗余的问题。提出一种轻量级单流多模态学习框架，实现高效融合音视频特征。**

- **链接: [http://arxiv.org/pdf/2506.07358v1](http://arxiv.org/pdf/2506.07358v1)**

> **作者:** Kuiyuan Zhang; Wenjie Pei; Rushi Lan; Yifang Guo; Zhongyun Hua
>
> **摘要:** Deepfakes are AI-synthesized multimedia data that may be abused for spreading misinformation. Deepfake generation involves both visual and audio manipulation. To detect audio-visual deepfakes, previous studies commonly employ two relatively independent sub-models to learn audio and visual features, respectively, and fuse them subsequently for deepfake detection. However, this may underutilize the inherent correlations between audio and visual features. Moreover, utilizing two isolated feature learning sub-models can result in redundant neural layers, making the overall model inefficient and impractical for resource-constrained environments. In this work, we design a lightweight network for audio-visual deepfake detection via a single-stream multi-modal learning framework. Specifically, we introduce a collaborative audio-visual learning block to efficiently integrate multi-modal information while learning the visual and audio features. By iteratively employing this block, our single-stream network achieves a continuous fusion of multi-modal features across its layers. Thus, our network efficiently captures visual and audio features without the need for excessive block stacking, resulting in a lightweight network design. Furthermore, we propose a multi-modal classification module that can boost the dependence of the visual and audio classifiers on modality content. It also enhances the whole resistance of the video classifier against the mismatches between audio and visual modalities. We conduct experiments on the DF-TIMIT, FakeAVCeleb, and DFDC benchmark datasets. Compared to state-of-the-art audio-visual joint detection methods, our method is significantly lightweight with only 0.48M parameters, yet it achieves superiority in both uni-modal and multi-modal deepfakes, as well as in unseen types of deepfakes.
>
---
#### [new 002] "In This Environment, As That Speaker": A Text-Driven Framework for Multi-Attribute Speech Conversion
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音转换任务，解决同时控制音色和环境声学的问题。通过文本驱动框架TES-VC实现独立控制，提升生成语音的准确性和可控性。**

- **链接: [http://arxiv.org/pdf/2506.07036v1](http://arxiv.org/pdf/2506.07036v1)**

> **作者:** Jiawei Jin; Zhuhan Yang; Yixuan Zhou; Zhiyong Wu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** We propose TES-VC (Text-driven Environment and Speaker controllable Voice Conversion), a text-driven voice conversion framework with independent control of speaker timbre and environmental acoustics. TES-VC processes simultaneous text inputs for target voice and environment, accurately generating speech matching described timbre/environment while preserving source content. Trained on synthetic data with decoupled vocal/environment features via latent diffusion modeling, our method eliminates interference between attributes. The Retrieval-Based Timbre Control (RBTC) module enables precise manipulation using abstract descriptions without paired data. Experiments confirm TES-VC effectively generates contextually appropriate speech in both timbre and environment with high content retention and superior controllability which demonstrates its potential for widespread applications.
>
---
#### [new 003] Towards a Unified Benchmark for Arabic Pronunciation Assessment: Quranic Recitation as Case Study
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于阿拉伯语发音评估任务，旨在解决MSA发音错误检测问题。工作包括构建统一基准、设计专用音素集和创建首个公开测试集。**

- **链接: [http://arxiv.org/pdf/2506.07722v1](http://arxiv.org/pdf/2506.07722v1)**

> **作者:** Yassine El Kheir; Omnia Ibrahim; Amit Meghanani; Nada Almarwani; Hawau Olamide Toyin; Sadeen Alharbi; Modar Alfadly; Lamya Alkanhal; Ibrahim Selim; Shehab Elbatal; Salima Mdhaffar; Thomas Hain; Yasser Hifny; Mostafa Shahin; Ahmed Ali
>
> **备注:** Accepted Interspeech 2025 and ArabicNLP Shared Task 2025
>
> **摘要:** We present a unified benchmark for mispronunciation detection in Modern Standard Arabic (MSA) using Qur'anic recitation as a case study. Our approach lays the groundwork for advancing Arabic pronunciation assessment by providing a comprehensive pipeline that spans data processing, the development of a specialized phoneme set tailored to the nuances of MSA pronunciation, and the creation of the first publicly available test set for this task, which we term as the Qur'anic Mispronunciation Benchmark (QuranMB.v1). Furthermore, we evaluate several baseline models to provide initial performance insights, thereby highlighting both the promise and the challenges inherent in assessing MSA pronunciation. By establishing this standardized framework, we aim to foster further research and development in pronunciation assessment in Arabic language technology and related applications.
>
---
#### [new 004] LeVo: High-Quality Song Generation with Multi-Preference Alignment
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决歌曲生成中的音质、音乐性和指令遵循问题。提出LeVo框架，结合多偏好对齐方法提升生成质量。**

- **链接: [http://arxiv.org/pdf/2506.07520v1](http://arxiv.org/pdf/2506.07520v1)**

> **作者:** Shun Lei; Yaoxun Xu; Zhiwei Lin; Huaicheng Zhang; Wei Tan; Hangting Chen; Jianwei Yu; Yixuan Zhang; Chenyu Yang; Haina Zhu; Shuai Wang; Zhiyong Wu; Dong Yu
>
> **摘要:** Recent advances in large language models (LLMs) and audio language models have significantly improved music generation, particularly in lyrics-to-song generation. However, existing approaches still struggle with the complex composition of songs and the scarcity of high-quality data, leading to limitations in sound quality, musicality, instruction following, and vocal-instrument harmony. To address these challenges, we introduce LeVo, an LM-based framework consisting of LeLM and a music codec. LeLM is capable of parallelly modeling two types of tokens: mixed tokens, which represent the combined audio of vocals and accompaniment to achieve vocal-instrument harmony, and dual-track tokens, which separately encode vocals and accompaniment for high-quality song generation. It employs two decoder-only transformers and a modular extension training strategy to prevent interference between different token types. To further enhance musicality and instruction following, we introduce a multi-preference alignment method based on Direct Preference Optimization (DPO). This method handles diverse human preferences through a semi-automatic data construction process and DPO post-training. Experimental results demonstrate that LeVo consistently outperforms existing methods on both objective and subjective metrics. Ablation studies further justify the effectiveness of our designs. Audio examples are available at https://levo-demo.github.io/.
>
---
#### [new 005] Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音端点检测任务，旨在提升对话系统的准确性和实时性。通过使用神经音频编解码器特征和标签延迟训练方法，有效减少截断错误并优化响应时间。**

- **链接: [http://arxiv.org/pdf/2506.07081v1](http://arxiv.org/pdf/2506.07081v1)**

> **作者:** Sathvik Udupa; Shinji Watanabe; Petr Schwarz; Jan Cernocky
>
> **摘要:** Accurate, low-latency endpointing is crucial for effective spoken dialogue systems. While traditional endpointers often rely on spectrum-based audio features, this work proposes real-time speech endpointing for multi-turn dialogues using streaming, low-bitrate Neural Audio Codec (NAC) features, building upon recent advancements in neural audio codecs. To further reduce cutoff errors, we introduce a novel label delay training scheme. At a fixed median latency of 160 ms, our combined NAC and label delay approach achieves significant relative cutoff error reductions: 42.7% for a single-stream endpointer and 37.5% for a two-stream configuration, compared to baseline methods. Finally, we demonstrate efficient integration with a codec-based pretrained speech large language model, improving its median response time by 1200 ms and reducing its cutoff error by 35%.
>
---
#### [new 006] RBA-FE: A Robust Brain-Inspired Audio Feature Extractor for Depression Diagnosis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于抑郁症诊断任务，旨在解决音频特征提取中的噪声问题。提出RBA-FE模型，结合改进的神经元模型提升鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.07118v1](http://arxiv.org/pdf/2506.07118v1)**

> **作者:** Yu-Xuan Wu; Ziyan Huang; Bin Hu; Zhi-Hong Guan
>
> **备注:** 14 pages
>
> **摘要:** This article proposes a robust brain-inspired audio feature extractor (RBA-FE) model for depression diagnosis, using an improved hierarchical network architecture. Most deep learning models achieve state-of-the-art performance for image-based diagnostic tasks, ignoring the counterpart audio features. In order to tailor the noise challenge, RBA-FE leverages six acoustic features extracted from the raw audio, capturing both spatial characteristics and temporal dependencies. This hybrid attribute helps alleviate the precision limitation in audio feature extraction within other learning models like deep residual shrinkage networks. To deal with the noise issues, our model incorporates an improved spiking neuron model, called adaptive rate smooth leaky integrate-and-fire (ARSLIF). The ARSLIF model emulates the mechanism of ``retuning of cellular signal selectivity" in the brain attention systems, which enhances the model robustness against environmental noises in audio data. Experimental results demonstrate that RBA-FE achieves state-of-the-art accuracy on the MODMA dataset, respectively with 0.8750, 0.8974, 0.8750 and 0.8750 in precision, accuracy, recall and F1 score. Extensive experiments on the AVEC2014 and DAIC-WOZ datasets both show enhancements in noise robustness. It is further indicated by comparison that the ARSLIF neuron model suggest the abnormal firing pattern within the feature extraction on depressive audio data, offering brain-inspired interpretability.
>
---
#### [new 007] Heart Rate Classification in ECG Signals Using Machine Learning and Deep Learning
- **分类: eess.SP; cs.CV; cs.LG**

- **简介: 该论文属于心电信号分类任务，旨在提高心跳分类的准确性。通过传统机器学习和深度学习方法进行比较，发现手选特征优于图像表示方法。**

- **链接: [http://arxiv.org/pdf/2506.06349v1](http://arxiv.org/pdf/2506.06349v1)**

> **作者:** Thien Nhan Vo; Thanh Xuan Truong
>
> **摘要:** This study addresses the classification of heartbeats from ECG signals through two distinct approaches: traditional machine learning utilizing hand-crafted features and deep learning via transformed images of ECG beats. The dataset underwent preprocessing steps, including downsampling, filtering, and normalization, to ensure consistency and relevance for subsequent analysis. In the first approach, features such as heart rate variability (HRV), mean, variance, and RR intervals were extracted to train various classifiers, including SVM, Random Forest, AdaBoost, LSTM, Bi-directional LSTM, and LightGBM. The second approach involved transforming ECG signals into images using Gramian Angular Field (GAF), Markov Transition Field (MTF), and Recurrence Plots (RP), with these images subsequently classified using CNN architectures like VGG and Inception. Experimental results demonstrate that the LightGBM model achieved the highest performance, with an accuracy of 99% and an F1 score of 0.94, outperforming the image-based CNN approach (F1 score of 0.85). Models such as SVM and AdaBoost yielded significantly lower scores, indicating limited suitability for this task. The findings underscore the superior ability of hand-crafted features to capture temporal and morphological variations in ECG signals compared to image-based representations of individual beats. Future investigations may benefit from incorporating multi-lead ECG signals and temporal dependencies across successive beats to enhance classification accuracy further.
>
---
#### [new 008] SynHate: Detecting Hate Speech in Synthetic Deepfake Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于仇恨言论检测任务，旨在解决合成语音中的仇恨言论识别问题。研究构建了多语言数据集SynHate，并评估了多种模型性能。**

- **链接: [http://arxiv.org/pdf/2506.06772v1](http://arxiv.org/pdf/2506.06772v1)**

> **作者:** Rishabh Ranjan; Kishan Pipariya; Mayank Vatsa; Richa Singh
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** The rise of deepfake audio and hate speech, powered by advanced text-to-speech, threatens online safety. We present SynHate, the first multilingual dataset for detecting hate speech in synthetic audio, spanning 37 languages. SynHate uses a novel four-class scheme: Real-normal, Real-hate, Fake-normal, and Fake-hate. Built from MuTox and ADIMA datasets, it captures diverse hate speech patterns globally and in India. We evaluate five leading self-supervised models (Whisper-small/medium, XLS-R, AST, mHuBERT), finding notable performance differences by language, with Whisper-small performing best overall. Cross-dataset generalization remains a challenge. By releasing SynHate and baseline code, we aim to advance robust, culturally sensitive, and multilingual solutions against synthetic hate speech. The dataset is available at https://www.iab-rubric.org/resources.
>
---
#### [new 009] Insights on Harmonic Tones from a Generative Music Experiment
- **分类: cs.SD; cs.HC; eess.AS; 68T01; J.5**

- **简介: 该论文属于音乐生成任务，探讨AI如何通过谐波音传递多音高，解决人类感知谐波作为独立音高的问题，实验显示AI能生成结构化多声部音乐。**

- **链接: [http://arxiv.org/pdf/2506.07073v1](http://arxiv.org/pdf/2506.07073v1)**

> **作者:** Emmanuel Deruty; Maarten Grachten
>
> **备注:** 15th International Workshop on Machine Learning and Music, September 9, 2024, Vilnius, Lithuania
>
> **摘要:** The ultimate purpose of generative music AI is music production. The studio-lab, a social form within the art-science branch of cross-disciplinarity, is a way to advance music production with AI music models. During a studio-lab experiment involving researchers, music producers, and an AI model for music generating bass-like audio, it was observed that the producers used the model's output to convey two or more pitches with a single harmonic complex tone, which in turn revealed that the model had learned to generate structured and coherent simultaneous melodic lines using monophonic sequences of harmonic complex tones. These findings prompt a reconsideration of the long-standing debate on whether humans can perceive harmonics as distinct pitches and highlight how generative AI can not only enhance musical creativity but also contribute to a deeper understanding of music.
>
---
#### [new 010] A Fast and Lightweight Model for Causal Audio-Visual Speech Separation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频-视觉语音分离任务，旨在解决实时处理问题。提出Swift-Net模型，采用轻量模块和因果机制，提升实时性能。**

- **链接: [http://arxiv.org/pdf/2506.06689v1](http://arxiv.org/pdf/2506.06689v1)**

> **作者:** Wendi Sang; Kai Li; Runxuan Yang; Jianqiang Huang; Xiaolin Hu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Audio-visual speech separation (AVSS) aims to extract a target speech signal from a mixed signal by leveraging both auditory and visual (lip movement) cues. However, most existing AVSS methods exhibit complex architectures and rely on future context, operating offline, which renders them unsuitable for real-time applications. Inspired by the pipeline of RTFSNet, we propose a novel streaming AVSS model, named Swift-Net, which enhances the causal processing capabilities required for real-time applications. Swift-Net adopts a lightweight visual feature extraction module and an efficient fusion module for audio-visual integration. Additionally, Swift-Net employs Grouped SRUs to integrate historical information across different feature spaces, thereby improving the utilization efficiency of historical information. We further propose a causal transformation template to facilitate the conversion of non-causal AVSS models into causal counterparts. Experiments on three standard benchmark datasets (LRS2, LRS3, and VoxCeleb2) demonstrated that under causal conditions, our proposed Swift-Net exhibited outstanding performance, highlighting the potential of this method for processing speech in complex environments.
>
---
#### [new 011] Towards Energy-Efficient and Low-Latency Voice-Controlled Smart Homes: A Proposal for Offline Speech Recognition and IoT Integration
- **分类: cs.SD; cs.CY; eess.AS**

- **简介: 该论文属于智能家庭领域，旨在解决云端语音识别的高延迟和依赖网络的问题。通过引入离线语音识别和分布式物联网架构，提升系统的低延迟、可靠性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.07494v1](http://arxiv.org/pdf/2506.07494v1)**

> **作者:** Peng Huang; Imdad Ullah; Xiaotong Wei; Tariq Ahamed Ahanger; Najm Hassan; Zawar Hussain Shah
>
> **摘要:** The smart home systems, based on AI speech recognition and IoT technology, enable people to control devices through verbal commands and make people's lives more efficient. However, existing AI speech recognition services are primarily deployed on cloud platforms on the Internet. When users issue a command, speech recognition devices like ``Amazon Echo'' will post a recording through numerous network nodes, reach multiple servers, and then receive responses through the Internet. This mechanism presents several issues, including unnecessary energy consumption, communication latency, and the risk of a single-point failure. In this position paper, we propose a smart home concept based on offline speech recognition and IoT technology: 1) integrating offline keyword spotting (KWS) technologies into household appliances with limited resource hardware to enable them to understand user voice commands; 2) designing a local IoT network with decentralized architecture to manage and connect various devices, enhancing the robustness and scalability of the system. This proposal of a smart home based on offline speech recognition and IoT technology will allow users to use low-latency voice control anywhere in the home without depending on the Internet and provide better scalability and energy sustainability.
>
---
#### [new 012] Methods for pitch analysis in contemporary popular music: Vitalic's use of tones that do not operate on the principle of acoustic resonance
- **分类: cs.SD; eess.AS; 00A65; J.5**

- **简介: 该论文属于音乐分析任务，探讨电子音乐中非共振音的使用，分析Vitalic作品中多旋律音的构造及其在流行音乐中的普遍性。**

- **链接: [http://arxiv.org/pdf/2506.07207v1](http://arxiv.org/pdf/2506.07207v1)**

> **作者:** Emmanuel Deruty; Pascal Arbez-Nicolas; David Meredith
>
> **摘要:** Vitalic is an electronic music producer who has been active since 2001. Vitalic's 2005 track "No Fun" features a main synthesiser part built from a sequence of single inharmonic tones that evoke two simultaneous melodies. This part serves as a starting point for examining Vitalic's use of tones that do not operate on the principle of acoustic resonance. The study considers tones that evoke two or more simultaneous pitches and examines various inharmonic partial layouts. Examples outside Vitalic's music are also provided to suggest that similar tone properties can be found elsewhere in contemporary popular music.
>
---
#### [new 013] Technical Report: A Practical Guide to Kaldi ASR Optimization
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升Kaldi ASR系统的准确性与效率。通过优化声学模型、超参数及语言模型，实现更优的识别效果。**

- **链接: [http://arxiv.org/pdf/2506.07149v1](http://arxiv.org/pdf/2506.07149v1)**

> **作者:** Mengze Hong; Di Jiang
>
> **摘要:** This technical report introduces innovative optimizations for Kaldi-based Automatic Speech Recognition (ASR) systems, focusing on acoustic model enhancement, hyperparameter tuning, and language model efficiency. We developed a custom Conformer block integrated with a multistream TDNN-F structure, enabling superior feature extraction and temporal modeling. Our approach includes advanced data augmentation techniques and dynamic hyperparameter optimization to boost performance and reduce overfitting. Additionally, we propose robust strategies for language model management, employing Bayesian optimization and $n$-gram pruning to ensure relevance and computational efficiency. These systematic improvements significantly elevate ASR accuracy and robustness, outperforming existing methods and offering a scalable solution for diverse speech recognition scenarios. This report underscores the importance of strategic optimizations in maintaining Kaldi's adaptability and competitiveness in rapidly evolving technological landscapes.
>
---
#### [new 014] Can Quantized Audio Language Models Perform Zero-Shot Spoofing Detection?
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频安全任务，研究量化对零样本语音欺骗检测的影响。通过实验评估五种模型在不同量化精度下的表现，发现FP16是平衡性能与效率的最佳选择。**

- **链接: [http://arxiv.org/pdf/2506.06756v1](http://arxiv.org/pdf/2506.06756v1)**

> **作者:** Bikash Dutta; Rishabh Ranjan; Shyam Sathvik; Mayank Vatsa; Richa Singh
>
> **备注:** Accepted in Interspeech 2025
>
> **摘要:** Quantization is essential for deploying large audio language models (LALMs) efficiently in resource-constrained environments. However, its impact on complex tasks, such as zero-shot audio spoofing detection, remains underexplored. This study evaluates the zero-shot capabilities of five LALMs, GAMA, LTU-AS, MERaLiON, Qwen-Audio, and SALMONN, across three distinct datasets: ASVspoof2019, In-the-Wild, and WaveFake, and investigates their robustness to quantization (FP32, FP16, INT8). Despite high initial spoof detection accuracy, our analysis demonstrates severe predictive biases toward spoof classification across all models, rendering their practical performance equivalent to random classification. Interestingly, quantization to FP16 precision resulted in negligible performance degradation compared to FP32, effectively halving memory and computational requirements without materially impacting accuracy. However, INT8 quantization intensified model biases, significantly degrading balanced accuracy. These findings highlight critical architectural limitations and emphasize FP16 quantization as an optimal trade-off, providing guidelines for practical deployment and future model refinement.
>
---
#### [new 015] Generative Voice Bursts during Phone Call
- **分类: cs.SD; cs.NE; eess.AS**

- **简介: 该论文属于通信任务，旨在解决紧急信息无法传递的问题。通过生成语音片段，在通话中传递紧急信息，提升通信效率与安全性。**

- **链接: [http://arxiv.org/pdf/2506.07526v1](http://arxiv.org/pdf/2506.07526v1)**

> **作者:** Paritosh Ranjan; Surajit Majumder; Prodip Roy
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** In critical situations, conventional mobile telephony fails to convey emergency voice messages to a callee already engaged in another call. The standard call waiting alert does not provide the urgency or content of the waiting call. This paper proposes a novel method for transmitting Generative Voice Bursts short, context aware audio messages during ongoing calls, from either preauthorized or dynamically prioritized callers. By leveraging generative AI techniques, the system automatically generates spoken messages from contextual inputs example like location, health data, images, background noise when the caller is unable to speak due to incapacitation or environmental constraints. The solution incorporates voice, text, and priority inference mechanisms, allowing high priority emergency messages to bypass conventional call waiting barriers. The approach employs models such as GPT Neo for generative text, which is synthesized into audio and delivered in configurable intervals G seconds and counts N times, ensuring minimal disruption while preserving urgency. This method holds potential for significant impact across telecom, mobile device manufacturing, and emergency communication platforms.
>
---
#### [new 016] An Open-Source Python Framework and Synthetic ECG Image Datasets for Digitization, Lead and Lead Name Detection, and Overlapping Signal Segmentation
- **分类: eess.SP; cs.CV; cs.LG**

- **简介: 该论文提出一个开源Python框架及合成ECG数据集，用于解决ECG图像数字化、导联检测与波形分割任务。**

- **链接: [http://arxiv.org/pdf/2506.06315v1](http://arxiv.org/pdf/2506.06315v1)**

> **作者:** Masoud Rahimi; Reza Karbasi; Abdol-Hossein Vahabie
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** We introduce an open-source Python framework for generating synthetic ECG image datasets to advance critical deep learning-based tasks in ECG analysis, including ECG digitization, lead region and lead name detection, and pixel-level waveform segmentation. Using the PTB-XL signal dataset, our proposed framework produces four open-access datasets: (1) ECG images in various lead configurations paired with time-series signals for ECG digitization, (2) ECG images annotated with YOLO-format bounding boxes for detection of lead region and lead name, (3)-(4) cropped single-lead images with segmentation masks compatible with U-Net-based models in normal and overlapping versions. In the overlapping case, waveforms from neighboring leads are superimposed onto the target lead image, while the segmentation masks remain clean. The open-source Python framework and datasets are publicly available at https://github.com/rezakarbasi/ecg-image-and-signal-dataset and https://doi.org/10.5281/zenodo.15484519, respectively.
>
---
#### [new 017] An introduction to pitch strength in contemporary popular music analysis and production
- **分类: cs.SD; eess.AS; 00A65; J.5**

- **简介: 该论文属于音乐信息检索任务，旨在解决AI模型与音乐制作脱节的问题。研究提出“音高强度”作为低层参数，提升AI在音乐生产中的适用性。**

- **链接: [http://arxiv.org/pdf/2506.07473v1](http://arxiv.org/pdf/2506.07473v1)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2024, Innovation in Music Conference, 14-16 June, 2024, Kristiania University College, Oslo, Norway
>
> **摘要:** Music information retrieval distinguishes between low- and high-level descriptions of music. Current generative AI models rely on text descriptions that are higher level than the controls familiar to studio musicians. Pitch strength, a low-level perceptual parameter of contemporary popular music, may be one feature that could make such AI models more suited to music production. Signal and perceptual analyses suggest that pitch strength (1) varies significantly across and inside songs; (2) contributes to both small- and large-scale structure; (3) contributes to the handling of polyphonic dissonance; and (4) may be a feature of upper harmonics made audible in a perspective of perceptual richness.
>
---
#### [new 018] Towards Generalized Source Tracing for Codec-Based Deepfake Speech
- **分类: cs.SD; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于语音源追踪任务，解决CodecFake生成语音的源追踪问题。通过引入SASTNet模型，结合语义与声学特征，提升追踪性能。**

- **链接: [http://arxiv.org/pdf/2506.07294v1](http://arxiv.org/pdf/2506.07294v1)**

> **作者:** Xuanjun Chen; I-Ming Lin; Lin Zhang; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Submitted to IEEE ASRU 2025
>
> **摘要:** Recent attempts at source tracing for codec-based deepfake speech (CodecFake), generated by neural audio codec-based speech generation (CoSG) models, have exhibited suboptimal performance. However, how to train source tracing models using simulated CoSG data while maintaining strong performance on real CoSG-generated audio remains an open challenge. In this paper, we show that models trained solely on codec-resynthesized data tend to overfit to non-speech regions and struggle to generalize to unseen content. To mitigate these challenges, we introduce the Semantic-Acoustic Source Tracing Network (SASTNet), which jointly leverages Whisper for semantic feature encoding and Wav2vec2 with AudioMAE for acoustic feature encoding. Our proposed SASTNet achieves state-of-the-art performance on the CoSG test set of the CodecFake+ dataset, demonstrating its effectiveness for reliable source tracing.
>
---
#### [new 019] Speech Recognition on TV Series with Video-guided Post-Correction
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决电视节目ASR准确性问题。通过视频引导的后校正框架，结合视频上下文信息提升转录精度。**

- **链接: [http://arxiv.org/pdf/2506.07323v1](http://arxiv.org/pdf/2506.07323v1)**

> **作者:** Haoyuan Yang; Yue Zhang; Liqiang Jing
>
> **摘要:** Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments.
>
---
#### [new 020] Audio synthesizer inversion in symmetric parameter spaces with approximately equivariant flow matching
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于音频合成器参数逆问题，解决因对称性导致的多解性问题。通过生成模型和等变流匹配提升逆过程性能。**

- **链接: [http://arxiv.org/pdf/2506.07199v1](http://arxiv.org/pdf/2506.07199v1)**

> **作者:** Ben Hayes; Charalampos Saitis; György Fazekas
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Many audio synthesizers can produce the same signal given different parameter configurations, meaning the inversion from sound to parameters is an inherently ill-posed problem. We show that this is largely due to intrinsic symmetries of the synthesizer, and focus in particular on permutation invariance. First, we demonstrate on a synthetic task that regressing point estimates under permutation symmetry degrades performance, even when using a permutation-invariant loss function or symmetry-breaking heuristics. Then, viewing equivalent solutions as modes of a probability distribution, we show that a conditional generative model substantially improves performance. Further, acknowledging the invariance of the implicit parameter distribution, we find that performance is further improved by using a permutation equivariant continuous normalizing flow. To accommodate intricate symmetries in real synthesizers, we also propose a relaxed equivariance strategy that adaptively discovers relevant symmetries from data. Applying our method to Surge XT, a full-featured open source synthesizer used in real world audio production, we find our method outperforms regression and generative baselines across audio reconstruction metrics.
>
---
#### [new 021] Benchmarking Early Agitation Prediction in Community-Dwelling People with Dementia Using Multimodal Sensors and Machine Learning
- **分类: eess.SP; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于行为预测任务，旨在通过多模态传感器和机器学习早期预测痴呆患者躁动，提升照护效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.06306v1](http://arxiv.org/pdf/2506.06306v1)**

> **作者:** Ali Abedi; Charlene H. Chu; Shehroz S. Khan
>
> **备注:** 16 pages, 4 figures, 2 tables
>
> **摘要:** Agitation is one of the most common responsive behaviors in people living with dementia, particularly among those residing in community settings without continuous clinical supervision. Timely prediction of agitation can enable early intervention, reduce caregiver burden, and improve the quality of life for both patients and caregivers. This study aimed to develop and benchmark machine learning approaches for the early prediction of agitation in community-dwelling older adults with dementia using multimodal sensor data. A new set of agitation-related contextual features derived from activity data was introduced and employed for agitation prediction. A wide range of machine learning and deep learning models was evaluated across multiple problem formulations, including binary classification for single-timestamp tabular sensor data and multi-timestamp sequential sensor data, as well as anomaly detection for single-timestamp tabular sensor data. The study utilized the Technology Integrated Health Management (TIHM) dataset, the largest publicly available dataset for remote monitoring of people living with dementia, comprising 2,803 days of in-home activity, physiology, and sleep data. The most effective setting involved binary classification of sensor data using the current 6-hour timestamp to predict agitation at the subsequent timestamp. Incorporating additional information, such as time of day and agitation history, further improved model performance, with the highest AUC-ROC of 0.9720 and AUC-PR of 0.4320 achieved by the light gradient boosting machine. This work presents the first comprehensive benchmarking of state-of-the-art techniques for agitation prediction in community-based dementia care using privacy-preserving sensor data. The approach enables accurate, explainable, and efficient agitation prediction, supporting proactive dementia care and aging in place.
>
---
#### [new 022] Deep Inertial Pose: A deep learning approach for human pose estimation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于人体姿态估计任务，旨在用深度学习替代传统复杂方法，提升姿态估计的准确性与成本效益。**

- **链接: [http://arxiv.org/pdf/2506.06850v1](http://arxiv.org/pdf/2506.06850v1)**

> **作者:** Sara M. Cerqueira; Manuel Palermo; Cristina P. Santos
>
> **摘要:** Inertial-based Motion capture system has been attracting growing attention due to its wearability and unsconstrained use. However, accurate human joint estimation demands several complex and expertise demanding steps, which leads to expensive software such as the state-of-the-art MVN Awinda from Xsens Technologies. This work aims to study the use of Neural Networks to abstract the complex biomechanical models and analytical mathematics required for pose estimation. Thus, it presents a comparison of different Neural Network architectures and methodologies to understand how accurately these methods can estimate human pose, using both low cost(MPU9250) and high end (Mtw Awinda) Magnetic, Angular Rate, and Gravity (MARG) sensors. The most efficient method was the Hybrid LSTM-Madgwick detached, which achieved an Quaternion Angle distance error of 7.96, using Mtw Awinda data. Also, an ablation study was conducted to study the impact of data augmentation, output representation, window size, loss function and magnetometer data on the pose estimation error. This work indicates that Neural Networks can be trained to estimate human pose, with results comparable to the state-of-the-art fusion filters.
>
---
#### [new 023] Bridging Audio and Vision: Zero-Shot Audiovisual Segmentation by Connecting Pretrained Models
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于音频视觉分割任务，解决传统方法依赖大量标注数据的问题。通过融合预训练模型，实现无需特定标注的零样本分割。**

- **链接: [http://arxiv.org/pdf/2506.06537v1](http://arxiv.org/pdf/2506.06537v1)**

> **作者:** Seung-jae Lee; Paul Hongsuck Seo
>
> **备注:** Accepted on INTERSPEECH2025
>
> **摘要:** Audiovisual segmentation (AVS) aims to identify visual regions corresponding to sound sources, playing a vital role in video understanding, surveillance, and human-computer interaction. Traditional AVS methods depend on large-scale pixel-level annotations, which are costly and time-consuming to obtain. To address this, we propose a novel zero-shot AVS framework that eliminates task-specific training by leveraging multiple pretrained models. Our approach integrates audio, vision, and text representations to bridge modality gaps, enabling precise sound source segmentation without AVS-specific annotations. We systematically explore different strategies for connecting pretrained models and evaluate their efficacy across multiple datasets. Experimental results demonstrate that our framework achieves state-of-the-art zero-shot AVS performance, highlighting the effectiveness of multimodal model integration for finegrained audiovisual segmentation.
>
---
#### [new 024] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音领域，解决语音基础模型在实际部署中因声学域变化导致的性能下降问题。提出E-BATS框架，实现高效无反向传播的测试时适应。**

- **链接: [http://arxiv.org/pdf/2506.07078v1](http://arxiv.org/pdf/2506.07078v1)**

> **作者:** Jiaheng Dong; Hong Jia; Soumyajit Chatterjee; Abhirup Ghosh; James Bailey; Ting Dang
>
> **备注:** Under Review
>
> **摘要:** Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4.1%-13.5% accuracy gains over backpropagation-free baselines and 2.0-6.4 times GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments.
>
---
#### [new 025] Non-Intrusive Load Monitoring Based on Image Load Signatures and Continual Learning
- **分类: cs.LG; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于非侵入式负载监测任务，旨在解决传统方法在特征鲁棒性和模型泛化上的不足。通过图像负载特征和持续学习提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.06637v1](http://arxiv.org/pdf/2506.06637v1)**

> **作者:** Olimjon Toirov; Wei Yu
>
> **备注:** 10 pages, 3 figures, 2025 2nd International Conference on Digital Society and Artificial Intelligence (DSAI 2025), Conference dates: May 23-25, 2025
>
> **摘要:** Non-Intrusive Load Monitoring (NILM) identifies the operating status and energy consumption of each electrical device in the circuit by analyzing the electrical signals at the bus, which is of great significance for smart power management. However, the complex and changeable load combinations and application environments lead to the challenges of poor feature robustness and insufficient model generalization of traditional NILM methods. To this end, this paper proposes a new non-intrusive load monitoring method that integrates "image load signature" and continual learning. This method converts multi-dimensional power signals such as current, voltage, and power factor into visual image load feature signatures, and combines deep convolutional neural networks to realize the identification and classification of multiple devices; at the same time, self-supervised pre-training is introduced to improve feature generalization, and continual online learning strategies are used to overcome model forgetting to adapt to the emergence of new loads. This paper conducts a large number of experiments on high-sampling rate load datasets, and compares a variety of existing methods and model variants. The results show that the proposed method has achieved significant improvements in recognition accuracy.
>
---
#### [new 026] Automatic Speech Recognition of African American English: Lexical and Contextual Effects
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别任务，研究AAE中CCR和ING-reduction对ASR的影响，分析无LM系统受词汇邻近效应影响更大。**

- **链接: [http://arxiv.org/pdf/2506.06888v1](http://arxiv.org/pdf/2506.06888v1)**

> **作者:** Hamid Mojarad; Kevin Tang
>
> **备注:** submitted to Interspeech 2025
>
> **摘要:** Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
>
---
#### [new 027] CAtCh: Cognitive Assessment through Cookie Thief
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于认知障碍预测任务，旨在通过语音分析识别更广泛的认知损伤。研究比较了多种方法，发现多模态和声学特征优于语言模型。**

- **链接: [http://arxiv.org/pdf/2506.06603v1](http://arxiv.org/pdf/2506.06603v1)**

> **作者:** Joseph T Colonel; Carolyn Hagler; Guiselle Wismer; Laura Curtis; Jacqueline Becker; Juan Wisnivesky; Alex Federman; Gaurav Pandey
>
> **摘要:** Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at https://github.com/JTColonel/catch.
>
---
#### [new 028] Neural Spectral Band Generation for Audio Coding
- **分类: eess.AS; cs.AI; eess.SP**

- **简介: 该论文属于音频编码任务，解决带宽受限音频的高频重建问题。通过结合DNN提取侧信息与带宽扩展，提出非盲参数化方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.06732v1](http://arxiv.org/pdf/2506.06732v1)**

> **作者:** Woongjib Choi; Byeong Hyeon Kim; Hyungseob Lim; Inseon Jang; Hong-Goo Kang
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Audio bandwidth extension is the task of reconstructing missing high frequency components of bandwidth-limited audio signals, where bandwidth limitation is a common issue for audio signals due to several reasons, including channel capacity and data constraints. While conventional spectral band replication is a well-established parametric approach to audio bandwidth extension, the SBR usually entails coarse feature extraction and reconstruction techniques, which leads to limitations when processing various types of audio signals. In parallel, numerous deep neural network-based audio bandwidth extension methods have been proposed. These DNN-based methods are usually referred to as blind BWE, as these methods do not rely on prior information extracted from original signals, and only utilize given low frequency band signals to estimate missing high frequency components. In order to replace conventional SBR with DNNs, simply adopting existing DNN-based methodologies results in suboptimal performance due to the blindness of these methods. My proposed research suggests a new approach to parametric non-blind bandwidth extension, as DNN-based side information extraction and DNN-based bandwidth extension are performed only at the front and end of the audio coding pipeline.
>
---
#### [new 029] W4S4: WaLRUS Meets S4 for Long-Range Sequence Modeling
- **分类: cs.LG; eess.AS; eess.IV; eess.SP**

- **简介: 该论文属于序列建模任务，旨在解决长程依赖建模问题。提出W4S4模型，利用小波框架改进状态空间模型，提升信息保留能力与计算效率。**

- **链接: [http://arxiv.org/pdf/2506.07920v1](http://arxiv.org/pdf/2506.07920v1)**

> **作者:** Hossein Babaei; Mel White; Richard G. Baraniuk
>
> **备注:** 10 pages, 2 figures, 3 tables
>
> **摘要:** State Space Models (SSMs) have emerged as powerful components for sequence modeling, enabling efficient handling of long-range dependencies via linear recurrence and convolutional computation. However, their effectiveness depends heavily on the choice and initialization of the state matrix. In this work, we build on the SaFARi framework and existing WaLRUS SSMs to introduce a new variant, W4S4 (WaLRUS for S4), a new class of SSMs constructed from redundant wavelet frames. WaLRUS admits a stable diagonalization and supports fast kernel computation without requiring low-rank approximations, making it both theoretically grounded and computationally efficient. We show that WaLRUS retains information over long horizons significantly better than HiPPO-based SSMs, both in isolation and when integrated into deep architectures such as S4. Our experiments demonstrate consistent improvements across delay reconstruction tasks, classification benchmarks, and long-range sequence modeling, confirming that high-quality, structured initialization enabled by wavelet-based state dynamic offers substantial advantages over existing alternatives. WaLRUS provides a scalable and versatile foundation for the next generation of deep SSM-based models.
>
---
#### [new 030] Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决情感理解缺乏解释性的问题。通过引入多任务框架和生成式推理，提升情感预测的准确性和解释性。**

- **链接: [http://arxiv.org/pdf/2506.06820v1](http://arxiv.org/pdf/2506.06820v1)**

> **作者:** Wenyu Zhang; Yingxu He; Geyu Lin; Zhuohan Liu; Shuo Sun; Bin Wang; Xunlong Zou; Jeremy H. M. Wong; Qiongqiong Wang; Hardik B. Sailor; Nancy F. Chen; Ai Ti Aw
>
> **摘要:** Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses.
>
---
#### [new 031] Accurate analysis of the pitch pulse-based magnitude/phase structure of natural vowels and assessment of three lightweight time/frequency voicing restoration methods
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音处理任务，旨在恢复 whispered speech 的 voicing 信息。通过分析语音的相位/幅度结构，并评估三种合成 voicing 方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.06675v1](http://arxiv.org/pdf/2506.06675v1)**

> **作者:** Aníbal J. S. Ferreira; Luis M. T. Jesus; Laurentino M. M. Leal; Jorge E. F. Spratley
>
> **备注:** 58 pages, 17 figures, 8 tables
>
> **摘要:** Whispered speech is produced when the vocal folds are not used, either intentionally, or due to a temporary or permanent voice condition. The essential difference between natural speech and whispered speech is that periodic signal components that exist in certain regions of the former, called voiced regions, as a consequence of the vibration of the vocal folds, are missing in the latter. The restoration of natural speech from whispered speech requires delicate signal processing procedures that are especially useful if they can be implemented on low-resourced portable devices, in real-time, and on-the-fly, taking advantage of the established source-filter paradigm of voice production and related models. This paper addresses two challenges that are intertwined and are key in informing and making viable this envisioned technological realization. The first challenge involves characterizing and modeling the evolution of the harmonic phase/magnitude structure of a sequence of individual pitch periods in a voiced region of natural speech comprising sustained or co-articulated vowels. This paper proposes a novel algorithm segmenting individual pitch pulses, which is then used to obtain illustrative results highlighting important differences between sustained and co-articulated vowels, and suggesting practical synthetic voicing approaches. The second challenge involves model-based synthetic voicing. Three implementation alternatives are described that differ in their signal reconstruction approaches: frequency-domain, combined frequency and time-domain, and physiologically-inspired separate filtering of glottal excitation pulses individually generated. The three alternatives are compared objectively using illustrative examples, and subjectively using the results of listening tests involving synthetic voicing of sustained and co-articulated vowels in word context.
>
---
#### [new 032] SPC to 3D: Novel View Synthesis from Binary SPC via I2I translation
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 该论文属于图像生成任务，旨在解决SPC图像信息丢失导致的3D合成难题。通过两阶段框架，将二值SPC图像转换为高质量彩色新视角。**

- **链接: [http://arxiv.org/pdf/2506.06890v1](http://arxiv.org/pdf/2506.06890v1)**

> **作者:** Sumit Sharma; Gopi Raju Matta; Kaushik Mitra
>
> **备注:** Accepted for publication at ICIP 2025
>
> **摘要:** Single Photon Avalanche Diodes (SPADs) represent a cutting-edge imaging technology, capable of detecting individual photons with remarkable timing precision. Building on this sensitivity, Single Photon Cameras (SPCs) enable image capture at exceptionally high speeds under both low and high illumination. Enabling 3D reconstruction and radiance field recovery from such SPC data holds significant promise. However, the binary nature of SPC images leads to severe information loss, particularly in texture and color, making traditional 3D synthesis techniques ineffective. To address this challenge, we propose a modular two-stage framework that converts binary SPC images into high-quality colorized novel views. The first stage performs image-to-image (I2I) translation using generative models such as Pix2PixHD, converting binary SPC inputs into plausible RGB representations. The second stage employs 3D scene reconstruction techniques like Neural Radiance Fields (NeRF) or Gaussian Splatting (3DGS) to generate novel views. We validate our two-stage pipeline (Pix2PixHD + Nerf/3DGS) through extensive qualitative and quantitative experiments, demonstrating significant improvements in perceptual quality and geometric consistency over the alternative baseline.
>
---
#### [new 033] Multimodal Spatial Language Maps for Robot Navigation and Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于机器人导航与操作任务，旨在解决语言与环境空间映射不精确的问题。通过构建多模态空间语言地图，融合视觉、音频和语言信息，提升机器人对多模态指令的定位与导航能力。**

- **链接: [http://arxiv.org/pdf/2506.06862v1](http://arxiv.org/pdf/2506.06862v1)**

> **作者:** Chenguang Huang; Oier Mees; Andy Zeng; Wolfram Burgard
>
> **备注:** accepted to International Journal of Robotics Research (IJRR). 24 pages, 18 figures. The paper contains texts from VLMaps(arXiv:2210.05714) and AVLMaps(arXiv:2303.07522). The project page is https://mslmaps.github.io/
>
> **摘要:** Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues.
>
---
#### [new 034] TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型训练任务，旨在解决依赖大量语音文本数据的问题。通过统一编码器和轻量投影网络，仅用文本训练模型，实现语音相关任务的高性能。**

- **链接: [http://arxiv.org/pdf/2506.06343v1](http://arxiv.org/pdf/2506.06343v1)**

> **作者:** Taesoo Kim; Jong Hwan Ko
>
> **摘要:** Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data.
>
---
#### [new 035] Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音标注任务，旨在提高日语语音合成数据的准确性。通过微调ASR模型并结合词典知识，提升音素和韵律标注质量。**

- **链接: [http://arxiv.org/pdf/2506.07646v1](http://arxiv.org/pdf/2506.07646v1)**

> **作者:** Rui Hu; Xiaolong Lin; Jiawang Liu; Shixi Huang; Zhenpeng Zhan
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this paper, we propose a method for annotating phonemic and prosodic labels on a given audio-transcript pair, aimed at constructing Japanese text-to-speech (TTS) datasets. Our approach involves fine-tuning a large-scale pre-trained automatic speech recognition (ASR) model, conditioned on ground truth transcripts, to simultaneously output phrase-level graphemes and annotation labels. To further correct errors in phonemic labeling, we employ a decoding strategy that utilizes dictionary prior knowledge. The objective evaluation results demonstrate that our proposed method outperforms previous approaches relying solely on text or audio. The subjective evaluation results indicate that the naturalness of speech synthesized by the TTS model, trained with labels annotated using our method, is comparable to that of a model trained with manual annotations.
>
---
#### [new 036] Speaker-Distinguishable CTC: Learning Speaker Distinction Using CTC for Multi-Talker Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，旨在解决Speaker Assignment失败导致的识别错误问题。提出SD-CTC框架，联合分配音素和说话人标签，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.07515v1](http://arxiv.org/pdf/2506.07515v1)**

> **作者:** Asahi Sakuma; Hiroaki Sato; Ryuga Sugano; Tadashi Kumano; Yoshihiko Kawai; Tetsuji Ogawa
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** This paper presents a novel framework for multi-talker automatic speech recognition without the need for auxiliary information. Serialized Output Training (SOT), a widely used approach, suffers from recognition errors due to speaker assignment failures. Although incorporating auxiliary information, such as token-level timestamps, can improve recognition accuracy, extracting such information from natural conversational speech remains challenging. To address this limitation, we propose Speaker-Distinguishable CTC (SD-CTC), an extension of CTC that jointly assigns a token and its corresponding speaker label to each frame. We further integrate SD-CTC into the SOT framework, enabling the SOT model to learn speaker distinction using only overlapping speech and transcriptions. Experimental comparisons show that multi-task learning with SD-CTC and SOT reduces the error rate of the SOT model by 26% and achieves performance comparable to state-of-the-art methods relying on auxiliary information.
>
---
## 更新

#### [replaced 001] Token Communications: A Unified Framework for Cross-modal Context-aware Semantic Communications
- **分类: cs.IT; cs.CV; cs.MM; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.12096v2](http://arxiv.org/pdf/2502.12096v2)**

> **作者:** Li Qiao; Mahdi Boloursaz Mashhadi; Zhen Gao; Rahim Tafazolli; Mehdi Bennis; Dusit Niyato
>
> **备注:** Submitted to IEEE Journals
>
> **摘要:** In this paper, we introduce token communications (TokCom), a large model-driven framework to leverage cross-modal context information in generative semantic communications (GenSC). TokCom is a new paradigm, motivated by the recent success of generative foundation models and multimodal large language models (GFM/MLLMs), where the communication units are tokens, enabling efficient transformer-based token processing at the transmitter and receiver. In this paper, we introduce the potential opportunities and challenges of leveraging context in GenSC, explore how to integrate GFM/MLLMs-based token processing into semantic communication systems to leverage cross-modal context effectively at affordable complexity, present the key principles for efficient TokCom at various layers in future wireless networks. In a typical image semantic communication setup, we demonstrate a significant improvement of the bandwidth efficiency, achieved by TokCom by leveraging the context information among tokens. Finally, the potential research directions are identified to facilitate adoption of TokCom in future wireless networks.
>
---
#### [replaced 002] Inter-Speaker Relative Cues for Text-Guided Target Speech Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.01483v3](http://arxiv.org/pdf/2506.01483v3)**

> **作者:** Wang Dai; Archontis Politis; Tuomas Virtanen
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We propose a novel approach that utilizes inter-speaker relative cues to distinguish target speakers and extract their voices from mixtures. Continuous cues (e.g., temporal order, age, pitch level) are grouped by relative differences, while discrete cues (e.g., language, gender, emotion) retain their categorical distinctions. Compared to fixed speech attribute classification, inter-speaker relative cues offer greater flexibility, facilitating much easier expansion of text-guided target speech extraction datasets. Our experiments show that combining all relative cues yields better performance than random subsets, with gender and temporal order being the most robust across languages and reverberant conditions. Additional cues, such as pitch level, loudness, distance, speaking duration, language, and pitch range, also demonstrate notable benefits in complex scenarios. Fine-tuning pre-trained WavLM Base+ CNN encoders improves overall performance over the Conv1d baseline.
>
---
#### [replaced 003] Training Articulatory Inversion Models for Interspeaker Consistency
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20529v3](http://arxiv.org/pdf/2505.20529v3)**

> **作者:** Charles McGhee; Mark J. F. Gales; Kate M. Knill
>
> **摘要:** Acoustic-to-Articulatory Inversion (AAI) attempts to model the inverse mapping from speech to articulation. Exact articulatory prediction from speech alone may be impossible, as speakers can choose different forms of articulation seemingly without reference to their vocal tract structure. However, once a speaker has selected an articulatory form, their productions vary minimally. Recent works in AAI have proposed adapting Self-Supervised Learning (SSL) models to single-speaker datasets, claiming that these single-speaker models provide a universal articulatory template. In this paper, we investigate whether SSL-adapted models trained on single and multi-speaker data produce articulatory targets which are consistent across speaker identities for English and Russian. We do this through the use of a novel evaluation method which extracts articulatory targets using minimal pair sets. We also present a training method which can improve interspeaker consistency using only speech data.
>
---
#### [replaced 004] Imagine to Hear: Auditory Knowledge Generation can be an Effective Assistant for Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16853v2](http://arxiv.org/pdf/2503.16853v2)**

> **作者:** Suho Yoo; Hyunjong Ok; Jaeho Lee
>
> **备注:** 12 pages, 5 figures, ACL Findings 2025
>
> **摘要:** Language models pretrained on text-only corpora often struggle with tasks that require auditory commonsense knowledge. Previous work addresses this problem by augmenting the language model to retrieve knowledge from external audio databases. This approach has several limitations, such as the potential lack of relevant audio in databases and the high costs associated with constructing the databases. To address these issues, we propose Imagine to Hear, a novel approach that dynamically generates auditory knowledge using generative models. Our framework detects multiple audio-related textual spans from the given prompt and generates corresponding auditory knowledge. We develop several mechanisms to efficiently process multiple auditory knowledge, including a CLAP-based rejection sampler and a language-audio fusion module. Our experiments show that our method achieves state-of-the-art performance on AuditoryBench without relying on external databases, highlighting the effectiveness of our generation-based approach.
>
---
#### [replaced 005] Towards Achieving Perfect Multimodal Alignment
- **分类: cs.LG; cs.AI; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.15352v2](http://arxiv.org/pdf/2503.15352v2)**

> **作者:** Abhi Kamboj; Minh N. Do
>
> **摘要:** Multimodal alignment constructs a joint latent vector space where modalities representing the same concept map to neighboring latent vectors. We formulate this as an inverse problem and show that, under certain conditions, paired data from each modality can map to equivalent latent vectors, which we refer to as perfect alignment. When perfect alignment cannot be achieved, it can be approximated using the Singular Value Decomposition (SVD) of a multimodal data matrix. Experiments on synthetic multimodal Gaussian data verify the effectiveness of our perfect alignment method compared to a learned contrastive alignment method. We further demonstrate the practical application of cross-modal transfer for human action recognition, showing that perfect alignment significantly enhances the model's accuracy. We conclude by discussing how these findings can be applied to various modalities and tasks and the limitations of our method. We hope these findings inspire further exploration of perfect alignment and its applications in representation learning.
>
---
#### [replaced 006] FLAM: Frame-Wise Language-Audio Modeling
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.05335v2](http://arxiv.org/pdf/2505.05335v2)**

> **作者:** Yusong Wu; Christos Tsirigotis; Ke Chen; Cheng-Zhi Anna Huang; Aaron Courville; Oriol Nieto; Prem Seetharaman; Justin Salamon
>
> **备注:** Accepted at ICML 2025 V2: fixed small typo on eq. 15 and eq. 17
>
> **摘要:** Recent multi-modal audio-language models (ALMs) excel at text-audio retrieval but struggle with frame-wise audio understanding. Prior works use temporal-aware labels or unsupervised training to improve frame-wise capabilities, but they still lack fine-grained labeling capability to pinpoint when an event occurs. While traditional sound event detection models can precisely localize events, they are limited to pre-defined categories, making them ineffective for real-world scenarios with out-of-distribution events. In this work, we introduce FLAM, an open-vocabulary contrastive audio-language model capable of localizing specific sound events. FLAM employs a memory-efficient and calibrated frame-wise objective with logit adjustment to address spurious correlations, such as event dependencies and label imbalances during training. To enable frame-wise supervision, we leverage a large-scale dataset with diverse audio events, LLM-generated captions and simulation. Experimental results and case studies demonstrate that FLAM significantly improves the open-vocabulary localization capability while maintaining strong performance in global retrieval and downstream tasks.
>
---
#### [replaced 007] LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.00493v3](http://arxiv.org/pdf/2503.00493v3)**

> **作者:** Boyi Kang; Xinfa Zhu; Zihan Zhang; Zhen Ye; Mingshuai Liu; Ziqian Wang; Yike Zhu; Guobin Ma; Jun Chen; Longshuai Xiao; Chao Weng; Wei Xue; Lei Xie
>
> **备注:** ACL2025 main, Codes available at https://github.com/Kevin-naticl/LLaSE-G1
>
> **摘要:** Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area.
>
---
#### [replaced 008] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00975v3](http://arxiv.org/pdf/2506.00975v3)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
#### [replaced 009] C3T: Cross-modal Transfer Through Time for Sensor-based Human Activity Recognition
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.16803v3](http://arxiv.org/pdf/2407.16803v3)**

> **作者:** Abhi Kamboj; Anh Duy Nguyen; Minh N. Do
>
> **摘要:** In order to unlock the potential of diverse sensors, we investigate a method to transfer knowledge between time-series modalities using a multimodal \textit{temporal} representation space for Human Activity Recognition (HAR). Specifically, we explore the setting where the modality used in testing has no labeled data during training, which we refer to as Unsupervised Modality Adaptation (UMA). We categorize existing UMA approaches as Student-Teacher or Contrastive Alignment methods. These methods typically compress continuous-time data samples into single latent vectors during alignment, inhibiting their ability to transfer temporal information through real-world temporal distortions. To address this, we introduce Cross-modal Transfer Through Time (C3T), which preserves temporal information during alignment to handle dynamic sensor data better. C3T achieves this by aligning a set of temporal latent vectors across sensing modalities. Our extensive experiments on various camera+IMU datasets demonstrate that C3T outperforms existing methods in UMA by at least 8% in accuracy and shows superior robustness to temporal distortions such as time-shift, misalignment, and dilation. Our findings suggest that C3T has significant potential for developing generalizable models for time-series sensor data, opening new avenues for various multimodal applications.
>
---
#### [replaced 010] Baseline Systems and Evaluation Metrics for Spatial Semantic Segmentation of Sound Scenes
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.22088v2](http://arxiv.org/pdf/2503.22088v2)**

> **作者:** Binh Thien Nguyen; Masahiro Yasuda; Daiki Takeuchi; Daisuke Niizumi; Yasunori Ohishi; Noboru Harada
>
> **备注:** Accepted to EUSIPCO2025
>
> **摘要:** Immersive communication has made significant advancements, especially with the release of the codec for Immersive Voice and Audio Services. Aiming at its further realization, the DCASE 2025 Challenge has recently introduced a task for spatial semantic segmentation of sound scenes (S5), which focuses on detecting and separating sound events in spatial sound scenes. In this paper, we explore methods for addressing the S5 task. Specifically, we present baseline S5 systems that combine audio tagging (AT) and label-queried source separation (LSS) models. We investigate two LSS approaches based on the ResUNet architecture: a) extracting a single source for each detected event and b) querying multiple sources concurrently. Since each separated source in S5 is identified by its sound event class label, we propose new class-aware metrics to evaluate both the sound sources and labels simultaneously. Experimental results on first-order ambisonics spatial audio demonstrate the effectiveness of the proposed systems and confirm the efficacy of the metrics.
>
---
#### [replaced 011] A Hypernetwork-Based Approach to KAN Representation of Audio Signals
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.02585v2](http://arxiv.org/pdf/2503.02585v2)**

> **作者:** Patryk Marszałek; Maciej Rut; Piotr Kawa; Przemysław Spurek; Piotr Syga
>
> **摘要:** Implicit neural representations (INR) have gained prominence for efficiently encoding multimedia data, yet their applications in audio signals remain limited. This study introduces the Kolmogorov-Arnold Network (KAN), a novel architecture using learnable activation functions, as an effective INR model for audio representation. KAN demonstrates superior perceptual performance over previous INRs, achieving the lowest Log-SpectralDistance of 1.29 and the highest Perceptual Evaluation of Speech Quality of 3.57 for 1.5 s audio. To extend KAN's utility, we propose FewSound, a hypernetwork-based architecture that enhances INR parameter updates. FewSound outperforms the state-of-the-art HyperSound, with a 33.3% improvement in MSE and 60.87% in SI-SNR. These results show KAN as a robust and adaptable audio representation with the potential for scalability and integration into various hypernetwork frameworks. The source code can be accessed at https://github.com/gmum/fewsound.git.
>
---
#### [replaced 012] DnR-nonverbal: Cinematic Audio Source Separation Dataset Containing Non-Verbal Sounds
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.02499v2](http://arxiv.org/pdf/2506.02499v2)**

> **作者:** Takuya Hasumi; Yusuke Fujita
>
> **备注:** Accepted to Interspeech 2025, 5 pages, 3 figures, dataset is available at https://zenodo.org/records/15470640
>
> **摘要:** We propose a new dataset for cinematic audio source separation (CASS) that handles non-verbal sounds. Existing CASS datasets only contain reading-style sounds as a speech stem. These datasets differ from actual movie audio, which is more likely to include acted-out voices. Consequently, models trained on conventional datasets tend to have issues where emotionally heightened voices, such as laughter and screams, are more easily separated as an effect, not speech. To address this problem, we build a new dataset, DnR-nonverbal. The proposed dataset includes non-verbal sounds like laughter and screams in the speech stem. From the experiments, we reveal the issue of non-verbal sound extraction by the current CASS model and show that our dataset can effectively address the issue in the synthetic and actual movie audio. Our dataset is available at https://zenodo.org/records/15470640.
>
---
