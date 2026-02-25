# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Quantifying Dimensional Independence in Speech: An Information-Theoretic Framework for Disentangled Representation Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音表示学习任务，旨在量化语音中不同维度的独立性。通过信息论框架，分析声源与声道的统计依赖，揭示情感、语言和病理维度的耦合程度。**

- **链接: [https://arxiv.org/pdf/2602.20592v1](https://arxiv.org/pdf/2602.20592v1)**

> **作者:** Bipasha Kashyap; Björn W. Schuller; Pubudu N. Pathirana
>
> **摘要:** Speech signals encode emotional, linguistic, and pathological information within a shared acoustic channel; however, disentanglement is typically assessed indirectly through downstream task performance. We introduce an information-theoretic framework to quantify cross-dimension statistical dependence in handcrafted acoustic features by integrating bounded neural mutual information (MI) estimation with non-parametric validation. Across six corpora, cross-dimension MI remains low, with tight estimation bounds ($< 0.15$ nats), indicating weak statistical coupling in the data considered, whereas Source--Filter MI is substantially higher (0.47 nats). Attribution analysis, defined as the proportion of total MI attributable to source versus filter components, reveals source dominance for emotional dimensions (80\%) and filter dominance for linguistic and pathological dimensions (60\% and 58\%, respectively). These findings provide a principled framework for quantifying dimensional independence in speech.
>
---
#### [new 002] Graph Modelling Analysis of Speech-Gesture Interaction for Aphasia Severity Estimation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语言障碍评估任务，旨在通过语音与手势的图模型分析，自动估计失语症严重程度，解决传统方法依赖孤立语言特征的问题。**

- **链接: [https://arxiv.org/pdf/2602.20163v1](https://arxiv.org/pdf/2602.20163v1)**

> **作者:** Navya Martin Kollapally; Christa Akers; Renjith Nelson Joseph
>
> **备注:** IJCAI
>
> **摘要:** Aphasia is an acquired language disorder caused by injury to the regions of the brain that are responsible for language. Aphasia may impair the use and comprehension of written and spoken language. The Western Aphasia Battery-Revised (WAB-R) is an assessment tool administered by speech-language pathologists (SLPs) to evaluate the aphasia type and severity. Because the WAB-R measures isolated linguistic skills, there has been growing interest in the assessment of discourse production as a more holistic representation of everyday language abilities. Recent advancements in speech analysis focus on automated estimation of aphasia severity from spontaneous speech, relying mostly in isolated linguistic or acoustical features. In this work, we propose a graph neural network-based framework for estimating aphasia severity. We represented each participant's discourse as a directed multi-modal graph, where nodes represent lexical items and gestures and edges encode word-word, gesture-word, and word-gesture transitions. GraphSAGE is employed to learn participant-level embeddings, thus integrating information from immediate neighbors and overall graph structure. Our results suggest that aphasia severity is not encoded in isolated lexical distribution, but rather emerges from structured interactions between speech and gesture. The proposed architecture offers a reliable automated aphasia assessment, with possible uses in bedside screening and telehealth-based monitoring.
>
---
#### [new 003] Assessing the Impact of Speaker Identity in Speech Spoofing Detection
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音欺骗检测任务，旨在解决说话人身份对检测系统的影响问题。通过提出两种方法，提升检测性能，降低错误率。**

- **链接: [https://arxiv.org/pdf/2602.20805v1](https://arxiv.org/pdf/2602.20805v1)**

> **作者:** Anh-Tuan Dao; Driss Matrouf; Nicholas Evans
>
> **摘要:** Spoofing detection systems are typically trained using diverse recordings from multiple speakers, often assuming that the resulting embeddings are independent of speaker identity. However, this assumption remains unverified. In this paper, we investigate the impact of speaker information on spoofing detection systems. We propose two approaches within our Speaker-Invariant Multi-Task framework, one that models speaker identity within the embeddings and another that removes it. SInMT integrates multi-task learning for joint speaker recognition and spoofing detection, incorporating a gradient reversal layer. Evaluated using four datasets, our speaker-invariant model reduces the average equal error rate by 17% compared to the baseline, with up to 48% reduction for the most challenging attacks (e.g., A11).
>
---
#### [new 004] Geometric Analysis of Speech Representation Spaces: Topological Disentanglement and Confound Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音分析任务，旨在解决多语言环境中病理语音与口音区分不清的问题。通过聚类框架评估语音特征的几何分离性，提出解决方案以提升临床系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.20823v1](https://arxiv.org/pdf/2602.20823v1)**

> **作者:** Bipasha Kashyap; Pubudu N. Pathirana
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Speech-based clinical tools are increasingly deployed in multilingual settings, yet whether pathological speech markers remain geometrically separable from accent variation remains unclear. Systems may misclassify healthy non-native speakers or miss pathology in multilingual patients. We propose a four-metric clustering framework to evaluate geometric disentanglement of emotional, linguistic, and pathological speech features across six corpora and eight dataset combinations. A consistent hierarchy emerges: emotional features form the tightest clusters (Silhouette 0.250), followed by pathological (0.141) and linguistic (0.077). Confound analysis shows pathological-linguistic overlap remains below 0.21, which is above the permutation null but bounded for clinical deployment. Trustworthiness analysis confirms embedding fidelity and robustness of the geometric conclusions. Our framework provides actionable guidelines for equitable and reliable speech health systems across diverse populations.
>
---
#### [new 005] Voices of the Mountains: Deep Learning-Based Vocal Error Detection System for Kurdish Maqams
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于自动歌唱评估任务，旨在解决 Kurdish maqam 唱腔错误检测问题。通过深度学习方法识别 pitch、rhythm 和 modal drift 错误，提升传统音乐的评估准确性。**

- **链接: [https://arxiv.org/pdf/2602.20744v1](https://arxiv.org/pdf/2602.20744v1)**

> **作者:** Darvan Shvan Khairaldeen; Hossein Hassani
>
> **摘要:** Maqam, a singing type, is a significant component of Kurdish music. A maqam singer receives training in a traditional face-to-face or through self-training. Automatic Singing Assessment (ASA) uses machine learning (ML) to provide the accuracy of singing styles and can help learners to improve their performance through error detection. Currently, the available ASA tools follow Western music rules. The musical composition requires all notes to stay within their expected pitch range from start to finish. The system fails to detect micro-intervals and pitch bends, so it identifies Kurdish maqam singing as incorrect even though the singer performs according to traditional rules. Kurdish maqam requires recognizing performance errors within microtonal spaces, which is beyond Western equal temperament. This research is the first attempt to address the mentioned gap. While many error types happen during singing, our focus is on pitch, rhythm, and modal stability errors in the context of Bayati-Kurd. We collected 50 songs from 13 vocalists ( 2-3 hours) and annotated 221 error spans (150 fine pitch, 46 rhythm, 25 modal drift). The data was segmented into 15,199 overlapping windows and converted to log-mel spectrograms. We developed a two-headed CNN-BiLSTM with attention mode to decide whether a window contains an error and to classify it based on the chosen errors. Trained for 20 epochs with early stopping at epoch 10, the model reached a validation macro-F1 of 0.468. On the full 50-song evaluation at a 0.750 threshold, recall was 39.4% and precision 25.8% . Within detected windows, type macro-F1 was 0.387, with F1 of 0.492 (fine pitch), 0.536 (rhythm), and 0.133 (modal drift); modal drift recall was 8.0%. The better performance on common error types shows that the method works, while the poor modal-drift recall shows that more data and balancing are needed.
>
---
#### [new 006] 823-OLT @ BUET DL Sprint 4.0: Context-Aware Windowing for ASR and Fine-Tuned Speaker Diarization in Bengali Long Form Audio
- **分类: cs.SD**

- **简介: 该论文聚焦于孟加拉语长语音的自动语音识别与说话人辨识任务，解决其在长语音技术中的不足。通过改进的窗口策略和微调模型，提升识别准确率与效率。**

- **链接: [https://arxiv.org/pdf/2602.21183v1](https://arxiv.org/pdf/2602.21183v1)**

> **作者:** Ratnajit Dhar; Arpita Mallik
>
> **摘要:** Bengali, despite being one of the most widely spoken languages globally, remains underrepresented in long form speech technology, particularly in systems addressing transcription and speaker attribution. We present frameworks for long form Bengali speech intelligence that address automatic speech recognition using a Whisper Medium based model and speaker diarization using a finetuned segmentation model. The ASR pipeline incorporates vocal separation, voice activity detection, and a gap aware windowing strategy to construct context preserving segments for stable decoding. For diarization, a pretrained speaker segmentation model is finetuned on the official competition dataset (provided as part of the DL Sprint 4.0 competition organized under BUET CSE Fest), to better capture Bengali conversational patterns. The resulting systems deliver both efficient transcription of long form audio and speaker aware transcription to provide scalable speech technology solutions for low resource languages.
>
---
#### [new 007] Training-Free Intelligibility-Guided Observation Addition for Noisy ASR
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，解决噪声环境下ASR性能下降问题。提出一种无需训练的 intelligibility-引导的观察融合方法，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2602.20967v1](https://arxiv.org/pdf/2602.20967v1)**

> **作者:** Haoyang Li; Changsong Liu; Wei Rao; Hao Shi; Sakriani Sakti; Eng Siong Chng
>
> **摘要:** Automatic speech recognition (ASR) degrades severely in noisy environments. Although speech enhancement (SE) front-ends effectively suppress background noise, they often introduce artifacts that harm recognition. Observation addition (OA) addressed this issue by fusing noisy and SE enhanced speech, improving recognition without modifying the parameters of the SE or ASR models. This paper proposes an intelligibility-guided OA method, where fusion weights are derived from intelligibility estimates obtained directly from the backend ASR. Unlike prior OA methods based on trained neural predictors, the proposed method is training-free, reducing complexity and enhances generalization. Extensive experiments across diverse SE-ASR combinations and datasets demonstrate strong robustness and improvements over existing OA baselines. Additional analyses of intelligibility-guided switching-based alternatives and frame versus utterance-level OA further validate the proposed design.
>
---
#### [new 008] Memory-guided Prototypical Co-occurrence Learning for Mixed Emotion Recognition
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于混合情绪识别任务，旨在解决真实场景中多情绪共存的问题。提出MPCL框架，通过记忆机制建模情绪共现模式，提升情绪分布预测性能。**

- **链接: [https://arxiv.org/pdf/2602.20530v1](https://arxiv.org/pdf/2602.20530v1)**

> **作者:** Ming Li; Yong-Jin Liu; Fang Liu; Huankun Sheng; Yeying Fan; Yixiang Wei; Minnan Luo; Weizhan Zhang; Wenping Wang
>
> **摘要:** Emotion recognition from multi-modal physiological and behavioral signals plays a pivotal role in affective computing, yet most existing models remain constrained to the prediction of singular emotions in controlled laboratory settings. Real-world human emotional experiences, by contrast, are often characterized by the simultaneous presence of multiple affective states, spurring recent interest in mixed emotion recognition as an emotion distribution learning problem. Current approaches, however, often neglect the valence consistency and structured correlations inherent among coexisting emotions. To address this limitation, we propose a Memory-guided Prototypical Co-occurrence Learning (MPCL) framework that explicitly models emotion co-occurrence patterns. Specifically, we first fuse multi-modal signals via a multi-scale associative memory mechanism. To capture cross-modal semantic relationships, we construct emotion-specific prototype memory banks, yielding rich physiological and behavioral representations, and employ prototype relation distillation to ensure cross-modal alignment in the latent prototype space. Furthermore, inspired by human cognitive memory systems, we introduce a memory retrieval strategy to extract semantic-level co-occurrence associations across emotion categories. Through this bottom-up hierarchical abstraction process, our model learns affectively informative representations for accurate emotion distribution prediction. Comprehensive experiments on two public datasets demonstrate that MPCL consistently outperforms state-of-the-art methods in mixed emotion recognition, both quantitatively and qualitatively.
>
---
## 更新

#### [replaced 001] MSR-Codec: A Low-Bitrate Multi-Stream Residual Codec for High-Fidelity Speech Generation with Information Disentanglement
- **分类: eess.AS**

- **简介: 该论文提出MSR-Codec，一种用于高保真语音生成的低比特率多流残差编解码器，解决语音合成与语音转换中的信息解耦问题。**

- **链接: [https://arxiv.org/pdf/2509.13068v3](https://arxiv.org/pdf/2509.13068v3)**

> **作者:** Jingyu Li; Guangyan Zhang; Zhen Ye; Yiwen Guo
>
> **摘要:** Audio codecs are a critical component of modern speech generation systems. This paper introduces a low-bitrate, multi-scale residual codec that encodes speech into four distinct streams: semantic, timbre, prosody, and residual. This architecture achieves high-fidelity speech reconstruction at competitive low bitrates while demonstrating an inherent ability for information disentanglement. We construct a two-stage language model for text-to-speech (TTS) synthesis using this codec, which, despite its lightweight design and minimal data requirements, achieves a state-of-the-art Word Error Rate (WER) and superior speaker similarity compared to several larger models. Furthermore, the codec's design proves highly effective for voice conversion, enabling independent manipulation of speaker timbre and prosody. Our inference code, pre-trained models, and audio samples are available at https://github.com/herbertLJY/MSRCodec.
>
---
#### [replaced 002] K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出K-Function，用于儿童语言评估，解决自动识别儿童语音困难的问题。通过结合声学模型和大语言模型，提升语音转录准确率并实现客观评分。**

- **链接: [https://arxiv.org/pdf/2507.03043v3](https://arxiv.org/pdf/2507.03043v3)**

> **作者:** Shuhe Li; Chenxu Guo; Jiachen Lian; Cheol Jun Cho; Wenshuo Zhao; Xiner Xu; Ruiyu Jin; Xiaoyu Shi; Xuanru Zhou; Dingkun Zhou; Sam Wang; Grace Wang; Jingze Yang; Jingyi Xu; Ruohan Bao; Xingrui Chen; Elise Brenner; Brandon In; Francesca Pei; Maria Luisa Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** Accepted to 2026 ICASSP
>
> **摘要:** Evaluating young children's language is challenging for automatic speech recognizers due to high-pitched voices, prolonged sounds, and limited data. We introduce K-Function, a framework that combines accurate sub-word transcription with objective, Large Language Model (LLM)-driven scoring. Its core, Kids-Weighted Finite State Transducer (K-WFST), merges an acoustic phoneme encoder with a phoneme-similarity model to capture child-specific speech errors while remaining fully interpretable. K-WFST achieves a 1.39 % phoneme error rate on MyST and 8.61 % on Multitudes-an absolute improvement of 10.47 % and 7.06 % over a greedy-search decoder. These high-quality transcripts are used by an LLM to grade verbal skills, developmental milestones, reading, and comprehension, with results that align closely with human evaluators. Our findings show that precise phoneme recognition is essential for creating an effective assessment framework, enabling scalable language screening for children.
>
---
#### [replaced 003] MultiAPI Spoof: A Multi-API Dataset and Local-Attention Network for Speech Anti-spoofing Detection
- **分类: cs.SD**

- **简介: 该论文属于语音反欺骗任务，解决现有基准与真实场景不匹配的问题。构建了多API语音数据集，并提出改进网络模型以提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.07352v2](https://arxiv.org/pdf/2512.07352v2)**

> **作者:** Xueping Zhang; Zhenshan Zhang; Yechen Wang; Linxi Li; Liwei Jin; Ming Li
>
> **备注:** Submited to Interspeech
>
> **摘要:** Existing speech anti-spoofing benchmarks rely on a narrow set of public models, creating a substantial gap from real-world scenarios in which commercial systems employ diverse, often proprietary APIs. To address this issue, we introduce MultiAPI Spoof, a multi-API audio anti-spoofing dataset comprising about 230 hours of synthetic speech generated by 30 distinct APIs, including commercial services, open-source models, and online platforms. Based on this dataset, we define the API tracing task, enabling fine-grained attribution of spoofed audio to its generation source. We further propose Nes2Net-LA, a local-attention enhanced variant of Nes2Net that improves local context modeling and fine-grained spoofing feature extraction. Experiments show that Nes2Net-LA achieves state-of-the-art performance and offers superior robustness, particularly under diverse and unseen spoofing conditions. Code \footnote{https://github.com/XuepingZhang/MultiAPI-Spoof} and dataset \footnote{https://xuepingzhang.github.io/MultiAPI-Spoof-Dataset/} have released.
>
---
#### [replaced 004] UBGAN: Enhancing Coded Speech with Blind and Guided Bandwidth Extension
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出UBGAN，解决语音编码中带宽扩展问题，提升编码语音的感知质量。通过生成对抗网络实现宽带到超宽带的扩展，适用于多种编码器。**

- **链接: [https://arxiv.org/pdf/2505.16404v2](https://arxiv.org/pdf/2505.16404v2)**

> **作者:** Kishan Gupta; Srikanth Korse; Andreas Brendel; Nicola Pia; Guillaume Fuchs
>
> **摘要:** In practical application of speech codecs, a multitude of factors such as the quality of the radio connection, limiting hardware or required user experience necessitate trade-offs between achievable perceptual quality, engendered bitrate and computational complexity. Most conventional and neural speech codecs operate on wideband (WB) speech signals to achieve this compromise. To further enhance the perceptual quality of coded speech, bandwidth extension (BWE) of the transmitted speech is an attractive and popular technique in conventional speech coding. In contrast, neural speech codecs are typically trained end-to-end to a specific set of requirements and are often not easily adaptable. In particular, they are typically trained to operate at a single fixed sampling rate. With the Universal Bandwidth Extension Generative Adversarial Network (UBGAN), we propose a modular and lightweight GAN-based solution that increases the operational flexibility of a wide range of conventional and neural codecs. Our model operates in the subband domain and extends the bandwidth of WB signals from 8 kHz to 16 kHz, resulting in super-wideband (SWB) signals. We further introduce two variants, guided-UBGAN and blind-UBGAN, where the guided version transmits quantized learned representation as a side information at a very low bitrate additional to the bitrate of the codec, while blind-BWE operates without such side-information. Our subjective assessments demonstrate the advantage of UBGAN applied to WB codecs and highlight the generalization capacity of our proposed method across multiple codecs and bitrates.
>
---
#### [replaced 005] Enroll-on-Wakeup: A First Comparative Study of Target Speech Extraction for Seamless Interaction in Real Noisy Human-Machine Dialogue Scenarios
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音提取任务，解决实时噪声环境下用户交互体验问题。提出EoW框架，利用唤醒词作为语音参考，提升交互流畅性。**

- **链接: [https://arxiv.org/pdf/2602.15519v2](https://arxiv.org/pdf/2602.15519v2)**

> **作者:** Yiming Yang; Guangyong Wang; Haixin Guan; Yanhua Long
>
> **备注:** This paper is submitted to Interspeech 2026
>
> **摘要:** Target speech extraction (TSE) typically relies on pre-recorded high-quality enrollment speech, which disrupts user experience and limits feasibility in spontaneous interaction. In this paper, we propose Enroll-on-Wakeup (EoW), a novel framework where the wake-word segment, captured naturally during human-machine interaction, is automatically utilized as the enrollment reference. This eliminates the need for pre-collected speech to enable a seamless experience. We perform the first systematic study of EoW-TSE, evaluating advanced discriminative and generative models under real diverse acoustic conditions. Given the short and noisy nature of wake-word segments, we investigate enrollment augmentation using LLM-based TTS. Results show that while current TSE models face performance degradation in EoW-TSE, TTS-based assistance significantly enhances the listening experience, though gaps remain in speech recognition accuracy.
>
---
#### [replaced 006] Hearing the Order: Investigating Position Bias in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频语言模型研究，探讨模型在有序选项任务中是否受位置偏见影响。通过实验发现模型性能受选项顺序影响，提出排列策略以减轻偏见。**

- **链接: [https://arxiv.org/pdf/2510.00628v2](https://arxiv.org/pdf/2510.00628v2)**

> **作者:** Yu-Xiang Lin; Chen-An Li; Sheng-Lun Wei; Po-Chun Chen; Hsin-Hsi Chen; Hung-yi Lee
>
> **备注:** The first two authors contributed equally. Submitted to Interspeech 2026
>
> **摘要:** Large audio-language models (LALMs) are often used in tasks that involve reasoning over ordered options. An open question is whether their predictions are influenced by the order of answer choices, which would indicate a form of position bias and undermine their reliability. In this paper, we identify and analyze this problem in LALMs. We demonstrate that no model is immune to this bias through extensive experiments on six LALMs across three widely used benchmarks and their spoken counterparts. Shuffling the order of answer options can cause performance fluctuations of up to 24% and even change model rankings, raising concerns about the reliability of current evaluation practices. We also study permutation-based strategies and show that they can mitigate bias in most cases. Our work represents the first systematic investigation of this issue in LALMs, and we hope it raises awareness and motivates further research in this direction.
>
---
#### [replaced 007] An Adaptive CMSA for Solving the Longest Filled Common Subsequence Problem with an Application in Audio Querying
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究LFCS问题，属于NP-hard难题，旨在解决大规模实例的高效求解。提出自适应CMSA框架，并应用于音频识别，验证了其有效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2509.12261v2](https://arxiv.org/pdf/2509.12261v2)**

> **作者:** Marko Djukanovic; Christian Blum; Aleksandar Kartelj; Ana Nikolikj; Guenther Raidl
>
> **摘要:** This paper addresses the Longest Filled Common Subsequence (LFCS) problem, a challenging NP-hard problem with applications in bioinformatics, including gene mutation prediction and genomic data reconstruction. Existing approaches, including exact, metaheuristic, and approximation algorithms, have primarily been evaluated on small-sized instances, which offer limited insights into their scalability. In this work, we introduce a new benchmark dataset with significantly larger instances and demonstrate that existing datasets lack the discriminative power needed to meaningfully assess algorithm performance at scale. To solve large instances efficiently, we utilize an adaptive Construct, Merge, Solve, Adapt (CMSA) framework that iteratively generates promising subproblems via component-based construction and refines them using feedback from prior iterations. Subproblems are solved using an external black-box solver. Extensive experiments on both standard and newly introduced benchmarks show that the proposed adaptive CMSA achieves state-of-the-art performance, outperforming five leading methods. Notably, on 1,510 problem instances with known optimal solutions, our approach solves 1,486 of them -- achieving over 99.9% optimal solution quality and demonstrating exceptional scalability. We additionally propose a novel application of LFCS for song identification from degraded audio excerpts as an engineering contribution, using real-world energy-profile instances from popular music. Finally, we conducted an empirical explainability analysis to identify critical feature combinations influencing algorithm performance, i.e., the key problem features contributing to success or failure of the approaches across different instance types are revealed.
>
---
#### [replaced 008] Sound Source Localization for Spatial Mapping of Surgical Actions in Dynamic Scenes
- **分类: cs.SD; cs.CV; eess.AS; eess.IV**

- **简介: 该论文属于手术场景理解任务，旨在解决动态场景中声源定位问题。通过融合3D音频与视觉信息，实现手术动作的时空定位与建模。**

- **链接: [https://arxiv.org/pdf/2510.24332v2](https://arxiv.org/pdf/2510.24332v2)**

> **作者:** Jonas Hein; Lazaros Vlachopoulos; Maurits Geert Laurent Olthof; Bastian Sigrist; Philipp Fürnstahl; Matthias Seibold
>
> **摘要:** Purpose: Surgical scene understanding is key to advancing computer-aided and intelligent surgical systems. Current approaches predominantly rely on visual data or end-to-end learning, which limits fine-grained contextual modeling. This work aims to enhance surgical scene representations by integrating 3D acoustic information, enabling temporally and spatially aware multimodal understanding of surgical environments. Methods: We propose a novel framework for generating 4D audio-visual representations of surgical scenes by projecting acoustic localization information from a phased microphone array onto dynamic point clouds from an RGB-D camera. A transformer-based acoustic event detection module identifies relevant temporal segments containing tool-tissue interactions which are spatially localized in the audio-visual scene representation. The system was experimentally evaluated in a realistic operating room setup during simulated surgical procedures performed by experts. Results: The proposed method successfully localizes surgical acoustic events in 3D space and associates them with visual scene elements. Experimental evaluation demonstrates accurate spatial sound localization and robust fusion of multimodal data, providing a comprehensive, dynamic representation of surgical activity. Conclusion: This work introduces the first approach for spatial sound localization in dynamic surgical scenes, marking a significant advancement toward multimodal surgical scene representations. By integrating acoustic and visual data, the proposed framework enables richer contextual understanding and provides a foundation for future intelligent and autonomous surgical systems.
>
---
#### [replaced 009] Uncertainty Calibration of Multi-Label Bird Sound Classifiers
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究生物声学中多标签鸟类声音分类器的不确定性校准问题，评估模型校准性能并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2511.08261v2](https://arxiv.org/pdf/2511.08261v2)**

> **作者:** Raphael Schwinger; Ben McEwen; Vincent S. Kather; René Heinrich; Lukas Rauch; Sven Tomforde
>
> **备注:** Accepted at ICAART 2026
>
> **摘要:** Passive acoustic monitoring enables large-scale biodiversity assessment, but reliable classification of bioacoustic sounds requires not only high accuracy but also well-calibrated uncertainty estimates to ground decision-making. In bioacoustics, calibration is challenged by overlapping vocalisations, long-tailed species distributions, and distribution shifts between training and deployment data. The calibration of multi-label deep learning classifiers within the domain of bioacoustics has not yet been assessed. We systematically benchmark the calibration of four state-of-the-art multi-label bird sound classifiers on the BirdSet benchmark, evaluating both global, per-dataset and per-class calibration using threshold-free calibration metrics (ECE, MCS) alongside discrimination metrics (cmAP). Model calibration varies significantly across datasets and classes. While Perch v2 and ConvNeXt$_{BS}$ show better global calibration, results vary between datasets. Both models indicate consistent underconfidence, while AudioProtoPNet and BirdMAE are mostly overconfident. Surprisingly, calibration seems to be better for less frequent classes. Using simple post hoc calibration methods we demonstrate a straightforward way to improve calibration. A small labelled calibration set is sufficient to significantly improve calibration with Platt scaling, while global calibration parameters suffer from dataset variability. Our findings highlight the importance of evaluating and improving uncertainty calibration in bioacoustic classifiers.
>
---
#### [replaced 010] Evaluating CNN with Stacked Feature Representations and Audio Spectrogram Transformer Models for Sound Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于环境声音分类任务，旨在提升CNN性能。通过特征叠加和对比Transformer模型，探索高效的声音分类方法。**

- **链接: [https://arxiv.org/pdf/2602.09321v2](https://arxiv.org/pdf/2602.09321v2)**

> **作者:** Parinaz Binandeh Dehaghania; Danilo Penab; A. Pedro Aguiar
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Environmental sound classification (ESC) has gained significant attention due to its diverse applications in smart city monitoring, fault detection, acoustic surveillance, and manufacturing quality control. To enhance CNN performance, feature stacking techniques have been explored to aggregate complementary acoustic descriptors into richer input representations. In this paper, we investigate CNN-based models employing various stacked feature combinations, including Log-Mel Spectrogram (LM), Spectral Contrast (SPC), Chroma (CH), Tonnetz (TZ), Mel-Frequency Cepstral Coefficients (MFCCs), and Gammatone Cepstral Coefficients (GTCC). Experiments are conducted on the widely used ESC-50 and UrbanSound8K datasets under different training regimes, including pretraining on ESC-50, fine-tuning on UrbanSound8K, and comparison with Audio Spectrogram Transformer (AST) models pretrained on large-scale corpora such as AudioSet. This experimental design enables an analysis of how feature-stacked CNNs compare with transformer-based models under varying levels of training data and pretraining diversity. The results indicate that feature-stacked CNNs offer a more computationally and data-efficient alternative when large-scale pretraining or extensive training data are unavailable, making them particularly well suited for resource-constrained and edge-level sound classification scenarios.
>
---
#### [replaced 011] Binaural Target Speaker Extraction using Individualized HRTF
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，旨在解决多说话人环境下提取目标说话人的问题。通过利用个体HRTF和复数神经网络，有效分离目标语音并保留双耳线索。**

- **链接: [https://arxiv.org/pdf/2507.19369v5](https://arxiv.org/pdf/2507.19369v5)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **摘要:** In this work, we address the problem of binaural target-speaker extraction in the presence of multiple simultane-ous talkers. We propose a novel approach that leverages the individual listener's Head-Related Transfer Function (HRTF) to isolate the target speaker. The proposed method is speaker-independent, as it does not rely on speaker embeddings. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier transform (STFT) of the mixed audio signals, and compare it to a Real-Imaginary (RI)-based neural network, demonstrating the advantages of the former. We first evaluate the method in an anechoic, noise-free scenario, achieving excellent extraction performance while preserving the binaural cues of the target signal. We then extend the evaluation to reverberant conditions. Our method proves robust, maintaining speech clarity and source directionality while simultaneously reducing reverberation. A comparative analysis with existing binaural Target Speaker Extraction (TSE) methods shows that the proposed approach achieves performance comparable to state-of-the-art techniques in terms of noise reduction and perceptual quality, while providing a clear advantage in preserving binaural cues. Demo-page: https://bi-ctse-hrtf.github.io
>
---
