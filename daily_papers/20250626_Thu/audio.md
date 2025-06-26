# 音频 cs.SD;  eess.SP

- **最新发布 9 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于枪支类型分类任务，旨在通过声学分析识别枪声并确定枪支类型，以低成本方式提升枪击事件的检测与响应效率。**

- **链接: [http://arxiv.org/pdf/2506.20609v1](http://arxiv.org/pdf/2506.20609v1)**

> **作者:** Ankit Shah; Rita Singh; Bhiksha Raj; Alexander Hauptmann
>
> **备注:** 4 pages + 1 References
>
> **摘要:** The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders.
>
---
#### [new 002] A Multi-Modal Spatial Risk Framework for EV Charging Infrastructure Using Remote Sensing
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于风险评估任务，旨在解决EV充电设施在环境压力下的脆弱性问题。通过融合多源数据和空间分析方法，构建了RSERI-EV框架以评估充电站的韧性。**

- **链接: [http://arxiv.org/pdf/2506.19860v1](http://arxiv.org/pdf/2506.19860v1)**

> **作者:** Oktay Karakuş; Padraig Corcoran
>
> **备注:** 11 pages, 4 figures, 2 tables
>
> **摘要:** Electric vehicle (EV) charging infrastructure is increasingly critical to sustainable transport systems, yet its resilience under environmental and infrastructural stress remains underexplored. In this paper, we introduce RSERI-EV, a spatially explicit and multi-modal risk assessment framework that combines remote sensing data, open infrastructure datasets, and spatial graph analytics to evaluate the vulnerability of EV charging stations. RSERI-EV integrates diverse data layers, including flood risk maps, land surface temperature (LST) extremes, vegetation indices (NDVI), land use/land cover (LULC), proximity to electrical substations, and road accessibility to generate a composite Resilience Score. We apply this framework to the country of Wales EV charger dataset to demonstrate its feasibility. A spatial $k$-nearest neighbours ($k$NN) graph is constructed over the charging network to enable neighbourhood-based comparisons and graph-aware diagnostics. Our prototype highlights the value of multi-source data fusion and interpretable spatial reasoning in supporting climate-resilient, infrastructure-aware EV deployment.
>
---
#### [new 003] Speaker Embeddings to Improve Tracking of Intermittent and Moving Speakers
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于说话人跟踪任务，解决间歇性与移动说话人身份分配问题。通过使用说话人嵌入进行跟踪后重识别，提升系统性能。**

- **链接: [http://arxiv.org/pdf/2506.19875v1](http://arxiv.org/pdf/2506.19875v1)**

> **作者:** Taous Iatariene; Can Cui; Alexandre Guérin; Romain Serizel
>
> **备注:** 33rd European Signal Processing Conference (EUSIPCO 2025), Sep 2025, Palerme (Italie), Italy
>
> **摘要:** Speaker tracking methods often rely on spatial observations to assign coherent track identities over time. This raises limits in scenarios with intermittent and moving speakers, i.e., speakers that may change position when they are inactive, thus leading to discontinuous spatial trajectories. This paper proposes to investigate the use of speaker embeddings, in a simple solution to this issue. We propose to perform identity reassignment post-tracking, using speaker embeddings. We leverage trajectory-related information provided by an initial tracking step and multichannel audio signal. Beamforming is used to enhance the signal towards the speakers' positions in order to compute speaker embeddings. These are then used to assign new track identities based on an enrollment pool. We evaluate the performance of the proposed speaker embedding-based identity reassignment method on a dataset where speakers change position during inactivity periods. Results show that it consistently improves the identity assignment performance of neural and standard tracking systems. In particular, we study the impact of beamforming and input duration for embedding extraction.
>
---
#### [new 004] Dynamic Bandwidth Allocation for Hybrid Event-RGB Transmission
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于多模态数据传输任务，解决事件相机与RGB相机融合时的带宽瓶颈问题。通过联合建模和自适应分配，提升传输效率与重建质量。**

- **链接: [http://arxiv.org/pdf/2506.20222v1](http://arxiv.org/pdf/2506.20222v1)**

> **作者:** Pujing Yang; Guangyi Zhang; Yunlong Cai; Lei Yu; Guanding Yu
>
> **摘要:** Event cameras asynchronously capture pixel-level intensity changes with extremely low latency. They are increasingly used in conjunction with RGB cameras for a wide range of vision-related applications. However, a major challenge in these hybrid systems lies in the transmission of the large volume of triggered events and RGB images. To address this, we propose a transmission scheme that retains efficient reconstruction performance of both sources while accomplishing real-time deblurring in parallel. Conventional RGB cameras and event cameras typically capture the same scene in different ways, often resulting in significant redundant information across their outputs. To address this, we develop a joint event and image (E-I) transmission framework to eliminate redundancy and thereby optimize channel bandwidth utilization. Our approach employs Bayesian modeling and the information bottleneck method to disentangle the shared and domain-specific information within the E-I inputs. This disentangled information bottleneck framework ensures both the compactness and informativeness of extracted shared and domain-specific information. Moreover, it adaptively allocates transmission bandwidth based on scene dynamics, i.e., more symbols are allocated to events for dynamic details or to images for static information. Simulation results demonstrate that the proposed scheme not only achieves superior reconstruction quality compared to conventional systems but also delivers enhanced deblurring performance.
>
---
#### [new 005] Lightweight Target-Speaker-Based Overlap Transcription for Practical Streaming ASR
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，解决重叠语音转录问题。通过结合说话人独立和条件模型，实现高效准确的重叠语音识别。**

- **链接: [http://arxiv.org/pdf/2506.20288v1](http://arxiv.org/pdf/2506.20288v1)**

> **作者:** Aleš Pražák; Marie Kunešová; Josef Psutka
>
> **摘要:** Overlapping speech remains a major challenge for automatic speech recognition (ASR) in real-world applications, particularly in broadcast media with dynamic, multi-speaker interactions. We propose a light-weight, target-speaker-based extension to an existing streaming ASR system to enable practical transcription of overlapping speech with minimal computational overhead. Our approach combines a speaker-independent (SI) model for standard operation with a speaker-conditioned (SC) model selectively applied in overlapping scenarios. Overlap detection is achieved using a compact binary classifier trained on frozen SI model output, offering accurate segmentation at negligible cost. The SC model employs Feature-wise Linear Modulation (FiLM) to incorporate speaker embeddings and is trained on synthetically mixed data to transcribe only the target speaker. Our method supports dynamic speaker tracking and reuses existing modules with minimal modifications. Evaluated on a challenging set of Czech television debates with 16% overlap, the system reduced WER on overlapping segments from 68.0% (baseline) to 35.78% while increasing total computational load by only 44%. The proposed system offers an effective and scalable solution for overlap transcription in continuous ASR services.
>
---
#### [new 006] VoxelOpt: Voxel-Adaptive Message Passing for Discrete Optimization in Deformable Abdominal CT Registration
- **分类: eess.IV; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于医学图像配准任务，旨在解决学习方法在数据不足和大形变下的性能问题，提出VoxelOpt框架结合学习与迭代方法，提升配准效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.19975v1](http://arxiv.org/pdf/2506.19975v1)**

> **作者:** Hang Zhang; Yuxi Zhang; Jiazheng Wang; Xiang Chen; Renjiu Hu; Xin Tian; Gaolei Li; Min Liu
>
> **备注:** Accepted for publication at MICCAI 2025
>
> **摘要:** Recent developments in neural networks have improved deformable image registration (DIR) by amortizing iterative optimization, enabling fast and accurate DIR results. However, learning-based methods often face challenges with limited training data, large deformations, and tend to underperform compared to iterative approaches when label supervision is unavailable. While iterative methods can achieve higher accuracy in such scenarios, they are considerably slower than learning-based methods. To address these limitations, we propose VoxelOpt, a discrete optimization-based DIR framework that combines the strengths of learning-based and iterative methods to achieve a better balance between registration accuracy and runtime. VoxelOpt uses displacement entropy from local cost volumes to measure displacement signal strength at each voxel, which differs from earlier approaches in three key aspects. First, it introduces voxel-wise adaptive message passing, where voxels with lower entropy receives less influence from their neighbors. Second, it employs a multi-level image pyramid with 27-neighbor cost volumes at each level, avoiding exponential complexity growth. Third, it replaces hand-crafted features or contrastive learning with a pretrained foundational segmentation model for feature extraction. In abdominal CT registration, these changes allow VoxelOpt to outperform leading iterative in both efficiency and accuracy, while matching state-of-the-art learning-based methods trained with label supervision. The source code will be available at https://github.com/tinymilky/VoxelOpt
>
---
#### [new 007] The role of audio-visual integration in the time course of phonetic encoding in self-supervised speech models
- **分类: eess.AS; cs.SD; eess.IV**

- **简介: 该论文属于语音处理任务，研究自监督模型在音素编码时间过程中的视听整合能力，旨在解决模型是否能捕捉视听信号的时序差异问题。**

- **链接: [http://arxiv.org/pdf/2506.20361v1](http://arxiv.org/pdf/2506.20361v1)**

> **作者:** Yi Wang; Oli Danyi Liu; Peter Bell
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Human speech perception is multimodal. In natural speech, lip movements can precede corresponding voicing by a non-negligible gap of 100-300 ms, especially for specific consonants, affecting the time course of neural phonetic encoding in human listeners. However, it remains unexplored whether self-supervised learning models, which have been used to simulate audio-visual integration in humans, can capture this asynchronicity between audio and visual cues. We compared AV-HuBERT, an audio-visual model, with audio-only HuBERT, by using linear classifiers to track their phonetic decodability over time. We found that phoneme information becomes available in AV-HuBERT embeddings only about 20 ms before HuBERT, likely due to AV-HuBERT's lower temporal resolution and feature concatenation process. It suggests AV-HuBERT does not adequately capture the temporal dynamics of multimodal speech perception, limiting its suitability for modeling the multimodal speech perception process.
>
---
#### [new 008] MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition
- **分类: eess.AS; cs.AI; cs.SD; 68T10**

- **简介: 该论文属于语音情感识别任务，解决自然语境下情感分类与属性预测问题。提出MATER框架，融合声学与文本特征，提升情感识别的准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2506.19887v1](http://arxiv.org/pdf/2506.19887v1)**

> **作者:** Hyo Jin Jon; Longbin Jin; Hyuntaek Jung; Hyunseo Kim; Donghun Min; Eun Yi Kim
>
> **备注:** 5 pages, 4 figures, 2 tables, 1 algorithm, Accepted to INTERSPEECH 2025
>
> **摘要:** This paper presents our contributions to the Speech Emotion Recognition in Naturalistic Conditions (SERNC) Challenge, where we address categorical emotion recognition and emotional attribute prediction. To handle the complexities of natural speech, including intra- and inter-subject variability, we propose Multi-level Acoustic-Textual Emotion Representation (MATER), a novel hierarchical framework that integrates acoustic and textual features at the word, utterance, and embedding levels. By fusing low-level lexical and acoustic cues with high-level contextualized representations, MATER effectively captures both fine-grained prosodic variations and semantic nuances. Additionally, we introduce an uncertainty-aware ensemble strategy to mitigate annotator inconsistencies, improving robustness in ambiguous emotional expressions. MATER ranks fourth in both tasks with a Macro-F1 of 41.01% and an average CCC of 0.5928, securing second place in valence prediction with an impressive CCC of 0.6941.
>
---
#### [new 009] An Exploration of ECAPA-TDNN and x-vector Speaker Representations in Zero-shot Multi-speaker TTS
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于零样本多说话人TTS任务，旨在比较不同说话人编码器的效果，解决如何选择有效说话人表示的问题。**

- **链接: [http://arxiv.org/pdf/2506.20190v1](http://arxiv.org/pdf/2506.20190v1)**

> **作者:** Marie Kunešová; Zdeněk Hanzlíček; Jindřich Matoušek
>
> **备注:** Accepted to TSD 2025
>
> **摘要:** Zero-shot multi-speaker text-to-speech (TTS) systems rely on speaker embeddings to synthesize speech in the voice of an unseen speaker, using only a short reference utterance. While many speaker embeddings have been developed for speaker recognition, their relative effectiveness in zero-shot TTS remains underexplored. In this work, we employ a YourTTS-based TTS system to compare three different speaker encoders - YourTTS's original H/ASP encoder, x-vector embeddings, and ECAPA-TDNN embeddings - within an otherwise fixed zero-shot TTS framework. All models were trained on the same dataset of Czech read speech and evaluated on 24 out-of-domain target speakers using both subjective and objective methods. The subjective evaluation was conducted via a listening test focused on speaker similarity, while the objective evaluation measured cosine distances between speaker embeddings extracted from synthesized and real utterances. Across both evaluations, the original H/ASP encoder consistently outperformed the alternatives, with ECAPA-TDNN showing better results than x-vectors. These findings suggest that, despite the popularity of ECAPA-TDNN in speaker recognition, it does not necessarily offer improvements for speaker similarity in zero-shot TTS in this configuration. Our study highlights the importance of empirical evaluation when reusing speaker recognition embeddings in TTS and provides a framework for additional future comparisons.
>
---
## 更新

#### [replaced 001] Representation Learning with Parameterised Quantum Circuits for Advancing Speech Emotion Recognition
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.12050v3](http://arxiv.org/pdf/2501.12050v3)**

> **作者:** Thejan Rajapakshe; Rajib Rana; Farina Riaz; Sara Khalifa; Björn W. Schuller
>
> **摘要:** Quantum machine learning (QML) offers a promising avenue for advancing representation learning in complex signal domains. In this study, we investigate the use of parameterised quantum circuits (PQCs) for speech emotion recognition (SER) a challenging task due to the subtle temporal variations and overlapping affective states in vocal signals. We propose a hybrid quantum classical architecture that integrates PQCs into a conventional convolutional neural network (CNN), leveraging quantum properties such as superposition and entanglement to enrich emotional feature representations. Experimental evaluations on three benchmark datasets IEMOCAP, RECOLA, and MSP-IMPROV demonstrate that our hybrid model achieves improved classification performance relative to a purely classical CNN baseline, with over 50% reduction in trainable parameters. This work provides early evidence of the potential for QML to enhance emotion recognition and lays the foundation for future quantum-enabled affective computing systems.
>
---
#### [replaced 002] Cross-attention Inspired Selective State Space Models for Target Sound Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.04803v5](http://arxiv.org/pdf/2409.04803v5)**

> **作者:** Donghang Wu; Yiwen Wang; Xihong Wu; Tianshu Qu
>
> **备注:** This is the preprint version of the paper published in ICASSP 2025. The final version is available at IEEE Xplore: https://ieeexplore.ieee.org/document/10890178
>
> **摘要:** The Transformer model, particularly its cross-attention module, is widely used for feature fusion in target sound extraction which extracts the signal of interest based on given clues. Despite its effectiveness, this approach suffers from low computational efficiency. Recent advancements in state space models, notably the latest work Mamba, have shown comparable performance to Transformer-based methods while significantly reducing computational complexity in various tasks. However, Mamba's applicability in target sound extraction is limited due to its inability to capture dependencies between different sequences as the cross-attention does. In this paper, we propose CrossMamba for target sound extraction, which leverages the hidden attention mechanism of Mamba to compute dependencies between the given clues and the audio mixture. The calculation of Mamba can be divided to the query, key and value. We utilize the clue to generate the query and the audio mixture to derive the key and value, adhering to the principle of the cross-attention mechanism in Transformers. Experimental results from two representative target sound extraction methods validate the efficacy of the proposed CrossMamba.
>
---
#### [replaced 003] BSM-iMagLS: ILD Informed Binaural Signal Matching for Reproduction with Head-Mounted Microphone Arrays
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.18227v2](http://arxiv.org/pdf/2501.18227v2)**

> **作者:** Or Berebi; Zamir Ben-Hur; David Lou Alon; Boaz Rafaely
>
> **备注:** 14 pages, 8 figures, Accepted to IEEE TASLP (IEEE Transactions on Audio, Speech and Language Processing, 2025)
>
> **摘要:** Headphone listening in applications such as augmented and virtual reality (AR and VR) relies on high-quality spatial audio to ensure immersion, making accurate binaural reproduction a critical component. As capture devices, wearable arrays with only a few microphones with irregular arrangement face challenges in achieving a reproduction quality comparable to that of arrays with a large number of microphones. Binaural signal matching (BSM) has recently been presented as a signal-independent approach for generating high-quality binaural signal using only a few microphones, which is further improved using magnitude-least squares (MagLS) optimization at high frequencies. This paper extends BSM with MagLS by introducing interaural level difference (ILD) into the MagLS, integrated into BSM (BSM-iMagLS). Using a deep neural network (DNN)-based solver, BSM-iMagLS achieves joint optimization of magnitude, ILD, and magnitude derivatives, improving spatial fidelity. Performance is validated through theoretical analysis, numerical simulations with diverse HRTFs and head-mounted array geometries, and listening experiments, demonstrating a substantial reduction in ILD errors while maintaining comparable magnitude accuracy to state-of-the-art solutions. The results highlight the potential of BSM-iMagLS to enhance binaural reproduction for wearable and portable devices.
>
---
#### [replaced 004] SLEEPING-DISCO 9M: A large-scale pre-training dataset for generative music modeling
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.14293v3](http://arxiv.org/pdf/2506.14293v3)**

> **作者:** Tawsif Ahmed; Andrej Radonjic; Gollam Rabby
>
> **摘要:** We present Sleeping-DISCO 9M, a large-scale pre-training dataset for music and song. To the best of our knowledge, there are no open-source high-quality dataset representing popular and well-known songs for generative music modeling tasks such as text-music, music-captioning, singing-voice synthesis, melody reconstruction and cross-model retrieval. Past contributions focused on isolated and constrained factors whose core perspective was to create synthetic or re-recorded music corpus (e.g. GTSinger, M4Singer) and arbitrarily large-scale audio datasets (e.g. DISCO-10M and LAIONDISCO-12M) had been another focus for the community. Unfortunately, adoption of these datasets has been below substantial in the generative music community as these datasets fail to reflect real-world music and its flavour. Our dataset changes this narrative and provides a dataset that is constructed using actual popular music and world-renowned artists.
>
---
#### [replaced 005] mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08400v2](http://arxiv.org/pdf/2506.08400v2)**

> **作者:** Luel Hagos Beyene; Vivek Verma; Min Ma; Jesujoba O. Alabi; Fabian David Schmidt; Joyce Nakatumba-Nabende; David Ifeoluwa Adelani
>
> **备注:** working paper
>
> **摘要:** Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
>
---
#### [replaced 006] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18671v2](http://arxiv.org/pdf/2506.18671v2)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to better maintain the relative positioning among dancers. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
