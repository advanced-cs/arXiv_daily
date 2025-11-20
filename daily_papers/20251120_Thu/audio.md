# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] LargeSHS: A large-scale dataset of music adaptation
- **分类: cs.SD**

- **简介: 论文提出LargeSHS数据集，解决参考驱动的音乐生成与改编分析问题。该数据集包含超170万条音乐元数据和90万音频链接，支持构建改编树与性能聚类，推动覆盖歌曲生成与适应感知音乐信息检索研究。**

- **链接: [https://arxiv.org/pdf/2511.15270v1](https://arxiv.org/pdf/2511.15270v1)**

> **作者:** Chih-Pin Tan; Hsuan-Kai Kao; Li Su; Yi-Hsuan Yang
>
> **备注:** submitted as an ISMIR 2025 late-breaking demo paper
>
> **摘要:** Recent advances in AI-based music generation have focused heavily on text-conditioned models, with less attention given to reference-based generation such as song adaptation. To support this line of research, we introduce LargeSHS, a large-scale dataset derived from SecondHandSongs, containing over 1.7 million metadata entries and approximately 900k publicly accessible audio links. Unlike existing datasets, LargeSHS includes structured adaptation relationships between musical works, enabling the construction of adaptation trees and performance clusters that represent cover song families. We provide comprehensive statistics and comparisons with existing datasets, highlighting the unique scale and richness of LargeSHS. This dataset paves the way for new research in cover song generation, reference-based music generation, and adaptation-aware MIR tasks.
>
---
#### [new 002] Fine-tuning Pre-trained Audio Models for COVID-19 Detection: A Technical Report
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究音频模型在新冠检测任务中的表现，旨在解决模型因人口统计学特征泄露而产生的虚假性能提升问题。作者对预训练音频模型进行微调与评估，发现严格的人口统计分层虽降低性能，但更真实反映模型能力，同时指出小样本限制了模型泛化效果。**

- **链接: [https://arxiv.org/pdf/2511.14939v1](https://arxiv.org/pdf/2511.14939v1)**

> **作者:** Daniel Oliveira de Brito; Letícia Gabriella de Souza; Marcelo Matheus Gauy; Marcelo Finger; Arnaldo Candido Junior
>
> **备注:** 11 pages
>
> **摘要:** This technical report investigates the performance of pre-trained audio models on COVID-19 detection tasks using established benchmark datasets. We fine-tuned Audio-MAE and three PANN architectures (CNN6, CNN10, CNN14) on the Coswara and COUGHVID datasets, evaluating both intra-dataset and cross-dataset generalization. We implemented a strict demographic stratification by age and gender to prevent models from exploiting spurious correlations between demographic characteristics and COVID-19 status. Intra-dataset results showed moderate performance, with Audio-MAE achieving the strongest result on Coswara (0.82 AUC, 0.76 F1-score), while all models demonstrated limited performance on Coughvid (AUC 0.58-0.63). Cross-dataset evaluation revealed severe generalization failure across all models (AUC 0.43-0.68), with Audio-MAE showing strong performance degradation (F1-score 0.00-0.08). Our experiments demonstrate that demographic balancing, while reducing apparent model performance, provides more realistic assessment of COVID-19 detection capabilities by eliminating demographic leakage - a confounding factor that inflate performance metrics. Additionally, the limited dataset sizes after balancing (1,219-2,160 samples) proved insufficient for deep learning models that typically require substantially larger training sets. These findings highlight fundamental challenges in developing generalizable audio-based COVID-19 detection systems and underscore the importance of rigorous demographic controls for clinically robust model evaluation.
>
---
#### [new 003] Voiced-Aware Style Extraction and Style Direction Adjustment for Expressive Text-to-Speech
- **分类: cs.SD; cs.AI**

- **简介: 论文提出SpotlightTTS，用于提升语音合成的表达力。通过聚焦有声区提取风格特征并调整风格方向，解决高质量情感语音合成难题，显著改善表达性、音质和风格迁移效果。**

- **链接: [https://arxiv.org/pdf/2511.14824v1](https://arxiv.org/pdf/2511.14824v1)**

> **作者:** Nam-Gyu Kim
>
> **备注:** Master's thesis, Korea University, 2025
>
> **摘要:** Recent advances in expressive text-to-speech (TTS) have introduced diverse methods based on style embedding extracted from reference speech. However, synthesizing high-quality expressive speech remains challenging. We propose SpotlightTTS, which exclusively emphasizes style via voiced-aware style extraction and style direction adjustment. Voiced-aware style extraction focuses on voiced regions highly related to style while maintaining continuity across different speech regions to improve expressiveness. We adjust the direction of the extracted style for optimal integration into the TTS model, which improves speech quality. Experimental results demonstrate that Spotlight-TTS achieves superior performance compared to baseline models in terms of expressiveness, overall speech quality, and style transfer capability.
>
---
#### [new 004] CASTELLA: Long Audio Dataset with Captions and Temporal Boundaries
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 论文提出CASTELLA，一个大规模人工标注的音频片段检索（AMR）数据集，解决现有数据集小且为合成数据导致模型性能不可靠的问题。工作包括构建24倍于先前数据集的真世界音频数据集，并建立基线模型验证其有效性。**

- **链接: [https://arxiv.org/pdf/2511.15131v1](https://arxiv.org/pdf/2511.15131v1)**

> **作者:** Hokuto Munakata; Takehiro Imamura; Taichi Nishimura; Tatsuya Komatsu
>
> **摘要:** We introduce CASTELLA, a human-annotated audio benchmark for the task of audio moment retrieval (AMR). Although AMR has various useful potential applications, there is still no established benchmark with real-world data. The early study of AMR trained the model with solely synthetic datasets. Moreover, the evaluation is based on annotated dataset of fewer than 100 samples. This resulted in less reliable reported performance. To ensure performance for applications in real-world environments, we present CASTELLA, a large-scale manually annotated AMR dataset. CASTELLA consists of 1,009, 213, and 640 audio recordings for train, valid, and test split, respectively, which is 24 times larger than the previous dataset. We also establish a baseline model for AMR using CASTELLA. Our experiments demonstrate that a model fine-tuned on CASTELLA after pre-training on the synthetic data outperformed a model trained solely on the synthetic data by 10.4 points in Recall1@0.7. CASTELLA is publicly available in https://h-munakata.github.io/CASTELLA-demo/.
>
---
#### [new 005] Quality-Controlled Multimodal Emotion Recognition in Conversations with Identity-Based Transfer Learning and MAMBA Fusion
- **分类: eess.AS; cs.AI; cs.LG; eess.IV; eess.SP**

- **简介: 论文聚焦多模态情感识别任务，解决数据质量差和低频情绪识别难的问题。通过构建质量控制管道、基于身份的迁移学习与MAMBA融合策略，提升模型性能，在MELD和IEMOCAP数据集上分别达到64.8%和74.3%准确率。**

- **链接: [https://arxiv.org/pdf/2511.14969v1](https://arxiv.org/pdf/2511.14969v1)**

> **作者:** Zanxu Wang; Homayoon Beigi
>
> **备注:** 8 pages, 14 images, 3 tables, Recognition Technologies, Inc. Technical Report RTI-20251118-01
>
> **摘要:** This paper addresses data quality issues in multimodal emotion recognition in conversation (MERC) through systematic quality control and multi-stage transfer learning. We implement a quality control pipeline for MELD and IEMOCAP datasets that validates speaker identity, audio-text alignment, and face detection. We leverage transfer learning from speaker and face recognition, assuming that identity-discriminative embeddings capture not only stable acoustic and Facial traits but also person-specific patterns of emotional expression. We employ RecoMadeEasy(R) engines for extracting 512-dimensional speaker and face embeddings, fine-tune MPNet-v2 for emotion-aware text representations, and adapt these features through emotion-specific MLPs trained on unimodal datasets. MAMBA-based trimodal fusion achieves 64.8% accuracy on MELD and 74.3% on IEMOCAP. These results show that combining identity-based audio and visual embeddings with emotion-tuned text representations on a quality-controlled subset of data yields consistent competitive performance for multimodal emotion recognition in conversation and provides a basis for further improvement on challenging, low-frequency emotion classes.
>
---
#### [new 006] Auden-Voice: General-Purpose Voice Encoder for Speech and Language Understanding
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出Auden-Voice，一种平衡捕捉语音身份与副语言特征的通用语音编码器。针对大音频语言模型中语音编码不平衡的问题，通过多任务训练实现更均衡表示，提升语音理解能力，并在与大语言模型结合时表现优异。**

- **链接: [https://arxiv.org/pdf/2511.15145v1](https://arxiv.org/pdf/2511.15145v1)**

> **作者:** Mingyue Huo; Wei-Cheng Tseng; Yiwen Shao; Hao Zhang; Dong Yu
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** Human voice encodes both identity and paralinguistic cues, yet encoders in large audio-language models (LALMs) rarely balance both aspects. In this work, we present a study toward building a general-purpose voice encoder that captures nuanced voice cues. Through a comprehensive evaluation, we find that multi-task training yields the most balanced representations, whereas contrastive language-audio pretraining (CLAP) primarily improves retrieval without enhancing paralinguistic understanding. Our final encoder, Auden-Voice, also demonstrates strong performance when integrated with LLMs. The code and training recipes will be released with the audio understanding toolkit Auden.
>
---
#### [new 007] A Novel CustNetGC Boosted Model with Spectral Features for Parkinson's Disease Prediction
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于医学诊断分类任务，旨在提高帕金森病（PD）的早期预测准确性。通过提取语音信号的光谱特征（L-mHP和频谱斜率），结合CNN与CatBoost的CustNetGC模型，在公开数据集上实现99.06%准确率，提升诊断效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2511.15485v1](https://arxiv.org/pdf/2511.15485v1)**

> **作者:** Abishek Karthik; Pandiyaraju V; Dominic Savio M; Rohit Swaminathan S
>
> **摘要:** Parkinson's disease is a neurodegenerative disorder that can be very tricky to diagnose and treat. Such early symptoms can include tremors, wheezy breathing, and changes in voice quality as critical indicators of neural damage. Notably, there has been growing interest in utilizing changes in vocal attributes as markers for the detection of PD early on. Based on this understanding, the present paper was designed to focus on the acoustic feature analysis based on voice recordings of patients diagnosed with PD and healthy controls (HC). In this paper, we introduce a novel classification and visualization model known as CustNetGC, combining a Convolutional Neural Network (CNN) with Custom Network Grad-CAM and CatBoost to enhance the efficiency of PD diagnosis. We use a publicly available dataset from Figshare, including voice recordings of 81 participants: 40 patients with PD and 41 healthy controls. From these recordings, we extracted the key spectral features: L-mHP and Spectral Slopes. The L-mHP feature combines three spectrogram representations: Log-Mel spectrogram, harmonic spectrogram, and percussive spectrogram, which are derived using Harmonic-Percussive Source Separation (HPSS). Grad-CAM was used to highlight the important regions in the data, thus making the PD predictions interpretable and effective. Our proposed CustNetGC model achieved an accuracy of 99.06% and precision of 95.83%, with the area under the ROC curve (AUC) recorded at 0.90 for the PD class and 0.89 for the HC class. Additionally, the combination of CatBoost, a gradient boosting algorithm, enhanced the robustness and the prediction performance by properly classifying PD and non-PD samples. Therefore, the results provide the potential improvement in the CustNetGC system in enhancing diagnostic accuracy and the interpretability of the Parkinson's Disease prediction model.
>
---
#### [new 008] OBHS: An Optimized Block Huffman Scheme for Real-Time Audio Compression
- **分类: cs.SD; eess.AS**

- **简介: 论文提出OBHS算法，解决实时音频流中高效压缩与低计算开销的矛盾。通过分块Huffman编码和规范码优化，在保持线性时间复杂度的同时实现高压缩比，适用于资源受限场景。**

- **链接: [https://arxiv.org/pdf/2511.14793v1](https://arxiv.org/pdf/2511.14793v1)**

> **作者:** Muntahi Safwan Mahfi; Md. Manzurul Hasan; Gahangir Hossain
>
> **备注:** 3 page, 2 figures, 2 tables
>
> **摘要:** In this paper, we introduce OBHS (Optimized Block Huffman Scheme), a novel lossless audio compression algorithm tailored for real-time streaming applications. OBHS leverages block-wise Huffman coding with canonical code representation and intelligent fallback mechanisms to achieve high compression ratios while maintaining low computational complexity. Our algorithm partitions audio data into fixed-size blocks, constructs optimal Huffman trees for each block, and employs canonical codes for efficient storage and transmission. Experimental results demonstrate that OBHS attains compression ratios of up to 93.6% for silence-rich audio and maintains competitive performance across various audio types, including pink noise, tones, and real-world recordings. With a linear time complexity of O(n) for n audio samples, OBHS effectively balances compression efficiency and computational demands, making it highly suitable for resource-constrained real-time audio streaming scenarios.
>
---
#### [new 009] IHearYou: Linking Acoustic Features to DSM-5 Depressive Behavior Indicators
- **分类: cs.SD**

- **简介: 论文提出IHearYou系统，通过分析语音声学特征自动检测抑郁症状，解决传统诊断依赖主观报告的问题。工作包括构建声学特征与DSM-5指标的关联框架、实现本地化隐私保护分析，并在真实数据集上验证可行性。**

- **链接: [https://arxiv.org/pdf/2511.14801v1](https://arxiv.org/pdf/2511.14801v1)**

> **作者:** Jonas Länzlinger; Katharina Müller; Bruno Rodrigues
>
> **摘要:** Depression affects over millions people worldwide, yet diagnosis still relies on subjective self-reports and interviews that may not capture authentic behavior. We present IHearYou, an approach to automated depression detection focused on speech acoustics. Using passive sensing in household environments, IHearYou extracts voice features and links them to DSM-5 (Diagnostic and Statistical Manual of Mental Disorders) indicators through a structured Linkage Framework instantiated for Major Depressive Disorder. The system runs locally to preserve privacy and includes a persistence schema and dashboard, presenting real-time throughput on a commodity laptop. To ensure reproducibility, we define a configuration-driven protocol with False Discovery Rate (FDR) correction and gender-stratified testing. Applied to the DAIC-WOZ dataset, this protocol reveals directionally consistent feature-indicator associations, while a TESS-based audio streaming experiment validates end-to-end feasibility. Our results show how passive voice sensing can be turned into explainable DSM-5 indicator scores, bridging the gap between black-box detection and clinically interpretable, on-device analysis.
>
---
#### [new 010] Aligning Generative Music AI with Human Preferences: Methods and Challenges
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文探讨如何通过偏好对齐技术提升生成式音乐AI与人类偏好的一致性，解决当前模型在时间连贯性、和声一致性和主观质量评估上的不足，提出多种方法并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2511.15038v1](https://arxiv.org/pdf/2511.15038v1)**

> **作者:** Dorien Herremans; Abhinaba Roy
>
> **备注:** Accepted at the AAAI-2026 Senior Member Track
>
> **摘要:** Recent advances in generative AI for music have achieved remarkable fidelity and stylistic diversity, yet these systems often fail to align with nuanced human preferences due to the specific loss functions they use. This paper advocates for the systematic application of preference alignment techniques to music generation, addressing the fundamental gap between computational optimization and human musical appreciation. Drawing on recent breakthroughs including MusicRL's large-scale preference learning, multi-preference alignment frameworks like diffusion-based preference optimization in DiffRhythm+, and inference-time optimization techniques like Text2midi-InferAlign, we discuss how these techniques can address music's unique challenges: temporal coherence, harmonic consistency, and subjective quality assessment. We identify key research challenges including scalability to long-form compositions, reliability amongst others in preference modelling. Looking forward, we envision preference-aligned music generation enabling transformative applications in interactive composition tools and personalized music services. This work calls for sustained interdisciplinary research combining advances in machine learning, music-theory to create music AI systems that truly serve human creative and experiential needs.
>
---
## 更新

#### [replaced 001] Scene-wide Acoustic Parameter Estimation
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2410.23523v3](https://arxiv.org/pdf/2410.23523v3)**

> **作者:** Ricardo Falcon-Perez; Ruohan Gao; Gregor Mueckl; Sebastia V. Amengual Gari; Ishwarya Ananthabhotla
>
> **备注:** Published in WASPAA 2025
>
> **摘要:** For augmented (AR) and virtual reality (VR) applications, accurate estimates of the acoustic characteristics of a scene are critical for creating a sense of immersion. However, directly estimating Room-impulse Responses (RIRs) from scene geometry is often a challenging, data-expensive task. We propose a method to instead infer spatially-distributed acoustic parameters (such as C50, T60, etc) for an entire scene from lightweight information readily available in an AR/VR context. We consider an image-to-image translation task to transform a 2D floormap, conditioned on a calibration RIR measurement, into 2D heatmaps of acoustic parameters. Moreover, we show that the method also works for directionally-dependent (i.e. beamformed) parameter prediction. We introduce and release a 1000-room, complex-scene dataset to study the task, and demonstrate improvements over strong statistical baselines.
>
---
#### [replaced 002] Efficient and Generalizable Speaker Diarization via Structured Pruning of Self-Supervised Models
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2506.18623v2](https://arxiv.org/pdf/2506.18623v2)**

> **作者:** Jiangyu Han; Petr Pálka; Marc Delcroix; Federico Landini; Johan Rohdin; Jan Cernocký; Lukáš Burget
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Self-supervised learning (SSL) models such as WavLM have substantially advanced speaker diarization by providing rich contextual speech representations. However, the high computational and memory costs of these models hinder deployment in real-time and resource-constrained scenarios. This work presents a systematic study on compressing SSL-based diarization models through structured pruning guided by knowledge distillation. We investigate pruning objectives that target both model parameters and computational complexity, and analyze alternative strategies, showing that a simple overall pruning approach provides the best balance between efficiency and accuracy. Our method achieves up to 80% model size reduction and 4x faster inference without performance degradation. Comprehensive experiments across eight public diarization datasets demonstrate that the pruned models consistently match or surpass the performance of their uncompressed counterparts. Furthermore, we show strong out-of-domain generalization on the CHiME-6 dataset, achieving accuracy comparable to the top systems in the CHiME-7 challenge without any domain adaptation. These results highlight that structured pruning, when guided by distillation, can yield efficient and generalizable diarization systems suitable for real-world applications.
>
---
#### [replaced 003] Retrieval Augmented Generation based context discovery for ASR
- **分类: cs.CL; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.19567v2](https://arxiv.org/pdf/2509.19567v2)**

> **作者:** Dimitrios Siskos; Stavros Papadopoulos; Pablo Peso Parada; Jisi Zhang; Karthikeyan Saravanan; Anastasios Drosou
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** This work investigates retrieval augmented generation as an efficient strategy for automatic context discovery in context-aware Automatic Speech Recognition (ASR) system, in order to improve transcription accuracy in the presence of rare or out-of-vocabulary terms. However, identifying the right context automatically remains an open challenge. This work proposes an efficient embedding-based retrieval approach for automatic context discovery in ASR. To contextualize its effectiveness, two alternatives based on large language models (LLMs) are also evaluated: (1) large language model (LLM)-based context generation via prompting, and (2) post-recognition transcript correction using LLMs. Experiments on the TED-LIUMv3, Earnings21 and SPGISpeech demonstrate that the proposed approach reduces WER by up to 17% (percentage difference) relative to using no-context, while the oracle context results in a reduction of up to 24.1%.
>
---
#### [replaced 004] Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models
- **分类: cs.LG; cs.AI; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.05171v2](https://arxiv.org/pdf/2511.05171v2)**

> **作者:** Davide Marincione; Donato Crisostomi; Roberto Dessi; Emanuele Rodolà; Emanuele Rossi
>
> **摘要:** Foundation models capable of generalizing across species and tasks represent a promising new frontier in bioacoustics, with NatureLM being one of the most prominent examples. While its domain-specific fine-tuning yields strong performance on bioacoustic benchmarks, we observe that it also introduces trade-offs in instruction-following flexibility. For instance, NatureLM achieves high accuracy when prompted for either the common or scientific name individually, but its accuracy drops significantly when both are requested in a single prompt. We address this by applying a simple model merging strategy that interpolates NatureLM with its base language model, recovering instruction-following capabilities with minimal loss of domain expertise. Finally, we show that the merged model exhibits markedly stronger zero-shot generalization, achieving over a 200% relative improvement and setting a new state-of-the-art in closed-set zero-shot classification of unseen species.
>
---
#### [replaced 005] Regularized Schrödinger Bridge: Alleviating Distortion and Exposure Bias in Solving Inverse Problems
- **分类: cs.LG; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.11686v3](https://arxiv.org/pdf/2511.11686v3)**

> **作者:** Qing Yao; Lijian Gao; Qirong Mao; Ming Dong
>
> **摘要:** Diffusion models serve as a powerful generative framework for solving inverse problems. However, they still face two key challenges: 1) the distortion-perception tradeoff, where improving perceptual quality often degrades reconstruction fidelity, and 2) the exposure bias problem, where the training-inference input mismatch leads to prediction error accumulation and reduced reconstruction quality. In this work, we propose the Regularized Schrödinger Bridge (RSB), an adaptation of Schrödinger Bridge tailored for inverse problems that addresses the above limitations. RSB employs a novel regularized training strategy that perturbs both the input states and targets, effectively mitigating exposure bias by exposing the model to simulated prediction errors and also alleviating distortion by well-designed interpolation via the posterior mean. Extensive experiments on two typical inverse problems for speech enhancement demonstrate that RSB outperforms state-of-the-art methods, significantly improving distortion metrics and effectively reducing exposure bias.
>
---
#### [replaced 006] Bridging the Modality Gap: Softly Discretizing Audio Representation for LLM-based Automatic Speech Recognition
- **分类: eess.AS**

- **链接: [https://arxiv.org/pdf/2506.05706v2](https://arxiv.org/pdf/2506.05706v2)**

> **作者:** Mu Yang; Szu-Jui Chen; Jiamin Xie; John Hansen
>
> **备注:** ASRU 2025
>
> **摘要:** One challenge of integrating speech input with large language models (LLMs) stems from the discrepancy between the continuous nature of audio data and the discrete token-based paradigm of LLMs. To mitigate this gap, we propose a method for integrating vector quantization (VQ) into LLM-based automatic speech recognition (ASR). Using the LLM embedding table as the VQ codebook, the VQ module aligns the continuous representations from the audio encoder with the discrete LLM inputs, enabling the LLM to operate on a discretized audio representation that better reflects the linguistic structure. We further create a soft "discretization" of the audio representation by updating the codebook and performing a weighted sum over the codebook embeddings. Empirical results demonstrate that our proposed method significantly improves upon the LLM-based ASR baseline, particularly in out-of-domain conditions. This work highlights the potential of soft discretization as a modality bridge in LLM-based ASR.
>
---
#### [replaced 007] Step-Audio-EditX Technical Report
- **分类: cs.CL; cs.AI; cs.HC; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.03601v2](https://arxiv.org/pdf/2511.03601v2)**

> **作者:** Chao Yan; Boyong Wu; Peng Yang; Pengfei Tan; Guoqiang Hu; Li Xie; Yuxin Zhang; Xiangyu; Zhang; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Shuchang Zhou; Gang Yu
>
> **摘要:** We present Step-Audio-EditX, the first open-source LLM-based audio model excelling at expressive and iterative audio editing encompassing emotion, speaking style, and paralinguistics alongside robust zero-shot text-to-speech (TTS) capabilities. Our core innovation lies in leveraging only large-margin synthetic data, which circumvents the need for embedding-based priors or auxiliary modules. This large-margin learning approach enables both iterative control and high expressivity across voices, and represents a fundamental pivot from the conventional focus on representation-level disentanglement. Evaluation results demonstrate that Step-Audio-EditX surpasses both MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 in emotion editing and other fine-grained control tasks.
>
---
#### [replaced 008] AcousTools: A 'Full-Stack', Python-Based, Acoustic Holography Library
- **分类: cs.SD; cs.ET**

- **链接: [https://arxiv.org/pdf/2511.07336v4](https://arxiv.org/pdf/2511.07336v4)**

> **作者:** Joshua Mukherjee; Giorgos Christopoulos; Zhouyang Shen; Sriram Subramanian; Ryuji Hirayama
>
> **备注:** 14 Pages, 7 Figures, 1 Table
>
> **摘要:** Acoustic Holography is an emerging field where mid-air ultrasound is controlled and manipulated for novel and exciting applications. These range from mid-air haptics, volumetric displays, contactless fabrication, and even chemical and biomedical applications such as drug delivery. To develop these applications, a software framework to predict acoustic behaviour and simulating resulting effects, such as applied forces or scattering patterns is desirable. There have been various software libraries and platforms that attempt to fill this role, but there is yet to be a single piece of software that acts as a 'full-stack' solution. We define this full-stack as the process from abstraction to physicalisation starting with setup, modelling acoustic propagation, transducer phase retrieval, sound field analysis, and control of the acoustic holographic hardware itself. Existing methods fail to fulfil one or more of these categories. To address this, we present AcousTools, a Python-based acoustic holography library, designed to support the full suite of acoustic holographic applications and we show AcousTools's ability to meet each step of the full-stack's requirements. AcousTools has the potential to become the standard code library for acoustic holography, with the uniquely complete suite of features wrapped in a language that is known to be easy to use, AcousTools will increase the ability for researchers to develop novel applications as well as accurately review other's work. The full-stack, aside from software, will also be useful for researchers - providing a way to view and compare methodologies by understanding where they fit into the stack.
>
---
#### [replaced 009] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement
- **分类: cs.SD; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.12074v2](https://arxiv.org/pdf/2511.12074v2)**

> **作者:** Xinyue Yu; Youqing Fang; Pingyu Wu; Guoyang Ye; Wenbo Zhou; Weiming Zhang; Song Xiao
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Generating expressive and controllable human speech is one of the core goals of generative artificial intelligence, but its progress has long been constrained by two fundamental challenges: the deep entanglement of speech factors and the coarse granularity of existing control mechanisms. To overcome these challenges, we have proposed a novel framework called MF-Speech, which consists of two core components: MF-SpeechEncoder and MF-SpeechGenerator. MF-SpeechEncoder acts as a factor purifier, adopting a multi-objective optimization strategy to decompose the original speech signal into highly pure and independent representations of content, timbre, and emotion. Subsequently, MF-SpeechGenerator functions as a conductor, achieving precise, composable and fine-grained control over these factors through dynamic fusion and Hierarchical Style Adaptive Normalization (HSAN). Experiments demonstrate that in the highly challenging multi-factor compositional speech generation task, MF-Speech significantly outperforms current state-of-the-art methods, achieving a lower word error rate (WER=4.67%), superior style control (SECS=0.5685, Corr=0.68), and the highest subjective evaluation scores(nMOS=3.96, sMOS_emotion=3.86, sMOS_style=3.78). Furthermore, the learned discrete factors exhibit strong transferability, demonstrating their significant potential as a general-purpose speech representation.
>
---
#### [replaced 010] MelodySim: Measuring Melody-aware Music Similarity for Plagiarism Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [https://arxiv.org/pdf/2505.20979v2](https://arxiv.org/pdf/2505.20979v2)**

> **作者:** Tongyu Lu; Charlotta-Marlena Geist; Jan Melechovsky; Abhinaba Roy; Dorien Herremans
>
> **摘要:** We propose MelodySim, a melody-aware music similarity model and dataset for plagiarism detection. First, we introduce a novel method to construct a dataset focused on melodic similarity. By augmenting Slakh2100, an existing MIDI dataset, we generate variations of each piece while preserving the melody through modifications such as note splitting, arpeggiation, minor track dropout, and re-instrumentation. A user study confirms that positive pairs indeed contain similar melodies, while other musical tracks are significantly changed. Second, we develop a segment-wise melodic-similarity detection model that uses a MERT encoder and applies a triplet neural network to capture melodic similarity. The resulting decision matrix highlights where plagiarism might occur. The experiments show that our model is able to outperform baseline models in detecting similar melodic fragments on the MelodySim test set.
>
---
#### [replaced 011] UniAV: Unified Audio-Visual Perception for Multi-Task Video Event Localization
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2404.03179v3](https://arxiv.org/pdf/2404.03179v3)**

> **作者:** Tiantian Geng; Teng Wang; Jinming Duan; Yanfu Zhang; Weili Guan; Feng Zheng; Ling shao
>
> **备注:** Published on IEEE TPAMI
>
> **摘要:** Video event localization tasks include temporal action localization (TAL), sound event detection (SED) and audio-visual event localization (AVEL). Existing methods tend to over-specialize on individual tasks, neglecting the equal importance of these different events for a complete understanding of video content. In this work, we aim to develop a unified framework to solve TAL, SED and AVEL tasks together to facilitate holistic video understanding. However, it is challenging since different tasks emphasize distinct event characteristics and there are substantial disparities in existing task-specific datasets (size/domain/duration). It leads to unsatisfactory results when applying a naive multi-task strategy. To tackle the problem, we introduce UniAV, a Unified Audio-Visual perception network to effectively learn and share mutually beneficial knowledge across tasks and modalities. Concretely, we propose a unified audio-visual encoder to derive generic representations from multiple temporal scales for videos from all tasks. Meanwhile, task-specific experts are designed to capture the unique knowledge specific to each task. Besides, instead of using separate prediction heads, we develop a novel unified language-aware classifier by utilizing semantic-aligned task prompts, enabling our model to flexibly localize various instances across tasks with an impressive open-set ability to localize novel categories. Extensive experiments demonstrate that UniAV, with its unified architecture, significantly outperforms both single-task models and the naive multi-task baseline across all three tasks. It achieves superior or on-par performances compared to the state-of-the-art task-specific methods on ActivityNet 1.3, DESED and UnAV-100 benchmarks.
>
---
