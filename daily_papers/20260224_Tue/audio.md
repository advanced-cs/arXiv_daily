# 音频 cs.SD;  eess.AS

- **最新发布 23 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，解决如何理解其编码的语音信息问题。通过分析96种语言，发现模型使用可解释的音素向量进行运算，实现音素向量算术。**

- **链接: [https://arxiv.org/pdf/2602.18899v1](https://arxiv.org/pdf/2602.18899v1)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Submitted to ACL, code planned to release after acceptance
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at https://github.com/juice500ml/phonetic-arithmetic .
>
---
#### [new 002] SongEcho: Towards Cover Song Generation via Instance-Adaptive Element-wise Linear Modulation
- **分类: cs.SD**

- **简介: 该论文提出SongEcho，用于生成伴奏和新演唱的翻唱歌曲。解决翻唱生成任务中的条件生成问题，通过IA-EiLM框架提升控制生成效果。**

- **链接: [https://arxiv.org/pdf/2602.19976v1](https://arxiv.org/pdf/2602.19976v1)**

> **作者:** Sifei Li; Yang Li; Zizhou Wang; Yuxin Zhang; Fuzhang Wu; Oliver Deussen; Tong-Yee Lee; Weiming Dong
>
> **备注:** Accepted at ICLR 2026. 21 pages (10 pages main text), 5 figures
>
> **摘要:** Cover songs constitute a vital aspect of musical culture, preserving the core melody of an original composition while reinterpreting it to infuse novel emotional depth and thematic emphasis. Although prior research has explored the reinterpretation of instrumental music through melody-conditioned text-to-music models, the task of cover song generation remains largely unaddressed. In this work, we reformulate our cover song generation as a conditional generation, which simultaneously generates new vocals and accompaniment conditioned on the original vocal melody and text prompts. To this end, we present SongEcho, which leverages Instance-Adaptive Element-wise Linear Modulation (IA-EiLM), a framework that incorporates controllable generation by improving both conditioning injection mechanism and conditional representation. To enhance the conditioning injection mechanism, we extend Feature-wise Linear Modulation (FiLM) to an Element-wise Linear Modulation (EiLM), to facilitate precise temporal alignment in melody control. For conditional representations, we propose Instance-Adaptive Condition Refinement (IACR), which refines conditioning features by interacting with the hidden states of the generative model, yielding instance-adaptive conditioning. Additionally, to address the scarcity of large-scale, open-source full-song datasets, we construct Suno70k, a high-quality AI song dataset enriched with comprehensive annotations. Experimental results across multiple datasets demonstrate that our approach generates superior cover songs compared to existing methods, while requiring fewer than 30% of the trainable parameters. The code, dataset, and demos are available at https://github.com/lsfhuihuiff/SongEcho_ICLR2026.
>
---
#### [new 003] AuditoryHuM: Auditory Scene Label Generation and Clustering using Human-MLLM Collaboration
- **分类: cs.SD**

- **简介: 该论文属于音频场景标签生成与聚类任务，解决人工标注耗时及标签粒度与可分性平衡问题。通过Human-MLLM协作生成标签，并优化聚类方法，提升标签质量与实用性。**

- **链接: [https://arxiv.org/pdf/2602.19409v1](https://arxiv.org/pdf/2602.19409v1)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard F. Lyon
>
> **摘要:** Manual annotation of audio datasets is labour intensive, and it is challenging to balance label granularity with acoustic separability. We introduce AuditoryHuM, a novel framework for the unsupervised discovery and clustering of auditory scene labels using a collaborative Human-Multimodal Large Language Model (MLLM) approach. By leveraging MLLMs (Gemma and Qwen) the framework generates contextually relevant labels for audio data. To ensure label quality and mitigate hallucinations, we employ zero-shot learning techniques (Human-CLAP) to quantify the alignment between generated text labels and raw audio content. A strategically targeted human-in-the-loop intervention is then used to refine the least aligned pairs. The discovered labels are grouped into thematically cohesive clusters using an adjusted silhouette score that incorporates a penalty parameter to balance cluster cohesion and thematic granularity. Evaluated across three diverse auditory scene datasets (ADVANCE, AHEAD-DS, and TAU 2019), AuditoryHuM provides a scalable, low-cost solution for creating standardised taxonomies. This solution facilitates the training of lightweight scene recognition models deployable to edge devices, such as hearing aids and smart home assistants. The project page and code: https://github.com/Australian-Future-Hearing-Initiative
>
---
#### [new 004] DECAF: Dynamic Envelope Context-Aware Fusion for Speech-Envelope Reconstruction from EEG
- **分类: cs.SD; eess.SP**

- **简介: 该论文属于语音-envelope重建任务，旨在提升从EEG中重构语音的准确性。针对现有方法忽略时间结构的问题，提出动态融合框架，结合神经信号与语音上下文，实现更精确的解码。**

- **链接: [https://arxiv.org/pdf/2602.19395v1](https://arxiv.org/pdf/2602.19395v1)**

> **作者:** Karan Thakkar; Mounya Elhilali
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Reconstructing the speech audio envelope from scalp neural recordings (EEG) is a central task for decoding a listener's attentional focus in applications like neuro-steered hearing aids. Current methods for this reconstruction, however, face challenges with fidelity and noise. Prevailing approaches treat it as a static regression problem, processing each EEG window in isolation and ignoring the rich temporal structure inherent in continuous speech. This study introduces a new, dynamic framework for envelope reconstruction that leverages this structure as a predictive temporal prior. We propose a state-space fusion model that combines direct neural estimates from EEG with predictions from recent speech context, using a learned gating mechanism to adaptively balance these cues. To validate this approach, we evaluate our model on the ICASSP 2023 Stimulus Reconstruction benchmark demonstrating significant improvements over static, EEG-only baselines. Our analyses reveal a powerful synergy between the neural and temporal information streams. Ultimately, this work reframes envelope reconstruction not as a simple mapping, but as a dynamic state-estimation problem, opening a new direction for developing more accurate and coherent neural decoding systems.
>
---
#### [new 005] RA-QA: Towards Respiratory Audio-based Health Question Answering
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出RA-QA数据集，解决呼吸音频与自然语言问答的结合问题，旨在开发智能诊断工具。**

- **链接: [https://arxiv.org/pdf/2602.18452v1](https://arxiv.org/pdf/2602.18452v1)**

> **作者:** Gaia A. Bertolino; Yuwei Zhang; Tong Xia; Domenico Talia; Cecilia Mascolo
>
> **摘要:** Respiratory diseases are a leading cause of death globally, highlighting the urgent need for early and accessible screening methods. While some lung auscultation analysis has been automated and machine learning audio based models are able to predict respiratory pathologies, there remains a critical gap: the lack of intelligent systems that can interact in real-time consultations using natural language. Unlike other clinical domains, such as electronic health records, radiological images, and biosignals, where numerous question-answering (QA) datasets and models have been established, audio-based modalities remain notably underdeveloped. We curated and harmonized data from 11 diverse respiratory audio datasets to construct the first Respiratory Audio Question Answering (RA-QA) dataset. As the first multimodal QA resource of its kind focused specifically on respiratory health, RA-QA bridges clinical audio and natural language in a structured, scalable format. This new data resource contains about 7.5 million QA pairs spanning more than 60 attributes and three question types: single verification, multiple choice, and open-ended questions. Building upon this dataset, we introduce a novel benchmark that compares audio-text generation models with traditional audio classifiers to evaluate their respective performance.\\Our experiments reveal interesting performance variations across different attributes and question types, establishing a baseline and paving the way for more advanced architectures that could further improve the performance. By bridging machine learning with real-world clinical dialogue, our work opens the door to the development of more interactive, intelligent, and accessible diagnostic tools in respiratory healthcare.
>
---
#### [new 006] Continuous Telemonitoring of Heart Failure using Personalised Speech Dynamics
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于心力衰竭远程监测任务，解决个体语音特征差异导致的模型准确性低问题，提出LIPT框架和PSE模型以提升监测效果。**

- **链接: [https://arxiv.org/pdf/2602.19674v1](https://arxiv.org/pdf/2602.19674v1)**

> **作者:** Yue Pan; Xingyao Wang; Hanyue Zhang; Liwei Liu; Changxin Li; Gang Yang; Rong Sheng; Yili Xia; Ming Chu
>
> **摘要:** Remote monitoring of heart failure (HF) via speech signals provides a non-invasive and cost-effective solution for long-term patient management. However, substantial inter-individual heterogeneity in vocal characteristics often limits the accuracy of traditional cross-sectional classification models. To address this, we propose a Longitudinal Intra-Patient Tracking (LIPT) scheme designed to capture the trajectory of relative symptomatic changes within individuals. Central to this framework is a Personalised Sequential Encoder (PSE), which transforms longitudinal speech recordings into context-aware latent representations. By incorporating historical data at each timestamp, the PSE facilitates a holistic assessment of the clinical trajectory rather than modelling discrete visits independently. Experimental results from a cohort of 225 patients demonstrate that the LIPT paradigm significantly outperforms the classic cross-sectional approaches, achieving a recognition accuracy of 99.7% for clinical status transitions. The model's high sensitivity was further corroborated by additional follow-up data, confirming its efficacy in predicting HF deterioration and its potential to secure patient safety in remote, home-based settings. Furthermore, this work addresses the gap in existing literature by providing a comprehensive analysis of different speech task designs and acoustic features. Taken together, the superior performance of the LIPT framework and PSE architecture validates their readiness for integration into long-term telemonitoring systems, offering a scalable solution for remote heart failure management.
>
---
#### [new 007] MDM-ASR: Bridging Accuracy and Efficiency in ASR with Diffusion-Based Non-Autoregressive Decoding
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决NAR模型精度不足与AR模型效率低的问题。提出基于扩散模型的NAR框架，提升精度并保持并行解码效率。**

- **链接: [https://arxiv.org/pdf/2602.18952v1](https://arxiv.org/pdf/2602.18952v1)**

> **作者:** Hao Yen; Pin-Jui Ku; Ante Jukić; Sabato Marco Siniscalchi
>
> **备注:** 10 pages, submitted to Interspeech 2026 Long Paper track
>
> **摘要:** In sequence-to-sequence Transformer ASR, autoregressive (AR) models achieve strong accuracy but suffer from slow decoding, while non-autoregressive (NAR) models enable parallel decoding at the cost of degraded performance. We propose a principled NAR ASR framework based on Masked Diffusion Models to reduce this gap. A pre-trained speech encoder is coupled with a Transformer diffusion decoder conditioned on acoustic features and partially masked transcripts for parallel token prediction. To mitigate the training-inference mismatch, we introduce Iterative Self-Correction Training that exposes the model to its own intermediate predictions. We also design a Position-Biased Entropy-Bounded Confidence-based sampler with positional bias to further boost results. Experiments across multiple benchmarks demonstrate consistent gains over prior NAR models and competitive performance with strong AR baselines, while retaining parallel decoding efficiency.
>
---
#### [new 008] DTT-BSR: GAN-based DTTNet with RoPE Transformer Enhancement for Music Source Restoration
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音乐源分离任务，旨在恢复混音中的原始音轨。提出DTT-BSR模型，结合Transformer和RNN，提升分离效果。**

- **链接: [https://arxiv.org/pdf/2602.19825v1](https://arxiv.org/pdf/2602.19825v1)**

> **作者:** Shihong Tan; Haoyu Wang; Youran Ni; Yingzhao Hou; Jiayue Luo; Zipei Hu; Han Dou; Zerui Han; Ningning Pan; Yuzhu Wang; Gongping Huang
>
> **摘要:** Music source restoration (MSR) aims to recover unprocessed stems from mixed and mastered recordings. The challenge lies in both separating overlapping sources and reconstructing signals degraded by production effects such as compression and reverberation. We therefore propose DTT-BSR, a hybrid generative adversarial network (GAN) combining rotary positional embeddings (RoPE) transformer for long-term temporal modeling with dual-path band-split recurrent neural network (RNN) for multi-resolution spectral processing. Our model achieved 3rd place on the objective leaderboard and 4th place on the subjective leaderboard on the ICASSP 2026 MSR Challenge, demonstrating exceptional generation fidelity and semantic alignment with a compact size of 7.1M parameters.
>
---
#### [new 009] Fairness-Aware Partial-label Domain Adaptation for Voice Classification of Parkinson's and ALS
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于跨域语音分类任务，旨在解决部分标签不匹配和性别不公平问题。提出一种混合框架，提升模型在不同群体间的泛化能力和公平性。**

- **链接: [https://arxiv.org/pdf/2602.18535v1](https://arxiv.org/pdf/2602.18535v1)**

> **作者:** Arianna Francesconi; Zhixiang Dai; Arthur Stefano Moscheni; Himesh Morgan Perera Kanattage; Donato Cappetta; Fabio Rebecchi; Paolo Soda; Valerio Guarrasi; Rosa Sicilia; Mary-Anne Hartley
>
> **备注:** 7 pages, 1 figure. Submitted to Pattern Recognition Letters
>
> **摘要:** Voice-based digital biomarkers can enable scalable, non-invasive screening and monitoring of Parkinson's disease (PD) and Amyotrophic Lateral Sclerosis (ALS). However, models trained on one cohort or device often fail on new acquisition settings due to cross-device and cross-cohort domain shift. This challenge is amplified in real-world scenarios with partial-label mismatch, where datasets may contain different disease labels and only partially overlap in class space. In addition, voice-based models may exploit demographic cues, raising concerns about gender-related unfairness, particularly when deployed across heterogeneous cohorts. To tackle these challenges, we propose a hybrid framework for unified three-class (healthy/PD/ALS) cross-domain voice classification from partially overlapping cohorts. The method combines style-based domain generalization with conditional adversarial alignment tailored to partial-label settings, reducing negative transfer. An additional adversarial gender branch promotes gender-invariant representations. We conduct a comprehensive evaluation across four heterogeneous sustained-vowel datasets, spanning distinct acquisition settings and devices, under both domain generalization and unsupervised domain adaptation protocols. The proposed approach is compared against twelve state-of-the-art machine learning and deep learning methods, and further evaluated through three targeted ablations, providing the first cross-cohort benchmark and end-to-end domain-adaptive framework for unified healthy/PD/ALS voice classification under partial-label mismatch and fairness constraints. Across all experimental settings, our method consistently achieves the best external generalization over the considered evaluation metrics, while maintaining reduced gender disparities. Notably, no competing method shows statistically significant gains in external performance.
>
---
#### [new 010] Musical Training, but not Mere Exposure to Music, Drives the Emergence of Chroma Equivalence in Artificial Neural Networks
- **分类: cs.SD; cs.NE**

- **简介: 该论文属于语音与音乐感知研究，探讨 chroma equivalence 是否由音乐训练引发。通过分析神经网络，发现只有受过音乐训练的模型才表现出 chroma equivalence，证明其为高级认知功能。**

- **链接: [https://arxiv.org/pdf/2602.18635v1](https://arxiv.org/pdf/2602.18635v1)**

> **作者:** Lukas Grasse; Matthew S. Tata
>
> **摘要:** Pitch is a fundamental aspect of auditory perception. Pitch perception is commonly described across two perceptual dimensions: pitch height is the sense that tones with varying frequencies seem to be higher or lower, and chroma equivalence is the cyclical similarity of notes octaves, corresponding to a doubling of fundamental frequency. Existing research is divided on whether chroma equivalence is a learned percept that varies according to musical experience and culture, or is an innate percept that develops automatically. Building on a recent framework that proposes to use ANNs to ask 'why' questions about the brain, we evaluated recent auditory ANNs using representational similarity analysis to test the emergence of pitch height and chroma equivalence in their learned representations. Additionally, we fine-tuned two models, Wav2Vec 2.0 and Data2Vec, on a self-supervised learning task using speech and music, and a supervised music transcription task. We found that all models exhibited varying degrees of pitch height representation, but that only models trained on the supervised music transcription task exhibited chroma equivalence. Mere exposure to music through self-supervised learning was not sufficient for chroma equivalence to emerge. This supports the view that chroma equivalence is a higher-order cognitive computation that emerges to support the specific task of music perception, distinct from other auditory perception such as speech listening. This work also highlights the usefulness of ANNs for probing the developmental conditions that give rise to perceptual representations in humans.
>
---
#### [new 011] Enhancing Automatic Chord Recognition via Pseudo-Labeling and Knowledge Distillation
- **分类: cs.SD; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于自动和弦识别任务，旨在解决标注数据稀缺的问题。通过伪标签和知识蒸馏方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.19778v1](https://arxiv.org/pdf/2602.19778v1)**

> **作者:** Nghia Phan; Rong Jin; Gang Liu; Xiao Dong
>
> **备注:** 9 pages, 6 figures, 3 tables
>
> **摘要:** Automatic Chord Recognition (ACR) is constrained by the scarcity of aligned chord labels, as well-aligned annotations are costly to acquire. At the same time, open-weight pre-trained models are currently more accessible than their proprietary training data. In this work, we present a two-stage training pipeline that leverages pre-trained models together with unlabeled audio. The proposed method decouples training into two stages. In the first stage, we use a pre-trained BTC model as a teacher to generate pseudo-labels for over 1,000 hours of diverse unlabeled audio and train a student model solely on these pseudo-labels. In the second stage, the student is continually trained on ground-truth labels as they become available, with selective knowledge distillation (KD) from the teacher applied as a regularizer to prevent catastrophic forgetting of the representations learned in the first stage. In our experiments, two models (BTC, 2E1D) were used as students. In stage 1, using only pseudo-labels, the BTC student achieves over 98% of the teacher's performance, while the 2E1D model achieves about 96% across seven standard mir_eval metrics. After a single training run for both students in stage 2, the resulting BTC student model surpasses the traditional supervised learning baseline by 2.5% and the original pre-trained teacher model by 1.55% on average across all metrics. And the resulting 2E1D student model improves from the traditional supervised learning baseline by 3.79% on average and achieves almost the same performance as the teacher. Both cases show the large gains on rare chord qualities.
>
---
#### [new 012] StyleStream: Real-Time Zero-Shot Voice Style Conversion
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音风格转换任务，旨在实现零样本实时风格转换。提出StyleStream系统，通过分离内容与风格，解决转换质量与实时性问题。**

- **链接: [https://arxiv.org/pdf/2602.20113v1](https://arxiv.org/pdf/2602.20113v1)**

> **作者:** Yisi Liu; Nicholas Lee; Gopala Anumanchipalli
>
> **摘要:** Voice style conversion aims to transform an input utterance to match a target speaker's timbre, accent, and emotion, with a central challenge being the disentanglement of linguistic content from style. While prior work has explored this problem, conversion quality remains limited, and real-time voice style conversion has not been addressed. We propose StyleStream, the first streamable zero-shot voice style conversion system that achieves state-of-the-art performance. StyleStream consists of two components: a Destylizer, which removes style attributes while preserving linguistic content, and a Stylizer, a diffusion transformer (DiT) that reintroduces target style conditioned on reference speech. Robust content-style disentanglement is enforced through text supervision and a highly constrained information bottleneck. This design enables a fully non-autoregressive architecture, achieving real-time voice style conversion with an end-to-end latency of 1 second. Samples and real-time demo: https://berkeley-speech-group.github.io/StyleStream/.
>
---
#### [new 013] Depth-Structured Music Recurrence: Budgeted Recurrent Attention for Full-Piece Symbolic Music Modeling
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于符号音乐建模任务，旨在解决长上下文建模在资源受限设备上的应用问题。提出DSMR模型，通过分层记忆调度实现高效长序列生成。**

- **链接: [https://arxiv.org/pdf/2602.19816v1](https://arxiv.org/pdf/2602.19816v1)**

> **作者:** Yungang Yi
>
> **摘要:** Long-context modeling is essential for symbolic music generation, since motif repetition and developmental variation can span thousands of musical events. However, practical composition and performance workflows frequently rely on resource-limited devices (e.g., electronic instruments and portable computers), making heavy memory and attention computation difficult to deploy. We introduce Depth-Structured Music Recurrence (DSMR), a recurrent long-context Transformer for full-piece symbolic music modeling that extends context beyond fixed-length excerpts via segment-level recurrence with detached cross-segment states, featuring a layer-wise memory-horizon schedule that budgets recurrent KV states across depth. DSMR is trained in a single left-to-right pass over each complete composition, akin to how a musician experiences it from beginning to end, while carrying recurrent cross-segment states forward. Within this recurrent framework, we systematically study how depth-wise horizon allocations affect optimization, best-checkpoint perplexity, and efficiency. By allocating different history-window lengths across layers while keeping the total recurrent-state budget fixed, DSMR creates depth-dependent temporal receptive fields within a recurrent attention stack without reducing compute depth. Our main instantiation is a two-scale DSMR schedule that allocates long history windows to lower layers and a uniform short window to the remaining layers. Experiments on the piano performance dataset MAESTRO demonstrate that two-scale DSMR provides a practical quality--efficiency recipe for full-length long-context symbolic music modeling with recurrent attention under limited computational resources.
>
---
#### [new 014] Mind the Gap: Detecting Cluster Exits for Robust Local Density-Based Score Normalization in Anomalous Sound Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于异常声音检测任务，解决局部密度归一化中邻域大小选择问题。通过检测聚类退出点，动态调整邻域大小，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.18777v1](https://arxiv.org/pdf/2602.18777v1)**

> **作者:** Kevin Wilkinghoff; Gordon Wichern; Jonathan Le Roux; Zheng-Hua Tan
>
> **摘要:** Local density-based score normalization is an effective component of distance-based embedding methods for anomalous sound detection, particularly when data densities vary across conditions or domains. In practice, however, performance depends strongly on neighborhood size. Increasing it can degrade detection accuracy when neighborhood expansion crosses cluster boundaries, violating the locality assumption of local density estimation. This observation motivates adapting the neighborhood size based on locality preservation rather than fixing it in advance. We realize this by proposing cluster exit detection, a lightweight mechanism that identifies distance discontinuities and selects neighborhood sizes accordingly. Experiments across multiple embedding models and datasets show improved robustness to neighborhood-size selection and consistent performance gains.
>
---
#### [new 015] CosyAccent: Duration-Controllable Accent Normalization Using Source-Synthesis Training Data
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音处理任务，解决accent normalization中的自然度与持续时间控制问题。提出源合成数据方法和非自回归模型CosyAccent，提升内容保留与自然度。**

- **链接: [https://arxiv.org/pdf/2602.19166v1](https://arxiv.org/pdf/2602.19166v1)**

> **作者:** Qibing Bai; Shuhao Shi; Shuai Wang; Yukai Ju; Yannan Wang; Haizhou Li
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Accent normalization (AN) systems often struggle with unnatural outputs and undesired content distortion, stemming from both suboptimal training data and rigid duration modeling. In this paper, we propose a "source-synthesis" methodology for training data construction. By generating source L2 speech and using authentic native speech as the training target, our approach avoids learning from TTS artifacts and, crucially, requires no real L2 data in training. Alongside this data strategy, we introduce CosyAccent, a non-autoregressive model that resolves the trade-off between prosodic naturalness and duration control. CosyAccent implicitly models rhythm for flexibility yet offers explicit control over total output duration. Experiments show that, despite being trained without any real L2 speech, CosyAccent achieves significantly improved content preservation and superior naturalness compared to strong baselines trained on real-world data.
>
---
#### [new 016] Multi-Channel Speech Enhancement for Cocktail Party Speech Emotion Recognition
- **分类: cs.SD**

- **简介: 该论文属于语音情感识别任务，解决混响环境下目标说话人语音提取问题。通过多通道语音增强技术提升情感识别效果。**

- **链接: [https://arxiv.org/pdf/2602.18802v1](https://arxiv.org/pdf/2602.18802v1)**

> **作者:** Youjun Chen; Guinan Li; Mengzhe Geng; Xurong Xie; Shujie Hu; Huimeng Wang; Haoning Xu; Chengxi Deng; Jiajun Deng; Zhaoqing Li; Mingyu Cui; Xunying Liu
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** This paper highlights the critical importance of multi-channel speech enhancement (MCSE) for speech emotion recognition (ER) in cocktail party scenarios. A multi-channel speech dereverberation and separation front-end integrating DNN-WPE and mask-based MVDR is used to extract the target speaker's speech from the mixture speech, before being fed into the downstream ER back-end using HuBERT- and ViT-based speech and visual features. Experiments on mixture speech constructed using the IEMOCAP and MSP-FACE datasets suggest the MCSE output consistently outperforms domain fine-tuned single-channel speech representations produced by: a) Conformer-based metric GANs; and b) WavLM SSL features with optional SE-ER dual task fine-tuning. Statistically significant increases in weighted, unweighted accuracy and F1 measures by up to 9.5%, 8.5% and 9.1% absolute (17.1%, 14.7% and 16.0% relative) are obtained over the above single-channel baselines. The generalization of IEMOCAP trained MCSE front-ends are also shown when being zero-shot applied to out-of-domain MSP-FACE data.
>
---
#### [new 017] CTC-TTS: LLM-based dual-streaming text-to-speech with CTC alignment
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音合成任务，旨在解决低延迟双流合成中的文本-语音对齐问题。提出CTC-TTS，采用CTC对齐和双词交错策略，提升合成质量与效率。**

- **链接: [https://arxiv.org/pdf/2602.19574v1](https://arxiv.org/pdf/2602.19574v1)**

> **作者:** Hanwen Liu; Saierdaer Yusuyin; Hao Huang; Zhijian Ou
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Large-language-model (LLM)-based text-to-speech (TTS) systems can generate natural speech, but most are not designed for low-latency dual-streaming synthesis. High-quality dual-streaming TTS depends on accurate text--speech alignment and well-designed training sequences that balance synthesis quality and latency. Prior work often relies on GMM-HMM based forced-alignment toolkits (e.g., MFA), which are pipeline-heavy and less flexible than neural aligners; fixed-ratio interleaving of text and speech tokens struggles to capture text--speech alignment regularities. We propose CTC-TTS, which replaces MFA with a CTC based aligner and introduces a bi-word based interleaving strategy. Two variants are designed: CTC-TTS-L (token concatenation along the sequence length) for higher quality and CTC-TTS-F (embedding stacking along the feature dimension) for lower latency. Experiments show that CTC-TTS outperforms fixed-ratio interleaving and MFA-based baselines on streaming synthesis and zero-shot tasks. Speech samples are available at https://ctctts.github.io/.
>
---
#### [new 018] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition
- **分类: cs.CV; cs.SD**

- **简介: 该论文针对统一语音识别任务，解决半监督训练中伪标签效率低和错误传播的问题，提出CTC驱动的教师强制方法，提升模型效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.19316v1](https://arxiv.org/pdf/2602.19316v1)**

> **作者:** Alexandros Haliassos; Rodrigo Mira; Stavros Petridis
>
> **备注:** ICLR 2026. Code: https://github.com/ahaliassos/usr2
>
> **摘要:** Unified Speech Recognition (USR) has emerged as a semi-supervised framework for training a single model for audio, visual, and audiovisual speech recognition, achieving state-of-the-art results on in-distribution benchmarks. However, its reliance on autoregressive pseudo-labelling makes training expensive, while its decoupled supervision of CTC and attention branches increases susceptibility to self-reinforcing errors, particularly under distribution shifts involving longer sequences, noise, or unseen domains. We propose CTC-driven teacher forcing, where greedily decoded CTC pseudo-labels are fed into the decoder to generate attention targets in a single forward pass. Although these can be globally incoherent, in the pseudo-labelling setting they enable efficient and effective knowledge transfer. Because CTC and CTC-driven attention pseudo-labels have the same length, the decoder can predict both simultaneously, benefiting from the robustness of CTC and the expressiveness of attention without costly beam search. We further propose mixed sampling to mitigate the exposure bias of the decoder relying solely on CTC inputs. The resulting method, USR 2.0, halves training time, improves robustness to out-of-distribution inputs, and achieves state-of-the-art results on LRS3, LRS2, and WildVSR, surpassing USR and modality-specific self-supervised baselines.
>
---
#### [new 019] Audio-Visual Continual Test-Time Adaptation without Forgetting
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于跨模态持续测试时适应任务，解决模型在非平稳领域中因分布变化导致的性能下降问题。提出AV-CTTA方法，通过动态检索融合层参数提升模型适应能力并减少遗忘。**

- **链接: [https://arxiv.org/pdf/2602.18528v1](https://arxiv.org/pdf/2602.18528v1)**

> **作者:** Sarthak Kumar Maharana; Akshay Mehra; Bhavya Ramakrishna; Yunhui Guo; Guan-Ming Su
>
> **摘要:** Audio-visual continual test-time adaptation involves continually adapting a source audio-visual model at test-time, to unlabeled non-stationary domains, where either or both modalities can be distributionally shifted, which hampers online cross-modal learning and eventually leads to poor accuracy. While previous works have tackled this problem, we find that SOTA methods suffer from catastrophic forgetting, where the model's performance drops well below the source model due to continual parameter updates at test-time. In this work, we first show that adapting only the modality fusion layer to a target domain not only improves performance on that domain but can also enhance performance on subsequent domains. Based on this strong cross-task transferability of the fusion layer's parameters, we propose a method, $\texttt{AV-CTTA}$, that improves test-time performance of the models without access to any source data. Our approach works by using a selective parameter retrieval mechanism that dynamically retrieves the best fusion layer parameters from a buffer using only a small batch of test data. These parameters are then integrated into the model, adapted to the current test distribution, and saved back for future use. Extensive experiments on benchmark datasets involving unimodal and bimodal corruptions show our proposed $\texttt{AV-CTTA}$ significantly outperforms existing methods while minimizing catastrophic forgetting.
>
---
#### [new 020] ReHear: Iterative Pseudo-Label Refinement for Semi-Supervised Speech Recognition via Audio Large Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决半监督学习中的伪标签偏差和错误累积问题。通过引入音频感知的大语言模型，迭代优化伪标签，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2602.18721v1](https://arxiv.org/pdf/2602.18721v1)**

> **作者:** Zefang Liu; Chenyang Zhu; Sangwoo Cho; Shi-Xiong Zhang
>
> **摘要:** Semi-supervised learning in automatic speech recognition (ASR) typically relies on pseudo-labeling, which often suffers from confirmation bias and error accumulation due to noisy supervision. To address this limitation, we propose ReHear, a framework for iterative pseudo-label refinement that integrates an instruction-tuned, audio-aware large language model (LLM) into the self-training loop. Unlike conventional text-based correctors, our approach conditions the LLM on both the ASR hypothesis and the source audio, allowing it to recover phonetically accurate transcripts even from severe recognition errors. These refined pseudo-labels serve as high-fidelity targets for fine-tuning the ASR model in an iterative cycle. Experimental results across diverse benchmarks demonstrate that ReHear effectively mitigates error propagation, consistently outperforming both supervised and pseudo-labeling baselines.
>
---
#### [new 021] JavisDiT++: Unified Modeling and Optimization for Joint Audio-Video Generation
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文聚焦于联合音视频生成任务，解决生成质量、时间同步和人类偏好对齐问题，提出JavisDiT++框架，包含多专家模型、时间对齐策略和直接偏好优化方法。**

- **链接: [https://arxiv.org/pdf/2602.19163v1](https://arxiv.org/pdf/2602.19163v1)**

> **作者:** Kai Liu; Yanhao Zheng; Kai Wang; Shengqiong Wu; Rongjunchen Zhang; Jiebo Luo; Dimitrios Hatzinakos; Ziwei Liu; Hao Fei; Tat-Seng Chua
>
> **备注:** Accepted by ICLR 2026. Homepage: https://JavisVerse.github.io/JavisDiT2-page
>
> **摘要:** AIGC has rapidly expanded from text-to-image generation toward high-quality multimodal synthesis across video and audio. Within this context, joint audio-video generation (JAVG) has emerged as a fundamental task that produces synchronized and semantically aligned sound and vision from textual descriptions. However, compared with advanced commercial models such as Veo3, existing open-source methods still suffer from limitations in generation quality, temporal synchrony, and alignment with human preferences. To bridge the gap, this paper presents JavisDiT++, a concise yet powerful framework for unified modeling and optimization of JAVG. First, we introduce a modality-specific mixture-of-experts (MS-MoE) design that enables cross-modal interaction efficacy while enhancing single-modal generation quality. Then, we propose a temporal-aligned RoPE (TA-RoPE) strategy to achieve explicit, frame-level synchronization between audio and video tokens. Besides, we develop an audio-video direct preference optimization (AV-DPO) method to align model outputs with human preference across quality, consistency, and synchrony dimensions. Built upon Wan2.1-1.3B-T2V, our model achieves state-of-the-art performance merely with around 1M public training entries, significantly outperforming prior approaches in both qualitative and quantitative evaluations. Comprehensive ablation studies have been conducted to validate the effectiveness of our proposed modules. All the code, model, and dataset are released at https://JavisVerse.github.io/JavisDiT2-page.
>
---
#### [new 022] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出JAEGER框架，解决2D感知与3D空间推理不匹配的问题，通过RGB-D和多通道音频实现联合3D视听定位与推理。**

- **链接: [https://arxiv.org/pdf/2602.18527v1](https://arxiv.org/pdf/2602.18527v1)**

> **作者:** Zhan Liu; Changli Tang; Yuxin Wang; Zhiyuan Zhu; Youjun Chen; Yiwen Shao; Tianzi Wang; Lei Ke; Zengrui Jin; Chao Zhang
>
> **摘要:** Current audio-visual large language models (AV-LLMs) are predominantly restricted to 2D perception, relying on RGB video and monaural audio. This design choice introduces a fundamental dimensionality mismatch that precludes reliable source localization and spatial reasoning in complex 3D environments. We address this limitation by presenting JAEGER, a framework that extends AV-LLMs to 3D space, to enable joint spatial grounding and reasoning through the integration of RGB-D observations and multi-channel first-order ambisonics. A core contribution of our work is the neural intensity vector (Neural IV), a learned spatial audio representation that encodes robust directional cues to enhance direction-of-arrival estimation, even in adverse acoustic scenarios with overlapping sources. To facilitate large-scale training and systematic evaluation, we propose SpatialSceneQA, a benchmark of 61k instruction-tuning samples curated from simulated physical environments. Extensive experiments demonstrate that our approach consistently surpasses 2D-centric baselines across diverse spatial perception and reasoning tasks, underscoring the necessity of explicit 3D modelling for advancing AI in physical environments. Our source code, pre-trained model checkpoints and datasets will be released upon acceptance.
>
---
#### [new 023] An LLM-Enabled Frequency-Aware Flow Diffusion Model for Natural-Language-Guided Power System Scenario Generation
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于电力系统场景生成任务，旨在解决传统方法依赖固定条件向量、灵活性不足的问题。提出LFFD框架，利用大语言模型和流扩散模型实现自然语言引导的高质量场景生成。**

- **链接: [https://arxiv.org/pdf/2602.19522v1](https://arxiv.org/pdf/2602.19522v1)**

> **作者:** Zhenghao Zhou; Yiyan Li; Fei Xie; Lu Wang; Bo Wang; Jiansheng Wang; Zheng Yan; Mo-Yuen Chow
>
> **摘要:** Diverse and controllable scenario generation (e.g., wind, solar, load, etc.) is critical for robust power system planning and operation. As AI-based scenario generation methods are becoming the mainstream, existing methods (e.g., Conditional Generative Adversarial Nets) mainly rely on a fixed-length numerical conditioning vector to control the generation results, facing challenges in user conveniency and generation flexibility. In this paper, a natural-language-guided scenario generation framework, named LLM-enabled Frequency-aware Flow Diffusion (LFFD), is proposed to enable users to generate desired scenarios using plain human language. First, a pretrained LLM module is introduced to convert generation requests described by unstructured natural languages into ordered semantic space. Second, instead of using standard diffusion models, a flow diffusion model employing a rectified flow matching objective is introduced to achieve efficient and high-quality scenario generation, taking the LLM output as the model input. During the model training process, a frequency-aware multi-objective optimization algorithm is introduced to mitigate the frequency-bias issue. Meanwhile, a dual-agent framework is designed to create text-scenario training sample pairs as well as to standardize semantic evaluation. Experiments based on large-scale photovoltaic and load datasets demonstrate the effectiveness of the proposed method.
>
---
## 更新

#### [replaced 001] The Universal Personalizer: Few-Shot Dysarthric Speech Recognition via Meta-Learning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，解决发音障碍者ASR个性化问题。通过元学习实现少样本快速个性化，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2509.15516v2](https://arxiv.org/pdf/2509.15516v2)**

> **作者:** Dhruuv Agarwal; Harry Zhang; Yang Yu; Quan Wang
>
> **摘要:** Personalizing dysarthric ASR is hindered by demanding enrollment collection and per-user training. We propose a hybrid meta-training method for a single model, enabling zero-shot and few-shot on-the-fly personalization via in-context learning (ICL). On Euphonia, it achieves 13.9% Word Error Rate (WER), surpassing speaker-independent baselines (17.5%). On SAP Test-1, our 5.3% WER outperforms the challenge-winning team (5.97%). On Test-2, our 9.49% trails only the winner (8.11%) but without relying on techniques like offline model-merging or custom audio chunking. Curation yields a 40% WER reduction using random same-speaker examples, validating active personalization. While static text curation fails to beat this baseline, oracle similarity reveals substantial headroom, highlighting dynamic acoustic retrieval as the next frontier. Data ablations confirm rapid low-resource speaker adaptation, establishing the model as a practical personalized solution.
>
---
#### [replaced 002] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音乐驱动的3D舞蹈生成任务，旨在解决传统方法对舞蹈风格条件利用不足的问题。提出MEGADance架构，通过分离舞蹈通用性与风格特异性，提升舞蹈质量和风格控制能力。**

- **链接: [https://arxiv.org/pdf/2505.17543v3](https://arxiv.org/pdf/2505.17543v3)**

> **作者:** Kaixing Yang; Xulong Tang; Ziqiao Peng; Yuxuan Hu; Jun He; Hongyan Liu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Music-driven 3D dance generation has attracted increasing attention in recent years, with promising applications in choreography, virtual reality, and creative content creation. Previous research has generated promising realistic dance movement from audio signals. However, traditional methods underutilize genre conditioning, often treating it as auxiliary modifiers rather than core semantic drivers. This oversight compromises music-motion synchronization and disrupts dance genre continuity, particularly during complex rhythmic transitions, thereby leading to visually unsatisfactory effects. To address the challenge, we propose MEGADance, a novel architecture for music-driven 3D dance generation. By decoupling choreographic consistency into dance generality and genre specificity, MEGADance demonstrates significant dance quality and strong genre controllability. It consists of two stages: (1) High-Fidelity Dance Quantization Stage (HFDQ), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) and reconstructs them with kinematic-dynamic constraints, and (2) Genre-Aware Dance Generation Stage (GADG), which maps music into the latent representation by synergistic utilization of Mixture-of-Experts (MoE) mechanism with Mamba-Transformer hybrid backbone. Extensive experiments on the FineDance and AIST++ dataset demonstrate the state-of-the-art performance of MEGADance both qualitatively and quantitatively. Code is available at https://github.com/XulongT/MEGADance.
>
---
#### [replaced 003] WAVE: Learning Unified & Versatile Audio-Visual Embeddings with Multimodal LLM
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出WAVE，一种统一的多模态嵌入模型，解决音频、视频与文本间的跨模态检索和问答问题。通过联合训练和层次特征融合，实现通用且灵活的多模态表示。**

- **链接: [https://arxiv.org/pdf/2509.21990v2](https://arxiv.org/pdf/2509.21990v2)**

> **作者:** Changli Tang; Qinfan Xiao; Ke Mei; Tianyi Wang; Fengyun Rao; Chao Zhang
>
> **摘要:** While embeddings from multimodal large language models (LLMs) excel as general-purpose representations, their application to dynamic modalities like audio and video remains underexplored. We introduce WAVE (\textbf{u}nified \& \textbf{v}ersatile \textbf{a}udio-\textbf{v}isual \textbf{e}mbeddings), the first LLM-based embedding that creates a unified representation space for text, audio, and video modalities. WAVE employs a novel hierarchical feature fusion strategy and a joint multi-modal, multi-task training approach to enable two key capabilities: any-to-any cross-modal retrieval and the generation of prompt-aware embeddings tailored to user instructions. Experimentally, WAVE sets a new state-of-the-art on the MMEB-v2 video benchmark and achieves superior results in audio and video-to-audio retrieval. Its prompt-aware nature also yields remarkable performance in multimodal question answering, significantly outperforming existing embedding models. Ablation studies validate our joint training strategy, demonstrating improved performance across all modalities. With a newly introduced benchmark for versatile audio-visual learning, WAVE opens up broad possibilities for cross-modal, any-to-any applications. Our code and checkpoints are released at \href{https://github.com/TCL606/WAVE}{https://github.com/TCL606/WAVE}.
>
---
#### [replaced 004] Bagpiper: Solving Open-Ended Audio Tasks via Rich Captions
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出Bagpiper，一个用于音频理解与生成的8B模型，通过丰富描述实现音频与概念的双向映射，解决传统模型任务单一、缺乏通用性的问题。**

- **链接: [https://arxiv.org/pdf/2602.05220v2](https://arxiv.org/pdf/2602.05220v2)**

> **作者:** Jinchuan Tian; Haoran Wang; Bo-Hao Su; Chien-yu Huang; Qingzheng Wang; Jiatong Shi; William Chen; Xun Gong; Siddhant Arora; Chin-Jou Li; Masao Someki; Takashi Maekaku; Keita Goto; Yusuke Shinohara; Jin Sakuma; Chao-Han Huck Yang; Shinji Watanabe
>
> **摘要:** Current audio foundation models typically rely on rigid, task-specific supervision, addressing isolated factors of audio rather than the whole. In contrast, human intelligence processes audio holistically, seamlessly bridging physical signals with abstract cognitive concepts to execute complex tasks. Grounded in this philosophy, we introduce Bagpiper, an 8B audio foundation model that interprets physical audio via rich captions, i.e., comprehensive natural language descriptions that encapsulate the critical cognitive concepts inherent in the signal (e.g., transcription, audio events). By pre-training on a massive corpus of 600B tokens, the model establishes a robust bidirectional mapping between raw audio and this high-level conceptual space. During fine-tuning, Bagpiper adopts a caption-then-process workflow, simulating an intermediate cognitive reasoning step to solve diverse tasks without task-specific priors. Experimentally, Bagpiper outperforms Qwen-2.5-Omni on MMAU and AIRBench for audio understanding and surpasses CosyVoice3 and TangoFlux in generation quality, capable of synthesizing arbitrary compositions of speech, music, and sound effects. To the best of our knowledge, Bagpiper is among the first works that achieve unified understanding generation for general audio. Model, data, and code are available at Bagpiper Home Page.
>
---
#### [replaced 005] PhoenixCodec: Taming Neural Speech Coding for Extreme Low-Resource Scenarios
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出PhoenixCodec，解决极端低资源下的语音编码问题。通过优化架构和训练策略，提升效率与质量，满足低计算、低延迟要求。**

- **链接: [https://arxiv.org/pdf/2510.21196v2](https://arxiv.org/pdf/2510.21196v2)**

> **作者:** Zixiang Wan; Haoran Zhao; Guochang Zhang; Runqiang Han; Jianqiang Wei; Yuexian Zou
>
> **备注:** Accepted by ICASSP 2026; 5 pages, 1 figure, 4 tables
>
> **摘要:** This paper presents PhoenixCodec, a comprehensive neural speech coding and decoding framework designed for extremely low-resource conditions. The proposed system integrates an optimized asymmetric frequency-time architecture, a Cyclical Calibration and Refinement (CCR) training strategy, and a noise-invariant fine-tuning procedure. Under stringent constraints - computation below 700 MFLOPs, latency less than 30 ms, and dual-rate support at 1 kbps and 6 kbps - existing methods face a trade-off between efficiency and quality. PhoenixCodec addresses these challenges by alleviating the resource scattering of conventional decoders, employing CCR to enhance optimization stability, and enhancing robustness through noisy-sample fine-tuning. In the LRAC 2025 Challenge Track 1, the proposed system ranked third overall and demonstrated the best performance at 1 kbps in both real-world noise and reverberation and intelligibility in clean tests, confirming its effectiveness.
>
---
#### [replaced 006] A Dual-Branch Parallel Network for Speech Enhancement and Restoration
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音增强与修复任务，旨在解决噪声、混响和带宽限制等问题。提出DBP-Net模型，通过双分支结构实现同时抑制干扰和重建频谱。**

- **链接: [https://arxiv.org/pdf/2409.08702v2](https://arxiv.org/pdf/2409.08702v2)**

> **作者:** Da-Hee Yang; Dail Kim; Joon-Hyuk Chang; Jeonghwan Choi; Han-gil Moon
>
> **备注:** Accepted for publication in Computer Speech & Language (2026). Final published version available at: https://doi.org/10.1016/j.csl.2026.101959
>
> **摘要:** We present a novel general speech restoration model, DBP-Net (dual-branch parallel network), designed to effectively handle complex real-world distortions including noise, reverberation, and bandwidth degradation. Unlike prior approaches that rely on a single processing path or separate models for enhancement and restoration, DBP-Net introduces a unified architecture with dual parallel branches-a masking-based branch for distortion suppression and a mapping-based branch for spectrum reconstruction. A key innovation behind DBP-Net lies in the parameter sharing between the two branches and a cross-branch skip fusion, where the output of the masking branch is explicitly fused into the mapping branch. This design enables DBP-Net to simultaneously leverage complementary learning strategies-suppression and generation-within a lightweight framework. Experimental results show that DBP-Net significantly outperforms existing baselines in comprehensive speech restoration tasks while maintaining a compact model size. These findings suggest that DBP-Net offers an effective and scalable solution for unified speech enhancement and restoration in diverse distortion scenarios.
>
---
#### [replaced 007] A Survey on Cross-Modal Interaction Between Music and Multimodal Data
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 本文综述音乐与多模态数据的跨模态交互，探讨其任务类型、数据表示及挑战，旨在推动计算音乐研究的发展。**

- **链接: [https://arxiv.org/pdf/2504.12796v2](https://arxiv.org/pdf/2504.12796v2)**

> **作者:** Sifei Li; Mining Tan; Feier Shen; Minyan Luo; Zijiao Yin; Fan Tang; Weiming Dong; Changsheng Xu
>
> **备注:** 34 pages, 7 figures
>
> **摘要:** Multimodal learning has driven innovation across various industries, particularly in the field of music. By enabling more intuitive interaction experiences and enhancing immersion, it not only lowers the entry barriers to the music but also increases its overall appeal. This survey aims to provide a comprehensive review of multimodal tasks related to music, outlining how music contributes to multimodal learning and offering insights for researchers seeking to expand the boundaries of computational music. Unlike text and images, which are often semantically or visually intuitive, music primarily interacts with humans through auditory perception, making its data representation inherently less intuitive. Therefore, this paper first introduces the representations of music and provides an overview of music datasets. Subsequently, we categorize cross-modal interactions between music and multimodal data into three types: music-driven cross-modal interactions, music-oriented cross-modal interactions, and bidirectional music cross-modal interactions. For each category, we systematically trace the development of relevant sub-tasks, analyze existing limitations, and discuss emerging trends. Furthermore, we provide a comprehensive summary of datasets and evaluation metrics used in multimodal tasks related to music, offering benchmark references for future research. Finally, we discuss the current challenges in cross-modal interactions involving music and propose potential directions for future research.
>
---
#### [replaced 008] Mathematical Foundations of Polyphonic Music Generation via Structural Inductive Bias
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，解决polyphonic音乐生成中的“缺失中间”问题，通过结构归纳偏置和数学理论验证，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.03612v3](https://arxiv.org/pdf/2601.03612v3)**

> **作者:** Joonwon Seo
>
> **备注:** 81 pages. A comprehensive monograph detailing the Smart Embedding architecture for polyphonic music generation, including theoretical proofs (Information Theory, Rademacher Complexity, RPTP) and human evaluation results
>
> **摘要:** This monograph introduces a novel approach to polyphonic music generation by addressing the "Missing Middle" problem through structural inductive bias. Focusing on Beethoven's piano sonatas as a case study, we empirically verify the independence of pitch and hand attributes using normalized mutual information (NMI=0.167) and propose the Smart Embedding architecture, achieving a 48.30% reduction in parameters. We provide rigorous mathematical proofs using information theory (negligible loss bounded at 0.153 bits), Rademacher complexity (28.09% tighter generalization bound), and category theory to demonstrate improved stability and generalization. Empirical results show a 9.47% reduction in validation loss, confirmed by SVD analysis and an expert listening study (N=53). This dual theoretical and applied framework bridges gaps in AI music generation, offering verifiable insights for mathematically grounded deep learning.
>
---
#### [replaced 009] AeroGPT: Leveraging Large-Scale Audio Model for Aero-Engine Bearing Fault Diagnosis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于航空发动机轴承故障诊断任务，旨在解决传统方法依赖后处理及缺乏可解释性的问题。提出AeroGPT框架，结合大模型与振动信号对齐，实现高效准确的故障分类。**

- **链接: [https://arxiv.org/pdf/2506.16225v2](https://arxiv.org/pdf/2506.16225v2)**

> **作者:** Jiale Liu; Dandan Peng; Huan Wang; Chenyu Liu; Yan-Fu Li; Min Xie
>
> **摘要:** Aerospace engines, as critical components in aviation and aerospace industries, require continuous and accurate fault diagnosis to ensure operational safety and prevent catastrophic failures. While deep learning techniques have been extensively studied in this context, they typically output logits or confidence scores, necessitating post-processing to obtain actionable insights. Furthermore, the potential of large-scale audio models for this task remains largely untapped. To address these limitations, this paper proposes AeroGPT, a novel framework that transfers knowledge from the general audio domain to aero-engine bearing fault diagnosis. AeroGPT leverages a large-scale audio model and incorporates Vibration Signal Alignment (VSA) to adapt general audio knowledge to domain-specific vibration patterns, along with Generative Fault Classification (GFC) to directly generate interpretable fault labels. This approach eliminates the need for label post-processing and supports interactive, interpretable, and actionable fault diagnosis, thereby enhancing industrial applicability. Through comprehensive experimental validation on two aero-engine bearing datasets, AeroGPT achieves 98.94% accuracy on the DIRG dataset and 100% accuracy on the HIT bearing dataset, outperforming representative deep learning approaches. Qualitative analysis and further discussion also demonstrate its potential for interactive diagnosis and real-world deployment, highlighting the promise of large-scale audio models to advance fault diagnosis in aerospace applications.
>
---
#### [replaced 010] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音基础模型的测试时适应任务，解决真实场景中声学域变化导致的性能下降问题。提出E-BATS框架，在无需反向传播的情况下实现高效准确的适应。**

- **链接: [https://arxiv.org/pdf/2506.07078v3](https://arxiv.org/pdf/2506.07078v3)**

> **作者:** Jiaheng Dong; Hong Jia; Soumyajit Chatterjee; Abhirup Ghosh; James Bailey; Ting Dang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4.1%-13.5% accuracy gains over backpropagation-free baselines and 2.0-6.4 times GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments.
>
---
#### [replaced 011] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization
- **分类: cs.CV; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出JavisDiT，解决音频视频同步生成任务。通过引入时空对齐机制和新基准，提升生成内容的质量与同步精度。**

- **链接: [https://arxiv.org/pdf/2503.23377v2](https://arxiv.org/pdf/2503.23377v2)**

> **作者:** Kai Liu; Wei Li; Lai Chen; Shengqiong Wu; Yanhao Zheng; Jiayi Ji; Fan Zhou; Jiebo Luo; Ziwei Liu; Hao Fei; Tat-Seng Chua
>
> **备注:** Accepted by ICLR 2026. Homepage: https://javisverse.github.io/JavisDiT-page/
>
> **摘要:** This paper introduces JavisDiT, a novel Joint Audio-Video Diffusion Transformer designed for synchronized audio-video generation (JAVG). Based on the powerful Diffusion Transformer (DiT) architecture, JavisDiT simultaneously generates high-quality audio and video content from open-ended user prompts in a unified framework. To ensure audio-video synchronization, we introduce a fine-grained spatio-temporal alignment mechanism through a Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator. This module extracts both global and fine-grained spatio-temporal priors, guiding the synchronization between the visual and auditory components. Furthermore, we propose a new benchmark, JavisBench, which consists of 10,140 high-quality text-captioned sounding videos and focuses on synchronization evaluation in diverse and complex real-world scenarios. Further, we specifically devise a robust metric for measuring the synchrony between generated audio-video pairs in real-world content. Experimental results demonstrate that JavisDiT significantly outperforms existing methods by ensuring both high-quality generation and precise synchronization, setting a new standard for JAVG tasks. Our code, model, and data are available at https://javisverse.github.io/JavisDiT-page/.
>
---
#### [replaced 012] Closing the Gap Between Text and Speech Understanding in LLMs
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于自然语言处理任务，解决LLM在语音理解上的性能差距问题。通过SALAD方法，提升语音与文本对齐效果，减少对合成数据的依赖。**

- **链接: [https://arxiv.org/pdf/2510.13632v2](https://arxiv.org/pdf/2510.13632v2)**

> **作者:** Santiago Cuervo; Skyler Seto; Maureen de Seyssel; Richard He Bai; Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly; Zakaria Aldeneh
>
> **摘要:** Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
>
---
#### [replaced 013] Binaural Target Speaker Extraction using HRTFs
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，解决多说话人场景下的目标说话人提取问题。通过利用HRTF和复数神经网络，有效分离目标语音并保留双耳线索。**

- **链接: [https://arxiv.org/pdf/2507.19369v4](https://arxiv.org/pdf/2507.19369v4)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **摘要:** In this work, we address the problem of binaural target-speaker extraction in the presence of multiple simultane-ous talkers. We propose a novel approach that leverages the individual listener's Head-Related Transfer Function (HRTF) to isolate the target speaker. The proposed method is speaker-independent, as it does not rely on speaker embeddings. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier transform (STFT) of the mixed audio signals, and compare it to a Real-Imaginary (RI)-based neural network, demonstrating the advantages of the former. We first evaluate the method in an anechoic, noise-free scenario, achieving excellent extraction performance while preserving the binaural cues of the target signal. We then extend the evaluation to reverberant conditions. Our method proves robust, maintaining speech clarity and source directionality while simultaneously reducing reverberation. A comparative analysis with existing binaural Target Speaker Extraction (TSE) methods shows that the proposed approach achieves performance comparable to state-of-the-art techniques in terms of noise reduction and perceptual quality, while providing a clear advantage in preserving binaural cues. Demo-page: https://bi-ctse-hrtf.github.io
>
---
#### [replaced 014] S-PRESSO: Ultra Low Bitrate Sound Effect Compression With Diffusion Autoencoders And Offline Quantization
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文属于音频压缩任务，旨在解决低比特率下音频质量下降的问题。提出S-PRESSO模型，通过扩散自编码器和离线量化实现超低比特率压缩，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2602.15082v2](https://arxiv.org/pdf/2602.15082v2)**

> **作者:** Zineb Lahrichi; Gaëtan Hadjeres; Gaël Richard; Geoffroy Peeters
>
> **摘要:** Neural audio compression models have recently achieved extreme compression rates, enabling efficient latent generative modeling. Conversely, latent generative models have been applied to compression, pushing the limits of continuous and discrete approaches. However, existing methods remain constrained to low-resolution audio and degrade substantially at very low bitrates, where audible artifacts are prominent. In this paper, we present S-PRESSO, a 48kHz sound effect compression model that produces both continuous and discrete embeddings at ultra-low bitrates, down to 0.096 kbps, via offline quantization. Our model relies on a pretrained latent diffusion model to decode compressed audio embeddings learned by a latent encoder. Leveraging the generative priors of the diffusion decoder, we achieve extremely low frame rates, down to 1Hz (750x compression rate), producing convincing and realistic reconstructions at the cost of exact fidelity. Despite operating at high compression rates, we demonstrate that S-PRESSO outperforms both continuous and discrete baselines in audio quality, acoustic similarity and reconstruction metrics.
>
---
