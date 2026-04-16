# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Few-Shot and Pseudo-Label Guided Speech Quality Evaluation with Large Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量评估任务，解决有限标注数据下的非侵入式评估问题。提出GatherMOS框架，结合大语言模型与伪标签，提升评估性能。**

- **链接: [https://arxiv.org/pdf/2604.13528](https://arxiv.org/pdf/2604.13528)**

> **作者:** Ryandhimas E. Zezario; Dyah A. M. G. Wisnu; Szu-Wei Fu; Sabato Marco Siniscalchi; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** In this paper, we introduce GatherMOS, a novel framework that leverages large language models (LLM) as meta-evaluators to aggregate diverse signals into quality predictions. GatherMOS integrates lightweight acoustic descriptors with pseudo-labels from DNSMOS and VQScore, enabling the LLM to reason over heterogeneous inputs and infer perceptual mean opinion scores (MOS). We further explore both zero-shot and few-shot in-context learning setups, showing that zero-shot GatherMOS maintains stable performance across diverse conditions, while few-shot guidance yields large gains when support samples match the test conditions. Experiments on the VoiceBank-DEMAND dataset demonstrate that GatherMOS consistently outperforms DNSMOS, VQScore, naive score averaging, and even learning-based models such as CNN-BLSTM and MOS-SSL when trained under limited labeled-data conditions. These results highlight the potential of LLM-based aggregation as a practical strategy for non-intrusive speech quality evaluation.
>
---
#### [new 002] SpeakerRPL v2: Robust Open-set Speaker Identification through Enhanced Few-shot Foundation Tuning and Model Fusion
- **分类: eess.AS**

- **简介: 该论文属于说话人识别任务，解决开放集识别中的鲁棒性问题。通过增强少样本微调和模型融合，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2604.13605](https://arxiv.org/pdf/2604.13605)**

> **作者:** Zhiyong Chen; Shuhang Wu; Yingjie Duan; Xinkang Xu; Xinhui Hu
>
> **备注:** ICASSP 2026. Code Available:this https URL
>
> **摘要:** This paper proposes an improved approach for open-set speaker identification based on pretrained speaker foundation models. Building upon the previous Speaker Reciprocal Points Learning framework (V1), we first introduce an enhanced open-set learning objective by integrating reciprocal points learning with logit normalization (LogitNorm) and incorporating adaptive anchor learning to better constrain target speaker representations and improve robustness. Second, we propose a model fusion strategy to stabilize and enhance the few-shot tuning process, effectively reducing result randomness and improving generalization. Furthermore, we introduce a model selection method to ensure optimal performance in model fusion. Experimental evaluations on the VoxCeleb, ESD and 3D-Speaker datasets demonstrate the effectiveness and robustness of the proposed method under diverse conditions. On a newly proposed Vox1-O-like test set, our method reduces the EER from 1.28% to 0.09%, achieving a relative reduction of approximately 93%.
>
---
#### [new 003] Towards Fine-grained Temporal Perception: Post-Training Large Audio-Language Models with Audio-Side Time Prompt
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频理解任务，旨在解决大音频语言模型在时间感知上的不足。通过引入音频侧时间提示和强化学习，提升模型在细粒度时间任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.13715](https://arxiv.org/pdf/2604.13715)**

> **作者:** Yanfeng Shi; Pengfei Cai; Jun Liu; Qing Gu; Nan Jiang; Lirong Dai; Ian McLoughlin; Yan Song
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Large Audio-Language Models (LALMs) enable general audio understanding and demonstrate remarkable performance across various audio tasks. However, these models still face challenges in temporal perception (e.g., inferring event onset and offset), leading to limited utility in fine-grained scenarios. To address this issue, we propose Audio-Side Time Prompt and leverage Reinforcement Learning (RL) to develop the TimePro-RL framework for fine-grained temporal perception. Specifically, we encode timestamps as embeddings and interleave them within the audio feature sequence as temporal coordinates to prompt the model. Furthermore, we introduce RL following Supervised Fine-Tuning (SFT) to directly optimize temporal alignment performance. Experiments demonstrate that TimePro-RL achieves significant performance gains across a range of audio temporal tasks, such as audio grounding, sound event detection, and dense audio captioning, validating its robust effectiveness.
>
---
#### [new 004] Melodic contour does not cluster: Reconsidering contour typology
- **分类: cs.SD**

- **简介: 该论文属于音乐分析任务，旨在检验旋律轮廓是否可划分为离散类型。通过数据分析发现旋律轮廓是连续的，而非离散聚类，质疑传统类型学的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.13119](https://arxiv.org/pdf/2604.13119)**

> **作者:** Bas Cornelissen; Willem Zuidema; John Ashley Burgoyne; Henkjan Honing
>
> **备注:** 16 pages, 8 figures, plus 5 pages of supplements
>
> **摘要:** How to describe the shape of a melodic phrase? Scholars have often relied on typologies with a small set of contour types. We question their adequacy: we find no evidence that phrase contours cluster into discrete types, neither in German or Chinese folksongs, nor in Gregorian chant. The test for clustering we propose applies the dist-dip test of multimodality after a UMAP dimensionality reduction. The test correctly identifies clustering in a synthetic dataset, but not in actual phrase contours. These results raise problems for discrete typologies. In particular, type frequencies may be unreliable, as we see with Huron's typology. We also show how a recent finding of four contour shapes may be an artefact of the analysis. Our findings suggest that melodic contour is best seen as a continuous phenomenon.
>
---
#### [new 005] Classical Machine Learning Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset
- **分类: eess.AS**

- **简介: 该论文属于深度伪造音频检测任务，旨在区分真实与合成语音。通过提取声学特征并训练经典机器学习模型，建立可解释的基线方法。**

- **链接: [https://arxiv.org/pdf/2604.13400](https://arxiv.org/pdf/2604.13400)**

> **作者:** Faheem Ahmad; Ajan Ahmed; Masudul Imtiaz
>
> **备注:** Accepted for Oral Presentation at The 35th IEEE Microelectronics Design and Test Symposium
>
> **摘要:** Deep learning has enabled highly realistic synthetic speech, raising concerns about fraud, impersonation, and disinformation. Despite rapid progress in neural detectors, transparent baselines are needed to reveal which acoustic cues reliably separate real from synthetic speech. This paper presents an interpretable classical machine learning baseline for deepfake audio detection using the Fake-or-Real (FoR) dataset. We extract prosodic, voice-quality, and spectral features from two-second clips at 44.1 kHz (high-fidelity) and 16 kHz (telephone-quality) sampling rates. Statistical analysis (ANOVA, correlation heatmaps) identifies features that differ significantly between real and fake speech. We then train multiple classifiers -- Logistic Regression, LDA, QDA, Gaussian Naive Bayes, SVMs, and GMMs -- and evaluate performance using accuracy, ROC-AUC, EER, and DET curves. Pairwise McNemar's tests confirm statistically significant differences between models. The best model, an RBF SVM, achieves ~93% test accuracy and ~7% EER on both sampling rates, while linear models reach ~75% accuracy. Feature analysis reveals that pitch variability and spectral richness (spectral centroid, bandwidth) are key discriminative cues. These results provide a strong, interpretable baseline for future deepfake audio detectors.
>
---
#### [new 006] ProSDD: Learning Prosodic Representations for Speech Deepfake Detection against Expressive and Emotional Attacks
- **分类: eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有系统在情感和表达攻击下泛化能力差的问题。通过构建两阶段框架ProSDD，提升模型对自然语音变化的感知能力，有效降低错误率。**

- **链接: [https://arxiv.org/pdf/2604.13229](https://arxiv.org/pdf/2604.13229)**

> **作者:** Aurosweta Mahapatra; Ismail Rasim Ulgen; Kong Aik Lee; Nicholas Andrews; Berrak Sisman
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Speech deepfake detection (SDD) systems perform well on standard benchmarks datasets but often fail to generalize to expressive and emotional spoofing attacks. Many methods rely on spoof-heavy training data, learning dataset-specific artifacts rather than transferable cues of natural speech. In contrast, humans internalize variability in real speech and detect fakes as deviations from it. We introduce ProSDD, a two-stage framework that enriches model embeddings through supervised masked prediction of speaker-conditioned prosodic variation based on pitch, voice activity, and energy. Stage I learns prosodic variability from real speech, and Stage II jointly optimizes this objective with spoof classification. ProSDD consistently outperforms baselines under both ASVspoof 2019 and 2024 training, reducing ASVspoof 2024 EER from 25.43% to 16.14% (2019-trained) and from 39.62% to 7.38% (2024-trained), while achieving 50% relative reductions on EmoFake and EmoSpoof-TTS.
>
---
#### [new 007] Comparison of window shapes and lengths in short-time feature extraction for classification of heart sound signals
- **分类: cs.SD; cs.AI**

- **简介: 论文研究心音信号分类任务，解决如何选择最佳窗口形状和长度以提高特征提取效果的问题。通过实验比较三种窗口形状和长度，评估其对biLSTM分类性能的影响。**

- **链接: [https://arxiv.org/pdf/2604.13567](https://arxiv.org/pdf/2604.13567)**

> **作者:** Mahmoud Fakhry; Abeer FathAllah Brery
>
> **摘要:** Heart sound signals, phonocardiography (PCG) signals, allow for the automatic diagnosis of potential cardiovascular pathology. Such classification task can be tackled using the bidirectional long short-term memory (biLSTM) network, trained on features extracted from labeled PCG signals. Regarding the non-stationarity of PCG signals, it is recommended to extract the features from multiple short-length segments of the signals using a sliding window of certain shape and length. However, some window contains unfavorable spectral side lobes, which distort the features. Accordingly, it is preferable to adapt the window shape and length in terms of classification performance. We propose an experimental evaluation for three window shapes, each with three window lengths. The biLSTM network is trained and tested on statistical features extracted, and the performance is reported in terms of the window shapes and lengths. Results show that the best performance is obtained when the Gaussian window is used for splitting the signals, and the triangular window competes with the Gaussian window for a length of 75 ms. Although the rectangular window is a commonly offered option, it is the worst choice for splitting the signals. Moreover, the classification performance obtained with a 75 ms Gaussian window outperforms that of a baseline method.
>
---
#### [new 008] Graph Propagated Projection Unlearning: A Unified Framework for Vision and Audio Discriminative Models
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文提出GPPU，解决深度学习模型中类级信息擦除问题，通过图传播和投影实现高效、可逆的模型遗忘，适用于视觉和音频任务。**

- **链接: [https://arxiv.org/pdf/2604.13127](https://arxiv.org/pdf/2604.13127)**

> **作者:** Shreyansh Pathak; Jyotishman Das
>
> **摘要:** The need to selectively and efficiently erase learned information from deep neural networks is becoming increasingly important for privacy, regulatory compliance, and adaptive system design. We introduce Graph-Propagated Projection Unlearning (GPPU), a unified and scalable algorithm for class-level unlearning that operates across both vision and audio models. GPPU employs graph-based propagation to identify class-specific directions in the feature space and projects representations onto the orthogonal subspace, followed by targeted fine-tuning, to ensure that target class information is effectively and irreversibly removed. Through comprehensive evaluations on six vision datasets and two large-scale audio benchmarks spanning a variety of architectures including CNNs, Vision Transformers, and Audio Transformers, we demonstrate that GPPU achieves highly efficient unlearning, realizing 10-20x speedups over prior methodologies while preserving model utility on retained classes. Our framework provides a principled and modality-agnostic approach to machine unlearning, evaluated at a scale that has received limited attention in prior work, contributing toward more efficient and responsible deep learning.
>
---
## 更新

#### [replaced 001] AudioX: A Unified Framework for Anything-to-Audio Generation
- **分类: cs.MM; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出AudioX，解决多模态到音频生成的任务，通过统一框架和高质量数据集，提升生成质量与跨模态对齐。**

- **链接: [https://arxiv.org/pdf/2503.10522](https://arxiv.org/pdf/2503.10522)**

> **作者:** Zeyue Tian; Zhaoyang Liu; Yizhu Jin; Ruibin Yuan; Liumeng Xue; Xu Tan; Qifeng Chen; Wei Xue; Yike Guo
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, and 2) large-scale, high-quality training data. As such, we propose AudioX, a unified framework for anything-to-audio generation that integrates varied multimodal conditions (i.e., text, video, and audio signals) in this work. The core design in this framework is a Multimodal Adaptive Fusion module, which enables the effective fusion of diverse multimodal inputs, enhancing cross-modal alignment and improving overall generation quality. To train this unified model, we construct a large-scale, high-quality dataset, IF-caps, comprising over 7 million samples curated through a structured data annotation pipeline. This dataset provides comprehensive supervision for multimodal-conditioned audio generation. We benchmark AudioX against state-of-the-art methods across a wide range of tasks, finding that our model achieves superior performance, especially in text-to-audio and text-to-music generation. These results demonstrate our method is capable of audio generation under multimodal control signals, showing powerful instruction-following potential. The code and datasets will be available at this https URL.
>
---
#### [replaced 002] Generative AI in Signal Processing Education: An Audio Foundation Model Based Approach
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于教育技术任务，旨在将生成式AI应用于信号处理教学。通过提出SPEduAFM框架，解决传统教学与AI创新结合的问题，提升教学互动与实践性。**

- **链接: [https://arxiv.org/pdf/2602.01249](https://arxiv.org/pdf/2602.01249)**

> **作者:** Muhammad Salman Khan; Ahmad Ullah; Siddique Latif; Junaid Qadir
>
> **备注:** accepted at IEEE EDUCON 2026
>
> **摘要:** Audio Foundation Models (AFMs), a specialized category of Generative AI (GenAI), have the potential to transform signal processing (SP) education by integrating core applications such as speech and audio enhancement, denoising, source separation, feature extraction, automatic classification, and real-time signal analysis into learning and research. This paper introduces SPEduAFM, a conceptual AFM tailored for SP education, bridging traditional SP principles with GenAI-driven innovations. Through an envisioned case study, we outline how AFMs can enable a range of applications, including automated lecture transcription, interactive demonstrations, and inclusive learning tools, showcasing their potential to transform abstract concepts into engaging, practical experiences. This paper also addresses challenges such as ethics, explainability, and customization by highlighting dynamic, real-time auditory interactions that foster experiential and authentic learning. By presenting SPEduAFM as a forward-looking vision, we aim to inspire broader adoption of GenAI in engineering education, enhancing accessibility, engagement, and innovation in the classroom and beyond.
>
---
