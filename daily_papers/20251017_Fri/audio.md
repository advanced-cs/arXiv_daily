# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Spatially Aware Self-Supervised Models for Multi-Channel Neural Speaker Diarization
- **分类: eess.AS**

- **简介: 该论文研究多通道神经说话人分离任务，旨在解决单通道预训练模型在多通道场景中无法充分利用空间信息的问题。作者提出一种轻量方法，通过在WavLM早期层插入通道通信模块并融合空间注意力加权的说话人嵌入，提升性能与效率。**

- **链接: [http://arxiv.org/pdf/2510.14551v1](http://arxiv.org/pdf/2510.14551v1)**

> **作者:** Jiangyu Han; Ruoyu Wang; Yoshiki Masuyama; Marc Delcroix; Johan Rohdin; Jun Du; Lukas Burget
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Self-supervised models such as WavLM have demonstrated strong performance for neural speaker diarization. However, these models are typically pre-trained on single-channel recordings, limiting their effectiveness in multi-channel scenarios. Existing diarization systems built on these models often rely on DOVER-Lap to combine outputs from individual channels. Although effective, this approach incurs substantial computational overhead and fails to fully exploit spatial information. In this work, building on DiariZen, a pipeline that combines WavLM-based local endto-end neural diarization with speaker embedding clustering, we introduce a lightweight approach to make pre-trained WavLM spatially aware by inserting channel communication modules into the early layers. Our method is agnostic to both the number of microphone channels and array topologies, ensuring broad applicability. We further propose to fuse multi-channel speaker embeddings by leveraging spatial attention weights. Evaluations on five public datasets show consistent improvements over single-channel baselines and demonstrate superior performance and efficiency compared with DOVER-Lap. Our source code is publicly available at https://github.com/BUTSpeechFIT/DiariZen.
>
---
#### [new 002] Big Data Approaches to Bovine Bioacoustics: A FAIR-Compliant Dataset and Scalable ML Framework for Precision Livestock Welfare
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文聚焦牛只生物声学分析，旨在解决畜牧养殖中生物音频数据利用不足的问题。作者构建了符合FAIR标准的大规模数据集，并开发可扩展的机器学习框架，实现发情、痛苦等行为的精准识别，推动基于声音的非侵入式牲畜福利监测。**

- **链接: [http://arxiv.org/pdf/2510.14443v1](http://arxiv.org/pdf/2510.14443v1)**

> **作者:** Mayuri Kate; Suresh Neethirajan
>
> **备注:** 40 pages, 14 figures, 9 Tables
>
> **摘要:** The convergence of IoT sensing, edge computing, and machine learning is transforming precision livestock farming. Yet bioacoustic data streams remain underused because of computational complexity and ecological validity challenges. We present one of the most comprehensive bovine vocalization datasets to date, with 569 curated clips covering 48 behavioral classes, recorded across three commercial dairy farms using multiple microphone arrays and expanded to 2900 samples through domain informed augmentation. This FAIR compliant resource addresses major Big Data challenges - volume (90 hours of recordings, 65.6 GB), variety (multi farm and multi zone acoustics), velocity (real time processing), and veracity (noise robust feature extraction). Our distributed processing framework integrates advanced denoising using iZotope RX, multimodal synchronization through audio and video alignment, and standardized feature engineering with 24 acoustic descriptors generated from Praat, librosa, and openSMILE. Preliminary benchmarks reveal distinct class level acoustic patterns for estrus detection, distress classification, and maternal communication. The datasets ecological realism, reflecting authentic barn acoustics rather than controlled settings, ensures readiness for field deployment. This work establishes a foundation for animal centered AI, where bioacoustic data enable continuous and non invasive welfare assessment at industrial scale. By releasing standardized pipelines and detailed metadata, we promote reproducible research that connects Big Data analytics, sustainable agriculture, and precision livestock management. The framework supports UN SDG 9, showing how data science can turn traditional farming into intelligent, welfare optimized systems that meet global food needs while upholding ethical animal care.
>
---
#### [new 003] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SpeechLLM-as-Judges新范式，旨在解决合成语音质量评估缺乏可解释性和泛化性的问题。通过构建多语言数据集SpeechEval，训练具备推理能力的语音质量评估大模型SQ-LLM，实现跨任务、跨语言的可解释语音质量评估。**

- **链接: [http://arxiv.org/pdf/2510.14664v1](http://arxiv.org/pdf/2510.14664v1)**

> **作者:** Hui Wang; Jinghua Zhao; Yifan Yang; Shujie Liu; Junyang Chen; Yanzhe Zhang; Shiwan Zhao; Jinyu Li; Jiaming Zhou; Haoqin Sun; Yan Lu; Yong Qin
>
> **摘要:** Generative speech technologies are progressing rapidly, but evaluating the perceptual quality of synthetic speech remains a core challenge. Existing methods typically rely on scalar scores or binary decisions, which lack interpretability and generalization across tasks and languages. We present SpeechLLM-as-Judges, a new paradigm for enabling large language models (LLMs) to conduct structured and explanation-based speech quality evaluation. To support this direction, we introduce SpeechEval, a large-scale dataset containing 32,207 multilingual speech clips and 128,754 annotations spanning four tasks: quality assessment, pairwise comparison, improvement suggestion, and deepfake detection. Based on this resource, we develop SQ-LLM, a speech-quality-aware LLM trained with chain-of-thought reasoning and reward optimization to improve capability. Experimental results show that SQ-LLM delivers strong performance across tasks and languages, revealing the potential of this paradigm for advancing speech quality evaluation. Relevant resources will be open-sourced.
>
---
#### [new 004] AudioEval: Automatic Dual-Perspective and Multi-Dimensional Evaluation of Text-to-Audio-Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对文本到音频生成（TTA）的评估难题，提出AudioEval数据集和Qwen-DisQA模型。通过双视角多维度人工标注，构建大规模评测基准，并训练模型实现自动、可靠的TTA质量评估。**

- **链接: [http://arxiv.org/pdf/2510.14570v1](http://arxiv.org/pdf/2510.14570v1)**

> **作者:** Hui Wang; Jinghua Zhao; Cheng Liu; Yuhang Jia; Haoqin Sun; Jiaming Zhou; Yong Qin
>
> **摘要:** Text-to-audio (TTA) is rapidly advancing, with broad potential in virtual reality, accessibility, and creative media. However, evaluating TTA quality remains difficult: human ratings are costly and limited, while existing objective metrics capture only partial aspects of perceptual quality. To address this gap, we introduce AudioEval, the first large-scale TTA evaluation dataset, containing 4,200 audio samples from 24 systems with 126,000 ratings across five perceptual dimensions, annotated by both experts and non-experts. Based on this resource, we propose Qwen-DisQA, a multimodal scoring model that jointly processes text prompts and generated audio to predict human-like quality ratings. Experiments show its effectiveness in providing reliable and scalable evaluation. The dataset will be made publicly available to accelerate future research.
>
---
#### [new 005] TASLA: Text-Aligned Speech Tokens with Multiple Layer-Aggregation
- **分类: cs.SD**

- **简介: 该论文研究语音-文本对齐的语音分词任务，旨在解决低帧率下语音重建中音质细节丢失问题。提出TASLA框架，结合多层动态注意力（MLDA）和有限标量量化（FSQ），通过融合编码器多层特征提升韵律与重建质量。**

- **链接: [http://arxiv.org/pdf/2510.14934v1](http://arxiv.org/pdf/2510.14934v1)**

> **作者:** Ming-Hao Hsu; Liang-Hsuan Tseng; Hung-yi Lee; Zhizheng Wu
>
> **摘要:** We propose Text-Aligned Speech Tokens with Multiple Layer-Aggregation (TASLA), which is a text-aligned speech tokenization framework that aims to address the problem that under a low-frame-rate and text-aligned regime, single-source speech tokens may lose acoustic details during reconstruction. On the other hand, this paper further explains how different encoder layers collaborate to capture comprehensive acoustic features for tokenization. Previous work, TASTE, proposed the text-aligned speech tokenization framework, which is a LM-friendly architecture, but struggles to capture acoustic details. We address this trade-off with two components: Multi-Layer Dynamic Attention (MLDA), which lets each text position adaptively mix shallow/deep features from a frozen speech encoder, and Finite Scalar Quantization (FSQ), a simple per-dimension discretization with smooth optimization. At about 2.62 Hz (tokens/s), TASLA consistently improves prosody and achieves competitive quality over TASTE on in-domain (LibriSpeech) and OOD (EXPRESSO, Voxceleb) sets. We further demonstrate that dynamic layer mixing is correlated with spectral flux and explains why MLDA preserves prosody under a low frame rate with extreme feature compression.
>
---
#### [new 006] Beat Detection as Object Detection
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文将节拍检测视为时序目标检测任务，提出用改进的FCOS模型检测音乐中的节拍与下节拍。通过替换骨干网络、引入特征金字塔，实现多尺度节拍检测，结合非极大抑制输出最终结果，在标准数据集上取得良好性能。**

- **链接: [http://arxiv.org/pdf/2510.14391v1](http://arxiv.org/pdf/2510.14391v1)**

> **作者:** Jaehoon Ahn; Moon-Ryul Jung
>
> **备注:** 11 pages, 4 figures, 5 tables
>
> **摘要:** Recent beat and downbeat tracking models (e.g., RNNs, TCNs, Transformers) output frame-level activations. We propose reframing this task as object detection, where beats and downbeats are modeled as temporal "objects." Adapting the FCOS detector from computer vision to 1D audio, we replace its original backbone with WaveBeat's temporal feature extractor and add a Feature Pyramid Network to capture multi-scale temporal patterns. The model predicts overlapping beat/downbeat intervals with confidence scores, followed by non-maximum suppression (NMS) to select final predictions. This NMS step serves a similar role to DBNs in traditional trackers, but is simpler and less heuristic. Evaluated on standard music datasets, our approach achieves competitive results, showing that object detection techniques can effectively model musical beats with minimal adaptation.
>
---
#### [new 007] Switchboard-Affect: Emotion Perception Labels from Conversational Speech
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文聚焦语音情感识别任务，旨在解决现有数据集情感标注不自然、标注标准不透明的问题。作者基于Switchboard语料库构建了自然对话情感标签集SWB-Affect，包含类别和维度情感标注，并分析标注依据，评估了主流模型在该数据上的表现。**

- **链接: [http://arxiv.org/pdf/2510.13906v1](http://arxiv.org/pdf/2510.13906v1)**

> **作者:** Amrit Romana; Jaya Narain; Tien Dung Tran; Andrea Davis; Jason Fong; Ramya Rasipuram; Vikramjit Mitra
>
> **备注:** 2025 13th International Conference on Affective Computing and Intelligent Interaction (ACII) https://github.com/apple/ml-switchboard-affect
>
> **摘要:** Understanding the nuances of speech emotion dataset curation and labeling is essential for assessing speech emotion recognition (SER) model potential in real-world applications. Most training and evaluation datasets contain acted or pseudo-acted speech (e.g., podcast speech) in which emotion expressions may be exaggerated or otherwise intentionally modified. Furthermore, datasets labeled based on crowd perception often lack transparency regarding the guidelines given to annotators. These factors make it difficult to understand model performance and pinpoint necessary areas for improvement. To address this gap, we identified the Switchboard corpus as a promising source of naturalistic conversational speech, and we trained a crowd to label the dataset for categorical emotions (anger, contempt, disgust, fear, sadness, surprise, happiness, tenderness, calmness, and neutral) and dimensional attributes (activation, valence, and dominance). We refer to this label set as Switchboard-Affect (SWB-Affect). In this work, we present our approach in detail, including the definitions provided to annotators and an analysis of the lexical and paralinguistic cues that may have played a role in their perception. In addition, we evaluate state-of-the-art SER models, and we find variable performance across the emotion categories with especially poor generalization for anger. These findings underscore the importance of evaluation with datasets that capture natural affective variations in speech. We release the labels for SWB-Affect to enable further analysis in this domain.
>
---
#### [new 008] Do Joint Language-Audio Embeddings Encode Perceptual Timbre Semantics?
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究联合语言-音频嵌入模型是否能捕捉人类感知的音色语义。针对音乐信息检索等任务中音色理解不足的问题，评估了MS-CLAP、LAION-CLAP和MuQ-MuLan三种模型，发现LAION-CLAP在对齐人类感知音色方面表现最优。**

- **链接: [http://arxiv.org/pdf/2510.14249v1](http://arxiv.org/pdf/2510.14249v1)**

> **作者:** Qixin Deng; Bryan Pardo; Thrasyvoulos N Pappas
>
> **摘要:** Understanding and modeling the relationship between language and sound is critical for applications such as music information retrieval,text-guided music generation, and audio captioning. Central to these tasks is the use of joint language-audio embedding spaces, which map textual descriptions and auditory content into a shared embedding space. While multimodal embedding models such as MS-CLAP, LAION-CLAP, and MuQ-MuLan have shown strong performance in aligning language and audio, their correspondence to human perception of timbre, a multifaceted attribute encompassing qualities such as brightness, roughness, and warmth, remains underexplored. In this paper, we evaluate the above three joint language-audio embedding models on their ability to capture perceptual dimensions of timbre. Our findings show that LAION-CLAP consistently provides the most reliable alignment with human-perceived timbre semantics across both instrumental sounds and audio effects.
>
---
#### [new 009] Revisit Modality Imbalance at the Decision Layer
- **分类: cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文研究多模态学习中的模态不平衡问题，发现偏差不仅存在于表征层，还显著存在于决策层。实验表明模型对音频等模态存在系统性偏好，根源在于特征空间与决策权重分布差异。作者指出融合时未校准的输出导致偏差，建议未来应关注决策层自适应权重分配机制。**

- **链接: [http://arxiv.org/pdf/2510.14411v1](http://arxiv.org/pdf/2510.14411v1)**

> **作者:** Xiaoyu Ma; Hao Chen
>
> **备注:** Some Insights in Balanced Multimodal Learning
>
> **摘要:** Multimodal learning integrates information from different modalities to enhance model performance, yet it often suffers from modality imbalance, where dominant modalities overshadow weaker ones during joint optimization. This paper reveals that such an imbalance not only occurs during representation learning but also manifests significantly at the decision layer. Experiments on audio-visual datasets (CREMAD and Kinetic-Sounds) show that even after extensive pretraining and balanced optimization, models still exhibit systematic bias toward certain modalities, such as audio. Further analysis demonstrates that this bias originates from intrinsic disparities in feature-space and decision-weight distributions rather than from optimization dynamics alone. We argue that aggregating uncalibrated modality outputs at the fusion stage leads to biased decision-layer weighting, hindering weaker modalities from contributing effectively. To address this, we propose that future multimodal systems should focus more on incorporate adaptive weight allocation mechanisms at the decision layer, enabling relative balanced according to the capabilities of each modality.
>
---
#### [new 010] Sound Masking Strategies for Interference with Mosquito Hearing
- **分类: physics.bio-ph; cs.SD**

- **简介: 该论文研究利用声音掩蔽干扰蚊子听觉通信的策略。针对病媒蚊，提出以集中频段或快速调频的声音进行有效掩蔽，阻碍其信息传递，从而控制种群。工作包括建模分析两类听觉系统，评估不同掩蔽方案效果。**

- **链接: [http://arxiv.org/pdf/2510.14921v1](http://arxiv.org/pdf/2510.14921v1)**

> **作者:** Justin Faber; Alexandros C Alampounti; Marcos Georgiades; Joerg T Albert; Dolores Bozovic
>
> **摘要:** The use of auditory masking has long been of interest in psychoacoustics and for engineering purposes, in order to cover sounds that are disruptive to humans or to species whose habitats overlap with ours. In most cases, we seek to minimize the disturbances to the communication of wildlife. However, in the case of pathogen-carrying insects, we may want to maximize these disturbances as a way to control populations. In the current work, we explore candidate masking strategies for a generic model of active auditory systems and a model of the mosquito auditory system. For both models, we find that masks with all acoustic power focused into just one or a few frequencies perform best. We propose that masks based on rapid frequency modulation are most effective for maximal disruption of information transfer and minimizing intelligibility. We hope that these results will serve to guide the avoidance or selection of possible acoustic signals for, respectively, maximizing or minimizing communication.
>
---
#### [new 011] TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG
- **分类: cs.AI; cs.CL; cs.LG; eess.AS; eess.SP**

- **简介: 该论文研究抑郁症自动检测，属多模态情感计算任务。针对现有研究缺乏系统比较与统一评估的问题，提出TRI-DEP框架，系统比较语音、文本和EEG的单、双、三模态组合，分析特征表示与融合策略，验证三模态结合预训练模型可提升性能。**

- **链接: [http://arxiv.org/pdf/2510.14922v1](http://arxiv.org/pdf/2510.14922v1)**

> **作者:** Annisaa Fitri Nurfidausi; Eleonora Mancini; Paolo Torroni
>
> **摘要:** Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection.
>
---
#### [new 012] A Robust Classification Method using Hybrid Word Embedding for Early Diagnosis of Alzheimer's Disease
- **分类: cs.CL; cs.AI; cs.LG; eess.AS; I.2.7; I.2.6**

- **简介: 该论文针对阿尔茨海默病早期诊断，提出一种融合Doc2Vec与ELMo的混合词嵌入分类方法。通过句子困惑度和语言特征分析语言能力变化，结合逻辑回归与超参数优化，实现91%准确率和97% AUC，优于现有模型，具备稳定性与临床应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.14332v1](http://arxiv.org/pdf/2510.14332v1)**

> **作者:** Yangyang Li
>
> **备注:** Peer-reviewed and published in Proceedings of the 2020 3rd International Conference on Algorithms, Computing and Artificial Intelligence (ACAI 2020). 7 pages, 5 figures
>
> **摘要:** Early detection of Alzheimer's Disease (AD) is greatly beneficial to AD patients, leading to early treatments that lessen symptoms and alleviating financial burden of health care. As one of the leading signs of AD, language capability changes can be used for early diagnosis of AD. In this paper, I develop a robust classification method using hybrid word embedding and fine-tuned hyperparameters to achieve state-of-the-art accuracy in the early detection of AD. Specifically, we create a hybrid word embedding based on word vectors from Doc2Vec and ELMo to obtain perplexity scores of the sentences. The scores identify whether a sentence is fluent or not and capture semantic context of the sentences. I enrich the word embedding by adding linguistic features to analyze syntax and semantics. Further, we input an embedded feature vector into logistic regression and fine tune hyperparameters throughout the pipeline. By tuning hyperparameters of the machine learning pipeline (e.g., model regularization parameter, learning rate and vector size of Doc2Vec, and vector size of ELMo), I achieve 91% classification accuracy and an Area Under the Curve (AUC) of 97% in distinguishing early AD from healthy subjects. Based on my knowledge, my model with 91% accuracy and 97% AUC outperforms the best existing NLP model for AD diagnosis with an accuracy of 88% [32]. I study the model stability through repeated experiments and find that the model is stable even though the training data is split randomly (standard deviation of accuracy = 0.0403; standard deviation of AUC = 0.0174). This affirms our proposed method is accurate and stable. This model can be used as a large-scale screening method for AD, as well as a complementary examination for doctors to detect AD.
>
---
#### [new 013] If You Hold Me Without Hurting Me: Pathways to Designing Game Audio for Healthy Escapism and Player Well-being
- **分类: cs.HC; cs.MM; cs.SD; H.5.5; H.5.2; J.5**

- **简介: 该论文探讨游戏音频在促进健康逃避主义与玩家福祉中的作用，旨在解决音频设计在调节情绪和自我调节中被忽视的问题。作者分析现有方法与可及性缺口，并提出改进路径，推动音频在游戏设计与研究中的整合。**

- **链接: [http://arxiv.org/pdf/2510.14691v1](http://arxiv.org/pdf/2510.14691v1)**

> **作者:** Caio Nunes; Bosco Borges; Georgia Cruz; Ticianne Darin
>
> **备注:** 5 pages. Presented and discussed at the CHI PLAY 2025 Workshop Exploring Future Directions for Healthy Escapism and Self-Regulation in Games, Pittsburgh, USA, October 13, 2025
>
> **摘要:** Escapism in games can support recovery or lead to harmful avoidance. Self-regulation, understood as combining autonomy with positive outcomes, is key to this distinction. We argue that audio, often overlooked, plays a central role in regulation. It can modulate arousal, mark transitions, and provide closure, yet its contribution to well-being remains underexplored. This paper identifies methodological and accessibility gaps that limit recognition of audio's potential and outlines ways to address them. We aim to encourage researchers and developers to integrate audio more deliberately into the design and study of healthier escapist play.
>
---
#### [new 014] Musical consonance: a review of theory and evidence on perception and preference of auditory roughness in humans and other animals
- **分类: physics.soc-ph; cs.SD; eess.AS**

- **简介: 该论文属综述任务，旨在探讨音乐协和感的成因，聚焦听觉粗糙度的理论与实证。作者评估现有模型，指出定义循环、测量依赖及过拟合等问题，主张未来研究应简化模型并扩大预测范围。**

- **链接: [http://arxiv.org/pdf/2510.14159v1](http://arxiv.org/pdf/2510.14159v1)**

> **作者:** John M. McBride
>
> **摘要:** The origins of consonance in human music has long been contested, and today there are three primary hypotheses: aversion to roughness, preference for harmonicity, and learned preferences from cultural exposure. While the evidence is currently insufficient to disentangle the contributions of these hypotheses, I propose several reasons why roughness is an especially promising area for future study. The aim of this review is to summarize and critically evaluate roughness theory and models, experimental data, to highlight areas that deserve further research. I identify 2 key areas: There are fundamental issues with the definition and interpretation of results due to tautology in the definition of roughness, and the lack of independence in empirical measurements. Despite extensive model development, there are many duplications and models have issues with data quality and overfitting. Future theory development should aim for model simplicity, and extra assumptions, features and parameters should be evaluated systematically. Model evaluation should aim to maximise the breadth of stimuli that are predicted.
>
---
## 更新

#### [replaced 001] Towards Inclusive Communication: A Unified Framework for Generating Spoken Language from Sign, Lip, and Audio
- **分类: cs.CV; cs.MM; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.20476v2](http://arxiv.org/pdf/2508.20476v2)**

> **作者:** Jeong Hun Yeo; Hyeongseop Rha; Sungjune Park; Junil Won; Yong Man Ro
>
> **摘要:** Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such audio-centric systems inherently exclude individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we propose the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and Audio-Visual Speech Recognition. Furthermore, our analysis reveals a key linguistic insight: explicitly modeling lip movements as a distinct modality significantly improves SLT performance by capturing critical non-manual cues.
>
---
#### [replaced 002] Non-invasive electromyographic speech neuroprosthesis: a geometric perspective
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2502.05762v2](http://arxiv.org/pdf/2502.05762v2)**

> **作者:** Harshavardhana T. Gowda; Lee M. Miller
>
> **摘要:** We present a high-bandwidth, egocentric neuromuscular speech interface that translates $silently$ voiced articulations directly into text. We record surface electromyographic (EMG) signals from multiple articulatory sites on the face and neck as participants $silently$ articulate speech, enabling direct EMG-to-text translation. Such an interface has the potential to restore communication for individuals who have lost the ability to produce intelligible speech due to laryngectomy, neuromuscular disease, stroke, or trauma-induced damage (e.g., radiotherapy toxicity) to the speech articulators. Prior work has largely focused on mapping EMG collected during $audible$ articulation to time-aligned audio targets or transferring these targets to $silent$ EMG recordings, which inherently requires audio and limits applicability to patients who can no longer speak. In contrast, we propose an efficient representation of high-dimensional EMG signals and demonstrate direct sequence-to-sequence EMG-to-text conversion at the phonemic level without relying on time-aligned audio. All data, code, and model checkpoints are open-sourced at The dataset and code are available at: https://github.com/HarshavardhanaTG/emg2speech .
>
---
#### [replaced 003] Pinhole Effect on Linkability and Dispersion in Speaker Anonymization
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2508.17134v2](http://arxiv.org/pdf/2508.17134v2)**

> **作者:** Kong Aik Lee; Zeyan Liu; Liping Chen; Zhenhua Ling
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Speaker anonymization aims to conceal speaker-specific attributes in speech signals, making the anonymized speech unlinkable to the original speaker identity. Recent approaches achieve this by disentangling speech into content and speaker components, replacing the latter with pseudo speakers. The anonymized speech can be mapped either to a common pseudo speaker shared across utterances or to distinct pseudo speakers unique to each utterance. This paper investigates the impact of these mapping strategies on three key dimensions: speaker linkability, dispersion in the anonymized speaker space, and de-identification from the original identity. Our findings show that using distinct pseudo speakers increases speaker dispersion and reduces linkability compared to common pseudo-speaker mapping, thereby enhancing privacy preservation. These observations are interpreted through the proposed pinhole effect, a conceptual framework introduced to explain the relationship between mapping strategies and anonymization performance. The hypothesis is validated through empirical evaluation.
>
---
#### [replaced 004] SPIRIT: Patching Speech Language Models against Jailbreak Attacks
- **分类: eess.AS; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.13541v2](http://arxiv.org/pdf/2505.13541v2)**

> **作者:** Amirbek Djanibekov; Nurdaulet Mukhituly; Kentaro Inui; Hanan Aldarmaki; Nils Lukas
>
> **摘要:** Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100% attack success rate in some instances. To improve security, we propose post-hoc patching defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.
>
---
#### [replaced 005] DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation
- **分类: eess.AS; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.12210v2](http://arxiv.org/pdf/2510.12210v2)**

> **作者:** Yakun Song; Xiaobin Zhuang; Jiawei Chen; Zhikang Niu; Guanrou Yang; Chenpeng Du; Dongya Jia; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **摘要:** Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on https://anonymous.4open.science/w/DiSTAR_demo.
>
---
