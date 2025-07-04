# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Analyzing and Improving Speaker Similarity Assessment for Speech Synthesis
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音合成中的说话人相似性评估任务，旨在解决ASV嵌入忽略动态特征的问题，提出U3D度量动态节奏模式。**

- **链接: [http://arxiv.org/pdf/2507.02176v1](http://arxiv.org/pdf/2507.02176v1)**

> **作者:** Marc-André Carbonneau; Benjamin van Niekerk; Hugo Seuté; Jean-Philippe Letendre; Herman Kamper; Julian Zaïdi
>
> **备注:** Accepted at SSW13 - Interspeech 2025 Speech Synthesis Workshop
>
> **摘要:** Modeling voice identity is challenging due to its multifaceted nature. In generative speech systems, identity is often assessed using automatic speaker verification (ASV) embeddings, designed for discrimination rather than characterizing identity. This paper investigates which aspects of a voice are captured in such representations. We find that widely used ASV embeddings focus mainly on static features like timbre and pitch range, while neglecting dynamic elements such as rhythm. We also identify confounding factors that compromise speaker similarity measurements and suggest mitigation strategies. To address these gaps, we propose U3D, a metric that evaluates speakers' dynamic rhythm patterns. This work contributes to the ongoing challenge of assessing speaker identity consistency in the context of ever-better voice cloning systems. We publicly release our code.
>
---
#### [new 002] Acoustic evaluation of a neural network dedicated to the detection of animal vocalisations
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于动物叫声检测任务，旨在解决自动检测系统性能评估问题，通过声学分析优化模型并估计叫声空间密度。**

- **链接: [http://arxiv.org/pdf/2507.01974v1](http://arxiv.org/pdf/2507.01974v1)**

> **作者:** Jérémy Rouch; M Ducrettet; S Haupert; R Emonet; F Sèbe
>
> **摘要:** The accessibility of long-duration recorders, adapted to sometimes demanding field conditions, has enabled the deployment of extensive animal population monitoring campaigns through ecoacoustics. The effectiveness of automatic signal detection methods, increasingly based on neural approaches, is frequently evaluated solely through machine learning metrics, while acoustic analysis of performance remains rare. As part of the acoustic monitoring of Rock Ptarmigan populations, we propose here a simple method for acoustic analysis of the detection system's performance. The proposed measure is based on relating the signal-to-noise ratio of synthetic signals to their probability of detection. We show how this measure provides information about the system and allows optimisation of its training. We also show how it enables modelling of the detection distance, thus offering the possibility of evaluating its dynamics according to the sound environment and accessing an estimation of the spatial density of calls.
>
---
#### [new 003] JoyTTS: LLM-based Spoken Chatbot With Voice Cloning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音对话任务，旨在实现基于大语言模型的语音聊天机器人，并解决语音克隆问题。通过结合LLM与TTS技术，构建了JoyTTS系统。**

- **链接: [http://arxiv.org/pdf/2507.02380v1](http://arxiv.org/pdf/2507.02380v1)**

> **作者:** Fangru Zhou; Jun Zhao; Guoxin Wang
>
> **摘要:** JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at https://github.com/jdh-algo/JoyTTS.git.
>
---
#### [new 004] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于语音安全任务，旨在解决语音克隆攻击问题。通过分析对抗扰动的防御效果，提出一种两阶段净化方法以提升防御能力。**

- **链接: [http://arxiv.org/pdf/2507.02606v1](http://arxiv.org/pdf/2507.02606v1)**

> **作者:** Wei Fan; Kejiang Chen; Chang Liu; Weiming Zhang; Nenghai Yu
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at https://de-antifake.github.io.
>
---
#### [new 005] Fx-Encoder++: Extracting Instrument-Wise Audio Effects Representations from Mixtures
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决智能音乐制作中对乐器级音频效果理解不足的问题。提出Fx-Encoder++模型，通过对比学习提取乐器级效果表示。**

- **链接: [http://arxiv.org/pdf/2507.02273v1](http://arxiv.org/pdf/2507.02273v1)**

> **作者:** Yen-Tung Yeh; Junghyun Koo; Marco A. Martínez-Ramírez; Wei-Hsiang Liao; Yi-Hsuan Yang; Yuki Mitsufuji
>
> **备注:** ISMIR 2025
>
> **摘要:** General-purpose audio representations have proven effective across diverse music information retrieval applications, yet their utility in intelligent music production remains limited by insufficient understanding of audio effects (Fx). Although previous approaches have emphasized audio effects analysis at the mixture level, this focus falls short for tasks demanding instrument-wise audio effects understanding, such as automatic mixing. In this work, we present Fx-Encoder++, a novel model designed to extract instrument-wise audio effects representations from music mixtures. Our approach leverages a contrastive learning framework and introduces an "extractor" mechanism that, when provided with instrument queries (audio or text), transforms mixture-level audio effects embeddings into instrument-wise audio effects embeddings. We evaluated our model across retrieval and audio effects parameter matching tasks, testing its performance across a diverse range of instruments. The results demonstrate that Fx-Encoder++ outperforms previous approaches at mixture level and show a novel ability to extract effects representation instrument-wise, addressing a critical capability gap in intelligent music production systems.
>
---
#### [new 006] Posterior Transition Modeling for Unsupervised Diffusion-Based Speech Enhancement
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，解决无监督环境下噪声语音的提升问题。通过改进扩散模型的逆向过程，提高增强效果和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.02391v1](http://arxiv.org/pdf/2507.02391v1)**

> **作者:** Mostafa Sadeghi; Jean-Eudes Ayilo; Romain Serizel; Xavier Alameda-Pineda
>
> **摘要:** We explore unsupervised speech enhancement using diffusion models as expressive generative priors for clean speech. Existing approaches guide the reverse diffusion process using noisy speech through an approximate, noise-perturbed likelihood score, combined with the unconditional score via a trade-off hyperparameter. In this work, we propose two alternative algorithms that directly model the conditional reverse transition distribution of diffusion states. The first method integrates the diffusion prior with the observation model in a principled way, removing the need for hyperparameter tuning. The second defines a diffusion process over the noisy speech itself, yielding a fully tractable and exact likelihood score. Experiments on the WSJ0-QUT and VoiceBank-DEMAND datasets demonstrate improved enhancement metrics and greater robustness to domain shifts compared to both supervised and unsupervised baselines.
>
---
#### [new 007] ASDA: Audio Spectrogram Differential Attention Mechanism for Self-Supervised Representation Learning
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频自监督表示学习任务，旨在解决Transformer注意力机制分配无效注意力的问题，提出ASDA模型通过差分注意力机制提升性能。**

- **链接: [http://arxiv.org/pdf/2507.02666v1](http://arxiv.org/pdf/2507.02666v1)**

> **作者:** Junyu Wang; Tianrui Wang; Meng Ge; Longbiao Wang; Jianwu Dang
>
> **备注:** Accepted at Interspeech2025
>
> **摘要:** In recent advancements in audio self-supervised representation learning, the standard Transformer architecture has emerged as the predominant approach, yet its attention mechanism often allocates a portion of attention weights to irrelevant information, potentially impairing the model's discriminative ability. To address this, we introduce a differential attention mechanism, which effectively mitigates ineffective attention allocation through the integration of dual-softmax operations and appropriately tuned differential coefficients. Experimental results demonstrate that our ASDA model achieves state-of-the-art (SOTA) performance across multiple benchmarks, including audio classification (49.0% mAP on AS-2M, 41.5% mAP on AS20K), keyword spotting (98.3% accuracy on SPC-2), and environmental sound classification (96.1% accuracy on ESC-50). These results highlight ASDA's effectiveness in audio tasks, paving the way for broader applications.
>
---
#### [new 008] TAGF: Time-aware Gated Fusion for Multimodal Valence-Arousal Estimation
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于多模态情感识别任务，旨在解决情绪评估中因噪声和模态错位导致的性能下降问题。提出TAGF框架，通过时间感知门控融合提升模型鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.02080v1](http://arxiv.org/pdf/2507.02080v1)**

> **作者:** Yubeen Lee; Sangeun Lee; Chaewon Park; Junyeop Cha; Eunil Park
>
> **备注:** 9 pages, 2 figures, 2 tables
>
> **摘要:** Multimodal emotion recognition often suffers from performance degradation in valence-arousal estimation due to noise and misalignment between audio and visual modalities. To address this challenge, we introduce TAGF, a Time-aware Gated Fusion framework for multimodal emotion recognition. The TAGF adaptively modulates the contribution of recursive attention outputs based on temporal dynamics. Specifically, the TAGF incorporates a BiLSTM-based temporal gating mechanism to learn the relative importance of each recursive step and effectively integrates multistep cross-modal features. By embedding temporal awareness into the recursive fusion process, the TAGF effectively captures the sequential evolution of emotional expressions and the complex interplay between modalities. Experimental results on the Aff-Wild2 dataset demonstrate that TAGF achieves competitive performance compared with existing recursive attention-based models. Furthermore, TAGF exhibits strong robustness to cross-modal misalignment and reliably models dynamic emotional transitions in real-world conditions.
>
---
#### [new 009] Parametric Neural Amp Modeling with Active Learning
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音频建模任务，旨在用最少样本训练吉他放大器模型。通过主动学习和梯度优化，提升模型训练效率。**

- **链接: [http://arxiv.org/pdf/2507.02109v1](http://arxiv.org/pdf/2507.02109v1)**

> **作者:** Florian Grötschla; Luca A. Lanzendörfer; Longxiang Jiao; Roger Wattenhofer
>
> **备注:** Accepted at ISMIR 2025 as Late-Breaking Demo (LBD)
>
> **摘要:** We introduce PANAMA, an active learning framework for the training of end-to-end parametric guitar amp models using a WaveNet-like architecture. With \model, one can create a virtual amp by recording samples that are determined by an active learning strategy to use a minimum amount of datapoints (i.e., amp knob settings). We show that gradient-based optimization algorithms can be used to determine the optimal datapoints to sample, and that the approach helps under a constrained number of samples.
>
---
#### [new 010] Padé Approximant Neural Networks for Enhanced Electric Motor Fault Diagnosis Using Vibration and Acoustic Data
- **分类: cs.LG; cs.SD; cs.SY; eess.SY**

- **简介: 该论文属于电机故障诊断任务，旨在提升感应电机故障检测性能。通过引入PadéNets模型，利用振动和声学数据进行对比实验，验证其优于传统CNN和Self-ONN方法。**

- **链接: [http://arxiv.org/pdf/2507.02599v1](http://arxiv.org/pdf/2507.02599v1)**

> **作者:** Sertac Kilickaya; Levent Eren
>
> **备注:** Submitted to the Journal of Vibration Engineering & Technologies
>
> **摘要:** Purpose: The primary aim of this study is to enhance fault diagnosis in induction machines by leveraging the Pad\'e Approximant Neuron (PAON) model. While accelerometers and microphones are standard in motor condition monitoring, deep learning models with nonlinear neuron architectures offer promising improvements in diagnostic performance. This research addresses the question: Can Pad\'e Approximant Neural Networks (Pad\'eNets) outperform conventional Convolutional Neural Networks (CNNs) and Self-Organized Operational Neural Networks (Self-ONNs) in diagnosing electrical and mechanical faults using vibration and acoustic data? Methods: We evaluate and compare the diagnostic capabilities of three deep learning architectures: one-dimensional CNNs, Self-ONNs, and Pad\'eNets. These models are tested on the University of Ottawa's publicly available constant-speed induction motor datasets, which include both vibration and acoustic sensor data. The Pad\'eNet model is designed to introduce enhanced nonlinearity and is compatible with unbounded activation functions such as Leaky ReLU. Results and Conclusion: Pad\'eNets consistently outperformed the baseline models, achieving diagnostic accuracies of 99.96%, 98.26%, 97.61%, and 98.33% for accelerometers 1, 2, 3, and the acoustic sensor, respectively. The enhanced nonlinearity of Pad\'eNets, together with their compatibility with unbounded activation functions, significantly improves fault diagnosis performance in induction motor condition monitoring.
>
---
#### [new 011] Benchmarking Akan ASR Models Across Domain-Specific Datasets: A Comparative Evaluation of Performance, Scalability, and Adaptability
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在评估Akan语言模型在不同领域的泛化能力，通过对比性能、可扩展性和适应性，揭示模型在跨领域场景下的表现差异。**

- **链接: [http://arxiv.org/pdf/2507.02407v1](http://arxiv.org/pdf/2507.02407v1)**

> **作者:** Mark Atta Mensah; Isaac Wiafe; Akon Ekpezu; Justice Kwame Appati; Jamal-Deen Abdulai; Akosua Nyarkoa Wiafe-Akenten; Frank Ernest Yeboah; Gifty Odame
>
> **备注:** This version has been reviewed and accepted for presentation at the Future Technologies Conference (FTC) 2025, to be held on 6 & 7 November 2025 in Munich, Germany. 17 pages, 4 figures, 1 table
>
> **摘要:** Most existing automatic speech recognition (ASR) research evaluate models using in-domain datasets. However, they seldom evaluate how they generalize across diverse speech contexts. This study addresses this gap by benchmarking seven Akan ASR models built on transformer architectures, such as Whisper and Wav2Vec2, using four Akan speech corpora to determine their performance. These datasets encompass various domains, including culturally relevant image descriptions, informal conversations, biblical scripture readings, and spontaneous financial dialogues. A comparison of the word error rate and character error rate highlighted domain dependency, with models performing optimally only within their training domains while showing marked accuracy degradation in mismatched scenarios. This study also identified distinct error behaviors between the Whisper and Wav2Vec2 architectures. Whereas fine-tuned Whisper Akan models led to more fluent but potentially misleading transcription errors, Wav2Vec2 produced more obvious yet less interpretable outputs when encountering unfamiliar inputs. This trade-off between readability and transparency in ASR errors should be considered when selecting architectures for low-resource language (LRL) applications. These findings highlight the need for targeted domain adaptation techniques, adaptive routing strategies, and multilingual training frameworks for Akan and other LRLs.
>
---
#### [new 012] Multi-Utterance Speech Separation and Association Trained on Short Segments
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多说话人语音分离任务，解决模型在短段训练后处理长音频时的性能问题。提出FTRNN模型，实现长音频的端到端分离与说话人关联。**

- **链接: [http://arxiv.org/pdf/2507.02562v1](http://arxiv.org/pdf/2507.02562v1)**

> **作者:** Yuzhu Wang; Archontis Politis; Konstantinos Drossos; Tuomas Virtanen
>
> **备注:** 5 pages, accepted by WASPAA 2025
>
> **摘要:** Current deep neural network (DNN) based speech separation faces a fundamental challenge -- while the models need to be trained on short segments due to computational constraints, real-world applications typically require processing significantly longer recordings with multiple utterances per speaker than seen during training. In this paper, we investigate how existing approaches perform in this challenging scenario and propose a frequency-temporal recurrent neural network (FTRNN) that effectively bridges this gap. Our FTRNN employs a full-band module to model frequency dependencies within each time frame and a sub-band module that models temporal patterns in each frequency band. Despite being trained on short fixed-length segments of 10 s, our model demonstrates robust separation when processing signals significantly longer than training segments (21-121 s) and preserves speaker association across utterance gaps exceeding those seen during training. Unlike the conventional segment-separation-stitch paradigm, our lightweight approach (0.9 M parameters) performs inference on long audio without segmentation, eliminating segment boundary distortions while simplifying deployment. Experimental results demonstrate the generalization ability of FTRNN for multi-utterance speech separation and speaker association.
>
---
#### [new 013] Self-Steering Deep Non-Linear Spatially Selective Filters for Efficient Extraction of Moving Speakers under Weak Guidance
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音增强任务，解决动态场景下说话人跟踪问题。通过结合低复杂度粒子滤波与空间选择性滤波器，提升跟踪精度和增强效果。**

- **链接: [http://arxiv.org/pdf/2507.02791v1](http://arxiv.org/pdf/2507.02791v1)**

> **作者:** Jakob Kienegger; Alina Mannanova; Huajian Fang; Timo Gerkmann
>
> **备注:** Accepted at IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** Recent works on deep non-linear spatially selective filters demonstrate exceptional enhancement performance with computationally lightweight architectures for stationary speakers of known directions. However, to maintain this performance in dynamic scenarios, resource-intensive data-driven tracking algorithms become necessary to provide precise spatial guidance conditioned on the initial direction of a target speaker. As this additional computational overhead hinders application in resource-constrained scenarios such as real-time speech enhancement, we present a novel strategy utilizing a low-complexity tracking algorithm in the form of a particle filter instead. Assuming a causal, sequential processing style, we introduce temporal feedback to leverage the enhanced speech signal of the spatially selective filter to compensate for the limited modeling capabilities of the particle filter. Evaluation on a synthetic dataset illustrates how the autoregressive interplay between both algorithms drastically improves tracking accuracy and leads to strong enhancement performance. A listening test with real-world recordings complements these findings by indicating a clear trend towards our proposed self-steering pipeline as preferred choice over comparable methods.
>
---
#### [new 014] Towards Perception-Informed Latent HRTF Representations
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于HRTF个性化任务，旨在解决传统方法在感知兼容性上的不足。通过构建感知引导的潜在空间提升空间音频体验。**

- **链接: [http://arxiv.org/pdf/2507.02815v1](http://arxiv.org/pdf/2507.02815v1)**

> **作者:** You Zhang; Andrew Francl; Ruohan Gao; Paul Calamia; Zhiyao Duan; Ishwarya Ananthabhotla
>
> **备注:** Accepted by IEEE WASPAA 2025, camera-ready version
>
> **摘要:** Personalized head-related transfer functions (HRTFs) are essential for ensuring a realistic auditory experience over headphones, because they take into account individual anatomical differences that affect listening. Most machine learning approaches to HRTF personalization rely on a learned low-dimensional latent space to generate or select custom HRTFs for a listener. However, these latent representations are typically learned in a manner that optimizes for spectral reconstruction but not for perceptual compatibility, meaning they may not necessarily align with perceptual distance. In this work, we first study whether traditionally learned HRTF representations are well correlated with perceptual relations using auditory-based objective perceptual metrics; we then propose a method for explicitly embedding HRTFs into a perception-informed latent space, leveraging a metric-based loss function and supervision via Metric Multidimensional Scaling (MMDS). Finally, we demonstrate the applicability of these learned representations to the task of HRTF personalization. We suggest that our method has the potential to render personalized spatial audio, leading to an improved listening experience.
>
---
#### [new 015] A Late Collaborative Perception Framework for 3D Multi-Object and Multi-Source Association and Fusion
- **分类: cs.RO; eess.IV; eess.SP**

- **简介: 该论文属于多目标多源融合任务，解决自动驾驶中协作感知的通信与模型保护问题。提出一种基于共享3D边界框属性的晚期融合框架，提升精度并降低误差。**

- **链接: [http://arxiv.org/pdf/2507.02430v1](http://arxiv.org/pdf/2507.02430v1)**

> **作者:** Maryem Fadili; Mohamed Anis Ghaoui; Louis Lecrosnier; Steve Pechberti; Redouane Khemmar
>
> **摘要:** In autonomous driving, recent research has increasingly focused on collaborative perception based on deep learning to overcome the limitations of individual perception systems. Although these methods achieve high accuracy, they rely on high communication bandwidth and require unrestricted access to each agent's object detection model architecture and parameters. These constraints pose challenges real-world autonomous driving scenarios, where communication limitations and the need to safeguard proprietary models hinder practical implementation. To address this issue, we introduce a novel late collaborative framework for 3D multi-source and multi-object fusion, which operates solely on shared 3D bounding box attributes-category, size, position, and orientation-without necessitating direct access to detection models. Our framework establishes a new state-of-the-art in late fusion, achieving up to five times lower position error compared to existing methods. Additionally, it reduces scale error by a factor of 7.5 and orientation error by half, all while maintaining perfect 100% precision and recall when fusing detections from heterogeneous perception systems. These results highlight the effectiveness of our approach in addressing real-world collaborative perception challenges, setting a new benchmark for efficient and scalable multi-agent fusion.
>
---
#### [new 016] DeSTA2.5-Audio: Toward General-Purpose Large Audio Language Model with Self-Generated Cross-Modal Alignment
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出DeSTA2.5-Audio，一种通用大音频语言模型，解决音频感知与指令跟随问题，通过自生成跨模态对齐策略提升性能。**

- **链接: [http://arxiv.org/pdf/2507.02768v1](http://arxiv.org/pdf/2507.02768v1)**

> **作者:** Ke-Han Lu; Zhehuai Chen; Szu-Wei Fu; Chao-Han Huck Yang; Sung-Feng Huang; Chih-Kai Yang; Chee-En Yu; Chun-Wei Chen; Wei-Chih Chen; Chien-yu Huang; Yi-Cheng Lin; Yu-Xiang Lin; Chi-An Fu; Chun-Yi Kuan; Wenze Ren; Xuanjun Chen; Wei-Ping Huang; En-Pei Hu; Tzu-Quan Lin; Yuan-Kuei Wu; Kuan-Po Huang; Hsiao-Ying Huang; Huang-Cheng Chou; Kai-Wei Chang; Cheng-Han Chiang; Boris Ginsburg; Yu-Chiang Frank Wang; Hung-yi Lee
>
> **备注:** Model and code available at: https://github.com/kehanlu/DeSTA2.5-Audio
>
> **摘要:** We introduce DeSTA2.5-Audio, a general-purpose Large Audio Language Model (LALM) designed for robust auditory perception and instruction-following, without requiring task-specific audio instruction-tuning. Recent LALMs typically augment Large Language Models (LLMs) with auditory capabilities by training on large-scale, manually curated or LLM-synthesized audio-instruction datasets. However, these approaches have often suffered from the catastrophic forgetting of the LLM's original language abilities. To address this, we revisit the data construction pipeline and propose DeSTA, a self-generated cross-modal alignment strategy in which the backbone LLM generates its own training targets. This approach preserves the LLM's native language proficiency while establishing effective audio-text alignment, thereby enabling zero-shot generalization without task-specific tuning. Using DeSTA, we construct DeSTA-AQA5M, a large-scale, task-agnostic dataset containing 5 million training samples derived from 7,000 hours of audio spanning 50 diverse datasets, including speech, environmental sounds, and music. DeSTA2.5-Audio achieves state-of-the-art or competitive performance across a wide range of audio-language benchmarks, including Dynamic-SUPERB, MMAU, SAKURA, Speech-IFEval, and VoiceBench. Comprehensive comparative studies demonstrate that our self-generated strategy outperforms widely adopted data construction and training strategies in both auditory perception and instruction-following capabilities. Our findings underscore the importance of carefully designed data construction in LALM development and offer practical insights for building robust, general-purpose LALMs.
>
---
## 更新

#### [replaced 001] Neural Scoring: A Refreshed End-to-End Approach for Speaker Recognition in Complex Conditions
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.16428v3](http://arxiv.org/pdf/2410.16428v3)**

> **作者:** Wan Lin; Junhui Chen; Tianhao Wang; Zhenyu Zhou; Lantian Li; Dong Wang
>
> **摘要:** Modern speaker verification systems primarily rely on speaker embeddings, followed by verification based on cosine similarity between the embedding vectors of the enrollment and test utterances. While effective, these methods struggle with multi-talker speech due to the unidentifiability of embedding vectors. In this paper, we propose Neural Scoring (NS), a refreshed end-to-end framework that directly estimates verification posterior probabilities without relying on test-side embeddings, making it more robust to complex conditions, e.g., with multiple talkers. To make the training of such an end-to-end model more efficient, we introduce a large-scale trial e2e training (LtE2E) strategy, where each test utterance pairs with a set of enrolled speakers, thus enabling the processing of large-scale verification trials per batch. Experiments on the VoxCeleb dataset demonstrate that NS consistently outperforms both the baseline and competitive methods across various conditions, achieving an overall 70.36% reduction in EER compared to the baseline.
>
---
#### [replaced 002] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v2](http://arxiv.org/pdf/2506.12479v2)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 003] ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.16889v3](http://arxiv.org/pdf/2506.16889v3)**

> **作者:** Junghyun Koo; Marco A. Martínez-Ramírez; Wei-Hsiang Liao; Giorgio Fabbro; Michele Mancusi; Yuki Mitsufuji
>
> **备注:** ISMIR 2025
>
> **摘要:** Music mastering style transfer aims to model and apply the mastering characteristics of a reference track to a target track, simulating the professional mastering process. However, existing methods apply fixed processing based on a reference track, limiting users' ability to fine-tune the results to match their artistic intent. In this paper, we introduce the ITO-Master framework, a reference-based mastering style transfer system that integrates Inference-Time Optimization (ITO) to enable finer user control over the mastering process. By optimizing the reference embedding during inference, our approach allows users to refine the output dynamically, making micro-level adjustments to achieve more precise mastering results. We explore both black-box and white-box methods for modeling mastering processors and demonstrate that ITO improves mastering performance across different styles. Through objective evaluation, subjective listening tests, and qualitative analysis using text-based conditioning with CLAP embeddings, we validate that ITO enhances mastering style similarity while offering increased adaptability. Our framework provides an effective and user-controllable solution for mastering style transfer, allowing users to refine their results beyond the initial style transfer.
>
---
#### [replaced 004] Prompt-Guided Turn-Taking Prediction
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21191v2](http://arxiv.org/pdf/2506.21191v2)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at SIGdial Meeting on Discourse and Dialogue 2025 (SIGDIAL 2025) and represents the author's version of the work
>
> **摘要:** Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
>
---
#### [replaced 005] Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.17772v3](http://arxiv.org/pdf/2501.17772v3)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** accepted for publication in IEEE TASLP
>
> **摘要:** Recent developments in Self-Supervised Learning (SSL) have demonstrated significant potential for Speaker Verification (SV), but closing the performance gap with supervised systems remains an ongoing challenge. SSL frameworks rely on anchor-positive pairs, constructed from segments of the same audio utterance. Hence, positives have channel characteristics similar to those of their corresponding anchors, even with extensive data-augmentation. Therefore, this positive sampling strategy is a fundamental limitation as it encodes too much information regarding the recording source in the learned representations. This article introduces Self-Supervised Positive Sampling (SSPS), a bootstrapped technique for sampling appropriate and diverse positives in SSL frameworks for SV. SSPS samples positives close to their anchor in the representation space, assuming that these pseudo-positives belong to the same speaker identity but correspond to different recording conditions. This method consistently demonstrates improvements in SV performance on VoxCeleb benchmarks when applied to major SSL frameworks, including SimCLR, SwAV, VICReg, and DINO. Using SSPS, SimCLR and DINO achieve 2.57% and 2.53% EER on VoxCeleb1-O, respectively. SimCLR yields a 58% relative reduction in EER, getting comparable performance to DINO with a simpler training framework. Furthermore, SSPS lowers intra-class variance and reduces channel information in speaker representations while exhibiting greater robustness without data-augmentation.
>
---
#### [replaced 006] KeyNode-Driven Geometry Coding for Real-World Scanned Human Dynamic Mesh Compression
- **分类: cs.CV; cs.MM; eess.SP**

- **链接: [http://arxiv.org/pdf/2501.01717v2](http://arxiv.org/pdf/2501.01717v2)**

> **作者:** Huong Hoang; Truong Nguyen; Pamela Cosman
>
> **摘要:** The compression of real-world scanned 3D human dynamic meshes is an emerging research area, driven by applications such as telepresence, virtual reality, and 3D digital streaming. Unlike synthesized dynamic meshes with fixed topology, scanned dynamic meshes often not only have varying topology across frames but also scan defects such as holes and outliers, increasing the complexity of prediction and compression. Additionally, human meshes often combine rigid and non-rigid motions, making accurate prediction and encoding significantly more difficult compared to objects that exhibit purely rigid motion. To address these challenges, we propose a compression method designed for real-world scanned human dynamic meshes, leveraging embedded key nodes. The temporal motion of each vertex is formulated as a distance-weighted combination of transformations from neighboring key nodes, requiring the transmission of solely the key nodes' transformations. To enhance the quality of the KeyNode-driven prediction, we introduce an octree-based residual coding scheme and a Dual-direction prediction mode, which uses I-frames from both directions. Extensive experiments demonstrate that our method achieves significant improvements over the state-of-the-art, with an average bitrate savings of 58.43% across the evaluated sequences, particularly excelling at low bitrates.
>
---
