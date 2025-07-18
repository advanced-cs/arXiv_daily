# 音频 cs.SD;  eess.SP

- **最新发布 19 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Autoregressive Speech Enhancement via Acoustic Tokens
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升录音质量与可懂度。研究提出使用声学标记并设计自回归模型，以更好保留说话人特征，相比语义标记效果更优。**

- **链接: [http://arxiv.org/pdf/2507.12825v1](http://arxiv.org/pdf/2507.12825v1)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** In speech processing pipelines, improving the quality and intelligibility of real-world recordings is crucial. While supervised regression is the primary method for speech enhancement, audio tokenization is emerging as a promising alternative for a smooth integration with other modalities. However, research on speech enhancement using discrete representations is still limited. Previous work has mainly focused on semantic tokens, which tend to discard key acoustic details such as speaker identity. Additionally, these studies typically employ non-autoregressive models, assuming conditional independence of outputs and overlooking the potential improvements offered by autoregressive modeling. To address these gaps we: 1) conduct a comprehensive study of the performance of acoustic tokens for speech enhancement, including the effect of bitrate and noise strength; 2) introduce a novel transducer-based autoregressive architecture specifically designed for this task. Experiments on VoiceBank and Libri1Mix datasets show that acoustic tokens outperform semantic tokens in terms of preserving speaker identity, and that our autoregressive approach can further improve performance. Nevertheless, we observe that discrete representations still fall short compared to continuous ones, highlighting the need for further research in this area.
>
---
#### [new 002] Best Practices and Considerations for Child Speech Corpus Collection and Curation in Educational, Clinical, and Forensic Scenarios
- **分类: cs.SD; cs.CY; eess.AS**

- **简介: 该论文属于儿童语音语料库构建任务，解决如何在教育、临床和法医场景中有效收集和管理儿童语音数据的问题。工作包括提出最佳实践和操作指南。**

- **链接: [http://arxiv.org/pdf/2507.12870v1](http://arxiv.org/pdf/2507.12870v1)**

> **作者:** John Hansen; Satwik Dutta; Ellen Grand
>
> **备注:** 5 pages, 0 figures, accepted at the 10th Workshop on Speech and Language Technology in Education (SLaTE 2025), a Satellite Workshop of the 2025 Interspeech Conference
>
> **摘要:** A child's spoken ability continues to change until their adult age. Until 7-8yrs, their speech sound development and language structure evolve rapidly. This dynamic shift in their spoken communication skills and data privacy make it challenging to curate technology-ready speech corpora for children. This study aims to bridge this gap and provide researchers and practitioners with the best practices and considerations for developing such a corpus based on an intended goal. Although primarily focused on educational goals, applications of child speech data have spread across fields including clinical and forensics fields. Motivated by this goal, we describe the WHO, WHAT, WHEN, and WHERE of data collection inspired by prior collection efforts and our experience/knowledge. We also provide a guide to establish collaboration, trust, and for navigating the human subjects research protocol. This study concludes with guidelines for corpus quality check, triage, and annotation.
>
---
#### [new 003] Evaluation of Neural Surrogates for Physical Modelling Synthesis of Nonlinear Elastic Plates
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频合成任务，旨在解决非线性弹性板振动的实时建模问题。通过比较神经网络方法，评估其在长序列预测中的表现与局限性。**

- **链接: [http://arxiv.org/pdf/2507.12563v1](http://arxiv.org/pdf/2507.12563v1)**

> **作者:** Carlos De La Vega Martin; Rodrigo Diaz Fernandez; Mark Sandler
>
> **摘要:** Physical modelling synthesis aims to generate audio from physical simulations of vibrating structures. Thin elastic plates are a common model for drum membranes. Traditional numerical methods like finite differences and finite elements offer high accuracy but are computationally demanding, limiting their use in real-time audio applications. This paper presents a comparative analysis of neural network-based approaches for solving the vibration of nonlinear elastic plates. We evaluate several state-of-the-art models, trained on short sequences, for prediction of long sequences in an autoregressive fashion. We show some of the limitations of these models, and why is not enough to look at the prediction error in the time domain. We discuss the implications for real-time audio synthesis and propose future directions for improving neural approaches to model nonlinear vibration.
>
---
#### [new 004] Cross-Modal Watermarking for Authentic Audio Recovery and Tamper Localization in Synthesized Audiovisual Forgeries
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音频视频伪造检测任务，旨在解决合成音频视频中真实音频恢复与篡改定位问题，提出跨模态水印框架实现有效防御。**

- **链接: [http://arxiv.org/pdf/2507.12723v1](http://arxiv.org/pdf/2507.12723v1)**

> **作者:** Minyoung Kim; Sehwan Park; Sungmin Cha; Paul Hongsuck Seo
>
> **备注:** 5 pages, 2 figures, Interspeech 2025
>
> **摘要:** Recent advances in voice cloning and lip synchronization models have enabled Synthesized Audiovisual Forgeries (SAVFs), where both audio and visuals are manipulated to mimic a target speaker. This significantly increases the risk of misinformation by making fake content seem real. To address this issue, existing methods detect or localize manipulations but cannot recover the authentic audio that conveys the semantic content of the message. This limitation reduces their effectiveness in combating audiovisual misinformation. In this work, we introduce the task of Authentic Audio Recovery (AAR) and Tamper Localization in Audio (TLA) from SAVFs and propose a cross-modal watermarking framework to embed authentic audio into visuals before manipulation. This enables AAR, TLA, and a robust defense against misinformation. Extensive experiments demonstrate the strong performance of our method in AAR and TLA against various manipulations, including voice cloning and lip synchronization.
>
---
#### [new 005] Enkidu: Universal Frequential Perturbation for Real-Time Audio Privacy Protection against Voice Deepfakes
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频隐私保护任务，旨在解决语音深度伪造带来的隐私泄露问题。通过引入通用频域扰动方法，实现高效实时的语音隐私保护。**

- **链接: [http://arxiv.org/pdf/2507.12932v1](http://arxiv.org/pdf/2507.12932v1)**

> **作者:** Zhou Feng; Jiahao Chen; Chunyi Zhou; Yuwen Pu; Qingming Li; Tianyu Du; Shouling Ji
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** The rapid advancement of voice deepfake technologies has raised serious concerns about user audio privacy, as attackers increasingly exploit publicly available voice data to generate convincing fake audio for malicious purposes such as identity theft, financial fraud, and misinformation campaigns. While existing defense methods offer partial protection, they face critical limitations, including weak adaptability to unseen user data, poor scalability to long audio, rigid reliance on white-box knowledge, and high computational and temporal costs during the encryption process. To address these challenges and defend against personalized voice deepfake threats, we propose Enkidu, a novel user-oriented privacy-preserving framework that leverages universal frequential perturbations generated through black-box knowledge and few-shot training on a small amount of user data. These highly malleable frequency-domain noise patches enable real-time, lightweight protection with strong generalization across variable-length audio and robust resistance to voice deepfake attacks, all while preserving perceptual quality and speech intelligibility. Notably, Enkidu achieves over 50 to 200 times processing memory efficiency (as low as 0.004 gigabytes) and 3 to 7000 times runtime efficiency (real-time coefficient as low as 0.004) compared to six state-of-the-art countermeasures. Extensive experiments across six mainstream text-to-speech models and five cutting-edge automated speaker verification models demonstrate the effectiveness, transferability, and practicality of Enkidu in defending against both vanilla and adaptive voice deepfake attacks.
>
---
#### [new 006] Early Detection of Furniture-Infesting Wood-Boring Beetles Using CNN-LSTM Networks and MFCC-Based Acoustic Features
- **分类: cs.SD; cs.HC; eess.AS**

- **简介: 该论文属于早期害虫检测任务，旨在解决传统方法在木蛀虫检测中的不足。通过结合CNN-LSTM网络与MFCC特征，实现非侵入式、自动化的声学分类。**

- **链接: [http://arxiv.org/pdf/2507.12793v1](http://arxiv.org/pdf/2507.12793v1)**

> **作者:** J. M. Chan Sri Manukalpa; H. S. Bopage; W. A. M. Jayawardena; P. K. P. G. Panduwawala
>
> **备注:** This is a preprint article
>
> **摘要:** Structural pests, such as termites, pose a serious threat to wooden buildings, resulting in significant economic losses due to their hidden and progressive damage. Traditional detection methods, such as visual inspections and chemical treatments, are invasive, labor intensive, and ineffective for early stage infestations. To bridge this gap, this study proposes a non invasive deep learning based acoustic classification framework for early termite detection. We aim to develop a robust, scalable model that distinguishes termite generated acoustic signals from background noise. We introduce a hybrid Convolutional Neural Network Long Short Term Memory architecture that captures both spatial and temporal features of termite activity. Audio data were collected from termite infested and clean wooden samples. We extracted Mel Frequency Cepstral Coefficients and trained the CNN LSTM model to classify the signals. Experimental results show high performance, with 94.5% accuracy, 93.2% precision, and 95.8% recall. Comparative analysis reveals that the hybrid model outperforms standalone CNN and LSTM architectures, underscoring its combined strength. Notably, the model yields low false-negative rates, which is essential for enabling timely intervention. This research contributes a non invasive, automated solution for early termite detection, with practical implications for improved pest monitoring, minimized structural damage, and better decision making by homeowners and pest control professionals. Future work may integrate IoT for real time alerts and extend detection to other structural pests.
>
---
#### [new 007] Sample-Constrained Black Box Optimization for Audio Personalization
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于黑盒优化任务，旨在通过混合查询方式（用户评分与直接元素反馈）提升音频个性化效果，优化用户满意度。**

- **链接: [http://arxiv.org/pdf/2507.12773v1](http://arxiv.org/pdf/2507.12773v1)**

> **作者:** Rajalaxmi Rajagopalan; Yu-Lin Wei; Romit Roy Choudhury
>
> **备注:** Published in AAAI 2024
>
> **摘要:** We consider the problem of personalizing audio to maximize user experience. Briefly, we aim to find a filter $h^*$, which applied to any music or speech, will maximize the user's satisfaction. This is a black-box optimization problem since the user's satisfaction function is unknown. Substantive work has been done on this topic where the key idea is to play audio samples to the user, each shaped by a different filter $h_i$, and query the user for their satisfaction scores $f(h_i)$. A family of ``surrogate" functions is then designed to fit these scores and the optimization method gradually refines these functions to arrive at the filter $\hat{h}^*$ that maximizes satisfaction. In certain applications, we observe that a second type of querying is possible where users can tell us the individual elements $h^*[j]$ of the optimal filter $h^*$. Consider an analogy from cooking where the goal is to cook a recipe that maximizes user satisfaction. A user can be asked to score various cooked recipes (e.g., tofu fried rice) or to score individual ingredients (say, salt, sugar, rice, chicken, etc.). Given a budget of $B$ queries, where a query can be of either type, our goal is to find the recipe that will maximize this user's satisfaction. Our proposal builds on Sparse Gaussian Process Regression (GPR) and shows how a hybrid approach can outperform any one type of querying. Our results are validated through simulations and real world experiments, where volunteers gave feedback on music/speech audio and were able to achieve high satisfaction levels. We believe this idea of hybrid querying opens new problems in black-box optimization and solutions can benefit other applications beyond audio personalization.
>
---
#### [new 008] Task-Specific Audio Coding for Machines: Machine-Learned Latent Features Are Codes for That Machine
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频编码任务，解决机器高效音频压缩问题。通过任务特定损失和残差向量量化，实现低比特率下的高性能音频编码。**

- **链接: [http://arxiv.org/pdf/2507.12701v1](http://arxiv.org/pdf/2507.12701v1)**

> **作者:** Anastasia Kuznetsova; Inseon Jang; Wootaek Lim; Minje Kim
>
> **摘要:** Neural audio codecs, leveraging quantization algorithms, have significantly impacted various speech/audio tasks. While high-fidelity reconstruction is paramount for human perception, audio coding for machines (ACoM) prioritizes efficient compression and downstream task performance, disregarding perceptual nuances. This work introduces an efficient ACoM method that can compress and quantize any chosen intermediate feature representation of an already trained speech/audio downstream model. Our approach employs task-specific loss guidance alongside residual vector quantization (RVQ) losses, providing ultra-low bitrates (i.e., less than 200 bps) with a minimal loss of the downstream model performance. The resulting tokenizer is adaptable to various bitrates and model sizes for flexible deployment. Evaluated on automatic speech recognition and audio classification, our method demonstrates its efficacy and potential for broader task and architectural applicability through appropriate regularization.
>
---
#### [new 009] SHIELD: A Secure and Highly Enhanced Integrated Learning for Robust Deepfake Detection against Adversarial Attacks
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于语音防伪任务，旨在解决深度伪造音频检测易受对抗攻击的问题，提出SHIELD方法提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13170v1](http://arxiv.org/pdf/2507.13170v1)**

> **作者:** Kutub Uddin; Awais Khan; Muhammad Umar Farooq; Khalid Malik
>
> **摘要:** Audio plays a crucial role in applications like speaker verification, voice-enabled smart devices, and audio conferencing. However, audio manipulations, such as deepfakes, pose significant risks by enabling the spread of misinformation. Our empirical analysis reveals that existing methods for detecting deepfake audio are often vulnerable to anti-forensic (AF) attacks, particularly those attacked using generative adversarial networks. In this article, we propose a novel collaborative learning method called SHIELD to defend against generative AF attacks. To expose AF signatures, we integrate an auxiliary generative model, called the defense (DF) generative model, which facilitates collaborative learning by combining input and output. Furthermore, we design a triplet model to capture correlations for real and AF attacked audios with real-generated and attacked-generated audios using auxiliary generative models. The proposed SHIELD strengthens the defense against generative AF attacks and achieves robust performance across various generative models. The proposed AF significantly reduces the average detection accuracy from 95.49% to 59.77% for ASVspoof2019, from 99.44% to 38.45% for In-the-Wild, and from 98.41% to 51.18% for HalfTruth for three different generative models. The proposed SHIELD mechanism is robust against AF attacks and achieves an average accuracy of 98.13%, 98.58%, and 99.57% in match, and 98.78%, 98.62%, and 98.85% in mismatch settings for the ASVspoof2019, In-the-Wild, and HalfTruth datasets, respectively.
>
---
#### [new 010] Multi-Class-Token Transformer for Multitask Self-supervised Music Information Retrieval
- **分类: cs.SD**

- **简介: 该论文属于音乐信息检索任务，旨在解决自监督学习中对比学习与等变学习的局限性。通过引入多类标记的Transformer架构，同时进行两种预训练任务，提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.12996v1](http://arxiv.org/pdf/2507.12996v1)**

> **作者:** Yuexuan Kong; Vincent Lostanlen; Romain Hennequin; Mathieu Lagrange; Gabriel Meseguer-Brocal
>
> **摘要:** Contrastive learning and equivariant learning are effective methods for self-supervised learning (SSL) for audio content analysis. Yet, their application to music information retrieval (MIR) faces a dilemma: the former is more effective on tagging (e.g., instrument recognition) but less effective on structured prediction (e.g., tonality estimation); The latter can match supervised methods on the specific task it is designed for, but it does not generalize well to other tasks. In this article, we adopt a best-of-both-worlds approach by training a deep neural network on both kinds of pretext tasks at once. The proposed new architecture is a Vision Transformer with 1-D spectrogram patches (ViT-1D), equipped with two class tokens, which are specialized to different self-supervised pretext tasks but optimized through the same model: hence the qualification of self-supervised multi-class-token multitask (MT2). The former class token optimizes cross-power spectral density (CPSD) for equivariant learning over the circle of fifths, while the latter optimizes normalized temperature-scaled cross-entropy (NT-Xent) for contrastive learning. MT2 combines the strengths of both pretext tasks and outperforms consistently both single-class-token ViT-1D models trained with either contrastive or equivariant learning. Averaging the two class tokens further improves performance on several tasks, highlighting the complementary nature of the representations learned by each class token. Furthermore, using the same single-linear-layer probing method on the features of last layer, MT2 outperforms MERT on all tasks except for beat tracking; achieving this with 18x fewer parameters thanks to its multitasking capabilities. Our SSL benchmark demonstrates the versatility of our multi-class-token multitask learning approach for MIR applications.
>
---
#### [new 011] Voxtral
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文介绍Voxtral系列多模态音频聊天模型，解决音频与文本理解任务，提升语音识别与对话能力，支持长音频处理和本地运行。**

- **链接: [http://arxiv.org/pdf/2507.13264v1](http://arxiv.org/pdf/2507.13264v1)**

> **作者:** Alexander H. Liu; Andy Ehrenberg; Andy Lo; Clément Denoix; Corentin Barreau; Guillaume Lample; Jean-Malo Delignon; Khyathi Raghavi Chandu; Patrick von Platen; Pavankumar Reddy Muddireddy; Sanchit Gandhi; Soham Ghosh; Srijan Mishra; Thomas Foubert; Abhinav Rastogi; Adam Yang; Albert Q. Jiang; Alexandre Sablayrolles; Amélie Héliou; Amélie Martin; Anmol Agarwal; Antoine Roux; Arthur Darcet; Arthur Mensch; Baptiste Bout; Baptiste Rozière; Baudouin De Monicault; Chris Bamford; Christian Wallenwein; Christophe Renaudin; Clémence Lanfranchi; Darius Dabert; Devendra Singh Chaplot; Devon Mizelle; Diego de las Casas; Elliot Chane-Sane; Emilien Fugier; Emma Bou Hanna; Gabrielle Berrada; Gauthier Delerce; Gauthier Guinet; Georgii Novikov; Guillaume Martin; Himanshu Jaju; Jan Ludziejewski; Jason Rute; Jean-Hadrien Chabran; Jessica Chudnovsky; Joachim Studnia; Joep Barmentlo; Jonas Amar; Josselin Somerville Roberts; Julien Denize; Karan Saxena; Karmesh Yadav; Kartik Khandelwal; Kush Jain; Lélio Renard Lavaud; Léonard Blier; Lingxiao Zhao; Louis Martin; Lucile Saulnier; Luyu Gao; Marie Pellat; Mathilde Guillaumin; Mathis Felardos; Matthieu Dinot; Maxime Darrin; Maximilian Augustin; Mickaël Seznec; Neha Gupta; Nikhil Raghuraman; Olivier Duchenne; Patricia Wang; Patryk Saffer; Paul Jacob; Paul Wambergue; Paula Kurylowicz; Philomène Chagniot; Pierre Stock; Pravesh Agrawal; Rémi Delacourt; Romain Sauvestre; Roman Soletskyi; Sagar Vaze; Sandeep Subramanian; Saurabh Garg; Shashwat Dalal; Siddharth Gandhi; Sumukh Aithal; Szymon Antoniak; Teven Le Scao; Thibault Schueller; Thibaut Lavril; Thomas Robert; Thomas Wang; Timothée Lacroix; Tom Bewley; Valeriia Nemychnikova; Victor Paltz; Virgile Richard; Wen-Ding Li; William Marshall; Xuanyu Zhang; Yihan Wan; Yunhao Tang
>
> **备注:** 17 pages
>
> **摘要:** We present Voxtral Mini and Voxtral Small, two multimodal audio chat models. Voxtral is trained to comprehend both spoken audio and text documents, achieving state-of-the-art performance across a diverse range of audio benchmarks, while preserving strong text capabilities. Voxtral Small outperforms a number of closed-source models, while being small enough to run locally. A 32K context window enables the model to handle audio files up to 40 minutes in duration and long multi-turn conversations. We also contribute three benchmarks for evaluating speech understanding models on knowledge and trivia. Both Voxtral models are released under Apache 2.0 license.
>
---
#### [new 012] From Neck to Head: Bio-Impedance Sensing for Head Pose Estimation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于头姿估计任务，旨在无需视线的头姿追踪。通过生物阻抗传感和深度学习方法，实现高精度头姿估计。**

- **链接: [http://arxiv.org/pdf/2507.12884v1](http://arxiv.org/pdf/2507.12884v1)**

> **作者:** Mengxi Liu; Lala Shakti Swarup Ray; Sizhen Bian; Ko Watanabe; Ankur Bhatt; Joanna Sorysz; Russel Torah; Bo Zhou; Paul Lukowicz
>
> **摘要:** We present NeckSense, a novel wearable system for head pose tracking that leverages multi-channel bio-impedance sensing with soft, dry electrodes embedded in a lightweight, necklace-style form factor. NeckSense captures dynamic changes in tissue impedance around the neck, which are modulated by head rotations and subtle muscle activations. To robustly estimate head pose, we propose a deep learning framework that integrates anatomical priors, including joint constraints and natural head rotation ranges, into the loss function design. We validate NeckSense on 7 participants using the current SOTA pose estimation model as ground truth. Our system achieves a mean per-vertex error of 25.9 mm across various head movements with a leave-one-person-out cross-validation method, demonstrating that a compact, line-of-sight-free bio-impedance wearable can deliver head-tracking performance comparable to SOTA vision-based methods.
>
---
#### [new 013] AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音评估任务，旨在解决传统评估系统设计复杂和与人类偏好相关性低的问题。通过构建AudioJudge模型，实现统一、高效的语音质量与偏好评估。**

- **链接: [http://arxiv.org/pdf/2507.12705v1](http://arxiv.org/pdf/2507.12705v1)**

> **作者:** Potsawee Manakul; Woody Haosheng Gan; Michael J. Ryan; Ali Sartaz Khan; Warit Sirichotedumrong; Kunat Pipatanakul; William Held; Diyi Yang
>
> **摘要:** Current speech evaluation suffers from two critical limitations: the need and difficulty of designing specialized systems targeting individual audio characteristics, and poor correlation between automatic evaluation methods and human preferences. This work presents a systematic study of Large Audio Model (LAM) as a Judge, AudioJudge, investigating whether it can provide a unified evaluation framework that addresses both challenges. We systematically explore AudioJudge across audio characteristic detection tasks, including pronunciation, speaking rate, speaker identification and speech quality, and system-level human preference simulation for automated benchmarking. We investigate different prompt engineering strategies, finding that audio concatenation combined with in-context learning significantly improves performance across both audio characteristic detection and human preference simulation tasks. We further introduce a multi-aspect ensemble AudioJudge to enable general-purpose multi-aspect audio evaluation. This method decomposes speech assessment into specialized judges for lexical content, speech quality, and paralinguistic features, achieving up to 0.91 Spearman correlation with human preferences on our system ranking benchmark. Robustness analysis reveals that while LAMs maintain strong performance under acoustic noise, they exhibit significant verbosity and positional biases that require careful mitigation.
>
---
#### [new 014] Keep the beat going: Automatic drum transcription with momentum
- **分类: math.NA; cs.NA; cs.SD; eess.AS**

- **简介: 该论文属于自动鼓音轨转录任务，旨在提高转录准确性。通过非负矩阵分解方法，比较了两种优化算法，发现带动量的梯度下降效果更优。**

- **链接: [http://arxiv.org/pdf/2507.12596v1](http://arxiv.org/pdf/2507.12596v1)**

> **作者:** Alisha L. Foster; Robert J. Webber
>
> **摘要:** A simple, interpretable way to perform automatic drum transcription is by factoring the magnitude spectrogram of a recorded musical piece using a partially fixed nonnegative matrix factorization. There are two natural ways to optimize the nonnegative matrix factorization, including a multiplicative update rule and projected gradient descent with momentum. The methods differ in their empirical accuracies and theoretical convergence guarantees. This paper summarizes the methods and their time complexities, and it applies the methods to the ENST-Drums data set and an original recording from the author's band, evaluating the empirical accuracy with respect to ground-truth drum annotations. The results indicate that projected gradient descent with momentum leads to higher accuracy for a fixed runtime, and it satisfies stronger convergence guarantees.
>
---
#### [new 015] AVFSNet: Audio-Visual Speech Separation for Flexible Number of Speakers with Multi-Scale and Multi-Task Learning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音分离任务，解决混合信号中未知说话人数的分离问题。提出AVFSNet模型，结合多尺度编码和视觉信息，实现多说话人分离与计数。**

- **链接: [http://arxiv.org/pdf/2507.12972v1](http://arxiv.org/pdf/2507.12972v1)**

> **作者:** Daning Zhang; Ying Wei
>
> **摘要:** Separating target speech from mixed signals containing flexible speaker quantities presents a challenging task. While existing methods demonstrate strong separation performance and noise robustness, they predominantly assume prior knowledge of speaker counts in mixtures. The limited research addressing unknown speaker quantity scenarios exhibits significantly constrained generalization capabilities in real acoustic environments. To overcome these challenges, this paper proposes AVFSNet -- an audio-visual speech separation model integrating multi-scale encoding and parallel architecture -- jointly optimized for speaker counting and multi-speaker separation tasks. The model independently separates each speaker in parallel while enhancing environmental noise adaptability through visual information integration. Comprehensive experimental evaluations demonstrate that AVFSNet achieves state-of-the-art results across multiple evaluation metrics and delivers outstanding performance on diverse datasets.
>
---
#### [new 016] NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations for Text-to-Speech
- **分类: cs.LG; cs.SD**

- **简介: 该论文属于语音合成任务，解决非语言声音数据不足的问题。构建了包含10类非语言发声和8种情感标注的17小时数据集NVTTS，提升TTS模型表现。**

- **链接: [http://arxiv.org/pdf/2507.13155v1](http://arxiv.org/pdf/2507.13155v1)**

> **作者:** Maksim Borisov; Egor Spirin; Daria Diatlova
>
> **摘要:** Current expressive speech synthesis models are constrained by the limited availability of open-source datasets containing diverse nonverbal vocalizations (NVs). In this work, we introduce NonverbalTTS (NVTTS), a 17-hour open-access dataset annotated with 10 types of NVs (e.g., laughter, coughs) and 8 emotional categories. The dataset is derived from popular sources, VoxCeleb and Expresso, using automated detection followed by human validation. We propose a comprehensive pipeline that integrates automatic speech recognition (ASR), NV tagging, emotion classification, and a fusion algorithm to merge transcriptions from multiple annotators. Fine-tuning open-source text-to-speech (TTS) models on the NVTTS dataset achieves parity with closed-source systems such as CosyVoice2, as measured by both human evaluation and automatic metrics, including speaker similarity and NV fidelity. By releasing NVTTS and its accompanying annotation guidelines, we address a key bottleneck in expressive TTS research. The dataset is available at https://huggingface.co/datasets/deepvk/NonverbalTTS.
>
---
#### [new 017] Large Language Models' Internal Perception of Symbolic Music
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究LLM在符号音乐领域的表现，旨在探索其隐式建模能力。通过文本生成音乐数据并进行分类与补全任务，评估其生成效果与局限性。**

- **链接: [http://arxiv.org/pdf/2507.12808v1](http://arxiv.org/pdf/2507.12808v1)**

> **作者:** Andrew Shin; Kunitake Kaneko
>
> **摘要:** Large language models (LLMs) excel at modeling relationships between strings in natural language and have shown promise in extending to other symbolic domains like coding or mathematics. However, the extent to which they implicitly model symbolic music remains underexplored. This paper investigates how LLMs represent musical concepts by generating symbolic music data from textual prompts describing combinations of genres and styles, and evaluating their utility through recognition and generation tasks. We produce a dataset of LLM-generated MIDI files without relying on explicit musical training. We then train neural networks entirely on this LLM-generated MIDI dataset and perform genre and style classification as well as melody completion, benchmarking their performance against established models. Our results demonstrate that LLMs can infer rudimentary musical structures and temporal relationships from text, highlighting both their potential to implicitly encode musical patterns and their limitations due to a lack of explicit musical context, shedding light on their generative capabilities for symbolic music.
>
---
#### [new 018] DiffRhythm+: Controllable and Flexible Full-Length Song Generation with Preference Optimization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音乐生成任务，解决全曲生成中的数据不平衡、控制不足和质量不一致问题。通过扩展数据集和引入多模态风格控制，提升生成歌曲的自然度与多样性。**

- **链接: [http://arxiv.org/pdf/2507.12890v1](http://arxiv.org/pdf/2507.12890v1)**

> **作者:** Huakang Chen; Yuepeng Jiang; Guobin Ma; Chunbo Hao; Shuai Wang; Jixun Yao; Ziqian Ning; Meng Meng; Jian Luan; Lei Xie
>
> **摘要:** Songs, as a central form of musical art, exemplify the richness of human intelligence and creativity. While recent advances in generative modeling have enabled notable progress in long-form song generation, current systems for full-length song synthesis still face major challenges, including data imbalance, insufficient controllability, and inconsistent musical quality. DiffRhythm, a pioneering diffusion-based model, advanced the field by generating full-length songs with expressive vocals and accompaniment. However, its performance was constrained by an unbalanced model training dataset and limited controllability over musical style, resulting in noticeable quality disparities and restricted creative flexibility. To address these limitations, we propose DiffRhythm+, an enhanced diffusion-based framework for controllable and flexible full-length song generation. DiffRhythm+ leverages a substantially expanded and balanced training dataset to mitigate issues such as repetition and omission of lyrics, while also fostering the emergence of richer musical skills and expressiveness. The framework introduces a multi-modal style conditioning strategy, enabling users to precisely specify musical styles through both descriptive text and reference audio, thereby significantly enhancing creative control and diversity. We further introduce direct performance optimization aligned with user preferences, guiding the model toward consistently preferred outputs across evaluation metrics. Extensive experiments demonstrate that DiffRhythm+ achieves significant improvements in naturalness, arrangement complexity, and listener satisfaction over previous systems.
>
---
#### [new 019] UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets
- **分类: eess.AS; cs.AI; cs.CL; cs.MM; cs.SD**

- **简介: 该论文属于语音语言理解任务，解决多任务模型分离导致的系统复杂和交互受限问题，提出UniSLU统一框架，融合ASR、NER和SA任务。**

- **链接: [http://arxiv.org/pdf/2507.12951v1](http://arxiv.org/pdf/2507.12951v1)**

> **作者:** Zhichao Sheng; Shilin Zhou; Chen Gong; Zhenghua Li
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Spoken Language Understanding (SLU) plays a crucial role in speech-centric multimedia applications, enabling machines to comprehend spoken language in scenarios such as meetings, interviews, and customer service interactions. SLU encompasses multiple tasks, including Automatic Speech Recognition (ASR), spoken Named Entity Recognition (NER), and spoken Sentiment Analysis (SA). However, existing methods often rely on separate model architectures for individual tasks such as spoken NER and SA, which increases system complexity, limits cross-task interaction, and fails to fully exploit heterogeneous datasets available across tasks. To address these limitations, we propose UniSLU, a unified framework that jointly models multiple SLU tasks within a single architecture. Specifically, we propose a unified representation for diverse SLU tasks, enabling full utilization of heterogeneous datasets across multiple tasks. Built upon this representation, we propose a unified generative method that jointly models ASR, spoken NER, and SA tasks, enhancing task interactions and enabling seamless integration with large language models to harness their powerful generative capabilities. Extensive experiments on public SLU datasets demonstrate the effectiveness of our approach, achieving superior SLU performance compared to several benchmark methods, making it well-suited for real-world speech-based multimedia scenarios. We will release all code and models at github to facilitate future research.
>
---
## 更新

#### [replaced 001] Can Large Language Models Predict Audio Effects Parameters from Natural Language?
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20770v2](http://arxiv.org/pdf/2505.20770v2)**

> **作者:** Seungheon Doh; Junghyun Koo; Marco A. Martínez-Ramírez; Wei-Hsiang Liao; Juhan Nam; Yuki Mitsufuji
>
> **备注:** Accepted for publication at The IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)
>
> **摘要:** In music production, manipulating audio effects (Fx) parameters through natural language has the potential to reduce technical barriers for non-experts. We present LLM2Fx, a framework leveraging Large Language Models (LLMs) to predict Fx parameters directly from textual descriptions without requiring task-specific training or fine-tuning. Our approach address the text-to-effect parameter prediction (Text2Fx) task by mapping natural language descriptions to the corresponding Fx parameters for equalization and reverberation. We demonstrate that LLMs can generate Fx parameters in a zero-shot manner that elucidates the relationship between timbre semantics and audio effects in music production. To enhance performance, we introduce three types of in-context examples: audio Digital Signal Processing (DSP) features, DSP function code, and few-shot examples. Our results demonstrate that LLM-based Fx parameter generation outperforms previous optimization approaches, offering competitive performance in translating natural language descriptions to appropriate Fx settings. Furthermore, LLMs can serve as text-driven interfaces for audio production, paving the way for more intuitive and accessible music production tools.
>
---
#### [replaced 002] Token Communications: A Large Model-Driven Framework for Cross-modal Context-aware Semantic Communications
- **分类: cs.MM; cs.CV; cs.IT; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.12096v4](http://arxiv.org/pdf/2502.12096v4)**

> **作者:** Li Qiao; Mahdi Boloursaz Mashhadi; Zhen Gao; Rahim Tafazolli; Mehdi Bennis; Dusit Niyato
>
> **备注:** Accepted at IEEE Wireless Communications Magazine
>
> **摘要:** In this paper, we introduce token communications (TokCom), a large model-driven framework to leverage cross-modal context information in generative semantic communications (GenSC). TokCom is a new paradigm, motivated by the recent success of generative foundation models and multimodal large language models (GFM/MLLMs), where the communication units are tokens, enabling efficient transformer-based token processing at the transmitter and receiver. In this paper, we introduce the potential opportunities and challenges of leveraging context in GenSC, explore how to integrate GFM/MLLMs-based token processing into semantic communication systems to leverage cross-modal context effectively at affordable complexity, present the key principles for efficient TokCom at various layers in future wireless networks. In a typical image semantic communication setup, we demonstrate a significant improvement of the bandwidth efficiency, achieved by TokCom by leveraging the context information among tokens. Finally, the potential research directions are identified to facilitate adoption of TokCom in future wireless networks.
>
---
#### [replaced 003] Learning Separated Representations for Instrument-based Music Similarity
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.17281v2](http://arxiv.org/pdf/2503.17281v2)**

> **作者:** Yuka Hashizume; Li Li; Atsushi Miyashita; Tomoki Toda
>
> **备注:** arXiv admin note: text overlap with arXiv:2404.06682
>
> **摘要:** A flexible recommendation and retrieval system requires music similarity in terms of multiple partial elements of musical pieces to allow users to select the element they want to focus on. A method for music similarity learning using multiple networks with individual instrumental signals is effective but faces the problem that using each clean instrumental signal as a query is impractical for retrieval systems and using separated instrumental signals reduces accuracy owing to artifacts. In this paper, we present instrumental-part-based music similarity learning with a single network that takes mixed signals as input instead of individual instrumental signals. Specifically, we designed a single similarity embedding space with separated subspaces for each instrument, extracted by Conditional Similarity Networks, which are trained using the triplet loss with masks. Experimental results showed that (1) the proposed method can obtain more accurate embedding representation than using individual networks using separated signals as input in the evaluation of an instrument that had low accuracy, (2) each sub-embedding space can hold the characteristics of the corresponding instrument, and (3) the selection of similar musical pieces focusing on each instrumental sound by the proposed method can obtain human acceptance, especially when focusing on timbre.
>
---
#### [replaced 004] Benchmarking Sub-Genre Classification For Mainstage Dance Music
- **分类: cs.SD; cs.AI; cs.MM; H.5.5; I.2.1**

- **链接: [http://arxiv.org/pdf/2409.06690v2](http://arxiv.org/pdf/2409.06690v2)**

> **作者:** Hongzhi Shu; Xinglin Li; Hongyu Jiang; Minghao Fu; Xinyu Li
>
> **备注:** WASPAA 2025
>
> **摘要:** Music classification, a cornerstone of music information retrieval, supports a wide array of applications. To address the lack of comprehensive datasets and effective methods for sub-genre classification in mainstage dance music, we introduce a novel benchmark featuring a new dataset and baseline. Our dataset expands the scope of sub-genres to reflect the diversity of recent mainstage live sets performed by leading DJs at global music festivals, capturing the vibrant and rapidly evolving electronic dance music (EDM) scene that engages millions of fans worldwide. We employ a continuous soft labeling approach to accommodate tracks blending multiple sub-genres, preserving their inherent complexity. Experiments demonstrate that even state-of-the-art multimodal large language models (MLLMs) struggle with this task, while our specialized baseline models achieve high accuracy. This benchmark supports applications such as music recommendation, DJ set curation, and interactive multimedia systems, with video demos provided. Our code and data are all open-sourced at https://github.com/Gariscat/housex-v2.git}{https://github.com/Gariscat/housex-v2.git.
>
---
#### [replaced 005] A lightweight and robust method for blind wideband-to-fullband extension of speech
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2412.11392v3](http://arxiv.org/pdf/2412.11392v3)**

> **作者:** Jan Büthe; Jean-Marc Valin
>
> **备注:** WASPAA 2025, 5 pages
>
> **摘要:** Reducing the bandwidth of speech is common practice in resource constrained environments like low-bandwidth speech transmission or low-complexity vocoding. We propose a lightweight and robust method for extending the bandwidth of wideband speech signals that is inspired by classical methods developed in the speech coding context. The resulting model has just ~370K parameters and a complexity of ~140 MFLOPS (or ~70 MMACS). With a frame size of 10 ms and a lookahead of only 0.27 ms, the model is well-suited for use with common wideband speech codecs. We evaluate the model's robustness by pairing it with the Opus SILK speech codec (1.5 release) and verify in a P.808 DCR listening test that it significantly improves quality from 6 to 12 kb/s. We also demonstrate that Opus 1.5 together with the proposed bandwidth extension at 9 kb/s meets the quality of 3GPP EVS at 9.6 kb/s and that of Opus 1.4 at 18 kb/s showing that the blind bandwidth extension can meet the quality of classical guided bandwidth extensions thus providing a way for backward-compatible quality improvement.
>
---
#### [replaced 006] Radif Corpus: A Symbolic Dataset for Non-Metric Iranian Classical Music
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.10456v2](http://arxiv.org/pdf/2507.10456v2)**

> **作者:** Maziar Kanani; Sean O Leary; James McDermott
>
> **摘要:** Non-metric music forms the core of the repertoire in Iranian classical music. Dastgahi music serves as the underlying theoretical system for both Iranian art music and certain folk traditions. At the heart of Iranian classical music lies the radif, a foundational repertoire that organizes melodic material central to performance and pedagogy. In this study, we introduce the first digital corpus representing the complete non-metrical radif repertoire, covering all 13 existing components of this repertoire. We provide MIDI files (about 281 minutes in total) and data spreadsheets describing notes, note durations, intervals, and hierarchical structures for 228 pieces of music. We faithfully represent the tonality including quarter-tones, and the non-metric aspect. Furthermore, we provide supporting basic statistics, and measures of complexity and similarity over the corpus. Our corpus provides a platform for computational studies of Iranian classical music. Researchers might employ it in studying melodic patterns, investigating improvisational styles, or for other tasks in music information retrieval, music theory, and computational (ethno)musicology.
>
---
#### [replaced 007] Stereo Reproduction in the Presence of Sample Rate Offsets
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.05402v2](http://arxiv.org/pdf/2507.05402v2)**

> **作者:** Srikanth Korse; Andreas Walther; Emanuel A. P. Habets
>
> **备注:** Accepted to IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** One of the main challenges in synchronizing wirelessly connected loudspeakers for spatial audio reproduction is clock skew. Clock skew arises from sample rate offsets ( SROs) between the loudspeakers, caused by the use of independent device clocks. While network-based protocols like Precision Time Protocol (PTP) and Network Time Protocol (NTP) are explored, the impact of SROs on spatial audio reproduction and its perceptual consequences remains underexplored. We propose an audio-domain SRO compensation method using spatial filtering to isolate loudspeaker contributions. These filtered signals, along with the original playback signal, are used to estimate the SROs, and their influence is compensated for prior to spatial audio reproduction. We evaluate the effect of the compensation method in a subjective listening test. The results of these tests as well as objective metrics demonstrate that the proposed method mitigates the perceptual degradation introduced by SROs by preserving the spatial cues.
>
---
#### [replaced 008] Speech-Forensics: Towards Comprehensive Synthetic Speech Dataset Establishment and Analysis
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.09032v3](http://arxiv.org/pdf/2412.09032v3)**

> **作者:** Zhoulin Ji; Chenhao Lin; Hang Wang; Chao Shen
>
> **备注:** IJCAI 2024
>
> **摘要:** Detecting synthetic from real speech is increasingly crucial due to the risks of misinformation and identity impersonation. While various datasets for synthetic speech analysis have been developed, they often focus on specific areas, limiting their utility for comprehensive research. To fill this gap, we propose the Speech-Forensics dataset by extensively covering authentic, synthetic, and partially forged speech samples that include multiple segments synthesized by different high-quality algorithms. Moreover, we propose a TEmporal Speech LocalizaTion network, called TEST, aiming at simultaneously performing authenticity detection, multiple fake segments localization, and synthesis algorithms recognition, without any complex post-processing. TEST effectively integrates LSTM and Transformer to extract more powerful temporal speech representations and utilizes dense prediction on multi-scale pyramid features to estimate the synthetic spans. Our model achieves an average mAP of 83.55% and an EER of 5.25% at the utterance level. At the segment level, it attains an EER of 1.07% and a 92.19% F1 score. These results highlight the model's robust capability for a comprehensive analysis of synthetic speech, offering a promising avenue for future research and practical applications in this field.
>
---
