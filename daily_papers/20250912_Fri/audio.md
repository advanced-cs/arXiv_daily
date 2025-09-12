# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] DeCodec: Rethinking Audio Codecs as Universal Disentangled Representation Learners
- **分类: cs.SD**

- **简介: 论文提出DeCodec，一种解耦音频表示的通用编码器，解决混合音频中语音与背景声分离的问题。通过子空间正交投影和表示交换训练，实现语音与背景声的独立量化与灵活特征选择，提升语音增强、语音识别等任务性能。**

- **链接: [http://arxiv.org/pdf/2509.09201v1](http://arxiv.org/pdf/2509.09201v1)**

> **作者:** Xiaoxue Luo; Jinwei Huang; Runyan Yang; Yingying Gao; Junlan Feng; Chao Deng; Shilei Zhang
>
> **摘要:** Universal audio codecs learn entangled representations across audio types, whereas some specific codecs offer decoupled representations but are limited to speech. Real-world audio, however, often contains mixed speech and background sounds, and downstream tasks require selective access to these components. Therefore, we rethink the audio codec as a universal disentangled representation learner to enable controllable feature selection across different audio tasks. To this end, we introduce DeCodec, a novel neural codec that learns to decouple audio representations into orthogonal subspaces dedicated to speech and background sound, and within speech, representations are further decomposed into semantic and paralinguistic components. This hierarchical disentanglement allows flexible feature selection, making DeCodec a universal front-end for multiple audio applications. Technically, built upon a codec framework, DeCodec incorporates two key innovations: a subspace orthogonal projection module that factorizes the input into two decoupled orthogonal subspaces, and a representation swap training procedure that ensures these two subspaces are correlate to the speech and background sound, respectively. These allows parallel RVQs to quantize speech and background sound components independently. Furthermore, we employ semantic guidance to the speech RVQ to achieve semantic and paralinguistic decomposition. Experimental results show that DeCodec maintains advanced signal reconstruction while enabling new capabilities: superior speech enhancement and effective one-shot voice conversion on noisy speech via representation recombination, improved ASR robustness through clean semantic features, and controllable background sound preservation/suppression in TTS. Demo Page: https://luo404.github.io/DeCodecV2/
>
---
#### [new 002] Finite Scalar Quantization Enables Redundant and Transmission-Robust Neural Audio Compression at Low Bit-rates
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出基于有限标量量化（FSQ）的神经音频编码器NeuCodec，用于解决低比特率下鲁棒的音频压缩问题。通过实验验证FSQ在编码冗余性和抗传输噪声方面的优势，替代传统残差向量量化（RVQ）。**

- **链接: [http://arxiv.org/pdf/2509.09550v1](http://arxiv.org/pdf/2509.09550v1)**

> **作者:** Harry Julia; Rachel Beeson; Lohith Konathala; Johanna Ulin; Jiameng Gao
>
> **摘要:** Neural Audio Codecs (NACs) have become increasingly adopted in speech processing tasks due to their excellent rate-distortion performance and compatibility with Large Language Models (LLMs) as discrete feature representations for audio generation. While most existing codecs rely on Residual Vector Quantization (RVQ), Finite Scalar Quantization (FSQ) has recently emerged as a compelling alternative that simplifies training and natively supports single codebooks. We introduce NeuCodec, an FSQ-based NAC, and show that FSQ encodes baked-in redundancy which produces an encoding which is robust when transmitted through noisy channels. First, through an encoder distillation experiment, we show that two different encoders can learn to encode identical audio into vastly different code sequences whilst maintaining comparable reconstruction quality with the same quantizer and decoder. Second, we demonstrate that FSQ has vastly superior bit-level perturbation robustness by comparing the performance of RVQ and FSQ codecs when simulating the transmission of code sequences through a noisy channel.
>
---
#### [new 003] Adaptive Knowledge Distillation using a Device-Aware Teacher for Low-Complexity Acoustic Scene Classification
- **分类: cs.SD; cs.AI**

- **简介: 论文针对DCASE 2025挑战赛任务1，解决低复杂度设备鲁棒声学场景分类问题。提出基于知识蒸馏的框架，使用设备感知教师模型提升泛化能力，并利用测试时设备标签进行微调，显著提升未见设备上的性能。**

- **链接: [http://arxiv.org/pdf/2509.09262v1](http://arxiv.org/pdf/2509.09262v1)**

> **作者:** Seung Gyu Jeong; Seong Eun Kim
>
> **摘要:** In this technical report, we describe our submission for Task 1, Low-Complexity Device-Robust Acoustic Scene Classification, of the DCASE 2025 Challenge. Our work tackles the dual challenges of strict complexity constraints and robust generalization to both seen and unseen devices, while also leveraging the new rule allowing the use of device labels at test time. Our proposed system is based on a knowledge distillation framework where an efficient CP-MobileNet student learns from a compact, specialized two-teacher ensemble. This ensemble combines a baseline PaSST teacher, trained with standard cross-entropy, and a 'generalization expert' teacher. This expert is trained using our novel Device-Aware Feature Alignment (DAFA) loss, adapted from prior work, which explicitly structures the feature space for device robustness. To capitalize on the availability of test-time device labels, the distilled student model then undergoes a final device-specific fine-tuning stage. Our proposed system achieves a final accuracy of 57.93\% on the development set, demonstrating a significant improvement over the official baseline, particularly on unseen devices.
>
---
#### [new 004] DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech
- **分类: cs.SD; cs.CL; cs.CV**

- **简介: 该论文提出DiFlow-TTS，用于零样本文本到语音合成任务，解决现有方法推理慢、重复 artifacts 问题。其采用纯离散流匹配，建模语音属性，实现高效、高质量的语音生成。**

- **链接: [http://arxiv.org/pdf/2509.09631v1](http://arxiv.org/pdf/2509.09631v1)**

> **作者:** Ngoc-Son Nguyen; Hieu-Nghia Huynh-Nguyen; Thanh V. T. Tran; Truong-Son Hy; Van Nguyen
>
> **摘要:** Zero-shot Text-to-Speech (TTS) aims to synthesize high-quality speech that mimics the voice of an unseen speaker using only a short reference sample, requiring not only speaker adaptation but also accurate modeling of prosodic attributes. Recent approaches based on language models, diffusion, and flow matching have shown promising results in zero-shot TTS, but still suffer from slow inference and repetition artifacts. Discrete codec representations have been widely adopted for speech synthesis, and recent works have begun to explore diffusion models in purely discrete settings, suggesting the potential of discrete generative modeling for speech synthesis. However, existing flow-matching methods typically embed these discrete tokens into a continuous space and apply continuous flow matching, which may not fully leverage the advantages of discrete representations. To address these challenges, we introduce DiFlow-TTS, which, to the best of our knowledge, is the first model to explore purely Discrete Flow Matching for speech synthesis. DiFlow-TTS explicitly models factorized speech attributes within a compact and unified architecture. It leverages in-context learning by conditioning on textual content, along with prosodic and acoustic attributes extracted from a reference speech, enabling effective attribute cloning in a zero-shot setting. In addition, the model employs a factorized flow prediction mechanism with distinct heads for prosody and acoustic details, allowing it to learn aspect-specific distributions. Experimental results demonstrate that DiFlow-TTS achieves promising performance in several key metrics, including naturalness, prosody, preservation of speaker style, and energy control. It also maintains a compact model size and achieves low-latency inference, generating speech up to 25.8 times faster than the latest existing baselines.
>
---
#### [new 005] Efficient Transformer-Based Piano Transcription With Sparse Attention Mechanisms
- **分类: cs.SD; cs.MM**

- **简介: 论文提出一种基于稀疏注意力机制的高效Transformer模型，用于钢琴自动转录任务。旨在解决传统Transformer因二次复杂度无法处理完整乐曲的问题，通过滑动窗口自注意力和混合跨注意力机制，降低计算成本，提升推理速度，同时保持转录性能。**

- **链接: [http://arxiv.org/pdf/2509.09318v1](http://arxiv.org/pdf/2509.09318v1)**

> **作者:** Weixing Wei; Kazuyoshi Yoshii
>
> **备注:** Accepted by APSIPA 2025
>
> **摘要:** This paper investigates automatic piano transcription based on computationally-efficient yet high-performant variants of the Transformer that can capture longer-term dependency over the whole musical piece. Recently, transformer-based sequence-to-sequence models have demonstrated excellent performance in piano transcription. These models, however, fail to deal with the whole piece at once due to the quadratic complexity of the self-attention mechanism, and music signals are thus typically processed in a sliding-window manner in practice. To overcome this limitation, we propose an efficient architecture with sparse attention mechanisms. Specifically, we introduce sliding-window self-attention mechanisms for both the encoder and decoder, and a hybrid global-local cross-attention mechanism that attends to various spans according to the MIDI token types. We also use a hierarchical pooling strategy between the encoder and decoder to further reduce computational load. Our experiments on the MAESTRO dataset showed that the proposed model achieved a significant reduction in computational cost and memory usage, accelerating inference speed, while maintaining transcription performance comparable to the full-attention baseline. This allows for training with longer audio contexts on the same hardware, demonstrating the viability of sparse attention for building efficient and high-performance piano transcription systems. The code is available at https://github.com/WX-Wei/efficient-seq2seq-piano-trans.
>
---
#### [new 006] Ultrafast Deep Learning-Based Scatter Estimation in Cone-Beam Computed Tomography
- **分类: eess.SP; cs.CV**

- **简介: 该论文提出一种基于深度学习的锥形束CT散射估计方法，通过调整网络分辨率以减少计算量和内存占用。任务是解决移动CBCT系统中因内存限制难以部署深度学习模型的问题，工作包括分析不同分辨率对性能的影响，并验证方法在真实数据上的有效性。**

- **链接: [http://arxiv.org/pdf/2509.08973v1](http://arxiv.org/pdf/2509.08973v1)**

> **作者:** Harshit Agrawal; Ari Hietanen; Simo Särkkä
>
> **摘要:** Purpose: Scatter artifacts drastically degrade the image quality of cone-beam computed tomography (CBCT) scans. Although deep learning-based methods show promise in estimating scatter from CBCT measurements, their deployment in mobile CBCT systems or edge devices is still limited due to the large memory footprint of the networks. This study addresses the issue by applying networks at varying resolutions and suggesting an optimal one, based on speed and accuracy. Methods: First, the reconstruction error in down-up sampling of CBCT scatter signal was examined at six resolutions by comparing four interpolation methods. Next, a recent state-of-the-art method was trained across five image resolutions and evaluated for the reductions in floating-point operations (FLOPs), inference times, and GPU memory requirements. Results: Reducing the input size and network parameters achieved a 78-fold reduction in FLOPs compared to the baseline method, while maintaining comarable performance in terms of mean-absolute-percentage-error (MAPE) and mean-square-error (MSE). Specifically, the MAPE decreased to 3.85% compared to 4.42%, and the MSE decreased to 1.34 \times 10^{-2} compared to 2.01 \times 10^{-2}. Inference time and GPU memory usage were reduced by factors of 16 and 12, respectively. Further experiments comparing scatter-corrected reconstructions on a large, simulated dataset and real CBCT scans from water and Sedentex CT phantoms clearly demonstrated the robustness of our method. Conclusion: This study highlights the underappreciated role of downsampling in deep learning-based scatter estimation. The substantial reduction in FLOPs and GPU memory requirements achieved by our method enables scatter correction in resource-constrained environments, such as mobile CBCT and edge devices.
>
---
#### [new 007] Bona fide Cross Testing Reveals Weak Spot in Audio Deepfake Detection Systems
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有评估方法对合成数据权重不均、真实语音多样性不足的问题。提出“真实交叉测试”框架，引入多样真实语音数据集，改进EER评估方式，提升检测系统的鲁棒性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.09204v1](http://arxiv.org/pdf/2509.09204v1)**

> **作者:** Chin Yuen Kwok; Jia Qi Yip; Zhen Qiu; Chi Hung Chi; Kwok Yan Lam
>
> **备注:** Published in Interspeech 2025
>
> **摘要:** Audio deepfake detection (ADD) models are commonly evaluated using datasets that combine multiple synthesizers, with performance reported as a single Equal Error Rate (EER). However, this approach disproportionately weights synthesizers with more samples, underrepresenting others and reducing the overall reliability of EER. Additionally, most ADD datasets lack diversity in bona fide speech, often featuring a single environment and speech style (e.g., clean read speech), limiting their ability to simulate real-world conditions. To address these challenges, we propose bona fide cross-testing, a novel evaluation framework that incorporates diverse bona fide datasets and aggregates EERs for more balanced assessments. Our approach improves robustness and interpretability compared to traditional evaluation methods. We benchmark over 150 synthesizers across nine bona fide speech types and release a new dataset to facilitate further research at https://github.com/cyaaronk/audio_deepfake_eval.
>
---
#### [new 008] In situ estimation of the acoustic surface impedance using simulation-based inference
- **分类: cs.SD; physics.data-an**

- **简介: 该论文提出一种基于模拟推理的贝叶斯框架，用于现场估计声学表面阻抗。通过神经网络映射模拟数据到参数后验分布，解决了传统测量方法在真实场景中精度不足的问题，实现了复杂几何体中阻抗的高效、准确估计。**

- **链接: [http://arxiv.org/pdf/2509.08873v1](http://arxiv.org/pdf/2509.08873v1)**

> **作者:** Jonas M. Schmid; Johannes D. Schmid; Martin Eser; Steffen Marburg
>
> **摘要:** Accurate acoustic simulations of enclosed spaces require precise boundary conditions, typically expressed through surface impedances for wave-based methods. Conventional measurement techniques often rely on simplifying assumptions about the sound field and mounting conditions, limiting their validity for real-world scenarios. To overcome these limitations, this study introduces a Bayesian framework for the in situ estimation of frequency-dependent acoustic surface impedances from sparse interior sound pressure measurements. The approach employs simulation-based inference, which leverages the expressiveness of modern neural network architectures to directly map simulated data to posterior distributions of model parameters, bypassing conventional sampling-based Bayesian approaches and offering advantages for high-dimensional inference problems. Impedance behavior is modeled using a damped oscillator model extended with a fractional calculus term. The framework is verified on a finite element model of a cuboid room and further tested with impedance tube measurements used as reference, achieving robust and accurate estimation of all six individual impedances. Application to a numerical car cabin model further demonstrates reliable uncertainty quantification and high predictive accuracy even for complex-shaped geometries. Posterior predictive checks and coverage diagnostics confirm well-calibrated inference, highlighting the method's potential for generalizable, efficient, and physically consistent characterization of acoustic boundary conditions in real-world interior environments.
>
---
#### [new 009] MoLEx: Mixture of LoRA Experts in Speech Self-Supervised Models for Audio Deepfake Detection
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出MoLEx框架，用于音频深度伪造检测任务。针对自监督学习模型微调成本高的问题，结合LoRA与专家路由机制，实现高效微调，在ASVSpoof 5数据集上取得SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.09175v1](http://arxiv.org/pdf/2509.09175v1)**

> **作者:** Zihan Pan; Sailor Hardik Bhupendra; Jinyang Wu
>
> **摘要:** While self-supervised learning (SSL)-based models have boosted audio deepfake detection accuracy, fully finetuning them is computationally expensive. To address this, we propose a parameter-efficient framework that combines Low-Rank Adaptation with a Mixture-of-Experts router, called Mixture of LoRA Experts (MoLEx). It preserves pre-trained knowledge of SSL models while efficiently finetuning only selected experts, reducing training costs while maintaining robust performance. The observed utility of experts during inference shows the router reactivates the same experts for similar attacks but switches to other experts for novel spoofs, confirming MoLEx's domain-aware adaptability. MoLEx additionally offers flexibility for domain adaptation by allowing extra experts to be trained without modifying the entire model. We mainly evaluate our approach on the ASVSpoof 5 dataset and achieve the state-of-the-art (SOTA) equal error rate (EER) of 5.56% on the evaluation set without augmentation.
>
---
#### [new 010] EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出EchoX，解决语音到语音大语言模型（SLLM）中知识与推理能力退化问题。通过动态生成语音训练目标，弥合声学-语义差距，保留强推理能力。实验表明其在问答基准上表现优异。属于语音生成与语义理解任务。**

- **链接: [http://arxiv.org/pdf/2509.09174v1](http://arxiv.org/pdf/2509.09174v1)**

> **作者:** Yuhao Zhang; Yuhao Du; Zhanchen Dai; Xiangnan Ma; Kaiqi Kou; Benyou Wang; Haizhou Li
>
> **摘要:** Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at https://github.com/FreedomIntelligence/EchoX.
>
---
#### [new 011] MAPSS: Manifold-based Assessment of Perceptual Source Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MAPSS方法，解决源分离系统的客观评估与主观感知不匹配问题。通过引入感知分离（PS）和感知匹配（PM）指标，利用流形学习量化泄漏与自失真，实现高相关性的人类感知评估。**

- **链接: [http://arxiv.org/pdf/2509.09212v1](http://arxiv.org/pdf/2509.09212v1)**

> **作者:** Amir Ivry; Samuele Cornell; Shinji Watanabe
>
> **备注:** Submitted to ICLR
>
> **摘要:** Objective assessment of source-separation systems still mismatches subjective human perception, especially when leakage and self-distortion interact. We introduce the Perceptual Separation (PS) and Perceptual Match (PM), the first pair of measures that functionally isolate these two factors. Our intrusive method begins with generating a bank of fundamental distortions for each reference waveform signal in the mixture. Distortions, references, and their respective system outputs from all sources are then independently encoded by a pre-trained self-supervised learning model. These representations are aggregated and projected onto a manifold via diffusion maps, which aligns Euclidean distances on the manifold with dissimilarities of the encoded waveforms. On this manifold, the PM measures the Mahalanobis distance from each output to its attributed cluster that consists of its reference and distortions embeddings, capturing self-distortion. The PS accounts for the Mahalanobis distance of the output to the attributed and to the closest non-attributed clusters, quantifying leakage. Both measures are differentiable and granular, operating at a resolution as low as 50 frames per second. We further derive, for both measures, deterministic error radius and non-asymptotic, high-probability confidence intervals (CIs). Experiments on English, Spanish, and music mixtures show that the PS and PM nearly always achieve the highest linear correlation coefficients with human mean-opinion scores than 14 competitors, reaching as high as 86.36% for speech and 87.21% for music. We observe, at worst, an error radius of 1.39% and a probabilistic 95% CI of 12.21% for these coefficients, which improves reliable and informed evaluation. Using mutual information, the measures complement each other most as their values decrease, suggesting they are jointly more informative as system performance degrades.
>
---
#### [new 012] Short-term cognitive fatigue of spatial selective attention after face-to-face conversations in virtual noisy environments
- **分类: eess.AS; cs.SD**

- **简介: 论文研究面对面交谈后空间选择性注意的短期认知疲劳。通过对比噪音环境与安静环境下的任务表现，探讨对话对注意力的影响，并发现训练效应显著。属于认知心理学与听觉注意领域。**

- **链接: [http://arxiv.org/pdf/2509.09479v1](http://arxiv.org/pdf/2509.09479v1)**

> **作者:** Ľuboš Hládek; Piotr Majdak; Robert Baumgartner
>
> **摘要:** Spatial selective attention is an important asset for communication in cocktail party situations but may be compromised by short-term cognitive fatigue. Here we tested whether an effortful conversation in a highly ecological setting depletes task performance in an auditory spatial selective attention task. Young participants with normal hearing performed the task before and after (1) having a real dyadic face-to-face conversation on a free topic in a virtual reverberant room with simulated interfering conversations and background babble noise at 72 dB SPL for 30 minutes, (2) passively listening to the interfering conversations and babble noise, or (3) having the conversation in quiet. Self-reported perceived effort and fatigue increased after conversations in noise and passive listening relative to the reports after conversations in quiet. In contrast to our expectations, response times in the attention task decreased, rather than increased, after conversation in noise and accuracy did not change systematically in any of the conditions on the group level. Unexpectedly, we observed strong training effects between the individual sessions in our within-subject design even after one hour of training on a different day.
>
---
#### [new 013] Automotive sound field reproduction using deep optimization with spatial domain constraint
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出SPMnet方法，解决汽车音响系统中音质与空间定位的矛盾。通过引入空间功率图约束和深度优化设计滤波器，提升复杂环境下的音质与空间定位精度。**

- **链接: [http://arxiv.org/pdf/2509.09149v1](http://arxiv.org/pdf/2509.09149v1)**

> **作者:** Yufan Qian; Tianshu Qu; Xihong Wu
>
> **备注:** 41 pages, 9 figures, Revised and submitted to The Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Sound field reproduction with undistorted sound quality and precise spatial localization is desirable for automotive audio systems. However, the complexity of automotive cabin acoustic environment often necessitates a trade-off between sound quality and spatial accuracy. To overcome this limitation, we propose Spatial Power Map Net (SPMnet), a learning-based sound field reproduction method that improves both sound quality and spatial localization in complex environments. We introduce a spatial power map (SPM) constraint, which characterizes the angular energy distribution of the reproduced field using beamforming. This constraint guides energy toward the intended direction to enhance spatial localization, and is integrated into a multi-channel equalization framework to also improve sound quality under reverberant conditions. To address the resulting non-convexity, deep optimization that use neural networks to solve optimization problems is employed for filter design. Both in situ objective and subjective evaluations confirm that our method enhances sound quality and improves spatial localization within the automotive cabin. Furthermore, we analyze the influence of different audio materials and the arrival angles of the virtual sound source in the reproduced sound field, investigating the potential underlying factors affecting these results.
>
---
#### [new 014] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; eess.SP**

- **简介: 该论文研究无线电法规领域的问答任务，提出一个电信专用的RAG框架，并构建首个多选评估数据集。通过定义领域特定检索指标，提升生成准确性，尤其在GPT-4o上相对提升12%。**

- **链接: [http://arxiv.org/pdf/2509.09651v1](http://arxiv.org/pdf/2509.09651v1)**

> **作者:** Zakaria El Kassimi; Fares Fourati; Mohamed-Slim Alouini
>
> **摘要:** We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at https://github.com/Zakaria010/Radio-RAG.
>
---
#### [new 015] Region-Specific Audio Tagging for Spatial Sound
- **分类: eess.AS; cs.SD**

- **简介: 论文提出区域特定音频标签任务，针对麦克风阵列记录的空间音频，对指定区域内的声音事件进行标注。研究结合谱、空间和位置特征，并扩展PANNs和AST模型，实验验证了方法的有效性及方向特征对全向标注的益处。**

- **链接: [http://arxiv.org/pdf/2509.09526v1](http://arxiv.org/pdf/2509.09526v1)**

> **作者:** Jinzheng Zhao; Yong Xu; Haohe Liu; Davide Berghi; Xinyuan Qian; Qiuqiang Kong; Junqi Zhao; Mark D. Plumbley; Wenwu Wang
>
> **备注:** DCASE2025 Workshop
>
> **摘要:** Audio tagging aims to label sound events appearing in an audio recording. In this paper, we propose region-specific audio tagging, a new task which labels sound events in a given region for spatial audio recorded by a microphone array. The region can be specified as an angular space or a distance from the microphone. We first study the performance of different combinations of spectral, spatial, and position features. Then we extend state-of-the-art audio tagging systems such as pre-trained audio neural networks (PANNs) and audio spectrogram transformer (AST) to the proposed region-specific audio tagging task. Experimental results on both the simulated and the real datasets show the feasibility of the proposed task and the effectiveness of the proposed method. Further experiments show that incorporating the directional features is beneficial for omnidirectional tagging.
>
---
#### [new 016] The Sound of Entanglement
- **分类: quant-ph; cs.ET; cs.MM; cs.SD**

- **简介: 论文探讨量子力学与艺术的结合，利用量子纠缠和随机性创作音乐表演。通过实时测量纠缠光子进行贝尔测试，将量子关联融入音乐与视觉表现，实现不可重复的视听体验，推动科学与艺术的融合。**

- **链接: [http://arxiv.org/pdf/2509.08892v1](http://arxiv.org/pdf/2509.08892v1)**

> **作者:** Enar de Dios Rodríguez; Philipp Haslinger; Johannes Kofler; Richard Kueng; Benjamin Orthner; Alexander Ploier; Martin Ringbauer; Clemens Wenger
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** The advent of quantum physics has revolutionized our understanding of the universe, replacing the deterministic framework of classical physics with a paradigm dominated by intrinsic randomness and quantum correlations. This shift has not only enabled groundbreaking technologies, such as quantum sensors, networks and computers, but has also unlocked entirely new possibilities for artistic expressions. In this paper, we explore the intersection of quantum mechanics and art, focusing on the use of quantum entanglement and inherent randomness as creative tools. Specifically, we present The Sound of Entanglement, a live musical performance driven by real-time measurements of entangled photons in a Bell test. By integrating the measured quantum correlations as a central compositional element and synchronizing live visuals with experimental data, the performance offers a unique and unrepeatable audiovisual experience that relies on quantum correlations which cannot be produced by any classical device. Through this fusion of science and art, we aim to provide a deeper appreciation of quantum phenomena while expanding the boundaries of creative expression.
>
---
## 更新

#### [replaced 001] Pretrained Conformers for Audio Fingerprinting and Retrieval
- **分类: cs.SD; cs.AI; cs.IR; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.11609v2](http://arxiv.org/pdf/2508.11609v2)**

> **作者:** Kemal Altwlkany; Elmedin Selmanovic; Sead Delalic
>
> **摘要:** Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes.
>
---
#### [replaced 002] The NTNU System at the S&I Challenge 2025 SLA Open Track
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.05121v2](http://arxiv.org/pdf/2506.05121v2)**

> **作者:** Hong-Yun Lin; Tien-Hong Lo; Yu-Hsuan Fang; Jhen-Ke Lin; Chung-Chun Wang; Hao-Chien Lu; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** A recent line of research on spoken language assessment (SLA) employs neural models such as BERT and wav2vec 2.0 (W2V) to evaluate speaking proficiency across linguistic and acoustic modalities. Although both models effectively capture features relevant to oral competence, each exhibits modality-specific limitations. BERT-based methods rely on ASR transcripts, which often fail to capture prosodic and phonetic cues for SLA. In contrast, W2V-based methods excel at modeling acoustic features but lack semantic interpretability. To overcome these limitations, we propose a system that integrates W2V with Phi-4 multimodal large language model (MLLM) through a score fusion strategy. The proposed system achieves a root mean square error (RMSE) of 0.375 on the official test set of the Speak & Improve Challenge 2025, securing second place in the competition. For comparison, the RMSEs of the top-ranked, third-ranked, and official baseline systems are 0.364, 0.384, and 0.444, respectively.
>
---
#### [replaced 003] AU-Harness: An Open-Source Toolkit for Holistic Evaluation of Audio LLMs
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.08031v2](http://arxiv.org/pdf/2509.08031v2)**

> **作者:** Sidharth Surapaneni; Hoang Nguyen; Jash Mehta; Aman Tiwari; Oluwanifemi Bamgbose; Akshay Kalkunte; Sai Rajeswar; Sathwik Tejaswi Madhusudhan
>
> **摘要:** Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient toolkits that limit fair comparison and systematic assessment. Current frameworks suffer from three critical issues: slow processing that bottlenecks large-scale studies, inconsistent prompting that hurts reproducibility, and narrow task coverage that misses important audio reasoning capabilities. We introduce AU-Harness, an efficient and comprehensive evaluation framework for LALMs. Our system achieves a speedup of up to 127% over existing toolkits through optimized batch processing and parallel execution, enabling large-scale evaluations previously impractical. We provide standardized prompting protocols and flexible configurations for fair model comparison across diverse scenarios. Additionally, we introduce two new evaluation categories: LLM-Adaptive Diarization for temporal audio understanding and Spoken Language Reasoning for complex audio-based cognitive tasks. Through evaluation across 380+ tasks, we reveal significant gaps in current LALMs, particularly in temporal understanding and complex spoken language reasoning tasks. Our findings also highlight a lack of standardization in instruction modality existent across audio benchmarks, which can lead up performance differences up to 9.5 absolute points on the challenging complex instruction following downstream tasks. AU-Harness provides both practical evaluation tools and insights into model limitations, advancing systematic LALM development.
>
---
#### [replaced 004] Enhancing Automatic Modulation Recognition With a Reconstruction-Driven Vision Transformer Under Limited Labels
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.20193v2](http://arxiv.org/pdf/2508.20193v2)**

> **作者:** Hossein Ahmadi; Banafsheh Saffari; Sajjad Emdadi Mahdimahalleh; Mohammad Esmaeil Safari; Aria Ahmadi
>
> **摘要:** Automatic modulation recognition (AMR) is critical for cognitive radio, spectrum monitoring, and secure wireless communication. However, existing solutions often rely on large labeled datasets or multi-stage training pipelines, which limit scalability and generalization in practice. We propose a unified Vision Transformer (ViT) framework that integrates supervised, self-supervised, and reconstruction objectives. The model combines a ViT encoder, a lightweight convolutional decoder, and a linear classifier; the reconstruction branch maps augmented signals back to their originals, anchoring the encoder to fine-grained I/Q structure. This strategy promotes robust, discriminative feature learning during pretraining, while partial label supervision in fine-tuning enables effective classification with limited labels. On the RML2018.01A dataset, our approach outperforms supervised CNN and ViT baselines in low-label regimes, approaches ResNet-level accuracy with only 15-20% labeled data, and maintains strong performance across varying SNR levels. Overall, the framework provides a simple, generalizable, and label-efficient solution for AMR.
>
---
#### [replaced 005] FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02521v2](http://arxiv.org/pdf/2509.02521v2)**

> **作者:** Yiqun Yao; Xiang Li; Xin Jiang; Xuezhi Fang; Naitong Yu; Wenjia Ma; Aixin Sun; Yequan Wang
>
> **摘要:** Full-duplex dialog models aim to listen and speak simultaneously, delivering rapid responses to dynamic user input. Among different solutions to full duplexity, a native solution merges multiple channels in each time step, achieving the lowest latency. However, prevailing designs break down the textual monologue sentences for word-level alignment with audio streams, which degrades language modeling abilities. To help address this issue, we introduce natural monologues, which are composed by continuous sentences and waiting intervals, mimicking humanoid cognitive behavior in dialogs. We find a proper training paradigm to be critical for semantically aligning natural monologues with audio. To this end, we develop a dual training paradigm that alternates the position of the monologues, either leading or trailing the audio, across different training stages. A combination of our natural monologue and dual training strategy is applied in developing FLM-Audio, our 7B spoken dialog chatbot with native full-duplexity. As confirmed by experimental results, FLM-Audio achieves superior response qualities and chatting experiences while requiring significantly less training data.
>
---
#### [replaced 006] Binaural Target Speaker Extraction using HRTFs
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19369v3](http://arxiv.org/pdf/2507.19369v3)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **摘要:** In this work, we aim to imitate the human ability to selectively attend to a single speaker, even in the presence of multiple simultaneous talkers. To achieve this, we propose a novel approach for binaural target speaker extraction that leverages the listener's Head-Related Transfer Function (HRTF) to isolate the desired speaker. Notably, our method does not rely on speaker embeddings, making it speaker-independent and enabling strong generalization across multiple speech datasets and languages. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier transform (STFT) of the mixed audio signals, and compare it to a Real-Imaginary (RI)-based neural network, demonstrating the advantages of the former. We first evaluate the method in an anechoic, noise-free scenario, achieving excellent extraction performance while preserving the binaural cues of the target signal. We then extend the evaluation to reverberant conditions. Our method proves robust, maintaining speech clarity and source directionality while simultaneously reducing reverberation. A comparative analysis with existing binaural Target Speaker Extraction (TSE) methods demonstrates that our approach attains performance on par with competing techniques in terms of noise reduction and perceptual quality, while offering a clear advantage in preserving binaural cues.Demo-page: https://bi-ctse-hrtf.github.io
>
---
#### [replaced 007] Preprocessing Algorithm Leveraging Geometric Modeling for Scale Correction in Hyperspectral Images for Improved Unmixing Performance
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.08431v2](http://arxiv.org/pdf/2508.08431v2)**

> **作者:** Praveen Sumanasekara; Athulya Ratnayake; Buddhi Wijenayake; Keshawa Ratnayake; Roshan Godaliyadda; Parakrama Ekanayake; Vijitha Herath
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Spectral variability significantly impacts the accuracy and convergence of hyperspectral unmixing algorithms. Many methods address complex spectral variability; yet large-scale distortions to the scale of the observed pixel signatures due to topography, illumination, and shadowing remain a major challenge. These variations often degrade unmixing performance and complicate model fitting. Because of this, correcting these variations can offer significant advantages in real-world GIS applications. In this paper, we propose a novel preprocessing algorithm that corrects scale-induced spectral variability prior to unmixing. By estimating and correcting these distortions to the scale of the pixel signatures, the algorithm produces pixel signatures with minimal distortions in scale. Since these distortions in scale (which hinder the performance of many unmixing methods) are greatly minimized in the output provided by the proposed method, the abundance estimation of the unmixing algorithms is significantly improved. We present a rigorous mathematical framework to describe and correct for scale variability and provide extensive experimental validation of the proposed algorithm. Furthermore, the algorithm's impact is evaluated across a wide range of state-of-the-art unmixing methods on two synthetic and two real hyperspectral datasets. The proposed preprocessing step consistently improves the performance of these algorithms, achieving error reductions of around 50%, even for algorithms specifically designed to handle spectral variability. This demonstrates that scale correction acts as a complementary step, facilitating more accurate unmixing with existing methods. The algorithm's generality, consistent impact, and significant influence highlight its potential as a key component in practical hyperspectral unmixing pipelines. The implementation code will be made publicly available upon publication.
>
---
#### [replaced 008] A Novel Data Augmentation Approach for Automatic Speaking Assessment on Opinion Expressions
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04077v2](http://arxiv.org/pdf/2506.04077v2)**

> **作者:** Chung-Chun Wang; Jhen-Ke Lin; Hao-Chien Lu; Hong-Yun Lin; Berlin Chen
>
> **备注:** submitted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Automated speaking assessment (ASA) on opinion expressions is often hampered by the scarcity of labeled recordings, which restricts prompt diversity and undermines scoring reliability. To address this challenge, we propose a novel training paradigm that leverages a large language models (LLM) to generate diverse responses of a given proficiency level, converts responses into synthesized speech via speaker-aware text-to-speech synthesis, and employs a dynamic importance loss to adaptively reweight training instances based on feature distribution differences between synthesized and real speech. Subsequently, a multimodal large language model integrates aligned textual features with speech signals to predict proficiency scores directly. Experiments conducted on the LTTC dataset show that our approach outperforms methods relying on real data or conventional augmentation, effectively mitigating low-resource constraints and enabling ASA on opinion expressions with cross-modal information.
>
---
#### [replaced 009] Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.08454v2](http://arxiv.org/pdf/2509.08454v2)**

> **作者:** Yujian Ma; Jinqiu Sang; Ruizhe Li
>
> **备注:** Work in process
>
> **摘要:** Large pre-trained speech models such as Whisper offer strong generalization but pose significant challenges for resource-efficient adaptation. Low-Rank Adaptation (LoRA) has become a popular parameter-efficient fine-tuning method, yet its underlying mechanisms in speech tasks remain poorly understood. In this work, we conduct the first systematic mechanistic interpretability study of LoRA within the Whisper encoder for speech emotion recognition (SER). Using a suite of analytical tools, including layer contribution probing, logit-lens inspection, and representational similarity via singular value decomposition (SVD) and centered kernel alignment (CKA), we reveal two key mechanisms: a delayed specialization process that preserves general features in early layers before consolidating task-specific information, and a forward alignment, backward differentiation dynamic between LoRA's matrices. Our findings clarify how LoRA reshapes encoder hierarchies, providing both empirical insights and a deeper mechanistic understanding for designing efficient and interpretable adaptation strategies in large speech models. Our code is available at https://github.com/harryporry77/Behind-the-Scenes.
>
---
