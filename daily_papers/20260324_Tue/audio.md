# 音频 cs.SD;  eess.AS

- **最新发布 27 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Disentangling Speaker Traits for Deepfake Source Verification via Chebyshev Polynomial and Riemannian Metric Learning
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音深度伪造源验证任务，旨在解决源与说话人特征混淆的问题。提出SDML框架，通过Chebyshev多项式和双曲空间投影实现特征解耦，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2603.21875](https://arxiv.org/pdf/2603.21875)**

> **作者:** Xi Xuan; Wenxin Zhang; Zhiyu Li; Jennifer Williams; Ville Hautamäki; Tomi H. Kinnunen
>
> **备注:** Submitted to Interspeech 2026; The code, evaluation protocols and demo website are available at this https URL
>
> **摘要:** Speech deepfake source verification systems aims to determine whether two synthetic speech utterances originate from the same source generator, often assuming that the resulting source embeddings are independent of speaker traits. However, this assumption remains unverified. In this paper, we first investigate the impact of speaker factors on source verification. We propose a speaker-disentangled metric learning (SDML) framework incorporating two novel loss functions. The first leverages Chebyshev polynomial to mitigate gradient instability during disentanglement optimization. The second projects source and speaker embeddings into hyperbolic space, leveraging Riemannian metric distances to reduce speaker information and learn more discriminative source features. Experimental results on MLAAD benchmark, evaluated under four newly proposed protocols designed for source-speaker disentanglement scenarios, demonstrate the effectiveness of SDML framework. The code, evaluation protocols and demo website are available at this https URL.
>
---
#### [new 002] WiRD-Gest: Gesture Recognition In The Real World Using Range-Doppler Wi-Fi Sensing on COTS Hardware
- **分类: eess.AS**

- **简介: 该论文属于手势识别任务，旨在解决Wi-Fi传感在真实环境中的稳定性与泛化问题。通过单设备获取范围-多普勒信息，提升识别准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22131](https://arxiv.org/pdf/2603.22131)**

> **作者:** Jessica Sanson; Rahul C. Shah; Yazhou Zhu; Rafael Rosales; Valerio Frascolla
>
> **摘要:** Wi-Fi sensing has emerged as a promising technique for gesture recognition, yet its practical deployment is hindered by environmental sensitivity and device placement challenges. To overcome these limitations we propose Wi-Fi Range and Doppler (WiRD)-Gest, a novel system that performs gesture recognition using a single, unmodified Wi-Fi transceiver on a commercial off-the-shelf (COTS) laptop. The system leverages an monostatic full duplex sensing pipeline capable of extracting Range-Doppler (RD) information. Utilizing this, we present the first benchmark of deep learning models for gesture recognition based on monostatic sensing. The key innovation lies in how monostatic sensing and spatial (range) information fundamentally transforms accuracy, robustness and generalization compared to prior approaches. We demonstrate excellent performance in crowded, unseen public spaces with dynamic interference and additional moving targets even when trained on data from controlled environments only. These are scenarios where prior Wi-Fi sensing approaches often fail, however, our system suffers minor degradation. The WiRD-Gest benchmark and dataset will also be released as open source.
>
---
#### [new 003] Emotion-Aware Quantization for Discrete Speech Representations: An Analysis of Emotion Preservation
- **分类: cs.SD**

- **简介: 该论文属于语音情感处理任务，旨在解决压缩导致情感信息丢失的问题。通过引入情感感知的量化方法，提升情感信息的保留效果。**

- **链接: [https://arxiv.org/pdf/2603.21224](https://arxiv.org/pdf/2603.21224)**

> **作者:** Haoguang Zhou; Siyi Wang; Jingyao Wu; James Bailey; Ting Dang
>
> **摘要:** Modern speech systems increasingly use discretized self-supervised speech representations for compression and integration with token-based models, yet their impact on emotional information remains unclear. We study how residual vector quantization (RVQ) reshapes emotional information in discrete speech representations from both representation- and task-level perspectives. Our analysis shows that aggressive compression disproportionately degrades emotion, with uneven loss across emotion classes and model architectures. To address this, we introduce emotion-aware quantization using emotion-specific and emotion-biased codebooks, improving the preservation of both hard and soft emotion perception. We further propose Emo-Q, a lightweight routed quantization method that selects emotion-specialized codebooks, improving emotion recognition performance at lower bitrates. These results highlight the importance of emotion-aware discretization for robust affective speech processing.
>
---
#### [new 004] SqueezeComposer: Temporal Speed-up is A Simple Trick for Long-form Music Composing
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出SqueezeComposer，解决长时音乐生成中的计算与内存限制问题。通过先生成加速音频再恢复原速，实现高效、高质量的长音乐生成。**

- **链接: [https://arxiv.org/pdf/2603.21073](https://arxiv.org/pdf/2603.21073)**

> **作者:** Jianyi Chen; Rongxiu Zhong; Shilei Zhang; Kun Qian; Jinglei Liu; Yike Guo; Wei Xue
>
> **备注:** Under Review
>
> **摘要:** Composing coherent long-form music remains a significant challenge due to the complexity of modeling long-range dependencies and the prohibitive memory and computational requirements associated with lengthy audio representations. In this work, we propose a simple yet powerful trick: we assume that AI models can understand and generate time-accelerated (speeded-up) audio at rates such as 2x, 4x, or even 8x. By first generating a high-speed version of the music, we greatly reduce the temporal length and resource requirements, making it feasible to handle long-form music that would otherwise exceed memory or computational limits. The generated audio is then restored to its original speed, recovering the full temporal structure. This temporal speed-up and slow-down strategy naturally follows the principle of hierarchical generation from abstract to detailed content, and can be conveniently applied to existing music generation models to enable long-form music generation. We instantiate this idea in SqueezeComposer, a framework that employs diffusion models for generation in the accelerated domain and refinement in the restored domain. We validate the effectiveness of this approach on two tasks: long-form music generation, which evaluates temporal-wise control (including continuation, completion, and generation from scratch), and whole-song singing accompaniment generation, which evaluates track-wise control. Experimental results demonstrate that our simple temporal speed-up trick enables efficient, scalable, and high-quality long-form music generation. Audio samples are available at this https URL.
>
---
#### [new 005] LL-SDR: Low-Latency Speech enhancement through Discrete Representations
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决噪声环境下语音清晰度问题。提出LL-SDR框架，通过离散表示分离语音与噪声，实现低延迟增强。**

- **链接: [https://arxiv.org/pdf/2603.20242](https://arxiv.org/pdf/2603.20242)**

> **作者:** Jingyi Li; Luca Della Libera; Mirco Ravanelli; Cem Subakan
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Many speech enhancement (SE) methods rely on continuous representations. Recently, discrete audio tokens have been explored to enable autoregressive generation for SE. However, it remains unclear whether discretization itself consistently improves SE performance. In this paper, we introduce LL-SDR, a token-based speech enhancement framework that explicitly leverages discretization to better separate speech and noise. Our first contribution is a Variance-Ordered Residual Vector Quantizer (VO-RVQ), designed to disentangle speech and noise distributions during tokenization. Second, we propose a latent-space discriminator to better align enhanced embeddings with semantic embeddings. Experiments show that LL-SDR outperforms continuous baselines and matches the performance of autoregressive token-based approaches, while enabling lightweight, low-latency speech enhancement in both reverberant and non-reverberant noisy environments. Demos and source code are available at our project websites.
>
---
#### [new 006] SelfTTS: cross-speaker style transfer through explicit embedding disentanglement and self-refinement using self-augmentation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SelfTTS，属于跨说话人风格迁移任务，解决无需外部编码器的问题。通过显式解耦和自增强策略，提升语音自然度与情感表达。**

- **链接: [https://arxiv.org/pdf/2603.22252](https://arxiv.org/pdf/2603.22252)**

> **作者:** Lucas H. Ueda; João G. T. Lima; Pedro R. Corrêa; Flávio O. Simões; Mário U. Neto; Paula D. P. Costa
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** This paper presents SelfTTS, a text-to-speech (TTS) model designed for cross-speaker style transfer that eliminates the need for external pre-trained speaker or emotion encoders. The architecture achieves emotional expressivity in neutral speakers through an explicit disentanglement strategy utilizing Gradient Reversal Layers (GRL) combined with cosine similarity loss to decouple speaker and emotion information. We introduce Multi Positive Contrastive Learning (MPCL) to induce clustered representations of speaker and emotion embeddings based on their respective labels. Furthermore, SelfTTS employs a self-refinement strategy via Self-Augmentation, exploiting the model's voice conversion capabilities to enhance the naturalness of synthesized speech. Experimental results demonstrate that SelfTTS achieves superior emotional naturalness (eMOS) and robust stability in target timbre and emotion compared to state-of-the-art baselines.
>
---
#### [new 007] HELIX: Scaling Raw Audio Understanding with Hybrid Mamba-Attention Beyond the Quadratic Limit
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在解决模型架构选择与输入表示之间关系的问题。通过对比纯Mamba、纯注意力及混合模型，发现不同架构对音频长度敏感，提出HELIX框架提升长序列处理能力。**

- **链接: [https://arxiv.org/pdf/2603.21316](https://arxiv.org/pdf/2603.21316)**

> **作者:** Khushiyant; Param Thakkar
>
> **备注:** 10 Pages, 8 Figures
>
> **摘要:** Audio representation learning typically evaluates design choices such as input frontend, sequence backbone, and sequence length in isolation. We show that these axes are coupled, and conclusions from one setting often do not transfer to others. We introduce HELIX, a controlled framework comparing pure Mamba, pure attention, and a minimal hybrid with a single attention bottleneck. All models are parameter-matched at about 8.3M parameters to isolate architectural effects. Across six datasets, we find that the preferred input representation depends on the backbone, and that attention hurts performance on short, stationary audio but becomes important at longer sequence lengths. On a 5-minute speaker identification task with 30,000 tokens, pure attention fails with out-of-memory errors, while HELIX closes an 11.5-point gap over pure Mamba.
>
---
#### [new 008] Enterprise Sales Copilot: Enabling Real-Time AI Support with Automatic Information Retrieval in Live Sales Calls
- **分类: cs.SD**

- **简介: 该论文提出SalesCopilot，解决销售通话中信息检索效率低的问题。通过实时AI技术自动检测问题并快速提供答案，提升销售效率与客户体验。**

- **链接: [https://arxiv.org/pdf/2603.21416](https://arxiv.org/pdf/2603.21416)**

> **作者:** Jielin Qiu; Liangwei Yang; Ming Zhu; Wenting Zhao; Zhiwei Liu; Juntao Tan; Zixiang Chen; Roshan Ram; Akshara Prabhakar; Rithesh Murthy; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** During live sales calls, customers frequently ask detailed product questions that require representatives to manually search internal databases and CRM systems. This process typically takes 25-65 seconds per query, creating awkward pauses that hurt customer experience and reduce sales efficiency. We present SalesCopilot, a real-time AI-powered assistant that eliminates this bottleneck by automatically detecting customer questions, retrieving relevant information from the product database, and displaying concise answers on the representative's dashboard in seconds. The system integrates streaming speech-to-text transcription, large language model (LLM)-based question detection, and retrieval-augmented generation (RAG) over a structured product database into a unified real-time pipeline. We demonstrate SalesCopilot on an insurance sales scenario with 50 products spanning 10 categories (2,490 FAQs, 290 coverage details, and 162 pricing tiers). In our benchmark evaluation, SalesCopilot achieves a measured mean response time of 2.8 seconds with 100% question detection rate, representing a 14xspeedup compared to manual CRM search in an internal study. The system is domain-agnostic and can be adapted to any enterprise sales domain by replacing the product database.
>
---
#### [new 009] SNAP: Speaker Nulling for Artifact Projection in Speech Deepfake Detection
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决模型对说话人信息的依赖问题。通过引入SNAP框架，抑制说话人相关特征，提升检测器对合成痕迹的敏感性。**

- **链接: [https://arxiv.org/pdf/2603.20686](https://arxiv.org/pdf/2603.20686)**

> **作者:** Kyudan Jung; Jihwan Kim; Minwoo Lee; Soyoon Kim; Jeonghoon Kim; Jaegul Choo; Cheonbok Park
>
> **备注:** 9 pages, 3 figures, 2 tables
>
> **摘要:** Recent advancements in text-to-speech technologies enable generating high-fidelity synthetic speech nearly indistinguishable from real human voices. While recent studies show the efficacy of self-supervised learning-based speech encoders for deepfake detection, these models struggle to generalize across unseen speakers. Our quantitative analysis suggests these encoder representations are substantially influenced by speaker information, causing detectors to exploit speaker-specific correlations rather than artifact-related cues. We call this phenomenon speaker entanglement. To mitigate this reliance, we introduce SNAP, a speaker-nulling framework. We estimate a speaker subspace and apply orthogonal projection to suppress speaker-dependent components, isolating synthesis artifacts within the residual features. By reducing speaker entanglement, SNAP encourages detectors to focus on artifact-related patterns, leading to state-of-the-art performance.
>
---
#### [new 010] LipsAM: Lipschitz-Continuous Amplitude Modifier for Audio Signal Processing and its Application to Plug-and-Play Dereverberation
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频信号处理任务，旨在解决DNN在音频处理中的稳定性问题。提出LipsAM架构，确保幅度修饰器的Lipschitz连续性，提升去混响算法的稳定性。**

- **链接: [https://arxiv.org/pdf/2603.21684](https://arxiv.org/pdf/2603.21684)**

> **作者:** Kazuki Matsumoto; Ren Uchida; Kohei Yatabe
>
> **备注:** Accepted for IEEE ICASSP 2026
>
> **摘要:** The robustness of deep neural networks (DNNs) can be certified through their Lipschitz continuity, which has made the construction of Lipschitz-continuous DNNs an active research field. However, DNNs for audio processing have not been a major focus due to their poor compatibility with existing results. In this paper, we consider the amplitude modifier (AM), a popular architecture for handling audio signals, and propose its Lipschitz-continuous variants, which we refer to as LipsAM. We prove a sufficient condition for an AM to be Lipschitz continuous and propose two architectures as examples of LipsAM. The proposed architectures were applied to a Plug-and-Play algorithm for speech dereverberation, and their improved stability is demonstrated through numerical experiments.
>
---
#### [new 011] Voice Privacy from an Attribute-based Perspective
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音隐私研究任务，旨在解决语音匿名化中的属性泄露问题。通过分析说话人属性的唯一性及攻击误差率，评估现有匿名化方法的隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2603.20301](https://arxiv.org/pdf/2603.20301)**

> **作者:** Mehtab Ur Rahman; Martha Larson; Cristian Tejedor García
>
> **备注:** Submitted to InterSpeech 2026
>
> **摘要:** Voice privacy approaches that preserve the anonymity of speakers modify speech in an attempt to break the link with the true identity of the speaker. Current benchmarks measure speaker protection based on signal-to-signal comparisons. In this paper, we introduce an attribute-based perspective, where we measure privacy protection in terms of comparisons between sets of speaker attributes. First, we analyze privacy impact by calculating speaker uniqueness for ground truth attributes, attributes inferred on the original speech, and attributes inferred on speech protected with standard anonymization. Next, we examine a threat scenario involving only a single utterance per speaker and calculate attack error rates. Overall, we observe that inferred attributes still present a risk despite attribute inference errors. Our research points to the importance of considering both attribute-related threats and protection mechanisms in future voice privacy research.
>
---
#### [new 012] ALICE: A Multifaceted Evaluation Framework of Large Audio-Language Models' In-Context Learning Ability
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频语言模型研究，旨在评估大音频语言模型的上下文学习能力。针对现有模型在音频条件下任务理解不足的问题，提出ALICE框架进行系统测试，发现模型仅能提升格式合规性，难以提升核心任务性能。**

- **链接: [https://arxiv.org/pdf/2603.20433](https://arxiv.org/pdf/2603.20433)**

> **作者:** Yen-Ting Piao; Jay Chiehen Liao; Wei-Tang Chien; Toshiki Ogimoto; Shang-Tse Chen; Yun-Nung Chen; Chun-Yi Lee; Shao-Yuan Lo
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While Large Audio-Language Models (LALMs) have been shown to exhibit degraded instruction-following capabilities, their ability to infer task patterns from in-context examples under audio conditioning remains unstudied. To address this gap, we present ALICE, a three-stage framework that progressively reduces textual guidance to systematically evaluate LALMs' in-context learning ability under audio conditioning. Evaluating six LALMs across four audio understanding tasks under two output constraint categories, we uncover a consistent asymmetry across all stages and LALMs: in-context demonstrations reliably improve format compliance but fail to improve, and often degrade, the core task performance. This suggests that LALMs can glean surface-level formatting patterns from demonstrations but may struggle to leverage cross-modal semantic grounding to reliably infer task objectives from audio-conditioned examples, highlighting potential limitations in current cross-modal integration.
>
---
#### [new 013] Adaptive Federated Fine-Tuning of Self-Supervised Speech Representations
- **分类: eess.AS**

- **简介: 该论文属于语音任务的联邦学习领域，解决联邦环境下的异构性和计算效率问题。通过引入轻量预测头和分层聚合策略，实现自适应微调，提升资源受限环境下的性能。**

- **链接: [https://arxiv.org/pdf/2603.21888](https://arxiv.org/pdf/2603.21888)**

> **作者:** Xin Guo; Chunrui Zhao; Hong Jia; Ting Dang; Gongping Huang; Xianrui Zheng; Yan Gao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Integrating Federated Learning (FL) with self-supervised learning (SSL) enables privacy-preserving fine-tuning for speech tasks. However, federated environments exhibit significant heterogeneity: clients differ in computational capacity, causing straggler effects under unified fine-tuning, while diverse downstream tasks require different representation depths, making full-model updates inefficient. To address these challenges, we propose an adaptive federated fine-tuning framework with early exits. Lightweight prediction heads are inserted at intermediate layers of the SSL backbone, allowing clients to terminate computation based on local constraints and task requirements. We further introduce a layer-wise, depth-aware partial aggregation strategy to better utilize representations from different network depths. Experiments show that the framework reduces edge overhead, supports heterogeneous hardware, and maintains competitive performance in resource-constrained federated environments.
>
---
#### [new 014] ERM-MinMaxGAP: Benchmarking and Mitigating Gender Bias in Multilingual Multimodal Speech-LLM Emotion Recognition
- **分类: cs.SD**

- **简介: 该论文属于多语言多模态情感识别任务，旨在解决性别偏见问题。通过构建基准数据集并提出ERM-MinMaxGAP方法，提升性能并减少性别差异。**

- **链接: [https://arxiv.org/pdf/2603.21050](https://arxiv.org/pdf/2603.21050)**

> **作者:** Zi Haur Pang; Xiaoxue Gao; Tatsuya Kawahara; Nancy F. Chen
>
> **摘要:** Speech emotion recognition (SER) systems can exhibit gender-related performance disparities, but how such bias manifests in multilingual speech LLMs across languages and modalities is unclear. We introduce a novel multilingual, multimodal benchmark built on MELD-ST, spanning English, Japanese, and German, to quantify language-specific SER performance and gender gaps. We find bias is strongly language-dependent, and multimodal fusion does not reliably improve fairness. To address these, we propose ERM-MinMaxGAP, a fairness-informed training objective, which augments empirical risk minimization (ERM) with a proposed adaptive fairness weight mechanism and a novel MinMaxGAP regularizer on the maximum male-female loss gap within each language and modality. Building upon the Qwen2-Audio backbone, our ERM-MinMaxGAP approach improves multilingual SER performance by 5.5% and 5.0% while reducing the overall gender bias gap by 0.1% and 1.4% in the unimodal and multimodal settings, respectively.
>
---
#### [new 015] DiT-Flow: Speech Enhancement Robust to Multiple Distortions based on Flow Matching in Latent Space and Diffusion Transformers
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决模型在多种噪声条件下泛化能力不足的问题。提出DiT-Flow框架，结合流匹配与扩散Transformer，在潜在空间中提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.21608](https://arxiv.org/pdf/2603.21608)**

> **作者:** Tianyu Cao; Helin Wang; Ari Frummer; Yuval Sieradzki; Adi Arbel; Laureano Moro Velazquez; Jesus Villalba; Oren Gal; Thomas Thebaud; Najim Dehak
>
> **摘要:** Recent advances in generative models, such as diffusion and flow matching, have shown strong performance in audio tasks. However, speech enhancement (SE) models are typically trained on limited datasets and evaluated under narrow conditions, limiting real-world applicability. To address this, we propose DiT-Flow, a flow matching-based SE framework built on the latent Diffusion Transformer (DiT) backbone and trained for robustness across diverse distortions, including noise, reverberation, and compression. DiT-Flow operates on compact variational auto-encoders (VAEs)-derived latent features. We validated our approach on StillSonicSet, a synthetic yet acoustically realistic dataset composed of LibriSpeech, FSD50K, FMA, and 90 Matterport3D scenes. Experiments show that DiT-Flow consistently outperforms state-of-the-art generative SE models, demonstrating the effectiveness of flow matching in multi-condition speech enhancement. Despite ongoing efforts to expand synthetic data realism, a persistent bottleneck in SE is the inevitable mismatch between training and deployment conditions. By integrating LoRA with the MoE framework, we achieve both parameter-efficient and high-performance training for DiT-Flow robust to multiple distortions with using 4.9% percentage of the total parameters to obtain a better performance on five unseen distortions.
>
---
#### [new 016] AnimalCLAP: Taxonomy-Aware Language-Audio Pretraining for Species Recognition and Trait Inference
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于物种识别与生态属性推断任务，旨在解决未见物种分类难题。通过构建包含6823种动物叫声的数据集，并引入分类学信息，提出AnimalCLAP模型，提升物种识别与属性推断效果。**

- **链接: [https://arxiv.org/pdf/2603.22053](https://arxiv.org/pdf/2603.22053)**

> **作者:** Risa Shinoda; Kaede Shiohara; Nakamasa Inoue; Hiroaki Santo; Fumio Okura
>
> **备注:** ICASSP 2026
>
> **摘要:** Animal vocalizations provide crucial insights for wildlife assessment, particularly in complex environments such as forests, aiding species identification and ecological monitoring. Recent advances in deep learning have enabled automatic species classification from their vocalizations. However, classifying species unseen during training remains challenging. To address this limitation, we introduce AnimalCLAP, a taxonomy-aware language-audio framework comprising a new dataset and model that incorporate hierarchical biological information. Specifically, our vocalization dataset consists of 4,225 hours of recordings covering 6,823 species, annotated with 22 ecological traits. The AnimalCLAP model is trained on this dataset to align audio and textual representations using taxonomic structures, improving the recognition of unseen species. We demonstrate that our proposed model effectively infers ecological and biological attributes of species directly from their vocalizations, achieving superior performance compared to CLAP. Our dataset, code, and models will be publicly available at this https URL.
>
---
#### [new 017] OmniCodec: Low Frame Rate Universal Audio Codec with Semantic-Acoustic Disentanglement
- **分类: eess.AS**

- **简介: 该论文提出OmniCodec，解决低帧率下跨音频领域统一编码问题，通过语义-声学解耦提升重建质量和语义信息，适用于语音、音乐等多样音频。**

- **链接: [https://arxiv.org/pdf/2603.20638](https://arxiv.org/pdf/2603.20638)**

> **作者:** Jingbin Hu; Haoyu Zhang; Dake Guo; Qirui Zhan; Wenhao Li; Huakang Chen; Guobin Ma; Hanke Xie; Chengyou Wang; Pengyuan Xie; Chuan Xie; Qiang Zhang; Lei Xie
>
> **摘要:** Large Language Models (LLMs) have advanced audio generation through discrete representation learning. However, most existing neural codecs focus on speech and emphasize reconstruction fidelity, overlooking unified low frame rate modeling across diverse audio domains, including speech, music, and general sound. Moreover, high reconstruction quality does not necessarily yield semantically informative representations, limiting effectiveness in downstream generation tasks. We propose OmniCodec, a universal neural audio codec tailored for low frame rate. It adopts a hierarchical multi-codebook design with semantic-acoustic decoupling by leveraging the audio encoder of the pre-trained understanding model, along with a self-guidance strategy to improve codebook utilization and reconstruction. Compared with the Mimi codec, experiments show that OmniCodec achieves outstanding performance at the same bitrate, delivering superior reconstruction quality while also providing more semantically informative representations that benefit downstream generation tasks. Our model and code will be open-sourced. Our demo page is available.
>
---
#### [new 018] End-to-End Multi-Task Learning for Adjustable Joint Noise Reduction and Hearing Loss Compensation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强与听力补偿任务，解决同时优化噪声抑制和听力损失补偿的问题。通过多任务学习框架，设计可调整的联合模型，提升性能并实现个性化。**

- **链接: [https://arxiv.org/pdf/2603.20387](https://arxiv.org/pdf/2603.20387)**

> **作者:** Philippe Gonzalez; Vera Margrethe Frederiksen; Torsten Dau; Tobias May
>
> **摘要:** A multi-task learning framework is proposed for optimizing a single deep neural network (DNN) for joint noise reduction (NR) and hearing loss compensation (HLC). A distinct training objective is defined for each task, and the DNN predicts two time-frequency masks. During inference, the amounts of NR and HLC can be adjusted independently by exponentiating each mask before combining them. In contrast to recent approaches that rely on training an auditory-model emulator to define a differentiable training objective, we propose an auditory model that is inherently differentiable, thus allowing end-to-end optimization. The audiogram is provided as an input to the DNN, thereby enabling listener-specific personalization without the need for retraining. Results show that the proposed approach not only allows adjusting the amounts of NR and HLC individually, but also improves objective metrics compared to optimizing a single training objective. It also outperforms a cascade of two DNNs that were separately trained for NR and HLC, and shows competitive HLC performance compared to a traditional hearing-aid prescription. To the best of our knowledge, this is the first study that uses an auditory model to train a single DNN for both NR and HLC across a wide range of listener profiles.
>
---
#### [new 019] Abjad-Kids: An Arabic Speech Classification Dataset for Primary Education
- **分类: cs.CL; cs.HC; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Abjad-Kids数据集，用于阿拉伯语儿童语音分类任务，解决低资源语言儿童语音数据不足的问题。通过CNN-LSTM模型进行分类研究。**

- **链接: [https://arxiv.org/pdf/2603.20255](https://arxiv.org/pdf/2603.20255)**

> **作者:** Abdul Aziz Snoubara; Baraa Al_Maradni; Haya Al_Naal; Malek Al_Madrmani; Roaa Jdini; Seedra Zarzour; Khloud Al Jallad
>
> **摘要:** Speech-based AI educational applications have gained significant interest in recent years, particularly for children. However, children speech research remains limited due to the lack of publicly available datasets, especially for low-resource languages such as this http URL paper presents Abjad-Kids, an Arabic speech dataset designed for kindergarten and primary education, focusing on fundamental learning of alphabets, numbers, and colors. The dataset consists of 46397 audio samples collected from children aged 3 - 12 years, covering 141 classes. All samples were recorded under controlled specifications to ensure consistency in duration, sampling rate, and format. To address high intra-class similarity among Arabic phonemes and the limited samples per class, we propose a hierarchical audio classification based on CNN-LSTM architectures. Our proposed methodology decomposes alphabet recognition into a two-stage process: an initial grouping classification model followed by specialized classifiers for each group. Both strategies: static linguistic-based grouping and dynamic clustering-based grouping, were evaluated. Experimental results demonstrate that static linguistic-based grouping achieves superior performance. Comparisons between traditional machine learning with deep learning approaches, highlight the effectiveness of CNN-LSTM models combined with data augmentation. Despite achieving promising results, most of our experiments indicate a challenge with overfitting, which is likely due to the limited number of samples, even after data augmentation and model regularization. Thus, future work may focus on collecting additional data to address this issue. Abjad-Kids will be publicly available. We hope that Abjad-Kids enrich children representation in speech dataset, and be a good resource for future research in Arabic speech classification for kids.
>
---
#### [new 020] TaigiSpeech: A Low-Resource Real-World Speech Intent Dataset and Preliminary Results with Scalable Data Mining In-the-Wild
- **分类: cs.CL; cs.LG; eess.AS**

- **简介: 该论文介绍了一个针对低资源语言的语音意图数据集TaigiSpeech，用于解决语音技术中语言资源不足的问题。通过数据挖掘策略构建数据集，支持医疗和家庭助手等应用场景。**

- **链接: [https://arxiv.org/pdf/2603.21478](https://arxiv.org/pdf/2603.21478)**

> **作者:** Kai-Wei Chang; Yi-Cheng Lin; Huang-Cheng Chou; Wenze Ren; Yu-Han Huang; Yun-Shao Tsai; Chien-Cheng Chen; Yu Tsao; Yuan-Fu Liao; Shrikanth Narayanan; James Glass; Hung-yi Lee
>
> **备注:** submitted to Interspeech 2026
>
> **摘要:** Speech technologies have advanced rapidly and serve diverse populations worldwide. However, many languages remain underrepresented due to limited resources. In this paper, we introduce \textbf{TaigiSpeech}, a real-world speech intent dataset in Taiwanese Taigi (aka Taiwanese Hokkien/Southern Min), which is a low-resource and primarily spoken language. The dataset is collected from older adults, comprising 21 speakers with a total of 3k utterances. It is designed for practical intent detection scenarios, including healthcare and home assistant applications. To address the scarcity of labeled data, we explore two data mining strategies with two levels of supervision: keyword match data mining with LLM pseudo labeling via an intermediate language and an audio-visual framework that leverages multimodal cues with minimal textual supervision. This design enables scalable dataset construction for low-resource and unwritten spoken languages. TaigiSpeech will be released under the CC BY 4.0 license to facilitate broad adoption and research on low-resource and unwritten languages. The project website and the dataset can be found on this https URL.
>
---
#### [new 021] TiCo: Time-Controllable Training for Spoken Dialogue Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决模型无法准确控制响应时长的问题。通过引入时间标记和强化学习，提升模型对时间约束的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.22267](https://arxiv.org/pdf/2603.22267)**

> **作者:** Kai-Wei Chang; Wei-Chih Chen; En-Pei Hu; Hung-yi Lee; James Glass
>
> **摘要:** We propose TiCo, a simple post-training method for enabling spoken dialogue models (SDMs) to follow time-constrained instructions and generate responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions (e.g., "Please generate a response lasting about 15 seconds"). Through an empirical evaluation of both open-source and commercial SDMs, we show that they frequently fail to satisfy such time-control requirements. TiCo addresses this limitation by enabling models to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is simple and efficient: it requires only a small amount of data and no additional question-answer pairs, relying instead on self-generation and reinforcement learning. Experimental results show that TiCo significantly improves adherence to duration constraints while preserving response quality.
>
---
#### [new 022] Semi-Blind Channel Estimation and Hybrid Receiver Beamforming in the Tera-Hertz Multi-User Massive MIMO Uplink
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于THz频段多用户大规模MIMO系统的信道估计与接收机设计任务，解决高训练开销和干扰问题，提出半盲信道估计和混合接收框架以提升性能。**

- **链接: [https://arxiv.org/pdf/2603.22258](https://arxiv.org/pdf/2603.22258)**

> **作者:** Abhisha Garg; Suraj Srivastava; Varsha Dubey; Aditya Jagannatham; Lajos Hanzo
>
> **摘要:** We develop a pragmatic multi-user (MU) massive multiple-input multiple-output (MIMO) channel model tailored to the THz band, encompassing factors such as molecular absorption, reflection losses and multipath diffused ray components. Next, we propose a novel semi-blind based channel state information (CSI) acquisition technique i.e. MU whitening decorrelation semi-blind (MU-WD-SB) that exploits the second order statistics corresponding to the unknown data symbols along with pilot vectors. A constrained Cramer-Rao Lower Bound (C-CRLB) is derived to bound the normalized mean square error (NMSE) performance of the proposed semi-blind learning technique. Our proposed scheme efficiently reduces the training overheads while enhancing the overall accuracy of the channel learning process. Furthermore, a novel hybrid receiver combiner framework is devised for MU THz massive MIMO systems, leveraging multiple measurement vector based sparse Bayesian learning (MMV-SBL) that relies on the estimated CSI acquired through our proposed semi-blind technique relying on low resolution analog-to-digital converters (ADCs). Finally, we propose an optimal hybrid combiner based on MMV-SBL, which directly reduces the MU interference. Extensive simulations are conducted to evaluate the performance gain of the proposed MU-WD-SB scheme over conventional training-based and other semi-blind learning techniques for a practical THz channel obtained from the high-resolution transmission (HITRAN) database. The metrics considered for quantifying the improvements include the NMSE, bit error rate (BER) and spectral-efficiency (SE).
>
---
#### [new 023] The Binding Effect: Analyzing How Multi-Dimensional Cues Form Gender Bias in Instruction TTS
- **分类: eess.SP; cs.SD**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在解决ITTS系统中因多维社会线索组合引发的性别偏见问题。工作包括建模社会线索、分析偏差模式及评估多样性提示的有效性。**

- **链接: [https://arxiv.org/pdf/2603.20743](https://arxiv.org/pdf/2603.20743)**

> **作者:** Kuan-Yu Chen; Yi-Cheng Lin; Po-Chung Hsieh; Huang-Cheng Chou; Chih-Fan Hsu; Jeng-Lin Li; Hung-yi Lee; Jian-Jiun Ding
>
> **备注:** 5 pages, 1 figure, 6 tables, Submitted to INTERSPEECH 2026
>
> **摘要:** Current bias evaluations in Instruction Text-to-Speech (ITTS) often rely on univariate testing, overlooking the compositional structure of social cues. In this work, we investigate gender bias by modeling prompts as combinations of Social Status, Career stereotypes, and Persona descriptors. Analyzing open-source ITTS models, we uncover systematic interaction effects where social dimensions modulate one another, creating complex bias patterns missed by univariate baselines. Crucially, our findings indicate that these biases extend beyond surface-level artifacts, demonstrating strong associations with the semantic priors of pre-trained text encoders and the skewed distributions inherent in training data. We further demonstrate that generic diversity prompting is insufficient to override these entrenched patterns, underscoring the need for compositional analysis to diagnose latent risks in generative speech.
>
---
#### [new 024] Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨语言语音识别任务，旨在解决帕金森病患者发音障碍检测中数据不足的问题。通过语言迁移方法调整语音表示，提升跨语言检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22225](https://arxiv.org/pdf/2603.22225)**

> **作者:** Abner Hernandez; Eunjung Yeo; Kwanghee Choi; Chin-Jou Li; Zhengjun Yue; Rohan Kumar Das; Jan Rusz; Mathew Magimai Doss; Juan Rafael Orozco-Arroyave; Tomás Arias-Vergara; Andreas Maier; Elmar Nöth; David R. Mortensen; David Harwath; Paula Andrea Perez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure.
>
---
#### [new 025] Fusing Memory and Attention: A study on LSTM, Transformer and Hybrid Architectures for Symbolic Music Generation
- **分类: cs.LG; cs.AI; cs.SD**

- **简介: 该论文属于符号音乐生成任务，旨在解决模型在局部连贯性与全局结构间的平衡问题。通过对比LSTM、Transformer及混合架构，分析其优缺点并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2603.21282](https://arxiv.org/pdf/2603.21282)**

> **作者:** Soudeep Ghoshal; Sandipan Chakraborty; Pradipto Chowdhury; Himanshu Buckchash
>
> **备注:** 20 pages, 6 figures. Published in Expert Systems with Applications (Elsevier), 2026. DOI: this https URL
>
> **摘要:** Machine learning techniques, such as Transformers and Long Short-Term Memory (LSTM) networks, play a crucial role in Symbolic Music Generation (SMG). Existing literature indicates a difference between LSTMs and Transformers regarding their ability to model local melodic continuity versus maintaining global structural coherence. However, their specific properties within the context of SMG have not been systematically studied. This paper addresses this gap by providing a fine-grained comparative analysis of LSTMs versus Transformers for SMG, examining local and global properties in detail using 17 musical quality metrics on the Deutschl dataset. We find that LSTM networks excel at capturing local patterns but fail to preserve long-range dependencies, while Transformers model global structure effectively but tend to produce irregular phrasing. Based on this analysis and leveraging their respective strengths, we propose a Hybrid architecture combining a Transformer Encoder with an LSTM Decoder and evaluate it against both baselines. We evaluated 1,000 generated melodies from each of the three architectures on the Deutschl dataset. The results show that the hybrid method achieves better local and global continuity and coherence compared to the baselines. Our work highlights the key characteristics of these models and demonstrates how their properties can be leveraged to design superior models. We also supported the experiments with ablation studies and human perceptual evaluations, which statistically support the findings and provide robust validation for this work.
>
---
#### [new 026] EARTalking: End-to-end GPT-style Autoregressive Talking Head Synthesis with Frame-wise Control
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于音频驱动的说话头生成任务，解决现有方法表达力不足、实时性差的问题。提出EARTalking模型，采用帧级控制和流式生成，提升生成质量和交互性。**

- **链接: [https://arxiv.org/pdf/2603.20307](https://arxiv.org/pdf/2603.20307)**

> **作者:** Yuzhe Weng; Haotian Wang; Yuanhong Yu; Jun Du; Shan He; Xiaoyan Wu; Haoran Xu
>
> **摘要:** Audio-driven talking head generation aims to create vivid and realistic videos from a static portrait and speech. Existing AR-based methods rely on intermediate facial representations, which limit their expressiveness and realism. Meanwhile, diffusion-based methods generate clip-by-clip, lacking fine-grained control and causing inherent latency due to overall denoising across the window. To address these limitations, we propose EARTalking, a novel end-to-end, GPT-style autoregressive model for interactive audio-driven talking head generation. Our method introduces a novel frame-by-frame, in-context, audio-driven streaming generation paradigm. For inherently supporting variable-length video generation with identity consistency, we propose the Sink Frame Window Attention (SFA) mechanism. Furthermore, to avoid the complex, separate networks that prior works required for diverse control signals, we propose a streaming Frame Condition In-Context (FCIC) scheme. This scheme efficiently injects diverse control signals in a streaming, in-context manner, enabling interactive control at every frame and at arbitrary moments. Experiments demonstrate that EARTalking outperforms existing autoregressive methods and achieves performance comparable to diffusion-based methods. Our work demonstrates the feasibility of in-context streaming autoregressive control, unlocking a scalable direction for flexible, efficient generation. The code will be released for reproducibility.
>
---
#### [new 027] Assessing the Ability of Neural TTS Systems to Model Consonant-Induced F0 Perturbation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音合成任务，旨在评估神经文本转语音系统对辅音引发的音高扰动的建模能力。通过对比合成与自然语音，发现模型在高频词表现良好，但低频词泛化能力差。**

- **链接: [https://arxiv.org/pdf/2603.21078](https://arxiv.org/pdf/2603.21078)**

> **作者:** Tianle Yang; Chengzhe Sun; Phil Rose; Cassandra L. Jacobs; Siwei Lyu
>
> **备注:** Accepted for publication in Computer Speech & Language
>
> **摘要:** This study proposes a segmental-level prosodic probing framework to evaluate neural TTS models' ability to reproduce consonant-induced f0 perturbation, a fine-grained segmental-prosodic effect that reflects local articulatory mechanisms. We compare synthetic and natural speech realizations for thousands of words, stratified by lexical frequency, using Tacotron 2 and FastSpeech 2 trained on the same speech corpus (LJ Speech). These controlled analyses are then complemented by a large-scale evaluation spanning multiple advanced TTS systems. Results show accurate reproduction for high-frequency words but poor generalization to low-frequency items, suggesting that the examined TTS architectures rely more on lexical-level memorization than on abstract segmental-prosodic encoding. This finding highlights a limitation in such TTS systems' ability to generalize prosodic detail beyond seen data. The proposed probe offers a linguistically informed diagnostic framework that may inform future TTS evaluation methods, and has implications for interpretability and authenticity assessment in synthetic speech.
>
---
## 更新

#### [replaced 001] Multi-Task Instruction Tuning via Data Scheduling for Low-Resource Arabic AudioLLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究低资源阿拉伯语音频大模型的多任务指令调优，解决语音理解与生成中的方言和情感识别问题。提出AraMega-SSum数据集及多种训练策略，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12494](https://arxiv.org/pdf/2601.12494)**

> **作者:** Hunzalah Hassan Bhatti; Firoj Alam; Shammur Absar Chowdhury
>
> **备注:** Foundation Models, Large Language Models, Native, Speech Models, Arabic
>
> **摘要:** Audio large language models (LLMs) enable unified speech understanding and generation, but adapting them to linguistically complex and dialect-rich settings such as Arabic-English remains challenging. We present a controlled study of multi-task instruction tuning for an Arabic-centric audio LLM across generative tasks including ASR and speech and text summarization, and discriminative tasks including dialect and emotion recognition, in a resource-constrained setting. To support end-to-end Arabic speech summarization, we introduce AraMega-SSum, a first speech summarization resource for training and benchmarking Arabic-centric Audio-LLMs. We compare four training strategies (i) Uniform Task Mixing, (ii) Task-Progressive Curriculum (TPC), (iiii) Aligner-Based Diverse Sampling (ADS) for training-time batch construction, and (iv) A two-stage TPC->ADS strategy. Our results show a clear efficiency-robustness trade-off. ADS speeds up early convergence and improves paralinguistic performance, however, it hurts other tasks. A two-stage TPC-> ADS strategy gives the most reliable overall balance across tasks, offering practical guidance for adapting omni audio LLMs to low-resource, dialect-rich environments. We will make AraMega-SSum and all experimental resources publicly available to the community.
>
---
#### [replaced 002] TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG
- **分类: cs.AI; cs.CL; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于抑郁症检测任务，旨在通过语音、文本和脑电（EEG）的多模态分析提升检测效果。研究比较了不同特征和模型配置，验证了预训练嵌入和融合策略的有效性。**

- **链接: [https://arxiv.org/pdf/2510.14922](https://arxiv.org/pdf/2510.14922)**

> **作者:** Annisaa Fitri Nurfidausi; Eleonora Mancini; Paolo Torroni
>
> **摘要:** Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection.
>
---
#### [replaced 003] Neural Directional Filtering Using a Compact Microphone Array
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决紧凑麦克风阵列方向性模式受限的问题。通过神经网络实现预设方向性模式的声场捕捉，提升方向性性能。**

- **链接: [https://arxiv.org/pdf/2511.07185](https://arxiv.org/pdf/2511.07185)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Mhd Modar Halimeh; Oliver Thiergart; Emanuël A. P. Habets
>
> **摘要:** Beamforming with desired directivity patterns using compact microphone arrays is essential in many audio applications. Directivity patterns achievable using traditional beamformers depend on the number of microphones and the array aperture. Generally, their effectiveness degrades for compact arrays. To overcome these limitations, we propose a neural directional filtering (NDF) approach that leverages deep neural networks to enable sound capture with a predefined directivity pattern. The NDF computes a single-channel complex mask from the microphone array signals, which is then applied to a reference microphone to produce an output that approximates a virtual directional microphone with the desired directivity pattern. We introduce training strategies and propose data-dependent metrics to evaluate the directivity pattern and directivity factor. We show that the proposed method: i) achieves a frequency-invariant directivity pattern even above the spatial aliasing frequency, ii) can approximate diverse and higher-order patterns, iii) can steer the pattern in different directions, and iv) generalizes to unseen conditions. Lastly, experimental comparisons demonstrate superior performance over conventional beamforming and parametric approaches.
>
---
#### [replaced 004] Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor
- **分类: eess.AS**

- **简介: 该论文属于后门攻击任务，旨在解决音频水印在数据中毒中的隐蔽性与效果问题。提出Bloodroot框架，通过对抗性LoRA微调实现高效且隐蔽的后门攻击。**

- **链接: [https://arxiv.org/pdf/2510.07909](https://arxiv.org/pdf/2510.07909)**

> **作者:** Kuan-Yu Chen; Yi-Cheng Lin; Jeng-Lin Li; Jian-Jiun Ding
>
> **备注:** 5 pages, 3 figures accepted to ICASSP 2026
>
> **摘要:** Backdoor data poisoning is a crucial technique for ownership protection and defending against malicious attacks. Embedding hidden triggers in training data can manipulate model outputs, enabling provenance verification, and deterring unauthorized use. However, current audio backdoor methods are suboptimal, as poisoned audio often exhibits degraded perceptual quality, which is noticeable to human listeners. This work explores the intrinsic stealthiness and effectiveness of audio watermarking in achieving successful poisoning. We propose a novel Watermark-as-Trigger concept, integrated into the Bloodroot backdoor framework via adversarial LoRA fine-tuning, which enhances perceptual quality while achieving a much higher trigger success rate and clean-sample accuracy. Experiments on speech recognition (SR) and speaker identification (SID) datasets show that watermark-based poisoning remains effective under acoustic filtering and model pruning. The proposed Bloodroot backdoor framework not only secures data-to-model ownership, but also well reveals the risk of adversarial misuse.
>
---
#### [replaced 005] Preliminary sonification of ENSO using traditional Javanese gamelan scales
- **分类: physics.soc-ph; cs.SD; physics.ao-ph**

- **简介: 该论文属于数据声学化任务，旨在将ENSO气候数据转化为音乐形式，通过分析音频轨迹验证其动力学特征。**

- **链接: [https://arxiv.org/pdf/2602.14560](https://arxiv.org/pdf/2602.14560)**

> **作者:** Sandy Hardian Susanto Herho; Rusmawan Suwarman; Nurjanna Joko Trilaksono; Iwan Pramesti Anwar; Faiz Rohman Fajary
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Sonification -- the mapping of data to non-speech audio -- offers an underexplored channel for representing complex dynamical systems. We treat El Niño-Southern Oscillation (ENSO), a canonical example of low-dimensional climate chaos, as a test case for culturally-situated sonification evaluated through complex systems diagnostics. Using parameter-mapping sonification of the Niño 3.4 sea surface temperature anomaly index (1870--2024), we encode ENSO variability into two traditional Javanese gamelan pentatonic systems (pelog and slendro) across four composition strategies, then analyze the resulting audio as trajectories in a two-dimensional acoustic phase space. Recurrence-based diagnostics, convex hull geometry, and coupling analysis reveal that the sonification pipeline preserves key dynamical signatures: alternating modes produce the highest trajectory recurrence rates, echoing ENSO's quasi-periodicity; layered polyphonic modes explore the broadest phase space regions; and the two scale families induce qualitatively distinct coupling regimes between spectral brightness and energy -- predominantly anti-phase in pelog but near-independent in slendro. Phase space trajectory analysis provides a rigorous geometric framework for comparing sonification designs within a complex systems context. Perceptual validation remains necessary; we contribute the dynamical systems methodology for evaluating such mappings.
>
---
#### [replaced 006] VorTEX: Various overlap ratio for Target speech EXtraction
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音分离任务，解决真实场景下不同重叠比例的语音提取问题。提出VorTEX模型和PORTE数据集，提升分离效果并避免抑制现象。**

- **链接: [https://arxiv.org/pdf/2603.14803](https://arxiv.org/pdf/2603.14803)**

> **作者:** Ro-hoon Oh; Jihwan Seol; Bugeun Kim
>
> **备注:** Submitted to InterSpeech 2026 (under review)
>
> **摘要:** Target speech extraction (TSE) aims to recover a target speaker's voice from a mixture. While recent text-prompted approaches have shown promise, most approaches assume fully overlapped mixtures, limiting insight into behavior across realistic overlap ratios. We introduce VorTEX (Various overlap ratio for Target speech EXtraction), a text-prompted TSE architecture with a Decoupled Adaptive Multi-branch (DAM) Fusion block that separates primary extraction from auxiliary regularization pathways. To enable controlled analysis, we construct PORTE, a two-speaker dataset spanning overlap ratios from 0% to 100%. We further propose Suppression Ratio on Energy (SuRE), a diagnostic metric that detects suppression behavior not captured by conventional measures. Experiments show that existing models exhibit suppression or residual interference under overlap, whereas VorTEX achieves the highest separation fidelity across 20-100% overlap (e.g., 5.50 dB at 20% and 2.04 dB at 100%) while maintaining zero SuRE, indicating robust extraction without suppression-driven artifacts.
>
---
#### [replaced 007] CALM: Class-Conditional Sparse Attention Vectors for Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出CALM方法，用于音频-语言模型的分类任务，解决传统方法中注意力头权重均等的问题，通过学习类别相关的权重提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.07077](https://arxiv.org/pdf/2602.07077)**

> **作者:** Videet Mehta; Liming Wang; Hilde Kuehne; Rogerio Feris; James R. Glass; M. Jehanzeb Mirza
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Large audio-language models (LALMs) exhibit strong zero-shot capabilities in multiple downstream tasks, such as audio question answering (AQA) and abstract reasoning; however, these models still lag behind specialized models for certain discriminative tasks (e.g., audio classification). Recent studies show that sparse subsets of attention heads within an LALM can serve as strong discriminative feature extractors for downstream tasks such as classification via simple voting schemes. However, these methods assign uniform weights to all selected heads, implicitly assuming that each head contributes equally across all semantic categories. In this work, we propose Class-Conditional Sparse Attention Vectors for Large Audio-Language Models, a few-shot classification method that learns class-dependent importance weights over attention heads. This formulation allows individual heads to specialize in distinct semantic categories and to contribute to ensemble predictions proportionally to their estimated reliability. Experiments on multiple few-shot audio and audiovisual classification benchmarks and tasks demonstrate that our method consistently outperforms state-of-the-art uniform voting-based approaches by up to 14.52%, 1.53%, 8.35% absolute gains for audio classification, audio-visual classification, and spoofing detection respectively.
>
---
#### [replaced 008] A Multimodal Data Fusion Generative Adversarial Network for Real Time Underwater Sound Speed Field Construction
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于水声参数建模任务，旨在无需现场数据实现高精度声速剖面估计。提出MDF-RAGAN模型，融合多源数据并引入注意力机制，提升建模精度。**

- **链接: [https://arxiv.org/pdf/2507.11812](https://arxiv.org/pdf/2507.11812)**

> **作者:** Wei Huang; Yuqiang Huang; Yanan Wu; Tianhe Xu; Tingting Lyu; Hao Zhang
>
> **摘要:** Sound speed profiles (SSPs) are essential parameters underwater that affects the propagation mode of underwater signals and has a critical impact on the energy efficiency of underwater acoustic communication and accuracy of underwater acoustic positioning. Traditionally, SSPs can be obtained by matching field processing (MFP), compressive sensing (CS), and deep learning (DL) methods. However, existing methods mainly rely on on-site underwater sonar observation data, which put forward strict requirements on the deployment of sonar observation systems. To achieve high-precision estimation of sound velocity distribution in a given sea area without on-site underwater data measurement, we propose a multi-modal data-fusion generative adversarial network model with residual attention block (MDF-RAGAN) for SSP construction. To improve the model's ability for capturing global spatial feature correlations, we embedded the attention mechanisms, and use residual modules for deeply capturing small disturbances in the deep ocean sound velocity distribution caused by changes of SST. Experimental results on real open dataset show that the proposed model outperforms other state-of-the-art methods, which achieves an accuracy with an error of less than 0.3m/s. Specifically, MDF-RAGAN not only outperforms convolutional neural network (CNN) and spatial interpolation (SITP) by nearly a factor of two, but also achieves about 65.8\% root mean square error (RMSE) reduction compared to mean profile, which fully reflects the enhancement of overall profile matching by multi-source fusion and cross-modal attention.
>
---
