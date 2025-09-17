# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] Can Large Audio Language Models Understand Audio Well? Speech, Scene and Events Understanding Benchmark for LALMs
- **分类: cs.SD**

- **简介: 该论文提出SSEU-Bench基准，用于评估大音频语言模型对语音、场景和事件的理解能力。针对现有基准忽略能量差异和联合理解的问题，设计独立与联合任务设置，并引入Chain-of-Thought方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2509.13148v1](http://arxiv.org/pdf/2509.13148v1)**

> **作者:** Han Yin; Jung-Woo Choi
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Recently, Large Audio Language Models (LALMs) have progressed rapidly, demonstrating their strong efficacy in universal audio understanding through cross-modal integration. To evaluate the LALM's audio understanding performance, researchers have proposed different benchmarks. However, key aspects for real-world interactions are underexplored in existing benchmarks, i.e., audio signals typically contain both speech and non-speech components, and energy levels of these components can vary significantly across different scenarios. Moreover, most benchmarks do not consider the joint understanding of speech, scene, and events within the same audio clip. In this work, we introduce SSEU-Bench, the first versatile audio understanding benchmark that explicitly accounts for energy differences between speech and non-speech audio, with both independent and joint understanding settings for speech, scene, and events. Furthermore, we demonstrate that some LALMs tend to underperform on certain tasks in a joint understanding setting. To address this issue, we introduce Chain-of-Thought, which effectively improves the LALM's joint audio understanding performance by decomposing complex tasks into simpler reasoning steps
>
---
#### [new 002] Contrastive timbre representations for musical instrument and synthesizer retrieval
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出对比学习框架，用于从音频中检索乐器音色，解决多乐器混音中高效检索特定乐器音色的难题。通过生成正负样本对，提升模型性能，在单/多乐器场景下均取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.13285v1](http://arxiv.org/pdf/2509.13285v1)**

> **作者:** Gwendal Le Vaillant; Yannick Molle
>
> **摘要:** Efficiently retrieving specific instrument timbres from audio mixtures remains a challenge in digital music production. This paper introduces a contrastive learning framework for musical instrument retrieval, enabling direct querying of instrument databases using a single model for both single- and multi-instrument sounds. We propose techniques to generate realistic positive/negative pairs of sounds for virtual musical instruments, such as samplers and synthesizers, addressing limitations in common audio data augmentation methods. The first experiment focuses on instrument retrieval from a dataset of 3,884 instruments, using single-instrument audio as input. Contrastive approaches are competitive with previous works based on classification pre-training. The second experiment considers multi-instrument retrieval with a mixture of instruments as audio input. In this case, the proposed contrastive framework outperforms related works, achieving 81.7\% top-1 and 95.7\% top-5 accuracies for three-instrument mixtures.
>
---
#### [new 003] Timbre-Adaptive Transcription: A Lightweight Architecture with Associative Memory for Dynamic Instrument Separation
- **分类: cs.SD; cs.IR**

- **简介: 论文提出一种轻量级多音色转录框架，解决现有模型泛化能力差和音源数量固定的问题。通过时域无关主干网络和联想记忆机制，实现对未见音色的动态分离，仅需少量训练数据即可高效完成多音色分离任务。**

- **链接: [http://arxiv.org/pdf/2509.12712v1](http://arxiv.org/pdf/2509.12712v1)**

> **作者:** Ruigang Li; Yongxu Zhu
>
> **摘要:** Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments and rigid source-count constraints. We address these limitations with a lightweight deep clustering solution featuring: 1) a timbre-agnostic backbone achieving state-of-the-art performance with only half the parameters of comparable models, and 2) a novel associative memory mechanism that mimics human auditory cognition to dynamically encode unseen timbres via attention-based clustering. Our biologically-inspired framework enables adaptive polyphonic separation with minimal training data (12.5 minutes), supported by a new synthetic dataset method offering cost-effective, high-precision multi-timbre generation. Experiments show the timbre-agnostic transcription model outperforms existing models on public benchmarks, while the separation module demonstrates promising timbre discrimination. This work provides an efficient framework for timbre-related music transcription and explores new directions for timbre-aware separation through cognitive-inspired architectures.
>
---
#### [new 004] Improving Anomalous Sound Detection with Attribute-aware Representation from Domain-adaptive Pre-training
- **分类: cs.SD; cs.AI**

- **简介: 论文提出一种基于领域自适应预训练的属性感知表示方法，用于改进异常声音检测任务。该方法通过聚类分配伪属性标签，解决属性标签缺失问题，最终在DCASE 2025数据集上取得新SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.12845v1](http://arxiv.org/pdf/2509.12845v1)**

> **作者:** Xin Fang; Guirui Zhong; Qing Wang; Fan Chu; Lei Wang; Mengui Qian; Mingqi Cai; Jiangzhao Wu; Jianqing Gao; Jun Du
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Anomalous Sound Detection (ASD) is often formulated as a machine attribute classification task, a strategy necessitated by the common scenario where only normal data is available for training. However, the exhaustive collection of machine attribute labels is laborious and impractical. To address the challenge of missing attribute labels, this paper proposes an agglomerative hierarchical clustering method for the assignment of pseudo-attribute labels using representations derived from a domain-adaptive pre-trained model, which are expected to capture machine attribute characteristics. We then apply model adaptation to this pre-trained model through supervised fine-tuning for machine attribute classification, resulting in a new state-of-the-art performance. Evaluation on the Detection and Classification of Acoustic Scenes and Events (DCASE) 2025 Challenge dataset demonstrates that our proposed approach yields significant performance gains, ultimately outperforming our previous top-ranking system in the challenge.
>
---
#### [new 005] More Similar than Dissimilar: Modeling Annotators for Cross-Corpus Speech Emotion Recognition
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 论文研究跨语料库语音情感识别中注释员建模问题。传统方法预测共识值，难以预测单个注释员标注。本文提出利用注释员间相似性，通过预训练模型找到相似注释员，实现低成本个性化预测，提升新注释员数据的标注效果。**

- **链接: [http://arxiv.org/pdf/2509.12295v1](http://arxiv.org/pdf/2509.12295v1)**

> **作者:** James Tavernor; Emily Mower Provost
>
> **备注:** \copyright 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Speech emotion recognition systems often predict a consensus value generated from the ratings of multiple annotators. However, these models have limited ability to predict the annotation of any one person. Alternatively, models can learn to predict the annotations of all annotators. Adapting such models to new annotators is difficult as new annotators must individually provide sufficient labeled training data. We propose to leverage inter-annotator similarity by using a model pre-trained on a large annotator population to identify a similar, previously seen annotator. Given a new, previously unseen, annotator and limited enrollment data, we can make predictions for a similar annotator, enabling off-the-shelf annotation of unseen data in target datasets, providing a mechanism for extremely low-cost personalization. We demonstrate our approach significantly outperforms other off-the-shelf approaches, paving the way for lightweight emotion adaptation, practical for real-world deployment.
>
---
#### [new 006] Osu2MIR: Beat Tracking Dataset Derived From Osu! Data
- **分类: cs.SD**

- **简介: 该论文属于音乐信息检索（MIR）任务，旨在利用Osu!游戏数据构建节拍追踪数据集。通过提取和筛选Osu! beatmaps中的节拍注释，提出一种可扩展的数据生成方法，并发布高质量数据子集osu2beat2025以支持相关研究。**

- **链接: [http://arxiv.org/pdf/2509.12667v1](http://arxiv.org/pdf/2509.12667v1)**

> **作者:** Ziyun Liu; Chris Donahue
>
> **备注:** 2 pages
>
> **摘要:** In this work, we explore the use of Osu!, a community-based rhythm game, as an alternative source of beat and downbeat annotations. Osu! beatmaps are created and refined by a large, diverse community and span underrepresented genres such as anime, Vocaloid, and video game music. We introduce a pipeline for extracting annotations from Osu! beatmaps and partition them into meaningful subsets. Through manual analysis, we find that beatmaps with a single timing point or widely spaced multiple timing points (>=5 seconds apart) provide reliable annotations, while closely spaced timing points (<5 seconds apart) often require additional curation. We also observe high consistency across multiple annotations of the same song. This study demonstrates the potential of Osu! data as a scalable, diverse, and community-driven resource for MIR research. We release our pipeline and a high-quality subset osu2beat2025 to support further exploration: https://github.com/ziyunliu4444/osu2mir.
>
---
#### [new 007] The CCF AATC 2025: Speech Restoration Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文介绍CCF AATC 2025语音修复挑战赛，旨在解决现实场景中多种语音失真共存的问题。任务是恢复受复杂噪声、混响、压缩及预处理模型影响的语音信号，提出综合数据集与评估方法，推动相关研究。**

- **链接: [http://arxiv.org/pdf/2509.12974v1](http://arxiv.org/pdf/2509.12974v1)**

> **作者:** Junan Zhang; Mengyao Zhu; Xin Xu; Hui Bu; Zhenhua Ling; Zhizheng Wu
>
> **备注:** Technical Report
>
> **摘要:** Real-world speech communication is often hampered by a variety of distortions that degrade quality and intelligibility. While many speech enhancement algorithms target specific degradations like noise or reverberation, they often fall short in realistic scenarios where multiple distortions co-exist and interact. To spur research in this area, we introduce the Speech Restoration Challenge as part of the China Computer Federation (CCF) Advanced Audio Technology Competition (AATC) 2025. This challenge focuses on restoring speech signals affected by a composite of three degradation types: (1) complex acoustic degradations including non-stationary noise and reverberation; (2) signal-chain artifacts such as those from MP3 compression; and (3) secondary artifacts introduced by other pre-processing enhancement models. We describe the challenge's background, the design of the task, the comprehensive dataset creation methodology, and the detailed evaluation protocol, which assesses both objective performance and model complexity. Homepage: https://ccf-aatc.org.cn/.
>
---
#### [new 008] A Traditional Approach to Symbolic Piano Continuation
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 论文提出一种传统方法用于符号化钢琴音乐续写任务，旨在解决单乐器音乐生成问题。通过简单地使用未增强的下一个token预测目标，在MIREX 2025挑战中竞争，强调数据与基础模型的重要性。**

- **链接: [http://arxiv.org/pdf/2509.12267v1](http://arxiv.org/pdf/2509.12267v1)**

> **作者:** Christian Zhou-Zheng; John Backsund; Dun Li Chan; Alex Coventry; Avid Eslami; Jyotin Goel; Xingwen Han; Danysh Soomro; Galen Wei
>
> **备注:** 3 pages, extended abstract, MIREX session at ISMIR 2025 LBD
>
> **摘要:** We present a traditional approach to symbolic piano music continuation for the MIREX 2025 Symbolic Music Generation challenge. While computational music generation has recently focused on developing large foundation models with sophisticated architectural modifications, we argue that simpler approaches remain more effective for constrained, single-instrument tasks. We thus return to a simple, unaugmented next-token-prediction objective on tokenized raw MIDI, aiming to outperform large foundation models by using better data and better fundamentals. We release model weights and code at https://github.com/christianazinn/mirex2025.
>
---
#### [new 009] A Lightweight Pipeline for Noisy Speech Voice Cloning and Accurate Lip Sync Synthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出一种轻量级语音克隆与唇形同步合成流水线，解决噪声环境下依赖大规模数据集和计算资源的问题。采用基于Transformer的扩散模型实现高保真零样本语音克隆，并用轻量GAN实现实时唇形同步，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2509.12831v1](http://arxiv.org/pdf/2509.12831v1)**

> **作者:** Javeria Amir; Farwa Attaria; Mah Jabeen; Umara Noor; Zahid Rashid
>
> **摘要:** Recent developments in voice cloning and talking head generation demonstrate impressive capabilities in synthesizing natural speech and realistic lip synchronization. Current methods typically require and are trained on large scale datasets and computationally intensive processes using clean studio recorded inputs that is infeasible in noisy or low resource environments. In this paper, we introduce a new modular pipeline comprising Tortoise text to speech. It is a transformer based latent diffusion model that can perform high fidelity zero shot voice cloning given only a few training samples. We use a lightweight generative adversarial network architecture for robust real time lip synchronization. The solution will contribute to many essential tasks concerning less reliance on massive pre training generation of emotionally expressive speech and lip synchronization in noisy and unconstrained scenarios. The modular structure of the pipeline allows an easy extension for future multi modal and text guided voice modulation and it could be used in real world systems.
>
---
#### [new 010] GLAD: Global-Local Aware Dynamic Mixture-of-Experts for Multi-Talker ASR
- **分类: cs.SD**

- **简介: 论文提出GLAD模型，用于多说话人语音识别（MTASR）任务，解决高重叠语音转录难题。通过融合全局与局部信息动态选择专家，提升识别准确率，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13093v1](http://arxiv.org/pdf/2509.13093v1)**

> **作者:** Yujie Guo; Jiaming Zhou; Yuhang Jia; Shiwan Zhao; Yong Qin
>
> **摘要:** End-to-end multi-talker automatic speech recognition (MTASR) faces significant challenges in accurately transcribing overlapping speech, especially under high-overlap conditions. To address these challenges, we proposed Global-Local Aware Dynamic (GLAD) Mixture-of-Experts, which dynamically fuse speaker-aware global information and fine-grained local features to guide expert selection. This mechanism enables speaker-specific routing by leveraging both global context and local acoustic cues. Experiments on LibriSpeechMix show that GLAD outperforms existing MTASR approaches, particularly in challenging multi-talker scenarios. To our best knowledge, this is the first work to apply Mixture-of-Experts (MoE) to end-to-end MTASR with a global-local fusion strategy. Our code and train dataset can be found at https://github.com/NKU-HLT/GLAD.
>
---
#### [new 011] Omni-CLST: Error-aware Curriculum Learning with guided Selective chain-of-Thought for audio questuin answering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出Omni-CLST框架，用于音频问答任务。通过错误感知课程学习和引导式思维丢弃机制，提升模型在多模态音频-语言理解中的准确性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.12275v1](http://arxiv.org/pdf/2509.12275v1)**

> **作者:** Jinghua Zhao; Hang Su; Lichun Fan; Zhenbo Luo; Jian Luan; Hui Wang; Haoqin Sun; Yong Qin
>
> **备注:** 5 pages, 1 figure, 2 tables
>
> **摘要:** We propose Omni-CLST, an error-aware Curriculum Learning framework with guided Selective Chain-of-Thought for audio question answering. The framework efficiently leverages existing high-quality dataset through two key strategies: an error-aware curriculum that organizes samples by difficulty, and a guided thought dropout mechanism that focuses reasoning on challenging cases. Integrated with GRPO training, these strategies enable the model to learn more effectively from informative samples. Experiments on MMAU-mini and MMAR demonstrate that Omni-CLST achieves competitive accuracy (73.80% on MMAU-mini) and establishes a new state of the art (64.30% on MMAR), highlighting its robustness and generalization capability in multimodal audio-language understanding.
>
---
#### [new 012] UTI-LLM: A Personalized Articulatory-Speech Therapy Assistance System Based on Multimodal Large Language Model
- **分类: cs.SD**

- **简介: 该论文提出基于多模态大语言模型的个性化言语治疗系统UTI-LLM，解决传统言语康复系统反馈不足的问题。通过融合超声舌像与语音信号，构建高质量数据集并实现时空融合训练，提供精准互动反馈，提升临床适应性。**

- **链接: [http://arxiv.org/pdf/2509.13145v1](http://arxiv.org/pdf/2509.13145v1)**

> **作者:** Yudong Yang; Xiaokang Liu; Shaofeng zhao; Rongfeng Su; Nan Yan; Lan Wang
>
> **摘要:** Speech therapy plays a critical role in training speech disorders caused by neurological impairments such as stroke. However, traditional manual and computer-assisted systems are limited in real-time accessibility and articulatory motion feedback, constraining their practical utility. Recent advances in multimodal large language models (MLLMs) have demonstrated significant potential in healthcare, particularly through their ability to integrate multimodal data for adaptive assessment and therapeutic feedback. Nevertheless, challenges including insufficient acquisition and fusion of articulatory information, inadequate parsing of articulatory organ motion trajectories, and the scarcity of high-quality domain-specific datasets hinder the application of MLLMs in speech therapy. To address these limitations, we propose an MLLM-based speech rehabilitation assistance system that synergistically leverages ultrasound tongue imaging and speech signals to deliver precise, interactive articulatory feedback. We construct a high-quality domain-specific dataset comprising UTI-speech dialogue pairs. This dataset facilitates fine-tuning to enhance the model's clinical adaptability. Building on this dataset, our methods achieves spatiotemporal fusion training strategy of ultrasound videos and speech signals, enabling fine-grained articulatory impairment analysis and ultimately generating actionable feedback.
>
---
#### [new 013] Beyond Bars: Distribution of Edit Operations in Historical Prints
- **分类: cs.SD**

- **简介: 该论文提出一种音乐学语料库比较研究方法，通过抽样乐谱小节减少数字化工作量。研究评估三种抽样方法，以贝多芬作品为例，旨在提高大规模分析效率与统计可靠性，促进对19世纪编辑实践的理解。**

- **链接: [http://arxiv.org/pdf/2509.12786v1](http://arxiv.org/pdf/2509.12786v1)**

> **作者:** Adrian Nachtwey; Fabian C. Moss; Anna Viktoria Katrin Plaksin
>
> **摘要:** In this paper, we present a method for conducting comparative corpus studies in musicology that reduces the time-consuming digitization process. Instead of encoding whole corpora of musical sources, we suggest sampling bars from these sources. We address the challenge of selecting representative samples and evaluate three different sampling methods. We used Beethoven's Bagatelles Op. 33 as a case study to find the method that works best in finding samples representative with respect to differences. We believe that this approach offers significant value to musicological research by enabling large-scale analyses and thereby statistically sound results. Moreover, we believe our work to be a valuable step toward understanding nineteenth-century editorial practices and enriching the field of scholarly editing of historical musical works.
>
---
#### [new 014] An Adaptive CMSA for Solving the Longest Filled Common Subsequence Problem with an Application in Audio Querying
- **分类: cs.SD; eess.AS**

- **简介: 论文提出一种自适应CMSA框架，用于高效求解NP难的最长填充公共子序列（LFCS）问题，并在音频查询中应用。通过新基准数据集验证其优越性能，实现99.9%最优解质量，同时揭示影响算法表现的关键特征。**

- **链接: [http://arxiv.org/pdf/2509.12261v1](http://arxiv.org/pdf/2509.12261v1)**

> **作者:** Marko Djukanovic; Christian Blum; Aleksandar Kartelj; Ana Nikolikj; Guenther Raidl
>
> **摘要:** This paper addresses the Longest Filled Common Subsequence (LFCS) problem, a challenging NP-hard problem with applications in bioinformatics, including gene mutation prediction and genomic data reconstruction. Existing approaches, including exact, metaheuristic, and approximation algorithms, have primarily been evaluated on small-sized instances, which offer limited insights into their scalability. In this work, we introduce a new benchmark dataset with significantly larger instances and demonstrate that existing datasets lack the discriminative power needed to meaningfully assess algorithm performance at scale. To solve large instances efficiently, we utilize an adaptive Construct, Merge, Solve, Adapt (CMSA) framework that iteratively generates promising subproblems via component-based construction and refines them using feedback from prior iterations. Subproblems are solved using an external black-box solver. Extensive experiments on both standard and newly introduced benchmarks show that the proposed adaptive CMSA achieves state-of-the-art performance, outperforming five leading methods. Notably, on 1,510 problem instances with known optimal solutions, our approach solves 1,486 of them -- achieving over 99.9% optimal solution quality and demonstrating exceptional scalability. We additionally propose a novel application of LFCS for song identification from degraded audio excerpts as an engineering contribution, using real-world energy-profile instances from popular music. Finally, we conducted an empirical explainability analysis to identify critical feature combinations influencing algorithm performance, i.e., the key problem features contributing to success or failure of the approaches across different instance types are revealed.
>
---
#### [new 015] FunAudio-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出FunAudio-ASR，一种结合大规模数据、大模型与LLM的ASR系统，通过强化学习优化，解决LLM幻觉问题，提升实际应用中的识别性能与鲁棒性。属于语音识别任务。**

- **链接: [http://arxiv.org/pdf/2509.12508v1](http://arxiv.org/pdf/2509.12508v1)**

> **作者:** Keyu An; Yanni Chen; Chong Deng; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Wen Wang; Wupeng Wang; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings.
>
---
#### [new 016] Multi-Modal Embedding-based Target Speaker Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 论文研究多模态融合策略在目标说话人增强中的鲁棒性，解决真实场景中模态丢失问题。提出结合唇、声纹、人脸和表情嵌入的方法，并通过高比例模态丢弃训练提升模型可靠性。**

- **链接: [http://arxiv.org/pdf/2509.12583v1](http://arxiv.org/pdf/2509.12583v1)**

> **作者:** Zhan Jin
>
> **摘要:** Target Speaker Extraction (TSE) is a critical challenge in cocktail party scenarios. While leveraging multiple modalities, such as voice, lip, face, and expression embeddings, can enhance performance, real-world applications often suffer from intermittent modality dropout. This paper presents a comprehensive study on the interactions and robustness of various multimodal fusion strategies under varying degrees of modality dropout. We build upon a state-of-the-art audio-visual speech enhancement system and integrate four distinct speaker identity cues: lip embeddings for synchronized contextual information, a voice speaker embedding extracted via cross-attention for acoustic consistency, a static face embedding for speaker identity, and a novel dynamic expression embedding for frame-wise emotional features. We systematically evaluate different combinations of these modalities under two key training regimes: zero dropout and 80% modality dropout. Extensive experiments demonstrate that while a full multimodal ensemble achieves optimal performance under ideal (zero dropout) conditions, its effectiveness diminishes significantly when test-time dropout occurs without prior exposure during training. Crucially, we show that training with a high (80%) modality dropout rate dramatically enhances model robustness, enabling the system to maintain superior performance even under severe test-time missing modalities. Our findings highlight that voice embeddings exhibit consistent robustness, while the proposed expression embedding provides valuable complementary information. This work underscores the importance of training strategies that account for real-world imperfection, moving beyond pure performance maximization to achieve practical reliability in multimodal speech enhancement systems.
>
---
#### [new 017] MoiréTac: A Dual-Mode Visuotactile Sensor for Multidimensional Perception Using Moiré Pattern Amplification
- **分类: cs.RO; eess.SP**

- **简介: 论文提出MoiréTac，一种双模式视觉触觉传感器，通过莫尔条纹放大微变形，实现高精度6轴力/扭矩测量与视觉感知。解决传统传感器分辨率低、力-图像关系不明确的问题，应用于机器人灵巧操作。**

- **链接: [http://arxiv.org/pdf/2509.12714v1](http://arxiv.org/pdf/2509.12714v1)**

> **作者:** Kit-Wa Sou; Junhao Gong; Shoujie Li; Chuqiao Lyu; Ziwu Song; Shilong Mu; Wenbo Ding
>
> **摘要:** Visuotactile sensors typically employ sparse marker arrays that limit spatial resolution and lack clear analytical force-to-image relationships. To solve this problem, we present \textbf{Moir\'eTac}, a dual-mode sensor that generates dense interference patterns via overlapping micro-gratings within a transparent architecture. When two gratings overlap with misalignment, they create moir\'e patterns that amplify microscopic deformations. The design preserves optical clarity for vision tasks while producing continuous moir\'e fields for tactile sensing, enabling simultaneous 6-axis force/torque measurement, contact localization, and visual perception. We combine physics-based features (brightness, phase gradient, orientation, and period) from moir\'e patterns with deep spatial features. These are mapped to 6-axis force/torque measurements, enabling interpretable regression through end-to-end learning. Experimental results demonstrate three capabilities: force/torque measurement with R^2 > 0.98 across tested axes; sensitivity tuning through geometric parameters (threefold gain adjustment); and vision functionality for object classification despite moir\'e overlay. Finally, we integrate the sensor into a robotic arm for cap removal with coordinated force and torque control, validating its potential for dexterous manipulation.
>
---
## 更新

#### [replaced 001] SuPreME: A Supervised Pre-training Framework for Multimodal ECG Representation Learning
- **分类: eess.SP; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19668v3](http://arxiv.org/pdf/2502.19668v3)**

> **作者:** Mingsheng Cai; Jiuming Jiang; Wenhao Huang; Che Liu; Rossella Arcucci
>
> **备注:** Findings of The 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP 2025)
>
> **摘要:** Cardiovascular diseases are a leading cause of death and disability worldwide. Electrocardiogram (ECG) is critical for diagnosing and monitoring cardiac health, but obtaining large-scale annotated ECG datasets is labor-intensive and time-consuming. Recent ECG Self-Supervised Learning (eSSL) methods mitigate this by learning features without extensive labels but fail to capture fine-grained clinical semantics and require extensive task-specific fine-tuning. To address these challenges, we propose $\textbf{SuPreME}$, a $\textbf{Su}$pervised $\textbf{Pre}$-training framework for $\textbf{M}$ultimodal $\textbf{E}$CG representation learning. SuPreME is pre-trained using structured diagnostic labels derived from ECG report entities through a one-time offline extraction with Large Language Models (LLMs), which help denoise, standardize cardiac concepts, and improve clinical representation learning. By fusing ECG signals with textual cardiac queries instead of fixed labels, SuPreME enables zero-shot classification of unseen conditions without further fine-tuning. We evaluate SuPreME on six downstream datasets covering 106 cardiac conditions, achieving superior zero-shot AUC performance of $77.20\%$, surpassing state-of-the-art eSSLs by $4.98\%$. Results demonstrate SuPreME's effectiveness in leveraging structured, clinically relevant knowledge for high-quality ECG representations.
>
---
#### [replaced 002] Quality Assessment of Noisy and Enhanced Speech with Limited Data: UWB-NTIS System for VoiceMOS 2024
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.00506v3](http://arxiv.org/pdf/2506.00506v3)**

> **作者:** Marie Kunešová; Aleš Pražák; Jan Lehečka
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** We present a system for non-intrusive prediction of speech quality in noisy and enhanced speech, developed for Track 3 of the VoiceMOS 2024 Challenge. The task required estimating the ITU-T P.835 metrics SIG, BAK, and OVRL without reference signals and with only 100 subjectively labeled utterances for training. Our approach uses wav2vec 2.0 with a two-stage transfer learning strategy: initial fine-tuning on automatically labeled noisy data, followed by adaptation to the challenge data. The system achieved the best performance on BAK prediction (LCC=0.867) and a very close second place in OVRL (LCC=0.711) in the official evaluation. Post-challenge experiments show that adding artificially degraded data to the first fine-tuning stage substantially improves SIG prediction, raising correlation with ground truth scores from 0.207 to 0.516. These results demonstrate that transfer learning with targeted data generation is effective for predicting P.835 scores under severe data constraints.
>
---
#### [replaced 003] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.05983v2](http://arxiv.org/pdf/2509.05983v2)**

> **作者:** Minh N. H. Nguyen; Anh Nguyen Tran; Dung Truong Dinh; Nam Van Vo
>
> **备注:** I need to withdraw the paper as there something wrong
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the subtle phonological shifts inherent in CS scenarios. The challenge is particularly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). The TSPC employs a phoneme-centric approach, built upon an extended Vietnamese phoneme set as an intermediate representation to facilitate mixed-lingual modeling. Experimental results demonstrate that TSPC consistently outperforms existing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 20.8\% with reduced training resources. Furthermore, the phonetic-based two-stage architecture enables phoneme adaptation and language conversion to enhance ASR performance in complex CS Vietnamese-English ASR scenarios.
>
---
#### [replaced 004] SwinSRGAN: Swin Transformer-based Generative Adversarial Network for High-Fidelity Speech Super-Resolution
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.03913v2](http://arxiv.org/pdf/2509.03913v2)**

> **作者:** Jiajun Yuan; Xiaochen Wang; Yuhang Xiao; Yulin Wu; Chenhao Hu; Xueyang Lv
>
> **备注:** 5 pages This work has been submitted to the IEEE for possible publication
>
> **摘要:** Speech super-resolution (SR) reconstructs high-frequency content from low-resolution speech signals. Existing systems often suffer from representation mismatch in two-stage mel-vocoder pipelines and from over-smoothing of hallucinated high-band content by CNN-only generators. Diffusion and flow models are computationally expensive, and their robustness across domains and sampling rates remains limited. We propose SwinSRGAN, an end-to-end framework operating on Modified Discrete Cosine Transform (MDCT) magnitudes. It is a Swin Transformer-based U-Net that captures long-range spectro-temporal dependencies with a hybrid adversarial scheme combines time-domain MPD/MSD discriminators with a multi-band MDCT discriminator specialized for the high-frequency band. We employs a sparse-aware regularizer on arcsinh-compressed MDCT to better preserve transient components. The system upsamples inputs at various sampling rates to 48 kHz in a single pass and operates in real time. On standard benchmarks, SwinSRGAN reduces objective error and improves ABX preference scores. In zero-shot tests on HiFi-TTS without fine-tuning, it outperforms NVSR and mdctGAN, demonstrating strong generalization across datasets
>
---
