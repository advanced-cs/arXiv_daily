# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] XAI-Driven Spectral Analysis of Cough Sounds for Respiratory Disease Characterization
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文通过XAI技术分析咳嗽声音频谱，旨在提升呼吸疾病诊断的可解释性。采用遮挡图突出关键频谱区域，揭示COPD等疾病在特定频段的差异，从而提取疾病特异性声学特征，增强咳嗽声分析的诊断能力。**

- **链接: [http://arxiv.org/pdf/2508.14949v1](http://arxiv.org/pdf/2508.14949v1)**

> **作者:** Patricia Amado-Caballero; Luis Miguel San-José-Revuelta; María Dolores Aguilar-García; José Ramón Garmendia-Leiza; Carlos Alberola-López; Pablo Casaseca-de-la-Higuera
>
> **摘要:** This paper proposes an eXplainable Artificial Intelligence (XAI)-driven methodology to enhance the understanding of cough sound analysis for respiratory disease management. We employ occlusion maps to highlight relevant spectral regions in cough spectrograms processed by a Convolutional Neural Network (CNN). Subsequently, spectral analysis of spectrograms weighted by these occlusion maps reveals significant differences between disease groups, particularly in patients with COPD, where cough patterns appear more variable in the identified spectral regions of interest. This contrasts with the lack of significant differences observed when analyzing raw spectrograms. The proposed approach extracts and analyzes several spectral features, demonstrating the potential of XAI techniques to uncover disease-specific acoustic signatures and improve the diagnostic capabilities of cough sound analysis by providing more interpretable results.
>
---
#### [new 002] Denoising by neural network for muzzle blast detection
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文针对战场枪声检测中环境噪音干扰问题，设计轻量级神经网络结合信号处理技术，显著提升脉冲枪声波形的检测率。**

- **链接: [http://arxiv.org/pdf/2508.14919v1](http://arxiv.org/pdf/2508.14919v1)**

> **作者:** Hadrien Pujol; Matteo Bevillacqua; Christophe Thirard; Thierry Mazoyer
>
> **备注:** INTER-NOISE 2024, Aug 2024, Nantes (France), France
>
> **摘要:** Acoem develops gunshot detection systems, consisting of a microphone array and software that detects and locates shooters on the battlefield. The performance of such systems is obviously affected by the acoustic environment in which they are operating: in particular, when mounted on a moving military vehicle, the presence of noise reduces the detection performance of the software. To limit the influence of the acoustic environment, a neural network has been developed. Instead of using a heavy convolutional neural network, a lightweight neural network architecture was chosen to limit the computational resources required to embed the algorithm on as many hardware platforms as possible. Thanks to the combination of a two hidden layer perceptron and appropriate signal processing techniques, the detection rate of impulsive muzzle blast waveforms (the wave coming from the detonation and indicating the position of the shooter) is significantly increased. With a rms value of noise of the same order as the muzzle blast peak amplitude, the detect rate is more than doubled with this denoising processing.
>
---
#### [new 003] An Enhanced Audio Feature Tailored for Anomalous Sound Detection Based on Pre-trained Models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 论文针对异常声音检测任务，解决异常定位不确定与冗余噪声干扰问题，提出等间距滤波器组特征与预训练模型参数自由增强方法，提升机器声音异常检测性能。**

- **链接: [http://arxiv.org/pdf/2508.15334v1](http://arxiv.org/pdf/2508.15334v1)**

> **作者:** Guirui Zhong; Qing Wang; Jun Du; Lei Wang; Mingqi Cai; Xin Fang
>
> **备注:** 13 pages, 3 figures, accepted by ICANN2025
>
> **摘要:** Anomalous Sound Detection (ASD) aims at identifying anomalous sounds from machines and has gained extensive research interests from both academia and industry. However, the uncertainty of anomaly location and much redundant information such as noise in machine sounds hinder the improvement of ASD system performance. This paper proposes a novel audio feature of filter banks with evenly distributed intervals, ensuring equal attention to all frequency ranges in the audio, which enhances the detection of anomalies in machine sounds. Moreover, based on pre-trained models, this paper presents a parameter-free feature enhancement approach to remove redundant information in machine audio. It is believed that this parameter-free strategy facilitates the effective transfer of universal knowledge from pre-trained tasks to the ASD task during model fine-tuning. Evaluation results on the Detection and Classification of Acoustic Scenes and Events (DCASE) 2024 Challenge dataset demonstrate significant improvements in ASD performance with our proposed methods.
>
---
#### [new 004] DualMark: Identifying Model and Training Data Origins in Generated Audio
- **分类: cs.SD**

- **简介: 该论文提出DualMark，解决音频生成模型无法追溯训练数据源的问题，通过双水印嵌入与损失函数设计，实现模型与数据来源联合标识，提升版权保护与责任追溯能力。**

- **链接: [http://arxiv.org/pdf/2508.15521v1](http://arxiv.org/pdf/2508.15521v1)**

> **作者:** Xuefeng Yang; Jian Guan; Feiyang Xiao; Congyi Fan; Haohe Liu; Qiaoxi Zhu; Dongli Xu; Youtian Lin
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Existing watermarking methods for audio generative models only enable model-level attribution, allowing the identification of the originating generation model, but are unable to trace the underlying training dataset. This significant limitation raises critical provenance questions, particularly in scenarios involving copyright and accountability concerns. To bridge this fundamental gap, we introduce DualMark, the first dual-provenance watermarking framework capable of simultaneously encoding two distinct attribution signatures, i.e., model identity and dataset origin, into audio generative models during training. Specifically, we propose a novel Dual Watermark Embedding (DWE) module to seamlessly embed dual watermarks into Mel-spectrogram representations, accompanied by a carefully designed Watermark Consistency Loss (WCL), which ensures reliable extraction of both watermarks from generated audio signals. Moreover, we establish the Dual Attribution Benchmark (DAB), the first robustness evaluation benchmark specifically tailored for joint model-data attribution. Extensive experiments validate that DualMark achieves outstanding attribution accuracy (97.01% F1-score for model attribution, and 91.51% AUC for dataset attribution), while maintaining exceptional robustness against aggressive pruning, lossy compression, additive noise, and sampling attacks, conditions that severely compromise prior methods. Our work thus provides a foundational step toward fully accountable audio generative models, significantly enhancing copyright protection and responsibility tracing capabilities.
>
---
#### [new 005] Comparative Evaluation of Text and Audio Simplification: A Methodological Replication Study
- **分类: cs.SD; eess.AS**

- **简介: 该论文通过复制研究比较文本与音频简化对医疗信息理解的影响，验证文本简化提升可及性的有效性，并分析教育水平和语言能力的作用。**

- **链接: [http://arxiv.org/pdf/2508.15088v1](http://arxiv.org/pdf/2508.15088v1)**

> **作者:** Prosanta Barai; Gondy Leroy; Arif Ahmed
>
> **摘要:** This study serves as a methodological replication of Leroy et al. (2022) research, which investigated the impact of text simplification on healthcare information comprehension in the evolving multimedia landscape. Building upon the original studys insights, our replication study evaluates audio content, recognizing its increasing importance in disseminating healthcare information in the digital age. Specifically, we explored the influence of text simplification on perceived and actual difficulty when users engage with audio content automatically generated from that text. Our replication involved 44 participants for whom we assessed their comprehension of healthcare information presented as audio created using Leroy et al. (2022) original and simplified texts. The findings from our study highlight the effectiveness of text simplification in enhancing perceived understandability and actual comprehension, aligning with the original studys results. Additionally, we examined the role of education level and language proficiency, shedding light on their potential impact on healthcare information access and understanding. This research underscores the practical value of text simplification tools in promoting health literacy. It suggests the need for tailored communication strategies to reach diverse audiences effectively in the healthcare domain.
>
---
#### [new 006] Human Feedback Driven Dynamic Speech Emotion Recognition
- **分类: cs.SD; cs.HC; cs.LG; eess.AS**

- **简介: 论文提出动态语音情感识别任务，解决时间序列情感建模问题，采用多阶段方法结合Dirichlet分布和人类反馈，提升模型效果。**

- **链接: [http://arxiv.org/pdf/2508.14920v1](http://arxiv.org/pdf/2508.14920v1)**

> **作者:** Ilya Fedorov; Dmitry Korobchenko
>
> **摘要:** This work proposes to explore a new area of dynamic speech emotion recognition. Unlike traditional methods, we assume that each audio track is associated with a sequence of emotions active at different moments in time. The study particularly focuses on the animation of emotional 3D avatars. We propose a multi-stage method that includes the training of a classical speech emotion recognition model, synthetic generation of emotional sequences, and further model improvement based on human feedback. Additionally, we introduce a novel approach to modeling emotional mixtures based on the Dirichlet distribution. The models are evaluated based on ground-truth emotions extracted from a dataset of 3D facial animations. We compare our models against the sliding window approach. Our experimental results show the effectiveness of Dirichlet-based approach in modeling emotional mixtures. Incorporating human feedback further improves the model quality while providing a simplified annotation procedure.
>
---
#### [new 007] Any-to-any Speaker Attribute Perturbation for Asynchronous Voice Anonymization
- **分类: cs.SD**

- **简介: 该论文针对异步语音匿名化任务，解决传统方法可能侵犯指定说话人隐私的问题，提出any-to-any策略通过批次均值损失生成伪说话人，并结合对抗训练实现有效匿名化，实验验证其有效性及局限性。**

- **链接: [http://arxiv.org/pdf/2508.15565v1](http://arxiv.org/pdf/2508.15565v1)**

> **作者:** Liping Chen; Chenyang Guo; Rui Wang; Kong Aik Lee; Zhenhua Ling
>
> **摘要:** Speaker attribute perturbation offers a feasible approach to asynchronous voice anonymization by employing adversarially perturbed speech as anonymized output. In order to enhance the identity unlinkability among anonymized utterances from the same original speaker, the targeted attack training strategy is usually applied to anonymize the utterances to a common designated speaker. However, this strategy may violate the privacy of the designated speaker who is an actual speaker. To mitigate this risk, this paper proposes an any-to-any training strategy. It is accomplished by defining a batch mean loss to anonymize the utterances from various speakers within a training mini-batch to a common pseudo-speaker, which is approximated as the average speaker in the mini-batch. Based on this, a speaker-adversarial speech generation model is proposed, incorporating the supervision from both the untargeted attack and the any-to-any strategies. The speaker attribute perturbations are generated and incorporated into the original speech to produce its anonymized version. The effectiveness of the proposed model was justified in asynchronous voice anonymization through experiments conducted on the VoxCeleb datasets. Additional experiments were carried out to explore the potential limitations of speaker-adversarial speech in voice privacy protection. With them, we aim to provide insights for future research on its protective efficacy against black-box speaker extractors \textcolor{black}{and adaptive attacks, as well as} generalization to out-of-domain datasets \textcolor{black}{and stability}. Audio samples and open-source code are published in https://github.com/VoicePrivacy/any-to-any-speaker-attribute-perturbation.
>
---
#### [new 008] AudioSet-R: A Refined AudioSet with Multi-Stage LLM Label Reannotation
- **分类: cs.SD**

- **简介: 论文针对音频分类任务中的标签准确性与完整性问题，提出多阶段LLM重新标注框架，通过跨模态提示策略优化AudioSet标签，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.15429v1](http://arxiv.org/pdf/2508.15429v1)**

> **作者:** Yulin Sun; Qisheng Xu; Yi Su; Qian Zhu; Yong Dou; Xinwang Liu; Kele Xu
>
> **备注:** 8 pages, 5 figures, accepted in ACM MM 2025 dataset track
>
> **摘要:** AudioSet is a widely used benchmark in the audio research community and has significantly advanced various audio-related tasks. However, persistent issues with label accuracy and completeness remain critical bottlenecks that limit performance in downstream applications.To address the aforementioned challenges, we propose a three-stage reannotation framework that harnesses general-purpose audio-language foundation models to systematically improve the label quality of AudioSet. The framework employs a cross-modal prompting strategy, inspired by the concept of prompt chaining, wherein prompts are sequentially composed to execute subtasks (audio comprehension, label synthesis, and semantic alignment). Leveraging this framework, we construct a high-quality, structured relabeled version of AudioSet-R. Extensive experiments conducted on representative audio classification models--including AST, PANNs, SSAST, and AudioMAE--consistently demonstrate substantial performance improvements, thereby validating the generalizability and effectiveness of the proposed approach in enhancing label reliability.The code is publicly available at: https://github.com/colaudiolab/AudioSet-R.
>
---
#### [new 009] ASCMamba: Multimodal Time-Frequency Mamba for Acoustic Scene Classification
- **分类: cs.SD**

- **简介: 论文针对多模态声学场景分类任务，解决传统仅依赖音频的局限性，通过融合文本信息（位置、时间）提出ASCMamba模型，集成DenseEncoder、Mamba块与伪标签机制，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2508.15632v1](http://arxiv.org/pdf/2508.15632v1)**

> **作者:** Bochao Sun; Dong Wang; Han Yin
>
> **摘要:** Acoustic Scene Classification (ASC) is a fundamental problem in computational audition, which seeks to classify environments based on the distinctive acoustic features. In the ASC task of the APSIPA ASC 2025 Grand Challenge, the organizers introduce a multimodal ASC task. Unlike traditional ASC systems that rely solely on audio inputs, this challenge provides additional textual information as inputs, including the location where the audio is recorded and the time of recording. In this paper, we present our proposed system for the ASC task in the APSIPA ASC 2025 Grand Challenge. Specifically, we propose a multimodal network, \textbf{ASCMamba}, which integrates audio and textual information for fine-grained acoustic scene understanding and effective multimodal ASC. The proposed ASCMamba employs a DenseEncoder to extract hierarchical spectral features from spectrograms, followed by a dual-path Mamba blocks that capture long-range temporal and frequency dependencies using Mamba-based state space models. In addition, we present a two-step pseudo-labeling mechanism to generate more reliable pseudo-labels. Results show that the proposed system outperforms all the participating teams and achieves a 6.2% improvement over the baseline. Code, model and pre-trained checkpoints are available at https://github.com/S-Orion/ASCMamba.git.
>
---
#### [new 010] Scalable FPGA Framework for Real-Time Denoising in High-Throughput Imaging: A DRAM-Optimized Pipeline using High-Level Synthesis
- **分类: cs.AR; cs.CV; cs.DC; eess.IV; eess.SP; physics.ins-det**

- **简介: 论文提出一种基于FPGA的实时去噪框架，用于处理高吞吐成像中的高速数据，通过HLS优化DRAM缓冲和突发模式接口，降低延迟并减少后续处理数据量。**

- **链接: [http://arxiv.org/pdf/2508.14917v1](http://arxiv.org/pdf/2508.14917v1)**

> **作者:** Weichien Liao
>
> **备注:** FPGA-based denoising pipeline for PRISM-scale imaging. Real-time frame subtraction and averaging via burst-mode AXI4 and DRAM buffering. Benchmarked against CPU/GPU workflows; scalable across multi-bank FPGA setups
>
> **摘要:** High-throughput imaging workflows, such as Parallel Rapid Imaging with Spectroscopic Mapping (PRISM), generate data at rates that exceed conventional real-time processing capabilities. We present a scalable FPGA-based preprocessing pipeline for real-time denoising, implemented via High-Level Synthesis (HLS) and optimized for DRAM-backed buffering. Our architecture performs frame subtraction and averaging directly on streamed image data, minimizing latency through burst-mode AXI4 interfaces. The resulting kernel operates below the inter-frame interval, enabling inline denoising and reducing dataset size for downstream CPU/GPU analysis. Validated under PRISM-scale acquisition, this modular FPGA framework offers a practical solution for latency-sensitive imaging workflows in spectroscopy and microscopy.
>
---
#### [new 011] Optimal Interference Signal for Masking an Acoustic Source
- **分类: math.AP; cs.SD; eess.AS; 35C05, 35Q93, 76Q05**

- **简介: 论文旨在设计最优干扰信号以掩盖声源，通过理论分析与数值方法减少目标区域残留振幅，应用于水下通信安全等领域。**

- **链接: [http://arxiv.org/pdf/2508.15023v1](http://arxiv.org/pdf/2508.15023v1)**

> **作者:** Hongyun Wang; Hong Zhou
>
> **备注:** 40 pages, a preprint
>
> **摘要:** In an environment where acoustic privacy or deliberate signal obfuscation is desired, it is necessary to mask the acoustic signature generated in essential operations. We consider the problem of masking the effect of an acoustic source in a target region where possible detection sensors are located. Masking is achieved by placing interference signals near the acoustic source. We introduce a theoretical and computational framework for designing such interference signals with the goal of minimizing the residual amplitude in the target region. For the three-dimensional (3D) forced wave equation with spherical symmetry, we derive analytical quasi-steady periodic solutions for several canonical cases. We examine the phenomenon of self-masking where an acoustic source with certain spatial forcing profile masks itself from detection outside its forcing footprint. We then use superposition of spherically symmetric solutions to investigate masking in a given target region. We analyze and optimize the performance of using one or two point-forces deployed near the acoustic source for masking in the target region. For the general case where the spatial forcing profile of the acoustic source lacks spherical symmetry, we develop an efficient numerical method for solving the 3D wave equation. Potential applications of this work include undersea acoustic communication security, undersea vehicles stealth, and protection against acoustic surveillance.
>
---
#### [new 012] A Chinese Heart Failure Status Speech Database with Universal and Personalised Classification
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文构建首个中文心衰语音数据库，通过对比住院前后语音记录，验证中文声调含心衰信息，并提出自适应频率滤波器，实现通用与个性化分类，解决中文心衰检测有效性问题。**

- **链接: [http://arxiv.org/pdf/2508.14908v1](http://arxiv.org/pdf/2508.14908v1)**

> **作者:** Yue Pan; Liwei Liu; Changxin Li; Xinyao Wang; Yili Xia; Hanyue Zhang; Ming Chu
>
> **摘要:** Speech is a cost-effective and non-intrusive data source for identifying acute and chronic heart failure (HF). However, there is a lack of research on whether Chinese syllables contain HF-related information, as observed in other well-studied languages. This study presents the first Chinese speech database of HF patients, featuring paired recordings taken before and after hospitalisation. The findings confirm the effectiveness of the Chinese language in HF detection using both standard 'patient-wise' and personalised 'pair-wise' classification approaches, with the latter serving as an ideal speaker-decoupled baseline for future research. Statistical tests and classification results highlight individual differences as key contributors to inaccuracy. Additionally, an adaptive frequency filter (AFF) is proposed for frequency importance analysis. The data and demonstrations are published at https://github.com/panyue1998/Voice_HF.
>
---
#### [new 013] UniCoM: A Universal Code-Switching Speech Generator
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文提出UniCoM框架，通过SWORDS算法生成自然代码切换语音样本，构建CS-FLEURS数据集，解决多语言语音技术中代码切换数据稀缺问题，提升ASR和S2TT性能。**

- **链接: [http://arxiv.org/pdf/2508.15244v1](http://arxiv.org/pdf/2508.15244v1)**

> **作者:** Sangmin Lee; Woojin Chung; Seyun Um; Hong-Goo Kang
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Code-switching (CS), the alternation between two or more languages within a single speaker's utterances, is common in real-world conversations and poses significant challenges for multilingual speech technology. However, systems capable of handling this phenomenon remain underexplored, primarily due to the scarcity of suitable datasets. To resolve this issue, we propose Universal Code-Mixer (UniCoM), a novel pipeline for generating high-quality, natural CS samples without altering sentence semantics. Our approach utilizes an algorithm we call Substituting WORDs with Synonyms (SWORDS), which generates CS speech by replacing selected words with their translations while considering their parts of speech. Using UniCoM, we construct Code-Switching FLEURS (CS-FLEURS), a multilingual CS corpus designed for automatic speech recognition (ASR) and speech-to-text translation (S2TT). Experimental results show that CS-FLEURS achieves high intelligibility and naturalness, performing comparably to existing datasets on both objective and subjective metrics. We expect our approach to advance CS speech technology and enable more inclusive multilingual systems.
>
---
#### [new 014] Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文针对基于语言模型的TTS系统生成幻觉问题，提出GOAT框架，通过分布对齐和GFlowNets优化生成过程，减少幻觉并提升稳定性，实验显示显著降低错误率。**

- **链接: [http://arxiv.org/pdf/2508.15442v1](http://arxiv.org/pdf/2508.15442v1)**

> **作者:** Chenlin Liu; Minghui Fang; Patrick Zhang; Wei Zhou; Jie Gao; Jiqing Han
>
> **摘要:** Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness.
>
---
#### [new 015] LLaSO: A Foundational Framework for Reproducible Research in Large Language and Speech Model
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; cs.SD**

- **简介: 该论文针对大规模语音-语言模型（LSLMs）的碎片化架构和数据透明度问题，提出LLaSO框架，提供对齐语料、多任务指令数据集及可重复基准，构建3.8B参数参考模型，推动研究统一与可重复性。**

- **链接: [http://arxiv.org/pdf/2508.15418v1](http://arxiv.org/pdf/2508.15418v1)**

> **作者:** Yirong Sun; Yizhong Geng; Peidong Wei; Yanjun Chen; Jinghan Yang; Rongfei Chen; Wei Zhang; Xiaoyu Shen
>
> **摘要:** The development of Large Speech-Language Models (LSLMs) has been slowed by fragmented architectures and a lack of transparency, hindering the systematic comparison and reproducibility of research. Unlike in the vision-language domain, the LSLM field suffers from the common practice of releasing model weights without their corresponding training data and configurations. To address these critical gaps, we introduce LLaSO, the first fully open, end-to-end framework for large-scale speech-language modeling. LLaSO provides the community with three essential resources: (1) LLaSO-Align, a 12M-instance speech-text alignment corpus; (2) LLaSO-Instruct, a 13.5M-instance multi-task instruction-tuning dataset; and (3) LLaSO-Eval, a reproducible benchmark for standardized evaluation. To validate our framework, we build and release LLaSO-Base, a 3.8B-parameter reference model trained exclusively on our public data. It achieves a normalized score of 0.72, establishing a strong, reproducible baseline that surpasses comparable models. Our analysis reveals that while broader training coverage enhances performance, significant generalization gaps persist on unseen tasks, particularly in pure audio scenarios. By releasing the complete stack of data, benchmarks, and models, LLaSO establishes a foundational open standard to unify research efforts and accelerate community-driven progress in LSLMs. We release the code, dataset, pretrained models, and results in https://github.com/EIT-NLP/LLaSO.
>
---
## 更新

#### [replaced 001] Fine-Tuning ASR for Stuttered Speech: Personalized vs. Generalized Approaches
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00853v2](http://arxiv.org/pdf/2506.00853v2)**

> **作者:** Dena Mujtaba; Nihar Mahapatra
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Stuttering -- characterized by involuntary disfluencies such as blocks, prolongations, and repetitions -- is often misinterpreted by automatic speech recognition (ASR) systems, resulting in elevated word error rates and making voice-driven technologies inaccessible to people who stutter. The variability of disfluencies across speakers and contexts further complicates ASR training, compounded by limited annotated stuttered speech data. In this paper, we investigate fine-tuning ASRs for stuttered speech, comparing generalized models (trained across multiple speakers) to personalized models tailored to individual speech characteristics. Using a diverse range of voice-AI scenarios, including virtual assistants and video interviews, we evaluate how personalization affects transcription accuracy. Our findings show that personalized ASRs significantly reduce word error rates, especially in spontaneous speech, highlighting the potential of tailored models for more inclusive voice technologies.
>
---
#### [replaced 002] FoleySpace: Vision-Aligned Binaural Spatial Audio Generation
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2508.12918v2](http://arxiv.org/pdf/2508.12918v2)**

> **作者:** Lei Zhao; Rujin Chen; Chi Zhang; Xiao-Lei Zhang; Xuelong Li
>
> **摘要:** Recently, with the advancement of AIGC, deep learning-based video-to-audio (V2A) technology has garnered significant attention. However, existing research mostly focuses on mono audio generation that lacks spatial perception, while the exploration of binaural spatial audio generation technologies, which can provide a stronger sense of immersion, remains insufficient. To solve this problem, we propose FoleySpace, a framework for video-to-binaural audio generation that produces immersive and spatially consistent stereo sound guided by visual information. Specifically, we develop a sound source estimation method to determine the sound source 2D coordinates and depth in each video frame, and then employ a coordinate mapping mechanism to convert the 2D source positions into a 3D trajectory. This 3D trajectory, together with the monaural audio generated by a pre-trained V2A model, serves as a conditioning input for a diffusion model to generate spatially consistent binaural audio. To support the generation of dynamic sound fields, we constructed a training dataset based on recorded Head-Related Impulse Responses that includes various sound source movement scenarios. Experimental results demonstrate that the proposed method outperforms existing approaches in spatial perception consistency, effectively enhancing the immersive quality of the audio-visual experience.
>
---
#### [replaced 003] DIFFA: Large Language Diffusion Models Can Listen and Understand
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.18452v2](http://arxiv.org/pdf/2507.18452v2)**

> **作者:** Jiaming Zhou; Hongjie Chen; Shiwan Zhao; Jian Kang; Jie Li; Enzhi Wang; Yujie Guo; Haoqin Sun; Hui Wang; Aobo Kong; Yong Qin; Xuelong Li
>
> **摘要:** Recent advances in large language models (LLMs) have shown remarkable capabilities across textual and multimodal domains. In parallel, diffusion-based language models have emerged as a promising alternative to the autoregressive paradigm, offering improved controllability, bidirectional context modeling, and robust generation. However, their application to the audio modality remains underexplored. In this work, we introduce \textbf{DIFFA}, the first diffusion-based large audio-language model designed to perform spoken language understanding. DIFFA integrates a frozen diffusion language model with a lightweight dual-adapter architecture that bridges speech understanding and natural language reasoning. We employ a two-stage training pipeline: first, aligning semantic representations via an ASR objective; then, learning instruction-following abilities through synthetic audio-caption pairs automatically generated by prompting LLMs. Despite being trained on only 960 hours of ASR and 127 hours of synthetic instruction data, DIFFA demonstrates competitive performance on major benchmarks, including MMSU, MMAU, and VoiceBench, outperforming several autoregressive open-source baselines. Our results reveal the potential of diffusion-based language models for efficient and scalable audio understanding, opening a new direction for speech-driven AI. Our code will be available at https://github.com/NKU-HLT/DIFFA.git.
>
---
#### [replaced 004] Machine Learning Approaches to Vocal Register Classification in Contemporary Male Pop Music
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11378v2](http://arxiv.org/pdf/2505.11378v2)**

> **作者:** Alexander Kim; Charlotte Botha
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** For singers of all experience levels, one of the most daunting challenges in learning technical repertoire is navigating placement and vocal register in and around the passagio (passage between chest voice and head voice registers). Particularly in pop music, where a single artist may use a variety of timbre's and textures to achieve a desired quality, it can be difficult to identify what vocal register within the vocal range a singer is using. This paper presents two methods for classifying vocal registers in an audio signal of male pop music through the analysis of textural features of mel-spectrogram images. Additionally, we will discuss the practical integration of these models for vocal analysis tools, and introduce a concurrently developed software called AVRA which stands for Automatic Vocal Register Analysis. Our proposed methods achieved consistent classification of vocal register through both Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models, which supports the promise of more robust classification possibilities across more voice types and genres of singing.
>
---
#### [replaced 005] Versatile Framework for Song Generation with Prompt-based Control
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.19062v4](http://arxiv.org/pdf/2504.19062v4)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Ruiqi Li; Jingyu Lu; Rongjie Huang; Ruiyuan Zhang; Zhiqing Hong; Ziyue Jiang; Zhou Zhao
>
> **备注:** Accepted by Findings of EMNLP 2025
>
> **摘要:** Song generation focuses on producing controllable high-quality songs based on various prompts. However, existing methods struggle to generate vocals and accompaniments with prompt-based control and proper alignment. Additionally, they fall short in supporting various tasks. To address these challenges, we introduce VersBand, a multi-task song generation framework for synthesizing high-quality, aligned songs with prompt-based control. VersBand comprises these primary models: 1) VocalBand, a decoupled model, leverages the flow-matching method for generating singing styles, pitches, and mel-spectrograms, allowing fast, high-quality vocal generation with style control. 2) AccompBand, a flow-based transformer model, incorporates the Band-MOE, selecting suitable experts for enhanced quality, alignment, and control. This model allows for generating controllable, high-quality accompaniments aligned with vocals. 3) Two generation models, LyricBand for lyrics and MelodyBand for melodies, contribute to the comprehensive multi-task song generation system, allowing for extensive control based on multiple prompts. Experimental results show that VersBand outperforms baseline models across multiple song generation tasks using objective and subjective metrics. Demos and codes are available at https://aaronz345.github.io/VersBandDemo and https://github.com/AaronZ345/VersBand.
>
---
#### [replaced 006] Prescriptive Agents based on RAG for Automated Maintenance (PARAM)
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.04714v2](http://arxiv.org/pdf/2508.04714v2)**

> **作者:** Chitranshu Harbola; Anupam Purwar
>
> **摘要:** Industrial machinery maintenance requires timely intervention to prevent catastrophic failures and optimize operational efficiency. This paper presents an integrated Large Language Model (LLM)-based intelligent system for prescriptive maintenance that extends beyond traditional anomaly detection to provide actionable maintenance recommendations. Building upon our prior LAMP framework for numerical data analysis, we develop a comprehensive solution that combines bearing vibration frequency analysis with multi agentic generation for intelligent maintenance planning. Our approach serializes bearing vibration data (BPFO, BPFI, BSF, FTF frequencies) into natural language for LLM processing, enabling few-shot anomaly detection with high accuracy. The system classifies fault types (inner race, outer race, ball/roller, cage faults) and assesses severity levels. A multi-agentic component processes maintenance manuals using vector embeddings and semantic search, while also conducting web searches to retrieve comprehensive procedural knowledge and access up-to-date maintenance practices for more accurate and in-depth recommendations. The Gemini model then generates structured maintenance recommendations includes immediate actions, inspection checklists, corrective measures, parts requirements, and timeline specifications. Experimental validation in bearing vibration datasets demonstrates effective anomaly detection and contextually relevant maintenance guidance. The system successfully bridges the gap between condition monitoring and actionable maintenance planning, providing industrial practitioners with intelligent decision support. This work advances the application of LLMs in industrial maintenance, offering a scalable framework for prescriptive maintenance across machinery components and industrial sectors.
>
---
