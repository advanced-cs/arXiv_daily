# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Confidence Boosts Trust-Based Resilience in Cooperative Multi-Robot Systems
- **分类: eess.SP; cs.MA; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多机器人系统安全任务，解决恶意机器人干扰问题。提出一种基于物理信道信任度的弹性协议，通过调整参数实现协调与效率的平衡。**

- **链接: [http://arxiv.org/pdf/2506.08807v1](http://arxiv.org/pdf/2506.08807v1)**

> **作者:** Luca Ballotta; Áron Vékássy; Stephanie Gil; Michal Yemini
>
> **备注:** This work has been submitted to IEEE for possible publication
>
> **摘要:** Wireless communication-based multi-robot systems open the door to cyberattacks that can disrupt safety and performance of collaborative robots. The physical channel supporting inter-robot communication offers an attractive opportunity to decouple the detection of malicious robots from task-relevant data exchange between legitimate robots. Yet, trustworthiness indications coming from physical channels are uncertain and must be handled with this in mind. In this paper, we propose a resilient protocol for multi-robot operation wherein a parameter {\lambda}t accounts for how confident a robot is about the legitimacy of nearby robots that the physical channel indicates. Analytical results prove that our protocol achieves resilient coordination with arbitrarily many malicious robots under mild assumptions. Tuning {\lambda}t allows a designer to trade between near-optimal inter-robot coordination and quick task execution; see Fig. 1. This is a fundamental performance tradeoff and must be carefully evaluated based on the task at hand. The effectiveness of our approach is numerically verified with experiments involving platoons of autonomous cars where some vehicles are maliciously spoofed.
>
---
#### [new 002] Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频问答任务，解决LALMs生成自然语音响应的问题。提出Step-Audio-AQAA模型，实现端到端音频问答，提升语音控制性能。**

- **链接: [http://arxiv.org/pdf/2506.08967v1](http://arxiv.org/pdf/2506.08967v1)**

> **作者:** Ailin Huang; Bingxin Li; Bruce Wang; Boyong Wu; Chao Yan; Chengli Feng; Heng Wang; Hongyu Zhou; Hongyuan Wang; Jingbei Li; Jianjian Sun; Joanna Wang; Mingrui Chen; Peng Liu; Ruihang Miao; Shilei Jiang; Tian Fei; Wang You; Xi Chen; Xuerui Yang; Yechang Huang; Yuxiang Zhang; Zheng Ge; Zheng Gong; Zhewei Huang; Zixin Zhang; Bin Wang; Bo Li; Buyun Ma; Changxin Miao; Changyi Wan; Chen Xu; Dapeng Shi; Dingyuan Hu; Enle Liu; Guanzhe Huang; Gulin Yan; Hanpeng Hu; Haonan Jia; Jiahao Gong; Jiaoren Wu; Jie Wu; Jie Yang; Junzhe Lin; Kaixiang Li; Lei Xia; Longlong Gu; Ming Li; Nie Hao; Ranchen Ming; Shaoliang Pang; Siqi Liu; Song Yuan; Tiancheng Cao; Wen Li; Wenqing He; Xu Zhao; Xuelin Zhang; Yanbo Yu; Yinmin Zhong; Yu Zhou; Yuanwei Liang; Yuanwei Lu; Yuxiang Yang; Zidong Yang; Zili Zhang; Binxing Jiao; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Xinhao Zhang; Yibo Zhu; Daxin Jiang; Shuchang Zhou; Chen Hu
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks.
>
---
#### [new 003] A Review on Score-based Generative Models for Audio Applications
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频生成领域，旨在解决扩散模型设计与应用中的问题。通过系统综述和开源代码，提升音频任务的生成质量与可复现性。**

- **链接: [http://arxiv.org/pdf/2506.08457v1](http://arxiv.org/pdf/2506.08457v1)**

> **作者:** Ge Zhu; Yutong Wen; Zhiyao Duan
>
> **摘要:** Diffusion models have emerged as powerful deep generative techniques, producing high-quality and diverse samples in applications in various domains including audio. These models have many different design choices suitable for different applications, however, existing reviews lack in-depth discussions of these design choices. The audio diffusion model literature also lacks principled guidance for the implementation of these design choices and their comparisons for different applications. This survey provides a comprehensive review of diffusion model design with an emphasis on design principles for quality improvement and conditioning for audio applications. We adopt the score modeling perspective as a unifying framework that accommodates various interpretations, including recent approaches like flow matching. We systematically examine the training and sampling procedures of diffusion models, and audio applications through different conditioning mechanisms. To address the lack of audio diffusion model codebases and to promote reproducible research and rapid prototyping, we introduce an open-source codebase at https://github.com/gzhu06/AudioDiffuser that implements our reviewed framework for various audio applications. We demonstrate its capabilities through three case studies: audio generation, speech enhancement, and text-to-speech synthesis, with benchmark evaluations on standard datasets.
>
---
#### [new 004] MD-ViSCo: A Unified Model for Multi-Directional Vital Sign Waveform Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多方向生命体征波形转换任务，旨在解决现有模型仅适用于特定波形对的问题。提出MD-ViSCo统一框架，用单一模型生成多种波形。**

- **链接: [http://arxiv.org/pdf/2506.08357v1](http://arxiv.org/pdf/2506.08357v1)**

> **作者:** Franck Meyer; Kyunghoon Hur; Edward Choi
>
> **备注:** Main paper (16 pages, 5 figures). Paper submitted for review. Code available at https://github.com/fr-meyer/MD-ViSCo
>
> **摘要:** Despite the remarkable progress of deep-learning methods generating a target vital sign waveform from a source vital sign waveform, most existing models are designed exclusively for a specific source-to-target pair. This requires distinct model architectures, optimization procedures, and pre-processing pipelines, resulting in multiple models that hinder usability in clinical settings. To address this limitation, we propose the Multi-Directional Vital-Sign Converter (MD-ViSCo), a unified framework capable of generating any target waveform such as electrocardiogram (ECG), photoplethysmogram (PPG), or arterial blood pressure (ABP) from any single input waveform with a single model. MD-ViSCo employs a shallow 1-Dimensional U-Net integrated with a Swin Transformer that leverages Adaptive Instance Normalization (AdaIN) to capture distinct waveform styles. To evaluate the efficacy of MD-ViSCo, we conduct multi-directional waveform generation on two publicly available datasets. Our framework surpasses state-of-the-art baselines (NabNet & PPG2ABP) on average across all waveform types, lowering Mean absolute error (MAE) by 8.8% and improving Pearson correlation (PC) by 4.9% over two datasets. In addition, the generated ABP waveforms satisfy the Association for the Advancement of Medical Instrumentation (AAMI) criterion and achieve Grade B on the British Hypertension Society (BHS) standard, outperforming all baselines. By eliminating the need for developing a distinct model for each task, we believe that this work offers a unified framework that can deal with any kind of vital sign waveforms with a single model in healthcare monitoring.
>
---
#### [new 005] Passive acoustic non-line-of-sight localization without a relay surface
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于非视距声源定位任务，解决传统方法依赖反射面的问题。通过利用边缘衍射信号实现三维定位，提出两种场景下的方法。**

- **链接: [http://arxiv.org/pdf/2506.08471v1](http://arxiv.org/pdf/2506.08471v1)**

> **作者:** Tal I. Sommer; Ori Katz
>
> **摘要:** The detection and localization of a source hidden outside the Line-of-Sight (LOS) traditionally rely on the acquisition of indirect signals, such as those reflected from visible relay surfaces such as floors or walls. These reflected signals are then utilized to reconstruct the obscured scene. In this study, we present an approach that utilize signals diffracted from an edge of an obstacle to achieve three-dimensional (3D) localization of an acoustic point source situated outside the LOS. We address two scenarios - a doorway and a convex corner - and propose a localization method for each of them. For the first scenario, we utilize the two edges of the door as virtual detector arrays. For the second scenario, we exploit the spectral signature of a knife-edge diffraction, inspired by the human perception of sound location by the head-related transfer function (HRTF). In both methods, knife-edge diffraction is utilized to extend the capabilities of non-line-of-sight (NLOS) acoustic sensing, enabling localization in environments where conventional relay-surface based approaches may be limited.
>
---
#### [new 006] SPBA: Utilizing Speech Large Language Model for Backdoor Attacks on Speech Classification Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音分类安全领域，针对后门攻击问题，提出利用语音大语言模型生成多样触发器的SPBA方法，提升攻击效果。**

- **链接: [http://arxiv.org/pdf/2506.08346v1](http://arxiv.org/pdf/2506.08346v1)**

> **作者:** Wenhan Yao; Fen Xiao; Xiarun Chen; Jia Liu; YongQiang He; Weiping Wen
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** Deep speech classification tasks, including keyword spotting and speaker verification, are vital in speech-based human-computer interaction. Recently, the security of these technologies has been revealed to be susceptible to backdoor attacks. Specifically, attackers use noisy disruption triggers and speech element triggers to produce poisoned speech samples that train models to become vulnerable. However, these methods typically create only a limited number of backdoors due to the inherent constraints of the trigger function. In this paper, we propose that speech backdoor attacks can strategically focus on speech elements such as timbre and emotion, leveraging the Speech Large Language Model (SLLM) to generate diverse triggers. Increasing the number of triggers may disproportionately elevate the poisoning rate, resulting in higher attack costs and a lower success rate per trigger. We introduce the Multiple Gradient Descent Algorithm (MGDA) as a mitigation strategy to address this challenge. The proposed attack is called the Speech Prompt Backdoor Attack (SPBA). Building on this foundation, we conducted attack experiments on two speech classification tasks, demonstrating that SPBA shows significant trigger effectiveness and achieves exceptional performance in attack metrics.
>
---
#### [new 007] Multimodal Zero-Shot Framework for Deepfake Hate Speech Detection in Low-Resource Languages
- **分类: cs.SD**

- **简介: 该论文属于多模态 hate speech 检测任务，解决低资源语言中 deepfake 音频的 hate speech 识别问题。提出一种零样本框架，结合音频与文本进行对比学习。**

- **链接: [http://arxiv.org/pdf/2506.08372v1](http://arxiv.org/pdf/2506.08372v1)**

> **作者:** Rishabh Ranjan; Likhith Ayinala; Mayank Vatsa; Richa Singh
>
> **备注:** Accepted in Interpseech 2025
>
> **摘要:** This paper introduces a novel multimodal framework for hate speech detection in deepfake audio, excelling even in zero-shot scenarios. Unlike previous approaches, our method uses contrastive learning to jointly align audio and text representations across languages. We present the first benchmark dataset with 127,290 paired text and synthesized speech samples in six languages: English and five low-resource Indian languages (Hindi, Bengali, Marathi, Tamil, Telugu). Our model learns a shared semantic embedding space, enabling robust cross-lingual and cross-modal classification. Experiments on two multilingual test sets show our approach outperforms baselines, achieving accuracies of 0.819 and 0.701, and generalizes well to unseen languages. This demonstrates the advantage of combining modalities for hate speech detection in synthetic media, especially in low-resource settings where unimodal models falter. The Dataset is available at https://www.iab-rubric.org/resources.
>
---
#### [new 008] Higher-Order Network Representation of J. S. Bach's Solo Violin Sonatas and Partitas: Topological and Geometrical Explorations
- **分类: cs.SD; eess.AS; physics.soc-ph**

- **简介: 该论文属于音乐分析任务，旨在解决传统方法无法捕捉音乐复杂性的难题。通过构建高阶网络模型，分析巴赫小提琴独奏曲的拓扑与几何特性。**

- **链接: [http://arxiv.org/pdf/2506.08540v1](http://arxiv.org/pdf/2506.08540v1)**

> **作者:** Dima Mrad; Sara Najem
>
> **摘要:** Music is inherently complex, with structures and interactions that unfold across multiple layers. Complex networks have emerged as powerful structures for the quantitative analysis of Western classical music, revealing significant features of its harmonic and structural organization. Although notable works have used these approaches to study music, dyadic representations of interactions fall short in conveying the underlying complexity and depth. In recent years, the limitations of traditional graph representations have been questioned and challenged in the context of interactions that could be higher-dimensional. Effective musical analysis requires models that capture higher-order interactions and a framework that simultaneously captures transitions between them. Subsequently, in this paper, we present a topological framework for analyzing J. S. Bach's Solo Violin Sonatas and Partitas that uses higher-order networks where single notes are vertices, two-note chords are edges, three-notes are triangles, etc. We subsequently account for the flow of music, by modeling transitions between successive notes. We identify genre-specific patterns in the works' geometric and topological properties. In particular, we find signatures in the trends of the evolution of the Euler characteristic and curvature, as well as examining adherence to the Gauss-Bonnet theorem across different movement types. The distinctions are revealed between slow movements, Fugues, and Baroque dance movements through their simplicial complex representation.
>
---
#### [new 009] Pureformer-VC: Non-parallel Voice Conversion with Pure Stylized Transformer Blocks and Triplet Discriminative Training
- **分类: cs.SD**

- **简介: 该论文属于语音转换任务，旨在解决非并行语音转换中语音元素编码与合成困难的问题。提出Pureformer-VC框架，结合Transformer结构与风格迁移机制，提升转换效果。**

- **链接: [http://arxiv.org/pdf/2506.08348v1](http://arxiv.org/pdf/2506.08348v1)**

> **作者:** Wenhan Yao; Fen Xiao; Xiarun Chen; Jia Liu; YongQiang He; Weiping Wen
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** As a foundational technology for intelligent human-computer interaction, voice conversion (VC) seeks to transform speech from any source timbre into any target timbre. Traditional voice conversion methods based on Generative Adversarial Networks (GANs) encounter significant challenges in precisely encoding diverse speech elements and effectively synthesising these elements into natural-sounding converted speech. To overcome these limitations, we introduce Pureformer-VC, an encoder-decoder framework that utilizes Conformer blocks to build a disentangled encoder and employs Zipformer blocks to create a style transfer decoder. We adopt a variational decoupled training approach to isolate speech components using a Variational Autoencoder (VAE), complemented by triplet discriminative training to enhance the speaker's discriminative capabilities. Furthermore, we incorporate the Attention Style Transfer Mechanism (ASTM) with Zipformer's shared weights to improve the style transfer performance in the decoder. We conducted experiments on two multi-speaker datasets. The experimental results demonstrate that the proposed model achieves comparable subjective evaluation scores while significantly enhancing objective metrics compared to existing approaches in many-to-many and many-to-one VC scenarios.
>
---
#### [new 010] Teaching Physical Awareness to LLMs through Sounds
- **分类: cs.SD; cs.AI; cs.MM; cs.RO**

- **简介: 该论文属于物理感知任务，旨在解决LLMs缺乏物理理解的问题。通过声音数据和物理模拟器，提升模型对物理现象的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.08524v1](http://arxiv.org/pdf/2506.08524v1)**

> **作者:** Weiguo Wang; Andy Nie; Wenrui Zhou; Yi Kai; Chengchen Hu
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world.
>
---
#### [new 011] Auto-Regressive vs Flow-Matching: a Comparative Study of Modeling Paradigms for Text-to-Music Generation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于文本到音乐生成任务，比较自回归与流匹配两种建模范式，分析其优缺点，以指导未来模型设计。**

- **链接: [http://arxiv.org/pdf/2506.08570v1](http://arxiv.org/pdf/2506.08570v1)**

> **作者:** Or Tal; Felix Kreuk; Yossi Adi
>
> **摘要:** Recent progress in text-to-music generation has enabled models to synthesize high-quality musical segments, full compositions, and even respond to fine-grained control signals, e.g. chord progressions. State-of-the-art (SOTA) systems differ significantly across many dimensions, such as training datasets, modeling paradigms, and architectural choices. This diversity complicates efforts to evaluate models fairly and pinpoint which design choices most influence performance. While factors like data and architecture are important, in this study we focus exclusively on the modeling paradigm. We conduct a systematic empirical analysis to isolate its effects, offering insights into associated trade-offs and emergent behaviors that can guide future text-to-music generation systems. Specifically, we compare the two arguably most common modeling paradigms: Auto-Regressive decoding and Conditional Flow-Matching. We conduct a controlled comparison by training all models from scratch using identical datasets, training configurations, and similar backbone architectures. Performance is evaluated across multiple axes, including generation quality, robustness to inference configurations, scalability, adherence to both textual and temporally aligned conditioning, and editing capabilities in the form of audio inpainting. This comparative study sheds light on distinct strengths and limitations of each paradigm, providing actionable insights that can inform future architectural and training decisions in the evolving landscape of text-to-music generation. Audio sampled examples are available at: https://huggingface.co/spaces/ortal1602/ARvsFM
>
---
#### [new 012] Implementing Keyword Spotting on the MCUX947 Microcontroller with Integrated NPU
- **分类: cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于语音识别任务，解决资源受限设备上的实时关键词检测问题，通过在MCXN947微控制器上集成NPU实现高效CNN模型部署。**

- **链接: [http://arxiv.org/pdf/2506.08911v1](http://arxiv.org/pdf/2506.08911v1)**

> **作者:** Petar Jakuš; Hrvoje Džapo
>
> **备注:** 4 pages
>
> **摘要:** This paper presents a keyword spotting (KWS) system implemented on the NXP MCXN947 microcontroller with an integrated Neural Processing Unit (NPU), enabling real-time voice interaction on resource-constrained devices. The system combines MFCC feature extraction with a CNN classifier, optimized using Quantization Aware Training to reduce model size with minimal accuracy drop. Experimental results demonstrate a 59x speedup in inference time when leveraging the NPU compared to CPU-only execution, achieving 97.06% accuracy with a model size of 30.58 KB, demonstrating the feasibility of efficient, low-power voice interfaces on embedded platforms.
>
---
#### [new 013] Neighbors and relatives: How do speech embeddings reflect linguistic connections across the world?
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语言关系分析任务，旨在通过语音嵌入研究全球语言间的联系。工作包括使用XLS-R模型提取语音特征，并与传统语言学指标对比验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.08564v1](http://arxiv.org/pdf/2506.08564v1)**

> **作者:** Tuukka Törö; Antti Suni; Juraj Šimko
>
> **备注:** 27 pages, 11 figures (+5 supplementary), submitted to PLOS One
>
> **摘要:** Investigating linguistic relationships on a global scale requires analyzing diverse features such as syntax, phonology and prosody, which evolve at varying rates influenced by internal diversification, language contact, and sociolinguistic factors. Recent advances in machine learning (ML) offer complementary alternatives to traditional historical and typological approaches. Instead of relying on expert labor in analyzing specific linguistic features, these new methods enable the exploration of linguistic variation through embeddings derived directly from speech, opening new avenues for large-scale, data-driven analyses. This study employs embeddings from the fine-tuned XLS-R self-supervised language identification model voxlingua107-xls-r-300m-wav2vec, to analyze relationships between 106 world languages based on speech recordings. Using linear discriminant analysis (LDA), language embeddings are clustered and compared with genealogical, lexical, and geographical distances. The results demonstrate that embedding-based distances align closely with traditional measures, effectively capturing both global and local typological patterns. Challenges in visualizing relationships, particularly with hierarchical clustering and network-based methods, highlight the dynamic nature of language change. The findings show potential for scalable analyses of language variation based on speech embeddings, providing new perspectives on relationships among languages. By addressing methodological considerations such as corpus size and latent space dimensionality, this approach opens avenues for studying low-resource languages and bridging macro- and micro-level linguistic variation. Future work aims to extend these methods to underrepresented languages and integrate sociolinguistic variation for a more comprehensive understanding of linguistic diversity.
>
---
#### [new 014] Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
- **分类: cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于语音识别审计任务，旨在解决ASR系统在评估语言障碍者（如失语症患者）时的偏差问题，通过分析现有审计方法的三个缺陷并提出改进框架。**

- **链接: [http://arxiv.org/pdf/2506.08846v1](http://arxiv.org/pdf/2506.08846v1)**

> **作者:** Katelyn Xiaoying Mei; Anna Seo Gyeong Choi; Hilke Schellmann; Mona Sloane; Allison Koenecke
>
> **摘要:** Automatic Speech Recognition (ASR) has transformed daily tasks from video transcription to workplace hiring. ASR systems' growing use warrants robust and standardized auditing approaches to ensure automated transcriptions of high and equitable quality. This is especially critical for people with speech and language disorders (such as aphasia) who may disproportionately depend on ASR systems to navigate everyday life. In this work, we identify three pitfalls in existing standard ASR auditing procedures, and demonstrate how addressing them impacts audit results via a case study of six popular ASR systems' performance for aphasia speakers. First, audits often adhere to a single method of text standardization during data pre-processing, which (a) masks variability in ASR performance from applying different standardization methods, and (b) may not be consistent with how users - especially those from marginalized speech communities - would want their transcriptions to be standardized. Second, audits often display high-level demographic findings without further considering performance disparities among (a) more nuanced demographic subgroups, and (b) relevant covariates capturing acoustic information from the input audio. Third, audits often rely on a single gold-standard metric -- the Word Error Rate -- which does not fully capture the extent of errors arising from generative AI models, such as transcription hallucinations. We propose a more holistic auditing framework that accounts for these three pitfalls, and exemplify its results in our case study, finding consistently worse ASR performance for aphasia speakers relative to a control group. We call on practitioners to implement these robust ASR auditing practices that remain flexible to the rapidly changing ASR landscape.
>
---
#### [new 015] mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决低资源语言在LLM评估中的缺失问题。提出mSTEB基准，评估多语言模型在语音和文本任务上的表现。**

- **链接: [http://arxiv.org/pdf/2506.08400v1](http://arxiv.org/pdf/2506.08400v1)**

> **作者:** Luel Hagos Beyene; Vivek Verma; Min Ma; Jesujoba O. Alabi; Fabian David Schmidt; Joyce Nakatumba-Nabende; David Ifeoluwa Adelani
>
> **备注:** working paper
>
> **摘要:** Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
>
---
#### [new 016] Multi-Teacher Language-Aware Knowledge Distillation for Multilingual Speech Emotion Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音情感识别任务，旨在通过多教师知识蒸馏方法提升模型在多种语言上的情感识别能力。**

- **链接: [http://arxiv.org/pdf/2506.08717v1](http://arxiv.org/pdf/2506.08717v1)**

> **作者:** Mehedi Hasan Bijoy; Dejan Porjazovski; Tamás Grósz; Mikko Kurimo
>
> **备注:** Accepted to INTERSPEECH 2025 conference
>
> **摘要:** Speech Emotion Recognition (SER) is crucial for improving human-computer interaction. Despite strides in monolingual SER, extending them to build a multilingual system remains challenging. Our goal is to train a single model capable of multilingual SER by distilling knowledge from multiple teacher models. To address this, we introduce a novel language-aware multi-teacher knowledge distillation method to advance SER in English, Finnish, and French. It leverages Wav2Vec2.0 as the foundation of monolingual teacher models and then distills their knowledge into a single multilingual student model. The student model demonstrates state-of-the-art performance, with a weighted recall of 72.9 on the English dataset and an unweighted recall of 63.4 on the Finnish dataset, surpassing fine-tuning and knowledge distillation baselines. Our method excels in improving recall for sad and neutral emotions, although it still faces challenges in recognizing anger and happiness.
>
---
#### [new 017] RadioDUN: A Physics-Inspired Deep Unfolding Network for Radio Map Estimation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于无线电图估计任务，旨在从稀疏样本中重建密集无线电图。通过结合物理模型和深度学习，提出RadioDUN网络提升估计性能。**

- **链接: [http://arxiv.org/pdf/2506.08418v1](http://arxiv.org/pdf/2506.08418v1)**

> **作者:** Taiqin Chen; Zikun Zhou; Zheng Fang; Wenzhen Zou; Kanjun Liu; Ke Chen; Yongbing Zhang; Yaowei Wang
>
> **摘要:** The radio map represents the spatial distribution of spectrum resources within a region, supporting efficient resource allocation and interference mitigation. However, it is difficult to construct a dense radio map as a limited number of samples can be measured in practical scenarios. While existing works have used deep learning to estimate dense radio maps from sparse samples, they are hard to integrate with the physical characteristics of the radio map. To address this challenge, we cast radio map estimation as the sparse signal recovery problem. A physical propagation model is further incorporated to decompose the problem into multiple factor optimization sub-problems, thereby reducing recovery complexity. Inspired by the existing compressive sensing methods, we propose the Radio Deep Unfolding Network (RadioDUN) to unfold the optimization process, achieving adaptive parameter adjusting and prior fitting in a learnable manner. To account for the radio propagation characteristics, we develop a dynamic reweighting module (DRM) to adaptively model the importance of each factor for the radio map. Inspired by the shadowing factor in the physical propagation model, we integrate obstacle-related factors to express the obstacle-induced signal stochastic decay. The shadowing loss is further designed to constrain the factor prediction and act as a supplementary supervised objective, which enhances the performance of RadioDUN. Extensive experiments have been conducted to demonstrate that the proposed method outperforms the state-of-the-art methods. Our code will be made publicly available upon publication.
>
---
## 更新

#### [replaced 001] Summarizing Speech: A Comprehensive Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.08024v2](http://arxiv.org/pdf/2504.08024v2)**

> **作者:** Fabian Retkowski; Maike Züfle; Andreas Sudmann; Dinah Pfau; Shinji Watanabe; Jan Niehues; Alexander Waibel
>
> **摘要:** Speech summarization has become an essential tool for efficiently managing and accessing the growing volume of spoken and audiovisual content. However, despite its increasing importance, speech summarization remains loosely defined. The field intersects with several research areas, including speech recognition, text summarization, and specific applications like meeting summarization. This survey not only examines existing datasets and evaluation protocols, which are crucial for assessing the quality of summarization approaches, but also synthesizes recent developments in the field, highlighting the shift from traditional systems to advanced models like fine-tuned cascaded architectures and end-to-end solutions. In doing so, we surface the ongoing challenges, such as the need for realistic evaluation benchmarks, multilingual datasets, and long-context handling.
>
---
#### [replaced 002] CASPER: A Large Scale Spontaneous Speech Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00267v2](http://arxiv.org/pdf/2506.00267v2)**

> **作者:** Cihan Xiao; Ruixing Liang; Xiangyu Zhang; Mehmet Emre Tiryaki; Veronica Bae; Lavanya Shankar; Rong Yang; Ethan Poon; Emmanuel Dupoux; Sanjeev Khudanpur; Leibny Paola Garcia Perera
>
> **摘要:** The success of large language models has driven interest in developing similar speech processing capabilities. However, a key challenge is the scarcity of high-quality spontaneous speech data, as most existing datasets contain scripted dialogues. To address this, we present a novel pipeline for eliciting and recording natural dialogues and release our dataset with 100+ hours of spontaneous speech. Our approach fosters fluid, natural conversations while encouraging a diverse range of topics and interactive exchanges. Unlike traditional methods, it facilitates genuine interactions, providing a reproducible framework for future data collection. This paper introduces our dataset and methodology, laying the groundwork for addressing the shortage of spontaneous speech data. We plan to expand this dataset in future stages, offering a growing resource for the research community.
>
---
#### [replaced 003] Are you really listening? Boosting Perceptual Awareness in Music-QA Benchmarks
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2504.00369v2](http://arxiv.org/pdf/2504.00369v2)**

> **作者:** Yongyi Zang; Sean O'Brien; Taylor Berg-Kirkpatrick; Julian McAuley; Zachary Novack
>
> **备注:** ISMIR 2025
>
> **摘要:** Large Audio Language Models (LALMs), where pretrained text LLMs are finetuned with audio input, have made remarkable progress in music understanding. However, current evaluation methodologies exhibit critical limitations: on the leading Music Question Answering benchmark, MuchoMusic, text-only LLMs without audio perception capabilities achieve surprisingly high accuracy of up to 56.4%, on par or above most LALMs. Furthermore, when presented with random Gaussian noise instead of actual audio, LALMs still perform significantly above chance. These findings suggest existing benchmarks predominantly assess reasoning abilities rather than audio perception. To overcome this challenge, we present RUListening: Robust Understanding through Listening, a framework that enhances perceptual evaluation in Music-QA benchmarks. We introduce the Perceptual Index (PI), a quantitative metric that measures a question's reliance on audio perception by analyzing log probability distributions from text-only language models. Using this metric, we generate synthetic, challenging distractors to create QA pairs that necessitate genuine audio perception. When applied to MuchoMusic, our filtered dataset successfully forces models to rely on perceptual information-text-only LLMs perform at chance levels, while LALMs similarly deteriorate when audio inputs are replaced with noise. These results validate our framework's effectiveness in creating benchmarks that more accurately evaluate audio perception capabilities.
>
---
#### [replaced 004] Structuring Concept Space with the Musical Circle of Fifths by Utilizing Music Grammar Based Activations
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2403.00790v3](http://arxiv.org/pdf/2403.00790v3)**

> **作者:** Tofara Moyo
>
> **备注:** Inaccuracies in script
>
> **摘要:** In this paper, we explore the intriguing similarities between the structure of a discrete neural network, such as a spiking network, and the composition of a piano piece. While both involve nodes or notes that are activated sequentially or in parallel, the latter benefits from the rich body of music theory to guide meaningful combinations. We propose a novel approach that leverages musical grammar to regulate activations in a spiking neural network, allowing for the representation of symbols as attractors. By applying rules for chord progressions from music theory, we demonstrate how certain activations naturally follow others, akin to the concept of attraction. Furthermore, we introduce the concept of modulating keys to navigate different basins of attraction within the network. Ultimately, we show that the map of concepts in our model is structured by the musical circle of fifths, highlighting the potential for leveraging music theory principles in deep learning algorithms.
>
---
#### [replaced 005] Towards Generalized Source Tracing for Codec-Based Deepfake Speech
- **分类: cs.SD; cs.CR; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07294v2](http://arxiv.org/pdf/2506.07294v2)**

> **作者:** Xuanjun Chen; I-Ming Lin; Lin Zhang; Haibin Wu; Hung-yi Lee; Jyh-Shing Roger Jang
>
> **备注:** Working in progress
>
> **摘要:** Recent attempts at source tracing for codec-based deepfake speech (CodecFake), generated by neural audio codec-based speech generation (CoSG) models, have exhibited suboptimal performance. However, how to train source tracing models using simulated CoSG data while maintaining strong performance on real CoSG-generated audio remains an open challenge. In this paper, we show that models trained solely on codec-resynthesized data tend to overfit to non-speech regions and struggle to generalize to unseen content. To mitigate these challenges, we introduce the Semantic-Acoustic Source Tracing Network (SASTNet), which jointly leverages Whisper for semantic feature encoding and Wav2vec2 with AudioMAE for acoustic feature encoding. Our proposed SASTNet achieves state-of-the-art performance on the CoSG test set of the CodecFake+ dataset, demonstrating its effectiveness for reliable source tracing.
>
---
#### [replaced 006] Voice Impression Control in Zero-Shot TTS
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.05688v2](http://arxiv.org/pdf/2506.05688v2)**

> **作者:** Keinichi Fujita; Shota Horiguchi; Yusuke Ijima
>
> **备注:** 5 pages,5 figures, Accepted to INTERSPEECH 2025
>
> **摘要:** Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization. Audio examples are available on our demo page (https://ntt-hilab-gensp.github.io/is2025voiceimpression/).
>
---
#### [replaced 007] Just Project! Multi-Channel Despeckling, the Easy Way
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2408.11531v2](http://arxiv.org/pdf/2408.11531v2)**

> **作者:** Loïc Denis; Emanuele Dalsasso; Florence Tupin
>
> **摘要:** Reducing speckle fluctuations in multi-channel SAR images is essential in many applications of SAR imaging such as polarimetric classification or interferometric height estimation. While single-channel despeckling has widely benefited from the application of deep learning techniques, extensions to multi-channel SAR images are much more challenging. This paper introduces MuChaPro, a generic framework that exploits existing single-channel despeckling methods. The key idea is to generate numerous single-channel projections, restore these projections, and recombine them into the final multi-channel estimate. This simple approach is shown to be effective in polarimetric and/or interferometric modalities. A special appeal of MuChaPro is the possibility to apply a self-supervised training strategy to learn sensor-specific networks for single-channel despeckling.
>
---
#### [replaced 008] LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.00493v4](http://arxiv.org/pdf/2503.00493v4)**

> **作者:** Boyi Kang; Xinfa Zhu; Zihan Zhang; Zhen Ye; Mingshuai Liu; Ziqian Wang; Yike Zhu; Guobin Ma; Jun Chen; Longshuai Xiao; Chao Weng; Wei Xue; Lei Xie
>
> **备注:** ACL2025 main, Codes available at https://github.com/Kevin-naticl/LLaSE-G1
>
> **摘要:** Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area.
>
---
#### [replaced 009] An introduction to pitch strength in contemporary popular music analysis and production
- **分类: cs.SD; eess.AS; 00A65; J.5**

- **链接: [http://arxiv.org/pdf/2506.07473v2](http://arxiv.org/pdf/2506.07473v2)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2024, Innovation in Music Conference, 14-16 June, 2024, Kristiania University College, Oslo, Norway
>
> **摘要:** Music information retrieval distinguishes between low- and high-level descriptions of music. Current generative AI models rely on text descriptions that are higher level than the controls familiar to studio musicians. Pitch strength, a low-level perceptual parameter of contemporary popular music, may be one feature that could make such AI models more suited to music production. Signal and perceptual analyses suggest that pitch strength (1) varies significantly across and inside songs; (2) contributes to both small- and large-scale structure; (3) contributes to the handling of polyphonic dissonance; and (4) may be a feature of upper harmonics made audible in a perspective of perceptual richness.
>
---
#### [replaced 010] Exploring SSL Discrete Speech Features for Zipformer-based Contextual ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.08797v2](http://arxiv.org/pdf/2409.08797v2)**

> **作者:** Mingyu Cui; Yifan Yang; Jiajun Deng; Jiawen Kang; Shujie Hu; Tianzi Wang; Zhaoqing Li; Shiliang Zhang; Xie Chen; Xunying Liu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Self-supervised learning (SSL) based discrete speech representations are highly compact and domain adaptable. In this paper, SSL discrete speech features extracted from WavLM models are used as additional cross-utterance acoustic context features in Zipformer-Transducer ASR systems. The efficacy of replacing Fbank features with discrete token features for modelling either cross-utterance contexts (from preceding and future segments), or current utterance's internal contexts alone, or both at the same time, are demonstrated thoroughly on the Gigaspeech 1000-hr corpus. The best Zipformer-Transducer system using discrete tokens based cross-utterance context features outperforms the baseline using utterance internal context only with statistically significant word error rate (WER) reductions of 0.32% to 0.41% absolute (2.78% to 3.54% relative) on the dev and test data. The lowest published WER of 11.15% and 11.14% were obtained on the dev and test sets. Our work is open-source and publicly available at https://github.com/open-creator/icefall/tree/master/egs/gigaspeech/Context\_ASR.
>
---
#### [replaced 011] Speech Synthesis By Unrolling Diffusion Process using Neural Network Layers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.09652v4](http://arxiv.org/pdf/2309.09652v4)**

> **作者:** Peter Ochieng
>
> **备注:** 10 pages
>
> **摘要:** This work introduces UDPNet, a novel architecture designed to accelerate the reverse diffusion process in speech synthesis. Unlike traditional diffusion models that rely on timestep embeddings and shared network parameters, UDPNet unrolls the reverse diffusion process directly into the network architecture, with successive layers corresponding to equally spaced steps in the diffusion schedule. Each layer progressively refines the noisy input, culminating in a high-fidelity estimation of the original data, \(x_0\). Additionally, we redefine the learning target by predicting latent variables instead of the conventional \(x_0\) or noise \(\epsilon_0\). This shift addresses the common issue of large prediction errors in early denoising stages, effectively reducing speech distortion. Extensive evaluations on single- and multi-speaker datasets demonstrate that UDPNet consistently outperforms state-of-the-art methods in both quality and efficiency, while generalizing effectively to unseen speech. These results position UDPNet as a robust solution for real-time speech synthesis applications. Sample audio is available at https://onexpeters.github.io/UDPNet/.
>
---
#### [replaced 012] Enhancing Retrieval-Augmented Audio Captioning with Generation-Assisted Multimodal Querying and Progressive Learning
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.10913v3](http://arxiv.org/pdf/2410.10913v3)**

> **作者:** Choi Changin; Lim Sungjun; Rhee Wonjong
>
> **摘要:** Retrieval-augmented generation can improve audio captioning by incorporating relevant audio-text pairs from a knowledge base. Existing methods typically rely solely on the input audio as a unimodal retrieval query. In contrast, we propose Generation-Assisted Multimodal Querying, which generates a text description of the input audio to enable multimodal querying. This approach aligns the query modality with the audio-text structure of the knowledge base, leading to more effective retrieval. Furthermore, we introduce a novel progressive learning strategy that gradually increases the number of interleaved audio-text pairs to enhance the training process. Our experiments on AudioCaps, Clotho, and Auto-ACD demonstrate that our approach achieves state-of-the-art results across these benchmarks.
>
---
