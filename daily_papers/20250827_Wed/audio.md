# 音频 cs.SD;  eess.SP

- **最新发布 13 篇**

- **更新 1 篇**

## 最新发布

#### [new 001] Cross-Learning Fine-Tuning Strategy for Dysarthric Speech Recognition Via CDSD database
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对运动性构音障碍患者的语音识别任务，解决传统单患者微调导致的特征冲突问题，提出跨学习多说话人联合微调策略，通过CDSD数据库验证，显著降低词错误率（WER）。**

- **链接: [http://arxiv.org/pdf/2508.18732v1](http://arxiv.org/pdf/2508.18732v1)**

> **作者:** Qing Xiao; Yingshan Peng; PeiPei Zhang
>
> **摘要:** Dysarthric speech recognition faces challenges from severity variations and disparities relative to normal speech. Conventional approaches individually fine-tune ASR models pre-trained on normal speech per patient to prevent feature conflicts. Counter-intuitively, experiments reveal that multi-speaker fine-tuning (simultaneously on multiple dysarthric speakers) improves recognition of individual speech patterns. This strategy enhances generalization via broader pathological feature learning, mitigates speaker-specific overfitting, reduces per-patient data dependence, and improves target-speaker accuracy - achieving up to 13.15% lower WER versus single-speaker fine-tuning.
>
---
#### [new 002] EMind: A Foundation Model for Multi-task Electromagnetic Signals Understanding
- **分类: eess.SP; cs.AI; cs.CV**

- **简介: 论文提出EMind模型，解决电磁信号多任务理解中的异质性、噪声及跨任务泛化难题，通过构建最大标准化数据集与物理适配方法，实现统一框架下的高效学习与广泛迁移。**

- **链接: [http://arxiv.org/pdf/2508.18785v1](http://arxiv.org/pdf/2508.18785v1)**

> **作者:** Luqing Luo; Wenjin Gui; Yunfei Liu; Ziyue Zhang; Yunxi Zhang; Fengxiang Wang; Zonghao Guo; Zizhi Ma; Xinzhu Liu; Hanxiang He; Jinhai Li; Xin Qiu; Wupeng Xie; Yangang Sun
>
> **摘要:** Deep understanding of electromagnetic signals is fundamental to dynamic spectrum management, intelligent transportation, autonomous driving and unmanned vehicle perception. The field faces challenges because electromagnetic signals differ greatly from text and images, showing high heterogeneity, strong background noise and complex joint time frequency structure, which prevents existing general models from direct use. Electromagnetic communication and sensing tasks are diverse, current methods lack cross task generalization and transfer efficiency, and the scarcity of large high quality datasets blocks the creation of a truly general multitask learning framework. To overcome these issue, we introduce EMind, an electromagnetic signals foundation model that bridges large scale pretraining and the unique nature of this modality. We build the first unified and largest standardized electromagnetic signal dataset covering multiple signal types and tasks. By exploiting the physical properties of electromagnetic signals, we devise a length adaptive multi-signal packing method and a hardware-aware training strategy that enable efficient use and representation learning from heterogeneous multi-source signals. Experiments show that EMind achieves strong performance and broad generalization across many downstream tasks, moving decisively from task specific models to a unified framework for electromagnetic intelligence. The code is available at: https://github.com/GabrielleTse/EMind.
>
---
#### [new 003] SegReConcat: A Data Augmentation Method for Voice Anonymization Attack
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出SegReConcat，通过分割重组匿名语音以干扰攻击者，提升语音去匿名化效果，解决语音匿名化中残余线索带来的隐私风险。**

- **链接: [http://arxiv.org/pdf/2508.18907v1](http://arxiv.org/pdf/2508.18907v1)**

> **作者:** Ridwan Arefeen; Xiaoxiao Miao; Rong Tong; Aik Beng Ng; Simon See
>
> **备注:** The Paper has been accepted by APCIPA ASC 2025
>
> **摘要:** Anonymization of voice seeks to conceal the identity of the speaker while maintaining the utility of speech data. However, residual speaker cues often persist, which pose privacy risks. We propose SegReConcat, a data augmentation method for attacker-side enhancement of automatic speaker verification systems. SegReConcat segments anonymized speech at the word level, rearranges segments using random or similarity-based strategies to disrupt long-term contextual cues, and concatenates them with the original utterance, allowing an attacker to learn source speaker traits from multiple perspectives. The proposed method has been evaluated in the VoicePrivacy Attacker Challenge 2024 framework across seven anonymization systems, SegReConcat improves de-anonymization on five out of seven systems.
>
---
#### [new 004] SwiftF0: Fast and Accurate Monophonic Pitch Detection
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文针对噪声环境下的单音轨音高检测难题，提出轻量模型SwiftF0，通过多领域数据训练实现高效准确检测，引入合成语音数据集SpeechSynth解决标注缺失问题，并推出统一评估指标与开源基准。**

- **链接: [http://arxiv.org/pdf/2508.18440v1](http://arxiv.org/pdf/2508.18440v1)**

> **作者:** Lars Nieradzik
>
> **摘要:** Accurate and real-time monophonic pitch estimation in noisy conditions, particularly on resource-constrained devices, remains an open challenge in audio processing. We present \emph{SwiftF0}, a novel, lightweight neural model that sets a new state-of-the-art for monophonic pitch estimation. Through training on diverse speech, music, and synthetic datasets with extensive data augmentation, SwiftF0 achieves robust generalization across acoustic domains while maintaining computational efficiency. SwiftF0 achieves a 91.80\% harmonic mean (HM) at 10 dB SNR, outperforming baselines like CREPE by over 12 percentage points and degrading by only 2.3 points from clean audio. SwiftF0 requires only 95,842 parameters and runs approximately 42x faster than CREPE on CPU, making it ideal for efficient, real-time deployment. To address the critical lack of perfectly accurate ground truth pitch in speech corpora (which typically rely on algorithmic estimators or laryngograph signals), we introduce \emph{SpeechSynth}. This synthetic speech dataset, generated by a phoneme-level TTS model, provides exact, on-demand ground-truth pitch curves, enabling more robust model training and evaluation. Furthermore, we propose a unified metric, combining six complementary performance measures for comprehensive and reliable pitch evaluation, and release an open-source pitch benchmark suite. A live demo of SwiftF0 is available at https://swift-f0.github.io/, the source code at https://github.com/lars76/swift-f0, and the benchmark framework at https://github.com/lars76/pitch-benchmark.
>
---
#### [new 005] H-PRM: A Pluggable Hotword Pre-Retrieval Module for Various Speech Recognition Systems
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 论文针对语音识别（ASR）中的热词定制问题，提出H-PRM模块通过声学相似性预检索提升热词识别率，适用于传统模型与Audio LLM，解决大规模热词导致的识别性能下降问题。**

- **链接: [http://arxiv.org/pdf/2508.18295v1](http://arxiv.org/pdf/2508.18295v1)**

> **作者:** Huangyu Dai; Lingtao Mao; Ben Chen; Zihan Wang; Zihan Liang; Ying Han; Chenyi Lei; Han Li
>
> **摘要:** Hotword customization is crucial in ASR to enhance the accuracy of domain-specific terms. It has been primarily driven by the advancements in traditional models and Audio large language models (LLMs). However, existing models often struggle with large-scale hotwords, as the recognition rate drops dramatically with the number of hotwords increasing. In this paper, we introduce a novel hotword customization system that utilizes a hotword pre-retrieval module (H-PRM) to identify the most relevant hotword candidate by measuring the acoustic similarity between the hotwords and the speech segment. This plug-and-play solution can be easily integrated into traditional models such as SeACo-Paraformer, significantly enhancing hotwords post-recall rate (PRR). Additionally, we incorporate H-PRM into Audio LLMs through a prompt-based approach, enabling seamless customization of hotwords. Extensive testing validates that H-PRM can outperform existing methods, showing a new direction for hotword customization in ASR.
>
---
#### [new 006] MDD: a Mask Diffusion Detector to Protect Speaker Verification Systems from Adversarial Perturbations
- **分类: eess.AS; cs.SD**

- **简介: 论文针对说话验证系统对抗扰动问题，提出MDD框架，基于文本条件掩码扩散模型，通过模拟退化与重建实现检测与净化，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.19180v1](http://arxiv.org/pdf/2508.19180v1)**

> **作者:** Yibo Bai; Sizhou Chen; Michele Panariello; Xiao-Lei Zhang; Massimiliano Todisco; Nicholas Evans
>
> **备注:** Accepted by APSIPA ASC 2025
>
> **摘要:** Speaker verification systems are increasingly deployed in security-sensitive applications but remain highly vulnerable to adversarial perturbations. In this work, we propose the Mask Diffusion Detector (MDD), a novel adversarial detection and purification framework based on a \textit{text-conditioned masked diffusion model}. During training, MDD applies partial masking to Mel-spectrograms and progressively adds noise through a forward diffusion process, simulating the degradation of clean speech features. A reverse process then reconstructs the clean representation conditioned on the input transcription. Unlike prior approaches, MDD does not require adversarial examples or large-scale pretraining. Experimental results show that MDD achieves strong adversarial detection performance and outperforms prior state-of-the-art methods, including both diffusion-based and neural codec-based approaches. Furthermore, MDD effectively purifies adversarially-manipulated speech, restoring speaker verification performance to levels close to those observed under clean conditions. These findings demonstrate the potential of diffusion-based masking strategies for secure and reliable speaker verification systems.
>
---
#### [new 007] The Sound of Risk: A Multimodal Physics-Informed Acoustic Model for Forecasting Market Volatility and Enhancing Market Interpretability
- **分类: cs.LG; cs.AI; cs.SD; eess.AS; 62P05, 68T0; I.2.7; J.4**

- **简介: 该论文提出一种多模态框架，结合文本情感与语音特征，利用物理约束的声学模型预测市场波动，解决传统文本分析在信息不对称下的不足，通过分析高管情感动态提升市场可解释性。**

- **链接: [http://arxiv.org/pdf/2508.18653v1](http://arxiv.org/pdf/2508.18653v1)**

> **作者:** Xiaoliang Chen; Xin Yu; Le Chang; Teng Jing; Jiashuai He; Ze Wang; Yangjun Luo; Xingyu Chen; Jiayue Liang; Yuchen Wang; Jiaying Xie
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Information asymmetry in financial markets, often amplified by strategically crafted corporate narratives, undermines the effectiveness of conventional textual analysis. We propose a novel multimodal framework for financial risk assessment that integrates textual sentiment with paralinguistic cues derived from executive vocal tract dynamics in earnings calls. Central to this framework is the Physics-Informed Acoustic Model (PIAM), which applies nonlinear acoustics to robustly extract emotional signatures from raw teleconference sound subject to distortions such as signal clipping. Both acoustic and textual emotional states are projected onto an interpretable three-dimensional Affective State Label (ASL) space-Tension, Stability, and Arousal. Using a dataset of 1,795 earnings calls (approximately 1,800 hours), we construct features capturing dynamic shifts in executive affect between scripted presentation and spontaneous Q&A exchanges. Our key finding reveals a pronounced divergence in predictive capacity: while multimodal features do not forecast directional stock returns, they explain up to 43.8% of the out-of-sample variance in 30-day realized volatility. Importantly, volatility predictions are strongly driven by emotional dynamics during executive transitions from scripted to spontaneous speech, particularly reduced textual stability and heightened acoustic instability from CFOs, and significant arousal variability from CEOs. An ablation study confirms that our multimodal approach substantially outperforms a financials-only baseline, underscoring the complementary contributions of acoustic and textual modalities. By decoding latent markers of uncertainty from verifiable biometric signals, our methodology provides investors and regulators a powerful tool for enhancing market interpretability and identifying hidden corporate uncertainty.
>
---
#### [new 008] DESAMO: A Device for Elder-Friendly Smart Homes Powered by Embedded LLM with Audio Modality
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文提出DESAMO，一种基于嵌入式音频LLM的智能家居设备，解决老年人语音模糊和非语音音频处理难题，通过直接处理原始音频实现意图识别与紧急事件检测。**

- **链接: [http://arxiv.org/pdf/2508.18918v1](http://arxiv.org/pdf/2508.18918v1)**

> **作者:** Youngwon Choi; Donghyuk Jung; Hwayeon Kim
>
> **备注:** 2 pages, 2 figures. Accepted for presentation as a UIST 2025 Poster
>
> **摘要:** We present DESAMO, an on-device smart home system for elder-friendly use powered by Audio LLM, that supports natural and private interactions. While conventional voice assistants rely on ASR-based pipelines or ASR-LLM cascades, often struggling with the unclear speech common among elderly users and unable to handle non-speech audio, DESAMO leverages an Audio LLM to process raw audio input directly, enabling a robust understanding of user intent and critical events, such as falls or calls for help.
>
---
#### [new 009] Improving Noise Robust Audio-Visual Speech Recognition via Router-Gated Cross-Modal Feature Fusion
- **分类: cs.CV; cs.AI; cs.MM; eess.AS; eess.SP**

- **简介: 论文针对噪声环境下的音频视觉语音识别（AVSR）鲁棒性问题，提出路由门控跨模态特征融合框架，动态调整音频与视觉特征权重，提升模型对音频质量退化的适应能力。**

- **链接: [http://arxiv.org/pdf/2508.18734v1](http://arxiv.org/pdf/2508.18734v1)**

> **作者:** DongHoon Lim; YoungChae Kim; Dong-Hyun Kim; Da-Hee Yang; Joon-Hyuk Chang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Robust audio-visual speech recognition (AVSR) in noisy environments remains challenging, as existing systems struggle to estimate audio reliability and dynamically adjust modality reliance. We propose router-gated cross-modal feature fusion, a novel AVSR framework that adaptively reweights audio and visual features based on token-level acoustic corruption scores. Using an audio-visual feature fusion-based router, our method down-weights unreliable audio tokens and reinforces visual cues through gated cross-attention in each decoder layer. This enables the model to pivot toward the visual modality when audio quality deteriorates. Experiments on LRS3 demonstrate that our approach achieves an 16.51-42.67% relative reduction in word error rate compared to AV-HuBERT. Ablation studies confirm that both the router and gating mechanism contribute to improved robustness under real-world acoustic noise.
>
---
#### [new 010] Emotion Omni: Enabling Empathetic Speech Response Generation through Large Language Models
- **分类: cs.CL; cs.SD; eess.AS; I.2.7**

- **简介: 论文提出Emotion Omni模型，用于生成情感丰富的语音回应，解决现有模型无法理解用户语音中情感 cues 的问题，通过新架构和20万情感对话数据集提升同理心交互。**

- **链接: [http://arxiv.org/pdf/2508.18655v1](http://arxiv.org/pdf/2508.18655v1)**

> **作者:** Haoyu Wang; Guangyan Zhang; Jiale Chen; Jingyu Li; Yuehai Wang; Yiwen Guo
>
> **备注:** 5 pages, 1 figure, submitted to ICASSP 2026
>
> **摘要:** With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models simply convert the response content into speech without fully understanding the rich emotional and paralinguistic cues embedded in the user's query. In many cases, the same sentence can have different meanings depending on the emotional expression. Furthermore, emotional understanding is essential for improving user experience in human-machine interaction. Currently, most speech LLMs with empathetic capabilities are trained on massive datasets. This approach requires vast amounts of data and significant computational resources. Therefore, a key challenge lies in how to develop a speech LLM capable of generating empathetic responses with limited data and without the need for large-scale training. To address this challenge, we propose Emotion Omni, a novel model architecture designed to understand the emotional content of user speech input and generate empathetic speech responses. Additionally, we developed a data generation pipeline based on an open-source TTS framework to construct a 200k emotional dialogue dataset, which supports the construction of an empathetic speech assistant. The demos are available at https://w311411.github.io/omni_demo/
>
---
#### [new 011] VibeVoice Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出VibeVoice，用于多说话人长篇语音合成，采用next-token diffusion与新型分词器，提升效率与音质，超越现有模型。**

- **链接: [http://arxiv.org/pdf/2508.19205v1](http://arxiv.org/pdf/2508.19205v1)**

> **作者:** Zhiliang Peng; Jianwei Yu; Wenhui Wang; Yaoyao Chang; Yutao Sun; Li Dong; Yi Zhu; Weijiang Xu; Hangbo Bao; Zehua Wang; Shaohan Huang; Yan Xia; Furu Wei
>
> **摘要:** This report presents VibeVoice, a novel model designed to synthesize long-form speech with multiple speakers by employing next-token diffusion, which is a unified method for modeling continuous data by autoregressively generating latent vectors via diffusion. To enable this, we introduce a novel continuous speech tokenizer that, when compared to the popular Encodec model, improves data compression by 80 times while maintaining comparable performance. The tokenizer effectively preserves audio fidelity while significantly boosting computational efficiency for processing long sequences. Thus, VibeVoice can synthesize long-form speech for up to 90 minutes (in a 64K context window length) with a maximum of 4 speakers, capturing the authentic conversational ``vibe'' and surpassing open-source and proprietary dialogue models.
>
---
#### [new 012] EAI-Avatar: Emotion-Aware Interactive Talking Head Generation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出EAI-Avatar，解决双向对话中情感适配不足的问题，通过结合大语言模型与Transformer遮罩生成器，实现情感感知的交互式虚拟头像生成。**

- **链接: [http://arxiv.org/pdf/2508.18337v1](http://arxiv.org/pdf/2508.18337v1)**

> **作者:** Haijie Yang; Zhenyu Zhang; Hao Tang; Jianjun Qian; Jian Yang
>
> **摘要:** Generative models have advanced rapidly, enabling impressive talking head generation that brings AI to life. However, most existing methods focus solely on one-way portrait animation. Even the few that support bidirectional conversational interactions lack precise emotion-adaptive capabilities, significantly limiting their practical applicability. In this paper, we propose EAI-Avatar, a novel emotion-aware talking head generation framework for dyadic interactions. Leveraging the dialogue generation capability of large language models (LLMs, e.g., GPT-4), our method produces temporally consistent virtual avatars with rich emotional variations that seamlessly transition between speaking and listening states. Specifically, we design a Transformer-based head mask generator that learns temporally consistent motion features in a latent mask space, capable of generating arbitrary-length, temporally consistent mask sequences to constrain head motions. Furthermore, we introduce an interactive talking tree structure to represent dialogue state transitions, where each tree node contains information such as child/parent/sibling nodes and the current character's emotional state. By performing reverse-level traversal, we extract rich historical emotional cues from the current node to guide expression synthesis. Extensive experiments demonstrate the superior performance and effectiveness of our method.
>
---
#### [new 013] Toward Responsible ASR for African American English Speakers: A Scoping Review of Bias and Equity in Speech Technology
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文通过系统综述探讨ASR技术在非裔美国人英语群体中的偏见与公平性问题，分析四个研究领域并提出治理框架，旨在推动负责任的语音技术开发。**

- **链接: [http://arxiv.org/pdf/2508.18288v1](http://arxiv.org/pdf/2508.18288v1)**

> **作者:** Jay L. Cunningham; Adinawa Adjagbodjou; Jeffrey Basoah; Jainaba Jawara; Kowe Kadoma; Aaleyah Lewis
>
> **备注:** 10 pages, 9 Pages (References and Appendices). The archival version has been accepted to AAAI (AIES 2025) without the extended Appendices. This extended version includes Appendices
>
> **摘要:** This scoping literature review examines how fairness, bias, and equity are conceptualized and operationalized in Automatic Speech Recognition (ASR) and adjacent speech and language technologies (SLT) for African American English (AAE) speakers and other linguistically diverse communities. Drawing from 44 peer-reviewed publications across Human-Computer Interaction (HCI), Machine Learning/Natural Language Processing (ML/NLP), and Sociolinguistics, we identify four major areas of inquiry: (1) how researchers understand ASR-related harms; (2) inclusive data practices spanning collection, curation, annotation, and model training; (3) methodological and theoretical approaches to linguistic inclusion; and (4) emerging practices and design recommendations for more equitable systems. While technical fairness interventions are growing, our review highlights a critical gap in governance-centered approaches that foreground community agency, linguistic justice, and participatory accountability. We propose a governance-centered ASR lifecycle as an emergent interdisciplinary framework for responsible ASR development and offer implications for researchers, practitioners, and policymakers seeking to address language marginalization in speech AI systems.
>
---
## 更新

#### [replaced 001] Revisiting SSL for sound event detection: complementary fusion and adaptive post-processing
- **分类: eess.AS; cs.AI; cs.SD; I.5.4; I.2.10; H.5.5**

- **链接: [http://arxiv.org/pdf/2505.11889v2](http://arxiv.org/pdf/2505.11889v2)**

> **作者:** Hanfang Cui; Longfei Song; Li Li; Dongxing Xu; Yanhua Long
>
> **备注:** 27 pages, 5 figures, accepted by Journal of King Saud University Computer and Information Sciences online
>
> **摘要:** Self-supervised learning (SSL) models offer powerful representations for sound event detection (SED), yet their synergistic potential remains underexplored. This study systematically evaluates state-of-the-art SSL models to guide optimal model selection and integration for SED. We propose a framework that combines heterogeneous SSL representations (e.g., BEATs, HuBERT, WavLM) through three fusion strategies: individual SSL embedding integration, dual-modal fusion, and full aggregation. Experiments on the DCASE 2023 Task 4 Challenge reveal that dual-modal fusion (e.g., CRNN+BEATs+WavLM) achieves complementary performance gains, while CRNN+BEATs alone delivers the best results among individual SSL models. We further introduce normalized sound event bounding boxes (nSEBBs), an adaptive post-processing method that dynamically adjusts event boundary predictions, improving PSDS1 by up to 4% for standalone SSL models. These findings highlight the compatibility and complementarity of SSL architectures, providing guidance for task-specific fusion and robust SED system design.
>
---
