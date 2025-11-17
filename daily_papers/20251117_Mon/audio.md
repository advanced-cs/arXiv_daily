# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] MSMT-FN: Multi-segment Multi-task Fusion Network for Marketing Audio Classification
- **分类: cs.SD; cs.AI**

- **简介: 论文提出MSMT-FN模型用于营销音频分类任务，解决从大量营销电话中高效识别客户购买倾向的问题。通过多段多任务融合网络，在自建MarketCalls数据集和公开基准测试中实现性能领先。**

- **链接: [https://arxiv.org/pdf/2511.11006v1](https://arxiv.org/pdf/2511.11006v1)**

> **作者:** HongYu Liu; Ruijie Wan; Yueju Han; Junxin Li; Liuxing Lu; Chao He; Lihua Cai
>
> **备注:** Accepted at The 21st International Conference on Advanced Data Mining and Applications (ADMA 2025). In book: Advanced Data Mining and Applications (pp.306-320)
>
> **摘要:** Audio classification plays an essential role in sentiment analysis and emotion recognition, especially for analyzing customer attitudes in marketing phone calls. Efficiently categorizing customer purchasing propensity from large volumes of audio data remains challenging. In this work, we propose a novel Multi-Segment Multi-Task Fusion Network (MSMT-FN) that is uniquely designed for addressing this business demand. Evaluations conducted on our proprietary MarketCalls dataset, as well as established benchmarks (CMU-MOSI, CMU-MOSEI, and MELD), show MSMT-FN consistently outperforms or matches state-of-the-art methods. Additionally, our newly curated MarketCalls dataset will be available upon request, and the code base is made accessible at GitHub Repository MSMT-FN, to facilitate further research and advancements in audio classification domain.
>
---
#### [new 002] DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦音频对话意图识别任务，解决话语依赖复杂与标注数据稀缺问题。提出DialogGraph-LLM框架，融合多关系对话注意力网络与多模态LLM，并设计自适应半监督学习策略，显著提升识别性能。**

- **链接: [https://arxiv.org/pdf/2511.11000v1](https://arxiv.org/pdf/2511.11000v1)**

> **作者:** HongYu Liu; Junxin Li; Changxi Guo; Hao Chen; Yaqian Huang; Yifu Guo; Huan Yang; Lihua Cai
>
> **备注:** 8 pages, 2 figures; Series: Frontiers in Artificial Intelligence and Applications, Volume 413: ECAI 2025
>
> **摘要:** Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
>
---
#### [new 003] Graph Neural Field with Spatial-Correlation Augmentation for HRTF Personalization
- **分类: cs.SD**

- **简介: 论文针对VR/AR中HRTF个人化问题，提出GraphNF-SCA模型，利用图神经网络和空间相关性增强，高效生成个体化HRTFs，避免传统测量耗时。**

- **链接: [https://arxiv.org/pdf/2511.10697v1](https://arxiv.org/pdf/2511.10697v1)**

> **作者:** De Hu; Junsheng Hu; Cuicui Jiang
>
> **摘要:** To achieve immersive spatial audio rendering on VR/AR devices, high-quality Head-Related Transfer Functions (HRTFs) are essential. In general, HRTFs are subject-dependent and position-dependent, and their measurement is time-consuming and tedious. To address this challenge, we propose the Graph Neural Field with Spatial-Correlation Augmentation (GraphNF-SCA) for HRTF personalization, which can be used to generate individual HRTFs for unseen subjects. The GraphNF-SCA consists of three key components: an HRTF personalization (HRTF-P) module, an HRTF upsampling (HRTF-U) module, and a fine-tuning stage. In the HRTF-P module, we predict HRTFs of the target subject via the Graph Neural Network (GNN) with an encoder-decoder architecture, where the encoder extracts universal features and the decoder incorporates the target-relevant features and produces individualized HRTFs. The HRTF-U module employs another GNN to model spatial correlations across HRTFs. This module is fine-tuned using the output of the HRTF-P module, thereby enhancing the spatial consistency of the predicted HRTFs. Unlike existing methods that estimate individual HRTFs position-by-position without spatial correlation modeling, the GraphNF-SCA effectively leverages inherent spatial correlations across HRTFs to enhance the performance of HRTF personalization. Experimental results demonstrate that the GraphNF-SCA achieves state-of-the-art results.
>
---
#### [new 004] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding
- **分类: cs.SD; cs.LG; q-bio.NC**

- **简介: 该论文针对BCI中文声调分类任务，提出CAT-Net框架融合EEG和EMG信号。通过交叉注意力机制和域对抗训练，实现最小通道（20 EEG, 5 EMG）下高准确率（可听87.83%，无声88.08%），跨被试准确率83.27%/85.10%，有效提升解码性能和泛化性。**

- **链接: [https://arxiv.org/pdf/2511.10935v1](https://arxiv.org/pdf/2511.10935v1)**

> **作者:** Yifan Zhuang; Calvin Huang; Zepeng Yu; Yongjie Zou; Jiawei Ju
>
> **备注:** This is the extended version with technical appendices. The version of record appears in AAAI-26. Please cite the AAAI version
>
> **摘要:** Brain-computer interface (BCI) speech decoding has emerged as a promising tool for assisting individuals with speech impairments. In this context, the integration of electroencephalography (EEG) and electromyography (EMG) signals offers strong potential for enhancing decoding performance. Mandarin tone classification presents particular challenges, as tonal variations convey distinct meanings even when phonemes remain identical. In this study, we propose a novel cross-subject multimodal BCI decoding framework that fuses EEG and EMG signals to classify four Mandarin tones under both audible and silent speech conditions. Inspired by the cooperative mechanisms of neural and muscular systems in speech production, our neural decoding architecture combines spatial-temporal feature extraction branches with a cross-attention fusion mechanism, enabling informative interaction between modalities. We further incorporate domain-adversarial training to improve cross-subject generalization. We collected 4,800 EEG trials and 4,800 EMG trials from 10 participants using only twenty EEG and five EMG channels, demonstrating the feasibility of minimal-channel decoding. Despite employing lightweight modules, our model outperforms state-of-the-art baselines across all conditions, achieving average classification accuracies of 87.83% for audible speech and 88.08% for silent speech. In cross-subject evaluations, it still maintains strong performance with accuracies of 83.27% and 85.10% for audible and silent speech, respectively. We further conduct ablation studies to validate the effectiveness of each component. Our findings suggest that tone-level decoding with minimal EEG-EMG channels is feasible and potentially generalizable across subjects, contributing to the development of practical BCI applications.
>
---
#### [new 005] Curved Worlds, Clear Boundaries: Generalizing Speech Deepfake Detection using Hyperbolic and Spherical Geometry Spaces
- **分类: eess.AS**

- **简介: 该论文解决语音深度伪造检测的跨范式泛化问题，针对传统TTS与扩散模型等不同合成技术无法通用的挑战。提出RHYME框架，利用双曲与球面几何映射融合嵌入表示，实现合成无关对齐，显著提升跨范式检测性能。**

- **链接: [https://arxiv.org/pdf/2511.10793v1](https://arxiv.org/pdf/2511.10793v1)**

> **作者:** Farhan Sheth; Girish; Mohd Mujtaba Akhtar; Muskaan Singh
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** In this work, we address the challenge of generalizable audio deepfake detection (ADD) across diverse speech synthesis paradigms-including conventional text-to-speech (TTS) systems and modern diffusion or flow-matching (FM) based generators. Prior work has mostly targeted individual synthesis families and often fails to generalize across paradigms due to overfitting to generation-specific artifacts. We hypothesize that synthetic speech, irrespective of its generative origin, leaves behind shared structural distortions in the embedding space that can be aligned through geometry-aware modeling. To this end, we propose RHYME, a unified detection framework that fuses utterance-level embeddings from diverse pretrained speech encoders using non-Euclidean projections. RHYME maps representations into hyperbolic and spherical manifolds-where hyperbolic geometry excels at modeling hierarchical generator families, and spherical projections capture angular, energy-invariant cues such as periodic vocoder artifacts. The fused representation is obtained via Riemannian barycentric averaging, enabling synthesis-invariant alignment. RHYME outperforms individual PTMs and homogeneous fusion baselines, achieving top performance and setting new state-of-the-art in cross-paradigm ADD.
>
---
#### [new 006] TimeAudio: Bridging Temporal Gaps in Large Audio-Language Models
- **分类: cs.SD**

- **简介: TimeAudio解决大音频-语言模型时间定位能力不足问题，通过时间标记、绝对时间编码和分段令牌合并提升长音频理解，构建新数据集评估细粒度任务性能。**

- **链接: [https://arxiv.org/pdf/2511.11039v1](https://arxiv.org/pdf/2511.11039v1)**

> **作者:** Hualei Wang; Yiming Li; Shuo Ma; Hong Liu; Xiangdong Wang
>
> **备注:** Accepted by The Fortieth AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Recent Large Audio-Language Models (LALMs) exhibit impressive capabilities in understanding audio content for conversational QA tasks. However, these models struggle to accurately understand timestamps for temporal localization (e.g., Temporal Audio Grounding) and are restricted to short audio perception, leading to constrained capabilities on fine-grained tasks. We identify three key aspects that limit their temporal localization and long audio understanding: (i) timestamp representation, (ii) architecture, and (iii) data. To address this, we introduce TimeAudio, a novel method that empowers LALMs to connect their understanding of audio content with precise temporal perception. Specifically, we incorporate unique temporal markers to improve time-sensitive reasoning and apply an absolute time-aware encoding that explicitly grounds the acoustic features with absolute time information. Moreover, to achieve end-to-end long audio understanding, we introduce a segment-level token merging module to substantially reduce audio token redundancy and enhance the efficiency of information extraction. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing audio datasets into a new dataset focused on temporal tasks and establish a series of metrics to evaluate the fine-grained performance. Evaluations show strong performance across a variety of fine-grained tasks, such as dense captioning, temporal grounding, and timeline speech summarization, demonstrating TimeAudio's robust temporal localization and reasoning capabilities.
>
---
#### [new 007] CLARITY: Contextual Linguistic Adaptation and Accent Retrieval for Dual-Bias Mitigation in Text-to-Speech Generation
- **分类: cs.SD; cs.CL**

- **简介: CLARITY解决TTS中的口音和语言偏差问题。通过上下文语言适应（本地化文本）和检索增强口音提示，提升12种英语口音的准确性和公平性，同时保持高质量语音。**

- **链接: [https://arxiv.org/pdf/2511.11104v1](https://arxiv.org/pdf/2511.11104v1)**

> **作者:** Crystal Min Hui Poon; Pai Chet Ng; Xiaoxiao Miao; Immanuel Jun Kai Loh; Bowen Zhang; Haoyu Song; Ian Mcloughlin
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Instruction-guided text-to-speech (TTS) research has reached a maturity level where excellent speech generation quality is possible on demand, yet two coupled biases persist: accent bias, where models default to dominant phonetic patterns, and linguistic bias, where dialect-specific lexical and cultural cues are ignored. These biases are interdependent, as authentic accent generation requires both accent fidelity and localized text. We present Contextual Linguistic Adaptation and Retrieval for Inclusive TTS sYnthesis (CLARITY), a backbone-agnostic framework that addresses these biases through dual-signal optimization: (i) contextual linguistic adaptation that localizes input text to the target dialect, and (ii) retrieval-augmented accent prompting (RAAP) that supplies accent-consistent speech prompts. Across twelve English accents, CLARITY improves accent accuracy and fairness while maintaining strong perceptual quality.
>
---
#### [new 008] Synthetic Voices, Real Threats: Evaluating Large Text-to-Speech Models in Generating Harmful Audio
- **分类: cs.SD; cs.AI; cs.CR; cs.MM; eess.AS**

- **简介: 论文研究TTS模型内容安全，提出HARMGEN攻击套件，通过语义混淆和音频模态技巧绕过安全过滤，实验证明能显著降低有害内容拒绝率并提升语音毒性。**

- **链接: [https://arxiv.org/pdf/2511.10913v1](https://arxiv.org/pdf/2511.10913v1)**

> **作者:** Guangke Chen; Yuhui Wang; Shouling Ji; Xiapu Luo; Ting Wang
>
> **摘要:** Modern text-to-speech (TTS) systems, particularly those built on Large Audio-Language Models (LALMs), generate high-fidelity speech that faithfully reproduces input text and mimics specified speaker identities. While prior misuse studies have focused on speaker impersonation, this work explores a distinct content-centric threat: exploiting TTS systems to produce speech containing harmful content. Realizing such threats poses two core challenges: (1) LALM safety alignment frequently rejects harmful prompts, yet existing jailbreak attacks are ill-suited for TTS because these systems are designed to faithfully vocalize any input text, and (2) real-world deployment pipelines often employ input/output filters that block harmful text and audio. We present HARMGEN, a suite of five attacks organized into two families that address these challenges. The first family employs semantic obfuscation techniques (Concat, Shuffle) that conceal harmful content within text. The second leverages audio-modality exploits (Read, Spell, Phoneme) that inject harmful content through auxiliary audio channels while maintaining benign textual prompts. Through evaluation across five commercial LALMs-based TTS systems and three datasets spanning two languages, we demonstrate that our attacks substantially reduce refusal rates and increase the toxicity of generated speech. We further assess both reactive countermeasures deployed by audio-streaming platforms and proactive defenses implemented by TTS providers. Our analysis reveals critical vulnerabilities: deepfake detectors underperform on high-fidelity audio; reactive moderation can be circumvented by adversarial perturbations; while proactive moderation detects 57-93% of attacks. Our work highlights a previously underexplored content-centric misuse vector for TTS and underscore the need for robust cross-modal safeguards throughout training and deployment.
>
---
#### [new 009] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak
- **分类: cs.SD**

- **简介: 该论文针对大型音频-语言模型（LAMs）的对齐漏洞问题，提出StyleBreak框架，通过风格感知音频转换和查询自适应策略，系统性扰动语音属性，高效实现音频越狱攻击。**

- **链接: [https://arxiv.org/pdf/2511.10692v1](https://arxiv.org/pdf/2511.10692v1)**

> **作者:** Hongyi Li; Chengxuan Zhou; Chu Wang; Sicheng Liang; Yanting Chen; Qinlin Xie; Jiawei Ye; Jie Wu
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Large Audio-language Models (LAMs) have recently enabled powerful speech-based interactions by coupling audio encoders with Large Language Models (LLMs). However, the security of LAMs under adversarial attacks remains underexplored, especially through audio jailbreaks that craft malicious audio prompts to bypass alignment. Existing efforts primarily rely on converting text-based attacks into speech or applying shallow signal-level perturbations, overlooking the impact of human speech's expressive variations on LAM alignment robustness. To address this gap, we propose StyleBreak, a novel style-aware audio jailbreak framework that systematically investigates how diverse human speech attributes affect LAM alignment robustness. Specifically, StyleBreak employs a two-stage style-aware transformation pipeline that perturbs both textual content and audio to control linguistic, paralinguistic, and extralinguistic attributes. Furthermore, we develop a query-adaptive policy network that automatically searches for adversarial styles to enhance the efficiency of LAM jailbreak exploration. Extensive evaluations demonstrate that LAMs exhibit critical vulnerabilities when exposed to diverse human speech attributes. Moreover, StyleBreak achieves substantial improvements in attack effectiveness and efficiency across multiple attack paradigms, highlighting the urgent need for more robust alignment in LAMs.
>
---
#### [new 010] Towards Attribution of Generators and Emotional Manipulation in Cross-Lingual Synthetic Speech using Geometric Learning
- **分类: eess.AS**

- **简介: 该论文解决合成语音中情感和操纵源的细粒度追溯问题。提出MiCuNet框架，融合Speech Foundation Models与听觉特征，通过混合曲率投影机制实现多任务学习，有效预测情感及操纵源，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2511.10790v1](https://arxiv.org/pdf/2511.10790v1)**

> **作者:** Girish; Mohd Mujtaba Akhtar; Farhan Sheth; Muskaan Singh
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** In this work, we address the problem of finegrained traceback of emotional and manipulation characteristics from synthetically manipulated speech. We hypothesize that combining semantic-prosodic cues captured by Speech Foundation Models (SFMs) with fine-grained spectral dynamics from auditory representations can enable more precise tracing of both emotion and manipulation source. To validate this hypothesis, we introduce MiCuNet, a novel multitask framework for fine-grained tracing of emotional and manipulation attributes in synthetically generated speech. Our approach integrates SFM embeddings with spectrogram-based auditory features through a mixed-curvature projection mechanism that spans Hyperbolic, Euclidean, and Spherical spaces guided by a learnable temporal gating mechanism. Our proposed method adopts a multitask learning setup to simultaneously predict original emotions, manipulated emotions, and manipulation sources on the EmoFake dataset (EFD) across both English and Chinese subsets. MiCuNet yields consistent improvements, consistently surpassing conventional fusion strategies. To the best of our knowledge, this work presents the first study to explore a curvature-adaptive framework specifically tailored for multitask tracking in synthetic speech.
>
---
#### [new 011] Do AI Voices Learn Social Nuances? A Case of Politeness and Speech Rate
- **分类: cs.CL; cs.AI; cs.HC; cs.SD**

- **简介: 该论文研究AI语音是否能隐式学习人类社会细微差别（如礼貌通过减慢语速）。通过测试22个AI语音在礼貌与随意提示下的语速，发现礼貌条件语速显著减慢，证明AI能内化社会规范并复制心理细微差别。**

- **链接: [https://arxiv.org/pdf/2511.10693v1](https://arxiv.org/pdf/2511.10693v1)**

> **作者:** Eyal Rabin; Zohar Elyoseph; Rotem Israel-Fishelson; Adi Dali; Ravit Nussinson
>
> **摘要:** Voice-based artificial intelligence is increasingly expected to adhere to human social conventions, but can it learn implicit cues that are not explicitly programmed? This study investigates whether state-of-the-art text-to-speech systems have internalized the human tendency to reduce speech rate to convey politeness - a non-obvious prosodic marker. We prompted 22 synthetic voices from two leading AI platforms (AI Studio and OpenAI) to read a fixed script under both "polite and formal" and "casual and informal" conditions and measured the resulting speech duration. Across both AI platforms, the polite prompt produced slower speech than the casual prompt with very large effect sizes, an effect that was statistically significant for all of AI Studio's voices and for a large majority of OpenAI's voices. These results demonstrate that AI can implicitly learn and replicate psychological nuances of human communication, highlighting its emerging role as a social actor capable of reinforcing human social norms.
>
---
#### [new 012] Proactive Hearing Assistants that Isolate Egocentric Conversations
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文提出主动听力辅助系统，解决自动识别对话伙伴问题。利用自语音作为锚点，通过双模型架构实时分离对话，实现多对话场景中的有效隔离。**

- **链接: [https://arxiv.org/pdf/2511.11473v1](https://arxiv.org/pdf/2511.11473v1)**

> **作者:** Guilin Hu; Malek Itani; Tuochao Chen; Shyamnath Gollakota
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** We introduce proactive hearing assistants that automatically identify and separate the wearer's conversation partners, without requiring explicit prompts. Our system operates on egocentric binaural audio and uses the wearer's self-speech as an anchor, leveraging turn-taking behavior and dialogue dynamics to infer conversational partners and suppress others. To enable real-time, on-device operation, we propose a dual-model architecture: a lightweight streaming model runs every 12.5 ms for low-latency extraction of the conversation partners, while a slower model runs less frequently to capture longer-range conversational dynamics. Results on real-world 2- and 3-speaker conversation test sets, collected with binaural egocentric hardware from 11 participants totaling 6.8 hours, show generalization in identifying and isolating conversational partners in multi-conversation settings. Our work marks a step toward hearing assistants that adapt proactively to conversational dynamics and engagement. More information can be found on our website: https://proactivehearing.cs.washington.edu/
>
---
#### [new 013] Towards Fine-Grained Code-Switch Speech Translation with Semantic Space Alignment
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文解决代码切换语音翻译任务中的语义建模复杂性和数据稀缺问题。提出MoE语音投影器实现细粒度语义建模，设计多阶段训练和过渡损失，利用单语数据有效提升翻译性能。**

- **链接: [https://arxiv.org/pdf/2511.10670v1](https://arxiv.org/pdf/2511.10670v1)**

> **作者:** Yan Gao; Yazheng Yang; Zhibin Lan; Yidong Chen; Min Zhang; Daimeng Wei; Hui Huang; Jinsong Su
>
> **备注:** Working in progress
>
> **摘要:** Code-switching (CS) speech translation (ST) refers to translating speech that alternates between two or more languages into a target language text, which poses significant challenges due to the complexity of semantic modeling and the scarcity of CS data. Previous studies tend to rely on the model itself to implicitly learn semantic modeling during training, and resort to inefficient and costly manual annotations for these two challenges. To mitigate these limitations, we propose enhancing Large Language Models (LLMs) with a Mixture of Experts (MoE) speech projector, where each expert specializes in the semantic subspace of a specific language, enabling fine-grained modeling of speech features. Additionally, we introduce a multi-stage training paradigm that utilizes readily available monolingual automatic speech recognition (ASR) and monolingual ST data, facilitating speech-text alignment and improving translation capabilities. During training, we leverage a combination of language-specific loss and intra-group load balancing loss to guide the MoE speech projector in efficiently allocating tokens to the appropriate experts, across expert groups and within each group, respectively. To bridge the data gap across different training stages and improve adaptation to the CS scenario, we further employ a transition loss, enabling smooth transitions of data between stages, to effectively address the scarcity of high-quality CS speech translation data. Extensive experiments on widely used datasets demonstrate the effectiveness and generality of our approach.
>
---
#### [new 014] AV-Dialog: Spoken Dialogue Models with Audio-Visual Input
- **分类: cs.CL; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: AV-Dialog解决嘈杂多说话人环境下的语音对话问题。它提出首个多模态框架，融合音频与视觉输入实现目标说话者跟踪、轮次预测及响应生成。通过多任务训练，该模型在干扰下显著减少转录错误，提升对话质量与自然流畅度。**

- **链接: [https://arxiv.org/pdf/2511.11124v1](https://arxiv.org/pdf/2511.11124v1)**

> **作者:** Tuochao Chen; Bandhav Veluri; Hongyu Gong; Shyamnath Gollakota
>
> **摘要:** Dialogue models falter in noisy, multi-speaker environments, often producing irrelevant responses and awkward turn-taking. We present AV-Dialog, the first multimodal dialog framework that uses both audio and visual cues to track the target speaker, predict turn-taking, and generate coherent responses. By combining acoustic tokenization with multi-task, multi-stage training on monadic, synthetic, and real audio-visual dialogue datasets, AV-Dialog achieves robust streaming transcription, semantically grounded turn-boundary detection and accurate responses, resulting in a natural conversational flow. Experiments show that AV-Dialog outperforms audio-only models under interference, reducing transcription errors, improving turn-taking prediction, and enhancing human-rated dialogue quality. These results highlight the power of seeing as well as hearing for speaker-aware interaction, paving the way for {spoken} dialogue agents that perform {robustly} in real-world, noisy environments.
>
---
#### [new 015] AccKV: Towards Efficient Audio-Video LLMs Inference via Adaptive-Focusing and Cross-Calibration KV Cache Optimization
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 论文针对音频-视频大语言模型推理效率问题，提出AccKV框架。解决KV缓存过大导致的模态混淆与性能下降，通过自适应聚焦和交叉校准优化缓存，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2511.11106v1](https://arxiv.org/pdf/2511.11106v1)**

> **作者:** Zhonghua Jiang; Kui Chen; Kunxi Li; Keting Yin; Yiyun Zhou; Zhaode Wang; Chengfei Lv; Shengyu Zhang
>
> **摘要:** Recent advancements in Audio-Video Large Language Models (AV-LLMs) have enhanced their capabilities in tasks like audio-visual question answering and multimodal dialog systems. Video and audio introduce an extended temporal dimension, resulting in a larger key-value (KV) cache compared to static image embedding. A naive optimization strategy is to selectively focus on and retain KV caches of audio or video based on task. However, in the experiment, we observed that the attention of AV-LLMs to various modalities in the high layers is not strictly dependent on the task. In higher layers, the attention of AV-LLMs shifts more towards the video modality. In addition, we also found that directly integrating temporal KV of audio and spatial-temporal KV of video may lead to information confusion and significant performance degradation of AV-LLMs. If audio and video are processed indiscriminately, it may also lead to excessive compression or reservation of a certain modality, thereby disrupting the alignment between modalities. To address these challenges, we propose AccKV, an Adaptive-Focusing and Cross-Calibration KV cache optimization framework designed specifically for efficient AV-LLMs inference. Our method is based on layer adaptive focusing technology, selectively focusing on key modalities according to the characteristics of different layers, and enhances the recognition of heavy hitter tokens through attention redistribution. In addition, we propose a Cross-Calibration technique that first integrates inefficient KV caches within the audio and video modalities, and then aligns low-priority modalities with high-priority modalities to selectively evict KV cache of low-priority modalities. The experimental results show that AccKV can significantly improve the computational efficiency of AV-LLMs while maintaining accuracy.
>
---
## 更新

#### [replaced 001] Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard
- **分类: cs.SD; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.10222v2](https://arxiv.org/pdf/2511.10222v2)**

> **作者:** Yudong Yang; Xuezhen Zhang; Zhifeng Han; Siyin Wang; Jimin Zhuang; Zengrui Jin; Jing Shao; Guangzhi Sun; Chao Zhang
>
> **摘要:** Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
>
---
#### [replaced 002] Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research
- **分类: cs.RO; cs.HC; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.17760v2](https://arxiv.org/pdf/2509.17760v2)**

> **作者:** Austin Wilson; Sahar Kapasi; Zane Greene; Alexis E. Block
>
> **摘要:** Legacy (unsupported) robotic platforms often lose research utility when manufacturer support ends, preventing integration of modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot featuring upgraded beamforming microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud-based and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot user study validating conversational performance, the Enhanced NAO delivered significantly higher conversational quality and elicited stronger user preference compared to the NAO AI Edition, without increasing response latency. The added visual and thermal sensing modalities established a foundation for future perception-driven interaction. Beyond this implementation, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction.
>
---
#### [replaced 003] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2511.08252v2](https://arxiv.org/pdf/2511.08252v2)**

> **作者:** Yi Yang; Haowen Li; Tianxiang Li; Boyu Cao; Xiaohan Zhang; Liqun Chen; Qi Liu
>
> **备注:** AAAI 2026 (Oral)
>
> **摘要:** Text-to-music generation technology is progressing rapidly, creating new opportunities for musical composition and editing. However, existing music editing methods often fail to preserve the source music's temporal structure, including melody and rhythm, when altering particular attributes like instrument, genre, and mood. To address this challenge, this paper conducts an in-depth probing analysis on attention maps within AudioLDM 2, a diffusion-based model commonly used as the backbone for existing music editing methods. We reveal a key finding: cross-attention maps encompass details regarding distinct musical characteristics, and interventions on these maps frequently result in ineffective modifications. In contrast, self-attention maps are essential for preserving the temporal structure of the source music during its conversion into the target music. Building upon this understanding, we present Melodia, a training-free technique that selectively manipulates self-attention maps in particular layers during the denoising process and leverages an attention repository to store source music information, achieving accurate modification of musical characteristics while preserving the original structure without requiring textual descriptions of the source music. Additionally, we propose two novel metrics to better evaluate music editing methods. Both objective and subjective experiments demonstrate that our approach achieves superior results in terms of textual adherence and structural integrity across various datasets. This research enhances comprehension of internal mechanisms within music generation models and provides improved control for music creation.
>
---
#### [replaced 004] Improving Speech Emotion Recognition with Mutual Information Regularized Generative Model
- **分类: cs.SD; cs.LG**

- **链接: [https://arxiv.org/pdf/2510.10078v2](https://arxiv.org/pdf/2510.10078v2)**

> **作者:** Chung-Soo Ahn; Rajib Rana; Sunil Sivadas; Carlos Busso; Jagath C. Rajapakse
>
> **摘要:** Although speech emotion recognition (SER) research has been advanced, thanks to deep learning methods, it still suffers from obtaining inputs from large quality-labelled training data. Data augmentation methods have been attempted to mitigate this issue, generative models have shown success among them recently. We propose a data augmentation framework that is aided by cross-modal information transfer and mutual information regularization. Mutual information based metric can serve as an indicator for the quality. Furthermore, we expand this data augmentation scope to multimodal inputs, thanks to mutual information ensureing dependency between modalities. Our framework was tested on three benchmark datasets: IEMOCAP, MSP-IMPROV and MSP-Podcast. The implementation was designed to generate input features that are fed into last layer for emotion classification. Our framework improved the performance of emotion prediction against existing works. Also, we discovered that our framework is able to generate new inputs without any cross-modal information.
>
---
#### [replaced 005] HI-TransPA: Hearing Impairments Translation Personal Assistant
- **分类: cs.CL; cs.MM; cs.SD**

- **链接: [https://arxiv.org/pdf/2511.09915v2](https://arxiv.org/pdf/2511.09915v2)**

> **作者:** Zhiming Ma; Shiyu Gan; Junhao Zhao; Xianming Li; Qingyun Pan; Peidong Wang; Mingjun Pan; Yuhao Mo; Jiajie Cheng; Chengxin Chen; Zhonglun Cao; Chonghan Liu; Shi Cheng
>
> **摘要:** Hearing-impaired individuals often face significant barriers in daily communication due to the inherent challenges of producing clear speech. To address this, we introduce the Omni-Model paradigm into assistive technology and present HI-TransPA, an instruction-driven audio-visual personal assistant. The model fuses indistinct speech with lip dynamics, enabling both translation and dialogue within a single multimodal framework. To address the distinctive pronunciation patterns of hearing-impaired speech and the limited adaptability of existing models, we develop a multimodal preprocessing and curation pipeline that detects facial landmarks, stabilizes the lip region, and quantitatively evaluates sample quality. These quality scores guide a curriculum learning strategy that first trains on clean, high-confidence samples and progressively incorporates harder cases to strengthen model robustness. Architecturally, we employs a novel unified 3D-Resampler to efficiently encode the lip dynamics, which is critical for accurate interpretation. Experiments on purpose-built HI-Dialogue dataset show that HI-TransPA achieves state-of-the-art performance in both literal accuracy and semantic fidelity. Our work establishes a foundation for applying Omni-Models to assistive communication technology, providing an end-to-end modeling framework and essential processing tools for future research.
>
---
#### [replaced 006] CO-VADA: A Confidence-Oriented Voice Augmentation Debiasing Approach for Fair Speech Emotion Recognition
- **分类: eess.AS; cs.CL**

- **链接: [https://arxiv.org/pdf/2506.06071v2](https://arxiv.org/pdf/2506.06071v2)**

> **作者:** Yun-Shao Tsai; Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by IEEE ASRU 2025
>
> **摘要:** Bias in speech emotion recognition (SER) systems often stems from spurious correlations between speaker characteristics and emotional labels, leading to unfair predictions across demographic groups. Many existing debiasing methods require model-specific changes or demographic annotations, limiting their practical use. We present CO-VADA, a Confidence-Oriented Voice Augmentation Debiasing Approach that mitigates bias without modifying model architecture or relying on demographic information. CO-VADA identifies training samples that reflect bias patterns present in the training data and then applies voice conversion to alter irrelevant attributes and generate samples. These augmented samples introduce speaker variations that differ from dominant patterns in the data, guiding the model to focus more on emotion-relevant features. Our framework is compatible with various SER models and voice conversion tools, making it a scalable and practical solution for improving fairness in SER systems.
>
---
#### [replaced 007] Video Echoed in Music: Semantic, Temporal, and Rhythmic Alignment for Video-to-Music Generation
- **分类: cs.SD; cs.MM**

- **链接: [https://arxiv.org/pdf/2511.09585v2](https://arxiv.org/pdf/2511.09585v2)**

> **作者:** Xinyi Tong; Yiran Zhu; Jishang Chen; Chunru Zhan; Tianle Wang; Sirui Zhang; Nian Liu; Tiezheng Ge; Duo Xu; Xin Jin; Feng Yu; Song-Chun Zhu
>
> **摘要:** Video-to-Music generation seeks to generate musically appropriate background music that enhances audiovisual immersion for videos. However, current approaches suffer from two critical limitations: 1) incomplete representation of video details, leading to weak alignment, and 2) inadequate temporal and rhythmic correspondence, particularly in achieving precise beat synchronization. To address the challenges, we propose Video Echoed in Music (VeM), a latent music diffusion that generates high-quality soundtracks with semantic, temporal, and rhythmic alignment for input videos. To capture video details comprehensively, VeM employs a hierarchical video parsing that acts as a music conductor, orchestrating multi-level information across modalities. Modality-specific encoders, coupled with a storyboard-guided cross-attention mechanism (SG-CAtt), integrate semantic cues while maintaining temporal coherence through position and duration encoding. For rhythmic precision, the frame-level transition-beat aligner and adapter (TB-As) dynamically synchronize visual scene transitions with music beats. We further contribute a novel video-music paired dataset sourced from e-commerce advertisements and video-sharing platforms, which imposes stricter transition-beat synchronization requirements. Meanwhile, we introduce novel metrics tailored to the task. Experimental results demonstrate superiority, particularly in semantic relevance and rhythmic precision.
>
---
#### [replaced 008] Golden Tonnetz
- **分类: cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.21428v2](https://arxiv.org/pdf/2509.21428v2)**

> **作者:** Yusuke Imai
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** For example, in the chromatic circle, the twelve tones are represented by twelve points on a circle, and in Tonnetz, the relationships among harmonies are represented by a triangular lattice. Recently, we have shown that several arrangements of tones on the regular icosahedron can be associated with chromatic scales, whole-tone scales, major tones, and minor tones through the golden ratio. Here, we investigate another type of connection between music and the golden ratio. We show that there exists an arrangement of 7 tones on a golden triangle that can represent a given major/minor scale and its tonic, dominant, and subdominant chords by golden triangles. By applying this finding, we propose ``golden Tonnetz" which represents all the major/minor scales and triads by the golden triangles or gnomons and also represents relative, parallel, and leading-tone exchange transformations in Neo-Riemannian theory by transformations among the golden triangles and gnomons
>
---
#### [replaced 009] SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models
- **分类: eess.AS; cs.AI**

- **链接: [https://arxiv.org/pdf/2511.06606v2](https://arxiv.org/pdf/2511.06606v2)**

> **作者:** S Sakshi; Vaibhavi Lokegaonkar; Neil Zhang; Ramani Duraiswami; Sreyan Ghosh; Dinesh Manocha; Lie Lu
>
> **备注:** Project: https://sakshi113.github.io/spur/
>
> **摘要:** Spatial perception is central to auditory intelligence, enabling accurate understanding of real-world acoustic scenes and advancing human-level perception of the world around us. While recent large audio-language models (LALMs) show strong reasoning over complex audios, most operate on monaural inputs and lack the ability to capture spatial cues such as direction, elevation, and distance. We introduce SPUR, a lightweight, plug-in approach that equips LALMs with spatial perception through minimal architectural changes. SPUR consists of: (i) a First-Order Ambisonics (FOA) encoder that maps (W, X, Y, Z) channels to rotation-aware, listener-centric spatial features, integrated into target LALMs via a multimodal adapter; and (ii) SPUR-Set, a spatial QA dataset combining open-source FOA recordings with controlled simulations, emphasizing relative direction, elevation, distance, and overlap for supervised spatial reasoning. Fine-tuning our model on the SPUR-Set consistently improves spatial QA and multi-speaker attribution while preserving general audio understanding. SPUR provides a simple recipe that transforms monaural LALMs into spatially aware models. Extensive ablations validate the effectiveness of our approach.
>
---
#### [replaced 010] MUDAS: Mote-scale Unsupervised Domain Adaptation in Multi-label Sound Classification
- **分类: cs.AI; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2506.11331v2](https://arxiv.org/pdf/2506.11331v2)**

> **作者:** Jihoon Yun; Chengzhang Li; Dhrubojyoti Roy; Anish Arora
>
> **备注:** BuildSys 25
>
> **摘要:** Unsupervised Domain Adaptation (UDA) is essential for adapting machine learning models to new, unlabeled environments where data distribution shifts can degrade performance. Existing UDA algorithms are designed for single-label tasks and rely on significant computational resources, limiting their use in multi-label scenarios and in resource-constrained IoT devices. Overcoming these limitations is particularly challenging in contexts such as urban sound classification, where overlapping sounds and varying acoustics require robust, adaptive multi-label capabilities on low-power, on-device systems. To address these limitations, we introduce Mote-scale Unsupervised Domain Adaptation for Sounds (MUDAS), a UDA framework developed for multi-label sound classification in resource-constrained IoT settings. MUDAS efficiently adapts models by selectively retraining the classifier in situ using high-confidence data, minimizing computational and memory requirements to suit on-device deployment. Additionally, MUDAS incorporates class-specific adaptive thresholds to generate reliable pseudo-labels and applies diversity regularization to improve multi-label classification accuracy. In evaluations on the SONYC Urban Sound Tagging (SONYC-UST) dataset recorded at various New York City locations, MUDAS demonstrates notable improvements in classification accuracy over existing UDA algorithms, achieving good performance in a resource-constrained IoT setting.
>
---
