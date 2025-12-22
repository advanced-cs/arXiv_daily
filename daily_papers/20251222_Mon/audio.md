# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Zero-Shot Recognition of Dysarthric Speech Using Commercial Automatic Speech Recognition and Multimodal Large Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属零-shot语音识别任务，旨在解决商用ASR与MLLM对构音障碍语音识别性能差的问题。研究评估8种商用系统在TORGO数据集上的词错误率、语义保持与成本延迟，发现严重构音障碍下WER超49%，GPT-4o提示微调可降WER 7.36点。**

- **链接: [https://arxiv.org/pdf/2512.17474v1](https://arxiv.org/pdf/2512.17474v1)**

> **作者:** Ali Alsayegh; Tariq Masood
>
> **摘要:** Voice-based human-machine interaction is a primary modality for accessing intelligent systems, yet individuals with dysarthria face systematic exclusion due to recognition performance gaps. Whilst automatic speech recognition (ASR) achieves word error rates (WER) below 5% on typical speech, performance degrades dramatically for dysarthric speakers. Multimodal large language models (MLLMs) offer potential for leveraging contextual reasoning to compensate for acoustic degradation, yet their zero-shot capabilities remain uncharacterised. This study evaluates eight commercial speech-to-text services on the TORGO dysarthric speech corpus: four conventional ASR systems (AssemblyAI, Whisper large-v3, Deepgram Nova-3, Nova-3 Medical) and four MLLM-based systems (GPT-4o, GPT-4o Mini, Gemini 2.5 Pro, Gemini 2.5 Flash). Evaluation encompasses lexical accuracy, semantic preservation, and cost-latency trade-offs. Results demonstrate severity-dependent degradation: mild dysarthria achieves 3-5% WER approaching typical-speech benchmarks, whilst severe dysarthria exceeds 49% WER across all systems. A verbatim-transcription prompt yields architecture-specific effects: GPT-4o achieves 7.36 percentage point WER reduction with consistent improvement across all tested speakers, whilst Gemini variants exhibit degradation. Semantic metrics indicate that communicative intent remains partially recoverable despite elevated lexical error rates. These findings establish empirical baselines enabling evidence-based technology selection for assistive voice interface deployment.
>
---
#### [new 002] InstructDubber: Instruction-based Alignment for Zero-shot Movie Dubbing
- **分类: cs.SD**

- **简介: 该论文属电影配音任务，旨在解决零样本场景下唇形与情感韵律对齐难的问题。提出InstructDubber：用多模态大模型生成语音速率与情感指令，再通过指令蒸馏时长、校准情感来驱动音频合成，实现鲁棒的零样本配音。**

- **链接: [https://arxiv.org/pdf/2512.17154v1](https://arxiv.org/pdf/2512.17154v1)**

> **作者:** Zhedong Zhang; Liang Li; Gaoxiang Cong; Chunshan Liu; Yuhan Gao; Xiaowan Wang; Tao Gu; Yuankai Qi
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Movie dubbing seeks to synthesize speech from a given script using a specific voice, while ensuring accurate lip synchronization and emotion-prosody alignment with the character's visual performance. However, existing alignment approaches based on visual features face two key limitations: (1)they rely on complex, handcrafted visual preprocessing pipelines, including facial landmark detection and feature extraction; and (2) they generalize poorly to unseen visual domains, often resulting in degraded alignment and dubbing quality. To address these issues, we propose InstructDubber, a novel instruction-based alignment dubbing method for both robust in-domain and zero-shot movie dubbing. Specifically, we first feed the video, script, and corresponding prompts into a multimodal large language model to generate natural language dubbing instructions regarding the speaking rate and emotion state depicted in the video, which is robust to visual domain variations. Second, we design an instructed duration distilling module to mine discriminative duration cues from speaking rate instructions to predict lip-aligned phoneme-level pronunciation duration. Third, for emotion-prosody alignment, we devise an instructed emotion calibrating module, which finetunes an LLM-based instruction analyzer using ground truth dubbing emotion as supervision and predicts prosody based on the calibrated emotion analysis. Finally, the predicted duration and prosody, together with the script, are fed into the audio decoder to generate video-aligned dubbing. Extensive experiments on three major benchmarks demonstrate that InstructDubber outperforms state-of-the-art approaches across both in-domain and zero-shot scenarios.
>
---
#### [new 003] Do Foundational Audio Encoders Understand Music Structure?
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属音乐信息检索任务，探究预训练基础音频编码器（FAEs）对音乐结构分析（MSA）的有效性。针对FAEs在MSA中应用不足、影响因素不明的问题，作者系统评估11种FAEs，发现基于音乐数据的掩码语言建模自监督方法最有效。**

- **链接: [https://arxiv.org/pdf/2512.17209v1](https://arxiv.org/pdf/2512.17209v1)**

> **作者:** Keisuke Toyama; Zhi Zhong; Akira Takahashi; Shusuke Takahashi; Yuki Mitsufuji
>
> **摘要:** In music information retrieval (MIR) research, the use of pretrained foundational audio encoders (FAEs) has recently become a trend. FAEs pretrained on large amounts of music and audio data have been shown to improve performance on MIR tasks such as music tagging and automatic music transcription. However, their use for music structure analysis (MSA) remains underexplored. Although many open-source FAE models are available, only a small subset has been examined for MSA, and the impact of factors such as learning methods, training data, and model context length on MSA performance remains unclear. In this study, we conduct comprehensive experiments on 11 types of FAEs to investigate how these factors affect MSA performance. Our results demonstrate that FAEs using selfsupervised learning with masked language modeling on music data are particularly effective for MSA. These findings pave the way for future research in MSA.
>
---
#### [new 004] When De-noising Hurts: A Systematic Study of Speech Enhancement Effects on Modern Medical ASR Systems
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文研究语音增强对现代医疗ASR系统的影响，挑战“去噪必有益”的共识。作者系统评估MetricGAN+在4种先进ASR模型上的效果，发现去噪反而显著降低语义词错率（semWER），表明大模型已具强噪声鲁棒性，传统增强可能损害关键声学特征。**

- **链接: [https://arxiv.org/pdf/2512.17562v1](https://arxiv.org/pdf/2512.17562v1)**

> **作者:** Sujal Chondhekar; Vasanth Murukuri; Rushabh Vasani; Sanika Goyal; Rajshree Badami; Anushree Rana; Sanjana SN; Karthik Pandia; Sulabh Katiyar; Neha Jagadeesh; Sankalp Gulati
>
> **备注:** Technical Report
>
> **摘要:** Speech enhancement methods are commonly believed to improve the performance of automatic speech recognition (ASR) in noisy environments. However, the effectiveness of these techniques cannot be taken for granted in the case of modern large-scale ASR models trained on diverse, noisy data. We present a systematic evaluation of MetricGAN-plus-voicebank denoising on four state-of-the-art ASR systems: OpenAI Whisper, NVIDIA Parakeet, Google Gemini Flash 2.0, Parrotlet-a using 500 medical speech recordings under nine noise conditions. ASR performance is measured using semantic WER (semWER), a normalized word error rate (WER) metric accounting for domain-specific normalizations. Our results reveal a counterintuitive finding: speech enhancement preprocessing degrades ASR performance across all noise conditions and models. Original noisy audio achieves lower semWER than enhanced audio in all 40 tested configurations (4 models x 10 conditions), with degradations ranging from 1.1% to 46.6% absolute semWER increase. These findings suggest that modern ASR models possess sufficient internal noise robustness and that traditional speech enhancement may remove acoustic features critical for ASR. For practitioners deploying medical scribe systems in noisy clinical environments, our results indicate that preprocessing audio with noise reduction techniques might not just be computationally wasteful but also be potentially harmful to the transcription accuracy.
>
---
#### [new 005] LibriVAD: A Scalable Open Dataset with Deep Learning Benchmarks for Voice Activity Detection
- **分类: cs.SD; cs.LG**

- **简介: 该论文面向语音活动检测（VAD）任务，旨在解决现有公开数据集规模小、噪声可控性差、泛化评估不足的问题。作者构建了可扩展开源数据集LibriVAD，提出ViT-MFCC模型，并验证其在OOD场景下的优越性，同时开源全部资源。**

- **链接: [https://arxiv.org/pdf/2512.17281v1](https://arxiv.org/pdf/2512.17281v1)**

> **作者:** Ioannis Stylianou; Achintya kr. Sarkar; Nauman Dawalatabad; James Glass; Zheng-Hua Tan
>
> **摘要:** Robust Voice Activity Detection (VAD) remains a challenging task, especially under noisy, diverse, and unseen acoustic conditions. Beyond algorithmic development, a key limitation in advancing VAD research is the lack of large-scale, systematically controlled, and publicly available datasets. To address this, we introduce LibriVAD - a scalable open-source dataset derived from LibriSpeech and augmented with diverse real-world and synthetic noise sources. LibriVAD enables systematic control over speech-to-noise ratio, silence-to-speech ratio (SSR), and noise diversity, and is released in three sizes (15 GB, 150 GB, and 1.5 TB) with two variants (LibriVAD-NonConcat and LibriVAD-Concat) to support different experimental setups. We benchmark multiple feature-model combinations, including waveform, Mel-Frequency Cepstral Coefficients (MFCC), and Gammatone filter bank cepstral coefficients, and introduce the Vision Transformer (ViT) architecture for VAD. Our experiments show that ViT with MFCC features consistently outperforms established VAD models such as boosted deep neural network and convolutional long short-term memory deep neural network across seen, unseen, and out-of-distribution (OOD) conditions, including evaluation on the real-world VOiCES dataset. We further analyze the impact of dataset size and SSR on model generalization, experimentally showing that scaling up dataset size and balancing SSR noticeably and consistently enhance VAD performance under OOD conditions. All datasets, trained models, and code are publicly released to foster reproducibility and accelerate progress in VAD research.
>
---
#### [new 006] Robust TTS Training via Self-Purifying Flow Matching for the WildSpoof 2026 TTS Track
- **分类: cs.SD; cs.AI**

- **简介: 该论文面向TTS任务，解决野外语音中标签噪声导致的鲁棒性差问题。提出Self-Purifying Flow Matching（SPFM）方法，通过对比条件/无条件流匹配损失动态净化训练样本，在Supertonic模型上实现轻量高效适配，显著降低WER并保持高感知质量。**

- **链接: [https://arxiv.org/pdf/2512.17293v1](https://arxiv.org/pdf/2512.17293v1)**

> **作者:** June Young Yi; Hyeongju Kim; Juheon Lee
>
> **备注:** 2 pages, preprint, This work has been submitted to the IEEE for possible publication. Submitted to ICASSP 2026 SPGC (WildSpoof Challenge, TTS track)
>
> **摘要:** This paper presents a lightweight text-to-speech (TTS) system developed for the WildSpoof Challenge TTS Track. Our approach fine-tunes the recently released open-weight TTS model, \textit{Supertonic}\footnote{\url{https://github.com/supertone-inc/supertonic}}, with Self-Purifying Flow Matching (SPFM) to enable robust adaptation to in-the-wild speech. SPFM mitigates label noise by comparing conditional and unconditional flow matching losses on each sample, routing suspicious text--speech pairs to unconditional training while still leveraging their acoustic information. The resulting model achieves the lowest Word Error Rate (WER) among all participating teams, while ranking second in perceptual metrics such as UTMOS and DNSMOS. These findings demonstrate that efficient, open-weight architectures like Supertonic can be effectively adapted to diverse real-world speech conditions when combined with explicit noise-handling mechanisms such as SPFM.
>
---
#### [new 007] Training Text-to-Speech Model with Purely Synthetic Data: Feasibility, Sensitivity, and Generalization Capability
- **分类: cs.SD**

- **简介: 该论文属TTS模型训练任务，探究纯合成数据训练的可行性、敏感性与泛化能力。通过控制文本丰富度、说话人多样性、噪声水平和说话风格等变量，系统验证合成数据的有效性，并发现其在理想条件下可超越真实数据性能。**

- **链接: [https://arxiv.org/pdf/2512.17356v1](https://arxiv.org/pdf/2512.17356v1)**

> **作者:** Tingxiao Zhou; Leying Zhang; Zhengyang Chen; Yanmin Qian
>
> **备注:** 14 pages, 5 figures, received by National Conference on Man-Machine Speech Communication (NCMMSC2025)
>
> **摘要:** The potential of synthetic data in text-to-speech (TTS) model training has gained increasing attention, yet its rationality and effectiveness require systematic validation. In this study, we systematically investigate the feasibility of using purely synthetic data for TTS training and explore how various factors--including text richness, speaker diversity, noise levels, and speaking styles--affect model performance. Our experiments reveal that increasing speaker and text diversity significantly enhances synthesis quality and robustness. Cleaner training data with minimal noise further improves performance. Moreover, we find that standard speaking styles facilitate more effective model learning. Our experiments indicate that models trained on synthetic data have great potential to outperform those trained on real data under similar conditions, due to the absence of real-world imperfections and noise.
>
---
#### [new 008] Review of MEMS Speakers for Audio Applications
- **分类: eess.AS; cs.SD**

- **简介: 该论文属综述任务，旨在解决MEMS扬声器在全频段音频应用中的性能瓶颈问题。工作包括按驱动原理分类、对比1990–2025年性能指标、指出压电式主导地位，并分析挑战与创新方向以推动MEMS-only扬声器实用化。**

- **链接: [https://arxiv.org/pdf/2512.17708v1](https://arxiv.org/pdf/2512.17708v1)**

> **作者:** Nils Wittek; Anton Melnikov; Bert Kaiser; André Zimmermann
>
> **备注:** 37 pages, 6 figures
>
> **摘要:** Microelectromechanical systems (MEMS) speakers are compact, scalable alternatives to traditional voice coil speakers, promising improved sound quality through precise semiconductor manufacturing. This review provides an overview of the research landscape, including ultrasound pulse-based and thermoacoustic sound generation, classifying MEMS speakers by actuation principle: electrodynamic, piezoelectric, and electrostatic. A comparative analysis of performance indicators from 1990-2025 highlights the dominance of piezoelectric MEMS with direct air displacement, focusing on miniaturization and efficiency. The review outlines upcoming research challenges and identifies potential candidates for achieving full-spectrum audio performance. A focus on innovative approaches could lead to wideband adoption of MEMS-only speakers.
>
---
#### [new 009] When Pamplona sounds different: the soundscape transformation of San Fermin through intelligent acoustic sensors and a sound repository
- **分类: cs.CY; cs.SD**

- **简介: 该论文属城市声景监测任务，旨在量化圣费尔明节对潘普洛纳声环境的影响。作者部署低成本智能声学传感器网络，采集节前、节中、节后连续音频数据，分析声压级与声景模式变化，并构建公开声音库保存节日声遗产。**

- **链接: [https://arxiv.org/pdf/2512.17740v1](https://arxiv.org/pdf/2512.17740v1)**

> **作者:** Amaia Sagasti; Frederic Font
>
> **备注:** 46 pages, 27 figures
>
> **摘要:** This study presents a use-case of a network of low-cost acoustic smart sensors deployed in the city of Pamplona to analyse changes in the urban soundscape during the San Fermin Festival. The sensors were installed in different areas of the city before, during, and after the event, capturing continuous acoustic data. Our analysis reveals a significant transformation in the city's sonic environment during the festive period: overall sound pressure levels increase significantly, soundscape patterns change, and the acoustic landscape becomes dominated by sounds associated with human activity. These findings highlight the potential of distributed smart acoustic monitoring systems to characterize the temporal dynamics of urban soundscapes and underscore how the large-scale event of San Fermin drastically reshapes the overall acoustic dynamics of the city of Pamplona. Additionally, to complement the objective measurements, a curated collection of real San Fermin sound recordings has been created and made publicly available, preserving the festival's unique sonic heritage.
>
---
#### [new 010] Speech-FT: Merging Pre-trained And Fine-Tuned Speech Representation Models For Cross-Task Generalization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属语音表征学习任务，旨在解决微调后模型跨任务泛化能力下降的问题。提出Speech-FT两阶段框架：先抑制表征漂移的微调，再与预训练模型权重插值，显著提升ASR、说话人识别等多任务性能与泛化性。**

- **链接: [https://arxiv.org/pdf/2502.12672v3](https://arxiv.org/pdf/2502.12672v3)**

> **作者:** Tzu-Quan Lin; Wei-Ping Huang; Hao Tang; Hung-yi Lee
>
> **备注:** Published in IEEE Transactions on Audio, Speech, and Language Processing (TASLP). Model and code available at: https://github.com/nervjack2/Speech-FT
>
> **摘要:** Fine-tuning speech representation models can enhance performance on specific tasks but often compromises their cross-task generalization ability. This degradation is often caused by excessive changes in the representations, making it difficult to retain information learned during pre-training. Existing approaches, such as regularizing weight changes during fine-tuning, may fail to maintain sufficiently high feature similarity with the pre-trained model, and thus could possibly lose cross-task generalization. To address this issue, we propose Speech-FT, a novel two-stage fine-tuning framework designed to maintain cross-task generalization while benefiting from fine-tuning. Speech-FT first applies fine-tuning specifically designed to reduce representational drift, followed by weight-space interpolation with the pre-trained model to restore cross-task generalization. Extensive experiments on HuBERT, wav2vec 2.0, DeCoAR 2.0, and WavLM Base+ demonstrate that Speech-FT consistently improves performance across a wide range of supervised, unsupervised, and multitask fine-tuning scenarios. Moreover, Speech-FT achieves superior cross-task generalization compared to fine-tuning baselines that explicitly constrain weight changes, such as weight-space regularization and LoRA fine-tuning. Our analysis reveals that Speech-FT maintains higher feature similarity to the pre-trained model compared to alternative strategies, despite allowing larger weight-space updates. Notably, Speech-FT achieves significant improvements on the SUPERB benchmark. For example, when fine-tuning HuBERT on automatic speech recognition, Speech-FT is able to reduce phone error rate from 5.17% to 3.94%, lower word error rate from 6.38% to 5.75%, and increase speaker identification accuracy from 81.86% to 84.11%. Speech-FT provides a simple yet powerful solution for further refining speech representation models after pre-training.
>
---
## 更新

#### [replaced 001] Unified Acoustic Representations for Screening Neurological and Respiratory Pathologies from Voice
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出MARVEL框架，属多任务语音健康筛查任务，旨在用统一声学表征同时检测9种神经/呼吸/嗓音疾病。它采用双分支共享骨干网络，仅需衍生声学特征，不传原始音频，提升隐私性与泛化性，在大型数据集上显著优于基线模型。**

- **链接: [https://arxiv.org/pdf/2508.20717v2](https://arxiv.org/pdf/2508.20717v2)**

> **作者:** Ran Piao; Yuan Lu; Hareld Kemps; Tong Xia; Aaqib Saeed
>
> **摘要:** Voice-based health assessment offers unprecedented opportunities for scalable, non-invasive disease screening, yet existing approaches typically focus on single conditions and fail to leverage the rich, multi-faceted information embedded in speech. We present MARVEL (Multi-task Acoustic Representations for Voice-based Health Analysis), a privacy-conscious multitask learning framework that simultaneously detects nine distinct neurological, respiratory, and voice disorders using only derived acoustic features, eliminating the need for raw audio transmission. Our dual-branch architecture employs specialized encoders with task-specific heads sharing a common acoustic backbone, enabling effective cross-condition knowledge transfer. Evaluated on the large-scale Bridge2AI-Voice v2.0 dataset, MARVEL achieves an overall AUROC of 0.78, with exceptional performance on neurological disorders (AUROC = 0.89), particularly for Alzheimer's disease/mild cognitive impairment (AUROC = 0.97). Our framework consistently outperforms single-modal baselines by 5-19% and surpasses state-of-the-art self-supervised models on 7 of 9 tasks, while correlation analysis reveals that the learned representations exhibit meaningful similarities with established acoustic features, indicating that the model's internal representations are consistent with clinically recognized acoustic patterns. By demonstrating that a single unified model can effectively screen for diverse conditions, this work establishes a foundation for deployable voice-based diagnostics in resource-constrained and remote healthcare settings.
>
---
#### [replaced 002] Set-theoretic solution for the tuning problem
- **分类: cs.SD; eess.AS**

- **简介: 该论文属音乐声学与数学交叉任务，旨在解决非谐波乐器的动态调音问题。提出基于集合论的量化 consonance（协和度）新框架，定义“亲和度”与“谐波性”两个测度，生成适配不同音色的动态音阶，统一谱干涉与谐波性对协和度的贡献。**

- **链接: [https://arxiv.org/pdf/2506.13969v2](https://arxiv.org/pdf/2506.13969v2)**

> **作者:** Vsevolod Vladimirovich Deriushkin
>
> **摘要:** In this paper I want to suggest a new solution to the problem of musical tuning. On one hand, I see it as a generalization of Just Intonation (JI) to inharmonic timbers, on another, as a unification of spectral interference and harmonicity contributions to consonance within a single framework. The main achievement of the work is the ability to mathematically quantify the phenomenon of musical consonance using set theory. That quantification is done by defining two measures of consonance: affinity and harmonicity. These measures naturally generate sets of intervals that can be used as dynamic tuning systems. The paper is aimed at a broad audience of people who may not be skilled in music and tuning theory or mathematics. Thus, I attempt to give as much details and explanations as I can, while keeping the number of pages as low as possible.
>
---
#### [replaced 003] Towards a Single ASR Model That Generalizes to Disordered Speech
- **分类: eess.AS**

- **简介: 该论文属语音识别（ASR）任务，旨在提升模型对言语障碍者语音的泛化能力。作者将约1000小时的无序语音数据用于微调先进ASR模型，显著提升其在提示语和自发对话两类障碍语音上的识别准确率（+33%、+26%），且不损害标准语音性能，推动无障碍语音技术发展。**

- **链接: [https://arxiv.org/pdf/2412.19315v2](https://arxiv.org/pdf/2412.19315v2)**

> **作者:** Jimmy Tobin; Katrin Tomanek; Subhashini Venugopalan
>
> **备注:** Accepted at ICASSP 2025
>
> **摘要:** This study investigates the impact of integrating a dataset of disordered speech recordings ($\sim$1,000 hours) into the fine-tuning of a near state-of-the-art ASR baseline system. Contrary to what one might expect, despite the data being less than 1% of the training data of the ASR system, we find a considerable improvement in disordered speech recognition accuracy. Specifically, we observe a 33% improvement on prompted speech, and a 26% improvement on a newly gathered spontaneous, conversational dataset of disordered speech. Importantly, there is no significant performance decline on standard speech recognition benchmarks. Further, we observe that the proposed tuning strategy helps close the gap between the baseline system and personalized models by 64% highlighting the significant progress as well as the room for improvement. Given the substantial benefits of our findings, this experiment suggests that from a fairness perspective, incorporating a small fraction of high quality disordered speech data in a training recipe is an easy step that could be done to make speech technology more accessible for users with speech disabilities.
>
---
#### [replaced 004] Fun-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-ASR，一种面向实际部署的LLM增强型语音识别系统。旨在解决LLM幻觉导致ASR实用性能下降的问题，通过数据/模型扩展、LLM深度集成与强化学习，提升流式识别、抗噪、语码转换等能力，在真实工业数据集上达到SOTA。**

- **链接: [https://arxiv.org/pdf/2509.12508v4](https://arxiv.org/pdf/2509.12508v4)**

> **作者:** Keyu An; Yanni Chen; Zhigao Chen; Chong Deng; Zhihao Du; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Ying Liu; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Haoxu Wang; Wen Wang; Wupeng Wang; Yuzhong Wu; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou; Yanqiao Zhu
>
> **备注:** Authors are listed in alphabetical order. Work in progress
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present Fun-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, Fun-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, Fun-ASR achieves state-of-the-art performance on real application datasets, demonstrating its effectiveness and robustness in practical settings. The code and models are accessible at https://github.com/FunAudioLLM/Fun-ASR .
>
---
#### [replaced 005] Fine-Tuning Large Audio-Language Models with LoRA for Precise Temporal Localization of Prolonged Exposure Therapy Elements
- **分类: eess.AS; cs.CL; cs.HC**

- **简介: 该论文属多模态时序定位任务，旨在自动定位PE疗法录音中三阶段（P1–P3）的起止时间。提出用LoRA微调Qwen2-Audio模型，以30秒音文窗口输入，预测归一化边界偏移，MAE达5.3秒，满足临床容错要求。**

- **链接: [https://arxiv.org/pdf/2506.09707v4](https://arxiv.org/pdf/2506.09707v4)**

> **作者:** Suhas BN; Andrew M. Sherrill; Jyoti Alaparthi; Dominik Mattioli; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Prolonged Exposure (PE) therapy is an effective treatment for post-traumatic stress disorder (PTSD), but evaluating therapist fidelity remains labor-intensive due to the need for manual review of session recordings. We present a method for the automatic temporal localization of key PE fidelity elements, identifying their start and stop times, directly from session audio and transcripts. Our approach fine-tunes a large pre-trained audio-language model, Qwen2-Audio, using Low-Rank Adaptation (LoRA) to process focused 30-second windows of audio-transcript input. Fidelity labels for three core protocol phases, therapist orientation (P1), imaginal exposure (P2), and post-imaginal processing (P3), are generated via LLM-based prompting and verified by trained raters. The model is trained to predict normalized boundary offsets using soft supervision guided by task-specific prompts. On a dataset of 308 real PE sessions, our best configuration (LoRA rank 8, 30s windows) achieves a mean absolute error (MAE) of 5.3s across tasks, within typical rater tolerance for timestamp review, enabling practical fidelity QC. We further analyze the effects of window size and LoRA rank, highlighting the importance of context granularity and model adaptation. This work introduces a privacy-preserving, scalable framework for fidelity tracking in PE therapy, with potential to support clinician training, supervision, and quality assurance.
>
---
