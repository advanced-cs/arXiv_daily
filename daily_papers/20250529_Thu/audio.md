# 音频 cs.SD;  eess.SP

- **最新发布 29 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] On-the-fly Routing for Zero-shot MoE Speaker Adaptation of Speech Foundation Models for Dysarthric Speech Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对构音障碍语音识别任务，解决零样本下高效适配不同说话人的难题。提出基于MoE的动态路由框架，通过实时预测说话人依赖参数，结合性别与障碍程度专家模块，并用KL散度增强专家多样性，实现零样本适配与实时处理，显著降低词错误率并提升速度。**

- **链接: [http://arxiv.org/pdf/2505.22072v1](http://arxiv.org/pdf/2505.22072v1)**

> **作者:** Shujie HU; Xurong Xie; Mengzhe Geng; Jiajun Deng; Huimeng Wang; Guinan Li; Chengxi Deng; Tianzi Wang; Mingyu Cui; Helen Meng; Xunying Liu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** This paper proposes a novel MoE-based speaker adaptation framework for foundation models based dysarthric speech recognition. This approach enables zero-shot adaptation and real-time processing while incorporating domain knowledge. Speech impairment severity and gender conditioned adapter experts are dynamically combined using on-the-fly predicted speaker-dependent routing parameters. KL-divergence is used to further enforce diversity among experts and their generalization to unseen speakers. Experimental results on the UASpeech corpus suggest that on-the-fly MoE-based adaptation produces statistically significant WER reductions of up to 1.34% absolute (6.36% relative) over the unadapted baseline HuBERT/WavLM models. Consistent WER reductions of up to 2.55% absolute (11.44% relative) and RTF speedups of up to 7 times are obtained over batch-mode adaptation across varying speaker-level data quantities. The lowest published WER of 16.35% (46.77% on very low intelligibility) is obtained.
>
---
#### [new 002] Effective and Efficient One-pass Compression of Speech Foundation Models Using Sparsity-aware Self-pinching Gates
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出语音模型单阶段压缩方法，通过层级自挤压门控（含单阈值）联合训练与剪枝，解决传统方法分阶段耗时且精度损失问题。实验表明，其将wav2vec2.0和HuBERT模型参数分别压缩65%和60%，保持最低WER（7.05%）且提速25%。**

- **链接: [http://arxiv.org/pdf/2505.22608v1](http://arxiv.org/pdf/2505.22608v1)**

> **作者:** Haoning Xu; Zhaoqing Li; Youjun Chen; Huimeng Wang; Guinan Li; Mengzhe Geng; Chengxi Deng; Xunying Liu
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This paper presents a novel approach for speech foundation models compression that tightly integrates model pruning and parameter update into a single stage. Highly compact layer-level tied self-pinching gates each containing only a single learnable threshold are jointly trained with uncompressed models and used in fine-grained neuron level pruning. Experiments conducted on the LibriSpeech-100hr corpus suggest that our approach reduces the number of parameters of wav2vec2.0-base and HuBERT-large models by 65% and 60% respectively, while incurring no statistically significant word error rate (WER) increase on the test-clean dataset. Compared to previously published methods on the same task, our approach not only achieves the lowest WER of 7.05% on the test-clean dataset under a comparable model compression ratio of 4.26x, but also operates with at least 25% less model compression time.
>
---
#### [new 003] Towards General Discrete Speech Codec for Complex Acoustic Environments: A Study of Reconstruction and Downstream Task Consistency
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究提升语音编解码器在复杂声学环境中的鲁棒性。针对现有方法在噪声环境下重建质量和下游任务（如语音识别）表现下降的问题，提出ERSB基准，评估编解码器对语音/非语音细节的重建能力及下游任务一致性。实验表明复杂环境显著降低这两项指标，指出现有模型不足及改进方向。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22515v1](http://arxiv.org/pdf/2505.22515v1)**

> **作者:** Haoran Wang; Guanyu Chen; Bohan Li; Hankun Wang; Yiwei Guo; Zhihan Li; Xie Chen; Kai Yu
>
> **备注:** Initial Upload
>
> **摘要:** Neural speech codecs excel in reconstructing clean speech signals; however, their efficacy in complex acoustic environments and downstream signal processing tasks remains underexplored. In this study, we introduce a novel benchmark named Environment-Resilient Speech Codec Benchmark (ERSB) to systematically evaluate whether neural speech codecs are environment-resilient. Specifically, we assess two key capabilities: (1) robust reconstruction, which measures the preservation of both speech and non-speech acoustic details, and (2) downstream task consistency, which ensures minimal deviation in downstream signal processing tasks when using reconstructed speech instead of the original. Our comprehensive experiments reveal that complex acoustic environments significantly degrade signal reconstruction and downstream task consistency. This work highlights the limitations of current speech codecs and raises a future direction that improves them for greater environmental resilience.
>
---
#### [new 004] RESOUND: Speech Reconstruction from Silent Videos via Acoustic-Semantic Decomposed Modeling
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于唇动到语音合成（L2S）任务，旨在解决无声视频语音重建的准确性与自然度不足问题。提出RESOUND系统，通过声学-语义分解建模分离韵律与语言特征，结合语音单元与梅尔频谱生成波形，提升可懂度与表达自然度，实验验证有效性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.22024v1](http://arxiv.org/pdf/2505.22024v1)**

> **作者:** Long-Khanh Pham; Thanh V. T. Tran; Minh-Tan Pham; Van Nguyen
>
> **备注:** accepted in Interspeech 2025
>
> **摘要:** Lip-to-speech (L2S) synthesis, which reconstructs speech from visual cues, faces challenges in accuracy and naturalness due to limited supervision in capturing linguistic content, accents, and prosody. In this paper, we propose RESOUND, a novel L2S system that generates intelligible and expressive speech from silent talking face videos. Leveraging source-filter theory, our method involves two components: an acoustic path to predict prosody and a semantic path to extract linguistic features. This separation simplifies learning, allowing independent optimization of each representation. Additionally, we enhance performance by integrating speech units, a proven unsupervised speech representation technique, into waveform generation alongside mel-spectrograms. This allows RESOUND to synthesize prosodic speech while preserving content and speaker identity. Experiments conducted on two standard L2S benchmarks confirm the effectiveness of the proposed method across various metrics.
>
---
#### [new 005] Weakly Supervised Data Refinement and Flexible Sequence Compression for Efficient Thai LLM-based ASR
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出EThai-ASR，首个基于LLM的泰语ASR系统，针对低资源场景下数据稀缺和高计算需求问题。通过自进化数据精炼策略优化弱标签数据提升语音编码器，并设计可插拔序列压缩模块降低计算量，实验达最优效果。**

- **链接: [http://arxiv.org/pdf/2505.22063v1](http://arxiv.org/pdf/2505.22063v1)**

> **作者:** Mingchen Shao; Xinfa Zhu; Chengyou Wang; Bingshen Mu; Hai Li; Ying Yan; Junhui Liu; Danming Xie; Lei Xie
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Despite remarkable achievements, automatic speech recognition (ASR) in low-resource scenarios still faces two challenges: high-quality data scarcity and high computational demands. This paper proposes EThai-ASR, the first to apply large language models (LLMs) to Thai ASR and create an efficient LLM-based ASR system. EThai-ASR comprises a speech encoder, a connection module and a Thai LLM decoder. To address the data scarcity and obtain a powerful speech encoder, EThai-ASR introduces a self-evolving data refinement strategy to refine weak labels, yielding an enhanced speech encoder. Moreover, we propose a pluggable sequence compression module used in the connection module with three modes designed to reduce the sequence length, thus decreasing computational demands while maintaining decent performance. Extensive experiments demonstrate that EThai-ASR has achieved state-of-the-art accuracy in multiple datasets. We release our refined text transcripts to promote further research.
>
---
#### [new 006] AudioGenie: A Training-Free Multi-Agent Framework for Diverse Multimodality-to-Multiaudio Generation
- **分类: cs.SD; cs.MA; cs.MM; eess.AS**

- **简介: 该论文属于多模态到多音频生成（MM2MA）任务，解决多模态输入生成多样化音频（如音效、语音等）时面临的数据稀缺和模型能力不足问题。提出AudioGenie框架，采用双层多智能体系统：生成团队通过任务分解、MoE动态选择与迭代优化生成音频；监督团队通过反馈确保时空一致性和输出质量。建立首个MM2MA基准MA-Bench，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22053v1](http://arxiv.org/pdf/2505.22053v1)**

> **作者:** Yan Rong; Jinting Wang; Shan Yang; Guangzhi Lei; Li Liu
>
> **摘要:** Multimodality-to-Multiaudio (MM2MA) generation faces significant challenges in synthesizing diverse and contextually aligned audio types (e.g., sound effects, speech, music, and songs) from multimodal inputs (e.g., video, text, images), owing to the scarcity of high-quality paired datasets and the lack of robust multi-task learning frameworks. Recently, multi-agent system shows great potential in tackling the above issues. However, directly applying it to MM2MA task presents three critical challenges: (1) inadequate fine-grained understanding of multimodal inputs (especially for video), (2) the inability of single models to handle diverse audio events, and (3) the absence of self-correction mechanisms for reliable outputs. To this end, we propose AudioGenie, a novel training-free multi-agent system featuring a dual-layer architecture with a generation team and a supervisor team. For the generation team, a fine-grained task decomposition and an adaptive Mixture-of-Experts (MoE) collaborative entity are designed for dynamic model selection, and a trial-and-error iterative refinement module is designed for self-correction. The supervisor team ensures temporal-spatial consistency and verifies outputs through feedback loops. Moreover, we build MA-Bench, the first benchmark for MM2MA tasks, comprising 198 annotated videos with multi-type audios. Experiments demonstrate that our AudioGenie outperforms state-of-the-art (SOTA) methods across 9 metrics in 8 tasks. User study further validate the effectiveness of the proposed method in terms of quality, accuracy, alignment, and aesthetic. The anonymous project website with samples can be found at https://audiogenie.github.io/.
>
---
#### [new 007] Delayed-KD: Delayed Knowledge Distillation based CTC for Low-Latency Streaming ASR
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于低延迟流式语音识别任务，解决小块分帧导致的精度下降和输出延迟问题。提出Delayed-KD方法，利用延迟知识蒸馏与Temporal Alignment Buffer（TAB）对齐CTC输出，控制延迟，提升精度。在中文数据集上，40ms延迟下达5.42%字符错误率，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.22069v1](http://arxiv.org/pdf/2505.22069v1)**

> **作者:** Longhao Li; Yangze Li; Hongfei Xue; Jie Liu; Shuai Fang; Kai Wang; Lei Xie
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** CTC-based streaming ASR has gained significant attention in real-world applications but faces two main challenges: accuracy degradation in small chunks and token emission latency. To mitigate these challenges, we propose Delayed-KD, which applies delayed knowledge distillation on CTC posterior probabilities from a non-streaming to a streaming model. Specifically, with a tiny chunk size, we introduce a Temporal Alignment Buffer (TAB) that defines a relative delay range compared to the non-streaming teacher model to align CTC outputs and mitigate non-blank token mismatches. Additionally, TAB enables fine-grained control over token emission delay. Experiments on 178-hour AISHELL-1 and 10,000-hour WenetSpeech Mandarin datasets show consistent superiority of Delayed-KD. Impressively, Delayed-KD at 40 ms latency achieves a lower character error rate (CER) of 5.42% on AISHELL-1, comparable to the competitive U2++ model running at 320 ms latency.
>
---
#### [new 008] Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对呼吸音分类任务中数据不足导致性能受限及集成模型推理成本高的问题，提出基于软标签的架构无关知识蒸馏方法。通过将多个教师模型的知识蒸馏至学生模型（甚至单教师与学生同架构），在ICHBI数据集获64.39分的新SOTA，超越先前最佳结果0.85分，验证了该方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22027v1](http://arxiv.org/pdf/2505.22027v1)**

> **作者:** Miika Toikkanen; June-Woo Kim
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Respiratory sound datasets are limited in size and quality, making high performance difficult to achieve. Ensemble models help but inevitably increase compute cost at inference time. Soft label training distills knowledge efficiently with extra cost only at training. In this study, we explore soft labels for respiratory sound classification as an architecture-agnostic approach to distill an ensemble of teacher models into a student model. We examine different variations of our approach and find that even a single teacher, identical to the student, considerably improves performance beyond its own capability, with optimal gains achieved using only a few teachers. We achieve the new state-of-the-art Score of 64.39 on ICHBI, surpassing the previous best by 0.85 and improving average Scores across architectures by more than 1.16. Our results highlight the effectiveness of knowledge distillation with soft labels for respiratory sound classification, regardless of size or architecture.
>
---
#### [new 009] Visual Cues Support Robust Turn-taking Prediction in Noise
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究噪音环境下对话轮换预测，发现现有音频模型在10dB音乐噪声中准确率骤降至52%。提出多模态模型（融合视觉特征）将准确率提升至72%，但泛化至新噪声类型受限。任务：提升噪声中预测鲁棒性；问题：模型对噪音敏感；方法：多模态建模与数据依赖性分析。**

- **链接: [http://arxiv.org/pdf/2505.22088v1](http://arxiv.org/pdf/2505.22088v1)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** 5 pages
>
> **摘要:** Accurate predictive turn-taking models (PTTMs) are essential for naturalistic human-robot interaction. However, little is known about their performance in noise. This study therefore explores PTTM performance in types of noise likely to be encountered once deployed. Our analyses reveal PTTMs are highly sensitive to noise. Hold/shift accuracy drops from 84% in clean speech to just 52% in 10 dB music noise. Training with noisy data enables a multimodal PTTM, which includes visual features to better exploit visual cues, with 72% accuracy in 10 dB music noise. The multimodal PTTM outperforms the audio-only PTTM across all noise types and SNRs, highlighting its ability to exploit visual cues; however, this does not always generalise to new types of noise. Analysis also reveals that successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions. We make code publicly available for future research.
>
---
#### [new 010] FGS-Audio: Fixed-Decoder Framework for Audio Steganography with Adversarial Perturbation Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出FGS-Audio框架，针对现有音频隐写需复杂训练及预训练模型的问题，采用固定解码器与对抗扰动生成技术。通过优化扰动嵌入秘密信息，设计轻量解码器实现高效提取，提升隐写音频抗分析能力及质量（PSNR超SOTA 10dB），无需依赖大模型。**

- **链接: [http://arxiv.org/pdf/2505.22266v1](http://arxiv.org/pdf/2505.22266v1)**

> **作者:** Jialin Yan; Yu Cheng; Zhaoxia Yin; Xinpeng Zhang; Shilin Wang; Tanfeng Sun; Xinghao Jiang
>
> **摘要:** The rapid development of Artificial Intelligence Generated Content (AIGC) has made high-fidelity generated audio widely available across the Internet, offering an abundant and versatile source of cover signals for covert communication. Driven by advances in deep learning, current audio steganography frameworks are mainly based on encoding-decoding network architectures. While these methods greatly improve the security of audio steganography, they typically employ elaborate training workflows and rely on extensive pre-trained models. To address the aforementioned issues, this paper pioneers a Fixed-Decoder Framework for Audio Steganography with Adversarial Perturbation Generation (FGS-Audio). The adversarial perturbations that carry secret information are embedded into cover audio to generate stego audio. The receiver only needs to share the structure and weights of the fixed decoding network to accurately extract the secret information from the stego audio, thus eliminating the reliance on large pre-trained models. In FGS-Audio, we propose an audio Adversarial Perturbation Generation (APG) strategy and design a lightweight fixed decoder. The fixed decoder guarantees reliable extraction of the hidden message, while the adversarial perturbations are optimized to keep the stego audio perceptually and statistically close to the cover audio, thereby improving resistance to steganalysis. The experimental results show that the method exhibits excellent anti-steganalysis performance under different relative payloads, outperforming existing SOTA approaches. In terms of stego audio quality, FGS-Audio achieves an average PSNR improvement of over 10 dB compared to SOTA method.
>
---
#### [new 011] Developing a Top-tier Framework in Naturalistic Conditions Challenge for Categorized Emotion Prediction: From Speech Foundation Models and Learning Objective to Data Augmentation and Engineering Choices
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别（SER）任务，针对自然情感表达的主观性和数据标签不平衡问题，提出SAILER系统。通过优化模型基础架构、学习目标、数据增强及工程选择，在INTERSPEECH 2025挑战赛中，单模型Macro-F1超0.4，超越95%参赛者，三模型集成跻身前三。**

- **链接: [http://arxiv.org/pdf/2505.22133v1](http://arxiv.org/pdf/2505.22133v1)**

> **作者:** Tiantian Feng; Thanathai Lertpetchpun; Dani Byrd; Shrikanth Narayanan
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** Speech emotion recognition (SER), particularly for naturally expressed emotions, remains a challenging computational task. Key challenges include the inherent subjectivity in emotion annotation and the imbalanced distribution of emotion labels in datasets. This paper introduces the \texttt{SAILER} system developed for participation in the INTERSPEECH 2025 Emotion Recognition Challenge (Task 1). The challenge dataset, which contains natural emotional speech from podcasts, serves as a valuable resource for studying imbalanced and subjective emotion annotations. Our system is designed to be simple, reproducible, and effective, highlighting critical choices in modeling, learning objectives, data augmentation, and engineering choices. Results show that even a single system (without ensembling) can outperform more than 95\% of the submissions, with a Macro-F1 score exceeding 0.4. Moreover, an ensemble of three systems further improves performance, achieving a competitively ranked score (top-3 performing team). Our model is at: https://github.com/tiantiaf0627/vox-profile-release.
>
---
#### [new 012] Two-stage Audio-Visual Target Speaker Extraction System for Real-Time Processing On Edge Device
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频视觉目标说话人提取任务，提出两阶段系统以解决现有方法计算复杂度高、无法实时处理于边缘设备的问题。首阶段用紧凑网络基于视觉信息进行语音活动检测，次阶段结合音频与检测结果分离目标语音，有效降噪且资源消耗低。**

- **链接: [http://arxiv.org/pdf/2505.22229v1](http://arxiv.org/pdf/2505.22229v1)**

> **作者:** Zixuan Li; Xueliang Zhang; Lei Miao; Zhipeng Yan
>
> **摘要:** Audio-Visual Target Speaker Extraction (AVTSE) aims to isolate a target speaker's voice in a multi-speaker environment with visual cues as auxiliary. Most of the existing AVTSE methods encode visual and audio features simultaneously, resulting in extremely high computational complexity and making it impractical for real-time processing on edge devices. To tackle this issue, we proposed a two-stage ultra-compact AVTSE system. Specifically, in the first stage, a compact network is employed for voice activity detection (VAD) using visual information. In the second stage, the VAD results are combined with audio inputs to isolate the target speaker's voice. Experiments show that the proposed system effectively suppresses background noise and interfering voices while spending little computational resources.
>
---
#### [new 013] Overlap-Adaptive Hybrid Speaker Diarization and ASR-Aware Observation Addition for MISP 2025 Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对MISP 2025挑战的说话人日志与语音识别任务，解决重叠语音分割及低信噪比下的识别问题。提出混合分割方法（WavLM端到端与传统聚类）优化不同重叠程度处理，并开发ASR感知观测添加技术提升低SNR场景下的GSS性能。系统集成两模块获Track2（9.48% CER）与Track3（11.56% cpCER）第一，验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.22013v1](http://arxiv.org/pdf/2505.22013v1)**

> **作者:** Shangkun Huang; Yuxuan Du; Jingwen Yang; Dejun Zhang; Xupeng Jia; Jing Deng; Jintao Kang; Rong Zheng
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This paper presents the system developed to address the MISP 2025 Challenge. For the diarization system, we proposed a hybrid approach combining a WavLM end-to-end segmentation method with a traditional multi-module clustering technique to adaptively select the appropriate model for handling varying degrees of overlapping speech. For the automatic speech recognition (ASR) system, we proposed an ASR-aware observation addition method that compensates for the performance limitations of Guided Source Separation (GSS) under low signal-to-noise ratio conditions. Finally, we integrated the speaker diarization and ASR systems in a cascaded architecture to address Track 3. Our system achieved character error rates (CER) of 9.48% on Track 2 and concatenated minimum permutation character error rate (cpCER) of 11.56% on Track 3, ultimately securing first place in both tracks and thereby demonstrating the effectiveness of the proposed methods in real-world meeting scenarios.
>
---
#### [new 014] VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **简介: 论文提出VoiceMark，属于零样本语音克隆抗水印任务。解决现有方法无法对抗零样本VC（无需训练直接生成克隆语音）的问题。通过利用说话人特定潜变量作为水印载体，并引入VC模拟增强和VAD损失，确保水印在克隆音频中保留。实验显示检测准确率超95%，显著优于现有方法的50%。**

- **链接: [http://arxiv.org/pdf/2505.21568v1](http://arxiv.org/pdf/2505.21568v1)**

> **作者:** Haiyun Li; Zhiyong Wu; Xiaofeng Xie; Jingran Xie; Yaoxun Xu; Hanyang Peng
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Voice cloning (VC)-resistant watermarking is an emerging technique for tracing and preventing unauthorized cloning. Existing methods effectively trace traditional VC models by training them on watermarked audio but fail in zero-shot VC scenarios, where models synthesize audio from an audio prompt without training. To address this, we propose VoiceMark, the first zero-shot VC-resistant watermarking method that leverages speaker-specific latents as the watermark carrier, allowing the watermark to transfer through the zero-shot VC process into the synthesized audio. Additionally, we introduce VC-simulated augmentations and VAD-based loss to enhance robustness against distortions. Experiments on multiple zero-shot VC models demonstrate that VoiceMark achieves over 95% accuracy in watermark detection after zero-shot VC synthesis, significantly outperforming existing methods, which only reach around 50%. See our code and demos at: https://huggingface.co/spaces/haiyunli/VoiceMark
>
---
#### [new 015] AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到音频（TTA）生成任务，旨在解决扩散模型推理速度慢的问题。现有rectified flow方法需从头训练且低步数效果差，作者提出AudioTurbo，利用预训练模型生成的确定性噪声样本对学习一阶ODE路径，实验证明其仅需10步即可超越此前模型，推理效率提升至3步。**

- **链接: [http://arxiv.org/pdf/2505.22106v1](http://arxiv.org/pdf/2505.22106v1)**

> **作者:** Junqi Zhao; Jinzheng Zhao; Haohe Liu; Yun Chen; Lu Han; Xubo Liu; Mark Plumbley; Wenwu Wang
>
> **摘要:** Diffusion models have significantly improved the quality and diversity of audio generation but are hindered by slow inference speed. Rectified flow enhances inference speed by learning straight-line ordinary differential equation (ODE) paths. However, this approach requires training a flow-matching model from scratch and tends to perform suboptimally, or even poorly, at low step counts. To address the limitations of rectified flow while leveraging the advantages of advanced pre-trained diffusion models, this study integrates pre-trained models with the rectified diffusion method to improve the efficiency of text-to-audio (TTA) generation. Specifically, we propose AudioTurbo, which learns first-order ODE paths from deterministic noise sample pairs generated by a pre-trained TTA model. Experiments on the AudioCaps dataset demonstrate that our model, with only 10 sampling steps, outperforms prior models and reduces inference to 3 steps compared to a flow-matching-based acceleration model.
>
---
#### [new 016] Advancing Hearing Assessment: An ASR-Based Frequency-Specific Speech Test for Diagnosing Presbycusis
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文开发基于ASR的频率特异性听力测试，解决传统测听无法准确评估老年性耳聋的阈上缺陷和频段感知问题。通过模拟听损的语音处理，分析音素混淆模式（如高频辅音替换/删除），验证测试可有效区分正常与受损听力，为精准听力评估提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.22231v1](http://arxiv.org/pdf/2505.22231v1)**

> **作者:** Stefan Bleeck
>
> **摘要:** Traditional audiometry often fails to fully characterize the functional impact of hearing loss on speech understanding, particularly supra-threshold deficits and frequency-specific perception challenges in conditions like presbycusis. This paper presents the development and simulated evaluation of a novel Automatic Speech Recognition (ASR)-based frequency-specific speech test designed to provide granular diagnostic insights. Our approach leverages ASR to simulate the perceptual effects of moderate sloping hearing loss by processing speech stimuli under controlled acoustic degradation and subsequently analyzing phoneme-level confusion patterns. Key findings indicate that simulated hearing loss introduces specific phoneme confusions, predominantly affecting high-frequency consonants (e.g., alveolar/palatal to labiodental substitutions) and leading to significant phoneme deletions, consistent with the acoustic cues degraded in presbycusis. A test battery curated from these ASR-derived confusions demonstrated diagnostic value, effectively differentiating between simulated normal-hearing and hearing-impaired listeners in a comprehensive simulation. This ASR-driven methodology offers a promising avenue for developing objective, granular, and frequency-specific hearing assessment tools that complement traditional audiometry. Future work will focus on validating these findings with human participants and exploring the integration of advanced AI models for enhanced diagnostic precision.
>
---
#### [new 017] Music Source Restoration
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出音乐源修复（MSR）任务，解决现有音乐源分离忽略制作中信号降级（如均衡、压缩、混响）的问题。构建首个包含578首歌曲未处理音源的RawStems数据集，提出U-Former基线模型，模拟多种降级并验证MSR可行性，公开数据集及工具。**

- **链接: [http://arxiv.org/pdf/2505.21827v1](http://arxiv.org/pdf/2505.21827v1)**

> **作者:** Yongyi Zang; Zheqi Dai; Mark D. Plumbley; Qiuqiang Kong
>
> **备注:** A modified version of this paper is in review
>
> **摘要:** We introduce Music Source Restoration (MSR), a novel task addressing the gap between idealized source separation and real-world music production. Current Music Source Separation (MSS) approaches assume mixtures are simple sums of sources, ignoring signal degradations employed during music production like equalization, compression, and reverb. MSR models mixtures as degraded sums of individually degraded sources, with the goal of recovering original, undegraded signals. Due to the lack of data for MSR, we present RawStems, a dataset annotation of 578 songs with unprocessed source signals organized into 8 primary and 17 secondary instrument groups, totaling 354.13 hours. To the best of our knowledge, RawStems is the first dataset that contains unprocessed music stems with hierarchical categories. We consider spectral filtering, dynamic range compression, harmonic distortion, reverb and lossy codec as possible degradations, and establish U-Former as a baseline method, demonstrating the feasibility of MSR on our dataset. We release the RawStems dataset annotations, degradation simulation pipeline, training code and pre-trained models to be publicly available.
>
---
#### [new 018] Leveraging LLM for Stuttering Speech: A Unified Architecture Bridging Recognition and Event Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出基于LLM的ASR-SED多任务框架，解决口吃语音识别与事件检测性能瓶颈。通过动态交互机制（ASR用CTC软提示优化LLM，SED输出嵌入增强理解），结合对比学习强化声学特征及Focal Loss处理类别不平衡，实现ASR误差率降低37.71%，SED F1提升46.58%。**

- **链接: [http://arxiv.org/pdf/2505.22005v1](http://arxiv.org/pdf/2505.22005v1)**

> **作者:** Shangkun Huang; Jing Deng; Jintao Kang; Rong Zheng
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The performance bottleneck of Automatic Speech Recognition (ASR) in stuttering speech scenarios has limited its applicability in domains such as speech rehabilitation. This paper proposed an LLM-driven ASR-SED multi-task learning framework that jointly optimized the ASR and Stuttering Event Detection (SED) tasks. We proposed a dynamic interaction mechanism where the ASR branch leveraged CTC-generated soft prompts to assist LLM context modeling, while the SED branch output stutter embeddings to enhance LLM comprehension of stuttered speech. We incorporated contrastive learning to strengthen the discriminative power of stuttering acoustic features and applied Focal Loss to mitigate the long-tailed distribution in stuttering event categories. Evaluations on the AS-70 Mandarin stuttering dataset demonstrated that our framework reduced the ASR character error rate (CER) to 5.45% (-37.71% relative reduction) and achieved an average SED F1-score of 73.63% (+46.58% relative improvement).
>
---
#### [new 019] Effective Context in Neural Speech Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究神经语音模型的有效上下文长度，解决模型实际使用上下文量量化问题。提出两种测量方法，分析监督与自监督模型（如HuBERT）的有效上下文差异，发现任务复杂度与所需上下文正相关，自监督模型早期层提升显著但整体仍较短，支持HuBERT无需修改即可流式处理。**

- **链接: [http://arxiv.org/pdf/2505.22487v1](http://arxiv.org/pdf/2505.22487v1)**

> **作者:** Yen Meng; Sharon Goldwater; Hao Tang
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Modern neural speech models benefit from having longer context, and many approaches have been proposed to increase the maximum context a model can use. However, few have attempted to measure how much context these models actually use, i.e., the effective context. Here, we propose two approaches to measuring the effective context, and use them to analyze different speech Transformers. For supervised models, we find that the effective context correlates well with the nature of the task, with fundamental frequency tracking, phone classification, and word classification requiring increasing amounts of effective context. For self-supervised models, we find that effective context increases mainly in the early layers, and remains relatively short -- similar to the supervised phone model. Given that these models do not use a long context during prediction, we show that HuBERT can be run in streaming mode without modification to the architecture and without further fine-tuning.
>
---
#### [new 020] An Investigation on Speaker Augmentation for End-to-End Speaker Extraction
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究端到端说话人提取任务，旨在解决目标混淆问题（系统误切换至非目标说话人）。提出基于时域重采样与重缩放的说话人增强策略，生成伪说话人提升嵌入泛化性，并通过特异性增强制造困难样本强化模型对真实说话特征的关注。实验显示该方法改善提取性能且可与度量学习结合提升效果。**

- **链接: [http://arxiv.org/pdf/2505.21805v1](http://arxiv.org/pdf/2505.21805v1)**

> **作者:** Zhenghai You; Zhenyu Zhou; Lantian Li; Dong Wang
>
> **摘要:** Target confusion, defined as occasional switching to non-target speakers, poses a key challenge for end-to-end speaker extraction (E2E-SE) systems. We argue that this problem is largely caused by the lack of generalizability and discrimination of the speaker embeddings, and introduce a simple yet effective speaker augmentation strategy to tackle the problem. Specifically, we propose a time-domain resampling and rescaling pipeline that alters speaker traits while preserving other speech properties. This generates a variety of pseudo-speakers to help establish a generalizable speaker embedding space, while the speaker-trait-specific augmentation creates hard samples that force the model to focus on genuine speaker characteristics. Experiments on WSJ0-2Mix and LibriMix show that our method mitigates the target confusion and improves extraction performance. Moreover, it can be combined with metric learning, another effective approach to address target confusion, leading to further gains.
>
---
#### [new 021] Voice Quality Dimensions as Interpretable Primitives for Speaking Style for Atypical Speech and Affect
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音质量建模任务，旨在通过可解释的声学维度分析非典型语音及情感的说话风格。研究基于SAP数据集训练七个语音质量探针模型（如可懂度、粗糙声等），验证其跨任务泛化能力，并实现零样本迁移至其他语言和情感语音数据，证明声学维度在说话风格分析中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.21809v1](http://arxiv.org/pdf/2505.21809v1)**

> **作者:** Jaya Narain; Vasudha Kowtha; Colin Lea; Lauren Tooley; Dianna Yee; Vikramjit Mitra; Zifang Huang; Miquel Espi Marques; Jon Huang; Carlos Avendano; Shirley Ren
>
> **备注:** accepted for Interspeech 2025
>
> **摘要:** Perceptual voice quality dimensions describe key characteristics of atypical speech and other speech modulations. Here we develop and evaluate voice quality models for seven voice and speech dimensions (intelligibility, imprecise consonants, harsh voice, naturalness, monoloudness, monopitch, and breathiness). Probes were trained on the public Speech Accessibility (SAP) project dataset with 11,184 samples from 434 speakers, using embeddings from frozen pre-trained models as features. We found that our probes had both strong performance and strong generalization across speech elicitation categories in the SAP dataset. We further validated zero-shot performance on additional datasets, encompassing unseen languages and tasks: Italian atypical speech, English atypical speech, and affective speech. The strong zero-shot performance and the interpretability of results across an array of evaluations suggests the utility of using voice quality dimensions in speaking style-related tasks.
>
---
#### [new 022] VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于越南语自动语音识别（ASR）任务，旨在解决低资源语言数据稀缺及现有系统成本高、延迟大的问题。提出VietASR方法，通过7万小时无标注数据预训练与50小时标注数据微调，构建轻量高效的模型，性能超越Whisper等现有系统。**

- **链接: [http://arxiv.org/pdf/2505.21527v1](http://arxiv.org/pdf/2505.21527v1)**

> **作者:** Jianheng Zhuo; Yifan Yang; Yiwen Shao; Yong Xu; Dong Yu; Kai Yu; Xie Chen
>
> **摘要:** Automatic speech recognition (ASR) has made remarkable progress but heavily relies on large-scale labeled data, which is scarce for low-resource languages like Vietnamese. While existing systems such as Whisper, USM, and MMS achieve promising performance, their efficacy remains inadequate in terms of training costs, latency, and accessibility. To address these issues, we propose VietASR, a novel ASR training pipeline that leverages vast amounts of unlabeled data and a small set of labeled data. Through multi-iteration ASR-biased self-supervised learning on a large-scale unlabeled dataset, VietASR offers a cost-effective and practical solution for enhancing ASR performance. Experiments demonstrate that pre-training on 70,000-hour unlabeled data and fine-tuning on merely 50-hour labeled data yield a lightweight but powerful ASR model. It outperforms Whisper Large-v3 and commercial ASR systems on real-world data. Our code and models will be open-sourced to facilitate research in low-resource ASR.
>
---
#### [new 023] Mitigating Audiovisual Mismatch in Visual-Guide Audio Captioning
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视觉引导音频描述任务，旨在解决视听不匹配问题（如配音或画外音）。提出熵感知门控融合框架，通过注意力熵分析抑制误导视觉信息，并开发批量视听混排技术增强模型鲁棒性，在AudioCaps基准上超越基线，速度提升6倍。**

- **链接: [http://arxiv.org/pdf/2505.22045v1](http://arxiv.org/pdf/2505.22045v1)**

> **作者:** Le Xu; Chenxing Li; Yong Ren; Yujie Chen; Yu Gu; Ruibo Fu; Shan Yang; Dong Yu
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Current vision-guided audio captioning systems frequently fail to address audiovisual misalignment in real-world scenarios, such as dubbed content or off-screen sounds. To bridge this critical gap, we present an entropy-aware gated fusion framework that dynamically modulates visual information flow through cross-modal uncertainty quantification. Our novel approach employs attention entropy analysis in cross-attention layers to automatically identify and suppress misleading visual cues during modal fusion. Complementing this architecture, we develop a batch-wise audiovisual shuffling technique that generates synthetic mismatched training pairs, greatly enhancing model resilience against alignment noise. Evaluations on the AudioCaps benchmark demonstrate our system's superior performance over existing baselines, especially in mismatched modality scenarios. Furthermore, our solution demonstrates an approximately 6x improvement in inference speed compared to the baseline.
>
---
#### [new 024] Loquacious Set: 25,000 Hours of Transcribed and Diverse English Speech Recognition Data for Research and Commercial Use
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文构建ASR数据集，解决现有数据集规模小、场景单一及许可限制问题。提出Loquacious Set：25000小时商业可用英语语音，涵盖多口音、朗读/自发/嘈杂等场景，助力学术与工业界研发真实环境ASR系统。（98字）**

- **链接: [http://arxiv.org/pdf/2505.21578v1](http://arxiv.org/pdf/2505.21578v1)**

> **作者:** Titouan Parcollet; Yuan Tseng; Shucong Zhang; Rogier van Dalen
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) research is driven by the availability of common datasets between industrial researchers and academics, encouraging comparisons and evaluations. LibriSpeech, despite its long success as an ASR benchmark, is now limited by its size and focus on clean, read speech, leading to near-zero word error rates. More recent datasets, including MOSEL, YODAS, Gigaspeech, OWSM, Libriheavy or People's Speech suffer from major limitations including licenses that researchers in the industry cannot use, unreliable transcriptions, incorrect audio data, or the lack of evaluation sets. This work presents the Loquacious Set, a 25,000-hour curated collection of commercially usable English speech. Featuring hundreds of thousands of speakers with diverse accents and a wide range of speech types (read, spontaneous, talks, clean, noisy), the Loquacious Set is designed to work for academics and researchers in the industry to build ASR systems in real-world scenarios.
>
---
#### [new 025] WhisperD: Dementia Speech Recognition and Filler Word Detection with Whisper
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于老年痴呆症语音识别与填充词检测任务，旨在解决Whisper模型因缺乏痴呆患者不连贯语音训练导致的转录错误问题。研究通过微调Whisper模型（使用DementiaBank和自建数据集），优化词错误率（WER）及填充词检测指标（FIR、F1），使中型模型WER降至0.24，显著优于原有模型，提升诊断与辅助技术的可行性。**

- **链接: [http://arxiv.org/pdf/2505.21551v1](http://arxiv.org/pdf/2505.21551v1)**

> **作者:** Emmanuel Akinrintoyo; Nadine Abdelhalim; Nicole Salomons
>
> **备注:** Submitted to Interspeech 2025 (Accepted)
>
> **摘要:** Whisper fails to correctly transcribe dementia speech because persons with dementia (PwDs) often exhibit irregular speech patterns and disfluencies such as pauses, repetitions, and fragmented sentences. It was trained on standard speech and may have had little or no exposure to dementia-affected speech. However, correct transcription is vital for dementia speech for cost-effective diagnosis and the development of assistive technology. In this work, we fine-tune Whisper with the open-source dementia speech dataset (DementiaBank) and our in-house dataset to improve its word error rate (WER). The fine-tuning also includes filler words to ascertain the filler inclusion rate (FIR) and F1 score. The fine-tuned models significantly outperformed the off-the-shelf models. The medium-sized model achieved a WER of 0.24, outperforming previous work. Similarly, there was a notable generalisability to unseen data and speech patterns.
>
---
#### [new 026] MultiFormer: A Multi-Person Pose Estimation System Based on CSI and Attention Mechanism
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于基于CSI的多人姿态估计任务，旨在解决多目标识别不准确及CSI特征学习效果差的问题。提出MultiFormer系统，采用Transformer的时频双token特征提取器建模CSI的子载波相关性和时序依赖，并通过多阶段特征融合网络（MSFN）强化解剖约束，实验显示其在高移动性关键点（手腕、肘部）的精度优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.22555v1](http://arxiv.org/pdf/2505.22555v1)**

> **作者:** Yanyi Qu; Haoyang Ma; Wenhui Xiong
>
> **摘要:** Human pose estimation based on Channel State Information (CSI) has emerged as a promising approach for non-intrusive and precise human activity monitoring, yet faces challenges including accurate multi-person pose recognition and effective CSI feature learning. This paper presents MultiFormer, a wireless sensing system that accurately estimates human pose through CSI. The proposed system adopts a Transformer based time-frequency dual-token feature extractor with multi-head self-attention. This feature extractor is able to model inter-subcarrier correlations and temporal dependencies of the CSI. The extracted CSI features and the pose probability heatmaps are then fused by Multi-Stage Feature Fusion Network (MSFN) to enforce the anatomical constraints. Extensive experiments conducted on on the public MM-Fi dataset and our self-collected dataset show that the MultiFormer achieves higher accuracy over state-of-the-art approaches, especially for high-mobility keypoints (wrists, elbows) that are particularly difficult for previous methods to accurately estimate.
>
---
#### [new 027] Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音口吃检测任务，旨在解决标注数据稀缺及合成数据自然度与多样性不足的问题。提出LLM-Dys数据集，利用LLM生成涵盖11类口吃的合成数据，并改进端到端检测框架，实验达最优性能，资源已开源。**

- **链接: [http://arxiv.org/pdf/2505.22029v1](http://arxiv.org/pdf/2505.22029v1)**

> **作者:** Jinming Zhang; Xuanru Zhou; Jiachen Lian; Shuhe Li; William Li; Zoe Ezzes; Rian Bogley; Lisa Wauters; Zachary Miller; Jet Vonk; Brittany Morin; Maria Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** Speech dysfluency detection is crucial for clinical diagnosis and language assessment, but existing methods are limited by the scarcity of high-quality annotated data. Although recent advances in TTS model have enabled synthetic dysfluency generation, existing synthetic datasets suffer from unnatural prosody and limited contextual diversity. To address these limitations, we propose LLM-Dys -- the most comprehensive dysfluent speech corpus with LLM-enhanced dysfluency simulation. This dataset captures 11 dysfluency categories spanning both word and phoneme levels. Building upon this resource, we improve an end-to-end dysfluency detection framework. Experimental validation demonstrates state-of-the-art performance. All data, models, and code are open-sourced at https://github.com/Berkeley-Speech-Group/LLM-Dys.
>
---
#### [new 028] Voice Adaptation for Swiss German
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属语音适应任务，旨在将标准德语文本转化为瑞士德语方言语音，解决方言语音克隆数据不足的问题。研究者预处理5000小时播客单词并自动标注方言类别，微调XTTSv2模型，取得CMOS-0.28、SMOS3.8的评估结果，推动小语种语音技术发展。**

- **链接: [http://arxiv.org/pdf/2505.22054v1](http://arxiv.org/pdf/2505.22054v1)**

> **作者:** Samuel Stucki; Jan Deriu; Mark Cieliebak
>
> **备注:** Submitted to Interspeech
>
> **摘要:** This work investigates the performance of Voice Adaptation models for Swiss German dialects, i.e., translating Standard German text to Swiss German dialect speech. For this, we preprocess a large dataset of Swiss podcasts, which we automatically transcribe and annotate with dialect classes, yielding approximately 5000 hours of weakly labeled training material. We fine-tune the XTTSv2 model on this dataset and show that it achieves good scores in human and automated evaluations and can correctly render the desired dialect. Our work shows a step towards adapting Voice Cloning technology to underrepresented languages. The resulting model achieves CMOS scores of up to -0.28 and SMOS scores of 3.8.
>
---
#### [new 029] Articulatory modeling of the S-shaped F2 trajectories observed in Öhman's spectrographic analysis of VCV syllables
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于发音建模任务，旨在解释Ohman研究中观察到的VCV音节S形F2轨迹成因。通过Maeda模型替代传统DRM模型，分析75个VCV序列，区分元音过渡与辅音影响，结合轨迹规划和共振峰轨迹方程分析，揭示S形轨迹由发音器官协同作用形成，而非单纯元音或辅音规划。**

- **链接: [http://arxiv.org/pdf/2505.22455v1](http://arxiv.org/pdf/2505.22455v1)**

> **作者:** Frédéric Berthommier
>
> **备注:** 5 pages, 4 figures, submitted to Interspeech 2025
>
> **摘要:** The synthesis of Ohman's VCV sequences with intervocalic plosive consonants was first achieved 30 years ago using the DRM model. However, this approach remains primarily acoustic and lacks articulatory constraints. In this study, the same 75 VCVs are analyzed, but generated with the Maeda model, using trajectory planning that differentiates vowel-to-vowel transitions from consonantal influences. Synthetic data exhibit similar characteristics to Ohman's sequences, including the presence of S-shaped F2 trajectories. Furthermore, locus equations (LEs) for F2 and F3 are computed from synthetic CV data to investigate their underlying determinism, leading to a reassessment of conventional interpretations. The findings indicate that, although articulatory planning is structured separately for vowel and consonant groups, S-shaped F2 trajectories emerge from a composite mechanism governed by the coordinated synergy of all articulators.
>
---
## 更新

#### [replaced 001] A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19663v2](http://arxiv.org/pdf/2505.19663v2)**

> **作者:** Yigitcan Özer; Woosung Choi; Joan Serrà; Mayank Kumar Singh; Wei-Hsiang Liao; Yuki Mitsufuji
>
> **备注:** 5 pages; 5 tables; accepted at INTERSPEECH 2025
>
> **摘要:** We introduce the Robust Audio Watermarking Benchmark (RAW-Bench), a benchmark for evaluating deep learning-based audio watermarking methods with standardized and systematic comparisons. To simulate real-world usage, we introduce a comprehensive audio attack pipeline with various distortions such as compression, background noise, and reverberation, along with a diverse test dataset including speech, environmental sounds, and music recordings. Evaluating four existing watermarking methods on RAW-bench reveals two main insights: (i) neural compression techniques pose the most significant challenge, even when algorithms are trained with such compressions; and (ii) training with audio attacks generally improves robustness, although it is insufficient in some cases. Furthermore, we find that specific distortions, such as polarity inversion, time stretching, or reverb, seriously affect certain methods. The evaluation framework is accessible at github.com/SonyResearch/raw_bench.
>
---
#### [replaced 002] VQ-CTAP: Cross-Modal Fine-Grained Sequence Representation Learning for Speech Processing
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2408.05758v2](http://arxiv.org/pdf/2408.05758v2)**

> **作者:** Chunyu Qiang; Wang Geng; Yi Zhao; Ruibo Fu; Tao Wang; Cheng Gong; Tianrui Wang; Qiuyu Liu; Jiangyan Yi; Zhengqi Wen; Chen Zhang; Hao Che; Longbiao Wang; Jianwu Dang; Jianhua Tao
>
> **摘要:** Deep learning has brought significant improvements to the field of cross-modal representation learning. For tasks such as text-to-speech (TTS), voice conversion (VC), and automatic speech recognition (ASR), a cross-modal fine-grained (frame-level) sequence representation is desired, emphasizing the semantic content of the text modality while de-emphasizing the paralinguistic information of the speech modality. We propose a method called "Vector Quantized Contrastive Token-Acoustic Pre-training (VQ-CTAP)", which uses the cross-modal aligned sequence transcoder to bring text and speech into a joint multimodal space, learning how to connect text and speech at the frame level. The proposed VQ-CTAP is a paradigm for cross-modal sequence representation learning, offering a promising solution for fine-grained generation and recognition tasks in speech processing. The VQ-CTAP can be directly applied to VC and ASR tasks without fine-tuning or additional structures. We propose a sequence-aware semantic connector, which connects multiple frozen pre-trained modules for the TTS task, exhibiting a plug-and-play capability. We design a stepping optimization strategy to ensure effective model convergence by gradually injecting and adjusting the influence of various loss components. Furthermore, we propose a semantic-transfer-wise paralinguistic consistency loss to enhance representational capabilities, allowing the model to better generalize to unseen data and capture the nuances of paralinguistic information. In addition, VQ-CTAP achieves high-compression speech coding at a rate of 25Hz from 24kHz input waveforms, which is a 960-fold reduction in the sampling rate. The audio demo is available at https://qiangchunyu.github.io/VQCTAP/
>
---
#### [replaced 003] Zero-Shot Mono-to-Binaural Speech Synthesis
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.08356v2](http://arxiv.org/pdf/2412.08356v2)**

> **作者:** Alon Levkovitch; Julian Salazar; Soroosh Mariooryad; RJ Skerry-Ryan; Nadav Bar; Bastiaan Kleijn; Eliya Nachmani
>
> **摘要:** We present ZeroBAS, a neural method to synthesize binaural audio from monaural audio recordings and positional information without training on any binaural data. To our knowledge, this is the first published zero-shot neural approach to mono-to-binaural audio synthesis. Specifically, we show that a parameter-free geometric time warping and amplitude scaling based on source location suffices to get an initial binaural synthesis that can be refined by iteratively applying a pretrained denoising vocoder. Furthermore, we find this leads to generalization across room conditions, which we measure by introducing a new dataset, TUT Mono-to-Binaural, to evaluate state-of-the-art monaural-to-binaural synthesis methods on unseen conditions. Our zero-shot method is perceptually on-par with the performance of supervised methods on the standard mono-to-binaural dataset, and even surpasses them on our out-of-distribution TUT Mono-to-Binaural dataset. Our results highlight the potential of pretrained generative audio models and zero-shot learning to unlock robust binaural audio synthesis.
>
---
#### [replaced 004] On the Within-class Variation Issue in Alzheimer's Disease Detection
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2409.16322v2](http://arxiv.org/pdf/2409.16322v2)**

> **作者:** Jiawen Kang; Dongrui Han; Lingwei Meng; Jingyan Zhou; Jinchao Li; Xixin Wu; Helen Meng
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** Alzheimer's Disease (AD) detection employs machine learning classification models to distinguish between individuals with AD and those without. Different from conventional classification tasks, we identify within-class variation as a critical challenge in AD detection: individuals with AD exhibit a spectrum of cognitive impairments. Therefore, simplistic binary AD classification may overlook two crucial aspects: within-class heterogeneity and instance-level imbalance. In this work, we found using a sample score estimator can generate sample-specific soft scores aligning with cognitive scores. We subsequently propose two simple yet effective methods: Soft Target Distillation (SoTD) and Instance-level Re-balancing (InRe), targeting two problems respectively. Based on the ADReSS and CU-MARVEL corpora, we demonstrated and analyzed the advantages of the proposed approaches in detection performance. These findings provide insights for developing robust and reliable AD detection models.
>
---
#### [replaced 005] Optimal Scalogram for Computational Complexity Reduction in Acoustic Recognition Using Deep Learning
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.13017v2](http://arxiv.org/pdf/2505.13017v2)**

> **作者:** Dang Thoai Phan; Tuan Anh Huynh; Van Tuan Pham; Cao Minh Tran; Van Thuan Mai; Ngoc Quy Tran
>
> **摘要:** The Continuous Wavelet Transform (CWT) is an effective tool for feature extraction in acoustic recognition using Convolutional Neural Networks (CNNs), particularly when applied to non-stationary audio. However, its high computational cost poses a significant challenge, often leading researchers to prefer alternative methods such as the Short-Time Fourier Transform (STFT). To address this issue, this paper proposes a method to reduce the computational complexity of CWT by optimizing the length of the wavelet kernel and the hop size of the output scalogram. Experimental results demonstrate that the proposed approach significantly reduces computational cost while maintaining the robust performance of the trained model in acoustic recognition tasks.
>
---
#### [replaced 006] GOAT-TTS: Expressive and Realistic Speech Generation via A Dual-Branch LLM
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.12339v2](http://arxiv.org/pdf/2504.12339v2)**

> **作者:** Yaodong Song; Hongjie Chen; Jie Lian; Yuxin Zhang; Guangmin Xia; Zehan Li; Genliang Zhao; Jian Kang; Jie Li; Yongxiang Li; Xuelong Li
>
> **摘要:** While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-n layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
>
---
#### [replaced 007] Solid State Bus-Comp: A Large-Scale and Diverse Dataset for Dynamic Range Compressor Virtual Analog Modeling
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2504.04589v3](http://arxiv.org/pdf/2504.04589v3)**

> **作者:** Yicheng Gu; Runsong Zhang; Lauri Juvela; Zhizheng Wu
>
> **摘要:** Virtual Analog (VA) modeling aims to simulate the behavior of hardware circuits via algorithms to replicate their tone digitally. Dynamic Range Compressor (DRC) is an audio processing module that controls the dynamics of a track by reducing and amplifying the volumes of loud and quiet sounds, which is essential in music production. In recent years, neural-network-based VA modeling has shown great potential in producing high-fidelity models. However, due to the lack of data quantity and diversity, their generalization ability in different parameter settings and input sounds is still limited. To tackle this problem, we present Solid State Bus-Comp, the first large-scale and diverse dataset for modeling the classical VCA compressor -- SSL 500 G-Bus. Specifically, we manually collected 175 unmastered songs from the Cambridge Multitrack Library. We recorded the compressed audio in 220 parameter combinations, resulting in an extensive 2528-hour dataset with diverse genres, instruments, tempos, and keys. Moreover, to facilitate the use of our proposed dataset, we conducted benchmark experiments in various open-sourced black-box and grey-box models, as well as white-box plugins. We also conducted ablation studies in different data subsets to illustrate the effectiveness of the improved data diversity and quantity. The dataset and demos are on our project page: https://www.yichenggu.com/SolidStateBusComp/.
>
---
#### [replaced 008] Data Standards in Audiology: A Mixed-Methods Exploration of Community Perspectives and Implementation Considerations
- **分类: cs.SD; eess.AS; physics.med-ph**

- **链接: [http://arxiv.org/pdf/2505.04728v2](http://arxiv.org/pdf/2505.04728v2)**

> **作者:** Charlotte Vercammen; Antje Heinrich; Christophe Lesimple; Alessia Paglialonga; Jan-Willem A. Wasmann; Mareike Buhl
>
> **摘要:** Objective: The purpose of this study was to explore options for data standardisation in audiology and document the global audiology community's current knowledge and views of data standards, explore their needs and preferences, and develop recommendations for data standardisation as a result. Design: A mixed-methods approach, combining a structured survey with an in-depth exploration of themes by experts during a special session on "Big Data and Data Standards in Audiology" at the 2024 Virtual Conference of Computational Audiology. Study Sample: The survey sample consisted of 82 members of the global audiology community; five experts joined the panel discussion. Results: Survey results emphasized the need for data standardisation in audiology aimed at facilitating research and improving patient care. Knowledge of existing initiatives was low: 38% were aware of initiatives. Yet, 90% envisioned contributing to them moving forward. The panel discussion explored emerging standardisation initiatives in audiology (OMOP, openEHR, HIMSA's Noah standard), challenges (e.g., data quality and privacy), and opportunities (e.g., conversion between approaches and synergies with other medical fields). Conclusions: The community support identified in this study could be leveraged to further develop standardisation initiatives for audiology, ensuring alignment between initiatives and with other medical fields.
>
---
#### [replaced 009] METEOR: Melody-aware Texture-controllable Symbolic Orchestral Music Generation
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.11753v2](http://arxiv.org/pdf/2409.11753v2)**

> **作者:** Dinh-Viet-Toan Le; Yi-Hsuan Yang
>
> **备注:** Accepted to 34rd International Joint Conference on Artificial Intelligence (IJCAI 2025) - AI, Arts and Creativity Special Track. Demo: https://dinhviettoanle.github.io/meteor
>
> **摘要:** Western music is often characterized by a homophonic texture, in which the musical content can be organized into a melody and an accompaniment. In orchestral music, in particular, the composer can select specific characteristics for each instrument's part within the accompaniment, while also needing to adapt the melody to suit the capabilities of the instruments performing it. In this work, we propose METEOR, a model for Melody-aware Texture-controllable Orchestral music generation. This model performs symbolic multi-track music style transfer with a focus on melodic fidelity. We allow bar- and track-level controllability of the accompaniment with various textural attributes while keeping a homophonic texture. We show that the model can achieve controllability performances similar to strong baselines while greatly improve melodic fidelity.
>
---
#### [replaced 010] Neurodyne: Neural Pitch Manipulation with Representation Learning and Cycle-Consistency GAN
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15368v3](http://arxiv.org/pdf/2505.15368v3)**

> **作者:** Yicheng Gu; Chaoren Wang; Zhizheng Wu; Lauri Juvela
>
> **摘要:** Pitch manipulation is the process of producers adjusting the pitch of an audio segment to a specific key and intonation, which is essential in music production. Neural-network-based pitch-manipulation systems have been popular in recent years due to their superior synthesis quality compared to classical DSP methods. However, their performance is still limited due to their inaccurate feature disentanglement using source-filter models and the lack of paired in- and out-of-tune training data. This work proposes Neurodyne to address these issues. Specifically, Neurodyne uses adversarial representation learning to learn a pitch-independent latent representation to avoid inaccurate disentanglement and cycle-consistency training to create paired training data implicitly. Experimental results on global-key and template-based pitch manipulation demonstrate the effectiveness of the proposed system, marking improved synthesis quality while maintaining the original singer identity.
>
---
#### [replaced 011] Hearing from Silence: Reasoning Audio Descriptions from Silent Videos via Vision-Language Model
- **分类: cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13062v3](http://arxiv.org/pdf/2505.13062v3)**

> **作者:** Yong Ren; Chenxing Li; Le Xu; Hao Gu; Duzhen Zhang; Yujie Chen; Manjie Xu; Ruibo Fu; Shan Yang; Dong Yu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Humans can intuitively infer sounds from silent videos, but whether multimodal large language models can perform modal-mismatch reasoning without accessing target modalities remains relatively unexplored. Current text-assisted-video-to-audio (VT2A) methods excel in video foley tasks but struggle to acquire audio descriptions during inference. We introduce the task of Reasoning Audio Descriptions from Silent Videos (SVAD) to address this challenge and investigate vision-language models' (VLMs) capabilities on this task. To further enhance the VLMs' reasoning capacity for the SVAD task, we construct a CoT-AudioCaps dataset and propose a Chain-of-Thought-based supervised fine-tuning strategy. Experiments on SVAD and subsequent VT2A tasks demonstrate our method's effectiveness in two key aspects: significantly improving VLMs' modal-mismatch reasoning for SVAD and effectively addressing the challenge of acquiring audio descriptions during VT2A inference.
>
---
#### [replaced 012] Towards Robust Automated Perceptual Voice Quality Assessment with Speech Foundation Models
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.21356v2](http://arxiv.org/pdf/2505.21356v2)**

> **作者:** Whenty Ariyanti; Kuan-Yu Chen; Sabato Marco Siniscalchi; Hsin-Min Wang; Yu Tsao
>
> **摘要:** Perceptual voice quality assessment is essential for diagnosing and monitoring voice disorders. Traditionally, expert raters use scales such as the CAPE-V and GRBAS. However, these are subjective and prone to inter-rater variability, motivating the need for automated, objective assessment methods. This study proposes VOQANet, a deep learning framework with an attention mechanism that leverages a Speech Foundation Model (SFM) to extract high-level acoustic and prosodic information from raw speech. To improve robustness and interpretability, we introduce VOQANet+, which integrates handcrafted acoustic features such as jitter, shimmer, and harmonics-to-noise ratio (HNR) with SFM embeddings into a hybrid representation. Unlike prior work focusing only on vowel-based phonation (PVQD-A subset) from the Perceptual Voice Quality Dataset (PVQD), we evaluate our models on both vowel-based and sentence-level speech (PVQD-S subset) for better generalizability. Results show that sentence-based input outperforms vowel-based input, particularly at the patient level, highlighting the benefit of longer utterances for capturing voice attributes. VOQANet consistently surpasses baseline methods in root mean squared error and Pearson correlation across CAPE-V and GRBAS dimensions, with VOQANet+ achieving further improvements. Additional tests under noisy conditions show that VOQANet+ maintains high prediction accuracy, supporting its use in real-world and telehealth settings. These findings demonstrate the value of combining SFM embeddings with domain-informed acoustic features for interpretable and robust voice quality assessment.
>
---
