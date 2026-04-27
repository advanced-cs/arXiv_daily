# 音频 cs.SD;  eess.AS

- **最新发布 9 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出UniSonate，解决语音、音乐和音效生成任务的统一问题，通过动态令牌注入和多阶段学习，实现跨模态生成。**

- **链接: [https://arxiv.org/pdf/2604.22209](https://arxiv.org/pdf/2604.22209)**

> **作者:** Chunyu Qiang; Xiaopeng Wang; Kang Yin; Yuzhe Liang; Yuxin Guo; Teng Ma; Ziyu Zhang; Tianrui Wang; Cheng Gong; Yushen Chen; Ruibo Fu; Chen Zhang; Longbiao Wang; Jianwu Dang
>
> **备注:** Accepted to ACL 2026 main conference (oral)
>
> **摘要:** Generative audio modeling has largely been fragmented into specialized tasks, text-to-speech (TTS), text-to-music (TTM), and text-to-audio (TTA), each operating under heterogeneous control paradigms. Unifying these modalities remains a fundamental challenge due to the intrinsic dissonance between structured semantic representations (speech/music) and unstructured acoustic textures (sound effects). In this paper, we introduce UniSonate, a unified flow-matching framework capable of synthesizing speech, music, and sound effects through a standardized, reference-free natural language instruction interface. To reconcile structural disparities, we propose a novel dynamic token injection mechanism that projects unstructured environmental sounds into a structured temporal latent space, enabling precise duration control within a phoneme-driven Multimodal Diffusion Transformer (MM-DiT). Coupled with a multi-stage curriculum learning strategy, this approach effectively mitigates cross-modal optimization conflicts. Extensive experiments demonstrate that UniSonate achieves state-of-the-art performance in instruction-based TTS (WER 1.47%) and TTM (SongEval Coherence 3.18), while maintaining competitive fidelity in TTA. Crucially, we observe positive transfer, where joint training on diverse audio data significantly enhances structural coherence and prosodic expressiveness compared to single-task baselines. Audio samples are available at this https URL.
>
---
#### [new 002] Transformer-Based Rhythm Quantization of Performance MIDI Using Beat Annotations
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音乐转录任务，解决MIDI性能的节奏量化问题。通过引入基于Transformer的模型，利用节拍信息提升量化效果，优化数据表示与模型结构，实现高精度的节奏对齐与输出。**

- **链接: [https://arxiv.org/pdf/2604.22290](https://arxiv.org/pdf/2604.22290)**

> **作者:** Maximilian Wachter; Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the 5th International Conference on SMART MULTIMEDIA (ICSM), 2025
>
> **摘要:** Rhythm transcription is a key subtask of notation-level Automatic Music Transcription (AMT). While deep learning models have been extensively used for detecting the metrical grid in audio and MIDI performances, beat-based rhythm quantization remains largely unexplored. In this work, we introduce a novel deep learning approach for quantizing MIDI performances using a priori beat information. Our method leverages the transformer architecture to effectively process synchronized score and performance data for training a quantization model. Key components of our approach include dataset preparation, a beat-based pre-quantization method to align performance and score times within a unified framework, and a MIDI tokenizer tailored for this task. We adapt a transformer model based on the T5 architecture to meet the specific requirements of rhythm quantization. The model is evaluated using a set of score-level metrics designed for objective assessment of quantization performance. Through systematic evaluation, we optimize both data representation and model architecture. Additionally, we apply performance and score augmentations, such as transposition, note deletion, and performance-side time jitter, to enhance the model's robustness. Finally, a qualitative analysis compares our model's quantization performance against state-of-the-art probabilistic and deep-learning models on various example pieces. Our model achieves an onset F1-score of 97.3% and a note value accuracy of 83.3% on the ASAP dataset. It generalizes well across time signatures, including those not seen during training, and produces readable score output. Fine-tuning on instrument-specific datasets further improves performance by capturing characteristic rhythmic and melodic patterns. This work contributes a robust and flexible framework for beat-based MIDI quantization using transformer models.
>
---
#### [new 003] Beyond Acoustic Sparsity and Linguistic Bias: A Prompt-Free Paradigm for Mispronunciation Detection and Diagnosis
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别中的发音错误检测与诊断任务，旨在解决现有系统因声学建模不足和语言偏见导致的误差问题。提出CROTTC-IF框架，提升检测精度。**

- **链接: [https://arxiv.org/pdf/2604.22133](https://arxiv.org/pdf/2604.22133)**

> **作者:** Haopeng Geng; Longfei Yang; Xi Chen; Haitong Sun; Daisuke Saito; Nobuaki Minematsu
>
> **摘要:** Mispronunciation Detection and Diagnosis (MDD) requires modeling fine-grained acoustic deviations. However, current ASR-derived MDD systems often face inherent limitations. In particular, CTC-based models favor sequence-level alignments that neglect transient mispronunciation cues, while explicit canonical priors bias predictions toward intended targets. To address these bottlenecks, we propose a prompt-free framework decoupling acoustic fidelity from canonical guidance. First, we introduce CROTTC, an acoustic model enforcing monotonic, frame-level alignment to accurately capture pronunciation deviations. Second, we implicitly inject mispronunciation information via the IF strategy under the knowledge transfer principle. Experiments show CROTTC-IF achieves a 71.77% F1-score on L2-ARCTIC and 71.70% F1-score on the Iqra'Eval2 leaderboard. With empirical analysis, we demonstrate that decoupling acoustics from explicit priors provides highly robust MDD.
>
---
#### [new 004] Advancing automatic speech recognition using feature fusion with self-supervised learning features: A case study on Fearless Steps Apollo corpus
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在提升自然场景下的识别性能。针对传统特征不足，融合自监督学习特征，并提出新的交叉注意力融合方法，有效降低词错误率。**

- **链接: [https://arxiv.org/pdf/2604.22203](https://arxiv.org/pdf/2604.22203)**

> **作者:** Szu-Jui Chen; John H.L. Hansen
>
> **备注:** Accepted to Speech Communication 2026
>
> **摘要:** Using self-supervised learning (SSL) models has significantly improved performance for downstream speech tasks, surpassing the capabilities of traditional hand-crafted features. This study investigates the amalgamation of SSL models, with the aim to leverage both their individual strengths and refine extracted features to achieve improved speech recognition models for naturalistic scenarios. Our research investigates the massive naturalistic Fearless Steps (FS) APOLLO resource, with particular focus on the FS Challenge (FSC) Phase-4 corpus, providing the inaugural analysis of this dataset. Additionally, we incorporate the CHiME-6 dataset to evaluate performance across diverse naturalistic speech scenarios. While exploring previously proposed Feature Refinement Loss and fusion methods, we found these methods to be less effective on the FSC Phase-4 corpus. To address this, we introduce a novel deep cross-attention (DCA) fusion method, designed to elevate performance, especially for the FSC Phase-4 corpus. Our objective is to foster creation of superior FS APOLLO community resources, catering to the diverse needs of researchers across various disciplines. The proposed solution achieves an absolute +1.1% improvement in WER, providing effective meta-data creation for the massive FS APOLLO community resource.
>
---
#### [new 005] DM-ASR: Diarization-aware Multi-speaker ASR with Large Language Models
- **分类: eess.AS**

- **简介: 该论文属于多说话人自动语音识别任务，旨在解决同时识别谁说、说什么及何时说的问题。提出DM-ASR框架，结合说话人分割与大语言模型，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2604.22467](https://arxiv.org/pdf/2604.22467)**

> **作者:** Li Li; Ming Cheng; Weixin Zhu; Yannan Wang; Juan Liu; Ming Li
>
> **摘要:** Multi-speaker automatic speech recognition (ASR) aims to transcribe conversational speech involving multiple speakers, requiring the model to capture not only what was said, but also who said it and sometimes when it was spoken. Recent Speech-LLM approaches have shown the potential of unified modeling for this task, but jointly learning speaker attribution, temporal structure, and lexical recognition remains difficult and data-intensive. At the current stage, leveraging reliable speaker diarization as an explicit structural prior provides a practical and efficient way to simplify this task. To effectively exploit such priors, we propose DM-ASR, a diarization-aware multi-speaker ASR framework that reformulates the task as a multi-turn dialogue generation process. Given an audio chunk and diarization results, DM-ASR decomposes transcription into a sequence of speaker- and time-conditioned queries, each corresponding to one speaker in one time segment. This formulation converts multi-speaker recognition into a series of structured sub-tasks, explicitly decoupling speaker-temporal structure from linguistic content and enabling effective integration of diarization cues with the reasoning capability of large language models. We further introduce an optional word-level timestamp prediction mechanism that interleaves word and timestamp tokens, yielding richer structured outputs and better transcription quality. Our analysis shows that diarization systems provide more reliable speaker identities and segment-level boundaries, while LLMs excel at modeling linguistic content and long-range dependencies, demonstrating their complementary strengths. Experiments on Mandarin and English benchmarks show that the proposed approach achieves strong performance with relatively small models and training data, while remaining competitive with or outperforming existing unified approaches.
>
---
#### [new 006] Spectrographic Portamento Gradient Analysis: A Quantitative Method for Historical Cello Recordings with Application to Beethoven's Piano and Cello Sonatas, 1930--2012
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐分析任务，旨在量化弦乐滑音的谱图梯度，解决滑音表达特征的定量描述问题，通过新方法分析历史录音中的滑音表现。**

- **链接: [https://arxiv.org/pdf/2604.22037](https://arxiv.org/pdf/2604.22037)**

> **作者:** Ignasi Sole
>
> **摘要:** Portamento in string performance has been studied primarily as a binary presence-or-absence phenomenon, with existing research measuring frequency of occurrence and, less commonly, duration in milliseconds. This paper introduces a third quantitative descriptor; the spectrographic gradient of the portamento slide, measured in Hz/second, and demonstrates its measurement using a protocol combining Sonic Visualizer's melodic spectrogram layer, GIMP pixel analysis, and metric calibration against the spectrogram's known frequency axis. The gradient captures what duration alone cannot: the steepness of the pitch trajectory, which encodes the expressive character of the slide independently of its length. Applied to the opening measures of. Specifically because their monophonic texture permits reliable spectrographic pitch tracking. The method yields gradient values ranging from approximately 600~Hz/s in late-period recordings to over 4,000~Hz/s in early twentieth-century performances. The paper further documents a gain-recovery protocol that extends the analysable corpus to analogue recordings from the 1930s where portamento traces are faint in digital transfer. Applying the method to a corpus of 22 recordings spanning 1930--2012, the paper tests the hypothesis that gradient steepness correlates negatively with tempo: that slower performances produce steeper, longer slides while faster performances produce shallower slides or none at all. The results support this hypothesis, suggesting that the widely documented decline of portamento across the twentieth century is not a binary transition from presence to absence but a continuou
>
---
#### [new 007] Listening with Time: Precise Temporal Awareness for Long-Form Audio Understanding
- **分类: eess.AS**

- **简介: 该论文属于长音频理解任务，旨在解决长音频中时间感知能力下降的问题。通过构建数据集和基准，提出LAT-Audio模型，提升长音频的时间对齐与理解能力。**

- **链接: [https://arxiv.org/pdf/2604.22245](https://arxiv.org/pdf/2604.22245)**

> **作者:** Mingchen Shao; Hang Su; Wenjie Tian; Bingshen Mu; Zhennan Lin; Lichun Fan; Zhenbo Luo; Jian Luan; Lei Xie
>
> **摘要:** While Large Audio Language Models (LALMs) achieve strong performance on short audio, they degrade on long-form inputs. This degradation is more severe in temporal awareness tasks, where temporal alignment becomes increasingly inaccurate as audio duration grows. We attribute these limitations to the lack of data, benchmarks, and modeling approaches tailored for long-form temporal awareness. To bridge this gap, we first construct LAT-Chronicle, a 1.2k hour long-form audio dataset with temporal annotations across real-world scenarios. We further develop LAT-Bench, the first human-verified benchmark supporting audio up to 30 minutes while covering three core tasks: Dense Audio Caption, Temporal Audio Grounding, and Targeted Audio Caption. Leveraging these resources, we propose LAT-Audio, formulating temporal awareness as a progressive global-to-local reasoning paradigm. A global timeline is first constructed as an aligned temporal-semantic context,and the Think-With-Audio Chain-of-Thought (TWA-CoT) is then introduced to perform iterative reasoning by incorporating local audio information via tool use. Experiments show that LAT-Audio surpasses existing models on long-form audio temporal awareness tasks and improves robustness to input duration. We release the dataset, benchmark, and model to facilitate future research at this https URL.
>
---
#### [new 008] Audio Effect Estimation with DNN-Based Prediction and Search Algorithm
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频效果估计任务，旨在从湿信号中估计效果配置。通过结合深度学习预测与搜索算法，提升估计效果。**

- **链接: [https://arxiv.org/pdf/2604.22276](https://arxiv.org/pdf/2604.22276)**

> **作者:** Youichi Okita; Haruhiro Katayose
>
> **备注:** Accepted for ICASSP2026
>
> **摘要:** Audio effects play an essential role in sound design. This research addresses the task of audio effect estimation, which aims to estimate the configuration of applied effects from a wet signal. Existing approaches to this problem can be categorized into predictive approaches, which use models pre-trained in a data-driven manner, and search-based approaches, which are based on wet signal reconstruction. In this study, we propose a novel approach that integrates these approaches: first, DNNs predict the dry signal and effect configuration, and then a search is performed based on wet signal reconstruction using these predictions. By estimating the dry signal in the prediction stage, it becomes possible to complement or improve the predictions using reconstruction similarity as an objective function. The experimental evaluation showed that methods based on the proposed approach outperformed the method solely based on the predictive approach. Furthermore, the findings suggest that the task division of predicting the effect type combination followed by the search-based estimation of order and parameters was the most effective across various metrics.
>
---
#### [new 009] TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出TTS-PRISM，用于语音合成的细粒度诊断。解决传统指标无法准确评估和解释语音质量的问题，通过多维框架、对抗合成和指令微调实现更精准的性能分析。**

- **链接: [https://arxiv.org/pdf/2604.22225](https://arxiv.org/pdf/2604.22225)**

> **作者:** Xi Wang; Jie Wang; Xingchen Song; Baijun Song; Jingran Xie; Jiahe Shao; Zijian Lin; Di Wu; Meng Meng; Jian Luan; Zhiyong Wu
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** While generative text-to-speech (TTS) models approach human-level quality, monolithic metrics fail to diagnose fine-grained acoustic artifacts or explain perceptual collapse. To address this, we propose TTS-PRISM, a multi-dimensional diagnostic framework for Mandarin. First, we establish a 12-dimensional schema spanning stability to advanced expressiveness. Second, we design a targeted synthesis pipeline with adversarial perturbations and expert anchors to build a high-quality diagnostic dataset. Third, schema-driven instruction tuning embeds explicit scoring criteria and reasoning into an efficient end-to-end model. Experiments on a 1,600-sample Gold Test Set show TTS-PRISM outperforms generalist models in human alignment. Profiling six TTS paradigms establishes intuitive diagnostic flags that reveal fine-grained capability differences. TTS-PRISM is open-source, with code and checkpoints at this https URL.
>
---
## 更新

#### [replaced 001] MOS-Bench: Benchmarking Generalization Abilities of Subjective Speech Quality Assessment Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于主观语音质量评估任务，旨在解决SSQA模型泛化能力不足的问题。通过构建MOS-Bench数据集，验证了多数据集训练的有效性。**

- **链接: [https://arxiv.org/pdf/2411.03715](https://arxiv.org/pdf/2411.03715)**

> **作者:** Wen-Chin Huang; Erica Cooper; Tomoki Toda
>
> **备注:** Accepted to Transactions on Audio, Speech and Language Processing
>
> **摘要:** In this paper, we study the task of subjective speech quality assessment (SSQA), which refers to predicting the perceptual quality of speech. Owing to the development of deep neural network models, SSQA has greatly advanced and has been widely applied in scientific papers to evaluate speech generation systems. Nonetheless, the insufficient out-of-domain (OOD) generalization ability of current SSQA models is underexplored and often overlooked by researchers. To study this problem systematically, we present MOS-Bench, a diverse SSQA dataset collection that currently contains 8 training sets and 17 test sets. Through extensive experiments, we first highlight the OOD generalization challenges of existing models. We then evaluate the efficacy of multiple-dataset training, comparing straightforward data pooling against AlignNet, an existing domain-aware method. We demonstrate that pooling multiple training sets provides a simple yet effective solution, and variation in the data is a key factor for robust generalization beyond training data size.
>
---
#### [replaced 002] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文针对藏语多方言、多说话人语音合成任务，解决低资源语言数据不足的问题，提出FMSD-TTS框架，生成高质量合成语音。**

- **链接: [https://arxiv.org/pdf/2505.14351](https://arxiv.org/pdf/2505.14351)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** This paper has been substantially restructured using a revised writing style. In addition, considering that maintaining two preprints simultaneously may not fully align with academic publishing ethics, we have withdrawn the previous version. Please refer to the updated manuscript at: arXiv:509.18060
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-Ü-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [replaced 003] HumDial-EIBench: A Human-Recorded Multi-Turn Emotional Intelligence Benchmark for Audio Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于情感智能评估任务，旨在解决现有基准依赖合成语音和单轮对话的问题。工作包括构建基于真实对话的多轮情感基准，并设计多项选择题和跨模态冲突任务以更客观地评估模型情感理解能力。**

- **链接: [https://arxiv.org/pdf/2604.11594](https://arxiv.org/pdf/2604.11594)**

> **作者:** Shuiyuan Wang; Zhixian Zhao; Hongfei Xue; Chengyou Wang; Shuai Wang; Hui Bu; Xin Xu; Lei Xie
>
> **摘要:** Evaluating the emotional intelligence (EI) of audio language models (ALMs) is critical. However, existing benchmarks mostly rely on synthesized speech, are limited to single-turn interactions, and depend heavily on open-ended scoring. This paper proposes HumDial-EIBench, a comprehensive benchmark for evaluating ALMs' EI. Using real-recorded human dialogues from the ICASSP 2026 HumDial Challenge, it reformulates emotional tracking and causal reasoning into multiple-choice questions with adversarial distractors, mitigating subjective scoring bias for cognitive tasks. It retains the generation of empathetic responses and introduces an acoustic-semantic conflict task to assess robustness against contradictory multimodal signals. Evaluations of eight ALMs reveal that most models struggle with multi-turn emotional tracking and implicit causal reasoning. Furthermore, all models exhibit decoupled textual and acoustic empathy, alongside a severe text-dominance bias during cross-modal conflicts.
>
---
#### [replaced 004] Can Hierarchical Cross-Modal Fusion Predict Human Perception of AI Dubbed Content?
- **分类: eess.AS**

- **简介: 该论文属于AI配音质量评估任务，旨在解决人工评分成本高、难以规模化的问题。通过多模态融合与轻量微调，实现对AI配音内容的自动评价。**

- **链接: [https://arxiv.org/pdf/2603.28717](https://arxiv.org/pdf/2603.28717)**

> **作者:** Ashwini Dasare; Nirmesh Shah; Ashishkumar Gudmalwar; Pankaj Wasnik
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Evaluating AI generated dubbed content is inherently multi-dimensional, shaped by synchronization, intelligibility, speaker consistency, emotional alignment, and semantic context. Human Mean Opinion Scores (MOS) remain the gold standard but are costly and impractical at scale. We present a hierarchical multimodal architecture for perceptually meaningful dubbing evaluation, integrating complementary cues from audio, video, and text. The model captures fine-grained features such as speaker identity, prosody, and content from audio, facial expressions and scene-level cues from video and semantic context from text, which are progressively fused through intra and inter-modal layers. Lightweight LoRA adapters enable parameter-efficient fine-tuning across modalities. To overcome limited subjective labels, we derive proxy MOS by aggregating objective metrics with weights optimized via active learning. The proposed architecture was trained on 12k Hindi-English bidirectional dubbed clips, followed by fine-tuning with human MOS. Our approach achieves strong perceptual alignment (PCC > 0.75), providing a scalable solution for automatic evaluation of AI-dubbed content.
>
---
#### [replaced 005] Full-Duplex Interaction in Spoken Dialogue Systems: A Comprehensive Study from the ICASSP 2026 HumDial Challenge
- **分类: eess.AS**

- **简介: 该论文属于对话系统任务，旨在解决传统系统在全双工交互中的不足。通过构建基准数据集和评估框架，提升系统处理实时干扰和动态对话的能力。**

- **链接: [https://arxiv.org/pdf/2604.21406](https://arxiv.org/pdf/2604.21406)**

> **作者:** Chengyou Wang; Hongfei Xue; Guojian Li; Zhixian Zhao; Shuiyuan Wang; Shuai Wang; Xin Xu; Hui Bu; Lei Xie
>
> **备注:** 5 pages, 1 figures
>
> **摘要:** Full-duplex interaction, where speakers and listeners converse simultaneously, is a key element of human communication often missing from traditional spoken dialogue systems. These systems, based on rigid turn-taking paradigms, struggle to respond naturally in dynamic conversations. The Full-Duplex Interaction Track of ICASSP 2026 Human-like Spoken Dialogue Systems Challenge (HumDial Challenge) aims to advance the evaluation of full-duplex systems by offering a framework for handling real-time interruptions, speech overlap, and dynamic turn negotiation. We introduce a comprehensive benchmark for full-duplex spoken dialogue systems, built from the HumDial Challenge. We release a high-quality dual-channel dataset of real human-recorded conversations, capturing interruptions, overlapping speech, and feedback mechanisms. This dataset forms the basis for the HumDial-FDBench benchmark, which assesses a system's ability to handle interruptions while maintaining conversational flow. Additionally, we create a public leaderboard to compare the performance of open-source and proprietary models, promoting transparent, reproducible evaluation. These resources support the development of more responsive, adaptive, and human-like dialogue systems.
>
---
#### [replaced 006] ActorMind: Emulating Human Actor Reasoning for Speech Role-Playing
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出ActorMind框架，解决语音角色扮演任务中的情感与角色一致性问题，通过多代理系统模拟人类演员推理过程，提升语音互动的真实性。**

- **链接: [https://arxiv.org/pdf/2604.11103](https://arxiv.org/pdf/2604.11103)**

> **作者:** Xi Chen; Wei Xue; Yike Guo
>
> **摘要:** Role-playing has garnered rising attention as it provides a strong foundation for human-machine interaction and facilitates sociological research. However, current work is confined to textual modalities, neglecting speech, which plays a predominant role in daily life, thus limiting genuine role-playing. To bridge this gap, we conceptualize and benchmark speech role-playing through ActorMindBench, and we present a corresponding reasoning framework, called ActorMind. Specifically, (1) Speech Role-Playing enables models to deliver spontaneous responses with personalized verbal traits based on their role, the scene, and spoken dialogue. (2) ActorMindBench is a hierarchical benchmark comprises Utterance-Level content with 7,653 utterances, Scene-Level content with 313 scenes, and Role-Level content with 6 roles. (3) ActorMind is an off-the-shelf, multi-agent, chain-of-though style reasoning framework that emulates how human actors perform in theaters. Concretely, ActorMind first reads its assigned role description via Eye Agent, then comprehends emotional cues within contextual spoken dialogues through Ear Agent. Subsequently, Brain Agent generates a descriptive emotional state, and finally, Mouth Agent delivers the scripts infused with corresponding emotion state. Experimental results demonstrate the effectiveness of ActorMind in enhancing speech role-playing.
>
---
