# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Semantic visually-guided acoustic highlighting with large vision-language models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频混合任务，旨在解决手动音频混音效率低的问题。通过视觉语义提示提升音频重混质量，实验表明相机焦点、色调和场景背景最有效。**

- **链接: [https://arxiv.org/pdf/2601.08871v1](https://arxiv.org/pdf/2601.08871v1)**

> **作者:** Junhua Huang; Chao Huang; Chenliang Xu
>
> **摘要:** Balancing dialogue, music, and sound effects with accompanying video is crucial for immersive storytelling, yet current audio mixing workflows remain largely manual and labor-intensive. While recent advancements have introduced the visually guided acoustic highlighting task, which implicitly rebalances audio sources using multimodal guidance, it remains unclear which visual aspects are most effective as conditioning signals.We address this gap through a systematic study of whether deep video understanding improves audio remixing. Using textual descriptions as a proxy for visual analysis, we prompt large vision-language models to extract six types of visual-semantic aspects, including object and character appearance, emotion, camera focus, tone, scene background, and inferred sound-related cues. Through extensive experiments, camera focus, tone, and scene background consistently yield the largest improvements in perceptual mix quality over state-of-the-art baselines. Our findings (i) identify which visual-semantic cues most strongly support coherent and visually aligned audio remixing, and (ii) outline a practical path toward automating cinema-grade sound design using lightweight guidance derived from large vision-language models.
>
---
#### [new 002] Towards Realistic Synthetic Data for Automatic Drum Transcription
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于自动鼓乐转录任务，解决缺乏配对音频-MIDI数据的问题。通过半监督方法构建高质量鼓样本文本，合成数据训练模型，取得最佳效果。**

- **链接: [https://arxiv.org/pdf/2601.09520v1](https://arxiv.org/pdf/2601.09520v1)**

> **作者:** Pierfrancesco Melucci; Paolo Merialdo; Taketo Akama
>
> **摘要:** Deep learning models define the state-of-the-art in Automatic Drum Transcription (ADT), yet their performance is contingent upon large-scale, paired audio-MIDI datasets, which are scarce. Existing workarounds that use synthetic data often introduce a significant domain gap, as they typically rely on low-fidelity SoundFont libraries that lack acoustic diversity. While high-quality one-shot samples offer a better alternative, they are not available in a standardized, large-scale format suitable for training. This paper introduces a new paradigm for ADT that circumvents the need for paired audio-MIDI training data. Our primary contribution is a semi-supervised method to automatically curate a large and diverse corpus of one-shot drum samples from unlabeled audio sources. We then use this corpus to synthesize a high-quality dataset from MIDI files alone, which we use to train a sequence-to-sequence transcription model. We evaluate our model on the ENST and MDB test sets, where it achieves new state-of-the-art results, significantly outperforming both fully supervised methods and previous synthetic-data approaches. The code for reproducing our experiments is publicly available at https://github.com/pier-maker92/ADT_STR
>
---
#### [new 003] Speech-Hands: A Self-Reflection Voice Agentic Approach to Speech Recognition and Audio Reasoning with Omni Perception
- **分类: cs.SD; cs.AI; cs.CL; cs.MA; eess.AS**

- **简介: 该论文提出Speech-Hands框架，解决语音识别与音频推理中的自我信任与外部感知决策问题，通过自省机制提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.09413v1](https://arxiv.org/pdf/2601.09413v1)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Jinchuan Tian; Hanrong Ye; Ankita Pasad; Szu-wei Fu; Arushi Goel; Ryo Hachiuma; Shizhe Diao; Kunal Dhawan; Sreyan Ghosh; Yusuke Hirota; Zhehuai Chen; Rafael Valle; Ehsan Hosseini Asl; Chenhui Chu; Shinji Watanabe; Yu-Chiang Frank Wang; Boris Ginsburg
>
> **备注:** Preprint. The version was submitted in October 2025
>
> **摘要:** We introduce a voice-agentic framework that learns one critical omni-understanding skill: knowing when to trust itself versus when to consult external audio perception. Our work is motivated by a crucial yet counterintuitive finding: naively fine-tuning an omni-model on both speech recognition and external sound understanding tasks often degrades performance, as the model can be easily misled by noisy hypotheses. To address this, our framework, Speech-Hands, recasts the problem as an explicit self-reflection decision. This learnable reflection primitive proves effective in preventing the model from being derailed by flawed external candidates. We show that this agentic action mechanism generalizes naturally from speech recognition to complex, multiple-choice audio reasoning. Across the OpenASR leaderboard, Speech-Hands consistently outperforms strong baselines by 12.1% WER on seven benchmarks. The model also achieves 77.37% accuracy and high F1 on audio QA decisions, showing robust generalization and reliability across diverse audio question answering datasets. By unifying perception and decision-making, our work offers a practical path toward more reliable and resilient audio intelligence.
>
---
#### [new 004] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD**

- **简介: 该论文属于语音建模任务，旨在解决语义与声学特征难以分离的问题。提出DSA-Tokenizer，通过优化约束分离语义和声学token，提升语音生成质量与可控性。**

- **链接: [https://arxiv.org/pdf/2601.09239v1](https://arxiv.org/pdf/2601.09239v1)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Jianping Wang; Linqi Song
>
> **摘要:** Speech tokenizers serve as the cornerstone of discrete Speech Large Language Models (Speech LLMs). Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose DSA-Tokenizer, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. To eliminate rigid length constraints between the two sequences, we introduce a hierarchical Flow-Matching decoder that further improve speech generation quality.Furthermore, We employ a joint reconstruction-recombination training strategy to enforce this separation. DSA-Tokenizer enables high fidelity reconstruction and flexible recombination through robust disentanglement, facilitating controllable generation in speech LLMs. Our analysis highlights disentangled tokenization as a pivotal paradigm for future speech modeling. Audio samples are avaialble at https://anonymous.4open.science/w/DSA_Tokenizer_demo/. The code and model will be made publicly available after the paper has been accepted.
>
---
#### [new 005] Linear Complexity Self-Supervised Learning for Music Understanding with Random Quantizer
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦音乐信息检索任务，旨在缩小基础模型规模。通过结合Branchformer与SummaryMixing，并引入随机量化，有效降低模型大小，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2601.09603v1](https://arxiv.org/pdf/2601.09603v1)**

> **作者:** Petros Vavaroutsos; Theodoros Palamas; Pantelis Vikatos
>
> **备注:** accepted by ACM/SIGAPP Symposium on Applied Computing (SAC 2026)
>
> **摘要:** In recent years, foundation models have become very popular due to their exceptional performance, mainly in natural language (NLP) tasks where they were first introduced. These models usually consist of hundreds of millions, or even billions, of parameters, making them resource-intensive during training and in production systems, leading to increased costs. This paper focuses on the reduction of a foundation's model size when applied to music information retrieval (MIR) tasks. Our research combines the Branchformer architecture with SummaryMixing, which were first applied in speech recognition, along with a random quantization process. To facilitate reproducibility, we conduct pre-training on publicly available datasets, complemented by a proprietary dataset comparable in scale to other private datasets reported in the literature. We ensure robust evaluation by using a framework consisting of a variety of downstream MIR tasks. Our results show that our architecture achieves competitive performance when compared with other state-of-the-art models that use multi-head self-attention, while reducing the model size from 8.5% up to 12.3%.
>
---
#### [new 006] Research on Piano Timbre Transformation System Based on Diffusion Model
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音乐音色转换任务，旨在将不同乐器的音乐精准转为钢琴音色。通过扩散模型结合音高和响度编码器，实现高质量音色转换。**

- **链接: [https://arxiv.org/pdf/2601.09333v1](https://arxiv.org/pdf/2601.09333v1)**

> **作者:** Chun-Chieh Hsu; Tsai-Ling Hsu; Chen-Chen Yeh; Shao-Chien Lu; Cheng-Han Wu; Bing-Ze Liu; Timothy K. Shih; Yu-Cheng Lin
>
> **摘要:** We propose a timbre conversion model based on the Diffusion architecture de-signed to precisely translate music played by various instruments into piano ver-sions. The model employs a Pitch Encoder and Loudness Encoder to extract pitch and loudness features of the music, which serve as conditional inputs to the Dif-fusion Model's decoder, generating high-quality piano timbres. Case analysis re-sults show that the model performs excellently in terms of pitch accuracy and timbral similarity, maintaining stable conversion across different musical styles (classical, jazz, pop) and lengths (from short clips to full pieces). Particularly, the model maintains high sound quality and accuracy even when dealing with rapidly changing notes and complex musical structures, demonstrating good generaliza-tion capability. Additionally, the model has the potential for real-time musical conversion and is suitable for live performances and digital music creation tools. Future research will focus on enhancing the handling of loudness dynamics and incorporating additional musical features (such as timbral variations and rhythmic complexity) to improve the model's adaptability and expressiveness. We plan to explore the model's application potential in other timbre conversion tasks, such as converting vocals to instrumental sounds or integration with MIDI digital pianos, further expanding the application scope of the Diffusion-based timbre conversion model in the field of music generation.
>
---
#### [new 007] SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing
- **分类: cs.SD; cs.CL; cs.MM**

- **简介: 该论文提出SLAM-LLM，一个面向语音、音频和音乐处理的多模态大语言模型框架，解决现有框架对音频支持不足的问题，提供模块化配置和训练方案。**

- **链接: [https://arxiv.org/pdf/2601.09385v1](https://arxiv.org/pdf/2601.09385v1)**

> **作者:** Ziyang Ma; Guanrou Yang; Wenxi Chen; Zhifu Gao; Yexing Du; Xiquan Li; Zhisheng Zheng; Haina Zhu; Jianheng Zhuo; Zheshu Song; Ruiyang Xu; Tiranrui Wang; Yifan Yang; Yanqiao Zhu; Zhikang Niu; Liumeng Xue; Yinghao Ma; Ruibin Yuan; Shiliang Zhang; Kai Yu; Eng Siong Chng; Xie Chen
>
> **备注:** Published in IEEE Journal of Selected Topics in Signal Processing (JSTSP)
>
> **摘要:** The recent surge in open-source Multimodal Large Language Models (MLLM) frameworks, such as LLaVA, provides a convenient kickoff for artificial intelligence developers and researchers. However, most of the MLLM frameworks take vision as the main input modality, and provide limited in-depth support for the modality of speech, audio, and music. This situation hinders the development of audio-language models, and forces researchers to spend a lot of effort on code writing and hyperparameter tuning. We present SLAM-LLM, an open-source deep learning framework designed to train customized MLLMs, focused on speech, language, audio, and music processing. SLAM-LLM provides a modular configuration of different encoders, projectors, LLMs, and parameter-efficient fine-tuning plugins. SLAM-LLM also includes detailed training and inference recipes for mainstream tasks, along with high-performance checkpoints like LLM-based Automatic Speech Recognition (ASR), Automated Audio Captioning (AAC), and Music Captioning (MC). Some of these recipes have already reached or are nearing state-of-the-art performance, and some relevant techniques have also been accepted by academic papers. We hope SLAM-LLM will accelerate iteration, development, data engineering, and model training for researchers. We are committed to continually pushing forward audio-based MLLMs through this open-source framework, and call on the community to contribute to the LLM-based speech, audio and music processing.
>
---
#### [new 008] Analysis of the Maximum Prediction Gain of Short-Term Prediction on Sustained Speech
- **分类: cs.SD**

- **简介: 该论文属于语音信号处理任务，旨在研究短时预测的最大预测增益。通过分析不同语音类型，比较线性与非线性预测效果，确定预测增益上限及影响因素。**

- **链接: [https://arxiv.org/pdf/2601.09461v1](https://arxiv.org/pdf/2601.09461v1)**

> **作者:** Reemt Hinrichs; Muhamad Fadli Damara; Stephan Preihs; Jörn Ostermann
>
> **备注:** Rejected at Eurasip for practical irrelevancy. Submitted here for reference. Originally accepted at DCC 2020 (Poster) but withdrawn due to page count limit
>
> **摘要:** Signal prediction is widely used in, e.g., economic forecasting, echo cancellation and in data compression, particularly in predictive coding of speech and music. Predictive coding algorithms reduce the bit-rate required for data transmission or storage by signal prediction. The prediction gain is a classic measure in applied signal coding of the quality of a predictor, as it links the mean-squared prediction error to the signal-to-quantization-noise of predictive coders. To evaluate predictor models, knowledge about the maximum achievable prediction gain independent of a predictor model is desirable. In this manuscript, Nadaraya-Watson kernel-regression (NWKR) and an information theoretic upper bound are applied to analyze the upper bound of the prediction gain on a newly recorded dataset of sustained speech/phonemes. It was found that for unvoiced speech a linear predictor always achieves the maximum prediction gain within at most 0.3 dB. On voiced speech, the optimum one-tap predictor was found to be linear but starting with two taps, the maximum achievable prediction gain was found to be about 2 dB to 6 dB above the prediction gain of the linear predictor. Significant differences between speakers/subjects were observed. The created dataset as well as the code can be obtained for research purpose upon request.
>
---
#### [new 009] Population-Aligned Audio Reproduction With LLM-Based Equalizers
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频处理任务，旨在解决传统音频均衡调整繁琐的问题。通过LLM将文本提示转换为均衡设置，实现更智能、个性化的音效调整。**

- **链接: [https://arxiv.org/pdf/2601.09448v1](https://arxiv.org/pdf/2601.09448v1)**

> **作者:** Ioannis Stylianou; Jon Francombe; Pablo Martinez-Nuevo; Sven Ewan Shepstone; Zheng-Hua Tan
>
> **备注:** 12 pages, 13 figures, 2 tables, IEEE JSTSP journal submission under first revision
>
> **摘要:** Conventional audio equalization is a static process that requires manual and cumbersome adjustments to adapt to changing listening contexts (e.g., mood, location, or social setting). In this paper, we introduce a Large Language Model (LLM)-based alternative that maps natural language text prompts to equalization settings. This enables a conversational approach to sound system control. By utilizing data collected from a controlled listening experiment, our models exploit in-context learning and parameter-efficient fine-tuning techniques to reliably align with population-preferred equalization settings. Our evaluation methods, which leverage distributional metrics that capture users' varied preferences, show statistically significant improvements in distributional alignment over random sampling and static preset baselines. These results indicate that LLMs could function as "artificial equalizers," contributing to the development of more accessible, context-aware, and expert-level audio tuning methods.
>
---
#### [new 010] Echoes of Ideology: Toward an Audio Analysis Pipeline to Unveil Character Traits in Historical Nazi Propaganda Films
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频分析任务，旨在通过计算方法揭示纳粹宣传片中的意识形态和角色特征。工作包括语音分割、转录及心理语言分析，以识别传播模式。**

- **链接: [https://arxiv.org/pdf/2601.08879v1](https://arxiv.org/pdf/2601.08879v1)**

> **作者:** Nicolas Ruth; Manuel Burghardt
>
> **摘要:** This study investigates the use of computational audio analysis to examine ideological narratives in Nazi propaganda films. Employing a three-step pipeline, speaker diarization, audio transcription and psycholinguistic analysis, it reveals ideological patterns in characters. Despite current issues with speaker diarization, the methodology provides insights into character traits and propaganda narratives, suggesting scalable applications.
>
---
## 更新

#### [replaced 001] Toward Conversational Hungarian Speech Recognition: Introducing the BEA-Large and BEA-Dialogue Datasets
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决匈牙利语对话语音数据不足的问题。通过构建两个新数据集并建立基线模型，推动匈牙利语语音技术发展。**

- **链接: [https://arxiv.org/pdf/2511.13529v2](https://arxiv.org/pdf/2511.13529v2)**

> **作者:** Máté Gedeon; Piroska Zsófia Barta; Péter Mihajlik; Tekla Etelka Gráczi; Anna Kohári; Katalin Mády
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** The advancement of automatic speech recognition (ASR) has been largely enhanced by extensive datasets in high-resource languages, while languages such as Hungarian remain underrepresented due to limited spontaneous and conversational corpora. To address this gap, we introduce two new datasets -- BEA-Large and BEA-Dialogue -- constructed from the previously unprocessed portions of the Hungarian speech corpus named BEA. BEA-Large extends BEA-Base with 255 hours of spontaneous speech from 433 speakers, enriched with detailed segment-level metadata. BEA-Dialogue, comprising 85 hours of spontaneous conversations, is a Hungarian speech corpus featuring natural dialogues partitioned into speaker-independent subsets, supporting research in conversational ASR and speaker diarization. We establish reproducible baselines on these datasets using publicly available ASR models, with the fine-tuned Fast Conformer model achieving word error rates as low as 14.18% on spontaneous and 4.8% on repeated speech. Diarization experiments yield diarization error rates between 12.46% and 17.40%, providing reference points for future improvements. The results highlight the persistent difficulty of conversational ASR, particularly due to disfluencies, overlaps, and informal speech patterns. By releasing these datasets and baselines, we aim to advance Hungarian speech technology and offer a methodological framework for developing spontaneous and conversational benchmarks in other languages.
>
---
#### [replaced 002] MATS: An Audio Language Model under Text-only Supervision
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出MATS，一种在纯文本监督下训练的音频语言模型，解决音频-语言对数据获取成本高的问题，通过文本训练实现音频理解。**

- **链接: [https://arxiv.org/pdf/2502.13433v3](https://arxiv.org/pdf/2502.13433v3)**

> **作者:** Wen Wang; Ruibing Hou; Hong Chang; Shiguang Shan; Xilin Chen
>
> **备注:** Accepted by ICML2025
>
> **摘要:** Large audio-language models (LALMs), built upon powerful Large Language Models (LLMs), have exhibited remarkable audio comprehension and reasoning capabilities. However, the training of LALMs demands a large corpus of audio-language pairs, which requires substantial costs in both data collection and training resources. In this paper, we propose \textbf{MATS}, an audio-language multimodal LLM designed to handle \textbf{M}ultiple \textbf{A}udio task using solely \textbf{T}ext-only \textbf{S}upervision. By leveraging pre-trained audio-language alignment models such as CLAP, we develop a text-only training strategy that projects the shared audio-language latent space into LLM latent space, endowing the LLM with audio comprehension capabilities without relying on audio data during training. To further bridge the modality gap between audio and language embeddings within CLAP, we propose the \textbf{S}trongly-rel\textbf{a}ted \textbf{n}oisy \textbf{t}ext with \textbf{a}udio (\textbf{Santa}) mechanism. Santa maps audio embeddings into CLAP language embedding space while preserving essential information from the audio input. Extensive experiments demonstrate that MATS, despite being trained exclusively on text data, achieves competitive performance compared to recent LALMs trained on large-scale audio-language pairs. The code is publicly available in \href{https://github.com/wangwen-banban/MATS}{https://github.com/wangwen-banban/MATS}.
>
---
#### [replaced 003] A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在提升情感分类的准确性。提出DCRF-BiLSTM模型，在多个数据集上实现高精度识别。**

- **链接: [https://arxiv.org/pdf/2507.07046v2](https://arxiv.org/pdf/2507.07046v2)**

> **作者:** Shahana Yasmin Chowdhury; Bithi Banik; Md Tamjidul Hoque; Shreya Banerjee
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets.
>
---
#### [replaced 004] MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转写与说话人辨识任务，旨在解决传统系统无法端到端处理、上下文受限等问题，提出MOSS Transcribe Diarize模型实现精准时间戳转写。**

- **链接: [https://arxiv.org/pdf/2601.01554v4](https://arxiv.org/pdf/2601.01554v4)**

> **作者:** MOSI. AI; :; Donghua Yu; Zhengyuan Lin; Chen Yang; Yiyang Zhang; Hanfu Chen; Jingqi Chen; Ke Chen; Liwei Fan; Yi Jiang; Jie Zhu; Muchen Li; Wenxuan Wang; Yang Wang; Zhe Xu; Yitian Gong; Yuqian Zhang; Wenbo Zhang; Zhaoye Fei; Songlin Wang; Zhiyu Wu; Qinyuan Cheng; Shimin Li; Xipeng Qiu
>
> **摘要:** Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks.
>
---
#### [replaced 005] Integrated Minimum Mean Squared Error Algorithms for Combined Acoustic Echo Cancellation and Noise Reduction
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决噪声和声学回声同时抑制的问题。提出集成方法，统一模型和优化目标，实现更优的AEC与NR联合处理。**

- **链接: [https://arxiv.org/pdf/2412.04267v2](https://arxiv.org/pdf/2412.04267v2)**

> **作者:** Arnout Roebben; Toon van Waterschoot; Jan Wouters; Marc Moonen
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** In many speech recording applications, noise and acoustic echo corrupt the desired speech. Consequently, combined noise reduction (NR) and acoustic echo cancellation (AEC) is required. Generally, a cascade approach is followed, i.e., the AEC and NR are designed in isolation by selecting a separate signal model, separate cost function, and separate solution strategy. The AEC and NR are then cascaded one after the other, not accounting for their interaction. In this paper, an integrated approach is proposed to consider this interaction in a general multi-microphone/multi-loudspeaker setup. Therefore, a single signal model of either the microphone signal vector or the extended signal vector, obtained by stacking microphone and loudspeaker signals, is selected, a single mean squared error cost function is formulated, and a common solution strategy is used. Using this microphone signal model, a multi-channel Wiener filter (MWF) is derived. Using the extended signal model, it is shown that an extended MWF (MWFext) can be derived, and several equivalent expressions can be found, which are nevertheless shown to be interpretable as cascade algorithms. Specifically, the MWFext is shown to be equivalent to algorithms where the AEC precedes the NR (AEC-NR), the NR precedes the AEC (NR-AEC), and the extended NR (NRext) precedes the AEC and post-filter (PF) (NRext-AEC-PF). Under rank-deficiency conditions the MWFext is non-unique. Equivalence then amounts to the expressions being specific, not necessarily minimum-norm solutions, for this MWFext. The practical performances differ due to non-stationarities and imperfect correlation matrix estimation, with the AEC-NR and NRext-AEC-PF attaining best overall performance.
>
---
#### [replaced 006] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多说话人语音识别任务，旨在解决单通道音频中多人重叠语音的识别与归属问题。工作包括梳理E2E架构、分析不同模型结构并评估其性能。**

- **链接: [https://arxiv.org/pdf/2505.10975v2](https://arxiv.org/pdf/2505.10975v2)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** Accepted for publication in Computer Speech & Language (CSL)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [replaced 007] MORE: Multi-Objective Adversarial Attacks on Speech Recognition
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文属于语音识别的对抗攻击任务，旨在同时降低识别准确率和推理效率。提出MORE方法，通过多目标优化实现双重破坏。**

- **链接: [https://arxiv.org/pdf/2601.01852v2](https://arxiv.org/pdf/2601.01852v2)**

> **作者:** Xiaoxue Gao; Zexin Li; Yiming Chen; Nancy F. Chen
>
> **备注:** 19 pages
>
> **摘要:** The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.
>
---
