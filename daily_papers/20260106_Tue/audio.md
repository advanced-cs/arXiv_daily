# 音频 cs.SD;  eess.AS

- **最新发布 21 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] BeatlesFC: Harmonic function annotations of Isophonics' The Beatles dataset
- **分类: cs.SD**

- **简介: 该论文属于音乐信息检索任务，旨在为The Beatles数据集添加和声功能标注，解决 chord label 与音乐结构之间的关联问题。工作包括构建 harmonic function 标注集。**

- **链接: [https://arxiv.org/pdf/2601.02099v1](https://arxiv.org/pdf/2601.02099v1)**

> **作者:** Ji Yeoung Sim; Rebecca Moranis; Johanna Devaney
>
> **备注:** International Society for Music Information Retrieval, Late-Breaking Demo 2024
>
> **摘要:** This paper presents BeatlesFC, a set of harmonic function annotations for Isophonics' The Beatles dataset. Harmonic function annotations characterize chord labels as stable (tonic) or unstable (predominant, dominant). They operate at the level of musical phrases, serving as a link between chord labels and higher-level formal structures.
>
---
#### [new 002] MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于多模态生成任务，旨在解决音频视频同步生成中语音控制与零样本语音克隆的问题。提出MM-Sonate框架，实现精准语言对齐和高质量语音克隆。**

- **链接: [https://arxiv.org/pdf/2601.01568v1](https://arxiv.org/pdf/2601.01568v1)**

> **作者:** Chunyu Qiang; Jun Wang; Xiaopeng Wang; Kang Yin; Yuxin Guo; Xijuan Zeng; Nan Li; Zihan Li; Yuzhe Liang; Ziyu Zhang; Teng Ma; Yushen Chen; Zhongliang Liu; Feng Deng; Chen Zhang; Pengfei Wan
>
> **摘要:** Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems.
>
---
#### [new 003] On the Role of Spatial Features in Foundation-Model-Based Speaker Diarization
- **分类: eess.AS**

- **简介: 该论文属于说话人日志任务，研究如何将空间特征融入基于基础模型的说话人日志系统，以提升性能。工作包括评估多通道空间特征对单通道系统的改进效果。**

- **链接: [https://arxiv.org/pdf/2601.02231v1](https://arxiv.org/pdf/2601.02231v1)**

> **作者:** Marc Deegen; Tobias Gburrek; Tobias Cord-Landwehr; Thilo von Neumann; Jiangyu Han; Lukáš Burget; Reinhold Haeb-Umbach
>
> **备注:** Accepted at HSCMA 2026
>
> **摘要:** Recent advances in speaker diarization exploit large pretrained foundation models, such as WavLM, to achieve state-of-the-art performance on multiple datasets. Systems like DiariZen leverage these rich single-channel representations, but are limited to single-channel audio, preventing the use of spatial cues available in multi-channel recordings. This work analyzes the impact of incorporating spatial information into a state-of-the-art single-channel diarization system by evaluating several strategies for conditioning the model on multi-channel spatial features. Experiments on meeting-style datasets indicate that spatial information can improve diarization performance, but the overall improvement is smaller than expected for the proposed system, suggesting that the features aggregated over all WavLM layers already capture much of the information needed for accurate speaker discrimination, also in overlapping speech regions. These findings provide insight into the potential and limitations of using spatial cues to enhance foundation model-based diarization.
>
---
#### [new 004] MORE: Multi-Objective Adversarial Attacks on Speech Recognition
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文属于语音识别鲁棒性研究，旨在解决对抗攻击下模型准确性和效率的问题。提出MORE方法，同时降低识别准确率并增加计算成本。**

- **链接: [https://arxiv.org/pdf/2601.01852v1](https://arxiv.org/pdf/2601.01852v1)**

> **作者:** Xiaoxue Gao; Zexin Li; Yiming Chen; Nancy F. Chen
>
> **备注:** 19 pages
>
> **摘要:** The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.
>
---
#### [new 005] DARC: Drum accompaniment generation with fine-grained rhythm control
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出DARC，解决鼓伴奏生成中节奏控制与音乐上下文感知的难题。通过结合音乐上下文和显式节奏提示，实现更精细的节奏控制。**

- **链接: [https://arxiv.org/pdf/2601.02357v1](https://arxiv.org/pdf/2601.02357v1)**

> **作者:** Trey Brosnan
>
> **摘要:** In music creation, rapid prototyping is essential for exploring and refining ideas, yet existing generative tools often fall short when users require both structural control and stylistic flexibility. Prior approaches in stem-to-stem generation can condition on other musical stems but offer limited control over rhythm, and timbre-transfer methods allow users to specify specific rhythms, but cannot condition on musical context. We introduce DARC, a generative drum accompaniment model that conditions both on musical context from other stems and explicit rhythm prompts such as beatboxing or tapping tracks. Using parameter-efficient fine-tuning, we augment STAGE, a state-of-the-art drum stem generator, with fine-grained rhythm control while maintaining musical context awareness.
>
---
#### [new 006] Improving Code-Switching Speech Recognition with TTS Data Augmentation
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音识别任务，旨在解决代码转换语音数据不足的问题。通过使用多语言TTS生成合成数据，提升ASR性能。**

- **链接: [https://arxiv.org/pdf/2601.00935v1](https://arxiv.org/pdf/2601.00935v1)**

> **作者:** Yue Heng Yeo; Yuchen Hu; Shreyas Gopal; Yizhou Peng; Hexin Liu; Eng Siong Chng
>
> **备注:** This paper was accepted by APSIPA 2025
>
> **摘要:** Automatic speech recognition (ASR) for conversational code-switching speech remains challenging due to the scarcity of realistic, high-quality labeled speech data. This paper explores multilingual text-to-speech (TTS) models as an effective data augmentation technique to address this shortage. Specifically, we fine-tune the multilingual CosyVoice2 TTS model on the SEAME dataset to generate synthetic conversational Chinese-English code-switching speech, significantly increasing the quantity and speaker diversity of available training data. Our experiments demonstrate that augmenting real speech with synthetic speech reduces the mixed error rate (MER) from 12.1 percent to 10.1 percent on DevMan and from 17.8 percent to 16.0 percent on DevSGE, indicating consistent performance gains. These results confirm that multilingual TTS is an effective and practical tool for enhancing ASR robustness in low-resource conversational code-switching scenarios.
>
---
#### [new 007] UltraEval-Audio: A Unified Framework for Comprehensive Evaluation of Audio Foundation Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出UltraEval-Audio，解决音频基础模型评估缺乏统一框架、编码器评估不全面及中文基准缺失的问题，涵盖多语言任务与24个模型的综合评估。**

- **链接: [https://arxiv.org/pdf/2601.01373v1](https://arxiv.org/pdf/2601.01373v1)**

> **作者:** Qundong Shi; Jie Zhou; Biyuan Lin; Junbo Cui; Guoyang Zeng; Yixuan Zhou; Ziyang Wang; Xin Liu; Zhen Luo; Yudong Wang; Zhiyuan Liu
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** The development of audio foundation models has accelerated rapidly since the emergence of GPT-4o. However, the lack of comprehensive evaluation has become a critical bottleneck for further progress in the field, particularly in audio generation. Current audio evaluation faces three major challenges: (1) audio evaluation lacks a unified framework, with datasets and code scattered across various sources, hindering fair and efficient cross-model comparison;(2) audio codecs, as a key component of audio foundation models, lack a widely accepted and holistic evaluation methodology; (3) existing speech benchmarks are heavily reliant on English, making it challenging to objectively assess models' performance on Chinese. To address the first issue, we introduce UltraEval-Audio, a unified evaluation framework for audio foundation models, specifically designed for both audio understanding and generation tasks. UltraEval-Audio features a modular architecture, supporting 10 languages and 14 core task categories, while seamlessly integrating 24 mainstream models and 36 authoritative benchmarks. To enhance research efficiency, the framework provides a one-command evaluation feature, accompanied by real-time public leaderboards. For the second challenge, UltraEval-Audio adopts a novel comprehensive evaluation scheme for audio codecs, evaluating performance across three key dimensions: semantic accuracy, timbre fidelity, and acoustic quality. To address the third issue, we propose two new Chinese benchmarks, SpeechCMMLU and SpeechHSK, designed to assess Chinese knowledge proficiency and language fluency. We wish that UltraEval-Audio will provide both academia and industry with a transparent, efficient, and fair platform for comparison of audio models. Our code, benchmarks, and leaderboards are available at https://github.com/OpenBMB/UltraEval-Audio.
>
---
#### [new 008] Index-ASR Technical Report
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决LLM-based ASR系统的幻觉错误和定制化不足问题，提出Index-ASR系统提升鲁棒性和热词识别能力。**

- **链接: [https://arxiv.org/pdf/2601.00890v1](https://arxiv.org/pdf/2601.00890v1)**

> **作者:** Zheshu Song; Lu Wang; Wei Deng; Zhuo Yang; Yong Wu; Bin Xia
>
> **备注:** Index-ASR technical report
>
> **摘要:** Automatic speech recognition (ASR) has witnessed remarkable progress in recent years, largely driven by the emergence of LLM-based ASR paradigm. Despite their strong performance on a variety of open-source benchmarks, existing LLM-based ASR systems still suffer from two critical limitations. First, they are prone to hallucination errors, often generating excessively long and repetitive outputs that are not well grounded in the acoustic input. Second, they provide limited support for flexible and fine-grained contextual customization. To address these challenges, we propose Index-ASR, a large-scale LLM-based ASR system designed to simultaneously enhance robustness and support customizable hotword recognition. The core idea of Index-ASR lies in the integration of LLM and large-scale training data enriched with background noise and contextual information. Experimental results show that our Index-ASR achieves strong performance on both open-source benchmarks and in-house test sets, highlighting its robustness and practicality for real-world ASR applications.
>
---
#### [new 009] Diffusion Timbre Transfer Via Mutual Information Guided Inpainting
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐音频风格迁移任务，旨在通过无训练的推理阶段操作实现音色转换，解决音色改变与结构保持的平衡问题。**

- **链接: [https://arxiv.org/pdf/2601.01294v1](https://arxiv.org/pdf/2601.01294v1)**

> **作者:** Ching Ho Lee; Javier Nistal; Stefan Lattner; Marco Pasini; George Fazekas
>
> **备注:** 6 pages, 2 figures, 3 tables
>
> **摘要:** We study timbre transfer as an inference-time editing problem for music audio. Starting from a strong pre-trained latent diffusion model, we introduce a lightweight procedure that requires no additional training: (i) a dimension-wise noise injection that targets latent channels most informative of instrument identity, and (ii) an early-step clamping mechanism that re-imposes the input's melodic and rhythmic structure during reverse diffusion. The method operates directly on audio latents and is compatible with text/audio conditioning (e.g., CLAP). We discuss design choices,analyze trade-offs between timbral change and structural preservation, and show that simple inference-time controls can meaningfully steer pre-trained models for style-transfer use cases.
>
---
#### [new 010] OV-InstructTTS: Towards Open-Vocabulary Instruct Text-to-Speech
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音任务，解决传统方法难以处理高阶指令的问题。提出OV-InstructTTS，通过开放词汇指令和推理框架提升语音合成的表达性和准确性。**

- **链接: [https://arxiv.org/pdf/2601.01459v1](https://arxiv.org/pdf/2601.01459v1)**

> **作者:** Yong Ren; Jiangyan Yi; Jianhua Tao; Haiyang Sun; Zhengqi Wen; Hao Gu; Le Xu; Ye Bai
>
> **摘要:** Instruct Text-to-Speech (InstructTTS) leverages natural language descriptions as style prompts to guide speech synthesis. However, existing InstructTTS methods mainly rely on a direct combination of audio-related labels or their diverse rephrasings, making it difficult to handle flexible, high-level instructions. Such rigid control is insufficient for users such as content creators who wish to steer generation with descriptive instructions. To address these constraints, we introduce OV-InstructTTS, a new paradigm for open-vocabulary InstructTTS. We propose a comprehensive solution comprising a newly curated dataset, OV-Speech, and a novel reasoning-driven framework. The OV-Speech dataset pairs speech with open-vocabulary instructions, each augmented with a reasoning process that connects high-level instructions to acoustic features. The reasoning-driven framework infers emotional, acoustic, and paralinguistic information from open-vocabulary instructions before synthesizing speech. Evaluations show that this reasoning-driven approach significantly improves instruction-following fidelity and speech expressiveness. We believe this work can inspire the next user-friendly InstructTTS systems with stronger generalization and real-world applicability. The dataset and demos are publicly available on our project page.
>
---
#### [new 011] IO-RAE: Information-Obfuscation Reversible Adversarial Example for Audio Privacy Protection
- **分类: cs.SD; cs.CR; cs.MM; eess.AS**

- **简介: 该论文属于音频隐私保护任务，旨在防止音频数据被非法分析。通过生成可逆对抗样本，有效混淆ASR系统，同时保持音频质量。**

- **链接: [https://arxiv.org/pdf/2601.01239v1](https://arxiv.org/pdf/2601.01239v1)**

> **作者:** Jiajie Zhu; Xia Du; Xiaoyuan Liu; Jizhe Zhou; Qizhen Xu; Zheng Lin; Chi-Man Pun
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** The rapid advancements in artificial intelligence have significantly accelerated the adoption of speech recognition technology, leading to its widespread integration across various applications. However, this surge in usage also highlights a critical issue: audio data is highly vulnerable to unauthorized exposure and analysis, posing significant privacy risks for businesses and individuals. This paper introduces an Information-Obfuscation Reversible Adversarial Example (IO-RAE) framework, the pioneering method designed to safeguard audio privacy using reversible adversarial examples. IO-RAE leverages large language models to generate misleading yet contextually coherent content, effectively preventing unauthorized eavesdropping by humans and Automatic Speech Recognition (ASR) systems. Additionally, we propose the Cumulative Signal Attack technique, which mitigates high-frequency noise and enhances attack efficacy by targeting low-frequency signals. Our approach ensures the protection of audio data without degrading its quality or our ability. Experimental evaluations demonstrate the superiority of our method, achieving a targeted misguidance rate of 96.5% and a remarkable 100% untargeted misguidance rate in obfuscating target keywords across multiple ASR models, including a commercial black-box system from Google. Furthermore, the quality of the recovered audio, measured by the Perceptual Evaluation of Speech Quality score, reached 4.45, comparable to high-quality original recordings. Notably, the recovered audio processed by ASR systems exhibited an error rate of 0%, indicating nearly lossless recovery. These results highlight the practical applicability and effectiveness of our IO-RAE framework in protecting sensitive audio privacy.
>
---
#### [new 012] SAFE-QAQ: End-to-End Slow-Thinking Audio-Text Fraud Detection via Reinforcement Learning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频欺诈检测任务，解决传统方法依赖文本转录导致的误差问题。提出SAFE-QAQ框架，通过音频直接分析实现高效、实时的欺诈检测。**

- **链接: [https://arxiv.org/pdf/2601.01392v1](https://arxiv.org/pdf/2601.01392v1)**

> **作者:** Peidong Wang; Zhiming Ma; Xin Dai; Yongkang Liu; Shi Feng; Xiaocui Yang; Wenxing Hu; Zhihao Wang; Mingjun Pan; Li Yuan; Daling Wang
>
> **摘要:** Existing fraud detection methods predominantly rely on transcribed text, suffering from ASR errors and missing crucial acoustic cues like vocal tone and environmental context. This limits their effectiveness against complex deceptive strategies. To address these challenges, we first propose \textbf{SAFE-QAQ}, an end-to-end comprehensive framework for audio-based slow-thinking fraud detection. First, the SAFE-QAQ framework eliminates the impact of transcription errors on detection performance. Secondly, we propose rule-based slow-thinking reward mechanisms that systematically guide the system to identify fraud-indicative patterns by accurately capturing fine-grained audio details, through hierarchical reasoning processes. Besides, our framework introduces a dynamic risk assessment framework during live calls, enabling early detection and prevention of fraud. Experiments on the TeleAntiFraud-Bench demonstrate that SAFE-QAQ achieves dramatic improvements over existing methods in multiple key dimensions, including accuracy, inference efficiency, and real-time processing capabilities. Currently deployed and analyzing over 70,000 calls daily, SAFE-QAQ effectively automates complex fraud detection, reducing human workload and financial losses. Code: https://anonymous.4open.science/r/SAFE-QAQ.
>
---
#### [new 013] MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转写任务，解决会议中精准识别说话人及时间戳的问题。提出MOSS Transcribe Diarize模型，实现端到端的说话人归属与时间标注。**

- **链接: [https://arxiv.org/pdf/2601.01554v1](https://arxiv.org/pdf/2601.01554v1)**

> **作者:** Donghua Yu; Zhengyuan Lin; Chen Yang; Yiyang Zhang; Zhaoye Fei; Hanfu Chen; Jingqi Chen; Ke Chen; Qinyuan Cheng; Liwei Fan; Yi Jiang; Jie Zhu; Muchen Li; Shimin Li; Wenxuan Wang; Yang Wang; Zhe Xu; Yitian Gong; Yuqian Zhang
>
> **摘要:** Speaker-Attributed, Time-Stamped Transcription (SATS) aims to transcribe what is said and to precisely determine the timing of each speaker, which is particularly valuable for meeting transcription. Existing SATS systems rarely adopt an end-to-end formulation and are further constrained by limited context windows, weak long-range speaker memory, and the inability to output timestamps. To address these limitations, we present MOSS Transcribe Diarize, a unified multimodal large language model that jointly performs Speaker-Attributed, Time-Stamped Transcription in an end-to-end paradigm. Trained on extensive real wild data and equipped with a 128k context window for up to 90-minute inputs, MOSS Transcribe Diarize scales well and generalizes robustly. Across comprehensive evaluations, it outperforms state-of-the-art commercial systems on multiple public and in-house benchmarks.
>
---
#### [new 014] A Mamba-Based Model for Automatic Chord Recognition
- **分类: cs.SD**

- **简介: 该论文属于音乐信息检索任务，旨在解决自动和弦识别问题。提出一种基于Mamba的模型BMACE，通过双向结构有效建模时间依赖，实现高效准确的和弦预测。**

- **链接: [https://arxiv.org/pdf/2601.02101v1](https://arxiv.org/pdf/2601.02101v1)**

> **作者:** Chunyu Yuan; Johanna Devaney
>
> **备注:** International Society of Music Information Retrieval, Late-Breaking Demo 2024
>
> **摘要:** In this work, we propose a new efficient solution, which is a Mamba-based model named BMACE (Bidirectional Mamba-based network, for Automatic Chord Estimation), which utilizes selective structured state-space models in a bidirectional Mamba layer to effectively model temporal dependencies. Our model achieves high prediction performance comparable to state-of-the-art models, with the advantage of requiring fewer parameters and lower computational resources
>
---
#### [new 015] Towards Prosodically Informed Mizo TTS without Explicit Tone Markings
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在为缺乏标注数据的Mizo语构建高质量TTS系统。通过对比Tacotron2和VITS模型，验证了非自回归框架的有效性。**

- **链接: [https://arxiv.org/pdf/2601.02073v1](https://arxiv.org/pdf/2601.02073v1)**

> **作者:** Abhijit Mohanta; Remruatpuii; Priyankoo Sarmah; Rohit Sinha; Wendy Lalhminghlui
>
> **摘要:** This paper reports on the development of a text-to-speech (TTS) system for Mizo, a low-resource, tonal, and Tibeto-Burman language spoken primarily in the Indian state of Mizoram. The TTS was built with only 5.18 hours of data; however, in terms of subjective and objective evaluations, the outputs were considered perceptually acceptable and intelligible. A baseline model using Tacotron2 was built, and then, with the same data, another TTS model was built with VITS. In both subjective and objective evaluations, the VITS model outperformed the Tacotron2 model. In terms of tone synthesis, the VITS model showed significantly lower tone errors than the Tacotron2 model. The paper demonstrates that a non-autoregressive, end-to-end framework can achieve synthesis of acceptable perceptual quality and intelligibility.
>
---
#### [new 016] Bayesian Negative Binomial Regression of Afrobeats Chart Persistence
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于回归分析任务，旨在探究合作歌曲在非洲节奏榜单上停留时间是否更长。通过贝叶斯负二项回归分析，控制流行度后发现合作歌曲停留时间略短。**

- **链接: [https://arxiv.org/pdf/2601.01391v1](https://arxiv.org/pdf/2601.01391v1)**

> **作者:** Ian Jacob Cabansag; Paul Ntegeka
>
> **摘要:** Afrobeats songs compete for attention on streaming platforms, where chart visibility can influence both revenue and cultural impact. This paper examines whether collaborations help songs remain on the charts longer, using daily Nigeria Spotify Top 200 data from 2024. Each track is summarized by the number of days it appears in the Top 200 during the year and its total annual streams in Nigeria. A Bayesian negative binomial regression is applied, with days on chart as the outcome and collaboration status (solo versus multi-artist) and log total streams as predictors. This approach is well suited for overdispersed count data and allows the effect of collaboration to be interpreted while controlling for overall popularity. Posterior inference is conducted using Markov chain Monte Carlo, and results are assessed using rate ratios, posterior probabilities, and predictive checks. The findings indicate that, after accounting for total streams, collaboration tracks tend to spend slightly fewer days on the chart than comparable solo tracks.
>
---
#### [new 017] Speak the Art: A Direct Speech to Image Generation Framework
- **分类: eess.AS; cs.AI; cs.MM**

- **简介: 该论文属于语音到图像生成任务，旨在解决现有方法在语义表示和生成稳定性上的不足。提出STA框架，结合语音编码与VQ-Diffusion网络，提升生成质量与多样性。**

- **链接: [https://arxiv.org/pdf/2601.00827v1](https://arxiv.org/pdf/2601.00827v1)**

> **作者:** Mariam Saeed; Manar Amr; Farida Adel; Nada Hassan; Nour Walid; Eman Mohamed; Mohamed Hussein; Marwan Torki
>
> **摘要:** Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called \textbf{Speak the Art (STA)} which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin.
>
---
#### [new 018] ARCADE: A City-Scale Corpus for Fine-Grained Arabic Dialect Tagging
- **分类: cs.CL; cs.CY; cs.SD**

- **简介: 该论文提出ARCADE数据集，用于城市级阿拉伯语方言识别。解决多方言语音与具体城市对应关系不明确的问题，通过收集和标注跨城市广播语音数据，支持细粒度方言分类任务。**

- **链接: [https://arxiv.org/pdf/2601.02209v1](https://arxiv.org/pdf/2601.02209v1)**

> **作者:** Omer Nacar; Serry Sibaee; Adel Ammar; Yasser Alhabashi; Nadia Samer Sibai; Yara Farouk Ahmed; Ahmed Saud Alqusaiyer; Sulieman Mahmoud AlMahmoud; Abdulrhman Mamdoh Mukhaniq; Lubaba Raed; Sulaiman Mohammed Alatwah; Waad Nasser Alqahtani; Yousif Abdulmajeed Alnasser; Mohamed Aziz Khadraoui; Wadii Boulila
>
> **摘要:** The Arabic language is characterized by a rich tapestry of regional dialects that differ substantially in phonetics and lexicon, reflecting the geographic and cultural diversity of its speakers. Despite the availability of many multi-dialect datasets, mapping speech to fine-grained dialect sources, such as cities, remains underexplored. We present ARCADE (Arabic Radio Corpus for Audio Dialect Evaluation), the first Arabic speech dataset designed explicitly with city-level dialect granularity. The corpus comprises Arabic radio speech collected from streaming services across the Arab world. Our data pipeline captures 30-second segments from verified radio streams, encompassing both Modern Standard Arabic (MSA) and diverse dialectal speech. To ensure reliability, each clip was annotated by one to three native Arabic reviewers who assigned rich metadata, including emotion, speech type, dialect category, and a validity flag for dialect identification tasks. The resulting corpus comprises 6,907 annotations and 3,790 unique audio segments spanning 58 cities across 19 countries. These fine-grained annotations enable robust multi-task learning, serving as a benchmark for city-level dialect tagging. We detail the data collection methodology, assess audio quality, and provide a comprehensive analysis of label distributions. The dataset is available on: https://huggingface.co/datasets/riotu-lab/ARCADE-full
>
---
#### [new 019] Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音转录分段任务，旨在生成多层级的目录结构。通过LoRA微调和高阶停顿特征，提升主题与子主题的分割效果。**

- **链接: [https://arxiv.org/pdf/2601.02128v1](https://arxiv.org/pdf/2601.02128v1)**

> **作者:** Steffen Freisinger; Philipp Seeberger; Thomas Ranzenberger; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Published in Proceedings of Interspeech 2025. Please cite the proceedings version (DOI: 10.21437/Interspeech.2025-2792)
>
> **摘要:** Segmenting speech transcripts into thematic sections benefits both downstream processing and users who depend on written text for accessibility. We introduce a novel approach to hierarchical topic segmentation in transcripts, generating multi-level tables of contents that capture both topic and subtopic boundaries. We compare zero-shot prompting and LoRA fine-tuning on large language models, while also exploring the integration of high-level speech pause features. Evaluations on English meeting recordings and multilingual lecture transcripts (Portuguese, German) show significant improvements over established topic segmentation baselines. Additionally, we adapt a common evaluation measure for multi-level segmentation, taking into account all hierarchical levels within one metric.
>
---
#### [new 020] HyperCLOVA X 8B Omni
- **分类: cs.LG; cs.AI; cs.CL; cs.SD**

- **简介: 该论文介绍HyperCLOVA X 8B Omni，一款支持文本、音频和视觉输入输出的多模态模型，解决跨模态交互问题，通过统一接口实现多模态理解和生成。**

- **链接: [https://arxiv.org/pdf/2601.01792v1](https://arxiv.org/pdf/2601.01792v1)**

> **作者:** NAVER Cloud HyperCLOVA X Team
>
> **备注:** Technical Report
>
> **摘要:** In this report, we present HyperCLOVA X 8B Omni, the first any-to-any omnimodal model in the HyperCLOVA X family that supports text, audio, and vision as both inputs and outputs. By consolidating multimodal understanding and generation into a single model rather than separate modality-specific pipelines, HyperCLOVA X 8B Omni serves as an 8B-scale omni-pathfinding point toward practical any-to-any omni assistants. At a high level, the model unifies modalities through a shared next-token prediction interface over an interleaved multimodal sequence, while vision and audio encoders inject continuous embeddings for fine-grained understanding and grounding. Empirical evaluations demonstrate competitive performance against comparably sized models across diverse input-output combinations spanning text, audio, and vision, in both Korean and English. We anticipate that the open-weight release of HyperCLOVA X 8B Omni will support a wide range of research and deployment scenarios.
>
---
#### [new 021] Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言对话语音识别任务，旨在解决LLM与端到端模型性能差距问题。通过融合语音编码器与LLM提升识别效果，提出交叉注意力融合机制，取得良好结果。**

- **链接: [https://arxiv.org/pdf/2601.01461v1](https://arxiv.org/pdf/2601.01461v1)**

> **作者:** Yuxiang Mei; Dongxing Xu; Jiaen Liang; Yanhua Long
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
>
---
## 更新

#### [replaced 001] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究多轮对话中语音模型的风格退化问题，属于自然语言处理任务。旨在解决模型无法持续保持指定说话风格的问题，并探索缓解方法。**

- **链接: [https://arxiv.org/pdf/2512.23578v2](https://arxiv.org/pdf/2512.23578v2)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** Submitted to ACL ARR January 2026
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that when SLMs are asked to recall the style instruction in later turns, they can recall the style instruction, but they fail to express it throughout the conversation. We also show that explicitly asking the model to recall the style instruction can partially mitigate style amnesia. In addition, we examine various prompting strategies and find that SLMs struggle to follow the required style when the instruction is placed in system messages rather than user messages, which contradicts the intended function of system prompts.
>
---
#### [replaced 002] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音处理任务，旨在解决说话人日志与识别（SDR）问题。提出SpeakerLM模型，实现端到端的SDR，并引入灵活的说话人注册机制，提升模型泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.06372v3](https://arxiv.org/pdf/2508.06372v3)**

> **作者:** Han Yin; Yafeng Chen; Chong Deng; Luyao Cheng; Hui Wang; Chao-Hong Tan; Qian Chen; Wen Wang; Xiangang Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** The Speaker Diarization and Recognition (SDR) task aims to predict "who spoke when and what" within an audio clip, which is a crucial task in various real-world multi-speaker scenarios such as meeting transcription and dialogue systems. Existing SDR systems typically adopt a cascaded framework, combining multiple modules such as speaker diarization (SD) and automatic speech recognition (ASR). The cascaded systems suffer from several limitations, such as error propagation, difficulty in handling overlapping speech, and lack of joint optimization for exploring the synergy between SD and ASR tasks. To address these limitations, we introduce SpeakerLM, a unified multimodal large language model for SDR that jointly performs SD and ASR in an end-to-end manner. Moreover, to facilitate diverse real-world scenarios, we incorporate a flexible speaker registration mechanism into SpeakerLM, enabling SDR under different speaker registration settings. SpeakerLM is progressively developed with a multi-stage training strategy on large-scale real data. Extensive experiments show that SpeakerLM demonstrates strong data scaling capability and generalizability, outperforming state-of-the-art cascaded baselines on both in-domain and out-of-domain public SDR benchmarks. Furthermore, experimental results show that the proposed speaker registration mechanism effectively ensures robust SDR performance of SpeakerLM across diverse speaker registration conditions and varying numbers of registered speakers.
>
---
#### [replaced 003] On the social bias of speech self-supervised models
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨自监督学习模型中的社会偏见问题，分析其成因并尝试通过模型压缩等方法进行去偏。**

- **链接: [https://arxiv.org/pdf/2406.04997v2](https://arxiv.org/pdf/2406.04997v2)**

> **作者:** Yi-Cheng Lin; Tzu-Quan Lin; Hsi-Che Lin; Andy T. Liu; Hung-yi Lee
>
> **备注:** Accepted by INTERSPEECH 2024, best paper runner-up for the special session "Responsible Speech Foundation Models"
>
> **摘要:** Self-supervised learning (SSL) speech models have achieved remarkable performance in various tasks, yet the biased outcomes, especially affecting marginalized groups, raise significant concerns. Social bias refers to the phenomenon where algorithms potentially amplify disparate properties between social groups present in the data used for training. Bias in SSL models can perpetuate injustice by automating discriminatory patterns and reinforcing inequitable systems. This work reveals that prevalent SSL models inadvertently acquire biased associations. We probe how various factors, such as model architecture, size, and training methodologies, influence the propagation of social bias within these models. Finally, we explore the efficacy of debiasing SSL models through regularization techniques, specifically via model compression. Our findings reveal that employing techniques such as row-pruning and training wider, shallower models can effectively mitigate social bias within SSL model.
>
---
#### [replaced 004] Perch 2.0: The Bittern Lesson for Bioacoustics
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文介绍Perch 2.0，一个用于生物声学的预训练模型，解决多物种分类与迁移学习问题。扩展至多类物种训练，提升性能并验证其在海洋任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2508.04665v2](https://arxiv.org/pdf/2508.04665v2)**

> **作者:** Bart van Merriënboer; Vincent Dumoulin; Jenny Hamer; Lauren Harrell; Andrea Burns; Tom Denton
>
> **摘要:** Perch is a performant pre-trained model for bioacoustics. It was trained in supervised fashion, providing both off-the-shelf classification scores for thousands of vocalizing species as well as strong embeddings for transfer learning. In this new release, Perch 2.0, we expand from training exclusively on avian species to a large multi-taxa dataset. The model is trained with self-distillation using a prototype-learning classifier as well as a new source-prediction training criterion. Perch 2.0 obtains state-of-the-art performance on the BirdSet and BEANS benchmarks. It also outperforms specialized marine models on marine transfer learning tasks, despite having almost no marine training data. We present hypotheses as to why fine-grained species classification is a particularly robust pre-training task for bioacoustics.
>
---
#### [replaced 005] DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，解决标准编解码器中音色与韵律纠缠的问题。提出DisCo-Speech框架，通过分离音色、韵律和内容，实现零样本可控语音生成。**

- **链接: [https://arxiv.org/pdf/2512.13251v3](https://arxiv.org/pdf/2512.13251v3)**

> **作者:** Tao Li; Wenshuo Ge; Zhichao Wang; Zihao Cui; Yong Ma; Yingying Gao; Chao Deng; Shilei Zhang; Junlan Feng
>
> **备注:** Updated with 6,000 hours of additional training data and improved performance. Expanded appendix with ablation studies, training objectives, and hyperparameter settings for better reproducibility. Audio and code links included
>
> **摘要:** Codec-based language models (LMs) have revolutionized text-to-speech (TTS). However, standard codecs entangle timbre and prosody, which hinders independent control in continuation-based LMs. To tackle this challenge, we propose DisCo-Speech, a zero-shot controllable TTS framework featuring a disentangled speech codec (DisCodec) and an LM-based generator. The core component DisCodec employs a two-stage design: 1) tri-factor disentanglement to separate speech into content, prosody, and timbre subspaces via parallel encoders and hybrid losses; and 2) fusion and reconstruction that merges content and prosody into unified content-prosody tokens suitable for LM prediction, while jointly optimizing reconstruction to address the disentanglement-reconstruction trade-off. This allows the LM to perform prosodic continuation from a style prompt while the decoder injects target timbre, enabling flexible zero-shot control. Experiments demonstrate that DisCo-Speech achieves competitive voice cloning and superior zero-shot prosody control. By resolving the core entanglement at the codec level, DisCo-Speech provides a robust foundation for controllable speech synthesis.
>
---
#### [replaced 006] AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives
- **分类: cs.SD; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于音频-语言模型任务，旨在解决模型生成内容与音频输入不一致的幻觉问题。通过构建高质量数据集和评估基准，提升模型的时序推理能力。**

- **链接: [https://arxiv.org/pdf/2512.24052v2](https://arxiv.org/pdf/2512.24052v2)**

> **作者:** Yanxi Chen; Wenhui Zhu; Xiwen Chen; Zhipeng Wang; Xin Li; Peijie Qiu; Hao Wang; Xuanzhao Dong; Yujian Xiong; Anderson Schneider; Yuriy Nevmyvaka; Yalin Wang
>
> **摘要:** Although Large Audio-Language Models (LALMs) deliver state-of-the-art (SOTA) performance, they frequently suffer from hallucinations, e.g. generating text not grounded in the audio input. We analyze these grounding failures and identify a distinct taxonomy: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error. To address this, we introduce the AHA (Audio Hallucination Alignment) framework. By leveraging counterfactual hard negative mining, our pipeline constructs a high-quality preference dataset that forces models to distinguish strict acoustic evidence from linguistically plausible fabrications. Additionally, we establish AHA-Eval, a diagnostic benchmark designed to rigorously test these fine-grained temporal reasoning capabilities. We apply this data to align Qwen2.5-Omni. The resulting model, Qwen-Audio-AHA, achieves a 13.7% improvement on AHA-Eval. Crucially, this benefit generalizes beyond our diagnostic set. Our model shows substantial gains on public benchmarks, including 1.3% on MMAU-Test and 1.6% on MMAR, outperforming latest SOTA methods. The model and dataset are open-sourced at https://github.com/LLM-VLM-GSL/AHA.
>
---
#### [replaced 007] Towards Practical Automatic Piano Reduction using BERT with Semi-supervised Learning
- **分类: cs.SD; cs.SC**

- **简介: 该论文属于自动钢琴简化任务，旨在解决手工处理耗时的问题。通过半监督学习和MidiBERT框架，提出两种解决方案，实现高效准确的钢琴简化。**

- **链接: [https://arxiv.org/pdf/2512.21324v2](https://arxiv.org/pdf/2512.21324v2)**

> **作者:** Wan Ki Wong; Ka Ho To; Chuck-jee Chau; Lucas Wong; Kevin Y. Yip; Irwin King
>
> **摘要:** In this study, we present a novel automatic piano reduction method with semi-supervised machine learning. Piano reduction is an important music transformation process, which helps musicians and composers as a musical sketch for performances and analysis. The automation of such is a highly challenging research problem but could bring huge conveniences as manually doing a piano reduction takes a lot of time and effort. While supervised machine learning is often a useful tool for learning input-output mappings, it is difficult to obtain a large quantity of labelled data. We aim to solve this problem by utilizing semi-supervised learning, so that the abundant available data in classical music can be leveraged to perform the task with little or no labelling effort. In this regard, we formulate a two-step approach of music simplification followed by harmonization. We further propose and implement two possible solutions making use of an existing machine learning framework -- MidiBERT. We show that our solutions can output practical and realistic samples with an accurate reduction that needs only small adjustments in post-processing. Our study forms the groundwork for the use of semi-supervised learning in automatic piano reduction, where future researchers can take reference to produce more state-of-the-art results.
>
---
#### [replaced 008] SAMUeL: Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐生成任务，旨在解决现有系统参数庞大、推理慢的问题。提出SAMUeL模型，通过软对齐注意力和潜在扩散实现高效伴奏生成。**

- **链接: [https://arxiv.org/pdf/2507.19991v3](https://arxiv.org/pdf/2507.19991v3)**

> **作者:** Hei Shing Cheung; Boya Zhang; Jonathan H. Chan
>
> **备注:** 7 pages, 3 figures, accepted to IEEE/WIC WI-IAT
>
> **摘要:** We present a lightweight latent diffusion model for vocal-conditioned musical accompaniment generation that addresses critical limitations in existing music AI systems. Our approach introduces a novel soft alignment attention mechanism that adaptively combines local and global temporal dependencies based on diffusion timesteps, enabling efficient capture of multi-scale musical structure. Operating in the compressed latent space of a pre-trained variational autoencoder, the model achieves a 220 times parameter reduction compared to state-of-the-art systems while delivering 52 times faster inference. Experimental evaluation demonstrates competitive performance with only 15M parameters, outperforming OpenAI Jukebox in production quality and content unity while maintaining reasonable musical coherence. The ultra-lightweight architecture enables real-time deployment on consumer hardware, making AI-assisted music creation accessible for interactive applications and resource-constrained environments.
>
---
#### [replaced 009] pyAMPACT: A Score-Audio Alignment Toolkit for Performance Data Estimation and Multi-modal Processing
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出pyAMPACT工具，用于音乐表演数据的音符对齐与多模态分析，解决符号音乐与音频之间的关联问题。**

- **链接: [https://arxiv.org/pdf/2412.05436v2](https://arxiv.org/pdf/2412.05436v2)**

> **作者:** Johanna Devaney; Daniel McKemie; Alex Morgan
>
> **备注:** Proceedings of the 2025 International Computer Music Conference
>
> **摘要:** pyAMPACT (Python-based Automatic Music Performance Analysis and Comparison Toolkit) links symbolic and audio music representations to facilitate score-informed estimation of performance data in audio as well as general linking of symbolic and audio music representations with a variety of annotations. pyAMPACT can read a range of symbolic formats and can output note-linked audio descriptors/performance data into MEI-formatted files. The audio analysis uses score alignment to calculate time-frequency regions of importance for each note in the symbolic representation from which to estimate a range of parameters. These include tuning-, dynamics-, and timbre-related performance descriptors, with timing-related information available from the score alignment. Beyond performance data estimation, pyAMPACT also facilitates multi-modal investigations through its infrastructure for linking symbolic representations and annotations to audio.
>
---
#### [replaced 010] Generating Piano Music with Transformers: A Comparative Study of Scale, Data, and Metrics
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于符号钢琴音乐生成任务，旨在研究设计选择对生成音乐质量的影响。通过比较数据集、模型架构等，提升生成音乐的品质。**

- **链接: [https://arxiv.org/pdf/2511.07268v2](https://arxiv.org/pdf/2511.07268v2)**

> **作者:** Jonathan Lehmkuhl; Ábel Ilyés-Kun; Nico Bremes; Cemhan Kaan Özaltan; Frederik Muthers; Jiayi Yuan
>
> **备注:** NeurIPS 2025 Workshop on AI for Music
>
> **摘要:** Although a variety of transformers have been proposed for symbolic music generation in recent years, there is still little comprehensive study on how specific design choices affect the quality of the generated music. In this work, we systematically compare different datasets, model architectures, model sizes, and training strategies for the task of symbolic piano music generation. To support model development and evaluation, we examine a range of quantitative metrics and analyze how well they correlate with human judgment collected through listening studies. Our best-performing model, a 950M-parameter transformer trained on 80K MIDI files from diverse genres, produces outputs that are often rated as human-composed in a Turing-style listening survey.
>
---
#### [replaced 011] CMDAR: A Chinese Multi-scene Dynamic Audio Reasoning Benchmark with Diverse Challenges
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出CMDAR基准，用于评估AI在多场景动态音频推理任务中的能力，解决现有数据单一、场景简单的问题。**

- **链接: [https://arxiv.org/pdf/2509.22461v2](https://arxiv.org/pdf/2509.22461v2)**

> **作者:** Hui Li; Changhao Jiang; Hongyu Wang; Ming Zhang; Jiajun Sun; Zhixiong Yang; Yifei Cao; Shihan Dou; Xiaoran Fan; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** The ability to reason from audio, including speech, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and English audio data and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce CMDAR, a Chinese benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. CMDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on CMDAR and observe that they exhibit limitations in complex reasoning tasks. In CMDAR-main, Qwen2.5-Omni achieves 76.67% accuracy, whereas GPT-4o Audio reaches 68.47%. However, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice with multiple audios and open-ended tasks. And we provide detail analysis corresponding suggestions for the future development of large audio language models.
>
---
