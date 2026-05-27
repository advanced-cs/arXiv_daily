# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，旨在降低高质量TTS系统的数据与资源需求。通过轻量架构和开源数据处理流程，实现高效合成与多种语音风格生成。**

- **链接: [https://arxiv.org/pdf/2605.27258](https://arxiv.org/pdf/2605.27258)**

> **作者:** Bowen Li; Shaotong Guo; Zhen Wang; Yang Xiang; Mingli Jin; Yihang Lin; Jiahui Zhao; Weibo Xiong; Dongrui Li; Keming Chen; Yunze Gao; Yuze Zhou; Zeyang Lin; Yue Liu
>
> **摘要:** Building state-of-the-art text-to-speech (TTS) systems typically demands millions of hours of proprietary data and complex multi-stage architectures, creating substantial barriers for resource-constrained research teams. In this report, we present PilotTTS, a lightweight autoregressive TTS system that achieves competitive performance through minimalist architecture and rigorous data engineering. PilotTTS is trained on only 200K hours of data processed entirely with open-source tools. Specifically, our contributions are: (1) a reproducible multi-stage data processing pipeline covering quality assessment, label annotation, and filtering, and (2) a compact model architecture that employs Q-Former-based conditioning to decouple speaker identity from speaking style via cross-sample paired training. Within a unified framework, PilotTTS supports zero-shot voice cloning, emotion synthesis (11 categories), paralinguistic synthesis (4 categories), and Chinese dialect synthesis (14 dialects). On the Seed-TTS Eval benchmark, PilotTTS achieves the lowest WER of 1.50% on test-en, a CER of 0.87% on test-zh, and the highest speaker similarity on both test sets (0.862 and 0.815), outperforming systems trained on significantly larger datasets. We release the complete data pipeline recipe, pretrained weights, and code at this https URL.
>
---
#### [new 002] CFMDCTCodec: A Low-Bitrate Neural Speech Codec with Noise-Prior-aware Conditional Flow Matching for MDCT-Spectral Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决低比特率下语音质量下降的问题。提出CFMDCTCodec，结合噪声先验和流匹配技术提升频谱细节，实现高质量低码率语音编码。**

- **链接: [https://arxiv.org/pdf/2605.26812](https://arxiv.org/pdf/2605.26812)**

> **作者:** Xiao-Hang Jiang; Yang Ai; Hui-Peng Du; Zhen-Hua Ling; Ji Wu
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** High-quality speech coding at low bitrates is crucial for bandwidth-constrained applications, yet remains challenging due to the severe loss of quality-critical information in highly compressed representations. To overcome this challenge, we propose CFMDCTCodec, a low-bitrate neural speech codec that operates entirely in the modified discrete cosine transform (MDCT) domain. CFMDCTCodec integrates a lightweight encoder-quantizer-decoder-style MDCT-spectral codec with a noise-prior-aware, conditional-flow-matching (CFM)-based MDCT-spectral enhancer. Within this framework, the codec serves as a base module that compactly discretizes the MDCT spectrum extracted from speech and produces an initial coarse reconstruction, while the enhancer further restores fine-grained spectral details. The enhancer improves the decoded MDCT spectrum by integrating a conditional MDCT velocity-field filter with an ordinary differential equation (ODE) solver, under the guidance of an MDCT-derived magnitude-adaptive noise prior, aiming to emphasize perceptually significant high-energy regions while stabilizing low-energy and silent regions. Finally, the enhanced MDCT spectrum is reconstructed into the decoded speech using the inverse MDCT. When optimizing CFMDCTCodec, we adopt a unified non-adversarial training strategy that jointly combines reconstruction, quantization and CFM objectives. Both objective and subjective evaluations show that CFMDCTCodec outperforms competitive baselines in low-bitrate regimes, e.g., 0.65 kbps, while approaching the perceptual quality of large-scale codecs with significantly fewer parameters and computations.
>
---
#### [new 003] MERIT: Learning Disentangled Music Representations for Audio Similarity
- **分类: cs.SD**

- **简介: 该论文属于音乐表示学习任务，旨在解决现有模型无法分离音乐维度的问题。通过MERIT框架，学习到独立的旋律、节奏和音色表示，提升相似性计算的可解释性与控制性。**

- **链接: [https://arxiv.org/pdf/2605.27346](https://arxiv.org/pdf/2605.27346)**

> **作者:** Abhinaba Roy; Junyi Liang; Dorien Herremans
>
> **摘要:** Current music similarity models typically compute a single, monolithic score, entangling distinct musical dimensions like melody, rhythm, and timbre. This limits user control and interpretability, making it impossible to execute nuanced queries. We introduce MERIT, a framework for learning disentangled, factor-specific music representations tailored to these three core dimensions. To overcome the lack of isolated musical variations in real-world audio, we use a novel training strategy that uses conditional audio generation and source-separated stems to strongly encourage single-factor variation in training data. Our evaluations demonstrate strong factor-wise disentanglement. Each head responds strongly to its intended perceptual dimension while remaining near chance on the others, a representational property that holds across both the synthetic training domain and independent real-world audio.
>
---
#### [new 004] An investigation of AI integration in sound designer workflows and experiences
- **分类: cs.SD; cs.AI; cs.CY**

- **简介: 论文研究AI在声音设计工作流程中的应用，解决工具与需求不匹配的问题。通过调查和访谈分析，提出五项主题并给出开发建议。**

- **链接: [https://arxiv.org/pdf/2605.27174](https://arxiv.org/pdf/2605.27174)**

> **作者:** Nelly Garcia; Joshua Reiss
>
> **摘要:** Artificial intelligence is increasingly being integrated into professional audio production workflows, yet a gap persists between the tools developers produce and the requirements of practising sound designers. This paper investigates this gap through a mixed-methods study comprising a survey of 76 practitioners and follow-up semi-structured interviews with 20 industry professionals. Results were analysed using descriptive statistical analysis and thematic analysis to identify patterns across both datasets. Five themes emerged from our analysis: Context, Workflow, Potential, Risks, and Right Use. Our work indicates that current AI tools perform adequately in fast-consumption media contexts but lack the narrative sophistication required for high-end sound design (films, immersive experiences etc). Practitioners demonstrate a preference for assistive, task-specific applications, particularly in audio restoration and library management, over end-to-end generative systems. This work contributes to the on-going discussion on the use of AI and AI-enhanced tools in the creative industries. We report on the current status of the field from the point of view of sound designers and creative audio practitioners, and offer a set of recommendation for sound technologist and developers based on our findings to guide the development of more informed AI tools for sound design.
>
---
#### [new 005] Eroding Trust in Real Speech: A Large-Scale Study of Human Audio Deepfake Perception
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频深度伪造感知研究，旨在探讨深度伪造对人类真实语音信任的影响。通过大规模实验发现，人们更不信任真实语音，而非更难识别伪造音频。**

- **链接: [https://arxiv.org/pdf/2605.26136](https://arxiv.org/pdf/2605.26136)**

> **作者:** Nicolas M. Müller; Wei Herng Choong
>
> **摘要:** Audio deepfakes have improved rapidly recently, yet their effect on human trust in real speech remains unstudied. We present the largest listening study on audio deepfake perception to date, collecting 35,532 judgments from 1,768 participants across 138 text-to-speech and voice conversion systems. Our central finding is a skepticism shift: compared to a 2021 baseline, human accuracy on fake samples barely changed (72.9% to 71.2%), but accuracy on real samples dropped from 72.7% to 64.1%. Participants are not worse at detecting synthesis artifacts; rather, they increasingly distrust authentic speech. Samples generated by commercial and autoregressive language model systems proved hardest to detect (61.3 - 65.9%), while those from traditional seq2seq and flow-matching models remain easier to spot (75.4 - 76.8%). An ML detector that served as a reference point maintained over 94.5% accuracy across all conditions. Our results suggest that the primary threat posed by modern deepfakes may not be mere deception, but the erosion of trust in genuine audio.
>
---
#### [new 006] PitchBench: Measuring Pitch Hearing in Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出PitchBench，用于评估音频语言模型的音高感知能力。针对当前模型在音高识别上的不可靠性，通过多种实验测试其在不同声学条件下的表现。**

- **链接: [https://arxiv.org/pdf/2605.26176](https://arxiv.org/pdf/2605.26176)**

> **作者:** Milan Liessens Dujardin; Song-Ze Yu; Craver Corbyn Thomas-Smith; David M. Chan; Karina Nguyen
>
> **备注:** Preprint
>
> **摘要:** Audio-language models (ALMs) are increasingly used in real-world applications that require understanding music, from music tutoring and transcription to captioning, recommendation systems, and music production. More broadly, they are becoming an important component of multimodal AI systems that must reason from sensory input rather than text alone. This makes reliable musical perception a critical prerequisite: if a model cannot accurately hear the structure of sound, it cannot be trusted to reason about, teach, transcribe, or act on audio in the real world. Yet existing benchmarks rarely assess one of the most fundamental musical abilities underlying such perception: pitch hearing. Current evaluations tend to probe pitch hearing only indirectly, through higher-level tasks and often in multiple-choice formats, leaving open how reliably ALMs identify fine-grained pitch across instruments, acoustic conditions, and response formats. We introduce PitchBench, an evaluation suite that systematically measures pitch hearing in ALMs. PitchBench comprises 28 experiments spanning absolute and relative pitch perception within sequences and chords, while varying loudness, note duration, sound source, time stretching, background noise, and other acoustic conditions. Tasks range from identifying individual pitches in isolation to tracking a melodic line within a four-part musical texture. Evaluating frontier ALMs, we find that pitch hearing remains highly unreliable: models perform consistently poorly across settings, with accuracy varying sharply by sound source, note duration, and notation format. Current ALMs do not yet possess stable pitch perception, even for controlled synthetic and instrumental stimuli. Alongside the benchmark, we release PitchBench as a Python package containing the evaluation data and data generation tools to support future work on pitch-aware audio-language modeling.
>
---
#### [new 007] Why Can't They Remember? Uncovering Representation and Retrieval Bottlenecks in Multi-Turn Acoustic Memory
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多轮语音记忆任务，旨在解决大音频语言模型在多轮交互中难以保留非语音信息的问题。通过构建EnvMem基准，分析表征与检索瓶颈，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2605.27039](https://arxiv.org/pdf/2605.27039)**

> **作者:** Yang Xiao; Siyi Wang; Han Yin; Hong Jia; Vidhyasaharan Sethu; Eun-Jung Holden; Ting Dang
>
> **摘要:** Large audio language models (LALMs) process both speech and environmental acoustic cues, yet struggle to retain non-speech information across multi-turn interactions. The performance gap between semantic (speech) and acoustic (non-speech) understanding remains poorly understood, and the underlying mechanisms of representation and retrieval are still unclear. This work introduces EnvMem, a controlled multi-turn benchmark designed to study this gap and identify the root causes of failures at the representation (i.e., latent embeddings) and retrieval levels (i.e., attention allocation). We further conduct post-hoc interventions to probe representational structure and attention dynamics. Our results reveal representational trajectory drift as the key failure mode, while showing that attention allocation plays a limited role in explaining the observed degradation. Overall, we provide a systematic framework for analyzing and improving non-linguistic memory in long-context LALMs, shedding light on future data and training design for robust acoustic memory modeling.
>
---
#### [new 008] LongAV-Compass: Towards Unified Evaluation of Minute-Scale Audio-Visual Generation Across T2AV, I2AV, and V2AV
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于音频视频生成任务，旨在解决长时序、多模态生成的评估问题。提出LongAV-Compass基准，支持T2AV、I2AV和V2AV的统一评估。**

- **链接: [https://arxiv.org/pdf/2605.26244](https://arxiv.org/pdf/2605.26244)**

> **作者:** Tengfei Liu; Yang Shi; Xuanyu Zhu; Jiafu Tang; Liu Yang; Qixun Wang; Zhuoran Zhang; Yuqi Tang; Fengxiang Wang; Yuhao Dong; Xinlong Chen; Bozhou Li; Bohan Zeng; Yue Ding; Xiaohan Zhang; Jialu Chen; Haotian Wang; Yuanxing Zhang; Pengfei Wan; Leye Wang
>
> **摘要:** Audio-visual generation is rapidly advancing from short clips to minute-long content, while existing evaluation protocols remain largely confined to short-form settings. Existing benchmarks primarily focus on 5--10 second text-conditioned generation and rarely support unified evaluation across text, image, and video conditioning modalities. Moreover, they provide limited insight into how identity consistency, narrative coherence, and audio-visual alignment degrade over extended temporal horizons. To bridge this gap, we introduce LongAV-Compass, a systematic benchmark for minute-long audio-visual generation. LongAV-Compass contains 284 curated test cases spanning text-to-audio-video (T2AV), image-to-audio-video (I2AV), and video-to-audio-video (V2AV), organized by application scenario and generation complexity. The benchmark combines taxonomy-guided benchmark construction with a unified evaluation framework that integrates MLLM-assisted assessment with complementary perceptual and multimodal metrics, including DINO-v2, ArcFace, CLIP, and ImageBind. The framework evaluates more than 20 fine-grained dimensions covering within-segment quality, cross-segment consistency, global narrative coherence, semantic alignment, and audio-visual synchronization. Through experiments on 11 representative models together with human-alignment validation, LongAV-Compass provides a diagnostic testbed for analyzing the limitations of current systems in sustaining coherent, semantically aligned, and temporally consistent minute-scale audio-visual generation across diverse input modalities.
>
---
#### [new 009] Learning When to Think While Listening in Large Audio-Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于语音问答任务，解决实时交互中推理时机与响应延迟的平衡问题。通过学习控制等待、思考和回答的时机，提升问答质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.27190](https://arxiv.org/pdf/2605.27190)**

> **作者:** Zhiyuan Song; Weici Zhao; Yang Xiao; Suhao Yu; Cheng Zhu; Jiatao Gu
>
> **备注:** 19 pages, 4 figures, 6 tables
>
> **摘要:** Recent advances in Large Audio-Language Models (LALMs) have made real-time, streaming spoken interaction increasingly practical. In this setting, reasoning quality and responsiveness are tightly coupled: delaying reasoning until the speech endpoint can improve answer quality but moves deliberation into user-visible response delay, while answering too early risks committing before decisive evidence arrives. We introduce a learnable wait-think-answer control formulation for LALMs. Motivated by the incremental nature of human conversation, the controller decides under partial audio evidence when to wait, when to externalize a compact reasoning update, and when to answer. Using Qwen2.5-Omni-7B as the base model, we construct aligned wait-think-answer traces from spoken reasoning data, train the controller with supervised fine-tuning (SFT), and then apply Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO). The reward combines answer correctness, action validity, update timing, latency synchronization, reasoning quality, and chain consistency, optimizing the complete wait-think-answer trajectory and not the final answer alone. On a six-task synthetic spoken reasoning question answering (SRQA) benchmark, the six-reward DAPO controller improves the row-weighted accuracy from 67.6% to 70.3% while reducing post-endpoint final-think length by 14% under the same Qwen deployment harness. On a 186-item human-recorded Real Audio Bench, a transfer check beyond text-to-speech (TTS)-rendered speech, the controller family remains functional: SFT achieves the strongest accuracy, while the six-reward DAPO controller is the only learned variant whose final-think length falls below the base. These results suggest that a streaming model should learn when to make intermediate reasoning explicit during the audio stream.
>
---
#### [new 010] PashtoTTS-Bench: automated screening for low-resource non-Latin-script text-to-speech
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于低资源非拉丁文字母语音合成评估任务，解决传统评估方法失效问题，提出INSV-A框架并构建PashtoTTS-Bench基准。**

- **链接: [https://arxiv.org/pdf/2605.26978](https://arxiv.org/pdf/2605.26978)**

> **作者:** Hanif Rahman
>
> **摘要:** Text-to-speech (TTS) evaluation for low-resource non-Latin-script languages can fail when it relies on a single ASR round-trip word error rate (WER). A system may produce no audio, speak a neighbouring language, preserve target script text only in an ASR transcript, or sound unnatural to native listeners. We introduce INSV (Intelligibility, Naturalness, Script fidelity, and Verification), a reporting framework that separates these cases. This paper reports INSV-A, the automated screening subset: synthesis completion, ASR WER/CER, transcript Script Fidelity Rate, and audio language identification. Native MOS and phonetic annotation are specified but not claimed in this release. We instantiate INSV-A as PashtoTTS-Bench, a dated benchmark for Pashto TTS. The April-May 2026 run evaluates Edge GulNawaz, Edge Latifa, OmniVoice clone, OmniVoice auto, and an Urdu negative control on 200 FLEURS and 200 filtered Common Voice 24 prompts. Under the independent omniASR_CTC_300M_v2, OmniVoice auto has the lowest WER (24.1% FLEURS, 27.4% CV24), followed by Edge GulNawaz (32.8%, 39.5%), Edge Latifa (35.6%, 47.7%), and OmniVoice clone (45.4%, 34.8%). WER below the natural-speech baseline reflects clean synthetic audio and should not be read as better than native speech. Whisper Large V3 returns 0.0% Pashto labels on checked Pashto TTS audio, while MMS-LID-4017 and SpeechBrain VoxLingua107 separate Pashto outputs from the Urdu control. The release provides provider metadata, per-sentence scores, LID audits, failure logs, and scripts for adding systems.
>
---
#### [new 011] DuoGesture: Neuro-Inspired and Biomechanically Informed Dual-Stream Co-Speech Gesture Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文属于语音驱动手势生成任务，旨在解决语义表达与生物力学合理性之间的矛盾。提出DuoGesture模型，通过双流架构分离语义和节奏手势，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26236](https://arxiv.org/pdf/2605.26236)**

> **作者:** Ferdinand Paar; Lanmiao Liu; Aslı Özyürek; Serge Thill; Esam Ghaleb
>
> **摘要:** Co-speech gesture generation requires both semantic expressivity and biomechanically plausible rhythmic motion. Existing holistic gesture models mix lexically grounded semantic gestures with frequent prosody-aligned beat gestures. This limits semantic grounding, speech-motion alignment, and kinematic smoothness. We propose \emph{DuoGesture}, a neuro-inspired and biomechanically informed dual-stream approach that decomposes co-speech gesture synthesis into coupled semantic and beat streams. The two streams are coordinated by a \emph{Semantic Variational Information Bottleneck}, a stochastic frame-level gate that learns when semantic gestures should override rhythmic beat motion. The semantic stream is controlled by \emph{Motion-Grounded Semantic Conditioning}, which replaces purely linguistic word embeddings with motion-language representations to provide motion-aligned semantic priors for long-tailed lexical triggers of gestures. The beat stream is further regularised by an \emph{Inertial Beat Prior}, an anthropometry-weighted arm-chain module that reduces jitter and improves rhythmic consistency without constraining semantic frames. Objective evaluations and subjective experiments show that DuoGesture outperforms strong holistic baselines, while component ablations confirm the complementary roles of semantic grounding, stochastic stream selection, and biomechanical regularisation.
>
---
#### [new 012] Can We Hear from Events? Generating Speech from Event Camera
- **分类: cs.MM; cs.SD**

- **简介: 该论文属于语音生成任务，解决传统方法因时间粒度不匹配导致的情感信息丢失问题。通过引入事件相机数据，提出EventSpeech框架，提升语音表达的准确性与清晰度。**

- **链接: [https://arxiv.org/pdf/2605.26672](https://arxiv.org/pdf/2605.26672)**

> **作者:** Jingping Fang; Lin Chen; Chenyang Xu; Tong Zhao; Weidong Cai; Xiaoming Chen
>
> **摘要:** Traditional RGB-based speech generation faces Temporal Granularity Mismatch since fixed camera exposure times inevitably blur the high-frequency articulatory transients essential for rendering emotional speech. To break this ceiling, we propose EventSpeech as a novel text-conditioned framework pioneering the use of neuromorphic events for expressive speech generation, since these microsecond-precise events naturally align with acoustic waveform dynamics. Our architecture integrates a dedicated Event Encoder to model sparse neuromorphic events alongside a multi-scale Audio Encoder featuring a Hierarchical Wavelet Contextualizer (HWC). A bidirectional alignment mechanism seamlessly synchronizes linguistic content and visual dynamics with dense acoustic features. Furthermore, we construct EVT-SPK as the first benchmark comprising large-scale synthetic data and real-world recordings from specialized neuromorphic hardware. Extensive evaluations demonstrate that EventSpeech significantly outperforms current baselines by preserving fine-grained emotions and resisting motion blur to establish a new paradigm for multimodal speech generation. Code and demo are available at this https URL.
>
---
#### [new 013] Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy
- **分类: cs.CL; cs.LG; cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文研究语音表征与认知评估层级的关系，解决MCI分类问题。通过对比传统特征与SSL嵌入，分析不同任务约束对性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.27189](https://arxiv.org/pdf/2605.27189)**

> **作者:** Serli Kopar; Roshan Prakash Rane; Christian Mychajliw; Lydia Federmann; Gerhard Eschweiler; Daniela Berg; Sam Gijsen; Paula Andrea Perez-Toro; Kerstin Ritter
>
> **摘要:** This study examines the relationship between speech representations and the hierarchical structure of cognitive assessment in mild cognitive impairment. Utilizing 5,754 German neuropsychological assessment recordings, we evaluate six cognitive tasks across three score levels: task, domain, and global levels. We compare hand-crafted acoustic features with self-supervised learning (SSL) embeddings. Results show that although SSL representations generally outperform hand-crafted features at lower levels, this trend reverses for MCI classification. Furthermore, task-specific constraints influence performance: tasks with greater response freedom exhibit performance dilution as hierarchical levels increase, suggesting ``specialist'' representations, whereas the performance of highly structured tasks increases toward higher levels, suggesting ``generalist'' representations. These findings show links between task constraints and assessment hierarchy in automated clinical speech analysis.
>
---
## 更新

#### [replaced 001] ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis
- **分类: cs.SD; cs.AI; cs.HC; cs.LG**

- **简介: 该论文提出ParsVoice，一个大规模多说话人波斯语语音语料库，用于文本转语音合成。解决波斯语资源不足的问题，通过自动化管道生成高质量数据，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.10774](https://arxiv.org/pdf/2510.10774)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **摘要:** Persian remains substantially underrepresented in open speech-text resources, limiting progress in multi-speaker text-to-speech (TTS), speech-language modelling, and low-resource speech processing. We introduce ParsVoice, the largest publicly available Persian speech-text corpus tailored for training multi-speaker TTS systems, along with a scalable pipeline to construct high-quality speech-text data from long-form audiobook recordings. The pipeline combines a fine-tuned ParsBERT sentence-completion classifier, ASR-based boundary optimization, punctuation restoration, speaker identification, and a multi-dimensional quality assessment that covers both audio and Persian-specific text properties. The resulting release contains a 2,200-hour TTS-ready subset with 1.36 million aligned segments from 1,815 automatically identified speaker IDs, making it more than 25 times larger than the previously largest open Persian TTS dataset. To validate the corpus, we fine-tune XTTS, a zero-shot multilingual TTS model that operates directly on raw Persian text without phoneme representations, achieving a naturalness MOS of 3.6/5 and speaker similarity MOS of 4.0/5. The ParsVoice dataset is publicly available at: this https URL.
>
---
#### [replaced 002] Genre Controlled Music Generation via Activation Steering
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决风格控制问题。通过在推理阶段干预模型激活，实现对生成音乐风格的精细控制。**

- **链接: [https://arxiv.org/pdf/2506.10225](https://arxiv.org/pdf/2506.10225)**

> **作者:** Swathi Narashiman; Pranay Mathur; Dipanshu Panda; Jayden Koshy Joe; Harshith M R; Anish Veerakumar; Aniruddh Krishna; Keerthiharan A
>
> **摘要:** Computational Music Generation is evolving towards non-conventional styles, demanding methods that enable precise and controllable blending of diverse music elements. In this work, we present a method for fine grained control using inference-time interventions on an autoregressive generative transformer, MusicGen. Through our approach, we achieve genre control by steering the residual stream using weights of a linear probe on it. By framing activation steering as a human-controllable interaction, our work highlights how interpretable model behaviors can empower in co-creative music this http URL samples demonstrating our method are available on our demo page.
>
---
#### [replaced 003] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决语义与声纹难以分离的问题。提出DSA-Tokenizer，通过优化约束分离语义和声纹token，并引入流匹配解码器提升生成质量与克隆能力。**

- **链接: [https://arxiv.org/pdf/2601.09239](https://arxiv.org/pdf/2601.09239)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Linqi Song
>
> **备注:** Submit to ACL ARR 2026 May
>
> **摘要:** Speech tokenizers are a key building block of fully discrete Speech LLMs. Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose \textbf{DSA-Tokenizer}, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. We further introduce a hierarchical Flow Matching decoder and a joint reconstruction-context inpainting training strategy, allowing the model to support both high-fidelity reconstruction and cross-utterance voice clone. To speed up inference, we distill the DiT decoder to reduce sampling steps of inference to 4 and improve synthesis quality with GAN fine-tuning. Experiments demonstrate that DSA-Tokenizer provides strong semantic-acoustic disentanglement, reliable controllable voice cloning, and efficient high-fidelity generation with low WER/CER. Moreover, our results suggest that disentangled tokenization provides a more effective interface for downstream large-model speech generation. Audio samples are avaialble at this https URL.
>
---
#### [replaced 004] CosyEdit2: Speech-Editing-Oriented Reinforcement Learning Unlocks Better Zero-Shot TTS
- **分类: cs.SD**

- **简介: 该论文属于语音编辑与零样本TTS任务，解决数据不足和优化信号粗糙的问题。提出CosyEdit2模型，通过两阶段训练提升编辑性能和零样本TTS能力。**

- **链接: [https://arxiv.org/pdf/2605.25930](https://arxiv.org/pdf/2605.25930)**

> **作者:** Junyang Chen; Yuhang Jia; Hui Wang; Jiaming Zhou; Yongchang Gan; Yong Qin
>
> **摘要:** Speech editing and zero-shot Text-to-Speech (TTS) share a similar generative foundation conditioned on speech prompts, yet speech editing demands far stricter local acoustic consistency with surrounding unedited content. While prior work has shown that Supervised Fine-Tuning (SFT) enables TTS models to acquire functional editing capability, this approach remains fundamentally bottlenecked by imperfect paired editing data and coarse-grained optimization signals. To address these limitations, we propose CosyEdit2, a speech editing model built on a two-stage post-training framework that progresses from supervised editing initialization to editing-oriented Group Relative Policy Optimization (GRPO) over target-speech-free data. Extensive experiments demonstrate that CosyEdit2 not only substantially advances speech editing performance, but also unlocks better zero-shot TTS capability, revealing a deeper mutual relationship between the two tasks. Audio samples are available at this https URL.
>
---
#### [replaced 005] Metric Analysis for Spatial Semantic Segmentation of Sound Scenes
- **分类: cs.SD**

- **简介: 该论文属于声场景空间语义分割任务，旨在解决分离与分类指标混淆的问题。提出CASA-SDR新度量，实现更准确的系统评估。**

- **链接: [https://arxiv.org/pdf/2511.07075](https://arxiv.org/pdf/2511.07075)**

> **作者:** Mayank Mishra; Paul Magron; Romain Serizel
>
> **备注:** 5 pages; content+bibliography
>
> **摘要:** Spatial semantic segmentation of sound scenes (S5) consists of jointly performing audio source separation and sound event classification from a multichannel audio mixture. Evaluating S5 systems with separation and classification metrics individually makes system comparison difficult, whereas existing joint metrics, such as the class-aware signal-to-distortion ratio (CA-SDR), can conflate separation and labeling errors. In particular, CA-SDR relies on predicted class labels for source matching, which may obscure label swaps or misclassifications when the underlying source estimates remain perceptually correct. In this work, we introduce the class and source-aware signal-to-distortion ratio (CASA-SDR), a new metric that performs permutation-invariant source matching before computing classification errors, thereby shifting from a classification-focused approach to a separation-focused approach. We first analyze CA-SDR in controlled scenarios with oracle separation and synthetic classification errors, as well as under controlled cross-contamination between sources, and compare its behavior to that of the classical SDR and CASA-SDR. We also study the impact of classification errors on the metrics by introducing error-based and source-based aggregation strategies. Finally, we compare CA-SDR and CASA-SDR on systems submitted to Task 4 of the DCASE 2025 challenge, highlighting the cases where CA-SDR over-penalizes label swaps or poorly separated sources, while CASA-SDR provides a more interpretable separation-centric assessment of S5 performance.
>
---
#### [replaced 006] PHALAR: Phasors for Learned Musical Audio Representations
- **分类: cs.SD; cs.AI; cs.LG; eess.SP**

- **简介: 该论文提出PHALAR，用于音乐音频表示学习，解决茎音提取任务中因丢失时间信息导致的性能问题。通过引入学习频谱池化层和复数头，提升模型精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.03929](https://arxiv.org/pdf/2605.03929)**

> **作者:** Davide Marincione; Michele Mancusi; Giorgio Strano; Luca Cerovaz; Donato Crisostomi; Roberto Ribuoli; Emanuele Rodolà
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Stem retrieval, the task of matching missing stems to a given audio submix, is a key challenge currently limited by models that discard temporal information. We introduce PHALAR, a contrastive framework achieving a relative accuracy increase of up to $\approx 70\%$ over the state-of-the-art while requiring $<50\%$ of the parameters and a 7$\times$ training speedup. By utilizing a Learned Spectral Pooling layer and a complex-valued head, PHALAR enforces pitch-equivariant and phase-equivariant biases. PHALAR establishes new retrieval state-of-the-art across MoisesDB, Slakh, and ChocoChorales, correlating significantly higher with human coherence judgment than semantic baselines. Finally, zero-shot beat tracking and linear chord probing confirm that PHALAR captures robust musical structures beyond the retrieval task.
>
---
#### [replaced 007] MetaSICL: Adapting Audiroty LLM via Meta Speech In-Context Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音与音频理解任务，旨在解决低资源场景下模型性能下降的问题。通过提出MetaSICL方法，增强模型的上下文学习能力，提升在少量标注数据下的表现。**

- **链接: [https://arxiv.org/pdf/2601.18904](https://arxiv.org/pdf/2601.18904)**

> **作者:** Haolong Zheng; Siyin Wang; Zengrui Jin; Mark Hasegawa-Johnson
>
> **摘要:** Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource tasks. In case in-domain labeled data are scarce or mismatched with the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that $\textit{Vanilla ICL}$, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest that this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose $\textbf{Meta Speech In-Context Learning (MetaSICL)}$, a post-training recipe utilizes only high resource speech data from various tasks intending to strengthen model's in-context learning capability. Experiments indicate our proposed method outperforms direct fine-tuning in low-resource scenario.
>
---
