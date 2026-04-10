# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Towards Real-Time Human-AI Musical Co-Performance: Accompaniment Generation with Latent Diffusion Models and MAX/MSP
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于实时人机音乐协作任务，解决AI与传统音乐工具实时交互的问题。通过结合MAX/MSP和扩散模型，实现低延迟的伴奏生成。**

- **链接: [https://arxiv.org/pdf/2604.07612](https://arxiv.org/pdf/2604.07612)**

> **作者:** Tornike Karchkhadze; Shlomo Dubnov
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** We present a framework for real-time human-AI musical co-performance, in which a latent diffusion model generates instrumental accompaniment in response to a live stream of context audio. The system combines a MAX/MSP front-end-handling real-time audio input, buffering, and playback-with a Python inference server running the generative model, communicating via OSC/UDP messages. This allows musicians to perform in MAX/MSP - a well-established, real-time capable environment - while interacting with a large-scale Python-based generative model, overcoming the fundamental disconnect between real-time music tools and state-of-the-art AI models. We formulate accompaniment generation as a sliding-window look-ahead protocol, training the model to predict future audio from partial context, where system latency is a critical constraint. To reduce latency, we apply consistency distillation to our diffusion model, achieving a 5.4x reduction in sampling time, with both models achieving real-time operation. Evaluated on musical coherence, beat alignment, and audio quality, both models achieve strong performance in the Retrospective regime and degrade gracefully as look-ahead increases. These results demonstrate the feasibility of diffusion-based real-time accompaniment and expose the fundamental trade-off between model latency, look-ahead depth, and generation quality that any such system must navigate.
>
---
#### [new 002] DeepForestSound: a multi-species automatic detector for passive acoustic monitoring in African tropical forests, a case study in Kibale National Park
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出DeepForestSound模型，解决非洲热带森林中生物多样性监测的自动声学检测问题，通过半监督学习提升多物种识别性能。**

- **链接: [https://arxiv.org/pdf/2604.08087](https://arxiv.org/pdf/2604.08087)**

> **作者:** Gabriel Dubus; Théau d'Audiffret; Claire Auger; Raphaël Cornette; Sylvain Haupert; Innocent Kasekendi; Raymond Katumba; Hugo Magaldi; Lise Pernel; Harold Rugonge; Jérôme Sueur; John Justice Tibesigwa; Sabrina Krief
>
> **备注:** 8 pages
>
> **摘要:** Passive Acoustic Monitoring (PAM) is widely used for biodiversity assessment. Its application in African tropical forests is limited by scarce annotated data, reducing the performance of general-purpose ecoacoustic models on underrepresented taxa. In this study, we introduce DeepForestSound (DFS), a multi-species automatic detection model designed for PAM in African tropical forests. DFS relies on a semi-supervised pipeline combining clustering of unannotated recordings with manual validation, followed by supervised fine-tuning of an Audio Spectrogram Transformer (AST) using low-rank adaptation, which is compared to a frozen-backbone linear baseline (DFS-Linear). The framework supports the detection of multiple taxonomic groups, including birds, primates, and elephants, from long-term acoustic recordings. DFS was trained on acoustic data collected in the Sebitoli area, in Kibale National Park, Uganda, and evaluated on an independent dataset recorded two years later at different locations within the same forest. This evaluation therefore assesses generalization across time and recording sites within a single tropical forest ecosystem. Across 8 out of 12 taxons, DFS outperforms existing automatic detection tools, particularly for non-avian taxa, achieving average AP values of 0.964 for primates and 0.961 for elephants. Results further show that LoRA-based fine-tuning substantially outperforms linear probing across taxa. Overall, these results demonstrate that task-oriented, region-specific training substantially improves detection performance in acoustically complex tropical environments, and highlight the potential of DFS as a practical tool for biodiversity monitoring and conservation in African rainforests.
>
---
#### [new 003] Ring Mixing with Auxiliary Signal-to-Consistency-Error Ratio Loss for Unsupervised Denoising in Speech Separation
- **分类: eess.AS**

- **简介: 该论文属于语音分离任务，旨在解决真实场景下噪声抑制的问题。通过引入环形混合和SCER损失函数，提升无监督去噪效果，使系统能从噪声数据中学习更泛化的模型。**

- **链接: [https://arxiv.org/pdf/2604.08415](https://arxiv.org/pdf/2604.08415)**

> **作者:** Matthew Maciejewski; Samuele Cornell
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Noisy speech separation systems are typically trained on fully-synthetic mixtures, limiting generalization to real-world scenarios. Though training on mixtures of in-domain (thus often noisy) speech is possible, we show that this leads to undesirable optima where mixture noise is retained in the estimates, due to the inseparability of the background noises and the loss function's symmetry. To address this, we propose ring mixing, a batch strategy of using each source in two mixtures, alongside a new Signal-to-Consistency-Error Ratio (SCER) auxiliary loss penalizing inconsistent estimates of the same source from different mixtures, breaking symmetry and incentivizing denoising. On a WHAM!-based benchmark, our method can reduce residual noise by upwards of half, effectively learning to denoise from only noisy recordings. This opens the door to training more generalizable systems using in-the-wild data, which we demonstrate via systems trained using naturally-noisy speech from VoxCeleb.
>
---
#### [new 004] TASU2: Controllable CTC Simulation for Alignment and Low-Resource Adaptation of Speech LLMs
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音大模型后训练任务，解决跨模态对齐和低资源适应问题。提出TASU2框架，可控模拟CTC后验分布，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.08384](https://arxiv.org/pdf/2604.08384)**

> **作者:** Jing Peng; Chenghao Wang; Yi Yang; Lirong Qian; Junjie Li; Yu Xi; Shuai Wang; Kai Yu
>
> **摘要:** Speech LLM post-training increasingly relies on efficient cross-modal alignment and robust low-resource adaptation, yet collecting large-scale audio-text pairs remains costly. Text-only alignment methods such as TASU reduce this burden by simulating CTC posteriors from transcripts, but they provide limited control over uncertainty and error rate, making curriculum design largely heuristic. We propose \textbf{TASU2}, a controllable CTC simulation framework that simulates CTC posterior distributions under a specified WER range, producing text-derived supervision that better matches the acoustic decoding interface. This enables principled post-training curricula that smoothly vary supervision difficulty without TTS. Across multiple source-to-target adaptation settings, TASU2 improves in-domain and out-of-domain recognition over TASU, and consistently outperforms strong baselines including text-only fine-tuning and TTS-based augmentation, while mitigating source-domain performance degradation.
>
---
#### [new 005] Semantic Noise Reduction via Teacher-Guided Dual-Path Audio-Visual Representation Learning
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于音频-视觉表示学习任务，旨在解决联合优化带来的语义噪声和干扰问题。通过解耦重建与对齐路径，并引入教师模型指导，提升跨模态检索性能。**

- **链接: [https://arxiv.org/pdf/2604.08147](https://arxiv.org/pdf/2604.08147)**

> **作者:** Linge Wang; Yingying Chen; Bingke Zhu; Lu Zhou; Jinqiao Wang
>
> **摘要:** Recent advances in audio-visual representation learning have shown the value of combining contrastive alignment with masked reconstruction. However, jointly optimizing these objectives in a single forward pass forces the contrastive branch to rely on randomly visible patches designed for reconstruction rather than cross-modal alignment, introducing semantic noise and optimization interference. We propose TG-DP, a Teacher-Guided Dual-Path framework that decouples reconstruction and alignment into separate optimization paths. By disentangling the masking regimes of the two branches, TG-DP enables the contrastive pathway to use a visibility pattern better suited to cross-modal alignment. A teacher model further provides auxiliary guidance for organizing visible tokens in this branch, helping reduce interference and stabilize cross-modal representation learning. TG-DP achieves state-of-the-art performance in zero-shot retrieval. On AudioSet, it improves R@1 from 35.2\% to 37.4\% for video-to-audio retrieval and from 27.9\% to 37.1\% for audio-to-video retrieval. The learned representations also remain semantically robust, achieving state-of-the-art linear-probe performance on AS20K and VGGSound. Taken together, our results suggest that decoupling multimodal objectives and introducing teacher-guided structure into the contrastive pathway provide an effective framework for improving large-scale audio-visual pretraining. Code is available at this https URL.
>
---
#### [new 006] Tracking Listener Attention: Gaze-Guided Audio-Visual Speech Enhancement Framework
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决多说话人环境下的目标说话人识别问题。通过结合眼神方向与视觉信息，提出GG-AVSE框架，提升语音增强效果。**

- **链接: [https://arxiv.org/pdf/2604.08359](https://arxiv.org/pdf/2604.08359)**

> **作者:** Hsiang-Cheng Yang; You-Jin Li; Rong Chao; Yu Tsao; Borching Su; Shao-Yi Chien
>
> **备注:** Accepted to IEEE ICASSP 2026
>
> **摘要:** This paper presents a Gaze-Guided Audio-Visual Speech Enhancement (GG-AVSE) framework to address the cocktail party problem. A major challenge in conventional AVSE is identifying the listener's intended speaker in multi-talker environments. GG-AVSE addresses this issue by exploiting gaze direction as a supervisory cue for target-speaker selection. Specifically, we propose the GG-VM module, which combines gaze signals with a YOLO5Face detector to extract the target speaker's facial features and integrates them with the pretrained AVSEMamba model through two strategies: zero-shot merging and partial visual fine-tuning. For evaluation, we introduce the AVSEC2-Gaze dataset. Experimental results show that GG-AVSE achieves substantial performance gains over gaze-free baselines: a 10.08% improvement in PESQ (2.370 to 2.609), a 5.18% improvement in STOI (0.8802 to 0.9258), and a 23.69% improvement in SI-SDR (9.16 to 11.33). These results confirm that gaze provides an effective cue for resolving target-speaker ambiguity and highlight the scalability of GG-AVSE for real-world applications.
>
---
#### [new 007] Selective Attention System (SAS): Device-Addressed Speech Detection for Real-Time On-Device Voice AI
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出SAS系统，解决设备定向语音检测问题，在边缘设备上实时判断是否转发音频。通过序列路由模型提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.08412](https://arxiv.org/pdf/2604.08412)**

> **作者:** David Joohun Kim; Daniyal Anjum; Bonny Banerjee; Omar Abbasi
>
> **摘要:** We study device-addressed speech detection under pre-ASR edge deployment constraints, where systems must decide whether to forward audio before transcription under strict latency and compute limits. We show that, in multi-speaker environments with temporally ambiguous utterances, this task is more effectively modelled as a sequential routing problem over interaction history than as an utterance-local classification task. We formalize this as Sequential Device-Addressed Routing (SDAR) and present the Selective Attention System (SAS), an on-device implementation that instantiates this formulation. On a held-out 60-hour multi-speaker English test set, the primary audio-only configuration achieves F1=0.86 (precision=0.89, recall=0.83); with an optional camera, audio+video fusion raises F1 to 0.95 (precision=0.97, recall=0.93). Removing causal interaction history (Stage~3) reduced F1 from 0.95 to 0.57+/-0.03 in the audio+video configuration under our evaluation protocol. Among the tested components, this was the largest observed ablation effect, indicating that short-horizon interaction history carries substantial decision-relevant information in the evaluated setting. SAS runs fully on-device on ARM Cortex-A class hardware (<150 ms latency, <20 MB footprint). All results are from internal evaluation on a proprietary dataset evaluated primarily in English; a 5-hour evaluation subset may be shared for independent verification (Section 8.8).
>
---
#### [new 008] DeepFense: A Unified, Modular, and Extensible Framework for Robust Deepfake Audio Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于深度伪造音频检测任务，旨在解决模型可复现性差和性能偏差问题。提出DeepFense框架，集成多种模型与数据增强方法，评估400余模型并分析性能影响因素。**

- **链接: [https://arxiv.org/pdf/2604.08450](https://arxiv.org/pdf/2604.08450)**

> **作者:** Yassine El Kheir; Arnab Das; Yixuan Xiao; Xin Wang; Feidi Kallel; Enes Erdem Erdogan; Ngoc Thang Vu; Tim Polzehl; Sebastian Moeller
>
> **备注:** Deepfense Toolkit
>
> **摘要:** Speech deepfake detection is a well-established research field with different models, datasets, and training strategies. However, the lack of standardized implementations and evaluation protocols limits reproducibility, benchmarking, and comparison across studies. In this work, we present DeepFense, a comprehensive, open-source PyTorch toolkit integrating the latest architectures, loss functions, and augmentation pipelines, alongside over 100 recipes. Using DeepFense, we conducted a large-scale evaluation of more than 400 models. Our findings reveal that while carefully curated training data improves cross-domain generalization, the choice of pre-trained front-end feature extractor dominates overall performance variance. Crucially, we show severe biases in high-performing models regarding audio quality, speaker gender, and language. DeepFense is expected to facilitate real-world deployment with the necessary tools to address equitable training data selection and front-end fine-tuning.
>
---
#### [new 009] AT-ADD: All-Type Audio Deepfake Detection Challenge Evaluation Plan
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有检测方法对非语音音频和真实场景适应性差的问题。提出AT-ADD挑战，涵盖语音和全类型音频检测，提升检测的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.08184](https://arxiv.org/pdf/2604.08184)**

> **作者:** Yuankun Xie; Haonan Cheng; Jiayi Zhou; Xiaoxuan Guo; Tao Wang; Jian Liu; Weiqiang Wang; Ruibo Fu; Xiaopeng Wang; Hengyan Huang; Xiaoying Huang; Long Ye; Guangtao Zhai
>
> **备注:** Accepted to the ACM Multimedia 2026 Grand Challenge
>
> **摘要:** The rapid advancement of Audio Large Language Models (ALLMs) has enabled cost-effective, high-fidelity generation and manipulation of both speech and non-speech audio, including sound effects, singing voices, and music. While these capabilities foster creativity and content production, they also introduce significant security and trust challenges, as realistic audio deepfakes can now be generated and disseminated at scale. Existing audio deepfake detection (ADD) countermeasures (CMs) and benchmarks, however, remain largely speech-centric, often relying on speech-specific artifacts and exhibiting limited robustness to real-world distortions, as well as restricted generalization to heterogeneous audio types and emerging spoofing techniques. To address these gaps, we propose the All-Type Audio Deepfake Detection (AT-ADD) Grand Challenge for ACM Multimedia 2026, designed to bridge controlled academic evaluation with practical multimedia forensics. AT-ADD comprises two tracks: (1) Robust Speech Deepfake Detection, which evaluates detectors under real-world scenarios and against unseen, state-of-the-art speech generation methods; and (2) All-Type Audio Deepfake Detection, which extends detection beyond speech to diverse, unknown audio types and promotes type-agnostic generalization across speech, sound, singing, and music. By providing standardized datasets, rigorous evaluation protocols, and reproducible baselines, AT-ADD aims to accelerate the development of robust and generalizable audio forensic technologies, supporting secure communication, reliable media verification, and responsible governance in an era of pervasive synthetic audio.
>
---
#### [new 010] CapTalk: Unified Voice Design for Single-Utterance and Dialogue Speech Generation
- **分类: cs.SD**

- **简介: 该论文属于语音生成任务，解决从文本生成特定音色和表达风格语音的问题。提出CapTalk框架，统一处理单句和对话语音生成，提升表达控制与上下文适应性。**

- **链接: [https://arxiv.org/pdf/2604.08363](https://arxiv.org/pdf/2604.08363)**

> **作者:** Xiaosu Su; Zihan Sun; Peilei Jia; Jun Gao
>
> **备注:** 14 pages, 2 figures
>
> **摘要:** Voice design from natural language descriptions is emerging as a new task in text-to-speech multimodal generation, aiming to synthesize speech with target timbre and speaking style without relying on reference audio. However, existing methods mainly focus on single-utterance generation, leaving conversational voice design largely unexplored. In this work, we extend voice design to dialogue, enabling better target speaker modeling and turn-level expressive control in natural conversational settings. We propose CapTalk, a unified caption-conditioned text-audio autoregressive framework for both single-utterance and dialogue voice design. CapTalk uses utterance-level captions for single-utterance voice design and speaker-level captions for dialogue speaker modeling, and further introduces a CoT control sequence in dialogue to explicitly plan turn-level dynamic attributes. To resolve the conflict between stable timbre preservation and context-adaptive expression, we propose a hierarchical variational conditioning module with an utterance-level speaker encoder to better balance stable timbre preservation and context-adaptive expression. This enables timbre reuse while keeping expression adaptive to the current utterance and, in dialogue, the surrounding context. We also build a comprehensive evaluation protocol for both single-utterance and dialogue settings. Experiments show that CapTalk achieves state-of-the-art performance on a single-utterance voice design benchmark and delivers better expression controllability and contextual appropriateness in multi-turn dialogue. Audio samples are available at: this https URL.
>
---
#### [new 011] Rethinking Entropy Allocation in LLM-based ASR: Understanding the Dynamics between Speech Encoders and LLMs
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决LLM与语音编码器间的熵分配问题，提升识别质量与鲁棒性。通过优化训练策略，减少幻觉并提高效率。**

- **链接: [https://arxiv.org/pdf/2604.08003](https://arxiv.org/pdf/2604.08003)**

> **作者:** Yuan Xie; Jiaqi Song; Guang Qiu; Xianliang Wang; Ming Lei; Jie Gao; Jie Wu
>
> **摘要:** Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a dominant paradigm. Although recent LLM-based ASR models have shown promising performance on public benchmarks, it remains challenging to balance recognition quality with latency and overhead, while hallucinations further limit real-world deployment. In this study, we revisit LLM-based ASR from an entropy allocation perspective and introduce three metrics to characterize how training paradigms allocate entropy reduction between the speech encoder and the LLM. To remedy entropy-allocation inefficiencies in prevailing approaches, we propose a principled multi-stage training strategy grounded in capability-boundary awareness, optimizing parameter efficiency and hallucination robustness. Specifically, we redesign the pretraining strategy to alleviate the speech-text modality gap, and further introduce an iterative asynchronous SFT stage between alignment and joint SFT to preserve functional decoupling and constrain encoder representation drift. Experiments on Mandarin and English benchmarks show that our method achieves competitive performance with state-of-the-art models using only 2.3B parameters, while also effectively mitigating hallucinations through our decoupling-oriented design.
>
---
#### [new 012] Semantic-Emotional Resonance Embedding: A Semi-Supervised Paradigm for Cross-Lingual Speech Emotion Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于跨语言语音情感识别任务，解决低资源语言性能不足的问题。提出SERE框架，无需目标语言标签或翻译对齐，通过半监督方法实现情感语义结构学习。**

- **链接: [https://arxiv.org/pdf/2604.07417](https://arxiv.org/pdf/2604.07417)**

> **作者:** Ya Zhao; Yinfeng Yu; Liejun Wang
>
> **备注:** Main paper (6 pages). Accepted for publication by IEEE International conference on Multimedia and Expo 2026 (ICME 2026)
>
> **摘要:** Cross-lingual Speech Emotion Recognition (CLSER) aims to identify emotional states in unseen languages. However, existing methods heavily rely on the semantic synchrony of complete labels and static feature stability, hindering low-resource languages from reaching high-resource performance. To address this, we propose a semi-supervised framework based on Semantic-Emotional Resonance Embedding (SERE), a cross-lingual dynamic feature paradigm that requires neither target language labels nor translation alignment. Specifically, SERE constructs an emotion-semantic structure using a small number of labeled samples. It learns human emotional experiences through an Instantaneous Resonance Field (IRF), enabling unlabeled samples to self-organize into this structure. This achieves semi-supervised semantic guidance and structural discovery. Additionally, we design a Triple-Resonance Interaction Chain (TRIC) loss to enable the model to reinforce the interaction and embedding capabilities between labeled and unlabeled samples during emotional highlights. Extensive experiments across multiple languages demonstrate the effectiveness of our method, requiring only 5-shot labeling in the source language.
>
---
#### [new 013] Contextual Earnings-22: A Speech Recognition Benchmark with Custom Vocabulary in the Wild
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决工业场景中自定义词汇识别准确率低的问题。通过构建包含真实自定义词汇的基准数据集，提升语音转文本系统的实用性。**

- **链接: [https://arxiv.org/pdf/2604.07354](https://arxiv.org/pdf/2604.07354)**

> **作者:** Berkin Durmus; Chen Cen; Eduardo Pacheco; Arda Okan; Atila Orhon
>
> **摘要:** The accuracy frontier of speech-to-text systems has plateaued on academic benchmarks.1 In contrast, industrial benchmarks and adoption in high-stakes domains suggest otherwise. We hypothesize that the primary difference between the two is contextual conditioning: Academic benchmarks are dominated by frequently encountered general vocabulary that is relatively easy to recognize compared with rare and context-defined custom vocabulary that has disproportionate impact on the usability of speech transcripts. Despite progress on contextual speech-to-text, there is no standardized benchmark. We introduce Contextual Earnings-22, an open dataset built upon Earnings-22, with realistic custom vocabulary contexts to foster research and reveal latent progress. We set six strong baselines for two dominant approaches: keyword prompting and keyword boosting. Experiments show both reach comparable and significantly improved accuracy when scaled from proof-of-concept to large-scale systems.
>
---
#### [new 014] Hybrid CNN-Transformer Architecture for Arabic Speech Emotion Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于阿拉伯语语音情感识别任务，旨在解决阿拉伯语数据稀缺的问题。提出混合CNN-Transformer模型，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2604.07357](https://arxiv.org/pdf/2604.07357)**

> **作者:** Youcef Soufiane Gheffari; Oussama Mustapha Benouddane; Samiya Silarbi
>
> **备注:** 7 pages, 4 figures. Master's thesis work, University of Science and Technology of Oran - Mohamed Boudiaf (USTO-MB)
>
> **摘要:** Recognizing emotions from speech using machine learning has become an active research area due to its importance in building human-centered applications. However, while many studies have been conducted in English, German, and other European and Asian languages, research in Arabic remains scarce because of the limited availability of annotated datasets. In this paper, we present an Arabic Speech Emotion Recognition (SER) system based on a hybrid CNN-Transformer architecture. The model leverages convolutional layers to extract discriminative spectral features from Mel-spectrogram inputs and Transformer encoders to capture long-range temporal dependencies in speech. Experiments were conducted on the EYASE (Egyptian Arabic speech emotion) corpus, and the proposed model achieved 97.8% accuracy and a macro F1-score of 0.98. These results demonstrate the effectiveness of combining convolutional feature extraction with attention-based modeling for Arabic SER and highlight the potential of Transformer-based approaches in low-resource languages.
>
---
#### [new 015] Bridging the Gap between Micro-scale Traffic Simulation and 4D Digital Cityscapes
- **分类: cs.HC; cs.SD**

- **简介: 该论文属于交通仿真与可视化任务，旨在解决微尺度交通模拟与高保真4D城市景观融合的问题，通过实时4D框架实现交通数据与虚拟现实的同步展示。**

- **链接: [https://arxiv.org/pdf/2604.08497](https://arxiv.org/pdf/2604.08497)**

> **作者:** Longxiang Jiao; Lukas Hofmann; Yiru Yang; Zhanyi Wu; Jonas Egeler
>
> **摘要:** While micro-scale traffic simulations provide essential data for urban planning, they are rarely coupled with the high-fidelity visualization or auralization necessary for effective stakeholder communication. In this work, we present a real-time 4D visualization framework that couples the SUMO traffic with a photorealistic, geospatially accurate VR representation of Zurich in Unreal Engine 5. Our architecture implements a robust C++ data pipeline for synchronized vehicle visualization and features an Open Sound Control (OSC) interface to support external auralization engines. We validate the framework through a user study assessing the correlation between simulated traffic dynamics and human perception. Results demonstrate a high degree of perceptual alignment, where users correctly interpret safety risks from the 4D simulation. Furthermore, our findings indicate that the inclusion of spatialized audio alters the user's sense of safety, showing the importance of multimodality in traffic simulations.
>
---
## 更新

#### [replaced 001] RiTTA: Modeling Event Relations in Text-to-Audio Generation
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决音频事件关系建模问题。提出基准数据集和评估指标，并设计微调框架以提升模型对音频事件关系的建模能力。**

- **链接: [https://arxiv.org/pdf/2412.15922](https://arxiv.org/pdf/2412.15922)**

> **作者:** Yuhang He; Yash Jain; Xubo Liu; Andrew Markham; Vibhav Vineet
>
> **备注:** EMNLP25, Project Site: this https URL. Code: this https URL
>
> **摘要:** Despite significant advancements in Text-to-Audio (TTA) generation models achieving high-fidelity audio with fine-grained context understanding, they struggle to model the relations between audio events described in the input text. However, previous TTA methods have not systematically explored audio event relation modeling, nor have they proposed frameworks to enhance this capability. In this work, we systematically study audio event relation modeling in TTA generation models. We first establish a benchmark for this task by: 1. proposing a comprehensive relation corpus covering all potential relations in real-world scenarios; 2. introducing a new audio event corpus encompassing commonly heard audios; and 3. proposing new evaluation metrics to assess audio event relation modeling from various perspectives. Furthermore, we propose a finetuning framework to enhance existing TTA models ability to model audio events relation. Code is available at: this https URL
>
---
#### [replaced 002] DHFP-PE: Dual-Precision Hybrid Floating Point Processing Element for AI Acceleration
- **分类: cs.AR; cs.RO; eess.AS; eess.IV**

- **简介: 该论文属于AI加速任务，旨在解决低功耗高吞吐量浮点乘加单元的设计问题。提出一种双精度浮点处理单元，支持多种低精度格式，提升硬件利用率并降低功耗。**

- **链接: [https://arxiv.org/pdf/2604.04507](https://arxiv.org/pdf/2604.04507)**

> **作者:** Shubham Kumar; Vijay Pratap Sharma; Vaibhav Neema; Santosh Kumar Vishvakarma
>
> **备注:** Accepted in ANRF-sponsored 2nd International Conference on Next Generation Electronics (NEleX-2026)
>
> **摘要:** The rapid adoption of low-precision arithmetic in artificial intelligence and edge computing has created a strong demand for energy-efficient and flexible floating-point multiply-accumulate (MAC) units. This paper presents a dual-precision floating-point MAC processing element supporting FP8 (E4M3, E5M2) and FP4 (2 x E2M1, 2 x E1M2) formats, specifically optimized for low-power and high-throughput AI workloads. The proposed architecture employs a novel bit-partitioning technique that enables a single 4-bit unit multiplier to operate either as a standard 4 x 4 multiplier for FP8 or as two parallel 2 x 2 multipliers for 2-bit operands, achieving maximum hardware utilization without duplicating logic. Implemented in 28 nm technology, the proposed PE achieves an operating frequency of 1.94 GHz with an area of 0.00396 mm^2 and power consumption of 2.13 mW, resulting in up to 60.4% area reduction and 86.6% power savings compared to state-of-the-art designs, making it well suited for energy-constrained AI inference and mixed-precision computing applications when deployed within larger accelerator architectures.
>
---
#### [replaced 003] YingMusic-Singer-Plus: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于歌唱语音合成任务，旨在解决修改歌词同时保持旋律一致的问题。提出YingMusic-Singer-Plus模型，实现灵活的歌词编辑和无需手动对齐的旋律控制。**

- **链接: [https://arxiv.org/pdf/2603.24589](https://arxiv.org/pdf/2603.24589)**

> **作者:** Chunbo Hao; Junjie Zheng; Guobin Ma; Yuepeng Jiang; Huakang Chen; Wenjie Tian; Gongyu Chen; Zihao Chen; Lei Xie
>
> **摘要:** Regenerating singing voices with altered lyrics while preserving melody consistency remains challenging, as existing methods either offer limited controllability or require laborious manual alignment. We propose YingMusic-Singer-Plus, a fully diffusion-based model enabling melody-controllable singing voice synthesis with flexible lyric manipulation. The model takes three inputs: an optional timbre reference, a melody-providing singing clip, and modified lyrics, without manual alignment. Trained with curriculum learning and Group Relative Policy Optimization, YingMusic-Singer-Plus achieves stronger melody preservation and lyric adherence than Vevo2, the most comparable baseline supporting melody control without manual alignment. We also introduce LyricEditBench, the first benchmark for melody-preserving lyric modification evaluation. The code, weights, benchmark, and demos are publicly available at this https URL.
>
---
#### [replaced 004] NSTR: Neural Spectral Transport Representation for Space-Varying Frequency Fields
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出NSTR，一种用于建模空间变化频谱的隐式神经表示框架，解决传统INR无法处理非平稳频域特征的问题。**

- **链接: [https://arxiv.org/pdf/2511.18384](https://arxiv.org/pdf/2511.18384)**

> **作者:** Plein Versace
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
>
> **摘要:** Implicit Neural Representations (INRs) have emerged as a powerful paradigm for representing signals such as images, audio, and 3D scenes. However, existing INR frameworks -- including MLPs with Fourier features, SIREN, and multiresolution hash grids -- implicitly assume a \textit{global and stationary} spectral basis. This assumption is fundamentally misaligned with real-world signals whose frequency characteristics vary significantly across space, exhibiting local high-frequency textures, smooth regions, and frequency drift phenomena. We propose \textbf{Neural Spectral Transport Representation (NSTR)}, the first INR framework that \textbf{explicitly models a spatially varying local frequency field}. NSTR introduces a learnable \emph{frequency transport equation}, a PDE that governs how local spectral compositions evolve across space. Given a learnable local spectrum field $S(x)$ and a frequency transport network $F_\theta$ enforcing $\nabla S(x) \approx F_\theta(x, S(x))$, NSTR reconstructs signals by spatially modulating a compact set of global sinusoidal bases. This formulation enables strong local adaptivity and offers a new level of interpretability via visualizing frequency flows. Experiments on 2D image regression, audio reconstruction, and implicit 3D geometry show that NSTR achieves significantly better accuracy-parameter trade-offs than SIREN, Fourier-feature MLPs, and Instant-NGP. NSTR requires fewer global frequencies, converges faster, and naturally explains signal structure through spectral transport fields. We believe NSTR opens a new direction in INR research by introducing explicit modeling of space-varying spectrum.
>
---
#### [replaced 005] Controllable Embedding Transformation for Mood-Guided Music Retrieval
- **分类: cs.SD**

- **简介: 该论文属于音乐检索任务，旨在实现对音乐嵌入的可控变换，解决仅调整单一属性（如情绪）而保持其他属性不变的问题。通过引入采样机制和联合目标函数，提升情绪引导的音乐检索效果。**

- **链接: [https://arxiv.org/pdf/2510.20759](https://arxiv.org/pdf/2510.20759)**

> **作者:** Julia Wilkins; Jaehun Kim; Matthew E. P. Davies; Juan Pablo Bello; Matthew C. McCallum
>
> **备注:** Preprint; under review
>
> **摘要:** Music representations are the backbone of modern recommendation systems, powering playlist generation, similarity search, and personalized discovery. Yet most embeddings offer little control for adjusting a single musical attribute, e.g., changing only the mood of a track while preserving its genre or instrumentation. In this work, we address the problem of controllable music retrieval through embedding-based transformation, where the objective is to retrieve songs that remain similar to a seed track but are modified along one chosen dimension. We propose a novel framework for mood-guided music embedding transformation, which learns a mapping from a seed audio embedding to a target embedding guided by mood labels, while preserving other musical attributes. Because mood cannot be directly altered in the seed audio, we introduce a sampling mechanism that retrieves proxy targets to balance diversity with similarity to the seed. We train a lightweight translation model using this sampling strategy and introduce a novel joint objective that encourages transformation and information preservation. Extensive experiments on two datasets show strong mood transformation performance while retaining genre and instrumentation far better than training-free baselines, establishing controllable embedding transformation as a promising paradigm for personalized music retrieval.
>
---
#### [replaced 006] AudioMoG: Guiding Audio Generation with Mixture-of-Guidance
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频生成任务，旨在提升文本到音频和视频到音频的生成质量。针对现有引导方法多样性不足的问题，提出AudioMoG混合引导框架，整合多种引导信号以优化生成效果。**

- **链接: [https://arxiv.org/pdf/2509.23727](https://arxiv.org/pdf/2509.23727)**

> **作者:** Junyou Wang; Zehua Chen; Binjie Yuan; Kaiwen Zheng; Chang Li; Yuxuan Jiang; Jun Zhu
>
> **备注:** Accepted at ICME 2026
>
> **摘要:** The design of diffusion-based audio generation systems has been investigated from diverse perspectives, such as data space, network architecture, and conditioning techniques, while most of these innovations require model re-training. In sampling, classifier-free guidance (CFG) has been uniformly adopted to enhance generation quality by strengthening condition alignment. However, CFG often compromises diversity, resulting in suboptimal performance. Although the recent autoguidance (AG) method proposes another direction of guidance that maintains diversity, its direct application in audio generation has so far underperformed CFG. In this work, we introduce AudioMoG, an improved sampling method that enhances text-to-audio (T2A) and video-to-audio (V2A) generation quality without requiring extensive training resources. We start with an analysis of both CFG and AG, examining their respective advantages and limitations for guiding diffusion models. Building upon our insights, we introduce a mixture-of-guidance framework that integrates diverse guidance signals with their interaction terms (e.g., the unconditional bad version of the model) to maximize cumulative advantages. Experiments show that, given the same inference speed, our approach consistently outperforms single guidance in T2A generation across sampling steps, concurrently showing advantages in V2A, text-to-music, and image generation. Demo samples are available at: this https URL.
>
---
#### [replaced 007] EvoTSE: Evolving Enrollment for Target Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文属于目标说话人提取任务，解决模型在混音中混淆说话人的问题。提出EvoTSE框架，通过动态更新语音档案提升效果。**

- **链接: [https://arxiv.org/pdf/2604.06810](https://arxiv.org/pdf/2604.06810)**

> **作者:** Zikai Liu; Ziqian Wang; Xingchen Li; Yike Zhu; Shuai Wang; Longshuai Xiao; Lei Xie
>
> **摘要:** Target Speaker Extraction (TSE) aims to isolate a specific speaker's voice from a mixture, guided by a pre-recorded enrollment. While TSE bypasses the global permutation ambiguity of blind source separation, it remains vulnerable to speaker confusion, where models mistakenly extract the interfering speaker. Furthermore, conventional TSE relies on static inference pipeline, where performance is limited by the quality of the fixed enrollment. To overcome these limitations, we propose EvoTSE, an evolving TSE framework in which the enrollment is continuously updated through reliability-filtered retrieval over high-confidence historical estimates. This mechanism reduces speaker confusion and relaxes the quality requirements for pre-recorded enrollment without relying on additional annotated data. Experiments across multiple benchmarks demonstrate that EvoTSE achieves consistent improvements, especially when evaluated on out-of-domain (OOD) scenarios. Our code and checkpoints are available.
>
---
