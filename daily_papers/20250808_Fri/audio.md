# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] A Scalable Pipeline for Enabling Non-Verbal Speech Generation and Understanding
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出了一种可扩展的非口头语音生成与理解管道，旨在解决现有系统仅关注语音内容而缺乏非语言信号的问题。通过引入NonVerbalSpeech-38K数据集（包含10类非语言信号），并验证F5-TTS等模型的有效性，实现了非语言信息的整合与交互优化。**

- **链接: [http://arxiv.org/pdf/2508.05385v1](http://arxiv.org/pdf/2508.05385v1)**

> **作者:** Runchuan Ye; Yixuan Zhou; Renjie Yu; Zijian Lin; Kehan Li; Xiang Li; Xin Liu; Guoyang Zeng; Zhiyong Wu
>
> **摘要:** Human spoken communication involves not only lexical content but also non-verbal vocalizations (NVs) such as laughter, sighs, and coughs, which convey emotions, intentions, and social signals. However, most existing speech systems focus solely on verbal content and lack the ability to understand and generate such non-verbal cues, reducing the emotional intelligence and communicative richness of spoken interfaces. In this work, we introduce $\textbf{NonVerbalSpeech-38K}$, a large and diverse dataset for non-verbal speech generation and understanding, collected from real-world media and annotated using an automatic pipeline. The dataset contains 38,718 samples (about 131 hours) with 10 categories of non-verbal cues, such as laughter, sniff, and throat clearing. We further validate the dataset by fine-tuning state-of-the-art models, including F5-TTS and Qwen2-Audio, demonstrating its effectiveness in non-verbal speech generation and understanding tasks. Our contributions are threefold: (1) We propose a practical pipeline for building natural and diverse non-verbal speech datasets; (2) We release a large-scale dataset to advance research on non-verbal speech generation and understanding; (3) We validate the dataset's effectiveness by demonstrating improvements in both non-verbal speech synthesis and captioning, thereby facilitating richer human-computer interaction.
>
---
#### [new 002] Estimating Musical Surprisal from Audio in Autoregressive Diffusion Model Noise Spaces
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究了利用自回归扩散模型（ADMs）的噪声空间估算音乐 surprisal，旨在通过比较不同ADMs与GIVT在捕捉音调和段落边界方面的表现，验证了噪声水平对音乐特征 surprisal 的影响。**

- **链接: [http://arxiv.org/pdf/2508.05306v1](http://arxiv.org/pdf/2508.05306v1)**

> **作者:** Mathias Rose Bjare; Stefan Lattner; Gerhard Widmer
>
> **备注:** 9 pages, 1 figure, 5 tables. Accepted at the 25th International Society for Music Information Retrieval Conference (ISMIR), Daejeon, South Korea, 2025 2025
>
> **摘要:** Recently, the information content (IC) of predictions from a Generative Infinite-Vocabulary Transformer (GIVT) has been used to model musical expectancy and surprisal in audio. We investigate the effectiveness of such modelling using IC calculated with autoregressive diffusion models (ADMs). We empirically show that IC estimates of models based on two different diffusion ordinary differential equations (ODEs) describe diverse data better, in terms of negative log-likelihood, than a GIVT. We evaluate diffusion model IC's effectiveness in capturing surprisal aspects by examining two tasks: (1) capturing monophonic pitch surprisal, and (2) detecting segment boundaries in multi-track audio. In both tasks, the diffusion models match or exceed the performance of a GIVT. We hypothesize that the surprisal estimated at different diffusion process noise levels corresponds to the surprisal of music and audio features present at different audio granularities. Testing our hypothesis, we find that, for appropriate noise levels, the studied musical surprisal tasks' results improve. Code is provided on github.com/SonyCSLParis/audioic.
>
---
#### [new 003] SpectroStream: A Versatile Neural Codec for General Audio
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出了一种新型神经音频编码器(SpectroStream)，解决多通道音频处理与高采样率重构问题，通过时间-频域表示和延迟融合策略提升音频质量，突破传统码率限制。**

- **链接: [http://arxiv.org/pdf/2508.05207v1](http://arxiv.org/pdf/2508.05207v1)**

> **作者:** Yunpeng Li; Kehang Han; Brian McWilliams; Zalan Borsos; Marco Tagliasacchi
>
> **摘要:** We propose SpectroStream, a full-band multi-channel neural audio codec. Successor to the well-established SoundStream, SpectroStream extends its capability beyond 24 kHz monophonic audio and enables high-quality reconstruction of 48 kHz stereo music at bit rates of 4--16 kbps. This is accomplished with a new neural architecture that leverages audio representation in the time-frequency domain, which leads to better audio quality especially at higher sample rate. The model also uses a delayed-fusion strategy to handle multi-channel audio, which is crucial in balancing per-channel acoustic quality and cross-channel phase consistency.
>
---
#### [new 004] Towards Hallucination-Free Music: A Reinforcement Learning Preference Optimization Framework for Reliable Song Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出了一种基于强化学习的音乐生成框架，旨在通过偏好优化技术解决内容幻觉问题，开发了PER数据集并实施了DPO、PPO和GRPO等优化策略，有效抑制幻觉同时保持音乐质量，为音乐生成提供了系统性解决方案。**

- **链接: [http://arxiv.org/pdf/2508.05011v1](http://arxiv.org/pdf/2508.05011v1)**

> **作者:** Huaicheng Zhang; Wei Tan; Guangzheng Li; Yixuan Zhang; Hangting Chen; Shun Lei; Chenyu Yang; Zhiyong Wu; Shuai Wang; Qijun Huang; Dong Yu
>
> **摘要:** Recent advances in audio-based generative language models have accelerated AI-driven lyric-to-song generation. However, these models frequently suffer from content hallucination, producing outputs misaligned with the input lyrics and undermining musical coherence. Current supervised fine-tuning (SFT) approaches, limited by passive label-fitting, exhibit constrained self-improvement and poor hallucination mitigation. To address this core challenge, we propose a novel reinforcement learning (RL) framework leveraging preference optimization for hallucination control. Our key contributions include: (1) Developing a robust hallucination preference dataset constructed via phoneme error rate (PER) computation and rule-based filtering to capture alignment with human expectations; (2) Implementing and evaluating three distinct preference optimization strategies within the RL framework: Direct Preference Optimization (DPO), Proximal Policy Optimization (PPO), and Group Relative Policy Optimization (GRPO). DPO operates off-policy to enhance positive token likelihood, achieving a significant 7.4% PER reduction. PPO and GRPO employ an on-policy approach, training a PER-based reward model to iteratively optimize sequences via reward maximization and KL-regularization, yielding PER reductions of 4.9% and 4.7%, respectively. Comprehensive objective and subjective evaluations confirm that our methods effectively suppress hallucinations while preserving musical quality. Crucially, this work presents a systematic, RL-based solution to hallucination control in lyric-to-song generation. The framework's transferability also unlocks potential for music style adherence and musicality enhancement, opening new avenues for future generative song research.
>
---
#### [new 005] SPGISpeech 2.0: Transcribed multi-speaker financial audio for speaker-tagged transcription
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出SPGISpeech 2.0，解决跨语言语音转录任务，通过扩展数据并添加多语种标注，提升基于端到端ASR模型在财务领域的性能。**

- **链接: [http://arxiv.org/pdf/2508.05554v1](http://arxiv.org/pdf/2508.05554v1)**

> **作者:** Raymond Grossman; Taejin Park; Kunal Dhawan; Andrew Titus; Sophia Zhi; Yulia Shchadilova; Weiqing Wang; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** To be presented at Interspeech 2025
>
> **摘要:** We introduce SPGISpeech 2.0, a dataset suitable for speaker-tagged transcription in the financial domain. SPGISpeech 2.0 improves the diversity of applicable modeling tasks while maintaining the core characteristic of the original SPGISpeech dataset: audio snippets and their corresponding fully formatted text transcriptions, usable for end-to-end automatic speech recognition (ASR). SPGISpeech 2.0 consists of 3,780 additional hours of professionally transcribed earnings calls. Furthermore, the dataset contains call and speaker information for each audio snippet facilitating multi-talker ASR. We validate the utility of SPGISpeech 2.0 through improvements in speaker-tagged ASR performance of popular speech recognition models after fine-tuning on SPGISpeech 2.0. Released free for non-commercial use, we expect SPGISpeech 2.0 to foster advancements in speech recognition technologies and inspire a wide range of research applications.
>
---
#### [new 006] Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS
- **分类: cs.SD; cs.AI; eess.AS; 68T50, 68T10, 94A12; I.2.7; H.3.3; C.2.2**

- **简介: 该论文提出了一套基于低延迟通信语音助手的AI架构，解决实时语音交互与低延迟传输问题。通过集成Streaming ASR、Quantized LLMs、Real-Time TTS等技术，构建了支持知识驱动的智能客服系统，实现了高效、跨域的 telecom 语音服务。**

- **链接: [http://arxiv.org/pdf/2508.04721v1](http://arxiv.org/pdf/2508.04721v1)**

> **作者:** Vignesh Ethiraj; Ashwath David; Sidhanth Menon; Divya Vijay
>
> **摘要:** We introduce a low-latency telecom AI voice agent pipeline for real-time, interactive telecommunications use, enabling advanced voice AI for call center automation, intelligent IVR (Interactive Voice Response), and AI-driven customer support. The solution is built for telecom, combining four specialized models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific Text-to-Speech (TTS) model. These models enable highly responsive, domain-adapted voice AI agents supporting knowledge-grounded spoken interactions with low latency. The pipeline integrates streaming ASR (TTE), conversational intelligence (TSLAM), retrieval augmented generation (RAG) over telecom documents, and real-time TTS (T-Synth), setting a new benchmark for telecom voice assistants. To evaluate the system, we built a dataset of 500 human-recorded telecom questions from RFCs, simulating real telecom agent queries. This framework allows analysis of latency, domain relevance, and real-time performance across the stack. Results show that TSLAM, TTE, and T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise, low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and T-Synth -- provide a foundation for next-generation telecom AI, enabling automated customer support, diagnostics, and more.
>
---
#### [new 007] Wearable Music2Emotion : Assessing Emotions Induced by AI-Generated Music through Portable EEG-fNIRS Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文旨在评估AI生成音乐对情绪的影响，解决刺激受限、模态单一及设备复杂性等问题，提出MEEtBrain框架，整合EEG-fNIRS与AI音乐生成，实现跨模态实时情感分析。**

- **链接: [http://arxiv.org/pdf/2508.04723v1](http://arxiv.org/pdf/2508.04723v1)**

> **作者:** Sha Zhao; Song Yi; Yangxuan Zhou; Jiadong Pan; Jiquan Wang; Jie Xia; Shijian Li; Shurong Dong; Gang Pan
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Emotions critically influence mental health, driving interest in music-based affective computing via neurophysiological signals with Brain-computer Interface techniques. While prior studies leverage music's accessibility for emotion induction, three key limitations persist: \textbf{(1) Stimulus Constraints}: Music stimuli are confined to small corpora due to copyright and curation costs, with selection biases from heuristic emotion-music mappings that ignore individual affective profiles. \textbf{(2) Modality Specificity}: Overreliance on unimodal neural data (e.g., EEG) ignores complementary insights from cross-modal signal fusion.\textbf{ (3) Portability Limitation}: Cumbersome setups (e.g., 64+ channel gel-based EEG caps) hinder real-world applicability due to procedural complexity and portability barriers. To address these limitations, we propose MEEtBrain, a portable and multimodal framework for emotion analysis (valence/arousal), integrating AI-generated music stimuli with synchronized EEG-fNIRS acquisition via a wireless headband. By MEEtBrain, the music stimuli can be automatically generated by AI on a large scale, eliminating subjective selection biases while ensuring music diversity. We use our developed portable device that is designed in a lightweight headband-style and uses dry electrodes, to simultaneously collect EEG and fNIRS recordings. A 14-hour dataset from 20 participants was collected in the first recruitment to validate the framework's efficacy, with AI-generated music eliciting target emotions (valence/arousal). We are actively expanding our multimodal dataset (44 participants in the latest dataset) and make it publicly available to promote further research and practical applications. \textbf{The dataset is available at https://zju-bmi-lab.github.io/ZBra.
>
---
#### [new 008] Enhancing Dialogue Annotation with Speaker Characteristics Leveraging a Frozen LLM
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文探讨了通过冷冻LLM增强对话注释的方法，利用语音特征（如年龄、性别、情绪）进行补充，解决了传统LLMs在对话质量上的局限性，并实现了高效且模块化的工作方式。**

- **链接: [http://arxiv.org/pdf/2508.04795v1](http://arxiv.org/pdf/2508.04795v1)**

> **作者:** Thomas Thebaud; Yen-Ju Lu; Matthew Wiesner; Peter Viechnicki; Najim Dehak
>
> **备注:** Accepted in the 2025 IEEE Automatic Speech Recognition and Understanding Workshop
>
> **摘要:** In dialogue transcription pipelines, Large Language Models (LLMs) are frequently employed in post-processing to improve grammar, punctuation, and readability. We explore a complementary post-processing step: enriching transcribed dialogues by adding metadata tags for speaker characteristics such as age, gender, and emotion. Some of the tags are global to the entire dialogue, while some are time-variant. Our approach couples frozen audio foundation models, such as Whisper or WavLM, with a frozen LLAMA language model to infer these speaker attributes, without requiring task-specific fine-tuning of either model. Using lightweight, efficient connectors to bridge audio and language representations, we achieve competitive performance on speaker profiling tasks while preserving modularity and speed. Additionally, we demonstrate that a frozen LLAMA model can compare x-vectors directly, achieving an Equal Error Rate of 8.8% in some scenarios.
>
---
#### [new 009] From Detection to Correction: Backdoor-Resilient Face Recognition via Vision-Language Trigger Detection and Noise-Based Neutralization
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出了一种基于视觉语言模型的反后门攻击方案TrueBiometric，解决了如何准确检测和修正受污染图像的问题，通过多模型联合检测与噪声校正技术实现了100%准确率，显著提升了人脸识别系统的可靠性。**

- **链接: [http://arxiv.org/pdf/2508.05409v1](http://arxiv.org/pdf/2508.05409v1)**

> **作者:** Farah Wahida; M. A. P. Chamikara; Yashothara Shanmugarasa; Mohan Baruwal Chhetri; Thilina Ranbaduge; Ibrahim Khalil
>
> **备注:** 19 Pages, 24 Figures
>
> **摘要:** Biometric systems, such as face recognition systems powered by deep neural networks (DNNs), rely on large and highly sensitive datasets. Backdoor attacks can subvert these systems by manipulating the training process. By inserting a small trigger, such as a sticker, make-up, or patterned mask, into a few training images, an adversary can later present the same trigger during authentication to be falsely recognized as another individual, thereby gaining unauthorized access. Existing defense mechanisms against backdoor attacks still face challenges in precisely identifying and mitigating poisoned images without compromising data utility, which undermines the overall reliability of the system. We propose a novel and generalizable approach, TrueBiometric: Trustworthy Biometrics, which accurately detects poisoned images using a majority voting mechanism leveraging multiple state-of-the-art large vision language models. Once identified, poisoned samples are corrected using targeted and calibrated corrective noise. Our extensive empirical results demonstrate that TrueBiometric detects and corrects poisoned images with 100\% accuracy without compromising accuracy on clean images. Compared to existing state-of-the-art approaches, TrueBiometric offers a more practical, accurate, and effective solution for mitigating backdoor attacks in face recognition systems.
>
---
#### [new 010] Embedding Alignment in Code Generation for Audio
- **分类: cs.MM; cs.AI; cs.SD; eess.AS**

- **简介: 该论文探讨了如何利用代码生成模型与音频映射关系来提升音乐性。研究解决了LLM在代码生成中难以生成符合音乐意图的音频问题，通过分析代码与音频嵌入空间的关系，构建了预测模型并提出了代码-音频映射模型。**

- **链接: [http://arxiv.org/pdf/2508.05473v1](http://arxiv.org/pdf/2508.05473v1)**

> **作者:** Sam Kouteili; Hiren Madhu; George Typaldos; Mark Santolucito
>
> **摘要:** LLM-powered code generation has the potential to revolutionize creative coding endeavors, such as live-coding, by enabling users to focus on structural motifs over syntactic details. In such domains, when prompting an LLM, users may benefit from considering multiple varied code candidates to better realize their musical intentions. Code generation models, however, struggle to present unique and diverse code candidates, with no direct insight into the code's audio output. To better establish a relationship between code candidates and produced audio, we investigate the topology of the mapping between code and audio embedding spaces. We find that code and audio embeddings do not exhibit a simple linear relationship, but supplement this with a constructed predictive model that shows an embedding alignment map could be learned. Supplementing the aim for musically diverse output, we present a model that given code predicts output audio embedding, constructing a code-audio embedding alignment map.
>
---
#### [new 011] Pitch Accent Detection improves Pretrained Automatic Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文旨在改进基于半监督的自动语音识别（ASR）系统，解决传统方法在融合补充信息时性能不足的问题，通过引入联合ASR与pitch accent检测模块，显著提升了F1分数并优化了有限资源下的训练效果。**

- **链接: [http://arxiv.org/pdf/2508.04814v1](http://arxiv.org/pdf/2508.04814v1)**

> **作者:** David Sasu; Natalie Schluter
>
> **摘要:** We show the performance of Automatic Speech Recognition (ASR) systems that use semi-supervised speech representations can be boosted by a complimentary pitch accent detection module, by introducing a joint ASR and pitch accent detection model. The pitch accent detection component of our model achieves a significant improvement on the state-of-the-art for the task, closing the gap in F1-score by 41%. Additionally, the ASR performance in joint training decreases WER by 28.3% on LibriSpeech, under limited resource fine-tuning. With these results, we show the importance of extending pretrained speech models to retain or re-learn important prosodic cues such as pitch accent.
>
---
#### [new 012] Prescriptive Agents based on Rag for Automated Maintenance (PARAM)
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; eess.SP**

- **简介: 该论文提出基于LLM的智能维护系统，解决工业设备维护效率不足的问题，通过整合振动数据分析与多智能体生成技术，实现故障诊断与结构化建议生成。**

- **链接: [http://arxiv.org/pdf/2508.04714v1](http://arxiv.org/pdf/2508.04714v1)**

> **作者:** Chitranshu Harbola; Anupam Purwar
>
> **摘要:** Industrial machinery maintenance requires timely intervention to prevent catastrophic failures and optimize operational efficiency. This paper presents an integrated Large Language Model (LLM)-based intelligent system for prescriptive maintenance that extends beyond traditional anomaly detection to provide actionable maintenance recommendations. Building upon our prior LAMP framework for numerical data analysis, we develop a comprehensive solution that combines bearing vibration frequency analysis with multi agentic generation for intelligent maintenance planning. Our approach serializes bearing vibration data (BPFO, BPFI, BSF, FTF frequencies) into natural language for LLM processing, enabling few-shot anomaly detection with high accuracy. The system classifies fault types (inner race, outer race, ball/roller, cage faults) and assesses severity levels. A multi-agentic component processes maintenance manuals using vector embeddings and semantic search, while also conducting web searches to retrieve comprehensive procedural knowledge and access up-to-date maintenance practices for more accurate and in-depth recommendations. The Gemini model then generates structured maintenance recommendations includes immediate actions, inspection checklists, corrective measures, parts requirements, and timeline specifications. Experimental validation in bearing vibration datasets demonstrates effective anomaly detection and contextually relevant maintenance guidance. The system successfully bridges the gap between condition monitoring and actionable maintenance planning, providing industrial practitioners with intelligent decision support. This work advances the application of LLMs in industrial maintenance, offering a scalable framework for prescriptive maintenance across machinery components and industrial sectors.
>
---
#### [new 013] RAP: Real-time Audio-driven Portrait Animation with Video Diffusion Transformer
- **分类: cs.GR; cs.CV; cs.SD; eess.AS**

- **简介: 该论文旨在解决实时音频驱动人像动画（RAP）的问题，通过引入混合注意力机制和静态-动态训练策略，实现了高效且高质量的实时合成。任务为生成逼真的人像视频，目标是平衡音频控制与视觉细节保留。**

- **链接: [http://arxiv.org/pdf/2508.05115v1](http://arxiv.org/pdf/2508.05115v1)**

> **作者:** Fangyu Du; Taiqing Li; Ziwei Zhang; Qian Qiao; Tan Yu; Dingcheng Zhen; Xu Jia; Yang Yang; Shunshun Yin; Siyuan Liu
>
> **备注:** 11 pages, 9 figures
>
> **摘要:** Audio-driven portrait animation aims to synthesize realistic and natural talking head videos from an input audio signal and a single reference image. While existing methods achieve high-quality results by leveraging high-dimensional intermediate representations and explicitly modeling motion dynamics, their computational complexity renders them unsuitable for real-time deployment. Real-time inference imposes stringent latency and memory constraints, often necessitating the use of highly compressed latent representations. However, operating in such compact spaces hinders the preservation of fine-grained spatiotemporal details, thereby complicating audio-visual synchronization RAP (Real-time Audio-driven Portrait animation), a unified framework for generating high-quality talking portraits under real-time constraints. Specifically, RAP introduces a hybrid attention mechanism for fine-grained audio control, and a static-dynamic training-inference paradigm that avoids explicit motion supervision. Through these techniques, RAP achieves precise audio-driven control, mitigates long-term temporal drift, and maintains high visual fidelity. Extensive experiments demonstrate that RAP achieves state-of-the-art performance while operating under real-time constraints.
>
---
#### [new 014] Keyword Spotting with Hyper-Matched Filters for Small Footprint Devices
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文旨在解决语音记录中关键词检测的问题，采用超匹配滤波器技术构建模型，通过小型语音编码器和检测网络实现高效泛化，取得小尺寸设备上的最佳性能。**

- **链接: [http://arxiv.org/pdf/2508.04857v1](http://arxiv.org/pdf/2508.04857v1)**

> **作者:** Yael Segal-Feldman; Ann R. Bradlow; Matthew Goldrick; Joseph Keshet
>
> **备注:** pre-print
>
> **摘要:** Open-vocabulary keyword spotting (KWS) refers to the task of detecting words or terms within speech recordings, regardless of whether they were included in the training data. This paper introduces an open-vocabulary keyword spotting model with state-of-the-art detection accuracy for small-footprint devices. The model is composed of a speech encoder, a target keyword encoder, and a detection network. The speech encoder is either a tiny Whisper or a tiny Conformer. The target keyword encoder is implemented as a hyper-network that takes the desired keyword as a character string and generates a unique set of weights for a convolutional layer, which can be considered as a keyword-specific matched filter. The detection network uses the matched-filter weights to perform a keyword-specific convolution, which guides the cross-attention mechanism of a Perceiver module in determining whether the target term appears in the recording. The results indicate that our system achieves state-of-the-art detection performance and generalizes effectively to out-of-domain conditions, including second-language (L2) speech. Notably, our smallest model, with just 4.2 million parameters, matches or outperforms models that are several times larger, demonstrating both efficiency and robustness.
>
---
## 更新

#### [replaced 001] ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.13053v3](http://arxiv.org/pdf/2506.13053v3)**

> **作者:** Han Zhu; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhaoqing Li; Weiji Zhuang; Long Lin; Daniel Povey
>
> **备注:** Accepted in ASRU 2025
>
> **摘要:** Existing large-scale zero-shot text-to-speech (TTS) models deliver high speech quality but suffer from slow inference speeds due to massive parameters. To address this issue, this paper introduces ZipVoice, a high-quality flow-matching-based zero-shot TTS model with a compact model size and fast inference speed. Key designs include: 1) a Zipformer-based vector field estimator to maintain adequate modeling capabilities under constrained size; 2) Average upsampling-based initial speech-text alignment and Zipformer-based text encoder to improve speech intelligibility; 3) A flow distillation method to reduce sampling steps and eliminate the inference overhead associated with classifier-free guidance. Experiments on 100k hours multilingual datasets show that ZipVoice matches state-of-the-art models in speech quality, while being 3 times smaller and up to 30 times faster than a DiT-based flow-matching baseline. Codes, model checkpoints and demo samples are publicly available at https://github.com/k2-fsa/ZipVoice.
>
---
#### [replaced 002] PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2309.02265v2](http://arxiv.org/pdf/2309.02265v2)**

> **作者:** Alain Riou; Stefan Lattner; Gaëtan Hadjeres; Geoffroy Peeters
>
> **备注:** Best Paper Award of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023
>
> **摘要:** In this paper, we address the problem of pitch estimation using Self Supervised Learning (SSL). The SSL paradigm we use is equivariance to pitch transposition, which enables our model to accurately perform pitch estimation on monophonic audio after being trained only on a small unlabeled dataset. We use a lightweight ($<$ 30k parameters) Siamese neural network that takes as inputs two different pitch-shifted versions of the same audio represented by its Constant-Q Transform. To prevent the model from collapsing in an encoder-only setting, we propose a novel class-based transposition-equivariant objective which captures pitch information. Furthermore, we design the architecture of our network to be transposition-preserving by introducing learnable Toeplitz matrices. We evaluate our model for the two tasks of singing voice and musical instrument pitch estimation and show that our model is able to generalize across tasks and datasets while being lightweight, hence remaining compatible with low-resource devices and suitable for real-time applications. In particular, our results surpass self-supervised baselines and narrow the performance gap between self-supervised and supervised methods for pitch estimation.
>
---
#### [replaced 003] Recent Advances in Speech Language Models: A Survey
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.03751v4](http://arxiv.org/pdf/2410.03751v4)**

> **作者:** Wenqian Cui; Dianzhi Yu; Xiaoqi Jiao; Ziqiao Meng; Guangyan Zhang; Qichao Wang; Yiwen Guo; Irwin King
>
> **备注:** The reduced version of this paper has been accepted at ACL 2025
>
> **摘要:** Large Language Models (LLMs) have recently garnered significant attention, primarily for their capabilities in text-based interactions. However, natural human interaction often relies on speech, necessitating a shift towards voice-based models. A straightforward approach to achieve this involves a pipeline of ``Automatic Speech Recognition (ASR) + LLM + Text-to-Speech (TTS)", where input speech is transcribed to text, processed by an LLM, and then converted back to speech. Despite being straightforward, this method suffers from inherent limitations, such as information loss during modality conversion, significant latency due to the complex pipeline, and error accumulation across the three stages. To address these issues, Speech Language Models (SpeechLMs) -- end-to-end models that generate speech without converting from text -- have emerged as a promising alternative. This survey paper provides the first comprehensive overview of recent methodologies for constructing SpeechLMs, detailing the key components of their architecture and the various training recipes integral to their development. Additionally, we systematically survey the various capabilities of SpeechLMs, categorize their evaluation metrics, and discuss the challenges and future research directions in this rapidly evolving field. The GitHub repository is available at https://github.com/dreamtheater123/Awesome-SpeechLM-Survey
>
---
#### [replaced 004] Towards Reliable Audio Deepfake Attribution and Model Recognition: A Multi-Level Autoencoder-Based Framework
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02521v2](http://arxiv.org/pdf/2508.02521v2)**

> **作者:** Andrea Di Pierno; Luca Guarnera; Dario Allegra; Sebastiano Battiato
>
> **摘要:** The proliferation of audio deepfakes poses a growing threat to trust in digital communications. While detection methods have advanced, attributing audio deepfakes to their source models remains an underexplored yet crucial challenge. In this paper we introduce LAVA (Layered Architecture for Voice Attribution), a hierarchical framework for audio deepfake detection and model recognition that leverages attention-enhanced latent representations extracted by a convolutional autoencoder trained solely on fake audio. Two specialized classifiers operate on these features: Audio Deepfake Attribution (ADA), which identifies the generation technology, and Audio Deepfake Model Recognition (ADMR), which recognize the specific generative model instance. To improve robustness under open-set conditions, we incorporate confidence-based rejection thresholds. Experiments on ASVspoof2021, FakeOrReal, and CodecFake show strong performance: the ADA classifier achieves F1-scores over 95% across all datasets, and the ADMR module reaches 96.31% macro F1 across six classes. Additional tests on unseen attacks from ASVpoof2019 LA and error propagation analysis confirm LAVA's robustness and reliability. The framework advances the field by introducing a supervised approach to deepfake attribution and model recognition under open-set conditions, validated on public benchmarks and accompanied by publicly released models and code. Models and code are available at https://www.github.com/adipiz99/lava-framework.
>
---
#### [replaced 005] Video Soundtrack Generation by Aligning Emotions and Temporal Boundaries
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.10154v2](http://arxiv.org/pdf/2502.10154v2)**

> **作者:** Serkan Sulun; Paula Viana; Matthew E. P. Davies
>
> **摘要:** We introduce EMSYNC, a video-based symbolic music generation model that aligns music with a video's emotional content and temporal boundaries. It follows a two-stage framework, where a pretrained video emotion classifier extracts emotional features, and a conditional music generator produces MIDI sequences guided by both emotional and temporal cues. We introduce boundary offsets, a novel temporal conditioning mechanism that enables the model to anticipate and align musical chords with scene cuts. Unlike existing models, our approach retains event-based encoding, ensuring fine-grained timing control and expressive musical nuances. We also propose a mapping scheme to bridge the video emotion classifier, which produces discrete emotion categories, with the emotion-conditioned MIDI generator, which operates on continuous-valued valence-arousal inputs. In subjective listening tests, EMSYNC outperforms state-of-the-art models across all subjective metrics, for music theory-aware participants as well as the general listeners.
>
---
#### [replaced 006] AudioGen-Omni: A Unified Multimodal Diffusion Transformer for Video-Synchronized Audio, Speech, and Song Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.00733v4](http://arxiv.org/pdf/2508.00733v4)**

> **作者:** Le Wang; Jun Wang; Chunyu Qiang; Feng Deng; Chen Zhang; Di Zhang; Kun Gai
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** We present AudioGen-Omni - a unified approach based on multimodal diffusion transformers (MMDit), capable of generating high-fidelity audio, speech, and song coherently synchronized with the input video. AudioGen-Omni introduces a novel joint training paradigm that seamlessly integrates large-scale video-text-audio corpora, enabling a model capable of generating semantically rich, acoustically diverse audio conditioned on multimodal inputs and adaptable to a wide range of audio generation tasks. AudioGen-Omni employs a unified lyrics-transcription encoder that encodes graphemes and phonemes from both song and spoken inputs into dense frame-level representations. Dense frame-level representations are fused using an AdaLN-based joint attention mechanism enhanced with phase-aligned anisotropic positional infusion (PAAPI), wherein RoPE is selectively applied to temporally structured modalities to ensure precise and robust cross-modal alignment. By unfreezing all modalities and masking missing inputs, AudioGen-Omni mitigates the semantic constraints of text-frozen paradigms, enabling effective cross-modal conditioning. This joint training approach enhances audio quality, semantic alignment, and lip-sync accuracy, while also achieving state-of-the-art results on Text-to-Audio/Speech/Song tasks. With an inference time of 1.91 seconds for 8 seconds of audio, it offers substantial improvements in both efficiency and generality.
>
---
