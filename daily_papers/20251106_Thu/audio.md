# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Quantifying Articulatory Coordination as a Biomarker for Schizophrenia
- **分类: eess.AS; cs.LG; eess.SP**

- **简介: 该论文提出一种可解释的语音生物标志物方法，利用eigenspectra和WSED量化发音协调性，以区分精神分裂症患者症状严重度及正负症状平衡，突破传统二元诊断局限，提升临床可解释性。**

- **链接: [http://arxiv.org/pdf/2511.03084v1](http://arxiv.org/pdf/2511.03084v1)**

> **作者:** Gowtham Premananth; Carol Espy-Wilson
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Advances in artificial intelligence (AI) and deep learning have improved diagnostic capabilities in healthcare, yet limited interpretability continues to hinder clinical adoption. Schizophrenia, a complex disorder with diverse symptoms including disorganized speech and social withdrawal, demands tools that capture symptom severity and provide clinically meaningful insights beyond binary diagnosis. Here, we present an interpretable framework that leverages articulatory speech features through eigenspectra difference plots and a weighted sum with exponential decay (WSED) to quantify vocal tract coordination. Eigenspectra plots effectively distinguished complex from simpler coordination patterns, and WSED scores reliably separated these groups, with ambiguity confined to a narrow range near zero. Importantly, WSED scores correlated not only with overall BPRS severity but also with the balance between positive and negative symptoms, reflecting more complex coordination in subjects with pronounced positive symptoms and the opposite trend for stronger negative symptoms. This approach offers a transparent, severity-sensitive biomarker for schizophrenia, advancing the potential for clinically interpretable speech-based assessment tools.
>
---
#### [new 002] Why Not Put a Microphone Near the Loudspeaker? A New Paradigm for Acoustic Echo Cancellation
- **分类: cs.SD**

- **简介: 该论文提出一种新型声学回声消除方法，通过在扬声器旁增设辅助麦克风捕获非线性失真信号，结合Wiener滤波抑制近端语音，再用深度神经网络消除残差回声与噪声，显著提升复杂环境下的回声消除效果。**

- **链接: [http://arxiv.org/pdf/2511.03244v1](http://arxiv.org/pdf/2511.03244v1)**

> **作者:** Fei Zhao; Zhong-Qiu Wang
>
> **摘要:** Acoustic echo cancellation (AEC) remains challenging in real-world environments due to nonlinear distortions caused by low-cost loudspeakers and complex room acoustics. To mitigate these issues, we introduce a dual-microphone configuration, where an auxiliary reference microphone is placed near the loudspeaker to capture the nonlinearly distorted far-end signal. Although this reference signal is contaminated by near-end speech, we propose a preprocessing module based on Wiener filtering to estimate a compressed time-frequency mask to suppress near-end components. This purified reference signal enables a more effective linear AEC stage, whose residual error signal is then fed to a deep neural network for joint residual echo and noise suppression. Evaluation results show that our method outperforms baseline approaches on matched test sets. To evaluate its robustness under strong nonlinearities, we further test it on a mismatched dataset and observe that it achieves substantial performance gains. These results demonstrate its effectiveness in practical scenarios where the nonlinear distortions are typically unknown.
>
---
#### [new 003] Speech-Based Prioritization for Schizophrenia Intervention
- **分类: eess.AS; eess.SP**

- **简介: 该论文提出一种基于语音的 schizophrenia 症状严重度排序模型，通过语音特征与Bradley-Terry模型进行 pairwise 比较，解决临床资源有限下优先级排序问题，超越传统回归方法，提升远程筛查与分诊效率。**

- **链接: [http://arxiv.org/pdf/2511.03086v1](http://arxiv.org/pdf/2511.03086v1)**

> **作者:** Gowtham Premananth; Philip Resnik; Sonia Bansal; Deanna L. Kelly; Carol Espy-Wilson
>
> **备注:** Submitted for ICASSP 2026
>
> **摘要:** Millions of people suffer from mental health conditions, yet many remain undiagnosed or receive delayed care due to limited clinical resources and labor-intensive assessment methods. While most machine-assisted approaches focus on diagnostic classification, estimating symptom severity is essential for prioritizing care, particularly in resource-constrained settings. Speech-based AI provides a scalable alternative by enabling automated, continuous, and remote monitoring, reducing reliance on subjective self-reports and time-consuming evaluations. In this paper, we introduce a speech-based model for pairwise comparison of schizophrenia symptom severity, leveraging articulatory and acoustic features. These comparisons are used to generate severity rankings via the Bradley-Terry model. Our approach outperforms previous regression-based models on ranking-based metrics, offering a more effective solution for clinical triage and prioritization.
>
---
#### [new 004] Seeing What You Say: Expressive Image Generation from Speech
- **分类: eess.AS; cs.CV; cs.MM**

- **简介: 论文提出VoxStudio，首个端到端语音到图像生成模型，直接从语音生成富有表现力的图像，通过语音信息瓶颈模块保留语调与情感，避免依赖文本中间表示，并构建了VoxEmoset情感语音图像数据集。**

- **链接: [http://arxiv.org/pdf/2511.03423v1](http://arxiv.org/pdf/2511.03423v1)**

> **作者:** Jiyoung Lee; Song Park; Sanghyuk Chun; Soo-Whan Chung
>
> **备注:** In progress
>
> **摘要:** This paper proposes VoxStudio, the first unified and end-to-end speech-to-image model that generates expressive images directly from spoken descriptions by jointly aligning linguistic and paralinguistic information. At its core is a speech information bottleneck (SIB) module, which compresses raw speech into compact semantic tokens, preserving prosody and emotional nuance. By operating directly on these tokens, VoxStudio eliminates the need for an additional speech-to-text system, which often ignores the hidden details beyond text, e.g., tone or emotion. We also release VoxEmoset, a large-scale paired emotional speech-image dataset built via an advanced TTS engine to affordably generate richly expressive utterances. Comprehensive experiments on the SpokenCOCO, Flickr8kAudio, and VoxEmoset benchmarks demonstrate the feasibility of our method and highlight key challenges, including emotional consistency and linguistic ambiguity, paving the way for future research.
>
---
#### [new 005] Open Source State-Of-the-Art Solution for Romanian Speech Recognition
- **分类: eess.AS; cs.AI**

- **简介: 该论文面向罗马尼亚语语音识别任务，首次将NVIDIA FastConformer架构应用于罗马尼亚语，利用超2600小时弱监督数据，结合CTC与TDT混合解码器，显著降低词错误率（最高27%），实现新SOTA性能并提升解码效率。**

- **链接: [http://arxiv.org/pdf/2511.03361v1](http://arxiv.org/pdf/2511.03361v1)**

> **作者:** Gabriel Pirlogeanu; Alexandru-Lucian Georgescu; Horia Cucu
>
> **备注:** 13th Conference on Speech Technology and Human-Computer Dialogue (SpeD 2025), Cluj-Napoca, Romania
>
> **摘要:** In this work, we present a new state-of-the-art Romanian Automatic Speech Recognition (ASR) system based on NVIDIA's FastConformer architecture--explored here for the first time in the context of Romanian. We train our model on a large corpus of, mostly, weakly supervised transcriptions, totaling over 2,600 hours of speech. Leveraging a hybrid decoder with both Connectionist Temporal Classification (CTC) and Token-Duration Transducer (TDT) branches, we evaluate a range of decoding strategies including greedy, ALSD, and CTC beam search with a 6-gram token-level language model. Our system achieves state-of-the-art performance across all Romanian evaluation benchmarks, including read, spontaneous, and domain-specific speech, with up to 27% relative WER reduction compared to previous best-performing systems. In addition to improved transcription accuracy, our approach demonstrates practical decoding efficiency, making it suitable for both research and deployment in low-latency ASR applications.
>
---
#### [new 006] TASU: Text-Only Alignment for Speech Understanding
- **分类: eess.AS**

- **简介: TASU提出一种仅用文本数据对齐语音理解的全新范式，解决传统方法依赖大规模音文配对数据、泛化差的问题，实现零样本语音识别并提升多任务性能，超越主流语音大模型。**

- **链接: [http://arxiv.org/pdf/2511.03310v1](http://arxiv.org/pdf/2511.03310v1)**

> **作者:** Jing Peng; Yi Yang; Xu Li; Yu Xi; Quanwei Tang; Yangui Fang; Junjie Li; Kai Yu
>
> **备注:** This paper is submitted to ICASSP 2026
>
> **摘要:** Recent advances in Speech Large Language Models (Speech LLMs) have paved the way for unified architectures across diverse speech understanding tasks. However, prevailing alignment paradigms rely heavily on large-scale audio-text paired data and computationally intensive training, yet often exhibit limited generalization to unseen domains or tasks. To address these limitations, we propose TASU (Text-only Alignment for Speech Understanding), a novel alignment paradigm that can leverage only unpaired text data to guide cross-modal alignment. Experiments show that TASU achieves competitive zero-shot speech recognition. Leveraging this property, it can further function as a pre-training stage in curriculum learning, enhancing domain generalization in speech recognition. Ultimately, TASU can extend its zero-shot generalization to a wide range of speech understanding tasks and notably outperforms prominent Speech LLMs including GLM-4-Voice and Step-Audio on the MMSU benchmark, establishing TASU as an efficient and scalable alignment paradigm for Speech LLMs.
>
---
#### [new 007] audio2chart: End to End Audio Transcription into playable Guitar Hero charts
- **分类: eess.AS; cs.SD**

- **简介: 论文提出audio2chart，将原始音频端到端转换为可玩的Guitar Hero谱面，将问题建模为序列预测任务，通过音频条件化显著提升音符预测准确率，并开源代码与预训练模型支持复现。**

- **链接: [http://arxiv.org/pdf/2511.03337v1](http://arxiv.org/pdf/2511.03337v1)**

> **作者:** Riccardo Tripodi
>
> **摘要:** This work introduces audio2chart, a framework for the automatic generation of Guitar Hero style charts directly from raw audio. The task is formalized as a sequence prediction problem, where models are trained to generate discrete chart tokens aligned with the audio on discrete time steps. An unconditional baseline demonstrates strong predictive performance, while the addition of audio conditioning yields consistent improvements across accuracy based metrics. This work demonstrates that incorporating audio conditioning is both feasible and effective for improving note prediction in automatic chart generation. The complete codebase for training and inference is publicly available on GitHub supporting reproducible research on neural chart generation. A family of pretrained models is released on Hugging Face.
>
---
#### [new 008] SyMuPe: Affective and Controllable Symbolic Music Performance
- **分类: cs.SD; cs.LG; cs.MM**

- **简介: 论文提出SyMuPe框架，旨在实现可控、情感化的人工智能钢琴演奏生成。通过PianoFlow模型，结合条件流匹配与情感文本嵌入，解决音乐表现力缺失问题，在生成质量与情感控制上超越基线模型。**

- **链接: [http://arxiv.org/pdf/2511.03425v1](http://arxiv.org/pdf/2511.03425v1)**

> **作者:** Ilya Borovik; Dmitrii Gavrilev; Vladimir Viro
>
> **备注:** ACM Multimedia 2025. Extended version with supplementary material
>
> **摘要:** Emotions are fundamental to the creation and perception of music performances. However, achieving human-like expression and emotion through machine learning models for performance rendering remains a challenging task. In this work, we present SyMuPe, a novel framework for developing and training affective and controllable symbolic piano performance models. Our flagship model, PianoFlow, uses conditional flow matching trained to solve diverse multi-mask performance inpainting tasks. By design, it supports both unconditional generation and infilling of music performance features. For training, we use a curated, cleaned dataset of 2,968 hours of aligned musical scores and expressive MIDI performances. For text and emotion control, we integrate a piano performance emotion classifier and tune PianoFlow with the emotion-weighted Flan-T5 text embeddings provided as conditional inputs. Objective and subjective evaluations against transformer-based baselines and existing models show that PianoFlow not only outperforms other approaches, but also achieves performance quality comparable to that of human-recorded and transcribed MIDI samples. For emotion control, we present and analyze samples generated under different text conditioning scenarios. The developed model can be integrated into interactive applications, contributing to the creation of more accessible and engaging music performance systems.
>
---
#### [new 009] A Computational Approach to Analyzing Disrupted Language in Schizophrenia: Integrating Surprisal and Coherence Measures
- **分类: cs.CL; eess.AS; eess.SP**

- **简介: 该论文属于计算语言学与精神医学交叉任务，旨在通过 surprisal 和语义连贯性量化精神分裂症患者的语言紊乱，区分患者与健康对照，并关联症状严重程度，为客观诊断提供计算指标。**

- **链接: [http://arxiv.org/pdf/2511.03089v1](http://arxiv.org/pdf/2511.03089v1)**

> **作者:** Gowtham Premananth; Carol Espy-Wilson
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Language disruptions are one of the well-known effects of schizophrenia symptoms. They are often manifested as disorganized speech and impaired discourse coherence. These abnormalities in spontaneous language production reflect underlying cognitive disturbances and have the potential to serve as objective markers for symptom severity and diagnosis of schizophrenia. This study focuses on how these language disruptions can be characterized in terms of two computational linguistic measures: surprisal and semantic coherence. By computing surprisal and semantic coherence of language using computational models, this study investigates how they differ between subjects with schizophrenia and healthy controls. Furthermore, this study provides further insight into how language disruptions in terms of these linguistic measures change with varying degrees of schizophrenia symptom severity.
>
---
#### [new 010] Step-Audio-EditX Technical Report
- **分类: cs.CL; cs.AI; cs.HC; cs.SD; eess.AS**

- **简介: Step-Audio-EditX是首个基于LLM的开源音频编辑模型，解决传统音频编辑依赖嵌入先验的问题，仅用大间隔合成数据实现情感、语调等细粒度迭代控制，兼具强零样本TTS能力，性能超越现有模型。**

- **链接: [http://arxiv.org/pdf/2511.03601v1](http://arxiv.org/pdf/2511.03601v1)**

> **作者:** Chao Yan; Boyong Wu; Peng Yang; Pengfei Tan; Guoqiang Hu; Yuxin Zhang; Xiangyu; Zhang; Fei Tian; Xuerui Yang; Xiangyu Zhang; Daxin Jiang; Gang Yu
>
> **摘要:** We present Step-Audio-EditX, the first open-source LLM-based audio model excelling at expressive and iterative audio editing encompassing emotion, speaking style, and paralinguistics alongside robust zero-shot text-to-speech (TTS) capabilities.Our core innovation lies in leveraging only large-margin synthetic data, which circumvents the need for embedding-based priors or auxiliary modules. This large-margin learning approach enables both iterative control and high expressivity across voices, and represents a fundamental pivot from the conventional focus on representation-level disentanglement. Evaluation results demonstrate that Step-Audio-EditX surpasses both MiniMax-2.6-hd and Doubao-Seed-TTS-2.0 in emotion editing and other fine-grained control tasks.
>
---
## 更新

#### [replaced 001] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.21448v3](http://arxiv.org/pdf/2506.21448v3)**

> **作者:** Huadai Liu; Kaicheng Luo; Jialei Wang; Wen Wang; Qian Chen; Zhou Zhao; Wei Xue
>
> **备注:** Accepted by NeurIPS 2025 Main
>
> **摘要:** While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, this generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present ThinkSound, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce AudioCoT, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics, and excels in the out-of-distribution Movie Gen Audio benchmark. The project page is available at https://ThinkSound-Project.github.io.
>
---
#### [replaced 002] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.15176v5](http://arxiv.org/pdf/2408.15176v5)**

> **作者:** Longshen Ou; Jingwei Zhao; Ziyu Wang; Gus Xia; Qihao Liang; Torin Hopkins Ye Wang
>
> **备注:** NeurIPS 2025 camera ready version
>
> **摘要:** We present a unified framework for automatic multitrack music arrangement that enables a single pre-trained symbolic music model to handle diverse arrangement scenarios, including reinterpretation, simplification, and additive generation. At its core is a segment-level reconstruction objective operating on token-level disentangled content and style, allowing for flexible any-to-any instrumentation transformations at inference time. To support track-wise modeling, we introduce REMI-z, a structured tokenization scheme for multitrack symbolic music that enhances modeling efficiency and effectiveness for both arrangement tasks and unconditional generation. Our method outperforms task-specific state-of-the-art models on representative tasks in different arrangement scenarios -- band arrangement, piano reduction, and drum arrangement, in both objective metrics and perceptual evaluations. Taken together, our framework demonstrates strong generality and suggests broader applicability in symbolic music-to-music transformation.
>
---
#### [replaced 003] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05724v3](http://arxiv.org/pdf/2507.05724v3)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **备注:** Accepted in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model Omni-router Transformer. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [replaced 004] Live Music Models
- **分类: cs.SD; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.04651v3](http://arxiv.org/pdf/2508.04651v3)**

> **作者:** Lyria Team; Antoine Caillon; Brian McWilliams; Cassie Tarakajian; Ian Simon; Ilaria Manco; Jesse Engel; Noah Constant; Yunpeng Li; Timo I. Denk; Alberto Lalama; Andrea Agostinelli; Cheng-Zhi Anna Huang; Ethan Manilow; George Brower; Hakan Erdogan; Heidi Lei; Itai Rolnick; Ivan Grishchenko; Manu Orsini; Matej Kastelic; Mauricio Zuluaga; Mauro Verzetti; Michael Dooley; Ondrej Skopek; Rafael Ferrer; Savvas Petridis; Zalán Borsos; Äaron van den Oord; Douglas Eck; Eli Collins; Jason Baldridge; Tom Hume; Chris Donahue; Kehang Han; Adam Roberts
>
> **摘要:** We introduce a new class of generative models for music called live music models that produce a continuous stream of music in real-time with synchronized user control. We release Magenta RealTime, an open-weights live music model that can be steered using text or audio prompts to control acoustic style. On automatic metrics of music quality, Magenta RealTime outperforms other open-weights music generation models, despite using fewer parameters and offering first-of-its-kind live generation capabilities. We also release Lyria RealTime, an API-based model with extended controls, offering access to our most powerful model with wide prompt coverage. These models demonstrate a new paradigm for AI-assisted music creation that emphasizes human-in-the-loop interaction for live music performance.
>
---
#### [replaced 005] MEDIC: Zero-shot Music Editing with Disentangled Inversion Control
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2407.13220v4](http://arxiv.org/pdf/2407.13220v4)**

> **作者:** Huadai Liu; Jialei Wang; Xiangtai Li; Wen Wang; Qian Chen; Rongjie Huang; Yang Liu; Jiayang Xu; Zhou Zhao
>
> **备注:** ACM Multimedia 2025
>
> **摘要:** Text-guided diffusion models revolutionize audio generation by adapting source audio to specific text prompts. However, existing zero-shot audio editing methods such as DDIM inversion accumulate errors across diffusion steps, reducing the effectiveness. Moreover, existing editing methods struggle with conducting complex non-rigid music edits while maintaining content integrity and high fidelity. To address these challenges, we propose MEDIC, a novel zero-shot music editing system based on innovative Disentangled Inversion Control (DIC) technique, which comprises Harmonized Attention Control and Disentangled Inversion. Disentangled Inversion disentangles the diffusion process into triple branches to rectify the deviated path of the source branch caused by DDIM inversion. Harmonized Attention Control unifies the mutual self-attention control and the cross-attention control with an intermediate Harmonic Branch to progressively generate the desired harmonic and melodic information in the target music. We also introduce ZoME-Bench, a comprehensive music editing benchmark with 1,100 samples covering ten distinct editing categories. ZoME-Bench facilitates both zero-shot and instruction-based music editing tasks. Our method outperforms state-of-the-art inversion techniques in editing fidelity and content preservation. The code and benchmark will be released. Audio samples are available at https://medic-edit.github.io/.
>
---
#### [replaced 006] High-Precision Modal Analysis of Multimode Waveguides from Amplitudes via Large-Step Nonconvex Optimization
- **分类: physics.comp-ph; cs.SD; physics.optics**

- **链接: [http://arxiv.org/pdf/2507.12299v2](http://arxiv.org/pdf/2507.12299v2)**

> **作者:** Jingtong Li; Dongting Huang; Minhui Xiong; Mingzhi Li
>
> **摘要:** Optimizing multimodal waveguide performance depends on modal analysis; however, existing methods focus predominantly on modal power distribution (MPD) and, limited by experimental hardware and conditions, exhibit low accuracy, poor adaptability, and high computational cost. This work presents a novel framework for comprehensive modal analysis (recovering both power and relative phase) using aperture field (AF) and far field (FF) amplitude measurements. We formulate the modal analysis as a nonconvex optimization problem under a power-normalization constraint and, inspired by recent advances in deep learning, introduce a large-step strategy to solve it. Our method retrieves both the MPD and the modal relative-phase distribution(MRPD). The effectiveness of the proposed method is validated through visualization of the nonconvex optimization process via its loss landscape. Under noiseless conditions, analysis results of $93$ electromagnetic modes indicate that the relative amplitude accuracy $\mathrm{MRE_{Modulus}}$, and the phase accuracy $\mathrm{MAE_{Phase}}$, both reach the level of machine precision. Through noise simulations of the AF and environmental background, the operational principles of the method are demonstrated under signal-to-noise ratio (SNR) conditions ranging from $10~\mathrm{dB}$ to $60~\mathrm{dB}$. Experiments further confirm that error suppression is effectively achieved by increasing the number of sampling points, thereby maintaining high accuracy and strong robustness. Within a unified evaluation framework, the absolute amplitude error $\mathrm{MAE_{Modulus}}$, and the phase error $\mathrm{MAE_{Phase}}$, are as low as $1.633\times10^{-8}$ and $0$, respectively. The accuracy is significantly superior to existing methods, while also exhibiting higher computational efficiency.
>
---
#### [replaced 007] Listen to Extract: Onset-Prompted Target Speaker Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.05114v2](http://arxiv.org/pdf/2505.05114v2)**

> **作者:** Pengjie Shen; Kangrui Chen; Shulin He; Pengru Chen; Shuqi Yuan; He Kong; Xueliang Zhang; Zhong-Qiu Wang
>
> **备注:** in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** We propose listen to extract (LExt), a highly-effective while extremely-simple algorithm for monaural target speaker extraction (TSE). Given an enrollment utterance of a target speaker, LExt aims at extracting the target speaker from the speaker's mixed speech with other speakers. For each mixture, LExt concatenates an enrollment utterance of the target speaker to the mixture signal at the waveform level, and trains deep neural networks (DNN) to extract the target speech based on the concatenated mixture signal. The rationale is that, this way, an artificial speech onset is created for the target speaker and it could prompt the DNN (a) which speaker is the target to extract; and (b) spectral-temporal patterns of the target speaker that could help extraction. This simple approach produces strong TSE performance on multiple public TSE datasets including WSJ0-2mix, WHAM! and WHAMR!.
>
---
#### [replaced 008] StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction
- **分类: eess.AS; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.18938v2](http://arxiv.org/pdf/2510.18938v2)**

> **作者:** Qianheng Xu
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems.
>
---
