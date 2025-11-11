# 音频 cs.SD;  eess.AS

- **最新发布 27 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] EchoMark: Perceptual Acoustic Environment Transfer with Watermark-Embedded Room Impulse Response
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: EchoMark提出一种可嵌入水印的声学环境迁移框架，解决RIR重建被滥用导致语音伪造的问题，在保持高感知质量（MOS 4.22）同时实现99%以上水印检测准确率。**

- **链接: [http://arxiv.org/pdf/2511.06458v1](http://arxiv.org/pdf/2511.06458v1)**

> **作者:** Chenpei Huang; Lingfeng Yao; Kyu In Lee; Lan Emily Zhang; Xun Chen; Miao Pan
>
> **摘要:** Acoustic Environment Matching (AEM) is the task of transferring clean audio into a target acoustic environment, enabling engaging applications such as audio dubbing and auditory immersive virtual reality (VR). Recovering similar room impulse response (RIR) directly from reverberant speech offers more accessible and flexible AEM solution. However, this capability also introduces vulnerabilities of arbitrary ``relocation" if misused by malicious user, such as facilitating advanced voice spoofing attacks or undermining the authenticity of recorded evidence. To address this issue, we propose EchoMark, the first deep learning-based AEM framework that generates perceptually similar RIRs with embedded watermark. Our design tackle the challenges posed by variable RIR characteristics, such as different durations and energy decays, by operating in the latent domain. By jointly optimizing the model with a perceptual loss for RIR reconstruction and a loss for watermark detection, EchoMark achieves both high-quality environment transfer and reliable watermark recovery. Experiments on diverse datasets validate that EchoMark achieves room acoustic parameter matching performance comparable to FiNS, the state-of-the-art RIR estimator. Furthermore, a high Mean Opinion Score (MOS) of 4.22 out of 5, watermark detection accuracy exceeding 99\%, and bit error rates (BER) below 0.3\% collectively demonstrate the effectiveness of EchoMark in preserving perceptual quality while ensuring reliable watermark embedding.
>
---
#### [new 002] SPUR: A Plug-and-Play Framework for Integrating Spatial Audio Understanding and Reasoning into Large Audio-Language Models
- **分类: eess.AS; cs.AI**

- **简介: SPUR提出一种轻量级插件框架，为单声道音频语言模型注入空间感知能力，通过FOA编码器与空间问答数据集SPUR-Set，实现对声源方向、高度、距离的精准理解与推理。**

- **链接: [http://arxiv.org/pdf/2511.06606v1](http://arxiv.org/pdf/2511.06606v1)**

> **作者:** S Sakshi; Vaibhavi Lokegaonkar; Neil Zhang; Ramani Duraiswami; Sreyan Ghosh; Dinesh Manocha; Lie Lu
>
> **备注:** Project: https://sakshi113.github.io/spur/
>
> **摘要:** Spatial perception is central to auditory intelligence, enabling accurate understanding of real-world acoustic scenes and advancing human-level perception of the world around us. While recent large audio-language models (LALMs) show strong reasoning over complex audios, most operate on monaural inputs and lack the ability to capture spatial cues such as direction, elevation, and distance. We introduce SPUR, a lightweight, plug-in approach that equips LALMs with spatial perception through minimal architectural changes. SPUR consists of: (i) a First-Order Ambisonics (FOA) encoder that maps (W, X, Y, Z) channels to rotation-aware, listener-centric spatial features, integrated into target LALMs via a multimodal adapter; and (ii) SPUR-Set, a spatial QA dataset combining open-source FOA recordings with controlled simulations, emphasizing relative direction, elevation, distance, and overlap for supervised spatial reasoning. Fine-tuning our model on the SPUR-Set consistently improves spatial QA and multi-speaker attribution while preserving general audio understanding. SPUR provides a simple recipe that transforms monaural LALMs into spatially aware models. Extensive ablations validate the effectiveness of our approach.
>
---
#### [new 003] SAR-LM: Symbolic Audio Reasoning with Large Language Models
- **分类: cs.SD**

- **简介: SAR-LM提出一种符号化音频推理框架，将音频转化为可解释的语音、声事件和音乐特征，替代传统稠密嵌入，提升推理透明性与错误可追溯性，在多个基准上实现竞争力性能。**

- **链接: [http://arxiv.org/pdf/2511.06483v1](http://arxiv.org/pdf/2511.06483v1)**

> **作者:** Termeh Taheri; Yinghao Ma; Emmanouil Benetos
>
> **摘要:** Large language models (LLMs) have advanced in text and vision, but their reasoning on audio remains limited. Most existing methods rely on dense audio embeddings, which are difficult to interpret and often fail on structured reasoning tasks. Caption-based approaches, introduced in recent benchmarks such as MMAU, improve performance by translating audio into text, yet still depend on dense embeddings as input, offering little insight when models fail. We present SAR-LM, a symbolic audio reasoning pipeline that builds on this caption-based paradigm by converting audio into structured, human-readable features across speech, sound events, and music. These symbolic inputs support both reasoning and transparent error analysis, enabling us to trace failures to specific features. Across three benchmarks, MMAU, MMAR, and OmniBench, SAR-LM achieves competitive results, while prioritizing interpretability as its primary contribution.
>
---
#### [new 004] ELEGANCE: Efficient LLM Guidance for Audio-Visual Target Speech Extraction
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 论文提出ELEGANCE框架，将大语言模型的 linguistic 知识引入音视频目标语音提取任务，通过三种引导策略提升模型在视觉信息受限、语种未知等挑战场景下的提取性能。**

- **链接: [http://arxiv.org/pdf/2511.06288v1](http://arxiv.org/pdf/2511.06288v1)**

> **作者:** Wenxuan Wu; Shuai Wang; Xixin Wu; Helen Meng; Haizhou Li
>
> **摘要:** Audio-visual target speaker extraction (AV-TSE) models primarily rely on visual cues from the target speaker. However, humans also leverage linguistic knowledge, such as syntactic constraints, next word prediction, and prior knowledge of conversation, to extract target speech. Inspired by this observation, we propose ELEGANCE, a novel framework that incorporates linguistic knowledge from large language models (LLMs) into AV-TSE models through three distinct guidance strategies: output linguistic constraints, intermediate linguistic prediction, and input linguistic prior. Comprehensive experiments with RoBERTa, Qwen3-0.6B, and Qwen3-4B on two AV-TSE backbones demon- strate the effectiveness of our approach. Significant improvements are observed in challenging scenarios, including visual cue impaired, unseen languages, target speaker switches, increased interfering speakers, and out-of-domain test set. Demo page: https://alexwxwu.github.io/ELEGANCE/.
>
---
#### [new 005] Twenty-Five Years of MIR Research: Achievements, Practices, Evaluations, and Future Challenges
- **分类: cs.SD; cs.AI**

- **简介: 该论文为综述性工作，系统回顾了音乐信息检索（MIR）25年的发展历程，梳理了核心成果、关键实践（如基准评测、开放研究、产学联动、多元包容），并展望未来挑战，旨在为领域提供全景式总结与方向指引。**

- **链接: [http://arxiv.org/pdf/2511.07205v1](http://arxiv.org/pdf/2511.07205v1)**

> **作者:** Geoffroy Peeters; Zafar Rafii; Magdalena Fuentes; Zhiyao Duan; Emmanouil Benetos; Juhan Nam; Yuki Mitsufuji
>
> **摘要:** In this paper, we trace the evolution of Music Information Retrieval (MIR) over the past 25 years. While MIR gathers all kinds of research related to music informatics, a large part of it focuses on signal processing techniques for music data, fostering a close relationship with the IEEE Audio and Acoustic Signal Processing Technical Commitee. In this paper, we reflect the main research achievements of MIR along the three EDICS related to music analysis, processing and generation. We then review a set of successful practices that fuel the rapid development of MIR research. One practice is the annual research benchmark, the Music Information Retrieval Evaluation eXchange, where participants compete on a set of research tasks. Another practice is the pursuit of reproducible and open research. The active engagement with industry research and products is another key factor for achieving large societal impacts and motivating younger generations of students to join the field. Last but not the least, the commitment to diversity, equity and inclusion ensures MIR to be a vibrant and open community where various ideas, methodologies, and career pathways collide. We finish by providing future challenges MIR will have to face.
>
---
#### [new 006] Metric Analysis for Spatial Semantic Segmentation of Sound Scenes
- **分类: cs.SD**

- **简介: 该论文研究声音场景的时空语义分割任务，旨在解决现有指标难以综合评估分离与分类性能的问题，提出改进的CA-SDR度量，引入类无关SDR与误差惩罚机制，提升评估准确性。**

- **链接: [http://arxiv.org/pdf/2511.07075v1](http://arxiv.org/pdf/2511.07075v1)**

> **作者:** Mayank Mishra; Paul Magron; Romain Serizel
>
> **备注:** 5 pages; content+bibliography
>
> **摘要:** Spatial semantic segmentation of sound scenes (S5) consists of jointly performing audio source separation and sound event classification from a multichannel audio mixture. To evaluate S5 systems, one can consider two individual metrics, i.e., one for source separation and another for sound event classification, but this approach makes it challenging to compare S5 systems. Thus, a joint class-aware signal-to-distortion ratio (CA-SDR) metric was proposed to evaluate S5 systems. In this work, we first compare the CA-SDR with the classical SDR on scenarios with only classification errors. We then analyze the cases where the metric might not allow proper comparison of the systems. To address this problem, we propose a modified version of the CA-SDR which first focuses on class-agnostic SDR and then accounts for the wrongly labeled sources. We also analyze the performance of the two metrics under cross-contamination between separated audio sources. Finally, we propose a first set of penalties in an attempt to make the metric more reflective of the labeling and separation errors.
>
---
#### [new 007] MT-HuBERT: Self-Supervised Mix-Training for Few-Shot Keyword Spotting in Mixed Speech
- **分类: cs.SD**

- **简介: 论文提出MT-HuBERT，一种自监督预训练框架，用于少样本关键词识别，解决混合语音中多关键词重叠检测难题。通过在预训练中引入混合训练准则，利用无标签数据提升模型在混响与干净语音下的性能。**

- **链接: [http://arxiv.org/pdf/2511.06296v1](http://arxiv.org/pdf/2511.06296v1)**

> **作者:** Junming Yuan; Ying Shi; Dong Wang; Lantian Li; Askar Hamdulla
>
> **摘要:** Few-shot keyword spotting aims to detect previously unseen keywords with very limited labeled samples. A pre-training and adaptation paradigm is typically adopted for this task. While effective in clean conditions, most existing approaches struggle with mixed keyword spotting--detecting multiple overlapping keywords within a single utterance--a capability essential for real-world applications. We have previously proposed a pre-training approach based on Mix-Training (MT) to tackle the mixed keyword detection problem and demonstrated its efficiency. However, this approach is fully supervised, unable to utilize vast unlabeled data. To this end, we propose Mix-Training HuBERT (MT-HuBERT), a self-supervised learning (SSL) pre-training framework that implements the MT criterion during pre-training. MT-HuBERT predicts, in a self-supervised manner, the clean acoustic units of each constituent signal from contextual cues, in contrast to predicting compositional patterns of mixed speech. Experiments conducted on the Google Speech Commands (GSC v2) corpus demonstrate that our proposed MT-HuBERT consistently outperforms several state-of-the-art baselines in few-shot KWS tasks under both mixed and clean conditions.
>
---
#### [new 008] IDMap: A Pseudo-Speaker Generator Framework Based on Speaker Identity Index to Vector Mapping
- **分类: eess.AS**

- **简介: 该论文提出IDMap框架，通过索引到声纹向量的前馈映射生成高唯一性伪说话人，解决现有方法唯一性不足与计算开销大的问题，提升语音匿名化隐私保护效果。**

- **链接: [http://arxiv.org/pdf/2511.06246v1](http://arxiv.org/pdf/2511.06246v1)**

> **作者:** Zeyan Liu; Liping Chen; Kong Aik Lee; Zhenhua Ling
>
> **摘要:** Facilitated by the speech generation framework that disentangles speech into content, speaker, and prosody, voice anonymization is accomplished by substituting the original speaker embedding vector with that of a pseudo-speaker. In this framework, the pseudo-speaker generation forms a fundamental challenge. Current pseudo-speaker generation methods demonstrate limitations in the uniqueness of pseudo-speakers, consequently restricting their effectiveness in voice privacy protection. Besides, existing model-based methods suffer from heavy computation costs. Especially, in the large-scale scenario where a huge number of pseudo-speakers are generated, the limitations of uniqueness and computational inefficiency become more significant. To this end, this paper proposes a framework for pseudo-speaker generation, which establishes a mapping from speaker identity index to speaker vector in the feedforward architecture, termed IDMap. Specifically, the framework is specified into two models: IDMap-MLP and IDMap-Diff. Experiments were conducted on both small- and large-scale evaluation datasets. Small-scale evaluations on the LibriSpeech dataset validated the effectiveness of the proposed IDMap framework in enhancing the uniqueness of pseudo-speakers, thereby improving voice privacy protection, while at a reduced computational cost. Large-scale evaluations on the MLS and Common Voice datasets further justified the superiority of the IDMap framework regarding the stability of the voice privacy protection capability as the number of pseudo-speakers increased. Audio samples and open-source code can be found in https://github.com/VoicePrivacy/IDMap.
>
---
#### [new 009] Generating Novel and Realistic Speakers for Voice Conversion
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SpeakerVAE，用于生成未见过的新型说话人声纹，解决语音转换中目标语音缺失的问题。其轻量级变分自编码器可即插即用，无需修改现有VC模型，即可生成高质量新说话人声音。**

- **链接: [http://arxiv.org/pdf/2511.07135v1](http://arxiv.org/pdf/2511.07135v1)**

> **作者:** Meiying Melissa Chen; Zhenyu Wang; Zhiyao Duan
>
> **摘要:** Voice conversion models modify timbre while preserving paralinguistic features, enabling applications like dubbing and identity protection. However, most VC systems require access to target utterances, limiting their use when target data is unavailable or when users desire conversion to entirely novel, unseen voices. To address this, we introduce a lightweight method SpeakerVAE to generate novel speakers for VC. Our approach uses a deep hierarchical variational autoencoder to model the speaker timbre space. By sampling from the trained model, we generate novel speaker representations for voice synthesis in a VC pipeline. The proposed method is a flexible plug-in module compatible with various VC models, without co-training or fine-tuning of the base VC system. We evaluated our approach with state-of-the-art VC models: FACodec and CosyVoice2. The results demonstrate that our method successfully generates novel, unseen speakers with quality comparable to that of the training speakers.
>
---
#### [new 010] BSCodec: A Band-Split Neural Codec for High-Quality Universal Audio Reconstruction
- **分类: eess.AS**

- **简介: BSCodec提出一种分频带神经音频编解码器，解决传统方法对语音与非语音音频频谱特性差异处理不足的问题，通过独立压缩各频带，在保持语音质量的同时显著提升音乐与环境音重建效果。**

- **链接: [http://arxiv.org/pdf/2511.06150v1](http://arxiv.org/pdf/2511.06150v1)**

> **作者:** Haoran Wang; Jiatong Shi; Jinchuan Tian; Bohan Li; Kai Yu; Shinji Watanabe
>
> **摘要:** Neural audio codecs have recently enabled high-fidelity reconstruction at high compression rates, especially for speech. However, speech and non-speech audio exhibit fundamentally different spectral characteristics: speech energy concentrates in narrow bands around pitch harmonics (80-400 Hz), while non-speech audio requires faithful reproduction across the full spectrum, particularly preserving higher frequencies that define timbre and texture. This poses a challenge: speech-optimized neural codecs suffer degradation on music or sound. Treating the full spectrum holistically is suboptimal: frequency bands have vastly different information density and perceptual importance by content type, yet full-band approaches apply uniform capacity across frequencies without accounting for these acoustic structures. To address this gap, we propose BSCodec (Band-Split Codec), a novel neural audio codec architecture that splits the spectral dimension into separate bands and compresses each band independently. Experimental results demonstrate that BSCodec achieves superior reconstruction over baselines across sound and music, while maintaining competitive quality in the speech domain, when trained on the same combined dataset of speech, music and sound. Downstream benchmark tasks further confirm that BSCodec shows strong potential for use in downstream applications.
>
---
#### [new 011] Omni-AVSR: Towards Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 论文提出Omni-AVSR，一个统一的多模态语音识别框架，整合ASR、VSR与AVSR任务，利用大语言模型与弹性推理，通过多粒度训练和LoRA适配，显著降低资源消耗并提升效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.07253v1](http://arxiv.org/pdf/2511.07253v1)**

> **作者:** Umberto Cappellazzo; Xubo Liu; Pingchuan Ma; Stavros Petridis; Maja Pantic
>
> **备注:** Project website: https://umbertocappellazzo.github.io/Omni-AVSR/
>
> **摘要:** Large language models (LLMs) have recently achieved impressive results in speech recognition across multiple modalities, including Auditory Speech Recognition (ASR), Visual Speech Recognition (VSR), and Audio-Visual Speech Recognition (AVSR). Despite this progress, current LLM-based approaches typically address each task independently, training separate models that raise computational and deployment resource use while missing potential cross-task synergies. They also rely on fixed-rate token compression, which restricts flexibility in balancing accuracy with efficiency. These limitations highlight the need for a unified framework that can support ASR, VSR, and AVSR while enabling elastic inference. To this end, we present Omni-AVSR, a unified audio-visual LLM that combines efficient multi-granularity training with parameter-efficient adaptation. Specifically, we adapt the matryoshka representation learning paradigm to efficiently train across multiple audio and visual granularities, reducing its inherent training resource use. Furthermore, we explore three LoRA-based strategies for adapting the backbone LLM, balancing shared and task-specific specialization. Experiments on LRS2 and LRS3 show that Omni-AVSR achieves comparable or superior accuracy to state-of-the-art baselines while training a single model at substantially lower training and deployment resource use. The model also remains robust under acoustic noise, and we analyze its scaling behavior as LLM size increases, providing insights into the trade-off between performance and efficiency.
>
---
#### [new 012] Factual and Musical Evaluation Metrics for Music Language Models
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文针对音乐语言模型（Music LMs）评估不足的问题，指出传统指标仅衡量语言流畅性而非事实正确性，提出新的音乐领域通用评估指标与事实性评估框架，以准确衡量模型回答的准确性。**

- **链接: [http://arxiv.org/pdf/2511.05550v1](http://arxiv.org/pdf/2511.05550v1)**

> **作者:** Daniel Chenyu Lin; Michael Freeman; John Thickstun
>
> **备注:** 18 pages; first submission
>
> **摘要:** Music language models (Music LMs), like vision language models, leverage multimodal representations to answer natural language queries about musical audio recordings. Although Music LMs are reportedly improving, we find that current evaluations fail to capture whether their answers are correct. Specifically, for all Music LMs that we examine, widely-used evaluation metrics such as BLEU, METEOR, and BERTScore fail to measure anything beyond linguistic fluency of the model's responses. To measure the true performance of Music LMs, we propose (1) a better general-purpose evaluation metric for Music LMs adapted to the music domain and (2) a factual evaluation framework to quantify the correctness of a Music LM's responses. Our framework is agnostic to the modality of the question-answering model and could be generalized to quantify performance in other open-ended question-answering domains. We use open datasets in our experiments and will release all code on publication.
>
---
#### [new 013] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis
- **分类: cs.SD; cs.AI; cs.CR; cs.LG**

- **简介: 论文提出E2E-VGuard，针对基于LLM的端到端语音合成系统，解决语音克隆攻击问题。通过编码器集成、ASR对抗样本和心理声学模型，实现不可感知的音色与发音保护，验证于16个开源系统与3个商业API。**

- **链接: [http://arxiv.org/pdf/2511.07099v1](http://arxiv.org/pdf/2511.07099v1)**

> **作者:** Zhisheng Zhang; Derui Wang; Yifan Mi; Zhiyong Wu; Jie Gao; Yuxin Cao; Kai Ye; Minhui Xue; Jie Hao
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at https://wxzyd123.github.io/e2e-vguard/.
>
---
#### [new 014] Generating Piano Music with Transformers: A Comparative Study of Scale, Data, and Metrics
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究基于Transformer的钢琴音乐生成任务，系统比较数据集、模型规模与训练策略对生成质量的影响，并评估量化指标与人类偏好相关性，最终实现950M参数模型生成的音乐被误认为人类创作。**

- **链接: [http://arxiv.org/pdf/2511.07268v1](http://arxiv.org/pdf/2511.07268v1)**

> **作者:** Jonathan Lehmkuhl; Ábel Ilyés-Kun; Nico Bremes; Cemhan Kaan Özaltan; Frederik Muthers; Jiayi Yuan
>
> **备注:** NeurIPS 2025 Workshop on AI for Music
>
> **摘要:** Although a variety of transformers have been proposed for symbolic music generation in recent years, there is still little comprehensive study on how specific design choices affect the quality of the generated music. In this work, we systematically compare different datasets, model architectures, model sizes, and training strategies for the task of symbolic piano music generation. To support model development and evaluation, we examine a range of quantitative metrics and analyze how well they correlate with human judgment collected through listening studies. Our best-performing model, a 950M-parameter transformer trained on 80K MIDI files from diverse genres, produces outputs that are often rated as human-composed in a Turing-style listening survey.
>
---
#### [new 015] AcousTools: A `Full-Stack', Python-Based, Acoustic Holography Library
- **分类: cs.SD; cs.ET**

- **简介: 论文提出AcousTools，一个全栈式Python声全息库，解决现有工具功能碎片化问题，整合从建模、相位优化到硬件控制的全流程，助力声全息应用开发与研究复现。**

- **链接: [http://arxiv.org/pdf/2511.07336v1](http://arxiv.org/pdf/2511.07336v1)**

> **作者:** Joshua Mukherjee; Giorgos Christopoulos; Zhouyang Shen; Sriram Subramanian; Ryuji Hirayama
>
> **备注:** 14 Pages, 7 Figures, 2 Tables, To be submitted to APL Computational Physics
>
> **摘要:** Acoustic Holography is an emerging field where mid-air ultrasound is controlled and manipulated for novel and exciting applications. These range from mid-air haptics, volumetric displays, contactless fabrication, and even chemical and biomedical applications such as drug delivery. To develop these applications, a software framework to predict acoustic behaviour and simulating resulting effects, such as applied forces or scattering patterns is desirable. There have been various software libraries and platforms that attempt to fill this role, but there is yet to be a single piece of software that acts as a 'full-stack' solution. We define this full-stack as the process from abstraction to physicalisation starting with setup, modelling acoustic propagation, transducer phase retrieval, sound field analysis, and control of the acoustic holographic hardware itself. Existing methods fail to fulfil one or more of these categories. To address this, we present AcousTools, a Python-based acoustic holography library, designed to support the full suite of acoustic holographic applications and we show AcousTools's ability to meet each step of the full-stack's requirements. AcousTools has the potential to become the standard code library for acoustic holography, with the uniquely complete suite of features wrapped in a language that is known to be easy to use, AcousTools will increase the ability for researchers to develop novel applications as well as accurately review other's work. The full-stack, aside from software, will also be useful for researchers - providing a way to view and compare methodologies by understanding where they fit into the stack.
>
---
#### [new 016] Neural Directional Filtering Using a Compact Microphone Array
- **分类: eess.AS**

- **简介: 该论文提出神经定向滤波（NDF），利用深度神经网络从紧凑麦克风阵列中生成虚拟定向麦克风，突破传统波束形成在高频和小阵列下的限制，实现频率不变、可 Steering 的高阶定向模式，显著提升音源捕捉性能。**

- **链接: [http://arxiv.org/pdf/2511.07185v1](http://arxiv.org/pdf/2511.07185v1)**

> **作者:** Weilong Huang; Srikanth Raj Chetupalli; Mhd Modar Halimeh; Oliver Thiergart; Emanuël Habets
>
> **摘要:** Beamforming with desired directivity patterns using compact microphone arrays is essential in many audio applications. Directivity patterns achievable using traditional beamformers depend on the number of microphones and the array aperture. Generally, their effectiveness degrades for compact arrays. To overcome these limitations, we propose a neural directional filtering (NDF) approach that leverages deep neural networks to enable sound capture with a predefined directivity pattern. The NDF computes a single-channel complex mask from the microphone array signals, which is then applied to a reference microphone to produce an output that approximates a virtual directional microphone with the desired directivity pattern. We introduce training strategies and propose data-dependent metrics to evaluate the directivity pattern and directivity factor. We show that the proposed method: i) achieves a frequency-invariant directivity pattern even above the spatial aliasing frequency, ii) can approximate diverse and higher-order patterns, iii) can steer the pattern in different directions, and iv) generalizes to unseen conditions. Lastly, experimental comparisons demonstrate superior performance over conventional beamforming and parametric approaches.
>
---
#### [new 017] We Can Hear You with mmWave Radar! An End-to-End Eavesdropping System
- **分类: cs.SD**

- **简介: 该论文提出mmSpeech系统，利用毫米波雷达非接触式捕获扬声器振动，重建透过墙壁的语音，解决隐私泄露问题。设计神经网络与合成训练管道，实现高保真语音恢复，支持跨说话人泛化。**

- **链接: [http://arxiv.org/pdf/2511.06205v1](http://arxiv.org/pdf/2511.06205v1)**

> **作者:** Dachao Han; Teng Huang; Han Ding; Cui Zhao; Fei Wang; Ge Wang; Wei Xi
>
> **摘要:** With the rise of voice-enabled technologies, loudspeaker playback has become widespread, posing increasing risks to speech privacy. Traditional eavesdropping methods often require invasive access or line-of-sight, limiting their practicality. In this paper, we present mmSpeech, an end-to-end mmWave-based eavesdropping system that reconstructs intelligible speech solely from vibration signals induced by loudspeaker playback, even through walls and without prior knowledge of the speaker. To achieve this, we reveal an optimal combination of vibrating material and radar sampling rate for capturing high- quality vibrations using narrowband mmWave signals. We then design a deep neural network that reconstructs intelligible speech from the estimated noisy spectrograms. To further support downstream speech understanding, we introduce a synthetic training pipeline and selectively fine-tune the encoder of a pre-trained ASR model. We implement mmSpeech with a commercial mmWave radar and validate its performance through extensive experiments. Results show that mmSpeech achieves state-of-the-art speech quality and generalizes well across unseen speakers and various conditions.
>
---
#### [new 018] BridgeVoC: Revitalizing Neural Vocoder from a Restoration Perspective
- **分类: cs.SD**

- **简介: 论文将声码器任务重构为音频恢复问题，提出BridgeVoC扩散模型，利用Schrodinger桥框架与子带感知网络，仅需4步甚至单步推理即可实现SOTA音质，参数更少、效率更高。**

- **链接: [http://arxiv.org/pdf/2511.07116v1](http://arxiv.org/pdf/2511.07116v1)**

> **作者:** Andong Li; Tong Lei; Rilin Chen; Kai Li; Meng Yu; Xiaodong Li; Dong Yu; Chengshi Zheng
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** This paper revisits the neural vocoder task through the lens of audio restoration and propose a novel diffusion vocoder called BridgeVoC. Specifically, by rank analysis, we compare the rank characteristics of Mel-spectrum with other common acoustic degradation factors, and cast the vocoder task as a specialized case of audio restoration, where the range-space spectral (RSS) surrogate of the target spectrum acts as the degraded input. Based on that, we introduce the Schrodinger bridge framework for diffusion modeling, which defines the RSS and target spectrum as dual endpoints of the stochastic generation trajectory. Further, to fully utilize the hierarchical prior of subbands in the time-frequency (T-F) domain, we elaborately devise a novel subband-aware convolutional diffusion network as the data predictor, where subbands are divided following an uneven strategy, and convolutional-style attention module is employed with large kernels for efficient T-F contextual modeling. To enable single-step inference, we propose an omnidirectional distillation loss to facilitate effective information transfer from the teacher model to the student model, and the performance is improved by combining target-related and bijective consistency losses. Comprehensive experiments are conducted on various benchmarks and out-of-distribution datasets. Quantitative and qualitative results show that while enjoying fewer parameters, lower computational cost, and competitive inference speed, the proposed BridgeVoC yields stateof-the-art performance over existing advanced GAN-, DDPMand flow-matching-based baselines with only 4 sampling steps. And consistent superiority is still achieved with single-step inference.
>
---
#### [new 019] Persian Musical Instruments Classification Using Polyphonic Data Augmentation
- **分类: cs.SD; cs.CL**

- **简介: 该论文针对波斯音乐乐器分类任务，解决非西方音乐数据稀缺问题，构建首个波斯传统乐器数据集，并提出文化感知的多声部数据增强方法，结合MERT模型显著提升真实混音场景下的分类性能。**

- **链接: [http://arxiv.org/pdf/2511.05717v1](http://arxiv.org/pdf/2511.05717v1)**

> **作者:** Diba Hadi Esfangereh; Mohammad Hossein Sameti; Sepehr Harfi Moridani; Leili Javidpour; Mahdieh Soleymani Baghshah
>
> **备注:** 9 pages, 2 figures, 4 tables
>
> **摘要:** Musical instrument classification is essential for music information retrieval (MIR) and generative music systems. However, research on non-Western traditions, particularly Persian music, remains limited. We address this gap by introducing a new dataset of isolated recordings covering seven traditional Persian instruments, two common but originally non-Persian instruments (i.e., violin, piano), and vocals. We propose a culturally informed data augmentation strategy that generates realistic polyphonic mixtures from monophonic samples. Using the MERT model (Music undERstanding with large-scale self-supervised Training) with a classification head, we evaluate our approach with out-of-distribution data which was obtained by manually labeling segments of traditional songs. On real-world polyphonic Persian music, the proposed method yielded the best ROC-AUC (0.795), highlighting complementary benefits of tonal and temporal coherence. These results demonstrate the effectiveness of culturally grounded augmentation for robust Persian instrument recognition and provide a foundation for culturally inclusive MIR and diverse music generation systems.
>
---
#### [new 020] Loud-loss: A Perceptually Motivated Loss Function for Speech Enhancement Based on Equal-Loudness Contours
- **分类: cs.SD**

- **简介: 该论文针对语音增强任务中MSE损失忽视人耳感知特性的缺陷，提出基于等响曲线的Loud-loss函数，加权频域误差以匹配听觉敏感度，显著提升感知质量，实验显示WB-PESQ提升显著。**

- **链接: [http://arxiv.org/pdf/2511.05945v1](http://arxiv.org/pdf/2511.05945v1)**

> **作者:** Zixuan Li; Xueliang Zhang; Changjiang Zhao; Shuai Gao; Lei Miao; Zhipeng Yan; Ying Sun; Chong Zhu
>
> **摘要:** The mean squared error (MSE) is a ubiquitous loss function for speech enhancement, but its problem is that the error cannot reflect the auditory perception quality. This is because MSE causes models to over-emphasize low-frequency components which has high energy, leading to the inadequate modeling of perceptually important high-frequency information. To overcome this limitation, we propose a perceptually-weighted loss function grounded in psychoacoustic principles. Specifically, it leverages equal-loudness contours to assign frequency-dependent weights to the reconstruction error, thereby penalizing deviations in a way aligning with human auditory sensitivity. The proposed loss is model-agnostic and flexible, demonstrating strong generality. Experiments on the VoiceBank+DEMAND dataset show that replacing MSE with our loss in a GTCRN model elevates the WB-PESQ score from 2.17 to 2.93-a significant improvement in perceptual quality.
>
---
#### [new 021] TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech
- **分类: cs.HC; cs.MM; cs.SD**

- **简介: 论文提出TalkSketch，一种融合手绘与语音输入的多模态AI绘图系统，解决设计师用文本提示生成创意时打断思维流的问题，支持实时语音驱动的草图迭代，推动AI融入设计过程而非仅输出结果。**

- **链接: [http://arxiv.org/pdf/2511.05817v1](http://arxiv.org/pdf/2511.05817v1)**

> **作者:** Weiyan Shi; Sunaya Upadhyay; Geraldine Quek; Kenny Tsu Wei Choo
>
> **备注:** Accepted at AAAI 2026 Workshop on Creative AI for Live Interactive Performances (CLIP). To be published in Springer CCIS series
>
> **摘要:** Sketching is a widely used medium for generating and exploring early-stage design concepts. While generative AI (GenAI) chatbots are increasingly used for idea generation, designers often struggle to craft effective prompts and find it difficult to express evolving visual concepts through text alone. In the formative study (N=6), we examined how designers use GenAI during ideation, revealing that text-based prompting disrupts creative flow. To address these issues, we developed TalkSketch, an embedded multimodal AI sketching system that integrates freehand drawing with real-time speech input. TalkSketch aims to support a more fluid ideation process through capturing verbal descriptions during sketching and generating context-aware AI responses. Our work highlights the potential of GenAI tools to engage the design process itself rather than focusing on output.
>
---
#### [new 022] Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing with Unified Representation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出Ming-UniAudio，首个统一语音理解、生成与编辑的语音大模型，通过连续音频分词器MingTok-Audio融合语义与声学特征，实现自然语言指令驱动的自由编辑，并构建首个相关评测基准。**

- **链接: [http://arxiv.org/pdf/2511.05516v1](http://arxiv.org/pdf/2511.05516v1)**

> **作者:** Canxiang Yan; Chunxiang Jin; Dawei Huang; Haibing Yu; Han Peng; Hui Zhan; Jie Gao; Jing Peng; Jingdong Chen; Jun Zhou; Kaimeng Ren; Ming Yang; Mingxue Yang; Qiang Xu; Qin Zhao; Ruijie Xiong; Shaoxiong Lin; Xuezhi Wang; Yi Yuan; Yifei Wu; Yongjie Lyu; Zhengyu He; Zhihao Qiu; Zhiqiang Fang; Ziyuan Huang
>
> **备注:** 32 pages, 8 figures
>
> **摘要:** Existing speech models suffer from competing requirements on token representations by understanding and generation tasks. This discrepancy in representation prevents speech language models from performing instruction-based free-form editing. To solve this challenge, we introduce a novel framework that unifies speech understanding, generation, and editing. The core of our unified model is a unified continuous speech tokenizer MingTok-Audio, the first continuous tokenizer to effectively integrate semantic and acoustic features, which makes it suitable for both understanding and generation tasks. Based on this unified continuous audio tokenizer, we developed the speech language model Ming-UniAudio, which achieved a balance between generation and understanding capabilities. Ming-UniAudio sets new state-of-the-art (SOTA) records on 8 out of 12 metrics on the ContextASR benchmark. Notably, for Chinese voice cloning, it achieves a highly competitive Seed-TTS-WER of 0.95. Leveraging this foundational model, we further trained a dedicated speech editing model Ming-UniAudio-Edit, the first speech language model that enables universal, free-form speech editing guided solely by natural language instructions, handling both semantic and acoustic modifications without timestamp condition. To rigorously assess the editing capability and establish a foundation for future research, we introduce Ming-Freeform-Audio-Edit, the first comprehensive benchmark tailored for instruction-based free-form speech editing, featuring diverse scenarios and evaluation dimensions spanning semantic correctness, acoustic quality, and instruction alignment. We open-sourced the continuous audio tokenizer, the unified foundational model, and the free-form instruction-based editing model to facilitate the development of unified audio understanding, generation, and manipulation.
>
---
#### [new 023] MedVoiceBias: A Controlled Study of Audio LLM Behavior in Clinical Decision-Making
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究音频大模型在临床决策中的偏见问题，发现语音特征（年龄、性别、情绪）显著影响医疗建议，导致高达35%的决策差异，揭示音频模态引入的公平性风险，呼吁构建偏见感知架构。**

- **链接: [http://arxiv.org/pdf/2511.06592v1](http://arxiv.org/pdf/2511.06592v1)**

> **作者:** Zhi Rui Tam; Yun-Nung Chen
>
> **摘要:** As large language models transition from text-based interfaces to audio interactions in clinical settings, they might introduce new vulnerabilities through paralinguistic cues in audio. We evaluated these models on 170 clinical cases, each synthesized into speech from 36 distinct voice profiles spanning variations in age, gender, and emotion. Our findings reveal a severe modality bias: surgical recommendations for audio inputs varied by as much as 35% compared to identical text-based inputs, with one model providing 80% fewer recommendations. Further analysis uncovered age disparities of up to 12% between young and elderly voices, which persisted in most models despite chain-of-thought prompting. While explicit reasoning successfully eliminated gender bias, the impact of emotion was not detected due to poor recognition performance. These results demonstrate that audio LLMs are susceptible to making clinical decisions based on a patient's voice characteristics rather than medical evidence, a flaw that risks perpetuating healthcare disparities. We conclude that bias-aware architectures are essential and urgently needed before the clinical deployment of these models.
>
---
#### [new 024] On the Joint Minimization of Regularization Loss Functions in Deep Variational Bayesian Methods for Attribute-Controlled Symbolic Music Generation
- **分类: cs.LG; cs.AI; eess.AS**

- **简介: 该论文研究属性控制的符号音乐生成，解决变分贝叶斯模型中KLD与属性正则化损失难以平衡的问题，提出通过属性变换实现潜在空间的可控性与正则化协同优化。**

- **链接: [http://arxiv.org/pdf/2511.07118v1](http://arxiv.org/pdf/2511.07118v1)**

> **作者:** Matteo Pettenó; Alessandro Ilic Mezza; Alberto Bernardini
>
> **备注:** IEEE Catalog No.: CFP2540S-ART ISBN: 978-9-46-459362-4
>
> **摘要:** Explicit latent variable models provide a flexible yet powerful framework for data synthesis, enabling controlled manipulation of generative factors. With latent variables drawn from a tractable probability density function that can be further constrained, these models enable continuous and semantically rich exploration of the output space by navigating their latent spaces. Structured latent representations are typically obtained through the joint minimization of regularization loss functions. In variational information bottleneck models, reconstruction loss and Kullback-Leibler Divergence (KLD) are often linearly combined with an auxiliary Attribute-Regularization (AR) loss. However, balancing KLD and AR turns out to be a very delicate matter. When KLD dominates over AR, generative models tend to lack controllability; when AR dominates over KLD, the stochastic encoder is encouraged to violate the standard normal prior. We explore this trade-off in the context of symbolic music generation with explicit control over continuous musical attributes. We show that existing approaches struggle to jointly minimize both regularization objectives, whereas suitable attribute transformations can help achieve both controllability and regularization of the target latent dimensions.
>
---
#### [new 025] Who Gets Heard? Rethinking Fairness in AI for Music Systems
- **分类: cs.CY; cs.MM; cs.SD; eess.AS**

- **简介: 该论文聚焦音乐AI系统中的文化与流派偏见问题，指出其对全球南方传统音乐的误现与文化侵蚀，提出从数据集、模型到界面的多层次公平性改进方案，属于AI公平性与文化代表性研究任务。**

- **链接: [http://arxiv.org/pdf/2511.05953v1](http://arxiv.org/pdf/2511.05953v1)**

> **作者:** Atharva Mehta; Shivam Chauhan; Megha Sharma; Gus Xia; Kaustuv Kanti Ganguli; Nishanth Chandran; Zeerak Talat; Monojit Choudhury
>
> **备注:** 7 pages, Accepted at NeurIPS'25 workshop on AI for Music
>
> **摘要:** In recent years, the music research community has examined risks of AI models for music, with generative AI models in particular, raised concerns about copyright, deepfakes, and transparency. In our work, we raise concerns about cultural and genre biases in AI for music systems (music-AI systems) which affect stakeholders including creators, distributors, and listeners shaping representation in AI for music. These biases can misrepresent marginalized traditions, especially from the Global South, producing inauthentic outputs (e.g., distorted ragas) that reduces creators' trust on these systems. Such harms risk reinforcing biases, limiting creativity, and contributing to cultural erasure. To address this, we offer recommendations at dataset, model and interface level in music-AI systems.
>
---
#### [new 026] Conditional Diffusion as Latent Constraints for Controllable Symbolic Music Generation
- **分类: cs.LG; cs.AI; eess.AS**

- **简介: 该论文提出用条件扩散模型作为潜变量约束，实现对符号音乐生成的精细可控。解决传统方法控制粒度不足的问题，首次在多音乐属性上验证其优于正则化方法，兼顾质量与多样性。**

- **链接: [http://arxiv.org/pdf/2511.07156v1](http://arxiv.org/pdf/2511.07156v1)**

> **作者:** Matteo Pettenó; Alessandro Ilic Mezza; Alberto Bernardini
>
> **摘要:** Recent advances in latent diffusion models have demonstrated state-of-the-art performance in high-dimensional time-series data synthesis while providing flexible control through conditioning and guidance. However, existing methodologies primarily rely on musical context or natural language as the main modality of interacting with the generative process, which may not be ideal for expert users who seek precise fader-like control over specific musical attributes. In this work, we explore the application of denoising diffusion processes as plug-and-play latent constraints for unconditional symbolic music generation models. We focus on a framework that leverages a library of small conditional diffusion models operating as implicit probabilistic priors on the latents of a frozen unconditional backbone. While previous studies have explored domain-specific use cases, this work, to the best of our knowledge, is the first to demonstrate the versatility of such an approach across a diverse array of musical attributes, such as note density, pitch range, contour, and rhythm complexity. Our experiments show that diffusion-driven constraints outperform traditional attribute regularization and other latent constraints architectures, achieving significantly stronger correlations between target and generated attributes while maintaining high perceptual quality and diversity.
>
---
#### [new 027] CLiFT-ASR: A Cross-Lingual Fine-Tuning Framework for Low-Resource Taiwanese Hokkien Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出CLiFT-ASR框架，用于低资源台语语音识别，解决单一标注方式信息缺失问题。通过两阶段跨语言微调，先利用台罗拼音学习音调特征，再结合汉字文本学习词汇语法，显著降低字符错误率。**

- **链接: [http://arxiv.org/pdf/2511.06860v1](http://arxiv.org/pdf/2511.06860v1)**

> **作者:** Hung-Yang Sung; Chien-Chun Wang; Kuan-Tang Huang; Tien-Hong Lo; Yu-Sheng Tsao; Yung-Chang Hsu; Berlin Chen
>
> **备注:** Accepted for an oral presentation at the 37th Conference on Computational Linguistics and Speech Processing (ROCLING 2025)
>
> **摘要:** Automatic speech recognition (ASR) for low-resource languages such as Taiwanese Hokkien is difficult due to the scarcity of annotated data. However, direct fine-tuning on Han-character transcriptions often fails to capture detailed phonetic and tonal cues, while training only on romanization lacks lexical and syntactic coverage. In addition, prior studies have rarely explored staged strategies that integrate both annotation types. To address this gap, we present CLiFT-ASR, a cross-lingual fine-tuning framework that builds on Mandarin HuBERT models and progressively adapts them to Taiwanese Hokkien. The framework employs a two-stage process in which it first learns acoustic and tonal representations from phonetic Tai-lo annotations and then captures vocabulary and syntax from Han-character transcriptions. This progressive adaptation enables effective alignment between speech sounds and orthographic structures. Experiments on the TAT-MOE corpus demonstrate that CLiFT-ASR achieves a 24.88\% relative reduction in character error rate (CER) compared with strong baselines. The results indicate that CLiFT-ASR provides an effective and parameter-efficient solution for Taiwanese Hokkien ASR and that it has potential to benefit other low-resource language scenarios.
>
---
## 更新

#### [replaced 001] WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.23238v3](http://arxiv.org/pdf/2509.23238v3)**

> **作者:** Goksenin Yuksel; Pierre Guetschel; Michael Tangermann; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.
>
---
#### [replaced 002] TTSOps: A Closed-Loop Corpus Optimization Framework for Training Multi-Speaker TTS Models from Dark Data
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15614v2](http://arxiv.org/pdf/2506.15614v2)**

> **作者:** Kentaro Seki; Shinnosuke Takamichi; Takaaki Saeki; Hiroshi Saruwatari
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** This paper presents TTSOps, a fully automated closed-loop framework for constructing multi-speaker text-to-speech (TTS) systems from noisy, uncurated web-scale speech data, often referred to as ``dark data,'' such as online videos. Conventional TTS training pipelines require well-curated corpora with high acoustic quality and accurate text-speech alignment, which severely limits scalability, speaker diversity, and real-world applicability. While recent studies have proposed acoustic-quality-based data selection techniques, they often overlook two critical aspects: (1) the inherent robustness of modern TTS models to noise, and (2) the potential contribution of perceptually low-quality yet informative samples. To address these issues, TTSOps introduces a data-centric training pipeline that integrates three core components: (1) automated data collection from dark data sources, (2) utterance-level dynamic selection of data cleansing methods based on training data quality, and (3) evaluation-in-the-loop data selection using automatically predicted mean opinion scores (MOS) to estimate each utterance's impact on model performance. Furthermore, TTSOps jointly optimizes the corpus and the TTS model in a closed-loop framework by dynamically adapting both data selection and data cleansing processes to the characteristics of the target TTS model. Extensive experiments on Japanese YouTube data demonstrate that TTSOps outperforms conventional acoustic-quality-based baselines in both the naturalness and speaker diversity of synthesized speech.
>
---
#### [replaced 003] DIFFA: Large Language Diffusion Models Can Listen and Understand
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.18452v3](http://arxiv.org/pdf/2507.18452v3)**

> **作者:** Jiaming Zhou; Hongjie Chen; Shiwan Zhao; Jian Kang; Jie Li; Enzhi Wang; Yujie Guo; Haoqin Sun; Hui Wang; Aobo Kong; Yong Qin; Xuelong Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Recent advances in large language models (LLMs) have shown remarkable capabilities across textual and multimodal domains. In parallel, diffusion-based language models have emerged as a promising alternative to the autoregressive paradigm, offering improved controllability, bidirectional context modeling, and robust generation. However, their application to the audio modality remains underexplored. In this work, we introduce \textbf{DIFFA}, the first diffusion-based large audio-language model designed to perform spoken language understanding. DIFFA integrates a frozen diffusion language model with a lightweight dual-adapter architecture that bridges speech understanding and natural language reasoning. We employ a two-stage training pipeline: first, aligning semantic representations via an ASR objective; then, learning instruction-following abilities through synthetic audio-caption pairs automatically generated by prompting LLMs. Despite being trained on only 960 hours of ASR and 127 hours of synthetic instruction data, DIFFA demonstrates competitive performance on major benchmarks, including MMSU, MMAU, and VoiceBench, outperforming several autoregressive open-source baselines. Our results reveal the potential of diffusion-based language models for efficient and scalable audio understanding, opening a new direction for speech-driven AI. Our code will be available at https://github.com/NKU-HLT/DIFFA.git.
>
---
#### [replaced 004] Privacy in Speech Technology
- **分类: eess.AS; cs.CR; cs.SD**

- **链接: [http://arxiv.org/pdf/2305.05227v4](http://arxiv.org/pdf/2305.05227v4)**

> **作者:** Tom Bäckström
>
> **摘要:** Speech technology for communication, accessing information, and services has rapidly improved in quality. It is convenient and appealing because speech is the primary mode of communication for humans. Such technology, however, also presents proven threats to privacy. Speech is a tool for communication and it will thus inherently contain private information. Importantly, it however also contains a wealth of side information, such as information related to health, emotions, affiliations, and relationships, all of which are private. Exposing such private information can lead to serious threats such as price gouging, harassment, extortion, and stalking. This paper is a tutorial on privacy issues related to speech technology, modeling their threats, approaches for protecting users' privacy, measuring the performance of privacy-protecting methods, perception of privacy as well as societal and legal consequences. In addition to a tutorial overview, it also presents lines for further development where improvements are most urgently needed.
>
---
#### [replaced 005] How Does a Deep Neural Network Look at Lexical Stress?
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.07229v2](http://arxiv.org/pdf/2508.07229v2)**

> **作者:** Itai Allouche; Itay Asael; Rotem Rousso; Vered Dassa; Ann Bradlow; Seung-Eun Kim; Matthew Goldrick; Joseph Keshet
>
> **备注:** 11 pages, 5 figures, submitted to the Journal of the Acoustical Society of America (JASA)
>
> **摘要:** Despite their success in speech processing, neural networks often operate as black boxes, prompting the question: what informs their decisions, and how can we interpret them? This work examines this issue in the context of lexical stress. A dataset of English disyllabic words was automatically constructed from read and spontaneous speech. Several Convolutional Neural Network (CNN) architectures were trained to predict stress position from a spectrographic representation of disyllabic words lacking minimal stress pairs (e.g., initial stress WAllet, final stress exTEND), achieving up to 92% accuracy on held-out test data. Layerwise Relevance Propagation (LRP), a technique for CNN interpretability analysis, revealed that predictions for held-out minimal pairs (PROtest vs. proTEST ) were most strongly influenced by information in stressed versus unstressed syllables, particularly the spectral properties of stressed vowels. However, the classifiers also attended to information throughout the word. A feature-specific relevance analysis is proposed, and its results suggest that our best-performing classifier is strongly influenced by the stressed vowel's first and second formants, with some evidence that its pitch and third formant also contribute. These results reveal deep learning's ability to acquire distributed cues to stress from naturally occurring data, extending traditional phonetic work based around highly controlled stimuli.
>
---
#### [replaced 006] Describe Where You Are: Improving Noise-Robustness for Speech Emotion Recognition with Text Description of the Environment
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.17716v2](http://arxiv.org/pdf/2407.17716v2)**

> **作者:** Seong-Gyun Leem; Daniel Fulford; Jukka-Pekka Onnela; David Gard; Carlos Busso
>
> **摘要:** Speech emotion recognition (SER) systems often struggle in real-world environments, where ambient noise severely degrades their performance. This paper explores a novel approach that exploits prior knowledge of testing environments to maximize SER performance under noisy conditions. To address this task, we propose a text-guided, environment-aware training where an SER model is trained with contaminated speech samples and their paired noise description. We use a pre-trained text encoder to extract the text-based environment embedding and then fuse it to a transformer-based SER model during training and inference. We demonstrate the effectiveness of our approach through our experiment with the MSP-Podcast corpus and real-world additive noise samples collected from the Freesound and DEMAND repositories. Our experiment indicates that the text-based environment descriptions processed by a large language model (LLM) produce representations that improve the noise-robustness of the SER system. With a contrastive learning (CL)-based representation, our proposed method can be improved by jointly fine-tuning the text encoder with the emotion recognition model. Under the -5dB signal-to-noise ratio (SNR) level, fine-tuning the text encoder improves our CL-based representation method by 76.4% (arousal), 100.0% (dominance), and 27.7% (valence).
>
---
#### [replaced 007] Bridging the Gap between Continuous and Informative Discrete Representations by Random Product Quantization
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2504.04721v2](http://arxiv.org/pdf/2504.04721v2)**

> **作者:** Xueqing Li; Hao Ma; Zehan Li; Rujin Chen; Boyu Zhu; Ruihao Jing; Jian Kang; Jie Li; Chi Zhang; Xiao-Lei Zhang; Xuelong Li
>
> **摘要:** Self-supervised learning (SSL) has become a core technique in speech processing, but the high dimensionality of its representations makes discretization essential for improving efficiency. However, existing discretization methods still suffer from significant information loss, resulting in a notable performance gap compared to continuous representations. To overcome these limitations, we propose two quantization-based discretization methods: Product Quantization (PQ) and Random Product Quantization (RPQ). PQ partitions the original feature space into multiple subspaces and independently quantizes each sub-vector, producing a fused set of discrete units that retain diverse information from different subspaces, thereby mitigating the loss associated with single-cluster quantization. RPQ further enhances representation diversity by randomly sampling a fixed proportion of feature dimensions multiple times to construct sub-vectors, thereby better capturing the variability in the data distribution. Theoretical analysis shows that RPQ reduces the correlation coefficient rho (where 0 <= rho <= 1) between sub-quantizers. Its quantization error is lower-bounded by the product of rho and epsilon-kms, where epsilon-kms denotes the quantization error of a single K-means quantizer. Experimental results on a combined dataset built from LibriSpeech and ML-SUPERB show that PQ and RPQ outperform standard K-means discretization, achieving relative improvements of 21.8 percent and 20.0 percent in WER on LibriSpeech, and 24.1 percent and 19.6 percent in CER on ML-SUPERB, respectively. Moreover, their performance is competitive with, and in some cases even surpasses, that of continuous SSL representations.
>
---
#### [replaced 008] Hybrid Pruning: In-Situ Compression of Self-Supervised Speech Models for Speaker Verification and Anti-Spoofing
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2508.16232v2](http://arxiv.org/pdf/2508.16232v2)**

> **作者:** Junyi Peng; Lin Zhang; Jiangyu Han; Oldřich Plchot; Johan Rohdin; Themos Stafylakis; Shuai Wang; Jan Černocký
>
> **摘要:** Although large-scale self-supervised learning (SSL) models like WavLM have achieved state-of-the-art performance in speech processing, their significant size impedes deployment on resource-constrained devices. While structured pruning is a key technique for model compression, existing methods typically separate it from task-specific fine-tuning. This multi-stage approach struggles to create optimal architectures tailored for diverse downstream tasks. In this work, we introduce a unified framework that integrates structured pruning into the downstream fine-tuning process. Our framework unifies these steps, jointly optimizing for task performance and model sparsity in a single stage. This allows the model to learn a compressed architecture specifically for the end task, eliminating the need for complex multi-stage pipelines and knowledge distillation. Our pruned models achieve up to a 70\% parameter reduction with negligible performance degradation on large-scale datasets, achieving equal error rates of 0.7\%, 0.8\%, and 1.6\% on Vox1-O, -E, and -H, respectively. Furthermore, our approach demonstrates improved generalization in low-resource scenarios, reducing overfitting and achieving a state-of-the-art 3.7\% EER on ASVspoof5.
>
---
#### [replaced 009] Compositional Phoneme Approximation for L1-Grounded L2 Pronunciation Training
- **分类: cs.CL; cs.SD; eess.AS; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.10927v5](http://arxiv.org/pdf/2411.10927v5)**

> **作者:** Jisang Park; Minu Kim; DaYoung Hong; Jongha Lee
>
> **备注:** Accepted to IJCNLP-AACL 2025
>
> **摘要:** Learners of a second language (L2) often map non-native phonemes to similar native-language (L1) phonemes, making conventional L2-focused training slow and effortful. To address this, we propose an L1-grounded pronunciation training method based on compositional phoneme approximation (CPA), a feature-based representation technique that approximates L2 sounds with sequences of L1 phonemes. Evaluations with 20 Korean non-native English speakers show that CPA-based training achieves a 76% in-box formant rate in acoustic analysis, 17.6% relative improvement in phoneme recognition accuracy, and over 80% of speech being rated as more native-like, with minimal training. Project page: https://gsanpark.github.io/CPA-Pronunciation.
>
---
#### [replaced 010] From Generation to Attribution: Music AI Agent Architectures for the Post-Streaming Era
- **分类: cs.IR; cs.HC; cs.MA; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.20276v2](http://arxiv.org/pdf/2510.20276v2)**

> **作者:** Wonil Kim; Hyeongseok Wi; Seungsoon Park; Taejun Kim; Sangeun Keum; Keunhyoung Kim; Taewan Kim; Jongmin Jung; Taehyoung Kim; Gaetan Guerrero; Mael Le Goff; Julie Po; Dongjoo Moon; Juhan Nam; Jongpil Lee
>
> **备注:** Accepted to the NeurIPS 2025 AI4Music Workshop
>
> **摘要:** Generative AI is reshaping music creation, but its rapid growth exposes structural gaps in attribution, rights management, and economic models. Unlike past media shifts, from live performance to recordings, downloads, and streaming, AI transforms the entire lifecycle of music, collapsing boundaries between creation, distribution, and monetization. However, existing streaming systems, with opaque and concentrated royalty flows, are ill-equipped to handle the scale and complexity of AI-driven production. We propose a content-based Music AI Agent architecture that embeds attribution directly into the creative workflow through block-level retrieval and agentic orchestration. Designed for iterative, session-based interaction, the system organizes music into granular components (Blocks) stored in BlockDB; each use triggers an Attribution Layer event for transparent provenance and real-time settlement. This framework reframes AI from a generative tool into infrastructure for a Fair AI Media Platform. By enabling fine-grained attribution, equitable compensation, and participatory engagement, it points toward a post-streaming paradigm where music functions not as a static catalog but as a collaborative and adaptive ecosystem.
>
---
#### [replaced 011] MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.04914v2](http://arxiv.org/pdf/2511.04914v2)**

> **作者:** Hardik B. Sailor; Aw Ai Ti; Chen Fang Yih Nancy; Chiu Ying Lay; Ding Yang; He Yingxu; Jiang Ridong; Li Jingtao; Liao Jingyi; Liu Zhuohan; Lu Yanfeng; Ma Yi; Manas Gupta; Muhammad Huzaifah Bin Md Shahrin; Nabilah Binte Md Johan; Nattadaporn Lertcheva; Pan Chunlei; Pham Minh Duc; Siti Maryam Binte Ahmad Subaidi; Siti Umairah Binte Mohammad Salleh; Sun Shuo; Tarun Kumar Vangani; Wang Qiongqiong; Won Cheng Yi Lewis; Wong Heng Meng Jeremy; Wu Jinyang; Zhang Huayun; Zhang Longyin; Zou Xunlong
>
> **备注:** https://huggingface.co/MERaLiON/MERaLiON-SER-v1
>
> **摘要:** We present MERaLiON-SER, a robust speech emotion recognition model de- signed for English and Southeast Asian languages. The model is trained using a hybrid objective combining weighted categorical cross-entropy and Concordance Correlation Coefficient (CCC) losses for joint discrete and dimensional emotion modelling. This dual approach enables the model to capture both the distinct categories of emotion (like happy or angry) and the fine-grained, such as arousal (intensity), valence (positivity/negativity), and dominance (sense of control), lead- ing to a more comprehensive and robust representation of human affect. Extensive evaluations across multilingual Singaporean languages (English, Chinese, Malay, and Tamil ) and other public benchmarks show that MERaLiON-SER consistently surpasses both open-source speech encoders and large Audio-LLMs. These results underscore the importance of specialised speech-only models for accurate paralin- guistic understanding and cross-lingual generalisation. Furthermore, the proposed framework provides a foundation for integrating emotion-aware perception into future agentic audio systems, enabling more empathetic and contextually adaptive multimodal reasoning.
>
---
#### [replaced 012] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2511.05350v2](http://arxiv.org/pdf/2511.05350v2)**

> **作者:** Mathias Rose Bjare; Giorgia Cantisani; Marco Pasini; Stefan Lattner; Gerhard Widmer
>
> **备注:** Accepted at NeurIPS 2025 - AI for Music Workshop, 11 pages, 5 figures, 1 table
>
> **摘要:** We argue that training autoencoders to reconstruct inputs from noised versions of their encodings, when combined with perceptual losses, yields encodings that are structured according to a perceptual hierarchy. We demonstrate the emergence of this hierarchical structure by showing that, after training an audio autoencoder in this manner, perceptually salient information is captured in coarser representation structures than with conventional training. Furthermore, we show that such perceptual hierarchies improve latent diffusion decoding in the context of estimating surprisal in music pitches and predicting EEG-brain responses to music listening. Pretrained weights are available on github.com/CPJKU/pa-audioic.
>
---
#### [replaced 013] Progressive Facial Granularity Aggregation with Bilateral Attribute-based Enhancement for Face-to-Speech Synthesis
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.07376v3](http://arxiv.org/pdf/2509.07376v3)**

> **作者:** Yejin Jeon; Youngjae Kim; Jihyun Lee; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** EMNLP Findings
>
> **摘要:** For individuals who have experienced traumatic events such as strokes, speech may no longer be a viable means of communication. While text-to-speech (TTS) can be used as a communication aid since it generates synthetic speech, it fails to preserve the user's own voice. As such, face-to-voice (FTV) synthesis, which derives corresponding voices from facial images, provides a promising alternative. However, existing methods rely on pre-trained visual encoders, and finetune them to align with speech embeddings, which strips fine-grained information from facial inputs such as gender or ethnicity, despite their known correlation with vocal traits. Moreover, these pipelines are multi-stage, which requires separate training of multiple components, thus leading to training inefficiency. To address these limitations, we utilize fine-grained facial attribute modeling by decomposing facial images into non-overlapping segments and progressively integrating them into a multi-granular representation. This representation is further refined through multi-task learning of speaker attributes such as gender and ethnicity at both the visual and acoustic domains. Moreover, to improve alignment robustness, we adopt a multi-view training strategy by pairing various visual perspectives of a speaker in terms of different angles and lighting conditions, with identical speech recordings. Extensive subjective and objective evaluations confirm that our approach substantially enhances face-voice congruence and synthesis stability.
>
---
#### [replaced 014] GRAM: Spatial general-purpose audio representation models for real-world applications
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00934v4](http://arxiv.org/pdf/2506.00934v4)**

> **作者:** Goksenin Yuksel; Marcel van Gerven; Kiki van der Heijden
>
> **备注:** Still under review
>
> **摘要:** Although audio foundations models have seen great progress on a wide variety of tasks, their application in real-world acoustic environments with reverberation and noise has been less successful. Moreover, as audio foundation models are typically trained on dry, single-channel audio clips, the inherent spatial nature of real-world sound scenes is overlooked and tasks involving sound localization ruled out. To address these limitations, we propose GRAM: a General-purpose Real-world Audio Model utilizing a multi-channel masked auto-encoder approach to efficiently learn spatial audio representations from high-quality simulated real-world scenes. To evaluate the performance of GRAM and other audio foundation models in real-world sound scenes, we release Nat-HEAR: A naturalistic version of the HEAR benchmark suite comprising a simulated real-world version, as well as two new sound localization tasks. We show that the performance of GRAM surpasses all state-of-the-art self-supervised audio foundation models and speech models on both HEAR and Nat-HEAR, while using only a fraction of the training data. GRAM also showcases state-of-the-art localization performance, surpassing even supervised sound localization approaches, and can be flexibly applied either to a two-channel, binaural sound format or a four-channel, Ambisonics format. Validating GRAM's performance on real-world sound recordings demonstrates robust transfer to real-world scenes. Taken together, GRAM presents a significant advancement towards robust, spatial audio foundation models for real-world applications.
>
---
#### [replaced 015] MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.03546v3](http://arxiv.org/pdf/2504.03546v3)**

> **作者:** Khai Le-Duc; Tuyen Tran; Bach Phan Tat; Nguyen Kim Hai Bui; Quan Dang; Hung-Phong Tran; Thanh-Thuy Nguyen; Ly Nguyen; Tuan-Minh Phan; Thi Thu Phuong Tran; Chris Ngo; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** EMNLP 2025
>
> **摘要:** Multilingual speech translation (ST) and machine translation (MT) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, and Simplified/Traditional Chinese, together with the models. With 290,000 samples, this is the largest medical MT dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most comprehensive ST analysis in the field's history, to our best knowledge, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: https://github.com/leduckhai/MultiMed-ST
>
---
#### [replaced 016] Adaptive Convolution for CNN-based Speech Enhancement Models
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.14224v2](http://arxiv.org/pdf/2502.14224v2)**

> **作者:** Dahan Wang; Xiaobin Rong; Shiruo Sun; Yuxiang Hu; Changbao Zhu; Jing Lu
>
> **备注:** Published in IEEE/ACM Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Deep learning-based speech enhancement methods have significantly improved speech quality and intelligibility. Convolutional neural networks (CNNs) have been proven to be essential components of many high-performance models. In this paper, we introduce adaptive convolution, an efficient and versatile convolutional module that enhances the model's capability to adaptively represent speech signals. Adaptive convolution performs frame-wise causal dynamic convolution, generating time-varying kernels for each frame by assembling multiple parallel candidate kernels. A lightweight attention mechanism is proposed for adaptive convolution, leveraging both current and historical information to assign adaptive weights to each candidate kernel. This enables the convolution operation to adapt to frame-level speech spectral features, leading to more efficient extraction and reconstruction. We integrate adaptive convolution into various CNN-based models, highlighting its generalizability. Experimental results demonstrate that adaptive convolution significantly improves the performance with negligible increases in computational complexity, especially for lightweight models. Moreover, we present an intuitive analysis revealing a strong correlation between kernel selection and signal characteristics. Furthermore, we propose the adaptive convolutional recurrent network (AdaptCRN), an ultra-lightweight model that incorporates adaptive convolution and an efficient encoder-decoder design, achieving superior performance compared to models with similar or even higher computational costs.
>
---
#### [replaced 017] MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.10287v2](http://arxiv.org/pdf/2503.10287v2)**

> **作者:** Hao Zhou; Xiaobao Guo; Yuzhe Zhu; Adams Wai-Kin Kong
>
> **备注:** Accepted at AAAI 2026. Code available at https://github.com/alxzzhou/MACS
>
> **摘要:** Propelled by the breakthrough in deep generative models, audio-to-image generation has emerged as a pivotal cross-modal task that converts complex auditory signals into rich visual representations. However, previous works only focus on single-source audio inputs for image generation, ignoring the multi-source characteristic in natural auditory scenes, thus limiting the performance in generating comprehensive visual content. To bridge this gap, we propose a method called MACS to conduct multi-source audio-to-image generation. To our best knowledge, this is the first work that explicitly separates multi-source audio to capture the rich audio components before image generation. MACS is a two-stage method. In the first stage, multi-source audio inputs are separated by a weakly supervised method, where the audio and text labels are semantically aligned by casting into a common space using the large pre-trained CLAP model. We introduce a ranking loss to consider the contextual significance of the separated audio signals. In the second stage, effective image generation is achieved by mapping the separated audio signals to the generation condition using only a trainable adapter and a MLP layer. We preprocess the LLP dataset as the first full multi-source audio-to-image generation benchmark. The experiments are conducted on multi-source, mixed-source, and single-source audio-to-image generation tasks. The proposed MACS outperforms the current state-of-the-art methods in 17 out of the 21 evaluation indexes on all tasks and delivers superior visual quality.
>
---
