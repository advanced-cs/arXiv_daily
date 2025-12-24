# 音频 cs.SD;  eess.AS

- **最新发布 15 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] SAM Audio: Segment Anything in Audio
- **分类: eess.AS; cs.CV**

- **简介: 该论文提出SAM Audio，面向通用音频源分离任务，解决现有模型领域受限、提示模态单一的问题。工作包括：构建支持文本/视觉/时间跨度多模态提示的扩散Transformer基础模型，用流匹配在大规模多类型音频上训练，并建立新基准与无参考评估模型。**

- **链接: [https://arxiv.org/pdf/2512.18099v1](https://arxiv.org/pdf/2512.18099v1)**

> **作者:** Bowen Shi; Andros Tjandra; John Hoffman; Helin Wang; Yi-Chiao Wu; Luya Gao; Julius Richter; Matt Le; Apoorv Vyas; Sanyuan Chen; Christoph Feichtenhofer; Piotr Dollár; Wei-Ning Hsu; Ann Lee
>
> **摘要:** General audio source separation is a key capability for multimodal AI systems that can perceive and reason about sound. Despite substantial progress in recent years, existing separation models are either domain-specific, designed for fixed categories such as speech or music, or limited in controllability, supporting only a single prompting modality such as text. In this work, we present SAM Audio, a foundation model for general audio separation that unifies text, visual, and temporal span prompting within a single framework. Built on a diffusion transformer architecture, SAM Audio is trained with flow matching on large-scale audio data spanning speech, music, and general sounds, and can flexibly separate target sources described by language, visual masks, or temporal spans. The model achieves state-of-the-art performance across a diverse suite of benchmarks, including general sound, speech, music, and musical instrument separation in both in-the-wild and professionally produced audios, substantially outperforming prior general-purpose and specialized systems. Furthermore, we introduce a new real-world separation benchmark with human-labeled multimodal prompts and a reference-free evaluation model that correlates strongly with human judgment.
>
---
#### [new 002] MMEDIT: A Unified Framework for Multi-Type Audio Editing via Audio Language Model
- **分类: cs.SD**

- **简介: 该论文提出MMEDIT框架，解决文本引导音频编辑中信号降质、数据稀缺及跨模态对齐弱等问题。它统一支持添加、替换、删除等多类编辑任务，构建细粒度配对数据集，并融合Qwen2-Audio与MMDiT实现精准局部编辑与高保真保持。**

- **链接: [https://arxiv.org/pdf/2512.20339v1](https://arxiv.org/pdf/2512.20339v1)**

> **作者:** Ye Tao; Xuenan Xu; Wen Wu; Shuai Wang; Mengyue Wu; Chao Zhang
>
> **备注:** Under review
>
> **摘要:** Text-guided audio editing aims to modify specific acoustic events while strictly preserving non-target content. Despite recent progress, existing approaches remain fundamentally limited. Training-free methods often suffer from signal degradation caused by diffusion inversion, while training-based methods, although achieving higher generation quality, are severely constrained by the scarcity of high-quality paired data and task formulations that cover only a narrow subset of editing operations. In addition, standard architectures typically decouple text and audio processing, limiting the ability to align instructions with specific acoustic contexts. To address these challenges, we propose MMEdit, an audio-language-model-driven framework for unified audio editing. We systematically extend task definitions to cover a comprehensive range of editing operations, including addition, replacement, removal, reordering, and attribute modification. Furthermore, we design a scalable data synthesis pipeline to construct large-scale paired datasets with fine-grained event-level annotations. To capture complex editing semantics, we integrate a Qwen2-Audio encoder with an MMDiT-based generator, enabling precise cross-modal alignment and localized editing. Experimental results demonstrate that our method achieves superior editing localization accuracy, robust instruction following, and high fidelity in non-edited regions.
>
---
#### [new 003] Aliasing-Free Neural Audio Synthesis
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属神经音频合成任务，旨在解决upsampling模型中的混叠失真问题。提出抗混叠激活函数（过采样+反导数）和替代ConvTranspose的重采样方法，构建Pupu-Vocoder/Codec，显著提升歌唱、音乐等音频质量。**

- **链接: [https://arxiv.org/pdf/2512.20211v1](https://arxiv.org/pdf/2512.20211v1)**

> **作者:** Yicheng Gu; Junan Zhang; Chaoren Wang; Jerry Li; Zhizheng Wu; Lauri Juvela
>
> **备注:** Submitted to TASLP
>
> **摘要:** Neural vocoders and codecs reconstruct waveforms from acoustic representations, which directly impact the audio quality. Among existing methods, upsampling-based time-domain models are superior in both inference speed and synthesis quality, achieving state-of-the-art performance. Still, despite their success in producing perceptually natural sound, their synthesis fidelity remains limited due to the aliasing artifacts brought by the inadequately designed model architectures. In particular, the unconstrained nonlinear activation generates an infinite number of harmonics that exceed the Nyquist frequency, resulting in ``folded-back'' aliasing artifacts. The widely used upsampling layer, ConvTranspose, copies the mirrored low-frequency parts to fill the empty high-frequency region, resulting in ``mirrored'' aliasing artifacts. Meanwhile, the combination of its inherent periodicity and the mirrored DC bias also brings ``tonal artifact,'' resulting in constant-frequency ringing. This paper aims to solve these issues from a signal processing perspective. Specifically, we apply oversampling and anti-derivative anti-aliasing to the activation function to obtain its anti-aliased form, and replace the problematic ConvTranspose layer with resampling to avoid the ``tonal artifact'' and eliminate aliased components. Based on our proposed anti-aliased modules, we introduce Pupu-Vocoder and Pupu-Codec, and release high-quality pre-trained checkpoints to facilitate audio generation research. We build a test signal benchmark to illustrate the effectiveness of the anti-aliased modules, and conduct experiments on speech, singing voice, music, and audio to validate our proposed models. Experimental results confirm that our lightweight Pupu-Vocoder and Pupu-Codec models can easily outperform existing systems on singing voice, music, and audio, while achieving comparable performance on speech.
>
---
#### [new 004] AUDRON: A Deep Learning Framework with Fused Acoustic Signatures for Drone Type Recognition
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文提出AUDRON框架，解决无人机声学类型识别任务，旨在通过声音区分无人机与噪声及不同机型。工作包括融合MFCC、STFT谱图与自编码器特征，结合CNN和RNN建模，实现高精度二类（98.51%）和多类（97.11%）识别。**

- **链接: [https://arxiv.org/pdf/2512.20407v1](https://arxiv.org/pdf/2512.20407v1)**

> **作者:** Rajdeep Chatterjee; Sudip Chakrabarty; Trishaani Acharjee; Deepanjali Mishra
>
> **备注:** Presented at the 2025 IEEE 22nd India Council International Conference (INDICON). 6 pages, 3 figures
>
> **摘要:** Unmanned aerial vehicles (UAVs), commonly known as drones, are increasingly used across diverse domains, including logistics, agriculture, surveillance, and defense. While these systems provide numerous benefits, their misuse raises safety and security concerns, making effective detection mechanisms essential. Acoustic sensing offers a low-cost and non-intrusive alternative to vision or radar-based detection, as drone propellers generate distinctive sound patterns. This study introduces AUDRON (AUdio-based Drone Recognition Network), a hybrid deep learning framework for drone sound detection, employing a combination of Mel-Frequency Cepstral Coefficients (MFCC), Short-Time Fourier Transform (STFT) spectrograms processed with convolutional neural networks (CNNs), recurrent layers for temporal modeling, and autoencoder-based representations. Feature-level fusion integrates complementary information before classification. Experimental evaluation demonstrates that AUDRON effectively differentiates drone acoustic signatures from background noise, achieving high accuracy while maintaining generalizability across varying conditions. AUDRON achieves 98.51 percent and 97.11 percent accuracy in binary and multiclass classification. The results highlight the advantage of combining multiple feature representations with deep learning for reliable acoustic drone detection, suggesting the framework's potential for deployment in security and surveillance applications where visual or radar sensing may be limited.
>
---
#### [new 005] LP-CFM: Perceptual Invariance-Aware Conditional Flow Matching for Speech Modeling
- **分类: eess.AS**

- **简介: 该论文属语音生成任务，旨在解决传统流匹配模型忽略语音感知不变性（如幅值缩放、时移）导致建模不鲁棒的问题。提出LP-CFM模型，将目标建模为感知等价变体上的投影对齐高斯分布，并设计VCS采样策略，显著提升低资源和少步长下的神经声码器性能。**

- **链接: [https://arxiv.org/pdf/2512.20314v1](https://arxiv.org/pdf/2512.20314v1)**

> **作者:** Doyeop Kwak; Youngjoon Jang; Joon Son Chung
>
> **摘要:** The goal of this paper is to provide a new perspective on speech modeling by incorporating perceptual invariances such as amplitude scaling and temporal shifts. Conventional generative formulations often treat each dataset sample as a fixed representative of the target distribution. From a generative standpoint, however, such samples are only one among many perceptually equivalent variants within the true speech distribution. To address this, we propose Linear Projection Conditional Flow Matching (LP-CFM), which models targets as projection-aligned elongated Gaussians along perceptually equivalent variants. We further introduce Vector Calibrated Sampling (VCS) to keep the sampling process aligned with the line-projection path. In neural vocoding experiments across model sizes, data scales, and sampling steps, the proposed approach consistently improves over the conventional optimal transport CFM, with particularly strong gains in low-resource and few-step scenarios. These results highlight the potential of LP-CFM and VCS to provide more robust and perceptually grounded generative modeling of speech.
>
---
#### [new 006] Spectral or spatial? Leveraging both for speaker extraction in challenging data conditions
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多通道语音分离任务，旨在解决参考信息（空间/频谱）不准确时的说话人提取问题。提出一种融合空间与频谱线索的鲁棒算法，通过动态加权机制自适应平衡或忽略不可靠线索，在DOA估计偏差和频谱注册噪声下仍保持高性能。**

- **链接: [https://arxiv.org/pdf/2512.20165v1](https://arxiv.org/pdf/2512.20165v1)**

> **作者:** Aviad Eisenberg; Sharon Gannot; Shlomo E. Chazan
>
> **摘要:** This paper presents a robust multi-channel speaker extraction algorithm designed to handle inaccuracies in reference information. While existing approaches often rely solely on either spatial or spectral cues to identify the target speaker, our method integrates both sources of information to enhance robustness. A key aspect of our approach is its emphasis on stability, ensuring reliable performance even when one of the features is degraded or misleading. Given a noisy mixture and two potentially unreliable cues, a dedicated network is trained to dynamically balance their contributions-or disregard the less informative one when necessary. We evaluate the system under challenging conditions by simulating inference-time errors using a simple direction of arrival (DOA) estimator and a noisy spectral enrollment process. Experimental results demonstrate that the proposed model successfully extracts the desired speaker even in the presence of substantial reference inaccuracies.
>
---
#### [new 007] ASK: Adaptive Self-improving Knowledge Framework for Audio Text Retrieval
- **分类: eess.AS; cs.IR; cs.LG; cs.MM; cs.SD**

- **简介: 该论文面向音频-文本检索（ATR）任务，旨在解决对比学习中“梯度局部性瓶颈”（GLB）和知识增强引发的“表征漂移失配”（RDM）双重问题。提出自适应自优化知识（ASK）框架，通过多粒度知识注入、动态知识精炼与可靠性加权机制，实现模型无关、即插即用的性能提升。**

- **链接: [https://arxiv.org/pdf/2512.19703v1](https://arxiv.org/pdf/2512.19703v1)**

> **作者:** Siyuan Fu; Xuchen Guo; Mingjun Liu; Hongxiang Li; Boyin Tan; Gongxi Zhu; Xianwei Zhuang; Jinghan Ru; Yuxin Xie; Yuguo Yin
>
> **摘要:** The dominant paradigm for Audio-Text Retrieval (ATR) relies on mini-batch-based contrastive learning. This process, however, is inherently limited by what we formalize as the Gradient Locality Bottleneck (GLB), which structurally prevents models from leveraging out-of-batch knowledge and thus impairs fine-grained and long-tail learning. While external knowledge-enhanced methods can alleviate the GLB, we identify a critical, unaddressed side effect: the Representation-Drift Mismatch (RDM), where a static knowledge base becomes progressively misaligned with the evolving model, turning guidance into noise. To address this dual challenge, we propose the Adaptive Self-improving Knowledge (ASK) framework, a model-agnostic, plug-and-play solution. ASK breaks the GLB via multi-grained knowledge injection, systematically mitigates RDM through dynamic knowledge refinement, and introduces a novel adaptive reliability weighting scheme to ensure consistent knowledge contributes to optimization. Experimental results on two benchmark datasets with superior, state-of-the-art performance justify the efficacy of our proposed ASK framework.
>
---
#### [new 008] EnvSSLAM-FFN: Lightweight Layer-Fused System for ESDD 2026 Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文面向环境声音深度伪造检测任务，解决 unseen generator 和 black-box 低资源场景下的检测难题。提出轻量级层融合系统 EnvSSLAM-FFN，冻结 SSLAM 编码器，融合第4–9层表征，结合类别加权训练，在两赛道 EER 分别达 1.20% 和 1.05%。**

- **链接: [https://arxiv.org/pdf/2512.20369v1](https://arxiv.org/pdf/2512.20369v1)**

> **作者:** Xiaoxuan Guo; Hengyan Huang; Jiayi Zhou; Renhe Sun; Jian Liu; Haonan Cheng; Long Ye; Qin Zhang
>
> **备注:** ESDD 2026 Challenge Technical Report
>
> **摘要:** Recent advances in generative audio models have enabled high-fidelity environmental sound synthesis, raising serious concerns for audio security. The ESDD 2026 Challenge therefore addresses environmental sound deepfake detection under unseen generators (Track 1) and black-box low-resource detection (Track 2) conditions. We propose EnvSSLAM-FFN, which integrates a frozen SSLAM self-supervised encoder with a lightweight FFN back-end. To effectively capture spoofing artifacts under severe data imbalance, we fuse intermediate SSLAM representations from layers 4-9 and adopt a class-weighted training objective. Experimental results show that the proposed system consistently outperforms the official baselines on both tracks, achieving Test Equal Error Rates (EERs) of 1.20% and 1.05%, respectively.
>
---
#### [new 009] SpatialNet with Binaural Loss Function for Correcting Binaural Signal Matching Outputs under Head Rotations
- **分类: eess.AS**

- **简介: 该论文属音频信号处理任务，旨在解决头戴设备在用户头部旋转时双耳信号匹配失准导致的空间与音色失真问题。提出将SpatialNet深度网络与感知驱动的双耳损失函数结合，对BSM-MagLS方法输出进行后处理校正，在仿真与听音实验中验证了其对大幅旋转的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.20122v1](https://arxiv.org/pdf/2512.20122v1)**

> **作者:** Dor Shamay; Boaz Rafaely
>
> **摘要:** Binaural reproduction is gaining increasing attention with the rise of devices such as virtual reality headsets, smart glasses, and head-tracked headphones. Achieving accurate binaural signals with these systems is challenging, as they often employ arbitrary microphone arrays with limited spatial resolution. The Binaural Signals Matching with Magnitude Least-Squares (BSM-MagLS) method was developed to address limitations of earlier BSM formulations, improving reproduction at high frequencies and under head rotation. However, its accuracy still degrades as head rotation increases, resulting in spatial and timbral artifacts, particularly when the virtual listener's ear moves farther from the nearest microphones. In this work, we propose the integration of deep learning with BSM-MagLS to mitigate these degradations. A post-processing framework based on the SpatialNet network is employed, leveraging its ability to process spatial information effectively and guided by both signal-level loss and a perceptually motivated binaural loss derived from a theoretical model of human binaural hearing. The effectiveness of the approach is investigated in a simulation study with a six-microphone semicircular array, showing its ability to perform robustly across head rotations. These findings are further studied in a listening experiment across different reverberant acoustic environments and head rotations, demonstrating that the proposed framework effectively mitigates BSM-MagLS degradations and provides robust correction across substantial head rotations.
>
---
#### [new 010] QuarkAudio Technical Report
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出QuarkAudio统一音频生成框架，解决多任务模型碎片化、扩展性差问题。工作包括：设计H-Codec离散音频编码器（融合SSL、支持48kHz与动态帧率），构建指令条件化的自回归语言模型，统一支持语音恢复、分离、转换及自然语言驱动的自由音频编辑。**

- **链接: [https://arxiv.org/pdf/2512.20151v1](https://arxiv.org/pdf/2512.20151v1)**

> **作者:** Chengwei Liu; Haoyin Yan; Shaofei Xue; Xiaotao Liang; Xiaofu Chen; Bin Gong; Zheng Xue; Gang Song
>
> **摘要:** Many existing audio processing and generation models rely on task-specific architectures, resulting in fragmented development efforts and limited extensibility. It is therefore promising to design a unified framework capable of handling multiple tasks, while providing robust instruction and audio understanding and high-quality audio generation. This requires a compatible paradigm design, a powerful backbone, and a high-fidelity audio reconstruction module. To meet these requirements, this technical report introduces QuarkAudio, a decoder-only autoregressive (AR) LM-based generative framework that unifies multiple tasks. The framework includes a unified discrete audio tokenizer, H-Codec, which incorporates self-supervised learning (SSL) representations into the tokenization and reconstruction process. We further propose several improvements to H-Codec, such as a dynamic frame-rate mechanism and extending the audio sampling rate to 48 kHz. QuarkAudio unifies tasks by using task-specific conditional information as the conditioning sequence of the decoder-only LM, and predicting discrete target audio tokens in an AR manner. The framework supports a wide range of audio processing and generation tasks, including speech restoration (SR), target speaker extraction (TSE), speech separation (SS), voice conversion (VC), and language-queried audio source separation (LASS). In addition, we extend downstream tasks to universal free-form audio editing guided by natural language instructions (including speech semantic editing and audio event editing). Experimental results show that H-Codec achieves high-quality audio reconstruction with a low frame rate, improving both the efficiency and performance of downstream audio generation, and that QuarkAudio delivers competitive or comparable performance to state-of-the-art task-specific or multi-task systems across multiple tasks.
>
---
#### [new 011] TAVID: Text-Driven Audio-Visual Interactive Dialogue Generation
- **分类: cs.CV; cs.AI; eess.AS; eess.IV**

- **简介: 该论文提出TAVID框架，解决文本驱动的音视频协同对话生成任务，旨在同步生成逼真互动人脸与自然对话语音。通过双向跨模态映射器（运动映射器与说话人映射器）融合视听信息，统一建模对话中的视听交互，提升交互真实性与流畅性。**

- **链接: [https://arxiv.org/pdf/2512.20296v1](https://arxiv.org/pdf/2512.20296v1)**

> **作者:** Ji-Hoon Kim; Junseok Ahn; Doyeop Kwak; Joon Son Chung; Shinji Watanabe
>
> **备注:** Project page: https://mm.kaist.ac.kr/projects/TAVID
>
> **摘要:** The objective of this paper is to jointly synthesize interactive videos and conversational speech from text and reference images. With the ultimate goal of building human-like conversational systems, recent studies have explored talking or listening head generation as well as conversational speech generation. However, these works are typically studied in isolation, overlooking the multimodal nature of human conversation, which involves tightly coupled audio-visual interactions. In this paper, we introduce TAVID, a unified framework that generates both interactive faces and conversational speech in a synchronized manner. TAVID integrates face and speech generation pipelines through two cross-modal mappers (i.e., a motion mapper and a speaker mapper), which enable bidirectional exchange of complementary information between the audio and visual modalities. We evaluate our system across four dimensions: talking face realism, listening head responsiveness, dyadic interaction fluency, and speech quality. Extensive experiments demonstrate the effectiveness of our approach across all these aspects.
>
---
#### [new 012] Fun-Audio-Chat Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-Audio-Chat，属大型音频语言模型（LALM）任务，旨在解决语音-文本模态间时序分辨率失配、计算开销大及文本LLM知识灾难性遗忘问题。通过双分辨率语音表征、Core-Cocktail训练和多任务DPO训练，实现高效高质语音理解与生成，并开源8B模型及代码。**

- **链接: [https://arxiv.org/pdf/2512.20156v1](https://arxiv.org/pdf/2512.20156v1)**

> **作者:** Qian Chen; Luyao Cheng; Chong Deng; Xiangang Li; Jiaqing Liu; Chao-Hong Tan; Wen Wang; Junhao Xu; Jieping Ye; Qinglin Zhang; Qiquan Zhang; Jingren Zhou
>
> **备注:** 21 pages, https://github.com/FunAudioLLM/Fun-Audio-Chat
>
> **摘要:** Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo.
>
---
#### [new 013] SpidR: Learning Fast and Stable Linguistic Units for Spoken Language Models Without Supervision
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出SpidR模型，面向无文本监督的口语语言建模任务，旨在从原始语音中直接学习稳定、语义丰富的离散语言单元。它通过掩码预测、自蒸馏与在线聚类联合训练，提升单元质量与预训练效率，显著优于wav2vec 2.0等基线，并开源代码与模型。**

- **链接: [https://arxiv.org/pdf/2512.20308v1](https://arxiv.org/pdf/2512.20308v1)**

> **作者:** Maxime Poli; Mahi Luthra; Youssef Benchekroun; Yosuke Higuchi; Martin Gleize; Jiayi Shen; Robin Algayres; Yu-An Chung; Mido Assran; Juan Pino; Emmanuel Dupoux
>
> **备注:** 30 pages, 16 figures
>
> **摘要:** The parallel advances in language modeling and speech representation learning have raised the prospect of learning language directly from speech without textual intermediates. This requires extracting semantic representations directly from speech. Our contributions are threefold. First, we introduce SpidR, a self-supervised speech representation model that efficiently learns representations with highly accessible phonetic information, which makes it particularly suited for textless spoken language modeling. It is trained on raw waveforms using a masked prediction objective combined with self-distillation and online clustering. The intermediate layers of the student model learn to predict assignments derived from the teacher's intermediate layers. This learning objective stabilizes the online clustering procedure compared to previous approaches, resulting in higher quality codebooks. SpidR outperforms wav2vec 2.0, HuBERT, WavLM, and DinoSR on downstream language modeling benchmarks (sWUGGY, sBLIMP, tSC). Second, we systematically evaluate across models and layers the correlation between speech unit quality (ABX, PNMI) and language modeling performance, validating these metrics as reliable proxies. Finally, SpidR significantly reduces pretraining time compared to HuBERT, requiring only one day of pretraining on 16 GPUs, instead of a week. This speedup is enabled by the pretraining method and an efficient codebase, which allows faster iteration and easier experimentation. We open-source the training code and model checkpoints at https://github.com/facebookresearch/spidr.
>
---
#### [new 014] OASI: Objective-Aware Surrogate Initialization for Multi-Objective Bayesian Optimization in TinyML Keyword Spotting
- **分类: cs.LG; cs.SD**

- **简介: 该论文面向TinyML关键词唤醒（KWS）任务，解决多目标贝叶斯优化（MOBO）在资源受限设备上初始化不佳导致帕累托前沿搜索低效的问题；提出目标感知代理初始化（OASI），利用多目标模拟退火生成高质量初始帕累托集，显著提升收敛性与多样性。**

- **链接: [https://arxiv.org/pdf/2512.19739v1](https://arxiv.org/pdf/2512.19739v1)**

> **作者:** Soumen Garai; Suman Samui
>
> **备注:** Baseline version
>
> **摘要:** Voice assistants utilize Keyword Spotting (KWS) to enable efficient, privacy-friendly activation. However, realizing accurate KWS models on ultra-low-power TinyML devices (often with less than $<2$ MB of flash memory) necessitates a delicate balance between accuracy with strict resource constraints. Multi-objective Bayesian Optimization (MOBO) is an ideal candidate for managing such a trade-off but is highly initialization-dependent, especially under the budgeted black-box setting. Existing methods typically fall back to naive, ad-hoc sampling routines (e.g., Latin Hypercube Sampling (LHS), Sobol sequences, or Random search) that are adapted to neither the Pareto front nor undergo rigorous statistical comparison. To address this, we propose Objective-Aware Surrogate Initialization (OASI), a novel initialization strategy that leverages Multi-Objective Simulated Annealing (MOSA) to generate a seed Pareto set of high-performing and diverse configurations that explicitly balance accuracy and model size. Evaluated in a TinyML KWS setting, OASI outperforms LHS, Sobol, and Random initialization, achieving the highest hypervolume (0.0627) and the lowest generational distance (0.0) across multiple runs, with only a modest increase in computation time (1934 s vs. $\sim$1500 s). A non-parametric statistical analysis using the Kruskal-Wallis test ($H = 5.40$, $p = 0.144$, $η^2 = 0.0007$) and Dunn's post-hoc test confirms OASI's superior consistency despite the non-significant overall difference with respect to the $α=0.05$ threshold.
>
---
#### [new 015] DDAVS: Disentangled Audio Semantics and Delayed Bidirectional Alignment for Audio-Visual Segmentation
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文面向音频-视觉分割（AVS）任务，旨在解决多声源纠缠与音视频时序/语义错位问题。提出DDAVS框架：通过可学习查询与音频原型记忆库解耦音频语义，并引入延迟双向交叉注意力增强音视频对齐。在多源、多实例场景下性能领先。**

- **链接: [https://arxiv.org/pdf/2512.20117v1](https://arxiv.org/pdf/2512.20117v1)**

> **作者:** Jingqi Tian; Yiheng Du; Haoji Zhang; Yuji Wang; Isaac Ning Lee; Xulong Bai; Tianrui Zhu; Jingxuan Niu; Yansong Tang
>
> **备注:** https://trilarflagz.github.io/DDAVS-page/
>
> **摘要:** Audio-Visual Segmentation (AVS) aims to localize sound-producing objects at the pixel level by jointly leveraging auditory and visual information. However, existing methods often suffer from multi-source entanglement and audio-visual misalignment, which lead to biases toward louder or larger objects while overlooking weaker, smaller, or co-occurring sources. To address these challenges, we propose DDAVS, a Disentangled Audio Semantics and Delayed Bidirectional Alignment framework. To mitigate multi-source entanglement, DDAVS employs learnable queries to extract audio semantics and anchor them within a structured semantic space derived from an audio prototype memory bank. This is further optimized through contrastive learning to enhance discriminability and robustness. To alleviate audio-visual misalignment, DDAVS introduces dual cross-attention with delayed modality interaction, improving the robustness of multimodal alignment. Extensive experiments on the AVS-Objects and VPO benchmarks demonstrate that DDAVS consistently outperforms existing approaches, exhibiting strong performance across single-source, multi-source, and multi-instance scenarios. These results validate the effectiveness and generalization ability of our framework under challenging real-world audio-visual segmentation conditions. Project page: https://trilarflagz.github.io/DDAVS-page/
>
---
## 更新

#### [replaced 001] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出DeepASA模型，面向复杂动态听觉场景分析（ASA）任务，解决多源重叠、运动导致的参数关联模糊问题。通过对象导向处理（OOP）、链式推理（CoI）与时间一致性匹配（TCM），统一实现声源分离、去混响、事件检测等多任务，达SOTA性能。**

- **链接: [https://arxiv.org/pdf/2509.17247v3](https://arxiv.org/pdf/2509.17247v3)**

> **作者:** Dongheon Lee; Younghoo Kwon; Jung-Woo Choi
>
> **备注:** 21 pages, 13 figures, 11 tables, published in NeurIPS 2025
>
> **摘要:** We propose DeepASA, a multi-purpose model for auditory scene analysis that performs multi-input multi-output (MIMO) source separation, dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a unified framework. DeepASA is designed for complex auditory scenes where multiple, often similar, sound sources overlap in time and move dynamically in space. To achieve robust and consistent inference across tasks, we introduce an object-oriented processing (OOP) strategy. This approach encapsulates diverse auditory features into object-centric representations and refines them through a chain-of-inference (CoI) mechanism. The pipeline comprises a dynamic temporal kernel-based feature extractor, a transformer-based aggregator, and an object separator that yields per-object features. These features feed into multiple task-specific decoders. Our object-centric representations naturally resolve the parameter association ambiguity inherent in traditional track-wise processing. However, early-stage object separation can lead to failure in downstream ASA tasks. To address this, we implement temporal coherence matching (TCM) within the chain-of-inference, enabling multi-task fusion and iterative refinement of object features using estimated auditory parameters. We evaluate DeepASA on representative spatial audio benchmark datasets, including ASA2, MC-FUSS, and STARSS23. Experimental results show that our model achieves state-of-the-art performance across all evaluated tasks, demonstrating its effectiveness in both source separation and auditory parameter estimation under diverse spatial auditory scenes.
>
---
#### [replaced 002] Spectral Bottleneck in Sinusoidal Representation Networks: Noise is All You Need
- **分类: eess.AS; cs.CV; cs.LG; cs.SD; eess.IV**

- **简介: 该论文研究隐式神经表示（SIREN）的频谱瓶颈问题：因初始化不当导致高频拟合失败。提出目标感知的WINNER初始化方法，通过调控激活频谱与NTK特性提升音频/图像拟合精度。**

- **链接: [https://arxiv.org/pdf/2509.09719v2](https://arxiv.org/pdf/2509.09719v2)**

> **作者:** Hemanth Chandravamsi; Dhanush V. Shenoy; Itay Zinn; Ziv Chen; Shimon Pisnoy; Steven H. Frankel
>
> **摘要:** This work identifies and attempts to address a fundamental limitation of implicit neural representations with sinusoidal activation. The fitting error of SIRENs is highly sensitive to the target frequency content and to the choice of initialization. In extreme cases, this sensitivity leads to a spectral bottleneck that can result in a zero-valued output. This phenomenon is characterized by analyzing the evolution of activation spectra and the empirical neural tangent kernel (NTK) during the training process. An unfavorable distribution of energy across frequency modes was noted to give rise to this failure mode. Furthermore, the effect of Gaussian perturbations applied to the baseline uniformly initialized weights is examined, showing how these perturbations influence activation spectra and the NTK eigenbasis of SIREN. Overall, initialization emerges as a central factor governing the evolution of SIRENs, indicating the need for adaptive, target-aware strategies as the target length increases and fine-scale detail becomes essential. The proposed weight initialization scheme (WINNER) represents a simple ad hoc step in this direction and demonstrates that fitting accuracy can be significantly improved by modifying the spectral profile of network activations through a target-aware initialization. The approach achieves state-of-the-art performance on audio fitting tasks and yields notable improvements in image fitting tasks.
>
---
#### [replaced 003] Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning
- **分类: eess.AS; cs.CL**

- **简介: 该论文面向低资源语音识别（ASR）域适应任务，解决Speech LLM在缺乏配对语音-文本数据时难以适配新领域的问题。提出仅用目标域无标注文本进行微调的方法，并引入实时评估机制保持语音-文本对齐，避免遗忘源域性能。**

- **链接: [https://arxiv.org/pdf/2506.05671v2](https://arxiv.org/pdf/2506.05671v2)**

> **作者:** Yangui Fang; Jing Peng; Xu Li; Yu Xi; Chengwei Zhang; Guohui Zhong; Kai Yu
>
> **备注:** This paper has been ACCEPTED for publication in ASRU
>
> **摘要:** Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
>
---
#### [replaced 004] Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction
- **分类: cs.CL; eess.AS**

- **简介: 该论文属ASR错误纠正任务，旨在解决LLM直接纠错易产生幻觉、误改正确文本的问题。提出三阶段无训练框架RLLM-CF：错误预检测、思维链迭代修正、推理过程验证，显著降低CER/WER。**

- **链接: [https://arxiv.org/pdf/2505.24347v3](https://arxiv.org/pdf/2505.24347v3)**

> **作者:** Yangui Fang; Baixu Chen; Jing Peng; Xu Li; Yu Xi; Chengwei Zhang; Guohui Zhong
>
> **备注:** This paper has been ACCEPTED for publication in ASRU
>
> **摘要:** Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
>
---
#### [replaced 005] Improving Speech Emotion Recognition with Mutual Information Regularized Generative Model
- **分类: cs.SD; cs.LG**

- **简介: 该论文属语音情感识别（SER）任务，旨在解决情感语音数据稀缺导致模型性能受限的问题。提出基于互信息正则化的生成框架，通过跨模态对齐与特征级合成，提升单/多模态SER性能，并将互信息用作可量化生成质量指标。**

- **链接: [https://arxiv.org/pdf/2510.10078v3](https://arxiv.org/pdf/2510.10078v3)**

> **作者:** Chung-Soo Ahn; Rajib Rana; Sunil Sivadas; Carlos Busso; Jagath C. Rajapakse
>
> **摘要:** Lack of large, well-annotated emotional speech corpora continues to limit the performance and robustness of speech emotion recognition (SER), particularly as models grow more complex and the demand for multimodal systems increases. While generative data augmentation offers a promising solution, existing approaches often produce emotionally inconsistent samples due to oversimplified conditioning on categorical labels. This paper introduces a novel mutual-information-regularised generative framework that combines cross-modal alignment with feature-level synthesis. Building on an InfoGAN-style architecture, our method first learns a semantically aligned audio-text representation space using pre-trained transformers and contrastive objectives. A feature generator is then trained to produce emotion-aware audio features while employing mutual information as a quantitative regulariser to ensure strong dependency between generated features and their conditioning variables. We extend this approach to multimodal settings, enabling the generation of novel, paired (audio, text) features. Comprehensive evaluation on three benchmark datasets (IEMOCAP, MSP-IMPROV, MSP-Podcast) demonstrates that our framework consistently outperforms existing augmentation methods, achieving state-of-the-art performance with improvements of up to 2.6% in unimodal SER and 3.2% in multimodal emotion recognition. Most importantly, we demonstrate that mutual information functions as both a regulariser and a measurable metric for generative quality, offering a systematic approach to data augmentation in affective computing.
>
---
#### [replaced 006] Unsupervised Single-Channel Audio Separation with Diffusion Source Priors
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究无监督单通道音频分离任务，旨在解决真实场景中配对训练数据稀缺导致泛化差的问题。提出基于扩散源先验的逆问题求解框架，设计抗梯度冲突的求解器、混合信号初始化策略及时频注意力网络，显著提升分离质量。**

- **链接: [https://arxiv.org/pdf/2512.07226v2](https://arxiv.org/pdf/2512.07226v2)**

> **作者:** Runwu Shi; Chang Li; Jiang Wang; Rui Zhang; Nabeela Khan; Benjamin Yen; Takeshi Ashizawa; Kazuhiro Nakadai
>
> **摘要:** Single-channel audio separation aims to separate individual sources from a single-channel mixture. Most existing methods rely on supervised learning with synthetically generated paired data. However, obtaining high-quality paired data in real-world scenarios is often difficult. This data scarcity can degrade model performance under unseen conditions and limit generalization ability. To this end, in this work, we approach this problem from an unsupervised perspective, framing it as a probabilistic inverse problem. Our method requires only diffusion priors trained on individual sources. Separation is then achieved by iteratively guiding an initial state toward the solution through reconstruction guidance. Importantly, we introduce an advanced inverse problem solver specifically designed for separation, which mitigates gradient conflicts caused by interference between the diffusion prior and reconstruction guidance during inverse denoising. This design ensures high-quality and balanced separation performance across individual sources. Additionally, we find that initializing the denoising process with an augmented mixture instead of pure Gaussian noise provides an informative starting point that significantly improves the final performance. To further enhance audio prior modeling, we design a novel time-frequency attention-based network architecture that demonstrates strong audio modeling capability. Collectively, these improvements lead to significant performance gains, as validated across speech-sound event, sound event, and speech separation tasks.
>
---
