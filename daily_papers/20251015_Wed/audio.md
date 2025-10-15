# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文提出DiSTAR，用于零样本文本到语音合成。针对现有方法在分布偏移下脆弱及可控性不足的问题，其在离散RVQ码空间中结合自回归模型与掩蔽扩散模型，实现块级并行生成，提升鲁棒性、自然度与可控性。**

- **链接: [http://arxiv.org/pdf/2510.12210v1](http://arxiv.org/pdf/2510.12210v1)**

> **作者:** Yakun Song; Xiaobin Zhuang; Jiawei Chen; Zhikang Niu; Guanrou Yang; Chenpeng Du; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **摘要:** Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on https://anonymous.4open.science/w/DiSTAR_demo.
>
---
#### [new 002] SeeingSounds: Learning Audio-to-Visual Alignment via Text
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究音频到图像生成任务，旨在无需配对音视频数据下实现可控生成。提出SeeingSounds框架，通过文本桥接音频与视觉模态，利用冻结的扩散模型和轻量适配器，实现高效跨模态对齐与细粒度控制。**

- **链接: [http://arxiv.org/pdf/2510.11738v1](http://arxiv.org/pdf/2510.11738v1)**

> **作者:** Simone Carnemolla; Matteo Pennisi; Chiara Russo; Simone Palazzo; Daniela Giordano; Concetto Spampinato
>
> **备注:** accepted to ACM Multimedia Asia 2025
>
> **摘要:** We introduce SeeingSounds, a lightweight and modular framework for audio-to-image generation that leverages the interplay between audio, language, and vision-without requiring any paired audio-visual data or training on visual generative models. Rather than treating audio as a substitute for text or relying solely on audio-to-text mappings, our method performs dual alignment: audio is projected into a semantic language space via a frozen language encoder, and, contextually grounded into the visual domain using a vision-language model. This approach, inspired by cognitive neuroscience, reflects the natural cross-modal associations observed in human perception. The model operates on frozen diffusion backbones and trains only lightweight adapters, enabling efficient and scalable learning. Moreover, it supports fine-grained and interpretable control through procedural text prompt generation, where audio transformations (e.g., volume or pitch shifts) translate into descriptive prompts (e.g., "a distant thunder") that guide visual outputs. Extensive experiments across standard benchmarks confirm that SeeingSounds outperforms existing methods in both zero-shot and supervised settings, establishing a new state of the art in controllable audio-to-visual generation.
>
---
#### [new 003] I-DCCRN-VAE: An Improved Deep Representation Learning Framework for Complex VAE-based Single-channel Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文研究单通道语音增强任务，旨在提升复杂VAE模型的泛化能力。作者改进DCCRN-VAE，去除跳连、引入β-VAE预训练，并让NSVAE同时生成语音与噪声隐表示，增强了模型在未匹配数据上的表现。**

- **链接: [http://arxiv.org/pdf/2510.12485v1](http://arxiv.org/pdf/2510.12485v1)**

> **作者:** Jiatong Li; Simon Doclo
>
> **摘要:** Recently, a complex variational autoencoder (VAE)-based single-channel speech enhancement system based on the DCCRN architecture has been proposed. In this system, a noise suppression VAE (NSVAE) learns to extract clean speech representations from noisy speech using pretrained clean speech and noise VAEs with skip connections. In this paper, we improve DCCRN-VAE by incorporating three key modifications: 1) removing the skip connections in the pretrained VAEs to encourage more informative speech and noise latent representations; 2) using $\beta$-VAE in pretraining to better balance reconstruction and latent space regularization; and 3) a NSVAE generating both speech and noise latent representations. Experiments show that the proposed system achieves comparable performance as the DCCRN and DCCRN-VAE baselines on the matched DNS3 dataset but outperforms the baselines on mismatched datasets (WSJ0-QUT, Voicebank-DEMEND), demonstrating improved generalization ability. In addition, an ablation study shows that a similar performance can be achieved with classical fine-tuning instead of adversarial training, resulting in a simpler training pipeline.
>
---
#### [new 004] Content Anonymization for Privacy in Long-form Audio
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究长语音中的隐私保护，指出传统声纹匿名化无法防御基于语言风格的内容攻击。作者提出在ASR-TTS流程中对文本进行上下文重写，通过 paraphrasing 消除说话者特有表达方式，实验证明该方法可有效防御内容攻击并保持语义可用性。**

- **链接: [http://arxiv.org/pdf/2510.12780v1](http://arxiv.org/pdf/2510.12780v1)**

> **作者:** Cristina Aggazzotti; Ashi Garg; Zexin Cai; Nicholas Andrews
>
> **摘要:** Voice anonymization techniques have been found to successfully obscure a speaker's acoustic identity in short, isolated utterances in benchmarks such as the VoicePrivacy Challenge. In practice, however, utterances seldom occur in isolation: long-form audio is commonplace in domains such as interviews, phone calls, and meetings. In these cases, many utterances from the same speaker are available, which pose a significantly greater privacy risk: given multiple utterances from the same speaker, an attacker could exploit an individual's vocabulary, syntax, and turns of phrase to re-identify them, even when their voice is completely disguised. To address this risk, we propose new content anonymization approaches. Our approach performs a contextual rewriting of the transcripts in an ASR-TTS pipeline to eliminate speaker-specific style while preserving meaning. We present results in a long-form telephone conversation setting demonstrating the effectiveness of a content-based attack on voice-anonymized speech. Then we show how the proposed content-based anonymization methods can mitigate this risk while preserving speech utility. Overall, we find that paraphrasing is an effective defense against content-based attacks and recommend that stakeholders adopt this step to ensure anonymity in long-form audio.
>
---
#### [new 005] FakeMark: Deepfake Speech Attribution With Watermarked Artifacts
- **分类: eess.AS**

- **简介: 该论文属语音取证任务，旨在解决深伪语音溯源中分类方法泛化性差和传统水印易被破坏的问题。提出FakeMark框架，通过关联深伪系统生成带伪影相关水印，实现鲁棒的源系统归属检测。**

- **链接: [http://arxiv.org/pdf/2510.12042v1](http://arxiv.org/pdf/2510.12042v1)**

> **作者:** Wanying Ge; Xin Wang; Junichi Yamagishi
>
> **摘要:** Deepfake speech attribution remains challenging for existing solutions. Classifier-based solutions often fail to generalize to domain-shifted samples, and watermarking-based solutions are easily compromised by distortions like codec compression or malicious removal attacks. To address these issues, we propose FakeMark, a novel watermarking framework that injects artifact-correlated watermarks associated with deepfake systems rather than pre-assigned bitstring messages. This design allows a detector to attribute the source system by leveraging both injected watermark and intrinsic deepfake artifacts, remaining effective even if one of these cues is elusive or removed. Experimental results show that FakeMark improves generalization to cross-dataset samples where classifier-based solutions struggle and maintains high accuracy under various distortions where conventional watermarking-based solutions fail.
>
---
#### [new 006] TFGA-Net: Temporal-Frequency Graph Attention Network for Brain-Controlled Speaker Extraction
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究脑控说话人提取任务，旨在利用EEG信号解码听者关注的目标语音。提出TFGA-Net模型，融合多尺度时频特征与皮层拓扑结构，结合图卷积与注意力机制提取EEG特征，并引入MossFormer2分离语音，在公开数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.12275v1](http://arxiv.org/pdf/2510.12275v1)**

> **作者:** Youhao Si; Yuan Liao; Qiushi Han; Yuhang Yang; Rui Dai; Liya Huang
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** The rapid development of auditory attention decoding (AAD) based on electroencephalography (EEG) signals offers the possibility EEG-driven target speaker extraction. However, how to effectively utilize the target-speaker common information between EEG and speech remains an unresolved problem. In this paper, we propose a model for brain-controlled speaker extraction, which utilizes the EEG recorded from the listener to extract the target speech. In order to effectively extract information from EEG signals, we derive multi-scale time--frequency features and further incorporate cortical topological structures that are selectively engaged during the task. Moreover, to effectively exploit the non-Euclidean structure of EEG signals and capture their global features, the graph convolutional networks and self-attention mechanism are used in the EEG encoder. In addition, to make full use of the fused EEG and speech feature and preserve global context and capture speech rhythm and prosody, we introduce MossFormer2 which combines MossFormer and RNN-Free Recurrent as separator. Experimental results on both the public Cocktail Party and KUL dataset in this paper show that our TFGA-Net model significantly outper-forms the state-of-the-art method in certain objective evaluation metrics. The source code is available at: https://github.com/LaoDa-X/TFGA-NET.
>
---
#### [new 007] A Phase Synthesizer for Decorrelation to Improve Acoustic Feedback Cancellation
- **分类: eess.AS**

- **简介: 该论文针对通信系统中的声学反馈问题，提出一种基于DFT滤波器组的相位合成器，融合频移与变延时相位调制以解相关扬声器与麦克风信号，结合自适应频域Kalman滤波，提升系统稳定性与语音质量。**

- **链接: [http://arxiv.org/pdf/2510.12377v1](http://arxiv.org/pdf/2510.12377v1)**

> **作者:** Klaus Linhard; Philipp Bulling
>
> **摘要:** Undesired acoustic feedback is a known issue in communication systems, such as speech in-car communication, public address systems, or hearing aids. Without additional precautions, there is a high risk that the adaptive filter - intended to cancel the feedback path - also suppresses parts of the desired signal. One solution is to decorrelate the loudspeaker and microphone signals. In this work, we combine the two decorrelation approaches frequency shifting and phase modulation in a unified framework: a so-called \textit{phase synthesizer}, implemented in a discrete Fourier transform (DFT) filter bank. Furthermore, we extend the phase modulation technique using variable delay lines, as known from vibrato and chorus effects. We demonstrate the benefits of the proposed phase synthesizer using an example from speech in-car communication, employing an adaptive frequency-domain Kalman filter. Improvements in system stability, speech quality measured by perceptual evaluation of speech quality (PESQ) are presented.
>
---
#### [new 008] UALM: Unified Audio Language Model for Understanding, Generation and Reasoning
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出统一音频语言模型UALM，旨在解决音频理解、文本到音频生成和多模态推理分离的问题。通过UALM-Gen和UALM-Reason，实现单模型多任务，支持跨模态生成与推理，首次在音频领域验证了跨模态生成式推理的有效性。**

- **链接: [http://arxiv.org/pdf/2510.12000v1](http://arxiv.org/pdf/2510.12000v1)**

> **作者:** Jinchuan Tian; Sang-gil Lee; Zhifeng Kong; Sreyan Ghosh; Arushi Goel; Chao-Han Huck Yang; Wenliang Dai; Zihan Liu; Hanrong Ye; Shinji Watanabe; Mohammad Shoeybi; Bryan Catanzaro; Rafael Valle; Wei Ping
>
> **摘要:** Recent advances in the audio language modeling (ALM) domain tackle audio understanding and text-to-audio generation as separate tasks. Very few studies attempt to unify these tasks -- an essential step toward advanced multimodal reasoning. This paper introduces U}nified Audio Language Model (UALM), which aims to unify audio understanding, text-to-audio generation, and multimodal reasoning in a single model. To achieve this goal, we first present UALM-Gen, a text-to-audio language model that directly predicts audio tokens and is comparable to state-of-the-art diffusion-based models. We then demonstrate, using proper data blending, training recipes, and inference techniques, that our single UALM model matches the quality of state-of-the-art specialized models in audio understanding, text-to-audio generation, and text reasoning. Furthermore, we present UALM-Reason, a multimodal reasoning model that utilizes both text and audio in the intermediate thinking steps to facilitate complex generation tasks. To our knowledge, this is the first demonstration in audio research of cross-modal generative reasoning, with its effectiveness confirmed by subjective evaluations.
>
---
#### [new 009] Serial-Parallel Dual-Path Architecture for Speaking Style Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究说话风格识别（SSR）任务，旨在解决现有方法对声学信息利用不足的问题。作者提出一种串并行双路径架构，融合声学与语言模态信息，在减少88.4%参数的同时，实现30.3%的准确率提升。**

- **链接: [http://arxiv.org/pdf/2510.11732v1](http://arxiv.org/pdf/2510.11732v1)**

> **作者:** Guojian Li; Qijie Shao; Zhixian Zhao; Shuiyuan Wang; Zhonghua Fu; Lei Xie
>
> **备注:** Accepted by NCMMSC2025
>
> **摘要:** Speaking Style Recognition (SSR) identifies a speaker's speaking style characteristics from speech. Existing style recognition approaches primarily rely on linguistic information, with limited integration of acoustic information, which restricts recognition accuracy improvements. The fusion of acoustic and linguistic modalities offers significant potential to enhance recognition performance. In this paper, we propose a novel serial-parallel dual-path architecture for SSR that leverages acoustic-linguistic bimodal information. The serial path follows the ASR+STYLE serial paradigm, reflecting a sequential temporal dependency, while the parallel path integrates our designed Acoustic-Linguistic Similarity Module (ALSM) to facilitate cross-modal interaction with temporal simultaneity. Compared to the existing SSR baseline -- the OSUM model, our approach reduces parameter size by 88.4% and achieves a 30.3% improvement in SSR accuracy for eight styles on the test set.
>
---
#### [new 010] Audio Palette: A Diffusion Transformer with Multi-Signal Conditioning for Controllable Foley Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文聚焦可控Foley音效合成任务，旨在解决扩散模型中细粒度声学控制不足的问题。作者提出Audio Palette模型，引入四种时变控制信号，结合LoRA高效微调，实现了高保真、语义对齐且可解释的音频生成。**

- **链接: [http://arxiv.org/pdf/2510.12175v1](http://arxiv.org/pdf/2510.12175v1)**

> **作者:** Junnuo Wang
>
> **备注:** Accepted for publication in the Journal of Artificial Intelligence Research (JAIR), Vol. 3 No. 2, December 2025
>
> **摘要:** Recent advances in diffusion-based generative models have enabled high-quality text-to-audio synthesis, but fine-grained acoustic control remains a significant challenge in open-source research. We present Audio Palette, a diffusion transformer (DiT) based model that extends the Stable Audio Open architecture to address this "control gap" in controllable audio generation. Unlike prior approaches that rely solely on semantic conditioning, Audio Palette introduces four time-varying control signals: loudness, pitch, spectral centroid, and timbre, for precise and interpretable manipulation of acoustic features. The model is efficiently adapted for the nuanced domain of Foley synthesis using Low-Rank Adaptation (LoRA) on a curated subset of AudioSet, requiring only 0.85 percent of the original parameters to be trained. Experiments demonstrate that Audio Palette achieves fine-grained, interpretable control of sound attributes. Crucially, it accomplishes this novel controllability while maintaining high audio quality and strong semantic alignment to text prompts, with performance on standard metrics such as Frechet Audio Distance (FAD) and LAION-CLAP scores remaining comparable to the original baseline model. We provide a scalable, modular pipeline for audio research, emphasizing sequence-based conditioning, memory efficiency, and a three-scale classifier-free guidance mechanism for nuanced inference-time control. This work establishes a robust foundation for controllable sound design and performative audio synthesis in open-source settings, enabling a more artist-centric workflow.
>
---
#### [new 011] DeePAQ: A Perceptual Audio Quality Metric Based On Foundational Models and Weakly Supervised Learning
- **分类: eess.AS**

- **简介: 该论文提出DeePAQ，一种基于基础模型和弱监督学习的音频质量评估方法。针对通用音频质量评价任务，利用MERT模型和度量学习，通过低秩适配微调，有效检测编码失真并泛化至分离失真，提升客观指标与主观感知的一致性。**

- **链接: [http://arxiv.org/pdf/2510.12326v1](http://arxiv.org/pdf/2510.12326v1)**

> **作者:** Guanxin Jiang; Andreas Brendel; Pablo M. Delgado; Jürgen Herre
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** This paper presents the Deep learning-based Perceptual Audio Quality metric (DeePAQ) for evaluating general audio quality. Our approach leverages metric learning together with the music foundation model MERT, guided by surrogate labels, to construct an embedding space that captures distortion intensity in general audio. To the best of our knowledge, DeePAQ is the first in the general audio quality domain to leverage weakly supervised labels and metric learning for fine-tuning a music foundation model with Low-Rank Adaptation (LoRA), a direction not yet explored by other state-of-the-art methods. We benchmark the proposed model against state-of-the-art objective audio quality metrics across listening tests spanning audio coding and source separation. Results show that our method surpasses existing metrics in detecting coding artifacts and generalizes well to unseen distortions such as source separation, highlighting its robustness and versatility.
>
---
#### [new 012] Audio-Guided Visual Perception for Audio-Visual Navigation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM**

- **简介: 该论文研究音频-视觉导航任务，解决现有方法在新声音或新环境中泛化能力差的问题。提出AGVP框架，通过音频引导视觉注意力，实现跨模态对齐，提升导航效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.11760v1](http://arxiv.org/pdf/2510.11760v1)**

> **作者:** Yi Wang; Yinfeng Yu; Fuchun Sun; Liejun Wang; Wendong Zheng
>
> **备注:** Main paper (6 pages). Accepted for publication by International Conference on Virtual Reality and Visualization 2025 (ICVRV 2025)
>
> **摘要:** Audio-Visual Embodied Navigation aims to enable agents to autonomously navigate to sound sources in unknown 3D environments using auditory cues. While current AVN methods excel on in-distribution sound sources, they exhibit poor cross-source generalization: navigation success rates plummet and search paths become excessively long when agents encounter unheard sounds or unseen environments. This limitation stems from the lack of explicit alignment mechanisms between auditory signals and corresponding visual regions. Policies tend to memorize spurious \enquote{acoustic fingerprint-scenario} correlations during training, leading to blind exploration when exposed to novel sound sources. To address this, we propose the AGVP framework, which transforms sound from policy-memorable acoustic fingerprint cues into spatial guidance. The framework first extracts global auditory context via audio self-attention, then uses this context as queries to guide visual feature attention, highlighting sound-source-related regions at the feature level. Subsequent temporal modeling and policy optimization are then performed. This design, centered on interpretable cross-modal alignment and region reweighting, reduces dependency on specific acoustic fingerprints. Experimental results demonstrate that AGVP improves both navigation efficiency and robustness while achieving superior cross-scenario generalization on previously unheard sounds.
>
---
#### [new 013] Omni-Captioner: Data Pipeline, Models, and Benchmark for Omni Detailed Perception
- **分类: cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究多模态细粒度感知任务，旨在解决现有模型在生成细节时易产生幻觉的问题。作者提出Omni-Detective数据生成 pipeline，训练Audio-Captioner和Omni-Captioner模型，并构建新基准Omni-Cloze，实现更准确、可靠的细粒度音频-视觉描述。**

- **链接: [http://arxiv.org/pdf/2510.12720v1](http://arxiv.org/pdf/2510.12720v1)**

> **作者:** Ziyang Ma; Ruiyang Xu; Zhenghao Xing; Yunfei Chu; Yuxuan Wang; Jinzheng He; Jin Xu; Pheng-Ann Heng; Kai Yu; Junyang Lin; Eng Siong Chng; Xie Chen
>
> **备注:** https://github.com/ddlBoJack/Omni-Captioner
>
> **摘要:** Fine-grained perception of multimodal information is critical for advancing human-AI interaction. With recent progress in audio-visual technologies, Omni Language Models (OLMs), capable of processing audio and video signals in parallel, have emerged as a promising paradigm for achieving richer understanding and reasoning. However, their capacity to capture and describe fine-grained details remains limited explored. In this work, we present a systematic and comprehensive investigation of omni detailed perception from the perspectives of the data pipeline, models, and benchmark. We first identify an inherent "co-growth" between detail and hallucination in current OLMs. To address this, we propose Omni-Detective, an agentic data generation pipeline integrating tool-calling, to autonomously produce highly detailed yet minimally hallucinatory multimodal data. Based on the data generated with Omni-Detective, we train two captioning models: Audio-Captioner for audio-only detailed perception, and Omni-Captioner for audio-visual detailed perception. Under the cascade evaluation protocol, Audio-Captioner achieves the best performance on MMAU and MMAR among all open-source models, surpassing Gemini 2.5 Flash and delivering performance comparable to Gemini 2.5 Pro. On existing detailed captioning benchmarks, Omni-Captioner sets a new state-of-the-art on VDC and achieves the best trade-off between detail and hallucination on the video-SALMONN 2 testset. Given the absence of a dedicated benchmark for omni detailed perception, we design Omni-Cloze, a novel cloze-style evaluation for detailed audio, visual, and audio-visual captioning that ensures stable, efficient, and reliable assessment. Experimental results and analysis demonstrate the effectiveness of Omni-Detective in generating high-quality detailed captions, as well as the superiority of Omni-Cloze in evaluating such detailed captions.
>
---
#### [new 014] Not in Sync: Unveiling Temporal Bias in Audio Chat Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究音频大模型在时间定位上的偏差问题，属于音频理解与多模态推理任务。作者发现模型预测事件时间戳存在系统性偏移，提出时序偏差指数（TBI）量化该问题，并通过实验分析其在不同数据、模型和事件中的表现，呼吁构建更及时准确的LALM架构。**

- **链接: [http://arxiv.org/pdf/2510.12185v1](http://arxiv.org/pdf/2510.12185v1)**

> **作者:** Jiayu Yao; Shenghua Liu; Yiwei Wang; Rundong Cheng; Lingrui Mei; Baolong Bi; Zhen Xiong; Xueqi Cheng
>
> **摘要:** Large Audio Language Models (LALMs) are increasingly applied to audio understanding and multimodal reasoning, yet their ability to locate when events occur remains underexplored. We present the first systematic study of temporal bias in LALMs, revealing a key limitation in their timestamp prediction. For example, when asked "At which second does the lecturer introduce the key formula?", models often predict timestamps that are consistently earlier or later than the ground truth. Through controlled experiments on timestamped datasets, we find that temporal bias (i) is prevalent across datasets and models, (ii) increases with audio length - even accumulating to tens of seconds in extended recordings, and (iii) varies across event types and positions. We quantify this effect with the Temporal Bias Index (TBI), measuring systematic misalignment in predicted event timings, and complement it with a visualization framework. Our findings highlight a fundamental limitation in current LALMs and call for the development of temporally robust architectures.
>
---
## 更新

#### [replaced 001] Assessing Latency in ASR Systems: A Methodological Perspective for Real-Time Use
- **分类: cs.SD; cs.AI; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2409.05674v3](http://arxiv.org/pdf/2409.05674v3)**

> **作者:** Carlos Arriaga; Alejandro Pozo; Javier Conde; Alvaro Alonso
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Automatic speech recognition (ASR) systems generate real-time transcriptions but often miss nuances that human interpreters capture. While ASR is useful in many contexts, interpreters-who already use ASR tools such as Dragon-add critical value, especially in sensitive settings such as diplomatic meetings where subtle language is key. Human interpreters not only perceive these nuances but can adjust in real time, improving accuracy, while ASR handles basic transcription tasks. However, ASR systems introduce a delay that does not align with real-time interpretation needs. The user-perceived latency of ASR systems differs from that of interpretation because it measures the time between speech and transcription delivery. To address this, we propose a new approach to measuring delay in ASR systems and validate if they are usable in live interpretation scenarios.
>
---
#### [replaced 002] A Fast and Lightweight Model for Causal Audio-Visual Speech Separation
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.06689v2](http://arxiv.org/pdf/2506.06689v2)**

> **作者:** Wendi Sang; Kai Li; Runxuan Yang; Jianqiang Huang; Xiaolin Hu
>
> **备注:** Accepted by ECAI 2025
>
> **摘要:** Audio-visual speech separation (AVSS) aims to extract a target speech signal from a mixed signal by leveraging both auditory and visual (lip movement) cues. However, most existing AVSS methods exhibit complex architectures and rely on future context, operating offline, which renders them unsuitable for real-time applications. Inspired by the pipeline of RTFSNet, we propose a novel streaming AVSS model, named Swift-Net, which enhances the causal processing capabilities required for real-time applications. Swift-Net adopts a lightweight visual feature extraction module and an efficient fusion module for audio-visual integration. Additionally, Swift-Net employs Grouped SRUs to integrate historical information across different feature spaces, thereby improving the utilization efficiency of historical information. We further propose a causal transformation template to facilitate the conversion of non-causal AVSS models into causal counterparts. Experiments on three standard benchmark datasets (LRS2, LRS3, and VoxCeleb2) demonstrated that under causal conditions, our proposed Swift-Net exhibited outstanding performance, highlighting the potential of this method for processing speech in complex environments.
>
---
#### [replaced 003] MRSAudio: A Large-Scale Multimodal Recorded Spatial Audio Dataset with Refined Annotations
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.10396v2](http://arxiv.org/pdf/2510.10396v2)**

> **作者:** Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Xintong Hu; Yu Zhang; Li Tang; Rui Yang; Han Wang; Zongbao Zhang; Yuhan Wang; Yixuan Chen; Hankun Xu; Ke Xu; Pengfei Fan; Zhetao Chen; Yanhao Yu; Qiange Huang; Fei Wu; Zhou Zhao
>
> **备注:** 24 pages
>
> **摘要:** Humans rely on multisensory integration to perceive spatial environments, where auditory cues enable sound source localization in three-dimensional space. Despite the critical role of spatial audio in immersive technologies such as VR/AR, most existing multimodal datasets provide only monaural audio, which limits the development of spatial audio generation and understanding. To address these challenges, we introduce MRSAudio, a large-scale multimodal spatial audio dataset designed to advance research in spatial audio understanding and generation. MRSAudio spans four distinct components: MRSLife, MRSSpeech, MRSMusic, and MRSSing, covering diverse real-world scenarios. The dataset includes synchronized binaural and ambisonic audio, exocentric and egocentric video, motion trajectories, and fine-grained annotations such as transcripts, phoneme boundaries, lyrics, scores, and prompts. To demonstrate the utility and versatility of MRSAudio, we establish five foundational tasks: audio spatialization, and spatial text to speech, spatial singing voice synthesis, spatial music generation and sound event localization and detection. Results show that MRSAudio enables high-quality spatial modeling and supports a broad range of spatial audio research. Demos and dataset access are available at https://mrsaudio.github.io.
>
---
#### [replaced 004] Stimulus Modality Matters: Impact of Perceptual Evaluations from Different Modalities on Speech Emotion Recognition System Performance
- **分类: eess.AS; cs.MM; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.10762v4](http://arxiv.org/pdf/2409.10762v4)**

> **作者:** Huang-Cheng Chou; Haibin Wu; Hung-yi Lee; Chi-Chun Lee
>
> **备注:** 5 pages, 2 figures, 4 tables, acceptance for ICASSP 2025
>
> **摘要:** Speech Emotion Recognition (SER) systems rely on speech input and emotional labels annotated by humans. However, various emotion databases collect perceptional evaluations in different ways. For instance, the IEMOCAP dataset uses video clips with sounds for annotators to provide their emotional perceptions. However, the most significant English emotion dataset, the MSP-PODCAST, only provides speech for raters to choose the emotional ratings. Nevertheless, using speech as input is the standard approach to training SER systems. Therefore, the open question is the emotional labels elicited by which scenarios are the most effective for training SER systems. We comprehensively compare the effectiveness of SER systems trained with labels elicited by different modality stimuli and evaluate the SER systems on various testing conditions. Also, we introduce an all-inclusive label that combines all labels elicited by various modalities. We show that using labels elicited by voice-only stimuli for training yields better performance on the test set, whereas labels elicited by voice-only stimuli.
>
---
#### [replaced 005] Investigating Faithfulness in Large Audio Language Models
- **分类: cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.22363v2](http://arxiv.org/pdf/2509.22363v2)**

> **作者:** Lovenya Jain; Pooneh Mousavi; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Faithfulness measures whether chain-of-thought (CoT) representations accurately reflect a model's decision process and can be used as reliable explanations. Prior work has shown that CoTs from text-based LLMs are often unfaithful. This question has not been explored for large audio-language models (LALMs), where faithfulness is critical for safety-sensitive applications. Reasoning in LALMs is also more challenging, as models must first extract relevant clues from audio before reasoning over them. In this paper, we investigate the faithfulness of CoTs produced by several LALMs by applying targeted interventions, including paraphrasing, filler token injection, early answering, and introducing mistakes, on two challenging reasoning datasets: SAKURA and MMAR. After going through the aforementioned interventions across several datasets and tasks, our experiments suggest that, LALMs generally produce CoTs that appear to be faithful to their underlying decision processes.
>
---
#### [replaced 006] ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis
- **分类: cs.SD; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.10774v2](http://arxiv.org/pdf/2510.10774v2)**

> **作者:** Mohammad Javad Ranjbar Kalahroodi; Heshaam Faili; Azadeh Shakery
>
> **摘要:** Existing Persian speech datasets are typically smaller than their English counterparts, which creates a key limitation for developing Persian speech technologies. We address this gap by introducing ParsVoice, the largest Persian speech corpus designed specifically for text-to-speech(TTS) applications. We created an automated pipeline that transforms raw audiobook content into TTS-ready data, incorporating components such as a BERT-based sentence completion detector, a binary search boundary optimization method for precise audio-text alignment, and audio-text quality assessment frameworks tailored to Persian. The pipeline processes 2,000 audiobooks, yielding 3,526 hours of clean speech, which was further filtered into a 1,804-hour high-quality subset suitable for TTS, featuring more than 470 speakers. To validate the dataset, we fine-tuned XTTS for Persian, achieving a naturalness Mean Opinion Score (MOS) of 3.6/5 and a Speaker Similarity Mean Opinion Score (SMOS) of 4.0/5 demonstrating ParsVoice's effectiveness for training multi-speaker TTS systems. ParsVoice is the largest high-quality Persian speech dataset, offering speaker diversity and audio quality comparable to major English corpora. The complete dataset has been made publicly available to accelerate the development of Persian speech technologies. The ParsVoice dataset is publicly available at: https://huggingface.co/datasets/MohammadJRanjbar/ParsVoice.
>
---
#### [replaced 007] AsynFusion: Towards Asynchronous Latent Consistency Models for Decoupled Whole-Body Audio-Driven Avatars
- **分类: cs.SD; cs.AI; cs.CV; cs.GR; eess.AS; 68T10**

- **链接: [http://arxiv.org/pdf/2505.15058v2](http://arxiv.org/pdf/2505.15058v2)**

> **作者:** Tianbao Zhang; Jian Zhao; Yuer Li; Zheng Zhu; Ping Hu; Zhaoxin Fan; Wenjun Wu; Xuelong Li
>
> **备注:** 15pages, conference
>
> **摘要:** Whole-body audio-driven avatar pose and expression generation is a critical task for creating lifelike digital humans and enhancing the capabilities of interactive virtual agents, with wide-ranging applications in virtual reality, digital entertainment, and remote communication. Existing approaches often generate audio-driven facial expressions and gestures independently, which introduces a significant limitation: the lack of seamless coordination between facial and gestural elements, resulting in less natural and cohesive animations. To address this limitation, we propose AsynFusion, a novel framework that leverages diffusion transformers to achieve harmonious expression and gesture synthesis. The proposed method is built upon a dual-branch DiT architecture, which enables the parallel generation of facial expressions and gestures. Within the model, we introduce a Cooperative Synchronization Module to facilitate bidirectional feature interaction between the two modalities, and an Asynchronous LCM Sampling strategy to reduce computational overhead while maintaining high-quality outputs. Extensive experiments demonstrate that AsynFusion achieves state-of-the-art performance in generating real-time, synchronized whole-body animations, consistently outperforming existing methods in both quantitative and qualitative evaluations.
>
---
#### [replaced 008] Padé Approximant Neural Networks for Enhanced Electric Motor Fault Diagnosis Using Vibration and Acoustic Data
- **分类: cs.LG; cs.SD; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.02599v2](http://arxiv.org/pdf/2507.02599v2)**

> **作者:** Sertac Kilickaya; Levent Eren
>
> **备注:** This version is the author's accepted manuscript. It has been peer-reviewed and accepted for publication in Journal of Vibration Engineering & Technologies. The final published version is available at https://doi.org/10.1007/s42417-025-02129-5
>
> **摘要:** Purpose: The primary aim of this study is to enhance fault diagnosis in induction machines by leveraging the Pad\'e Approximant Neuron (PAON) model. While accelerometers and microphones are standard in motor condition monitoring, deep learning models with nonlinear neuron architectures offer promising improvements in diagnostic performance. This research investigates whether Pad\'e Approximant Neural Networks (Pad\'eNets) can outperform conventional Convolutional Neural Networks (CNNs) and Self-Organized Operational Neural Networks (Self-ONNs) in the diagnosis of electrical and mechanical faults from vibration and acoustic data. Methods: We evaluate and compare the diagnostic capabilities of three deep learning architectures: one-dimensional CNNs, Self-ONNs, and Pad\'eNets. These models are tested on the University of Ottawa's publicly available constant-speed induction motor datasets, which include both vibration and acoustic sensor data. The Pad\'eNet model is designed to introduce enhanced nonlinearity and is compatible with unbounded activation functions such as LeakyReLU. Results and Conclusion: Pad\'eNets consistently outperformed the baseline models, achieving diagnostic accuracies of 99.96%, 98.26%, 97.61%, and 98.33% for accelerometers 1, 2, 3, and the acoustic sensor, respectively. The enhanced nonlinearity of Pad\'eNets, together with their compatibility with unbounded activation functions, significantly improves fault diagnosis performance in induction motor condition monitoring.
>
---
#### [replaced 009] TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.15666v3](http://arxiv.org/pdf/2509.15666v3)**

> **作者:** Yongsheng Feng; Yuetonghui Xu; Jiehui Luo; Hongjia Liu; Xiaobing Li; Feng Yu; Wei Li
>
> **备注:** Submitted to ICASSP 2026.(C) 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work
>
> **摘要:** Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates early-split multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation. Code is available at https://github.com/WingSingFung/TISDiSS.
>
---
