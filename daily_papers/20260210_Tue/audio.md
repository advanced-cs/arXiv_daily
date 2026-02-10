# 音频 cs.SD;  eess.AS

- **最新发布 21 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Detect, Attend and Extract: Keyword Guided Target Speaker Extraction
- **分类: eess.AS**

- **简介: 该论文属于目标说话人提取任务，解决无清洁注册语音时的说话人分离问题。提出DAE-TSE框架，通过关键词定位并提取目标说话人语音。**

- **链接: [https://arxiv.org/pdf/2602.07977v1](https://arxiv.org/pdf/2602.07977v1)**

> **作者:** Haoyu Li; Yu Xi; Yidi Jiang; Shuai Wang; Kate Knill; Mark Gales; Haizhou Li; Kai Yu
>
> **备注:** 4 figures, 4 tables. Submitted to IJCAI-ECAI 2026
>
> **摘要:** Target speaker extraction (TSE) aims to extract the speech of a target speaker from mixtures containing multiple competing speakers. Conventional TSE systems predominantly rely on speaker cues, such as pre-enrolled speech, to identify and isolate the target speaker. However, in many practical scenarios, clean enrollment utterances are unavailable, limiting the applicability of existing approaches. In this work, we propose DAE-TSE, a keyword-guided TSE framework that specifies the target speaker through distinct keywords they utter. By leveraging keywords (i.e., partial transcriptions) as cues, our approach provides a flexible and practical alternative to enrollment-based TSE. DAE-TSE follows the Detect-Attend-Extract (DAE) paradigm: it first detects the presence of the given keywords, then attends to the corresponding speaker based on the keyword content, and finally extracts the target speech. Experimental results demonstrate that DAE-TSE outperforms standard TSE systems that rely on clean enrollment speech. To the best of our knowledge, this is the first study to utilize partial transcription as a cue for specifying the target speaker in TSE, offering a flexible and practical solution for real-world scenarios. Our code and demo page are now publicly available.
>
---
#### [new 002] Input-Adaptive Spectral Feature Compression by Sequence Modeling for Source Separation
- **分类: eess.AS**

- **简介: 该论文属于语音源分离任务，旨在解决传统频带分割模块输入非自适应和参数冗余的问题，提出SFC模块实现高效频谱特征压缩。**

- **链接: [https://arxiv.org/pdf/2602.08671v1](https://arxiv.org/pdf/2602.08671v1)**

> **作者:** Kohei Saijo; Yoshiaki Bando
>
> **备注:** Accepted by IEEE TASLP. \c{opyright} 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Time-frequency domain dual-path models have demonstrated strong performance and are widely used in source separation. Because their computational cost grows with the number of frequency bins, these models often use the band-split (BS) module in high-sampling-rate tasks such as music source separation (MSS) and cinematic audio source separation (CASS). The BS encoder compresses frequency information by encoding features for each predefined subband. It achieves effective compression by introducing an inductive bias that places greater emphasis on low-frequency parts. Despite its success, the BS module has two inherent limitations: (i) it is not input-adaptive, preventing the use of input-dependent information, and (ii) the parameter count is large, since each subband requires a dedicated module. To address these issues, we propose Spectral Feature Compression (SFC). SFC compresses the input using a single sequence modeling module, making it both input-adaptive and parameter-efficient. We investigate two variants of SFC, one based on cross-attention and the other on Mamba, and introduce inductive biases inspired by the BS module to make them suitable for frequency information compression. Experiments on MSS and CASS tasks demonstrate that the SFC module consistently outperforms the BS module across different separator sizes and compression ratios. We also provide an analysis showing that SFC adaptively captures frequency patterns from the input.
>
---
#### [new 003] Prototype-Based Disentanglement for Controllable Dysarthric Speech Synthesis
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音合成任务，旨在解决失语症语音合成中可控性差的问题。提出ProtoDisent-TTS框架，分离说话人音色与病理特征，实现健康与失语语音的双向转换。**

- **链接: [https://arxiv.org/pdf/2602.08696v1](https://arxiv.org/pdf/2602.08696v1)**

> **作者:** Haoshen Wang; Xueli Zhong; Bingbing Lin; Jia Huang; Xingduo Pan; Shengxiang Liang; Nizhuan Wang; Wai Ting Siok
>
> **摘要:** Dysarthric speech exhibits high variability and limited labeled data, posing major challenges for both automatic speech recognition (ASR) and assistive speech technologies. Existing approaches rely on synthetic data augmentation or speech reconstruction, yet often entangle speaker identity with pathological articulation, limiting controllability and robustness. In this paper, we propose ProtoDisent-TTS, a prototype-based disentanglement TTS framework built on a pre-trained text-to-speech backbone that factorizes speaker timbre and dysarthric articulation within a unified latent space. A pathology prototype codebook provides interpretable and controllable representations of healthy and dysarthric speech patterns, while a dual-classifier objective with a gradient reversal layer enforces invariance of speaker embeddings to pathological attributes. Experiments on the TORGO dataset demonstrate that this design enables bidirectional transformation between healthy and dysarthric speech, leading to consistent ASR performance gains and robust, speaker-aware speech reconstruction.
>
---
#### [new 004] Tutti: Expressive Multi-Singer Synthesis via Structure-Level Timbre Control and Vocal Texture Modeling
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于歌唱语音合成任务，旨在解决多歌手编排与音色控制问题。提出Tutti框架，实现结构化多歌手生成与声学纹理建模。**

- **链接: [https://arxiv.org/pdf/2602.08233v1](https://arxiv.org/pdf/2602.08233v1)**

> **作者:** Jiatao Chen; Xing Tang; Xiaoyue Duan; Yutang Feng; Jinchao Zhang; Jie Zhou
>
> **摘要:** While existing Singing Voice Synthesis systems achieve high-fidelity solo performances, they are constrained by global timbre control, failing to address dynamic multi-singer arrangement and vocal texture within a single song. To address this, we propose Tutti, a unified framework designed for structured multi-singer generation. Specifically, we introduce a Structure-Aware Singer Prompt to enable flexible singer scheduling evolving with musical structure, and propose Complementary Texture Learning via Condition-Guided VAE to capture implicit acoustic textures (e.g., spatial reverberation and spectral fusion) that are complementary to explicit controls. Experiments demonstrate that Tutti excels in precise multi-singer scheduling and significantly enhances the acoustic realism of choral generation, offering a novel paradigm for complex multi-singer arrangement. Audio samples are available at https://annoauth123-ctrl.github.io/Tutii_Demo/.
>
---
#### [new 005] SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于歌唱语音合成任务，旨在解决开放源代码系统在鲁棒性和零样本泛化上的问题。提出SoulX-Singer系统，支持多种语言和控制方式，提升合成质量与实用性。**

- **链接: [https://arxiv.org/pdf/2602.07803v1](https://arxiv.org/pdf/2602.07803v1)**

> **作者:** Jiale Qian; Hao Meng; Tian Zheng; Pengcheng Zhu; Haopeng Lin; Yuhang Dai; Hanke Xie; Wenxiao Cao; Ruixuan Shang; Jun Wu; Hongmei Liu; Hanlin Wen; Jian Zhao; Zhonglin Jiang; Yong Chen; Shunshun Yin; Ming Tao; Jianguo Wei; Lei Xie; Xinsheng Wang
>
> **备注:** Technical Report
>
> **摘要:** While recent years have witnessed rapid progress in speech synthesis, open-source singing voice synthesis (SVS) systems still face significant barriers to industrial deployment, particularly in terms of robustness and zero-shot generalization. In this report, we introduce SoulX-Singer, a high-quality open-source SVS system designed with practical deployment considerations in mind. SoulX-Singer supports controllable singing generation conditioned on either symbolic musical scores (MIDI) or melodic representations, enabling flexible and expressive control in real-world production workflows. Trained on more than 42,000 hours of vocal data, the system supports Mandarin Chinese, English, and Cantonese and consistently achieves state-of-the-art synthesis quality across languages under diverse musical conditions. Furthermore, to enable reliable evaluation of zero-shot SVS performance in practical scenarios, we construct SoulX-Singer-Eval, a dedicated benchmark with strict training-test disentanglement, facilitating systematic assessment in zero-shot settings.
>
---
#### [new 006] Massive Sound Embedding Benchmark (MSEB)
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出Massive Sound Embedding Benchmark (MSEB)，用于评估多模态系统中的音频能力。旨在解决音频理解与处理任务，通过八项核心任务和多样数据集推动音频智能发展。**

- **链接: [https://arxiv.org/pdf/2602.07143v1](https://arxiv.org/pdf/2602.07143v1)**

> **作者:** Georg Heigold; Ehsan Variani; Tom Bagby; Cyril Allauzen; Ji Ma; Shankar Kumar; Michael Riley
>
> **摘要:** Audio is a critical component of multimodal perception, and any truly intelligent system must demonstrate a wide range of auditory capabilities. These capabilities include transcription, classification, retrieval, reasoning, segmentation, clustering, reranking, and reconstruction. Fundamentally, each task involves transforming a raw audio signal into a meaningful 'embedding' - be it a single vector, a sequence of continuous or discrete representations, or another structured form - which then serves as the basis for generating the task's final response. To accelerate progress towards robust machine auditory intelligence, we present the Massive Sound Embedding Benchmark (MSEB): an extensible framework designed to evaluate the auditory components of any multimodal system. In its first release, MSEB offers a comprehensive suite of eight core tasks, with more planned for the future, supported by diverse datasets, including the new, large-scale Simple Voice Questions (SVQ) dataset. Our initial experiments establish clear performance headrooms, highlighting the significant opportunity to improve real-world multimodal experiences where audio is a core signal. We encourage the research community to use MSEB to assess their algorithms and contribute to its growth. The library is publicly hosted at github.
>
---
#### [new 007] No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting
- **分类: cs.SD**

- **简介: 该论文属于开放词汇关键词识别任务，解决前缀偏差导致的误触发问题。通过引入POB数据集和EPS方法，提升模型性能并减少错误率。**

- **链接: [https://arxiv.org/pdf/2602.08930v1](https://arxiv.org/pdf/2602.08930v1)**

> **作者:** Yi Liu; Chuan-Che; Huang; Xiao Quan
>
> **备注:** Published in ICASSP 2026
>
> **摘要:** Open-vocabulary keyword spotting (OV-KWS) enables personalized device control via arbitrary voice commands. Recently, researchers have explored using audio-text joint embeddings, allowing users to enroll phrases with text, and proposed techniques to disambiguate similar utterances. We find that existing OV-KWS solutions often overly bias the beginning phonemes of an enrollment, causing false triggers when negative enrollment-query-pairs share a prefix (``turn the volume up'' vs. ``turn the volume down''). We trace this to two factors: training data bias and position-biased cross-modal scoring. To address these limitations, we introduce the Partial Overlap Benchmark (POB) with two datasets, POB-Spark and POB-LibriPhrase (POB-LP), containing mismatched audio-text pairs with shared prefixes, and propose Equal-weighting Position Scoring (EPS), a lightweight decision layer. Using EPS alone reduces EER on POB-Spark from 64.4\% to 29.3\% and improves POB-LP accuracy from 87.6\% to 96.8\%, while maintaining performance on LibriPhrase and Google Speech Commands (GSC). With POB data added in training, our work achieves the best POB benchmark results while incurring the least amount of degradation on prior metrics among baselines. This degradation is most pronounced in GSC, which contains only one-word commands. We surface mitigating this trade-off as future work.
>
---
#### [new 008] SNC: A Stem-Native Codec for Efficient Lossless Audio Storage with Adaptive Playback Capabilities
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SNC音频编码格式，解决传统格式在压缩与功能间的平衡问题。通过分离音轨和低能残留，实现高效无损存储与灵活播放。**

- **链接: [https://arxiv.org/pdf/2602.08148v1](https://arxiv.org/pdf/2602.08148v1)**

> **作者:** Shaad Sufi
>
> **摘要:** Current audio formats present a fundamental trade-off between file size and functionality: lossless formats like FLAC preserve quality but lack adaptability, while lossy formats reduce size at the cost of fidelity and offer no stem-level access.We introduce the Stem-Native Codec (SNC), a novel audio container format that stores music as independently encoded stems plus a low-energy mastering residual. By exploiting the lower information entropy of separated stems compared to mixed audio, SNC achieves a 38.2% file size reduction versus FLAC (7.76 MB vs. 12.55 MB for a 2:18 test track) while maintaining perceptual transparency (STOI = 0.996). Unlike existing formats, SNC enables context-aware adaptive playback, spatial audio rendering, and user-controlled remixing without requiring additional storage. Our experimental validation demonstrates that the stems-plus residual architecture successfully decouples the conflicting requirements of compression efficiency and feature richness, offering a practical path toward next-generation audio distribution systems.
>
---
#### [new 009] Physics-Guided Variational Model for Unsupervised Sound Source Tracking
- **分类: eess.AS**

- **简介: 该论文属于声源跟踪任务，解决无监督定位问题。提出一种物理引导的变分模型，无需标签即可实现单声源跟踪，并扩展至多声源，具有强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.08484v1](https://arxiv.org/pdf/2602.08484v1)**

> **作者:** Luan Vinícius Fiorio; Ivana Nikoloska; Bruno Defraene; Alex Young; Johan David; Ronald M. Aarts
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Sound source tracking is often performed using classical array-processing algorithms. Alternative methods, such as machine learning, rely on ground truth position labels, which are costly to obtain. We propose a variational model that can perform single-source unsupervised sound source tracking in latent space, aided by a physics-based decoder. Our experiments demonstrate that the proposed method surpasses traditional baselines and achieves performance and computational complexity comparable to state-of-the-art supervised models. We also show that the method presents substantial robustness to altered microphone array geometries and corrupted microphone position metadata. Finally, the method is extended to multi-source sound tracking and the basic theoretical changes are proposed.
>
---
#### [new 010] MENASpeechBank: A Reference Voice Bank with Persona-Conditioned Multi-Turn Conversations for AudioLLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出MENASpeechBank，解决AudioLLMs缺乏多样化对话数据的问题，通过合成数据生成多语言、多角色的对话，提升模型对话能力。**

- **链接: [https://arxiv.org/pdf/2602.07036v1](https://arxiv.org/pdf/2602.07036v1)**

> **作者:** Zien Sheikh Ali; Hunzalah Hassan Bhatti; Rabindra Nath Nandi; Shammur Absar Chowdhury; Firoj Alam
>
> **备注:** Foundation Models, Large Language Models, Native, Speech Models, Arabic, AI-persona, Persona-conditioned-conversations
>
> **摘要:** Audio large language models (AudioLLMs) enable instruction-following over speech and general audio, but progress is increasingly limited by the lack of diverse, conversational, instruction-aligned speech-text data. This bottleneck is especially acute for persona-grounded interactions and dialectal coverage, where collecting and releasing real multi-speaker recordings is costly and slow. We introduce MENASpeechBank, a reference speech bank comprising about 18K high-quality utterances from 124 speakers spanning multiple MENA countries, covering English, Modern Standard Arabic (MSA), and regional Arabic varieties. Building on this resource, we develop a controllable synthetic data pipeline that: (i) constructs persona profiles enriched with World Values Survey-inspired attributes, (ii) defines a taxonomy of about 5K conversational scenarios, (iii) matches personas to scenarios via semantic similarity, (iv) generates about 417K role-play conversations with an LLM where the user speaks as the persona and the assistant behaves as a helpful agent, and (v) synthesizes the user turns by conditioning on reference speaker audio to preserve speaker identity and diversity. We evaluate both synthetic and human-recorded conversations and provide detailed analysis. We will release MENASpeechBank and the generated conversations publicly for the community.
>
---
#### [new 011] Cross-Modal Bottleneck Fusion For Noise Robust Audio-Visual Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于音频-视觉语音识别任务，旨在提升噪声环境下语音识别的鲁棒性。提出CoBRA框架，通过跨模态瓶颈融合，有效利用视觉信息，增强模型在噪声条件下的性能。**

- **链接: [https://arxiv.org/pdf/2602.08293v1](https://arxiv.org/pdf/2602.08293v1)**

> **作者:** Seaone Ok; Min Jun Choi; Eungbeom Kim; Seungu Han; Kyogu Lee
>
> **备注:** 5 pages, 3 figures, ICASSP 2026 Accepted
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) leverages both acoustic and visual cues to improve speech recognition under noisy conditions. A central question is how to design a fusion mechanism that allows the model to effectively exploit visual information when the audio signal is degraded, while maintaining strong performance on clean speech. We propose CoBRA (Cross-modal Bottleneck for Robust AVSR), a bottleneck-based fusion framework that introduces a compact set of learnable tokens to mediate cross-modal exchange. By regulating information flow through these tokens, the audio stream can reliably access essential visual cues even under adverse or out-of-domain noise. Despite limited training data, our model surpasses comparable baselines and remains competitive with large-scale systems through noise-adaptive fusion, demonstrating both efficiency and robustness. Ablation studies highlight that the depth of fusion is the most critical factor, underscoring its importance in designing robust AVSR systems.
>
---
#### [new 012] Global Rotation Equivariant Phase Modeling for Speech Enhancement with Deep Magnitude-Phase Interaction
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决相位建模困难的问题。通过引入全局旋转等变的双流框架，提升相位建模效果。**

- **链接: [https://arxiv.org/pdf/2602.08556v1](https://arxiv.org/pdf/2602.08556v1)**

> **作者:** Chengzhong Wang; Andong Li; Dingding Yao; Junfeng Li
>
> **备注:** Submitted to IEEE TASLP
>
> **摘要:** While deep learning has advanced speech enhancement (SE), effective phase modeling remains challenging, as conventional networks typically operate within a flat Euclidean feature space, which is not easy to model the underlying circular topology of the phase. To address this, we propose a manifold-aware magnitude-phase dual-stream framework that aligns the phase stream with its intrinsic circular geometry by enforcing Global Rotation Equivariance (GRE) characteristic. Specifically, we introduce a Magnitude-Phase Interactive Convolutional Module (MPICM) for modulus-based information exchange and a Hybrid-Attention Dual-FFN (HADF) bottleneck for unified feature fusion, both of which are designed to preserve GRE in the phase stream. Comprehensive evaluations are conducted across phase retrieval, denoising, dereverberation, and bandwidth extension tasks to validate the superiority of the proposed method over multiple advanced baselines. Notably, the proposed architecture reduces Phase Distance by over 20\% in the phase retrieval task and improves PESQ by more than 0.1 in zero-shot cross-corpus denoising evaluations. The overall superiority is also established in universal SE tasks involving mixed distortions. Qualitative analysis further reveals that the learned phase features exhibit distinct periodic patterns, which are consistent with the intrinsic circular nature of the phase. The source code is available at https://github.com/wangchengzhong/RENet.
>
---
#### [new 013] CALM: Class-Conditional Sparse Attention Vectors for Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频分类任务，解决LALM在特定任务中表现不足的问题。提出CALM方法，通过学习类别相关的注意力权重提升分类性能。**

- **链接: [https://arxiv.org/pdf/2602.07077v1](https://arxiv.org/pdf/2602.07077v1)**

> **作者:** Videet Mehta; Liming Wang; Hilde Kuehne; Rogerio Feris; James R. Glass; M. Jehanzeb Mirza
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Large audio-language models (LALMs) exhibit strong zero-shot capabilities in multiple downstream tasks, such as audio question answering (AQA) and abstract reasoning; however, these models still lag behind specialized models for certain discriminative tasks (e.g., audio classification). Recent studies show that sparse subsets of attention heads within an LALM can serve as strong discriminative feature extractors for downstream tasks such as classification via simple voting schemes. However, these methods assign uniform weights to all selected heads, implicitly assuming that each head contributes equally across all semantic categories. In this work, we propose Class-Conditional Sparse Attention Vectors for Large Audio-Language Models, a few-shot classification method that learns class-dependent importance weights over attention heads. This formulation allows individual heads to specialize in distinct semantic categories and to contribute to ensemble predictions proportionally to their estimated reliability. Experiments on multiple few-shot audio and audiovisual classification benchmarks and tasks demonstrate that our method consistently outperforms state-of-the-art uniform voting-based approaches by up to 14.52%, 1.53%, 8.35% absolute gains for audio classification, audio-visual classification, and spoofing detection respectively.
>
---
#### [new 014] Beyond Transcripts: A Renewed Perspective on Audio Chaptering
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频章节划分任务，解决如何有效利用音频信息、处理ASR错误及无文本评估的问题。通过对比不同模型和评估方法，提出音频专用架构AudioSeg，提升分割效果。**

- **链接: [https://arxiv.org/pdf/2602.08979v1](https://arxiv.org/pdf/2602.08979v1)**

> **作者:** Fabian Retkowski; Maike Züfle; Thai Binh Nguyen; Jan Niehues; Alexander Waibel
>
> **摘要:** Audio chaptering, the task of automatically segmenting long-form audio into coherent sections, is increasingly important for navigating podcasts, lectures, and videos. Despite its relevance, research remains limited and text-based, leaving key questions unresolved about leveraging audio information, handling ASR errors, and transcript-free evaluation. We address these gaps through three contributions: (1) a systematic comparison between text-based models with acoustic features, a novel audio-only architecture (AudioSeg) operating on learned audio representations, and multimodal LLMs; (2) empirical analysis of factors affecting performance, including transcript quality, acoustic features, duration, and speaker composition; and (3) formalized evaluation protocols contrasting transcript-dependent text-space protocols with transcript-invariant time-space protocols. Our experiments on YTSeg reveal that AudioSeg substantially outperforms text-based approaches, pauses provide the largest acoustic gains, and MLLMs remain limited by context length and weak instruction following, yet MLLMs are promising on shorter audio.
>
---
#### [new 015] Equipping LLM with Directional Multi-Talker Speech Understanding Capabilities
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音理解任务，旨在解决多说话人、多麦克风场景下的方向性语音理解问题。通过两种方法增强LLM的定向能力，提升其在智能眼镜中的应用效果。**

- **链接: [https://arxiv.org/pdf/2602.07211v1](https://arxiv.org/pdf/2602.07211v1)**

> **作者:** Ju Lin; Jing Pan; Ruizhi Li; Ming Sun; Yuzong Liu; Alaa Hassan; Jing Zheng; Florian Metze
>
> **摘要:** Recent studies have demonstrated that prompting large language models (LLM) with audio encodings enables effective speech understanding capabilities. However, most speech LLMs are trained on single-channel, single-talker data, which makes it challenging to directly apply them to multi-talker and multi-channel speech understanding task. In this work, we present a comprehensive investigation on how to enable directional multi-talker speech understanding capabilities for LLMs, specifically in smart glasses usecase. We propose two novel approaches to integrate directivity into LLMs: (1) a cascaded system that leverages a source separation front-end module, and (2) an end-to-end system that utilizes serialized output training. All of the approaches utilize a multi-microphone array embedded in smart glasses to optimize directivity interpretation and processing in a streaming manner. Experimental results demonstrate the efficacy of our proposed methods in endowing LLMs with directional speech understanding capabilities, achieving strong performance in both speech recognition and speech translation tasks.
>
---
#### [new 016] VocalNet-MDM: Accelerating Streaming Speech LLM via Self-Distilled Masked Diffusion Modeling
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音大模型任务，解决自回归模型效率低、延迟高的问题。提出VocalNet-MDM，通过掩码扩散建模和自蒸馏技术提升生成速度与自然度。**

- **链接: [https://arxiv.org/pdf/2602.08607v1](https://arxiv.org/pdf/2602.08607v1)**

> **作者:** Ziyang Cheng; Yuhao Wang; Heyang Liu; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** Recent Speech Large Language Models~(LLMs) have achieved impressive capabilities in end-to-end speech interaction. However, the prevailing autoregressive paradigm imposes strict serial constraints, limiting generation efficiency and introducing exposure bias. In this paper, we investigate Masked Diffusion Modeling~(MDM) as a non-autoregressive paradigm for speech LLMs and introduce VocalNet-MDM. To adapt MDM for streaming speech interaction, we address two critical challenges: training-inference mismatch and iterative overhead. We propose Hierarchical Block-wise Masking to align training objectives with the progressive masked states encountered during block diffusion decoding, and Iterative Self-Distillation to compress multi-step refinement into fewer steps for low-latency inference. Trained on a limited scale of only 6K hours of speech data, VocalNet-MDM achieves a 3.7$\times$--10$\times$ decoding speedup and reduces first-chunk latency by 34\% compared to AR baselines. It maintains competitive recognition accuracy while achieving state-of-the-art text quality and speech naturalness, demonstrating that MDM is a promising and scalable alternative for low-latency, efficient speech LLMs.
>
---
#### [new 017] MOVA: Towards Scalable and Synchronized Video-Audio Generation
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出MOVA，解决视频音频同步生成问题，采用Mixture-of-Experts架构，支持高质量多模态内容生成。**

- **链接: [https://arxiv.org/pdf/2602.08794v1](https://arxiv.org/pdf/2602.08794v1)**

> **作者:** SII-OpenMOSS Team; :; Donghua Yu; Mingshu Chen; Qi Chen; Qi Luo; Qianyi Wu; Qinyuan Cheng; Ruixiao Li; Tianyi Liang; Wenbo Zhang; Wenming Tu; Xiangyu Peng; Yang Gao; Yanru Huo; Ying Zhu; Yinze Luo; Yiyang Zhang; Yuerong Song; Zhe Xu; Zhiyu Zhang; Chenchen Yang; Cheng Chang; Chushu Zhou; Hanfu Chen; Hongnan Ma; Jiaxi Li; Jingqi Tong; Junxi Liu; Ke Chen; Shimin Li; Songlin Wang; Wei Jiang; Zhaoye Fei; Zhiyuan Ning; Chunguo Li; Chenhui Li; Ziwei He; Zengfeng Huang; Xie Chen; Xipeng Qiu
>
> **备注:** Technical report for MOVA (open-source video-audio generation model). 38 pages, 10 figures, 22 tables. Project page: https://mosi.cn/models/mova Code: https://github.com/OpenMOSS/MOVA Models: https://huggingface.co/collections/OpenMOSS-Team/mova. Qinyuan Cheng and Tianyi Liang are project leader. Xie Chen and Xipeng Qiu are corresponding authors
>
> **摘要:** Audio is indispensable for real-world video, yet generation models have largely overlooked audio components. Current approaches to producing audio-visual content often rely on cascaded pipelines, which increase cost, accumulate errors, and degrade overall quality. While systems such as Veo 3 and Sora 2 emphasize the value of simultaneous generation, joint multimodal modeling introduces unique challenges in architecture, data, and training. Moreover, the closed-source nature of existing systems limits progress in the field. In this work, we introduce MOVA (MOSS Video and Audio), an open-source model capable of generating high-quality, synchronized audio-visual content, including realistic lip-synced speech, environment-aware sound effects, and content-aligned music. MOVA employs a Mixture-of-Experts (MoE) architecture, with a total of 32B parameters, of which 18B are active during inference. It supports IT2VA (Image-Text to Video-Audio) generation task. By releasing the model weights and code, we aim to advance research and foster a vibrant community of creators. The released codebase features comprehensive support for efficient inference, LoRA fine-tuning, and prompt enhancement.
>
---
#### [new 018] PTS-SNN: A Prompt-Tuned Temporal Shift Spiking Neural Networks for Efficient Speech Emotion Recognition
- **分类: cs.AI; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决传统模型计算成本高和SNN与SSL表示不匹配的问题。提出PTS-SNN框架，通过参数高效方法实现高效情感识别。**

- **链接: [https://arxiv.org/pdf/2602.08240v1](https://arxiv.org/pdf/2602.08240v1)**

> **作者:** Xun Su; Huamin Wang; Qi Zhang
>
> **摘要:** Speech Emotion Recognition (SER) is widely deployed in Human-Computer Interaction, yet the high computational cost of conventional models hinders their implementation on resource-constrained edge devices. Spiking Neural Networks (SNNs) offer an energy-efficient alternative due to their event-driven nature; however, their integration with continuous Self-Supervised Learning (SSL) representations is fundamentally challenged by distribution mismatch, where high-dynamic-range embeddings degrade the information coding capacity of threshold-based neurons. To resolve this, we propose Prompt-Tuned Spiking Neural Networks (PTS-SNN), a parameter-efficient neuromorphic adaptation framework that aligns frozen SSL backbones with spiking dynamics. Specifically, we introduce a Temporal Shift Spiking Encoder to capture local temporal dependencies via parameter-free channel shifts, establishing a stable feature basis. To bridge the domain gap, we devise a Context-Aware Membrane Potential Calibration strategy. This mechanism leverages a Spiking Sparse Linear Attention module to aggregate global semantic context into learnable soft prompts, which dynamically regulate the bias voltages of Parametric Leaky Integrate-and-Fire (PLIF) neurons. This regulation effectively centers the heterogeneous input distribution within the responsive firing range, mitigating functional silence or saturation. Extensive experiments on five multilingual datasets (e.g., IEMOCAP, CASIA, EMODB) demonstrate that PTS-SNN achieves 73.34\% accuracy on IEMOCAP, comparable to competitive Artificial Neural Networks (ANNs), while requiring only 1.19M trainable parameters and 0.35 mJ inference energy per sample.
>
---
#### [new 019] Rho-Perfect: Correlation Ceiling For Subjective Evaluation Datasets
- **分类: cs.LG; eess.AS; stat.ML**

- **简介: 该论文属于模型评估任务，旨在解决主观评价数据中模型与人类评分相关性受限的问题。提出$ρ$-Perfect方法，估计模型在主观数据集上的最大可能相关性，区分模型局限与数据质量问题。**

- **链接: [https://arxiv.org/pdf/2602.08552v1](https://arxiv.org/pdf/2602.08552v1)**

> **作者:** Fredrik Cumlin
>
> **摘要:** Subjective ratings contain inherent noise that limits the model-human correlation, but this reliability issue is rarely quantified. In this paper, we present $ρ$-Perfect, a practical estimation of the highest achievable correlation of a model on subjectively rated datasets. We define $ρ$-Perfect to be the correlation between a perfect predictor and human ratings, and derive an estimate of the value based on heteroscedastic noise scenarios, a common occurrence in subjectively rated datasets. We show that $ρ$-Perfect squared estimates test-retest correlation and use this to validate the estimate. We demonstrate the use of $ρ$-Perfect on a speech quality dataset and show how the measure can distinguish between model limitations and data quality issues.
>
---
#### [new 020] Speech Emotion Recognition Leveraging OpenAI's Whisper Representations and Attentive Pooling Methods
- **分类: cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决数据不足和特征提取效率问题。通过引入Whisper模型和注意力池化方法，提升情感特征保留与模型效率。**

- **链接: [https://arxiv.org/pdf/2602.06000v1](https://arxiv.org/pdf/2602.06000v1)**

> **作者:** Ali Shendabadi; Parnia Izadirad; Mostafa Salehi; Mahmoud Bijankhan
>
> **摘要:** Speech Emotion Recognition (SER) research has faced limitations due to the lack of standard and sufficiently large datasets. Recent studies have leveraged pre-trained models to extract features for downstream tasks such as SER. This work explores the capabilities of Whisper, a pre-trained ASR system, in speech emotion recognition by proposing two attention-based pooling methods, Multi-head Attentive Average Pooling and QKV Pooling, designed to efficiently reduce the dimensionality of Whisper representations while preserving emotional features. We experiment on English and Persian, using the IEMOCAP and ShEMO datasets respectively, with Whisper Tiny and Small. Our multi-head QKV architecture achieves state-of-the-art results on the ShEMO dataset, with a 2.47% improvement in unweighted accuracy. We further compare the performance of different Whisper encoder layers and find that intermediate layers often perform better for SER on the Persian dataset, providing a lightweight and efficient alternative to much larger models such as HuBERT X-Large. Our findings highlight the potential of Whisper as a representation extractor for SER and demonstrate the effectiveness of attention-based pooling for dimension reduction.
>
---
#### [new 021] Video-based Music Generation
- **分类: cs.LG; cs.AI; cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出EMSYNC，解决视频配乐自动化问题。通过视频情绪分类、情感音乐生成和时间边界对齐，实现情绪与节奏同步的音乐生成。**

- **链接: [https://arxiv.org/pdf/2602.07063v1](https://arxiv.org/pdf/2602.07063v1)**

> **作者:** Serkan Sulun
>
> **备注:** PhD thesis, University of Porto
>
> **摘要:** As the volume of video content on the internet grows rapidly, finding a suitable soundtrack remains a significant challenge. This thesis presents EMSYNC (EMotion and SYNChronization), a fast, free, and automatic solution that generates music tailored to the input video, enabling content creators to enhance their productions without composing or licensing music. Our model creates music that is emotionally and rhythmically synchronized with the video. A core component of EMSYNC is a novel video emotion classifier. By leveraging pretrained deep neural networks for feature extraction and keeping them frozen while training only fusion layers, we reduce computational complexity while improving accuracy. We show the generalization abilities of our method by obtaining state-of-the-art results on Ekman-6 and MovieNet. Another key contribution is a large-scale, emotion-labeled MIDI dataset for affective music generation. We then present an emotion-based MIDI generator, the first to condition on continuous emotional values rather than discrete categories, enabling nuanced music generation aligned with complex emotional content. To enhance temporal synchronization, we introduce a novel temporal boundary conditioning method, called "boundary offset encodings," aligning musical chords with scene changes. Combining video emotion classification, emotion-based music generation, and temporal boundary conditioning, EMSYNC emerges as a fully automatic video-based music generator. User studies show that it consistently outperforms existing methods in terms of music richness, emotional alignment, temporal synchronization, and overall preference, setting a new state-of-the-art in video-based music generation.
>
---
## 更新

#### [replaced 001] Pronunciation Editing for Finnish Speech using Phonetic Posteriorgrams
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决低资源语言L2语音合成难题。通过PPG2Speech模型，无需文本对齐即可编辑单个音素，提升语音自然度与说话人相似性。**

- **链接: [https://arxiv.org/pdf/2507.02115v3](https://arxiv.org/pdf/2507.02115v3)**

> **作者:** Zirui Li; Lauri Juvela; Mikko Kurimo
>
> **备注:** Accepted by Proceeding of 13th edition of the Speech Synthesis Workshop; 5 pages, 1 figure
>
> **摘要:** Synthesizing second-language (L2) speech is potentially highly valued for L2 language learning experience and feedback. However, due to the lack of L2 speech synthesis datasets, it is difficult to synthesize L2 speech for low-resourced languages. In this paper, we provide a practical solution for editing native speech to approximate L2 speech and present PPG2Speech, a diffusion-based multispeaker Phonetic-Posteriorgrams-to-Speech model that is capable of editing a single phoneme without text alignment. We use Matcha-TTS's flow-matching decoder as the backbone, transforming Phonetic Posteriorgrams (PPGs) to mel-spectrograms conditioned on external speaker embeddings and pitch. PPG2Speech strengthens the Matcha-TTS's flow-matching decoder with Classifier-free Guidance (CFG) and Sway Sampling. We also propose a new task-specific objective evaluation metric, the Phonetic Aligned Consistency (PAC), between the edited PPGs and the PPGs extracted from the synthetic speech for editing effects. We validate the effectiveness of our method on Finnish, a low-resourced, nearly phonetic language, using approximately 60 hours of data. We conduct objective and subjective evaluations of our approach to compare its naturalness, speaker similarity, and editing effectiveness with TTS-based editing. Our source code is published at https://github.com/aalto-speech/PPG2Speech.
>
---
#### [replaced 002] Measuring Audio's Impact on Correctness: Audio-Contribution-Aware Post-Training of Large Audio Language Models
- **分类: eess.AS**

- **简介: 该论文属于音频语言模型的后训练任务，旨在解决模型依赖文本而非音频的问题。通过构建数据集和提出新的训练策略，提升模型对音频信息的利用能力。**

- **链接: [https://arxiv.org/pdf/2509.21060v3](https://arxiv.org/pdf/2509.21060v3)**

> **作者:** Haolin He; Xingjian Du; Renhe Sun; Zheqi Dai; Yujia Xiao; Mingru Yang; Jiayi Zhou; Xiquan Li; Zhengxi Liu; Zining Liang; Chunyat Wu; Qianhua He; Tan Lee; Xie Chen; Wei-Long Zheng; Weiqiang Wang; Mark Plumbley; Jian Liu; Qiuqiang Kong
>
> **摘要:** Large Audio Language Models (LALMs) represent an important frontier in multimodal AI, addressing diverse audio tasks. Recently, post-training of LALMs has received increasing attention due to significant performance improvements over foundation models. While single-stage post-training such as reinforcement learning (RL) has demonstrated promising results, multi-stage approaches such as supervised fine-tuning (SFT) followed by RL remain suboptimal. The allocation of data across multiple training stages to maximize LALM capabilities has not been fully explored, and large-scale, high-quality datasets for such research are also lacking. To address these problems, we firstly present AudioMCQ, a comprehensive audio multiple-choice question dataset comprising 571k samples with two kinds of chain-of-thought annotations. Secondly, we investigate the prevalent zero audio-contribution phenomenon in LALMs, where models derive correct answers solely from textual information without processing audio content. We propose Audio-Contribution Filtering to partition data into weak and strong audio-contribution subsets. Based on these insights, we develop two effective post-training paradigms: Weak-to-Strong (SFT on weak audio-contribution data followed by RL on strong audio-contribution data) and Mixed-to-Strong (SFT on mixed audio-contribution data followed by RL on strong audio-contribution data). We achieve first place in the DCASE 2025 Audio-Question-Answering challenge by using AudioMCQ. Additionally, leveraging our dataset with different training strategies, we achieve 78.2\% on MMAU-test-mini, 75.6\% on MMAU, 67.1\% on MMAR, and 70.7\% on MMSU, establishing new state-of-the-art performance.
>
---
#### [replaced 003] Non-Intrusive Automatic Speech Recognition Refinement: A Survey
- **分类: eess.AS**

- **简介: 该论文属于语音识别优化任务，旨在解决ASR系统在语音多样性、环境干扰和领域术语上的准确性问题。通过综述非侵入式优化方法，提出评估标准与研究方向。**

- **链接: [https://arxiv.org/pdf/2508.07285v2](https://arxiv.org/pdf/2508.07285v2)**

> **作者:** Mohammad Reza Peyghan; Saman Soleimani Roudi; Saeedreza Zouashkiani; Sajjad Amini; Fatemeh Rajabi; Shahrokh Ghaemmaghami
>
> **摘要:** Automatic Speech Recognition (ASR) has become an integral component of modern technology, powering applications such as voice-activated assistants, transcription services, and accessibility tools. Yet ASR systems continue to struggle with the inherent variability of human speech, such as accents, dialects, and speaking styles, as well as environmental interference, including background noise. Moreover, domain-specific conversations often employ specialized terminology, which can exacerbate transcription errors. These shortcomings not only degrade raw ASR accuracy but also propagate mistakes through subsequent natural language processing pipelines. Because redesigning an ASR model is costly and time-consuming, non-intrusive refinement techniques that leave the model's architecture unchanged have become increasingly popular. In this survey, we review current non-intrusive refinement approaches and group them into five classes: fusion, re-scoring, correction, distillation, and training adjustment. For each class, we outline the main methods, advantages, drawbacks, and ideal application scenarios. Beyond method classification, this work surveys adaptation techniques aimed at refining ASR in domain-specific contexts, reviews commonly used evaluation datasets along with their construction processes, and proposes a standardized set of metrics to facilitate fair comparisons. Finally, we identify open research gaps and suggest promising directions for future work. By providing this structured overview, we aim to equip researchers and practitioners with a clear foundation for developing more robust, accurate ASR refinement pipelines.
>
---
#### [replaced 004] Emotion-Aligned Generation in Diffusion Text to Speech Models via Preference-Guided Optimization
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音生成任务，旨在解决情感对齐问题。通过引入EASPO框架，实现更精细的情感控制，提升语音表达的自然度与表现力。**

- **链接: [https://arxiv.org/pdf/2509.25416v2](https://arxiv.org/pdf/2509.25416v2)**

> **作者:** Jiacheng Shi; Hongfei Du; Yangfan He; Y. Alicia Hong; Ye Gao
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Emotional text-to-speech seeks to convey affect while preserving intelligibility and prosody, yet existing methods rely on coarse labels or proxy classifiers and receive only utterance-level feedback. We introduce Emotion-Aware Stepwise Preference Optimization (EASPO), a post-training framework that aligns diffusion TTS with fine-grained emotional preferences at intermediate denoising steps. Central to our approach is EASPM, a time-conditioned model that scores noisy intermediate speech states and enables automatic preference pair construction. EASPO optimizes generation to match these stepwise preferences, enabling controllable emotional shaping. Experiments show superior performance over existing methods in both expressiveness and naturalness.
>
---
#### [replaced 005] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于对话系统任务，旨在解决多轮全双工语音语言模型评估不足的问题。提出MTR-DuplexBench基准，全面评估模型的对话质量、指令遵循与安全性。**

- **链接: [https://arxiv.org/pdf/2511.10262v2](https://arxiv.org/pdf/2511.10262v2)**

> **作者:** He Zhang; Wenqian Cui; Haoning Xu; Xiaohui Li; Lei Zhu; Haoli Bai; Shaohua Ma; Irwin King
>
> **备注:** Work in progress
>
> **摘要:** Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions, neglecting the complexities of multi-round communication. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. Also, existing benchmarks often focus solely on evaluating conversational features, neglecting other critical aspects. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark designed for a comprehensive multi-round evaluation of FD-SLMs. MTR-DuplexBench not only segments continuous full-duplex dialogues into discrete turns for turn-by-turn assessment but also incorporates various evaluation aspects, including conversational features, dialogue quality, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our benchmark. The benchmark and code will be available in the future.
>
---
#### [replaced 006] Nord-Parl-TTS: Finnish and Swedish TTS Dataset from Parliament Speech
- **分类: eess.AS**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决低资源语言数据不足的问题。通过整理北欧议会录音，构建了芬兰语和瑞典语的大规模开放TTS数据集。**

- **链接: [https://arxiv.org/pdf/2509.17988v2](https://arxiv.org/pdf/2509.17988v2)**

> **作者:** Zirui Li; Jens Edlund; Yicheng Gu; Nhan Phan; Lauri Juvela; Mikko Kurimo
>
> **备注:** Accepted by ICASSP 2026. 5 pages, 2 figures
>
> **摘要:** Text-to-speech (TTS) development is limited by scarcity of high-quality, publicly available speech data for most languages outside a few high-resource languages. We present Nord-Parl-TTS, an open TTS dataset for Finnish and Swedish based on speech found in the wild. Using recordings of Nordic parliamentary proceedings, we extract 900 hours of Finnish and 5090 hours of Swedish speech suitable for TTS training. The dataset is built using an adapted version of the Emilia data processing pipeline and includes unified evaluation sets to support model development and benchmarking. By offering open, large-scale data for Finnish and Swedish, Nord-Parl-TTS narrows the resource gap in TTS between high- and lower-resourced languages.
>
---
#### [replaced 007] The Combination of Several Decorrelation Methods to Improve Acoustic Feedback Cancellation
- **分类: eess.AS**

- **简介: 该论文属于声学反馈消除任务，旨在提升系统性能。通过结合多种去相关方法，优化系统结构，提高语音质量。**

- **链接: [https://arxiv.org/pdf/2602.06921v2](https://arxiv.org/pdf/2602.06921v2)**

> **作者:** Klaus Linhard; Philipp Bulling
>
> **摘要:** This paper extends an acoustic feedback cancellation system by incorporating multiple decorrelation methods. The baseline system is based on a frequency-domain Kalman filter implemented in a multi-delay structure. The proposed extensions include a variable time delay line, prediction, distortion compensation, and a simplified reverberation model. Each extension is analyzed, and a practical parameter range is defined. While existing literature often focuses on a single extension, such as prediction, to describe an optimal system, this work demonstrates that each individual extension contributes to performance improvements. Furthermore, the combination of all proposed extensions results in a superior system. The evaluation is conducted using publicly available datasets, with performance assessed through system distance metrics and the objective speech quality measure PSEQ.
>
---
#### [replaced 008] DegDiT: Controllable Audio Generation with Dynamic Event Graph Guided Diffusion Transformer
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于可控文本到音频生成任务，旨在解决时间定位、开放词汇和效率之间的平衡问题。提出DegDiT框架，利用动态事件图引导扩散Transformer实现高质量音频生成。**

- **链接: [https://arxiv.org/pdf/2508.13786v2](https://arxiv.org/pdf/2508.13786v2)**

> **作者:** Yisu Liu; Chenxing Li; Wanqian Zhang; Wenfu Wang; Meng Yu; Ruibo Fu; Zheng Lin; Weiping Wang; Dong Yu
>
> **摘要:** Controllable text-to-audio generation aims to synthesize audio from textual descriptions while satisfying user-specified constraints, including event types, temporal sequences, and onset and offset timestamps. This enables precise control over both the content and temporal structure of the generated audio. Despite recent progress, existing methods still face inherent trade-offs among accurate temporal localization, open-vocabulary scalability, and practical efficiency. To address these challenges, we propose DegDiT, a novel dynamic event graph-guided diffusion transformer framework for open-vocabulary controllable audio generation. DegDiT encodes the events in the description as structured dynamic graphs. The nodes in each graph are designed to represent three aspects: semantic features, temporal attributes, and inter-event connections. A graph transformer is employed to integrate these nodes and produce contextualized event embeddings that serve as guidance for the diffusion model. To ensure high-quality and diverse training data, we introduce a quality-balanced data selection pipeline that combines hierarchical event annotation with multi-criteria quality scoring, resulting in a curated dataset with semantic diversity. Furthermore, we present consensus preference optimization, facilitating audio generation through consensus among multiple reward signals. Extensive experiments on AudioCondition, DESED, and AudioTime datasets demonstrate that DegDiT achieves state-of-the-art performances across a variety of objective and subjective evaluation metrics.
>
---
#### [replaced 009] Fed-PISA: Federated Voice Cloning via Personalized Identity-Style Adaptation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音克隆任务，旨在解决联邦学习中通信成本高和风格个性化不足的问题。提出Fed-PISA，通过分离的LoRA机制降低通信开销，并利用协同过滤增强风格多样性。**

- **链接: [https://arxiv.org/pdf/2509.16010v2](https://arxiv.org/pdf/2509.16010v2)**

> **作者:** Qi Wang; Shituo Ma; Guoxin Yu; Hanyang Peng; Yue Yu
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Voice cloning for Text-to-Speech (TTS) aims to generate expressive and personalized speech from text using limited data from a target speaker. Federated Learning (FL) offers a collaborative and privacy-preserving framework for this task, but existing approaches suffer from high communication costs and tend to suppress stylistic heterogeneity, resulting in insufficient personalization. To address these issues, we propose Fed-PISA, which stands for Federated Personalized Identity-Style Adaptation. To minimize communication costs, Fed-PISA introduces a disentangled Low-Rank Adaptation (LoRA) mechanism: the speaker's timbre is retained locally through a private ID-LoRA, while only a lightweight style-LoRA is transmitted to the server, thereby minimizing parameter exchange. To harness heterogeneity, our aggregation method, inspired by collaborative filtering, is introduced to create custom models for each client by learning from stylistically similar peers. Experiments show that Fed-PISA improves style expressivity, naturalness, and speaker similarity, outperforming standard federated baselines with minimal communication costs.
>
---
#### [replaced 010] A Lightweight Architecture for Multi-instrument Transcription with Practical Optimizations
- **分类: cs.SD; cs.IR**

- **简介: 该论文属于多乐器音高转录任务，解决现有模型泛化能力差、计算量大等问题。提出轻量级架构，结合时序编码与深度聚类，实现高效准确的多乐器转录与分离。**

- **链接: [https://arxiv.org/pdf/2509.12712v3](https://arxiv.org/pdf/2509.12712v3)**

> **作者:** Ruigang Li; Yongxu Zhu
>
> **摘要:** Existing multi-timbre transcription models struggle with generalization beyond pre-trained instruments, rigid source-count constraints, and high computational demands that hinder deployment on low-resource devices. We address these limitations with a lightweight model that extends a timbre-agnostic transcription backbone with a dedicated timbre encoder and performs deep clustering at the note level, enabling joint transcription and dynamic separation of arbitrary instruments given a specified number of instrument classes. Practical optimizations including spectral normalization, dilated convolutions, and contrastive clustering further improve efficiency and robustness. Despite its small size and fast inference, the model achieves competitive performance with heavier baselines in terms of transcription accuracy and separation quality, and shows promising generalization ability, making it highly suitable for real-world deployment in practical and resource-constrained settings.
>
---
#### [replaced 011] VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音对话系统任务，旨在解决现有基准不足的问题。构建了VCB Bench，一个基于真实中文语音的评估基准，从多维度评估大语言模型的性能。**

- **链接: [https://arxiv.org/pdf/2510.11098v4](https://arxiv.org/pdf/2510.11098v4)**

> **作者:** Jiliang Hu; Wenfu Wang; Zuchao Li; Chenxing Li; Yiyang Zhao; Hanzhao Li; Liqiang Zhang; Meng Yu; Dong Yu
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.
>
---
#### [replaced 012] Differentiable Grouped Feedback Delay Networks for Learning Coupled Volume Acoustics
- **分类: eess.AS**

- **简介: 该论文属于声学建模任务，旨在解决动态混响渲染问题。提出DiffGFDN模型，高效模拟多斜率混响，适用于移动源和听者的XR应用。**

- **链接: [https://arxiv.org/pdf/2508.06686v2](https://arxiv.org/pdf/2508.06686v2)**

> **作者:** Orchisama Das; Gloria Dal Santo; Sebastian J. Schlecht; Vesa Valimaki; Zoran Cvetkovic
>
> **摘要:** Rendering dynamic reverberation in a complicated acoustic space for moving sources and listeners is challenging but crucial for enhancing user immersion in extended-reality (XR) applications. Capturing spatially varying room impulse responses (RIRs) is costly and often impractical. Moreover, dynamic convolution with measured RIRs is computationally expensive with high memory demands, typically not available on wearable computing devices. Grouped Feedback Delay Networks (GFDNs), on the other hand, allow efficient rendering of coupled room acoustics. However, its parameters need to be tuned to match the reverberation profile of a coupled space. In this work, we propose the concept of Differentiable GFDNs (DiffGFDNs), which have tunable parameters that are optimised to match the late reverberation profile of a set of RIRs captured from a space that exhibits multi-slope decay. Once trained on a finite set of measurements, the DiffGFDN interpolates to unmeasured locations in the space. We propose a parallel processing pipeline that has multiple DiffGFDNs with frequency-independent parameters processing each octave band. The parameters of the DiffGFDN can be updated rapidly during inferencing as sources and listeners move. We evaluate the proposed architecture against the Common Slopes (CS) model on a dataset of RIRs for three coupled rooms. The proposed architecture generates multi-slope late reverberation with low memory and computational requirements, achieving a better energy decay relief (EDR) error and slightly worse octave-band energy decay curve (EDC) errors compared to the CS model. Furthermore, DiffGFDN requires an order of magnitude fewer floating-point operations per sample than the CS renderer.
>
---
#### [replaced 013] STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出STITCH方法，解决语音语言模型在回应前缺乏内部思考的问题。通过交替生成思考和回应块，实现同步思考与说话，提升推理能力并保持低延迟。**

- **链接: [https://arxiv.org/pdf/2507.15375v2](https://arxiv.org/pdf/2507.15375v2)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Linjie Li; Chung-Ching Lin; Kevin Lin; Shujie Liu; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **备注:** ICLR 2026 camera-ready version. Project page: https://d223302.github.io/STITCH/
>
> **摘要:** Spoken Language Models (SLMs) are designed to take speech inputs and produce spoken responses. However, current SLMs lack the ability to perform an internal, unspoken thinking process before responding. In contrast, humans typically engage in complex mental reasoning internally, enabling them to communicate ideas clearly and concisely. Thus, integrating an unspoken thought process into SLMs is highly desirable. While naively generating a complete chain-of-thought (CoT) reasoning before starting to talk can enable thinking for SLMs, this induces additional latency for the speech response, as the CoT reasoning can be arbitrarily long. To solve this issue, we propose Stitch, a novel generation method that alternates between the generation of unspoken reasoning chunks and spoken response chunks. Since the audio duration of a chunk of spoken response is much longer than the time to generate the tokens in a chunk of spoken response, we use the remaining free time to generate the unspoken reasoning tokens. When a chunk of audio is played to the user, the model continues to generate the next unspoken reasoning chunk, achieving simultaneous thinking and talking. Remarkably, Stitch matches the latency of baselines that cannot generate unspoken CoT by design while outperforming those baselines by 15% on math reasoning datasets; Stitch also performs equally well on non-reasoning datasets as those baseline models. Some animations and demonstrations are on the project page: https://d223302.github.io/STITCH.
>
---
#### [replaced 014] Cross-Lingual F5-TTS: Towards Language-Agnostic Voice Cloning and Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，解决跨语言语音克隆中依赖参考文本的问题。通过强制对齐获取词边界，并训练语速预测器实现无需文本的跨语言合成。**

- **链接: [https://arxiv.org/pdf/2509.14579v4](https://arxiv.org/pdf/2509.14579v4)**

> **作者:** Qingyu Liu; Yushen Chen; Zhikang Niu; Chunhui Wang; Yunting Yang; Bowen Zhang; Jian Zhao; Pengcheng Zhu; Kai Yu; Xie Chen
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models have shown high-quality speech synthesis. However, most current flow-matching-based TTS models still rely on reference transcripts corresponding to the audio prompt for synthesis. This dependency prevents cross-lingual voice cloning when audio prompt transcripts are unavailable, particularly for unseen languages. The key challenges for flow-matching-based TTS models to remove audio prompt transcripts are identifying word boundaries during training and determining appropriate duration during inference. In this paper, we introduce Cross-Lingual F5-TTS, a framework that enables cross-lingual voice cloning without audio prompt transcripts. Our method preprocesses audio prompts by forced alignment to obtain word boundaries, enabling direct synthesis from audio prompts while excluding transcripts during training. To address the duration modeling challenge, we train speaking rate predictors at different linguistic granularities to derive duration from speaker pace. Experiments show that our approach matches the performance of F5-TTS while enabling cross-lingual voice cloning.
>
---
