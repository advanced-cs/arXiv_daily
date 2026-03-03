# 音频 cs.SD;  eess.AS

- **最新发布 32 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] ViTex: Visual Texture Control for Multi-Track Symbolic Music Generation via Discrete Diffusion Models
- **分类: cs.SD; cs.SC**

- **简介: 该论文属于多轨音乐生成任务，旨在通过视觉纹理控制实现更自然的人机交互。提出ViTex表示法，结合扩散模型生成高质量多轨音乐。**

- **链接: [https://arxiv.org/pdf/2603.01984](https://arxiv.org/pdf/2603.01984)**

> **作者:** Xiaoyu Yi; Qi He; Gus Xia; Ziyu Wang
>
> **摘要:** In automatic music generation, a central challenge is to design controls that enable meaningful human-machine interaction. Existing systems often rely on extrinsic inputs such as text prompts or metadata, which do not allow humans to directly shape the composition. While prior work has explored intrinsic controls such as chords or hierarchical structure, these approaches mainly address piano or vocal-accompaniment settings, leaving multitrack symbolic music largely underexplored. We identify instrumentation, the choice of instruments and their roles, as a natural dimension of control in multi-track composition, and propose ViTex, a visual representation of instrumental texture. In ViTex, color encodes instrument choice, spatial position represents pitch and time, and stroke properties capture local textures. Building on this representation, we develop a discrete diffusion model conditioned on ViTex and chord progressions to generate 8-measure multi-track symbolic music, enabling explicit texture-level control while maintaining strong unconditional generation quality. The demo page and code are avaliable at this https URL.
>
---
#### [new 002] Whisper-MLA: Reducing GPU Memory Consumption of ASR Models based on MHA2MLA Conversion
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决Transformer模型中MHA机制导致的GPU内存消耗过高的问题。通过引入MLA机制，有效降低KV缓存占用，提升内存效率。**

- **链接: [https://arxiv.org/pdf/2603.00563](https://arxiv.org/pdf/2603.00563)**

> **作者:** Sen Zhang; Jianguo Wei; Wenhuan Lu; Xianghu Yue; Wei Li; Qiang Li; Pengcheng Zhao; Ming Cai; Luo Si
>
> **备注:** 5 pages, 3 figures, accepted at ICASSP 2026
>
> **摘要:** The Transformer-based Whisper model has achieved state-of-the-art performance in Automatic Speech Recognition (ASR). However, its Multi-Head Attention (MHA) mechanism results in significant GPU memory consumption due to the linearly growing Key-Value (KV) cache usage, which is problematic for many applications especially with long-form audio. To address this, we introduce Whisper-MLA, a novel architecture that incorporates Multi-Head Latent Attention (MLA) into the Whisper model. Specifically, we adapt MLA for Whisper's absolute positional embeddings and systematically investigate its application across encoder self-attention, decoder self-attention, and cross-attention modules. Empirical results indicate that applying MLA exclusively to decoder self-attention yields the desired balance between performance and memory efficiency. Our proposed approach allows conversion of a pretrained Whisper model to Whisper-MLA with minimal fine-tuning. Extensive experiments on the LibriSpeech benchmark validate the effectiveness of this conversion, demonstrating that Whisper-MLA reduces the KV cache size by up to 87.5% while maintaining competitive accuracy.
>
---
#### [new 003] Analytical Exploration of Spatial Audio Cues: A Differentiable Multi-Sphere Scattering Model
- **分类: cs.SD**

- **简介: 该论文属于声学建模任务，旨在解决水下声散射建模问题。提出一种可微分的多球散射模型，用于提升声源定位与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.02205](https://arxiv.org/pdf/2603.02205)**

> **作者:** Siminfar Samakoush Galougah; Pranav Pulijala; Ramani Duraiswami
>
> **摘要:** A primary challenge in developing synthetic spatial hearing systems, particularly underwater, is accurately modeling sound scattering. Biological organisms achieve 3D spatial hearing by exploiting sound scattering off their bodies to generate location-dependent interaural level and time differences (ITD/ILD). While Head-Related Transfer Function (HRTF) models based on rigid scattering suffice for terrestrial humans, they fail in underwater environments due to the near-impedance match between water and soft tissue. Motivated by the acoustic anatomy of underwater animals, we introduce a novel, analytically derived, closed-form forward model for scattering from a semi-transparent sphere containing two rigid spherical scatterers. This model accurately maps source direction, frequency, and material properties to the pressure field, capturing the complex physics of layered, penetrable structures. Critically, our model is implemented in a fully differentiable setting, enabling its integration with a machine learning algorithm to optimize a cost function for active localization. We demonstrate enhanced convergence for localization under noise using a physics-informed frequency weighting scheme, and present accurate moving-source tracking via an Extended Kalman Filter (EKF) with analytically computed Jacobians. Our work suggests that differentiable models of scattering from layered rigid and transparent geometries offer a promising new foundation for microphone arrays that leverage scattering-based spatial cues over conventional beamforming, applicable to both terrestrial and underwater applications. Our model will be made open source.
>
---
#### [new 004] VoiceAgengRAG: Solving the RAG Latency Bottleneck in Real-Time Voice Agents Using Dual-Agent Architectures
- **分类: cs.SD**

- **简介: 该论文属于实时语音代理任务，解决RAG延迟瓶颈问题。通过双代理架构，分离检索与生成，提升响应速度。**

- **链接: [https://arxiv.org/pdf/2603.02206](https://arxiv.org/pdf/2603.02206)**

> **作者:** Jielin Qiu; Jianguo Zhang; Zixiang Chen; Liangwei Yang; Ming Zhu; Juntao Tan; Haolin Chen; Wenting Zhao; Rithesh Murthy; Roshan Ram; Akshara Prabhakar; Shelby Heinecke; Caiming; Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** We present VoiceAgentRAG, an open-source dual-agent memory router that decouples retrieval from response generation. A background Slow Thinker agent continuously monitors the conversation stream, predicts likely follow-up topics using an LLM, and pre-fetches relevant document chunks into a FAISS-backed semantic cache. A foreground Fast Talker agent reads only from this sub-millisecond cache, bypassing the vector database entirely on cache hits.
>
---
#### [new 005] Entropy-Guided GRVQ for Ultra-Low Bitrate Neural Speech Codec
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音编码任务，旨在解决超低比特率下高质量语音重建与语义建模的难题。提出熵引导的GRVQ方法，提升编码效率和语音质量。**

- **链接: [https://arxiv.org/pdf/2603.01476](https://arxiv.org/pdf/2603.01476)**

> **作者:** Yanzhou Ren; Noboru Harada; Daiki Takeuchi; Siyu Chen; Wei Liu; Xiao Zhang; Liyuan Zhang; Takehiro Moriya; Shoji Makino
>
> **摘要:** Neural audio codec (NAC) is essential for reconstructing high-quality speech signals and generating discrete representations for downstream speech language models. However, ensuring accurate semantic modeling while maintaining high-fidelity reconstruction under ultra-low bitrate constraints remains challenging. We propose an entropy-guided group residual vector quantization (EG-GRVQ) for an ultra-low bitrate neural speech codec, which retains a semantic branch for linguistic information and incorporates an entropy-guided grouping strategy in the acoustic branch. Assuming that channel activations follow approximately Gaussian statistics, the variance of each channel can serve as a principled proxy for its information content. Based on this assumption, we partition the encoder output such that each group carries an equal share of the total information. This balanced allocation improves codebook efficiency and reduces redundancy. Trained on LibriTTS and VCTK, our model shows improvements in perceptual quality and intelligibility metrics under ultra-low bitrate conditions, with a focus on codec-level fidelity for communication-oriented scenarios.
>
---
#### [new 006] End-to-End Simultaneous Dysarthric Speech Reconstruction with Frame-Level Adaptor and Multiple Wait-k Knowledge Distillation
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音重建任务，旨在解决慢速发音导致的延迟及语音识别误差问题。提出端到端系统，引入帧级适配器和多视角知识蒸馏，提升重建效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.01382](https://arxiv.org/pdf/2603.01382)**

> **作者:** Minghui Wu; Haitao Tang; Jiahuan Fan; Ruizhi Liao; Yanyong Zhang
>
> **备注:** Submitted to 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Dysarthric speech reconstruction (DSR) typically employs a cascaded system that combines automatic speech recognition (ASR) and sentence-level text-to-speech (TTS) to convert dysarthric speech into normally-prosodied speech. However, dysarthric individuals often speak more slowly, leading to excessively long response times in such systems, rendering them impractical in long-speech scenarios. Cascaded DSR systems based on streaming ASR and incremental TTS can help reduce latency. However, patients with differing dysarthria severity exhibit substantial pronunciation variability for the same text, resulting in poor robustness of ASR and limiting the intelligibility of reconstructed speech. In addition, incremental TTS suffers from poor prosodic feature prediction due to a limited receptive field. In this study, we propose an end-to-end simultaneous DSR system with two key innovations: 1) A frame-level adaptor module is introduced to bridge ASR and TTS. By employing explicit-implicit semantic information fusion and joint module training, it enhances the error tolerance of TTS to ASR outputs. 2) A multiple wait-k autoregressive TTS module is designed to mitigate prosodic degradation via multi-view knowledge distillation. Our system has an average response time of 1.03 seconds on Tesla A100, with an average real-time factor (RTF) of 0.71. On the UASpeech dataset, it attains a mean opinion score (MOS) of 4.67 and demonstrates a 54.25% relative reduction in word error rate (WER) compared to the state-of-the-art. Our demo is available at: this https URL
>
---
#### [new 007] VietSuperSpeech: A Large-Scale Vietnamese Conversational Speech Dataset for ASR Fine-Tuning in Chatbot, Customer Support, and Call Center Applications
- **分类: cs.SD**

- **简介: 该论文提出VietSuperSpeech，一个大规模越南语对话语音数据集，用于解决ASR在聊天机器人等场景中的口语识别问题。**

- **链接: [https://arxiv.org/pdf/2603.01894](https://arxiv.org/pdf/2603.01894)**

> **作者:** Loan Do; Thanh Ngoc Nguyen; Thanh Pham; Vinh Do; Hien Nguyen; Charlotte Nguyen
>
> **摘要:** We introduce VietSuperSpeech, a large-scale Vietnamese automatic speech recognition (ASR) dataset of 52,023 audio-text pairs totaling 267.39 hours, with a distinctive focus on casual conversational speech. Unlike existing Vietnamese ASR corpora that predominantly feature read speech, news narration, or audiobook content, VietSuperSpeech is sourced from four publicly accessible YouTube channels spanning everyday conversation, personal vlogging, overseas Vietnamese community dialogue, and informal commentary - the very speech styles encountered in real-world chatbot, customer support, call center, and hotline deployments. All audio is standardized to 16 kHz mono PCM WAV and segmented into 3-30 second utterances. Transcriptions are generated via pseudo-labeling using the Zipformer-30M-RNNT-6000h model (Nguyen, 2025) deployed through Sherpa-ONNX, pre-trained on 6,000 hours of Vietnamese speech. After quality filtering, the dataset is split into 46,822 training samples (240.67 hours) and 5,201 development/test samples (26.72 hours) with a fixed random seed. The text averages 266 characters per utterance, totaling 13.8 million fully diacritically marked Vietnamese characters. We demonstrate that VietSuperSpeech fills a critical gap in the Vietnamese ASR ecosystem: while corpora such as VLSP2020, VIET_BUD500, VietSpeech, FLEURS, VietMed, Sub-GigaSpeech2-Vi, viVoice, and Sub-PhoAudioBook provide broad coverage of formal and read speech, none specifically targets the casual, spontaneous register indispensable for conversational AI applications. VietSuperSpeech is publicly released at this https URL.
>
---
#### [new 008] Aurchestra: Fine-Grained, Real-Time Soundscape Control on Resource-Constrained Hearables
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出Aurchestra系统，解决听觉设备在复杂声场中无法精细控制的问题。通过动态界面和实时多输出网络，实现对不同声音的独立调整。**

- **链接: [https://arxiv.org/pdf/2603.00395](https://arxiv.org/pdf/2603.00395)**

> **作者:** Seunghyun Oh; Malek Itani; Aseem Gauri; Shyamnath Gollakota
>
> **备注:** 15 pages, 11 figures, 4 tables, submitted to ACM MobiSys 2026
>
> **摘要:** Hearables are becoming ubiquitous, yet their sound controls remain blunt: users can either enable global noise suppression or focus on a single target sound. Real-world acoustic scenes, however, contain many simultaneous sources that users may want to adjust independently. We introduce Aurchestra, the first system to provide fine-grained, real-time soundscape control on resource-constrained hearables. Our system has two key components: (1) a dynamic interface that surfaces only active sound classes and (2) a real-time, on-device multi-output extraction network that generates separate streams for each selected class, achieving robust performance for upto 5 overlapping target sounds, and letting users mix their environment by customizing per-class volumes, much like an audio engineer mixes tracks. We optimize the model architecture for multiple compute-limited platforms and demonstrate real-time performance on 6 ms streaming audio chunks. Across real-world environments in previously unseen indoor and outdoor scenarios, our system enables expressive per-class sound control and achieves substantial improvements in target-class enhancement and interference suppression. Our results show that the world need not be heard as a single, undifferentiated stream: with Aurchestra, the soundscape becomes truly programmable.
>
---
#### [new 009] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于多轨音乐生成任务，旨在解决节奏不稳定和不同轨同步不足的问题。提出SyncTrack模型，通过共享与专用模块提升节奏一致性和轨道同步性。**

- **链接: [https://arxiv.org/pdf/2603.01101](https://arxiv.org/pdf/2603.01101)**

> **作者:** Hongrui Wang; Fan Zhang; Zhiyuan Yu; Ziya Zhou; Xi Chen; Can Yang; Yang Wang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Multi-track music generation has garnered significant research interest due to its precise mixing and remixing capabilities. However, existing models often overlook essential attributes such as rhythmic stability and synchronization, leading to a focus on differences between tracks rather than their inherent properties. In this paper, we introduce SyncTrack, a synchronous multi-track waveform music generation model designed to capture the unique characteristics of multi-track music. SyncTrack features a novel architecture that includes track-shared modules to establish a common rhythm across all tracks and track-specific modules to accommodate diverse timbres and pitch ranges. Each track-shared module employs two cross-track attention mechanisms to synchronize rhythmic information, while each track-specific module utilizes learnable instrument priors to better represent timbre and other unique features. Additionally, we enhance the evaluation of multi-track music quality by introducing rhythmic consistency through three novel metrics: Inner-track Rhythmic Stability (IRS), Cross-track Beat Synchronization (CBS), and Cross-track Beat Dispersion (CBD). Experiments demonstrate that SyncTrack significantly improves the multi-track music quality by enhancing rhythmic consistency.
>
---
#### [new 010] Investigating Group Relative Policy Optimization for Diffusion Transformer based Text-to-Audio Generation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到音频生成任务，旨在解决复杂文本提示渲染不准确和文本-音频对齐问题。通过引入强化学习方法GRPO和大语言模型生成高质量音频描述，提升合成质量与提示匹配度。**

- **链接: [https://arxiv.org/pdf/2603.01565](https://arxiv.org/pdf/2603.01565)**

> **作者:** Yi Gu; Yanqing Liu; Chen Yang; Sheng Zhao
>
> **摘要:** Text-to-audio (T2A) generation has advanced considerably in recent years, yet existing methods continue to face challenges in accurately rendering complex text prompts, particularly those involving intricate audio effects, and achieving precise text-audio alignment. While prior approaches have explored data augmentation, explicit timing conditioning, and reinforcement learning, overall synthesis quality remains constrained. In this work, we experiment with reinforcement learning to further enhance T2A generation quality, building on diffusion transformer (DiT)-based architectures. Our method first employs a large language model (LLM) to generate high-fidelity, richly detailed audio captions, substantially improving text-audio semantic alignment, especially for ambiguous or underspecified prompts. We then apply Group Relative Policy Optimization (GRPO), a recently introduced reinforcement learning algorithm, to fine-tune the T2A model. Through systematic experimentation with diverse reward functions (including CLAP, KL, FAD, and their combinations), we identify the key drivers of effective RL in audio synthesis and analyze how reward design impacts final audio quality. Experimental results demonstrate that GRPO-based fine-tuning yield substantial gains in synthesis fidelity and prompt adherence.
>
---
#### [new 011] Voices of Civilizations: A Multilingual QA Benchmark for Global Music Understanding
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一个跨语言问答基准，用于评估音频大模型对音乐文化的理解能力，解决音乐文化理解不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.00533](https://arxiv.org/pdf/2603.00533)**

> **作者:** Shangda Wu; Ziya Zhou; Yongyi Zang; Yutong Zheng; Dafang Liang; Ruibin Yuan; Qiuqiang Kong
>
> **备注:** 2 pages, 2 figures, 1 table, accepted by ISMIR 2025 LBD
>
> **摘要:** We introduce Voices of Civilizations, the first multilingual QA benchmark for evaluating audio LLMs' cultural comprehension on full-length music recordings. Covering 380 tracks across 38 languages, our automated pipeline yields 1,190 multiple-choice questions through four stages - each followed by manual verification: 1) compiling a representative music list; 2) generating cultural-background documents for each sample in the music list via LLMs; 3) extracting key attributes from those documents; and 4) constructing multiple-choice questions probing language, region associations, mood, and thematic content. We evaluate models under four conditions and report per-language accuracy. Our findings demonstrate that even state-of-the-art audio LLMs struggle to capture subtle cultural nuances without rich textual context and exhibit systematic biases in interpreting music from different cultural traditions. The dataset is publicly available on Hugging Face to foster culturally inclusive music understanding research.
>
---
#### [new 012] Conversational Speech Naturalness Predictor
- **分类: eess.AS**

- **简介: 该论文属于语音自然度评估任务，旨在解决多说话人对话自然度预测问题。提出双通道模型，提升与人类判断的相关性。**

- **链接: [https://arxiv.org/pdf/2603.01467](https://arxiv.org/pdf/2603.01467)**

> **作者:** Anfeng Xu; Yashesh Gaur; Naoyuki Kanda; Zhicheng Ouyang; Katerina Zmolikova; Desh Raj; Simone Merello; Anna Sun; Ozlem Kalinli
>
> **备注:** Under review for Interspeech 2026
>
> **摘要:** Evaluation of conversational naturalness is essential for developing human-like speech agents. However, existing speech naturalness predictors are often designed to assess utterances from a single speaker, failing to capture conversation-level naturalness qualities. In this paper, we present a framework for an automatic naturalness predictor for two-speaker, multi-turn conversations. We first show that existing naturalness estimators have low, or sometimes even negative, correlations with conversational naturalness, based on conversational recordings annotated with human ratings. We then propose a dual-channel naturalness estimator, in which we investigate multiple pre-trained encoders with data augmentation. Our proposed model achieves substantially higher correlation with human judgments compared to existing naturalness predictors for both in-domain and out-of-domain conditions.
>
---
#### [new 013] A SUPERB-Style Benchmark of Self-Supervised Speech Models for Audio Deepfake Detection
- **分类: eess.AS; cs.AI; cs.LG; eess.SP**

- **简介: 该论文属于音频深度伪造检测任务，旨在评估自监督学习模型在该任务中的表现。工作包括构建基准Spoof-SUPERB，测试20种模型，并分析其鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.01482](https://arxiv.org/pdf/2603.01482)**

> **作者:** Hashim Ali; Nithin Sai Adupa; Surya Subramani; Hafiz Malik
>
> **备注:** Accepted at ICASSP
>
> **摘要:** Self-supervised learning (SSL) has transformed speech processing, with benchmarks such as SUPERB establishing fair comparisons across diverse downstream tasks. Despite it's security-critical importance, Audio deepfake detection has remained outside these efforts. In this work, we introduce Spoof-SUPERB, a benchmark for audio deepfake detection that systematically evaluates 20 SSL models spanning generative, discriminative, and spectrogram-based architectures. We evaluated these models on multiple in-domain and out-of-domain datasets. Our results reveal that large-scale discriminative models such as XLS-R, UniSpeech-SAT, and WavLM Large consistently outperform other models, benefiting from multilingual pretraining, speaker-aware objectives, and model scale. We further analyze the robustness of these models under acoustic degradations, showing that generative approaches degrade sharply, while discriminative models remain resilient. This benchmark establishes a reproducible baseline and provides practical insights into which SSL representations are most reliable for securing speech systems against audio deepfakes.
>
---
#### [new 014] CodecFlow: Efficient Bandwidth Extension via Conditional Flow Matching in Neural Codec Latent Space
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音带宽扩展任务，旨在提升低带宽语音的清晰度和可懂性。提出CodecFlow框架，通过神经编解码器潜空间实现高效重建，解决高频频谱保真度不足问题。**

- **链接: [https://arxiv.org/pdf/2603.02022](https://arxiv.org/pdf/2603.02022)**

> **作者:** Bowen Zhang; Junchuan Zhao; Ian McLoughlin; Ye Wang; A S Madhukumar
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Speech Bandwidth Extension improves clarity and intelligibility by restoring/inferring appropriate high-frequency content for low-bandwidth speech. Existing methods often rely on spectrogram or waveform modeling, which can incur higher computational cost and have limited high-frequency fidelity. Neural audio codecs offer compact latent representations that better preserve acoustic detail, yet accurately recovering high-resolution latent information remains challenging due to representation mismatch. We present CodecFlow, a neural codec-based BWE framework that performs efficient speech reconstruction in a compact latent space. CodecFlow employs a voicing-aware conditional flow converter on continuous codec embeddings and a structure-constrained residual vector quantizer to improve latent alignment stability. Optimized end-to-end, CodecFlow achieves strong spectral fidelity and enhanced perceptual quality on 8 kHz to 16 kHz and 44.1 kHz speech BWE tasks.
>
---
#### [new 015] Using Songs to Improve Kazakh Automatic Speech Recognition
- **分类: eess.AS**

- **简介: 该论文属于自动语音识别任务，旨在解决低资源语言Kazakh ASR数据不足的问题。通过使用歌曲数据进行模型微调，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2603.00961](https://arxiv.org/pdf/2603.00961)**

> **作者:** Rustem Yeshpanov
>
> **备注:** 9 pages, 7 tables, to appear in Proceedings of the 2026 Language Resources and Evaluation Conference
>
> **摘要:** Developing automatic speech recognition (ASR) systems for low-resource languages is hindered by the scarcity of transcribed corpora. This proof-of-concept study explores songs as an unconventional yet promising data source for Kazakh ASR. We curate a dataset of 3,013 audio-text pairs (about 4.5 hours) from 195 songs by 36 artists, segmented at the lyric-line level. Using Whisper as the base recogniser, we fine-tune models under seven training scenarios involving Songs, Common Voice Corpus (CVC), and FLEURS, and evaluate them on three benchmarks: CVC, FLEURS, and Kazakh Speech Corpus 2 (KSC2). Results show that song-based fine-tuning improves performance over zero-shot baselines. For instance, Whisper Large-V3 Turbo trained on a mixture of Songs, CVC, and FLEURS achieves 27.6% normalised WER on CVC and 11.8% on FLEURS, while halving the error on KSC2 (39.3% vs. 81.2%) relative to the zero-shot model. Although these gains remain below those of models trained on the 1,100-hour KSC2 corpus, they demonstrate that even modest song-speech mixtures can yield meaningful adaptation improvements in low-resource ASR. The dataset is released on Hugging Face for research purposes under a gated, non-commercial licence.
>
---
#### [new 016] TQCodec: Towards neural audio codec for high-fidelity music streaming
- **分类: cs.SD**

- **简介: 该论文提出TQCodec，属于音频编码任务，解决高保真音乐流媒体的编码问题，通过改进网络结构和比特分配策略，提升音质。**

- **链接: [https://arxiv.org/pdf/2603.01592](https://arxiv.org/pdf/2603.01592)**

> **作者:** Lixing He; Zhouxuan Chen; Mingshuai Liu; Xinran Sun; Wucheng Wang; Minfu Li; Lingcheng Kong; Weifeng Zhao; Wenjiang Zhou
>
> **摘要:** We propose TQCodec, a neural audio codec designed for high-bitrate, high-fidelity music streaming. Unlike existing neural codecs that primarily target ultra-low bitrates (<= 16kbps), TQCodec operates at 44.1 kHz and supports bitrates from 32 kbps to 128 kbps, aligning with the standard quality of modern music streaming platforms. The model adopts an encoder-decoder architecture based on SEANet for efficient on-device computation and introduces several enhancements: an imbalanced network design for improved quality with low overhead, SimVQ for mid-frequency detail preservation, and a phase-aware waveform loss. Additionally, we introduce a perception-driven band-wise bit allocation strategy to prioritize perceptually critical lower frequencies. Evaluations on diverse music datasets demonstrate that TQCodec achieves superior audio quality at target bitrates, making it well-suited for high-quality audio applications.
>
---
#### [new 017] CMI-RewardBench: Evaluating Music Reward Models with Compositional Multimodal Instruction
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成评估任务，解决音乐奖励模型评价不足的问题。构建了CMI-RewardBench基准和相关数据集，提出高效奖励模型CMI-RM，提升音乐生成质量与对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.00610](https://arxiv.org/pdf/2603.00610)**

> **作者:** Yinghao Ma; Haiwen Xia; Hewei Gao; Weixiong Chen; Yuxin Ye; Yuchen Yang; Sungkyun Chang; Mingshuo Ding; Yizhi Li; Ruibin Yuan; Simon Dixon; Emmanouil Benetos
>
> **摘要:** While music generation models have evolved to handle complex multimodal inputs mixing text, lyrics, and reference audio, evaluation mechanisms have lagged behind. In this paper, we bridge this critical gap by establishing a comprehensive ecosystem for music reward modeling under Compositional Multimodal Instruction (CMI), where the generated music may be conditioned on text descriptions, lyrics, and audio prompts. We first introduce CMI-Pref-Pseudo, a large-scale preference dataset comprising 110k pseudo-labeled samples, and CMI-Pref, a high-quality, human-annotated corpus tailored for fine-grained alignment tasks. To unify the evaluation landscape, we propose CMI-RewardBench, a unified benchmark that evaluates music reward models on heterogeneous samples across musicality, text-music alignment, and compositional instruction alignment. Leveraging these resources, we develop CMI reward models (CMI-RMs), a parameter-efficient reward model family capable of processing heterogeneous inputs. We evaluate their correlation with human judgments scores on musicality and alignment on CMI-Pref along with previous datasets. Further experiments demonstrate that CMI-RM not only correlates strongly with human judgments, but also enables effective inference-time scaling via top-k filtering. The necessary training data, benchmarks, and reward models are publicly available.
>
---
#### [new 018] DARS: Dysarthria-Aware Rhythm-Style Synthesis for ASR Enhancement
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决失语症语音识别困难的问题。提出DARS框架，通过合成失语症语音增强ASR性能。**

- **链接: [https://arxiv.org/pdf/2603.01369](https://arxiv.org/pdf/2603.01369)**

> **作者:** Minghui Wu; Xueling Liu; Jiahuan Fan; Haitao Tang; Yanyong Zhang; Yue Zhang
>
> **备注:** Submitted to 2025 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)
>
> **摘要:** Dysarthric speech exhibits abnormal prosody and significant speaker variability, presenting persistent challenges for automatic speech recognition (ASR). While text-to-speech (TTS)-based data augmentation has shown potential, existing methods often fail to accurately model the pathological rhythm and acoustic style of dysarthric speech. To address this, we propose DARS, a dysarthria-aware rhythm-style synthesis framework based on the Matcha-TTS architecture. DARS incorporates a multi-stage rhythm predictor optimized by contrastive preferences between normal and dysarthric speech, along with a dysarthric-style conditional flow matching mechanism, jointly enhancing temporal rhythm reconstruction and pathological acoustic style simulation. Experiments on the TORGO dataset demonstrate that DARS achieves a Mean Cepstral Distortion (MCD) of 4.29, closely approximating real dysarthric speech. Adapting a Whisper-based ASR system with synthetic dysarthric speech from DARS achieves a 54.22% relative reduction in word error rate (WER) compared to state-of-the-art methods, demonstrating the framework's effectiveness in enhancing recognition performance.
>
---
#### [new 019] VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging Speaker Modeling
- **分类: eess.AS; cs.CL; cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出VoxKnesset数据集，用于研究语音随年龄变化的问题。任务为说话人建模与年龄预测，解决语音系统在长期变化中的性能下降问题。工作包括数据收集、模型基准测试及结果分析。**

- **链接: [https://arxiv.org/pdf/2603.01270](https://arxiv.org/pdf/2603.01270)**

> **作者:** Yanir Marmor; Arad Zulti; David Krongauz; Adam Gabet; Yoad Snapir; Yair Lifshitz; Eran Segal
>
> **备注:** 4 pages, 5 figures, 2 tables
>
> **摘要:** Speech processing systems face a fundamental challenge: the human voice changes with age, yet few datasets support rigorous longitudinal evaluation. We introduce VoxKnesset, an open-access dataset of ~2,300 hours of Hebrew parliamentary speech spanning 2009-2025, comprising 393 speakers with recording spans of up to 15 years. Each segment includes aligned transcripts and verified demographic metadata from official parliamentary records. We benchmark modern speech embeddings (WavLM-Large, ECAPA-TDNN, Wav2Vec2-XLSR-1B) on age prediction and speaker verification under longitudinal conditions. Speaker verification EER rises from 2.15\% to 4.58\% over 15 years for the strongest model, and cross-sectionally trained age regressors fail to capture within-speaker aging, while longitudinally trained models recover a meaningful temporal signal. We publicly release the dataset and pipeline to support aging-robust speech systems and Hebrew speech processing.
>
---
#### [new 020] Efficient Long-Sequence Diffusion Modeling for Symbolic Music Generation
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于符号音乐生成任务，解决长序列音乐生成中训练和推理成本高的问题。提出SMDIM模型，结合全局结构与局部细化，提升生成质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.00576](https://arxiv.org/pdf/2603.00576)**

> **作者:** Jinhan Xu; Xing Tang; Houpeng Yang; Haoran Zhang; Shenghua Yuan; Jiatao Chen; Tianming Xi; Jing Wang; Jiaojiao Yu; Guangli Xiang
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Symbolic music generation is a challenging task in multimedia generation, involving long sequences with hierarchical temporal structures, long-range dependencies, and fine-grained local details. Though recent diffusion-based models produce high quality generations, they tend to suffer from high training and inference costs with long symbolic sequences due to iterative denoising and sequence-length-related costs. To deal with such problem, we put forth a diffusing strategy named SMDIM to combine efficient global structure construction and light local refinement. SMDIM uses structured state space models to capture long range musical context at near linear cost, and selectively refines local musical details via a hybrid refinement scheme. Experiments performed on a wide range of symbolic music datasets which encompass various Western classical music, popular music and traditional folk music show that the SMDIM model outperforms the other state-of-the-art approaches on both the generation quality and the computational efficiency, and it has robust generalization to underexplored musical styles. These results show that SMDIM offers a principled solution for long-sequence symbolic music generation, including associated attributes that accompany the sequences. We provide a project webpage with audio examples and supplementary materials at this https URL.
>
---
#### [new 021] AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching
- **分类: cs.SD; cs.AI; cs.LG; cs.MM**

- **简介: 该论文属于音频生成任务，解决音频流匹配中表示对齐的层选择问题。提出AG-REPA方法，通过因果贡献评估选择关键层，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2603.01006](https://arxiv.org/pdf/2603.01006)**

> **作者:** Pengfei Zhang; Tianxin Xie; Minghao Yang; Li Liu
>
> **备注:** 13 pages, 4 figures, 4 tables
>
> **摘要:** REPresentation Alignment (REPA) improves the training of generative flow models by aligning intermediate hidden states with pretrained teacher features, but its effectiveness in token-conditioned audio Flow Matching critically depends on the choice of supervised layers, which is typically made heuristically based on the depth. In this work, we introduce Attribution-Guided REPresentation Alignment (AG-REPA), a novel causal layer selection strategy for representation alignment in audio Flow Matching. Firstly, we find that layers that best store semantic/acoustic information (high teacher-space similarity) are not necessarily the layers that contribute most to the velocity field that drives generation, and we call it Store-Contribute Dissociation (SCD). To turn this insight into an actionable training guidance, we propose a forward-only gate ablation (FoG-A) that quantifies each layer's causal contribution via the induced change in the predicted velocity field, enabling sparse layer selection and adaptive weighting for alignment. Across unified speech and general-audio training (LibriSpeech + AudioSet) under different token-conditioning topologies, AG-REPA consistently outperforms REPA baselines. Overall, our results show that alignment is most effective when applied to the causally dominant layers that drive the velocity field, rather than to layers that are representationally rich but functionally passive.
>
---
#### [new 022] Inter-Speaker Relative Cues for Two-Stage Text-Guided Target Speech Extraction
- **分类: eess.AS**

- **简介: 该论文属于语音提取任务，旨在提升文本引导的目标语音分离效果。通过引入相对线索，构建两阶段框架，增强分类准确性和分离性能。**

- **链接: [https://arxiv.org/pdf/2603.01316](https://arxiv.org/pdf/2603.01316)**

> **作者:** Wang Dai; Archontis Politis; Tuomas Virtanen
>
> **摘要:** This paper investigates the use of relative cues for text-based target speech extraction (TSE). We first provide a theoretical justification for relative cues from the perspectives of human perception and label quantization, showing that relative cues preserve fine-grained distinctions often lost in absolute categorical representations. Building on this analysis, we propose a two-stage TSE framework, in which a speech separation model generates candidate sources, followed by a text-guided classifier that selects the target speaker based on embedding similarity. Using this framework, we train two separate classification models to evaluate the advantages of relative cues over independent cues in terms of both classification accuracy and TSE performance. Experimental results demonstrate that (i) relative cues achieve higher overall classification accuracy and improved TSE performance compared with independent cues, (ii) the two-stage framework substantially outperforms single-stage text-conditioned extraction methods on both signal-level and objective perceptual metrics, and (iii) certain relative cues (language, gender, loudness, distance, temporal order, speaking duration, random cue and all cue) can surpass the performance of an audio-based TSE system. Further analysis reveals notable differences in discriminative power across cue types, providing insights into the effectiveness of different relative cues for TSE.
>
---
#### [new 023] TCG CREST System Description for the DISPLACE-M Challenge
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于说话人二分类任务，旨在提升医疗场景下的语音分离性能。通过比较不同VAD方法和聚类算法，优化了二分类系统，提升了准确率。**

- **链接: [https://arxiv.org/pdf/2603.02030](https://arxiv.org/pdf/2603.02030)**

> **作者:** Nikhil Raghav; Md Sahidullah
>
> **备注:** Report submitted for the DISPLACE-M challenge
>
> **摘要:** This report presents the TCG CREST system description for Track 1 (Speaker Diarization) of the DISPLACE-M challenge, focusing on naturalistic medical conversations in noisy rural-healthcare scenarios. Our study evaluates the impact of various voice activity detection (VAD) methods and advanced clustering algorithms on overall speaker diarization (SD) performance. We compare and analyze two SD frameworks: a modular pipeline utilizing SpeechBrain with ECAPA-TDNN embeddings, and a state-of-the-art (SOTA) hybrid end-to-end neural diarization system, Diarizen, built on top of a pre-trained WavLM. With these frameworks, we explore diverse clustering techniques, including agglomerative hierarchical clustering (AHC), and multiple novel variants of spectral clustering, such as SC-adapt, SC-PNA, and SC-MK. Experimental results demonstrate that the Diarizen system provides an approximate $39\%$ relative improvement in the diarization error rate (DER) on the post-evaluation analysis of Phase~I compared to the SpeechBrain baseline. Our best-performing submitted system employing the Diarizen baseline with AHC employing a median filtering with a larger context window of $29$ achieved a DER of 10.37\% on the development and 9.21\% on the evaluation sets, respectively. Our team ranked sixth out of the 11 participating teams after the Phase~I evaluation.
>
---
#### [new 024] The USTC-NERCSLIP Systems for the CHiME-9 MCoRec Challenge
- **分类: eess.AS**

- **简介: 该论文针对多对话识别与聚类任务，解决室内社交场景下多并发对话的语音识别与 speaker 聚类问题。通过融合音视频信息提升识别与聚类效果。**

- **链接: [https://arxiv.org/pdf/2603.01415](https://arxiv.org/pdf/2603.01415)**

> **作者:** Ya Jiang; Ruoyu Wang; Jingxuan Zhang; Jun Du; Yi Han; Zihao Quan; Hang Chen; Yeran Yang; Kongzhi Zheng; Zhuo Chen; Yanhui Tu; Shutong Niu; Changfeng Xi; Mengzhi Wang; Zhongbin Wu; Jieru Chen; Henghui Zhi; Weiyi Shi; Shuhang Wu; Genshun Wan; Jia Pan; Jianqing Gao
>
> **摘要:** This report details our submission to the CHiME-9 MCoRec Challenge on recognizing and clustering multiple concurrent natural conversations within indoor social settings. Unlike conventional meetings centered on a single shared topic, this scenario contains multiple parallel dialogues--up to eight speakers across up to four simultaneous conversations--with a speech overlap rate exceeding 90%. To tackle this, we propose a multimodal cascaded system that leverages per-speaker visual streams extracted from synchronized 360 degree video together with single-channel audio. Our system improves three components of the pipeline by leveraging enhanced audio-visual pretrained models: Active Speaker Detection (ASD), Audio-Visual Target Speech Extraction (AVTSE), and Audio-Visual Speech Recognition (AVSR). The AVSR module further incorporates Whisper and LLM techniques to boost transcription accuracy. Our best single cascaded system achieves a Speaker Word Error Rate (WER) of 32.44% on the development set. By further applying ROVER to fuse outputs from diverse front-end and back-end variants, we reduce Speaker WER to 31.40%. Notably, our LLM-based zero-shot conversational clustering achieves a speaker clustering F1 score of 1.0, yielding a final Joint ASR-Clustering Error Rate (JACER) of 15.70%.
>
---
#### [new 025] SpectroFusion-ViT: A Lightweight Transformer for Speech Emotion Recognition Using Harmonic Mel-Chroma Fusion
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音情感识别任务，旨在解决传统方法在准确性和效率上的不平衡问题。提出SpectroFusion-ViT模型，融合频谱特征，实现高效准确的情感分类。**

- **链接: [https://arxiv.org/pdf/2603.00746](https://arxiv.org/pdf/2603.00746)**

> **作者:** Faria Ahmed; Rafi Hassan Chowdhury; Fatema Tuz Zohora Moon; Sabbir Ahmed
>
> **摘要:** Speech is a natural means of conveying emotions, making it an effective method for understanding and representing human feelings. Reliable speech emotion recognition (SER) is central to applications in human-computer interaction, healthcare, education, and customer service. However, most SER methods depend on heavy backbone models or hand-crafted features that fail to balance accuracy and efficiency, particularly for low-resource languages like Bangla. In this work, we present SpectroFusion-ViT, a lightweight SER framework built utilizing EfficientViT-b0, a compact Vision Transformer architecture equipped with self-attention to capture long-range temporal and spectral patterns. The model contains only 2.04M parameters and requires 0.1 GFLOPs, enabling deployment in resource-constrained settings without compromising accuracy. Our pipeline first performs preprocessing and augmentation on raw audio, then extracts Chroma and Mel-frequency cepstral coefficient (MFCC) features. These representations are fused into a complementary time-frequency descriptor that preserves both fine-grained spectral detail and broader harmonic structure. Using transfer learning, EfficientViT-b0 is fine-tuned for multi-class emotion classification. We evaluate the system on two benchmark Bangla emotional speech datasets, SUBESCO and BanglaSER, which vary in speaker diversity, recording conditions, and acoustic characteristics. The proposed approach achieves 92.56% accuracy on SUBESCO and 82.19% on BanglaSER, surpassing existing state-of-the-art methods. These findings demonstrate that lightweight transformer architectures can deliver robust SER performance while remaining computationally efficient for real-world deployment.
>
---
#### [new 026] StethoLM: Audio Language Model for Cardiopulmonary Analysis Across Clinical Tasks
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出StethoLM，一个用于心肺听诊的音频语言模型，解决临床听诊自动化与可解释性问题，通过多阶段训练提升任务性能。**

- **链接: [https://arxiv.org/pdf/2603.00355](https://arxiv.org/pdf/2603.00355)**

> **作者:** Yishan Wang; Tsai-Ning Wang; Mathias Funk; Aaqib Saeed
>
> **备注:** To be published in TMLR
>
> **摘要:** Listening to heart and lung sounds - auscultation - is one of the first and most fundamental steps in a clinical examination. Despite being fast and non-invasive, it demands years of experience to interpret subtle audio cues. Recent deep learning methods have made progress in automating cardiopulmonary sound analysis, yet most are restricted to simple classification and offer little clinical interpretability or decision support. We present StethoLM, the first audio-language model specialized for cardiopulmonary auscultation, capable of performing instruction-driven clinical tasks across the full spectrum of auscultation analysis. StethoLM integrates audio encoding with a medical language model backbone and is trained on StethoBench, a comprehensive benchmark comprising 77,027 instruction-response pairs synthesized from 16,125 labeled cardiopulmonary recordings spanning seven clinical task categories: binary classification, detection, reporting, reasoning, differential diagnosis, comparison, and location-based analysis. Through multi-stage training that combines supervised fine-tuning and direct preference optimization, StethoLM achieves substantial gains in performance and robustness on out-of-distribution data. Our work establishes a foundation for instruction-following AI systems in clinical auscultation.
>
---
#### [new 027] Anatomy of the Modality Gap: Dissecting the Internal States of End-to-End Speech LLMs
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音与文本模态间的性能差距，分析端到端语音大模型的内部表示演化，揭示其结构稳定性及瓶颈所在。**

- **链接: [https://arxiv.org/pdf/2603.01502](https://arxiv.org/pdf/2603.01502)**

> **作者:** Ming-Hao Hsu; Xueyao Zhang; Xiaohai Tian; Jun Zhang; Zhizheng Wu
>
> **摘要:** Recent advancements in Large Speech-Language Models have significantly bridged the gap between acoustic signals and linguistic understanding. However, a persistent performance disparity remains in speech-based input tasks compared to direct text inference. In this paper, we investigate the dynamic roots of this modality gap beyond static geometric alignment, analyzing how speech and text representations evolve layer-by-layer. We evaluate four open-weight end-to-end models on SpeechMMLU and VoiceBench BBH. Using cross-layer CKA analysis with speech-text token alignment, we find that speech representations exhibit a broad cross-layer alignment band, attributable to the redundant nature of speech where semantic content spans multiple frames. We show that these alignment patterns are structurally stable across different analysis configurations. Crucially, simple statistical calibration is insufficient and can be detrimental when applied at the input layer, indicating that the modality gap is not a mere distribution shift. Overall, our results suggest that the bottleneck lies in condensing redundant speech into stable late-layer decisions, motivating future solutions that operate at the token or temporal granularity instead of feature-level matching.
>
---
#### [new 028] Acoustic Sensing for Universal Jamming Grippers
- **分类: cs.RO; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于机器人感知任务，旨在解决传统传感器影响抓取性能的问题。通过声学传感实现柔性抓取器的自感知，提升物体识别与抓取能力。**

- **链接: [https://arxiv.org/pdf/2603.00351](https://arxiv.org/pdf/2603.00351)**

> **作者:** Lion Weber; Theodor Wienert; Martin Splettstößer; Alexander Koenig; Oliver Brock
>
> **备注:** Accepted at ICRA 2026, supplementary material under this https URL
>
> **摘要:** Universal jamming grippers excel at grasping unknown objects due to their compliant bodies. Traditional tactile sensors can compromise this compliance, reducing grasping performance. We present acoustic sensing as a form of morphological sensing, where the gripper's soft body itself becomes the sensor. A speaker and microphone are placed inside the gripper cavity, away from the deformable membrane, fully preserving compliance. Sound propagates through the gripper and object, encoding object properties, which are then reconstructed via machine learning. Our sensor achieves high spatial resolution in sensing object size (2.6 mm error) and orientation (0.6 deg error), remains robust to external noise levels of 80 dBA, and discriminates object materials (up to 100% accuracy) and 16 everyday objects (85.6% accuracy). We validate the sensor in a realistic tactile object sorting task, achieving 53 minutes of uninterrupted grasping and sensing, confirming the preserved grasping performance. Finally, we demonstrate that disentangled acoustic representations can be learned, improving robustness to irrelevant acoustic variations.
>
---
#### [new 029] Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于法语临床对话的语音转录与说话人辨识任务，旨在降低转录错误率。通过多轮LLM后处理提升准确性，实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00086](https://arxiv.org/pdf/2603.00086)**

> **作者:** Ambre Marie; Thomas Bertin; Guillaume Dardenne; Gwenolé Quellec
>
> **摘要:** Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p < 0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment.
>
---
#### [new 030] Towards Orthographically-Informed Evaluation of Speech Recognition Systems for Indian Languages
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别评估任务，解决印度语言ASR系统评价不准确的问题，通过引入考虑拼写变体的OIWER指标，提升评估与实际表现的一致性。**

- **链接: [https://arxiv.org/pdf/2603.00941](https://arxiv.org/pdf/2603.00941)**

> **作者:** Kaushal Santosh Bhogale; Tahir Javed; Greeshma Susan John; Dhruv Rathi; Akshayasree Padmanaban; Niharika Parasa; Mitesh M. Khapra
>
> **备注:** Accepted in ICASSP 2026
>
> **摘要:** Evaluating ASR systems for Indian languages is challenging due to spelling variations, suffix splitting flexibility, and non-standard spellings in code-mixed words. Traditional Word Error Rate (WER) often presents a bleaker picture of system performance than what human users perceive. Better aligning evaluation with real-world performance requires capturing permissible orthographic variations, which is extremely challenging for under-resourced Indian languages. Leveraging recent advances in LLMs, we propose a framework for creating benchmarks that capture permissible variations. Through extensive experiments, we demonstrate that OIWER, by accounting for orthographic variations, reduces pessimistic error rates (an average improvement of 6.3 points), narrows inflated model gaps (e.g., Gemini-Canary performance difference drops from 18.1 to 11.5 points), and aligns more closely with human perception than prior methods like WER-SN by 4.9 points.
>
---
#### [new 031] FlowPortrait: Reinforcement Learning for Audio-Driven Portrait Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.SD**

- **简介: 该论文提出FlowPortrait，用于生成高质量的音频驱动人脸视频，解决唇形同步、动作自然度等问题，通过强化学习提升生成效果。**

- **链接: [https://arxiv.org/pdf/2603.00159](https://arxiv.org/pdf/2603.00159)**

> **作者:** Weiting Tan; Andy T. Liu; Ming Tu; Xinghua Qu; Philipp Koehn; Lu Lu
>
> **摘要:** Generating realistic talking-head videos remains challenging due to persistent issues such as imperfect lip synchronization, unnatural motion, and evaluation metrics that correlate poorly with human perception. We propose FlowPortrait, a reinforcement-learning framework for audio-driven portrait animation built on a multimodal backbone for autoregressive audio-to-video generation. FlowPortrait introduces a human-aligned evaluation system based on Multimodal Large Language Models (MLLMs) to assess lip-sync accuracy, expressiveness, and motion quality. These signals are combined with perceptual and temporal consistency regularizers to form a stable composite reward, which is used to post-train the generator via Group Relative Policy Optimization (GRPO). Extensive experiments, including both automatic evaluations and human preference studies, demonstrate that FlowPortrait consistently produces higher-quality talking-head videos, highlighting the effectiveness of reinforcement learning for portrait animation.
>
---
#### [new 032] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出UniTalking，用于生成高保真语音和唇形同步视频的统一框架，解决音频视频生成中的同步与质量问题。**

- **链接: [https://arxiv.org/pdf/2603.01418](https://arxiv.org/pdf/2603.01418)**

> **作者:** Hebeizi Li; Zihao Liang; Benyuan Sun; Zihao Yin; Xiao Sha; Chenliang Wang; Yi Yang
>
> **备注:** Accepted at CVPR 2026 (Findings Track)
>
> **摘要:** While state-of-the-art audio-video generation models like Veo3 and Sora2 demonstrate remarkable capabilities, their closed-source nature makes their architectures and training paradigms inaccessible. To bridge this gap in accessibility and performance, we introduce UniTalking, a unified, end-to-end diffusion framework for generating high-fidelity speech and lip-synchronized video. At its core, our framework employs Multi-Modal Transformer Blocks to explicitly model the fine-grained temporal correspondence between audio and video latent tokens via a shared self-attention mechanism. By leveraging powerful priors from a pre-trained video generation model, our framework ensures state-of-the-art visual fidelity while enabling efficient training. Furthermore, UniTalking incorporates a personalized voice cloning capability, allowing the generation of speech in a target style from a brief audio reference. Qualitative and quantitative results demonstrate that our method produces highly realistic talking portraits, achieving superior performance over existing open-source approaches in lip-sync accuracy, audio naturalness, and overall perceptual quality.
>
---
## 更新

#### [replaced 001] GACA-DiT: Diffusion-based Dance-to-Music Generation with Genre-Adaptive Rhythm and Context-Aware Alignment
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于舞蹈到音乐生成任务，旨在解决节奏不一致和时间错位问题。提出GACA-DiT框架，包含自适应节奏提取和上下文对齐模块，提升音乐与舞蹈的同步性。**

- **链接: [https://arxiv.org/pdf/2510.26818](https://arxiv.org/pdf/2510.26818)**

> **作者:** Jinting Wang; Chenxing Li; Li Liu
>
> **备注:** 5 pages, 4 figures, submitted to Interspeech2026
>
> **摘要:** Dance-to-music (D2M) generation aims to automatically compose music that is rhythmically and temporally aligned with dance movements. Existing methods typically rely on coarse rhythm embeddings, such as global motion features or binarized joint-based rhythm values, which discard fine-grained motion cues and result in weak rhythmic alignment. Moreover, temporal mismatches introduced by feature downsampling further hinder precise synchronization between dance and music. To address these problems, we propose \textbf{GACA-DiT}, a diffusion transformer-based framework with two novel modules for rhythmically consistent and temporally aligned music generation. First, a \textbf{genre-adaptive rhythm extraction} module combines multi-scale temporal wavelet analysis and spatial phase histograms with adaptive joint weighting to capture fine-grained, genre-specific rhythm patterns. Second, a \textbf{context-aware temporal alignment} module resolves temporal mismatches using learnable context queries to align music latents with relevant dance rhythm features. Extensive experiments on the AIST++ and TikTok datasets demonstrate that GACA-DiT outperforms state-of-the-art methods in both objective metrics and human evaluation. Project page: this https URL.
>
---
#### [replaced 002] Speech Emotion Recognition with ASR Integration
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别任务，旨在解决真实场景下情感识别困难的问题。通过集成自动语音识别技术，提升情感识别的鲁棒性和实用性。**

- **链接: [https://arxiv.org/pdf/2601.17901](https://arxiv.org/pdf/2601.17901)**

> **作者:** Yuanchao Li
>
> **备注:** PhD Thesis
>
> **摘要:** Speech Emotion Recognition (SER) plays a pivotal role in understanding human communication, enabling emotionally intelligent systems, and serving as a fundamental component in the development of Artificial General Intelligence (AGI). However, deploying SER in real-world, spontaneous, and low-resource scenarios remains a significant challenge due to the complexity of emotional expression and the limitations of current speech and language technologies. This thesis investigates the integration of Automatic Speech Recognition (ASR) into SER, with the goal of enhancing the robustness, scalability, and practical applicability of emotion recognition from spoken language.
>
---
#### [replaced 003] Universal Robust Speech Adaptation for Cross-Domain Speech Recognition and Enhancement
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音识别与增强任务，旨在解决域迁移导致的性能下降问题。提出URSA-GAN框架，通过双编码器和动态扰动提升模型在噪声和信道失配下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.04307](https://arxiv.org/pdf/2602.04307)**

> **作者:** Chien-Chun Wang; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to IEEE Transactions on Audio, Speech and Language Processing (IEEE TASLP)
>
> **摘要:** Pre-trained models for automatic speech recognition (ASR) and speech enhancement (SE) have exhibited remarkable capabilities under matched noise and channel conditions. However, these models often suffer from severe performance degradation when confronted with domain shifts, particularly in the presence of unseen noise and channel distortions. In view of this, we in this paper present URSA-GAN, a unified and domain-aware generative framework specifically designed to mitigate mismatches in both noise and channel conditions. URSA-GAN leverages a dual-embedding architecture that consists of a noise encoder and a channel encoder, each pre-trained with limited in-domain data to capture domain-relevant representations. These embeddings condition a GAN-based speech generator, facilitating the synthesis of speech that is acoustically aligned with the target domain while preserving phonetic content. To enhance generalization further, we propose dynamic stochastic perturbation, a novel regularization technique that introduces controlled variability into the embeddings during generation, promoting robustness to unseen domains. Empirical results demonstrate that URSA-GAN effectively reduces character error rates in ASR and improves perceptual metrics in SE across diverse noisy and mismatched channel scenarios. Notably, evaluations on compound test conditions with both channel and noise degradations confirm the generalization ability of URSA-GAN, yielding relative improvements of 16.16% in ASR performance and 15.58% in SE metrics.
>
---
#### [replaced 004] Depth-Structured Music Recurrence: Budgeted Recurrent Attention for Full-Piece Symbolic Music Modeling
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于符号音乐生成任务，解决长上下文建模问题。提出Depth-Structured Music Recurrence（DSMR），通过分层记忆调度，在有限资源下高效建模完整乐曲。**

- **链接: [https://arxiv.org/pdf/2602.19816](https://arxiv.org/pdf/2602.19816)**

> **作者:** Yungang Yi; Weihua Li; Matthew Kuo; Catherine Shi; Quan Bai
>
> **摘要:** Long-context modeling is essential for symbolic music generation, since motif repetition and developmental variation can span thousands of musical events, yet practical workflows frequently rely on resource-limited hardware. We introduce Depth-Structured Music Recurrence (DSMR), a training-time design that learns from complete compositions end to end by streaming each piece left-to-right with stateful recurrent attention and distributing layer-wise memory horizons under a fixed recurrent-state budget. Our main instantiation, two-scale DSMR, assigns long history windows to lower layers and a uniform short window to the remaining layers. On the MAESTRO piano performance dataset, two-scale DSMR matches a full-memory recurrent reference in perplexity (5.96 vs. 5.98) while using approximately 59% less GPU memory and achieving roughly 36% higher throughput. Variant analyses further show strong layer substitutability under binary-horizon schedules: performance depends primarily on total allocated memory rather than which layers carry it.
>
---
#### [replaced 005] Score-Informed Transformer for Refining MIDI Velocity in Automatic Music Transcription
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于自动音乐转录任务，旨在解决MIDI速度估计不准确的问题。通过引入轻量级的Transformer模块，提升速度估计精度并增强跨数据集的泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.07757](https://arxiv.org/pdf/2508.07757)**

> **作者:** Zhanhong He; Roberto Togneri; David Huang
>
> **备注:** Submitted to SMC2026 Conference
>
> **摘要:** MIDI velocity is crucial for capturing expressive dynamics in human performances. In practical scenarios, a music score with inaccurate velocities may be available alongside the performance audio (e.g., music education and free online archives), enabling the task of score-informed MIDI velocity estimation. In this work, we propose a modular, lightweight score-informed Transformer correction module that refines the velocity estimates of Automatic Music Transcription (AMT) systems. We integrate the proposed module into multiple AMT systems (HPT, HPPNet, and DynEst). Trained exclusively on the MAESTRO training split, our method consistently reduces velocity estimation errors on MAESTRO and improves cross-dataset generalization to SMD and MAPS datasets. Under this training protocol, integrating our score-informed module with HPT (named Score-HPT) establishes a new state-of-the-art performance, outperforms existing score-informed methods and velocity-enabled AMT systems while adding only 1 M parameters.
>
---
#### [replaced 006] WAXAL: A Large-Scale Multilingual African Language Speech Corpus
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文介绍WAXAL，一个用于24种非洲语言的大规模语音数据集，旨在解决低资源语言技术发展不平衡的问题。任务为语音识别与合成，工作包括数据收集、标注及质量控制。**

- **链接: [https://arxiv.org/pdf/2602.02734](https://arxiv.org/pdf/2602.02734)**

> **作者:** Abdoulaye Diack; Perry Nelson; Kwaku Agbesi; Angela Nakalembe; MohamedElfatih MohamedKhair; Vusumuzi Dube; Tavonga Siyavora; Subhashini Venugopalan; Jason Hickey; Uche Okonkwo; Abhishek Bapna; Isaac Wiafe; Raynard Dodzi Helegah; Elikem Doe Atsakpo; Charles Nutrokpor; Fiifi Baffoe Payin Winful; Kafui Kwashie Solaga; Jamal-Deen Abdulai; Akon Obu Ekpezu; Audace Niyonkuru; Samuel Rutunda; Boris Ishimwe; Michael Melese; Engineer Bainomugisha; Joyce Nakatumba-Nabende; Andrew Katumba; Claire Babirye; Jonathan Mukiibi; Vincent Kimani; Samuel Kibacia; James Maina; Fridah Emmah; Ahmed Ibrahim Shekarau; Ibrahim Shehu Adamu; Yusuf Abdullahi; Howard Lakougna; Bob MacDonald; Hadar Shemtov; Aisha Walcott-Bryant; Moustapha Cisse; Avinatan Hassidim; Jeff Dean; Yossi Matias
>
> **备注:** Initial dataset release with added TTS, some more to come
>
> **摘要:** The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 24 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with around 235 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at this https URL under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages.
>
---
#### [replaced 007] Learning Vocal-Tract Area and Radiation with a Physics-Informed Webster Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在通过物理模型准确估计声门和辐射参数，解决传统方法稳定性差的问题。工作包括训练物理感知神经网络并验证其性能。**

- **链接: [https://arxiv.org/pdf/2602.13834](https://arxiv.org/pdf/2602.13834)**

> **作者:** Minhui Lu; Joshua D. Reiss
>
> **备注:** Accepted at IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2026
>
> **摘要:** We present a physics-informed voiced backend renderer for singing-voice synthesis. Given synthetic single-channel audio and a fund-amental--frequency trajectory, we train a time-domain Webster model as a physics-informed neural network to estimate an interpretable vocal-tract area function and an open-end radiation coefficient. Training enforces partial differential equation and boundary consistency; a lightweight DDSP path is used only to stabilize learning, while inference is purely physics-based. On sustained vowels (/a/, /i/, /u/), parameters rendered by an independent finite-difference time-domain Webster solver reproduce spectral envelopes competitively with a compact DDSP baseline and remain stable under changes in discretization, moderate source variations, and about ten percent pitch shifts. The in-graph waveform remains breathier than the reference, motivating periodicity-aware objectives and explicit glottal priors in future work.
>
---
#### [replaced 008] Echo: Towards Advanced Audio Comprehension via Audio-Interleaved Reasoning
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音频理解任务，旨在解决传统模型对复杂音频理解不足的问题。通过引入音频交织推理机制，提升模型的音频感知与分析能力。**

- **链接: [https://arxiv.org/pdf/2602.11909](https://arxiv.org/pdf/2602.11909)**

> **作者:** Daiqing Wu; Xuan Zhang; Dongbao Yang; Jiashu Yao; Longfei Chen; Qingsong Liu; Sicheng Zhao; Can Ma; Yangyang Kang; Yu Zhou
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** The maturation of Large Audio Language Models (LALMs) has raised growing expectations for them to comprehend complex audio much like humans. Current efforts primarily replicate text-based reasoning by contextualizing audio content through a one-time encoding, which introduces a critical information bottleneck. Drawing inspiration from human cognition, we propose audio-interleaved reasoning to break through this bottleneck. It treats audio as an active reasoning component, enabling sustained audio engagement and perception-grounded analysis. To instantiate it, we introduce a two-stage training framework, first teaching LALMs to localize salient audio segments through supervised fine-tuning, and then incentivizing proficient re-listening via reinforcement learning. In parallel, a structured data generation pipeline is developed to produce high-quality training data. Consequently, we present Echo, a LALM capable of dynamically re-listening to audio in demand during reasoning. On audio comprehension benchmarks, Echo achieves overall superiority in both challenging expert-level and general-purpose tasks. Comprehensive analysis further confirms the efficiency and generalizability of audio-interleaved reasoning, establishing it as a promising direction for advancing audio comprehension. Project page: this https URL.
>
---
#### [replaced 009] GDiffuSE: Diffusion-based speech enhancement with noise model guidance
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升噪声环境下的语音质量。通过引入噪声模型引导的扩散模型，提高对未知噪声的适应能力。**

- **链接: [https://arxiv.org/pdf/2510.04157](https://arxiv.org/pdf/2510.04157)**

> **作者:** Efrayim Yanir; David Burshtein; Sharon Gannot
>
> **摘要:** This paper introduces a novel speech enhancement (SE) approach based on a denoising diffusion probabilistic model (DDPM), termed Guided diffusion for speech enhancement (GDiffuSE). In contrast to conventional methods that directly map noisy speech to clean speech, our method employs a lightweight helper model to estimate the noise distribution, which is then incorporated into the diffusion denoising process via a guidance mechanism. This design improves robustness by enabling seamless adaptation to unseen noise types and by leveraging large-scale DDPMs originally trained for speech generation in the context of SE. We evaluate our approach on noisy signals obtained by adding noise samples from the BBC sound effects database to LibriSpeech utterances, showing consistent improvements over state-of-the-art baselines under mismatched noise conditions. Examples are available at our project webpage.
>
---
#### [replaced 010] SounDiT: Geo-Contextual Soundscape-to-Landscape Generation
- **分类: cs.SD; cs.AI; cs.GR; cs.HC; eess.AS**

- **简介: 该论文提出GeoS2L任务，旨在从环境声景生成地理真实的景观图像。构建了两个多模态数据集，提出SounDiT模型和PSS评估框架，解决声景与图像一致性问题。**

- **链接: [https://arxiv.org/pdf/2505.12734](https://arxiv.org/pdf/2505.12734)**

> **作者:** Junbo Wang; Haofeng Tan; Bowen Liao; Albert Jiang; Teng Fei; Qixing Huang; Bing Zhou; Zhengzhong Tu; Shan Ye; Yuhao Kang
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Recent audio-to-image models have shown impressive performance in generating images of specific objects conditioned on their corresponding sounds. However, these models fail to reconstruct real-world landscapes conditioned on environmental soundscapes. To address this gap, we present Geo-contextual Soundscape-to-Landscape (GeoS2L) generation, a novel and practically significant task that aims to synthesize geographically realistic landscape images from environmental soundscapes. To support this task, we construct two large-scale geo-contextual multi-modal datasets, SoundingSVI and SonicUrban, which pair diverse environmental soundscapes with real-world landscape images. We propose SounDiT, a diffusion transformer (DiT)-based model that incorporates environmental soundscapes and geo-contextual scene conditioning to synthesize geographically coherent landscape images. Furthermore, we propose the Place Similarity Score (PSS), a practically-informed geo-contextual evaluation framework to measure consistency between input soundscapes and generated landscape images. Extensive experiments demonstrate that SounDiT outperforms existing baselines in the GeoS2L, while the PSS effectively captures multi-level generation consistency across element, scene,and human perception. Project page: this https URL
>
---
#### [replaced 011] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction
- **分类: cs.AI; cs.SD**

- **简介: 该论文属于对话系统评估任务，旨在检验语音到语音系统的人类相似性。通过首次图灵测试，发现现有系统未达人类水平，并分析其不足之处。**

- **链接: [https://arxiv.org/pdf/2602.24080](https://arxiv.org/pdf/2602.24080)**

> **作者:** Xiang Li; Jiabao Gao; Sipei Lin; Xuan Zhou; Chi Zhang; Bo Cheng; Jiale Han; Benyou Wang
>
> **备注:** Accepted by ICLR 2026 Conference
>
> **摘要:** The pursuit of human-like conversational agents has long been guided by the Turing test. For modern speech-to-speech (S2S) systems, a critical yet unanswered question is whether they can converse like humans. To tackle this, we conduct the first Turing test for S2S systems, collecting 2,968 human judgments on dialogues between 9 state-of-the-art S2S systems and 28 human participants. Our results deliver a clear finding: no existing evaluated S2S system passes the test, revealing a significant gap in human-likeness. To diagnose this failure, we develop a fine-grained taxonomy of 18 human-likeness dimensions and crowd-annotate our collected dialogues accordingly. Our analysis shows that the bottleneck is not semantic understanding but stems from paralinguistic features, emotional expressivity, and conversational persona. Furthermore, we find that off-the-shelf AI models perform unreliably as Turing test judges. In response, we propose an interpretable model that leverages the fine-grained human-likeness ratings and delivers accurate and transparent human-vs-machine discrimination, offering a powerful tool for automatic human-likeness evaluation. Our work establishes the first human-likeness evaluation for S2S systems and moves beyond binary outcomes to enable detailed diagnostic insights, paving the way for human-like improvements in conversational AI systems.
>
---
#### [replaced 012] Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音合成检测任务，旨在识别深度伪造语音中的合成词语。通过微调Whisper模型，利用下一个词预测实现高效检测。**

- **链接: [https://arxiv.org/pdf/2602.22658](https://arxiv.org/pdf/2602.22658)**

> **作者:** Hoan My Tran; Xin Wang; Wanying Ge; Xuechen Liu; Junichi Yamagishi
>
> **备注:** Submitted to Interspeech. To quote: Interspeech no longer enforces an anonymity period for submissions. While uploading a version online is permitted, your official submission to Interspeech must not contain any author-identifying information. ... a note indicating that the paper was submitted for review to (or, eventually, accepted at) Interspeech should be included in the posting
>
> **摘要:** Deepfake speech utterances can be forged by replacing one or more words in a bona fide utterance with semantically different words synthesized with speech-generative models. While a dedicated synthetic word detector could be developed, we developed a cost-effective method that fine-tunes a pre-trained Whisper model to detect synthetic words while transcribing the input utterance via next-token prediction. We further investigate using partially vocoded utterances as the fine-tuning data, thus reducing the cost of data collection. Our experiments demonstrate that, on in-domain test data, the fine-tuned Whisper yields low synthetic-word detection error rates and transcription error rates. On out-of-domain test data with synthetic words produced with unseen speech-generative models, the fine-tuned Whisper remains on par with a dedicated ResNet-based detection model; however, the overall performance degradation calls for strategies to improve its generalization capability.
>
---
#### [replaced 013] Interpreting Multi-Branch Anti-Spoofing Architectures: Correlating Internal Strategy with Empirical Performance
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究音频反欺骗任务，解决多分支网络内部决策机制不透明的问题，通过分析分支协作与竞争模式，量化模型策略并识别性能问题。**

- **链接: [https://arxiv.org/pdf/2602.17711](https://arxiv.org/pdf/2602.17711)**

> **作者:** Ivan Viakhirev; Kirill Borodin; Mikhail Gorodnichev; Grach Mkrtchian
>
> **备注:** Published at MDPI Mathematics (see at this https URL)
>
> **摘要:** Multi-branch deep neural networks like AASIST3 achieve state-of-the-art comparable performance in audio anti-spoofing, yet their internal decision dynamics remain opaque compared to traditional input-level saliency methods. While existing interpretability efforts largely focus on visualizing input artifacts, the way individual architectural branches cooperate or compete under different spoofing attacks is not well characterized. This paper develops a framework for interpreting AASIST3 at the component level. Intermediate activations from fourteen branches and global attention modules are modeled with covariance operators whose leading eigenvalues form low-dimensional spectral signatures. These signatures train a CatBoost meta-classifier to generate TreeSHAP-based branch attributions, which we convert into normalized contribution shares and confidence scores (Cb) to quantify the model's operational strategy. By analyzing 13 spoofing attacks from the ASVspoof 2019 benchmark, we identify four operational archetypes-ranging from Effective Specialization (e.g., A09, Equal Error Rate (EER) 0.04%, C=1.56) to Ineffective Consensus (e.g., A08, EER 3.14%, C=0.33). Crucially, our analysis exposes a Flawed Specialization mode where the model places high confidence in an incorrect branch, leading to severe performance degradation for attacks A17 and A18 (EER 14.26% and 28.63%, respectively). These quantitative findings link internal architectural strategy directly to empirical reliability, highlighting specific structural dependencies that standard performance metrics overlook.
>
---
#### [replaced 014] ERIS: Evolutionary Real-world Interference Scheme for Jailbreaking Audio Large Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频大模型安全任务，旨在解决现实干扰下模型越狱问题。通过遗传算法生成融合恶意指令的自然音频，有效绕过安全机制。**

- **链接: [https://arxiv.org/pdf/2509.11128](https://arxiv.org/pdf/2509.11128)**

> **作者:** Yibo Zhang; Liang Lin
>
> **摘要:** Existing Audio Large Models (ALMs) alignment focuses on clean inputs, neglecting security risks in complex environments. We propose ERIS, a framework transforming real-world interference into a strategically optimized carrier for jailbreaking ALMs. Unlike methods relying on manually designed acoustic patterns, ERIS uses a genetic algorithm to optimize the selection and synthesis of naturalistic signals. Through population initialization, crossover fusion, and probabilistic mutation, it evolves audio fusing malicious instructions with real-world interference. To humans and safety filters, these samples present as natural speech with harmless background noise, yet bypass alignment. Evaluations on multiple ALMs show ERIS significantly outperforms both text and audio jailbreak baselines. Our findings reveal that seemingly innocuous real-world interference can be leveraged to circumvent safety constraints, providing new insights for defensive mechanisms in complex acoustic scenarios.
>
---
#### [replaced 015] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音系统评估任务，旨在解决主观与客观评价不一致的问题。提出TTSDS2指标，并提供数据集和基准以更准确评估合成语音质量。**

- **链接: [https://arxiv.org/pdf/2506.19441](https://arxiv.org/pdf/2506.19441)**

> **作者:** Christoph Minixhofer; Ondrej Klejch; Peter Bell
>
> **摘要:** Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for continually recreating a multilingual test dataset to avoid data leakage; and a continually updated benchmark for TTS in 14 languages.
>
---
#### [replaced 016] Discovering and Steering Interpretable Concepts in Large Generative Music Models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成模型研究任务，旨在发现并解释模型中的可理解概念。通过稀疏自编码器提取特征，揭示音乐结构规律，并用于引导模型生成。**

- **链接: [https://arxiv.org/pdf/2505.18186](https://arxiv.org/pdf/2505.18186)**

> **作者:** Nikhil Singh; Manuel Cherep; Pattie Maes
>
> **备注:** ICLR 2026, 20 pages, 12 figures
>
> **摘要:** The fidelity with which neural networks can now generate content such as music presents a scientific opportunity: these systems appear to have learned implicit theories of such content's structure through statistical learning alone. This offers a potentially new lens on theories of human-generated media. When internal representations align with traditional constructs (e.g. chord progressions in music), they show how such categories can emerge from statistical regularities; when they diverge, they expose limits of existing frameworks and patterns we may have overlooked but that nonetheless carry explanatory power. In this paper, focusing on autoregressive music generators, we introduce a method for discovering interpretable concepts using sparse autoencoders (SAEs), extracting interpretable features from the residual stream of a transformer model. We make this approach scalable and evaluable using automated labeling and validation pipelines. Our results reveal both familiar musical concepts and coherent but uncodified patterns lacking clear counterparts in theory or language. As an extension, we show such concepts can be used to steer model generations. Beyond improving model transparency, our work provides an empirical tool for uncovering organizing principles that have eluded traditional methods of analysis and synthesis.
>
---
#### [replaced 017] Mathematical Foundations of Polyphonic Music Generation via Structural Inductive Bias
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，解决多声部音乐生成中的“缺失中间层”问题，通过结构归纳偏置和数学理论验证，提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.03612](https://arxiv.org/pdf/2601.03612)**

> **作者:** Joonwon Seo
>
> **备注:** 81 pages. A comprehensive monograph detailing the Smart Embedding architecture for polyphonic music generation, including theoretical proofs (Information Theory, Rademacher Complexity, RPTP) and human evaluation results
>
> **摘要:** This monograph introduces a novel approach to polyphonic music generation by addressing the "Missing Middle" problem through structural inductive bias. Focusing on Beethoven's piano sonatas as a case study, we empirically verify the independence of pitch and hand attributes using normalized mutual information (NMI=0.167) and propose the Smart Embedding architecture, achieving a 48.30% reduction in parameters. We provide rigorous mathematical proofs using information theory (negligible loss bounded at 0.153 bits), Rademacher complexity (28.09% tighter generalization bound), and category theory to demonstrate improved stability and generalization. Empirical results show a 9.47% reduction in validation loss, confirmed by SVD analysis and an expert listening study (N=53). This dual theoretical and applied framework bridges gaps in AI music generation, offering verifiable insights for mathematically grounded deep learning.
>
---
#### [replaced 018] Chain of Correction for Full-text Speech Recognition with Large Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别后处理任务，旨在解决全文本纠错中的稳定性、可控性等问题。提出Chain of Correction方法，分段纠正错误，提升纠错效果。**

- **链接: [https://arxiv.org/pdf/2504.01519](https://arxiv.org/pdf/2504.01519)**

> **作者:** Zhiyuan Tang; Dong Wang; Zhikai Zhou; Yong Liu; Shen Huang; Shidong Shang
>
> **备注:** ICASSP 2026
>
> **摘要:** Full-text error correction with Large Language Models (LLMs) for Automatic Speech Recognition (ASR) is attracting increased attention for its ability to address a wide range of error types, such as punctuation restoration and inverse text normalization, across long context. However, challenges remain regarding stability, controllability, completeness, and fluency. To mitigate these issues, this paper proposes the Chain of Correction (CoC), which uses a multi-turn chat format to correct errors segment by segment, guided by pre-recognized text and full-text context for better semantic understanding. Utilizing the open-sourced ChFT dataset, we fine-tune a pre-trained LLM to evaluate CoC's performance. Experiments show that CoC significantly outperforms baseline and benchmark systems in correcting full-text ASR outputs. We also analyze correction thresholds to balance under-correction and over-rephrasing, extrapolate CoC on extra-long ASR outputs, and explore using other types of information to guide error correction.
>
---
#### [replaced 019] TC-BiMamba: Trans-Chunk bidirectionally within BiMamba for unified streaming and non-streaming ASR
- **分类: eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，解决统一流式与非流式识别中动态块大小训练的问题。提出TC-BiMamba模型，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.11546](https://arxiv.org/pdf/2602.11546)**

> **作者:** Qingshun She; Jing Peng; Yangui Fang; Yu Xi; Kai Yu
>
> **备注:** Critical experimental errors found in key results. Resubmitting after correction
>
> **摘要:** This work investigates bidirectional Mamba (BiMamba) for unified streaming and non-streaming automatic speech recognition (ASR). Dynamic chunk size training enables a single model for offline decoding and streaming decoding with various latency settings. In contrast, existing BiMamba based streaming method is limited to fixed chunk size decoding. When dynamic chunk size training is applied, training overhead increases substantially. To tackle this issue, we propose the Trans-Chunk BiMamba (TC-BiMamba) for dynamic chunk size training. Trans-Chunk mechanism trains both bidirectional sequences in an offline style with dynamic chunk size. On the one hand, compared to traditional chunk-wise processing, TC-BiMamba simultaneously achieves 1.3 times training speedup, reduces training memory by 50%, and improves model performance since it can capture bidirectional context. On the other hand, experimental results show that TC-BiMamba outperforms U2++ and matches LC-BiMmaba with smaller model size.
>
---
#### [replaced 020] FOCAL: A Novel Benchmarking Technique for Multi-modal Agents
- **分类: cs.SD**

- **简介: 该论文提出FOCAL框架，用于评估多模态代理的端到端推理和错误传播，解决多模态系统测试与分析问题。**

- **链接: [https://arxiv.org/pdf/2601.07367](https://arxiv.org/pdf/2601.07367)**

> **作者:** Anupam Purwar; Aditya Choudhary
>
> **备注:** We present a framework for evaluation of Multi-modal Agents consisting of Voice-to-voice model components viz. Text to Speech (TTS), Retrieval Augmented Generation (RAG) and Speech-to-text (STT)
>
> **摘要:** With the recent advancements in reasoning capabilities, tool calling using MCP servers and Audio Language Models (ALMs), development and integration of multi-modal agents (with voice and text support) has come to the industry forefront. Cascading pipelines for voice agents still play a central role in the industry owing to their superior reasoning capabilities facilitated by LLMs. Although, cascading pipelines often present error propagation through the pipeline. We propose a framework, FOCAL to benchmark end-to-end reasoning, component-wise error propagation and error analysis for automated as well as human-assisted testing of multi-modal agents (voice to voice + text input). We also share two novel metrics viz. Reasoning and Semantic scores to evaluate efficacy of the agent in having meaningful conversations in voice mode.
>
---
#### [replaced 021] Adapting Speech Foundation Models for Unified Multimodal Speech Recognition with Large Language Models
- **分类: eess.AS**

- **简介: 该论文属于多模态语音识别任务，解决SFMs在多模态场景下适应性不足的问题。通过引入LLMs作为解码器，实现统一的VSR、ASR和AVSR任务优化。**

- **链接: [https://arxiv.org/pdf/2510.22961](https://arxiv.org/pdf/2510.22961)**

> **作者:** Jing-Xuan Zhang; Genshun Wan; Jin Li; Jianqing Gao; Duo Zhao; Zhen-Hua Ling
>
> **备注:** 10 pages, 4 figures, 5 tables
>
> **摘要:** While speech foundation models (SFMs) have demonstrated remarkable performance in audio-only tasks, their adaptation to multimodal scenarios remains underexplored. This work presents UASR-LLM, a novel framework that adapts frozen SFMs to unified visual speech recognition (VSR), automatic speech recognition (ASR), and audio-visual speech recognition (AVSR) by leveraging large language models (LLMs) as text decoders. Visual representations are injected into multiple SFM layers via visual injection modules, enabling multimodal fusion and unified representation learning. The augmented SFMs are connected to decoder-only LLMs through a feed-forward adaptor, where concatenated representations and instruction prompts guide transcription. We propose a two-stage training strategy consisting of visual injection pretraining followed by speech recognition finetuning. The pretraining stage aligns audio, visual, and audio-visual representations within the frozen SFM backbone, while the finetuning stage integrates LLMs for unified optimization across speech recognition tasks. Experimental results demonstrate superior performance over state-of-the-art baselines across VSR, ASR, and AVSR under both clean and noisy conditions. Ablation studies further confirm generalization across various SFMs and LLMs, validating the effectiveness of the proposed training strategy.
>
---
#### [replaced 022] MAPSS: Manifold-based Assessment of Perceptual Source Separation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音源分离评估任务，旨在解决客观评估与主观感知不匹配的问题。提出PS和PM两个指标，通过编码与流形学习分离干扰和失真因素，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2509.09212](https://arxiv.org/pdf/2509.09212)**

> **作者:** Amir Ivry; Samuele Cornell; Shinji Watanabe
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Objective assessment of audio source-separation systems still mismatches subjective human perception, especially when interference from competing talkers and distortion of the target signal interact. We introduce Perceptual Separation (PS) and Perceptual Match (PM), a complementary pair of measures that, by design, isolate these leakage and distortion factors. Our intrusive approach generates a set of fundamental distortions, e.g., clipping, notch filter, and pitch shift from each reference waveform signal in the mixture. Distortions, references, and system outputs from all sources are independently encoded by a pre-trained self-supervised model, then aggregated and embedded with a manifold learning technique called diffusion maps, which aligns Euclidean distances on the manifold with dissimilarities of the encoded waveform representations. On this manifold, PM captures the self-distortion of a source by measuring distances from its output to its reference and associated distortions, while PS captures leakage by also accounting for distances from the output to non-attributed references and distortions. Both measures are differentiable and operate at a resolution as high as 75 frames per second, allowing granular optimization and analysis. We further derive, for both measures, frame-level deterministic error radius and non-asymptotic, high-probability confidence intervals. Experiments on English, Spanish, and music mixtures show that, against 18 widely used measures, the PS and PM are almost always placed first or second in linear and rank correlations with subjective human mean-opinion scores.
>
---
#### [replaced 023] DTT-BSR: GAN-based DTTNet with RoPE Transformer Enhancement for Music Source Restoration
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音乐源分离任务，旨在恢复混音中的原始音轨。通过结合GAN与RoPE Transformer及双路径RNN，提升分离效果与信号质量。**

- **链接: [https://arxiv.org/pdf/2602.19825](https://arxiv.org/pdf/2602.19825)**

> **作者:** Shihong Tan; Haoyu Wang; Youran Ni; Yingzhao Hou; Jiayue Luo; Zipei Hu; Han Dou; Zerui Han; Ningning Pan; Yuzhu Wang; Gongping Huang
>
> **备注:** 3 pages, accepted by ICASSP 2026
>
> **摘要:** Music source restoration (MSR) aims to recover unprocessed stems from mixed and mastered recordings. The challenge lies in both separating overlapping sources and reconstructing signals degraded by production effects such as compression and reverberation. We therefore propose DTT-BSR, a hybrid generative adversarial network (GAN) combining rotary positional embeddings (RoPE) transformer for long-term temporal modeling with dual-path band-split recurrent neural network (RNN) for multi-resolution spectral processing. Our model achieved 3rd place on the objective leaderboard and 4th place on the subjective leaderboard on the ICASSP 2026 MSR Challenge, demonstrating exceptional generation fidelity and semantic alignment with a compact size of 7.1M parameters.
>
---
#### [replaced 024] JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型安全研究任务，旨在解决LALMs面临的越狱攻击问题。工作包括构建JALMBench基准，评估攻击与防御方法，分析模型安全性。**

- **链接: [https://arxiv.org/pdf/2505.17568](https://arxiv.org/pdf/2505.17568)**

> **作者:** Zifan Peng; Yule Liu; Zhen Sun; Mingchen Li; Zeren Luo; Jingyi Zheng; Wenhan Dong; Xinlei He; Xuechao Wang; Yingjie Xue; Shengmin Xu; Xinyi Huang
>
> **摘要:** Large Audio Language Models (LALMs) have made significant progress. While increasingly deployed in real-world applications, LALMs face growing safety risks from jailbreak attacks that bypass safety alignment. However, there remains a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare jailbreak attacks against them. To address this gap, we introduce JALMBench, a comprehensive benchmark that assesses LALM safety against jailbreak attacks, comprising 11,316 text samples and 245,355 audio samples (>1,000 hours). JALMBench supports 12 mainstream LALMs, 8 attack methods (4 text-transferred and 4 audio-originated), and 5 defenses. We conduct in-depth analysis on attack efficiency, topic sensitivity, voice diversity, and model architecture. Additionally, we explore mitigation strategies for the attacks at both the prompt and response levels. Our systematic evaluation reveals that LALMs' safety is strongly influenced by modality and architectural choices: text-based safety alignment can partially transfer to audio inputs, and interleaved audio-text strategies enable more robust cross-modal generalization. Existing general-purpose moderation methods only slightly improve security, highlighting the need for defense methods specifically designed for LALMs. We hope our work can shed light on the design principles for building more robust LALMs.
>
---
