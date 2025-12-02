# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] LLM2Fx-Tools: Tool Calling For Music Post-Production
- **分类: cs.SD**

- **简介: 该论文提出LLM2Fx-Tools，一个用于音乐后期制作的多模态工具调用框架，通过大语言模型实现音频效果链的自动生成与参数估计。针对音乐制作中效果配置复杂、缺乏自动化的问题，研究构建了带思维链标注的指令跟随数据集LP-Fx，利用工具调用与链式推理实现可解释、可控的音频处理，验证了其在风格迁移与生成质量上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01559v1](https://arxiv.org/pdf/2512.01559v1)**

> **作者:** Seungheon Doh; Junghyun Koo; Marco A. Martínez-Ramírez; Woosung Choi; Wei-Hsiang Liao; Qiyu Wu; Juhan Nam; Yuki Mitsufuji
>
> **摘要:** This paper introduces LLM2Fx-Tools, a multimodal tool-calling framework that generates executable sequences of audio effects (Fx-chain) for music post-production. LLM2Fx-Tools uses a large language model (LLM) to understand audio inputs, select audio effects types, determine their order, and estimate parameters, guided by chain-of-thought (CoT) planning. We also present LP-Fx, a new instruction-following dataset with structured CoT annotations and tool calls for audio effects modules. Experiments show that LLM2Fx-Tools can infer an Fx-chain and its parameters from pairs of unprocessed and processed audio, enabled by autoregressive sequence modeling, tool calling, and CoT reasoning. We further validate the system in a style transfer setting, where audio effects information is transferred from a reference source and applied to new content. Finally, LLM-as-a-judge evaluation demonstrates that our approach generates appropriate CoT reasoning and responses for music production queries. To our knowledge, this is the first work to apply LLM-based tool calling to audio effects modules, enabling interpretable and controllable music production.
>
---
#### [new 002] Parallel Delayed Memory Units for Enhanced Temporal Modeling in Biomedical and Bioacoustic Signal Analysis
- **分类: cs.SD; cs.NE**

- **简介: 该论文针对生物信号分析中时序建模的挑战，提出并行延迟记忆单元（PDMU），通过门控延时线与勒让德记忆单元结合，提升短期时序建模效率与记忆能力，增强模型在数据稀缺下的表现，支持并行训练与实时推理，适用于音频与生物信号处理任务。**

- **链接: [https://arxiv.org/pdf/2512.01626v1](https://arxiv.org/pdf/2512.01626v1)**

> **作者:** Pengfei Sun; Wenyu Jiang; Paul Devos; Dick Botteldooren
>
> **备注:** Accepted for publication in IEEE Transactions on Audio, Speech and Language Processing, 2025
>
> **摘要:** Advanced deep learning architectures, particularly recurrent neural networks (RNNs), have been widely applied in audio, bioacoustic, and biomedical signal analysis, especially in data-scarce environments. While gated RNNs remain effective, they can be relatively over-parameterised and less training-efficient in some regimes, while linear RNNs tend to fall short in capturing the complexity inherent in bio-signals. To address these challenges, we propose the Parallel Delayed Memory Unit (PDMU), a {delay-gated state-space module for short-term temporal credit assignment} targeting audio and bioacoustic signals, which enhances short-term temporal state interactions and memory efficiency via a gated delay-line mechanism. Unlike previous Delayed Memory Units (DMU) that embed temporal dynamics into the delay-line architecture, the PDMU further compresses temporal information into vector representations using Legendre Memory Units (LMU). This design serves as a form of causal attention, allowing the model to dynamically adjust its reliance on past states and improve real-time learning performance. Notably, in low-information scenarios, the gating mechanism behaves similarly to skip connections by bypassing state decay and preserving early representations, thereby facilitating long-term memory retention. The PDMU is modular, supporting parallel training and sequential inference, and can be easily integrated into existing linear RNN frameworks. Furthermore, we introduce bidirectional, efficient, and spiking variants of the architecture, each offering additional gains in performance or energy efficiency. Experimental results on diverse audio and biomedical benchmarks demonstrate that the PDMU significantly enhances both memory capacity and overall model performance.
>
---
#### [new 003] MoLT: Mixture of Layer-Wise Tokens for Efficient Audio-Visual Learning
- **分类: cs.SD; cs.CV; cs.MM**

- **简介: 该论文针对音频-视觉学习中的参数与内存效率问题，提出MoLT框架。通过在深层网络中并行提取和融合层间令牌，实现轻量化适配，有效避免早期层误差传播，提升性能。实验表明，MoLT在多个跨模态任务上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00115v1](https://arxiv.org/pdf/2512.00115v1)**

> **作者:** Kyeongha Rho; Hyeongkeun Lee; Jae Won Cho; Joon Son Chung
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** In this paper, we propose Mixture of Layer-Wise Tokens (MoLT), a parameter- and memory-efficient adaptation framework for audio-visual learning. The key idea of MoLT is to replace conventional, computationally heavy sequential adaptation at every transformer layer with a parallel, lightweight scheme that extracts and fuses layer-wise tokens only from the late layers. We adopt two types of adapters to distill modality-specific information and cross-modal interaction into compact latent tokens in a layer-wise manner. A token fusion module then dynamically fuses these layer-wise tokens by taking into account their relative significance. To prevent the redundancy of latent tokens, we apply an orthogonality regularization between latent tokens during training. Through the systematic analysis of the position of adaptation in the pre-trained transformers, we extract latent tokens only from the late layers of the transformers. This strategic adaptation approach avoids error propagation from the volatile early-layer features, thereby maximizing the adaptation performance while maintaining parameter and memory efficiency. Through extensive experiments, we demonstrate that MoLT outperforms existing methods on diverse audio-visual benchmarks, including Audio-Visual Question Answering, Audio-Visual Segmentation, and Audio-Visual Event Localization.
>
---
#### [new 004] Melody or Machine: Detecting Synthetic Music with Dual-Stream Contrastive Learning
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对AI生成音乐的检测任务，解决现有模型在新生成器上泛化能力差的问题。提出大规模基准MoM和双流对比学习架构CLAM，通过分析人声与乐器间的机器不一致特征，提升对合成音乐的检测精度，实现F1 0.925的新纪录。**

- **链接: [https://arxiv.org/pdf/2512.00621v1](https://arxiv.org/pdf/2512.00621v1)**

> **作者:** Arnesh Batra; Dev Sharma; Krish Thukral; Ruhani Bhatia; Naman Batra; Aditya Gautam
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** The rapid evolution of end-to-end AI music generation poses an escalating threat to artistic authenticity and copyright, demanding detection methods that can keep pace. While foundational, existing models like SpecTTTra falter when faced with the diverse and rapidly advancing ecosystem of new generators, exhibiting significant performance drops on out-of-distribution (OOD) content. This generalization failure highlights a critical gap: the need for more challenging benchmarks and more robust detection architectures. To address this, we first introduce Melody or Machine (MoM), a new large-scale benchmark of over 130,000 songs (6,665 hours). MoM is the most diverse dataset to date, built with a mix of open and closed-source models and a curated OOD test set designed specifically to foster the development of truly generalizable detectors. Alongside this benchmark, we introduce CLAM, a novel dual-stream detection architecture. We hypothesize that subtle, machine-induced inconsistencies between vocal and instrumental elements, often imperceptible in a mixed signal, offer a powerful tell-tale sign of synthesis. CLAM is designed to test this hypothesis by employing two distinct pre-trained audio encoders (MERT and Wave2Vec2) to create parallel representations of the audio. These representations are fused by a learnable cross-aggregation module that models their inter-dependencies. The model is trained with a dual-loss objective: a standard binary cross-entropy loss for classification, complemented by a contrastive triplet loss which trains the model to distinguish between coherent and artificially mismatched stream pairings, enhancing its sensitivity to synthetic artifacts without presuming a simple feature alignment. CLAM establishes a new state-of-the-art in synthetic music forensics. It achieves an F1 score of 0.925 on our challenging MoM benchmark.
>
---
#### [new 005] A Low-Complexity Speech Codec Using Parametric Dithering for ASR
- **分类: eess.AS**

- **简介: 该论文针对语音识别（ASR）中语音压缩导致的性能下降问题，提出一种低复杂度参数化抖动编码方法。通过分析最优ASR在有损压缩下的表现，设计适用于1-3比特分辨率的抖动技术，在保持低码率的同时显著提升识别准确率，相对字符错误率降低25%以上。**

- **链接: [https://arxiv.org/pdf/2512.00511v1](https://arxiv.org/pdf/2512.00511v1)**

> **作者:** Ellison Murray; Morriel Kasher; Predrag Spasojevic
>
> **备注:** 10 pages, 8 figures, Accepted 2026 Data Compression Conference
>
> **摘要:** Dithering is a technique commonly used to improve the perceptual quality of lossy data compression. In this work, we analytically and experimentally justify the use of dithering for ASR input compression. We formalize an understanding of optimal ASR performance under lossy input compression and leverage this to propose a parametric dithering technique for a low-complexity speech compression pipeline. The method performs well at 1-bit resolution, showing a 25\% relative CER improvement, while also demonstrating improvements of 32.4\% and 33.5\% at 2- and 3-bit resolution, respectively, with our second dither choice yielding a reduced data rate. The proposed codec is adaptable to meet performance targets or stay within entropy constraints.
>
---
#### [new 006] Explainable Multi-Modal Deep Learning for Automatic Detection of Lung Diseases from Respiratory Audio Signals
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文针对呼吸系统疾病自动检测任务，解决传统听诊主观性强、受环境干扰等问题。提出可解释的多模态深度学习框架，融合卷积-双向LSTM注意力网络与手工声学特征，通过晚期融合提升性能，并利用Grad-CAM等方法生成临床可解释性分析，实现高精度、可信赖的肺部疾病筛查。**

- **链接: [https://arxiv.org/pdf/2512.00563v1](https://arxiv.org/pdf/2512.00563v1)**

> **作者:** S M Asiful Islam Saky; Md Rashidul Islam; Md Saiful Arefin; Shahaba Alam
>
> **摘要:** Respiratory diseases remain major global health challenges, and traditional auscultation is often limited by subjectivity, environmental noise, and inter-clinician variability. This study presents an explainable multimodal deep learning framework for automatic lung-disease detection using respiratory audio signals. The proposed system integrates two complementary representations: a spectral-temporal encoder based on a CNN-BiLSTM Attention architecture, and a handcrafted acoustic-feature encoder capturing physiologically meaningful descriptors such as MFCCs, spectral centroid, spectral bandwidth, and zero-crossing rate. These branches are combined through late-stage fusion to leverage both data-driven learning and domain-informed acoustic cues. The model is trained and evaluated on the Asthma Detection Dataset Version 2 using rigorous preprocessing, including resampling, normalization, noise filtering, data augmentation, and patient-level stratified partitioning. The study achieved strong generalization with 91.21% accuracy, 0.899 macro F1-score, and 0.9866 macro ROC-AUC, outperforming all ablated variants. An ablation study confirms the importance of temporal modeling, attention mechanisms, and multimodal fusion. The framework incorporates Grad-CAM, Integrated Gradients, and SHAP, generating interpretable spectral, temporal, and feature-level explanations aligned with known acoustic biomarkers to build clinical transparency. The findings demonstrate the framework's potential for telemedicine, point-of-care diagnostics, and real-world respiratory screening.
>
---
#### [new 007] Arabic TTS with FastPitch: Reproducible Baselines, Adversarial Training, and Oversmoothing Analysis
- **分类: eess.AS**

- **简介: 该论文聚焦阿拉伯语文本到语音（TTS）任务，针对资源有限与发音复杂导致的输出平滑过度问题。提出基于FastPitch的可复现基线，引入倒谱域度量分析过度平滑，并采用轻量级对抗性谱图损失有效缓解该问题，同时通过合成语音增强多说话人多样性。**

- **链接: [https://arxiv.org/pdf/2512.00937v1](https://arxiv.org/pdf/2512.00937v1)**

> **作者:** Lars Nippert
>
> **摘要:** Arabic text-to-speech (TTS) remains challenging due to limited resources and complex phonological patterns. We present reproducible baselines for Arabic TTS built on the FastPitch architecture and introduce cepstral-domain metrics for analyzing oversmoothing in mel-spectrogram prediction. While traditional Lp reconstruction losses yield smooth but over-averaged outputs, the proposed metrics reveal their temporal and spectral effects throughout training. To address this, we incorporate a lightweight adversarial spectrogram loss, which trains stably and substantially reduces oversmoothing. We further explore multi-speaker Arabic TTS by augmenting FastPitch with synthetic voices generated using XTTSv2, resulting in improved prosodic diversity without loss of stability. The code, pretrained models, and training recipes are publicly available at: https://github.com/nipponjo/tts-arabic-pytorch.
>
---
#### [new 008] STCTS: Generative Semantic Compression for Ultra-Low Bitrate Speech via Explicit Text-Prosody-Timbre Decomposition
- **分类: cs.SD; cs.MM**

- **简介: 该论文提出STCTS，一种面向超低比特率语音压缩的生成式框架。针对带宽受限场景下传统编码器效率低、现有语义方法丢失韵律与音色的问题，通过显式分解文本、韵律、音色并分别压缩，实现约80 bps的高效传输，同时保持高感知质量，支持隐私保护与边缘部署。**

- **链接: [https://arxiv.org/pdf/2512.00451v1](https://arxiv.org/pdf/2512.00451v1)**

> **作者:** Siyu Wang; Haitao Li
>
> **备注:** The complete source code and online speech reconstruction demo is publicly available at https://github.com/dywsy21/STCTS
>
> **摘要:** Voice communication in bandwidth-constrained environments--maritime, satellite, and tactical networks--remains prohibitively expensive. Traditional codecs struggle below 1 kbps, while existing semantic approaches (STT-TTS) sacrifice prosody and speaker identity. We present STCTS, a generative semantic compression framework enabling natural voice communication at approximately 80 bps. STCTS explicitly decomposes speech into linguistic content, prosodic expression, and speaker timbre, applying tailored compression: context-aware text encoding (approximately 70 bps), sparse prosody transmission via TTS interpolation (less than 14 bps at 0.1-1 Hz), and amortized speaker embedding. Evaluations on LibriSpeech demonstrate a 75x bitrate reduction versus Opus (6 kbps) and 12x versus EnCodec (1 kbps), while maintaining perceptual quality (NISQA MOS greater than 4.26). We also discover a bimodal quality distribution with prosody sampling rate: sparse and dense updates both achieve high quality, while mid-range rates degrade due to perceptual discontinuities--guiding optimal configuration design. Beyond efficiency, our modular architecture supports privacy-preserving encryption, human-interpretable transmission, and flexible deployment on edge devices, offering a robust solution for ultra-low bandwidth scenarios.
>
---
#### [new 009] Art2Music: Generating Music for Art Images with Multi-modal Feeling Alignment
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; cs.MM**

- **简介: 该论文提出Art2Music，解决艺术图像与音乐间的感知自然、情感对齐的跨模态生成问题。构建ArtiCaps数据集，设计轻量级框架，通过融合图像与文本特征生成高质量音乐，显著提升音色保真度与情感一致性，适用于互动艺术与个性化声景。**

- **链接: [https://arxiv.org/pdf/2512.00120v1](https://arxiv.org/pdf/2512.00120v1)**

> **作者:** Jiaying Hong; Ting Zhu; Thanet Markchom; Huizhi Liang
>
> **摘要:** With the rise of AI-generated content (AIGC), generating perceptually natural and feeling-aligned music from multimodal inputs has become a central challenge. Existing approaches often rely on explicit emotion labels that require costly annotation, underscoring the need for more flexible feeling-aligned methods. To support multimodal music generation, we construct ArtiCaps, a pseudo feeling-aligned image-music-text dataset created by semantically matching descriptions from ArtEmis and MusicCaps. We further propose Art2Music, a lightweight cross-modal framework that synthesizes music from artistic images and user comments. In the first stage, images and text are encoded with OpenCLIP and fused using a gated residual module; the fused representation is decoded by a bidirectional LSTM into Mel-spectrograms with a frequency-weighted L1 loss to enhance high-frequency fidelity. In the second stage, a fine-tuned HiFi-GAN vocoder reconstructs high-quality audio waveforms. Experiments on ArtiCaps show clear improvements in Mel-Cepstral Distortion, Frechet Audio Distance, Log-Spectral Distance, and cosine similarity. A small LLM-based rating study further verifies consistent cross-modal feeling alignment and offers interpretable explanations of matches and mismatches across modalities. These results demonstrate improved perceptual naturalness, spectral fidelity, and semantic consistency. Art2Music also maintains robust performance with only 50k training samples, providing a scalable solution for feeling-aligned creative audio generation in interactive art, personalized soundscapes, and digital art exhibitions.
>
---
#### [new 010] Q2D2: A Geometry-Aware Audio Codec Leveraging Two-Dimensional Quantization
- **分类: cs.SD; cs.AI; cs.IT; cs.LG; eess.SP**

- **简介: 该论文针对神经音频编码器中量化方法限制隐空间几何结构、影响表示效率的问题，提出二维量化Q2D2。通过将特征对投影至规则2D网格进行量化，构建隐式码本，提升代码本利用率与压缩效率，实现低比特率下优异的重建质量。**

- **链接: [https://arxiv.org/pdf/2512.01537v1](https://arxiv.org/pdf/2512.01537v1)**

> **作者:** Tal Shuster; Eliya Nachmani
>
> **摘要:** Recent neural audio codecs have achieved impressive reconstruction quality, typically relying on quantization methods such as Residual Vector Quantization (RVQ), Vector Quantization (VQ) and Finite Scalar Quantization (FSQ). However, these quantization techniques limit the geometric structure of the latent space, make it harder to capture correlations between features leading to inefficiency in representation learning, codebook utilization and token rate. In this paper we introduce Two Dimensional Quantization (Q2D2), a quantization scheme in which feature pairs are projected onto structured 2D grids such as hexagonal, rhombic, or rectangular tiling and quantized to the nearest grid values, yielding an implicit codebook defined by the product of grid levels, with codebook sizes comparable to conventional methods. Despite its simple geometric formulation, Q2D2 improves audio compression efficiency, with low token rates and high codebook utilization while maintaining state of the art reconstruction quality. Specifically, Q2D2 achieves competitive to superior performance in various objective and subjective reconstruction metrics, across extensive experiments in speech domain compared to state of the art models. Comprehensive ablation studies further confirm the effectiveness of our design choices.
>
---
#### [new 011] Identifiability Conditions for Acoustic Feedback Cancellation with the Two-Channel Adaptive Feedback Canceller Algorithm
- **分类: eess.AS**

- **简介: 该论文研究声学反馈消除中的可辨识性问题，针对两通道自适应反馈抵消（2ch-AFC）算法，提出基于逆矩阵可逆性的新可辨识条件，证明当前向路径滤波器阶数高于AR模型阶数时即可实现正确识别，并以相关矩阵条件数作为可辨识性监测指标。**

- **链接: [https://arxiv.org/pdf/2512.01466v1](https://arxiv.org/pdf/2512.01466v1)**

> **作者:** Arnout Roebben; Toon van Waterschoot; Jan Wouters; Marc Moonen
>
> **备注:** Accepted for publication in IEEE Open Journal of Signal Processing (OJSP)
>
> **摘要:** In audio signal processing applications with a microphone and a loudspeaker within the same acoustic environment, the loudspeaker signals can feed back into the microphone, thereby creating a closed-loop system that potentially leads to system instability. To remove this acoustic coupling, prediction error method (PEM) feedback cancellation algorithms aim to identify the feedback path between the loudspeaker and the microphone by assuming that the input signal can be modelled by means of an autoregressive (AR) model. It has previously been shown that this PEM framework and resulting algorithms can identify the feedback path correctly in cases where the forward path from microphone to loudspeaker is sufficiently time-varying or non-linear, or when the forward path delay equals or exceeds the order of the AR model. In this paper, it is shown that this delay-based condition can be generalised for one particular PEM-based algorithm, the so-called two-channel adaptive feedback canceller (2ch-AFC), to an invertibility-based condition, for which it is shown that identifiability can be achieved when the order of the forward path feedforward filter exceeds the order of the AR model. Additionally, the condition number of inversion of the correlation matrix as used in the 2ch-AFC algorithm can serve as a measure for monitoring the identifiability.
>
---
#### [new 012] Beyond Performance: Probing Representation Dynamics In Speech Enhancement Models
- **分类: eess.AS**

- **简介: 该论文研究语音增强模型在不同信噪比下的表征动态。针对噪声影响下模型内部表示不稳定的问题，利用CKA与扩散距离分析MUSE模型各层激活，发现编码器鲁棒性强，解码器敏感，且层间响应差异显著，揭示了深度依赖的抗噪特性，提出应设计信噪比感知的条件调制与优化策略。**

- **链接: [https://arxiv.org/pdf/2512.00482v1](https://arxiv.org/pdf/2512.00482v1)**

> **作者:** Yair Amar; Amir Ivry; Israel Cohen
>
> **摘要:** We probe internal representations of a speech enhancement (SE) model across noise conditions. Using MUSE, a transformer-convolutional model trained on VoiceBank DEMAND, we analyze activations in encoder, latent, decoder, and refinement blocks while sweeping input signal-to-noise-ratios (SNRs) from -10 to 30 dB. We use Centered Kernel Alignment (CKA) to measure point-wise representation similarity and diffusion distance to capture distributional shifts across SNRs. Results show that the encoder CKA between noisy and clean inputs remains stable and latent and decoder CKA drop sharply as SNR decreases. Linear fits of CKA versus SNR reveal a depth-dependent robustness-sensitivity trade-off. The diffusion distance varies incrementally with SNR within each layer but differs strongly across layers, especially at low SNRs. Together, these findings indicate that noise levels differentially activate model regions and induce distinct inter-layer dynamics, motivating SNR-aware conditioning and refinement strategies for SE.
>
---
#### [new 013] Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出首个音频-视觉世界模型（AVWM）框架，解决多模态环境建模中视听同步与任务奖励预测问题。构建AVW-4k数据集，设计AV-CDiT模型实现高保真视听动态预测，显著提升连续导航任务性能。**

- **链接: [https://arxiv.org/pdf/2512.00883v1](https://arxiv.org/pdf/2512.00883v1)**

> **作者:** Jiahua Wang; Shannan Yan; Leqi Zheng; Jialong Wu; Yaoxin Mao
>
> **摘要:** World models simulate environmental dynamics to enable agents to plan and reason about future states. While existing approaches have primarily focused on visual observations, real-world perception inherently involves multiple sensory modalities. Audio provides crucial spatial and temporal cues such as sound source localization and acoustic scene properties, yet its integration into world models remains largely unexplored. No prior work has formally defined what constitutes an audio-visual world model or how to jointly capture binaural spatial audio and visual dynamics under precise action control with task reward prediction. This work presents the first formal framework for Audio-Visual World Models (AVWM), formulating multimodal environment simulation as a partially observable Markov decision process with synchronized audio-visual observations, fine-grained actions, and task rewards. To address the lack of suitable training data, we construct AVW-4k, a dataset comprising 30 hours of binaural audio-visual trajectories with action annotations and reward signals across 76 indoor environments. We propose AV-CDiT, an Audio-Visual Conditional Diffusion Transformer with a novel modality expert architecture that balances visual and auditory learning, optimized through a three-stage training strategy for effective multimodal integration. Extensive experiments demonstrate that AV-CDiT achieves high-fidelity multimodal prediction across visual and auditory modalities with reward. Furthermore, we validate its practical utility in continuous audio-visual navigation tasks, where AVWM significantly enhances the agent's performance.
>
---
#### [new 014] Masked Symbol Modeling for Demodulation of Oversampled Baseband Communication Signals in Impulsive Noise-Dominated Channels
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文提出掩码符号建模（MSM）框架，用于在脉冲噪声主导的信道中解调过采样基带信号。针对传统方法忽视符号间干扰（ISC）上下文的问题，将ISC视为确定性上下文信息，利用Transformer通过掩码预测学习基带波形的隐式语法，实现对受损符号的上下文推断，推动物理层向语义理解演进。**

- **链接: [https://arxiv.org/pdf/2512.01428v1](https://arxiv.org/pdf/2512.01428v1)**

> **作者:** Oguz Bedir; Nurullah Sevim; Mostafa Ibrahim; Sabit Ekin
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG), non-archival
>
> **摘要:** Recent breakthroughs in natural language processing show that attention mechanism in Transformer networks, trained via masked-token prediction, enables models to capture the semantic context of the tokens and internalize the grammar of language. While the application of Transformers to communication systems is a burgeoning field, the notion of context within physical waveforms remains under-explored. This paper addresses that gap by re-examining inter-symbol contribution (ISC) caused by pulse-shaping overlap. Rather than treating ISC as a nuisance, we view it as a deterministic source of contextual information embedded in oversampled complex baseband signals. We propose Masked Symbol Modeling (MSM), a framework for the physical (PHY) layer inspired by Bidirectional Encoder Representations from Transformers methodology. In MSM, a subset of symbol aligned samples is randomly masked, and a Transformer predicts the missing symbol identifiers using the surrounding "in-between" samples. Through this objective, the model learns the latent syntax of complex baseband waveforms. We illustrate MSM's potential by applying it to the task of demodulating signals corrupted by impulsive noise, where the model infers corrupted segments by leveraging the learned context. Our results suggest a path toward receivers that interpret, rather than merely detect communication signals, opening new avenues for context-aware PHY layer design.
>
---
#### [new 015] MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification
- **分类: cs.CL; cs.LG; cs.NE; cs.SD**

- **简介: 该论文针对脑磁图（MEG）信号的语音检测与音素分类任务，提出基于Conformer的轻量级解码器。通过适配306通道原始MEG数据，引入专用增强、加权策略与归一化方法，有效缓解分布偏移与样本不均衡问题，在竞赛中分别取得88.9%和65.8%的F1-macro成绩，显著优于基线。**

- **链接: [https://arxiv.org/pdf/2512.01443v1](https://arxiv.org/pdf/2512.01443v1)**

> **作者:** Xabier de Zuazo; Ibon Saratxaga; Eva Navas
>
> **备注:** 10 pages, 5 figures, 4 tables, LibriBrain Workshop, NeurIPS 2025
>
> **摘要:** We present Conformer-based decoders for the LibriBrain 2025 PNPL competition, targeting two foundational MEG tasks: Speech Detection and Phoneme Classification. Our approach adapts a compact Conformer to raw 306-channel MEG signals, with a lightweight convolutional projection layer and task-specific heads. For Speech Detection, a MEG-oriented SpecAugment provided a first exploration of MEG-specific augmentation. For Phoneme Classification, we used inverse-square-root class weighting and a dynamic grouping loader to handle 100-sample averaged examples. In addition, a simple instance-level normalization proved critical to mitigate distribution shifts on the holdout split. Using the official Standard track splits and F1-macro for model selection, our best systems achieved 88.9% (Speech) and 65.8% (Phoneme) on the leaderboard, surpassing the competition baselines and ranking within the top-10 in both tasks. For further implementation details, the technical documentation, source code, and checkpoints are available at https://github.com/neural2speech/libribrain-experiments.
>
---
#### [new 016] ZO-ASR: Zeroth-Order Fine-Tuning of Speech Foundation Models without Back-Propagation
- **分类: cs.MM; cs.SD**

- **简介: 该论文针对自动语音识别（ASR）中微调预训练模型时GPU内存消耗大的问题，提出无需反向传播的ZO-ASR方法。通过前向传播估计梯度，仅需推理内存即可完成微调，在监督与无监督场景下均有效，尤其在资源受限环境下具有实用价值。**

- **链接: [https://arxiv.org/pdf/2512.01267v1](https://arxiv.org/pdf/2512.01267v1)**

> **作者:** Yuezhang Peng; Yuxin Liu; Yao Li; Sheng Wang; Fei Wen; Xie Chen
>
> **备注:** 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)
>
> **摘要:** Fine-tuning pre-trained speech foundation models for Automatic Speech Recognition (ASR) is prevalent, yet constrained by substantial GPU memory requirements. We introduce ZO-ASR, a memory-efficient Zeroth-Order (ZO) method that avoids Back-Propagation (BP) and activation memory by estimating gradients via forward passes. When combined with SGD optimizer, ZO-ASR-SGD fine-tunes ASR models using only inference memory. Our evaluation spans supervised and unsupervised tasks. For Supervised Domain Adaptation on Whisper-Large-V3, ZO-ASR's multiple query mechanism enhances robustness and achieves up to an 18.9\% relative Word Error Rate reduction over zero-shot baselines, outperforming existing ZO methods. For unsupervised Test-Time Adaptation on Wav2Vec2-Base, ZO-ASR exhibits moderately lower performance compared to first-order optimizer Adam. Our BP-free approach provides a viable solution for fine-tuning ASR models in computationally resource-constrained or gradient-inaccessible scenarios.
>
---
## 更新

#### [replaced 001] Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis
- **分类: eess.AS**

- **简介: 该论文针对零样本语音合成中声谱图冗余导致对齐效率低的问题，提出Semantic-VAE框架。通过语义对齐正则化，在高维潜在空间中捕捉语义结构，缓解重建与生成间的权衡，提升语音合成质量与训练效率。**

- **链接: [https://arxiv.org/pdf/2509.22167v2](https://arxiv.org/pdf/2509.22167v2)**

> **作者:** Zhikang Niu; Shujie Hu; Jeongsoo Choi; Yushen Chen; Peining Chen; Pengcheng Zhu; Yunting Yang; Bowen Zhang; Jian Zhao; Chunhui Wang; Xie Chen
>
> **备注:** Submitted to ICASSP2026
>
> **摘要:** While mel-spectrograms have been widely utilized as intermediate representations in zero-shot text-to-speech (TTS), their inherent redundancy leads to inefficiency in learning text-speech alignment. Compact VAE-based latent representations have recently emerged as a stronger alternative, but they also face a fundamental optimization dilemma: higher-dimensional latent spaces improve reconstruction quality and speaker similarity, but degrade intelligibility, while lower-dimensional spaces improve intelligibility at the expense of reconstruction fidelity. To overcome this dilemma, we propose Semantic-VAE, a novel VAE framework that utilizes semantic alignment regularization in the latent space. This design alleviates the reconstruction-generation trade-off by capturing semantic structure in high-dimensional latent representations. Extensive experiments demonstrate that Semantic-VAE significantly improves synthesis quality and training efficiency. When integrated into F5-TTS, our method achieves 2.10% WER and 0.64 speaker similarity on LibriSpeech-PC, outperforming mel-based systems (2.23%, 0.60) and vanilla acoustic VAE baselines (2.65%, 0.59). We also release the code and models to facilitate further research.
>
---
#### [replaced 002] Comparative Evaluation of Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对日语角色语音合成任务，比较VITS与Style-BERT-VITS2 JP Extra模型。针对日语特有的高低音敏感与风格多样性问题，基于三个角色数据集，评估自然度、可懂度与说话人一致性。结果表明，SBV2JE在自然度和可懂度上接近真人，且支持音高控制，适用于语言学习与角色对话生成。**

- **链接: [https://arxiv.org/pdf/2505.17320v2](https://arxiv.org/pdf/2505.17320v2)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **备注:** Accepted to IEEE UEMCON 2025
>
> **摘要:** Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper empirically evaluates two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands.
>
---
#### [replaced 003] Multilingual DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对多语言语音识别中低资源语言性能不足的问题，提出DistilWhisper方法。通过引入语言特定专家与知识蒸馏，对Whisper-small进行轻量级微调，有效提升多语言ASR性能，同时保持多任务、多语言特性，参数开销极小。**

- **链接: [https://arxiv.org/pdf/2311.01070v4](https://arxiv.org/pdf/2311.01070v4)**

> **作者:** Thomas Palmeira Ferraz; Marcely Zanon Boito; Caroline Brun; Vassilina Nikoulina
>
> **备注:** Accepted and Presented at IEEE ICASSP 2024 (this is extended version)
>
> **摘要:** Whisper is a multitask and multilingual speech model covering 99 languages. It yields commendable automatic speech recognition (ASR) results in a subset of its covered languages, but the model still underperforms on a non-negligible number of under-represented languages, a problem exacerbated in smaller model versions. In this work, we propose DistilWhisper, an approach able to bridge the performance gap in ASR for these languages while retaining the advantages of multitask and multilingual capabilities. Our approach involves two key strategies: lightweight modular ASR fine-tuning of whisper-small using language-specific experts, and knowledge distillation from whisper-large-v2. This dual approach allows us to effectively boost ASR performance while keeping the robustness inherited from the multitask and multilingual pre-training. Results demonstrate that our approach is more effective than standard fine-tuning or LoRA adapters, boosting performance in the targeted languages for both in- and out-of-domain test sets, while introducing only a negligible parameter overhead at inference.
>
---
#### [replaced 004] RIFT: Entropy-Optimised Fractional Wavelet Constellations for Ideal Time-Frequency Estimation
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出RIFT方法，用于高分辨率时频分析。针对非平稳信号的时频表示中交叉项干扰与分辨率不足问题，通过优化分数阶小波变换星座，结合熵稀疏度与正则化反卷积，实现无交叉项、高精度的时频重构，并可提取信号轨迹用于语音音乐分析。**

- **链接: [https://arxiv.org/pdf/2501.15764v2](https://arxiv.org/pdf/2501.15764v2)**

> **作者:** James M. Cozens; Simon J. Godsill
>
> **摘要:** We introduce a new method for estimating the Ideal Time-Frequency Representation (ITFR) of complex nonstationary signals. The Reconstructive Ideal Fractional Transform (RIFT) computes a constellation of Continuous Fractional Wavelet Transforms (CFWTs) aligned to different local time-frequency curvatures. This constellation is combined into a single optimised time-frequency energy representation via a localised entropy-based sparsity measure, designed to resolve auto-terms and attenuate cross-terms. Finally, a positivity-constrained Lucy-Richardson deconvolution with total-variation regularisation is applied to estimate the ITFR, achieving auto-term resolution comparable to that of the Wigner-Ville Distribution (WVD), yielding the high-resolution RIFT representation. The required Cohen's class convolutional kernels are fully derived in the paper for the chosen CFWT constellations. Additionally, the optimisation yields an Instantaneous Phase Direction (IPD) field, which allows the localised curvature in speech or music extracts to be visualised and utilised within a Kalman tracking scheme, enabling the extraction of signal component trajectories and the construction of the Spline-RIFT variant. Evaluation on synthetic and real-world signals demonstrates the algorithm's ability to effectively suppress cross-terms and achieve superior time-frequency precision relative to competing methods. This advance holds significant potential for a wide range of applications requiring high-resolution cross-term-free time-frequency analysis.
>
---
#### [replaced 005] Discrete Optimal Transport and Voice Conversion
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文研究语音转换任务，旨在通过向量对齐实现跨说话人音频特征的精准映射。提出使用离散最优传输方法进行音频嵌入对齐，提升转换质量；同时发现该方法在生成音频后处理中可能导致合成语音被误判为真实语音。**

- **链接: [https://arxiv.org/pdf/2505.04382v3](https://arxiv.org/pdf/2505.04382v3)**

> **作者:** Anton Selitskiy; Maitreya Kocharekar
>
> **备注:** 4 pages, 6 figures, 1 table
>
> **摘要:** In this work, we address the voice conversion (VC) task using a vector-based interface. To align audio embeddings between speakers, we employ discrete optimal transport mapping. Our evaluation results demonstrate the high quality and effectiveness of this method. Additionally, we show that applying discrete optimal transport as a post-processing step in audio generation can lead to the incorrect classification of synthetic audio as real.
>
---
#### [replaced 006] Probabilistic Fusion and Calibration of Neural Speaker Diarization Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对端到端神经说话人聚类（EEND）系统中置信度评分不可靠的问题，提出概率级融合与校准框架。通过连续概率输出实现更优的模型融合与校准，提升诊断错误率（DER）性能，并提供可靠的置信度估计，优于传统DOVER-Lap方法。**

- **链接: [https://arxiv.org/pdf/2511.22696v2](https://arxiv.org/pdf/2511.22696v2)**

> **作者:** Juan Ignacio Alvarez-Trejos; Sergio A. Balanya; Daniel Ramos; Alicia Lozano-Diez
>
> **摘要:** End-to-End Neural Diarization (EEND) systems produce frame-level probabilistic speaker activity estimates, yet since evaluation focuses primarily on Diarization Error Rate (DER), the reliability and calibration of these confidence scores have been largely neglected. When fusing multiple diarization systems, DOVER-Lap remains the only established approach, operating at the segment level with hard decisions. We propose working with continuous probability outputs, which enables more sophisticated fusion and calibration techniques that can leverage model uncertainty and complementary strengths across different architectures. This paper presents the first comprehensive framework for calibrating and fusing EEND models at the probability level. We investigate two output formulations (multilabel and powerset representations) and their impact on calibration and fusion effectiveness. Through extensive experiments on the CallHome two-speaker benchmark, we demonstrate that proper calibration provides substantial improvements even for individual models (up to 19% relative DER reduction), in some cases mitigating the absence of domain adaptation. We reveal that joint calibration in powerset space consistently outperforms independent per-speaker calibration, that fusion substantially improves over individual models, and that the Fuse-then-Calibrate ordering generally outperforms both calibrating before fusion and uncalibrated fusion while requiring calibration of only a single combined model. Our best configuration outperforms DOVER-Lap in terms of DER while providing reliable confidence estimates essential for downstream applications. This work proposes best practices for probability-level fusion of EEND systems and demonstrates the advantages of leveraging soft outputs over hard decisions.
>
---
#### [replaced 007] The Extended SONICOM HRTF Dataset and Spatial Audio Metrics Toolbox
- **分类: eess.AS; cs.SD**

- **简介: 该论文面向个性化空间音频任务，解决HRTF数据稀缺与分析困难问题。构建了300人扩展版SONICOM HRTF数据集，包含实测与合成数据及优化3D扫描，支持算法迭代与形态研究；并推出SAM工具箱，实现高效HRTF分析与可视化，推动个性化空间音频发展。**

- **链接: [https://arxiv.org/pdf/2507.05053v2](https://arxiv.org/pdf/2507.05053v2)**

> **作者:** Katarina C. Poole; Julie Meyer; Vincent Martin; Rapolas Daugintis; Nils Marggraf-Turley; Jack Webb; Ludovic Pirard; Nicola La Magna; Oliver Turvey; Lorenzo Picinali
>
> **备注:** For dataset: https://www.axdesign.co.uk/tools-and-devices/sonicom-hrtf-dataset. For toolbox: https://github.com/Katarina-Poole/Spatial-Audio-Metrics. Conference: Forum Acusticum 2025 V2: Fixed clickable links in pdf
>
> **摘要:** Headphone-based spatial audio uses head-related transfer functions (HRTFs) to simulate real-world acoustic environments. HRTFs are unique to everyone, due to personal morphology, shaping how sound waves interact with the body before reaching the eardrums. Here we present the extended SONICOM HRTF dataset which expands on the previous version released in 2023. The total number of measured subjects has now been increased to 300, with demographic information for a subset of the participants, providing context for the dataset's population and relevance. The dataset incorporates synthesised HRTFs for 200 of the 300 subjects, generated using Mesh2HRTF, alongside pre-processed 3D scans of the head and ears, optimised for HRTF synthesis. This rich dataset facilitates rapid and iterative optimisation of HRTF synthesis algorithms, allowing the automatic generation of large data. The optimised scans enable seamless morphological modifications, providing insights into how anatomical changes impact HRTFs, and the larger sample size enhances the effectiveness of machine learning approaches. To support analysis, we also introduce the Spatial Audio Metrics (SAM) Toolbox, a Python package designed for efficient analysis and visualisation of HRTF data, offering customisable tools for advanced research. Together, the extended dataset and toolbox offer a comprehensive resource for advancing personalised spatial audio research and development.
>
---
#### [replaced 008] SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models
- **分类: cs.CL; cs.AI; cs.SC; cs.SD; eess.AS**

- **简介: 该论文提出SpeechIQ，一种基于认知层次的语音理解评估框架，针对大语言模型的语音理解能力。它超越传统词错误率，从记忆、理解、应用三层次评估，解决现有评测指标单一、难以发现幻觉与标注错误的问题，统一比较端到端与级联模型，推动多模态训练研究。**

- **链接: [https://arxiv.org/pdf/2507.19361v2](https://arxiv.org/pdf/2507.19361v2)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Yahan Yu; Jinchuan Tian; Sheng Li; Ke Hu; Zhehuai Chen; Shinji Watanabe; Fei Cheng; Chenhui Chu; Sadao Kurohashi
>
> **备注:** ACL 2025 main. Our Speech-IQ leaderboard is hosted at huggingface.co/spaces/nvidia/Speech-IQ-leaderboard. Speech-IQ Calculator: https://github.com/YukinoWan/SpeechIQ
>
> **摘要:** We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training. Our code and data will be open source to encourage future studies.
>
---
#### [replaced 009] IMSE: Efficient U-Net-based Speech Enhancement using Inception Depthwise Convolution and Amplitude-Aware Linear Attention
- **分类: cs.SD; cs.AI; cs.CV**

- **简介: 该论文针对资源受限设备上的语音增强任务，提出IMSE模型。针对现有方法MUSE的效率瓶颈，创新性地引入幅度感知线性注意力（MALA）和因式分解深度卷积（IDConv），在减少16.8%参数量的同时保持优异语音质量，实现了轻量化与高性能的平衡。**

- **链接: [https://arxiv.org/pdf/2511.14515v2](https://arxiv.org/pdf/2511.14515v2)**

> **作者:** Xinxin Tang; Bin Qin; Yufang Li
>
> **摘要:** Achieving a balance between lightweight design and high performance remains a significant challenge for speech enhancement (SE) tasks on resource-constrained devices. Existing state-of-the-art methods, such as MUSE, have established a strong baseline with only 0.51M parameters by introducing a Multi-path Enhanced Taylor (MET) transformer and Deformable Embedding (DE). However, an in-depth analysis reveals that MUSE still suffers from efficiency bottlenecks: the MET module relies on a complex "approximate-compensate" mechanism to mitigate the limitations of Taylor-expansion-based attention, while the offset calculation for deformable embedding introduces additional computational burden. This paper proposes IMSE, a systematically optimized and ultra-lightweight network. We introduce two core innovations: 1) Replacing the MET module with Amplitude-Aware Linear Attention (MALA). MALA fundamentally rectifies the "amplitude-ignoring" problem in linear attention by explicitly preserving the norm information of query vectors in the attention calculation, achieving efficient global modeling without an auxiliary compensation branch. 2) Replacing the DE module with Inception Depthwise Convolution (IDConv). IDConv borrows the Inception concept, decomposing large-kernel operations into efficient parallel branches (square, horizontal, and vertical strips), thereby capturing spectrogram features with extremely low parameter redundancy. Extensive experiments on the VoiceBank+DEMAND dataset demonstrate that, compared to the MUSE baseline, IMSE significantly reduces the parameter count by 16.8\% (from 0.513M to 0.427M) while achieving competitive performance comparable to the state-of-the-art on the PESQ metric (3.373). This study sets a new benchmark for the trade-off between model size and speech quality in ultra-lightweight speech enhancement.
>
---
#### [replaced 010] AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文针对大音频语言模型（LALMs）指令敏感问题，提出AHAMask方法。通过掩码解码器中部分注意力头，无需指令即可触发特定音频任务功能。实验表明，该方法性能优于或相当指令，揭示了模型内部存在可激活的“功能路径”。**

- **链接: [https://arxiv.org/pdf/2509.01787v3](https://arxiv.org/pdf/2509.01787v3)**

> **作者:** Yiwei Guo; Bohan Li; Hankun Wang; Zhihan Li; Shuai Wang; Xie Chen; Kai Yu
>
> **备注:** 15 pages, 10 tables, 6 figures. This is the camera ready version for AAAI 2026, plus an appendix for supplementary experimental details and results
>
> **摘要:** Although current large audio language models (LALMs) extend text large language models (LLMs) with generic acoustic understanding abilities, they usually suffer from prompt sensitivity, where different instructions of the same intention can yield drastically different outcomes. In this work, we propose AHAMask, where we simply mask some of the attention heads in the decoder-only LLM backbone of LALMs, to trigger specific acoustic task functionalities without instructions. These masks are efficiently obtained by training on an LALM, with the number of trainable parameters equal to the attention head count in its LLM backbone. We show by experiments that applying such selective attention head masks achieves comparable or even better performance than using instructions, either on single or composite tasks. Besides achieving reliable acoustic task specification for LALMs, this also reveals that LALMs exhibit certain "functional pathways" in their attention heads.
>
---
#### [replaced 011] Safeguarding Privacy in Edge Speech Understanding with Tiny Foundation Models
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文针对边缘设备语音识别中的隐私泄露问题，提出SpeechShield框架。利用小型语音基础模型在本地实时检测并掩蔽敏感实体，保护隐私同时保持高转录准确率。实验表明其内存占用小、速度快、效率高，显著优于现有方案。**

- **链接: [https://arxiv.org/pdf/2502.01649v2](https://arxiv.org/pdf/2502.01649v2)**

> **作者:** Afsara Benazir; Felix Xiaozhu Lin
>
> **摘要:** Robust speech recognition systems rely on cloud service providers for inference. It needs to ensure that an untrustworthy provider cannot deduce the sensitive content in speech. Sanitization can be done on speech content keeping in mind that it has to avoid compromising transcription accuracy. Realizing the under utilized capabilities of tiny speech foundation models (FMs), for the first time, we propose a novel use: enhancing speech privacy on resource-constrained devices. We introduce SpeechShield, an edge/cloud privacy preserving speech inference engine that can filter sensitive entities without compromising transcript accuracy. We utilize a timestamp based on-device masking approach that utilizes a token to entity prediction model to filter sensitive entities. Our choice of mask strategically conceals parts of the input and hides sensitive data. The masked input is sent to a trusted cloud service or to a local hub to generate the masked output. The effectiveness of SpeechShield hinges on how well the entity time segments are masked. Our recovery is a confidence score based approach that chooses the best prediction between cloud and on-device model. We implement SpeechShield on a 64 bit Raspberry Pi 4B. Experiments show that our solution leads to robust speech recognition without forsaking privacy. SpeechShield with < 100 MB memory, achieves state-of-the-art (SOTA) speech transcription performance while filtering about 83% of private entities directly on-device. SpeechShield is 16x smaller in memory, 3.3x faster and 17x more compute efficient than prior privacy preserving speech frameworks and has a relative reduction in word error rate (WER) by 38.8-77.5% when compared to existing offline transcription services.
>
---
#### [replaced 012] NeKo: Cross-Modality Post-Recognition Error Correction with Tasks-Guided Mixture-of-Experts Language Model
- **分类: cs.CL; cs.AI; cs.LG; cs.MA; eess.AS**

- **简介: 该论文针对多模态语音、语言、视觉到文本的识别后纠错任务，提出基于任务引导专家混合（MoE）的NeKo模型。通过让专家学习特定数据集特征并动态路由，实现单模型高效融合多领域知识，显著降低错误率，在多个基准上超越现有模型。**

- **链接: [https://arxiv.org/pdf/2411.05945v2](https://arxiv.org/pdf/2411.05945v2)**

> **作者:** Yen-Ting Lin; Zhehuai Chen; Piotr Zelasko; Zhen Wan; Xuesong Yang; Zih-Ching Chen; Krishna C Puvvada; Szu-Wei Fu; Ke Hu; Jun Wei Chiu; Jagadeesh Balam; Boris Ginsburg; Yu-Chiang Frank Wang; Chao-Han Huck Yang
>
> **备注:** ACL 2025 Industry Track. NeKo LMs: https://huggingface.co/nvidia/NeKo-v0-post-correction
>
> **摘要:** Construction of a general-purpose post-recognition error corrector poses a crucial question: how can we most effectively train a model on a large mixture of domain datasets? The answer would lie in learning dataset-specific features and digesting their knowledge in a single model. Previous methods achieve this by having separate correction language models, resulting in a significant increase in parameters. In this work, we present Mixture-of-Experts as a solution, highlighting that MoEs are much more than a scalability tool. We propose a Multi-Task Correction MoE, where we train the experts to become an ``expert'' of speech-to-text, language-to-text and vision-to-text datasets by learning to route each dataset's tokens to its mapped expert. Experiments on the Open ASR Leaderboard show that we explore a new state-of-the-art performance by achieving an average relative 5.0% WER reduction and substantial improvements in BLEU scores for speech and translation tasks. On zero-shot evaluation, NeKo outperforms GPT-3.5 and Claude-Opus with 15.5% to 27.6% relative WER reduction in the Hyporadise benchmark. NeKo performs competitively on grammar and post-OCR correction as a multi-task model.
>
---
#### [replaced 013] MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion via Mean Flows
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对零样本语音转换（VC）任务，解决现有流式方法在轻量化、高效性与泛化能力间的矛盾。提出MeanVC，通过分块自回归扩散变换器与均值流机制，实现单步采样、低参数量下的高质量语音转换，结合对抗后训练提升音质，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2510.08392v2](https://arxiv.org/pdf/2510.08392v2)**

> **作者:** Guobin Ma; Jixun Yao; Ziqian Ning; Yuepeng Jiang; Lingxin Xiong; Lei Xie; Pengcheng Zhu
>
> **摘要:** Zero-shot voice conversion (VC) aims to transfer timbre from a source speaker to any unseen target speaker while preserving linguistic content. Growing application scenarios demand models with streaming inference capabilities. This has created a pressing need for models that are simultaneously fast, lightweight, and high-fidelity. However, existing streaming methods typically rely on either autoregressive (AR) or non-autoregressive (NAR) frameworks, which either require large parameter sizes to achieve strong performance or struggle to generalize to unseen speakers. In this study, we propose MeanVC, a lightweight and streaming zero-shot VC approach. MeanVC introduces a diffusion transformer with a chunk-wise autoregressive denoising strategy, combining the strengths of both AR and NAR paradigms for efficient streaming processing. By introducing mean flows, MeanVC regresses the average velocity field during training, enabling zero-shot VC with superior speech quality and speaker similarity in a single sampling step by directly mapping from the start to the endpoint of the flow trajectory. Additionally, we incorporate diffusion adversarial post-training to mitigate over-smoothing and further enhance speech quality. Experimental results demonstrate that MeanVC significantly outperforms existing zero-shot streaming VC systems, achieving superior conversion quality with higher efficiency and significantly fewer parameters. Audio demos and code are publicly available at https://aslp-lab.github.io/MeanVC.
>
---
#### [replaced 014] SpeechJudge: Towards Human-Level Judgment for Speech Naturalness
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文针对语音合成中自然度评价缺乏大规模人类反馈数据的问题，提出SpeechJudge体系。构建99K语音对的人类偏好数据集，建立评估基准SpeechJudge-Eval，开发基于Qwen2.5的生成式奖励模型SpeechJudge-GRM，显著提升自然度判断准确率，助力语音模型与人类感知对齐。**

- **链接: [https://arxiv.org/pdf/2511.07931v2](https://arxiv.org/pdf/2511.07931v2)**

> **作者:** Xueyao Zhang; Chaoren Wang; Huan Liao; Ziniu Li; Yuancheng Wang; Li Wang; Dongya Jia; Yuanzhe Chen; Xiulin Li; Zhuo Chen; Zhizheng Wu
>
> **备注:** Dataset, Model, and Code: https://github.com/AmphionTeam/SpeechJudge
>
> **摘要:** Aligning large generative models with human feedback is a critical challenge. In speech synthesis, this is particularly pronounced due to the lack of a large-scale human preference dataset, which hinders the development of models that truly align with human perception. To address this, we introduce SpeechJudge, a comprehensive suite comprising a dataset, a benchmark, and a reward model centered on naturalness--one of the most fundamental subjective metrics for speech synthesis. First, we present SpeechJudge-Data, a large-scale human feedback corpus of 99K speech pairs. The dataset is constructed using a diverse set of advanced zero-shot text-to-speech (TTS) models across diverse speech styles and multiple languages, with human annotations for both intelligibility and naturalness preference. From this, we establish SpeechJudge-Eval, a challenging benchmark for speech naturalness judgment. Our evaluation reveals that existing metrics and AudioLLMs struggle with this task; the leading model, Gemini-2.5-Flash, achieves less than 70% agreement with human judgment, highlighting a significant gap for improvement. To bridge this gap, we develop SpeechJudge-GRM, a generative reward model (GRM) based on Qwen2.5-Omni-7B. It is trained on SpeechJudge-Data via a two-stage post-training process: Supervised Fine-Tuning (SFT) with Chain-of-Thought rationales followed by Reinforcement Learning (RL) with GRPO on challenging cases. On the SpeechJudge-Eval benchmark, the proposed SpeechJudge-GRM demonstrates superior performance, achieving 77.2% accuracy (and 79.4% after inference-time scaling @10) compared to a classic Bradley-Terry reward model (72.7%). Furthermore, SpeechJudge-GRM can be also employed as a reward function during the post-training of speech generation models to facilitate their alignment with human preferences.
>
---
#### [replaced 015] Speech Audio Generation from dynamic MRI via a Knowledge Enhanced Conditional Variational Autoencoder
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于语音生成任务，旨在从动态MRI序列中恢复受损语音。针对MRI采集环境导致的数据丢失与噪声问题，提出知识增强的条件变分自编码器（KE-CVAE），通过无标签数据增强与变分推理提升生成质量，实现高保真语音重建，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2503.06588v2](https://arxiv.org/pdf/2503.06588v2)**

> **作者:** Yaxuan Li; Han Jiang; Yifei Ma; Shihua Qin; Jonghye Woo; Fangxu Xing
>
> **摘要:** Dynamic Magnetic Resonance Imaging (MRI) of the vocal tract has become an increasingly adopted imaging modality for speech motor studies. Beyond image signals, systematic data loss, noise pollution, and audio file corruption can occur due to the unpredictability of the MRI acquisition environment. In such cases, generating audio from images is critical for data recovery in both clinical and research applications. However, this remains challenging due to hardware constraints, acoustic interference, and data corruption. Existing solutions, such as denoising and multi-stage synthesis methods, face limitations in audio fidelity and generalizability. To address these challenges, we propose a Knowledge Enhanced Conditional Variational Autoencoder (KE-CVAE), a novel two-step "knowledge enhancement + variational inference" framework for generating speech audio signals from cine dynamic MRI sequences. This approach introduces two key innovations: (1) integration of unlabeled MRI data for knowledge enhancement, and (2) a variational inference architecture to improve generative modeling capacity. To the best of our knowledge, this is one of the first attempts at synthesizing speech audio directly from dynamic MRI video sequences. The proposed method was trained and evaluated on an open-source dynamic vocal tract MRI dataset recorded during speech. Experimental results demonstrate its effectiveness in generating natural speech waveforms while addressing MRI-specific acoustic challenges, outperforming conventional deep learning-based synthesis approaches.
>
---
