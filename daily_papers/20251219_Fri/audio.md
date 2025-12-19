# 音频 cs.SD;  eess.AS

- **最新发布 9 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] CogSR: Semantic-Aware Speech Super-Resolution via Chain-of-Thought Guided Flow Matching
- **分类: cs.SD**

- **简介: 该论文属语音超分辨率任务，旨在解决极低采样率录音因缺失声学线索导致的语义失真与幻觉问题。提出CogSR框架，融合大音频语言模型的链式思维推理（语义锚定）与显式声学先验，引导校正流模型生成高保真、语言准确的高频细节。**

- **链接: [https://arxiv.org/pdf/2512.16304v1](https://arxiv.org/pdf/2512.16304v1)**

> **作者:** Jiajun Yuan; Xiaochen Wang; Yuhang Xiao; Yulin Wu; Chenhao Hu; Xueyang Lv
>
> **备注:** 7 pages
>
> **摘要:** Applying speech super-resolution (SR) to recordings with severely low sampling rates is a critical challenge in digital archiving and investigative audio recovery. In these scenarios, the input lacks essential acoustic cues. Consequently, existing generative models often fail; without sufficient context, they hallucinate phonetic content, guessing words based on probability rather than meaning. To address this, we propose CogSR, a framework designed specifically for high-precision, offline restoration. Our approach shifts the focus from simple signal mapping to cognitive reconstruction. By integrating a Large Audio-Language Model, we employ Chain-of-Thought reasoning to act as a semantic anchor, while explicit acoustic priors ensure the speaker's identity remains consistent. This guides a Rectified Flow backbone to synthesize high-frequency details that are not only realistic but linguistically accurate. Evaluations show that CogSR effectively eliminates ambiguity in severe degradation regimes, making it a robust solution for restoring high-value legacy and surveillance audio.
>
---
#### [new 002] BEST-STD2.0: Balanced and Efficient Speech Tokenizer for Spoken Term Detection
- **分类: eess.AS**

- **简介: 该论文面向查询示例式口语词检测（STD）任务，旨在提升噪声/混响下鲁棒性与令牌利用效率。提出噪声增强训练、最优传输正则化及TF-IDF检索机制，显著提升检索精度与速度。**

- **链接: [https://arxiv.org/pdf/2512.16395v1](https://arxiv.org/pdf/2512.16395v1)**

> **作者:** Anup Singh; Kris Demuynck; Vipul Arora
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Fast and accurate spoken content retrieval is vital for applications such as voice search. Query-by-Example Spoken Term Detection (STD) involves retrieving matching segments from an audio database given a spoken query. Token-based STD systems, which use discrete speech representations, enable efficient search but struggle with robustness to noise and reverberation, and with inefficient token utilization. We address these challenges by proposing a noise and reverberation-augmented training strategy to improve tokenizer robustness. In addition, we introduce optimal transport-based regularization to ensure balanced token usage and enhance token efficiency. To further speed up retrieval, we adopt a TF-IDF-based search mechanism. Empirical evaluations demonstrate that the proposed method outperforms STD baselines across various distortion levels while maintaining high search efficiency.
>
---
#### [new 003] Pseudo-Cepstrum: Pitch Modification for Mel-Based Neural Vocoders
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出伪倒谱（Pseudo-Cepstrum）方法，解决mel谱图的无训练、通用化音高修改问题。通过mel逆变换→DCT转倒谱→平移倒谱峰→IDCT与mel重映射，实现兼容任意mel基神经声码器的音高调整，无需模型修改或重训练。**

- **链接: [https://arxiv.org/pdf/2512.16519v1](https://arxiv.org/pdf/2512.16519v1)**

> **作者:** Nikolaos Ellinas; Alexandra Vioni; Panos Kakoulidis; Georgios Vamvoukakis; Myrsini Christidou; Konstantinos Markopoulos; Junkwang Oh; Gunu Jho; Inchul Hwang; Aimilios Chalamandaris; Pirros Tsiakoulis
>
> **摘要:** This paper introduces a cepstrum-based pitch modification method that can be applied to any mel-spectrogram representation. As a result, this method is compatible with any mel-based vocoder without requiring any additional training or changes to the model. This is achieved by directly modifying the cepstrum feature space in order to shift the harmonic structure to the desired target. The spectrogram magnitude is computed via the pseudo-inverse mel transform, then converted to the cepstrum by applying DCT. In this domain, the cepstral peak is shifted without having to estimate its position and the modified mel is recomputed by applying IDCT and mel-filterbank. These pitch-shifted mel-spectrogram features can be converted to speech with any compatible vocoder. The proposed method is validated experimentally with objective and subjective metrics on various state-of-the-art neural vocoders as well as in comparison with traditional pitch modification methods.
>
---
#### [new 004] Learning Recursive Attenuation Filters Under Noisy Conditions
- **分类: eess.AS**

- **简介: 该论文属音频信号处理任务，旨在解决噪声干扰下递归衰减滤波器（用于反馈延迟网络）的梯度优化失准问题。提出显式建模噪声的方法，改善损失函数景观，确保低信噪比下收敛至正确极小值，并通过80组实验验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.16318v1](https://arxiv.org/pdf/2512.16318v1)**

> **作者:** Gloria Dal Santo; Karolina Prawda; Sebastian J. Schlecht; Vesa Välimäki
>
> **备注:** Submitted to the Journal of Audio Engineering Society
>
> **摘要:** Recursion is a fundamental concept in the design of filters and audio systems. In particular, artificial reverberation systems that use delay networks depend on recursive paths to control both echo density and the decay rate of modal components. The differentiable digital signal processing framework has shown promise in automatically tuning both recursive and non-recursive elements given a target room impulse response. This is done by applying gradient descent to loss functions based on energy-decay or spectrogram differences. However, these representations are highly sensitive to background noise, which is ubiquitous in real measurements, producing spurious loss minima and leading to incorrect attenuation. This paper addresses the problem of tuning recursive attenuation filters of a feedback delay network when targets are noisy. We examine the loss landscape associated with different optimization objectives and propose a method that ensures correct minima under low signal-to-noise conditions. We demonstrate the effectiveness of the proposed approach through statistical analysis on 80 individual optimization examples. The results reveal that explicitly modeling the noise restores correct minima. Furthermore, we identify the sensitivity of attenuation filter parameters tuning to perturbations in frequency-independent parameters. These findings provide practical guidelines for more robust and reproducible gradient-based optimization of feedback delay networks.
>
---
#### [new 005] DPDFNet: Boosting DeepFilterNet2 via Dual-Path RNN
- **分类: cs.SD**

- **简介: 该论文面向因果单通道语音增强任务，旨在解决低信噪比、多语言真实场景下的语音质量与实时性兼顾难题。作者提出DPDFNet模型，引入双路径RNN增强时频建模，设计抗过衰减损失与“常开”微调策略，并构建新评测集、提出PRISM综合指标，验证其在边缘NPU上的实时部署能力。**

- **链接: [https://arxiv.org/pdf/2512.16420v1](https://arxiv.org/pdf/2512.16420v1)**

> **作者:** Daniel Rika; Nino Sapir; Ido Gus
>
> **摘要:** We present DPDFNet, a causal single-channel speech enhancement model that extends DeepFilterNet2 architecture with dual-path blocks in the encoder, strengthening long-range temporal and cross-band modeling while preserving the original enhancement framework. In addition, we demonstrate that adding a loss component to mitigate over-attenuation in the enhanced speech, combined with a fine-tuning phase tailored for "always-on" applications, leads to substantial improvements in overall model performance. To compare our proposed architecture with a variety of causal open-source models, we created a new evaluation set comprising long, low-SNR recordings in 12 languages across everyday noise scenarios, better reflecting real-world conditions than commonly used benchmarks. On this evaluation set, DPDFNet delivers superior performance to other causal open-source models, including some that are substantially larger and more computationally demanding. We also propose an holistic metric named PRISM, a composite, scale-normalized aggregate of intrusive and non-intrusive metrics, which demonstrates clear scalability with the number of dual-path blocks. We further demonstrate on-device feasibility by deploying DPDFNet on Ceva-NeuPro-Nano edge NPUs. Results indicate that DPDFNet-4, our second-largest model, achieves real-time performance on NPN32 and runs even faster on NPN64, confirming that state-of-the-art quality can be sustained within strict embedded power and latency constraints.
>
---
#### [new 006] Domain-Agnostic Causal-Aware Audio Transformer for Infant Cry Classification
- **分类: cs.SD; cs.AI**

- **简介: 该论文面向婴儿哭声分类任务，旨在解决现有模型因依赖相关性特征而鲁棒性差、泛化能力弱的问题。提出DACH-TIC模型，融合因果注意力、分层表征、多任务学习与对抗域泛化，提升噪声鲁棒性、跨环境泛化性与因果可解释性。**

- **链接: [https://arxiv.org/pdf/2512.16271v1](https://arxiv.org/pdf/2512.16271v1)**

> **作者:** Geofrey Owino; Bernard Shibwabo Kasamani; Ahmed M. Abdelmoniem; Edem Wornyo
>
> **备注:** This paper has been published in the IEEE proceedings of the 8th International Conference of Computer and Informatics Engineering (IC2IE)
>
> **摘要:** Accurate and interpretable classification of infant cry paralinguistics is essential for early detection of neonatal distress and clinical decision support. However, many existing deep learning methods rely on correlation-driven acoustic representations, which makes them vulnerable to noise, spurious cues, and domain shifts across recording environments. We propose DACH-TIC, a Domain-Agnostic Causal-Aware Hierarchical Audio Transformer for robust infant cry classification. The model integrates causal attention, hierarchical representation learning, multi-task supervision, and adversarial domain generalization within a unified framework. DACH-TIC employs a structured transformer backbone with local token-level and global semantic encoders, augmented by causal attention masking and controlled perturbation training to approximate counterfactual acoustic variations. A domain-adversarial objective promotes environment-invariant representations, while multi-task learning jointly optimizes cry type recognition, distress intensity estimation, and causal relevance prediction. The model is evaluated on the Baby Chillanto and Donate-a-Cry datasets, with ESC-50 environmental noise overlays for domain augmentation. Experimental results show that DACH-TIC outperforms state-of-the-art baselines, including HTS-AT and SE-ResNet Transformer, achieving improvements of 2.6 percent in accuracy and 2.2 points in macro-F1 score, alongside enhanced causal fidelity. The model generalizes effectively to unseen acoustic environments, with a domain performance gap of only 2.4 percent, demonstrating its suitability for real-world neonatal acoustic monitoring systems.
>
---
#### [new 007] From Minutes to Days: Scaling Intracranial Speech Decoding with Supervised Pretraining
- **分类: cs.SD; cs.CL; q-bio.NC**

- **简介: 该论文属脑机接口中的侵入式语音解码任务，旨在解决训练数据稀缺与跨日神经信号漂移问题。作者利用患者数天临床监测的颅内+音频长时数据预训练对比学习模型，大幅提升解码性能，并揭示需建模日间变异性。**

- **链接: [https://arxiv.org/pdf/2512.15830v1](https://arxiv.org/pdf/2512.15830v1)**

> **作者:** Linnea Evanson; Mingfang; Zhang; Hubert Banville; Saarang Panchavati; Pierre Bourdillon; Jean-Rémi King
>
> **备注:** Linnea Evanson* and Mingfang (Lucy) Zhang* are joint first authors. Pierre Bourdillon** and Jean-Rémi King** are joint last authors
>
> **摘要:** Decoding speech from brain activity has typically relied on limited neural recordings collected during short and highly controlled experiments. Here, we introduce a framework to leverage week-long intracranial and audio recordings from patients undergoing clinical monitoring, effectively increasing the training dataset size by over two orders of magnitude. With this pretraining, our contrastive learning model substantially outperforms models trained solely on classic experimental data, with gains that scale log-linearly with dataset size. Analysis of the learned representations reveals that, while brain activity represents speech features, its global structure largely drifts across days, highlighting the need for models that explicitly account for cross-day variability. Overall, our approach opens a scalable path toward decoding and modeling brain representations in both real-life and controlled task settings.
>
---
#### [new 008] Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属语音翻译任务，探究SpeechLLM（端到端语音大模型）是否优于传统级联方案。作者构建首个综合评测套件Hearing to Translate，对比5个SpeechLLM与16个级联系统，在16项基准、13语对、9种挑战条件下评估，发现级联系统整体更可靠，SpeechLLM仅在部分场景匹敌。**

- **链接: [https://arxiv.org/pdf/2512.16378v1](https://arxiv.org/pdf/2512.16378v1)**

> **作者:** Sara Papi; Javier Garcia Gilabert; Zachary Hopton; Vilém Zouhar; Carlos Escolano; Gerard I. Gállego; Jorge Iranzo-Sánchez; Ahrii Kim; Dominik Macháček; Patricia Schmidtova; Maike Züfle
>
> **备注:** Project available at https://github.com/sarapapi/hearing2translate
>
> **摘要:** As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
>
---
#### [new 009] Poster: Recognizing Hidden-in-the-Ear Private Key for Reliable Silent Speech Interface Using Multi-Task Learning
- **分类: cs.HC; eess.AS**

- **简介: 该论文提出HEar-ID，一种基于多任务学习的静音语音接口，利用普通降噪耳机同时实现静音拼写识别与说话人身份认证。旨在解决SSI系统缺乏身份验证的问题，通过共享编码器联合优化两个任务，在同一模型中兼顾可靠性与实用性。**

- **链接: [https://arxiv.org/pdf/2512.16518v1](https://arxiv.org/pdf/2512.16518v1)**

> **作者:** Xuefu Dong; Liqiang Xu; Lixing He; Zengyi Han; Ken Christofferson; Yifei Chen; Akihito Taya; Yuuki Nishiyama; Kaoru Sezaki
>
> **备注:** UbiComp Poster 2025
>
> **摘要:** Silent speech interface (SSI) enables hands-free input without audible vocalization, but most SSI systems do not verify speaker identity. We present HEar-ID, which uses consumer active noise-canceling earbuds to capture low-frequency "whisper" audio and high-frequency ultrasonic reflections. Features from both streams pass through a shared encoder, producing embeddings that feed a contrastive branch for user authentication and an SSI head for silent spelling recognition. This design supports decoding of 50 words while reliably rejecting impostors, all on commodity earbuds with a single model. Experiments demonstrate that HEar-ID achieves strong spelling accuracy and robust authentication.
>
---
## 更新

#### [replaced 001] DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec
- **分类: cs.SD**

- **简介: 该论文属语音合成任务，旨在解决现有语音编解码器中韵律与音色强耦合导致的零样本可控性差问题。提出DisCo-Speech框架，含解耦语音编解码器（DisCodec）和LM生成器：DisCodec通过三因子解耦（内容/韵律/音色）与融合重建，实现韵律独立控制与音色注入。**

- **链接: [https://arxiv.org/pdf/2512.13251v2](https://arxiv.org/pdf/2512.13251v2)**

> **作者:** Tao Li; Wenshuo Ge; Zhichao Wang; Zihao Cui; Yong Ma; Yingying Gao; Chao Deng; Shilei Zhang; Junlan Feng
>
> **摘要:** Recent codec-based language models~(LMs) have revolutionized text-to-speech~(TTS). However, since standard codecs tightly couple timbre and prosody, continuation-based LMs inevitably replicate this entanglement, hindering independent control. Recent efforts attempt to break this entanglement via codec design, but insufficient decoupling remains a critical bottleneck. To tackle this challenge, we propose DisCo-Speech, a zero-shot controllable TTS framework that enables prosody control and voice cloning via a disentangled speech codec (DisCodec) and an LM-based generator. The core component, DisCodec, contains two core stages: 1) Tri-factor disentanglement, which explicitly factorizes speech into content, prosody, and timbre subspaces via parallel encoders and hybrid losses; and 2) Fusion and reconstruction, which fuses content and prosody into unified content-prosody tokens suitable for LM prediction, while jointly optimizing reconstruction quality to resolve the disentanglement-reconstruction trade-off. With this design, the LM performs prosodic continuation from a style prompt while the decoder handles target timbre injection, enabling flexible zero-shot control. Experiments show that DisCo-Speech matches state-of-the-art voice cloning performance while outperforming baselines in zero-shot prosody control. By resolving the core entanglement at the codec level, DisCo-Speech provides a robust foundation for controllable speech synthesis. Audio samples are available at https://github.com/disco-speech/DisCo-Speech, and the code and weights will be released at the same link.
>
---
#### [replaced 002] Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属AI评估方法学任务，旨在解决语音基础模型评价标准零散、不匹配的问题。作者提出三维正交分类法（评估维度、模型能力、任务要求），系统梳理现有评测基准，揭示评测缺口，并为模型与评测的合理匹配提供理论框架与实践指南。**

- **链接: [https://arxiv.org/pdf/2510.19509v2](https://arxiv.org/pdf/2510.19509v2)**

> **作者:** Maureen de Seyssel; Eeshan Gunesh Dhekane
>
> **备注:** 57 pages (26 main, 25 appendix, 6 references)
>
> **摘要:** Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the evaluation aspect being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.
>
---
#### [replaced 003] Spoken DialogSum: An Emotion-Rich Conversational Dataset for Spoken Dialogue Summarization
- **分类: cs.CL; cs.AI; cs.LG; eess.AS**

- **简介: 该论文面向口语对话摘要任务，旨在解决情感感知与语音建模缺乏对齐数据的问题。作者构建了首个语音-文本-情感对齐数据集Spoken DialogSum，含13,460条带情绪标签的合成对话及双类型摘要，并验证了端到端Audio-LLM在情感摘要上的优势。**

- **链接: [https://arxiv.org/pdf/2512.14687v2](https://arxiv.org/pdf/2512.14687v2)**

> **作者:** Yen-Ju Lu; Kunxiao Gao; Mingrui Liang; Helin Wang; Thomas Thebaud; Laureano Moro-Velazquez; Najim Dehak; Jesus Villalba
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Recent audio language models can follow long conversations. However, research on emotion-aware or spoken dialogue summarization is constrained by the lack of data that links speech, summaries, and paralinguistic cues. We introduce Spoken DialogSum, the first corpus aligning raw conversational audio with factual summaries, emotion-rich summaries, and utterance-level labels for speaker age, gender, and emotion. The dataset is built in two stages: first, an LLM rewrites DialogSum scripts with Switchboard-style fillers and back-channels, then tags each utterance with emotion, pitch, and speaking rate. Second, an expressive TTS engine synthesizes speech from the tagged scripts, aligned with paralinguistic labels. Spoken DialogSum comprises 13,460 emotion-diverse dialogues, each paired with both a factual and an emotion-focused summary. We release an online demo at https://fatfat-emosum.github.io/EmoDialog-Sum-Audio-Samples/, with plans to release the full dataset in the near future. Baselines show that an Audio-LLM raises emotional-summary ROUGE-L by 28% relative to a cascaded ASR-LLM system, confirming the value of end-to-end speech modeling.
>
---
#### [replaced 004] Optimizing tiny colorless feedback delay networks
- **分类: eess.AS; cs.SD**

- **简介: 该论文属音频信号处理任务，旨在解决人工混响中因延迟线少导致的频谱着色（金属 ringing）问题。提出一种可微小反馈延迟网络（仅4条延迟线），联合优化反馈矩阵与增益，以提升频谱平坦度并保持脉冲响应密度，有效降低听觉着色且计算更高效。**

- **链接: [https://arxiv.org/pdf/2402.11216v4](https://arxiv.org/pdf/2402.11216v4)**

> **作者:** Gloria Dal Santo; Karolina Prawda; Sebastian J. Schlecht; Vesa Välimäki
>
> **摘要:** A common bane of artificial reverberation algorithms is spectral coloration in the synthesized sound, typically manifesting as metallic ringing, leading to a degradation in the perceived sound quality. In delay network methods, coloration is more pronounced when fewer delay lines are used. This paper presents an optimization framework in which a tiny differentiable feedback delay network, with as few as four delay lines, is used to learn a set of parameters to iteratively reduce coloration. The parameters under optimization include the feedback matrix, as well as the input and output gains. The optimization objective is twofold: to maximize spectral flatness through a spectral loss while maintaining temporal density by penalizing sparseness in the parameter values. A favorable narrow distribution of modal excitation is achieved while maintaining the desired impulse response density. In a subjective assessment, the new method proves effective in reducing perceptual coloration of late reverberation. Compared to the author's previous work, which serves as the baseline and utilizes a sparsity loss in the time domain, the proposed method achieves computational savings while maintaining performance. The effectiveness of this work is demonstrated through two application scenarios where smooth-sounding synthetic room impulse responses are obtained via the introduction of attenuation filters and an optimizable scattering feedback matrix.
>
---
#### [replaced 005] Similarity Metrics For Late Reverberation
- **分类: eess.AS**

- **简介: 该论文属音频信号处理任务，旨在解决现有通用音频相似度度量不适用于 late reverberation 评估的问题。提出两种可微、面向晚期混响统计特性的新度量（基于平均功率与频带能量衰减），并在大规模房间脉冲响应数据集上验证其优于基线方法。**

- **链接: [https://arxiv.org/pdf/2408.14836v2](https://arxiv.org/pdf/2408.14836v2)**

> **作者:** Gloria Dal Santo; Karolina Prawda; Sebastian J. Schlecht; Vesa Välimäki
>
> **摘要:** Automatic tuning of reverberation algorithms relies on the optimization of a cost function. While general audio similarity metrics are useful, they are not optimized for the specific statistical properties of reverberation in rooms. This paper presents two novel metrics for assessing the similarity of late reverberation in room impulse responses. These metrics are differentiable and can be utilized within a machine-learning framework. We compare the performance of these metrics to two popular audio metrics using a large dataset of room impulse responses encompassing various room configurations and microphone positions. The results indicate that the proposed functions based on averaged power and frequency-band energy decay outperform the baselines with the former exhibiting the most suitable profile towards the minimum. The proposed work holds promise as an improvement to the design and evaluation of reverberation similarity metrics.
>
---
#### [replaced 006] Adaptive Edge-Cloud Inference for Speech-to-Action Systems Using ASR and Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对语音控制IoT设备中云与边缘推理的权衡问题，提出自适应边缘-云协同推理框架ASTA。它融合端侧ASR与轻量语言模型、云端大模型，依据实时系统指标动态路由，并引入命令验证与修复机制，提升鲁棒性与资源效率。**

- **链接: [https://arxiv.org/pdf/2512.12769v2](https://arxiv.org/pdf/2512.12769v2)**

> **作者:** Mohammad Jalili Torkamani; Israt Zarin
>
> **备注:** preprint, 6 pages, 7 figures, 1 table
>
> **摘要:** Voice-based interaction has emerged as a natural and intuitive modality for controlling IoT devices. However, speech-driven edge devices face a fundamental trade-off between cloud-based solutions, which offer stronger language understanding capabilities at the cost of latency, connectivity dependence, and privacy concerns, and edge-based solutions, which provide low latency and improved privacy but are limited by computational constraints. This paper presents ASTA, an adaptive speech-to-action solution that dynamically routes voice commands between edge and cloud inference to balance performance and system resource utilization. ASTA integrates on-device automatic speech recognition and lightweight offline language-model inference with cloud-based LLM processing, guided by real-time system metrics such as CPU workload, device temperature, and network latency. A metric-aware routing mechanism selects the inference path at runtime, while a rule-based command validation and repair component ensures successful end-to-end command execution. We implemented our solution on an NVIDIA Jetson-based edge platform and evaluated it using a diverse dataset of 80 spoken commands. Experimental results show that ASTA successfully routes all input commands for execution, achieving a balanced distribution between online and offline inference. The system attains an ASR accuracy of 62.5% and generates executable commands without repair for only 47.5% of inputs, highlighting the importance of the repair mechanism in improving robustness. These results suggest that adaptive edge-cloud orchestration is a viable approach for resilient and resource-aware voice-controlled IoT systems.
>
---
