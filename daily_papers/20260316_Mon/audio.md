# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Room Impulse Response Completion Using Signal-Prediction Diffusion Models Conditioned on Simulated Early Reflections
- **分类: eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决RIR生成中早期反射缺失的问题。通过扩散模型结合ISG模拟的早期反射，生成更真实的RIR。**

- **链接: [https://arxiv.org/pdf/2603.12442](https://arxiv.org/pdf/2603.12442)**

> **作者:** Zeyu Xu; Andreas Brendel; Albert G. Prinn; Emanuël A. P. Habets
>
> **备注:** The following article has been submitted for review to Interspeech 2026
>
> **摘要:** Room impulse responses (RIRs) are fundamental to audio data augmentation, acoustic signal processing, and immersive audio rendering. While geometric simulators such as the image source method (ISM) can efficiently generate early reflections, they lack the realism of measured RIRs due to missing acoustic wave effects. We propose a diffusion-based RIR completion method using signal-prediction conditioned on ISM-simulated direct-path and early reflections. Unlike state-of-the-art methods, our approach imposes no fixed duration constraint on the input early reflections. We further incorporate classifier-free guidance to steer generation toward a target distribution learned from physically realistic RIRs simulated with the Treble SDK. Objective evaluation demonstrates that the proposed method outperforms a state-of-the-art baseline in early RIR completion and energy decay curve reconstruction.
>
---
#### [new 002] Speech-Worthy Alignment for Japanese SpeechLLMs via Direct Preference Optimization
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音生成任务，旨在解决日本SpeechLLM输出风格不适合语音合成的问题。通过偏好对齐方法优化输出，使其更口语化、自然。**

- **链接: [https://arxiv.org/pdf/2603.12565](https://arxiv.org/pdf/2603.12565)**

> **作者:** Mengjie Zhao; Lianbo Liu; Yusuke Fujita; Hao Shi; Yuan Gao; Roman Koshkin; Yui Sudo
>
> **摘要:** SpeechLLMs typically combine ASR-trained encoders with text-based LLM backbones, leading them to inherit written-style output patterns unsuitable for text-to-speech synthesis. This mismatch is particularly pronounced in Japanese, where spoken and written registers differ substantially in politeness markers, sentence-final particles, and syntactic complexity. We propose a preference-based alignment approach to adapt Japanese SpeechLLMs for speech-worthy outputs: text that is concise, conversational, and readily synthesized as natural speech. To rigorously evaluate this task, we introduce SpokenElyza, a benchmark for Japanese speech-worthiness derived from ELYZA-tasks-100 with auditory verification by native experts. Experiments show that our approach achieves substantial improvement on SpokenElyza while largely preserving performance on the original written-style evaluation. We will release SpokenElyza to support future research on Japanese spoken dialog systems.
>
---
#### [new 003] Mask2Flow-TSE: Two-Stage Target Speaker Extraction with Masking and Flow Matching
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于目标说话人提取任务，旨在从混合语音中提取特定说话人的声音。提出Mask2Flow-TSE框架，结合掩码和流匹配，在单次推理中实现高质量语音重建。**

- **链接: [https://arxiv.org/pdf/2603.12837](https://arxiv.org/pdf/2603.12837)**

> **作者:** Junwon Moon; Hyunjin Choi; Hansol Park; Heeseung Kim; Kyuhong Shim
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Target speaker extraction (TSE) extracts the target speaker's voice from overlapping speech mixtures given a reference utterance. Existing approaches typically fall into two categories: discriminative and generative. Discriminative methods apply time-frequency masking for fast inference but often over-suppress the target signal, while generative methods synthesize high-quality speech at the cost of numerous iterative steps. We propose Mask2Flow-TSE, a two-stage framework combining the strengths of both paradigms. The first stage applies discriminative masking for coarse separation, and the second stage employs flow matching to refine the output toward target speech. Unlike generative approaches that synthesize speech from Gaussian noise, our method starts from the masked spectrogram, enabling high-quality reconstruction in a single inference step. Experiments show that Mask2Flow-TSE achieves comparable performance to existing generative TSE methods with approximately 85M parameters.
>
---
#### [new 004] DAST: A Dual-Stream Voice Anonymization Attacker with Staged Training
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音匿名化攻击任务，旨在破解语音匿名化技术泄露的说话人信息。提出双流攻击模型，通过三阶段训练提升攻击效果，有效识别匿名语音中的说话人特征。**

- **链接: [https://arxiv.org/pdf/2603.12840](https://arxiv.org/pdf/2603.12840)**

> **作者:** Ridwan Arefeen; Xiaoxiao Miao; Rong Tong; Aik Beng Ng; Simon See; Timothy Liu
>
> **摘要:** Voice anonymization masks vocal traits while preserving linguistic content, which may still leak speaker-specific patterns. To assess and strengthen privacy evaluation, we propose a dual-stream attacker that fuses spectral and self-supervised learning features via parallel encoders with a three-stage training strategy. Stage I establishes foundational speaker-discriminative representations. Stage II leverages the shared identity-transformation characteristics of voice conversion and anonymization, exposing the model to diverse converted speech to build cross-system robustness. Stage III provides lightweight adaptation to target anonymized data. Results on the VoicePrivacy Attacker Challenge (VPAC) dataset demonstrate that Stage II is the primary driver of generalization, enabling strong attacking performance on unseen anonymization datasets. With Stage III, fine-tuning on only 10\% of the target anonymization dataset surpasses current state-of-the-art attackers in terms of EER.
>
---
#### [new 005] MamTra: A Hybrid Mamba-Transformer Backbone for Speech Synthesis
- **分类: eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决Transformer模型计算复杂度高、Mamba模型缺乏全局上下文的问题。提出MamTra框架，结合两者优势，并通过知识迁移提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2603.12342](https://arxiv.org/pdf/2603.12342)**

> **作者:** Tan Dat Nguyen; Sangmin Bae; Joon Son Chung; Ji-Hoon Kim
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Despite the remarkable quality of LLM-based text-to-speech systems, their reliance on autoregressive Transformers leads to quadratic computational complexity, which severely limits practical applications. Linear-time alternatives, notably Mamba, offer a potential remedy; however, they often sacrifice the global context essential for expressive synthesis. In this paper, we propose MamTra, an interleaved Mamba-Transformer framework designed to leverage the advantages of Mamba's efficiency and Transformers' modeling capability. We also introduce novel knowledge transfer strategies to distill insights from a pretrained Transformer into our hybrid architecture, thereby bypassing the prohibitive costs of training from scratch. Systematic experiments identify the optimal hybrid configuration, and demonstrate that MamTra reduces inference VRAM usage by up to 34% without compromising speech fidelity - even trained on only 2% of the original training dataset. Audio samples are available at this https URL.
>
---
#### [new 006] Self-Supervised Speech Models Encode Phonetic Context via Position-dependent Orthogonal Subspaces
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型如何编码音素及其上下文，属于语音表示学习任务。解决的问题是理解单帧表示如何组合音素信息。工作包括提出音素向量在位置相关正交子空间中叠加的结构。**

- **链接: [https://arxiv.org/pdf/2603.12642](https://arxiv.org/pdf/2603.12642)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David R. Mortensen; David Harwath
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Transformer-based self-supervised speech models (S3Ms) are often described as contextualized, yet what this entails remains unclear. Here, we focus on how a single frame-level S3M representation can encode phones and their surrounding context. Prior work has shown that S3Ms represent phones compositionally; for example, phonological vectors such as voicing, bilabiality, and nasality vectors are superposed in the S3M representation of [m]. We extend this view by proposing that phonological information from a sequence of neighboring phones is also compositionally encoded in a single frame, such that vectors corresponding to previous, current, and next phones are superposed within a single frame-level representation. We show that this structure has several properties, including orthogonality between relative positions, and emergence of implicit phonetic boundaries. Together, our findings advance our understanding of context-dependent S3M representations.
>
---
#### [new 007] Perpetual Dialogues: A Computational Analysis of Voice-Guitar Interaction in Carlos Paredes's Discography
- **分类: cs.SD**

- **简介: 该论文属于音乐分析任务，旨在研究卡洛斯·帕雷德斯作品中人声与吉他互动的计算方法，解决无乐谱传统音乐的结构分析问题，通过音频分析技术揭示其表现协调特征。**

- **链接: [https://arxiv.org/pdf/2603.12854](https://arxiv.org/pdf/2603.12854)**

> **作者:** Gilberto Bernardes; Nádia Moura; António Sá Pinto
>
> **备注:** 8 pages, 8 figures, to be published in ICMC 2026
>
> **摘要:** Computational musicology enables systematic analysis of performative and structural traits in recorded music, yet existing approaches remain largely tailored to notated, score-based repertoires. This study advances a methodology for analyzing voice-guitar interaction in Carlos Paredes's vocal collaborations - an oral-tradition context where compositional and performative layers co-emerge. Using source-separated stems, physics-informed harmonic modelling, and beat-level audio descriptors, we examine melodic, harmonic, and rhythmic relationships across eight recordings with four singers. Our commonality-diversity framework, combining multi-scale correlation analysis with residual-based detection of structural deviations, reveals that expressive coordination is predominantly piece-specific rather than corpus-wide. Diversity events systematically align with formal boundaries and textural shifts, demonstrating that the proposed approach can identify musically salient reorganizations with minimal human annotation. The framework further offers a generalizable computational strategy for repertoires without notated blueprints, extending Music Performance Analysis into oral-tradition and improvisation-inflected practices.
>
---
#### [new 008] Bounds on Agreement between Subjective and Objective Measurements
- **分类: eess.AS; eess.IV**

- **简介: 该论文属于多媒体质量评估任务，旨在解决主观与客观测量间一致性评估问题。通过建立理论边界，提供无需投票方差信息的计算方法，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2603.13204](https://arxiv.org/pdf/2603.13204)**

> **作者:** Jaden Pieper; Stephen D. Voran
>
> **备注:** Currently under review at IEEE Transactions on Multimedia. Submitted 5 November 2025, revised 3 March 2026
>
> **摘要:** Objective estimators of multimedia quality are often judged by comparing estimates with subjective "truth data," most often via Pearson correlation coefficient (PCC) or mean-squared error (MSE). But subjective test results contain noise, so striving for a PCC of 1.0 or an MSE of 0.0 is neither realistic nor repeatable. Numerous efforts have been made to acknowledge and appropriately accommodate subjective test noise in objective-subjective comparisons, typically resulting in new analysis frameworks and figures-of-merit. We take a different approach. By making only basic assumptions, we derive bounds on PCC and MSE that can be expected for a subjective test. Consistent with intuition, these bounds are functions of subjective vote variance. When a subjective test includes vote variance information, the calculation of the bounds is easy, and in this case we say the resulting bounds are "fully data-driven." We provide two options for calculating bounds in cases where vote variance information is not available. One option is to use vote variance information from other subjective tests that do provide such information, and the second option is to use a model for subjective votes. Thus we introduce a binomial-based model for subjective votes (BinoVotes) that naturally leads to a mean opinion score (MOS) model, named BinoMOS, with multiple unique desirable properties. BinoMOS reproduces the discrete nature of MOS values and its dependence on the number of votes per file. This modeling provides vote variance information required by the PCC and MSE bounds and we compare this modeling with data from 18 subjective tests. The modeling yields PCC and MSE bounds that agree very well with those found from the data directly. These results allow one to set expectations for the PCC and MSE that might be achieved for any subjective test, even those where vote variance information is not available.
>
---
#### [new 009] TASTE-Streaming: Towards Streamable Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音语言建模任务，解决语音与文本对齐的延迟问题。提出TASTE-S，实现实时流式处理，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.12350](https://arxiv.org/pdf/2603.12350)**

> **作者:** Liang-Hsuan Tseng; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** Text-speech joint spoken language modeling (SLM) aims at natural and intelligent speech-based interactions, but developing such a system may suffer from modality mismatch: speech unit sequences are much longer than text tokens. Prior work reduces this gap with text-aligned tokenization and embedding (TASTE), producing speech tokens that align in lengths with their textual counterparts. However, the dependence on an external ASR system and the use of a non-causal decoder limits streaming use. To address this limitation, we propose TASTE-S, a streamable extension of TASTE suitable for real-time usage. TASTE-S integrates a CTC-based ASR module into the encoder for instant dual-modality encoding. We also redesign the unit decoder to enable on-the-fly decoding. With joint training, we show that TASTE-S matches TASTE's performance while significantly reducing latency. Further investigations reveal that TASTE-S remains robust to transcriptions and enables long-form encoding and decoding.
>
---
#### [new 010] RadEar: A Self-Supervised RF Backscatter System for Voice Eavesdropping and Separation
- **分类: cs.NI; cs.SD**

- **简介: 该论文提出RadEar系统，用于通过墙壁进行语音窃听与分离。属于隐私安全任务，解决弱信号和语音重叠问题，采用自监督学习实现语音分离与降噪。**

- **链接: [https://arxiv.org/pdf/2603.12446](https://arxiv.org/pdf/2603.12446)**

> **作者:** Qijun Wang; Peihao Yan; Chunqi Qian; Huacheng Zeng
>
> **备注:** Accepted by IEEE INFOCOM 2026
>
> **摘要:** Eavesdropping on voice conversations presents a growing threat to personal privacy and information security. In this paper, we present RadEar, a novel RF backscatter-based system designed to enable covert voice eavesdropping through walls. RadEar consists of two key components: (i) a batteryless RF backscatter tag covertly deployed inside the target space, and (ii) an RF reader located outside the room that performs signal demodulation, voice separation, and denoising. The tag features a compact, dual-resonator design that achieves energy-efficient frequency modulation for continuous voice eavesdropping while mitigating self-interference by separating excitation and reflection frequencies. To overcome the challenges of weak signal reception and overlapping speech, the RF reader employs self-supervised learning models for voice separation and denoising, trained using a remix-based objective without requiring ground-truth labels. We fabricate and evaluate RadEar in real-world scenarios, demonstrating its ability to recover and separate human speech with high fidelity under practical constraints.
>
---
## 更新

#### [replaced 001] Which Data Matter? Embedding-Based Data Selection for Speech Recognition
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决专用模型在广泛数据中训练时的性能问题。通过嵌入表示选择相关数据子集，提升目标领域的识别效果。**

- **链接: [https://arxiv.org/pdf/2603.05819](https://arxiv.org/pdf/2603.05819)**

> **作者:** Zakaria Aldeneh; Skyler Seto; Maureen de Seyssel; Jie Chi; Zijin Gu; Takuya Higuchi; Jee-weon Jung; Shinji Watanabe; David Grangier; Barry-John Theobald; Tatiana Likhomanenko
>
> **摘要:** Modern ASR systems are typically trained on large-scale pseudo-labeled, in-the-wild data spanning multiple domains. While such heterogeneous data benefit generalist models designed for broad deployment, they pose challenges for specialist models targeting specific domains: specialist models lack the capacity to learn from all available data, and one must pay closer attention to addressing the mismatch between training and test conditions. In this work, we study targeted data selection as a strategy to address these challenges, selecting relevant subsets from 100k hours of in-the-wild training data to optimize performance on target domains. We represent speech samples using embeddings that capture complementary characteristic--speaker attributes, phonetic content, and semantic meaning--and analyze how relevance and diversity along these axes when performing data selection affect downstream ASR performance. Our experiments with CTC-based Conformer models show that training on a strategically selected 5% subset can exceed the performance of models trained on the full dataset by up to 36.8% relative WER reduction on target domains.
>
---
#### [replaced 002] Mitigating Latent Mismatch in cVAE-Based Singing Voice Synthesis via Flow Matching
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于歌唱语音合成任务，旨在解决cVAE模型中训练与推理时潜在表示不匹配的问题。通过流匹配方法优化潜在空间，提升合成质量。**

- **链接: [https://arxiv.org/pdf/2601.00217](https://arxiv.org/pdf/2601.00217)**

> **作者:** Minhyeok Yun; Yong-Hoon Choi
>
> **摘要:** Singing voice synthesis (SVS) aims to generate natural and expressive singing waveforms from symbolic musical scores. In cVAE-based SVS, however, a mismatch arises because the decoder is trained with latent representations inferred from target singing signals, while inference relies on latent representations predicted only from conditioning inputs. This discrepancy can weaken fine expressive acoustic details in the synthesized output. To mitigate this issue, we propose FM-Singer, a flow-matching-based latent refinement framework for cVAE-based singing voice synthesis. Rather than redesigning the acoustic decoder, the proposed method learns a continuous vector field that transports inference-time latent samples toward posterior-like latent representations through ODE-based integration before waveform generation. Because the refinement is performed in latent space, the method remains lightweight and compatible with a strong parallel synthesis backbone. Experimental results on Korean and Chinese singing datasets show that the proposed latent refinement improves objective metrics and perceptual quality while maintaining practical synthesis efficiency. These results suggest that reducing training-inference latent mismatch is a useful direction for improving expressive singing voice synthesis. Code, pre-trained checkpoints, and audio demos are available at this https URL.
>
---
#### [replaced 003] Lightweight speech enhancement guided target speech extraction in noisy multi-speaker scenarios
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决噪声环境下多说话人场景中的目标语音提取问题。提出轻量级模型GTCRN和两种扩展方法，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.19583](https://arxiv.org/pdf/2508.19583)**

> **作者:** Ziling Huang; Junnan Wu; Lichun Fan; Zhenbo Luo; Jian Luan; Haixin Guan; Yanhua Long
>
> **备注:** Submitted to Computer Speech & Language
>
> **摘要:** Target speech extraction (TSE) has achieved strong performance in relatively simple conditions such as one-speaker-plus-noise and two-speaker mixtures, but its performance remains unsatisfactory in noisy multi-speaker scenarios. To address this issue, we introduce a lightweight speech enhancement model, GTCRN, to better guide TSE in noisy environments. Building on our competitive previous speaker embedding/encoder-free framework SEF-PNet, we propose two extensions: LGTSE and D-LGTSE. LGTSE incorporates noise-agnostic enrollment guidance by denoising the input noisy speech before context interaction with enrollment speech, thereby reducing noise interference. D-LGTSE further improves system robustness against speech distortion by leveraging denoised speech as an additional noisy input during training, expanding the dynamic range of noisy conditions and enabling the model to directly learn from distorted signals. Furthermore, we propose a two-stage training strategy, first with GTCRN enhancement-guided pre-training and then joint fine-tuning, to fully exploit model this http URL on the Libri2Mix dataset demonstrate significant improvements of 0.89 dB in SISDR, 0.16 in PESQ, and 1.97% in STOI, validating the effectiveness of our approach.
>
---
#### [replaced 004] Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言语音识别任务，旨在提升多语言对话场景下的识别准确率。通过创新的编码器-适配器-大语言模型架构和多阶段训练策略，实现了优异的识别性能。**

- **链接: [https://arxiv.org/pdf/2507.17288](https://arxiv.org/pdf/2507.17288)**

> **作者:** Miaomiao Gao; Xiaoxiao Xiang; Yiwen Guo
>
> **备注:** Accepted By Interspeech 2025 MLC-SLM workshop
>
> **摘要:** This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking.
>
---
#### [replaced 005] TripleC Learning and Lightweight Speech Enhancement for Multi-Condition Target Speech Extraction
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，解决多条件目标语音提取问题。通过引入TripleC学习策略和轻量级前端，提升模型在不同场景下的泛化能力与性能。**

- **链接: [https://arxiv.org/pdf/2512.04945](https://arxiv.org/pdf/2512.04945)**

> **作者:** Ziling Huang
>
> **备注:** in submisssion
>
> **摘要:** In our recent work, we proposed Lightweight Speech Enhancement Guided Target Speech Extraction (LGTSE) and demonstrated its effectiveness in multi-speaker-plus-noise scenarios. However, real-world applications often involve more diverse and complex conditions, such as one-speaker-plus-noise or two-speaker-without-noise. To address this challenge, we extend LGTSE with a Cross-Condition Consistency learning strategy, termed TripleC Learning. This strategy is first validated under multi-speaker-plus-noise condition and then evaluated for its generalization across diverse scenarios. Moreover, building upon the lightweight front-end denoiser in LGTSE, which can flexibly process both noisy and clean mixtures and shows strong generalization to unseen conditions, we integrate TripleC learning with a proposed parallel universal training scheme that organizes batches containing multiple scenarios for the same target speaker. By enforcing consistent extraction across different conditions, easier cases can assist harder ones, thereby fully exploiting diverse training data and fostering a robust universal model. Experimental results on the Libri2Mix three-condition tasks demonstrate that the proposed LGTSE with TripleC learning achieves superior performance over condition-specific models, highlighting its strong potential for universal deployment in real-world speech applications.
>
---
#### [replaced 006] LESS: Large Language Model Enhanced Semi-Supervised Learning for Speech Foundational Models Using in-the-wild Data
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音基础模型的半监督学习任务，旨在解决真实场景数据中伪标签质量低的问题。通过引入大语言模型优化伪标签，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.04586](https://arxiv.org/pdf/2506.04586)**

> **作者:** Wen Ding; Fan Qian
>
> **备注:** Accepted by ICASSP 2026
>
> **摘要:** Although state-of-the-art Speech Foundational Models can produce high-quality text pseudo-labels, applying Semi-Supervised Learning (SSL) for in-the-wild real-world data remains challenging due to its richer and more complex acoustics compared to curated datasets. To address the challenges, we introduce LESS (Large Language Model Enhanced Semi-supervised Learning), a versatile framework that uses Large Language Models (LLMs) to correct pseudo-labels generated on in-the-wild data. In the LESS framework, pseudo-labeled text from Automatic Speech Recognition (ASR) or Automatic Speech Translation (AST) of the unsupervised data is refined by an LLM, and further improved by a data filtering strategy. Across Mandarin ASR and Spanish-to-English AST evaluations, LESS delivers consistent gains, with an absolute Word Error Rate reduction of 3.8% on WenetSpeech, and BLEU score increase of 0.8 and 0.7, achieving 34.0 on Callhome and 64.7 on Fisher testsets respectively. These results highlight LESS's effectiveness across diverse languages, tasks, and domains. We have released the recipe as open source to facilitate further research in this area.
>
---
#### [replaced 007] nlm: Real-Time Non-linear Modal Synthesis in Max
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决实时非线性模态合成的问题。通过开发Max外挂模块nlm，实现对弦、膜、板的物理参数交互控制与多通道输出。**

- **链接: [https://arxiv.org/pdf/2603.10240](https://arxiv.org/pdf/2603.10240)**

> **作者:** Rodrigo Diaz; Rodrigo Constanzo; Mark Sandler
>
> **备注:** accepted to PdMaxCon25~ (this https URL)
>
> **摘要:** We present nlm, a set of Max externals that enable efficient real-time non-linear modal synthesis for strings, membranes, and plates. The externals, implemented in C++, offer interactive control of physical parameters, allow the loading of custom modal data, and provide multichannel output. By integrating interactive physical-modelling capabilities into a familiar environment, nlm lowers the barrier for composers, performers, and sound designers to explore the expressive potential of non-linear modal synthesis. The externals are available as open-source software at this https URL.
>
---
#### [replaced 008] OmniForcing: Unleashing Real-time Joint Audio-Visual Generation
- **分类: cs.MM; cs.CV; cs.SD**

- **简介: 该论文提出OmniForcing，解决音频-视觉生成中的实时性问题。通过蒸馏技术将双向模型转为流式自回归生成，提升推理速度并保持多模态同步与质量。**

- **链接: [https://arxiv.org/pdf/2603.11647](https://arxiv.org/pdf/2603.11647)**

> **作者:** Yaofeng Su; Yuming Li; Zeyue Xue; Jie Huang; Siming Fu; Haoran Li; Ying Li; Zezhong Qian; Haoyang Huang; Nan Duan
>
> **备注:** 14 pages
>
> **摘要:** Recent joint audio-visual diffusion models achieve remarkable generation quality but suffer from high latency due to their bidirectional attention dependencies, hindering real-time applications. We propose OmniForcing, the first framework to distill an offline, dual-stream bidirectional diffusion model into a high-fidelity streaming autoregressive generator. However, naively applying causal distillation to such dual-stream architectures triggers severe training instability, due to the extreme temporal asymmetry between modalities and the resulting token sparsity. We address the inherent information density gap by introducing an Asymmetric Block-Causal Alignment with a zero-truncation Global Prefix that prevents multi-modal synchronization drift. The gradient explosion caused by extreme audio token sparsity during the causal shift is further resolved through an Audio Sink Token mechanism equipped with an Identity RoPE constraint. Finally, a Joint Self-Forcing Distillation paradigm enables the model to dynamically self-correct cumulative cross-modal errors from exposure bias during long rollouts. Empowered by a modality-independent rolling KV-cache inference scheme, OmniForcing achieves state-of-the-art streaming generation at $\sim$25 FPS on a single GPU, maintaining multi-modal synchronization and visual quality on par with the bidirectional teacher.\textbf{Project Page:} \href{this https URL}{this https URL}
>
---
#### [replaced 009] MAGE: A Coarse-to-Fine Speech Enhancer with Masked Generative Model
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决效率与感知质量的平衡问题。提出MAGE模型，采用粗到细的掩码策略和轻量校正模块，提升效果并减少参数量。**

- **链接: [https://arxiv.org/pdf/2509.19881](https://arxiv.org/pdf/2509.19881)**

> **作者:** Hieu Pham; Tan Dat Nguyen; Phuong Thanh Tran; Joon Son Chung; Duc Dung Nguyen
>
> **备注:** ICASSP 2026
>
> **摘要:** Speech enhancement remains challenging due to the trade-off between efficiency and perceptual quality. In this paper, we introduce MAGE, a Masked Audio Generative Enhancer that advances generative speech enhancement through a compact and robust design. Unlike prior masked generative models with random masking, MAGE employs a scarcity-aware coarse-to-fine masking strategy that prioritizes frequent tokens in early steps and rare tokens in later refinements, improving efficiency and generalization. We also propose a lightweight corrector module that further stabilizes inference by detecting low-confidence predictions and re-masking them for refinement. Built on BigCodec and finetuned from Qwen2.5-0.5B, MAGE is reduced to 200M parameters through selective layer retention. Experiments on DNS Challenge and noisy LibriSpeech show that MAGE achieves state-of-the-art perceptual quality and significantly reduces word error rate for downstream recognition, outperforming larger baselines. Audio examples are available at this https URL.
>
---
#### [replaced 010] On Deepfake Voice Detection -- It's All in the Presentation
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音防伪任务，旨在解决深度伪造音频在真实场景中检测效果差的问题。通过改进数据集和研究方法，提升了检测准确率。**

- **链接: [https://arxiv.org/pdf/2509.26471](https://arxiv.org/pdf/2509.26471)**

> **作者:** Héctor Delgado; Giorgio Ramondetti; Emanuele Dalmasso; Gennady Karvitsky; Daniele Colibro; Haydar Talib
>
> **备注:** ICASSP 2026. \c{opyright}IEEE Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** While the technologies empowering malicious audio deepfakes have dramatically evolved in recent years due to generative AI advances, the same cannot be said of global research into spoofing (deepfake) countermeasures. This paper highlights how current deepfake datasets and research methodologies led to systems that failed to generalize to real world application. The main reason is due to the difference between raw deepfake audio, and deepfake audio that has been presented through a communication channel, e.g. by phone. We propose a new framework for data creation and research methodology, allowing for the development of spoofing countermeasures that would be more effective in real-world scenarios. By following the guidelines outlined here we improved deepfake detection accuracy by 39% in more robust and realistic lab setups, and by 57% on a real-world benchmark. We also demonstrate how improvement in datasets would have a bigger impact on deepfake detection accuracy than the choice of larger SOTA models would over smaller models; that is, it would be more important for the scientific community to make greater investment on comprehensive data collection programs than to simply train larger models with higher computational demands.
>
---
#### [replaced 011] Dynamically Slimmable Speech Enhancement Network with Metric-Guided Training
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在降低轻量模型的复杂度。提出DSN网络和MGT方法，动态调整计算资源，提升效率。**

- **链接: [https://arxiv.org/pdf/2510.11395](https://arxiv.org/pdf/2510.11395)**

> **作者:** Haixin Zhao; Kaixuan Yang; Nilesh Madhu
>
> **备注:** Accepted by ICASSP2026
>
> **摘要:** To further reduce the complexity of lightweight speech enhancement models, we introduce a gating-based Dynamically Slimmable Network (DSN). The DSN comprises static and dynamic components. For architecture-independent applicability, we introduce distinct dynamic structures targeting the commonly used components, namely, grouped recurrent neural network units, multi-head attention, convolutional, and fully connected layers. A policy module adaptively governs the use of dynamic parts at a frame-wise resolution according to the input signal quality, controlling computational load. We further propose Metric-Guided Training (MGT) to explicitly guide the policy module in assessing input speech quality. Experimental results demonstrate that the DSN achieves comparable enhancement performance in instrumental metrics to the state-of-the-art lightweight baseline, while using only 73% of its computational load on average. Evaluations of dynamic component usage ratios indicate that the MGT-DSN can appropriately allocate network resources according to the severity of input signal distortion.
>
---
