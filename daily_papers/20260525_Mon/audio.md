# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] Word-Level Modeling with Alignment-Aware Acoustic Fusion for Text-Assisted Intelligibility Prediction in Listeners with Hearing Loss
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音 intelligibility 预测任务，解决听力障碍者在CPC3中的文本辅助听觉理解预测问题。通过词级建模和对齐感知的声学融合提升预测精度。**

- **链接: [https://arxiv.org/pdf/2605.23604](https://arxiv.org/pdf/2605.23604)**

> **作者:** Kazushi Nakazawa
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We address text-assisted speech intelligibility prediction for hearing-impaired listeners in CPC3. Although the target is a sentence-level percentage, it is determined by reference-word recognition outcomes. We formulate prediction as reference-conditioned word-level correctness modeling: a frozen Whisper encoder analyzes degraded speech, a teacher-forced decoder conditions on the canonical transcript, and sentence intelligibility is obtained by averaging predicted correctness probabilities over valid reference words. To complement transcript-conditioned decoder states, we add a word-aligned local acoustic branch based on character-level cross-attention alignment and an utterance-level global acoustic branch for calibration. On the official evaluation set, the decoder baseline obtains RMSE 24.92 and correlation 0.795, while joint fusion improves to incorrect-word F1 0.778, MCC 0.626, correlation 0.806, and RMSE 24.39. A similar trend with Whisper medium suggests that the gain comes from prediction granularity and alignment-aware fusion.
>
---
#### [new 002] Natural Yet Challenging to Detect: Robust In-the-Wild TTS through EMA and Dual-Scoring Prompt Selection -- Submission for WildSpoof 2026 TTS Track
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在生成自然且难以检测的语音。通过引入EMA和双评分提示选择，提升语音质量和欺骗性。**

- **链接: [https://arxiv.org/pdf/2605.23859](https://arxiv.org/pdf/2605.23859)**

> **作者:** Renhe Sun; Jiayi Zhou; Haolin He; Yueying Feng; Jian Liu
>
> **摘要:** In this technical report, we describe our submission for the WildSpoof Challenge TTS Track: Text-to-Speech with In-the-Wild Data. We introduce F5-TTS-DPS, a model built upon the F5-TTS architecture. Our approach integrates Exponential Moving Average (EMA) into supervised fine-tuning to stabilize training and improve generalization. To enhance synthesis fidelity, we leverage large language models (LLMs) and large audio language models (LALMs) for dual-scoring prompt selection, filtering reference audio and text prompts to ensure quality while addressing alignment issues in noisy datasets. Experimental evaluation demonstrates that F5-TTS-DPS achieves strong performance with UTMOS of 3.20 and speaker similarity of 0.51 on the development set. More importantly, our model achieves the best a-DCF scores of 0.1582, 0.5233, and 0.2562 across three advanced SASV systems among all submissions, indicating our synthesized speech is the most difficult to detect and exhibits the highest degree of naturalness and authenticity. Combined with competitive WER performance, these results validate the effectiveness of our approach in generating natural-sounding speech with strong spoofing capabilities.
>
---
#### [new 003] Evaluating the Temporal Detection Capability of Integrated Gradients Applied on Sound Classifier
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文研究音频分类中集成梯度的时序检测能力，解决无时序监督下声音事件定位问题。通过合成数据评估IG方法的定位效果，与不同监督方式的CNN对比，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.23293](https://arxiv.org/pdf/2605.23293)**

> **作者:** Martynas Dumpis; Tuomas Virtanen
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Gradient-based attribution methods can highlight input regions important for neural network predictions, but their effectiveness for temporal sound event detection in audio classification has not been systematically evaluated. This paper assesses whether integrated gradients (IG) can temporally detect sound events when applied to a classifier trained without temporal supervision. We use synthetic polyphonic audio with ground truth timestamps to measure alignment between IG attributions and event boundaries. On a 10-class domestic sound dataset, IG achieves mean Intersection over Union (IoU) of 0.39, frame-level F1 of 0.52, and Pointing Game accuracy of 82.6\%. For comparison, a framewise CNN trained with weak supervision (FW-WS, clip-level training labels) achieves 0.42 IoU, 0.55 F1, and 97.3\% PG, while a strongly supervised variant (FW-SS, frame-level training labels) reaches 0.45 IoU, 0.58 F1, and 97.9\% PG. Overall, these results suggest that post-hoc IG captures meaningful temporal activity patterns of sound events, with localization performance approaching models that explicitly produce frame-level predictions. All methods substantially outperform random and energy-based baselines.
>
---
#### [new 004] AffectCodec: Emotion-Preserving Neural Speech Codec with Block-Diagonal Residual FSQ
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，旨在解决情绪信息在低比特率下丢失的问题。通过结构化量化方法，AffectCodec有效保留情绪特征，同时保持语音质量。**

- **链接: [https://arxiv.org/pdf/2605.23373](https://arxiv.org/pdf/2605.23373)**

> **作者:** Zhaoyang Meng; Zhengyao Ma; Kecan Mao; Yingming Gao; Ya Li
>
> **摘要:** Neural speech codecs have become the discrete interface between raw audio and speech language models, yet they remain optimized primarily for acoustic reconstruction fidelity, which leaves emotion-relevant cues vulnerable to being discarded during quantization, limiting the affective capacity of downstream models. We trace this degradation to two mechanisms: reconstruction-driven bit allocation under limited bitrate and cross-stream leakage in concatenation-based codecs, where acoustic gradients can overwrite nominally emotion-reserved dimensions. We propose AffectCodec, an emotion-preserving neural speech codec built on Block-Diagonal Residual Finite Scalar Quantization (BD-RFSQ). By imposing block-diagonal input and output projections over emotion and acoustic subspaces, BD-RFSQ transforms bit allocation from implicit and loss-driven to explicit and structurally guaranteed, while still preserving a flat token interface for downstream speech language models. AffectCodec further combines this structurally constrained quantizer with multi-granularity emotion conditioning and multi-rate training, enabling robust affect preservation at low bitrates. Experiments across multiple emotional speech benchmarks show that AffectCodec substantially improves emotion preservation, especially in the low-bitrate regime, while maintaining competitive acoustic quality and intelligibility. These results suggest that structurally protected quantization is an effective principle for preserving emotion-relevant information and may provide a general route toward attribute-aware neural speech compression.
>
---
#### [new 005] MixFake: Benchmarking and Enhancing Audio Deepfake Detection in Diverse Real-world Mixed Audio
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决真实场景下混合语音检测难题。提出MixFake数据集和多流提示调优框架，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2605.23201](https://arxiv.org/pdf/2605.23201)**

> **作者:** Qingcao Li; Yipeng Lin; Weichen Lian; Zhongjie Ba; Peng Cheng; Zhichao Lian
>
> **备注:** Accepted by ICME2026
>
> **摘要:** Speech deepfake detection has achieved remarkable success in clean environments but faces significant challenges in complex, real-world scenarios where speech is often mixed with background music or noise. Current state-of-the-art methods rely on semantic features from self-supervised learning (SSL) models, which often fail when processing non-speech or mixed-source audio. In this paper, we first introduce MixFake, a large-scale benchmark dataset designed to simulate diverse acoustic environments with varying SNR levels and mixed authenticity components. To address the "semantic-centric" limitation, we propose a Multi-stream Prompt Tuning framework that injects signal-level priors into SSL backbones. By integrating base, frequency, and texture streams through deep prompt injection, our model effectively captures acoustic artifacts. Experimental results demonstrate that our method significantly outperforms existing baselines, achieving a 0.95% EER in foreground detection and a substantial 7.72% absolute improvement in complex background detection tasks. Our dataset and code are available at this https URL.
>
---
#### [new 006] A study on weakly-supervised training approaches for phoneme-level pronunciation scoring
- **分类: eess.AS**

- **简介: 该论文属于语音识别中的发音评估任务，旨在减少对音素级标注的依赖。通过弱监督学习，利用更高层级标签训练模型，提升音素级评分效果。**

- **链接: [https://arxiv.org/pdf/2605.23593](https://arxiv.org/pdf/2605.23593)**

> **作者:** Jazmín Vidal; Luciana Ferrer
>
> **摘要:** Phoneme-level computer-assisted pronunciation training systems typically rely on phoneme-level annotations, which are costly and scarce. In this work, we investigate whether phoneme-level mispronunciation information can be learned without phoneme-level supervision by exploiting higher-level pronunciation labels. Specifically, we study a weakly supervised setting in which models are trained using only utterance- or word-level pronunciation labels and analyze whether this supervision induces useful phoneme-level score predictions. We further consider a two-stage training scenario in which a model trained only with utterance-level labels is finetuned using a limited number of carefully-selected phoneme-level labeled utterances. We find that, using our proposed architecture and selection process, the two-stage process leads to comparable results to those obtained with full phoneme-level supervision, requiring only a small fraction of phoneme-level labels.
>
---
#### [new 007] StepAudio 2.5 Technical Report
- **分类: eess.AS**

- **简介: 该论文提出StepAudio 2.5，解决统一音频语言模型在ASR、TTS和实时交互中的性能不足问题，通过RLHF和专用解码实现多任务优化。**

- **链接: [https://arxiv.org/pdf/2605.23463](https://arxiv.org/pdf/2605.23463)**

> **作者:** Bin Lin; Bo Zhao; Boyong Wu; Chao Yan; Chen Wu; Cheng Yi; Chengyuan Yao; Daijiao Liu; Fei Tian; Feng Tian; Haiyang Sun; Haoyang Zhang; Jiangjie Zhen; Jinglan Gong; Jun Chen; Li Xie; Peilin Li; Peng Yang; Pengfei Tan; Qingjian Lin; Runze Li; Shenghua Hu; Siyi Zhou; Wenwen Qu; Xiangyu Li; Xiangyu Tony Zhang; Xuerui Yang; Yang Yang; Yechang Huang; Yu Fu; Yuchu Luo; Yuxin Li; Yuxin Zhang; Zhengyan Sheng; Brian Li; Chang Zeng; Changlin Zhang; Chen Geng; Chenghao Dong; Chengli Feng; Dan Zhou; Danni Wan; Di Chen; Die Zhang; Dongqing Pang; Guanglong Yang; Guoqiang Hu; Huangxi Zhu; Jianzheng Gao; Jinghua Liang; Jinmei Wan; Junjie Yuan; Kang An; Lei Lei; Limin Zhong; Lun Cai; Mengqiang Ren; Min Xu; Mingliang Li; Mingxiao Li; Na Wang; Qiang Tong; Qiaoling Huang; Qingfu Du; Rui Wang; Shengchen Zhou; Shi Qiu; Shihao Peng; Shiliang Yang; Siqi Tu; Tianjiao Deng; Ting Xu; Tong Wang; WeiMing Niu; Wuxun Xie; Xianwei Zhang; Xianyu Feng; Xiaojia Liu; Xing Chen; Xiongbin Wu; Yan Wu; Yang Li; Yi Liu; Yifan Zhang; Yile Liu; Yongshen Long; Yu Luo; Yuanhao Ding; Yuhao Wang; Yuhe Yin; Yunfang Xu; Yuxiang Yang; Zhiguo Huang; Zhiyue Wu; Zichao Li; Zichao Zhou; Daxin Jiang; Future Li; Gang Yu; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** Unified audio-language modeling has emerged as a prominent trend in modern speech systems, promising to bring the reasoning capabilities of large language models to auditory tasks. However, existing unified foundations often struggle to match the depth of specialized systems across automatic speech recognition (ASR), text-to-speech synthesis (TTS), and realtime spoken interaction. Bridging this gap remains an open challenge. This report presents StepAudio 2.5, a unified audio-language foundation model that matches or exceeds specialized systems across all three capabilities. Rather than treating these tasks as architecturally distinct, we operate on the premise that once text and audio share a multimodal representational space, task specialization becomes a matter of operational regimes: data construction, optimization targets, and decoding constraints. Guided by this insight, we advance the post-training paradigm from standard supervised learning to task-tailored Reinforcement Learning from Human Feedback (RLHF), using it as the primary mechanism to define complex optimization targets. We leverage this RLHF-centric alignment, alongside specialized decoding, to shape a shared backbone into three distinct operational modes. Concretely, the ASR branch advances transcription efficiency via verifiable multi-token decoding; the TTS branch achieves controllable, expressive synthesis through preference-based RLHF and context-rich supervision; and the Realtime branch realizes low-latency, persona-consistent dialogue via generative reward modeling within an RLHF framework. On standard benchmarks, StepAudio 2.5 achieves state-of-the-art results across ASR, TTS, and Realtime, demonstrating that a singular audio-language foundation can successfully internalize the distinct deployment objectives of speech understanding, generation, and live interaction.
>
---
#### [new 008] Frame-Aligned Fusion of Canary and WavLM for Non-Intrusive Intelligibility Prediction of Hearing-Aid-Processed Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于非侵入式语音可懂性预测任务，旨在评估助听器处理后语音的可理解性。通过融合Canary和WavLM模型，在帧对齐基础上提升预测性能。**

- **链接: [https://arxiv.org/pdf/2605.23619](https://arxiv.org/pdf/2605.23619)**

> **作者:** Kazushi Nakazawa
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** Non-intrusive intelligibility prediction estimates how well hearing-impaired listeners understand hearing-aid-processed speech without a clean reference. We study this task in the 3rd Clarity Prediction Challenge using two frozen speech encoders, Canary and WavLM. The central question is not only whether complementary pretrained representations should be combined, but where their interaction should occur. We compare single-backbone baselines, uniform score averaging, pool-late fusion, cross-attention, frame-aligned fusion, and reverse alignment under a shared left/right-preserving binaural framework. Among the compared systems, the best model temporally prepares WavLM with a learnable strided convolution and fuses it with Canary on the coarser Canary timeline before pooling, reaching Eval RMSE 24.96$\pm$0.06 and Eval Corr 0.796$\pm$0.001. Severity, enhancement-system, layer-window, and temporal-shift analyses indicate that coarse local temporal correspondence before pooling is a useful inductive bias for this task.
>
---
#### [new 009] UniSRM: A Unified Speech Reward Model for Reasoning-Based Fine-grained Assessment
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音生成评估任务，旨在解决人工评价成本高、主观性强的问题。提出UniSRM模型，实现多维度、可解释的语音质量评估。**

- **链接: [https://arxiv.org/pdf/2605.23261](https://arxiv.org/pdf/2605.23261)**

> **作者:** Yuanyuan Wang; Dongchao Yang; Yayue Deng; Zhiyong Wu; Yiwen Guo; Helen Meng; Xixin Wu
>
> **备注:** Accepted by ACL 2026(Main)
>
> **摘要:** Evaluating speech generation still relies heavily on human judgments, such as Mean Opinion Score (MOS), which are expensive, subjective, and difficult to reproduce at scale. While a few recent studies have begun to explore AudioLLM-based judge models, existing efforts typically target only a narrow set of scenarios (e.g., utterance-level quality or single-turn dialogue) and provide limited coverage of diverse speech generation tasks and evaluation dimensions. In this work, we propose UniSRM, a unified speech reward model that can support multi-dimensional, interpretable reward signals with reliable reasoning. To support training and evaluation, we introduce UniSRM-Data and UniSRM-Bench, covering speech evaluation tasks from utterance-level quality to context-level coherence. Based on this dataset, we present the unified speech reward model, UniSRM, with a two-stage pipeline that enables reasoning-based fine-grained assessment. Furthermore, we introduce Reasoning-Consistent Rewards to improve the reliability of the reasoning process. Experiments show that UniSRM delivers more reliable and human-aligned judgments across a broad range of speech evaluation tasks, offering a practical foundation for scalable and unified evaluation of speech quality.
>
---
#### [new 010] Articulatory strategy as a source of variation in acoustic vowel dynamics
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音学研究，探讨发音策略如何影响元音声学动态。通过分析36名英语母语者的数据，发现舌位变化与元音共振峰动态相关，揭示了个体发音差异的机制。**

- **链接: [https://arxiv.org/pdf/2605.23416](https://arxiv.org/pdf/2605.23416)**

> **作者:** Patrycja Strycharczuk; Justin J. H. Lo; Sam Kirkham
>
> **摘要:** Acoustic vowel dynamics have some speaker-identifying characteristics, which have been ascribed to individual properties of articulatory strategies: formant transitions have a particular shape because speakers move their articulators, using specific and practised movements. However, there is little existing evidence that different articulatory strategies systematically affect formant dynamics. The present study corroborates the link between the two. Ultrasound tongue imaging data from 36 speakers of Northern-Anglo English are used to identify distinct articulatory strategies for the production of palatal vowel /i/. Tongue shape in /i/ is found to be a significant predictor of formant dynamics in diphthongs with a palatal offglide. The observed relationships can be explained by the characteristics of articulatory movement conditioned by vocal tract shape. Greater articulatory displacement of tongue root and/or dorsum produces greater distortion from the mean tongue shape in palatal vowels, and it also requires higher articulatory velocities, resulting in relatively earlier and steeper formant transitions. The results contribute to the conceptual understanding of individuality in speech, by illuminating the regularising and individual aspects of articulatory compensation.
>
---
## 更新

#### [replaced 001] Codec-Robust Attacks on Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频大语言模型的对抗攻击问题，提出CodecAttack方法，在编码器潜空间生成鲁棒扰动，突破压缩防御，验证了编解码器感知攻击对实际部署系统的威胁。**

- **链接: [https://arxiv.org/pdf/2605.20519](https://arxiv.org/pdf/2605.20519)**

> **作者:** Jaechul Roh; Jean-Philippe Monteuuis; Jonathan Petit; Amir Houmansadr
>
> **摘要:** Prior attacks on Audio Large Language Models (Audio LLMs) demonstrated that carefully crafted waveform-domain perturbations can force targeted adversarial outputs. As a defense mechanism against these attacks, real-world codec compression preprocessing has been studied to both detect and remove the perturbations. Yet no existing attack has demonstrated robustness against these compressions. We introduce CodecAttack, which optimizes a perturbation in a neural audio codec's continuous latent space rather than directly perturbing the audio waveform. We show that the codec's compression channel, which discards waveform perturbations, transmits perturbations crafted in its own latent space. To further harden the attack across real-world compression channels, we apply multi-bitrate straight-through Expectation-over-Transformation (EoT), all without modifying the target model. Across three realistic Audio LLM deployment scenarios and three target models, CodecAttack achieves an average 85.5% target-substring attack success rate (ASR) on Opus at moderate bitrates, while the waveform baseline trained with identical EoT hardening does not exceed 26% at any bitrate. The attack transfers to held-out codecs, reaching up to 100% ASR on MP3 and 84% on AAC-LC without retraining. A per-band energy analysis shows that the latent perturbation concentrates below 4kHz, exactly where codecs allocate the most bits, while the waveform baseline spreads into higher frequencies that codecs discard. These results demonstrate that lossy compression is not a reliable defense against adversarial audio and that codec-aware attacks pose a practical threat to deployed Audio LLM systems.
>
---
#### [replaced 002] Data Augmentation for Pathological Speech Enhancement
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在提升帕金森病患者病理语音的增强效果。通过数据增强策略，研究不同方法对模型性能的影响，发现噪声增强效果最佳。**

- **链接: [https://arxiv.org/pdf/2602.14671](https://arxiv.org/pdf/2602.14671)**

> **作者:** Mingchi Hou; Enno Hermann; Ina Kodrasi
>
> **备注:** Accepted at EUSIPCO 2026
>
> **摘要:** The performance of state-of-the-art speech enhancement (SE) models considerably degrades for pathological speech due to atypical acoustic characteristics and limited data availability. This paper systematically investigates data augmentation (DA) strategies to improve SE performance for pathological speakers affected by Parkinson`s disease, evaluating both predictive and generative SE models. We examine three DA categories, i.e., transformative, generative, and noise augmentation, assessing their impact with objective SE metrics. Experimental results show that noise augmentation consistently delivers the largest and most robust gains, transformative augmentations provide moderate improvements, while generative augmentation yields limited benefits and can harm performance as the amount of synthetic data increases. Furthermore, we show that the effectiveness of DA varies depending on the SE model, with DA being more beneficial for predictive SE models. While our results demonstrate that DA improves SE performance for pathological speakers, a performance gap between neurotypical and pathological speech persists, highlighting the need for future research on targeted DA strategies for pathological speech.
>
---
#### [replaced 003] XAttnMark: Learning Robust Audio Watermarking with Cross-Attention
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于音频水印任务，旨在解决版权保护与真实性验证问题。提出XAttnMark方法，通过交叉注意力机制和时频掩码损失提升水印的鲁棒性和不可感知性。**

- **链接: [https://arxiv.org/pdf/2502.04230](https://arxiv.org/pdf/2502.04230)**

> **作者:** Yixin Liu; Lie Lu; Jihui Jin; Lichao Sun; Andrea Fanelli
>
> **备注:** Accepted at ICML'25
>
> **摘要:** The rapid proliferation of generative audio synthesis and editing technologies has raised serious concerns about copyright infringement, data provenance, and the spread of misinformation via deepfake audio. Watermarking offers a proactive solution by embedding imperceptible yet identifiable and traceable signals into audio content. While recent neural network-based watermarking methods like WavMark and AudioSeal have improved robustness and quality, they struggle to jointly optimize both robust detection and accurate attribution. This paper introduces Cross-Attention Robust Audio Watermark (XATTNMARK), which bridges this gap by leveraging partial parameter sharing between the generator and the detector, a cross-attention mechanism for efficient message retrieval, and a temporal conditioning module for improved message distribution. Additionally, we propose a psychoacoustic-aligned time-frequency (TF) masking loss that captures fine-grained auditory masking effects, improving watermark imperceptibility. XATTNMARK achieves state-of-the-art performance in both detection and attribution, demonstrating superior robustness against a wide range of audio transformations, including challenging generative editing at varying strengths. This work advances audio watermarking for protecting intellectual property and ensuring authenticity in the era of generative AI.
>
---
#### [replaced 004] Real-time, EDM-inspired sonification of the activity of a supercomputer
- **分类: cs.SD**

- **简介: 论文探讨实时将超级计算机活动数据转化为音乐的声景化方法，属于数据声景化任务。旨在实现持续监控系统行为，解决长期监听中信息可理解与吸引人的问题。通过选择EDM风格，生成连贯且具持续性的音乐。**

- **链接: [https://arxiv.org/pdf/2605.21874](https://arxiv.org/pdf/2605.21874)**

> **作者:** Marco Alunno; Paolo Bientinesi
>
> **备注:** 7 pages, 2 figures, accepted conference paper
>
> **摘要:** The project described in this paper explores the informative sonification of data received in real time from a supercomputer. These data capture the current activities in all the nodes of the computer, therefore, their sonification functions as a form of continuous monitoring of the nodes' behavior and, by extension, of the system as a whole. Because such monitoring is theoretically unending, the resulting sonification must be musically capable of conveying information through sound in a way that remains both intelligible and engaging over long durations. Rather than imposing a predefined musical style onto the data, we sought to identify one which the data themselves could plausibly support. From a small set of candidates, we selected EDM because it is a family of genres whose structural and temporal characteristics align well with continuous, data-driven processes and long-term listening. Through this style-based approach, this research builds on the long tradition of computer data sonification while uniquely combining three elements rarely addressed together: monitoring (rather than debugging) as the primary goal, real-time (rather than post-mortem) data interpretation, and generation of virtually infinite and stylistically coherent (rather than incongruous) music.
>
---
