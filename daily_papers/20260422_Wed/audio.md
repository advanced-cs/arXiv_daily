# 音频 cs.SD;  eess.AS

- **最新发布 19 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Deep Supervised Contrastive Learning of Pitch Contours for Robust Pitch Accent Classification in Seoul Korean
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音处理任务，旨在解决 Seoul Korean 中连续 F0 轮廓到离散语调类别的映射问题。提出 Dual-Glob 框架，通过深度对比学习捕捉整体轮廓特征，提升音高重音分类性能。**

- **链接: [https://arxiv.org/pdf/2604.19477](https://arxiv.org/pdf/2604.19477)**

> **作者:** Hyunjung Joo; GyeongTaek Lee
>
> **摘要:** The intonational structure of Seoul Korean has been defined with discrete tonal categories within the Autosegmental-Metrical model of intonational phonology. However, it is challenging to map continuous $F_0$ contours to these invariant categories due to variable $F_0$ realizations in real-world speech. Our paper proposes Dual-Glob, a deep supervised contrastive learning framework to robustly classify fine-grained pitch accent patterns in Seoul Korean. Unlike conventional local predictive models, our approach captures holistic $F_0$ contour shapes by enforcing structural consistency between clean and augmented views in a shared latent space. To this aim, we introduce the first large-scale benchmark dataset, consisting of manually annotated 10,093 Accentual Phrases in Seoul Korean. Experimental results show that our Dual-Glob significantly outperforms strong baseline models with state-of-the-art accuracy (77.75%) and F1-score (51.54%). Therefore, our work supports AM-based intonational phonology using data-driven methodology, showing that deep contrastive learning effectively captures holistic structural features of continuous $F_0$ contours.
>
---
#### [new 002] Self-Noise Reduction for Capacitive Sensors via Photoelectric DC Servo: Application to Condenser Microphones
- **分类: eess.AS**

- **简介: 该论文属于噪声抑制任务，旨在解决电容传感器自噪声限制灵敏度的问题。通过引入光电直流伺服电路，有效降低噪声并扩展带宽，提升麦克风性能。**

- **链接: [https://arxiv.org/pdf/2604.18969](https://arxiv.org/pdf/2604.18969)**

> **作者:** Hirotaka Obo; Atsushi Tsuchiya; Tadashi Ebihara; Naoto Wakatsuki
>
> **摘要:** The self-noise of capacitive sensors, primarily caused by thermal noise from the gate-bias resistor in the preamplifier, imposes a fundamental limit on measurement sensitivity. In electret condenser microphones (ECMs), this resistor simultaneously determines the noise low-pass cutoff frequency and the signal high-pass cutoff frequency through a single RC time constant, creating a trade-off between noise reduction and signal bandwidth. This paper proposes PDS-Amp (Photoelectric DC Servo Amplifier), a circuit technique that replaces the gate-bias resistor with a photoelectric element functioning as an ultra-high-impedance current source. A DC servo loop using lag-lead compensation feeds back the preamplifier output through an LED to control the photocurrent, thereby stabilizing the gate bias while decoupling the noise and signal cutoff frequencies. A custom photosensor based on the external photoelectric effect of a zinc photocathode was fabricated to achieve sub-picoampere dark current, overcoming the limitations of commercial semiconductor photodiodes. Combined with a cascode JFET preamplifier that minimizes input capacitance through bootstrap action, PDS-Amp achieved a self-noise of 11 dBA with a 12 pF dummy microphone. Despite using a small-diameter ECM capsule, this performance is comparable to that of large-diaphragm condenser microphones costing several thousand dollars. Recording experiments with an actual ECM capsule qualitatively confirmed a significant reduction in background noise. The proposed technique is applicable not only to microphones but broadly to capacitive sensors including accelerometers, pressure sensors, and pyroelectric sensors.
>
---
#### [new 003] Audio Spoof Detection with GaborNet
- **分类: cs.SD**

- **简介: 该论文属于音频伪造检测任务，旨在提升检测效果。通过引入GaborNet替代SincNet，结合RawNet2和RawGAT-ST架构进行改进，并研究音频增强方法。**

- **链接: [https://arxiv.org/pdf/2604.19209](https://arxiv.org/pdf/2604.19209)**

> **作者:** Waldek Maciejko
>
> **备注:** Industrial conference materials
>
> **摘要:** An direction of development in the extraction of features from audio signals is based on processing raw samples in the time domain. Such an approach appears to be effective, especially in the era of neural networks. An example is SincNet. In this solution, the core of the neural network layer is a set of sinc functions that are convolved with the input signal. Due to the finite length of sinc functions, distortions appear in the frequency domain of the convolved signal, the same as in the case of windowing the signal. Recently, a new approach has been developed that uses Gabor filters to replace sinc functions. Due to the complex results, further modifications had to be applied, such as squared modulus or Gaussian Lowpass Pooling. In this work, an ingestion layer based on a bank of Gabor filters, named GaborNet, and its modifications are intensively examined within the popular RawNet2 and RawGAT- ST architectures. These have been developed for the purpose of audio spoof detection. Another issue that has been investigated was audio augmentation using codec conversions, room responses, and additive noises.
>
---
#### [new 004] Comparison of sEMG Encoding Accuracy Across Speech Modes Using Articulatory and Phoneme Features
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音编码分析任务，旨在比较不同说话方式下SPARC特征与音素特征对sEMG的预测效果，验证SPARC在静默语音建模中的有效性。**

- **链接: [https://arxiv.org/pdf/2604.18920](https://arxiv.org/pdf/2604.18920)**

> **作者:** Chenqian Le; Ruisi Li; Beatrice Fumagalli; Xupeng Chen; Amirhossein Khalilian-Gourtani; Tianyu He; Adeen Flinker; Yao Wang
>
> **摘要:** We test whether Speech Articulatory Coding (SPARC) features can linearly predict surface electromyography (sEMG) envelopes across aloud, mimed, and subvocal speech in twenty-four subjects. Using elastic-net multivariate temporal response function (mTRF) with sentence-level cross-validation, SPARC yields higher prediction accuracy than phoneme one-hot representations on nearly all electrodes and in all speech modes. Aloud and mimed speech perform comparably, and subvocal speech remains above chance, indicating detectable articulatory activity. Variance partitioning shows a substantial unique contribution from SPARC and a minimal unique contribution from phoneme features. mTRF weight patterns reveal anatomically interpretable relationships between electrode sites and articulatory movements that remain consistent across modes. This study focuses on representation/encoding analysis (not end-to-end decoding) and supports SPARC as a robust and interpretable intermediate target for sEMG-based silent-speech modeling.
>
---
#### [new 005] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频语言模型中的幻觉检测任务，旨在解决音频领域幻觉问题研究不足的问题。工作包括构建首个大规模音频幻觉基准HalluAudio，并评估多种模型的幻觉情况。**

- **链接: [https://arxiv.org/pdf/2604.19300](https://arxiv.org/pdf/2604.19300)**

> **作者:** Feiyu Zhao; Yiming Chen; Wenhuan Lu; Daipeng Zhang; Xianghu Yue; Jianguo Wei
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Large Audio-Language Models (LALMs) have recently achieved strong performance across various audio-centric tasks. However, hallucination, where models generate responses that are semantically incorrect or acoustically unsupported, remains largely underexplored in the audio domain. Existing hallucination benchmarks mainly focus on text or vision, while the few audio-oriented studies are limited in scale, modality coverage, and diagnostic depth. We therefore introduce HalluAudio, the first large-scale benchmark for evaluating hallucinations across speech, environmental sound, and music. HalluAudio comprises over 5K human-verified QA pairs and spans diverse task types, including binary judgments, multi-choice reasoning, attribute verification, and open-ended QA. To systematically induce hallucinations, we design adversarial prompts and mixed-audio conditions. Beyond accuracy, our evaluation protocol measures hallucination rate, yes/no bias, error-type analysis, and refusal rate, enabling a fine-grained analysis of LALM failure modes. We benchmark a broad range of open-source and proprietary models, providing the first large-scale comparison across speech, sound, and music. Our results reveal significant deficiencies in acoustic grounding, temporal reasoning, and music attribute understanding, underscoring the need for reliable and robust LALMs.
>
---
#### [new 006] Towards Streaming Target Speaker Extraction via Chunk-wise Interleaved Splicing of Autoregressive Language Model
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于目标说话人提取任务，解决实时语音分离问题。通过引入分块交错拼接机制，提升模型在低延迟下的稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2604.19635](https://arxiv.org/pdf/2604.19635)**

> **作者:** Shuhai Peng; Hui Lu; Jinjiang Liu; Liyang Chen; Guiping Zhong; Jiakui Li; Huimeng Wang; Haiyun Li; Liang Cao; Shiyin Kang; Zhiyong Wu
>
> **摘要:** While generative models have set new benchmarks for Target Speaker Extraction (TSE), their inherent reliance on global context precludes deployment in real-time applications. Direct adaptation to streaming scenarios often leads to catastrophic inference performance degradation due to the severe mismatch between training and streaming inference. To bridge this gap, we present the first autoregressive (AR) models tailored for streaming TSE. Our approach introduces a Chunk-wise Interleaved Splicing Paradigm that ensures highly efficient and stable streaming inference. To ensure the coherence between the extracted speech segments, we design a historical context refinement mechanism that mitigates boundary discontinuities by leveraging historical information. Experiments on Libri2Mix show that while AR generative baseline exhibits performance degradation at low latencies, our approach maintains 100% stability and superior intelligibility. Furthermore, our streaming results are comparable to or even surpass offline baselines. Additionally, our model achieves a Real-Time-Factor (RTF) of 0.248 on consumer-level GPUs. This work provides empirical evidence that AR generative backbones are viable for latency-sensitive applications through the Chunk-wise Interleaved Splicing Paradigm.
>
---
#### [new 007] APRVOS: 1st Place Winner of 5th PVUW MeViS-Audio Track
- **分类: cs.SD**

- **简介: 该论文属于音频引导的视频目标分割任务，解决语音查询噪声及视觉不存在目标的问题，通过语音转文本、视觉验证、粗分割和代理优化等步骤提升分割精度。**

- **链接: [https://arxiv.org/pdf/2604.18665](https://arxiv.org/pdf/2604.18665)**

> **作者:** Deshui Miao; Yameng Gu; Chao Yang; Xin Li; Haijun Zhang; Ming-Hsuan Yang
>
> **摘要:** This report presents an Audio-aware Referring Video Object Segmentation (Ref-VOS) pipeline tailored to the MEVIS\_Audio setting, where the referring expression is provided in spoken form rather than as clean text. Compared with a standard Sa2VA-based Ref-VOS pipeline, the proposed system introduces two additional front-end stages: speech transcription and visual existence verification. Specifically, we first employ VibeVoice-ASR to convert long-form spoken input into a structured textual transcript. Since audio-derived queries are inherently noisy and may describe entities that are not visually present in the video, we then introduce an Omni-based judgment module to determine whether the transcribed target can be grounded in the visual content. If the target is judged to be absent, the pipeline terminates early and outputs all-zero masks. Otherwise, the transcript is transformed into a segmentation-oriented prompt and fed into Sa2VA to obtain a coarse mask trajectory over the full video. Importantly, this trajectory is treated as an initial semantic hypothesis rather than a final prediction. On top of it, an agentic refinement layer evaluates query reliability, temporal relevance, anchor quality, and potential error sources, and may invoke SAM3 to improve spatial boundary precision and temporal consistency. The resulting framework explicitly decomposes the MEVIS\_Audio task into audio-to-text conversion, visual existence verification, coarse video segmentation, and agent-guided refinement. Such a staged design is substantially more appropriate for audio-conditioned Ref-VOS than directly sending noisy ASR outputs into a segmentation model.
>
---
#### [new 008] Reducing the Offline-Streaming Gap for Unified ASR Transducer with Consistency Regularization
- **分类: eess.AS; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决统一模型在离线与实时流式设置中性能差距的问题。通过引入一致性正则化方法，提升流式识别准确率并保持离线性能。**

- **链接: [https://arxiv.org/pdf/2604.19079](https://arxiv.org/pdf/2604.19079)**

> **作者:** Andrei Andrusenko; Vladimir Bataev; Lilit Grigoryan; Nune Tadevosyan; Vitaly Lavrukhin; Boris Ginsburg
>
> **摘要:** Unification of automatic speech recognition (ASR) systems reduces development and maintenance costs, but training a single model to perform well in both offline and low-latency streaming settings remains challenging. We present a Unified ASR framework for Transducer (RNNT) training that supports both offline and streaming decoding within a single model, using chunk-limited attention with right context and dynamic chunked convolutions. To further close the gap between offline and streaming performance, we introduce an efficient Triton implementation of mode-consistency regularization for RNNT (MCR-RNNT), which encourages agreement across training modes. Experiments show that the proposed approach improves streaming accuracy at low latency while preserving offline performance and scaling to larger model sizes and training datasets. The proposed Unified ASR framework and the English model checkpoint are open-sourced.
>
---
#### [new 009] BEAT: Tokenizing and Generating Symbolic Music by Uniform Temporal Steps
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，旨在解决音乐符号表示中时间不统一的问题。提出以固定时长的节拍为基本单元进行编码，提升模型对音乐结构和长期模式的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2604.19532](https://arxiv.org/pdf/2604.19532)**

> **作者:** Lekai Qian; Haoyu Gu; Jingwei Zhao; Ziyu Wang
>
> **备注:** Preprint. 20 pages, 8 figures
>
> **摘要:** Tokenizing music to fit the general framework of language models is a compelling challenge, especially considering the diverse symbolic structures in which music can be represented (e.g., sequences, grids, and graphs). To date, most approaches tokenize symbolic music as sequences of musical events, such as onsets, pitches, time shifts, or compound note events. This strategy is intuitive and has proven effective in Transformer-based models, but it treats the regularity of musical time implicitly: individual tokens may span different durations, resulting in non-uniform time progression. In this paper, we instead consider whether an alternative tokenization is possible, where a uniform-length musical step (e.g., a beat) serves as the basic unit. Specifically, we encode all events within a single time step at the same pitch as one token, and group tokens explicitly by time step, which resembles a sparse encoding of a piano-roll representation. We evaluate the proposed tokenization on music continuation and accompaniment generation tasks, comparing it with mainstream event-based methods. Results show improved musical quality and structural coherence, while additional analyses confirm higher efficiency and more effective capture of long-range patterns with the proposed tokenization.
>
---
#### [new 010] ATRIE: Adaptive Tuning for Robust Inference and Emotion in Persona-Driven Speech Synthesis
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决角色语音在不同情绪下保持一致性格特征的问题。提出ATRIE框架，通过双轨架构实现稳定身份识别和丰富情感表达。**

- **链接: [https://arxiv.org/pdf/2604.19055](https://arxiv.org/pdf/2604.19055)**

> **作者:** Aoduo Li; Haoran Lv; Shengmin Li; Sihao Qin; Hongjian Xu
>
> **备注:** 10 pages, 6 figures. Accepted to ACM ICMR 2026
>
> **摘要:** High-fidelity character voice synthesis is a cornerstone of immersive multimedia applications, particularly for interacting with anime avatars and digital humans. However, existing systems struggle to maintain consistent persona traits across diverse emotional contexts. To bridge this gap, we present ATRIE, a unified framework utilizing a Persona-Prosody Dual-Track (P2-DT) architecture. Our system disentangles generation into a static Timbre Track (via Scalar Quantization) and a dynamic Prosody Track (via Hierarchical Flow-Matching), distilled from a 14B LLM teacher. This design enables robust identity preservation (Zero-Shot Speaker Verification EER: 0.04) and rich emotional expression. Evaluated on our extended AnimeTTS-Bench (50 characters), ATRIE achieves state-of-the-art performance in both generation and cross-modal retrieval (mAP: 0.75), establishing a new paradigm for persona-driven multimedia content creation.
>
---
#### [new 011] Tadabur: A Large-Scale Quran Audio Dataset
- **分类: cs.SD; cs.AI**

- **简介: 该论文介绍Tadabur，一个大规模的《古兰经》音频数据集，旨在解决现有数据集规模小、多样性不足的问题，通过收集超过1400小时的不同诵读者音频，支持 Quranic 语音研究。**

- **链接: [https://arxiv.org/pdf/2604.18932](https://arxiv.org/pdf/2604.18932)**

> **作者:** Faisal Alherran
>
> **备注:** Project page: this https URL
>
> **摘要:** Despite growing interest in Quranic data research, existing Quran datasets remain limited in both scale and diversity. To address this gap, we present Tadabur, a large-scale Quran audio dataset. Tadabur comprises more than 1400+ hours of recitation audio from over 600 distinct reciters, providing substantial variation in recitation styles, vocal characteristics, and recording conditions. This diversity makes Tadabur a comprehensive and representative resource for Quranic speech research and analysis. By significantly expanding both the total duration and variability of available Quran data, Tadabur aims to support future research and facilitate the development of standardized Quranic speech benchmarks.
>
---
#### [new 012] Towards Revised Tempo Indications for Beethoven's Piano and Cello Sonatas: Czerny, Moscheles, Kolisch, and Recorded Practice 1930-2012
- **分类: cs.SD**

- **简介: 该论文属于音乐学术研究任务，旨在解决贝多芬钢琴与大提琴奏鸣曲速度指示的准确性问题。通过分析历史标记与录音数据，提出基于实证的修订速度建议。**

- **链接: [https://arxiv.org/pdf/2604.18631](https://arxiv.org/pdf/2604.18631)**

> **作者:** Ignasi Sole
>
> **摘要:** Historical metronome indications for Beethoven's five piano and cello sonatas (as transmitted by Czerny, Moscheles, and Kolisch), have long been regarded as problematic by performers and scholars alike. This paper presents the first systematic empirical assessment of those indications against a corpus of over one hundred movement-level recordings spanning 1930--2012, encompassing first, second, and third movements across all five sonatas (Op.~5 Nos.~1 and~2; Op.~69; Op.~102 Nos.~1 and~2). The core findings are threefold. First, Czerny's and Moscheles's markings are consistently and substantially exceeded by the entire recording corpus: gaps of 15--39\% are documented across movements, with the largest divergences in slow Adagio movements and the smallest in fast Allegro finales. Second, Kolisch's 1943 markings align considerably more closely with recorded practice than either Czerny's or Moscheles's, a striking result given that Kolisch was reasoning without corpus data. Third, the central Allegro tempo traditions for each movement are stable across eight decades; not because all performers play alike, but because three coexisting slow, mid-range, and fast traditions persist simultaneously, with the mid-range dominant throughout. Building on these findings, this paper proposes a set of revised tempo indications grounded in the statistical modal tempi of the corpus, presented as ranges reflecting the documented spectrum of expert interpretive practice rather than single prescriptive values. These indications are offered not as claims about Beethoven's intentions but as evidence-based reference points for performers and scholars navigating the gap between historical prescription and performable reality.
>
---
#### [new 013] Text-To-Speech with Chain-of-Details: modeling temporal dynamics in speech generation
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成任务，解决传统方法参数多、自然度不足的问题。提出CoD框架，通过级联结构显式建模时间动态，提升语音自然度并减少参数量。**

- **链接: [https://arxiv.org/pdf/2604.19330](https://arxiv.org/pdf/2604.19330)**

> **作者:** Jianbo Ma; Richard Cartwright
>
> **摘要:** Recent advances in Text-To-Speech (TTS) synthesis have seen the popularity of multi-stage approaches that first predict semantic tokens and then generate acoustic tokens. In this paper, we extend the coarse-to-fine generation paradigm to the temporal domain and introduce Chain-of-Details (CoD), a novel framework that explicitly models temporal coarse-to-fine dynamics in speech generation using a cascaded architecture. Our method progressively refines temporal details across multiple stages, with each stage targeting a specific temporal granularity. All temporal detail predictions are performed using a shared decoder, enabling efficient parameter utilization across different temporal resolutions. Notably, we observe that the lowest detail level naturally performs phonetic planning without the need for an explicit phoneme duration predictor. We evaluate our method on several datasets and compare it against several baselines. Experimental results show that CoD achieves competitive performance with significantly fewer parameters than existing approaches. Our findings demonstrate that explicit modeling of temporal dynamics with the CoD framework leads to more natural speech synthesis.
>
---
#### [new 014] A Complementary Visualisation Suite for Empirical Performance Analysis: Tempographs, Histograms, Ridgeline Plots, Stacked Bar Charts, and Combination Charts Applied to Beethoven's Piano and Cello Sonatas
- **分类: cs.SD**

- **简介: 该论文属于音乐性能分析任务，旨在通过多种可视化工具揭示数据不同特征，解决单一图表信息局限问题，提出五种互补图表并应用于贝多芬钢琴与大提琴奏鸣曲分析。**

- **链接: [https://arxiv.org/pdf/2604.18630](https://arxiv.org/pdf/2604.18630)**

> **作者:** Ignasi Sole
>
> **摘要:** The choice of visualisation in empirical performance analysis is not a neutral presentation decision but an analytical one: different graphical forms reveal different features of the same dataset, and reliance on any single type systematically conceals what the others expose. This paper presents and argues for a suite of five complementary visualisation tools; tempographs, histograms with spline-smoothed probability density functions, ridgeline plots, stacked bar charts, and combination charts. These are applied to bar-level beats-per-minute data from recordings of Beethoven's five piano and cello sonatas (Op.~5 Nos.~1 and~2; Op.~69; Op.~102 Nos.~1 and~2) spanning 1930--2012. Each tool is described formally, its analytical properties characterised, its implementation detailed in working Python and MATLAB code, and its specific contribution demonstrated on a worked example using two recordings of Op.~5 No.~1 (Casals/Horszowski 1930--39 and Isserlis/Levin 2012) separated by eight decades. A five-panel composite figure applies all five tools to the same two recordings simultaneously, making the complementarity argument concrete: the tempograph reveals moment-to-moment structural parallels invisible in aggregate statistics; the spline-smoothed histogram exposes bimodality and secondary peaks suppressed by binning artefacts; the ridgeline plot positions both recordings within the full distributional space; the stacked bar chart shows divergent sectional pacing concealed by identical movement means; and the combination chart integrates mean tempo, variability, and historical reference marks in a single view. The spline-CDF smoothing method, applied to histogram data via cubic spline interpolation with zero-slope boundary conditions, is presented as a novel contribution to the performance analysis toolkit. Full implementation code is publicly available.
>
---
#### [new 015] Virtual boundary integral neural network for three-dimensional exterior acoustic problems
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出一种虚拟边界积分神经网络，用于解决三维外部声学问题。通过引入虚拟边界和神经网络表示源密度，避免了传统方法中的奇异积分问题，提高了计算稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2604.18636](https://arxiv.org/pdf/2604.18636)**

> **作者:** Jiahao Li; Qiang Xi; Ilia Marchevskiy; Zhuojia Fu
>
> **摘要:** This paper presents a virtual boundary integral neural network (VBINN) for exterior acoustic problems in three dimensions. The method introduces a virtual boundary inside the scatterer or vibrating body and represents the associated source density with a neural network. Coupled with the acoustic fundamental solution, this representation satisfies the Sommerfeld radiation condition by construction and enables direct evaluation of the acoustic pressure and its normal derivative at arbitrary field points. Because the integration surface is separated from the physical boundary, the formulation avoids the singular and near singular kernel evaluations associated with coincident source and collocation points in conventional boundary integral learning methods. To reduce sensitivity to boundary placement, the geometric parameters of the virtual boundary are optimized jointly with the source density during training. Numerical examples for acoustic scattering, multiple body interaction, and underwater acoustic propagation show close agreement with analytical solutions and COMSOL results, and the Burton Miller extension further improves stability near characteristic frequencies. These results demonstrate the potential of VBINN for exterior acoustic analysis in three dimensions.
>
---
#### [new 016] Environmental Sound Deepfake Detection Using Deep-Learning Framework
- **分类: cs.SD; cs.AI**

- **简介: 本文研究环境声音深度伪造检测任务，旨在区分音频中声音场景与事件的真实性。通过实验对比不同模型和方法，提出有效检测方案。**

- **链接: [https://arxiv.org/pdf/2604.19652](https://arxiv.org/pdf/2604.19652)**

> **作者:** Lam Pham; Khoi Vu; Dat Tran; Phat Lam; Vu Nguyen; David Fischinger; Alexander Schindler; Martin Boyer; Son Le
>
> **摘要:** In this paper, we propose a deep-learning framework for environmental sound deepfake detection (ESDD) -- the task of identifying whether the sound scene and sound event in an input audio recording is fake or not. To this end, we conducted extensive experiments to explore how individual spectrograms, a wide range of network architectures and pre-trained models, ensemble of spectrograms or network architectures affect the ESDD task performance. The experimental results on the benchmark datasets of EnvSDD and ESDD-Challenge-TestSet indicate that detecting deepfake audio of sound scene and detecting deepfake audio of sound event should be considered as individual tasks. We also indicate that the approach of finetuning a pre-trained model is more effective compared with training a model from scratch for the ESDD task. Eventually, our best model, which was finetuned from the pre-trained WavLM model with the proposed three-stage training strategy, achieve the Accuracy of 0.98, F1 Score of 0.95, AuC of 0.99 on EnvSDD Test subset and the Accuracy of 0.88, F1 Score of 0.77, and AuC of 0.92 on ESDD-Challenge-TestSet dataset.
>
---
#### [new 017] UAF: A Unified Audio Front-end LLM for Full-Duplex Speech Interaction
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出UAF模型，解决全双工语音交互中的前端任务整合问题。通过统一处理VAD、TD等任务，提升交互响应速度与准确性。**

- **链接: [https://arxiv.org/pdf/2604.19221](https://arxiv.org/pdf/2604.19221)**

> **作者:** Yadong Li; Guoxin Wu; Haiping Hou; Biye Li
>
> **摘要:** Full-duplex speech interaction, as the most natural and intuitive mode of human communication, is driving artificial intelligence toward more human-like conversational systems. Traditional cascaded speech processing pipelines suffer from critical limitations, including accumulated latency, information loss, and error propagation across modules. To address these issues, recent efforts focus on the end-to-end audio large language models (LLMs) like GPT-4o, which primarily unify speech understanding and generation task. However, most of these models are inherently half-duplex, and rely on a suite of separate, task-specific front-end components, such as voice activity detection (VAD) and turn-taking detection (TD). In our development of speech assistant, we observed that optimizing the speech front-end is equally crucial as advancing the back-end unified model for achieving seamless, responsive interactions. To bridge this gap, we propose the first unified audio front-end LLM (UAF) tailored for full-duplex speech systems. Our model reformulates diverse audio front-end tasks into a single auto-regressive sequence prediction problem, including VAD, TD, speaker recognition (SR), automatic speech recognition (ASR) and question answer (QA). It takes streaming fixed-duration audio chunk (e.g., 600 ms) as input, leverages a reference audio prompt to anchor the target speaker at the beginning, and regressively generates discrete tokens encoding both semantic content and system-level state controls (e.g., interruption signals). Experiments demonstrate that our model achieves leading performance across multiple audio front-end tasks and significantly enhances response latency and interruption accuracy in real-world interaction scenarios.
>
---
#### [new 018] Voice of India: A Large-Scale Benchmark for Real-World Speech Recognition in India
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决印度语种真实场景下语音识别的基准不足问题。构建了包含15种语言的大规模语音数据集，分析了多种影响因素，提升实际应用中的识别效果。**

- **链接: [https://arxiv.org/pdf/2604.19151](https://arxiv.org/pdf/2604.19151)**

> **作者:** Kaushal Bhogale; Manas Dhir; Amritansh Walecha; Manmeet Kaur; Vanshika Chhabra; Aaditya Pareek; Hanuman Sidh; Sagar Jain; Bhaskar Singh; Utkarsh Singh; Tahir Javed; Shobhit Banga; Mitesh M. Khapra
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Existing Indic ASR benchmarks often use scripted, clean speech and leaderboard driven evaluation that encourages dataset specific overfitting. In addition, strict single reference WER penalizes natural spelling variation in Indian languages, including non standardized spellings of code-mixed English origin words. To address these limitations, we introduce Voice of India, a closed source benchmark built from unscripted telephonic conversations covering 15 major Indian languages across 139 regional clusters. The dataset contains 306230 utterances, totaling 536 hours of speech from 36691 speakers with transcripts accounting for spelling variations. We also analyze performance geographically at the district level, revealing disparities. Finally, we provide detailed analysis across factors such as audio quality, speaking rate, gender, and device type, highlighting where current ASR systems struggle and offering insights for improving real world Indic ASR systems.
>
---
#### [new 019] Hybrid SMI Realization via Matrix Completion and Riemannian Manifold Optimization on Narrowband Sub-Array Based Architectures
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于信号处理任务，解决混合波束成形中因硬件限制导致的协方差矩阵不完整问题。通过矩阵补全和黎曼优化方法，重建协方差矩阵，提升混合SMI性能。**

- **链接: [https://arxiv.org/pdf/2604.18748](https://arxiv.org/pdf/2604.18748)**

> **作者:** Tarun Suman Cousik; Rohit Rangaraj; Nishith Tripathi; Jeffrey H Reed; Daniel Jakubisin; Jon Kraft
>
> **备注:** Accepted in 2026 IEEE AESS RadarConf
>
> **摘要:** Hybrid beamforming architectures reduce hardware complexity but restrict access to full array observations, rendering direct implementation of classical covariance based methods such as minimum variance distortionless response (MVDR) and sample matrix inversion (SMI) infeasible. This work introduces a structured covariance completion framework, termed Rock Road to Dublin (RR2D), which estimates the unobservable analytical covariance matrix (ACM) from a partially observed sample covariance matrix (SCM). RR2D exploits signal stationarity across the array and enforces physical measurement consistency using Dykstra's alternating projection algorithm with positive semidefinite, Toeplitz, and block constraints. The reconstructed virtual ACM enables a realizable hybrid SMI (HSMI) formulation that remains fully compatible with existing hybrid MVDR optimization frameworks. Empirical results for a 32 element hybrid array demonstrate both the expected degradation of HSMI implemented directly under prior HMVDR formulations and the performance gains achieved through RR2D. The proposed HSMI consistently outperforms previous hybrid SMI and partial digital baselines, achieving performance close to the HMVDR reference. Overall, RR2D bridges the gap between theoretical HMVDR formulations and practical hybrid hardware by enabling structured covariance reconstruction from incomplete observations.
>
---
## 更新

#### [replaced 001] DASB - Discrete Audio and Speech Benchmark
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频处理任务，旨在解决离散音频表示的优化问题。通过构建DASB基准，评估不同配置下的离散音频令牌性能，探索其在语音、音乐等领域的适用性。**

- **链接: [https://arxiv.org/pdf/2406.14294](https://arxiv.org/pdf/2406.14294)**

> **作者:** Pooneh Mousavi; Jarod Duret; Darius Petermann; Artem Ploujnikov; Luca Della Libera; Anastasia Kuznetsova; Cem Subakan; Mirco Ravanelli
>
> **摘要:** Discrete audio tokens have recently gained considerable attention for their potential to bridge audio and language processing, enabling multimodal language models that can both generate and understand audio. However, preserving key information such as phonetic content, speaker identity, and paralinguistic cues remains a major challenge. Identifying the optimal tokenizer and configuration is further complicated by inconsistent evaluation settings across existing studies. To address this, we introduce the Discrete Audio and Speech Benchmark (DASB), a comprehensive framework for benchmarking discrete audio tokens across speech, general audio, and music domains on a range of discriminative and generative tasks. Our results show that discrete representations are less robust than continuous ones and require careful tuning of factors such as model architecture, data size, learning rate, and capacity. Semantic tokens generally outperform acoustic tokens, but a gap remains between discrete tokens and continuous features, highlighting the need for further research. DASB codes, evaluation setup, and leaderboards are publicly available at this https URL.
>
---
#### [replaced 002] Computational Narrative Understanding for Expressive Text-to-Speech
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于文本到语音合成任务，旨在提升语音表达性。通过构建包含5.3K小时表达性语音的LibriQuote数据集，研究发现微调模型可显著提高语音的表达性和可懂度。**

- **链接: [https://arxiv.org/pdf/2509.04072](https://arxiv.org/pdf/2509.04072)**

> **作者:** Gaspard Michel; Elena V. Epure; Christophe Cerisara
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Recent advances in text-to-speech (TTS) have been driven by large, multi-domain speech corpora, yet the expressive potential of audiobook data remains underexamined. We argue that human-narrated audiobooks, particularly fictional works, contain rich and diverse prosodic cues arising from the natural alternation between neutral narration and expressive character dialogue. Building from this observation, we introduce LibriQuote, a large-scale 5.3K hours of expressive speech drawn from character quotations. Each quote is supplemented with contextual pseudo-labels for speech verbs and adverbs that characterize the intended delivery of direct speech (e.g., "he whispered softly"). We found that fine-tuning a flow-matching model on LibriQuote yields substantial improvements in expressivity and intelligibility, while training from scratch enhances expressiveness of an autoregressive TTS model. Benchmarking on LibriQuote-test highlights significant variability across systems in generating expressive speech. We publicly release the dataset, code, and evaluation resources to facilitate reproducibility. Audio samples can be found at this https URL.
>
---
#### [replaced 003] Protecting Bystander Privacy via Selective Hearing in Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频大模型隐私保护任务，旨在解决无意中泄露旁观者语音隐私的问题。提出SH-Bench基准和BPFT训练方法，提升模型在保持主讲人理解的同时保护旁观者隐私。**

- **链接: [https://arxiv.org/pdf/2512.06380](https://arxiv.org/pdf/2512.06380)**

> **作者:** Xiao Zhan; Guangzhi Sun; Jose Such; Phil Woodland
>
> **备注:** To Appear at ACL 2026 main conference; Dataset: this https URL
>
> **摘要:** Audio Large language models (LLMs) are increasingly deployed in the real world, where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences did not consider. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures, including both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. In addition, we propose Selective Efficacy (SE), a novel metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LLMs reveals substantial bystander privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we also present Bystander Privacy Fine-Tuning (BPFT), a novel training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. We show that BPFT yields substantial gains, achieving an absolute 47% higher bystander accuracy under selective mode and an absolute 16% higher SE compared to Gemini 2.5 Pro, which is the best audio LLM without BPFT. Together, SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio LLMs.
>
---
#### [replaced 004] Real-Time Streamable Generative Speech Restoration with Flow Matching
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决实时语音生成的延迟问题。提出Stream.FM模型，实现低延迟的流式语音恢复，支持多种语音处理任务。**

- **链接: [https://arxiv.org/pdf/2512.19442](https://arxiv.org/pdf/2512.19442)**

> **作者:** Simon Welker; Bunlong Lay; Maris Hillemann; Tal Peer; Timo Gerkmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Diffusion-based generative models have greatly impacted the speech processing field in recent years, exhibiting high speech naturalness and spawning a new research direction. Their application in real-time communication is, however, still lagging behind due to their computation-heavy nature involving multiple calls of large DNNs. Here, we present Stream$.$FM, a frame-causal flow-based generative model with an algorithmic latency of 32 milliseconds (ms) and a total latency of 48 ms, paving the way for generative speech processing in real-time communication. We propose a buffered streaming inference scheme and an optimized DNN architecture, show how learned few-step numerical solvers can boost output quality at a fixed compute budget, explore model weight compression to find favorable points along a compute/quality tradeoff, and contribute a model variant with 24 ms total latency for the speech enhancement task. Our work looks beyond theoretical latencies, showing that high-quality streaming generative speech processing can be realized on consumer GPUs available today. Stream$.$FM can solve a variety of speech processing tasks in a streaming fashion: speech enhancement, dereverberation, codec post-filtering, bandwidth extension, STFT phase retrieval, and Mel vocoding. As we verify through comprehensive evaluations and a MUSHRA listening test, Stream$.$FM establishes a state-of-the-art for generative streaming speech restoration, exhibits only a reasonable reduction in quality compared to a non-streaming variant, and outperforms our recent work (Diffusion Buffer) on generative streaming speech enhancement while operating at a lower latency.
>
---
#### [replaced 005] OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出OmniVoice，解决多语言零样本文本到语音问题。采用扩散语言模型架构，直接将文本映射到音频标记，提升效率与覆盖范围。**

- **链接: [https://arxiv.org/pdf/2604.00688](https://arxiv.org/pdf/2604.00688)**

> **作者:** Han Zhu; Lingxuan Ye; Wei Kang; Zengwei Yao; Liyong Guo; Fangjun Kuang; Zhifeng Han; Weiji Zhuang; Long Lin; Daniel Povey
>
> **摘要:** We present OmniVoice, a massively multilingual zero-shot text-to-speech (TTS) model that scales to over 600 languages. At its core is a novel diffusion language model-style discrete non-autoregressive (NAR) architecture. Unlike conventional discrete NAR models that suffer from performance bottlenecks in complex two-stage (text-to-semantic-to-acoustic) pipelines, OmniVoice directly maps text to multi-codebook acoustic tokens. This simplified approach is facilitated by two key technical innovations: (1) a full-codebook random masking strategy for efficient training, and (2) initialization from a pre-trained LLM to ensure superior intelligibility. By leveraging a 581k-hour multilingual dataset curated entirely from open-source data, OmniVoice achieves the broadest language coverage to date and delivers state-of-the-art performance across Chinese, English, and diverse multilingual benchmarks. Our code and pre-trained models are publicly available at this https URL.
>
---
#### [replaced 006] NVBench: A Benchmark for Speech Synthesis with Non-Verbal Vocalizations
- **分类: cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决非语言发声评估标准不足的问题。提出NVBench基准，评估系统生成非语言发声的准确性、位置和显著性。**

- **链接: [https://arxiv.org/pdf/2604.16211](https://arxiv.org/pdf/2604.16211)**

> **作者:** Liumeng Xue; Weizhen Bian; Jiahao Pan; Wenxuan Wang; Yilin Ren; Boyi Kang; Jingbin Hu; Ziyang Ma; Shuai Wang; Xinyuan Qian; Hung-yi Lee; Yike Guo
>
> **摘要:** Non-verbal vocalizations (NVVs) like laugh, sigh, and sob are essential for human-like speech, yet standardized evaluation remains limited in jointly assessing whether systems can generate the intended NVVs, place them correctly, and keep them salient without harming speech. We present Non-verbal Vocalization Benchmark (NVBench), a bilingual (English/Chinese) benchmark that evaluates speech synthesis with NVVs. NVBench pairs a unified 45-type taxonomy with a curated bilingual dataset and introduces a multi-axis protocol that separates general speech naturalness and quality from NVV-specific controllability, placement, and salience. We benchmark 15 TTS systems using objective metrics, listening tests, and an LLM-based multi-rater evaluation. Results reveal that NVVs controllability often decouples from quality, while low-SNR oral cues and long-duration affective NVVs remain persistent bottlenecks. NVBench enables fair cross-system comparison across diverse control interfaces under a unified, standardized framework.
>
---
#### [replaced 007] Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations
- **分类: cs.SD**

- **简介: 该论文属于情感语音合成任务，旨在解决非语言发声（如笑、叹气）在开放场景中数据不足且缺乏监督的问题。提出Affectron框架，增强NV生成的多样性与上下文一致性。**

- **链接: [https://arxiv.org/pdf/2603.14432](https://arxiv.org/pdf/2603.14432)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Nonverbal vocalizations (NVs), such as laughter and sighs, are central to the expression of affective cues in emotional speech synthesis. However, learning diverse and contextually aligned NVs remains challenging in open settings due to limited NV data and the lack of explicit supervision. Motivated by this challenge, we propose Affectron as a framework for affective and contextually aligned NV generation. Built on a small-scale open and decoupled corpus, Affectron introduces an NV-augmented training strategy that expands the distribution of NV types and insertion locations. We further incorporate NV structural masking into a speech backbone pre-trained on purely verbal speech to enable diverse and natural NV synthesis. Experimental results demonstrate that Affectron produces more expressive and diverse NVs than baseline systems while preserving the naturalness of the verbal speech stream.
>
---
#### [replaced 008] Speculative End-Turn Detector for Efficient Speech Chatbot Assistant
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决端轮检测问题，提出SpeculativeETD框架，结合轻量与高性能模型提升检测效率与准确率。**

- **链接: [https://arxiv.org/pdf/2503.23439](https://arxiv.org/pdf/2503.23439)**

> **作者:** Hyunjong Ok; Suho Yoo; Jaeho Lee
>
> **备注:** ACL 2026
>
> **摘要:** Spoken dialogue systems powered by large language models have demonstrated remarkable abilities in understanding human speech and generating appropriate spoken responses. However, these systems struggle with end-turn detection (ETD) -- the ability to distinguish between user turn completion and hesitation. This limitation often leads to premature or delayed responses, disrupting the flow of spoken conversations. In this paper, we introduce the ETD Dataset, the first public dataset for end-turn detection. The ETD dataset consists of both synthetic speech data generated with text-to-speech models and real-world speech data collected from web sources. We also propose SpeculativeETD, a novel collaborative inference framework that balances efficiency and accuracy to improve real-time ETD in resource-constrained environments. Our approach jointly employs a lightweight GRU-based model, which rapidly detects the non-speaking units in real-time on local devices, and a high-performance Wav2vec-based model running on the server to make a more challenging classification of distinguishing turn ends from mere pauses. Experiments demonstrate that the proposed SpeculativeETD significantly improves ETD accuracy while keeping the required computations low. Datasets and code will be available after the review.
>
---
#### [replaced 009] Qwen3.5-Omni Technical Report
- **分类: cs.CL; eess.AS**

- **简介: 该论文介绍Qwen3.5-Omni，解决多模态理解和交互问题，通过大规模数据训练，提升音频视频处理能力，并引入ARIA优化语音合成。**

- **链接: [https://arxiv.org/pdf/2604.15804](https://arxiv.org/pdf/2604.15804)**

> **作者:** Qwen Team
>
> **摘要:** In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis, often caused by encoding efficiency discrepancies between text and speech tokenizers, we introduce ARIA. ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding.
>
---
