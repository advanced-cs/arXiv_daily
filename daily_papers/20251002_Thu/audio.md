# 音频 cs.SD;  eess.SP

- **最新发布 32 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Dereverberation Using Binary Residual Masking with Time-Domain Consistency
- **分类: cs.SD; eess.AS; I.2.m; I.5.1**

- **简介: 该论文属于语音去混响任务，旨在提升实时音频处理中语音清晰度。通过残差掩码和时域一致性优化，实现高效准确的去混响效果。**

- **链接: [http://arxiv.org/pdf/2510.00356v1](http://arxiv.org/pdf/2510.00356v1)**

> **作者:** Daniel G. Williams
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Vocal dereverberation remains a challenging task in audio processing, particularly for real-time applications where both accuracy and efficiency are crucial. Traditional deep learning approaches often struggle to suppress reverberation without degrading vocal clarity, while recent methods that jointly predict magnitude and phase have significant computational cost. We propose a real-time dereverberation framework based on residual mask prediction in the short-time Fourier transform (STFT) domain. A U-Net architecture is trained to estimate a residual reverberation mask that suppresses late reflections while preserving direct speech components. A hybrid objective combining binary cross-entropy, residual magnitude reconstruction, and time-domain consistency further encourages both accurate suppression and perceptual quality. Together, these components enable low-latency dereverberation suitable for real-world speech and singing applications.
>
---
#### [new 002] When Silence Matters: The Impact of Irrelevant Audio on Text Reasoning in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于多模态AI任务，研究无关音频对文本推理的影响，发现噪声和静音会降低模型准确性，提出自一致性方法提升稳定性。**

- **链接: [http://arxiv.org/pdf/2510.00626v1](http://arxiv.org/pdf/2510.00626v1)**

> **作者:** Chen-An Li; Tzu-Han Lin; Hung-yi Lee
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) unify speech and text processing, but their robustness in noisy real-world settings remains underexplored. We investigate how irrelevant audio, such as silence, synthetic noise, and environmental sounds, affects text reasoning tasks where audio is unnecessary. Across three text-based benchmarks, we find that even non-informative audio reduces accuracy and increases prediction volatility; the severity of interference scales with longer durations, higher amplitudes, and elevated decoding temperatures. Silence, often assumed neutral, destabilizes outputs as strongly as synthetic noise. While larger models show greater resilience, vulnerabilities persist across all evaluated systems. We further test mitigation strategies and find that prompting shows limited effectiveness, whereas self-consistency improves stability at the cost of increased computation. Our results reveal cross-modal interference as a key robustness challenge and highlight the need for efficient fusion strategies that preserve reasoning performance in the presence of irrelevant inputs.
>
---
#### [new 003] Temporal-Aware Iterative Speech Model for Dementia Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于 dementia detection 任务，旨在解决传统方法忽视语音动态时间特征的问题。提出 TAI-Speech 模型，通过时序建模和语音特征对齐提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.00030v1](http://arxiv.org/pdf/2510.00030v1)**

> **作者:** Chukwuemeka Ugwu; Oluwafemi Oyeleke
>
> **摘要:** Deep learning systems often struggle with processing long sequences, where computational complexity can become a bottleneck. Current methods for automated dementia detection using speech frequently rely on static, time-agnostic features or aggregated linguistic content, lacking the flexibility to model the subtle, progressive deterioration inherent in speech production. These approaches often miss the dynamic temporal patterns that are critical early indicators of cognitive decline. In this paper, we introduce TAI-Speech, a Temporal Aware Iterative framework that dynamically models spontaneous speech for dementia detection. The flexibility of our method is demonstrated through two key innovations: 1) Optical Flow-inspired Iterative Refinement: By treating spectrograms as sequential frames, this component uses a convolutional GRU to capture the fine-grained, frame-to-frame evolution of acoustic features. 2) Cross-Attention Based Prosodic Alignment: This component dynamically aligns spectral features with prosodic patterns, such as pitch and pauses, to create a richer representation of speech production deficits linked to functional decline (IADL). TAI-Speech adaptively models the temporal evolution of each utterance, enhancing the detection of cognitive markers. Experimental results on the DementiaBank dataset show that TAI-Speech achieves a strong AUC of 0.839 and 80.6\% accuracy, outperforming text-based baselines without relying on ASR. Our work provides a more flexible and robust solution for automated cognitive assessment, operating directly on the dynamics of raw audio.
>
---
#### [new 004] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates
- **分类: cs.SD**

- **简介: 该论文属于语音编码任务，解决低帧率下语义信息丢失问题。提出FlexiCodec，通过动态帧率和双流结构提升语义保留与音频质量。**

- **链接: [http://arxiv.org/pdf/2510.00981v1](http://arxiv.org/pdf/2510.00981v1)**

> **作者:** Jiaqi Li; Yao Qian; Yuxuan Hu; Leying Zhang; Xiaofei Wang; Heng Lu; Manthan Thakker; Jinyu Li; Shang Zhao; Zhizheng Wu
>
> **摘要:** Neural audio codecs are foundational to speech language models. It is expected to have a low frame rate and decoupled semantic and acoustic information. A lower frame rate codec can reduce the computational cost of speech language models by shortening the sequence length. Recent studies have developed 12.5Hz low-frame-rate audio codecs, but even lower frame rate codecs remain underexplored. We find that a major challenge for very low frame rate tokens is missing semantic information. This paper introduces FlexiCodec to address this limitation. FlexiCodec improves semantic preservation with a dynamic frame rate approach and introduces a novel architecture featuring an ASR feature-assisted dual stream encoding and Transformer bottlenecks. With dynamic frame rates, it uses less frames at information-sparse regions through adaptively merging semantically similar frames. A dynamic frame rate also allows FlexiCodec to support inference-time controllable frame rates between 3Hz and 12.5Hz. Experiments on 6.25Hz, 8.3Hz and 12.5Hz average frame rates confirm that FlexiCodec excels over baseline systems in semantic information preservation and delivers a high audio reconstruction quality. We also validate the effectiveness of FlexiCodec in language model-based TTS. Demos are available at: https://flexicodec.github.io
>
---
#### [new 005] NLDSI-BWE: Non Linear Dynamical Systems-Inspired Multi Resolution Discriminators for Speech Bandwidth Extension
- **分类: cs.SD**

- **简介: 该论文属于语音带宽扩展任务，旨在通过模拟语音的非线性混沌特性，设计更高效的判别器以减少参数量并提升性能。**

- **链接: [http://arxiv.org/pdf/2510.01109v1](http://arxiv.org/pdf/2510.01109v1)**

> **作者:** Tarikul Islam Tamiti; Anomadarshi Barua
>
> **摘要:** In this paper, we design two nonlinear dynamical systems-inspired discriminators -- the Multi-Scale Recurrence Discriminator (MSRD) and the Multi-Resolution Lyapunov Discriminator (MRLD) -- to \textit{explicitly} model the inherent deterministic chaos of speech. MSRD is designed based on Recurrence representations to capture self-similarity dynamics. MRLD is designed based on Lyapunov exponents to capture nonlinear fluctuations and sensitivity to initial conditions. Through extensive design optimization and the use of depthwise-separable convolutions in the discriminators, our framework surpasses prior AP-BWE model with a 44x reduction in the discriminator parameter count \textbf{($\sim$ 22M vs $\sim$ 0.48M)}. To the best of our knowledge, for the first time, this paper demonstrates how BWE can be supervised by the subtle non-linear chaotic physics of voiced sound production to achieve a significant reduction in the discriminator size.
>
---
#### [new 006] From Scores to Preferences: Redefining MOS Benchmarking for Speech Quality Reward Modeling
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决传统MOS评分的主观性和不一致性问题。通过构建MOS-RMBench基准，提出多种奖励模型，提升自动语音质量评估的准确性与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.00743v1](http://arxiv.org/pdf/2510.00743v1)**

> **作者:** Yifei Cao; Changhao Jiang; Jiabao Zhuang; Jiajun Sun; Ming Zhang; Zhiheng Xi; Hui Li; Shihan Dou; Yuran Wang; Yunke Zhang; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Assessing the perceptual quality of synthetic speech is crucial for guiding the development and refinement of speech generation models. However, it has traditionally relied on human subjective ratings such as the Mean Opinion Score (MOS), which depend on manual annotations and often suffer from inconsistent rating standards and poor reproducibility. To address these limitations, we introduce MOS-RMBench, a unified benchmark that reformulates diverse MOS datasets into a preference-comparison setting, enabling rigorous evaluation across different datasets. Building on MOS-RMBench, we systematically construct and evaluate three paradigms for reward modeling: scalar reward models, semi-scalar reward models, and generative reward models (GRMs). Our experiments reveal three key findings: (1) scalar models achieve the strongest overall performance, consistently exceeding 74% accuracy; (2) most models perform considerably worse on synthetic speech than on human speech; and (3) all models struggle on pairs with very small MOS differences. To improve performance on these challenging pairs, we propose a MOS-aware GRM that incorporates an MOS-difference-based reward function, enabling the model to adaptively scale rewards according to the difficulty of each sample pair. Experimental results show that the MOS-aware GRM significantly improves fine-grained quality discrimination and narrows the gap with scalar models on the most challenging cases. We hope this work will establish both a benchmark and a methodological framework to foster more rigorous and scalable research in automatic speech quality assessment.
>
---
#### [new 007] XPPG-PCA: Reference-free automatic speech severity evaluation with principal components
- **分类: cs.SD**

- **简介: 该论文属于语音病理严重程度评估任务，旨在解决依赖专家评估的主观性与低效问题。提出XPPG-PCA方法，实现无参考的自动评估。**

- **链接: [http://arxiv.org/pdf/2510.00657v1](http://arxiv.org/pdf/2510.00657v1)**

> **作者:** Bence Mark Halpern; Thomas B. Tienkamp; Teja Rebernik; Rob J. J. H. van Son; Sebastiaan A. H. J. de Visscher; Max J. H. Witjes; Defne Abur; Tomoki Toda
>
> **备注:** 14 pages, 4 figures. Author Accepted Manuscript version of the IEEE Selected Topics in Signal Processing with the same title
>
> **摘要:** Reliably evaluating the severity of a speech pathology is crucial in healthcare. However, the current reliance on expert evaluations by speech-language pathologists presents several challenges: while their assessments are highly skilled, they are also subjective, time-consuming, and costly, which can limit the reproducibility of clinical studies and place a strain on healthcare resources. While automated methods exist, they have significant drawbacks. Reference-based approaches require transcriptions or healthy speech samples, restricting them to read speech and limiting their applicability. Existing reference-free methods are also flawed; supervised models often learn spurious shortcuts from data, while handcrafted features are often unreliable and restricted to specific speech tasks. This paper introduces XPPG-PCA (x-vector phonetic posteriorgram principal component analysis), a novel, unsupervised, reference-free method for speech severity evaluation. Using three Dutch oral cancer datasets, we demonstrate that XPPG-PCA performs comparably to, or exceeds established reference-based methods. Our experiments confirm its robustness against data shortcuts and noise, showing its potential for real-world clinical use. Taken together, our results show that XPPG-PCA provides a robust, generalizable solution for the objective assessment of speech pathology, with the potential to significantly improve the efficiency and reliability of clinical evaluations across a range of disorders. An open-source implementation is available.
>
---
#### [new 008] PodEval: A Multimodal Evaluation Framework for Podcast Audio Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音生成评估任务，解决长文本音频生成无标准评价的问题。提出PodEval框架，结合多模态评估方法，涵盖内容与格式，提升生成质量评估准确性。**

- **链接: [http://arxiv.org/pdf/2510.00485v1](http://arxiv.org/pdf/2510.00485v1)**

> **作者:** Yujia Xiao; Liumeng Xue; Lei He; Xinyi Chen; Aemon Yat Fei Chiu; Wenjie Tian; Shaofei Zhang; Qiuqiang Kong; Xinfa Zhu; Wei Xue; Tan Lee
>
> **摘要:** Recently, an increasing number of multimodal (text and audio) benchmarks have emerged, primarily focusing on evaluating models' understanding capability. However, exploration into assessing generative capabilities remains limited, especially for open-ended long-form content generation. Significant challenges lie in no reference standard answer, no unified evaluation metrics and uncontrollable human judgments. In this work, we take podcast-like audio generation as a starting point and propose PodEval, a comprehensive and well-designed open-source evaluation framework. In this framework: 1) We construct a real-world podcast dataset spanning diverse topics, serving as a reference for human-level creative quality. 2) We introduce a multimodal evaluation strategy and decompose the complex task into three dimensions: text, speech and audio, with different evaluation emphasis on "Content" and "Format". 3) For each modality, we design corresponding evaluation methods, involving both objective metrics and subjective listening test. We leverage representative podcast generation systems (including open-source, close-source, and human-made) in our experiments. The results offer in-depth analysis and insights into podcast generation, demonstrating the effectiveness of PodEval in evaluating open-ended long-form audio. This project is open-source to facilitate public use: https://github.com/yujxx/PodEval.
>
---
#### [new 009] Low Resource Audio Codec Challenge Baseline Systems
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于低资源音频编码任务，旨在提升受限环境下的语音编码性能。解决噪声和混响下的透明传输与增强问题，提出基于卷积神经网络的基线系统。**

- **链接: [http://arxiv.org/pdf/2510.00264v1](http://arxiv.org/pdf/2510.00264v1)**

> **作者:** Yusuf Ziya Isik; Rafał Łaganowski
>
> **备注:** Low-Resource Audio Codec Challenge 2025
>
> **摘要:** The Low-Resource Audio Codec (LRAC) Challenge aims to advance neural audio coding for deployment in resource-constrained environments. The first edition focuses on low-resource neural speech codecs that must operate reliably under everyday noise and reverberation, while satisfying strict constraints on computational complexity, latency, and bitrate. Track 1 targets transparency codecs, which aim to preserve the perceptual transparency of input speech under mild noise and reverberation. Track 2 addresses enhancement codecs, which combine coding and compression with denoising and dereverberation. This paper presents the official baseline systems for both tracks in the 2025 LRAC Challenge. The baselines are convolutional neural codec models with Residual Vector Quantization, trained end-to-end using a combination of adversarial and reconstruction objectives. We detail the data filtering and augmentation strategies, model architectures, optimization procedures, and checkpoint selection criteria.
>
---
#### [new 010] A Robust Proactive Communication Strategy for Distributed Active Noise Control Systems
- **分类: eess.SP; eess.AS**

- **简介: 该论文属于分布式主动降噪任务，解决通信开销导致的系统不稳定问题，提出一种结合自适应与固定滤波器的主动通信策略。**

- **链接: [http://arxiv.org/pdf/2510.00934v1](http://arxiv.org/pdf/2510.00934v1)**

> **作者:** Junwei Ji; Dongyuan Shi; Zhengding Luo; Boxiang Wang; Ziyi Yang; Haowen Li; Woon-Seng Gan
>
> **摘要:** Distributed multichannel active noise control (DMCANC) systems assign the high computational load of conventional centralized algorithms across multiple processing nodes, leveraging inter-node communication to collaboratively suppress unwanted noise. However, communication overhead can undermine algorithmic stability and degrade overall performance. To address this challenge, we propose a robust communication framework that integrates adaptive-fixed-filter switching and the mixed-gradient combination strategy. In this approach, each node independently executes a single-channel filtered reference least mean square (FxLMS) algorithm while monitoring real-time noise reduction levels. When the current noise reduction performance degrades compared to the previous state, the node halts its adaptive algorithm, switches to a fixed filter, and simultaneously initiates a communication request. The exchanged information comprises the difference between the current control filter and the filter at the time of the last communication, equivalent to the accumulated gradient sum during non-communication intervals. Upon receiving neighboring cumulative gradients, the node employs a mixed-gradient combination method to update its control filter, subsequently reverting to the adaptive mode. This proactive communication strategy and adaptive-fixed switching mechanism ensure system robustness by mitigating instability risks caused by communication issues. Simulations demonstrate that the proposed method achieves noise reduction performance comparable to centralized algorithms while maintaining stability under communication constraints, highlighting its practical applicability in real-world distributed ANC scenarios.
>
---
#### [new 011] HVAC-EAR: Eavesdropping Human Speech Using HVAC Systems
- **分类: cs.SD; cs.CR**

- **简介: 该论文属于语音重建任务，解决通过HVAC系统压力数据窃听语音的问题。工作包括使用复数变换器模型和处理噪声方法，实现低采样率下的清晰语音恢复。**

- **链接: [http://arxiv.org/pdf/2510.01082v1](http://arxiv.org/pdf/2510.01082v1)**

> **作者:** Tarikul Islam Tamiti; Biraj Joshi; Rida Hasan; Anomadarshi Barua
>
> **摘要:** Pressure sensors are widely integrated into modern Heating, Ventilation and Air Conditioning (HVAC) systems. As they are sensitive to acoustic pressure, they can be a source of eavesdropping. This paper introduces HVAC-EAR, which reconstructs intelligible speech from low-resolution, noisy pressure data with two key contributions: (i) We achieve intelligible reconstruction from as low as 0.5 kHz sampling rate, surpassing prior work limited to hot word detection, by employing a complex-valued conformer with a Complex Unified Attention Block to capture phoneme dependencies; (ii) HVAC-EAR mitigates transient HVAC noise by reconstructing both magnitude and phase of missing frequencies. For the first time, evaluations on real-world HVAC deployments show significant intelligibility, raising novel privacy concerns.
>
---
#### [new 012] Unpacking Musical Symbolism in Online Communities: Content-Based and Network-Centric Approaches
- **分类: cs.SD; cs.CL; cs.CY; cs.MM; eess.AS**

- **简介: 该论文研究在线社区中音乐符号的生成与传播，结合音乐分析与网络视角，分析歌词和音频特征，揭示音乐风格与情绪的关系。**

- **链接: [http://arxiv.org/pdf/2510.00006v1](http://arxiv.org/pdf/2510.00006v1)**

> **作者:** Kajwan Ziaoddini
>
> **摘要:** This paper examines how musical symbolism is produced and circulated in online communities by combining content-based music analysis with a lightweight network perspective on lyrics. Using a curated corpus of 275 chart-topping songs enriched with audio descriptors (energy, danceability, loudness, liveness, valence, acousticness, speechiness, popularity) and full lyric transcripts, we build a reproducible pipeline that (i) quantifies temporal trends in sonic attributes, (ii) models lexical salience and co-occurrence, and (iii) profiles mood by genre. We find a decade-long decline in energy (79 -> 58) alongside a rise in danceability (59 -> 73); valence peaks in 2013 (63) and dips in 2014-2016 (42) before partially recovering. Correlation analysis shows strong coupling of energy with loudness (r = 0.74) and negative associations for acousticness with both energy (r = -0.54) and loudness (r = -0.51); danceability is largely orthogonal to other features (|r| < 0.20). Lyric tokenization (>114k tokens) reveals a pronoun-centric lexicon "I/you/me/my" and a dense co-occurrence structure in which interpersonal address anchors mainstream narratives. Mood differs systematically by style: R&B exhibits the highest mean valence (96), followed by K-Pop/Pop (77) and Indie/Pop (70), whereas Latin/Reggaeton is lower (37) despite high danceability. Read through a subcultural identity lens, these patterns suggest the mainstreaming of previously peripheral codes and a commercial preference for relaxed yet rhythmically engaging productions that sustain collective participation without maximal intensity. Methodologically, we contribute an integrated MIR-plus-network workflow spanning summary statistics, correlation structure, lexical co-occurrence matrices, and genre-wise mood profiling that is robust to modality sparsity and suitable for socially aware recommendation or community-level diffusion studies.
>
---
#### [new 013] SAGE-Music: Low-Latency Symbolic Music Generation via Attribute-Specialized Key-Value Head Sharing
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于符号音乐生成任务，解决实时创作中速度与质量的矛盾。提出AS-KVHS方法，在多轨设置下提升推理速度并保持音乐质量。**

- **链接: [http://arxiv.org/pdf/2510.00395v1](http://arxiv.org/pdf/2510.00395v1)**

> **作者:** Jiaye Tan; Haonan Luo; Linfeng Song; Shuaiqi Chen; Yishan Lyu; Zian Zhong; Roujia Wang; Daniel Jiang; Haoran Zhang; Jiaming Bai; Haoran Cheng; Q. Vera Liao; Hao-Wen Dong
>
> **摘要:** Low-latency symbolic music generation is essential for real-time improvisation and human-AI co-creation. Existing transformer-based models, however, face a trade-off between inference speed and musical quality. Traditional acceleration techniques such as embedding pooling significantly degrade quality, while recently proposed Byte Pair Encoding (BPE) methods - though effective on single-track piano data - suffer large performance drops in multi-track settings, as revealed by our analysis. We propose Attribute-Specialized Key-Value Head Sharing (AS-KVHS), adapted to music's structured symbolic representation, achieving about 30% inference speedup with only a negligible (about 0.4%) quality drop in objective evaluations and slight improvements in subjective listening tests. Our main contributions are (1) the first systematic study of BPE's generalizability in multi-track symbolic music, and (2) the introduction of AS-KVHS for low-latency symbolic music generation. Beyond these, we also release SAGE-Music, an open-source benchmark that matches or surpasses state-of-the-art models in generation quality.
>
---
#### [new 014] ARIONet: An Advanced Self-supervised Contrastive Representation Network for Birdsong Classification and Future Frame Prediction
- **分类: cs.SD**

- **简介: 该论文属于鸟类叫声分类与未来帧预测任务，解决依赖标注数据和忽略时序动态的问题，提出ARIONet模型实现自监督学习。**

- **链接: [http://arxiv.org/pdf/2510.00522v1](http://arxiv.org/pdf/2510.00522v1)**

> **作者:** Md. Abdur Rahman; Selvarajah Thuseethan; Kheng Cher Yeo; Reem E. Mohamed; Sami Azam
>
> **摘要:** Automated birdsong classification is essential for advancing ecological monitoring and biodiversity studies. Despite recent progress, existing methods often depend heavily on labeled data, use limited feature representations, and overlook temporal dynamics essential for accurate species identification. In this work, we propose a self-supervised contrastive network, ARIONet (Acoustic Representation for Interframe Objective Network), that jointly optimizes contrastive classification and future frame prediction using augmented audio representations. The model simultaneously integrates multiple complementary audio features within a transformer-based encoder model. Our framework is designed with two key objectives: (1) to learn discriminative species-specific representations for contrastive learning through maximizing similarity between augmented views of the same audio segment while pushing apart different samples, and (2) to model temporal dynamics by predicting future audio frames, both without requiring large-scale annotations. We validate our framework on four diverse birdsong datasets, including the British Birdsong Dataset, Bird Song Dataset, and two extended Xeno-Canto subsets (A-M and N-Z). Our method consistently outperforms existing baselines and achieves classification accuracies of 98.41%, 93.07%, 91.89%, and 91.58%, and F1-scores of 97.84%, 94.10%, 91.29%, and 90.94%, respectively. Furthermore, it demonstrates low mean absolute errors and high cosine similarity, up to 95%, in future frame prediction tasks. Extensive experiments further confirm the effectiveness of our self-supervised learning strategy in capturing complex acoustic patterns and temporal dependencies, as well as its potential for real-world applicability in ecological conservation and monitoring.
>
---
#### [new 015] WaveMind: Towards a Conversational EEG Foundation Model Aligned to Textual and Visual Modalities
- **分类: eess.SP; cs.AI; cs.CL; cs.LG; q-bio.NC**

- **简介: 该论文属于脑信号分析任务，旨在解决EEG与文本、视觉模态对齐问题。通过构建统一语义空间和跨任务数据集，提升EEG的分类准确性和对话能力。**

- **链接: [http://arxiv.org/pdf/2510.00032v1](http://arxiv.org/pdf/2510.00032v1)**

> **作者:** Ziyi Zeng; Zhenyang Cai; Yixi Cai; Xidong Wang; Junying Chen; Rongsheng Wang; Yipeng Liu; Siqi Cai; Benyou Wang; Zhiguo Zhang; Haizhou Li
>
> **摘要:** Electroencephalography (EEG) interpretation using multimodal large language models (MLLMs) offers a novel approach for analyzing brain signals. However, the complex nature of brain activity introduces critical challenges: EEG signals simultaneously encode both cognitive processes and intrinsic neural states, creating a mismatch in EEG paired-data modality that hinders effective cross-modal representation learning. Through a pivot investigation, we uncover complementary relationships between these modalities. Leveraging this insight, we propose mapping EEG signals and their corresponding modalities into a unified semantic space to achieve generalized interpretation. To fully enable conversational capabilities, we further introduce WaveMind-Instruct-338k, the first cross-task EEG dataset for instruction tuning. The resulting model demonstrates robust classification accuracy while supporting flexible, open-ended conversations across four downstream tasks, thereby offering valuable insights for both neuroscience research and the development of general-purpose EEG models.
>
---
#### [new 016] Reference-free automatic speech severity evaluation using acoustic unit language modelling
- **分类: cs.SD**

- **简介: 该论文属于语音严重程度评估任务，旨在解决现有模型依赖参考数据和泛化能力差的问题。提出无需参考的SpeechLMScore方法，并构建新数据集进行验证。**

- **链接: [http://arxiv.org/pdf/2510.00639v1](http://arxiv.org/pdf/2510.00639v1)**

> **作者:** Bence Mark Halpern; Tomoki Toda
>
> **备注:** 5 pages. Proceedings of the 6th ACM International Conference on Multimedia in Asia Workshops
>
> **摘要:** Speech severity evaluation is becoming increasingly important as the economic burden of speech disorders grows. Current speech severity models often struggle with generalization, learning dataset-specific acoustic cues rather than meaningful correlates of speech severity. Furthermore, many models require reference speech or a transcript, limiting their applicability in ecologically valid scenarios, such as spontaneous speech evaluation. Previous research indicated that automatic speech naturalness evaluation scores correlate strongly with severity evaluation scores, leading us to explore a reference-free method, SpeechLMScore, which does not rely on pathological speech data. Additionally, we present the NKI-SpeechRT dataset, based on the NKI-CCRT dataset, to provide a more comprehensive foundation for speech severity evaluation. This study evaluates whether SpeechLMScore outperforms traditional acoustic feature-based approaches and assesses the performance gap between reference-free and reference-based models. Moreover, we examine the impact of noise on these models by utilizing subjective noise ratings in the NKI-SpeechRT dataset. The results demonstrate that SpeechLMScore is robust to noise and offers superior performance compared to traditional approaches.
>
---
#### [new 017] Hearing the Order: Investigating Selection Bias in Large Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频语言模型研究，探讨模型在有序选项任务中的选择偏差问题。通过实验发现模型预测受选项顺序影响，提出排列策略缓解偏差。**

- **链接: [http://arxiv.org/pdf/2510.00628v1](http://arxiv.org/pdf/2510.00628v1)**

> **作者:** Yu-Xiang Lin; Chen-An Li; Sheng-Lun Wei; Po-Chun Chen; Hsin-Hsi Chen; Hung-yi Lee
>
> **备注:** The first two authors contributed equally. Submitted to ICASSP 2026
>
> **摘要:** Large audio-language models (LALMs) are often used in tasks that involve reasoning over ordered options. An open question is whether their predictions are influenced by the order of answer choices, which would indicate a form of selection bias and undermine their reliability. In this paper, we identify and analyze this problem in LALMs. We demonstrate that no model is immune to this bias through extensive experiments on six LALMs across three widely used benchmarks and their spoken counterparts. Shuffling the order of answer options can cause performance fluctuations of up to 24% and even change model rankings, raising concerns about the reliability of current evaluation practices. We also study permutation-based strategies and show that they can mitigate bias in most cases. Our work represents the first systematic investigation of this issue in LALMs, and we hope it raises awareness and motivates further research in this direction.
>
---
#### [new 018] A Recall-First CNN for Sleep Apnea Screening from Snoring Audio
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于睡眠呼吸暂停检测任务，旨在通过鼾声音频进行筛查。研究使用卷积神经网络，提高检测召回率，以低成本方式早期识别高风险人群。**

- **链接: [http://arxiv.org/pdf/2510.00052v1](http://arxiv.org/pdf/2510.00052v1)**

> **作者:** Anushka Mallick; Afiya Noorain; Ashwin Menon; Ashita Solanki; Keertan Balaji
>
> **摘要:** Sleep apnea is a serious sleep-related breathing disorder that is common and can impact health if left untreated. Currently the traditional method for screening and diagnosis is overnight polysomnography. Polysomnography is expensive and takes a lot of time, and is not practical for screening large groups of people. In this paper, we explored a more accessible option, using respiratory audio recordings to spot signs of apnea.We utilized 18 audio files.The approach involved converting breathing sounds into spectrograms, balancing the dataset by oversampling apnea segments, and applying class weights to reduce bias toward the majority class. The model reached a recall of 90.55 for apnea detection. Intentionally, prioritizing catching apnea events over general accuracy. Despite low precision,the high recall suggests potential as a low-cost screening tool that could be used at home or in basic clinical setups, potentially helping identify at-risk individuals much earlier.
>
---
#### [new 019] SAGE-LD: Towards Scalable and Generalizable End-to-End Language Diarization via Simulated Data Augmentation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语言辨识任务，解决多语言场景下的数据不足与模型泛化问题。通过模拟数据增强和多语言预训练，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.00582v1](http://arxiv.org/pdf/2510.00582v1)**

> **作者:** Sangmin Lee; Woongjib Choi; Jihyun Kim; Hong-Goo Kang
>
> **摘要:** In this paper, we present a neural spoken language diarization model that supports an unconstrained span of languages within a single framework. Our approach integrates a learnable query-based architecture grounded in multilingual awareness, with large-scale pretraining on simulated code-switching data. By jointly leveraging these two components, our method overcomes the limitations of conventional approaches in data scarcity and architecture optimization, and generalizes effectively to real-world multilingual settings across diverse environments. Experimental results demonstrate that our approach achieves state-of-the-art performance on several language diarization benchmarks, with a relative performance improvement of 23% to 52% over previous methods. We believe that this work not only advances research in language diarization but also establishes a foundational framework for code-switching speech technologies.
>
---
#### [new 020] Object-AVEdit: An Object-level Audio-Visual Editing Model
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视频编辑任务，解决对象级音视频操作难题。提出Object-AVEdit模型，实现对象添加、替换和移除，同时保持结构信息。**

- **链接: [http://arxiv.org/pdf/2510.00050v1](http://arxiv.org/pdf/2510.00050v1)**

> **作者:** Youquan Fu; Ruiyang Si; Hongfa Wang; Dongzhan Zhou; Jiacheng Sun; Ping Luo; Di Hu; Hongyuan Zhang; Xuelong Li
>
> **摘要:** There is a high demand for audio-visual editing in video post-production and the film making field. While numerous models have explored audio and video editing, they struggle with object-level audio-visual operations. Specifically, object-level audio-visual editing requires the ability to perform object addition, replacement, and removal across both audio and visual modalities, while preserving the structural information of the source instances during the editing process. In this paper, we present \textbf{Object-AVEdit}, achieving the object-level audio-visual editing based on the inversion-regeneration paradigm. To achieve the object-level controllability during editing, we develop a word-to-sounding-object well-aligned audio generation model, bridging the gap in object-controllability between audio and current video generation models. Meanwhile, to achieve the better structural information preservation and object-level editing effect, we propose an inversion-regeneration holistically-optimized editing algorithm, ensuring both information retention during the inversion and better regeneration effect. Extensive experiments demonstrate that our editing model achieved advanced results in both audio-video object-level editing tasks with fine audio-visual semantic alignment. In addition, our developed audio generation model also achieved advanced performance. More results on our project page: https://gewu-lab.github.io/Object_AVEdit-website/.
>
---
#### [new 021] Backdoor Attacks Against Speech Language Models
- **分类: cs.CL; cs.CR; cs.SD**

- **简介: 该论文属于语音语言模型安全研究，解决后门攻击问题。通过系统分析音频后门攻击，验证其有效性并提出防御方法。**

- **链接: [http://arxiv.org/pdf/2510.01157v1](http://arxiv.org/pdf/2510.01157v1)**

> **作者:** Alexandrine Fortier; Thomas Thebaud; Jesús Villalba; Najim Dehak; Patrick Cardinal
>
> **摘要:** Large Language Models (LLMs) and their multimodal extensions are becoming increasingly popular. One common approach to enable multimodality is to cascade domain-specific encoders with an LLM, making the resulting model inherit vulnerabilities from all of its components. In this work, we present the first systematic study of audio backdoor attacks against speech language models. We demonstrate its effectiveness across four speech encoders and three datasets, covering four tasks: automatic speech recognition (ASR), speech emotion recognition, and gender and age prediction. The attack consistently achieves high success rates, ranging from 90.76% to 99.41%. To better understand how backdoors propagate, we conduct a component-wise analysis to identify the most vulnerable stages of the pipeline. Finally, we propose a fine-tuning-based defense that mitigates the threat of poisoned pretrained encoders.
>
---
#### [new 022] DiffAU: Diffusion-Based Ambisonics Upscaling
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于Ambisonics上采样任务，旨在提升一阶Ambisonics的分辨率以获得更高阶的3D音频，通过扩散模型实现高效可靠的上采样。**

- **链接: [http://arxiv.org/pdf/2510.00180v1](http://arxiv.org/pdf/2510.00180v1)**

> **作者:** Amit Milstein; Nir Shlezinger; Boaz Rafaely
>
> **摘要:** Spatial audio enhances immersion by reproducing 3D sound fields, with Ambisonics offering a scalable format for this purpose. While first-order Ambisonics (FOA) notably facilitates hardware-efficient acquisition and storage of sound fields as compared to high-order Ambisonics (HOA), its low spatial resolution limits realism, highlighting the need for Ambisonics upscaling (AU) as an approach for increasing the order of Ambisonics signals. In this work we propose DiffAU, a cascaded AU method that leverages recent developments in diffusion models combined with novel adaptation to spatial audio to generate 3rd order Ambisonics from FOA. By learning data distributions, DiffAU provides a principled approach that rapidly and reliably reproduces HOA in various settings. Experiments in anechoic conditions with multiple speakers, show strong objective and perceptual performance.
>
---
#### [new 023] Descriptor:: Extended-Length Audio Dataset for Synthetic Voice Detection and Speaker Recognition (ELAD-SVDSR)
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出ELAD-SVDSR数据集，用于合成语音检测与说话人识别任务，旨在提升深度伪造语音的生成与检测能力。**

- **链接: [http://arxiv.org/pdf/2510.00218v1](http://arxiv.org/pdf/2510.00218v1)**

> **作者:** Rahul Vijaykumar; Ajan Ahmed; John Parker; Dinesh Pendyala; Aidan Collins; Stephanie Schuckers; Masudul H. Imtiaz
>
> **摘要:** This paper introduces the Extended Length Audio Dataset for Synthetic Voice Detection and Speaker Recognition (ELAD SVDSR), a resource specifically designed to facilitate the creation of high quality deepfakes and support the development of detection systems trained against them. The dataset comprises 45 minute audio recordings from 36 participants, each reading various newspaper articles recorded under controlled conditions and captured via five microphones of differing quality. By focusing on extended duration audio, ELAD SVDSR captures a richer range of speech attributes such as pitch contours, intonation patterns, and nuanced delivery enabling models to generate more realistic and coherent synthetic voices. In turn, this approach allows for the creation of robust deepfakes that can serve as challenging examples in datasets used to train and evaluate synthetic voice detection methods. As part of this effort, 20 deepfake voices have already been created and added to the dataset to showcase its potential. Anonymized metadata accompanies the dataset on speaker demographics. ELAD SVDSR is expected to spur significant advancements in audio forensics, biometric security, and voice authentication systems.
>
---
#### [new 024] Spiralformer: Low Latency Encoder for Streaming Speech Recognition with Circular Layer Skipping and Early Exiting
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在降低流式处理中的编码延迟。通过提出Spiralformer模型，结合层跳过和提前退出机制，有效减少平均令牌发射延迟。**

- **链接: [http://arxiv.org/pdf/2510.00982v1](http://arxiv.org/pdf/2510.00982v1)**

> **作者:** Emiru Tsunoo; Hayato Futami; Yosuke Kashiwagi; Siddhant Arora; Shinji Watanabe
>
> **备注:** Accepted for ASRU 2025
>
> **摘要:** For streaming speech recognition, a Transformer-based encoder has been widely used with block processing. Although many studies addressed improving emission latency of transducers, little work has been explored for improving encoding latency of the block processing. We seek to reduce latency by frequently emitting a chunk with a small shift rather than scarce large-chunk emissions, resulting in higher computational costs. To efficiently compute with the small chunk shift, we propose a new encoder, Spiralformer, tailored for block processing by combining layer dropping and early exiting. We skip layer computation in a cyclic manner and shift the computed layer in each block spirally, which completes computation for all the layers over the block processing. Experimentally, we observed that our method achieved 21.6% reduction in the averaged token emission delay in Librispeech, and 7.0% in CSJ, compared with the baseline with similar computational cost and word error rates.
>
---
#### [new 025] CL-UZH submission to the NIST SRE 2024 Speaker Recognition Evaluation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于说话人识别任务，解决NIST SRE 2024挑战中的固定和开放条件问题。团队提交了音频和视听系统，使用X-vector和预训练模型进行实验。**

- **链接: [http://arxiv.org/pdf/2510.00952v1](http://arxiv.org/pdf/2510.00952v1)**

> **作者:** Aref Farhadipour; Shiran Liu; Masoumeh Chapariniya; Valeriia Perepelytsia; Srikanth Madikeri; Teodora Vukovic; Volker Dellwo
>
> **备注:** CL-UZH submission for the NIST SRE 2024 Evaluation plan
>
> **摘要:** The CL-UZH team submitted one system each for the fixed and open conditions of the NIST SRE 2024 challenge. For the closed-set condition, results for the audio-only trials were achieved using the X-vector system developed with Kaldi. For the audio-visual results we used only models developed for the visual modality. Two sets of results were submitted for the open-set and closed-set conditions, one based on a pretrained model using the VoxBlink2 and VoxCeleb2 datasets. An Xvector-based model was trained from scratch using the CTS superset dataset for the closed set. In addition to the submission of the results of the SRE24 evaluation to the competition website, we talked about the performance of the proposed systems on the SRE24 evaluation in this report.
>
---
#### [new 026] Improved Hyperspectral Anomaly Detection via Unsupervised Subspace Modeling in the Signed Cumulative Distribution Transform Domain
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于 hyperspectral anomaly detection 任务，旨在解决复杂环境中异常像素检测问题。通过引入基于运输的数学模型和无监督子空间建模，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2510.00148v1](http://arxiv.org/pdf/2510.00148v1)**

> **作者:** Abu Hasnat Mohammad Rubaiyat; Jordan Vincent; Colin Olson
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Hyperspectral anomaly detection (HAD), a crucial approach for many civilian and military applications, seeks to identify pixels with spectral signatures that are anomalous relative to a preponderance of background signatures. Significant effort has been made to improve HAD techniques, but challenges arise due to complex real-world environments and, by definition, limited prior knowledge of potential signatures of interest. This paper introduces a novel HAD method by proposing a transport-based mathematical model to describe the pixels comprising a given hyperspectral image. In this approach, hyperspectral pixels are viewed as observations of a template pattern undergoing unknown deformations that enables their representation in the signed cumulative distribution transform (SCDT) domain. An unsupervised subspace modeling technique is then used to construct a model of abundant background signals in this domain, whereupon anomalous signals are detected as deviations from the learned model. Comprehensive evaluations across five distinct datasets illustrate the superiority of our approach compared to state-of-the-art methods.
>
---
#### [new 027] Learning Domain-Robust Bioacoustic Representations for Mosquito Species Classification with Contrastive Learning and Distribution Alignment
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于蚊虫物种分类任务，解决跨领域性能下降问题。通过对比学习和分布对齐方法，提升模型在不同环境下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.00346v1](http://arxiv.org/pdf/2510.00346v1)**

> **作者:** Yuanbo Hou; Zhaoyi Liu; Xin Shen; Stephen Roberts
>
> **摘要:** Mosquito Species Classification (MSC) is crucial for vector surveillance and disease control. The collection of mosquito bioacoustic data is often limited by mosquito activity seasons and fieldwork. Mosquito recordings across regions, habitats, and laboratories often show non-biological variations from the recording environment, which we refer to as domain features. This study finds that models directly trained on audio recordings with domain features tend to rely on domain information rather than the species' acoustic cues for identification, resulting in illusory good performance while actually performing poor cross-domain generalization. To this end, we propose a Domain-Robust Bioacoustic Learning (DR-BioL) framework that combines contrastive learning with distribution alignment. Contrastive learning aims to promote cohesion within the same species and mitigate inter-domain discrepancies, and species-conditional distribution alignment further enhances cross-domain species representation. Experiments on a multi-domain mosquito bioacoustic dataset from diverse environments show that the DR-BioL improves the accuracy and robustness of baselines, highlighting its potential for reliable cross-domain MSC in the real world.
>
---
#### [new 028] UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于音频超分辨率任务，解决传统方法依赖声码器的问题。通过流匹配模型直接重建波形，提升音频质量与效率。**

- **链接: [http://arxiv.org/pdf/2510.00771v1](http://arxiv.org/pdf/2510.00771v1)**

> **作者:** Woongjib Choi; Sangmin Lee; Hyungseob Lim; Hong-Goo Kang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** In this paper, we present a vocoder-free framework for audio super-resolution that employs a flow matching generative model to capture the conditional distribution of complex-valued spectral coefficients. Unlike conventional two-stage diffusion-based approaches that predict a mel-spectrogram and then rely on a pre-trained neural vocoder to synthesize waveforms, our method directly reconstructs waveforms via the inverse Short-Time Fourier Transform (iSTFT), thereby eliminating the dependence on a separate vocoder. This design not only simplifies end-to-end optimization but also overcomes a critical bottleneck of two-stage pipelines, where the final audio quality is fundamentally constrained by vocoder performance. Experiments show that our model consistently produces high-fidelity 48 kHz audio across diverse upsampling factors, achieving state-of-the-art performance on both speech and general audio datasets.
>
---
#### [new 029] Post-Training Quantization for Audio Diffusion Transformers
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频生成任务，解决Diffusion Transformers计算和存储开销大的问题，通过后训练量化技术提升效率与精度平衡。**

- **链接: [http://arxiv.org/pdf/2510.00313v1](http://arxiv.org/pdf/2510.00313v1)**

> **作者:** Tanmay Khandelwal; Magdalena Fuentes
>
> **备注:** 5 pages, 4 figures, accepted at IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** Diffusion Transformers (DiTs) enable high-quality audio synthesis but are often computationally intensive and require substantial storage, which limits their practical deployment. In this paper, we present a comprehensive evaluation of post-training quantization (PTQ) techniques for audio DiTs, analyzing the trade-offs between static and dynamic quantization schemes. We explore two practical extensions (1) a denoising-timestep-aware smoothing method that adapts quantization scales per-input-channel and timestep to mitigate activation outliers, and (2) a lightweight low-rank adapter (LoRA)-based branch derived from singular value decomposition (SVD) to compensate for residual weight errors. Using Stable Audio Open we benchmark W8A8 and W4A8 configurations across objective metrics and human perceptual ratings. Our results show that dynamic quantization preserves fidelity even at lower precision, while static methods remain competitive with lower latency. Overall, our findings show that low-precision DiTs can retain high-fidelity generation while reducing memory usage by up to 79%.
>
---
#### [new 030] Subjective quality evaluation of personalized own voice reconstruction systems
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决个性化语音重建系统的质量评估问题。通过数据增强和微调提升系统性能，并对比客观与主观评价结果。**

- **链接: [http://arxiv.org/pdf/2510.00256v1](http://arxiv.org/pdf/2510.00256v1)**

> **作者:** Mattes Ohlenbusch; Christian Rollwage; Simon Doclo; Jan Rennies
>
> **备注:** Submitted to Acta Acustica
>
> **摘要:** Own voice pickup technology for hearable devices facilitates communication in noisy environments. Own voice reconstruction (OVR) systems enhance the quality and intelligibility of the recorded noisy own voice signals. Since disturbances affecting the recorded own voice signals depend on individual factors, personalized OVR systems have the potential to outperform generic OVR systems. In this paper, we propose personalizing OVR systems through data augmentation and fine-tuning, comparing them to their generic counterparts. We investigate the influence of personalization on speech quality assessed by objective metrics and conduct a subjective listening test to evaluate quality under various conditions. In addition, we assess the prediction accuracy of the objective metrics by comparing predicted quality with subjectively measured quality. Our findings suggest that personalized OVR provides benefits over generic OVR for some talkers only. Our results also indicate that performance comparisons between systems are not always accurately predicted by objective metrics. In particular, certain disturbances lead to a consistent overestimation of quality compared to actual subjective ratings.
>
---
#### [new 031] Audio Driven Real-Time Facial Animation for Social Telepresence
- **分类: cs.GR; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于语音驱动的实时面部动画任务，旨在解决虚拟现实社交中面部表情生成的延迟与准确性问题。通过编码器-解码器架构和扩散模型实现快速、逼真的面部动画。**

- **链接: [http://arxiv.org/pdf/2510.01176v1](http://arxiv.org/pdf/2510.01176v1)**

> **作者:** Jiye Lee; Chenghui Li; Linh Tran; Shih-En Wei; Jason Saragih; Alexander Richard; Hanbyul Joo; Shaojie Bai
>
> **备注:** SIGGRAPH Asia 2025. Project page: https://jiyewise.github.io/projects/AudioRTA
>
> **摘要:** We present an audio-driven real-time system for animating photorealistic 3D facial avatars with minimal latency, designed for social interactions in virtual reality for anyone. Central to our approach is an encoder model that transforms audio signals into latent facial expression sequences in real time, which are then decoded as photorealistic 3D facial avatars. Leveraging the generative capabilities of diffusion models, we capture the rich spectrum of facial expressions necessary for natural communication while achieving real-time performance (<15ms GPU time). Our novel architecture minimizes latency through two key innovations: an online transformer that eliminates dependency on future inputs and a distillation pipeline that accelerates iterative denoising into a single step. We further address critical design challenges in live scenarios for processing continuous audio signals frame-by-frame while maintaining consistent animation quality. The versatility of our framework extends to multimodal applications, including semantic modalities such as emotion conditions and multimodal sensors with head-mounted eye cameras on VR headsets. Experimental results demonstrate significant improvements in facial animation accuracy over existing offline state-of-the-art baselines, achieving 100 to 1000 times faster inference speed. We validate our approach through live VR demonstrations and across various scenarios such as multilingual speeches.
>
---
#### [new 032] Room Impulse Response Synthesis via Differentiable Feedback Delay Networks for Efficient Spatial Audio Rendering
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于空间音频渲染任务，解决传统方法计算成本高、延迟大的问题。通过优化FDN参数实现高效实时RIR生成。**

- **链接: [http://arxiv.org/pdf/2510.00238v1](http://arxiv.org/pdf/2510.00238v1)**

> **作者:** Armin Gerami; Ramani Duraiswami
>
> **摘要:** We introduce a computationally efficient and tunable feedback delay network (FDN) architecture for real-time room impulse response (RIR) rendering that addresses the computational and latency challenges inherent in traditional convolution and Fourier transform based methods. Our approach directly optimizes FDN parameters to match target RIR acoustic and psychoacoustic metrics such as clarity and definition through novel differentiable programming-based optimization. Our method enables dynamic, real-time adjustments of room impulse responses that accommodates listener and source movement. When combined with previous work on representation of head-related impulse responses via infinite impulse responses, an efficient rendering of auditory objects is possible when the HRIR and RIR are known. Our method produces renderings with quality similar to convolution with long binaural room impulse response (BRIR) filters, but at a fraction of the computational cost.
>
---
## 更新

#### [replaced 001] DualCodec: A Low-Frame-Rate, Semantically-Enhanced Neural Audio Codec for Speech Generation
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13000v2](http://arxiv.org/pdf/2505.13000v2)**

> **作者:** Jiaqi Li; Xiaolong Lin; Zhekai Li; Shixi Huang; Yuancheng Wang; Chaoren Wang; Zhenpeng Zhan; Zhizheng Wu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Neural audio codecs form the foundational building blocks for language model (LM)-based speech generation. Typically, there is a trade-off between frame rate and audio quality. This study introduces a low-frame-rate, semantically enhanced codec model. Existing approaches distill semantically rich self-supervised (SSL) representations into the first-layer codec tokens. This work proposes DualCodec, a dual-stream encoding approach that integrates SSL and waveform representations within an end-to-end codec framework. In this setting, DualCodec enhances the semantic information in the first-layer codec and enables the codec system to maintain high audio quality while operating at a low frame rate. Note that a low-frame-rate codec improves the efficiency of speech generation. Experimental results on audio codec and speech generation tasks confirm the effectiveness of the proposed DualCodec compared to state-of-the-art codec systems, such as Mimi Codec, SpeechTokenizer, DAC, and Encodec. Demos are available at: https://dualcodec.github.io, code is available at: https://github.com/jiaqili3/DualCodec
>
---
#### [replaced 002] A dataset and model for recognition of audiologically relevant environments for hearing aids: AHEAD-DS and YAMNet+
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.10360v3](http://arxiv.org/pdf/2508.10360v3)**

> **作者:** Henry Zhong; Jörg M. Buchholz; Julian Maclaren; Simon Carlile; Richard Lyon
>
> **摘要:** Scene recognition of audiologically relevant environments is important for hearing aids; however, it is challenging, in part because of the limitations of existing datasets. Datasets often lack public accessibility, completeness, or audiologically relevant labels, hindering systematic comparison of machine learning models. Deploying these models on resource-constrained edge devices presents another challenge. Our solution is two-fold: we leverage several open source datasets to create AHEAD-DS, a dataset designed for scene recognition of audiologically relevant environments, and introduce YAMNet+, a sound recognition model. AHEAD-DS aims to provide a standardised, publicly available dataset with consistent labels relevant to hearing aids, facilitating model comparison. YAMNet+ is designed for deployment on edge devices like smartphones connected to hearing devices, such as hearing aids and wireless earphones with hearing aid functionality; serving as a baseline model for sound-based scene recognition. YAMNet+ achieved a mean average precision of 0.83 and accuracy of 0.93 on the testing set of AHEAD-DS across fourteen categories of audiologically relevant environments. We found that applying transfer learning from the pretrained YAMNet model was essential. We demonstrated real-time sound-based scene recognition capabilities on edge devices by deploying YAMNet+ to an Android smartphone. Even with a Google Pixel 3 (a phone with modest specifications, released in 2018), the model processes audio with approximately 50ms of latency to load the model, and an approximate linear increase of 30ms per 1 second of audio. Our website and code https://github.com/Australian-Future-Hearing-Initiative .
>
---
#### [replaced 003] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.15957v3](http://arxiv.org/pdf/2505.15957v3)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** EMNLP 2025 (Main). Project Website: https://github.com/ckyang1124/LALM-Evaluation-Survey
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [replaced 004] Deep Learning for Tuberculosis Screening in a High-burden Setting using Cough Analysis and Speech Foundation Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09746v2](http://arxiv.org/pdf/2509.09746v2)**

> **作者:** Ning Ma; Bahman Mirheidari; Guy J. Brown; Nsala Sanjase; Minyoi M. Maimbolwa; Solomon Chifwamba; Seke Muzazu; Monde Muyoyeta; Mary Kagujje
>
> **备注:** submitted to IEEE Journal of Biomedical and Health Informatics
>
> **摘要:** Artificial intelligence (AI) systems can detect disease-related acoustic patterns in cough sounds, offering a scalable and cost-effective approach to tuberculosis (TB) screening in high-burden, resource-limited settings. Previous studies have been limited by small datasets, under-representation of symptomatic non-TB patients, and recordings collected in controlled environments. In this study, we enrolled 512 participants at two hospitals in Zambia, categorised into three groups: bacteriologically confirmed TB (TB+), symptomatic patients with other respiratory diseases (OR), and healthy controls (HC). Usable cough recordings with demographic and clinical data were obtained from 500 participants. Deep learning classifiers based on pre-trained speech foundation models were fine-tuned on cough recordings to predict diagnostic categories. The best-performing model, trained on 3-second audio clips, achieved an AUROC of 85.2% for distinguishing TB coughs from all other participants (TB+/Rest) and 80.1% for TB+ versus symptomatic OR participants (TB+/OR). Incorporating demographic and clinical features improved performance to 92.1% for TB+/Rest and 84.2% for TB+/OR. At a probability threshold of 0.38, the multimodal model reached 90.3% sensitivity and 73.1% specificity for TB+/Rest, meeting WHO target product profile benchmarks for TB screening. Adversarial testing and stratified analyses shows that the model was robust to confounding factors including background noise, recording time, and device variability. These results demonstrate the feasibility of cough-based AI for TB screening in real-world, low-resource settings.
>
---
#### [replaced 005] An Agent-Based Framework for Automated Higher-Voice Harmony Generation
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.24463v2](http://arxiv.org/pdf/2509.24463v2)**

> **作者:** Nia D'Souza Ganapathy; Arul Selvamani Shaja
>
> **摘要:** The generation of musically coherent and aesthetically pleasing harmony remains a significant challenge in the field of algorithmic composition. This paper introduces an innovative Agentic AI-enabled Higher Harmony Music Generator, a multi-agent system designed to create harmony in a collaborative and modular fashion. Our framework comprises four specialized agents: a Music-Ingestion Agent for parsing and standardizing input musical scores; a Chord-Knowledge Agent, powered by a Chord-Former (Transformer model), to interpret and provide the constituent notes of complex chord symbols; a Harmony-Generation Agent, which utilizes a Harmony-GPT and a Rhythm-Net (RNN) to compose a melodically and rhythmically complementary harmony line; and an Audio-Production Agent that employs a GAN-based Symbolic-to-Audio Synthesizer to render the final symbolic output into high-fidelity audio. By delegating specific tasks to specialized agents, our system effectively mimics the collaborative process of human musicians. This modular, agent-based approach allows for robust data processing, deep theoretical understanding, creative composition, and realistic audio synthesis, culminating in a system capable of generating sophisticated and contextually appropriate higher-voice harmonies for given melodies.
>
---
#### [replaced 006] DeepASA: An Object-Oriented One-for-All Network for Auditory Scene Analysis
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.17247v2](http://arxiv.org/pdf/2509.17247v2)**

> **作者:** Dongheon Lee; Younghoo Kwon; Jung-Woo Choi
>
> **备注:** 19 pages, 13 figures, 8 tables, accepted in NeurIPS 2025
>
> **摘要:** We propose DeepASA, a multi-purpose model for auditory scene analysis that performs multi-input multi-output (MIMO) source separation, dereverberation, sound event detection (SED), audio classification, and direction-of-arrival estimation (DoAE) within a unified framework. DeepASA is designed for complex auditory scenes where multiple, often similar, sound sources overlap in time and move dynamically in space. To achieve robust and consistent inference across tasks, we introduce an object-oriented processing (OOP) strategy. This approach encapsulates diverse auditory features into object-centric representations and refines them through a chain-of-inference (CoI) mechanism. The pipeline comprises a dynamic temporal kernel-based feature extractor, a transformer-based aggregator, and an object separator that yields per-object features. These features feed into multiple task-specific decoders. Our object-centric representations naturally resolve the parameter association ambiguity inherent in traditional track-wise processing. However, early-stage object separation can lead to failure in downstream ASA tasks. To address this, we implement temporal coherence matching (TCM) within the chain-of-inference, enabling multi-task fusion and iterative refinement of object features using estimated auditory parameters. We evaluate DeepASA on representative spatial audio benchmark datasets, including ASA2, MC-FUSS, and STARSS23. Experimental results show that our model achieves state-of-the-art performance across all evaluated tasks, demonstrating its effectiveness in both source separation and auditory parameter estimation under diverse spatial auditory scenes.
>
---
#### [replaced 007] Toward a Robust R2D2 Paradigm for Radio-interferometric Imaging: Revisiting Deep Neural Network Training and Architecture
- **分类: astro-ph.IM; cs.CV; cs.LG; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.02554v2](http://arxiv.org/pdf/2503.02554v2)**

> **作者:** Amir Aghabiglou; Chung San Chu; Chao Tang; Arwa Dabbech; Yves Wiaux
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** The R2D2 Deep Neural Network (DNN) series was recently introduced for image formation in radio interferometry. It can be understood as a learned version of CLEAN, whose minor cycles are substituted with DNNs. We revisit R2D2 on the grounds of series convergence, training methodology, and DNN architecture, improving its robustness in terms of generalizability beyond training conditions, capability to deliver high data fidelity, and epistemic uncertainty. First, while still focusing on telescope-specific training, we enhance the learning process by randomizing Fourier sampling integration times, incorporating multiscan multinoise configurations, and varying imaging settings, including pixel resolution and visibility-weighting scheme. Second, we introduce a convergence criterion whereby the reconstruction process stops when the data residual is compatible with noise, rather than simply using all available DNNs. This not only increases the reconstruction efficiency by reducing its computational cost, but also refines training by pruning out the data/image pairs for which optimal data fidelity is reached before training the next DNN. Third, we substitute R2D2's early U-Net DNN with a novel architecture (U-WDSR) combining U-Net and WDSR, which leverages wide activation, dense skip connections, weight normalization, and low-rank convolution to improve feature reuse and reconstruction precision. As previously, R2D2 was trained for monochromatic intensity imaging with the Very Large Array at fixed $512 \times 512$ image size. Simulations on a wide range of inverse problems and a case study on real data reveal that the new R2D2 model consistently outperforms its earlier version in image reconstruction quality, data fidelity, and epistemic uncertainty.
>
---
#### [replaced 008] Nonlinear Framework for Speech Bandwidth Extension
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.15970v2](http://arxiv.org/pdf/2507.15970v2)**

> **作者:** Tarikul Islam Tamiti; Nursad Mamun; Anomadarshi Barua
>
> **摘要:** Recovering high-frequency components lost to bandwidth constraints is crucial for applications ranging from telecommunications to high-fidelity audio on limited resources. We introduce NDSI-BWE, a new adversarial Band Width Extension (BWE) framework that leverage four new discriminators inspired by nonlinear dynamical system to capture diverse temporal behaviors: a Multi-Resolution Lyapunov Discriminator (MRLD) for determining sensitivity to initial conditions by capturing deterministic chaos, a Multi-Scale Recurrence Discriminator (MS-RD) for self-similar recurrence dynamics, a Multi-Scale Detrended Fractal Analysis Discriminator (MSDFA) for long range slow variant scale invariant relationship, a Multi-Resolution Poincar\'e Plot Discriminator (MR-PPD) for capturing hidden latent space relationship, a Multi-Period Discriminator (MPD) for cyclical patterns, a Multi-Resolution Amplitude Discriminator (MRAD) and Multi-Resolution Phase Discriminator (MRPD) for capturing intricate amplitude-phase transition statistics. By using depth-wise convolution at the core of the convolutional block with in each discriminators, NDSI-BWE attains an eight-times parameter reduction. These seven discriminators guide a complex-valued ConformerNeXt based genetor with a dual stream Lattice-Net based architecture for simultaneous refinement of magnitude and phase. The genertor leverage the transformer based conformer's global dependency modeling and ConvNeXt block's local temporal modeling capability. Across six objective evaluation metrics and subjective based texts comprises of five human judges, NDSI-BWE establishes a new SoTA in BWE.
>
---
