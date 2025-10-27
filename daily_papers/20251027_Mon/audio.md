# 音频 cs.SD;  eess.AS

- **最新发布 17 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] WhaleVAD-BPN: Improving Baleen Whale Call Detection with Boundary Proposal Networks and Post-processing Optimisation
- **分类: eess.AS; cs.AI; cs.LG; cs.SD; q-bio.QM**

- **简介: 该论文针对海洋中须鲸叫声检测任务，解决现有声事件检测系统误报多、少数类检测差的问题。提出边界提议网络（BPN）优化模型输出，并采用前后向搜索法优化后处理超参数，显著提升精度与F1分数，实现9.8%的绝对性能改进。**

- **链接: [http://arxiv.org/pdf/2510.21280v1](http://arxiv.org/pdf/2510.21280v1)**

> **作者:** Christiaan M. Geldenhuys; Günther Tonitz; Thomas R. Niesler
>
> **摘要:** While recent sound event detection (SED) systems can identify baleen whale calls in marine audio, challenges related to false positive and minority-class detection persist. We propose the boundary proposal network (BPN), which extends an existing lightweight SED system. The BPN is inspired by work in image object detection and aims to reduce the number of false positive detections. It achieves this by using intermediate latent representations computed within the backbone classification model to gate the final output. When added to an existing SED system, the BPN achieves a 16.8 % absolute increase in precision, as well as 21.3 % and 9.4 % improvements in the F1-score for minority-class d-calls and bp-calls, respectively. We further consider two approaches to the selection of post-processing hyperparameters: a forward-search and a backward-search. By separately optimising event-level and frame-level hyperparameters, these two approaches lead to considerable performance improvements over parameters selected using empirical methods. The complete WhaleVAD-BPN system achieves a cross-validated development F1-score of 0.475, which is a 9.8 % absolute improvement over the baseline.
>
---
#### [new 002] Can large audio language models understand child stuttering speech? speech summarization, and source separation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究大音频语言模型（LALMs）对儿童口吃语音的理解能力，聚焦于混合语音中的儿童语音分离与保留临床相关口吃特征的摘要生成。通过对比模型在访谈和朗读任务中的表现，结合自动评估与人工评分，揭示了模型在真实场景下的有效性与局限性，为临床与教育应用提供指导。**

- **链接: [http://arxiv.org/pdf/2510.20850v1](http://arxiv.org/pdf/2510.20850v1)**

> **作者:** Chibuzor Okocha; Maya Bakri; Christan Grant
>
> **备注:** 7 pages, 1 Figure, 8 tables, Under review ICASSP 2026
>
> **摘要:** Child speech differs from adult speech in acoustics, prosody, and language development, and disfluencies (repetitions, prolongations, blocks) further challenge Automatic Speech Recognition (ASR) and downstream Natural Language Processing (NLP). Recent large audio-language models (LALMs) demonstrate strong cross-modal audio understanding; however, their behavior in disfluent child speech remains underexplored. We evaluate several state-of-the-art LALMs in two settings: an interview (mixed speakers) and a reading task (single child). The tasks are (i) single-channel source separation to isolate the child and (ii) child-only summarization that preserves clinically relevant disfluencies and avoids adult-speech leakage. Evaluation combines Large Language Model (LLM) as a judge, human expert ratings, and BERTScore (F1), and we report agreement between models and between models and humans to assess reliability. Our findings delineate the conditions under which LALMs produce faithful child-only summaries from mixed audio and where they fail, offering practical guidance for clinical and educational deployments. We provide prompts and evaluation scripts to support replication.
>
---
#### [new 003] PhoenixCodec: Taming Neural Speech Coding for Extreme Low-Resource Scenarios
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对极低资源下的神经语音编码任务，提出PhoenixCodec框架。通过优化的异构时频架构、循环校准与精炼训练策略及抗噪微调，有效提升编码效率与鲁棒性，在计算量低于700 MFLOPs、延迟小于30 ms条件下实现1 kbps与6 kbps双速率编码，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21196v1](http://arxiv.org/pdf/2510.21196v1)**

> **作者:** Zixiang Wan; Haoran Zhao; Guochang Zhang; Runqiang Han; Jianqiang Wei; Yuexian Zou
>
> **备注:** 5 pages, 1 figure, 4 tables
>
> **摘要:** This paper presents PhoenixCodec, a comprehensive neural speech coding and decoding framework designed for extremely low-resource conditions. The proposed system integrates an optimized asymmetric frequency-time architecture, a Cyclical Calibration and Refinement (CCR) training strategy, and a noise-invariant fine-tuning procedure. Under stringent constraints - computation below 700 MFLOPs, latency less than 30 ms, and dual-rate support at 1 kbps and 6 kbps - existing methods face a trade-off between efficiency and quality. PhoenixCodec addresses these challenges by alleviating the resource scattering of conventional decoders, employing CCR to escape local optima, and enhancing robustness through noisy-sample fine-tuning. In the LRAC 2025 Challenge Track 1, the proposed system ranked third overall and demonstrated the best performance at 1 kbps in both real-world noise and reverberation and intelligibility in clean tests, confirming its effectiveness.
>
---
#### [new 004] Are These Even Words? Quantifying the Gibberishness of Generative Speech Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对生成语音模型产生的荒谬语音（如无意义词）的评估问题，提出一种无需参考信号的无监督评估方法。通过语言模型量化语音内容的合理性，发布高质量合成荒谬语音数据集，并提供评分代码，旨在提升对生成语音中语义错误的检测能力。**

- **链接: [http://arxiv.org/pdf/2510.21317v1](http://arxiv.org/pdf/2510.21317v1)**

> **作者:** Danilo de Oliveira; Tal Peer; Jonas Rochdi; Timo Gerkmann
>
> **摘要:** Significant research efforts are currently being dedicated to non-intrusive quality and intelligibility assessment, especially given how it enables curation of large scale datasets of in-the-wild speech data. However, with the increasing capabilities of generative models to synthesize high quality speech, new types of artifacts become relevant, such as generative hallucinations. While intrusive metrics are able to spot such sort of discrepancies from a reference signal, it is not clear how current non-intrusive methods react to high-quality phoneme confusions or, more extremely, gibberish speech. In this paper we explore how to factor in this aspect under a fully unsupervised setting by leveraging language models. Additionally, we publish a dataset of high-quality synthesized gibberish speech for further development of measures to assess implausible sentences in spoken language, alongside code for calculating scores from a variety of speech language models.
>
---
#### [new 005] SpecTokenizer: A Lightweight Streaming Codec in the Compressed Spectrum Domain
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SpecTokenizer，一种轻量级流式音频编解码器，工作于压缩频谱域。针对现有轻量级神经音频编解码器计算与参数开销高的问题，通过交替使用CNN和RNN实现多尺度建模，在4kbps下仅需20%计算量和10%参数即可达到或超越当前最优性能。**

- **链接: [http://arxiv.org/pdf/2510.21209v1](http://arxiv.org/pdf/2510.21209v1)**

> **作者:** Zixiang Wan; Guochang Zhang; Yifeng He; Jianqiang Wei
>
> **备注:** Accepted by Interspeech 2025; 5 pages, 1 figure, 5 tables
>
> **摘要:** Neural Audio Codecs (NACs) have gained growing attention in recent years as technologies for audio compression and audio representation in speech language models. While mainstream NACs typically require G-level computation and M-level parameters, the performance of lightweight and streaming NACs remains underexplored. This paper proposes SpecTokenizer, a lightweight streaming codec that operates in the compressed spectral domain. Composed solely of alternating CNN and RNN layers, SpecTokenizer achieves greater efficiency and better representational capability through multi-scale modeling in the compressed spectrum domain. At 4 kbps, the proposed SpecTokenizer achieves comparable or superior performance compared to the codec with state-of-the-art lightweight architecture while requiring only 20% of the computation and 10% of the parameters. Furthermore, it significantly outperforms the codec when using similar computational and storage resources.
>
---
#### [new 006] Beyond Hearing: Learning Task-agnostic ExG Representations from Earphones via Physiology-informed Tokenization
- **分类: eess.AS; cs.CL; cs.SD; 68T01**

- **简介: 该论文针对日常场景下生理信号（ExG）建模中数据少、任务依赖性强的问题，提出基于耳塞设备采集自由活动数据，并设计生理启发的多频段分词（PiMT）方法，实现跨任务的通用表示学习。在新构建的DailySense数据集及多个公开基准上验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2510.20853v1](http://arxiv.org/pdf/2510.20853v1)**

> **作者:** Hyungjun Yoon; Seungjoo Lee; Yu Yvonne Wu; Xiaomeng Chen; Taiting Lu; Freddy Yifei Liu; Taeckyung Lee; Hyeongheon Cha; Haochen Zhao; Gaoteng Zhao; Sung-Ju Lee; Cecilia Mascolo; Dongyao Chen; Lili Qiu
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Electrophysiological (ExG) signals offer valuable insights into human physiology, yet building foundation models that generalize across everyday tasks remains challenging due to two key limitations: (i) insufficient data diversity, as most ExG recordings are collected in controlled labs with bulky, expensive devices; and (ii) task-specific model designs that require tailored processing (i.e., targeted frequency filters) and architectures, which limit generalization across tasks. To address these challenges, we introduce an approach for scalable, task-agnostic ExG monitoring in the wild. We collected 50 hours of unobtrusive free-living ExG data with an earphone-based hardware prototype to narrow the data diversity gap. At the core of our approach is Physiology-informed Multi-band Tokenization (PiMT), which decomposes ExG signals into 12 physiology-informed tokens, followed by a reconstruction task to learn robust representations. This enables adaptive feature recognition across the full frequency spectrum while capturing task-relevant information. Experiments on our new DailySense dataset-the first to enable ExG-based analysis across five human senses-together with four public ExG benchmarks, demonstrate that PiMT consistently outperforms state-of-the-art methods across diverse tasks.
>
---
#### [new 007] StylePitcher: Generating Style-Following and Expressive Pitch Curves for Versatile Singing Tasks
- **分类: cs.SD**

- **简介: 该论文提出StylePitcher，一个通用的音高曲线生成模型，旨在解决现有方法忽视歌手风格表达及任务泛化能力差的问题。通过学习参考音频中的演唱风格，结合乐谱与音高上下文，实现多样歌唱任务下的风格一致且精准的音高生成。**

- **链接: [http://arxiv.org/pdf/2510.21685v1](http://arxiv.org/pdf/2510.21685v1)**

> **作者:** Jingyue Huang; Qihui Yang; Fei Yueh Chen; Julian McAuley; Randal Leistikow; Perry R. Cook; Yongyi Zang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Existing pitch curve generators face two main challenges: they often neglect singer-specific expressiveness, reducing their ability to capture individual singing styles. And they are typically developed as auxiliary modules for specific tasks such as pitch correction, singing voice synthesis, or voice conversion, which restricts their generalization capability. We propose StylePitcher, a general-purpose pitch curve generator that learns singer style from reference audio while preserving alignment with the intended melody. Built upon a rectified flow matching architecture, StylePitcher flexibly incorporates symbolic music scores and pitch context as conditions for generation, and can seamlessly adapt to diverse singing tasks without retraining. Objective and subjective evaluations across various singing tasks demonstrate that StylePitcher improves style similarity and audio quality while maintaining pitch accuracy comparable to task-specific baselines.
>
---
#### [new 008] Smule Renaissance Small: Efficient General-Purpose Vocal Restoration
- **分类: cs.SD**

- **简介: 该论文提出Smule Renaissance Small（SRS），一种高效端到端的语音修复模型，针对消费设备录音中的噪声、混响、带宽限制和削波等多重退化问题。模型在复数短时傅里叶变换域工作，结合相位感知损失，实现高分辨率与低延迟。在真实极端退化数据集EDB上表现优异，超越多数开源方法，接近商业系统性能。**

- **链接: [http://arxiv.org/pdf/2510.21659v1](http://arxiv.org/pdf/2510.21659v1)**

> **作者:** Yongyi Zang; Chris Manchester; David Young; Ivan Ivanov; Jeffrey Lufkin; Martin Vladimirov; PJ Solomon; Svetoslav Kepchelev; Fei Yueh Chen; Dongting Cai; Teodor Naydenov; Randal Leistikow
>
> **备注:** Technical Report
>
> **摘要:** Vocal recordings on consumer devices commonly suffer from multiple concurrent degradations: noise, reverberation, band-limiting, and clipping. We present Smule Renaissance Small (SRS), a compact single-stage model that performs end-to-end vocal restoration directly in the complex STFT domain. By incorporating phase-aware losses, SRS enables large analysis windows for improved frequency resolution while achieving 10.5x real-time inference on iPhone 12 CPU at 48 kHz. On the DNS 5 Challenge blind set, despite no speech training, SRS outperforms a strong GAN baseline and closely matches a computationally expensive flow-matching system. To enable evaluation under realistic multi-degradation scenarios, we introduce the Extreme Degradation Bench (EDB): 87 singing and speech recordings captured under severe acoustic conditions. On EDB, SRS surpasses all open-source baselines on singing and matches commercial systems, while remaining competitive on speech despite no speech-specific training. We release both SRS and EDB under the MIT License.
>
---
#### [new 009] Robust Distortion-Free Watermark for Autoregressive Audio Generation Models
- **分类: cs.SD**

- **简介: 该论文针对自回归语音生成模型的版权保护问题，提出一种无失真水印技术Aligned-IS。针对传统方法因“重分词不匹配”导致检测失效的问题，利用聚类使同簇令牌等价，提升水印鲁棒性与可检测性，同时保持音频质量，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.21115v1](http://arxiv.org/pdf/2510.21115v1)**

> **作者:** Yihan Wu; Georgios Milis; Ruibo Chen; Heng Huang
>
> **摘要:** The rapid advancement of next-token-prediction models has led to widespread adoption across modalities, enabling the creation of realistic synthetic media. In the audio domain, while autoregressive speech models have propelled conversational interactions forward, the potential for misuse, such as impersonation in phishing schemes or crafting misleading speech recordings, has also increased. Security measures such as watermarking have thus become essential to ensuring the authenticity of digital media. Traditional statistical watermarking methods used for autoregressive language models face challenges when applied to autoregressive audio models, due to the inevitable ``retokenization mismatch'' - the discrepancy between original and retokenized discrete audio token sequences. To address this, we introduce Aligned-IS, a novel, distortion-free watermark, specifically crafted for audio generation models. This technique utilizes a clustering approach that treats tokens within the same cluster equivalently, effectively countering the retokenization mismatch issue. Our comprehensive testing on prevalent audio generation platforms demonstrates that Aligned-IS not only preserves the quality of generated audio but also significantly improves the watermark detectability compared to the state-of-the-art distortion-free watermarking adaptations, establishing a new benchmark in secure audio technology applications.
>
---
#### [new 010] refess-qi: reference-free evaluation for speech separation with joint quality and intelligibility scoring
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对语音分离任务，提出一种无需参考音频的评估框架（refess-qi），利用自监督学习表示联合预测音质（SI-SNR）与语音可懂度（WER）。解决了真实场景下无参考信号时无法评估的问题，实验表明其在WHAMR!数据集上表现稳健。**

- **链接: [http://arxiv.org/pdf/2510.21014v1](http://arxiv.org/pdf/2510.21014v1)**

> **作者:** Ari Frummer; Helin Wang; Tianyu Cao; Adi Arbel; Yuval Sieradzki; Oren Gal; Jesús Villalba; Thomas Thebaud; Najim Dehak
>
> **摘要:** Source separation is a crucial pre-processing step for various speech processing tasks, such as automatic speech recognition (ASR). Traditionally, the evaluation metrics for speech separation rely on the matched reference audios and corresponding transcriptions to assess audio quality and intelligibility. However, they cannot be used to evaluate real-world mixtures for which no reference exists. This paper introduces a text-free reference-free evaluation framework based on self-supervised learning (SSL) representations. The proposed framework utilize the mixture and separated tracks to predict jointly audio quality, through the Scale Invariant Signal to Noise Ratio (SI-SNR) metric, and speech intelligibility through the Word Error Rate (WER) metric. We conducted experiments on the WHAMR! dataset, which shows a WER estimation with a mean absolute error (MAE) of 17\% and a Pearson correlation coefficient (PCC) of 0.77; and SI-SNR estimation with an MAE of 1.38 and PCC of 0.95. We further demonstrate the robustness of our estimator by using various SSL representations.
>
---
#### [new 011] FlexIO: Flexible Single- and Multi-Channel Speech Separation and Enhancement
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出FlexIO，一种灵活的单/多通道语音分离与增强系统。针对现有方法在输入（麦克风数）和输出（说话人数）灵活性上的局限，通过提示向量实现任意数量说话人的条件分离，并引入无阵列依赖的通道通信机制，统一处理不同配置的多通道输入。实验验证其在1–5麦克风、1–3说话人场景及真实数据上的有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21485v1](http://arxiv.org/pdf/2510.21485v1)**

> **作者:** Yoshiki Masuyama; Kohei Saijo; Francesco Paissan; Jiangyu Han; Marc Delcroix; Ryo Aihara; François G. Germain; Gordon Wichern; Jonathan Le Roux
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech separation and enhancement (SSE) has advanced remarkably and achieved promising results in controlled settings, such as a fixed number of speakers and a fixed array configuration. Towards a universal SSE system, single-channel systems have been extended to deal with a variable number of speakers (i.e., outputs). Meanwhile, multi-channel systems accommodating various array configurations (i.e., inputs) have been developed. However, these attempts have been pursued separately. In this paper, we propose a flexible input and output SSE system, named FlexIO. It performs conditional separation using prompt vectors, one per speaker as a condition, allowing separation of an arbitrary number of speakers. Multi-channel mixtures are processed together with the prompt vectors via an array-agnostic channel communication mechanism. Our experiments demonstrate that FlexIO successfully covers diverse conditions with one to five microphones and one to three speakers. We also confirm the robustness of FlexIO on CHiME-4 real data.
>
---
#### [new 012] Compressing Quaternion Convolutional Neural Networks for Audio Classification
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文针对音频分类任务，解决量子卷积神经网络（QCNN）计算复杂度高、难以部署于资源受限设备的问题。通过剪枝技术压缩QCNN，显著降低参数量与计算成本，同时保持优异性能，在多个基准数据集上优于或媲美传统CNN与Transformer模型。**

- **链接: [http://arxiv.org/pdf/2510.21388v1](http://arxiv.org/pdf/2510.21388v1)**

> **作者:** Arshdeep Singh; Vinayak Abrol; Mark D. Plumbley
>
> **备注:** Under review in IEEE TASLPRO
>
> **摘要:** Conventional Convolutional Neural Networks (CNNs) in the real domain have been widely used for audio classification. However, their convolution operations process multi-channel inputs independently, limiting the ability to capture correlations among channels. This can lead to suboptimal feature learning, particularly for complex audio patterns such as multi-channel spectrogram representations. Quaternion Convolutional Neural Networks (QCNNs) address this limitation by employing quaternion algebra to jointly capture inter-channel dependencies, enabling more compact models with fewer learnable parameters while better exploiting the multi-dimensional nature of audio signals. However, QCNNs exhibit higher computational complexity due to the overhead of quaternion operations, resulting in increased inference latency and reduced efficiency compared to conventional CNNs, posing challenges for deployment on resource-constrained platforms. To address this challenge, this study explores knowledge distillation (KD) and pruning, to reduce the computational complexity of QCNNs while maintaining performance. Our experiments on audio classification reveal that pruning QCNNs achieves similar or superior performance compared to KD while requiring less computational effort. Compared to conventional CNNs and Transformer-based architectures, pruned QCNNs achieve competitive performance with a reduced learnable parameter count and computational complexity. On the AudioSet dataset, pruned QCNNs reduce computational cost by 50\% and parameter count by 80\%, while maintaining performance comparable to the conventional CNNs. Furthermore, pruned QCNNs generalize well across multiple audio classification benchmarks, including GTZAN for music genre recognition, ESC-50 for environmental sound classification and RAVDESS for speech emotion recognition.
>
---
#### [new 013] Data-Centric Lessons To Improve Speech-Language Pretraining
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文聚焦语音-语言预训练任务，针对现有模型性能提升机制不明确的问题，通过受控实验探究音频数据处理、合成数据构建与序列交织策略。基于发现，构建3.8B参数模型SpeLangy，显著优于更大模型，凸显数据质量对语音语言模型的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.20860v1](http://arxiv.org/pdf/2510.20860v1)**

> **作者:** Vishaal Udandarao; Zhiyun Lu; Xuankai Chang; Yongqiang Wang; Violet Z. Yao; Albin Madapally Jose; Fartash Faghri; Josh Gardner; Chung-Cheng Chiu
>
> **备注:** Tech Report
>
> **摘要:** Spoken Question-Answering (SQA) is a core capability for useful and interactive artificial intelligence systems. Recently, several speech-language models (SpeechLMs) have been released with a specific focus on improving their SQA performance. However, a lack of controlled ablations of pretraining data processing and curation makes it challenging to understand what factors account for performance, despite substantial gains from similar studies in other data modalities. In this work, we address this gap by conducting a data-centric exploration for pretraining SpeechLMs. We focus on three research questions fundamental to speech-language pretraining data: (1) how to process raw web-crawled audio content for speech-text pretraining, (2) how to construct synthetic pretraining datasets to augment web-crawled data and (3) how to interleave (text, audio) segments into training sequences. We apply the insights from our controlled data-centric ablations to pretrain a 3.8B-parameter SpeechLM, called SpeLangy, that outperforms models that are up to 3x larger by 10.2% absolute performance. We hope our findings highlight the impact of effective data curation for speech-language pretraining and guide future data-centric exploration in SpeechLMs.
>
---
#### [new 014] HiFi-HARP: A High-Fidelity 7th-Order Ambisonic Room Impulse Response Dataset
- **分类: cs.SD**

- **简介: 该论文提出HiFi-HARP，一个7阶全向声学脉冲响应（HOA-RIR）数据集，用于高保真空间音频研究。针对真实复杂室内场景中高精度声学模拟难题，结合几何建模与混合波/射线仿真，生成超10万条7阶Ambisonic RIRs，支持机器学习与音频算法开发。**

- **链接: [http://arxiv.org/pdf/2510.21257v1](http://arxiv.org/pdf/2510.21257v1)**

> **作者:** Shivam Saini; Jürgen Peissig
>
> **备注:** Under review for ICASSP 2026
>
> **摘要:** We introduce HiFi-HARP, a large-scale dataset of 7th-order Higher-Order Ambisonic Room Impulse Responses (HOA-RIRs) consisting of more than 100,000 RIRs generated via a hybrid acoustic simulation in realistic indoor scenes. HiFi-HARP combines geometrically complex, furnished room models from the 3D-FRONT repository with a hybrid simulation pipeline: low-frequency wave-based simulation (finite-difference time-domain) up to 900 Hz is used, while high frequencies above 900 Hz are simulated using a ray-tracing approach. The combined raw RIRs are encoded into the spherical-harmonic domain (AmbiX ACN) for direct auralization. Our dataset extends prior work by providing 7th-order Ambisonic RIRs that combine wave-theoretic accuracy with realistic room content. We detail the generation pipeline (scene and material selection, array design, hybrid simulation, ambisonic encoding) and provide dataset statistics (room volumes, RT60 distributions, absorption properties). A comparison table highlights the novelty of HiFi-HARP relative to existing RIR collections. Finally, we outline potential benchmarks such as FOA-to-HOA upsampling, source localization, and dereverberation. We discuss machine learning use cases (spatial audio rendering, acoustic parameter estimation) and limitations (e.g., simulation approximations, static scenes). Overall, HiFi-HARP offers a rich resource for developing spatial audio and acoustics algorithms in complex environments.
>
---
#### [new 015] FlowSynth: Instrument Generation Through Distributional Flow Matching and Test-Time Search
- **分类: cs.SD**

- **简介: 该论文针对虚拟乐器合成中音高与力度下音色一致性难题，提出FlowSynth方法。通过分布流匹配建模预测不确定性，并结合测试时搜索优化，提升单音质量与跨音符音色一致性，显著优于现有基线，为实时专业级乐器合成提供新路径。**

- **链接: [http://arxiv.org/pdf/2510.21667v1](http://arxiv.org/pdf/2510.21667v1)**

> **作者:** Qihui Yang; Randal Leistikow; Yongyi Zang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Virtual instrument generation requires maintaining consistent timbre across different pitches and velocities, a challenge that existing note-level models struggle to address. We present FlowSynth, which combines distributional flow matching (DFM) with test-time optimization for high-quality instrument synthesis. Unlike standard flow matching that learns deterministic mappings, DFM parameterizes the velocity field as a Gaussian distribution and optimizes via negative log-likelihood, enabling the model to express uncertainty in its predictions. This probabilistic formulation allows principled test-time search: we sample multiple trajectories weighted by model confidence and select outputs that maximize timbre consistency. FlowSynth outperforms the current state-of-the-art TokenSynth baseline in both single-note quality and cross-note consistency. Our approach demonstrates that modeling predictive uncertainty in flow matching, combined with music-specific consistency objectives, provides an effective path to professional-quality virtual instruments suitable for real-time performance.
>
---
#### [new 016] Foley Control: Aligning a Frozen Latent Text-to-Audio Model to Video
- **分类: cs.CV; cs.SD**

- **简介: 该论文提出Foley Control，用于视频引导的音效生成任务。针对现有方法需重训练模型、参数多的问题，提出冻结预训练音视频模型，仅添加轻量级交叉注意力桥接，实现高效同步。通过视频控制音频时序与细节，保留文本语义控制，支持模块灵活替换，显著减少参数并提升实用性。**

- **链接: [http://arxiv.org/pdf/2510.21581v1](http://arxiv.org/pdf/2510.21581v1)**

> **作者:** Ciara Rowles; Varun Jampani; Simon Donné; Shimon Vainer; Julian Parker; Zach Evans
>
> **备注:** Project Page: https://stability-ai.github.io/foleycontrol.github.io/
>
> **摘要:** Foley Control is a lightweight approach to video-guided Foley that keeps pretrained single-modality models frozen and learns only a small cross-attention bridge between them. We connect V-JEPA2 video embeddings to a frozen Stable Audio Open DiT text-to-audio (T2A) model by inserting compact video cross-attention after the model's existing text cross-attention, so prompts set global semantics while video refines timing and local dynamics. The frozen backbones retain strong marginals (video; audio given text) and the bridge learns the audio-video dependency needed for synchronization -- without retraining the audio prior. To cut memory and stabilize training, we pool video tokens before conditioning. On curated video-audio benchmarks, Foley Control delivers competitive temporal and semantic alignment with far fewer trainable parameters than recent multi-modal systems, while preserving prompt-driven controllability and production-friendly modularity (swap/upgrade encoders or the T2A backbone without end-to-end retraining). Although we focus on Video-to-Foley, the same bridge design can potentially extend to other audio modalities (e.g., speech).
>
---
#### [new 017] Can Current Detectors Catch Face-to-Voice Deepfake Attacks?
- **分类: cs.CR; cs.LG; cs.MM; cs.SD**

- **简介: 该论文研究音频深度伪造检测任务，针对FOICE攻击——仅用人脸图像生成逼真语音的问题。研究发现现有检测器在干净与噪声环境下均失效；提出针对性微调策略提升检测效果，但存在对新生成器泛化能力下降的权衡。揭示了当前防御体系的根本缺陷。**

- **链接: [http://arxiv.org/pdf/2510.21004v1](http://arxiv.org/pdf/2510.21004v1)**

> **作者:** Nguyen Linh Bao Nguyen; Alsharif Abuadbba; Kristen Moore; Tingming Wu
>
> **备注:** 8 pages, Accepted at Workshop on AI for Cyber Threat Intelligence, co-located with ACSAC 2025
>
> **摘要:** The rapid advancement of generative models has enabled the creation of increasingly stealthy synthetic voices, commonly referred to as audio deepfakes. A recent technique, FOICE [USENIX'24], demonstrates a particularly alarming capability: generating a victim's voice from a single facial image, without requiring any voice sample. By exploiting correlations between facial and vocal features, FOICE produces synthetic voices realistic enough to bypass industry-standard authentication systems, including WeChat Voiceprint and Microsoft Azure. This raises serious security concerns, as facial images are far easier for adversaries to obtain than voice samples, dramatically lowering the barrier to large-scale attacks. In this work, we investigate two core research questions: (RQ1) can state-of-the-art audio deepfake detectors reliably detect FOICE-generated speech under clean and noisy conditions, and (RQ2) whether fine-tuning these detectors on FOICE data improves detection without overfitting, thereby preserving robustness to unseen voice generators such as SpeechT5. Our study makes three contributions. First, we present the first systematic evaluation of FOICE detection, showing that leading detectors consistently fail under both standard and noisy conditions. Second, we introduce targeted fine-tuning strategies that capture FOICE-specific artifacts, yielding significant accuracy improvements. Third, we assess generalization after fine-tuning, revealing trade-offs between specialization to FOICE and robustness to unseen synthesis pipelines. These findings expose fundamental weaknesses in today's defenses and motivate new architectures and training protocols for next-generation audio deepfake detection.
>
---
## 更新

#### [replaced 001] Speaker Disentanglement of Speech Pre-trained Model Based on Interpretability
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17851v2](http://arxiv.org/pdf/2507.17851v2)**

> **作者:** Xiaoxu Zhu; Junhua Li; Aaron J. Li; Yiming Ren; Baoxiang Li
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** Self-supervised speech models learn representations that capture both content and speaker information. Yet this entanglement creates problems: content tasks suffer from speaker bias, and privacy concerns arise when speaker identity leaks through supposedly anonymized representations. We present two contributions to address these challenges. First, we develop InterpTRQE-SptME (Timbre Residual Quantitative Evaluation Benchmark of Speech pre-training Models Encoding via Interpretability), a benchmark that directly measures residual speaker information in content embeddings using SHAP-based interpretability analysis. Unlike existing indirect metrics, our approach quantifies the exact proportion of speaker information remaining after disentanglement. Second, we propose InterpTF-SptME, which uses these interpretability insights to filter speaker information from embeddings. Testing on VCTK with seven models including HuBERT, WavLM, and ContentVec, we find that SHAP Noise filtering reduces speaker residuals from 18.05% to nearly zero while maintaining recognition accuracy (CTC loss increase under 1%). The method is model-agnostic and requires no retraining.
>
---
#### [replaced 002] Robust Residual Finite Scalar Quantization for Neural Compression
- **分类: eess.IV; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.15860v2](http://arxiv.org/pdf/2508.15860v2)**

> **作者:** Xiaoxu Zhu; Jiakui Li; Ken Zheng; Guiping Zhong; Huimeng Wang; Shiyin Kang; Dahua Lin
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Finite Scalar Quantization (FSQ) offers simplified training but suffers from residual magnitude decay in multi-stage settings, where subsequent stages receive exponentially weaker signals. We propose Robust Residual Finite Scalar Quantization (RFSQ), addressing this fundamental limitation through two novel conditioning strategies: learnable scaling factors and invertible layer normalization. Our experiments across audio and image modalities demonstrate RFSQ's effectiveness and generalizability. In audio reconstruction at 24 bits/frame, RFSQ-LayerNorm achieves 3.646 DNSMOS, a 3.6% improvement over state-of-the-art RVQ (3.518). On ImageNet, RFSQ achieves 0.102 L1 loss and 0.100 perceptual loss, with LayerNorm providing 9.7% L1 improvement and 17.4% perceptual improvement over unconditioned variants. The LayerNorm strategy consistently outperforms alternatives by maintaining normalized input statistics across stages, effectively preventing exponential magnitude decay that limits naive residual approaches. RFSQ combines FSQ's simplicity with multi-stage quantization's representational power, establishing a new standard for neural compression across diverse modalities.
>
---
#### [replaced 003] Visual Cues Support Robust Turn-taking Prediction in Noise
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22088v2](http://arxiv.org/pdf/2505.22088v2)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** Accepted to INTERSPEECH 2025, 10.21437/Interspeech.2025-668
>
> **摘要:** Accurate predictive turn-taking models (PTTMs) are essential for naturalistic human-robot interaction. However, little is known about their performance in noise. This study therefore explores PTTM performance in types of noise likely to be encountered once deployed. Our analyses reveal PTTMs are highly sensitive to noise. Hold/shift accuracy drops from 84% in clean speech to just 52% in 10 dB music noise. Training with noisy data enables a multimodal PTTM, which includes visual features to better exploit visual cues, with 72% accuracy in 10 dB music noise. The multimodal PTTM outperforms the audio-only PTTM across all noise types and SNRs, highlighting its ability to exploit visual cues; however, this does not always generalise to new types of noise. Analysis also reveals that successful training relies on accurate transcription, limiting the use of ASR-derived transcriptions to clean conditions. We make code publicly available for future research.
>
---
#### [replaced 004] LipDiffuser: Lip-to-Speech Generation with Conditional Diffusion Models
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.11391v3](http://arxiv.org/pdf/2505.11391v3)**

> **作者:** Julius Richter; Danilo de Oliveira; Tal Peer; Timo Gerkmann
>
> **摘要:** We present LipDiffuser, a conditional diffusion model for lip-to-speech generation synthesizing natural and intelligible speech directly from silent video recordings. Our approach leverages the magnitude-preserving ablated diffusion model (MP-ADM) architecture as a denoiser model. To effectively condition the model, we incorporate visual features using magnitude-preserving feature-wise linear modulation (MP-FiLM) alongside speaker embeddings. A neural vocoder then reconstructs the speech waveform from the generated mel-spectrograms. Evaluations on LRS3 demonstrate that LipDiffuser outperforms existing lip-to-speech baselines in perceptual speech quality and speaker similarity, while remaining competitive in downstream automatic speech recognition. These findings are also supported by a formal listening experiment.
>
---
#### [replaced 005] Continuous-Token Diffusion for Speaker-Referenced TTS in Multimodal LLMs
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.12995v2](http://arxiv.org/pdf/2510.12995v2)**

> **作者:** Xinlu He; Swayambhu Nath Ray; Harish Mallidi; Jia-Hong Huang; Ashwin Bellur; Chander Chandak; M. Maruf; Venkatesh Ravichandran
>
> **摘要:** Unified architectures in multimodal large language models (MLLM) have shown promise in handling diverse tasks within a single framework. In the text-to-speech (TTS) task, current MLLM-based approaches rely on discrete token representations, which disregard the inherently continuous nature of speech and can lead to loss of fine-grained acoustic information. In this work, we investigate the TTS within the MLLM paradigm using continuous speech representations. We design a dual-head architecture and implement two complementary training strategies for a robust model. (1) A diffusion head generating continuous speech representations is added on the MLLM, which is on frame-level and strictly autoregressive. (2) The original language model head is retained to preserve multitask capability and to control the start and end of speech synthesis. (3) Masked training is employed to address exposure bias in autoregressive decoding. (4) To stabilize optimization, we propose a two-stage scheme where the LM is frozen in the second stage, ensuring the diffusion head learns from a fixed input distribution. Evaluations on LibriSpeech(PC) test-clean show that our approach achieves state-of-the-art autoregressive performance, with a WER of 1.95%, speaker similarity of 0.54, and UTMOS of 4.00. The two-stage training yields a 46% relative WER reduction over the one-stage training baseline. These results highlight the effectiveness of combining autoregressive modeling with continuous-token diffusion, supported by a two-stage training procedure.
>
---
#### [replaced 006] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.01957v4](http://arxiv.org/pdf/2501.01957v4)**

> **作者:** Chaoyou Fu; Haojia Lin; Xiong Wang; Yi-Fan Zhang; Yunhang Shen; Xiaoyu Liu; Haoyu Cao; Zuwei Long; Heting Gao; Ke Li; Long Ma; Xiawu Zheng; Rongrong Ji; Xing Sun; Caifeng Shan; Ran He
>
> **备注:** NeurIPS 2025 Spotlight, Code 2.4K Stars: https://github.com/VITA-MLLM/VITA
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) have typically focused on integrating visual and textual modalities, with less emphasis placed on the role of speech in enhancing interaction. However, speech plays a crucial role in multimodal dialogue systems, and implementing high-performance in both vision and speech tasks remains a significant challenge due to the fundamental modality differences. In this paper, we propose a carefully designed multi-stage training methodology that progressively trains LLM to understand both visual and speech information, ultimately enabling fluent vision and speech interaction. Our approach not only preserves strong vision-language capacity, but also enables efficient speech-to-speech dialogue capabilities without separate ASR and TTS modules, significantly accelerating multimodal end-to-end response speed. By comparing our method against state-of-the-art counterparts across benchmarks for image, video, and speech tasks, we demonstrate that our model is equipped with both strong visual and speech capabilities, making near real-time vision and speech interaction. Code has been released at https://github.com/VITA-MLLM/VITA.
>
---
#### [replaced 007] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11217v2](http://arxiv.org/pdf/2505.11217v2)**

> **作者:** Yanhao Jia; Ji Xie; S Jivaganesh; Hao Li; Xu Wu; Mengmi Zhang
>
> **备注:** NeurIPS 2025, Spotlight
>
> **摘要:** Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we propose a neuroscience-inspired model, EchoPin, which uses a stereo audio-image dataset generated via 3D simulations. Even with limited training data, EchoPin surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy.
>
---
#### [replaced 008] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13181v2](http://arxiv.org/pdf/2505.13181v2)**

> **作者:** Zhengrui Ma; Yang Feng; Chenze Shao; Fandong Meng; Jie Zhou; Min Zhang
>
> **备注:** NeurIPS 2025; Demos and code are available at https://github.com/ictnlp/SLED-TTS
>
> **摘要:** We introduce SLED, an alternative approach to speech language modeling by encoding speech waveforms into sequences of continuous latent representations and modeling them autoregressively using an energy distance objective. The energy distance offers an analytical measure of the distributional gap by contrasting simulated and target samples, enabling efficient training to capture the underlying continuous autoregressive distribution. By bypassing reliance on residual vector quantization, SLED avoids discretization errors and eliminates the need for the complicated hierarchical architectures common in existing speech language models. It simplifies the overall modeling pipeline while preserving the richness of speech information and maintaining inference efficiency. Empirical results demonstrate that SLED achieves strong performance in both zero-shot and streaming speech synthesis, showing its potential for broader applications in general-purpose speech language models.
>
---
#### [replaced 009] Variational autoencoders stabilise TCN performance when classifying weakly labelled bioacoustics data: an interdisciplinary approach
- **分类: cs.SD; eess.AS; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2410.17006v2](http://arxiv.org/pdf/2410.17006v2)**

> **作者:** Laia Garrobé Fonollosa; Douglas Gillespie; Lina Stankovic; Vladimir Stankovic; Luke Rendell
>
> **摘要:** Passive acoustic monitoring (PAM) data is often weakly labelled, audited at the scale of detection presence or absence on timescales of minutes to hours. Moreover, this data exhibits great variability from one deployment to the next, due to differences in ambient noise and the signals across sources and geographies. This study proposes a two-step solution to leverage weakly annotated data for training Deep Learning (DL) detection models. Our case study involves binary classification of the presence/absence of sperm whale (\textit{Physeter macrocephalus}) click trains in 4-minute-long recordings from a dataset comprising diverse sources and deployment conditions to maximise generalisability. We tested methods for extracting acoustic features from lengthy audio segments and integrated Temporal Convolutional Networks (TCNs) trained on the extracted features for sequence classification. For feature extraction, we introduced a new approach using Variational AutoEncoders (VAEs) to extract information from both waveforms and spectrograms, which eliminates the necessity for manual threshold setting or time-consuming strong labelling. For classification, TCNs were trained separately on sequences of either VAE embeddings or handpicked acoustic features extracted from the waveform and spectrogram representations using classical methods, to compare the efficacy of the two approaches. The TCN demonstrated robust classification capabilities on a validation set, achieving accuracies exceeding 85\% when applied to 4-minute acoustic recordings. Notably, TCNs trained on handpicked acoustic features exhibited greater variability in performance across recordings from diverse deployment conditions, whereas those trained on VAEs showed a more consistent performance, highlighting the robust transferability of VAEs for feature extraction across different deployment conditions.
>
---
