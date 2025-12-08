# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] A Multi-Channel Auditory Signal Encoder with Adaptive Resolution Using Volatile Memristors
- **分类: eess.AS**

- **简介: 该论文提出一种基于易失性忆阻器的多通道听觉信号编码器，解决传统编码能耗高、动态适应性差的问题。通过CMOS-忆阻器混合电路实现自适应阈值的异步 delta 调制，增强信号起始响应，提升编码效率与时间精度。**

- **链接: [https://arxiv.org/pdf/2512.05701v1](https://arxiv.org/pdf/2512.05701v1)**

> **作者:** Dongxu Guo; Deepika Yadav; Patrick Foster; Spyros Stathopoulos; Mingyi Chen; Themis Prodromakis; Shiwei Wang
>
> **备注:** 11 pages, 17 figures, submitted to IEEE Transactions on Circuits and Systems I: Regular Papers for possible publications
>
> **摘要:** We demonstrate and experimentally validate an end-to-end hybrid CMOS-memristor auditory encoder that realises adaptive-threshold, asynchronous delta-modulation (ADM)-based spike encoding by exploiting the inherent volatility of HfTiOx devices. A spike-triggered programming pulse rapidly raises the ADM threshold Delta (desensitisation); the device's volatility then passively lowers Delta when activity subsides (resensitisation), emphasising onsets while restoring sensitivity without static control energy. Our prototype couples an 8-channel 130 nm encoder IC to off-chip HfTiOx devices via a switch interface and an off-chip controller that monitors spike activity and issues programming events. An on-chip current-mirror transimpedance amplifier (TIA) converts device current into symmetric thresholds, enabling both sensitive and conservative encoding regimes. Evaluated with gammatone-filtered speech, the adaptive loop-at matched spike budget-sharpens onsets and preserves fine temporal detail that a fixed-Delta baseline misses; multi-channel spike cochleagrams show the same trend. Together, these results establish a practical hybrid CMOS-memristor pathway to onset-salient, spike-efficient neuromorphic audio front-ends and motivate low-power single-chip integration.
>
---
#### [new 002] Speech World Model: Causal State-Action Planning with Explicit Reasoning for Speech
- **分类: eess.AS**

- **简介: 该论文提出语音世界模型，面向语音理解中的弱推理问题，通过模块化因果图分解语音状态与动作，实现可解释的显式推理。属语音语言建模任务，解决了传统模型在稀疏监督下推理能力差的问题。**

- **链接: [https://arxiv.org/pdf/2512.05933v1](https://arxiv.org/pdf/2512.05933v1)**

> **作者:** Xuanru Zhou; Jiachen Lian; Henry Hong; Xinyi Yang; Gopala Anumanchipalli
>
> **摘要:** Current speech-language models (SLMs) typically use a cascade of speech encoder and large language model, treating speech understanding as a single black box. They analyze the content of speech well but reason weakly about other aspects, especially under sparse supervision. Thus, we argue for explicit reasoning over speech states and actions with modular and transparent decisions. Inspired by cognitive science we adopt a modular perspective and a world model view in which the system learns forward dynamics over latent states. We factorize speech understanding into four modules that communicate through a causal graph, establishing a cognitive state search space. Guided by posterior traces from this space, an instruction-tuned language model produces a concise causal analysis and a user-facing response, enabling counterfactual interventions and interpretability under partial supervision. We present the first graph based modular speech model for explicit reasoning and we will open source the model and data to promote the development of advanced speech understanding.
>
---
#### [new 003] Lyrics Matter: Exploiting the Power of Learnt Representations for Music Popularity Prediction
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属音乐流行度预测任务，旨在解决歌词信息利用不足的问题。作者提出自动化LLM歌词特征提取管道LyricsAENet，并构建多模态模型HitMusicLyricNet，融合音频、歌词与社交元数据，显著提升预测性能。**

- **链接: [https://arxiv.org/pdf/2512.05508v1](https://arxiv.org/pdf/2512.05508v1)**

> **作者:** Yash Choudhary; Preeti Rao; Pushpak Bhattacharyya
>
> **备注:** 8 pages
>
> **摘要:** Accurately predicting music popularity is a critical challenge in the music industry, offering benefits to artists, producers, and streaming platforms. Prior research has largely focused on audio features, social metadata, or model architectures. This work addresses the under-explored role of lyrics in predicting popularity. We present an automated pipeline that uses LLM to extract high-dimensional lyric embeddings, capturing semantic, syntactic, and sequential information. These features are integrated into HitMusicLyricNet, a multimodal architecture that combines audio, lyrics, and social metadata for popularity score prediction in the range 0-100. Our method outperforms existing baselines on the SpotGenTrack dataset, which contains over 100,000 tracks, achieving 9% and 20% improvements in MAE and MSE, respectively. Ablation confirms that gains arise from our LLM-driven lyrics feature pipeline (LyricsAENet), underscoring the value of dense lyric representations.
>
---
#### [new 004] The T12 System for AudioMOS Challenge 2025: Audio Aesthetics Score Prediction System Using KAN- and VERSA-based Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频美学评分（AES）预测任务，提出T12系统，结合KAN和VERSA模型预测音频在多个维度的美学得分。通过改进网络结构与集成学习，提升评分与主观感知的一致性，在多个评测级别上取得最优相关性。**

- **链接: [https://arxiv.org/pdf/2512.05592v1](https://arxiv.org/pdf/2512.05592v1)**

> **作者:** Katsuhiko Yamamoto; Koichi Miyazaki; Shogo Seki
>
> **备注:** Accepted by IEEE ASRU 2025
>
> **摘要:** We propose an audio aesthetics score (AES) prediction system by CyberAgent (AESCA) for AudioMOS Challenge 2025 (AMC25) Track 2. The AESCA comprises a Kolmogorov--Arnold Network (KAN)-based audiobox aesthetics and a predictor from the metric scores using the VERSA toolkit. In the KAN-based predictor, we replaced each multi-layer perceptron layer in the baseline model with a group-rational KAN and trained the model with labeled and pseudo-labeled audio samples. The VERSA-based predictor was designed as a regression model using extreme gradient boosting, incorporating outputs from existing metrics. Both the KAN- and VERSA-based models predicted the AES, including the four evaluation axes. The final AES values were calculated using an ensemble model that combined four KAN-based models and a VERSA-based model. Our proposed T12 system yielded the best correlations among the submitted systems, in three axes at the utterance level, two axes at the system level, and the overall average.
>
---
#### [new 005] SyncVoice: Towards Video Dubbing with Vision-Augmented Pretrained TTS Model
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.MM; cs.SD**

- **简介: 该论文研究视频配音任务，旨在解决现有方法在语音自然度、音画同步及多语言支持上的不足。作者提出SyncVoice框架，基于预训练TTS模型引入视觉信息并设计双说话人编码器，提升跨语言合成效果与音画一致性。**

- **链接: [https://arxiv.org/pdf/2512.05126v1](https://arxiv.org/pdf/2512.05126v1)**

> **作者:** Kaidi Wang; Yi He; Wenhao Guan; Weijie Wu; Hongwu Ding; Xiong Zhang; Di Wu; Meng Meng; Jian Luan; Lin Li; Qingyang Hong
>
> **摘要:** Video dubbing aims to generate high-fidelity speech that is precisely temporally aligned with the visual content. Existing methods still suffer from limitations in speech naturalness and audio-visual synchronization, and are limited to monolingual settings. To address these challenges, we propose SyncVoice, a vision-augmented video dubbing framework built upon a pretrained text-to-speech (TTS) model. By fine-tuning the TTS model on audio-visual data, we achieve strong audiovisual consistency. We propose a Dual Speaker Encoder to effectively mitigate inter-language interference in cross-lingual speech synthesis and explore the application of video dubbing in video translation scenarios. Experimental results show that SyncVoice achieves high-fidelity speech generation with strong synchronization performance, demonstrating its potential in video dubbing tasks.
>
---
#### [new 006] Decoding Selective Auditory Attention to Musical Elements in Ecologically Valid Music Listening
- **分类: q-bio.NC; cs.LG; cs.SD; eess.AS; eess.SP**

- **简介: 该论文研究自然音乐聆听中对音乐元素的选择性注意解码，旨在解决缺乏客观量化听觉注意力工具的问题。使用四通道消费级EEG设备，实现对真实音乐场景下脑响应的解码，验证了跨歌曲与被试的可行性，并提升了模型性能。**

- **链接: [https://arxiv.org/pdf/2512.05528v1](https://arxiv.org/pdf/2512.05528v1)**

> **作者:** Taketo Akama; Zhuohao Zhang; Tsukasa Nagashima; Takagi Yutaka; Shun Minamikawa; Natalia Polouliakh
>
> **摘要:** Art has long played a profound role in shaping human emotion, cognition, and behavior. While visual arts such as painting and architecture have been studied through eye tracking, revealing distinct gaze patterns between experts and novices, analogous methods for auditory art forms remain underdeveloped. Music, despite being a pervasive component of modern life and culture, still lacks objective tools to quantify listeners' attention and perceptual focus during natural listening experiences. To our knowledge, this is the first attempt to decode selective attention to musical elements using naturalistic, studio-produced songs and a lightweight consumer-grade EEG device with only four electrodes. By analyzing neural responses during real world like music listening, we test whether decoding is feasible under conditions that minimize participant burden and preserve the authenticity of the musical experience. Our contributions are fourfold: (i) decoding music attention in real studio-produced songs, (ii) demonstrating feasibility with a four-channel consumer EEG, (iii) providing insights for music attention decoding, and (iv) demonstrating improved model ability over prior work. Our findings suggest that musical attention can be decoded not only for novel songs but also across new subjects, showing performance improvements compared to existing approaches under our tested conditions. These findings show that consumer-grade devices can reliably capture signals, and that neural decoding in music could be feasible in real-world settings. This paves the way for applications in education, personalized music technologies, and therapeutic interventions.
>
---
#### [new 007] Noise Suppression for Time Difference of Arrival: Performance Evaluation of a Generalized Cross-Correlation Method Using Mean Signal and Inverse Filter
- **分类: eess.SP; eess.AS**

- **简介: 该论文研究噪声环境下到达时间差（TDOA）估计任务，旨在提升低信噪比和未知带宽时的估计精度。提出GCC-MSIF方法，利用多通道均值信号与逆滤波器抑制带外噪声，仿真表明其性能优于传统方法，且随阵元数增加而提升。**

- **链接: [https://arxiv.org/pdf/2512.05355v1](https://arxiv.org/pdf/2512.05355v1)**

> **作者:** Hirotaka Obo; Yuki Fujita; Masahisa Ishii; Hideki Moriyama; Ryota Tsuchiya; Yuta Ohashi; Kotaro Seki
>
> **摘要:** This paper proposes a novel generalized cross-correlation (GCC) method, termed GCC-MSIF, to improve time difference of arrival (TDOA) estimation accuracy in noisy environments. Conventional GCC methods often suffer from performance degradation under low signal-to-noise ratio (SNR) conditions, particularly when the signal bandwidth is unknown. GCC-MSIF introduces a "mean signal" estimated from multi-channel inputs and an "inverse filter" to virtually reconstruct the source signal, enabling adaptive suppression of out-of-band noise. Numerical simulations simulating a small-scale array demonstrate that GCC-MSIF significantly outperforms conventional methods, such as GCC-PHAT and GCC-SCOT, in low SNR regions and achieves robustness comparable to or exceeding the maximum likelihood (GCC-ML) method. Furthermore, the estimation accuracy improves scalably with the number of array elements. These results suggest that GCC-MSIF is a promising solution for robust passive localization in practical blind environments.
>
---
#### [new 008] MuMeNet: A Network Simulator for Musical Metaverse Communications
- **分类: cs.NI; cs.SD**

- **简介: 该论文针对音乐元宇宙（MM）中网络服务提供问题，提出一种面向5G/6G的新型网络仿真器MuMeNet。通过建模MM会话的服务与网络图，设计适配其交互性、异构性和组播特性的仿真框架，并验证了编排策略在真实工作负载下的性能。**

- **链接: [https://arxiv.org/pdf/2512.05201v1](https://arxiv.org/pdf/2512.05201v1)**

> **作者:** Ali Al Housseini; Jaime Llorca; Luca Turchet; Tiziano Leidi; Cristina Rottondi; Omran Ayoub
>
> **备注:** To be published in 2025 IEEE 6th International Symposium on the Internet of Sounds (IS2)
>
> **摘要:** The Metaverse, a shared and spatially organized digital continuum, is transforming various industries, with music emerging as a leading use case. Live concerts, collaborative composition, and interactive experiences are driving the Musical Metaverse (MM), but the requirements of the underlying network and service infrastructures hinder its growth. These challenges underscore the need for a novel modeling and simulation paradigm tailored to the unique characteristics of MM sessions, along with specialized service provisioning strategies capable of capturing their interactive, heterogeneous, and multicast-oriented nature. To this end, we make a first attempt to formally model and analyze the problem of service provisioning for MM sessions in 5G/6G networks. We first formalize service and network graph models for the MM, using "live audience interaction in a virtual concert" as a reference scenario. We then present MuMeNet, a novel discrete-event network simulator specifically tailored to the requirements and the traffic dynamics of the MM. We showcase the effectiveness of MuMeNet by running a linear programming based orchestration policy on the reference scenario and providing performance analysis under realistic MM workloads.
>
---
## 更新

#### [replaced 001] Wavehax: Aliasing-Free Neural Waveform Synthesis Based on 2D Convolution and Harmonic Prior for Reliable Complex Spectrogram Estimation
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究神经语音合成中的混叠问题，旨在提升高基频语音的生成质量。提出Wavehax模型，结合2D卷积与谐波先验，实现无混叠、高效且鲁棒的波形合成，在降低计算成本的同时保持高保真音质。**

- **链接: [https://arxiv.org/pdf/2411.06807v2](https://arxiv.org/pdf/2411.06807v2)**

> **作者:** Reo Yoneyama; Atsushi Miyashita; Ryuichi Yamamoto; Tomoki Toda
>
> **备注:** 13 pages, 5 figures. A peer-reviewed and revised version of this work has been accepted for publication in IEEE TASLP and is available as open access (https://ieeexplore.ieee.org/document/11216102). The accepted paper includes more solid discussions and additional experiments that are not reflected in this preprint
>
> **摘要:** Neural vocoders often struggle with aliasing in latent feature spaces, caused by time-domain nonlinear operations and resampling layers. Aliasing folds high-frequency components into the low-frequency range, making aliased and original frequency components indistinguishable and introducing two practical issues. First, aliasing complicates the waveform generation process, as the subsequent layers must address these aliasing effects, increasing the computational complexity. Second, it limits extrapolation performance, particularly in handling high fundamental frequencies, which degrades the perceptual quality of generated speech waveforms. This paper demonstrates that 1) time-domain nonlinear operations inevitably introduce aliasing but provide a strong inductive bias for harmonic generation, and 2) time-frequency-domain processing can achieve aliasing-free waveform synthesis but lacks the inductive bias for effective harmonic generation. Building on this insight, we propose Wavehax, an aliasing-free neural WAVEform generator that integrates 2D convolution and a HArmonic prior for reliable Complex Spectrogram estimation. Experimental results show that Wavehax achieves speech quality comparable to existing high-fidelity neural vocoders and exhibits exceptional robustness in scenarios requiring high fundamental frequency extrapolation, where aliasing effects become typically severe. Moreover, Wavehax requires less than 5% of the multiply-accumulate operations and model parameters compared to HiFi-GAN V1, while achieving over four times faster CPU inference speed.
>
---
#### [replaced 002] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
- **分类: cs.LG; cs.CL; eess.AS**

- **简介: 该论文研究同步语音翻译（SimulST），旨在平衡翻译质量与延迟。提出基于信息熵的正则化损失REINA，指导模型仅在获取新信息时等待输入，优化权衡。基于开源或合成数据训练，实现多语言SOTA性能，并提升流式效率达21%。**

- **链接: [https://arxiv.org/pdf/2508.04946v3](https://arxiv.org/pdf/2508.04946v3)**

> **作者:** Nameer Hirschkind; Joseph Liu; Xiao Yu; Mahesh Kumar Nandwana
>
> **备注:** Accepted to AAAI 2026 (Oral Track)
>
> **摘要:** Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.
>
---
#### [replaced 003] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking
- **分类: cs.CR; cs.SD; eess.AS**

- **简介: 该论文研究神经音频水印的安全漏洞，提出“覆写攻击”以伪造水印并消除原水印。针对白盒、灰盒、黑盒场景设计攻击方法，实验表明其在多种先进水印系统中接近100%成功，揭示现有方案的安全缺陷，强调需加强水印系统的安全性设计。**

- **链接: [https://arxiv.org/pdf/2509.05835v2](https://arxiv.org/pdf/2509.05835v2)**

> **作者:** Lingfeng Yao; Chenpei Huang; Shengyao Wang; Junpei Xue; Hanqing Guo; Jiang Liu; Phone Lin; Tomoaki Ohtsuki; Miao Pan
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.
>
---
