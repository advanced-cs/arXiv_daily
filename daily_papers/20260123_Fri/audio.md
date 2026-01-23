# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Bridging the Perception Gap: A Lightweight Coarse-to-Fine Architecture for Edge Audio Systems
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频理解任务，解决边缘计算中感知深度与效率的矛盾。提出CoFi-Agent架构，通过条件性边缘-云协作提升准确率。**

- **链接: [https://arxiv.org/pdf/2601.15676v1](https://arxiv.org/pdf/2601.15676v1)**

> **作者:** Hengfan Zhang; Yueqian Lin; Hai Helen Li; Yiran Chen
>
> **备注:** 10 pages, 3 figures, 2 tables. Preprint
>
> **摘要:** Deploying Audio-Language Models (Audio-LLMs) on edge infrastructure exposes a persistent tension between perception depth and computational efficiency. Lightweight local models tend to produce passive perception - generic summaries that miss the subtle evidence required for multi-step audio reasoning - while indiscriminate cloud offloading incurs unacceptable latency, bandwidth cost, and privacy risk. We propose CoFi-Agent (Tool-Augmented Coarse-to-Fine Agent), a hybrid architecture targeting edge servers and gateways. It performs fast local perception and triggers conditional forensic refinement only when uncertainty is detected. CoFi-Agent runs an initial single-pass on a local 7B Audio-LLM, then a cloud controller gates difficult cases and issues lightweight plans for on-device tools such as temporal re-listening and local ASR. On the MMAR benchmark, CoFi-Agent improves accuracy from 27.20% to 53.60%, while achieving a better accuracy-efficiency trade-off than an always-on investigation pipeline. Overall, CoFi-Agent bridges the perception gap via tool-enabled, conditional edge-cloud collaboration under practical system constraints.
>
---
#### [new 002] U3-xi: Pushing the Boundaries of Speaker Recognition via Incorporating Uncertainty
- **分类: cs.SD**

- **简介: 该论文属于说话人识别任务，旨在解决帧级信息不均衡问题。通过估计帧级不确定性并赋予自适应权重，提出U3-xi框架提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.15719v1](https://arxiv.org/pdf/2601.15719v1)**

> **作者:** Junjie Li; Kong Aik Lee
>
> **摘要:** An utterance-level speaker embedding is typically obtained by aggregating a sequence of frame-level representations. However, in real-world scenarios, individual frames encode not only speaker-relevant information but also various nuisance factors. As a result, different frames contribute unequally to the final utterance-level speaker representation for Automatic Speaker Verification systems. To address this issue, we propose to estimate the inherent uncertainty of each frame and assign adaptive weights accordingly, where frames with higher uncertainty receive lower attention. Based on this idea, we present U3-xi, a comprehensive framework designed to produce more reliable and interpretable uncertainty estimates for speaker embeddings. Specifically, we introduce several strategies for uncertainty supervision. First, we propose speaker-level uncertainty supervision via a Stochastic Variance Loss, where the distance between an utterance embedding and its corresponding speaker centroid serves as a pseudo ground truth for uncertainty learning. Second, we incorporate global-level uncertainty supervision by injecting the predicted uncertainty into the sof tmax scale during training. This adaptive scaling mechanism adjusts the sharpness of the decision boundary according to sample difficulty, providing global guidance. Third, we redesign the uncertainty estimation module by integrating a Transformer encoder with multi-view self-attention, enabling the model to capture rich local and long-range temporal dependencies. Comprehensive experiments demonstrate that U3-xi is model-agnostic and can be seamlessly applied to various speaker encoders. In particular, when applied to ECAPA-TDNN, it achieves 21.1% and 15.57% relative improvements on the VoxCeleb1 test sets in terms of EER and minDCF, respectively.
>
---
#### [new 003] Timbre-Aware LLM-based Direct Speech-to-Speech Translation Extendable to Multiple Language Pairs
- **分类: eess.AS; cs.HC**

- **简介: 该论文属于语音到语音翻译任务，旨在解决数据稀缺、说话人信息丢失和多语言扩展问题。提出DS2ST-LM框架，结合大语言模型和音色控制合成，提升翻译效果与自然度。**

- **链接: [https://arxiv.org/pdf/2601.16023v1](https://arxiv.org/pdf/2601.16023v1)**

> **作者:** Lalaram Arya; Mrinmoy Bhattacharjee; Adarsh C. R.; S. R. Mahadeva Prasanna
>
> **备注:** 13 pages
>
> **摘要:** Direct Speech-to-Speech Translation (S2ST) has gained increasing attention for its ability to translate speech from one language to another, while reducing error propagation and latency inherent in traditional cascaded pipelines. However, existing direct S2ST systems continue to face notable challenges, including instability in semantic-acoustic alignment when parallel speech data is scarce, difficulty in preserving speaker identity, and limited multilingual scalability. In this work, we introduce DS2ST-LM, a scalable, single-stage direct S2ST framework leveraging a multilingual Large Language Model (LLM). The architecture integrates a Whisper speech encoder, a learnable projection module, a Qwen2-0.5B LLM, and a timbre-controlled vocoder. We construct GigaS2S-1000, a 1000-hour bilingual corpus by extending the GigaST dataset with high-fidelity synthetic target speech, and show that this synthetic data alleviates data scarcity to some extent. We investigate two semantic token generation strategies: speech-derived S3 tokens and text-derived tokens generated by a pre-trained LLM, and analyze their impact on training stability and semantic consistency. We further evaluate three projection architectures (Linear, Conv1D-Linear, and Q-Former) and observe that while higher-capacity projectors converge faster, the simple Linear projector achieves higher performance. Extensive experiments demonstrate that DS2ST-LM outperforms traditional cascaded and ST (Qwen-Audio) + TTS baselines across both lexical (BLEU, METEOR) and semantic (BLEURT, COMET) metrics, while extending to multiple language pairs, including French, Spanish, German, Hindi, Bengali, and Urdu. Furthermore, we incorporate timbre-aware speech synthesis to preserve speaker information, enabling DS2ST-LM to surpass prior direct S2ST systems in both speaker similarity and perceptual naturalness.
>
---
#### [new 004] Distillation-based Layer Dropping (DLD) Effective End-to-end Framework for Dynamic Speech Networks
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于语音识别任务，旨在解决动态网络中层剪枝导致的性能下降问题。提出DLD框架，结合知识蒸馏与层剪枝，提升动态语音网络性能。**

- **链接: [https://arxiv.org/pdf/2601.16117v1](https://arxiv.org/pdf/2601.16117v1)**

> **作者:** Abdul Hannan; Daniele Falavigna; Shah Nawaz; Mubashir Noman; Markus Schedl; Alessio Brutti
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Edge devices operate in constrained and varying resource settings, requiring dynamic architectures that can adapt to limitations of the available resources. To meet such demands, layer dropping ($\mathcal{LD}$) approach is typically used to transform static models into dynamic ones by skipping parts of the network along with reducing overall computational complexity. However, existing $\mathcal{LD}$ methods greatly impact the dynamic model's performance for low and high dropping cases, deteriorating the performance-computation trade-off. To this end, we propose a distillation-based layer dropping (DLD) framework that effectively combines the capabilities of knowledge distillation and $\mathcal{LD}$ in an end-to-end fashion, thereby achieving state-of-the-art performance for dynamic speech networks. Comprehensive experimentation utilizing well-known speech recognition methods, including conformer and WavLM, on three public benchmarks demonstrates the effectiveness of our framework, reducing the word error rate by $9.32\%$ and $2.25\%$ for high and no dropping cases with $33.3\%$ reduction in training time.
>
---
#### [new 005] Abusive music and song transformation using GenAI and LLMs
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于内容安全任务，旨在解决音乐中暴力内容影响听众的问题。通过GenAI和LLMs转换歌词与演唱方式，降低攻击性，同时保持音乐完整性。**

- **链接: [https://arxiv.org/pdf/2601.15348v1](https://arxiv.org/pdf/2601.15348v1)**

> **作者:** Jiyang Choi; Rohitash Chandra
>
> **摘要:** Repeated exposure to violence and abusive content in music and song content can influence listeners' emotions and behaviours, potentially normalising aggression or reinforcing harmful stereotypes. In this study, we explore the use of generative artificial intelligence (GenAI) and Large Language Models (LLMs) to automatically transform abusive words (vocal delivery) and lyrical content in popular music. Rather than simply muting or replacing a single word, our approach transforms the tone, intensity, and sentiment, thus not altering just the lyrics, but how it is expressed. We present a comparative analysis of four selected English songs and their transformed counterparts, evaluating changes through both acoustic and sentiment-based lenses. Our findings indicate that Gen-AI significantly reduces vocal aggressiveness, with acoustic analysis showing improvements in Harmonic to Noise Ratio, Cepstral Peak Prominence, and Shimmer. Sentiment analysis reduced aggression by 63.3-85.6\% across artists, with major improvements in chorus sections (up to 88.6\% reduction). The transformed versions maintained musical coherence while mitigating harmful content, offering a promising alternative to traditional content moderation that avoids triggering the "forbidden fruit" effect, where the censored content becomes more appealing simply because it is restricted. This approach demonstrates the potential for GenAI to create safer listening experiences while preserving artistic expression.
>
---
#### [new 006] Domain-Incremental Continual Learning for Robust and Efficient Keyword Spotting in Resource Constrained Systems
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于关键词检测任务，解决资源受限设备中因领域变化导致的准确性和鲁棒性问题。提出一种持续学习框架，结合高效降噪和原型更新，提升模型适应能力。**

- **链接: [https://arxiv.org/pdf/2601.16158v1](https://arxiv.org/pdf/2601.16158v1)**

> **作者:** Prakash Dhungana; Sayed Ahmad Salehi
>
> **备注:** 12 pages, 8 figures, and 3 tables
>
> **摘要:** Keyword Spotting (KWS) systems with small footprint models deployed on edge devices face significant accuracy and robustness challenges due to domain shifts caused by varying noise and recording conditions. To address this, we propose a comprehensive framework for continual learning designed to adapt to new domains while maintaining computational efficiency. The proposed pipeline integrates a dual-input Convolutional Neural Network, utilizing both Mel Frequency Cepstral Coefficients (MFCC) and Mel-spectrogram features, supported by a multi-stage denoising process, involving discrete wavelet transform and spectral subtraction techniques, plus model and prototype update blocks. Unlike prior methods that restrict updates to specific layers, our approach updates the complete quantized model, made possible due to compact model architecture. A subset of input samples are selected during runtime using class prototypes and confidence-driven filtering, which are then pseudo-labeled and combined with rehearsal buffer for incremental model retraining. Experimental results on noisy test dataset demonstrate the framework's effectiveness, achieving 99.63\% accuracy on clean data and maintaining robust performance (exceeding 94\% accuracy) across diverse noisy environments, even at -10 dB Signal-to-Noise Ratio. The proposed framework work confirms that integrating efficient denoising with prototype-based continual learning enables KWS models to operate autonomously and robustly in resource-constrained, dynamic environments.
>
---
#### [new 007] PF-D2M: A Pose-free Diffusion Model for Universal Dance-to-Music Generation
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于舞蹈到音乐生成任务，解决多舞者和非人类舞蹈场景下的音乐生成问题。提出PF-D2M模型，利用视频视觉特征和渐进训练策略提升性能。**

- **链接: [https://arxiv.org/pdf/2601.15872v1](https://arxiv.org/pdf/2601.15872v1)**

> **作者:** Jaekwon Im; Natalia Polouliakh; Taketo Akama
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** Dance-to-music generation aims to generate music that is aligned with dance movements. Existing approaches typically rely on body motion features extracted from a single human dancer and limited dance-to-music datasets, which restrict their performance and applicability to real-world scenarios involving multiple dancers and non-human dancers. In this paper, we propose PF-D2M, a universal diffusion-based dance-to-music generation model that incorporates visual features extracted from dance videos. PF-D2M is trained with a progressive training strategy that effectively addresses data scarcity and generalization challenges. Both objective and subjective evaluations show that PF-D2M achieves state-of-the-art performance in dance-music alignment and music quality.
>
---
#### [new 008] Pay (Cross) Attention to the Melody: Curriculum Masking for Single-Encoder Melodic Harmonization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐生成任务，解决单编码器旋律和声生成中的注意力不足问题。提出FF训练课程，通过逐步解码增强旋律与和声的交互，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16150v1](https://arxiv.org/pdf/2601.16150v1)**

> **作者:** Maximos Kaliakatsos-Papakostas; Dimos Makris; Konstantinos Soiledis; Konstantinos-Theodoros Tsamis; Vassilis Katsouros; Emilios Cambouropoulos
>
> **摘要:** Melodic harmonization, the task of generating harmonic accompaniments for a given melody, remains a central challenge in computational music generation. Recent single encoder transformer approaches have framed harmonization as a masked sequence modeling problem, but existing training curricula inspired by discrete diffusion often result in weak (cross) attention between melody and harmony. This leads to limited exploitation of melodic cues, particularly in out-of-domain contexts. In this work, we introduce a training curriculum, FF (full-to-full), which keeps all harmony tokens masked for several training steps before progressively unmasking entire sequences during training to strengthen melody-harmony interactions. We systematically evaluate this approach against prior curricula across multiple experimental axes, including temporal quantization (quarter vs. sixteenth note), bar-level vs. time-signature conditioning, melody representation (full range vs. pitch class), and inference-time unmasking strategies. Models are trained on the HookTheory dataset and evaluated both in-domain and on a curated collection of jazz standards, using a comprehensive set of metrics that assess chord progression structure, harmony-melody alignment, and rhythmic coherence. Results demonstrate that the proposed FF curriculum consistently outperforms baselines in nearly all metrics, with particularly strong gains in out-of-domain evaluations where harmonic adaptability to novel melodic queues is crucial. We further find that quarter-note quantization, intertwining of bar tokens, and pitch-class melody representations are advantageous in the FF setting. Our findings highlight the importance of training curricula in enabling effective melody conditioning and suggest that full-to-full unmasking offers a robust strategy for single encoder harmonization.
>
---
#### [new 009] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning
- **分类: cs.SD**

- **简介: 该论文属于语音情感理解任务，旨在解决现有模型解释性差、依赖分类的问题。提出EmotionThinker，通过强化学习生成可解释的情感预测，提升情感理解的准确性和解释质量。**

- **链接: [https://arxiv.org/pdf/2601.15668v1](https://arxiv.org/pdf/2601.15668v1)**

> **作者:** Dingdong Wang; Shujie Liu; Tianhua Zhang; Youjun Chen; Jinyu Li; Helen Meng
>
> **摘要:** Emotional information in speech plays a unique role in multimodal perception. However, current Speech Large Language Models (SpeechLLMs), similar to conventional speech emotion recognition (SER) systems, still treat emotion understanding as a simple classification problem. This provides limited interpretability of predictions, while leaving the LLMs' expressive and reasoning capabilities underutilized. In this work, we take the first step to reformulate SER as a deep reasoning problem through reinforcement learning (RL). We propose EmotionThinker, which is designed to generate accurate emotion predictions with interpretable explanations grounded in fine-grained acoustic cues. To achieve this, we first construct EmotionCoT-35K, an emotional reasoning dataset with Chain-of-Thought annotations and detailed captions. Second, we observe that current SpeechLLMs exhibit weak prosody perception, whereas prosodic cues constitute fundamental signals for interpreting emotions. To address this, we develop the prosody-enhanced foundation model EmotionThinker-Base, and demonstrate that prosody enhancement improves emotion understanding. Third, we introduce Group-Relative-Policy-Optimization with Progressive-Trust-aware-Reasoning-Reward (GRPO-PTR) for RL. Different from standard GRPO, which relies only on rule-based outcome rewards, GRPO-PTR progressively introduces reasoning reward, dynamically adjusts it with a trustworthiness weight reflecting the alignment between reasoning and outcome, and evaluates the overall reasoning quality with a reward model based on multi-dimensional criteria. EmotionThinker outperforms previous state-of-the-art evaluation models both in emotion accuracy and explanation quality, advancing SER toward interpretable multimodal reasoning. Project page: https://github.com/dingdongwang/EmotionThinker
>
---
#### [new 010] Distributed Multichannel Active Noise Control with Asynchronous Communication
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于主动降噪任务，解决分布式系统中通信效率低的问题。提出异步通信策略，减少数据交换，提升系统效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2601.15653v1](https://arxiv.org/pdf/2601.15653v1)**

> **作者:** Junwei Ji; Dongyuan Shi; Boxiang Wang; Ziyi Yang; Haowen Li; Woon-Seng Gan
>
> **摘要:** Distributed multichannel active noise control (DMCANC) offers effective noise reduction across large spatial areas by distributing the computational load of centralized control to multiple low-cost nodes. Conventional DMCANC methods, however, typically assume synchronous communication and require frequent data exchange, resulting in high communication overhead. To enhance efficiency and adaptability, this work proposes an asynchronous communication strategy where each node executes a weight-constrained filtered-x LMS (WCFxLMS) algorithm and independently requests communication only when its local noise reduction performance degrades. Upon request, other nodes transmit the weight difference between their local control filter and the center point in WCFxLMS, which are then integrated to update both the control filter and the center point. This design enables nodes to operate asynchronously while preserving cooperative behavior. Simulation results demonstrate that the proposed asynchronous communication DMCANC (ACDMCANC) system maintains effective noise reduction with significantly reduced communication load, offering improved scalability for heterogeneous networks.
>
---
#### [new 011] DynamicSound simulator for simulating moving sources and microphone arrays
- **分类: eess.AS**

- **简介: 该论文属于声学仿真任务，旨在解决静态声源和固定麦克风阵列的局限性。提出DynamicSound框架，模拟移动声源和麦克风阵列的三维音频，支持真实声传播效果。**

- **链接: [https://arxiv.org/pdf/2601.15433v1](https://arxiv.org/pdf/2601.15433v1)**

> **作者:** Luca Barbisan; Marco Levorato; Fabrizio Riente
>
> **摘要:** Developing algorithms for sound classification, detection, and localization requires large amounts of flexible and realistic audio data, especially when leveraging modern machine learning and beamforming techniques. However, most existing acoustic simulators are tailored for indoor environments and are limited to static sound sources, making them unsuitable for scenarios involving moving sources, moving microphones, or long-distance propagation. This paper presents DynamicSound an open-source acoustic simulation framework for generating multichannel audio from one or more sound sources with the possibility to move them continuously in three-dimensional space and recorded by arbitrarily configured microphone arrays. The proposed model explicitly accounts for finite sound propagation delays, Doppler effects, distance-dependent attenuation, air absorption, and first-order reflections from planar surfaces, yielding temporally consistent spatial audio signals. Unlike conventional mono or stereo simulators, the proposed system synthesizes audio for an arbitrary number of virtual microphones, accurately reproducing inter-microphone time delays, level differences, and spectral coloration induced by the environment. Comparative evaluations with existing open-source tools demonstrate that the generated signals preserve high spatial fidelity across varying source positions and acoustic conditions. By enabling the generation of realistic multichannel audio under controlled and repeatable conditions, the proposed open framework provides a flexible and reproducible tool for the development, training, and evaluation of modern spatial audio and sound-source localization algorithms.
>
---
#### [new 012] Loose coupling of spectral and spatial models for multi-channel diarization and enhancement of meetings in dynamic environments
- **分类: eess.AS**

- **简介: 该论文属于会议语音处理任务，解决动态环境中说话人定位与信号增强问题。通过松耦合的谱-空联合模型，提升多通道语音分离效果。**

- **链接: [https://arxiv.org/pdf/2601.16077v1](https://arxiv.org/pdf/2601.16077v1)**

> **作者:** Adrian Meise; Tobias Cord-Landwehr; Christoph Boeddeker; Marc Delcroix; Tomohiro Nakatani; Reinhold Haeb-Umbach
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Sound capture by microphone arrays opens the possibility to exploit spatial, in addition to spectral, information for diarization and signal enhancement, two important tasks in meeting transcription. However, there is no one-to-one mapping of positions in space to speakers if speakers move. Here, we address this by proposing a novel joint spatial and spectral mixture model, whose two submodels are loosely coupled by modeling the relationship between speaker and position index probabilistically. Thus, spatial and spectral information can be jointly exploited, while at the same time allowing for speakers speaking from different positions. Experiments on the LibriCSS data set with simulated speaker position changes show great improvements over tightly coupled subsystems.
>
---
#### [new 013] DeepASMR: LLM-Based Zero-Shot ASMR Speech Generation for Anyone of Any Voice
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决零样本ASMR语音生成问题。通过引入DeepASMR框架，利用少量语音片段实现高质量ASMR合成。**

- **链接: [https://arxiv.org/pdf/2601.15596v1](https://arxiv.org/pdf/2601.15596v1)**

> **作者:** Leying Zhang; Tingxiao Zhou; Haiyang Sun; Mengxiao Bi; Yanmin Qian
>
> **摘要:** While modern Text-to-Speech (TTS) systems achieve high fidelity for read-style speech, they struggle to generate Autonomous Sensory Meridian Response (ASMR), a specialized, low-intensity speech style essential for relaxation. The inherent challenges include ASMR's subtle, often unvoiced characteristics and the demand for zero-shot speaker adaptation. In this paper, we introduce DeepASMR, the first framework designed for zero-shot ASMR generation. We demonstrate that a single short snippet of a speaker's ordinary, read-style speech is sufficient to synthesize high-fidelity ASMR in their voice, eliminating the need for whispered training data from the target speaker. Methodologically, we first identify that discrete speech tokens provide a soft factorization of ASMR style from speaker timbre. Leveraging this insight, we propose a two-stage pipeline incorporating a Large Language Model (LLM) for content-style encoding and a flow-matching acoustic decoder for timbre reconstruction. Furthermore, we contribute DeepASMR-DB, a comprehensive 670-hour English-Chinese multi-speaker ASMR speech corpus, and introduce a novel evaluation protocol integrating objective metrics, human listening tests, LLM-based scoring and unvoiced speech analysis. Extensive experiments confirm that DeepASMR achieves state-of-the-art naturalness and style fidelity in ASMR generation for anyone of any voice, while maintaining competitive performance on normal speech synthesis.
>
---
#### [new 014] A Stabilized Hybrid Active Noise Control Algorithm of GFANC and FxNLMS with Online Clustering
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于噪声控制任务，旨在解决FxNLMS收敛慢和GFANC适应性差的问题。提出混合算法结合两者优势，并引入在线聚类提升稳定性。**

- **链接: [https://arxiv.org/pdf/2601.15889v1](https://arxiv.org/pdf/2601.15889v1)**

> **作者:** Zhengding Luo; Haozhe Ma; Boxiang Wang; Ziyi Yang; Dongyuan Shi; Woon-Seng Gan
>
> **备注:** Accepted by 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** The Filtered-x Normalized Least Mean Square (FxNLMS) algorithm suffers from slow convergence and a risk of divergence, although it can achieve low steady-state errors after sufficient adaptation. In contrast, the Generative Fixed-Filter Active Noise Control (GFANC) method offers fast response speed, but its lack of adaptability may lead to large steady-state errors. This paper proposes a hybrid GFANC-FxNLMS algorithm to leverage the complementary advantages of both approaches. In the hybrid GFANC-FxNLMS algorithm, GFANC provides a frame-level control filter as an initialization for FxNLMS, while FxNLMS performs continuous adaptation at the sampling rate. Small variations in the GFANC-generated filter may repeatedly reinitialize FxNLMS, interrupting its adaptation process and destabilizing the system. An online clustering module is introduced to avoid unnecessary re-initializations and improve system stability. Simulation results show that the proposed algorithm achieves fast response, very low steady-state error, and high stability, requiring only one pre-trained broadband filter.
>
---
#### [new 015] Qwen3-TTS Technical Report
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在提升多语言、可控、鲁棒的语音生成效果。提出Qwen3-TTS模型，支持快速语音克隆和描述控制，采用双轨架构与两个高效语音分词器，实现低延迟实时合成。**

- **链接: [https://arxiv.org/pdf/2601.15621v1](https://arxiv.org/pdf/2601.15621v1)**

> **作者:** Hangrui Hu; Xinfa Zhu; Ting He; Dake Guo; Bin Zhang; Xiong Wang; Zhifang Guo; Ziyue Jiang; Hongkun Hao; Zishan Guo; Xinyu Zhang; Pei Zhang; Baosong Yang; Jin Xu; Jingren Zhou; Junyang Lin
>
> **备注:** https://github.com/QwenLM/Qwen3-TTS
>
> **摘要:** In this report, we present the Qwen3-TTS series, a family of advanced multilingual, controllable, robust, and streaming text-to-speech models. Qwen3-TTS supports state-of-the-art 3-second voice cloning and description-based control, allowing both the creation of entirely novel voices and fine-grained manipulation over the output speech. Trained on over 5 million hours of speech data spanning 10 languages, Qwen3-TTS adopts a dual-track LM architecture for real-time synthesis, coupled with two speech tokenizers: 1) Qwen-TTS-Tokenizer-25Hz is a single-codebook codec emphasizing semantic content, which offers seamlessly integration with Qwen-Audio and enables streaming waveform reconstruction via a block-wise DiT. 2) Qwen-TTS-Tokenizer-12Hz achieves extreme bitrate reduction and ultra-low-latency streaming, enabling immediate first-packet emission ($97\,\mathrm{ms}$) through its 12.5 Hz, 16-layer multi-codebook design and a lightweight causal ConvNet. Extensive experiments indicate state-of-the-art performance across diverse objective and subjective benchmark (e.g., TTS multilingual test set, InstructTTSEval, and our long speech test set). To facilitate community research and development, we release both tokenizers and models under the Apache 2.0 license.
>
---
#### [new 016] Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音大模型任务，解决领域术语识别不足问题。提出LOGIC框架，在解码层进行上下文偏置，提升实体识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.15397v1](https://arxiv.org/pdf/2601.15397v1)**

> **作者:** Peidong Wang
>
> **摘要:** The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
>
---
## 更新

#### [replaced 001] AudioMotionBench: Evaluating Auditory Motion Perception in Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频语言模型任务，旨在解决模型在听觉运动感知上的缺陷。研究提出了AudioMotionBench基准，评估模型对声音源运动方向和轨迹的识别能力，发现模型表现不佳，揭示了人机听觉空间推理的差距。**

- **链接: [https://arxiv.org/pdf/2511.13273v2](https://arxiv.org/pdf/2511.13273v2)**

> **作者:** Zhe Sun; Yujun Cai; Jiayu Yao; Yiwei Wang
>
> **摘要:** Large Audio-Language Models (LALMs) have recently shown impressive progress in speech recognition, audio captioning, and auditory question answering. Yet, whether these models can perceive spatial dynamics, particularly the motion of sound sources, remains unclear. In this work, we uncover a systematic motion perception deficit in current ALLMs. To investigate this issue, we introduce AudioMotionBench, the first benchmark explicitly designed to evaluate auditory motion understanding. AudioMotionBench introduces a controlled question-answering benchmark designed to evaluate whether Audio-Language Models (LALMs) can infer the direction and trajectory of moving sound sources from binaural audio. Comprehensive quantitative and qualitative analyses reveal that current models struggle to reliably recognize motion cues or distinguish directional patterns. The average accuracy remains below 50\%, underscoring a fundamental limitation in auditory spatial reasoning. Our study highlights a fundamental gap between human and model auditory spatial reasoning, providing both a diagnostic tool and new insight for enhancing spatial cognition in future Audio-Language Models.
>
---
#### [replaced 002] WavLink: Compact Audio-Text Embeddings with a Global Whisper Token
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文提出WavLink，一种紧凑的音频-文本嵌入模型，解决音频特征表示效率问题。通过融合Whisper和可学习全局标记，提升检索性能并减小嵌入规模。**

- **链接: [https://arxiv.org/pdf/2601.15118v2](https://arxiv.org/pdf/2601.15118v2)**

> **作者:** Gokul Karthik Kumar; Ludovick Lepauloux; Hakim Hacid
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Whisper has become the de-facto encoder for extracting general-purpose audio features in large audio-language models, where a 30-second clip is typically represented by 1500 frame features projected into an LLM. In contrast, audio-text embedding models like CLAP-based models have largely relied on alternative audio encoders (e.g., HTS-AT, PaSST), and have not leveraged Whisper effectively. We present WavLink, a compact audio-text embedding model that augments Whisper encoder with a learnable global token, trained jointly with a text encoder. Through a systematic study of design choices, including pretrained text encoders, loss functions, training modes, and data mixtures, we identify configurations that yield state-of-the-art retrieval performance. Our two-stage training recipe across three model sizes, combined with Matryoshka-style supervision, improves scalability, enabling 8x smaller embeddings with minimal performance drop. WavLink also demonstrates competitive performance on AIR-Bench with MCQs and zero-shot classification.
>
---
#### [replaced 003] Multi-Task Transformer for Explainable Speech Deepfake Detection via Formant Modeling
- **分类: cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在区分真实与虚假语音，并提供可解释性。通过多任务Transformer模型预测音高轨迹和发音模式，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.14850v2](https://arxiv.org/pdf/2601.14850v2)**

> **作者:** Viola Negroni; Luca Cuccovillo; Paolo Bestagini; Patrick Aichroth; Stefano Tubaro
>
> **备注:** Accepted @ IEEE ICASSP 2026
>
> **摘要:** In this work, we introduce a multi-task transformer for speech deepfake detection, capable of predicting formant trajectories and voicing patterns over time, ultimately classifying speech as real or fake, and highlighting whether its decisions rely more on voiced or unvoiced regions. Building on a prior speaker-formant transformer architecture, we streamline the model with an improved input segmentation strategy, redesign the decoding process, and integrate built-in explainability. Compared to the baseline, our model requires fewer parameters, trains faster, and provides better interpretability, without sacrificing prediction performance.
>
---
#### [replaced 004] Configurations, Tessellations and Tone Networks
- **分类: math.CO; eess.AS; math.AG**

- **简介: 论文探讨音乐理论中的调性网络，构建基于图论的音程结构，分析和声与对位关系，为作曲提供新方法。属于音乐数学建模任务。**

- **链接: [https://arxiv.org/pdf/2505.08752v5](https://arxiv.org/pdf/2505.08752v5)**

> **作者:** Jeffrey R. Boland; Lane P. Hughston
>
> **备注:** 59 pages, 23 figures
>
> **摘要:** The Eulerian tonnetz, which associates three minor chords to each major chord and three major chords to each minor chord, can be represented by a bipartite graph with twelve white vertices denoting major chords and twelve black vertices denoting minor chords. This so-called Levi graph determines a configuration of twelve points and twelve lines in $\mathbb R^2$ with the property that three points lie on each line and three lines pass through each point. Interesting features of the tonnetz, such as the existence of the four hexatonic cycles and the three octatonic cycles, crucial for the understanding of nineteenth-century harmony and voice leading, can be read off directly as properties of this configuration $\{12_3\}$ and its Levi graph. Analogous tone networks together with their Levi graphs and configurations can be constructed for pentatonic music and twelve-tone music. These and other new tonnetze offer the promise of new methods of composition. If the constraints of the Eulerian tonnetz are relaxed so as to allow movements between major and minor triads with variations at exactly two tones, the resulting bipartite graph has two components, each generating a tessellation of the plane, of a type known to Kepler, based on hexagons, squares and dodecagons. When the same combinatorial idea is applied to tetrachords of the 'Tristan' genus (dominant sevenths and half-diminished sevenths) the cycles of the resulting bipartite graph are sufficiently ample in girth to ensure the existence of a second configuration $\{12_3\}$, distinct from the Eulerian tonnetz as an incidence geometry, which can be used for a new approach to the analysis of the rich tetradic harmonies of the nineteenth century common practice.
>
---
#### [replaced 005] Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决情感细微变化捕捉和跨数据集泛化问题。通过融合MFCC特征与1D-CNN及注意力机制，提升识别精度。**

- **链接: [https://arxiv.org/pdf/2507.03251v3](https://arxiv.org/pdf/2507.03251v3)**

> **作者:** HyeYoung Lee; Muhammad Nadeem
>
> **备注:** After posting, we discovered that part of the material included in the manuscript should not have been publicly distributed in this form. We are withdrawing the paper while we address the issue
>
> **摘要:** Speech Emotion Recognition (SER) traditionally relies on auditory data analysis for emotion classification. Several studies have adopted different methods for SER. However, existing SER methods often struggle to capture subtle emotional variations and generalize across diverse datasets. In this article, we use Mel-Frequency Cepstral Coefficients (MFCCs) as spectral features to bridge the gap between computational emotion processing and human auditory perception. To further improve robustness and feature diversity, we propose a novel 1D-CNN-based SER framework that integrates data augmentation techniques. MFCC features extracted from the augmented data are processed using a 1D Convolutional Neural Network (CNN) architecture enhanced with channel and spatial attention mechanisms. These attention modules allow the model to highlight key emotional patterns, enhancing its ability to capture subtle variations in speech signals. The proposed method delivers cutting-edge performance, achieving the accuracy of 97.49% for SAVEE, 99.23% for RAVDESS, 89.31% for CREMA-D, 99.82% for TESS, 99.53% for EMO-DB, and 96.39% for EMOVO. Experimental results show new benchmarks in SER, demonstrating the effectiveness of our approach in recognizing emotional expressions with high precision. Our evaluation demonstrates that the integration of advanced Deep Learning (DL) methods substantially enhances generalization across diverse datasets, underscoring their potential to advance SER for real-world deployment in assistive technologies and human-computer interaction.
>
---
#### [replaced 006] Lightweight and perceptually-guided voice conversion for electro-laryngeal speech
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音转换任务，旨在提升电声喉语音的自然度和可懂度。通过轻量级模型优化，结合感知损失提升性能，显著降低字符错误率并提高评分。**

- **链接: [https://arxiv.org/pdf/2601.03892v2](https://arxiv.org/pdf/2601.03892v2)**

> **作者:** Benedikt Mayrhofer; Franz Pernkopf; Philipp Aichinger; Martin Hagmüller
>
> **备注:** 5 pages, 5 figures. Paper accepted for ICASSP 2026. Audio samples available at https://spsc-tugraz.github.io/lw-elvc-icassp26/
>
> **摘要:** Electro-laryngeal (EL) speech is characterized by constant pitch, limited prosody, and mechanical noise, reducing naturalness and intelligibility. We propose a lightweight adaptation of the state-of-the-art StreamVC framework to this setting by removing pitch and energy modules and combining self-supervised pretraining with supervised fine-tuning on parallel EL and healthy (HE) speech data, guided by perceptual and intelligibility losses. Objective and subjective evaluations across different loss configurations confirm their influence: the best model variant, based on WavLM features and human-feedback predictions (+WavLM+HF), drastically reduces character error rate (CER) of EL inputs, raises naturalness mean opinion score (nMOS) from 1.1 to 3.3, and consistently narrows the gap to HE ground-truth speech in all evaluated metrics. These findings demonstrate the feasibility of adapting lightweight voice conversion architectures to EL voice rehabilitation while also identifying prosody generation and intelligibility improvements as the main remaining bottlenecks.
>
---
#### [replaced 007] Principled Coarse-Grained Acceptance for Speculative Decoding in Speech
- **分类: eess.AS; cs.LG**

- **简介: 该论文属于语音生成任务，解决传统方法中token匹配过严导致效率低的问题。通过引入基于声学相似性的粗粒度验证方法，提升解码速度与接受率。**

- **链接: [https://arxiv.org/pdf/2511.13732v4](https://arxiv.org/pdf/2511.13732v4)**

> **作者:** Moran Yanuka; Paul Dixon; Eyal Finkelshtein; Daniel Rotman; Raja Giryes
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Speculative decoding accelerates autoregressive speech generation by letting a fast draft model propose tokens that a larger target model verifies. However, for speech LLMs that generate acoustic tokens, exact token matching is overly restrictive: many discrete tokens are acoustically or semantically interchangeable, reducing acceptance rates and limiting speedups. We introduce Principled Coarse-Graining (PCG), which verifies proposals at the level of Acoustic Similarity Groups (ASGs) derived from the target model's embedding space. By splitting each token's probability mass across the overlapping groups that contain it, we define an overlap-aware coarse-grained distribution and perform rejection sampling on the resulting group variable. This yields an exactness guarantee at the group level while allowing the accepted draft token to stand in for any member of the group in practice. On LibriTTS, PCG increases acceptance and throughput relative to standard speculative decoding and prior speech-specific relaxations while maintaining intelligibility and speaker similarity. These results suggest acoustically aware, group-level acceptance as a simple and general way to accelerate speech token generation while maintaining speech quality.
>
---
#### [replaced 008] Mitigation of multi-path propagation artefacts in acoustic targets with adaptive cepstral filtering
- **分类: cs.SD; cs.CE**

- **简介: 该论文属于声学目标识别任务，旨在解决多路径传播引起的信号失真问题。通过自适应倒谱滤波方法分离目标信号与反射成分，提升分类性能和时延估计精度。**

- **链接: [https://arxiv.org/pdf/2512.11165v2](https://arxiv.org/pdf/2512.11165v2)**

> **作者:** Lucas C. F. Domingos; Russell S. A. Brinkworth; Paulo E. Santos; Karl Sammut
>
> **摘要:** Passive acoustic sensing is a cost-effective solution for monitoring moving targets such as vessels and aircraft, but its performance is hindered by complex propagation effects like multi-path reflections and motion-induced artefacts. Existing filtering techniques do not properly incorporate the characteristics of the environment or account for variability in medium properties, limiting their effectiveness in separating source and reflection components. This paper proposes a method for separating target signals from their reflections in a spectrogram. Temporal filtering is applied to cepstral coefficients using an adaptive band-stop filter, which dynamically adjusts its bandwidth based on the relative intensity of the quefrency components. The method improved the signal-to-noise ratio (SNR) and log-spectral distance (LSD) across velocities ranging from 10 to 100 metres per second in aircraft noise with simulated motion. It also enhanced the performance of ship-type classification in underwater tasks by 2.28 and 2.62 Matthews Correlation Coefficient percentage points for the DeepShip and VTUAD v2 datasets, respectively. These results demonstrate the potential of the proposed pipeline to improve acoustic target classification and time-delay estimation in multi-path environments, with future work aimed at amplitude preservation and multi-sensor applications.
>
---
#### [replaced 009] Attentive AV-FusionNet: Audio-Visual Quality Prediction with Hybrid Attention
- **分类: eess.AS; cs.MM; eess.IV**

- **简介: 该论文属于音频-视频质量预测任务，解决多模态质量评估问题。通过融合音频和视频特征，并引入注意力机制提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2509.16994v2](https://arxiv.org/pdf/2509.16994v2)**

> **作者:** Ina Salaj; Arijit Biswas
>
> **备注:** Accepted to 51st IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 04-08 May 2026
>
> **摘要:** We introduce a novel deep learning-based audio-visual quality (AVQ) prediction model that leverages internal features from state-of-the-art unimodal predictors. Unlike prior approaches that rely on simple fusion strategies, our model employs a hybrid representation that combines learned Generative Machine Listener (GML) audio features with hand-crafted Video Multimethod Assessment Fusion (VMAF) video features. Attention mechanisms capture cross-modal interactions and intra-modal relationships, yielding context-aware quality representations. A modality relevance estimator quantifies each modality's contribution per content, potentially enabling adaptive bitrate allocation. Experiments demonstrate improved AVQ prediction accuracy and robustness across diverse content types.
>
---
#### [replaced 010] Xi+: Uncertainty Supervision for Robust Speaker Embedding
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于说话人识别任务，旨在提升说话人嵌入的鲁棒性。针对现有方法未充分考虑帧间时序关系和不确定性监督的问题，提出xi+模型，引入时序注意力和新损失函数，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2509.05993v4](https://arxiv.org/pdf/2509.05993v4)**

> **作者:** Junjie Li; Kong Aik Lee; Duc-Tuan Truong; Tianchi Liu; Man-Wai Mak
>
> **摘要:** There are various factors that can influence the performance of speaker recognition systems, such as emotion, language and other speaker-related or context-related variations. Since individual speech frames do not contribute equally to the utterance-level representation, it is essential to estimate the importance or reliability of each frame. The xi-vector model addresses this by assigning different weights to frames based on uncertainty estimation. However, its uncertainty estimation model is implicitly trained through classification loss alone and does not consider the temporal relationships between frames, which may lead to suboptimal supervision. In this paper, we propose an improved architecture, xi+. Compared to xi-vector, xi+ incorporates a temporal attention module to capture frame-level uncertainty in a context-aware manner. In addition, we introduce a novel loss function, Stochastic Variance Loss, which explicitly supervises the learning of uncertainty. Results demonstrate consistent performance improvements of about 10\% on the VoxCeleb1-O set and 11\% on the NIST SRE 2024 evaluation set.
>
---
#### [replaced 011] Towards Evaluating Generative Audio: Insights from Neural Audio Codec Embedding Distances
- **分类: eess.AS**

- **简介: 该论文属于音频质量评估任务，旨在解决如何有效评估生成音频质量的问题。通过比较不同方法，研究显示神经音频编解码器在压缩和感知评估中具有双重价值。**

- **链接: [https://arxiv.org/pdf/2509.18823v2](https://arxiv.org/pdf/2509.18823v2)**

> **作者:** Arijit Biswas; Lars Villemoes
>
> **备注:** Accepted to 51st IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 04-08 May 2026
>
> **摘要:** Neural audio codecs (NACs) achieve low-bitrate compression by learning compact audio representations, which can also serve as features for perceptual quality evaluation. We introduce DACe, an enhanced, higher-fidelity version of the Descript Audio Codec (DAC), trained on diverse real and synthetic tonal data with balanced sampling. We systematically compare Fréchet Audio Distance (FAD) and Maximum Mean Discrepancy (MMD) on MUSHRA tests across speech, music, and mixed content. FAD consistently outperforms MMD, and embeddings from higher-fidelity NACs (such as DACe) show stronger correlations with human judgments. While CLAP LAION Music (CLAP-M) and OpenL3 Mel128 (OpenL3-128M) embeddings achieve higher correlations, NAC embeddings provide a practical zero-shot approach to audio quality assessment, requiring only unencoded audio for training. These results demonstrate the dual utility of NACs for compression and perceptually informed audio evaluation.
>
---
#### [replaced 012] MOSA: Mixture of Simple Adapters Outperforms Monolithic Approaches in LLM-based Multilingual ASR
- **分类: eess.AS**

- **简介: 该论文属于多语言语音识别任务，解决数据稀缺问题。提出MOSA模型，通过多个简单适配器提升性能，减少参数干扰，提高效率。**

- **链接: [https://arxiv.org/pdf/2508.18998v2](https://arxiv.org/pdf/2508.18998v2)**

> **作者:** Junjie Li; Jing Peng; Yangui Fang; Shuai Wang; Kai Yu
>
> **备注:** 5 pages, 3 figures, accepted to ICASSP 2026
>
> **摘要:** LLM-based ASR overcomes multilingual data scarcity by projecting speech representations into the LLM space to leverage its robust semantic and reasoning capabilities. However, while previous approaches typically enhance performance by scaling data or model parameters, a single projector often struggles to effectively align representations across different languages. In this work, we propose an MoE-based projector named MOSA (Mixture of Simple Adapters). By aggregating multiple simple adapters, this architecture enables different experts to specialize in learning either language-shared or language-specific knowledge. This approach not only mitigates parameter interference between languages but also facilitates positive transfer from high-resource to low-resource languages, effectively alleviating data scarcity issues. Experimental results demonstrate that MOSA-Base achieves a 15.4% relative reduction in average WER compared to the Ideal-LLM Base, consistently outperforming it across all languages. Notably, MOSA achieves a 13.3% WER reduction over the Ideal-LLM Base while utilizing only 60% of its parameters. These findings highlight MOSA's superior parameter efficiency and robustness against data imbalance, suggesting that a mixture of simple adapters is more suitable for multilingual LLM-based ASR than complex single-adapter designs.
>
---
#### [replaced 013] Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究LoRA在Whisper模型中的机制，用于语音情感识别任务。解决LoRA在语音任务中机制不明确的问题，通过分析工具揭示其工作原理。**

- **链接: [https://arxiv.org/pdf/2509.08454v3](https://arxiv.org/pdf/2509.08454v3)**

> **作者:** Yujian Ma; Xikun Lu; Jinqiu Sang; Xianquan Jiang; Ruizhe Li
>
> **备注:** Accepted at ICASSP 2026
>
> **摘要:** Large pre-trained speech models such as Whisper offer strong generalization but pose significant challenges for resource-efficient adaptation. Low-Rank Adaptation (LoRA) has become a popular parameter-efficient fine-tuning method, yet its underlying mechanisms in speech tasks remain poorly understood. In this work, we conduct the first systematic mechanistic interpretability study of LoRA within the Whisper encoder for speech emotion recognition (SER). Using a suite of analytical tools, including layer contribution probing, logit-lens inspection, and representational similarity via singular value decomposition (SVD) and centered kernel alignment (CKA), we reveal two key mechanisms: a delayed specialization process that preserves general features in early layers before consolidating task-specific information, and a forward alignment, backward differentiation dynamic between LoRA's matrices. Our findings clarify how LoRA reshapes encoder hierarchies, providing both empirical insights and a deeper mechanistic understanding for designing efficient and interpretable adaptation strategies in large speech models. Our code is available at https://github.com/harryporry77/Behind-the-Scenes.
>
---
#### [replaced 014] Quantization-Based Score Calibration for Few-Shot Keyword Spotting with Dynamic Time Warping in Noisy Environments
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于关键词检测任务，解决噪声环境下阈值选择不当导致性能下降的问题。通过嵌入量化和归一化提升DTW的检测效果。**

- **链接: [https://arxiv.org/pdf/2510.15432v2](https://arxiv.org/pdf/2510.15432v2)**

> **作者:** Kevin Wilkinghoff; Alessia Cornaggia-Urrigshardt; Zheng-Hua Tan
>
> **摘要:** Detecting occurrences of keywords with keyword spotting (KWS) systems requires thresholding continuous detection scores. Selecting appropriate thresholds is a non-trivial task, typically relying on optimizing performance on a validation dataset. However, such greedy threshold selection often leads to suboptimal performance on unseen data, particularly in varying or noisy acoustic environments or few-shot settings. In this work, we investigate detection threshold estimation for template-based open-set few-shot KWS using dynamic time warping on noisy speech data. To mitigate the performance degradation caused by suboptimal thresholds, we propose a score calibration approach that operates at the embedding level by quantizing learned representations and applying quantization error-based normalization prior to DTW-based scoring and thresholding. Experiments on KWS-DailyTalk with simulated high frequency radio channels show that the proposed calibration approach simplifies the selection of robust detection thresholds and significantly improves the resulting performance.
>
---
#### [replaced 015] Competitive Audio-Language Models with Data-Efficient Single-Stage Training on Public Data
- **分类: cs.SD; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Falcon3-Audio，一种高效音频-语言模型，解决音频与语言融合不足的问题。使用少量公开数据，实现高性能，强调数据效率与训练简洁性。**

- **链接: [https://arxiv.org/pdf/2509.07526v3](https://arxiv.org/pdf/2509.07526v3)**

> **作者:** Gokul Karthik Kumar; Rishabh Saraf; Ludovick Lepauloux; Abdul Muneer; Billel Mokeddem; Hakim Hacid
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** Large language models (LLMs) have transformed NLP, yet their integration with audio remains underexplored despite audio's centrality to human communication. We introduce Falcon3-Audio, a family of Audio-Language Models (ALMs) built on instruction-tuned LLMs and Whisper encoders. Using a remarkably small amount of public audio data, less than 30K hours (5K unique), Falcon3-Audio-7B matches the best reported performance among open-weight models on the MMAU benchmark, with a score of 64.14, matching R1-AQA, while distinguishing itself through superior data and parameter efficiency, single-stage training, and transparency. Notably, our smallest 1B model remains competitive with larger open models ranging from 2B to 13B parameters. Through extensive ablations, we find that common complexities such as curriculum learning, multiple audio encoders, and intricate cross-attention connectors are not required for strong performance, even compared to models trained on over 500K hours of data.
>
---
#### [replaced 016] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出MMSU基准，用于评估语音语言理解与推理能力。解决现有模型在语音细粒度感知和复杂推理上的不足，涵盖多任务评测与语言理论分析。**

- **链接: [https://arxiv.org/pdf/2506.04779v2](https://arxiv.org/pdf/2506.04779v2)**

> **作者:** Dingdong Wang; Jincenzi Wu; Junan Li; Dongchao Yang; Xueyuan Chen; Tianhua Zhang; Helen Meng
>
> **备注:** MMSU benchmark is available at https://huggingface.co/datasets/ddwang2000/MMSU. Evaluation Code is available at https://github.com/dingdongwang/MMSU_Bench
>
> **摘要:** Speech inherently contains rich acoustic information that extends far beyond the textual language. In real-world spoken language understanding, effective interpretation often requires integrating semantic meaning (e.g., content), paralinguistic features (e.g., emotions, speed, pitch) and phonological characteristics (e.g., prosody, intonation, rhythm), which are embedded in speech. While recent multimodal Speech Large Language Models (SpeechLLMs) have demonstrated remarkable capabilities in processing audio information, their ability to perform fine-grained perception and complex reasoning in natural speech remains largely unexplored. To address this gap, we introduce MMSU, a comprehensive benchmark designed specifically for understanding and reasoning in spoken language. MMSU comprises 5,000 meticulously curated audio-question-answer triplets across 47 distinct tasks. To ground our benchmark in linguistic theory, we systematically incorporate a wide range of linguistic phenomena, including phonetics, prosody, rhetoric, syntactics, semantics, and paralinguistics. Through a rigorous evaluation of 14 advanced SpeechLLMs, we identify substantial room for improvement in existing models, highlighting meaningful directions for future optimization. MMSU establishes a new standard for comprehensive assessment of spoken language understanding, providing valuable insights for developing more sophisticated human-AI speech interaction systems. MMSU benchmark is available at https://huggingface.co/datasets/ddwang2000/MMSU. Evaluation Code is available at https://github.com/dingdongwang/MMSU_Bench.
>
---
#### [replaced 017] AQA-TTRL: Self-Adaptation in Audio Question Answering with Test-Time Reinforcement Learning
- **分类: eess.AS**

- **简介: 该论文属于音频问答任务，解决模型部署后无法适应新数据的问题。提出AQA-TTRL框架，利用测试时强化学习和自生成标签实现模型自适应优化。**

- **链接: [https://arxiv.org/pdf/2510.05478v2](https://arxiv.org/pdf/2510.05478v2)**

> **作者:** Haoyu Zhang; Jiaxian Guo; Yusuke Iwasawa; Yutaka Matsuo
>
> **摘要:** Large Audio Language Models (LALMs) demonstrate impressive general audio understanding, but once deployed, they are static and fail to improve with new real-world audio data. As traditional supervised fine-tuning is costly, we introduce a novel framework for test-time audio understanding, AQA-TTRL, where an LALM evolves on-the-fly using only unlabeled test data. It first generates pseudo-labels from the prediction via majority voting, then optimizes the model via reinforcement learning. To handle the inherent noise in these self-generated labels, we introduce a confidence-based weighting method to adjust training signals. Furthermore, a multiple-attempt sampling operation mitigates advantage collapse and stabilizes training. On the MMAU (test-mini/test), MMAR, and MMSU benchmarks, AQA-TTRL achieves significant average improvements of 4.42% for the Qwen2.5-Omni 7B model and 11.04% for the 3B model. Notably, the adapted 3B model consistently outperforms the direct inference of the unadapted 7B model, highlighting the effectiveness of previously unexplored test-time adaptations in audio understanding.
>
---
