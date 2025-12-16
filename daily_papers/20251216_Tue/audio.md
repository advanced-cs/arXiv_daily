# 音频 cs.SD;  eess.AS

- **最新发布 12 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] A comparative study of generative models for child voice conversion
- **分类: cs.SD**

- **简介: 该论文研究成人到儿童语音转换任务，旨在解决现有生成模型在儿童语音合成中目标说话人相似度不足的问题。作者对比了扩散、流模型、VAE和GAN四种模型，并提出一种高效频率扭曲技术提升音色匹配度，通过主客观评估验证效果。**

- **链接: [https://arxiv.org/pdf/2512.12129v1](https://arxiv.org/pdf/2512.12129v1)**

> **作者:** Protima Nomo Sudro; Anton Ragni; Thomas Hain
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Generative models are a popular choice for adult-to-adult voice conversion (VC) because of their efficient way of modelling unlabelled data. To this point their usefulness in producing children speech and in particular adult to child VC has not been investigated. For adult to child VC, four generative models are compared: diffusion model, flow based model, variational autoencoders, and generative adversarial network. Results show that although converted speech outputs produce by those models appear plausible, they exhibit insufficient similarity with the target speaker characteristics. We introduce an efficient frequency warping technique that can be applied to the output of models, and which shows significant reduction of the mismatch between adult and child. The output of all the models are evaluated using both objective and subjective measures. In particular we compare specific speaker pairing using a unique corpus collected for dubbing of children speech.
>
---
#### [new 002] Privacy-Aware Ambient Audio Sensing for Healthy Indoor Spaces
- **分类: cs.SD**

- **简介: 该论文属隐私保护的环境感知任务，旨在解决室内空气传播健康风险监测中侵入性、高成本及缺乏实时性的问题。作者提出基于环境音频的非侵入式方法，利用现有麦克风实时估计通风、气溶胶排放和人员分布，并设计隐私保护机制，实现室内空气质量与健康风险的实时、低侵入监测。**

- **链接: [https://arxiv.org/pdf/2512.12471v1](https://arxiv.org/pdf/2512.12471v1)**

> **作者:** Bhawana Chhaglani
>
> **摘要:** Indoor airborne transmission poses a significant health risk, yet current monitoring solutions are invasive, costly, or fail to address it directly. My research explores the untapped potential of ambient audio sensing to estimate key transmission risk factors such as ventilation, aerosol emissions, and occupant distribution non-invasively and in real time. I develop privacy-preserving systems that leverage existing microphones to monitor the whole spectrum of indoor air quality which can have a significant effect on an individual's health. This work lays the foundation for privacy-aware airborne risk monitoring using everyday devices.
>
---
#### [new 003] BUT Systems for WildSpoof Challenge: SASV in the Wild
- **分类: eess.AS**

- **简介: 该论文面向WildSpoof挑战的SASV任务，解决野外环境下语音欺骗攻击导致的说话人验证鲁棒性问题。提出融合多源自监督前端（如Dasheng、WavLM）与因子化注意力后端的框架，并引入基于分布不确定性的特征增强策略，联合CM与ASV系统优化a-DCFs和EER。**

- **链接: [https://arxiv.org/pdf/2512.12851v1](https://arxiv.org/pdf/2512.12851v1)**

> **作者:** Junyi Peng; Jin Li; Johan Rohdin; Lin Zhang; Miroslav Hlaváček; Oldrich Plchot
>
> **备注:** 4 pages
>
> **摘要:** This paper presents the BUT submission to the WildSpoof Challenge, focusing on the Spoofing-robust Automatic Speaker Verification (SASV) track. We propose a SASV framework designed to bridge the gap between general audio understanding and specialized speech analysis. Our subsystem integrates diverse Self-Supervised Learning front-ends ranging from general audio models (e.g., Dasheng) to speech-specific encoders (e.g., WavLM). These representations are aggregated via a lightweight Multi-Head Factorized Attention back-end for corresponding subtasks. Furthermore, we introduce a feature domain augmentation strategy based on Distribution Uncertainty to explicitly model and mitigate the domain shift caused by unseen neural vocoders and recording environments. By fusing these robust CM scores with state-of-the-art ASV systems, our approach achieves superior minimization of the a-DCFs and EERs.
>
---
#### [new 004] REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification
- **分类: eess.AS**

- **简介: 该论文属联邦学习中的音频分类任务，旨在解决客户端异质性与投毒攻击导致的全局模型鲁棒性下降问题。提出REVERB-FL：一种轻量级服务器端防御方法，结合小规模预留集、预/后聚合微调及对抗训练，无需客户端修改，有效缓解模型投毒并提升收敛性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.13647v1](https://arxiv.org/pdf/2512.13647v1)**

> **作者:** Sathwika Peechara; Rajeev Sahay
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Federated learning (FL) enables a privacy-preserving training paradigm for audio classification but is highly sensitive to client heterogeneity and poisoning attacks, where adversarially compromised clients can bias the global model and hinder the performance of audio classifiers. To mitigate the effects of model poisoning for audio signal classification, we present REVERB-FL, a lightweight, server-side defense that couples a small reserve set (approximately 5%) with pre- and post-aggregation retraining and adversarial training. After each local training round, the server refines the global model on the reserve set with either clean or additional adversarially perturbed data, thereby counteracting non-IID drift and mitigating potential model poisoning without adding substantial client-side cost or altering the aggregation process. We theoretically demonstrate the feasibility of our framework, showing faster convergence and a reduced steady-state error relative to baseline federated averaging. We validate our framework on two open-source audio classification datasets with varying IID and Dirichlet non-IID partitions and demonstrate that REVERB-FL mitigates global model poisoning under multiple designs of local data poisoning.
>
---
#### [new 005] SAMAY: System for Acoustic Measurement and Analysis
- **分类: cs.SD; cs.RO**

- **简介: 该论文提出SAMAY系统，属智能环境监测任务，旨在解决野外鸟类声学数据自动采集与分析难题。工作包括设计基于STM32F407的便携式录音设备，支持4麦克风、128GB存储、太阳能供电，并实现USB/Wi-Fi实时配置，支撑物种识别、种群监测与环境影响分析。**

- **链接: [https://arxiv.org/pdf/2512.13284v1](https://arxiv.org/pdf/2512.13284v1)**

> **作者:** Adheep Arya G R; Vaibhav Pratap Singh; Mayank Kumar; Niyathi Shenoy; Tejas Suryawanshi; Ruchi Juyal; Sangit Saha; Kaushik Nanda; Hari Babu Pasupuleti; S D Sudarsan
>
> **摘要:** This paper describes an automatic bird call recording system called SAMAY, which is developed to study bird species by creating a database of large amounts of bird acoustic data. By analysing the recorded bird call data, the system can also be used for automatic classification of bird species, monitoring bird populations and analysing the impact of environmental changes. The system is driven through a powerful STM32F407 series microcontroller, supports 4 microphones, is equipped with 128 GB of storage capacity, and is powered by a 10400 mAh battery pack interfaced with a solar charger. In addition, the device is user-configurable over USB and Wi-Fi during runtime, ensuring user-friendly operation during field deployment.
>
---
#### [new 006] Adaptive Edge-Cloud Inference for Speech-to-Action Systems Using ASR and Large Language Models (ASTA)
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出ASTA系统，解决语音控制IoT设备中边缘与云推理的权衡问题。它通过实时系统指标动态路由语音命令至边缘或云端，并结合ASR、轻量语言模型与LLM，辅以命令验证与修复，提升鲁棒性与资源效率。**

- **链接: [https://arxiv.org/pdf/2512.12769v1](https://arxiv.org/pdf/2512.12769v1)**

> **作者:** Mohammad Jalili Torkamani; Israt Zarin
>
> **备注:** preprint, 6 pages, 7 figures, 1 table
>
> **摘要:** Voice-based interaction has emerged as a natural and intuitive modality for controlling IoT devices. However, speech-driven edge devices face a fundamental trade-off between cloud-based solutions, which offer stronger language understanding capabilities at the cost of latency, connectivity dependence, and privacy concerns, and edge-based solutions, which provide low latency and improved privacy but are limited by computational constraints. This paper presents ASTA, an adaptive speech-to-action solution that dynamically routes voice commands between edge and cloud inference to balance performance and system resource utilization. ASTA integrates on-device automatic speech recognition and lightweight offline language-model inference with cloud-based LLM processing, guided by real-time system metrics such as CPU workload, device temperature, and network latency. A metric-aware routing mechanism selects the inference path at runtime, while a rule-based command validation and repair component ensures successful end-to-end command execution. We implemented our solution on an NVIDIA Jetson-based edge platform and evaluated it using a diverse dataset of 80 spoken commands. Experimental results show that ASTA successfully routes all input commands for execution, achieving a balanced distribution between online and offline inference. The system attains an ASR accuracy of 62.5% and generates executable commands without repair for only 47.5% of inputs, highlighting the importance of the repair mechanism in improving robustness. These results suggest that adaptive edge-cloud orchestration is a viable approach for resilient and resource-aware voice-controlled IoT systems.
>
---
#### [new 007] HQ-MPSD: A Multilingual Artifact-Controlled Benchmark for Partial Deepfake Speech Detection
- **分类: cs.SD**

- **简介: 该论文面向部分深度伪造语音检测任务，旨在解决现有数据集因合成过时、含人工伪影而导致模型泛化差的问题。作者构建了高质量、多语言、带真实背景噪声的HQ-MPSD基准数据集，并验证其显著降低SOTA模型性能，凸显现实挑战。**

- **链接: [https://arxiv.org/pdf/2512.13012v1](https://arxiv.org/pdf/2512.13012v1)**

> **作者:** Menglu Li; Majd Alber; Ramtin Asgarianamiri; Lian Zhao; Xiao-Ping Zhang
>
> **备注:** 6 pages, 4 figures, 2 tables
>
> **摘要:** Detecting partial deepfake speech is challenging because manipulations occur only in short regions while the surrounding audio remains authentic. However, existing detection methods are fundamentally limited by the quality of available datasets, many of which rely on outdated synthesis systems and generation procedures that introduce dataset-specific artifacts rather than realistic manipulation cues. To address this gap, we introduce HQ-MPSD, a high-quality multilingual partial deepfake speech dataset. HQ-MPSD is constructed using linguistically coherent splice points derived from fine-grained forced alignment, preserving prosodic and semantic continuity and minimizing audible and visual boundary artifacts. The dataset contains 350.8 hours of speech across eight languages and 550 speakers, with background effects added to better reflect real-world acoustic conditions. MOS evaluations and spectrogram analysis confirm the high perceptual naturalness of the samples. We benchmark state-of-the-art detection models through cross-language and cross-dataset evaluations, and all models experience performance drops exceeding 80% on HQ-MPSD. These results demonstrate that HQ-MPSD exposes significant generalization challenges once low-level artifacts are removed and multilingual and acoustic diversity are introduced, providing a more realistic and demanding benchmark for partial deepfake detection. The dataset can be found at: https://zenodo.org/records/17929533.
>
---
#### [new 008] Procedural Music Generation Systems in Games
- **分类: cs.SD**

- **简介: 该论文属综述任务，旨在弥合PMG学术研究与游戏工业应用间的鸿沟。它提出两维度分类法，对比分析算法实现、音乐质量与游戏集成等挑战，并指出未来应聚焦任务导向、上下文感知设计、质量评估与工具整合。**

- **链接: [https://arxiv.org/pdf/2512.12834v1](https://arxiv.org/pdf/2512.12834v1)**

> **作者:** Shangxuan Luo; Joshua Reiss
>
> **摘要:** Procedural Music Generation (PMG) is an emerging field that algorithmically creates music content for video games. By leveraging techniques from simple rule-based approaches to advanced machine learning algorithms, PMG has the potential to significantly improve development efficiency, provide richer musical experiences, and enhance player immersion. However, academic prototypes often diverge from applications due to differences in priorities such as novelty, reliability, and allocated resources. This paper bridges the gap between research and applications by presenting a systematic overview of current PMG techniques in both fields, offering a two-aspect taxonomy. Through a comparative analysis, this study identifies key research challenges in algorithm implementation, music quality and game integration. Finally, the paper outlines future research directions, emphasising task-oriented and context-aware design, more comprehensive quality evaluation methods, and improved research tool integration to provide actionable insights for developers, composers, and researchers seeking to advance PMG in game contexts.
>
---
#### [new 009] DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec
- **分类: cs.SD**

- **简介: 该论文属零-shot可控语音合成任务，旨在解决语音编解码器中音色与韵律耦合导致的控制不灵活问题。提出DisCo-Speech框架，含解耦语音编解码器（DisCodec）和LM生成器，通过三因子解耦与融合重建实现韵律独立控制与语音克隆。**

- **链接: [https://arxiv.org/pdf/2512.13251v1](https://arxiv.org/pdf/2512.13251v1)**

> **作者:** Tao Li; Wengshuo Ge; Zhichao Wang; Zihao Cui; Yong Ma; Yingying Gao; Chao Deng; Shilei Zhang; Junlan Feng
>
> **摘要:** Recent codec-based language models~(LMs) have revolutionized text-to-speech~(TTS). However, since standard codecs tightly couple timbre and prosody, continuation-based LMs inevitably replicate this entanglement, hindering independent control. Recent efforts attempt to break this entanglement via codec design, but insufficient decoupling remains a critical bottleneck. To tackle this challenge, we propose DisCo-Speech, a zero-shot controllable TTS framework that enables prosody control and voice cloning via a disentangled speech codec (DisCodec) and an LM-based generator. The core component, DisCodec, contains two core stages: 1) Tri-factor disentanglement, which explicitly factorizes speech into content, prosody, and timbre subspaces via parallel encoders and hybrid losses; and 2) Fusion and reconstruction, which fuses content and prosody into unified content-prosody tokens suitable for LM prediction, while jointly optimizing reconstruction quality to resolve the disentanglement-reconstruction trade-off. With this design, the LM performs prosodic continuation from a style prompt while the decoder handles target timbre injection, enabling flexible zero-shot control. Experiments show that DisCo-Speech matches state-of-the-art voice cloning performance while outperforming baselines in zero-shot prosody control. By resolving the core entanglement at the codec level, DisCo-Speech provides a robust foundation for controllable speech synthesis. Audio samples are available at https://github.com/disco-speech/DisCo-Speech, and the code and weights will be released at the same link.
>
---
#### [new 010] AutoMV: An Automatic Multi-Agent System for Music Video Generation
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **简介: 该论文面向音乐视频（MV）生成任务，解决现有方法难以生成全长、结构一致、音画对齐MV的问题。提出多智能体系统AutoMV，通过音乐分析、脚本与导演协同、分场景生成及验证机制，实现端到端MV生成，并构建新评估基准验证其显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.12196v1](https://arxiv.org/pdf/2512.12196v1)**

> **作者:** Xiaoxuan Tang; Xinping Lei; Chaoran Zhu; Shiyun Chen; Ruibin Yuan; Yizhi Li; Changjae Oh; Ge Zhang; Wenhao Huang; Emmanouil Benetos; Yang Liu; Jiaheng Liu; Yinghao Ma
>
> **摘要:** Music-to-Video (M2V) generation for full-length songs faces significant challenges. Existing methods produce short, disjointed clips, failing to align visuals with musical structure, beats, or lyrics, and lack temporal consistency. We propose AutoMV, a multi-agent system that generates full music videos (MVs) directly from a song. AutoMV first applies music processing tools to extract musical attributes, such as structure, vocal tracks, and time-aligned lyrics, and constructs these features as contextual inputs for following agents. The screenwriter Agent and director Agent then use this information to design short script, define character profiles in a shared external bank, and specify camera instructions. Subsequently, these agents call the image generator for keyframes and different video generators for "story" or "singer" scenes. A Verifier Agent evaluates their output, enabling multi-agent collaboration to produce a coherent longform MV. To evaluate M2V generation, we further propose a benchmark with four high-level categories (Music Content, Technical, Post-production, Art) and twelve ine-grained criteria. This benchmark was applied to compare commercial products, AutoMV, and human-directed MVs with expert human raters: AutoMV outperforms current baselines significantly across all four categories, narrowing the gap to professional MVs. Finally, we investigate using large multimodal models as automatic MV judges; while promising, they still lag behind human expert, highlighting room for future work.
>
---
#### [new 011] Schrodinger Audio-Visual Editor: Object-Level Audiovisual Removal
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文提出“薛定谔音视频编辑器（SAVE）”，解决对象级音视频联合编辑任务，旨在同步移除目标对象并保持其余内容及音视频对齐。为此构建了配对数据集SAVEBench，并设计基于薛定谔桥的端到端流匹配模型，实现更优时序同步与语义一致性。**

- **链接: [https://arxiv.org/pdf/2512.12875v1](https://arxiv.org/pdf/2512.12875v1)**

> **作者:** Weihan Xu; Kan Jen Cheng; Koichi Saito; Muhammad Jehanzeb Mirza; Tingle Li; Yisi Liu; Alexander H. Liu; Liming Wang; Masato Ishii; Takashi Shibuya; Yuki Mitsufuji; Gopala Anumanchipalli; Paul Pu Liang
>
> **摘要:** Joint editing of audio and visual content is crucial for precise and controllable content creation. This new task poses challenges due to the limitations of paired audio-visual data before and after targeted edits, and the heterogeneity across modalities. To address the data and modeling challenges in joint audio-visual editing, we introduce SAVEBench, a paired audiovisual dataset with text and mask conditions to enable object-grounded source-to-target learning. With SAVEBench, we train the Schrodinger Audio-Visual Editor (SAVE), an end-to-end flow-matching model that edits audio and video in parallel while keeping them aligned throughout processing. SAVE incorporates a Schrodinger Bridge that learns a direct transport from source to target audiovisual mixtures. Our evaluation demonstrates that the proposed SAVE model is able to remove the target objects in audio and visual content while preserving the remaining content, with stronger temporal synchronization and audiovisual semantic correspondence compared with pairwise combinations of an audio editor and a video editor.
>
---
#### [new 012] Towards Unified Co-Speech Gesture Generation via Hierarchical Implicit Periodicity Learning
- **分类: cs.AI; cs.CV; cs.GR; cs.MM; cs.SD**

- **简介: 该论文属“语音驱动3D手势生成”任务，旨在解决现有方法难以建模头、身、手运动间内在关联导致动作不自然、协调性差的问题。提出分层隐式周期性（HIP）学习方法，通过周期自编码器解耦运动相位，并以级联引导建模多部位层级关系。**

- **链接: [https://arxiv.org/pdf/2512.13131v1](https://arxiv.org/pdf/2512.13131v1)**

> **作者:** Xin Guo; Yifan Zhao; Jia Li
>
> **备注:** IEEE Transactions on Image Processing
>
> **摘要:** Generating 3D-based body movements from speech shows great potential in extensive downstream applications, while it still suffers challenges in imitating realistic human movements. Predominant research efforts focus on end-to-end generation schemes to generate co-speech gestures, spanning GANs, VQ-VAE, and recent diffusion models. As an ill-posed problem, in this paper, we argue that these prevailing learning schemes fail to model crucial inter- and intra-correlations across different motion units, i.e. head, body, and hands, thus leading to unnatural movements and poor coordination. To delve into these intrinsic correlations, we propose a unified Hierarchical Implicit Periodicity (HIP) learning approach for audio-inspired 3D gesture generation. Different from predominant research, our approach models this multi-modal implicit relationship by two explicit technique insights: i) To disentangle the complicated gesture movements, we first explore the gesture motion phase manifolds with periodic autoencoders to imitate human natures from realistic distributions while incorporating non-period ones from current latent states for instance-level diversities. ii) To model the hierarchical relationship of face motions, body gestures, and hand movements, driving the animation with cascaded guidance during learning. We exhibit our proposed approach on 3D avatars and extensive experiments show our method outperforms the state-of-the-art co-speech gesture generation methods by both quantitative and qualitative evaluations. Code and models will be publicly available.
>
---
## 更新

#### [replaced 001] Configurations, Tessellations and Tone Networks
- **分类: math.CO; eess.AS; math.AG**

- **简介: 该论文属音乐理论与几何建模交叉任务，旨在用组合几何方法解析调性网络。它构建并分析Eulerian tonnetz的Levi图与{12₃}构型，推广至五声音阶、十二音体系，并松弛约束生成Kepler式平面铺砌；还基于Tristan类四和弦提出新{12₃}构型，用于浪漫派音乐分析。**

- **链接: [https://arxiv.org/pdf/2505.08752v4](https://arxiv.org/pdf/2505.08752v4)**

> **作者:** Jeffrey R. Boland; Lane P. Hughston
>
> **备注:** 55 pages, 21 figures, new title
>
> **摘要:** The Eulerian tonnetz, which associates three minor chords to each major chord and three major chords to each minor chord, can be represented by a bipartite graph with twelve white vertices denoting major chords and twelve black vertices denoting minor chords. This so-called Levi graph determines a configuration of twelve points and twelve lines in $\mathbb R^2$ with the property that three points lie on each line and three lines pass through each point. Interesting features of the tonnetz, such as the existence of the four hexatonic hexacycles and the three octatonic octacycles, crucial for the understanding of nineteenth-century harmony and voice leading, can be read off rather directly as properties of this $\{12_3\}$ and its Levi graph. Analogous tone networks together with their associated Levi graphs and configurations can be constructed for pentatonic music and twelve-tone music, offering the promise of new methods of composition. When the constraints of the Eulerian tonnetz are relaxed so as to allow movements between major and minor triads with variations at exactly two tones, the resulting bipartite graph has two components, each of which generates a tessellation of the plane, of a type known to Kepler, based on hexagons, squares and dodecagons. When the same combinatorial idea is applied to tetrachords of the Tristan genus (dominant sevenths and minor sixths) the cycles of the resulting bipartite graph are sufficiently ample in girth to ensure the existence of a second geometrical configuration of type $\{12_3\}$, distinct from the Eulerian tonnetz as an incidence geometry, which can be used as the basis for a new approach to the analysis of the music of Chopin, Wagner, Tchaikovsky, Brahms and their contemporaries.
>
---
#### [replaced 002] Generative AI-based data augmentation for improved bioacoustic classification in noisy environments
- **分类: cs.SD; eess.AS; stat.AP**

- **简介: 该论文属生物声学分类任务，旨在解决稀有物种音频数据稀缺及风场噪声干扰导致模型性能差的问题。作者提出用ACGAN和DDPM生成逼真鸟鸣频谱图进行数据增强，验证DDPM效果更优，并构建了640小时爱尔兰风场鸟类音频新数据集。**

- **链接: [https://arxiv.org/pdf/2412.01530v3](https://arxiv.org/pdf/2412.01530v3)**

> **作者:** Anthony Gibbons; Emma King; Ian Donohue; Andrew Parnell
>
> **备注:** 25 pages, 4 tables, 7 figures
>
> **摘要:** Obtaining data to train robust artificial intelligence (AI)-based models for species classification can be challenging, particularly for rare species. Data augmentation can boost classification accuracy by increasing the diversity of training data and is cheaper to obtain than expert-labelled data. However, many classic image-based augmentation techniques are not suitable for audio spectrograms. We investigate two generative AI models as data augmentation tools to synthesise spectrograms and supplement audio data: Auxiliary Classifier Generative Adversarial Networks (ACGAN) and Denoising Diffusion Probabilistic Models (DDPMs). The latter performed particularly well in terms of both realism of generated spectrograms and accuracy in a resulting classification task. Alongside these new approaches, we present a new audio data set of 640 hours of bird calls from wind farm sites in Ireland, approximately 800 samples of which have been labelled by experts. Wind farm data are particularly challenging for classification models given the background wind and turbine noise. Training an ensemble of classification models on real and synthetic data combined compared well with highly confident BirdNET predictions. Each classifier we used was improved by including synthetic data, and classification metrics generally improved in line with the amount of synthetic data added. Our approach can be used to augment acoustic signals for more species and other land-use types, and has the potential to bring about advances in our capacity to develop reliable AI-based detection of rare species. Our code is available at https://github.com/gibbona1/SpectrogramGenAI.
>
---
#### [replaced 003] Layer-aware TDNN: Speaker Recognition Using Multi-Layer Features from Pre-Trained Models
- **分类: eess.AS; cs.AI**

- **简介: 该论文面向说话人识别任务，解决现有方法未充分利用预训练SSL模型多层特征的问题。提出层感知TDNN（L-TDNN），直接建模层-帧维度，融合层感知卷积、自适应层聚合与注意力统计池化，提升验证性能，兼顾模型紧凑性与推理效率。**

- **链接: [https://arxiv.org/pdf/2409.07770v2](https://arxiv.org/pdf/2409.07770v2)**

> **作者:** Jin Sob Kim; Hyun Joon Park; Wooseok Shin; Juan Yun; Sung Won Han
>
> **备注:** Accepted for publication in ICAIIC 2026
>
> **摘要:** Recent advances in self-supervised learning (SSL) on Transformers have significantly improved speaker verification (SV) by providing domain-general speech representations. However, existing approaches have underutilized the multi-layered nature of SSL encoders. To address this limitation, we propose the layer-aware time-delay neural network (L-TDNN), which directly performs layer/frame-wise processing on the layer-wise hidden state outputs from pre-trained models, extracting fixed-size speaker vectors. L-TDNN comprises a layer-aware convolutional network, a frame-adaptive layer aggregation, and attentive statistic pooling, explicitly modeling of the recognition and processing of previously overlooked layer dimension. We evaluated L-TDNN across multiple speech SSL Transformers and diverse speech-speaker corpora against other approaches for leveraging pre-trained encoders. L-TDNN consistently demonstrated robust verification performance, achieving the lowest error rates throughout the experiments. Concurrently, it stood out in terms of model compactness and exhibited inference efficiency comparable to the existing systems. These results highlight the advantages derived from the proposed layer-aware processing approach. Future work includes exploring joint training with SSL frontends and the incorporation of score calibration to further enhance state-of-the-art verification performance.
>
---
#### [replaced 004] SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SAC神经语音编解码器，解决现有语音编解码器难以兼顾高质量重建与语义丰富性的问题。通过语义-声学双流量化实现表征解耦，在重建质量、自然度、可懂度及语义表达上均显著提升，并支持高性能单阶段AR文本到语音合成。**

- **链接: [https://arxiv.org/pdf/2510.16841v2](https://arxiv.org/pdf/2510.16841v2)**

> **作者:** Wenxi Chen; Xinsheng Wang; Ruiqi Yan; Yushen Chen; Zhikang Niu; Ziyang Ma; Xiquan Li; Yuzhe Liang; Hanlin Wen; Shunshun Yin; Ming Tao; Xie Chen
>
> **摘要:** Speech codecs that convert continuous speech signals into discrete tokens have become essential for speech language models. However, existing codecs struggle to balance high-quality reconstruction with semantically rich representations, limiting their effectiveness in both generative and understanding tasks. In this work, we propose SAC, a neural speech codec with semantic-acoustic dual-stream quantization. By disentangling semantic and acoustic modeling into two dedicated streams, SAC enables each to be optimized for its respective role. Comprehensive evaluations show that SAC achieves strong reconstruction performance across diverse bitrates under both clean and noisy conditions, with particularly high scores on UTMOS and WER, indicating superior naturalness and intelligibility. Moreover, SAC substantially surpasses prior codecs in semantic representation, approaching the level of continuous self-supervised embeddings. When used as a tokenizer for LLM-based text-to-speech, SAC enables a single-stage autoregressive (AR) TTS model that clearly outperforms state-of-the-art AR systems. Our disentanglement analysis further validates the effectiveness of the dual-stream design, offering new potential for controllable speech generation.
>
---
#### [replaced 005] Audio-Visual Speech Enhancement: Architectural Design and Deployment Strategies
- **分类: cs.SD; eess.SP**

- **简介: 该论文研究音频-视觉语音增强（AVSE）任务，旨在提升噪声环境下的语音质量与可懂度。提出基于CNN-LSTM的多模态融合模型，并对比云、边缘辅助和终端三种部署架构，在延迟、计算开销和语音质量间权衡优化。**

- **链接: [https://arxiv.org/pdf/2508.08468v2](https://arxiv.org/pdf/2508.08468v2)**

> **作者:** Anis Hamadouche; Haifeng Luo; Mathini Sellathurai; Tharm Ratnarajah
>
> **摘要:** This paper introduces a new AI-based Audio-Visual Speech Enhancement (AVSE) system and presents a comparative performance analysis of different deployment architectures. The proposed AVSE system employs convolutional neural networks (CNNs) for spectral feature extraction and long short-term memory (LSTM) networks for temporal modeling, enabling robust speech enhancement through multimodal fusion of audio and visual cues. Multiple deployment scenarios are investigated, including cloud-based, edge-assisted, and standalone device implementations. Their performance is evaluated in terms of speech quality improvement, latency, and computational overhead. Real-world experiments are conducted across various network conditions, including Ethernet, Wi-Fi, 4G, and 5G, to analyze the trade-offs between processing delay, communication latency, and perceptual speech quality. The results show that while cloud deployment achieves the highest enhancement quality, edge-assisted architectures offer the best balance between latency and intelligibility, meeting real-time requirements under 5G and Wi-Fi 6 conditions. These findings provide practical guidelines for selecting and optimizing AVSE deployment architectures in diverse applications, including assistive hearing devices, telepresence, and industrial communications.
>
---
#### [replaced 006] RapVerse: Coherent Vocals and Whole-Body Motions Generations from Text
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出RapVerse任务：从文本歌词同步生成3D全身动作与说唱人声。为解决现有工作孤立处理音频与动作的问题，作者构建RapVerse数据集，并设计统一多模态Transformer框架，联合建模文本、量化语音与离散动作token，实现高一致性联合生成。**

- **链接: [https://arxiv.org/pdf/2405.20336v3](https://arxiv.org/pdf/2405.20336v3)**

> **作者:** Jiaben Chen; Xin Yan; Yihang Chen; Siyuan Cen; Zixin Wang; Qinwei Ma; Haoyu Zhen; Kaizhi Qian; Lie Lu; Chuang Gan
>
> **备注:** ICCV 2025, Project website: https://jiabenchen.github.io/RapVerse/
>
> **摘要:** In this work, we introduce a challenging task for simultaneously generating 3D holistic body motions and singing vocals directly from textual lyrics inputs, advancing beyond existing works that typically address these two modalities in isolation. To facilitate this, we first collect the RapVerse dataset, a large dataset containing synchronous rapping vocals, lyrics, and high-quality 3D holistic body meshes. With the RapVerse dataset, we investigate the extent to which scaling autoregressive multimodal transformers across language, audio, and motion can enhance the coherent and realistic generation of vocals and whole-body human motions. For modality unification, a vector-quantized variational autoencoder is employed to encode whole-body motion sequences into discrete motion tokens, while a vocal-to-unit model is leveraged to obtain quantized audio tokens preserving content, prosodic information and singer identity. By jointly performing transformer modeling on these three modalities in a unified way, our framework ensures a seamless and realistic blend of vocals and human motions. Extensive experiments demonstrate that our unified generation framework not only produces coherent and realistic singing vocals alongside human motions directly from textual inputs, but also rivals the performance of specialized single-modality generation systems, establishing new benchmarks for joint vocal-motion generation.
>
---
#### [replaced 007] SwinSRGAN: Swin Transformer-based Generative Adversarial Network for High-Fidelity Speech Super-Resolution
- **分类: cs.SD; eess.AS**

- **简介: 该论文面向语音超分辨率任务，旨在从低采样率语音中高保真重建高频内容。针对现有方法存在表征失配、过平滑及计算昂贵等问题，提出SwinSRGAN：基于Swin Transformer的端到端MDCT域生成对抗网络，融合多尺度时域与多频带MDCT判别器，并引入稀疏感知正则化，支持实时跨采样率上采样至48 kHz。**

- **链接: [https://arxiv.org/pdf/2509.03913v3](https://arxiv.org/pdf/2509.03913v3)**

> **作者:** Jiajun Yuan; Xiaochen Wang; Yuhang Xiao; Yulin Wu; Chenhao Hu; Xueyang Lv
>
> **备注:** 5 pages Submitted to ICASSP 2026
>
> **摘要:** Speech super-resolution (SR) reconstructs high-frequency content from low-resolution speech signals. Existing systems often suffer from representation mismatch in two-stage mel-vocoder pipelines and from over-smoothing of hallucinated high-band content by CNN-only generators. Diffusion and flow models are computationally expensive, and their robustness across domains and sampling rates remains limited. We propose SwinSRGAN, an end-to-end framework operating on Modified Discrete Cosine Transform (MDCT) magnitudes. It is a Swin Transformer-based U-Net that captures long-range spectro-temporal dependencies with a hybrid adversarial scheme combines time-domain MPD/MSD discriminators with a multi-band MDCT discriminator specialized for the high-frequency band. We employs a sparse-aware regularizer on arcsinh-compressed MDCT to better preserve transient components. The system upsamples inputs at various sampling rates to 48 kHz in a single pass and operates in real time. On standard benchmarks, SwinSRGAN reduces objective error and improves ABX preference scores. In zero-shot tests on HiFi-TTS without fine-tuning, it outperforms NVSR and mdctGAN, demonstrating strong generalization across datasets
>
---
#### [replaced 008] Protecting Bystander Privacy via Selective Hearing in Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦音频大模型的旁观者隐私保护任务，解决其在多说话人场景中无意泄露旁观者语音信息的问题。提出首个选择性听觉基准SH-Bench和新指标Selective Efficacy，并设计旁观者隐私微调（BPFT）方法，显著提升模型对主讲人理解与旁观者隐私保护的双重能力。**

- **链接: [https://arxiv.org/pdf/2512.06380v2](https://arxiv.org/pdf/2512.06380v2)**

> **作者:** Xiao Zhan; Guangzhi Sun; Jose Such; Phil Woodland
>
> **备注:** Dataset: https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench
>
> **摘要:** Audio Large language models (LLMs) are increasingly deployed in the real world, where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences did not consider. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures, including both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. In addition, we propose Selective Efficacy (SE), a novel metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LLMs reveals substantial bystander privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we also present Bystander Privacy Fine-Tuning (BPFT), a novel training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. We show that BPFT yields substantial gains, achieving an absolute 47% higher bystander accuracy under selective mode and an absolute 16% higher SE compared to Gemini 2.5 Pro, which is the best audio LLM without BPFT. Together, SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio LLMs.
>
---
#### [replaced 009] MR-FlowDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation
- **分类: cs.SD**

- **简介: 该论文属文本到音乐生成任务，旨在解决生成音乐与人类主观偏好对齐难的问题。提出MR-FlowDPO方法：基于流匹配模型，融合多维音乐奖励（文本对齐、音质、语义一致性），通过多奖励DPO优化与提示增强，并引入自监督表征提升节奏稳定性。**

- **链接: [https://arxiv.org/pdf/2512.10264v2](https://arxiv.org/pdf/2512.10264v2)**

> **作者:** Alon Ziv; Sanyuan Chen; Andros Tjandra; Yossi Adi; Wei-Ning Hsu; Bowen Shi
>
> **摘要:** A key challenge in music generation models is their lack of direct alignment with human preferences, as music evaluation is inherently subjective and varies widely across individuals. We introduce MR-FlowDPO, a novel approach that enhances flow-matching-based music generation models - a major class of modern music generative models, using Direct Preference Optimization (DPO) with multiple musical rewards. The rewards are crafted to assess music quality across three key dimensions: text alignment, audio production quality, and semantic consistency, utilizing scalable off-the-shelf models for each reward prediction. We employ these rewards in two ways: (i) By constructing preference data for DPO and (ii) by integrating the rewards into text prompting. To address the ambiguity in musicality evaluation, we propose a novel scoring mechanism leveraging semantic self-supervised representations, which significantly improves the rhythmic stability of generated music. We conduct an extensive evaluation using a variety of music-specific objective metrics as well as a human study. Results show that MR-FlowDPO significantly enhances overall music generation quality and is consistently preferred over highly competitive baselines in terms of audio quality, text alignment, and musicality. Our code is publicly available at https://github.com/lonzi/mrflow_dpo. Samples are provided in our demo page at https://lonzi.github.io/mr_flowdpo_demopage/.
>
---
#### [replaced 010] DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components
- **分类: cs.SD**

- **简介: 该论文面向音频深度伪造检测任务，旨在解决现有方法泛化性差、难以适应新伪造技术及多任务（如归因、定位）的问题。作者分析音频编码器与文本LLM组件影响，提出DFALLM架构，显著提升跨域检测与多任务性能，达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.08403v2](https://arxiv.org/pdf/2512.08403v2)**

> **作者:** Yupei Li; Li Wang; Yuxiang Wang; Lei Wang; Rizhao Cai; Jie Shi; Björn W. Schuller; Zhizheng Wu
>
> **摘要:** Audio deepfake detection has recently garnered public concern due to its implications for security and reliability. Traditional deep learning methods have been widely applied to this task but often lack generalisability when confronted with newly emerging spoofing techniques and more tasks such as spoof attribution recognition rather than simple binary classification. In principle, Large Language Models (LLMs) are considered to possess the needed generalisation capabilities. However, previous research on Audio LLMs (ALLMs) indicates a generalization bottleneck in audio deepfake detection performance, even when sufficient data is available. Consequently, this study investigates the model architecture and examines the effects of the primary components of ALLMs, namely the audio encoder and the text-based LLM. Our experiments demonstrate that the careful selection and combination of audio encoders and text-based LLMs are crucial for unlocking the deepfake detection potential of ALLMs. We further propose an ALLM structure capable of generalizing deepfake detection abilities to out-of-domain spoofing tests and other deepfake tasks, such as spoof positioning and spoof attribution recognition. Our proposed model architecture achieves state-of-the-art (SOTA) performance across multiple datasets, including ASVSpoof2019, InTheWild, and Demopage, with accuracy reaching up to 95.76% on average, and exhibits competitive capabilities in other deepfake detection tasks such as attribution, and localisation compared to SOTA audio understanding models. Data and codes are provided in supplementary materials.
>
---
