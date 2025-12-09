# 音频 cs.SD;  eess.AS

- **最新发布 23 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Physics-Guided Deepfake Detection for Voice Authentication Systems
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对语音认证系统面临的深度伪造攻击与联邦学习中的控制面投毒问题，提出一种融合物理引导特征与不确定性感知的边缘学习框架，通过多模态集成和贝叶斯方法实现鲁棒的深伪检测。**

- **链接: [https://arxiv.org/pdf/2512.06040v1](https://arxiv.org/pdf/2512.06040v1)**

> **作者:** Alireza Mohammadi; Keshav Sood; Dhananjay Thiruvady; Asef Nazari
>
> **摘要:** Voice authentication systems deployed at the network edge face dual threats: a) sophisticated deepfake synthesis attacks and b) control-plane poisoning in distributed federated learning protocols. We present a framework coupling physics-guided deepfake detection with uncertainty-aware in edge learning. The framework fuses interpretable physics features modeling vocal tract dynamics with representations coming from a self-supervised learning module. The representations are then processed via a Multi-Modal Ensemble Architecture, followed by a Bayesian ensemble providing uncertainty estimates. Incorporating physics-based characteristics evaluations and uncertainty estimates of audio samples allows our proposed framework to remain robust to both advanced deepfake attacks and sophisticated control-plane poisoning, addressing the complete threat model for networked voice authentication.
>
---
#### [new 002] Introduction to Ambisonics, Part 1: The Part With No Math
- **分类: eess.AS**

- **简介: 该论文属于技术介绍类任务，旨在帮助读者直观理解Ambisonics的基本概念。它解释了Ambisonic信号的获取、处理与播放方法，不涉及数学细节，并提供音频示例辅助理解，适用于希望实践应用的读者。**

- **链接: [https://arxiv.org/pdf/2512.07570v1](https://arxiv.org/pdf/2512.07570v1)**

> **作者:** Jens Ahrens
>
> **摘要:** The present document is Part 1 of a 2-part introduction to ambisonics and aims at readers who would like to work practically with ambisonics. We leave out deep technical details in this part and focus on helping the reader to develop an intuitive understanding of the underlying concept. We explain what ambisonic signals are, how they can be obtained, what manipulations can be applied to them, and how they can be reproduced to a listener. We provide a variety of audio examples that illustrate the matter. Part 2 of this introduction into ambisonics is provided in a separate document and aims at readers who would like to understand the mathematical details.
>
---
#### [new 003] XM-ALIGN: Unified Cross-Modal Embedding Alignment for Face-Voice Association
- **分类: cs.SD; cs.CV**

- **简介: 该论文针对跨模态人脸-语音关联任务，解决不同语言下音视频模态对齐问题。提出XM-ALIGN框架，结合显式与隐式对齐机制，通过共享分类器和MSE损失优化嵌入对齐，并采用数据增强提升性能，在MAV-Celeb数据集上表现优异。**

- **链接: [https://arxiv.org/pdf/2512.06757v1](https://arxiv.org/pdf/2512.06757v1)**

> **作者:** Zhihua Fang; Shumei Tao; Junxu Wang; Liang He
>
> **备注:** FAME 2026 Technical Report
>
> **摘要:** This paper introduces our solution, XM-ALIGN (Unified Cross-Modal Embedding Alignment Framework), proposed for the FAME challenge at ICASSP 2026. Our framework combines explicit and implicit alignment mechanisms, significantly improving cross-modal verification performance in both "heard" and "unheard" languages. By extracting feature embeddings from both face and voice encoders and jointly optimizing them using a shared classifier, we employ mean squared error (MSE) as the embedding alignment loss to ensure tight alignment between modalities. Additionally, data augmentation strategies are applied during model training to enhance generalization. Experimental results show that our approach demonstrates superior performance on the MAV-Celeb dataset. The code will be released at https://github.com/PunkMale/XM-ALIGN.
>
---
#### [new 004] Incorporating Structure and Chord Constraints in Symbolic Transformer-based Melodic Harmonization
- **分类: cs.SD; cs.AI; cs.SC**

- **简介: 该论文研究基于Transformer的旋律和声化任务，解决预设和弦约束的融入问题。提出B*算法，结合束搜索、A*与回溯，强制模型在指定位置生成目标和弦，兼顾结构约束，为后续优化提供基础。**

- **链接: [https://arxiv.org/pdf/2512.07627v1](https://arxiv.org/pdf/2512.07627v1)**

> **作者:** Maximos Kaliakatsos-Papakostas; Konstantinos Soiledis; Theodoros Tsamis; Dimos Makris; Vassilis Katsouros; Emilios Cambouropoulos
>
> **备注:** Proceedings of the 6th Conference on AI Music Creativity (AIMC 2025), Brussels, Belgium, September 10th-12th
>
> **摘要:** Transformer architectures offer significant advantages regarding the generation of symbolic music; their capabilities for incorporating user preferences toward what they generate is being studied under many aspects. This paper studies the inclusion of predefined chord constraints in melodic harmonization, i.e., where a desired chord at a specific location is provided along with the melody as inputs and the autoregressive transformer model needs to incorporate the chord in the harmonization that it generates. The peculiarities of involving such constraints is discussed and an algorithm is proposed for tackling this task. This algorithm is called B* and it combines aspects of beam search and A* along with backtracking to force pretrained transformers to satisfy the chord constraints, at the correct onset position within the correct bar. The algorithm is brute-force and has exponential complexity in the worst case; however, this paper is a first attempt to highlight the difficulties of the problem and proposes an algorithm that offers many possibilities for improvements since it accommodates the involvement of heuristics.
>
---
#### [new 005] MultiAPI Spoof: A Multi-API Dataset and Local-Attention Network for Speech Anti-spoofing Detection
- **分类: cs.SD**

- **简介: 该论文聚焦语音反欺骗任务，针对现有数据集API多样性不足的问题，构建了包含30种API的多源伪造语音数据集MultiAPI Spoof，并提出Nes2Net-LA模型，通过局部注意力增强细粒度特征提取，提升对未知和多样化伪造语音的检测性能。**

- **链接: [https://arxiv.org/pdf/2512.07352v1](https://arxiv.org/pdf/2512.07352v1)**

> **作者:** Xueping Zhang; Zhenshan Zhang; Yechen Wang; Linxi Li; Liwei Jin; Ming Li
>
> **摘要:** Existing speech anti-spoofing benchmarks rely on a narrow set of public models, creating a substantial gap from real-world scenarios in which commercial systems employ diverse, often proprietary APIs. To address this issue, we introduce MultiAPI Spoof, a multi-API audio anti-spoofing dataset comprising about 230 hours of synthetic speech generated by 30 distinct APIs, including commercial services, open-source models, and online platforms. Based on this dataset, we define the API tracing task, enabling fine-grained attribution of spoofed audio to its generation source. We further propose Nes2Net-LA, a local-attention enhanced variant of Nes2Net that improves local context modeling and fine-grained spoofing feature extraction. Experiments show that Nes2Net-LA achieves state-of-the-art performance and offers superior robustness, particularly under diverse and unseen spoofing conditions. Code \footnote{https://github.com/XuepingZhang/MultiAPI-Spoof} and dataset \footnote{https://xuepingzhang.github.io/MultiAPI-Spoof-Dataset/} have released.
>
---
#### [new 006] What Needs to be Known in Order to Perform a Meaningful Scientific Comparison Between Animal Communications and Human Spoken Language
- **分类: cs.SD**

- **简介: 该论文探讨如何科学比较动物交流与人类口语，指出需评估七个关键因素。属于理论分析任务，旨在明确比较的前提条件，解决因标准缺失导致的误比问题，提出系统性评估框架。**

- **链接: [https://arxiv.org/pdf/2512.06890v1](https://arxiv.org/pdf/2512.06890v1)**

> **作者:** Roger K. Moore
>
> **备注:** 5 pages, 1 figure, Proc. Vocal Interactivity in-and-between Humans, Animals and Robots (VIHAR-24), Kos, Greece, 6 Sept. 2024
>
> **摘要:** Human spoken language has long been the subject of scientific investigation, particularly with regard to the mechanisms underpinning speech production. Likewise, the study of animal communications has a substantial literature, with many studies focusing on vocalisation. More recently, there has been growing interest in comparing animal communications and human speech. However, it is proposed here that such a comparison necessitates the appraisal of a minimum set of critical phenomena: i) the number of degrees-of-freedom of the vocal apparatus, ii) the ability to control those degrees-of-freedom independently, iii) the properties of the acoustic environment in which communication takes place, iv) the perceptual salience of the generated sounds, v) the degree to which sounds are contrastive, vi) the presence/absence of compositionality, and vii) the information rate(s) of the resulting communications.
>
---
#### [new 007] Technical Report of Nomi Team in the Environmental Sound Deepfake Detection Challenge 2026
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对环境声音深伪检测任务，解决 unseen 生成器和低资源黑盒场景下的检测难题。团队提出音频-文本跨注意力模型，在 EnvSDD 数据集上取得优于基线的 EER 表现。**

- **链接: [https://arxiv.org/pdf/2512.06041v1](https://arxiv.org/pdf/2512.06041v1)**

> **作者:** Candy Olivia Mawalim; Haotian Zhang; Shogo Okada
>
> **摘要:** This paper presents our work for the ICASSP 2026 Environmental Sound Deepfake Detection (ESDD) Challenge. The challenge is based on the large-scale EnvSDD dataset that consists of various synthetic environmental sounds. We focus on addressing the complexities of unseen generators and low-resource black-box scenarios by proposing an audio-text cross-attention model. Experiments with individual and combined text-audio models demonstrate competitive EER improvements over the challenge baseline (BEATs+AASIST model).
>
---
#### [new 008] JEPA as a Neural Tokenizer: Learning Robust Speech Representations with Density Adaptive Attention
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文提出一种两阶段自监督语音表征学习框架，结合JEPA与密度自适应注意力机制，通过语义特征学习和高效神经量化，实现低码率、高保真的语音表示与重建，提升压缩效率与语言模型兼容性。**

- **链接: [https://arxiv.org/pdf/2512.07168v1](https://arxiv.org/pdf/2512.07168v1)**

> **作者:** Georgios Ioannides; Christos Constantinou; Aman Chadha; Aaron Elkins; Linsey Pang; Ravid Shwartz-Ziv; Yann LeCun
>
> **备注:** UniReps: Unifying Representations in Neural Models (NeurIPS 2025 Workshop)
>
> **摘要:** We introduce a two-stage self-supervised framework that combines the Joint-Embedding Predictive Architecture (JEPA) with a Density Adaptive Attention Mechanism (DAAM) for learning robust speech representations. Stage~1 uses JEPA with DAAM to learn semantic audio features via masked prediction in latent space, fully decoupled from waveform reconstruction. Stage~2 leverages these representations for efficient tokenization using Finite Scalar Quantization (FSQ) and a mixed-radix packing scheme, followed by high-fidelity waveform reconstruction with a HiFi-GAN decoder. By integrating Gaussian mixture-based density-adaptive gating into the JEPA encoder, the model performs adaptive temporal feature selection and discovers hierarchical speech structure at a low frame rate of 2.5~Hz. The resulting tokens (47.5 tokens/sec) provide a reversible, highly compressed, and language-model-friendly representation that is competitive with, and often more efficient than, existing neural audio codecs.
>
---
#### [new 009] Multi-Accent Mandarin Dry-Vocal Singing Dataset: Benchmark for Singing Accent Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文聚焦歌唱口音识别任务，旨在解决现有歌唱数据集缺乏地域口音标注和干声细节的问题。作者构建了包含4206人、覆盖九个地区的多口音普通话干声歌唱数据集MADVSD，并开展基准实验，验证其在歌唱口音分析与方言影响研究中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.07005v1](https://arxiv.org/pdf/2512.07005v1)**

> **作者:** Zihao Wang; Ruibin Yuan; Ziqi Geng; Hengjia Li; Xingwei Qu; Xinyi Li; Songye Chen; Haoying Fu; Roger B. Dannenberg; Kejun Zhang
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Singing accent research is underexplored compared to speech accent studies, primarily due to the scarcity of suitable datasets. Existing singing datasets often suffer from detail loss, frequently resulting from the vocal-instrumental separation process. Additionally, they often lack regional accent annotations. To address this, we introduce the Multi-Accent Mandarin Dry-Vocal Singing Dataset (MADVSD). MADVSD comprises over 670 hours of dry vocal recordings from 4,206 native Mandarin speakers across nine distinct Chinese regions. In addition to each participant recording audio of three popular songs in their native accent, they also recorded phonetic exercises covering all Mandarin vowels and a full octave range. We validated MADVSD through benchmark experiments in singing accent recognition, demonstrating its utility for evaluating state-of-the-art speech models in singing contexts. Furthermore, we explored dialectal influences on singing accent and analyzed the role of vowels in accentual variations, leveraging MADVSD's unique phonetic exercises.
>
---
#### [new 010] DreamFoley: Scalable VLMs for High-Fidelity Video-to-Audio Generation
- **分类: cs.SD; cs.MM**

- **简介: 该论文研究视频到音频生成任务，旨在解决生成与视频同步的高质量音频问题。提出DreamFoley模型，基于大视觉语言模型，结合双视觉编码器、残差向量量化音频 tokenizer 和无分类器引导策略，提升音视频对齐与生成质量，并构建高效数据 pipeline 支持大规模训练。**

- **链接: [https://arxiv.org/pdf/2512.06022v1](https://arxiv.org/pdf/2512.06022v1)**

> **作者:** Fu Li; Weichao Zhao; You Li; Zhichao Zhou; Dongliang He
>
> **备注:** 10 pages; Bytedance
>
> **摘要:** Recent advances in video generation have achieved remarkable improvements in visual content fidelity. However, the absence of synchronized audio severely undermines immersive experience and restricts practical applications of these technologies. To address this challenge, several pioneering works have explored diffusion transformer architectures for generating plausible video-synchronized audio, including Kling-foley, HunyuanVideo-foley and Thinksound. Distinct from existing works, we introduce an autoregressive audio generation architecture (DreamFoley) that harnesses the capabilities of large vision-language models (VLMs) to jointly model sequential interactions among video, audio, and text modalities. Our approach features a dual-visual encoder module that effectively captures both audio-aligned and text-aligned visual features. Additionally, we employ a Residual Vector Quantization audio tokenizer with a delay-pattern generation scheme to balance the trade-off between training efficiency and audio quality. Moreover, we introduce the classifier-free guidance strategy into VLMs to bootstrap generated audio quality. Furthermore, we establish an efficient data production pipeline to scale audio-video-text triple collection. Finally, extensive experiments are conducted to validate the effectiveness of our model, achieving promising performance across popular benchmarks. We hope that the findings in this study provide a strong foundation for future video-to-audio generation research. We also release the previously missing audio-visual textual descriptions from the public benchmark, aiming to facilitate subsequent researchers in conducting more convenient and effective evaluations and comparisons.
>
---
#### [new 011] Who Will Top the Charts? Multimodal Music Popularity Prediction via Adaptive Fusion of Modality Experts and Temporal Engagement Modeling
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文研究音乐流行度预测任务，旨在解决现有方法忽略时序动态、歌词结构、历史表现及模态融合不佳的问题。作者提出GAMENet模型，通过自适应融合音频、歌词与社会元数据专家网络，并引入职业轨迹特征，显著提升预测性能。**

- **链接: [https://arxiv.org/pdf/2512.06259v1](https://arxiv.org/pdf/2512.06259v1)**

> **作者:** Yash Choudhary; Preeti Rao; Pushpak Bhattacharyya
>
> **备注:** 8 pages
>
> **摘要:** Predicting a song's commercial success prior to its release remains an open and critical research challenge for the music industry. Early prediction of music popularity informs strategic decisions, creative planning, and marketing. Existing methods suffer from four limitations:(i) temporal dynamics in audio and lyrics are averaged away; (ii) lyrics are represented as a bag of words, disregarding compositional structure and affective semantics; (iii) artist- and song-level historical performance is ignored; and (iv) multimodal fusion approaches rely on simple feature concatenation, resulting in poorly aligned shared representations. To address these limitations, we introduce GAMENet, an end-to-end multimodal deep learning architecture for music popularity prediction. GAMENet integrates modality-specific experts for audio, lyrics, and social metadata through an adaptive gating mechanism. We use audio features from Music4AllOnion processed via OnionEnsembleAENet, a network of autoencoders designed for robust feature extraction; lyric embeddings derived through a large language model pipeline; and newly introduced Career Trajectory Dynamics (CTD) features that capture multi-year artist career momentum and song-level trajectory statistics. Using the Music4All dataset (113k tracks), previously explored in MIR tasks but not popularity prediction, GAMENet achieves a 12% improvement in R^2 over direct multimodal feature concatenation. Spotify audio descriptors alone yield an R^2 of 0.13. Integrating aggregate CTD features increases this to 0.69, with an additional 7% gain from temporal CTD features. We further validate robustness using the SpotGenTrack Popularity Dataset (100k tracks), achieving a 16% improvement over the previous baseline. Extensive ablations confirm the model's effectiveness and the distinct contribution of each modality.
>
---
#### [new 012] KidSpeak: A General Multi-purpose LLM for Kids' Speech Recognition and Screening
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对儿童语音识别与语言障碍筛查任务，解决现有模型因依赖成人语音数据而对儿童语音表现差的问题。提出KidSpeak多任务语音大模型及FASA自动对齐工具，提升儿童语音数据质量与模型性能。**

- **链接: [https://arxiv.org/pdf/2512.05994v1](https://arxiv.org/pdf/2512.05994v1)**

> **作者:** Rohan Sharma; Dancheng Liu; Jingchen Sun; Shijie Zhou; Jiayu Qin; Jinjun Xiong; Changyou Chen
>
> **摘要:** With the rapid advancement of conversational and diffusion-based AI, there is a growing adoption of AI in educational services, ranging from grading and assessment tools to personalized learning systems that provide targeted support for students. However, this adaptability has yet to fully extend to the domain of children's speech, where existing models often fail due to their reliance on datasets designed for clear, articulate adult speech. Children, particularly those in early developmental stages or with speech and language pathologies, present unique challenges that current AI models and datasets are ill-equipped to handle. To address this, we introduce KidSpeak, a multi-task speech-enhanced Foundation Model capable of both generative and discriminative tasks specifically tailored to children's speech patterns. Our framework employs a two-stage training process that incorporates phonetic knowledge into the speech encoder, achieving an average accuracy of 87% across four separate tasks. Furthermore, recognizing the limitations of scalable human annotation and existing speech alignment tools, we propose the Flexible and Automatic Speech Aligner (FASA) and leverage the method to construct high quality datasets for training and evaluation. This novel alignment tool significantly improves the quality of aligned children's speech from noisy data, enhancing data quality by 13.6x compared to human annotations, as demonstrated on the CHILDES dataset. To the best of our knowledge, KidSpeak and FASA represent the first comprehensive solution designed for speech and language therapy in children, offering both a multi-purpose speech LLM and a robust alignment tool.
>
---
#### [new 013] Singing Timbre Popularity Assessment Based on Multimodal Large Foundation Model
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究无参考的歌唱音色流行度自动评估，旨在克服依赖参考音频和单一评分的局限。提出新数据集Sing-MD、高效多模态模型VocalVerse及人类参与的H-TPR评测基准，实现多维度、描述性歌唱性能评估。**

- **链接: [https://arxiv.org/pdf/2512.06999v1](https://arxiv.org/pdf/2512.06999v1)**

> **作者:** Zihao Wang; Ruibin Yuan; Ziqi Geng; Hengjia Li; Xingwei Qu; Xinyi Li; Songye Chen; Haoying Fu; Roger B. Dannenberg; Kejun Zhang
>
> **备注:** Accepted to ACMMM 2025 oral
>
> **摘要:** Automated singing assessment is crucial for education and entertainment. However, existing systems face two fundamental limitations: reliance on reference tracks, which stifles creative expression, and the simplification of complex performances into non-diagnostic scores based solely on pitch and rhythm. We advocate for a shift from discriminative to descriptive evaluation, creating a complete ecosystem for reference-free, multi-dimensional assessment. First, we introduce Sing-MD, a large-scale dataset annotated by experts across four dimensions: breath control, timbre quality, emotional expression, and vocal technique. Our analysis reveals significant annotation inconsistencies among experts, challenging the validity of traditional accuracy-based metrics. Second, addressing the memory limitations of Multimodal Large Language Models (MLLMs) in analyzing full-length songs, we propose VocalVerse. This efficient hybrid architecture leverages a lightweight acoustic encoder to model global performance features and long-term dependencies. Third, to address automated metric shortcomings, we establish the H-TPR (Human-in-the-loop Tiered Perceptual Ranking) benchmark, which evaluates a model's ability to generate perceptually valid rankings rather than predicting noisy ground-truth scores.
>
---
#### [new 014] Unsupervised Single-Channel Audio Separation with Diffusion Source Priors
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究单通道音频分离任务，旨在解决无监督条件下缺乏配对训练数据的问题。提出基于扩散先验的无监督方法，设计新型网络架构与逆问题求解策略，通过重建引导和混合信号初始化实现高质量分离。**

- **链接: [https://arxiv.org/pdf/2512.07226v1](https://arxiv.org/pdf/2512.07226v1)**

> **作者:** Runwu Shi; Chang Li; Jiang Wang; Rui Zhang; Nabeela Khan; Benjamin Yen; Takeshi Ashizawa; Kazuhiro Nakadai
>
> **备注:** 15 pages, 31 figures, accepted by The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
>
> **摘要:** Single-channel audio separation aims to separate individual sources from a single-channel mixture. Most existing methods rely on supervised learning with synthetically generated paired data. However, obtaining high-quality paired data in real-world scenarios is often difficult. This data scarcity can degrade model performance under unseen conditions and limit generalization ability. To this end, in this work, we approach this problem from an unsupervised perspective, framing it as a probabilistic inverse problem. Our method requires only diffusion priors trained on individual sources. Separation is then achieved by iteratively guiding an initial state toward the solution through reconstruction guidance. Importantly, we introduce an advanced inverse problem solver specifically designed for separation, which mitigates gradient conflicts caused by interference between the diffusion prior and reconstruction guidance during inverse denoising. This design ensures high-quality and balanced separation performance across individual sources. Additionally, we find that initializing the denoising process with an augmented mixture instead of pure Gaussian noise provides an informative starting point that significantly improves the final performance. To further enhance audio prior modeling, we design a novel time-frequency attention-based network architecture that demonstrates strong audio modeling capability. Collectively, these improvements lead to significant performance gains, as validated across speech-sound event, sound event, and speech separation tasks.
>
---
#### [new 015] Degrading Voice: A Comprehensive Overview of Robust Voice Conversion Through Input Manipulation
- **分类: eess.AS; cs.AI; cs.CR; cs.SD**

- **简介: 该论文聚焦语音转换（VC）模型在输入退化下的鲁棒性问题，旨在探究噪声、混响等干扰对输出质量的影响。作者从输入操控角度分类攻防方法，并在可懂度、自然度等维度评估性能，最后指出开放问题与未来方向。**

- **链接: [https://arxiv.org/pdf/2512.06304v1](https://arxiv.org/pdf/2512.06304v1)**

> **作者:** Xining Song; Zhihua Wei; Rui Wang; Haixiao Hu; Yanxiang Chen; Meng Han
>
> **摘要:** Identity, accent, style, and emotions are essential components of human speech. Voice conversion (VC) techniques process the speech signals of two input speakers and other modalities of auxiliary information such as prompts and emotion tags. It changes para-linguistic features from one to another, while maintaining linguistic contents. Recently, VC models have made rapid advancements in both generation quality and personalization capabilities. These developments have attracted considerable attention for diverse applications, including privacy preservation, voice-print reproduction for the deceased, and dysarthric speech recovery. However, these models only learn non-robust features due to the clean training data. Subsequently, it results in unsatisfactory performances when dealing with degraded input speech in real-world scenarios, including additional noise, reverberation, adversarial attacks, or even minor perturbation. Hence, it demands robust deployments, especially in real-world settings. Although latest researches attempt to find potential attacks and countermeasures for VC systems, there remains a significant gap in the comprehensive understanding of how robust the VC model is under input manipulation. here also raises many questions: For instance, to what extent do different forms of input degradation attacks alter the expected output of VC models? Is there potential for optimizing these attack and defense strategies? To answer these questions, we classify existing attack and defense methods from the perspective of input manipulation and evaluate the impact of degraded input speech across four dimensions, including intelligibility, naturalness, timbre similarity, and subjective perception. Finally, we outline open issues and future directions.
>
---
#### [new 016] Protecting Bystander Privacy via Selective Hearing in LALMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究大型音频语言模型中的旁观者隐私保护问题，提出“选择性聆听”任务。作者构建SH-Bench基准和SE评估指标，发现现有模型存在隐私泄露，并提出BPFT训练方法，有效提升模型在理解主说话人的同时保护旁观者隐私的能力。**

- **链接: [https://arxiv.org/pdf/2512.06380v1](https://arxiv.org/pdf/2512.06380v1)**

> **作者:** Xiao Zhan; Guangzhi Sun; Jose Such; Phil Woodland
>
> **备注:** Dataset: https://huggingface.co/datasets/BrianatCambridge/SelectiveHearingBench
>
> **摘要:** Large audio language models (LALMs) are increasingly deployed in real-world settings where they inevitably capture speech from unintended nearby bystanders, raising privacy risks that existing benchmarks and defences largely overlook. We introduce SH-Bench, the first benchmark designed to evaluate selective hearing: a model's ability to attend to an intended main speaker while refusing to process or reveal information about incidental bystander speech. SH-Bench contains 3,968 multi-speaker audio mixtures spanning both real-world and synthetic scenarios, paired with 77k multiple-choice questions that probe models under general and selective operating modes. We propose Selective Efficacy (SE), a unified metric capturing both multi-speaker comprehension and bystander-privacy protection. Our evaluation of state-of-the-art open-source and proprietary LALMs reveals substantial privacy leakage, with strong audio understanding failing to translate into selective protection of bystander privacy. To mitigate this gap, we introduce Bystander Privacy Fine-Tuning (BPFT), a training pipeline that teaches models to refuse bystander-related queries without degrading main-speaker comprehension. BPFT yields substantial gains which improve SE by up to 15.9% over Gemini 2.5 Pro, demonstrating that selective hearing is learnable but far from achieved in current LALMs. SH-Bench and BPFT provide the first systematic framework for measuring and improving bystander privacy in audio foundation models.
>
---
#### [new 017] Lightweight Wasserstein Audio-Visual Model for Unified Speech Enhancement and Separation
- **分类: cs.CV; eess.AS**

- **简介: 该论文研究语音增强与分离的统一任务，旨在解决传统方法需多阶段、重参数且依赖监督训练的问题。提出轻量级无监督音视频模型UniVoiceLite，利用唇动和面部身份信息，结合Wasserstein正则化，实现高效、泛化性强的语音提取。**

- **链接: [https://arxiv.org/pdf/2512.06689v1](https://arxiv.org/pdf/2512.06689v1)**

> **作者:** Jisoo Park; Seonghak Lee; Guisik Kim; Taewoo Kim; Junseok Kwon
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Speech Enhancement (SE) and Speech Separation (SS) have traditionally been treated as distinct tasks in speech processing. However, real-world audio often involves both background noise and overlapping speakers, motivating the need for a unified solution. While recent approaches have attempted to integrate SE and SS within multi-stage architectures, these approaches typically involve complex, parameter-heavy models and rely on supervised training, limiting scalability and generalization. In this work, we propose UniVoiceLite, a lightweight and unsupervised audio-visual framework that unifies SE and SS within a single model. UniVoiceLite leverages lip motion and facial identity cues to guide speech extraction and employs Wasserstein distance regularization to stabilize the latent space without requiring paired noisy-clean data. Experimental results demonstrate that UniVoiceLite achieves strong performance in both noisy and multi-speaker scenarios, combining efficiency with robust generalization. The source code is available at https://github.com/jisoo-o/UniVoiceLite.
>
---
#### [new 018] DeepAgent: A Dual Stream Multi Agent Fusion for Robust Multimodal Deepfake Detection
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属多模态深度伪造检测任务，旨在解决单模型易受模态噪声和不一致影响的问题。提出DeepAgent框架，通过双流多代理（视觉与音视频不一致性分析）融合决策，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.07351v1](https://arxiv.org/pdf/2512.07351v1)**

> **作者:** Sayeem Been Zaman; Wasimul Karim; Arefin Ittesafun Abian; Reem E. Mohamed; Md Rafiqul Islam; Asif Karim; Sami Azam
>
> **摘要:** The increasing use of synthetic media, particularly deepfakes, is an emerging challenge for digital content verification. Although recent studies use both audio and visual information, most integrate these cues within a single model, which remains vulnerable to modality mismatches, noise, and manipulation. To address this gap, we propose DeepAgent, an advanced multi-agent collaboration framework that simultaneously incorporates both visual and audio modalities for the effective detection of deepfakes. DeepAgent consists of two complementary agents. Agent-1 examines each video with a streamlined AlexNet-based CNN to identify the symbols of deepfake manipulation, while Agent-2 detects audio-visual inconsistencies by combining acoustic features, audio transcriptions from Whisper, and frame-reading sequences of images through EasyOCR. Their decisions are fused through a Random Forest meta-classifier that improves final performance by taking advantage of the different decision boundaries learned by each agent. This study evaluates the proposed framework using three benchmark datasets to demonstrate both component-level and fused performance. Agent-1 achieves a test accuracy of 94.35% on the combined Celeb-DF and FakeAVCeleb datasets. On the FakeAVCeleb dataset, Agent-2 and the final meta-classifier attain accuracies of 93.69% and 81.56%, respectively. In addition, cross-dataset validation on DeepFakeTIMIT confirms the robustness of the meta-classifier, which achieves a final accuracy of 97.49%, and indicates a strong capability across diverse datasets. These findings confirm that hierarchy-based fusion enhances robustness by mitigating the weaknesses of individual modalities and demonstrate the effectiveness of a multi-agent approach in addressing diverse types of manipulations in deepfakes.
>
---
#### [new 019] Hankel-FNO: Fast Underwater Acoustic Charting Via Physics-Encoded Fourier Neural Operator
- **分类: cs.LG; cs.SD**

- **简介: 该论文研究水下声学快速建图任务，旨在解决传统数值方法计算慢、现有深度学习模型泛化差的问题。提出Hankel-FNO模型，融合物理知识与傅里叶神经算子，实现高效高精度声场预测，适应多变环境与远距离传播。**

- **链接: [https://arxiv.org/pdf/2512.06417v1](https://arxiv.org/pdf/2512.06417v1)**

> **作者:** Yifan Sun; Lei Cheng; Jianlong Li; Peter Gerstoft
>
> **摘要:** Fast and accurate underwater acoustic charting is crucial for downstream tasks such as environment-aware sensor placement optimization and autonomous vehicle path planning. Conventional methods rely on computationally expensive while accurate numerical solvers, which are not scalable for large-scale or real-time applications. Although deep learning-based surrogate models can accelerate these computations, they often suffer from limitations such as fixed-resolution constraints or dependence on explicit partial differential equation formulations. These issues hinder their applicability and generalization across diverse environments. We propose Hankel-FNO, a Fourier Neural Operator (FNO)-based model for efficient and accurate acoustic charting. By incorporating sound propagation knowledge and bathymetry, our method has high accuracy while maintaining high computational speed. Results demonstrate that Hankel-FNO outperforms traditional solvers in speed and surpasses data-driven alternatives in accuracy, especially in long-range predictions. Experiments show the model's adaptability to diverse environments and sound source settings with minimal fine-tuning.
>
---
#### [new 020] TeluguST-46: A Benchmark Corpus and Comprehensive Evaluation for Telugu-English Speech Translation
- **分类: cs.CL; eess.AS**

- **简介: 该论文聚焦泰卢固语-英语语音翻译任务，旨在解决低资源语言对研究不足的问题。作者构建了高质量基准数据集TeluguST-46，系统比较级联与端到端模型，并评估自动评测指标的有效性，为形态复杂语言的语音翻译提供可复现基准和实用指导。**

- **链接: [https://arxiv.org/pdf/2512.07265v1](https://arxiv.org/pdf/2512.07265v1)**

> **作者:** Bhavana Akkiraju; Srihari Bandarupalli; Swathi Sambangi; Vasavi Ravuri; R Vijaya Saraswathi; Anil Kumar Vuppala
>
> **备注:** Submitted to AACL IJCNLP 2025
>
> **摘要:** Despite Telugu being spoken by over 80 million people, speech translation research for this morphologically rich language remains severely underexplored. We address this gap by developing a high-quality Telugu--English speech translation benchmark from 46 hours of manually verified CSTD corpus data (30h/8h/8h train/dev/test split). Our systematic comparison of cascaded versus end-to-end architectures shows that while IndicWhisper + IndicMT achieves the highest performance due to extensive Telugu-specific training data, finetuned SeamlessM4T models demonstrate remarkable competitiveness despite using significantly less Telugu-specific training data. This finding suggests that with careful hyperparameter tuning and sufficient parallel data (potentially less than 100 hours), end-to-end systems can achieve performance comparable to cascaded approaches in low-resource settings. Our metric reliability study evaluating BLEU, METEOR, ChrF++, ROUGE-L, TER, and BERTScore against human judgments reveals that traditional metrics provide better quality discrimination than BERTScore for Telugu--English translation. The work delivers three key contributions: a reproducible Telugu--English benchmark, empirical evidence of competitive end-to-end performance potential in low-resource scenarios, and practical guidance for automatic evaluation in morphologically complex language pairs.
>
---
#### [new 021] Efficient ASR for Low-Resource Languages: Leveraging Cross-Lingual Unlabeled Data
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究低资源语言的语音识别，旨在解决标注数据少、计算资源需求高的问题。通过跨语言无监督数据持续预训练和形态感知分词，构建高效模型，在较少参数和标注数据下实现优越性能。**

- **链接: [https://arxiv.org/pdf/2512.07277v1](https://arxiv.org/pdf/2512.07277v1)**

> **作者:** Srihari Bandarupalli; Bhavana Akkiraju; Charan Devarakonda; Vamsiraghusimha Narsinga; Anil Kumar Vuppala
>
> **备注:** Accepted in AACL IJCNLP 2025
>
> **摘要:** Automatic speech recognition for low-resource languages remains fundamentally constrained by the scarcity of labeled data and computational resources required by state-of-the-art models. We present a systematic investigation into cross-lingual continuous pretraining for low-resource languages, using Perso-Arabic languages (Persian, Arabic, and Urdu) as our primary case study. Our approach demonstrates that strategic utilization of unlabeled speech data can effectively bridge the resource gap without sacrificing recognition accuracy. We construct a 3,000-hour multilingual corpus through a scalable unlabeled data collection pipeline and employ targeted continual pretraining combined with morphologically-aware tokenization to develop a 300M parameter model that achieves performance comparable to systems 5 times larger. Our model outperforms Whisper Large v3 (1.5B parameters) on Persian and achieves competitive results on Arabic and Urdu despite using significantly fewer parameters and substantially less labeled data. These findings challenge the prevailing assumption that ASR quality scales primarily with model size, revealing instead that data relevance and strategic pretraining are more critical factors for low-resource scenarios. This work provides a practical pathway toward inclusive speech technology, enabling effective ASR for underrepresented languages without dependence on massive computational infrastructure or proprietary datasets.
>
---
#### [new 022] Coherent Audio-Visual Editing via Conditional Audio Generation Following Video Edits
- **分类: cs.MM; cs.LG; cs.SD**

- **简介: 该论文研究音视频协同编辑任务，旨在解决视频编辑后音频与画面不一致的问题。提出一种新模型，通过条件音频生成，结合源音频、目标视频和文本提示，实现与视觉变化同步的音频编辑，提升音视频一致性与内容保真度。**

- **链接: [https://arxiv.org/pdf/2512.07209v1](https://arxiv.org/pdf/2512.07209v1)**

> **作者:** Masato Ishii; Akio Hayakawa; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** We introduce a novel pipeline for joint audio-visual editing that enhances the coherence between edited video and its accompanying audio. Our approach first applies state-of-the-art video editing techniques to produce the target video, then performs audio editing to align with the visual changes. To achieve this, we present a new video-to-audio generation model that conditions on the source audio, target video, and a text prompt. We extend the model architecture to incorporate conditional audio input and propose a data augmentation strategy that improves training efficiency. Furthermore, our model dynamically adjusts the influence of the source audio based on the complexity of the edits, preserving the original audio structure where possible. Experimental results demonstrate that our method outperforms existing approaches in maintaining audio-visual alignment and content integrity.
>
---
#### [new 023] A multimodal Bayesian Network for symptom-level depression and anxiety prediction from voice and speech data
- **分类: cs.LG; cs.SD**

- **简介: 该论文提出一种基于贝叶斯网络的多模态模型，用于从语音和言语数据预测抑郁和焦虑症状。旨在解决临床评估中多源信息整合难、模型可解释性差等问题，验证了在大规模数据下症状级预测的有效性、公平性及临床可用性。**

- **链接: [https://arxiv.org/pdf/2512.07741v1](https://arxiv.org/pdf/2512.07741v1)**

> **作者:** Agnes Norbury; George Fairs; Alexandra L. Georgescu; Matthew M. Nour; Emilia Molimpakis; Stefano Goria
>
> **摘要:** During psychiatric assessment, clinicians observe not only what patients report, but important nonverbal signs such as tone, speech rate, fluency, responsiveness, and body language. Weighing and integrating these different information sources is a challenging task and a good candidate for support by intelligence-driven tools - however this is yet to be realized in the clinic. Here, we argue that several important barriers to adoption can be addressed using Bayesian network modelling. To demonstrate this, we evaluate a model for depression and anxiety symptom prediction from voice and speech features in large-scale datasets (30,135 unique speakers). Alongside performance for conditions and symptoms (for depression, anxiety ROC-AUC=0.842,0.831 ECE=0.018,0.015; core individual symptom ROC-AUC>0.74), we assess demographic fairness and investigate integration across and redundancy between different input modality types. Clinical usefulness metrics and acceptability to mental health service users are explored. When provided with sufficiently rich and large-scale multimodal data streams and specified to represent common mental conditions at the symptom rather than disorder level, such models are a principled approach for building robust assessment support tools: providing clinically-relevant outputs in a transparent and explainable format that is directly amenable to expert clinical supervision.
>
---
## 更新

#### [replaced 001] DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究语音生成任务，旨在解决连续语音表征自回归生成中的高计算成本与生成质量不佳问题。提出DiTAR框架，结合语言模型与扩散Transformer，采用分块生成策略和基于时间点的温度控制，提升生成效率与音质。**

- **链接: [https://arxiv.org/pdf/2502.03930v4](https://arxiv.org/pdf/2502.03930v4)**

> **作者:** Dongya Jia; Zhuo Chen; Jiawei Chen; Chenpeng Du; Jian Wu; Jian Cong; Xiaobin Zhuang; Chumin Li; Zhen Wei; Yuping Wang; Yuxuan Wang
>
> **备注:** ByteDance Seed template, ICML 2025
>
> **摘要:** Several recent studies have attempted to autoregressively generate continuous speech representations without discrete speech tokens by combining diffusion and autoregressive models, yet they often face challenges with excessive computational loads or suboptimal outcomes. In this work, we propose Diffusion Transformer Autoregressive Modeling (DiTAR), a patch-based autoregressive framework combining a language model with a diffusion transformer. This approach significantly enhances the efficacy of autoregressive models for continuous tokens and reduces computational demands. DiTAR utilizes a divide-and-conquer strategy for patch generation, where the language model processes aggregated patch embeddings and the diffusion transformer subsequently generates the next patch based on the output of the language model. For inference, we propose defining temperature as the time point of introducing noise during the reverse diffusion ODE to balance diversity and determinism. We also show in the extensive scaling analysis that DiTAR has superb scalability. In zero-shot speech generation, DiTAR achieves state-of-the-art performance in robustness, speaker similarity, and naturalness.
>
---
#### [replaced 002] Is Self-Supervised Learning Enough to Fill in the Gap? A Study on Speech Inpainting
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究语音修复任务，探索自监督学习（SSL）模型是否可直接用于填补缺失语音。作者基于HuBERT和HiFi-GAN构建两种微调策略，验证其在不同场景下的重建效果，表明SSL预训练能有效迁移至语音修复，无需额外标注数据。**

- **链接: [https://arxiv.org/pdf/2405.20101v2](https://arxiv.org/pdf/2405.20101v2)**

> **作者:** Ihab Asaad; Maxime Jacquelin; Olivier Perrotin; Laurent Girin; Thomas Hueber
>
> **备注:** Accepted for publication to Computer Speech and Language journal (to appear)
>
> **摘要:** Speech inpainting consists in reconstructing corrupted or missing speech segments using surrounding context, a process that closely resembles the pretext tasks in Self-Supervised Learning (SSL) for speech encoders. This study investigates using SSL-trained speech encoders for inpainting without any additional training beyond the initial pretext task, and simply adding a decoder to generate a waveform. We compare this approach to supervised fine-tuning of speech encoders for a downstream task -- here, inpainting. Practically, we integrate HuBERT as the SSL encoder and HiFi-GAN as the decoder in two configurations: (1) fine-tuning the decoder to align with the frozen pre-trained encoder's output and (2) fine-tuning the encoder for an inpainting task based on a frozen decoder's input. Evaluations are conducted under single- and multi-speaker conditions using in-domain datasets and out-of-domain datasets (including unseen speakers, diverse speaking styles, and noise). Both informed and blind inpainting scenarios are considered, where the position of the corrupted segment is either known or unknown. The proposed SSL-based methods are benchmarked against several baselines, including a text-informed method combining automatic speech recognition with zero-shot text-to-speech synthesis. Performance is assessed using objective metrics and perceptual evaluations. The results demonstrate that both approaches outperform baselines, successfully reconstructing speech segments up to 200 ms, and sometimes up to 400 ms. Notably, fine-tuning the SSL encoder achieves more accurate speech reconstruction in single-speaker settings, while a pre-trained encoder proves more effective for multi-speaker scenarios. This demonstrates that an SSL pretext task can transfer to speech inpainting, enabling successful speech reconstruction with a pre-trained encoder.
>
---
#### [replaced 003] SteerMusic: Enhanced Musical Consistency for Zero-shot Text-guided and Personalized Music Editing
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文研究零样本文本引导的音乐编辑任务，旨在解决现有方法在编辑时难以保持音乐内容一致性的问题。作者提出SteerMusic和SteerMusic+，利用分数蒸馏提升编辑保真度与风格控制能力。**

- **链接: [https://arxiv.org/pdf/2504.10826v3](https://arxiv.org/pdf/2504.10826v3)**

> **作者:** Xinlei Niu; Kin Wai Cheuk; Jing Zhang; Naoki Murata; Chieh-Hsin Lai; Michele Mancusi; Woosung Choi; Giorgio Fabbro; Wei-Hsiang Liao; Charles Patrick Martin; Yuki Mitsufuji
>
> **备注:** Accepted by AAAI2026
>
> **摘要:** Music editing is an important step in music production, which has broad applications, including game development and film production. Most existing zero-shot text-guided editing methods rely on pretrained diffusion models by involving forward-backward diffusion processes. However, these methods often struggle to preserve the musical content. Additionally, text instructions alone usually fail to accurately describe the desired music. In this paper, we propose two music editing methods that improve the consistency between the original and edited music by leveraging score distillation. The first method, SteerMusic, is a coarse-grained zero-shot editing approach using delta denoising score. The second method, SteerMusic+, enables fine-grained personalized music editing by manipulating a concept token that represents a user-defined musical style. SteerMusic+ allows for the editing of music into user-defined musical styles that cannot be achieved by the text instructions alone. Experimental results show that our methods outperform existing approaches in preserving both music content consistency and editing fidelity. User studies further validate that our methods achieve superior music editing quality.
>
---
#### [replaced 004] Audio Palette: A Diffusion Transformer with Multi-Signal Conditioning for Controllable Foley Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频生成任务，旨在解决可控Foley音效合成中细粒度声学控制不足的问题。作者提出Audio Palette模型，基于扩散Transformer引入四种时变控制信号，并采用LoRA高效微调，实现精确、可解释的声音属性调控，同时保持高质量生成与语义对齐。**

- **链接: [https://arxiv.org/pdf/2510.12175v4](https://arxiv.org/pdf/2510.12175v4)**

> **作者:** Junnuo Wang
>
> **备注:** Accepted for publication in the Artificial Intelligence Technology Research (AITR)
>
> **摘要:** Recent advances in diffusion-based generative models have enabled high-quality text-to-audio synthesis, but fine-grained acoustic control remains a significant challenge in open-source research. We present Audio Palette, a diffusion transformer (DiT) based model that extends the Stable Audio Open architecture to address this "control gap" in controllable audio generation. Unlike prior approaches that rely solely on semantic conditioning, Audio Palette introduces four time-varying control signals, loudness, pitch, spectral centroid, and timbre, for precise and interpretable manipulation of acoustic features. The model is efficiently adapted for the nuanced domain of Foley synthesis using Low-Rank Adaptation (LoRA) on a curated subset of AudioSet, requiring only 0.85% of the original parameters to be trained. Experiments demonstrate that Audio Palette achieves fine-grained, interpretable control of sound attributes. Crucially, it accomplishes this novel controllability while maintaining high audio quality and strong semantic alignment to text prompts, with performance on standard metrics such as Frechet Audio Distance (FAD) and LAION-CLAP scores remaining comparable to the original baseline model. We provide a scalable, modular pipeline for audio research, emphasizing sequence-based conditioning, memory efficiency, and a three-scale classifier-free guidance mechanism for nuanced inference-time control. This work establishes a robust foundation for controllable sound design and performative audio synthesis in open-source settings, enabling a more artist-centric workflow in the broader context of music and sound information retrieval.
>
---
#### [replaced 005] MAVERIX: Multimodal Audio-Visual Evaluation and Recognition IndeX
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出MAVERIX，首个专注视频、音频、文本多模态理解的基准，旨在解决现有模型缺乏统一评估框架的问题。构建了2,556个需音视频融合理解的问题，评估先进模型性能显著低于人类，揭示理解差距。**

- **链接: [https://arxiv.org/pdf/2503.21699v2](https://arxiv.org/pdf/2503.21699v2)**

> **作者:** Liuyue Xie; Avik Kuthiala; George Z. Wei; Ce Zheng; Ananya Bal; Mosam Dabhi; Liting Wen; Taru Rustagi; Ethan Lai; Sushil Khyalia; Rohan Choudhury; Morteza Ziyadi; Xu Zhang; Hao Yang; László A. Jeni
>
> **摘要:** We introduce MAVERIX (Multimodal audiovisual Evaluation and Recognition IndeX), a unified benchmark to probe the video understanding in multimodal LLMs, encompassing video, audio, text inputs with human performance baselines. Although recent advancements in models with vision and audio understanding capabilities have shown substantial progress, the field lacks a standardized evaluation framework to thoroughly assess their cross-modality comprehension performance. MAVERIX curates 2,556 questions from 700 videos, in the form of both multiple-choice and open-ended formats, explicitly designed to evaluate multimodal models through questions that necessitate tight integration of video and audio information, spanning a broad spectrum of agentic scenarios. MAVERIX uniquely provides models with audiovisual questions, closely mimicking the multimodal perceptual experiences available to humans during inference and decision-making processes. To our knowledge, MAVERIX is the first benchmark aimed explicitly at assessing comprehensive audiovisual integration in such granularity. Experiments with state-of-the-art models, including Qwen 2.5 Omni and Gemini 2.5 Flash-Lite, show performance around 64% accuracy, while human experts reach near-ceiling performance of 92.8%, exposing a substantial gap to human-level comprehension. With standardized evaluation protocols, a rigorously annotated pipeline, and a public toolkit, MAVERIX establishes a challenging testbed for advancing audiovisual multimodal intelligence.
>
---
#### [replaced 006] Target Speaker Extraction through Comparing Noisy Positive and Negative Audio Enrollments
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究单通道目标说话人提取，解决在无纯净语音条件下利用含噪注册音频分离目标语音的问题。提出通过对比目标说话人有声与无声片段（正负注册）编码身份信息的新方法，并设计两阶段训练策略，显著提升性能并加速收敛，实现当前最优效果。**

- **链接: [https://arxiv.org/pdf/2502.16611v3](https://arxiv.org/pdf/2502.16611v3)**

> **作者:** Shitong Xu; Yiyuan Yang; Niki Trigoni; Andrew Markham
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Target speaker extraction focuses on isolating a specific speaker's voice from an audio mixture containing multiple speakers. To provide information about the target speaker's identity, prior works have utilized clean audio samples as conditioning inputs. However, such clean audio examples are not always readily available. For instance, obtaining a clean recording of a stranger's voice at a cocktail party without leaving the noisy environment is generally infeasible. Limited prior research has explored extracting the target speaker's characteristics from noisy enrollments, which may contain overlapping speech from interfering speakers. In this work, we explore a novel enrollment strategy that encodes target speaker information from the noisy enrollment by comparing segments where the target speaker is talking (Positive Enrollments) with segments where the target speaker is silent (Negative Enrollments). Experiments show the effectiveness of our model architecture, which achieves over 2.1 dB higher SI-SNRi compared to prior works in extracting the monaural speech from the mixture of two speakers. Additionally, the proposed two-stage training strategy accelerates convergence, reducing the number of optimization steps required to reach 3 dB SNR by 60%. Overall, our method achieves state-of-the-art performance in the monaural target speaker extraction conditioned on noisy enrollments. Our implementation is available at https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll .
>
---
#### [replaced 007] MuMeNet: A Network Simulator for Musical Metaverse Communications
- **分类: cs.NI; cs.SD**

- **简介: 该论文聚焦音乐元宇宙（MM）通信的服务供给问题，旨在解决其交互性、异构性和组播需求对网络带来的挑战。工作包括建立MM服务与网络图模型，设计专用离散事件网络模拟器MuMeNet，并通过线性规划编排策略验证其在虚拟音乐会场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.05201v2](https://arxiv.org/pdf/2512.05201v2)**

> **作者:** Ali Al Housseini; Jaime Llorca; Luca Turchet; Tiziano Leidi; Cristina Rottondi; Omran Ayoub
>
> **备注:** To appear in 2025 IEEE 6th International Symposium on the Internet of Sounds (IS2) proceedings
>
> **摘要:** The Metaverse, a shared and spatially organized digital continuum, is transforming various industries, with music emerging as a leading use case. Live concerts, collaborative composition, and interactive experiences are driving the Musical Metaverse (MM), but the requirements of the underlying network and service infrastructures hinder its growth. These challenges underscore the need for a novel modeling and simulation paradigm tailored to the unique characteristics of MM sessions, along with specialized service provisioning strategies capable of capturing their interactive, heterogeneous, and multicast-oriented nature. To this end, we make a first attempt to formally model and analyze the problem of service provisioning for MM sessions in 5G/6G networks. We first formalize service and network graph models for the MM, using "live audience interaction in a virtual concert" as a reference scenario. We then present MuMeNet, a novel discrete-event network simulator specifically tailored to the requirements and the traffic dynamics of the MM. We showcase the effectiveness of MuMeNet by running a linear programming based orchestration policy on the reference scenario and providing performance analysis under realistic MM workloads.
>
---
#### [replaced 008] Scaling to Multimodal and Multichannel Heart Sound Classification with Synthetic and Augmented Biosignals
- **分类: cs.SD; cs.LG; eess.SP**

- **简介: 该论文研究多模态与多通道心音分类任务，旨在解决同步与多通道数据稀缺导致模型性能受限的问题。作者结合信号处理与扩散模型生成合成数据，增强数据集后微调Wav2Vec 2.0模型，显著提升分类性能。**

- **链接: [https://arxiv.org/pdf/2509.11606v3](https://arxiv.org/pdf/2509.11606v3)**

> **作者:** Milan Marocchi; Matthew Fynn; Kayapanda Mandana; Yue Rong
>
> **备注:** 35 pages, 37 figures, 19 tables
>
> **摘要:** Cardiovascular diseases (CVDs) are the leading cause of death worldwide, accounting for approximately 17.9 million deaths each year. Early detection is critical, creating a demand for accurate and inexpensive pre-screening methods. Deep learning has recently been applied to classify abnormal heart sounds indicative of CVDs using synchronised phonocardiogram (PCG) and electrocardiogram (ECG) signals, as well as multichannel PCG (mPCG). However, state-of-the-art architectures remain underutilised due to the limited availability of synchronised and multichannel datasets. Augmented datasets and pre-trained models provide a pathway to overcome these limitations, enabling transformer-based architectures to be trained effectively. This work combines traditional signal processing with denoising diffusion models, WaveGrad and DiffWave, to create an augmented dataset to fine-tune a Wav2Vec 2.0-based classifier on multimodal and multichannel heart sound datasets. The approach achieves state-of-the-art performance. On the Computing in Cardiology (CinC) 2016 dataset of single channel PCG, accuracy, unweighted average recall (UAR), sensitivity, specificity and Matthew's correlation coefficient (MCC) reach 92.48%, 93.05%, 93.63%, 92.48%, 94.93% and 0.8283, respectively. Using the synchronised PCG and ECG signals of the training-a dataset from CinC, 93.14%, 92.21%, 94.35%, 90.10%, 95.12% and 0.8380 are achieved for accuracy, UAR, sensitivity, specificity and MCC, respectively. Using a wearable vest dataset consisting of mPCG data, the model achieves 77.13% accuracy, 74.25% UAR, 86.47% sensitivity, 62.04% specificity, and 0.5082 MCC. These results demonstrate the effectiveness of transformer-based models for CVD detection when supported by augmented datasets, highlighting their potential to advance multimodal and multichannel heart sound classification.
>
---
