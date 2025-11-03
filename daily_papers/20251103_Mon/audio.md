# 音频 cs.SD;  eess.AS

- **最新发布 11 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Beamforming in the Reproducing Kernel Domain Based on Spatial Differentiation
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出一种基于再生核域的空间微分的波束成形新框架，将方向响应建模为声场的空间微分，实现任意波束图（包括非轴对称）的构造。通过霍布森定理支持的再生核表达，统一并推广了传统球谐域波束成形，揭示其微分算子本质。二维仿真验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.27143v1](http://arxiv.org/pdf/2510.27143v1)**

> **作者:** Takahiro Iwami; Naohisa Inoue; Akira Omoto
>
> **摘要:** This paper proposes a novel beamforming framework in the reproducing kernel domain, derived from a unified interpretation of directional response as spatial differentiation of the sound field. By representing directional response using polynomial differential operators, the proposed method enables the formulation of arbitrary beam patterns including non-axisymmetric. The derivation of the reproducing kernel associated with the interior fields is mathematically supported by Hobson's theorem, which allows concise analytical expressions. Furthermore, the proposed framework generalizes conventional spherical harmonic domain beamformers by reinterpreting them as spatial differential operators, thereby clarifying their theoretical structure and extensibility. Three numerical simulations conducted in two-dimensional space confirm the validity of the method.
>
---
#### [new 002] Cross-Corpus Validation of Speech Emotion Recognition in Urdu using Domain-Knowledge Acoustic Features
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究乌尔都语语音情感识别（SER）任务，针对低资源语言在跨语料库设置下的模型泛化问题。通过三个乌尔都语情感语料库的交叉验证，采用eGeMAPS和ComParE声学特征与分类器结合，发现自语料库评估易高估性能，强调跨语料库验证对真实评估模型鲁棒性的重要性。**

- **链接: [http://arxiv.org/pdf/2510.26823v1](http://arxiv.org/pdf/2510.26823v1)**

> **作者:** Unzela Talpur; Zafi Sherhan Syed; Muhammad Shehram Shah Syed; Abbas Shah Syed
>
> **备注:** Conference paper, 4 pages, including 3 figures and 3 tables
>
> **摘要:** Speech Emotion Recognition (SER) is a key affective computing technology that enables emotionally intelligent artificial intelligence. While SER is challenging in general, it is particularly difficult for low-resource languages such as Urdu. This study investigates Urdu SER in a cross-corpus setting, an area that has remained largely unexplored. We employ a cross-corpus evaluation framework across three different Urdu emotional speech datasets to test model generalization. Two standard domain-knowledge based acoustic feature sets, eGeMAPS and ComParE, are used to represent speech signals as feature vectors which are then passed to Logistic Regression and Multilayer Perceptron classifiers. Classification performance is assessed using unweighted average recall (UAR) whilst considering class-label imbalance. Results show that Self-corpus validation often overestimates performance, with UAR exceeding cross-corpus evaluation by up to 13%, underscoring that cross-corpus evaluation offers a more realistic measure of model robustness. Overall, this work emphasizes the importance of cross-corpus validation for Urdu SER and its implications contribute to advancing affective computing research for underrepresented language communities.
>
---
#### [new 003] Audio-Visual Speech Enhancement In Complex Scenarios With Separation And Dereverberation Joint Modeling
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文针对复杂场景下的音视频语音增强任务，解决混响与干扰噪声导致的语音质量差问题。提出“先分离后去混响”的联合建模框架，有效提升语音清晰度与可懂度，在第四届音视频语音增强挑战赛中三项客观指标领先，并获主观听感第一名。**

- **链接: [http://arxiv.org/pdf/2510.26825v1](http://arxiv.org/pdf/2510.26825v1)**

> **作者:** Jiarong Du; Zhan Jin; Peijun Yang; Juan Liu; Zhuo Li; Xin Liu; Ming Li
>
> **摘要:** Audio-visual speech enhancement (AVSE) is a task that uses visual auxiliary information to extract a target speaker's speech from mixed audio. In real-world scenarios, there often exist complex acoustic environments, accompanied by various interfering sounds and reverberation. Most previous methods struggle to cope with such complex conditions, resulting in poor perceptual quality of the extracted speech. In this paper, we propose an effective AVSE system that performs well in complex acoustic environments. Specifically, we design a "separation before dereverberation" pipeline that can be extended to other AVSE networks. The 4th COGMHEAR Audio-Visual Speech Enhancement Challenge (AVSEC) aims to explore new approaches to speech processing in multimodal complex environments. We validated the performance of our system in AVSEC-4: we achieved excellent results in the three objective metrics on the competition leaderboard, and ultimately secured first place in the human subjective listening test.
>
---
#### [new 004] GACA-DiT: Diffusion-based Dance-to-Music Generation with Genre-Adaptive Rhythm and Context-Aware Alignment
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文聚焦舞蹈到音乐生成任务，旨在实现节奏与时间上精准对齐的音乐合成。针对现有方法因粗粒度节奏嵌入和下采样导致的同步偏差问题，提出GACA-DiT框架，通过风格自适应节奏提取与上下文感知时间对齐模块，捕捉细粒度节奏特征并修复时序错位，显著提升生成音乐的质量与同步性。**

- **链接: [http://arxiv.org/pdf/2510.26818v1](http://arxiv.org/pdf/2510.26818v1)**

> **作者:** Jinting Wang; Chenxing Li; Li Liu
>
> **备注:** 5 pages, 3 figures, submitted to ICASSP 2026
>
> **摘要:** Dance-to-music (D2M) generation aims to automatically compose music that is rhythmically and temporally aligned with dance movements. Existing methods typically rely on coarse rhythm embeddings, such as global motion features or binarized joint-based rhythm values, which discard fine-grained motion cues and result in weak rhythmic alignment. Moreover, temporal mismatches introduced by feature downsampling further hinder precise synchronization between dance and music. To address these problems, we propose \textbf{GACA-DiT}, a diffusion transformer-based framework with two novel modules for rhythmically consistent and temporally aligned music generation. First, a \textbf{genre-adaptive rhythm extraction} module combines multi-scale temporal wavelet analysis and spatial phase histograms with adaptive joint weighting to capture fine-grained, genre-specific rhythm patterns. Second, a \textbf{context-aware temporal alignment} module resolves temporal mismatches using learnable context queries to align music latents with relevant dance rhythm features. Extensive experiments on the AIST++ and TikTok datasets demonstrate that GACA-DiT outperforms state-of-the-art methods in both objective metrics and human evaluation. Project page: https://beria-moon.github.io/GACA-DiT/.
>
---
#### [new 005] Oral Tradition-Encoded NanyinHGNN: Integrating Nanyin Music Preservation and Generation through a Pipa-Centric Dataset
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，针对非遗南音因口传心授导致的乐谱数据缺失问题。构建以琵琶为中心的MIDI数据集，提出NanyinTok分词与图转换方法，设计异构图神经网络模型，将装饰音生成建模为图节点生成，结合表演规则实现无标注训练，成功生成四乐器合奏的仿真南音作品。**

- **链接: [http://arxiv.org/pdf/2510.26817v1](http://arxiv.org/pdf/2510.26817v1)**

> **作者:** Jianbing Xiahou; Weixi Zhai; Xu Cui
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** We propose NanyinHGNN, a heterogeneous graph network model for generating Nanyin instrumental music. As a UNESCO-recognized intangible cultural heritage, Nanyin follows a heterophonic tradition centered around the pipa, where core melodies are notated in traditional notation while ornamentations are passed down orally, presenting challenges for both preservation and contemporary innovation. To address this, we construct a Pipa-Centric MIDI dataset, develop NanyinTok as a specialized tokenization method, and convert symbolic sequences into graph structures using a Graph Converter to ensure that key musical features are preserved. Our key innovation reformulates ornamentation generation as the creation of ornamentation nodes within a heterogeneous graph. First, a graph neural network generates melodic outlines optimized for ornamentations. Then, a rule-guided system informed by Nanyin performance practices refines these outlines into complete ornamentations without requiring explicit ornamentation annotations during training. Experimental results demonstrate that our model successfully generates authentic heterophonic ensembles featuring four traditional instruments. These findings validate that integrating domain-specific knowledge into model architecture can effectively mitigate data scarcity challenges in computational ethnomusicology.
>
---
#### [new 006] Expressive Range Characterization of Open Text-to-Audio Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究文本到音频生成模型的表达范围，旨在量化其输出的多样性与保真度。针对生成音频的宽泛性，提出基于标准提示的表达范围分析（ERA）框架，通过声学特征分析评估模型性能，为生成音频模型提供可量化的探索性评估方法。**

- **链接: [http://arxiv.org/pdf/2510.27102v1](http://arxiv.org/pdf/2510.27102v1)**

> **作者:** Jonathan Morse; Azadeh Naderi; Swen Gaudl; Mark Cartwright; Amy K. Hoover; Mark J. Nelson
>
> **备注:** Accepted at the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment (AIIDE 2025)
>
> **摘要:** Text-to-audio models are a type of generative model that produces audio output in response to a given textual prompt. Although level generators and the properties of the functional content that they create (e.g., playability) dominate most discourse in procedurally generated content (PCG), games that emotionally resonate with players tend to weave together a range of creative and multimodal content (e.g., music, sounds, visuals, narrative tone), and multimodal models have begun seeing at least experimental use for this purpose. However, it remains unclear what exactly such models generate, and with what degree of variability and fidelity: audio is an extremely broad class of output for a generative system to target. Within the PCG community, expressive range analysis (ERA) has been used as a quantitative way to characterize generators' output space, especially for level generators. This paper adapts ERA to text-to-audio models, making the analysis tractable by looking at the expressive range of outputs for specific, fixed prompts. Experiments are conducted by prompting the models with several standardized prompts derived from the Environmental Sound Classification (ESC-50) dataset. The resulting audio is analyzed along key acoustic dimensions (e.g., pitch, loudness, and timbre). More broadly, this paper offers a framework for ERA-based exploratory evaluation of generative audio models.
>
---
#### [new 007] See the Speaker: Crafting High-Resolution Talking Faces from Speech with Prior Guidance and Region Refinement
- **分类: eess.AS; cs.AI; cs.CV; cs.SD**

- **简介: 该论文聚焦语音驱动高分辨率说话人脸生成任务，解决仅依赖语音输入时外观与动作同步难题。提出两阶段框架：先用语音条件扩散模型结合面部先验生成人脸，再通过区域增强模块优化唇音同步与表情动态，最终融合离散码本与渲染网络实现端到端高清视频生成。**

- **链接: [http://arxiv.org/pdf/2510.26819v1](http://arxiv.org/pdf/2510.26819v1)**

> **作者:** Jinting Wang; Jun Wang; Hei Victor Cheng; Li Liu
>
> **备注:** 16 pages,15 figures, accepted by TASLP
>
> **摘要:** Unlike existing methods that rely on source images as appearance references and use source speech to generate motion, this work proposes a novel approach that directly extracts information from the speech, addressing key challenges in speech-to-talking face. Specifically, we first employ a speech-to-face portrait generation stage, utilizing a speech-conditioned diffusion model combined with statistical facial prior and a sample-adaptive weighting module to achieve high-quality portrait generation. In the subsequent speech-driven talking face generation stage, we embed expressive dynamics such as lip movement, facial expressions, and eye movements into the latent space of the diffusion model and further optimize lip synchronization using a region-enhancement module. To generate high-resolution outputs, we integrate a pre-trained Transformer-based discrete codebook with an image rendering network, enhancing video frame details in an end-to-end manner. Experimental results demonstrate that our method outperforms existing approaches on the HDTF, VoxCeleb, and AVSpeech datasets. Notably, this is the first method capable of generating high-resolution, high-quality talking face videos exclusively from a single speech input.
>
---
#### [new 008] Multi-Representation Attention Framework for Underwater Bioacoustic Denoising and Recognition
- **分类: eess.AS; cs.LG; cs.SD; stat.AP; stat.ML**

- **简介: 该论文针对海洋哺乳动物声学信号的去噪与识别任务，解决低频到超声波信号重叠、噪声复杂等问题。提出多表示注意力框架，通过谱图分割生成软掩码，融合掩码与原始输入进行多带去噪分类，实现高精度识别与强泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.26838v1](http://arxiv.org/pdf/2510.26838v1)**

> **作者:** Amine Razig; Youssef Soulaymani; Loubna Benabbou; Pierre Cauchy
>
> **摘要:** Automated monitoring of marine mammals in the St. Lawrence Estuary faces extreme challenges: calls span low-frequency moans to ultrasonic clicks, often overlap, and are embedded in variable anthropogenic and environmental noise. We introduce a multi-step, attention-guided framework that first segments spectrograms to generate soft masks of biologically relevant energy and then fuses these masks with the raw inputs for multi-band, denoised classification. Image and mask embeddings are integrated via mid-level fusion, enabling the model to focus on salient spectrogram regions while preserving global context. Using real-world recordings from the Saguenay St. Lawrence Marine Park Research Station in Canada, we demonstrate that segmentation-driven attention and mid-level fusion improve signal discrimination, reduce false positive detections, and produce reliable representations for operational marine mammal monitoring across diverse environmental conditions and signal-to-noise ratios. Beyond in-distribution evaluation, we further assess the generalization of Mask-Guided Classification (MGC) under distributional shifts by testing on spectrograms generated with alternative acoustic transformations. While high-capacity baseline models lose accuracy in this Out-of-distribution (OOD) setting, MGC maintains stable performance, with even simple fusion mechanisms (gated, concat) achieving comparable results across distributions. This robustness highlights the capacity of MGC to learn transferable representations rather than overfitting to a specific transformation, thereby reinforcing its suitability for large-scale, real-world biodiversity monitoring. We show that in all experimental settings, the MGC framework consistently outperforms baseline architectures, yielding substantial gains in accuracy on both in-distribution and OOD data.
>
---
#### [new 009] Representing Classical Compositions through Implication-Realization Temporal-Gestalt Graphs
- **分类: cs.SD; cs.LG; cs.SI; H.5.5; G.2.2; I.5.4**

- **简介: 该论文提出基于蕴含-实现（I-R）与时间格式塔理论的图模型，将古典音乐片段建模为带认知标签的图结构，通过图核与嵌入分析音乐结构与感知相似性。旨在解决音乐结构与听觉体验的计算表征问题，融合认知模型与图神经网络，实现对风格与结构的深层分析。**

- **链接: [http://arxiv.org/pdf/2510.27530v1](http://arxiv.org/pdf/2510.27530v1)**

> **作者:** A. V. Bomediano; R. J. Conanan; L. D. Santuyo; A. Coronel
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** Understanding the structural and cognitive underpinnings of musical compositions remains a key challenge in music theory and computational musicology. While traditional methods focus on harmony and rhythm, cognitive models such as the Implication-Realization (I-R) model and Temporal Gestalt theory offer insight into how listeners perceive and anticipate musical structure. This study presents a graph-based computational approach that operationalizes these models by segmenting melodies into perceptual units and annotating them with I-R patterns. These segments are compared using Dynamic Time Warping and organized into k-nearest neighbors graphs to model intra- and inter-segment relationships. Each segment is represented as a node in the graph, and nodes are further labeled with melodic expectancy values derived from Schellenberg's two-factor I-R model-quantifying pitch proximity and pitch reversal at the segment level. This labeling enables the graphs to encode both structural and cognitive information, reflecting how listeners experience musical tension and resolution. To evaluate the expressiveness of these graphs, we apply the Weisfeiler-Lehman graph kernel to measure similarity between and within compositions. Results reveal statistically significant distinctions between intra- and inter-graph structures. Segment-level analysis via multidimensional scaling confirms that structural similarity at the graph level reflects perceptual similarity at the segment level. Graph2vec embeddings and clustering demonstrate that these representations capture stylistic and structural features that extend beyond composer identity. These findings highlight the potential of graph-based methods as a structured, cognitively informed framework for computational music analysis, enabling a more nuanced understanding of musical structure and style through the lens of listener perception.
>
---
#### [new 010] Reference Microphone Selection for Guided Source Separation based on the Normalized L-p Norm
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对远场语音识别中的引导源分离（GSS）任务，解决参考麦克风选择问题。传统基于信噪比（SNR）的方法忽略混响差异。本文提出基于归一化ℓ_p-范数的选参方法，兼顾SNR与早期/晚期混响比（ELR），实验表明其能有效降低词错误率，提升系统性能。**

- **链接: [http://arxiv.org/pdf/2510.27198v1](http://arxiv.org/pdf/2510.27198v1)**

> **作者:** Anselm Lohmann; Tomohiro Nakatani; Rintaro Ikeshita; Marc Delcroix; Shoko Araki; Simon Doclo
>
> **摘要:** Guided Source Separation (GSS) is a popular front-end for distant automatic speech recognition (ASR) systems using spatially distributed microphones. When considering spatially distributed microphones, the choice of reference microphone may have a large influence on the quality of the output signal and the downstream ASR performance. In GSS-based speech enhancement, reference microphone selection is typically performed using the signal-to-noise ratio (SNR), which is optimal for noise reduction but may neglect differences in early-to-late-reverberant ratio (ELR) across microphones. In this paper, we propose two reference microphone selection methods for GSS-based speech enhancement that are based on the normalized $\ell_p$-norm, either using only the normalized $\ell_p$-norm or combining the normalized $\ell_p$-norm and the SNR to account for both differences in SNR and ELR across microphones. Experimental evaluation using a CHiME-8 distant ASR system shows that the proposed $\ell_p$-norm-based methods outperform the baseline method, reducing the macro-average word error rate.
>
---
#### [new 011] Inferring trust in recommendation systems from brain, behavioural, and physiological data
- **分类: cs.HC; eess.AS; eess.SP**

- **简介: 该论文研究推荐系统中用户对AI的信任机制，旨在解决传统自评方法主观且干扰使用的问题。通过音乐推荐任务，结合脑电（EEG）与瞳孔变化等多模态数据，发现系统准确率影响信任及偏好，且与神经活动和奖励预测误差相关，为可信AI提供神经基础。**

- **链接: [http://arxiv.org/pdf/2510.27272v1](http://arxiv.org/pdf/2510.27272v1)**

> **作者:** Vincent K. M. Cheung; Pei-Cheng Shih; Masato Hirano; Masataka Goto; Shinichi Furuya
>
> **摘要:** As people nowadays increasingly rely on artificial intelligence (AI) to curate information and make decisions, assigning the appropriate amount of trust in automated intelligent systems has become ever more important. However, current measurements of trust in automation still largely rely on self-reports that are subjective and disruptive to the user. Here, we take music recommendation as a model to investigate the neural and cognitive processes underlying trust in automation. We observed that system accuracy was directly related to users' trust and modulated the influence of recommendation cues on music preference. Modelling users' reward encoding process with a reinforcement learning model further revealed that system accuracy, expected reward, and prediction error were related to oscillatory neural activity recorded via EEG and changes in pupil diameter. Our results provide a neurally grounded account of calibrating trust in automation and highlight the promises of a multimodal approach towards developing trustable AI systems.
>
---
## 更新

#### [replaced 001] UTI-LLM: A Personalized Articulatory-Speech Therapy Assistance System Based on Multimodal Large Language Model
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.13145v2](http://arxiv.org/pdf/2509.13145v2)**

> **作者:** Yudong Yang; Xiaokang Liu; Shaofeng zhao; Rongfeng Su; Nan Yan; Lan Wang
>
> **摘要:** Speech therapy is essential for rehabilitating speech disorders caused by neurological impairments such as stroke. However, traditional manual and computer-assisted systems are limited in real-time accessibility and articulatory motion feedback. Recent advances in multimodal large language models (MLLMs) have demonstrated significant potential in healthcare, especially through their adaptive assessment and therapeutic feedback capabilities. Nevertheless, challenges including insufficient acquisition and fusion of articulatory information, inadequate parsing of articulatory organ motion trajectories, and the scarcity of domain-specific datasets hinder the application of MLLMs in speech therapy. To address these limitations, we propose an MLLM-based speech rehabilitation assistance system that leverages ultrasound tongue imaging and speech signals to deliver precise, interactive articulatory feedback. We construct a high-quality domain-specific dataset comprising ultrasound-speech dialogue pairs. This dataset facilitates fine-tuning to enhance the model's clinical adaptability. Furthermore, our method develops spatiotemporal fusion training strategy of ultrasound videos and speech signals, enabling fine-grained articulatory impairment analysis and ultimately generating actionable feedback. Experimental results demonstrate the effectiveness of our model in articulatory analysis and clinical assessment.
>
---
#### [replaced 002] 'Studies for': A Human-AI Co-Creative Sound Artwork Using a Real-time Multi-channel Sound Generation Model
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.25228v2](http://arxiv.org/pdf/2510.25228v2)**

> **作者:** Chihiro Nagashima; Akira Takahashi; Zhi Zhong; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted at NeurIPS Creative AI Track 2025, 9 pages, 6 figures, 1 table, Demo page: https://sony.github.io/studies-for/
>
> **摘要:** This paper explores the integration of AI technologies into the artistic workflow through the creation of Studies for, a generative sound installation developed in collaboration with sound artist Evala (https://www.ntticc.or.jp/en/archive/works/studies-for/). The installation employs SpecMaskGIT, a lightweight yet high-quality sound generation AI model, to generate and playback eight-channel sound in real-time, creating an immersive auditory experience over the course of a three-month exhibition. The work is grounded in the concept of a "new form of archive," which aims to preserve the artistic style of an artist while expanding beyond artists' past artworks by continued generation of new sound elements. This speculative approach to archival preservation is facilitated by training the AI model on a dataset consisting of over 200 hours of Evala's past sound artworks. By addressing key requirements in the co-creation of art using AI, this study highlights the value of the following aspects: (1) the necessity of integrating artist feedback, (2) datasets derived from an artist's past works, and (3) ensuring the inclusion of unexpected, novel outputs. In Studies for, the model was designed to reflect the artist's artistic identity while generating new, previously unheard sounds, making it a fitting realization of the concept of "a new form of archive." We propose a Human-AI co-creation framework for effectively incorporating sound generation AI models into the sound art creation process and suggest new possibilities for creating and archiving sound art that extend an artist's work beyond their physical existence. Demo page: https://sony.github.io/studies-for/
>
---
