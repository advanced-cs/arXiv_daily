# 音频 cs.SD;  eess.SP

- **最新发布 39 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] The Multimodal Information Based Speech Processing (MISP) 2025 Challenge: Audio-Visual Diarization and Recognition
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文聚焦多模态语音处理任务，针对复杂会议场景下的说话人分离与识别难题，通过组织MISP 2025挑战赛，提出视听说话人分离（AVSD）、语音识别（AVSR）及联合任务（AVDR），提供数据集与基线系统，评估多模态方法性能，结果显示最佳模型在误差率指标上显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.13971v1](http://arxiv.org/pdf/2505.13971v1)**

> **作者:** Ming Gao; Shilong Wu; Hang Chen; Jun Du; Chin-Hui Lee; Shinji Watanabe; Jingdong Chen; Siniscalchi Sabato Marco; Odette Scharenborg
>
> **备注:** Accepted by Interspeech 2025. Camera-ready version
>
> **摘要:** Meetings are a valuable yet challenging scenario for speech applications due to complex acoustic conditions. This paper summarizes the outcomes of the MISP 2025 Challenge, hosted at Interspeech 2025, which focuses on multi-modal, multi-device meeting transcription by incorporating video modality alongside audio. The tasks include Audio-Visual Speaker Diarization (AVSD), Audio-Visual Speech Recognition (AVSR), and Audio-Visual Diarization and Recognition (AVDR). We present the challenge's objectives, tasks, dataset, baseline systems, and solutions proposed by participants. The best-performing systems achieved significant improvements over the baseline: the top AVSD model achieved a Diarization Error Rate (DER) of 8.09%, improving by 7.43%; the top AVSR system achieved a Character Error Rate (CER) of 9.48%, improving by 10.62%; and the best AVDR system achieved a concatenated minimum-permutation Character Error Rate (cpCER) of 11.56%, improving by 72.49%.
>
---
#### [new 002] ClapFM-EVC: High-Fidelity and Flexible Emotional Voice Conversion with Dual Control from Natural Language and Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于情绪语音转换（EVC）任务，旨在解决高保真度与灵活可控情绪转换的难题。提出ClapFM-EVC框架，通过EVC-CLAP模型对齐语音与文本情绪特征，利用FuEncoder融合情感强度与音素特征，并采用流匹配模型重建高质量语音，实现自然语言或参考语音驱动的可调节情绪转换。**

- **链接: [http://arxiv.org/pdf/2505.13805v1](http://arxiv.org/pdf/2505.13805v1)**

> **作者:** Yu Pan; Yanni Hu; Yuguang Yang; Jixun Yao; Jianhao Ye; Hongbin Zhou; Lei Ma; Jianjun Zhao
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** Despite great advances, achieving high-fidelity emotional voice conversion (EVC) with flexible and interpretable control remains challenging. This paper introduces ClapFM-EVC, a novel EVC framework capable of generating high-quality converted speech driven by natural language prompts or reference speech with adjustable emotion intensity. We first propose EVC-CLAP, an emotional contrastive language-audio pre-training model, guided by natural language prompts and categorical labels, to extract and align fine-grained emotional elements across speech and text modalities. Then, a FuEncoder with an adaptive intensity gate is presented to seamless fuse emotional features with Phonetic PosteriorGrams from a pre-trained ASR model. To further improve emotion expressiveness and speech naturalness, we propose a flow matching model conditioned on these captured features to reconstruct Mel-spectrogram of source speech. Subjective and objective evaluations validate the effectiveness of ClapFM-EVC.
>
---
#### [new 003] Complexity of frequency fluctuations and the interpretive style in the bass viola da gamba
- **分类: cs.SD; eess.AS; stat.AP**

- **简介: 该论文通过复杂网络模型分析中提维奥拉演奏音频信号，研究频率波动复杂性与演奏风格的关系。任务为量化音乐事件与演奏者风格间的关联，解决如何通过网络分析揭示频率波动规律的问题。工作包括音频频谱分解、统计分布拟合、中心性计算及最大团识别，以发现稳定声音组和互动功能组，关联统计规律与演奏特征。**

- **链接: [http://arxiv.org/pdf/2505.14448v1](http://arxiv.org/pdf/2505.14448v1)**

> **作者:** Igor Lugo; Martha G. Alatriste-Contreras; Rafael Sánchez-Guevara
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Audio signals in a set of musical pieces are modeled as a complex network for studying the relationship between the complexity of frequency fluctuations and the interpretive style of the bass viola da gamba. Based on interdisciplinary scientific and music approaches, we compute the spectral decomposition and translated its frequency components to a network of sounds. We applied a best fit analysis for identifying the statistical distributions that describe more precisely the behavior of such frequencies and computed the centrality measures and identify cliques for characterizing such a network. Findings suggested statistical regularities in the type of statistical distribution that best describes frequency fluctuations. The centrality measure confirmed the most influential and stable group of sounds in a piece of music, meanwhile the identification of the largest clique indicated functional groups of sounds that interact closely for identifying the emergence of complex frequency fluctuations. Therefore, by modeling the sound as a complex network, we can clearly associate the presence of large-scale statistical regularities with the presence of similar frequency fluctuations related to different musical events played by a same musician.
>
---
#### [new 004] Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Vox-Profile基准，用于综合评估语音基础模型在分析说话人静态特征（如年龄、性别）和动态属性（如情感、语流）的能力。针对现有方法仅关注单一维度的不足，其通过整合15个公开数据集和多模型实验，构建多维度评测体系，并展示其在提升语音识别分析、评估生成系统质量等应用。工具已开源。**

- **链接: [http://arxiv.org/pdf/2505.14648v1](http://arxiv.org/pdf/2505.14648v1)**

> **作者:** Tiantian Feng; Jihwan Lee; Anfeng Xu; Yoonjeong Lee; Thanathai Lertpetchpun; Xuan Shi; Helin Wang; Thomas Thebaud; Laureano Moro-Velazquez; Dani Byrd; Najim Dehak; Shrikanth Narayanan
>
> **摘要:** We introduce Vox-Profile, a comprehensive benchmark to characterize rich speaker and speech traits using speech foundation models. Unlike existing works that focus on a single dimension of speaker traits, Vox-Profile provides holistic and multi-dimensional profiles that reflect both static speaker traits (e.g., age, sex, accent) and dynamic speech properties (e.g., emotion, speech flow). This benchmark is grounded in speech science and linguistics, developed with domain experts to accurately index speaker and speech characteristics. We report benchmark experiments using over 15 publicly available speech datasets and several widely used speech foundation models that target various static and dynamic speaker and speech properties. In addition to benchmark experiments, we showcase several downstream applications supported by Vox-Profile. First, we show that Vox-Profile can augment existing speech recognition datasets to analyze ASR performance variability. Vox-Profile is also used as a tool to evaluate the performance of speech generation systems. Finally, we assess the quality of our automated profiles through comparison with human evaluation and show convergent validity. Vox-Profile is publicly available at: https://github.com/tiantiaf0627/vox-profile-release.
>
---
#### [new 005] PersonaTAB: Predicting Personality Traits using Textual, Acoustic, and Behavioral Cues in Fully-Duplex Speech Dialogs
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于对话系统中的人格预测任务，旨在解决语音数据缺乏人格标注导致的个性感知代理研究不足问题。团队通过预处理音频生成标注数据集，结合ASR提取文本并利用大模型预测对话人格，经人类评估验证效果更优。**

- **链接: [http://arxiv.org/pdf/2505.14356v1](http://arxiv.org/pdf/2505.14356v1)**

> **作者:** Sho Inoue; Shai Wang; Haizhou Li
>
> **备注:** This is accepted to Interspeech 2025; Added an extra page for supplementary figures; Project page: https://github.com/shinshoji01/Personality-Prediction-for-Conversation-Agents
>
> **摘要:** Despite significant progress in neural spoken dialog systems, personality-aware conversation agents -- capable of adapting behavior based on personalities -- remain underexplored due to the absence of personality annotations in speech datasets. We propose a pipeline that preprocesses raw audio recordings to create a dialogue dataset annotated with timestamps, response types, and emotion/sentiment labels. We employ an automatic speech recognition (ASR) system to extract transcripts and timestamps, then generate conversation-level annotations. Leveraging these annotations, we design a system that employs large language models to predict conversational personality. Human evaluators were engaged to identify conversational characteristics and assign personality labels. Our analysis demonstrates that the proposed system achieves stronger alignment with human judgments compared to existing approaches.
>
---
#### [new 006] S2SBench: A Benchmark for Quantifying Intelligence Degradation in Speech-to-Speech Large Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出S2SBench基准，用于量化语音到语音LLMs的智能退化问题。任务是评估音频输入导致的推理与生成性能下降。通过构建诊断数据集（句子续写、常识推理）及基于困惑度差异的配对评估协议，系统衡量语音模型相较文本输入的退化，并验证于Baichuan-Audio训练分析。**

- **链接: [http://arxiv.org/pdf/2505.14438v1](http://arxiv.org/pdf/2505.14438v1)**

> **作者:** Yuanbo Fang; Haoze Sun; Jun Liu; Tao Zhang; Zenan Zhou; Weipeng Chen; Xiaofen Xing; Xiangmin Xu
>
> **摘要:** End-to-end speech large language models ((LLMs)) extend the capabilities of text-based models to directly process and generate audio tokens. However, this often leads to a decline in reasoning and generation performance compared to text input, a phenomenon referred to as intelligence degradation. To systematically evaluate this gap, we propose S2SBench, a benchmark designed to quantify performance degradation in Speech LLMs. It includes diagnostic datasets targeting sentence continuation and commonsense reasoning under audio input. We further introduce a pairwise evaluation protocol based on perplexity differences between plausible and implausible samples to measure degradation relative to text input. We apply S2SBench to analyze the training process of Baichuan-Audio, which further demonstrates the benchmark's effectiveness. All datasets and evaluation code are available at https://github.com/undobug/S2SBench.
>
---
#### [new 007] Score-Based Training for Energy-Based TTS Models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于能量基语音合成（TTS）模型训练任务，旨在解决现有方法（NCE和SSM）忽视对数似然函数形式导致一阶优化推理效果不佳的问题。提出新评分学习准则，优化能量模型的梯度学习，实验对比了不同训练方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.13771v1](http://arxiv.org/pdf/2505.13771v1)**

> **作者:** Wanli Sun; Anton Ragni
>
> **摘要:** Noise contrastive estimation (NCE) is a popular method for training energy-based models (EBM) with intractable normalisation terms. The key idea of NCE is to learn by comparing unnormalised log-likelihoods of the reference and noisy samples, thus avoiding explicitly computing normalisation terms. However, NCE critically relies on the quality of noisy samples. Recently, sliced score matching (SSM) has been popularised by closely related diffusion models (DM). Unlike NCE, SSM learns a gradient of log-likelihood, or score, by learning distribution of its projections on randomly chosen directions. However, both NCE and SSM disregard the form of log-likelihood function, which is problematic given that EBMs and DMs make use of first-order optimisation during inference. This paper proposes a new criterion that learns scores more suitable for first-order schemes. Experiments contrasts these approaches for training EBMs.
>
---
#### [new 008] MatchDance: Collaborative Mamba-Transformer Architecture Matching for High-Quality 3D Dance Synthesis
- **分类: cs.SD; cs.GR; cs.MM; eess.AS**

- **简介: 该论文属于音乐到舞蹈生成任务，旨在解决现有方法编舞一致性不足的问题。提出MatchDance框架：1）通过Kinematic-Dynamic量化阶段（KDQS）将舞蹈动作编码为受物理约束的潜在表示；2）采用Mamba-Transformer混合架构将音乐映射到该潜在空间，解码生成高质量3D舞蹈，实验显示达到新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.14222v1](http://arxiv.org/pdf/2505.14222v1)**

> **作者:** Kaixing Yang; Xulong Tang; Yuxuan Hu; Jiahao Yang; Hongyan Liu; Qinnan Zhang; Jun He; Zhaoxin Fan
>
> **摘要:** Music-to-dance generation represents a challenging yet pivotal task at the intersection of choreography, virtual reality, and creative content generation. Despite its significance, existing methods face substantial limitation in achieving choreographic consistency. To address the challenge, we propose MatchDance, a novel framework for music-to-dance generation that constructs a latent representation to enhance choreographic consistency. MatchDance employs a two-stage design: (1) a Kinematic-Dynamic-based Quantization Stage (KDQS), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) with kinematic-dynamic constraints and reconstructs them with high fidelity, and (2) a Hybrid Music-to-Dance Generation Stage(HMDGS), which uses a Mamba-Transformer hybrid architecture to map music into the latent representation, followed by the KDQS decoder to generate 3D dance motions. Additionally, a music-dance retrieval framework and comprehensive metrics are introduced for evaluation. Extensive experiments on the FineDance dataset demonstrate state-of-the-art performance. Code will be released upon acceptance.
>
---
#### [new 009] Source Verification for Speech Deepfakes
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音深伪源验证任务，解决开放环境下追溯未知生成模型的难题。提出基于源归因嵌入的距离评分方法，评估多模型在不同场景的表现，分析影响因素，首次探索该任务的潜力与漏洞。**

- **链接: [http://arxiv.org/pdf/2505.14188v1](http://arxiv.org/pdf/2505.14188v1)**

> **作者:** Viola Negroni; Davide Salvi; Paolo Bestagini; Stefano Tubaro
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** With the proliferation of speech deepfake generators, it becomes crucial not only to assess the authenticity of synthetic audio but also to trace its origin. While source attribution models attempt to address this challenge, they often struggle in open-set conditions against unseen generators. In this paper, we introduce the source verification task, which, inspired by speaker verification, determines whether a test track was produced using the same model as a set of reference signals. Our approach leverages embeddings from a classifier trained for source attribution, computing distance scores between tracks to assess whether they originate from the same source. We evaluate multiple models across diverse scenarios, analyzing the impact of speaker diversity, language mismatch, and post-processing operations. This work provides the first exploration of source verification, highlighting its potential and vulnerabilities, and offers insights for real-world forensic applications.
>
---
#### [new 010] FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出FMSD-TTS框架，解决藏语三大方言（Ü-Tsang、Amdo、Kham）低资源下的多说话人多方言语音合成问题。通过设计说话人-方言融合模块和DSDR-Net，捕捉方言差异同时保持说话人特征，公开合成语料库及评估工具。**

- **链接: [http://arxiv.org/pdf/2505.14351v1](http://arxiv.org/pdf/2505.14351v1)**

> **作者:** Yutong Liu; Ziyue Zhang; Ban Ma-bao; Yuqing Cai; Yongbin Yu; Renzeng Duojie; Xiangxiang Wang; Fan Gao; Cheng Huang; Nyima Tashi
>
> **备注:** 13 pages
>
> **摘要:** Tibetan is a low-resource language with minimal parallel speech corpora spanning its three major dialects-\"U-Tsang, Amdo, and Kham-limiting progress in speech modeling. To address this issue, we propose FMSD-TTS, a few-shot, multi-speaker, multi-dialect text-to-speech framework that synthesizes parallel dialectal speech from limited reference audio and explicit dialect labels. Our method features a novel speaker-dialect fusion module and a Dialect-Specialized Dynamic Routing Network (DSDR-Net) to capture fine-grained acoustic and linguistic variations across dialects while preserving speaker identity. Extensive objective and subjective evaluations demonstrate that FMSD-TTS significantly outperforms baselines in both dialectal expressiveness and speaker similarity. We further validate the quality and utility of the synthesized speech through a challenging speech-to-speech dialect conversion task. Our contributions include: (1) a novel few-shot TTS system tailored for Tibetan multi-dialect speech synthesis, (2) the public release of a large-scale synthetic Tibetan speech corpus generated by FMSD-TTS, and (3) an open-source evaluation toolkit for standardized assessment of speaker similarity, dialect consistency, and audio quality.
>
---
#### [new 011] Combining Deterministic Enhanced Conditions with Dual-Streaming Encoding for Diffusion-Based Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于扩散模型驱动的语音增强任务，旨在解决如何有效利用确定性方法生成的可靠条件来提升扩散过程的性能。研究对比了仅用确定性特征或结合噪声特征的两种条件方式，提出双流编码Repair-Diffusion模型（DERDM-SE），结合粗细粒度的确定性模型，平衡性能与稳定性，在CHiME4数据集上取得更好效果。**

- **链接: [http://arxiv.org/pdf/2505.13983v1](http://arxiv.org/pdf/2505.13983v1)**

> **作者:** Hao Shi; Xugang Lu; Kazuki Shimada; Tatsuya Kawahara
>
> **摘要:** Diffusion-based speech enhancement (SE) models need to incorporate correct prior knowledge as reliable conditions to generate accurate predictions. However, providing reliable conditions using noisy features is challenging. One solution is to use features enhanced by deterministic methods as conditions. However, the information distortion and loss caused by deterministic methods might affect the diffusion process. In this paper, we first investigate the effects of using different deterministic SE models as conditions for diffusion. We validate two conditions depending on whether the noisy feature was used as part of the condition: one using only the deterministic feature (deterministic-only), and the other using both deterministic and noisy features (deterministic-noisy). Preliminary investigation found that using deterministic enhanced conditions improves hearing experiences on real data, while the choice between using deterministic-only or deterministic-noisy conditions depends on the deterministic models. Based on these findings, we propose a dual-streaming encoding Repair-Diffusion Model for SE (DERDM-SE) to more effectively utilize both conditions. Moreover, we found that fine-grained deterministic models have greater potential in objective evaluation metrics, while UNet-based deterministic models provide more stable diffusion performance. Therefore, in the DERDM-SE, we propose a deterministic model that combines coarse- and fine-grained processing. Experimental results on CHiME4 show that the proposed models effectively leverage deterministic models to achieve better SE evaluation scores, along with more stable performance compared to other diffusion-based SE models.
>
---
#### [new 012] AquaSignal: An Integrated Framework for Robust Underwater Acoustic Analysis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出AquaSignal框架，解决水下复杂环境中声学信号分析的鲁棒性问题。整合U-Net降噪、ResNet18分类及自编码器异常检测模块，提升预处理、分类和新颖信号识别性能，适用于海洋监测等实时场景。**

- **链接: [http://arxiv.org/pdf/2505.14285v1](http://arxiv.org/pdf/2505.14285v1)**

> **作者:** Eirini Panteli; Paulo E. Santos; Nabil Humphrey
>
> **备注:** 8 pages; 9 figures
>
> **摘要:** This paper presents AquaSignal, a modular and scalable pipeline for preprocessing, denoising, classification, and novelty detection of underwater acoustic signals. Designed to operate effectively in noisy and dynamic marine environments, AquaSignal integrates state-of-the-art deep learning architectures to enhance the reliability and accuracy of acoustic signal analysis. The system is evaluated on a combined dataset from the Deepship and Ocean Networks Canada (ONC) benchmarks, providing a diverse set of real-world underwater scenarios. AquaSignal employs a U-Net architecture for denoising, a ResNet18 convolutional neural network for classifying known acoustic events, and an AutoEncoder-based model for unsupervised detection of novel or anomalous signals. To our knowledge, this is the first comprehensive study to apply and evaluate this combination of techniques on maritime vessel acoustic data. Experimental results show that AquaSignal improves signal clarity and task performance, achieving 71% classification accuracy and 91% accuracy in novelty detection. Despite slightly lower classification performance compared to some state-of-the-art models, differences in data partitioning strategies limit direct comparisons. Overall, AquaSignal demonstrates strong potential for real-time underwater acoustic monitoring in scientific, environmental, and maritime domains.
>
---
#### [new 013] Representation Learning for Semantic Alignment of Language, Audio, and Visual Modalities
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出单阶段对比学习方法，解决现有两阶段多模态对齐的数据分布不匹配问题，通过联合优化音频、视觉、文本的统一表征，在AVCaps数据集实现更优性能，音频驱动视觉检索精度提升一倍。**

- **链接: [http://arxiv.org/pdf/2505.14562v1](http://arxiv.org/pdf/2505.14562v1)**

> **作者:** Parthasaarathy Sudarsanam; Irene Martín-Morató; Tuomas Virtanen
>
> **备注:** Accepted to European Signal Processing Conference (EUSIPCO 2025)
>
> **摘要:** This paper proposes a single-stage training approach that semantically aligns three modalities - audio, visual, and text using a contrastive learning framework. Contrastive training has gained prominence for multimodal alignment, utilizing large-scale unlabeled data to learn shared representations. Existing deep learning approach for trimodal alignment involves two-stages, that separately align visual-text and audio-text modalities. This approach suffers from mismatched data distributions, resulting in suboptimal alignment. Leveraging the AVCaps dataset, which provides audio, visual and audio-visual captions for video clips, our method jointly optimizes the representation of all the modalities using contrastive training. Our results demonstrate that the single-stage approach outperforms the two-stage method, achieving a two-fold improvement in audio based visual retrieval, highlighting the advantages of unified multimodal representation learning.
>
---
#### [new 014] Bridging Speech Emotion Recognition and Personality: Dataset and Temporal Interaction Condition Network
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别（SER）任务，旨在利用人格特质提升情感识别精度。通过扩展IEMOCAP数据集的人格标注，发现人格与情感的显著关联。提出TICN模型融合声学与人格特征，显著提升效价识别（CCC从0.698至0.785），并开发自动人格识别模块实现11.17%的相对提升，验证了人格感知SER的有效性。**

- **链接: [http://arxiv.org/pdf/2505.13978v1](http://arxiv.org/pdf/2505.13978v1)**

> **作者:** Yuan Gao; Hao Shi; Yahui Fu; Chenhui Chu; Tatsuya Kawahara
>
> **摘要:** This study investigates the interaction between personality traits and emotional expression, exploring how personality information can improve speech emotion recognition (SER). We collected personality annotation for the IEMOCAP dataset, and the statistical analysis identified significant correlations between personality traits and emotional expressions. To extract finegrained personality features, we propose a temporal interaction condition network (TICN), in which personality features are integrated with Hubert-based acoustic features for SER. Experiments show that incorporating ground-truth personality traits significantly enhances valence recognition, improving the concordance correlation coefficient (CCC) from 0.698 to 0.785 compared to the baseline without personality information. For practical applications in dialogue systems where personality information about the user is unavailable, we develop a front-end module of automatic personality recognition. Using these automatically predicted traits as inputs to our proposed TICN model, we achieve a CCC of 0.776 for valence recognition, representing an 11.17% relative improvement over the baseline. These findings confirm the effectiveness of personality-aware SER and provide a solid foundation for further exploration in personality-aware speech processing applications.
>
---
#### [new 015] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型任务，旨在解决现有模型在声音细粒度语义推理上的不足。提出AudSemThinker模型，基于人类听觉语义框架构建推理结构，并构建AudSem数据集，通过多阶段生成过滤数据污染，提升零样本评估可靠性。实验表明其推理性能超越现有模型，成果已开源。**

- **链接: [http://arxiv.org/pdf/2505.14142v1](http://arxiv.org/pdf/2505.14142v1)**

> **作者:** Gijs Wijngaard; Elia Formisano; Michele Esposito; Michel Dumontier
>
> **摘要:** Audio-language models have shown promising results in various sound understanding tasks, yet they remain limited in their ability to reason over the fine-grained semantics of sound. In this paper, we present AudSemThinker, a model whose reasoning is structured around a framework of auditory semantics inspired by human cognition. To support this, we introduce AudSem, a novel dataset specifically curated for semantic descriptor reasoning in audio-language models. AudSem addresses the persistent challenge of data contamination in zero-shot evaluations by providing a carefully filtered collection of audio samples paired with captions generated through a robust multi-stage pipeline. Our experiments demonstrate that AudSemThinker outperforms state-of-the-art models across multiple training settings, highlighting its strength in semantic audio reasoning. Both AudSemThinker and the AudSem dataset are released publicly.
>
---
#### [new 016] BiCrossMamba-ST: Speech Deepfake Detection with Bidirectional Mamba Spectro-Temporal Cross-Attention
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出BiCrossMamba-ST框架，用于语音深度伪造检测。针对合成语音细微特征捕捉难题，其采用双分支架构分别处理频谱子带和时序信息，结合双向Mamba块、跨模态注意力及2D卷积注意力机制，直接分析原始声学特征，在ASVSpoof基准测试中显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13930v1](http://arxiv.org/pdf/2505.13930v1)**

> **作者:** Yassine El Kheir; Tim Polzehl; Sebastian Möller
>
> **备注:** Accepted Interspeech 2025
>
> **摘要:** We propose BiCrossMamba-ST, a robust framework for speech deepfake detection that leverages a dual-branch spectro-temporal architecture powered by bidirectional Mamba blocks and mutual cross-attention. By processing spectral sub-bands and temporal intervals separately and then integrating their representations, BiCrossMamba-ST effectively captures the subtle cues of synthetic speech. In addition, our proposed framework leverages a convolution-based 2D attention map to focus on specific spectro-temporal regions, enabling robust deepfake detection. Operating directly on raw features, BiCrossMamba-ST achieves significant performance improvements, a 67.74% and 26.3% relative gain over state-of-the-art AASIST on ASVSpoof LA21 and ASVSpoof DF21 benchmarks, respectively, and a 6.80% improvement over RawBMamba on ASVSpoof DF21. Code and models will be made publicly available.
>
---
#### [new 017] PAST: Phonetic-Acoustic Speech Tokenizer
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文提出PAST，一种端到端语音分词框架，联合建模音素与声学信息，无需外部预训练模型。旨在解决传统方法依赖预训练模型及实时性不足的问题，通过监督学习与因果变体设计，提升语音表示、重建及实时应用性能，增强语音语言模型效果。**

- **链接: [http://arxiv.org/pdf/2505.14470v1](http://arxiv.org/pdf/2505.14470v1)**

> **作者:** Nadav Har-Tuv; Or Tal; Yossi Adi
>
> **摘要:** We present PAST, a novel end-to-end framework that jointly models phonetic information alongside signal reconstruction, eliminating the need for external pretrained models. Unlike previous approaches that rely on pretrained self-supervised models, PAST employs supervised phonetic data, directly integrating domain knowledge into the tokenization process via auxiliary tasks. Additionally, we introduce a streamable, causal variant of PAST, enabling real-time speech applications. Results demonstrate that PAST surpasses existing evaluated baseline tokenizers across common evaluation metrics, including phonetic representation and speech reconstruction. Notably, PAST also achieves superior performance when serving as a speech representation for speech language models, further highlighting its effectiveness as a foundation for spoken language generation. To foster further research, we release the full implementation. For code, model checkpoints, and samples see: https://pages.cs.huji.ac.il/adiyoss-lab/PAST
>
---
#### [new 018] VocalAgent: Large Language Models for Vocal Health Diagnostics with Safety-Aware Evaluation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于声带健康诊断任务，解决全球语音障碍诊断不便的问题。提出VocalAgent模型，基于Qwen-Audio-Chat微调医疗数据，通过安全评估、跨语言分析等多维度验证，在声带疾病分类中优于现有方法，提供可扩展的诊断方案并强调伦理验证。**

- **链接: [http://arxiv.org/pdf/2505.13577v1](http://arxiv.org/pdf/2505.13577v1)**

> **作者:** Yubin Kim; Taehan Kim; Wonjune Kang; Eugene Park; Joonsik Yoon; Dongjae Lee; Xin Liu; Daniel McDuff; Hyeonhoon Lee; Cynthia Breazeal; Hae Won Park
>
> **摘要:** Vocal health plays a crucial role in peoples' lives, significantly impacting their communicative abilities and interactions. However, despite the global prevalence of voice disorders, many lack access to convenient diagnosis and treatment. This paper introduces VocalAgent, an audio large language model (LLM) to address these challenges through vocal health diagnosis. We leverage Qwen-Audio-Chat fine-tuned on three datasets collected in-situ from hospital patients, and present a multifaceted evaluation framework encompassing a safety assessment to mitigate diagnostic biases, cross-lingual performance analysis, and modality ablation studies. VocalAgent demonstrates superior accuracy on voice disorder classification compared to state-of-the-art baselines. Its LLM-based method offers a scalable solution for broader adoption of health diagnostics, while underscoring the importance of ethical and technical validation.
>
---
#### [new 019] Forensic deepfake audio detection using segmental speech features
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于深度伪造音频检测任务，旨在通过分段语音特征提升检测效果。针对现有方法对全局特征依赖但效果有限的问题，研究探索了与人类发声机制相关的分段声学特征（如音素、重音等），发现其在识别深度伪造音频中更有效，而全局特征价值较低，为法医语音鉴定提供了新方法。**

- **链接: [http://arxiv.org/pdf/2505.13847v1](http://arxiv.org/pdf/2505.13847v1)**

> **作者:** Tianle Yang; Chengzhe Sun; Siwei Lyu; Phil Rose
>
> **摘要:** This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection differently for forensic voice comparison and offer a new perspective on leveraging segmental features for this purpose.
>
---
#### [new 020] Direction-Aware Neural Acoustic Fields for Few-Shot Interpolation of Ambisonic Impulse Responses
- **分类: eess.AS; cs.AI; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于声场建模任务，解决现有神经场方法无法准确捕捉声场方向特性的问题。提出方向感知神经声场（DANF），利用Ambisonic格式RIR显式整合方向信息，并设计方向损失函数，同时研究其少样本适配新环境（如低秩调整）的能力。**

- **链接: [http://arxiv.org/pdf/2505.13617v1](http://arxiv.org/pdf/2505.13617v1)**

> **作者:** Christopher Ick; Gordon Wichern; Yoshiki Masuyama; François Germain; Jonathan Le Roux
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** The characteristics of a sound field are intrinsically linked to the geometric and spatial properties of the environment surrounding a sound source and a listener. The physics of sound propagation is captured in a time-domain signal known as a room impulse response (RIR). Prior work using neural fields (NFs) has allowed learning spatially-continuous representations of RIRs from finite RIR measurements. However, previous NF-based methods have focused on monaural omnidirectional or at most binaural listeners, which does not precisely capture the directional characteristics of a real sound field at a single point. We propose a direction-aware neural field (DANF) that more explicitly incorporates the directional information by Ambisonic-format RIRs. While DANF inherently captures spatial relations between sources and listeners, we further propose a direction-aware loss. In addition, we investigate the ability of DANF to adapt to new rooms in various ways including low-rank adaptation.
>
---
#### [new 021] SeamlessEdit: Background Noise Aware Zero-Shot Speech Editing with in-Context Enhancement
- **分类: eess.AS; cs.SD; 68T45; I.2.7; H.5.5**

- **简介: 该论文属于零样本语音编辑任务，针对现实场景中背景噪声干扰问题，提出SeamlessEdit框架。其通过频段感知降噪模块和上下文优化策略，解决语音与噪声频段重叠时的编辑难题，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14066v1](http://arxiv.org/pdf/2505.14066v1)**

> **作者:** Kuan-Yu Chen; Jeng-Lin Li; Jian-Jiun Ding
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** With the fast development of zero-shot text-to-speech technologies, it is possible to generate high-quality speech signals that are indistinguishable from the real ones. Speech editing, including speech insertion and replacement, appeals to researchers due to its potential applications. However, existing studies only considered clean speech scenarios. In real-world applications, the existence of environmental noise could significantly degrade the quality of the generation. In this study, we propose a noise-resilient speech editing framework, SeamlessEdit, for noisy speech editing. SeamlessEdit adopts a frequency-band-aware noise suppression module and an in-content refinement strategy. It can well address the scenario where the frequency bands of voice and background noise are not separated. The proposed SeamlessEdit framework outperforms state-of-the-art approaches in multiple quantitative and qualitative evaluations.
>
---
#### [new 022] Steering Deep Non-Linear Spatially Selective Filters for Weakly Guided Extraction of Moving Speakers in Dynamic Scenarios
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音提取任务，旨在解决动态场景下移动说话人提取的挑战，如目标方向变化和说话人交叉导致的歧义。提出仅依赖初始位置的弱指导方法，结合深度跟踪算法与联合训练策略，提升动态场景中的提取性能，超越强指导方法。**

- **链接: [http://arxiv.org/pdf/2505.14517v1](http://arxiv.org/pdf/2505.14517v1)**

> **作者:** Jakob Kienegger; Timo Gerkmann
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Recent speaker extraction methods using deep non-linear spatial filtering perform exceptionally well when the target direction is known and stationary. However, spatially dynamic scenarios are considerably more challenging due to time-varying spatial features and arising ambiguities, e.g. when moving speakers cross. While in a static scenario it may be easy for a user to point to the target's direction, manually tracking a moving speaker is impractical. Instead of relying on accurate time-dependent directional cues, which we refer to as strong guidance, in this paper we propose a weakly guided extraction method solely depending on the target's initial position to cope with spatial dynamic scenarios. By incorporating our own deep tracking algorithm and developing a joint training strategy on a synthetic dataset, we demonstrate the proficiency of our approach in resolving spatial ambiguities and even outperform a mismatched, but strongly guided extraction method.
>
---
#### [new 023] Articulatory Feature Prediction from Surface EMG during Speech Production
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出一种结合卷积层与Transformer的模型，从表面肌电图（EMG）预测发音特征并合成可懂语音，首次通过发音特征解码语音波形，分析电极位置对预测的影响，优化EMG配置。任务为EMG语音合成，解决肌电信号到语音的高精度转换问题。**

- **链接: [http://arxiv.org/pdf/2505.13814v1](http://arxiv.org/pdf/2505.13814v1)**

> **作者:** Jihwan Lee; Kevin Huang; Kleanthis Avramidis; Simon Pistrosch; Monica Gonzalez-Machorro; Yoonjeong Lee; Björn Schuller; Louis Goldstein; Shrikanth Narayanan
>
> **备注:** Accepted for Interspeech2025
>
> **摘要:** We present a model for predicting articulatory features from surface electromyography (EMG) signals during speech production. The proposed model integrates convolutional layers and a Transformer block, followed by separate predictors for articulatory features. Our approach achieves a high prediction correlation of approximately 0.9 for most articulatory features. Furthermore, we demonstrate that these predicted articulatory features can be decoded into intelligible speech waveforms. To our knowledge, this is the first method to decode speech waveforms from surface EMG via articulatory features, offering a novel approach to EMG-based speech synthesis. Additionally, we analyze the relationship between EMG electrode placement and articulatory feature predictability, providing knowledge-driven insights for optimizing EMG electrode configurations. The source code and decoded speech samples are publicly available.
>
---
#### [new 024] SSPS: Self-Supervised Positive Sampling for Robust Self-Supervised Speaker Verification
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于自监督说话人验证任务，旨在解决传统正样本采样过度依赖同源语音导致模型编码通道信息而非说话人特征的问题。提出SSPS方法，利用聚类和记忆队列在潜在空间筛选同一说话人不同录音条件的正样本，降低 intra-variance，提升模型鲁棒性，在VoxCeleb1-O达2.53%-2.57% EER，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14561v1](http://arxiv.org/pdf/2505.14561v1)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** accepted at Interspeech 2025
>
> **摘要:** Self-Supervised Learning (SSL) has led to considerable progress in Speaker Verification (SV). The standard framework uses same-utterance positive sampling and data-augmentation to generate anchor-positive pairs of the same speaker. This is a major limitation, as this strategy primarily encodes channel information from the recording condition, shared by the anchor and positive. We propose a new positive sampling technique to address this bottleneck: Self-Supervised Positive Sampling (SSPS). For a given anchor, SSPS aims to find an appropriate positive, i.e., of the same speaker identity but a different recording condition, in the latent space using clustering assignments and a memory queue of positive embeddings. SSPS improves SV performance for both SimCLR and DINO, reaching 2.57% and 2.53% EER, outperforming SOTA SSL methods on VoxCeleb1-O. In particular, SimCLR-SSPS achieves a 58% EER reduction by lowering intra-speaker variance, providing comparable performance to DINO-SSPS.
>
---
#### [new 025] Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究针对语音大模型（Speech-LLMs）的通用声学对抗攻击，旨在通过插入固定音频段控制模型输出。提出两种攻击：强制无响应或覆盖原始指令，以及基于说话人性别/语言等属性的选择性触发攻击。实验揭示Qwen2-Audio和Granite-Speech存在漏洞，强调需提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.14286v1](http://arxiv.org/pdf/2505.14286v1)**

> **作者:** Rao Ma; Mengjie Qian; Vyas Raina; Mark Gales; Kate Knill
>
> **摘要:** The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.
>
---
#### [new 026] Pushing the Frontiers of Self-Distillation Prototypes Network with Dimension Regularization and Score Normalization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于说话人验证任务，旨在提升无监督自监督模型性能。针对自监督方法与全监督方法的性能差距及嵌入坍塌问题，提出在Self-Distillation Prototypes Network（SDPN）中引入维度正则化（解决嵌入坍塌）和分数归一化（借鉴监督方法技术），实现VoxCeleb1基准测试新SOTA，EER降低22%-28%。**

- **链接: [http://arxiv.org/pdf/2505.13826v1](http://arxiv.org/pdf/2505.13826v1)**

> **作者:** Yafeng Chen; Chong Deng; Hui Wang; Yiheng Jiang; Han Yin; Qian Chen; Wen Wang
>
> **摘要:** Developing robust speaker verification (SV) systems without speaker labels has been a longstanding challenge. Earlier research has highlighted a considerable performance gap between self-supervised and fully supervised approaches. In this paper, we enhance the non-contrastive self-supervised framework, Self-Distillation Prototypes Network (SDPN), by introducing dimension regularization that explicitly addresses the collapse problem through the application of regularization terms to speaker embeddings. Moreover, we integrate score normalization techniques from fully supervised SV to further bridge the gap toward supervised verification performance. SDPN with dimension regularization and score normalization sets a new state-of-the-art on the VoxCeleb1 speaker verification evaluation benchmark, achieving Equal Error Rate 1.29%, 1.60%, and 2.80% for trial VoxCeleb1-{O,E,H} respectively. These results demonstrate relative improvements of 28.3%, 19.6%, and 22.6% over the current best self-supervised methods, thereby advancing the frontiers of SV technology.
>
---
#### [new 027] Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于提升LLM驱动零样本TTS抗噪性的任务。针对噪声音频提示导致合成语音质量下降的问题，提出基于神经编码器的去噪框架：通过音频编码器提取噪声声学标记，经token去噪模块预测干净标记，再通过嵌入优化生成高质量语音。集成到LauraTTS后显著提升抗噪性能，优于传统语音增强方法。（98字）**

- **链接: [http://arxiv.org/pdf/2505.13830v1](http://arxiv.org/pdf/2505.13830v1)**

> **作者:** Ye-Xin Lu; Hui-Peng Du; Fei Liu; Yang Ai; Zhen-Hua Ling
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
>
---
#### [new 028] A Semantic Information-based Hierarchical Speech Enhancement Method Using Factorized Codec and Diffusion Model
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，针对传统方法忽视语义及声学细节导致复杂环境性能下降的问题，提出基于语义信息的分层增强方法，结合分解式编解码器与扩散模型，分层建模语音语义和声学属性，提升恢复鲁棒性及下游TTS性能，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.13843v1](http://arxiv.org/pdf/2505.13843v1)**

> **作者:** Yang Xiang; Canan Huang; Desheng Hu; Jingguang Tian; Xinhui Hu; Chao Zhang
>
> **备注:** Accepted by interspeech 2025
>
> **摘要:** Most current speech enhancement (SE) methods recover clean speech from noisy inputs by directly estimating time-frequency masks or spectrums. However, these approaches often neglect the distinct attributes, such as semantic content and acoustic details, inherent in speech signals, which can hinder performance in downstream tasks. Moreover, their effectiveness tends to degrade in complex acoustic environments. To overcome these challenges, we propose a novel, semantic information-based, step-by-step factorized SE method using factorized codec and diffusion model. Unlike traditional SE methods, our hierarchical modeling of semantic and acoustic attributes enables more robust clean speech recovery, particularly in challenging acoustic scenarios. Moreover, this method offers further advantages for downstream TTS tasks. Experimental results demonstrate that our algorithm not only outperforms SOTA baselines in terms of speech quality but also enhances TTS performance in noisy environments.
>
---
#### [new 029] Sat2Sound: A Unified Framework for Zero-Shot Soundscape Mapping
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: Sat2Sound提出零样本声音景观映射框架，解决现有方法依赖配对数据难以捕捉声音多样性的问题。通过VLM生成卫星图像的语义声音描述，结合跨模态对比学习，构建共享声音概念代码本，提升卫星图像与音频的跨模态检索性能，并实现基于位置的声音合成应用。**

- **链接: [http://arxiv.org/pdf/2505.13777v1](http://arxiv.org/pdf/2505.13777v1)**

> **作者:** Subash Khanal; Srikumar Sastry; Aayush Dhakal; Adeel Ahmad; Nathan Jacobs
>
> **摘要:** We present Sat2Sound, a multimodal representation learning framework for soundscape mapping, designed to predict the distribution of sounds at any location on Earth. Existing methods for this task rely on satellite image and paired geotagged audio samples, which often fail to capture the diversity of sound sources at a given location. To address this limitation, we enhance existing datasets by leveraging a Vision-Language Model (VLM) to generate semantically rich soundscape descriptions for locations depicted in satellite images. Our approach incorporates contrastive learning across audio, audio captions, satellite images, and satellite image captions. We hypothesize that there is a fixed set of soundscape concepts shared across modalities. To this end, we learn a shared codebook of soundscape concepts and represent each sample as a weighted average of these concepts. Sat2Sound achieves state-of-the-art performance in cross-modal retrieval between satellite image and audio on two datasets: GeoSound and SoundingEarth. Additionally, building on Sat2Sound's ability to retrieve detailed soundscape captions, we introduce a novel application: location-based soundscape synthesis, which enables immersive acoustic experiences. Our code and models will be publicly available.
>
---
#### [new 030] Pairwise Evaluation of Accent Similarity in Speech Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音合成中口音相似性评估任务，旨在解决现有主观和客观评估方法不足的问题，尤其针对少数群体口音。工作包括优化XAB测试（提供转录、标注差异、筛选可靠度），引入基于元音共振峰和语音后验图的客观指标，并指出WER等传统指标的局限性。**

- **链接: [http://arxiv.org/pdf/2505.14410v1](http://arxiv.org/pdf/2505.14410v1)**

> **作者:** Jinzuomu Zhong; Suyuan Liu; Dan Wells; Korin Richmond
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Despite growing interest in generating high-fidelity accents, evaluating accent similarity in speech synthesis has been underexplored. We aim to enhance both subjective and objective evaluation methods for accent similarity. Subjectively, we refine the XAB listening test by adding components that achieve higher statistical significance with fewer listeners and lower costs. Our method involves providing listeners with transcriptions, having them highlight perceived accent differences, and implementing meticulous screening for reliability. Objectively, we utilise pronunciation-related metrics, based on distances between vowel formants and phonetic posteriorgrams, to evaluate accent generation. Comparative experiments reveal that these metrics, alongside accent similarity, speaker similarity, and Mel Cepstral Distortion, can be used. Moreover, our findings underscore significant limitations of common metrics like Word Error Rate in assessing underrepresented accents.
>
---
#### [new 031] Listen, Analyze, and Adapt to Learn New Attacks: An Exemplar-Free Class Incremental Learning Method for Audio Deepfake Source Tracing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频深度伪造溯源任务，解决模型在增量学习新攻击时的灾难性遗忘问题。提出AnaST方法，固定特征提取器，通过闭式解更新分类器，无需存储旧数据，实现高效在线训练，优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.14601v1](http://arxiv.org/pdf/2505.14601v1)**

> **作者:** Yang Xiao; Rohan Kumar Das
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** As deepfake speech becomes common and hard to detect, it is vital to trace its source. Recent work on audio deepfake source tracing (ST) aims to find the origins of synthetic or manipulated speech. However, ST models must adapt to learn new deepfake attacks while retaining knowledge of the previous ones. A major challenge is catastrophic forgetting, where models lose the ability to recognize previously learned attacks. Some continual learning methods help with deepfake detection, but multi-class tasks such as ST introduce additional challenges as the number of classes grows. To address this, we propose an analytic class incremental learning method called AnaST. When new attacks appear, the feature extractor remains fixed, and the classifier is updated with a closed-form analytical solution in one epoch. This approach ensures data privacy, optimizes memory usage, and is suitable for online training. The experiments carried out in this work show that our method outperforms the baselines.
>
---
#### [new 032] AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models
- **分类: cs.CR; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文研究针对端到端大型音频语言模型（LALMs）的越狱攻击任务，解决现有攻击效果差、适用性窄及需完全控制用户输入的问题。提出AudioJailbreak方法，通过异步音频设计、通用扰动生成、隐蔽策略及环境鲁棒优化，实现高效隐蔽的攻击，扩展了攻击场景并经实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.14103v1](http://arxiv.org/pdf/2505.14103v1)**

> **作者:** Guangke Chen; Fu Song; Zhe Zhao; Xiaojun Jia; Yang Liu; Yanchen Qiao; Weizhe Zhang
>
> **摘要:** Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website https://audiojailbreak.github.io/AudioJailbreak.
>
---
#### [new 033] U-SAM: An audio language Model for Unified Speech, Audio, and Music Understanding
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文提出U-SAM模型，针对语音、通用音频和音乐的统一理解任务。解决现有模型跨模态对齐不足及冗余特征处理问题，通过多专家混合特征融合与语义对比损失模块优化，结合预训练语言模型提升跨模态对齐。实验显示其性能优于现有模型并具备泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.13880v1](http://arxiv.org/pdf/2505.13880v1)**

> **作者:** Ziqian Wang; Xianjun Xia; Xinfa Zhu; Lei Xie
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The text generation paradigm for audio tasks has opened new possibilities for unified audio understanding. However, existing models face significant challenges in achieving a comprehensive understanding across diverse audio types, such as speech, general audio events, and music. Furthermore, their exclusive reliance on cross-entropy loss for alignment often falls short, as it treats all tokens equally and fails to account for redundant audio features, leading to weaker cross-modal alignment. To deal with the above challenges, this paper introduces U-SAM, an advanced audio language model that integrates specialized encoders for speech, audio, and music with a pre-trained large language model (LLM). U-SAM employs a Mixture of Experts (MoE) projector for task-aware feature fusion, dynamically routing and integrating the domain-specific encoder outputs. Additionally, U-SAM incorporates a Semantic-Aware Contrastive Loss Module, which explicitly identifies redundant audio features under language supervision and rectifies their semantic and spectral representations to enhance cross-modal alignment. Extensive experiments demonstrate that U-SAM consistently outperforms both specialized models and existing audio language models across multiple benchmarks. Moreover, it exhibits emergent capabilities on unseen tasks, showcasing its generalization potential. Code is available (https://github.com/Honee-W/U-SAM/).
>
---
#### [new 034] Teaching Audio-Aware Large Language Models What Does Not Hear: Mitigating Hallucinations through Synthesized Negative Samples
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文针对音频感知大语言模型（ALLMs）虚构不存在声音的问题，提出LISTEN方法。通过合成负样本对比训练，增强模型区分真实与不存在声音的能力，采用轻量适配器无需修改模型参数。实验显示有效减少幻觉且计算高效。**

- **链接: [http://arxiv.org/pdf/2505.14518v1](http://arxiv.org/pdf/2505.14518v1)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Recent advancements in audio-aware large language models (ALLMs) enable them to process and understand audio inputs. However, these models often hallucinate non-existent sound events, reducing their reliability in real-world applications. To address this, we propose LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method that enhances ALLMs' ability to distinguish between present and absent sounds using synthesized data from the backbone LLM. Unlike prior approaches, our method requires no modification to LLM parameters and efficiently integrates audio representations via a lightweight adapter. Experiments show that LISTEN effectively mitigates hallucinations while maintaining impressive performance on existing audio question and reasoning benchmarks. At the same time, it is more efficient in both data and computation.
>
---
#### [new 035] Naturalness-Aware Curriculum Learning with Dynamic Temperature for Speech Deepfake Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有模型忽视语音自然度导致检测鲁棒性不足的问题。提出自然度感知课程学习框架，结合动态温度缩放，通过样本难度排序（基于标签和MOS）逐步训练模型，提升泛化能力，实验显示EER降低23%。**

- **链接: [http://arxiv.org/pdf/2505.13976v1](http://arxiv.org/pdf/2505.13976v1)**

> **作者:** Taewoo Kim; Guisik Kim; Choongsang Cho; Young Han Lee
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Recent advances in speech deepfake detection (SDD) have significantly improved artifacts-based detection in spoofed speech. However, most models overlook speech naturalness, a crucial cue for distinguishing bona fide speech from spoofed speech. This study proposes naturalness-aware curriculum learning, a novel training framework that leverages speech naturalness to enhance the robustness and generalization of SDD. This approach measures sample difficulty using both ground-truth labels and mean opinion scores, and adjusts the training schedule to progressively introduce more challenging samples. To further improve generalization, a dynamic temperature scaling method based on speech naturalness is incorporated into the training process. A 23% relative reduction in the EER was achieved in the experiments on the ASVspoof 2021 DF dataset, without modifying the model architecture. Ablation studies confirmed the effectiveness of naturalness-aware training strategies for SDD tasks.
>
---
#### [new 036] Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach
- **分类: eess.AS; cs.CV; cs.MM; cs.SD**

- **简介: 该论文属于视听语音识别（AVSR）任务，旨在解决基于大语言模型（LLM）的AVSR计算成本过高问题。提出Llama-SMoP方法，通过稀疏投影器混合（SMoP）模块扩展模型容量而不增加推理成本，采用模态专用路由和专家配置（DEDR），提升多模态LLM的效率与性能，消融实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.14336v1](http://arxiv.org/pdf/2505.14336v1)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Stavros Petridis; Daniele Falavigna; Alessio Brutti
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
>
---
#### [new 037] Single-Channel Target Speech Extraction Utilizing Distance and Room Clues
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于单通道目标语音提取任务，旨在解决仅依赖距离线索的语音分离模型难以跨房间泛化的问题。提出结合房间维度与混响时间信息，在时频域构建带可学习嵌入的联合模型，提升语音提取的环境适应性。**

- **链接: [http://arxiv.org/pdf/2505.14433v1](http://arxiv.org/pdf/2505.14433v1)**

> **作者:** Runwu Shi; Zirui Lin; Benjamin Yen; Jiang Wang; Ragib Amin Nihal; Kazuhiro Nakadai
>
> **备注:** 5 pages, 3 figures, accepted by Eusipco 2025
>
> **摘要:** This paper aims to achieve single-channel target speech extraction (TSE) in enclosures utilizing distance clues and room information. Recent works have verified the feasibility of distance clues for the TSE task, which can imply the sound source's direct-to-reverberation ratio (DRR) and thus can be utilized for speech separation and TSE systems. However, such distance clue is significantly influenced by the room's acoustic characteristics, such as dimension and reverberation time, making it challenging for TSE systems that rely solely on distance clues to generalize across a variety of different rooms. To solve this, we suggest providing room environmental information (room dimensions and reverberation time) for distance-based TSE for better generalization capabilities. Especially, we propose a distance and environment-based TSE model in the time-frequency (TF) domain with learnable distance and room embedding. Results on both simulated and real collected datasets demonstrate its feasibility. Demonstration materials are available at https://runwushi.github.io/distance-room-demo-page/.
>
---
#### [new 038] Recreating Neural Activity During Speech Production with Language and Speech Model Embeddings
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于神经活动重建任务，旨在探究语言及语音模型的嵌入是否能有效复现人类说话时的脑神经信号。研究通过映射预训练模型的嵌入特征到脑电数据，评估其时空动态保真度，结果表明嵌入可高度（0.79-0.99皮尔逊系数）重建多被试的神经活动。**

- **链接: [http://arxiv.org/pdf/2505.14074v1](http://arxiv.org/pdf/2505.14074v1)**

> **作者:** Owais Mujtaba Khanday; Pablo Rodroguez San Esteban; Zubair Ahmad Lone; Marc Ouellet; Jose Andres Gonzalez Lopez
>
> **备注:** Accepted for presentation at Interspeech2025
>
> **摘要:** Understanding how neural activity encodes speech and language production is a fundamental challenge in neuroscience and artificial intelligence. This study investigates whether embeddings from large-scale, self-supervised language and speech models can effectively reconstruct neural activity recordings captured during speech production. We leverage pre-trained embeddings from deep learning models trained on linguistic and acoustic data to represent high-level speech features and map them onto neural signals. We analyze the extent to which these embeddings preserve the spatio-temporal dynamics of brain activity. We evaluate reconstructed neural signals against ground truth recordings using correlation metrics and signal reconstruction quality assessments. The results indicate that neural activity can be effectively reconstructed using embeddings from large language and speech models across all study participants, yielding Pearson correlation coefficients ranging from 0.79 to 0.99.
>
---
#### [new 039] AdaKWS: Towards Robust Keyword Spotting with Test-Time Adaptation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音关键词检测（KWS）任务，旨在解决模型在未见环境或噪声下的性能下降问题。提出AdaKWS方法，通过预测熵最小化筛选可靠样本并调整批归一化参数，结合伪关键词一致性约束，提升噪声场景下的检测鲁棒性，实验显示其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14600v1](http://arxiv.org/pdf/2505.14600v1)**

> **作者:** Yang Xiao; Tianyi Peng; Yanghao Zhou; Rohan Kumar Das
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Spoken keyword spotting (KWS) aims to identify keywords in audio for wide applications, especially on edge devices. Current small-footprint KWS systems focus on efficient model designs. However, their inference performance can decline in unseen environments or noisy backgrounds. Test-time adaptation (TTA) helps models adapt to test samples without needing the original training data. In this study, we present AdaKWS, the first TTA method for robust KWS to the best of our knowledge. Specifically, 1) We initially optimize the model's confidence by selecting reliable samples based on prediction entropy minimization and adjusting the normalization statistics in each batch. 2) We introduce pseudo-keyword consistency (PKC) to identify critical, reliable features without overfitting to noise. Our experiments show that AdaKWS outperforms other methods across various conditions, including Gaussian noise and real-scenario noises. The code will be released in due course.
>
---
## 更新

#### [replaced 001] The 1st SpeechWellness Challenge: Detecting Suicide Risk Among Adolescents
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.06474v2](http://arxiv.org/pdf/2501.06474v2)**

> **作者:** Wen Wu; Ziyun Cui; Chang Lei; Yinan Duan; Diyang Qu; Ji Wu; Bowen Zhou; Runsen Chen; Chao Zhang
>
> **摘要:** The 1st SpeechWellness Challenge (SW1) aims to advance methods for detecting current suicide risk in adolescents using speech analysis techniques. Suicide among adolescents is a critical public health issue globally. Early detection of suicidal tendencies can lead to timely intervention and potentially save lives. Traditional methods of assessment often rely on self-reporting or clinical interviews, which may not always be accessible. The SW1 challenge addresses this gap by exploring speech as a non-invasive and readily available indicator of mental health. We release the SW1 dataset which contains speech recordings from 600 adolescents aged 10-18 years. By focusing on speech generated from natural tasks, the challenge seeks to uncover patterns and markers that correlate with current suicide risk.
>
---
#### [replaced 002] USEF-TSE: Universal Speaker Embedding Free Target Speaker Extraction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.02615v3](http://arxiv.org/pdf/2409.02615v3)**

> **作者:** Bang Zeng; Ming Li
>
> **备注:** Accepted by IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Target speaker extraction aims to separate the voice of a specific speaker from mixed speech. Traditionally, this process has relied on extracting a speaker embedding from a reference speech, in which a speaker recognition model is required. However, identifying an appropriate speaker recognition model can be challenging, and using the target speaker embedding as reference information may not be optimal for target speaker extraction tasks. This paper introduces a Universal Speaker Embedding-Free Target Speaker Extraction (USEF-TSE) framework that operates without relying on speaker embeddings. USEF-TSE utilizes a multi-head cross-attention mechanism as a frame-level target speaker feature extractor. This innovative approach allows mainstream speaker extraction solutions to bypass the dependency on speaker recognition models and better leverage the information available in the enrollment speech, including speaker characteristics and contextual details. Additionally, USEF-TSE can seamlessly integrate with other time-domain or time-frequency domain speech separation models to achieve effective speaker extraction. Experimental results show that our proposed method achieves state-of-the-art (SOTA) performance in terms of Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) on the WSJ0-2mix, WHAM!, and WHAMR! datasets, which are standard benchmarks for monaural anechoic, noisy and noisy-reverberant two-speaker speech separation and speaker extraction. The results on the LibriMix and the blind test set of the ICASSP 2023 DNS Challenge demonstrate that the model performs well on more diverse and out-of-domain data. For access to the source code, please visit: https://github.com/ZBang/USEF-TSE.
>
---
#### [replaced 003] Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.17772v2](http://arxiv.org/pdf/2501.17772v2)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** submitted to IEEE/ACM TASLP in January 2025
>
> **摘要:** Recent developments in Self-Supervised Learning (SSL) have demonstrated significant potential for Speaker Verification (SV), but closing the performance gap with supervised systems remains an ongoing challenge. Standard SSL frameworks rely on anchor-positive pairs extracted from the same audio utterances. Hence, positives have channel characteristics similar to those of their corresponding anchors, even with extensive data-augmentation. Therefore, this positive sampling strategy is a fundamental limitation as it encodes too much information regarding the recording source in the learned representations. This article introduces Self-Supervised Positive Sampling (SSPS), a bootstrapped technique for sampling appropriate and diverse positives in SSL frameworks for SV. SSPS samples positives close to their anchor in the representation space, under the assumption that these pseudo-positives belong to the same speaker identity but correspond to different recording conditions. This method demonstrates consistent improvements in SV performance on VoxCeleb benchmarks when implemented in major SSL frameworks, such as SimCLR, SwAV, VICReg, and DINO. Using SSPS, SimCLR, and DINO achieve 2.57% and 2.53% EER on VoxCeleb1-O. SimCLR yields a 58% relative reduction in EER, getting comparable performance to DINO with a simpler training framework. Furthermore, SSPS lowers intra-class variance and reduces channel information in speaker representations while exhibiting greater robustness without data-augmentation.
>
---
#### [replaced 004] Frozen Large Language Models Can Perceive Paralinguistic Aspects of Speech
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.01162v2](http://arxiv.org/pdf/2410.01162v2)**

> **作者:** Wonjune Kang; Junteng Jia; Chunyang Wu; Wei Zhou; Egor Lakomkin; Yashesh Gaur; Leda Sari; Suyoun Kim; Ke Li; Jay Mahadeokar; Ozlem Kalinli
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This work studies the capabilities of a large language model (LLM) to understand paralinguistic aspects of speech without fine-tuning its weights. We utilize an end-to-end system with a speech encoder, which is trained to produce token embeddings such that the LLM's response to an expressive speech prompt is aligned with its response to a semantically matching text prompt that has also been conditioned on the user's speaking style. This framework enables the encoder to generate tokens that capture both linguistic and paralinguistic information and effectively convey them to the LLM, even when the LLM's weights remain completely frozen. To the best of our knowledge, our work is the first to explore how to induce a frozen LLM to understand more than just linguistic content from speech inputs in a general interaction setting. Experiments demonstrate that our system is able to produce higher quality and more empathetic responses to expressive speech prompts compared to several baselines.
>
---
#### [replaced 005] aTENNuate: Optimized Real-time Speech Enhancement with Deep SSMs on Raw Audio
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.03377v4](http://arxiv.org/pdf/2409.03377v4)**

> **作者:** Yan Ru Pei; Ritik Shrivastava; FNU Sidharth
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We present aTENNuate, a simple deep state-space autoencoder configured for efficient online raw speech enhancement in an end-to-end fashion. The network's performance is primarily evaluated on raw speech denoising, with additional assessments on tasks such as super-resolution and de-quantization. We benchmark aTENNuate on the VoiceBank + DEMAND and the Microsoft DNS1 synthetic test sets. The network outperforms previous real-time denoising models in terms of PESQ score, parameter count, MACs, and latency. Even as a raw waveform processing model, the model maintains high fidelity to the clean signal with minimal audible artifacts. In addition, the model remains performant even when the noisy input is compressed down to 4000Hz and 4 bits, suggesting general speech enhancement capabilities in low-resource environments. Try it out by pip install attenuate
>
---
#### [replaced 006] F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.06885v3](http://arxiv.org/pdf/2410.06885v3)**

> **作者:** Yushen Chen; Zhikang Niu; Ziyang Ma; Keqi Deng; Chunhui Wang; Jian Zhao; Kai Yu; Xie Chen
>
> **备注:** 17 pages, 9 tables, 3 figures
>
> **摘要:** This paper introduces F5-TTS, a fully non-autoregressive text-to-speech system based on flow matching with Diffusion Transformer (DiT). Without requiring complex designs such as duration model, text encoder, and phoneme alignment, the text input is simply padded with filler tokens to the same length as input speech, and then the denoising is performed for speech generation, which was originally proved feasible by E2 TTS. However, the original design of E2 TTS makes it hard to follow due to its slow convergence and low robustness. To address these issues, we first model the input with ConvNeXt to refine the text representation, making it easy to align with the speech. We further propose an inference-time Sway Sampling strategy, which significantly improves our model's performance and efficiency. This sampling strategy for flow step can be easily applied to existing flow matching based models without retraining. Our design allows faster training and achieves an inference RTF of 0.15, which is greatly improved compared to state-of-the-art diffusion-based TTS models. Trained on a public 100K hours multilingual dataset, our F5-TTS exhibits highly natural and expressive zero-shot ability, seamless code-switching capability, and speed control efficiency. We have released all codes and checkpoints to promote community development, at https://SWivid.github.io/F5-TTS/.
>
---
#### [replaced 007] xLSTM-SENet: xLSTM for Single-Channel Speech Enhancement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.06146v2](http://arxiv.org/pdf/2501.06146v2)**

> **作者:** Nikolai Lund Kühne; Jan Østergaard; Jesper Jensen; Zheng-Hua Tan
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** While attention-based architectures, such as Conformers, excel in speech enhancement, they face challenges such as scalability with respect to input sequence length. In contrast, the recently proposed Extended Long Short-Term Memory (xLSTM) architecture offers linear scalability. However, xLSTM-based models remain unexplored for speech enhancement. This paper introduces xLSTM-SENet, the first xLSTM-based single-channel speech enhancement system. A comparative analysis reveals that xLSTM-and notably, even LSTM-can match or outperform state-of-the-art Mamba- and Conformer-based systems across various model sizes in speech enhancement on the VoiceBank+Demand dataset. Through ablation studies, we identify key architectural design choices such as exponential gating and bidirectionality contributing to its effectiveness. Our best xLSTM-based model, xLSTM-SENet2, outperforms state-of-the-art Mamba- and Conformer-based systems of similar complexity on the Voicebank+DEMAND dataset.
>
---
#### [replaced 008] Fast Text-to-Audio Generation with Adversarial Post-Training
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.08175v3](http://arxiv.org/pdf/2505.08175v3)**

> **作者:** Zachary Novack; Zach Evans; Zack Zukowski; Josiah Taylor; CJ Carr; Julian Parker; Adnan Al-Sinan; Gian Marco Iodice; Julian McAuley; Taylor Berg-Kirkpatrick; Jordi Pons
>
> **摘要:** Text-to-audio systems, while increasingly performant, are slow at inference time, thus making their latency unpractical for many creative applications. We present Adversarial Relativistic-Contrastive (ARC) post-training, the first adversarial acceleration algorithm for diffusion/flow models not based on distillation. While past adversarial post-training methods have struggled to compare against their expensive distillation counterparts, ARC post-training is a simple procedure that (1) extends a recent relativistic adversarial formulation to diffusion/flow post-training and (2) combines it with a novel contrastive discriminator objective to encourage better prompt adherence. We pair ARC post-training with a number optimizations to Stable Audio Open and build a model capable of generating $\approx$12s of 44.1kHz stereo audio in $\approx$75ms on an H100, and $\approx$7s on a mobile edge-device, the fastest text-to-audio model to our knowledge.
>
---
#### [replaced 009] Unified Microphone Conversion: Many-to-Many Device Mapping via Feature-wise Linear Modulation
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.18322v2](http://arxiv.org/pdf/2410.18322v2)**

> **作者:** Myeonghoon Ryu; Hongseok Oh; Suji Lee; Han Park
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We present Unified Microphone Conversion, a unified generative framework designed to bolster sound event classification (SEC) systems against device variability. While our prior CycleGAN-based methods effectively simulate device characteristics, they require separate models for each device pair, limiting scalability. Our approach overcomes this constraint by conditioning the generator on frequency response data, enabling many-to-many device mappings through unpaired training. We integrate frequency-response information via Feature-wise Linear Modulation, further enhancing scalability. Additionally, incorporating synthetic frequency response differences improves the applicability of our framework for real-world application. Experimental results show that our method outperforms the state-of-the-art by 2.6% and reduces variability by 0.8% in macro-average F1 score.
>
---
#### [replaced 010] TF-Mamba: A Time-Frequency Network for Sound Source Localization
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.05034v2](http://arxiv.org/pdf/2409.05034v2)**

> **作者:** Yang Xiao; Rohan Kumar Das
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Sound source localization (SSL) determines the position of sound sources using multi-channel audio data. It is commonly used to improve speech enhancement and separation. Extracting spatial features is crucial for SSL, especially in challenging acoustic environments. Recently, a novel structure referred to as Mamba demonstrated notable performance across various sequence-based modalities. This study introduces the Mamba for SSL tasks. We consider the Mamba-based model to analyze spatial features from speech signals by fusing both time and frequency features, and we develop an SSL system called TF-Mamba. This system integrates time and frequency fusion, with Bidirectional Mamba managing both time-wise and frequency-wise processing. We conduct the experiments on the simulated and real datasets. Experiments show that TF-Mamba significantly outperforms other advanced methods. The code will be publicly released in due course.
>
---
#### [replaced 011] DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.12623v2](http://arxiv.org/pdf/2502.12623v2)**

> **作者:** Zhuoyuan Mao; Mengjie Zhao; Qiyu Wu; Hiromi Wakaki; Yuki Mitsufuji
>
> **摘要:** Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-LLM fusion Transformer to enhance modality fusion prior to input into text LLMs, tailoring DeepResonance for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We plan to open-source the models and the newly constructed datasets.
>
---
