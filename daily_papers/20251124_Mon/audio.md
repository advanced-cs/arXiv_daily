# 音频 cs.SD;  eess.AS

- **最新发布 12 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] The Artist is Present: Traces of Artists Resigind and Spawning in Text-to-Audio AI
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究文本到音频（TTA）系统中艺术家风格的可诱导性，旨在揭示模型如何隐式学习并再现特定艺术家的声学特征。通过元标签提示工程，实证发现可精准生成接近如Bon Iver等艺术家的音色，证明艺术家作品被用作训练基础。研究提出可复现的审计方法，揭示版权、授权与伦理问题，推动对AI生成内容治理的反思。**

- **链接: [https://arxiv.org/pdf/2511.17404v1](https://arxiv.org/pdf/2511.17404v1)**

> **作者:** Guilherme Coelho
>
> **摘要:** Text-to-audio (TTA) systems are rapidly transforming music creation and distribution, with platforms like Udio and Suno generating thousands of tracks daily and integrating into mainstream music platforms and ecosystems. These systems, trained on vast and largely undisclosed datasets, are fundamentally reshaping how music is produced, reproduced and consumed. This paper presents empirical evidence that artist-conditioned regions can be systematically microlocated through metatag-based prompt design, effectively enabling the spawning of artist-like content through strategic prompt engineering. Through systematic exploration of metatag-based prompt engineering techniques this research reveals how users can access the distinctive sonic signatures of specific artists, evidencing their inclusion in training datasets. Using descriptor constellations drawn from public music taxonomies, the paper demonstrates reproducible proximity to artists such as Bon Iver, Philip Glass, Panda Bear and William Basinski. The results indicate stable text-audio correspondences consistent with artist-specific training signals, enabling precise traversal of stylistic microlocations without explicitly naming artists. This capacity to summon artist-specific outputs shows that artists' creative works fuction as foundational material from which these systems generate new content, often without explicit consent or attribuition. Conceptually, the work clarifies how textual descriptors act as navigational cues in high-dimensional representation spaces; methodologically, it provides a replicable protocol for auditing stylistic inducibility. The findings raise immediate queestions for governance-attribution, consent and disclosure standards-and for creative practice, where induced stylistic proximity complicates boundaries between ownership, reproduction, imitation, creative agency and the ethics of algorithmic creation.
>
---
#### [new 002] MusicAIR: A Multimodal AI Music Generation Framework Powered by an Algorithm-Driven Core
- **分类: cs.SD; cs.AI; cs.CL; cs.MM**

- **简介: 该论文提出MusicAIR框架，解决生成式AI音乐因依赖大数据带来的版权与成本问题。基于算法驱动的符号化音乐核心，实现从歌词、文本或图像自动生成符合音乐理论的乐谱，支持多模态输入。实验表明其生成作品在调性一致性上优于人类作曲家，具备教育与创作辅助价值。**

- **链接: [https://arxiv.org/pdf/2511.17323v1](https://arxiv.org/pdf/2511.17323v1)**

> **作者:** Callie C. Liao; Duoduo Liao; Ellie L. Zhang
>
> **备注:** Accepted by IEEE Big Data 2025
>
> **摘要:** Recent advances in generative AI have made music generation a prominent research focus. However, many neural-based models rely on large datasets, raising concerns about copyright infringement and high-performance costs. In contrast, we propose MusicAIR, an innovative multimodal AI music generation framework powered by a novel algorithm-driven symbolic music core, effectively mitigating copyright infringement risks. The music core algorithms connect critical lyrical and rhythmic information to automatically derive musical features, creating a complete, coherent melodic score solely from the lyrics. The MusicAIR framework facilitates music generation from lyrics, text, and images. The generated score adheres to established principles of music theory, lyrical structure, and rhythmic conventions. We developed Generate AI Music (GenAIM), a web tool using MusicAIR for lyric-to-song, text-to-music, and image-to-music generation. In our experiments, we evaluated AI-generated music scores produced by the system using both standard music metrics and innovative analysis that compares these compositions with original works. The system achieves an average key confidence of 85%, outperforming human composers at 79%, and aligns closely with established music theory standards, demonstrating its ability to generate diverse, human-like compositions. As a co-pilot tool, GenAIM can serve as a reliable music composition assistant and a possible educational composition tutor while simultaneously lowering the entry barrier for all aspiring musicians, which is innovative and significantly contributes to AI for music generation.
>
---
#### [new 003] Semantic and Semiotic Interplays in Text-to-Audio AI: Exploring Cognitive Dynamics and Musical Interactions
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究文本到音频生成的AI模型，探讨其在音乐创作与认知中的作用。针对语言提示如何转化为复杂声学对象的问题，结合语义学与认知理论，分析了AI在音乐符号化过程中的动态机制，揭示其作为“准对象”的双重功能，并以Udio为例，提出其促进结构性聆听与美学反思的潜力。**

- **链接: [https://arxiv.org/pdf/2511.17429v1](https://arxiv.org/pdf/2511.17429v1)**

> **作者:** Guilherme Coelho
>
> **摘要:** This paper investigates the emerging text-to-audio paradigm in artificial intelligence (AI), examining its transformative implications for musical creation, interpretation, and cognition. I explore the complex semantic and semiotic interplays that occur when descriptive natural language prompts are translated into nuanced sound objects across the text-to-audio modality. Drawing from structuralist and post-structuralist perspectives, as well as cognitive theories of schema dynamics and metacognition, the paper explores how these AI systems reconfigure musical signification processes and navigate established cognitive frameworks. The research analyzes some of the cognitive dynamics at play in AI-mediated musicking, including processes of schema assimilation and accommodation, metacognitive reflection, and constructive perception. The paper argues that text-to-audio AI models function as quasi-objects of musical signification, simultaneously stabilizing and destabilizing conventional forms while fostering new modes of listening and aesthetic reflexivity.Using Udio as a primary case study, this study explores how these models navigate the liminal spaces between linguistic prompts and sonic outputs. This process not only generates novel musical expressions but also prompts listeners to engage in forms of critical and "structurally-aware listening.", encouraging a deeper understanding of music's structures, semiotic nuances, and the socio-cultural contexts that shape our musical cognition. The paper concludes by reflecting on the potential of text-to-audio AI models to serve as epistemic tools and quasi-objects, facilitating a significant shift in musical interactions and inviting users to develop a more nuanced comprehension of the cognitive and cultural foundations of music.
>
---
#### [new 004] Enhancing Quranic Learning: A Multimodal Deep Learning Approach for Arabic Phoneme Recognition
- **分类: cs.SD; cs.AI**

- **简介: 该论文针对阿拉伯语诵读中音素误读检测难题，提出基于Transformer的多模态深度学习框架，融合UniSpeech声学嵌入与BERT文本嵌入，通过早期、中期、晚期融合策略提升识别精度。实验验证了其在29个音素上的有效性，推动了智能语音辅助《古兰经》学习系统的发展。**

- **链接: [https://arxiv.org/pdf/2511.17477v1](https://arxiv.org/pdf/2511.17477v1)**

> **作者:** Ayhan Kucukmanisa; Derya Gelmez; Sukru Selim Calik; Zeynep Hilal Kilimci
>
> **备注:** 11 pages, 2 figures, 3 tables
>
> **摘要:** Recent advances in multimodal deep learning have greatly enhanced the capability of systems for speech analysis and pronunciation assessment. Accurate pronunciation detection remains a key challenge in Arabic, particularly in the context of Quranic recitation, where subtle phonetic differences can alter meaning. Addressing this challenge, the present study proposes a transformer-based multimodal framework for Arabic phoneme mispronunciation detection that combines acoustic and textual representations to achieve higher precision and robustness. The framework integrates UniSpeech-derived acoustic embeddings with BERT-based textual embeddings extracted from Whisper transcriptions, creating a unified representation that captures both phonetic detail and linguistic context. To determine the most effective integration strategy, early, intermediate, and late fusion methods were implemented and evaluated on two datasets containing 29 Arabic phonemes, including eight hafiz sounds, articulated by 11 native speakers. Additional speech samples collected from publicly available YouTube recordings were incorporated to enhance data diversity and generalization. Model performance was assessed using standard evaluation metrics: accuracy, precision, recall, and F1-score, allowing a detailed comparison of the fusion strategies. Experimental findings show that the UniSpeech-BERT multimodal configuration provides strong results and that fusion-based transformer architectures are effective for phoneme-level mispronunciation detection. The study contributes to the development of intelligent, speaker-independent, and multimodal Computer-Aided Language Learning (CALL) systems, offering a practical step toward technology-supported Quranic pronunciation training and broader speech-based educational applications.
>
---
#### [new 005] Is Phase Really Needed for Weakly-Supervised Dereverberation ?
- **分类: cs.SD; cs.AI; eess.SP; physics.class-ph; stat.ML**

- **简介: 该论文研究弱监督语音去混响任务，针对训练时无法获取干净语音的问题，探究混响信号相位信息的必要性。基于统计波场理论，证明晚期混响使相位呈随机噪声分布，几乎无用。实验表明，去除相位损失可显著提升模型性能，证实相位非必需。**

- **链接: [https://arxiv.org/pdf/2511.17346v1](https://arxiv.org/pdf/2511.17346v1)**

> **作者:** Marius Rodrigues; Louis Bahrman; Roland Badeau; Gaël Richard
>
> **摘要:** In unsupervised or weakly-supervised approaches for speech dereverberation, the target clean (dry) signals are considered to be unknown during training. In that context, evaluating to what extent information can be retrieved from the sole knowledge of reverberant (wet) speech becomes critical. This work investigates the role of the reverberant (wet) phase in the time-frequency domain. Based on Statistical Wave Field Theory, we show that late reverberation perturbs phase components with white, uniformly distributed noise, except at low frequencies. Consequently, the wet phase carries limited useful information and is not essential for weakly supervised dereverberation. To validate this finding, we train dereverberation models under a recent weak supervision framework and demonstrate that performance can be significantly improved by excluding the reverberant phase from the loss function.
>
---
#### [new 006] Device-Guided Music Transfer
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出DeMT，解决未见过设备音质差异导致的音乐播放不适配问题。通过将扬声器频率响应曲线转为图像，利用视觉语言模型提取设备特征，再以特征调制方式实现音乐风格迁移，支持少样本适配，提升跨设备播放质量。**

- **链接: [https://arxiv.org/pdf/2511.17136v1](https://arxiv.org/pdf/2511.17136v1)**

> **作者:** Manh Pham Hung; Changshuo Hu; Ting Dang; Dong Ma
>
> **摘要:** Device-guided music transfer adapts playback across unseen devices for users who lack them. Existing methods mainly focus on modifying the timbre, rhythm, harmony, or instrumentation to mimic genres or artists, overlooking the diverse hardware properties of the playback device (i.e., speaker). Therefore, we propose DeMT, which processes a speaker's frequency response curve as a line graph using a vision-language model to extract device embeddings. These embeddings then condition a hybrid transformer via feature-wise linear modulation. Fine-tuned on a self-collected dataset, DeMT enables effective speaker-style transfer and robust few-shot adaptation for unseen devices, supporting applications like device-style augmentation and quality enhancement.
>
---
#### [new 007] AI in Music and Sound: Pedagogical Reflections, Post-Structuralist Approaches and Creative Outcomes in Seminar Practice
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于教育技术与艺术实践交叉研究，旨在探索AI在音乐教学中的应用。通过设计“配对练习”模式，引导学生反思AI工具的表征局限与创造性潜能，结合理论与实践，提升学生的媒介意识与批判性创作能力，提出可复用的教学设计模式。**

- **链接: [https://arxiv.org/pdf/2511.17425v1](https://arxiv.org/pdf/2511.17425v1)**

> **作者:** Guilherme Coelho
>
> **摘要:** This paper presents a pedagogical and conceptual account of the course AI in Music and Sound: Modalities, Tools and Creative Applications, offered within the Music Informatics and Media Art module of an M.Sc. in Audio Communication. The course engaged students with a range of AI modalities such as symbolic composition, voice synthesis, timbre transfer, neural audio synthesis, and text-to-audio systems, combining theoretical reflection with practice-based experimentation. Its central pedagogical move is a paired-études design: each modality is approached first through its intended affordances and then through a deliberately reframed or "misused" exercise that surfaces representational limits and alternative behaviours. Framed by medium theory and post-structuralist inquiry, we treated AI as a transmodal conduit-a system that translates and perturbs musical signs across textual, symbolic, timbral and audio domains. Evidence from student work and reflection indicates growth in technical fluency, medium awareness, and critical literacy, alongside the cultivation of experimental method and process-oriented listening. The paper outlines the course architecture, assessment design, and representative projects, and distils a set of design patterns for AI-music pedagogy (eg., prompt-conditioned interplays and semantic destabilisation in text-to-audio; latent space materialism in timbre transfer). It concludes with pedagogical recommendations that integrate creative practice with medium awareness and with cultural-epistemic analysis of AI technologies, preparing students to participate in how AI is understood, developed, and deployed with creative communities.
>
---
#### [new 008] Revisiting Audio-language Pretraining for Learning General-purpose Audio Representation
- **分类: eess.AS; cs.AI**

- **简介: 该论文研究音频-语言预训练，旨在构建通用音频表示。针对现有模型在大规模数据、多样标题及系统评估方面的不足，提出10.7M多源音频-文本数据集CaptionStew，对比了对比学习与标题生成目标的性能，揭示二者在不同规模下的互补优势，验证了其在语音、音乐和环境音任务中的有效性，推动通用音频理解发展。**

- **链接: [https://arxiv.org/pdf/2511.16757v1](https://arxiv.org/pdf/2511.16757v1)**

> **作者:** Wei-Cheng Tseng; Xuanru Zhou; Mingyue Huo; Yiwen Shao; Hao Zhang; Dong Yu
>
> **备注:** Work in progress
>
> **摘要:** Audio-language pretraining holds promise for general-purpose audio understanding, yet remains underexplored compared to its vision counterpart. While vision-language models like CLIP serve as widely adopted foundations, existing audio-language models primarily excel at retrieval tasks with limited adoption as general-purpose encoders. We identify three key barriers: limited large-scale audio-text corpora, insufficient caption diversity, and lack of systematic exploration and evaluation. To this end, we introduce CaptionStew, a 10.7M caption dataset aggregating diverse open-source audio-text corpora across multiple domains and captioning styles. Using this resource, we conduct the first comprehensive evaluation comparing contrastive and captioning objectives for audio representation learning across speech, music, and environmental sound tasks. Our results demonstrate that audio-language pretraining yields competitive, transferable representations. Through systematic data-scaling experiments, we reveal complementary objective strengths: contrastive learning achieves superior data efficiency at smaller scales, while captioning demonstrates better scalability on language-involved audio understanding tasks. We also find that common supervised initialization practices provide diminishing returns at scale, challenging current approaches. These findings establish audio-language pretraining as a viable pathway toward general-purpose audio representations, guiding future research. To accelerate progress, we release data preparation recipes, training protocols, and pretrained models, paving the way toward universal audio understanding.
>
---
#### [new 009] Investigating self-supervised representations for audio-visual deepfake detection
- **分类: cs.CV; cs.LG; cs.SD**

- **简介: 该论文研究自监督表示在音视频深度伪造检测中的应用。针对现有方法对自监督特征利用不足的问题，系统评估其在音频、视频及多模态下的检测效果、可解释性与跨模态互补性。结果表明，这些特征能捕捉深层伪造信息且具互补性，但泛化能力差，主要受数据集特性影响。**

- **链接: [https://arxiv.org/pdf/2511.17181v1](https://arxiv.org/pdf/2511.17181v1)**

> **作者:** Dragos-Alexandru Boldisor; Stefan Smeu; Dan Oneata; Elisabeta Oneata
>
> **摘要:** Self-supervised representations excel at many vision and speech tasks, but their potential for audio-visual deepfake detection remains underexplored. Unlike prior work that uses these features in isolation or buried within complex architectures, we systematically evaluate them across modalities (audio, video, multimodal) and domains (lip movements, generic visual content). We assess three key dimensions: detection effectiveness, interpretability of encoded information, and cross-modal complementarity. We find that most self-supervised features capture deepfake-relevant information, and that this information is complementary. Moreover, models primarily attend to semantically meaningful regions rather than spurious artifacts. Yet none generalize reliably across datasets. This generalization failure likely stems from dataset characteristics, not from the features themselves latching onto superficial patterns. These results expose both the promise and fundamental challenges of self-supervised representations for deepfake detection: while they learn meaningful patterns, achieving robust cross-domain performance remains elusive.
>
---
#### [new 010] Better audio representations are more brain-like: linking model-brain alignment with performance in downstream auditory tasks
- **分类: cs.LG; cs.SD**

- **简介: 该论文研究自监督音频模型的表征是否更接近大脑活动。通过对比36个模型在两个fMRI数据集上的脑-模型对齐度，发现性能越强的模型与大脑听觉皮层信号越相似，且这种相似性在预训练早期即显现，表明脑相似性是学习重建自然音频的副产品。**

- **链接: [https://arxiv.org/pdf/2511.16849v1](https://arxiv.org/pdf/2511.16849v1)**

> **作者:** Leonardo Pepino; Pablo Riera; Juan Kamienkowski; Luciana Ferrer
>
> **摘要:** Artificial neural networks (ANNs) are increasingly powerful models of brain computation, yet it remains unclear whether improving their task performance also makes their internal representations more similar to brain signals. To address this question in the auditory domain, we quantified the alignment between the internal representations of 36 different audio models and brain activity from two independent fMRI datasets. Using voxel-wise and component-wise regression, and representation similarity analysis (RSA), we found that recent self-supervised audio models with strong performance in diverse downstream tasks are better predictors of auditory cortex activity than older and more specialized models. To assess the quality of the audio representations, we evaluated these models in 6 auditory tasks from the HEAREval benchmark, spanning music, speech, and environmental sounds. This revealed strong positive Pearson correlations ($r>0.7$) between a model's overall task performance and its alignment with brain representations. Finally, we analyzed the evolution of the similarity between audio and brain representations during the pretraining of EnCodecMAE. We discovered that brain similarity increases progressively and emerges early during pretraining, despite the model not being explicitly optimized for this objective. This suggests that brain-like representations can be an emergent byproduct of learning to reconstruct missing information from naturalistic audio data.
>
---
#### [new 011] Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM
- **分类: cs.RO; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文聚焦人机协作中的动作确认与规划任务，针对现有方法忽略长视频上下文依赖、文本信息抽象过度的问题，提出融合左右上下文的长程Q-former与文本条件化机制，通过VideoLLaMA3提升多模态理解能力，显著改善动作确认与规划准确率。**

- **链接: [https://arxiv.org/pdf/2511.17335v1](https://arxiv.org/pdf/2511.17335v1)**

> **作者:** Chiori Hori; Yoshiki Masuyama; Siddarth Jain; Radu Corcodel; Devesh Jha; Diego Romeres; Jonathan Le Roux
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
>
---
#### [new 012] A new kid on the block: Distributional semantics predicts the word-specific tone signatures of monosyllabic words in conversational Taiwan Mandarin
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究普通话单音节词在口语中的声调实现，探究语义对音高的影响。通过广义加性模型分析语料，发现词汇语义是声调轮廓的重要预测因子，基于上下文词嵌入可准确预测具体发音，挑战传统声调理论，支持非对称性词库模型。**

- **链接: [https://arxiv.org/pdf/2511.17337v1](https://arxiv.org/pdf/2511.17337v1)**

> **作者:** Xiaoyun Jin; Mirjam Ernestus; R. Harald Baayen
>
> **备注:** arXiv admin note: text overlap with arXiv:2409.07891
>
> **摘要:** We present a corpus-based investigation of how the pitch contours of monosyllabic words are realized in spontaneous conversational Mandarin, focusing on the effects of words' meanings. We used the generalized additive model to decompose a given observed pitch contour into a set of component pitch contours that are tied to different control variables and semantic predictors. Even when variables such as word duration, gender, speaker identity, tonal context, vowel height, and utterance position are controlled for, the effect of word remains a strong predictor of tonal realization. We present evidence that this effect of word is a semantic effect: word sense is shown to be a better predictor than word, and heterographic homophones are shown to have different pitch contours. The strongest evidence for the importance of semantics is that the pitch contours of individual word tokens can be predicted from their contextualized embeddings with an accuracy that substantially exceeds a permutation baseline. For phonetics, distributional semantics is a new kid on the block. Although our findings challenge standard theories of Mandarin tone, they fit well within the theoretical framework of the Discriminative Lexicon Model.
>
---
## 更新

#### [replaced 001] A Differentiable Alignment Framework for Sequence-to-Sequence Modeling via Optimal Transport
- **分类: cs.LG; cs.SD; eess.AS; stat.ML**

- **链接: [https://arxiv.org/pdf/2502.01588v3](https://arxiv.org/pdf/2502.01588v3)**

> **作者:** Yacouba Kaloga; Shashi Kumar; Petr Motlicek; Ina Kodrasi
>
> **摘要:** Accurate sequence-to-sequence (seq2seq) alignment is critical for applications like medical speech analysis and language learning tools relying on automatic speech recognition (ASR). State-of-the-art end-to-end (E2E) ASR systems, such as the Connectionist Temporal Classification (CTC) and transducer-based models, suffer from peaky behavior and alignment inaccuracies. In this paper, we propose a novel differentiable alignment framework based on one-dimensional optimal transport, enabling the model to learn a single alignment and perform ASR in an E2E manner. We introduce a pseudo-metric, called Sequence Optimal Transport Distance (SOTD), over the sequence space and discuss its theoretical properties. Based on the SOTD, we propose Optimal Temporal Transport Classification (OTTC) loss for ASR and contrast its behavior with CTC. Experimental results on the TIMIT, AMI, and LibriSpeech datasets show that our method considerably improves alignment performance compared to CTC and the more recently proposed Consistency-Regularized CTC, though with a trade-off in ASR performance. We believe this work opens new avenues for seq2seq alignment research, providing a solid foundation for further exploration and development within the community. Our code is publicly available at: https://github.com/idiap/OTTC
>
---
#### [replaced 002] AV-Lip-Sync+: Leveraging AV-HuBERT to Exploit Multimodal Inconsistency for Deepfake Detection of Frontal Face Videos
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [https://arxiv.org/pdf/2311.02733v2](https://arxiv.org/pdf/2311.02733v2)**

> **作者:** Sahibzada Adil Shahzad; Ammarah Hashmi; Yan-Tsung Peng; Yu Tsao; Hsin-Min Wang
>
> **摘要:** Multimodal manipulations (also known as audio-visual deepfakes) make it difficult for unimodal deepfake detectors to detect forgeries in multimedia content. To avoid the spread of false propaganda and fake news, timely detection is crucial. The damage to either modality (i.e., visual or audio) can only be discovered through multimodal models that can exploit both pieces of information simultaneously. However, previous methods mainly adopt unimodal video forensics and use supervised pre-training for forgery detection. This study proposes a new method based on a multimodal self-supervised-learning (SSL) feature extractor to exploit inconsistency between audio and visual modalities for multimodal video forgery detection. We use the transformer-based SSL pre-trained Audio-Visual HuBERT (AV-HuBERT) model as a visual and acoustic feature extractor and a multi-scale temporal convolutional neural network to capture the temporal correlation between the audio and visual modalities. Since AV-HuBERT only extracts visual features from the lip region, we also adopt another transformer-based video model to exploit facial features and capture spatial and temporal artifacts caused during the deepfake generation process. Experimental results show that our model outperforms all existing models and achieves new state-of-the-art performance on the FakeAVCeleb and DeepfakeTIMIT datasets.
>
---
#### [replaced 003] Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?
- **分类: eess.AS; cs.SD**

- **链接: [https://arxiv.org/pdf/2505.09439v3](https://arxiv.org/pdf/2505.09439v3)**

> **作者:** Andrew Rouditchenko; Saurabhchand Bhati; Edson Araujo; Samuel Thomas; Hilde Kuehne; Rogerio Feris; James Glass
>
> **摘要:** We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
>
---
