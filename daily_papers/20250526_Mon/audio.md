# 音频 cs.SD;  eess.SP

- **最新发布 27 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Understanding the Algorithm Behind Audio Key Detection
- **分类: cs.SD; eess.AS; 94A12, 00A69, 68T10; H.5.5; I.5.4**

- **简介: 该论文属于音乐信息检索（MIR）领域，聚焦自动音调检测任务。旨在通过数字信号处理分析音频的音调特征，并与理论音调模板对比，开发算法以自动识别音频的音乐音调，解决旋律与和弦进行的和声语境自动化判定问题。**

- **链接: [http://arxiv.org/pdf/2505.17259v1](http://arxiv.org/pdf/2505.17259v1)**

> **作者:** Henrique Perez G. Silva
>
> **备注:** Preprint. Describes an algorithmic approach to musical key detection implemented in Python. Includes conceptual explanation of audio feature extraction and key profile matching
>
> **摘要:** The determination of musical key is a fundamental aspect of music theory and perception, providing a harmonic context for melodies and chord progressions. Automating this process, known as automatic key detection, is a significant task in the field of Music Information Retrieval (MIR). This article outlines an algorithmic methodology for estimating the musical key of an audio recording by analyzing its tonal content through digital signal processing techniques and comparison with theoretical key profiles.
>
---
#### [new 002] LLM-based Generative Error Correction for Rare Words with Synthetic Data and Phonetic Context
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于ASR后处理的生成式错误纠正任务，针对罕见词纠正效果差及过度修正问题，提出合成数据增强罕见词训练数据，并整合ASR的N-best假设与语音上下文以减少过度修正，提升多语言纠错效果。**

- **链接: [http://arxiv.org/pdf/2505.17410v1](http://arxiv.org/pdf/2505.17410v1)**

> **作者:** Natsuo Yamashita; Masaaki Yamamoto; Hiroaki Kokubo; Yohei Kawaguchi
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Generative error correction (GER) with large language models (LLMs) has emerged as an effective post-processing approach to improve automatic speech recognition (ASR) performance. However, it often struggles with rare or domain-specific words due to limited training data. Furthermore, existing LLM-based GER approaches primarily rely on textual information, neglecting phonetic cues, which leads to over-correction. To address these issues, we propose a novel LLM-based GER approach that targets rare words and incorporates phonetic information. First, we generate synthetic data to contain rare words for fine-tuning the GER model. Second, we integrate ASR's N-best hypotheses along with phonetic context to mitigate over-correction. Experimental results show that our method not only improves the correction of rare words but also reduces the WER and CER across both English and Japanese datasets.
>
---
#### [new 003] CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于多语言语音合成任务，旨在解决前作CosyVoice 2在语言覆盖、数据规模及生成质量上的局限。通过引入语音分词器、可微奖励模型，扩展数据量至百万小时及模型参数至15亿，提升零样本多语言合成的韵律自然度和内容一致性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.17589v1](http://arxiv.org/pdf/2505.17589v1)**

> **作者:** Zhihao Du; Changfeng Gao; Yuxuan Wang; Fan Yu; Tianyu Zhao; Hao Wang; Xiang Lv; Hui Wang; Xian Shi; Keyu An; Guanrou Yang; Yabin Li; Yanni Chen; Zhifu Gao; Qian Chen; Yue Gu; Mengzhe Chen; Yafeng Chen; Shiliang Zhang; Wen Wang; Jieping Ye
>
> **备注:** Preprint, work in progress
>
> **摘要:** In our prior works, we introduced a scalable streaming speech synthesis model, CosyVoice 2, which integrates a large language model (LLM) and a chunk-aware flow matching (FM) model, and achieves low-latency bi-streaming speech synthesis and human-parity quality. Despite these advancements, CosyVoice 2 exhibits limitations in language coverage, domain diversity, data volume, text formats, and post-training techniques. In this paper, we present CosyVoice 3, an improved model designed for zero-shot multilingual speech synthesis in the wild, surpassing its predecessor in content consistency, speaker similarity, and prosody naturalness. Key features of CosyVoice 3 include: 1) A novel speech tokenizer to improve prosody naturalness, developed via supervised multi-task training, including automatic speech recognition, speech emotion recognition, language identification, audio event detection, and speaker analysis. 2) A new differentiable reward model for post-training applicable not only to CosyVoice 3 but also to other LLM-based speech synthesis models. 3) Dataset Size Scaling: Training data is expanded from ten thousand hours to one million hours, encompassing 9 languages and 18 Chinese dialects across various domains and text formats. 4) Model Size Scaling: Model parameters are increased from 0.5 billion to 1.5 billion, resulting in enhanced performance on our multilingual benchmark due to the larger model capacity. These advancements contribute significantly to the progress of speech synthesis in the wild. We encourage readers to listen to the demo at https://funaudiollm.github.io/cosyvoice3.
>
---
#### [new 004] ReMi: A Random Recurrent Neural Network Approach to Music Production
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出ReMi方法，属于音乐生成任务。旨在解决生成AI的能耗高、版权争议及 creativity下降问题。通过随机初始化的循环神经网络生成琶音和低频振荡，无需训练数据且计算效率高，辅助音乐人扩展创造力而非替代其创作。**

- **链接: [http://arxiv.org/pdf/2505.17023v1](http://arxiv.org/pdf/2505.17023v1)**

> **作者:** Hugo Chateau-Laurent; Tara Vanhatalo
>
> **备注:** Accepted for an Innovation Showcase Demo at International Computer Music Conference
>
> **摘要:** Generative artificial intelligence raises concerns related to energy consumption, copyright infringement and creative atrophy. We show that randomly initialized recurrent neural networks can produce arpeggios and low-frequency oscillations that are rich and configurable. In contrast to end-to-end music generation that aims to replace musicians, our approach expands their creativity while requiring no data and much less computational power. More information can be found at: https://allendia.com/
>
---
#### [new 005] UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出UniTTS和DistilCodec，解决现有LLM-TTS系统因声学/语义信息分离导致的音频信息利用不足问题。DistilCodec将多码本编码器精简为高利用率单码本，支持无标签音频训练；UniTTS整合三模态自回归任务，实现混合文本/语音输入，通过三阶段训练提升端到端TTS性能。**

- **链接: [http://arxiv.org/pdf/2505.17426v1](http://arxiv.org/pdf/2505.17426v1)**

> **作者:** Rui Wang; Qianguo Sun; Tianrong Chen; Zhiyun Zeng; Junlong Wu; Jiaxing Zhang
>
> **摘要:** The emergence of multi-codebook neutral audio codecs such as Residual Vector Quantization (RVQ) and Group Vector Quantization (GVQ) has significantly advanced Large-Language-Model (LLM) based Text-to-Speech (TTS) systems. These codecs are crucial in separating semantic and acoustic information while efficiently harnessing semantic priors. However, since semantic and acoustic information cannot be fully aligned, a significant drawback of these methods when applied to LLM-based TTS is that large language models may have limited access to comprehensive audio information. To address this limitation, we propose DistilCodec and UniTTS, which collectively offer the following advantages: 1) This method can distill a multi-codebook audio codec into a single-codebook audio codec with 32,768 codes while achieving a near 100\% utilization. 2) As DistilCodec does not employ a semantic alignment scheme, a large amount of high-quality unlabeled audio (such as audiobooks with sound effects, songs, etc.) can be incorporated during training, further expanding data diversity and broadening its applicability. 3) Leveraging the comprehensive audio information modeling of DistilCodec, we integrated three key tasks into UniTTS's pre-training framework: audio modality autoregression, text modality autoregression, and speech-text cross-modal autoregression. This allows UniTTS to accept interleaved text and speech/audio prompts while substantially preserving LLM's text capabilities. 4) UniTTS employs a three-stage training process: Pre-Training, Supervised Fine-Tuning (SFT), and Alignment. Source code and model checkpoints are publicly available at https://github.com/IDEA-Emdoor-Lab/UniTTS and https://github.com/IDEA-Emdoor-Lab/DistilCodec.
>
---
#### [new 006] MEGADance: Mixture-of-Experts Architecture for Genre-Aware 3D Dance Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出MEGADance解决音乐驱动3D舞蹈生成中舞蹈类型条件不足的问题。传统方法未充分利用音乐类型，导致动作同步与连续性差。MEGADance通过双阶段架构：高保真量化编码舞蹈动作，混合专家与Transformer协同将音乐映射至潜空间，提升舞蹈质量和类型可控性，实验显示效果最佳。**

- **链接: [http://arxiv.org/pdf/2505.17543v1](http://arxiv.org/pdf/2505.17543v1)**

> **作者:** Kaixing Yang; Xulong Tang; Ziqiao Peng; Yuxuan Hu; Jun He; Hongyan Liu
>
> **备注:** arXiv admin note: text overlap with arXiv:2505.14222
>
> **摘要:** Music-driven 3D dance generation has attracted increasing attention in recent years, with promising applications in choreography, virtual reality, and creative content creation. Previous research has generated promising realistic dance movement from audio signals. However, traditional methods underutilize genre conditioning, often treating it as auxiliary modifiers rather than core semantic drivers. This oversight compromises music-motion synchronization and disrupts dance genre continuity, particularly during complex rhythmic transitions, thereby leading to visually unsatisfactory effects. To address the challenge, we propose MEGADance, a novel architecture for music-driven 3D dance generation. By decoupling choreographic consistency into dance generality and genre specificity, MEGADance demonstrates significant dance quality and strong genre controllability. It consists of two stages: (1) High-Fidelity Dance Quantization Stage (HFDQ), which encodes dance motions into a latent representation by Finite Scalar Quantization (FSQ) and reconstructs them with kinematic-dynamic constraints, and (2) Genre-Aware Dance Generation Stage (GADG), which maps music into the latent representation by synergistic utilization of Mixture-of-Experts (MoE) mechanism with Mamba-Transformer hybrid backbone. Extensive experiments on the FineDance and AIST++ dataset demonstrate the state-of-the-art performance of MEGADance both qualitatively and quantitatively. Code will be released upon acceptance.
>
---
#### [new 007] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出DualTalk框架，解决3D虚拟对话中自然切换说话与倾听的问题。通过整合双角色动态行为，生成连贯互动，创建50小时多轮对话数据集，实验验证提升自然度和表现力。**

- **链接: [http://arxiv.org/pdf/2505.18096v1](http://arxiv.org/pdf/2505.18096v1)**

> **作者:** Ziqiao Peng; Yanbo Fan; Haoyu Wu; Xuan Wang; Hongyan Liu; Jun He; Zhaoxin Fan
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In face-to-face conversations, individuals need to switch between speaking and listening roles seamlessly. Existing 3D talking head generation models focus solely on speaking or listening, neglecting the natural dynamics of interactive conversation, which leads to unnatural interactions and awkward transitions. To address this issue, we propose a new task -- multi-round dual-speaker interaction for 3D talking head generation -- which requires models to handle and generate both speaking and listening behaviors in continuous conversation. To solve this task, we introduce DualTalk, a novel unified framework that integrates the dynamic behaviors of speakers and listeners to simulate realistic and coherent dialogue interactions. This framework not only synthesizes lifelike talking heads when speaking but also generates continuous and vivid non-verbal feedback when listening, effectively capturing the interplay between the roles. We also create a new dataset featuring 50 hours of multi-round conversations with over 1,000 characters, where participants continuously switch between speaking and listening roles. Extensive experiments demonstrate that our method significantly enhances the naturalness and expressiveness of 3D talking heads in dual-speaker conversations. We recommend watching the supplementary video: https://ziqiaopeng.github.io/dualtalk.
>
---
#### [new 008] Analyzing Mitigation Strategies for Catastrophic Forgetting in End-to-End Training of Spoken Language Models
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 论文研究端到端口语语言模型训练中的灾难性遗忘问题，评估模型合并、LoRA缩放折扣及经验回放策略，发现后者效果最佳，结合其他方法进一步提升，为更稳健的训练提供见解。**

- **链接: [http://arxiv.org/pdf/2505.17496v1](http://arxiv.org/pdf/2505.17496v1)**

> **作者:** Chi-Yuan Hsiao; Ke-Han Lu; Kai-Wei Chang; Chih-Kai Yang; Wei-Chih Chen; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** End-to-end training of Spoken Language Models (SLMs) commonly involves adapting pre-trained text-based Large Language Models (LLMs) to the speech modality through multi-stage training on diverse tasks such as ASR, TTS and spoken question answering (SQA). Although this multi-stage continual learning equips LLMs with both speech understanding and generation capabilities, the substantial differences in task and data distributions across stages can lead to catastrophic forgetting, where previously acquired knowledge is lost. This paper investigates catastrophic forgetting and evaluates three mitigation strategies-model merging, discounting the LoRA scaling factor, and experience replay to balance knowledge retention with new learning. Results show that experience replay is the most effective, with further gains achieved by combining it with other methods. These findings provide insights for developing more robust and efficient SLM training pipelines.
>
---
#### [new 009] Semantic-Aware Interpretable Multimodal Music Auto-Tagging
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐自动打标签任务，针对现有模型可解释性不足的问题，提出结合多模态特征（信号处理、深度学习等）的可解释框架：通过语义聚类及期望最大化算法为特征组分配权重，提升决策透明度，实现高性能且用户可理解的音乐标签系统。**

- **链接: [http://arxiv.org/pdf/2505.17233v1](http://arxiv.org/pdf/2505.17233v1)**

> **作者:** Andreas Patakis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **摘要:** Music auto-tagging is essential for organizing and discovering music in extensive digital libraries. While foundation models achieve exceptional performance in this domain, their outputs often lack interpretability, limiting trust and usability for researchers and end-users alike. In this work, we present an interpretable framework for music auto-tagging that leverages groups of musically meaningful multimodal features, derived from signal processing, deep learning, ontology engineering, and natural language processing. To enhance interpretability, we cluster features semantically and employ an expectation maximization algorithm, assigning distinct weights to each group based on its contribution to the tagging process. Our method achieves competitive tagging performance while offering a deeper understanding of the decision-making process, paving the way for more transparent and user-centric music tagging systems.
>
---
#### [new 010] Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; 68T10; I.2.7**

- **简介: 该论文属于语音处理任务，研究帧率对语音分词器的影响，旨在解决不同语言下优化帧率选择的问题。通过对比汉语和英语在不同帧率下的语音识别效果，分析帧率与音位密度、语言声学特征的关联，提出帧率选择的优化策略。（98字）**

- **链接: [http://arxiv.org/pdf/2505.17076v1](http://arxiv.org/pdf/2505.17076v1)**

> **作者:** Haoyang Zhang; Hexin Liu; Xiangyu Zhang; Qiquan Zhang; Yuchen Hu; Junqi Zhao; Fei Tian; Xuerui Yang; Eng Siong Chng
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
>
---
#### [new 011] Toward Optimal ANC: Establishing Mutual Information Lower Bound
- **分类: cs.IT; cs.AI; cs.LG; cs.SD; eess.AS; math.IT**

- **简介: 该论文属于主动降噪（ANC）优化任务，解决其理论极限缺失问题。提出基于信息熵与频率约束的NMSE下限模型，量化信息处理和物理限制对降噪性能的边界，并通过实验验证模型有效性。**

- **链接: [http://arxiv.org/pdf/2505.17877v1](http://arxiv.org/pdf/2505.17877v1)**

> **作者:** François Derrida; Shahar Lutati; Eliya Nachmani
>
> **摘要:** Active Noise Cancellation (ANC) algorithms aim to suppress unwanted acoustic disturbances by generating anti-noise signals that destructively interfere with the original noise in real time. Although recent deep learning-based ANC algorithms have set new performance benchmarks, there remains a shortage of theoretical limits to rigorously assess their improvements. To address this, we derive a unified lower bound on cancellation performance composed of two components. The first component is information-theoretic: it links residual error power to the fraction of disturbance entropy captured by the anti-noise signal, thereby quantifying limits imposed by information-processing capacity. The second component is support-based: it measures the irreducible error arising in frequency bands that the cancellation path cannot address, reflecting fundamental physical constraints. By taking the maximum of these two terms, our bound establishes a theoretical ceiling on the Normalized Mean Squared Error (NMSE) attainable by any ANC algorithm. We validate its tightness empirically on the NOISEX dataset under varying reverberation times, demonstrating robustness across diverse acoustic conditions.
>
---
#### [new 012] Swedish Whispers; Leveraging a Massive Speech Corpus for Swedish Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文聚焦瑞典语语音识别任务，针对其作为中等资源语言在多语种模型中的表现不足问题，通过微调Whisper模型并利用大规模多样化语料库训练，使最佳模型在多个基准测试中WER较OpenAI原版降低47%。**

- **链接: [http://arxiv.org/pdf/2505.17538v1](http://arxiv.org/pdf/2505.17538v1)**

> **作者:** Leonora Vesterbacka; Faton Rekathati; Robin Kurtz; Justyna Sikora; Agnes Toftgård
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This work presents a suite of fine-tuned Whisper models for Swedish, trained on a dataset of unprecedented size and variability for this mid-resourced language. As languages of smaller sizes are often underrepresented in multilingual training datasets, substantial improvements in performance can be achieved by fine-tuning existing multilingual models, as shown in this work. This work reports an overall improvement across model sizes compared to OpenAI's Whisper evaluated on Swedish. Most notably, we report an average 47% reduction in WER comparing our best performing model to OpenAI's whisper-large-v3, in evaluations across FLEURS, Common Voice, and NST.
>
---
#### [new 013] Benchmarking Expressive Japanese Character Text-to-Speech with VITS and Style-BERT-VITS2
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文基准测试了VITS和SBV2JE模型在日语角色语音合成任务中的表现，解决音调敏感与风格多样化的挑战。通过三个角色数据集评估自然度、可懂度及说话人一致性，结果显示SBV2JE在自然度（MOS 4.37近似人类水平）和WER上优于VITS，但计算需求更高。**

- **链接: [http://arxiv.org/pdf/2505.17320v1](http://arxiv.org/pdf/2505.17320v1)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **摘要:** Synthesizing expressive Japanese character speech poses unique challenges due to pitch-accent sensitivity and stylistic variability. This paper benchmarks two open-source text-to-speech models--VITS and Style-BERT-VITS2 JP Extra (SBV2JE)--on in-domain, character-driven Japanese speech. Using three character-specific datasets, we evaluate models across naturalness (mean opinion and comparative mean opinion score), intelligibility (word error rate), and speaker consistency. SBV2JE matches human ground truth in naturalness (MOS 4.37 vs. 4.38), achieves lower WER, and shows slight preference in CMOS. Enhanced by pitch-accent controls and a WavLM-based discriminator, SBV2JE proves effective for applications like language learning and character dialogue generation, despite higher computational demands.
>
---
#### [new 014] Private kNN-VC: Interpretable Anonymization of Converted Speech
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音匿名化任务，旨在解决语音转换后说话人身份泄露问题。针对kNN-VC模型因韵律信息泄漏导致匿名效果差的问题，提出通过添加可解释组件匿名化音素时长和变异，证实韵律特征被用于身份识别，并分析目标选择算法对隐私攻击的影响。**

- **链接: [http://arxiv.org/pdf/2505.17584v1](http://arxiv.org/pdf/2505.17584v1)**

> **作者:** Carlos Franzreb; Arnab Das; Tim Polzehl; Sebastian Möller
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Speaker anonymization seeks to conceal a speaker's identity while preserving the utility of their speech. The achieved privacy is commonly evaluated with a speaker recognition model trained on anonymized speech. Although this represents a strong attack, it is unclear which aspects of speech are exploited to identify the speakers. Our research sets out to unveil these aspects. It starts with kNN-VC, a powerful voice conversion model that performs poorly as an anonymization system, presumably because of prosody leakage. To test this hypothesis, we extend kNN-VC with two interpretable components that anonymize the duration and variation of phones. These components increase privacy significantly, proving that the studied prosodic factors encode speaker identity and are exploited by the privacy attack. Additionally, we show that changes in the target selection algorithm considerably influence the outcome of the privacy attack.
>
---
#### [new 015] Exploring the Effect of Segmentation and Vocabulary Size on Speech Tokenization for Speech Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究语音分词中分段宽度和词汇规模对语音语言模型（SLMs）的影响，旨在优化离散表示以提升模型性能。通过对比固定/可变分段及不同聚类规模，发现中等粗分段与较大集群能提升零样本语音理解效果，最佳模型减少50%训练数据与70%训练时间，强调多token融合的重要性。**

- **链接: [http://arxiv.org/pdf/2505.17446v1](http://arxiv.org/pdf/2505.17446v1)**

> **作者:** Shunsuke Kando; Yusuke Miyao; Shinnosuke Takamichi
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** The purpose of speech tokenization is to transform a speech signal into a sequence of discrete representations, serving as the foundation for speech language models (SLMs). While speech tokenization has many options, their effect on the performance of SLMs remains unclear. This paper investigates two key aspects of speech tokenization: the segmentation width and the cluster size of discrete units. First, we segment speech signals into fixed/variable widths and pooled representations. We then train K-means models in multiple cluster sizes. Through the evaluation on zero-shot spoken language understanding benchmarks, we find the positive effect of moderately coarse segmentation and bigger cluster size. Notably, among the best-performing models, the most efficient one achieves a 50% reduction in training data and a 70% decrease in training runtime. Our analysis highlights the importance of combining multiple tokens to enhance fine-grained spoken language understanding.
>
---
#### [new 016] Enhancing Fourier-based Doppler Resolution with Diffusion Models
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于雷达信号处理任务，旨在提升多普勒分辨率以更好检测慢速目标。针对传统FFT受硬件限制导致分辨率不足的问题，提出结合零填充FFT与扩散模型的生成式神经网络，优化雷达成像中的range-Doppler图，实现对近距离目标的有效分离。**

- **链接: [http://arxiv.org/pdf/2505.17567v1](http://arxiv.org/pdf/2505.17567v1)**

> **作者:** Denisa Qosja; Kilian Barth; Simon Wagner
>
> **备注:** Published at International Radar Symposium (IRS) 2025
>
> **摘要:** In radar systems, high resolution in the Doppler dimension is important for detecting slow-moving targets as it allows for more distinct separation between these targets and clutter, or stationary objects. However, achieving sufficient resolution is constrained by hardware capabilities and physical factors, leading to the development of processing techniques to enhance the resolution after acquisition. In this work, we leverage artificial intelligence to increase the Doppler resolution in range-Doppler maps. Based on a zero-padded FFT, a refinement via the generative neural networks of diffusion models is achieved. We demonstrate that our method overcomes the limitations of traditional FFT, generating data where closely spaced targets are effectively separated.
>
---
#### [new 017] Speechless: Speech Instruction Training Without Speech for Low Resource Languages
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出Speechless方法，针对低资源语言语音助手训练中缺乏语音指令数据及TTS模型的问题，通过在语义表示层合成数据并对其与Whisper编码器对齐，使大模型仅用文本指令微调即可处理语音指令，简化低资源语言语音系统开发。**

- **链接: [http://arxiv.org/pdf/2505.17417v1](http://arxiv.org/pdf/2505.17417v1)**

> **作者:** Alan Dao; Dinh Bach Vu; Huy Hoang Ha; Tuan Le Duc Anh; Shreyas Gopal; Yue Heng Yeo; Warren Keng Hoong Low; Eng Siong Chng; Jia Qi Yip
>
> **备注:** This paper was accepted by INTERSPEECH 2025
>
> **摘要:** The rapid growth of voice assistants powered by large language models (LLM) has highlighted a need for speech instruction data to train these systems. Despite the abundance of speech recognition data, there is a notable scarcity of speech instruction data, which is essential for fine-tuning models to understand and execute spoken commands. Generating high-quality synthetic speech requires a good text-to-speech (TTS) model, which may not be available to low resource languages. Our novel approach addresses this challenge by halting synthesis at the semantic representation level, bypassing the need for TTS. We achieve this by aligning synthetic semantic representations with the pre-trained Whisper encoder, enabling an LLM to be fine-tuned on text instructions while maintaining the ability to understand spoken instructions during inference. This simplified training process is a promising approach to building voice assistant for low-resource languages.
>
---
#### [new 018] What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection
- **分类: cs.LG; cs.CL; cs.SD; eess.AS; 53-04**

- **简介: 该论文研究深度伪造语音检测中的语言敏感性，通过文本级对抗攻击揭示现有系统（含商业产品）对语言变异的脆弱性。发现轻微语言扰动使检测准确率骤降（如某商业系统从100%降至32%），分析显示语言复杂度和音频嵌入相似度是关键因素，提出需结合语言特征设计更鲁棒的检测系统。**

- **链接: [http://arxiv.org/pdf/2505.17513v1](http://arxiv.org/pdf/2505.17513v1)**

> **作者:** Binh Nguyen; Shuji Shi; Ryan Ofman; Thai Le
>
> **备注:** 15 pages, 2 fogures
>
> **摘要:** Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.
>
---
#### [new 019] Audio-to-Audio Emotion Conversion With Pitch And Duration Style Transfer
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出A2A-ZEST框架，属于零样本音频情绪风格迁移任务。旨在解决如何在保留源语音内容和说话人特征的同时，将参考语音的情绪（含音高、时长特征）转换至目标语音的问题。方法通过分解语音为语义、说话人表征和情感嵌入，结合自监督训练的分析-合成模块实现风格转换，实验显示其效果优于先前方法且无需平行训练数据。**

- **链接: [http://arxiv.org/pdf/2505.17655v1](http://arxiv.org/pdf/2505.17655v1)**

> **作者:** Soumya Dutta; Avni Jain; Sriram Ganapathy
>
> **备注:** 11 pages, 9 figures, 5 tables
>
> **摘要:** Given a pair of source and reference speech recordings, audio-to-audio (A2A) style transfer involves the generation of an output speech that mimics the style characteristics of the reference while preserving the content and speaker attributes of the source. In this paper, we propose a novel framework, termed as A2A Zero-shot Emotion Style Transfer (A2A-ZEST), that enables the transfer of reference emotional attributes to the source while retaining its speaker and speech contents. The A2A-ZEST framework consists of an analysis-synthesis pipeline, where the analysis module decomposes speech into semantic tokens, speaker representations, and emotion embeddings. Using these representations, a pitch contour estimator and a duration predictor are learned. Further, a synthesis module is designed to generate speech based on the input representations and the derived factors. This entire paradigm of analysis-synthesis is trained purely in a self-supervised manner with an auto-encoding loss. For A2A emotion style transfer, the emotion embedding extracted from the reference speech along with the rest of the representations from the source speech are used in the synthesis module to generate the style translated speech. In our experiments, we evaluate the converted speech on content/speaker preservation (w.r.t. source) as well as on the effectiveness of the emotion style transfer (w.r.t. reference). The proposal, A2A-ZEST, is shown to improve over other prior works on these evaluations, thereby enabling style transfer without any parallel training data. We also illustrate the application of the proposed work for data augmentation in emotion recognition tasks.
>
---
#### [new 020] JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出JALMBench，首个评估音频语言模型(ALMs)抗越狱攻击能力的基准测试框架。针对ALMs安全性研究不足及缺乏统一评测标准的问题，构建含2.2万文本及5万音频样本的数据集，支持12种ALMs、8类攻击及5种防御方法，分析攻击效率与防御策略，填补音频模态安全评估空白。**

- **链接: [http://arxiv.org/pdf/2505.17568v1](http://arxiv.org/pdf/2505.17568v1)**

> **作者:** Zifan Peng; Yule Liu; Zhen Sun; Mingchen Li; Zeren Luo; Jingyi Zheng; Wenhan Dong; Xinlei He; Xuechao Wang; Yingjie Xue; Shengmin Xu; Xinyi Huang
>
> **摘要:** Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.
>
---
#### [new 021] Reverse-Speech-Finder: A Neural Network Backtracking Architecture for Generating Alzheimer's Disease Speech Samples and Improving Diagnosis Performance
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Reverse-Speech-Finder（RSF），一种神经网络回溯架构，用于生成阿尔茨海默病（AD）语音样本并提升诊断性能。针对真实AD语音数据稀缺及模型可解释性差问题，RSF通过识别AD相关神经元（MPNs）和回溯至语音标记（MPTs），发现新型语音特征，增强模型解释性与数据量。实验显示其准确率和F1值分别提升3.5%和3.2%，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.17477v1](http://arxiv.org/pdf/2505.17477v1)**

> **作者:** Victor OK Li; Yang Han; Jacqueline CK Lam; Lawrence YL Cheung
>
> **摘要:** This study introduces Reverse-Speech-Finder (RSF), a groundbreaking neural network backtracking architecture designed to enhance Alzheimer's Disease (AD) diagnosis through speech analysis. Leveraging the power of pre-trained large language models, RSF identifies and utilizes the most probable AD-specific speech markers, addressing both the scarcity of real AD speech samples and the challenge of limited interpretability in existing models. RSF's unique approach consists of three core innovations: Firstly, it exploits the observation that speech markers most probable of predicting AD, defined as the most probable speech-markers (MPMs), must have the highest probability of activating those neurons (in the neural network) with the highest probability of predicting AD, defined as the most probable neurons (MPNs). Secondly, it utilizes a speech token representation at the input layer, allowing backtracking from MPNs to identify the most probable speech-tokens (MPTs) of AD. Lastly, it develops an innovative backtracking method to track backwards from the MPNs to the input layer, identifying the MPTs and the corresponding MPMs, and ingeniously uncovering novel speech markers for AD detection. Experimental results demonstrate RSF's superiority over traditional methods such as SHAP and Integrated Gradients, achieving a 3.5% improvement in accuracy and a 3.2% boost in F1-score. By generating speech data that encapsulates novel markers, RSF not only mitigates the limitations of real data scarcity but also significantly enhances the robustness and accuracy of AD diagnostic models. These findings underscore RSF's potential as a transformative tool in speech-based AD detection, offering new insights into AD-related linguistic deficits and paving the way for more effective non-invasive early intervention strategies.
>
---
#### [new 022] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering
- **分类: cs.IR; cs.SD; eess.AS**

- **简介: 该论文提出VoxRAG，一种无需转录的语音问答检索增强生成系统，直接从语音查询中检索相关音频片段。任务为解决传统RAG系统依赖文本转录的问题，通过沉默感知分段、说话人分离、CLAP嵌入及FAISS检索实现语音到语音的直接交互。实验显示其检索与生成效果初步可行，但精度仍需提升。**

- **链接: [http://arxiv.org/pdf/2505.17326v1](http://arxiv.org/pdf/2505.17326v1)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **备注:** Accepted to ACL 2025 Workshop MAGMaR
>
> **摘要:** We introduce VoxRAG, a modular speech-to-speech retrieval-augmented generation system that bypasses transcription to retrieve semantically relevant audio segments directly from spoken queries. VoxRAG employs silence-aware segmentation, speaker diarization, CLAP audio embeddings, and FAISS retrieval using L2-normalized cosine similarity. We construct a 50-query test set recorded as spoken input by a native English speaker. Retrieval quality was evaluated using LLM-as-a-judge annotations. For very relevant segments, cosine similarity achieved a Recall@10 of 0.34. For somewhat relevant segments, Recall@10 rose to 0.60 and nDCG@10 to 0.27, highlighting strong topical alignment. Answer quality was judged on a 0--2 scale across relevance, accuracy, completeness, and precision, with mean scores of 0.84, 0.58, 0.56, and 0.46 respectively. While precision and retrieval quality remain key limitations, VoxRAG shows that transcription-free speech-to-speech retrieval is feasible in RAG systems.
>
---
#### [new 023] Effects of auditory distance cues and reverberation on spatial perception and listening strategies
- **分类: eess.AS; cs.SD**

- **简介: 该研究探讨生态效度下听觉距离与混响对空间感知的影响，旨在解决现有简化模型无法反映真实场景的问题。通过让被试在无回声/混响环境进行无指导的主动定位任务，发现混响环境中头部运动更频繁（适应性策略），而距离不影响策略但影响定位精度。**

- **链接: [http://arxiv.org/pdf/2505.18020v1](http://arxiv.org/pdf/2505.18020v1)**

> **作者:** Fulvio Missoni; Katarina Poole; Lorenzo Picinali; Andrea Canessa
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Spatial hearing, the brain's ability to use auditory cues to identify the origin of sounds, is crucial for everyday listening. While simplified paradigms have advanced the understanding of spatial hearing, their lack of ecological validity limits their applicability to real-life conditions. This study aims to address this gap by investigating the effects of listener movement, reverberation, and distance on localisation accuracy in a more ecologically valid context. Participants performed active localisation tasks with no specific instructions on listening strategy, in either anechoic or reverberant conditions. The results indicate that the head movements were more frequent in reverberant environments, suggesting an adaptive strategy to mitigate uncertainty in binaural cues due to reverberation. While distance did not affect the listening strategy, it influenced the localisation performance. Our outcomes suggest that listening behaviour is adapted depending on the current acoustic conditions to support an effective perception of the space.
>
---
#### [new 024] From Weak Labels to Strong Results: Utilizing 5,000 Hours of Noisy Classroom Transcripts with Minimal Accurate Data
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于课堂环境下的自动语音识别（ASR）任务，解决如何利用大量弱标注（噪声）文本与少量高质量数据提升低资源场景性能的问题。提出Weakly Supervised Pretraining（WSP）方法：先用弱数据预训练模型，再用精数据微调，实验显示其优于其他方法，有效应对低成本标注与稀缺优质数据的矛盾。**

- **链接: [http://arxiv.org/pdf/2505.17088v1](http://arxiv.org/pdf/2505.17088v1)**

> **作者:** Ahmed Adel Attia; Dorottya Demszky; Jing Liu; Carol Espy-Wilson
>
> **摘要:** Recent progress in speech recognition has relied on models trained on vast amounts of labeled data. However, classroom Automatic Speech Recognition (ASR) faces the real-world challenge of abundant weak transcripts paired with only a small amount of accurate, gold-standard data. In such low-resource settings, high transcription costs make re-transcription impractical. To address this, we ask: what is the best approach when abundant inexpensive weak transcripts coexist with limited gold-standard data, as is the case for classroom speech data? We propose Weakly Supervised Pretraining (WSP), a two-step process where models are first pretrained on weak transcripts in a supervised manner, and then fine-tuned on accurate data. Our results, based on both synthetic and real weak transcripts, show that WSP outperforms alternative methods, establishing it as an effective training methodology for low-resource ASR in real-world scenarios.
>
---
#### [new 025] Improving endpoint detection in end-to-end streaming ASR for conversational speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦端点检测（EP）在端到端流式ASR中的优化，旨在解决转导模型（T-ASR）因延迟输出导致的EP错误（如截断结果或延迟响应）。提出在词尾添加结束标记并引入延迟惩罚，结合辅助网络的帧级语音活动检测改进EP，实验基于Switchboard语料库验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.17070v1](http://arxiv.org/pdf/2505.17070v1)**

> **作者:** Anandh C; Karthik Pandia Durai; Jeena Prakash; Manickavela Arumugam; Kadri Hacioglu; S. Pavankumar Dubagunta; Andreas Stolcke; Shankar Venkatesan; Aravind Ganapathiraju
>
> **备注:** Submitted to Interspeech 2024
>
> **摘要:** ASR endpointing (EP) plays a major role in delivering a good user experience in products supporting human or artificial agents in human-human/machine conversations. Transducer-based ASR (T-ASR) is an end-to-end (E2E) ASR modelling technique preferred for streaming. A major limitation of T-ASR is delayed emission of ASR outputs, which could lead to errors or delays in EP. Inaccurate EP will cut the user off while speaking, returning incomplete transcript while delays in EP will increase the perceived latency, degrading the user experience. We propose methods to improve EP by addressing delayed emission along with EP mistakes. To address the delayed emission problem, we introduce an end-of-word token at the end of each word, along with a delay penalty. The EP delay is addressed by obtaining a reliable frame-level speech activity detection using an auxiliary network. We apply the proposed methods on Switchboard conversational speech corpus and evaluate it against a delay penalty method.
>
---
#### [new 026] Source Separation of Small Classical Ensembles: Challenges and Opportunities
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于音乐源分离（MSS）任务，旨在解决古典音乐合奏分离难题。针对数据稀缺、乐器歧义等问题，团队创建合成木管数据库弥补现有数据不足，采用ConvTasNet模型对比因果/非因果方法，在Bach10和URMP数据集验证，发现合成与真实数据存在显著差距，提出需提升合成数据质量或增加真实训练数据。**

- **链接: [http://arxiv.org/pdf/2505.17823v1](http://arxiv.org/pdf/2505.17823v1)**

> **作者:** Gerardo Roa-Dabike; Trevor J. Cox; Jon P. Barker; Michael A. Akeroyd; Scott Bannister; Bruno Fazenda; Jennifer Firth; Simone Graetzer; Alinka Greasley; Rebecca R. Vos; William M. Whitmer
>
> **备注:** 5 pages, 4 figures, 2 tables, submitted to WASSPA 2025
>
> **摘要:** Musical (MSS) source separation of western popular music using non-causal deep learning can be very effective. In contrast, MSS for classical music is an unsolved problem. Classical ensembles are harder to separate than popular music because of issues such as the inherent greater variation in the music; the sparsity of recordings with ground truth for supervised training; and greater ambiguity between instruments. The Cadenza project has been exploring MSS for classical music. This is being done so music can be remixed to improve listening experiences for people with hearing loss. To enable the work, a new database of synthesized woodwind ensembles was created to overcome instrumental imbalances in the EnsembleSet. For the MSS, a set of ConvTasNet models was used with each model being trained to extract a string or woodwind instrument. ConvTasNet was chosen because it enabled both causal and non-causal approaches to be tested. Non-causal approaches have dominated MSS work and are useful for recorded music, but for live music or processing on hearing aids, causal signal processing is needed. The MSS performance was evaluated on the two small datasets (Bach10 and URMP) of real instrument recordings where the ground-truth is available. The performances of the causal and non-causal systems were similar. Comparing the average Signal-to-Distortion (SDR) of the synthesized validation set (6.2 dB causal; 6.9 non-causal), to the real recorded evaluation set (0.3 dB causal, 0.4 dB non-causal), shows that mismatch between synthesized and recorded data is a problem. Future work needs to either gather more real recordings that can be used for training, or to improve the realism and diversity of the synthesized recordings to reduce the mismatch...
>
---
#### [new 027] Large Language Models Implicitly Learn to See and Hear Just By Reading
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于多模态理解任务，探索文本LLM是否能通过预训练直接处理图像/音频。研究发现仅用文本训练的自回归LLM可内在学习跨模态理解能力，输入图像块或音频波形即可输出分类结果，在FSD-50K、CIFAR-10等数据集验证，证明文本模型内部形成通用感知模块，无需针对新任务从头训练。**

- **链接: [http://arxiv.org/pdf/2505.17091v1](http://arxiv.org/pdf/2505.17091v1)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 4 tables. Under Review WASPAA 2025
>
> **摘要:** This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time.
>
---
## 更新

#### [replaced 001] Enhancing Low-Resource Language and Instruction Following Capabilities of Audio Language Models
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.10999v2](http://arxiv.org/pdf/2409.10999v2)**

> **作者:** Potsawee Manakul; Guangzhi Sun; Warit Sirichotedumrong; Kasima Tharnpipitchai; Kunat Pipatanakul
>
> **备注:** Interspeech 2025
>
> **摘要:** Audio language models process audio inputs using textual prompts for tasks like speech recognition and audio captioning. Although built on multilingual pre-trained components, most are trained primarily on English, limiting their usability for other languages. This paper evaluates audio language models on Thai, a low-resource language, and finds that they lack emergent cross-lingual abilities despite their multilingual foundations. To address this, we explore data mixtures that optimize audio language models for both a target language and English while integrating audio comprehension and speech instruction-following into a unified model. Our experiments provide insights into improving instruction-following in low-resource languages by balancing language-specific and multilingual training data. The proposed model, Typhoon-Audio, significantly outperforms existing open-source models and achieves performance comparable to state-of-the-art Gemini-1.5-Pro in both English and Thai.
>
---
#### [replaced 002] Impact of Microphone Array Mismatches to Learning-based Replay Speech Detection
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.07357v2](http://arxiv.org/pdf/2503.07357v2)**

> **作者:** Michael Neri; Tuomas Virtanen
>
> **备注:** Accepted for publication in EUSIPCO 2025
>
> **摘要:** In this work, we investigate the generalization of a multi-channel learning-based replay speech detector, which employs adaptive beamforming and detection, across different microphone arrays. In general, deep neural network-based microphone array processing techniques generalize poorly to unseen array types, i.e., showing a significant training-test mismatch of performance. We employ the ReMASC dataset to analyze performance degradation due to inter- and intra-device mismatches, assessing both single- and multi-channel configurations. Furthermore, we explore fine-tuning to mitigate the performance loss when transitioning to unseen microphone arrays. Our findings reveal that array mismatches significantly decrease detection accuracy, with intra-device generalization being more robust than inter-device. However, fine-tuning with as little as ten minutes of target data can effectively recover performance, providing insights for practical deployment of replay detection systems in heterogeneous automatic speaker verification environments.
>
---
#### [replaced 003] NBM: an Open Dataset for the Acoustic Monitoring of Nocturnal Migratory Birds in Europe
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.03633v4](http://arxiv.org/pdf/2412.03633v4)**

> **作者:** Louis Airale; Adrien Pajot; Juliette Linossier
>
> **摘要:** The persisting threats on migratory bird populations highlight the urgent need for effective monitoring techniques that could assist in their conservation. Among these, passive acoustic monitoring is an essential tool, particularly for nocturnal migratory species that are difficult to track otherwise. This work presents the Nocturnal Bird Migration (NBM) dataset, a collection of 13,359 annotated vocalizations from 117 species of the Western Palearctic. The dataset includes precise time and frequency annotations, gathered by dozens of bird enthusiasts across France, enabling novel downstream acoustic analysis. In particular, we prove the utility of this database by training an original two-stage deep object detection model tailored for the processing of audio data. While allowing the precise localization of bird calls in spectrograms, this model shows competitive accuracy on the 45 main species of the dataset with state-of-the-art systems trained on much larger audio collections. These results highlight the interest of fostering similar open-science initiatives to acquire costly but valuable fine-grained annotations of audio files. All data and code are made openly available.
>
---
#### [replaced 004] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.15957v2](http://arxiv.org/pdf/2505.15957v2)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** Project Website: https://github.com/ckyang1124/LALM-Evaluation-Survey
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [replaced 005] EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16691v2](http://arxiv.org/pdf/2505.16691v2)**

> **作者:** Advait Joglekar; Divyanshu Singh; Rooshil Rohit Bhatia; S. Umesh
>
> **摘要:** Voice Conversion research in recent times has increasingly focused on improving the zero-shot capabilities of existing methods. Despite remarkable advancements, current architectures still tend to struggle in zero-shot cross-lingual settings. They are also often unable to generalize for speakers of unseen languages and accents. In this paper, we adopt a simple yet effective approach that combines discrete speech representations from self-supervised models with a non-autoregressive Diffusion-Transformer based conditional flow matching speech decoder. We show that this architecture allows us to train a voice-conversion model in a purely textless, self-supervised fashion. Our technique works without requiring multiple encoders to disentangle speech features. Our model also manages to excel in zero-shot cross-lingual settings even for unseen languages. For Demo: https://ez-vc.github.io/EZ-VC-Demo/
>
---
#### [replaced 006] U-SAM: An audio language Model for Unified Speech, Audio, and Music Understanding
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.13880v2](http://arxiv.org/pdf/2505.13880v2)**

> **作者:** Ziqian Wang; Xianjun Xia; Xinfa Zhu; Lei Xie
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The text generation paradigm for audio tasks has opened new possibilities for unified audio understanding. However, existing models face significant challenges in achieving a comprehensive understanding across diverse audio types, such as speech, general audio events, and music. Furthermore, their exclusive reliance on cross-entropy loss for alignment often falls short, as it treats all tokens equally and fails to account for redundant audio features, leading to weaker cross-modal alignment. To deal with the above challenges, this paper introduces U-SAM, an advanced audio language model that integrates specialized encoders for speech, audio, and music with a pre-trained large language model (LLM). U-SAM employs a Mixture of Experts (MoE) projector for task-aware feature fusion, dynamically routing and integrating the domain-specific encoder outputs. Additionally, U-SAM incorporates a Semantic-Aware Contrastive Loss Module, which explicitly identifies redundant audio features under language supervision and rectifies their semantic and spectral representations to enhance cross-modal alignment. Extensive experiments demonstrate that U-SAM consistently outperforms both specialized models and existing audio language models across multiple benchmarks. Moreover, it exhibits emergent capabilities on unseen tasks, showcasing its generalization potential. Code is available (https://github.com/Honee-W/U-SAM/).
>
---
#### [replaced 007] SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.13237v2](http://arxiv.org/pdf/2505.13237v2)**

> **作者:** Chih-Kai Yang; Neo Ho; Yen-Ting Piao; Hung-yi Lee
>
> **备注:** Accepted to Interspeech 2025. Project page: https://github.com/ckyang1124/SAKURA
>
> **摘要:** Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research.
>
---
#### [replaced 008] CrossMuSim: A Cross-Modal Framework for Music Similarity Retrieval with LLM-Powered Text Description Sourcing and Mining
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.23128v2](http://arxiv.org/pdf/2503.23128v2)**

> **作者:** Tristan Tsoi; Jiajun Deng; Yaolong Ju; Benno Weck; Holger Kirchhoff; Simon Lui
>
> **备注:** Accepted by ICME2025
>
> **摘要:** Music similarity retrieval is fundamental for managing and exploring relevant content from large collections in streaming platforms. This paper presents a novel cross-modal contrastive learning framework that leverages the open-ended nature of text descriptions to guide music similarity modeling, addressing the limitations of traditional uni-modal approaches in capturing complex musical relationships. To overcome the scarcity of high-quality text-music paired data, this paper introduces a dual-source data acquisition approach combining online scraping and LLM-based prompting, where carefully designed prompts leverage LLMs' comprehensive music knowledge to generate contextually rich descriptions. Exten1sive experiments demonstrate that the proposed framework achieves significant performance improvements over existing benchmarks through objective metrics, subjective evaluations, and real-world A/B testing on the Huawei Music streaming platform.
>
---
#### [replaced 009] Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.07435v3](http://arxiv.org/pdf/2503.07435v3)**

> **作者:** Riccardo Mazzieri; Jacopo Pegoraro; Michele Rossi
>
> **摘要:** The adoption of Millimeter-Wave (mmWave) radar devices for human sensing, particularly gait recognition, has recently gathered significant attention due to their efficiency, resilience to environmental conditions, and privacy-preserving nature. In this work, we tackle the challenging problem of Open-set Gait Recognition (OSGR) from sparse mmWave radar point clouds. Unlike most existing research, which assumes a closed-set scenario, our work considers the more realistic open-set case, where unknown subjects might be present at inference time, and should be correctly recognized by the system. Point clouds are well-suited for edge computing applications with resource constraints, but are more significantly affected by noise and random fluctuations than other representations, like the more common micro-Doppler signature. This is the first work addressing open-set gait recognition with sparse point cloud data. To do so, we propose a novel neural network architecture that combines supervised classification with unsupervised reconstruction of the point clouds, creating a robust, rich, and highly regularized latent space of gait features. To detect unknown subjects at inference time, we introduce a probabilistic novelty detection algorithm that leverages the structured latent space and offers a tunable trade-off between inference speed and prediction accuracy. Along with this paper, we release mmGait10, an original human gait dataset featuring over five hours of measurements from ten subjects, under varied walking modalities. Extensive experimental results show that our solution attains F1-Score improvements by 24% over state-of-the-art methods, on average, and across multiple openness levels.
>
---
#### [replaced 010] Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.19759v2](http://arxiv.org/pdf/2502.19759v2)**

> **作者:** Heeseung Kim; Che Hyun Lee; Sangkwon Park; Jiheum Yeom; Nohil Park; Sangwon Yu; Sungroh Yoon
>
> **备注:** ACL 2025 Findings, Project Page: https://contextdialog.github.io/
>
> **摘要:** Recent advancements in multi-turn voice interaction models have improved user-model communication. However, while closed-source models effectively retain and recall past utterances, whether open-source models share this ability remains unexplored. To fill this gap, we systematically evaluate how well open-source interaction models utilize past utterances using ContextDialog, a benchmark we proposed for this purpose. Our findings show that speech-based models have more difficulty than text-based ones, especially when recalling information conveyed in speech, and even with retrieval-augmented generation, models still struggle with questions about past utterances. These insights highlight key limitations in open-source models and suggest ways to improve memory retention and retrieval robustness.
>
---
