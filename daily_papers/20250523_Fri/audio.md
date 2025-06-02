# 音频 cs.SD;  eess.SP

- **最新发布 20 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Source Separation by Flow Matching
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出FLOSS方法，解决单通道音频源分离的欠定问题。通过流匹配技术，利用源的联合分布与混合分布样本，添加人工噪声匹配维度，并采用等变神经网络处理源排列不变性，确保混合一致性，实现多语音分离。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16119v1](http://arxiv.org/pdf/2505.16119v1)**

> **作者:** Robin Scheibler; John R. Hershey; Arnaud Doucet; Henry Li
>
> **备注:** 5 pages, 3 figures, 2 tables
>
> **摘要:** We consider the problem of single-channel audio source separation with the goal of reconstructing $K$ sources from their mixture. We address this ill-posed problem with FLOSS (FLOw matching for Source Separation), a constrained generation method based on flow matching, ensuring strict mixture consistency. Flow matching is a general methodology that, when given samples from two probability distributions defined on the same space, learns an ordinary differential equation to output a sample from one of the distributions when provided with a sample from the other. In our context, we have access to samples from the joint distribution of $K$ sources and so the corresponding samples from the lower-dimensional distribution of their mixture. To apply flow matching, we augment these mixture samples with artificial noise components to ensure the resulting "augmented" distribution matches the dimensionality of the $K$ source distribution. Additionally, as any permutation of the sources yields the same mixture, we adopt an equivariant formulation of flow matching which relies on a suitable custom-designed neural network architecture. We demonstrate the performance of the method for the separation of overlapping speech.
>
---
#### [new 002] EZ-VC: Easy Zero-shot Any-to-Any Voice Conversion
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音转换任务，针对现有方法在零样本跨语言场景下对未见语言/口音泛化能力不足的问题，提出结合自监督离散语音表征与非自回归扩散-Transformer解码器的模型，实现无文本自监督训练且无需多编码器分离特征，在零样本跨语言转换中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.16691v1](http://arxiv.org/pdf/2505.16691v1)**

> **作者:** Advait Joglekar; Divyanshu Singh; Rooshil Rohit Bhatia; S. Umesh
>
> **备注:** Submitted to EMNLP 2025, 7 pages, 2 figures, 5 Tables
>
> **摘要:** Voice Conversion research in recent times has increasingly focused on improving the zero-shot capabilities of existing methods. Despite remarkable advancements, current architectures still tend to struggle in zero-shot cross-lingual settings. They are also often unable to generalize for speakers of unseen languages and accents. In this paper, we adopt a simple yet effective approach that combines discrete speech representations from self-supervised models with a non-autoregressive Diffusion-Transformer based conditional flow matching speech decoder. We show that this architecture allows us to train a voice-conversion model in a purely textless, self-supervised fashion. Our technique works without requiring multiple encoders to disentangle speech features. Our model also manages to excel in zero-shot cross-lingual settings even for unseen languages.
>
---
#### [new 003] Discrete Tokens Exhibit Interlanguage Speech Intelligibility Benefit: an Analytical Study Towards Accent-robust ASR Only with Native Speech Data
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究口音鲁棒ASR任务，旨在仅用母语数据提升系统对非母语口音的识别。通过分析自监督模型提取的离散语音token，验证了"中介语可懂度增益"现象（共享母语的非母语者识别更优），实验表明该方法可提升ASR跨口音 robustness，且无需非母语数据。**

- **链接: [http://arxiv.org/pdf/2505.16182v1](http://arxiv.org/pdf/2505.16182v1)**

> **作者:** Kentaro Onda; Keisuke Imoto; Satoru Fukayama; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** In this study, we gained insight that contributes to achieving accent-robust ASR using only native speech data. In human perception of non-native speech, the phenomenon known as "interlanguage speech intelligibility benefit" (ISIB) is observed, where non-native listeners who share the native language with the speaker understand the speech better compared even to native listeners. Based on the idea that discrete tokens extracted from self-supervised learning (SSL) models represent the human perception of speech, we conducted an analytical study on the robustness of discrete token-based ASR to non-native speech, varying the language used for training the tokenization, which is viewed as a technical implementation of ISIB. The results showed that ISIB actually occurred in the discrete token-based ASR. Since our approach relies only on native speech data to simulate the behavior of human perception, it is expected to be applicable to a wide range of accents for which speech data is scarce.
>
---
#### [new 004] Selective Invocation for Multilingual ASR: A Cost-effective Approach Adapting to Speech Recognition Difficulty
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多语言自动语音识别（ASR）任务，解决语言差异导致性能不均衡及传统语言识别（LID）方法成本高、误分类的问题。提出SIMA方法，基于口语化语言模型（SLLM）评估输入难度，动态决定直接转录或调用高性能ASR模型，降低词错率18.7%并减少一半调用成本，验证于三数据集。**

- **链接: [http://arxiv.org/pdf/2505.16168v1](http://arxiv.org/pdf/2505.16168v1)**

> **作者:** Hongfei Xue; Yufeng Tang; Jun Zhang; Xuelong Geng; Lei Xie
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Although multilingual automatic speech recognition (ASR) systems have significantly advanced, enabling a single model to handle multiple languages, inherent linguistic differences and data imbalances challenge SOTA performance across all languages. While language identification (LID) models can route speech to the appropriate ASR model, they incur high costs from invoking SOTA commercial models and suffer from inaccuracies due to misclassification. To overcome these, we propose SIMA, a selective invocation for multilingual ASR that adapts to the difficulty level of the input speech. Built on a spoken large language model (SLLM), SIMA evaluates whether the input is simple enough for direct transcription or requires the invocation of a SOTA ASR model. Our approach reduces word error rates by 18.7% compared to the SLLM and halves invocation costs compared to LID-based methods. Tests on three datasets show that SIMA is a scalable, cost-effective solution for multilingual ASR applications.
>
---
#### [new 005] A Novel Deep Learning Framework for Efficient Multichannel Acoustic Feedback Control
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出了一种基于深度学习的多通道声学反馈控制框架，旨在解决传统信号处理方法在处理高相关噪声时收敛困难的问题。研究设计了卷积循环网络结合空间-时间处理，并采用三种训练方法（In-a-Loop、Teacher Forcing及混合多通道维纳滤波策略），提升了复杂环境下的语音增强效果，降低了计算需求，为实际应用提供了高效鲁棒的解决方案。**

- **链接: [http://arxiv.org/pdf/2505.15914v1](http://arxiv.org/pdf/2505.15914v1)**

> **作者:** Yuan-Kuei Wu; Juan Azcarreta; Kashyap Patel; Buye Xu; Jung-Suk Lee; Sanha Lee; Ashutosh Pandey
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** This study presents a deep-learning framework for controlling multichannel acoustic feedback in audio devices. Traditional digital signal processing methods struggle with convergence when dealing with highly correlated noise such as feedback. We introduce a Convolutional Recurrent Network that efficiently combines spatial and temporal processing, significantly enhancing speech enhancement capabilities with lower computational demands. Our approach utilizes three training methods: In-a-Loop Training, Teacher Forcing, and a Hybrid strategy with a Multichannel Wiener Filter, optimizing performance in complex acoustic environments. This scalable framework offers a robust solution for real-world applications, making significant advances in Acoustic Feedback Control technology.
>
---
#### [new 006] Dialogue in Resonance: An Interactive Music Piece for Piano and Real-Time Automatic Transcription System
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出《Dialogue in Resonance》，开发人钢琴家与计算机控制钢琴的交互系统，通过实时自动乐谱转录技术，平衡预设乐曲结构与动态互动。解决传统即兴交互缺乏结构的问题，实现计算机对演奏的实时解析与回应，融合创作意图与现场不可预测性。工作包括技术实现、排练及演出流程设计。**

- **链接: [http://arxiv.org/pdf/2505.16259v1](http://arxiv.org/pdf/2505.16259v1)**

> **作者:** Hayeon Bang; Taegyun Kwon; Juhan Nam
>
> **摘要:** This paper presents <Dialogue in Resonance>, an interactive music piece for a human pianist and a computer-controlled piano that integrates real-time automatic music transcription into a score-driven framework. Unlike previous approaches that primarily focus on improvisation-based interactions, our work establishes a balanced framework that combines composed structure with dynamic interaction. Through real-time automatic transcription as its core mechanism, the computer interprets and responds to the human performer's input in real time, creating a musical dialogue that balances compositional intent with live interaction while incorporating elements of unpredictability. In this paper, we present the development process from composition to premiere performance, including technical implementation, rehearsal process, and performance considerations.
>
---
#### [new 007] Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐信息检索领域，旨在探究自监督学习模型的层级表征特性。针对模型编码信息的具体意义及适用性不足的问题，通过分析MusicFM和MuQ模型，验证其跨任务优势，研究各层对不同任务的专精度，并比较层选择对性能的影响，以优化模型应用。**

- **链接: [http://arxiv.org/pdf/2505.16306v1](http://arxiv.org/pdf/2505.16306v1)**

> **作者:** Yizhi Zhou; Haina Zhu; Hangting Chen
>
> **摘要:** Recently, pre-trained models for music information retrieval based on self-supervised learning (SSL) are becoming popular, showing success in various downstream tasks. However, there is limited research on the specific meanings of the encoded information and their applicability. Exploring these aspects can help us better understand their capabilities and limitations, leading to more effective use in downstream tasks. In this study, we analyze the advanced music representation model MusicFM and the newly emerged SSL model MuQ. We focus on three main aspects: (i) validating the advantages of SSL models across multiple downstream tasks, (ii) exploring the specialization of layer-wise information for different tasks, and (iii) comparing performance differences when selecting specific layers. Through this analysis, we reveal insights into the structure and potential applications of SSL models in music information retrieval.
>
---
#### [new 008] Differentiable K-means for Fully-optimized Discrete Token-based ASR
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对离散语音 token 生成与下游任务（如 ASR）脱节的问题，提出基于可微分 k-means 的联合优化方法，通过同步优化 token 化、SSL 模型参数及多层权重，提升 ASR 准确率并增强 token 的音素信息纯度。**

- **链接: [http://arxiv.org/pdf/2505.16207v1](http://arxiv.org/pdf/2505.16207v1)**

> **作者:** Kentaro Onda; Yosuke Kashiwagi; Emiru Tsunoo; Hayato Futami; Shinji Watanabe
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** Recent studies have highlighted the potential of discrete tokens derived from self-supervised learning (SSL) models for various speech-related tasks. These tokens serve not only as substitutes for text in language modeling but also as intermediate representations for tasks such as automatic speech recognition (ASR). However, discrete tokens are typically obtained via k-means clustering of SSL features independently of downstream tasks, making them suboptimal for specific applications. This paper proposes the use of differentiable k-means, enabling the joint optimization of tokenization and downstream tasks. This approach enables the fine-tuning of the SSL parameters and learning weights for outputs from multiple SSL layers. Experiments were conducted with ASR as a downstream task. ASR accuracy successfully improved owing to the optimized tokens. The acquired tokens also exhibited greater purity of phonetic information, which were found to be useful even in speech resynthesis.
>
---
#### [new 009] X-ARES: A Comprehensive Framework for Assessing Audio Encoder Performance
- **分类: cs.SD**

- **简介: 该论文提出X-ARES框架，用于系统评估音频编码器在语音、环境声、音乐等多领域的性能。针对现有评估方法覆盖不全的问题，设计包含22项任务的基准，采用线性微调和无参数评估两种方式，揭示了编码器跨任务表现差异，强调通用音频表征学习的复杂性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16369v1](http://arxiv.org/pdf/2505.16369v1)**

> **作者:** Junbo Zhang; Heinrich Dinkel; Yadong Niu; Chenyu Liu; Si Cheng; Anbei Zhao; Jian Luan
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** We introduces X-ARES (eXtensive Audio Representation and Evaluation Suite), a novel open-source benchmark designed to systematically assess audio encoder performance across diverse domains. By encompassing tasks spanning speech, environmental sounds, and music, X-ARES provides two evaluation approaches for evaluating audio representations: linear fine-tuning and unparameterized evaluation. The framework includes 22 distinct tasks that cover essential aspects of audio processing, from speech recognition and emotion detection to sound event classification and music genre identification. Our extensive evaluation of state-of-the-art audio encoders reveals significant performance variations across different tasks and domains, highlighting the complexity of general audio representation learning.
>
---
#### [new 010] Prosodically Enhanced Foreign Accent Simulation by Discrete Token-based Resynthesis Only with Native Speech Corpora
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于外语口音模拟任务。针对前人方法无法重现持续时间相关口音的缺陷，提出在离散令牌重合成框架中加入韵律时长修正模块，利用母语语料库更准确模拟二语者时长特征差异。实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.16191v1](http://arxiv.org/pdf/2505.16191v1)**

> **作者:** Kentaro Onda; Keisuke Imoto; Satoru Fukayama; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** Recently, a method for synthesizing foreign-accented speech only with native speech data using discrete tokens obtained from self-supervised learning (SSL) models was proposed. Considering limited availability of accented speech data, this method is expected to make it much easier to simulate foreign accents. By using the synthesized accented speech as listening materials for humans or training data for automatic speech recognition (ASR), both of them will acquire higher robustness against foreign accents. However, the previous method has a fatal flaw that it cannot reproduce duration-related accents. Durational accents are commonly seen when L2 speakers, whose native language has syllable-timed or mora-timed rhythm, speak stress-timed languages, such as English. In this paper, we integrate duration modification to the previous method to simulate foreign accents more accurately. Experiments show that the proposed method successfully replicates durational accents seen in real L2 speech.
>
---
#### [new 011] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出AudioTrust框架，用于评估音频大语言模型（ALLMs）的多维度可信度。针对现有评估方法忽视音频特有风险（如隐私、鲁棒性）的问题，构建含6个评估维度、18种实验设置及4420个真实场景样本的基准测试，并设计9项音频专用指标。实验揭示了当前模型在高风险场景中的局限性，助力安全部署。**

- **链接: [http://arxiv.org/pdf/2505.16211v1](http://arxiv.org/pdf/2505.16211v1)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Zhe Wang; Xingjian Du; Shun Zhang; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Xiaojun Jia; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Haoyang Li; Yiming Li; Xiaobin Zhuang; Yang Liu; Haibo Hu; Zhuo Chen; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; XiaoFeng Wang; Wenyuan Xu; Wei Dong; Xinfeng Li
>
> **备注:** Technical Report
>
> **摘要:** The rapid advancement and expanding applications of Audio Large Language Models (ALLMs) demand a rigorous understanding of their trustworthiness. However, systematic research on evaluating these models, particularly concerning risks unique to the audio modality, remains largely unexplored. Existing evaluation frameworks primarily focus on the text modality or address only a restricted set of safety dimensions, failing to adequately account for the unique characteristics and application scenarios inherent to the audio modality. We introduce AudioTrust-the first multifaceted trustworthiness evaluation framework and benchmark specifically designed for ALLMs. AudioTrust facilitates assessments across six key dimensions: fairness, hallucination, safety, privacy, robustness, and authentication. To comprehensively evaluate these dimensions, AudioTrust is structured around 18 distinct experimental setups. Its core is a meticulously constructed dataset of over 4,420 audio/text samples, drawn from real-world scenarios (e.g., daily conversations, emergency calls, voice assistant interactions), specifically designed to probe the multifaceted trustworthiness of ALLMs. For assessment, the benchmark carefully designs 9 audio-specific evaluation metrics, and we employ a large-scale automated pipeline for objective and scalable scoring of model outputs. Experimental results reveal the trustworthiness boundaries and limitations of current state-of-the-art open-source and closed-source ALLMs when confronted with various high-risk audio scenarios, offering valuable insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are available at https://github.com/JusperLee/AudioTrust.
>
---
#### [new 012] SpecMaskFoley: Steering Pretrained Spectral Masked Generative Transformer Toward Synchronized Video-to-audio Synthesis via ControlNet
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.IV**

- **简介: 该论文属于视频同步 Foley音频合成任务，旨在缩小ControlNet适配预训练模型与从头训练模型的性能差距。针对ControlNet依赖手工时序条件而效果不足的问题，提出SpecMaskFoley方法，通过频率感知时序对齐器将视频特征与预训练SpecMaskGIT的频谱特性匹配，简化条件机制，提升生成质量，实验显示其超越了从头训练的基线模型。**

- **链接: [http://arxiv.org/pdf/2505.16195v1](http://arxiv.org/pdf/2505.16195v1)**

> **作者:** Zhi Zhong; Akira Takahashi; Shuyang Cui; Keisuke Toyama; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** 4 pages, 2 figures, 2 tables. Demo page: https://zzaudio.github.io/SpecMaskFoley_Demo/
>
> **摘要:** Foley synthesis aims to synthesize high-quality audio that is both semantically and temporally aligned with video frames. Given its broad application in creative industries, the task has gained increasing attention in the research community. To avoid the non-trivial task of training audio generative models from scratch, adapting pretrained audio generative models for video-synchronized foley synthesis presents an attractive direction. ControlNet, a method for adding fine-grained controls to pretrained generative models, has been applied to foley synthesis, but its use has been limited to handcrafted human-readable temporal conditions. In contrast, from-scratch models achieved success by leveraging high-dimensional deep features extracted using pretrained video encoders. We have observed a performance gap between ControlNet-based and from-scratch foley models. To narrow this gap, we propose SpecMaskFoley, a method that steers the pretrained SpecMaskGIT model toward video-synchronized foley synthesis via ControlNet. To unlock the potential of a single ControlNet branch, we resolve the discrepancy between the temporal video features and the time-frequency nature of the pretrained SpecMaskGIT via a frequency-aware temporal feature aligner, eliminating the need for complicated conditioning mechanisms widely used in prior arts. Evaluations on a common foley synthesis benchmark demonstrate that SpecMaskFoley could even outperform strong from-scratch baselines, substantially advancing the development of ControlNet-based foley synthesis models. Demo page: https://zzaudio.github.io/SpecMaskFoley_Demo/
>
---
#### [new 013] UBGAN: Enhancing Coded Speech with Blind and Guided Bandwidth Extension
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出UBGAN，一种基于GAN的模块化带宽扩展方法，解决现有语音编解码器在灵活性和带宽扩展上的局限。通过将8kHz宽频信号扩展至16kHz超宽频，其两种变体（需低比特率侧信道的引导式与无需信道的盲式）提升了多种编解码器的音质与适应性，主观测试验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.16404v1](http://arxiv.org/pdf/2505.16404v1)**

> **作者:** Kishan Gupta; Srikanth Korse; Andreas Brendel; Nicola Pia; Guillaume Fuchs
>
> **摘要:** In practical application of speech codecs, a multitude of factors such as the quality of the radio connection, limiting hardware or required user experience necessitate trade-offs between achievable perceptual quality, engendered bitrate and computational complexity. Most conventional and neural speech codecs operate on wideband (WB) speech signals to achieve this compromise. To further enhance the perceptual quality of coded speech, bandwidth extension (BWE) of the transmitted speech is an attractive and popular technique in conventional speech coding. In contrast, neural speech codecs are typically trained end-to-end to a specific set of requirements and are often not easily adaptable. In particular, they are typically trained to operate at a single fixed sampling rate. With the Universal Bandwidth Extension Generative Adversarial Network (UBGAN), we propose a modular and lightweight GAN-based solution that increases the operational flexibility of a wide range of conventional and neural codecs. Our model operates in the subband domain and extends the bandwidth of WB signals from 8 kHz to 16 kHz, resulting in super-wideband (SWB) signals. We further introduce two variants, guided-UBGAN and blind-UBGAN, where the guided version transmits quantized learned representation as a side information at a very low bitrate additional to the bitrate of the codec, while blind-BWE operates without such side-information. Our subjective assessments demonstrate the advantage of UBGAN applied to WB codecs and highlight the generalization capacity of our proposed method across multiple codecs and bitrates.
>
---
#### [new 014] Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于大型音频语言模型（LALM）评估任务，旨在解决现有评估方法碎片化、缺乏系统分类的问题。提出四维评估分类法（听觉处理、知识推理、对话能力、公平安全），总结挑战并指明未来方向，同时公开论文集支持研究。**

- **链接: [http://arxiv.org/pdf/2505.15957v1](http://arxiv.org/pdf/2505.15957v1)**

> **作者:** Chih-Kai Yang; Neo S. Ho; Hung-yi Lee
>
> **备注:** Project Website: https://github.com/b08202033/LALM-Evaluation-Survey
>
> **摘要:** With advancements in large audio-language models (LALMs), which enhance large language models (LLMs) with auditory capabilities, these models are expected to demonstrate universal proficiency across various auditory tasks. While numerous benchmarks have emerged to assess LALMs' performance, they remain fragmented and lack a structured taxonomy. To bridge this gap, we conduct a comprehensive survey and propose a systematic taxonomy for LALM evaluations, categorizing them into four dimensions based on their objectives: (1) General Auditory Awareness and Processing, (2) Knowledge and Reasoning, (3) Dialogue-oriented Ability, and (4) Fairness, Safety, and Trustworthiness. We provide detailed overviews within each category and highlight challenges in this field, offering insights into promising future directions. To the best of our knowledge, this is the first survey specifically focused on the evaluations of LALMs, providing clear guidelines for the community. We will release the collection of the surveyed papers and actively maintain it to support ongoing advancements in the field.
>
---
#### [new 015] Unsupervised Network Anomaly Detection with Autoencoders and Traffic Images
- **分类: cs.CV; cs.CR; eess.IV; eess.SP**

- **简介: 该论文属于无监督网络异常检测任务，旨在解决海量异构设备流量数据的高效安全检测问题。提出基于流量图像的紧凑表示方法（1秒时间窗），通过自编码器实现轻量化的无监督异常识别，减少复杂计算需求。**

- **链接: [http://arxiv.org/pdf/2505.16650v1](http://arxiv.org/pdf/2505.16650v1)**

> **作者:** Michael Neri; Sara Baldoni
>
> **备注:** Accepted for publication in EUSIPCO 2025
>
> **摘要:** Due to the recent increase in the number of connected devices, the need to promptly detect security issues is emerging. Moreover, the high number of communication flows creates the necessity of processing huge amounts of data. Furthermore, the connected devices are heterogeneous in nature, having different computational capacities. For this reason, in this work we propose an image-based representation of network traffic which allows to realize a compact summary of the current network conditions with 1-second time windows. The proposed representation highlights the presence of anomalies thus reducing the need for complex processing architectures. Finally, we present an unsupervised learning approach which effectively detects the presence of anomalies. The code and the dataset are available at https://github.com/michaelneri/image-based-network-traffic-anomaly-detection.
>
---
#### [new 016] Unlocking Temporal Flexibility: Neural Speech Codec with Variable Frame Rate
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于神经语音编解码优化任务。针对恒定帧率（CFR）在处理时间信息密度变化的语音段（如静音与有声区域）时效率不足的问题，提出时序灵活编码（TFC）技术，首次引入可变帧率（VFR），动态分配帧率以匹配时间熵，提升编码效率与灵活性，实验验证其在低帧率下的竞争力。**

- **链接: [http://arxiv.org/pdf/2505.16845v1](http://arxiv.org/pdf/2505.16845v1)**

> **作者:** Hanglei Zhang; Yiwei Guo; Zhihan Li; Xiang Hao; Xie Chen; Kai Yu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Most neural speech codecs achieve bitrate adjustment through intra-frame mechanisms, such as codebook dropout, at a Constant Frame Rate (CFR). However, speech segments inherently have time-varying information density (e.g., silent intervals versus voiced regions). This property makes CFR not optimal in terms of bitrate and token sequence length, hindering efficiency in real-time applications. In this work, we propose a Temporally Flexible Coding (TFC) technique, introducing variable frame rate (VFR) into neural speech codecs for the first time. TFC enables seamlessly tunable average frame rates and dynamically allocates frame rates based on temporal entropy. Experimental results show that a codec with TFC achieves optimal reconstruction quality with high flexibility, and maintains competitive performance even at lower frame rates. Our approach is promising for the integration with other efforts to develop low-frame-rate neural speech codecs for more efficient downstream tasks.
>
---
#### [new 017] Attractor-Based Speech Separation of Multiple Utterances by Unknown Number of Speakers
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出基于吸引子的语音分离模型，解决单通道未知说话人数及多utterance分离问题。通过集成吸引子模块，同时实现语音分离、动态估计说话者数量及检测活动，有效处理混响噪声环境，在合成数据集上表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.16607v1](http://arxiv.org/pdf/2505.16607v1)**

> **作者:** Yuzhu Wang; Archontis Politis; Konstantinos Drossos; Tuomas Virtanen
>
> **备注:** 5 pages, 4 figures, accepted by Interspeech 2025
>
> **摘要:** This paper addresses the problem of single-channel speech separation, where the number of speakers is unknown, and each speaker may speak multiple utterances. We propose a speech separation model that simultaneously performs separation, dynamically estimates the number of speakers, and detects individual speaker activities by integrating an attractor module. The proposed system outperforms existing methods by introducing an attractor-based architecture that effectively combines local and global temporal modeling for multi-utterance scenarios. To evaluate the method in reverberant and noisy conditions, a multi-speaker multi-utterance dataset was synthesized by combining Librispeech speech signals with WHAM! noise signals. The results demonstrate that the proposed system accurately estimates the number of sources. The system effectively detects source activities and separates the corresponding utterances into correct outputs in both known and unknown source count scenarios.
>
---
#### [new 018] Analyzing the Impact of Accent on English Speech: Acoustic and Articulatory Perspectives
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音处理研究，旨在解决非母语口音对AI语音系统识别的挑战。通过声学（如 eigenspectra）和发音协调特征分析，发现非母语者存在更简单的发音模式和更高音调，并提出无需依赖复杂转录的口音强度量化方法，助力开发更包容的语音技术。**

- **链接: [http://arxiv.org/pdf/2505.15965v1](http://arxiv.org/pdf/2505.15965v1)**

> **作者:** Gowtham Premananth; Vinith Kugathasan; Carol Espy-Wilson
>
> **备注:** Accepted to be presented at Interspeech 2025
>
> **摘要:** Advancements in AI-driven speech-based applications have transformed diverse industries ranging from healthcare to customer service. However, the increasing prevalence of non-native accented speech in global interactions poses significant challenges for speech-processing systems, which are often trained on datasets dominated by native speech. This study investigates accented English speech through articulatory and acoustic analysis, identifying simpler coordination patterns and higher average pitch than native speech. Using eigenspectra and Vocal Tract Variable-based coordination features, we establish an efficient method for quantifying accent strength without relying on resource-intensive phonetic transcriptions. Our findings provide a new avenue for research on the impacts of accents on speech intelligibility and offer insights for developing inclusive, robust speech processing systems that accommodate diverse linguistic communities.
>
---
#### [new 019] Multimodal Biomarkers for Schizophrenia: Towards Individual Symptom Severity Estimation
- **分类: eess.AS; cs.LG; eess.IV; eess.SP**

- **简介: 该论文属于精神分裂症个体症状严重程度估计任务。针对传统深度学习仅分类患病与否、忽略病情复杂性的不足，提出整合语音、视频和文本的多模态框架，开发单模态与多模态模型以提升诊断精准度，支持个性化治疗，提供客观可扩展的评估工具。（99字）**

- **链接: [http://arxiv.org/pdf/2505.16044v1](http://arxiv.org/pdf/2505.16044v1)**

> **作者:** Gowtham Premananth; Philip Resnik; Sonia Bansal; Deanna L. Kelly; Carol Espy-Wilson
>
> **备注:** Accepted to be presented at Interspeech 2025
>
> **摘要:** Studies on schizophrenia assessments using deep learning typically treat it as a classification task to detect the presence or absence of the disorder, oversimplifying the condition and reducing its clinical applicability. This traditional approach overlooks the complexity of schizophrenia, limiting its practical value in healthcare settings. This study shifts the focus to individual symptom severity estimation using a multimodal approach that integrates speech, video, and text inputs. We develop unimodal models for each modality and a multimodal framework to improve accuracy and robustness. By capturing a more detailed symptom profile, this approach can help in enhancing diagnostic precision and support personalized treatment, offering a scalable and objective tool for mental health assessment.
>
---
#### [new 020] From Tens of Hours to Tens of Thousands: Scaling Back-Translation for Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出Speech Back-Translation方法，解决多语言语音识别中数据稀缺问题。通过将文本转为合成语音，仅用少量真实数据生成大量高质量合成语音，建立可扩展的训练 pipeline。其开发智能性评估框架，验证合成数据对ASR的提升效果，最终使Whisper模型错误率下降超30%。**

- **链接: [http://arxiv.org/pdf/2505.16972v1](http://arxiv.org/pdf/2505.16972v1)**

> **作者:** Tianduo Wang; Lu Xu; Wei Lu; Shanbo Cheng
>
> **摘要:** Recent advances in Automatic Speech Recognition (ASR) have been largely fueled by massive speech corpora. However, extending coverage to diverse languages with limited resources remains a formidable challenge. This paper introduces Speech Back-Translation, a scalable pipeline that improves multilingual ASR models by converting large-scale text corpora into synthetic speech via off-the-shelf text-to-speech (TTS) models. We demonstrate that just tens of hours of real transcribed speech can effectively train TTS models to generate synthetic speech at hundreds of times the original volume while maintaining high quality. To evaluate synthetic speech quality, we develop an intelligibility-based assessment framework and establish clear thresholds for when synthetic data benefits ASR training. Using Speech Back-Translation, we generate more than 500,000 hours of synthetic speech in ten languages and continue pre-training Whisper-large-v3, achieving average transcription error reductions of over 30\%. These results highlight the scalability and effectiveness of Speech Back-Translation for enhancing multilingual ASR systems.
>
---
## 更新

#### [replaced 001] Slamming: Training a Speech Language Model on One GPU in a Day
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.15814v2](http://arxiv.org/pdf/2502.15814v2)**

> **作者:** Gallil Maimon; Avishai Elmakies; Yossi Adi
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** We introduce Slam, a recipe for training high-quality Speech Language Models (SLMs) on a single academic GPU in 24 hours. We do so through empirical analysis of model initialisation and architecture, synthetic training data, preference optimisation with synthetic data and tweaking all other components. We empirically demonstrate that this training recipe also scales well with more compute getting results on par with leading SLMs in a fraction of the compute cost. We hope these insights will make SLM training and research more accessible. In the context of SLM scaling laws, our results far outperform predicted compute optimal performance, giving an optimistic view to SLM feasibility. See code, data, models, samples at - https://pages.cs.huji.ac.il/adiyoss-lab/slamming .
>
---
#### [replaced 002] Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.13830v2](http://arxiv.org/pdf/2505.13830v2)**

> **作者:** Ye-Xin Lu; Hui-Peng Du; Fei Liu; Yang Ai; Zhen-Hua Ling
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
>
---
#### [replaced 003] ShiftySpeech: A Large-Scale Synthetic Speech Dataset with Distribution Shifts
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.05674v4](http://arxiv.org/pdf/2502.05674v4)**

> **作者:** Ashi Garg; Zexin Cai; Lin Zhang; Henry Li Xinyuan; Leibny Paola García-Perera; Kevin Duh; Sanjeev Khudanpur; Matthew Wiesner; Nicholas Andrews
>
> **摘要:** The problem of synthetic speech detection has enjoyed considerable attention, with recent methods achieving low error rates across several established benchmarks. However, to what extent can low error rates on academic benchmarks translate to more realistic conditions? In practice, while the training set is fixed at one point in time, test-time conditions may exhibit distribution shifts relative to the training conditions, such as changes in speaker characteristics, emotional expressiveness, language and acoustic conditions, and the emergence of novel synthesis methods. Although some existing datasets target subsets of these distribution shifts, systematic analysis remains difficult due to inconsistencies between source data and synthesis systems across datasets. This difficulty is further exacerbated by the rapid development of new text-to-speech (TTS) and vocoder systems, which continually expand the diversity of synthetic speech. To enable systematic benchmarking of model performance under distribution shifts, we introduce ShiftySpeech, a large-scale benchmark comprising over 3,000 hours of synthetic speech across 7 source domains, 6 TTS systems, 12 vocoders, and 3 languages. ShiftySpeech is specifically designed to evaluate model generalization under controlled distribution shifts while ensuring broad coverage of modern synthetic speech generation techniques. It fills a key gap in current benchmarks by supporting fine-grained, controlled analysis of generalization robustness. All tested distribution shifts significantly degrade detection performance of state-of-the-art detection approaches based on self-supervised features. Overall, our findings suggest that reliance on synthetic speech detection methods in production environments should be carefully evaluated based on anticipated distribution shifts.
>
---
#### [replaced 004] Long-Form Text-to-Music Generation with Adaptive Prompts: A Case Study in Tabletop Role-Playing Games Soundtracks
- **分类: cs.SD; cs.AI; cs.MM; cs.NE; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.03948v3](http://arxiv.org/pdf/2411.03948v3)**

> **作者:** Felipe Marra; Lucas N. Ferreira
>
> **备注:** Proceedings of the 1st Latin American Music Information Retrieval Workshop (LAMIR), pg 80
>
> **摘要:** This paper investigates the capabilities of text-to-audio music generation models in producing long-form music with prompts that change over time, focusing on soundtrack generation for Tabletop Role-Playing Games (TRPGs). We introduce Babel Bardo, a system that uses Large Language Models (LLMs) to transform speech transcriptions into music descriptions for controlling a text-to-music model. Four versions of Babel Bardo were compared in two TRPG campaigns: a baseline using direct speech transcriptions, and three LLM-based versions with varying approaches to music description generation. Evaluations considered audio quality, story alignment, and transition smoothness. Results indicate that detailed music descriptions improve audio quality while maintaining consistency across consecutive descriptions enhances story alignment and transition smoothness.
>
---
#### [replaced 005] Neurodyne: Neural Pitch Manipulation with Representation Learning and Cycle-Consistency GAN
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15368v2](http://arxiv.org/pdf/2505.15368v2)**

> **作者:** Yicheng Gu; Chaoren Wang; Zhizheng Wu; Lauri Juvela
>
> **摘要:** Pitch manipulation is the process of producers adjusting the pitch of an audio segment to a specific key and intonation, which is essential in music production. Neural-network-based pitch-manipulation systems have been popular in recent years due to their superior synthesis quality compared to classical DSP methods. However, their performance is still limited due to their inaccurate feature disentanglement using source-filter models and the lack of paired in- and out-of-tune training data. This work proposes Neurodyne to address these issues. Specifically, Neurodyne uses adversarial representation learning to learn a pitch-independent latent representation to avoid inaccurate disentanglement and cycle-consistency training to create paired training data implicitly. Experimental results on global-key and template-based pitch manipulation demonstrate the effectiveness of the proposed system, marking improved synthesis quality while maintaining the original singer identity.
>
---
#### [replaced 006] TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.07053v2](http://arxiv.org/pdf/2504.07053v2)**

> **作者:** Liang-Hsuan Tseng; Yi-Chang Chen; Kuan-Yi Lee; Da-Shan Shiu; Hung-yi Lee
>
> **备注:** Preprint
>
> **摘要:** Recent efforts target spoken language models (SLMs) that not only listen but also speak for more natural human-LLM interaction. Joint speech-text modeling is a promising direction to achieve this. However, the effectiveness of recent speech tokens for joint modeling remains underexplored. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through a attention-based aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. With TASTE, we perform straightforward joint spoken language modeling by using Low-Rank Adaptation on the pre-trained text LLM. Experimental results show that TASTE-based SLMs perform comparable to previous work on SALMON and StoryCloze; while significantly outperform other pre-trained SLMs on speech continuation across subjective and objective evaluations. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and model are available at https://mtkresearch.github.io/TASTE-SpokenLM.github.io.
>
---
