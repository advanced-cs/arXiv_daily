# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Denoising GER: A Noise-Robust Generative Error Correction with LLM for Speech Recognition
- **分类: cs.SD**

- **简介: 论文提出Denoising GER框架，针对语音识别中噪声环境下的生成错误校正（GER）问题，通过噪声自适应编码器、异构特征融合及强化学习提升模型鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.04392v1](http://arxiv.org/pdf/2509.04392v1)**

> **作者:** Yanyan Liu; Minqiang Xu; Yihao Chen; Liang He; Lei Fang; Sian Fang; Lin Liu
>
> **摘要:** In recent years, large language models (LLM) have made significant progress in the task of generation error correction (GER) for automatic speech recognition (ASR) post-processing. However, in complex noisy environments, they still face challenges such as poor adaptability and low information utilization, resulting in limited effectiveness of GER. To address these issues, this paper proposes a noise-robust multi-modal GER framework (Denoising GER). The framework enhances the model's adaptability to different noisy scenarios through a noise-adaptive acoustic encoder and optimizes the integration of multi-modal information via a heterogeneous feature compensation dynamic fusion (HFCDF) mechanism, improving the LLM's utilization of multi-modal information. Additionally, reinforcement learning (RL) training strategies are introduced to enhance the model's predictive capabilities. Experimental results demonstrate that Denoising GER significantly improves accuracy and robustness in noisy environments and exhibits good generalization abilities in unseen noise scenarios.
>
---
#### [new 002] Open-Source Full-Duplex Conversational Datasets for Natural and Interactive Speech Synthesis
- **分类: cs.SD**

- **简介: 论文构建中英文全双工对话数据集，用于提升语音合成的自然性和互动性。通过真实对话记录与标注，验证数据集在微调TTS模型中的有效性，提高合成语音质量。**

- **链接: [http://arxiv.org/pdf/2509.04093v1](http://arxiv.org/pdf/2509.04093v1)**

> **作者:** Zhitong Zhou; Qingqing Zhang; Lei Luo; Jiechen Liu; Ruohua Zhou
>
> **摘要:** Full-duplex, spontaneous conversational data are essential for enhancing the naturalness and interactivity of synthesized speech in conversational TTS systems. We present two open-source dual-track conversational speech datasets, one in Chinese and one in English, designed to enhance the naturalness of synthesized speech by providing more realistic conversational data. The two datasets contain a total of 15 hours of natural, spontaneous conversations recorded in isolated rooms, which produces separate high-quality audio tracks for each speaker. The conversations cover diverse daily topics and domains, capturing realistic interaction patterns including frequent overlaps, backchannel responses, laughter, and other non-verbal vocalizations. We introduce the data collection procedure, transcription and annotation methods. We demonstrate the utility of these corpora by fine-tuning a baseline TTS model with the proposed datasets. The fine-tuned TTS model achieves higher subjective and objective evaluation metrics compared to the baseline, indicating improved naturalness and conversational realism in synthetic speech. All data, annotations, and supporting code for fine-tuning and evaluation are made available to facilitate further research in conversational speech synthesis.
>
---
#### [new 003] Enhancing Self-Supervised Speaker Verification Using Similarity-Connected Graphs and GCN
- **分类: cs.SD**

- **简介: 该论文针对自监督说话人验证中伪标签噪声问题，提出基于相似性连接图与GCN的改进聚类框架，优化伪标签生成以提升系统性能。**

- **链接: [http://arxiv.org/pdf/2509.04147v1](http://arxiv.org/pdf/2509.04147v1)**

> **作者:** Zhaorui Sun; Yihao Chen; Jialong Wang; Minqiang Xu; Lei Fang; Sian Fang; Lin Liu
>
> **摘要:** With the continuous development of speech recognition technology, speaker verification (SV) has become an important method for identity authentication. Traditional SV methods rely on handcrafted feature extraction, while deep learning has significantly improved system performance. However, the scarcity of labeled data still limits the widespread application of deep learning in SV. Self-supervised learning, by mining latent information in large unlabeled datasets, enhances model generalization and is a key technology to address this issue. DINO is an efficient self-supervised learning method that generates pseudo-labels from unlabeled speech data through clustering, supporting subsequent training. However, clustering may produce noisy pseudo-labels, which can reduce overall recognition performance. To address this issue, this paper proposes an improved clustering framework based on similarity connection graphs and Graph Convolutional Networks. By leveraging GCNs' ability to model structured data and incorporating relational information between nodes in the similarity connection graph, the clustering process is optimized, improving pseudo-label accuracy and enhancing the robustness and performance of the self-supervised speaker verification system. Experimental results show that this method significantly improves system performance and provides a new approach for self-supervised speaker verification. Index Terms: Speaker Verification, Self-Supervised Learning, DINO, Clustering Algorithm, Graph Convolutional Network, Similarity Connection Graph
>
---
#### [new 004] WenetSpeech-Yue: A Large-scale Cantonese Speech Corpus with Multi-dimensional Annotation
- **分类: cs.SD**

- **简介: 论文构建多维标注的粤语大规模语音语料库，解决资源不足导致的ASR/TTS性能问题。提出集成流水线，生成21800小时数据及评估基准，提升模型表现。**

- **链接: [http://arxiv.org/pdf/2509.03959v1](http://arxiv.org/pdf/2509.03959v1)**

> **作者:** Longhao Li; Zhao Guo; Hongjie Chen; Yuhang Dai; Ziyu Zhang; Hongfei Xue; Tianlun Zuo; Chengyou Wang; Shuiyuan Wang; Jie Li; Xin Xu; Hui Bu; Binbin Zhang; Ruibin Yuan; Ziya Zhou; Wei Xue; Lei Xie
>
> **摘要:** The development of speech understanding and generation has been significantly accelerated by the availability of large-scale, high-quality speech datasets. Among these, ASR and TTS are regarded as the most established and fundamental tasks. However, for Cantonese (Yue Chinese), spoken by approximately 84.9 million native speakers worldwide, limited annotated resources have hindered progress and resulted in suboptimal ASR and TTS performance. To address this challenge, we propose WenetSpeech-Pipe, an integrated pipeline for building large-scale speech corpus with multi-dimensional annotation tailored for speech understanding and generation. It comprises six modules: Audio Collection, Speaker Attributes Annotation, Speech Quality Annotation, Automatic Speech Recognition, Text Postprocessing and Recognizer Output Voting, enabling rich and high-quality annotations. Based on this pipeline, we release WenetSpeech-Yue, the first large-scale Cantonese speech corpus with multi-dimensional annotation for ASR and TTS, covering 21,800 hours across 10 domains with annotations including ASR transcription, text confidence, speaker identity, age, gender, speech quality scores, among other annotations. We also release WSYue-eval, a comprehensive Cantonese benchmark with two components: WSYue-ASR-eval, a manually annotated set for evaluating ASR on short and long utterances, code-switching, and diverse acoustic conditions, and WSYue-TTS-eval, with base and coverage subsets for standard and generalization testing. Experimental results show that models trained on WenetSpeech-Yue achieve competitive results against state-of-the-art (SOTA) Cantonese ASR and TTS systems, including commercial and LLM-based models, highlighting the value of our dataset and pipeline.
>
---
#### [new 005] PianoBind: A Multimodal Joint Embedding Model for Pop-piano Music
- **分类: cs.SD; cs.IR; cs.MM**

- **简介: 该论文提出PianoBind，解决通用音乐模型在钢琴音乐语义捕捉和多模态处理上的不足，通过多源训练与联合嵌入优化，提升文本到音乐检索性能。**

- **链接: [http://arxiv.org/pdf/2509.04215v1](http://arxiv.org/pdf/2509.04215v1)**

> **作者:** Hayeon Bang; Eunjin Choi; Seungheon Doh; Juhan Nam
>
> **备注:** Accepted for publication at the 26th International Society for Music Information Retrieval Conference (ISMIR 2025)
>
> **摘要:** Solo piano music, despite being a single-instrument medium, possesses significant expressive capabilities, conveying rich semantic information across genres, moods, and styles. However, current general-purpose music representation models, predominantly trained on large-scale datasets, often struggle to captures subtle semantic distinctions within homogeneous solo piano music. Furthermore, existing piano-specific representation models are typically unimodal, failing to capture the inherently multimodal nature of piano music, expressed through audio, symbolic, and textual modalities. To address these limitations, we propose PianoBind, a piano-specific multimodal joint embedding model. We systematically investigate strategies for multi-source training and modality utilization within a joint embedding framework optimized for capturing fine-grained semantic distinctions in (1) small-scale and (2) homogeneous piano datasets. Our experimental results demonstrate that PianoBind learns multimodal representations that effectively capture subtle nuances of piano music, achieving superior text-to-music retrieval performance on in-domain and out-of-domain piano datasets compared to general-purpose music joint embedding models. Moreover, our design choices offer reusable insights for multimodal representation learning with homogeneous datasets beyond piano music.
>
---
#### [new 006] Wav2DF-TSL: Two-stage Learning with Efficient Pre-training and Hierarchical Experts Fusion for Robust Audio Deepfake Detection
- **分类: cs.SD**

- **简介: 该论文提出Wav2DF-TSL方法，针对音频深度伪造检测中领域偏差问题，通过两阶段学习：预训练用适配器学习伪造语音伪影，微调采用HA-MoE融合多级线索，显著提升跨域检测性能。**

- **链接: [http://arxiv.org/pdf/2509.04161v1](http://arxiv.org/pdf/2509.04161v1)**

> **作者:** Yunqi Hao; Yihao Chen; Minqiang Xu; Jianbo Zhan; Liang He; Lei Fang; Sian Fang; Lin Liu
>
> **摘要:** In recent years, self-supervised learning (SSL) models have made significant progress in audio deepfake detection (ADD) tasks. However, existing SSL models mainly rely on large-scale real speech for pre-training and lack the learning of spoofed samples, which leads to susceptibility to domain bias during the fine-tuning process of the ADD task. To this end, we propose a two-stage learning strategy (Wav2DF-TSL) based on pre-training and hierarchical expert fusion for robust audio deepfake detection. In the pre-training stage, we use adapters to efficiently learn artifacts from 3000 hours of unlabelled spoofed speech, improving the adaptability of front-end features while mitigating catastrophic forgetting. In the fine-tuning stage, we propose the hierarchical adaptive mixture of experts (HA-MoE) method to dynamically fuse multi-level spoofing cues through multi-expert collaboration with gated routing. Experimental results show that the proposed method significantly outperforms the baseline system on all four benchmark datasets, especially on the cross-domain In-the-wild dataset, achieving a 27.5% relative improvement in equal error rate (EER), outperforming the existing state-of-the-art systems. Index Terms: audio deepfake detection, self-supervised learning, parameter-efficient fine-tuning, mixture of experts
>
---
#### [new 007] Contextualized Token Discrimination for Speech Search Query Correction
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出CTD方法，用于语音搜索中的查询拼写纠正。通过BERT生成上下文表示，结合组合层增强语义，对比原始与上下文表示纠正错误，并构建新基准数据集。**

- **链接: [http://arxiv.org/pdf/2509.04393v1](http://arxiv.org/pdf/2509.04393v1)**

> **作者:** Junyu Lu; Di Jiang; Mengze Hong; Victor Junqiu Wei; Qintian Guo; Zhiyang Su
>
> **摘要:** Query spelling correction is an important function of modern search engines since it effectively helps users express their intentions clearly. With the growing popularity of speech search driven by Automated Speech Recognition (ASR) systems, this paper introduces a novel method named Contextualized Token Discrimination (CTD) to conduct effective speech query correction. In CTD, we first employ BERT to generate token-level contextualized representations and then construct a composition layer to enhance semantic information. Finally, we produce the correct query according to the aggregated token representation, correcting the incorrect tokens by comparing the original token representations and the contextualized representations. Extensive experiments demonstrate the superior performance of our proposed method across all metrics, and we further present a new benchmark dataset with erroneous ASR transcriptions to offer comprehensive evaluations for audio query correction.
>
---
#### [new 008] AUDETER: A Large-scale Dataset for Deepfake Audio Detection in Open Worlds
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 论文提出AUDETER，首个大规模深度伪造音频数据集，解决现有数据集在真实环境泛化能力不足的问题。通过整合多种TTS和声码器生成300万音频片段，验证方法在跨域检测中错误率降低44.1%-51.6%。**

- **链接: [http://arxiv.org/pdf/2509.04345v1](http://arxiv.org/pdf/2509.04345v1)**

> **作者:** Qizhou Wang; Hanxun Huang; Guansong Pang; Sarah Erfani; Christopher Leckie
>
> **摘要:** Speech generation systems can produce remarkably realistic vocalisations that are often indistinguishable from human speech, posing significant authenticity challenges. Although numerous deepfake detection methods have been developed, their effectiveness in real-world environments remains unrealiable due to the domain shift between training and test samples arising from diverse human speech and fast evolving speech synthesis systems. This is not adequately addressed by current datasets, which lack real-world application challenges with diverse and up-to-date audios in both real and deep-fake categories. To fill this gap, we introduce AUDETER (AUdio DEepfake TEst Range), a large-scale, highly diverse deepfake audio dataset for comprehensive evaluation and robust development of generalised models for deepfake audio detection. It consists of over 4,500 hours of synthetic audio generated by 11 recent TTS models and 10 vocoders with a broad range of TTS/vocoder patterns, totalling 3 million audio clips, making it the largest deepfake audio dataset by scale. Through extensive experiments with AUDETER, we reveal that i) state-of-the-art (SOTA) methods trained on existing datasets struggle to generalise to novel deepfake audio samples and suffer from high false positive rates on unseen human voice, underscoring the need for a comprehensive dataset; and ii) these methods trained on AUDETER achieve highly generalised detection performance and significantly reduce detection error rate by 44.1% to 51.6%, achieving an error rate of only 4.17% on diverse cross-domain samples in the popular In-the-Wild dataset, paving the way for training generalist deepfake audio detectors. AUDETER is available on GitHub.
>
---
#### [new 009] SwinSRGAN: Swin Transformer-based Generative Adversarial Network for High-Fidelity Speech Super-Resolution
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SwinSRGAN，解决语音超分辨率中表示不匹配和过平滑问题。通过Swin Transformer与混合对抗框架结合，采用多频段判别器和稀疏正则化，实现高效实时的高质量语音上采样。**

- **链接: [http://arxiv.org/pdf/2509.03913v1](http://arxiv.org/pdf/2509.03913v1)**

> **作者:** Jiajun Yuan; Xiaochen Wang; Yuhang Xiao; Yulin Wu; Chenhao Hu; Xueyang Lv
>
> **备注:** 5 pages
>
> **摘要:** Speech super-resolution (SR) reconstructs high-frequency content from low-resolution speech signals. Existing systems often suffer from representation mismatch in two-stage mel-vocoder pipelines and from over-smoothing of hallucinated high-band content by CNN-only generators. Diffusion and flow models are computationally expensive, and their robustness across domains and sampling rates remains limited. We propose SwinSRGAN, an end-to-end framework operating on Modified Discrete Cosine Transform (MDCT) magnitudes. It is a Swin Transformer-based U-Net that captures long-range spectro-temporal dependencies with a hybrid adversarial scheme combines time-domain MPD/MSD discriminators with a multi-band MDCT discriminator specialized for the high-frequency band. We employs a sparse-aware regularizer on arcsinh-compressed MDCT to better preserve transient components. The system upsamples inputs at various sampling rates to 48 kHz in a single pass and operates in real time. On standard benchmarks, SwinSRGAN reduces objective error and improves ABX preference scores. In zero-shot tests on HiFi-TTS without fine-tuning, it outperforms NVSR and mdctGAN, demonstrating strong generalization across datasets
>
---
#### [new 010] VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出VoxRole基准，解决语音角色扮演代理评估中忽视语音特征与缺乏标准基准的问题，构建包含65.6小时电影对话数据的资源，通过自动化流程生成角色档案并进行多维评估。**

- **链接: [http://arxiv.org/pdf/2509.03940v1](http://arxiv.org/pdf/2509.03940v1)**

> **作者:** Weihao Wu; Liang Cao; Xinyu Wu; Zhiwei Lin; Rui Niu; Jingbei Li; Zhiyong Wu
>
> **摘要:** Recent significant advancements in Large Language Models (LLMs) have greatly propelled the development of Role-Playing Conversational Agents (RPCAs). These systems aim to create immersive user experiences through consistent persona adoption. However, current RPCA research faces dual limitations. First, existing work predominantly focuses on the textual modality, entirely overlooking critical paralinguistic features including intonation, prosody, and rhythm in speech, which are essential for conveying character emotions and shaping vivid identities. Second, the speech-based role-playing domain suffers from a long-standing lack of standardized evaluation benchmarks. Most current spoken dialogue datasets target only fundamental capability assessments, featuring thinly sketched or ill-defined character profiles. Consequently, they fail to effectively quantify model performance on core competencies like long-term persona consistency. To address this critical gap, we introduce VoxRole, the first comprehensive benchmark specifically designed for the evaluation of speech-based RPCAs. The benchmark comprises 13335 multi-turn dialogues, totaling 65.6 hours of speech from 1228 unique characters across 261 movies. To construct this resource, we propose a novel two-stage automated pipeline that first aligns movie audio with scripts and subsequently employs an LLM to systematically build multi-dimensional profiles for each character. Leveraging VoxRole, we conduct a multi-dimensional evaluation of contemporary spoken dialogue models, revealing crucial insights into their respective strengths and limitations in maintaining persona consistency.
>
---
#### [new 011] Accelerated Interactive Auralization of Highly Reverberant Spaces using Graphics Hardware
- **分类: eess.AS; cs.SD**

- **简介: 论文提出基于GPU加速的实时多通道听觉化系统，解决高混响空间声学合成延迟问题，通过GPU卷积与反馈消除整合，实现低延迟的交互式虚拟声场重建。**

- **链接: [http://arxiv.org/pdf/2509.04390v1](http://arxiv.org/pdf/2509.04390v1)**

> **作者:** Hannes Rosseel; Toon van Waterschoot
>
> **备注:** 8 pages, 6 figures, submitted to Journal of the Audio Engineering Society
>
> **摘要:** Interactive acoustic auralization allows users to explore virtual acoustic environments in real-time, enabling the acoustic recreation of concert hall or Historical Worship Spaces (HWS) that are either no longer accessible, acoustically altered, or impractical to visit. Interactive acoustic synthesis requires real-time convolution of input signals with a set of synthesis filters that model the space-time acoustic response of the space. The acoustics in concert halls and HWS are both characterized by a long reverberation time, resulting in synthesis filters containing many filter taps. As a result, the convolution process can be computationally demanding, introducing significant latency that limits the real-time interactivity of the auralization system. In this paper, the implementation of a real-time multichannel loudspeaker-based auralization system is presented. This system is capable of synthesizing the acoustics of highly reverberant spaces in real-time using GPU-acceleration. A comparison between traditional CPU-based convolution and GPU-accelerated convolution is presented, showing that the latter can achieve real-time performance with significantly lower latency. Additionally, the system integrates acoustic synthesis with acoustic feedback cancellation on the GPU, creating a unified loudspeaker-based auralization framework that minimizes processing latency.
>
---
#### [new 012] Crossing the Species Divide: Transfer Learning from Speech to Animal Sounds
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; 68T07; I.5.4; I.2.6; H.5.5**

- **简介: 该论文研究自监督语音模型（如HuBERT、WavLM）在生物声学检测分类中的迁移学习效果，分析时间信息及噪声影响，结果与微调模型相当，证明其在生物声学研究中的潜力。（99字）**

- **链接: [http://arxiv.org/pdf/2509.04166v1](http://arxiv.org/pdf/2509.04166v1)**

> **作者:** Jules Cauzinille; Marius Miron; Olivier Pietquin; Masato Hagiwara; Ricard Marxer; Arnaud Rey; Benoit Favre
>
> **备注:** 5 pages, 3 figures, uses dcase2025.sty, submitted to DCASE 2025
>
> **摘要:** Self-supervised speech models have demonstrated impressive performance in speech processing, but their effectiveness on non-speech data remains underexplored. We study the transfer learning capabilities of such models on bioacoustic detection and classification tasks. We show that models such as HuBERT, WavLM, and XEUS can generate rich latent representations of animal sounds across taxa. We analyze the models properties with linear probing on time-averaged representations. We then extend the approach to account for the effect of time-wise information with other downstream architectures. Finally, we study the implication of frequency range and noise on performance. Notably, our results are competitive with fine-tuned bioacoustic pre-trained models and show the impact of noise-robust pre-training setups. These findings highlight the potential of speech-based self-supervised learning as an efficient framework for advancing bioacoustic research.
>
---
#### [new 013] LibriQuote: A Speech Dataset of Fictional Character Utterances for Expressive Zero-Shot Speech Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出LibriQuote数据集，用于零样本语音合成任务，解决现有数据集规模小、表达性语音比例不明的问题。数据集包含12.7K小时非表达性语音及5.3K小时角色引用表达性语音，设计测试集评估系统生成表达性语音能力，并验证微调后系统效果提升。**

- **链接: [http://arxiv.org/pdf/2509.04072v1](http://arxiv.org/pdf/2509.04072v1)**

> **作者:** Gaspard Michel; Elena V. Epure; Christophe Cerisara
>
> **摘要:** Text-to-speech (TTS) systems have recently achieved more expressive and natural speech synthesis by scaling to large speech datasets. However, the proportion of expressive speech in such large-scale corpora is often unclear. Besides, existing expressive speech corpora are typically smaller in scale and primarily used for benchmarking TTS systems. In this paper, we introduce the LibriQuote dataset, an English corpus derived from read audiobooks, designed for both fine-tuning and benchmarking expressive zero-shot TTS system. The training dataset includes 12.7K hours of read, non-expressive speech and 5.3K hours of mostly expressive speech drawn from character quotations. Each utterance in the expressive subset is supplemented with the context in which it was written, along with pseudo-labels of speech verbs and adverbs used to describe the quotation (\textit{e.g. ``he whispered softly''}). Additionally, we provide a challenging 7.5 hour test set intended for benchmarking TTS systems: given a neutral reference speech as input, we evaluate system's ability to synthesize an expressive utterance while preserving reference timbre. We validate qualitatively the test set by showing that it covers a wide range of emotions compared to non-expressive speech, along with various accents. Extensive subjective and objective evaluations show that fine-tuning a baseline TTS system on LibriQuote significantly improves its synthesized speech intelligibility, and that recent systems fail to synthesize speech as expressive and natural as the ground-truth utterances. The dataset and evaluation code are freely available. Audio samples can be found at https://libriquote.github.io/.
>
---
#### [new 014] PARCO: Phoneme-Augmented Robust Contextual ASR via Contrastive Entity Disambiguation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文针对ASR中命名实体识别与同音词歧义问题，提出PARCO方法。通过音素感知编码、对比实体消歧等技术，提升多token实体识别准确率，减少误判，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.04357v1](http://arxiv.org/pdf/2509.04357v1)**

> **作者:** Jiajun He; Naoki Sawada; Koichi Miyazaki; Tomoki Toda
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Automatic speech recognition (ASR) systems struggle with domain-specific named entities, especially homophones. Contextual ASR improves recognition but often fails to capture fine-grained phoneme variations due to limited entity diversity. Moreover, prior methods treat entities as independent tokens, leading to incomplete multi-token biasing. To address these issues, we propose Phoneme-Augmented Robust Contextual ASR via COntrastive entity disambiguation (PARCO), which integrates phoneme-aware encoding, contrastive entity disambiguation, entity-level supervision, and hierarchical entity filtering. These components enhance phonetic discrimination, ensure complete entity retrieval, and reduce false positives under uncertainty. Experiments show that PARCO achieves CER of 4.22% on Chinese AISHELL-1 and WER of 11.14% on English DATA2 under 1,000 distractors, significantly outperforming baselines. PARCO also demonstrates robust gains on out-of-domain datasets like THCHS-30 and LibriSpeech.
>
---
## 更新

#### [replaced 001] Beyond-Voice: Towards Continuous 3D Hand Pose Tracking on Commercial Home Assistant Devices
- **分类: cs.SD; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2306.17477v3](http://arxiv.org/pdf/2306.17477v3)**

> **作者:** Yin Li; Rohan Reddy; Cheng Zhang; Rajalakshmi Nandakumar
>
> **备注:** Accepted by IPSN 2024
>
> **摘要:** The surging popularity of home assistants and their voice user interface (VUI) have made them an ideal central control hub for smart home devices. However, current form factors heavily rely on VUI, which poses accessibility and usability issues; some latest ones are equipped with additional cameras and displays, which are costly and raise privacy concerns. These concerns jointly motivate Beyond-Voice, a novel high-fidelity acoustic sensing system that allows commodity home assistant devices to track and reconstruct hand poses continuously. It transforms the home assistant into an active sonar system using its existing onboard microphones and speakers. We feed a high-resolution range profile to the deep learning model that can analyze the motions of multiple body parts and predict the 3D positions of 21 finger joints, bringing the granularity for acoustic hand tracking to the next level. It operates across different environments and users without the need for personalized training data. A user study with 11 participants in 3 different environments shows that Beyond-Voice can track joints with an average mean absolute error of 16.47mm without any training data provided by the testing subject.
>
---
#### [replaced 002] CoPlay: Audio-agnostic Cognitive Scaling for Acoustic Sensing
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2403.10796v2](http://arxiv.org/pdf/2403.10796v2)**

> **作者:** Yin Li; Bo Liu; Rajalakshmi Nanadakumar
>
> **备注:** ICCCN'25
>
> **摘要:** Acoustic sensing manifests great potential in various applications that encompass health monitoring, gesture interface and imaging by leveraging the speakers and microphones on smart devices. However, in ongoing research and development in acoustic sensing, one problem is often overlooked: the same speaker, when used concurrently for sensing and other traditional applications (like playing music), could cause interference in both making it impractical to use in the real world. The strong ultrasonic sensing signals mixed with music would overload the speaker's mixer. To confront this issue of overloaded signals, current solutions are clipping or down-scaling, both of which affect the music playback quality and also sensing range and accuracy. To address this challenge, we propose CoPlay, a deep learning based optimization algorithm to cognitively adapt the sensing signal. It can 1) maximize the sensing signal magnitude within the available bandwidth left by the concurrent music to optimize sensing range and accuracy and 2) minimize any consequential frequency distortion that can affect music playback. In this work, we design a deep learning model and test it on common types of sensing signals (sine wave or Frequency Modulated Continuous Wave FMCW) as inputs with various agnostic concurrent music and speech. First, we evaluated the model performance to show the quality of the generated signals. Then we conducted field studies of downstream acoustic sensing tasks in the real world. A study with 12 users proved that respiration monitoring and gesture recognition using our adapted signal achieve similar accuracy as no-concurrent-music scenarios, while clipping or down-scaling manifests worse accuracy. A qualitative study also manifests that the music play quality is not degraded, unlike traditional clipping or down-scaling methods.
>
---
#### [replaced 003] AImoclips: A Benchmark for Evaluating Emotion Conveyance in Text-to-Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.00813v2](http://arxiv.org/pdf/2509.00813v2)**

> **作者:** Gyehun Go; Satbyul Han; Ahyeon Choi; Eunjin Choi; Juhan Nam; Jeong Mi Park
>
> **备注:** to be published in HCMIR25: 3rd Workshop on Human-Centric Music Information Research
>
> **摘要:** Recent advances in text-to-music (TTM) generation have enabled controllable and expressive music creation using natural language prompts. However, the emotional fidelity of TTM systems remains largely underexplored compared to human preference or text alignment. In this study, we introduce AImoclips, a benchmark for evaluating how well TTM systems convey intended emotions to human listeners, covering both open-source and commercial models. We selected 12 emotion intents spanning four quadrants of the valence-arousal space, and used six state-of-the-art TTM systems to generate over 1,000 music clips. A total of 111 participants rated the perceived valence and arousal of each clip on a 9-point Likert scale. Our results show that commercial systems tend to produce music perceived as more pleasant than intended, while open-source systems tend to perform the opposite. Emotions are more accurately conveyed under high-arousal conditions across all models. Additionally, all systems exhibit a bias toward emotional neutrality, highlighting a key limitation in affective controllability. This benchmark offers valuable insights into model-specific emotion rendering characteristics and supports future development of emotionally aligned TTM systems.
>
---
#### [replaced 004] MultiGen: Child-Friendly Multilingual Speech Generator with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.08715v3](http://arxiv.org/pdf/2508.08715v3)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** 5 pages
>
> **摘要:** Generative speech models have demonstrated significant potential in improving human-machine interactions, offering valuable real-world applications such as language learning for children. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiGen, a multilingual speech generation model with child-friendly interaction, leveraging LLM architecture for speech generation tailored for low-resource languages. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, which can be used to facilitate young children's communication with AI systems through culturally relevant context in three low-resource languages: Singaporean accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiGen compared to baseline methods.
>
---
#### [replaced 005] Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated Piano Hand Motion Synthesis
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.09885v2](http://arxiv.org/pdf/2504.09885v2)**

> **作者:** Zihao Liu; Mingwen Ou; Zunnan Xu; Jiaqi Huang; Haonan Han; Ronghui Li; Xiu Li
>
> **备注:** 15 pages, 7 figures, Accepted to ACMMM 2025
>
> **摘要:** Automating the synthesis of coordinated bimanual piano performances poses significant challenges, particularly in capturing the intricate choreography between the hands while preserving their distinct kinematic signatures. In this paper, we propose a dual-stream neural framework designed to generate synchronized hand gestures for piano playing from audio input, addressing the critical challenge of modeling both hand independence and coordination. Our framework introduces two key innovations: (i) a decoupled diffusion-based generation framework that independently models each hand's motion via dual-noise initialization, sampling distinct latent noise for each while leveraging a shared positional condition, and (ii) a Hand-Coordinated Asymmetric Attention (HCAA) mechanism suppresses symmetric (common-mode) noise to highlight asymmetric hand-specific features, while adaptively enhancing inter-hand coordination during denoising. Comprehensive evaluations demonstrate that our framework outperforms existing state-of-the-art methods across multiple metrics. Our project is available at https://monkek123king.github.io/S2C_page/.
>
---
#### [replaced 006] Auto-Regressive vs Flow-Matching: a Comparative Study of Modeling Paradigms for Text-to-Music Generation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08570v3](http://arxiv.org/pdf/2506.08570v3)**

> **作者:** Or Tal; Felix Kreuk; Yossi Adi
>
> **摘要:** Recent progress in text-to-music generation has enabled models to synthesize high-quality musical segments, full compositions, and even respond to fine-grained control signals, e.g. chord progressions. State-of-the-art (SOTA) systems differ significantly in many dimensions, such as training datasets, modeling paradigms, and architectural choices. This diversity complicates efforts to evaluate models fairly and identify which design choices influence performance the most. While factors like data and architecture are important, in this study we focus exclusively on the modeling paradigm. We conduct a systematic empirical analysis to isolate its effects, offering insights into associated trade-offs and emergent behaviors that can guide future text-to-music generation systems. Specifically, we compare the two arguably most common modeling paradigms: auto-regressive decoding and conditional flow-matching. We conduct a controlled comparison by training all models from scratch using identical datasets, training configurations, and similar backbone architectures. Performance is evaluated across multiple axes, including generation quality, robustness to inference configurations, scalability, adherence to both textual and temporally aligned conditioning, and editing capabilities in the form of audio inpainting. This comparative study sheds light on distinct strengths and limitations of each paradigm, providing actionable insights that can inform future architectural and training decisions in the evolving landscape of text-to-music generation. Audio sampled examples are available at: https://huggingface.co/spaces/ortal1602/ARvsFM
>
---
#### [replaced 007] EZhouNet:A framework based on graph neural network and anchor interval for the respiratory sound event detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.01153v2](http://arxiv.org/pdf/2509.01153v2)**

> **作者:** Yun Chu; Qiuhao Wang; Enze Zhou; Qian Liu; Gang Zheng
>
> **摘要:** Auscultation is a key method for early diagnosis of respiratory and pulmonary diseases, relying on skilled healthcare professionals. However, the process is often subjective, with variability between experts. As a result, numerous deep learning-based automatic classification methods have emerged, most of which focus on respiratory sound classification. In contrast, research on respiratory sound event detection remains limited. Existing sound event detection methods typically rely on frame-level predictions followed by post-processing to generate event-level outputs, making interval boundaries challenging to learn directly. Furthermore, many approaches can only handle fixed-length audio, limiting their applicability to variable-length respiratory sounds. Additionally, the impact of respiratory sound location information on detection performance has not been extensively explored. To address these issues, we propose a graph neural network-based framework with anchor intervals, capable of handling variable-length audio and providing more precise temporal localization for abnormal respiratory sound events. Our method improves both the flexibility and applicability of respiratory sound detection. Experiments on the SPRSound 2024 and HF Lung V1 datasets demonstrate the effectiveness of the proposed approach, and incorporating respiratory position information enhances the discrimination between abnormal sounds. The reference implementation is available at https://github.com/chumingqian/EzhouNet.
>
---
#### [replaced 008] AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation
- **分类: cs.SD; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.02349v2](http://arxiv.org/pdf/2509.02349v2)**

> **作者:** Lu Wang; Hao Chen; Siyu Wu; Zhiyue Wu; Hao Zhou; Chengfeng Zhang; Ting Wang; Haodi Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been widely applied in speech and music. This tendency has led to a focus on audio tokenization for Large Models (LMs). Unlike semantic-only text tokens, audio tokens must both capture global semantic content and preserve fine-grained acoustic details. Moreover, they provide a discrete method for speech and music that can be effectively integrated into MLLMs. However, existing research is unsuitable in the definitions of semantic tokens and acoustic tokens. In addition, the evaluation of different codecs typically concentrates on specific domains or tasks, such as reconstruction or Automatic Speech Recognition (ASR) task, which prevents fair and comprehensive comparisons. To address these problems, this paper provides suitable definitions for semantic and acoustic tokens and introduces a systematic evaluation framework. This framework allows for a comprehensive assessment of codecs' capabilities which evaluate across four dimensions: audio reconstruction metric, codebook index (ID) stability, decoder-only transformer perplexity, and performance on downstream probe tasks. Our results show the correctness of the provided suitable definitions and the correlation among reconstruction metrics, codebook ID stability, downstream probe tasks and perplexity.
>
---
#### [replaced 009] NADI 2025: The First Multidialectal Arabic Speech Processing Shared Task
- **分类: cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.02038v2](http://arxiv.org/pdf/2509.02038v2)**

> **作者:** Bashar Talafha; Hawau Olamide Toyin; Peter Sullivan; AbdelRahim Elmadany; Abdurrahman Juma; Amirbek Djanibekov; Chiyu Zhang; Hamad Alshehhi; Hanan Aldarmaki; Mustafa Jarrar; Nizar Habash; Muhammad Abdul-Mageed
>
> **摘要:** We present the findings of the sixth Nuanced Arabic Dialect Identification (NADI 2025) Shared Task, which focused on Arabic speech dialect processing across three subtasks: spoken dialect identification (Subtask 1), speech recognition (Subtask 2), and diacritic restoration for spoken dialects (Subtask 3). A total of 44 teams registered, and during the testing phase, 100 valid submissions were received from eight unique teams. The distribution was as follows: 34 submissions for Subtask 1 "five teams{\ae}, 47 submissions for Subtask 2 "six teams", and 19 submissions for Subtask 3 "two teams". The best-performing systems achieved 79.8% accuracy on Subtask 1, 35.68/12.20 WER/CER (overall average) on Subtask 2, and 55/13 WER/CER on Subtask 3. These results highlight the ongoing challenges of Arabic dialect speech processing, particularly in dialect identification, recognition, and diacritic restoration. We also summarize the methods adopted by participating teams and briefly outline directions for future editions of NADI.
>
---
#### [replaced 010] Speech Intelligibility Assessment with Uncertainty-Aware Whisper Embeddings and sLSTM
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.03013v2](http://arxiv.org/pdf/2509.03013v2)**

> **作者:** Ryandhimas E. Zezario; Dyah A. M. G. Wisnu; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to APSIPA ASC 2025
>
> **摘要:** Non-intrusive speech intelligibility prediction remains challenging due to variability in speakers, noise conditions, and subjective perception. We propose an uncertainty-aware approach that leverages Whisper embeddings in combination with statistical features, specifically the mean, standard deviation, and entropy computed across the embedding dimensions. The entropy, computed via a softmax over the feature dimension, serves as a proxy for uncertainty, complementing global information captured by the mean and standard deviation. To model the sequential structure of speech, we adopt a scalar long short-term memory (sLSTM) network, which efficiently captures long-range dependencies. Building on this foundation, we propose iMTI-Net, an improved multi-target intelligibility prediction network that integrates convolutional neural network (CNN) and sLSTM components within a multitask learning framework. It jointly predicts human intelligibility scores and machine-based word error rates (WER) from Google ASR and Whisper. Experimental results show that iMTI-Net outperforms the original MTI-Net across multiple evaluation metrics, demonstrating the effectiveness of incorporating uncertainty-aware features and the CNN-sLSTM architecture.
>
---
#### [replaced 011] CUHK-EE Systems for the vTAD Challenge at NCMMSC 2025
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.23266v2](http://arxiv.org/pdf/2507.23266v2)**

> **作者:** Aemon Yat Fei Chiu; Jingyu Li; Yusheng Tian; Guangyan Zhang; Tan Lee
>
> **备注:** Accepted at China's 20th National Conference on Man-Machine Speech Communication (NCMMSC 2025)
>
> **摘要:** This paper presents the Voice Timbre Attribute Detection (vTAD) systems developed by the Digital Signal Processing & Speech Technology Laboratory (DSP&STL) of the Department of Electronic Engineering (EE) at The Chinese University of Hong Kong (CUHK) for the 20th National Conference on Human-Computer Speech Communication (NCMMSC 2025) vTAD Challenge. The proposed systems leverage WavLM-Large embeddings with attentive statistical pooling (ASTP) to extract robust speaker representations, followed by two variants of Diff-Net, i.e., Feed-Forward Neural Network (FFN) and Squeeze-and-Excitation-enhanced Residual FFN (SE-ResFFN), to compare timbre attribute intensities between utterance pairs. Experimental results demonstrate that the WavLM-Large+FFN system generalises better to unseen speakers, achieving 77.96% accuracy and 21.79% equal error rate (EER), while the WavLM-Large+SE-ResFFN model excels in the 'Seen' setting with 94.42% accuracy and 5.49% EER. These findings highlight a trade-off between model complexity and generalisation, and underscore the importance of architectural choices in fine-grained speaker modelling. Our analysis also reveals the impact of speaker identity, annotation subjectivity, and data imbalance on system performance, pointing to future directions for improving robustness and fairness in timbre attribute detection.
>
---
#### [replaced 012] FireRedTTS-2: Towards Long Conversational Speech Generation for Podcast and Chatbot
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.02020v2](http://arxiv.org/pdf/2509.02020v2)**

> **作者:** Kun Xie; Feiyu Shen; Junjie Li; Fenglong Xie; Xu Tang; Yao Hu
>
> **摘要:** Current dialogue generation approaches typically require the complete dialogue text before synthesis and produce a single, inseparable speech containing all voices, making them unsuitable for interactive chat; moreover, they suffer from unstable synthesis, inaccurate speaker transitions, and incoherent prosody. In this work, we present FireRedTTS-2, a long-form streaming TTS system for multi-speaker dialogue generation, delivering stable, natural speech with reliable speaker switching and context-aware prosody. A new 12.5Hz streaming speech tokenizer accelerates training and inference, extends maximum dialogue length, encodes richer semantics to stabilize text-to-token modeling and supports high-fidelity streaming generation for real-time applications. We adopt a text-speech interleaved format, concatenating speaker-labeled text with aligned speech tokens in chronological order, and model it with a dual-transformer: a large decoder-only transformer predicts tokens at the first layer, and a smaller one completes subsequent layers. Experimental results show that FireRedTTS-2 integrates seamlessly with chat frameworks and, with minimal fine-tuning, produces emotionally expressive speech guided by implicit contextual cues. In podcast generation, it surpasses existing systems including MoonCast, Zipvoice-Dialogue, and MOSS-TTSD in objective intelligibility, speaker-turn reliability, and perceived naturalness with context-consistent prosody. Our demos are available at https://fireredteam.github.io/demos/firered_tts_2.
>
---
