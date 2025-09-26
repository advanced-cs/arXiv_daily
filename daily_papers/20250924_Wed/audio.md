# 音频 cs.SD;  eess.SP

- **最新发布 29 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] A Dimensional Approach to Canine Bark Analysis for Assistance Dog Seizure Signaling
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出一种基于二维情绪空间（唤醒-愉悦）的回归方法，用于分析导盲犬的吠叫信号。针对数据稀疏和伦理限制问题，采用改进的Siamese网络学习吠叫间的连续距离，减少分类误差，验证了在有限数据下分析犬吠的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18375v1](http://arxiv.org/pdf/2509.18375v1)**

> **作者:** Hailin Song; Shelley Brady; Tomás Ward; Alan F. Smeaton
>
> **摘要:** Standard classification of canine vocalisations is severely limited for assistance dogs, where sample data is sparse and variable across dogs and where capture of the full range of bark types is ethically constrained. We reframe this problem as a continuous regression task within a two-dimensional arousal-valence space. Central to our approach is an adjusted Siamese Network trained not on binary similarity, but on the ordinal and numeric distance between input sample pairs. Trained on a public dataset, our model reduces Turn-around Percentage by up to 50% on the challenging valence dimension compared to a regression baseline. Qualitative validation on a real-world dataset confirms the learned space is semantically meaningful, establishing a proof-of-concept for analysing canine barking under severe data limitations.
>
---
#### [new 002] Identifying birdsong syllables without labelled data
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出一种无需标注数据的鸟类鸣叫音节识别方法，属于语音分解任务。旨在解决传统方法依赖标注数据、适用性受限的问题。通过事件检测、聚类和匹配追踪实现音节序列自动分解，并验证了个体识别能力。**

- **链接: [http://arxiv.org/pdf/2509.18412v1](http://arxiv.org/pdf/2509.18412v1)**

> **作者:** Mélisande Teng; Julien Boussard; David Rolnick; Hugo Larochelle
>
> **摘要:** Identifying sequences of syllables within birdsongs is key to tackling a wide array of challenges, including bird individual identification and better understanding of animal communication and sensory-motor learning. Recently, machine learning approaches have demonstrated great potential to alleviate the need for experts to label long audio recordings by hand. However, they still typically rely on the availability of labelled data for model training, restricting applicability to a few species and datasets. In this work, we build the first fully unsupervised algorithm to decompose birdsong recordings into sequences of syllables. We first detect syllable events, then cluster them to extract templates --syllable representations-- before performing matching pursuit to decompose the recording as a sequence of syllables. We evaluate our automatic annotations against human labels on a dataset of Bengalese finch songs and find that our unsupervised method achieves high performance. We also demonstrate that our approach can distinguish individual birds within a species through their unique vocal signatures, for both Bengalese finches and another species, the great tit.
>
---
#### [new 003] Scattering Transformer: A Training-Free Transformer Architecture for Heart Murmur Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出一种无需训练的Scattering Transformer模型，用于心音杂音检测。针对医疗资源受限场景下深度学习方法计算量大的问题，采用小波散射网络结合Transformer结构，在CirCor DigiScope数据集上取得与先进方法相当的性能。**

- **链接: [http://arxiv.org/pdf/2509.18424v1](http://arxiv.org/pdf/2509.18424v1)**

> **作者:** Rami Zewail
>
> **摘要:** In an attempt to address the need for skilled clinicians in heart sound interpretation, recent research efforts on automating cardiac auscultation have explored deep learning approaches. The majority of these approaches have been based on supervised learning that is always challenged in occasions where training data is limited. More recently, there has been a growing interest in potentials of pre-trained self-supervised audio foundation models for biomedical end tasks. Despite exhibiting promising results, these foundational models are typically computationally intensive. Within the context of automatic cardiac auscultation, this study explores a lightweight alternative to these general-purpose audio foundation models by introducing the Scattering Transformer, a novel, training-free transformer architecture for heart murmur detection. The proposed method leverages standard wavelet scattering networks by introducing contextual dependencies in a transformer-like architecture without any backpropagation. We evaluate our approach on the public CirCor DigiScope dataset, directly comparing it against leading general-purpose foundational models. The Scattering Transformer achieves a Weighted Accuracy(WAR) of 0.786 and an Unweighted Average Recall(UAR) of 0.697, demonstrating performance highly competitive with contemporary state of the art methods. This study establishes the Scattering Transformer as a viable and promising alternative in resource-constrained setups.
>
---
#### [new 004] Explore the Reinforcement Learning for the LLM based ASR and TTS system
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音识别（ASR）与语音合成（TTS）任务，旨在探索强化学习（RL）在基于大语言模型的音频系统中的应用。提出了一个轻量级RL框架，并通过实验验证了RL在提升ASR和TTS性能上的有效性，尤其在数据有限的情况下。**

- **链接: [http://arxiv.org/pdf/2509.18569v1](http://arxiv.org/pdf/2509.18569v1)**

> **作者:** Changfeng Gao; Yabin Li; Keyu An; Zhifu Gao; Zhihao Du; Han Zhao; Xiangang Li
>
> **摘要:** In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps.
>
---
#### [new 005] StereoFoley: Object-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文提出StereoFoley，一个视频到立体声音频生成框架，旨在解决现有模型在语义对齐、时间同步和空间定位上的不足。通过构建基础模型、合成数据管道及引入新评估指标，实现了对象感知的高质量立体声生成。**

- **链接: [http://arxiv.org/pdf/2509.18272v1](http://arxiv.org/pdf/2509.18272v1)**

> **作者:** Tornike Karchkhadze; Kuan-Lin Chen; Mojtaba; Heydari; Robert Henzel; Alessandro Toso; Mehrez Souden; Joshua Atkins
>
> **摘要:** We present StereoFoley, a video-to-audio generation framework that produces semantically aligned, temporally synchronized, and spatially accurate stereo sound at 48 kHz. While recent generative video-to-audio models achieve strong semantic and temporal fidelity, they largely remain limited to mono or fail to deliver object-aware stereo imaging, constrained by the lack of professionally mixed, spatially accurate video-to-audio datasets. First, we develop and train a base model that generates stereo audio from video, achieving state-of-the-art in both semantic accuracy and synchronization. Next, to overcome dataset limitations, we introduce a synthetic data generation pipeline that combines video analysis, object tracking, and audio synthesis with dynamic panning and distance-based loudness controls, enabling spatially accurate object-aware sound. Finally, we fine-tune the base model on this synthetic dataset, yielding clear object-audio correspondence. Since no established metrics exist, we introduce stereo object-awareness measures and validate it through a human listening study, showing strong correlation with perception. This work establishes the first end-to-end framework for stereo object-aware video-to-audio generation, addressing a critical gap and setting a new benchmark in the field.
>
---
#### [new 006] Scalable Evaluation for Audio Identification via Synthetic Latent Fingerprint Generation
- **分类: cs.SD; cs.IR; eess.AS; H.5.5; I.2.6**

- **简介: 该论文针对音频指纹识别的大规模评估难题，提出一种无需真实音频的合成潜在指纹方法。通过训练Rectified Flow模型生成逼真的指纹数据，模拟大规模检索性能，验证其与真实数据趋势一致，实现了音频指纹系统的可扩展性评估。**

- **链接: [http://arxiv.org/pdf/2509.18620v1](http://arxiv.org/pdf/2509.18620v1)**

> **作者:** Aditya Bhattacharjee; Marco Pasini; Emmanouil Benetos
>
> **备注:** Under review for International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Barcelona, 2026
>
> **摘要:** The evaluation of audio fingerprinting at a realistic scale is limited by the scarcity of large public music databases. We present an audio-free approach that synthesises latent fingerprints which approximate the distribution of real fingerprints. Our method trains a Rectified Flow model on embeddings extracted by pre-trained neural audio fingerprinting systems. The synthetic fingerprints generated using our system act as realistic distractors and enable the simulation of retrieval performance at a large scale without requiring additional audio. We assess the fidelity of synthetic fingerprints by comparing the distributions to real data. We further benchmark the retrieval performances across multiple state-of-the-art audio fingerprinting frameworks by augmenting real reference databases with synthetic distractors, and show that the scaling trends obtained with synthetic distractors closely track those obtained with real distractors. Finally, we scale the synthetic distractor database to model retrieval performance for very large databases, providing a practical metric of system scalability that does not depend on access to audio corpora.
>
---
#### [new 007] MNV-17: A High-Quality Performative Mandarin Dataset for Nonverbal Vocalization Recognition in Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出MNV-17，一个高质量的表演型普通话数据集，用于非语言发声识别。旨在解决现有语音识别系统对非语言发声（如笑声、咳嗽）识别能力不足的问题。工作包括构建17类平衡的NV标注数据，并在主流ASR模型上进行基准测试。**

- **链接: [http://arxiv.org/pdf/2509.18196v1](http://arxiv.org/pdf/2509.18196v1)**

> **作者:** Jialong Mai; Jinxin Ji; Xiaofen Xing; Chen Yang; Weidong Chen; Jingyuan Xing; Xiangmin Xu
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Mainstream Automatic Speech Recognition (ASR) systems excel at transcribing lexical content, but largely fail to recognize nonverbal vocalizations (NVs) embedded in speech, such as sighs, laughs, and coughs. This capability is important for a comprehensive understanding of human communication, as NVs convey crucial emotional and intentional cues. Progress in NV-aware ASR has been hindered by the lack of high-quality, well-annotated datasets. To address this gap, we introduce MNV-17, a 7.55-hour performative Mandarin speech dataset. Unlike most existing corpora that rely on model-based detection, MNV-17's performative nature ensures high-fidelity, clearly articulated NV instances. To the best of our knowledge, MNV-17 provides the most extensive set of nonverbal vocalization categories, comprising 17 distinct and well-balanced classes of common NVs. We benchmarked MNV-17 on four mainstream ASR architectures, evaluating their joint performance on semantic transcription and NV classification. The dataset and the pretrained model checkpoints will be made publicly available to facilitate future research in expressive ASR.
>
---
#### [new 008] An overview of neural architectures for self-supervised audio representation learning from masked spectrograms
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文综述了基于掩码频谱图的自监督音频表征学习方法，重点对比了Transformer、Mamba和xLSTM架构。针对长序列建模问题，研究在统一框架下评估了三种模型在10项下游音频分类任务中的表现，旨在为相关应用提供选择依据。**

- **链接: [http://arxiv.org/pdf/2509.18691v1](http://arxiv.org/pdf/2509.18691v1)**

> **作者:** Sarthak Yadav; Sergios Theodoridis; Zheng-Hua Tan
>
> **摘要:** In recent years, self-supervised learning has amassed significant interest for training deep neural representations without labeled data. One such self-supervised learning approach is masked spectrogram modeling, where the objective is to learn semantically rich contextual representations by predicting removed or hidden portions of the input audio spectrogram. With the Transformer neural architecture at its core, masked spectrogram modeling has emerged as the prominent approach for learning general purpose audio representations, a.k.a. audio foundation models. Meanwhile, addressing the issues of the Transformer architecture, in particular the underlying Scaled Dot-product Attention operation, which scales quadratically with input sequence length, has led to renewed interest in recurrent sequence modeling approaches. Among them, Selective structured state space models (such as Mamba) and extended Long Short-Term Memory (xLSTM) are the two most promising approaches which have experienced widespread adoption. While the body of work on these two topics continues to grow, there is currently a lack of an adequate overview encompassing the intersection of these topics. In this paper, we present a comprehensive overview of the aforementioned research domains, covering masked spectrogram modeling and the previously mentioned neural sequence modeling architectures, Mamba and xLSTM. Further, we compare Transformers, Mamba and xLSTM based masked spectrogram models in a unified, reproducible framework on ten diverse downstream audio classification tasks, which will help interested readers to make informed decisions regarding suitability of the evaluated approaches to adjacent applications.
>
---
#### [new 009] XMUspeech Systems for the ASVspoof 5 Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音反伪造任务，旨在解决语音深度伪造检测问题。作者改进了模型输入长度，并结合自监督模型与AMFF方法，优化损失函数配置，提升了检测性能。**

- **链接: [http://arxiv.org/pdf/2509.18102v1](http://arxiv.org/pdf/2509.18102v1)**

> **作者:** Wangjie Li; Xingjia Xie; Yishuang Li; Wenhao Guan; Kaidi Wang; Pengyu Ren; Lin Li; Qingyang Hong
>
> **摘要:** In this paper, we present our submitted XMUspeech systems to the speech deepfake detection track of the ASVspoof 5 Challenge. Compared to previous challenges, the audio duration in ASVspoof 5 database has significantly increased. And we observed that merely adjusting the input audio length can substantially improve system performance. To capture artifacts at multiple levels, we explored the performance of AASIST, HM-Conformer, Hubert, and Wav2vec2 with various input features and loss functions. Specifically, in order to obtain artifact-related information, we trained self-supervised models on the dataset containing spoofing utterances as the feature extractors. And we applied an adaptive multi-scale feature fusion (AMFF) method to integrate features from multiple Transformer layers with the hand-crafted feature to enhance the detection capability. In addition, we conducted extensive experiments on one-class loss functions and provided optimized configurations to better align with the anti-spoofing task. Our fusion system achieved a minDCF of 0.4783 and an EER of 20.45% in the closed condition, and a minDCF of 0.2245 and an EER of 9.36% in the open condition.
>
---
#### [new 010] Pay More Attention To Audio: Mitigating Imbalance of Cross-Modal Attention in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文针对大音视语言模型（LALM）中音频-文本注意力不平衡问题，提出训练免费方法MATA。通过动态增强音频token的注意力权重，提升音频推理性能，在MMAU和MMAR基准上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.18816v1](http://arxiv.org/pdf/2509.18816v1)**

> **作者:** Junyu Wang; Ziyang Ma; Zhengding Luo; Tianrui Wang; Meng Ge; Xiaobao Wang; Longbiao Wang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large Audio-Language Models (LALMs) often suffer from audio-textual attention imbalance, prioritizing text over acoustic information, particularly in the multi-modal fusion layers of the Transformer architecture. This bias hinders their ability to fully utilize acoustic cues, causing suboptimal performance on audio reasoning tasks. To mitigate this, we propose \textbf{MATA}, a novel training-free method that dynamically pushes LALMs to pay \textbf{M}ore \textbf{A}ttention \textbf{T}o \textbf{A}udio tokens within the self-attention mechanism. Specifically, MATA intervenes post raw attention scoring, targeting only the last token in intermediate layers without introducing additional parameters or computational overhead. Experiments on the MMAU and MMAR benchmarks confirm MATA's effectiveness, with consistent performance gains. Notably, on MMAR, MATA enables an open-source model to surpass the proprietary Gemini 2.0 Flash for the first time. Our work provides an efficient solution to mitigate attention bias and opens a new research direction for enhancing the audio-processing capabilities of multi-modal models.
>
---
#### [new 011] Finding My Voice: Generative Reconstruction of Disordered Speech for Automated Clinical Evaluation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出ChiReSSD，用于重建儿童语音障碍者的语音，在抑制发音错误的同时保留说话人身份。通过解耦风格的TTS方法，实现了语音质量与身份保持的提升，并在临床评估中表现出良好的自动评分相关性。**

- **链接: [http://arxiv.org/pdf/2509.19231v1](http://arxiv.org/pdf/2509.19231v1)**

> **作者:** Karen Rosero; Eunjung Yeo; David R. Mortensen; Cortney Van't Slot; Rami R. Hallac; Carlos Busso
>
> **摘要:** We present ChiReSSD, a speech reconstruction framework that preserves children speaker's identity while suppressing mispronunciations. Unlike prior approaches trained on healthy adult speech, ChiReSSD adapts to the voices of children with speech sound disorders (SSD), with particular emphasis on pitch and prosody. We evaluate our method on the STAR dataset and report substantial improvements in lexical accuracy and speaker identity preservation. Furthermore, we automatically predict the phonetic content in the original and reconstructed pairs, where the proportion of corrected consonants is comparable to the percentage of correct consonants (PCC), a clinical speech assessment metric. Our experiments show Pearson correlation of 0.63 between automatic and human expert annotations, highlighting the potential to reduce the manual transcription burden. In addition, experiments on the TORGO dataset demonstrate effective generalization for reconstructing adult dysarthric speech. Our results indicate that disentangled, style-based TTS reconstruction can provide identity-preserving speech across diverse clinical populations.
>
---
#### [new 012] MECap-R1: Emotion-aware Policy with Reinforcement Learning for Multimodal Emotion Captioning
- **分类: cs.SD**

- **简介: 该论文提出MECap-R1，利用强化学习进行多模态情感描述生成。针对语音情感描述任务中传统方法表达不足的问题，采用情感感知奖励策略优化生成多样性与准确性。**

- **链接: [http://arxiv.org/pdf/2509.18729v1](http://arxiv.org/pdf/2509.18729v1)**

> **作者:** Haoqin Sun; Chenyang Lyu; Xiangyu Kong; Shiwan Zhao; Jiaming Zhou; Hui Wang; Aobo Kong; Jinghua Zhao; Longyue Wang; Weihua Luo; Kaifu Zhang; Yong Qin
>
> **摘要:** Speech Emotion Captioning (SEC) has emerged as a notable research direction. The inherent complexity of emotional content in human speech makes it challenging for traditional discrete classification methods to provide an adequate representation. Consequently, utilizing natural language to describe speech emotions presents a novel avenue for more effectively capturing and expressing affect. In this paper, we propose MECap-R1, a pioneering emotion-aware policy with reinforcement learning for multimodal emotion captioning. By employing Group Relative Policy Optimization with emotion-aware reward (Emo-GRPO), the framework precisely captures the emotion and semantic features, thereby addressing the shortcomings of rigid rules in handling the dynamic and flexible nature of captions. Experimental results on the EmotionTalk dataset demonstrate that MECap-R1 performs well in generating emotion descriptions and achieves substantial gains in both accuracy and diversity.
>
---
#### [new 013] Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在提升自动和弦识别性能。作者提出一种基于大语言模型（LLM）的5阶段推理框架，整合多个MIR工具的信息，通过文本表示与音乐理论知识实现更准确的和弦识别。**

- **链接: [http://arxiv.org/pdf/2509.18700v1](http://arxiv.org/pdf/2509.18700v1)**

> **作者:** Chih-Cheng Chang; Bo-Yu Chen; Lu-Rong Chen; Li Su
>
> **摘要:** Music Information Retrieval (MIR) encompasses a broad range of computational techniques for analyzing and understanding musical content, with recent deep learning advances driving substantial improvements. Building upon these advances, this paper explores how large language models (LLMs) can serve as an integrative bridge to connect and integrate information from multiple MIR tools, with a focus on enhancing automatic chord recognition performance. We present a novel approach that positions text-based LLMs as intelligent coordinators that process and integrate outputs from diverse state-of-the-art MIR tools-including music source separation, key detection, chord recognition, and beat tracking. Our method converts audio-derived musical information into textual representations, enabling LLMs to perform reasoning and correction specifically for chord recognition tasks. We design a 5-stage chain-of-thought framework that allows GPT-4o to systematically analyze, compare, and refine chord recognition results by leveraging music-theoretical knowledge to integrate information across different MIR components. Experimental evaluation on three datasets demonstrates consistent improvements across multiple evaluation metrics, with overall accuracy gains of 1-2.77% on the MIREX metric. Our findings demonstrate that LLMs can effectively function as integrative bridges in MIR pipelines, opening new directions for multi-tool coordination in music information retrieval tasks.
>
---
#### [new 014] Improving Test-Time Performance of RVQ-based Neural Codecs
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对基于RVQ的神经音频编解码器在测试时的合成质量优化问题，提出了一种新的编码算法，用于减少量化误差，提升音频还原效果。**

- **链接: [http://arxiv.org/pdf/2509.19186v1](http://arxiv.org/pdf/2509.19186v1)**

> **作者:** Hyeongju Kim; Junhyeok Lee; Jacob Morton; Juheon Lee; Jinhyeok Yang
>
> **备注:** 5 pages, preprint
>
> **摘要:** The residual vector quantization (RVQ) technique plays a central role in recent advances in neural audio codecs. These models effectively synthesize high-fidelity audio from a limited number of codes due to the hierarchical structure among quantization levels. In this paper, we propose an encoding algorithm to further enhance the synthesis quality of RVQ-based neural codecs at test-time. Firstly, we point out the suboptimal nature of quantized vectors generated by conventional methods. We demonstrate that quantization error can be mitigated by selecting a different set of codes. Subsequently, we present our encoding algorithm, designed to identify a set of discrete codes that achieve a lower quantization error. We then apply the proposed method to pre-trained models and evaluate its efficacy using diverse metrics. Our experimental findings validate that our method not only reduces quantization errors, but also improves synthesis quality.
>
---
#### [new 015] Training Flow Matching Models with Reliable Labels via Self-Purification
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出一种名为SPFM的自净化流匹配方法，用于解决训练数据中标签污染的问题。通过在训练过程中利用模型自身识别不可靠样本，无需预训练模型或额外模块，提升了模型在噪声标签下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.19091v1](http://arxiv.org/pdf/2509.19091v1)**

> **作者:** Hyeongju Kim; Yechan Yu; June Young Yi; Juheon Lee
>
> **备注:** 5 pages, 3 figures, preprint
>
> **摘要:** Training datasets are inherently imperfect, often containing mislabeled samples due to human annotation errors, limitations of tagging models, and other sources of noise. Such label contamination can significantly degrade the performance of a trained model. In this work, we introduce Self-Purifying Flow Matching (SPFM), a principled approach to filtering unreliable data within the flow-matching framework. SPFM identifies suspicious data using the model itself during the training process, bypassing the need for pretrained models or additional modules. Our experiments demonstrate that models trained with SPFM generate samples that accurately adhere to the specified conditioning, even when trained on noisy labels. Furthermore, we validate the robustness of SPFM on the TITW dataset, which consists of in-the-wild speech data, achieving performance that surpasses existing baselines.
>
---
#### [new 016] HarmoniFuse: A Component-Selective and Prompt-Adaptive Framework for Multi-Task Speech Language Modeling
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出HarmoniFuse，一种多任务语音语言建模框架，旨在解决ASR和SER任务因信息需求不同导致的性能下降问题。通过组件选择与提示自适应机制，实现任务相关特征的融合与优化，提升模型在有限数据下的鲁棒性与效果。**

- **链接: [http://arxiv.org/pdf/2509.18570v1](http://arxiv.org/pdf/2509.18570v1)**

> **作者:** Yuke Si; Runyan Yang; Yingying Gao; Junlan Feng; Chao Deng; Shilei Zhang
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** Recent advances in large language models have facilitated the development of unified speech language models (SLMs) capable of supporting multiple speech tasks within a shared architecture. However, tasks such as automatic speech recognition (ASR) and speech emotion recognition (SER) rely on distinct types of information: ASR primarily depends on linguistic content, whereas SER requires the integration of both linguistic and paralinguistic cues. Existing multitask SLMs typically adopt naive parameter sharing or prompt-based conditioning without explicitly modeling the differences in information composition required by each task. Such designs risk task interference and performance degradation, especially under limited data conditions. To address these limitations, we propose HarmoniFuse, a component-selective and prompt-adaptive framework for multi-task speech language modeling. HarmoniFuse is designed to harmonize heterogeneous task demands by selecting and fusing task-relevant components of speech representations. Specifically, it integrates a gated speech encoder to extract task-specific acoustic features and a prompt-adaptive dynamic fusion module to aggregate transformer layers based on task characteristics. In addition, a batch-interleaved training strategy enables leveraging separate ASR and SER datasets without requiring joint annotation. Experimental results demonstrate that HarmoniFuse improves both ASR and SER performance, offering a scalable and robust solution for multitask speech understanding under realistic data constraints.
>
---
#### [new 017] LOTUSDIS: A Thai far-field meeting corpus for robust conversational ASR
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出了LOTUSDIS，一个用于提升远场对话ASR鲁棒性的泰语会议语料库。针对远场语音识别性能下降的问题，构建了包含114小时真实对话的数据集，并通过基线实验验证了距离多样数据对提升识别效果的重要性。**

- **链接: [http://arxiv.org/pdf/2509.18722v1](http://arxiv.org/pdf/2509.18722v1)**

> **作者:** Pattara Tipaksorn; Sumonmas Thatphithakkul; Vataya Chunwijitra; Kwanchiva Thangthai
>
> **摘要:** We present LOTUSDIS, a publicly available Thai meeting corpus designed to advance far-field conversational ASR. The dataset comprises 114 hours of spontaneous, unscripted dialogue collected in 15-20 minute sessions with three participants, where overlapping speech is frequent and natural. Speech was recorded simultaneously by nine independent single-channel devices spanning six microphone types at distances from 0.12 m to 10 m, preserving the authentic effects of reverberation, noise, and device coloration without relying on microphone arrays. We provide standard train, dev, test splits and release a reproducible baseline system. We benchmarked several Whisper variants under zero-shot and fine-tuned conditions. Off-the-shelf models showed strong degradation with distance, confirming a mismatch between pre-training data and Thai far-field speech. Fine-tuning on LOTUSDIS dramatically improved robustness: a Thai Whisper baseline reduced overall WER from 64.3 to 38.3 and far-field WER from 81.6 to 49.5, with especially large gains on the most distant microphones. These results underscore the importance of distance-diverse training data for robust ASR. The corpus is available under CC-BY-SA 4.0. We also release training and evaluation scripts as a baseline system to promote reproducible research in this field.
>
---
#### [new 018] Trace Is In Sentences: Unbiased Lightweight ChatGPT-Generated Text Detector
- **分类: cs.CL; eess.SP**

- **简介: 该论文提出一种轻量级AI生成文本检测方法，任务是识别原始及经简单修改的ChatGPT生成文本。针对现有方法易受词级变换影响、存在偏见等问题，通过建模句子结构关系并结合对比学习与因果图，提取稳定结构特征以提升检测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18535v1](http://arxiv.org/pdf/2509.18535v1)**

> **作者:** Mo Mu; Dianqiao Lei; Chang Li
>
> **摘要:** The widespread adoption of ChatGPT has raised concerns about its misuse, highlighting the need for robust detection of AI-generated text. Current word-level detectors are vulnerable to paraphrasing or simple prompts (PSP), suffer from biases induced by ChatGPT's word-level patterns (CWP) and training data content, degrade on modified text, and often require large models or online LLM interaction. To tackle these issues, we introduce a novel task to detect both original and PSP-modified AI-generated texts, and propose a lightweight framework that classifies texts based on their internal structure, which remains invariant under word-level changes. Our approach encodes sentence embeddings from pre-trained language models and models their relationships via attention. We employ contrastive learning to mitigate embedding biases from autoregressive generation and incorporate a causal graph with counterfactual methods to isolate structural features from topic-related biases. Experiments on two curated datasets, including abstract comparisons and revised life FAQs, validate the effectiveness of our method.
>
---
#### [new 019] WaveletGaussian: Wavelet-domain Diffusion for Sparse-view 3D Gaussian Object Reconstruction
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 该论文提出WaveletGaussian，用于稀疏视角下的3D高斯物体重建。针对现有方法计算量大的问题，将扩散模型移至小波域，仅对低频部分进行扩散，并采用轻量网络优化高频部分，同时引入高效的在线随机掩码策略，显著提升了训练效率和重建质量。**

- **链接: [http://arxiv.org/pdf/2509.19073v1](http://arxiv.org/pdf/2509.19073v1)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **摘要:** 3D Gaussian Splatting (3DGS) has become a powerful representation for image-based object reconstruction, yet its performance drops sharply in sparse-view settings. Prior works address this limitation by employing diffusion models to repair corrupted renders, subsequently using them as pseudo ground truths for later optimization. While effective, such approaches incur heavy computation from the diffusion fine-tuning and repair steps. We present WaveletGaussian, a framework for more efficient sparse-view 3D Gaussian object reconstruction. Our key idea is to shift diffusion into the wavelet domain: diffusion is applied only to the low-resolution LL subband, while high-frequency subbands are refined with a lightweight network. We further propose an efficient online random masking strategy to curate training pairs for diffusion fine-tuning, replacing the commonly used, but inefficient, leave-one-out strategy. Experiments across two benchmark datasets, Mip-NeRF 360 and OmniObject3D, show WaveletGaussian achieves competitive rendering quality while substantially reducing training time.
>
---
#### [new 020] Teaching Audio Models to Reason: A Unified Framework for Source- and Layer-wise Distillation
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出一种统一的知识蒸馏框架，用于将文本模型的推理能力迁移至音频模型。针对音频与文本模态差异及缺乏结构监督的问题，引入源级和层级蒸馏策略，提升音频模型的推理性能。属于音频建模任务。**

- **链接: [http://arxiv.org/pdf/2509.18579v1](http://arxiv.org/pdf/2509.18579v1)**

> **作者:** Runyan Yang; Yuke Si; Yingying Gao; Junlan Feng; Chao Deng; Shilei Zhang
>
> **备注:** 5 pages; submitted to ICASSP 2026
>
> **摘要:** While large audio language models excel at tasks like ASR and emotion recognition, they still struggle with complex reasoning due to the modality gap between audio and text as well as the lack of structured intermediate supervision. To address this, we propose a unified knowledge distillation framework to transfer reasoning capabilities from a high-capacity textual teacher model to a student audio models while preserving its acoustic competence. Our method introduces two key dimensions: source-wise distillation, which leverages both textual and acoustic teachers to provide complementary modality-specific supervision; and layer-wise distillation, which aligns teacher signals with appropriate student layers to improve transfer efficiency. This dual-dimensional strategy enables fine-grained control over the distillation process, effectively bridging the gap between symbolic reasoning and speech representations. Experimental results show significant improvements in audio reasoning performance, demonstrating the effectiveness of our framework as a reasoning transfer solution for audio modeling.
>
---
#### [new 021] FlexSED: Towards Open-Vocabulary Sound Event Detection
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出FlexSED，面向开放词汇声事件检测任务。旨在解决现有系统无法处理自由文本查询、缺乏零样本和少样本能力的问题。通过结合预训练音频模型与CLAP文本编码器，并引入编码-解码结构与自适应融合策略，提升了检测性能与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.18606v1](http://arxiv.org/pdf/2509.18606v1)**

> **作者:** Jiarui Hai; Helin Wang; Weizhe Guo; Mounya Elhilali
>
> **摘要:** Despite recent progress in large-scale sound event detection (SED) systems capable of handling hundreds of sound classes, existing multi-class classification frameworks remain fundamentally limited. They cannot process free-text sound queries, which enable more flexible and user-friendly interaction, and they lack zero-shot capabilities and offer poor few-shot adaptability. Although text-query-based separation methods have been explored, they primarily focus on source separation and are ill-suited for SED tasks that require precise temporal localization and efficient detection across large and diverse sound vocabularies. In this paper, we propose FlexSED, an open-vocabulary sound event detection system. FlexSED builds on a pretrained audio SSL model and the CLAP text encoder, introducing an encoder-decoder composition and an adaptive fusion strategy to enable effective continuous training from pretrained weights. To ensure robust supervision, it also employs large language models (LLMs) to assist in event query selection during training, addressing challenges related to missing labels. As a result, FlexSED achieves superior performance compared to vanilla SED models on AudioSet-Strong, while demonstrating strong zero-shot and few-shot capabilities. We release the code and pretrained models to support future research and applications based on FlexSED.
>
---
#### [new 022] SynSonic: Augmenting Sound Event Detection through Text-to-Audio Diffusion ControlNet and Effective Sample Filtering
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文聚焦于声音事件检测（SED）任务，针对标注数据稀缺和样本多样性不足的问题，提出SynSonic方法。利用文本到音频扩散模型与ControlNet生成时序一致的合成音频，并通过双分类器过滤提升样本质量，从而增强SED性能。**

- **链接: [http://arxiv.org/pdf/2509.18603v1](http://arxiv.org/pdf/2509.18603v1)**

> **作者:** Jiarui Hai; Mounya Elhilali
>
> **摘要:** Data synthesis and augmentation are essential for Sound Event Detection (SED) due to the scarcity of temporally labeled data. While augmentation methods like SpecAugment and Mix-up can enhance model performance, they remain constrained by the diversity of existing samples. Recent generative models offer new opportunities, yet their direct application to SED is challenging due to the lack of precise temporal annotations and the risk of introducing noise through unreliable filtering. To address these challenges and enable generative-based augmentation for SED, we propose SynSonic, a data augmentation method tailored for this task. SynSonic leverages text-to-audio diffusion models guided by an energy-envelope ControlNet to generate temporally coherent sound events. A joint score filtering strategy with dual classifiers ensures sample quality, and we explore its practical integration into training pipelines. Experimental results show that SynSonic improves Polyphonic Sound Detection Scores (PSDS1 and PSDS2), enhancing both temporal localization and sound class discrimination.
>
---
#### [new 023] SloPalSpeech: A 2,8000-Hour Slovak Speech Corpus from Parliamentary Data
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文聚焦于低资源语言（斯洛伐克语）的自动语音识别任务，旨在解决训练数据不足的问题。作者构建了SloPalSpeech数据集（2806小时），并基于此微调Whisper模型，显著降低了词错误率，推动了低资源语音识别研究。**

- **链接: [http://arxiv.org/pdf/2509.19270v1](http://arxiv.org/pdf/2509.19270v1)**

> **作者:** Erik Božík; Marek Šuppa
>
> **摘要:** Automatic Speech Recognition (ASR) for low-resource languages like Slovak is hindered by the scarcity of training data. To address this, we introduce SloPalSpeech, a new, large-scale Slovak ASR dataset containing 2,806 hours of speech from parliamentary proceedings. We developed a robust processing pipeline to align and segment long-form recordings into clean, 30-second audio-transcript pairs suitable for model training. We use this dataset to fine-tune several OpenAI Whisper models (small, medium, large-v3, and large-v3-turbo), achieving significant Word Error Rate (WER) reductions on standard Slovak benchmarks like Common Voice and FLEURS. For instance, the fine-tuned Whisper-small model's WER dropped by up to 70\%, approaching the baseline performance of the much larger Whisper-large-v3 model. To foster future research in low-resource speech recognition, we publicly release the complete SloPalSpeech dataset, the fully segmented transcripts (60 million words), and all our fine-tuned models.
>
---
#### [new 024] HD-PPT: Hierarchical Decoding of Content- and Prompt-Preference Tokens for Instruction-based TTS
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出HD-PPT，用于指令驱动的语音合成（Instruct-TTS）任务。旨在解决多级语音与单级文本指令间的模态差异导致的精细控制不足问题。通过引入新语音编解码器和分层解码策略，实现更精确可控的语音生成。**

- **链接: [http://arxiv.org/pdf/2509.19001v1](http://arxiv.org/pdf/2509.19001v1)**

> **作者:** Sihang Nie; Xiaofen Xing; Jingyuan Xing; Baiji Liu; Xiangmin Xu
>
> **备注:** 5 pages, 2 figures, submitted to ICASSP2026
>
> **摘要:** Large Language Model (LLM)-based Text-to-Speech (TTS) models have already reached a high degree of naturalness. However, the precision control of TTS inference is still challenging. Although instruction-based Text-to-Speech (Instruct-TTS) models are proposed, these models still lack fine-grained control due to the modality gap between single-level text instructions and multilevel speech tokens. To address this limitation, we propose HD-PPT, a framework that transforms speech synthesis into a structured, hierarchical task. To enable fine-grained control, we introduce a novel speech codec to extract distinct prompt-preference and content-preference tokens from the complex speech tokens, supervised by automatic speech recognition (ASR) and cross-lingual audio-text pre-training (CLAP) objectives. To bridge the modality gap of these tokens, we propose a hierarchical decoding strategy, where the LLM generates tokens in a structured order: first semantic, then fine-grained style, and finally complete acoustic representation. Extensive experiments demonstrate that this hierarchical paradigm significantly improves instruction adherence and achieves state-of-the-art naturalness, validating our approach for precise and controllable speech synthesis. Audio samples are available at https://xxh333.github.io/.
>
---
#### [new 025] Automated Analysis of Naturalistic Recordings in Early Childhood: Applications, Challenges, and Opportunities
- **分类: eess.AS; cs.SD**

- **简介: 该论文探讨如何利用语音技术和机器学习自动分析幼儿自然场景录音，解决儿童早期发展研究中技术适配不足的问题，总结了应用、挑战与机遇，并呼吁跨学科合作推动技术发展。**

- **链接: [http://arxiv.org/pdf/2509.18235v1](http://arxiv.org/pdf/2509.18235v1)**

> **作者:** Jialu Li; Marvin Lavechin; Xulin Fan; Nancy L. McElwain; Alejandrina Cristia; Paola Garcia-Perera; Mark Hasegawa-Johnson
>
> **备注:** Accepted to IEEE Signal Processing Magazine
>
> **摘要:** Naturalistic recordings capture audio in real-world environments where participants behave naturally without interference from researchers or experimental protocols. Naturalistic long-form recordings extend this concept by capturing spontaneous and continuous interactions over extended periods, often spanning hours or even days, in participants' daily lives. Naturalistic recordings have been extensively used to study children's behaviors, including how they interact with others in their environment, in the fields of psychology, education, cognitive science, and clinical research. These recordings provide an unobtrusive way to observe children in real-world settings beyond controlled and constrained experimental environments. Advancements in speech technology and machine learning have provided an initial step for researchers to automatically and systematically analyze large-scale naturalistic recordings of children. Despite the imperfect accuracy of machine learning models, these tools still offer valuable opportunities to uncover important insights into children's cognitive and social development. Several critical speech technologies involved include speaker diarization, vocalization classification, word count estimate from adults, speaker verification, and language diarization for code-switching. Most of these technologies have been primarily developed for adults, and speech technologies applied to children specifically are still vastly under-explored. To fill this gap, we discuss current progress, challenges, and opportunities in advancing these technologies to analyze naturalistic recordings of children during early development (<3 years of age). We strive to inspire the signal processing community and foster interdisciplinary collaborations to further develop this emerging technology and address its unique challenges and opportunities.
>
---
#### [new 026] Qubit Instrumentation of Entanglement
- **分类: quant-ph; cs.SD; eess.AS**

- **简介: 该论文探索通过量子纠缠模拟音乐家之间的“人类纠缠”，利用MIDI捕捉音乐家音调关系，输入嵌入式设备进行量子仿真，并将结果反馈至乐器，实现量子-音乐表达的新形式，推动未来“纠缠合奏”的发展。**

- **链接: [http://arxiv.org/pdf/2509.18340v1](http://arxiv.org/pdf/2509.18340v1)**

> **作者:** Mark Carney
>
> **备注:** 28 pages, 4 figures, book chapter
>
> **摘要:** This chapter and the experiments described within explore how `human entanglement' might be represented and even emulated by physical entanglement. To achieve this, a notion of `tonal centrality' between two musicians is captured via MIDI and passed as a parameter into a quantum simulation taking place on an embedded device (a Raspberry Pi Pico). The results of these simulations are then coded back into MIDI and sent to the players' instruments. The closer the musicians' tonality is, the more their instruments will be entangled in a $|\Phi^+ \rangle$ state, and the further away they are the more their instruments will be entangled in a $|\Psi^+ \rangle$ state. The intention is to create random parameters that are correlative - \emph{i.e.} the same on both instruments - or anti-correlative - \emph{i.e.} the bit-wise opposite of each other, influenced by the tonal relationship from the players. These random parameters sharing these particular properties add a new dimension for quantum-musical expression. This concept was realised experimentally, and the full code and sample outputs are provided. This work aims to pave the way for musicians to explore and experience quantum emulations of their own musical experiences, adding a new nuance and possibilities for the future of \emph{entangled ensembles.}
>
---
#### [new 027] No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对TTS任务中韵律自然性不足的问题，提出一种基于少量人类偏好数据的迭代DPO方法，在KoCC-TTS数据集上实现了优于GRPO和商业基线的自然语音生成效果。**

- **链接: [http://arxiv.org/pdf/2509.18531v1](http://arxiv.org/pdf/2509.18531v1)**

> **作者:** Seungyoun Shin; Dongha Ahn; Jiwoo Kim; Sungwook Jeon
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{https://tts.ch.dev}
>
---
#### [new 028] SoundCompass: Navigating Target Sound Extraction With Effective Directional Clue Integration In Complex Acoustic Scenes
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出SoundCompass，用于复杂声学场景中的目标声音提取（TSE）。针对传统方法丢失空间信息的问题，设计SPIN模块融合多通道频谱的空间相关性，并结合球谐编码与迭代推理策略，提升提取性能。**

- **链接: [http://arxiv.org/pdf/2509.18561v1](http://arxiv.org/pdf/2509.18561v1)**

> **作者:** Dayun Choi; Jung-Woo Choi
>
> **备注:** 5 pages, 4 figures, submitted to ICASSP 2026
>
> **摘要:** Recent advances in target sound extraction (TSE) utilize directional clues derived from direction of arrival (DoA), which represent an inherent spatial property of sound available in any acoustic scene. However, previous DoA-based methods rely on hand-crafted features or discrete encodings, which lose fine-grained spatial information and limit adaptability. We propose SoundCompass, an effective directional clue integration framework centered on a Spectral Pairwise INteraction (SPIN) module that captures cross-channel spatial correlations in the complex spectrogram domain to preserve full spatial information in multichannel signals. The input feature expressed in terms of spatial correlations is fused with a DoA clue represented as spherical harmonics (SH) encoding. The fusion is carried out across overlapping frequency subbands, inheriting the benefits reported in the previous band-split architectures. We also incorporate the iterative refinement strategy, chain-of-inference (CoI), in the TSE framework, which recursively fuses DoA with sound event activation estimated from the previous inference stage. Experiments demonstrate that SoundCompass, combining SPIN, SH embedding, and CoI, robustly extracts target sources across diverse signal classes and spatial configurations.
>
---
#### [new 029] Audio-Based Pedestrian Detection in the Presence of Vehicular Noise
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文研究基于音频的行人检测任务，旨在解决车辆噪声干扰下的检测难题。作者构建了一个1321小时的路边音频数据集，评估了噪声环境对模型性能的影响，并分析了跨数据集表现和模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.19295v1](http://arxiv.org/pdf/2509.19295v1)**

> **作者:** Yonghyun Kim; Chaeyeon Han; Akash Sarode; Noah Posner; Subhrajit Guhathakurta; Alexander Lerch
>
> **备注:** Accepted to the 10th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2025
>
> **摘要:** Audio-based pedestrian detection is a challenging task and has, thus far, only been explored in noise-limited environments. We present a new dataset, results, and a detailed analysis of the state-of-the-art in audio-based pedestrian detection in the presence of vehicular noise. In our study, we conduct three analyses: (i) cross-dataset evaluation between noisy and noise-limited environments, (ii) an assessment of the impact of noisy data on model performance, highlighting the influence of acoustic context, and (iii) an evaluation of the model's predictive robustness on out-of-domain sounds. The new dataset is a comprehensive 1321-hour roadside dataset. It incorporates traffic-rich soundscapes. Each recording includes 16kHz audio synchronized with frame-level pedestrian annotations and 1fps video thumbnails.
>
---
## 更新

#### [replaced 001] DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.15690v2](http://arxiv.org/pdf/2507.15690v2)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** Accepted to VCIP 2025
>
> **摘要:** Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
>
---
#### [replaced 002] SupertonicTTS: Towards Highly Efficient and Streamlined Text-to-Speech System
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.23108v3](http://arxiv.org/pdf/2503.23108v3)**

> **作者:** Hyeongju Kim; Jinhyeok Yang; Yechan Yu; Seunghun Ji; Jacob Morton; Frederik Bous; Joon Byun; Juheon Lee
>
> **备注:** 22 pages, preprint
>
> **摘要:** We introduce SupertonicTTS, a novel text-to-speech (TTS) system designed for efficient and streamlined speech synthesis. SupertonicTTS comprises three components: a speech autoencoder for continuous latent representation, a text-to-latent module leveraging flow-matching for text-to-latent mapping, and an utterance-level duration predictor. To enable a lightweight architecture, we employ a low-dimensional latent space, temporal compression of latents, and ConvNeXt blocks. The TTS pipeline is further simplified by operating directly on raw character-level text and employing cross-attention for text-speech alignment, thus eliminating the need for grapheme-to-phoneme (G2P) modules and external aligners. In addition, we propose context-sharing batch expansion that accelerates loss convergence and stabilizes text-speech alignment with minimal memory and I/O overhead. Experimental results demonstrate that SupertonicTTS delivers performance comparable to contemporary zero-shot TTS models with only 44M parameters, while significantly reducing architectural complexity and computational cost. Audio samples are available at: https://supertonictts.github.io/.
>
---
#### [replaced 003] Large Language Models Implicitly Learn to See and Hear Just By Reading
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17091v2](http://arxiv.org/pdf/2505.17091v2)**

> **作者:** Prateek Verma; Mert Pilanci
>
> **备注:** 6 pages, 3 figures, 4 tables. Added BLIP reference
>
> **摘要:** This paper presents a fascinating find: By training an auto-regressive LLM model on text tokens, the text model inherently develops internally an ability to understand images and audio, thereby developing the ability to see and hear just by reading. Popular audio and visual LLM models fine-tune text LLM models to give text output conditioned on images and audio embeddings. On the other hand, our architecture takes in patches of images, audio waveforms or tokens as input. It gives us the embeddings or category labels typical of a classification pipeline. We show the generality of text weights in aiding audio classification for datasets FSD-50K and GTZAN. Further, we show this working for image classification on CIFAR-10 and Fashion-MNIST, as well on image patches. This pushes the notion of text-LLMs learning powerful internal circuits that can be utilized by activating necessary connections for various applications rather than training models from scratch every single time.
>
---
#### [replaced 004] PoolingVQ: A VQVAE Variant for Reducing Audio Redundancy and Boosting Multi-Modal Fusion in Music Emotion Analysis
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.11976v3](http://arxiv.org/pdf/2509.11976v3)**

> **作者:** Dinghao Zou; Yicheng Gong; Xiaokang Li; Xin Cao; Sunbowen Lee
>
> **摘要:** Multimodal music emotion analysis leverages both audio and MIDI modalities to enhance performance. While mainstream approaches focus on complex feature extraction networks, we propose that shortening the length of audio sequence features to mitigate redundancy, especially in contrast to MIDI's compact representation, may effectively boost task performance. To achieve this, we developed PoolingVQ by combining Vector Quantized Variational Autoencoder (VQVAE) with spatial pooling, which directly compresses audio feature sequences through codebook-guided local aggregation to reduce redundancy, then devised a two-stage co-attention approach to fuse audio and MIDI information. Experimental results on the public datasets EMOPIA and VGMIDI demonstrate that our multimodal framework achieves state-of-the-art performance, with PoolingVQ yielding effective improvement. Our proposed metho's code is available at Anonymous GitHub
>
---
#### [replaced 005] Compositional Phoneme Approximation for L1-Grounded L2 Pronunciation Training
- **分类: cs.CL; cs.SD; eess.AS; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.10927v4](http://arxiv.org/pdf/2411.10927v4)**

> **作者:** Jisang Park; Minu Kim; DaYoung Hong; Jongha Lee
>
> **摘要:** Learners of a second language (L2) often map non-native phonemes with similar native-language (L1) phonemes, making conventional L2-focused training slow and effortful. To address this, we propose an L1-grounded pronunciation training method based on compositional phoneme approximation (CPA), a feature-based representation technique that approximates L2 sounds with sequences of L1 phonemes. Evaluations with 20 Korean non-native English speakers show that CPA-based training achieves a 76% in-box formant rate in acoustic analysis, over 20% relative improvement in phoneme recognition accuracy, and over 80% of speech being rated as more native-like, with minimal training.
>
---
#### [replaced 006] mRadNet: A Compact Radar Object Detector with MetaFormer
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.16223v2](http://arxiv.org/pdf/2509.16223v2)**

> **作者:** Huaiyu Chen; Fahed Hassanat; Robert Laganiere; Martin Bouchard
>
> **备注:** 5 pages, 2 figures, submitted to IEEE ICASSP 2026. Code availble at https://github.com/huaiyu-chen/mRadNet
>
> **摘要:** Frequency-modulated continuous wave radars have gained increasing popularity in the automotive industry. Its robustness against adverse weather conditions makes it a suitable choice for radar object detection in advanced driver assistance systems. These real-time embedded systems have requirements for the compactness and efficiency of the model, which have been largely overlooked in previous work. In this work, we propose mRadNet, a novel radar object detection model with compactness in mind. mRadNet employs a U-net style architecture with MetaFormer blocks, in which separable convolution and attention token mixers are used to capture both local and global features effectively. More efficient token embedding and merging strategies are introduced to further facilitate the lightweight design. The performance of mRadNet is validated on the CRUW dataset, improving state-of-the-art performance with the least number of parameters and FLOPs.
>
---
#### [replaced 007] FUN-SSL: Full-band Layer Followed by U-Net with Narrow-band Layers for Multiple Moving Sound Source Localization
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2509.17490v2](http://arxiv.org/pdf/2509.17490v2)**

> **作者:** Yuseon Choi; Hyeonseung Kim; Jewoo Jun; Jong Won Shin
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Dual-path processing along the temporal and spectral dimensions has shown to be effective in various speech processing applications. While the sound source localization (SSL) models utilizing dual-path processing such as the FN-SSL and IPDnet demonstrated impressive performances in localizing multiple moving sources, they require significant amount of computation. In this paper, we propose an architecture for SSL which introduces a U-Net to perform narrow-band processing in multiple resolutions to reduce computational complexity. The proposed model replaces the full-narrow network block in the IPDnet consisting of one full-band LSTM layer along the spectral dimension followed by one narrow-band LSTM layer along the temporal dimension with the FUN block composed of one Full-band layer followed by a U-net with Narrow-band layers in multiple scales. On top of the skip connections within each U-Net, we also introduce the skip connections between FUN blocks to enrich information. Experimental results showed that the proposed FUN-SSL outperformed previously proposed approaches with computational complexity much lower than that of the IPDnet.
>
---
#### [replaced 008] DeepResonance: Enhancing Multimodal Music Understanding via Music-centric Multi-way Instruction Tuning
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.12623v3](http://arxiv.org/pdf/2502.12623v3)**

> **作者:** Zhuoyuan Mao; Mengjie Zhao; Qiyu Wu; Hiromi Wakaki; Yuki Mitsufuji
>
> **备注:** Accepted to EMNLP 2025 main conference
>
> **摘要:** Recent advancements in music large language models (LLMs) have significantly improved music understanding tasks, which involve the model's ability to analyze and interpret various musical elements. These improvements primarily focused on integrating both music and text inputs. However, the potential of incorporating additional modalities such as images, videos and textual music features to enhance music understanding remains unexplored. To bridge this gap, we propose DeepResonance, a multimodal music understanding LLM fine-tuned via multi-way instruction tuning with multi-way aligned music, text, image, and video data. To this end, we construct Music4way-MI2T, Music4way-MV2T, and Music4way-Any2T, three 4-way training and evaluation datasets designed to enable DeepResonance to integrate both visual and textual music feature content. We also introduce multi-sampled ImageBind embeddings and a pre-LLM fusion Transformer to enhance modality fusion prior to input into text LLMs, tailoring for multi-way instruction tuning. Our model achieves state-of-the-art performances across six music understanding tasks, highlighting the benefits of the auxiliary modalities and the structural superiority of DeepResonance. We open-source the codes, models and datasets we constructed: github.com/sony/DeepResonance.
>
---
#### [replaced 009] WavReward: Spoken Dialogue Models With Generalist Reward Evaluators
- **分类: eess.AS; cs.AI; cs.LG; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.09558v2](http://arxiv.org/pdf/2505.09558v2)**

> **作者:** Shengpeng Ji; Tianle Liang; Yangzhuo Li; Jialong Zuo; Minghui Fang; Jinzheng He; Yifu Chen; Zhengqing Liu; Ziyue Jiang; Xize Cheng; Siqi Zheng; Jin Xu; Junyang Lin; Zhou Zhao
>
> **摘要:** End-to-end spoken dialogue models such as GPT-4o-audio have recently garnered significant attention in the speech domain. However, the evaluation of spoken dialogue models' conversational performance has largely been overlooked. This is primarily due to the intelligent chatbots convey a wealth of non-textual information which cannot be easily measured using text-based language models like ChatGPT. To address this gap, we propose WavReward, a reward feedback model based on audio language models that can evaluate both the IQ and EQ of spoken dialogue systems with speech input. Specifically, 1) based on audio language models, WavReward incorporates the deep reasoning process and the nonlinear reward mechanism for post-training. By utilizing multi-sample feedback via the reinforcement learning algorithm, we construct a specialized evaluator tailored to spoken dialogue models. 2) We introduce ChatReward-30K, a preference dataset used to train WavReward. ChatReward-30K includes both comprehension and generation aspects of spoken dialogue models. These scenarios span various tasks, such as text-based chats, nine acoustic attributes of instruction chats, and implicit chats. WavReward outperforms previous state-of-the-art evaluation models across multiple spoken dialogue scenarios, achieving a substantial improvement about Qwen2.5-Omni in objective accuracy from 53.4$\%$ to 91.5$\%$. In subjective A/B testing, WavReward also leads by a margin of 83$\%$. Comprehensive ablation studies confirm the necessity of each component of WavReward. All data and code will be publicly at https://github.com/jishengpeng/WavReward after the paper is accepted.
>
---
