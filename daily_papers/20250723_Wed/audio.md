# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] A new XML conversion process for mensural music encoding : CMME\_to\_MEI (via Verovio)
- **分类: cs.SD; cs.DB**

- **简介: 论文介绍了一个将CMME格式的15世纪音乐数据转换为现代MEI标准的工具开发任务。为解决CMME格式过时、难以使用的问题，研究团队在Verovio中开发了CMME到MEI的转换器，使历史音乐数据得以重用，并可导入现代软件进行编辑与展示。**

- **链接: [http://arxiv.org/pdf/2507.15991v1](http://arxiv.org/pdf/2507.15991v1)**

> **作者:** David Fiala; Laurent Pugin; Marnix van Berchum; Martha Thomae; Kévin Roger
>
> **摘要:** The Ricercar Lab - the musicological research team at the Center for advanced Studies in the Renaissance at the University of Tours - has decided to make available in open access, thanks to the support of the French digital infrastructure Biblissima, a large corpus of about 3500 XML files of 15th-c. music. This corpus was produced by the German musicologist Clemens Goldberg who encoded since 2010 onwards the musical content of 34 major 15th-c. music manuscripts and other complementary files, in order to offer on his foundation's website PDF files of complete collections of works by Du Fay, Binchois, Okeghem, Busnoys and most of their major contemporaries, focusing on their secular output. This corpus was encoded in an XML format named CMME (Computerized Mensural Music Editing), specifically conceived for mensural music by Theodor Dumitrescu in the 2000s, together with editorial and publication tools which have not been updated since then. This article focuses on the development of a set of conversion tools for these CMME files to meet more up-to-date standards of music encoding, namely MEI. A workshop was organised in September 2024 at the Campus Condorcet in Paris, gathering experts with a wide range of knowledge on mensural music notation, XML formats and programming. A converter was developped directly in the open-source rendering library Verovio, allowing the conversion from CMME to MEI mensural. A conversion to MEI CMN was implemented afterwards, enabling to load these files in common engraving softwares such as MuseScore with minimal loss of information. With the availability of a direct import of CMME-XML into Verovio, the corpus of existing CMME files gets a new life. Furthermore, since the stand-alone CMME editor still works fine and no alternative is available yet for native MEI, the converter offers a new pipeline for encoding and editing mensural music.
>
---
#### [new 002] SALM: Spatial Audio Language Model with Structured Embeddings for Understanding and Editing
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SALM模型，属于音频-语言多模态任务，旨在解决空间音频理解与编辑困难的问题。通过结构化嵌入，将空间音频分解为语义与空间成分，实现跨模态对齐，并支持零样本方向分类与文本驱动的空间音频编辑。**

- **链接: [http://arxiv.org/pdf/2507.16724v1](http://arxiv.org/pdf/2507.16724v1)**

> **作者:** Jinbo Hu; Yin Cao; Ming Wu; Feiran Yang; Jun Yang
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Spatial audio understanding is essential for accurately perceiving and interpreting acoustic environments. However, existing audio-language models struggle with processing spatial audio and perceiving spatial acoustic scenes. We introduce the Spatial Audio Language Model (SALM), a novel framework that bridges spatial audio and language via multi-modal contrastive learning. SALM consists of a text encoder and a dual-branch audio encoder, decomposing spatial sound into semantic and spatial components through structured audio embeddings. Key features of SALM include seamless alignment of spatial and text representations, separate and joint extraction of spatial and semantic information, zero-shot direction classification and robust support for spatial audio editing. Experimental results demonstrate that SALM effectively captures and aligns cross-modal representations. Furthermore, it supports advanced editing capabilities, such as altering directional audio using text-based embeddings.
>
---
#### [new 003] SDBench: A Comprehensive Benchmark Suite for Speaker Diarization
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音处理中的说话人日志（speaker diarization）任务，旨在解决不同系统在多样本数据上的表现差异大、难以统一评估的问题。作者构建了SDBench基准测试套件，集成13个数据集并提供统一评估工具，支持新系统的便捷接入。通过SDBench，作者开发了高效系统SpeakerKit，比Pyannote v3快9.6倍，同时评估了多个前沿系统在准确率与速度间的权衡。**

- **链接: [http://arxiv.org/pdf/2507.16136v1](http://arxiv.org/pdf/2507.16136v1)**

> **作者:** Eduardo Pacheco; Atila Orhon; Berkin Durmus; Blaise Munyampirwa; Andrey Leonov
>
> **摘要:** Even state-of-the-art speaker diarization systems exhibit high variance in error rates across different datasets, representing numerous use cases and domains. Furthermore, comparing across systems requires careful application of best practices such as dataset splits and metric definitions to allow for apples-to-apples comparison. We propose SDBench (Speaker Diarization Benchmark), an open-source benchmark suite that integrates 13 diverse datasets with built-in tooling for consistent and fine-grained analysis of speaker diarization performance for various on-device and server-side systems. SDBench enables reproducible evaluation and easy integration of new systems over time. To demonstrate the efficacy of SDBench, we built SpeakerKit, an inference efficiency-focused system built on top of Pyannote v3. SDBench enabled rapid execution of ablation studies that led to SpeakerKit being 9.6x faster than Pyannote v3 while achieving comparable error rates. We benchmark 6 state-of-the-art systems including Deepgram, AWS Transcribe, and Pyannote AI API, revealing important trade-offs between accuracy and speed.
>
---
#### [new 004] LENS-DF: Deepfake Detection and Temporal Localization for Long-Form Noisy Speech
- **分类: cs.SD; cs.CR**

- **简介: 该论文属于音频深度伪造检测与定位任务，旨在解决复杂真实场景下长时噪声语音的深伪检测难题。作者提出了LENS-DF方法，通过可控生成具备多种真实特征的音频数据，结合自监督学习模型，实现更鲁棒的检测与时间定位，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.16220v1](http://arxiv.org/pdf/2507.16220v1)**

> **作者:** Xuechen Liu; Wanying Ge; Xin Wang; Junichi Yamagishi
>
> **备注:** Accepted by IEEE International Joint Conference on Biometrics (IJCB) 2025, Osaka, Japan
>
> **摘要:** This study introduces LENS-DF, a novel and comprehensive recipe for training and evaluating audio deepfake detection and temporal localization under complicated and realistic audio conditions. The generation part of the recipe outputs audios from the input dataset with several critical characteristics, such as longer duration, noisy conditions, and containing multiple speakers, in a controllable fashion. The corresponding detection and localization protocol uses models. We conduct experiments based on self-supervised learning front-end and simple back-end. The results indicate that models trained using data generated with LENS-DF consistently outperform those trained via conventional recipes, demonstrating the effectiveness and usefulness of LENS-DF for robust audio deepfake detection and localization. We also conduct ablation studies on the variations introduced, investigating their impact on and relevance to realistic challenges in the field.
>
---
#### [new 005] LABNet: A Lightweight Attentive Beamforming Network for Ad-hoc Multichannel Microphone Invariant Real-Time Speech Enhancement
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决多麦克风条件下实时语音恢复与麦克风不变性的权衡问题。作者提出了轻量级注意力波束成形网络LABNet，通过三级框架实现高效通道内建模与跨通道交互，兼顾低复杂度与麦克风配置适应性。**

- **链接: [http://arxiv.org/pdf/2507.16190v1](http://arxiv.org/pdf/2507.16190v1)**

> **作者:** Haoyin Yan; Jie Zhang; Chengqian Jiang; Shuang Zhang
>
> **摘要:** Multichannel speech enhancement (SE) aims to restore clean speech from noisy measurements by leveraging spatiotemporal signal features. In ad-hoc array conditions, microphone invariance (MI) requires systems to handle different microphone numbers and array geometries. From a practical perspective, multichannel recordings inevitably increase the computational burden for edge-device applications, highlighting the necessity of lightweight and efficient deployments. In this work, we propose a lightweight attentive beamforming network (LABNet) to integrate MI in a low-complexity real-time SE system. We design a three-stage framework for efficient intra-channel modeling and inter-channel interaction. A cross-channel attention module is developed to aggregate features from each channel selectively. Experimental results demonstrate our LABNet achieves impressive performance with ultra-light resource overhead while maintaining the MI, indicating great potential for ad-hoc array processing.
>
---
#### [new 006] Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于声音事件检测任务，旨在解决现有方法受限于预定义类别、无法泛化到新类的问题。作者提出DASM模型，通过多模态查询实现开集声音事件检测，采用双流解码器解耦事件识别与时间定位，并设计注意力掩码策略提升对新类的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.16343v1](http://arxiv.org/pdf/2507.16343v1)**

> **作者:** Pengfei Cai; Yan Song; Qing Gu; Nan Jiang; Haoyu Song; Ian McLoughlin
>
> **备注:** Accepted by MM 2025
>
> **摘要:** Most existing sound event detection~(SED) algorithms operate under a closed-set assumption, restricting their detection capabilities to predefined classes. While recent efforts have explored language-driven zero-shot SED by exploiting audio-language models, their performance is still far from satisfactory due to the lack of fine-grained alignment and cross-modal feature fusion. In this work, we propose the Detect Any Sound Model (DASM), a query-based framework for open-vocabulary SED guided by multi-modal queries. DASM formulates SED as a frame-level retrieval task, where audio features are matched against query vectors derived from text or audio prompts. To support this formulation, DASM introduces a dual-stream decoder that explicitly decouples event recognition and temporal localization: a cross-modality event decoder performs query-feature fusion and determines the presence of sound events at the clip-level, while a context network models temporal dependencies for frame-level localization. Additionally, an inference-time attention masking strategy is proposed to leverage semantic relations between base and novel classes, substantially enhancing generalization to novel classes. Experiments on the AudioSet Strong dataset demonstrate that DASM effectively balances localization accuracy with generalization to novel classes, outperforming CLAP-based methods in open-vocabulary setting (+ 7.8 PSDS) and the baseline in the closed-set setting (+ 6.9 PSDS). Furthermore, in cross-dataset zero-shot evaluation on DESED, DASM achieves a PSDS1 score of 42.2, even exceeding the supervised CRNN baseline. The project page is available at https://cai525.github.io/Transformer4SED/demo_page/DASM/.
>
---
#### [new 007] TTMBA: Towards Text To Multiple Sources Binaural Audio Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决现有方法生成单声道音频、缺乏空间信息的问题。论文提出TTMBA方法，通过级联结构生成多声源双耳音频，包含文本解析、单声道音频生成、双耳渲染及时间排列等步骤，提升音频生成质量与空间感知效果。**

- **链接: [http://arxiv.org/pdf/2507.16564v1](http://arxiv.org/pdf/2507.16564v1)**

> **作者:** Yuxuan He; Xiaoran Yang; Ningning Pan; Gongping Huang
>
> **备注:** 5 pages,3 figures,2 tables
>
> **摘要:** Most existing text-to-audio (TTA) generation methods produce mono outputs, neglecting essential spatial information for immersive auditory experiences. To address this issue, we propose a cascaded method for text-to-multisource binaural audio generation (TTMBA) with both temporal and spatial control. First, a pretrained large language model (LLM) segments the text into a structured format with time and spatial details for each sound event. Next, a pretrained mono audio generation network creates multiple mono audios with varying durations for each event. These mono audios are transformed into binaural audios using a binaural rendering neural network based on spatial data from the LLM. Finally, the binaural audios are arranged by their start times, resulting in multisource binaural audio. Experimental results demonstrate the superiority of the proposed method in terms of both audio generation quality and spatial perceptual accuracy.
>
---
#### [new 008] Nonlinear Framework for Speech Bandwidth Extension
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音带宽扩展任务，旨在恢复带宽限制导致的高频成分损失。论文提出了NDSI-BWE框架，结合七种基于非线性动力系统的判别器，减少参数的同时提升语音质量。使用复杂值ConformerNeXt生成器优化幅度与相位，取得当前最优性能。**

- **链接: [http://arxiv.org/pdf/2507.15970v1](http://arxiv.org/pdf/2507.15970v1)**

> **作者:** Tarikul Islam Tamiti; Nursad Mamun; Anomadarshi Barua
>
> **摘要:** Recovering high-frequency components lost to bandwidth constraints is crucial for applications ranging from telecommunications to high-fidelity audio on limited resources. We introduce NDSI-BWE, a new adversarial Band Width Extension (BWE) framework that leverage four new discriminators inspired by nonlinear dynamical system to capture diverse temporal behaviors: a Multi-Resolution Lyapunov Discriminator (MRLD) for determining sensitivity to initial conditions by capturing deterministic chaos, a Multi-Scale Recurrence Discriminator (MS-RD) for self-similar recurrence dynamics, a Multi-Scale Detrended Fractal Analysis Discriminator (MSDFA) for long range slow variant scale invariant relationship, a Multi-Resolution Poincar\'e Plot Discriminator (MR-PPD) for capturing hidden latent space relationship, a Multi-Period Discriminator (MPD) for cyclical patterns, a Multi-Resolution Amplitude Discriminator (MRAD) and Multi-Resolution Phase Discriminator (MRPD) for capturing intricate amplitude-phase transition statistics. By using depth-wise convolution at the core of the convolutional block with in each discriminators, NDSI-BWE attains an eight-times parameter reduction. These seven discriminators guide a complex-valued ConformerNeXt based genetor with a dual stream Lattice-Net based architecture for simultaneous refinement of magnitude and phase. The genertor leverage the transformer based conformer's global dependency modeling and ConvNeXt block's local temporal modeling capability. Across six objective evaluation metrics and subjective based texts comprises of five human judges, NDSI-BWE establishes a new SoTA in BWE.
>
---
#### [new 009] Robust Bioacoustic Detection via Richly Labelled Synthetic Soundscape Augmentation
- **分类: cs.SD**

- **简介: 该论文属于生物声学检测任务，旨在解决标注训练数据耗时费力的问题。作者提出一种合成数据框架，通过合成真实声景并自动生成动态标签，有效提升了模型鲁棒性与泛化能力，减少了对大量真实数据的依赖和人工标注成本。**

- **链接: [http://arxiv.org/pdf/2507.16235v1](http://arxiv.org/pdf/2507.16235v1)**

> **作者:** Kaspar Soltero; Tadeu Siqueira; Stefanie Gutschmidt
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Passive Acoustic Monitoring (PAM) analysis is often hindered by the intensive manual effort needed to create labelled training data. This study introduces a synthetic data framework to generate large volumes of richly labelled training data from very limited source material, improving the robustness of bioacoustic detection models. Our framework synthesises realistic soundscapes by combining clean background noise with isolated target vocalisations (little owl), automatically generating dynamic labels like bounding boxes during synthesis. A model fine-tuned on this data generalised well to real-world soundscapes, with performance remaining high even when the diversity of source vocalisations was drastically reduced, indicating the model learned generalised features without overfitting. This demonstrates that synthetic data generation is a highly effective strategy for training robust bioacoustic detectors from small source datasets. The approach significantly reduces manual labelling effort, overcoming a key bottleneck in computational bioacoustics and enhancing ecological assessment capabilities.
>
---
#### [new 010] Interpretable Embeddings of Speech Enhance and Explain Brain Encoding Performance of Audio Models
- **分类: q-bio.NC; cs.SD**

- **简介: 该论文属于语音处理与神经科学交叉任务，旨在探究语音模型如何更好地解释大脑对语音的编码机制。论文提出可解释性特征模型，结合自监督语音模型（SSM），分析其对脑电响应的预测能力，并揭示SSM中影响神经编码的关键成分。**

- **链接: [http://arxiv.org/pdf/2507.16080v1](http://arxiv.org/pdf/2507.16080v1)**

> **作者:** Riki Shimizu; Richard J. Antonello; Chandan Singh; Nima Mesgarani
>
> **备注:** 7pages, 4 figures
>
> **摘要:** Self-supervised speech models (SSMs) are increasingly hailed as more powerful computational models of human speech perception than models based on traditional hand-crafted features. However, since their representations are inherently black-box, it remains unclear what drives their alignment with brain responses. To remedy this, we built linear encoding models from six interpretable feature families: mel-spectrogram, Gabor filter bank features, speech presence, phonetic, syntactic, and semantic Question-Answering features, and contextualized embeddings from three state-of-the-art SSMs (Whisper, HuBERT, WavLM), quantifying the shared and unique neural variance captured by each feature class. Contrary to prevailing assumptions, our interpretable model predicted electrocorticography (ECoG) responses to speech more accurately than any SSM. Moreover, augmenting SSM representations with interpretable features yielded the best overall neural predictions, significantly outperforming either class alone. Further variance-partitioning analyses revealed previously unresolved components of SSM representations that contribute to their neural alignment: 1. Despite the common assumption that later layers of SSMs discard low-level acoustic information, these models compress and preferentially retain frequency bands critical for neural encoding of speech (100-1000 Hz). 2. Contrary to previous claims, SSMs encode brain-relevant semantic information that cannot be reduced to lower-level features, improving with context length and model size. These results highlight the importance of using refined, interpretable features in understanding speech perception.
>
---
#### [new 011] FISHER: A Foundation Model for Multi-Modal Industrial Signal Comprehensive Representation
- **分类: cs.LG; cs.AI; cs.MM; cs.SD**

- **简介: 论文提出FISHER，一种用于多模态工业信号表征的基础模型，旨在解决工业信号异构性强（M5问题）导致的分析难题。通过统一建模不同采样率的信号子带，采用自监督学习框架预训练，在多个工业健康任务中表现优异，提升综合表征能力。**

- **链接: [http://arxiv.org/pdf/2507.16696v1](http://arxiv.org/pdf/2507.16696v1)**

> **作者:** Pingyi Fan; Anbai Jiang; Shuwei Zhang; Zhiqiang Lv; Bing Han; Xinhu Zheng; Wenrui Liang; Junjie Li; Wei-Qiang Zhang; Yanmin Qian; Xie Chen; Cheng Lu; Jia Liu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** With the rapid deployment of SCADA systems, how to effectively analyze industrial signals and detect abnormal states is an urgent need for the industry. Due to the significant heterogeneity of these signals, which we summarize as the M5 problem, previous works only focus on small sub-problems and employ specialized models, failing to utilize the synergies between modalities and the powerful scaling law. However, we argue that the M5 signals can be modeled in a unified manner due to the intrinsic similarity. As a result, we propose FISHER, a Foundation model for multi-modal Industrial Signal compreHEnsive Representation. To support arbitrary sampling rates, FISHER considers the increment of sampling rate as the concatenation of sub-band information. Specifically, FISHER takes the STFT sub-band as the modeling unit and adopts a teacher student SSL framework for pre-training. We also develop the RMIS benchmark, which evaluates the representations of M5 industrial signals on multiple health management tasks. Compared with top SSL models, FISHER showcases versatile and outstanding capabilities with a general performance gain up to 5.03%, along with much more efficient scaling curves. We also investigate the scaling law on downstream tasks and derive potential avenues for future works. FISHER is now open-sourced on https://github.com/jianganbai/FISHER
>
---
#### [new 012] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文介绍了Step-Audio 2，一种用于音频理解和语音对话的多模态大模型。它通过集成音频编码器和强化学习，提升语音识别和对话响应能力，并结合检索增强生成减少幻觉。模型训练于海量语音数据，适用于多种对话场景。**

- **链接: [http://arxiv.org/pdf/2507.16632v1](http://arxiv.org/pdf/2507.16632v1)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **摘要:** This paper presents Step-Audio~2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [new 013] Universal Wavelet Units in 3D Retinal Layer Segmentation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医学图像分割任务，旨在提升视网膜层分割精度。为解决传统下采样方法丢失高频细节的问题，作者将三种可学习小波模块集成到网络中，实现3D OCT图像更精确的分割，尤其LS-BiorthLattUwU效果最佳。**

- **链接: [http://arxiv.org/pdf/2507.16119v1](http://arxiv.org/pdf/2507.16119v1)**

> **作者:** An D. Le; Hung Nguyen; Melanie Tran; Jesse Most; Dirk-Uwe G. Bartsch; William R Freeman; Shyamanga Borooah; Truong Q. Nguyen; Cheolhong An
>
> **摘要:** This paper presents the first study to apply tunable wavelet units (UwUs) for 3D retinal layer segmentation from Optical Coherence Tomography (OCT) volumes. To overcome the limitations of conventional max-pooling, we integrate three wavelet-based downsampling modules, OrthLattUwU, BiorthLattUwU, and LS-BiorthLattUwU, into a motion-corrected MGU-Net architecture. These modules use learnable lattice filter banks to preserve both low- and high-frequency features, enhancing spatial detail and structural consistency. Evaluated on the Jacobs Retina Center (JRC) OCT dataset, our framework shows significant improvement in accuracy and Dice score, particularly with LS-BiorthLattUwU, highlighting the benefits of tunable wavelet filters in volumetric medical image segmentation.
>
---
#### [new 014] An approach to measuring the performance of Automatic Speech Recognition (ASR) models in the context of Large Language Model (LLM) powered applications
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别与自然语言处理任务，旨在解决在大语言模型（LLM）驱动的应用中，如何有效评估自动语音识别（ASR）性能的问题。论文分析了LLM对ASR错误的修正能力，并提出了新的评估指标。**

- **链接: [http://arxiv.org/pdf/2507.16456v1](http://arxiv.org/pdf/2507.16456v1)**

> **作者:** Sujith Pulikodan; Sahapthan K; Prasanta Kumar Ghosh; Visruth Sanka; Nihar Desai
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Automatic Speech Recognition (ASR) plays a crucial role in human-machine interaction and serves as an interface for a wide range of applications. Traditionally, ASR performance has been evaluated using Word Error Rate (WER), a metric that quantifies the number of insertions, deletions, and substitutions in the generated transcriptions. However, with the increasing adoption of large and powerful Large Language Models (LLMs) as the core processing component in various applications, the significance of different types of ASR errors in downstream tasks warrants further exploration. In this work, we analyze the capabilities of LLMs to correct errors introduced by ASRs and propose a new measure to evaluate ASR performance for LLM-powered applications.
>
---
#### [new 015] Stop-band Energy Constraint for Orthogonal Tunable Wavelet Units in Convolutional Neural Networks for Computer Vision problems
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于计算机视觉任务，旨在提升卷积神经网络在图像分类与异常检测中的性能。作者提出了一种基于正交可调小波单元的滤波器停带能量约束方法，应用于ResNet-18和ResNet-34中，显著提高了CIFAR-10和纹理数据集的分类准确率，并在MVTec榛子异常检测任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2507.16114v1](http://arxiv.org/pdf/2507.16114v1)**

> **作者:** An D. Le; Hung Nguyen; Sungbal Seo; You-Suk Bae; Truong Q. Nguyen
>
> **摘要:** This work introduces a stop-band energy constraint for filters in orthogonal tunable wavelet units with a lattice structure, aimed at improving image classification and anomaly detection in CNNs, especially on texture-rich datasets. Integrated into ResNet-18, the method enhances convolution, pooling, and downsampling operations, yielding accuracy gains of 2.48% on CIFAR-10 and 13.56% on the Describable Textures dataset. Similar improvements are observed in ResNet-34. On the MVTec hazelnut anomaly detection task, the proposed method achieves competitive results in both segmentation and detection, outperforming existing approaches.
>
---
## 更新

#### [replaced 001] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05724v2](http://arxiv.org/pdf/2507.05724v2)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model Omni-router Transformer. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [replaced 002] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling
- **分类: cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.14915v2](http://arxiv.org/pdf/2507.14915v2)**

> **作者:** Xiaojie Li; Ronghui Li; Shukai Fang; Shuzhao Xie; Xiaoyang Guo; Jiaqing Zhou; Junkun Peng; Zhi Wang
>
> **摘要:** Well-coordinated, music-aligned holistic dance enhances emotional expressiveness and audience engagement. However, generating such dances remains challenging due to the scarcity of holistic 3D dance datasets, the difficulty of achieving cross-modal alignment between music and dance, and the complexity of modeling interdependent motion across the body, hands, and face. To address these challenges, we introduce SoulDance, a high-precision music-dance paired dataset captured via professional motion capture systems, featuring meticulously annotated holistic dance movements. Building on this dataset, we propose SoulNet, a framework designed to generate music-aligned, kinematically coordinated holistic dance sequences. SoulNet consists of three principal components: (1) Hierarchical Residual Vector Quantization, which models complex, fine-grained motion dependencies across the body, hands, and face; (2) Music-Aligned Generative Model, which composes these hierarchical motion units into expressive and coordinated holistic dance; (3) Music-Motion Retrieval Module, a pre-trained cross-modal model that functions as a music-dance alignment prior, ensuring temporal synchronization and semantic coherence between generated dance and input music throughout the generation process. Extensive experiments demonstrate that SoulNet significantly surpasses existing approaches in generating high-quality, music-coordinated, and well-aligned holistic 3D dance sequences.
>
---
#### [replaced 003] ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting
- **分类: eess.AS; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.20630v4](http://arxiv.org/pdf/2504.20630v4)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating Drama-MOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos are available at https://aaronz345.github.io/ISDramaDemo. We provide the dataset and the evaluation code at https://huggingface.co/datasets/AaronZ345/MRSDrama and https://github.com/AaronZ345/ISDrama.
>
---
#### [replaced 004] ReMi: A Random Recurrent Neural Network Approach to Music Production
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17023v2](http://arxiv.org/pdf/2505.17023v2)**

> **作者:** Hugo Chateau-Laurent; Tara Vanhatalo; Wei-Tung Pan; Xavier Hinaut
>
> **备注:** Accepted for an Innovation Showcase Demo at International Computer Music Conference
>
> **摘要:** Generative artificial intelligence raises concerns related to energy consumption, copyright infringement and creative atrophy. We show that randomly initialized recurrent neural networks can produce arpeggios and low-frequency oscillations that are rich and configurable. In contrast to end-to-end music generation that aims to replace musicians, our approach expands their creativity while requiring no data and much less computational power. More information can be found at: https://allendia.com/
>
---
#### [replaced 005] Audio Geolocation: A Natural Sounds Benchmark
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18726v2](http://arxiv.org/pdf/2505.18726v2)**

> **作者:** Mustafa Chasmai; Wuao Liu; Subhransu Maji; Grant Van Horn
>
> **摘要:** Can we determine someone's geographic location purely from the sounds they hear? Are acoustic signals enough to localize within a country, state, or even city? We tackle the challenge of global-scale audio geolocation, formalize the problem, and conduct an in-depth analysis with wildlife audio from the iNatSounds dataset. Adopting a vision-inspired approach, we convert audio recordings to spectrograms and benchmark existing image geolocation techniques. We hypothesize that species vocalizations offer strong geolocation cues due to their defined geographic ranges and propose an approach that integrates species range prediction with retrieval-based geolocation. We further evaluate whether geolocation improves when analyzing species-rich recordings or when aggregating across spatiotemporal neighborhoods. Finally, we introduce case studies from movies to explore multimodal geolocation using both audio and visual content. Our work highlights the advantages of integrating audio and visual cues, and sets the stage for future research in audio geolocation.
>
---
