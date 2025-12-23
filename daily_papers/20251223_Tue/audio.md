# 音频 cs.SD;  eess.AS

- **最新发布 27 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Let the Model Learn to Feel: Mode-Guided Tonality Injection for Symbolic Music Emotion Recognition
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文面向符号音乐情感识别（SMER）任务，旨在解决预训练模型（如MIDIBERT）忽略调式（mode）导致情感识别性能受限的问题。提出模式引导增强（MoGE）策略，通过模式感知的特征线性调制（MoFi）注入调式信息，显著提升情感识别准确率。**

- **链接: [https://arxiv.org/pdf/2512.17946v1](https://arxiv.org/pdf/2512.17946v1)**

> **作者:** Haiying Xia; Zhongyi Huang; Yumei Tan; Shuxiang Song
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Music emotion recognition is a key task in symbolic music understanding (SMER). Recent approaches have shown promising results by fine-tuning large-scale pre-trained models (e.g., MIDIBERT, a benchmark in symbolic music understanding) to map musical semantics to emotional labels. While these models effectively capture distributional musical semantics, they often overlook tonal structures, particularly musical modes, which play a critical role in emotional perception according to music psychology. In this paper, we investigate the representational capacity of MIDIBERT and identify its limitations in capturing mode-emotion associations. To address this issue, we propose a Mode-Guided Enhancement (MoGE) strategy that incorporates psychological insights on mode into the model. Specifically, we first conduct a mode augmentation analysis, which reveals that MIDIBERT fails to effectively encode emotion-mode correlations. We then identify the least emotion-relevant layer within MIDIBERT and introduce a Mode-guided Feature-wise linear modulation injection (MoFi) framework to inject explicit mode features, thereby enhancing the model's capability in emotional representation and inference. Extensive experiments on the EMOPIA and VGMIDI datasets demonstrate that our mode injection strategy significantly improves SMER performance, achieving accuracies of 75.2% and 59.1%, respectively. These results validate the effectiveness of mode-guided modeling in symbolic music emotion recognition.
>
---
#### [new 002] Speaker Recognition -- Wavelet Packet Based Multiresolution Feature Extraction Approach
- **分类: cs.SD**

- **简介: 该论文面向文本无关的说话人识别任务，旨在提升特征鲁棒性与判别性。提出融合MFCC与小波包变换（WPT）的多分辨率特征提取方法，结合GMM（识别）和HMM（验证）分类器，在VoxForge和TIMIT数据集上验证，显著提升噪声鲁棒性与识别性能。**

- **链接: [https://arxiv.org/pdf/2512.18902v1](https://arxiv.org/pdf/2512.18902v1)**

> **作者:** Saurabh Bhardwaj; Smriti Srivastava; Abhishek Bhandari; Krit Gupta; Hitesh Bahl; J. R. P. Gupta
>
> **备注:** This paper was originally written in Summer 2013 and previously made available on Figshare. The present submission is uploaded for archival and citation purposes
>
> **摘要:** This paper proposes a novel Wavelet Packet based feature extraction approach for the task of text independent speaker recognition. The features are extracted by using the combination of Mel Frequency Cepstral Coefficient (MFCC) and Wavelet Packet Transform (WPT).Hybrid Features technique uses the advantage of human ear simulation offered by MFCC combining it with multi-resolution property and noise robustness of WPT. To check the validity of the proposed approach for the text independent speaker identification and verification we have used the Gaussian Mixture Model (GMM) and Hidden Markov Model (HMM) respectively as the classifiers. The proposed paradigm is tested on voxforge speech corpus and CSTR US KED Timit database. The paradigm is also evaluated after adding standard noise signal at different level of SNRs for evaluating the noise robustness. Experimental results show that better results are achieved for the tasks of both speaker identification as well as speaker verification.
>
---
#### [new 003] X-Talk: On the Underestimated Potential of Modular Speech-to-Speech Dialogue System
- **分类: cs.SD**

- **简介: 该论文提出X-Talk框架，面向语音到语音（S2S）对话系统任务，解决端到端模型难以兼顾多目标与低延迟的问题。工作包括构建模块化级联架构，集成VAD、ASR、情感分析、RAG和工具调用等专用组件，在保障亚秒延迟的同时提升灵活性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.18706v1](https://arxiv.org/pdf/2512.18706v1)**

> **作者:** Zhanxun Liu; Yifan Duan; Mengmeng Wang; Pengchao Feng; Haotian Zhang; Xiaoyu Xing; Yijia Shan; Haina Zhu; Yuhang Dai; Chaochao Lu; Xipeng Qiu; Lei Xie; Lan Wang; Nan Yan; Zilong Zheng; Ziyang Ma; Kai Yu; Xie Chen
>
> **备注:** 14 pages
>
> **摘要:** We present X-Talk, an open-source framework that champions a decoupled, modular design for LLM-driven speech-to-speech (S2S) systems. While the dominant trend favors end-to-end (E2E) modeling to optimize information flow, these "omni-models" often struggle to balance the competing objectives of complex speech tasks within a single network. X-Talk challenges this paradigm by demonstrating that a systematically optimized cascaded pipeline can achieve sub-second latency without sacrificing modular flexibility. Our framework seamlessly integrates specialized front-end components (e.g., VAD, speech enhancement) and diverse understanding models (e.g., ASR, emotion, and environmental sound analysis) with LLM capabilities like retrieval-augmented generation (RAG) and tool use. By revitalizing the cascaded approach, X-Talk highlights the underestimated potential of modular S2S systems and provides a robust foundation for future research and applications.
>
---
#### [new 004] Pushing the Frontier of Audiovisual Perception with Large-Scale Multimodal Correspondence Learning
- **分类: cs.SD; cs.CV; cs.LG**

- **简介: 该论文提出PE-AV模型，解决多模态（音频-视频-文本）表征学习不统一、跨模态对齐弱的问题。通过构建亿级音视频-文本对数据集，采用十种对比学习目标联合训练，实现统一跨模态嵌入，并拓展至帧级对齐（PE-A-Frame），提升语音检索、声音事件检测等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.19687v1](https://arxiv.org/pdf/2512.19687v1)**

> **作者:** Apoorv Vyas; Heng-Jui Chang; Cheng-Fu Yang; Po-Yao Huang; Luya Gao; Julius Richter; Sanyuan Chen; Matt Le; Piotr Dollár; Christoph Feichtenhofer; Ann Lee; Wei-Ning Hsu
>
> **摘要:** We introduce Perception Encoder Audiovisual, PE-AV, a new family of encoders for audio and video understanding trained with scaled contrastive learning. Built on PE, PE-AV makes several key contributions to extend representations to audio, and natively support joint embeddings across audio-video, audio-text, and video-text modalities. PE-AV's unified cross-modal embeddings enable novel tasks such as speech retrieval, and set a new state of the art across standard audio and video benchmarks. We unlock this by building a strong audiovisual data engine that synthesizes high-quality captions for O(100M) audio-video pairs, enabling large-scale supervision consistent across modalities. Our audio data includes speech, music, and general sound effects-avoiding single-domain limitations common in prior work. We exploit ten pairwise contrastive objectives, showing that scaling cross-modality and caption-type pairs strengthens alignment and improves zero-shot performance. We further develop PE-A-Frame by fine-tuning PE-AV with frame-level contrastive objectives, enabling fine-grained audio-frame-to-text alignment for tasks such as sound event detection.
>
---
#### [new 005] A Data-Centric Approach to Generalizable Speech Deepfake Detection
- **分类: cs.SD; eess.SP**

- **简介: 该论文属语音深度伪造检测（SDD）任务，旨在提升模型对未知伪造方法的泛化能力。针对数据组成影响被忽视的问题，提出数据-centric方法：分析数据规模与多样性规律，并设计多样性优化采样策略（DOSS），显著提升检测性能与数据效率。**

- **链接: [https://arxiv.org/pdf/2512.18210v1](https://arxiv.org/pdf/2512.18210v1)**

> **作者:** Wen Huang; Yuchen Mao; Yanmin Qian
>
> **摘要:** Achieving robust generalization in speech deepfake detection (SDD) remains a primary challenge, as models often fail to detect unseen forgery methods. While research has focused on model-centric and algorithm-centric solutions, the impact of data composition is often underexplored. This paper proposes a data-centric approach, analyzing the SDD data landscape from two practical perspectives: constructing a single dataset and aggregating multiple datasets. To address the first perspective, we conduct a large-scale empirical study to characterize the data scaling laws for SDD, quantifying the impact of source and generator diversity. To address the second, we propose the Diversity-Optimized Sampling Strategy (DOSS), a principled framework for mixing heterogeneous data with two implementations: DOSS-Select (pruning) and DOSS-Weight (re-weighting). Our experiments show that DOSS-Select outperforms the naive aggregation baseline while using only 3% of the total available data. Furthermore, our final model, trained on a 12k-hour curated data pool using the optimal DOSS-Weight strategy, achieves state-of-the-art performance, outperforming large-scale baselines with greater data and model efficiency on both public benchmarks and a new challenge set of various commercial APIs.
>
---
#### [new 006] Explainable Transformer-CNN Fusion for Noise-Robust Speech Emotion Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文面向噪声鲁棒的语音情感识别（SER）任务，解决真实环境中噪声干扰导致性能下降及模型不可解释问题。提出可解释的Transformer-CNN融合框架，结合Wav2Vec 2.0与1D-CNN，引入注意力时序池化和SHAP/Score-CAM可视化解释机制。**

- **链接: [https://arxiv.org/pdf/2512.18298v1](https://arxiv.org/pdf/2512.18298v1)**

> **作者:** Sudip Chakrabarty; Pappu Bishwas; Rajdeep Chatterjee
>
> **摘要:** Speech Emotion Recognition (SER) systems often degrade in performance when exposed to the unpredictable acoustic interference found in real-world environments. Additionally, the opacity of deep learning models hinders their adoption in trust-sensitive applications. To bridge this gap, we propose a Hybrid Transformer-CNN framework that unifies the contextual modeling of Wav2Vec 2.0 with the spectral stability of 1D-Convolutional Neural Networks. Our dual-stream architecture processes raw waveforms to capture long-range temporal dependencies while simultaneously extracting noise-resistant spectral features (MFCC, ZCR, RMSE) via a custom Attentive Temporal Pooling mechanism. We conducted extensive validation across four diverse benchmark datasets: RAVDESS, TESS, SAVEE, and CREMA-D. To rigorously test robustness, we subjected the model to non-stationary acoustic interference using real-world noise profiles from the SAS-KIIT dataset. The proposed framework demonstrates superior generalization and state-of-the-art accuracy across all datasets, significantly outperforming single-branch baselines under realistic environmental interference. Furthermore, we address the ``black-box" problem by integrating SHAP and Score-CAM into the evaluation pipeline. These tools provide granular visual explanations, revealing how the model strategically shifts attention between temporal and spectral cues to maintain reliability in the presence of complex environmental noise.
>
---
#### [new 007] MeanFlow-TSE: One-Step Generative Target Speaker Extraction with Mean Flow
- **分类: eess.AS**

- **简介: 该论文面向目标说话人提取（TSE）任务，旨在解决现有扩散/流匹配模型需多步采样、实时性差的问题。作者提出MeanFlow-TSE，基于均值流目标实现单步生成式说话人分离，在保持高质量的同时显著提升推理速度。**

- **链接: [https://arxiv.org/pdf/2512.18572v1](https://arxiv.org/pdf/2512.18572v1)**

> **作者:** Riki Shimizu; Xilin Jiang; Nima Mesgarani
>
> **备注:** 6 pages, 2 figures, 2 tables
>
> **摘要:** Target speaker extraction (TSE) aims to isolate a desired speaker's voice from a multi-speaker mixture using auxiliary information such as a reference utterance. Although recent advances in diffusion and flow-matching models have improved TSE performance, these methods typically require multi-step sampling, which limits their practicality in low-latency settings. In this work, we propose MeanFlow-TSE, a one-step generative TSE framework trained with mean-flow objectives, enabling fast and high-quality generation without iterative refinement. Building on the AD-FlowTSE paradigm, our method defines a flow between the background and target source that is governed by the mixing ratio (MR). Experiments on the Libri2Mix corpus show that our approach outperforms existing diffusion- and flow-matching-based TSE models in separation quality and perceptual metrics while requiring only a single inference step. These results demonstrate that mean-flow-guided one-step generation offers an effective and efficient alternative for real-time target speaker extraction. Code is available at https://github.com/rikishimizu/MeanFlow-TSE.
>
---
#### [new 008] Enhancing Fully Formatted End-to-End Speech Recognition with Knowledge Distillation via Multi-Codebook Vector Quantization
- **分类: eess.AS**

- **简介: 该论文属端到端语音识别任务，旨在解决传统ASR输出无标点/大小写的可读性差及级联后处理带来的延迟问题。提出基于多码本向量量化知识蒸馏的全格式E2E模型，直接输出带标点与大小写的文本，在WER和PER上达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.18967v1](https://arxiv.org/pdf/2512.18967v1)**

> **作者:** Jian You; Xiangfeng Li; Erwan Zerhouni
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Conventional automatic speech recognition (ASR) models typically produce outputs as normalized texts lacking punctuation and capitalization, necessitating post-processing models to enhance readability. This approach, however, introduces additional complexity and latency due to the cascaded system design. In response to this challenge, there is a growing trend to develop end-to-end (E2E) ASR models capable of directly predicting punctuation and capitalization, though this area remains underexplored. In this paper, we propose an enhanced fully formatted E2E ASR model that leverages knowledge distillation (KD) through multi-codebook vector quantization (MVQ). Experimental results demonstrate that our model significantly outperforms previous works in word error rate (WER) both with and without punctuation and capitalization, and in punctuation error rate (PER). Evaluations on the LibriSpeech-PC test-clean and test-other subsets show that our model achieves state-of-the-art results.
>
---
#### [new 009] chatter: a Python library for applying information theory and AI/ML models to animal communication
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出chatter——一个Python库，用于在连续隐空间中分析动物通信。它解决传统离散分类法丢失复杂性的问题，采用VAE、ViT等模型将声音序列映射为隐空间轨迹，支持端到端分析，量化复杂度、可预测性等特性。**

- **链接: [https://arxiv.org/pdf/2512.17935v1](https://arxiv.org/pdf/2512.17935v1)**

> **作者:** Mason Youngblood
>
> **摘要:** The study of animal communication often involves categorizing units into types (e.g. syllables in songbirds, or notes in humpback whales). While this approach is useful in many cases, it necessarily flattens the complexity and nuance present in real communication systems. chatter is a new Python library for analyzing animal communication in continuous latent space using information theory and modern machine learning techniques. It is taxonomically agnostic, and has been tested with the vocalizations of birds, bats, whales, and primates. By leveraging a variety of different architectures, including variational autoencoders and vision transformers, chatter represents vocal sequences as trajectories in high-dimensional latent space, bypassing the need for manual or automatic categorization of units. The library provides an end-to-end workflow -- from preprocessing and segmentation to model training and feature extraction -- that enables researchers to quantify the complexity, predictability, similarity, and novelty of vocal sequences.
>
---
#### [new 010] SAM Audio: Segment Anything in Audio
- **分类: eess.AS**

- **简介: 该论文提出SAM Audio，面向通用音频源分离任务，解决现有模型领域受限、提示模态单一的问题。工作包括：构建支持文本/视觉/时序多模态提示的扩散Transformer基础模型，基于流匹配在大规模异构音频上训练，并建立新基准与无参考评估模型。**

- **链接: [https://arxiv.org/pdf/2512.18099v1](https://arxiv.org/pdf/2512.18099v1)**

> **作者:** Bowen Shi; Andros Tjandra; John Hoffman; Helin Wang; Yi-Chiao Wu; Luya Gao; Julius Richter; Matt Le; Apoorv Vyas; Sanyuan Chen; Christoph Feichtenhofer; Piotr Dollár; Wei-Ning Hsu; Ann Lee
>
> **摘要:** General audio source separation is a key capability for multimodal AI systems that can perceive and reason about sound. Despite substantial progress in recent years, existing separation models are either domain-specific, designed for fixed categories such as speech or music, or limited in controllability, supporting only a single prompting modality such as text. In this work, we present SAM Audio, a foundation model for general audio separation that unifies text, visual, and temporal span prompting within a single framework. Built on a diffusion transformer architecture, SAM Audio is trained with flow matching on large-scale audio data spanning speech, music, and general sounds, and can flexibly separate target sources described by language, visual masks, or temporal spans. The model achieves state-of-the-art performance across a diverse suite of benchmarks, including general sound, speech, music, and musical instrument separation in both in-the-wild and professionally produced audios, substantially outperforming prior general-purpose and specialized systems. Furthermore, we introduce a new real-world separation benchmark with human-labeled multimodal prompts and a reference-free evaluation model that correlates strongly with human judgment.
>
---
#### [new 011] TICL+: A Case Study On Speech In-Context Learning for Children's Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属儿童语音识别任务，旨在解决儿童语音声学/语言变异大、标注数据少导致的ASR性能差问题。提出TICL+方法，在检索式语音上下文学习（SICL）中引入声学重排序，联合语义与声学对齐选取示例，显著降低词错误率。**

- **链接: [https://arxiv.org/pdf/2512.18263v1](https://arxiv.org/pdf/2512.18263v1)**

> **作者:** Haolong Zheng; Yekaterina Yegorova; Mark Hasegawa-Johnson
>
> **备注:** Published at IEEE ASRU 2025 Satellite Workshop-AI for Children's Speech and Language
>
> **摘要:** Children's speech recognition remains challenging due to substantial acoustic and linguistic variability, limited labeled data, and significant differences from adult speech. Speech foundation models can address these challenges through Speech In-Context Learning (SICL), allowing adaptation to new domains without fine-tuning. However, the effectiveness of SICL depends on how in-context examples are selected. We extend an existing retrieval-based method, Text-Embedding KNN for SICL (TICL), introducing an acoustic reranking step to create TICL+. This extension prioritizes examples that are both semantically and acoustically aligned with the test input. Experiments on four children's speech corpora show that TICL+ achieves up to a 53.3% relative word error rate reduction over zero-shot performance and 37.6% over baseline TICL, highlighting the value of combining semantic and acoustic information for robust, scalable ASR in children's speech.
>
---
#### [new 012] DeepGESI: A Non-Intrusive Objective Evaluation Model for Predicting Speech Intelligibility in Hearing-Impaired Listeners
- **分类: cs.SD**

- **简介: 该论文提出DeepGESI模型，解决听力障碍者语音可懂度的非侵入式客观评估问题。现有指标（如STOI、GESI）需清洁参考语音，不适用于真实场景。DeepGESI基于深度学习，无需参考信号，可快速准确预测GESI分数，在CPC2数据集上验证了其高相关性与高效性。**

- **链接: [https://arxiv.org/pdf/2512.19374v1](https://arxiv.org/pdf/2512.19374v1)**

> **作者:** Wenyu Luo; Jinhui Chen
>
> **摘要:** Speech intelligibility assessment is essential for many speech-related applications. However, most objective intelligibility metrics are intrusive, as they require clean reference speech in addition to the degraded or processed signal for evaluation. Furthermore, existing metrics such as STOI are primarily designed for normal hearing listeners, and their predictive accuracy for hearing impaired speech intelligibility remains limited. On the other hand, the GESI (Gammachirp Envelope Similarity Index) can be used to estimate intelligibility for hearing-impaired listeners, but it is also intrusive, as it depends on reference signals. This requirement limits its applicability in real-world scenarios. To overcome this limitation, this study proposes DeepGESI, a non-intrusive deep learning-based model capable of accurately and efficiently predicting the speech intelligibility of hearing-impaired listeners without requiring any clean reference speech. Experimental results demonstrate that, under the test conditions of the 2nd Clarity Prediction Challenge(CPC2) dataset, the GESI scores predicted by DeepGESI exhibit a strong correlation with the actual GESI scores. In addition, the proposed model achieves a substantially faster prediction speed compared to conventional methods.
>
---
#### [new 013] Task Vector in TTS: Toward Emotionally Expressive Dialectal Speech Synthesis
- **分类: cs.SD; cs.LG**

- **简介: 该论文面向情感化方言语音合成任务，解决方言与情感联合建模因标注数据稀缺而难以实现的问题。提出分两阶段的HE-Vector方法：先独立建模方言和情感的E-Vector并加权优化单风格合成；再分层融合二者，实现无需联合标注的可控情感方言TTS。**

- **链接: [https://arxiv.org/pdf/2512.18699v1](https://arxiv.org/pdf/2512.18699v1)**

> **作者:** Pengchao Feng; Yao Xiao; Ziyang Ma; Zhikang Niu; Shuai Fan; Yao Li; Sheng Wang; Xie Chen
>
> **摘要:** Recent advances in text-to-speech (TTS) have yielded remarkable improvements in naturalness and intelligibility. Building on these achievements, research has increasingly shifted toward enhancing the expressiveness of generated speech, such as dialectal and emotional TTS. However, cross-style synthesis combining both dialect and emotion remains challenging and largely unexplored, mainly due to the scarcity of dialectal data with emotional labels. To address this, we propose Hierarchical Expressive Vector (HE-Vector), a two-stage method for Emotional Dialectal TTS. In the first stage, we construct different task vectors to model dialectal and emotional styles independently, and then enhance single-style synthesis by adjusting their weights, a method we refer to as Expressive Vector (E-Vector). For the second stage, we hierarchically integrate these vectors to achieve controllable emotionally expressive dialect synthesis without requiring jointly labeled data, corresponding to Hierarchical Expressive Vector (HE-Vector). Experimental results demonstrate that HE-Vectors achieve superior performance in dialect synthesis, and promising results in synthesizing emotionally expressive dialectal speech in a zero-shot setting.
>
---
#### [new 014] Continual Learning for Acoustic Event Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究持续学习下的声学事件分类任务，旨在解决边缘设备上增量学习新类别时的灾难性遗忘问题。提出RK方法：基于样本分类不确定性选择多样性历史数据，结合数据增强与知识蒸馏，降低内存与计算开销。**

- **链接: [https://arxiv.org/pdf/2512.17932v1](https://arxiv.org/pdf/2512.17932v1)**

> **作者:** Yang Xiao
>
> **备注:** Master project report
>
> **摘要:** Continuously learning new classes without catastrophic forgetting is a challenging problem for on-device acoustic event classification given the restrictions on computation resources (e.g., model size, running memory). To alleviate such an issue, we propose two novel diversity-aware incremental learning method for Spoken Keyword Spotting and Environmental Sound Classification. Our method selects the historical data for the training by measuring the per-sample classification uncertainty. For the Spoken Keyword Spotting application, the proposed RK approach introduces a diversity-aware sampler to select a diverse set from historical and incoming keywords by calculating classification uncertainty. As a result, the RK approach can incrementally learn new tasks without forgetting prior knowledge. Besides, the RK approach also proposes data augmentation and knowledge distillation loss function for efficient memory management on the edge device. For the Environmental Sound Classification application, we measure the uncertainty by observing how the classification probability of data fluctuates against the parallel perturbations added to the classifier embedding. In this way, the computation cost can be significantly reduced compared with adding perturbation to the raw data. Experimental results show that the proposed RK approach achieves 4.2% absolute improvement in terms of average accuracy over the best baseline on Google Speech Command dataset with less required memory. Experimental results on the DCASE 2019 Task 1 and ESC-50 dataset show that our proposed method outperforms baseline continual learning methods on classification accuracy and computational efficiency, indicating our method can efficiently and incrementally learn new classes without the catastrophic forgetting problem for on-device environmental sound classification
>
---
#### [new 015] Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform
- **分类: cs.SD; cs.AI; cs.CR**

- **简介: 该论文属音频水印任务，旨在解决TTS扩散模型的知识产权保护与语音溯源难题。提出通用水印方案Smark，利用离散小波变换（DWT）在低频区域嵌入水印，适配各类TTS扩散模型，兼顾高音质与强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.18791v1](https://arxiv.org/pdf/2512.18791v1)**

> **作者:** Yichuan Zhang; Chengxin Li; Yujie Gu
>
> **摘要:** Text-to-Speech (TTS) diffusion models generate high-quality speech, which raises challenges for the model intellectual property protection and speech tracing for legal use. Audio watermarking is a promising solution. However, due to the structural differences among various TTS diffusion models, existing watermarking methods are often designed for a specific model and degrade audio quality, which limits their practical applicability. To address this dilemma, this paper proposes a universal watermarking scheme for TTS diffusion models, termed Smark. This is achieved by designing a lightweight watermark embedding framework that operates in the common reverse diffusion paradigm shared by all TTS diffusion models. To mitigate the impact on audio quality, Smark utilizes the discrete wavelet transform (DWT) to embed watermarks into the relatively stable low-frequency regions of the audio, which ensures seamless watermark-audio integration and is resistant to removal during the reverse diffusion process. Extensive experiments are conducted to evaluate the audio quality and watermark performance in various simulated real-world attack scenarios. The experimental results show that Smark achieves superior performance in both audio quality and watermark extraction accuracy.
>
---
#### [new 016] Reliable Audio Deepfake Detection in Variable Conditions via Quantum-Kernel SVMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文属音频深度伪造检测任务，旨在解决小样本、多变录音条件下模型泛化差、误报率高的问题。提出用经典计算机模拟的量子核SVM（QSVM），仅替换核函数，不增参数，在四个数据集上显著降低等错误率与误报率。**

- **链接: [https://arxiv.org/pdf/2512.18797v1](https://arxiv.org/pdf/2512.18797v1)**

> **作者:** Lisan Al Amin; Vandana P. Janeja
>
> **备注:** This paper is accepted in ICDM 2025-MLC workshop
>
> **摘要:** Detecting synthetic speech is challenging when labeled data are scarce and recording conditions vary. Existing end-to-end deep models often overfit or fail to generalize, and while kernel methods can remain competitive, their performance heavily depends on the chosen kernel. Here, we show that using a quantum kernel in audio deepfake detection reduces falsepositive rates without increasing model size. Quantum feature maps embed data into high-dimensional Hilbert spaces, enabling the use of expressive similarity measures and compact classifiers. Building on this motivation, we compare quantum-kernel SVMs (QSVMs) with classical SVMs using identical mel-spectrogram preprocessing and stratified 5-fold cross-validation across four corpora (ASVspoof 2019 LA, ASVspoof 5 (2024), ADD23, and an In-the-Wild set). QSVMs achieve consistently lower equalerror rates (EER): 0.183 vs. 0.299 on ASVspoof 5 (2024), 0.081 vs. 0.188 on ADD23, 0.346 vs. 0.399 on ASVspoof 2019, and 0.355 vs. 0.413 In-the-Wild. At the EER operating point (where FPR equals FNR), these correspond to absolute false-positiverate reductions of 0.116 (38.8%), 0.107 (56.9%), 0.053 (13.3%), and 0.058 (14.0%), respectively. We also report how consistent the results are across cross-validation folds and margin-based measures of class separation, using identical settings for both models. The only modification is the kernel; the features and SVM remain unchanged, no additional trainable parameters are introduced, and the quantum kernel is computed on a conventional computer.
>
---
#### [new 017] Influence of string register locations on vibratos among violoncellists
- **分类: cs.SD**

- **简介: 该论文属音乐声学/演奏行为分析任务，探究大提琴手在不同把位（弦上手指位置）时颤音（vibrato）的声学深度与手指运动幅度变化关系。通过分析94段演奏片段，发现靠近琴桥时声学颤音加深但手指摆幅减小，揭示演奏者存在但不充分的补偿性控制。**

- **链接: [https://arxiv.org/pdf/2512.18162v1](https://arxiv.org/pdf/2512.18162v1)**

> **作者:** Steven Hu; Sophia H. Kim; Helena H. Kim; Hugo Mackay; Eric J. Heller
>
> **摘要:** This study analyzes how vibrato changes with finger position along the cello string. Examining 94 excerpts, we found moving the finger toward the bridge strongly increases acoustic vibrato depth ($ρ=0.6902$, $p=1.408\cdot 10^{-14}$). However, the performer's physical finger amplitude simultaneously decreases ($ρ=-0.6391$, $p=4.172\cdot 10^{-12}$). This shows players reduce finger motion in higher positions, but not enough to counteract the greater pitch deviation there, revealing both the presence and limits of compensatory vibrato behavior.
>
---
#### [new 018] LIWhiz: A Non-Intrusive Lyric Intelligibility Prediction System for the Cadenza Challenge
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出LIWhiz系统，解决歌词可懂度预测任务，即非侵入式估计演唱中歌词的清晰度。它利用Whisper提取鲁棒语音特征，并通过可训练后端预测 intelligibility 分数，在Cadenza挑战赛CLIP测试集上显著优于STOI基线。**

- **链接: [https://arxiv.org/pdf/2512.17937v1](https://arxiv.org/pdf/2512.17937v1)**

> **作者:** Ram C. M. C. Shekar; Iván López-Espejo
>
> **摘要:** We present LIWhiz, a non-intrusive lyric intelligibility prediction system submitted to the ICASSP 2026 Cadenza Challenge. LIWhiz leverages Whisper for robust feature extraction and a trainable back-end for score prediction. Tested on the Cadenza Lyric Intelligibility Prediction (CLIP) evaluation set, LIWhiz achieves a 22.4% relative root mean squared error reduction over the STOI-based baseline, yielding a substantial improvement in normalized cross-correlation.
>
---
#### [new 019] JoyVoice: Long-Context Conditioning for Anthropomorphic Multi-Speaker Conversational Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出JoyVoice模型，解决多说话人、长对话语音合成任务中上下文建模僵化、音色与韵律不自然的问题。工作包括：设计端到端Transformer-DiT架构、低比特率MM-Tokenizer、鲁棒文本前端，支持最多八人自由对话，提升韵律连续性、节奏丰富性与零-shot克隆能力。**

- **链接: [https://arxiv.org/pdf/2512.19090v1](https://arxiv.org/pdf/2512.19090v1)**

> **作者:** Fan Yu; Tao Wang; You Wu; Lin Zhu; Wei Deng; Weisheng Han; Wenchao Wang; Lin Hu; Xiangyu Liang; Xiaodong He; Yankun Huang; Yu Gu; Yuan Liu; Yuxuan Wang; Zhangyu Xiao; Ziteng Wang; Boya Dong; Feng Dang; Jinming Chen; Jingdong Li; Jun Wang; Yechen Jin; Yuan Zhang; Zhengyan Sheng; Xin Wang
>
> **摘要:** Large speech generation models are evolving from single-speaker, short sentence synthesis to multi-speaker, long conversation geneartion. Current long-form speech generation models are predominately constrained to dyadic, turn-based interactions. To address this, we introduce JoyVoice, a novel anthropomorphic foundation model designed for flexible, boundary-free synthesis of up to eight speakers. Unlike conventional cascaded systems, JoyVoice employs a unified E2E-Transformer-DiT architecture that utilizes autoregressive hidden representations directly for diffusion inputs, enabling holistic end-to-end optimization. We further propose a MM-Tokenizer operating at a low bitrate of 12.5 Hz, which integrates multitask semantic and MMSE losses to effectively model both semantic and acoustic information. Additionally, the model incorporates robust text front-end processing via large-scale data perturbation. Experiments show that JoyVoice achieves state-of-the-art results in multilingual generation (Chinese, English, Japanese, Korean) and zero-shot voice cloning. JoyVoice achieves top-tier results on both the Seed-TTS-Eval Benchmark and multi-speaker long-form conversational voice cloning tasks, demonstrating superior audio quality and generalization. It achieves significant improvements in prosodic continuity for long-form speech, rhythm richness in multi-speaker conversations, paralinguistic naturalness, besides superior intelligibility. We encourage readers to listen to the demo at https://jea-speech.github.io/JoyVoice
>
---
#### [new 020] AutoSchA: Automatic Hierarchical Music Representations via Multi-Relational Node Isolation
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出AutoSchA，属自动音乐分析任务，旨在解决人工分层音乐分析（如Schenkerian分析）耗时费力、难数字化的问题。工作包括：构建图神经网络框架、设计基于节点隔离的图池化机制，并集成成端到端自动分层表示模型，在巴洛克赋格主题上达专家水平。**

- **链接: [https://arxiv.org/pdf/2512.18232v1](https://arxiv.org/pdf/2512.18232v1)**

> **作者:** Stephen Ni-Hahn; Rico Zhu; Jerry Yin; Yue Jiang; Cynthia Rudin; Simon Mak
>
> **摘要:** Hierarchical representations provide powerful and principled approaches for analyzing many musical genres. Such representations have been broadly studied in music theory, for instance via Schenkerian analysis (SchA). Hierarchical music analyses, however, are highly cost-intensive; the analysis of a single piece of music requires a great deal of time and effort from trained experts. The representation of hierarchical analyses in a computer-readable format is a further challenge. Given recent developments in hierarchical deep learning and increasing quantities of computer-readable data, there is great promise in extending such work for an automatic hierarchical representation framework. This paper thus introduces a novel approach, AutoSchA, which extends recent developments in graph neural networks (GNNs) for hierarchical music analysis. AutoSchA features three key contributions: 1) a new graph learning framework for hierarchical music representation, 2) a new graph pooling mechanism based on node isolation that directly optimizes learned pooling assignments, and 3) a state-of-the-art architecture that integrates such developments for automatic hierarchical music analysis. We show, in a suite of experiments, that AutoSchA performs comparably to human experts when analyzing Baroque fugue subjects.
>
---
#### [new 021] What Does the Speaker Embedding Encode?
- **分类: eess.AS**

- **简介: 该论文属语音表示分析任务，旨在探究i-vector、d-vector和s-vector等说话人嵌入究竟编码了哪些属性（如身份、性别、语速、文本内容等）。通过多维度分类实验揭示各方法优劣，并提出融合i-vector与s-vector的i-s-vector新嵌入，在RSR2015上显著降低内容不匹配场景下的EER。**

- **链接: [https://arxiv.org/pdf/2512.18286v1](https://arxiv.org/pdf/2512.18286v1)**

> **作者:** Shuai Wang; Yanmin Qian; Kai Yu
>
> **备注:** This paper was accepted by Interspeech 2017. However, no public version is currently available, as the original link provided by ISCA is no longer accessible. The version uploaded herein has undergone automatic English polishing using GPT (Expanded for better calarity)
>
> **摘要:** Developing a good speaker embedding has received tremendous interest in the speech community, with representations such as i-vector and d-vector demonstrating remarkable performance across various tasks. Despite their widespread adoption, a fundamental question remains largely unexplored: what properties are actually encoded in these embeddings? To address this gap, we conduct a comprehensive analysis of three prominent speaker embedding methods: i-vector, d-vector, and RNN/LSTM-based sequence-vector (s-vector). Through carefully designed classification tasks, we systematically investigate their encoding capabilities across multiple dimensions, including speaker identity, gender, speaking rate, text content, word order, and channel information. Our analysis reveals distinct strengths and limitations of each embedding type: i-vector excels at speaker discrimination but encodes limited sequential information; s-vector captures text content and word order effectively but struggles with speaker identity; d-vector shows balanced performance but loses sequential information through averaging. Based on these insights, we propose a novel multi-task learning framework that integrates i-vector and s-vector, resulting in a new speaker embedding (i-s-vector) that combines their complementary advantages. Experimental results on RSR2015 demonstrate that the proposed i-s-vector achieves more than 50% EER reduction compared to the i-vector baseline on content mismatch trials, validating the effectiveness of our approach.
>
---
#### [new 022] Phoneme-based speech recognition driven by large language models and sampling marginalization
- **分类: eess.AS; cs.SD**

- **简介: 该论文属语音识别任务，旨在解决LLM-P2G方法中TKM策略路径多样性不足、训练低效等问题。提出采样边缘化（SKM）训练策略，以随机采样替代束搜索生成候选音素路径，提升收敛速度、识别性能与效率，并验证其跨语言适用性。**

- **链接: [https://arxiv.org/pdf/2512.18371v1](https://arxiv.org/pdf/2512.18371v1)**

> **作者:** Te Ma; Nanjie Li; Hao Huang; Zhijian Ou
>
> **备注:** Published at NCMMSC 2025, in Chinese language
>
> **摘要:** Recently, the Large Language Model-based Phoneme-to-Grapheme (LLM-P2G) method has shown excellent performance in speech recognition tasks and has become a feasible direction to replace the traditional WFST decoding method. This framework takes into account both recognition accuracy and system scalability through two-stage modeling of phoneme prediction and text generation. However, the existing LLM-P2G adopts the Top-K Marginalized (TKM) training strategy, and its candidate phoneme sequences rely on beam search generation, which has problems such as insufficient path diversity, low training efficiency, and high resource overhead. To this end, this paper proposes a sampling marginalized training strategy (Sampling-K Marginalized, SKM), which replaces beam search with random sampling to generate candidate paths, improving marginalized modeling and training efficiency. Experiments were conducted on Polish and German datasets, and the results showed that SKM further improved the model learning convergence speed and recognition performance while maintaining the complexity of the model. Comparative experiments with a speech recognition method that uses a projector combined with a large language model (SpeechLLM) also show that the SKM-driven LLM-P2G has more advantages in recognition accuracy and structural simplicity. The study verified the practical value and application potential of this method in cross-language speech recognition systems.
>
---
#### [new 023] Sonified Quantum Seizures. Sonification of time series in epileptic seizures and simulation of seizures via quantum modelling
- **分类: quant-ph; cs.ET; cs.SD**

- **简介: 该论文属交叉学科任务，旨在用声学化（sonification）与量子计算联合分析癫痫发作。它将真实ECoG信号转为声音，再以两种量子模型模拟发作并声学化，通过对比声学特征评估模型保真度，为癫痫建模提供新验证方法。**

- **链接: [https://arxiv.org/pdf/2512.19272v1](https://arxiv.org/pdf/2512.19272v1)**

> **作者:** Maria Mannone; Paulo Vitor Itaborai; Omar Costa Hamido; Miriam Goldack; Norbert Marwan; Peppino Fazio; Patrizia Ribino
>
> **备注:** Presented at ISQCMC '25: 3rd International Symposium on Quantum Computing and Musical Creativity
>
> **摘要:** We apply sonification strategies and quantum computing to the analysis of an episode of seizure. We first sonify the signal from a selection of channels (from real ECoG data), obtaining a polyphonic sequence. Then, we propose two quantum approaches to simulate a similar episode of seizure, and we sonify the results. The comparison of sonifications can give hints on similarities and discrepancies between real data and simulations, helping refine the \textit{in silico} model. This is a pioneering approach, showing how the combination of quantum computing and sonification can broaden the perspective of real-data investigation, and helping define a new test bench for analysis and prediction of seizures.
>
---
#### [new 024] Tempo as the Stable Cue: Hierarchical Mixture of Tempo and Beat Experts for Music to 3D Dance Generation
- **分类: cs.CV; cs.MM; cs.SD**

- **简介: 该论文面向音乐驱动3D舞蹈生成任务，解决现有方法依赖噪声大、不普适的音乐风格标签导致节奏错位与风格漂移的问题。提出TempoMoE模块，以稳定且普适的节拍（BPM）为线索，分层混合节奏与节拍专家，实现无需人工标签的高精度节奏对齐舞蹈生成。**

- **链接: [https://arxiv.org/pdf/2512.18804v1](https://arxiv.org/pdf/2512.18804v1)**

> **作者:** Guangtao Lyu; Chenghao Xu; Qi Liu; Jiexi Yan; Muli Yang; Fen Fang; Cheng Deng
>
> **摘要:** Music to 3D dance generation aims to synthesize realistic and rhythmically synchronized human dance from music. While existing methods often rely on additional genre labels to further improve dance generation, such labels are typically noisy, coarse, unavailable, or insufficient to capture the diversity of real-world music, which can result in rhythm misalignment or stylistic drift. In contrast, we observe that tempo, a core property reflecting musical rhythm and pace, remains relatively consistent across datasets and genres, typically ranging from 60 to 200 BPM. Based on this finding, we propose TempoMoE, a hierarchical tempo-aware Mixture-of-Experts module that enhances the diffusion model and its rhythm perception. TempoMoE organizes motion experts into tempo-structured groups for different tempo ranges, with multi-scale beat experts capturing fine- and long-range rhythmic dynamics. A Hierarchical Rhythm-Adaptive Routing dynamically selects and fuses experts from music features, enabling flexible, rhythm-aligned generation without manual genre labels. Extensive experiments demonstrate that TempoMoE achieves state-of-the-art results in dance quality and rhythm alignment.
>
---
#### [new 025] MEGState: Phoneme Decoding from Magnetoencephalography Signals
- **分类: q-bio.NC; cs.LG; cs.SD**

- **简介: 该论文属神经语音解码任务，旨在解决MEG信号信噪比低、时序维度高导致的音素解码困难问题。作者提出新模型MEGState，精准捕捉听觉刺激诱发的精细皮层响应，在LibriBrain数据集上显著超越基线，推动非侵入式语音脑机接口发展。**

- **链接: [https://arxiv.org/pdf/2512.17978v1](https://arxiv.org/pdf/2512.17978v1)**

> **作者:** Shuntaro Suzuki; Chia-Chun Dan Hsu; Yu Tsao; Komei Sugiura
>
> **备注:** Accepted for presentation at LibriBrain Competition, NeurIPS 2025
>
> **摘要:** Decoding linguistically meaningful representations from non-invasive neural recordings remains a central challenge in neural speech decoding. Among available neuroimaging modalities, magnetoencephalography (MEG) provides a safe and repeatable means of mapping speech-related cortical dynamics, yet its low signal-to-noise ratio and high temporal dimensionality continue to hinder robust decoding. In this work, we introduce MEGState, a novel architecture for phoneme decoding from MEG signals that captures fine-grained cortical responses evoked by auditory stimuli. Extensive experiments on the LibriBrain dataset demonstrate that MEGState consistently surpasses baseline model across multiple evaluation metrics. These findings highlight the potential of MEG-based phoneme decoding as a scalable pathway toward non-invasive brain-computer interfaces for speech.
>
---
#### [new 026] Real-Time Streamable Generative Speech Restoration with Flow Matching
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文提出Stream.FM，一种实时流式生成式语音恢复模型，解决扩散模型计算重、延迟高、难用于实时通信的问题。通过帧因果流匹配、缓冲流推理、轻量架构与数值求解器优化，在消费级GPU上实现24–48ms低延迟，支持语音增强等多种任务，达生成式流式语音修复SOTA。**

- **链接: [https://arxiv.org/pdf/2512.19442v1](https://arxiv.org/pdf/2512.19442v1)**

> **作者:** Simon Welker; Bunlong Lay; Maris Hillemann; Tal Peer; Timo Gerkmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Diffusion-based generative models have greatly impacted the speech processing field in recent years, exhibiting high speech naturalness and spawning a new research direction. Their application in real-time communication is, however, still lagging behind due to their computation-heavy nature involving multiple calls of large DNNs. Here, we present Stream.FM, a frame-causal flow-based generative model with an algorithmic latency of 32 milliseconds (ms) and a total latency of 48 ms, paving the way for generative speech processing in real-time communication. We propose a buffered streaming inference scheme and an optimized DNN architecture, show how learned few-step numerical solvers can boost output quality at a fixed compute budget, explore model weight compression to find favorable points along a compute/quality tradeoff, and contribute a model variant with 24 ms total latency for the speech enhancement task. Our work looks beyond theoretical latencies, showing that high-quality streaming generative speech processing can be realized on consumer GPUs available today. Stream.FM can solve a variety of speech processing tasks in a streaming fashion: speech enhancement, dereverberation, codec post-filtering, bandwidth extension, STFT phase retrieval, and Mel vocoding. As we verify through comprehensive evaluations and a MUSHRA listening test, Stream.FM establishes a state-of-the-art for generative streaming speech restoration, exhibits only a reasonable reduction in quality compared to a non-streaming variant, and outperforms our recent work (Diffusion Buffer) on generative streaming speech enhancement while operating at a lower latency.
>
---
#### [new 027] MauBERT: Universal Phonetic Inductive Biases for Few-Shot Acoustic Units Discovery
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MauBERT，属语音表征学习任务，旨在解决多语言语音模型跨语言泛化弱、语音单元发现样本效率低的问题。通过在55种语言上结合发音特征监督继续预训练HuBERT，提升音素判别力与零样本迁移能力。**

- **链接: [https://arxiv.org/pdf/2512.19612v1](https://arxiv.org/pdf/2512.19612v1)**

> **作者:** Angelo Ortiz Tandazo; Manel Khentout; Youssef Benchekroun; Thomas Hueber; Emmanuel Dupoux
>
> **摘要:** This paper introduces MauBERT, a multilingual extension of HuBERT that leverages articulatory features for robust cross-lingual phonetic representation learning. We continue HuBERT pre-training with supervision based on a phonetic-to-articulatory feature mapping in 55 languages. Our models learn from multilingual data to predict articulatory features or phones, resulting in language-independent representations that capture multilingual phonetic properties. Through comprehensive ABX discriminability testing, we show MauBERT models produce more context-invariant representations than state-of-the-art multilingual self-supervised learning models. Additionally, the models effectively adapt to unseen languages and casual speech with minimal self-supervised fine-tuning (10 hours of speech). This establishes an effective approach for instilling linguistic inductive biases in self-supervised speech models.
>
---
## 更新

#### [replaced 001] A Comprehensive Survey on Generative AI for Video-to-Music Generation
- **分类: eess.AS; cs.AI; cs.MM**

- **简介: 该论文属视频到音乐生成任务，旨在解决该领域缺乏系统综述的问题。工作包括：细粒度划分视频与音乐模态；按条件输入构建、条件机制、音乐生成框架三部分分类梳理方法；总结数据集、评估指标及挑战。**

- **链接: [https://arxiv.org/pdf/2502.12489v2](https://arxiv.org/pdf/2502.12489v2)**

> **作者:** Shulei Ji; Songruoyao Wu; Zihao Wang; Shuyu Li; Kejun Zhang
>
> **摘要:** The burgeoning growth of video-to-music generation can be attributed to the ascendancy of multimodal generative models. However, there is a lack of literature that comprehensively combs through the work in this field. To fill this gap, this paper presents a comprehensive review of video-to-music generation using deep generative AI techniques, focusing on three key components: conditioning input construction, conditioning mechanism, and music generation frameworks. We categorize existing approaches based on their designs for each component, clarifying the roles of different strategies. Preceding this, we provide a fine-grained categorization of video and music modalities, illustrating how different categories influence the design of components within the generation pipelines. Furthermore, we summarize available multimodal datasets and evaluation metrics while highlighting ongoing challenges in the field.
>
---
#### [replaced 002] ASR-Synchronized Speaker-Role Diarization
- **分类: eess.AS; cs.AI; cs.LG**

- **简介: 该论文研究说话人角色二值化（RD），即识别医生/患者等语义角色，而非仅区分说话人。为提升ASR与RD联合性能，提出冻结ASR模型、并行训练专用RD模块，利用高层ASR特征与强制对齐路径上的交叉熵损失，显著降低角色词错误率。**

- **链接: [https://arxiv.org/pdf/2507.17765v3](https://arxiv.org/pdf/2507.17765v3)**

> **作者:** Arindam Ghosh; Mark Fuhs; Bongjun Kim; Anurag Chowdhury; Monika Woszczyna
>
> **备注:** Work in progress
>
> **摘要:** Speaker-role diarization (RD), such as doctor vs. patient or lawyer vs. client, is practically often more useful than conventional speaker diarization (SD), which assigns only generic labels (speaker-1, speaker-2). The state-of-the-art end-to-end ASR+RD approach uses a single transducer that serializes word and role predictions (role at the end of a speaker's turn), but at the cost of degraded ASR performance. To address this, we adapt a recent joint ASR+SD framework to ASR+RD by freezing the ASR transducer and training an auxiliary RD transducer in parallel to assign a role to each ASR-predicted word. For this, we first show that SD and RD are fundamentally different tasks, exhibiting different dependencies on acoustic and linguistic information. Motivated by this, we propose (1) task-specific predictor networks and (2) using higher-layer ASR encoder features as input to the RD encoder. Additionally, we replace the blank-shared RNNT loss by cross-entropy loss along the 1-best forced-alignment path to further improve performance while reducing computational and memory requirements during RD training. Experiments on a public and a private dataset of doctor-patient conversations demonstrate that our method outperforms the best baseline with relative reductions of 6.2% and 4.5% in role-based word diarization error rate (R-WDER), respectively
>
---
#### [replaced 003] MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion via Mean Flows
- **分类: eess.AS; cs.SD**

- **简介: 该论文面向零-shot语音转换任务，解决现有流式方法在轻量性、实时性与跨说话人泛化间的矛盾。提出MeanVC：基于均值流的轻量流式模型，融合扩散Transformer与分块自回归去噪，并引入扩散对抗后训练，实现单步高质量转换。**

- **链接: [https://arxiv.org/pdf/2510.08392v3](https://arxiv.org/pdf/2510.08392v3)**

> **作者:** Guobin Ma; Jixun Yao; Ziqian Ning; Yuepeng Jiang; Lingxin Xiong; Lei Xie; Pengcheng Zhu
>
> **摘要:** Zero-shot voice conversion (VC) aims to transfer timbre from a source speaker to any unseen target speaker while preserving linguistic content. Growing application scenarios demand models with streaming inference capabilities. This has created a pressing need for models that are simultaneously fast, lightweight, and high-fidelity. However, existing streaming methods typically rely on either autoregressive (AR) or non-autoregressive (NAR) frameworks, which either require large parameter sizes to achieve strong performance or struggle to generalize to unseen speakers. In this study, we propose MeanVC, a lightweight and streaming zero-shot VC approach. MeanVC introduces a diffusion transformer with a chunk-wise autoregressive denoising strategy, combining the strengths of both AR and NAR paradigms for efficient streaming processing. By introducing mean flows, MeanVC regresses the average velocity field during training, enabling zero-shot VC with superior speech quality and speaker similarity in a single sampling step by directly mapping from the start to the endpoint of the flow trajectory. Additionally, we incorporate diffusion adversarial post-training to mitigate over-smoothing and further enhance speech quality. Experimental results demonstrate that MeanVC significantly outperforms existing zero-shot streaming VC systems, achieving superior conversion quality with higher efficiency and significantly fewer parameters. Audio demos and code are publicly available at https://aslp-lab.github.io/MeanVC.
>
---
#### [replaced 004] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 该论文面向说话人日志与识别（SDR）任务，旨在解决传统级联系统误差传播、难处理重叠语音等问题。提出端到端多模态大模型SpeakerLM，联合完成说话人划分与语音识别，并引入灵活说话人注册机制，支持多种注册场景。**

- **链接: [https://arxiv.org/pdf/2508.06372v2](https://arxiv.org/pdf/2508.06372v2)**

> **作者:** Han Yin; Yafeng Chen; Chong Deng; Luyao Cheng; Hui Wang; Chao-Hong Tan; Qian Chen; Wen Wang; Xiangang Li
>
> **摘要:** The Speaker Diarization and Recognition (SDR) task aims to predict "who spoke when and what" within an audio clip, which is a crucial task in various real-world multi-speaker scenarios such as meeting transcription and dialogue systems. Existing SDR systems typically adopt a cascaded framework, combining multiple modules such as speaker diarization (SD) and automatic speech recognition (ASR). The cascaded systems suffer from several limitations, such as error propagation, difficulty in handling overlapping speech, and lack of joint optimization for exploring the synergy between SD and ASR tasks. To address these limitations, we introduce SpeakerLM, a unified multimodal large language model for SDR that jointly performs SD and ASR in an end-to-end manner. Moreover, to facilitate diverse real-world scenarios, we incorporate a flexible speaker registration mechanism into SpeakerLM, enabling SDR under different speaker registration settings. SpeakerLM is progressively developed with a multi-stage training strategy on large-scale real data. Extensive experiments show that SpeakerLM demonstrates strong data scaling capability and generalizability, outperforming state-of-the-art cascaded baselines on both in-domain and out-of-domain public SDR benchmarks. Furthermore, experimental results show that the proposed speaker registration mechanism effectively ensures robust SDR performance of SpeakerLM across diverse speaker registration conditions and varying numbers of registered speakers.
>
---
#### [replaced 005] Machine Unlearning in Speech Emotion Recognition via Forget Set Alone
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于机器遗忘任务，旨在解决语音情感识别模型中仅用“遗忘集”高效删除特定用户敏感语音数据的隐私问题。提出一种仅基于待遗忘数据的对抗攻击式微调方法，在不依赖原始训练数据前提下实现有效遗忘，同时保持模型识别性能。**

- **链接: [https://arxiv.org/pdf/2510.04251v2](https://arxiv.org/pdf/2510.04251v2)**

> **作者:** Zhao Ren; Rathi Adarshi Rammohan; Kevin Scheck; Tanja Schultz
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.
>
---
