# 音频 cs.SD;  eess.AS

- **最新发布 13 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] AudioGAN: A Compact and Efficient Framework for Real-Time High-Fidelity Text-to-Audio Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决现有模型速度慢、成本高的问题。提出AudioGAN框架，采用GAN结构实现高效实时生成。**

- **链接: [https://arxiv.org/pdf/2512.22166v1](https://arxiv.org/pdf/2512.22166v1)**

> **作者:** HaeChun Chung
>
> **备注:** 10 pages, 6 figures, Accepted to AES AIMLA 2025
>
> **摘要:** Text-to-audio (TTA) generation can significantly benefit the media industry by reducing production costs and enhancing work efficiency. However, most current TTA models (primarily diffusion-based) suffer from slow inference speeds and high computational costs. In this paper, we introduce AudioGAN, the first successful Generative Adversarial Networks (GANs)-based TTA framework that generates audio in a single pass, thereby reducing model complexity and inference time. To overcome the inherent difficulties in training GANs, we integrate multiple ,contrastive losses and propose innovative components Single-Double-Triple (SDT) Attention and Time-Frequency Cross-Attention (TF-CA). Extensive experiments on the AudioCaps dataset demonstrate that AudioGAN achieves state-of-the-art performance while using 90% fewer parameters and running 20 times faster, synthesizing audio in under one second. These results establish AudioGAN as a practical and powerful solution for real-time TTA.
>
---
#### [new 002] Rethinking Leveraging Pre-Trained Multi-Layer Representations for Speaker Verification
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于说话人验证任务，旨在提升预训练模型多层表示的聚合效果。提出LAP方法动态评估各层重要性，结合ASTP提取嵌入，提升性能并减少训练时间。**

- **链接: [https://arxiv.org/pdf/2512.22148v1](https://arxiv.org/pdf/2512.22148v1)**

> **作者:** Jin Sob Kim; Hyun Joon Park; Wooseok Shin; Sung Won Han
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Recent speaker verification studies have achieved notable success by leveraging layer-wise output from pre-trained Transformer models. However, few have explored the advancements in aggregating these multi-level features beyond the static weighted average. We present Layer Attentive Pooling (LAP), a novel strategy for aggregating inter-layer representations from pre-trained speech models for speaker verification. LAP assesses the significance of each layer from multiple perspectives time-dynamically, and employs max pooling instead of averaging. Additionally, we propose a lightweight backend speaker model comprising LAP and Attentive Statistical Temporal Pooling (ASTP) to extract speaker embeddings from pre-trained model output. Experiments on the VoxCeleb benchmark reveal that our compact architecture achieves state-of-the-art performance while greatly reducing the training time. We further analyzed LAP design and its dynamic weighting mechanism for capturing speaker characteristics.
>
---
#### [new 003] Marco-ASR: A Principled and Metric-Driven Framework for Fine-Tuning Large-Scale ASR Models for Domain Adaptation
- **分类: cs.SD**

- **简介: 该论文属于语音识别领域，解决ASR模型在特定领域性能下降的问题。通过优化学习率和数据增强，提升模型适应性。**

- **链接: [https://arxiv.org/pdf/2512.22165v1](https://arxiv.org/pdf/2512.22165v1)**

> **作者:** Xuanfan Ni; Fei Yang; Fengping Tian; Qingjuan Li; Chenyang Lyu; Yichao Du; Longyue Wang; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report
>
> **摘要:** Automatic Speech Recognition (ASR) models have achieved remarkable accuracy in general settings, yet their performance often degrades in domain-specific applications due to data mismatch and linguistic variability. This challenge is amplified for modern Large Language Model (LLM)-based ASR systems, whose massive scale and complex training dynamics make effective fine-tuning non-trivial. To address this gap, this paper proposes a principled and metric-driven fine-tuning framework for adapting both traditional and LLM-based ASR models to specialized domains. The framework emphasizes learning rate optimization based on performance metrics, combined with domain-specific data transformation and augmentation. We empirically evaluate our framework on state-of-the-art models, including Whisper, Whisper-Turbo, and Qwen2-Audio, across multi-domain, multilingual, and multi-length datasets. Our results not only validate the proposed framework but also establish practical protocols for improving domain-specific ASR performance while preventing overfitting.
>
---
#### [new 004] A Robust framework for sound event localization and detection on real recordings
- **分类: cs.SD**

- **简介: 该论文属于声事件定位与检测任务，旨在准确识别声音事件及其位置。通过设计鲁棒框架，结合数据增强和模型集成，提升真实场景下的性能。**

- **链接: [https://arxiv.org/pdf/2512.22156v1](https://arxiv.org/pdf/2512.22156v1)**

> **作者:** Jin Sob Kim; Hyun Joon Park; Wooseok Shin; Sung Won Han
>
> **备注:** Technical Report submitted to DCASE 2022 Challenge Task 3 (Winner of the Judge's Award)
>
> **摘要:** This technical report describes the systems submitted to the DCASE2022 challenge task 3: sound event localization and detection (SELD). The task aims to detect occurrences of sound events and specify their class, furthermore estimate their position. Our system utilizes a ResNet-based model under a proposed robust framework for SELD. To guarantee the generalized performance on the real-world sound scenes, we design the total framework with augmentation techniques, a pipeline of mixing datasets from real-world sound scenes and emulations, and test time augmentation. Augmentation techniques and exploitation of external sound sources enable training diverse samples and keeping the opportunity to train the real-world context enough by maintaining the number of the real recording samples in the batch. In addition, we design a test time augmentation and a clustering-based model ensemble method to aggregate confident predictions. Experimental results show that the model under a proposed framework outperforms the baseline methods and achieves competitive performance in real-world sound recordings.
>
---
#### [new 005] Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation
- **分类: eess.AS**

- **简介: 该论文提出Flow2GAN，解决音频生成中的质量与效率问题。结合流匹配与GAN，采用多分辨率网络，实现高质量、少步骤音频生成。**

- **链接: [https://arxiv.org/pdf/2512.23278v1](https://arxiv.org/pdf/2512.23278v1)**

> **作者:** Zengwei Yao; Wei Kang; Han Zhu; Liyong Guo; Lingxuan Ye; Fangjun Kuang; Weiji Zhuang; Zhaoqing Li; Zhifeng Han; Long Lin; Daniel Povey
>
> **摘要:** Existing dominant methods for audio generation include Generative Adversarial Networks (GANs) and diffusion-based methods like Flow Matching. GANs suffer from slow convergence and potential mode collapse during training, while diffusion methods require multi-step inference that introduces considerable computational overhead. In this work, we introduce Flow2GAN, a two-stage framework that combines Flow Matching training for learning generative capabilities with GAN fine-tuning for efficient few-step inference. Specifically, given audio's unique properties, we first improve Flow Matching for audio modeling through: 1) reformulating the objective as endpoint estimation, avoiding velocity estimation difficulties when involving empty regions; 2) applying spectral energy-based loss scaling to emphasize perceptually salient quieter regions. Building on these Flow Matching adaptations, we demonstrate that a further stage of lightweight GAN fine-tuning enables us to obtain one-step generator that produces high-quality audio. In addition, we develop a multi-branch network architecture that processes Fourier coefficients at different time-frequency resolutions, which improves the modeling capabilities compared to prior single-resolution designs. Experimental results indicate that our Flow2GAN delivers high-fidelity audio generation from Mel-spectrograms or discrete audio tokens, achieving better quality-efficiency trade-offs than existing state-of-the-art GAN-based and Flow Matching-based methods. Online demo samples are available at https://flow2gan.github.io, and the source code is released at https://github.com/k2-fsa/Flow2GAN.
>
---
#### [new 006] Mobile-Efficient Speech Emotion Recognition Using DistilHuBERT: A Cross-Corpus Validation Study
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音情感识别任务，旨在解决模型计算需求高、难以部署于移动设备的问题。通过使用DistilHuBERT模型，实现高效且准确的情感识别系统。**

- **链接: [https://arxiv.org/pdf/2512.23435v1](https://arxiv.org/pdf/2512.23435v1)**

> **作者:** Saifelden M. Ismail
>
> **备注:** 5 pages, 2 tables, 1 figure. Submitted to IEEE conference
>
> **摘要:** Speech Emotion Recognition (SER) has significant potential for mobile applications, yet deployment remains constrained by the computational demands of state-of-the-art transformer architectures. This paper presents a mobile-efficient SER system based on DistilHuBERT, a distilled and 8-bit quantized transformer that achieves 92% parameter reduction compared to full-scale Wav2Vec 2.0 models while maintaining competitive accuracy. We conduct a rigorous 5-fold Leave-One-Session-Out (LOSO) cross-validation on the IEMOCAP dataset to ensure speaker independence, augmented with cross-corpus training on CREMA-D to enhance generalization. Cross-corpus training with CREMA-D yields a 1.2% improvement in Weighted Accuracy, a 1.4% gain in Macro F1-score, and a 32% reduction in cross-fold variance, with the Neutral class showing the most substantial benefit at 5.4% F1-score improvement. Our approach achieves an Unweighted Accuracy of 61.4% with a quantized model footprint of only 23 MB, representing approximately 91% of full-scale baseline performance. Cross-corpus evaluation on RAVDESS reveals that the theatrical nature of acted emotions causes predictions to cluster by arousal level rather than valence: happiness is systematically confused with anger due to acoustic saturation in high-energy expressions. Despite this theatricality effect reducing overall RAVDESS accuracy to 43.29%, the model maintains robust arousal detection with 97% recall for anger and 64% for sadness. These findings establish a Pareto-optimal tradeoff between model size and accuracy, enabling practical affect recognition on resource-constrained mobile devices.
>
---
#### [new 007] Single Channel Blind Dereverberation of Speech Signals
- **分类: eess.AS**

- **简介: 该论文属于语音去混响任务，旨在通过NMFD方法提升混响语音的频谱质量。工作包括提出新方法并结合时序模型进行比较分析。**

- **链接: [https://arxiv.org/pdf/2512.23322v1](https://arxiv.org/pdf/2512.23322v1)**

> **作者:** Dhruv Nigam
>
> **摘要:** Dereverberation of recorded speech signals is one of the most pertinent problems in speech processing. In the present work, the objective is to understand and implement dereverberation techniques that aim at enhancing the magnitude spectrogram of reverberant speech signals to remove the reverberant effects introduced. An approach to estimate a clean speech spectrogram from the reverberant speech spectrogram is proposed. This is achieved through non-negative matrix factor deconvolution(NMFD). Further, this approach is extended using the NMF representation for speech magnitude spectrograms. To exploit temporal dependencies, a convolutive NMF-based representation and a frame-stacked model are incorporated into the NMFD framework for speech. A novel approach for dereverberation by applying NMFD to the activation matrix of the reverberated magnitude spectrogram is also proposed. Finally, a comparative analysis of the performance of the listed techniques, using sentence recordings from the TIMIT database and recorded room impulse responses from the Reverb 2014 challenge, is presented based on two key objective measures - PESQ and Cepstral Distortion.\\ Although we were qualitatively able to verify the claims made in literature regarding these techniques, exact results could not be matched. The novel approach, as it is suggested, provides improvement in quantitative metrics, but is not consistent
>
---
#### [new 008] Spatial Interpolation of Room Impulse Responses based on Deeper Physics-Informed Neural Networks with Residual Connections
- **分类: eess.AS**

- **简介: 该论文属于声学逆问题，旨在提高房间脉冲响应的插值精度。通过设计更深的物理信息神经网络，结合残差连接和正弦激活函数，提升估计性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.22915v1](https://arxiv.org/pdf/2512.22915v1)**

> **作者:** Ken Kurata; Gen Sato; Izumi Tsunokuni; Yusuke Ikeda
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The room impulse response (RIR) characterizes sound propagation in a room from a loudspeaker to a microphone under the linear time-invariant assumption. Estimating RIRs from a limited number of measurement points is crucial for sound propagation analysis and visualization. Physics-informed neural networks (PINNs) have recently been introduced for accurate RIR estimation by embedding governing physical laws into deep learning models; however, the role of network depth has not been systematically investigated. In this study, we developed a deeper PINN architecture with residual connections and analyzed how network depth affects estimation performance. We further compared activation functions, including tanh and sinusoidal activations. Our results indicate that the residual PINN with sinusoidal activations achieves the highest accuracy for both interpolation and extrapolation of RIRs. Moreover, the proposed architecture enables stable training as the depth increases and yields notable improvements in estimating reflection components. These results provide practical guidelines for designing deep and stable PINNs for acoustic-inverse problems.
>
---
#### [new 009] Geometry-Aware Optimization for Respiratory Sound Classification: Enhancing Sensitivity with SAM-Optimized Audio Spectrogram Transformers
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文属于呼吸音分类任务，旨在解决数据集小、噪声高和类别不平衡的问题。通过优化音频频谱Transformer模型，提升分类敏感度。**

- **链接: [https://arxiv.org/pdf/2512.22564v1](https://arxiv.org/pdf/2512.22564v1)**

> **作者:** Atakan Işık; Selin Vulga Işık; Ahmet Feridun Işık; Mahşuk Taylan
>
> **备注:** 10 pages, 3 figures,2 tables
>
> **摘要:** Respiratory sound classification is hindered by the limited size, high noise levels, and severe class imbalance of benchmark datasets like ICBHI 2017. While Transformer-based models offer powerful feature extraction capabilities, they are prone to overfitting and often converge to sharp minima in the loss landscape when trained on such constrained medical data. To address this, we introduce a framework that enhances the Audio Spectrogram Transformer (AST) using Sharpness-Aware Minimization (SAM). Instead of merely minimizing the training loss, our approach optimizes the geometry of the loss surface, guiding the model toward flatter minima that generalize better to unseen patients. We also implement a weighted sampling strategy to handle class imbalance effectively. Our method achieves a state-of-the-art score of 68.10% on the ICBHI 2017 dataset, outperforming existing CNN and hybrid baselines. More importantly, it reaches a sensitivity of 68.31%, a crucial improvement for reliable clinical screening. Further analysis using t-SNE and attention maps confirms that the model learns robust, discriminative features rather than memorizing background noise.
>
---
#### [new 010] Chord Recognition with Deep Learning
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音乐信息检索任务，旨在解决自动和弦识别问题。通过实验分析现有方法，发现稀有和弦识别效果差，并提出使用音高增强提升准确率。**

- **链接: [https://arxiv.org/pdf/2512.22621v1](https://arxiv.org/pdf/2512.22621v1)**

> **作者:** Pierre Mackenzie
>
> **摘要:** Progress in automatic chord recognition has been slow since the advent of deep learning in the field. To understand why, I conduct experiments on existing methods and test hypotheses enabled by recent developments in generative models. Findings show that chord classifiers perform poorly on rare chords and that pitch augmentation boosts accuracy. Features extracted from generative models do not help and synthetic data presents an exciting avenue for future work. I conclude by improving the interpretability of model outputs with beat detection, reporting some of the best results in the field and providing qualitative analysis. Much work remains to solve automatic chord recognition, but I hope this thesis will chart a path for others to try.
>
---
#### [new 011] EEG-to-Voice Decoding of Spoken and Imagined speech Using Non-Invasive EEG
- **分类: eess.SP; cs.LG; cs.SD**

- **简介: 该论文属于脑机接口中的语音重建任务，旨在通过非侵入式EEG信号直接解码口语和想象语音，解决传统方法依赖时间对齐的问题。**

- **链接: [https://arxiv.org/pdf/2512.22146v1](https://arxiv.org/pdf/2512.22146v1)**

> **作者:** Hanbeot Park; Yunjeong Cho; Hunhee Kim
>
> **备注:** 20 pages, 7 figures, 4 tables
>
> **摘要:** Restoring speech communication from neural signals is a central goal of brain-computer interface research, yet EEG-based speech reconstruction remains challenging due to limited spatial resolution, susceptibility to noise, and the absence of temporally aligned acoustic targets in imagined speech. In this study, we propose an EEG-to-Voice paradigm that directly reconstructs speech from non-invasive EEG signals without dynamic time warping (DTW) or explicit temporal alignment. The proposed pipeline generates mel-spectrograms from EEG in an open-loop manner using a subject-specific generator, followed by pretrained vocoder and automatic speech recognition (ASR) modules to synthesize speech waveforms and decode text. Separate generators were trained for spoken speech and imagined speech, and transfer learning-based domain adaptation was applied by pretraining on spoken speech and adapting to imagined speech. A minimal language model-based correction module was optionally applied to correct limited ASR errors while preserving semantic structure. The framework was evaluated under 2 s and 4 s speech conditions using acoustic-level metrics (PCC, RMSE, MCD) and linguistic-level metrics (CER, WER). Stable acoustic reconstruction and comparable linguistic accuracy were observed for both spoken speech and imagined speech. While acoustic similarity decreased for longer utterances, text-level decoding performance was largely preserved, and word-position analysis revealed a mild increase in decoding errors toward later parts of sentences. The language model-based correction consistently reduced CER and WER without introducing semantic distortion. These results demonstrate the feasibility of direct, open-loop EEG-to-Voice reconstruction for spoken speech and imagined speech without explicit temporal alignment.
>
---
#### [new 012] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究多轮对话中语音模型的风格退化问题，属于自然语言处理任务。旨在解决模型无法持续保持指定说话风格的问题，并通过实验验证不同策略的缓解效果。**

- **链接: [https://arxiv.org/pdf/2512.23578v1](https://arxiv.org/pdf/2512.23578v1)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** Work in progress
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that when SLMs are asked to recall the style instruction in later turns, they can recall the style instruction, but they fail to express it throughout the conversation. We also show that explicitly asking the model to recall the style instruction can partially mitigate style amnesia. In addition, we examine various prompting strategies and find that SLMs struggle to follow the required style when the instruction is placed in system messages rather than user messages, which contradicts the intended function of system prompts.
>
---
#### [new 013] PROFASR-BENCH: A Benchmark for Context-Conditioned ASR in High-Stakes Professional Speech
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于高风险专业语音识别任务，旨在解决领域术语复杂、错误容忍度低的问题。提出ProfASR-Bench基准，评估上下文条件下的ASR性能。**

- **链接: [https://arxiv.org/pdf/2512.23686v1](https://arxiv.org/pdf/2512.23686v1)**

> **作者:** Deepak Babu Piskala
>
> **备注:** Benchmark dataset and evaluation suite. Data and code available at: https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench https://github.com/prdeepakbabu/ProfASR-Bench
>
> **摘要:** Automatic Speech Recognition (ASR) in professional settings faces challenges that existing benchmarks underplay: dense domain terminology, formal register variation, and near-zero tolerance for critical entity errors. We present ProfASR-Bench, a professional-talk evaluation suite for high-stakes applications across finance, medicine, legal, and technology. Each example pairs a natural-language prompt (domain cue and/or speaker profile) with an entity-rich target utterance, enabling controlled measurement of context-conditioned recognition. The corpus supports conventional ASR metrics alongside entity-aware scores and slice-wise reporting by accent and gender. Using representative families Whisper (encoder-decoder ASR) and Qwen-Omni (audio language models) under matched no-context, profile, domain+profile, oracle, and adversarial conditions, we find a consistent pattern: lightweight textual context produces little to no change in average word error rate (WER), even with oracle prompts, and adversarial prompts do not reliably degrade performance. We term this the context-utilization gap (CUG): current systems are nominally promptable yet underuse readily available side information. ProfASR-Bench provides a standardized context ladder, entity- and slice-aware reporting with confidence intervals, and a reproducible testbed for comparing fusion strategies across model families. Dataset: https://huggingface.co/datasets/prdeepakbabu/ProfASR-Bench Code: https://github.com/prdeepakbabu/ProfASR-Bench
>
---
## 更新

#### [replaced 001] Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation
- **分类: eess.AS; cs.CL; cs.HC; cs.LG; cs.SD**

- **简介: 该论文属于EEG语音解码任务，旨在解决数据噪声、数据量少及复杂任务性能差的问题。通过VAE数据增强和序列模型提升解码效果。**

- **链接: [https://arxiv.org/pdf/2501.04359v2](https://arxiv.org/pdf/2501.04359v2)**

> **作者:** Terrance Yu-Hao Chen; Yulin Chen; Pontus Soederhaell; Sadrishya Agrawal; Kateryna Shapovalenko
>
> **备注:** 19 pages, 15 figures, 2 tables
>
> **摘要:** Decoding speech from non-invasive brain signals, such as electroencephalography (EEG), has the potential to advance brain-computer interfaces (BCIs), with applications in silent communication and assistive technologies for individuals with speech impairments. However, EEG-based speech decoding faces major challenges, such as noisy data, limited datasets, and poor performance on complex tasks like speech perception. This study attempts to address these challenges by employing variational autoencoders (VAEs) for EEG data augmentation to improve data quality and applying a state-of-the-art (SOTA) sequence-to-sequence deep learning architecture, originally successful in electromyography (EMG) tasks, to EEG-based speech decoding. Additionally, we adapt this architecture for word classification tasks. Using the Brennan dataset, which contains EEG recordings of subjects listening to narrated speech, we preprocess the data and evaluate both classification and sequence-to-sequence models for EEG-to-words/sentences tasks. Our experiments show that VAEs have the potential to reconstruct artificial EEG data for augmentation. Meanwhile, our sequence-to-sequence model achieves more promising performance in generating sentences compared to our classification model, though both remain challenging tasks. These findings lay the groundwork for future research on EEG speech perception decoding, with possible extensions to speech production tasks such as silent or imagined speech.
>
---
#### [replaced 002] Dub-S2ST: Textless Speech-to-Speech Translation for Seamless Dubbing
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音翻译任务，旨在解决跨语言配音中语音模式不匹配的问题。提出一种基于离散扩散的翻译模型，实现时间对齐和语速适应，生成自然流畅的翻译结果。**

- **链接: [https://arxiv.org/pdf/2505.20899v2](https://arxiv.org/pdf/2505.20899v2)**

> **作者:** Jeongsoo Choi; Jaehun Kim; Joon Son Chung
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** This paper introduces a cross-lingual dubbing system that translates speech from one language to another while preserving key characteristics such as duration, speaker identity, and speaking speed. Despite the strong translation quality of existing speech translation approaches, they often overlook the transfer of speech patterns, leading to mismatches with source speech and limiting their suitability for dubbing applications. To address this, we propose a discrete diffusion-based speech-to-unit translation model with explicit duration control, enabling time-aligned translation. We then synthesize speech based on the translated units and source speaker's identity using a conditional flow matching model. Additionally, we introduce a unit-based speed adaptation mechanism that guides the translation model to produce speech at a rate consistent with the source, without relying on any text. Extensive experiments demonstrate that our framework generates natural and fluent translations that align with the original speech's duration and speaking pace, while achieving competitive translation performance. The code is available at https://github.com/kaistmm/Dub-S2ST.
>
---
#### [replaced 003] A Data-Centric Approach to Generalizable Speech Deepfake Detection
- **分类: cs.SD; eess.SP**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决模型泛化能力不足的问题。通过数据驱动方法，构建数据集并优化采样策略，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.18210v3](https://arxiv.org/pdf/2512.18210v3)**

> **作者:** Wen Huang; Yuchen Mao; Yanmin Qian
>
> **摘要:** Achieving robust generalization in speech deepfake detection (SDD) remains a primary challenge, as models often fail to detect unseen forgery methods. While research has focused on model-centric and algorithm-centric solutions, the impact of data composition is often underexplored. This paper proposes a data-centric approach, analyzing the SDD data landscape from two practical perspectives: constructing a single dataset and aggregating multiple datasets. To address the first perspective, we conduct a large-scale empirical study to characterize the data scaling laws for SDD, quantifying the impact of source and generator diversity. To address the second, we propose the Diversity-Optimized Sampling Strategy (DOSS), a principled framework for mixing heterogeneous data with two implementations: DOSS-Select (pruning) and DOSS-Weight (re-weighting). Our experiments show that DOSS-Select outperforms the naive aggregation baseline while using only 3% of the total available data. Furthermore, our final model, trained on a 12k-hour curated data pool using the optimal DOSS-Weight strategy, achieves state-of-the-art performance, outperforming large-scale baselines with greater data and model efficiency on both public benchmarks and a new challenge set of various commercial APIs.
>
---
#### [replaced 004] The CCF AATC 2025 Speech Restoration Challenge: A Retrospective
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音恢复任务，旨在解决多类型语音退化问题。通过挑战赛分析不同模型性能，揭示效率与生成模型的权衡及评估指标缺陷。**

- **链接: [https://arxiv.org/pdf/2509.12974v2](https://arxiv.org/pdf/2509.12974v2)**

> **作者:** Junan Zhang; Mengyao Zhu; Xin Xu; Hui Bu; Zhenhua Ling; Zhizheng Wu
>
> **备注:** Technical Report. Homepage: https://ccf-aatc.org.cn. Code & Data: https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc
>
> **摘要:** Real-world speech communication is rarely affected by a single type of degradation. Instead, it suffers from a complex interplay of acoustic interference, codec compression, and, increasingly, secondary artifacts introduced by upstream enhancement algorithms. To bridge the gap between academic research and these realistic scenarios, we introduced the CCF AATC 2025 Challenge. This challenge targets universal blind speech restoration, requiring a single model to handle three distinct distortion categories: acoustic degradation, codec distortion, and secondary processing artifacts. In this paper, we provide a comprehensive retrospective of the challenge, detailing the dataset construction, task design, and a systematic analysis of the 25 participating systems. We report three key findings that define the current state of the field: (1) Efficiency vs. Scale: Contrary to the trend of massive generative models, top-performing systems demonstrated that lightweight discriminative architectures (<10M parameters) can achieve state-of-the-art performance, balancing restoration quality with deployment constraints. (2) Generative Trade-off: While generative and hybrid models excel in theoretical perceptual metrics, breakdown analysis reveals they suffer from "reconstruction bias" in high-SNR codec tasks and struggle with hallucination in complex secondary artifact scenarios. (3) Metric Gap: Most critically, our rank correlation analysis exposes a strong negative correlation (\r{ho}=-0.8) between widely-used reference-free metrics (e.g., DNSMOS) and human MOS when evaluating hybrid systems. This indicates that current metrics may over-reward artificial spectral smoothness at the expense of perceptual naturalness. This paper aims to serve as a reference for future research in robust speech restoration and calls for the development of next-generation evaluation metrics sensitive to generative artifacts.
>
---
#### [replaced 005] Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决大模型在SER中出现的幻觉和不稳定性问题。提出C$^2$SER方法，结合上下文感知与思维链，提升识别准确性和稳定性。**

- **链接: [https://arxiv.org/pdf/2502.18186v3](https://arxiv.org/pdf/2502.18186v3)**

> **作者:** Zhixian Zhao; Xinfa Zhu; Xinsheng Wang; Shuiyuan Wang; Xuelong Geng; Wenjie Tian; Lei Xie
>
> **备注:** This work has been published in IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Large-scale audio language models (ALMs), such as Qwen2-Audio, are capable of comprehending diverse audio signal, performing audio analysis and generating textual responses. However, in speech emotion recognition (SER), ALMs often suffer from hallucinations, resulting in misclassifications or irrelevant outputs. To address these challenges, we propose C$^2$SER, a novel ALM designed to enhance the stability and accuracy of SER through Contextual perception and Chain of Thought (CoT). C$^2$SER integrates the Whisper encoder for semantic perception and Emotion2Vec-S for acoustic perception, where Emotion2Vec-S extends Emotion2Vec with semi-supervised learning to enhance emotional discrimination. Additionally, C$^2$SER employs a CoT approach, processing SER in a step-by-step manner while leveraging speech content and speaking styles to improve recognition. To further enhance stability, C$^2$SER introduces self-distillation from explicit CoT to implicit CoT, mitigating error accumulation and boosting recognition accuracy. Extensive experiments show that C$^2$SER outperforms existing popular ALMs, such as Qwen2-Audio and SECap, delivering more stable and precise emotion recognition. We release the training code, checkpoints, and test sets to facilitate further research.
>
---
#### [replaced 006] SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文提出SonicMaster，一个统一的音乐修复与母带处理模型，解决非专业环境下的音频质量问题。通过文本控制实现音频增强，提升音质并改善立体声效果。**

- **链接: [https://arxiv.org/pdf/2508.03448v3](https://arxiv.org/pdf/2508.03448v3)**

> **作者:** Jan Melechovsky; Ambuj Mehrish; Abhinaba Roy; Dorien Herremans
>
> **摘要:** Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration. To train this model, we construct the SonicMaster dataset, a large dataset of paired degraded and high-quality tracks by simulating common degradation types with nineteen degradation functions belonging to five enhancements groups: equalization, dynamics, reverb, amplitude, and stereo. Our approach leverages a flow-matching generative training paradigm to learn an audio transformation that maps degraded inputs to their cleaned, mastered versions guided by text prompts. Objective audio quality metrics demonstrate that SonicMaster significantly improves sound quality across all artifact categories. Furthermore, subjective listening tests confirm that listeners prefer SonicMaster's enhanced outputs over other baselines.
>
---
#### [replaced 007] Distinctive Feature Codec: An Adaptive Efficient Speech Representation for Depression Detection
- **分类: eess.AS**

- **简介: 该论文属于语音抑郁检测任务，旨在解决传统固定帧处理破坏时间动态的问题。提出DFC框架，通过自适应分割保留时间信息，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2505.18516v2](https://arxiv.org/pdf/2505.18516v2)**

> **作者:** Xiangyu Zhang; Fuming Fang; Peng Gao; Bin Qin; Beena Ahmed; Julien Epps
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable success across diverse fields, establishing a powerful paradigm for complex information processing. This has inspired the integration of speech into LLM frameworks, often by tokenizing continuous audio via neural speech codecs, enabling powerful speech language models. However, this dominant tokenization strategy relies on uniform frame-based processing at fixed time intervals. This fixed-rate approach, while effective for linguistic content, destroys the temporal dynamics. These dynamics are not noise but are established as primary biomarkers in clinical applications such as depression detection. To address this gap, we introduce the Distinctive Feature Codec (DFC), an adaptive framework engineered to preserve this vital timing information. Drawing from linguistic theory, DFC abandons fixed-interval processing and instead learns to dynamically segment the signal at perceptually significant acoustic transitions. This generates variable-length tokens that efficiently encode the temporal structure. As a key contribution, this work is the first to integrate traditional distinctive features into a modern deep learning codec for a temporally sensitive task such as depression detection. We also introduce the Group-wise Scalar Quantization (GSQ) approach to stably quantize these variable-length segments. Our distinctive feature-based approach offers a promising alternative to conventional frame-based processing and advances interpretable representation learning in the modern deep learning speech depression detection framework.
>
---
#### [replaced 008] Fun-Audio-Chat Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文提出Fun-Audio-Chat，解决语音与文本模型间的时间分辨率不匹配及知识遗忘问题，通过双分辨率语音表示和核心鸡尾酒训练提升音频理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2512.20156v2](https://arxiv.org/pdf/2512.20156v2)**

> **作者:** Tongyi Fun Team; Qian Chen; Luyao Cheng; Chong Deng; Xiangang Li; Jiaqing Liu; Chao-Hong Tan; Wen Wang; Junhao Xu; Jieping Ye; Qinglin Zhang; Qiquan Zhang; Jingren Zhou
>
> **备注:** Authors are listed in alphabetical order, 21 pages, https://github.com/FunAudioLLM/Fun-Audio-Chat
>
> **摘要:** Recent advancements in joint speech-text models show great potential for seamless voice interactions. However, existing models face critical challenges: temporal resolution mismatch between speech tokens (25Hz) and text tokens (~3Hz) dilutes semantic information, incurs high computational costs, and causes catastrophic forgetting of text LLM knowledge. We introduce Fun-Audio-Chat, a Large Audio Language Model addressing these limitations via two innovations from our previous work DrVoice. First, Dual-Resolution Speech Representations (DRSR): the Shared LLM processes audio at efficient 5Hz (via token grouping), while the Speech Refined Head generates high-quality tokens at 25Hz, balancing efficiency (~50% GPU reduction) and quality. Second, Core-Cocktail Training, a two-stage fine-tuning with intermediate merging that mitigates catastrophic forgetting. We then apply Multi-Task DPO Training to enhance robustness, audio understanding, instruction-following and voice empathy. This multi-stage post-training enables Fun-Audio-Chat to retain text LLM knowledge while gaining powerful audio understanding, reasoning, and generation. Unlike recent LALMs requiring large-scale audio-text pre-training, Fun-Audio-Chat leverages pre-trained models and extensive post-training. Fun-Audio-Chat 8B and MoE 30B-A3B achieve competitive performance on Speech-to-Text and Speech-to-Speech tasks, ranking top among similar-scale models on Spoken QA benchmarks. They also achieve competitive to superior performance on Audio Understanding, Speech Function Calling, Instruction-Following and Voice Empathy. We develop Fun-Audio-Chat-Duplex, a full-duplex variant with strong performance on Spoken QA and full-duplex interactions. We open-source Fun-Audio-Chat-8B with training and inference code, and provide an interactive demo, at https://github.com/FunAudioLLM/Fun-Audio-Chat .
>
---
#### [replaced 009] Unrolled Creative Adversarial Network For Generating Novel Musical Pieces
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决传统方法在创意和多样性上的不足。提出两种基于对抗网络的系统，提升音乐生成的创新性与变化性。**

- **链接: [https://arxiv.org/pdf/2501.00452v3](https://arxiv.org/pdf/2501.00452v3)**

> **作者:** Pratik Nag
>
> **摘要:** Music generation has emerged as a significant topic in artificial intelligence and machine learning. While recurrent neural networks (RNNs) have been widely employed for sequence generation, generative adversarial networks (GANs) remain relatively underexplored in this domain. This paper presents two systems based on adversarial networks for music generation. The first system learns a set of music pieces without differentiating between styles, while the second system focuses on learning and deviating from specific composers' styles to create innovative music. By extending the Creative Adversarial Networks (CAN) framework to the music domain, this work introduces unrolled CAN to address mode collapse, evaluating both GAN and CAN in terms of creativity and variation.
>
---
