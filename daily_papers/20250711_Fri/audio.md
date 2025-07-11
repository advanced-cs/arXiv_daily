# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Re-Bottleneck: Latent Re-Structuring for Neural Audio Autoencoders
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频压缩与表示学习任务，旨在解决预训练模型潜在空间结构不足的问题。通过引入Re-Bottleneck框架，优化潜在表示以适应不同应用需求。**

- **链接: [http://arxiv.org/pdf/2507.07867v1](http://arxiv.org/pdf/2507.07867v1)**

> **作者:** Dimitrios Bralios; Jonah Casebeer; Paris Smaragdis
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** Neural audio codecs and autoencoders have emerged as versatile models for audio compression, transmission, feature-extraction, and latent-space generation. However, a key limitation is that most are trained to maximize reconstruction fidelity, often neglecting the specific latent structure necessary for optimal performance in diverse downstream applications. We propose a simple, post-hoc framework to address this by modifying the bottleneck of a pre-trained autoencoder. Our method introduces a "Re-Bottleneck", an inner bottleneck trained exclusively through latent space losses to instill user-defined structure. We demonstrate the framework's effectiveness in three experiments. First, we enforce an ordering on latent channels without sacrificing reconstruction quality. Second, we align latents with semantic embeddings, analyzing the impact on downstream diffusion modeling. Third, we introduce equivariance, ensuring that a filtering operation on the input waveform directly corresponds to a specific transformation in the latent space. Ultimately, our Re-Bottleneck framework offers a flexible and efficient way to tailor representations of neural audio models, enabling them to seamlessly meet the varied demands of different applications with minimal additional training.
>
---
#### [new 002] SecureSpeech: Prompt-based Speaker and Content Protection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音隐私保护任务，解决身份和内容泄露问题。通过生成不可关联的说话人身份和替换敏感内容，实现语音匿名化。**

- **链接: [http://arxiv.org/pdf/2507.07799v1](http://arxiv.org/pdf/2507.07799v1)**

> **作者:** Belinda Soh Hui Hui; Xiaoxiao Miao; Xin Wang
>
> **备注:** Accepted by IEEE International Joint Conference on Biometrics (IJCB) 2025
>
> **摘要:** Given the increasing privacy concerns from identity theft and the re-identification of speakers through content in the speech field, this paper proposes a prompt-based speech generation pipeline that ensures dual anonymization of both speaker identity and spoken content. This is addressed through 1) generating a speaker identity unlinkable to the source speaker, controlled by descriptors, and 2) replacing sensitive content within the original text using a name entity recognition model and a large language model. The pipeline utilizes the anonymized speaker identity and text to generate high-fidelity, privacy-friendly speech via a text-to-speech synthesis model. Experimental results demonstrate an achievement of significant privacy protection while maintaining a decent level of content retention and audio quality. This paper also investigates the impact of varying speaker descriptions on the utility and privacy of generated speech to determine potential biases.
>
---
#### [new 003] VP-SelDoA: Visual-prompted Selective DoA Estimation of Target Sound via Semantic-Spatial Matching
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频-视觉声源定位任务，解决多源声源分离、语义与空间特征对齐及依赖配对数据的问题，提出VP-SelDoA方法实现目标声音的精准定位。**

- **链接: [http://arxiv.org/pdf/2507.07384v1](http://arxiv.org/pdf/2507.07384v1)**

> **作者:** Yu Chen; Xinyuan Qian; Hongxu Zhu; Jiadong Wang; Kainan Chen; Haizhou Li
>
> **备注:** Under Review
>
> **摘要:** Audio-visual sound source localization (AV-SSL) identifies the position of a sound source by exploiting the complementary strengths of auditory and visual signals. However, existing AV-SSL methods encounter three major challenges: 1) inability to selectively isolate the target sound source in multi-source scenarios, 2) misalignment between semantic visual features and spatial acoustic features, and 3) overreliance on paired audio-visual data. To overcome these limitations, we introduce Cross-Instance Audio-Visual Localization (CI-AVL), a novel task that leverages images from different instances of the same sound event category to localize target sound sources, thereby reducing dependence on paired data while enhancing generalization capabilities. Our proposed VP-SelDoA tackles this challenging task through a semantic-level modality fusion and employs a Frequency-Temporal ConMamba architecture to generate target-selective masks for sound isolation. We further develop a Semantic-Spatial Matching mechanism that aligns the heterogeneous semantic and spatial features via integrated cross- and self-attention mechanisms. To facilitate the CI-AVL research, we construct a large-scale dataset named VGG-SSL, comprising 13,981 spatial audio clips across 296 sound event categories. Extensive experiments show that our proposed method outperforms state-of-the-art audio-visual localization methods, achieving a mean absolute error (MAE) of 12.04 and an accuracy (ACC) of 78.23%.
>
---
#### [new 004] Input Conditioned Layer Dropping in Speech Foundation Models
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于语音模型优化任务，解决边缘设备计算资源动态变化的问题。提出输入驱动的层跳过方法，在不改变架构的前提下动态调整计算量，提升模型效率。**

- **链接: [http://arxiv.org/pdf/2507.07954v1](http://arxiv.org/pdf/2507.07954v1)**

> **作者:** Abdul Hannan; Daniele Falavigna; Alessio Brutti
>
> **备注:** Accepted at IEEE MLSP 2025
>
> **摘要:** Curating foundation speech models for edge and IoT settings, where computational resources vary over time, requires dynamic architectures featuring adaptable reduction strategies. One emerging approach is layer dropping ($\mathcal{LD}$) which skips fraction of the layers of a backbone network during inference to reduce the computational load. This allows transforming static models into dynamic ones. However, existing approaches exhibit limitations either in the mode of selecting layers or by significantly modifying the neural architecture. To this end, we propose input-driven $\mathcal{LD}$ that employs the network's input features and a lightweight layer selecting network to determine the optimum combination of processing layers. Extensive experimentation on 4 speech and audio public benchmarks, using two different pre-trained foundation models, demonstrates the effectiveness of our approach, thoroughly outperforming random dropping and producing on-par (or better) results to early exit.
>
---
#### [new 005] Audio-Visual Speech Separation via Bottleneck Iterative Network
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音频-视觉语音分离任务，旨在提升模型性能同时降低计算成本。提出Bottleneck Iterative Network（BIN）方法，在保持模型轻量的同时增强表示能力。**

- **链接: [http://arxiv.org/pdf/2507.07270v1](http://arxiv.org/pdf/2507.07270v1)**

> **作者:** Sidong Zhang; Shiv Shankar; Trang Nguyen; Andrea Fanelli; Madalina Fiterau
>
> **备注:** Accepted to the 42nd International Conference on Machine Learning Workshop on Machine Learning for Audio
>
> **摘要:** Integration of information from non-auditory cues can significantly improve the performance of speech-separation models. Often such models use deep modality-specific networks to obtain unimodal features, and risk being too costly or lightweight but lacking capacity. In this work, we present an iterative representation refinement approach called Bottleneck Iterative Network (BIN), a technique that repeatedly progresses through a lightweight fusion block, while bottlenecking fusion representations by fusion tokens. This helps improve the capacity of the model, while avoiding major increase in model size and balancing between the model performance and training cost. We test BIN on challenging noisy audio-visual speech separation tasks, and show that our approach consistently outperforms state-of-the-art benchmark models with respect to SI-SDRi on NTCD-TIMIT and LRS3+WHAM! datasets, while simultaneously achieving a reduction of more than 50% in training and GPU inference time across nearly all settings.
>
---
#### [new 006] DMF2Mel: A Dynamic Multiscale Fusion Network for EEG-Driven Mel Spectrogram Reconstruction
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于脑电信号驱动的梅尔频谱重建任务，旨在解决连续想象语音的精确重建问题。提出DMF2Mel网络，通过多尺度融合与动态机制提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07526v1](http://arxiv.org/pdf/2507.07526v1)**

> **作者:** Cunhang Fan; Sheng Zhang; Jingjing Zhang; Enrui Liu; Xinhui Li; Minggang Zhao; Zhao Lv
>
> **备注:** Accepted by ACM MM 2025
>
> **摘要:** Decoding speech from brain signals is a challenging research problem. Although existing technologies have made progress in reconstructing the mel spectrograms of auditory stimuli at the word or letter level, there remain core challenges in the precise reconstruction of minute-level continuous imagined speech: traditional models struggle to balance the efficiency of temporal dependency modeling and information retention in long-sequence decoding. To address this issue, this paper proposes the Dynamic Multiscale Fusion Network (DMF2Mel), which consists of four core components: the Dynamic Contrastive Feature Aggregation Module (DC-FAM), the Hierarchical Attention-Guided Multi-Scale Network (HAMS-Net), the SplineMap attention mechanism, and the bidirectional state space module (convMamba). Specifically, the DC-FAM separates speech-related "foreground features" from noisy "background features" through local convolution and global attention mechanisms, effectively suppressing interference and enhancing the representation of transient signals. HAMS-Net, based on the U-Net framework,achieves cross-scale fusion of high-level semantics and low-level details. The SplineMap attention mechanism integrates the Adaptive Gated Kolmogorov-Arnold Network (AGKAN) to combine global context modeling with spline-based local fitting. The convMamba captures long-range temporal dependencies with linear complexity and enhances nonlinear dynamic modeling capabilities. Results on the SparrKULee dataset show that DMF2Mel achieves a Pearson correlation coefficient of 0.074 in mel spectrogram reconstruction for known subjects (a 48% improvement over the baseline) and 0.048 for unknown subjects (a 35% improvement over the baseline).Code is available at: https://github.com/fchest/DMF2Mel.
>
---
#### [new 007] Assessing the Alignment of Audio Representations with Timbre Similarity Ratings
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频表征与人类感知对齐任务，旨在评估不同音频表示与音色相似性判断的匹配度，通过比较距离和排序来验证深度学习模型的有效性。**

- **链接: [http://arxiv.org/pdf/2507.07764v1](http://arxiv.org/pdf/2507.07764v1)**

> **作者:** Haokun Tian; Stefan Lattner; Charalampos Saitis
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Psychoacoustical so-called "timbre spaces" map perceptual similarity ratings of instrument sounds onto low-dimensional embeddings via multidimensional scaling, but suffer from scalability issues and are incapable of generalization. Recent results from audio (music and speech) quality assessment as well as image similarity have shown that deep learning is able to produce embeddings that align well with human perception while being largely free from these constraints. Although the existing human-rated timbre similarity data is not large enough to train deep neural networks (2,614 pairwise ratings on 334 audio samples), it can serve as test-only data for audio models. In this paper, we introduce metrics to assess the alignment of diverse audio representations with human judgments of timbre similarity by comparing both the absolute values and the rankings of embedding distances to human similarity ratings. Our evaluation involves three signal-processing-based representations, twelve representations extracted from pre-trained models, and three representations extracted from a novel sound matching model. Among them, the style embeddings inspired by image style transfer, extracted from the CLAP model and the sound matching model, remarkably outperform the others, showing their potential in modeling timbre similarity.
>
---
#### [new 008] mmFlux: Crowd Flow Analytics with Commodity mmWave MIMO Radar
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于人群流动分析任务，旨在通过mmWave雷达提取人群运动模式和语义。工作包括生成高保真流场、构建几何图，并利用curl和divergence分析获取关键语义。**

- **链接: [http://arxiv.org/pdf/2507.07331v1](http://arxiv.org/pdf/2507.07331v1)**

> **作者:** Anurag Pallaprolu; Winston Hurst; Yasamin Mostofi
>
> **摘要:** In this paper, we present a novel framework for extracting underlying crowd motion patterns and inferring crowd semantics using mmWave radar. First, our proposed signal processing pipeline combines optical flow estimation concepts from vision with novel statistical and morphological noise filtering to generate high-fidelity mmWave flow fields - compact 2D vector representations of crowd motion. We then introduce a novel approach that transforms these fields into directed geometric graphs, where edges capture dominant flow currents, vertices mark crowd splitting or merging, and flow distribution is quantified across edges. Finally, we show that by analyzing the local Jacobian and computing the corresponding curl and divergence, we can extract key crowd semantics for both structured and diffused crowds. We conduct 21 experiments on crowds of up to (and including) 20 people across 3 areas, using commodity mmWave radar. Our framework achieves high-fidelity graph reconstruction of the underlying flow structure, even for complex crowd patterns, demonstrating strong spatial alignment and precise quantitative characterization of flow split ratios. Finally, our curl and divergence analysis accurately infers key crowd semantics, e.g., abrupt turns, boundaries where flow directions shift, dispersions, and gatherings. Overall, these findings validate our framework, underscoring its potential for various crowd analytics applications.
>
---
#### [new 009] Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，解决在边缘设备上部署高精度模型的问题。通过研究量化方法，评估不同位宽对模型性能的影响，提出高效压缩方案。**

- **链接: [http://arxiv.org/pdf/2507.07877v1](http://arxiv.org/pdf/2507.07877v1)**

> **作者:** Chen Feng; Yicheng Lin; Shaojie Zhuo; Chenzheng Su; Ramchalam Kinattinkara Ramakrishnan; Zhaocong Yuan; Xiaopeng Zhang
>
> **摘要:** Recent advances in Automatic Speech Recognition (ASR) have demonstrated remarkable accuracy and robustness in diverse audio applications, such as live transcription and voice command processing. However, deploying these models on resource constrained edge devices (e.g., IoT device, wearables) still presents substantial challenges due to strict limits on memory, compute and power. Quantization, particularly Post-Training Quantization (PTQ), offers an effective way to reduce model size and inference cost without retraining. Despite its importance, the performance implications of various advanced quantization methods and bit-width configurations on ASR models remain unclear. In this work, we present a comprehensive benchmark of eight state-of-the-art (SOTA) PTQ methods applied to two leading edge-ASR model families, Whisper and Moonshine. We systematically evaluate model performances (i.e., accuracy, memory I/O and bit operations) across seven diverse datasets from the open ASR leaderboard, analyzing the impact of quantization and various configurations on both weights and activations. Built on an extension of the LLM compression toolkit, our framework integrates edge-ASR models, diverse advanced quantization algorithms, a unified calibration and evaluation data pipeline, and detailed analysis tools. Our results characterize the trade-offs between efficiency and accuracy, demonstrating that even 3-bit quantization can succeed on high capacity models when using advanced PTQ techniques. These findings provide valuable insights for optimizing ASR models on low-power, always-on edge devices.
>
---
#### [new 010] LISTEN: Lightweight Industrial Sound-representable Transformer for Edge Notification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于工业声学分析任务，旨在解决深度学习模型依赖大量数据和计算资源的问题。提出轻量级模型LISTEN，可在边缘设备实时运行，实现高效工业监测。**

- **链接: [http://arxiv.org/pdf/2507.07879v1](http://arxiv.org/pdf/2507.07879v1)**

> **作者:** Changheon Han; Yun Seok Kang; Yuseop Sim; Martin Byung-Guk Jun; Hyung Wook Park
>
> **摘要:** Deep learning-based machine listening is broadening the scope of industrial acoustic analysis for applications like anomaly detection and predictive maintenance, thereby improving manufacturing efficiency and reliability. Nevertheless, its reliance on large, task-specific annotated datasets for every new task limits widespread implementation on shop floors. While emerging sound foundation models aim to alleviate data dependency, they are too large and computationally expensive, requiring cloud infrastructure or high-end hardware that is impractical for on-site, real-time deployment. We address this gap with LISTEN (Lightweight Industrial Sound-representable Transformer for Edge Notification), a kilobyte-sized industrial sound foundation model. Using knowledge distillation, LISTEN runs in real-time on low-cost edge devices. On benchmark downstream tasks, it performs nearly identically to its much larger parent model, even when fine-tuned with minimal datasets and training resource. Beyond the model itself, we demonstrate its real-world utility by integrating LISTEN into a complete machine monitoring framework on an edge device with an Industrial Internet of Things (IIoT) sensor and system, validating its performance and generalization capabilities on a live manufacturing shop floor.
>
---
#### [new 011] End-to-end Acoustic-linguistic Emotion and Intent Recognition Enhanced by Semi-supervised Learning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感与意图识别任务，旨在解决标注数据不足的问题。通过半监督学习结合大量未标注数据，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.07806v1](http://arxiv.org/pdf/2507.07806v1)**

> **作者:** Zhao Ren; Rathi Adarshi Rammohan; Kevin Scheck; Sheng Li; Tanja Schultz
>
> **备注:** Accepted by EMBC 2025
>
> **摘要:** Emotion and intent recognition from speech is essential and has been widely investigated in human-computer interaction. The rapid development of social media platforms, chatbots, and other technologies has led to a large volume of speech data streaming from users. Nevertheless, annotating such data manually is expensive, making it challenging to train machine learning models for recognition purposes. To this end, we propose applying semi-supervised learning to incorporate a large scale of unlabelled data alongside a relatively smaller set of labelled data. We train end-to-end acoustic and linguistic models, each employing multi-task learning for emotion and intent recognition. Two semi-supervised learning approaches, including fix-match learning and full-match learning, are compared. The experimental results demonstrate that the semi-supervised learning approaches improve model performance in speech emotion and intent recognition from both acoustic and text data. The late fusion of the best models outperforms the acoustic and text baselines by joint recognition balance metrics of 12.3% and 10.4%, respectively.
>
---
#### [new 012] SonicMotion: Dynamic Spatial Audio Soundscapes with Latent Diffusion Models
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于空间音频生成任务，旨在解决动态3D声场生成问题。提出SonicMotion模型及新数据集，实现精准声源定位与高质量音频合成。**

- **链接: [http://arxiv.org/pdf/2507.07318v1](http://arxiv.org/pdf/2507.07318v1)**

> **作者:** Christian Templin; Yanda Zhu; Hao Wang
>
> **摘要:** Spatial audio is an integral part of immersive entertainment, such as VR/AR, and has seen increasing popularity in cinema and music as well. The most common format of spatial audio is described as first-order Ambisonics (FOA). We seek to extend recent advancements in FOA generative AI models to enable the generation of 3D scenes with dynamic sound sources. Our proposed end-to-end model, SonicMotion, comes in two variations which vary in their user input and level of precision in sound source localization. In addition to our model, we also present a new dataset of simulated spatial audio-caption pairs. Evaluation of our models demonstrate that they are capable of matching the semantic alignment and audio quality of state of the art models while capturing the desired spatial attributes.
>
---
#### [new 013] IML-Spikeformer: Input-aware Multi-Level Spiking Transformer for Speech Processing
- **分类: cs.MM; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决SNN在大规模语音任务中性能不足和计算开销大的问题。提出IML-Spikeformer架构，提升性能并降低能耗。**

- **链接: [http://arxiv.org/pdf/2507.07396v1](http://arxiv.org/pdf/2507.07396v1)**

> **作者:** Zeyang Song; Shimin Zhang; Yuhong Chou; Jibin Wu; Haizhou Li
>
> **备注:** Under review of TNNLS
>
> **摘要:** Spiking Neural Networks (SNNs), inspired by biological neural mechanisms, represent a promising neuromorphic computing paradigm that offers energy-efficient alternatives to traditional Artificial Neural Networks (ANNs). Despite proven effectiveness, SNN architectures have struggled to achieve competitive performance on large-scale speech processing task. Two key challenges hinder progress: (1) the high computational overhead during training caused by multi-timestep spike firing, and (2) the absence of large-scale SNN architectures tailored to speech processing tasks. To overcome the issues, we introduce Input-aware Multi-Level Spikeformer, i.e. IML-Spikeformer, a spiking Transformer architecture specifically designed for large-scale speech processing. Central to our design is the Input-aware Multi-Level Spike (IMLS) mechanism, which simulate multi-timestep spike firing within a single timestep using an adaptive, input-aware thresholding scheme. IML-Spikeformer further integrates a Reparameterized Spiking Self-Attention (RepSSA) module with a Hierarchical Decay Mask (HDM), forming the HD-RepSSA module. This module enhances the precision of attention maps and enables modeling of multi-scale temporal dependencies in speech signals. Experiments demonstrate that IML-Spikeformer achieves word error rates of 6.0\% on AiShell-1 and 3.4\% on Librispeech-960, comparable to conventional ANN transformers while reducing theoretical inference energy consumption by 4.64$\times$ and 4.32$\times$ respectively. IML-Spikeformer marks an advance of scalable SNN architectures for large-scale speech processing in both task performance and energy efficiency.
>
---
#### [new 014] Interpretable EEG-to-Image Generation with Semantic Prompts
- **分类: cs.CV; cs.LG; eess.SP**

- **简介: 该论文属于脑信号到图像的解码任务，旨在解决EEG空间细节不足导致的图像重建难题。通过语义提示对齐EEG与多层级语义描述，实现可解释的视觉解码。**

- **链接: [http://arxiv.org/pdf/2507.07157v1](http://arxiv.org/pdf/2507.07157v1)**

> **作者:** Arshak Rezvani; Ali Akbari; Kosar Sanjar Arani; Maryam Mirian; Emad Arasteh; Martin J. McKeown
>
> **备注:** Actionable Interpretability Workshop (non-archival) at the 42 International Conference on Machine Learning
>
> **摘要:** Decoding visual experience from brain signals offers exciting possibilities for neuroscience and interpretable AI. While EEG is accessible and temporally precise, its limitations in spatial detail hinder image reconstruction. Our model bypasses direct EEG-to-image generation by aligning EEG signals with multilevel semantic captions -- ranging from object-level to abstract themes -- generated by a large language model. A transformer-based EEG encoder maps brain activity to these captions through contrastive learning. During inference, caption embeddings retrieved via projection heads condition a pretrained latent diffusion model for image generation. This text-mediated framework yields state-of-the-art visual decoding on the EEGCVPR dataset, with interpretable alignment to known neurocognitive pathways. Dominant EEG-caption associations reflected the importance of different semantic levels extracted from perceived images. Saliency maps and t-SNE projections reveal semantic topography across the scalp. Our model demonstrates how structured semantic mediation enables cognitively aligned visual decoding from EEG.
>
---
#### [new 015] Generic Speech Enhancement with Self-Supervised Representation Space Loss
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决模型泛化能力不足的问题。通过引入自监督表示空间损失，提升模型在多个下游任务中的性能。**

- **链接: [http://arxiv.org/pdf/2507.07631v1](http://arxiv.org/pdf/2507.07631v1)**

> **作者:** Hiroshi Sato; Tsubasa Ochiai; Marc Delcroix; Takafumi Moriya; Takanori Ashihara; Ryo Masumura
>
> **备注:** 22 pages, 3 figures. Accepted for Frontiers in signal processing
>
> **摘要:** Single-channel speech enhancement is utilized in various tasks to mitigate the effect of interfering signals. Conventionally, to ensure the speech enhancement performs optimally, the speech enhancement has needed to be tuned for each task. Thus, generalizing speech enhancement models to unknown downstream tasks has been challenging. This study aims to construct a generic speech enhancement front-end that can improve the performance of back-ends to solve multiple downstream tasks. To this end, we propose a novel training criterion that minimizes the distance between the enhanced and the ground truth clean signal in the feature representation domain of self-supervised learning models. Since self-supervised learning feature representations effectively express high-level speech information useful for solving various downstream tasks, the proposal is expected to make speech enhancement models preserve such information. Experimental validation demonstrates that the proposal improves the performance of multiple speech tasks while maintaining the perceptual quality of the enhanced signal.
>
---
## 更新

#### [replaced 001] What do self-supervised speech models know about Dutch? Analyzing advantages of language-specific pre-training
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00981v2](http://arxiv.org/pdf/2506.00981v2)**

> **作者:** Marianne de Heer Kloots; Hosein Mohebbi; Charlotte Pouw; Gaofei Shen; Willem Zuidema; Martijn Bentum
>
> **备注:** Accepted to Interspeech 2025. For model, code, and materials, see https://github.com/mdhk/SSL-NL-eval
>
> **摘要:** How language-specific are speech representations learned by self-supervised models? Existing work has shown that a range of linguistic features can be successfully decoded from end-to-end models trained only on speech recordings. However, it's less clear to what extent pre-training on specific languages improves language-specific linguistic information. Here we test the encoding of Dutch phonetic and lexical information in internal representations of self-supervised Wav2Vec2 models. Pre-training exclusively on Dutch improves the representation of Dutch linguistic features as compared to pre-training on similar amounts of English or larger amounts of multilingual data. This language-specific advantage is well-detected by trained clustering or classification probes, and partially observable using zero-shot metrics. Furthermore, the language-specific benefit on linguistic feature encoding aligns with downstream performance on Automatic Speech Recognition.
>
---
#### [replaced 002] "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.00718v2](http://arxiv.org/pdf/2502.00718v2)**

> **作者:** Isha Gupta; David Khachaturov; Robert Mullins
>
> **摘要:** The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.
>
---
#### [replaced 003] Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.03251v2](http://arxiv.org/pdf/2507.03251v2)**

> **作者:** HyeYoung Lee; Muhammad Nadeem
>
> **摘要:** Speech Emotion Recognition (SER) traditionally relies on auditory data analysis for emotion classification. Several studies have adopted different methods for SER. However, existing SER methods often struggle to capture subtle emotional variations and generalize across diverse datasets. In this article, we use Mel-Frequency Cepstral Coefficients (MFCCs) as spectral features to bridge the gap between computational emotion processing and human auditory perception. To further improve robustness and feature diversity, we propose a novel 1D-CNN-based SER framework that integrates data augmentation techniques. MFCC features extracted from the augmented data are processed using a 1D Convolutional Neural Network (CNN) architecture enhanced with channel and spatial attention mechanisms. These attention modules allow the model to highlight key emotional patterns, enhancing its ability to capture subtle variations in speech signals. The proposed method delivers cutting-edge performance, achieving the accuracy of 97.49% for SAVEE, 99.23% for RAVDESS, 89.31% for CREMA-D, 99.82% for TESS, 99.53% for EMO-DB, and 96.39% for EMOVO. Experimental results show new benchmarks in SER, demonstrating the effectiveness of our approach in recognizing emotional expressions with high precision. Our evaluation demonstrates that the integration of advanced Deep Learning (DL) methods substantially enhances generalization across diverse datasets, underscoring their potential to advance SER for real-world deployment in assistive technologies and human-computer interaction.
>
---
#### [replaced 004] A Voice-based Triage for Type 2 Diabetes using a Conversational Virtual Assistant in the Home Environment
- **分类: cs.SD; eess.AS; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.19204v3](http://arxiv.org/pdf/2411.19204v3)**

> **作者:** Kelvin Summoogum; Debayan Das; Sathish Kumaran; Sumit Bhagra
>
> **备注:** 8 pages
>
> **摘要:** Incorporating cloud technology with Internet of Medical Things for ubiquitous healthcare has seen many successful applications in the last decade with the advent of machine learning and deep learning techniques. One of these applications, namely voice-based pathology, has yet to receive notable attention from academia and industry. Applying voice analysis to early detection of fatal diseases holds much promise to improve health outcomes and quality of life of patients. In this paper, we propose a novel application of acoustic machine learning based triaging into commoditised conversational virtual assistant systems to pre-screen for onset of diabetes. Specifically, we developed a triaging system which extracts acoustic features from the voices of n=24 older adults when they converse with a virtual assistant and predict the incidence of Diabetes Mellitus (Type 2) or not. Our triaging system achieved hit-rates of 70% and 60% for male and female older adult subjects, respectively. Our proposed triaging uses 7 non-identifiable voice-based features and can operate within resource-constrained embedded systems running voice-based virtual assistants. This application demonstrates the feasibility of applying voice-based pathology analysis to improve health outcomes of older adults within the home environment by early detection of life-changing chronic conditions like diabetes.
>
---
#### [replaced 005] Long-Form Speech Generation with Spoken Language Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.18603v2](http://arxiv.org/pdf/2412.18603v2)**

> **作者:** Se Jin Park; Julian Salazar; Aren Jansen; Keisuke Kinoshita; Yong Man Ro; RJ Skerry-Ryan
>
> **备注:** Accepted to ICML 2025 (oral)
>
> **摘要:** We consider the generative modeling of speech over multiple minutes, a requirement for long-form multimedia generation and audio-native voice assistants. However, textless spoken language models struggle to generate plausible speech past tens of seconds, due to high temporal resolution of speech tokens causing loss of coherence, architectural issues with long-sequence training or extrapolation, and memory costs at inference time. From these considerations we derive SpeechSSM, the first speech language model family to learn from and sample long-form spoken audio (e.g., 16 minutes of read or extemporaneous speech) in a single decoding session without text intermediates. SpeechSSMs leverage recent advances in linear-time sequence modeling to greatly surpass current Transformer spoken LMs in coherence and efficiency on multi-minute generations while still matching them at the utterance level. As we found current spoken language evaluations uninformative, especially in this new long-form setting, we also introduce: LibriSpeech-Long, a benchmark for long-form speech evaluation; new embedding-based and LLM-judged metrics; and quality measurements over length and time. Speech samples, the LibriSpeech-Long dataset, and any future code or model releases can be found at https://google.github.io/tacotron/publications/speechssm/.
>
---
#### [replaced 006] video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models
- **分类: cs.CV; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.15220v2](http://arxiv.org/pdf/2506.15220v2)**

> **作者:** Changli Tang; Yixuan Li; Yudong Yang; Jimin Zhuang; Guangzhi Sun; Wei Li; Zejun Ma; Chao Zhang
>
> **摘要:** Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimisation (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimised using DPO. To further improve training, we propose a novel multi-round DPO (MrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initialising the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilise the process. Experimental results show that MrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing the captioning error rates by 28\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining highly competitive performance to the state-of-the-art on widely used video question-answering benchmarks among models of similar size. Codes are available at \href{https://github.com/bytedance/video-SALMONN-2}{https://github.com/bytedance/video-SALMONN-2}.
>
---
#### [replaced 007] C3T: Cross-modal Transfer Through Time for Sensor-based Human Activity Recognition
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.16803v4](http://arxiv.org/pdf/2407.16803v4)**

> **作者:** Abhi Kamboj; Anh Duy Nguyen; Minh N. Do
>
> **摘要:** In order to unlock the potential of diverse sensors, we investigate a method to transfer knowledge between time-series modalities using a multimodal \textit{temporal} representation space for Human Activity Recognition (HAR). Specifically, we explore the setting where the modality used in testing has no labeled data during training, which we refer to as Unsupervised Modality Adaptation (UMA). We categorize existing UMA approaches as Student-Teacher or Contrastive Alignment methods. These methods typically compress continuous-time data samples into single latent vectors during alignment, inhibiting their ability to transfer temporal information through real-world temporal distortions. To address this, we introduce Cross-modal Transfer Through Time (C3T), which preserves temporal information during alignment to handle dynamic sensor data better. C3T achieves this by aligning a set of temporal latent vectors across sensing modalities. Our extensive experiments on various camera+IMU datasets demonstrate that C3T outperforms existing methods in UMA by at least 8% in accuracy and shows superior robustness to temporal distortions such as time-shift, misalignment, and dilation. Our findings suggest that C3T has significant potential for developing generalizable models for time-series sensor data, opening new avenues for various multimodal applications.
>
---
#### [replaced 008] Tiny-Align: Bridging Automatic Speech Recognition and Large Language Model on the Edge
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.13766v3](http://arxiv.org/pdf/2411.13766v3)**

> **作者:** Ruiyang Qin; Dancheng Liu; Gelei Xu; Zheyu Yan; Chenhui Xu; Yuting Hu; X. Sharon Hu; Jinjun Xiong; Yiyu Shi
>
> **备注:** Accepted by ICCAD'25
>
> **摘要:** The combination of Large Language Models (LLM) and Automatic Speech Recognition (ASR), when deployed on edge devices (called edge ASR-LLM), can serve as a powerful personalized assistant to enable audio-based interaction for users. Compared to text-based interaction, edge ASR-LLM allows accessible and natural audio interactions. Unfortunately, existing ASR-LLM models are mainly trained in high-performance computing environments and produce substantial model weights, making them difficult to deploy on edge devices. More importantly, to better serve users' personalized needs, the ASR-LLM must be able to learn from each distinct user, given that audio input often contains highly personalized characteristics that necessitate personalized on-device training. Since individually fine-tuning the ASR or LLM often leads to suboptimal results due to modality-specific limitations, end-to-end training ensures seamless integration of audio features and language understanding (cross-modal alignment), ultimately enabling a more personalized and efficient adaptation on edge devices. However, due to the complex training requirements and substantial computational demands of existing approaches, cross-modal alignment between ASR audio and LLM can be challenging on edge devices. In this work, we propose a resource-efficient cross-modal alignment framework that bridges ASR and LLMs on edge devices to handle personalized audio input. Our framework enables efficient ASR-LLM alignment on resource-constrained devices like NVIDIA Jetson Orin (8GB RAM), achieving 50x training time speedup while improving the alignment quality by more than 50\%. To the best of our knowledge, this is the first work to study efficient ASR-LLM alignment on resource-constrained edge devices.
>
---
#### [replaced 009] Discrete Optimal Transport and Voice Conversion
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.04382v2](http://arxiv.org/pdf/2505.04382v2)**

> **作者:** Anton Selitskiy; Maitreya Kocharekar
>
> **备注:** 4 pages, 6 figures, 1 table
>
> **摘要:** In this work, we address the voice conversion (VC) task using a vector-based interface. To align audio embeddings between speakers, we employ discrete optimal transport mapping. Our evaluation results demonstrate the high quality and effectiveness of this method. Additionally, we show that applying discrete optimal transport as a post-processing step in audio generation can lead to the incorrect classification of synthetic audio as real.
>
---
#### [replaced 010] Inter-linguistic Phonetic Composition (IPC): A Theoretical and Computational Approach to Enhance Second Language Pronunciation
- **分类: cs.CL; cs.SD; eess.AS; H.5.5**

- **链接: [http://arxiv.org/pdf/2411.10927v3](http://arxiv.org/pdf/2411.10927v3)**

> **作者:** Jisang Park; Minu Kim; DaYoung Hong; Jongha Lee
>
> **摘要:** Learners of a second language (L2) often unconsciously substitute unfamiliar L2 phonemes with similar phonemes from their native language (L1), even though native speakers of the L2 perceive these sounds as distinct and non-interchangeable. This phonemic substitution leads to deviations from the standard phonological patterns of the L2, creating challenges for learners in acquiring accurate L2 pronunciation. To address this, we propose Inter-linguistic Phonetic Composition (IPC), a novel computational method designed to minimize incorrect phonological transfer by reconstructing L2 phonemes as composite sounds derived from multiple L1 phonemes. Tests with two automatic speech recognition models demonstrated that when L2 speakers produced IPC-generated composite sounds, the recognition rate of target L2 phonemes improved by 20% compared to when their pronunciation was influenced by original phonological transfer patterns. The improvement was observed within a relatively shorter time frame, demonstrating rapid acquisition of the composite sound.
>
---
#### [replaced 011] Benchmarking Time-localized Explanations for Audio Classification Models
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04391v2](http://arxiv.org/pdf/2506.04391v2)**

> **作者:** Cecilia Bolaños; Leonardo Pepino; Martin Meza; Luciana Ferrer
>
> **摘要:** Most modern approaches for audio processing are opaque, in the sense that they do not provide an explanation for their decisions. For this reason, various methods have been proposed to explain the outputs generated by these models. Good explanations can result in interesting insights about the data or the model, as well as increase trust in the system. Unfortunately, evaluating the quality of explanations is far from trivial since, for most tasks, there is no clear ground truth explanation to use as reference. In this work, we propose a benchmark for time-localized explanations for audio classification models that uses time annotations of target events as a proxy for ground truth explanations. We use this benchmark to systematically optimize and compare various approaches for model-agnostic post-hoc explanation, obtaining, in some cases, close to perfect explanations. Finally, we illustrate the utility of the explanations for uncovering spurious correlations.
>
---
