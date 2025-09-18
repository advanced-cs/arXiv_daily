# 音频 cs.SD;  eess.SP

- **最新发布 16 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Noise Supervised Contrastive Learning and Feature-Perturbed for Anomalous Sound Detection
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于异常声音检测任务，旨在仅使用正常音频数据检测未知异常声。提出OS-SCL方法，通过特征扰动和噪声监督对比学习，结合TFgram时频特征，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2509.13853v1](http://arxiv.org/pdf/2509.13853v1)**

> **作者:** Shun Huang; Zhihua Fang; Liang He
>
> **备注:** Accept ICASSP 2025
>
> **摘要:** Unsupervised anomalous sound detection aims to detect unknown anomalous sounds by training a model using only normal audio data. Despite advancements in self-supervised methods, the issue of frequent false alarms when handling samples of the same type from different machines remains unresolved. This paper introduces a novel training technique called one-stage supervised contrastive learning (OS-SCL), which significantly addresses this problem by perturbing features in the embedding space and employing a one-stage noisy supervised contrastive learning approach. On the DCASE 2020 Challenge Task 2, it achieved 94.64\% AUC, 88.42\% pAUC, and 89.24\% mAUC using only Log-Mel features. Additionally, a time-frequency feature named TFgram is proposed, which is extracted from raw audio. This feature effectively captures critical information for anomalous sound detection, ultimately achieving 95.71\% AUC, 90.23\% pAUC, and 91.23\% mAUC. The source code is available at: \underline{www.github.com/huangswt/OS-SCL}.
>
---
#### [new 002] Comprehensive Evaluation of CNN-Based Audio Tagging Models on Resource-Constrained Devices
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究在资源受限设备（如树莓派）上部署音频标签模型的任务，评估多种CNN架构的性能与稳定性，解决计算效率与散热问题，将模型转为ONNX格式以提升跨平台部署能力。**

- **链接: [http://arxiv.org/pdf/2509.14049v1](http://arxiv.org/pdf/2509.14049v1)**

> **作者:** Jordi Grau-Haro; Ruben Ribes-Serrano; Javier Naranjo-Alcazar; Marta Garcia-Ballesteros; Pedro Zuccarello
>
> **备注:** Accepted at Computing Conference 2026, London, UK
>
> **摘要:** Convolutional Neural Networks (CNNs) have demonstrated exceptional performance in audio tagging tasks. However, deploying these models on resource-constrained devices like the Raspberry Pi poses challenges related to computational efficiency and thermal management. In this paper, a comprehensive evaluation of multiple convolutional neural network (CNN) architectures for audio tagging on the Raspberry Pi is conducted, encompassing all 1D and 2D models from the Pretrained Audio Neural Networks (PANNs) framework, a ConvNeXt-based model adapted for audio classification, as well as MobileNetV3 architectures. In addition, two PANNs-derived networks, CNN9 and CNN13, recently proposed, are also evaluated. To enhance deployment efficiency and portability across diverse hardware platforms, all models are converted to the Open Neural Network Exchange (ONNX) format. Unlike previous works that focus on a single model, our analysis encompasses a broader range of architectures and involves continuous 24-hour inference sessions to assess performance stability. Our experiments reveal that, with appropriate model selection and optimization, it is possible to maintain consistent inference latency and manage thermal behavior effectively over extended periods. These findings provide valuable insights for deploying audio tagging models in real-world edge computing scenarios.
>
---
#### [new 003] AnyAccomp: Generalizable Accompaniment Generation via Quantized Melodic Bottleneck
- **分类: cs.SD; eess.SP**

- **简介: 该论文提出AnyAccomp框架，解决伴奏生成中因依赖分离语音而泛化能力差的问题。通过量化旋律瓶颈提取鲁棒旋律表示，并用流匹配模型生成伴奏，实现对真实人声和纯乐器曲目的泛化生成。属于音乐生成任务。**

- **链接: [http://arxiv.org/pdf/2509.14052v1](http://arxiv.org/pdf/2509.14052v1)**

> **作者:** Junan Zhang; Yunjia Zhang; Xueyao Zhang; Zhizheng Wu
>
> **备注:** Demo audio and code: https://anyaccomp.github.io
>
> **摘要:** Singing Accompaniment Generation (SAG) is the process of generating instrumental music for a given clean vocal input. However, existing SAG techniques use source-separated vocals as input and overfit to separation artifacts. This creates a critical train-test mismatch, leading to failure on clean, real-world vocal inputs. We introduce AnyAccomp, a framework that resolves this by decoupling accompaniment generation from source-dependent artifacts. AnyAccomp first employs a quantized melodic bottleneck, using a chromagram and a VQ-VAE to extract a discrete and timbre-invariant representation of the core melody. A subsequent flow-matching model then generates the accompaniment conditioned on these robust codes. Experiments show AnyAccomp achieves competitive performance on separated-vocal benchmarks while significantly outperforming baselines on generalization test sets of clean studio vocals and, notably, solo instrumental tracks. This demonstrates a qualitative leap in generalization, enabling robust accompaniment for instruments - a task where existing models completely fail - and paving the way for more versatile music co-creation tools. Demo audio and code: https://anyaccomp.github.io
>
---
#### [new 004] Field of View Enhanced Signal Dependent Binauralization with Mixture of Experts Framework for Continuous Source Motion
- **分类: cs.SD; stat.ML**

- **简介: 该论文提出一种基于专家混合框架的信号依赖双耳化方法，用于增强视野范围，解决连续声源运动中的空间音频渲染问题。通过隐式定位实时跟踪声源，实现动态空间音频渲染，适用于AR/VR等场景。**

- **链接: [http://arxiv.org/pdf/2509.13548v1](http://arxiv.org/pdf/2509.13548v1)**

> **作者:** Manan Mittal; Thomas Deppisch; Joseph Forrer; Chris Le Sueur; Zamir Ben-Hur; David Lou Along; Daniel D. E. Wong
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** We propose a novel mixture of experts framework for field-of-view enhancement in binaural signal matching. Our approach enables dynamic spatial audio rendering that adapts to continuous talker motion, allowing users to emphasize or suppress sounds from selected directions while preserving natural binaural cues. Unlike traditional methods that rely on explicit direction-of-arrival estimation or operate in the Ambisonics domain, our signal-dependent framework combines multiple binaural filters in an online manner using implicit localization. This allows for real-time tracking and enhancement of moving sound sources, supporting applications such as speech focus, noise reduction, and world-locked audio in augmented and virtual reality. The method is agnostic to array geometry offering a flexible solution for spatial audio capture and personalized playback in next-generation consumer audio devices.
>
---
#### [new 005] Neural Speech Separation with Parallel Amplitude and Phase Spectrum Estimation
- **分类: cs.SD**

- **简介: 该论文提出APSS模型，用于语音分离任务，解决传统方法相位估计不足的问题。通过并行估计幅度和相位谱，结合时频Transformer处理，提升分离效果，实验表明其优于现有方法，具有良好的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.13825v1](http://arxiv.org/pdf/2509.13825v1)**

> **作者:** Fei Liu; Yang Ai; Zhen-Hua Ling
>
> **备注:** Accepted by APSIPA2025
>
> **摘要:** This paper proposes APSS, a novel neural speech separation model with parallel amplitude and phase spectrum estimation. Unlike most existing speech separation methods, the APSS distinguishes itself by explicitly estimating the phase spectrum for more complete and accurate separation. Specifically, APSS first extracts the amplitude and phase spectra from the mixed speech signal. Subsequently, the extracted amplitude and phase spectra are fused by a feature combiner into joint representations, which are then further processed by a deep processor with time-frequency Transformers to capture temporal and spectral dependencies. Finally, leveraging parallel amplitude and phase separators, the APSS estimates the respective spectra for each speaker from the resulting features, which are then combined via inverse short-time Fourier transform (iSTFT) to reconstruct the separated speech signals. Experimental results indicate that APSS surpasses both time-domain separation methods and implicit-phase-estimation-based time-frequency approaches. Also, APSS achieves stable and competitive results on multiple datasets, highlighting its strong generalization capability and practical applicability.
>
---
#### [new 006] A Domain Knowledge Informed Approach for Anomaly Detection of Electric Vehicle Interior Sounds
- **分类: cs.SD; cs.AI; cs.CV; cs.LG; eess.AS; I.2.1; I.2.6; I.2.10; I.5.1; I.5.2; J.2; J.7**

- **简介: 论文提出一种基于领域知识的模型选择方法，用于检测电动汽车舱内异常声音。任务为无监督异常检测，解决缺乏故障标签导致的模型选择难题。通过构造代理异常数据验证模型，显著优于传统策略。**

- **链接: [http://arxiv.org/pdf/2509.13390v1](http://arxiv.org/pdf/2509.13390v1)**

> **作者:** Deepti Kunte; Bram Cornelis; Claudio Colangeli; Karl Janssens; Brecht Van Baelen; Konstantinos Gryllias
>
> **备注:** Submitted to: Mechanical Systems and Signal Processing
>
> **摘要:** The detection of anomalies in automotive cabin sounds is critical for ensuring vehicle quality and maintaining passenger comfort. In many real-world settings, this task is more appropriately framed as an unsupervised learning problem rather than the supervised case due to the scarcity or complete absence of labeled faulty data. In such an unsupervised setting, the model is trained exclusively on healthy samples and detects anomalies as deviations from normal behavior. However, in the absence of labeled faulty samples for validation and the limited reliability of commonly used metrics, such as validation reconstruction error, effective model selection remains a significant challenge. To overcome these limitations, a domain-knowledge-informed approach for model selection is proposed, in which proxy-anomalies engineered through structured perturbations of healthy spectrograms are used in the validation set to support model selection. The proposed methodology is evaluated on a high-fidelity electric vehicle dataset comprising healthy and faulty cabin sounds across five representative fault types viz., Imbalance, Modulation, Whine, Wind, and Pulse Width Modulation. This dataset, generated using advanced sound synthesis techniques, and validated via expert jury assessments, has been made publicly available to facilitate further research. Experimental evaluations on the five fault cases demonstrate the selection of optimal models using proxy-anomalies, significantly outperform conventional model selection strategies.
>
---
#### [new 007] RFM-Editing: Rectified Flow Matching for Text-guided Audio Editing
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出RFM-Editing框架，解决文本引导的音频编辑任务，旨在精准修改音频内容而不破坏其余部分。通过构建多事件音频数据集，实现无需辅助信息的高效、高质量编辑。**

- **链接: [http://arxiv.org/pdf/2509.14003v1](http://arxiv.org/pdf/2509.14003v1)**

> **作者:** Liting Gao; Yi Yuan; Yaru Chen; Yuelan Cheng; Zhenbo Li; Juan Wen; Shubin Zhang; Wenwu Wang
>
> **摘要:** Diffusion models have shown remarkable progress in text-to-audio generation. However, text-guided audio editing remains in its early stages. This task focuses on modifying the target content within an audio signal while preserving the rest, thus demanding precise localization and faithful editing according to the text prompt. Existing training-based and zero-shot methods that rely on full-caption or costly optimization often struggle with complex editing or lack practicality. In this work, we propose a novel end-to-end efficient rectified flow matching-based diffusion framework for audio editing, and construct a dataset featuring overlapping multi-event audio to support training and benchmarking in complex scenarios. Experiments show that our model achieves faithful semantic alignment without requiring auxiliary captions or masks, while maintaining competitive editing quality across metrics.
>
---
#### [new 008] A Lightweight Fourier-based Network for Binaural Speech Enhancement with Spatial Cue Preservation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出轻量级GAF-Net网络，用于双耳语音增强，解决性能与计算效率的矛盾。通过融合傅里叶与gammatone特征、全局自适应傅里叶调制及动态门控机制，在保持空间线索的同时提升性能。**

- **链接: [http://arxiv.org/pdf/2509.14076v1](http://arxiv.org/pdf/2509.14076v1)**

> **作者:** Xikun Lu; Yujian Ma; Xianquan Jiang; Xuelong Wang; Jinqiu Sang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Binaural speech enhancement faces a severe trade-off challenge, where state-of-the-art performance is achieved by computationally intensive architectures, while lightweight solutions often come at the cost of significant performance degradation. To bridge this gap, we propose the Global Adaptive Fourier Network (GAF-Net), a lightweight deep complex network that aims to establish a balance between performance and computational efficiency. The GAF-Net architecture consists of three components. First, a dual-feature encoder combining short-time Fourier transform and gammatone features enhances the robustness of acoustic representation. Second, a channel-independent globally adaptive Fourier modulator efficiently captures long-term temporal dependencies while preserving the spatial cues. Finally, a dynamic gating mechanism is implemented to reduce processing artifacts. Experimental results show that GAF-Net achieves competitive performance, particularly in terms of binaural cues (ILD and IPD error) and objective intelligibility (MBSTOI), with fewer parameters and computational cost. These results confirm that GAF-Net provides a feasible way to achieve high-fidelity binaural processing on resource-constrained devices.
>
---
#### [new 009] Summary on The Multilingual Conversational Speech Language Model Challenge: Datasets, Tasks, Baselines, and Methods
- **分类: eess.AS; cs.SD**

- **简介: 该论文总结了Interspeech2025多语言对话语音大模型挑战赛，介绍了任务设置、1604小时真实多语言对话数据集及基线系统，分析参赛团队成果，旨在推动多语言对话语音大模型的发展。**

- **链接: [http://arxiv.org/pdf/2509.13785v1](http://arxiv.org/pdf/2509.13785v1)**

> **作者:** Bingshen Mu; Pengcheng Guo; Zhaokai Sun; Shuai Wang; Hexin Liu; Mingchen Shao; Lei Xie; Eng Siong Chng; Longshuai Xiao; Qiangze Feng; Daliang Wang
>
> **摘要:** This paper summarizes the Interspeech2025 Multilingual Conversational Speech Language Model (MLC-SLM) challenge, which aims to advance the exploration of building effective multilingual conversational speech LLMs (SLLMs). We provide a detailed description of the task settings for the MLC-SLM challenge, the released real-world multilingual conversational speech dataset totaling approximately 1,604 hours, and the baseline systems for participants. The MLC-SLM challenge attracts 78 teams from 13 countries to participate, with 489 valid leaderboard results and 14 technical reports for the two tasks. We distill valuable insights on building multilingual conversational SLLMs based on submissions from participants, aiming to contribute to the advancement of the community.
>
---
#### [new 010] CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出CS-FLEURS数据集，用于开发和评估跨多种语言的代码切换语音识别与翻译系统。数据集包含4个测试集和训练集，覆盖113种代码切换语言对，旨在推动低资源语言的代码切换语音研究。**

- **链接: [http://arxiv.org/pdf/2509.14161v1](http://arxiv.org/pdf/2509.14161v1)**

> **作者:** Brian Yan; Injy Hamed; Shuichiro Shimizu; Vasista Lodagala; William Chen; Olga Iakovenko; Bashar Talafha; Amir Hussein; Alexander Polok; Kalvin Chang; Dominik Klement; Sara Althubaiti; Puyuan Peng; Matthew Wiesner; Thamar Solorio; Ahmed Ali; Sanjeev Khudanpur; Shinji Watanabe; Chih-Chen Chen; Zhen Wu; Karim Benharrak; Anuj Diwan; Samuele Cornell; Eunjung Yeo; Kwanghee Choi; Carlos Carvalho; Karen Rosero
>
> **摘要:** We present CS-FLEURS, a new dataset for developing and evaluating code-switched speech recognition and translation systems beyond high-resourced languages. CS-FLEURS consists of 4 test sets which cover in total 113 unique code-switched language pairs across 52 languages: 1) a 14 X-English language pair set with real voices reading synthetically generated code-switched sentences, 2) a 16 X-English language pair set with generative text-to-speech 3) a 60 {Arabic, Mandarin, Hindi, Spanish}-X language pair set with the generative text-to-speech, and 4) a 45 X-English lower-resourced language pair test set with concatenative text-to-speech. Besides the four test sets, CS-FLEURS also provides a training set with 128 hours of generative text-to-speech data across 16 X-English language pairs. Our hope is that CS-FLEURS helps to broaden the scope of future code-switched speech research. Dataset link: https://huggingface.co/datasets/byan/cs-fleurs.
>
---
#### [new 011] Network representations reveal structured uncertainty in music
- **分类: physics.soc-ph; cs.SD**

- **简介: 论文研究音乐网络表示中的结构化不确定性，通过比较八种网络模型，分析特征选择对结构和认知匹配的影响。发现简单模型更符合人类感知，复杂模型引入认知低效。任务是探索音乐结构与认知的关系，方法包括拓扑分析与熵分析。**

- **链接: [http://arxiv.org/pdf/2509.14053v1](http://arxiv.org/pdf/2509.14053v1)**

> **作者:** Lluc Bono Rosselló; Robert Jankowski; Hugues Bersini; Marián Boguñá; M. Ángeles Serrano
>
> **摘要:** Music, as a structured yet perceptually rich experience, can be modeled as a network to uncover how humans encode and process auditory information. While network-based representations of music are increasingly common, the impact of feature selection on structural properties and cognitive alignment remains underexplored. In this study, we evaluated eight network models, each constructed from symbolic representations of piano compositions using distinct combinations of pitch, octave, duration, and interval, designed to be representative of existing approaches in the literature. By comparing these models through topological metrics, entropy analysis, and divergence with respect to inferred cognitive representations, we assessed both their structural and perceptual efficiency. Our findings reveal that simpler, feature-specific models better match human perception, whereas complex, multidimensional representations introduce cognitive inefficiencies. These results support the view that humans rely on modular, parallel cognitive networks--an architecture consistent with theories of predictive processing and free energy minimization. Moreover, we find that musical networks are structurally organized to guide attention toward transitions that are both uncertain and inferable. The resulting structure concentrates uncertainty in a few frequently visited nodes, creating local entropy gradients that alternate between stable and unpredictable regions, thereby enabling the expressive dynamics of tension and release that define the musical experience. These findings show that network structures make the organization of uncertainty in music observable, offering new insight into how patterned flows of expectation shape perception, and open new directions for studying how musical structures evolve across genres, cultures, and historical periods through the lens of network science.
>
---
#### [new 012] Invisible Ears at Your Fingertips: Acoustic Eavesdropping via Mouse Sensors
- **分类: cs.CR; cs.SD**

- **简介: 该论文提出Mic-E-Mouse攻击，利用光学鼠标传感器窃听语音。属于侧信道攻击任务，解决通过鼠标数据提取音频信号的问题，设计滤波管道提升语音识别准确率。**

- **链接: [http://arxiv.org/pdf/2509.13581v1](http://arxiv.org/pdf/2509.13581v1)**

> **作者:** Mohamad Fakih; Rahul Dharmaji; Youssef Mahmoud; Halima Bouzidi; Mohammad Abdullah Al Faruque
>
> **备注:** Appearing in the Annual Computer Security Applications Conference (ACSAC 2025)
>
> **摘要:** Modern optical mouse sensors, with their advanced precision and high responsiveness, possess an often overlooked vulnerability: they can be exploited for side-channel attacks. This paper introduces Mic-E-Mouse, the first-ever side-channel attack that targets high-performance optical mouse sensors to covertly eavesdrop on users. We demonstrate that audio signals can induce subtle surface vibrations detectable by a mouse's optical sensor. Remarkably, user-space software on popular operating systems can collect and broadcast this sensitive side channel, granting attackers access to raw mouse data without requiring direct system-level permissions. Initially, the vibration signals extracted from mouse data are of poor quality due to non-uniform sampling, a non-linear frequency response, and significant quantization. To overcome these limitations, Mic-E-Mouse employs a sophisticated end-to-end data filtering pipeline that combines Wiener filtering, resampling corrections, and an innovative encoder-only spectrogram neural filtering technique. We evaluate the attack's efficacy across diverse conditions, including speaking volume, mouse polling rate and DPI, surface materials, speaker languages, and environmental noise. In controlled environments, Mic-E-Mouse improves the signal-to-noise ratio (SNR) by up to +19 dB for speech reconstruction. Furthermore, our results demonstrate a speech recognition accuracy of roughly 42% to 61% on the AudioMNIST and VCTK datasets. All our code and datasets are publicly accessible on https://sites.google.com/view/mic-e-mouse.
>
---
#### [new 013] DSpAST: Disentangled Representations for Spatial Audio Reasoning with Large Language Models
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文提出DSpAST音频编码器，用于空间音频推理任务，解决单一编码器难以独立捕捉声音事件类型、方向和距离信息的问题。通过学习解耦表示，提升空间音频理解性能，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.13927v1](http://arxiv.org/pdf/2509.13927v1)**

> **作者:** Kevin Wilkinghoff; Zheng-Hua Tan
>
> **摘要:** Reasoning about spatial audio with large language models requires a spatial audio encoder as an acoustic front-end to obtain audio embeddings for further processing. Such an encoder needs to capture all information required to detect the type of sound events, as well as the direction and distance of their corresponding sources. Accomplishing this with a single audio encoder is demanding as the information required for each of these tasks is mostly independent of each other. As a result, the performance obtained with a single encoder is often worse than when using task-specific audio encoders. In this work, we present DSpAST, a novel audio encoder based on SpatialAST that learns disentangled representations of spatial audio while having only 0.2% additional parameters. Experiments on SpatialSoundQA with the spatial audio reasoning system BAT demonstrate that DSpAST significantly outperforms SpatialAST.
>
---
#### [new 014] Enhancing Speaker-Independent Dysarthric Speech Severity Classification with DSSCNet and Cross-Corpus Adaptation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决说话人无关的构音障碍语音严重程度分类问题。提出DSSCNet模型并结合跨语料库微调框架，提升模型泛化能力，在多个数据集上取得优于现有方法的分类准确率。**

- **链接: [http://arxiv.org/pdf/2509.13442v1](http://arxiv.org/pdf/2509.13442v1)**

> **作者:** Arnab Kumar Roy; Hemant Kumar Kathania; Paban Sapkota
>
> **备注:** Speaker-independent experiments on classification of dysarthric speech severity
>
> **摘要:** Dysarthric speech severity classification is crucial for objective clinical assessment and progress monitoring in individuals with motor speech disorders. Although prior methods have addressed this task, achieving robust generalization in speaker-independent (SID) scenarios remains challenging. This work introduces DSSCNet, a novel deep neural architecture that combines Convolutional, Squeeze-Excitation (SE), and Residual network, helping it extract discriminative representations of dysarthric speech from mel spectrograms. The addition of SE block selectively focuses on the important features of the dysarthric speech, thereby minimizing loss and enhancing overall model performance. We also propose a cross-corpus fine-tuning framework for severity classification, adapted from detection-based transfer learning approaches. DSSCNet is evaluated on two benchmark dysarthric speech corpora: TORGO and UA-Speech under speaker-independent evaluation protocols: One-Speaker-Per-Severity (OSPS) and Leave-One-Speaker-Out (LOSO) protocols. DSSCNet achieves accuracies of 56.84% and 62.62% under OSPS and 63.47% and 64.18% under LOSO setting on TORGO and UA-Speech respectively outperforming existing state-of-the-art methods. Upon fine-tuning, the performance improves substantially, with DSSCNet achieving up to 75.80% accuracy on TORGO and 68.25% on UA-Speech in OSPS, and up to 77.76% and 79.44%, respectively, in LOSO. These results demonstrate the effectiveness and generalizability of DSSCNet for fine-grained severity classification across diverse dysarthric speech datasets.
>
---
#### [new 015] Lightweight Implicit Neural Network for Binaural Audio Synthesis
- **分类: eess.AS; cs.SD**

- **简介: 论文提出轻量隐式神经网络LINN，用于双耳音频合成。旨在解决现有方法计算资源消耗大、难以应用于边缘设备的问题。通过两阶段框架，结合时间域变形与隐式修正模块，实现高保真音质与高效计算的平衡。**

- **链接: [http://arxiv.org/pdf/2509.14069v1](http://arxiv.org/pdf/2509.14069v1)**

> **作者:** Xikun Lu; Fang Liu; Weizhi Shi; Jinqiu Sang
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** High-fidelity binaural audio synthesis is crucial for immersive listening, but existing methods require extensive computational resources, limiting their edge-device application. To address this, we propose the Lightweight Implicit Neural Network (LINN), a novel two-stage framework. LINN first generates initial estimates using a time-domain warping, which is then refined by an Implicit Binaural Corrector (IBC) module. IBC is an implicit neural network that predicts amplitude and phase corrections directly, resulting in a highly compact model architecture. Experimental results show that LINN achieves statistically comparable perceptual quality to the best-performing baseline model while significantly improving computational efficiency. Compared to the most efficient existing method, LINN achieves a 72.7% reduction in parameters and significantly fewer compute operations (MACs). This demonstrates that our approach effectively addresses the trade-off between synthesis quality and computational efficiency, providing a new solution for high-fidelity edge-device spatial audio applications.
>
---
#### [new 016] Mixture of Low-Rank Adapter Experts in Generalizable Audio Deepfake Detection
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 论文提出一种混合低秩适配器专家（MoE-LoRA）方法，用于提升音频深度伪造检测的泛化能力。该方法通过集成多个低秩适配器并引入路由机制，增强模型对新型攻击的适应性，有效降低出域错误率，解决传统微调模型泛化能力不足的问题。**

- **链接: [http://arxiv.org/pdf/2509.13878v1](http://arxiv.org/pdf/2509.13878v1)**

> **作者:** Janne Laakkonen; Ivan Kukanov; Ville Hautamäki
>
> **备注:** 6 pages, 3 figures, 1 table
>
> **摘要:** Foundation models such as Wav2Vec2 excel at representation learning in speech tasks, including audio deepfake detection. However, after being fine-tuned on a fixed set of bonafide and spoofed audio clips, they often fail to generalize to novel deepfake methods not represented in training. To address this, we propose a mixture-of-LoRA-experts approach that integrates multiple low-rank adapters (LoRA) into the model's attention layers. A routing mechanism selectively activates specialized experts, enhancing adaptability to evolving deepfake attacks. Experimental results show that our method outperforms standard fine-tuning in both in-domain and out-of-domain scenarios, reducing equal error rates relative to baseline models. Notably, our best MoE-LoRA model lowers the average out-of-domain EER from 8.55\% to 6.08\%, demonstrating its effectiveness in achieving generalizable audio deepfake detection.
>
---
## 更新

#### [replaced 001] Omni-CLST: Error-aware Curriculum Learning with guided Selective chain-of-Thought for audio question answering
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.12275v2](http://arxiv.org/pdf/2509.12275v2)**

> **作者:** Jinghua Zhao; Hang Su; Lichun Fan; Zhenbo Luo; Jian Luan; Hui Wang; Haoqin Sun; Yong Qin
>
> **备注:** 5 pages, 1 figure, 2 tables submitted to icassp, under prereview
>
> **摘要:** With the rapid progress of large audio-language models (LALMs), audio question answering (AQA) has emerged as a challenging task requiring both fine-grained audio understanding and complex reasoning. While current methods mainly rely on constructing new datasets via captioning or reasoning traces, existing high-quality AQA data remains underutilized. To address this, we propose Omni-CLST, an error-aware Curriculum Learning framework with guided Selective Chain-of-Thought. The framework efficiently leverages existing high-quality dataset through two key strategies: an error-aware curriculum that organizes samples by difficulty, and a guided thought dropout mechanism that focuses reasoning on challenging cases. Experiments show that Omni-CLST achieves 73.80% on MMAU-mini and a new state of the art of 64.30% on MMAR, demonstrating robust generalization in multimodal audio-language understanding.
>
---
#### [replaced 002] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11051v2](http://arxiv.org/pdf/2505.11051v2)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Under review at ICASSP
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [replaced 003] Improving Generalizability of Kolmogorov-Arnold Networks via Error-Correcting Output Codes
- **分类: cs.LG; cs.CV; eess.IV; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.05798v2](http://arxiv.org/pdf/2505.05798v2)**

> **作者:** Youngjoon Lee; Jinu Gong; Joonhyuk Kang
>
> **备注:** Accepted to IEEE BioCAS 2025
>
> **摘要:** Kolmogorov-Arnold Networks (KAN) offer universal function approximation using univariate spline compositions without nonlinear activations. In this work, we integrate Error-Correcting Output Codes (ECOC) into the KAN framework to transform multi-class classification into multiple binary tasks, improving robustness via Hamming distance decoding. Our proposed KAN with ECOC framework outperforms vanilla KAN on a challenging blood cell classification dataset, achieving higher accuracy across diverse hyperparameter settings. Ablation studies further confirm that ECOC consistently enhances performance across FastKAN and FasterKAN variants. These results demonstrate that ECOC integration significantly boosts KAN generalizability in critical healthcare AI applications. To the best of our knowledge, this is the first work of ECOC with KAN for enhancing multi-class medical image classification performance.
>
---
#### [replaced 004] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18614v3](http://arxiv.org/pdf/2505.18614v3)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** Accepted to EMNLP 2025, Project Page: https://k1064190.github.io/papers/paper1.html, our codes and datasets are available at https://github.com/k1064190/MAVL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 005] Empathy Omni: Enabling Empathetic Speech Response Generation through Large Language Models
- **分类: cs.CL; cs.SD; eess.AS; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.18655v3](http://arxiv.org/pdf/2508.18655v3)**

> **作者:** Haoyu Wang; Guangyan Zhang; Jiale Chen; Jingyu Li; Yuehai Wang; Yiwen Guo
>
> **备注:** 5 pages, 1 figure, submitted to ICASSP 2026
>
> **摘要:** With the development of speech large language models (speech LLMs), users can now interact directly with assistants via speech. However, most existing models only convert response content into speech without fully capturing the rich emotional cues in user queries, where the same sentence may convey different meanings depending on the expression. Emotional understanding is thus essential for improving human-machine interaction. Most empathetic speech LLMs rely on massive datasets, demanding high computational cost. A key challenge is to build models that generate empathetic responses with limited data and without large-scale training. To this end, we propose Emotion Omni, a model that understands emotional content in user speech and generates empathetic responses. We further developed a data pipeline to construct a 200k emotional dialogue dataset supporting empathetic speech assistants. Experiments show that Emotion Omni achieves comparable instruction-following ability without large-scale pretraining, while surpassing existing models in speech quality (UTMOS:4.41) and empathy (Emotion GPT Score: 3.97). These results confirm its improvements in both speech fidelity and emotional expressiveness. Demos are available at https://w311411.github.io/omni_demo/.
>
---
#### [replaced 006] KALL-E:Autoregressive Speech Synthesis with Next-Distribution Prediction
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2412.16846v2](http://arxiv.org/pdf/2412.16846v2)**

> **作者:** Kangxiang Xia; Xinfa Zhu; Jixun Yao; Wenjie Tian; Wenhao Li; Lei Xie
>
> **备注:** 6 figures, 5 tables
>
> **摘要:** We introduce KALL-E, a novel autoregressive (AR) language model for text-to-speech (TTS) synthesis that operates by predicting the next distribution of continuous speech frames. Unlike existing methods, KALL-E directly models the continuous speech distribution conditioned on text, eliminating the need for any diffusion-based components. Specifically, we utilize a Flow-VAE to extract a continuous latent speech representation from waveforms, instead of relying on discrete speech tokens. A single AR Transformer is then trained to predict these continuous speech distributions from text, optimizing a Kullback-Leibler divergence loss as its objective. Experimental results demonstrate that KALL-E achieves superior speech synthesis quality and can even adapt to a target speaker from just a single sample. Importantly, KALL-E provides a more direct and effective approach for utilizing continuous speech representations in TTS.
>
---
#### [replaced 007] Spike Encoding for Environmental Sound: A Comparative Benchmark
- **分类: cs.SD; cs.ET; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.11206v3](http://arxiv.org/pdf/2503.11206v3)**

> **作者:** Andres Larroza; Javier Naranjo-Alcazar; Vicent Ortiz; Pedro Zuccarello
>
> **备注:** Under review ICASSP 2026
>
> **摘要:** Spiking Neural Networks (SNNs) offer energy efficient processing suitable for edge applications, but conventional sensor data must first be converted into spike trains for neuromorphic processing. Environmental sound, including urban soundscapes, poses challenges due to variable frequencies, background noise, and overlapping acoustic events, while most spike based audio encoding research has focused on speech. This paper analyzes three spike encoding methods, Threshold Adaptive Encoding (TAE), Step Forward (SF), and Moving Window (MW) across three datasets: ESC10, UrbanSound8K, and TAU Urban Acoustic Scenes. Our multiband analysis shows that TAE consistently outperforms SF and MW in reconstruction quality, both per frequency band and per class across datasets. Moreover, TAE yields the lowest spike firing rates, indicating superior energy efficiency. For downstream environmental sound classification with a standard SNN, TAE also achieves the best performance among the compared encoders. Overall, this work provides foundational insights and a comparative benchmark to guide the selection of spike encoders for neuromorphic environmental sound processing.
>
---
#### [replaced 008] Video-Foley: Two-Stage Video-To-Sound Generation via Temporal Event Condition For Foley Sound
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2408.11915v3](http://arxiv.org/pdf/2408.11915v3)**

> **作者:** Junwon Lee; Jaekwon Im; Dabin Kim; Juhan Nam
>
> **备注:** Accepted at IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Foley sound synthesis is crucial for multimedia production, enhancing user experience by synchronizing audio and video both temporally and semantically. Recent studies on automating this labor-intensive process through video-to-sound generation face significant challenges. Systems lacking explicit temporal features suffer from poor alignment and controllability, while timestamp-based models require costly and subjective human annotation. We propose Video-Foley, a video-to-sound system using Root Mean Square (RMS) as an intuitive condition with semantic timbre prompts (audio or text). RMS, a frame-level intensity envelope closely related to audio semantics, acts as a temporal event feature to guide audio generation from video. The annotation-free self-supervised learning framework consists of two stages, Video2RMS and RMS2Sound, incorporating novel ideas including RMS discretization and RMS-ControlNet with a pretrained text-to-audio model. Our extensive evaluation shows that Video-Foley achieves state-of-the-art performance in audio-visual alignment and controllability for sound timing, intensity, timbre, and nuance. Source code, model weights and demos are available on our companion website. (https://jnwnlee.github.io/video-foley-demo)
>
---
#### [replaced 009] ECHO: Frequency-aware Hierarchical Encoding for Variable-length Signals
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.14689v2](http://arxiv.org/pdf/2508.14689v2)**

> **作者:** Yucong Zhang; Juan Liu; Ming Li
>
> **备注:** submitted to ICASSP 2026
>
> **摘要:** Pre-trained foundation models have demonstrated remarkable success in audio, vision and language, yet their potential for general machine signal modeling with arbitrary sampling rates-covering acoustic, vibration, and other industrial sensor data-remains under-explored. In this work, we propose a novel foundation model ECHO that integrates an advanced band-split architecture with frequency positional embeddings, enabling spectral localization across arbitrary sampling configurations. Moreover, the model incorporates sliding patches to support inputs of variable length without padding or cropping, producing a concise embedding that retains both temporal and spectral fidelity and naturally extends to streaming scenarios. We evaluate our method on various kinds of machine signal datasets, including previous DCASE task 2 challenges (2020-2025), and widely-used industrial signal corpora. Experimental results demonstrate consistent state-of-the-art performance in machine signal anomaly detection and fault classification, confirming the effectiveness and generalization capability of the proposed model. We open-sourced ECHO on https://github.com/yucongzh/ECHO.
>
---
