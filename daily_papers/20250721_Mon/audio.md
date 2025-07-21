# 音频 cs.SD;  eess.SP

- **最新发布 5 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] OpenBEATs: A Fully Open-Source General-Purpose Audio Encoder
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频表示学习任务，旨在解决BEATs模型因未开源预训练代码和单一训练数据导致的适用性受限问题。论文提出了OpenBEATs框架，通过多领域音频预训练扩展BEATs，并在多个音频任务和数据集中验证其性能，实现了更优的音频理解效果。**

- **链接: [http://arxiv.org/pdf/2507.14129v1](http://arxiv.org/pdf/2507.14129v1)**

> **作者:** Shikhar Bharadwaj; Samuele Cornell; Kwanghee Choi; Satoru Fukayama; Hye-jin Shim; Soham Deshmukh; Shinji Watanabe
>
> **摘要:** Masked token prediction has emerged as a powerful pre-training objective across language, vision, and speech, offering the potential to unify these diverse modalities through a single pre-training task. However, its application for general audio understanding remains underexplored, with BEATs being the only notable example. BEATs has seen limited modifications due to the absence of open-source pre-training code. Furthermore, BEATs was trained only on AudioSet, restricting its broader downstream applicability. To address these gaps, we present OpenBEATs, an open-source framework that extends BEATs via multi-domain audio pre-training. We conduct comprehensive evaluations across six types of tasks, twenty five datasets, and three audio domains, including audio reasoning tasks such as audio question answering, entailment, and captioning. OpenBEATs achieves state-of-the-art performance on six bioacoustics datasets, two environmental sound datasets and five reasoning datasets, performing better than models exceeding a billion parameters at one-fourth their parameter size. These results demonstrate the effectiveness of multi-domain datasets and masked token prediction task to learn general-purpose audio representations. To promote further research and reproducibility, we release all pre-training and evaluation code, pretrained and fine-tuned checkpoints, and training logs at https://shikhar-s.github.io/OpenBEATs
>
---
#### [new 002] Temporal Adaptation of Pre-trained Foundation Models for Music Structure Analysis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索中的音乐结构分析任务，旨在解决预训练模型在长音频处理中的效率与偏差问题。论文提出了一种时间适应方法，通过音频窗口扩展和低分辨率适应，实现对完整歌曲的高效单次分析，提升了边界检测与结构功能预测效果。**

- **链接: [http://arxiv.org/pdf/2507.13572v1](http://arxiv.org/pdf/2507.13572v1)**

> **作者:** Yixiao Zhang; Haonan Chen; Ju-Chiang Wang; Jitong Chen
>
> **备注:** Accepted to WASPAA 2025. Project Page: https://sites.google.com/view/temporal-adaptation-for-msa/
>
> **摘要:** Audio-based music structure analysis (MSA) is an essential task in Music Information Retrieval that remains challenging due to the complexity and variability of musical form. Recent advances highlight the potential of fine-tuning pre-trained music foundation models for MSA tasks. However, these models are typically trained with high temporal feature resolution and short audio windows, which limits their efficiency and introduces bias when applied to long-form audio. This paper presents a temporal adaptation approach for fine-tuning music foundation models tailored to MSA. Our method enables efficient analysis of full-length songs in a single forward pass by incorporating two key strategies: (1) audio window extension and (2) low-resolution adaptation. Experiments on the Harmonix Set and RWC-Pop datasets show that our method significantly improves both boundary detection and structural function prediction, while maintaining comparable memory usage and inference speed.
>
---
#### [new 003] Controlling the Parameterized Multi-channel Wiener Filter using a tiny neural network
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决噪声抑制与语音失真之间的平衡问题。作者提出NeuralPMWF系统，通过轻量级神经网络控制参数化多通道维纳滤波器（PMWF），在低计算资源下实现高效语音增强，取得了优于基线方法的效果。**

- **链接: [http://arxiv.org/pdf/2507.13863v1](http://arxiv.org/pdf/2507.13863v1)**

> **作者:** Eric Grinstein; Ashutosh Pandey; Cole Li; Shanmukha Srinivas; Juan Azcarreta; Jacob Donley; Sanha Lee; Ali Aroudi; Cagdas Bilen
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** Noise suppression and speech distortion are two important aspects to be balanced when designing multi-channel Speech Enhancement (SE) algorithms. Although neural network models have achieved state-of-the-art noise suppression, their non-linear operations often introduce high speech distortion. Conversely, classical signal processing algorithms such as the Parameterized Multi-channel Wiener Filter ( PMWF) beamformer offer explicit mechanisms for controlling the suppression/distortion trade-off. In this work, we present NeuralPMWF, a system where the PMWF is entirely controlled using a low-latency, low-compute neural network, resulting in a low-complexity system offering high noise reduction and low speech distortion. Experimental results show that our proposed approach results in significantly better perceptual and objective speech enhancement in comparison to several competitive baselines using similar computational resources.
>
---
#### [new 004] A Data-Centric Framework for Addressing Phonetic and Prosodic Challenges in Russian Speech Generative Models
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决俄语语音生成中的音系与韵律挑战，如元音弱化、辅音清化等。作者构建了高质量俄语语音数据集Balalaika，并展示了其在语音合成与增强任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13563v1](http://arxiv.org/pdf/2507.13563v1)**

> **作者:** Kirill Borodin; Nikita Vasiliev; Vasiliy Kudryavtsev; Maxim Maslov; Mikhail Gorodnichev; Oleg Rogov; Grach Mkrtchian
>
> **备注:** The work is still in progress
>
> **摘要:** Russian speech synthesis presents distinctive challenges, including vowel reduction, consonant devoicing, variable stress patterns, homograph ambiguity, and unnatural intonation. This paper introduces Balalaika, a novel dataset comprising more than 2,000 hours of studio-quality Russian speech with comprehensive textual annotations, including punctuation and stress markings. Experimental results show that models trained on Balalaika significantly outperform those trained on existing datasets in both speech synthesis and enhancement tasks. We detail the dataset construction pipeline, annotation methodology, and results of comparative evaluations.
>
---
#### [new 005] Unifying Listener Scoring Scales: Comparison Learning Framework for Speech Quality Assessment and Continuous Speech Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音技术任务，旨在解决语音质量评估与情感识别中因听众评分尺度偏差导致的模型性能问题。论文提出一种统一分数尺度建模方法，通过比较评分关系提升预测效果，验证了其有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13626v1](http://arxiv.org/pdf/2507.13626v1)**

> **作者:** Cheng-Hung Hu; Yusuke Yasud; Akifumi Yoshimoto; Tomoki Toda
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Speech Quality Assessment (SQA) and Continuous Speech Emotion Recognition (CSER) are two key tasks in speech technology, both relying on listener ratings. However, these ratings are inherently biased due to individual listener factors. Previous approaches have introduced a mean listener scoring scale and modeled all listener scoring scales in the training set. However, the mean listener approach is prone to distortion from averaging ordinal data, leading to potential biases. Moreover, learning multiple listener scoring scales while inferring based only on the mean listener scale limits effectiveness. In contrast, our method focuses on modeling a unified listener scoring scale, using comparison scores to correctly capture the scoring relationships between utterances. Experimental results show that our method effectively improves prediction performance in both SQA and CSER tasks, proving its effectiveness and robustness.
>
---
## 更新

#### [replaced 001] A lightweight and robust method for blind wideband-to-fullband extension of speech
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2412.11392v4](http://arxiv.org/pdf/2412.11392v4)**

> **作者:** Jan Büthe; Jean-Marc Valin
>
> **备注:** WASPAA 2025, 5 pages
>
> **摘要:** Reducing the bandwidth of speech is common practice in resource constrained environments like low-bandwidth speech transmission or low-complexity vocoding. We propose a lightweight and robust method for extending the bandwidth of wideband speech signals that is inspired by classical methods developed in the speech coding context. The resulting model has just ~370K parameters and a complexity of ~140 MFLOPS (or ~70 MMACS). With a frame size of 10 ms and a lookahead of only 0.27 ms, the model is well-suited for use with common wideband speech codecs. We evaluate the model's robustness by pairing it with the Opus SILK speech codec (1.5 release) and verify in a P.808 DCR listening test that it significantly improves quality from 6 to 12 kb/s. We also demonstrate that Opus 1.5 together with the proposed bandwidth extension at 9 kb/s meets the quality of 3GPP EVS at 9.6 kb/s and that of Opus 1.4 at 18 kb/s showing that the blind bandwidth extension can meet the quality of classical guided bandwidth extensions thus providing a way for backward-compatible quality improvement.
>
---
#### [replaced 002] Align Your Rhythm: Generating Highly Aligned Dance Poses with Gating-Enhanced Rhythm-Aware Feature Representation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.17340v2](http://arxiv.org/pdf/2503.17340v2)**

> **作者:** Congyi Fan; Jian Guan; Xuanjia Zhao; Dongli Xu; Youtian Lin; Tong Ye; Pengming Feng; Haiwei Pan
>
> **备注:** ICCV 2025 Accept, Project page: https://danceba.github.io/
>
> **摘要:** Automatically generating natural, diverse and rhythmic human dance movements driven by music is vital for virtual reality and film industries. However, generating dance that naturally follows music remains a challenge, as existing methods lack proper beat alignment and exhibit unnatural motion dynamics. In this paper, we propose Danceba, a novel framework that leverages gating mechanism to enhance rhythm-aware feature representation for music-driven dance generation, which achieves highly aligned dance poses with enhanced rhythmic sensitivity. Specifically, we introduce Phase-Based Rhythm Extraction (PRE) to precisely extract rhythmic information from musical phase data, capitalizing on the intrinsic periodicity and temporal structures of music. Additionally, we propose Temporal-Gated Causal Attention (TGCA) to focus on global rhythmic features, ensuring that dance movements closely follow the musical rhythm. We also introduce Parallel Mamba Motion Modeling (PMMM) architecture to separately model upper and lower body motions along with musical features, thereby improving the naturalness and diversity of generated dance movements. Extensive experiments confirm that Danceba outperforms state-of-the-art methods, achieving significantly better rhythmic alignment and motion diversity. Project page: https://danceba.github.io/ .
>
---
#### [replaced 003] Source Separation by Flow Matching
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.16119v2](http://arxiv.org/pdf/2505.16119v2)**

> **作者:** Robin Scheibler; John R. Hershey; Arnaud Doucet; Henry Li
>
> **备注:** 5 pages, 3 figures, 2 tables, accepted at WASPAA 2025
>
> **摘要:** We consider the problem of single-channel audio source separation with the goal of reconstructing $K$ sources from their mixture. We address this ill-posed problem with FLOSS (FLOw matching for Source Separation), a constrained generation method based on flow matching, ensuring strict mixture consistency. Flow matching is a general methodology that, when given samples from two probability distributions defined on the same space, learns an ordinary differential equation to output a sample from one of the distributions when provided with a sample from the other. In our context, we have access to samples from the joint distribution of $K$ sources and so the corresponding samples from the lower-dimensional distribution of their mixture. To apply flow matching, we augment these mixture samples with artificial noise components to match the dimensionality of the $K$ source distribution. Additionally, as any permutation of the sources yields the same mixture, we adopt an equivariant formulation of flow matching which relies on a neural network architecture that is equivariant by design. We demonstrate the performance of the method for the separation of overlapping speech.
>
---
#### [replaced 004] WildFX: A DAW-Powered Pipeline for In-the-Wild Audio FX Graph Modeling
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.10534v2](http://arxiv.org/pdf/2507.10534v2)**

> **作者:** Qihui Yang; Taylor Berg-Kirkpatrick; Julian McAuley; Zachary Novack
>
> **摘要:** Despite rapid progress in end-to-end AI music generation, AI-driven modeling of professional Digital Signal Processing (DSP) workflows remains challenging. In particular, while there is growing interest in neural black-box modeling of audio effect graphs (e.g. reverb, compression, equalization), AI-based approaches struggle to replicate the nuanced signal flow and parameter interactions used in professional workflows. Existing differentiable plugin approaches often diverge from real-world tools, exhibiting inferior performance relative to simplified neural controllers under equivalent computational constraints. We introduce WildFX, a pipeline containerized with Docker for generating multi-track audio mixing datasets with rich effect graphs, powered by a professional Digital Audio Workstation (DAW) backend. WildFX supports seamless integration of cross-platform commercial plugins or any plugins in the wild, in VST/VST3/LV2/CLAP formats, enabling structural complexity (e.g., sidechains, crossovers) and achieving efficient parallelized processing. A minimalist metadata interface simplifies project/plugin configuration. Experiments demonstrate the pipeline's validity through blind estimation of mixing graphs, plugin/gain parameters, and its ability to bridge AI research with practical DSP demands. The code is available on: https://github.com/IsaacYQH/WildFX.
>
---
#### [replaced 005] MuteSwap: Visual-informed Silent Video Identity Conversion
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00498v2](http://arxiv.org/pdf/2507.00498v2)**

> **作者:** Yifan Liu; Yu Fang; Zhouhan Lin
>
> **摘要:** Conventional voice conversion modifies voice characteristics from a source speaker to a target speaker, relying on audio input from both sides. However, this process becomes infeasible when clean audio is unavailable, such as in silent videos or noisy environments. In this work, we focus on the task of Silent Face-based Voice Conversion (SFVC), which does voice conversion entirely from visual inputs. i.e., given images of a target speaker and a silent video of a source speaker containing lip motion, SFVC generates speech aligning the identity of the target speaker while preserving the speech content in the source silent video. As this task requires generating intelligible speech and converting identity using only visual cues, it is particularly challenging. To address this, we introduce MuteSwap, a novel framework that employs contrastive learning to align cross-modality identities and minimize mutual information to separate shared visual features. Experimental results show that MuteSwap achieves impressive performance in both speech synthesis and identity conversion, especially under noisy conditions where methods dependent on audio input fail to produce intelligible results, demonstrating both the effectiveness of our training approach and the feasibility of SFVC.
>
---
#### [replaced 006] Instruct-MusicGen: Unlocking Text-to-Music Editing for Music Language Models via Instruction Tuning
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2405.18386v3](http://arxiv.org/pdf/2405.18386v3)**

> **作者:** Yixiao Zhang; Yukara Ikemiya; Woosung Choi; Naoki Murata; Marco A. Martínez-Ramírez; Liwei Lin; Gus Xia; Wei-Hsiang Liao; Yuki Mitsufuji; Simon Dixon
>
> **备注:** Accepted at ISMIR 2025 Conference. Code and demo are available at: https://github.com/ldzhangyx/instruct-musicgen
>
> **摘要:** Recent advances in text-to-music editing, which employ text queries to modify music (e.g.\ by changing its style or adjusting instrumental components), present unique challenges and opportunities for AI-assisted music creation. Previous approaches in this domain have been constrained by the necessity to train specific editing models from scratch, which is both resource-intensive and inefficient; other research uses large language models to predict edited music, resulting in imprecise audio reconstruction. To Combine the strengths and address these limitations, we introduce Instruct-MusicGen, a novel approach that finetunes a pretrained MusicGen model to efficiently follow editing instructions such as adding, removing, or separating stems. Our approach involves a modification of the original MusicGen architecture by incorporating a text fusion module and an audio fusion module, which allow the model to process instruction texts and audio inputs concurrently and yield the desired edited music. Remarkably, Instruct-MusicGen only introduces 8% new parameters to the original MusicGen model and only trains for 5K steps, yet it achieves superior performance across all tasks compared to existing baselines, and demonstrates performance comparable to the models trained for specific tasks. This advancement not only enhances the efficiency of text-to-music editing but also broadens the applicability of music language models in dynamic music production environments.
>
---
#### [replaced 007] SpecMaskFoley: Steering Pretrained Spectral Masked Generative Transformer Toward Synchronized Video-to-audio Synthesis via ControlNet
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.16195v2](http://arxiv.org/pdf/2505.16195v2)**

> **作者:** Zhi Zhong; Akira Takahashi; Shuyang Cui; Keisuke Toyama; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** WASPAA 2025. 4 pages, 2 figures, 2 tables. Demo page: https://zzaudio.github.io/SpecMaskFoley_Demo/
>
> **摘要:** Foley synthesis aims to synthesize high-quality audio that is both semantically and temporally aligned with video frames. Given its broad application in creative industries, the task has gained increasing attention in the research community. To avoid the non-trivial task of training audio generative models from scratch, adapting pretrained audio generative models for video-synchronized foley synthesis presents an attractive direction. ControlNet, a method for adding fine-grained controls to pretrained generative models, has been applied to foley synthesis, but its use has been limited to handcrafted human-readable temporal conditions. In contrast, from-scratch models achieved success by leveraging high-dimensional deep features extracted using pretrained video encoders. We have observed a performance gap between ControlNet-based and from-scratch foley models. To narrow this gap, we propose SpecMaskFoley, a method that steers the pretrained SpecMaskGIT model toward video-synchronized foley synthesis via ControlNet. To unlock the potential of a single ControlNet branch, we resolve the discrepancy between the temporal video features and the time-frequency nature of the pretrained SpecMaskGIT via a frequency-aware temporal feature aligner, eliminating the need for complicated conditioning mechanisms widely used in prior arts. Evaluations on a common foley synthesis benchmark demonstrate that SpecMaskFoley could even outperform strong from-scratch baselines, substantially advancing the development of ControlNet-based foley synthesis models. Demo page: https://zzaudio.github.io/SpecMaskFoley_Demo/
>
---
#### [replaced 008] Incremental Averaging Method to Improve Graph-Based Time-Difference-of-Arrival Estimation
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.07087v2](http://arxiv.org/pdf/2507.07087v2)**

> **作者:** Klaus Brümann; Kouei Yamaoka; Nobutaka Ono; Simon Doclo
>
> **摘要:** Estimating the position of a speech source based on time-differences-of-arrival (TDOAs) is often adversely affected by background noise and reverberation. A popular method to estimate the TDOA between a microphone pair involves maximizing a generalized cross-correlation with phase transform (GCC-PHAT) function. Since the TDOAs across different microphone pairs satisfy consistency relations, generally only a small subset of microphone pairs are used for source position estimation. Although the set of microphone pairs is often determined based on a reference microphone, recently a more robust method has been proposed to determine the set of microphone pairs by computing the minimum spanning tree (MST) of a signal graph of GCC-PHAT function reliabilities. To reduce the influence of noise and reverberation on the TDOA estimation accuracy, in this paper we propose to compute the GCC-PHAT functions of the MST based on an average of multiple cross-power spectral densities (CPSDs) using an incremental method. In each step of the method, we increase the number of CPSDs over which we average by considering CPSDs computed indirectly via other microphones from previous steps. Using signals recorded in a noisy and reverberant laboratory with an array of spatially distributed microphones, the performance of the proposed method is evaluated in terms of TDOA estimation error and 2D source position estimation error. Experimental results for different source and microphone configurations and three reverberation conditions show that the proposed method considering multiple CPSDs improves the TDOA estimation and source position estimation accuracy compared to the reference microphone- and MST-based methods that rely on a single CPSD as well as steered-response power-based source position estimation.
>
---
