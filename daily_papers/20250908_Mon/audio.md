# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] A Multiclass Acoustic Dataset and Interactive Tool for Analyzing Drone Signatures in Real-World Environments
- **分类: cs.SD; eess.AS**

- **简介: 论文构建多类无人机声学数据集及交互工具，解决现有视觉/雷达检测的局限，通过音频、频谱图和MFCC数据支持分类与分析，促进无人机声学研究。**

- **链接: [http://arxiv.org/pdf/2509.04715v1](http://arxiv.org/pdf/2509.04715v1)**

> **作者:** Mia Y. Wang; Mackenzie Linn; Andrew P. Berg; Qian Zhang
>
> **备注:** This article extends our previous work presented in the 2024 Artificial Intelligence x Humanities, Education, and Art (2024 AIxHeart) Conference
>
> **摘要:** The rapid proliferation of drones across various industries has introduced significant challenges related to privacy, security, and noise pollution. Current drone detection systems, primarily based on visual and radar technologies, face limitations under certain conditions, highlighting the need for effective acoustic-based detection methods. This paper presents a unique and comprehensive dataset of drone acoustic signatures, encompassing 32 different categories differentiated by brand and model. The dataset includes raw audio recordings, spectrogram plots, and Mel-frequency cepstral coefficient (MFCC) plots for each drone. Additionally, we introduce an interactive web application that allows users to explore this dataset by selecting specific drone categories, listening to the associated audio, and viewing the corresponding spectrogram and MFCC plots. This tool aims to facilitate research in drone detection, classification, and acoustic analysis, supporting both technological advancements and educational initiatives. The paper details the dataset creation process, the design and implementation of the web application, and provides experimental results and user feedback. Finally, we discuss potential applications and future work to expand and enhance the project.
>
---
#### [new 002] Training a Perceptual Model for Evaluating Auditory Similarity in Music Adversarial Attack
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出PAMT框架，解决音乐对抗攻击中模型与人类听觉感知不匹配的问题。通过心理声学条件的对比变压器，提升感知评估相关性与鲁棒性，改善音乐信息检索任务的抗攻击能力。**

- **链接: [http://arxiv.org/pdf/2509.04985v1](http://arxiv.org/pdf/2509.04985v1)**

> **作者:** Yuxuan Liu; Rui Sang; Peihong Zhang; Zhixin Li; Shengchen Li
>
> **摘要:** Music Information Retrieval (MIR) systems are highly vulnerable to adversarial attacks that are often imperceptible to humans, primarily due to a misalignment between model feature spaces and human auditory perception. Existing defenses and perceptual metrics frequently fail to adequately capture these auditory nuances, a limitation supported by our initial listening tests showing low correlation between common metrics and human judgments. To bridge this gap, we introduce Perceptually-Aligned MERT Transformer (PAMT), a novel framework for learning robust, perceptually-aligned music representations. Our core innovation lies in the psychoacoustically-conditioned sequential contrastive transformer, a lightweight projection head built atop a frozen MERT encoder. PAMT achieves a Spearman correlation coefficient of 0.65 with subjective scores, outperforming existing perceptual metrics. Our approach also achieves an average of 9.15\% improvement in robust accuracy on challenging MIR tasks, including Cover Song Identification and Music Genre Classification, under diverse perceptual adversarial attacks. This work pioneers architecturally-integrated psychoacoustic conditioning, yielding representations significantly more aligned with human perception and robust against music adversarial attacks.
>
---
#### [new 003] Quantum Fourier Transform Based Denoising: Unitary Filtering for Enhanced Speech Clarity
- **分类: cs.SD; cs.ET; eess.AS**

- **简介: 该论文提出基于量子傅里叶变换（QFT）的语音去噪方法，替代传统FFT以提升语音清晰度。通过保持能量与相位一致性，QFT在低信噪比和非平稳噪声下实现显著SNR增益（最高15dB），减少伪影，展现高效鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.04851v1](http://arxiv.org/pdf/2509.04851v1)**

> **作者:** Rajeshwar Tripathi; Sahil Tomar; Sandeep Kumar; Monika Aggarwal
>
> **备注:** 8 pages
>
> **摘要:** This paper introduces a quantum-inspired denoising framework that integrates the Quantum Fourier Transform (QFT) into classical audio enhancement pipelines. Unlike conventional Fast Fourier Transform (FFT) based methods, QFT provides a unitary transformation with global phase coherence and energy preservation, enabling improved discrimination between speech and noise. The proposed approach replaces FFT in Wiener and spectral subtraction filters with a QFT operator, ensuring consistent hyperparameter settings for fair comparison. Experiments on clean speech, synthetic tones, and noisy mixtures across diverse signal to noise ratio (SNR) conditions, demonstrate statistically significant gains in SNR, with up to 15 dB improvement and reduced artifact generation. Results confirm that QFT based denoising offers robustness under low SNR and nonstationary noise scenarios without additional computational overhead, highlighting its potential as a scalable pathway toward quantum-enhanced speech processing.
>
---
#### [new 004] Ecologically Valid Benchmarking and Adaptive Attention: Scalable Marine Bioacoustic Monitoring
- **分类: cs.SD; cs.AI; cs.CV; cs.IR; cs.LG; eess.AS**

- **简介: 该论文提出GetNetUPAM框架和ARPA-N网络，解决海洋生物声学监测中噪声干扰和信号依赖问题，提升模型稳定性与泛化能力，实现高精度、可扩展的生态监测。**

- **链接: [http://arxiv.org/pdf/2509.04682v1](http://arxiv.org/pdf/2509.04682v1)**

> **作者:** Nicholas R. Rasmussen; Rodrigue Rizk; Longwei Wang; KC Santosh
>
> **备注:** Under review as an anonymous submission to IEEETAI - We are allowed an archive submission. Final formatting is yet to be determined
>
> **摘要:** Underwater Passive Acoustic Monitoring (UPAM) provides rich spatiotemporal data for long-term ecological analysis, but intrinsic noise and complex signal dependencies hinder model stability and generalization. Multilayered windowing has improved target sound localization, yet variability from shifting ambient noise, diverse propagation effects, and mixed biological and anthropogenic sources demands robust architectures and rigorous evaluation. We introduce GetNetUPAM, a hierarchical nested cross-validation framework designed to quantify model stability under ecologically realistic variability. Data are partitioned into distinct site-year segments, preserving recording heterogeneity and ensuring each validation fold reflects a unique environmental subset, reducing overfitting to localized noise and sensor artifacts. Site-year blocking enforces evaluation against genuine environmental diversity, while standard cross-validation on random subsets measures generalization across UPAM's full signal distribution, a dimension absent from current benchmarks. Using GetNetUPAM as the evaluation backbone, we propose the Adaptive Resolution Pooling and Attention Network (ARPA-N), a neural architecture for irregular spectrogram dimensions. Adaptive pooling with spatial attention extends the receptive field, capturing global context without excessive parameters. Under GetNetUPAM, ARPA-N achieves a 14.4% gain in average precision over DenseNet baselines and a log2-scale order-of-magnitude drop in variability across all metrics, enabling consistent detection across site-year folds and advancing scalable, accurate bioacoustic monitoring.
>
---
#### [new 005] MAIA: An Inpainting-Based Approach for Music Adversarial Attacks
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出MAIA框架，针对音乐信息检索系统设计对抗攻击，通过重要性分析与生成式修复模型实现隐蔽性强、成功率高的攻击，验证了当前MIR系统的脆弱性。**

- **链接: [http://arxiv.org/pdf/2509.04980v1](http://arxiv.org/pdf/2509.04980v1)**

> **作者:** Yuxuan Liu; Peihong Zhang; Rui Sang; Zhixin Li; Shengchen Li
>
> **备注:** Accepted at ISMIR2025
>
> **摘要:** Music adversarial attacks have garnered significant interest in the field of Music Information Retrieval (MIR). In this paper, we present Music Adversarial Inpainting Attack (MAIA), a novel adversarial attack framework that supports both white-box and black-box attack scenarios. MAIA begins with an importance analysis to identify critical audio segments, which are then targeted for modification. Utilizing generative inpainting models, these segments are reconstructed with guidance from the output of the attacked model, ensuring subtle and effective adversarial perturbations. We evaluate MAIA on multiple MIR tasks, demonstrating high attack success rates in both white-box and black-box settings while maintaining minimal perceptual distortion. Additionally, subjective listening tests confirm the high audio fidelity of the adversarial samples. Our findings highlight vulnerabilities in current MIR systems and emphasize the need for more robust and secure models.
>
---
#### [new 006] Learning and composing of classical music using restricted Boltzmann machines
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文利用受限玻尔兹曼机学习巴赫音乐风格，解决复杂模型难以分析的问题，通过简单结构探究内部状态，实现音乐生成。**

- **链接: [http://arxiv.org/pdf/2509.04899v1](http://arxiv.org/pdf/2509.04899v1)**

> **作者:** Mutsumi Kobayashi; Hiroshi Watanabe
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Recently, software has been developed that uses machine learning to mimic the style of a particular composer, such as J. S. Bach. However, since such software often adopts machine learning models with complex structures, it is difficult to analyze how the software understands the characteristics of the composer's music. In this study, we adopted J. S. Bach's music for training of a restricted Boltzmann machine (RBM). Since the structure of RBMs is simple, it allows us to investigate the internal states after learning. We found that the learned RBM is able to compose music.
>
---
#### [new 007] Recomposer: Event-roll-guided generative audio editing
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 论文提出Recomposer系统，通过文本指令和事件时间图谱，利用生成模型编辑复杂音频场景中的单个声音事件，解决重叠声源的编辑问题，支持删除、插入和增强操作，并评估描述要素的重要性。**

- **链接: [http://arxiv.org/pdf/2509.05256v1](http://arxiv.org/pdf/2509.05256v1)**

> **作者:** Daniel P. W. Ellis; Eduardo Fonseca; Ron J. Weiss; Kevin Wilson; Scott Wisdom; Hakan Erdogan; John R. Hershey; Aren Jansen; R. Channing Moore; Manoj Plakal
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Editing complex real-world sound scenes is difficult because individual sound sources overlap in time. Generative models can fill-in missing or corrupted details based on their strong prior understanding of the data domain. We present a system for editing individual sound events within complex scenes able to delete, insert, and enhance individual sound events based on textual edit descriptions (e.g., ``enhance Door'') and a graphical representation of the event timing derived from an ``event roll'' transcription. We present an encoder-decoder transformer working on SoundStream representations, trained on synthetic (input, desired output) audio example pairs formed by adding isolated sound events to dense, real-world backgrounds. Evaluation reveals the importance of each part of the edit descriptions -- action, class, timing. Our work demonstrates ``recomposition'' is an important and practical application.
>
---
#### [new 008] WildScore: Benchmarking MLLMs in-the-Wild Symbolic Music Reasoning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出WildScore基准，评估多模态大语言模型（MLLMs）在符号音乐推理任务中的能力，解决其在音乐领域推理能力不足的问题，通过真实音乐数据与多选题框架进行系统评测。**

- **链接: [http://arxiv.org/pdf/2509.04744v1](http://arxiv.org/pdf/2509.04744v1)**

> **作者:** Gagan Mundada; Yash Vishe; Amit Namburi; Xin Xu; Zachary Novack; Julian McAuley; Junda Wu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities across various vision-language tasks. However, their reasoning abilities in the multimodal symbolic music domain remain largely unexplored. We introduce WildScore, the first in-the-wild multimodal symbolic music reasoning and analysis benchmark, designed to evaluate MLLMs' capacity to interpret real-world music scores and answer complex musicological queries. Each instance in WildScore is sourced from genuine musical compositions and accompanied by authentic user-generated questions and discussions, capturing the intricacies of practical music analysis. To facilitate systematic evaluation, we propose a systematic taxonomy, comprising both high-level and fine-grained musicological ontologies. Furthermore, we frame complex music reasoning as multiple-choice question answering, enabling controlled and scalable assessment of MLLMs' symbolic music understanding. Empirical benchmarking of state-of-the-art MLLMs on WildScore reveals intriguing patterns in their visual-symbolic reasoning, uncovering both promising directions and persistent challenges for MLLMs in symbolic music reasoning and analysis. We release the dataset and code.
>
---
#### [new 009] Inferring the Graph Structure of Images for Graph Neural Networks
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文旨在改进图像的图结构表示，以提升图神经网络的分类性能。通过构建基于像素相关性的行、列和乘积图，替代传统网格图和超像素方法，实验表明新方法在MNIST和Fashion-MNIST数据集上提高了准确性。**

- **链接: [http://arxiv.org/pdf/2509.04677v1](http://arxiv.org/pdf/2509.04677v1)**

> **作者:** Mayur S Gowda; John Shi; Augusto Santos; José M. F. Moura
>
> **摘要:** Image datasets such as MNIST are a key benchmark for testing Graph Neural Network (GNN) architectures. The images are traditionally represented as a grid graph with each node representing a pixel and edges connecting neighboring pixels (vertically and horizontally). The graph signal is the values (intensities) of each pixel in the image. The graphs are commonly used as input to graph neural networks (e.g., Graph Convolutional Neural Networks (Graph CNNs) [1, 2], Graph Attention Networks (GAT) [3], GatedGCN [4]) to classify the images. In this work, we improve the accuracy of downstream graph neural network tasks by finding alternative graphs to the grid graph and superpixel methods to represent the dataset images, following the approach in [5, 6]. We find row correlation, column correlation, and product graphs for each image in MNIST and Fashion-MNIST using correlations between the pixel values building on the method in [5, 6]. Experiments show that using these different graph representations and features as input into downstream GNN models improves the accuracy over using the traditional grid graph and superpixel methods in the literature.
>
---
#### [new 010] Layer-wise Analysis for Quality of Multilingual Synthesized Speech
- **分类: eess.AS; cs.SD**

- **简介: 论文针对多语言合成语音质量分析，探讨无监督方法中SSL和ASR模型的层编码机制，通过参考建模发现早期SSL层与人类评分相关，后期ASR层可预测非神经系统质量及可懂度，并强调参考数据匹配的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04830v1](http://arxiv.org/pdf/2509.04830v1)**

> **作者:** Erica Cooper; Takuma Okamoto; Yamato Ohtani; Tomoki Toda; Hisashi Kawai
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** While supervised quality predictors for synthesized speech have demonstrated strong correlations with human ratings, their requirement for in-domain labeled training data hinders their generalization ability to new domains. Unsupervised approaches based on pretrained self-supervised learning (SSL) based models and automatic speech recognition (ASR) models are a promising alternative; however, little is known about how these models encode information about speech quality. Towards the goal of better understanding how different aspects of speech quality are encoded in a multilingual setting, we present a layer-wise analysis of multilingual pretrained speech models based on reference modeling. We find that features extracted from early SSL layers show correlations with human ratings of synthesized speech, and later layers of ASR models can predict quality of non-neural systems as well as intelligibility. We also demonstrate the importance of using well-matched reference data.
>
---
#### [new 011] Lightweight DNN for Full-Band Speech Denoising on Mobile Devices: Exploiting Long and Short Temporal Patterns
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 本论文提出轻量级DNN，用于移动设备全带语音降噪，解决资源受限及低延迟问题。方法结合长短时序模式，采用修改UNet架构与因果实例归一化，实现低延迟高效率。**

- **链接: [http://arxiv.org/pdf/2509.05079v1](http://arxiv.org/pdf/2509.05079v1)**

> **作者:** Konstantinos Drossos; Mikko Heikkinen; Paschalis Tsiaflakis
>
> **备注:** Accepted for publication in Proceedings of the 2025 IEEE 27th International Workshop on Multimedia Signal Processing (MMSP)
>
> **摘要:** Speech denoising (SD) is an important task of many, if not all, modern signal processing chains used in devices and for everyday-life applications. While there are many published and powerful deep neural network (DNN)-based methods for SD, few are optimized for resource-constrained platforms such as mobile devices. Additionally, most DNN-based methods for SD are not focusing on full-band (FB) signals, i.e. having 48 kHz sampling rate, and/or low latency cases. In this paper we present a causal, low latency, and lightweight DNN-based method for full-band SD, leveraging both short and long temporal patterns. The method is based on a modified UNet architecture employing look-back frames, temporal spanning of convolutional kernels, and recurrent neural networks for exploiting short and long temporal patterns in the signal and estimated denoising mask. The DNN operates on a causal frame-by-frame basis taking as an input the STFT magnitude, utilizes inverted bottlenecks inspired by MobileNet, employs causal instance normalization for channel-wise normalization, and achieves a real-time factor below 0.02 when deployed on a modern mobile phone. The proposed method is evaluated using established speech denoising metrics and publicly available datasets, demonstrating its effectiveness in achieving an (SI-)SDR value that outperforms existing FB and low latency SD methods.
>
---
#### [new 012] Exploring Situated Stabilities of a Rhythm Generation System through Variational Cross-Examination
- **分类: cs.HC; cs.AI; cs.SD; eess.AS**

- **简介: 论文通过变分交叉检验（VCE）方法研究节奏生成系统GrooveTransformer的多稳定性，分析其在不同艺术情境中的应用，探讨系统不变量、跨学科合作和处境性的影响，评估VCE作为数字音乐乐器设计方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.05145v1](http://arxiv.org/pdf/2509.05145v1)**

> **作者:** Błażej Kotowski; Nicholas Evans; Behzad Haki; Frederic Font; Sergi Jordà
>
> **备注:** AI Music Creativity 2025
>
> **摘要:** This paper investigates GrooveTransformer, a real-time rhythm generation system, through the postphenomenological framework of Variational Cross-Examination (VCE). By reflecting on its deployment across three distinct artistic contexts, we identify three stabilities: an autonomous drum accompaniment generator, a rhythmic control voltage sequencer in Eurorack format, and a rhythm driver for a harmonic accompaniment system. The versatility of its applications was not an explicit goal from the outset of the project. Thus, we ask: how did this multistability emerge? Through VCE, we identify three key contributors to its emergence: the affordances of system invariants, the interdisciplinary collaboration, and the situated nature of its development. We conclude by reflecting on the viability of VCE as a descriptive and analytical method for Digital Musical Instrument (DMI) design, emphasizing its value in uncovering how technologies mediate, co-shape, and are co-shaped by users and contexts.
>
---
#### [new 013] Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 论文提出基于序列化输出提示（SOP）的多说话人语音识别方法，解决LLM在复杂多说话人场景下性能不足的问题。通过分离器与CTC层提取序列化内容，结合三阶段训练策略，提升LLM对混合语音的识别效果。**

- **链接: [http://arxiv.org/pdf/2509.04488v1](http://arxiv.org/pdf/2509.04488v1)**

> **作者:** Hao Shi; Yusuke Fujita; Tomoya Mizumoto; Lianbo Liu; Atsushi Kojima; Yui Sudo
>
> **摘要:** Prompts are crucial for task definition and for improving the performance of large language models (LLM)-based systems. However, existing LLM-based multi-talker (MT) automatic speech recognition (ASR) systems either omit prompts or rely on simple task-definition prompts, with no prior work exploring the design of prompts to enhance performance. In this paper, we propose extracting serialized output prompts (SOP) and explicitly guiding the LLM using structured prompts to improve system performance (SOP-MT-ASR). A Separator and serialized Connectionist Temporal Classification (CTC) layers are inserted after the speech encoder to separate and extract MT content from the mixed speech encoding in a first-speaking-first-out manner. Subsequently, the SOP, which serves as a prompt for LLMs, is obtained by decoding the serialized CTC outputs using greedy search. To train the model effectively, we design a three-stage training strategy, consisting of serialized output training (SOT) fine-tuning, serialized speech information extraction, and SOP-based adaptation. Experimental results on the LibriMix dataset show that, although the LLM-based SOT model performs well in the two-talker scenario, it fails to fully leverage LLMs under more complex conditions, such as the three-talker scenario. The proposed SOP approach significantly improved performance under both two- and three-talker conditions.
>
---
#### [new 014] On Time Delay Interpolation for Improved Acoustic Reflector Localization
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对声学反射体定位中传统TDE时间分辨率不足的问题，提出基于sinc和Whittaker-Shannon插值的方法，提升混响环境下的定位精度。**

- **链接: [http://arxiv.org/pdf/2509.04629v1](http://arxiv.org/pdf/2509.04629v1)**

> **作者:** Hannes Rosseel; Toon van Waterschoot
>
> **备注:** 20 pages, 13 figures, 2 tables, submitted to J. Acoust. Soc. Am
>
> **摘要:** The localization of acoustic reflectors is a fundamental component in various applications, including room acoustics analysis, sound source localization, and acoustic scene analysis. Time Delay Estimation (TDE) is essential for determining the position of reflectors relative to a sensor array. Traditional TDE algorithms generally yield time delays that are integer multiples of the operating sampling period, potentially lacking sufficient time resolution. To achieve subsample TDE accuracy, various interpolation methods, including parabolic, Gaussian, frequency, and sinc interpolation, have been proposed. This paper presents a comprehensive study on time delay interpolation to achieve subsample accuracy for acoustic reflector localization in reverberant conditions. We derive the Whittaker-Shannon interpolation formula from the previously proposed sinc interpolation in the context of short-time windowed TDE for acoustic reflector localization. Simulations show that sinc and Whittaker-Shannon interpolation outperform existing methods in terms of time delay error and positional error for critically sampled and band-limited reflections. Performance is evaluated on real-world measurements from the MYRiAD dataset, showing that sinc and Whittaker-Shannon interpolation consistently provide reliable performance across different sensor-source pairs and loudspeaker positions. These results can enhance the precision of acoustic reflector localization systems, vital for applications such as room acoustics analysis, sound source localization, and acoustic scene analysis.
>
---
#### [new 015] Efficient Video-to-Audio Generation via Multiple Foundation Models Mapper
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 本文提出MFM-Mapper，通过融合双视觉编码器与GPT-2，解决V2A生成中训练成本高的问题，实现高效且性能优越的音频生成。**

- **链接: [http://arxiv.org/pdf/2509.04957v1](http://arxiv.org/pdf/2509.04957v1)**

> **作者:** Gehui Chen; Guan'an Wang; Xiaowen Huang; Jitao Sang
>
> **摘要:** Recent Video-to-Audio (V2A) generation relies on extracting semantic and temporal features from video to condition generative models. Training these models from scratch is resource intensive. Consequently, leveraging foundation models (FMs) has gained traction due to their cross-modal knowledge transfer and generalization capabilities. One prior work has explored fine-tuning a lightweight mapper network to connect a pre-trained visual encoder with a text-to-audio generation model for V2A. Inspired by this, we introduce the Multiple Foundation Model Mapper (MFM-Mapper). Compared to the previous mapper approach, MFM-Mapper benefits from richer semantic and temporal information by fusing features from dual visual encoders. Furthermore, by replacing a linear mapper with GPT-2, MFM-Mapper improves feature alignment, drawing parallels between cross-modal features mapping and autoregressive translation tasks. Our MFM-Mapper exhibits remarkable training efficiency. It achieves better performance in semantic and temporal consistency with fewer training consuming, requiring only 16\% of the training scale compared to previous mapper-based work, yet achieves competitive performance with models trained on a much larger scale.
>
---
#### [new 016] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出VARSTok，一种动态语音分词器，通过自适应聚类和隐式时长编码，解决固定帧率分词导致的信息密度不匹配问题，实现更高效、自然的语音表示。**

- **链接: [http://arxiv.org/pdf/2509.04685v1](http://arxiv.org/pdf/2509.04685v1)**

> **作者:** Rui-Chen Zheng; Wenrui Liu; Hui-Peng Du; Qinglin Zhang; Chong Deng; Qian Chen; Wen Wang; Yang Ai; Zhen-Hua Ling
>
> **摘要:** Existing speech tokenizers typically assign a fixed number of tokens per second, regardless of the varying information density or temporal fluctuations in the speech signal. This uniform token allocation mismatches the intrinsic structure of speech, where information is distributed unevenly over time. To address this, we propose VARSTok, a VAriable-frame-Rate Speech Tokenizer that adapts token allocation based on local feature similarity. VARSTok introduces two key innovations: (1) a temporal-aware density peak clustering algorithm that adaptively segments speech into variable-length units, and (2) a novel implicit duration coding scheme that embeds both content and temporal span into a single token index, eliminating the need for auxiliary duration predictors. Extensive experiments show that VARSTok significantly outperforms strong fixed-rate baselines. Notably, it achieves superior reconstruction naturalness while using up to 23% fewer tokens than a 40 Hz fixed-frame-rate baseline. VARSTok further yields lower word error rates and improved naturalness in zero-shot text-to-speech synthesis. To the best of our knowledge, this is the first work to demonstrate that a fully dynamic, variable-frame-rate acoustic speech tokenizer can be seamlessly integrated into downstream speech language models. Speech samples are available at https://zhengrachel.github.io/VARSTok.
>
---
#### [new 017] MEAN-RIR: Multi-Modal Environment-Aware Network for Robust Room Impulse Response Estimation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出MEAN-RIR模型，解决多模态环境信息下的鲁棒房间脉冲响应（RIR）估计问题。通过编码器-解码器框架融合音频、视觉与文本输入，利用交叉注意力模块交互多模态信息，分离生成直接声/早期反射与后期混响，提升RIR估计精度。**

- **链接: [http://arxiv.org/pdf/2509.05205v1](http://arxiv.org/pdf/2509.05205v1)**

> **作者:** Jiajian Chen; Jiakang Chen; Hang Chen; Qing Wang; Yu Gao; Jun Du
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** This paper presents a Multi-Modal Environment-Aware Network (MEAN-RIR), which uses an encoder-decoder framework to predict room impulse response (RIR) based on multi-level environmental information from audio, visual, and textual sources. Specifically, reverberant speech capturing room acoustic properties serves as the primary input, which is combined with panoramic images and text descriptions as supplementary inputs. Each input is processed by its respective encoder, and the outputs are fed into cross-attention modules to enable effective interaction between different modalities. The MEAN-RIR decoder generates two distinct components: the first component captures the direct sound and early reflections, while the second produces masks that modulate learnable filtered noise to synthesize the late reverberation. These two components are mixed to reconstruct the final RIR. The results show that MEAN-RIR significantly improves RIR estimation, with notable gains in acoustic parameters.
>
---
## 更新

#### [replaced 001] WenetSpeech-Yue: A Large-scale Cantonese Speech Corpus with Multi-dimensional Annotation
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2509.03959v2](http://arxiv.org/pdf/2509.03959v2)**

> **作者:** Longhao Li; Zhao Guo; Hongjie Chen; Yuhang Dai; Ziyu Zhang; Hongfei Xue; Tianlun Zuo; Chengyou Wang; Shuiyuan Wang; Jie Li; Jian Kang; Xin Xu; Hui Bu; Binbin Zhang; Ruibin Yuan; Ziya Zhou; Wei Xue; Lei Xie
>
> **摘要:** The development of speech understanding and generation has been significantly accelerated by the availability of large-scale, high-quality speech datasets. Among these, ASR and TTS are regarded as the most established and fundamental tasks. However, for Cantonese (Yue Chinese), spoken by approximately 84.9 million native speakers worldwide, limited annotated resources have hindered progress and resulted in suboptimal ASR and TTS performance. To address this challenge, we propose WenetSpeech-Pipe, an integrated pipeline for building large-scale speech corpus with multi-dimensional annotation tailored for speech understanding and generation. It comprises six modules: Audio Collection, Speaker Attributes Annotation, Speech Quality Annotation, Automatic Speech Recognition, Text Postprocessing and Recognizer Output Voting, enabling rich and high-quality annotations. Based on this pipeline, we release WenetSpeech-Yue, the first large-scale Cantonese speech corpus with multi-dimensional annotation for ASR and TTS, covering 21,800 hours across 10 domains with annotations including ASR transcription, text confidence, speaker identity, age, gender, speech quality scores, among other annotations. We also release WSYue-eval, a comprehensive Cantonese benchmark with two components: WSYue-ASR-eval, a manually annotated set for evaluating ASR on short and long utterances, code-switching, and diverse acoustic conditions, and WSYue-TTS-eval, with base and coverage subsets for standard and generalization testing. Experimental results show that models trained on WenetSpeech-Yue achieve competitive results against state-of-the-art (SOTA) Cantonese ASR and TTS systems, including commercial and LLM-based models, highlighting the value of our dataset and pipeline.
>
---
#### [replaced 002] Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.15442v3](http://arxiv.org/pdf/2508.15442v3)**

> **作者:** Chenlin Liu; Minghui Fang; Patrick Zhang; Wei Zhou; Jie Gao; Jiqing Han
>
> **备注:** Accepted to EMNLP 2025 Main Conference (Oral)
>
> **摘要:** Language Model (LM)-based Text-to-Speech (TTS) systems often generate hallucinated speech that deviates from input text. Existing mitigation strategies either demand excessive training resources or introduce significant inference latency. In this paper, we propose GFlOwNet-guided distribution AlignmenT (GOAT) for LM-based TTS, a post-training framework that mitigates hallucinations without relying on massive resources or inference cost. Specifically, we first conduct an uncertainty analysis, revealing a strong positive correlation between hallucination and model uncertainty. Based on this, we reformulate TTS generation as a trajectory flow optimization problem and introduce an enhanced Subtrajectory Balance objective together with a sharpened internal reward as target distribution. We further integrate reward temperature decay and learning rate optimization for stability and performance balance. Extensive experiments show that GOAT reduce over 50% character error rates on challenging test cases and lowering uncertainty by up to 58%, demonstrating its strong generalization ability and effectiveness.
>
---
