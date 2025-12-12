# 音频 cs.SD;  eess.AS

- **最新发布 8 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] MR-FlowDPO: Multi-Reward Direct Preference Optimization for Flow-Matching Text-to-Music Generation
- **分类: cs.SD**

- **简介: 该论文属于文本到音乐生成任务，旨在解决生成音乐与人类主观偏好对齐困难的问题。作者提出MR-FlowDPO方法，结合多奖励直接偏好优化与流程匹配模型，通过文本对齐、音质和语义一致性三个维度提升生成质量，并引入自监督表示增强节奏稳定性。**

- **链接: [https://arxiv.org/pdf/2512.10264v1](https://arxiv.org/pdf/2512.10264v1)**

> **作者:** Alon Ziv; Sanyuan Chen; Andros Tjandra; Yossi Adi; Wei-Ning Hsu; Bowen Shi
>
> **摘要:** A key challenge in music generation models is their lack of direct alignment with human preferences, as music evaluation is inherently subjective and varies widely across individuals. We introduce MR-FlowDPO, a novel approach that enhances flow-matching-based music generation models - a major class of modern music generative models, using Direct Preference Optimization (DPO) with multiple musical rewards. The rewards are crafted to assess music quality across three key dimensions: text alignment, audio production quality, and semantic consistency, utilizing scalable off-the-shelf models for each reward prediction. We employ these rewards in two ways: (i) By constructing preference data for DPO and (ii) by integrating the rewards into text prompting. To address the ambiguity in musicality evaluation, we propose a novel scoring mechanism leveraging semantic self-supervised representations, which significantly improves the rhythmic stability of generated music. We conduct an extensive evaluation using a variety of music-specific objective metrics as well as a human study. Results show that MR-FlowDPO significantly enhances overall music generation quality and is consistently preferred over highly competitive baselines in terms of audio quality, text alignment, and musicality. Our code is publicly available at https://github.com/lonzi/mrflow_dpo; Samples are provided in our demo page at https://lonzi.github.io/mr_flowdpo_demopage/.
>
---
#### [new 002] Semantic-Aware Confidence Calibration for Automated Audio Captioning
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对自动音频描述生成中模型过度自信且缺乏语义准确性的置信度校准问题，提出一种语义感知的置信度校准框架。通过引入基于CLAP和句向量的语义相似性度量，结合置信度预测头，实现更可靠的校准与更优的生成质量。**

- **链接: [https://arxiv.org/pdf/2512.10170v1](https://arxiv.org/pdf/2512.10170v1)**

> **作者:** Lucas Dunker; Sai Akshay Menta; Snigdha Mohana Addepalli; Venkata Krishna Rayalu Garapati
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Automated audio captioning models frequently produce overconfident predictions regardless of semantic accuracy, limiting their reliability in deployment. This deficiency stems from two factors: evaluation metrics based on n-gram overlap that fail to capture semantic correctness, and the absence of calibrated confidence estimation. We present a framework that addresses both limitations by integrating confidence prediction into audio captioning and redefining correctness through semantic similarity. Our approach augments a Whisper-based audio captioning model with a learned confidence prediction head that estimates uncertainty from decoder hidden states. We employ CLAP audio-text embeddings and sentence transformer similarities (FENSE) to define semantic correctness, enabling Expected Calibration Error (ECE) computation that reflects true caption quality rather than surface-level text overlap. Experiments on Clotho v2 demonstrate that confidence-guided beam search with semantic evaluation achieves dramatically improved calibration (CLAP-based ECE of 0.071) compared to greedy decoding baselines (ECE of 0.488), while simultaneously improving caption quality across standard metrics. Our results establish that semantic similarity provides a more meaningful foundation for confidence calibration in audio captioning than traditional n-gram metrics.
>
---
#### [new 003] BRACE: A Benchmark for Robust Audio Caption Quality Evaluation
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦音频描述质量评估任务，旨在解决无参考文本时的评估难题。作者提出BRACE基准，包含两个子任务，用于评测音频描述对齐与幻觉检测，通过实验揭示现有CLAPScore与大模型的局限性。**

- **链接: [https://arxiv.org/pdf/2512.10403v1](https://arxiv.org/pdf/2512.10403v1)**

> **作者:** Tianyu Guo; Hongyu Chen; Hao Liang; Meiyi Qiang; Bohan Zeng; Linzhuang Sun; Bin Cui; Wentao Zhang
>
> **摘要:** Automatic audio captioning is essential for audio understanding, enabling applications such as accessibility and content indexing. However, evaluating the quality of audio captions remains a major challenge, especially in reference-free settings where high-quality ground-truth captions are unavailable. While CLAPScore is currently the most widely used reference-free Audio Caption Evaluation Metric(ACEM), its robustness under diverse conditions has not been systematically validated. To address this gap, we introduce BRACE, a new benchmark designed to evaluate audio caption alignment quality in a reference-free setting. BRACE is primarily designed for assessing ACEMs, and can also be extended to measure the modality alignment abilities of Large Audio Language Model(LALM). BRACE consists of two sub-benchmarks: BRACE-Main for fine-grained caption comparison and BRACE-Hallucination for detecting subtle hallucinated content. We construct these datasets through high-quality filtering, LLM-based corruption, and human annotation. Given the widespread adoption of CLAPScore as a reference-free ACEM and the increasing application of LALMs in audio-language tasks, we evaluate both approaches using the BRACE benchmark, testing CLAPScore across various CLAP model variants and assessing multiple LALMs. Notably, even the best-performing CLAP-based ACEM achieves only a 70.01 F1-score on the BRACE-Main benchmark, while the best LALM reaches just 63.19. By revealing the limitations of CLAP models and LALMs, our BRACE benchmark offers valuable insights into the direction of future research.
>
---
#### [new 004] Exploring Perceptual Audio Quality Measurement on Stereo Processing Using the Open Dataset of Audio Quality
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频质量评估任务，旨在探究立体声处理对感知音频质量的影响。基于ODAQ数据集，研究了现有客观指标在复杂听觉场景下的局限性，强调需结合音色与空间因素及自下而上与自上而下的心理声学机制以提升预测性能。**

- **链接: [https://arxiv.org/pdf/2512.10689v1](https://arxiv.org/pdf/2512.10689v1)**

> **作者:** Pablo M. Delgado; Sascha Dick; Christoph Thompson; Chih-Wei Wu; Phillip A. Williams
>
> **备注:** Presented at the 159 Audio Engineering Society Convention. Paper Number:366. https://aes2.org/publications/elibrary-page/?id=23040
>
> **摘要:** ODAQ (Open Dataset of Audio Quality) provides a comprehensive framework for exploring both monaural and binaural audio quality degradations across a range of distortion classes and signals, accompanied by subjective quality ratings. A recent update of ODAQ, focusing on the impact of stereo processing methods such as Mid/Side (MS) and Left/Right (LR), provides test signals and subjective ratings for the in-depth investigation of state-of-the-art objective audio quality metrics. Our evaluation results suggest that, while timbre-focused metrics often yield robust results under simpler conditions, their prediction performance tends to suffer under the conditions with a more complex presentation context. Our findings underscore the importance of modeling the interplay of bottom-up psychoacoustic processes and top-down contextual factors, guiding future research toward models that more effectively integrate both timbral and spatial dimensions of perceived audio quality.
>
---
#### [new 005] VocSim: A Training-free Benchmark for Zero-shot Content Identity in Single-source Audio
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出VocSim，一个无需训练的零样本音频内容身份识别基准，旨在评估音频表示的内在几何对齐性。通过单源音频构建数据集，衡量模型在语音、动物叫声等上的零样本泛化能力，揭示现有模型在低资源语音上几何结构崩溃的问题。**

- **链接: [https://arxiv.org/pdf/2512.10120v1](https://arxiv.org/pdf/2512.10120v1)**

> **作者:** Maris Basha; Anja Zai; Sabine Stoll; Richard Hahnloser
>
> **摘要:** General-purpose audio representations aim to map acoustically variable instances of the same event to nearby points, resolving content identity in a zero-shot setting. Unlike supervised classification benchmarks that measure adaptability via parameter updates, we introduce VocSim, a training-free benchmark probing the intrinsic geometric alignment of frozen embeddings. VocSim aggregates 125k single-source clips from 19 corpora spanning human speech, animal vocalizations, and environmental sounds. By restricting to single-source audio, we isolate content representation from the confound of source separation. We evaluate embeddings using Precision@k for local purity and the Global Separation Rate (GSR) for point-wise class separation. To calibrate GSR, we report lift over an empirical permutation baseline. Across diverse foundation models, a simple pipeline, frozen Whisper encoder features, time-frequency pooling, and label-free PCA, yields strong zero-shot performance. However, VocSim also uncovers a consistent generalization gap. On blind, low-resource speech, local retrieval drops sharply. While performance remains statistically distinguishable from chance, the absolute geometric structure collapses, indicating a failure to generalize to unseen phonotactics. As external validation, our top embeddings predict avian perceptual similarity, improve bioacoustic classification, and achieve state-of-the-art results on the HEAR benchmark. We posit that the intrinsic geometric quality measured here proxies utility in unlisted downstream applications. We release data, code, and a public leaderboard to standardize the evaluation of intrinsic audio geometry.
>
---
#### [new 006] Neural personal sound zones with flexible bright zone control
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究个人声区（PSZ）生成，旨在解决传统方法依赖固定麦克风阵列、成本高的问题。提出基于3D CNN的方法，实现灵活控制点布局与多目标再现，仅需一次训练即可适应变化的声场目标。**

- **链接: [https://arxiv.org/pdf/2512.10375v1](https://arxiv.org/pdf/2512.10375v1)**

> **作者:** Wenye Zhu; Jun Tang; Xiaofei Li
>
> **摘要:** Personal sound zone (PSZ) reproduction system, which attempts to create distinct virtual acoustic scenes for different listeners at their respective positions within the same spatial area using one loudspeaker array, is a fundamental technology in the application of virtual reality. For practical applications, the reconstruction targets must be measured on the same fixed receiver array used to record the local room impulse responses (RIRs) from the loudspeaker array to the control points in each PSZ, which makes the system inconvenient and costly for real-world use. In this paper, a 3D convolutional neural network (CNN) designed for PSZ reproduction with flexible control microphone grid and alternative reproduction target is presented, utilizing the virtual target scene as inputs and the PSZ pre-filters as output. Experimental results of the proposed method are compared with the traditional method, demonstrating that the proposed method is able to handle varied reproduction targets on flexible control point grid using only one training session. Furthermore, the proposed method also demonstrates the capability to learn global spatial information from sparse sampling points distributed in PSZs.
>
---
#### [new 007] Building Audio-Visual Digital Twins with Smartphones
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文致力于构建可编辑的音视频数字孪生系统。针对现有数字孪生缺乏声学建模的问题，提出AV-Twin，利用智能手机实现房间脉冲响应采集与视觉辅助声学场建模，恢复表面材质属性，支持几何、布局与材料的编辑，并同步更新音视频。**

- **链接: [https://arxiv.org/pdf/2512.10778v1](https://arxiv.org/pdf/2512.10778v1)**

> **作者:** Zitong Lan; Yiwei Tang; Yuhan Wang; Haowen Lai; Yido Hao; Mingmin Zhao
>
> **备注:** Under Mobisys 2026 review, single blind
>
> **摘要:** Digital twins today are almost entirely visual, overlooking acoustics-a core component of spatial realism and interaction. We introduce AV-Twin, the first practical system that constructs editable audio-visual digital twins using only commodity smartphones. AV-Twin combines mobile RIR capture and a visual-assisted acoustic field model to efficiently reconstruct room acoustics. It further recovers per-surface material properties through differentiable acoustic rendering, enabling users to modify materials, geometry, and layout while automatically updating both audio and visuals. Together, these capabilities establish a practical path toward fully modifiable audio-visual digital twins for real-world environments.
>
---
#### [new 008] Investigating training objective for flow matching-based speech enhancement
- **分类: cs.SD**

- **简介: 该论文研究基于流匹配的语音增强，旨在提升效率与性能。作者系统比较三种训练目标，并引入感知与信号损失，优化训练过程，显著提高语音质量和收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.10382v1](https://arxiv.org/pdf/2512.10382v1)**

> **作者:** Liusha Yang; Ziru Ge; Gui Zhang; Junan Zhang; Zhizheng Wu
>
> **摘要:** Speech enhancement(SE) aims to recover clean speech from noisy recordings. Although generative approaches such as score matching and Schrodinger bridge have shown strong effectiveness, they are often computationally expensive. Flow matching offers a more efficient alternative by directly learning a velocity field that maps noise to data. In this work, we present a systematic study of flow matching for SE under three training objectives: velocity prediction, $x_1$ prediction, and preconditioned $x_1$ prediction. We analyze their impact on training dynamics and overall performance. Moreover, by introducing perceptual(PESQ) and signal-based(SI-SDR) objectives, we further enhance convergence efficiency and speech quality, yielding substantial improvements across evaluation metrics.
>
---
## 更新

#### [replaced 001] Towards Robust Assessment of Pathological Voices via Combined Low-Level Descriptors and Foundation Model Representations
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究病理性语音评估任务，旨在解决传统主观评价方法可靠性低的问题。作者提出VOQANet及VOQANet+模型，结合语音基础模型与低层声学特征，提升客观评估的准确性和鲁棒性，适用于真实场景与远程医疗。**

- **链接: [https://arxiv.org/pdf/2505.21356v4](https://arxiv.org/pdf/2505.21356v4)**

> **作者:** Whenty Ariyanti; Kuan-Yu Chen; Sabato Marco Siniscalchi; Hsin-Min Wang; Yu Tsao
>
> **摘要:** Perceptual voice quality assessment plays a vital role in diagnosing and monitoring voice disorders. Traditional methods, such as the Consensus Auditory-Perceptual Evaluation of Voice (CAPE-V) and the Grade, Roughness, Breathiness, Asthenia, and Strain (GRBAS) scales, rely on expert raters and are prone to inter-rater variability, emphasizing the need for objective solutions. This study introduces the Voice Quality Assessment Network (VOQANet), a deep learning framework that employs an attention mechanism and Speech Foundation Model (SFM) embeddings to extract high-level features. To further enhance performance, we propose VOQANet+, which integrates self-supervised SFM embeddings with low-level acoustic descriptors-namely jitter, shimmer, and harmonics-to-noise ratio (HNR). Unlike previous approaches that focus solely on vowel-based phonation (PVQD-A), our models are evaluated on both vowel-level and sentence-level speech (PVQD-S) to assess generalizability. Experimental results demonstrate that sentence-based inputs yield higher accuracy, particularly at the patient level. Overall, VOQANet consistently outperforms baseline models in terms of root mean squared error (RMSE) and Pearson correlation coefficient across CAPE-V and GRBAS dimensions, with VOQANet+ achieving even greater performance gains. Additionally, VOQANet+ maintains consistent performance under noisy conditions, suggesting enhanced robustness for real-world and telehealth applications. This work highlights the value of combining SFM embeddings with low-level features for accurate and robust pathological voice assessment.
>
---
#### [replaced 002] Forensic deepfake audio detection using segmental speech features
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究深伪音频检测，提出利用可解释性强的音段语音特征（与发音机制相关）进行识别。工作包括验证此类特征在区分真/假音频中的有效性，并提出面向个体说话人的检测框架，以提升法医场景下的可解释性与敏感性。**

- **链接: [https://arxiv.org/pdf/2505.13847v3](https://arxiv.org/pdf/2505.13847v3)**

> **作者:** Tianle Yang; Chengzhe Sun; Siwei Lyu; Phil Rose
>
> **备注:** Accepted for publication in Forensic Science International
>
> **摘要:** This study explores the potential of using acoustic features of segmental speech sounds to detect deepfake audio. These features are highly interpretable because of their close relationship with human articulatory processes and are expected to be more difficult for deepfake models to replicate. The results demonstrate that certain segmental features commonly used in forensic voice comparison (FVC) are effective in identifying deep-fakes, whereas some global features provide little value. These findings underscore the need to approach audio deepfake detection using methods that are distinct from those employed in traditional FVC, and offer a new perspective on leveraging segmental features for this purpose. In addition, the present study proposes a speaker-specific framework for deepfake detection, which differs fundamentally from the speaker-independent systems that dominate current benchmarks. While speaker-independent frameworks aim at broad generalization, the speaker-specific approach offers advantages in forensic contexts where case-by-case interpretability and sensitivity to individual phonetic realization are essential.
>
---
#### [replaced 003] Lightweight Model Attribution and Detection of Synthetic Speech via Audio Residual Fingerprints
- **分类: eess.AS; cs.CR; cs.LG**

- **简介: 该论文研究合成语音的检测与溯源，提出一种无需训练的轻量方法，通过音频残差指纹识别生成模型并区分真假语音，在多场景下均表现优异，适用于数字取证与安全。**

- **链接: [https://arxiv.org/pdf/2411.14013v4](https://arxiv.org/pdf/2411.14013v4)**

> **作者:** Matías Pizarro; Mike Laszkiewicz; Dorothea Kolossa; Asja Fischer
>
> **摘要:** As speech generation technologies advance, so do risks of impersonation, misinformation, and spoofing. We present a lightweight, training-free approach for detecting synthetic speech and attributing it to its source model. Our method addresses three tasks: (1) single-model attribution in an open-world setting, (2) multi-model attribution in a closed-world setting, and (3) real vs. synthetic speech classification. The core idea is simple: we compute standardized average residuals--the difference between an audio signal and its filtered version--to extract model-agnostic fingerprints that capture synthesis artifacts. Experiments across multiple synthesis systems and languages show AUROC scores above 99%, with strong reliability even when only a subset of model outputs is available. The method maintains high performance under common audio distortions, including echo and moderate background noise, while data augmentation can improve results in more challenging conditions. In addition, out-of-domain detection is performed using Mahalanobis distances to in-domain residual fingerprints, achieving an F1 score of 0.91 on unseen models, reinforcing the method's efficiency, generalizability, and suitability for digital forensics and security applications.
>
---
#### [replaced 004] A Low-Complexity Speech Codec Using Parametric Dithering for ASR
- **分类: eess.AS**

- **简介: 该论文研究语音压缩对ASR的影响，提出一种基于参数化抖动的低复杂度语音编解码方法。通过引入抖动技术提升压缩后语音的ASR识别性能，在1-3比特量化下显著降低词错误率，并支持适应性调整以满足性能或熵约束需求。**

- **链接: [https://arxiv.org/pdf/2512.00511v3](https://arxiv.org/pdf/2512.00511v3)**

> **作者:** Ellison Murray; Morriel Kasher; Predrag Spasojevic
>
> **备注:** 10 pages, 8 figures, Accepted 2026 Data Compression Conference
>
> **摘要:** Dithering is a technique commonly used to improve the perceptual quality of lossy data compression. In this work, we analytically and experimentally justify the use of dithering for ASR input compression. We formalize an understanding of optimal ASR performance under lossy input compression and leverage this to propose a parametric dithering technique for a low-complexity speech compression pipeline. The method performs well at 1-bit resolution, showing a 25\% relative CER improvement, while also demonstrating improvements of 32.4\% and 33.5\% at 2- and 3-bit resolution, respectively, with our second dither choice yielding a reduced data rate. The proposed codec is adaptable to meet performance targets or stay within entropy constraints.
>
---
#### [replaced 005] It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models
- **分类: cs.MM; cs.CV; cs.LG; eess.AS**

- **简介: 该论文属心理健康检测任务，旨在解决传统大模型无法有效融合非语言音视频线索的问题。作者提出一种多模态大模型框架，通过在时间戳级对齐音视频特征，提升抑郁检测性能，并验证了其优越性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.19877v2](https://arxiv.org/pdf/2511.19877v2)**

> **作者:** Xiangyu Zhao; Yaling Shen; Yiwen Jiang; Zimu Wang; Jiahe Liu; Maxmartwell H Cheng; Guilherme C Oliveira; Robert Desimone; Dominic Dwyer; Zongyuan Ge
>
> **摘要:** Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
>
---
#### [replaced 006] Universal Discrete-Domain Speech Enhancement
- **分类: cs.SD**

- **简介: 该论文研究语音增强任务，旨在解决多种失真并存时的通用性问题。提出UDSE模型，将增强视为离散域分类任务，通过预测量化语音标记恢复干净语音，实现对复合及非常规失真的有效处理。**

- **链接: [https://arxiv.org/pdf/2510.09974v2](https://arxiv.org/pdf/2510.09974v2)**

> **作者:** Fei Liu; Yang Ai; Ye-Xin Lu; Rui-Chen Zheng; Hui-Peng Du; Zhen-Hua Ling
>
> **摘要:** In real-world scenarios, speech signals are inevitably corrupted by various types of interference, making speech enhancement (SE) a critical task for robust speech processing. However, most existing SE methods only handle a limited range of distortions, such as additive noise, reverberation, or band limitation, while the study of SE under multiple simultaneous distortions remains limited. This gap affects the generalization and practical usability of SE methods in real-world environments.To address this gap, this paper proposes a novel Universal Discrete-domain SE model called UDSE.Unlike regression-based SE models that directly predict clean speech waveform or continuous features, UDSE redefines SE as a discrete-domain classification task, instead predicting the clean discrete tokens quantized by the residual vector quantizer (RVQ) of a pre-trained neural speech codec.Specifically, UDSE first extracts global features from the degraded speech. Guided by these global features, the clean token prediction for each VQ follows the rules of RVQ, where the prediction of each VQ relies on the results of the preceding ones. Finally, the predicted clean tokens from all VQs are decoded to reconstruct the clean speech waveform. During training, the UDSE model employs a teacher-forcing strategy, and is optimized with cross-entropy loss. Experimental results confirm that the proposed UDSE model can effectively enhance speech degraded by various conventional and unconventional distortions, e.g., additive noise, reverberation, band limitation, clipping, phase distortion, and compression distortion, as well as their combinations. These results demonstrate the superior universality and practicality of UDSE compared to advanced regression-based SE methods.
>
---
