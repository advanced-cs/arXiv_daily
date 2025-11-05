# 音频 cs.SD;  eess.AS

- **最新发布 10 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Improving DF-Conformer Using Hydra For High-Fidelity Generative Speech Enhancement on Discrete Codec Token
- **分类: cs.SD**

- **简介: 该论文面向离散编解码器令牌的高保真生成式语音增强任务，提出用Hydra替代DF-Conformer中的FAVOR+，以消除近似误差并保持线性复杂度，提升全局序列建模能力，显著优于原模型。**

- **链接: [http://arxiv.org/pdf/2511.02454v1](http://arxiv.org/pdf/2511.02454v1)**

> **作者:** Shogo Seki; Shaoxiang Dang; Li Li
>
> **备注:** Submitted to ICASSP 2026. Audio samples available at https://s-seki.github.io/dc_hydra/
>
> **摘要:** The Dilated FAVOR Conformer (DF-Conformer) is an efficient variant of the Conformer architecture designed for speech enhancement (SE). It employs fast attention through positive orthogonal random features (FAVOR+) to mitigate the quadratic complexity associated with self-attention, while utilizing dilated convolution to expand the receptive field. This combination results in impressive performance across various SE models. In this paper, we propose replacing FAVOR+ with bidirectional selective structured state-space sequence models to achieve two main objectives:(1) enhancing global sequential modeling by eliminating the approximations inherent in FAVOR+, and (2) maintaining linear complexity relative to the sequence length. Specifically, we utilize Hydra, a bidirectional extension of Mamba, framed within the structured matrix mixer framework. Experiments conducted using a generative SE model on discrete codec tokens, known as Genhancer, demonstrate that the proposed method surpasses the performance of the DF-Conformer.
>
---
#### [new 002] From the perspective of perceptual speech quality: The robustness of frequency bands to noise
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文研究语音频带对噪声的感知质量鲁棒性，采用MUSHRA方法评估32个频带在噪声下的主观质量得分，发现中频段最易受噪声影响，建议未来语音质量提升应重点关注中频段。**

- **链接: [http://arxiv.org/pdf/2511.02252v1](http://arxiv.org/pdf/2511.02252v1)**

> **作者:** Junyi Fan; Donald S. Williamson
>
> **备注:** Accepted to J. Acoust. Soc. Am. (JASA) 155, 1916-1927 (2024)
>
> **摘要:** Speech quality is one of the main foci of speech-related research, where it is frequently studied with speech intelligibility, another essential measurement. Band-level perceptual speech intelligibility, however, has been studied frequently, whereas speech quality has not been thoroughly analyzed. In this paper, a Multiple Stimuli With Hidden Reference and Anchor (MUSHRA) inspired approach was proposed to study the individual robustness of frequency bands to noise with perceptual speech quality as the measure. Speech signals were filtered into thirty-two frequency bands with compromising real-world noise employed at different signal-to-noise ratios. Robustness to noise indices of individual frequency bands was calculated based on the human-rated perceptual quality scores assigned to the reconstructed noisy speech signals. Trends in the results suggest the mid-frequency region appeared less robust to noise in terms of perceptual speech quality. These findings suggest future research aiming at improving speech quality should pay more attention to the mid-frequency region of the speech signals accordingly.
>
---
#### [new 003] Multiplexing Neural Audio Watermarks
- **分类: eess.AS**

- **简介: 该论文面向音频水印任务，解决现有方法易受神经重建等攻击的问题，提出五种多路复用策略，其中PA-TFM无需训练即可显著提升鲁棒性，尤其对抗新型神经攻击。**

- **链接: [http://arxiv.org/pdf/2511.02278v1](http://arxiv.org/pdf/2511.02278v1)**

> **作者:** Zheqi Yuan; Yucheng Huang; Guangzhi Sun; Zengrui Jin; Chao Zhang
>
> **备注:** Submission of IEEE ICASSP 2026
>
> **摘要:** Audio watermarking is a promising tool to ensure authenticity of speech content. However, existing watermarking methods remain vulnerable to more advanced dilution attacks such as lossy compression and neural reconstruction. In this paper, we propose to multiplex neural audio watermarking techniques to leverage their complementarity under different types of attacks. Specifically, five different multiplexing designs are investigated, including parallel, sequential, frequency-division, time-division and perceptual adaptive time-frequency multiplexing (PA-TFM). We evaluate our multiplexing technique on LibriSpeech data with 11 different attack methods, including 2 new neural reconstruction attacks featuring recent advancements in speech processing. As a result, the proposed PA-TFM as a training-free multiplexing method achieves better performance than single watermarking baselines by clear margins, showcasing a more robust way of using watermarks for audio.
>
---
#### [new 004] Augmenting Open-Vocabulary Dysarthric Speech Assessment with Human Perceptual Supervision
- **分类: eess.AS**

- **简介: 该论文属于语音障碍评估任务，旨在提升自监督模型对构音障碍语音的评估性能。通过引入语音合成评估中的人类感知标注作为跨域监督信号，实现知识迁移，显著提升模型效果。**

- **链接: [http://arxiv.org/pdf/2511.02270v1](http://arxiv.org/pdf/2511.02270v1)**

> **作者:** Kaimeng Jia; Minzhu Tu; Zengrui Jin; Siyin Wang; Chao Zhang
>
> **备注:** Submission of IEEE ICASSP 2026
>
> **摘要:** Dysarthria is a speech disorder characterized by impaired intelligibility and reduced communicative effectiveness. Automatic dysarthria assessment provides a scalable, cost-effective approach for supporting the diagnosis and treatment of neurological conditions such as Parkinson's disease, Alzheimer's disease, and stroke. This study investigates leveraging human perceptual annotations from speech synthesis assessment as reliable out-of-domain knowledge for dysarthric speech assessment. Experimental results suggest that such supervision can yield consistent and substantial performance improvements in self-supervised learning pre-trained models. These findings suggest that perceptual ratings aligned with human judgments from speech synthesis evaluations represent valuable resources for dysarthric speech modeling, enabling effective cross-domain knowledge transfer.
>
---
#### [new 005] Perceived Femininity in Singing Voice: Analysis and Prediction
- **分类: cs.SD**

- **简介: 该论文研究歌唱声音的感知女性化（PSVF），填补了语音性别研究在歌唱领域的空白。通过问卷调查收集数据，并基于x-vector构建自动预测模型，旨在揭示性别刻板印象，助力音乐内容中的非二元性别分析。**

- **链接: [http://arxiv.org/pdf/2511.02726v1](http://arxiv.org/pdf/2511.02726v1)**

> **作者:** Yuexuan Kong; Viet-Anh Tran; Romain Hennequin
>
> **摘要:** This paper focuses on the often-overlooked aspect of perceived voice femininity in singing voices. While existing research has examined perceived voice femininity in speech, the same concept has not yet been studied in singing voice. The analysis of gender bias in music content could benefit from such study. To address this gap, we design a stimuli-based survey to measure perceived singing voice femininity (PSVF), and collect responses from 128 participants. Our analysis reveals intriguing insights into how PSVF varies across different demographic groups. Furthermore, we propose an automatic PSVF prediction model by fine-tuning an x-vector model, offering a novel tool for exploring gender stereotypes related to voices in music content analysis beyond binary sex classification. This study contributes to a deeper understanding of the complexities surrounding perceived femininity in singing voices by analyzing survey and proposes an automatic tool for future research.
>
---
#### [new 006] Toward Objective and Interpretable Prosody Evaluation in Text-to-Speech: A Linguistically Motivated Approach
- **分类: eess.AS**

- **简介: 该论文面向TTS语音韵律评估，解决传统主观评分（如MOS）低效且不可解释的问题，提出一种基于语言学的半自动双层评估框架，融合离散与连续声学指标，实现客观、可解释的韵律诊断与优化。**

- **链接: [http://arxiv.org/pdf/2511.02104v1](http://arxiv.org/pdf/2511.02104v1)**

> **作者:** Cedric Chan; Jianjing Kuang
>
> **摘要:** Prosody is essential for speech technology, shaping comprehension, naturalness, and expressiveness. However, current text-to-speech (TTS) systems still struggle to accurately capture human-like prosodic variation, in part because existing evaluation methods for prosody remain limited. Traditional metrics like Mean Opinion Score (MOS) are resource-intensive, inconsistent, and offer little insight into why a system sounds unnatural. This study introduces a linguistically informed, semi-automatic framework for evaluating TTS prosody through a two-tier architecture that mirrors human prosodic organization. The method uses quantitative linguistic criteria to evaluate synthesized speech against human speech corpora across multiple acoustic dimensions. By integrating discrete and continuous prosodic measures, it provides objective and interpretable metrics of both event placement and cue realization, while accounting for the natural variability observed across speakers and prosodic cues. Results show strong correlations with perceptual MOS ratings while revealing model-specific weaknesses that traditional perceptual tests alone cannot capture. This approach provides a principled path toward diagnosing, benchmarking, and ultimately improving the prosodic naturalness of next-generation TTS systems.
>
---
#### [new 007] H-Infinity Filter Enhanced CNN-LSTM for Arrhythmia Detection from Heart Sound Recordings
- **分类: cs.LG; cs.AI; cs.SD; cs.SY; eess.SY**

- **简介: 该论文提出一种CNN-H-Infinity-LSTM模型，用于从心音信号中自动检测心律失常，解决传统方法依赖人工、泛化能力差的问题，通过引入H∞滤波增强鲁棒性，在PhysioNet数据集上达到99.42%准确率。**

- **链接: [http://arxiv.org/pdf/2511.02379v1](http://arxiv.org/pdf/2511.02379v1)**

> **作者:** Rohith Shinoj Kumar; Rushdeep Dinda; Aditya Tyagi; Annappa B.; Naveen Kumar M. R
>
> **备注:** This is a preprint of a paper to appear at the 15th IEEE International Conference on Systems Engineering and Technology (ICSET 2025)
>
> **摘要:** Early detection of heart arrhythmia can prevent severe future complications in cardiac patients. While manual diagnosis still remains the clinical standard, it relies heavily on visual interpretation and is inherently subjective. In recent years, deep learning has emerged as a powerful tool to automate arrhythmia detection, offering improved accuracy, consistency, and efficiency. Several variants of convolutional and recurrent neural network architectures have been widely explored to capture spatial and temporal patterns in physiological signals. However, despite these advancements, current models often struggle to generalize well in real-world scenarios, especially when dealing with small or noisy datasets, which are common challenges in biomedical applications. In this paper, a novel CNN-H-Infinity-LSTM architecture is proposed to identify arrhythmic heart signals from heart sound recordings. This architecture introduces trainable parameters inspired by the H-Infinity filter from control theory, enhancing robustness and generalization. Extensive experimentation on the PhysioNet CinC Challenge 2016 dataset, a public benchmark of heart audio recordings, demonstrates that the proposed model achieves stable convergence and outperforms existing benchmarks, with a test accuracy of 99.42% and an F1 score of 98.85%.
>
---
#### [new 008] An unscented Kalman filter method for real time input-parameter-state estimation
- **分类: eess.SP; cs.AI; cs.CV; cs.SY; eess.AS; eess.SY; 68T05 (Learning and adaptive systems); I.2.6; I.2.8**

- **简介: 该论文提出一种无迹卡尔曼滤波方法，实现线性与非线性系统中输入、参数与状态的实时联合估计，解决传统输出仅参数辨识无法同时获取系统全状态的问题，并通过扰动分析证明其唯一可辨识性。**

- **链接: [http://arxiv.org/pdf/2511.02717v1](http://arxiv.org/pdf/2511.02717v1)**

> **作者:** Marios Impraimakis; Andrew W. Smyth
>
> **备注:** author-accepted manuscript (AAM) published in Mechanical Systems and Signal Processing
>
> **摘要:** The input-parameter-state estimation capabilities of a novel unscented Kalman filter is examined herein on both linear and nonlinear systems. The unknown input is estimated in two stages within each time step. Firstly, the predicted dynamic states and the system parameters provide an estimation of the input. Secondly, the corrected with measurements states and parameters provide a final estimation. Importantly, it is demonstrated using the perturbation analysis that, a system with at least a zero or a non-zero known input can potentially be uniquely identified. This output-only methodology allows for a better understanding of the system compared to classical output-only parameter identification strategies, given that all the dynamic states, the parameters, and the input are estimated jointly and in real-time.
>
---
#### [new 009] An Evaluation of Interleaved Instruction Tuning on Semantic Reasoning Performance in an Audio MLLM
- **分类: cs.MM; cs.CL; cs.SD**

- **简介: 该论文研究音频多模态大模型中交错指令微调对语义推理的影响，旨在提升模型对同义/上位词的音频推理能力。通过新构建的SHARD数据集，证明交错提示微调可增强推理性能，但会削弱音频标注能力。**

- **链接: [http://arxiv.org/pdf/2511.02234v1](http://arxiv.org/pdf/2511.02234v1)**

> **作者:** Jiawei Liu; Enis Berk Çoban; Zarina Schevchenko; Hao Tang; Zhigang Zhu; Michael I Mandel; Johanna Devaney
>
> **摘要:** Standard training for Multi-modal Large Language Models (MLLMs) involves concatenating non-textual information, like vision or audio, with a text prompt. This approach may not encourage deep integration of modalities, limiting the model's ability to leverage the core language model's reasoning capabilities. This work examined the impact of interleaved instruction tuning in an audio MLLM, where audio tokens are interleaved within the prompt. Using the Listen, Think, and Understand (LTU) model as a testbed, we conduct an experiment using the Synonym and Hypernym Audio Reasoning Dataset (SHARD), our newly created reasoning benchmark for audio-based semantic reasoning focusing on synonym and hypernym recognition. Our findings show that while even zero-shot interleaved prompting improves performance on our reasoning tasks, a small amount of fine-tuning using interleaved training prompts improves the results further, however, at the expense of the MLLM's audio labeling ability.
>
---
#### [new 010] Condition-Invariant fMRI Decoding of Speech Intelligibility with Deep State Space Model
- **分类: q-bio.NC; cs.LG; cs.SD; eess.AS; eess.SP**

- **简介: 该论文旨在解决语音可懂度在不同听觉环境下神经表征是否不变的问题，提出一种深度状态空间模型，首次实现跨条件fMRI解码语音可懂度，并揭示了大脑中存在条件不变的抽象语言编码。**

- **链接: [http://arxiv.org/pdf/2511.01868v1](http://arxiv.org/pdf/2511.01868v1)**

> **作者:** Ching-Chih Sung; Shuntaro Suzuki; Francis Pingfan Chien; Komei Sugiura; Yu Tsao
>
> **摘要:** Clarifying the neural basis of speech intelligibility is critical for computational neuroscience and digital speech processing. Recent neuroimaging studies have shown that intelligibility modulates cortical activity beyond simple acoustics, primarily in the superior temporal and inferior frontal gyri. However, previous studies have been largely confined to clean speech, leaving it unclear whether the brain employs condition-invariant neural codes across diverse listening environments. To address this gap, we propose a novel architecture built upon a deep state space model for decoding intelligibility from fMRI signals, specifically tailored to their high-dimensional temporal structure. We present the first attempt to decode intelligibility across acoustically distinct conditions, showing our method significantly outperforms classical approaches. Furthermore, region-wise analysis highlights contributions from auditory, frontal, and parietal regions, and cross-condition transfer indicates the presence of condition-invariant neural codes, thereby advancing understanding of abstract linguistic representations in the brain.
>
---
## 更新

#### [replaced 001] MultiSoundGen: Video-to-Audio Generation for Multi-Event Scenarios via SlowFast Contrastive Audio-Visual Pretraining and Direct Preference Optimization
- **分类: cs.MM; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.19999v2](http://arxiv.org/pdf/2509.19999v2)**

> **作者:** Jianxuan Yang; Xiaoran Yang; Lipan Zhang; Xinyue Guo; Zhao Wang; Gongping Huang
>
> **摘要:** Current video-to-audio (V2A) methods struggle in complex multi-event scenarios (video scenarios involving multiple sound sources, sound events, or transitions) due to two critical limitations. First, existing methods face challenges in precisely aligning intricate semantic information together with rapid dynamic features. Second, foundational training lacks quantitative preference optimization for semantic-temporal alignment and audio quality. As a result, it fails to enhance integrated generation quality in cluttered multi-event scenes. To address these core limitations, this study proposes a novel V2A framework: MultiSoundGen. It introduces direct preference optimization (DPO) into the V2A domain, leveraging audio-visual pretraining (AVP) to enhance performance in complex multi-event scenarios. Our contributions include two key innovations: the first is SlowFast Contrastive AVP (SF-CAVP), a pioneering AVP model with a unified dual-stream architecture. SF-CAVP explicitly aligns core semantic representations and rapid dynamic features of audio-visual data to handle multi-event complexity; second, we integrate the DPO method into V2A task and propose AVP-Ranked Preference Optimization (AVP-RPO). It uses SF-CAVP as a reward model to quantify and prioritize critical semantic-temporal matches while enhancing audio quality. Experiments demonstrate that MultiSoundGen achieves state-of-the-art (SOTA) performance in multi-event scenarios, delivering comprehensive gains across distribution matching, audio quality, semantic alignment, and temporal synchronization. Demos are available at https://v2aresearch.github.io/MultiSoundGen/.
>
---
#### [replaced 002] Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.20410v4](http://arxiv.org/pdf/2509.20410v4)**

> **作者:** Weijie Wu; Wenhao Guan; Kaidi Wang; Peijie Chen; Zhuanling Zha; Junbo Li; Jun Fang; Lin Li; Qingyang Hong
>
> **备注:** It requires internal PR approval
>
> **摘要:** Spoken dialogue models have significantly advanced intelligent human-computer interaction, yet they lack a plug-and-play full-duplex prediction module for semantic endpoint detection, hindering seamless audio interactions. In this paper, we introduce Phoenix-VAD, an LLM-based model that enables streaming semantic endpoint detection. Specifically, Phoenix-VAD leverages the semantic comprehension capability of the LLM and a sliding window training strategy to achieve reliable semantic endpoint detection while supporting streaming inference. Experiments on both semantically complete and incomplete speech scenarios indicate that Phoenix-VAD achieves excellent and competitive performance. Furthermore, this design enables the full-duplex prediction module to be optimized independently of the dialogue model, providing more reliable and flexible support for next-generation human-computer interaction.
>
---
#### [replaced 003] AuthGlass: Enhancing Voice Authentication on Smart Glasses via Air-Bone Acoustic Features
- **分类: cs.HC; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.20799v2](http://arxiv.org/pdf/2509.20799v2)**

> **作者:** Weiye Xu; Zhang Jiang; Siqi Zheng; Xiyuxing Zhang; Yankai Zhao; Changhao Zhang; Jian Liu; Weiqiang Wang; Yuntao Wang
>
> **备注:** This version has been updated to remove proprietary details pending a patent filing
>
> **摘要:** With the rapid advancement of smart glasses, voice interaction has become widely deployed due to its naturalness and convenience. However, its practicality is often undermined by the vulnerability to spoofing attacks and interference from surrounding sounds, making seamless voice authentication crucial for smart glasses usage. To address this challenge, we propose AuthGlass, a voice authentication approach that leverages both air- and bone-conducted speech features to enhance accuracy and liveness detection. Aiming to gain comprehensive knowledge on speech-related acoustic and vibration features, we built a smart glasses prototype with redundant synchronized microphones: 14 air-conductive microphones and 2 bone-conductive units. In a study with 42 participants, we validated that combining sound-field and vibration features significantly improves authentication robustness and attack resistance. Furthermore, experiments demonstrated that AuthGlass maintains competitive accuracy even under various practical scenarios, highlighting its applicability and scalability for real-world deployment.
>
---
#### [replaced 004] Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08039v3](http://arxiv.org/pdf/2508.08039v3)**

> **作者:** Shu Wu; Chenxing Li; Wenfu Wang; Hao Zhang; Hualei Wang; Meng Yu; Dong Yu
>
> **备注:** preprint
>
> **摘要:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.
>
---
#### [replaced 005] DARAS: Dynamic Audio-Room Acoustic Synthesis for Blind Room Impulse Response Estimation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.08135v2](http://arxiv.org/pdf/2507.08135v2)**

> **作者:** Chunxi Wang; Maoshen Jia; Wenyu Jin
>
> **备注:** 14 pages, 9 figures, accepted for publication in IEEE/ACM Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Room Impulse Responses (RIRs) accurately characterize acoustic properties of indoor environments and play a crucial role in applications such as speech enhancement, speech recognition, and audio rendering in augmented reality (AR) and virtual reality (VR). Existing blind estimation methods struggle to achieve practical accuracy. To overcome this challenge, we propose the dynamic audio-room acoustic synthesis (DARAS) model, a novel deep learning framework that is explicitly designed for blind RIR estimation from monaural reverberant speech signals. First, a dedicated deep audio encoder effectively extracts relevant nonlinear latent space features. Second, the Mamba-based self-supervised blind room parameter estimation (MASS-BRPE) module, utilizing the efficient Mamba state space model (SSM), accurately estimates key room acoustic parameters and features. Third, the system incorporates a hybrid-path cross-attention feature fusion module, enhancing deep integration between audio and room acoustic features. Finally, our proposed dynamic acoustic tuning (DAT) decoder adaptively segments early reflections and late reverberation to improve the realism of synthesized RIRs. Experimental results, including a MUSHRA-based subjective listening study, demonstrate that DARAS substantially outperforms existing baseline models, providing a robust and effective solution for practical blind RIR estimation in real-world acoustic environments.
>
---
#### [replaced 006] Prevailing Research Areas for Music AI in the Era of Foundation Models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS; 68T05, 68T20; I.2; I.5.4; I.2.6; I.2.7; H.5.5**

- **链接: [http://arxiv.org/pdf/2409.09378v3](http://arxiv.org/pdf/2409.09378v3)**

> **作者:** Megan Wei; Mateusz Modrzejewski; Aswin Sivaraman; Dorien Herremans
>
> **摘要:** Parallel to rapid advancements in foundation model research, the past few years have witnessed a surge in music AI applications. As AI-generated and AI-augmented music become increasingly mainstream, many researchers in the music AI community may wonder: what research frontiers remain unexplored? This paper outlines several key areas within music AI research that present significant opportunities for further investigation. We begin by examining foundational representation models and highlight emerging efforts toward explainability and interpretability. We then discuss the evolution toward multimodal systems, provide an overview of the current landscape of music datasets and their limitations, and address the growing importance of model efficiency in both training and deployment. Next, we explore applied directions, focusing first on generative models. We review recent systems, their computational constraints, and persistent challenges related to evaluation and controllability. We then examine extensions of these generative approaches to multimodal settings and their integration into artists' workflows, including applications in music editing, captioning, production, transcription, source separation, performance, discovery, and education. Finally, we explore copyright implications of generative music and propose strategies to safeguard artist rights. While not exhaustive, this survey aims to illuminate promising research directions enabled by recent developments in music foundation models.
>
---
