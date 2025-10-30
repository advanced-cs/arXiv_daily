# 音频 cs.SD;  eess.AS

- **最新发布 14 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Controlling Contrastive Self-Supervised Learning with Knowledge-Driven Multiple Hypothesis: Application to Beat Tracking
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音乐节拍追踪任务，解决因数据歧义导致的多合理解问题。提出基于知识驱动多假设的对比自监督学习方法，通过筛选最合理的假设来学习鲁棒表示，显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.25560v1](http://arxiv.org/pdf/2510.25560v1)**

> **作者:** Antonin Gagnere; Slim Essid; Geoffroy Peeters
>
> **摘要:** Ambiguities in data and problem constraints can lead to diverse, equally plausible outcomes for a machine learning task. In beat and downbeat tracking, for instance, different listeners may adopt various rhythmic interpretations, none of which would necessarily be incorrect. To address this, we propose a contrastive self-supervised pre-training approach that leverages multiple hypotheses about possible positive samples in the data. Our model is trained to learn representations compatible with different such hypotheses, which are selected with a knowledge-based scoring function to retain the most plausible ones. When fine-tuned on labeled data, our model outperforms existing methods on standard benchmarks, showcasing the advantages of integrating domain knowledge with multi-hypothesis selection in music representation learning in particular.
>
---
#### [new 002] A Parameter-Efficient Multi-Scale Convolutional Adapter for Synthetic Speech Detection
- **分类: cs.SD**

- **简介: 该论文针对合成语音检测任务，解决现有参数高效微调方法缺乏多尺度时序特征建模能力的问题。提出多尺度卷积适配器（MultiConvAdapter），在预训练自监督模型中引入并行卷积模块，以低参数量（3.17M）有效捕捉短时与长时伪音特征，显著降低计算开销并提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.24852v1](http://arxiv.org/pdf/2510.24852v1)**

> **作者:** Yassine El Kheir; Fabian Ritter-Guttierez; Arnab Das; Tim Polzehl; Sebastian Möller
>
> **备注:** 6 pages
>
> **摘要:** Recent synthetic speech detection models typically adapt a pre-trained SSL model via finetuning, which is computationally demanding. Parameter-Efficient Fine-Tuning (PEFT) offers an alternative. However, existing methods lack the specific inductive biases required to model the multi-scale temporal artifacts characteristic of spoofed audio. This paper introduces the Multi-Scale Convolutional Adapter (MultiConvAdapter), a parameter-efficient architecture designed to address this limitation. MultiConvAdapter integrates parallel convolutional modules within the SSL encoder, facilitating the simultaneous learning of discriminative features across multiple temporal resolutions, capturing both short-term artifacts and long-term distortions. With only $3.17$M trainable parameters ($1\%$ of the SSL backbone), MultiConvAdapter substantially reduces the computational burden of adaptation. Evaluations on five public datasets, demonstrate that MultiConvAdapter achieves superior performance compared to full fine-tuning and established PEFT methods.
>
---
#### [new 003] Separating peripheral and higher-level effects on speech intelligibility using a hearing loss simulator and an objective intelligibility measure
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究听力损失对语音可懂度的影响，旨在分离外周听力损失与高级认知过程的作用。通过模拟老年听障者听力并结合客观可懂度指标GESI，对比年轻正常听力者，发现部分老年人更高阶处理能力更强，验证了方法的有效性，为个体化研究提供工具。**

- **链接: [http://arxiv.org/pdf/2510.25235v1](http://arxiv.org/pdf/2510.25235v1)**

> **作者:** Toshio Irino; Ayako Yamamoto; Fuki Miyazaki
>
> **备注:** This is a manuscript that was submitted to Trends in Hearing on October 29, 2025
>
> **摘要:** This paper presents a new method for separating the effects of peripheral hearing loss (HL) and higher-level processes on speech intelligibility (SI). In a previous study, we conducted an SI experiment with 14 older adult (OA) listeners, using speech-in-noise sounds that were either processed with an ideal ratio mask (IRM) enhancement technique or left unprocessed. The current study involved an SI experiment with 15 young, normal-hearing (YNH) listeners. This experiment used simulated HL sounds processed with the WHIS simulator that reflected the hearing level of a specific OA from the previous study. The results showed that the target OA's SI scores were higher than the average YNH scores. This implies that the target OA's higher-level processes may be more effective than those of the average YNH. To understand the characteristics of other OAs, we used the GESI objective intelligibility measure to predict SI. First, we confirmed that GESI could fairly accurately predict the SI scores for both the YNH and OA listeners. Next, we predicted the SI scores of the 14 OA listeners using the parameters estimated in the YNH experiment. The results showed that some OAs had higher SI scores than the average YNH, while one OA had lower scores. These differences in SI scores may reflect variations in the efficiency of higher-level processes.These results imply that WHIS and GESI could facilitate contrastive experiments between YNH and OA listeners, regardless of hearing level. This would allow us to study the effects of higher-level processes in OA listeners individually.
>
---
#### [new 004] Efficient Vocal Source Separation Through Windowed Sink Attention
- **分类: cs.SD**

- **简介: 该论文针对语音分离任务，解决现有模型因全时域自注意力导致计算量过大的问题。通过分析发现注意力具有局部性，提出窗口化感知注意力（WSA），结合小窗口与注意力汇点，在仅损失8% SDR性能下，将计算量降低44.5倍。**

- **链接: [http://arxiv.org/pdf/2510.25745v1](http://arxiv.org/pdf/2510.25745v1)**

> **作者:** Christodoulos Benetatos; Yongyi Zang; Randal Leistikow
>
> **摘要:** State-of-the-art vocal separation models like Mel-Band-Roformer rely on full temporal self-attention mechanisms, where each temporal frame interacts with every other frames. This incurs heavy computational costs that scales quadratically with input audio length, motivating chunking and windowing approaches. Through analysis of a pre-trained vocal separation model, we discovered that temporal attention patterns are highly localized. Building on this insight, we replaced full attention with windowed sink attention (WSA) with small temporal attention window and attention sinks. We show empirically that fine-tuning from the original checkpoint recovers 92% of the original SDR performance while reducing FLOPs by 44.5x. We release our code and checkpoints under MIT license at https://github.com/smulelabs/windowed-roformer.
>
---
#### [new 005] Joint Analysis of Acoustic Scenes and Sound Events Based on Semi-Supervised Training of Sound Events With Partial Labels
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对音频检测中标注成本高的问题，提出联合声景分类与部分标签声音事件检测的半监督学习框架。通过利用声景上下文构建部分标签，并引入自蒸馏方法进行标签优化，有效降低标注成本的同时提升检测性能。**

- **链接: [http://arxiv.org/pdf/2510.25075v1](http://arxiv.org/pdf/2510.25075v1)**

> **作者:** Keisuke Imoto
>
> **备注:** Accepted to APSIPA Transactions on Signal and Information Processing
>
> **摘要:** Annotating time boundaries of sound events is labor-intensive, limiting the scalability of strongly supervised learning in audio detection. To reduce annotation costs, weakly-supervised learning with only clip-level labels has been widely adopted. As an alternative, partial label learning offers a cost-effective approach, where a set of possible labels is provided instead of exact weak annotations. However, partial label learning for audio analysis remains largely unexplored. Motivated by the observation that acoustic scenes provide contextual information for constructing a set of possible sound events, we utilize acoustic scene information to construct partial labels of sound events. On the basis of this idea, in this paper, we propose a multitask learning framework that jointly performs acoustic scene classification and sound event detection with partial labels of sound events. While reducing annotation costs, weakly-supervised and partial label learning often suffer from decreased detection performance due to lacking the precise event set and their temporal annotations. To better balance between annotation cost and detection performance, we also explore a semi-supervised framework that leverages both strong and partial labels. Moreover, to refine partial labels and achieve better model training, we propose a label refinement method based on self-distillation for the proposed approach with partial labels.
>
---
#### [new 006] PitchFlower: A flow-based neural audio codec with pitch controllability
- **分类: eess.AS; cs.LG**

- **简介: 该论文提出PitchFlower，一种具显式音高可控性的流模型音频编解码器。通过训练时平滑并随机偏移基频轮廓，结合向量量化瓶颈实现音高解耦，使音高控制更精准。实验表明其音高控制优于WORLD，且在质量上超越SiFiGAN，同时为语音其他属性的解耦提供通用框架。**

- **链接: [http://arxiv.org/pdf/2510.25566v1](http://arxiv.org/pdf/2510.25566v1)**

> **作者:** Diego Torres; Axel Roebel; Nicolas Obin
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** We present PitchFlower, a flow-based neural audio codec with explicit pitch controllability. Our approach enforces disentanglement through a simple perturbation: during training, F0 contours are flattened and randomly shifted, while the true F0 is provided as conditioning. A vector-quantization bottleneck prevents pitch recovery, and a flow-based decoder generates high quality audio. Experiments show that PitchFlower achieves more accurate pitch control than WORLD at much higher audio quality, and outperforms SiFiGAN in controllability while maintaining comparable quality. Beyond pitch, this framework provides a simple and extensible path toward disentangling other speech attributes.
>
---
#### [new 007] Retaining Mixture Representations for Domain Generalized Anomalous Sound Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对域泛化异常声音检测任务，解决现有方法在混合噪声下泛化能力差的问题。提出保留混合表示而非去噪的策略，结合多标签音频标记与混合对齐损失，使模型更好保留混合源信息，提升对分布偏移的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.25182v1](http://arxiv.org/pdf/2510.25182v1)**

> **作者:** Phurich Saengthong; Tomoya Nishida; Kota Dohi; Natsuo Yamashita; Yohei Kawaguchi
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Anomalous sound detection (ASD) in the wild requires robustness to distribution shifts such as unseen low-SNR input mixtures of machine and noise types. State-of-the-art systems extract embeddings from an adapted audio encoder and detect anomalies via nearest-neighbor search, but fine tuning on noisy machine sounds often acts like a denoising objective, suppressing noise and reducing generalization under mismatched mixtures or inconsistent labeling. Training-free systems with frozen self-supervised learning (SSL) encoders avoid this issue and show strong first-shot generalization, yet their performance drops when mixture embeddings deviate from clean-source embeddings. We propose to improve SSL backbones with a retain-not-denoise strategy that better preserves information from mixed sound sources. The approach combines a multi-label audio tagging loss with a mixture alignment loss that aligns student mixture embeddings to convex teacher embeddings of clean and noise inputs. Controlled experiments on stationary, non-stationary, and mismatched noise subsets demonstrate improved robustness under distribution shifts, narrowing the gap toward oracle mixture representations.
>
---
#### [new 008] SFMS-ALR: Script-First Multilingual Speech Synthesis with Adaptive Locale Resolution
- **分类: cs.SD; cs.AI; eess.AS; I.2.7; H.5.5**

- **简介: 该论文提出SFMS-ALR框架，解决多语言语音合成中因语言切换导致的语调不连贯、脚本混杂等问题。通过按字符集分段、自适应语言与方言识别、情感感知声调归一化，生成统一SSML并单次调用合成，实现跨引擎、无需重训的流畅多语言语音合成。**

- **链接: [http://arxiv.org/pdf/2510.25178v1](http://arxiv.org/pdf/2510.25178v1)**

> **作者:** Dharma Teja Donepudi
>
> **备注:** 10 pages, 2 figures, 1 table. Demonstration prototype available at https://sfml-tts-proxy-253495793487.us-central1.run.app
>
> **摘要:** Intra-sentence multilingual speech synthesis (code-switching TTS) remains a major challenge due to abrupt language shifts, varied scripts, and mismatched prosody between languages. Conventional TTS systems are typically monolingual and fail to produce natural, intelligible speech in mixed-language contexts. We introduce Script-First Multilingual Synthesis with Adaptive Locale Resolution (SFMS-ALR), an engine-agnostic framework for fluent, real-time code-switched speech generation. SFMS-ALR segments input text by Unicode script, applies adaptive language identification to determine each segment's language and locale, and normalizes prosody using sentiment-aware adjustments to preserve expressive continuity across languages. The algorithm generates a unified SSML representation with appropriate "lang" or "voice" spans and synthesizes the utterance in a single TTS request. Unlike end-to-end multilingual models, SFMS-ALR requires no retraining and integrates seamlessly with existing voices from Google, Apple, Amazon, and other providers. Comparative analysis with data-driven pipelines such as Unicom and Mask LID demonstrates SFMS-ALR's flexibility, interpretability, and immediate deployability. The framework establishes a modular baseline for high-quality, engine-independent multilingual TTS and outlines evaluation strategies for intelligibility, naturalness, and user preference.
>
---
#### [new 009] Studies for : A Human-AI Co-Creative Sound Artwork Using a Real-time Multi-channel Sound Generation Model
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出一种人机协同的声景艺术创作框架，通过实时多通道声生成模型SpecMaskGIT，基于200小时艺术家Evala的作品数据，持续生成新声音，构建“新型档案”概念。旨在解决艺术创作中AI与人类艺术家协同、风格延续与创新平衡问题，实现艺术遗产的动态延续。**

- **链接: [http://arxiv.org/pdf/2510.25228v1](http://arxiv.org/pdf/2510.25228v1)**

> **作者:** Chihiro Nagashima; Akira Takahashi; Zhi Zhong; Shusuke Takahashi; Yuki Mitsufuji
>
> **备注:** Accepted at NeurIPS Creative AI Track 2025, 9 pages, 6 figures, 1 table, Demo page: https://sony.github.io/studies-for/
>
> **摘要:** This paper explores the integration of AI technologies into the artistic workflow through the creation of Studies for, a generative sound installation developed in collaboration with sound artist Evala (https://www.ntticc.or.jp/en/archive/works/studies-for/). The installation employs SpecMaskGIT, a lightweight yet high-quality sound generation AI model, to generate and playback eight-channel sound in real-time, creating an immersive auditory experience over the course of a three-month exhibition. The work is grounded in the concept of a "new form of archive," which aims to preserve the artistic style of an artist while expanding beyond artists' past artworks by continued generation of new sound elements. This speculative approach to archival preservation is facilitated by training the AI model on a dataset consisting of over 200 hours of Evala's past sound artworks. By addressing key requirements in the co-creation of art using AI, this study highlights the value of the following aspects: (1) the necessity of integrating artist feedback, (2) datasets derived from an artist's past works, and (3) ensuring the inclusion of unexpected, novel outputs. In Studies for, the model was designed to reflect the artist's artistic identity while generating new, previously unheard sounds, making it a fitting realization of the concept of "a new form of archive." We propose a Human-AI co-creation framework for effectively incorporating sound generation AI models into the sound art creation process and suggest new possibilities for creating and archiving sound art that extend an artist's work beyond their physical existence. Demo page: https://sony.github.io/studies-for/
>
---
#### [new 010] EasyEyes: Online hearing research using speakers calibrated by phones
- **分类: eess.AS**

- **简介: 该论文针对在线听力研究中扬声器未校准的问题，提出EasyEyes.app工具，利用手机麦克风模型库实现电脑扬声器的快速在线校准。通过非同步最大长度序列算法，三分钟内完成校准，确保输出频谱平坦（标准差<3 dB）。支持80%以上美英参与者，提升研究效率与包容性。**

- **链接: [http://arxiv.org/pdf/2510.25048v1](http://arxiv.org/pdf/2510.25048v1)**

> **作者:** Ivan Vican; Hugo De Moraes; Chongjun Liao; Nathnael H. Tsegaye; William O'Gara; Jasper Inamoto; Denis G. Pelli
>
> **摘要:** Hearing research requires a calibrated sound source, traditionally as lab equipment. Online research is quicker and more inclusive, but most participants lack calibration equipment and their sound sources are uncalibrated and diverse. This article explains how the open-source EasyEyes.app calibrates loudspeakers online. A library of smartphone-microphone profiles allows EasyEyes to use the participant's phone to calibrate their computer's loudspeaker in three minutes. Participants select their phone model, which is verified by screen size. Calibration employs the Novak et al. nonsynchronous maximum-length-sequence (MLS) algorithm. The computer's loudspeaker is corrected by convolving its input with the inverse of its impulse response. Researchers can contribute to the open-access library by calibrating phones with a measurement microphone. In the library, each profile is linked back to the profile used to produce it, back to the manufacturer profile of a measurement microphone. Correction accuracy is such that playing the flat-spectrum MLS through the corrected loudspeaker produces a nearly flat spectrum, with standard deviation less than 3 dB. A survey shows that a library of 94 phone models from major brands will support most participants in the USA (87%) and UK (80%). This method facilitates efficient and inclusive online hearing research.
>
---
#### [new 011] Lost in Phonation: Voice Quality Variation as an Evaluation Dimension for Speech Foundation Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文研究语音基础模型（SFM）对语音质量（如沙哑、气声）的敏感性，属于语音理解任务。针对现有评估基准忽视非词汇性语音特征的问题，提出基于开放式生成与情绪识别的新评测方法，并构建了包含语音质量调控的并行数据集，首次系统考察SFM对语音质量变化的响应一致性。**

- **链接: [http://arxiv.org/pdf/2510.25577v1](http://arxiv.org/pdf/2510.25577v1)**

> **作者:** Harm Lameris; Shree Harsha Bokkahalli Satish; Joakim Gustafson; Éva Székely
>
> **备注:** 8 pages, 3 figures, 4 tables, submitted to LREC 2026
>
> **摘要:** Recent advances in speech foundation models (SFMs) have enabled the direct processing of spoken language from raw audio, bypassing intermediate textual representations. This capability allows SFMs to be exposed to, and potentially respond to, rich paralinguistic variations embedded in the input speech signal. One under-explored dimension of paralinguistic variation is voice quality, encompassing phonation types such as creaky and breathy voice. These phonation types are known to influence how listeners infer affective state, stance and social meaning in speech. Existing benchmarks for speech understanding largely rely on multiple-choice question answering (MCQA) formats, which are prone to failure and therefore unreliable in capturing the nuanced ways paralinguistic features influence model behaviour. In this paper, we probe SFMs through open-ended generation tasks and speech emotion recognition, evaluating whether model behaviours are consistent across different phonation inputs. We introduce a new parallel dataset featuring synthesized modifications to voice quality, designed to evaluate SFM responses to creaky and breathy voice. Our work provides the first examination of SFM sensitivity to these particular non-lexical aspects of speech perception.
>
---
#### [new 012] Binaspect -- A Python Library for Binaural Audio Analysis, Visualization & Feature Generation
- **分类: cs.SD**

- **简介: 该论文提出Binaspect，一个用于双耳音频分析、可视化与特征生成的开源Python库。针对双耳线索在编码与渲染中被破坏却难观测的问题，通过构建可解释的方位图，实现对多声源定位及失真情况的直观呈现，并生成可用于机器学习的结构化特征。**

- **链接: [http://arxiv.org/pdf/2510.25714v1](http://arxiv.org/pdf/2510.25714v1)**

> **作者:** Dan Barry; Davoud Shariat Panah; Alessandro Ragano; Jan Skoglund; Andrew Hines
>
> **摘要:** We present Binaspect, an open-source Python library for binaural audio analysis, visualization, and feature generation. Binaspect generates interpretable "azimuth maps" by calculating modified interaural time and level difference spectrograms, and clustering those time-frequency (TF) bins into stable time-azimuth histogram representations. This allows multiple active sources to appear as distinct azimuthal clusters, while degradations manifest as broadened, diffused, or shifted distributions. Crucially, Binaspect operates blindly on audio, requiring no prior knowledge of head models. These visualizations enable researchers and engineers to observe how binaural cues are degraded by codec and renderer design choices, among other downstream processes. We demonstrate the tool on bitrate ladders, ambisonic rendering, and VBAP source positioning, where degradations are clearly revealed. In addition to their diagnostic value, the proposed representations can be exported as structured features suitable for training machine learning models in quality prediction, spatial audio classification, and other binaural tasks. Binaspect is released under an open-source license with full reproducibility scripts at https://github.com/QxLabIreland/Binaspect.
>
---
#### [new 013] State Space and Self-Attention Collaborative Network with Feature Aggregation for DOA Estimation
- **分类: eess.SP; cs.SD**

- **简介: 该论文针对声源方向（DOA）估计任务，解决时频变化下特征聚合与高效建模的难题。提出FA-Stateformer网络，融合特征聚合、轻量Conformer与双向Mamba模块，协同建模时空依赖，提升精度与计算效率。**

- **链接: [http://arxiv.org/pdf/2510.25193v1](http://arxiv.org/pdf/2510.25193v1)**

> **作者:** Qi You; Qinghua Huang; Yi-Cheng Lin
>
> **摘要:** Accurate direction-of-arrival (DOA) estimation for sound sources is challenging due to the continuous changes in acoustic characteristics across time and frequency. In such scenarios, accurate localization relies on the ability to aggregate relevant features and model temporal dependencies effectively. In time series modeling, achieving a balance between model performance and computational efficiency remains a significant challenge. To address this, we propose FA-Stateformer, a state space and self-attention collaborative network with feature aggregation. The proposed network first employs a feature aggregation module to enhance informative features across both temporal and spectral dimensions. This is followed by a lightweight Conformer architecture inspired by the squeeze-and-excitation mechanism, where the feedforward layers are compressed to reduce redundancy and parameter overhead. Additionally, a temporal shift mechanism is incorporated to expand the receptive field of convolutional layers while maintaining a compact kernel size. To further enhance sequence modeling capabilities, a bidirectional Mamba module is introduced, enabling efficient state-space-based representation of temporal dependencies in both forward and backward directions. The remaining self-attention layers are combined with the Mamba blocks, forming a collaborative modeling framework that achieves a balance between representation capacity and computational efficiency. Extensive experiments demonstrate that FA-Stateformer achieves superior performance and efficiency compared to conventional architectures.
>
---
#### [new 014] Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音情感识别任务，针对语音语义与情感表达不一致的情况，评估四种语音语言模型的表现。结果表明，模型主要依赖文本语义而非语音情感特征，揭示了模型对声学模态的整合不足。作者发布了代码和EMIS数据集以促进研究。**

- **链接: [http://arxiv.org/pdf/2510.25054v1](http://arxiv.org/pdf/2510.25054v1)**

> **作者:** Pedro Corrêa; João Lima; Victor Moreno; Paula Dornhofer Paro Costa
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Advancements in spoken language processing have driven the development of spoken language models (SLMs), designed to achieve universal audio understanding by jointly learning text and audio representations for a wide range of tasks. Although promising results have been achieved, there is growing discussion regarding these models' generalization capabilities and the extent to which they truly integrate audio and text modalities in their internal representations. In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results indicate that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.
>
---
## 更新

#### [replaced 001] Predicting speech intelligibility in older adults for speech enhancement using the Gammachirp Envelope Similarity Index, GESI
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2504.14437v3](http://arxiv.org/pdf/2504.14437v3)**

> **作者:** Ayako Yamamoto; Fuki Miyazaki; Toshio Irino
>
> **备注:** This is a copy of the final version that was accepted for publication in Speech Communication on October 12, 2025
>
> **摘要:** We propose an objective intelligibility measure (OIM), called the Gammachirp Envelope Similarity Index (GESI), that can predict speech intelligibility (SI) in older adults. GESI is a bottom-up model based on psychoacoustic knowledge from the peripheral to the central auditory system. It computes the single SI metric using the gammachirp filterbank (GCFB), the modulation filterbank, and the extended cosine similarity measure. It takes into account not only the hearing level represented in the audiogram, but also the temporal processing characteristics captured by the temporal modulation transfer function (TMTF). To evaluate performance, SI experiments were conducted with older adults of various hearing levels using speech-in-noise with ideal speech enhancement on familiarity-controlled Japanese words. The prediction performance was compared with HASPIw2, which was developed for keyword SI prediction. The results showed that GESI predicted the subjective SI scores more accurately than HASPIw2. GESI was also found to be at least as effective as, if not more effective than, HASPIv2 in predicting English sentence-level SI. The effect of introducing TMTF into the GESI algorithm was insignificant, suggesting that TMTF measurements and models are not yet mature. Therefore, it may be necessary to perform TMTF measurements with bandpass noise and to improve the incorporation of temporal characteristics into the model.
>
---
#### [replaced 002] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17937v3](http://arxiv.org/pdf/2507.17937v3)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).
>
---
#### [replaced 003] Application of Whisper in Clinical Practice: the Post-Stroke Speech Assessment during a Naming Task
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17326v2](http://arxiv.org/pdf/2507.17326v2)**

> **作者:** Milena Davudova; Ziyuan Cai; Valentina Giunchiglia; Dragos C. Gruia; Giulia Sanguedolce; Adam Hampshire; Fatemeh Geranmayeh
>
> **摘要:** Detailed assessment of language impairment following stroke remains a cognitively complex and clinician-intensive task, limiting timely and scalable diagnosis. Automatic Speech Recognition (ASR) foundation models offer a promising pathway to augment human evaluation through intelligent systems, but their effectiveness in the context of speech and language impairment remains uncertain. In this study, we evaluate whether Whisper, a state-of-the-art ASR foundation model, can be applied to transcribe and analyze speech from patients with stroke during a commonly used picture-naming task. We assess both verbatim transcription accuracy and the model's ability to support downstream prediction of language function, which has major implications for outcomes after stroke. Our results show that the baseline Whisper model performs poorly on single-word speech utterances. Nevertheless, fine-tuning Whisper significantly improves transcription accuracy (reducing Word Error Rate by 87.72% in healthy speech and 71.22% in speech from patients). Further, learned representations from the model enable accurate prediction of speech quality (average F1 Macro of 0.74 for healthy, 0.75 for patients). However, evaluations on an unseen (TORGO) dataset reveal limited generalizability, highlighting the inability of Whisper to perform zero-shot transcription of single-word utterances on out-of-domain clinical speech and emphasizing the need to adapt models to specific clinical populations. While challenges remain in cross-domain generalization, these findings highlight the potential of foundation models, when appropriately fine-tuned, to advance automated speech and language assessment and rehabilitation for stroke-related impairments.
>
---
#### [replaced 004] Quantifying Multimodal Imbalance: A GMM-Guided Adaptive Loss for Audio-Visual Learning
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.21797v2](http://arxiv.org/pdf/2510.21797v2)**

> **作者:** Zhaocheng Liu; Zhiwen Yu; Xiaoqing Liu
>
> **摘要:** The heterogeneity of multimodal data leads to inconsistencies and imbalance, allowing a dominant modality to steer gradient updates. Existing solutions mainly focus on optimization- or data-based strategies but rarely exploit the information inherent in multimodal imbalance or conduct its quantitative analysis. To address this gap, we propose a novel quantitative analysis framework for Multimodal Imbalance and design a sample-level adaptive loss function. We define the Modality Gap as the Softmax score difference between modalities for the correct class and model its distribution using a bimodal Gaussian Mixture Model(GMM), representing balanced and imbalanced samples. Using Bayes' theorem, we estimate each sample's posterior probability of belonging to these two groups. Based on this, our adaptive loss (1) minimizes the overall Modality Gap, (2) aligns imbalanced samples with balanced ones, and (3) adaptively penalizes each according to its imbalance degree. A two-stage training strategy-warm-up and adaptive phases,yields state-of-the-art performance on CREMA-D (80.65%), AVE (70.40%), and KineticSound (72.42%). Fine-tuning with high-quality samples identified by the GMM further improves results, highlighting their value for effective multimodal fusion.
>
---
#### [replaced 005] PromptReverb: Multimodal Room Impulse Response Generation Through Latent Rectified Flow Matching
- **分类: cs.SD; cs.AI; I.2.6, H.5.5**

- **链接: [http://arxiv.org/pdf/2510.22439v2](http://arxiv.org/pdf/2510.22439v2)**

> **作者:** Ali Vosoughi; Yongyi Zang; Qihui Yang; Nathan Paek; Randal Leistikow; Chenliang Xu
>
> **备注:** 9 pages, 2 figures, 4 tables; v2: corrected spelling of a co-author name; no content changes
>
> **摘要:** Room impulse response (RIR) generation remains a critical challenge for creating immersive virtual acoustic environments. Current methods suffer from two fundamental limitations: the scarcity of full-band RIR datasets and the inability of existing models to generate acoustically accurate responses from diverse input modalities. We present PromptReverb, a two-stage generative framework that addresses these challenges. Our approach combines a variational autoencoder that upsamples band-limited RIRs to full-band quality (48 kHz), and a conditional diffusion transformer model based on rectified flow matching that generates RIRs from descriptions in natural language. Empirical evaluation demonstrates that PromptReverb produces RIRs with superior perceptual quality and acoustic accuracy compared to existing methods, achieving 8.8% mean RT60 error compared to -37% for widely used baselines and yielding more realistic room-acoustic parameters. Our method enables practical applications in virtual reality, architectural acoustics, and audio production where flexible, high-quality RIR synthesis is essential.
>
---
#### [replaced 006] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.01200v2](http://arxiv.org/pdf/2509.01200v2)**

> **作者:** Chenyang Le; Bing Han; Jinshun Li; Songyong Chen; Yanmin Qian
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Simultaneous Speech Translation (SimulST) enables real-time cross-lingual communication by jointly optimizing speech recognition and machine translation under strict latency constraints. Existing systems struggle to balance translation quality, latency, and semantic coherence, particularly in multilingual many-to-many scenarios where divergent read and write policies hinder unified strategy learning. In this paper, we present SimulMEGA (Simultaneous Generation by Mixture-of-Experts Gating), an unsupervised policy learning framework that combines prefix-based training with a Mixture-of-Experts refiner to learn effective read and write decisions in an implicit manner, without adding inference-time overhead. Our design requires only minimal modifications to standard transformer architectures and generalizes across both speech-to-text and text-to-speech streaming tasks. Through comprehensive evaluation on six language pairs, our 500M parameter speech-to-text model outperforms the Seamless baseline, achieving under 7 percent BLEU degradation at 1.5 seconds average lag and under 3 percent at 3 seconds. We further demonstrate the versatility of SimulMEGA by extending it to streaming TTS with a unidirectional backbone, yielding superior latency quality tradeoffs.
>
---
#### [replaced 007] Artificial Neural Networks Trained on Noisy Speech Exhibit the McGurk Effect
- **分类: cs.SD; cs.MM; cs.NE; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.05715v2](http://arxiv.org/pdf/2411.05715v2)**

> **作者:** Lukas Grasse; Matthew S. Tata
>
> **摘要:** Humans are able to fuse information from both auditory and visual modalities to help with understanding speech. This is demonstrated through a phenomenon known as the McGurk Effect, during which a listener is presented with incongruent auditory and visual speech that fuse together into the percept of illusory intermediate phonemes. Building on a recent framework that proposes how to address developmental 'why' questions using artificial neural networks, we evaluated a set of recent artificial neural networks trained on audiovisual speech by testing them with audiovisually incongruent words designed to elicit the McGurk effect. We show that networks trained entirely on congruent audiovisual speech nevertheless exhibit the McGurk percept. We further investigated 'why' by comparing networks trained on clean speech to those trained on noisy speech, and discovered that training with noisy speech led to a pronounced increase in both visual responses and McGurk responses across all models. Furthermore, we observed that systematically increasing the level of auditory noise during ANN training also increased the amount of audiovisual integration up to a point, but at extreme noise levels, this integration failed to develop. These results suggest that excessive noise exposure during critical periods of audiovisual learning may negatively influence the development of audiovisual speech integration. This work also demonstrates that the McGurk effect reliably emerges untrained from the behaviour of both supervised and unsupervised networks, even networks trained only on congruent speech. This supports the notion that artificial neural networks might be useful models for certain aspects of perception and cognition.
>
---
