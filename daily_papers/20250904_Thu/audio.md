# 音频 cs.SD;  eess.SP

- **最新发布 11 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Multi-level SSL Feature Gating for Audio Deepfake Detection
- **分类: cs.SD; cs.AI; cs.MM; I.2.7**

- **简介: 该论文提出多级SSL特征门控方法，用于音频深度伪造检测。针对现有方法泛化能力不足的问题，设计XLS-R前端提取特征，MultiConv后端捕捉局部/全局语音特征，并引入CKA增强特征多样性，实现跨领域检测的SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.03409v1](http://arxiv.org/pdf/2509.03409v1)**

> **作者:** Hoan My Tran; Damien Lolive; Aghilas Sini; Arnaud Delhay; Pierre-François Marteau; David Guennec
>
> **备注:** This paper has been accepted by ACM MM 2025
>
> **摘要:** Recent advancements in generative AI, particularly in speech synthesis, have enabled the generation of highly natural-sounding synthetic speech that closely mimics human voices. While these innovations hold promise for applications like assistive technologies, they also pose significant risks, including misuse for fraudulent activities, identity theft, and security threats. Current research on spoofing detection countermeasures remains limited by generalization to unseen deepfake attacks and languages. To address this, we propose a gating mechanism extracting relevant feature from the speech foundation XLS-R model as a front-end feature extractor. For downstream back-end classifier, we employ Multi-kernel gated Convolution (MultiConv) to capture both local and global speech artifacts. Additionally, we introduce Centered Kernel Alignment (CKA) as a similarity metric to enforce diversity in learned features across different MultiConv layers. By integrating CKA with our gating mechanism, we hypothesize that each component helps improving the learning of distinct synthetic speech patterns. Experimental results demonstrate that our approach achieves state-of-the-art performance on in-domain benchmarks while generalizing robustly to out-of-domain datasets, including multilingual speech samples. This underscores its potential as a versatile solution for detecting evolving speech deepfake threats.
>
---
#### [new 002] Analysis of Speaker Verification Performance Trade-offs with Neural Audio Codec Transmission
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究神经音频编解码器对说话人验证性能的影响，比较传统与NACs在不同比特率下的表现，发现NACs在低比特率下优于Opus，但高比特率下因丢失关键特征略有下降，建议开发说话人感知的NACs以优化性能。**

- **链接: [http://arxiv.org/pdf/2509.02771v1](http://arxiv.org/pdf/2509.02771v1)**

> **作者:** Nirmalya Mallick Thakur; Jia Qi Yip; Eng Siong Chng
>
> **备注:** Accepted by APSIPA ASC 2025
>
> **摘要:** Neural audio codecs (NACs) have made significant advancements in recent years and are rapidly being adopted in many audio processing pipelines. However, they can introduce audio distortions which degrade speaker verification (SV) performance. This study investigates the impact of both traditional and neural audio codecs at varying bitrates on three state of-the-art SV models evaluated on the VoxCeleb1 dataset. Our findings reveal a consistent degradation in SV performance across all models and codecs as bitrates decrease. Notably, NACs do not fundamentally break SV performance when compared to traditional codecs. They outperform Opus by 6-8% at low-bitrates (< 12 kbps) and remain marginally behind at higher bitrates ($\approx$ 24 kbps), with an EER increase of only 0.4-0.7%. The disparity at higher bitrates is likely due to the primary optimization of NACs for perceptual quality, which can inadvertently discard critical speaker-discriminative features, unlike Opus which was designed to preserve vocal characteristics. Our investigation suggests that NACs are a feasible alternative to traditional codecs, especially under bandwidth limitations. To bridge the gap at higher bitrates, future work should focus on developing speaker-aware NACs or retraining and adapting SV models.
>
---
#### [new 003] Speech DF Arena: A Leaderboard for Speech DeepFake Detection Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Speech DF Arena基准，包含14数据集和评估指标，用于比较语音深度伪造检测模型，解决缺乏标准化评估的问题。**

- **链接: [http://arxiv.org/pdf/2509.02859v1](http://arxiv.org/pdf/2509.02859v1)**

> **作者:** Sandipana Dowerah; Atharva Kulkarni; Ajinkya Kulkarni; Hoan My Tran; Joonas Kalda; Artem Fedorchenko; Benoit Fauve; Damien Lolive; Tanel Alumäe; Matthew Magimai Doss
>
> **摘要:** Parallel to the development of advanced deepfake audio generation, audio deepfake detection has also seen significant progress. However, a standardized and comprehensive benchmark is still missing. To address this, we introduce Speech DeepFake (DF) Arena, the first comprehensive benchmark for audio deepfake detection. Speech DF Arena provides a toolkit to uniformly evaluate detection systems, currently across 14 diverse datasets and attack scenarios, standardized evaluation metrics and protocols for reproducibility and transparency. It also includes a leaderboard to compare and rank the systems to help researchers and developers enhance their reliability and robustness. We include 14 evaluation sets, 12 state-of-the-art open-source and 3 proprietary detection systems. Our study presents many systems exhibiting high EER in out-of-domain scenarios, highlighting the need for extensive cross-domain evaluation. The leaderboard is hosted on Huggingface1 and a toolkit for reproducing results across the listed datasets is available on GitHub.
>
---
#### [new 004] Improving Perceptual Audio Aesthetic Assessment via Triplet Loss and Self-Supervised Embeddings
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出一种基于三元组损失和自监督嵌入的音频审美评估系统，解决生成音频跨领域质量预测问题。通过结合BEATs与多分支LSTM，优化嵌入空间结构，提升模型对合成数据的泛化能力，实现领域鲁棒的多维度音频质量评估。**

- **链接: [http://arxiv.org/pdf/2509.03292v1](http://arxiv.org/pdf/2509.03292v1)**

> **作者:** Dyah A. M. G. Wisnu; Ryandhimas E. Zezario; Stefano Rini; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted by IEEE Automatic Speech Recognition and Understanding Workshop(ASRU), 2025
>
> **摘要:** We present a system for automatic multi-axis perceptual quality prediction of generative audio, developed for Track 2 of the AudioMOS Challenge 2025. The task is to predict four Audio Aesthetic Scores--Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness--for audio generated by text-to-speech (TTS), text-to-audio (TTA), and text-to-music (TTM) systems. A main challenge is the domain shift between natural training data and synthetic evaluation data. To address this, we combine BEATs, a pretrained transformer-based audio representation model, with a multi-branch long short-term memory (LSTM) predictor and use a triplet loss with buffer-based sampling to structure the embedding space by perceptual similarity. Our results show that this improves embedding discriminability and generalization, enabling domain-robust audio quality assessment without synthetic training data.
>
---
#### [new 005] Comparison of End-to-end Speech Assessment Models for the NOCASA 2025 Challenge
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对儿童挪威语第二语言发音评估任务，比较了三种端到端模型，提出基于CTC的GOP特征与定制损失函数，实现高精度自动词级发音评分，超越基准模型并登顶挑战赛 leaderboard。**

- **链接: [http://arxiv.org/pdf/2509.03256v1](http://arxiv.org/pdf/2509.03256v1)**

> **作者:** Aleksei Žavoronkov; Tanel Alumäe
>
> **备注:** Published at IEEE MLSP 2025
>
> **摘要:** This paper presents an analysis of three end-to-end models developed for the NOCASA 2025 Challenge, aimed at automatic word-level pronunciation assessment for children learning Norwegian as a second language. Our models include an encoder-decoder Siamese architecture (E2E-R), a prefix-tuned direct classification model leveraging pretrained wav2vec2.0 representations, and a novel model integrating alignment-free goodness-of-pronunciation (GOP) features computed via CTC. We introduce a weighted ordinal cross-entropy loss tailored for optimizing metrics such as unweighted average recall and mean absolute error. Among the explored methods, our GOP-CTC-based model achieved the highest performance, substantially surpassing challenge baselines and attaining top leaderboard scores.
>
---
#### [new 006] A Study on Zero-Shot Non-Intrusive Speech Intelligibility for Hearing Aids Using Large Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出GPT-Whisper-HA模型，基于LLMs实现助听器零样本非侵入式语音可懂度评估，整合听力损失模拟、ASR模块及GPT-4o预测得分，实验显示RMSE提升2.59%，提升预测准确性。**

- **链接: [http://arxiv.org/pdf/2509.03021v1](http://arxiv.org/pdf/2509.03021v1)**

> **作者:** Ryandhimas E. Zezario; Dyah A. M. G. Wisnu; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to IEEE ICCE-TW 2025
>
> **摘要:** This work focuses on zero-shot non-intrusive speech assessment for hearing aids (HA) using large language models (LLMs). Specifically, we introduce GPT-Whisper-HA, an extension of GPT-Whisper, a zero-shot non-intrusive speech assessment model based on LLMs. GPT-Whisper-HA is designed for speech assessment for HA, incorporating MSBG hearing loss and NAL-R simulations to process audio input based on each individual's audiogram, two automatic speech recognition (ASR) modules for audio-to-text representation, and GPT-4o to predict two corresponding scores, followed by score averaging for the final estimated score. Experimental results indicate that GPT-Whisper-HA achieves a 2.59% relative root mean square error (RMSE) improvement over GPT-Whisper, confirming the potential of LLMs for zero-shot speech assessment in predicting subjective intelligibility for HA users.
>
---
#### [new 007] An Effective Strategy for Modeling Score Ordinality and Non-uniform Intervals in Automated Speaking Assessment
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文针对自动口语评估（ASA）中忽视分数序数结构与非均匀间隔的问题，提出结合自监督学习与手工特征的新方法，引入多边距序数损失以同时建模评分顺序与间隔差异。**

- **链接: [http://arxiv.org/pdf/2509.03372v1](http://arxiv.org/pdf/2509.03372v1)**

> **作者:** Tien-Hong Lo; Szu-Yu Chen; Yao-Ting Sung; Berlin Chen
>
> **备注:** Accepted at ASRU 2025
>
> **摘要:** A recent line of research on automated speaking assessment (ASA) has benefited from self-supervised learning (SSL) representations, which capture rich acoustic and linguistic patterns in non-native speech without underlying assumptions of feature curation. However, speech-based SSL models capture acoustic-related traits but overlook linguistic content, while text-based SSL models rely on ASR output and fail to encode prosodic nuances. Moreover, most prior arts treat proficiency levels as nominal classes, ignoring their ordinal structure and non-uniform intervals between proficiency labels. To address these limitations, we propose an effective ASA approach combining SSL with handcrafted indicator features via a novel modeling paradigm. We further introduce a multi-margin ordinal loss that jointly models both the score ordinality and non-uniform intervals of proficiency labels. Extensive experiments on the TEEMI corpus show that our method consistently outperforms strong baselines and generalizes well to unseen prompts.
>
---
#### [new 008] Non-Intrusive Intelligibility Prediction for Hearing Aids: Recent Advances, Trends, and Challenges
- **分类: eess.AS; cs.SD**

- **简介: 该论文综述非侵入式助听器语音可懂度预测研究，聚焦鲁棒特征提取、模型架构与泛化策略，旨在提升系统在复杂环境下的可靠性，同时指出数据集不足与跨场景泛化挑战。**

- **链接: [http://arxiv.org/pdf/2509.03017v1](http://arxiv.org/pdf/2509.03017v1)**

> **作者:** Ryandhimas E. Zezario
>
> **备注:** APSIPA ASC 2025 perspective paper
>
> **摘要:** This paper provides an overview of recent progress in non-intrusive speech intelligibility prediction for hearing aids (HA). We summarize developments in robust acoustic feature extraction, hearing loss modeling, and the use of emerging architectures for long-sequence processing. Listener-specific adaptation strategies and domain generalization approaches that aim to improve robustness in unseen acoustic environments are also discussed. Remaining challenges, such as the need for large-scale, diverse datasets and reliable cross-profile generalization, are acknowledged. Our goal is to offer a perspective on current trends, ongoing challenges, and possible future directions toward practical and reliable HA-oriented intelligibility prediction systems.
>
---
#### [new 009] IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文提出IS³网络，解决音频场景中瞬态与稳态声音分离问题，通过深度过滤和定制数据生成，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.02622v1](http://arxiv.org/pdf/2509.02622v1)**

> **作者:** Berger Clémentine; Stamadiatis Paraskevas; Badeau Roland; Essid Slim
>
> **摘要:** We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics. To this end, we introduce IS${}^3$, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics.
>
---
#### [new 010] Gaussian Process Regression of Steering Vectors With Physics-Aware Deep Composite Kernels for Augmented Listening
- **分类: eess.AS; cs.LG; cs.SD; eess.SP**

- **简介: 论文提出基于物理感知深度复合核与高斯过程回归的转向向量建模方法，解决声场散射效应下的空间音频表示问题。通过整合神经场与概率框架，实现数据稀疏条件下的超分辨率，提升空间滤波与双耳渲染性能。**

- **链接: [http://arxiv.org/pdf/2509.02571v1](http://arxiv.org/pdf/2509.02571v1)**

> **作者:** Diego Di Carlo; Koyama Shoichi; Nugraha Aditya Arie; Fontaine Mathieu; Bando Yoshiaki; Yoshii Kazuyoshi
>
> **摘要:** This paper investigates continuous representations of steering vectors over frequency and position of microphone and source for augmented listening (e.g., spatial filtering and binaural rendering) with precise control of the sound field perceived by the user. Steering vectors have typically been used for representing the spatial characteristics of the sound field as a function of the listening position. The basic algebraic representation of steering vectors assuming an idealized environment cannot deal with the scattering effect of the sound field. One may thus collect a discrete set of real steering vectors measured in dedicated facilities and super-resolve (i.e., upsample) them. Recently, physics-aware deep learning methods have been effectively used for this purpose. Such deterministic super-resolution, however, suffers from the overfitting problem due to the non-uniform uncertainty over the measurement space. To solve this problem, we integrate an expressive representation based on the neural field (NF) into the principled probabilistic framework based on the Gaussian process (GP). Specifically, we propose a physics-aware composite kernel that model the directional incoming waves and the subsequent scattering effect. Our comprehensive comparative experiment showed the effectiveness of the proposed method under data insufficiency conditions. In downstream tasks such as speech enhancement and binaural rendering using the simulated data of the SPEAR challenge, the oracle performances were attained with less than ten times fewer measurements.
>
---
#### [new 011] Speech Intelligibility Assessment with Uncertainty-Aware Whisper Embeddings and sLSTM
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对非侵入式语音可懂度预测任务，提出iMTI-Net模型。通过融合不确定性感知的Whisper嵌入与sLSTM网络，结合CNN与多任务学习框架，联合预测人类可懂度评分与机器WER，提升语音理解评估效果。**

- **链接: [http://arxiv.org/pdf/2509.03013v1](http://arxiv.org/pdf/2509.03013v1)**

> **作者:** Ryandhimas E. Zezario; Dyah A. M. G. Wisnu; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to APSIPA ASC 2025
>
> **摘要:** Non-intrusive speech intelligibility prediction remains challenging due to variability in speakers, noise conditions, and subjective perception. We propose an uncertainty-aware approach that leverages Whisper embeddings in combination with statistical features, specifically the mean, standard deviation, and entropy computed across the embedding dimensions. The entropy, computed via a softmax over the feature dimension, serves as a proxy for uncertainty, complementing global information captured by the mean and standard deviation. To model the sequential structure of speech, we adopt a scalar long short-term memory (sLSTM) network, which efficiently captures long-range dependencies. Building on this foundation, we propose iMTI-Net, an improved multi-target intelligibility prediction network that integrates convolutional neural network (CNN) and sLSTM components within a multitask learning framework. It jointly predicts human intelligibility scores and machine-based word error rates (WER) from Google ASR and Whisper. Experimental results show that iMTI-Net outperforms the original MTI-Net across multiple evaluation metrics, demonstrating the effectiveness of incorporating uncertainty-aware features and the CNN-sLSTM architecture.
>
---
## 更新

#### [replaced 001] Binaural Target Speaker Extraction using HRTFs
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.19369v2](http://arxiv.org/pdf/2507.19369v2)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **摘要:** In this work, we aim to imitate the human ability to selectively attend to a single speaker, even in the presence of multiple simultaneous talkers. To achieve this, we propose a novel approach for binaural target speaker extraction that leverages the listener's Head-Related Transfer Function (HRTF) to isolate the desired speaker. Notably, our method does not rely on speaker embeddings, making it speaker-independent and enabling strong generalization across multiple speech datasets and languages. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier transform (STFT) of the mixed audio signals, and compare it to a Real-Imaginary (RI)-based neural network, demonstrating the advantages of the former. We first evaluate the method in an anechoic, noise-free scenario, achieving excellent extraction performance while preserving the binaural cues of the target signal. We then extend the evaluation to reverberant conditions. Our method proves robust, maintaining speech clarity and source directionality while simultaneously reducing reverberation. A comparative analysis with existing binaural Target Speaker Extraction (TSE) methods demonstrates that our approach attains performance on par with competing techniques in terms of noise reduction and perceptual quality, while offering a clear advantage in preserving binaural cues.Demo-page: https://bi-ctse-hrtf.github.io
>
---
#### [replaced 002] You Sound a Little Tense: L2 Tailored Clear TTS Using Durational Vowel Properties
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.23367v2](http://arxiv.org/pdf/2506.23367v2)**

> **作者:** Paige Tuttösí; H. Henny Yeung; Yue Wang; Jean-Julien Aucouturier; Angelica Lim
>
> **备注:** Accepted to ISCA Speech Synthesis Workshop, 2025, Project webpage here: https://rosielab.github.io/clear_speech/ Code here: https://github.com/chocobearz/Matcha-TTS-L2-clarity
>
> **摘要:** We present the first text-to-speech (TTS) system tailored to second language (L2) speakers. We use duration differences between American English tense (longer) and lax (shorter) vowels to create a "clarity mode" for Matcha-TTS. Our perception studies showed that French-L1, English-L2 listeners had fewer (at least 9.15%) transcription errors when using our clarity mode, and found it more encouraging and respectful than overall slowed down speech. Remarkably, listeners were not aware of these effects: despite the decreased word error rate in clarity mode, listeners still believed that slowing all target words was the most intelligible, suggesting that actual intelligibility does not correlate with perceived intelligibility. Additionally, we found that Whisper-ASR did not use the same cues as L2 speakers to differentiate difficult vowels and is not sufficient to assess the intelligibility of TTS systems for these individuals.
>
---
#### [replaced 003] OSUM-EChat: Enhancing End-to-End Empathetic Spoken Chatbot via Understanding-Driven Spoken Dialogue
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2508.09600v2](http://arxiv.org/pdf/2508.09600v2)**

> **作者:** Xuelong Geng; Qijie Shao; Hongfei Xue; Shuiyuan Wang; Hanke Xie; Zhao Guo; Yi Zhao; Guojian Li; Wenjie Tian; Chengyou Wang; Zhixian Zhao; Kangxiang Xia; Ziyu Zhang; Zhennan Lin; Tianlun Zuo; Mingchen Shao; Yuang Cao; Guobin Ma; Longhao Li; Yuhang Dai; Dehui Gao; Dake Guo; Lei Xie
>
> **摘要:** Empathy is crucial in enabling natural interactions within spoken dialogue systems, allowing machines to recognize and respond appropriately to paralinguistic cues such as age, gender, and emotion. Recent advancements in end-to-end speech language models, which unify speech understanding and generation, provide promising solutions. However, several challenges persist, including an over-reliance on large-scale dialogue datasets, insufficient extraction of paralinguistic cues vital for conveying empathy, and the lack of empathy-specific datasets and evaluation frameworks. To address these issues, we introduce OSUM-EChat, an open-source, end-to-end spoken dialogue system designed to enhance empathetic interactions, particularly in resource-limited settings. OSUM-EChat introduces two key innovations: (1) a three-stage understanding-driven spoken dialogue training strategy that extends the capabilities of a large speech understanding model to spoken dialogue tasks, and (2) a linguistic-paralinguistic dual thinking mechanism that integrates paralinguistic understanding through a chain of thought with dialogue generation, enabling the system to produce more empathetic responses. This approach reduces reliance on large-scale dialogue datasets while maintaining high-quality empathetic interactions. Additionally, we introduce the EChat-200K dataset, a rich corpus of empathetic speech-to-speech dialogues, and the EChat-eval benchmark, a comprehensive framework for evaluating the empathetic capabilities of dialogue systems. Experimental results demonstrate that OSUM-EChat outperforms end-to-end spoken dialogue models regarding empathetic responsiveness, validating its effectiveness.
>
---
#### [replaced 004] FELLE: Autoregressive Speech Synthesis with Token-Wise Coarse-to-Fine Flow Matching
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.11128v2](http://arxiv.org/pdf/2502.11128v2)**

> **作者:** Hui Wang; Shujie Liu; Lingwei Meng; Jinyu Li; Yifan Yang; Shiwan Zhao; Haiyang Sun; Yanqing Liu; Haoqin Sun; Jiaming Zhou; Yan Lu; Yong Qin
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** To advance continuous-valued token modeling and temporal-coherence enforcement, we propose FELLE, an autoregressive model that integrates language modeling with token-wise flow matching. By leveraging the autoregressive nature of language models and the generative efficacy of flow matching, FELLE effectively predicts continuous-valued tokens (mel-spectrograms). For each continuous-valued token, FELLE modifies the general prior distribution in flow matching by incorporating information from the previous step, improving coherence and stability. Furthermore, to enhance synthesis quality, FELLE introduces a coarse-to-fine flow-matching mechanism, generating continuous-valued tokens hierarchically, conditioned on the language model's output. Experimental results demonstrate the potential of incorporating flow-matching techniques in autoregressive mel-spectrogram modeling, leading to significant improvements in TTS generation quality, as shown in https://aka.ms/felle.
>
---
#### [replaced 005] IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.21619v2](http://arxiv.org/pdf/2506.21619v2)**

> **作者:** Siyi Zhou; Yiquan Zhou; Yi He; Xun Zhou; Jinchao Wang; Wei Deng; Jingchen Shu
>
> **摘要:** Existing autoregressive large-scale text-to-speech (TTS) models have advantages in speech naturalness, but their token-by-token generation mechanism makes it difficult to precisely control the duration of synthesized speech. This becomes a significant limitation in applications requiring strict audio-visual synchronization, such as video dubbing. This paper introduces IndexTTS2, which proposes a novel, general, and autoregressive model-friendly method for speech duration control. The method supports two generation modes: one explicitly specifies the number of generated tokens to precisely control speech duration; the other freely generates speech in an autoregressive manner without specifying the number of tokens, while faithfully reproducing the prosodic features of the input prompt. Furthermore, IndexTTS2 achieves disentanglement between emotional expression and speaker identity, enabling independent control over timbre and emotion. In the zero-shot setting, the model can accurately reconstruct the target timbre (from the timbre prompt) while perfectly reproducing the specified emotional tone (from the style prompt). To enhance speech clarity in highly emotional expressions, we incorporate GPT latent representations and design a novel three-stage training paradigm to improve the stability of the generated speech. Additionally, to lower the barrier for emotional control, we designed a soft instruction mechanism based on text descriptions by fine-tuning Qwen3, effectively guiding the generation of speech with the desired emotional orientation. Finally, experimental results on multiple datasets show that IndexTTS2 outperforms state-of-the-art zero-shot TTS models in terms of word error rate, speaker similarity, and emotional fidelity. Audio samples are available at: https://index-tts.github.io/index-tts2.github.io/
>
---
#### [replaced 006] I2TTS: Image-indicated Immersive Text-to-speech Synthesis with Spatial Perception
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.13314v4](http://arxiv.org/pdf/2411.13314v4)**

> **作者:** Jiawei Zhang; Tian-Hao Zhang; Jun Wang; Jiaran Gao; Xinyuan Qian; Xu-Cheng Yin
>
> **备注:** Accepted by APSIPA ASC2025
>
> **摘要:** Controlling the style and characteristics of speech synthesis is crucial for adapting the output to specific contexts and user requirements. Previous Text-to-speech (TTS) works have focused primarily on the technical aspects of producing natural-sounding speech, such as intonation, rhythm, and clarity. However, they overlook the fact that there is a growing emphasis on spatial perception of synthesized speech, which may provide immersive experience in gaming and virtual reality. To solve this issue, in this paper, we present a novel multi-modal TTS approach, namely Image-indicated Immersive Text-to-speech Synthesis (I2TTS). Specifically, we introduce a scene prompt encoder that integrates visual scene prompts directly into the synthesis pipeline to control the speech generation process. Additionally, we propose a reverberation classification and refinement technique that adjusts the synthesized mel-spectrogram to enhance the immersive experience, ensuring that the involved reverberation condition matches the scene accurately. Experimental results demonstrate that our model achieves high-quality scene and spatial matching without compromising speech naturalness, marking a significant advancement in the field of context-aware speech synthesis. Project demo page: https://spatialTTS.github.io/ Index Terms-Speech synthesis, scene prompt, spatial perception
>
---
