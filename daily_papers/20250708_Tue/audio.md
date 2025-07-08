# 音频 cs.SD;  eess.SP

- **最新发布 43 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] EXPOTION: Facial Expression and Motion Control for Multimodal Music Generation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于多模态音乐生成任务，旨在通过面部表情和肢体动作控制生成同步且富有表现力的音乐。工作包括引入PEFT方法和时间对齐策略，提升音乐质量与视频同步性。**

- **链接: [http://arxiv.org/pdf/2507.04955v1](http://arxiv.org/pdf/2507.04955v1)**

> **作者:** Fathinah Izzati; Xinyue Li; Gus Xia
>
> **摘要:** We propose Expotion (Facial Expression and Motion Control for Multimodal Music Generation), a generative model leveraging multimodal visual controls - specifically, human facial expressions and upper-body motion - as well as text prompts to produce expressive and temporally accurate music. We adopt parameter-efficient fine-tuning (PEFT) on the pretrained text-to-music generation model, enabling fine-grained adaptation to the multimodal controls using a small dataset. To ensure precise synchronization between video and music, we introduce a temporal smoothing strategy to align multiple modalities. Experiments demonstrate that integrating visual features alongside textual descriptions enhances the overall quality of generated music in terms of musicality, creativity, beat-tempo consistency, temporal alignment with the video, and text adherence, surpassing both proposed baselines and existing state-of-the-art video-to-music generation models. Additionally, we introduce a novel dataset consisting of 7 hours of synchronized video recordings capturing expressive facial and upper-body gestures aligned with corresponding music, providing significant potential for future research in multimodal and interactive music generation.
>
---
#### [new 002] SAFERad: A Framework to Enable Radar Data for Safety-Relevant Perception Tasks
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于自动驾驶感知任务，解决雷达数据噪声问题。提出一种基于关键性评分的过滤框架，提升对潜在危险目标的检测准确率。**

- **链接: [http://arxiv.org/pdf/2507.03959v1](http://arxiv.org/pdf/2507.03959v1)**

> **作者:** Tim Brühl; Jenny Glönkler; Robin Schwager; Tin Stribor Sohn; Tim Dieter Eberhardt; Sören Hohmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Radar sensors play a crucial role for perception systems in automated driving but suffer from a high level of noise. In the past, this could be solved by strict filters, which remove most false positives at the expense of undetected objects. Future highly automated functions are much more demanding with respect to error rate. Hence, if the radar sensor serves as a component of perception systems for such functions, a simple filter strategy cannot be applied. In this paper, we present a modified filtering approach which is characterized by the idea to vary the filtering depending on the potential of harmful collision with the object which is potentially represented by the radar point. We propose an algorithm which determines a criticality score for each point based on the planned or presumable trajectory of the automated vehicle. Points identified as very critical can trigger manifold actions to confirm or deny object presence. Our pipeline introduces criticality regions. The filter threshold in these criticality regions is omitted. Commonly known radar data sets do not or barely feature critical scenes. Thus, we present an approach to evaluate our framework by adapting the planned trajectory towards vulnerable road users, which serve as ground truth critical points. Evaluation of the criticality metric prove high recall rates. Besides, our post-processing algorithm lowers the rate of non-clustered critical points by 74.8 % in an exemplary setup compared to a moderate, generic filter.
>
---
#### [new 003] Speaker-agnostic Emotion Vector for Cross-speaker Emotion Intensity Control
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感控制任务，解决跨说话人情感强度控制问题。提出一种与说话人无关的情感向量，实现情感强度可控且保持说话人一致性。**

- **链接: [http://arxiv.org/pdf/2507.03382v1](http://arxiv.org/pdf/2507.03382v1)**

> **作者:** Masato Murata; Koichi Miyazaki; Tomoki Koriyama
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Cross-speaker emotion intensity control aims to generate emotional speech of a target speaker with desired emotion intensities using only their neutral speech. A recently proposed method, emotion arithmetic, achieves emotion intensity control using a single-speaker emotion vector. Although this prior method has shown promising results in the same-speaker setting, it lost speaker consistency in the cross-speaker setting due to mismatches between the emotion vector of the source and target speakers. To overcome this limitation, we propose a speaker-agnostic emotion vector designed to capture shared emotional expressions across multiple speakers. This speaker-agnostic emotion vector is applicable to arbitrary speakers. Experimental results demonstrate that the proposed method succeeds in cross-speaker emotion intensity control while maintaining speaker consistency, speech quality, and controllability, even in the unseen speaker case.
>
---
#### [new 004] High-Resolution Sustain Pedal Depth Estimation from Piano Audio Across Room Acoustics
- **分类: cs.SD; cs.AI; cs.IR; eess.AS**

- **简介: 该论文属于音频处理任务，解决钢琴延音踏板深度估计问题。通过引入Transformer模型，实现连续深度预测，提升音乐表达的准确性。**

- **链接: [http://arxiv.org/pdf/2507.04230v1](http://arxiv.org/pdf/2507.04230v1)**

> **作者:** Kun Fang; Hanwen Zhang; Ziyu Wang; Ichiro Fujinaga
>
> **摘要:** Piano sustain pedal detection has previously been approached as a binary on/off classification task, limiting its application in real-world piano performance scenarios where pedal depth significantly influences musical expression. This paper presents a novel approach for high-resolution estimation that predicts continuous pedal depth values. We introduce a Transformer-based architecture that not only matches state-of-the-art performance on the traditional binary classification task but also achieves high accuracy in continuous pedal depth estimation. Furthermore, by estimating continuous values, our model provides musically meaningful predictions for sustain pedal usage, whereas baseline models struggle to capture such nuanced expressions with their binary detection approach. Additionally, this paper investigates the influence of room acoustics on sustain pedal estimation using a synthetic dataset that includes varied acoustic conditions. We train our model with different combinations of room settings and test it in an unseen new environment using a "leave-one-out" approach. Our findings show that the two baseline models and ours are not robust to unseen room conditions. Statistical analysis further confirms that reverberation influences model predictions and introduces an overestimation bias.
>
---
#### [new 005] Towards Human-in-the-Loop Onset Detection: A Transfer Learning Approach for Maracatu
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音乐起始点检测任务，旨在解决非西方音乐传统中数据不足的问题。通过迁移学习，利用少量标注数据优化模型，提升检测性能。**

- **链接: [http://arxiv.org/pdf/2507.04858v1](http://arxiv.org/pdf/2507.04858v1)**

> **作者:** António Sá Pinto
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** We explore transfer learning strategies for musical onset detection in the Afro-Brazilian Maracatu tradition, which features complex rhythmic patterns that challenge conventional models. We adapt two Temporal Convolutional Network architectures: one pre-trained for onset detection (intra-task) and another for beat tracking (inter-task). Using only 5-second annotated snippets per instrument, we fine-tune these models through layer-wise retraining strategies for five traditional percussion instruments. Our results demonstrate significant improvements over baseline performance, with F1 scores reaching up to 0.998 in the intra-task setting and improvements of over 50 percentage points in best-case scenarios. The cross-task adaptation proves particularly effective for time-keeping instruments, where onsets naturally align with beat positions. The optimal fine-tuning configuration varies by instrument, highlighting the importance of instrument-specific adaptation strategies. This approach addresses the challenges of underrepresented musical traditions, offering an efficient human-in-the-loop methodology that minimizes annotation effort while maximizing performance. Our findings contribute to more inclusive music information retrieval tools applicable beyond Western musical contexts.
>
---
#### [new 006] Direction Estimation of Sound Sources Using Microphone Arrays and Signal Strength
- **分类: cs.SD; cs.SY; eess.AS; eess.SY**

- **简介: 该论文属于声源方向估计任务，旨在解决声源定位中的方向识别问题。通过三麦克风阵列和信号强度分析，实现高精度、低成本的声源方向检测。**

- **链接: [http://arxiv.org/pdf/2507.03466v1](http://arxiv.org/pdf/2507.03466v1)**

> **作者:** Mahdi Ali Pour; Utku Gunay Acer
>
> **备注:** 5 pages
>
> **摘要:** Sound-tracking refers to the process of determining the direction from which a sound originates, making it a fundamental component of sound source localization. This capability is essential in a variety of applications, including security systems, acoustic monitoring, and speaker tracking, where accurately identifying the direction of a sound source enables real-time responses, efficient resource allocation, and improved situational awareness. While sound-tracking is closely related to localization, it specifically focuses on identifying the direction of the sound source rather than estimating its exact position in space. Despite its utility, sound-tracking systems face several challenges, such as maintaining directional accuracy and precision, along with the need for sophisticated hardware configurations and complex signal processing algorithms. This paper presents a sound-tracking method using three electret microphones. We estimate the direction of a sound source using a lightweight method that analyzes signals from three strategically placed microphones. By comparing the average power of the received signals, the system infers the most probable direction of the sound. The results indicate that the power level from each microphone effectively determines the sound source direction. Our system employs a straightforward and cost-effective hardware design, ensuring simplicity and affordability in implementation. It achieves a localization error of less than 6 degrees and a precision of 98%. Additionally, its effortless integration with various systems makes it versatile and adaptable. Consequently, this technique presents a robust and reliable solution for sound-tracking and localization, with potential applications spanning diverse domains such as security systems, smart homes, and acoustic monitoring.
>
---
#### [new 007] OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出OMAR-RQ，一种基于多特征掩码令牌预测的开源音乐音频表示模型，旨在提升音乐信息检索任务的性能。**

- **链接: [http://arxiv.org/pdf/2507.03482v1](http://arxiv.org/pdf/2507.03482v1)**

> **作者:** Pablo Alonso-Jiménez; Pedro Ramoneda; R. Oguz Araz; Andrea Poltronieri; Dmitry Bogdanov
>
> **摘要:** Developing open-source foundation models is essential for advancing research in music audio understanding and ensuring access to powerful, multipurpose representations for music information retrieval. We present OMAR-RQ, a model trained with self-supervision via masked token classification methodologies using a large-scale dataset with over 330,000 hours of music audio. We experiment with different input features and quantization options, and achieve state-of-the-art performance in music tagging, pitch estimation, chord recognition, beat tracking, segmentation, and difficulty estimation among open self-supervised models. We open-source our training and evaluation pipelines and model weights, available at https://github.com/mtg/omar-rq.
>
---
#### [new 008] Music Boomerang: Reusing Diffusion Models for Data Augmentation and Audio Manipulation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频处理任务，旨在通过扩散模型实现数据增强和音频操作。工作包括应用Boomerang采样改进节拍跟踪器性能，并尝试文本驱动的乐器替换。**

- **链接: [http://arxiv.org/pdf/2507.04864v1](http://arxiv.org/pdf/2507.04864v1)**

> **作者:** Alexander Fichtinger; Jan Schlüter; Gerhard Widmer
>
> **备注:** Accepted at SMC 2025. Code at https://malex1106.github.io/boomify/
>
> **摘要:** Generative models of music audio are typically used to generate output based solely on a text prompt or melody. Boomerang sampling, recently proposed for the image domain, allows generating output close to an existing example, using any pretrained diffusion model. In this work, we explore its application in the audio domain as a tool for data augmentation or content manipulation. Specifically, implementing Boomerang sampling for Stable Audio Open, we augment training data for a state-of-the-art beat tracker, and attempt to replace musical instruments in recordings. Our results show that the rhythmic structure of existing examples is mostly preserved, that it improves performance of the beat tracker, but only in scenarios of limited training data, and that it can accomplish text-based instrument replacement on monophonic inputs. We publish our implementation to invite experiments on data augmentation in other tasks and explore further applications.
>
---
#### [new 009] Differentiable High-Performance Ray Tracing-Based Simulation of Radio Propagation with Point Clouds
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于无线传播仿真任务，解决环境模型与电磁属性建模问题，提出基于点云的可微光线追踪模拟方法，实现高效多路径仿真。**

- **链接: [http://arxiv.org/pdf/2507.04021v1](http://arxiv.org/pdf/2507.04021v1)**

> **作者:** Niklas Vaara; Pekka Sangi; Miguel Bordallo López; Janne Heikkilä
>
> **摘要:** Ray tracing is a widely used deterministic method for radio propagation simulations, capable of producing physically accurate multipath components. The accuracy depends on the quality of the environment model and its electromagnetic properties. Recent advances in computer vision and machine learning have made it possible to reconstruct detailed environment models augmented with semantic segmentation labels. In this letter, we propose a differentiable ray tracing-based radio propagation simulator that operates directly on point clouds. We showcase the efficiency of our method by simulating multi-bounce propagation paths with up to five interactions with specular reflections and diffuse scattering in two indoor scenarios, each completing in less than 90 ms. Lastly, we demonstrate how the differentiability of electromagnetic computations can be combined with segmentation labels to learn the electromagnetic properties of the environment.
>
---
#### [new 010] Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决情感细微变化捕捉和跨数据集泛化问题。通过融合MFCC特征与1D-CNN及注意力机制，提升了识别精度。**

- **链接: [http://arxiv.org/pdf/2507.03251v1](http://arxiv.org/pdf/2507.03251v1)**

> **作者:** HyeYoung Lee; Muhammad Nadeem
>
> **摘要:** Speech Emotion Recognition (SER) traditionally relies on auditory data analysis for emotion classification. Several studies have adopted different methods for SER. However, existing SER methods often struggle to capture subtle emotional variations and generalize across diverse datasets. In this article, we use Mel-Frequency Cepstral Coefficients (MFCCs) as spectral features to bridge the gap between computational emotion processing and human auditory perception. To further improve robustness and feature diversity, we propose a novel 1D-CNN-based SER framework that integrates data augmentation techniques. MFCC features extracted from the augmented data are processed using a 1D Convolutional Neural Network (CNN) architecture enhanced with channel and spatial attention mechanisms. These attention modules allow the model to highlight key emotional patterns, enhancing its ability to capture subtle variations in speech signals. The proposed method delivers cutting-edge performance, achieving the accuracy of 97.49% for SAVEE, 99.23% for RAVDESS, 89.31% for CREMA-D, 99.82% for TESS, 99.53% for EMO-DB, and 96.39% for EMOVO. Experimental results show new benchmarks in SER, demonstrating the effectiveness of our approach in recognizing emotional expressions with high precision. Our evaluation demonstrates that the integration of advanced Deep Learning (DL) methods substantially enhances generalization across diverse datasets, underscoring their potential to advance SER for real-world deployment in assistive technologies and human-computer interaction.
>
---
#### [new 011] LAPS-Diff: A Diffusion-Based Framework for Singing Voice Synthesis With Language Aware Prosody-Style Guided Learning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于歌唱语音合成任务，旨在解决低资源场景下风格和语种特征捕捉难题。通过引入语言感知嵌入和风格引导学习机制，提升合成质量。**

- **链接: [http://arxiv.org/pdf/2507.04966v1](http://arxiv.org/pdf/2507.04966v1)**

> **作者:** Sandipan Dhar; Mayank Gupta; Preeti Rao
>
> **备注:** 10 pages, 5 figures, 3 Tables
>
> **摘要:** The field of Singing Voice Synthesis (SVS) has seen significant advancements in recent years due to the rapid progress of diffusion-based approaches. However, capturing vocal style, genre-specific pitch inflections, and language-dependent characteristics remains challenging, particularly in low-resource scenarios. To address this, we propose LAPS-Diff, a diffusion model integrated with language-aware embeddings and a vocal-style guided learning mechanism, specifically designed for Bollywood Hindi singing style. We curate a Hindi SVS dataset and leverage pre-trained language models to extract word and phone-level embeddings for an enriched lyrics representation. Additionally, we incorporated a style encoder and a pitch extraction model to compute style and pitch losses, capturing features essential to the naturalness and expressiveness of the synthesized singing, particularly in terms of vocal style and pitch variations. Furthermore, we utilize MERT and IndicWav2Vec models to extract musical and contextual embeddings, serving as conditional priors to refine the acoustic feature generation process further. Based on objective and subjective evaluations, we demonstrate that LAPS-Diff significantly improves the quality of the generated samples compared to the considered state-of-the-art (SOTA) model for our constrained dataset that is typical of the low resource scenario.
>
---
#### [new 012] TTS-CtrlNet: Time varying emotion aligned text-to-speech generation with ControlNet
- **分类: cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决时间变化情感控制难题。提出TTS-CtrlNet方法，在不改变原模型的前提下，实现高效、可扩展的情感控制。**

- **链接: [http://arxiv.org/pdf/2507.04349v1](http://arxiv.org/pdf/2507.04349v1)**

> **作者:** Jaeseok Jeong; Yuna Lee; Mingi Kwon; Youngjung Uh
>
> **摘要:** Recent advances in text-to-speech (TTS) have enabled natural speech synthesis, but fine-grained, time-varying emotion control remains challenging. Existing methods often allow only utterance-level control and require full model fine-tuning with a large emotion speech dataset, which can degrade performance. Inspired by adding conditional control to the existing model in ControlNet (Zhang et al, 2023), we propose the first ControlNet-based approach for controllable flow-matching TTS (TTS-CtrlNet), which freezes the original model and introduces a trainable copy of it to process additional conditions. We show that TTS-CtrlNet can boost the pretrained large TTS model by adding intuitive, scalable, and time-varying emotion control while inheriting the ability of the original model (e.g., zero-shot voice cloning & naturalness). Furthermore, we provide practical recipes for adding emotion control: 1) optimal architecture design choice with block analysis, 2) emotion-specific flow step, and 3) flexible control scale. Experiments show that ours can effectively add an emotion controller to existing TTS, and achieves state-of-the-art performance with emotion similarity scores: Emo-SIM and Aro-Val SIM. The project page is available at: https://curryjung.github.io/ttsctrlnet_project_page
>
---
#### [new 013] Fast-VGAN: Lightweight Voice Conversion with Explicit Control of F0 and Duration Parameters
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音转换任务，旨在实现对音高和时长的精确控制。通过构建轻量级模型，直接调节F0和发音序列，提升转换效果与灵活性。**

- **链接: [http://arxiv.org/pdf/2507.04817v1](http://arxiv.org/pdf/2507.04817v1)**

> **作者:** Mathilde Abrassart; Nicolas Obin; Axel Roebel
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Precise control over speech characteristics, such as pitch, duration, and speech rate, remains a significant challenge in the field of voice conversion. The ability to manipulate parameters like pitch and syllable rate is an important element for effective identity conversion, but can also be used independently for voice transformation, achieving goals that were historically addressed by vocoder-based methods. In this work, we explore a convolutional neural network-based approach that aims to provide means for modifying fundamental frequency (F0), phoneme sequences, intensity, and speaker identity. Rather than relying on disentanglement techniques, our model is explicitly conditioned on these factors to generate mel spectrograms, which are then converted into waveforms using a universal neural vocoder. Accordingly, during inference, F0 contours, phoneme sequences, and speaker embeddings can be freely adjusted, allowing for intuitively controlled voice transformations. We evaluate our approach on speaker conversion and expressive speech tasks using both perceptual and objective metrics. The results suggest that the proposed method offers substantial flexibility, while maintaining high intelligibility and speaker similarity.
>
---
#### [new 014] Improving BERT for Symbolic Music Understanding Using Token Denoising and Pianoroll Prediction
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐理解任务，旨在提升BERT模型在符号音乐处理上的性能。通过引入token去噪和钢琴卷预测作为预训练目标，增强模型对音乐知识的学习能力。**

- **链接: [http://arxiv.org/pdf/2507.04776v1](http://arxiv.org/pdf/2507.04776v1)**

> **作者:** Jun-You Wang; Li Su
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** We propose a pre-trained BERT-like model for symbolic music understanding that achieves competitive performance across a wide range of downstream tasks. To achieve this target, we design two novel pre-training objectives, namely token correction and pianoroll prediction. First, we sample a portion of note tokens and corrupt them with a limited amount of noise, and then train the model to denoise the corrupted tokens; second, we also train the model to predict bar-level and local pianoroll-derived representations from the corrupted note tokens. We argue that these objectives guide the model to better learn specific musical knowledge such as pitch intervals. For evaluation, we propose a benchmark that incorporates 12 downstream tasks ranging from chord estimation to symbolic genre classification. Results confirm the effectiveness of the proposed pre-training objectives on downstream tasks.
>
---
#### [new 015] CLEP-DG: Contrastive Learning for Speech Emotion Domain Generalization via Soft Prompt Tuning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决模型在不同声学条件下的泛化问题。通过对比学习和软提示微调增强模型鲁棒性，提升情感识别效果。**

- **链接: [http://arxiv.org/pdf/2507.04048v1](http://arxiv.org/pdf/2507.04048v1)**

> **作者:** Jiacheng Shi; Yanfu Zhang; Ye Gao
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** Speech Emotion Recognition (SER) is fundamental to affective computing and human-computer interaction, yet existing models struggle to generalize across diverse acoustic conditions. While Contrastive Language-Audio Pretraining (CLAP) provides strong multimodal alignment, it lacks dedicated mechanisms for capturing emotional cues, making it suboptimal for SER. To address this, we propose CLEP-DG, a framework that enhances CLAP's robustness in emotion recognition. First, we fine-tune CLAP to obtain CLEP, adapting it on large-scale emotional speech datasets to better encode emotion-relevant features. Then, we introduce Acoustic Context Prompt Tuning (ACPT), a text-driven augmentation strategy that optimizes learnable prompt vectors to model diverse acoustic environments without additional labeled audio. Finally, leveraging cross-modal transferability, we train a classifier on text-derived embeddings and apply it to the audio encoder during inference, mitigating domain shifts between textual supervision and audio-based emotion recognition. Experiments across five benchmark datasets show that CLEP-DG outperforms prior CLAP-based approaches, achieving state-of-the-art performance in both supervised and domain generalization settings.
>
---
#### [new 016] RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决帕金森病早期检测问题。提出RECA-PD方法，结合可解释特征与自监督表示，提升模型可解释性与性能。**

- **链接: [http://arxiv.org/pdf/2507.03594v1](http://arxiv.org/pdf/2507.03594v1)**

> **作者:** Terry Yi Zhong; Cristian Tejedor-Garcia; Martha Larson; Bastiaan R. Bloem
>
> **备注:** Accepted for TSD 2025
>
> **摘要:** Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Speech-based PD detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance.
>
---
#### [new 017] Robust Localization of Partially Fake Speech: Metrics, Models, and Out-of-Domain Evaluation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音防伪任务，解决部分虚假语音定位问题。针对现有方法在域外表现差的问题，提出使用序列异常检测和阈值相关指标，评估模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.03468v1](http://arxiv.org/pdf/2507.03468v1)**

> **作者:** Hieu-Thi Luong; Inbal Rimons; Haim Permuter; Kong Aik Lee; Eng Siong Chng
>
> **备注:** Submitted to APSIPA 2025
>
> **摘要:** Partial audio deepfake localization pose unique challenges and remain underexplored compared to full-utterance spoofing detection. While recent methods report strong in-domain performance, their real-world utility remains unclear. In this analysis, we critically examine the limitations of current evaluation practices, particularly the widespread use of Equal Error Rate (EER), which often obscures generalization and deployment readiness. We propose reframing the localization task as a sequential anomaly detection problem and advocate for the use of threshold-dependent metrics such as accuracy, precision, recall, and F1-score, which better reflect real-world behavior. Specifically, we analyze the performance of the open-source Coarse-to-Fine Proposal Refinement Framework (CFPRF), which achieves a 20-ms EER of 7.61% on the in-domain PartialSpoof evaluation set, but 43.25% and 27.59% on the LlamaPartialSpoof and Half-Truth out-of-domain test sets. Interestingly, our reproduced version of the same model performs worse on in-domain data (9.84%) but better on the out-of-domain sets (41.72% and 14.98%, respectively). This highlights the risks of over-optimizing for in-domain EER, which can lead to models that perform poorly in real-world scenarios. It also suggests that while deep learning models can be effective on in-domain data, they generalize poorly to out-of-domain scenarios, failing to detect novel synthetic samples and misclassifying unfamiliar bona fide audio. Finally, we observe that adding more bona fide or fully synthetic utterances to the training data often degrades performance, whereas adding partially fake utterances improves it.
>
---
#### [new 018] Eigenvoice Synthesis based on Model Editing for Speaker Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在生成未见过的说话人声音。解决如何定义有效说话人空间的问题，通过在DNN参数空间中定义说话人空间并直接采样生成多样语音。**

- **链接: [http://arxiv.org/pdf/2507.03377v1](http://arxiv.org/pdf/2507.03377v1)**

> **作者:** Masato Murata; Koichi Miyazaki; Tomoki Koriyama; Tomoki Toda
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Speaker generation task aims to create unseen speaker voice without reference speech. The key to the task is defining a speaker space that represents diverse speakers to determine the generated speaker trait. However, the effective way to define this speaker space remains unclear. Eigenvoice synthesis is one of the promising approaches in the traditional parametric synthesis framework, such as HMM-based methods, which define a low-dimensional speaker space using pre-stored speaker features. This study proposes a novel DNN-based eigenvoice synthesis method via model editing. Unlike prior methods, our method defines a speaker space in the DNN model parameter space. By directly sampling new DNN model parameters in this space, we can create diverse speaker voices. Experimental results showed the capability of our method to generate diverse speakers' speech. Moreover, we discovered a gender-dominant axis in the created speaker space, indicating the potential to control speaker attributes.
>
---
#### [new 019] Machine Learning in Acoustics: A Review and Open-Source Repository
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于机器学习在声学领域的应用任务，旨在解决声学数据分析与模式识别问题，通过开源工具展示ML技术的实践效果。**

- **链接: [http://arxiv.org/pdf/2507.04419v1](http://arxiv.org/pdf/2507.04419v1)**

> **作者:** Ryan A. McCarthy; You Zhang; Samuel A. Verburg; William F. Jenkins; Peter Gerstoft
>
> **备注:** Accepted by npj Acoustics, 22 pages, 12 figures
>
> **摘要:** Acoustic data provide scientific and engineering insights in fields ranging from bioacoustics and communications to ocean and earth sciences. In this review, we survey recent advances and the transformative potential of machine learning (ML) in acoustics, including deep learning (DL). Using the Python high-level programming language, we demonstrate a broad collection of ML techniques to detect and find patterns for classification, regression, and generation in acoustics data automatically. We have ML examples including acoustic data classification, generative modeling for spatial audio, and physics-informed neural networks. This work includes AcousticsML, a set of practical Jupyter notebook examples on GitHub demonstrating ML benefits and encouraging researchers and practitioners to apply reproducible data-driven approaches to acoustic challenges.
>
---
#### [new 020] Multi-Step Prediction and Control of Hierarchical Emotion Distribution in Text-to-Speech Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在解决情感表达的多层级控制问题。通过引入多步骤层次化情感分布预测模块，提升情感表现力与控制精度。**

- **链接: [http://arxiv.org/pdf/2507.04598v1](http://arxiv.org/pdf/2507.04598v1)**

> **作者:** Sho Inoue; Kun Zhou; Shuai Wang; Haizhou Li
>
> **备注:** Accepted to APSIPA Transactions on Signal and Information Processing
>
> **摘要:** We investigate hierarchical emotion distribution (ED) for achieving multi-level quantitative control of emotion rendering in text-to-speech synthesis (TTS). We introduce a novel multi-step hierarchical ED prediction module that quantifies emotion variance at the utterance, word, and phoneme levels. By predicting emotion variance in a multi-step manner, we leverage global emotional context to refine local emotional variations, thereby capturing the intrinsic hierarchical structure of speech emotion. Our approach is validated through its integration into a variance adaptor and an external module design compatible with various TTS systems. Both objective and subjective evaluations demonstrate that the proposed framework significantly enhances emotional expressiveness and enables precise control of emotion rendering across multiple speech granularities.
>
---
#### [new 021] Robust Node Localization for Rough and Extreme Deployment Environments
- **分类: eess.SP; cs.RO; math.OC**

- **简介: 该论文属于无线传感器网络定位任务，解决恶劣环境下节点定位误差问题。通过压缩感知方法和优化锚点配置，实现可靠定位。**

- **链接: [http://arxiv.org/pdf/2507.03856v1](http://arxiv.org/pdf/2507.03856v1)**

> **作者:** Abiy Tasissa; Waltenegus Dargie
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Many applications have been identified which require the deployment of large-scale low-power wireless sensor networks. Some of the deployment environments, however, impose harsh operation conditions due to intense cross-technology interference, extreme weather conditions (heavy rainfall, excessive heat, etc.), or rough motion, thereby affecting the quality and predictability of the wireless links the nodes establish. In localization tasks, these conditions often lead to significant errors in estimating the position of target nodes. Motivated by the practical deployments of sensors on the surface of different water bodies, we address the problem of identifying susceptible nodes and robustly estimating their positions. We formulate these tasks as a compressive sensing problem and propose algorithms for both node identification and robust estimation. Additionally, we design an optimal anchor configuration to maximize the robustness of the position estimation task. Our numerical results and comparisons with competitive methods demonstrate that the proposed algorithms achieve both objectives with a modest number of anchors. Since our method relies only on target-to-anchor distances, it is broadly applicable and yields resilient, robust localization.
>
---
#### [new 022] Self-supervised learning of speech representations with Dutch archival data
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音表示学习任务，旨在利用荷兰广播数据提升wav2vec 2.0模型性能。解决数据质量与预处理问题，探索单语与多语训练效果，最终获得更优的荷兰语模型。**

- **链接: [http://arxiv.org/pdf/2507.04554v1](http://arxiv.org/pdf/2507.04554v1)**

> **作者:** Nik Vaessen; David A. van Leeuwen; Roeland Ordelman
>
> **备注:** accepted at interspeech 2025
>
> **摘要:** This paper explores the use of Dutch archival television broadcast data for self-supervised learning of speech foundation models, specifically wav2vec 2.0. We first study data quality assumptions for pre-training, and show how music, noise and speaker overlap affect SSL convergence and downstream fine-tuning performance. Secondly, we explore effectively pre-processing strategies to convert the noisy broadcast dataset into a qualitative dataset for pre-training, by using Whisper and WhisperX., Thirdly, we compare mono-lingual and multi-lingual pre-training with equivalent amounts of data, and show that mono-lingual pre-training is more robust to out-of-domain data. Lastly, we achieve a state-of-the-art LARGE wav2vec 2.0 model for the Dutch language, by a continuation of pre-training a wav2vec 2.0 XLS-R model checkpoint with our 55k hour archival dataset.
>
---
#### [new 023] Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.SP**

- **简介: 该论文提出Audio-JEPA，用于音频表征学习，解决自监督音频特征提取问题。通过预测频谱图块的潜在表示，实现高效预训练。**

- **链接: [http://arxiv.org/pdf/2507.02915v1](http://arxiv.org/pdf/2507.02915v1)**

> **作者:** Ludovic Tuncay; Etienne Labbé; Emmanouil Benetos; Thomas Pellegrini
>
> **摘要:** Building on the Joint-Embedding Predictive Architecture (JEPA) paradigm, a recent self-supervised learning framework that predicts latent representations of masked regions in high-level feature spaces, we propose Audio-JEPA (Audio Joint-Embedding Predictive Architecture), tailored specifically for audio data. Audio-JEPA uses a simple Vision Transformer backbone to predict latent representations of masked spectrogram patches rather than reconstructing raw audio. We pre-train on unlabeled AudioSet clips (10s, 32kHz) with random patch masking on mel-spectrograms. We evaluate on the X-ARES suite covering speech, music, and environmental sound tasks. Although our implementation is a straightforward translation of the original model to audio, the results still show comparable performance to wav2vec 2.0 and data2vec while using less than one-fifth of their training data and with no hyper-parameter tuning. All code and pretrained checkpoints will be released on GitHub.
>
---
#### [new 024] Modeling the Difficulty of Saxophone Music
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决风乐器乐谱难度评估问题。通过分析音符转换成本，建立难度模型，适用于萨克斯等木管乐器。**

- **链接: [http://arxiv.org/pdf/2507.04963v1](http://arxiv.org/pdf/2507.04963v1)**

> **作者:** Šimon Libřický; Jan Hajič jr
>
> **摘要:** In learning music, difficulty is an important factor in choice of repertoire, choice of tempo, and structure of practice. These choices are typically done with the guidance of a teacher; however, not all learners have access to one. While piano and strings have had some attention devoted to automated difficulty estimation, wind instruments have so far been under-served. In this paper, we propose a method for estimating the difficulty of pieces for winds and implement it for the tenor saxophone. We take the cost-of-traversal approach, modelling the part as a sequence of transitions -- note pairs. We estimate transition costs from newly collected recordings of trill speeds, comparing representations of saxophone fingerings at various levels of expert input. We then compute and visualise the cost of the optimal path through the part, at a given tempo. While we present this model for the tenor saxophone, the same pipeline can be applied to other woodwinds, and our experiments show that with appropriate feature design, only a small proportion of possible trills is needed to estimate the costs well. Thus, we present a practical way of diversifying the capabilities of MIR in music education to the wind family of instruments.
>
---
#### [new 025] MaskBeat: Loopable Drum Beat Generation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在解决 drum beat 的循环生成问题。提出 MaskBeat 方法，通过双向注意力和自定义损失函数，实现并行且连贯的鼓点生成。**

- **链接: [http://arxiv.org/pdf/2507.03395v1](http://arxiv.org/pdf/2507.03395v1)**

> **作者:** Luca A. Lanzendörfer; Florian Grötschla; Karim Galal; Roger Wattenhofer
>
> **备注:** Extended Abstract ISMIR 2025
>
> **摘要:** We present MaskBeat, a transformer-based approach for loopable drum pattern generation. Rather than predicting drum hits sequentially, our method uses bidirectional attention with iterative refinement, allowing instruments to be generated in parallel while maintaining musical coherence. Additionally, we introduce custom loss functions that capture drum-specific musical relationships. Our experiments show that MaskBeat generates higher quality and more musically coherent drum patterns than baseline approaches.
>
---
#### [new 026] MusGO: A Community-Driven Framework For Assessing Openness in Music-Generative AI
- **分类: cs.SD; cs.AI; cs.CY; eess.AS**

- **简介: 该论文属于音乐生成AI的开放性评估任务，旨在解决“开放模型”定义模糊的问题。通过构建MusGO框架，评估16个模型并建立开放排行榜。**

- **链接: [http://arxiv.org/pdf/2507.03599v1](http://arxiv.org/pdf/2507.03599v1)**

> **作者:** Roser Batlle-Roca; Laura Ibáñez-Martínez; Xavier Serra; Emilia Gómez; Martín Rocamora
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Since 2023, generative AI has rapidly advanced in the music domain. Despite significant technological advancements, music-generative models raise critical ethical challenges, including a lack of transparency and accountability, along with risks such as the replication of artists' works, which highlights the importance of fostering openness. With upcoming regulations such as the EU AI Act encouraging open models, many generative models are being released labelled as 'open'. However, the definition of an open model remains widely debated. In this article, we adapt a recently proposed evidence-based framework for assessing openness in LLMs to the music domain. Using feedback from a survey of 110 participants from the Music Information Retrieval (MIR) community, we refine the framework into MusGO (Music-Generative Open AI), which comprises 13 openness categories: 8 essential and 5 desirable. We evaluate 16 state-of-the-art generative models and provide an openness leaderboard that is fully open to public scrutiny and community contributions. Through this work, we aim to clarify the concept of openness in music-generative AI and promote its transparent and responsible development.
>
---
#### [new 027] Evaluation of an Uncertainty-Aware Late Fusion Algorithm for Multi-Source Bird's Eye View Detections Under Controlled Noise
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于多源感知融合任务，解决检测误差对融合性能评估的影响。通过引入可控噪声，提出UniKF算法提升BEV检测融合效果。**

- **链接: [http://arxiv.org/pdf/2507.03381v1](http://arxiv.org/pdf/2507.03381v1)**

> **作者:** Maryem Fadili; Louis Lecrosnier; Steve Pechberti; Redouane Khemmar
>
> **摘要:** Reliable multi-source fusion is crucial for robust perception in autonomous systems. However, evaluating fusion performance independently of detection errors remains challenging. This work introduces a systematic evaluation framework that injects controlled noise into ground-truth bounding boxes to isolate the fusion process. We then propose Unified Kalman Fusion (UniKF), a late-fusion algorithm based on Kalman filtering to merge Bird's Eye View (BEV) detections while handling synchronization issues. Experiments show that UniKF outperforms baseline methods across various noise levels, achieving up to 3x lower object's positioning and orientation errors and 2x lower dimension estimation errors, while maintaining nearperfect precision and recall between 99.5% and 100%.
>
---
#### [new 028] Navigating Speech Recording Collections with AI-Generated Illustrations
- **分类: cs.IR; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于语音信息导航任务，旨在解决从大量语音数据中高效检索和探索的问题。通过生成图像和思维导图，提升用户对语音档案的浏览体验。**

- **链接: [http://arxiv.org/pdf/2507.04182v1](http://arxiv.org/pdf/2507.04182v1)**

> **作者:** Sirina Håland; Trond Karlsen Strøm; Petra Galuščáková
>
> **摘要:** Although the amount of available spoken content is steadily increasing, extracting information and knowledge from speech recordings remains challenging. Beyond enhancing traditional information retrieval methods such as speech search and keyword spotting, novel approaches for navigating and searching spoken content need to be explored and developed. In this paper, we propose a novel navigational method for speech archives that leverages recent advances in language and multimodal generative models. We demonstrate our approach with a Web application that organizes data into a structured format using interactive mind maps and image generation tools. The system is implemented using the TED-LIUM~3 dataset, which comprises over 2,000 speech transcripts and audio files of TED Talks. Initial user tests using a System Usability Scale (SUS) questionnaire indicate the application's potential to simplify the exploration of large speech collections.
>
---
#### [new 029] A Unified Speech LLM for Diarization and Speech Recognition in Multilingual Conversations
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文聚焦多语言对话中的语音识别与说话人分离任务，提出统一的语音大模型解决数据不足导致的性能问题，实现端到端联合处理。**

- **链接: [http://arxiv.org/pdf/2507.02927v1](http://arxiv.org/pdf/2507.02927v1)**

> **作者:** Phurich Saengthong; Boonnithi Jiaramaneepinit; Sheng Li; Manabu Okumura; Takahiro Shinozaki
>
> **摘要:** Speech Large Language Models (Speech LLMs) have emerged as a crucial paradigm in recent years, extending the capabilities of traditional LLMs to speech tasks such as automatic speech recognition (ASR) and spoken dialogue modeling. However, their effectiveness in real-world multilingual conversations remains limited by the scarcity of data that captures natural conversational phenomena. To address this, the MLC-SLM Challenge provides a multilingual conversational dataset and evaluates models on two tasks: ASR with oracle segmentation (Task I) and joint diarization and recognition without oracle information (Task II). In this paper, we focus on Task II and propose a unified speech LLM that jointly performs diarization and ASR in an end-to-end manner. By reformulating the training data format and modifying the inference procedure, our model addresses the ambiguity inherent in pre-segmented audio and achieves a 54.87\% relative improvement in tcpWER/tcpCER over the baseline, ranking 8th overall, despite using a smaller LLM backbone. We also report results from Task I using a fine-tuned speech LLM.
>
---
#### [new 030] Hear-Your-Click: Interactive Video-to-Audio Generation via Object-aware Contrastive Audio-Visual Fine-tuning
- **分类: cs.CV; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于视频到音频生成任务，解决现有方法难以精准生成特定物体声音的问题。通过交互式点击和对象感知的音频视觉对齐技术实现更精确的音频生成。**

- **链接: [http://arxiv.org/pdf/2507.04959v1](http://arxiv.org/pdf/2507.04959v1)**

> **作者:** Yingshan Liang; Keyu Fan; Zhicheng Du; Yiran Wang; Qingyang Shi; Xinyu Zhang; Jiasheng Lu; Peiwu Qin
>
> **摘要:** Video-to-audio (V2A) generation shows great potential in fields such as film production. Despite significant advances, current V2A methods, which rely on global video information, struggle with complex scenes and often fail to generate audio tailored to specific objects or regions in the videos. To address these limitations, we introduce Hear-Your-Click, an interactive V2A framework that enables users to generate sounds for specific objects in the videos by simply clicking on the frame. To achieve this, we propose Object-aware Contrastive Audio-Visual Fine-tuning (OCAV) with a Mask-guided Visual Encoder (MVE) to obtain object-level visual features aligned with corresponding audio segments. Furthermore, we tailor two data augmentation strategies: Random Video Stitching (RVS) and Mask-guided Loudness Modulation (MLM), aimed at enhancing the model's sensitivity to the segmented objects. To effectively measure the audio-visual correspondence, we design a new evaluation metric, the CAV score, for evaluation. Extensive experiments demonstrate that our framework offers more precise control and improved generation performance across various metrics. Project Page: https://github.com/SynapGrid/Hear-Your-Click
>
---
#### [new 031] Latent FxLMS: Accelerating Active Noise Control with Neural Adaptive Filters
- **分类: cs.LG; cs.SD; cs.SY; eess.AS; eess.SY; nlin.AO; stat.ML**

- **简介: 该论文属于主动降噪任务，旨在加速FxLMS算法。通过引入神经自编码器约束滤波器权重，使模型在潜在空间中更新，提升收敛速度。**

- **链接: [http://arxiv.org/pdf/2507.03854v1](http://arxiv.org/pdf/2507.03854v1)**

> **作者:** Kanad Sarkar; Austin Lu; Manan Mittal; Yongjie Zhuang; Ryan Corey; Andrew Singer
>
> **备注:** 8 pages, Submitted at Forum Acousticum Euronoise 2025
>
> **摘要:** Filtered-X LMS (FxLMS) is commonly used for active noise control (ANC), wherein the soundfield is minimized at a desired location. Given prior knowledge of the spatial region of the noise or control sources, we could improve FxLMS by adapting along the low-dimensional manifold of possible adaptive filter weights. We train an auto-encoder on the filter coefficients of the steady-state adaptive filter for each primary source location sampled from a given spatial region and constrain the weights of the adaptive filter to be the output of the decoder for a given state of latent variables. Then, we perform updates in the latent space and use the decoder to generate the cancellation filter. We evaluate how various neural network constraints and normalization techniques impact the convergence speed and steady-state mean squared error. Under certain conditions, our Latent FxLMS model converges in fewer steps with comparable steady-state error to the standard FxLMS.
>
---
#### [new 032] OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于情感语音交互任务，旨在解决 empathetic LSLMs 闭源问题。提出 OpenS2S 模型，实现透明、低延迟的情感语音生成与训练。**

- **链接: [http://arxiv.org/pdf/2507.05177v1](http://arxiv.org/pdf/2507.05177v1)**

> **作者:** Chen Wang; Tianyu Peng; Wen Yang; Yinan Bai; Guangfu Wang; Jun Lin; Lanpeng Jia; Lingxiang Wu; Jinqiao Wang; Chengqing Zong; Jiajun Zhang
>
> **备注:** Technical Report
>
> **摘要:** Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at https://casia-lm.github.io/OpenS2S
>
---
#### [new 033] The Overview of Segmental Durations Modification Algorithms on Speech Signal Characteristics
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音信号处理任务，旨在解决在不改变语音本质属性的情况下任意修改语音段持续时间的问题，研究了多种相关算法。**

- **链接: [http://arxiv.org/pdf/2507.04264v1](http://arxiv.org/pdf/2507.04264v1)**

> **作者:** Kyeomeun Jang; Jiaying Li; Yinuo Wang
>
> **摘要:** This paper deeply evaluates and analyzes several mainstream algorithms that can arbitrarily modify the duration of any portion of a given speech signal without changing the essential properties (e.g., pitch contour, power spectrum, etc.) of the original signal. Arbitrary modification in this context means that the duration of any region of the signal can be changed by specifying the starting and ending time for modification or the target duration of the specified interval, which can be either a fixed value of duration in the time domain or a scaling factor of the original duration. In addition, arbitrary modification also indicates any number of intervals can be modified at the same time.
>
---
#### [new 034] DiceHuBERT: Distilling HuBERT with a Self-Supervised Learning Objective
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在压缩HuBERT模型。通过自监督学习目标进行知识蒸馏，提升模型效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.02911v1](http://arxiv.org/pdf/2507.02911v1)**

> **作者:** Hyung Gun Chi; Zakaria Aldeneh; Tatiana Likhomanenko; Oggi Rudovic; Takuya Higuchi; Li-Wei Chen; Shinji Watanabe; Ahmed Hussen Abdelaziz
>
> **备注:** 5 pages, 1 figure, interspeech accepted paper
>
> **摘要:** We introduce DiceHuBERT, a knowledge distillation framework for compressing HuBERT, a widely used self-supervised learning (SSL)-based speech foundation model. Unlike existing distillation methods that rely on layer-wise and feature-wise mapping between teacher and student models, DiceHuBERT leverages HuBERT's iterative self-distillation mechanism by directly replacing the original model with a student model. This replacement allows the student to be trained using the same SSL objective used when pre-training HuBERT, eliminating the need for additional modules or architectural constraints. Experimental results on SUPERB show that DiceHuBERT consistently outperforms existing distillation methods, improving phoneme recognition performance by over 21% and ASR performance by more than 14%. Furthermore, DiceHuBERT demonstrates competitive performance across multiple tasks, highlighting its clear advantage.
>
---
#### [new 035] Assessing the Viability of Wave Field Synthesis in VR-Based Cognitive Research
- **分类: cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于虚拟现实中的音频研究任务，旨在评估波场合成（WFS）在增强VR听觉沉浸感方面的可行性，通过实验比较WFS与传统立体声的定位效果。**

- **链接: [http://arxiv.org/pdf/2507.03797v1](http://arxiv.org/pdf/2507.03797v1)**

> **作者:** Benjamin Kahl
>
> **备注:** 35 pages
>
> **摘要:** This paper investigates the viability of Wave Field Synthesis (WFS) for enhancing auditory immersion in VR-based cognitive research. While Virtual Reality (VR) offers significant advantages for studying human perception and behavior, auditory cues are often underutilized. WFS, an advanced audio rendering technique, can create highly realistic and spatially accurate soundscapes, potentially increasing ecological validity. This study evaluates WFS by implementing a sample experiment where participants localize static and moving sound sources in both a WFS-rendered environment and a conventional stereo headphone setup. The research explores the impact of virtual environments, sound types, and durations on localization accuracy and search behavior. Findings indicate that while stereo setups can achieve higher accuracy, WFS provides a more natural and intuitive auditory experience, particularly for directional cues. The study also highlights limitations of current WFS systems, such as the lack of height localization, occlusion simulation, and user-dependent optimization, which affect performance, especially for centrally located sound sources. Despite these challenges, WFS shows promise for specialized auditory perception research, particularly for complex soundscapes where directional information is paramount.
>
---
#### [new 036] Adaptive Slimming for Scalable and Efficient Speech Enhancement
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音增强任务，解决资源受限设备上模型性能与效率的平衡问题。通过动态裁剪技术，使模型自适应选择不同复杂度，提升效率并保持质量。**

- **链接: [http://arxiv.org/pdf/2507.04879v1](http://arxiv.org/pdf/2507.04879v1)**

> **作者:** Riccardo Miccini; Minje Kim; Clément Laroche; Luca Pezzarossa; Paris Smaragdis
>
> **备注:** Accepted for publication at the 2025 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA 2025)
>
> **摘要:** Speech enhancement (SE) enables robust speech recognition, real-time communication, hearing aids, and other applications where speech quality is crucial. However, deploying such systems on resource-constrained devices involves choosing a static trade-off between performance and computational efficiency. In this paper, we introduce dynamic slimming to DEMUCS, a popular SE architecture, making it scalable and input-adaptive. Slimming lets the model operate at different utilization factors (UF), each corresponding to a different performance/efficiency trade-off, effectively mimicking multiple model sizes without the extra storage costs. In addition, a router subnet, trained end-to-end with the backbone, determines the optimal UF for the current input. Thus, the system saves resources by adaptively selecting smaller UFs when additional complexity is unnecessary. We show that our solution is Pareto-optimal against individual UFs, confirming the benefits of dynamic routing. When training the proposed dynamically-slimmable model to use 10% of its capacity on average, we obtain the same or better speech quality as the equivalent static 25% utilization while reducing MACs by 29%.
>
---
#### [new 037] Prosody Labeling with Phoneme-BERT and Speech Foundation Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音标注任务，旨在提升音调和语义边界预测的准确性。通过结合语音和语言模型特征，提高了日语韵律标签的预测效果。**

- **链接: [http://arxiv.org/pdf/2507.03912v1](http://arxiv.org/pdf/2507.03912v1)**

> **作者:** Tomoki Koriyama
>
> **备注:** Accepted to Speech Synthesis Workshop 2025 (SSW13)
>
> **摘要:** This paper proposes a model for automatic prosodic label annotation, where the predicted labels can be used for training a prosody-controllable text-to-speech model. The proposed model utilizes not only rich acoustic features extracted by a self-supervised-learning (SSL)-based model or a Whisper encoder, but also linguistic features obtained from phoneme-input pretrained linguistic foundation models such as PnG BERT and PL-BERT. The concatenation of acoustic and linguistic features is used to predict phoneme-level prosodic labels. In the experimental evaluation on Japanese prosodic labels, including pitch accents and phrase break indices, it was observed that the combination of both speech and linguistic foundation models enhanced the prediction accuracy compared to using either a speech or linguistic input alone. Specifically, we achieved 89.8% prediction accuracy in accent labels, 93.2% in high-low pitch accents, and 94.3% in break indices.
>
---
#### [new 038] Ambisonics Encoder for Wearable Array with Improved Binaural Reproduction
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于音频编码任务，旨在解决可穿戴麦克风阵列在双耳再现中的精度问题。通过改进损失函数，结合ASM与BSM提升双耳音质。**

- **链接: [http://arxiv.org/pdf/2507.04108v1](http://arxiv.org/pdf/2507.04108v1)**

> **作者:** Yhonatan Gayer; Vladimir Tourbabin; Zamir Ben-Hur; David Alon; Boaz Rafaely
>
> **备注:** Published in Forum Acousticum 2025, 6 pages, 2 figures
>
> **摘要:** Ambisonics Signal Matching (ASM) is a recently proposed signal-independent approach to encoding Ambisonic signal from wearable microphone arrays, enabling efficient and standardized spatial sound reproduction. However, reproduction accuracy is currently limited due to the non-ideal layout of the microphones. This research introduces an enhanced ASM encoder that reformulates the loss function by integrating a Binaural Signal Matching (BSM) term into the optimization framework. The aim of this reformulation is to improve the accuracy of binaural reproduction when integrating the Ambisonic signal with Head-Related Transfer Functions (HRTFs), making the encoded Ambisonic signal better suited for binaural reproduction. This paper first presents the mathematical formulation developed to align the ASM and BSM objectives in a single loss function, followed by a simulation study with a simulated microphone array mounted on a rigid sphere representing a head-mounted wearable array. The analysis shows that improved binaural reproduction with the encoded Ambisonic signal can be achieved using this joint ASM-BSM optimization, thereby enabling higher-quality binaural playback for virtual and augmented reality applications based on Ambisonics.
>
---
#### [new 039] WSCoach: Wearable Real-time Auditory Feedback for Reducing Unwanted Words in Daily Communication
- **分类: cs.HC; cs.SD**

- **简介: 该论文属于行为干预任务，旨在减少日常交流中的不当用语。通过可穿戴设备提供实时听觉反馈，设计并评估了WSCoach系统，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.04238v1](http://arxiv.org/pdf/2507.04238v1)**

> **作者:** Zhang Youpeng; Nuwan Janaka; Ashwin Ram; Yin Peilin; Tian Yang; Shengdong Zhao; Pierre Dragicevic
>
> **备注:** 30 pages, 9 figures
>
> **摘要:** The rise of wearable smart devices raises unprecedented opportunities for self-improvement through ubiquitous behavior tracking and guidance. However, the design of effective wearable behavior intervention systems remains relatively unexplored. To address this gap, we conducted controlled studies focusing on the reduction of unwanted words (e.g., filler words, swear words) in daily communication through auditory feedback using wearable technology. We started with a design space exploration, considering various factors such as the type, duration, and timing of the auditory feedback. Then, we conducted pilot studies to reduce the space of design choices and prototyped a system called WSCoach (Wearable Speech Coach), which informs users when they utter unwanted words in near-real-time. To evaluate WSCoach, we compared it with a state-of-the-art mobile application supporting post-hoc conversation analysis. Both approaches were effective in reducing the occurrence of unwanted words, but WSCoach appears to be more effective in the long run. Finally, we discuss guidelines for the design of wearable audio-based behavior monitoring and intervention systems and highlight the potential of wearable technology for facilitating behavior correction and improvement. For supplementary material, please see the META Appendix and our OSF project at https://osf.io/6vhwn/?view_only=489498d3ac2d4703a17475fc6ca65dfa.
>
---
#### [new 040] K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于儿童语言评估任务，解决语音识别困难与反馈缺失问题。提出K-Function框架，结合语音转录与客观评分，提升诊断准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.03043v1](http://arxiv.org/pdf/2507.03043v1)**

> **作者:** Shuhe Li; Chenxu Guo; Jiachen Lian; Cheol Jun Cho; Wenshuo Zhao; Xuanru Zhou; Dingkun Zhou; Sam Wang; Grace Wang; Jingze Yang; Jingyi Xu; Ruohan Bao; Elise Brenner; Brandon In; Francesca Pei; Maria Luisa Gorno-Tempini; Gopala Anumanchipalli
>
> **摘要:** Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment.
>
---
#### [new 041] DeepGesture: A conversational gesture synthesis system based on emotions and semantics
- **分类: cs.HC; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于手势生成任务，解决数字人自然动作生成问题。提出DeepGesture框架，结合语义与情感信息生成更逼真手势。**

- **链接: [http://arxiv.org/pdf/2507.03147v1](http://arxiv.org/pdf/2507.03147v1)**

> **作者:** Thanh Hoang-Minh
>
> **备注:** Video Demo: https://www.youtube.com/watch?v=eZghfNGmZn8
>
> **摘要:** Along with the explosion of large language models, improvements in speech synthesis, advancements in hardware, and the evolution of computer graphics, the current bottleneck in creating digital humans lies in generating character movements that correspond naturally to text or speech inputs. In this work, we present DeepGesture, a diffusion-based gesture synthesis framework for generating expressive co-speech gestures conditioned on multimodal signals-text, speech, emotion, and seed motion. Built upon the DiffuseStyleGesture model, DeepGesture introduces novel architectural enhancements that improve semantic alignment and emotional expressiveness in generated gestures. Specifically, we integrate fast text transcriptions as semantic conditioning and implement emotion-guided classifier-free diffusion to support controllable gesture generation across affective states. A lightweight Transformer backbone combines full self-attention and cross-local attention for effective feature fusion of heterogeneous modalities. To visualize results, we implement a full rendering pipeline in Unity based on BVH output from the model. Evaluation on the ZeroEGGS dataset shows that DeepGesture produces gestures with improved human-likeness and contextual appropriateness, outperforming baselines on Mean Opinion Score and Frechet Gesture Distance metrics. Our system supports interpolation between emotional states and demonstrates generalization to out-of-distribution speech, including synthetic voices-marking a step forward toward fully multimodal, emotionally aware digital humans.
>
---
#### [new 042] Improving Low-Resource Dialect Classification Using Retrieval-based Voice Conversion
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于低资源德语方言分类任务，旨在解决数据稀缺问题。通过使用检索语音转换技术增强数据，提升分类性能。**

- **链接: [http://arxiv.org/pdf/2507.03641v1](http://arxiv.org/pdf/2507.03641v1)**

> **作者:** Lea Fischbach; Akbar Karimi; Caroline Kleen; Alfred Lameli; Lucie Flek
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Deep learning models for dialect identification are often limited by the scarcity of dialectal data. To address this challenge, we propose to use Retrieval-based Voice Conversion (RVC) as an effective data augmentation method for a low-resource German dialect classification task. By converting audio samples to a uniform target speaker, RVC minimizes speaker-related variability, enabling models to focus on dialect-specific linguistic and phonetic features. Our experiments demonstrate that RVC enhances classification performance when utilized as a standalone augmentation method. Furthermore, combining RVC with other augmentation methods such as frequency masking and segment removal leads to additional performance gains, highlighting its potential for improving dialect classification in low-resource scenarios.
>
---
#### [new 043] Spatial and Semantic Embedding Integration for Stereo Sound Event Localization and Detection in Regular Videos
- **分类: eess.AS; cs.LG; eess.IV; eess.SP**

- **简介: 该论文属于声源定位与检测任务，解决常规视频中立体声事件的定位与识别问题。通过融合语义嵌入和多模态模型提升性能。**

- **链接: [http://arxiv.org/pdf/2507.04845v1](http://arxiv.org/pdf/2507.04845v1)**

> **作者:** Davide Berghi; Philip J. B. Jackson
>
> **摘要:** This report presents our systems submitted to the audio-only and audio-visual tracks of the DCASE2025 Task 3 Challenge: Stereo Sound Event Localization and Detection (SELD) in Regular Video Content. SELD is a complex task that combines temporal event classification with spatial localization, requiring reasoning across spatial, temporal, and semantic dimensions. The last is arguably the most challenging to model. Traditional SELD architectures rely on multichannel input, which limits their ability to leverage large-scale pre-training due to data constraints. To address this, we enhance standard SELD architectures with semantic information by integrating pre-trained, contrastive language-aligned models: CLAP for audio and OWL-ViT for visual inputs. These embeddings are incorporated into a modified Conformer module tailored for multimodal fusion, which we refer to as the Cross-Modal Conformer. Additionally, we incorporate autocorrelation-based acoustic features to improve distance estimation. We pre-train our models on curated synthetic audio and audio-visual datasets and apply a left-right channel swapping augmentation to further increase the training data. Both our audio-only and audio-visual systems substantially outperform the challenge baselines on the development set, demonstrating the effectiveness of our strategy. Performance is further improved through model ensembling and a visual post-processing step based on human keypoints. Future work will investigate the contribution of each modality and explore architectural variants to further enhance results.
>
---
## 更新

#### [replaced 001] A Framework for Synthetic Audio Conversations Generation using Large Language Models
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.00946v3](http://arxiv.org/pdf/2409.00946v3)**

> **作者:** Kaung Myat Kyaw; Jonathan Hoyin Chan
>
> **备注:** This work has been accepted at the WI-IAT'24. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media
>
> **摘要:** In this paper, we introduce ConversaSynth, a framework designed to generate synthetic conversation audio using large language models (LLMs) with multiple persona settings. The framework first creates diverse and coherent text-based dialogues across various topics, which are then converted into audio using text-to-speech (TTS) systems. Our experiments demonstrate that ConversaSynth effectively generates highquality synthetic audio datasets, which can significantly enhance the training and evaluation of models for audio tagging, audio classification, and multi-speaker speech recognition. The results indicate that the synthetic datasets generated by ConversaSynth exhibit substantial diversity and realism, making them suitable for developing robust, adaptable audio-based AI systems.
>
---
#### [replaced 002] AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals
- **分类: cs.SD; cs.HC; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.16780v2](http://arxiv.org/pdf/2501.16780v2)**

> **作者:** Dongliang Zhou; Yakun Zhang; Jinghan Wu; Xingyu Zhang; Liang Xie; Erwei Yin
>
> **备注:** The paper has been accepted by IEEE Transactions on Human-Machine Systems
>
> **摘要:** The global aging population faces considerable challenges, particularly in communication, due to the prevalence of hearing and speech impairments. To address these, we introduce the AVE speech, a comprehensive multi-modal dataset for speech recognition tasks. The dataset includes a 100-sentence Mandarin corpus with audio signals, lip-region video recordings, and six-channel electromyography (EMG) data, collected from 100 participants. Each subject read the entire corpus ten times, with each sentence averaging approximately two seconds in duration, resulting in over 55 hours of multi-modal speech data per modality. Experiments demonstrate that combining these modalities significantly improves recognition performance, particularly in cross-subject and high-noise environments. To our knowledge, this is the first publicly available sentence-level dataset integrating these three modalities for large-scale Mandarin speech recognition. We expect this dataset to drive advancements in both acoustic and non-acoustic speech recognition research, enhancing cross-modal learning and human-machine interaction.
>
---
#### [replaced 003] An introduction to pitch strength in contemporary popular music analysis and production
- **分类: cs.SD; eess.AS; 00A65; J.5**

- **链接: [http://arxiv.org/pdf/2506.07473v4](http://arxiv.org/pdf/2506.07473v4)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2024, Innovation in Music Conference, 14-16 June, 2024, Kristiania University College, Oslo, Norway
>
> **摘要:** Music information retrieval distinguishes between low- and high-level descriptions of music. Current generative AI models rely on text descriptions that are higher level than the controls familiar to studio musicians. Pitch strength, a low-level perceptual parameter of contemporary popular music, may be one feature that could make such AI models more suited to music production. Signal and perceptual analyses suggest that pitch strength (1) varies significantly across and inside songs; (2) contributes to both small- and large-scale structure; (3) contributes to the handling of polyphonic dissonance; and (4) may be a feature of upper harmonics made audible in a perspective of perceptual richness.
>
---
#### [replaced 004] ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.19937v2](http://arxiv.org/pdf/2505.19937v2)**

> **作者:** Pooneh Mousavi; Yingzhi Wang; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Large Language Models (LLMs) are increasingly used in Spoken Language Understanding (SLU), where effective multimodal learning depends on the alignment between audio and text. Despite various fusion methods, no standard metric exists to assess this alignment. This work introduces ALAS (Automatic Latent Alignment Score), a metric that evaluates alignment by measuring correlations between audio and text representations across transformer layers. Experiments on Spoken Question Answering and Emotion Recognition show that ALAS captures meaningful patterns across tasks and layers.
>
---
#### [replaced 005] Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.13596v2](http://arxiv.org/pdf/2506.13596v2)**

> **作者:** Tuan Nguyen; Long-Vu Hoang; Huy-Dat Tran
>
> **备注:** Accepted to Interspeech MLCSLM-2025 Workshop
>
> **摘要:** This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model.
>
---
#### [replaced 006] AADNet: An End-to-End Deep Learning Model for Auditory Attention Decoding
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2410.13059v2](http://arxiv.org/pdf/2410.13059v2)**

> **作者:** Nhan Duc Thanh Nguyen; Huy Phan; Simon Geirnaert; Kaare Mikkelsen; Preben Kidmose
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Auditory attention decoding (AAD) is the process of identifying the attended speech in a multi-talker environment using brain signals, typically recorded through electroencephalography (EEG). Over the past decade, AAD has undergone continuous development, driven by its promising application in neuro-steered hearing devices. Most AAD algorithms are relying on the increase in neural entrainment to the envelope of attended speech, as compared to unattended speech, typically using a two-step approach. First, the algorithm predicts representations of the attended speech signal envelopes; second, it identifies the attended speech by finding the highest correlation between the predictions and the representations of the actual speech signals. In this study, we proposed a novel end-to-end neural network architecture, named AADNet, which combines these two stages into a direct approach to address the AAD problem. We compare the proposed network against the traditional approaches, including linear stimulus reconstruction, canonical correlation analysis, and an alternative non-linear stimulus reconstruction using two different datasets. AADNet shows a significant performance improvement for both subject-specific and subject-independent models. Notably, the average subject-independent classification accuracies from 56.1 % to 82.7 % with analysis window lengths ranging from 1 to 40 seconds, respectively, show a significantly improved ability to generalize to data from unseen subjects. These results highlight the potential of deep learning models for advancing AAD, with promising implications for future hearing aids, assistive devices, and clinical assessments.
>
---
#### [replaced 007] Manipulated Regions Localization For Partially Deepfake Audio: A Survey
- **分类: cs.SD; cs.MM**

- **链接: [http://arxiv.org/pdf/2506.14396v2](http://arxiv.org/pdf/2506.14396v2)**

> **作者:** Jiayi He; Jiangyan Yi; Jianhua Tao; Siding Zeng; Hao Gu
>
> **备注:** Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** With the development of audio deepfake techniques, attacks with partially deepfake audio are beginning to rise. Compared to fully deepfake, it is much harder to be identified by the detector due to the partially cryptic manipulation, resulting in higher security risks. Although some studies have been launched, there is no comprehensive review to systematically introduce the current situations and development trends for addressing this issue. Thus, in this survey, we are the first to outline a systematic introduction for partially deepfake audio manipulated region localization tasks, including the fundamentals, branches of existing methods, current limitations and potential trends, providing a revealing insight into this scope.
>
---
#### [replaced 008] Music102: An $D_{12}$-equivariant transformer for chord progression accompaniment
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.18151v2](http://arxiv.org/pdf/2410.18151v2)**

> **作者:** Weiliang Luo
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** We present Music102, an advanced model aimed at enhancing chord progression accompaniment through a $D_{12}$-equivariant transformer. Inspired by group theory and symbolic music structures, Music102 leverages musical symmetry--such as transposition and reflection operations--integrating these properties into the transformer architecture. By encoding prior music knowledge, the model maintains equivariance across both melody and chord sequences. The POP909 dataset was employed to train and evaluate Music102, revealing significant improvements over the non-equivariant Music101 prototype Music101 in both weighted loss and exact accuracy metrics, despite using fewer parameters. This work showcases the adaptability of self-attention mechanisms and layer normalization to the discrete musical domain, addressing challenges in computational music analysis. With its stable and flexible neural framework, Music102 sets the stage for further exploration in equivariant music generation and computational composition tools, bridging mathematical theory with practical music performance.
>
---
#### [replaced 009] Event-based Photometric Bundle Adjustment
- **分类: cs.CV; cs.RO; eess.SP; math.OC**

- **链接: [http://arxiv.org/pdf/2412.14111v2](http://arxiv.org/pdf/2412.14111v2)**

> **作者:** Shuang Guo; Guillermo Gallego
>
> **备注:** 21 pages, 19 figures, 10 tables. Project page: https://github.com/tub-rip/epba
>
> **摘要:** We tackle the problem of bundle adjustment (i.e., simultaneous refinement of camera poses and scene map) for a purely rotating event camera. Starting from first principles, we formulate the problem as a classical non-linear least squares optimization. The photometric error is defined using the event generation model directly in the camera rotations and the semi-dense scene brightness that triggers the events. We leverage the sparsity of event data to design a tractable Levenberg-Marquardt solver that handles the very large number of variables involved. To the best of our knowledge, our method, which we call Event-based Photometric Bundle Adjustment (EPBA), is the first event-only photometric bundle adjustment method that works on the brightness map directly and exploits the space-time characteristics of event data, without having to convert events into image-like representations. Comprehensive experiments on both synthetic and real-world datasets demonstrate EPBA's effectiveness in decreasing the photometric error (by up to 90%), yielding results of unparalleled quality. The refined maps reveal details that were hidden using prior state-of-the-art rotation-only estimation methods. The experiments on modern high-resolution event cameras show the applicability of EPBA to panoramic imaging in various scenarios (without map initialization, at multiple resolutions, and in combination with other methods, such as IMU dead reckoning or previous event-based rotation estimation methods). We make the source code publicly available. https://github.com/tub-rip/epba
>
---
#### [replaced 010] Evolving music theory for emerging musical languages
- **分类: cs.SD; 00A65; J.5**

- **链接: [http://arxiv.org/pdf/2506.14504v2](http://arxiv.org/pdf/2506.14504v2)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2025, Innovation in Music Conference. 20-22 June, 2025, Bath Spa University, Bath, UK
>
> **摘要:** This chapter reconsiders the concept of pitch in contemporary popular music (CPM), particularly in electronic contexts where traditional assumptions may fail. Drawing on phenomenological and inductive methods, it argues that pitch is not an ontologically objective property but a perceptual construct shaped by listeners and conditions. Analyses of quasi-harmonic tones reveal that a single tone can convey multiple pitches, giving rise to tonal fission. The perception of pitch may also be multistable, varying for the same listener over time. In this framework, the tuning system may emerge from a tone's internal structure. A parallel with the coastline paradox supports a model of pitch grounded in perceptual variability, challenging inherited theoretical norms.
>
---
#### [replaced 011] AVTENet: A Human-Cognition-Inspired Audio-Visual Transformer-Based Ensemble Network for Video Deepfake Detection
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2310.13103v2](http://arxiv.org/pdf/2310.13103v2)**

> **作者:** Ammarah Hashmi; Sahibzada Adil Shahzad; Chia-Wen Lin; Yu Tsao; Hsin-Min Wang
>
> **摘要:** The recent proliferation of hyper-realistic deepfake videos has drawn attention to the threat of audio and visual forgeries. Most previous studies on detecting artificial intelligence-generated fake videos only utilize visual modality or audio modality. While some methods exploit audio and visual modalities to detect forged videos, they have not been comprehensively evaluated on multimodal datasets of deepfake videos involving acoustic and visual manipulations, and are mostly based on convolutional neural networks with low detection accuracy. Considering that human cognition instinctively integrates multisensory information including audio and visual cues to perceive and interpret content and the success of transformer in various fields, this study introduces the audio-visual transformer-based ensemble network (AVTENet). This innovative framework tackles the complexities of deepfake technology by integrating both acoustic and visual manipulations to enhance the accuracy of video forgery detection. Specifically, the proposed model integrates several purely transformer-based variants that capture video, audio, and audio-visual salient cues to reach a consensus in prediction. For evaluation, we use the recently released benchmark multimodal audio-video FakeAVCeleb dataset. For a detailed analysis, we evaluate AVTENet, its variants, and several existing methods on multiple test sets of the FakeAVCeleb dataset. Experimental results show that the proposed model outperforms all existing methods and achieves state-of-the-art performance on Testset-I and Testset-II of the FakeAVCeleb dataset. We also compare AVTENet against humans in detecting video forgery. The results show that AVTENet significantly outperforms humans.
>
---
#### [replaced 012] SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.06612v2](http://arxiv.org/pdf/2406.06612v2)**

> **作者:** Rishit Dagli; Shivesh Prakash; Robert Wu; Houman Khosravani
>
> **备注:** Project Page: https://see2sound.github.io/
>
> **摘要:** Generating combined visual and auditory sensory experiences is critical for the consumption of immersive content. Recent advances in neural generative models have enabled the creation of high-resolution content across multiple modalities such as images, text, speech, and videos. Despite these successes, there remains a significant gap in the generation of high-quality spatial audio that complements generated visual content. Furthermore, current audio generation models excel in either generating natural audio or speech or music but fall short in integrating spatial audio cues necessary for immersive experiences. In this work, we introduce SEE-2-SOUND, a zero-shot approach that decomposes the task into (1) identifying visual regions of interest; (2) locating these elements in 3D space; (3) generating mono-audio for each; and (4) integrating them into spatial audio. Using our framework, we demonstrate compelling results for generating spatial audio for high-quality videos, images, and dynamic images from the internet, as well as media generated by learned approaches.
>
---
#### [replaced 013] UniForm: A Unified Multi-Task Diffusion Transformer for Audio-Video Generation
- **分类: cs.MM; cs.AI; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.03897v5](http://arxiv.org/pdf/2502.03897v5)**

> **作者:** Lei Zhao; Linfeng Feng; Dongxu Ge; Rujin Chen; Fangqiu Yi; Chi Zhang; Xiao-Lei Zhang; Xuelong Li
>
> **备注:** Our demos are available at https://uniform-t2av.github.io/
>
> **摘要:** With the rise of diffusion models, audio-video generation has been revolutionized. However, most existing methods rely on separate modules for each modality, with limited exploration of unified generative architectures. In addition, many are confined to a single task and small-scale datasets. To overcome these limitations, we introduce UniForm, a unified multi-task diffusion transformer that generates both audio and visual modalities in a shared latent space. By using a unified denoising network, UniForm captures the inherent correlations between sound and vision. Additionally, we propose task-specific noise schemes and task tokens, enabling the model to support multiple tasks with a single set of parameters, including video-to-audio, audio-to-video and text-to-audio-video generation. Furthermore, by leveraging large language models and a large-scale text-audio-video combined dataset, UniForm achieves greater generative diversity than prior approaches. Experiments show that UniForm achieves performance close to the state-of-the-art single-task models across three generation tasks, with generated content that is not only highly aligned with real-world data distributions but also enables more diverse and fine-grained generation.
>
---
#### [replaced 014] SwitchCodec: A High-Fidelity Nerual Audio Codec With Sparse Quantization
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24437v2](http://arxiv.org/pdf/2505.24437v2)**

> **作者:** Jin Wang; Wenbin Jiang; Xiangbo Wang
>
> **备注:** 11 pages,7 figures
>
> **摘要:** Neural audio compression has emerged as a promising technology for efficiently representing speech, music, and general audio. However, existing methods suffer from significant performance degradation at limited bitrates, where the available embedding space is sharply constrained. To address this, we propose a universal high-fidelity neural audio compression algorithm featuring Residual Experts Vector Quantization (REVQ), which substantially expands the embedding space with minimal impact on bandwidth. A gentle load-balancing strategy is introduced to ensure the full utilization of this expanded space. Furthermore, we develop a novel multi-tiered discriminator that periodically stratifies STFT spectra, guiding the generator to focus on critical spectral regions. To support multiple bitrates without quality loss at the lower end, we adopt an efficient post-training strategy. Our proposed model achieves impressive performance, with PESQ and ViSQOL scores of 2.87 and 4.27, respectively, at 2.67 kbps bandwidth. The approach effectively reduces spectral blur, decreasing the distance to the original mel-spectrogram by 13%. Notably, our post-training strategy achieves performance comparable to dedicated fixed-bitrate models while reducing the required training time by half. Extensive ablation studies confirm the superiority of our method over baselines.
>
---
#### [replaced 015] Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds
- **分类: cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.07435v4](http://arxiv.org/pdf/2503.07435v4)**

> **作者:** Riccardo Mazzieri; Jacopo Pegoraro; Michele Rossi
>
> **摘要:** The adoption of Millimeter-Wave (mmWave) radar devices for human sensing, particularly gait recognition, has recently gathered significant attention due to their efficiency, resilience to environmental conditions, and privacy-preserving nature. In this work, we tackle the challenging problem of Open-set Gait Recognition (OSGR) from sparse mmWave radar point clouds. Unlike most existing research, which assumes a closed-set scenario, our work considers the more realistic open-set case, where unknown subjects might be present at inference time, and should be correctly recognized by the system. Point clouds are well-suited for edge computing applications with resource constraints, but are more significantly affected by noise and random fluctuations than other representations, like the more common micro-Doppler signature. This is the first work addressing open-set gait recognition with sparse point cloud data. To do so, we propose a novel neural network architecture that combines supervised classification with unsupervised reconstruction of the point clouds, creating a robust, rich, and highly regularized latent space of gait features. To detect unknown subjects at inference time, we introduce a probabilistic novelty detection algorithm that leverages the structured latent space and offers a tunable trade-off between inference speed and prediction accuracy. Along with this paper, we release mmGait10, an original human gait dataset featuring over five hours of measurements from ten subjects, under varied walking modalities. Extensive experimental results show that our solution attains F1-Score improvements by 24% over state-of-the-art methods adapted for point clouds, on average, and across multiple openness levels.
>
---
#### [replaced 016] UltrasonicSpheres: Localized, Multi-Channel Sound Spheres Using Off-the-Shelf Speakers and Earables
- **分类: cs.SD; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.02715v2](http://arxiv.org/pdf/2506.02715v2)**

> **作者:** Michael Küttner; Valeria Zitz; Kathrin Gerling; Michael Beigl; Tobias Röddiger
>
> **摘要:** We present a demo of UltrasonicSpheres, a novel system for location-specific audio delivery using wearable earphones that decode ultrasonic signals into audible sound. Unlike conventional beamforming setups, UltrasonicSpheres relies on single ultrasonic speakers to broadcast localized audio with multiple channels, each encoded on a distinct ultrasonic carrier frequency. Users wearing our acoustically transparent earphones can demodulate their selected stream, such as exhibit narrations in a chosen language, while remaining fully aware of ambient environmental sounds. The experience preserves spatial audio perception, giving the impression that the sound originates directly from the physical location of the source. This enables personalized, localized audio without requiring pairing, tracking, or additional infrastructure. Importantly, visitors not equipped with the earphones are unaffected, as the ultrasonic signals are inaudible to the human ear. Our demo invites participants to explore multiple co-located audio zones and experience how UltrasonicSpheres supports unobtrusive delivery of personalized sound in public spaces.
>
---
#### [replaced 017] Self-Steering Deep Non-Linear Spatially Selective Filters for Efficient Extraction of Moving Speakers under Weak Guidance
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.02791v2](http://arxiv.org/pdf/2507.02791v2)**

> **作者:** Jakob Kienegger; Alina Mannanova; Huajian Fang; Timo Gerkmann
>
> **备注:** Accepted at IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025. Video demonstration: https://youtu.be/aSKOSh5JZ3o
>
> **摘要:** Recent works on deep non-linear spatially selective filters demonstrate exceptional enhancement performance with computationally lightweight architectures for stationary speakers of known directions. However, to maintain this performance in dynamic scenarios, resource-intensive data-driven tracking algorithms become necessary to provide precise spatial guidance conditioned on the initial direction of a target speaker. As this additional computational overhead hinders application in resource-constrained scenarios such as real-time speech enhancement, we present a novel strategy utilizing a low-complexity tracking algorithm in the form of a particle filter instead. Assuming a causal, sequential processing style, we introduce temporal feedback to leverage the enhanced speech signal of the spatially selective filter to compensate for the limited modeling capabilities of the particle filter. Evaluation on a synthetic dataset illustrates how the autoregressive interplay between both algorithms drastically improves tracking accuracy and leads to strong enhancement performance. A listening test with real-world recordings complements these findings by indicating a clear trend towards our proposed self-steering pipeline as preferred choice over comparable methods.
>
---
#### [replaced 018] Serenade: A Singing Style Conversion Framework Based On Audio Infilling
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.12388v2](http://arxiv.org/pdf/2503.12388v2)**

> **作者:** Lester Phillip Violeta; Wen-Chin Huang; Tomoki Toda
>
> **备注:** Accepted to EUSIPCO 2025
>
> **摘要:** We propose Serenade, a novel framework for the singing style conversion (SSC) task. Although singer identity conversion has made great strides in the previous years, converting the singing style of a singer has been an unexplored research area. We find three main challenges in SSC: modeling the target style, disentangling source style, and retaining the source melody. To model the target singing style, we use an audio infilling task by predicting a masked segment of the target mel-spectrogram with a flow-matching model using the complement of the masked target mel-spectrogram along with disentangled acoustic features. On the other hand, to disentangle the source singing style, we use a cyclic training approach, where we use synthetic converted samples as source inputs and reconstruct the original source mel-spectrogram as a target. Finally, to retain the source melody better, we investigate a post-processing module using a source-filter-based vocoder and resynthesize the converted waveforms using the original F0 patterns. Our results showed that the Serenade framework can handle generalized SSC tasks with the best overall similarity score, especially in modeling breathy and mixed singing styles. We also found that resynthesizing with the original F0 patterns alleviated out-of-tune singing and improved naturalness, but found a slight tradeoff in similarity due to not changing the F0 patterns into the target style.
>
---
