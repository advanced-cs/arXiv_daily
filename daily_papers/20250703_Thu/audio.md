# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] Exploring Classical Piano Performance Generation with Expressive Music Variational AutoEncoder
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在模拟作曲家与演奏家的双重角色，生成高质量古典钢琴表演。通过提出XMVAE模型实现这一目标。**

- **链接: [http://arxiv.org/pdf/2507.01582v1](http://arxiv.org/pdf/2507.01582v1)**

> **作者:** Jing Luo; Xinyu Yang; Jie Wei
>
> **备注:** Accepted by IEEE SMC 2025
>
> **摘要:** The creativity of classical music arises not only from composers who craft the musical sheets but also from performers who interpret the static notations with expressive nuances. This paper addresses the challenge of generating classical piano performances from scratch, aiming to emulate the dual roles of composer and pianist in the creative process. We introduce the Expressive Compound Word (ECP) representation, which effectively captures both the metrical structure and expressive nuances of classical performances. Building on this, we propose the Expressive Music Variational AutoEncoder (XMVAE), a model featuring two branches: a Vector Quantized Variational AutoEncoder (VQ-VAE) branch that generates score-related content, representing the Composer, and a vanilla VAE branch that produces expressive details, fulfilling the role of Pianist. These branches are jointly trained with similar Seq2Seq architectures, leveraging a multiscale encoder to capture beat-level contextual information and an orthogonal Transformer decoder for efficient compound tokens decoding. Both objective and subjective evaluations demonstrate that XMVAE generates classical performances with superior musical quality compared to state-of-the-art models. Furthermore, pretraining the Composer branch on extra musical score datasets contribute to a significant performance gain.
>
---
#### [new 002] A Dataset for Automatic Assessment of TTS Quality in Spanish
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于TTS质量评估任务，旨在提升西班牙语自然度预测模型的准确性。研究构建了首个西班牙语TTS质量数据集，并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2507.01805v1](http://arxiv.org/pdf/2507.01805v1)**

> **作者:** Alejandro Sosa Welford; Leonardo Pepino
>
> **备注:** 5 pages, 2 figures. Accepted at Interspeech 2025
>
> **摘要:** This work addresses the development of a database for the automatic assessment of text-to-speech (TTS) systems in Spanish, aiming to improve the accuracy of naturalness prediction models. The dataset consists of 4,326 audio samples from 52 different TTS systems and human voices and is, up to our knowledge, the first of its kind in Spanish. To label the audios, a subjective test was designed based on the ITU-T Rec. P.807 standard and completed by 92 participants. Furthermore, the utility of the collected dataset was validated by training automatic naturalness prediction systems. We explored two approaches: fine-tuning an existing model originally trained for English, and training small downstream networks on top of frozen self-supervised speech models. Our models achieve a mean absolute error of 0.8 on a five-point MOS scale. Further analysis demonstrates the quality and diversity of the developed dataset, and its potential to advance TTS research in Spanish.
>
---
#### [new 003] Real-Time Emergency Vehicle Siren Detection with Efficient CNNs on Embedded Hardware
- **分类: cs.SD; cs.AI; eess.AS; 68T07 (Primary), 68T10 (Secondary); B.1.5; B.4.5; C.3; C.4; I.2; K.4; J.2**

- **简介: 该论文属于声音事件检测任务，解决紧急车辆警报实时识别问题。通过优化CNN模型和构建高质量数据集，实现在嵌入式设备上的低延迟检测。**

- **链接: [http://arxiv.org/pdf/2507.01563v1](http://arxiv.org/pdf/2507.01563v1)**

> **作者:** Marco Giordano; Stefano Giacomelli; Claudia Rinaldi; Fabio Graziosi
>
> **备注:** 10 pages, 10 figures, submitted to https://internetofsounds2025.ieee-is2.org/. arXiv admin note: text overlap with arXiv:2506.23437
>
> **摘要:** We present a full-stack emergency vehicle (EV) siren detection system designed for real-time deployment on embedded hardware. The proposed approach is based on E2PANNs, a fine-tuned convolutional neural network derived from EPANNs, and optimized for binary sound event detection under urban acoustic conditions. A key contribution is the creation of curated and semantically structured datasets - AudioSet-EV, AudioSet-EV Augmented, and Unified-EV - developed using a custom AudioSet-Tools framework to overcome the low reliability of standard AudioSet annotations. The system is deployed on a Raspberry Pi 5 equipped with a high-fidelity DAC+microphone board, implementing a multithreaded inference engine with adaptive frame sizing, probability smoothing, and a decision-state machine to control false positive activations. A remote WebSocket interface provides real-time monitoring and facilitates live demonstration capabilities. Performance is evaluated using both framewise and event-based metrics across multiple configurations. Results show the system achieves low-latency detection with improved robustness under realistic audio conditions. This work demonstrates the feasibility of deploying IoS-compatible SED solutions that can form distributed acoustic monitoring networks, enabling collaborative emergency vehicle tracking across smart city infrastructures through WebSocket connectivity on low-cost edge devices.
>
---
#### [new 004] User-guided Generative Source Separation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐源分离任务，旨在解决传统方法灵活性不足的问题。提出GuideSep模型，通过用户引导实现更灵活的乐器分离。**

- **链接: [http://arxiv.org/pdf/2507.01339v1](http://arxiv.org/pdf/2507.01339v1)**

> **作者:** Yutong Wen; Minje Kim; Paris Smaragdis
>
> **摘要:** Music source separation (MSS) aims to extract individual instrument sources from their mixture. While most existing methods focus on the widely adopted four-stem separation setup (vocals, bass, drums, and other instruments), this approach lacks the flexibility needed for real-world applications. To address this, we propose GuideSep, a diffusion-based MSS model capable of instrument-agnostic separation beyond the four-stem setup. GuideSep is conditioned on multiple inputs: a waveform mimicry condition, which can be easily provided by humming or playing the target melody, and mel-spectrogram domain masks, which offer additional guidance for separation. Unlike prior approaches that relied on fixed class labels or sound queries, our conditioning scheme, coupled with the generative approach, provides greater flexibility and applicability. Additionally, we design a mask-prediction baseline using the same model architecture to systematically compare predictive and generative approaches. Our objective and subjective evaluations demonstrate that GuideSep achieves high-quality separation while enabling more versatile instrument extraction, highlighting the potential of user participation in the diffusion-based generative process for MSS. Our code and demo page are available at https://yutongwen.github.io/GuideSep/
>
---
#### [new 005] Generalizable Detection of Audio Deepfakes
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在提升模型的泛化能力。通过实验对比不同预训练模型和数据增强策略，优化检测效果。**

- **链接: [http://arxiv.org/pdf/2507.01750v1](http://arxiv.org/pdf/2507.01750v1)**

> **作者:** Jose A. Lopez; Georg Stemmer; Héctor Cordourier Maruri
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** In this paper, we present our comprehensive study aimed at enhancing the generalization capabilities of audio deepfake detection models. We investigate the performance of various pre-trained backbones, including Wav2Vec2, WavLM, and Whisper, across a diverse set of datasets, including those from the ASVspoof challenges and additional sources. Our experiments focus on the effects of different data augmentation strategies and loss functions on model performance. The results of our research demonstrate substantial enhancements in the generalization capabilities of audio deepfake detection models, surpassing the performance of the top-ranked single system in the ASVspoof 5 Challenge. This study contributes valuable insights into the optimization of audio models for more robust deepfake detection and facilitates future research in this critical area.
>
---
#### [new 006] Hello Afrika: Speech Commands in Kinyarwanda
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音命令识别任务，旨在解决非洲语言语音模型不足的问题。研究构建了基隆瓦语语音命令数据集，并在多种设备上部署模型进行测试。**

- **链接: [http://arxiv.org/pdf/2507.01024v1](http://arxiv.org/pdf/2507.01024v1)**

> **作者:** George Igwegbe; Martins Awojide; Mboh Bless; Nirel Kadzo
>
> **备注:** Data Science Africa, 2024
>
> **摘要:** Voice or Speech Commands are a subset of the broader Spoken Word Corpus of a language which are essential for non-contact control of and activation of larger AI systems in devices used in everyday life especially for persons with disabilities. Currently, there is a dearth of speech command models for African languages. The Hello Afrika project aims to address this issue and its first iteration is focused on the Kinyarwanda language since the country has shown interest in developing speech recognition technologies culminating in one of the largest datasets on Mozilla Common Voice. The model was built off a custom speech command corpus made up of general directives, numbers, and a wake word. The final model was deployed on multiple devices (PC, Mobile Phone and Edge Devices) and the performance was assessed using suitable metrics.
>
---
#### [new 007] Scalable Offline ASR for Command-Style Dictation in Courtrooms
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决法庭命令式录音的高效处理问题。提出一种开源框架，通过VAD和并行Whisper模型实现低延迟批处理，提升计算效率。**

- **链接: [http://arxiv.org/pdf/2507.01021v1](http://arxiv.org/pdf/2507.01021v1)**

> **作者:** Kumarmanas Nethil; Vaibhav Mishra; Kriti Anandan; Kavya Manohar
>
> **备注:** Accepted to Interspeech 2025 Show & Tell
>
> **摘要:** We propose an open-source framework for Command-style dictation that addresses the gap between resource-intensive Online systems and high-latency Batch processing. Our approach uses Voice Activity Detection (VAD) to segment audio and transcribes these segments in parallel using Whisper models, enabling efficient multiplexing across audios. Unlike proprietary systems like SuperWhisper, this framework is also compatible with most ASR architectures, including widely used CTC-based models. Our multiplexing technique maximizes compute utilization in real-world settings, as demonstrated by its deployment in around 15% of India's courtrooms. Evaluations on live data show consistent latency reduction as user concurrency increases, compared to sequential batch processing. The live demonstration will showcase our open-sourced implementation and allow attendees to interact with it in real-time.
>
---
#### [new 008] QHARMA-GAN: Quasi-Harmonic Neural Vocoder based on Autoregressive Moving Average Model
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于语音合成任务，旨在解决神经声码器黑箱性强、生成效率低的问题。提出QHARMA-GAN框架，结合ARMA模型与神经网络，提升合成质量与灵活性。**

- **链接: [http://arxiv.org/pdf/2507.01611v1](http://arxiv.org/pdf/2507.01611v1)**

> **作者:** Shaowen Chen; Tomoki Toda
>
> **备注:** This manuscript is currently under review for publication in the IEEE Transactions on Audio, Speech, and Language Processing. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Vocoders, encoding speech signals into acoustic features and allowing for speech signal reconstruction from them, have been studied for decades. Recently, the rise of deep learning has particularly driven the development of neural vocoders to generate high-quality speech signals. On the other hand, the existing end-to-end neural vocoders suffer from a black-box nature that blinds the speech production mechanism and the intrinsic structure of speech, resulting in the ambiguity of separately modeling source excitation and resonance characteristics and the loss of flexibly synthesizing or modifying speech with high quality. Moreover, their sequence-wise waveform generation usually requires complicated networks, leading to substantial time consumption. In this work, inspired by the quasi-harmonic model (QHM) that represents speech as sparse components, we combine the neural network and QHM synthesis process to propose a novel framework for the neural vocoder. Accordingly, speech signals can be encoded into autoregressive moving average (ARMA) functions to model the resonance characteristics, yielding accurate estimates of the amplitudes and phases of quasi-harmonics at any frequency. Subsequently, the speech can be resynthesized and arbitrarily modified in terms of pitch shifting and time stretching with high quality, whereas the time consumption and network size decrease. The experiments indicate that the proposed method leverages the strengths of QHM, the ARMA model, and neural networks, leading to the outperformance of our methods over other methods in terms of generation speed, synthesis quality, and modification flexibility.
>
---
#### [new 009] IdolSongsJp Corpus: A Multi-Singer Song Corpus in the Style of Japanese Idol Groups
- **分类: eess.AS; cs.SD**

- **简介: 该论文介绍了一个名为IdolSongsJp的日本偶像歌曲语料库，用于音乐信息处理任务，如歌手辨识和音源分离。研究者创作了15首符合偶像风格的歌曲，并提供多轨道音频和标注数据以支持相关技术评估。**

- **链接: [http://arxiv.org/pdf/2507.01349v1](http://arxiv.org/pdf/2507.01349v1)**

> **作者:** Hitoshi Suda; Junya Koguchi; Shunsuke Yoshida; Tomohiko Nakamura; Satoru Fukayama; Jun Ogata
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Japanese idol groups, comprising performers known as "idols," are an indispensable part of Japanese pop culture. They frequently appear in live concerts and television programs, entertaining audiences with their singing and dancing. Similar to other J-pop songs, idol group music covers a wide range of styles, with various types of chord progressions and instrumental arrangements. These tracks often feature numerous instruments and employ complex mastering techniques, resulting in high signal loudness. Additionally, most songs include a song division (utawari) structure, in which members alternate between singing solos and performing together. Hence, these songs are well-suited for benchmarking various music information processing techniques such as singer diarization, music source separation, and automatic chord estimation under challenging conditions. Focusing on these characteristics, we constructed a song corpus titled IdolSongsJp by commissioning professional composers to create 15 tracks in the style of Japanese idol groups. This corpus includes not only mastered audio tracks but also stems for music source separation, dry vocal tracks, and chord annotations. This paper provides a detailed description of the corpus, demonstrates its diversity through comparisons with real-world idol group songs, and presents its application in evaluating several music information processing techniques.
>
---
#### [new 010] SpeechAccentLLM: A Unified Framework for Foreign Accent Conversion and Text to Speech
- **分类: eess.AS; cs.SD; I.2.7**

- **简介: 该论文属于语音处理任务，解决外语 accents 转换问题。提出 SpeechAccentLLM 框架，结合 VAE 和多任务学习，提升转换效果与语音质量。**

- **链接: [http://arxiv.org/pdf/2507.01348v1](http://arxiv.org/pdf/2507.01348v1)**

> **作者:** Cheng Zhuangfei; Zhang Guangyan; Tu Zehai; Song Yangyang; Mao Shuiyang; Jiao Xiaoqi; Li Jingyu; Guo Yiwen; Wu Jiasong
>
> **备注:** 10 pages, includes references, 4 figures, 4 tables
>
> **摘要:** Foreign accent conversion (FAC) in speech processing remains a challenging task. Building on the remarkable success of large language models (LLMs) in Text-to-Speech (TTS) tasks, this study investigates the adaptation of LLM-based techniques for FAC, which we term SpeechAccentLLM. At the core of this framework, we introduce SpeechCodeVAE, the first model to integrate connectionist temporal classification (CTC) directly into codebook discretization for speech content tokenization. This novel architecture generates tokens with a unique "locality" property, as validated by experiments demonstrating optimal trade-offs among content faithfulness, temporal coherence, and structural recoverability. Then, to address data scarcity for the FAC module, we adopted a multitask learning strategy that jointly trains the FAC and TTS modules. Beyond mitigating data limitations, this approach yielded accelerated convergence and superior speech quality compared to standalone FAC training. Moreover, leveraging the salient properties of our discrete speech representations, we introduce SpeechRestorer, a postprocessing architecture designed to refine LLM-generated outputs. This module effectively mitigates stochastic errors prevalent in LLM inference pipelines while enhancing prosodic continuity, as validated by ablation experiments.
>
---
#### [new 011] Voice Conversion for Likability Control via Automated Rating of Speech Synthesis Corpora
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音转换任务，旨在控制语音的吸引力。通过训练预测模型自动标注数据，实现对语音 likability 的有效控制，同时保持说话人身份和内容。**

- **链接: [http://arxiv.org/pdf/2507.01356v1](http://arxiv.org/pdf/2507.01356v1)**

> **作者:** Hitoshi Suda; Shinnosuke Takamichi; Satoru Fukayama
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Perceived voice likability plays a crucial role in various social interactions, such as partner selection and advertising. A system that provides reference likable voice samples tailored to target audiences would enable users to adjust their speaking style and voice quality, facilitating smoother communication. To this end, we propose a voice conversion method that controls the likability of input speech while preserving both speaker identity and linguistic content. To improve training data scalability, we train a likability predictor on an existing voice likability dataset and employ it to automatically annotate a large speech synthesis corpus with likability ratings. Experimental evaluations reveal a significant correlation between the predictor's outputs and human-provided likability ratings. Subjective and objective evaluations further demonstrate that the proposed approach effectively controls voice likability while preserving both speaker identity and linguistic content.
>
---
#### [new 012] Low-Complexity Neural Wind Noise Reduction for Audio Recordings
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频降噪任务，旨在解决户外录音中的风噪问题。提出一种低复杂度的单通道神经网络模型，有效抑制风噪，适用于嵌入式设备。**

- **链接: [http://arxiv.org/pdf/2507.01821v1](http://arxiv.org/pdf/2507.01821v1)**

> **作者:** Hesam Eftekhari; Srikanth Raj Chetupalli; Shrishti Saha Shetu; Emanuël A. P. Habets; Oliver Thiergart
>
> **摘要:** Wind noise significantly degrades the quality of outdoor audio recordings, yet remains difficult to suppress in real-time on resource-constrained devices. In this work, we propose a low-complexity single-channel deep neural network that leverages the spectral characteristics of wind noise. Experimental results show that our method achieves performance comparable to the state-of-the-art low-complexity ULCNet model. The proposed model, with only 249K parameters and roughly 73 MHz of computational power, is suitable for embedded and mobile audio applications.
>
---
#### [new 013] Workflow-Based Evaluation of Music Generation Systems
- **分类: eess.AS; cs.HC; cs.LG; cs.MM; cs.SD**

- **简介: 该论文属于音乐生成系统评估任务，旨在解决MGS在音乐制作流程中的应用问题，通过实验分析其功能与局限性，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2507.01022v1](http://arxiv.org/pdf/2507.01022v1)**

> **作者:** Shayan Dadman; Bernt Arild Bremdal; Andreas Bergsland
>
> **备注:** 54 pages, 3 figures, 6 tables, 5 appendices
>
> **摘要:** This study presents an exploratory evaluation of Music Generation Systems (MGS) within contemporary music production workflows by examining eight open-source systems. The evaluation framework combines technical insights with practical experimentation through criteria specifically designed to investigate the practical and creative affordances of the systems within the iterative, non-linear nature of music production. Employing a single-evaluator methodology as a preliminary phase, this research adopts a mixed approach utilizing qualitative methods to form hypotheses subsequently assessed through quantitative metrics. The selected systems represent architectural diversity across both symbolic and audio-based music generation approaches, spanning composition, arrangement, and sound design tasks. The investigation addresses limitations of current MGS in music production, challenges and opportunities for workflow integration, and development potential as collaborative tools while maintaining artistic authenticity. Findings reveal these systems function primarily as complementary tools enhancing rather than replacing human expertise. They exhibit limitations in maintaining thematic and structural coherence that emphasize the indispensable role of human creativity in tasks demanding emotional depth and complex decision-making. This study contributes a structured evaluation framework that considers the iterative nature of music creation. It identifies methodological refinements necessary for subsequent comprehensive evaluations and determines viable areas for AI integration as collaborative tools in creative workflows. The research provides empirically-grounded insights to guide future development in the field.
>
---
#### [new 014] A Review on Sound Source Localization in Robotics: Focusing on Deep Learning Methods
- **分类: cs.RO; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于声音源定位任务，解决机器人中声源定位问题，综述了传统方法和深度学习方法，分析了数据与训练策略，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.01143v1](http://arxiv.org/pdf/2507.01143v1)**

> **作者:** Reza Jalayer; Masoud Jalayer; Amirali Baniasadi
>
> **备注:** 35 pages
>
> **摘要:** Sound source localization (SSL) adds a spatial dimension to auditory perception, allowing a system to pinpoint the origin of speech, machinery noise, warning tones, or other acoustic events, capabilities that facilitate robot navigation, human-machine dialogue, and condition monitoring. While existing surveys provide valuable historical context, they typically address general audio applications and do not fully account for robotic constraints or the latest advancements in deep learning. This review addresses these gaps by offering a robotics-focused synthesis, emphasizing recent progress in deep learning methodologies. We start by reviewing classical methods such as Time Difference of Arrival (TDOA), beamforming, Steered-Response Power (SRP), and subspace analysis. Subsequently, we delve into modern machine learning (ML) and deep learning (DL) approaches, discussing traditional ML and neural networks (NNs), convolutional neural networks (CNNs), convolutional recurrent neural networks (CRNNs), and emerging attention-based architectures. The data and training strategy that are the two cornerstones of DL-based SSL are explored. Studies are further categorized by robot types and application domains to facilitate researchers in identifying relevant work for their specific contexts. Finally, we highlight the current challenges in SSL works in general, regarding environmental robustness, sound source multiplicity, and specific implementation constraints in robotics, as well as data and learning strategies in DL-based SSL. Also, we sketch promising directions to offer an actionable roadmap toward robust, adaptable, efficient, and explainable DL-based SSL for next-generation robots.
>
---
#### [new 015] Adaptability of ASR Models on Low-Resource Language: A Comparative Study of Whisper and Wav2Vec-BERT on Bangla
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，研究如何提升低资源语言（如孟加拉语）的ASR性能，对比了Whisper和Wav2Vec-BERT模型的效果与效率。**

- **链接: [http://arxiv.org/pdf/2507.01931v1](http://arxiv.org/pdf/2507.01931v1)**

> **作者:** Md Sazzadul Islam Ridoy; Sumi Akter; Md. Aminur Rahman
>
> **摘要:** In recent years, neural models trained on large multilingual text and speech datasets have shown great potential for supporting low-resource languages. This study investigates the performances of two state-of-the-art Automatic Speech Recognition (ASR) models, OpenAI's Whisper (Small & Large-V2) and Facebook's Wav2Vec-BERT on Bangla, a low-resource language. We have conducted experiments using two publicly available datasets: Mozilla Common Voice-17 and OpenSLR to evaluate model performances. Through systematic fine-tuning and hyperparameter optimization, including learning rate, epochs, and model checkpoint selection, we have compared the models based on Word Error Rate (WER), Character Error Rate (CER), Training Time, and Computational Efficiency. The Wav2Vec-BERT model outperformed Whisper across all key evaluation metrics, demonstrated superior performance while requiring fewer computational resources, and offered valuable insights to develop robust speech recognition systems in low-resource linguistic settings.
>
---
## 更新

#### [replaced 001] Multi-interaction TTS toward professional recording reproduction
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.00808v2](http://arxiv.org/pdf/2507.00808v2)**

> **作者:** Hiroki Kanagawa; Kenichi Fujita; Aya Watanabe; Yusuke Ijima
>
> **备注:** 7 pages,6 figures, Accepted to Speech Synthesis Workshop 2025 (SSW13)
>
> **摘要:** Voice directors often iteratively refine voice actors' performances by providing feedback to achieve the desired outcome. While this iterative feedback-based refinement process is important in actual recordings, it has been overlooked in text-to-speech synthesis (TTS). As a result, fine-grained style refinement after the initial synthesis is not possible, even though the synthesized speech often deviates from the user's intended style. To address this issue, we propose a TTS method with multi-step interaction that allows users to intuitively and rapidly refine synthesized speech. Our approach models the interaction between the TTS model and its user to emulate the relationship between voice actors and voice directors. Experiments show that the proposed model with its corresponding dataset enables iterative style refinements in accordance with users' directions, thus demonstrating its multi-interaction capability. Sample audios are available: https://ntt-hilab-gensp.github.io/ssw13multiinteractiontts/
>
---
#### [replaced 002] Embedding-Space Diffusion for Zero-Shot Environmental Sound Classification
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.03771v2](http://arxiv.org/pdf/2412.03771v2)**

> **作者:** Ysobel Sims; Alexandre Mendes; Stephan Chalup
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Zero-shot learning enables models to generalise to unseen classes by leveraging semantic information, bridging the gap between training and testing sets with non-overlapping classes. While much research has focused on zero-shot learning in computer vision, the application of these methods to environmental audio remains underexplored, with poor performance in existing studies. Generative methods, which have demonstrated success in computer vision, are notably absent from zero-shot environmental sound classification studies. To address this gap, this work investigates generative methods for zero-shot learning in environmental audio. Two successful generative models from computer vision are adapted: a cross-aligned and distribution-aligned variational autoencoder (CADA-VAE) and a leveraging invariant side generative adversarial network (LisGAN). Additionally, we introduced a novel diffusion model conditioned on class auxiliary data. Synthetic embeddings generated by the diffusion model are combined with seen class embeddings to train a classifier. Experiments are conducted on five environmental audio datasets, ESC-50, ARCA23K-FSD, FSC22, UrbanSound8k and TAU Urban Acoustics 2019, and one music classification dataset, GTZAN. Results show that the diffusion model outperforms all baseline methods on average across six audio datasets. This work establishes the diffusion model as a promising approach for zero-shot learning and introduces the first benchmark of generative methods for zero-shot environmental sound classification, providing a foundation for future research.
>
---
#### [replaced 003] TARO: Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning for Synchronized Video-to-Audio Synthesis
- **分类: cs.SD; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.05684v2](http://arxiv.org/pdf/2504.05684v2)**

> **作者:** Tri Ton; Ji Woo Hong; Chang D. Yoo
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** This paper introduces Timestep-Adaptive Representation Alignment with Onset-Aware Conditioning (TARO), a novel framework for high-fidelity and temporally coherent video-to-audio synthesis. Built upon flow-based transformers, which offer stable training and continuous transformations for enhanced synchronization and audio quality, TARO introduces two key innovations: (1) Timestep-Adaptive Representation Alignment (TRA), which dynamically aligns latent representations by adjusting alignment strength based on the noise schedule, ensuring smooth evolution and improved fidelity, and (2) Onset-Aware Conditioning (OAC), which integrates onset cues that serve as sharp event-driven markers of audio-relevant visual moments to enhance synchronization with dynamic visual events. Extensive experiments on the VGGSound and Landscape datasets demonstrate that TARO outperforms prior methods, achieving relatively 53% lower Frechet Distance (FD), 29% lower Frechet Audio Distance (FAD), and a 97.19% Alignment Accuracy, highlighting its superior audio quality and synchronization precision.
>
---
#### [replaced 004] PSELDNets: Pre-trained Neural Networks on a Large-scale Synthetic Dataset for Sound Event Localization and Detection
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2411.06399v2](http://arxiv.org/pdf/2411.06399v2)**

> **作者:** Jinbo Hu; Yin Cao; Ming Wu; Fang Kang; Feiran Yang; Wenwu Wang; Mark D. Plumbley; Jun Yang
>
> **备注:** 16 pages, 9 figures, accepted by IEEE Transactions on Audio, Speech, and Language Processing. The code is available at https://github.com/Jinbo-Hu/PSELDNets
>
> **摘要:** Sound event localization and detection (SELD) has seen substantial advancements through learning-based methods. These systems, typically trained from scratch on specific datasets, have shown considerable generalization capabilities. Recently, deep neural networks trained on large-scale datasets have achieved remarkable success in the sound event classification (SEC) field, prompting an open question of whether these advances can be extended to the development of SELD foundation models. In this paper, leveraging the power of pre-trained SEC models, we propose pre-trained SELD networks (PSELDNets) on a large-scale synthetic dataset. The synthetic dataset, generated by convolving sound events with simulated spatial room impulse responses (SRIRs), contains 1,167 hours of audio clips with an ontology of 170 sound classes. These PSELDNets are applied to various SELD scenarios. When we adapt PSELDNets to specific scenarios, particularly in cases of low-resource data, we introduce a data-efficient fine-tuning method, AdapterBit. PSELDNets are evaluated on synthetic-test-set using collected SRIRs from the TAU Spatial Room Impulse Response Database (TAU-SRIR DB) and achieve satisfactory performance. We also carried out experiments to validate the transferability of PSELDNets to three publicly available datasets and our own real-world recordings. The results demonstrate that PSELDNets surpass state-of-the-art systems across all publicly available datasets. Given the need for direction-of-arrival estimation, SELD generally relies on sufficient multi-channel audio clips. However, incorporating the AdapterBit, PSELDNets show more efficient adaptability to various scenarios using minimal multi-channel or even just monophonic audio clips, outperforming traditional fine-tuning approaches.
>
---
#### [replaced 005] Real-Time AIoT for AAV Antenna Interference Detection via Edge-Cloud Collaboration
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03055v3](http://arxiv.org/pdf/2412.03055v3)**

> **作者:** Jun Dong; Jintao Cheng; Jin Wu; Chengxi Zhang; Shunyi Zhao; Xiaoyu Tang
>
> **摘要:** In the fifth-generation (5G) era, eliminating communication interference sources is crucial for maintaining network performance. Interference often originates from unauthorized or malfunctioning antennas, and radio monitoring agencies must address numerous sources of such antennas annually. Unmanned aerial vehicles (UAVs) can improve inspection efficiency. However, the data transmission delay in the existing cloud-only (CO) artificial intelligence (AI) mode fails to meet the low latency requirements for real-time performance. Therefore, we propose a computer vision-based AI of Things (AIoT) system to detect antenna interference sources for UAVs. The system adopts an optimized edge-cloud collaboration (ECC+) mode, combining a keyframe selection algorithm (KSA), focusing on reducing end-to-end latency (E2EL) and ensuring reliable data transmission, which aligns with the core principles of ultra-reliable low-latency communication (URLLC). At the core of our approach is an end-to-end antenna localization scheme based on the tracking-by-detection (TBD) paradigm, including a detector (EdgeAnt) and a tracker (AntSort). EdgeAnt achieves state-of-the-art (SOTA) performance with a mean average precision (mAP) of 42.1% on our custom antenna interference source dataset, requiring only 3 million parameters and 14.7 GFLOPs. On the COCO dataset, EdgeAnt achieves 38.9% mAP with 5.4 GFLOPs. We deployed EdgeAnt on Jetson Xavier NX (TRT) and Raspberry Pi 4B (NCNN), achieving real-time inference speeds of 21.1 (1088) and 4.8 (640) frames per second (FPS), respectively. Compared with CO mode, the ECC+ mode reduces E2EL by 88.9%, increases accuracy by 28.2%. Additionally, the system offers excellent scalability for coordinated multiple UAVs inspections. The detector code is publicly available at https://github.com/SCNU-RISLAB/EdgeAnt.
>
---
#### [replaced 006] An accurate measurement of parametric array using a spurious sound filter topologically equivalent to a half-wavelength resonator
- **分类: cs.SD; eess.AS; physics.app-ph**

- **链接: [http://arxiv.org/pdf/2504.12398v3](http://arxiv.org/pdf/2504.12398v3)**

> **作者:** Woongji Kim; Beomseok Oh; Junsuk Rho; Wonkyu Moon
>
> **备注:** 12 pages, 11 figures. Published in Applied Acoustics
>
> **摘要:** Parametric arrays (PA) offer exceptional directivity and compactness compared to conventional loudspeakers, facilitating various acoustic applications. However, accurate measurement of audio signals generated by PA remains challenging due to spurious ultrasonic sounds arising from microphone nonlinearities. Existing filtering methods, including Helmholtz resonators, phononic crystals, polymer films, and grazing incidence techniques, exhibit practical constraints such as size limitations, fabrication complexity, or insufficient attenuation. To address these issues, we propose and demonstrate a novel acoustic filter based on the design of a half-wavelength resonator. The developed filter exploits the nodal plane in acoustic pressure distribution, effectively minimizing microphone exposure to targeted ultrasonic frequencies. Fabrication via stereolithography (SLA) 3D printing ensures high dimensional accuracy, which is crucial for high-frequency acoustic filters. Finite element method (FEM) simulations guided filter optimization for suppression frequencies at 40 kHz and 60 kHz, achieving high transmission loss (TL) around 60 dB. Experimental validations confirm the filter's superior performance in significantly reducing spurious acoustic signals, as reflected in frequency response, beam pattern, and propagation curve measurements. The proposed filter ensures stable and precise acoustic characterization, independent of measurement distances and incidence angles. This new approach not only improves measurement accuracy but also enhances reliability and reproducibility in parametric array research and development.
>
---
#### [replaced 007] Unifying Global and Near-Context Biasing in a Single Trie Pass
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.13514v2](http://arxiv.org/pdf/2409.13514v2)**

> **作者:** Iuliia Thorbecke; Esaú Villatoro-Tello; Juan Zuluaga-Gomez; Shashi Kumar; Sergio Burdisso; Pradeep Rangappa; Andrés Carofilis; Srikanth Madikeri; Petr Motlicek; Karthik Pandia; Kadri Hacioğlu; Andreas Stolcke
>
> **备注:** Accepted to TSD2025
>
> **摘要:** Despite the success of end-to-end automatic speech recognition (ASR) models, challenges persist in recognizing rare, out-of-vocabulary words - including named entities (NE) - and in adapting to new domains using only text data. This work presents a practical approach to address these challenges through an unexplored combination of an NE bias list and a word-level n-gram language model (LM). This solution balances simplicity and effectiveness, improving entities' recognition while maintaining or even enhancing overall ASR performance. We efficiently integrate this enriched biasing method into a transducer-based ASR system, enabling context adaptation with almost no computational overhead. We present our results on three datasets spanning four languages and compare them to state-of-the-art biasing strategies. We demonstrate that the proposed combination of keyword biasing and n-gram LM improves entity recognition by up to 32% relative and reduces overall WER by up to a 12% relative.
>
---
#### [replaced 008] Melody predominates over harmony in the evolution of musical scales across 96 countries
- **分类: cs.SD; eess.AS; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2408.12633v3](http://arxiv.org/pdf/2408.12633v3)**

> **作者:** John M McBride; Elizabeth Phillips; Patrick E Savage; Steven Brown; Tsvi Tlusty
>
> **摘要:** The standard theory of musical scales since antiquity has been based on harmony, rather than melody. While recent analyses provide mixed support for a role of melody as well as harmony, we lack a comparative analysis based on cross-cultural data. We address this longstanding problem through a rigorous computational comparison of the main theories using 1,314 scales from 96 countries. There is near-universal support for melodic theories, which predict step-sizes of 1-3 semitones. Harmony accounts for the prevalence of certain simple-integer-ratio intervals, particularly for music-theoretic scales from Eurasian societies, which may explain their dominance amongst Western scholars. However, harmony is a poor predictor of scales measured from ethnographic recordings, particularly outside of Eurasia. Overall, we show that the historical emphasis on harmony is misguided and that melody is the primary determinant of the world's musical scales.
>
---
