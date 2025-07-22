# 音频 cs.SD;  eess.SP

- **最新发布 21 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于语音伪造检测任务，旨在解决部分伪造语音检测中依赖昂贵帧级标注的问题。通过分析帧间时间差异，发现伪造语音具有不自然的局部变化模式。提出TDAM模块与双层差异表示方法，结合自适应平均池化，在无需帧级标注的情况下实现高效检测，取得最优性能。**

- **链接: [http://arxiv.org/pdf/2507.15101v1](http://arxiv.org/pdf/2507.15101v1)**

> **作者:** Menglu Li; Xiao-Ping Zhang; Lian Zhao
>
> **备注:** 5 pages, 4 figures, 4 tables. Accepted to IEEE SPL
>
> **摘要:** Detecting partial deepfake speech is essential due to its potential for subtle misinformation. However, existing methods depend on costly frame-level annotations during training, limiting real-world scalability. Also, they focus on detecting transition artifacts between bonafide and deepfake segments. As deepfake generation techniques increasingly smooth these transitions, detection has become more challenging. To address this, our work introduces a new perspective by analyzing frame-level temporal differences and reveals that deepfake speech exhibits erratic directional changes and unnatural local transitions compared to bonafide speech. Based on this finding, we propose a Temporal Difference Attention Module (TDAM) that redefines partial deepfake detection as identifying unnatural temporal variations, without relying on explicit boundary annotations. A dual-level hierarchical difference representation captures temporal irregularities at both fine and coarse scales, while adaptive average pooling preserves essential patterns across variable-length inputs to minimize information loss. Our TDAM-AvgPool model achieves state-of-the-art performance, with an EER of 0.59% on the PartialSpoof dataset and 0.03% on the HAD dataset, which significantly outperforms the existing methods without requiring frame-level supervision.
>
---
#### [new 002] Exploiting Context-dependent Duration Features for Voice Anonymization Attack Systems
- **分类: cs.SD; cs.CL; cs.CR; eess.AS**

- **简介: 该论文属于语音匿名化与攻击分析任务，旨在研究语音中的时序动态特征对说话人识别的影响。论文提出了一种基于上下文相关时长嵌入的说话人特征表示方法，并构建了攻击模型，用于评估语音匿名化系统的安全性。实验表明，该方法在原始和匿名化语音上均提升了攻击效果。**

- **链接: [http://arxiv.org/pdf/2507.15214v1](http://arxiv.org/pdf/2507.15214v1)**

> **作者:** Natalia Tomashenko; Emmanuel Vincent; Marc Tommasi
>
> **备注:** Accepted at Interspeech-2025
>
> **摘要:** The temporal dynamics of speech, encompassing variations in rhythm, intonation, and speaking rate, contain important and unique information about speaker identity. This paper proposes a new method for representing speaker characteristics by extracting context-dependent duration embeddings from speech temporal dynamics. We develop novel attack models using these representations and analyze the potential vulnerabilities in speaker verification and voice anonymization systems.The experimental results show that the developed attack models provide a significant improvement in speaker verification performance for both original and anonymized data in comparison with simpler representations of speech temporal dynamics reported in the literature.
>
---
#### [new 003] Neuro-MSBG: An End-to-End Neural Model for Hearing Loss Simulation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音信号处理任务，旨在解决传统听力损失模拟模型计算复杂、延迟高的问题。作者提出了Neuro-MSBG，一个轻量级端到端神经网络模型，通过个性化听力图编码器实现高效时频建模。实验表明，该模型在保持语音可懂度和感知质量的同时，大幅提升了计算效率，适用于实时应用。**

- **链接: [http://arxiv.org/pdf/2507.15396v1](http://arxiv.org/pdf/2507.15396v1)**

> **作者:** Hui-Guan Yuan; Ryandhimas E. Zezario; Shafique Ahmed; Hsin-Min Wang; Kai-Lung Hua; Yu Tsao
>
> **摘要:** Hearing loss simulation models are essential for hearing aid deployment. However, existing models have high computational complexity and latency, which limits real-time applications and lack direct integration with speech processing systems. To address these issues, we propose Neuro-MSBG, a lightweight end-to-end model with a personalized audiogram encoder for effective time-frequency modeling. Experiments show that Neuro-MSBG supports parallel inference and retains the intelligibility and perceptual quality of the original MSBG, with a Spearman's rank correlation coefficient (SRCC) of 0.9247 for Short-Time Objective Intelligibility (STOI) and 0.8671 for Perceptual Evaluation of Speech Quality (PESQ). Neuro-MSBG reduces simulation runtime by a factor of 46 (from 0.970 seconds to 0.021 seconds for a 1 second input), further demonstrating its efficiency and practicality.
>
---
#### [new 004] Multichannel Keyword Spotting for Noisy Conditions
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决噪声环境下关键词识别性能下降的问题。作者提出一种多通道注意力机制神经网络架构，提升关键词识别的鲁棒性，并在多种数据集上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.15558v1](http://arxiv.org/pdf/2507.15558v1)**

> **作者:** Dzmitry Saladukha; Ivan Koriabkin; Kanstantsin Artsiom; Aliaksei Rak; Nikita Ryzhikov
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** This article presents a method for improving a keyword spotter (KWS) algorithm in noisy environments. Although beamforming (BF) and adaptive noise cancellation (ANC) techniques are robust in some conditions, they may degrade the performance of the activation system by distorting or suppressing useful signals. The authors propose a neural network architecture that uses several input channels and an attention mechanism that allows the network to determine the most useful channel or their combination. The improved quality of the algorithm was demonstrated on two datasets: from a laboratory with controlled conditions and from smart speakers in natural conditions. The proposed algorithm was compared against several baselines in terms of the quality of noise reduction metrics, KWS metrics, and computing resources in comparison with existing solutions.
>
---
#### [new 005] The Rest is Silence: Leveraging Unseen Species Models for Computational Musicology
- **分类: cs.SD; eess.AS; stat.AP**

- **简介: 该论文属于计算音乐学任务，旨在解决音乐学数据不完整性问题。通过引入生态学中的“未见物种模型”（USMs），对音乐数据集的缺失部分进行定量估计。论文通过四个案例研究展示模型应用，如估计RISM缺失的作曲家人数、格里高利圣咏来源的编目比例、不同乐谱版本间的差异预期、民俗音乐传统中歌曲的覆盖率，以及作曲家和声词汇的规模估计。**

- **链接: [http://arxiv.org/pdf/2507.14638v1](http://arxiv.org/pdf/2507.14638v1)**

> **作者:** Fabian C. Moss; Jan Hajič jr.; Adrian Nachtwey; Laurent Pugin
>
> **摘要:** For many decades, musicologists have engaged in creating large databases serving different purposes for musicological research and scholarship. With the rise of fields like music information retrieval and digital musicology, there is now a constant and growing influx of musicologically relevant datasets and corpora. In historical or observational settings, however, these datasets are necessarily incomplete, and the true extent of a collection of interest remains unknown -- silent. Here, we apply, for the first time, so-called Unseen Species models (USMs) from ecology to areas of musicological activity. After introducing the models formally, we show in four case studies how USMs can be applied to musicological data to address quantitative questions like: How many composers are we missing in RISM? What percentage of medieval sources of Gregorian chant have we already cataloged? How many differences in music prints do we expect to find between editions? How large is the coverage of songs from genres of a folk music tradition? And, finally, how close are we in estimating the size of the harmonic vocabulary of a large number of composers?
>
---
#### [new 006] Multi-Sampling-Frequency Naturalness MOS Prediction Using Self-Supervised Learning Model with Sampling-Frequency-Independent Layer
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决多采样频率下语音MOS预测问题。作者提出了一种结合自监督学习与采样频率无关卷积层的模型，并采用知识蒸馏和大规模数据预训练提升性能，在AMC 2025 Track 3中取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2507.14647v1](http://arxiv.org/pdf/2507.14647v1)**

> **作者:** Go Nishikawa; Wataru Nakata; Yuki Saito; Kanami Imamura; Hiroshi Saruwatari; Tomohiko Nakamura
>
> **备注:** 4 pages, 2 figures
>
> **摘要:** We introduce our submission to the AudioMOS Challenge (AMC) 2025 Track 3: mean opinion score (MOS) prediction for speech with multiple sampling frequencies (SFs). Our submitted model integrates an SF-independent (SFI) convolutional layer into a self-supervised learning (SSL) model to achieve SFI speech feature extraction for MOS prediction. We present some strategies to improve the MOS prediction performance of our model: distilling knowledge from a pretrained non-SFI-SSL model and pretraining with a large-scale MOS dataset. Our submission to the AMC 2025 Track 3 ranked the first in one evaluation metric and the fourth in the final ranking. We also report the results of our ablation study to investigate essential factors of our model.
>
---
#### [new 007] A2TTS: TTS for Low Resource Indian Languages
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决低资源印度语言中对未见说话人生成自然语音的问题。作者提出了一个基于扩散模型的TTS系统，结合说话人编码器和注意力机制，提升多说话人语音生成的自然度与准确性，并支持多种印度语言。**

- **链接: [http://arxiv.org/pdf/2507.15272v1](http://arxiv.org/pdf/2507.15272v1)**

> **作者:** Ayush Singh Bhadoriya; Abhishek Nikunj Shinde; Isha Pandey; Ganesh Ramakrishnan
>
> **摘要:** We present a speaker conditioned text-to-speech (TTS) system aimed at addressing challenges in generating speech for unseen speakers and supporting diverse Indian languages. Our method leverages a diffusion-based TTS architecture, where a speaker encoder extracts embeddings from short reference audio samples to condition the DDPM decoder for multispeaker generation. To further enhance prosody and naturalness, we employ a cross-attention based duration prediction mechanism that utilizes reference audio, enabling more accurate and speaker consistent timing. This results in speech that closely resembles the target speaker while improving duration modeling and overall expressiveness. Additionally, to improve zero-shot generation, we employed classifier free guidance, allowing the system to generate speech more near speech for unknown speakers. Using this approach, we trained language-specific speaker-conditioned models. Using the IndicSUPERB dataset for multiple Indian languages such as Bengali, Gujarati, Hindi, Marathi, Malayalam, Punjabi and Tamil.
>
---
#### [new 008] EchoVoices: Preserving Generational Voices and Memories for Seniors and Children
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音与语言处理任务，旨在解决老年人与儿童语音识别与合成效果不佳的问题。作者提出了EchoVoices系统，结合改进的语音识别、语音合成与大语言模型，用于保存老年人与儿童的声音与记忆，提升跨代交流与数字遗产的持久性。**

- **链接: [http://arxiv.org/pdf/2507.15221v1](http://arxiv.org/pdf/2507.15221v1)**

> **作者:** Haiying Xu; Haoze Liu; Mingshi Li; Siyu Cai; Guangxuan Zheng; Yuhuang Jia; Jinghua Zhao; Yong Qin
>
> **摘要:** Recent breakthroughs in intelligent speech and digital human technologies have primarily targeted mainstream adult users, often overlooking the distinct vocal patterns and interaction styles of seniors and children. These demographics possess distinct vocal characteristics, linguistic styles, and interaction patterns that challenge conventional ASR, TTS, and LLM systems. To address this, we introduce EchoVoices, an end-to-end digital human pipeline dedicated to creating persistent digital personas for seniors and children, ensuring their voices and memories are preserved for future generations. Our system integrates three core innovations: a k-NN-enhanced Whisper model for robust speech recognition of atypical speech; an age-adaptive VITS model for high-fidelity, speaker-aware speech synthesis; and an LLM-driven agent that automatically generates persona cards and leverages a RAG-based memory system for conversational consistency. Our experiments, conducted on the SeniorTalk and ChildMandarin datasets, demonstrate significant improvements in recognition accuracy, synthesis quality, and speaker similarity. EchoVoices provides a comprehensive framework for preserving generational voices, offering a new means of intergenerational connection and the creation of lasting digital legacies.
>
---
#### [new 009] Traffic Signal Phase and Timing Estimation with Large-Scale Floating Car Data
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于交通信号分析任务，旨在解决准确估计信号相位与配时（SPaT）的问题。由于传统方法依赖固定假设且适用性有限，论文提出了一种基于大规模浮动车数据（FCD）的工业级分析框架，实现从数据预处理到SPaT估计的全流程处理。该方法适应不同路口结构和周期性变化，具备强鲁棒性，并已应用于实际导航平台。**

- **链接: [http://arxiv.org/pdf/2507.14190v1](http://arxiv.org/pdf/2507.14190v1)**

> **作者:** Mingcheng Liao; Zebang Feng; Miao Fan; Shengtong Xu; Haoyi Xiong
>
> **备注:** Accepted by ITSC'25
>
> **摘要:** Effective modern transportation systems depend critically on accurate Signal Phase and Timing (SPaT) estimation. However, acquiring ground-truth SPaT information faces significant hurdles due to communication challenges with transportation departments and signal installers. As a result, Floating Car Data (FCD) has become the primary source for large-scale SPaT analyses. Current FCD approaches often simplify the problem by assuming fixed schedules and basic intersection designs for specific times and locations. These methods fail to account for periodic signal changes, diverse intersection structures, and the inherent limitations of real-world data, thus lacking a comprehensive framework that is universally applicable. Addressing this limitation, we propose an industrial-grade FCD analysis suite that manages the entire process, from initial data preprocessing to final SPaT estimation. Our approach estimates signal phases, identifies time-of-day (TOD) periods, and determines the durations of red and green lights. The framework's notable stability and robustness across diverse conditions, regardless of road geometry, is a key feature. Furthermore, we provide a cleaned, de-identified FCD dataset and supporting parameters to facilitate future research. Currently operational within our navigation platform, the system analyses over 15 million FCD records daily, supporting over two million traffic signals in mainland China, with more than 75\% of estimations demonstrating less than five seconds of error.
>
---
#### [new 010] U-DREAM: Unsupervised Dereverberation guided by a Reverberation Model
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于语音信号处理任务，旨在解决缺乏成对干净与混响数据的语音去混响问题。作者提出U-DREAM方法，通过基于混响模型的无监督学习策略，仅使用混响信号和声学模型进行训练，有效提升了低资源场景下的去混响效果。**

- **链接: [http://arxiv.org/pdf/2507.14237v1](http://arxiv.org/pdf/2507.14237v1)**

> **作者:** Louis Bahrman; Mathieu Fontaine; Gaël Richard
>
> **备注:** Submitted to IEEE Transactions on Audio, Speech and Language Processing (TASLPRO)
>
> **摘要:** This paper explores the outcome of training state-ofthe-art dereverberation models with supervision settings ranging from weakly-supervised to fully unsupervised, relying solely on reverberant signals and an acoustic model for training. Most of the existing deep learning approaches typically require paired dry and reverberant data, which are difficult to obtain in practice. We develop instead a sequential learning strategy motivated by a bayesian formulation of the dereverberation problem, wherein acoustic parameters and dry signals are estimated from reverberant inputs using deep neural networks, guided by a reverberation matching loss. Our most data-efficient variant requires only 100 reverberation-parameter-labelled samples to outperform an unsupervised baseline, demonstrating the effectiveness and practicality of the proposed method in low-resource scenarios.
>
---
#### [new 011] School Attendance Control System Based on RFID Technology with Raspberry Pi and Arduino: EDURFID
- **分类: eess.SP; cs.CY**

- **简介: 论文设计了基于RFID技术的校园考勤系统EDURFID，旨在解决秘鲁农村学校考勤效率低的问题。系统使用Raspberry Pi、Arduino和RC522模块，结合Python Django开发网页平台，实现高精度、低延迟的自动化考勤，节省行政时间并提供实时报表。**

- **链接: [http://arxiv.org/pdf/2507.14191v1](http://arxiv.org/pdf/2507.14191v1)**

> **作者:** Cliver Oliver Turpo Benique
>
> **备注:** 27 pages, 4 figures. Educational technology system for rural schools in Peru. Implements RFID-based attendance control using open-source hardware (Raspberry Pi, Arduino). System validation conducted at T\'upac Amaru Secondary Educational Institution, Coasa, Puno
>
> **摘要:** This paper presents EDURFID, an automated school attendance control system based on RFID technology designed for rural educational institutions in Peru. The system integrates open-source hardware (Raspberry Pi 5, Arduino UNO R3) with RC522 RFID modules operating at 13.56 MHz, implementing a web architecture developed in Python Django. The system demonstrates 100% precision in RFID readings with 0.03-second response time, achieving 94% cost reduction compared to commercial solutions. Validation at T\'upac Amaru Secondary Educational Institution showed successful automation of attendance processes, saving 50 daily minutes of administrative time while providing real-time reporting capabilities.
>
---
#### [new 012] MeMo: Attentional Momentum for Real-time Audio-visual Speaker Extraction under Impaired Visual Conditions
- **分类: cs.SD; cs.MM**

- **简介: 该论文属于音视频目标说话人提取（AV-TSE）任务，旨在解决在视觉线索缺失或严重退化时系统无法准确提取目标说话人的问题。作者提出MeMo框架，通过引入两个自适应记忆库来维持注意力，实现即使在无视觉线索下也能持续跟踪目标说话人。实验表明该方法相较基线提升了至少2 dB的SI-SNR。**

- **链接: [http://arxiv.org/pdf/2507.15294v1](http://arxiv.org/pdf/2507.15294v1)**

> **作者:** Junjie Li; Wenxuan Wu; Shuai Wang; Zexu Pan; Kong Aik Lee; Helen Meng; Haizhou Li
>
> **摘要:** Audio-visual Target Speaker Extraction (AV-TSE) aims to isolate a target speaker's voice from multi-speaker environments by leveraging visual cues as guidance. However, the performance of AV-TSE systems heavily relies on the quality of these visual cues. In extreme scenarios where visual cues are missing or severely degraded, the system may fail to accurately extract the target speaker. In contrast, humans can maintain attention on a target speaker even in the absence of explicit auxiliary information. Motivated by such human cognitive ability, we propose a novel framework called MeMo, which incorporates two adaptive memory banks to store attention-related information. MeMo is specifically designed for real-time scenarios: once initial attention is established, the system maintains attentional momentum over time, even when visual cues become unavailable. We conduct comprehensive experiments to verify the effectiveness of MeMo. Experimental results demonstrate that our proposed framework achieves SI-SNR improvements of at least 2 dB over the corresponding baseline.
>
---
#### [new 013] Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音评估任务，旨在解决发音错误检测问题。现有方法难以准确识别因口音或不流利导致的发音差异。作者提出一种基于多任务训练的语音素相似性建模框架，并开源含发音错误的模拟数据集VCTK-accent，同时设计新评估指标，建立发音错误检测的新基准。**

- **链接: [http://arxiv.org/pdf/2507.14346v1](http://arxiv.org/pdf/2507.14346v1)**

> **作者:** Xuanru Zhou; Jiachen Lian; Cheol Jun Cho; Tejas Prabhune; Shuhe Li; William Li; Rodrigo Ortiz; Zoe Ezzes; Jet Vonk; Brittany Morin; Rian Bogley; Lisa Wauters; Zachary Miller; Maria Gorno-Tempini; Gopala Anumanchipalli
>
> **备注:** 2025 Interspeech
>
> **摘要:** Phonetic error detection, a core subtask of automatic pronunciation assessment, identifies pronunciation deviations at the phoneme level. Speech variability from accents and dysfluencies challenges accurate phoneme recognition, with current models failing to capture these discrepancies effectively. We propose a verbatim phoneme recognition framework using multi-task training with novel phoneme similarity modeling that transcribes what speakers actually say rather than what they're supposed to say. We develop and open-source \textit{VCTK-accent}, a simulated dataset containing phonetic errors, and propose two novel metrics for assessing pronunciation differences. Our work establishes a new benchmark for phonetic error detection.
>
---
#### [new 014] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出了一种用于零样本在线语音转换的模型Conan，旨在实现实时语音转换中内容保留、音色和风格匹配。论文属于语音处理任务，解决现有模型在实时性、语义保真和风格适应上的不足，通过设计流式内容提取、自适应风格编码和因果声码器组件提升效果。**

- **链接: [http://arxiv.org/pdf/2507.14534v1](http://arxiv.org/pdf/2507.14534v1)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [new 015] Adapting Whisper for Lightweight and Efficient Automatic Speech Recognition of Children for On-device Edge Applications
- **分类: eess.AS; cs.HC; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决儿童语音识别中隐私和设备资源受限的问题。通过优化Whisper模型，实现轻量化并在树莓派上高效运行。实验表明，压缩后的模型在保持较低词错误率的同时，显著减少了计算需求。**

- **链接: [http://arxiv.org/pdf/2507.14451v1](http://arxiv.org/pdf/2507.14451v1)**

> **作者:** Satwik Dutta; Shruthigna Chandupatla; John Hansen
>
> **备注:** 5 pages, 5 figures, accepted for presentation at the 2025 Workshop on Child Computer Interaction (WOCCI 2025), a Satellite Workshop of the 2025 Interspeech Conference
>
> **摘要:** Reliability on cloud providers for ASR inference to support child-centered voice-based applications is becoming challenging due to regulatory and privacy challenges. Motivated by a privacy-preserving design, this study aims to develop a lightweight & efficient Whisper ASR system capable of running on a Raspberry Pi. Upon evaluation of the MyST corpus and by examining various filtering strategies to fine-tune the `tiny.en' model, a Word Error Rate (WER) of 15.9% was achieved (11.8% filtered). A low-rank compression reduces the encoder size by 0.51M with 1.26x faster inference in GPU, with 11% relative WER increase. During inference on Pi, the compressed version required ~2 GFLOPS fewer computations. The RTF for both the models ranged between [0.23-0.41] for various input audio durations. Analyzing the RAM usage and CPU temperature showed that the PI was capable of handling both the tiny models, however it was noticed that small models initiated additional overhead/thermal throttling.
>
---
#### [new 016] Developing an AI-Guided Assistant Device for the Deaf and Hearing Impaired
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于智能辅助任务，旨在帮助听障人士实时定位和识别声源。论文提出了包含声音方向检测（JerryNet）、音频分类（基于CLAP模型）和多模态融合定位（结合视觉与音频）的系统。通过定制硬件与深度学习模型，实现高精度方向识别与声音分类，提升辅助设备性能。**

- **链接: [http://arxiv.org/pdf/2507.14215v1](http://arxiv.org/pdf/2507.14215v1)**

> **作者:** Jiayu; Liu
>
> **摘要:** This study aims to develop a deep learning system for an accessibility device for the deaf or hearing impaired. The device will accurately localize and identify sound sources in real time. This study will fill an important gap in current research by leveraging machine learning techniques to target the underprivileged community. The system includes three main components. 1. JerryNet: A custom designed CNN architecture that determines the direction of arrival (DoA) for nine possible directions. 2. Audio Classification: This model is based on fine-tuning the Contrastive Language-Audio Pretraining (CLAP) model to identify the exact sound classes only based on audio. 3. Multimodal integration model: This is an accurate sound localization model that combines audio, visual, and text data to locate the exact sound sources in the images. The part consists of two modules, one object detection using Yolov9 to generate all the bounding boxes of the objects, and an audio visual localization model to identify the optimal bounding box using complete Intersection over Union (CIoU). The hardware consists of a four-microphone rectangular formation and a camera mounted on glasses with a wristband for displaying necessary information like direction. On a custom collected data set, JerryNet achieved a precision of 91. 1% for the sound direction, outperforming all the baseline models. The CLAP model achieved 98.5% and 95% accuracy on custom and AudioSet datasets, respectively. The audio-visual localization model within component 3 yielded a cIoU of 0.892 and an AUC of 0.658, surpassing other similar models. There are many future potentials to this study, paving the way to creating a new generation of accessibility devices.
>
---
#### [new 017] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于3D舞蹈生成任务，旨在解决音乐对齐与全身动作协调难题。作者提出了SoulDance数据集与SoulNet框架，通过分层建模实现高质量、音乐同步的全身舞蹈生成。**

- **链接: [http://arxiv.org/pdf/2507.14915v1](http://arxiv.org/pdf/2507.14915v1)**

> **作者:** Xiaojie Li; Ronghui Li; Shukai Fang; Shuzhao Xie; Xiaoyang Guo; Jiaqing Zhou; Junkun Peng; Zhi Wang
>
> **摘要:** Well-coordinated, music-aligned holistic dance enhances emotional expressiveness and audience engagement. However, generating such dances remains challenging due to the scarcity of holistic 3D dance datasets, the difficulty of achieving cross-modal alignment between music and dance, and the complexity of modeling interdependent motion across the body, hands, and face. To address these challenges, we introduce SoulDance, a high-precision music-dance paired dataset captured via professional motion capture systems, featuring meticulously annotated holistic dance movements. Building on this dataset, we propose SoulNet, a framework designed to generate music-aligned, kinematically coordinated holistic dance sequences. SoulNet consists of three principal components: (1) Hierarchical Residual Vector Quantization, which models complex, fine-grained motion dependencies across the body, hands, and face; (2) Music-Aligned Generative Model, which composes these hierarchical motion units into expressive and coordinated holistic dance; (3) Music-Motion Retrieval Module, a pre-trained cross-modal model that functions as a music-dance alignment prior, ensuring temporal synchronization and semantic coherence between generated dance and input music throughout the generation process. Extensive experiments demonstrate that SoulNet significantly surpasses existing approaches in generating high-quality, music-coordinated, and well-aligned holistic 3D dance sequences.
>
---
#### [new 018] DWTGS: Rethinking Frequency Regularization for Sparse-view 3D Gaussian Splatting
- **分类: cs.CV; eess.IV; eess.SP**

- **简介: 本文属于3D重建任务，旨在解决稀疏视角下高斯点绘（3DGS）因高频过拟合导致的新视角生成质量差的问题。作者提出DWTGS方法，通过小波空间中的低频监督与高频稀疏约束进行正则化，提升泛化能力并减少高频伪影。**

- **链接: [http://arxiv.org/pdf/2507.15690v1](http://arxiv.org/pdf/2507.15690v1)**

> **作者:** Hung Nguyen; Runfa Li; An Le; Truong Nguyen
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Sparse-view 3D Gaussian Splatting (3DGS) presents significant challenges in reconstructing high-quality novel views, as it often overfits to the widely-varying high-frequency (HF) details of the sparse training views. While frequency regularization can be a promising approach, its typical reliance on Fourier transforms causes difficult parameter tuning and biases towards detrimental HF learning. We propose DWTGS, a framework that rethinks frequency regularization by leveraging wavelet-space losses that provide additional spatial supervision. Specifically, we supervise only the low-frequency (LF) LL subbands at multiple DWT levels, while enforcing sparsity on the HF HH subband in a self-supervised manner. Experiments across benchmarks show that DWTGS consistently outperforms Fourier-based counterparts, as this LF-centric strategy improves generalization and reduces HF hallucinations.
>
---
#### [new 019] Parameter-Efficient Fine-Tuning of Foundation Models for CLP Speech Classification
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音分类任务，旨在解决基于发音特征的唇腭裂（CLP）检测与严重程度分类问题。作者通过参数高效微调（PEFT）方法优化基础模型，结合自监督和弱监督模型提取特征，并在英语和印度语数据集上验证性能，取得了优于传统方法的结果。**

- **链接: [http://arxiv.org/pdf/2507.14898v1](http://arxiv.org/pdf/2507.14898v1)**

> **作者:** Susmita Bhattacharjee; Jagabandhu Mishra; H. S. Shekhawat; S. R. Mahadeva Prasanna
>
> **备注:** 6 pages, 5 figures, conference
>
> **摘要:** We propose the use of parameter-efficient fine-tuning (PEFT) of foundation models for cleft lip and palate (CLP) detection and severity classification. In CLP, nasalization increases with severity due to the abnormal passage between the oral and nasal tracts; this causes oral stops to be replaced by glottal stops and alters formant trajectories and vowel space. Since foundation models are trained for grapheme prediction or long-term quantized representation prediction, they may better discriminate CLP severity when fine-tuned on domain-specific data. We conduct experiments on two datasets: English (NMCPC) and Kannada (AIISH). We perform a comparative analysis using embeddings from self-supervised models Wav2Vec2 and WavLM, and the weakly supervised Whisper, each paired with SVM classifiers, and compare them with traditional handcrafted features eGeMAPS and ComParE. Finally, we fine-tune the best-performing Whisper model using PEFT techniques: Low-Rank Adapter (LoRA) and Decomposed Low-Rank Adapter (DoRA). Our results demonstrate that the proposed approach achieves relative improvements of 26.4% and 63.4% in macro-average F1 score over the best foundation model and handcrafted feature baselines on the NMCPC dataset, and improvements of 6.1% and 52.9% on the AIISH dataset, respectively.
>
---
#### [new 020] Fiduciary AI for the Future of Brain-Technology Interactions
- **分类: cs.CY; cs.AI; cs.HC; cs.LG; eess.SP; K.4.0; I.2.0; J.4**

- **简介: 该论文探讨脑机接口（BCI）与基础模型结合带来的风险，提出将受托责任（忠诚、关怀、保密）嵌入系统设计。任务是解决神经信号被滥用、用户控制权不足的问题，通过技术架构与治理机制保障用户权益，确保AI在脑技术交互中忠实于用户利益。**

- **链接: [http://arxiv.org/pdf/2507.14339v1](http://arxiv.org/pdf/2507.14339v1)**

> **作者:** Abhishek Bhattacharjee; Jack Pilkington; Nita Farahany
>
> **备注:** 32 pages
>
> **摘要:** Brain foundation models represent a new frontier in AI: instead of processing text or images, these models interpret real-time neural signals from EEG, fMRI, and other neurotechnologies. When integrated with brain-computer interfaces (BCIs), they may enable transformative applications-from thought controlled devices to neuroprosthetics-by interpreting and acting on brain activity in milliseconds. However, these same systems pose unprecedented risks, including the exploitation of subconscious neural signals and the erosion of cognitive liberty. Users cannot easily observe or control how their brain signals are interpreted, creating power asymmetries that are vulnerable to manipulation. This paper proposes embedding fiduciary duties-loyalty, care, and confidentiality-directly into BCI-integrated brain foundation models through technical design. Drawing on legal traditions and recent advancements in AI alignment techniques, we outline implementable architectural and governance mechanisms to ensure these systems act in users' best interests. Placing brain foundation models on a fiduciary footing is essential to realizing their potential without compromising self-determination.
>
---
#### [new 021] An Investigation of Test-time Adaptation for Audio Classification under Background Noise
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音频分类任务，旨在解决背景噪声引起的域移问题。作者采用测试时自适应（TTA）方法，如TTT、TENT和改进的CoNMix，在AudioMNIST和SpeechCommands数据集上评估其性能。实验表明，改进的CoNMix在噪声环境下表现最佳，且该研究首次将TTA应用于音频分类领域。**

- **链接: [http://arxiv.org/pdf/2507.15523v1](http://arxiv.org/pdf/2507.15523v1)**

> **作者:** Weichuang Shao; Iman Yi Liao; Tomas Henrique Bode Maul; Tissa Chandesa
>
> **摘要:** Domain shift is a prominent problem in Deep Learning, causing a model pre-trained on a source dataset to suffer significant performance degradation on test datasets. This research aims to address the issue of audio classification under domain shift caused by background noise using Test-Time Adaptation (TTA), a technique that adapts a pre-trained model during testing using only unlabelled test data before making predictions. We adopt two common TTA methods, TTT and TENT, and a state-of-the-art method CoNMix, and investigate their respective performance on two popular audio classification datasets, AudioMNIST (AM) and SpeechCommands V1 (SC), against different types of background noise and noise severity levels. The experimental results reveal that our proposed modified version of CoNMix produced the highest classification accuracy under domain shift (5.31% error rate under 10 dB exercise bike background noise and 12.75% error rate under 3 dB running tap background noise for AM) compared to TTT and TENT. The literature search provided no evidence of similar works, thereby motivating the work reported here as the first study to leverage TTA techniques for audio classification under domain shift.
>
---
## 更新

#### [replaced 001] Modeling nonuniform energy decay through the modal decomposition of acoustic radiance transfer (MoD-ART)
- **分类: cs.SD; cs.SY; eess.AS; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.04534v2](http://arxiv.org/pdf/2412.04534v2)**

> **作者:** Matteo Scerbo; Sebastian J. Schlecht; Randall Ali; Lauri Savioja; Enzo De Sena
>
> **摘要:** Modeling late reverberation in real-time interactive applications is a challenging task when multiple sound sources and listeners are present in the same environment. This is especially problematic when the environment is geometrically complex and/or features uneven energy absorption (e.g. coupled volumes), because in such cases the late reverberation is dependent on the sound sources' and listeners' positions, and therefore must be adapted to their movements in real time. We present a novel approach to the task, named modal decomposition of acoustic radiance transfer (MoD-ART), which can handle highly complex scenarios with efficiency. The approach is based on the geometrical acoustics method of acoustic radiance transfer, from which we extract a set of energy decay modes and their positional relationships with sources and listeners. In this paper, we describe the physical and mathematical significance of MoD-ART, highlighting its advantages and applicability to different scenarios. Through an analysis of the method's computational complexity, we show that it compares very favorably with ray-tracing. We also present simulation results showing that MoD-ART can capture multiple decay slopes and flutter echoes.
>
---
#### [replaced 002] Knowing When to Quit: Probabilistic Early Exits for Speech Separation
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09768v2](http://arxiv.org/pdf/2507.09768v2)**

> **作者:** Kenny Falkær Olsen; Mads Østergaard; Karl Ulbæk; Søren Føns Nielsen; Rasmus Malik Høegh Lindrup; Bjørn Sand Jensen; Morten Mørup
>
> **摘要:** In recent years, deep learning-based single-channel speech separation has improved considerably, in large part driven by increasingly compute- and parameter-efficient neural network architectures. Most such architectures are, however, designed with a fixed compute and parameter budget, and consequently cannot scale to varying compute demands or resources, which limits their use in embedded and heterogeneous devices such as mobile phones and hearables. To enable such use-cases we design a neural network architecture for speech separation capable of early-exit, and we propose an uncertainty-aware probabilistic framework to jointly model the clean speech signal and error variance which we use to derive probabilistic early-exit conditions in terms of desired signal-to-noise ratios. We evaluate our methods on both speech separation and enhancement tasks, and we show that a single early-exit model can be competitive with state-of-the-art models trained at many compute and parameter budgets. Our framework enables fine-grained dynamic compute-scaling of speech separation networks while achieving state-of-the-art performance and interpretable exit conditions.
>
---
#### [replaced 003] The Perception of Phase Intercept Distortion and its Application in Data Augmentation
- **分类: eess.SP; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.14571v2](http://arxiv.org/pdf/2506.14571v2)**

> **作者:** Venkatakrishnan Vaidyanathapuram Krishnan; Nathaniel Condit-Schultz
>
> **备注:** Accepted to the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025. Camera-ready version
>
> **摘要:** Phase distortion refers to the alteration of the phase relationships between frequencies in a signal, which can be perceptible. In this paper, we discuss a special case of phase distortion known as phase-intercept distortion, which is created by a frequency-independent phase shift. We hypothesize that, though this form of distortion changes a signal's waveform significantly, the distortion is imperceptible. Human-subject experiment results are reported which are consistent with this hypothesis. Furthermore, we discuss how the imperceptibility of phase-intercept distortion can be useful for machine learning, specifically for data augmentation. We conducted multiple experiments using phase-intercept distortion as a novel approach to data augmentation, and obtained improved results for audio machine learning tasks.
>
---
#### [replaced 004] Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09116v3](http://arxiv.org/pdf/2507.09116v3)**

> **作者:** Bingshen Mu; Kun Wei; Pengcheng Guo; Lei Xie
>
> **备注:** IEEE Transactions on Audio, Speech and Language Processing
>
> **摘要:** Despite improvements in automatic speech recognition, performance drops with accented speech. Generative error correction (GER) leverages the linguistic knowledge of large language models (LLMs), outperforming typical language model methods. However, it lacks specificity in accented speech scenarios. Accents represent deviations from standard pronunciation, making multi-granularity pronunciation and semantic information essential for accented speech recognition. Moreover, accents exhibit considerable diversity, with each accent possessing distinct characteristics. In this study, we leverage GER to improve transcription accuracy by addressing the two primary features. We propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level pronunciation information. These methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through low-rank adaptation (LoRA) fine-tuning. We employ a three-stage strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge mono-accent LoRA experts within a single multi-modal GER to overcome accent diversity challenges. Furthermore, multi-granularity GER leverages N-best word-level and phoneme-level hypotheses from the HDMoLE model to predict final transcriptions. Experiments on a multi-accent English dataset show that our methods reduce word error rate by 67.35% compared to the baseline vanilla Whisper-large-v3 model.
>
---
#### [replaced 005] End-to-end Joint Punctuated and Normalized ASR with a Limited Amount of Punctuated Training Data
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2311.17741v3](http://arxiv.org/pdf/2311.17741v3)**

> **作者:** Can Cui; Imran Ahamad Sheikh; Mostafa Sadeghi; Emmanuel Vincent
>
> **摘要:** Joint punctuated and normalized automatic speech recognition (ASR) aims at outputing transcripts with and without punctuation and casing. This task remains challenging due to the lack of paired speech and punctuated text data in most ASR corpora. We propose two approaches to train an end-to-end joint punctuated and normalized ASR system using limited punctuated data. The first approach uses a language model to convert normalized training transcripts into punctuated transcripts. This achieves a better performance on out-of-domain test data, with up to 17% relative Punctuation-Case-aware Word Error Rate (PC-WER) reduction. The second approach uses a single decoder conditioned on the type of output. This yields a 42% relative PC-WER reduction compared to Whisper-base and a 4% relative (normalized) WER reduction compared to the normalized output of a punctuated-only model. Additionally, our proposed model demonstrates the feasibility of a joint ASR system using as little as 5% punctuated training data with a moderate (2.42% absolute) PC-WER increase.
>
---
#### [replaced 006] SC-TSE: Speaker Consistency-Aware Target Speaker Extraction
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.09510v2](http://arxiv.org/pdf/2507.09510v2)**

> **作者:** Shu Wu; Anbin Qi; Yanzhang Xie; Xiang Xie
>
> **备注:** Due to concerns regarding data and citations that may compromise academic rigor, the paper has been retracted
>
> **摘要:** Target Speaker Extraction (TSE) uses a reference cue to extract the target speech from a mixture. In TSE systems relying on audio cues, the speaker embedding from the enrolled speech is crucial to performance. However, these embeddings may suffer from speaker identity confusion. Unlike previous studies that focus on improving speaker embedding extraction, we improve TSE performance from the perspective of speaker consistency. In this paper, we propose a speaker consistency-aware target speaker extraction method that incorporates a centroid-based speaker consistency loss. This approach enhances TSE performance by ensuring speaker consistency between the enrolled and extracted speech. In addition, we integrate conditional loss suppression into the training process. The experimental results validate the effectiveness of our proposed methods in advancing the TSE performance. A speech demo is available online:https://sc-tse.netlify.app/
>
---
#### [replaced 007] RingFormer: A Neural Vocoder with Ring Attention and Convolution-Augmented Transformer
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.01182v2](http://arxiv.org/pdf/2501.01182v2)**

> **作者:** Seongho Hong; Yong-Hoon Choi
>
> **备注:** Accepted for publication in IEEE Transactions on Human-Machine Systems (THMS)
>
> **摘要:** While transformers demonstrate outstanding performance across various audio tasks, their application to neural vocoders remains challenging. Neural vocoders require the generation of long audio signals at the sample level, which demands high temporal resolution. This results in significant computational costs for attention map generation and limits their ability to efficiently process both global and local information. Additionally, the sequential nature of sample generation in neural vocoders poses difficulties for real-time processing, making the direct adoption of transformers impractical. To address these challenges, we propose RingFormer, a neural vocoder that incorporates the ring attention mechanism into a lightweight transformer variant, the convolution-augmented transformer (Conformer). Ring attention effectively captures local details while integrating global information, making it well-suited for processing long sequences and enabling real-time audio generation. RingFormer is trained using adversarial training with two discriminators. The proposed model is applied to the decoder of the text-to-speech model VITS and compared with state-of-the-art vocoders such as HiFi-GAN, iSTFT-Net, and BigVGAN under identical conditions using various objective and subjective metrics. Experimental results show that RingFormer achieves comparable or superior performance to existing models, particularly excelling in real-time audio generation. Our code and audio samples are available on GitHub.
>
---
#### [replaced 008] Supporting SENCOTEN Language Documentation Efforts with Automatic Speech Recognition
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.10827v2](http://arxiv.org/pdf/2507.10827v2)**

> **作者:** Mengzhe Geng; Patrick Littell; Aidan Pine; PENÁĆ; Marc Tessier; Roland Kuhn
>
> **备注:** Accepted by ComputEL-8
>
> **摘要:** The SENCOTEN language, spoken on the Saanich peninsula of southern Vancouver Island, is in the midst of vigorous language revitalization efforts to turn the tide of language loss as a result of colonial language policies. To support these on-the-ground efforts, the community is turning to digital technology. Automatic Speech Recognition (ASR) technology holds great promise for accelerating language documentation and the creation of educational resources. However, developing ASR systems for SENCOTEN is challenging due to limited data and significant vocabulary variation from its polysynthetic structure and stress-driven metathesis. To address these challenges, we propose an ASR-driven documentation pipeline that leverages augmented speech data from a text-to-speech (TTS) system and cross-lingual transfer learning with Speech Foundation Models (SFMs). An n-gram language model is also incorporated via shallow fusion or n-best restoring to maximize the use of available data. Experiments on the SENCOTEN dataset show a word error rate (WER) of 19.34% and a character error rate (CER) of 5.09% on the test set with a 57.02% out-of-vocabulary (OOV) rate. After filtering minor cedilla-related errors, WER improves to 14.32% (26.48% on unseen words) and CER to 3.45%, demonstrating the potential of our ASR-driven pipeline to support SENCOTEN language documentation.
>
---
#### [replaced 009] Unifying Listener Scoring Scales: Comparison Learning Framework for Speech Quality Assessment and Continuous Speech Emotion Recognition
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.13626v2](http://arxiv.org/pdf/2507.13626v2)**

> **作者:** Cheng-Hung Hu; Yusuke Yasuda; Akifumi Yoshimoto; Tomoki Toda
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Speech Quality Assessment (SQA) and Continuous Speech Emotion Recognition (CSER) are two key tasks in speech technology, both relying on listener ratings. However, these ratings are inherently biased due to individual listener factors. Previous approaches have introduced a mean listener scoring scale and modeled all listener scoring scales in the training set. However, the mean listener approach is prone to distortion from averaging ordinal data, leading to potential biases. Moreover, learning multiple listener scoring scales while inferring based only on the mean listener scale limits effectiveness. In contrast, our method focuses on modeling a unified listener scoring scale, using comparison scores to correctly capture the scoring relationships between utterances. Experimental results show that our method effectively improves prediction performance in both SQA and CSER tasks, proving its effectiveness and robustness.
>
---
#### [replaced 010] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.06656v3](http://arxiv.org/pdf/2409.06656v3)**

> **作者:** Taejin Park; Ivan Medennikov; Kunal Dhawan; Weiqing Wang; He Huang; Nithin Rao Koluguri; Krishna C. Puvvada; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Published at ICML 2025
>
> **摘要:** Sortformer is an encoder-based speaker diarization model designed for supervising speaker tagging in speech-to-text models. Instead of relying solely on permutation invariant loss (PIL), Sortformer introduces Sort Loss to resolve the permutation problem, either independently or in tandem with PIL. In addition, we propose a streamlined multi-speaker speech-to-text architecture that leverages Sortformer for speaker supervision, embedding speaker labels into the encoder using sinusoidal kernel functions. This design addresses the speaker permutation problem through sorted objectives, effectively bridging timestamps and tokens to supervise speaker labels in the output transcriptions. Experiments demonstrate that Sort Loss can boost speaker diarization performance, and incorporating the speaker supervision from Sortformer improves multi-speaker transcription accuracy. We anticipate that the proposed Sortformer and multi-speaker architecture will enable the seamless integration of speaker tagging capabilities into foundational speech-to-text systems and multimodal large language models (LLMs), offering an easily adoptable and user-friendly mechanism to enhance their versatility and performance in speaker-aware tasks. The code and trained models are made publicly available through the NVIDIA NeMo Framework.
>
---
#### [replaced 011] Towards the Next Frontier in Speech Representation Learning Using Disentanglement
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.02543v2](http://arxiv.org/pdf/2407.02543v2)**

> **作者:** Varun Krishna; Sriram Ganapathy
>
> **备注:** There were some bugs in the Code that was used to produce the results in the paper. The results reported in the paper are not valid
>
> **摘要:** The popular frameworks for self-supervised learning of speech representations have largely focused on frame-level masked prediction of speech regions. While this has shown promising downstream task performance for speech recognition and related tasks, this has largely ignored factors of speech that are encoded at coarser level, like characteristics of the speaker or channel that remain consistent through-out a speech utterance. In this work, we propose a framework for Learning Disentangled Self Supervised (termed as Learn2Diss) representations of speech, which consists of frame-level and an utterance-level encoder modules. The two encoders are initially learned independently, where the frame-level model is largely inspired by existing self supervision techniques, thereby learning pseudo-phonemic representations, while the utterance-level encoder is inspired by constrastive learning of pooled embeddings, thereby learning pseudo-speaker representations. The joint learning of these two modules consists of disentangling the two encoders using a mutual information based criterion. With several downstream evaluation experiments, we show that the proposed Learn2Diss achieves state-of-the-art results on a variety of tasks, with the frame-level encoder representations improving semantic tasks, while the utterance-level representations improve non-semantic tasks.
>
---
#### [replaced 012] ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting
- **分类: eess.AS; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.20630v3](http://arxiv.org/pdf/2504.20630v3)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating Drama-MOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos are available at https://aaronz345.github.io/ISDramaDemo. We provide the dataset and the evaluation code at https://huggingface.co/datasets/AaronZ345/MRSDrama and https://github.com/AaronZ345/ISDrama.
>
---
#### [replaced 013] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.06273v2](http://arxiv.org/pdf/2503.06273v2)**

> **作者:** Jeong Hun Yeo; Minsu Kim; Chae Won Kim; Stavros Petridis; Yong Man Ro
>
> **备注:** Accepted at ICCV 2025. Code available at: https://github.com/JeongHun0716/zero-avsr
>
> **摘要:** We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
>
---
