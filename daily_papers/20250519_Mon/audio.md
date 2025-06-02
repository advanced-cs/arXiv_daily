# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Improving Inference-Time Optimisation for Vocal Effects Style Transfer with a Gaussian Prior
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频风格迁移任务，解决现有方法因忽略参数分布导致的偏差问题。通过引入基于DiffVox数据集的高斯先验约束，将推理时优化转换为最大后验估计，在MedleyDB数据集上显著降低参数误差33%，主客观评估均优于基线方法，提升了声音效果迁移的真实性和效果。**

- **链接: [http://arxiv.org/pdf/2505.11315v1](http://arxiv.org/pdf/2505.11315v1)**

> **作者:** Chin-Yun Yu; Marco A. Martínez-Ramírez; Junghyun Koo; Wei-Hsiang Liao; Yuki Mitsufuji; György Fazekas
>
> **备注:** Submitted to WASPAA 2025
>
> **摘要:** Style Transfer with Inference-Time Optimisation (ST-ITO) is a recent approach for transferring the applied effects of a reference audio to a raw audio track. It optimises the effect parameters to minimise the distance between the style embeddings of the processed audio and the reference. However, this method treats all possible configurations equally and relies solely on the embedding space, which can lead to unrealistic or biased results. We address this pitfall by introducing a Gaussian prior derived from a vocal preset dataset, DiffVox, over the parameter space. The resulting optimisation is equivalent to maximum-a-posteriori estimation. Evaluations on vocal effects transfer on the MedleyDB dataset show significant improvements across metrics compared to baselines, including a blind audio effects estimator, nearest-neighbour approaches, and uncalibrated ST-ITO. The proposed calibration reduces parameter mean squared error by up to 33% and matches the reference style better. Subjective evaluations with 16 participants confirm our method's superiority, especially in limited data regimes. This work demonstrates how incorporating prior knowledge in inference time enhances audio effects transfer, paving the way for more effective and realistic audio processing systems.
>
---
#### [new 002] Multi-Stage Speaker Diarization for Noisy Classrooms
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究说话人日志任务，解决嘈杂教室中因背景噪音、重叠语音和儿童声音导致的识别难题。提出多阶段模型，结合降噪、混合语音检测（融合ASR时间戳）优化Nvidia NeMo流程，实验显示降噪和混合方法显著降低了错误率，教师-学生场景DER达17%。**

- **链接: [http://arxiv.org/pdf/2505.10879v1](http://arxiv.org/pdf/2505.10879v1)**

> **作者:** Ali Sartaz Khan; Tolulope Ogunremi; Ahmed Attia; Dorottya Demszky
>
> **摘要:** Speaker diarization, the process of identifying "who spoke when" in audio recordings, is essential for understanding classroom dynamics. However, classroom settings present distinct challenges, including poor recording quality, high levels of background noise, overlapping speech, and the difficulty of accurately capturing children's voices. This study investigates the effectiveness of multi-stage diarization models using Nvidia's NeMo diarization pipeline. We assess the impact of denoising on diarization accuracy and compare various voice activity detection (VAD) models, including self-supervised transformer-based frame-wise VAD models. We also explore a hybrid VAD approach that integrates Automatic Speech Recognition (ASR) word-level timestamps with frame-level VAD predictions. We conduct experiments using two datasets from English speaking classrooms to separate teacher vs. student speech and to separate all speakers. Our results show that denoising significantly improves the Diarization Error Rate (DER) by reducing the rate of missed speech. Additionally, training on both denoised and noisy datasets leads to substantial performance gains in noisy conditions. The hybrid VAD model leads to further improvements in speech detection, achieving a DER as low as 17% in teacher-student experiments and 45% in all-speaker experiments. However, we also identified trade-offs between voice activity detection and speaker confusion. Overall, our study highlights the effectiveness of multi-stage diarization models and integrating ASR-based information for enhancing speaker diarization in noisy classroom environments.
>
---
#### [new 003] $\mathcal{A}LLM4ADD$: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有音频大语言模型（ALLM）在零样本下检测伪造音频效果差的问题。研究提出ALLM4ADD框架，将检测任务重构为音频问答问题，通过监督微调使ALLM识别音频真实性，在数据稀缺场景中实现高效检测，为开发更优检测系统提供新思路。**

- **链接: [http://arxiv.org/pdf/2505.11079v1](http://arxiv.org/pdf/2505.11079v1)**

> **作者:** Hao Gu; Jiangyan Yi; Chenglong Wang; Jianhua Tao; Zheng Lian; Jiayi He; Yong Ren; Yujie Chen; Zhengqi Wen
>
> **摘要:** Audio deepfake detection (ADD) has grown increasingly important due to the rise of high-fidelity audio generative models and their potential for misuse. Given that audio large language models (ALLMs) have made significant progress in various audio processing tasks, a heuristic question arises: Can ALLMs be leveraged to solve ADD?. In this paper, we first conduct a comprehensive zero-shot evaluation of ALLMs on ADD, revealing their ineffectiveness in detecting fake audio. To enhance their performance, we propose $\mathcal{A}LLM4ADD$, an ALLM-driven framework for ADD. Specifically, we reformulate ADD task as an audio question answering problem, prompting the model with the question: "Is this audio fake or real?". We then perform supervised fine-tuning to enable the ALLM to assess the authenticity of query audio. Extensive experiments are conducted to demonstrate that our ALLM-based method can achieve superior performance in fake audio detection, particularly in data-scarce scenarios. As a pioneering study, we anticipate that this work will inspire the research community to leverage ALLMs to develop more effective ADD systems.
>
---
#### [new 004] Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文属于文本到语音（TTS）评估任务，旨在解决中文TTS系统评测中传统方法（如MOS）的主观性、维度单一等问题。提出多维度中文数据集ATT-Corpus及基于图灵测试的评测协议，通过人类判听简化评估，并开发自动评测工具Auto-ATT，验证其与人工评价的一致性，提升评测效率和客观性。**

- **链接: [http://arxiv.org/pdf/2505.11200v1](http://arxiv.org/pdf/2505.11200v1)**

> **作者:** Xihuai Wang; Ziyi Zhao; Siyu Ren; Shao Zhang; Song Li; Xiaoyu Li; Ziwen Wang; Lin Qiu; Guanglu Wan; Xuezhi Cao; Xunliang Cai; Weinan Zhang
>
> **备注:** Under Review
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved text-to-speech (TTS) systems, enhancing control over speech style, naturalness, and emotional expression, which brings TTS Systems closer to human-level performance. Although the Mean Opinion Score (MOS) remains the standard for TTS System evaluation, it suffers from subjectivity, environmental inconsistencies, and limited interpretability. Existing evaluation datasets also lack a multi-dimensional design, often neglecting factors such as speaking styles, context diversity, and trap utterances, which is particularly evident in Chinese TTS evaluation. To address these challenges, we introduce the Audio Turing Test (ATT), a multi-dimensional Chinese corpus dataset ATT-Corpus paired with a simple, Turing-Test-inspired evaluation protocol. Instead of relying on complex MOS scales or direct model comparisons, ATT asks evaluators to judge whether a voice sounds human. This simplification reduces rating bias and improves evaluation robustness. To further support rapid model development, we also finetune Qwen2-Audio-Instruct with human judgment data as Auto-ATT for automatic evaluation. Experimental results show that ATT effectively differentiates models across specific capability dimensions using its multi-dimensional design. Auto-ATT also demonstrates strong alignment with human evaluations, confirming its value as a fast and reliable assessment tool. The white-box ATT-Corpus and Auto-ATT can be found in ATT Hugging Face Collection (https://huggingface.co/collections/meituan/audio-turing-test-682446320368164faeaf38a4).
>
---
#### [new 005] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI models in Sound Localization
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **简介: 该论文研究多模态AI的声源定位任务，解决视听冲突下模型模态偏差问题。通过对比实验发现AI过度依赖视觉，性能低于人类；提出利用3D仿真数据微调模型，使其在冲突场景中表现提升并呈现类似人类的左右定位偏好，揭示了感官输入质量与架构对多模态表征的影响。**

- **链接: [http://arxiv.org/pdf/2505.11217v1](http://arxiv.org/pdf/2505.11217v1)**

> **作者:** Yanhao Jia; Ji Xie; S Jivaganesh; Hao Li; Xu Wu; Mengmi Zhang
>
> **备注:** 16 pages, 14 figures
>
> **摘要:** Imagine hearing a dog bark and turning toward the sound only to see a parked car, while the real, silent dog sits elsewhere. Such sensory conflicts test perception, yet humans reliably resolve them by prioritizing sound over misleading visuals. Despite advances in multimodal AI integrating vision and audio, little is known about how these systems handle cross-modal conflicts or whether they favor one modality. In this study, we systematically examine modality bias and conflict resolution in AI sound localization. We assess leading multimodal models and benchmark them against human performance in psychophysics experiments across six audiovisual conditions, including congruent, conflicting, and absent cues. Humans consistently outperform AI, demonstrating superior resilience to conflicting or missing visuals by relying on auditory information. In contrast, AI models often default to visual input, degrading performance to near chance levels. To address this, we finetune a state-of-the-art model using a stereo audio-image dataset generated via 3D simulations. Even with limited training data, the refined model surpasses existing benchmarks. Notably, it also mirrors human-like horizontal localization bias favoring left-right precision-likely due to the stereo audio structure reflecting human ear placement. These findings underscore how sensory input quality and system architecture shape multimodal representation accuracy.
>
---
#### [new 006] Machine Learning Approaches to Vocal Register Classification in Contemporary Male Pop Music
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频分类任务，旨在解决男性流行音乐中歌手声乐音区（如胸声、头声）的自动识别难题。通过分析梅尔频谱图纹理特征，提出支持向量机（SVM）和卷积神经网络（CNN）两种分类方法，并开发了AVRA软件工具。研究验证了模型在音区分类中的有效性，为声乐分析技术提供支持。**

- **链接: [http://arxiv.org/pdf/2505.11378v1](http://arxiv.org/pdf/2505.11378v1)**

> **作者:** Alexander Kim; Charlotte Botha
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** For singers of all experience levels, one of the most daunting challenges in learning technical repertoire is navigating placement and vocal register in and around the passagio (passage between chest voice and head voice registers). Particularly in pop music, where a single artist may use a variety of timbre's and textures to achieve a desired quality, it can be difficult to identify what vocal register within the vocal range a singer is using. This paper presents two methods for classifying vocal registers in an audio signal of male pop music through the analysis of textural features of mel-spectrogram images. Additionally, we will discuss the practical integration of these models for vocal analysis tools, and introduce a concurrently developed software called AVRA which stands for Automatic Vocal Register Analysis. Our proposed methods achieved consistent classification of vocal register through both Support Vector Machine (SVM) and Convolutional Neural Network (CNN) models, which supports the promise of more robust classification possibilities across more voice types and genres of singing.
>
---
#### [new 007] BanglaFake: Constructing and Evaluating a Specialized Bengali Deepfake Audio Dataset
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于深度伪造音频检测任务，旨在解决低资源语言（孟加拉语）因数据集不足和声学特征差异细微导致的检测难题。研究者构建了包含12,260真实和13,260伪造音频的BanglaFake数据集，利用先进TTS模型生成高质量合成语音，并通过MOS评分、t-SNE可视化等方法评估数据特性，为孟加拉语深度伪造检测研究提供关键资源。**

- **链接: [http://arxiv.org/pdf/2505.10885v1](http://arxiv.org/pdf/2505.10885v1)**

> **作者:** Istiaq Ahmed Fahad; Kamruzzaman Asif; Sifat Sikder
>
> **备注:** 5 page
>
> **摘要:** Deepfake audio detection is challenging for low-resource languages like Bengali due to limited datasets and subtle acoustic features. To address this, we introduce BangalFake, a Bengali Deepfake Audio Dataset with 12,260 real and 13,260 deepfake utterances. Synthetic speech is generated using SOTA Text-to-Speech (TTS) models, ensuring high naturalness and quality. We evaluate the dataset through both qualitative and quantitative analyses. Mean Opinion Score (MOS) from 30 native speakers shows Robust-MOS of 3.40 (naturalness) and 4.01 (intelligibility). t-SNE visualization of MFCCs highlights real vs. fake differentiation challenges. This dataset serves as a crucial resource for advancing deepfake detection in Bengali, addressing the limitations of low-resource language research.
>
---
#### [new 008] Classifying Shelf Life Quality of Pineapples by Combining Audio and Visual Features
- **分类: cs.CV; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于多模态分类任务，旨在通过非破坏性方法解决菠萝保质期质量分级问题。研究构建了融合多视角音频（敲击声）和视觉（多角度图像）特征的分类模型，创建了包含500个样本的PQC500数据集，并改进对比学习框架训练跨模态分类器。实验表明音频主导采样策略使模型准确率达84%，优于单模态方法。**

- **链接: [http://arxiv.org/pdf/2505.11020v1](http://arxiv.org/pdf/2505.11020v1)**

> **作者:** Yi-Lu Jiang; Wen-Chang Chang; Ching-Lin Wang; Kung-Liang Hsu; Chih-Yi Chiu
>
> **摘要:** Determining the shelf life quality of pineapples using non-destructive methods is a crucial step to reduce waste and increase income. In this paper, a multimodal and multiview classification model was constructed to classify pineapples into four quality levels based on audio and visual characteristics. For research purposes, we compiled and released the PQC500 dataset consisting of 500 pineapples with two modalities: one was tapping pineapples to record sounds by multiple microphones and the other was taking pictures by multiple cameras at different locations, providing multimodal and multi-view audiovisual features. We modified the contrastive audiovisual masked autoencoder to train the cross-modal-based classification model by abundant combinations of audio and visual pairs. In addition, we proposed to sample a compact size of training data for efficient computation. The experiments were evaluated under various data and model configurations, and the results demonstrated that the proposed cross-modal model trained using audio-major sampling can yield 84% accuracy, outperforming the unimodal models of only audio and only visual by 6% and 18%, respectively.
>
---
#### [new 009] NeoLightning: A Modern Reimagination of Gesture-Based Sound Design
- **分类: cs.HC; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于人机交互与数字音乐技术领域，旨在解决传统手势音乐设备（如Buchla Lightning）技术过时、交互受限的问题。通过整合MediaPipe深度学习手势识别、Max/MSP和Processing多媒体处理，实现了低延迟的精准手势捕捉与沉浸式3D交互，为现代用户重构了基于手势的实时声音设计系统。**

- **链接: [http://arxiv.org/pdf/2505.10686v1](http://arxiv.org/pdf/2505.10686v1)**

> **作者:** Yonghyun Kim; Sangheon Park; Marcus Parker; Donghoon Seu; Alexandria Smith
>
> **备注:** Accepted to the 50th International Computer Music Conference (ICMC), 2025
>
> **摘要:** This paper introduces NeoLightning, a modern reinterpretation of the Buchla Lightning. NeoLightning preserves the innovative spirit of Don Buchla's "Buchla Lightning" (introduced in the 1990s) while making its gesture-based interaction accessible to contemporary users. While the original Buchla Lightning and many other historical instruments were groundbreaking in their time, they are now largely unsupported, limiting user interaction to indirect experiences. To address this, NeoLightning leverages MediaPipe for deep learning-based gesture recognition and employs Max/MSP and Processing for real-time multimedia processing. The redesigned system offers precise, low-latency gesture recognition and immersive 3D interaction. By merging the creative spirit of the original Lightning with modern advancements, NeoLightning redefines gesture-based musical interaction, expanding possibilities for expressive performance and interactive sound design.
>
---
#### [new 010] Anti-aliasing of neural distortion effects via model fine tuning
- **分类: eess.AS; cs.LG; eess.SP**

- **简介: 该论文属于音频信号处理任务，解决神经网络吉他失真模型中高频输入导致的频率混叠问题。作者提出师生微调方法：用冻结的教师模型生成无混叠训练数据，调整学生模型参数以抑制混叠。实验表明该方法在LSTM和TCN中效果优于两倍过采样，但可能影响谐波成分，其中LSTM在抗混叠与音质保真间表现最佳。**

- **链接: [http://arxiv.org/pdf/2505.11375v1](http://arxiv.org/pdf/2505.11375v1)**

> **作者:** Alistair Carson; Alec Wright; Stefan Bilbao
>
> **备注:** Accepted for DAFx25
>
> **摘要:** Neural networks have become ubiquitous with guitar distortion effects modelling in recent years. Despite their ability to yield perceptually convincing models, they are susceptible to frequency aliasing when driven by high frequency and high gain inputs. Nonlinear activation functions create both the desired harmonic distortion and unwanted aliasing distortion as the bandwidth of the signal is expanded beyond the Nyquist frequency. Here, we present a method for reducing aliasing in neural models via a teacher-student fine tuning approach, where the teacher is a pre-trained model with its weights frozen, and the student is a copy of this with learnable parameters. The student is fine-tuned against an aliasing-free dataset generated by passing sinusoids through the original model and removing non-harmonic components from the output spectra. Our results show that this method significantly suppresses aliasing for both long-short-term-memory networks (LSTM) and temporal convolutional networks (TCN). In the majority of our case studies, the reduction in aliasing was greater than that achieved by two times oversampling. One side-effect of the proposed method is that harmonic distortion components are also affected. This adverse effect was found to be model-dependent, with the LSTM models giving the best balance between anti-aliasing and preserving the perceived similarity to an analog reference device.
>
---
#### [new 011] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属多说话人语音识别综述，针对单通道音频中数据稀缺、重叠语音下说话人及内容识别难题，系统梳理端到端架构（SIMO/SISO范式）的技术演进，分析算法改进、长语音处理策略，并通过基准评估对比方法性能，总结挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2505.10975v1](http://arxiv.org/pdf/2505.10975v1)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** 13 pages. Submitted to IEEE/ACM Transaction on Audio Speech and Language Processing (TASLP)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [new 012] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 本文提出CAMEO多语言情感语音数据集，用于语音情感识别任务，解决数据分散、复现困难和缺乏统一基准的问题。通过筛选、标准化处理多语种数据，构建公开平台并提供模型性能评估，建立跨语言/情感的标准化评测基准。**

- **链接: [http://arxiv.org/pdf/2505.11051v1](http://arxiv.org/pdf/2505.11051v1)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Under review at NeurIPS
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [new 013] LegoSLM: Connecting LLM with Speech Encoder using CTC Posteriors
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言处理任务，旨在解决预训练语音编码器与LLM结合性能不佳的问题。提出LegoSLM框架，通过CTC后验矩阵将语音特征转换为LLM词表概率分布，重构伪音频嵌入并与文本嵌入融合，实现ASR与语音翻译性能提升（49% WERR）。支持模块化替换语音编码器，并通过温度控制调节多模态权重。**

- **链接: [http://arxiv.org/pdf/2505.11352v1](http://arxiv.org/pdf/2505.11352v1)**

> **作者:** Rao Ma; Tongzhou Chen; Kartik Audhkhasi; Bhuvana Ramabhadran
>
> **摘要:** Recently, large-scale pre-trained speech encoders and Large Language Models (LLMs) have been released, which show state-of-the-art performance on a range of spoken language processing tasks including Automatic Speech Recognition (ASR). To effectively combine both models for better performance, continuous speech prompts, and ASR error correction have been adopted. However, these methods are prone to suboptimal performance or are inflexible. In this paper, we propose a new paradigm, LegoSLM, that bridges speech encoders and LLMs using the ASR posterior matrices. The speech encoder is trained to generate Connectionist Temporal Classification (CTC) posteriors over the LLM vocabulary, which are used to reconstruct pseudo-audio embeddings by computing a weighted sum of the LLM input embeddings. These embeddings are concatenated with text embeddings in the LLM input space. Using the well-performing USM and Gemma models as an example, we demonstrate that our proposed LegoSLM method yields good performance on both ASR and speech translation tasks. By connecting USM with Gemma models, we can get an average of 49% WERR over the USM-CTC baseline on 8 MLS testsets. The trained model also exhibits modularity in a range of settings -- after fine-tuning the Gemma model weights, the speech encoder can be switched and combined with the LLM in a zero-shot fashion. Additionally, we propose to control the decode-time influence of the USM and LLM using a softmax temperature, which shows effectiveness in domain adaptation.
>
---
#### [new 014] LipDiffuser: Lip-to-Speech Generation with Conditional Diffusion Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究唇语转语音任务，解决无声视频生成自然清晰语音的问题。提出LipDiffuser模型，基于条件扩散架构（MP-ADM），通过MP-FiLM融合视觉特征与说话人嵌入，配合神经声码器重构语音。实验表明其在语音质量、说话人相似度优于基线，并保持ASR竞争力。**

- **链接: [http://arxiv.org/pdf/2505.11391v1](http://arxiv.org/pdf/2505.11391v1)**

> **作者:** Danilo de Oliveira; Julius Richter; Tal Peer; Timo Germann
>
> **摘要:** We present LipDiffuser, a conditional diffusion model for lip-to-speech generation synthesizing natural and intelligible speech directly from silent video recordings. Our approach leverages the magnitude-preserving ablated diffusion model (MP-ADM) architecture as a denoiser model. To effectively condition the model, we incorporate visual features using magnitude-preserving feature-wise linear modulation (MP-FiLM) alongside speaker embeddings. A neural vocoder then reconstructs the speech waveform from the generated mel-spectrograms. Evaluations on LRS3 and TCD-TIMIT demonstrate that LipDiffuser outperforms existing lip-to-speech baselines in perceptual speech quality and speaker similarity, while remaining competitive in downstream automatic speech recognition (ASR). These findings are also supported by a formal listening experiment. Extensive ablation studies and cross-dataset evaluation confirm the effectiveness and generalization capabilities of our approach.
>
---
## 更新

#### [replaced 001] Supervised contrastive learning from weakly-labeled audio segments for musical version matching
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.16936v3](http://arxiv.org/pdf/2502.16936v3)**

> **作者:** Joan Serrà; R. Oguz Araz; Dmitry Bogdanov; Yuki Mitsufuji
>
> **备注:** 17 pages, 6 figures, 8 tables (includes Appendix); accepted at ICML25
>
> **摘要:** Detecting musical versions (different renditions of the same piece) is a challenging task with important applications. Because of the ground truth nature, existing approaches match musical versions at the track level (e.g., whole song). However, most applications require to match them at the segment level (e.g., 20s chunks). In addition, existing approaches resort to classification and triplet losses, disregarding more recent losses that could bring meaningful improvements. In this paper, we propose a method to learn from weakly annotated segments, together with a contrastive loss variant that outperforms well-studied alternatives. The former is based on pairwise segment distance reductions, while the latter modifies an existing loss following decoupling, hyper-parameter, and geometric considerations. With these two elements, we do not only achieve state-of-the-art results in the standard track-level evaluation, but we also obtain a breakthrough performance in a segment-level evaluation. We believe that, due to the generality of the challenges addressed here, the proposed methods may find utility in domains beyond audio or musical version matching.
>
---
#### [replaced 002] On the Role of Speech Data in Reducing Toxicity Detection Bias
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.08135v2](http://arxiv.org/pdf/2411.08135v2)**

> **作者:** Samuel J. Bell; Mariano Coria Meglioli; Megan Richards; Eduardo Sánchez; Christophe Ropers; Skyler Wang; Adina Williams; Levent Sagun; Marta R. Costa-jussà
>
> **备注:** Accepted at NAACL 2025
>
> **摘要:** Text toxicity detection systems exhibit significant biases, producing disproportionate rates of false positives on samples mentioning demographic groups. But what about toxicity detection in speech? To investigate the extent to which text-based biases are mitigated by speech-based systems, we produce a set of high-quality group annotations for the multilingual MuTox dataset, and then leverage these annotations to systematically compare speech- and text-based toxicity classifiers. Our findings indicate that access to speech data during inference supports reduced bias against group mentions, particularly for ambiguous and disagreement-inducing samples. Our results also suggest that improving classifiers, rather than transcription pipelines, is more helpful for reducing group bias. We publicly release our annotations and provide recommendations for future toxicity dataset construction.
>
---
#### [replaced 003] JamendoMaxCaps: A Large Scale Music-caption Dataset with Imputed Metadata
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.07461v2](http://arxiv.org/pdf/2502.07461v2)**

> **作者:** Abhinaba Roy; Renhang Liu; Tongyu Lu; Dorien Herremans
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** We introduce JamendoMaxCaps, a large-scale music-caption dataset featuring over 362,000 freely licensed instrumental tracks from the renowned Jamendo platform. The dataset includes captions generated by a state-of-the-art captioning model, enhanced with imputed metadata. We also introduce a retrieval system that leverages both musical features and metadata to identify similar songs, which are then used to fill in missing metadata using a local large language model (LLLM). This approach allows us to provide a more comprehensive and informative dataset for researchers working on music-language understanding tasks. We validate this approach quantitatively with five different measurements. By making the JamendoMaxCaps dataset publicly available, we provide a high-quality resource to advance research in music-language understanding tasks such as music retrieval, multimodal representation learning, and generative music models.
>
---
#### [replaced 004] SlimSpeech: Lightweight and Efficient Text-to-Speech with Slim Rectified Flow
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07776v2](http://arxiv.org/pdf/2504.07776v2)**

> **作者:** Kaidi Wang; Wenhao Guan; Shenghui Lu; Jianglong Yao; Lin Li; Qingyang Hong
>
> **摘要:** Recently, flow matching based speech synthesis has significantly enhanced the quality of synthesized speech while reducing the number of inference steps. In this paper, we introduce SlimSpeech, a lightweight and efficient speech synthesis system based on rectified flow. We have built upon the existing speech synthesis method utilizing the rectified flow model, modifying its structure to reduce parameters and serve as a teacher model. By refining the reflow operation, we directly derive a smaller model with a more straight sampling trajectory from the larger model, while utilizing distillation techniques to further enhance the model performance. Experimental results demonstrate that our proposed method, with significantly reduced model parameters, achieves comparable performance to larger models through one-step sampling.
>
---
#### [replaced 005] SupertonicTTS: Towards Highly Scalable and Efficient Text-to-Speech System
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.23108v2](http://arxiv.org/pdf/2503.23108v2)**

> **作者:** Hyeongju Kim; Jinhyeok Yang; Yechan Yu; Seunghun Ji; Jacob Morton; Frederik Bous; Joon Byun; Juheon Lee
>
> **备注:** 21 pages, preprint
>
> **摘要:** We present a novel text-to-speech (TTS) system, namely SupertonicTTS, for improved scalability and efficiency in speech synthesis. SupertonicTTS comprises three components: a speech autoencoder for continuous latent representation, a text-to-latent module leveraging flow-matching for text-to-latent mapping, and an utterance-level duration predictor. To enable a lightweight architecture, we employ a low-dimensional latent space, temporal compression of latents, and ConvNeXt blocks. We further simplify the TTS pipeline by operating directly on raw character-level text and employing cross-attention for text-speech alignment, thus eliminating the need for grapheme-to-phoneme (G2P) modules and external aligners. In addition, we introduce context-sharing batch expansion that accelerates loss convergence and stabilizes text-speech alignment. Experimental results demonstrate that SupertonicTTS achieves competitive performance while significantly reducing architectural complexity and computational overhead compared to contemporary TTS models. Audio samples demonstrating the capabilities of SupertonicTTS are available at: https://supertonictts.github.io/.
>
---
#### [replaced 006] ImprovNet -- Generating Controllable Musical Improvisations with Iterative Corruption Refinement
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.04522v4](http://arxiv.org/pdf/2502.04522v4)**

> **作者:** Keshav Bhandari; Sungkyun Chang; Tongyu Lu; Fareza R. Enus; Louis B. Bradshaw; Dorien Herremans; Simon Colton
>
> **备注:** 10 pages, 6 figures, IJCNN 2025 conference
>
> **摘要:** Despite deep learning's remarkable advances in style transfer across various domains, generating controllable performance-level musical style transfer for complete symbolically represented musical works remains a challenging area of research. Much of this is owed to limited datasets, especially for genres such as jazz, and the lack of unified models that can handle multiple music generation tasks. This paper presents ImprovNet, a transformer-based architecture that generates expressive and controllable musical improvisations through a self-supervised corruption-refinement training strategy. The improvisational style transfer is aimed at making meaningful modifications to one or more musical elements - melody, harmony or rhythm of the original composition with respect to the target genre. ImprovNet unifies multiple capabilities within a single model: it can perform cross-genre and intra-genre improvisations, harmonize melodies with genre-specific styles, and execute short prompt continuation and infilling tasks. The model's iterative generation framework allows users to control the degree of style transfer and structural similarity to the original composition. Objective and subjective evaluations demonstrate ImprovNet's effectiveness in generating musically coherent improvisations while maintaining structural relationships with the original pieces. The model outperforms Anticipatory Music Transformer in short continuation and infilling tasks and successfully achieves recognizable genre conversion, with 79\% of participants correctly identifying jazz-style improvisations of classical pieces. Our code and demo page can be found at https://github.com/keshavbhandari/improvnet.
>
---
