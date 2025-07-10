# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 3 篇**

## 最新发布

#### [new 001] Advances in Intelligent Hearing Aids: Deep Learning Approaches to Selective Noise Cancellation
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文属于智能助听器任务，解决噪声抑制问题。通过深度学习方法提升降噪效果，分析模型性能与部署挑战。**

- **链接: [http://arxiv.org/pdf/2507.07043v1](http://arxiv.org/pdf/2507.07043v1)**

> **作者:** Haris Khan; Shumaila Asif; Hassan Nasir
>
> **备注:** 22 pages, 4 figures, submitted as a systematic literature review in AI-based hearing assistance. (June 2025)
>
> **摘要:** The integration of artificial intelligence into hearing assistance marks a paradigm shift from traditional amplification-based systems to intelligent, context-aware audio processing. This systematic literature review evaluates advances in AI-driven selective noise cancellation (SNC) for hearing aids, highlighting technological evolution, implementation challenges, and future research directions. We synthesize findings across deep learning architectures, hardware deployment strategies, clinical validation studies, and user-centric design. The review traces progress from early machine learning models to state-of-the-art deep networks, including Convolutional Recurrent Networks for real-time inference and Transformer-based architectures for high-accuracy separation. Key findings include significant gains over traditional methods, with recent models achieving up to 18.3 dB SI-SDR improvement on noisy-reverberant benchmarks, alongside sub-10 ms real-time implementations and promising clinical outcomes. Yet, challenges remain in bridging lab-grade models with real-world deployment - particularly around power constraints, environmental variability, and personalization. Identified research gaps include hardware-software co-design, standardized evaluation protocols, and regulatory considerations for AI-enhanced hearing devices. Future work must prioritize lightweight models, continual learning, contextual-based classification and clinical translation to realize transformative hearing solutions for millions globally.
>
---
#### [new 002] IMPACT: Industrial Machine Perception via Acoustic Cognitive Transformer
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于工业音频分析任务，解决工业机器声音检测与理解问题。提出DINOS数据集和IMPACT模型，提升小样本下的泛化能力与性能。**

- **链接: [http://arxiv.org/pdf/2507.06481v1](http://arxiv.org/pdf/2507.06481v1)**

> **作者:** Changheon Han; Yuseop Sim; Hoin Jung; Jiho Lee; Hojun Lee; Yun Seok Kang; Sucheol Woo; Garam Kim; Hyung Wook Park; Martin Byung-Guk Jun
>
> **摘要:** Acoustic signals from industrial machines offer valuable insights for anomaly detection, predictive maintenance, and operational efficiency enhancement. However, existing task-specific, supervised learning methods often scale poorly and fail to generalize across diverse industrial scenarios, whose acoustic characteristics are distinct from general audio. Furthermore, the scarcity of accessible, large-scale datasets and pretrained models tailored for industrial audio impedes community-driven research and benchmarking. To address these challenges, we introduce DINOS (Diverse INdustrial Operation Sounds), a large-scale open-access dataset. DINOS comprises over 74,149 audio samples (exceeding 1,093 hours) collected from various industrial acoustic scenarios. We also present IMPACT (Industrial Machine Perception via Acoustic Cognitive Transformer), a novel foundation model for industrial machine sound analysis. IMPACT is pretrained on DINOS in a self-supervised manner. By jointly optimizing utterance and frame-level losses, it captures both global semantics and fine-grained temporal structures. This makes its representations suitable for efficient fine-tuning on various industrial downstream tasks with minimal labeled data. Comprehensive benchmarking across 30 distinct downstream tasks (spanning four machine types) demonstrates that IMPACT outperforms existing models on 24 tasks, establishing its superior effectiveness and robustness, while providing a new performance benchmark for future research.
>
---
#### [new 003] MixAssist: An Audio-Language Dataset for Co-Creative AI Assistance in Music Mixing
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频-语言任务，旨在解决音乐混音中AI协作指导不足的问题。通过构建MixAssist数据集，支持训练能理解并回应音乐制作对话的模型。**

- **链接: [http://arxiv.org/pdf/2507.06329v1](http://arxiv.org/pdf/2507.06329v1)**

> **作者:** Michael Clemens; Ana Marasović
>
> **备注:** Published at COLM 2025. Code and dataset are available here http://mclemcrew.github.io/mixassist-website
>
> **摘要:** While AI presents significant potential for enhancing music mixing and mastering workflows, current research predominantly emphasizes end-to-end automation or generation, often overlooking the collaborative and instructional dimensions vital for co-creative processes. This gap leaves artists, particularly amateurs seeking to develop expertise, underserved. To bridge this, we introduce MixAssist, a novel audio-language dataset capturing the situated, multi-turn dialogue between expert and amateur music producers during collaborative mixing sessions. Comprising 431 audio-grounded conversational turns derived from 7 in-depth sessions involving 12 producers, MixAssist provides a unique resource for training and evaluating audio-language models that can comprehend and respond to the complexities of real-world music production dialogues. Our evaluations, including automated LLM-as-a-judge assessments and human expert comparisons, demonstrate that fine-tuning models such as Qwen-Audio on MixAssist can yield promising results, with Qwen significantly outperforming other tested models in generating helpful, contextually relevant mixing advice. By focusing on co-creative instruction grounded in audio context, MixAssist enables the development of intelligent AI assistants designed to support and augment the creative process in music mixing.
>
---
#### [new 004] Latent Acoustic Mapping for Direction of Arrival Estimation: A Self-Supervised Approach
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于声源定位任务，解决传统方法计算量大、深度学习方法依赖标注数据的问题。提出自监督的LAM模型，生成高分辨率声学图，提升定位性能与适应性。**

- **链接: [http://arxiv.org/pdf/2507.07066v1](http://arxiv.org/pdf/2507.07066v1)**

> **作者:** Adrian S. Roman; Iran R. Roman; Juan P. Bello
>
> **摘要:** Acoustic mapping techniques have long been used in spatial audio processing for direction of arrival estimation (DoAE). Traditional beamforming methods for acoustic mapping, while interpretable, often rely on iterative solvers that can be computationally intensive and sensitive to acoustic variability. On the other hand, recent supervised deep learning approaches offer feedforward speed and robustness but require large labeled datasets and lack interpretability. Despite their strengths, both methods struggle to consistently generalize across diverse acoustic setups and array configurations, limiting their broader applicability. We introduce the Latent Acoustic Mapping (LAM) model, a self-supervised framework that bridges the interpretability of traditional methods with the adaptability and efficiency of deep learning methods. LAM generates high-resolution acoustic maps, adapts to varying acoustic conditions, and operates efficiently across different microphone arrays. We assess its robustness on DoAE using the LOCATA and STARSS benchmarks. LAM achieves comparable or superior localization performance to existing supervised methods. Additionally, we show that LAM's acoustic maps can serve as effective features for supervised models, further enhancing DoAE accuracy and underscoring its potential to advance adaptive, high-performance sound localization systems.
>
---
#### [new 005] Constraint Optimized Multichannel Mixer-limiter Design
- **分类: cs.SD; eess.AS; eess.SP; math.OC**

- **简介: 该论文属于音频处理任务，解决多通道混音器与限幅器耦合设计问题。通过线性约束二次规划优化，降低失真并提升实时计算效率。**

- **链接: [http://arxiv.org/pdf/2507.06769v1](http://arxiv.org/pdf/2507.06769v1)**

> **作者:** Yuancheng Luo; Dmitriy Yamkovoy; Guillermo Garcia
>
> **备注:** For submission to ICASSP 2026
>
> **摘要:** Multichannel audio mixer and limiter designs are conventionally decoupled for content reproduction over loudspeaker arrays due to high computational complexity and run-time costs. We propose a coupled mixer-limiter-envelope design formulated as an efficient linear-constrained quadratic program that minimizes a distortion objective over multichannel gain variables subject to sample mixture constraints. Novel methods for asymmetric constant overlap-add window optimization, objective function approximation, variable and constraint reduction are presented. Experiments demonstrate distortion reduction of the coupled design, and computational trade-offs required for efficient real-time processing.
>
---
#### [new 006] Physics-Informed Direction-Aware Neural Acoustic Fields
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于声场建模任务，旨在解决FOA RIRs的建模问题。通过引入物理先验，扩展PINN框架以准确描述FOA信号的物理关系。**

- **链接: [http://arxiv.org/pdf/2507.06826v1](http://arxiv.org/pdf/2507.06826v1)**

> **作者:** Yoshiki Masuyama; François G. Germain; Gordon Wichern; Christopher Ick; Jonathan Le Roux
>
> **备注:** Accepted to WASPAA 2025
>
> **摘要:** This paper presents a physics-informed neural network (PINN) for modeling first-order Ambisonic (FOA) room impulse responses (RIRs). PINNs have demonstrated promising performance in sound field interpolation by combining the powerful modeling capability of neural networks and the physical principles of sound propagation. In room acoustics, PINNs have typically been trained to represent the sound pressure measured by omnidirectional microphones where the wave equation or its frequency-domain counterpart, i.e., the Helmholtz equation, is leveraged. Meanwhile, FOA RIRs additionally provide spatial characteristics and are useful for immersive audio generation with a wide range of applications. In this paper, we extend the PINN framework to model FOA RIRs. We derive two physics-informed priors for FOA RIRs based on the correspondence between the particle velocity and the (X, Y, Z)-channels of FOA. These priors associate the predicted W-channel and other channels through their partial derivatives and impose the physically feasible relationship on the four channels. Our experiments confirm the effectiveness of the proposed method compared with a neural network without the physics-informed prior.
>
---
#### [new 007] STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于歌唱语音标注任务，解决手动标注耗时问题，提出STARS框架实现统一的转录、对齐与风格标注。**

- **链接: [http://arxiv.org/pdf/2507.06670v1](http://arxiv.org/pdf/2507.06670v1)**

> **作者:** Wenxiang Guo; Yu Zhang; Changhao Pan; Zhiyuan Zhu; Ruiqi Li; Zhetao Chen; Wenhao Xu; Fei Wu; Zhou Zhao
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Recent breakthroughs in singing voice synthesis (SVS) have heightened the demand for high-quality annotated datasets, yet manual annotation remains prohibitively labor-intensive and resource-intensive. Existing automatic singing annotation (ASA) methods, however, primarily tackle isolated aspects of the annotation pipeline. To address this fundamental challenge, we present STARS, which is, to our knowledge, the first unified framework that simultaneously addresses singing transcription, alignment, and refined style annotation. Our framework delivers comprehensive multi-level annotations encompassing: (1) precise phoneme-audio alignment, (2) robust note transcription and temporal localization, (3) expressive vocal technique identification, and (4) global stylistic characterization including emotion and pace. The proposed architecture employs hierarchical acoustic feature processing across frame, word, phoneme, note, and sentence levels. The novel non-autoregressive local acoustic encoders enable structured hierarchical representation learning. Experimental validation confirms the framework's superior performance across multiple evaluation dimensions compared to existing annotation approaches. Furthermore, applications in SVS training demonstrate that models utilizing STARS-annotated data achieve significantly enhanced perceptual naturalness and precise style control. This work not only overcomes critical scalability challenges in the creation of singing datasets but also pioneers new methodologies for controllable singing voice synthesis. Audio samples are available at https://gwx314.github.io/stars-demo/.
>
---
#### [new 008] Exploring State-Space-Model based Language Model in Music Generation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于文本到音乐生成任务，旨在探索基于Mamba的模型在该任务中的表现，通过单层代码本表示和SiMBA解码器实现高效生成。**

- **链接: [http://arxiv.org/pdf/2507.06674v1](http://arxiv.org/pdf/2507.06674v1)**

> **作者:** Wei-Jaw Lee; Fang-Chih Hsieh; Xuanjun Chen; Fang-Duo Tsai; Yi-Hsuan Yang
>
> **备注:** Accepted at ISMIR 2025 as Late-Breaking Demo (LBD)
>
> **摘要:** The recent surge in State Space Models (SSMs), particularly the emergence of Mamba, has established them as strong alternatives or complementary modules to Transformers across diverse domains. In this work, we aim to explore the potential of Mamba-based architectures for text-to-music generation. We adopt discrete tokens of Residual Vector Quantization (RVQ) as the modeling representation and empirically find that a single-layer codebook can capture semantic information in music. Motivated by this observation, we focus on modeling a single-codebook representation and adapt SiMBA, originally designed as a Mamba-based encoder, to function as a decoder for sequence modeling. We compare its performance against a standard Transformer-based decoder. Our results suggest that, under limited-resource settings, SiMBA achieves much faster convergence and generates outputs closer to the ground truth. This demonstrates the promise of SSMs for efficient and expressive text-to-music generation. We put audio examples on Github.
>
---
#### [new 009] A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在提升模型在多数据集上的泛化能力。提出DCRF-BiLSTM模型，在多个数据集上取得高准确率。**

- **链接: [http://arxiv.org/pdf/2507.07046v1](http://arxiv.org/pdf/2507.07046v1)**

> **作者:** Shahana Yasmin Chowdhury; Bithi Banik; Md Tamjidul Hoque; Shreya Banerjee
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets.
>
---
#### [new 010] Revealing the Hidden Temporal Structure of HubertSoft Embeddings based on the Russian Phonetic Corpus
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，研究SSL模型是否保留语音的时序结构。通过分析HubertSoft嵌入，验证其在音素边界处是否反映音素身份和顺序。**

- **链接: [http://arxiv.org/pdf/2507.06794v1](http://arxiv.org/pdf/2507.06794v1)**

> **作者:** Anastasia Ananeva; Anton Tomilov; Marina Volkova
>
> **备注:** 11 pages, 5 figures, Specom 2025 conference
>
> **摘要:** Self-supervised learning (SSL) models such as Wav2Vec 2.0 and HuBERT have shown remarkable success in extracting phonetic information from raw audio without labelled data. While prior work has demonstrated that SSL embeddings encode phonetic features at the frame level, it remains unclear whether these models preserve temporal structure, specifically, whether embeddings at phoneme boundaries reflect the identity and order of adjacent phonemes. This study investigates the extent to which boundary-sensitive embeddings from HubertSoft, a soft-clustering variant of HuBERT, encode phoneme transitions. Using the CORPRES Russian speech corpus, we labelled 20 ms embedding windows with triplets of phonemes corresponding to their start, centre, and end segments. A neural network was trained to predict these positions separately, and multiple evaluation metrics, such as ordered, unordered accuracy and a flexible centre accuracy, were used to assess temporal sensitivity. Results show that embeddings extracted at phoneme boundaries capture both phoneme identity and temporal order, with especially high accuracy at segment boundaries. Confusion patterns further suggest that the model encodes articulatory detail and coarticulatory effects. These findings contribute to our understanding of the internal structure of SSL speech representations and their potential for phonological analysis and fine-grained transcription tasks.
>
---
#### [new 011] Comparative Analysis of CNN and Transformer Architectures with Heart Cycle Normalization for Automated Phonocardiogram Classification
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于心音图自动分类任务，旨在解决心杂音检测问题。通过对比CNN与Transformer模型，采用不同归一化方法提升分类效果。**

- **链接: [http://arxiv.org/pdf/2507.07058v1](http://arxiv.org/pdf/2507.07058v1)**

> **作者:** Martin Sondermann; Pinar Bisgin; Niklas Tschorn; Anja Burmann; Christoph M. Friedrich
>
> **备注:** Preprint Version. Accepted at EMBC 2025
>
> **摘要:** The automated classification of phonocardiogram (PCG) recordings represents a substantial advancement in cardiovascular diagnostics. This paper presents a systematic comparison of four distinct models for heart murmur detection: two specialized convolutional neural networks (CNNs) and two zero-shot universal audio transformers (BEATs), evaluated using fixed-length and heart cycle normalization approaches. Utilizing the PhysioNet2022 dataset, a custom heart cycle normalization method tailored to individual cardiac rhythms is introduced. The findings indicate the following AUROC values: the CNN model with fixed-length windowing achieves 79.5%, the CNN model with heart cycle normalization scores 75.4%, the BEATs transformer with fixed-length windowing achieves 65.7%, and the BEATs transformer with heart cycle normalization results in 70.1%. The findings indicate that physiological signal constraints, especially those introduced by different normalization strategies, have a substantial impact on model performance. The research provides evidence-based guidelines for architecture selection in clinical settings, emphasizing the need for a balance between accuracy and computational efficiency. Although specialized CNNs demonstrate superior performance overall, the zero-shot transformer models may offer promising efficiency advantages during development, such as faster training and evaluation cycles, despite their lower classification accuracy. These findings highlight the potential of automated classification systems to enhance cardiac diagnostics and improve patient care.
>
---
#### [new 012] Data-Balanced Curriculum Learning for Audio Question Answering
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频问答任务，解决数据不平衡和训练不稳定问题。通过结合课程学习与数据平衡方法，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.06815v1](http://arxiv.org/pdf/2507.06815v1)**

> **作者:** Gijs Wijngaard; Elia Formisano; Michele Esposito; Michel Dumontier
>
> **摘要:** Audio question answering (AQA) requires models to understand acoustic content and perform complex reasoning. Current models struggle with dataset imbalances and unstable training dynamics. This work combines curriculum learning with statistical data balancing to address these challenges. The method labels question difficulty using language models, then trains progressively from easy to hard examples. Statistical filtering removes overrepresented audio categories, and guided decoding constrains outputs to valid multiple-choice formats. Experiments on the DCASE 2025 training set and five additional public datasets show that data curation improves accuracy by 11.7% over baseline models, achieving 64.2% on the DCASE 2025 benchmark.
>
---
#### [new 013] Open-Set Source Tracing of Audio Deepfake Systems
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频深度伪造源追踪任务，解决开放集场景下的追踪问题。通过引入Softmax能量（SME）等方法，提升追踪准确率，FPR95达到8.3%。**

- **链接: [http://arxiv.org/pdf/2507.06470v1](http://arxiv.org/pdf/2507.06470v1)**

> **作者:** Nicholas Klein; Hemlata Tak; Elie Khoury
>
> **备注:** Accepted by INTERSPEECH 2025 as part of the special session "Source Tracing: The Origins of Synthetic or Manipulated Speech"
>
> **摘要:** Existing research on source tracing of audio deepfake systems has focused primarily on the closed-set scenario, while studies that evaluate open-set performance are limited to a small number of unseen systems. Due to the large number of emerging audio deepfake systems, robust open-set source tracing is critical. We leverage the protocol of the Interspeech 2025 special session on source tracing to evaluate methods for improving open-set source tracing performance. We introduce a novel adaptation to the energy score for out-of-distribution (OOD) detection, softmax energy (SME). We find that replacing the typical temperature-scaled energy score with SME provides a relative average improvement of 31% in the standard FPR95 (false positive rate at true positive rate of 95%) measure. We further explore SME-guided training as well as copy synthesis, codec, and reverberation augmentations, yielding an FPR95 of 8.3%.
>
---
#### [new 014] Incremental Averaging Method to Improve Graph-Based Time-Difference-of-Arrival Estimation
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于声源定位任务，旨在解决噪声和混响影响下的TDOA估计问题。通过改进的增量平均方法提升GCC-PHAT性能，提高定位精度。**

- **链接: [http://arxiv.org/pdf/2507.07087v1](http://arxiv.org/pdf/2507.07087v1)**

> **作者:** Klaus Brümann; Kouei Yamaoka; Nobutaka Ono; Simon Doclo
>
> **摘要:** Estimating the position of a speech source based on time-differences-of-arrival (TDOAs) is often adversely affected by background noise and reverberation. A popular method to estimate the TDOA between a microphone pair involves maximizing a generalized cross-correlation with phase transform (GCC-PHAT) function. Since the TDOAs across different microphone pairs satisfy consistency relations, generally only a small subset of microphone pairs are used for source position estimation. Although the set of microphone pairs is often determined based on a reference microphone, recently a more robust method has been proposed to determine the set of microphone pairs by computing the minimum spanning tree (MST) of a signal graph of GCC-PHAT function reliabilities. To reduce the influence of noise and reverberation on the TDOA estimation accuracy, in this paper we propose to compute the GCC-PHAT functions of the MST based on an average of multiple cross-power spectral densities (CPSDs) using an incremental method. In each step of the method, we increase the number of CPSDs over which we average by considering CPSDs computed indirectly via other microphones from previous steps. Using signals recorded in a noisy and reverberant laboratory with an array of spatially distributed microphones, the performance of the proposed method is evaluated in terms of TDOA estimation error and 2D source position estimation error. Experimental results for different source and microphone configurations and three reverberation conditions show that the proposed method considering multiple CPSDs improves the TDOA estimation and source position estimation accuracy compared to the reference microphone- and MST-based methods that rely on a single CPSD as well as steered-response power-based source position estimation.
>
---
#### [new 015] Deep Feed-Forward Neural Network for Bangla Isolated Speech Recognition
- **分类: eess.AS; cs.SD; 68T05; I.2.7; I.5.1; H.5.2**

- **简介: 该论文属于语音识别任务，旨在解决Bangla孤立语音识别问题。通过使用MFCC和深度前馈神经网络，实现了93.42%的识别准确率。**

- **链接: [http://arxiv.org/pdf/2507.07068v1](http://arxiv.org/pdf/2507.07068v1)**

> **作者:** Dipayan Bhadra; Mehrab Hosain; Fatema Alam
>
> **备注:** 12 pages, 3 figures, 4 tables. published in Jatiya Kabi Kazi Nazrul Islam University, Vol. 10 No. 1-2, 2025 https://jkkniu.edu.bd/13817-2/
>
> **摘要:** As the most important human-machine interfacing tool, an insignificant amount of work has been carried out on Bangla Speech Recognition compared to the English language. Motivated by this, in this work, the performance of speaker-independent isolated speech recognition systems has been implemented and analyzed using a dataset that is created containing both isolated Bangla and English spoken words. An approach using the Mel Frequency Cepstral Coefficient (MFCC) and Deep Feed-Forward Fully Connected Neural Network (DFFNN) of 7 layers as a classifier is proposed in this work to recognize isolated spoken words. This work shows 93.42% recognition accuracy which is better compared to most of the works done previously on Bangla speech recognition considering the number of classes and dataset size.
>
---
#### [new 016] Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于语音情感研究任务，旨在探索如何通过调整语音参数增强"可爱"感。研究分析了TTS和游戏角色语音，发现特定频率调整可提升kawaii效果。**

- **链接: [http://arxiv.org/pdf/2507.06235v1](http://arxiv.org/pdf/2507.06235v1)**

> **作者:** Yuto Mandai; Katie Seaborn; Tomoyasu Nakano; Xin Sun; Yijia Wang; Jun Kato
>
> **备注:** CHI '25
>
> **摘要:** "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice.
>
---
#### [new 017] Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于安全领域，研究音频大模型的漏洞。通过构造噪声干扰，攻击者可操控模型行为，影响用户体验，提出防御思路。**

- **链接: [http://arxiv.org/pdf/2507.06256v1](http://arxiv.org/pdf/2507.06256v1)**

> **作者:** Vinu Sankar Sadasivan; Soheil Feizi; Rajiv Mathews; Lun Wang
>
> **摘要:** This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.
>
---
## 更新

#### [replaced 001] XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.23325v2](http://arxiv.org/pdf/2506.23325v2)**

> **作者:** Yitian Gong; Luozhijie Jin; Ruifan Deng; Dong Zhang; Xin Zhang; Qinyuan Cheng; Zhaoye Fei; Shimin Li; Xipeng Qiu
>
> **摘要:** Speech codecs serve as bridges between speech signals and large language models. An ideal codec for speech language models should not only preserve acoustic information but also capture rich semantic information. However, existing speech codecs struggle to balance high-quality audio reconstruction with ease of modeling by language models. In this study, we analyze the limitations of previous codecs in balancing semantic richness and acoustic fidelity. We propose XY-Tokenizer, a novel codec that mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning. Experimental results demonstrate that XY-Tokenizer achieves performance in both semantic and acoustic tasks comparable to that of state-of-the-art codecs operating at similar bitrates, even though those existing codecs typically excel in only one aspect. Specifically, XY-Tokenizer achieves strong text alignment, surpassing distillation-based semantic modeling methods such as SpeechTokenizer and Mimi, while maintaining a speaker similarity score of 0.83 between reconstructed and original audio. The reconstruction performance of XY-Tokenizer is comparable to that of BigCodec, the current state-of-the-art among acoustic-only codecs, which achieves a speaker similarity score of 0.84 at a similar bitrate. Code and models are available at https://github.com/gyt1145028706/XY-Tokenizer.
>
---
#### [replaced 002] Rethinking Non-Negative Matrix Factorization with Implicit Neural Representations
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2404.04439v2](http://arxiv.org/pdf/2404.04439v2)**

> **作者:** Krishna Subramani; Paris Smaragdis; Takuya Higuchi; Mehrez Souden
>
> **备注:** WASPAA 2025, Code: https://github.com/SubramaniKrishna/in-nmf
>
> **摘要:** Non-negative Matrix Factorization (NMF) is a powerful technique for analyzing regularly-sampled data, i.e., data that can be stored in a matrix. For audio, this has led to numerous applications using time-frequency (TF) representations like the Short-Time Fourier Transform. However extending these applications to irregularly-spaced TF representations, like the Constant-Q transform, wavelets, or sinusoidal analysis models, has not been possible since these representations cannot be directly stored in matrix form. In this paper, we formulate NMF in terms of learnable functions (instead of vectors) and show that NMF can be extended to a wider variety of signal classes that need not be regularly sampled.
>
---
#### [replaced 003] A Voice-based Triage for Type 2 Diabetes using a Conversational Virtual Assistant in the Home Environment
- **分类: cs.SD; eess.AS; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.19204v2](http://arxiv.org/pdf/2411.19204v2)**

> **作者:** Kelvin Summoogum; Debayan Das; Sathish Kumaran
>
> **备注:** 8 pages
>
> **摘要:** Incorporating cloud technology with Internet of Medical Things for ubiquitous healthcare has seen many successful applications in the last decade with the advent of machine learning and deep learning techniques. One of these applications, namely voice-based pathology, has yet to receive notable attention from academia and industry. Applying voice analysis to early detection of fatal diseases holds much promise to improve health outcomes and quality of life of patients. In this paper, we propose a novel application of acoustic machine learning based triaging into commoditised conversational virtual assistant systems to pre-screen for onset of diabetes. Specifically, we developed a triaging system which extracts acoustic features from the voices of n=24 older adults when they converse with a virtual assistant and predict the incidence of Diabetes Mellitus (Type 2) or not. Our triaging system achieved hit-rates of 70% and 60% for male and female older adult subjects, respectively. Our proposed triaging uses 7 non-identifiable voice-based features and can operate within resource-constrained embedded systems running voice-based virtual assistants. This application demonstrates the feasibility of applying voice-based pathology analysis to improve health outcomes of older adults within the home environment by early detection of life-changing chronic conditions like diabetes.
>
---
