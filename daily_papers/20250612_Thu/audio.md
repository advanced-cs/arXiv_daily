# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] OWSM-Biasing: Contextualizing Open Whisper-Style Speech Models for Automatic Speech Recognition with Dynamic Vocabulary
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于自动语音识别任务，旨在解决罕见词识别问题。通过结合上下文偏置方法与预训练模型，提升识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.09448v1](http://arxiv.org/pdf/2506.09448v1)**

> **作者:** Yui Sudo; Yusuke Fujita; Atsushi Kojima; Tomoya Mizumoto; Lianbo Liu
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Speech foundation models (SFMs), such as Open Whisper-Style Speech Models (OWSM), are trained on massive datasets to achieve accurate automatic speech recognition. However, even SFMs struggle to accurately recognize rare and unseen words. While contextual biasing (CB) is a promising approach to improve recognition of such words, most CB methods are trained from scratch, resulting in lower performance than SFMs due to the lack of pre-trained knowledge. This paper integrates an existing CB method with OWSM v3.1 while freezing its pre-trained parameters. By leveraging the knowledge embedded in SFMs, the proposed method enables effective CB while preserving the advantages of SFMs, even with a small dataset. Experimental results show that the proposed method improves the biasing word error rate (B-WER) by 11.6 points, resulting in a 0.9 point improvement in the overall WER while reducing the real-time factor by 7.5% compared to the non-biasing baseline on the LibriSpeech 100 test-clean set.
>
---
#### [new 002] Fractional Fourier Sound Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频处理任务，旨在利用分数阶傅里叶变换实现新型声音合成，解决传统方法难以实现的时频分析问题。通过数学分析与实验验证，提出alpha合成等新技术。**

- **链接: [http://arxiv.org/pdf/2506.09189v1](http://arxiv.org/pdf/2506.09189v1)**

> **作者:** Esteban Gutiérrez; Rodrigo Cádiz; Carlos Sing Long; Frederic Font; Xavier Serra
>
> **备注:** Accepted to the International Computer Music Conference (ICMC) 2025 held in Boston, USA. 6 pages and 2 figures
>
> **摘要:** This paper explores the innovative application of the Fractional Fourier Transform (FrFT) in sound synthesis, highlighting its potential to redefine time-frequency analysis in audio processing. As an extension of the classical Fourier Transform, the FrFT introduces fractional order parameters, enabling a continuous interpolation between time and frequency domains and unlocking unprecedented flexibility in signal manipulation. Crucially, the FrFT also opens the possibility of directly synthesizing sounds in the alpha-domain, providing a unique framework for creating timbral and dynamic characteristics unattainable through conventional methods. This work delves into the mathematical principles of the FrFT, its historical evolution, and its capabilities for synthesizing complex audio textures. Through experimental analyses, we showcase novel sound design techniques, such as alpha-synthesis and alpha-filtering, which leverage the FrFT's time-frequency rotation properties to produce innovative sonic results. The findings affirm the FrFT's value as a transformative tool for composers, sound designers, and researchers seeking to push the boundaries of auditory creativity.
>
---
#### [new 003] UmbraTTS: Adapting Text-to-Speech to Environmental Contexts with Flow Matching
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决将语音与复杂环境音频融合的问题。提出UmbraTTS模型，通过流匹配生成语音和环境音频，实现自然、环境感知的音频合成。**

- **链接: [http://arxiv.org/pdf/2506.09874v1](http://arxiv.org/pdf/2506.09874v1)**

> **作者:** Neta Glazer; Aviv Navon; Yael Segal; Aviv Shamsian; Hilit Segev; Asaf Buchnick; Menachem Pirchi; Gil Hetz; Joseph Keshet
>
> **摘要:** Recent advances in Text-to-Speech (TTS) have enabled highly natural speech synthesis, yet integrating speech with complex background environments remains challenging. We introduce UmbraTTS, a flow-matching based TTS model that jointly generates both speech and environmental audio, conditioned on text and acoustic context. Our model allows fine-grained control over background volume and produces diverse, coherent, and context-aware audio scenes. A key challenge is the lack of data with speech and background audio aligned in natural context. To overcome the lack of paired training data, we propose a self-supervised framework that extracts speech, background audio, and transcripts from unannotated recordings. Extensive evaluations demonstrate that UmbraTTS significantly outperformed existing baselines, producing natural, high-quality, environmentally aware audios.
>
---
#### [new 004] Training-Free Voice Conversion with Factorized Optimal Transport
- **分类: cs.SD; cs.CV; cs.LG; eess.AS**

- **简介: 该论文属于语音转换任务，解决短参考音频下的跨语言语音转换问题。提出MKL-VC方法，利用因子化最优传输提升转换质量与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.09709v1](http://arxiv.org/pdf/2506.09709v1)**

> **作者:** Alexander Lobashev; Assel Yermekova; Maria Larchenko
>
> **备注:** Interspeech 2025
>
> **摘要:** This paper introduces Factorized MKL-VC, a training-free modification for kNN-VC pipeline. In contrast with original pipeline, our algorithm performs high quality any-to-any cross-lingual voice conversion with only 5 second of reference audio. MKL-VC replaces kNN regression with a factorized optimal transport map in WavLM embedding subspaces, derived from Monge-Kantorovich Linear solution. Factorization addresses non-uniform variance across dimensions, ensuring effective feature transformation. Experiments on LibriSpeech and FLEURS datasets show MKL-VC significantly improves content preservation and robustness with short reference audio, outperforming kNN-VC. MKL-VC achieves performance comparable to FACodec, especially in cross-lingual voice conversion domain.
>
---
#### [new 005] SimClass: A Classroom Speech Dataset Generated via Game Engine Simulation For Automatic Speech Recognition Research
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决课堂语音数据稀缺问题。通过游戏引擎生成合成课堂噪声和语音数据，构建SimClass数据集，提升语音模型的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.09206v1](http://arxiv.org/pdf/2506.09206v1)**

> **作者:** Ahmed Adel Attia; Jing Liu; Carl Espy-Wilson
>
> **摘要:** The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Public classroom datasets remain limited, and the lack of a dedicated classroom noise corpus prevents the use of standard data augmentation techniques. In this paper, we introduce a scalable methodology for synthesizing classroom noise using game engines, a framework that extends to other domains. Using this methodology, we present SimClass, a dataset that includes both a synthesized classroom noise corpus and a simulated classroom speech dataset. The speech data is generated by pairing a public children's speech corpus with YouTube lecture videos to approximate real classroom interactions in clean conditions. Our experiments on clean and noisy speech demonstrate that SimClass closely approximates real classroom speech, making it a valuable resource for developing robust speech recognition and enhancement models.
>
---
#### [new 006] Incorporating Linguistic Constraints from External Knowledge Source for Audio-Visual Target Speech Extraction
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音频-视觉目标语音提取任务，旨在提升语音质量和可懂度。通过引入预训练语言模型作为语言约束，增强模型性能，且不增加推理成本。**

- **链接: [http://arxiv.org/pdf/2506.09792v1](http://arxiv.org/pdf/2506.09792v1)**

> **作者:** Wenxuan Wu; Shuai Wang; Xixin Wu; Helen Meng; Haizhou Li
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Audio-visual target speaker extraction (AV-TSE) models primarily rely on target visual cues to isolate the target speaker's voice from others. We know that humans leverage linguistic knowledge, such as syntax and semantics, to support speech perception. Inspired by this, we explore the potential of pre-trained speech-language models (PSLMs) and pre-trained language models (PLMs) as auxiliary knowledge sources for AV-TSE. In this study, we propose incorporating the linguistic constraints from PSLMs or PLMs for the AV-TSE model as additional supervision signals. Without introducing any extra computational cost during inference, the proposed approach consistently improves speech quality and intelligibility. Furthermore, we evaluate our method in multi-language settings and visual cue-impaired scenarios and show robust performance gains.
>
---
#### [new 007] BemaGANv2: A Tutorial and Comparative Survey of GAN-based Vocoders for Long-Term Audio Generation
- **分类: cs.SD; cs.AI; cs.LG; cs.LO; eess.AS; I.2.6; H.5.5; I.5.1**

- **简介: 该论文属于语音生成任务，旨在提升长时音频的质量与稳定性。通过改进GAN架构，引入AMP模块和MED、MRD等 discriminator 结构，优化了周期性建模与长期依赖捕捉。**

- **链接: [http://arxiv.org/pdf/2506.09487v1](http://arxiv.org/pdf/2506.09487v1)**

> **作者:** Taesoo Park; Mungwi Jeong; Mingyu Park; Narae Kim; Junyoung Kim; Mujung Kim; Jisang Yoo; Hoyun Lee; Sanghoon Kim; Soonchul Kwon
>
> **备注:** 11 pages, 7 figures. Survey and tutorial paper. Currently under review at ICT Express as an extended version of our ICAIIC 2025 paper
>
> **摘要:** This paper presents a tutorial-style survey and implementation guide of BemaGANv2, an advanced GAN-based vocoder designed for high-fidelity and long-term audio generation. Built upon the original BemaGAN architecture, BemaGANv2 incorporates major architectural innovations by replacing traditional ResBlocks in the generator with the Anti-aliased Multi-Periodicity composition (AMP) module, which internally applies the Snake activation function to better model periodic structures. In the discriminator framework, we integrate the Multi-Envelope Discriminator (MED), a novel architecture we originally proposed, to extract rich temporal envelope features crucial for periodicity detection. Coupled with the Multi-Resolution Discriminator (MRD), this combination enables more accurate modeling of long-range dependencies in audio. We systematically evaluate various discriminator configurations, including MSD + MED, MSD + MRD, and MPD + MED + MRD, using objective metrics (FAD, SSIM, PLCC, MCD) and subjective evaluations (MOS, SMOS). This paper also provides a comprehensive tutorial on the model architecture, training methodology, and implementation to promote reproducibility. The code and pre-trained models are available at: https://github.com/dinhoitt/BemaGANv2.
>
---
#### [new 008] CoLMbo: Speaker Language Model for Descriptive Profiling
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出CoLMbo，一种用于生成详细说话人描述的语音语言模型，解决传统系统在提取人口统计特征上的不足。**

- **链接: [http://arxiv.org/pdf/2506.09375v1](http://arxiv.org/pdf/2506.09375v1)**

> **作者:** Massa Baali; Shuo Han; Syed Abdul Hannan; Purusottam Samal; Karanveer Singh; Soham Deshmukh; Rita Singh; Bhiksha Raj
>
> **摘要:** Speaker recognition systems are often limited to classification tasks and struggle to generate detailed speaker characteristics or provide context-rich descriptions. These models primarily extract embeddings for speaker identification but fail to capture demographic attributes such as dialect, gender, and age in a structured manner. This paper introduces CoLMbo, a Speaker Language Model (SLM) that addresses these limitations by integrating a speaker encoder with prompt-based conditioning. This allows for the creation of detailed captions based on speaker embeddings. CoLMbo utilizes user-defined prompts to adapt dynamically to new speaker characteristics and provides customized descriptions, including regional dialect variations and age-related traits. This innovative approach not only enhances traditional speaker profiling but also excels in zero-shot scenarios across diverse datasets, marking a significant advancement in the field of speaker recognition.
>
---
#### [new 009] Ming-Omni: A Unified Multimodal Model for Perception and Generation
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出Ming-Omni，一个统一的多模态模型，解决跨模态感知与生成任务，支持图像、文本、音频和视频处理，无需额外调整。**

- **链接: [http://arxiv.org/pdf/2506.09344v1](http://arxiv.org/pdf/2506.09344v1)**

> **作者:** Inclusion AI; Biao Gong; Cheng Zou; Chuanyang Zheng; Chunluan Zhou; Canxiang Yan; Chunxiang Jin; Chunjie Shen; Dandan Zheng; Fudong Wang; Furong Xu; GuangMing Yao; Jun Zhou; Jingdong Chen; Jianxin Sun; Jiajia Liu; Jianjiang Zhu; Jun Peng; Kaixiang Ji; Kaiyou Song; Kaimeng Ren; Libin Wang; Lixiang Ru; Lele Xie; Longhua Tan; Lyuxin Xue; Lan Wang; Mochen Bai; Ning Gao; Pei Chen; Qingpei Guo; Qinglong Zhang; Qiang Xu; Rui Liu; Ruijie Xiong; Sirui Gao; Tinghao Liu; Taisong Li; Weilong Chai; Xinyu Xiao; Xiaomei Wang; Xiaoxue Chen; Xiao Lu; Xiaoyu Li; Xingning Dong; Xuzheng Yu; Yi Yuan; Yuting Gao; Yunxiao Sun; Yipeng Chen; Yifei Wu; Yongjie Lyu; Ziping Ma; Zipeng Feng; Zhijiang Fang; Zhihao Qiu; Ziyuan Huang; Zhengyu He
>
> **备注:** 18 pages,8 figures
>
> **摘要:** We propose Ming-Omni, a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-Omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-Omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-Omni offers a powerful solution for unified perception and generation across all modalities. Notably, our proposed Ming-Omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community.
>
---
#### [new 010] A Technique for Isolating Lexically-Independent Phonetic Dependencies in Generative CNNs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决DNN如何表征语音规则的问题。通过调整网络结构，研究发现卷积层可独立于全连接层进行语音依赖性泛化。**

- **链接: [http://arxiv.org/pdf/2506.09218v1](http://arxiv.org/pdf/2506.09218v1)**

> **作者:** Bruno Ferenc Šegedin
>
> **摘要:** The ability of deep neural networks (DNNs) to represent phonotactic generalizations derived from lexical learning remains an open question. This study (1) investigates the lexically-invariant generalization capacity of generative convolutional neural networks (CNNs) trained on raw audio waveforms of lexical items and (2) explores the consequences of shrinking the fully-connected layer (FC) bottleneck from 1024 channels to 8 before training. Ultimately, a novel technique for probing a model's lexically-independent generalizations is proposed that works only under the narrow FC bottleneck: generating audio outputs by bypassing the FC and inputting randomized feature maps into the convolutional block. These outputs are equally biased by a phonotactic restriction in training as are outputs generated with the FC. This result shows that the convolutional layers can dynamically generalize phonetic dependencies beyond lexically-constrained configurations learned by the FC.
>
---
#### [new 011] InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属于多概念人体动画生成任务，解决多主体交互中条件控制不足的问题。通过引入区域绑定的条件注入机制，实现高质量多概念视频生成。**

- **链接: [http://arxiv.org/pdf/2506.09984v1](http://arxiv.org/pdf/2506.09984v1)**

> **作者:** Zhenzhi Wang; Jiaqi Yang; Jianwen Jiang; Chao Liang; Gaojie Lin; Zerong Zheng; Ceyuan Yang; Dahua Lin
>
> **备注:** TL;DR: The first multi-person dialogue video generation method from pairs of reference image and audio via explicit layout-aligned condition injection. See project page https://zhenzhiwang.github.io/interacthuman/ for more details
>
> **摘要:** End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios that multiple concepts could appears in the same video with rich human-human interactions and human-object interactions. Such global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in a iterative manner. This design enables the high-quality generation of controllable multi-concept human-centric videos. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods.
>
---
#### [new 012] A Study on Speech Assessment with Visual Cues
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于语音质量评估任务，旨在在无参考信号情况下提升语音可懂性预测。通过融合音频与视觉特征，提出多模态框架，显著提高了PESQ和STOI的预测精度。**

- **链接: [http://arxiv.org/pdf/2506.09549v1](http://arxiv.org/pdf/2506.09549v1)**

> **作者:** Shafique Ahmed; Ryandhimas E. Zezario; Nasir Saleem; Amir Hussain; Hsin-Min Wang; Yu Tsao
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Non-intrusive assessment of speech quality and intelligibility is essential when clean reference signals are unavailable. In this work, we propose a multimodal framework that integrates audio features and visual cues to predict PESQ and STOI scores. It employs a dual-branch architecture, where spectral features are extracted using STFT, and visual embeddings are obtained via a visual encoder. These features are then fused and processed by a CNN-BLSTM with attention, followed by multi-task learning to simultaneously predict PESQ and STOI. Evaluations on the LRS3-TED dataset, augmented with noise from the DEMAND corpus, show that our model outperforms the audio-only baseline. Under seen noise conditions, it improves LCC by 9.61% (0.8397->0.9205) for PESQ and 11.47% (0.7403->0.8253) for STOI. These results highlight the effectiveness of incorporating visual cues in enhancing the accuracy of non-intrusive speech assessment.
>
---
#### [new 013] Regularizing Learnable Feature Extraction for Automatic Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于自动语音识别任务，旨在解决可学习特征提取器易过拟合的问题。通过音频扰动和频域掩码等正则化方法提升性能。**

- **链接: [http://arxiv.org/pdf/2506.09804v1](http://arxiv.org/pdf/2506.09804v1)**

> **作者:** Peter Vieting; Maximilian Kannen; Benedikt Hilmes; Ralf Schlüter; Hermann Ney
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Neural front-ends are an appealing alternative to traditional, fixed feature extraction pipelines for automatic speech recognition (ASR) systems since they can be directly trained to fit the acoustic model. However, their performance often falls short compared to classical methods, which we show is largely due to their increased susceptibility to overfitting. This work therefore investigates regularization methods for training ASR models with learnable feature extraction front-ends. First, we examine audio perturbation methods and show that larger relative improvements can be obtained for learnable features. Additionally, we identify two limitations in the standard use of SpecAugment for these front-ends and propose masking in the short time Fourier transform (STFT)-domain as a simple but effective modification to address these challenges. Finally, integrating both regularization approaches effectively closes the performance gap between traditional and learnable features.
>
---
#### [new 014] PHRASED: Phrase Dictionary Biasing for Speech Translation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，旨在解决短语翻译困难的问题。通过引入短语字典偏置方法，提升模型对短语的翻译效果。**

- **链接: [http://arxiv.org/pdf/2506.09175v1](http://arxiv.org/pdf/2506.09175v1)**

> **作者:** Peidong Wang; Jian Xue; Rui Zhao; Junkun Chen; Aswin Shanmugam Subramanian; Jinyu Li
>
> **摘要:** Phrases are essential to understand the core concepts in conversations. However, due to their rare occurrence in training data, correct translation of phrases is challenging in speech translation tasks. In this paper, we propose a phrase dictionary biasing method to leverage pairs of phrases mapping from the source language to the target language. We apply the phrase dictionary biasing method to two types of widely adopted models, a transducer-based streaming speech translation model and a multimodal large language model. Experimental results show that the phrase dictionary biasing method outperforms phrase list biasing by 21% relatively for the streaming speech translation model. In addition, phrase dictionary biasing enables multimodal large language models to use external phrase information, achieving 85% relative improvement in phrase recall.
>
---
## 更新

#### [replaced 001] Teaching Physical Awareness to LLMs through Sounds
- **分类: cs.SD; cs.AI; cs.MM; cs.RO; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08524v2](http://arxiv.org/pdf/2506.08524v2)**

> **作者:** Weiguo Wang; Andy Nie; Wenrui Zhou; Yi Kai; Chengchen Hu
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world.
>
---
#### [replaced 002] Annotation-Free MIDI-to-Audio Synthesis via Concatenative Synthesis and Generative Refinement
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.16785v2](http://arxiv.org/pdf/2410.16785v2)**

> **作者:** Osamu Take; Taketo Akama
>
> **备注:** Work in progress; 7 pages, 4 figures, 3 tables
>
> **摘要:** Recent MIDI-to-audio synthesis methods using deep neural networks have successfully generated high-quality, expressive instrumental tracks. However, these methods require MIDI annotations for supervised training, limiting the diversity of instrument timbres and expression styles in the output. We propose CoSaRef, a MIDI-to-audio synthesis method that does not require MIDI-audio paired datasets. CoSaRef first generates a synthetic audio track using concatenative synthesis based on MIDI input, then refines it with a diffusion-based deep generative model trained on datasets without MIDI annotations. This approach improves the diversity of timbres and expression styles. Additionally, it allows detailed control over timbres and expression through audio sample selection and extra MIDI design, similar to traditional functions in digital audio workstations. Experiments showed that CoSaRef could generate realistic tracks while preserving fine-grained timbre control via one-shot samples. Moreover, despite not being supervised on MIDI annotation, CoSaRef outperformed the state-of-the-art timbre-controllable method based on MIDI supervision in both objective and subjective evaluation.
>
---
#### [replaced 003] LID Models are Actually Accent Classifiers: Implications and Solutions for LID on Accented Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00628v2](http://arxiv.org/pdf/2506.00628v2)**

> **作者:** Niyati Bafna; Matthew Wiesner
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Prior research indicates that LID model performance significantly declines on accented speech; however, the specific causes, extent, and characterization of these errors remain under-explored. (i) We identify a common failure mode on accented speech whereby LID systems often misclassify L2 accented speech as the speaker's native language or a related language. (ii) We present evidence suggesting that state-of-the-art models are invariant to permutations of short spans of speech, implying they classify on the basis of short phonotactic features indicative of accent rather than language. Our analysis reveals a simple method to enhance model robustness to accents through input chunking. (iii) We present an approach that integrates sequence-level information into our model without relying on monolingual ASR systems; this reduces accent-language confusion and significantly enhances performance on accented speech while maintaining comparable results on standard LID.
>
---
#### [replaced 004] Channel Adaptation for Speaker Verification Using Optimal Transport with Pseudo Label
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.09396v2](http://arxiv.org/pdf/2409.09396v2)**

> **作者:** Wenhao Yang; Jianguo Wei; Wenhuan Lu; Lei Li; Xugang Lu
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Domain gap often degrades the performance of speaker verification (SV) systems when the statistical distributions of training data and real-world test speech are mismatched. Channel variation, a primary factor causing this gap, is less addressed than other issues (e.g., noise). Although various domain adaptation algorithms could be applied to handle this domain gap problem, most algorithms could not take the complex distribution structure in domain alignment with discriminative learning. In this paper, we propose a novel unsupervised domain adaptation method, i.e., Joint Partial Optimal Transport with Pseudo Label (JPOT-PL), to alleviate the channel mismatch problem. Leveraging the geometric-aware distance metric of optimal transport in distribution alignment, we further design a pseudo label-based discriminative learning where the pseudo label can be regarded as a new type of soft speaker label derived from the optimal coupling. With the JPOT-PL, we carry out experiments on the SV channel adaptation task with VoxCeleb as the basis corpus. Experiments show our method reduces EER by over 10% compared with several state-of-the-art channel adaptation algorithms.
>
---
#### [replaced 005] Listen, Chat, and Remix: Text-Guided Soundscape Remixing for Enhanced Auditory Experience
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2402.03710v2](http://arxiv.org/pdf/2402.03710v2)**

> **作者:** Xilin Jiang; Cong Han; Yinghao Aaron Li; Nima Mesgarani
>
> **备注:** Accepted by IEEE Journal of Selected Topics in Signal Processing (JSTSP)
>
> **摘要:** In daily life, we encounter a variety of sounds, both desirable and undesirable, with limited control over their presence and volume. Our work introduces "Listen, Chat, and Remix" (LCR), a novel multimodal sound remixer that controls each sound source in a mixture based on user-provided text instructions. LCR distinguishes itself with a user-friendly text interface and its unique ability to remix multiple sound sources simultaneously within a mixture, without needing to separate them. Users input open-vocabulary text prompts, which are interpreted by a large language model to create a semantic filter for remixing the sound mixture. The system then decomposes the mixture into its components, applies the semantic filter, and reassembles filtered components back to the desired output. We developed a 160-hour dataset with over 100k mixtures, including speech and various audio sources, along with text prompts for diverse remixing tasks including extraction, removal, and volume control of single or multiple sources. Our experiments demonstrate significant improvements in signal quality across all remixing tasks and robust performance in zero-shot scenarios with varying numbers and types of sound sources. An audio demo is available at: https://listenchatremix.github.io/demo.
>
---
#### [replaced 006] Towards Energy-Efficient and Low-Latency Voice-Controlled Smart Homes: A Proposal for Offline Speech Recognition and IoT Integration
- **分类: cs.SD; cs.CY; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07494v2](http://arxiv.org/pdf/2506.07494v2)**

> **作者:** Peng Huang; Imdad Ullah; Xiaotong Wei; Tariq Ahamed Ahanger; Najm Hassan; Zawar Hussain Shah
>
> **摘要:** The smart home systems, based on AI speech recognition and IoT technology, enable people to control devices through verbal commands and make people's lives more efficient. However, existing AI speech recognition services are primarily deployed on cloud platforms on the Internet. When users issue a command, speech recognition devices like ``Amazon Echo'' will post a recording through numerous network nodes, reach multiple servers, and then receive responses through the Internet. This mechanism presents several issues, including unnecessary energy consumption, communication latency, and the risk of a single-point failure. In this position paper, we propose a smart home concept based on offline speech recognition and IoT technology: 1) integrating offline keyword spotting (KWS) technologies into household appliances with limited resource hardware to enable them to understand user voice commands; 2) designing a local IoT network with decentralized architecture to manage and connect various devices, enhancing the robustness and scalability of the system. This proposal of a smart home based on offline speech recognition and IoT technology will allow users to use low-latency voice control anywhere in the home without depending on the Internet and provide better scalability and energy sustainability.
>
---
#### [replaced 007] Auto-Regressive vs Flow-Matching: a Comparative Study of Modeling Paradigms for Text-to-Music Generation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08570v2](http://arxiv.org/pdf/2506.08570v2)**

> **作者:** Or Tal; Felix Kreuk; Yossi Adi
>
> **摘要:** Recent progress in text-to-music generation has enabled models to synthesize high-quality musical segments, full compositions, and even respond to fine-grained control signals, e.g. chord progressions. State-of-the-art (SOTA) systems differ significantly across many dimensions, such as training datasets, modeling paradigms, and architectural choices. This diversity complicates efforts to evaluate models fairly and pinpoint which design choices most influence performance. While factors like data and architecture are important, in this study we focus exclusively on the modeling paradigm. We conduct a systematic empirical analysis to isolate its effects, offering insights into associated trade-offs and emergent behaviors that can guide future text-to-music generation systems. Specifically, we compare the two arguably most common modeling paradigms: Auto-Regressive decoding and Conditional Flow-Matching. We conduct a controlled comparison by training all models from scratch using identical datasets, training configurations, and similar backbone architectures. Performance is evaluated across multiple axes, including generation quality, robustness to inference configurations, scalability, adherence to both textual and temporally aligned conditioning, and editing capabilities in the form of audio inpainting. This comparative study sheds light on distinct strengths and limitations of each paradigm, providing actionable insights that can inform future architectural and training decisions in the evolving landscape of text-to-music generation. Audio sampled examples are available at: https://huggingface.co/spaces/ortal1602/ARvsFM
>
---
#### [replaced 008] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.16794v3](http://arxiv.org/pdf/2502.16794v3)**

> **作者:** Xilin Jiang; Sukru Samet Dindar; Vishal Choudhari; Stephan Bickel; Ashesh Mehta; Guy M McKhann; Daniel Friedman; Adeen Flinker; Nima Mesgarani
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Auditory foundation models, including auditory large language models (LLMs), process all sound inputs equally, independent of listener perception. However, human auditory perception is inherently selective: listeners focus on specific speakers while ignoring others in complex auditory scenes. Existing models do not incorporate this selectivity, limiting their ability to generate perception-aligned responses. To address this, we introduce Intention-Informed Auditory Scene Understanding (II-ASU) and present Auditory Attention-Driven LLM (AAD-LLM), a prototype system that integrates brain signals to infer listener attention. AAD-LLM extends an auditory LLM by incorporating intracranial electroencephalography (iEEG) recordings to decode which speaker a listener is attending to and refine responses accordingly. The model first predicts the attended speaker from neural activity, then conditions response generation on this inferred attentional state. We evaluate AAD-LLM on speaker description, speech transcription and extraction, and question answering in multitalker scenarios, with both objective and subjective ratings showing improved alignment with listener intention. By taking a first step toward intention-aware auditory AI, this work explores a new paradigm where listener perception informs machine listening, paving the way for future listener-centered auditory systems. Demo and code available: https://aad-llm.github.io.
>
---
#### [replaced 009] An introduction to pitch strength in contemporary popular music analysis and production
- **分类: cs.SD; eess.AS; 00A65; J.5**

- **链接: [http://arxiv.org/pdf/2506.07473v3](http://arxiv.org/pdf/2506.07473v3)**

> **作者:** Emmanuel Deruty
>
> **备注:** In Music 2024, Innovation in Music Conference, 14-16 June, 2024, Kristiania University College, Oslo, Norway
>
> **摘要:** Music information retrieval distinguishes between low- and high-level descriptions of music. Current generative AI models rely on text descriptions that are higher level than the controls familiar to studio musicians. Pitch strength, a low-level perceptual parameter of contemporary popular music, may be one feature that could make such AI models more suited to music production. Signal and perceptual analyses suggest that pitch strength (1) varies significantly across and inside songs; (2) contributes to both small- and large-scale structure; (3) contributes to the handling of polyphonic dissonance; and (4) may be a feature of upper harmonics made audible in a perspective of perceptual richness.
>
---
#### [replaced 010] Weakly Supervised Multiple Instance Learning for Whale Call Detection and Temporal Localization in Long-Duration Passive Acoustic Monitoring
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.20838v2](http://arxiv.org/pdf/2502.20838v2)**

> **作者:** Ragib Amin Nihal; Benjamin Yen; Runwu Shi; Kazuhiro Nakadai
>
> **摘要:** Marine ecosystem monitoring via Passive Acoustic Monitoring (PAM) generates vast data, but deep learning often requires precise annotations and short segments. We introduce DSMIL-LocNet, a Multiple Instance Learning framework for whale call detection and localization using only bag-level labels. Our dual-stream model processes 2-30 minute audio segments, leveraging spectral and temporal features with attention-based instance selection. Tests on Antarctic whale data show longer contexts improve classification (F1: 0.8-0.9) while medium instances ensure localization precision (0.65-0.70). This suggests MIL can enhance scalable marine monitoring. Code: https://github.com/Ragib-Amin-Nihal/DSMIL-Loc
>
---
#### [replaced 011] CASPER: A Large Scale Spontaneous Speech Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00267v3](http://arxiv.org/pdf/2506.00267v3)**

> **作者:** Cihan Xiao; Ruixing Liang; Xiangyu Zhang; Mehmet Emre Tiryaki; Veronica Bae; Lavanya Shankar; Rong Yang; Ethan Poon; Emmanuel Dupoux; Sanjeev Khudanpur; Leibny Paola Garcia Perera
>
> **摘要:** The success of large language models has driven interest in developing similar speech processing capabilities. However, a key challenge is the scarcity of high-quality spontaneous speech data, as most existing datasets contain scripted dialogues. To address this, we present a novel pipeline for eliciting and recording natural dialogues and release our dataset with 100+ hours of spontaneous speech. Our approach fosters fluid, natural conversations while encouraging a diverse range of topics and interactive exchanges. Unlike traditional methods, it facilitates genuine interactions, providing a reproducible framework for future data collection. This paper introduces our dataset and methodology, laying the groundwork for addressing the shortage of spontaneous speech data. We plan to expand this dataset in future stages, offering a growing resource for the research community.
>
---
#### [replaced 012] Speech Synthesis By Unrolling Diffusion Process using Neural Network Layers
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2309.09652v5](http://arxiv.org/pdf/2309.09652v5)**

> **作者:** Peter Ochieng
>
> **备注:** 10 pages
>
> **摘要:** This work introduces UDPNet, a novel architecture designed to accelerate the reverse diffusion process in speech synthesis. Unlike traditional diffusion models that rely on timestep embeddings and shared network parameters, UDPNet unrolls the reverse diffusion process directly into the network architecture, with successive layers corresponding to equally spaced steps in the diffusion schedule. Each layer progressively refines the noisy input, culminating in a high-fidelity estimation of the original data, \(x_0\). Additionally, we redefine the learning target by predicting latent variables instead of the conventional \(x_0\) or noise \(\epsilon_0\). This shift addresses the common issue of large prediction errors in early denoising stages, effectively reducing speech distortion. Extensive evaluations on single- and multi-speaker datasets demonstrate that UDPNet consistently outperforms state-of-the-art methods in both quality and efficiency, while generalizing effectively to unseen speech. These results position UDPNet as a robust solution for real-time speech synthesis applications. Sample audio is available at https://onexpeters.github.io/UDPNet.
>
---
#### [replaced 013] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00975v4](http://arxiv.org/pdf/2506.00975v4)**

> **作者:** Qichao Wang; Ziqiao Meng; Wenqian Cui; Yifei Zhang; Pengcheng Wu; Bingzhe Wu; Irwin King; Liang Chen; Peilin Zhao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Inspired by the impressive capabilities of GPT-4o, there is growing interest in enabling speech language models (SLMs) to engage in natural, fluid spoken interactions with humans. Recent advancements have led to the development of several SLMs that demonstrate promising results in this area. However, current approaches have yet to fully exploit dual-channel speech data, which inherently captures the structure and dynamics of human conversation. In this work, we systematically explore the use of dual-channel speech data in the context of modern large language models, and introduce a novel generative modeling paradigm, Next-Token-Pair Prediction (NTPP), to enable speaker-independent dual-channel spoken dialogue learning using decoder-only architectures for the first time. We evaluate our approach on standard benchmarks, and empirical results show that our proposed method, NTPP, significantly improves the conversational abilities of SLMs in terms of turn-taking prediction, response coherence, and naturalness. Moreover, compared to existing methods, NTPP achieves substantially lower inference latency, highlighting its practical efficiency for real-time applications.
>
---
