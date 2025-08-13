# 音频 cs.SD;  eess.SP

- **最新发布 17 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] QAMRO: Quality-aware Adaptive Margin Ranking Optimization for Human-aligned Assessment of Audio Generation Systems
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 论文提出QAMRO框架，通过融合多视角回归目标，解决音频生成系统主观评估中的相对性问题，提升与人类评估的一致性。**

- **链接: [http://arxiv.org/pdf/2508.08957v1](http://arxiv.org/pdf/2508.08957v1)**

> **作者:** Chien-Chun Wang; Kuan-Tang Huang; Cheng-Yeh Yang; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Evaluating audio generation systems, including text-to-music (TTM), text-to-speech (TTS), and text-to-audio (TTA), remains challenging due to the subjective and multi-dimensional nature of human perception. Existing methods treat mean opinion score (MOS) prediction as a regression problem, but standard regression losses overlook the relativity of perceptual judgments. To address this limitation, we introduce QAMRO, a novel Quality-aware Adaptive Margin Ranking Optimization framework that seamlessly integrates regression objectives from different perspectives, aiming to highlight perceptual differences and prioritize accurate ratings. Our framework leverages pre-trained audio-text models such as CLAP and Audiobox-Aesthetics, and is trained exclusively on the official AudioMOS Challenge 2025 dataset. It demonstrates superior alignment with human evaluations across all dimensions, significantly outperforming robust baseline models.
>
---
#### [new 002] Neutone SDK: An Open Source Framework for Neural Audio Processing
- **分类: cs.SD; cs.SE; eess.AS**

- **简介: 论文提出Neutone SDK开源框架，为PyTorch神经音频模型提供统一接口，解决DAWs集成难题，支持实时/离线处理与Python开发，应用于音频效果、音色转移等场景。**

- **链接: [http://arxiv.org/pdf/2508.09126v1](http://arxiv.org/pdf/2508.09126v1)**

> **作者:** Christopher Mitcheltree; Bogdan Teleaga; Andrew Fyfe; Naotake Masuda; Matthias Schäfer; Alfie Bradic; Nao Tokui
>
> **备注:** Accepted to AES International Conference on Artificial Intelligence and Machine Learning for Audio 2025
>
> **摘要:** Neural audio processing has unlocked novel methods of sound transformation and synthesis, yet integrating deep learning models into digital audio workstations (DAWs) remains challenging due to real-time / neural network inference constraints and the complexities of plugin development. In this paper, we introduce the Neutone SDK: an open source framework that streamlines the deployment of PyTorch-based neural audio models for both real-time and offline applications. By encapsulating common challenges such as variable buffer sizes, sample rate conversion, delay compensation, and control parameter handling within a unified, model-agnostic interface, our framework enables seamless interoperability between neural models and host plugins while allowing users to work entirely in Python. We provide a technical overview of the interfaces needed to accomplish this, as well as the corresponding SDK implementations. We also demonstrate the SDK's versatility across applications such as audio effect emulation, timbre transfer, and sample generation, as well as its adoption by researchers, educators, companies, and artists alike. The Neutone SDK is available at https://github.com/Neutone/neutone_sdk
>
---
#### [new 003] Sound Signal Synthesis with Auxiliary Classifier GAN, COVID-19 cough as an example
- **分类: cs.SD; cs.LG**

- **简介: 论文提出使用辅助分类GAN合成COVID-19咳嗽数据，提升分类器准确性，通过数据增强达到75%测试精度。**

- **链接: [http://arxiv.org/pdf/2508.08892v1](http://arxiv.org/pdf/2508.08892v1)**

> **作者:** Yahya Sherif Solayman Mohamed Saleh; Ahmed Mohammed Dabbous; Lama Alkhaled; Hum Yan Chai; Muhammad Ehsan Rana; Hamam Mokayed
>
> **摘要:** One of the fastest-growing domains in AI is healthcare. Given its importance, it has been the interest of many researchers to deploy ML models into the ever-demanding healthcare domain to aid doctors and increase accessibility. Delivering reliable models, however, demands a sizable amount of data, and the recent COVID-19 pandemic served as a reminder of the rampant and scary nature of healthcare that makes training models difficult. To alleviate such scarcity, many published works attempted to synthesize radiological cough data to train better COVID-19 detection models on the respective radiological data. To accommodate the time sensitivity expected during a pandemic, this work focuses on detecting COVID-19 through coughs using synthetic data to improve the accuracy of the classifier. The work begins by training a CNN on a balanced subset of the Coughvid dataset, establishing a baseline classification test accuracy of 72%. The paper demonstrates how an Auxiliary Classification GAN (ACGAN) may be trained to conditionally generate novel synthetic Mel Spectrograms of both healthy and COVID-19 coughs. These coughs are used to augment the training dataset of the CNN classifier, allowing it to reach a new test accuracy of 75%. The work highlights the expected messiness and inconsistency in training and offers insights into detecting and handling such shortcomings.
>
---
#### [new 004] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models
- **分类: cs.SD; eess.AS**

- **简介: 论文提出DualSpeechLM框架，通过双语音节建模解决语音理解与生成的模态差距及任务差异问题，提升统一模型性能。**

- **链接: [http://arxiv.org/pdf/2508.08961v1](http://arxiv.org/pdf/2508.08961v1)**

> **作者:** Yuanyuan Wang; Dongchao Yang; Yiwen Shao; Hangting Chen; Jiankun Zhao; Zhiyong Wu; Helen Meng; Xixin Wu
>
> **摘要:** Extending pre-trained Large Language Models (LLMs)'s speech understanding or generation abilities by introducing various effective speech tokens has attracted great attention in the speech community. However, building a unified speech understanding and generation model still faces the following challenges: (1) Due to the huge modality gap between speech tokens and text tokens, extending text LLMs to unified speech LLMs relies on large-scale paired data for fine-tuning, and (2) Generation and understanding tasks prefer information at different levels, e.g., generation benefits from detailed acoustic features, while understanding favors high-level semantics. This divergence leads to difficult performance optimization in one unified model. To solve these challenges, in this paper, we present two key insights in speech tokenization and speech language modeling. Specifically, we first propose an Understanding-driven Speech Tokenizer (USTokenizer), which extracts high-level semantic information essential for accomplishing understanding tasks using text LLMs. In this way, USToken enjoys better modality commonality with text, which reduces the difficulty of modality alignment in adapting text LLMs to speech LLMs. Secondly, we present DualSpeechLM, a dual-token modeling framework that concurrently models USToken as input and acoustic token as output within a unified, end-to-end framework, seamlessly integrating speech understanding and generation capabilities. Furthermore, we propose a novel semantic supervision loss and a Chain-of-Condition (CoC) strategy to stabilize model training and enhance speech generation performance. Experimental results demonstrate that our proposed approach effectively fosters a complementary relationship between understanding and generation tasks, highlighting the promising strategy of mutually enhancing both tasks in one unified model.
>
---
#### [new 005] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization
- **分类: cs.SD; cs.CL**

- **简介: 论文提出SSPO方法解决视频配音时长对齐问题，通过分段采样与细粒度损失优化，提升音频视频同步性能。**

- **链接: [http://arxiv.org/pdf/2508.08550v1](http://arxiv.org/pdf/2508.08550v1)**

> **作者:** Chaoqun Cui; Liangbin Huang; Shijing Wang; Zhe Tong; Zhaolong Huang; Xiao Zeng; Xiaofeng Liu
>
> **备注:** This paper is accepted by ACL2025 (Main)
>
> **摘要:** Video dubbing aims to translate original speech in visual media programs from the source language to the target language, relying on neural machine translation and text-to-speech technologies. Due to varying information densities across languages, target speech often mismatches the source speech duration, causing audio-video synchronization issues that significantly impact viewer experience. In this study, we approach duration alignment in LLM-based video dubbing machine translation as a preference optimization problem. We propose the Segment Supervised Preference Optimization (SSPO) method, which employs a segment-wise sampling strategy and fine-grained loss to mitigate duration mismatches between source and target lines. Experimental results demonstrate that SSPO achieves superior performance in duration alignment tasks.
>
---
#### [new 006] Opening Musical Creativity? Embedded Ideologies in Generative-AI Music Systems
- **分类: cs.SD; cs.AI; cs.HC**

- **简介: 论文分析四种生成式AI音乐系统中的意识形态，探讨其开发者如何营销民主化，发现共享的个人主义、全球主义等总意识形态，模糊责任并重塑音乐本质。**

- **链接: [http://arxiv.org/pdf/2508.08805v1](http://arxiv.org/pdf/2508.08805v1)**

> **作者:** Liam Pram; Fabio Morreale
>
> **备注:** Extended version of the presentation at The First International Conference in AI Music Studies 2024
>
> **摘要:** AI systems for music generation are increasingly common and easy to use, granting people without any musical background the ability to create music. Because of this, generative-AI has been marketed and celebrated as a means of democratizing music making. However, inclusivity often functions as marketable rhetoric rather than a genuine guiding principle in these industry settings. In this paper, we look at four generative-AI music making systems available to the public as of mid-2025 (AIVA, Stable Audio, Suno, and Udio) and track how they are rhetoricized by their developers, and received by users. Our aim is to investigate ideologies that are driving the early-stage development and adoption of generative-AI in music making, with a particular focus on democratization. A combination of autoethnography and digital ethnography is used to examine patterns and incongruities in rhetoric when positioned against product functionality. The results are then collated to develop a nuanced, contextual discussion. The shared ideology we map between producers and consumers is individualist, globalist, techno-liberal, and ethically evasive. It is a 'total ideology' which obfuscates individual responsibility, and through which the nature of music and musical practice is transfigured to suit generative outcomes.
>
---
#### [new 007] Revealing the Role of Audio Channels in ASR Performance Degradation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 论文研究音频通道对ASR性能的影响，提出规范化技术对齐模型特征以提升跨通道泛化能力，解决通道差异导致的性能下降问题。**

- **链接: [http://arxiv.org/pdf/2508.08967v1](http://arxiv.org/pdf/2508.08967v1)**

> **作者:** Kuan-Tang Huang; Li-Wei Chen; Hung-Shin Lee; Berlin Chen; Hsin-Min Wang
>
> **备注:** Accepted to IEEE ASRU 2025
>
> **摘要:** Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences.
>
---
#### [new 008] SonicRadiation: A Hybrid Numerical Solution for Sound Radiation without Ghost Cells
- **分类: cs.SD; cs.GR; cs.NA; math.NA**

- **简介: 论文提出一种混合数值方法，结合FDTD和TDBEM处理复杂边界，无需ghost cells，提升声辐射模拟精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.08775v1](http://arxiv.org/pdf/2508.08775v1)**

> **作者:** Xutong Jin; Guoping Wang; Sheng Li
>
> **备注:** 11 pages
>
> **摘要:** Interactive synthesis of physical sound effects is crucial in digital media production. Sound radiation simulation, a key component of physically based sound synthesis, has posed challenges in the context of complex object boundaries. Previous methods, such as ghost cell-based finite-difference time-domain (FDTD) wave solver, have struggled to address these challenges, leading to large errors and failures in complex boundaries because of the limitation of ghost cells. We present SonicRadiation, a hybrid numerical solution capable of handling complex and dynamic object boundaries in sound radiation simulation without relying on ghost cells. We derive a consistent formulation to connect the physical quantities on grid cells in FDTD with the boundary elements in the time-domain boundary element method (TDBEM). Hereby, we propose a boundary grid synchronization strategy to seamlessly integrate TDBEM with FDTD while maintaining high numerical accuracy. Our method holds both advantages from the accuracy of TDBEM for the near-field and the efficiency of FDTD for the far-field. Experimental results demonstrate the superiority of our method in sound radiation simulation over previous approaches in terms of accuracy and efficiency, particularly in complex scenes, further validating its effectiveness.
>
---
#### [new 009] Multi-Target Backdoor Attacks Against Speaker Recognition
- **分类: cs.SD; cs.LG**

- **简介: 论文提出多目标语音识别后门攻击方法，利用位置无关点击声作为触发器，同时攻击50名说话者，通过调整信噪比平衡隐蔽性与效果，并扩展至验证任务。**

- **链接: [http://arxiv.org/pdf/2508.08559v1](http://arxiv.org/pdf/2508.08559v1)**

> **作者:** Alexandrine Fortier; Sonal Joshi; Thomas Thebaud; Jesus Villalba Lopez; Najim Dehak; Patrick Cardinal
>
> **备注:** Accepted to IEEE Automatic Speech Recognition and Understanding Workshop 2025
>
> **摘要:** In this work, we propose a multi-target backdoor attack against speaker identification using position-independent clicking sounds as triggers. Unlike previous single-target approaches, our method targets up to 50 speakers simultaneously, achieving success rates of up to 95.04%. To simulate more realistic attack conditions, we vary the signal-to-noise ratio between speech and trigger, demonstrating a trade-off between stealth and effectiveness. We further extend the attack to the speaker verification task by selecting the most similar training speaker - based on cosine similarity - as the target. The attack is most effective when target and enrolled speaker pairs are highly similar, reaching success rates of up to 90% in such cases.
>
---
#### [new 010] Where is the Boundary: Multimodal Sensor Fusion Test Bench for Tissue Boundary Delineation
- **分类: eess.SP; cs.RO**

- **简介: 论文提出多模态传感器融合测试平台，用于解决机器人辅助神经外科中组织边界识别难题，通过整合视觉、力传感器等，提升材料分类准确率，并提供实时可视化及可扩展的硬件软件方案。**

- **链接: [http://arxiv.org/pdf/2508.08257v1](http://arxiv.org/pdf/2508.08257v1)**

> **作者:** Zacharias Chen; Alexa Cristelle Cahilig; Sarah Dias; Prithu Kolar; Ravi Prakash; Patrick J. Codd
>
> **备注:** 4 pages, 5 figures
>
> **摘要:** Robot-assisted neurological surgery is receiving growing interest due to the improved dexterity, precision, and control of surgical tools, which results in better patient outcomes. However, such systems often limit surgeons' natural sensory feedback, which is crucial in identifying tissues -- particularly in oncological procedures where distinguishing between healthy and tumorous tissue is vital. While imaging and force sensing have addressed the lack of sensory feedback, limited research has explored multimodal sensing options for accurate tissue boundary delineation. We present a user-friendly, modular test bench designed to evaluate and integrate complementary multimodal sensors for tissue identification. Our proposed system first uses vision-based guidance to estimate boundary locations with visual cues, which are then refined using data acquired by contact microphones and a force sensor. Real-time data acquisition and visualization are supported via an interactive graphical interface. Experimental results demonstrate that multimodal fusion significantly improves material classification accuracy. The platform provides a scalable hardware-software solution for exploring sensor fusion in surgical applications and demonstrates the potential of multimodal approaches in real-time tissue boundary delineation.
>
---
#### [new 011] Audio-Visual Speech Enhancement: Architectural Design and Deployment Strategies
- **分类: cs.SD; eess.SP**

- **简介: 论文提出基于CNN和LSTM的音频视觉语音增强系统，对比分析不同部署架构（云、边缘、设备）的性能，研究多模态融合下的延迟与质量权衡，揭示边缘架构在5G/Wi-Fi 6下满足实时需求的可行性，为应用场景选择提供指导。**

- **链接: [http://arxiv.org/pdf/2508.08468v1](http://arxiv.org/pdf/2508.08468v1)**

> **作者:** Anis Hamadouche; Haifeng Luo; Mathini Sellathurai; Tharm Ratnarajah
>
> **摘要:** This paper introduces a new AI-based Audio-Visual Speech Enhancement (AVSE) system and presents a comparative performance analysis of different deployment architectures. The proposed AVSE system employs convolutional neural networks (CNNs) for spectral feature extraction and long short-term memory (LSTM) networks for temporal modeling, enabling robust speech enhancement through multimodal fusion of audio and visual cues. Multiple deployment scenarios are investigated, including cloud-based, edge-assisted, and standalone device implementations. Their performance is evaluated in terms of speech quality improvement, latency, and computational overhead. Real-world experiments are conducted across various network conditions, including Ethernet, Wi-Fi, 4G, and 5G, to analyze the trade-offs between processing delay, communication latency, and perceptual speech quality. The results show that while cloud deployment achieves the highest enhancement quality, edge-assisted architectures offer the best balance between latency and intelligibility, meeting real-time requirements under 5G and Wi-Fi 6 conditions. These findings provide practical guidelines for selecting and optimizing AVSE deployment architectures in diverse applications, including assistive hearing devices, telepresence, and industrial communications.
>
---
#### [new 012] LPGNet: A Lightweight Network with Parallel Attention and Gated Fusion for Multimodal Emotion Recognition
- **分类: eess.AS; cs.SD**

- **简介: 论文提出LPGNet，针对多模态情感识别任务，解决Transformer模型计算成本高、依赖说话人信息的问题，通过并行注意力与门控融合实现高效特征学习，提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.08925v1](http://arxiv.org/pdf/2508.08925v1)**

> **作者:** Zhining He; Yang Xiao
>
> **备注:** Under peering review
>
> **摘要:** Emotion recognition in conversations (ERC) aims to predict the emotional state of each utterance by using multiple input types, such as text and audio. While Transformer-based models have shown strong performance in this task, they often face two major issues: high computational cost and heavy dependence on speaker information. These problems reduce their ability to generalize in real-world conversations. To solve these challenges, we propose LPGNet, a Lightweight network with Parallel attention and Gated fusion for multimodal ERC. The main part of LPGNet is the Lightweight Parallel Interaction Attention (LPIA) module. This module replaces traditional stacked Transformer layers with parallel dot-product attention, which can model both within-modality and between-modality relationships more efficiently. To improve emotional feature learning, LPGNet also uses a dual-gated fusion method. This method filters and combines features from different input types in a flexible and dynamic way. In addition, LPGNet removes speaker embeddings completely, which allows the model to work independently of speaker identity. Experiments on the IEMOCAP dataset show that LPGNet reaches over 87% accuracy and F1-score in 4-class emotion classification. It outperforms strong baseline models while using fewer parameters and showing better generalization across speakers.
>
---
#### [new 013] MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **简介: 论文提出MultiAiTutor，利用LLM架构生成儿童友好的多语言语音，针对新加坡口音 Mandarin、马来语、泰米尔等低资源语言，通过文化相关任务提升语言学习效果，实验验证其优于基线方法。**

- **链接: [http://arxiv.org/pdf/2508.08715v1](http://arxiv.org/pdf/2508.08715v1)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** 5 figures
>
> **摘要:** Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
>
---
#### [new 014] Transient Noise Removal via Diffusion-based Speech Inpainting
- **分类: eess.AS; cs.SD**

- **简介: 论文提出基于扩散的语音修复框架PGDI，解决瞬态噪声及长间隔修复难题，通过音素级分类器引导提升重建质量，适用于现实场景。**

- **链接: [http://arxiv.org/pdf/2508.08890v1](http://arxiv.org/pdf/2508.08890v1)**

> **作者:** Mordehay Moradi; Sharon Gannot
>
> **备注:** 23 pages, 3 figures, signal processing paper on speech inpainting
>
> **摘要:** In this paper, we present PGDI, a diffusion-based speech inpainting framework for restoring missing or severely corrupted speech segments. Unlike previous methods that struggle with speaker variability or long gap lengths, PGDI can accurately reconstruct gaps of up to one second in length while preserving speaker identity, prosody, and environmental factors such as reverberation. Central to this approach is classifier guidance, specifically phoneme-level guidance, which substantially improves reconstruction fidelity. PGDI operates in a speaker-independent manner and maintains robustness even when long segments are completely masked by strong transient noise, making it well-suited for real-world applications, such as fireworks, door slams, hammer strikes, and construction noise. Through extensive experiments across diverse speakers and gap lengths, we demonstrate PGDI's superior inpainting performance and its ability to handle challenging acoustic conditions. We consider both scenarios, with and without access to the transcript during inference, showing that while the availability of text further enhances performance, the model remains effective even in its absence. For audio samples, visit: https://mordehaym.github.io/PGDI/
>
---
#### [new 015] Listen through the Sound: Generative Speech Restoration Leveraging Acoustic Context Representation
- **分类: eess.AS; cs.SD**

- **简介: 论文提出基于声学上下文嵌入的语音恢复方法，通过ACX表示优化扩散模型UNIVERSE++，提升失真抑制效果。任务为语音信号恢复，解决失真处理难题，工作包括引入CLAP特征与ACX结构。**

- **链接: [http://arxiv.org/pdf/2508.08953v1](http://arxiv.org/pdf/2508.08953v1)**

> **作者:** Soo-Whan Chung; Min-Seok Choi
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** This paper introduces a novel approach to speech restoration by integrating a context-related conditioning strategy. Specifically, we employ the diffusion-based generative restoration model, UNIVERSE++, as a backbone to evaluate the effectiveness of contextual representations. We incorporate acoustic context embeddings extracted from the CLAP model, which capture the environmental attributes of input audio. Additionally, we propose an Acoustic Context (ACX) representation that refines CLAP embeddings to better handle various distortion factors and their intensity in speech signals. Unlike content-based approaches that rely on linguistic and speaker attributes, ACX provides contextual information that enables the restoration model to distinguish and mitigate distortions better. Experimental results indicate that context-aware conditioning improves both restoration performance and its stability across diverse distortion conditions, reducing variability compared to content-based methods.
>
---
#### [new 016] Exploring Disentangled Neural Speech Codecs from Self-Supervised Representations
- **分类: eess.AS; eess.SP**

- **简介: 论文提出基于自监督表示的解耦语音编码器，解决传统NACs无法分离语义与非语义信息的问题，通过结构化解耦提升语音转换等任务效果。**

- **链接: [http://arxiv.org/pdf/2508.08399v1](http://arxiv.org/pdf/2508.08399v1)**

> **作者:** Ryo Aihara; Yoshiki Masuyama; Gordon Wichern; François G. Germain; Jonathan Le Roux
>
> **摘要:** Neural audio codecs (NACs), which use neural networks to generate compact audio representations, have garnered interest for their applicability to many downstream tasks -- especially quantized codecs due to their compatibility with large language models. However, unlike text, speech conveys not only linguistic content but also rich paralinguistic features. Encoding these elements in an entangled fashion may be suboptimal, as it limits flexibility. For instance, voice conversion (VC) aims to convert speaker characteristics while preserving the original linguistic content, which requires a disentangled representation. Inspired by VC methods utilizing $k$-means quantization with self-supervised features to disentangle phonetic information, we develop a discrete NAC capable of structured disentanglement. Experimental evaluations show that our approach achieves reconstruction performance on par with conventional NACs that do not explicitly perform disentanglement, while also matching the effectiveness of conventional VC techniques.
>
---
#### [new 017] Preprocessing Algorithm Leveraging Geometric Modeling for Scale Correction in Hyperspectral Images for Improved Unmixing Performance
- **分类: eess.IV; cs.CV; eess.SP**

- **简介: 论文提出基于几何建模的预处理算法，针对高光谱图像尺度变化问题，通过修正大尺度乘法效应提升解混性能，误差减少近50%。**

- **链接: [http://arxiv.org/pdf/2508.08431v1](http://arxiv.org/pdf/2508.08431v1)**

> **作者:** Praveen Sumanasekara; Athulya Ratnayake; Buddhi Wijenayake; Keshawa Ratnayake; Roshan Godaliyadda; Parakrama Ekanayake; Vijitha Herath
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** Spectral variability significantly impacts the accuracy and convergence of hyperspectral unmixing algorithms. While many methods address complex spectral variability, large-scale variations in spectral signature scale caused by factors such as topography, illumination, and shadowing remain a major challenge. These variations often degrade unmixing performance and complicate model fitting. In this paper, we propose a novel preprocessing algorithm that corrects scale-induced spectral variability prior to unmixing. By isolating and compensating for these large-scale multiplicative effects, the algorithm provides a cleaner input, enabling unmixing methods to focus more effectively on modeling nonlinear spectral variability and abundance estimation. We present a rigorous mathematical framework to describe scale variability and extensive experimental validation of the proposed algorithm. Furthermore, the algorithm's impact is evaluated across a broad spectrum of state-of-the-art unmixing algorithms on two synthetic and two real hyperspectral datasets. The proposed preprocessing step consistently improves the performance of these algorithms, including those specifically designed to handle spectral variability, with error reductions close to 50% in many cases. This demonstrates that scale correction acts as a complementary step, facilitating more accurate unmixing by existing methods. The algorithm's generality and significant impact highlight its potential as a key component in practical hyperspectral unmixing pipelines. The implementation code will be made publicly available upon publication.
>
---
## 更新

#### [replaced 001] Dopamine Audiobook: A Training-free MLLM Agent for Emotional and Immersive Audiobook Generation
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.11002v2](http://arxiv.org/pdf/2504.11002v2)**

> **作者:** Yan Rong; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **摘要:** Audiobook generation aims to create rich, immersive listening experiences from multimodal inputs, but current approaches face three critical challenges: (1) the lack of synergistic generation of diverse audio types (e.g., speech, sound effects, and music) with precise temporal and semantic alignment; (2) the difficulty in conveying expressive, fine-grained emotions, which often results in machine-like vocal outputs; and (3) the absence of automated evaluation frameworks that align with human preferences for complex and diverse audio. To address these issues, we propose Dopamine Audiobook, a novel unified training-free multi-agent system, where a multimodal large language model (MLLM) serves two specialized roles (i.e., speech designer and audio designer) for emotional, human-like, and immersive audiobook generation and evaluation. Specifically, we firstly propose a flow-based, context-aware framework for diverse audio generation with word-level semantic and temporal alignment. To enhance expressiveness, we then design word-level paralinguistic augmentation, utterance-level prosody retrieval, and adaptive TTS model selection. Finally, for evaluation, we introduce a novel MLLM-based evaluation framework incorporating self-critique, perspective-taking, and psychological MagicEmo prompts to ensure human-aligned and self-aligned assessments. Experimental results demonstrate that our method achieves state-of-the-art (SOTA) performance on multiple metrics. Importantly, our evaluation framework shows better alignment with human preferences and transferability across audio tasks.
>
---
#### [replaced 002] 3DFacePolicy: Audio-Driven 3D Facial Animation Based on Action Control
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.10848v2](http://arxiv.org/pdf/2409.10848v2)**

> **作者:** Xuanmeng Sha; Liyun Zhang; Tomohiro Mashita; Naoya Chiba; Yuki Uranishi
>
> **摘要:** Audio-driven 3D facial animation has achieved significant progress in both research and applications. While recent baselines struggle to generate natural and continuous facial movements due to their frame-by-frame vertex generation approach, we propose 3DFacePolicy, a pioneer work that introduces a novel definition of vertex trajectory changes across consecutive frames through the concept of "action". By predicting action sequences for each vertex that encode frame-to-frame movements, we reformulate vertex generation approach into an action-based control paradigm. Specifically, we leverage a robotic control mechanism, diffusion policy, to predict action sequences conditioned on both audio and vertex states. Extensive experiments on VOCASET and BIWI datasets demonstrate that our approach significantly outperforms state-of-the-art methods and is particularly expert in dynamic, expressive and naturally smooth facial animations.
>
---
#### [replaced 003] Learning Marmoset Vocal Patterns with a Masked Autoencoder for Robust Call Segmentation, Classification, and Caller Identification
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.23279v4](http://arxiv.org/pdf/2410.23279v4)**

> **作者:** Bin Wu; Shinnosuke Takamichi; Sakriani Sakti; Satoshi Nakamura
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** The marmoset, a highly vocal primate, is a key model for studying social-communicative behavior. Unlike human speech, marmoset vocalizations are less structured, highly variable, and recorded in noisy, low-resource conditions. Learning marmoset communication requires joint call segmentation, classification, and caller identification -- challenging domain tasks. Previous CNNs handle local patterns but struggle with long-range temporal structure. We applied Transformers using self-attention for global dependencies. However, Transformers show overfitting and instability on small, noisy annotated datasets. To address this, we pretrain Transformers with MAE -- a self-supervised method reconstructing masked segments from hundreds of hours of unannotated marmoset recordings. The pretraining improved stability and generalization. Results show MAE-pretrained Transformers outperform CNNs, demonstrating modern self-supervised architectures effectively model low-resource non-human vocal communication.
>
---
#### [replaced 004] Audio-Thinker: Guiding Audio Language Model When and How to Think via Reinforcement Learning
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08039v2](http://arxiv.org/pdf/2508.08039v2)**

> **作者:** Shu Wu; Chenxing Li; Wenfu Wang; Hao Zhang; Hualei Wang; Meng Yu; Dong Yu
>
> **备注:** preprint
>
> **摘要:** Recent advancements in large language models, multimodal large language models, and large audio language models (LALMs) have significantly improved their reasoning capabilities through reinforcement learning with rule-based rewards. However, the explicit reasoning process has yet to show significant benefits for audio question answering, and effectively leveraging deep reasoning remains an open challenge, with LALMs still falling short of human-level auditory-language reasoning. To address these limitations, we propose Audio-Thinker, a reinforcement learning framework designed to enhance the reasoning capabilities of LALMs, with a focus on improving adaptability, consistency, and effectiveness. Our approach introduces an adaptive think accuracy reward, enabling the model to adjust its reasoning strategies based on task complexity dynamically. Furthermore, we incorporate an external reward model to evaluate the overall consistency and quality of the reasoning process, complemented by think-based rewards that help the model distinguish between valid and flawed reasoning paths during training. Experimental results demonstrate that our Audio-Thinker model outperforms existing reasoning-oriented LALMs across various benchmark tasks, exhibiting superior reasoning and generalization capabilities.
>
---
#### [replaced 005] Gotta Hear Them All: Towards Sound Source Aware Audio Generation
- **分类: cs.MM; cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.15447v4](http://arxiv.org/pdf/2411.15447v4)**

> **作者:** Wei Guo; Heng Wang; Jianbo Ma; Weidong Cai
>
> **备注:** 17 pages, 12 figures, source code available at https://github.com/wguo86/SSV2A
>
> **摘要:** Audio synthesis has broad applications in multimedia. Recent advancements have made it possible to generate relevant audios from inputs describing an audio scene, such as images or texts. However, the immersiveness and expressiveness of the generation are limited. One possible problem is that existing methods solely rely on the global scene and overlook details of local sounding objects (i.e., sound sources). To address this issue, we propose a Sound Source-Aware Audio (SS2A) generator. SS2A is able to locally perceive multimodal sound sources from a scene with visual detection and cross-modality translation. It then contrastively learns a Cross-Modal Sound Source (CMSS) Manifold to semantically disambiguate each source. Finally, we attentively mix their CMSS semantics into a rich audio representation, from which a pretrained audio generator outputs the sound. To model the CMSS manifold, we curate a novel single-sound-source visual-audio dataset VGGS3 from VGGSound. We also design a Sound Source Matching Score to clearly measure localized audio relevance. With the effectiveness of explicit sound source modeling, SS2A achieves state-of-the-art performance in extensive image-to-audio tasks. We also qualitatively demonstrate SS2A's ability to achieve intuitive synthesis control by compositing vision, text, and audio conditions. Furthermore, we show that our sound source modeling can achieve competitive video-to-audio performance with a straightforward temporal aggregation mechanism.
>
---
#### [replaced 006] Task-Oriented Feature Compression for Multimodal Understanding via Device-Edge Co-Inference
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12926v2](http://arxiv.org/pdf/2503.12926v2)**

> **作者:** Cheng Yuan; Zhening Liu; Jiashu Lv; Jiawei Shao; Yufei Jiang; Jun Zhang; Xuelong Li
>
> **摘要:** With the rapid development of large multimodal models (LMMs), multimodal understanding applications are emerging. As most LMM inference requests originate from edge devices with limited computational capabilities, the predominant inference pipeline involves directly forwarding the input data to an edge server which handles all computations. However, this approach introduces high transmission latency due to limited uplink bandwidth of edge devices and significant computation latency caused by the prohibitive number of visual tokens, thus hindering delay-sensitive tasks and degrading user experience. To address this challenge, we propose a task-oriented feature compression (TOFC) method for multimodal understanding in a device-edge co-inference framework, where visual features are merged by clustering and encoded by a learnable and selective entropy model before feature projection. Specifically, we employ density peaks clustering based on K nearest neighbors to reduce the number of visual features, thereby minimizing both data transmission and computational complexity. Subsequently, a learnable entropy model with hyperprior is utilized to encode and decode merged features, further reducing transmission overhead. To enhance compression efficiency, multiple entropy models are adaptively selected based on the characteristics of the visual features, enabling a more accurate estimation of the probability distribution. Comprehensive experiments on seven visual question answering benchmarks validate the effectiveness of the proposed TOFC method. Results show that TOFC achieves up to 52% reduction in data transmission overhead and 63% reduction in system latency while maintaining identical task performance, compared with neural compression ELIC.
>
---
#### [replaced 007] Marco-Voice Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.02038v3](http://arxiv.org/pdf/2508.02038v3)**

> **作者:** Fengping Tian; Chenyang Lyu; Xuanfan Ni; Haoqin Sun; Qingjuan Li; Zhiqiang Qian; Haijun Li; Longyue Wang; Zhao Xu; Weihua Luo; Kaifu Zhang
>
> **备注:** Technical Report. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively
>
> **摘要:** This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis. Our code and dataset are publicly available at https://github.com/AIDC-AI/Marco-Voice and https://huggingface.co/datasets/AIDC-AI/CSEMOTIONS respectively.
>
---
#### [replaced 008] TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated Phrase-Boosting Tree
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2508.07014v2](http://arxiv.org/pdf/2508.07014v2)**

> **作者:** Andrei Andrusenko; Vladimir Bataev; Lilit Grigoryan; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Recognizing specific key phrases is an essential task for contextualized Automatic Speech Recognition (ASR). However, most existing context-biasing approaches have limitations associated with the necessity of additional model training, significantly slow down the decoding process, or constrain the choice of the ASR system type. This paper proposes a universal ASR context-biasing framework that supports all major types: CTC, Transducers, and Attention Encoder-Decoder models. The framework is based on a GPU-accelerated word boosting tree, which enables it to be used in shallow fusion mode for greedy and beam search decoding without noticeable speed degradation, even with a vast number of key phrases (up to 20K items). The obtained results showed high efficiency of the proposed method, surpassing the considered open-source context-biasing approaches in accuracy and decoding speed. Our context-biasing framework is open-sourced as a part of the NeMo toolkit.
>
---
