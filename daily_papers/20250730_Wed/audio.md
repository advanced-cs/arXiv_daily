# 音频 cs.SD;  eess.SP

- **最新发布 10 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Whilter: A Whisper-based Data Filter for "In-the-Wild" Speech Corpora Using Utterance-level Multi-Task Classification
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于语音数据过滤任务，旨在解决“in-the-wild”语音语料中存在的多说话人、非目标语言和音乐等干扰问题。作者提出了Whilter模型，基于Whisper编码器和注意力分类器进行多任务分类，实现了高效过滤，并在多个指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.21642v1](http://arxiv.org/pdf/2507.21642v1)**

> **作者:** William Ravenscroft; George Close; Kit Bower-Morris; Jamie Stacey; Dmitry Sityaev; Kris Y. Hong
>
> **备注:** Accepted for Interspeech 2025
>
> **摘要:** Large-scale in-the-wild speech datasets have become more prevalent in recent years due to increased interest in models that can learn useful features from unlabelled data for tasks such as speech recognition or synthesis. These datasets often contain undesirable features, such as multiple speakers, non-target languages, and music, which may impact model learning. The Whilter model is proposed as a multitask solution to identify these undesirable samples. Whilter uses a Whisper encoder with an attention-based classifier to solve five diverse classification problems at once. In addition, an annotated dataset is published for a subset of two popular in-the-wild corpora. Whilter achieves F1 scores above 85% and equal error rates of 6.5% to 7.8% for three of five subtasks, outperforming a state-of-the-art BEATs classifier on speech-specific classes, with a notable decrease in processing time compared to a combination of single-task alternatives.
>
---
#### [new 002] Hyperbolic Embeddings for Order-Aware Classification of Audio Effect Chains
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频信号处理任务，旨在解决从湿音频信号中联合估计音频效果类型及其顺序的问题。现有方法多仅关注效果类型估计，忽视顺序影响。作者提出一种基于超平面嵌入的神经网络方法，利用超平面空间的指数扩展特性，对效果链的树状结构进行建模，从而更准确地识别效果类型及顺序。实验表明该方法优于传统欧几里得空间方法。**

- **链接: [http://arxiv.org/pdf/2507.20624v1](http://arxiv.org/pdf/2507.20624v1)**

> **作者:** Aogu Wada; Tomohiko Nakamura; Hiroshi Saruwatari
>
> **备注:** 7 pages, 3 figures, accepted for the 28th International Conference on Digital Audio Effects (DAFx25)
>
> **摘要:** Audio effects (AFXs) are essential tools in music production, frequently applied in chains to shape timbre and dynamics. The order of AFXs in a chain plays a crucial role in determining the final sound, particularly when non-linear (e.g., distortion) or time-variant (e.g., chorus) processors are involved. Despite its importance, most AFX-related studies have primarily focused on estimating effect types and their parameters from a wet signal. To address this gap, we formulate AFX chain recognition as the task of jointly estimating AFX types and their order from a wet signal. We propose a neural-network-based method that embeds wet signals into a hyperbolic space and classifies their AFX chains. Hyperbolic space can represent tree-structured data more efficiently than Euclidean space due to its exponential expansion property. Since AFX chains can be represented as trees, with AFXs as nodes and edges encoding effect order, hyperbolic space is well-suited for modeling the exponentially growing and non-commutative nature of ordered AFX combinations, where changes in effect order can result in different final sounds. Experiments using guitar sounds demonstrate that, with an appropriate curvature, the proposed method outperforms its Euclidean counterpart. Further analysis based on AFX type and chain length highlights the effectiveness of the proposed method in capturing AFX order.
>
---
#### [new 003] Combolutional Neural Networks
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音频信息检索任务，旨在解决传统模型处理音频时计算复杂、特征提取不足的问题。作者提出了一种新的“combolutional层”，作为卷积层的替代，能高效提取音频中的谐波特征，适用于需要精确谐波分析的音频任务。**

- **链接: [http://arxiv.org/pdf/2507.21202v1](http://arxiv.org/pdf/2507.21202v1)**

> **作者:** Cameron Churchwell; Minje Kim; Paris Smaragdis
>
> **备注:** 4 pages, 3 figures, accepted to WASPAA 2025
>
> **摘要:** Selecting appropriate inductive biases is an essential step in the design of machine learning models, especially when working with audio, where even short clips may contain millions of samples. To this end, we propose the combolutional layer: a learned-delay IIR comb filter and fused envelope detector, which extracts harmonic features in the time domain. We demonstrate the efficacy of the combolutional layer on three information retrieval tasks, evaluate its computational cost relative to other audio frontends, and provide efficient implementations for training. We find that the combolutional layer is an effective replacement for convolutional layers in audio tasks where precise harmonic analysis is important, e.g., piano transcription, speaker classification, and key detection. Additionally, the combolutional layer has several other key benefits over existing frontends, namely: low parameter count, efficient CPU inference, strictly real-valued computations, and improved interpretability.
>
---
#### [new 004] Relationship between objective and subjective perceptual measures of speech in individuals with head and neck cancer
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于临床语音评估任务，旨在探究头颈癌患者语音的主客观评价关系。研究分析了大量头颈癌患者的语音数据，通过训练有素的听者进行主观评分，并与客观声学测量结果对比。结果显示主观的可懂度、发音和声音质量高度相关，且部分客观测量与主观评价一致，表明单一可懂度指标可能足以用于临床监测。**

- **链接: [http://arxiv.org/pdf/2507.21426v1](http://arxiv.org/pdf/2507.21426v1)**

> **作者:** Bence Mark Halpern; Thomas Tienkamp; Teja Rebernik; Rob J. J. H. van Son; Martijn Wieling; Defne Abur; Tomoki Toda
>
> **备注:** 5 pages, 1 figure, 1 table. Accepted at Interspeech 2025
>
> **摘要:** Meaningful speech assessment is vital in clinical phonetics and therapy monitoring. This study examined the link between perceptual speech assessments and objective acoustic measures in a large head and neck cancer (HNC) dataset. Trained listeners provided ratings of intelligibility, articulation, voice quality, phonation, speech rate, nasality, and background noise on speech. Strong correlations were found between subjective intelligibility, articulation, and voice quality, likely due to a shared underlying cause of speech symptoms in our speaker population. Objective measures of intelligibility and speech rate aligned with their subjective counterpart. Our results suggest that a single intelligibility measure may be sufficient for the clinical monitoring of speakers treated for HNC using concomitant chemoradiation.
>
---
#### [new 005] SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决现有数据集规模小、多样性不足的问题。作者构建了大规模多语言数据集SpeechFake，包含300万伪造语音样本，覆盖46种语言和40种生成工具，并训练检测模型，验证了其在多种场景下的优越性能。**

- **链接: [http://arxiv.org/pdf/2507.21463v1](http://arxiv.org/pdf/2507.21463v1)**

> **作者:** Wen Huang; Yanmei Gu; Zhiming Wang; Huijia Zhu; Yanmin Qian
>
> **备注:** Published in ACL 2025. Dataset available at: https://github.com/YMLLG/SpeechFake
>
> **摘要:** As speech generation technology advances, the risk of misuse through deepfake audio has become a pressing concern, which underscores the critical need for robust detection systems. However, many existing speech deepfake datasets are limited in scale and diversity, making it challenging to train models that can generalize well to unseen deepfakes. To address these gaps, we introduce SpeechFake, a large-scale dataset designed specifically for speech deepfake detection. SpeechFake includes over 3 million deepfake samples, totaling more than 3,000 hours of audio, generated using 40 different speech synthesis tools. The dataset encompasses a wide range of generation techniques, including text-to-speech, voice conversion, and neural vocoder, incorporating the latest cutting-edge methods. It also provides multilingual support, spanning 46 languages. In this paper, we offer a detailed overview of the dataset's creation, composition, and statistics. We also present baseline results by training detection models on SpeechFake, demonstrating strong performance on both its own test sets and various unseen test sets. Additionally, we conduct experiments to rigorously explore how generation methods, language diversity, and speaker variation affect detection performance. We believe SpeechFake will be a valuable resource for advancing speech deepfake detection and developing more robust models for evolving generation techniques.
>
---
#### [new 006] Hierarchical Graph Neural Network for Compressed Speech Steganalysis
- **分类: cs.CR; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音隐写分析任务，旨在解决深度学习方法在VoIP语音流隐写检测中计算复杂、泛化能力差的问题。论文首次引入图神经网络GraphSAGE，通过构建语音数据的图结构，捕获多层次隐写特征，提升了检测精度与效率，尤其在短样本、低嵌入率场景下表现优越。**

- **链接: [http://arxiv.org/pdf/2507.21591v1](http://arxiv.org/pdf/2507.21591v1)**

> **作者:** Mustapha Hemis; Hamza Kheddar; Mohamed Chahine Ghanem; Bachir Boudraa
>
> **摘要:** Steganalysis methods based on deep learning (DL) often struggle with computational complexity and challenges in generalizing across different datasets. Incorporating a graph neural network (GNN) into steganalysis schemes enables the leveraging of relational data for improved detection accuracy and adaptability. This paper presents the first application of a Graph Neural Network (GNN), specifically the GraphSAGE architecture, for steganalysis of compressed voice over IP (VoIP) speech streams. The method involves straightforward graph construction from VoIP streams and employs GraphSAGE to capture hierarchical steganalysis information, including both fine grained details and high level patterns, thereby achieving high detection accuracy. Experimental results demonstrate that the developed approach performs well in uncovering quantization index modulation (QIM)-based steganographic patterns in VoIP signals. It achieves detection accuracy exceeding 98 percent even for short 0.5 second samples, and 95.17 percent accuracy under challenging conditions with low embedding rates, representing an improvement of 2.8 percent over the best performing state of the art methods. Furthermore, the model exhibits superior efficiency, with an average detection time as low as 0.016 seconds for 0.5-second samples an improvement of 0.003 seconds. This makes it efficient for online steganalysis tasks, providing a superior balance between detection accuracy and efficiency under the constraint of short samples with low embedding rates.
>
---
#### [new 007] Sync-TVA: A Graph-Attention Framework for Multimodal Emotion Recognition with Cross-Modal Fusion
- **分类: cs.MM; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多模态情感识别任务，旨在解决跨模态交互不足和模态贡献不平衡问题。作者提出Sync-TVA框架，结合图注意力机制与动态增强模块，构建跨模态图并融合多模态信息。实验表明其在MELD和IEMOCAP数据集上性能优于现有模型，尤其在类别不平衡条件下表现更优。**

- **链接: [http://arxiv.org/pdf/2507.21395v1](http://arxiv.org/pdf/2507.21395v1)**

> **作者:** Zeyu Deng; Yanhui Lu; Jiashu Liao; Shuang Wu; Chongfeng Wei
>
> **摘要:** Multimodal emotion recognition (MER) is crucial for enabling emotionally intelligent systems that perceive and respond to human emotions. However, existing methods suffer from limited cross-modal interaction and imbalanced contributions across modalities. To address these issues, we propose Sync-TVA, an end-to-end graph-attention framework featuring modality-specific dynamic enhancement and structured cross-modal fusion. Our design incorporates a dynamic enhancement module for each modality and constructs heterogeneous cross-modal graphs to model semantic relations across text, audio, and visual features. A cross-attention fusion mechanism further aligns multimodal cues for robust emotion inference. Experiments on MELD and IEMOCAP demonstrate consistent improvements over state-of-the-art models in both accuracy and weighted F1 score, especially under class-imbalanced conditions.
>
---
#### [new 008] TTS-1 Technical Report
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升文本到语音的生成质量与效率。论文提出了两个Transformer模型TTS-1和TTS-1-Max，分别面向高效实时与高质量场景，通过预训练、微调与强化学习优化模型，实现多语言、低延迟、高分辨率语音生成，并开源代码。**

- **链接: [http://arxiv.org/pdf/2507.21138v1](http://arxiv.org/pdf/2507.21138v1)**

> **作者:** Oleg Atamanenko; Anna Chalova; Joseph Coombes; Nikki Cope; Phillip Dang; Zhifeng Deng; Jimmy Du; Michael Ermolenko; Feifan Fan; Yufei Feng; Cheryl Fichter; Pavel Filimonov; Louis Fischer; Kylan Gibbs; Valeria Gusarova; Pavel Karpik; Andreas Assad Kottner; Ian Lee; Oliver Louie; Jasmine Mai; Mikhail Mamontov; Suri Mao; Nurullah Morshed; Igor Poletaev; Florin Radu; Dmytro Semernia; Evgenii Shingarev; Vikram Sivaraja; Peter Skirko; Rinat Takhautdinov; Robert Villahermosa; Jean Wang
>
> **备注:** 20 pages, 10 figures. For associated modeling and training code, see https://github.com/inworld-ai/tts
>
> **摘要:** We introduce Inworld TTS-1, a set of two Transformer-based autoregressive text-to-speech (TTS) models. Our largest model, TTS-1-Max, has 8.8B parameters and is designed for utmost quality and expressiveness in demanding applications. TTS-1 is our most efficient model, with 1.6B parameters, built for real-time speech synthesis and on-device use cases. By scaling train-time compute and applying a sequential process of pre-training, fine-tuning, and RL-alignment of the speech-language model (SpeechLM) component, both models achieve state-of-the-art performance on a variety of benchmarks, demonstrating exceptional quality relying purely on in-context learning of the speaker's voice. Inworld TTS-1 and TTS-1-Max can generate high-resolution 48 kHz speech with low latency, and support 11 languages with fine-grained emotional control and non-verbal vocalizations through audio markups. We additionally open-source our training and modeling code under an MIT license.
>
---
#### [new 009] Model-free Speculative Decoding for Transformer-based ASR with Token Map Drafting
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决Transformer模型解码计算量大、部署受限的问题。论文提出无需额外模型的“Token Map Drafting”方法，利用预计算的n-gram token map进行推测解码，提升解码速度，实验证明在低复杂度领域有效且不损失准确率。**

- **链接: [http://arxiv.org/pdf/2507.21522v1](http://arxiv.org/pdf/2507.21522v1)**

> **作者:** Tuan Vu Ho; Hiroaki Kokubo; Masaaki Yamamoto; Yohei Kawaguchi
>
> **备注:** Accepted at EUSIPCO 2025
>
> **摘要:** End-to-end automatic speech recognition (ASR) systems based on transformer architectures, such as Whisper, offer high transcription accuracy and robustness. However, their autoregressive decoding is computationally expensive, hence limiting deployment on CPU-based and resource-constrained devices. Speculative decoding (SD) mitigates this issue by using a smaller draft model to propose candidate tokens, which are then verified by the main model. However, this approach is impractical for devices lacking hardware accelerators like GPUs. To address this, we propose \emph{Token Map Drafting}, a model-free SD technique that eliminates the need for a separate draft model. Instead, we leverage a precomputed n-gram token map derived from domain-specific training data, enabling efficient speculative decoding with minimal overhead. Our method significantly accelerates ASR inference in structured, low-perplexity domains without sacrificing transcription accuracy. Experimental results demonstrate decoding speed-ups of $1.27\times$ on the CI-AVSR dataset and $1.37\times$ on our internal dataset without degrading recognition accuracy. Additionally, our approach achieves a $10\%$ absolute improvement in decoding speed over the Distill-spec baseline running on CPU, highlighting its effectiveness for on-device ASR applications.
>
---
#### [new 010] A Deep Learning Automatic Speech Recognition Model for Shona Language
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决低资源语言绍纳语因数据稀缺和声调复杂导致的识别准确率低问题。研究采用卷积神经网络与长短期记忆网络混合架构，并运用数据增强、迁移学习和注意力机制，最终实现74%准确率，显著提升识别效果。**

- **链接: [http://arxiv.org/pdf/2507.21331v1](http://arxiv.org/pdf/2507.21331v1)**

> **作者:** Leslie Wellington Sirora; Mainford Mutandavari
>
> **摘要:** This study presented the development of a deep learning-based Automatic Speech Recognition system for Shona, a low-resource language characterized by unique tonal and grammatical complexities. The research aimed to address the challenges posed by limited training data, lack of labelled data, and the intricate tonal nuances present in Shona speech, with the objective of achieving significant improvements in recognition accuracy compared to traditional statistical models. The research first explored the feasibility of using deep learning to develop an accurate ASR system for Shona. Second, it investigated the specific challenges involved in designing and implementing deep learning architectures for Shona speech recognition and proposed strategies to mitigate these challenges. Lastly, it compared the performance of the deep learning-based model with existing statistical models in terms of accuracy. The developed ASR system utilized a hybrid architecture consisting of a Convolutional Neural Network for acoustic modelling and a Long Short-Term Memory network for language modelling. To overcome the scarcity of data, data augmentation techniques and transfer learning were employed. Attention mechanisms were also incorporated to accommodate the tonal nature of Shona speech. The resulting ASR system achieved impressive results, with a Word Error Rate of 29%, Phoneme Error Rate of 12%, and an overall accuracy of 74%. These metrics indicated the potential of deep learning to enhance ASR accuracy for under-resourced languages like Shona. This study contributed to the advancement of ASR technology for under-resourced languages like Shona, ultimately fostering improved accessibility and communication for Shona speakers worldwide.
>
---
## 更新

#### [replaced 001] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.10082v2](http://arxiv.org/pdf/2507.10082v2)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [replaced 002] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation
- **分类: cs.SD; cs.AI; cs.CV; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.05130v3](http://arxiv.org/pdf/2502.05130v3)**

> **作者:** Yusheng Dai; Chenxi Wang; Chang Li; Chen Wang; Jun Du; Kewei Li; Ruoyu Wang; Jiefeng Ma; Lei Sun; Jianqing Gao
>
> **摘要:** This paper introduces Swap Forward (SaFa), a modality-agnostic and efficient method to generate seamless and coherence long spectrum and panorama through latent swap joint diffusion across multi-views. We first investigate the spectrum aliasing problem in spectrum-based audio generation caused by existing joint diffusion methods. Through a comparative analysis of the VAE latent representation of Mel-spectra and RGB images, we identify that the failure arises from excessive suppression of high-frequency components during the spectrum denoising process due to the averaging operator. To address this issue, we propose Self-Loop Latent Swap, a frame-level bidirectional swap applied to the overlapping region of adjacent views. Leveraging stepwise differentiated trajectories of adjacent subviews, this swap operator adaptively enhances high-frequency components and avoid spectrum distortion. Furthermore, to improve global cross-view consistency in non-overlapping regions, we introduce Reference-Guided Latent Swap, a unidirectional latent swap operator that provides a centralized reference trajectory to synchronize subview diffusions. By refining swap timing and intervals, we can achieve a cross-view similarity-diversity balance in a forward-only manner. Quantitative and qualitative experiments demonstrate that SaFa significantly outperforms existing joint diffusion methods and even training-based methods in audio generation using both U-Net and DiT models, along with effective longer length adaptation. It also adapts well to panorama generation, achieving comparable performance with 2 $\sim$ 20 $\times$ faster speed and greater model generalizability. More generation demos are available at https://swapforward.github.io/
>
---
#### [replaced 003] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling
- **分类: cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.14915v3](http://arxiv.org/pdf/2507.14915v3)**

> **作者:** Xiaojie Li; Ronghui Li; Shukai Fang; Shuzhao Xie; Xiaoyang Guo; Jiaqing Zhou; Junkun Peng; Zhi Wang
>
> **摘要:** Well-coordinated, music-aligned holistic dance enhances emotional expressiveness and audience engagement. However, generating such dances remains challenging due to the scarcity of holistic 3D dance datasets, the difficulty of achieving cross-modal alignment between music and dance, and the complexity of modeling interdependent motion across the body, hands, and face. To address these challenges, we introduce SoulDance, a high-precision music-dance paired dataset captured via professional motion capture systems, featuring meticulously annotated holistic dance movements. Building on this dataset, we propose SoulNet, a framework designed to generate music-aligned, kinematically coordinated holistic dance sequences. SoulNet consists of three principal components: (1) Hierarchical Residual Vector Quantization, which models complex, fine-grained motion dependencies across the body, hands, and face; (2) Music-Aligned Generative Model, which composes these hierarchical motion units into expressive and coordinated holistic dance; (3) Music-Motion Retrieval Module, a pre-trained cross-modal model that functions as a music-dance alignment prior, ensuring temporal synchronization and semantic coherence between generated dance and input music throughout the generation process. Extensive experiments demonstrate that SoulNet significantly surpasses existing approaches in generating high-quality, music-coordinated, and well-aligned holistic 3D dance sequences.
>
---
#### [replaced 004] Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.08128v2](http://arxiv.org/pdf/2507.08128v2)**

> **作者:** Arushi Goel; Sreyan Ghosh; Jaehyeon Kim; Sonal Kumar; Zhifeng Kong; Sang-gil Lee; Chao-Han Huck Yang; Ramani Duraiswami; Dinesh Manocha; Rafael Valle; Bryan Catanzaro
>
> **备注:** Code, Datasets, and Models: https://research.nvidia.com/labs/adlr/AF3/ ; Updates in v2: Updated results for new thinking mode ckpts, added qualitative figure, added note on fully open claim, add email ID for corresponding authors
>
> **摘要:** We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
>
---
#### [replaced 005] LLAMAPIE: Proactive In-Ear Conversation Assistants
- **分类: cs.LG; cs.CL; cs.HC; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04066v2](http://arxiv.org/pdf/2505.04066v2)**

> **作者:** Tuochao Chen; Nicholas Batchelder; Alisa Liu; Noah Smith; Shyamnath Gollakota
>
> **备注:** Published by ACL2025 (Findings)
>
> **摘要:** We introduce LlamaPIE, the first real-time proactive assistant designed to enhance human conversations through discreet, concise guidance delivered via hearable devices. Unlike traditional language models that require explicit user invocation, this assistant operates in the background, anticipating user needs without interrupting conversations. We address several challenges, including determining when to respond, crafting concise responses that enhance conversations, leveraging knowledge of the user for context-aware assistance, and real-time, on-device processing. To achieve this, we construct a semi-synthetic dialogue dataset and propose a two-model pipeline: a small model that decides when to respond and a larger model that generates the response. We evaluate our approach on real-world datasets, demonstrating its effectiveness in providing helpful, unobtrusive assistance. User studies with our assistant, implemented on Apple Silicon M2 hardware, show a strong preference for the proactive assistant over both a baseline with no assistance and a reactive model, highlighting the potential of LlamaPie to enhance live conversations.
>
---
#### [replaced 006] EEG-CLIP : Learning EEG representations from natural language descriptions
- **分类: cs.CL; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.16531v2](http://arxiv.org/pdf/2503.16531v2)**

> **作者:** Tidiane Camaret Ndir; Robin Tibor Schirrmeister; Tonio Ball
>
> **摘要:** Deep networks for electroencephalogram (EEG) decoding are often only trained to solve one specific task, such as pathology or age decoding. A more general task-agnostic approach is to train deep networks to match a (clinical) EEG recording to its corresponding textual medical report and vice versa. This approach was pioneered in the computer vision domain matching images and their text captions and subsequently allowed to do successful zero-shot decoding using textual class prompts. In this work, we follow this approach and develop a contrastive learning framework, EEG-CLIP, that aligns the EEG time series and the descriptions of the corresponding clinical text in a shared embedding space. We investigated its potential for versatile EEG decoding, evaluating performance in a range of few-shot and zero-shot settings. Overall, we show that EEG-CLIP manages to non-trivially align text and EEG representations. Our work presents a promising approach to learn general EEG representations, which could enable easier analyses of diverse decoding questions through zero-shot decoding or training task-specific models from fewer training examples. The code for reproducing our results is available at https://github.com/tidiane-camaret/EEGClip
>
---
#### [replaced 007] Multi-Microphone and Multi-Modal Emotion Recognition in Reverberant Environment
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.09545v3](http://arxiv.org/pdf/2409.09545v3)**

> **作者:** Ohad Cohen; Gershon Hazan; Sharon Gannot
>
> **备注:** 5 pages, 4 figures, 2 tables. Accepted to EUSIPCO 2025
>
> **摘要:** This paper presents a Multi-modal Emotion Recognition (MER) system designed to enhance emotion recognition accuracy in challenging acoustic conditions. Our approach combines a modified and extended Hierarchical Token-semantic Audio Transformer (HTS-AT) for multi-channel audio processing with an R(2+1)D Convolutional Neural Networks (CNN) model for video analysis. We evaluate our proposed method on a reverberated version of the Ryerson audio-visual database of emotional speech and song (RAVDESS) dataset using synthetic and real-world Room Impulse Responsess (RIRs). Our results demonstrate that integrating audio and video modalities yields superior performance compared to uni-modal approaches, especially in challenging acoustic conditions. Moreover, we show that the multimodal (audiovisual) approach that utilizes multiple microphones outperforms its single-microphone counterpart.
>
---
