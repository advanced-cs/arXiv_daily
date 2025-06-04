# 音频 cs.SD;  eess.SP

- **最新发布 39 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Speaker Diarization with Overlapping Community Detection Using Graph Attention Networks and Label Propagation Algorithm
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音说话人日志（speaker diarization）任务，旨在解决传统聚类方法在处理复杂嵌入分布和重叠语音段时的不足。论文提出OCDGALP方法，结合图注意力网络优化嵌入和节点连接，以及标签传播算法实现重叠社区检测。实验表明其显著降低了Diarization Error Rate，在DIHARD-III数据集上取得优异性能。**

- **链接: [http://arxiv.org/pdf/2506.02610v1](http://arxiv.org/pdf/2506.02610v1)**

> **作者:** Zhaoyang Li; Jie Wang; XiaoXiao Li; Wangjie Li; Longjie Luo; Lin Li; Qingyang Hong
>
> **摘要:** In speaker diarization, traditional clustering-based methods remain widely used in real-world applications. However, these methods struggle with the complex distribution of speaker embeddings and overlapping speech segments. To address these limitations, we propose an Overlapping Community Detection method based on Graph Attention networks and the Label Propagation Algorithm (OCDGALP). The proposed framework comprises two key components: (1) a graph attention network that refines speaker embeddings and node connections by aggregating information from neighboring nodes, and (2) a label propagation algorithm that assigns multiple community labels to each node, enabling simultaneous clustering and overlapping community detection. Experimental results show that the proposed method significantly reduces the Diarization Error Rate (DER), achieving a state-of-the-art 15.94% DER on the DIHARD-III dataset without oracle Voice Activity Detection (VAD), and an impressive 11.07% with oracle VAD.
>
---
#### [new 002] Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于雷达数据模拟任务，旨在解决现有雷达仿真方法依赖硬件参数、效率低及可控性差的问题。作者提出SA-Radar，通过波形参数嵌入实现属性可控的雷达张量生成，并构建ICFAR-Net网络与混合数据集，提升模拟数据的真实性与应用效果。**

- **链接: [http://arxiv.org/pdf/2506.03134v1](http://arxiv.org/pdf/2506.03134v1)**

> **作者:** Weiqing Xiao; Hao Huang; Chonghao Zhong; Yujie Lin; Nan Wang; Xiaoxue Chen; Zhaoxi Chen; Saining Zhang; Shuocheng Yang; Pierre Merriaux; Lei Lei; Hao Zhao
>
> **备注:** Code: https://github.com/zhuxing0/SA-Radar Project page: https://zhuxing0.github.io/projects/SA-Radar
>
> **摘要:** We present SA-Radar (Simulate Any Radar), a radar simulation approach that enables controllable and efficient generation of radar cubes conditioned on customizable radar attributes. Unlike prior generative or physics-based simulators, SA-Radar integrates both paradigms through a waveform-parameterized attribute embedding. We design ICFAR-Net, a 3D U-Net conditioned on radar attributes encoded via waveform parameters, which captures signal variations induced by different radar configurations. This formulation bypasses the need for detailed radar hardware specifications and allows efficient simulation of range-azimuth-Doppler (RAD) tensors across diverse sensor settings. We further construct a mixed real-simulated dataset with attribute annotations to robustly train the network. Extensive evaluations on multiple downstream tasks-including 2D/3D object detection and radar semantic segmentation-demonstrate that SA-Radar's simulated data is both realistic and effective, consistently improving model performance when used standalone or in combination with real data. Our framework also supports simulation in novel sensor viewpoints and edited scenes, showcasing its potential as a general-purpose radar data engine for autonomous driving applications. Code and additional materials are available at https://zhuxing0.github.io/projects/SA-Radar.
>
---
#### [new 003] Trusted Fake Audio Detection Based on Dirichlet Distribution
- **分类: cs.SD; cs.MM; eess.AS**

- **简介: 该论文属于音频安全任务，旨在解决深度学习生成的虚假音频带来的网络安全问题。通过引入基于Dirichlet分布的方法，建模检测结果的不确定性，提升模型决策的可信度。论文在ASVspoof系列数据集上验证了所提方法在准确性、鲁棒性和可信性方面的优越性能。**

- **链接: [http://arxiv.org/pdf/2506.02401v1](http://arxiv.org/pdf/2506.02401v1)**

> **作者:** Chi Ding; Junxiao Xue; Cong Wang; Hao Zhou
>
> **摘要:** With the continuous development of deep learning-based speech conversion and speech synthesis technologies, the cybersecurity problem posed by fake audio has become increasingly serious. Previously proposed models for defending against fake audio have attained remarkable performance. However, they all fall short in modeling the trustworthiness of the decisions made by the models themselves. Based on this, we put forward a plausible fake audio detection approach based on the Dirichlet distribution with the aim of enhancing the reliability of fake audio detection. Specifically, we first generate evidence through a neural network. Uncertainty is then modeled using the Dirichlet distribution. By modeling the belief distribution with the parameters of the Dirichlet distribution, an estimate of uncertainty can be obtained for each decision. Finally, the predicted probabilities and corresponding uncertainty estimates are combined to form the final opinion. On the ASVspoof series dataset (i.e., ASVspoof 2019 LA, ASVspoof 2021 LA, and DF), we conduct a number of comparison experiments to verify the excellent performance of the proposed model in terms of accuracy, robustness, and trustworthiness.
>
---
#### [new 004] DnR-nonverbal: Cinematic Audio Source Separation Dataset Containing Non-Verbal Sounds
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频源分离任务，旨在解决电影音效中非语言声音（如笑声、尖叫声）易被误分类为效果声的问题。作者构建了新数据集DnR-nonverbal，将非语言声纳入语音轨道，提升了模型对真实电影音频的分离效果。**

- **链接: [http://arxiv.org/pdf/2506.02499v1](http://arxiv.org/pdf/2506.02499v1)**

> **作者:** Takuya Hasumi; Yusuke Fujita
>
> **备注:** Accepted to Interspeech 2025, 5 pages, 3 figures, dataset is available at https://zenodo.org/records/15470640
>
> **摘要:** We propose a new dataset for cinematic audio source separation (CASS) that handles non-verbal sounds. Existing CASS datasets only contain reading-style sounds as a speech stem. These datasets differ from actual movie audio, which is more likely to include acted-out voices. Consequently, models trained on conventional datasets tend to have issues where emotionally heightened voices, such as laughter and screams, are more easily separated as an effect, not speech. To address this problem, we build a new dataset, DnR-nonverbal. The proposed dataset includes non-verbal sounds like laughter and screams in the speech stem. From the experiments, we reveal the issue of non-verbal sound extraction by the current CASS model and show that our dataset can effectively address the issue in the synthetic and actual movie audio. Our dataset is available at https://zenodo.org/records/15470640.
>
---
#### [new 005] Breaking the Barriers of Text-Hungry and Audio-Deficient AI
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在解决当前AI依赖文本、忽视音频语言的问题。工作包括提出无需文本的音频到音频翻译框架，引入多尺度音频语义变换（MAST）及结合分数扩散模型，实现高保真语音生成，服务于无文字语言群体。**

- **链接: [http://arxiv.org/pdf/2506.02443v1](http://arxiv.org/pdf/2506.02443v1)**

> **作者:** Hamidou Tembine; Issa Bamia; Massa NDong; Bakary Coulibaly; Oumar Issiaka Traore; Moussa Traore; Moussa Sanogo; Mamadou Eric Sangare; Salif Kante; Daryl Noupa Yongueng; Hafiz Tiomoko Ali; Malik Tiomoko; Frejus Laleye; Boualem Djehiche; Wesmanegda Elisee Dipama; Idris Baba Saje; Hammid Mohammed Ibrahim; Moumini Sanogo; Marie Coursel Nininahazwe; Abdul-Latif Siita; Haine Mhlongo; Teddy Nelvy Dieu Merci Kouka; Mariam Serine Jeridi; Mutiyamuogo Parfait Mupenge; Lekoueiry Dehah; Abdoul Aziz Bio Sidi Bouko; Wilfried Franceslas Zokoue; Odette Richette Sambila; Alina RS Mbango; Mady Diagouraga; Oumarou Moussa Sanoussi; Gizachew Dessalegn; Mohamed Lamine Samoura; Bintou Laetitia Audrey Coulibaly
>
> **备注:** 61 pages, 16 figures, 14 tables, 25 languages, 13 blockaudio per language. Presented at AI Mali, May 2025
>
> **摘要:** While global linguistic diversity spans more than 7164 recognized languages, the current dominant architecture of machine intelligence remains fundamentally biased toward written text. This bias excludes over 700 million people particularly in rural and remote regions who are audio-literate. In this work, we introduce a fully textless, audio-to-audio machine intelligence framework designed to serve this underserved population, and all the people who prefer audio-efficiency. Our contributions include novel Audio-to-Audio translation architectures that bypass text entirely, including spectrogram-, scalogram-, wavelet-, and unit-based models. Central to our approach is the Multiscale Audio-Semantic Transform (MAST), a representation that encodes tonal, prosodic, speaker, and expressive features. We further integrate MAST into a fractional diffusion of mean-field-type framework powered by fractional Brownian motion. It enables the generation of high-fidelity, semantically consistent speech without reliance on textual supervision. The result is a robust and scalable system capable of learning directly from raw audio, even in languages that are unwritten or rarely digitized. This work represents a fundamental shift toward audio-native machine intelligence systems, expanding access to language technologies for communities historically left out of the current machine intelligence ecosystem.
>
---
#### [new 006] Comparison of spectrogram scaling in multi-label Music Genre Recognition
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于多标签音乐流派识别任务，旨在解决流派界限模糊及组合多样带来的分类难题。论文比较了不同频谱图缩放方法与模型训练策略，在一个自建的18000条标注数据集上进行了实验验证。**

- **链接: [http://arxiv.org/pdf/2506.02091v1](http://arxiv.org/pdf/2506.02091v1)**

> **作者:** Bartosz Karpiński; Cyryl Leszczyński
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** As the accessibility and ease-of-use of digital audio workstations increases, so does the quantity of music available to the average listener; additionally, differences between genres are not always well defined and can be abstract, with widely varying combinations of genres across individual records. In this article, multiple preprocessing methods and approaches to model training are described and compared, accounting for the eclectic nature of today's albums. A custom, manually labeled dataset of more than 18000 entries has been used to perform the experiments.
>
---
#### [new 007] LASPA: Language Agnostic Speaker Disentanglement with Prefix-Tuned Cross-Attention
- **分类: cs.SD; cs.AI; cs.LG; cs.MM**

- **简介: 论文提出LASPA模型，旨在解决多语言环境下说话人识别中语言与说话人特征纠缠的问题。通过前缀调优的跨注意力机制，实现语言无关的说话人解耦，提升跨语言及单语场景下的识别准确率，有效分离语言信息与说话人嵌入特征。**

- **链接: [http://arxiv.org/pdf/2506.02083v1](http://arxiv.org/pdf/2506.02083v1)**

> **作者:** Aditya Srinivas Menon; Raj Prakash Gohil; Kumud Tripathi; Pankaj Wasnik
>
> **备注:** Accepted at Interspeech 2025, Netherlands
>
> **摘要:** Speaker recognition models face challenges in multi-lingual settings due to the entanglement of linguistic information within speaker embeddings. The overlap between vocal traits such as accent, vocal anatomy, and a language's phonetic structure complicates separating linguistic and speaker information. Disentangling these components can significantly improve speaker recognition accuracy. To this end, we propose a novel disentanglement learning strategy that integrates joint learning through prefix-tuned cross-attention. This approach is particularly effective when speakers switch between languages. Experimental results show the model generalizes across monolingual and multi-lingual settings, including unseen languages. Notably, the proposed model improves the equal error rate across multiple datasets, highlighting its ability to separate language information from speaker embeddings and enhance recognition in diverse linguistic conditions.
>
---
#### [new 008] SALF-MOS: Speaker Agnostic Latent Features Downsampled for MOS Prediction
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音质量评估任务，旨在解决传统客观指标无法准确预测主观MOS评分的问题。作者提出了SALF-MOS模型，通过卷积序列提取音频潜在特征，实现端到端、小规模且高效的MOS预测，取得了良好的评估效果。**

- **链接: [http://arxiv.org/pdf/2506.02082v1](http://arxiv.org/pdf/2506.02082v1)**

> **作者:** Saurabh Agrawal; Raj Gohil; Gopal Kumar Agrawal; Vikram C M; Kushal Verma
>
> **摘要:** Speech quality assessment is a critical process in selecting text-to-speech synthesis (TTS) or voice conversion models. Evaluation of voice synthesis can be done using objective metrics or subjective metrics. Although there are many objective metrics like the Perceptual Evaluation of Speech Quality (PESQ), Perceptual Objective Listening Quality Assessment (POLQA) or Short-Time Objective Intelligibility (STOI) but none of them is feasible in selecting the best model. On the other hand subjective metric like Mean Opinion Score is highly reliable but it requires a lot of manual efforts and are time-consuming. To counter the issues in MOS Evaluation, we have developed a novel model, Speaker Agnostic Latent Features (SALF)-Mean Opinion Score (MOS) which is a small-sized, end-to-end, highly generalized and scalable model for predicting MOS score on a scale of 5. We use the sequences of convolutions and stack them to get the latent features of the audio samples to get the best state-of-the-art results based on mean squared error (MSE), Linear Concordance Correlation coefficient (LCC), Spearman Rank Correlation Coefficient (SRCC) and Kendall Rank Correlation Coefficient (KTAU).
>
---
#### [new 009] On the Language and Gender Biases in PSTN, VoIP and Neural Audio Codecs
- **分类: cs.SD; eess.AS**

- **简介: 该论文研究了语音技术中音频编解码器的语言和性别偏见问题，属于公平性与包容性任务。为解决音频转码可能加剧社会偏见的问题，分析了超过200万个多语种音频文件，发现PSTN编解码器存在性别偏见，神经编解码器引入语言偏见。**

- **链接: [http://arxiv.org/pdf/2506.02545v1](http://arxiv.org/pdf/2506.02545v1)**

> **作者:** Kemal Altwlkany; Amar Kuric; Emanuel Lacic
>
> **备注:** Submitted to INTERSPEECH2025
>
> **摘要:** In recent years, there has been a growing focus on fairness and inclusivity within speech technology, particularly in areas such as automatic speech recognition and speech sentiment analysis. When audio is transcoded prior to processing, as is the case in streaming or real-time applications, any inherent bias in the coding mechanism may result in disparities. This not only affects user experience but can also have broader societal implications by perpetuating stereotypes and exclusion. Thus, it is important that audio coding mechanisms are unbiased. In this work, we contribute towards the scarce research with respect to language and gender biases of audio codecs. By analyzing the speech quality of over 2 million multilingual audio files after transcoding through a representative subset of codecs (PSTN, VoIP and neural), our results indicate that PSTN codecs are strongly biased in terms of gender and that neural codecs introduce language biases.
>
---
#### [new 010] Cocktail-Party Audio-Visual Speech Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频-视觉语音识别（AVSR）任务，旨在解决鸡尾酒会场景中复杂噪音下的语音识别问题。作者构建了一个包含说话和静默面部片段的新数据集，并提出新方法，在极端噪声环境下将词错误率显著降低67%，无需显式分割线索。**

- **链接: [http://arxiv.org/pdf/2506.02178v1](http://arxiv.org/pdf/2506.02178v1)**

> **作者:** Thai-Binh Nguyen; Ngoc-Quan Pham; Alexander Waibel
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Audio-Visual Speech Recognition (AVSR) offers a robust solution for speech recognition in challenging environments, such as cocktail-party scenarios, where relying solely on audio proves insufficient. However, current AVSR models are often optimized for idealized scenarios with consistently active speakers, overlooking the complexities of real-world settings that include both speaking and silent facial segments. This study addresses this gap by introducing a novel audio-visual cocktail-party dataset designed to benchmark current AVSR systems and highlight the limitations of prior approaches in realistic noisy conditions. Additionally, we contribute a 1526-hour AVSR dataset comprising both talking-face and silent-face segments, enabling significant performance gains in cocktail-party environments. Our approach reduces WER by 67% relative to the state-of-the-art, reducing WER from 119% to 39.2% in extreme noise, without relying on explicit segmentation cues.
>
---
#### [new 011] DGMO: Training-Free Audio Source Separation through Diffusion-Guided Mask Optimization
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语言查询音频源分离任务，旨在通过自然语言实现开放词汇的声音分离。现有方法需特定训练，而本文提出无需训练的DGmo框架，利用预训练扩散模型优化频谱掩码，实现零样本分离，解决了跨模态与无需标注数据的问题。**

- **链接: [http://arxiv.org/pdf/2506.02858v1](http://arxiv.org/pdf/2506.02858v1)**

> **作者:** Geonyoung Lee; Geonhee Han; Paul Hongsuck Seo
>
> **备注:** Interspeech 2025
>
> **摘要:** Language-queried Audio Source Separation (LASS) enables open-vocabulary sound separation via natural language queries. While existing methods rely on task-specific training, we explore whether pretrained diffusion models, originally designed for audio generation, can inherently perform separation without further training. In this study, we introduce a training-free framework leveraging generative priors for zero-shot LASS. Analyzing na\"ive adaptations, we identify key limitations arising from modality-specific challenges.To address these issues, we propose Diffusion-Guided Mask Optimization (DGMO), a test-time optimization framework that refines spectrogram masks for precise, input-aligned separation. Our approach effectively repurposes pretrained diffusion models for source separation, achieving competitive performance without task-specific supervision. This work expands the application of diffusion models beyond generation, establishing a new paradigm for zero-shot audio separation. The code is available at: https://wltschmrz.github.io/DGMO/
>
---
#### [new 012] UltrasonicSpheres: Localized, Multi-Channel Sound Spheres Using Off-the-Shelf Speakers and Earables
- **分类: cs.SD; cs.HC; eess.AS**

- **简介: 论文提出UltrasonicSpheres系统，属于音频技术任务，旨在解决公共空间中个性化、定位音频传输的问题。利用超声波扬声器广播多通道音频，用户通过耳塞式耳机选择并解码所需音轨，实现在不干扰他人的情况下享受本地化声音体验。**

- **链接: [http://arxiv.org/pdf/2506.02715v1](http://arxiv.org/pdf/2506.02715v1)**

> **作者:** Michael Küttner; Valeria Sitz; Kathrin Gerling; Michael Beigl; Tobias Röddiger
>
> **摘要:** We present a demo ofUltrasonicSpheres, a novel system for location-specific audio delivery using wearable earphones that decode ultrasonic signals into audible sound. Unlike conventional beamforming setups, UltrasonicSpheres relies on single ultrasonic speakers to broadcast localized audio with multiple channels, each encoded on a distinct ultrasonic carrier frequency. Users wearing our acoustically transparent earphones can demodulate their selected stream, such as exhibit narrations in a chosen language, while remaining fully aware of ambient environmental sounds. The experience preserves spatial audio perception, giving the impression that the sound originates directly from the physical location of the source. This enables personalized, localized audio without requiring pairing, tracking, or additional infrastructure. Importantly, visitors not equipped with the earphones are unaffected, as the ultrasonic signals are inaudible to the human ear. Our demo invites participants to explore multiple co-located audio zones and experience how UltrasonicSpheres supports unobtrusive delivery of personalized sound in public spaces.
>
---
#### [new 013] SOVA-Bench: Benchmarking the Speech Conversation Ability for LLM-based Voice Assistant
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音交互系统评估任务，旨在解决缺乏对语音助手生成语音质量的系统评估问题。作者提出了SOVA-Bench基准，综合评估语音大模型的语言理解、语音识别及语义与声学生成能力，推动语音交互系统发展。**

- **链接: [http://arxiv.org/pdf/2506.02457v1](http://arxiv.org/pdf/2506.02457v1)**

> **作者:** Yixuan Hou; Heyang Liu; Yuhao Wang; Ziyang Cheng; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** Thanks to the steady progress of large language models (LLMs), speech encoding algorithms and vocoder structure, recent advancements have enabled generating speech response directly from a user instruction. However, benchmarking the generated speech quality has been a neglected but critical issue, considering the shift from the pursuit of semantic accuracy to vivid and spontaneous speech flow. Previous evaluation focused on the speech-understanding ability, lacking a quantification of acoustic quality. In this paper, we propose Speech cOnversational Voice Assistant Benchmark (SOVA-Bench), providing a comprehension comparison of the general knowledge, speech recognition and understanding, along with both semantic and acoustic generative ability between available speech LLMs. To the best of our knowledge, SOVA-Bench is one of the most systematic evaluation frameworks for speech LLMs, inspiring the direction of voice interaction systems.
>
---
#### [new 014] Cross-attention and Self-attention for Audio-visual Speaker Diarization in MISP-Meeting Challenge
- **分类: cs.SD**

- **简介: 该论文属于多模态语音处理任务，旨在解决音视频说话人日志（AVSD）问题。作者提出了CASA-Net模型，融合交叉注意力与自注意力机制以捕捉跨模态及上下文信息，并结合伪标签优化、中值滤波等方法提升识别准确率与平滑预测结果。**

- **链接: [http://arxiv.org/pdf/2506.02621v1](http://arxiv.org/pdf/2506.02621v1)**

> **作者:** Zhaoyang Li; Haodong Zhou; Longjie Luo; Xiaoxiao Li; Yongxin Chen; Lin Li; Qingyang Hong
>
> **摘要:** This paper presents the system developed for Task 1 of the Multi-modal Information-based Speech Processing (MISP) 2025 Challenge. We introduce CASA-Net, an embedding fusion method designed for end-to-end audio-visual speaker diarization (AVSD) systems. CASA-Net incorporates a cross-attention (CA) module to effectively capture cross-modal interactions in audio-visual signals and employs a self-attention (SA) module to learn contextual relationships among audio-visual frames. To further enhance performance, we adopt a training strategy that integrates pseudo-label refinement and retraining, improving the accuracy of timestamp predictions. Additionally, median filtering and overlap averaging are applied as post-processing techniques to eliminate outliers and smooth prediction labels. Our system achieved a diarization error rate (DER) of 8.18% on the evaluation set, representing a relative improvement of 47.3% over the baseline DER of 15.52%.
>
---
#### [new 015] Unveiling Audio Deepfake Origins: A Deep Metric learning And Conformer Network Approach With Ensemble Fusion
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频源追踪任务，旨在解决音频深度伪造的溯源问题。通过结合深度度量学习、Conformer网络与集成融合方法，提升对真实与伪造语音的辨别能力及源系统追踪准确性。**

- **链接: [http://arxiv.org/pdf/2506.02085v1](http://arxiv.org/pdf/2506.02085v1)**

> **作者:** Ajinkya Kulkarni; Sandipana Dowerah; Tanel Alumae; Mathew Magimai. -Doss
>
> **备注:** Accepted at Interspeech 2025, Netherlands
>
> **摘要:** Audio deepfakes are acquiring an unprecedented level of realism with advanced AI. While current research focuses on discerning real speech from spoofed speech, tracing the source system is equally crucial. This work proposes a novel audio source tracing system combining deep metric multi-class N-pair loss with Real Emphasis and Fake Dispersion framework, a Conformer classification network, and ensemble score-embedding fusion. The N-pair loss improves discriminative ability, while Real Emphasis and Fake Dispersion enhance robustness by focusing on differentiating real and fake speech patterns. The Conformer network captures both global and local dependencies in the audio signal, crucial for source tracing. The proposed ensemble score-embedding fusion shows an optimal trade-off between in-domain and out-of-domain source tracing scenarios. We evaluate our method using Frechet Distance and standard metrics, demonstrating superior performance in source tracing over the baseline system.
>
---
#### [new 016] Enhancing Speech Emotion Recognition with Graph-Based Multimodal Fusion and Prosodic Features for the Speech Emotion Recognition in Naturalistic Conditions Challenge at Interspeech 2025
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于语音情感识别（SER）任务，旨在解决自然场景下情感识别准确率低的问题。作者提出了一种结合图注意力网络的多模态融合方法，并利用基频量化与预训练音频标签模型提升性能，最终在Interspeech 2025挑战赛中取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2506.02088v1](http://arxiv.org/pdf/2506.02088v1)**

> **作者:** Alef Iury Siqueira Ferreira; Lucas Rafael Gris; Alexandre Ferro Filho; Lucas Ólives; Daniel Ribeiro; Luiz Fernando; Fernanda Lustosa; Rodrigo Tanaka; Frederico Santos de Oliveira; Arlindo Galvão Filho
>
> **摘要:** Training SER models in natural, spontaneous speech is especially challenging due to the subtle expression of emotions and the unpredictable nature of real-world audio. In this paper, we present a robust system for the INTERSPEECH 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge, focusing on categorical emotion recognition. Our method combines state-of-the-art audio models with text features enriched by prosodic and spectral cues. In particular, we investigate the effectiveness of Fundamental Frequency (F0) quantization and the use of a pretrained audio tagging model. We also employ an ensemble model to improve robustness. On the official test set, our system achieved a Macro F1-score of 39.79% (42.20% on validation). Our results underscore the potential of these methods, and analysis of fusion techniques confirmed the effectiveness of Graph Attention Networks. Our source code is publicly available.
>
---
#### [new 017] Learning More with Less: Self-Supervised Approaches for Low-Resource Speech Emotion Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决低资源语言因标注数据不足导致的识别难题。作者采用对比学习和BYOL两种自监督方法，提升跨语言情感识别性能，并在多种语言上取得显著效果，为构建包容性强的情感识别系统提供了新思路。**

- **链接: [http://arxiv.org/pdf/2506.02059v1](http://arxiv.org/pdf/2506.02059v1)**

> **作者:** Ziwei Gong; Pengyuan Shi; Kaan Donbekci; Lin Ai; Run Chen; David Sasu; Zehui Wu; Julia Hirschberg
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Speech Emotion Recognition (SER) has seen significant progress with deep learning, yet remains challenging for Low-Resource Languages (LRLs) due to the scarcity of annotated data. In this work, we explore unsupervised learning to improve SER in low-resource settings. Specifically, we investigate contrastive learning (CL) and Bootstrap Your Own Latent (BYOL) as self-supervised approaches to enhance cross-lingual generalization. Our methods achieve notable F1 score improvements of 10.6% in Urdu, 15.2% in German, and 13.9% in Bangla, demonstrating their effectiveness in LRLs. Additionally, we analyze model behavior to provide insights on key factors influencing performance across languages, and also highlighting challenges in low-resource SER. This work provides a foundation for developing more inclusive, explainable, and robust emotion recognition systems for underrepresented languages.
>
---
#### [new 018] TalkingMachines: Real-Time Audio-Driven FaceTime-Style Video via Autoregressive Diffusion Models
- **分类: cs.SD; cs.AI; cs.GR**

- **简介: 该论文属于视频生成任务，旨在实现音频驱动的实时视频通话风格人物动画生成。通过改进预训练模型，采用知识蒸馏与高效推理流水线，解决了实时性、无限视频流及低延迟问题，提升了对话体验。**

- **链接: [http://arxiv.org/pdf/2506.03099v1](http://arxiv.org/pdf/2506.03099v1)**

> **作者:** Chetwin Low; Weimin Wang
>
> **摘要:** In this paper, we present TalkingMachines -- an efficient framework that transforms pretrained video generation models into real-time, audio-driven character animators. TalkingMachines enables natural conversational experiences by integrating an audio large language model (LLM) with our video generation foundation model. Our primary contributions include: (1) We adapt a pretrained SOTA image-to-video DiT into an audio-driven avatar generation model of 18 billion parameters; (2) We enable infinite video streaming without error accumulation through asymmetric knowledge distillation from a bidirectional teacher model into a sparse causal, autoregressive student model; (3) We design a high-throughput, low-latency inference pipeline incorporating several key engineering optimizations such as: (a) disaggregation of the DiT and VAE decoder across separate devices, (b) efficient overlap of inter-device communication and computation using CUDA streams, (c) elimination of redundant recomputations to maximize frame-generation throughput. Please see demo videos here - https://aaxwaz.github.io/TalkingMachines/
>
---
#### [new 019] MotionRAG-Diff: A Retrieval-Augmented Diffusion Framework for Long-Term Music-to-Dance Generation
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **简介: 该论文属于音乐驱动舞蹈生成任务，旨在解决长期舞蹈序列生成中连贯性、同步性和创造性不足的问题。作者提出MotionRAG-Diff框架，结合检索增强生成与扩散模型，实现高质量、与音乐同步的舞蹈生成。**

- **链接: [http://arxiv.org/pdf/2506.02661v1](http://arxiv.org/pdf/2506.02661v1)**

> **作者:** Mingyang Huang; Peng Zhang; Bang Zhang
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Generating long-term, coherent, and realistic music-conditioned dance sequences remains a challenging task in human motion synthesis. Existing approaches exhibit critical limitations: motion graph methods rely on fixed template libraries, restricting creative generation; diffusion models, while capable of producing novel motions, often lack temporal coherence and musical alignment. To address these challenges, we propose $\textbf{MotionRAG-Diff}$, a hybrid framework that integrates Retrieval-Augmented Generation (RAG) with diffusion-based refinement to enable high-quality, musically coherent dance generation for arbitrary long-term music inputs. Our method introduces three core innovations: (1) A cross-modal contrastive learning architecture that aligns heterogeneous music and dance representations in a shared latent space, establishing unsupervised semantic correspondence without paired data; (2) An optimized motion graph system for efficient retrieval and seamless concatenation of motion segments, ensuring realism and temporal coherence across long sequences; (3) A multi-condition diffusion model that jointly conditions on raw music signals and contrastive features to enhance motion quality and global synchronization. Extensive experiments demonstrate that MotionRAG-Diff achieves state-of-the-art performance in motion quality, diversity, and music-motion synchronization accuracy. This work establishes a new paradigm for music-driven dance generation by synergizing retrieval-based template fidelity with diffusion-based creative enhancement.
>
---
#### [new 020] Synthetic Speech Source Tracing using Metric Learning
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频取证任务，旨在解决合成语音的源追踪问题。现有方法多关注欺骗检测，缺乏有效追踪生成系统的手段。受说话人识别启发，作者采用分类和度量学习方法，在MLAADv5数据集上测试ResNet与自监督模型效果，结果显示ResNet表现优异，表明其在源追踪中的可行性，并指出优化自监督表示的重要性。**

- **链接: [http://arxiv.org/pdf/2506.02590v1](http://arxiv.org/pdf/2506.02590v1)**

> **作者:** Dimitrios Koutsianos; Stavros Zacharopoulos; Yannis Panagakis; Themos Stafylakis
>
> **备注:** Submitted to Interspeech 2025
>
> **摘要:** This paper addresses source tracing in synthetic speech-identifying generative systems behind manipulated audio via speaker recognition-inspired pipelines. While prior work focuses on spoofing detection, source tracing lacks robust solutions. We evaluate two approaches: classification-based and metric-learning. We tested our methods on the MLAADv5 benchmark using ResNet and self-supervised learning (SSL) backbones. The results show that ResNet achieves competitive performance with the metric learning approach, matching and even exceeding SSL-based systems. Our work demonstrates ResNet's viability for source tracing while underscoring the need to optimize SSL representations for this task. Our work bridges speaker recognition methodologies with audio forensic challenges, offering new directions for combating synthetic media manipulation.
>
---
#### [new 021] Fast-Converging Distributed Signal Estimation in Topology-Unconstrained Wireless Acoustic Sensor Networks
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于分布式信号估计任务，旨在解决拓扑无约束无线声学传感器网络中算法收敛速度慢的问题。作者提出了改进的TI-DANSE+算法，通过利用来自每个邻居的部分网络和信号，并结合树剪枝策略，加快收敛速度，同时节省通信带宽，提升了原有方法的性能与适用性。**

- **链接: [http://arxiv.org/pdf/2506.02797v1](http://arxiv.org/pdf/2506.02797v1)**

> **作者:** Paul Didier; Toon van Waterschoot; Simon Doclo; Jörg Bitzer; Marc Moonen
>
> **摘要:** This paper focuses on distributed signal estimation in topology-unconstrained wireless acoustic sensor networks (WASNs) where sensor nodes only transmit fused versions of their local sensor signals. For this task, the topology-independent (TI) distributed adaptive node-specific signal estimation (DANSE) algorithm (TI-DANSE) has previously been proposed. It converges towards the centralized signal estimation solution in non-fully connected and time-varying network topologies. However, the applicability of TI-DANSE in real-world scenarios is limited due to its slow convergence. The latter results from the fact that, in TI-DANSE, nodes only have access to the in-network sum of all fused signals in the WASN. We address this low convergence speed by introducing an improved TI-DANSE algorithm, referred to as TI-DANSE+, in which updating nodes separately use the partial in-network sums of fused signals coming from each of their neighbors. Nodes can maximize the number of available degrees of freedom in their local optimization problem, leading to faster convergence. This is further exploited by combining TI-DANSE+ with a tree-pruning strategy that maximizes the number of neighbors at the updating node. In fully connected WASNs, TI-DANSE+ converges as fast as the original DANSE algorithm (the latter only defined for fully connected WASNs) while using peer-to-peer data transmission instead of broadcasting and thus saving communication bandwidth. If link failures occur, the convergence of TI-DANSE+ towards the centralized solution is preserved without any change in its formulation. Altogether, the proposed TI-DANSE+ algorithm can be viewed as an all-round alternative to DANSE and TI-DANSE which (i) merges the advantages of both, (ii) reconciliates their differences into a single formulation, and (iii) shows advantages of its own in terms of communication bandwidth usage.
>
---
#### [new 022] Prompt-Unseen-Emotion: Zero-shot Expressive Speech Synthesis with Prompt-LLM Contextual Knowledge for Mixed Emotions
- **分类: eess.AS; cs.AI; cs.SD; eess.SP**

- **简介: 该论文属于语音合成任务，旨在解决现有系统仅能生成有限预定义情绪的问题。作者提出Prompt-Unseen-Emotion（PUE）方法，在零样本设置下生成未见过的情绪语音，通过LLM-TTS架构实现情绪提示与语音的一致性，并可量化每种情绪的权重。**

- **链接: [http://arxiv.org/pdf/2506.02742v1](http://arxiv.org/pdf/2506.02742v1)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **摘要:** Existing expressive text-to-speech (TTS) systems primarily model a limited set of categorical emotions, whereas human conversations extend far beyond these predefined emotions, making it essential to explore more diverse emotional speech generation for more natural interactions. To bridge this gap, this paper proposes a novel prompt-unseen-emotion (PUE) approach to generate unseen emotional speech via emotion-guided prompt learning. PUE is trained utilizing an LLM-TTS architecture to ensure emotional consistency between categorical emotion-relevant prompts and emotional speech, allowing the model to quantitatively capture different emotion weightings per utterance. During inference, mixed emotional speech can be generated by flexibly adjusting emotion proportions and leveraging LLM contextual knowledge, enabling the model to quantify different emotional styles. Our proposed PUE successfully facilitates expressive speech synthesis of unseen emotions in a zero-shot setting.
>
---
#### [new 023] Enhancing Lyrics Transcription on Music Mixtures with Consistency Loss
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于自动歌词转录（ALT）任务，旨在提升在音乐伴奏中识别歌词的准确性。为解决歌唱声音与语音识别模型适配性差的问题，作者采用低秩自适应（LoRA），并引入一致性损失优化双域训练，减少对歌声分离的依赖，取得了一定效果。**

- **链接: [http://arxiv.org/pdf/2506.02339v1](http://arxiv.org/pdf/2506.02339v1)**

> **作者:** Jiawen Huang; Felipe Sousa; Emir Demirel; Emmanouil Benetos; Igor Gadelha
>
> **备注:** submitted to Interspeech
>
> **摘要:** Automatic Lyrics Transcription (ALT) aims to recognize lyrics from singing voices, similar to Automatic Speech Recognition (ASR) for spoken language, but faces added complexity due to domain-specific properties of the singing voice. While foundation ASR models show robustness in various speech tasks, their performance degrades on singing voice, especially in the presence of musical accompaniment. This work focuses on this performance gap and explores Low-Rank Adaptation (LoRA) for ALT, investigating both single-domain and dual-domain fine-tuning strategies. We propose using a consistency loss to better align vocal and mixture encoder representations, improving transcription on mixture without relying on singing voice separation. Our results show that while na\"ive dual-domain fine-tuning underperforms, structured training with consistency loss yields modest but consistent gains, demonstrating the potential of adapting ASR foundation models for music.
>
---
#### [new 024] Leveraging Large Language Models in Visual Speech Recognition: Model Scaling, Context-Aware Decoding, and Iterative Polishing
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于视觉语音识别（VSR）任务，旨在通过利用大语言模型（LLMs）提升VSR性能。论文研究了LLM规模对识别效果的影响，并提出上下文感知解码和迭代优化方法，以提高识别准确率。**

- **链接: [http://arxiv.org/pdf/2506.02012v1](http://arxiv.org/pdf/2506.02012v1)**

> **作者:** Zehua Liu; Xiaolou Li; Li Guo; Lantian Li; Dong Wang
>
> **摘要:** Visual Speech Recognition (VSR) transcribes speech by analyzing lip movements. Recently, Large Language Models (LLMs) have been integrated into VSR systems, leading to notable performance improvements. However, the potential of LLMs has not been extensively studied, and how to effectively utilize LLMs in VSR tasks remains unexplored. This paper systematically explores how to better leverage LLMs for VSR tasks and provides three key contributions: (1) Scaling Test: We study how the LLM size affects VSR performance, confirming a scaling law in the VSR task. (2) Context-Aware Decoding: We add contextual text to guide the LLM decoding, improving recognition accuracy. (3) Iterative Polishing: We propose iteratively refining LLM outputs, progressively reducing recognition errors. Extensive experiments demonstrate that by these designs, the great potential of LLMs can be largely harnessed, leading to significant VSR performance improvement.
>
---
#### [new 025] Singing Voice Graph Modeling for SingFake Detection
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于音频伪造检测任务，旨在解决歌唱语音深伪（SingFake）的鉴别问题。为应对现有模型难以适应新攻击的问题，作者提出了SingGraph模型，融合MERT与wav2vec2.0进行音高、节奏及歌词分析，并引入基于音乐知识的数据增强技术，显著提升了在多种场景下的检测性能。**

- **链接: [http://arxiv.org/pdf/2406.03111v2](http://arxiv.org/pdf/2406.03111v2)**

> **作者:** Xuanjun Chen; Haibin Wu; Jyh-Shing Roger Jang; Hung-yi Lee
>
> **备注:** Accepted by Interspeech 2024; Our code is available at https://github.com/xjchenGit/SingGraph.git
>
> **摘要:** Detecting singing voice deepfakes, or SingFake, involves determining the authenticity and copyright of a singing voice. Existing models for speech deepfake detection have struggled to adapt to unseen attacks in this unique singing voice domain of human vocalization. To bridge the gap, we present a groundbreaking SingGraph model. The model synergizes the capabilities of the MERT acoustic music understanding model for pitch and rhythm analysis with the wav2vec2.0 model for linguistic analysis of lyrics. Additionally, we advocate for using RawBoost and beat matching techniques grounded in music domain knowledge for singing voice augmentation, thereby enhancing SingFake detection performance. Our proposed method achieves new state-of-the-art (SOTA) results within the SingFake dataset, surpassing the previous SOTA model across three distinct scenarios: it improves EER relatively for seen singers by 13.2%, for unseen singers by 24.3%, and unseen singers using different codecs by 37.1%.
>
---
#### [new 026] No Audiogram: Leveraging Existing Scores for Personalized Speech Intelligibility Prediction
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于语音可懂度预测任务，旨在解决个性化预测中仅依赖听力图导致精度不足的问题。作者提出SSIPNet模型，利用已有（音频，得分）对构建高维表征，实现对新音频的准确预测，无需额外听众特征。实验表明方法优于传统听力图预测。**

- **链接: [http://arxiv.org/pdf/2506.02039v1](http://arxiv.org/pdf/2506.02039v1)**

> **作者:** Haoshuai Zhou; Changgeng Mo; Boxuan Cao; Linkai Li; Shan Xiang Wang
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Personalized speech intelligibility prediction is challenging. Previous approaches have mainly relied on audiograms, which are inherently limited in accuracy as they only capture a listener's hearing threshold for pure tones. Rather than incorporating additional listener features, we propose a novel approach that leverages an individual's existing intelligibility data to predict their performance on new audio. We introduce the Support Sample-Based Intelligibility Prediction Network (SSIPNet), a deep learning model that leverages speech foundation models to build a high-dimensional representation of a listener's speech recognition ability from multiple support (audio, score) pairs, enabling accurate predictions for unseen audio. Results on the Clarity Prediction Challenge dataset show that, even with a small number of support (audio, score) pairs, our method outperforms audiogram-based predictions. Our work presents a new paradigm for personalized speech intelligibility prediction.
>
---
#### [new 027] Overcoming Data Scarcity in Multi-Dialectal Arabic ASR via Whisper Fine-Tuning
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多方言阿拉伯语语音识别任务，旨在解决方言语音识别中数据不足的问题。通过微调Whisper模型，结合标准阿拉伯语和多种方言数据训练，发现小规模标准语数据可显著提升效果，且混合方言模型表现与单一方言模型相当，为低资源场景提供了有效方案。**

- **链接: [http://arxiv.org/pdf/2506.02627v1](http://arxiv.org/pdf/2506.02627v1)**

> **作者:** Ömer Tarik Özyilmaz; Matt Coler; Matias Valdenegro-Toro
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Although commercial Arabic automatic speech recognition (ASR) systems support Modern Standard Arabic (MSA), they struggle with dialectal speech. We investigate the effect of fine-tuning OpenAI's Whisper on five major Arabic dialects (Gulf, Levantine, Iraqi, Egyptian, Maghrebi) using Mozilla Common Voice for MSA and the MASC dataset for dialectal speech. We evaluate MSA training size effects, benefits of pre-training on MSA data, and dialect-specific versus dialect-pooled models. We find that small amounts of MSA fine-tuning data yield substantial improvements for smaller models, matching larger non-fine-tuned models. While MSA pre-training shows minimal benefit, suggesting limited shared features between MSA and dialects, our dialect-pooled models perform comparably to dialect-specific ones. This indicates that pooling dialectal data, when properly balanced, can help address data scarcity in low-resource ASR without significant performance loss.
>
---
#### [new 028] CNVSRC 2024: The Second Chinese Continuous Visual Speech Recognition Challenge
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文属于中文连续视觉语音识别任务，旨在解决大词汇量下的视觉语音识别问题。论文通过改进基线系统和引入新数据集提升识别效果，推动相关技术发展。**

- **链接: [http://arxiv.org/pdf/2506.02010v1](http://arxiv.org/pdf/2506.02010v1)**

> **作者:** Zehua Liu; Xiaolou Li; Chen Chen; Lantian Li; Dong Wang
>
> **备注:** to be published in INTERSPEECH 2025
>
> **摘要:** This paper presents the second Chinese Continuous Visual Speech Recognition Challenge (CNVSRC 2024), which builds on CNVSRC 2023 to advance research in Chinese Large Vocabulary Continuous Visual Speech Recognition (LVC-VSR). The challenge evaluates two test scenarios: reading in recording studios and Internet speech. CNVSRC 2024 uses the same datasets as its predecessor CNVSRC 2023, which involves CN-CVS for training and CNVSRC-Single/Multi for development and evaluation. However, CNVSRC 2024 introduced two key improvements: (1) a stronger baseline system, and (2) an additional dataset, CN-CVS2-P1, for open tracks to improve data volume and diversity. The new challenge has demonstrated several important innovations in data preprocessing, feature extraction, model design, and training strategies, further pushing the state-of-the-art in Chinese LVC-VSR. More details and resources are available at the official website.
>
---
#### [new 029] Investigating the Reasonable Effectiveness of Speaker Pre-Trained Models and their Synergistic Power for SingMOS Prediction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音质量评估任务，旨在解决预测合成歌声的主观评分（SingMOS）问题。作者提出使用说话人识别预训练模型（如x-vector、ECAPA）提升预测效果，并设计了基于Bhattacharya距离的融合框架BATCH，实现优于现有方法的性能表现。**

- **链接: [http://arxiv.org/pdf/2506.02232v1](http://arxiv.org/pdf/2506.02232v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Swarup Ranjan Behera; Pailla Balakrishna Reddy; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this study, we focus on Singing Voice Mean Opinion Score (SingMOS) prediction. Previous research have shown the performance benefit with the use of state-of-the-art (SOTA) pre-trained models (PTMs). However, they haven't explored speaker recognition speech PTMs (SPTMs) such as x-vector, ECAPA and we hypothesize that it will be the most effective for SingMOS prediction. We believe that due to their speaker recognition pre-training, it equips them to capture fine-grained vocal features (e.g., pitch, tone, intensity) from synthesized singing voices in a much more better way than other PTMs. Our experiments with SOTA PTMs including SPTMs and music PTMs validates the hypothesis. Additionally, we introduce a novel fusion framework, BATCH that uses Bhattacharya Distance for fusion of PTMs. Through BATCH with the fusion of speaker recognition SPTMs, we report the topmost performance comparison to all the individual PTMs and baseline fusion techniques as well as setting SOTA.
>
---
#### [new 030] Towards Machine Unlearning for Paralinguistic Speech Processing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别与抑郁检测任务，旨在解决模型遗忘学习后性能下降问题。作者提出了SISA++方法，通过模型权重平均提升遗忘效果，并提供特征表示和架构选择建议，以减轻遗忘带来的性能损失。**

- **链接: [http://arxiv.org/pdf/2506.02230v1](http://arxiv.org/pdf/2506.02230v1)**

> **作者:** Orchid Chetia Phukan; Girish; Mohd Mujtaba Akhtar; Shubham Singh; Swarup Ranjan Behera; Vandana Rajan; Muskaan Singh; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In this work, we pioneer the study of Machine Unlearning (MU) for Paralinguistic Speech Processing (PSP). We focus on two key PSP tasks: Speech Emotion Recognition (SER) and Depression Detection (DD). To this end, we propose, SISA++, a novel extension to previous state-of-the-art (SOTA) MU method, SISA by merging models trained on different shards with weight-averaging. With such modifications, we show that SISA++ preserves performance more in comparison to SISA after unlearning in benchmark SER (CREMA-D) and DD (E-DAIC) datasets. Also, to guide future research for easier adoption of MU for PSP, we present ``cookbook recipes'' - actionable recommendations for selecting optimal feature representations and downstream architectures that can mitigate performance degradation after the unlearning process.
>
---
#### [new 031] CapSpeech: Enabling Downstream Applications in Style-Captioned Text-to-Speech
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于文本到语音合成任务，旨在解决风格标注语音生成的数据与应用难题。作者提出了CapSpeech基准，包含千万级标注数据及新子任务，支持多种语音风格生成，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.02863v1](http://arxiv.org/pdf/2506.02863v1)**

> **作者:** Helin Wang; Jiarui Hai; Dading Chong; Karan Thakkar; Tiantian Feng; Dongchao Yang; Junhyeok Lee; Laureano Moro Velazquez; Jesus Villalba; Zengyi Qin; Shrikanth Narayanan; Mounya Elhiali; Najim Dehak
>
> **摘要:** Recent advancements in generative artificial intelligence have significantly transformed the field of style-captioned text-to-speech synthesis (CapTTS). However, adapting CapTTS to real-world applications remains challenging due to the lack of standardized, comprehensive datasets and limited research on downstream tasks built upon CapTTS. To address these gaps, we introduce CapSpeech, a new benchmark designed for a series of CapTTS-related tasks, including style-captioned text-to-speech synthesis with sound events (CapTTS-SE), accent-captioned TTS (AccCapTTS), emotion-captioned TTS (EmoCapTTS), and text-to-speech synthesis for chat agent (AgentTTS). CapSpeech comprises over 10 million machine-annotated audio-caption pairs and nearly 0.36 million human-annotated audio-caption pairs. In addition, we introduce two new datasets collected and recorded by a professional voice actor and experienced audio engineers, specifically for the AgentTTS and CapTTS-SE tasks. Alongside the datasets, we conduct comprehensive experiments using both autoregressive and non-autoregressive models on CapSpeech. Our results demonstrate high-fidelity and highly intelligible speech synthesis across a diverse range of speaking styles. To the best of our knowledge, CapSpeech is the largest available dataset offering comprehensive annotations for CapTTS-related tasks. The experiments and findings further provide valuable insights into the challenges of developing CapTTS systems.
>
---
#### [new 032] StarVC: A Unified Auto-Regressive Framework for Joint Text and Speech Generation in Voice Conversion
- **分类: cs.MM; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音转换任务，旨在解决传统方法忽略语言内容建模的问题。作者提出StarVC框架，先预测文本再合成语音，更好分离说话人身份与语言内容，提升转换效果。**

- **链接: [http://arxiv.org/pdf/2506.02414v1](http://arxiv.org/pdf/2506.02414v1)**

> **作者:** Fengjin Li; Jie Wang; Yadong Niu; Yongqing Wang; Meng Meng; Jian Luan; Zhiyong Wu
>
> **备注:** 5 pages, 2 figures, Accepted by Interspeech 2025, Demo: https://thuhcsi.github.io/StarVC/
>
> **摘要:** Voice Conversion (VC) modifies speech to match a target speaker while preserving linguistic content. Traditional methods usually extract speaker information directly from speech while neglecting the explicit utilization of linguistic content. Since VC fundamentally involves disentangling speaker identity from linguistic content, leveraging structured semantic features could enhance conversion performance. However, previous attempts to incorporate semantic features into VC have shown limited effectiveness, motivating the integration of explicit text modeling. We propose StarVC, a unified autoregressive VC framework that first predicts text tokens before synthesizing acoustic features. The experiments demonstrate that StarVC outperforms conventional VC methods in preserving both linguistic content (i.e., WER and CER) and speaker characteristics (i.e., SECS and MOS). Audio demo can be found at: https://thuhcsi.github.io/StarVC/.
>
---
#### [new 033] On the influence of language similarity in non-target speaker verification trials
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究了非目标说话人验证中语言相似性的影响，属于说话人验证任务。通过分析不同语言组合的评分分布，探讨语言相似性对验证效果的作用，特别是在训练未见语言上的影响。使用ECAPA-TDNN系统，在多语言数据上进行了实验。**

- **链接: [http://arxiv.org/pdf/2506.02777v1](http://arxiv.org/pdf/2506.02777v1)**

> **作者:** Paul M. Reuter; Michael Jessen
>
> **备注:** accepted to Interspeech 2025
>
> **摘要:** In this paper, we investigate the influence of language similarity in cross-lingual non-target speaker verification trials using a state-of-the-art speaker verification system, ECAPA-TDNN, trained on multilingual and monolingual variants of the VoxCeleb dataset. Our analysis of the score distribution patterns on multilingual Globalphone and LDC CTS reveals a clustering effect in speaker comparisons involving a training language, whereby the choice of comparison language only minimally impacts scores. Conversely, we observe a language similarity effect in trials involving languages not included in the training set of the speaker verification system, with scores correlating with language similarity measured by a language classification system, especially when using multilingual training data.
>
---
#### [new 034] Inter(sectional) Alia(s): Ambiguity in Voice Agent Identity via Intersectional Japanese Self-Referents
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.SD; eess.AS**

- **简介: 该论文研究了日本语境中语音代理的身份模糊性问题，特别是通过交叉性视角分析日语自我指称词对身份感知的影响。任务是探索语音代理在性别、年龄和正式程度等社会身份维度上的认知效果。工作包括招募204名日本参与者，评估三种ChatGPT语音与七种自我指称词的组合表现。结果揭示了语音性别化现象及某些自指词的去性别化潜力，强调文化敏感性和交叉性分析的重要性。**

- **链接: [http://arxiv.org/pdf/2506.01998v1](http://arxiv.org/pdf/2506.01998v1)**

> **作者:** Takao Fujii; Katie Seaborn; Madeleine Steeds; Jun Kato
>
> **备注:** CHI '25
>
> **摘要:** Conversational agents that mimic people have raised questions about the ethics of anthropomorphizing machines with human social identity cues. Critics have also questioned assumptions of identity neutrality in humanlike agents. Recent work has revealed that intersectional Japanese pronouns can elicit complex and sometimes evasive impressions of agent identity. Yet, the role of other "neutral" non-pronominal self-referents (NPSR) and voice as a socially expressive medium remains unexplored. In a crowdsourcing study, Japanese participants (N = 204) evaluated three ChatGPT voices (Juniper, Breeze, and Ember) using seven self-referents. We found strong evidence of voice gendering alongside the potential of intersectional self-referents to evade gendering, i.e., ambiguity through neutrality and elusiveness. Notably, perceptions of age and formality intersected with gendering as per sociolinguistic theories, especially boku and watakushi. This work provides a nuanced take on agent identity perceptions and champions intersectional and culturally-sensitive work on voice agents.
>
---
#### [new 035] PartialEdit: Identifying Partial Deepfakes in the Era of Neural Speech Editing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音伪造检测任务，旨在解决神经语音编辑技术带来的部分伪造语音难以检测的问题。作者构建了PartialEdit数据集，并验证现有模型对此类深伪语音的检测能力不足，同时探讨了检测方法与神经音频编解码器相关的潜在特征。**

- **链接: [http://arxiv.org/pdf/2506.02958v1](http://arxiv.org/pdf/2506.02958v1)**

> **作者:** You Zhang; Baotong Tian; Lin Zhang; Zhiyao Duan
>
> **备注:** Interspeech 2025 camera ready. Project page: https://yzyouzhang.com/PartialEdit/
>
> **摘要:** Neural speech editing enables seamless partial edits to speech utterances, allowing modifications to selected content while preserving the rest of the audio unchanged. This useful technique, however, also poses new risks of deepfakes. To encourage research on detecting such partially edited deepfake speech, we introduce PartialEdit, a deepfake speech dataset curated using advanced neural editing techniques. We explore both detection and localization tasks on PartialEdit. Our experiments reveal that models trained on the existing PartialSpoof dataset fail to detect partially edited speech generated by neural speech editing models. As recent speech editing models almost all involve neural audio codecs, we also provide insights into the artifacts the model learned on detecting these deepfakes. Further information about the PartialEdit dataset and audio samples can be found on the project page: https://yzyouzhang.com/PartialEdit/index.html.
>
---
#### [new 036] DYNAC: Dynamic Vocabulary based Non-Autoregressive Contextualization for Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文属于语音识别任务，旨在解决上下文偏置建模中推理速度慢和动态词汇依赖建模不足的问题。工作提出DYNAC方法，将动态词汇集成到CTC模型中间层，实现编码器自条件化，有效捕捉静态与动态token依赖，显著提升推理效率。**

- **链接: [http://arxiv.org/pdf/2506.00422v1](http://arxiv.org/pdf/2506.00422v1)**

> **作者:** Yui Sudo; Yosuke Fukumoto; Muhammad Shakeel; Yifan Peng; Chyi-Jiunn Lin; Shinji Watanabe
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Contextual biasing (CB) improves automatic speech recognition for rare and unseen phrases. Recent studies have introduced dynamic vocabulary, which represents context phrases as expandable tokens in autoregressive (AR) models. This method improves CB accuracy but with slow inference speed. While dynamic vocabulary can be applied to non-autoregressive (NAR) models, such as connectionist temporal classification (CTC), the conditional independence assumption fails to capture dependencies between static and dynamic tokens. This paper proposes DYNAC (Dynamic Vocabulary-based NAR Contextualization), a self-conditioned CTC method that integrates dynamic vocabulary into intermediate layers. Conditioning the encoder on dynamic vocabulary, DYNAC effectively captures dependencies between static and dynamic tokens while reducing the real-time factor (RTF). Experimental results show that DYNAC reduces RTF by 81% with a 0.1-point degradation in word error rate on the LibriSpeech 960 test-clean set.
>
---
#### [new 037] Are Mamba-based Audio Foundation Models the Best Fit for Non-Verbal Emotion Recognition?
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究非语言情感识别任务，旨在解决现有模型对非语言情感特征捕捉不足的问题。论文首次将基于Mamba的音频基础模型（MAFMs）应用于此任务，并提出RENO方法融合MAFMs与AAFMs，提升情感识别效果，取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2506.02258v1](http://arxiv.org/pdf/2506.02258v1)**

> **作者:** Mohd Mujtaba Akhtar; Orchid Chetia Phukan; Girish; Swarup Ranjan Behera; Ananda Chandra Nayak; Sanjib Kumar Nayak; Arun Balaji Buduru; Rajesh Sharma
>
> **备注:** Accepted to EUSIPCO 2025
>
> **摘要:** In this work, we focus on non-verbal vocal sounds emotion recognition (NVER). We investigate mamba-based audio foundation models (MAFMs) for the first time for NVER and hypothesize that MAFMs will outperform attention-based audio foundation models (AAFMs) for NVER by leveraging its state-space modeling to capture intrinsic emotional structures more effectively. Unlike AAFMs, which may amplify irrelevant patterns due to their attention mechanisms, MAFMs will extract more stable and context-aware representations, enabling better differentiation of subtle non-verbal emotional cues. Our experiments with state-of-the-art (SOTA) AAFMs and MAFMs validates our hypothesis. Further, motivated from related research such as speech emotion recognition, synthetic speech detection, where fusion of foundation models (FMs) have showed improved performance, we also explore fusion of FMs for NVER. To this end, we propose, RENO, that uses renyi-divergence as a novel loss function for effective alignment of the FMs. It also makes use of self-attention for better intra-representation interaction of the FMs. With RENO, through the heterogeneous fusion of MAFMs and AAFMs, we show the topmost performance in comparison to individual FMs, its fusion and also setting SOTA in comparison to previous SOTA work.
>
---
#### [new 038] Adaptive Differential Denoising for Respiratory Sounds Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于呼吸音分类任务，旨在解决背景噪声干扰和去噪不足导致的分类性能下降问题。作者提出了自适应差分去噪网络（ADD-RSC），包含频谱掩码、差分注意力层和偏置去噪损失，有效抑制噪声并保留病理特征，分类性能提升1.99%。**

- **链接: [http://arxiv.org/pdf/2506.02505v1](http://arxiv.org/pdf/2506.02505v1)**

> **作者:** Gaoyang Dong; Zhicheng Zhang; Ping Sun; Minghui Zhang
>
> **备注:** accepted at Interspeech2025
>
> **摘要:** Automated respiratory sound classification faces practical challenges from background noise and insufficient denoising in existing systems. We propose Adaptive Differential Denoising network, that integrates noise suppression and pathological feature preservation via three innovations: 1) Adaptive Frequency Filter with learnable spectral masks and soft shrink to eliminate noise while retaining diagnostic high-frequency components; 2) A Differential Denoise Layer using differential attention to reduce noise-induced variations through augmented sample comparisons; 3) A bias denoising loss jointly optimizing classification and robustness without clean labels. Experiments on the ICBHI2017 dataset show that our method achieves 65.53\% of the Score, which is improved by 1.99\% over the previous sota method. The code is available in https://github.com/deegy666/ADD-RSC
>
---
#### [new 039] AuralNet: Hierarchical Attention-based 3D Binaural Localization of Overlapping Speakers
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于3D声音定位任务，旨在解决混叠说话人在复杂环境中的方位和仰角定位问题。作者提出了AuralNet模型，采用分层注意力机制与粗到精结构，实现无需先验源数的精准定位，并设计多任务损失函数优化检测与角度估计。**

- **链接: [http://arxiv.org/pdf/2506.02773v1](http://arxiv.org/pdf/2506.02773v1)**

> **作者:** Linya Fu; Yu Liu; Zhijie Liu; Zedong Yang; Zhong-Qiu Wang; Youfu Li; He Kong
>
> **备注:** Accepted and to appear at Interspeech 2025
>
> **摘要:** We propose AuralNet, a novel 3D multi-source binaural sound source localization approach that localizes overlapping sources in both azimuth and elevation without prior knowledge of the number of sources. AuralNet employs a gated coarse-tofine architecture, combining a coarse classification stage with a fine-grained regression stage, allowing for flexible spatial resolution through sector partitioning. The model incorporates a multi-head self-attention mechanism to capture spatial cues in binaural signals, enhancing robustness in noisy-reverberant environments. A masked multi-task loss function is designed to jointly optimize sound detection, azimuth, and elevation estimation. Extensive experiments in noisy-reverberant conditions demonstrate the superiority of AuralNet over recent methods
>
---
## 更新

#### [replaced 001] Ola: Pushing the Frontiers of Omni-Modal Language Model
- **分类: cs.CV; cs.CL; cs.MM; cs.SD; eess.AS; eess.IV**

- **链接: [http://arxiv.org/pdf/2502.04328v3](http://arxiv.org/pdf/2502.04328v3)**

> **作者:** Zuyan Liu; Yuhao Dong; Jiahui Wang; Ziwei Liu; Winston Hu; Jiwen Lu; Yongming Rao
>
> **摘要:** Recent advances in large language models, particularly following GPT-4o, have sparked increasing interest in developing omni-modal models capable of understanding more modalities. While some open-source alternatives have emerged, there is still a notable lag behind specialized single-modality models in performance. In this paper, we present Ola, an Omni-modal Language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts, pushing the frontiers of the omni-modal language model to a large extent. We conduct a comprehensive exploration of architectural design, data curation, and training strategies essential for building a robust omni-modal model. Ola incorporates advanced visual understanding and audio recognition capabilities through several critical and effective improvements over mainstream baselines. Moreover, we rethink inter-modal relationships during omni-modal training, emphasizing cross-modal alignment with video as a central bridge, and propose a progressive training pipeline that begins with the most distinct modalities and gradually moves towards closer modality alignment. Extensive experiments demonstrate that Ola surpasses existing open omni-modal LLMs across all modalities while achieving highly competitive performance compared to state-of-the-art specialized models of similar sizes. We aim to make Ola a fully open omni-modal understanding solution to advance future research in this emerging field. Model weights, code, and data are open-sourced at https://github.com/Ola-Omni/Ola.
>
---
#### [replaced 002] OmniAudio: Generating Spatial Audio from 360-Degree Video
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.14906v3](http://arxiv.org/pdf/2504.14906v3)**

> **作者:** Huadai Liu; Tianyi Luo; Kaicheng Luo; Qikai Jiang; Peiwen Sun; Jialei Wang; Rongjie Huang; Qian Chen; Wen Wang; Xiangtai Li; Shiliang Zhang; Zhijie Yan; Zhou Zhao; Wei Xue
>
> **备注:** ICML 2025
>
> **摘要:** Traditional video-to-audio generation techniques primarily focus on perspective video and non-spatial audio, often missing the spatial cues necessary for accurately representing sound sources in 3D environments. To address this limitation, we introduce a novel task, 360V2SA, to generate spatial audio from 360-degree videos, specifically producing First-order Ambisonics (FOA) audio - a standard format for representing 3D spatial audio that captures sound directionality and enables realistic 3D audio reproduction. We first create Sphere360, a novel dataset tailored for this task that is curated from real-world data. We also design an efficient semi-automated pipeline for collecting and cleaning paired video-audio data. To generate spatial audio from 360-degree video, we propose a novel framework OmniAudio, which leverages self-supervised pre-training using both spatial audio data (in FOA format) and large-scale non-spatial data. Furthermore, OmniAudio features a dual-branch framework that utilizes both panoramic and perspective video inputs to capture comprehensive local and global information from 360-degree videos. Experimental results demonstrate that OmniAudio achieves state-of-the-art performance across both objective and subjective metrics on Sphere360. Code and datasets are available at https://github.com/liuhuadai/OmniAudio. The project website is available at https://OmniAudio-360V2SA.github.io.
>
---
#### [replaced 003] Improving Transformer Performance for French Clinical Notes Classification Using Mixture of Experts on a Limited Dataset
- **分类: cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2303.12892v3](http://arxiv.org/pdf/2303.12892v3)**

> **作者:** Thanh-Dung Le; Philippe Jouvet; Rita Noumeir
>
> **备注:** Accepted for publication in the IEEE Journal of Translational Engineering in Health and Medicine
>
> **摘要:** Transformer-based models have shown outstanding results in natural language processing but face challenges in applications like classifying small-scale clinical texts, especially with constrained computational resources. This study presents a customized Mixture of Expert (MoE) Transformer models for classifying small-scale French clinical texts at CHU Sainte-Justine Hospital. The MoE-Transformer addresses the dual challenges of effective training with limited data and low-resource computation suitable for in-house hospital use. Despite the success of biomedical pre-trained models such as CamemBERT-bio, DrBERT, and AliBERT, their high computational demands make them impractical for many clinical settings. Our MoE-Transformer model not only outperforms DistillBERT, CamemBERT, FlauBERT, and Transformer models on the same dataset but also achieves impressive results: an accuracy of 87\%, precision of 87\%, recall of 85\%, and F1-score of 86\%. While the MoE-Transformer does not surpass the performance of biomedical pre-trained BERT models, it can be trained at least 190 times faster, offering a viable alternative for settings with limited data and computational resources. Although the MoE-Transformer addresses challenges of generalization gaps and sharp minima, demonstrating some limitations for efficient and accurate clinical text classification, this model still represents a significant advancement in the field. It is particularly valuable for classifying small French clinical narratives within the privacy and constraints of hospital-based computational resources.
>
---
#### [replaced 004] Unsupervised Time-Series Signal Analysis with Autoencoders and Vision Transformers: A Review of Architectures and Applications
- **分类: cs.LG; cs.AI; cs.CV; eess.SP; I.5.4; I.2.6; H.2.8**

- **链接: [http://arxiv.org/pdf/2504.16972v2](http://arxiv.org/pdf/2504.16972v2)**

> **作者:** Hossein Ahmadi; Sajjad Emdadi Mahdimahalleh; Arman Farahat; Banafsheh Saffari
>
> **摘要:** The rapid growth of unlabeled time-series data in domains such as wireless communications, radar, biomedical engineering, and the Internet of Things (IoT) has driven advancements in unsupervised learning. This review synthesizes recent progress in applying autoencoders and vision transformers for unsupervised signal analysis, focusing on their architectures, applications, and emerging trends. We explore how these models enable feature extraction, anomaly detection, and classification across diverse signal types, including electrocardiograms, radar waveforms, and IoT sensor data. The review highlights the strengths of hybrid architectures and self-supervised learning, while identifying challenges in interpretability, scalability, and domain generalization. By bridging methodological innovations and practical applications, this work offers a roadmap for developing robust, adaptive models for signal intelligence.
>
---
#### [replaced 005] Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15380v2](http://arxiv.org/pdf/2505.15380v2)**

> **作者:** Zijian Lin; Yang Zhang; Yougen Yuan; Yuming Yan; Jinjiang Liu; Zhiyong Wu; Pengfei Hu; Qun Yu
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Modern autoregressive speech synthesis models leveraging language models have demonstrated remarkable performance. However, the sequential nature of next token prediction in these models leads to significant latency, hindering their deployment in scenarios where inference speed is critical. In this work, we propose Speech Speculative Decoding (SSD), a novel framework for autoregressive speech synthesis acceleration. Specifically, our method employs a lightweight draft model to generate candidate token sequences, which are subsequently verified in parallel by the target model using the proposed SSD framework. Experimental results demonstrate that SSD achieves a significant speedup of 1.4x compared with conventional autoregressive decoding, while maintaining high fidelity and naturalness. Subjective evaluations further validate the effectiveness of SSD in preserving the perceptual quality of the target model while accelerating inference.
>
---
#### [replaced 006] Effect of laboratory conditions on the perception of virtual stages for music
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.20552v2](http://arxiv.org/pdf/2505.20552v2)**

> **作者:** Ernesto Accolti
>
> **摘要:** This manuscript presents initial findings critical for supporting augmented acoustics experiments in custom-made hearing booths, addressing a key challenge in ensuring perceptual validity and experimental rigor in these highly sensitive setups. This validation ensures our proposed methodology is sound, guarantees the reliability of future results, and lays the foundational groundwork for subsequent perceptual studies and the development of robust guidelines for laboratory design in virtual acoustics research. A preliminary study on the effect of the acoustical conditions of three different rooms on the perception of virtual stages for music is presented: an anechoic room, a custom-made hearing booth with insufficient sound absorption, and another custom-made hearing booth with achievable sound absorption. The goal of this study is to assess the impact of these different conditions on the perception of virtual stages for music. The results show that the anechoic room and the hearing booth with achievable sound absorption have a difference between the total sound and the virtual sound below the just-noticeable difference, which means that the virtual sound is not perceived louder than it should. In contrast, the hearing booth with insufficient sound absorption has a difference above the just-noticeable difference, which means that the virtual sound is perceived louder than it should. This study provides a preliminary validation of the proposed methodology for assessing the acoustical conditions of custom-made hearing booths in stage acoustics experiments. Future work will include a more comprehensive analysis of the results, including the effect of different sound sources. Supplementary audio files illustrating key simulation results are available at https://zenodo.org/records/15579861
>
---
#### [replaced 007] Score-informed Music Source Separation: Improving Synthetic-to-real Generalization in Classical Music
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2503.07352v2](http://arxiv.org/pdf/2503.07352v2)**

> **作者:** Eetu Tunturi; David Diaz-Guerra; Archontis Politis; Tuomas Virtanen
>
> **备注:** 5 pages, 2 figures, accepted to EUSIPCO 2025
>
> **摘要:** Music source separation is the task of separating a mixture of instruments into constituent tracks. Music source separation models are typically trained using only audio data, although additional information can be used to improve the model's separation capability. In this paper, we propose two ways of using musical scores to aid music source separation: a score-informed model where the score is concatenated with the magnitude spectrogram of the audio mixture as the input of the model, and a model where we use only the score to calculate the separation mask. We train our models on synthetic data in the SynthSOD dataset and evaluate our methods on the URMP and Aalto anechoic orchestra datasets, comprised of real recordings. The score-informed model improves separation results compared to a baseline approach, but struggles to generalize from synthetic to real data, whereas the score-only model shows a clear improvement in synthetic-to-real generalization.
>
---
#### [replaced 008] FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.12266v2](http://arxiv.org/pdf/2410.12266v2)**

> **作者:** Huadai Liu; Jialei Wang; Rongjie Huang; Yang Liu; Heng Lu; Zhou Zhao; Wei Xue
>
> **备注:** ACL 2025 Main
>
> **摘要:** Recent advancements in latent diffusion models (LDMs) have markedly enhanced text-to-audio generation, yet their iterative sampling processes impose substantial computational demands, limiting practical deployment. While recent methods utilizing consistency-based distillation aim to achieve few-step or single-step inference, their one-step performance is constrained by curved trajectories, preventing them from surpassing traditional diffusion models. In this work, we introduce FlashAudio with rectified flows to learn straight flow for fast simulation. To alleviate the inefficient timesteps allocation and suboptimal distribution of noise, FlashAudio optimizes the time distribution of rectified flow with Bifocal Samplers and proposes immiscible flow to minimize the total distance of data-noise pairs in a batch vias assignment. Furthermore, to address the amplified accumulation error caused by the classifier-free guidance (CFG), we propose Anchored Optimization, which refines the guidance scale by anchoring it to a reference trajectory. Experimental results on text-to-audio generation demonstrate that FlashAudio's one-step generation performance surpasses the diffusion-based models with hundreds of sampling steps on audio quality and enables a sampling speed of 400x faster than real-time on a single NVIDIA 4090Ti GPU. Code will be available at https://github.com/liuhuadai/FlashAudio.
>
---
#### [replaced 009] Continual Speech Learning with Fused Speech Features
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.01496v2](http://arxiv.org/pdf/2506.01496v2)**

> **作者:** Guitao Wang; Jinming Zhao; Hao Yang; Guilin Qi; Tongtong Wu; Gholamreza Haffari
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Rapid growth in speech data demands adaptive models, as traditional static methods fail to keep pace with dynamic and diverse speech information. We introduce continuous speech learning, a new set-up targeting at bridging the adaptation gap in current speech models. We use the encoder-decoder Whisper model to standardize speech tasks into a generative format. We integrate a learnable gated-fusion layer on the top of the encoder to dynamically select task-specific features for downstream tasks. Our approach improves accuracy significantly over traditional methods in six speech processing tasks, demonstrating gains in adapting to new speech tasks without full retraining.
>
---
#### [replaced 010] Improving Multilingual Speech Models on ML-SUPERB 2.0: Fine-tuning with Data Augmentation and LID-Aware CTC
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.24200v2](http://arxiv.org/pdf/2505.24200v2)**

> **作者:** Qingzheng Wang; Jiancheng Sun; Yifan Peng; Shinji Watanabe
>
> **摘要:** Multilingual speech processing with self-supervised or supervised pre-trained Speech Foundation Models (SFM) has achieved strong performance on tasks like Language Identification (LID) and Automatic Speech Recognition (ASR). However, these models struggle with limited resources during fine-tuning. This paper enhances multilingual LID and ASR on ML-SUPERB 2.0 by exploring multiple strategies for adapting SFMs, including frozen upstream training, partial fine-tuning, and low-rank adaptation. Furthermore, we employ data augmentation to mitigate performance gaps in few-shot settings and introduce LID Connectionist Temporal Classification (CTC) loss for regularization. Our approach achieves a 14% relative improvement in LID accuracy and a 30% relative reduction in ASR CER over the baseline on ML-SUPERB 2.0, securing second place in the Interspeech 2025 ML-SUPERB 2.0 Challenge.
>
---
