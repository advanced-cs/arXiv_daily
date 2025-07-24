# 音频 cs.SD;  eess.SP

- **最新发布 9 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Application of Whisper in Clinical Practice: the Post-Stroke Speech Assessment during a Naming Task
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别与临床语言评估任务，旨在解决中风后语言障碍自动评估的问题。研究验证了Whisper模型在命名任务中对健康人和中风患者语音的转录准确性，并探索其在语言功能预测中的应用。结果显示，微调后模型转录准确率显著提升，能有效预测语言质量，但跨数据集泛化能力有限。**

- **链接: [http://arxiv.org/pdf/2507.17326v1](http://arxiv.org/pdf/2507.17326v1)**

> **作者:** Milena Davudova; Ziyuan Cai; Valentina Giunchiglia; Dragos C. Gruia; Giulia Sanguedolce; Adam Hampshire; Fatemeh Geranmayeh
>
> **摘要:** Detailed assessment of language impairment following stroke remains a cognitively complex and clinician-intensive task, limiting timely and scalable diagnosis. Automatic Speech Recognition (ASR) foundation models offer a promising pathway to augment human evaluation through intelligent systems, but their effectiveness in the context of speech and language impairment remains uncertain. In this study, we evaluate whether Whisper, a state-of-the-art ASR foundation model, can be applied to transcribe and analyze speech from patients with stroke during a commonly used picture-naming task. We assess both verbatim transcription accuracy and the model's ability to support downstream prediction of language function, which has major implications for outcomes after stroke. Our results show that the baseline Whisper model performs poorly on single-word speech utterances. Nevertheless, fine-tuning Whisper significantly improves transcription accuracy (reducing Word Error Rate by 87.72% in healthy speech and 71.22% in speech from patients). Further, learned representations from the model enable accurate prediction of speech quality (average F1 Macro of 0.74 for healthy, 0.75 for patients). However, evaluations on an unseen (TORGO) dataset reveal limited generalizability, highlighting the inability of Whisper to perform zero-shot transcription of single-word utterances on out-of-domain clinical speech and emphasizing the need to adapt models to specific clinical populations. While challenges remain in cross-domain generalization, these findings highlight the potential of foundation models, when appropriately fine-tuned, to advance automated speech and language assessment and rehabilitation for stroke-related impairments.
>
---
#### [new 002] Audio-Vision Contrastive Learning for Phonological Class Recognition
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于语音分类任务，旨在解决基于发音器官运动与语音信号的音系类别识别问题。作者提出了一种多模态深度学习框架，结合实时磁共振成像与语音信号，通过对比学习实现音视频融合，显著提升了分类性能。**

- **链接: [http://arxiv.org/pdf/2507.17682v1](http://arxiv.org/pdf/2507.17682v1)**

> **作者:** Daiqi Liu; Tomás Arias-Vergara; Jana Hutter; Andreas Maier; Paula Andrea Pérez-Toro
>
> **备注:** conference to TSD 2025
>
> **摘要:** Accurate classification of articulatory-phonological features plays a vital role in understanding human speech production and developing robust speech technologies, particularly in clinical contexts where targeted phonemic analysis and therapy can improve disease diagnosis accuracy and personalized rehabilitation. In this work, we propose a multimodal deep learning framework that combines real-time magnetic resonance imaging (rtMRI) and speech signals to classify three key articulatory dimensions: manner of articulation, place of articulation, and voicing. We perform classification on 15 phonological classes derived from the aforementioned articulatory dimensions and evaluate the system with four audio/vision configurations: unimodal rtMRI, unimodal audio signals, multimodal middle fusion, and contrastive learning-based audio-vision fusion. Experimental results on the USC-TIMIT dataset show that our contrastive learning-based approach achieves state-of-the-art performance, with an average F1-score of 0.81, representing an absolute increase of 0.23 over the unimodal baseline. The results confirm the effectiveness of contrastive representation learning for multimodal articulatory analysis. Our code and processed dataset will be made publicly available at https://github.com/DaE-plz/AC_Contrastive_Phonology to support future research.
>
---
#### [new 003] On Temporal Guidance and Iterative Refinement in Audio Source Separation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频源分离任务，旨在提升复杂声学场景中声源分离的精度。传统方法因缺乏细粒度时间信息受限，本文提出结合事件检测与迭代优化的新方法，有效提升分离效果，并在DCASE 2025挑战赛中取得第二名。**

- **链接: [http://arxiv.org/pdf/2507.17297v1](http://arxiv.org/pdf/2507.17297v1)**

> **作者:** Tobias Morocutti; Jonathan Greif; Paul Primus; Florian Schmid; Gerhard Widmer
>
> **摘要:** Spatial semantic segmentation of sound scenes (S5) involves the accurate identification of active sound classes and the precise separation of their sources from complex acoustic mixtures. Conventional systems rely on a two-stage pipeline - audio tagging followed by label-conditioned source separation - but are often constrained by the absence of fine-grained temporal information critical for effective separation. In this work, we address this limitation by introducing a novel approach for S5 that enhances the synergy between the event detection and source separation stages. Our key contributions are threefold. First, we fine-tune a pre-trained Transformer to detect active sound classes. Second, we utilize a separate instance of this fine-tuned Transformer to perform sound event detection (SED), providing the separation module with detailed, time-varying guidance. Third, we implement an iterative refinement mechanism that progressively enhances separation quality by recursively reusing the separator's output from previous iterations. These advancements lead to significant improvements in both audio tagging and source separation performance, as demonstrated by our system's second-place finish in Task 4 of the DCASE Challenge 2025. Our implementation and model checkpoints are available in our GitHub repository: https://github.com/theMoro/dcase25task4 .
>
---
#### [new 004] Weak Supervision Techniques towards Enhanced ASR Models in Industry-level CRM Systems
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决通用语音识别模型难以适应工业CRM系统中行业特定语音识别需求的问题。作者提出了一种微调行业专用ASR模型的方法，显著提升了其在实际工业应用中的性能，并已落地应用。**

- **链接: [http://arxiv.org/pdf/2507.16843v1](http://arxiv.org/pdf/2507.16843v1)**

> **作者:** Zhongsheng Wang; Sijie Wang; Jia Wang; Yung-I Liang; Yuxi Zhang; Jiamou Liu
>
> **备注:** Accepted by ICONIP 2024
>
> **摘要:** In the design of customer relationship management (CRM) systems, accurately identifying customer types and offering personalized services are key to enhancing customer satisfaction and loyalty. However, this process faces the challenge of discerning customer voices and intentions, and general pre-trained automatic speech recognition (ASR) models make it difficult to effectively address industry-specific speech recognition tasks. To address this issue, we innovatively proposed a solution for fine-tuning industry-specific ASR models, which significantly improved the performance of the fine-tuned ASR models in industry applications. Experimental results show that our method substantially improves the crucial auxiliary role of the ASR model in industry CRM systems, and this approach has also been adopted in actual industrial applications.
>
---
#### [new 005] BoSS: Beyond-Semantic Speech
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音技术任务，旨在解决当前语音系统无法捕捉超越语义的隐含信息问题。提出了“超越语义语音”（BoSS）概念，构建了语音交互系统能力等级框架（L1-L5），并基于认知相关理论和机器学习分析语音的多维特征，揭示了当前模型在理解超越语义信号上的不足。**

- **链接: [http://arxiv.org/pdf/2507.17563v1](http://arxiv.org/pdf/2507.17563v1)**

> **作者:** Qing Wang; Zehan Li; Hang Lv; Hongjie Chen; Yaodong Song; Jian Kang; Jie Lian; Jie Li; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Human communication involves more than explicit semantics, with implicit signals and contextual cues playing a critical role in shaping meaning. However, modern speech technologies, such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) often fail to capture these beyond-semantic dimensions. To better characterize and benchmark the progression of speech intelligence, we introduce Spoken Interaction System Capability Levels (L1-L5), a hierarchical framework illustrated the evolution of spoken dialogue systems from basic command recognition to human-like social interaction. To support these advanced capabilities, we propose Beyond-Semantic Speech (BoSS), which refers to the set of information in speech communication that encompasses but transcends explicit semantics. It conveys emotions, contexts, and modifies or extends meanings through multidimensional features such as affective cues, contextual dynamics, and implicit semantics, thereby enhancing the understanding of communicative intentions and scenarios. We present a formalized framework for BoSS, leveraging cognitive relevance theories and machine learning models to analyze temporal and contextual speech dynamics. We evaluate BoSS-related attributes across five different dimensions, reveals that current spoken language models (SLMs) are hard to fully interpret beyond-semantic signals. These findings highlight the need for advancing BoSS research to enable richer, more context-aware human-machine communication.
>
---
#### [new 006] Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音到语音的同声传译任务，旨在解决现有自动同声传译系统在翻译质量、实时性、多说话人混淆和语音克隆延迟等方面的问题。作者提出了Seed-LiveInterpret 2.0，通过双工语音理解和生成框架、大规模预训练与强化学习，实现了低延迟、高保真和语音克隆的端到端同声传译系统。**

- **链接: [http://arxiv.org/pdf/2507.17527v1](http://arxiv.org/pdf/2507.17527v1)**

> **作者:** Shanbo Cheng; Yu Bao; Zhichao Huang; Yu Lu; Ningxin Peng; Lu Xu; Runsheng Yu; Rong Cao; Ting Han; Zeyang Li; Sitong Liu; Shengtao Ma; Shiguang Pan; Jiongchen Xiao; Nuo Xu; Meng Yang; Rong Ye; Yiming Yu; Ruofei Zhang; Wanyi Zhang; Wenhao Zhu; Liehao Zou; Lu Lu; Yuxuan Wang; Yonghui Wu
>
> **备注:** Seed-LiveInterpret 2.0 Technical Report
>
> **摘要:** Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
>
---
#### [new 007] Clustering-based hard negative sampling for supervised contrastive speaker verification
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 论文属于说话人验证任务，旨在解决对比学习中难负样本选取的问题。提出CHNS方法，通过聚类相似说话人调整训练批次中的难易负样本比例，提升模型性能。实验表明其在VoxCeleb数据集上优于现有对比学习和分类方法。**

- **链接: [http://arxiv.org/pdf/2507.17540v1](http://arxiv.org/pdf/2507.17540v1)**

> **作者:** Piotr Masztalski; Michał Romaniuk; Jakub Żak; Mateusz Matuszewski; Konrad Kowalczyk
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** In speaker verification, contrastive learning is gaining popularity as an alternative to the traditionally used classification-based approaches. Contrastive methods can benefit from an effective use of hard negative pairs, which are different-class samples particularly challenging for a verification model due to their similarity. In this paper, we propose CHNS - a clustering-based hard negative sampling method, dedicated for supervised contrastive speaker representation learning. Our approach clusters embeddings of similar speakers, and adjusts batch composition to obtain an optimal ratio of hard and easy negatives during contrastive loss calculation. Experimental evaluation shows that CHNS outperforms a baseline supervised contrastive approach with and without loss-based hard negative sampling, as well as a state-of-the-art classification-based approach to speaker verification by as much as 18 % relative EER and minDCF on the VoxCeleb dataset using two lightweight model architectures.
>
---
#### [new 008] Enhancing Lung Disease Diagnosis via Semi-Supervised Machine Learning
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于医疗信号检测任务，旨在解决肺部疾病诊断中依赖大量手动标注数据的问题。通过引入半监督学习方法（如Mix Match、Co-Refinement等），结合MFCC+CNN模型，提升了肺音信号检测准确率至92.9%，减少了对标注数据的依赖，提高了诊断效率。**

- **链接: [http://arxiv.org/pdf/2507.16845v1](http://arxiv.org/pdf/2507.16845v1)**

> **作者:** Xiaoran Xua; In-Ho Rab; Ravi Sankarc
>
> **摘要:** Lung diseases, including lung cancer and COPD, are significant health concerns globally. Traditional diagnostic methods can be costly, time-consuming, and invasive. This study investigates the use of semi supervised learning methods for lung sound signal detection using a model combination of MFCC+CNN. By introducing semi supervised learning modules such as Mix Match, Co-Refinement, and Co Refurbishing, we aim to enhance the detection performance while reducing dependence on manual annotations. With the add-on semi-supervised modules, the accuracy rate of the MFCC+CNN model is 92.9%, an increase of 3.8% to the baseline model. The research contributes to the field of lung disease sound detection by addressing challenges such as individual differences, feature insufficient labeled data.
>
---
#### [new 009] Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音处理任务，旨在解决非母语口音语音转换为母语口音的问题。作者提出了一种基于自监督离散token和非平行数据的新方法，通过提取语音token并转换，再利用流匹配合成语音，在保持说话人身份的同时有效降低口音程度，提升自然度。**

- **链接: [http://arxiv.org/pdf/2507.17735v1](http://arxiv.org/pdf/2507.17735v1)**

> **作者:** Qibing Bai; Sho Inoue; Shuai Wang; Zhongjie Jiang; Yannan Wang; Haizhou Li
>
> **备注:** Accepted to INTERSPEECH 2025
>
> **摘要:** Accent normalization converts foreign-accented speech into native-like speech while preserving speaker identity. We propose a novel pipeline using self-supervised discrete tokens and non-parallel training data. The system extracts tokens from source speech, converts them through a dedicated model, and synthesizes the output using flow matching. Our method demonstrates superior performance over a frame-to-frame baseline in naturalness, accentedness reduction, and timbre preservation across multiple English accents. Through token-level phonetic analysis, we validate the effectiveness of our token-based approach. We also develop two duration preservation methods, suitable for applications such as dubbing.
>
---
## 更新

#### [replaced 001] Miipher-2: A Universal Speech Restoration Model for Million-Hour Scale Data Restoration
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.04457v4](http://arxiv.org/pdf/2505.04457v4)**

> **作者:** Shigeki Karita; Yuma Koizumi; Heiga Zen; Haruko Ishikawa; Robin Scheibler; Michiel Bacchiani
>
> **备注:** Accepted to IEEE WASPAA2025
>
> **摘要:** Training data cleaning is a new application for generative model-based speech restoration (SR). This paper introduces Miipher-2, an SR model designed for million-hour scale data, for training data cleaning for large-scale generative models like large language models. Key challenges addressed include generalization to unseen languages, operation without explicit conditioning (e.g., text, speaker ID), and computational efficiency. Miipher-2 utilizes a frozen, pre-trained Universal Speech Model (USM), supporting over 300 languages, as a robust, conditioning-free feature extractor. To optimize efficiency and minimize memory, Miipher-2 incorporates parallel adapters for predicting clean USM features from noisy inputs and employs the WaveFit neural vocoder for waveform synthesis. These components were trained on 3,000 hours of multi-lingual, studio-quality recordings with augmented degradations, while USM parameters remained fixed. Experimental results demonstrate Miipher-2's superior or comparable performance to conventional SR models in word-error-rate, speaker similarity, and both objective and subjective sound quality scores across all tested languages. Miipher-2 operates efficiently on consumer-grade accelerators, achieving a real-time factor of 0.0078, enabling the processing of a million-hour speech dataset in approximately three days using only 100 such accelerators.
>
---
#### [replaced 002] Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.14534v2](http://arxiv.org/pdf/2507.14534v2)**

> **作者:** Yu Zhang; Baotong Tian; Zhiyao Duan
>
> **摘要:** Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at https://aaronz345.github.io/ConanDemo.
>
---
#### [replaced 003] UniCUE: Unified Recognition and Generation Framework for Chinese Cued Speech Video-to-Speech Generation
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04134v2](http://arxiv.org/pdf/2506.04134v2)**

> **作者:** Jinting Wang; Shan Yang; Li Liu
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Cued Speech (CS) enhances lipreading through hand coding, providing precise speech perception support for the hearing-impaired. CS Video-to-Speech generation (CSV2S) task aims to convert the CS visual expressions (CS videos) of hearing-impaired individuals into comprehensible speech signals. Direct generation of speech from CS video (called single CSV2S) yields poor performance due to insufficient CS data. Current research mostly focuses on CS Recognition (CSR), which convert video content into linguistic text. Based on this, one straightforward way of CSV2S is to combine CSR with a Text-to-Speech system. This combined architecture relies on text as an intermediate medium for stepwise cross-modal alignment, which may lead to error propagation and temporal misalignment between speech and video dynamics. To address these challenges, we propose a novel approach that directly generates speech from CS videos without relying on intermediate text. Building upon this, we propose UniCUE, the first unified framework for CSV2S, whose core innovation lies in the integration of the CSR task that provides fine-grained visual-semantic information to facilitate speech generation from CS videos. More precisely, (1) a novel fine-grained semantic alignment pool to ensure precise mapping between visual features and speech contents; (2) a VisioPhonetic adapter to bridge cross-task representations, ensuring seamless compatibility between two distinct tasks (i.e., CSV2S and CSR); (3) a pose-aware visual processor is introduced to enhance fine-grained spatiotemporal correlations between lip and hand movements in CS video. Experiments on our new established Chinese CS dataset show that our UniCUE achieves state-of-the-art performance across various metrics.
>
---
#### [replaced 004] HiFi-Stream: Streaming Speech Enhancement with Generative Adversarial Networks
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.17141v2](http://arxiv.org/pdf/2503.17141v2)**

> **作者:** Ekaterina Dmitrieva; Maksim Kaledin
>
> **备注:** 5 pages (4 content pages + 1 page of references)
>
> **摘要:** Speech Enhancement techniques have become core technologies in mobile devices and voice software. Still, modern deep learning solutions often require high amount of computational resources what makes their usage on low-resource devices challenging. We present HiFi-Stream, an optimized version of recently published HiFi++ model. Our experiments demonstrate that HiFi-Stream saves most of the qualities of the original model despite its size and computational complexity improved in comparison to the original HiFi++ making it one of the smallest and fastest models available. The model is evaluated in streaming setting where it demonstrates its superior performance in comparison to modern baselines.
>
---
#### [replaced 005] Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.05236v2](http://arxiv.org/pdf/2502.05236v2)**

> **作者:** Shehzeen Hussain; Paarth Neekhara; Xuesong Yang; Edresson Casanova; Subhankar Ghosh; Mikyas T. Desta; Roy Fejgin; Rafael Valle; Jason Li
>
> **摘要:** While autoregressive speech token generation models produce speech with remarkable variety and naturalness, their inherent lack of controllability often results in issues such as hallucinations and undesired vocalizations that do not conform to conditioning inputs. We introduce Koel-TTS, a suite of enhanced encoder-decoder Transformer TTS models that address these challenges by incorporating preference alignment techniques guided by automatic speech recognition and speaker verification models. Additionally, we incorporate classifier-free guidance to further improve synthesis adherence to the transcript and reference speaker audio. Our experiments demonstrate that these optimizations significantly enhance target speaker similarity, intelligibility, and naturalness of synthesized speech. Notably, Koel-TTS directly maps text and context audio to acoustic tokens, and on the aforementioned metrics, outperforms state-of-the-art TTS models, despite being trained on a significantly smaller dataset. Audio samples and demos are available on our website.
>
---
#### [replaced 006] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography
- **分类: eess.IV; cs.CV; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.10876v4](http://arxiv.org/pdf/2409.10876v4)**

> **作者:** Tianao Li; Manxiu Cui; Cheng Ma; Emma Alexander
>
> **备注:** Accepted to IEEE/CVF International Conference on Computer Vision (ICCV), 2025
>
> **摘要:** Photoacoustic computed tomography (PACT) is a non-invasive imaging modality, similar to ultrasound, with wide-ranging medical applications. Conventional PACT images are degraded by wavefront distortion caused by the heterogeneous speed of sound (SOS) in tissue. Accounting for these effects can improve image quality and provide medically useful information, but measuring the SOS directly is burdensome and the existing joint reconstruction method is computationally expensive. Traditional supervised learning techniques are currently inaccessible in this data-starved domain. In this work, we introduce an efficient, self-supervised joint reconstruction method that recovers SOS and high-quality images for ring array PACT systems. To solve this semi-blind inverse problem, we parametrize the SOS using either a pixel grid or a neural field (NF) and update it directly by backpropagating the gradients through a differentiable imaging forward model. Our method removes SOS aberrations more accurately and 35x faster than the current SOTA. We demonstrate the success of our method quantitatively in simulation and qualitatively on experimentally-collected and in vivo data. Our code and synthetic numerical phantoms are available on our project page: https://lukeli0425.github.io/Coord-SoS-PACT/.
>
---
#### [replaced 007] ISDrama: Immersive Spatial Drama Generation through Multimodal Prompting
- **分类: eess.AS; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.20630v5](http://arxiv.org/pdf/2504.20630v5)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Zhiyuan Zhu; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ACM Multimedia 2025
>
> **摘要:** Multimodal immersive spatial drama generation focuses on creating continuous multi-speaker binaural speech with dramatic prosody based on multimodal prompts, with potential applications in AR, VR, and others. This task requires simultaneous modeling of spatial information and dramatic prosody based on multimodal inputs, with high data collection costs. To the best of our knowledge, our work is the first attempt to address these challenges. We construct MRSDrama, the first multimodal recorded spatial drama dataset, containing binaural drama audios, scripts, videos, geometric poses, and textual prompts. Then, we propose ISDrama, the first immersive spatial drama generation model through multimodal prompting. ISDrama comprises these primary components: 1) Multimodal Pose Encoder, based on contrastive learning, considering the Doppler effect caused by moving speakers to extract unified pose information from multimodal prompts. 2) Immersive Drama Transformer, a flow-based mamba-transformer model that generates high-quality drama, incorporating Drama-MOE to select proper experts for enhanced prosody and pose control. We also design a context-consistent classifier-free guidance strategy to coherently generate complete drama. Experimental results show that ISDrama outperforms baseline models on objective and subjective metrics. The demos are available at https://aaronz345.github.io/ISDramaDemo. We provide the dataset and the evaluation code at https://huggingface.co/datasets/AaronZ345/MRSDrama and https://github.com/AaronZ345/ISDrama.
>
---
