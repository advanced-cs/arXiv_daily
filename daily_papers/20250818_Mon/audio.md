# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 4 篇**

## 最新发布

#### [new 001] LD-LAudio-V1: Video-to-Long-Form-Audio Generation Extension with Dual Lightweight Adapters
- **分类: cs.SD; cs.AI; cs.CV; eess.AS**

- **简介: 该论文属于视频到长音频生成任务，解决长时音频同步与质量问题，提出LD-LAudio-V1模型，采用双轻量适配器提升生成效果。**

- **链接: [http://arxiv.org/pdf/2508.11074v1](http://arxiv.org/pdf/2508.11074v1)**

> **作者:** Haomin Zhang; Kristin Qi; Shuxin Yang; Zihao Chen; Chaofan Ding; Xinhan Di
>
> **备注:** Gen4AVC@ICCV: 1st Workshop on Generative AI for Audio-Visual Content Creation
>
> **摘要:** Generating high-quality and temporally synchronized audio from video content is essential for video editing and post-production tasks, enabling the creation of semantically aligned audio for silent videos. However, most existing approaches focus on short-form audio generation for video segments under 10 seconds or rely on noisy datasets for long-form video-to-audio zsynthesis. To address these limitations, we introduce LD-LAudio-V1, an extension of state-of-the-art video-to-audio models and it incorporates dual lightweight adapters to enable long-form audio generation. In addition, we release a clean and human-annotated video-to-audio dataset that contains pure sound effects without noise or artifacts. Our method significantly reduces splicing artifacts and temporal inconsistencies while maintaining computational efficiency. Compared to direct fine-tuning with short training videos, LD-LAudio-V1 achieves significant improvements across multiple metrics: $FD_{\text{passt}}$ 450.00 $\rightarrow$ 327.29 (+27.27%), $FD_{\text{panns}}$ 34.88 $\rightarrow$ 22.68 (+34.98%), $FD_{\text{vgg}}$ 3.75 $\rightarrow$ 1.28 (+65.87%), $KL_{\text{panns}}$ 2.49 $\rightarrow$ 2.07 (+16.87%), $KL_{\text{passt}}$ 1.78 $\rightarrow$ 1.53 (+14.04%), $IS_{\text{panns}}$ 4.17 $\rightarrow$ 4.30 (+3.12%), $IB_{\text{score}}$ 0.25 $\rightarrow$ 0.28 (+12.00%), $Energy\Delta10\text{ms}$ 0.3013 $\rightarrow$ 0.1349 (+55.23%), $Energy\Delta10\text{ms(vs.GT)}$ 0.0531 $\rightarrow$ 0.0288 (+45.76%), and $Sem.\,Rel.$ 2.73 $\rightarrow$ 3.28 (+20.15%). Our dataset aims to facilitate further research in long-form video-to-audio generation and is available at https://github.com/deepreasonings/long-form-video2audio.
>
---
#### [new 002] Mitigating Category Imbalance: Fosafer System for the Multimodal Emotion and Intent Joint Understanding Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多模态情感与意图联合理解任务，解决类别不平衡问题。通过数据增强、损失函数改进和模型微调等方法提升识别效果。**

- **链接: [http://arxiv.org/pdf/2508.11362v1](http://arxiv.org/pdf/2508.11362v1)**

> **作者:** Honghong Wang; Yankai Wang; Dejun Zhang; Jing Deng; Rong Zheng
>
> **备注:** 2 pages. pubilshed by ICASSP2025
>
> **摘要:** This paper presents Fosafer approach to the Track 2 Mandarin in the Multimodal Emotion and Intent Joint Understandingchallenge, which focuses on achieving joint recognition of emotion and intent in Mandarin, despite the issue of category imbalance. To alleviate this issue, we use a variety of data augmentation techniques across text, video, and audio modalities. Additionally, we introduce the SampleWeighted Focal Contrastive loss, designed to address the challenges of recognizing minority class samples and those that are semantically similar but difficult to distinguish. Moreover, we fine-tune the Hubert model to adapt the emotion and intent joint recognition. To mitigate modal competition, we introduce a modal dropout strategy. For the final predictions, a plurality voting approach is used to determine the results. The experimental results demonstrate the effectiveness of our method, which achieves the second-best performance in the Track 2 Mandarin challenge.
>
---
#### [new 003] Temporally-Similar Structure-Aware Spatiotemporal Fusion of Satellite Images
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于卫星图像时空融合任务，旨在解决噪声环境下结构信息丢失问题。通过引入TGTV和TGEC机制，提升融合效果与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.11259v1](http://arxiv.org/pdf/2508.11259v1)**

> **作者:** Ryosuke Isono; Shunsuke Ono
>
> **备注:** Submitted to IEEE Transactions on Geoscience and Remote Sensing. arXiv admin note: text overlap with arXiv:2308.00500
>
> **摘要:** This paper proposes a novel spatiotemporal (ST) fusion framework for satellite images, named Temporally-Similar Structure-Aware ST fusion (TSSTF). ST fusion is a promising approach to address the trade-off between the spatial and temporal resolution of satellite images. In real-world scenarios, observed satellite images are severely degraded by noise due to measurement equipment and environmental conditions. Consequently, some recent studies have focused on enhancing the robustness of ST fusion methods against noise. However, existing noise-robust ST fusion approaches often fail to capture fine spatial structure, leading to oversmoothing and artifacts. To address this issue, TSSTF introduces two key mechanisms: Temporally-Guided Total Variation (TGTV) and Temporally-Guided Edge Constraint (TGEC). TGTV is a novel regularization function that promotes spatial piecewise smoothness while preserving structural details, guided by a reference high spatial resolution image acquired on a nearby date. TGEC enforces consistency in edge locations between two temporally adjacent images, while allowing for spectral variations. We formulate the ST fusion task as a constrained optimization problem incorporating TGTV and TGEC, and develop an efficient algorithm based on a preconditioned primal-dual splitting method. Experimental results demonstrate that TSSTF performs comparably to state-of-the-art methods under noise-free conditions and outperforms them under noisy conditions. Additionally, we provide a comprehensive set of recommended parameter values that consistently yield high performance across diverse target regions and noise conditions, aiming to enhance reproducibility and practical utility.
>
---
#### [new 004] Perturbed Public Voices (P$^{2}$V): A Dataset for Robust Audio Deepfake Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有检测器在真实场景下失效的问题。通过构建P$^{2}$V数据集，研究了噪声和高级克隆技术对检测的影响，并验证了新模型的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.10949v1](http://arxiv.org/pdf/2508.10949v1)**

> **作者:** Chongyang Gao; Marco Postiglione; Isabel Gortner; Sarit Kraus; V. S. Subrahmanian
>
> **摘要:** Current audio deepfake detectors cannot be trusted. While they excel on controlled benchmarks, they fail when tested in the real world. We introduce Perturbed Public Voices (P$^{2}$V), an IRB-approved dataset capturing three critical aspects of malicious deepfakes: (1) identity-consistent transcripts via LLMs, (2) environmental and adversarial noise, and (3) state-of-the-art voice cloning (2020-2025). Experiments reveal alarming vulnerabilities of 22 recent audio deepfake detectors: models trained on current datasets lose 43% performance when tested on P$^{2}$V, with performance measured as the mean of F1 score on deepfake audio, AUC, and 1-EER. Simple adversarial perturbations induce up to 16% performance degradation, while advanced cloning techniques reduce detectability by 20-30%. In contrast, P$^{2}$V-trained models maintain robustness against these attacks while generalizing to existing datasets, establishing a new benchmark for robust audio deepfake detection. P$^{2}$V will be publicly released upon acceptance by a conference/journal.
>
---
#### [new 005] Benchmarking Prosody Encoding in Discrete Speech Tokens
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决离散语音标记在捕捉语气信息上的不足，通过分析其对人工修改语气的敏感性，提供设计指南。**

- **链接: [http://arxiv.org/pdf/2508.11224v1](http://arxiv.org/pdf/2508.11224v1)**

> **作者:** Kentaro Onda; Satoru Fukayama; Daisuke Saito; Nobuaki Minematsu
>
> **备注:** Accepted by ASRU2025
>
> **摘要:** Recently, discrete tokens derived from self-supervised learning (SSL) models via k-means clustering have been actively studied as pseudo-text in speech language models and as efficient intermediate representations for various tasks. However, these discrete tokens are typically learned in advance, separately from the training of language models or downstream tasks. As a result, choices related to discretization, such as the SSL model used or the number of clusters, must be made heuristically. In particular, speech language models are expected to understand and generate responses that reflect not only the semantic content but also prosodic features. Yet, there has been limited research on the ability of discrete tokens to capture prosodic information. To address this gap, this study conducts a comprehensive analysis focusing on prosodic encoding based on their sensitivity to the artificially modified prosody, aiming to provide practical guidelines for designing discrete tokens.
>
---
#### [new 006] Pretrained Conformers for Audio Fingerprinting and Retrieval
- **分类: cs.SD; cs.AI; cs.IR; eess.AS**

- **简介: 该论文属于音频指纹识别与检索任务，旨在解决小段音频的高效嵌入生成问题。通过自监督对比学习训练Conformer模型，实现高鲁棒性音频检索。**

- **链接: [http://arxiv.org/pdf/2508.11609v1](http://arxiv.org/pdf/2508.11609v1)**

> **作者:** Kemal Altwlkany; Elmedin Selmanovic; Sead Delalic
>
> **摘要:** Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes.
>
---
#### [new 007] Speech Emotion Recognition Using Fine-Tuned DWFormer:A Study on Track 1 of the IERPChallenge 2024
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在提升情绪预测精度。通过微调DWFormer模型并结合数据增强与评分融合策略，解决个体情感表达差异问题，取得Track 1第一名。**

- **链接: [http://arxiv.org/pdf/2508.11371v1](http://arxiv.org/pdf/2508.11371v1)**

> **作者:** Honghong Wang; Xupeng Jia; Jing Deng; Rong Zheng
>
> **备注:** 5 pages,1 figures
>
> **摘要:** The field of artificial intelligence has a strong interest in the topic of emotion recognition. The majority of extant emotion recognition models are oriented towards enhancing the precision of discrete emotion label prediction. Given the direct relationship between human personality and emotion, as well as the significant inter-individual differences in subjective emotional expression, the IERP Challenge 2024 incorporates personality traits into emotion recognition research. This paper presents the Fosafer submissions to the Track 1 of the IERP Challenge 2024. This task primarily concerns the recognition of emotions in audio, while also providing text and audio features. In Track 1, we utilized exclusively audio-based features and fine-tuned a pre-trained speech emotion recognition model, DWFormer, through the integration of data augmentation and score fusion strategies, thereby achieving the first place among the participating teams.
>
---
#### [new 008] CleanCTG: A Deep Learning Model for Multi-Artefact Detection and Reconstruction in Cardiotocography
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于医学信号处理任务，旨在解决CTG信号中的多类型伪影检测与重建问题。提出CleanCTG模型，实现端到端的伪影识别与修复，提升诊断准确性。**

- **链接: [http://arxiv.org/pdf/2508.10928v1](http://arxiv.org/pdf/2508.10928v1)**

> **作者:** Sheng Wong; Beth Albert; Gabriel Davis Jones
>
> **摘要:** Cardiotocography (CTG) is essential for fetal monitoring but is frequently compromised by diverse artefacts which obscure true fetal heart rate (FHR) patterns and can lead to misdiagnosis or delayed intervention. Current deep-learning approaches typically bypass comprehensive noise handling, applying minimal preprocessing or focusing solely on downstream classification, while traditional methods rely on simple interpolation or rule-based filtering that addresses only missing samples and fail to correct complex artefact types. We present CleanCTG, an end-to-end dual-stage model that first identifies multiple artefact types via multi-scale convolution and context-aware cross-attention, then reconstructs corrupted segments through artefact-specific correction branches. Training utilised over 800,000 minutes of physiologically realistic, synthetically corrupted CTGs derived from expert-verified "clean" recordings. On synthetic data, CleanCTG achieved perfect artefact detection (AU-ROC = 1.00) and reduced mean squared error (MSE) on corrupted segments to 2.74 x 10^-4 (clean-segment MSE = 2.40 x 10^-6), outperforming the next best method by more than 60%. External validation on 10,190 minutes of clinician-annotated segments yielded AU-ROC = 0.95 (sensitivity = 83.44%, specificity 94.22%), surpassing six comparator classifiers. Finally, when integrated with the Dawes-Redman system on 933 clinical CTG recordings, denoised traces increased specificity (from 80.70% to 82.70%) and shortened median time to decision by 33%. These findings suggest that explicit artefact removal and signal reconstruction can both maintain diagnostic accuracy and enable shorter monitoring sessions, offering a practical route to more reliable CTG interpretation.
>
---
#### [new 009] Novel Parasitic Dual-Scale Modeling for Efficient and Accurate Multilingual Speech Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言语音翻译任务，旨在解决统一模型参数大、效率低的问题。通过提出 Parasitic Dual-Scale 方法，提升推理速度与性能。**

- **链接: [http://arxiv.org/pdf/2508.11189v1](http://arxiv.org/pdf/2508.11189v1)**

> **作者:** Chenyang Le; Yinfeng Xia; Huiyan Li; Manhong Wang; Yutao Sun; Xingyang Ma; Yanmin Qian
>
> **备注:** Interspeech 2025
>
> **摘要:** Recent advancements in speech-to-text translation have led to the development of multilingual models capable of handling multiple language pairs simultaneously. However, these unified models often suffer from large parameter sizes, making it challenging to balance inference efficiency and performance, particularly in local deployment scenarios. We propose an innovative Parasitic Dual-Scale Approach, which combines an enhanced speculative sampling method with model compression and knowledge distillation techniques. Building on the Whisper Medium model, we enhance it for multilingual speech translation into whisperM2M, and integrate our novel KVSPN module, achieving state-of-the-art (SOTA) performance across six popular languages with improved inference efficiency. KVSPN enables a 40\% speedup with no BLEU score degradation. Combined with distillation methods, it represents a 2.6$\times$ speedup over the original Whisper Medium with superior performance.
>
---
#### [new 010] ASAudio: A Survey of Advanced Spatial Audio Research
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于空间音频领域的综述任务，旨在系统整理和分析相关研究方法与技术，解决现有研究缺乏全面总结的问题。**

- **链接: [http://arxiv.org/pdf/2508.10924v1](http://arxiv.org/pdf/2508.10924v1)**

> **作者:** Zhiyuan Zhu; Yu Zhang; Wenxiang Guo; Changhao Pan; Zhou Zhao
>
> **摘要:** With the rapid development of spatial audio technologies today, applications in AR, VR, and other scenarios have garnered extensive attention. Unlike traditional mono sound, spatial audio offers a more realistic and immersive auditory experience. Despite notable progress in the field, there remains a lack of comprehensive surveys that systematically organize and analyze these methods and their underlying technologies. In this paper, we provide a comprehensive overview of spatial audio and systematically review recent literature in the area. To address this, we chronologically outlining existing work related to spatial audio and categorize these studies based on input-output representations, as well as generation and understanding tasks, thereby summarizing various research aspects of spatial audio. In addition, we review related datasets, evaluation metrics, and benchmarks, offering insights from both training and evaluation perspectives. Related materials are available at https://github.com/dieKarotte/ASAudio.
>
---
#### [new 011] MoE-TTS: Enhancing Out-of-Domain Text Understanding for Description-based TTS via Mixture-of-Experts
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于文本到语音合成任务，解决模型在处理域外文本描述时理解能力不足的问题。通过引入混合专家架构增强预训练语言模型的文本理解能力。**

- **链接: [http://arxiv.org/pdf/2508.11326v1](http://arxiv.org/pdf/2508.11326v1)**

> **作者:** Heyang Xue; Xuchen Song; Yu Tang; Jianyu Chen; Yanru Chen; Yang Li; Yahui Zhou
>
> **摘要:** Description-based text-to-speech (TTS) models exhibit strong performance on in-domain text descriptions, i.e., those encountered during training. However, in real-world applications, the diverse range of user-generated descriptions inevitably introduces numerous out-of-domain inputs that challenge the text understanding capabilities of these systems. To address this issue, we propose MoE-TTS, a description-based TTS model designed to enhance the understanding of out-of-domain text descriptions. MoE-TTS employs a modality-based mixture-of-experts (MoE) approach to augment a pre-trained textual large language model (LLM) with a set of specialized weights adapted to the speech modality while maintaining the original LLM frozen during training. This approach allows MoE-TTS to effectively leverage the pre-trained knowledge and text understanding abilities of textual LLMs. Our experimental results indicate that: first, even the most advanced closed-source commercial products can be challenged by carefully designed out-of-domain description test sets; second, MoE-TTS achieves superior performance in generating speech that more accurately reflects the descriptions. We encourage readers to listen to the demos at https://welkinyang.github.io/MoE-TTS/.
>
---
#### [new 012] Representing Speech Through Autoregressive Prediction of Cochlear Tokens
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出AuriStream，一种基于听觉处理机制的语音表示学习模型，解决语音编码与语义理解问题，通过两阶段框架提取并生成语音特征。**

- **链接: [http://arxiv.org/pdf/2508.11598v1](http://arxiv.org/pdf/2508.11598v1)**

> **作者:** Greta Tuckute; Klemen Kotar; Evelina Fedorenko; Daniel L. K. Yamins
>
> **摘要:** We introduce AuriStream, a biologically inspired model for encoding speech via a two-stage framework inspired by the human auditory processing hierarchy. The first stage transforms raw audio into a time-frequency representation based on the human cochlea, from which we extract discrete \textbf{cochlear tokens}. The second stage applies an autoregressive sequence model over the cochlear tokens. AuriStream learns meaningful phoneme and word representations, and state-of-the-art lexical semantics. AuriStream shows competitive performance on diverse downstream SUPERB speech tasks. Complementing AuriStream's strong representational capabilities, it generates continuations of audio which can be visualized in a spectrogram space and decoded back into audio, providing insights into the model's predictions. In summary, we present a two-stage framework for speech representation learning to advance the development of more human-like models that efficiently handle a range of speech-based tasks.
>
---
#### [new 013] UWB-PostureGuard: A Privacy-Preserving RF Sensing System for Continuous Ergonomic Sitting Posture Monitoring
- **分类: cs.CV; cs.HC; eess.SP**

- **简介: 该论文属于姿态监测任务，解决传统方法隐私和舒适性问题。通过UWB技术实现无接触、持续的坐姿监控，提升健康管理水平。**

- **链接: [http://arxiv.org/pdf/2508.11115v1](http://arxiv.org/pdf/2508.11115v1)**

> **作者:** Haotang Li; Zhenyu Qi; Sen He; Kebin Peng; Sheng Tan; Yili Ren; Tomas Cerny; Jiyue Zhao; Zi Wang
>
> **摘要:** Improper sitting posture during prolonged computer use has become a significant public health concern. Traditional posture monitoring solutions face substantial barriers, including privacy concerns with camera-based systems and user discomfort with wearable sensors. This paper presents UWB-PostureGuard, a privacy-preserving ultra-wideband (UWB) sensing system that advances mobile technologies for preventive health management through continuous, contactless monitoring of ergonomic sitting posture. Our system leverages commercial UWB devices, utilizing comprehensive feature engineering to extract multiple ergonomic sitting posture features. We develop PoseGBDT to effectively capture temporal dependencies in posture patterns, addressing limitations of traditional frame-wise classification approaches. Extensive real-world evaluation across 10 participants and 19 distinct postures demonstrates exceptional performance, achieving 99.11% accuracy while maintaining robustness against environmental variables such as clothing thickness, additional devices, and furniture configurations. Our system provides a scalable, privacy-preserving mobile health solution on existing platforms for proactive ergonomic management, improving quality of life at low costs.
>
---
#### [new 014] Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音检索任务，解决根据自然语言描述的说话风格检索语音的问题。通过训练语音和文本编码器，将两者映射到共同空间，实现基于风格描述的语音检索。**

- **链接: [http://arxiv.org/pdf/2508.11187v1](http://arxiv.org/pdf/2508.11187v1)**

> **作者:** Wonjune Kang; Deb Roy
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** We introduce the task of expressive speech retrieval, where the goal is to retrieve speech utterances spoken in a given style based on a natural language description of that style. While prior work has primarily focused on performing speech retrieval based on what was said in an utterance, we aim to do so based on how something was said. We train speech and text encoders to embed speech and text descriptions of speaking styles into a joint latent space, which enables using free-form text prompts describing emotions or styles as queries to retrieve matching expressive speech segments. We perform detailed analyses of various aspects of our proposed framework, including encoder architectures, training criteria for effective cross-modal alignment, and prompt augmentation for improved generalization to arbitrary text queries. Experiments on multiple datasets encompassing 22 speaking styles demonstrate that our approach achieves strong retrieval performance as measured by Recall@k.
>
---
## 更新

#### [replaced 001] Neurodyne: Neural Pitch Manipulation with Representation Learning and Cycle-Consistency GAN
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15368v4](http://arxiv.org/pdf/2505.15368v4)**

> **作者:** Yicheng Gu; Chaoren Wang; Zhizheng Wu; Lauri Juvela
>
> **摘要:** Pitch manipulation is the process of producers adjusting the pitch of an audio segment to a specific key and intonation, which is essential in music production. Neural-network-based pitch-manipulation systems have been popular in recent years due to their superior synthesis quality compared to classical DSP methods. However, their performance is still limited due to their inaccurate feature disentanglement using source-filter models and the lack of paired in- and out-of-tune training data. This work proposes Neurodyne to address these issues. Specifically, Neurodyne uses adversarial representation learning to learn a pitch-independent latent representation to avoid inaccurate disentanglement and cycle-consistency training to create paired training data implicitly. Experimental results on global-key and template-based pitch manipulation demonstrate the effectiveness of the proposed system, marking improved synthesis quality while maintaining the original singer identity.
>
---
#### [replaced 002] MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.08715v2](http://arxiv.org/pdf/2508.08715v2)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** We are withdrawing the manuscript to revise the title and contents of figures for better alignment with the paper's contributions
>
> **摘要:** Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
>
---
#### [replaced 003] Generalizable speech deepfake detection via meta-learned LoRA
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.10838v2](http://arxiv.org/pdf/2502.10838v2)**

> **作者:** Janne Laakkonen; Ivan Kukanov; Ville Hautamäki
>
> **备注:** 10 pages, 5 figures, 7 tables
>
> **摘要:** Reliable detection of speech deepfakes (spoofs) must remain effective when the distribution of spoofing attacks shifts. We frame the task as domain generalization and show that inserting Low-Rank Adaptation (LoRA) adapters into every attention head of a self-supervised (SSL) backbone, then training only those adapters with Meta-Learning Domain Generalization (MLDG), yields strong zero-shot performance. The resulting model updates about 3.6 million parameters, roughly 1.1% of the 318 million updated in full fine-tuning, yet surpasses a fully fine-tuned counterpart on five of six evaluation corpora. A first-order MLDG loop encourages the adapters to focus on cues that persist across attack types, lowering the average EER from 8.84% for the fully fine-tuned model to 5.30% with our best MLDG-LoRA configuration. Our findings show that combining meta-learning with parameter-efficient adaptation offers an effective method for zero-shot, distribution-shift-aware speech deepfake detection.
>
---
#### [replaced 004] L3AC: Towards a Lightweight and Lossless Audio Codec
- **分类: cs.SD; cs.AI; 68T07; I.2.m**

- **链接: [http://arxiv.org/pdf/2504.04949v2](http://arxiv.org/pdf/2504.04949v2)**

> **作者:** Linwei Zhai; Han Ding; Cui Zhao; fei wang; Ge Wang; Wang Zhi; Wei Xi
>
> **摘要:** Neural audio codecs have recently gained traction for their ability to compress high-fidelity audio and provide discrete tokens for generative modeling. However, leading approaches often rely on resource-intensive models and complex multi-quantizer architectures, limiting their practicality in real-world applications. In this work, we introduce L3AC, a lightweight neural audio codec that addresses these challenges by leveraging a single quantizer and a highly efficient architecture. To enhance reconstruction fidelity while minimizing model complexity, L3AC explores streamlined convolutional networks and local Transformer modules, alongside TConv--a novel structure designed to capture acoustic variations across multiple temporal scales. Despite its compact design, extensive experiments across diverse datasets demonstrate that L3AC matches or exceeds the reconstruction quality of leading codecs while reducing computational overhead by an order of magnitude. The single-quantizer design further enhances its adaptability for downstream tasks. The source code is publicly available at https://github.com/zhai-lw/L3AC.
>
---
