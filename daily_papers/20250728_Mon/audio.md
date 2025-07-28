# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 7 篇**

## 最新发布

#### [new 001] Assessing the Reliability and Validity of a Balance Mat for Measuring Postural Stability: A Combined Robot-Human Approach
- **分类: eess.SP; cs.RO**

- **简介: 该论文评估了一种新型便携式平衡垫（BM）测量姿势稳定性的信度和效度，旨在解决传统力板（FP）便携性差、操作复杂的问题。研究通过机器人模拟和人体实验验证BM性能，结果显示其在校准后具有良好的一致性和准确性。**

- **链接: [http://arxiv.org/pdf/2507.18943v1](http://arxiv.org/pdf/2507.18943v1)**

> **作者:** Abishek Shrestha; Damith Herath; Angie Fearon; Maryam Ghahramani
>
> **摘要:** Postural sway assessment is important for detecting balance problems and identifying people at risk of falls. Force plates (FP) are considered the gold standard postural sway assessment method in laboratory conditions, but their lack of portability and requirement of high-level expertise limit their widespread usage. This study evaluates the reliability and validity of a novel Balance Mat (BM) device, a low-cost portable alternative that uses optical fibre technology. The research includes two studies: a robot study and a human study. In the robot study, a UR10 robotic arm was used to obtain controlled sway patterns to assess the reliability and sensitivity of the BM. In the human study, 51 healthy young participants performed balance tasks on the BM in combination with an FP to evaluate the BM's validity. Sway metrics such as sway mean, sway absolute mean, sway root mean square (RMS), sway path, sway range, and sway velocity were calculated from both BM and FP and compared. Reliability was evaluated using the intra-class correlation coefficient (ICC), where values greater than 0.9 were considered excellent and values between 0.75 and 0.9 were considered good. Results from the robot study demonstrated good to excellent ICC values in both single and double-leg stances. The human study showed moderate to strong correlations for sway path and range. Using Bland-Altman plots for agreement analysis revealed proportional bias between the BM and the FP where the BM overestimated sway metrics compared to the FP. Calibration was used to improve the agreement between the devices. The device demonstrated consistent sway measurement across varied stance conditions, establishing both reliability and validity following appropriate calibration.
>
---
#### [new 002] HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决现有语音编码在高压缩率下保真度不足、计算复杂度高的问题。作者提出了HH-Codec，采用单量化器结构和非对称编解码架构，实现高保真、低带宽（0.3 kbps）的语音重建，并优化码本利用与生成模型适配能力。**

- **链接: [http://arxiv.org/pdf/2507.18897v1](http://arxiv.org/pdf/2507.18897v1)**

> **作者:** Rongkun Xue; Yazhe Niu; Shuai Hu; Zixin Yin; Yongqiang Yao; Jing Yang
>
> **摘要:** Discrete speech tokenization is a fundamental component in speech codecs. However, in large-scale speech-to-speech systems, the complexity of parallel streams from multiple quantizers and the computational cost of high-time-dimensional codecs pose significant challenges. In this paper, we introduce HH-Codec, a neural codec that achieves extreme compression at 24 tokens per second for 24 kHz audio while relying on single-quantizer inference. Our approach involves a carefully designed Vector Quantization space for Spoken Language Modeling, optimizing compression efficiency while minimizing information loss. Building on this, we propose an asymmetric encoder-decoder architecture (Audio-VQ-Mel-Audio) that leverages dual supervision and progressive training to enhance reconstruction stability and fidelity. HH-Codec achieves state-of-the-art performance in speech reconstruction with an ultra-low bandwidth of 0.3 kbps. We further evaluate its effectiveness in codebook utilization and generative model adaptation, with extensive ablations validating the necessity of each module. HH-Codec is available at https://github.com/opendilab/HH-Codec.
>
---
#### [new 003] MLLM-based Speech Recognition: When and How is Multimodality Beneficial?
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于多模态语音识别任务，旨在研究多模态大语言模型（MLLMs）在噪声环境中提升语音识别准确率的条件与方法。通过实验分析不同模态的互补性、模态同步性、视觉编码质量等因素对ASR的影响，提供优化策略并加深对多模态语音识别的理解。**

- **链接: [http://arxiv.org/pdf/2507.19037v1](http://arxiv.org/pdf/2507.19037v1)**

> **作者:** Yiwen Guan; Viet Anh Trinh; Vivek Voleti; Jacob Whitehill
>
> **摘要:** Recent advances in multi-modal large language models (MLLMs) have opened new possibilities for unified modeling of speech, text, images, and other modalities. Building on our prior work, this paper examines the conditions and model architectures under which multiple input modalities can improve automatic speech recognition (ASR) accuracy in noisy environments. Through experiments on synthetic and real-world data, we find that (1) harnessing more modalities usually improves ASR accuracy, as each modality provides complementary information, but the improvement depends on the amount of auditory noise. (2) Synchronized modalities (e.g., lip movements) are more useful at high noise levels whereas unsynchronized modalities (e.g., image context) are most helpful at moderate noise levels. (3) Higher-quality visual representations consistently improve ASR accuracy, highlighting the importance of developing more powerful visual encoders. (4) Mamba exhibits similar trends regarding the benefits of multimodality as do Transformers. (5) The input order of modalities as well as their weights in the loss function can significantly impact accuracy. These findings both offer practical insights and help to deepen our understanding of multi-modal speech recognition under challenging conditions.
>
---
#### [new 004] The Eloquence team submission for task 1 of MLC-SLM challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文参与MLC-SLM挑战任务1，旨在提升多语言对话语音识别（ASR）。作者评估官方基线，使用不同基础模型训练两种投影器，并探索对比学习与扩展对话上下文对识别鲁棒性的影响。**

- **链接: [http://arxiv.org/pdf/2507.19308v1](http://arxiv.org/pdf/2507.19308v1)**

> **作者:** Lorenzo Concina; Jordi Luque; Alessio Brutti; Marco Matassoni; Yuchen Zhang
>
> **备注:** Technical Report for MLC-SLM Challenge of Interspeech2025
>
> **摘要:** In this paper, we present our studies and experiments carried out for the task 1 of the Challenge and Workshop on Multilingual Conversational Speech Language Model (MLC-SLM), which focuses on advancing multilingual conversational speech recognition through the development of speech language models architectures. Given the increasing relevance of real-world conversational data for building robust Spoken Dialogue Systems, we explore three approaches to multilingual ASR. First, we conduct an evaluation of the official baseline to better understand its strengths and limitations, by training two projectors (linear and qformer) with different foundation models. Second we leverage the SLAM-ASR framework to train a custom multilingual linear projector. Finally we investigate the role of contrastive learning and the extended conversational context in enhancing the robustness of recognition.
>
---
#### [new 005] Face2VoiceSync: Lightweight Face-Voice Consistency for Text-Driven Talking Face Generation
- **分类: cs.SD; cs.CV; cs.MM; eess.AS**

- **简介: 该论文属于文本驱动的说话人脸生成任务，旨在解决语音与人脸外貌不匹配的问题。论文提出了Face2VoiceSync框架，实现人脸与语音的一致性生成，并在生成质量、多样性、可控性及训练效率方面取得进展。**

- **链接: [http://arxiv.org/pdf/2507.19225v1](http://arxiv.org/pdf/2507.19225v1)**

> **作者:** Fang Kang; Yin Cao; Haoyu Chen
>
> **摘要:** Recent studies in speech-driven talking face generation achieve promising results, but their reliance on fixed-driven speech limits further applications (e.g., face-voice mismatch). Thus, we extend the task to a more challenging setting: given a face image and text to speak, generating both talking face animation and its corresponding speeches. Accordingly, we propose a novel framework, Face2VoiceSync, with several novel contributions: 1) Voice-Face Alignment, ensuring generated voices match facial appearance; 2) Diversity \& Manipulation, enabling generated voice control over paralinguistic features space; 3) Efficient Training, using a lightweight VAE to bridge visual and audio large-pretrained models, with significantly fewer trainable parameters than existing methods; 4) New Evaluation Metric, fairly assessing the diversity and identity consistency. Experiments show Face2VoiceSync achieves both visual and audio state-of-the-art performances on a single 40GB GPU.
>
---
#### [new 006] SCORE-SET: A dataset of GuitarPro files for Music Phrase Generation and Sequence Learning
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐生成与序列学习任务，旨在解决现有数据缺乏真实吉他演奏细节的问题。作者构建了SCORE-SET数据集，将MAESTRO和GiantMIDI中的MIDI音符转换为包含滑音、弯音等演奏技巧的吉他谱文件，适用于音乐生成和表演感知学习。**

- **链接: [http://arxiv.org/pdf/2507.18723v1](http://arxiv.org/pdf/2507.18723v1)**

> **作者:** Vishakh Begari
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** A curated dataset of Guitar Pro tablature files (.gp5 format), tailored for tasks involving guitar music generation, sequence modeling, and performance-aware learning is provided. The dataset is derived from MIDI notes in MAESTRO and GiantMIDI which have been adapted into rhythm guitar tracks. These tracks are further processed to include a variety of expression settings typical of guitar performance, such as bends, slides, vibrato, and palm muting, to better reflect the nuances of real-world guitar playing.
>
---
#### [new 007] From Continuous to Discrete: Cross-Domain Collaborative General Speech Enhancement via Hierarchical Language Models
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决真实场景中多种失真（如噪声、混响、带宽限制等）同时存在时的语音质量下降问题。论文提出了OmniGSE框架，结合判别式与生成式方法，通过连续特征增强与离散语言模型协同优化，提升复杂场景下的语音恢复效果。**

- **链接: [http://arxiv.org/pdf/2507.19062v1](http://arxiv.org/pdf/2507.19062v1)**

> **作者:** Zhaoxi Mu; Rilin Chen; Andong Li; Meng Yu; Xinyu Yang; Dong Yu
>
> **备注:** ACMMM 2025
>
> **摘要:** This paper introduces OmniGSE, a novel general speech enhancement (GSE) framework designed to mitigate the diverse distortions that speech signals encounter in real-world scenarios. These distortions include background noise, reverberation, bandwidth limitations, signal clipping, and network packet loss. Existing methods typically focus on optimizing for a single type of distortion, often struggling to effectively handle the simultaneous presence of multiple distortions in complex scenarios. OmniGSE bridges this gap by integrating the strengths of discriminative and generative approaches through a two-stage architecture that enables cross-domain collaborative optimization. In the first stage, continuous features are enhanced using a lightweight channel-split NAC-RoFormer. In the second stage, discrete tokens are generated to reconstruct high-quality speech through language models. Specifically, we designed a hierarchical language model structure consisting of a RootLM and multiple BranchLMs. The RootLM models general acoustic features across codebook layers, while the BranchLMs explicitly capture the progressive relationships between different codebook levels. Experimental results demonstrate that OmniGSE surpasses existing models across multiple benchmarks, particularly excelling in scenarios involving compound distortions. These findings underscore the framework's potential for robust and versatile speech enhancement in real-world applications.
>
---
#### [new 008] Latent Granular Resynthesis using Neural Audio Codecs
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 论文提出一种基于神经音频编解码器的潜在粒状重合成技术，用于音频风格迁移任务。该方法通过构建源音频的潜在向量码本，将目标音频的潜在粒与码本匹配并合成，实现保留目标时序结构、融合源音色特征的音频生成，无需训练模型，适用于多种音频材料。**

- **链接: [http://arxiv.org/pdf/2507.19202v1](http://arxiv.org/pdf/2507.19202v1)**

> **作者:** Nao Tokui; Tom Baker
>
> **备注:** Accepted at ISMIR 2025 Late Breaking Demos
>
> **摘要:** We introduce a novel technique for creative audio resynthesis that operates by reworking the concept of granular synthesis at the latent vector level. Our approach creates a "granular codebook" by encoding a source audio corpus into latent vector segments, then matches each latent grain of a target audio signal to its closest counterpart in the codebook. The resulting hybrid sequence is decoded to produce audio that preserves the target's temporal structure while adopting the source's timbral characteristics. This technique requires no model training, works with diverse audio materials, and naturally avoids the discontinuities typical of traditional concatenative synthesis through the codec's implicit interpolation during decoding. We include supplementary material at https://github.com/naotokui/latentgranular/ , as well as a proof-of-concept implementation to allow users to experiment with their own sounds at https://huggingface.co/spaces/naotokui/latentgranular .
>
---
#### [new 009] Binaural Target Speaker Extraction using HRTFs and a Complex-Valued Neural Network
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音信号处理任务，旨在解决多人同时说话环境下目标说话人分离的问题。作者提出了一种基于HRTF和复数值神经网络的方法，无需依赖说话人嵌入，实现了跨语言和数据集的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.19369v1](http://arxiv.org/pdf/2507.19369v1)**

> **作者:** Yoav Ellinson; Sharon Gannot
>
> **摘要:** In this work, we aim to imitate the human ability to selectively attend to a single speaker, even in the presence of multiple simultaneous talkers. We propose a novel approach for binaural target speaker extraction that leverages the listener's Head-Related Transfer Function (HRTF) to isolate the desired speaker. Notably, our method does not rely on speaker embeddings, making it speaker-independent and enabling strong generalization across multiple speech datasets in different languages. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier Transform (STFT) of the mixed audio signals. This deviates from conventional approaches that use spectrograms or treat the real and imaginary components of the STFT as separate real-valued inputs. We first evaluate the method in an anechoic, noise-free scenario, where it demonstrates excellent extraction performance while effectively preserving the binaural cues of the target signal. We then test a modified variant under mild reverberation conditions. This version remains robust in reverberant environments, maintaining speech clarity, preserving source directionality, and simultaneously reducing reverberation.
>
---
#### [new 010] Assessment of Personality Dimensions Across Situations Using Conversational Speech
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文研究通过对话语音分析不同情境下的个性感知，属于自动个性感知任务。旨在解决传统方法忽略情境对个性判断影响的问题。工作包括分析两种工作情境中语音特征与个性评分的关系，发现情境影响个性判断，特定声学特征在不同情境下对个性预测效果不同，且非神经特征优于嵌入表示。**

- **链接: [http://arxiv.org/pdf/2507.19137v1](http://arxiv.org/pdf/2507.19137v1)**

> **作者:** Alice Zhang; Skanda Muralidhar; Daniel Gatica-Perez; Mathew Magimai-Doss
>
> **摘要:** Prior research indicates that users prefer assistive technologies whose personalities align with their own. This has sparked interest in automatic personality perception (APP), which aims to predict an individual's perceived personality traits. Previous studies in APP have treated personalities as static traits, independent of context. However, perceived personalities can vary by context and situation as shown in psychological research. In this study, we investigate the relationship between conversational speech and perceived personality for participants engaged in two work situations (a neutral interview and a stressful client interaction). Our key findings are: 1) perceived personalities differ significantly across interactions, 2) loudness, sound level, and spectral flux features are indicative of perceived extraversion, agreeableness, conscientiousness, and openness in neutral interactions, while neuroticism correlates with these features in stressful contexts, 3) handcrafted acoustic features and non-verbal features outperform speaker embeddings in inference of perceived personality, and 4) stressful interactions are more predictive of neuroticism, aligning with existing psychological research.
>
---
#### [new 011] CatchPhrase: EXPrompt-Guided Encoder Adaptation for Audio-to-Image Generation
- **分类: cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于音频到图像生成任务，旨在解决音频与图像语义不一致问题。通过挖掘跨模态语义提示（EXPrompt Mining）并选择最匹配的提示（EXPrompt Selector），结合轻量映射网络，提升音频驱动图像生成的质量与语义对齐度。**

- **链接: [http://arxiv.org/pdf/2507.18750v1](http://arxiv.org/pdf/2507.18750v1)**

> **作者:** Hyunwoo Oh; SeungJu Cha; Kwanyoung Lee; Si-Woo Kim; Dong-Jin Kim
>
> **摘要:** We propose CatchPhrase, a novel audio-to-image generation framework designed to mitigate semantic misalignment between audio inputs and generated images. While recent advances in multi-modal encoders have enabled progress in cross-modal generation, ambiguity stemming from homographs and auditory illusions continues to hinder accurate alignment. To address this issue, CatchPhrase generates enriched cross-modal semantic prompts (EXPrompt Mining) from weak class labels by leveraging large language models (LLMs) and audio captioning models (ACMs). To address both class-level and instance-level misalignment, we apply multi-modal filtering and retrieval to select the most semantically aligned prompt for each audio sample (EXPrompt Selector). A lightweight mapping network is then trained to adapt pre-trained text-to-image generation models to audio input. Extensive experiments on multiple audio classification datasets demonstrate that CatchPhrase improves audio-to-image alignment and consistently enhances generation quality by mitigating semantic misalignment.
>
---
#### [new 012] SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models
- **分类: cs.CL; cs.AI; cs.SC; cs.SD; eess.AS**

- **简介: 该论文提出了SpeechIQ（SIQ），一种基于认知层次的语音理解评估框架，用于评估语音大模型（LLM Voice）在语音理解方面的能力。它结合了Bloom分类法的三个认知层次：记忆、理解和应用，旨在超越传统的WER指标，提供更全面的模型评估与比较，并检测标注错误和幻觉问题。**

- **链接: [http://arxiv.org/pdf/2507.19361v1](http://arxiv.org/pdf/2507.19361v1)**

> **作者:** Zhen Wan; Chao-Han Huck Yang; Yahan Yu; Jinchuan Tian; Sheng Li; Ke Hu; Zhehuai Chen; Shinji Watanabe; Fei Cheng; Chenhui Chu; Sadao Kurohashi
>
> **备注:** Our Speech-IQ leaderboard will be hosted at huggingface.co/spaces/nvidia/Speech-IQ-leaderboard. ACL 2025 main
>
> **摘要:** We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training.
>
---
#### [new 013] KuiSCIMA v2.0: Improved Baselines, Calibration, and Cross-Notation Generalization for Historical Chinese Music Notations in Jiang Kui's Baishidaoren Gequ
- **分类: cs.CV; cs.DL; cs.SD; eess.AS**

- **简介: 该论文属于光学音乐识别（OMR）任务，旨在解决历史中文乐谱识别中数据稀缺、类别不平衡的问题。论文改进了基线模型，提升了识别准确率与模型校准，并扩展了数据集以增强跨版本泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18741v1](http://arxiv.org/pdf/2507.18741v1)**

> **作者:** Tristan Repolusk; Eduardo Veas
>
> **备注:** International Conference on Document Analysis and Recognition. This preprint has not undergone any post-submission improvements or corrections. The Version of Record of this contribution is published in "19th International Conference on Document Analysis and Recognition (ICDAR 2025), Wuhan, China, September 16-21, 2025, Proceedings", and is available online at the External DOI field below
>
> **摘要:** Optical Music Recognition (OMR) for historical Chinese musical notations, such as suzipu and l\"ul\"upu, presents unique challenges due to high class imbalance and limited training data. This paper introduces significant advancements in OMR for Jiang Kui's influential collection Baishidaoren Gequ from 1202. In this work, we develop and evaluate a character recognition model for scarce imbalanced data. We improve upon previous baselines by reducing the Character Error Rate (CER) from 10.4% to 7.1% for suzipu, despite working with 77 highly imbalanced classes, and achieve a remarkable CER of 0.9% for l\"ul\"upu. Our models outperform human transcribers, with an average human CER of 15.9% and a best-case CER of 7.6%. We employ temperature scaling to achieve a well-calibrated model with an Expected Calibration Error (ECE) below 0.0162. Using a leave-one-edition-out cross-validation approach, we ensure robust performance across five historical editions. Additionally, we extend the KuiSCIMA dataset to include all 109 pieces from Baishidaoren Gequ, encompassing suzipu, l\"ul\"upu, and jianzipu notations. Our findings advance the digitization and accessibility of historical Chinese music, promoting cultural diversity in OMR and expanding its applicability to underrepresented music traditions.
>
---
#### [new 014] Should Top-Down Clustering Affect Boundaries in Unsupervised Word Discovery?
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于无监督词发现任务，旨在将未标注语音分割为类词单元并聚类生成词典。论文比较了自下而上和自上而下的聚类方法对边界选择的影响，发现两者在多语言数据上表现相当，但自下而上方法更快。分析表明，聚类步骤是性能瓶颈，建议未来研究更优聚类技术和表征学习。**

- **链接: [http://arxiv.org/pdf/2507.19204v1](http://arxiv.org/pdf/2507.19204v1)**

> **作者:** Simon Malan; Benjamin van Niekerk; Herman Kamper
>
> **备注:** 5 figures, 5 tables
>
> **摘要:** We investigate the problem of segmenting unlabeled speech into word-like units and clustering these to create a lexicon. Prior work can be categorized into two frameworks. Bottom-up methods first determine boundaries and then cluster the fixed segmented words into a lexicon. In contrast, top-down methods incorporate information from the clustered words to inform boundary selection. However, it is unclear whether top-down information is necessary to improve segmentation. To explore this, we look at two similar approaches that differ in whether top-down clustering informs boundary selection. Our simple bottom-up strategy predicts word boundaries using the dissimilarity between adjacent self-supervised features, then clusters the resulting segments to construct a lexicon. Our top-down system is an updated version of the ES-KMeans dynamic programming method that iteratively uses K-means to update its boundaries. On the five-language ZeroSpeech benchmarks, both approaches achieve comparable state-of-the-art results, with the bottom-up system being nearly five times faster. Through detailed analyses, we show that the top-down influence of ES-KMeans can be beneficial (depending on factors like the candidate boundaries), but in many cases the simple bottom-up method performs just as well. For both methods, we show that the clustering step is a limiting factor. Therefore, we recommend that future work focus on improved clustering techniques and learning more discriminative word-like representations. Project code repository: https://github.com/s-malan/prom-seg-clus.
>
---
## 更新

#### [replaced 001] GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.18119v2](http://arxiv.org/pdf/2507.18119v2)**

> **作者:** Hongjie Chen; Zehan Li; Yaodong Song; Wenming Deng; Yitong Yao; Yuxin Zhang; Hang Lv; Xuechao Zhu; Jian Kang; Jie Lian; Jie Li; Chao Wang; Shuangyong Song; Yongxiang Li; Zhongjiang He; Xuelong Li
>
> **摘要:** Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems.
>
---
#### [replaced 002] Acoustically Precise Hesitation Tagging Is Essential for End-to-End Verbatim Transcription Systems
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.04076v2](http://arxiv.org/pdf/2506.04076v2)**

> **作者:** Jhen-Ke Lin; Hao-Chien Lu; Chung-Chun Wang; Hong-Yun Lin; Berlin Chen
>
> **备注:** accepted to the ISCA SLaTE-2025 Workshop
>
> **摘要:** Verbatim transcription for automatic speaking assessment demands accurate capture of disfluencies, crucial for downstream tasks like error analysis and feedback. However, many ASR systems discard or generalize hesitations, losing important acoustic details. We fine-tune Whisper models on the Speak & Improve 2025 corpus using low-rank adaptation (LoRA), without recourse to external audio training data. We compare three annotation schemes: removing hesitations (Pure), generic tags (Rich), and acoustically precise fillers inferred by Gemini 2.0 Flash from existing audio-transcript pairs (Extra). Our challenge system achieved 6.47% WER (Pure) and 5.81% WER (Extra). Post-challenge experiments reveal that fine-tuning Whisper Large V3 Turbo with the "Extra" scheme yielded a 5.5% WER, an 11.3% relative improvement over the "Pure" scheme (6.2% WER). This demonstrates that explicit, realistic filled-pause labeling significantly enhances ASR accuracy for verbatim L2 speech transcription.
>
---
#### [replaced 003] Self-Supervised Frameworks for Speaker Verification via Bootstrapped Positive Sampling
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.17772v4](http://arxiv.org/pdf/2501.17772v4)**

> **作者:** Theo Lepage; Reda Dehak
>
> **备注:** accepted for publication in IEEE TASLP
>
> **摘要:** Recent developments in Self-Supervised Learning (SSL) have demonstrated significant potential for Speaker Verification (SV), but closing the performance gap with supervised systems remains an ongoing challenge. SSL frameworks rely on anchor-positive pairs, constructed from segments of the same audio utterance. Hence, positives have channel characteristics similar to those of their corresponding anchors, even with extensive data-augmentation. Therefore, this positive sampling strategy is a fundamental limitation as it encodes too much information regarding the recording source in the learned representations. This article introduces Self-Supervised Positive Sampling (SSPS), a bootstrapped technique for sampling appropriate and diverse positives in SSL frameworks for SV. SSPS samples positives close to their anchor in the representation space, assuming that these pseudo-positives belong to the same speaker identity but correspond to different recording conditions. This method consistently demonstrates improvements in SV performance on VoxCeleb benchmarks when applied to major SSL frameworks, including SimCLR, SwAV, VICReg, and DINO. Using SSPS, SimCLR and DINO achieve 2.57% and 2.53% EER on VoxCeleb1-O, respectively. SimCLR yields a 58% relative reduction in EER, getting comparable performance to DINO with a simpler training framework. Furthermore, SSPS lowers intra-class variance and reduces channel information in speaker representations while exhibiting greater robustness without data-augmentation.
>
---
#### [replaced 004] AI Flow: Perspectives, Scenarios, and Approaches
- **分类: cs.AI; cs.CL; cs.CV; cs.DC; eess.SP**

- **链接: [http://arxiv.org/pdf/2506.12479v3](http://arxiv.org/pdf/2506.12479v3)**

> **作者:** Hongjun An; Wenhan Hu; Sida Huang; Siqi Huang; Ruanjun Li; Yuanzhi Liang; Jiawei Shao; Yiliang Song; Zihan Wang; Cheng Yuan; Chi Zhang; Hongyuan Zhang; Wenhao Zhuang; Xuelong Li
>
> **备注:** Authors are with Institute of Artificial Intelligence (TeleAI), China Telecom, China. Author names are listed alphabetically by surname. This work was conducted at TeleAI, facilitated by Dr. Jiawei Shao (e-mail: shaojw2@chinatelecom.cn) under the leadership of Prof. Xuelong Li. The corresponding author is Prof. Xuelong Li (e-mail: xuelong li@ieee.org), the CTO and Chief Scientist of China Telecom
>
> **摘要:** Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems.
>
---
#### [replaced 005] Integrating IP Broadcasting with Audio Tags: Workflow and Challenges
- **分类: eess.AS; cs.AI; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2407.15423v3](http://arxiv.org/pdf/2407.15423v3)**

> **作者:** Rhys Burchett-Vass; Arshdeep Singh; Gabriel Bibbó; Mark D. Plumbley
>
> **备注:** Accepted for publication in 2025 AES International Conference on Artificial Intelligence and Machine Learning for Audio
>
> **摘要:** The broadcasting industry has adopted IP technologies, revolutionising both live and pre-recorded content production, from news gathering to live music events. IP broadcasting allows for the transport of audio and video signals in an easily configurable way, aligning with modern networking techniques. This shift towards an IP workflow allows for much greater flexibility, not only in routing signals but with the integration of tools using standard web development techniques. One possible tool could include the use of live audio tagging, which has a number of uses in the production of content. These could include adding sound effects to automated closed captioning or identifying unwanted sound events within a scene. In this paper, we describe the process of containerising an audio tagging model into a microservice, a small segregated code module that can be integrated into a multitude of different network setups. The goal is to develop a modular, accessible, and flexible tool capable of seamless deployment into broadcasting workflows of all sizes, from small productions to large corporations. Challenges surrounding latency of the selected audio tagging model and its effect on the usefulness of the end product are discussed.
>
---
#### [replaced 006] Incremental Averaging Method to Improve Graph-Based Time-Difference-of-Arrival Estimation
- **分类: eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.07087v3](http://arxiv.org/pdf/2507.07087v3)**

> **作者:** Klaus Brümann; Kouei Yamaoka; Nobutaka Ono; Simon Doclo
>
> **备注:** This paper was accepted for presentation at the IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA) 2025
>
> **摘要:** Estimating the position of a speech source based on time-differences-of-arrival (TDOAs) is often adversely affected by background noise and reverberation. A popular method to estimate the TDOA between a microphone pair involves maximizing a generalized cross-correlation with phase transform (GCC-PHAT) function. Since the TDOAs across different microphone pairs satisfy consistency relations, generally only a small subset of microphone pairs are used for source position estimation. Although the set of microphone pairs is often determined based on a reference microphone, recently a more robust method has been proposed to determine the set of microphone pairs by computing the minimum spanning tree (MST) of a signal graph of GCC-PHAT function reliabilities. To reduce the influence of noise and reverberation on the TDOA estimation accuracy, in this paper we propose to compute the GCC-PHAT functions of the MST based on an average of multiple cross-power spectral densities (CPSDs) using an incremental method. In each step of the method, we increase the number of CPSDs over which we average by considering CPSDs computed indirectly via other microphones from previous steps. Using signals recorded in a noisy and reverberant laboratory with an array of spatially distributed microphones, the performance of the proposed method is evaluated in terms of TDOA estimation error and 2D source position estimation error. Experimental results for different source and microphone configurations and three reverberation conditions show that the proposed method considering multiple CPSDs improves the TDOA estimation and source position estimation accuracy compared to the reference microphone- and MST-based methods that rely on a single CPSD as well as steered-response power-based source position estimation.
>
---
#### [replaced 007] SALM-Duplex: Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v4](http://arxiv.org/pdf/2505.15670v4)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
