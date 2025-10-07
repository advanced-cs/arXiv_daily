# 音频 cs.SD;  eess.SP

- **最新发布 27 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] GDiffuSE: Diffusion-based speech enhancement with noise model guidance
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决复杂噪声环境下语音质量提升问题。作者提出GDiffuSE方法，利用扩散模型结合噪声估计引导机制，实现对未知噪声类型的鲁棒增强，取得了优于现有方法的效果。**

- **链接: [http://arxiv.org/pdf/2510.04157v1](http://arxiv.org/pdf/2510.04157v1)**

> **作者:** Efrayim Yanir; David Burshtein; Sharon Gannot
>
> **摘要:** This paper introduces a novel speech enhancement (SE) approach based on a denoising diffusion probabilistic model (DDPM), termed Guided diffusion for speech enhancement (GDiffuSE). In contrast to conventional methods that directly map noisy speech to clean speech, our method employs a lightweight helper model to estimate the noise distribution, which is then incorporated into the diffusion denoising process via a guidance mechanism. This design improves robustness by enabling seamless adaptation to unseen noise types and by leveraging large-scale DDPMs originally trained for speech generation in the context of SE. We evaluate our approach on noisy signals obtained by adding noise samples from the BBC sound effects database to LibriSpeech utterances, showing consistent improvements over state-of-the-art baselines under mismatched noise conditions. Examples are available at our project webpage.
>
---
#### [new 002] Linguistic and Audio Embedding-Based Machine Learning for Alzheimer's Dementia and Mild Cognitive Impairment Detection: Insights from the PROCESS Challenge
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于医疗诊断任务，旨在通过语音数据早期检测阿尔茨海默病（AD）和轻度认知障碍（MCI）。研究利用机器学习模型，结合音频嵌入和语言特征，对认知状态进行分类并预测MMSE评分。结果表明，融合语言特征的模型在分类任务中表现最佳，而基于Whisper音频嵌入的模型在评分预测上误差最小，展示了多模态语音分析在非侵入性认知评估中的潜力。**

- **链接: [http://arxiv.org/pdf/2510.03336v1](http://arxiv.org/pdf/2510.03336v1)**

> **作者:** Adharsha Sam Edwin Sam Devahi; Sohail Singh Sangha; Prachee Priyadarshinee; Jithin Thilakan; Ivan Fu Xing Tan; Christopher Johann Clarke; Sou Ka Lon; Balamurali B T; Yow Wei Quin; Chen Jer-Ming
>
> **摘要:** Early detection of Alzheimer's Dementia (AD) and Mild Cognitive Impairment (MCI) is critical for timely intervention, yet current diagnostic approaches remain resource-intensive and invasive. Speech, encompassing both acoustic and linguistic dimensions, offers a promising non-invasive biomarker for cognitive decline. In this study, we present a machine learning framework for the PROCESS Challenge, leveraging both audio embeddings and linguistic features derived from spontaneous speech recordings. Audio representations were extracted using Whisper embeddings from the Cookie Theft description task, while linguistic features-spanning pronoun usage, syntactic complexity, filler words, and clause structure-were obtained from transcriptions across Semantic Fluency, Phonemic Fluency, and Cookie Theft picture description. Classification models aimed to distinguish between Healthy Controls (HC), MCI, and AD participants, while regression models predicted Mini-Mental State Examination (MMSE) scores. Results demonstrated that voted ensemble models trained on concatenated linguistic features achieved the best classification performance (F1 = 0.497), while Whisper embedding-based ensemble regressors yielded the lowest MMSE prediction error (RMSE = 2.843). Comparative evaluation within the PROCESS Challenge placed our models among the top submissions in regression task, and mid-range for classification, highlighting the complementary strengths of linguistic and audio embeddings. These findings reinforce the potential of multimodal speech-based approaches for scalable, non-invasive cognitive assessment and underline the importance of integrating task-specific linguistic and acoustic markers in dementia detection.
>
---
#### [new 003] Désentrelacement Fréquentiel Doux pour les Codecs Audio Neuronaux
- **分类: cs.SD; eess.AS; q-bio.NC**

- **简介: 该论文属于音频编解码任务，旨在提升神经音频编解码器的表示可解释性。针对现有方法依赖特定数据集或任务的问题，作者提出一种基于频谱分解的神经音频编解码方法。实验表明，该方法在重建保真度和感知质量上优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.03741v1](http://arxiv.org/pdf/2510.03741v1)**

> **作者:** Benoît Giniès; Xiaoyu Bie; Olivier Fercoq; Gaël Richard
>
> **备注:** in French language, Groupe de Recherche et d'Etudes du Traitement du Signal et des Images (GRETSI 2025), Aug 2025, Strasbourg, France
>
> **摘要:** While neural-based models have led to significant advancements in audio feature extraction, the interpretability of the learned representations remains a critical challenge. To address this, disentanglement techniques have been integrated into discrete neural audio codecs to impose structure on the extracted tokens. However, these approaches often exhibit strong dependencies on specific datasets or task formulations. In this work, we propose a disentangled neural audio codec that leverages spectral decomposition of time-domain signals to enhance representation interpretability. Experimental evaluations demonstrate that our method surpasses a state-of-the-art baseline in both reconstruction fidelity and perceptual quality.
>
---
#### [new 004] Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; eess.AS**

- **简介: 论文提出MAVE模型，用于高质量语音编辑和零样本文本到语音合成。基于交叉注意力Mamba架构，实现上下文感知的语音编辑与跨模态对齐，在语音编辑任务中表现优异，多数编辑结果难以与原声区分，并在零样本TTS中优于现有方法，具有更低内存消耗。**

- **链接: [http://arxiv.org/pdf/2510.04738v1](http://arxiv.org/pdf/2510.04738v1)**

> **作者:** Baher Mohammad; Magauiya Zhussip; Stamatios Lefkimmiatis
>
> **摘要:** We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE - edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE - demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with VoiceCraft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ~6x less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention.
>
---
#### [new 005] Soft Disentanglement in Frequency Bands for Neural Audio Codecs
- **分类: cs.SD**

- **简介: 论文属于音频信号处理任务，旨在解决神经音频编解码器中特征解耦的问题。现有方法依赖数据特性或特定任务，本文提出一种通用方法，通过频谱分解和多分支编解码结构，实现更优的特征解耦，提升了音频重建和感知效果，并有助于修复任务。**

- **链接: [http://arxiv.org/pdf/2510.03735v1](http://arxiv.org/pdf/2510.03735v1)**

> **作者:** Benoit Ginies; Xiaoyu Bie; Olivier Fercoq; Gaël Richard
>
> **摘要:** In neural-based audio feature extraction, ensuring that representations capture disentangled information is crucial for model interpretability. However, existing disentanglement methods often rely on assumptions that are highly dependent on data characteristics or specific tasks. In this work, we introduce a generalizable approach for learning disentangled features within a neural architecture. Our method applies spectral decomposition to time-domain signals, followed by a multi-branch audio codec that operates on the decomposed components. Empirical evaluations demonstrate that our approach achieves better reconstruction and perceptual performance compared to a state-of-the-art baseline while also offering potential advantages for inpainting tasks.
>
---
#### [new 006] Language Model Based Text-to-Audio Generation: Anti-Causally Aligned Collaborative Residual Transformers
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于文本到音频生成任务，旨在解决语言模型在多层残差向量量化（RVQ）下的生成能力不足问题。作者提出Siren框架，采用多隔离Transformer与因果条件及反因果对齐策略，通过强化学习优化生成效果，最终在音频生成质量和多模态统一框架上取得进展。**

- **链接: [http://arxiv.org/pdf/2510.04577v1](http://arxiv.org/pdf/2510.04577v1)**

> **作者:** Juncheng Wang; Chao Xu; Cheng Yu; Zhe Hu; Haoyu Xie; Guoqi Yu; Lei Shang; Shujun Wang
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** While language models (LMs) paired with residual vector quantization (RVQ) tokenizers have shown promise in text-to-audio (T2A) generation, they still lag behind diffusion-based models by a non-trivial margin. We identify a critical dilemma underpinning this gap: incorporating more RVQ layers improves audio reconstruction fidelity but exceeds the generation capacity of conventional LMs. To address this, we first analyze RVQ dynamics and uncover two key limitations: 1) orthogonality of features across RVQ layers hinders effective LMs training, and 2) descending semantic richness in tokens from deeper RVQ layers exacerbates exposure bias during autoregressive decoding. Based on these insights, we propose Siren, a novel LM-based framework that employs multiple isolated transformers with causal conditioning and anti-causal alignment via reinforcement learning. Extensive experiments demonstrate that Siren outperforms both existing LM-based and diffusion-based T2A systems, achieving state-of-the-art results. By bridging the representational strengths of LMs with the fidelity demands of audio synthesis, our approach repositions LMs as competitive contenders against diffusion models in T2A tasks. Moreover, by aligning audio representations with linguistic structures, Siren facilitates a promising pathway toward unified multi-modal generation frameworks.
>
---
#### [new 007] A Study on the Data Distribution Gap in Music Emotion Recognition
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐情感识别任务，旨在解决跨数据集泛化能力差的问题。作者分析了五个包含不同音乐风格的数据集，探讨了流派与情感的关系及数据偏倚问题，并提出一种结合Jukebox模型嵌入与色度特征的框架，提升了模型的泛化性能。**

- **链接: [http://arxiv.org/pdf/2510.04688v1](http://arxiv.org/pdf/2510.04688v1)**

> **作者:** Joann Ching; Gerhard Widmer
>
> **备注:** Accepted at the 17th International Symposium on Computer Music Multidisciplinary Research (CMMR) 2025
>
> **摘要:** Music Emotion Recognition (MER) is a task deeply connected to human perception, relying heavily on subjective annotations collected from contributors. Prior studies tend to focus on specific musical styles rather than incorporating a diverse range of genres, such as rock and classical, within a single framework. In this paper, we address the task of recognizing emotion from audio content by investigating five datasets with dimensional emotion annotations -- EmoMusic, DEAM, PMEmo, WTC, and WCMED -- which span various musical styles. We demonstrate the problem of out-of-distribution generalization in a systematic experiment. By closely looking at multiple data and feature sets, we provide insight into genre-emotion relationships in existing data and examine potential genre dominance and dataset biases in certain feature representations. Based on these experiments, we arrive at a simple yet effective framework that combines embeddings extracted from the Jukebox model with chroma features and demonstrate how, alongside a combination of several diverse training sets, this permits us to train models with substantially improved cross-dataset generalization capabilities.
>
---
#### [new 008] Machine Unlearning in Speech Emotion Recognition via Forget Set Alone
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决隐私数据删除问题。现有遗忘方法依赖额外数据，而本文提出仅使用待遗忘数据进行对抗攻击微调，实现有效遗忘且保持模型性能。**

- **链接: [http://arxiv.org/pdf/2510.04251v1](http://arxiv.org/pdf/2510.04251v1)**

> **作者:** Zhao Ren; Rathi Adarshi Rammohan; Kevin Scheck; Tanja Schultz
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.
>
---
#### [new 009] Pitch-Conditioned Instrument Sound Synthesis From an Interactive Timbre Latent Space
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于音频合成任务，旨在解决现有方法在音色控制与导航上的不足。论文提出一种两阶段框架：先用变分自编码器学习音高-音色解耦的2D表示，再以此作为条件输入训练Transformer生成模型。最终实现音高准确、音色可控制的高质量乐器声音合成，并通过网页应用展示其交互潜力。**

- **链接: [http://arxiv.org/pdf/2510.04339v1](http://arxiv.org/pdf/2510.04339v1)**

> **作者:** Christian Limberg; Fares Schulz; Zhe Zhang; Stefan Weinzierl
>
> **备注:** 8 pages, accepted to the Proceedings of the 28-th Int. Conf. on Digital Audio Effects (DAFx25) - demo: https://pgesam.faresschulz.com
>
> **摘要:** This paper presents a novel approach to neural instrument sound synthesis using a two-stage semi-supervised learning framework capable of generating pitch-accurate, high-quality music samples from an expressive timbre latent space. Existing approaches that achieve sufficient quality for music production often rely on high-dimensional latent representations that are difficult to navigate and provide unintuitive user experiences. We address this limitation through a two-stage training paradigm: first, we train a pitch-timbre disentangled 2D representation of audio samples using a Variational Autoencoder; second, we use this representation as conditioning input for a Transformer-based generative model. The learned 2D latent space serves as an intuitive interface for navigating and exploring the sound landscape. We demonstrate that the proposed method effectively learns a disentangled timbre space, enabling expressive and controllable audio generation with reliable pitch conditioning. Experimental results show the model's ability to capture subtle variations in timbre while maintaining a high degree of pitch accuracy. The usability of our method is demonstrated in an interactive web application, highlighting its potential as a step towards future music production environments that are both intuitive and creatively empowering: https://pgesam.faresschulz.com
>
---
#### [new 010] Lightweight and Generalizable Acoustic Scene Representations via Contrastive Fine-Tuning and Distillation
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于音频场景分类任务，旨在解决模型在边缘设备上适应新声学类别的泛化能力不足问题。作者提出ContrastASC方法，通过对比微调和知识蒸馏学习可迁移的声学表示，使模型在少量样本下能适应新类别，同时保持原有分类性能。**

- **链接: [http://arxiv.org/pdf/2510.03728v1](http://arxiv.org/pdf/2510.03728v1)**

> **作者:** Kuang Yuan; Yang Gao; Xilin Li; Xinhao Mei; Syavosh Zadissa; Tarun Pruthi; Saeed Bagheri Sereshki
>
> **摘要:** Acoustic scene classification (ASC) models on edge devices typically operate under fixed class assumptions, lacking the transferability needed for real-world applications that require adaptation to new or refined acoustic categories. We propose ContrastASC, which learns generalizable acoustic scene representations by structuring the embedding space to preserve semantic relationships between scenes, enabling adaptation to unseen categories without retraining. Our approach combines supervised contrastive fine-tuning of pre-trained models with contrastive representation distillation to transfer this structured knowledge to compact student models. Our evaluation shows that ContrastASC demonstrates improved few-shot adaptation to unseen categories while maintaining strong closed-set performance.
>
---
#### [new 011] Evaluating Self-Supervised Speech Models via Text-Based LLMS
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音模型评估任务，旨在解决自监督语音模型（SSL）在下游任务评估中的高成本问题。论文提出了一种无需额外训练或调参的新评估指标，利用大语言模型（LLMs）输入SSL生成的离散token序列和少量领域提示，计算均值对数似然作为评分。实验表明该评分与语音识别任务表现相关，并发现LLM还能提供可用于说话人验证的推理嵌入。**

- **链接: [http://arxiv.org/pdf/2510.04463v1](http://arxiv.org/pdf/2510.04463v1)**

> **作者:** Takashi Maekaku; Keita Goto; Jinchuan Tian; Yusuke Shinohara; Shinji Watanabe
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Self-Supervised Learning (SSL) has gained traction for its ability to learn rich representations with low labeling costs, applicable across diverse downstream tasks. However, assessing the downstream-task performance remains challenging due to the cost of extra training and evaluation. Existing methods for task-agnostic evaluation also require extra training or hyperparameter tuning. We propose a novel evaluation metric using large language models (LLMs). By inputting discrete token sequences and minimal domain cues derived from SSL models into LLMs, we obtain the mean log-likelihood; these cues guide in-context learning, rendering the score more reliable without extra training or hyperparameter tuning. Experimental results show a correlation between LLM-based scores and automatic speech recognition task. Additionally, our findings reveal that LLMs not only functions as an SSL evaluation tools but also provides inference-time embeddings that are useful for speaker verification task.
>
---
#### [new 012] Audio Forensics Evaluation (SAFE) Challenge
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音频取证任务，旨在解决合成语音检测难题。随着TTS技术进步，合成语音更逼真且经处理后更难识别。论文提出SAFE挑战，提供包含90小时音频的多模型数据集，评估检测系统在不同复杂场景下的性能，推动相关研究发展。**

- **链接: [http://arxiv.org/pdf/2510.03387v1](http://arxiv.org/pdf/2510.03387v1)**

> **作者:** Kirill Trapeznikov; Paul Cummer; Pranay Pherwani; Jai Aslam; Michael S. Davinroy; Peter Bautista; Laura Cassani; Matthew Stamm; Jill Crisman
>
> **摘要:** The increasing realism of synthetic speech generated by advanced text-to-speech (TTS) models, coupled with post-processing and laundering techniques, presents a significant challenge for audio forensic detection. In this paper, we introduce the SAFE (Synthetic Audio Forensics Evaluation) Challenge, a fully blind evaluation framework designed to benchmark detection models across progressively harder scenarios: raw synthetic speech, processed audio (e.g., compression, resampling), and laundered audio intended to evade forensic analysis. The SAFE challenge consisted of a total of 90 hours of audio and 21,000 audio samples split across 21 different real sources and 17 different TTS models and 3 tasks. We present the challenge, evaluation design and tasks, dataset details, and initial insights into the strengths and limitations of current approaches, offering a foundation for advancing synthetic audio detection research. More information is available at \href{https://stresearch.github.io/SAFE/}{https://stresearch.github.io/SAFE/}.
>
---
#### [new 013] Adapting Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech Recognition
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 论文提出了一种基于Whisper的说话人标注多说话人语音识别模型，结合目标说话人建模与序列化输出训练。模型使用Diarization-Conditioned Whisper编码器提取目标说话人嵌入，通过共享解码器联合解码，实现对重叠语音的序列化转录，附带说话人标签和时间戳，提升了多说话人混合语音的识别效果。**

- **链接: [http://arxiv.org/pdf/2510.03723v1](http://arxiv.org/pdf/2510.03723v1)**

> **作者:** Martin Kocour; Martin Karafiat; Alexander Polok; Dominik Klement; Lukáš Burget; Jan Černocký
>
> **摘要:** We propose a speaker-attributed (SA) Whisper-based model for multi-talker speech recognition that combines target-speaker modeling with serialized output training (SOT). Our approach leverages a Diarization-Conditioned Whisper (DiCoW) encoder to extract target-speaker embeddings, which are concatenated into a single representation and passed to a shared decoder. This enables the model to transcribe overlapping speech as a serialized output stream with speaker tags and timestamps. In contrast to target-speaker ASR systems such as DiCoW, which decode each speaker separately, our approach performs joint decoding, allowing the decoder to condition on the context of all speakers simultaneously. Experiments show that the model outperforms existing SOT-based approaches and surpasses DiCoW on multi-talker mixtures (e.g., LibriMix).
>
---
#### [new 014] Probing Whisper for Dysarthric Speech in Detection and Assessment
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究大规模语音模型Whisper在构音障碍语音检测与评估中的应用。旨在分析Whisper-Medium模型编码器各层对病理语音的表示能力，通过线性分类、轮廓系数和互信息评估层重要性，并探讨微调后的适应性变化。属于语音病理分析任务，用于指导临床评估工具的构建。**

- **链接: [http://arxiv.org/pdf/2510.04219v1](http://arxiv.org/pdf/2510.04219v1)**

> **作者:** Zhengjun Yue; Devendra Kayande; Zoran Cvetkovic; Erfan Loweimi
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Large-scale end-to-end models such as Whisper have shown strong performance on diverse speech tasks, but their internal behavior on pathological speech remains poorly understood. Understanding how dysarthric speech is represented across layers is critical for building reliable and explainable clinical assessment tools. This study probes the Whisper-Medium model encoder for dysarthric speech for detection and assessment (i.e., severity classification). We evaluate layer-wise embeddings with a linear classifier under both single-task and multi-task settings, and complement these results with Silhouette scores and mutual information to provide perspectives on layer informativeness. To examine adaptability, we repeat the analysis after fine-tuning Whisper on a dysarthric speech recognition task. Across metrics, the mid-level encoder layers (13-15) emerge as most informative, while fine-tuning induces only modest changes. The findings improve the interpretability of Whisper's embeddings and highlight the potential of probing analyses to guide the use of large-scale pretrained models for pathological speech.
>
---
#### [new 015] Enhancing Speaker Verification with w2v-BERT 2.0 and Knowledge Distillation guided Structured Pruning
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于说话人验证任务，旨在提升验证准确率并压缩模型。作者使用大规模预训练模型w2v-BERT 2.0，结合MFA结构和LoRA进行高效微调，取得最优性能。同时引入知识蒸馏指导的结构化剪枝，在显著压缩模型的同时仅造成微小性能损失。**

- **链接: [http://arxiv.org/pdf/2510.04213v1](http://arxiv.org/pdf/2510.04213v1)**

> **作者:** Ze Li; Ming Cheng; Ming Li
>
> **摘要:** Large-scale self-supervised Pre-Trained Models (PTMs) have shown significant improvements in the speaker verification (SV) task by providing rich feature representations. In this paper, we utilize w2v-BERT 2.0, a model with approximately 600 million parameters trained on 450 million hours of unlabeled data across 143 languages, for the SV task. The MFA structure with Layer Adapter is employed to process the multi-layer feature outputs from the PTM and extract speaker embeddings. Additionally, we incorporate LoRA for efficient fine-tuning. Our model achieves state-of-the-art results with 0.12% and 0.55% EER on the Vox1-O and Vox1-H test sets, respectively. Furthermore, we apply knowledge distillation guided structured pruning, reducing the model size by 80% while achieving only a 0.04% EER degradation. Source code and models are released at https://github.com/ZXHY-82/w2v-BERT-2.0_SV.
>
---
#### [new 016] Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多说话人自动语音识别（ASR）任务，旨在解决现有系统因需为每位说话人单独运行ASR模型而导致的高推理成本问题。论文提出将说话人活动信号转换为与说话人无关的两个流，以降低计算开销，同时设计新策略保持识别性能。实验表明该方法在AMI和ICSI数据集上有效。**

- **链接: [http://arxiv.org/pdf/2510.03630v1](http://arxiv.org/pdf/2510.03630v1)**

> **作者:** Xiluo He; Alexander Polok; Jesús Villalba; Thomas Thebaud; Matthew Maciejewski
>
> **摘要:** An increasingly common training paradigm for multi-talker automatic speech recognition (ASR) is to use speaker activity signals to adapt single-speaker ASR models for overlapping speech. Although effective, these systems require running the ASR model once per speaker, resulting in inference costs that scale with the number of speakers and limiting their practicality. In this work, we propose a method that decouples the inference cost of activity-conditioned ASR systems from the number of speakers by converting speaker-specific activity outputs into two speaker-agnostic streams. A central challenge is that na\"ively merging speaker activities into streams significantly degrades recognition, since pretrained ASR models assume contiguous, single-speaker inputs. To address this, we design new heuristics aimed at preserving conversational continuity and maintaining compatibility with existing systems. We show that our approach is compatible with Diarization-Conditioned Whisper (DiCoW) to greatly reduce runtimes on the AMI and ICSI meeting datasets while retaining competitive performance.
>
---
#### [new 017] From Qubits to Rhythm: Exploring Quantum Random Walks in Rhythmspaces
- **分类: quant-ph; cs.CY; cs.SD; eess.AS**

- **简介: 该论文属于音乐与量子计算交叉领域的任务，旨在解决如何利用量子计算生成复杂节奏模式的问题。论文设计了一种将二维量子随机游走映射到节奏空间的算法，通过分解为两个一维游走以降低电路深度，并引入经典势场调控概率分布，最终将游走路径转化为MIDI鼓点模式，实现了量子计算在音乐生成中的应用。**

- **链接: [http://arxiv.org/pdf/2510.03836v1](http://arxiv.org/pdf/2510.03836v1)**

> **作者:** María Aguado-Yáñez; Karl Jansen; Daniel Gómez-Marín; Sergi Jordà
>
> **备注:** 17 pages. 11 figures. Papers from arXiv cited: arXiv:2311.13313, arXiv:2411.09549
>
> **摘要:** A quantum computing algorithm for rhythm generation is presented, which aims to expand and explore quantum computing applications in the arts, particularly in music. The algorithm maps quantum random walk trajectories onto a rhythmspace -- a 2D interface that interpolates rhythmic patterns. The methodology consists of three stages. The first stage involves designing quantum computing algorithms and establishing a mapping between the qubit space and the rhythmspace. To minimize circuit depth, a decomposition of a 2D quantum random walk into two 1D quantum random walks is applied. The second stage focuses on biasing the directionality of quantum random walks by introducing classical potential fields, adjusting the probability distribution of the wave function based on the position gradient within these fields. Four potential fields are implemented: a null potential, a linear field, a Gaussian potential, and a Gaussian potential under inertial dynamics. The third stage addresses the sonification of these paths by generating MIDI drum pattern messages and transmitting them to a Digital Audio Workstation (DAW). This work builds upon existing literature that applies quantum computing to simpler qubit spaces with a few positions, extending the formalism to a 2D x-y plane. It serves as a proof of concept for scalable quantum computing-based generative random walk algorithms in music and audio applications. Furthermore, the approach is applicable to generic multidimensional sound spaces, as the algorithms are not strictly constrained to rhythm generation and can be adapted to different musical structures.
>
---
#### [new 018] Cross-Lingual Multi-Granularity Framework for Interpretable Parkinson's Disease Diagnosis from Speech
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于医疗诊断任务，旨在通过语音实现跨语言的帕金森病检测。现有方法分析整段语音，可能忽略特定语音单元的诊断价值。作者提出多粒度框架，提取音素、音节和单词级别特征，使用双向LSTM与多头注意力模型，在多种语言数据上验证。音素级分析表现最佳，并发现关键语音特征与临床方法一致。**

- **链接: [http://arxiv.org/pdf/2510.03758v1](http://arxiv.org/pdf/2510.03758v1)**

> **作者:** Ilias Tougui; Mehdi Zakroum; Mounir Ghogho
>
> **摘要:** Parkinson's Disease (PD) affects over 10 million people worldwide, with speech impairments in up to 89% of patients. Current speech-based detection systems analyze entire utterances, potentially overlooking the diagnostic value of specific phonetic elements. We developed a granularity-aware approach for multilingual PD detection using an automated pipeline that extracts time-aligned phonemes, syllables, and words from recordings. Using Italian, Spanish, and English datasets, we implemented a bidirectional LSTM with multi-head attention to compare diagnostic performance across the different granularity levels. Phoneme-level analysis achieved superior performance with AUROC of 93.78% +- 2.34% and accuracy of 92.17% +- 2.43%. This demonstrates enhanced diagnostic capability for cross-linguistic PD detection. Importantly, attention analysis revealed that the most informative speech features align with those used in established clinical protocols: sustained vowels (/a/, /e/, /o/, /i/) at phoneme level, diadochokinetic syllables (/ta/, /pa/, /la/, /ka/) at syllable level, and /pataka/ sequences at word level. Source code will be available at https://github.com/jetliqs/clearpd.
>
---
#### [new 019] Robustness assessment of large audio language models in multiple-choice evaluation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型评估任务，旨在解决现有评估方法忽略选项顺序和表述变化影响的问题。作者通过分析多个模型和数据集，发现模型对选项顺序和问题改写敏感，进而提出了一种更全面的评估协议和指标。**

- **链接: [http://arxiv.org/pdf/2510.04584v1](http://arxiv.org/pdf/2510.04584v1)**

> **作者:** Fernando López; Santosh Kesiraju; Jordi Luque
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Recent advances in large audio language models (LALMs) have primarily been assessed using a multiple-choice question answering (MCQA) framework. However, subtle changes, such as shifting the order of choices, result in substantially different results. Existing MCQA frameworks do not account for this variability and report a single accuracy number per benchmark or category. We dive into the MCQA evaluation framework and conduct a systematic study spanning three benchmarks (MMAU, MMAR and MMSU) and four models: Audio Flamingo 2, Audio Flamingo 3, Qwen2.5-Omni-7B-Instruct, and Kimi-Audio-7B-Instruct. Our findings indicate that models are sensitive not only to the ordering of choices, but also to the paraphrasing of the question and the choices. Finally, we propose a simpler evaluation protocol and metric that account for subtle variations and provide a more detailed evaluation report of LALMs within the MCQA framework.
>
---
#### [new 020] Drax: Speech Recognition with Discrete Flow Matching
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决非自回归（NAR）模型在语音识别中的潜力未被充分探索的问题。作者提出了Drax，一种基于离散流匹配的框架，通过构建音频条件概率路径，提升识别准确率与解码效率。**

- **链接: [http://arxiv.org/pdf/2510.04162v1](http://arxiv.org/pdf/2510.04162v1)**

> **作者:** Aviv Navon; Aviv Shamsian; Neta Glazer; Yael Segal-Feldman; Gill Hetz; Joseph Keshet; Ethan Fetaya
>
> **摘要:** Diffusion and flow-based non-autoregressive (NAR) models have shown strong promise in large language modeling, however, their potential for automatic speech recognition (ASR) remains largely unexplored. We propose Drax, a discrete flow matching framework for ASR that enables efficient parallel decoding. To better align training with inference, we construct an audio-conditioned probability path that guides the model through trajectories resembling likely intermediate inference errors, rather than direct random noise to target transitions. Our theoretical analysis links the generalization gap to divergences between training and inference occupancies, controlled by cumulative velocity errors, thereby motivating our design choice. Empirical evaluation demonstrates that our approach attains recognition accuracy on par with state-of-the-art speech models while offering improved accuracy-efficiency trade-offs, highlighting discrete flow matching as a promising direction for advancing NAR ASR.
>
---
#### [new 021] A Multilingual Framework for Dysarthria: Detection, Severity Classification, Speech-to-Text, and Clean Speech Generation
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多语言语音处理任务，旨在解决构音障碍患者的语音识别与交流困难问题。研究提出统一AI框架，涵盖检测、分类、语音转文字、语音生成等六项功能，跨语言（英、俄、德）验证有效性，尤其在低资源场景下表现突出，有助于提升构音障碍者的沟通质量。**

- **链接: [http://arxiv.org/pdf/2510.03986v1](http://arxiv.org/pdf/2510.03986v1)**

> **作者:** Ananya Raghu; Anisha Raghu; Nithika Vivek; Sofie Budman; Omar Mansour
>
> **摘要:** Dysarthria is a motor speech disorder that results in slow and often incomprehensible speech. Speech intelligibility significantly impacts communication, leading to barriers in social interactions. Dysarthria is often a characteristic of neurological diseases including Parkinson's and ALS, yet current tools lack generalizability across languages and levels of severity. In this study, we present a unified AI-based multilingual framework that addresses six key components: (1) binary dysarthria detection, (2) severity classification, (3) clean speech generation, (4) speech-to-text conversion, (5) emotion detection, and (6) voice cloning. We analyze datasets in English, Russian, and German, using spectrogram-based visualizations and acoustic feature extraction to inform model training. Our binary detection model achieved 97% accuracy across all three languages, demonstrating strong generalization across languages. The severity classification model also reached 97% test accuracy, with interpretable results showing model attention focused on lower harmonics. Our translation pipeline, trained on paired Russian dysarthric and clean speech, reconstructed intelligible outputs with low training (0.03) and test (0.06) L1 losses. Given the limited availability of English dysarthric-clean pairs, we fine-tuned the Russian model on English data and achieved improved losses of 0.02 (train) and 0.03 (test), highlighting the promise of cross-lingual transfer learning for low-resource settings. Our speech-to-text pipeline achieved a Word Error Rate of 0.1367 after three epochs, indicating accurate transcription on dysarthric speech and enabling downstream emotion recognition and voice cloning from transcribed speech. Overall, the results and products of this study can be used to diagnose dysarthria and improve communication and understanding for patients across different languages.
>
---
#### [new 022] Evaluating High-Resolution Piano Sustain Pedal Depth Estimation with Musically Informed Metrics
- **分类: cs.IR; cs.SD; eess.AS**

- **简介: 该论文属于音乐信号处理任务，旨在解决钢琴踏板深度估计评估不全面的问题。现有方法仅依赖帧级指标，忽略音乐特征。作者提出新评估框架，结合动作级和手势级分析，更准确衡量踏板动作的方向、时序和轮廓相似性。实验表明引入MIDI信息的模型在动作和手势级别表现更优。**

- **链接: [http://arxiv.org/pdf/2510.03750v1](http://arxiv.org/pdf/2510.03750v1)**

> **作者:** Hanwen Zhang; Kun Fang; Ziyu Wang; Ichiro Fujinaga
>
> **摘要:** Evaluation for continuous piano pedal depth estimation tasks remains incomplete when relying only on conventional frame-level metrics, which overlook musically important features such as direction-change boundaries and pedal curve contours. To provide more interpretable and musically meaningful insights, we propose an evaluation framework that augments standard frame-level metrics with an action-level assessment measuring direction and timing using segments of press/hold/release states and a gesture-level analysis that evaluates contour similarity of each press-release cycle. We apply this framework to compare an audio-only baseline with two variants: one incorporating symbolic information from MIDI, and another trained in a binary-valued setting, all within a unified architecture. Results show that the MIDI-informed model significantly outperforms the others at action and gesture levels, despite modest frame-level gains. These findings demonstrate that our framework captures musically relevant improvements indiscernible by traditional metrics, offering a more practical and effective approach to evaluating pedal depth estimation models.
>
---
#### [new 023] Efficiency vs. Efficacy: Assessing the Compression Ratio-Dice Score Relationship through a Simple Benchmarking Framework for Cerebrovascular 3D Segmentation
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于医学图像处理任务，旨在解决3D医学影像数据过大影响协作的问题。研究使用ZFP压缩技术，在不影响脑血管分割性能的前提下，实现高效数据压缩，验证了其在保持高分割精度（Dice系数约0.87656）的同时显著减少数据量（最高压缩比22.89:1），从而促进大规模医学数据的共享与研究。**

- **链接: [http://arxiv.org/pdf/2510.03769v1](http://arxiv.org/pdf/2510.03769v1)**

> **作者:** Shimaa Elbana; Ahmad Kamal; Shahd Ahmed Ali; Ahmad Al-Kabbany
>
> **摘要:** The increasing size and complexity of medical imaging datasets, particularly in 3D formats, present significant barriers to collaborative research and transferability. This study investigates whether the ZFP compression technique can mitigate these challenges without compromising the performance of automated cerebrovascular segmentation, a critical first step in intracranial aneurysm detection. We apply ZFP in both its error tolerance and fixed-rate modes to a large scale, and one of the most recent, datasets in the literature, 3D medical dataset containing ground-truth vascular segmentations. The segmentation quality on the compressed volumes is rigorously compared to the uncompressed baseline (Dice approximately equals 0.8774). Our findings reveal that ZFP can achieve substantial data reduction--up to a 22.89:1 ratio in error tolerance mode--while maintaining a high degree of fidelity, with the mean Dice coefficient remaining high at 0.87656. These results demonstrate that ZFP is a viable and powerful tool for enabling more efficient and accessible research on large-scale medical datasets, fostering broader collaboration across the community.
>
---
#### [new 024] UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models
- **分类: eess.AS; cs.SD**

- **简介: 论文提出UniVoice，统一自动语音识别（ASR）与文本到语音（TTS）任务，通过连续表示与大语言模型实现。解决现有方法分离建模导致的信息损失与性能限制，采用自回归与流匹配结合，并设计双注意力机制与语音填补方法，实现高质量语音识别与零样本语音克隆。**

- **链接: [http://arxiv.org/pdf/2510.04593v1](http://arxiv.org/pdf/2510.04593v1)**

> **作者:** Wenhao Guan; Zhikang Niu; Ziyue Jiang; Kaidi Wang; Peijie Chen; Qingyang Hong; Lin Li; Xie Chen
>
> **摘要:** Large language models (LLMs) have demonstrated promising performance in both automatic speech recognition (ASR) and text-to-speech (TTS) systems, gradually becoming the mainstream approach. However, most current approaches address these tasks separately rather than through a unified framework. This work aims to integrate these two tasks into one unified model. Although discrete speech tokenization enables joint modeling, its inherent information loss limits performance in both recognition and generation. In this work, we present UniVoice, a unified LLM framework through continuous representations that seamlessly integrates speech recognition and synthesis within a single model. Our approach combines the strengths of autoregressive modeling for speech recognition with flow matching for high-quality generation. To mitigate the inherent divergence between autoregressive and flow-matching models, we further design a dual attention mechanism, which switches between a causal mask for recognition and a bidirectional attention mask for synthesis. Furthermore, the proposed text-prefix-conditioned speech infilling method enables high-fidelity zero-shot voice cloning. Experimental results demonstrate that our method can achieve or exceed current single-task modeling methods in both ASR and zero-shot TTS tasks. This work explores new possibilities for end-to-end speech understanding and generation.
>
---
#### [new 025] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文属于音频-视觉语音识别（AVSR）任务，旨在解决现有模型计算成本高、压缩率固定、跨尺度泛化能力弱的问题。作者提出MoME框架，结合Matryoshka表示学习与稀疏专家混合机制，实现多粒度压缩与动态资源分配，提升了模型效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.04136v1](http://arxiv.org/pdf/2510.04136v1)**

> **作者:** Umberto Cappellazzo; Minsu Kim; Pingchuan Ma; Honglie Chen; Xubo Liu; Stavros Petridis; Maja Pantic
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models (LLMs) have recently shown strong potential in audio-visual speech recognition (AVSR), but their high computational demands and sensitivity to token granularity limit their practicality in resource-constrained settings. Token compression methods can reduce inference cost, but they require fixing a compression rate in advance and produce a single fixed-length output, offering no flexibility to balance information density and efficiency at inference time. Matryoshka representation learning (MRL) addresses this by enabling a single model to operate across multiple token granularities, allowing compression rates to be adjusted dynamically. However, current MRL-based methods treat each scale independently during training, limiting cross-scale generalization, robustness at high compression, and interpretability. To overcome these limitations, we propose MoME (Mixture of Matryoshka Experts), a novel framework that integrates sparse Mixture-of-Experts (MoE) into MRL-based LLMs for AVSR. MoME augments a frozen LLM with top-k routed and shared experts, allowing dynamic capacity allocation across scales and modalities. A shared router promotes consistent expert activation across granularities, enabling compressed sequences to benefit from representations learned at lower compression. Experiments on LRS2 and LRS3 demonstrate that MoME achieves state-of-the-art performance across AVSR, ASR, and VSR tasks, while requiring significantly fewer parameters and maintaining robustness under noise. MoME unifies the adaptability of MRL with the efficiency of MoE, offering a scalable and interpretable solution for resource-aware speech recognition.
>
---
#### [new 026] Differentiable physics for sound field reconstruction
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于声音场重建任务，旨在从有限空间采样中重建声场。作者提出一种基于可微分物理的方法，利用神经网络近似波动方程的初始条件，并通过可微分数值求解器计算微分算子，以物理约束提升重建效果。实验表明，该方法在极端数据稀缺条件下优于传统物理信息神经网络。**

- **链接: [http://arxiv.org/pdf/2510.04459v1](http://arxiv.org/pdf/2510.04459v1)**

> **作者:** Samuel A. Verburg; Efren Fernandez-Grande; Peter Gerstoft
>
> **备注:** 28 pages plus references, 8 figures, full journal paper
>
> **摘要:** Sound field reconstruction involves estimating sound fields from a limited number of spatially distributed observations. This work introduces a differentiable physics approach for sound field reconstruction, where the initial conditions of the wave equation are approximated with a neural network, and the differential operator is computed with a differentiable numerical solver. The use of a numerical solver enables a stable network training while enforcing the physics as a strong constraint, in contrast to conventional physics-informed neural networks, which include the physics as a constraint in the loss function. We introduce an additional sparsity-promoting constraint to achieve meaningful solutions even under severe undersampling conditions. Experiments demonstrate that the proposed approach can reconstruct sound fields under extreme data scarcity, achieving higher accuracy and better convergence compared to physics-informed neural networks.
>
---
#### [new 027] A MATLAB toolbox for Computation of Speech Transmission Index (STI)
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音信号处理任务，旨在解决STI计算工具不开放、依赖专用设备的问题。作者基于MATLAB实现了符合IEC标准的STI计算工具，支持直接和间接方法及STIPA协议，并通过测试和对比验证了其准确性，提供开源代码。**

- **链接: [http://arxiv.org/pdf/2510.03825v1](http://arxiv.org/pdf/2510.03825v1)**

> **作者:** Pavel Rajmic; Jiří Schimmel; Šimon Cieslar
>
> **摘要:** The speech transmission index (STI) is a popular simple metric for the prediction of speech intelligibility when speech is passed through a transmission channel. Computation of STI from acoustic measurements is described in the IEC 60268-16:2020 standard. Though, reliable implementations of STI are not publicly accessible and are frequently limited to the use with a proprietary measurement hardware. We present a Matlab STI implementation of both the direct and indirect approaches according to the standard, including the shortened STIPA protocol. The suggested implementation meets prescribed requirements, as evidenced by tests on reference signals. Additionally, we conducted a verification measurement in comparison to a commercial measurement device. Our software comes with open source code.
>
---
## 更新

#### [replaced 001] VibE-SVC: Vibrato Extraction with High-frequency F0 Contour for Singing Voice Conversion
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20794v2](http://arxiv.org/pdf/2505.20794v2)**

> **作者:** Joon-Seung Choi; Dong-Min Byun; Hyung-Seok Oh; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech
>
> **摘要:** Controlling singing style is crucial for achieving an expressive and natural singing voice. Among the various style factors, vibrato plays a key role in conveying emotions and enhancing musical depth. However, modeling vibrato remains challenging due to its dynamic nature, making it difficult to control in singing voice conversion. To address this, we propose VibESVC, a controllable singing voice conversion model that explicitly extracts and manipulates vibrato using discrete wavelet transform. Unlike previous methods that model vibrato implicitly, our approach decomposes the F0 contour into frequency components, enabling precise transfer. This allows vibrato control for enhanced flexibility. Experimental results show that VibE-SVC effectively transforms singing styles while preserving speaker similarity. Both subjective and objective evaluations confirm high-quality conversion.
>
---
#### [replaced 002] Low Resource Audio Codec Challenge Baseline Systems
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.00264v2](http://arxiv.org/pdf/2510.00264v2)**

> **作者:** Yusuf Ziya Isik; Rafał Łaganowski
>
> **备注:** Low-Resource Audio Codec Challenge 2025
>
> **摘要:** The Low-Resource Audio Codec (LRAC) Challenge aims to advance neural audio coding for deployment in resource-constrained environments. The first edition focuses on low-resource neural speech codecs that must operate reliably under everyday noise and reverberation, while satisfying strict constraints on computational complexity, latency, and bitrate. Track 1 targets transparency codecs, which aim to preserve the perceptual transparency of input speech under mild noise and reverberation. Track 2 addresses enhancement codecs, which combine coding and compression with denoising and dereverberation. This paper presents the official baseline systems for both tracks in the 2025 LRAC Challenge. The baselines are convolutional neural codec models with Residual Vector Quantization, trained end-to-end using a combination of adversarial and reconstruction objectives. We detail the data filtering and augmentation strategies, model architectures, optimization procedures, and checkpoint selection criteria.
>
---
#### [replaced 003] StereoFoley: Object-Aware Stereo Audio Generation from Video
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.18272v3](http://arxiv.org/pdf/2509.18272v3)**

> **作者:** Tornike Karchkhadze; Kuan-Lin Chen; Mojtaba Heydari; Robert Henzel; Alessandro Toso; Mehrez Souden; Joshua Atkins
>
> **摘要:** We present StereoFoley, a video-to-audio generation framework that produces semantically aligned, temporally synchronized, and spatially accurate stereo sound at 48 kHz. While recent generative video-to-audio models achieve strong semantic and temporal fidelity, they largely remain limited to mono or fail to deliver object-aware stereo imaging, constrained by the lack of professionally mixed, spatially accurate video-to-audio datasets. First, we develop and train a base model that generates stereo audio from video, achieving state-of-the-art in both semantic accuracy and synchronization. Next, to overcome dataset limitations, we introduce a synthetic data generation pipeline that combines video analysis, object tracking, and audio synthesis with dynamic panning and distance-based loudness controls, enabling spatially accurate object-aware sound. Finally, we fine-tune the base model on this synthetic dataset, yielding clear object-audio correspondence. Since no established metrics exist, we introduce stereo object-awareness measures and validate it through a human listening study, showing strong correlation with perception. This work establishes the first end-to-end framework for stereo object-aware video-to-audio generation, addressing a critical gap and setting a new benchmark in the field.
>
---
#### [replaced 004] Robust MRI Reconstruction by Smoothed Unrolling (SMUG)
- **分类: eess.IV; cs.AI; cs.CV; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2312.07784v3](http://arxiv.org/pdf/2312.07784v3)**

> **作者:** Shijun Liang; Van Hoang Minh Nguyen; Jinghan Jia; Ismail Alkhouri; Sijia Liu; Saiprasad Ravishankar
>
> **摘要:** As the popularity of deep learning (DL) in the field of magnetic resonance imaging (MRI) continues to rise, recent research has indicated that DL-based MRI reconstruction models might be excessively sensitive to minor input disturbances, including worst-case additive perturbations. This sensitivity often leads to unstable, aliased images. This raises the question of how to devise DL techniques for MRI reconstruction that can be robust to train-test variations. To address this problem, we propose a novel image reconstruction framework, termed Smoothed Unrolling (SMUG), which advances a deep unrolling-based MRI reconstruction model using a randomized smoothing (RS)-based robust learning approach. RS, which improves the tolerance of a model against input noises, has been widely used in the design of adversarial defense approaches for image classification tasks. Yet, we find that the conventional design that applies RS to the entire DL-based MRI model is ineffective. In this paper, we show that SMUG and its variants address the above issue by customizing the RS process based on the unrolling architecture of a DL-based MRI reconstruction model. Compared to the vanilla RS approach, we show that SMUG improves the robustness of MRI reconstruction with respect to a diverse set of instability sources, including worst-case and random noise perturbations to input measurements, varying measurement sampling rates, and different numbers of unrolling steps. Furthermore, we theoretically analyze the robustness of our method in the presence of perturbations.
>
---
#### [replaced 005] Advanced Clustering Techniques for Speech Signal Enhancement: A Review and Metanalysis of Fuzzy C-Means, K-Means, and Kernel Fuzzy C-Means Methods
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.19448v2](http://arxiv.org/pdf/2409.19448v2)**

> **作者:** Abdulhady Abas Abdullah; Aram Mahmood Ahmed; Tarik Rashid; Hadi Veisi
>
> **摘要:** Speech signal processing is a cornerstone of modern communication technologies, tasked with improving the clarity and comprehensibility of audio data in noisy environments. The primary challenge in this field is the effective separation and recognition of speech from background noise, crucial for applications ranging from voice-activated assistants to automated transcription services. The quality of speech recognition directly impacts user experience and accessibility in technology-driven communication. This review paper explores advanced clustering techniques, particularly focusing on the Kernel Fuzzy C-Means (KFCM) method, to address these challenges. Our findings indicate that KFCM, compared to traditional methods like K-Means (KM) and Fuzzy C-Means (FCM), provides superior performance in handling non-linear and non-stationary noise conditions in speech signals. The most notable outcome of this review is the adaptability of KFCM to various noisy environments, making it a robust choice for speech enhancement applications. Additionally, the paper identifies gaps in current methodologies, such as the need for more dynamic clustering algorithms that can adapt in real time to changing noise conditions without compromising speech recognition quality. Key contributions include a detailed comparative analysis of current clustering algorithms and suggestions for further integrating hybrid models that combine KFCM with neural networks to enhance speech recognition accuracy. Through this review, we advocate for a shift towards more sophisticated, adaptive clustering techniques that can significantly improve speech enhancement and pave the way for more resilient speech processing systems.
>
---
#### [replaced 006] Automated Defect Detection for Mass-Produced Electronic Components Based on YOLO Object Detection Models
- **分类: cs.CV; cs.AI; cs.LG; eess.SP; 68T07, 68U10; I.4.8; I.2.10**

- **链接: [http://arxiv.org/pdf/2510.01914v2](http://arxiv.org/pdf/2510.01914v2)**

> **作者:** Wei-Lung Mao; Chun-Chi Wang; Po-Heng Chou; Yen-Ting Liu
>
> **备注:** 12 pages, 16 figures, 7 tables, and published in IEEE Sensors Journal
>
> **摘要:** Since the defect detection of conventional industry components is time-consuming and labor-intensive, it leads to a significant burden on quality inspection personnel and makes it difficult to manage product quality. In this paper, we propose an automated defect detection system for the dual in-line package (DIP) that is widely used in industry, using digital camera optics and a deep learning (DL)-based model. The two most common defect categories of DIP are examined: (1) surface defects, and (2) pin-leg defects. However, the lack of defective component images leads to a challenge for detection tasks. To solve this problem, the ConSinGAN is used to generate a suitable-sized dataset for training and testing. Four varieties of the YOLO model are investigated (v3, v4, v7, and v9), both in isolation and with the ConSinGAN augmentation. The proposed YOLOv7 with ConSinGAN is superior to the other YOLO versions in accuracy of 95.50\%, detection time of 285 ms, and is far superior to threshold-based approaches. In addition, the supervisory control and data acquisition (SCADA) system is developed, and the associated sensor architecture is described. The proposed automated defect detection can be easily established with numerous types of defects or insufficient defect data.
>
---
#### [replaced 007] StressTest: Can YOUR Speech LM Handle the Stress?
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.22765v2](http://arxiv.org/pdf/2505.22765v2)**

> **作者:** Iddo Yosha; Gallil Maimon; Yossi Adi
>
> **摘要:** Sentence stress refers to emphasis on words within a spoken utterance to highlight or contrast an idea. It is often used to imply an underlying intention not explicitly stated. Recent speech-aware language models (SLMs) have enabled direct audio processing, allowing models to access the full richness of speech to perform audio reasoning tasks such as spoken question answering. Despite the crucial role of sentence stress in shaping meaning and intent, it remains largely overlooked in evaluation and development of SLMs. We address this gap by introducing StressTest, a benchmark designed to evaluate models' ability to distinguish between meanings of speech based on the stress pattern. We evaluate leading SLMs, and find that despite their overall capabilities, they perform poorly on such tasks. Hence, we propose a novel data generation pipeline, and create Stress-17k, a training set that simulates change of meaning implied by stress variation. Results suggest, that our finetuned model, StresSLM, generalizes well to real recordings and notably outperforms existing SLMs on sentence stress reasoning and detection. Models, code, data, samples - pages.cs.huji.ac.il/adiyoss-lab/stresstest.
>
---
#### [replaced 008] SonicMaster: Towards Controllable All-in-One Music Restoration and Mastering
- **分类: cs.SD; cs.AI; cs.MM; eess.AS; 68T07, 94A12, 68U10; I.2.10; H.5.5; J.5**

- **链接: [http://arxiv.org/pdf/2508.03448v2](http://arxiv.org/pdf/2508.03448v2)**

> **作者:** Jan Melechovsky; Ambuj Mehrish; Abhinaba Roy; Dorien Herremans
>
> **摘要:** Music recordings often suffer from audio quality issues such as excessive reverberation, distortion, clipping, tonal imbalances, and a narrowed stereo image, especially when created in non-professional settings without specialized equipment or expertise. These problems are typically corrected using separate specialized tools and manual adjustments. In this paper, we introduce SonicMaster, the first unified generative model for music restoration and mastering that addresses a broad spectrum of audio artifacts with text-based control. SonicMaster is conditioned on natural language instructions to apply targeted enhancements, or can operate in an automatic mode for general restoration. To train this model, we construct the SonicMaster dataset, a large dataset of paired degraded and high-quality tracks by simulating common degradation types with nineteen degradation functions belonging to five enhancements groups: equalization, dynamics, reverb, amplitude, and stereo. Our approach leverages a flow-matching generative training paradigm to learn an audio transformation that maps degraded inputs to their cleaned, mastered versions guided by text prompts. Objective audio quality metrics demonstrate that SonicMaster significantly improves sound quality across all artifact categories. Furthermore, subjective listening tests confirm that listeners prefer SonicMaster's enhanced outputs over the original degraded audio, highlighting the effectiveness of our unified approach.
>
---
#### [replaced 009] Prompt-aware classifier free guidance for diffusion models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.22728v2](http://arxiv.org/pdf/2509.22728v2)**

> **作者:** Xuanhao Zhang; Chang Li
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Diffusion models have achieved remarkable progress in image and audio generation, largely due to Classifier-Free Guidance. However, the choice of guidance scale remains underexplored: a fixed scale often fails to generalize across prompts of varying complexity, leading to oversaturation or weak alignment. We address this gap by introducing a prompt-aware framework that predicts scale-dependent quality and selects the optimal guidance at inference. Specifically, we construct a large synthetic dataset by generating samples under multiple scales and scoring them with reliable evaluation metrics. A lightweight predictor, conditioned on semantic embeddings and linguistic complexity, estimates multi-metric quality curves and determines the best scale via a utility function with regularization. Experiments on MSCOCO~2014 and AudioCaps show consistent improvements over vanilla CFG, enhancing fidelity, alignment, and perceptual preference. This work demonstrates that prompt-aware scale selection provides an effective, training-free enhancement for pretrained diffusion backbones.
>
---
#### [replaced 010] TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation
- **分类: cs.IR; cs.AI; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.09685v3](http://arxiv.org/pdf/2509.09685v3)**

> **作者:** Keunwoo Choi; Seungheon Doh; Juhan Nam
>
> **摘要:** We present TalkPlayData 2, a synthetic dataset for multimodal conversational music recommendation generated by an agentic data pipeline. In the proposed pipeline, multiple large language model (LLM) agents are created under various roles with specialized prompts and access to different parts of information, and the chat data is acquired by logging the conversation between the Listener LLM and the Recsys LLM. To cover various conversation scenarios, for each conversation, the Listener LLM is conditioned on a finetuned conversation goal. Finally, all the LLMs are multimodal with audio and images, allowing a simulation of multimodal recommendation and conversation. In the LLM-as-a-judge and subjective evaluation experiments, TalkPlayData 2 achieved the proposed goal in various aspects related to training a generative recommendation model for music. TalkPlayData 2 and its generation code are open-sourced at https://talkpl.ai/talkplaydata2.
>
---
#### [replaced 011] Fun-ASR Technical Report
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.12508v3](http://arxiv.org/pdf/2509.12508v3)**

> **作者:** Keyu An; Yanni Chen; Chong Deng; Changfeng Gao; Zhifu Gao; Bo Gong; Xiangang Li; Yabin Li; Xiang Lv; Yunjie Ji; Yiheng Jiang; Bin Ma; Haoneng Luo; Chongjia Ni; Zexu Pan; Yiping Peng; Zhendong Peng; Peiyao Wang; Hao Wang; Wen Wang; Wupeng Wang; Biao Tian; Zhentao Tan; Nan Yang; Bin Yuan; Jieping Ye; Jixing Yu; Qinglin Zhang; Kun Zou; Han Zhao; Shengkui Zhao; Jingren Zhou
>
> **备注:** Authors are listed in alphabetical order
>
> **摘要:** In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present Fun-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, Fun-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, Fun-ASR achieves state-of-the-art performance on real application datasets, demonstrating its effectiveness and robustness in practical settings.
>
---
#### [replaced 012] HNote: Extending YNote with Hexadecimal Encoding for Fine-Tuning LLMs in Music Modeling
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.25694v2](http://arxiv.org/pdf/2509.25694v2)**

> **作者:** Hung-Ying Chu; Shao-Yu Wei; Guan-Wei Chen; Tzu-Wei Hung; ChengYang Tsai; Yu-Cheng Lin
>
> **摘要:** Recent advances in large language models (LLMs) have created new opportunities for symbolic music generation. However, existing formats such as MIDI, ABC, and MusicXML are either overly complex or structurally inconsistent, limiting their suitability for token-based learning architectures. To address these challenges, we propose HNote, a novel hexadecimal-based notation system extended from YNote, which encodes both pitch and duration within a fixed 32-unit measure framework. This design ensures alignment, reduces ambiguity, and is directly compatible with LLM architectures. We converted 12,300 Jiangnan-style songs generated from traditional folk pieces from YNote into HNote, and fine-tuned LLaMA-3.1(8B) using parameter-efficient LoRA. Experimental results show that HNote achieves a syntactic correctness rate of 82.5%, and BLEU and ROUGE evaluations demonstrate strong symbolic and structural similarity, producing stylistically coherent compositions. This study establishes HNote as an effective framework for integrating LLMs with cultural music modeling.
>
---
#### [replaced 013] HiKE: Hierarchical Evaluation Framework for Korean-English Code-Switching Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.24613v2](http://arxiv.org/pdf/2509.24613v2)**

> **作者:** Gio Paik; Yongbeom Kim; Soungmin Lee; Sangmin Ahn; Chanwoo Kim
>
> **备注:** Updated table 2 and 3 due to bug fix, Under Review
>
> **摘要:** Despite advances in multilingual automatic speech recognition (ASR), code-switching (CS), the mixing of languages within an utterance common in daily speech, remains a severely underexplored challenge. In this paper, we introduce HiKE: the Hierarchical Korean-English code-switching benchmark, the first globally accessible evaluation framework for Korean-English CS, aiming to provide a means for the precise evaluation of multilingual ASR models and to foster research in the field. The proposed framework not only consists of high-quality, natural CS data across various topics, but also provides meticulous loanword labels and a hierarchical CS-level labeling scheme (word, phrase, and sentence) that together enable a systematic evaluation of a model's ability to handle each distinct level of code-switching. Through evaluations of diverse multilingual ASR models and fine-tuning experiments, this paper demonstrates that although most multilingual ASR models initially exhibit inadequate CS-ASR performance, this capability can be enabled through fine-tuning with synthetic CS data. HiKE is available at https://github.com/ThetaOne-AI/HiKE
>
---
#### [replaced 014] Poisson multi-Bernoulli mixture filter for trajectory measurements
- **分类: eess.SP; cs.CV; stat.AP**

- **链接: [http://arxiv.org/pdf/2504.08421v2](http://arxiv.org/pdf/2504.08421v2)**

> **作者:** Marco Fontana; Ángel F. García-Fernández; Simon Maskell
>
> **备注:** 16 pages, 9 figures, journal paper
>
> **摘要:** This paper presents a Poisson multi-Bernoulli mixture (PMBM) filter for multi-target filtering based on sensor measurements that are sets of trajectories in the last two-time step window. The proposed filter, the trajectory measurement PMBM (TM-PMBM) filter, propagates a PMBM density on the set of target states. In prediction, the filter obtains the PMBM density on the set of trajectories over the last two time steps. This density is then updated with the set of trajectory measurements. After the update step, the PMBM posterior on the set of two-step trajectories is marginalised to obtain a PMBM density on the set of target states. The filter provides a closed-form solution for multi-target filtering based on sets of trajectory measurements, estimating the set of target states at the end of each time window. Additionally, the paper proposes computationally lighter alternatives to the TM-PMBM filter by deriving a Poisson multi-Bernoulli (PMB) density through Kullback-Leibler divergence minimisation in an augmented space with auxiliary variables. The performance of the proposed filters are evaluated in a simulation study.
>
---
#### [replaced 015] Latent Multi-view Learning for Robust Environmental Sound Representations
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.02500v2](http://arxiv.org/pdf/2510.02500v2)**

> **作者:** Sivan Ding; Julia Wilkins; Magdalena Fuentes; Juan Pablo Bello
>
> **备注:** Accepted to DCASE 2025 Workshop. 4+1 pages, 2 figures, 2 tables
>
> **摘要:** Self-supervised learning (SSL) approaches, such as contrastive and generative methods, have advanced environmental sound representation learning using unlabeled data. However, how these approaches can complement each other within a unified framework remains relatively underexplored. In this work, we propose a multi-view learning framework that integrates contrastive principles into a generative pipeline to capture sound source and device information. Our method encodes compressed audio latents into view-specific and view-common subspaces, guided by two self-supervised objectives: contrastive learning for targeted information flow between subspaces, and reconstruction for overall information preservation. We evaluate our method on an urban sound sensor network dataset for sound source and sensor classification, demonstrating improved downstream performance over traditional SSL techniques. Additionally, we investigate the model's potential to disentangle environmental sound attributes within the structured latent space under varied training configurations.
>
---
#### [replaced 016] TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling
- **分类: cs.IR; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.01698v2](http://arxiv.org/pdf/2510.01698v2)**

> **作者:** Seungheon Doh; Keunwoo Choi; Juhan Nam
>
> **备注:** Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
>
> **摘要:** While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
>
---
#### [replaced 017] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18671v4](http://arxiv.org/pdf/2506.18671v4)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to encode temporal and identity information. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
