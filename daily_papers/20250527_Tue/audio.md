# 音频 cs.SD;  eess.SP

- **最新发布 53 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Self-supervised learning method using multiple sampling strategies for general-purpose audio representation
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出基于多采样策略（含帧级、任务特定策略）的自监督学习方法，用于通用音频表征。解决传统单clip-level采样在帧级分类和音高检测等任务上的不足，通过多视角对比损失学习，预训练于Audioset子集，在下游任务中提升分类、事件检测和音高检测性能25%、20%和3.6%。**

- **链接: [http://arxiv.org/pdf/2505.18984v1](http://arxiv.org/pdf/2505.18984v1)**

> **作者:** Ibuki Kuroyanagi; Tatsuya Komatsu
>
> **备注:** 5 pages, 1 figure, 2 tables, ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
>
> **摘要:** We propose a self-supervised learning method using multiple sampling strategies to obtain general-purpose audio representation. Multiple sampling strategies are used in the proposed method to construct contrastive losses from different perspectives and learn representations based on them. In this study, in addition to the widely used clip-level sampling strategy, we introduce two new strategies, a frame-level strategy and a task-specific strategy. The proposed multiple strategies improve the performance of frame-level classification and other tasks like pitch detection, which are not the focus of the conventional single clip-level sampling strategy. We pre-trained the method on a subset of Audioset and applied it to a downstream task with frozen weights. The proposed method improved clip classification, sound event detection, and pitch detection performance by 25%, 20%, and 3.6%.
>
---
#### [new 002] Audio Geolocation: A Natural Sounds Benchmark
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文研究音频地理定位任务，解决如何仅通过声音确定地理位置的问题。通过将音频转为频谱图测试图像定位技术，结合物种分布预测与检索方法，评估物种丰富度及时空聚合的影响，并探索音视频多模态定位，验证声学与视觉信息融合的优势。**

- **链接: [http://arxiv.org/pdf/2505.18726v1](http://arxiv.org/pdf/2505.18726v1)**

> **作者:** Mustafa Chasmai; Wuao Liu; Subhransu Maji; Grant Van Horn
>
> **摘要:** Can we determine someone's geographic location purely from the sounds they hear? Are acoustic signals enough to localize within a country, state, or even city? We tackle the challenge of global-scale audio geolocation, formalize the problem, and conduct an in-depth analysis with wildlife audio from the iNatSounds dataset. Adopting a vision-inspired approach, we convert audio recordings to spectrograms and benchmark existing image geolocation techniques. We hypothesize that species vocalizations offer strong geolocation cues due to their defined geographic ranges and propose an approach that integrates species range prediction with retrieval-based geolocation. We further evaluate whether geolocation improves when analyzing species-rich recordings or when aggregating across spatiotemporal neighborhoods. Finally, we introduce case studies from movies to explore multimodal geolocation using both audio and visual content. Our work highlights the advantages of integrating audio and visual cues, and sets the stage for future research in audio geolocation.
>
---
#### [new 003] EmoSphere-SER: Enhancing Speech Emotion Recognition Through Spherical Representation with Auxiliary Classification
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属语音情感识别任务，旨在提升情感维度（VAD）预测精度。提出EmoSphere-SER模型，将VAD值转为球面坐标并划分区域，通过辅助分类任务引导回归，结合动态加权与多头自注意力风格池化层，增强特征捕捉与预测一致性，实验验证优于基线方法。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19693v1](http://arxiv.org/pdf/2505.19693v1)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Speech emotion recognition predicts a speaker's emotional state from speech signals using discrete labels or continuous dimensions such as arousal, valence, and dominance (VAD). We propose EmoSphere-SER, a joint model that integrates spherical VAD region classification to guide VAD regression for improved emotion prediction. In our framework, VAD values are transformed into spherical coordinates that are divided into multiple spherical regions, and an auxiliary classification task predicts which spherical region each point belongs to, guiding the regression process. Additionally, we incorporate a dynamic weighting scheme and a style pooling layer with multi-head self-attention to capture spectral and temporal dynamics, further boosting performance. This combined training strategy reinforces structured learning and improves prediction consistency. Experimental results show that our approach exceeds baseline methods, confirming the validity of the proposed framework.
>
---
#### [new 004] Towards Reliable Large Audio Language Model
- **分类: cs.SD; cs.CL; cs.HC; cs.MM; eess.AS**

- **简介: 该论文研究提升大型音频语言模型（LALM）可靠性的方法，解决其无法主动识别知识边界和拒绝未知问题的缺陷。工作包括系统评估无训练的多模态思维链和有监督微调等方法，提出可靠性增益指数（RGI）新评估指标，并发现可靠性意识可跨音频模态迁移。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19294v1](http://arxiv.org/pdf/2505.19294v1)**

> **作者:** Ziyang Ma; Xiquan Li; Yakun Song; Wenxi Chen; Chenpeng Du; Jian Wu; Yuanzhe Chen; Zhuo Chen; Yuping Wang; Yuxuan Wang; Xie Chen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Recent advancements in large audio language models (LALMs) have demonstrated impressive results and promising prospects in universal understanding and reasoning across speech, music, and general sound. However, these models still lack the ability to recognize their knowledge boundaries and refuse to answer questions they don't know proactively. While there have been successful attempts to enhance the reliability of LLMs, reliable LALMs remain largely unexplored. In this paper, we systematically investigate various approaches towards reliable LALMs, including training-free methods such as multi-modal chain-of-thought (MCoT), and training-based methods such as supervised fine-tuning (SFT). Besides, we identify the limitations of previous evaluation metrics and propose a new metric, the Reliability Gain Index (RGI), to assess the effectiveness of different reliable methods. Our findings suggest that both training-free and training-based methods enhance the reliability of LALMs to different extents. Moreover, we find that awareness of reliability is a "meta ability", which can be transferred across different audio modalities, although significant structural and content differences exist among sound, music, and speech.
>
---
#### [new 005] Large Language Model-Driven Distributed Integrated Multimodal Sensing and Semantic Communications
- **分类: eess.SP; cs.AI; cs.CV**

- **简介: 该论文提出LLM-DiSAC框架，解决传统单模态传感系统在复杂环境中的局限性。通过多设备协作，融合RF与视觉数据，设计RF-视觉融合网络、LLM语义传输网络及自适应聚合模型，并采用分布式学习保护隐私。实验验证其提升感知与通信性能。**

- **链接: [http://arxiv.org/pdf/2505.18194v1](http://arxiv.org/pdf/2505.18194v1)**

> **作者:** Yubo Peng; Luping Xiang; Bingxin Zhang; Kun Yang
>
> **摘要:** Traditional single-modal sensing systems-based solely on either radio frequency (RF) or visual data-struggle to cope with the demands of complex and dynamic environments. Furthermore, single-device systems are constrained by limited perspectives and insufficient spatial coverage, which impairs their effectiveness in urban or non-line-of-sight scenarios. To overcome these challenges, we propose a novel large language model (LLM)-driven distributed integrated multimodal sensing and semantic communication (LLM-DiSAC) framework. Specifically, our system consists of multiple collaborative sensing devices equipped with RF and camera modules, working together with an aggregation center to enhance sensing accuracy. First, on sensing devices, LLM-DiSAC develops an RF-vision fusion network (RVFN), which employs specialized feature extractors for RF and visual data, followed by a cross-attention module for effective multimodal integration. Second, a LLM-based semantic transmission network (LSTN) is proposed to enhance communication efficiency, where the LLM-based decoder leverages known channel parameters, such as transceiver distance and signal-to-noise ratio (SNR), to mitigate semantic distortion. Third, at the aggregation center, a transformer-based aggregation model (TRAM) with an adaptive aggregation attention mechanism is developed to fuse distributed features and enhance sensing accuracy. To preserve data privacy, a two-stage distributed learning strategy is introduced, allowing local model training at the device level and centralized aggregation model training using intermediate features. Finally, evaluations on a synthetic multi-view RF-visual dataset generated by the Genesis simulation engine show that LLM-DiSAC achieves a good performance.
>
---
#### [new 006] CloneShield: A Framework for Universal Perturbation Against Zero-Shot Voice Cloning
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出CloneShield框架，针对零样本语音克隆的隐私威胁，通过生成时域对抗扰动防御非法克隆。任务为保护语音隐私，解决仅需少量音频即可高保真复制声音的问题。方法包括多目标优化（MGDA算法）和Mel频谱分解扰动，平衡听感保真（PESQ 3.90）与克隆失效（PESQ降至1.07），实验证明有效。**

- **链接: [http://arxiv.org/pdf/2505.19119v1](http://arxiv.org/pdf/2505.19119v1)**

> **作者:** Renyuan Li; Zhibo Liang; Haichuan Zhang; Tianyu Shi; Zhiyuan Cheng; Jia Shi; Carl Yang; Mingjie Tang
>
> **备注:** 10pages, 4figures
>
> **摘要:** Recent breakthroughs in text-to-speech (TTS) voice cloning have raised serious privacy concerns, allowing highly accurate vocal identity replication from just a few seconds of reference audio, while retaining the speaker's vocal authenticity. In this paper, we introduce CloneShield, a universal time-domain adversarial perturbation framework specifically designed to defend against zero-shot voice cloning. Our method provides protection that is robust across speakers and utterances, without requiring any prior knowledge of the synthesized text. We formulate perturbation generation as a multi-objective optimization problem, and propose Multi-Gradient Descent Algorithm (MGDA) to ensure the robust protection across diverse utterances. To preserve natural auditory perception for users, we decompose the adversarial perturbation via Mel-spectrogram representations and fine-tune it for each sample. This design ensures imperceptibility while maintaining strong degradation effects on zero-shot cloned outputs. Experiments on three state-of-the-art zero-shot TTS systems, five benchmark datasets and evaluations from 60 human listeners demonstrate that our method preserves near-original audio quality in protected inputs (PESQ = 3.90, SRS = 0.93) while substantially degrading both speaker similarity and speech quality in cloned samples (PESQ = 1.07, SRS = 0.08).
>
---
#### [new 007] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于自监督语音表征学习任务，解决如何高效去除说话人信息而不损害语音内容的问题。提出通过线性分解方法将SSL表示分离为说话人特定和无关成分，生成解纠缠表征，在语音转换等任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19273v1](http://arxiv.org/pdf/2505.19273v1)**

> **作者:** Giuseppe Ruggiero; Matteo Testa; Jurgen Van de Walle; Luigi Di Caro
>
> **备注:** Full paper accepted at ACL 2025
>
> **摘要:** Self-supervised learning (SSL) has reduced the reliance on expensive labeling in speech technologies by learning meaningful representations from unannotated data. Since most SSL-based downstream tasks prioritize content information in speech, ideal representations should disentangle content from unwanted variations like speaker characteristics in the SSL representations. However, removing speaker information often degrades other speech components, and existing methods either fail to fully disentangle speaker identity or require resource-intensive models. In this paper, we propose a novel disentanglement method that linearly decomposes SSL representations into speaker-specific and speaker-independent components, effectively generating speaker disentangled representations. Comprehensive experiments show that our approach achieves speaker independence and as such, when applied to content-driven tasks such as voice conversion, our representations yield significant improvements over state-of-the-art methods.
>
---
#### [new 008] Automated evaluation of children's speech fluency for low-resource languages
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出结合多语言ASR模型、客观指标（如语音错误率、停顿比例）及GPT分类器，解决低资源语言（泰米尔语、马来语）儿童语音流利度自动评估问题。通过对比Random Forest、XGBoost及ChatGPT-4o，实验显示其方法准确率显著更高。**

- **链接: [http://arxiv.org/pdf/2505.19671v1](http://arxiv.org/pdf/2505.19671v1)**

> **作者:** Bowen Zhang; Nur Afiqah Abdul Latiff; Justin Kan; Rong Tong; Donny Soh; Xiaoxiao Miao; Ian McLoughlin
>
> **备注:** 5 pages, 2 figures, conference
>
> **摘要:** Assessment of children's speaking fluency in education is well researched for majority languages, but remains highly challenging for low resource languages. This paper proposes a system to automatically assess fluency by combining a fine-tuned multilingual ASR model, an objective metrics extraction stage, and a generative pre-trained transformer (GPT) network. The objective metrics include phonetic and word error rates, speech rate, and speech-pause duration ratio. These are interpreted by a GPT-based classifier guided by a small set of human-evaluated ground truth examples, to score fluency. We evaluate the proposed system on a dataset of children's speech in two low-resource languages, Tamil and Malay and compare the classification performance against Random Forest and XGBoost, as well as using ChatGPT-4o to predict fluency directly from speech input. Results demonstrate that the proposed approach achieves significantly higher accuracy than multimodal GPT or other methods.
>
---
#### [new 009] EnvSDD: Benchmarking Environmental Sound Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文聚焦环境声音深度伪造检测任务，针对现有方法对环境音效适应性差及数据集规模小的问题，构建了首个大规模数据集EnvSDD（含45.25小时真实与316.74小时伪造音频），并提出基于预训练音频模型的检测系统，实验显示其优于语音/歌声领域的现有最优方法。**

- **链接: [http://arxiv.org/pdf/2505.19203v1](http://arxiv.org/pdf/2505.19203v1)**

> **作者:** Han Yin; Yang Xiao; Rohan Kumar Das; Jisheng Bai; Haohe Liu; Wenwu Wang; Mark D Plumbley
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Audio generation systems now create very realistic soundscapes that can enhance media production, but also pose potential risks. Several studies have examined deepfakes in speech or singing voice. However, environmental sounds have different characteristics, which may make methods for detecting speech and singing deepfakes less effective for real-world sounds. In addition, existing datasets for environmental sound deepfake detection are limited in scale and audio types. To address this gap, we introduce EnvSDD, the first large-scale curated dataset designed for this task, consisting of 45.25 hours of real and 316.74 hours of fake audio. The test set includes diverse conditions to evaluate the generalizability, such as unseen generation models and unseen datasets. We also propose an audio deepfake detection system, based on a pre-trained audio foundation model. Results on EnvSDD show that our proposed system outperforms the state-of-the-art systems from speech and singing domains.
>
---
#### [new 010] Novel Loss-Enhanced Universal Adversarial Patches for Sustainable Speaker Privacy
- **分类: cs.SD; cs.AI; cs.CR; eess.AS**

- **简介: 该论文属于说话人隐私保护任务，旨在解决现有通用对抗补丁（UAP）方法在语音处理中音频质量下降、跨模型迁移性差及依赖音频长度的问题。提出新型指数总方差损失函数与可扩展UAP插入方法，提升对抗补丁的攻击强度与自然度，实现对多种音频长度的高效隐私保护。**

- **链接: [http://arxiv.org/pdf/2505.19951v1](http://arxiv.org/pdf/2505.19951v1)**

> **作者:** Elvir Karimov; Alexander Varlamov; Danil Ivanov; Dmitrii Korzh; Oleg Y. Rogov
>
> **备注:** 5 pages, 3 figures, 1 table; Submitted to Interspeech 2025
>
> **摘要:** Deep learning voice models are commonly used nowadays, but the safety processing of personal data, such as human identity and speech content, remains suspicious. To prevent malicious user identification, speaker anonymization methods were proposed. Current methods, particularly based on universal adversarial patch (UAP) applications, have drawbacks such as significant degradation of audio quality, decreased speech recognition quality, low transferability across different voice biometrics models, and performance dependence on the input audio length. To mitigate these drawbacks, in this work, we introduce and leverage the novel Exponential Total Variance (TV) loss function and provide experimental evidence that it positively affects UAP strength and imperceptibility. Moreover, we present a novel scalable UAP insertion procedure and demonstrate its uniformly high performance for various audio lengths.
>
---
#### [new 011] A Comprehensive Real-World Assessment of Audio Watermarking Algorithms: Will They Survive Neural Codecs?
- **分类: cs.SD; cs.AI; cs.CR; cs.LG; eess.AS**

- **简介: 该论文属于音频水印算法鲁棒性评估任务，针对现有方法在现实攻击（如神经压缩、极性反转）下失效的问题，构建标准化评估框架，设计含压缩、噪声、混响等攻击的测试管道及多类型音频数据集，评估四算法并揭示其脆弱性。**

- **链接: [http://arxiv.org/pdf/2505.19663v1](http://arxiv.org/pdf/2505.19663v1)**

> **作者:** Yigitcan Özer; Woosung Choi; Joan Serrà; Mayank Kumar Singh; Wei-Hsiang Liao; Yuki Mitsufuji
>
> **备注:** 5 pages; 5 tables; accepted at INTERSPEECH 2025
>
> **摘要:** We present a framework to foster the evaluation of deep learning-based audio watermarking algorithms, establishing a standardized benchmark and allowing systematic comparisons. To simulate real-world usage, we introduce a comprehensive audio attack pipeline, featuring various distortions such as compression, background noise, and reverberation, and propose a diverse test dataset, including speech, environmental sounds, and music recordings. By assessing the performance of four existing watermarking algorithms on our framework, two main insights stand out: (i) neural compression techniques pose the most significant challenge, even when algorithms are trained with such compressions; and (ii) training with audio attacks generally improves robustness, although it is insufficient in some cases. Furthermore, we find that specific distortions, such as polarity inversion, time stretching, or reverb, seriously affect certain algorithms. Our contributions strengthen the robustness and perceptual assessment of audio watermarking algorithms across a wide range of applications, while ensuring a fair and consistent evaluation approach. The evaluation framework, including the attack pipeline, is accessible at github.com/SonyResearch/wm_robustness_eval.
>
---
#### [new 012] ABHINAYA -- A System for Speech Emotion Recognition In Naturalistic Conditions Challenge
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 论文提出ABHINAYA系统，针对自然环境下语音情感识别的变异性、录音条件复杂及类别不平衡问题，整合语音表征模型、文本语境模型及跨模态联合建模，采用自监督语音大模型与大语言模型，结合定制损失函数与投票策略优化，获Interspeech挑战赛第4名并达领域SOTA。**

- **链接: [http://arxiv.org/pdf/2505.18217v1](http://arxiv.org/pdf/2505.18217v1)**

> **作者:** Soumya Dutta; Smruthi Balaji; Varada R; Viveka Salinamakki; Sriram Ganapathy
>
> **备注:** 5 pages, 2 figures, 4 tables, accepted at Interspeech 2025
>
> **摘要:** Speech emotion recognition (SER) in naturalistic settings remains a challenge due to the intrinsic variability, diverse recording conditions, and class imbalance. As participants in the Interspeech Naturalistic SER Challenge which focused on these complexities, we present Abhinaya, a system integrating speech-based, text-based, and speech-text models. Our approach fine-tunes self-supervised and speech large language models (SLLM) for speech representations, leverages large language models (LLM) for textual context, and employs speech-text modeling with an SLLM to capture nuanced emotional cues. To combat class imbalance, we apply tailored loss functions and generate categorical decisions through majority voting. Despite one model not being fully trained, the Abhinaya system ranked 4th among 166 submissions. Upon completion of training, it achieved state-of-the-art performance among published results, demonstrating the effectiveness of our approach for SER in real-world conditions.
>
---
#### [new 013] MPE-TTS: Customized Emotion Zero-Shot Text-To-Speech Using Multi-Modal Prompt
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出MPE-TTS系统，属于零样本文本到语音任务，解决现有方法依赖单一提示导致灵活性不足的问题。通过多模态情感编码器提取文本/图像/语音情感提示，结合解耦的语音成分（内容、音色、情感、韵律），引入韵律预测器和情感一致性损失，提升生成语音的自然度与情感匹配度。**

- **链接: [http://arxiv.org/pdf/2505.18453v1](http://arxiv.org/pdf/2505.18453v1)**

> **作者:** Zhichao Wu; Yueteng Kang; Songjun Cao; Long Ma; Qiulin Li; Qun Yang
>
> **备注:** Accepted by InterSpeech
>
> **摘要:** Most existing Zero-Shot Text-To-Speech(ZS-TTS) systems generate the unseen speech based on single prompt, such as reference speech or text descriptions, which limits their flexibility. We propose a customized emotion ZS-TTS system based on multi-modal prompt. The system disentangles speech into the content, timbre, emotion and prosody, allowing emotion prompts to be provided as text, image or speech. To extract emotion information from different prompts, we propose a multi-modal prompt emotion encoder. Additionally, we introduce an prosody predictor to fit the distribution of prosody and propose an emotion consistency loss to preserve emotion information in the predicted prosody. A diffusion-based acoustic model is employed to generate the target mel-spectrogram. Both objective and subjective experiments demonstrate that our system outperforms existing systems in terms of naturalness and similarity. The samples are available at https://mpetts-demo.github.io/mpetts_demo/.
>
---
#### [new 014] Decoding Speaker-Normalized Pitch from EEG for Mandarin Perception
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于脑电信号解码任务，旨在从EEG中提取与说话人无关的音高信息。通过让受试者聆听不同发音人 Mandarin 单字，提出CE-ViViT模型直接解码音高轮廓，发现归一化音高解码更准确，验证大脑编码相对音高而非绝对音高。**

- **链接: [http://arxiv.org/pdf/2505.19626v1](http://arxiv.org/pdf/2505.19626v1)**

> **作者:** Jiaxin Chen; Yiming Wang; Ziyu Zhang; Jiayang Han; Yin-Long Liu; Rui Feng; Xiuyuan Liang; Zhen-Hua Ling; Jiahong Yuan
>
> **摘要:** The same speech content produced by different speakers exhibits significant differences in pitch contour, yet listeners' semantic perception remains unaffected. This phenomenon may stem from the brain's perception of pitch contours being independent of individual speakers' pitch ranges. In this work, we recorded electroencephalogram (EEG) while participants listened to Mandarin monosyllables with varying tones, phonemes, and speakers. The CE-ViViT model is proposed to decode raw or speaker-normalized pitch contours directly from EEG. Experimental results demonstrate that the proposed model can decode pitch contours with modest errors, achieving performance comparable to state-of-the-art EEG regression methods. Moreover, speaker-normalized pitch contours were decoded more accurately, supporting the neural encoding of relative pitch.
>
---
#### [new 015] Automated data curation for self-supervised learning in underwater acoustic analysis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出自动化数据整理管道，整合AIS与水听器数据，通过层次k-means聚类生成平衡数据集，解决水下声学数据量大、未标记问题，支持自监督学习，用于监测海洋哺乳动物及评估声污染。**

- **链接: [http://arxiv.org/pdf/2505.20066v1](http://arxiv.org/pdf/2505.20066v1)**

> **作者:** Hilde I Hummel; Sandjai Bhulai; Burooj Ghani; Rob van der Mei
>
> **摘要:** The sustainability of the ocean ecosystem is threatened by increased levels of sound pollution, making monitoring crucial to understand its variability and impact. Passive acoustic monitoring (PAM) systems collect a large amount of underwater sound recordings, but the large volume of data makes manual analysis impossible, creating the need for automation. Although machine learning offers a potential solution, most underwater acoustic recordings are unlabeled. Self-supervised learning models have demonstrated success in learning from large-scale unlabeled data in various domains like computer vision, Natural Language Processing, and audio. However, these models require large, diverse, and balanced datasets for training in order to generalize well. To address this, a fully automated self-supervised data curation pipeline is proposed to create a diverse and balanced dataset from raw PAM data. It integrates Automatic Identification System (AIS) data with recordings from various hydrophones in the U.S. waters. Using hierarchical k-means clustering, the raw audio data is sampled and then combined with AIS samples to create a balanced and diverse dataset. The resulting curated dataset enables the development of self-supervised learning models, facilitating various tasks such as monitoring marine mammals and assessing sound pollution.
>
---
#### [new 016] AI- Enhanced Stethoscope in Remote Diagnostics for Cardiopulmonary Diseases
- **分类: eess.SP; cs.CV**

- **简介: 论文提出基于AI的低成本听诊器模型，结合MFCC特征提取与GRU-CNN混合网络，分析心肺听诊音频，实时诊断六类肺病及五类心脏病，解决偏远地区医疗资源不足导致的诊断延迟问题。**

- **链接: [http://arxiv.org/pdf/2505.18184v1](http://arxiv.org/pdf/2505.18184v1)**

> **作者:** Hania Ghouse; Juveria Tanveen; Abdul Muqtadir Ahmed; Uma N. Dulhare
>
> **摘要:** The increase in cardiac and pulmonary diseases presents an alarming and pervasive health challenge on a global scale responsible for unexpected and premature mortalities. In spite of how serious these conditions are, existing methods of detection and treatment encounter challenges, particularly in achieving timely diagnosis for effective medical intervention. Manual screening processes commonly used for primary detection of cardiac and respiratory problems face inherent limitations, increased by a scarcity of skilled medical practitioners in remote or under-resourced areas. To address this, our study introduces an innovative yet efficient model which integrates AI for diagnosing lung and heart conditions concurrently using the auscultation sounds. Unlike the already high-priced digital stethoscope, our proposed model has been particularly designed to deploy on low-cost embedded devices and thus ensure applicability in under-developed regions that actually face an issue of accessing medical care. Our proposed model incorporates MFCC feature extraction and engineering techniques to ensure that the signal is well analyzed for accurate diagnostics through the hybrid model combining Gated Recurrent Unit with CNN in processing audio signals recorded from the low-cost stethoscope. Beyond its diagnostic capabilities, the model generates digital audio records that facilitate in classifying six pulmonary and five cardiovascular diseases. Hence, the integration of a cost effective stethoscope with an efficient AI empowered model deployed on a web app providing real-time analysis, represents a transformative step towards standardized healthcare
>
---
#### [new 017] Token-Level Logits Matter: A Closer Look at Speech Foundation Models for Ambiguous Emotion Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文聚焦模糊情感识别任务，探究语音基础模型（SFMs）在处理模糊情绪时的潜力。针对现有模型忽视情绪复杂性的问题，提出基于提示的预测方法及两种分析策略（生成文本分析和token-level logits解析），揭示SFMs虽文本生成不够准确，但能通过内部logits利用先验知识解析模糊情绪，展现鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.18484v1](http://arxiv.org/pdf/2505.18484v1)**

> **作者:** Jule Valendo Halim; Siyi Wang; Hong Jia; Ting Dang
>
> **备注:** Accepted at INTERSPEECH 2025
>
> **摘要:** Emotional intelligence in conversational AI is crucial across domains like human-computer interaction. While numerous models have been developed, they often overlook the complexity and ambiguity inherent in human emotions. In the era of large speech foundation models (SFMs), understanding their capability in recognizing ambiguous emotions is essential for the development of next-generation emotion-aware models. This study examines the effectiveness of SFMs in ambiguous emotion recognition. We designed prompts for ambiguous emotion prediction and introduced two novel approaches to infer ambiguous emotion distributions: one analysing generated text responses and the other examining the internal processing of SFMs through token-level logits. Our findings suggest that while SFMs may not consistently generate accurate text responses for ambiguous emotions, they can interpret such emotions at the token level based on prior knowledge, demonstrating robustness across different prompts.
>
---
#### [new 018] STOPA: A Database of Systematic VariaTion Of DeePfake Audio for Open-Set Source Tracing and Attribution
- **分类: cs.SD; cs.AI; cs.CR; eess.AS; 68T45, 68T10, 94A08; I.2.7; I.5.4; K.4.1**

- **简介: 该论文属于深度伪造语音的来源追踪与归因任务，旨在解决现有数据集缺乏系统化变异和元数据导致的源追踪精度不足问题。研究构建了STOPA数据库，涵盖8种声学模型、6种声码器及多样化参数设置（70万样本），通过系统控制生成因素提升归因可靠性，助力伪造检测与模型透明化。**

- **链接: [http://arxiv.org/pdf/2505.19644v1](http://arxiv.org/pdf/2505.19644v1)**

> **作者:** Anton Firc; Manasi Chibber; Jagabandhu Mishra; Vishwanath Pratap Singh; Tomi Kinnunen; Kamil Malinka
>
> **备注:** Accepted to Interspeech 2025 conference
>
> **摘要:** A key research area in deepfake speech detection is source tracing - determining the origin of synthesised utterances. The approaches may involve identifying the acoustic model (AM), vocoder model (VM), or other generation-specific parameters. However, progress is limited by the lack of a dedicated, systematically curated dataset. To address this, we introduce STOPA, a systematically varied and metadata-rich dataset for deepfake speech source tracing, covering 8 AMs, 6 VMs, and diverse parameter settings across 700k samples from 13 distinct synthesisers. Unlike existing datasets, which often feature limited variation or sparse metadata, STOPA provides a systematically controlled framework covering a broader range of generative factors, such as the choice of the vocoder model, acoustic model, or pretrained weights, ensuring higher attribution reliability. This control improves attribution accuracy, aiding forensic analysis, deepfake detection, and generative model transparency.
>
---
#### [new 019] Evaluation in EEG Emotion Recognition: State-of-the-Art Review and Unified Framework
- **分类: eess.SP; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于EEG情绪识别评估方法研究，旨在解决领域内缺乏统一评估协议的问题。通过分析216篇论文，揭示了现有评估中的不一致性（如ground truth定义、指标选择、数据划分方式等），并提出开源框架EEGain，标准化预处理、数据划分、评估指标及数据集加载，支持六大数据集和四种常用方法验证，推动领域可重复性与可比性。**

- **链接: [http://arxiv.org/pdf/2505.18175v1](http://arxiv.org/pdf/2505.18175v1)**

> **作者:** Natia Kukhilava; Tatia Tsmindashvili; Rapael Kalandadze; Anchit Gupta; Sofio Katamadze; François Brémond; Laura M. Ferrari; Philipp Müller; Benedikt Emanuel Wirth
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Electroencephalography-based Emotion Recognition (EEG-ER) has become a growing research area in recent years. Analyzing 216 papers published between 2018 and 2023, we uncover that the field lacks a unified evaluation protocol, which is essential to fairly define the state of the art, compare new approaches and to track the field's progress. We report the main inconsistencies between the used evaluation protocols, which are related to ground truth definition, evaluation metric selection, data splitting types (e.g., subject-dependent or subject-independent) and the use of different datasets. Capitalizing on this state-of-the-art research, we propose a unified evaluation protocol, EEGain (https://github.com/EmotionLab/EEGain), which enables an easy and efficient evaluation of new methods and datasets. EEGain is a novel open source software framework, offering the capability to compare - and thus define - state-of-the-art results. EEGain includes standardized methods for data pre-processing, data splitting, evaluation metrics, and the ability to load the six most relevant datasets (i.e., AMIGOS, DEAP, DREAMER, MAHNOB-HCI, SEED, SEED-IV) in EEG-ER with only a single line of code. In addition, we have assessed and validated EEGain using these six datasets on the four most common publicly available methods (EEGNet, DeepConvNet, ShallowConvNet, TSception). This is a significant step to make research on EEG-ER more reproducible and comparable, thereby accelerating the overall progress of the field.
>
---
#### [new 020] RA-CLAP: Relation-Augmented Emotional Speaking Style Contrastive Language-Audio Pretraining For Speech Retrieval
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音检索任务，针对情感说话风格描述（ESSD）中跨模态对比预训练不足的问题，提出新任务情感说话风格检索（ESSR）及ESS-CLAP模型，进一步通过RA-CLAP采用自蒸馏学习局部匹配关系，克服传统二元关系限制，提升泛化能力，实验验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.19437v1](http://arxiv.org/pdf/2505.19437v1)**

> **作者:** Haoqin Sun; Jingguang Tian; Jiaming Zhou; Hui Wang; Jiabei He; Shiwan Zhao; Xiangyu Kong; Desheng Hu; Xinkang Xu; Xinhui Hu; Yong Qin
>
> **摘要:** The Contrastive Language-Audio Pretraining (CLAP) model has demonstrated excellent performance in general audio description-related tasks, such as audio retrieval. However, in the emerging field of emotional speaking style description (ESSD), cross-modal contrastive pretraining remains largely unexplored. In this paper, we propose a novel speech retrieval task called emotional speaking style retrieval (ESSR), and ESS-CLAP, an emotional speaking style CLAP model tailored for learning relationship between speech and natural language descriptions. In addition, we further propose relation-augmented CLAP (RA-CLAP) to address the limitation of traditional methods that assume a strict binary relationship between caption and audio. The model leverages self-distillation to learn the potential local matching relationships between speech and descriptions, thereby enhancing generalization ability. The experimental results validate the effectiveness of RA-CLAP, providing valuable reference in ESSD.
>
---
#### [new 021] Learning Emotion-Invariant Speaker Representations for Speaker Verification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于说话人验证任务，旨在解决说话人表征受情绪变化影响的问题。提出三方面改进：1）用CopyPaste数据增强获取同一说话人不同情绪的平行数据；2）采用余弦相似度损失减少类内差异，降低情绪相关性；3）通过基于语音能量的情绪感知掩码强化表征情绪不变性。实验显示EER下降19.29%。**

- **链接: [http://arxiv.org/pdf/2505.18498v1](http://arxiv.org/pdf/2505.18498v1)**

> **作者:** Jingguang Tian; Xinhui Hu; Xinkang Xu
>
> **摘要:** In recent years, the rapid progress in speaker verification (SV) technology has been driven by the extraction of speaker representations based on deep learning. However, such representations are still vulnerable to emotion variability. To address this issue, we propose multiple improvements to train speaker encoders to increase emotion robustness. Firstly, we utilize CopyPaste-based data augmentation to gather additional parallel data, which includes different emotional expressions from the same speaker. Secondly, we apply cosine similarity loss to restrict parallel sample pairs and minimize intra-class variation of speaker representations to reduce their correlation with emotional information. Finally, we use emotion-aware masking (EM) based on the speech signal energy on the input parallel samples to further strengthen the speaker representation and make it emotion-invariant. We conduct a comprehensive ablation study to demonstrate the effectiveness of these various components. Experimental results show that our proposed method achieves a relative 19.29\% drop in EER compared to the baseline system.
>
---
#### [new 022] Room Impulse Response as a Prompt for Acoustic Echo Cancellation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声学回声消除（AEC）任务，旨在解决数据驱动AEC模型在未知真实环境（如不可见的房间脉冲响应RIR）中泛化能力差的问题。提出将RIR作为训练提示，并探索四种融合方法，通过模拟与真实场景验证了该方法显著提升了模型在未知环境中的性能。**

- **链接: [http://arxiv.org/pdf/2505.19480v1](http://arxiv.org/pdf/2505.19480v1)**

> **作者:** Fei Zhao; Shulin He; Xueliang Zhang
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Data-driven acoustic echo cancellation (AEC) methods, predominantly trained on synthetic or constrained real-world datasets, encounter performance declines in unseen echo scenarios, especially in real environments where echo paths are not directly observable. Our proposed method counters this limitation by integrating room impulse response (RIR) as a pivotal training prompt, aiming to improve the generalization of AEC models in such unforeseen conditions. We also explore four RIR prompt fusion methods. Comprehensive evaluations, including both simulated RIR under unknown conditions and recorded RIR in real, demonstrate that the proposed approach significantly improves performance compared to baseline models. These results substantiate the effectiveness of our RIR-guided approach in strengthening the model's generalization capabilities.
>
---
#### [new 023] Serial-OE: Anomalous sound detection based on serial method with outlier exposure capable of using small amounts of anomalous data for training
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Serial-OE方法，针对异常声音检测中异常数据稀缺的问题，结合离群值暴露框架，利用正常数据、伪异常数据及少量真实异常数据训练模型，提升检测性能。实验显示其优于现有方法，并分析数据影响，支持动态适应实际场景。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18982v1](http://arxiv.org/pdf/2505.18982v1)**

> **作者:** Ibuki Kuroyanagi; Tomoki Hayashi; Kazuya Takeda; Tomoki Toda
>
> **备注:** 39 pages, 5 figures, 5 tables, APSIPA Transactions on Signal and Information Processing
>
> **摘要:** We introduce Serial-OE, a new approach to anomalous sound detection (ASD) that leverages small amounts of anomalous data to improve the performance. Conventional ASD methods rely primarily on the modeling of normal data, due to the cost of collecting anomalous data from various possible types of equipment breakdowns. Our method improves upon existing ASD systems by implementing an outlier exposure framework that utilizes normal and pseudo-anomalous data for training, with the capability to also use small amounts of real anomalous data. A comprehensive evaluation using the DCASE2020 Task2 dataset shows that our method outperforms state-of-the-art ASD models. We also investigate the impact on performance of using a small amount of anomalous data during training, of using data without machine ID information, and of using contaminated training data. Our experimental results reveal the potential of using a very limited amount of anomalous data during training to address the limitations of existing methods using only normal data for training due to the scarcity of anomalous data. This study contributes to the field by presenting a method that can be dynamically adapted to include anomalous data during the operational phase of an ASD system, paving the way for more accurate ASD.
>
---
#### [new 024] Improving Anomalous Sound Detection through Pseudo-anomalous Set Selection and Pseudo-label Utilization under Unlabeled Conditions
- **分类: cs.SD; eess.AS**

- **简介: 该论文针对无标签条件下异常声音检测（ASD）因缺乏相似机器数据和标签导致性能下降的问题，提出结合伪异常集选择、三元组学习生成伪标签及迭代优化的集成方法，提升检测准确率，在DCASE数据集上AUC提升6.6分，减少标注需求。**

- **链接: [http://arxiv.org/pdf/2505.18980v1](http://arxiv.org/pdf/2505.18980v1)**

> **作者:** Ibuki Kuroyanagi; Takuya Fujimura; Kazuya Takeda; Tomoki Toda
>
> **备注:** 33 pages, 3 figures, 7 tables, APSIPA Transactions on Signal and Information Processing
>
> **摘要:** This paper addresses performance degradation in anomalous sound detection (ASD) when neither sufficiently similar machine data nor operational state labels are available. We present an integrated pipeline that combines three complementary components derived from prior work and extends them to the unlabeled ASD setting. First, we adapt an anomaly score based selector to curate external audio data resembling the normal sounds of the target machine. Second, we utilize triplet learning to assign pseudo-labels to unlabeled data, enabling finer classification of operational sounds and detection of subtle anomalies. Third, we employ iterative training to refine both the pseudo-anomalous set selection and pseudo-label assignment, progressively improving detection accuracy. Experiments on the DCASE2022-2024 Task 2 datasets demonstrate that, in unlabeled settings, our approach achieves an average AUC increase of over 6.6 points compared to conventional methods. In labeled settings, incorporating external data from the pseudo-anomalous set further boosts performance. These results highlight the practicality and robustness of our methods in scenarios with scarce machine data and labels, facilitating ASD deployment across diverse industrial settings with minimal annotation effort.
>
---
#### [new 025] Towards Video to Piano Music Generation with Chain-of-Perform Support Benchmarks
- **分类: cs.SD; cs.CV; eess.AS**

- **简介: 该论文属于视频到钢琴音乐生成任务，解决现有数据集无法准确评估音画同步复杂性的问题。提出CoP基准数据集，含多模态标注与评估框架，并开源促进研究。**

- **链接: [http://arxiv.org/pdf/2505.20038v1](http://arxiv.org/pdf/2505.20038v1)**

> **作者:** Chang Liu; Haomin Zhang; Shiyu Xia; Zihao Chen; Chaofan Ding; Xin Yue; Huizhe Chen; Xinhan Di
>
> **备注:** 4 pages, 1 figure, accepted by CVPR 2025 MMFM Workshop
>
> **摘要:** Generating high-quality piano audio from video requires precise synchronization between visual cues and musical output, ensuring accurate semantic and temporal alignment.However, existing evaluation datasets do not fully capture the intricate synchronization required for piano music generation. A comprehensive benchmark is essential for two primary reasons: (1) existing metrics fail to reflect the complexity of video-to-piano music interactions, and (2) a dedicated benchmark dataset can provide valuable insights to accelerate progress in high-quality piano music generation. To address these challenges, we introduce the CoP Benchmark Dataset-a fully open-sourced, multimodal benchmark designed specifically for video-guided piano music generation. The proposed Chain-of-Perform (CoP) benchmark offers several compelling features: (1) detailed multimodal annotations, enabling precise semantic and temporal alignment between video content and piano audio via step-by-step Chain-of-Perform guidance; (2) a versatile evaluation framework for rigorous assessment of both general-purpose and specialized video-to-piano generation tasks; and (3) full open-sourcing of the dataset, annotations, and evaluation protocols. The dataset is publicly available at https://github.com/acappemin/Video-to-Audio-and-Piano, with a continuously updated leaderboard to promote ongoing research in this domain.
>
---
#### [new 026] Training-Free Multi-Step Audio Source Separation
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于音频源分离任务，旨在提升分离性能而无需额外训练。针对传统单步推理未充分挖掘模型潜力的问题，提出多步迭代方法：通过最优混合比例迭代融合混合信号与前一步结果，理论证明其有效性并分析与扩散模型的关联。实验显示该方法在多个任务中超越单步推理，性能媲美模型扩增或数据增强。**

- **链接: [http://arxiv.org/pdf/2505.19534v1](http://arxiv.org/pdf/2505.19534v1)**

> **作者:** Yongyi Zang; Jingyi Li; Qiuqiang Kong
>
> **摘要:** Audio source separation aims to separate a mixture into target sources. Previous audio source separation systems usually conduct one-step inference, which does not fully explore the separation ability of models. In this work, we reveal that pretrained one-step audio source separation models can be leveraged for multi-step separation without additional training. We propose a simple yet effective inference method that iteratively applies separation by optimally blending the input mixture with the previous step's separation result. At each step, we determine the optimal blending ratio by maximizing a metric. We prove that our method always yield improvement over one-step inference, provide error bounds based on model smoothness and metric robustness, and provide theoretical analysis connecting our method to denoising along linear interpolation paths between noise and clean distributions, a property we link to denoising diffusion bridge models. Our approach effectively delivers improved separation performance as a "free lunch" from existing models. Our empirical results demonstrate that our multi-step separation approach consistently outperforms one-step inference across both speech enhancement and music source separation tasks, and can achieve scaling performance similar to training a larger model, using more data, or in some cases employing a multi-step training objective. These improvements appear not only on the optimization metric during multi-step inference, but also extend to nearly all non-optimized metrics (with one exception). We also discuss limitations of our approach and directions for future research.
>
---
#### [new 027] DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文针对文本到语音中跨说话人情感迁移任务，解决情感与说话人特征分离不彻底的问题。提出DiEmo-TTS方法，通过自监督蒸馏、集群驱动采样、信息扰动及双条件Transformer，实现情感保留与说话人信息去除，提升合成质量。**

- **链接: [http://arxiv.org/pdf/2505.19687v1](http://arxiv.org/pdf/2505.19687v1)**

> **作者:** Deok-Hyeon Cho; Hyung-Seok Oh; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Cross-speaker emotion transfer in speech synthesis relies on extracting speaker-independent emotion embeddings for accurate emotion modeling without retaining speaker traits. However, existing timbre compression methods fail to fully separate speaker and emotion characteristics, causing speaker leakage and degraded synthesis quality. To address this, we propose DiEmo-TTS, a self-supervised distillation method to minimize emotional information loss and preserve speaker identity. We introduce cluster-driven sampling and information perturbation to preserve emotion while removing irrelevant factors. To facilitate this process, we propose an emotion clustering and matching approach using emotional attribute prediction and speaker embeddings, enabling generalization to unlabeled data. Additionally, we designed a dual conditioning transformer to integrate style features better. Experimental results confirm the effectiveness of our method in learning speaker-irrelevant emotion embeddings.
>
---
#### [new 028] Discovering Interpretable Concepts in Large Generative Music Models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于生成模型可解释性研究任务，旨在揭示大型音乐生成模型内部学习的隐含音乐概念。通过稀疏自编码器提取Transformer模型残差流中的可解释特征，建立自动评估流程，发现传统音乐理论已知概念及无明确语言描述的新模式，推动模型透明度与音乐理论创新。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18186v1](http://arxiv.org/pdf/2505.18186v1)**

> **作者:** Nikhil Singh; Manuel Cherep; Pattie Maes
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** The fidelity with which neural networks can now generate content such as music presents a scientific opportunity: these systems appear to have learned implicit theories of the structure of such content through statistical learning alone. This could offer a novel lens on theories of human-generated media. Where these representations align with traditional constructs (e.g. chord progressions in music), they demonstrate how these can be inferred from statistical regularities. Where they diverge, they highlight potential limits in our theoretical frameworks -- patterns that we may have overlooked but that nonetheless hold significant explanatory power. In this paper, we focus on the specific case of music generators. We introduce a method to discover musical concepts using sparse autoencoders (SAEs), extracting interpretable features from the residual stream activations of a transformer model. We evaluate this approach by extracting a large set of features and producing an automatic labeling and evaluation pipeline for them. Our results reveal both familiar musical concepts and counterintuitive patterns that lack clear counterparts in existing theories or natural language altogether. Beyond improving model transparency, our work provides a new empirical tool that might help discover organizing principles in ways that have eluded traditional methods of analysis and synthesis.
>
---
#### [new 029] Multi-Channel Acoustic Echo Cancellation Based on Direction-of-Arrival Estimation
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多通道声学回声消除任务，旨在提升复杂环境下的回声抑制性能。针对传统方法对空间信息利用不足的问题，提出两阶段算法：首阶段用轻量DNN预测声源方向，次阶段融合方向信息与多通道信号优化AEC网络，实验证明其泛化性更强。**

- **链接: [http://arxiv.org/pdf/2505.19493v1](http://arxiv.org/pdf/2505.19493v1)**

> **作者:** Fei Zhao; Xueliang Zhang; Zhong-Qiu Wang
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Acoustic echo cancellation (AEC) is an important speech signal processing technology that can remove echoes from microphone signals to enable natural-sounding full-duplex speech communication. While single-channel AEC is widely adopted, multi-channel AEC can leverage spatial cues afforded by multiple microphones to achieve better performance. Existing multi-channel AEC approaches typically combine beamforming with deep neural networks (DNN). This work proposes a two-stage algorithm that enhances multi-channel AEC by incorporating sound source directional cues. Specifically, a lightweight DNN is first trained to predict the sound source directions, and then the predicted directional information, multi-channel microphone signals, and single-channel far-end signal are jointly fed into an AEC network to estimate the near-end signal. Evaluation results show that the proposed algorithm outperforms baseline approaches and exhibits robust generalization across diverse acoustic environments.
>
---
#### [new 030] BR-ASR: Efficient and Scalable Bias Retrieval Framework for Contextual Biasing ASR in Speech LLM
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文提出BR-ASR框架，针对语音大模型ASR中大规模上下文偏置（如专有名词、罕见词）的挑战，通过语音-偏置对比学习和动态课程学习解决同音词混淆问题，实现高效可扩展的偏置词检索。实验显示其在200k偏置词下仍保持SOTA性能，仅增加20ms延迟。**

- **链接: [http://arxiv.org/pdf/2505.19179v1](http://arxiv.org/pdf/2505.19179v1)**

> **作者:** Xun Gong; Anqi Lv; Zhiming Wang; Huijia Zhu; Yanmin Qian
>
> **备注:** Accepted by InterSpeech 2025
>
> **摘要:** While speech large language models (SpeechLLMs) have advanced standard automatic speech recognition (ASR), contextual biasing for named entities and rare words remains challenging, especially at scale. To address this, we propose BR-ASR: a Bias Retrieval framework for large-scale contextual biasing (up to 200k entries) via two innovations: (1) speech-and-bias contrastive learning to retrieve semantically relevant candidates; (2) dynamic curriculum learning that mitigates homophone confusion which negatively impacts the final performance. The is a general framework that allows seamless integration of the retrieved candidates into diverse ASR systems without fine-tuning. Experiments on LibriSpeech test-clean/-other achieve state-of-the-art (SOTA) biased word error rates (B-WER) of 2.8%/7.1% with 2000 bias words, delivering 45% relative improvement over prior methods. BR-ASR also demonstrates high scalability: when expanding the bias list to 200k where traditional methods generally fail, it induces only 0.3 / 2.9% absolute WER / B-WER degradation with a 99.99% pruning rate and only 20ms latency per query on test-other.
>
---
#### [new 031] Accelerating Diffusion-based Text-to-Speech Model Training with Dual Modality Alignment
- **分类: eess.AS; cs.SD**

- **简介: 该论文属文本到语音生成任务，旨在加速扩散模型训练并降低计算成本。提出A-DMA方法，通过文本引导与语音引导的双模态对齐策略，优化隐层状态与判别特征的匹配，减少复杂表示学习对扩散过程的依赖，实现训练速度翻倍且性能提升。**

- **链接: [http://arxiv.org/pdf/2505.19595v1](http://arxiv.org/pdf/2505.19595v1)**

> **作者:** Jeongsoo Choi; Zhikang Niu; Ji-Hoon Kim; Chunhui Wang; Joon Son Chung; Chen Xie
>
> **备注:** Interspeech 2025
>
> **摘要:** The goal of this paper is to optimize the training process of diffusion-based text-to-speech models. While recent studies have achieved remarkable advancements, their training demands substantial time and computational costs, largely due to the implicit guidance of diffusion models in learning complex intermediate representations. To address this, we propose A-DMA, an effective strategy for Accelerating training with Dual Modality Alignment. Our method introduces a novel alignment pipeline leveraging both text and speech modalities: text-guided alignment, which incorporates contextual representations, and speech-guided alignment, which refines semantic representations. By aligning hidden states with discriminative features, our training scheme reduces the reliance on diffusion models for learning complex representations. Extensive experiments demonstrate that A-DMA doubles the convergence speed while achieving superior performance over baselines. Code and demo samples are available at: https://github.com/ZhikangNiu/A-DMA
>
---
#### [new 032] Improving Speech Emotion Recognition Through Cross Modal Attention Alignment and Balanced Stacking Model
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音情感识别任务，针对自然语音中情感数据不平衡和跨模态融合效果不佳的问题，提出结合跨模态注意力对齐与平衡堆叠模型的方法。通过加权交叉熵损失及中性表达软-margin损失解决类别不平衡，并训练12个模型后集成，实现8类情感识别的MacroF1 0.4094和准确率0.4128。**

- **链接: [http://arxiv.org/pdf/2505.20007v1](http://arxiv.org/pdf/2505.20007v1)**

> **作者:** Lucas Ueda; João Lima; Leonardo Marques; Paula Costa
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Emotion plays a fundamental role in human interaction, and therefore systems capable of identifying emotions in speech are crucial in the context of human-computer interaction. Speech emotion recognition (SER) is a challenging problem, particularly in natural speech and when the available data is imbalanced across emotions. This paper presents our proposed system in the context of the 2025 Speech Emotion Recognition in Naturalistic Conditions Challenge. Our proposed architecture leverages cross-modality, utilizing cross-modal attention to fuse representations from different modalities. To address class imbalance, we employed two training designs: (i) weighted crossentropy loss (WCE); and (ii) WCE with an additional neutralexpressive soft margin loss and balancing. We trained a total of 12 multimodal models, which were ensembled using a balanced stacking model. Our proposed system achieves a MacroF1 score of 0.4094 and an accuracy of 0.4128 on 8-class speech emotion recognition.
>
---
#### [new 033] From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文提出BALSa方法，通过合成数据解决音频语言模型（ALLMs）训练中灾难性遗忘和依赖大量标注数据的问题。其工作包括生成跨模态对齐的合成数据、提出LISTEN对比训练提升声音区分能力，及扩展多音频场景训练，有效减少音频幻觉并保持模型性能。**

- **链接: [http://arxiv.org/pdf/2505.20166v1](http://arxiv.org/pdf/2505.20166v1)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. However, this adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where important textual capabilities such as instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about their reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making the process resource-intensive. To address these issues, we leverage the backbone LLMs from ALLMs to synthesize general-purpose caption-style alignment data. We refer to this process as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Building on BALSa, we introduce LISTEN (Learning to Identify Sounds Through Extended Negative Samples), a contrastive-like training method designed to improve ALLMs' ability to distinguish between present and absent sounds. We further extend BALSa to multi-audio scenarios, where the model either explains the differences between audio inputs or produces a unified caption that describes them all, thereby enhancing audio-language alignment. Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance in audio understanding, reasoning, and instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to the development of ALLMs.
>
---
#### [new 034] MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于关键词检测（KWS）任务，针对传统ASR方法效率低、未优先检测关键词的问题，提出MFA-KWS框架：采用CTC-Transducer异步解码架构，结合多头异步解码与Token-and-Duration Transducer，探索得分融合策略，提升检测准确率与速度（47%-63%加速），适用于设备端部署。**

- **链接: [http://arxiv.org/pdf/2505.19577v1](http://arxiv.org/pdf/2505.19577v1)**

> **作者:** Yu Xi; Haoyu Li; Xiaoyu Gu; Yidi Jiang; Kai Yu
>
> **备注:** TASLP under review
>
> **摘要:** Keyword spotting (KWS) is essential for voice-driven applications, demanding both accuracy and efficiency. Traditional ASR-based KWS methods, such as greedy and beam search, explore the entire search space without explicitly prioritizing keyword detection, often leading to suboptimal performance. In this paper, we propose an effective keyword-specific KWS framework by introducing a streaming-oriented CTC-Transducer-combined frame-asynchronous system with multi-head frame-asynchronous decoding (MFA-KWS). Specifically, MFA-KWS employs keyword-specific phone-synchronous decoding for CTC and replaces conventional RNN-T with Token-and-Duration Transducer to enhance both performance and efficiency. Furthermore, we explore various score fusion strategies, including single-frame-based and consistency-based methods. Extensive experiments demonstrate the superior performance of MFA-KWS, which achieves state-of-the-art results on both fixed keyword and arbitrary keywords datasets, such as Snips, MobvoiHotwords, and LibriKWS-20, while exhibiting strong robustness in noisy environments. Among fusion strategies, the consistency-based CDC-Last method delivers the best performance. Additionally, MFA-KWS achieves a 47% to 63% speed-up over the frame-synchronous baselines across various datasets. Extensive experimental results confirm that MFA-KWS is an effective and efficient KWS framework, making it well-suited for on-device deployment.
>
---
#### [new 035] Deep learning based spatial aliasing reduction in beamforming for audio capture
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频捕获波束成形领域，旨在解决空间混叠导致的高频方向模糊及精度下降问题。提出基于U-Net的信号依赖去混叠滤波器，设计独立通道与跨通道依赖两种模型，在立体声及一阶Ambisonics场景验证，显著提升客观与感知效果，证明深度学习在减少波束成形混叠的潜力。**

- **链接: [http://arxiv.org/pdf/2505.19781v1](http://arxiv.org/pdf/2505.19781v1)**

> **作者:** Mateusz Guzik; Giulio Cengarle; Daniel Arteaga
>
> **备注:** 5 pages, 4 figures; accepted for presentation in Interspeech 2025
>
> **摘要:** Spatial aliasing affects spaced microphone arrays, causing directional ambiguity above certain frequencies, degrading spatial and spectral accuracy of beamformers. Given the limitations of conventional signal processing and the scarcity of deep learning approaches to spatial aliasing mitigation, we propose a novel approach using a U-Net architecture to predict a signal-dependent de-aliasing filter, which reduces aliasing in conventional beamforming for spatial capture. Two types of multichannel filters are considered, one which treats the channels independently and a second one that models cross-channel dependencies. The proposed approach is evaluated in two common spatial capture scenarios: stereo and first-order Ambisonics. The results indicate a very significant improvement, both objective and perceptual, with respect to conventional beamforming. This work shows the potential of deep learning to reduce aliasing in beamforming, leading to improvements in multi-microphone setups.
>
---
#### [new 036] Reshaping Representation Space to Balance the Safety and Over-rejection in Large Audio Language Models
- **分类: cs.CL; cs.MM; cs.SD; eess.AS**

- **简介: 该论文针对大音频语言模型（LALM）的安全对齐不足及过度拒绝问题，提出无监督安全微调策略，通过重塑模型表示空间，在提升安全性的同时仅小幅增加过度拒绝率（0.88%），适用于多模态输入。**

- **链接: [http://arxiv.org/pdf/2505.19670v1](http://arxiv.org/pdf/2505.19670v1)**

> **作者:** Hao Yang; Lizhen Qu; Ehsan Shareghi; Gholamreza Haffari
>
> **摘要:** Large Audio Language Models (LALMs) have extended the capabilities of Large Language Models (LLMs) by enabling audio-based human interactions. However, recent research has revealed that LALMs remain vulnerable to harmful queries due to insufficient safety-alignment. Despite advances in defence measures for text and vision LLMs, effective safety-alignment strategies and audio-safety dataset specifically targeting LALMs are notably absent. Meanwhile defence measures based on Supervised Fine-tuning (SFT) struggle to address safety improvement while avoiding over-rejection issues, significantly compromising helpfulness. In this work, we propose an unsupervised safety-fine-tuning strategy as remedy that reshapes model's representation space to enhance existing LALMs safety-alignment while balancing the risk of over-rejection. Our experiments, conducted across three generations of Qwen LALMs, demonstrate that our approach significantly improves LALMs safety under three modality input conditions (audio-text, text-only, and audio-only) while increasing over-rejection rate by only 0.88% on average. Warning: this paper contains harmful examples.
>
---
#### [new 037] WHISTRESS: Enriching Transcriptions with Sentence Stress Detection
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音处理任务，旨在通过检测句子重音增强语音转录系统。提出无需对齐的WHISTRESS模型及合成数据集TINYSTRESS-15K，解决现有方法依赖额外输入或标注数据的问题，实验显示其超越基线并具备跨领域零样本泛化能力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.19103v1](http://arxiv.org/pdf/2505.19103v1)**

> **作者:** Iddo Yosha; Dorin Shteyman; Yossi Adi
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** Spoken language conveys meaning not only through words but also through intonation, emotion, and emphasis. Sentence stress, the emphasis placed on specific words within a sentence, is crucial for conveying speaker intent and has been extensively studied in linguistics. In this work, we introduce WHISTRESS, an alignment-free approach for enhancing transcription systems with sentence stress detection. To support this task, we propose TINYSTRESS-15K, a scalable, synthetic training data for the task of sentence stress detection which resulted from a fully automated dataset creation process. We train WHISTRESS on TINYSTRESS-15K and evaluate it against several competitive baselines. Our results show that WHISTRESS outperforms existing methods while requiring no additional input priors during training or inference. Notably, despite being trained on synthetic data, WHISTRESS demonstrates strong zero-shot generalization across diverse benchmarks. Project page: https://pages.cs.huji.ac.il/adiyoss-lab/whistress.
>
---
#### [new 038] LiSTEN: Learning Soft Token Embeddings for Neural Audio LLMs
- **分类: cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音频-语言任务适应，解决LLMs在多任务音频处理中因环境差异导致的过拟合及数据依赖问题。提出LiSTEN框架，通过动态提示选择与可学习键值对平衡通用与任务知识，减少参数，简化单阶段训练，提升可解释性。**

- **链接: [http://arxiv.org/pdf/2505.18517v1](http://arxiv.org/pdf/2505.18517v1)**

> **作者:** Pooneh Mousavi; Shubham Gupta; Cem Subakan; Mirco Ravanelli
>
> **摘要:** Foundation models based on large language models (LLMs) have shown great success in handling various tasks and modalities. However, adapting these models for general-purpose audio-language tasks is challenging due to differences in acoustic environments and task variations. In this work, we introduce LiSTEN Learning Soft Token Embeddings for Neural Audio LLMs), a framework for adapting LLMs to speech and audio tasks. LiSTEN uses a dynamic prompt selection strategy with learnable key-value pairs, allowing the model to balance general and task-specific knowledge while avoiding overfitting in a multitask setting. Our approach reduces dependence on large-scale ASR or captioning datasets, achieves competitive performance with fewer trainable parameters, and simplifies training by using a single-stage process. Additionally, LiSTEN enhances interpretability by analyzing the diversity and overlap of selected prompts across different tasks.
>
---
#### [new 039] Accelerating Flow-Matching-Based Text-to-Speech via Empirically Pruned Step Sampling
- **分类: eess.AS; cs.SD**

- **简介: 该论文针对基于流匹配的文本到语音合成（TTS）模型推理速度慢的问题，提出无训练方法Fast F5-TTS。通过分析采样轨迹，提出非均匀采样策略EPSS，剪除冗余步骤，使F5-TTS推理速度提升4倍（RTF 0.03），并适用于E2 TTS，保持性能。**

- **链接: [http://arxiv.org/pdf/2505.19931v1](http://arxiv.org/pdf/2505.19931v1)**

> **作者:** Qixi Zheng; Yushen Chen; Zhikang Niu; Ziyang Ma; Xiaofei Wang; Kai Yu; Xie Chen
>
> **摘要:** Flow-matching-based text-to-speech (TTS) models, such as Voicebox, E2 TTS, and F5-TTS, have attracted significant attention in recent years. These models require multiple sampling steps to reconstruct speech from noise, making inference speed a key challenge. Reducing the number of sampling steps can greatly improve inference efficiency. To this end, we introduce Fast F5-TTS, a training-free approach to accelerate the inference of flow-matching-based TTS models. By inspecting the sampling trajectory of F5-TTS, we identify redundant steps and propose Empirically Pruned Step Sampling (EPSS), a non-uniform time-step sampling strategy that effectively reduces the number of sampling steps. Our approach achieves a 7-step generation with an inference RTF of 0.030 on an NVIDIA RTX 3090 GPU, making it 4 times faster than the original F5-TTS while maintaining comparable performance. Furthermore, EPSS performs well on E2 TTS models, demonstrating its strong generalization ability.
>
---
#### [new 040] Acoustic and Machine Learning Methods for Speech-Based Suicide Risk Assessment: A Systematic Review
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属系统综述，探讨AI/ML通过语音声学分析评估自杀风险，以改进早期检测。分析33项研究，发现高危人群声学特征（如抖动、基频）显著差异，多模态ML效果更优，但受限于样本小、方法不一等，建议标准化及数据扩展。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18195v1](http://arxiv.org/pdf/2505.18195v1)**

> **作者:** Ambre Marie; Marine Garnier; Thomas Bertin; Laura Machart; Guillaume Dardenne; Gwenolé Quellec; Sofian Berrouiguet
>
> **备注:** Preprint version of a manuscript submitted to Computers in Biology and Medicine. 25 pages, 7 figures, 8 tables
>
> **摘要:** Suicide remains a public health challenge, necessitating improved detection methods to facilitate timely intervention and treatment. This systematic review evaluates the role of Artificial Intelligence (AI) and Machine Learning (ML) in assessing suicide risk through acoustic analysis of speech. Following PRISMA guidelines, we analyzed 33 articles selected from PubMed, Cochrane, Scopus, and Web of Science databases. These studies primarily explored acoustic differences between individuals at risk of suicide (RS) and those not at risk (NRS), and evaluated ML classifier performance. Findings consistently showed significant acoustic feature variations between RS and NRS populations, particularly involving jitter, fundamental frequency (F0), Mel-frequency cepstral coefficients (MFCC), and power spectral density (PSD). Classifier effectiveness varied based on algorithms, modalities, and speech elicitation methods, with multimodal approaches integrating acoustic, linguistic, and metadata features demonstrating superior performance. However, limitations such as methodological variability, small sample sizes, lack of longitudinal data, and limited linguistic and demographic diversity restrict generalizability. Future research should focus on standardizing methods, expanding multimodal analyses, and utilizing larger, diverse datasets to support AI integration in clinical suicide risk assessment.
>
---
#### [new 041] DeepDialogue: A Multi-Turn Emotionally-Rich Spoken Dialogue Dataset
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出DeepDialogue数据集，解决现有对话数据在多回合、情感表达、领域多样性和模态上的不足。构建含40,150个多领域对话的多模态数据集，覆盖20种情绪并合成情感一致语音，通过多模型生成与筛选，揭示对话连贯性规律，推动跨模态对话系统发展。**

- **链接: [http://arxiv.org/pdf/2505.19978v1](http://arxiv.org/pdf/2505.19978v1)**

> **作者:** Alkis Koudounas; Moreno La Quatra; Elena Baralis
>
> **备注:** Currently under review. See the official website: https://salt-research.github.io/DeepDialogue
>
> **摘要:** Recent advances in conversational AI have demonstrated impressive capabilities in single-turn responses, yet multi-turn dialogues remain challenging for even the most sophisticated language models. Current dialogue datasets are limited in their emotional range, domain diversity, turn depth, and are predominantly text-only, hindering progress in developing more human-like conversational systems across modalities. To address these limitations, we present DeepDialogue, a large-scale multimodal dataset containing 40,150 high-quality multi-turn dialogues spanning 41 domains and incorporating 20 distinct emotions with coherent emotional progressions. Our approach pairs 9 different language models (4B-72B parameters) to generate 65,600 initial conversations, which we then evaluate through a combination of human annotation and LLM-based quality filtering. The resulting dataset reveals fundamental insights: smaller models fail to maintain coherence beyond 6 dialogue turns; concrete domains (e.g., "cars," "travel") yield more meaningful conversations than abstract ones (e.g., "philosophy"); and cross-model interactions produce more coherent dialogues than same-model conversations. A key contribution of DeepDialogue is its speech component, where we synthesize emotion-consistent voices for all 40,150 dialogues, creating the first large-scale open-source multimodal dialogue dataset that faithfully preserves emotional context across multi-turn conversations.
>
---
#### [new 042] Evaluating the Usefulness of Non-Diagnostic Speech Data for Developing Parkinson's Disease Classifiers
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于帕金森病（PD）分类任务，探讨非诊断语音数据（如日常对话的Turn-Taking数据集）在PD检测中的有效性。针对现有研究依赖诊断专用语音数据的问题，作者比较了其与传统PC-GITA数据集的性能，分析数据特征（如平衡性别/患者分布、合并录音）对分类的影响，并揭示跨数据集模型泛化差异及个体说话人变异性对结果的影响。**

- **链接: [http://arxiv.org/pdf/2505.18722v1](http://arxiv.org/pdf/2505.18722v1)**

> **作者:** Terry Yi Zhong; Esther Janse; Cristian Tejedor-Garcia; Louis ten Bosch; Martha Larson
>
> **备注:** Accepted for Interspeech 2025 (Camera-Ready)
>
> **摘要:** Speech-based Parkinson's disease (PD) detection has gained attention for its automated, cost-effective, and non-intrusive nature. As research studies usually rely on data from diagnostic-oriented speech tasks, this work explores the feasibility of diagnosing PD on the basis of speech data not originally intended for diagnostic purposes, using the Turn-Taking (TT) dataset. Our findings indicate that TT can be as useful as diagnostic-oriented PD datasets like PC-GITA. We also investigate which specific dataset characteristics impact PD classification performance. The results show that concatenating audio recordings and balancing participants' gender and status distributions can be beneficial. Cross-dataset evaluation reveals that models trained on PC-GITA generalize poorly to TT, whereas models trained on TT perform better on PC-GITA. Furthermore, we provide insights into the high variability across folds, which is mainly due to large differences in individual speaker performance.
>
---
#### [new 043] GSA-TTS : Toward Zero-Shot Speech Synthesis based on Gradual Style Adaptor
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于零样本语音合成任务，旨在解决无需目标说话人数据的情况下生成自然且相似语音的挑战。提出GSA-TTS模型，通过渐进式风格适配器分层编码声学参考的局部与全局风格，利用自注意力融合风格特征，提升语音自然度、可懂度及说话人相似度，同时探索模型的可控性与可解释性。**

- **链接: [http://arxiv.org/pdf/2505.19384v1](http://arxiv.org/pdf/2505.19384v1)**

> **作者:** Seokgi Lee; Jungjun Kim
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** We present the gradual style adaptor TTS (GSA-TTS) with a novel style encoder that gradually encodes speaking styles from an acoustic reference for zero-shot speech synthesis. GSA first captures the local style of each semantic sound unit. Then the local styles are combined by self-attention to obtain a global style condition. This semantic and hierarchical encoding strategy provides a robust and rich style representation for an acoustic model. We test GSA-TTS on unseen speakers and obtain promising results regarding naturalness, speaker similarity, and intelligibility. Additionally, we explore the potential of GSA in terms of interpretability and controllability, which stems from its hierarchical structure.
>
---
#### [new 044] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文提出多模态多语言歌词翻译任务，解决动画歌曲翻译中语义、节奏、音节及视觉听觉同步难题。构建MAVL数据集整合文本/音频/视频，并设计SylAVL-CoT模型，利用音视频线索和音节约束提升歌词可唱性与准确性。**

- **链接: [http://arxiv.org/pdf/2505.18614v1](http://arxiv.org/pdf/2505.18614v1)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [new 045] Efficient Speech Translation through Model Compression and Knowledge Distillation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文针对语音翻译任务中大型模型计算需求高的问题，提出结合迭代层剪枝、4-bit量化低秩适配（QLoRA）和知识蒸馏的压缩方法。通过参数减半（存储降50%）同时保持97-100%翻译质量，提升模型部署效率。**

- **链接: [http://arxiv.org/pdf/2505.20237v1](http://arxiv.org/pdf/2505.20237v1)**

> **作者:** Yasmin Moslem
>
> **备注:** IWSLT 2025
>
> **摘要:** Efficient deployment of large audio-language models for speech translation remains challenging due to their significant computational requirements. In this paper, we address this challenge through our system submissions to the "Model Compression" track at the International Conference on Spoken Language Translation (IWSLT 2025). We experiment with a combination of approaches including iterative layer pruning based on layer importance evaluation, low-rank adaptation with 4-bit quantization (QLoRA), and knowledge distillation. In our experiments, we use Qwen2-Audio-7B-Instruct for speech translation into German and Chinese. Our pruned (student) models achieve up to a 50% reduction in both model parameters and storage footprint, while retaining 97-100% of the translation quality of the in-domain (teacher) models.
>
---
#### [new 046] ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多模态大语言模型的语音语言理解任务，旨在解决缺乏语音-文本跨模态对齐质量评估标准的问题。提出ALAS指标，通过分析Transformer层中音频与文本表征的关联性，评估模型在问答和情感识别任务中的跨模态对齐效果。**

- **链接: [http://arxiv.org/pdf/2505.19937v1](http://arxiv.org/pdf/2505.19937v1)**

> **作者:** Pooneh Mousavi; Yingzhi Wang; Mirco Ravanelli; Cem Subakan
>
> **摘要:** Large Language Models (LLMs) are widely used in Spoken Language Understanding (SLU). Recent SLU models process audio directly by adapting speech input into LLMs for better multimodal learning. A key consideration for these models is the cross-modal alignment between text and audio modalities, which is a telltale sign as to whether or not LLM is able to associate semantic meaning to audio segments. While various methods exist for fusing these modalities, there is no standard metric to evaluate alignment quality in LLMs. In this work, we propose a new metric, ALAS (Automatic Latent Alignment Score). Our study examines the correlation between audio and text representations across transformer layers, for two different tasks (Spoken Question Answering and Emotion Recognition). We showcase that our metric behaves as expected across different layers and different tasks.
>
---
#### [new 047] SoloSpeech: Enhancing Intelligibility and Quality in Target Speech Extraction through a Cascaded Generative Pipeline
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于目标语音提取（TSE）任务，旨在解决现有模型在语音分离中存在音质缺陷、自然度不足及环境敏感性问题。提出SoloSpeech级联生成框架，通过压缩、提取、重建和校正模块，利用线索音频的潜在空间对齐混合语音，无需说话人嵌入，实现高质量、高可懂度的语音分离，在Libri2Mix数据集及真实场景中达新SOTA。**

- **链接: [http://arxiv.org/pdf/2505.19314v1](http://arxiv.org/pdf/2505.19314v1)**

> **作者:** Helin Wang; Jiarui Hai; Dongchao Yang; Chen Chen; Kai Li; Junyi Peng; Thomas Thebaud; Laureano Moro Velazquez; Jesus Villalba; Najim Dehak
>
> **摘要:** Target Speech Extraction (TSE) aims to isolate a target speaker's voice from a mixture of multiple speakers by leveraging speaker-specific cues, typically provided as auxiliary audio (a.k.a. cue audio). Although recent advancements in TSE have primarily employed discriminative models that offer high perceptual quality, these models often introduce unwanted artifacts, reduce naturalness, and are sensitive to discrepancies between training and testing environments. On the other hand, generative models for TSE lag in perceptual quality and intelligibility. To address these challenges, we present SoloSpeech, a novel cascaded generative pipeline that integrates compression, extraction, reconstruction, and correction processes. SoloSpeech features a speaker-embedding-free target extractor that utilizes conditional information from the cue audio's latent space, aligning it with the mixture audio's latent space to prevent mismatches. Evaluated on the widely-used Libri2Mix dataset, SoloSpeech achieves the new state-of-the-art intelligibility and quality in target speech extraction and speech separation tasks while demonstrating exceptional generalization on out-of-domain data and real-world scenarios.
>
---
#### [new 048] Mel-McNet: A Mel-Scale Framework for Online Multichannel Speech Enhancement
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于在线多通道语音增强任务，旨在解决传统线性频域方法计算效率低且较少采用更符合人类听觉的梅尔频谱的问题。提出Mel-McNet框架，通过STFT-to-Mel模块压缩多通道频谱并改进McNet骨干网络直接处理梅尔域特征，实现计算量降低60%的同时保持增强与ASR性能，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19576v1](http://arxiv.org/pdf/2505.19576v1)**

> **作者:** Yujie Yang; Bing Yang; Xiaofei Li
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Online multichannel speech enhancement has been intensively studied recently. Though Mel-scale frequency is more matched with human auditory perception and computationally efficient than linear frequency, few works are implemented in a Mel-frequency domain. To this end, this work proposes a Mel-scale framework (namely Mel-McNet). It processes spectral and spatial information with two key components: an effective STFT-to-Mel module compressing multi-channel STFT features into Mel-frequency representations, and a modified McNet backbone directly operating in the Mel domain to generate enhanced LogMel spectra. The spectra can be directly fed to vocoders for waveform reconstruction or ASR systems for transcription. Experiments on CHiME-3 show that Mel-McNet can reduce computational complexity by 60% while maintaining comparable enhancement and ASR performance to the original McNet. Mel-McNet also outperforms other SOTA methods, verifying the potential of Mel-scale speech enhancement.
>
---
#### [new 049] VoiceStar: Robust Zero-Shot Autoregressive TTS with Duration Control and Extrapolation
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出VoiceStar，首个零样本自回归TTS模型，解决长语音生成时的时长控制与外推问题。通过PM-RoPE位置嵌入和CPM训练，实现文本-语音对齐、时长指示及超长语音生成，提升可懂度与自然度，在长语音基准测试中表现最优。**

- **链接: [http://arxiv.org/pdf/2505.19462v1](http://arxiv.org/pdf/2505.19462v1)**

> **作者:** Puyuan Peng; Shang-Wen Li; Abdelrahman Mohamed; David Harwath
>
> **摘要:** We present VoiceStar, the first zero-shot TTS model that achieves both output duration control and extrapolation. VoiceStar is an autoregressive encoder-decoder neural codec language model, that leverages a novel Progress-Monitoring Rotary Position Embedding (PM-RoPE) and is trained with Continuation-Prompt Mixed (CPM) training. PM-RoPE enables the model to better align text and speech tokens, indicates the target duration for the generated speech, and also allows the model to generate speech waveforms much longer in duration than those seen during. CPM training also helps to mitigate the training/inference mismatch, and significantly improves the quality of the generated speech in terms of speaker similarity and intelligibility. VoiceStar outperforms or is on par with current state-of-the-art models on short-form benchmarks such as Librispeech and Seed-TTS, and significantly outperforms these models on long-form/extrapolation benchmarks (20-50s) in terms of intelligibility and naturalness. Code and model weights: https://github.com/jasonppy/VoiceStar
>
---
#### [new 050] SpeakStream: Streaming Text-to-Speech with Interleaved Data
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于流式文本转语音任务，旨在解决传统TTS系统因完整语句处理导致的高延迟问题，尤其在对接流式LLM时首词延迟过大的痛点。提出SpeakStream系统，采用解码器架构与交错文本-语音数据训练，实现边接收文本边生成音频，兼顾低延迟与高质量。**

- **链接: [http://arxiv.org/pdf/2505.19206v1](http://arxiv.org/pdf/2505.19206v1)**

> **作者:** Richard He Bai; Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** The latency bottleneck of traditional text-to-speech (TTS) systems fundamentally hinders the potential of streaming large language models (LLMs) in conversational AI. These TTS systems, typically trained and inferenced on complete utterances, introduce unacceptable delays, even with optimized inference speeds, when coupled with streaming LLM outputs. This is particularly problematic for creating responsive conversational agents where low first-token latency is critical. In this paper, we present SpeakStream, a streaming TTS system that generates audio incrementally from streaming text using a decoder-only architecture. SpeakStream is trained using a next-step prediction loss on interleaved text-speech data. During inference, it generates speech incrementally while absorbing streaming input text, making it particularly suitable for cascaded conversational AI agents where an LLM streams text to a TTS system. Our experiments demonstrate that SpeakStream achieves state-of-the-art latency results in terms of first-token latency while maintaining the quality of non-streaming TTS systems.
>
---
#### [new 051] FlowSE: Efficient and High-Quality Speech Enhancement via Flow Matching
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决现有语言模型量化损失及扩散模型训练复杂、推理延迟高的问题。提出FlowSE模型，通过流匹配学习噪声与清晰语音分布的连续变换，单次推理实现高质量增强，支持文本辅助或无文本场景，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.19476v1](http://arxiv.org/pdf/2505.19476v1)**

> **作者:** Ziqian Wang; Zikai Liu; Xinfa Zhu; Yike Zhu; Mingshuai Liu; Jun Chen; Longshuai Xiao; Chao Weng; Lei Xie
>
> **备注:** Accepted to InterSpeech 2025
>
> **摘要:** Generative models have excelled in audio tasks using approaches such as language models, diffusion, and flow matching. However, existing generative approaches for speech enhancement (SE) face notable challenges: language model-based methods suffer from quantization loss, leading to compromised speaker similarity and intelligibility, while diffusion models require complex training and high inference latency. To address these challenges, we propose FlowSE, a flow-matching-based model for SE. Flow matching learns a continuous transformation between noisy and clean speech distributions in a single pass, significantly reducing inference latency while maintaining high-quality reconstruction. Specifically, FlowSE trains on noisy mel spectrograms and optional character sequences, optimizing a conditional flow matching loss with ground-truth mel spectrograms as supervision. It implicitly learns speech's temporal-spectral structure and text-speech alignment. During inference, FlowSE can operate with or without textual information, achieving impressive results in both scenarios, with further improvements when transcripts are available. Extensive experiments demonstrate that FlowSE significantly outperforms state-of-the-art generative methods, establishing a new paradigm for generative-based SE and demonstrating the potential of flow matching to advance the field. Our code, pre-trained checkpoints, and audio samples are available.
>
---
#### [new 052] SACM: SEEG-Audio Contrastive Matching for Chinese Speech Decoding
- **分类: cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于中文语音解码脑机接口任务，旨在帮助语言障碍患者恢复交流能力。通过收集癫痫患者SEEG脑电与同步音频数据，提出对比学习框架SACM，实现高精度语音检测与解码，并发现单个运动皮层电极即可达到阵列级效果。**

- **链接: [http://arxiv.org/pdf/2505.19652v1](http://arxiv.org/pdf/2505.19652v1)**

> **作者:** Hongbin Wang; Zhihong Jia; Yuanzhong Shen; Ziwei Wang; Siyang Li; Kai Shu; Feng Hu; Dongrui Wu
>
> **摘要:** Speech disorders such as dysarthria and anarthria can severely impair the patient's ability to communicate verbally. Speech decoding brain-computer interfaces (BCIs) offer a potential alternative by directly translating speech intentions into spoken words, serving as speech neuroprostheses. This paper reports an experimental protocol for Mandarin Chinese speech decoding BCIs, along with the corresponding decoding algorithms. Stereo-electroencephalography (SEEG) and synchronized audio data were collected from eight drug-resistant epilepsy patients as they conducted a word-level reading task. The proposed SEEG and Audio Contrastive Matching (SACM), a contrastive learning-based framework, achieved decoding accuracies significantly exceeding chance levels in both speech detection and speech decoding tasks. Electrode-wise analysis revealed that a single sensorimotor cortex electrode achieved performance comparable to that of the full electrode array. These findings provide valuable insights for developing more accurate online speech decoding BCIs.
>
---
#### [new 053] Enhancing Generalization of Speech Large Language Models with Multi-Task Behavior Imitation and Speech-Text Interleaving
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音大语言模型泛化任务。针对标注语音数据不足导致对齐效率低和泛化差的问题，提出MTBI方法：结合多任务行为模仿与语音-文本交织技术，利用配对数据使模型生成一致响应，提升泛化能力。实验显示其优于现有方法且数据需求更少。（99字）**

- **链接: [http://arxiv.org/pdf/2505.18644v1](http://arxiv.org/pdf/2505.18644v1)**

> **作者:** Jingran Xie; Xiang Li; Hui Wang; Yue Yu; Yang Xiang; Xixin Wu; Zhiyong Wu
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Large language models (LLMs) have shown remarkable generalization across tasks, leading to increased interest in integrating speech with LLMs. These speech LLMs (SLLMs) typically use supervised fine-tuning to align speech with text-based LLMs. However, the lack of annotated speech data across a wide range of tasks hinders alignment efficiency, resulting in poor generalization. To address these issues, we propose a novel multi-task 'behavior imitation' method with speech-text interleaving, called MTBI, which relies solely on paired speech and transcripts. By ensuring the LLM decoder generates equivalent responses to paired speech and text, we achieve a more generalized SLLM. Interleaving is used to further enhance alignment efficiency. We introduce a simple benchmark to evaluate prompt and task generalization across different models. Experimental results demonstrate that our MTBI outperforms SOTA SLLMs on both prompt and task generalization, while requiring less supervised speech data.
>
---
## 更新

#### [replaced 001] Bemba Speech Translation: Exploring a Low-Resource African Language
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.02518v2](http://arxiv.org/pdf/2505.02518v2)**

> **作者:** Muhammad Hazim Al Farouq; Aman Kassahun Wassie; Yasmin Moslem
>
> **备注:** IWSLT 2025
>
> **摘要:** This paper describes our system submission to the International Conference on Spoken Language Translation (IWSLT 2025), low-resource languages track, namely for Bemba-to-English speech translation. We built cascaded speech translation systems based on Whisper and NLLB-200, and employed data augmentation techniques, such as back-translation. We investigate the effect of using synthetic data and discuss our experimental setup.
>
---
#### [replaced 002] LipDiffuser: Lip-to-Speech Generation with Conditional Diffusion Models
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.11391v2](http://arxiv.org/pdf/2505.11391v2)**

> **作者:** Danilo de Oliveira; Julius Richter; Tal Peer; Timo Gerkmann
>
> **摘要:** We present LipDiffuser, a conditional diffusion model for lip-to-speech generation synthesizing natural and intelligible speech directly from silent video recordings. Our approach leverages the magnitude-preserving ablated diffusion model (MP-ADM) architecture as a denoiser model. To effectively condition the model, we incorporate visual features using magnitude-preserving feature-wise linear modulation (MP-FiLM) alongside speaker embeddings. A neural vocoder then reconstructs the speech waveform from the generated mel-spectrograms. Evaluations on LRS3 and TCD-TIMIT demonstrate that LipDiffuser outperforms existing lip-to-speech baselines in perceptual speech quality and speaker similarity, while remaining competitive in downstream automatic speech recognition (ASR). These findings are also supported by a formal listening experiment. Extensive ablation studies and cross-dataset evaluation confirm the effectiveness and generalization capabilities of our approach.
>
---
#### [replaced 003] FireRedTTS-1S: An Upgraded Streamable Foundation Text-to-Speech System
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.20499v3](http://arxiv.org/pdf/2503.20499v3)**

> **作者:** Hao-Han Guo; Yao Hu; Fei-Yu Shen; Xu Tang; Yi-Chen Wu; Feng-Long Xie; Kun Xie
>
> **摘要:** In this work, we upgrade FireRedTTS to a new version, FireRedTTS-1S, a high-quality streaming foundation text-to-speech system. FireRedTTS-1S achieves streaming speech generation via two steps: text-to-semantic decoding and semantic-to-acoustic decoding. In text-to-semantic decoding, a semantic-aware speech tokenizer converts the speech signal into semantic tokens, which can be synthesized from the text via a language model in an auto-regressive manner. Meanwhile, the semantic-to-acoustic decoding module simultaneously translates generated semantic tokens into the speech signal in a streaming way. We implement two approaches to achieve this module: 1) a chunk-wise streamable flow-matching approach, and 2) a multi-stream language model-based approach. They both present high-quality and streamable speech generation but differ in real-time factor (RTF) and latency. Specifically, flow-matching decoding can generate speech by chunks, presenting a lower RTF of 0.1 but a higher latency of 300ms. Instead, the multi-stream language model generates speech by frames in an autoregressive manner, presenting a higher RTF of 0.3 but a low latency of 150ms. In experiments on zero-shot voice cloning, the objective results validate FireRedTTS-1S as a high-quality foundation model with comparable intelligibility and speaker similarity over industrial baseline systems. Furthermore, the subjective score of FireRedTTS-1S highlights its impressive synthesis performance, achieving comparable quality to the ground-truth recordings. These results validate FireRedTTS-1S as a high-quality streaming foundation TTS system.
>
---
#### [replaced 004] Deep Active Speech Cancellation with Mamba-Masking Network
- **分类: cs.SD; cs.AI; cs.LG; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2502.01185v2](http://arxiv.org/pdf/2502.01185v2)**

> **作者:** Yehuda Mishaly; Lior Wolf; Eliya Nachmani
>
> **摘要:** We present a novel deep learning network for Active Speech Cancellation (ASC), advancing beyond Active Noise Cancellation (ANC) methods by effectively canceling both noise and speech signals. The proposed Mamba-Masking architecture introduces a masking mechanism that directly interacts with the encoded reference signal, enabling adaptive and precisely aligned anti-signal generation-even under rapidly changing, high-frequency conditions, as commonly found in speech. Complementing this, a multi-band segmentation strategy further improves phase alignment across frequency bands. Additionally, we introduce an optimization-driven loss function that provides near-optimal supervisory signals for anti-signal generation. Experimental results demonstrate substantial performance gains, achieving up to 7.2dB improvement in ANC scenarios and 6.2dB in ASC, significantly outperforming existing methods.
>
---
#### [replaced 005] The Faetar Benchmark: Speech Recognition in a Very Under-Resourced Language
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.08103v4](http://arxiv.org/pdf/2409.08103v4)**

> **作者:** Michael Ong; Sean Robertson; Leo Peckham; Alba Jorquera Jimenez de Aberasturi; Paula Arkhangorodsky; Robin Huo; Aman Sakhardande; Mark Hallap; Naomi Nagy; Ewan Dunbar
>
> **备注:** To appear in INTERSPEECH 2025
>
> **摘要:** We introduce the Faetar Automatic Speech Recognition Benchmark, a benchmark corpus designed to push the limits of current approaches to low-resource speech recognition. Faetar, a Franco-Proven\c{c}al variety spoken primarily in Italy, has no standard orthography, has virtually no existing textual or speech resources other than what is included in the benchmark, and is quite different from other forms of Franco-Proven\c{c}al. The corpus comes from field recordings, most of which are noisy, for which only 5 hrs have matching transcriptions, and for which forced alignment is of variable quality. The corpus contains an additional 20 hrs of unlabelled speech. We report baseline results from state-of-the-art multilingual speech foundation models with a best phone error rate of 30.4%, using a pipeline that continues pre-training on the foundation model using the unlabelled set.
>
---
#### [replaced 006] Towards End-to-End Training of Automatic Speech Recognition for Nigerian Pidgin
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2010.11123v2](http://arxiv.org/pdf/2010.11123v2)**

> **作者:** Amina Mardiyyah Rufai; Afolabi Abeeb; Esther Oduntan; Tayo Arulogun; Oluwabukola Adegboro; Daniel Ajisafe
>
> **备注:** Updated empirical results, included additional architectures, and added a zero-shot baseline
>
> **摘要:** The prevalence of automatic speech recognition (ASR) systems in spoken language applications has increased significantly in recent years. Notably, many African languages lack sufficient linguistic resources to support the robustness of these systems. This paper focuses on the development of an end-to-end speech recognition system customized for Nigerian Pidgin English. We investigated and evaluated different pretrained state-of-the-art architectures on a new dataset. Our empirical results demonstrate a notable performance of the variant Wav2Vec2 XLSR-53 on our dataset, achieving a word error rate (WER) of 29.6% on the test set, surpassing other architectures such as NEMO QUARTZNET and Wav2Vec2.0 BASE-100H in quantitative assessments. Additionally, we demonstrate that pretrained state-of-the-art architectures do not work well out-of-the-box. We performed zero-shot evaluation using XLSR-English as the baseline, chosen for its similarity to Nigerian Pidgin. This yielded a higher WER of 73.7%. By adapting this architecture to nuances represented in our dataset, we reduce error by 59.84%. Our dataset comprises 4,288 recorded utterances from 10 native speakers, partitioned into training, validation, and test sets. This study underscores the potential for improving ASR systems for under-resourced languages like Nigerian Pidgin English, contributing to greater inclusion in speech technology applications. We publicly release our unique parallel dataset (speech-to-text) on Nigerian Pidgin, as well as the model weights on Hugging Face. Our code would be made available to foster future research from the community.
>
---
#### [replaced 007] Personalized Voice Synthesis through Human-in-the-Loop Coordinate Descent
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2408.17068v5](http://arxiv.org/pdf/2408.17068v5)**

> **作者:** Yusheng Tian; Junbin Liu; Tan Lee
>
> **备注:** work in progress
>
> **摘要:** This paper describes a human-in-the-loop approach to personalized voice synthesis in the absence of reference speech data from the target speaker. It is intended to help vocally disabled individuals restore their lost voices without requiring any prior recordings. The proposed approach leverages a learned speaker embedding space. Starting from an initial voice, users iteratively refine the speaker embedding parameters through a coordinate descent-like process, guided by auditory perception. By analyzing the latent space, it is noted that that the embedding parameters correspond to perceptual voice attributes, including pitch, vocal tension, brightness, and nasality, making the search process intuitive. Computer simulations and real-world user studies demonstrate that the proposed approach is effective in approximating target voices across a diverse range of test cases.
>
---
#### [replaced 008] LoopGen: Training-Free Loopable Music Generation
- **分类: cs.SD; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04466v3](http://arxiv.org/pdf/2504.04466v3)**

> **作者:** Davide Marincione; Giorgio Strano; Donato Crisostomi; Roberto Ribuoli; Emanuele Rodolà
>
> **摘要:** Loops--short audio segments designed for seamless repetition--are central to many music genres, particularly those rooted in dance and electronic styles. However, current generative music models struggle to produce truly loopable audio, as generating a short waveform alone does not guarantee a smooth transition from its endpoint back to its start, often resulting in audible discontinuities. We address this gap by modifying a non-autoregressive model (MAGNeT) to generate tokens in a circular pattern, letting the model attend to the beginning of the audio when creating its ending. This inference-only approach results in generations that are aware of future context and loop naturally, without the need for any additional training or data. We evaluate the consistency of loop transitions by computing token perplexity around the seam of the loop, observing a 55% improvement. Blind listening tests further confirm significant perceptual gains over baseline methods, improving mean ratings by 70%. Taken together, these results highlight the effectiveness of inference-only approaches in improving generative models and underscore the advantages of non-autoregressive methods for context-aware music generation.
>
---
#### [replaced 009] SepALM: Audio Language Models Are Error Correctors for Robust Speech Separation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03273v2](http://arxiv.org/pdf/2505.03273v2)**

> **作者:** Zhaoxi Mu; Xinyu Yang; Gang Wang
>
> **备注:** Appears in IJCAI 2025
>
> **摘要:** While contemporary speech separation technologies adeptly process lengthy mixed audio waveforms, they are frequently challenged by the intricacies of real-world environments, including noisy and reverberant settings, which can result in artifacts or distortions in the separated speech. To overcome these limitations, we introduce SepALM, a pioneering approach that employs audio language models (ALMs) to rectify and re-synthesize speech within the text domain following preliminary separation. SepALM comprises four core components: a separator, a corrector, a synthesizer, and an aligner. By integrating an ALM-based end-to-end error correction mechanism, we mitigate the risk of error accumulation and circumvent the optimization hurdles typically encountered in conventional methods that amalgamate automatic speech recognition (ASR) with large language models (LLMs). Additionally, we have developed Chain-of-Thought (CoT) prompting and knowledge distillation techniques to facilitate the reasoning and training processes of the ALM. Our experiments substantiate that SepALM not only elevates the precision of speech separation but also markedly bolsters adaptability in novel acoustic environments.
>
---
#### [replaced 010] DGSNA: prompt-based Dynamic Generative Scene-based Noise Addition method
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.12363v5](http://arxiv.org/pdf/2411.12363v5)**

> **作者:** Zihao Chen; Zhentao Lin; Bi Zeng; Linyi Huang; Zhi Li; Jia Cai
>
> **摘要:** To ensure the reliable operation of speech systems across diverse environments, noise addition methods have emerged as the prevailing solution. However, existing methods offer limited coverage of real-world noisy scenes and depend on pre-existing scene-based information and noise. This paper presents prompt-based Dynamic Generative Scene-based Noise Addition (DGSNA), a novel noise addition methodology that integrates Dynamic Generation of Scene-based Information (DGSI) with Scene-based Noise Addition for Speech (SNAS). This integration facilitates automated scene-based noise addition by transforming clean speech into various noise environments, thereby providing a more comprehensive and realistic simulation of diverse noise conditions. Experimental results demonstrate that DGSNA significantly enhances the robustness of speech recognition and keyword spotting models across various noise conditions, achieving a relative improvement of up to 11.21%. Furthermore, DGSNA can be effectively integrated with other noise addition methods to enhance performance. Our implementation and demonstrations are available at https://dgsna.github.io.
>
---
#### [replaced 011] Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.13772v2](http://arxiv.org/pdf/2501.13772v2)**

> **作者:** Hao Cheng; Erjia Xiao; Jing Shao; Yichi Wang; Le Yang; Chao Sheng; Philip Torr; Jindong Gu; Renjing Xu
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant security risks, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific Jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also a range of audio editing techniques. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs, establishing the most comprehensive audio jailbreak benchmark to date. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.
>
---
#### [replaced 012] GraphemeAug: A Systematic Approach to Synthesized Hard Negative Keyword Spotting Examples
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.14814v2](http://arxiv.org/pdf/2505.14814v2)**

> **作者:** Harry Zhang; Kurt Partridge; Pai Zhu; Neng Chen; Hyun Jin Park; Dhruuv Agarwal; Quan Wang
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Spoken Keyword Spotting (KWS) is the task of distinguishing between the presence and absence of a keyword in audio. The accuracy of a KWS model hinges on its ability to correctly classify examples close to the keyword and non-keyword boundary. These boundary examples are often scarce in training data, limiting model performance. In this paper, we propose a method to systematically generate adversarial examples close to the decision boundary by making insertion/deletion/substitution edits on the keyword's graphemes. We evaluate this technique on held-out data for a popular keyword and show that the technique improves AUC on a dataset of synthetic hard negatives by 61% while maintaining quality on positives and ambient negative audio data.
>
---
#### [replaced 013] "Alexa, can you forget me?" Machine Unlearning Benchmark in Spoken Language Understanding
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15700v2](http://arxiv.org/pdf/2505.15700v2)**

> **作者:** Alkis Koudounas; Claudio Savelli; Flavio Giobergia; Elena Baralis
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Machine unlearning, the process of efficiently removing specific information from machine learning models, is a growing area of interest for responsible AI. However, few studies have explored the effectiveness of unlearning methods on complex tasks, particularly speech-related ones. This paper introduces UnSLU-BENCH, the first benchmark for machine unlearning in spoken language understanding (SLU), focusing on four datasets spanning four languages. We address the unlearning of data from specific speakers as a way to evaluate the quality of potential "right to be forgotten" requests. We assess eight unlearning techniques and propose a novel metric to simultaneously better capture their efficacy, utility, and efficiency. UnSLU-BENCH sets a foundation for unlearning in SLU and reveals significant differences in the effectiveness and computational feasibility of various techniques.
>
---
#### [replaced 014] PITCH: AI-assisted Tagging of Deepfake Audio Calls using Challenge-Response
- **分类: cs.SD; cs.CR; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.18085v4](http://arxiv.org/pdf/2402.18085v4)**

> **作者:** Govind Mittal; Arthur Jakobsson; Kelly O. Marshall; Chinmay Hegde; Nasir Memon
>
> **备注:** To appear in ASIA CCS 2025. Human Instrument, Code and Dataset at https://govindm.me/pitch
>
> **摘要:** The rise of AI voice-cloning technology, particularly audio Real-time Deepfakes (RTDFs), has intensified social engineering attacks by enabling real-time voice impersonation that bypasses conventional enrollment-based authentication. This technology represents an existential threat to phone-based authentication systems, while total identity fraud losses reached $43 billion. Unlike traditional robocalls, these personalized AI-generated voice attacks target high-value accounts and circumvent existing defensive measures, creating an urgent cybersecurity challenge. To address this, we propose PITCH, a robust challenge-response method to detect and tag interactive deepfake audio calls. We developed a comprehensive taxonomy of audio challenges based on the human auditory system, linguistics, and environmental factors, yielding 20 prospective challenges. Testing against leading voice-cloning systems using a novel dataset (18,600 original and 1.6 million deepfake samples from 100 users), PITCH's challenges enhanced machine detection capabilities to 88.7% AUROC score, enabling us to identify 10 highly-effective challenges. For human evaluation, we filtered a challenging, balanced subset on which human evaluators independently achieved 72.6% accuracy, while machines scored 87.7%. Recognizing that call environments require human control, we developed a novel human-AI collaborative system that tags suspicious calls as "Deepfake-likely." Contrary to prior findings, we discovered that integrating human intuition with machine precision offers complementary advantages, giving users maximum control while boosting detection accuracy to 84.5%. This significant improvement situates PITCH's potential as an AI-assisted pre-screener for verifying calls, offering an adaptable approach to combat real-time voice-cloning attacks while maintaining human decision authority.
>
---
#### [replaced 015] TALKPLAY: Multimodal Music Recommendation with Large Language Models
- **分类: cs.IR; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.13713v4](http://arxiv.org/pdf/2502.13713v4)**

> **作者:** Seungheon Doh; Keunwoo Choi; Juhan Nam
>
> **摘要:** We present TALKPLAY, a novel multimodal music recommendation system that reformulates recommendation as a token generation problem using large language models (LLMs). By leveraging the instruction-following and natural language generation capabilities of LLMs, our system effectively recommends music from diverse user queries while generating contextually relevant responses. While pretrained LLMs are primarily designed for text modality, TALKPLAY extends their scope through two key innovations: a multimodal music tokenizer that encodes audio features, lyrics, metadata, semantic tags, and playlist co-occurrence signals; and a vocabulary expansion mechanism that enables unified processing and generation of both linguistic and music-relevant tokens. By integrating the recommendation system directly into the LLM architecture, TALKPLAY transforms conventional systems by: (1) unifying previous two-stage conversational recommendation systems (recommendation engines and dialogue managers) into a cohesive end-to-end system, (2) effectively utilizing long conversational context for recommendation while maintaining strong performance in extended multi-turn interactions, and (3) generating natural language responses for seamless user interaction. Our qualitative and quantitative evaluation demonstrates that TALKPLAY significantly outperforms unimodal approaches based solely on text or listening history in both recommendation performance and conversational naturalness.
>
---
#### [replaced 016] Fast Differentiable Modal Simulation of Non-linear Strings, Membranes, and Plates
- **分类: cs.SD; cs.LG; eess.AS; physics.comp-ph**

- **链接: [http://arxiv.org/pdf/2505.05940v2](http://arxiv.org/pdf/2505.05940v2)**

> **作者:** Rodrigo Diaz; Mark Sandler
>
> **备注:** accepted to DAFx 2025
>
> **摘要:** Modal methods for simulating vibrations of strings, membranes, and plates are widely used in acoustics and physically informed audio synthesis. However, traditional implementations, particularly for non-linear models like the von K\'arm\'an plate, are computationally demanding and lack differentiability, limiting inverse modelling and real-time applications. We introduce a fast, differentiable, GPU-accelerated modal framework built with the JAX library, providing efficient simulations and enabling gradient-based inverse modelling. Benchmarks show that our approach significantly outperforms CPU and GPU-based implementations, particularly for simulations with many modes. Inverse modelling experiments demonstrate that our approach can recover physical parameters, including tension, stiffness, and geometry, from both synthetic and experimental data. Although fitting physical parameters is more sensitive to initialisation compared to other methods, it provides greater interpretability and more compact parameterisation. The code is released as open source to support future research and applications in differentiable physical modelling and sound synthesis.
>
---
#### [replaced 017] Bridging The Multi-Modality Gaps of Audio, Visual and Linguistic for Speech Enhancement
- **分类: cs.SD; cs.LG; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.13375v2](http://arxiv.org/pdf/2501.13375v2)**

> **作者:** Meng-Ping Lin; Jen-Cheng Hou; Chia-Wei Chen; Shao-Yi Chien; Jun-Cheng Chen; Xugang Lu; Yu Tsao
>
> **摘要:** Speech enhancement (SE) aims to improve the quality and intelligibility of speech in noisy environments. Recent studies have shown that incorporating visual cues in audio signal processing can enhance SE performance. Given that human speech communication naturally involves audio, visual, and linguistic modalities, it is reasonable to expect additional improvements by integrating linguistic information. However, effectively bridging these modality gaps, particularly during knowledge transfer remains a significant challenge. In this paper, we propose a novel multi-modal learning framework, termed DLAV-SE, which leverages a diffusion-based model integrating audio, visual, and linguistic information for audio-visual speech enhancement (AVSE). Within this framework, the linguistic modality is modeled using a pretrained language model (PLM), which transfers linguistic knowledge to the audio-visual domain through a cross-modal knowledge transfer (CMKT) mechanism during training. After training, the PLM is no longer required at inference, as its knowledge is embedded into the AVSE model through the CMKT process. We conduct a series of SE experiments to evaluate the effectiveness of our approach. Results show that the proposed DLAV-SE system significantly improves speech quality and reduces generative artifacts, such as phonetic confusion, compared to state-of-the-art (SOTA) methods. Furthermore, visualization analyses confirm that the CMKT method enhances the generation quality of the AVSE outputs. These findings highlight both the promise of diffusion-based methods for advancing AVSE and the value of incorporating linguistic information to further improve system performance.
>
---
#### [replaced 018] DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2502.03930v3](http://arxiv.org/pdf/2502.03930v3)**

> **作者:** Dongya Jia; Zhuo Chen; Jiawei Chen; Chenpeng Du; Jian Wu; Jian Cong; Xiaobin Zhuang; Chumin Li; Zhen Wei; Yuping Wang; Yuxuan Wang
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Several recent studies have attempted to autoregressively generate continuous speech representations without discrete speech tokens by combining diffusion and autoregressive models, yet they often face challenges with excessive computational loads or suboptimal outcomes. In this work, we propose Diffusion Transformer Autoregressive Modeling (DiTAR), a patch-based autoregressive framework combining a language model with a diffusion transformer. This approach significantly enhances the efficacy of autoregressive models for continuous tokens and reduces computational demands. DiTAR utilizes a divide-and-conquer strategy for patch generation, where the language model processes aggregated patch embeddings and the diffusion transformer subsequently generates the next patch based on the output of the language model. For inference, we propose defining temperature as the time point of introducing noise during the reverse diffusion ODE to balance diversity and determinism. We also show in the extensive scaling analysis that DiTAR has superb scalability. In zero-shot speech generation, DiTAR achieves state-of-the-art performance in robustness, speaker similarity, and naturalness.
>
---
#### [replaced 019] On the Relevance of Clinical Assessment Tasks for the Automatic Detection of Parkinson's Disease Medication State from Speech
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.15378v2](http://arxiv.org/pdf/2505.15378v2)**

> **作者:** David Gimeno-Gómez; Rubén Solera-Ureña; Anna Pompili; Carlos-D. Martínez-Hinarejos; Rita Cardoso; Isabel Guimarães; Joaquim J. Ferreira; Alberto Abad
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** The automatic identification of medication states of Parkinson's disease (PD) patients can assist clinicians in monitoring and scheduling personalized treatments, as well as studying the effects of medication in alleviating the motor symptoms that characterize the disease. This paper explores speech as a non-invasive and accessible biomarker for identifying PD medication states, introducing a novel approach that addresses this task from a speaker-independent perspective. While traditional machine learning models achieve competitive results, self-supervised speech representations prove essential for optimal performance, significantly surpassing knowledge-based acoustic descriptors. Experiments across diverse speech assessment tasks highlight the relevance of prosody and continuous speech in distinguishing medication states, reaching an F1-score of 88.2%. These findings may streamline clinicians' work and reduce patient effort in voice recordings.
>
---
#### [replaced 020] ALMA: a mathematics-driven approach for determining tuning parameters in generalized LASSO problems, with applications to MRI
- **分类: eess.IV; cs.CV; eess.SP; physics.med-ph; 92C55, 62J07, 65K10; I.4.2; I.4.5; J.2; J.3**

- **链接: [http://arxiv.org/pdf/2406.19239v2](http://arxiv.org/pdf/2406.19239v2)**

> **作者:** Gianluca Giacchi; Isidoros Iakovidis; Bastien Milani; Micah Murray; Benedetta Franceschiello
>
> **备注:** Modified pictures, authors and fixed some typo
>
> **摘要:** Magnetic Resonance Imaging (MRI) is a powerful technique employed for non-invasive in vivo visualization of internal structures. Sparsity is often deployed to accelerate the signal acquisition or overcome the presence of motion artifacts, improving the quality of image reconstruction. Image reconstruction algorithms use TV-regularized LASSO (Total Variation-regularized LASSO) to retrieve the missing information of undersampled signals, by cleaning the data of noise and while optimizing sparsity. A tuning parameter moderates the balance between these two aspects; its choice affecting the quality of the reconstructions. Currently, there is a lack of general deterministic techniques to choose these parameters, which are oftentimes manually selected and thus hinder the reliability of the reconstructions. Here, we present ALMA (Algorithm for Lagrange Multipliers Approximation), an iterative mathematics-inspired technique that computes tuning parameters for generalized LASSO problems during MRI reconstruction. We analyze quantitatively the performance of these parameters for imaging reconstructions via TV-LASSO in an MRI context on phantoms. Although our study concentrates on TV-LASSO, the techniques developed here hold significant promise for a wide array of applications. ALMA is not only adaptable to more generalized LASSO problems but is also robust to accommodate other forms of regularization beyond total variation. Moreover, it extends effectively to handle non-Cartesian sampling trajectories, broadening its utility in complex data reconstruction scenarios. More generally, ALMA provides a powerful tool for numerically solving constrained optimization problems across various disciplines, offering a versatile and impactful solution for advanced computational challenges.
>
---
#### [replaced 021] The Impact of LoRA Adapters for LLMs on Clinical NLP Classification Under Data Limitations
- **分类: cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2407.19299v2](http://arxiv.org/pdf/2407.19299v2)**

> **作者:** Thanh-Dung Le; Ti Ti Nguyen; Vu Nguyen Ha; Symeon Chatzinotas; Philippe Jouvet; Rita Noumeir
>
> **备注:** Under revisions
>
> **摘要:** Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to the domain gap and limited data availability. This study investigates the effectiveness of various adapter techniques, equivalent to Low-Rank Adaptation (LoRA), for fine-tuning LLMs in a resource-constrained hospital environment. We experimented with four structures-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN)-as final layers for clinical notes classification. We fine-tuned biomedical pre-trained models, including CamemBERT-bio, AliBERT, and DrBERT, alongside two Transformer-based models. Our extensive experimental results indicate that i) employing adapter structures does not yield significant improvements in fine-tuning biomedical pre-trained LLMs, and ii) simpler Transformer-based models, trained from scratch, perform better under resource constraints. Among the adapter structures, GRN demonstrated superior performance with accuracy, precision, recall, and an F1 score of 0.88. Moreover, the total training time for LLMs exceeded 1000 hours, compared to under 6 hours for simpler transformer-based models, highlighting that LLMs are more suitable for environments with extensive computational resources and larger datasets. Consequently, this study demonstrates that simpler Transformer-based models can be effectively trained from scratch, providing a viable solution for clinical NLP tasks in low-resource environments with limited data availability. By identifying the GRN as the most effective adapter structure, we offer a practical approach to enhance clinical note classification without requiring extensive computational resources.
>
---
#### [replaced 022] Semantic-Aware Interpretable Multimodal Music Auto-Tagging
- **分类: cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.17233v2](http://arxiv.org/pdf/2505.17233v2)**

> **作者:** Andreas Patakis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Music auto-tagging is essential for organizing and discovering music in extensive digital libraries. While foundation models achieve exceptional performance in this domain, their outputs often lack interpretability, limiting trust and usability for researchers and end-users alike. In this work, we present an interpretable framework for music auto-tagging that leverages groups of musically meaningful multimodal features, derived from signal processing, deep learning, ontology engineering, and natural language processing. To enhance interpretability, we cluster features semantically and employ an expectation maximization algorithm, assigning distinct weights to each group based on its contribution to the tagging process. Our method achieves competitive tagging performance while offering a deeper understanding of the decision-making process, paving the way for more transparent and user-centric music tagging systems.
>
---
#### [replaced 023] DualTalk: Dual-Speaker Interaction for 3D Talking Head Conversations
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.18096v2](http://arxiv.org/pdf/2505.18096v2)**

> **作者:** Ziqiao Peng; Yanbo Fan; Haoyu Wu; Xuan Wang; Hongyan Liu; Jun He; Zhaoxin Fan
>
> **备注:** Accepted by CVPR 2025
>
> **摘要:** In face-to-face conversations, individuals need to switch between speaking and listening roles seamlessly. Existing 3D talking head generation models focus solely on speaking or listening, neglecting the natural dynamics of interactive conversation, which leads to unnatural interactions and awkward transitions. To address this issue, we propose a new task -- multi-round dual-speaker interaction for 3D talking head generation -- which requires models to handle and generate both speaking and listening behaviors in continuous conversation. To solve this task, we introduce DualTalk, a novel unified framework that integrates the dynamic behaviors of speakers and listeners to simulate realistic and coherent dialogue interactions. This framework not only synthesizes lifelike talking heads when speaking but also generates continuous and vivid non-verbal feedback when listening, effectively capturing the interplay between the roles. We also create a new dataset featuring 50 hours of multi-round conversations with over 1,000 characters, where participants continuously switch between speaking and listening roles. Extensive experiments demonstrate that our method significantly enhances the naturalness and expressiveness of 3D talking heads in dual-speaker conversations. We recommend watching the supplementary video: https://ziqiaopeng.github.io/dualtalk.
>
---
#### [replaced 024] vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2409.01995v4](http://arxiv.org/pdf/2409.01995v4)**

> **作者:** Yiwei Guo; Zhihan Li; Junjie Li; Chenpeng Du; Hankun Wang; Shuai Wang; Xie Chen; Kai Yu
>
> **备注:** 5 pages, 3 figures, 2 tables. Demo page: https://cantabile-kwok.github.io/vec2wav2/
>
> **摘要:** We propose a new speech discrete token vocoder, vec2wav 2.0, which advances voice conversion (VC). We use discrete tokens from speech self-supervised models as the content features of source speech, and treat VC as a prompted vocoding task. To amend the loss of speaker timbre in the content tokens, vec2wav 2.0 utilizes the WavLM features to provide strong timbre-dependent information. A novel adaptive Snake activation function is proposed to better incorporate timbre into the waveform reconstruction process. In this way, vec2wav 2.0 learns to alter the speaker timbre appropriately given different reference prompts. Also, no supervised data is required for vec2wav 2.0 to be effectively trained. Experimental results demonstrate that vec2wav 2.0 outperforms all other baselines to a considerable margin in terms of audio quality and speaker similarity in any-to-any VC. Ablation studies verify the effects made by the proposed techniques. Moreover, vec2wav 2.0 achieves competitive cross-lingual VC even only trained on monolingual corpus. Thus, vec2wav 2.0 shows timbre can potentially be manipulated only by speech token vocoders, pushing the frontiers of VC and speech synthesis.
>
---
