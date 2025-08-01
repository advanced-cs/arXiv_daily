# 音频 cs.SD;  eess.SP

- **最新发布 14 篇**

- **更新 6 篇**

## 最新发布

#### [new 001] Enforcing Speech Content Privacy in Environmental Sound Recordings using Segment-wise Waveform Reversal
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音隐私保护任务，旨在去除环境录音中的可识别语音，同时保留音频质量和场景完整性。通过反转波形段和语音活动检测实现隐私保护。**

- **链接: [http://arxiv.org/pdf/2507.08412v1](http://arxiv.org/pdf/2507.08412v1)**

> **作者:** Modan Tailleur; Mathieu Lagrange; Pierre Aumond; Vincent Tourre
>
> **摘要:** Environmental sound recordings often contain intelligible speech, raising privacy concerns that limit analysis, sharing and reuse of data. In this paper, we introduce a method that renders speech unintelligible while preserving both the integrity of the acoustic scene, and the overall audio quality. Our approach involves reversing waveform segments to distort speech content. This process is enhanced through a voice activity detection and speech separation pipeline, which allows for more precise targeting of speech. In order to demonstrate the effectivness of the proposed approach, we consider a three-part evaluation protocol that assesses: 1) speech intelligibility using Word Error Rate (WER), 2) sound sources detectability using Sound source Classification Accuracy-Drop (SCAD) from a widely used pre-trained model, and 3) audio quality using the Fr\'echet Audio Distance (FAD), computed with our reference dataset that contains unaltered speech. Experiments on this simulated evaluation dataset, which consists of linear mixtures of speech and environmental sound scenes, show that our method achieves satisfactory speech intelligibility reduction (97.9% WER), minimal degradation of the sound sources detectability (2.7% SCAD), and high perceptual quality (FAD of 1.40). An ablation study further highlights the contribution of each component of the pipeline. We also show that incorporating random splicing to our speech content privacy enforcement method can enhance the algorithm's robustness to attempt to recover the clean speech, at a slight cost of audio quality.
>
---
#### [new 002] Active Learning for Text-to-Speech Synthesis with Informative Sample Collection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在解决大规模数据集构建中的存储与效率问题。通过主动学习方法，迭代收集更具信息量的数据，提升合成质量。**

- **链接: [http://arxiv.org/pdf/2507.08319v1](http://arxiv.org/pdf/2507.08319v1)**

> **作者:** Kentaro Seki; Shinnosuke Takamichi; Takaaki Saeki; Hiroshi Saruwatari
>
> **摘要:** The construction of high-quality datasets is a cornerstone of modern text-to-speech (TTS) systems. However, the increasing scale of available data poses significant challenges, including storage constraints. To address these issues, we propose a TTS corpus construction method based on active learning. Unlike traditional feed-forward and model-agnostic corpus construction approaches, our method iteratively alternates between data collection and model training, thereby focusing on acquiring data that is more informative for model improvement. This approach enables the construction of a data-efficient corpus. Experimental results demonstrate that the corpus constructed using our method enables higher-quality speech synthesis than corpora of the same size.
>
---
#### [new 003] Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音深度伪造检测任务，旨在提高检测的准确性和可解释性。通过音素级分析构建说话人特征，实现更细粒度的伪造识别。**

- **链接: [http://arxiv.org/pdf/2507.08626v1](http://arxiv.org/pdf/2507.08626v1)**

> **作者:** Davide Salvi; Viola Negroni; Sara Mandelli; Paolo Bestagini; Stefano Tubaro
>
> **备注:** Accepted at ICCV Workshop - Authenticity & Provenance in the age of Generative AI
>
> **摘要:** Recent advances in generative AI have made the creation of speech deepfakes widely accessible, posing serious challenges to digital trust. To counter this, various speech deepfake detection strategies have been proposed, including Person-of-Interest (POI) approaches, which focus on identifying impersonations of specific individuals by modeling and analyzing their unique vocal traits. Despite their excellent performance, the existing methods offer limited granularity and lack interpretability. In this work, we propose a POI-based speech deepfake detection method that operates at the phoneme level. Our approach decomposes reference audio into phonemes to construct a detailed speaker profile. In inference, phonemes from a test sample are individually compared against this profile, enabling fine-grained detection of synthetic artifacts. The proposed method achieves comparable accuracy to traditional approaches while offering superior robustness and interpretability, key aspects in multimedia forensics. By focusing on phoneme analysis, this work explores a novel direction for explainable, speaker-centric deepfake detection.
>
---
#### [new 004] Distilling Spectrograms into Tokens: Fast and Lightweight Bioacoustic Classification for BirdCLEF+ 2025
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.08236v1](http://arxiv.org/pdf/2507.08236v1)**

> **作者:** Anthony Miyaguchi; Murilo Gustineli; Adrian Cheung
>
> **备注:** Working note submitted to CLEF 2025 under the LifeCLEF lab
>
> **摘要:** The BirdCLEF+ 2025 challenge requires classifying 206 species, including birds, mammals, insects, and amphibians, from soundscape recordings under a strict 90-minute CPU-only inference deadline, making many state-of-the-art deep learning approaches impractical. To address this constraint, the DS@GT BirdCLEF team explored two strategies. First, we establish competitive baselines by optimizing pre-trained models from the Bioacoustics Model Zoo for CPU inference. Using TFLite, we achieved a nearly 10x inference speedup for the Perch model, enabling it to run in approximately 16 minutes and achieve a final ROC-AUC score of 0.729 on the public leaderboard post-competition and 0.711 on the private leaderboard. The best model from the zoo was BirdSetEfficientNetB1, with a public score of 0.810 and a private score of 0.778. Second, we introduce a novel, lightweight pipeline named Spectrogram Token Skip-Gram (STSG) that treats bioacoustics as a sequence modeling task. This method converts audio into discrete "spectrogram tokens" by clustering Mel-spectrograms using Faiss K-means and then learns high-quality contextual embeddings for these tokens in an unsupervised manner with a Word2Vec skip-gram model. For classification, embeddings within a 5-second window are averaged and passed to a linear model. With a projected inference time of 6 minutes for a 700-minute test set, the STSG approach achieved a final ROC-AUC public score of 0.559 and a private score of 0.520, demonstrating the viability of fast tokenization approaches with static embeddings for bioacoustic classification. Supporting code for this paper can be found at https://github.com/dsgt-arc/birdclef-2025.
>
---
#### [new 005] On Barriers to Archival Audio Processing
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音处理任务，探讨档案音频中语言识别和说话人识别的挑战，尤其关注多语言和跨年龄录音的影响。研究测试了现有方法的鲁棒性并指出说话人嵌入的脆弱性。**

- **链接: [http://arxiv.org/pdf/2507.08768v1](http://arxiv.org/pdf/2507.08768v1)**

> **作者:** Peter Sullivan; Muhammad Abdul-Mageed
>
> **备注:** Update with Acknowledgements of ICNSLP 2025 paper
>
> **摘要:** In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing.
>
---
#### [new 006] FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation
- **分类: cs.SD; cs.AI; cs.MM; eess.AS**

- **简介: 该论文属于文本到音频生成任务，解决长格式文本精确时间控制生成问题。提出FreeAudio框架，无需训练即可实现高精度时间控制的音频合成。**

- **链接: [http://arxiv.org/pdf/2507.08557v1](http://arxiv.org/pdf/2507.08557v1)**

> **作者:** Yuxuan Jiang; Zehua Chen; Zeqian Ju; Chang Li; Weibei Dou; Jun Zhu
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** Text-to-audio (T2A) generation has achieved promising results with the recent advances in generative models. However, because of the limited quality and quantity of temporally-aligned audio-text pairs, existing T2A methods struggle to handle the complex text prompts that contain precise timing control, e.g., "owl hooted at 2.4s-5.2s". Recent works have explored data augmentation techniques or introduced timing conditions as model inputs to enable timing-conditioned 10-second T2A generation, while their synthesis quality is still limited. In this work, we propose a novel training-free timing-controlled T2A framework, FreeAudio, making the first attempt to enable timing-controlled long-form T2A generation, e.g., "owl hooted at 2.4s-5.2s and crickets chirping at 0s-24s". Specifically, we first employ an LLM to plan non-overlapping time windows and recaption each with a refined natural language description, based on the input text and timing prompts. Then we introduce: 1) Decoupling and Aggregating Attention Control for precise timing control; 2) Contextual Latent Composition for local smoothness and Reference Guidance for global consistency. Extensive experiments show that: 1) FreeAudio achieves state-of-the-art timing-conditioned T2A synthesis quality among training-free methods and is comparable to leading training-based methods; 2) FreeAudio demonstrates comparable long-form generation quality with training-based Stable Audio and paves the way for timing-controlled long-form T2A synthesis. Demo samples are available at: https://freeaudio.github.io/FreeAudio/
>
---
#### [new 007] Audio Inpanting using Discrete Diffusion Model
- **分类: cs.SD; cs.AI; cs.IT; cs.LG; eess.AS; math.IT**

- **简介: 该论文属于音频修复任务，解决长间隙音频重建问题。提出基于离散扩散模型的方法，在离散潜在空间中进行生成，提升修复质量与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.08333v1](http://arxiv.org/pdf/2507.08333v1)**

> **作者:** Tali Dror; Iftach Shoham; Moshe Buchris; Oren Gal; Haim Permuter; Gilad Katz; Eliya Nachmani
>
> **摘要:** Audio inpainting refers to the task of reconstructing missing segments in corrupted audio recordings. While prior approaches-including waveform and spectrogram-based diffusion models-have shown promising results for short gaps, they often degrade in quality when gaps exceed 100 milliseconds (ms). In this work, we introduce a novel inpainting method based on discrete diffusion modeling, which operates over tokenized audio representations produced by a pre-trained audio tokenizer. Our approach models the generative process directly in the discrete latent space, enabling stable and semantically coherent reconstruction of missing audio. We evaluate the method on the MusicNet dataset using both objective and perceptual metrics across gap durations up to 300 ms. We further evaluated our approach on the MTG dataset, extending the gap duration to 500 ms. Experimental results demonstrate that our method achieves competitive or superior performance compared to existing baselines, particularly for longer gaps, offering a robust solution for restoring degraded musical recordings. Audio examples of our proposed method can be found at https://iftach21.github.io/
>
---
#### [new 008] MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐表演合成任务，旨在提升MIDI到音频的转换质量。通过引入MIDI-VALLE模型，结合音频和MIDI作为离散标记，增强模型泛化能力，显著降低音频距离并提高合成质量。**

- **链接: [http://arxiv.org/pdf/2507.08530v1](http://arxiv.org/pdf/2507.08530v1)**

> **作者:** Jingjing Tang; Xin Wang; Zhe Zhang; Junichi Yamagishi; Geraint Wiggins; George Fazekas
>
> **备注:** Accepted by ISMIR 2025
>
> **摘要:** Generating expressive audio performances from music scores requires models to capture both instrument acoustics and human interpretation. Traditional music performance synthesis pipelines follow a two-stage approach, first generating expressive performance MIDI from a score, then synthesising the MIDI into audio. However, the synthesis models often struggle to generalise across diverse MIDI sources, musical styles, and recording environments. To address these challenges, we propose MIDI-VALLE, a neural codec language model adapted from the VALLE framework, which was originally designed for zero-shot personalised text-to-speech (TTS) synthesis. For performance MIDI-to-audio synthesis, we improve the architecture to condition on a reference audio performance and its corresponding MIDI. Unlike previous TTS-based systems that rely on piano rolls, MIDI-VALLE encodes both MIDI and audio as discrete tokens, facilitating a more consistent and robust modelling of piano performances. Furthermore, the model's generalisation ability is enhanced by training on an extensive and diverse piano performance dataset. Evaluation results show that MIDI-VALLE significantly outperforms a state-of-the-art baseline, achieving over 75% lower Frechet Audio Distance on the ATEPP and Maestro datasets. In the listening test, MIDI-VALLE received 202 votes compared to 58 for the baseline, demonstrating improved synthesis quality and generalisation across diverse performance MIDI inputs.
>
---
#### [new 009] Modèle physique variationnel pour l'estimation de réponses impulsionnelles de salles
- **分类: cs.SD; eess.AS; eess.SP; physics.class-ph**

- **简介: 该论文属于语音去混响任务，旨在提高自动语音识别效果。通过结合统计与物理模型，提出一种新方法估计房间脉冲响应，优于传统解卷积方法。**

- **链接: [http://arxiv.org/pdf/2507.08051v1](http://arxiv.org/pdf/2507.08051v1)**

> **作者:** Louis Lalay; Mathieu Fontaine; Roland Badeau
>
> **备注:** in French language. GRETSI, Aug 2025, Strasbourg (67000), France
>
> **摘要:** Room impulse response estimation is essential for tasks like speech dereverberation, which improves automatic speech recognition. Most existing methods rely on either statistical signal processing or deep neural networks designed to replicate signal processing principles. However, combining statistical and physical modeling for RIR estimation remains largely unexplored. This paper proposes a novel approach integrating both aspects through a theoretically grounded model. The RIR is decomposed into interpretable parameters: white Gaussian noise filtered by a frequency-dependent exponential decay (e.g. modeling wall absorption) and an autoregressive filter (e.g. modeling microphone response). A variational free-energy cost function enables practical parameter estimation. As a proof of concept, we show that given dry and reverberant speech signals, the proposed method outperforms classical deconvolution in noisy environments, as validated by objective metrics.
>
---
#### [new 010] Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频理解任务，旨在提升语音、声音和音乐的推理与理解能力。工作包括设计多模态音频编码器、支持长音频处理及对话交互，并在多个基准上取得最佳成绩。**

- **链接: [http://arxiv.org/pdf/2507.08128v1](http://arxiv.org/pdf/2507.08128v1)**

> **作者:** Arushi Goel; Sreyan Ghosh; Jaehyeon Kim; Sonal Kumar; Zhifeng Kong; Sang-gil Lee; Chao-Han Huck Yang; Ramani Duraiswami; Dinesh Manocha; Rafael Valle; Bryan Catanzaro
>
> **备注:** Code, Datasets and Models: https://research.nvidia.com/labs/adlr/AF3/
>
> **摘要:** We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
>
---
#### [new 011] DARAS: Dynamic Audio-Room Acoustic Synthesis for Blind Room Impulse Response Estimation
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.08135v1](http://arxiv.org/pdf/2507.08135v1)**

> **作者:** Chunxi Wang; Maoshen Jia; Wenyu Jin
>
> **备注:** 14 pages, 9 figures, submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing
>
> **摘要:** Room Impulse Responses (RIRs) accurately characterize acoustic properties of indoor environments and play a crucial role in applications such as speech enhancement, speech recognition, and audio rendering in augmented reality (AR) and virtual reality (VR). Existing blind estimation methods struggle to achieve practical accuracy. To overcome this challenge, we propose the dynamic audio-room acoustic synthesis (DARAS) model, a novel deep learning framework that is explicitly designed for blind RIR estimation from monaural reverberant speech signals. First, a dedicated deep audio encoder effectively extracts relevant nonlinear latent space features. Second, the Mamba-based self-supervised blind room parameter estimation (MASS-BRPE) module, utilizing the efficient Mamba state space model (SSM), accurately estimates key room acoustic parameters and features. Third, the system incorporates a hybrid-path cross-attention feature fusion module, enhancing deep integration between audio and room acoustic features. Finally, our proposed dynamic acoustic tuning (DAT) decoder adaptively segments early reflections and late reverberation to improve the realism of synthesized RIRs. Experimental results, including a MUSHRA-based subjective listening study, demonstrate that DARAS substantially outperforms existing baseline models, providing a robust and effective solution for practical blind RIR estimation in real-world acoustic environments.
>
---
#### [new 012] RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于文本转语音任务，旨在提升语音控制的准确性和稳定性。通过重复微调和主成分分析，发现并引入新特征以增强模型可控性。**

- **链接: [http://arxiv.org/pdf/2507.08012v1](http://arxiv.org/pdf/2507.08012v1)**

> **作者:** Atli Sigurgeirsson; Simon King
>
> **摘要:** A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics. We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model.
>
---
#### [new 013] RawTFNet: A Lightweight CNN Architecture for Speech Anti-spoofing
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音反欺骗任务，旨在提升自动说话人验证系统的安全性。针对现有模型计算量大的问题，提出轻量级CNN模型RawTFNet，有效捕捉合成语音细节，性能接近先进模型但资源消耗更低。**

- **链接: [http://arxiv.org/pdf/2507.08227v1](http://arxiv.org/pdf/2507.08227v1)**

> **作者:** Yang Xiao; Ting Dang; Rohan Kumar Das
>
> **备注:** Submitted to APSIPA ASC 2025
>
> **摘要:** Automatic speaker verification (ASV) systems are often affected by spoofing attacks. Recent transformer-based models have improved anti-spoofing performance by learning strong feature representations. However, these models usually need high computing power. To address this, we introduce RawTFNet, a lightweight CNN model designed for audio signals. The RawTFNet separates feature processing along time and frequency dimensions, which helps to capture the fine-grained details of synthetic speech. We tested RawTFNet on the ASVspoof 2021 LA and DF evaluation datasets. The results show that RawTFNet reaches comparable performance to that of the state-of-the-art models, while also using fewer computing resources. The code and models will be made publicly available.
>
---
#### [new 014] Unlocking Speech Instruction Data Potential with Query Rewriting
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音指令数据构建任务，解决数据不足与质量低的问题。通过查询重写和多模型融合，提升语音合成数据质量，提高数据可用性。**

- **链接: [http://arxiv.org/pdf/2507.08603v1](http://arxiv.org/pdf/2507.08603v1)**

> **作者:** Yonghua Hei; Yibo Yan; Shuliang Liu; Huiyu Zhou; Linfeng Zhang; Xuming Hu
>
> **备注:** ACL 2025 Findings
>
> **摘要:** End-to-end Large Speech Language Models~(\textbf{LSLMs}) demonstrate strong potential in response latency and speech comprehension capabilities, showcasing general intelligence across speech understanding tasks. However, the ability to follow speech instructions has not been fully realized due to the lack of datasets and heavily biased training tasks. Leveraging the rich ASR datasets, previous approaches have used Large Language Models~(\textbf{LLMs}) to continue the linguistic information of speech to construct speech instruction datasets. Yet, due to the gap between LLM-generated results and real human responses, the continuation methods further amplify these shortcomings. Given the high costs of collecting and annotating speech instruction datasets by humans, using speech synthesis to construct large-scale speech instruction datasets has become a balanced and robust alternative. Although modern Text-To-Speech~(\textbf{TTS}) models have achieved near-human-level synthesis quality, it is challenging to appropriately convert out-of-distribution text instruction to speech due to the limitations of the training data distribution in TTS models. To address this issue, we propose a query rewriting framework with multi-LLM knowledge fusion, employing multiple agents to annotate and validate the synthesized speech, making it possible to construct high-quality speech instruction datasets without relying on human annotation. Experiments show that this method can transform text instructions into distributions more suitable for TTS models for speech synthesis through zero-shot rewriting, increasing data usability from 72\% to 93\%. It also demonstrates unique advantages in rewriting tasks that require complex knowledge and context-related abilities.
>
---
## 更新

#### [replaced 001] Token Communications: A Unified Framework for Cross-modal Context-aware Semantic Communications
- **分类: cs.IT; cs.CV; cs.MM; eess.SP; math.IT**

- **链接: [http://arxiv.org/pdf/2502.12096v3](http://arxiv.org/pdf/2502.12096v3)**

> **作者:** Li Qiao; Mahdi Boloursaz Mashhadi; Zhen Gao; Rahim Tafazolli; Mehdi Bennis; Dusit Niyato
>
> **备注:** Accepted at IEEE Wireless Communications Magazine
>
> **摘要:** In this paper, we introduce token communications (TokCom), a large model-driven framework to leverage cross-modal context information in generative semantic communications (GenSC). TokCom is a new paradigm, motivated by the recent success of generative foundation models and multimodal large language models (GFM/MLLMs), where the communication units are tokens, enabling efficient transformer-based token processing at the transmitter and receiver. In this paper, we introduce the potential opportunities and challenges of leveraging context in GenSC, explore how to integrate GFM/MLLMs-based token processing into semantic communication systems to leverage cross-modal context effectively at affordable complexity, present the key principles for efficient TokCom at various layers in future wireless networks. In a typical image semantic communication setup, we demonstrate a significant improvement of the bandwidth efficiency, achieved by TokCom by leveraging the context information among tokens. Finally, the potential research directions are identified to facilitate adoption of TokCom in future wireless networks.
>
---
#### [replaced 002] End-to-end multi-channel speaker extraction and binaural speech synthesis
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2410.05739v2](http://arxiv.org/pdf/2410.05739v2)**

> **作者:** Cheng Chi; Xiaoyu Li; Yuxuan Ke; Qunping Ni; Yao Ge; Xiaodong Li; Chengshi Zheng
>
> **摘要:** Speech clarity and spatial audio immersion are the two most critical factors in enhancing remote conferencing experiences. Existing methods are often limited: either due to the lack of spatial information when using only one microphone, or because their performance is highly dependent on the accuracy of direction-of-arrival estimation when using microphone array. To overcome this issue, we introduce an end-to-end deep learning framework that has the capacity of mapping multi-channel noisy and reverberant signals to clean and spatialized binaural speech directly. This framework unifies source extraction, noise suppression, and binaural rendering into one network. In this framework, a novel magnitude-weighted interaural level difference loss function is proposed that aims to improve the accuracy of spatial rendering. Extensive evaluations show that our method outperforms established baselines in terms of both speech quality and spatial fidelity.
>
---
#### [replaced 003] MuCodec: Ultra Low-Bitrate Music Codec
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.13216v3](http://arxiv.org/pdf/2409.13216v3)**

> **作者:** Yaoxun Xu; Hangting Chen; Jianwei Yu; Wei Tan; Rongzhi Gu; Shun Lei; Zhiwei Lin; Zhiyong Wu
>
> **摘要:** Music codecs are a vital aspect of audio codec research, and ultra low-bitrate compression holds significant importance for music transmission and generation. Due to the complexity of music backgrounds and the richness of vocals, solely relying on modeling semantic or acoustic information cannot effectively reconstruct music with both vocals and backgrounds. To address this issue, we propose MuCodec, specifically targeting music compression and reconstruction tasks at ultra low bitrates. MuCodec employs MuEncoder to extract both acoustic and semantic features, discretizes them with RVQ, and obtains Mel-VAE features via flow-matching. The music is then reconstructed using a pre-trained MEL-VAE decoder and HiFi-GAN. MuCodec can reconstruct high-fidelity music at ultra low (0.35kbps) or high bitrates (1.35kbps), achieving the best results to date in both subjective and objective metrics. Code and Demo: https://xuyaoxun.github.io/MuCodec_demo/.
>
---
#### [replaced 004] LISTEN: Lightweight Industrial Sound-representable Transformer for Edge Notification
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.07879v2](http://arxiv.org/pdf/2507.07879v2)**

> **作者:** Changheon Han; Yun Seok Kang; Yuseop Sim; Hyung Wook Park; Martin Byung-Guk Jun
>
> **摘要:** Deep learning-based machine listening is broadening the scope of industrial acoustic analysis for applications like anomaly detection and predictive maintenance, thereby improving manufacturing efficiency and reliability. Nevertheless, its reliance on large, task-specific annotated datasets for every new task limits widespread implementation on shop floors. While emerging sound foundation models aim to alleviate data dependency, they are too large and computationally expensive, requiring cloud infrastructure or high-end hardware that is impractical for on-site, real-time deployment. We address this gap with LISTEN (Lightweight Industrial Sound-representable Transformer for Edge Notification), a kilobyte-sized industrial sound foundation model. Using knowledge distillation, LISTEN runs in real-time on low-cost edge devices. On benchmark downstream tasks, it performs nearly identically to its much larger parent model, even when fine-tuned with minimal datasets and training resource. Beyond the model itself, we demonstrate its real-world utility by integrating LISTEN into a complete machine monitoring framework on an edge device with an Industrial Internet of Things (IIoT) sensor and system, validating its performance and generalization capabilities on a live manufacturing shop floor.
>
---
#### [replaced 005] Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
- **分类: cs.CY; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08846v2](http://arxiv.org/pdf/2506.08846v2)**

> **作者:** Katelyn Xiaoying Mei; Anna Seo Gyeong Choi; Hilke Schellmann; Mona Sloane; Allison Koenecke
>
> **摘要:** Automatic Speech Recognition (ASR) has transformed daily tasks from video transcription to workplace hiring. ASR systems' growing use warrants robust and standardized auditing approaches to ensure automated transcriptions of high and equitable quality. This is especially critical for people with speech and language disorders (such as aphasia) who may disproportionately depend on ASR systems to navigate everyday life. In this work, we identify three pitfalls in existing standard ASR auditing procedures, and demonstrate how addressing them impacts audit results via a case study of six popular ASR systems' performance for aphasia speakers. First, audits often adhere to a single method of text standardization during data pre-processing, which (a) masks variability in ASR performance from applying different standardization methods, and (b) may not be consistent with how users - especially those from marginalized speech communities - would want their transcriptions to be standardized. Second, audits often display high-level demographic findings without further considering performance disparities among (a) more nuanced demographic subgroups, and (b) relevant covariates capturing acoustic information from the input audio. Third, audits often rely on a single gold-standard metric -- the Word Error Rate -- which does not fully capture the extent of errors arising from generative AI models, such as transcription hallucinations. We propose a more holistic auditing framework that accounts for these three pitfalls, and exemplify its results in our case study, finding consistently worse ASR performance for aphasia speakers relative to a control group. We call on practitioners to implement these robust ASR auditing practices that remain flexible to the rapidly changing ASR landscape.
>
---
#### [replaced 006] UmbraTTS: Adapting Text-to-Speech to Environmental Contexts with Flow Matching
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.09874v2](http://arxiv.org/pdf/2506.09874v2)**

> **作者:** Neta Glazer; Aviv Navon; Yael Segal; Aviv Shamsian; Hilit Segev; Asaf Buchnick; Menachem Pirchi; Gil Hetz; Joseph Keshet
>
> **备注:** ICML Workshop on Machine Learning for Audio 2025
>
> **摘要:** Recent advances in Text-to-Speech (TTS) have enabled highly natural speech synthesis, yet integrating speech with complex background environments remains challenging. We introduce UmbraTTS, a flow-matching based TTS model that jointly generates both speech and environmental audio, conditioned on text and acoustic context. Our model allows fine-grained control over background volume and produces diverse, coherent, and context-aware audio scenes. A key challenge is the lack of data with speech and background audio aligned in natural context. To overcome the lack of paired training data, we propose a self-supervised framework that extracts speech, background audio, and transcripts from unannotated recordings. Extensive evaluations demonstrate that UmbraTTS significantly outperformed existing baselines, producing natural, high-quality, environmentally aware audios.
>
---
