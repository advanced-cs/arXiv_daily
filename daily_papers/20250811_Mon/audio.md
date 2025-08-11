# 音频 cs.SD;  eess.SP

- **最新发布 12 篇**

- **更新 2 篇**

## 最新发布

#### [new 001] Robust Target Speaker Diarization and Separation via Augmented Speaker Embedding Sampling
- **分类: cs.SD; cs.AI**

- **简介: 本文提出一种无需先验知识的语音分离与说话人分组方法，通过增强说话人嵌入采样实现鲁棒性，结合双阶段训练和重叠频谱损失提升分组精度，显著优于现有基线。**

- **链接: [http://arxiv.org/pdf/2508.06393v1](http://arxiv.org/pdf/2508.06393v1)**

> **作者:** Md Asif Jalal; Luca Remaggi; Vasileios Moschopoulos; Thanasis Kotsiopoulos; Vandana Rajan; Karthikeyan Saravanan; Anastasis Drosou; Junho Heo; Hyuk Oh; Seokyeong Jeong
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Traditional speech separation and speaker diarization approaches rely on prior knowledge of target speakers or a predetermined number of participants in audio signals. To address these limitations, recent advances focus on developing enrollment-free methods capable of identifying targets without explicit speaker labeling. This work introduces a new approach to train simultaneous speech separation and diarization using automatic identification of target speaker embeddings, within mixtures. Our proposed model employs a dual-stage training pipeline designed to learn robust speaker representation features that are resilient to background noise interference. Furthermore, we present an overlapping spectral loss function specifically tailored for enhancing diarization accuracy during overlapped speech frames. Experimental results show significant performance gains compared to the current SOTA baseline, achieving 71% relative improvement in DER and 69% in cpWER.
>
---
#### [new 002] DAFMSVC: One-Shot Singing Voice Conversion with Dual Attention Mechanism and Flow Matching
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 论文提出DAFMSVC，通过替换目标音频的自监督特征解决音色泄露问题，结合双注意力机制与流匹配模块提升SVC质量，显著改善音色相似度与自然度。**

- **链接: [http://arxiv.org/pdf/2508.05978v1](http://arxiv.org/pdf/2508.05978v1)**

> **作者:** Wei Chen; Binzhu Sha; Dan Luo; Jing Yang; Zhuo Wang; Fan Fan; Zhiyong Wu
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** Singing Voice Conversion (SVC) transfers a source singer's timbre to a target while keeping melody and lyrics. The key challenge in any-to-any SVC is adapting unseen speaker timbres to source audio without quality degradation. Existing methods either face timbre leakage or fail to achieve satisfactory timbre similarity and quality in the generated audio. To address these challenges, we propose DAFMSVC, where the self-supervised learning (SSL) features from the source audio are replaced with the most similar SSL features from the target audio to prevent timbre leakage. It also incorporates a dual cross-attention mechanism for the adaptive fusion of speaker embeddings, melody, and linguistic content. Additionally, we introduce a flow matching module for high quality audio generation from the fused features. Experimental results show that DAFMSVC significantly enhances timbre similarity and naturalness, outperforming state-of-the-art methods in both subjective and objective evaluations.
>
---
#### [new 003] Improved Dysarthric Speech to Text Conversion via TTS Personalization
- **分类: cs.SD; cs.HC**

- **简介: 论文提出通过个性化TTS生成合成口吃语音，优化零样本下的语音识别性能，降低字符错误率。任务为改进口吃语音转文本转换，解决现有ASR模型在零样本场景下误差高的问题，采用合成语音微调提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.06391v1](http://arxiv.org/pdf/2508.06391v1)**

> **作者:** Péter Mihajlik; Éva Székely; Piroska Barta; Máté Soma Kádár; Gergely Dobsinszki; László Tóth
>
> **摘要:** We present a case study on developing a customized speech-to-text system for a Hungarian speaker with severe dysarthria. State-of-the-art automatic speech recognition (ASR) models struggle with zero-shot transcription of dysarthric speech, yielding high error rates. To improve performance with limited real dysarthric data, we fine-tune an ASR model using synthetic speech generated via a personalized text-to-speech (TTS) system. We introduce a method for generating synthetic dysarthric speech with controlled severity by leveraging premorbidity recordings of the given speaker and speaker embedding interpolation, enabling ASR fine-tuning on a continuum of impairments. Fine-tuning on both real and synthetic dysarthric speech reduces the character error rate (CER) from 36-51% (zero-shot) to 7.3%. Our monolingual FastConformer_Hu ASR model significantly outperforms Whisper-turbo when fine-tuned on the same data, and the inclusion of synthetic speech contributes to an 18% relative CER reduction. These results highlight the potential of personalized ASR systems for improving accessibility for individuals with severe speech impairments.
>
---
#### [new 004] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models
- **分类: cs.SD; cs.AI**

- **简介: 论文提出SpeakerLM，通过多模态大语言模型实现端到端的SDR任务，解决传统分层框架的误差传播、重叠语音处理及联合优化问题，引入灵活注册机制提升跨场景适应性。**

- **链接: [http://arxiv.org/pdf/2508.06372v1](http://arxiv.org/pdf/2508.06372v1)**

> **作者:** Han Yin; Yafeng Chen; Chong Deng; Luyao Cheng; Hui Wang; Chao-Hong Tan; Qian Chen; Wen Wang; Xiangang Li
>
> **摘要:** The Speaker Diarization and Recognition (SDR) task aims to predict "who spoke when and what" within an audio clip, which is a crucial task in various real-world multi-speaker scenarios such as meeting transcription and dialogue systems. Existing SDR systems typically adopt a cascaded framework, combining multiple modules such as speaker diarization (SD) and automatic speech recognition (ASR). The cascaded systems suffer from several limitations, such as error propagation, difficulty in handling overlapping speech, and lack of joint optimization for exploring the synergy between SD and ASR tasks. To address these limitations, we introduce SpeakerLM, a unified multimodal large language model for SDR that jointly performs SD and ASR in an end-to-end manner. Moreover, to facilitate diverse real-world scenarios, we incorporate a flexible speaker registration mechanism into SpeakerLM, enabling SDR under different speaker registration settings. SpeakerLM is progressively developed with a multi-stage training strategy on large-scale real data. Extensive experiments show that SpeakerLM demonstrates strong data scaling capability and generalizability, outperforming state-of-the-art cascaded baselines on both in-domain and out-of-domain public SDR benchmarks. Furthermore, experimental results show that the proposed speaker registration mechanism effectively ensures robust SDR performance of SpeakerLM across diverse speaker registration conditions and varying numbers of registered speakers.
>
---
#### [new 005] EmoAugNet: A Signal-Augmented Hybrid CNN-LSTM Framework for Speech Emotion Recognition
- **分类: cs.SD; cs.HC; cs.LG**

- **简介: 论文提出EmoAugNet，结合CNN-LSTM框架与数据增强，提升语音情感识别性能。**

- **链接: [http://arxiv.org/pdf/2508.06321v1](http://arxiv.org/pdf/2508.06321v1)**

> **作者:** Durjoy Chandra Paul; Gaurob Saha; Md Amjad Hossain
>
> **备注:** To be published in ICCCNT 2025 (16th International Conference on Computing Communication and Networking Technologies)
>
> **摘要:** Recognizing emotional signals in speech has a significant impact on enhancing the effectiveness of human-computer interaction (HCI). This study introduces EmoAugNet, a hybrid deep learning framework, that incorporates Long Short-Term Memory (LSTM) layers with one-dimensional Convolutional Neural Networks (1D-CNN) to enable reliable Speech Emotion Recognition (SER). The quality and variety of the features that are taken from speech signals have a significant impact on how well SER systems perform. A comprehensive speech data augmentation strategy was used to combine both traditional methods, such as noise addition, pitch shifting, and time stretching, with a novel combination-based augmentation pipeline to enhance generalization and reduce overfitting. Each audio sample was transformed into a high-dimensional feature vector using root mean square energy (RMSE), Mel-frequency Cepstral Coefficient (MFCC), and zero-crossing rate (ZCR). Our model with ReLU activation has a weighted accuracy of 95.78\% and unweighted accuracy of 92.52\% on the IEMOCAP dataset and, with ELU activation, has a weighted accuracy of 96.75\% and unweighted accuracy of 91.28\%. On the RAVDESS dataset, we get a weighted accuracy of 94.53\% and 94.98\% unweighted accuracy for ReLU activation and 93.72\% weighted accuracy and 94.64\% unweighted accuracy for ELU activation. These results highlight EmoAugNet's effectiveness in improving the robustness and performance of SER systems through integated data augmentation and hybrid modeling.
>
---
#### [new 006] Llasa+: Free Lunch for Accelerated and Streaming Llama-Based Speech Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出Llasa+，通过引入多token预测模块和验证机制，加速LLM-based语音合成并实现流式输出，解决传统模型速度与质量矛盾，提升合成效率。**

- **链接: [http://arxiv.org/pdf/2508.06262v1](http://arxiv.org/pdf/2508.06262v1)**

> **作者:** Wenjie Tian; Xinfa Zhu; Hanke Xie; Zhen Ye; Wei Xue; Lei Xie
>
> **摘要:** Recent progress in text-to-speech (TTS) has achieved impressive naturalness and flexibility, especially with the development of large language model (LLM)-based approaches. However, existing autoregressive (AR) structures and large-scale models, such as Llasa, still face significant challenges in inference latency and streaming synthesis. To deal with the limitations, we introduce Llasa+, an accelerated and streaming TTS model built on Llasa. Specifically, to accelerate the generation process, we introduce two plug-and-play Multi-Token Prediction (MTP) modules following the frozen backbone. These modules allow the model to predict multiple tokens in one AR step. Additionally, to mitigate potential error propagation caused by inaccurate MTP, we design a novel verification algorithm that leverages the frozen backbone to validate the generated tokens, thus allowing Llasa+ to achieve speedup without sacrificing generation quality. Furthermore, we design a causal decoder that enables streaming speech reconstruction from tokens. Extensive experiments show that Llasa+ achieves a 1.48X speedup without sacrificing generation quality, despite being trained only on LibriTTS. Moreover, the MTP-and-verification framework can be applied to accelerate any LLM-based model. All codes and models are publicly available at https://github.com/ASLP-lab/LLaSA_Plus.
>
---
#### [new 007] Training chord recognition models on artificially generated audio
- **分类: cs.SD; cs.LG**

- **简介: 论文对比两种Transformer模型，探索人工生成音频在和弦识别任务中的应用，解决版权数据不足问题，验证人工数据可作为替代训练集用于流行音乐和弦预测。**

- **链接: [http://arxiv.org/pdf/2508.05878v1](http://arxiv.org/pdf/2508.05878v1)**

> **作者:** Martyna Majchrzak; Jacek Mańdziuk
>
> **摘要:** One of the challenging problems in Music Information Retrieval is the acquisition of enough non-copyrighted audio recordings for model training and evaluation. This study compares two Transformer-based neural network models for chord sequence recognition in audio recordings and examines the effectiveness of using an artificially generated dataset for this purpose. The models are trained on various combinations of Artificial Audio Multitracks (AAM), Schubert's Winterreise Dataset, and the McGill Billboard Dataset and evaluated with three metrics: Root, MajMin and Chord Content Metric (CCM). The experiments prove that even though there are certainly differences in complexity and structure between artificially generated and human-composed music, the former can be useful in certain scenarios. Specifically, AAM can enrich a smaller training dataset of music composed by a human or can even be used as a standalone training set for a model that predicts chord sequences in pop music, if no other data is available.
>
---
#### [new 008] MeanAudio: Fast and Faithful Text-to-Audio Generation with Mean Flows
- **分类: cs.SD; cs.AI**

- **简介: 本文提出MeanAudio，一种基于均流的文本到音频生成模型，解决速度慢问题，通过平均速度场加速生成并结合CFG和渐进式课程学习提升效率。**

- **链接: [http://arxiv.org/pdf/2508.06098v1](http://arxiv.org/pdf/2508.06098v1)**

> **作者:** Xiquan Li; Junxi Liu; Yuzhe Liang; Zhikang Niu; Wenxi Chen; Xie Chen
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Recent developments in diffusion- and flow- based models have significantly advanced Text-to-Audio Generation (TTA). While achieving great synthesis quality and controllability, current TTA systems still suffer from slow inference speed, which significantly limits their practical applicability. This paper presents MeanAudio, a novel MeanFlow-based model tailored for fast and faithful text-to-audio generation. Built on a Flux-style latent transformer, MeanAudio regresses the average velocity field during training, enabling fast generation by mapping directly from the start to the endpoint of the flow trajectory. By incorporating classifier-free guidance (CFG) into the training target, MeanAudio incurs no additional cost in the guided sampling process. To further stabilize training, we propose an instantaneous-to-mean curriculum with flow field mix-up, which encourages the model to first learn the foundational instantaneous dynamics, and then gradually adapt to mean flows. This strategy proves critical for enhancing training efficiency and generation quality. Experimental results demonstrate that MeanAudio achieves state-of-the-art performance in single-step audio generation. Specifically, it achieves a real time factor (RTF) of 0.013 on a single NVIDIA RTX 3090, yielding a 100x speedup over SOTA diffusion-based TTA systems. Moreover, MeanAudio also demonstrates strong performance in multi-step generation, enabling smooth and coherent transitions across successive synthesis steps.
>
---
#### [new 009] Multivariate Fields of Experts
- **分类: eess.IV; cs.CV; cs.LG; eess.SP**

- **简介: 该论文提出多变量专家场框架，用于图像先验学习，解决逆向问题，通过Moreau envelopes构造多变量势函数，提升性能、速度与可解释性。**

- **链接: [http://arxiv.org/pdf/2508.06490v1](http://arxiv.org/pdf/2508.06490v1)**

> **作者:** Stanislas Ducotterd; Michael Unser
>
> **摘要:** We introduce the multivariate fields of experts, a new framework for the learning of image priors. Our model generalizes existing fields of experts methods by incorporating multivariate potential functions constructed via Moreau envelopes of the $\ell_\infty$-norm. We demonstrate the effectiveness of our proposal across a range of inverse problems that include image denoising, deblurring, compressed-sensing magnetic-resonance imaging, and computed tomography. The proposed approach outperforms comparable univariate models and achieves performance close to that of deep-learning-based regularizers while being significantly faster, requiring fewer parameters, and being trained on substantially fewer data. In addition, our model retains a relatively high level of interpretability due to its structured design.
>
---
#### [new 010] Acoustic Non-Stationarity Objective Assessment with Hard Label Criteria for Supervised Learning Models
- **分类: eess.AS; eess.SP**

- **简介: 本文提出基于硬标签准则（HLC）的非稳态声学评估方法，解决传统客观指标计算复杂的问题，提出NANSA网络实现99%准确率。**

- **链接: [http://arxiv.org/pdf/2508.06405v1](http://arxiv.org/pdf/2508.06405v1)**

> **作者:** Guilherme Zucatelli; Ricardo Barioni; Gabriela Dantas
>
> **备注:** Manuscript under review
>
> **摘要:** Objective non-stationarity measures are resource intensive and impose critical limitations for real-time processing solutions. In this paper, a novel Hard Label Criteria (HLC) algorithm is proposed to generate a global non-stationarity label for acoustic signals, enabling supervised learning strategies to be trained as stationarity estimators. The HLC is first evaluated on state-of-the-art general-purpose acoustic models, demonstrating that these models encode stationarity information. Furthermore, the first-of-its-kind HLC-based Network for Acoustic Non-Stationarity Assessment (NANSA) is proposed. NANSA models outperform competing approaches, achieving up to 99\% classification accuracy, while solving the computational infeasibility of traditional objective measures.
>
---
#### [new 011] NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 本文提出NanoCodec，通过低帧率编码优化语音LLM推理，解决高帧率导致的慢问题，实现高质量压缩并设新基准。**

- **链接: [http://arxiv.org/pdf/2508.05835v1](http://arxiv.org/pdf/2508.05835v1)**

> **作者:** Edresson Casanova; Paarth Neekhara; Ryan Langman; Shehzeen Hussain; Subhankar Ghosh; Xuesong Yang; Ante Jukić; Jason Li; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Large Language Models (LLMs) have significantly advanced audio processing by leveraging audio codecs to discretize audio into tokens, enabling the application of language modeling techniques to speech data. However, existing audio codecs often operate at high frame rates, leading to slow training and inference, particularly for autoregressive models. To address this, there is growing interest in low frame-rate audio codecs, which reduce the number of autoregressive steps required to generate one second of audio. In this paper, we conduct ablation studies to examine the impact of frame rate, bitrate, and causality on codec reconstruction quality. Based on our findings, we introduce NanoCodec, a state-of-the-art audio codec that achieves high-quality compression at just 12.5 frames per second (FPS). NanoCodec outperforms related works across various bitrate ranges, establishing a new benchmark for low-latency and efficient Speech LLM training and inference.
>
---
#### [new 012] Large Language Model Data Generation for Enhanced Intent Recognition in German Speech
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 论文提出通过合成LLM数据提升德语语音意图识别，解决现有方法对短命令和英语的局限，采用适应Whisper和微调Transformer，结合合成数据测试，显示合成数据提升性能，LeoLM优于ChatGPT。**

- **链接: [http://arxiv.org/pdf/2508.06277v1](http://arxiv.org/pdf/2508.06277v1)**

> **作者:** Theresa Pekarek Rosin; Burak Can Kaplan; Stefan Wermter
>
> **备注:** 11 pages, 3 figures, accepted at KONVENS 2025
>
> **摘要:** Intent recognition (IR) for speech commands is essential for artificial intelligence (AI) assistant systems; however, most existing approaches are limited to short commands and are predominantly developed for English. This paper addresses these limitations by focusing on IR from speech by elderly German speakers. We propose a novel approach that combines an adapted Whisper ASR model, fine-tuned on elderly German speech (SVC-de), with Transformer-based language models trained on synthetic text datasets generated by three well-known large language models (LLMs): LeoLM, Llama3, and ChatGPT. To evaluate the robustness of our approach, we generate synthetic speech with a text-to-speech model and conduct extensive cross-dataset testing. Our results show that synthetic LLM-generated data significantly boosts classification performance and robustness to different speaking styles and unseen vocabulary. Notably, we find that LeoLM, a smaller, domain-specific 13B LLM, surpasses the much larger ChatGPT (175B) in dataset quality for German intent recognition. Our approach demonstrates that generative AI can effectively bridge data gaps in low-resource domains. We provide detailed documentation of our data generation and training process to ensure transparency and reproducibility.
>
---
## 更新

#### [replaced 001] SAMUeL: Efficient Vocal-Conditioned Music Generation via Soft Alignment Attention and Latent Diffusion
- **分类: cs.SD; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.19991v2](http://arxiv.org/pdf/2507.19991v2)**

> **作者:** Hei Shing Cheung; Boya Zhang; Jonathan H. Chan
>
> **备注:** 7 page, 3 figures, submitted to IEEE/WIC WI-IAT
>
> **摘要:** We present a lightweight latent diffusion model for vocal-conditioned musical accompaniment generation that addresses critical limitations in existing music AI systems. Our approach introduces a novel soft alignment attention mechanism that adaptively combines local and global temporal dependencies based on diffusion timesteps, enabling efficient capture of multi-scale musical structure. Operating in the compressed latent space of a pre-trained variational autoencoder, the model achieves a 220 times parameter reduction compared to state-of-the-art systems while delivering 52 times faster inference. Experimental evaluation demonstrates competitive performance with only 15M parameters, outperforming OpenAI Jukebox in production quality and content unity while maintaining reasonable musical coherence. The ultra-lightweight architecture enables real-time deployment on consumer hardware, making AI-assisted music creation accessible for interactive applications and resource-constrained environments.
>
---
#### [replaced 002] Survey on the Evaluation of Generative Models in Music
- **分类: cs.SD; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05104v2](http://arxiv.org/pdf/2506.05104v2)**

> **作者:** Alexander Lerch; Claire Arthur; Nick Bryan-Kinns; Corey Ford; Qianyi Sun; Ashvala Vinay
>
> **备注:** Minor Revision submitted to ACM CSUR on 08-Aug-2025, original manuscript submitted on 26-Jun-2024
>
> **摘要:** Research on generative systems in music has seen considerable attention and growth in recent years. A variety of attempts have been made to systematically evaluate such systems. We present an interdisciplinary review of the common evaluation targets, methodologies, and metrics for the evaluation of both system output and model use, covering subjective and objective approaches, qualitative and quantitative approaches, as well as empirical and computational methods. We examine the benefits and limitations of these approaches from a musicological, an engineering, and an HCI perspective.
>
---
