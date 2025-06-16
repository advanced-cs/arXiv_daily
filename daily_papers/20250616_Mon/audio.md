# 音频 cs.SD;  eess.SP

- **最新发布 31 篇**

- **更新 8 篇**

## 最新发布

#### [new 001] (SimPhon Speech Test): A Data-Driven Method for In Silico Design and Validation of a Phonetically Balanced Speech Test
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音测试设计任务，旨在解决听力损失对语音理解影响评估不足的问题。通过计算方法构建平衡的语音测试集，提升诊断效率。**

- **链接: [http://arxiv.org/pdf/2506.11620v1](http://arxiv.org/pdf/2506.11620v1)**

> **作者:** Stefan Bleeck
>
> **摘要:** Traditional audiometry often provides an incomplete characterization of the functional impact of hearing loss on speech understanding, particularly for supra-threshold deficits common in presbycusis. This motivates the development of more diagnostically specific speech perception tests. We introduce the Simulated Phoneme Speech Test (SimPhon Speech Test) methodology, a novel, multi-stage computational pipeline for the in silico design and validation of a phonetically balanced minimal-pair speech test. This methodology leverages a modern Automatic Speech Recognition (ASR) system as a proxy for a human listener to simulate the perceptual effects of sensorineural hearing loss. By processing speech stimuli under controlled acoustic degradation, we first identify the most common phoneme confusion patterns. These patterns then guide the data-driven curation of a large set of candidate word pairs derived from a comprehensive linguistic corpus. Subsequent phases involving simulated diagnostic testing, expert human curation, and a final, targeted sensitivity analysis systematically reduce the candidates to a final, optimized set of 25 pairs (the SimPhon Speech Test-25). A key finding is that the diagnostic performance of the SimPhon Speech Test-25 test items shows no significant correlation with predictions from the standard Speech Intelligibility Index (SII), suggesting the SimPhon Speech Test captures perceptual deficits beyond simple audibility. This computationally optimized test set offers a significant increase in efficiency for audiological test development, ready for initial human trials.
>
---
#### [new 002] Abstract Sound Fusion with Unconditioned Inversion Model
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于声音合成任务，旨在实现超越简单叠加的音效融合。通过无条件逆向模型，保留原始特征并可控合成新声音。**

- **链接: [http://arxiv.org/pdf/2506.11811v1](http://arxiv.org/pdf/2506.11811v1)**

> **作者:** Jing Liu; EnQi Lian
>
> **摘要:** An abstract sound is defined as a sound that does not disclose identifiable real-world sound events to a listener. Sound fusion aims to synthesize an original sound and a reference sound to generate a novel sound that exhibits auditory features beyond mere additive superposition of the sound constituents. To achieve this fusion, we employ inversion techniques that preserve essential features of the original sample while enabling controllable synthesis. We propose novel SDE and ODE inversion models based on DPMSolver++ samplers that reverse the sampling process by configuring model outputs as constants, eliminating circular dependencies incurred by noise prediction terms. Our inversion approach requires no prompt conditioning while maintaining flexible guidance during sampling.
>
---
#### [new 003] Enabling automatic transcription of child-centered audio recordings from real-world environments
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于语音识别任务，旨在解决儿童日常录音自动转录难题。通过检测可准确转录的语句，提升转录效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.11747v1](http://arxiv.org/pdf/2506.11747v1)**

> **作者:** Daniil Kocharov; Okko Räsänen
>
> **备注:** pre-print
>
> **摘要:** Longform audio recordings obtained with microphones worn by children-also known as child-centered daylong recordings-have become a standard method for studying children's language experiences and their impact on subsequent language development. Transcripts of longform speech audio would enable rich analyses at various linguistic levels, yet the massive scale of typical longform corpora prohibits comprehensive manual annotation. At the same time, automatic speech recognition (ASR)-based transcription faces significant challenges due to the noisy, unconstrained nature of real-world audio, and no existing study has successfully applied ASR to transcribe such data. However, previous attempts have assumed that ASR must process each longform recording in its entirety. In this work, we present an approach to automatically detect those utterances in longform audio that can be reliably transcribed with modern ASR systems, allowing automatic and relatively accurate transcription of a notable proportion of all speech in typical longform data. We validate the approach on four English longform audio corpora, showing that it achieves a median word error rate (WER) of 0% and a mean WER of 18% when transcribing 13% of the total speech in the dataset. In contrast, transcribing all speech without any filtering yields a median WER of 52% and a mean WER of 51%. We also compare word log-frequencies derived from the automatic transcripts with those from manual annotations and show that the frequencies correlate at r = 0.92 (Pearson) for all transcribed words and r = 0.98 for words that appear at least five times in the automatic transcripts. Overall, the work provides a concrete step toward increasingly detailed automated linguistic analyses of child-centered longform audio.
>
---
#### [new 004] Dissecting the Segmentation Model of End-to-End Diarization with Vector Clustering
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于说话人日志任务，研究端到端语音分割模型的优化。通过分析不同编码器、解码器、损失函数和块大小的影响，提升系统性能。**

- **链接: [http://arxiv.org/pdf/2506.11605v1](http://arxiv.org/pdf/2506.11605v1)**

> **作者:** Alexis Plaquet; Naohiro Tawara; Marc Delcroix; Shota Horiguchi; Atsushi Ando; Shoko Araki; Hervé Bredin
>
> **备注:** 37 pages, 18 figures. Submitted to Computer Speech & Language
>
> **摘要:** End-to-End Neural Diarization with Vector Clustering is a powerful and practical approach to perform Speaker Diarization. Multiple enhancements have been proposed for the segmentation model of these pipelines, but their synergy had not been thoroughly evaluated. In this work, we provide an in-depth analysis on the impact of major architecture choices on the performance of the pipeline. We investigate different encoders (SincNet, pretrained and finetuned WavLM), different decoders (LSTM, Mamba, and Conformer), different losses (multilabel and multiclass powerset), and different chunk sizes. Through in-depth experiments covering nine datasets, we found that the finetuned WavLM-based encoder always results in the best systems by a wide margin. The LSTM decoder is outclassed by Mamba- and Conformer-based decoders, and while we found Mamba more robust to other architecture choices, it is slightly inferior to our best architecture, which uses a Conformer encoder. We found that multilabel and multiclass powerset losses do not have the same distribution of errors. We confirmed that the multiclass loss helps almost all models attain superior performance, except when finetuning WavLM, in which case, multilabel is the superior choice. We also evaluated the impact of the chunk size on all aforementioned architecture choices and found that newer architectures tend to better handle long chunk sizes, which can greatly improve pipeline performance. Our best system achieved state-of-the-art results on five widely used speaker diarization datasets.
>
---
#### [new 005] Reimagining Dance: Real-time Music Co-creation between Dancers and AI
- **分类: cs.SD; cs.AI; cs.HC; eess.AS**

- **简介: 该论文属于舞蹈与AI协同创作任务，旨在解决舞蹈与音乐单向互动的问题。通过多模态系统实现舞者动态影响音乐生成，建立双向创作关系。**

- **链接: [http://arxiv.org/pdf/2506.12008v1](http://arxiv.org/pdf/2506.12008v1)**

> **作者:** Olga Vechtomova; Jeff Bos
>
> **备注:** Accepted for publication at ICCC 2025 (International Conference on Computational Creativity)
>
> **摘要:** Dance performance traditionally follows a unidirectional relationship where movement responds to music. While AI has advanced in various creative domains, its application in dance has primarily focused on generating choreography from musical input. We present a system that enables dancers to dynamically shape musical environments through their movements. Our multi-modal architecture creates a coherent musical composition by intelligently combining pre-recorded musical clips in response to dance movements, establishing a bidirectional creative partnership where dancers function as both performers and composers. Through correlation analysis of performance data, we demonstrate emergent communication patterns between movement qualities and audio features. This approach reconceptualizes the role of AI in performing arts as a responsive collaborator that expands possibilities for both professional dance performance and improvisational artistic expression across broader populations.
>
---
#### [new 006] End-to-End Diarization utilizing Attractor Deep Clustering
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音处理中的说话人日志任务，旨在提高说话人分离的准确性和效率。通过结合深度聚类和注意力机制，优化了说话人表示和分离效果。**

- **链接: [http://arxiv.org/pdf/2506.11090v1](http://arxiv.org/pdf/2506.11090v1)**

> **作者:** David Palzer; Matthew Maciejewski; Eric Fosler-Lussier
>
> **备注:** To appear at INTERSPEECH 2025
>
> **摘要:** Speaker diarization remains challenging due to the need for structured speaker representations, efficient modeling, and robustness to varying conditions. We propose a performant, compact diarization framework that integrates conformer decoders, transformer-updated attractors, and a deep clustering style angle loss. Our approach refines speaker representations with an enhanced conformer structure, incorporating cross-attention to attractors and an additional convolution module. To enforce structured embeddings, we extend deep clustering by constructing label-attractor vectors, aligning their directional structure with audio embeddings. We also impose orthogonality constraints on active attractors for better speaker separation while suppressing non-active attractors to prevent false activations. Finally, a permutation invariant training binary cross-entropy loss refines speaker detection. Experiments show that our method achieves low diarization error while maintaining parameter count.
>
---
#### [new 007] A correlation-permutation approach for speech-music encoders model merging
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音频模型融合任务，解决不同模型对齐难题。通过相关性-排列方法对齐语音和音乐编码器，实现高效融合，提升音乐性能。**

- **链接: [http://arxiv.org/pdf/2506.11403v1](http://arxiv.org/pdf/2506.11403v1)**

> **作者:** Fabian Ritter-Gutierrez; Yi-Cheng Lin; Jeremy H. M Wong; Hung-yi Lee; Eng Siong Chng; Nancy F. Chen
>
> **备注:** Under review
>
> **摘要:** Creating a unified speech and music model requires expensive pre-training. Model merging can instead create an unified audio model with minimal computational expense. However, direct merging is challenging when the models are not aligned in the weight space. Motivated by Git Re-Basin, we introduce a correlation-permutation approach that aligns a music encoder's internal layers with a speech encoder. We extend previous work to the case of merging transformer layers. The method computes a permutation matrix that maximizes the model's features-wise cross-correlations layer by layer, enabling effective fusion of these otherwise disjoint models. The merged model retains speech capabilities through this method while significantly enhancing music performance, achieving an improvement of 14.83 points in average score compared to linear interpolation model merging. This work allows the creation of unified audio models from independently trained encoders.
>
---
#### [new 008] Amplifying Artifacts with Speech Enhancement in Voice Anti-spoofing
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音防欺骗任务，旨在提升伪造语音检测效果。通过增强语音中的伪影特征，提出一种与模型无关的增强流程，显著提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.11542v1](http://arxiv.org/pdf/2506.11542v1)**

> **作者:** Thanapat Trachu; Thanathai Lertpetchpun; Ekapol Chuangsuwanich
>
> **备注:** Accepted to Interspeech2025
>
> **摘要:** Spoofed utterances always contain artifacts introduced by generative models. While several countermeasures have been proposed to detect spoofed utterances, most primarily focus on architectural improvements. In this work, we investigate how artifacts remain hidden in spoofed speech and how to enhance their presence. We propose a model-agnostic pipeline that amplifies artifacts using speech enhancement and various types of noise. Our approach consists of three key steps: noise addition, noise extraction, and noise amplification. First, we introduce noise into the raw speech. Then, we apply speech enhancement to extract the entangled noise and artifacts. Finally, we amplify these extracted features. Moreover, our pipeline is compatible with different speech enhancement models and countermeasure architectures. Our method improves spoof detection performance by up to 44.44\% on ASVspoof2019 and 26.34\% on ASVspoof2021.
>
---
#### [new 009] Assessing the Impact of Anisotropy in Neural Representations of Speech: A Case Study on Keyword Spotting
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文研究语音表示中的各向异性问题，针对关键词识别任务，验证了wav2vec2在无转录情况下的有效性。**

- **链接: [http://arxiv.org/pdf/2506.11096v1](http://arxiv.org/pdf/2506.11096v1)**

> **作者:** Guillaume Wisniewski; Séverine Guillaume; Clara Rosina Fernández
>
> **摘要:** Pretrained speech representations like wav2vec2 and HuBERT exhibit strong anisotropy, leading to high similarity between random embeddings. While widely observed, the impact of this property on downstream tasks remains unclear. This work evaluates anisotropy in keyword spotting for computational documentary linguistics. Using Dynamic Time Warping, we show that despite anisotropy, wav2vec2 similarity measures effectively identify words without transcription. Our results highlight the robustness of these representations, which capture phonetic structures and generalize across speakers. Our results underscore the importance of pretraining in learning rich and invariant speech representations.
>
---
#### [new 010] LiLAC: A Lightweight Latent ControlNet for Musical Audio Generation
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于音乐音频生成任务，旨在解决现有模型缺乏细粒度时间控制的问题。提出轻量级架构LiLAC，减少参数量并提升控制灵活性。**

- **链接: [http://arxiv.org/pdf/2506.11476v1](http://arxiv.org/pdf/2506.11476v1)**

> **作者:** Tom Baker; Javier Nistal
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** Text-to-audio diffusion models produce high-quality and diverse music but many, if not most, of the SOTA models lack the fine-grained, time-varying controls essential for music production. ControlNet enables attaching external controls to a pre-trained generative model by cloning and fine-tuning its encoder on new conditionings. However, this approach incurs a large memory footprint and restricts users to a fixed set of controls. We propose a lightweight, modular architecture that considerably reduces parameter count while matching ControlNet in audio quality and condition adherence. Our method offers greater flexibility and significantly lower memory usage, enabling more efficient training and deployment of independent controls. We conduct extensive objective and subjective evaluations and provide numerous audio examples on the accompanying website at https://lightlatentcontrol.github.io
>
---
#### [new 011] Confidence-Based Self-Training for EMG-to-Speech: Leveraging Synthetic EMG for Robust Modeling
- **分类: cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于EMG到语音的重建任务，旨在解决数据稀缺问题。通过自训练方法和合成数据提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11862v1](http://arxiv.org/pdf/2506.11862v1)**

> **作者:** Xiaodan Chen; Xiaoxue Gao; Mathias Quoy; Alexandre Pitti; Nancy F. Chen
>
> **摘要:** Voiced Electromyography (EMG)-to-Speech (V-ETS) models reconstruct speech from muscle activity signals, facilitating applications such as neurolaryngologic diagnostics. Despite its potential, the advancement of V-ETS is hindered by a scarcity of paired EMG-speech data. To address this, we propose a novel Confidence-based Multi-Speaker Self-training (CoM2S) approach, along with a newly curated Libri-EMG dataset. This approach leverages synthetic EMG data generated by a pre-trained model, followed by a proposed filtering mechanism based on phoneme-level confidence to enhance the ETS model through the proposed self-training techniques. Experiments demonstrate our method improves phoneme accuracy, reduces phonological confusion, and lowers word error rate, confirming the effectiveness of our CoM2S approach for V-ETS. In support of future research, we will release the codes and the proposed Libri-EMG dataset-an open-access, time-aligned, multi-speaker voiced EMG and speech recordings.
>
---
#### [new 012] GLAP: General contrastive audio-text pretraining across domains and languages
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出GLAP，解决跨语言和跨领域的音频-文本预训练问题。通过多语言和多任务实验，验证其在音频检索与分类中的优越性能。**

- **链接: [http://arxiv.org/pdf/2506.11350v1](http://arxiv.org/pdf/2506.11350v1)**

> **作者:** Heinrich Dinkel; Zhiyong Yan; Tianzi Wang; Yongqing Wang; Xingwei Sun; Yadong Niu; Jizhong Liu; Gang Li; Junbo Zhang; Jian Luan
>
> **摘要:** Contrastive Language Audio Pretraining (CLAP) is a widely-used method to bridge the gap between audio and text domains. Current CLAP methods enable sound and music retrieval in English, ignoring multilingual spoken content. To address this, we introduce general language audio pretraining (GLAP), which expands CLAP with multilingual and multi-domain abilities. GLAP demonstrates its versatility by achieving competitive performance on standard audio-text retrieval benchmarks like Clotho and AudioCaps, while significantly surpassing existing methods in speech retrieval and classification tasks. Additionally, GLAP achieves strong results on widely used sound-event zero-shot benchmarks, while simultaneously outperforming previous methods on speech content benchmarks. Further keyword spotting evaluations across 50 languages emphasize GLAP's advanced multilingual capabilities. Finally, multilingual sound and music understanding is evaluated across four languages. Checkpoints and Source: https://github.com/xiaomi-research/dasheng-glap.
>
---
#### [new 013] Fifteen Years of Child-Centered Long-Form Recordings: Promises, Resources, and Remaining Challenges to Validity
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于儿童语言研究任务，解决长时录音数据自动化分析问题，总结现有资源并提出质量评估策略。**

- **链接: [http://arxiv.org/pdf/2506.11075v1](http://arxiv.org/pdf/2506.11075v1)**

> **作者:** Loann Peurey; Marvin Lavechin; Tarek Kunze; Manel Khentout; Lucas Gautheron; Emmanuel Dupoux; Alejandrina Cristia
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Audio-recordings collected with a child-worn device are a fundamental tool in child language research. Long-form recordings collected over whole days promise to capture children's input and production with minimal observer bias, and therefore high validity. The sheer volume of resulting data necessitates automated analysis to extract relevant metrics for researchers and clinicians. This paper summarizes collective knowledge on this technique, providing entry points to existing resources. We also highlight various sources of error that threaten the accuracy of automated annotations and the interpretation of resulting metrics. To address this, we propose potential troubleshooting metrics to help users assess data quality. While a fully automated quality control system is not feasible, we outline practical strategies for researchers to improve data collection and contextualize their analyses.
>
---
#### [new 014] S2ST-Omni: An Efficient and Scalable Multilingual Speech-to-Speech Translation Framework via Seamlessly Speech-Text Alignment and Streaming Speech Decoder
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多语言语音到语音翻译任务，旨在解决高质低延时翻译及依赖平行语料的问题。通过分解任务并利用预训练模型实现高效翻译。**

- **链接: [http://arxiv.org/pdf/2506.11160v1](http://arxiv.org/pdf/2506.11160v1)**

> **作者:** Yu Pan; Yuguang Yang; Yanni Hu; Jianhao Ye; Xiang Zhang; Hongbin Zhou; Lei Ma; Jianjun Zhao
>
> **备注:** Working in progress
>
> **摘要:** Multilingual speech-to-speech translation (S2ST) aims to directly convert spoken utterances from multiple source languages into natural and intelligible speech in a target language. Despite recent progress, significant challenges remain: (1) achieving high-quality and low-latency S2ST remains a critical hurdle; (2) existing S2ST approaches heavily rely on large-scale parallel speech corpora, which are extremely difficult to collect. To address these issues, we propose S2ST-Omni, an efficient and scalable framework for multilingual speech-to-speech translation. Specifically, we decompose the S2ST task into speech-to-text translation (S2TT) and text-to-speech synthesis (TTS), unifying them within a single end-to-end speech-language model. To achieve high-quality S2TT while reducing dependence on parallel corpora, we leverage large-scale pretrained models -- Whisper for audio understanding and Qwen 3.0 for text understanding. A lightweight speech adapter is introduced to align speech and text representations, enabling effective use of pretrained multimodal knowledge. To ensure both translation quality and real-time performance, we adopt a pretrained streaming speech decoder in the TTS stage to generate target speech in an autoregressive manner. Extensive experiments on the CVSS benchmark demonstrate that S2ST-Omni outperforms state-of-the-art S2ST baselines while maintaining comparable latency, highlighting its effectiveness and practical potential for real-world deployment.
>
---
#### [new 015] Tracking of Intermittent and Moving Speakers : Dataset and Metrics
- **分类: eess.AS; cs.SD; eess.SP**

- **简介: 该论文属于语音跟踪任务，解决间歇性移动说话人跟踪问题。提出LibriJump数据集和关联度量，用于评估跟踪性能。**

- **链接: [http://arxiv.org/pdf/2506.11145v1](http://arxiv.org/pdf/2506.11145v1)**

> **作者:** Taous Iatariene; Alexandre Guérin; Romain Serizel
>
> **摘要:** This paper presents the problem of tracking intermittent and moving sources, i.e, sources that may change position when they are inactive. This issue is seldom explored, and most current tracking methods rely on spatial observations for track identity management. They are either based on a previous localization step, or designed to perform joint localization and tracking by predicting ordered position estimates. This raises concerns about whether such methods can maintain reliable track identity assignment performance when dealing with discontinuous spatial tracks, which may be caused by a change of direction during silence. We introduce LibriJump, a novel dataset of acoustic scenes in the First Order Ambisonics format focusing on speaker tracking. The dataset contains speakers with changing positions during inactivity periods, thus simulating discontinuous tracks. To measure the identity assignment performance, we propose to use tracking association metrics adapted from the computer vision community. We provide experiments showing the complementarity of association metrics with previously used tracking metrics, given continuous and discontinuous spatial tracks.
>
---
#### [new 016] Customizing Speech Recognition Model with Large Language Model Feedback
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别领域，解决领域不匹配导致的命名实体识别问题。通过强化学习和大语言模型反馈优化ASR模型，提升转录质量。**

- **链接: [http://arxiv.org/pdf/2506.11091v1](http://arxiv.org/pdf/2506.11091v1)**

> **作者:** Shaoshi Ling; Guoli Ye
>
> **摘要:** Automatic speech recognition (ASR) systems have achieved strong performance on general transcription tasks. However, they continue to struggle with recognizing rare named entities and adapting to domain mismatches. In contrast, large language models (LLMs), trained on massive internet-scale datasets, are often more effective across a wide range of domains. In this work, we propose a reinforcement learning based approach for unsupervised domain adaptation, leveraging unlabeled data to enhance transcription quality, particularly the named entities affected by domain mismatch, through feedback from a LLM. Given contextual information, our framework employs a LLM as the reward model to score the hypotheses from the ASR model. These scores serve as reward signals to fine-tune the ASR model via reinforcement learning. Our method achieves a 21\% improvement on entity word error rate over conventional self-training methods.
>
---
#### [new 017] MUDAS: Mote-scale Unsupervised Domain Adaptation in Multi-label Sound Classification
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于多标签声音分类任务，解决资源受限物联网设备上的无监督域适应问题。提出MUDAS框架，通过选择性重训练和自适应阈值提升分类性能。**

- **链接: [http://arxiv.org/pdf/2506.11331v1](http://arxiv.org/pdf/2506.11331v1)**

> **作者:** Jihoon Yun; Chengzhang Li; Dhrubojyoti Roy; Anish Arora
>
> **摘要:** Unsupervised Domain Adaptation (UDA) is essential for adapting machine learning models to new, unlabeled environments where data distribution shifts can degrade performance. Existing UDA algorithms are designed for single-label tasks and rely on significant computational resources, limiting their use in multi-label scenarios and in resource-constrained IoT devices. Overcoming these limitations is particularly challenging in contexts such as urban sound classification, where overlapping sounds and varying acoustics require robust, adaptive multi-label capabilities on low-power, on-device systems. To address these limitations, we introduce Mote-scale Unsupervised Domain Adaptation for Sounds (MUDAS), a UDA framework developed for multi-label sound classification in resource-constrained IoT settings. MUDAS efficiently adapts models by selectively retraining the classifier in situ using high-confidence data, minimizing computational and memory requirements to suit on-device deployment. Additionally, MUDAS incorporates class-specific adaptive thresholds to generate reliable pseudo-labels and applies diversity regularization to improve multi-label classification accuracy. In evaluations on the SONYC Urban Sound Tagging (SONYC-UST) dataset recorded at various New York City locations, MUDAS demonstrates notable improvements in classification accuracy over existing UDA algorithms, achieving good performance in a resource-constrained IoT setting.
>
---
#### [new 018] Wi-CBR: WiFi-based Cross-domain Behavior Recognition via Multimodal Collaborative Awareness
- **分类: cs.CV; eess.SP**

- **简介: 该论文属于行为识别任务，旨在通过融合WiFi信号的相位和多普勒频移数据，提升跨领域行为识别的准确性。**

- **链接: [http://arxiv.org/pdf/2506.11616v1](http://arxiv.org/pdf/2506.11616v1)**

> **作者:** Ruobei Zhang; Shengeng Tang; Huan Yan; Xiang Zhang; Richang Hong
>
> **摘要:** WiFi-based human behavior recognition aims to recognize gestures and activities by analyzing wireless signal variations. However, existing methods typically focus on a single type of data, neglecting the interaction and fusion of multiple features. To this end, we propose a novel multimodal collaborative awareness method. By leveraging phase data reflecting changes in dynamic path length and Doppler Shift (DFS) data corresponding to frequency changes related to the speed of gesture movement, we enable efficient interaction and fusion of these features to improve recognition accuracy. Specifically, we first introduce a dual-branch self-attention module to capture spatial-temporal cues within each modality. Then, a group attention mechanism is applied to the concatenated phase and DFS features to mine key group features critical for behavior recognition. Through a gating mechanism, the combined features are further divided into PD-strengthen and PD-weaken branches, optimizing information entropy and promoting cross-modal collaborative awareness. Extensive in-domain and cross-domain experiments on two large publicly available datasets, Widar3.0 and XRF55, demonstrate the superior performance of our method.
>
---
#### [new 019] Challenges in Automated Processing of Speech from Child Wearables: The Case of Voice Type Classifier
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音处理任务，探讨儿童可穿戴设备语音数据自动处理的挑战，重点解决语音类型分类问题，通过实验分析发现数据相关性与数量对性能提升更为关键。**

- **链接: [http://arxiv.org/pdf/2506.11074v1](http://arxiv.org/pdf/2506.11074v1)**

> **作者:** Tarek Kunze; Marianne Métais; Hadrien Titeux; Lucas Elbert; Joseph Coffey; Emmanuel Dupoux; Alejandrina Cristia; Marvin Lavechin
>
> **备注:** 5 pages, 3 figures
>
> **摘要:** Recordings gathered with child-worn devices promised to revolutionize both fundamental and applied speech sciences by allowing the effortless capture of children's naturalistic speech environment and language production. This promise hinges on speech technologies that can transform the sheer mounds of data thus collected into usable information. This paper demonstrates several obstacles blocking progress by summarizing three years' worth of experiments aimed at improving one fundamental task: Voice Type Classification. Our experiments suggest that improvements in representation features, architecture, and parameter search contribute to only marginal gains in performance. More progress is made by focusing on data relevance and quantity, which highlights the importance of collecting data with appropriate permissions to allow sharing.
>
---
#### [new 020] Tracking of Spatially Dynamic Room Impulse Responses Along Locally Linearized Trajectories
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于声学环境建模任务，解决动态房间脉冲响应跟踪问题。通过分段线性化轨迹扩展方法，提升在复杂环境中的适用性。**

- **链接: [http://arxiv.org/pdf/2506.11703v1](http://arxiv.org/pdf/2506.11703v1)**

> **作者:** Kathleen MacWilliam; Thomas Dietzen; Toon van Waterschoot
>
> **备注:** 8 pages, 6 figures. Accepted paper for conference: Forum Acousticum Euronoise 2025 (fa-euronoise2025)
>
> **摘要:** Measuring room impulse responses (RIRs) at multiple spatial points is a time-consuming task, while simulations require detailed knowledge of the room's acoustic environment. In prior work, we proposed a method for estimating the early part of RIRs along a linear trajectory in a time-varying acoustic scenario involving a static sound source and a microphone moving at constant velocity. This approach relies on measured RIRs at the start and end points of the trajectory and assumes that the time intervals occupied by the direct sound and individual reflections along the trajectory are non-overlapping. The method's applicability is therefore restricted to relatively small areas within a room, and its performance has yet to be validated with real-world data. In this paper, we propose a practical extension of the method to more realistic scenarios by segmenting longer trajectories into smaller linear intervals where the assumptions approximately hold. Applying the method piecewise along these segments extends its applicability to more complex room environments. We demonstrate its effectiveness using the trajectoRIR database, which includes moving microphone recordings and RIR measurements at discrete points along a controlled L-shaped trajectory in a real room.
>
---
#### [new 021] Improved in-car sound pick-up using multichannel Wiener filter
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于语音增强任务，旨在解决车内多麦克风信号处理中的噪声和回声问题。通过使用多通道维纳滤波器提升语音质量。**

- **链接: [http://arxiv.org/pdf/2506.11157v1](http://arxiv.org/pdf/2506.11157v1)**

> **作者:** Juhi Khalid; Martin Bouchard
>
> **备注:** 6 pages
>
> **摘要:** With advancements in automotive electronics and sensors, the sound pick-up using multiple microphones has become feasible for hands-free telephony and voice command in-car applications. However, challenges remain in effectively processing multiple microphone signals due to bandwidth or processing limitations. This work explores the use of the Multichannel Wiener Filter algorithm with a two-microphone in-car system, to enhance speech quality for driver and passenger voice, i.e., to mitigate notch-filtering effects caused by echoes and improve background noise reduction. We evaluate its performance under various noise conditions using modern objective metrics like Deep Noise Suppression Mean Opinion Score. The effect of head movements of driver/passenger is also investigated. The proposed method is shown to provide significant improvements over a simple mixing of microphone signals.
>
---
#### [new 022] Advances in Small-Footprint Keyword Spotting: A Comprehensive Review of Efficient Models and Algorithms
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于小尺寸关键词检测任务，旨在解决在资源受限设备上高效实现语音关键词识别的问题。论文综述了七类关键技术，为该领域提供系统性参考。**

- **链接: [http://arxiv.org/pdf/2506.11169v1](http://arxiv.org/pdf/2506.11169v1)**

> **作者:** Soumen Garai; Suman Samui
>
> **备注:** 61 pages, 21 figures
>
> **摘要:** Small-Footprint Keyword Spotting (SF-KWS) has gained popularity in today's landscape of smart voice-activated devices, smartphones, and Internet of Things (IoT) applications. This surge is attributed to the advancements in Deep Learning, enabling the identification of predefined words or keywords from a continuous stream of words. To implement the SF-KWS model on edge devices with low power and limited memory in real-world scenarios, a efficient Tiny Machine Learning (TinyML) framework is essential. In this study, we explore seven distinct categories of techniques namely, Model Architecture, Learning Techniques, Model Compression, Attention Awareness Architecture, Feature Optimization, Neural Network Search, and Hybrid Approaches, which are suitable for developing an SF-KWS system. This comprehensive overview will serve as a valuable resource for those looking to understand, utilize, or contribute to the field of SF-KWS. The analysis conducted in this work enables the identification of numerous potential research directions, encompassing insights from automatic speech recognition research and those specifically pertinent to the realm of spoken SF-KWS.
>
---
#### [new 023] SUTA-LM: Bridging Test-Time Adaptation and Language Model Rescoring for Robust ASR
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决领域不匹配导致的性能下降问题。通过结合测试时自适应与语言模型重排序，提出SUTA-LM方法提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.11121v1](http://arxiv.org/pdf/2506.11121v1)**

> **作者:** Wei-Ping Huang; Guan-Ting Lin; Hung-yi Lee
>
> **摘要:** Despite progress in end-to-end ASR, real-world domain mismatches still cause performance drops, which Test-Time Adaptation (TTA) aims to mitigate by adjusting models during inference. Recent work explores combining TTA with external language models, using techniques like beam search rescoring or generative error correction. In this work, we identify a previously overlooked challenge: TTA can interfere with language model rescoring, revealing the nontrivial nature of effectively combining the two methods. Based on this insight, we propose SUTA-LM, a simple yet effective extension of SUTA, an entropy-minimization-based TTA approach, with language model rescoring. SUTA-LM first applies a controlled adaptation process guided by an auto-step selection mechanism leveraging both acoustic and linguistic information, followed by language model rescoring to refine the outputs. Experiments on 18 diverse ASR datasets show that SUTA-LM achieves robust results across a wide range of domains.
>
---
#### [new 024] Benchmarking Foundation Speech and Language Models for Alzheimer's Disease and Related Dementia Detection from Spontaneous Speech
- **分类: cs.CL; cs.SD; eess.AS; 68T10 (Primary), 68U99 (Secondary); I.2.1; J.3**

- **简介: 该论文属于阿尔茨海默病检测任务，旨在利用自发语音中的声学和语言特征进行早期诊断。研究对比了多种基础模型的分类效果。**

- **链接: [http://arxiv.org/pdf/2506.11119v1](http://arxiv.org/pdf/2506.11119v1)**

> **作者:** Jingyu Li; Lingchao Mao; Hairong Wang; Zhendong Wang; Xi Mao; Xuelei Sherry Ni
>
> **摘要:** Background: Alzheimer's disease and related dementias (ADRD) are progressive neurodegenerative conditions where early detection is vital for timely intervention and care. Spontaneous speech contains rich acoustic and linguistic markers that may serve as non-invasive biomarkers for cognitive decline. Foundation models, pre-trained on large-scale audio or text data, produce high-dimensional embeddings encoding contextual and acoustic features. Methods: We used the PREPARE Challenge dataset, which includes audio recordings from over 1,600 participants with three cognitive statuses: healthy control (HC), mild cognitive impairment (MCI), and Alzheimer's Disease (AD). We excluded non-English, non-spontaneous, or poor-quality recordings. The final dataset included 703 (59.13%) HC, 81 (6.81%) MCI, and 405 (34.06%) AD cases. We benchmarked a range of open-source foundation speech and language models to classify cognitive status into the three categories. Results: The Whisper-medium model achieved the highest performance among speech models (accuracy = 0.731, AUC = 0.802). Among language models, BERT with pause annotation performed best (accuracy = 0.662, AUC = 0.744). ADRD detection using state-of-the-art automatic speech recognition (ASR) model-generated audio embeddings outperformed others. Including non-semantic features like pause patterns consistently improved text-based classification. Conclusion: This study introduces a benchmarking framework using foundation models and a clinically relevant dataset. Acoustic-based approaches -- particularly ASR-derived embeddings -- demonstrate strong potential for scalable, non-invasive, and cost-effective early detection of ADRD.
>
---
#### [new 025] Improving Child Speech Recognition and Reading Mistake Detection by Using Prompts
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于儿童语音识别与阅读错误检测任务，旨在提升自动阅读评估效果。通过使用提示优化Whisper和大语言模型，显著降低了识别错误率并提高了检测准确率。**

- **链接: [http://arxiv.org/pdf/2506.11079v1](http://arxiv.org/pdf/2506.11079v1)**

> **作者:** Lingyun Gao; Cristian Tejedor-Garcia; Catia Cucchiarini; Helmer Strik
>
> **备注:** This paper is accepted to Interspeech 2025. This publication is part of the project Responsible AI for Voice Diagnostics (RAIVD) with file number NGF.1607.22.013 of the research programme NGF AiNed Fellowship Grants which is financed by the Dutch Research Council (NWO)
>
> **摘要:** Automatic reading aloud evaluation can provide valuable support to teachers by enabling more efficient scoring of reading exercises. However, research on reading evaluation systems and applications remains limited. We present a novel multimodal approach that leverages audio and knowledge from text resources. In particular, we explored the potential of using Whisper and instruction-tuned large language models (LLMs) with prompts to improve transcriptions for child speech recognition, as well as their effectiveness in downstream reading mistake detection. Our results demonstrate the effectiveness of prompting Whisper and prompting LLM, compared to the baseline Whisper model without prompting. The best performing system achieved state-of-the-art recognition performance in Dutch child read speech, with a word error rate (WER) of 5.1%, improving the baseline WER of 9.4%. Furthermore, it significantly improved reading mistake detection, increasing the F1 score from 0.39 to 0.73.
>
---
#### [new 026] From Sharpness to Better Generalization for Speech Deepfake Detection
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决模型泛化能力不足的问题。通过分析模型尖锐性，提出使用SAM方法提升检测性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.11532v1](http://arxiv.org/pdf/2506.11532v1)**

> **作者:** Wen Huang; Xuechen Liu; Xin Wang; Junichi Yamagishi; Yanmin Qian
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Generalization remains a critical challenge in speech deepfake detection (SDD). While various approaches aim to improve robustness, generalization is typically assessed through performance metrics like equal error rate without a theoretical framework to explain model performance. This work investigates sharpness as a theoretical proxy for generalization in SDD. We analyze how sharpness responds to domain shifts and find it increases in unseen conditions, indicating higher model sensitivity. Based on this, we apply Sharpness-Aware Minimization (SAM) to reduce sharpness explicitly, leading to better and more stable performance across diverse unseen test sets. Furthermore, correlation analysis confirms a statistically significant relationship between sharpness and generalization in most test settings. These findings suggest that sharpness can serve as a theoretical indicator for generalization in SDD and that sharpness-aware training offers a promising strategy for improving robustness.
>
---
#### [new 027] PMF-CEC: Phoneme-augmented Multimodal Fusion for Context-aware ASR Error Correction with Error-specific Selective Decoding
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别后处理任务，旨在解决罕见词及同音词的纠错问题。通过引入语音信息和优化检测机制，提升纠错准确率。**

- **链接: [http://arxiv.org/pdf/2506.11064v1](http://arxiv.org/pdf/2506.11064v1)**

> **作者:** Jiajun He; Tomoki Toda
>
> **备注:** Accepted by IEEE TASLP 2025
>
> **摘要:** End-to-end automatic speech recognition (ASR) models often struggle to accurately recognize rare words. Previously, we introduced an ASR postprocessing method called error detection and context-aware error correction (ED-CEC), which leverages contextual information such as named entities and technical terms to improve the accuracy of ASR transcripts. Although ED-CEC achieves a notable success in correcting rare words, its accuracy remains low when dealing with rare words that have similar pronunciations but different spellings. To address this issue, we proposed a phoneme-augmented multimodal fusion method for context-aware error correction (PMF-CEC) method on the basis of ED-CEC, which allowed for better differentiation between target rare words and homophones. Additionally, we observed that the previous ASR error detection module suffers from overdetection. To mitigate this, we introduced a retention probability mechanism to filter out editing operations with confidence scores below a set threshold, preserving the original operation to improve error detection accuracy. Experiments conducted on five datasets demonstrated that our proposed PMF-CEC maintains reasonable inference speed while further reducing the biased word error rate compared with ED-CEC, showing a stronger advantage in correcting homophones. Moreover, our method outperforms other contextual biasing methods, and remains valuable compared with LLM-based methods in terms of faster inference and better robustness under large biasing lists.
>
---
#### [new 028] Can We Trust Machine Learning? The Reliability of Features from Open-Source Speech Analysis Tools for Speech Modeling
- **分类: eess.AS; cs.CL; cs.CY; cs.SD; stat.AP; K.4; J.4; I.2**

- **简介: 该论文属于语音分析任务，探讨开源工具提取特征的可靠性问题。研究旨在解决特征不可靠导致模型偏差的问题，通过评估OpenSMILE和Praat在自闭症青少年中的表现，发现特征差异影响模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11072v1](http://arxiv.org/pdf/2506.11072v1)**

> **作者:** Tahiya Chowdhury; Veronica Romero
>
> **备注:** 5 pages, 1 figure, 3 tables
>
> **摘要:** Machine learning-based behavioral models rely on features extracted from audio-visual recordings. The recordings are processed using open-source tools to extract speech features for classification models. These tools often lack validation to ensure reliability in capturing behaviorally relevant information. This gap raises concerns about reproducibility and fairness across diverse populations and contexts. Speech processing tools, when used outside of their design context, can fail to capture behavioral variations equitably and can then contribute to bias. We evaluate speech features extracted from two widely used speech analysis tools, OpenSMILE and Praat, to assess their reliability when considering adolescents with autism. We observed considerable variation in features across tools, which influenced model performance across context and demographic groups. We encourage domain-relevant verification to enhance the reliability of machine learning models in clinical applications.
>
---
#### [new 029] Efficient Speech Enhancement via Embeddings from Pre-trained Generative Audioencoders
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在提升语音清晰度。通过预训练音频编码器提取嵌入，再经去噪网络和声码器生成干净语音，有效提升语音质量与说话人保真度。**

- **链接: [http://arxiv.org/pdf/2506.11514v1](http://arxiv.org/pdf/2506.11514v1)**

> **作者:** Xingwei Sun; Heinrich Dinkel; Yadong Niu; Linzhang Wang; Junbo Zhang; Jian Luan
>
> **备注:** Accepted by Interspeech 2025
>
> **摘要:** Recent research has delved into speech enhancement (SE) approaches that leverage audio embeddings from pre-trained models, diverging from time-frequency masking or signal prediction techniques. This paper introduces an efficient and extensible SE method. Our approach involves initially extracting audio embeddings from noisy speech using a pre-trained audioencoder, which are then denoised by a compact encoder network. Subsequently, a vocoder synthesizes the clean speech from denoised embeddings. An ablation study substantiates the parameter efficiency of the denoise encoder with a pre-trained audioencoder and vocoder. Experimental results on both speech enhancement and speaker fidelity demonstrate that our generative audioencoder-based SE system outperforms models utilizing discriminative audioencoders. Furthermore, subjective listening tests validate that our proposed system surpasses an existing state-of-the-art SE model in terms of perceptual quality.
>
---
#### [new 030] A Self-Refining Framework for Enhancing ASR Using TTS-Synthesized Data
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在提升ASR性能。通过自 refining 框架，利用未标注数据和TTS合成数据优化模型，有效降低错误率。**

- **链接: [http://arxiv.org/pdf/2506.11130v1](http://arxiv.org/pdf/2506.11130v1)**

> **作者:** Cheng Kang Chou; Chan-Jan Hsu; Ho-Lam Chung; Liang-Hsuan Tseng; Hsi-Chun Cheng; Yu-Kuan Fu; Kuan Po Huang; Hung-Yi Lee
>
> **摘要:** We propose a self-refining framework that enhances ASR performance with only unlabeled datasets. The process starts with an existing ASR model generating pseudo-labels on unannotated speech, which are then used to train a high-fidelity text-to-speech (TTS) system. Then, synthesized speech text pairs are bootstrapped into the original ASR system, completing the closed-loop self-improvement cycle. We demonstrated the effectiveness of the framework on Taiwanese Mandarin speech. Leveraging 6,000 hours of unlabeled speech, a moderate amount of text data, and synthetic content from the AI models, we adapt Whisper-large-v2 into a specialized model, Twister. Twister reduces error rates by up to 20% on Mandarin and 50% on Mandarin-English code-switching benchmarks compared to Whisper. Results highlight the framework as a compelling alternative to pseudo-labeling self-distillation approaches and provides a practical pathway for improving ASR performance in low-resource or domain-specific settings.
>
---
#### [new 031] Regularized Federated Learning for Privacy-Preserving Dysarthric and Elderly Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决隐私保护下失语和老年语音识别问题，通过正则化联邦学习提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.11069v1](http://arxiv.org/pdf/2506.11069v1)**

> **作者:** Tao Zhong; Mengzhe Geng; Shujie Hu; Guinan Li; Xunying Liu
>
> **摘要:** Accurate recognition of dysarthric and elderly speech remains challenging to date. While privacy concerns have driven a shift from centralized approaches to federated learning (FL) to ensure data confidentiality, this further exacerbates the challenges of data scarcity, imbalanced data distribution and speaker heterogeneity. To this end, this paper conducts a systematic investigation of regularized FL techniques for privacy-preserving dysarthric and elderly speech recognition, addressing different levels of the FL process by 1) parameter-based, 2) embedding-based and 3) novel loss-based regularization. Experiments on the benchmark UASpeech dysarthric and DementiaBank Pitt elderly speech corpora suggest that regularized FL systems consistently outperform the baseline FedAvg system by statistically significant WER reductions of up to 0.55\% absolute (2.13\% relative). Further increasing communication frequency to one exchange per batch approaches centralized training performance.
>
---
## 更新

#### [replaced 001] Disentangling Dual-Encoder Masked Autoencoder for Respiratory Sound Classification
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.10698v2](http://arxiv.org/pdf/2506.10698v2)**

> **作者:** Peidong Wei; Shiyu Miao; Lin Li
>
> **备注:** (Accepted at Interspeech 2025)
>
> **摘要:** Deep neural networks have been applied to audio spectrograms for respiratory sound classification, but it remains challenging to achieve satisfactory performance due to the scarcity of available data. Moreover, domain mismatch may be introduced into the trained models as a result of the respiratory sound samples being collected from various electronic stethoscopes, patient demographics, and recording environments. To tackle this issue, we proposed a modified MaskedAutoencoder(MAE) model, named Disentangling Dual-Encoder MAE (DDE-MAE) for respiratory sound classification. Two independent encoders were designed to capture disease-related and disease-irrelevant information separately, achieving feature disentanglement to reduce the domain mismatch. Our method achieves a competitive performance on the ICBHI dataset.
>
---
#### [replaced 002] Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08967v2](http://arxiv.org/pdf/2506.08967v2)**

> **作者:** Ailin Huang; Bingxin Li; Bruce Wang; Boyong Wu; Chao Yan; Chengli Feng; Heng Wang; Hongyu Zhou; Hongyuan Wang; Jingbei Li; Jianjian Sun; Joanna Wang; Mingrui Chen; Peng Liu; Ruihang Miao; Shilei Jiang; Tian Fei; Wang You; Xi Chen; Xuerui Yang; Yechang Huang; Yuxiang Zhang; Zheng Ge; Zheng Gong; Zhewei Huang; Zixin Zhang; Bin Wang; Bo Li; Buyun Ma; Changxin Miao; Changyi Wan; Chen Xu; Dapeng Shi; Dingyuan Hu; Enle Liu; Guanzhe Huang; Gulin Yan; Hanpeng Hu; Haonan Jia; Jiahao Gong; Jiaoren Wu; Jie Wu; Jie Yang; Junzhe Lin; Kaixiang Li; Lei Xia; Longlong Gu; Ming Li; Nie Hao; Ranchen Ming; Shaoliang Pang; Siqi Liu; Song Yuan; Tiancheng Cao; Wen Li; Wenqing He; Xu Zhao; Xuelin Zhang; Yanbo Yu; Yinmin Zhong; Yu Zhou; Yuanwei Liang; Yuanwei Lu; Yuxiang Yang; Zidong Yang; Zili Zhang; Binxing Jiao; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Xinhao Zhang; Yibo Zhu; Daxin Jiang; Shuchang Zhou; Chen Hu
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Large Audio-Language Models (LALMs) have significantly advanced intelligent human-computer interaction, yet their reliance on text-based outputs limits their ability to generate natural speech responses directly, hindering seamless audio interactions. To address this, we introduce Step-Audio-AQAA, a fully end-to-end LALM designed for Audio Query-Audio Answer (AQAA) tasks. The model integrates a dual-codebook audio tokenizer for linguistic and semantic feature extraction, a 130-billion-parameter backbone LLM and a neural vocoder for high-fidelity speech synthesis. Our post-training approach employs interleaved token-output of text and audio to enhance semantic coherence and combines Direct Preference Optimization (DPO) with model merge to improve performance. Evaluations on the StepEval-Audio-360 benchmark demonstrate that Step-Audio-AQAA excels especially in speech control, outperforming the state-of-art LALMs in key areas. This work contributes a promising solution for end-to-end LALMs and highlights the critical role of token-based vocoder in enhancing overall performance for AQAA tasks.
>
---
#### [replaced 003] In This Environment, As That Speaker: A Text-Driven Framework for Multi-Attribute Speech Conversion
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.07036v2](http://arxiv.org/pdf/2506.07036v2)**

> **作者:** Jiawei Jin; Zhihan Yang; Yixuan Zhou; Zhiyong Wu
>
> **备注:** Accepted by Interspeech2025
>
> **摘要:** We propose TES-VC (Text-driven Environment and Speaker controllable Voice Conversion), a text-driven voice conversion framework with independent control of speaker timbre and environmental acoustics. TES-VC processes simultaneous text inputs for target voice and environment, accurately generating speech matching described timbre/environment while preserving source content. Trained on synthetic data with decoupled vocal/environment features via latent diffusion modeling, our method eliminates interference between attributes. The Retrieval-Based Timbre Control (RBTC) module enables precise manipulation using abstract descriptions without paired data. Experiments confirm TES-VC effectively generates contextually appropriate speech in both timbre and environment with high content retention and superior controllability which demonstrates its potential for widespread applications.
>
---
#### [replaced 004] Impact of Frame Rates on Speech Tokenizer: A Case Study on Mandarin and English
- **分类: cs.CL; cs.AI; cs.SD; eess.AS; 68T10; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.17076v3](http://arxiv.org/pdf/2505.17076v3)**

> **作者:** Haoyang Zhang; Hexin Liu; Xiangyu Zhang; Qiquan Zhang; Yuchen Hu; Junqi Zhao; Fei Tian; Xuerui Yang; Leibny Paola Garcia; Eng Siong Chng
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** The speech tokenizer plays a crucial role in recent speech tasks, generally serving as a bridge between speech signals and language models. While low-frame-rate codecs are widely employed as speech tokenizers, the impact of frame rates on speech tokens remains underexplored. In this study, we investigate how varying frame rates affect speech tokenization by examining Mandarin and English, two typologically distinct languages. We encode speech at different frame rates and evaluate the resulting semantic tokens in the speech recognition task. Our findings reveal that frame rate variations influence speech tokenization differently for each language, highlighting the interplay between frame rates, phonetic density, and language-specific acoustic features. The results provide insights into optimizing frame rate selection for speech tokenizers, with implications for automatic speech recognition, text-to-speech, and other speech-related applications.
>
---
#### [replaced 005] Non-intrusive Speech Quality Assessment with Diffusion Models Trained on Clean Speech
- **分类: eess.AS; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2410.17834v2](http://arxiv.org/pdf/2410.17834v2)**

> **作者:** Danilo de Oliveira; Julius Richter; Jean-Marie Lemercier; Simon Welker; Timo Gerkmann
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Diffusion models have found great success in generating high quality, natural samples of speech, but their potential for density estimation for speech has so far remained largely unexplored. In this work, we leverage an unconditional diffusion model trained only on clean speech for the assessment of speech quality. We show that the quality of a speech utterance can be assessed by estimating the likelihood of a corresponding sample in the terminating Gaussian distribution, obtained via a deterministic noising process. The resulting method is purely unsupervised, trained only on clean speech, and therefore does not rely on annotations. Our diffusion-based approach leverages clean speech priors to assess quality based on how the input relates to the learned distribution of clean data. Our proposed log-likelihoods show promising results, correlating well with intrusive speech quality metrics and showing the best correlation with human scores in a listening experiment.
>
---
#### [replaced 006] Improving Acoustic Scene Classification with City Features
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.16862v2](http://arxiv.org/pdf/2503.16862v2)**

> **作者:** Yiqiang Cai; Yizhou Tan; Shengchen Li; Xi Shao; Mark D. Plumbley
>
> **摘要:** Acoustic scene recordings are often collected from a diverse range of cities. Most existing acoustic scene classification (ASC) approaches focus on identifying common acoustic scene patterns across cities to enhance generalization. However, the potential acoustic differences introduced by city-specific environmental and cultural factors are overlooked. In this paper, we hypothesize that the city-specific acoustic features are beneficial for the ASC task rather than being treated as noise or bias. To this end, we propose City2Scene, a novel framework that leverages city features to improve ASC. Unlike conventional approaches that may discard or suppress city information, City2Scene transfers the city-specific knowledge from pre-trained city classification models to scene classification model using knowledge distillation. We evaluate City2Scene on three datasets of DCASE Challenge Task 1, which include both scene and city labels. Experimental results demonstrate that city features provide valuable information for classifying scenes. By distilling city-specific knowledge, City2Scene effectively improves accuracy across a variety of lightweight CNN backbones, achieving competitive performance to the top-ranked solutions of DCASE Challenge in recent years.
>
---
#### [replaced 007] Autonomous Robotic Radio Source Localization via a Novel Gaussian Mixture Filtering Approach
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2503.10349v3](http://arxiv.org/pdf/2503.10349v3)**

> **作者:** Sukkeun Kim; Sangwoo Moon; Ivan Petrunin; Hyo-Sang Shin; Shehryar Khattak
>
> **摘要:** This study proposes a new Gaussian Mixture Filter (GMF) to improve the estimation performance for the autonomous robotic radio signal source search and localization problem in unknown environments. The proposed filter is first tested with a benchmark numerical problem to validate the performance with other state-of-the-practice approaches such as Particle Filter (PF) and Particle Gaussian Mixture (PGM) filters. Then the proposed approach is tested and compared against PF and PGM filters in real-world robotic field experiments to validate its impact for real-world applications. The considered real-world scenarios have partial observability with the range-only measurement and uncertainty with the measurement model. The results show that the proposed filter can handle this partial observability effectively whilst showing improved performance compared to PF, reducing the computation requirements while demonstrating improved robustness over compared techniques.
>
---
#### [replaced 008] Real-Time AIoT for UAV Antenna Interference Detection via Edge-Cloud Collaboration
- **分类: eess.SP; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.03055v2](http://arxiv.org/pdf/2412.03055v2)**

> **作者:** Jun Dong; Jintao Cheng; Jin Wu; Chengxi Zhang; Shunyi Zhao; Xiaoyu Tang
>
> **摘要:** In the fifth-generation (5G) era, eliminating communication interference sources is crucial for maintaining network performance. Interference often originates from unauthorized or malfunctioning antennas, and radio monitoring agencies must address numerous sources of such antennas annually. Unmanned aerial vehicles (UAVs) can improve inspection efficiency. However, the data transmission delay in the existing cloud-only (CO) artificial intelligence (AI) mode fails to meet the low latency requirements for real-time performance. Therefore, we propose a computer vision-based AI of Things (AIoT) system to detect antenna interference sources for UAVs. The system adopts an optimized edge-cloud collaboration (ECC+) mode, combining a keyframe selection algorithm (KSA), focusing on reducing end-to-end latency (E2EL) and ensuring reliable data transmission, which aligns with the core principles of ultra-reliable low-latency communication (URLLC). At the core of our approach is an end-to-end antenna localization scheme based on the tracking-by-detection (TBD) paradigm, including a detector (EdgeAnt) and a tracker (AntSort). EdgeAnt achieves state-of-the-art (SOTA) performance with a mean average precision (mAP) of 42.1% on our custom antenna interference source dataset, requiring only 3 million parameters and 14.7 GFLOPs. On the COCO dataset, EdgeAnt achieves 38.9% mAP with 5.4 GFLOPs. We deployed EdgeAnt on Jetson Xavier NX (TRT) and Raspberry Pi 4B (NCNN), achieving real-time inference speeds of 21.1 (1088) and 4.8 (640) frames per second (FPS), respectively. Compared with CO mode, the ECC+ mode reduces E2EL by 88.9%, increases accuracy by 28.2%. Additionally, the system offers excellent scalability for coordinated multiple UAVs inspections. The detector code is publicly available at https://github.com/SCNU-RISLAB/EdgeAnt.
>
---
