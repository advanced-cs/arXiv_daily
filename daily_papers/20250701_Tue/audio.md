# 音频 cs.SD;  eess.SP

- **最新发布 29 篇**

- **更新 12 篇**

## 最新发布

#### [new 001] Optical Waveguide-based Spider Web Enables Resilient Impact Detection and Localization
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于传感技术任务，旨在解决振动检测与定位问题。通过仿生光学波导结构实现鲁棒的冲击检测与定位。**

- **链接: [http://arxiv.org/pdf/2506.22472v1](http://arxiv.org/pdf/2506.22472v1)**

> **作者:** Dylan Wilson; Marco Pontin; Peter Walters; Perla Maiolino
>
> **摘要:** Spiders use their webs as multifunctional tools that enable capturing and localizing prey and more general environmental sensing through vibrations. Inspired by their biological function, we present a spider web-inspired optical waveguide system for resilient impulse detection and localization. The structure consists of six clear thermoplastic polyurethane (TPU) waveguides arranged radially and interconnected by a spiral TPU thread, mimicking orb spider webs. Light transmission losses, induced by vibrations, are measured via coupled LEDs and photo-diodes, allowing real-time detection. We systematically characterize individual waveguides, analyzing key parameters such as tension, impulse position, and break angle to optimize vibrational response. The complete system is validated through controlled experiments, revealing a 5 ms propagation delay in vibration transfer between adjacent radii, enhancing localization capabilities. We demonstrate a robust impulse detection and localization algorithm leveraging time delay analysis, achieving reliable event identification even in cases of sensor failure. This study highlights the potential of bioinspired optical waveguide structures for adaptive sensing, with applications in soft robotics, structural monitoring, and environmental sensing.
>
---
#### [new 002] WavShape: Information-Theoretic Speech Representation Learning for Fair and Privacy-Aware Audio Processing
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音表示学习任务，旨在解决语音嵌入中敏感信息泄露问题。通过信息论方法优化嵌入，减少敏感属性关联，同时保留任务相关信息。**

- **链接: [http://arxiv.org/pdf/2506.22789v1](http://arxiv.org/pdf/2506.22789v1)**

> **作者:** Oguzhan Baser; Ahmet Ege Tanriverdi; Kaan Kale; Sandeep P. Chinchali; Sriram Vishwanath
>
> **备注:** 5 pages, 4 figures, Published at The Proceedings of Interspeech 2025, code is available at http://www.github.com/UTAustin-SwarmLab/WavShape
>
> **摘要:** Speech embeddings often retain sensitive attributes such as speaker identity, accent, or demographic information, posing risks in biased model training and privacy leakage. We propose WavShape, an information-theoretic speech representation learning framework that optimizes embeddings for fairness and privacy while preserving task-relevant information. We leverage mutual information (MI) estimation using the Donsker-Varadhan formulation to guide an MI-based encoder that systematically filters sensitive attributes while maintaining speech content essential for downstream tasks. Experimental results on three known datasets show that WavShape reduces MI between embeddings and sensitive attributes by up to 81% while retaining 97% of task-relevant information. By integrating information theory with self-supervised speech models, this work advances the development of fair, privacy-aware, and resource-efficient speech systems.
>
---
#### [new 003] Emergent musical properties of a transformer under contrastive self-supervised learning
- **分类: cs.SD; cs.IR; cs.LG; eess.AS**

- **简介: 该论文研究音乐信息检索中的序列建模任务，探讨对比自监督学习与Transformer结合在局部任务中的表现，发现其能自发生成音乐特征。**

- **链接: [http://arxiv.org/pdf/2506.23873v1](http://arxiv.org/pdf/2506.23873v1)**

> **作者:** Yuexuan Kong; Gabriel Meseguer-Brocal; Vincent Lostanlen; Mathieu Lagrange; Romain Hennequin
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** In music information retrieval (MIR), contrastive self-supervised learning for general-purpose representation models is effective for global tasks such as automatic tagging. However, for local tasks such as chord estimation, it is widely assumed that contrastively trained general-purpose self-supervised models are inadequate and that more sophisticated SSL is necessary; e.g., masked modeling. Our paper challenges this assumption by revealing the potential of contrastive SSL paired with a transformer in local MIR tasks. We consider a lightweight vision transformer with one-dimensional patches in the time--frequency domain (ViT-1D) and train it with simple contrastive SSL through normalized temperature-scaled cross-entropy loss (NT-Xent). Although NT-Xent operates only over the class token, we observe that, potentially thanks to weight sharing, informative musical properties emerge in ViT-1D's sequence tokens. On global tasks, the temporal average of class and sequence tokens offers a performance increase compared to the class token alone, showing useful properties in the sequence tokens. On local tasks, sequence tokens perform unexpectedly well, despite not being specifically trained for. Furthermore, high-level musical features such as onsets emerge from layer-wise attention maps and self-similarity matrices show different layers capture different musical dimensions. Our paper does not focus on improving performance but advances the musical interpretation of transformers and sheds light on some overlooked abilities of contrastive SSL paired with transformers for sequence modeling in MIR.
>
---
#### [new 004] Enhancing Neural Audio Fingerprint Robustness to Audio Degradation for Music Identification
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐识别任务，旨在提升音频指纹在音频退化下的鲁棒性。通过改进自监督学习和评估不同度量学习方法，提出更有效的解决方案。**

- **链接: [http://arxiv.org/pdf/2506.22661v1](http://arxiv.org/pdf/2506.22661v1)**

> **作者:** R. O. Araz; G. Cortes-Sebastia; E. Molina; J. Serra; X. Serra; Y. Mitsufuji; D. Bogdanov
>
> **备注:** Accepted to ISMIR2025
>
> **摘要:** Audio fingerprinting (AFP) allows the identification of unknown audio content by extracting compact representations, termed audio fingerprints, that are designed to remain robust against common audio degradations. Neural AFP methods often employ metric learning, where representation quality is influenced by the nature of the supervision and the utilized loss function. However, recent work unrealistically simulates real-life audio degradation during training, resulting in sub-optimal supervision. Additionally, although several modern metric learning approaches have been proposed, current neural AFP methods continue to rely on the NT-Xent loss without exploring the recent advances or classical alternatives. In this work, we propose a series of best practices to enhance the self-supervision by leveraging musical signal properties and realistic room acoustics. We then present the first systematic evaluation of various metric learning approaches in the context of AFP, demonstrating that a self-supervised adaptation of the triplet loss yields superior performance. Our results also reveal that training with multiple positive samples per anchor has critically different effects across loss functions. Our approach is built upon these insights and achieves state-of-the-art performance on both a large, synthetically degraded dataset and a real-world dataset recorded using microphones in diverse music venues.
>
---
#### [new 005] A Self-Training Approach for Whisper to Enhance Long Dysarthric Speech Recognition
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升长篇构音障碍语音的识别效果。通过自训练方法增强Whisper模型，解决数据不足和语音不完整问题。**

- **链接: [http://arxiv.org/pdf/2506.22810v1](http://arxiv.org/pdf/2506.22810v1)**

> **作者:** Shiyao Wang; Jiaming Zhou; Shiwan Zhao; Yong Qin
>
> **备注:** accepted by Interspeech 2025
>
> **摘要:** Dysarthric speech recognition (DSR) enhances the accessibility of smart devices for dysarthric speakers with limited mobility. Previously, DSR research was constrained by the fact that existing datasets typically consisted of isolated words, command phrases, and a limited number of sentences spoken by a few individuals. This constrained research to command-interaction systems and speaker adaptation. The Speech Accessibility Project (SAP) changed this by releasing a large and diverse English dysarthric dataset, leading to the SAP Challenge to build speaker- and text-independent DSR systems. We enhanced the Whisper model's performance on long dysarthric speech via a novel self-training method. This method increased training data and adapted the model to handle potentially incomplete speech segments encountered during inference. Our system achieved second place in both Word Error Rate and Semantic Score in the SAP Challenge.
>
---
#### [new 006] From Large-scale Audio Tagging to Real-Time Explainable Emergency Vehicle Sirens Detection
- **分类: cs.SD; cs.AI; eess.AS; 68T07; E.1; H.1; I.2; I.5; J.2; K.4; C.4**

- **简介: 该论文属于音频事件检测任务，旨在解决应急车辆警报实时识别问题。通过优化轻量级模型E2PANNs，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.23437v1](http://arxiv.org/pdf/2506.23437v1)**

> **作者:** Stefano Giacomelli; Marco Giordano; Claudia Rinaldi; Fabio Graziosi
>
> **备注:** pre-print (submitted to the IEEE/ACM Transactions on Audio, Speech, and Language Processing)
>
> **摘要:** Accurate recognition of Emergency Vehicle (EV) sirens is critical for the integration of intelligent transportation systems, smart city monitoring systems, and autonomous driving technologies. Modern automatic solutions are limited by the lack of large scale, curated datasets and by the computational demands of state of the art sound event detection models. This work introduces E2PANNs (Efficient Emergency Pre trained Audio Neural Networks), a lightweight Convolutional Neural Network architecture derived from the PANNs framework, specifically optimized for binary EV siren detection. Leveraging our dedicated subset of AudioSet (AudioSet EV) we fine-tune and evaluate E2PANNs across multiple reference datasets and test its viability on embedded hardware. The experimental campaign includes ablation studies, cross-domain benchmarking, and real-time inference deployment on edge device. Interpretability analyses exploiting Guided Backpropagation and ScoreCAM algorithms provide insights into the model internal representations and validate its ability to capture distinct spectrotemporal patterns associated with different types of EV sirens. Real time performance is assessed through frame wise and event based detection metrics, as well as a detailed analysis of false positive activations. Results demonstrate that E2PANNs establish a new state of the art in this research domain, with high computational efficiency, and suitability for edge-based audio monitoring and safety-critical applications.
>
---
#### [new 007] You Sound a Little Tense: L2 Tailored Clear TTS Using Durational Vowel Properties
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音合成任务，旨在提升二语学习者的语音可懂度。通过调整元音时长设计清晰模式，减少转录错误，解决L2语音识别难题。**

- **链接: [http://arxiv.org/pdf/2506.23367v1](http://arxiv.org/pdf/2506.23367v1)**

> **作者:** Paige Tuttösí; H. Henny Yeung; Yue Wang; Jean-Julien Aucouturier; Angelica Lim
>
> **备注:** Accepted to ISCA Speech Synthesis Workshop, 2025
>
> **摘要:** We present the first text-to-speech (TTS) system tailored to second language (L2) speakers. We use duration differences between American English tense (longer) and lax (shorter) vowels to create a "clarity mode" for Matcha-TTS. Our perception studies showed that French-L1, English-L2 listeners had fewer (at least 9.15%) transcription errors when using our clarity mode, and found it more encouraging and respectful than overall slowed down speech. Remarkably, listeners were not aware of these effects: despite the decreased word error rate in clarity mode, listeners still believed that slowing all target words was the most intelligible, suggesting that actual intelligibility does not correlate with perceived intelligibility. Additionally, we found that Whisper-ASR did not use the same cues as L2 speakers to differentiate difficult vowels and is not sufficient to assess the intelligibility of TTS systems for these individuals.
>
---
#### [new 008] Scaling Self-Supervised Representation Learning for Symbolic Piano Performance
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文研究生成式Transformer模型在钢琴演奏符号数据上的自监督学习，解决音乐生成与分类问题，通过预训练和微调提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.23869v1](http://arxiv.org/pdf/2506.23869v1)**

> **作者:** Louis Bradshaw; Honglu Fan; Alexander Spangher; Stella Biderman; Simon Colton
>
> **备注:** ISMIR (2025)
>
> **摘要:** We study the capabilities of generative autoregressive transformer models trained on large amounts of symbolic solo-piano transcriptions. After first pretraining on approximately 60,000 hours of music, we use a comparatively smaller, high-quality subset, to finetune models to produce musical continuations, perform symbolic classification tasks, and produce general-purpose contrastive MIDI embeddings by adapting the SimCLR framework to symbolic music. When evaluating piano continuation coherence, our generative model outperforms leading symbolic generation techniques and remains competitive with proprietary audio generation models. On MIR classification benchmarks, frozen representations from our contrastive model achieve state-of-the-art results in linear probe experiments, while direct finetuning demonstrates the generalizability of pretrained representations, often requiring only a few hundred labeled examples to specialize to downstream tasks.
>
---
#### [new 009] XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音编码任务，旨在解决语义与声学信息平衡问题。提出XY-Tokenizer，通过多阶段学习实现语义与声学性能的兼顾。**

- **链接: [http://arxiv.org/pdf/2506.23325v1](http://arxiv.org/pdf/2506.23325v1)**

> **作者:** Yitian Gong; Luozhijie Jin; Ruifan Deng; Dong Zhang; Xin Zhang; Qinyuan Cheng; Zhaoye Fei; Shimin Li; Xipeng Qiu
>
> **摘要:** Speech codecs serve as bridges between speech signals and large language models. An ideal codec for speech language models should not only preserve acoustic information but also capture rich semantic information. However, existing speech codecs struggle to balance high-quality audio reconstruction with ease of modeling by language models. In this study, we analyze the limitations of previous codecs in balancing semantic richness and acoustic fidelity. We propose XY-Tokenizer, a novel codec that mitigates the conflict between semantic and acoustic capabilities through multi-stage, multi-task learning. Experimental results demonstrate that XY-Tokenizer achieves performance in both semantic and acoustic tasks comparable to that of state-of-the-art codecs operating at similar bitrates, even though those existing codecs typically excel in only one aspect. Specifically, XY-Tokenizer achieves strong text alignment, surpassing distillation-based semantic modeling methods such as SpeechTokenizer and Mimi, while maintaining a speaker similarity score of 0.83 between reconstructed and original audio. The reconstruction performance of XY-Tokenizer is comparable to that of BigCodec, the current state-of-the-art among acoustic-only codecs, which achieves a speaker similarity score of 0.84 at a similar bitrate. Code and models are available at https://github.com/gyt1145028706/XY-Tokenizer.
>
---
#### [new 010] SegmentAnyMuscle: A universal muscle segmentation model across different locations in MRI
- **分类: eess.SP; cs.CV**

- **简介: 该论文属于医学图像分割任务，旨在解决MRI中肌肉精准分割的问题。通过开发一个通用模型，实现跨部位和序列的自动化肌肉分割。**

- **链接: [http://arxiv.org/pdf/2506.22467v1](http://arxiv.org/pdf/2506.22467v1)**

> **作者:** Roy Colglazier; Jisoo Lee; Haoyu Dong; Hanxue Gu; Yaqian Chen; Joseph Cao; Zafer Yildiz; Zhonghao Liu; Nicholas Konz; Jichen Yang; Jikai Zhang; Yuwen Chen; Lin Li; Adrian Camarena; Maciej A. Mazurowski
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** The quantity and quality of muscles are increasingly recognized as important predictors of health outcomes. While MRI offers a valuable modality for such assessments, obtaining precise quantitative measurements of musculature remains challenging. This study aimed to develop a publicly available model for muscle segmentation in MRIs and demonstrate its applicability across various anatomical locations and imaging sequences. A total of 362 MRIs from 160 patients at a single tertiary center (Duke University Health System, 2016-2020) were included, with 316 MRIs from 114 patients used for model development. The model was tested on two separate sets: one with 28 MRIs representing common sequence types, achieving an average Dice Similarity Coefficient (DSC) of 88.45%, and another with 18 MRIs featuring less frequent sequences and abnormalities such as muscular atrophy, hardware, and significant noise, achieving 86.21% DSC. These results demonstrate the feasibility of a fully automated deep learning algorithm for segmenting muscles on MRI across diverse settings. The public release of this model enables consistent, reproducible research into the relationship between musculature and health.
>
---
#### [new 011] TOMI: Transforming and Organizing Music Ideas for Multi-Track Compositions with Full-Song Structure
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出TOMI模型，用于生成结构完整的多轨电子音乐，解决音乐创作中概念层次组织问题。**

- **链接: [http://arxiv.org/pdf/2506.23094v1](http://arxiv.org/pdf/2506.23094v1)**

> **作者:** Qi He; Gus Xia; Ziyu Wang
>
> **备注:** 9 pages, 4 figures, 2 tables. To be published in ISMIR 2025
>
> **摘要:** Hierarchical planning is a powerful approach to model long sequences structurally. Aside from considering hierarchies in the temporal structure of music, this paper explores an even more important aspect: concept hierarchy, which involves generating music ideas, transforming them, and ultimately organizing them--across musical time and space--into a complete composition. To this end, we introduce TOMI (Transforming and Organizing Music Ideas) as a novel approach in deep music generation and develop a TOMI-based model via instruction-tuned foundation LLM. Formally, we represent a multi-track composition process via a sparse, four-dimensional space characterized by clips (short audio or MIDI segments), sections (temporal positions), tracks (instrument layers), and transformations (elaboration methods). Our model is capable of generating multi-track electronic music with full-song structure, and we further integrate the TOMI-based model with the REAPER digital audio workstation, enabling interactive human-AI co-creation. Experimental results demonstrate that our approach produces higher-quality electronic music with stronger structural coherence compared to baselines.
>
---
#### [new 012] The Florence Price Art Song Dataset and Piano Accompaniment Generator
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在复原并生成符合Florence Price风格的钢琴伴奏。研究构建了她的艺术歌曲数据集，并训练了一个生成模型进行伴奏创作。**

- **链接: [http://arxiv.org/pdf/2506.23130v1](http://arxiv.org/pdf/2506.23130v1)**

> **作者:** Tao-Tao He; Martin E. Malandro; Douglas Shadle
>
> **备注:** 8 pages, 4 figures. To appear in the proceedings of ISMIR 2025
>
> **摘要:** Florence B. Price was a composer in the early 20th century whose music reflects her upbringing in the American South, her African heritage, and her Western classical training. She is noted as the first African-American woman to have a symphony performed by a major orchestra. Her music has recently received renewed attention from both the public and the research community, decades after her death. In addition to other genres, Price was a prolific composer for solo voice and piano. Music historians have documented the existence of 134 art songs and piano/voice arrangements for spirituals and folk songs written by Price. We release a digital catalog of 112 of these works in MuseScore, MusicXML, MIDI, and PDF format. We also use this dataset to fine-tune a symbolic music generation model to generate accompaniments to melodies, and we conduct a blind listening experiment that shows that accompaniments generated by our model are perceived as being reflective of Florence Price's style more frequently than accompaniments generated by a baseline model. We release our model as the Florence Price Piano Accompaniment Generator alongside our dataset.
>
---
#### [new 013] Privacy-aware IoT Fall Detection Services For Aging in Place
- **分类: eess.SP; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属于智能健康监护任务，旨在解决老年人跌倒检测中的隐私和数据不足问题。通过设计FDaaS框架和使用UWB传感器及FD-GPT模型实现高效检测。**

- **链接: [http://arxiv.org/pdf/2506.22462v1](http://arxiv.org/pdf/2506.22462v1)**

> **作者:** Abdallah Lakhdari; Jiajie Li; Amani Abusafia; Athman Bouguettaya
>
> **备注:** 11 pages, 12 figures, This paper is accepted in the 2025 IEEE International Conference on Web Services (ICWS 2025)
>
> **摘要:** Fall detection is critical to support the growing elderly population, projected to reach 2.1 billion by 2050. However, existing methods often face data scarcity challenges or compromise privacy. We propose a novel IoT-based Fall Detection as a Service (FDaaS) framework to assist the elderly in living independently and safely by accurately detecting falls. We design a service-oriented architecture that leverages Ultra-wideband (UWB) radar sensors as an IoT health-sensing service, ensuring privacy and minimal intrusion. We address the challenges of data scarcity by utilizing a Fall Detection Generative Pre-trained Transformer (FD-GPT) that uses augmentation techniques. We developed a protocol to collect a comprehensive dataset of the elderly daily activities and fall events. This resulted in a real dataset that carefully mimics the elderly's routine. We rigorously evaluate and compare various models using this dataset. Experimental results show our approach achieves 90.72% accuracy and 89.33% precision in distinguishing between fall events and regular activities of daily living.
>
---
#### [new 014] Efficient Interleaved Speech Modeling through Knowledge Distillation
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决大模型部署受限问题。通过知识蒸馏构建轻量模型TinyWave，在保持性能的同时减少3倍参数量，适用于多种语音应用。**

- **链接: [http://arxiv.org/pdf/2506.23670v1](http://arxiv.org/pdf/2506.23670v1)**

> **作者:** Mohammadmahdi Nouriborji; Morteza Rohanian
>
> **摘要:** Current speech language models exceed the size and latency constraints of many deployment environments. We build compact, expressive speech generation models through layer-aligned distillation, matching hidden states, attention maps, and softened logits to compress large multimodal transformers by 3x with minimal loss in performance. We introduce TinyWave, a family of 2B-parameter models for speech-to-speech and interleaved speech-text generation, trained on 50,000 hours of public audio. TinyWave supports (i) speech-only generation using phonetic or expressive tokens and (ii) mixed speech-text continuations. Evaluation on Libri-Light shows TinyWave within 1.4 normalized perplexity points of its teacher. Accuracy on spoken StoryCloze and SALMon reaches 93-97% of the teacher's performance, outperforming size-matched baselines. These models are optimized for deployment on commodity hardware, enabling applications in real-time conversational agents, assistive technologies, and low-resource environments. We release models, training code, and evaluation scripts to support reproducible research on compact, expressive speech generation.
>
---
#### [new 015] RELATE: Subjective evaluation dataset for automatic evaluation of relevance between text and audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于文本与音频相关性评估任务，旨在解决主观评价成本高、客观评价不明确的问题。构建了RELATE数据集，并提出一种自动预测模型。**

- **链接: [http://arxiv.org/pdf/2506.23582v1](http://arxiv.org/pdf/2506.23582v1)**

> **作者:** Yusuke Kanamori; Yuki Okamoto; Taisei Takano; Shinnosuke Takamichi; Yuki Saito; Hiroshi Saruwatari
>
> **备注:** Accepted to INTERSPEECH2025
>
> **摘要:** In text-to-audio (TTA) research, the relevance between input text and output audio is an important evaluation aspect. Traditionally, it has been evaluated from both subjective and objective perspectives. However, subjective evaluation is costly in terms of money and time, and objective evaluation is unclear regarding the correlation to subjective evaluation scores. In this study, we construct RELATE, an open-sourced dataset that subjectively evaluates the relevance. Also, we benchmark a model for automatically predicting the subjective evaluation score from synthesized audio. Our model outperforms a conventional CLAPScore model, and that trend extends to many sound categories.
>
---
#### [new 016] StreamFlow: Streaming Flow Matching with Block-wise Guided Attention Mask for Speech Token Decoding
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音生成任务，解决实时语音生成中的流式处理与音频质量问题。提出StreamFlow架构，采用分块注意力机制提升效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.23986v1](http://arxiv.org/pdf/2506.23986v1)**

> **作者:** Dake Guo; Jixun Yao; Linhan Ma; Wang He; Lei Xie
>
> **摘要:** Recent advancements in discrete token-based speech generation have highlighted the importance of token-to-waveform generation for audio quality, particularly in real-time interactions. Traditional frameworks integrating semantic tokens with flow matching (FM) struggle with streaming capabilities due to their reliance on a global receptive field. Additionally, directly implementing token-by-token streaming speech generation often results in degraded audio quality. To address these challenges, we propose StreamFlow, a novel neural architecture that facilitates streaming flow matching with diffusion transformers (DiT). To mitigate the long-sequence extrapolation issues arising from lengthy historical dependencies, we design a local block-wise receptive field strategy. Specifically, the sequence is first segmented into blocks, and we introduce block-wise attention masks that enable the current block to receive information from the previous or subsequent block. These attention masks are combined hierarchically across different DiT-blocks to regulate the receptive field of DiTs. Both subjective and objective experimental results demonstrate that our approach achieves performance comparable to non-streaming methods while surpassing other streaming methods in terms of speech quality, all the while effectively managing inference time during long-sequence generation. Furthermore, our method achieves a notable first-packet latency of only 180 ms.\footnote{Speech samples: https://dukguo.github.io/StreamFlow/}
>
---
#### [new 017] Evaluating Sound Similarity Metrics for Differentiable, Iterative Sound-Matching
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于声音匹配任务，旨在评估不同相似性度量在迭代声音匹配中的效果。通过实验分析不同损失函数与合成器的性能关系，探讨是否存在通用最优损失函数。**

- **链接: [http://arxiv.org/pdf/2506.22628v1](http://arxiv.org/pdf/2506.22628v1)**

> **作者:** Amir Salimi; Abram Hindle; Osmar R. Zaiane
>
> **摘要:** Manual sound design with a synthesizer is inherently iterative: an artist compares the synthesized output to a mental target, adjusts parameters, and repeats until satisfied. Iterative sound-matching automates this workflow by continually programming a synthesizer under the guidance of a loss function (or similarity measure) toward a target sound. Prior comparisons of loss functions have typically favored one metric over another, but only within narrow settings: limited synthesis methods, few loss types, often without blind listening tests. This leaves open the question of whether a universally optimal loss exists, or the choice of loss remains a creative decision conditioned on the synthesis method and the sound designer's preference. We propose differentiable iterative sound-matching as the natural extension of the available literature, since it combines the manual approach to sound design with modern advances in machine learning. To analyze the variability of loss function performance across synthesizers, we implemented a mix of four novel and established differentiable loss functions, and paired them with differentiable subtractive, additive, and AM synthesizers. For each of the sixteen synthesizer--loss combinations, we ran 300 randomized sound-matching trials. Performance was measured using parameter differences, spectrogram-distance metrics, and manually assigned listening scores. We observed a moderate level of consistency among the three performance measures. Our post-hoc analysis shows that the loss function performance is highly dependent on the synthesizer. These findings underscore the value of expanding the scope of sound-matching experiments and developing new similarity metrics tailored to specific synthesis techniques rather than pursuing one-size-fits-all solutions.
>
---
#### [new 018] Mind the Gap: Entity-Preserved Context-Aware ASR Structured Transcriptions
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决实体和数字识别不准的问题。通过扩展上下文窗口和优化实体分配，提升长文本的准确性和格式正确性。**

- **链接: [http://arxiv.org/pdf/2506.22858v1](http://arxiv.org/pdf/2506.22858v1)**

> **作者:** Duygu Altinok
>
> **备注:** This is the accepted version of an article accepted to the TSD 2025 conference, published in Springer Lecture Notes in Artificial Intelligence (LNAI). The final authenticated version is available online at SpringerLink
>
> **摘要:** Automatic Speech Recognition (ASR) systems, such as Whisper, achieve high transcription accuracy but struggle with named entities and numerical data, especially when proper formatting is required. These issues increase word error rate (WER) and impair semantic understanding in critical domains like legal, financial, and medical applications. We propose a novel training approach that extends the semantic context of ASR models by adding overlapping context windows during training. By sliding 5-second overlaps on both sides of 30-second chunks, we create a 40-second "effective semantic window," improving entity recognition and formatting while focusing predictions on the central 30 seconds. To address entities spanning chunk boundaries, we reassign such entities entirely to the right-hand chunk, ensuring proper formatting. Additionally, enriched training data with embedded entity labels enables the model to learn both recognition and type-specific formatting. Evaluated on the Spoken Wikipedia dataset, our method improves performance across semantic tasks, including named entity recognition (NER) and entity formatting. These results highlight the effectiveness of context-aware training in addressing ASR limitations for long-form transcription and complex entity recognition tasks.
>
---
#### [new 019] Less is More: Data Curation Matters in Scaling Speech Enhancement
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，解决大规模数据训练效果不佳的问题。通过数据筛选，证明高质量数据比数据量更重要。**

- **链接: [http://arxiv.org/pdf/2506.23859v1](http://arxiv.org/pdf/2506.23859v1)**

> **作者:** Chenda Li; Wangyou Zhang; Wei Wang; Robin Scheibler; Kohei Saijo; Samuele Cornell; Yihui Fu; Marvin Sach; Zhaoheng Ni; Anurag Kumar; Tim Fingscheidt; Shinji Watanabe; Yanmin Qian
>
> **备注:** Submitted to ASRU2025
>
> **摘要:** The vast majority of modern speech enhancement systems rely on data-driven neural network models. Conventionally, larger datasets are presumed to yield superior model performance, an observation empirically validated across numerous tasks in other domains. However, recent studies reveal diminishing returns when scaling speech enhancement data. We focus on a critical factor: prevalent quality issues in ``clean'' training labels within large-scale datasets. This work re-examines this phenomenon and demonstrates that, within large-scale training sets, prioritizing high-quality training data is more important than merely expanding the data volume. Experimental findings suggest that models trained on a carefully curated subset of 700 hours can outperform models trained on the 2,500-hour full dataset. This outcome highlights the crucial role of data curation in scaling speech enhancement systems effectively.
>
---
#### [new 020] URGENT-PK: Perceptually-Aligned Ranking Model Designed for Speech Enhancement Competition
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音增强任务，旨在解决MOS评估依赖人工标注的问题。提出URGENT-PK模型，通过配对比较进行系统排名，提升有限数据下的排序性能。**

- **链接: [http://arxiv.org/pdf/2506.23874v1](http://arxiv.org/pdf/2506.23874v1)**

> **作者:** Jiahe Wang; Chenda Li; Wei Wang; Wangyou Zhang; Samuele Cornell; Marvin Sach; Robin Scheibler; Kohei Saijo; Yihui Fu; Zhaoheng Ni; Anurag Kumar; Tim Fingscheidt; Shinji Watanabe; Yanmin Qian
>
> **备注:** Submitted to ASRU2025
>
> **摘要:** The Mean Opinion Score (MOS) is fundamental to speech quality assessment. However, its acquisition requires significant human annotation. Although deep neural network approaches, such as DNSMOS and UTMOS, have been developed to predict MOS to avoid this issue, they often suffer from insufficient training data. Recognizing that the comparison of speech enhancement (SE) systems prioritizes a reliable system comparison over absolute scores, we propose URGENT-PK, a novel ranking approach leveraging pairwise comparisons. URGENT-PK takes homologous enhanced speech pairs as input to predict relative quality rankings. This pairwise paradigm efficiently utilizes limited training data, as all pairwise permutations of multiple systems constitute a training instance. Experiments across multiple open test sets demonstrate URGENT-PK's superior system-level ranking performance over state-of-the-art baselines, despite its simple network architecture and limited training data.
>
---
#### [new 021] Human-CLAP: Human-perception-based contrastive language-audio pretraining
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频与文本匹配任务，旨在解决CLAPScore与人类主观评价相关性低的问题，通过引入人类感知优化模型，提升两者间的相关性。**

- **链接: [http://arxiv.org/pdf/2506.23553v1](http://arxiv.org/pdf/2506.23553v1)**

> **作者:** Taisei Takano; Yuki Okamoto; Yusuke Kanamori; Yuki Saito; Ryotaro Nagase; Hiroshi Saruwatari
>
> **摘要:** Contrastive language-audio pretraining (CLAP) is widely used for audio generation and recognition tasks. For example, CLAPScore, which utilizes the similarity of CLAP embeddings, has been a major metric for the evaluation of the relevance between audio and text in text-to-audio. However, the relationship between CLAPScore and human subjective evaluation scores is still unclarified. We show that CLAPScore has a low correlation with human subjective evaluation scores. Additionally, we propose a human-perception-based CLAP called Human-CLAP by training a contrastive language-audio model using the subjective evaluation score. In our experiments, the results indicate that our Human-CLAP improved the Spearman's rank correlation coefficient (SRCC) between the CLAPScore and the subjective evaluation scores by more than 0.25 compared with the conventional CLAP.
>
---
#### [new 022] Unsupervised Discovery of Behavioral Primitives from Sensorimotor Dynamic Functional Connectivity
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于行为基元发现任务，旨在从传感器-运动数据中解析动态功能连接，识别运动模块与协同机制。**

- **链接: [http://arxiv.org/pdf/2506.22473v1](http://arxiv.org/pdf/2506.22473v1)**

> **作者:** Fernando Diaz Ledezma; Valentin Marcel; Matej Hoffmann
>
> **备注:** 8 pages with 6 figures
>
> **摘要:** The movements of both animals and robots give rise to streams of high-dimensional motor and sensory information. Imagine the brain of a newborn or the controller of a baby humanoid robot trying to make sense of unprocessed sensorimotor time series. Here, we present a framework for studying the dynamic functional connectivity between the multimodal sensory signals of a robotic agent to uncover an underlying structure. Using instantaneous mutual information, we capture the time-varying functional connectivity (FC) between proprioceptive, tactile, and visual signals, revealing the sensorimotor relationships. Using an infinite relational model, we identified sensorimotor modules and their evolving connectivity. To further interpret these dynamic interactions, we employed non-negative matrix factorization, which decomposed the connectivity patterns into additive factors and their corresponding temporal coefficients. These factors can be considered the agent's motion primitives or movement synergies that the agent can use to make sense of its sensorimotor space and later for behavior selection. In the future, the method can be deployed in robot learning as well as in the analysis of human movement trajectories or brain signals.
>
---
#### [new 023] Investigating an Overfitting and Degeneration Phenomenon in Self-Supervised Multi-Pitch Estimation
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于多音高估计任务，旨在解决自监督学习中过拟合与退化问题。通过结合自监督目标提升模型性能，但发现模型在监督数据上过拟合，在自监督数据上退化。**

- **链接: [http://arxiv.org/pdf/2506.23371v1](http://arxiv.org/pdf/2506.23371v1)**

> **作者:** Frank Cwitkowitz; Zhiyao Duan
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** Multi-Pitch Estimation (MPE) continues to be a sought after capability of Music Information Retrieval (MIR) systems, and is critical for many applications and downstream tasks involving pitch, including music transcription. However, existing methods are largely based on supervised learning, and there are significant challenges in collecting annotated data for the task. Recently, self-supervised techniques exploiting intrinsic properties of pitch and harmonic signals have shown promise for both monophonic and polyphonic pitch estimation, but these still remain inferior to supervised methods. In this work, we extend the classic supervised MPE paradigm by incorporating several self-supervised objectives based on pitch-invariant and pitch-equivariant properties. This joint training results in a substantial improvement under closed training conditions, which naturally suggests that applying the same objectives to a broader collection of data will yield further improvements. However, in doing so we uncover a phenomenon whereby our model simultaneously overfits to the supervised data while degenerating on data used for self-supervision only. We demonstrate and investigate this and offer our insights on the underlying problem.
>
---
#### [new 024] JAM-Flow: Joint Audio-Motion Synthesis with Flow Matching
- **分类: cs.CV; cs.SD; eess.AS**

- **简介: 该论文提出JAM-Flow，解决跨模态生成任务中的语音与面部动作同步问题，通过联合建模实现音频和视觉的统一生成。**

- **链接: [http://arxiv.org/pdf/2506.23552v1](http://arxiv.org/pdf/2506.23552v1)**

> **作者:** Mingi Kwon; Joonghyuk Shin; Jaeseok Jung; Jaesik Park; Youngjung Uh
>
> **备注:** project page: https://joonghyuk.com/jamflow-web Under review. Preprint published on arXiv
>
> **摘要:** The intrinsic link between facial motion and speech is often overlooked in generative modeling, where talking head synthesis and text-to-speech (TTS) are typically addressed as separate tasks. This paper introduces JAM-Flow, a unified framework to simultaneously synthesize and condition on both facial motion and speech. Our approach leverages flow matching and a novel Multi-Modal Diffusion Transformer (MM-DiT) architecture, integrating specialized Motion-DiT and Audio-DiT modules. These are coupled via selective joint attention layers and incorporate key architectural choices, such as temporally aligned positional embeddings and localized joint attention masking, to enable effective cross-modal interaction while preserving modality-specific strengths. Trained with an inpainting-style objective, JAM-Flow supports a wide array of conditioning inputs-including text, reference audio, and reference motion-facilitating tasks such as synchronized talking head generation from text, audio-driven animation, and much more, within a single, coherent model. JAM-Flow significantly advances multi-modal generative modeling by providing a practical solution for holistic audio-visual synthesis. project page: https://joonghyuk.com/jamflow-web
>
---
#### [new 025] VisionScores -- A system-segmented image score dataset for deep learning tasks
- **分类: cs.CV; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文提出VisionScores数据集，用于深度学习任务。旨在解决音乐乐谱图像分析问题，通过构建结构丰富的图像数据集，支持不同作曲家和作品类型的分析。**

- **链接: [http://arxiv.org/pdf/2506.23030v1](http://arxiv.org/pdf/2506.23030v1)**

> **作者:** Alejandro Romero Amezcua; Mariano José Juan Rivera Meraz
>
> **备注:** Comments: 5 pages, 3 figures. Accepted for presentation at the 2025 IEEE International Conference on Image Processing (ICIP). \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for any other use
>
> **摘要:** VisionScores presents a novel proposal being the first system-segmented image score dataset, aiming to offer structure-rich, high information-density images for machine and deep learning tasks. Delimited to two-handed piano pieces, it was built to consider not only certain graphic similarity but also composition patterns, as this creative process is highly instrument-dependent. It provides two scenarios in relation to composer and composition type. The first, formed by 14k samples, considers works from different authors but the same composition type, specifically, Sonatinas. The latter, consisting of 10.8K samples, presents the opposite case, various composition types from the same author, being the one selected Franz Liszt. All of the 24.8k samples are formatted as grayscale jpg images of $128 \times 512$ pixels. VisionScores supplies the users not only the formatted samples but the systems' order and pieces' metadata. Moreover, unsegmented full-page scores and the pre-formatted images are included for further analysis.
>
---
#### [new 026] Feasibility of spectral-element modeling of wave propagation through the anatomy of marine mammals
- **分类: cs.CE; cs.SD; eess.AS; q-bio.TO**

- **简介: 该论文属于生物声学仿真任务，旨在解决海洋哺乳动物头部超声波传播模拟问题。采用SEM方法克服FEM局限，构建详细解剖模型进行仿真研究。**

- **链接: [http://arxiv.org/pdf/2506.22944v1](http://arxiv.org/pdf/2506.22944v1)**

> **作者:** Carlos García A.; Vladimiro Boselli; Aida Hejazi Nooghabi; Andrea Colombi; Lapo Boschi
>
> **摘要:** This study introduces the first 3D spectral-element method (SEM) simulation of ultrasonic wave propagation in a bottlenose dolphin (Tursiops truncatus) head. Unlike traditional finite-element methods (FEM), which struggle with high-frequency simulations due to costly linear-system inversions and slower convergence, SEM offers exponential convergence and efficient parallel computation. Using Computed Tomography (CT) scan data, we developed a detailed hexahedral mesh capturing complex anatomical features, such as acoustic fats and jaws. Our simulations of plane and spherical waves confirm SEM's effectiveness for ultrasonic time-domain modeling. This approach opens new avenues for marine biology, contributing to research in echolocation, the impacts of anthropogenic marine noise pollution and the biophysics of hearing and click generation in marine mammals. By overcoming FEM's limitations, SEM provides a powerful scalable tool to test hypotheses about dolphin bioacoustics, with significant implications for conservation and understanding marine mammal auditory systems under increasing environmental challenges.
>
---
#### [new 027] Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，旨在提升CTC模型的语义建模能力。通过引入LLM辅助的中间损失函数，增强语言建模同时保持高效解码。**

- **链接: [http://arxiv.org/pdf/2506.22846v1](http://arxiv.org/pdf/2506.22846v1)**

> **作者:** Duygu Altinok
>
> **备注:** This is the accepted version of an article accepted to the TSD 2025 conference, published in Springer Lecture Notes in Artificial Intelligence (LNAI). The final authenticated version is available online at SpringerLink
>
> **摘要:** End-to-end (E2E) automatic speech recognition (ASR) systems have revolutionized the field by integrating all components into a single neural network, with attention-based encoder-decoder models achieving state-of-the-art performance. However, their autoregressive decoding process limits inference speed, making them unsuitable for real-time applications. In contrast, CTC-based models offer faster, non-autoregressive decoding but struggle to model linguistic dependencies effectively. Addressing this challenge, we propose a novel auxiliary loss framework called Language-Aware Intermediate Loss (LAIL) to enhance CTC-based ASR using the linguistic knowledge of large language models (LLMs). By attaching connector layers to intermediate encoder layers, LAIL maps outputs to the embedding space of an LLM and computes a causal language modeling loss during training. This approach enhances linguistic modeling while preserving the computational efficiency of CTC decoding. Using the Conformer architecture and various LLaMA models, we demonstrate significant improvements in Word Error Rate (WER) on the LibriSpeech, TEDLIUM2, and WSJ corpora, achieving state-of-the-art performance for CTC-based ASR with minimal computational overhead.
>
---
#### [new 028] AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks
- **分类: cs.AI; cs.CL; cs.SD; eess.AS; 68T42, 68T50,; I.2.7; I.2.11; H.5.5**

- **简介: 该论文提出AURA，一个用于语音任务的智能代理，解决多轮对话中工具使用与推理的问题，通过集成ASR、TTS和LLMs实现复杂任务处理。**

- **链接: [http://arxiv.org/pdf/2506.23049v1](http://arxiv.org/pdf/2506.23049v1)**

> **作者:** Leander Melroy Maben; Gayathri Ganesh Lakshmy; Srijith Radhakrishnan; Siddhant Arora; Shinji Watanabe
>
> **摘要:** Despite advances in language and speech technologies, no open-source system enables full speech-to-speech, multi-turn dialogue with integrated tool use and agentic reasoning. We introduce AURA (Agent for Understanding, Reasoning, and Automated Tool Use), the first open-source, speech-native assistant capable of completing complex, goal-driven tasks through dynamic tool invocation and multi-turn conversation. AURA combines open-weight ASR, TTS, and LLMs in a cascaded pipeline and supports tools such as calendar booking, contact lookup, web search, and email. Its modular design allows easy integration of new tools using natural language prompts and action classes. On VoiceBench, AURA scores 92.75% on OpenBookQA-outperforming all open-weight systems and nearing GPT-4o-and 4.39 on AlpacaEval, competitive with other open-weight systems. Human evaluation shows 90% task success on complex, multi-turn speech tasks.
>
---
#### [new 029] Speaker Targeting via Self-Speaker Adaptation for Multi-talker ASR
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于多说话人语音识别任务，解决严重语音重叠下的目标说话人识别问题。通过自适应方法动态调整ASR模型，无需显式 speaker query。**

- **链接: [http://arxiv.org/pdf/2506.22646v1](http://arxiv.org/pdf/2506.22646v1)**

> **作者:** Weiqing Wang; Taejin Park; Ivan Medennikov; Jinhan Wang; Kunal Dhawan; He Huang; Nithin Rao Koluguri; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted by INTERSPEECH 2025
>
> **摘要:** We propose a self-speaker adaptation method for streaming multi-talker automatic speech recognition (ASR) that eliminates the need for explicit speaker queries. Unlike conventional approaches requiring target speaker embeddings or enrollment audio, our technique dynamically adapts individual ASR instances through speaker-wise speech activity prediction. The key innovation involves injecting speaker-specific kernels generated via speaker supervision activations into selected ASR encoder layers. This enables instantaneous speaker adaptation to target speakers while handling fully overlapped speech even in a streaming scenario. Experiments show state-of-the-art performance in both offline and streaming scenarios, demonstrating that our self-adaptive method effectively addresses severe speech overlap through streamlined speaker-focused recognition. The results validate the proposed self-speaker adaptation approach as a robust solution for multi-talker ASR under severe overlapping speech conditions.
>
---
## 更新

#### [replaced 001] From Alignment to Advancement: Bootstrapping Audio-Language Alignment with Synthetic Data
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.20166v2](http://arxiv.org/pdf/2505.20166v2)**

> **作者:** Chun-Yi Kuan; Hung-yi Lee
>
> **备注:** Submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing. Project Website: https://kuan2jiu99.github.io/Balsa
>
> **摘要:** Audio-aware large language models (ALLMs) have recently made great strides in understanding and processing audio inputs. These models are typically adapted from text-based large language models (LLMs) through additional training on audio-related tasks. However, this adaptation process presents two major limitations. First, ALLMs often suffer from catastrophic forgetting, where crucial textual capabilities like instruction-following are lost after training on audio data. In some cases, models may even hallucinate sounds that are not present in the input audio, raising concerns about reliability. Second, achieving cross-modal alignment between audio and language typically relies on large collections of task-specific question-answer pairs for instruction tuning, making it resource-intensive. To address these issues, previous works have leveraged the backbone LLMs to synthesize general-purpose, caption-style alignment data. In this paper, we propose a data generation framework that produces contrastive-like training data, designed to enhance ALLMs' ability to differentiate between present and absent sounds. We further extend our approach to multi-audio scenarios, enabling the model to either explain differences between audio inputs or produce unified captions that describe all inputs, thereby enhancing audio-language alignment. We refer to the entire ALLM training framework as bootstrapping audio-language alignment via synthetic data generation from backbone LLMs (BALSa). Experimental results indicate that our method effectively mitigates audio hallucinations while reliably maintaining strong performance on audio understanding and reasoning benchmarks, as well as instruction-following skills. Moreover, incorporating multi-audio training further enhances the model's comprehension and reasoning capabilities. Overall, BALSa offers an efficient and scalable approach to developing ALLMs.
>
---
#### [replaced 002] NatureLM-audio: an Audio-Language Foundation Model for Bioacoustics
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.07186v2](http://arxiv.org/pdf/2411.07186v2)**

> **作者:** David Robinson; Marius Miron; Masato Hagiwara; Benno Weck; Sara Keen; Milad Alizadeh; Gagan Narula; Matthieu Geist; Olivier Pietquin
>
> **备注:** Demo page: https://earthspecies.github.io/naturelm-audio-demo/
>
> **摘要:** Large language models (LLMs) prompted with text and audio have achieved state-of-the-art performance across various auditory tasks, including speech, music, and general audio, showing emergent abilities on unseen tasks. However, their potential has yet to be fully demonstrated in bioacoustics tasks, such as detecting animal vocalizations in large recordings, classifying rare and endangered species, and labeling context and behavior -- tasks that are crucial for conservation, biodiversity monitoring, and animal behavior studies. In this work, we present NatureLM-audio, the first audio-language foundation model specifically designed for bioacoustics. Our training dataset consists of carefully curated text-audio pairs spanning bioacoustics, speech, and music, designed to address the field's limited availability of annotated data. We demonstrate successful transfer of learned representations from music and speech to bioacoustics, and our model shows promising generalization to unseen taxa and tasks. We evaluate NatureLM-audio on a novel benchmark (BEANS-Zero) and it sets a new state of the art on several bioacoustics tasks, including zero-shot classification of unseen species. To advance bioacoustics research, we release our model weights, benchmark data, and open-source the code for training and benchmark data generation and model training.
>
---
#### [replaced 003] Video-Guided Text-to-Music Generation Using Public Domain Movie Collections
- **分类: cs.SD; cs.MM; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.12573v3](http://arxiv.org/pdf/2506.12573v3)**

> **作者:** Haven Kim; Zachary Novack; Weihan Xu; Julian McAuley; Hao-Wen Dong
>
> **备注:** ISMIR 2025 regular paper. Dataset, code, and demo available at https://havenpersona.github.io/ossl-v1
>
> **摘要:** Despite recent advancements in music generation systems, their application in film production remains limited, as they struggle to capture the nuances of real-world filmmaking, where filmmakers consider multiple factors-such as visual content, dialogue, and emotional tone-when selecting or composing music for a scene. This limitation primarily stems from the absence of comprehensive datasets that integrate these elements. To address this gap, we introduce Open Screen Soundtrack Library (OSSL), a dataset consisting of movie clips from public domain films, totaling approximately 36.5 hours, paired with high-quality soundtracks and human-annotated mood information. To demonstrate the effectiveness of our dataset in improving the performance of pre-trained models on film music generation tasks, we introduce a new video adapter that enhances an autoregressive transformer-based text-to-music model by adding video-based conditioning. Our experimental results demonstrate that our proposed approach effectively enhances MusicGen-Medium in terms of both objective measures of distributional and paired fidelity, and subjective compatibility in mood and genre. To facilitate reproducibility and foster future work, we publicly release the dataset, code, and demo.
>
---
#### [replaced 004] AI-Generated Song Detection via Lyrics Transcripts
- **分类: cs.SD; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.18488v2](http://arxiv.org/pdf/2506.18488v2)**

> **作者:** Markus Frohmann; Elena V. Epure; Gabriel Meseguer-Brocal; Markus Schedl; Romain Hennequin
>
> **备注:** Accepted to ISMIR 2025
>
> **摘要:** The recent rise in capabilities of AI-based music generation tools has created an upheaval in the music industry, necessitating the creation of accurate methods to detect such AI-generated content. This can be done using audio-based detectors; however, it has been shown that they struggle to generalize to unseen generators or when the audio is perturbed. Furthermore, recent work used accurate and cleanly formatted lyrics sourced from a lyrics provider database to detect AI-generated music. However, in practice, such perfect lyrics are not available (only the audio is); this leaves a substantial gap in applicability in real-life use cases. In this work, we instead propose solving this gap by transcribing songs using general automatic speech recognition (ASR) models. We do this using several detectors. The results on diverse, multi-genre, and multi-lingual lyrics show generally strong detection performance across languages and genres, particularly for our best-performing model using Whisper large-v2 and LLM2Vec embeddings. In addition, we show that our method is more robust than state-of-the-art audio-based ones when the audio is perturbed in different ways and when evaluated on different music generators. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [replaced 005] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.21448v2](http://arxiv.org/pdf/2506.21448v2)**

> **作者:** Huadai Liu; Jialei Wang; Kaicheng Luo; Wen Wang; Qian Chen; Zhou Zhao; Wei Xue
>
> **摘要:** While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, such generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present ThinkSound, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce AudioCoT, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics and excels in out-of-distribution Movie Gen Audio benchmark. The demo page is available at https://ThinkSound-Project.github.io.
>
---
#### [replaced 006] Spotlight-TTS: Spotlighting the Style via Voiced-Aware Style Extraction and Style Direction Adjustment for Expressive Text-to-Speech
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.20868v2](http://arxiv.org/pdf/2505.20868v2)**

> **作者:** Nam-Gyu Kim; Deok-Hyeon Cho; Seung-Bin Kim; Seong-Whan Lee
>
> **备注:** Proceedings of Interspeech 2025
>
> **摘要:** Recent advances in expressive text-to-speech (TTS) have introduced diverse methods based on style embedding extracted from reference speech. However, synthesizing high-quality expressive speech remains challenging. We propose Spotlight-TTS, which exclusively emphasizes style via voiced-aware style extraction and style direction adjustment. Voiced-aware style extraction focuses on voiced regions highly related to style while maintaining continuity across different speech regions to improve expressiveness. We adjust the direction of the extracted style for optimal integration into the TTS model, which improves speech quality. Experimental results demonstrate that Spotlight-TTS achieves superior performance compared to baseline models in terms of expressiveness, overall speech quality, and style transfer capability. Our audio samples are publicly available.
>
---
#### [replaced 007] Acousto-optic reconstruction of exterior sound field based on concentric circle sampling with circular harmonic expansion
- **分类: cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2311.01715v3](http://arxiv.org/pdf/2311.01715v3)**

> **作者:** Phuc Duc Nguyen; Kenji Ishikawa; Noboru Harada; Takehiro Moriya
>
> **备注:** Published in IEEE Transactions on Instrumentation and Measurement, Volume 74, 09 June 2025, Article Sequence Number: 4511312,
>
> **摘要:** Acousto-optic sensing provides an alternative approach to traditional microphone arrays by shedding light on the interaction of light with an acoustic field. Sound field reconstruction is a fascinating and advanced technique used in acousto-optics sensing. Current challenges in sound-field reconstruction methods pertain to scenarios in which the sound source is located within the reconstruction area, known as the exterior problem. Existing reconstruction algorithms, primarily designed for interior scenarios, often exhibit suboptimal performance when applied to exterior cases. This paper introduces a novel technique for exterior sound-field reconstruction. The proposed method leverages concentric circle sampling and a two-dimensional exterior sound-field reconstruction approach based on circular harmonic extensions. To evaluate the efficacy of this approach, both numerical simulations and practical experiments are conducted. The results highlight the superior accuracy of the proposed method when compared to conventional reconstruction methods, all while utilizing a minimal amount of measured projection data.
>
---
#### [replaced 008] FreeCodec: A disentangled neural speech codec with fewer tokens
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2412.01053v3](http://arxiv.org/pdf/2412.01053v3)**

> **作者:** Youqiang Zheng; Weiping Tu; Yueteng Kang; Jie Chen; Yike Zhang; Li Xiao; Yuhong Yang; Long Ma
>
> **备注:** 5 pages, 2 figures, 3 tables.Code and Demo page:https://github.com/exercise-book-yq/FreeCodec. Accepted to Interspeech 2025
>
> **摘要:** Neural speech codecs have gained great attention for their outstanding reconstruction with discrete token representations. It is a crucial component in generative tasks such as speech coding and large language models (LLM). However, most works based on residual vector quantization perform worse with fewer tokens due to low coding efficiency for modeling complex coupled information. In this paper, we propose a neural speech codec named FreeCodec which employs a more effective encoding framework by decomposing intrinsic properties of speech into different components: 1) a global vector is extracted as the timbre information, 2) a prosody encoder with a long stride level is used to model the prosody information, 3) the content information is from a content encoder. Using different training strategies, FreeCodec achieves state-of-the-art performance in reconstruction and disentanglement scenarios. Results from subjective and objective experiments demonstrate that our framework outperforms existing methods.
>
---
#### [replaced 009] MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.19577v3](http://arxiv.org/pdf/2505.19577v3)**

> **作者:** Yu Xi; Haoyu Li; Xiaoyu Gu; Yidi Jiang; Kai Yu
>
> **备注:** Accepted by TASLP
>
> **摘要:** Keyword spotting (KWS) is essential for voice-driven applications, demanding both accuracy and efficiency. Traditional ASR-based KWS methods, such as greedy and beam search, explore the entire search space without explicitly prioritizing keyword detection, often leading to suboptimal performance. In this paper, we propose an effective keyword-specific KWS framework by introducing a streaming-oriented CTC-Transducer-combined frame-asynchronous system with multi-head frame-asynchronous decoding (MFA-KWS). Specifically, MFA-KWS employs keyword-specific phone-synchronous decoding for CTC and replaces conventional RNN-T with Token-and-Duration Transducer to enhance both performance and efficiency. Furthermore, we explore various score fusion strategies, including single-frame-based and consistency-based methods. Extensive experiments demonstrate the superior performance of MFA-KWS, which achieves state-of-the-art results on both fixed keyword and arbitrary keywords datasets, such as Snips, MobvoiHotwords, and LibriKWS-20, while exhibiting strong robustness in noisy environments. Among fusion strategies, the consistency-based CDC-Last method delivers the best performance. Additionally, MFA-KWS achieves a 47% to 63% speed-up over the frame-synchronous baselines across various datasets. Extensive experimental results confirm that MFA-KWS is an effective and efficient KWS framework, making it well-suited for on-device deployment.
>
---
#### [replaced 010] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.15981v2](http://arxiv.org/pdf/2506.15981v2)**

> **作者:** Markus Frohmann; Gabriel Meseguer-Brocal; Markus Schedl; Elena V. Epure
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** The rapid advancement of AI-based music generation tools is revolutionizing the music industry but also posing challenges to artists, copyright holders, and providers alike. This necessitates reliable methods for detecting such AI-generated content. However, existing detectors, relying on either audio or lyrics, face key practical limitations: audio-based detectors fail to generalize to new or unseen generators and are vulnerable to audio perturbations; lyrics-based methods require cleanly formatted and accurate lyrics, unavailable in practice. To overcome these limitations, we propose a novel, practically grounded approach: a multimodal, modular late-fusion pipeline that combines automatically transcribed sung lyrics and speech features capturing lyrics-related information within the audio. By relying on lyrical aspects directly from audio, our method enhances robustness, mitigates susceptibility to low-level artifacts, and enables practical applicability. Experiments show that our method, DE-detect, outperforms existing lyrics-based detectors while also being more robust to audio perturbations. Thus, it offers an effective, robust solution for detecting AI-generated music in real-world scenarios. Our code is available at https://github.com/deezer/robust-AI-lyrics-detection.
>
---
#### [replaced 011] CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.12285v2](http://arxiv.org/pdf/2506.12285v2)**

> **作者:** Yinghao Ma; Siyou Li; Juntao Yu; Emmanouil Benetos; Akira Maezawa
>
> **备注:** Accepted by ISMIR 2025
>
> **摘要:** Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multi-choice evaluations that fail to reflect the complexity of real-world music analysis. We reinterpret a broad range of traditional MIR annotations as instruction-following formats and introduce CMI-Bench, a comprehensive music instruction following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. These include genre classification, emotion regression, emotion tagging, instrument classification, pitch estimation, key detection, lyrics transcription, melody extraction, vocal technique recognition, instrument performance technique detection, music tagging, music captioning, and (down)beat tracking: reflecting core challenges in MIR research. Unlike previous benchmarks, CMI-Bench adopts standardized evaluation metrics consistent with previous state-of-the-art MIR models, ensuring direct comparability with supervised approaches. We provide an evaluation toolkit supporting all open-source audio-textual LLMs, including LTU, Qwen-audio, SALMONN, MusiLingo, etc. Experiment results reveal significant performance gaps between LLMs and supervised models, along with their culture, chronological and gender bias, highlighting the potential and limitations of current models in addressing MIR tasks. CMI-Bench establishes a unified foundation for evaluating music instruction following, driving progress in music-aware LLMs.
>
---
#### [replaced 012] METEOR: Melody-aware Texture-controllable Symbolic Orchestral Music Generation via Transformer VAE
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.11753v3](http://arxiv.org/pdf/2409.11753v3)**

> **作者:** Dinh-Viet-Toan Le; Yi-Hsuan Yang
>
> **备注:** Accepted to 34rd International Joint Conference on Artificial Intelligence (IJCAI 2025) - AI, Arts and Creativity Special Track. Demo: https://dinhviettoanle.github.io/meteor
>
> **摘要:** Re-orchestration is the process of adapting a music piece for a different set of instruments. By altering the original instrumentation, the orchestrator often modifies the musical texture while preserving a recognizable melodic line and ensures that each part is playable within the technical and expressive capabilities of the chosen instruments. In this work, we propose METEOR, a model for generating Melody-aware Texture-controllable re-Orchestration with a Transformer-based variational auto-encoder (VAE). This model performs symbolic instrumental and textural music style transfers with a focus on melodic fidelity and controllability. We allow bar- and track-level controllability of the accompaniment with various textural attributes while keeping a homophonic texture. With both subjective and objective evaluations, we show that our model outperforms style transfer models on a re-orchestration task in terms of generation quality and controllability. Moreover, it can be adapted for a lead sheet orchestration task as a zero-shot learning model, achieving performance comparable to a model specifically trained for this task.
>
---
