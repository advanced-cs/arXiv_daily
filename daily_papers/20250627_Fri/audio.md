# 音频 cs.SD;  eess.SP

- **最新发布 15 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] A Multi-Stage Framework for Multimodal Controllable Speech Synthesis
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，旨在解决多模态控制下的语音生成问题。通过提出三阶段框架，提升生成语音的质量与多样性。**

- **链接: [http://arxiv.org/pdf/2506.20945v1](http://arxiv.org/pdf/2506.20945v1)**

> **作者:** Rui Niu; Weihao Wu; Jie Chen; Long Ma; Zhiyong Wu
>
> **备注:** Accepted by ICME2025
>
> **摘要:** Controllable speech synthesis aims to control the style of generated speech using reference input, which can be of various modalities. Existing face-based methods struggle with robustness and generalization due to data quality constraints, while text prompt methods offer limited diversity and fine-grained control. Although multimodal approaches aim to integrate various modalities, their reliance on fully matched training data significantly constrains their performance and applicability. This paper proposes a 3-stage multimodal controllable speech synthesis framework to address these challenges. For face encoder, we use supervised learning and knowledge distillation to tackle generalization issues. Furthermore, the text encoder is trained on both text-face and text-speech data to enhance the diversity of the generated speech. Experimental results demonstrate that this method outperforms single-modal baseline methods in both face based and text prompt based speech synthesis, highlighting its effectiveness in generating high-quality speech.
>
---
#### [new 002] Exploring Adapter Design Tradeoffs for Low Resource Music Generation
- **分类: cs.SD; cs.AI; cs.CL; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于音乐生成任务，旨在探索低资源环境下适配器设计的权衡。通过研究不同适配器配置，分析其在音乐细节和长程依赖上的表现，以优化模型性能与资源消耗。**

- **链接: [http://arxiv.org/pdf/2506.21298v1](http://arxiv.org/pdf/2506.21298v1)**

> **作者:** Atharva Mehta; Shivam Chauhan; Monojit Choudhury
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Fine-tuning large-scale music generation models, such as MusicGen and Mustango, is a computationally expensive process, often requiring updates to billions of parameters and, therefore, significant hardware resources. Parameter-Efficient Fine-Tuning (PEFT) techniques, particularly adapter-based methods, have emerged as a promising alternative, enabling adaptation with minimal trainable parameters while preserving model performance. However, the design choices for adapters, including their architecture, placement, and size, are numerous, and it is unclear which of these combinations would produce optimal adapters and why, for a given case of low-resource music genre. In this paper, we attempt to answer this question by studying various adapter configurations for two AI music models, MusicGen and Mustango, on two genres: Hindustani Classical and Turkish Makam music. Our findings reveal distinct trade-offs: convolution-based adapters excel in capturing fine-grained local musical details such as ornamentations and short melodic phrases, while transformer-based adapters better preserve long-range dependencies crucial for structured improvisation. Additionally, we analyze computational resource requirements across different adapter scales, demonstrating how mid-sized adapters (40M parameters) achieve an optimal balance between expressivity and quality. Furthermore, we find that Mustango, a diffusion-based model, generates more diverse outputs with better adherence to the description in the input prompt while lacking in providing stability in notes, rhythm alignment, and aesthetics. Also, it is computationally intensive and requires significantly more time to train. In contrast, autoregressive models like MusicGen offer faster training and are more efficient, and can produce better quality output in comparison, but have slightly higher redundancy in their generations.
>
---
#### [new 003] PeakNetFP: Peak-based Neural Audio Fingerprinting Robust to Extreme Time Stretching
- **分类: cs.SD; cs.IR; eess.AS; H.3.1; H.3.3; H.3.4**

- **简介: 该论文属于音频指纹任务，解决时间拉伸下的音频匹配问题。提出PeakNetFP，结合峰值特征与神经网络，提升鲁棒性与效率。**

- **链接: [http://arxiv.org/pdf/2506.21086v1](http://arxiv.org/pdf/2506.21086v1)**

> **作者:** Guillem Cortès-Sebastià; Benjamin Martin; Emilio Molina; Xavier Serra; Romain Hennequin
>
> **备注:** Accepted at ISMIR 2025
>
> **摘要:** This work introduces PeakNetFP, the first neural audio fingerprinting (AFP) system designed specifically around spectral peaks. This novel system is designed to leverage the sparse spectral coordinates typically computed by traditional peak-based AFP methods. PeakNetFP performs hierarchical point feature extraction techniques similar to the computer vision model PointNet++, and is trained using contrastive learning like in the state-of-the-art deep learning AFP, NeuralFP. This combination allows PeakNetFP to outperform conventional AFP systems and achieves comparable performance to NeuralFP when handling challenging time-stretched audio data. In extensive evaluation, PeakNetFP maintains a Top-1 hit rate of over 90% for stretching factors ranging from 50% to 200%. Moreover, PeakNetFP offers significant efficiency advantages: compared to NeuralFP, it has 100 times fewer parameters and uses 11 times smaller input data. These features make PeakNetFP a lightweight and efficient solution for AFP tasks where time stretching is involved. Overall, this system represents a promising direction for future AFP technologies, as it successfully merges the lightweight nature of peak-based AFP with the adaptability and pattern recognition capabilities of neural network-based approaches, paving the way for more scalable and efficient solutions in the field.
>
---
#### [new 004] SmoothSinger: A Conditional Diffusion Model for Singing Voice Synthesis with Multi-Resolution Architecture
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，旨在解决歌唱语音合成中的自然度和质量问题。提出SmoothSinger模型，通过扩散机制直接优化音频，提升合成效果。**

- **链接: [http://arxiv.org/pdf/2506.21478v1](http://arxiv.org/pdf/2506.21478v1)**

> **作者:** Kehan Sui; Jinxu Xiang; Fang Jin
>
> **摘要:** Singing voice synthesis (SVS) aims to generate expressive and high-quality vocals from musical scores, requiring precise modeling of pitch, duration, and articulation. While diffusion-based models have achieved remarkable success in image and video generation, their application to SVS remains challenging due to the complex acoustic and musical characteristics of singing, often resulting in artifacts that degrade naturalness. In this work, we propose SmoothSinger, a conditional diffusion model designed to synthesize high quality and natural singing voices. Unlike prior methods that depend on vocoders as a final stage and often introduce distortion, SmoothSinger refines low-quality synthesized audio directly in a unified framework, mitigating the degradation associated with two-stage pipelines. The model adopts a reference-guided dual-branch architecture, using low-quality audio from any baseline system as a reference to guide the denoising process, enabling more expressive and context-aware synthesis. Furthermore, it enhances the conventional U-Net with a parallel low-frequency upsampling path, allowing the model to better capture pitch contours and long term spectral dependencies. To improve alignment during training, we replace reference audio with degraded ground truth audio, addressing temporal mismatch between reference and target signals. Experiments on the Opencpop dataset, a large-scale Chinese singing corpus, demonstrate that SmoothSinger achieves state-of-the-art results in both objective and subjective evaluations. Extensive ablation studies confirm its effectiveness in reducing artifacts and improving the naturalness of synthesized voices.
>
---
#### [new 005] Integrating Vehicle Acoustic Data for Enhanced Urban Traffic Management: A Study on Speed Classification in Suzhou
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于交通管理中的速度分类任务，通过融合声学数据提升城市交通监控。研究提出BMCNN模型，有效识别车辆速度，助力智能交通系统优化。**

- **链接: [http://arxiv.org/pdf/2506.21269v1](http://arxiv.org/pdf/2506.21269v1)**

> **作者:** Pengfei Fan; Yuli Zhang; Xinheng Wang; Ruiyuan Jiang; Hankang Gu; Dongyao Jia; Shangbo Wang
>
> **摘要:** This study presents and publicly releases the Suzhou Urban Road Acoustic Dataset (SZUR-Acoustic Dataset), which is accompanied by comprehensive data-acquisition protocols and annotation guidelines to ensure transparency and reproducibility of the experimental workflow. To model the coupling between vehicular noise and driving speed, we propose a bimodal-feature-fusion deep convolutional neural network (BMCNN). During preprocessing, an adaptive denoising and normalization strategy is applied to suppress environmental background interference; in the network architecture, parallel branches extract Mel-frequency cepstral coefficients (MFCCs) and wavelet-packet energy features, which are subsequently fused via a cross-modal attention mechanism in the intermediate feature space to fully exploit time-frequency information. Experimental results demonstrate that BMCNN achieves a classification accuracy of 87.56% on the SZUR-Acoustic Dataset and 96.28% on the public IDMT-Traffic dataset. Ablation studies and robustness tests on the Suzhou dataset further validate the contributions of each module to performance improvement and overfitting mitigation. The proposed acoustics-based speed classification method can be integrated into smart-city traffic management systems for real-time noise monitoring and speed estimation, thereby optimizing traffic flow control, reducing roadside noise pollution, and supporting sustainable urban planning.
>
---
#### [new 006] A Hierarchical Deep Learning Approach for Minority Instrument Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于音乐信息检索任务，旨在解决少数乐器检测问题。通过构建层次化深度学习模型，提升粗粒度乐器识别的可靠性。**

- **链接: [http://arxiv.org/pdf/2506.21167v1](http://arxiv.org/pdf/2506.21167v1)**

> **作者:** Dylan Sechet; Francesca Bugiotti; Matthieu Kowalski; Edouard d'Hérouville; Filip Langiewicz
>
> **备注:** International Conference on Digital Audio Effects (DAFx)
>
> **摘要:** Identifying instrument activities within audio excerpts is vital in music information retrieval, with significant implications for music cataloging and discovery. Prior deep learning endeavors in musical instrument recognition have predominantly emphasized instrument classes with ample data availability. Recent studies have demonstrated the applicability of hierarchical classification in detecting instrument activities in orchestral music, even with limited fine-grained annotations at the instrument level. Based on the Hornbostel-Sachs classification, such a hierarchical classification system is evaluated using the MedleyDB dataset, renowned for its diversity and richness concerning various instruments and music genres. This work presents various strategies to integrate hierarchical structures into models and tests a new class of models for hierarchical music prediction. This study showcases more reliable coarse-level instrument detection by bridging the gap between detailed instrument identification and group-level recognition, paving the way for further advancements in this domain.
>
---
#### [new 007] Learnable Adaptive Time-Frequency Representation via Differentiable Short-Time Fourier Transform
- **分类: cs.SD; cs.LG; eess.AS; eess.SP**

- **简介: 该论文属于信号处理任务，旨在解决STFT参数调优困难的问题。通过提出可微分STFT，实现参数的梯度优化，提升时频表示效果。**

- **链接: [http://arxiv.org/pdf/2506.21440v1](http://arxiv.org/pdf/2506.21440v1)**

> **作者:** Maxime Leiber; Yosra Marnissi; Axel Barrau; Sylvain Meignen; Laurent Massoulié
>
> **备注:** DSTFT, STFT, spectrogram, time-frequency, IEEE Transactions on Signal Processing, 10 pages
>
> **摘要:** The short-time Fourier transform (STFT) is widely used for analyzing non-stationary signals. However, its performance is highly sensitive to its parameters, and manual or heuristic tuning often yields suboptimal results. To overcome this limitation, we propose a unified differentiable formulation of the STFT that enables gradient-based optimization of its parameters. This approach addresses the limitations of traditional STFT parameter tuning methods, which often rely on computationally intensive discrete searches. It enables fine-tuning of the time-frequency representation (TFR) based on any desired criterion. Moreover, our approach integrates seamlessly with neural networks, allowing joint optimization of the STFT parameters and network weights. The efficacy of the proposed differentiable STFT in enhancing TFRs and improving performance in downstream tasks is demonstrated through experiments on both simulated and real-world data.
>
---
#### [new 008] CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音编码任务，解决固定帧率导致的冗余问题，提出CodecSlime方法实现动态帧率压缩，提升重建质量。**

- **链接: [http://arxiv.org/pdf/2506.21074v1](http://arxiv.org/pdf/2506.21074v1)**

> **作者:** Hankun Wang; Yiwei Guo; Chongtian Shao; Bohan Li; Xie Chen; Kai Yu
>
> **备注:** 16 pages, 5 figures, 9 tables
>
> **摘要:** Neural speech codecs have been widely used in audio compression and various downstream tasks. Current mainstream codecs are fixed-frame-rate (FFR), which allocate the same number of tokens to every equal-duration slice. However, speech is inherently non-uniform in temporal information density. As a result, many tokens are wasted on steady-state segments like long vowels and silences. To address this mismatch, we present CodecSlime, a plugin-style method for compressing temporal redundancy through supporting dynamic frame rate (DFR) on neural speech codecs for the first time. Our method is unsupervised and architecture-agnostic, combining two key innovations, ScheDFR and Melt-and-Cool, for adapting inference and training, respectively. When integrated into a typical VQ-GAN codec backbone and operating at 40 Hz DFR ($\approx$ 600 bps), the reconstruction WER of CodecSlime is reduced by up to 46% relative to conventional FFR baselines with the same model architecture and similar bitrates, while other metrics are also competitive. CodecSlime also enables flexible trade-offs between reconstruction quality and bitrate: a single model supports inference at multiple frame rates and consistently outperforms FFR models at the corresponding frame rates. Audio samples are available at https://acadarmeria.github.io/codecslime/.
>
---
#### [new 009] MedPrompt: LLM-CNN Fusion with Weight Routing for Medical Image Segmentation and Classification
- **分类: cs.CV; eess.SP**

- **简介: 该论文提出MedPrompt，解决医疗图像分割与分类任务中系统灵活性不足的问题，通过融合LLM与CNN实现高效、可扩展的框架。**

- **链接: [http://arxiv.org/pdf/2506.21199v1](http://arxiv.org/pdf/2506.21199v1)**

> **作者:** Shadman Sobhan; Kazi Abrar Mahmud; Abduz Zami
>
> **备注:** 40 pages, 8 Tables, 9 Figures
>
> **摘要:** Current medical image analysis systems are typically task-specific, requiring separate models for classification and segmentation, and lack the flexibility to support user-defined workflows. To address these challenges, we introduce MedPrompt, a unified framework that combines a few-shot prompted Large Language Model (Llama-4-17B) for high-level task planning with a modular Convolutional Neural Network (DeepFusionLab) for low-level image processing. The LLM interprets user instructions and generates structured output to dynamically route task-specific pretrained weights. This weight routing approach avoids retraining the entire framework when adding new tasks-only task-specific weights are required, enhancing scalability and deployment. We evaluated MedPrompt across 19 public datasets, covering 12 tasks spanning 5 imaging modalities. The system achieves a 97% end-to-end correctness in interpreting and executing prompt-driven instructions, with an average inference latency of 2.5 seconds, making it suitable for near real-time applications. DeepFusionLab achieves competitive segmentation accuracy (e.g., Dice 0.9856 on lungs) and strong classification performance (F1 0.9744 on tuberculosis). Overall, MedPrompt enables scalable, prompt-driven medical imaging by combining the interpretability of LLMs with the efficiency of modular CNNs.
>
---
#### [new 010] Hybrid Deep Learning and Signal Processing for Arabic Dialect Recognition in Low-Resource Settings
- **分类: eess.AS; cs.CL; cs.SD; eess.SP**

- **简介: 该论文属于阿拉伯语方言识别任务，旨在解决低资源环境下数据不足的问题。通过结合传统信号处理与深度学习模型进行实验，验证了MFCC+CNN的有效性。**

- **链接: [http://arxiv.org/pdf/2506.21386v1](http://arxiv.org/pdf/2506.21386v1)**

> **作者:** Ghazal Al-Shwayyat; Omer Nezih Gerek
>
> **摘要:** Arabic dialect recognition presents a significant challenge in speech technology due to the linguistic diversity of Arabic and the scarcity of large annotated datasets, particularly for underrepresented dialects. This research investigates hybrid modeling strategies that integrate classical signal processing techniques with deep learning architectures to address this problem in low-resource scenarios. Two hybrid models were developed and evaluated: (1) Mel-Frequency Cepstral Coefficients (MFCC) combined with a Convolutional Neural Network (CNN), and (2) Discrete Wavelet Transform (DWT) features combined with a Recurrent Neural Network (RNN). The models were trained on a dialect-filtered subset of the Common Voice Arabic dataset, with dialect labels assigned based on speaker metadata. Experimental results demonstrate that the MFCC + CNN architecture achieved superior performance, with an accuracy of 91.2% and strong precision, recall, and F1-scores, significantly outperforming the Wavelet + RNN configuration, which achieved an accuracy of 66.5%. These findings highlight the effectiveness of leveraging spectral features with convolutional models for Arabic dialect recognition, especially when working with limited labeled data. The study also identifies limitations related to dataset size, potential regional overlaps in labeling, and model optimization, providing a roadmap for future research. Recommendations for further improvement include the adoption of larger annotated corpora, integration of self-supervised learning techniques, and exploration of advanced neural architectures such as Transformers. Overall, this research establishes a strong baseline for future developments in Arabic dialect recognition within resource-constrained environments.
>
---
#### [new 011] Global and Local Contrastive Learning for Joint Representations from Cardiac MRI and ECG
- **分类: eess.IV; cs.AI; cs.CV; eess.SP**

- **简介: 该论文属于多模态学习任务，旨在通过结合ECG和CMR数据提升心脏诊断能力。解决ECG无法直接测量功能参数的问题，通过对比学习框架PTACL增强ECG表示。**

- **链接: [http://arxiv.org/pdf/2506.20683v1](http://arxiv.org/pdf/2506.20683v1)**

> **作者:** Alexander Selivanov; Philip Müller; Özgün Turgut; Nil Stolt-Ansó; Daniel Rückert
>
> **备注:** accepted to MICCAI 2025 (Springer LNCS)
>
> **摘要:** An electrocardiogram (ECG) is a widely used, cost-effective tool for detecting electrical abnormalities in the heart. However, it cannot directly measure functional parameters, such as ventricular volumes and ejection fraction, which are crucial for assessing cardiac function. Cardiac magnetic resonance (CMR) is the gold standard for these measurements, providing detailed structural and functional insights, but is expensive and less accessible. To bridge this gap, we propose PTACL (Patient and Temporal Alignment Contrastive Learning), a multimodal contrastive learning framework that enhances ECG representations by integrating spatio-temporal information from CMR. PTACL uses global patient-level contrastive loss and local temporal-level contrastive loss. The global loss aligns patient-level representations by pulling ECG and CMR embeddings from the same patient closer together, while pushing apart embeddings from different patients. Local loss enforces fine-grained temporal alignment within each patient by contrasting encoded ECG segments with corresponding encoded CMR frames. This approach enriches ECG representations with diagnostic information beyond electrical activity and transfers more insights between modalities than global alignment alone, all without introducing new learnable weights. We evaluate PTACL on paired ECG-CMR data from 27,951 subjects in the UK Biobank. Compared to baseline approaches, PTACL achieves better performance in two clinically relevant tasks: (1) retrieving patients with similar cardiac phenotypes and (2) predicting CMR-derived cardiac function parameters, such as ventricular volumes and ejection fraction. Our results highlight the potential of PTACL to enhance non-invasive cardiac diagnostics using ECG. The code is available at: https://github.com/alsalivan/ecgcmr
>
---
#### [new 012] Step-by-Step Video-to-Audio Synthesis via Negative Audio Guidance
- **分类: cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决多音轨合成问题。通过分步生成音频并利用负向音频引导，提升合成音频质量。**

- **链接: [http://arxiv.org/pdf/2506.20995v1](http://arxiv.org/pdf/2506.20995v1)**

> **作者:** Akio Hayakawa; Masato Ishii; Takashi Shibuya; Yuki Mitsufuji
>
> **摘要:** We propose a novel step-by-step video-to-audio generation method that sequentially produces individual audio tracks, each corresponding to a specific sound event in the video. Our approach mirrors traditional Foley workflows, aiming to capture all sound events induced by a given video comprehensively. Each generation step is formulated as a guided video-to-audio synthesis task, conditioned on a target text prompt and previously generated audio tracks. This design is inspired by the idea of concept negation from prior compositional generation frameworks. To enable this guided generation, we introduce a training framework that leverages pre-trained video-to-audio models and eliminates the need for specialized paired datasets, allowing training on more accessible data. Experimental results demonstrate that our method generates multiple semantically distinct audio tracks for a single input video, leading to higher-quality composite audio synthesis than existing baselines.
>
---
#### [new 013] ThinkSound: Chain-of-Thought Reasoning in Multimodal Large Language Models for Audio Generation and Editing
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文提出ThinkSound框架，解决视频到音频生成中的高保真与语义一致性问题，通过链式推理实现分步音频生成与编辑。**

- **链接: [http://arxiv.org/pdf/2506.21448v1](http://arxiv.org/pdf/2506.21448v1)**

> **作者:** Huadai Liu; Jialei Wang; Kaicheng Luo; Wen Wang; Qian Chen; Zhou Zhao; Wei Xue
>
> **摘要:** While end-to-end video-to-audio generation has greatly improved, producing high-fidelity audio that authentically captures the nuances of visual content remains challenging. Like professionals in the creative industries, such generation requires sophisticated reasoning about items such as visual dynamics, acoustic environments, and temporal relationships. We present \textbf{ThinkSound}, a novel framework that leverages Chain-of-Thought (CoT) reasoning to enable stepwise, interactive audio generation and editing for videos. Our approach decomposes the process into three complementary stages: foundational foley generation that creates semantically coherent soundscapes, interactive object-centric refinement through precise user interactions, and targeted editing guided by natural language instructions. At each stage, a multimodal large language model generates contextually aligned CoT reasoning that guides a unified audio foundation model. Furthermore, we introduce \textbf{AudioCoT}, a comprehensive dataset with structured reasoning annotations that establishes connections between visual content, textual descriptions, and sound synthesis. Experiments demonstrate that ThinkSound achieves state-of-the-art performance in video-to-audio generation across both audio metrics and CoT metrics and excels in out-of-distribution Movie Gen Audio benchmark. The demo page is available at https://ThinkSound-Demo.github.io.
>
---
#### [new 014] Aligning Spoken Dialogue Models from User Interactions
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于对话系统任务，解决实时语音对话模型的偏好对齐问题。通过构建大规模数据集并微调模型，提升对话的准确性、安全性和上下文一致性。**

- **链接: [http://arxiv.org/pdf/2506.21463v1](http://arxiv.org/pdf/2506.21463v1)**

> **作者:** Anne Wu; Laurent Mazaré; Neil Zeghidour; Alexandre Défossez
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** We propose a novel preference alignment framework for improving spoken dialogue models on real-time conversations from user interactions. Current preference learning methods primarily focus on text-based language models, and are not directly suited to the complexities of real-time speech interactions, with richer dynamics (e.g. interruption, interjection) and no explicit segmentation between speaker turns.We create a large-scale dataset of more than 150,000 preference pairs from raw multi-turn speech conversations, annotated with AI feedback, to cover preferences over both linguistic content and temporal context variations. We leverage offline alignment methods to finetune a full-duplex autoregressive speech-to-speech model. Extensive experiments demonstrate that feedback on generic conversations can be consistently effective in improving spoken dialogue models to produce more factual, safer and more contextually aligned interactions. We deploy the finetuned model and conduct holistic human evaluations to assess the impact beyond single-turn conversations. Our findings shed light on the importance of a well-calibrated balance among various dynamics, crucial for natural real-time speech dialogue systems.
>
---
#### [new 015] Prompt-Guided Turn-Taking Prediction
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于对话系统中的任务，解决turn-taking预测问题。通过引入文本提示控制预测行为，提升模型的灵活性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.21191v1](http://arxiv.org/pdf/2506.21191v1)**

> **作者:** Koji Inoue; Mikey Elmers; Yahui Fu; Zi Haur Pang; Divesh Lala; Keiko Ochi; Tatsuya Kawahara
>
> **备注:** This paper has been accepted for presentation at SIGdial Meeting on Discourse and Dialogue 2025 (SIGDIAL 2025) and represents the author's version of the work
>
> **摘要:** Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
>
---
## 更新

#### [replaced 001] IndieFake Dataset: A Benchmark Dataset for Audio Deepfake Detection
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.19014v2](http://arxiv.org/pdf/2506.19014v2)**

> **作者:** Abhay Kumar; Kunal Verma; Omkar More
>
> **备注:** Project Website: https://indie-fake-dataset.netlify.app/
>
> **摘要:** Advancements in audio deepfake technology offers benefits like AI assistants, better accessibility for speech impairments, and enhanced entertainment. However, it also poses significant risks to security, privacy, and trust in digital communications. Detecting and mitigating these threats requires comprehensive datasets. Existing datasets lack diverse ethnic accents, making them inadequate for many real-world scenarios. Consequently, models trained on these datasets struggle to detect audio deepfakes in diverse linguistic and cultural contexts such as in South-Asian countries. Ironically, there is a stark lack of South-Asian speaker samples in the existing datasets despite constituting a quarter of the worlds population. This work introduces the IndieFake Dataset (IFD), featuring 27.17 hours of bonafide and deepfake audio from 50 English speaking Indian speakers. IFD offers balanced data distribution and includes speaker-level characterization, absent in datasets like ASVspoof21 (DF). We evaluated various baselines on IFD against existing ASVspoof21 (DF) and In-The-Wild (ITW) datasets. IFD outperforms ASVspoof21 (DF) and proves to be more challenging compared to benchmark ITW dataset. The complete dataset, along with documentation and sample reference clips, is publicly accessible for research use on project website.
>
---
#### [replaced 002] TCDiff++: An End-to-end Trajectory-Controllable Diffusion Model for Harmonious Music-Driven Group Choreography
- **分类: cs.SD; cs.CV; cs.GR; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.18671v3](http://arxiv.org/pdf/2506.18671v3)**

> **作者:** Yuqin Dai; Wanlu Zhu; Ronghui Li; Xiu Li; Zhenyu Zhang; Jun Li; Jian Yang
>
> **摘要:** Music-driven dance generation has garnered significant attention due to its wide range of industrial applications, particularly in the creation of group choreography. During the group dance generation process, however, most existing methods still face three primary issues: multi-dancer collisions, single-dancer foot sliding and abrupt swapping in the generation of long group dance. In this paper, we propose TCDiff++, a music-driven end-to-end framework designed to generate harmonious group dance. Specifically, to mitigate multi-dancer collisions, we utilize a dancer positioning embedding to better maintain the relative positioning among dancers. Additionally, we incorporate a distance-consistency loss to ensure that inter-dancer distances remain within plausible ranges. To address the issue of single-dancer foot sliding, we introduce a swap mode embedding to indicate dancer swapping patterns and design a Footwork Adaptor to refine raw motion, thereby minimizing foot sliding. For long group dance generation, we present a long group diffusion sampling strategy that reduces abrupt position shifts by injecting positional information into the noisy input. Furthermore, we integrate a Sequence Decoder layer to enhance the model's ability to selectively process long sequences. Extensive experiments demonstrate that our TCDiff++ achieves state-of-the-art performance, particularly in long-duration scenarios, ensuring high-quality and coherent group dance generation.
>
---
#### [replaced 003] Rapid Gyroscope Calibration: A Deep Learning Approach
- **分类: cs.LG; cs.AI; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.00488v3](http://arxiv.org/pdf/2409.00488v3)**

> **作者:** Yair Stolero; Itzik Klein
>
> **备注:** 10 Pages, 14 Figures
>
> **摘要:** Low-cost gyroscope calibration is essential for ensuring the accuracy and reliability of gyroscope measurements. Stationary calibration estimates the deterministic parts of measurement errors. To this end, a common practice is to average the gyroscope readings during a predefined period and estimate the gyroscope bias. Calibration duration plays a crucial role in performance, therefore, longer periods are preferred. However, some applications require quick startup times and calibration is therefore allowed only for a short time. In this work, we focus on reducing low-cost gyroscope calibration time using deep learning methods. We propose an end-to-end convolutional neural network for the application of gyroscope calibration. We explore the possibilities of using multiple real and virtual gyroscopes to improve the calibration performance of single gyroscopes. To train and validate our approach, we recorded a dataset consisting of 186.6 hours of gyroscope readings, using 36 gyroscopes of four different brands. We also created a virtual dataset consisting of simulated gyroscope readings. The six datasets were used to evaluate our proposed approach. One of our key achievements in this work is reducing gyroscope calibration time by up to 89% using three low-cost gyroscopes. Our dataset is publicly available to allow reproducibility of our work and to increase research in the field.
>
---
#### [replaced 004] Aliasing Reduction in Neural Amp Modeling by Smoothing Activations
- **分类: eess.AS; cs.SD; eess.SP**

- **链接: [http://arxiv.org/pdf/2505.04082v2](http://arxiv.org/pdf/2505.04082v2)**

> **作者:** Ryota Sato; Julius O. Smith III
>
> **备注:** Accepted to DAFx 2025
>
> **摘要:** The increasing demand for high-quality digital emulations of analog audio hardware, such as vintage tube guitar amplifiers, led to numerous works on neural network-based black-box modeling, with deep learning architectures like WaveNet showing promising results. However, a key limitation in all of these models was the aliasing artifacts stemming from nonlinear activation functions in neural networks. In this paper, we investigated novel and modified activation functions aimed at mitigating aliasing within neural amplifier models. Supporting this, we introduced a novel metric, the Aliasing-to-Signal Ratio (ASR), which quantitatively assesses the level of aliasing with high accuracy. Measuring also the conventional Error-to-Signal Ratio (ESR), we conducted studies on a range of preexisting and modern activation functions with varying stretch factors. Our findings confirmed that activation functions with smoother curves tend to achieve lower ASR values, indicating a noticeable reduction in aliasing. Notably, this improvement in aliasing reduction was achievable without a substantial increase in ESR, demonstrating the potential for high modeling accuracy with reduced aliasing in neural amp models.
>
---
#### [replaced 005] ITO-Master: Inference-Time Optimization for Audio Effects Modeling of Music Mastering Processors
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.16889v2](http://arxiv.org/pdf/2506.16889v2)**

> **作者:** Junghyun Koo; Marco A. Martinez-Ramirez; Wei-Hsiang Liao; Giorgio Fabbro; Michele Mancusi; Yuki Mitsufuji
>
> **备注:** ISMIR 2025
>
> **摘要:** Music mastering style transfer aims to model and apply the mastering characteristics of a reference track to a target track, simulating the professional mastering process. However, existing methods apply fixed processing based on a reference track, limiting users' ability to fine-tune the results to match their artistic intent. In this paper, we introduce the ITO-Master framework, a reference-based mastering style transfer system that integrates Inference-Time Optimization (ITO) to enable finer user control over the mastering process. By optimizing the reference embedding during inference, our approach allows users to refine the output dynamically, making micro-level adjustments to achieve more precise mastering results. We explore both black-box and white-box methods for modeling mastering processors and demonstrate that ITO improves mastering performance across different styles. Through objective evaluation, subjective listening tests, and qualitative analysis using text-based conditioning with CLAP embeddings, we validate that ITO enhances mastering style similarity while offering increased adaptability. Our framework provides an effective and user-controllable solution for mastering style transfer, allowing users to refine their results beyond the initial style transfer.
>
---
