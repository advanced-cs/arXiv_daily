# 音频 cs.SD;  eess.AS

- **最新发布 30 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Low-Resource Audio Codec (LRAC): 2025 Challenge Description
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出2025低资源音频编码挑战，旨在推动神经与混合编码器在计算受限场景下的应用。针对低比特率、低延迟及噪声/混响鲁棒性问题，提供数据集、基线系统与评估框架，促进高效编码器设计及其与语音增强的融合。**

- **链接: [http://arxiv.org/pdf/2510.23312v1](http://arxiv.org/pdf/2510.23312v1)**

> **作者:** Kamil Wojcicki; Yusuf Ziya Isik; Laura Lechler; Mansur Yesilbursa; Ivana Balić; Wolfgang Mack; Rafał Łaganowski; Guoqing Zhang; Yossi Adi; Minje Kim; Shinji Watanabe
>
> **摘要:** While recent neural audio codecs deliver superior speech quality at ultralow bitrates over traditional methods, their practical adoption is hindered by obstacles related to low-resource operation and robustness to acoustic distortions. Edge deployment scenarios demand codecs that operate under stringent compute constraints while maintaining low latency and bitrate. The presence of background noise and reverberation further necessitates designs that are resilient to such degradations. The performance of neural codecs under these constraints and their integration with speech enhancement remain largely unaddressed. To catalyze progress in this area, we introduce the 2025 Low-Resource Audio Codec Challenge, which targets the development of neural and hybrid codecs for resource-constrained applications. Participants are supported with a standardized training dataset, two baseline systems, and a comprehensive evaluation framework. The challenge is expected to yield valuable insights applicable to both codec design and related downstream audio tasks.
>
---
#### [new 002] Streaming Generation for Music Accompaniment
- **分类: cs.SD**

- **简介: 该论文研究实时音频到音频伴奏生成任务，旨在解决模型在接收实时输入时同步生成连贯伴奏的问题。提出考虑系统延迟的模型设计，通过调节未来可见性与输出块长度，在延迟与质量间权衡，并发现传统训练方法不足，需引入前瞻与代理目标以提升实时协作性能。**

- **链接: [http://arxiv.org/pdf/2510.22105v1](http://arxiv.org/pdf/2510.22105v1)**

> **作者:** Yusong Wu; Mason Wang; Heidi Lei; Stephen Brade; Lancelot Blanchard; Shih-Lun Wu; Aaron Courville; Anna Huang
>
> **摘要:** Music generation models can produce high-fidelity coherent accompaniment given complete audio input, but are limited to editing and loop-based workflows. We study real-time audio-to-audio accompaniment: as a model hears an input audio stream (e.g., a singer singing), it has to also simultaneously generate in real-time a coherent accompanying stream (e.g., a guitar accompaniment). In this work, we propose a model design considering inevitable system delays in practical deployment with two design variables: future visibility $t_f$, the offset between the output playback time and the latest input time used for conditioning, and output chunk duration $k$, the number of frames emitted per call. We train Transformer decoders across a grid of $(t_f,k)$ and show two consistent trade-offs: increasing effective $t_f$ improves coherence by reducing the recency gap, but requires faster inference to stay within the latency budget; increasing $k$ improves throughput but results in degraded accompaniment due to a reduced update rate. Finally, we observe that naive maximum-likelihood streaming training is insufficient for coherent accompaniment where future context is not available, motivating advanced anticipatory and agentic objectives for live jamming.
>
---
#### [new 003] PromptReverb: Multimodal Room Impulse Response Generation Through Latent Rectified Flow Matching
- **分类: cs.SD; cs.AI; I.2.6, H.5.5**

- **简介: 该论文提出PromptReverb，解决全频段RIR数据稀缺与多模态输入生成不准的问题。通过变分自编码器上采样至48kHz，并结合基于修正流匹配的条件扩散变换器，实现从自然语言描述生成高保真、声学准确的房间混响响应，显著提升感知质量与参数真实性。**

- **链接: [http://arxiv.org/pdf/2510.22439v1](http://arxiv.org/pdf/2510.22439v1)**

> **作者:** Ali Vosoughi; Yongyi Zang; Qihui Yang; Nathan Peak; Randal Leistikow; Chenliang Xu
>
> **备注:** 9 pages, 2 figures, 4 tables
>
> **摘要:** Room impulse response (RIR) generation remains a critical challenge for creating immersive virtual acoustic environments. Current methods suffer from two fundamental limitations: the scarcity of full-band RIR datasets and the inability of existing models to generate acoustically accurate responses from diverse input modalities. We present PromptReverb, a two-stage generative framework that addresses these challenges. Our approach combines a variational autoencoder that upsamples band-limited RIRs to full-band quality (48 kHz), and a conditional diffusion transformer model based on rectified flow matching that generates RIRs from descriptions in natural language. Empirical evaluation demonstrates that PromptReverb produces RIRs with superior perceptual quality and acoustic accuracy compared to existing methods, achieving 8.8% mean RT60 error compared to -37% for widely used baselines and yielding more realistic room-acoustic parameters. Our method enables practical applications in virtual reality, architectural acoustics, and audio production where flexible, high-quality RIR synthesis is essential.
>
---
#### [new 004] GuitarFlow: Realistic Electric Guitar Synthesis From Tablatures via Flow Matching and Style Transfer
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出GuitarFlow，针对电吉他音色合成任务，解决传统方法在表达吉他特有演奏技巧（如推弦、闷音）时的局限性。通过将六线谱转化为音频并利用流匹配进行风格迁移，实现高效、高保真的电吉他音色生成，仅需不足6小时数据训练，显著提升合成音频的现实感。**

- **链接: [http://arxiv.org/pdf/2510.21872v1](http://arxiv.org/pdf/2510.21872v1)**

> **作者:** Jackson Loth; Pedro Sarmento; Mark Sandler; Mathieu Barthet
>
> **备注:** To be published in Proceedings of the 17th International Symposium on Computer Music and Multidisciplinary Research (CMMR)
>
> **摘要:** Music generation in the audio domain using artificial intelligence (AI) has witnessed steady progress in recent years. However for some instruments, particularly the guitar, controllable instrument synthesis remains limited in expressivity. We introduce GuitarFlow, a model designed specifically for electric guitar synthesis. The generative process is guided using tablatures, an ubiquitous and intuitive guitar-specific symbolic format. The tablature format easily represents guitar-specific playing techniques (e.g. bends, muted strings and legatos), which are more difficult to represent in other common music notation formats such as MIDI. Our model relies on an intermediary step of first rendering the tablature to audio using a simple sample-based virtual instrument, then performing style transfer using Flow Matching in order to transform the virtual instrument audio into more realistic sounding examples. This results in a model that is quick to train and to perform inference, requiring less than 6 hours of training data. We present the results of objective evaluation metrics, together with a listening test, in which we show significant improvement in the realism of the generated guitar audio from tablatures.
>
---
#### [new 005] SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出SoulX-Podcast，面向多轮多说话人播客对话的语音合成任务，解决现有TTS系统在多说话人连贯性、方言与副语言多样性上的不足。通过集成多语种及方言支持、副语言控制，实现90分钟以上稳定自然的对话生成，兼具高保真单人语音与流畅多人对话表现。**

- **链接: [http://arxiv.org/pdf/2510.23541v1](http://arxiv.org/pdf/2510.23541v1)**

> **作者:** Hanke Xie; Haopeng Lin; Wenxiao Cao; Dake Guo; Wenjie Tian; Jun Wu; Hanlin Wen; Ruixuan Shang; Hongmei Liu; Zhiqi Jiang; Yuepeng Jiang; Wenxi Chen; Ruiqi Yan; Jiale Qian; Yichao Yan; Shunshun Yin; Ming Tao; Xie Chen; Lei Xie; Xinsheng Wang
>
> **摘要:** Recent advances in text-to-speech (TTS) synthesis have significantly improved speech expressiveness and naturalness. However, most existing systems are tailored for single-speaker synthesis and fall short in generating coherent multi-speaker conversational speech. This technical report presents SoulX-Podcast, a system designed for podcast-style multi-turn, multi-speaker dialogic speech generation, while also achieving state-of-the-art performance in conventional TTS tasks. To meet the higher naturalness demands of multi-turn spoken dialogue, SoulX-Podcast integrates a range of paralinguistic controls and supports both Mandarin and English, as well as several Chinese dialects, including Sichuanese, Henanese, and Cantonese, enabling more personalized podcast-style speech generation. Experimental results demonstrate that SoulX-Podcast can continuously produce over 90 minutes of conversation with stable speaker timbre and smooth speaker transitions. Moreover, speakers exhibit contextually adaptive prosody, reflecting natural rhythm and intonation changes as dialogues progress. Across multiple evaluation metrics, SoulX-Podcast achieves state-of-the-art performance in both monologue TTS and multi-turn conversational speech synthesis.
>
---
#### [new 006] Evaluating Multimodal Large Language Models on Core Music Perception Tasks
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文评估多模态大模型在核心音乐感知任务（节奏、转调、和弦识别）上的表现，揭示其对音频的“听觉”能力不足。通过分离输入模态、示例暴露与推理策略，发现模型在MIDI上表现优异但在音频上退化，即使采用结构化推理也仍脆弱。研究明确了感知与推理的边界，为构建以音频为核心的音乐系统提供指导。**

- **链接: [http://arxiv.org/pdf/2510.22455v1](http://arxiv.org/pdf/2510.22455v1)**

> **作者:** Brandon James Carone; Iran R. Roman; Pablo Ripollés
>
> **备注:** Accepted to the NeurIPS 2025 Workshop on AI for Music (AI4Music), 16 pages, 1 figure, 3 tables
>
> **摘要:** Multimodal Large Language Models (LLMs) claim "musical understanding" via evaluations that conflate listening with score reading. We benchmark three SOTA LLMs (Gemini 2.5 Pro, Gemini 2.5 Flash, and Qwen2.5-Omni) across three core music skills: Syncopation Scoring, Transposition Detection, and Chord Quality Identification. Moreover, we separate three sources of variability: (i) perceptual limitations (audio vs. MIDI inputs), (ii) exposure to examples (zero- vs. few-shot manipulations), and (iii) reasoning strategies (Standalone, CoT, LogicLM). For the latter we adapt LogicLM, a framework combining LLMs with symbolic solvers to perform structured reasoning, to music. Results reveal a clear perceptual gap: models perform near ceiling on MIDI but show accuracy drops on audio. Reasoning and few-shot prompting offer minimal gains. This is expected for MIDI, where performance reaches saturation, but more surprising for audio, where LogicLM, despite near-perfect MIDI accuracy, remains notably brittle. Among models, Gemini Pro achieves the highest performance across most conditions. Overall, current systems reason well over symbols (MIDI) but do not yet "listen" reliably from audio. Our method and dataset make the perception-reasoning boundary explicit and offer actionable guidance for building robust, audio-first music systems.
>
---
#### [new 007] Learning Linearity in Audio Consistency Autoencoders via Implicit Regularization
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文针对音频自编码器隐空间非线性导致难以进行直观操作的问题，提出通过数据增强诱导高压缩一致性自编码器的线性特性。无需修改架构或损失函数，即可实现缩放与加法不变性，提升隐空间可操作性，应用于音乐源混合与分离，实现了更高效、直观的音频处理。**

- **链接: [http://arxiv.org/pdf/2510.23530v1](http://arxiv.org/pdf/2510.23530v1)**

> **作者:** Bernardo Torres; Manuel Moussallam; Gabriel Meseguer-Brocal
>
> **摘要:** Audio autoencoders learn useful, compressed audio representations, but their non-linear latent spaces prevent intuitive algebraic manipulation such as mixing or scaling. We introduce a simple training methodology to induce linearity in a high-compression Consistency Autoencoder (CAE) by using data augmentation, thereby inducing homogeneity (equivariance to scalar gain) and additivity (the decoder preserves addition) without altering the model's architecture or loss function. When trained with our method, the CAE exhibits linear behavior in both the encoder and decoder while preserving reconstruction fidelity. We test the practical utility of our learned space on music source composition and separation via simple latent arithmetic. This work presents a straightforward technique for constructing structured latent spaces, enabling more intuitive and efficient audio processing.
>
---
#### [new 008] Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMS
- **分类: eess.AS; cs.CV; cs.SD**

- **简介: 该论文研究多模态语音识别中大语言模型的注意力陷阱与过激活问题。针对音频-视觉语音识别（AVSR）任务，发现除起始符外，中间低语义标记也存在注意力集中和特征激活异常。提出去相关损失函数，降低起始符与其他标记的余弦相似度，有效缓解该问题，并提升在高特征下采样率下的识别准确率。**

- **链接: [http://arxiv.org/pdf/2510.22603v1](http://arxiv.org/pdf/2510.22603v1)**

> **作者:** Anand; Umberto Cappellazzo; Stavros Petridis; Maja Pantic
>
> **备注:** The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
>
> **摘要:** Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
>
---
#### [new 009] Treble10: A high-quality dataset for far-field speech recognition, dereverberation, and enhancement
- **分类: eess.AS; cs.LG**

- **简介: 该论文提出Treble10，一个大规模、物理准确的远场语音数据集，用于语音识别、去混响和增强任务。针对现有数据集在真实性和可扩展性间的权衡问题，采用混合波-几何声学仿真，生成高保真房间脉冲响应与配对语音场景，支持多通道与多种格式，实现物理真实且可复现的评估与数据增强。**

- **链接: [http://arxiv.org/pdf/2510.23141v1](http://arxiv.org/pdf/2510.23141v1)**

> **作者:** Sarabeth S. Mullins; Georg Götz; Eric Bezzam; Steven Zheng; Daniel Gert Nielsen
>
> **摘要:** Accurate far-field speech datasets are critical for tasks such as automatic speech recognition (ASR), dereverberation, speech enhancement, and source separation. However, current datasets are limited by the trade-off between acoustic realism and scalability. Measured corpora provide faithful physics but are expensive, low-coverage, and rarely include paired clean and reverberant data. In contrast, most simulation-based datasets rely on simplified geometrical acoustics, thus failing to reproduce key physical phenomena like diffraction, scattering, and interference that govern sound propagation in complex environments. We introduce Treble10, a large-scale, physically accurate room-acoustic dataset. Treble10 contains over 3000 broadband room impulse responses (RIRs) simulated in 10 fully furnished real-world rooms, using a hybrid simulation paradigm implemented in the Treble SDK that combines a wave-based and geometrical acoustics solver. The dataset provides six complementary subsets, spanning mono, 8th-order Ambisonics, and 6-channel device RIRs, as well as pre-convolved reverberant speech scenes paired with LibriSpeech utterances. All signals are simulated at 32 kHz, accurately modelling low-frequency wave effects and high-frequency reflections. Treble10 bridges the realism gap between measurement and simulation, enabling reproducible, physically grounded evaluation and large-scale data augmentation for far-field speech tasks. The dataset is openly available via the Hugging Face Hub, and is intended as both a benchmark and a template for next-generation simulation-driven audio research.
>
---
#### [new 010] UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models
- **分类: eess.AS; cs.CL**

- **简介: 该论文针对语音对话模型缺乏细粒度风格控制的问题，提出UltraVoice数据集，涵盖830小时多风格对话，支持情绪、语速、音量等六维控制。通过微调主流模型，显著提升风格可控性与对话能力，验证了数据集在语音合成与对话系统中的高价值。**

- **链接: [http://arxiv.org/pdf/2510.22588v1](http://arxiv.org/pdf/2510.22588v1)**

> **作者:** Wenming Tu; Guanrou Yang; Ruiqi Yan; Wenxi Chen; Ziyang Ma; Yipeng Kang; Kai Yu; Xie Chen; Zilong Zheng
>
> **备注:** 23 pages, 4 figures
>
> **摘要:** Spoken dialogue models currently lack the ability for fine-grained speech style control, a critical capability for human-like interaction that is often overlooked in favor of purely functional capabilities like reasoning and question answering. To address this limitation, we introduce UltraVoice, the first large-scale speech dialogue dataset engineered for multiple fine-grained speech style control. Encompassing over 830 hours of speech dialogues, UltraVoice provides instructions across six key speech stylistic dimensions: emotion, speed, volume, accent, language, and composite styles. Fine-tuning leading models such as SLAM-Omni and VocalNet on UltraVoice significantly enhances their fine-grained speech stylistic controllability without degrading core conversational abilities. Specifically, our fine-tuned models achieve improvements of 29.12-42.33% in Mean Opinion Score (MOS) and 14.61-40.09 percentage points in Instruction Following Rate (IFR) on multi-dimensional control tasks designed in the UltraVoice. Moreover, on the URO-Bench benchmark, our fine-tuned models demonstrate substantial gains in core understanding, reasoning, and conversational abilities, with average improvements of +10.84% on the Basic setting and +7.87% on the Pro setting. Furthermore, the dataset's utility extends to training controllable Text-to-Speech (TTS) models, underscoring its high quality and broad applicability for expressive speech synthesis. The complete dataset and model checkpoints are available at: https://github.com/bigai-nlco/UltraVoice.
>
---
#### [new 011] Adapting Speech Foundation Models with Large Language Models for Unified Speech Recognition
- **分类: eess.AS**

- **简介: 该论文针对统一语音识别任务，解决语音基础模型在多模态场景下适应性差的问题。提出UASR-LLM框架，通过视觉注入模块与语言模型结合，实现听觉、视觉及视听联合识别，采用两阶段训练策略，在冻结语音模型的前提下提升性能。**

- **链接: [http://arxiv.org/pdf/2510.22961v1](http://arxiv.org/pdf/2510.22961v1)**

> **作者:** Jing-Xuan Zhang; Genshun Wan; Jin Li; Jianqing Gao
>
> **备注:** submitted to Pattern Recognition
>
> **摘要:** Unified speech recognition aims to perform auditory, visual, and audiovisual speech recognition within a single model framework. While speech foundation models (SFMs) have demonstrated remarkable performance in auditory tasks, their adaptation to multimodal scenarios remains underexplored. This paper presents UASR-LLM, a novel framework that adapts frozen SFMs to unified VSR, ASR, and AVSR tasks by leveraging large language models (LLMs) as text decoders. Our approach introduces visual representations into multiple SFM layers through visual injection modules, enabling multimodal input processing and unified hidden representations. The augmented SFMs connect with decoder-only LLMs via a feed-forward adaptor, where concatenated representations and instruction prompts guide speech transcription. We implement a twostage training strategy: visual injection pretraining followed by speech recognition finetuning. SFM parameters remain frozen throughout training, with only visual injection modules optimized initially, and LLMs finetuned using LoRA parameters subsequently. Experimental results demonstrate superior performance over state-of-the-art baselines across VSR, ASR, and AVSR tasks under both clean and noisy conditions. Ablation studies confirm generalization across various SFMs and LLMs, validating the proposed training strategy.
>
---
#### [new 012] LibriConvo: Simulating Conversations from Read Literature for ASR and Diarization
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出LibriConvo，一个基于文学文本的多说话人对话模拟数据集，用于语音识别（ASR）与说话人分离（Diarization）任务。针对现有数据语义断裂、时间间隔不真实的问题，通过语境一致的语音组织与空间合理的声音模拟，实现自然对话动态。实验表明其有效提升模型性能，为多说话人语音处理提供高质量基准。**

- **链接: [http://arxiv.org/pdf/2510.23320v1](http://arxiv.org/pdf/2510.23320v1)**

> **作者:** Máté Gedeon; Péter Mihajlik
>
> **备注:** Submitted to LREC 2026
>
> **摘要:** We introduce LibriConvo, a simulated multi-speaker conversational dataset based on speaker-aware conversation simulation (SASC), designed to support training and evaluation of speaker diarization and automatic speech recognition (ASR) systems. Unlike prior resources that mostly rely on semantically disconnected utterances and implausible temporal gaps, LibriConvo ensures semantic coherence and realistic conversational timing. Our pipeline leverages CallHome with external VAD for reliable boundaries, applies compression to reduce unnaturally long silences, and organizes LibriTTS utterances by book to maintain contextual consistency. Acoustic realism is enhanced via a novel room impulse response selection procedure that ranks speaker-microphone configurations by spatial plausibility, balancing realism and diversity. The dataset comprises 240.1 hours across 1,496 dialogues with 830 unique speakers, split in a speaker-disjoint manner for robust evaluation. Baselines show that the sortformer model outperforms the pyannote pipeline in diarization, while a fine-tuned Fast Conformer-CTC XLarge with Serialized Output Training achieves 7.29\% WER for ASR, surpassing zero-shot Whisper-large-v3. LibriConvo provides a valuable resource for advancing multi-speaker speech processing research with realistic conversational dynamics and controlled experimental conditions.
>
---
#### [new 013] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出SAO-Instruct，一种基于自然语言指令自由编辑音频的模型。针对现有方法需完整描述或受限于预设指令的问题，构建音频编辑三元组数据集，利用合成数据训练，实现对真实音频的灵活编辑。实验表明其在客观与主观评价上均表现优异。**

- **链接: [http://arxiv.org/pdf/2510.22795v1](http://arxiv.org/pdf/2510.22795v1)**

> **作者:** Michael Ungersböck; Florian Grötschla; Luca A. Lanzendörfer; June Young Yi; Changho Choi; Roger Wattenhofer
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Generative models have made significant progress in synthesizing high-fidelity audio from short textual descriptions. However, editing existing audio using natural language has remained largely underexplored. Current approaches either require the complete description of the edited audio or are constrained to predefined edit instructions that lack flexibility. In this work, we introduce SAO-Instruct, a model based on Stable Audio Open capable of editing audio clips using any free-form natural language instruction. To train our model, we create a dataset of audio editing triplets (input audio, edit instruction, output audio) using Prompt-to-Prompt, DDPM inversion, and a manual editing pipeline. Although partially trained on synthetic data, our model generalizes well to real in-the-wild audio clips and unseen edit instructions. We demonstrate that SAO-Instruct achieves competitive performance on objective metrics and outperforms other audio editing approaches in a subjective listening study. To encourage future research, we release our code and model weights.
>
---
#### [new 014] SRP-PHAT-NET: A Reliability-Driven DNN for Reverberant Speaker Localization
- **分类: eess.AS**

- **简介: 该论文针对混响环境下声源定位任务，提出SRP-PHAT-NET模型，利用SRP-PHAT方向图作为特征，并引入内置可靠性评估机制。通过高斯加权标签训练，实现对预测置信度的精准估计，实验表明高置信度预测可显著提升定位精度，有效解决深度学习方法缺乏可靠性评估的问题。**

- **链接: [http://arxiv.org/pdf/2510.22682v1](http://arxiv.org/pdf/2510.22682v1)**

> **作者:** Bar Shaybet; Vladimir Tourbabin; Boaz Rafaely
>
> **备注:** In submission process to the IEEE Transactions on Audio, Speech and Language Processing, 2025
>
> **摘要:** Accurate Direction-of-Arrival (DOA) estimation in reverberant environments remains a fundamental challenge for spatial audio applications. While deep learning methods have shown strong performance in such conditions, they typically lack a mechanism to assess the reliability of their predictions - an essential feature for real-world deployment. In this work, we present the SRP-PHAT-NET, a deep neural network framework that leverages SRP-PHAT directional maps as spatial features and introduces a built-in reliability estimation. To enable meaningful reliability scoring, the model is trained using Gaussian-weighted labels centered around the true direction. We systematically analyze the influence of label smoothing on accuracy and reliability, demonstrating that the choice of Gaussian kernel width can be tuned to application-specific requirements. Experimental results show that selectively using high-confidence predictions yields significantly improved localization accuracy, highlighting the practical benefits of integrating reliability into deep learning-based DOA estimation.
>
---
#### [new 015] Bridging the Perceptual - Statistical Gap in Dysarthria Assessment: Why Machine Learning Still Falls Short
- **分类: eess.AS; cs.LG**

- **简介: 该论文聚焦于语音识别中的构音障碍评估任务，旨在解决机器学习模型在自动评估构音障碍时与人类专家表现间的“感知-统计差距”。作者分析了人类感知机制与机器学习表征的差异，提出通过感知启发特征、自监督预训练等策略缩小差距，并倡导临床导向的评估方法。**

- **链接: [http://arxiv.org/pdf/2510.22237v1](http://arxiv.org/pdf/2510.22237v1)**

> **作者:** Krishna Gurugubelli
>
> **摘要:** Automated dysarthria detection and severity assessment from speech have attracted significant research attention due to their potential clinical impact. Despite rapid progress in acoustic modeling and deep learning, models still fall short of human expert performance. This manuscript provides a comprehensive analysis of the reasons behind this gap, emphasizing a conceptual divergence we term the ``perceptual-statistical gap''. We detail human expert perceptual processes, survey machine learning representations and methods, review existing literature on feature sets and modeling strategies, and present a theoretical analysis of limits imposed by label noise and inter-rater variability. We further outline practical strategies to narrow the gap, perceptually motivated features, self-supervised pretraining, ASR-informed objectives, multimodal fusion, human-in-the-loop training, and explainability methods. Finally, we propose experimental protocols and evaluation metrics aligned with clinical goals to guide future research toward clinically reliable and interpretable dysarthria assessment tools.
>
---
#### [new 016] Binaural Signal Matching with Wearable Arrays for Near-Field Sources and Directional Focus
- **分类: eess.AS**

- **简介: 该论文研究可穿戴麦克风阵列在近场声源下的双耳信号匹配（BSM）技术。针对传统远场BSM在近场失效的问题，提出近场扩展的NF-BSM，并引入视场加权（FoV）提升感知质量。通过仿真与听音实验验证，新方法在近距离和头部转动下显著优于传统方法，提升了可穿戴空间音频系统的近场再现性能。**

- **链接: [http://arxiv.org/pdf/2510.22258v1](http://arxiv.org/pdf/2510.22258v1)**

> **作者:** Sapir Goldring; Zamir Ben Hur; David Lou Alon; Chad McKell; Sebastian Prepelita; Boaz Rafaely
>
> **摘要:** This paper investigates the performance of Binaural Signal Matching (BSM) methods for near-field sound reproduction using a wearable glasses-mounted microphone array. BSM is a flexible, signal-independent approach for binaural rendering with arbitrary arrays, but its conventional formulation assumes far-field sources. In our previous work, we proposed a near-field extension of BSM (NF-BSM) that incorporates distance-dependent modeling and showed improved performance over far-field BSM using analytic data, though degradation persisted for sources very close to the array. In this study, we extend that analysis by using realistic simulated data of near-field Head-Related Transfer Functions (HRTFs) and Acoustic Transfer Functions (ATFs) of the array, accounting for listener head rotation and evaluating binaural cues such as interaural level and time differences (ILD and ITD). A key contribution is the introduction of a Field of View (FoV) weighting, designed to emphasize perceptually relevant directions and improve robustness under challenging conditions. Results from both simulation and a listening test confirm that NF-BSM outperforms traditional far-field BSM in near-field scenarios, and that the proposed NF-FoV-BSM method achieves the best perceptual and objective quality among all tested methods, particularly at close source distances and under head rotation. These findings highlight the limitations for far-field models in near-field sources and demonstrate that incorporating source distance and directional weighting can significantly improve binaural reproduction performance for wearable spatial audio systems.
>
---
#### [new 017] A Unified Framework for Direction and Diffuseness Estimation Using Tight-Frame Microphone Arrays
- **分类: eess.AS**

- **简介: 该论文提出一种统一框架，用于基于不同空间配置麦克风阵列估计声场方向与扩散度。针对传统方法依赖模式白化或球谐分解的问题，提出仅用速度协方差的方案，实现跨阵列的一致性评估。通过仿真与实验对比A格式、刚性球和新提出的紧框架阵列，验证了紧框架阵列在保持紧凑结构下接近高阶球阵的性能，支持宽带空间声场稳健表征。**

- **链接: [http://arxiv.org/pdf/2510.22183v1](http://arxiv.org/pdf/2510.22183v1)**

> **作者:** Akira Omoto
>
> **备注:** 36 pages including 14 files
>
> **摘要:** This work presents a unified framework for estimating both sound-field direction and diffuseness using practical microphone arrays with different spatial configurations. Building on covariance-based diffuseness models, we formulate a velocity-only covariance approach that enables consistent diffuseness evaluation across heterogeneous array geometries without requiring mode whitening or spherical-harmonic decomposition. Three array types -- an A-format array, a rigid-sphere array, and a newly proposed tight-frame array -- are modeled and compared through both simulations and measurement-based experiments. The results show that the tight-frame configuration achieves near-isotropic directional sampling and reproduces diffuseness characteristics comparable to those of higher-order spherical arrays, while maintaining a compact physical structure. We further examine the accuracy of direction-of-arrival estimation based on acoustic intensity within the same framework. These findings connect theoretical diffuseness analysis with implementable array designs and support the development of robust, broadband methods for spatial-sound-field characterization.
>
---
#### [new 018] Matching Reverberant Speech Through Learned Acoustic Embeddings and Feedback Delay Networks
- **分类: eess.AS**

- **简介: 该论文针对听觉增强现实中实时生成逼真混响的难题，提出基于学习声学先验的盲估计方法。通过反馈延迟网络（FDN）建模目标空间的频率相关衰减与直达声比，实现无需精确测量的高效、感知一致混响渲染。**

- **链接: [http://arxiv.org/pdf/2510.23158v1](http://arxiv.org/pdf/2510.23158v1)**

> **作者:** Philipp Götz; Gloria Dal Santo; Sebastian J. Schlecht; Vesa Välimäki; Emanuël A. P. Habets
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Reverberation conveys critical acoustic cues about the environment, supporting spatial awareness and immersion. For auditory augmented reality (AAR) systems, generating perceptually plausible reverberation in real time remains a key challenge, especially when explicit acoustic measurements are unavailable. We address this by formulating blind estimation of artificial reverberation parameters as a reverberant signal matching task, leveraging a learned room-acoustic prior. Furthermore, we propose a feedback delay network (FDN) structure that reproduces both frequency-dependent decay times and the direct-to-reverberation ratio of a target space. Experimental evaluation against a leading automatic FDN tuning method demonstrates improvements in estimated room-acoustic parameters and perceptual plausibility of artificial reverberant speech. These results highlight the potential of our approach for efficient, perceptually consistent reverberation rendering in AAR applications.
>
---
#### [new 019] HyBeam: Hybrid Microphone-Beamforming Array-Agnostic Speech Enhancement for Wearables
- **分类: eess.AS; eess.SP**

- **简介: 该论文针对可穿戴设备语音增强任务，解决传统方法对麦克风阵列几何形状依赖性强的问题。提出HyBeam框架，融合低频原始麦克风信号与高频波束成形信号，实现阵列无关的混合增强，显著提升不同场景下的语音质量与清晰度。**

- **链接: [http://arxiv.org/pdf/2510.22637v1](http://arxiv.org/pdf/2510.22637v1)**

> **作者:** Yuval Bar Ilan; Boaz Rafaely; Vladimir Tourbabin
>
> **摘要:** Speech enhancement is a fundamental challenge in signal processing, particularly when robustness is required across diverse acoustic conditions and microphone setups. Deep learning methods have been successful for speech enhancement, but often assume fixed array geometries, limiting their use in mobile, embedded, and wearable devices. Existing array-agnostic approaches typically rely on either raw microphone signals or beamformer outputs, but both have drawbacks under changing geometries. We introduce HyBeam, a hybrid framework that uses raw microphone signals at low frequencies and beamformer signals at higher frequencies, exploiting their complementary strengths while remaining highly array-agnostic. Simulations across diverse rooms and wearable array configurations demonstrate that HyBeam consistently surpasses microphone-only and beamformer-only baselines in PESQ, STOI, and SI-SDR. A bandwise analysis shows that the hybrid approach leverages beamformer directivity at high frequencies and microphone cues at low frequencies, outperforming either method alone across all bands.
>
---
#### [new 020] DialoSpeech: Dual-Speaker Dialogue Generation with LLM and Flow Matching
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出DialoSpeech，解决多轮双说话人对话语音合成中自然度、连贯性与交互动态性不足的问题。结合大语言模型与分块流匹配，构建双轨架构，支持中英文及跨语言对话生成，并设计数据处理流程以提升训练效率与效果。**

- **链接: [http://arxiv.org/pdf/2510.08373v1](http://arxiv.org/pdf/2510.08373v1)**

> **作者:** Hanke Xie; Dake Guo; Chengyou Wang; Yue Li; Wenjie Tian; Xinfa Zhu; Xinsheng Wang; Xiulin Li; Guanqiong Miao; Bo Liu; Lei Xie
>
> **摘要:** Recent advances in text-to-speech (TTS) synthesis, particularly those leveraging large language models (LLMs), have significantly improved expressiveness and naturalness. However, generating human-like, interactive dialogue speech remains challenging. Current systems face limitations due to the scarcity of dual-track data and difficulties in achieving naturalness, contextual coherence, and interactional dynamics, such as turn-taking, overlapping speech, and speaker consistency, in multi-turn conversations. To address these challenges, we propose DialoSpeech, a dual-track architecture combining a large language model with Chunked Flow Matching for expressive, human-like dialogue speech synthesis. DialoSpeech generates natural multi-turn conversations with coherent speaker turns and natural overlaps, supporting both Chinese and English and cross-lingual speech synthesis. We introduce a data processing pipeline to construct dual-track dialogue datasets, facilitating scalable training and experimental validation. Experiments show that our model outperforms baselines, offering a solution for generating human-like spoken dialogues. Audio samples are available at https://tiamojames.github.io/DialoSpeech
>
---
#### [new 021] Empowering Multimodal Respiratory Sound Classification with Counterfactual Adversarial Debiasing for Out-of-Distribution Robustness
- **分类: eess.AS**

- **简介: 该论文针对多模态呼吸音分类中的分布外泛化问题，旨在消除年龄、性别等非因果属性带来的虚假关联。提出基于因果图的反事实去偏、对抗去偏及反事实元数据增强方法，学习与元数据无关的鲁棒表示，显著提升模型在跨机构数据上的性能。**

- **链接: [http://arxiv.org/pdf/2510.22263v1](http://arxiv.org/pdf/2510.22263v1)**

> **作者:** Heejoon Koo; Miika Toikkanen; Yoon Tae Kim; Soo Yong Kim; June-Woo Kim
>
> **备注:** 3 figures, 4 Tables, and 5 pages
>
> **摘要:** Multimodal respiratory sound classification offers promise for early pulmonary disease detection by integrating bioacoustic signals with patient metadata. Nevertheless, current approaches remain vulnerable to spurious correlations from attributes such as age, sex, or acquisition device, which hinder their generalization, especially under distribution shifts across clinical sites. To this end, we propose a counterfactual adversarial debiasing framework. First, we employ a causal graph-based counterfactual debiasing strategy to suppress non-causal dependencies from patient metadata. Second, we introduce adversarial debiasing to learn metadata-insensitive representations and reduce metadata-specific biases. Third, we design counterfactual metadata augmentation to mitigate spurious correlations further and strengthen metadata-invariant representations. By doing so, our method consistently outperforms strong baselines in evaluations under both in-distribution and distribution shifts. The code is available at https://github.com/RSC-Toolkit/BTS-CARD.
>
---
#### [new 022] ISA-Bench: Benchmarking Instruction Sensitivity for Large Audio Language Models
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文针对大音频语言模型（LALMs）对指令表述高度敏感的问题，提出ISA-Bench基准，从指令描述、输出格式、任务组合三方面系统评估其敏感性。通过实验发现主流LALMs性能受指令影响显著，进而提出微调策略提升指令遵循能力，但引发灾难性遗忘。研究为构建鲁棒音频理解系统提供标准化评估与改进路径。**

- **链接: [http://arxiv.org/pdf/2510.23558v1](http://arxiv.org/pdf/2510.23558v1)**

> **作者:** Bohan Li; Wenbin Huang; Yuhang Qiu; Yiwei Guo; Hankun Wang; Zhihan Li; Jing Peng; Ziyang Ma; Xie Chen; Kai Yu
>
> **备注:** submitted to icassp 2026
>
> **摘要:** Large Audio Language Models (LALMs), which couple acoustic perception with large language models (LLMs) to extract and understand diverse information from audio, have attracted intense interest from both academic and industrial communities. However, existing LALMs are highly sensitive to how instructions are phrased, affecting both (i) instruction-following rates and (ii) task performance. Yet, no existing benchmarks offer a systematic and comprehensive evaluation of this sensitivity. We introduce ISA-Bench, a dynamic benchmark evaluating instruction sensitivity for LALMs along three axes: instruction description, output format, and task composition. We assess recent open-source and proprietary LALMs using ISA-Bench, profiling both compliance and accuracy under controlled instruction variations. Experimental results reveal that even state-of-the-art LALMs suffer significant instruction sensitivity, leading to degraded performance on fundamental audio understanding tasks. To mitigate this issue, we fine-tune Qwen2-Audio on a specifically constructed complex instruction-variant dataset, achieving a marked improvement in instruction-following performance. However, this also induces nontrivial catastrophic forgetting: the model loses some previously mastered task capabilities when exposed to new instruction styles. Our benchmark provides a standardized basis for assessing and improving instruction sensitivity in LALMs, underscoring the need for instruction-robust audio understanding in real-world pipelines.
>
---
#### [new 023] M-CIF: Multi-Scale Alignment For CIF-Based Non-Autoregressive ASR
- **分类: cs.SD; cs.CL**

- **简介: 该论文针对非自回归语音识别中的对齐不稳问题，提出多尺度CIF（M-CIF）机制，通过融合字符与音素层级监督，逐步优化子词表示的对齐。实验表明，M-CIF显著降低WER，尤其在德语和法语上提升明显，验证了多层级监督对增强声文对齐的有效性。**

- **链接: [http://arxiv.org/pdf/2510.22172v1](http://arxiv.org/pdf/2510.22172v1)**

> **作者:** Ruixiang Mao; Xiangnan Ma; Qing Yang; Ziming Zhu; Yucheng Qiao; Yuan Ge; Tong Xiao; Shengxiang Gao; Zhengtao Yu; Jingbo Zhu
>
> **摘要:** The Continuous Integrate-and-Fire (CIF) mechanism provides effective alignment for non-autoregressive (NAR) speech recognition. This mechanism creates a smooth and monotonic mapping from acoustic features to target tokens, achieving performance on Mandarin competitive with other NAR approaches. However, without finer-grained guidance, its stability degrades in some languages such as English and French. In this paper, we propose Multi-scale CIF (M-CIF), which performs multi-level alignment by integrating character and phoneme level supervision progressively distilled into subword representations, thereby enhancing robust acoustic-text alignment. Experiments show that M-CIF reduces WER compared to the Paraformer baseline, especially on CommonVoice by 4.21% in German and 3.05% in French. To further investigate these gains, we define phonetic confusion errors (PE) and space-related segmentation errors (SE) as evaluation metrics. Analysis of these metrics across different M-CIF settings reveals that the phoneme and character layers are essential for enhancing progressive CIF alignment.
>
---
#### [new 024] Evaluation of Spherical Wavelet Framework in Comparsion with Ambisonics
- **分类: eess.AS**

- **简介: 该论文属于音频空间表示任务，旨在比较球面小波框架（SWF）与声学球谐函数（Ambisonics）的性能。研究通过感知指标（IACC、ITD、ILD）和听音测试，评估不同网格下SWF与Ambisonics的空间与音色保真度。结果表明，SWF在多数情况下更接近参考，但对球面划分敏感且无法自然表示连续方向入射波。**

- **链接: [http://arxiv.org/pdf/2510.23403v1](http://arxiv.org/pdf/2510.23403v1)**

> **作者:** Ş. Ekmen; H. Lee
>
> **备注:** 13 pages, 8 figures. Submitted to IEEE TASLP
>
> **摘要:** Recently, the Spherical Wavelet Framework (SWF) was proposed to combine the benefits of Ambisonics and Object-Based Audio (OBA) by utilising highly localised basis functions. SWF can enhance the sweet-spot area and reduce localisation blur while still enabling a sparse representation of the complete sound field, making storage and transmission more efficient. Initial vector analysis and listening test of SWF have shown promising results; however, these findings are limited to very specific conditions and do not include perceptual metrics. The present study investigates SWF in greater detail, comparing it with Ambisonics. The comparison was carried out using IACC, ITD, and ILD estimations, as well as listening tests with ecologically valid sound sources. Various reproduction layouts: regular polyhedron, t-design, and Lebedev grid with their corresponding Ambisonics orders and channel counts were evaluated. Results indicate that SWF is rated significantly more similar to the reference than Ambisonics is, in terms of overall spatial and timbral fidelity; however, it is considerably dependent on the subdivison of the sphere. Moreover, it cannot natively represent a wave arriving at a continuous direction. Possible solutions are proposed.
>
---
#### [new 025] FOA Tokenizer: Low-bitrate Neural Codec for First Order Ambisonics with Spatial Consistency Loss
- **分类: cs.SD**

- **简介: 该论文提出首个面向一阶全向声场（FOA）的离散神经音频编解码器FOA Tokenizer，解决低比特率下空间音频压缩与方向保真问题。基于WavTokenizer架构，引入空间一致性损失，实现24kHz四通道信号每秒75个离散标记（0.9kbps），有效保留声源方向信息，并在多个场景下验证了高重建精度及对下游任务的适用性。**

- **链接: [http://arxiv.org/pdf/2510.22241v1](http://arxiv.org/pdf/2510.22241v1)**

> **作者:** Parthasaarathy Sudarsanam; Sebastian Braun; Hannes Gamper
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Neural audio codecs have been widely studied for mono and stereo signals, but spatial audio remains largely unexplored. We present the first discrete neural spatial audio codec for first-order ambisonics (FOA). Building on the WavTokenizer architecture, we extend it to support four-channel FOA signals and introduce a novel spatial consistency loss to preserve directional cues in the reconstructed signals under a highly compressed representation. Our codec compresses 4-channel FOA audio at 24 kHz into 75 discrete tokens per second, corresponding to a bit rate of 0.9 kbps. Evaluations on simulated reverberant mixtures, non-reverberant clean speech, and FOA mixtures with real room impulse responses show accurate reconstruction, with mean angular errors of 13.76{\deg}, 3.96{\deg}, and 25.83{\deg}, respectively, across the three conditions. In addition, discrete latent representations derived from our codec provide useful features for downstream spatial audio tasks, as demonstrated on sound event localization and detection with STARSS23 real recordings.
>
---
#### [new 026] DiffRhythm 2: Efficient and High Fidelity Song Generation via Block Flow Matching
- **分类: eess.AS**

- **简介: 该论文提出DiffRhythm 2，面向高保真可控歌曲生成任务。针对歌词与演唱对齐难、多偏好优化性能下降问题，提出基于块流匹配的半自回归架构与跨对偏好优化方法，结合音乐VAE与随机块对齐损失，实现高效高质量长序列歌曲生成。**

- **链接: [http://arxiv.org/pdf/2510.22950v1](http://arxiv.org/pdf/2510.22950v1)**

> **作者:** Yuepeng Jiang; Huakang Chen; Ziqian Ning; Jixun Yao; Zerui Han; Di Wu; Meng Meng; Jian Luan; Zhonghua Fu; Lei Xie
>
> **摘要:** Generating full-length, high-quality songs is challenging, as it requires maintaining long-term coherence both across text and music modalities and within the music modality itself. Existing non-autoregressive (NAR) frameworks, while capable of producing high-quality songs, often struggle with the alignment between lyrics and vocal. Concurrently, catering to diverse musical preferences necessitates reinforcement learning from human feedback (RLHF). However, existing methods often rely on merging multiple models during multi-preference optimization, which results in significant performance degradation. To address these challenges, we introduce DiffRhythm 2, an end-to-end framework designed for high-fidelity, controllable song generation. To tackle the lyric alignment problem, DiffRhythm 2 employs a semi-autoregressive architecture based on block flow matching. This design enables faithful alignment of lyrics to singing vocals without relying on external labels and constraints, all while preserving the high generation quality and efficiency of NAR models. To make this framework computationally tractable for long sequences, we implement a music variational autoencoder (VAE) that achieves a low frame rate of 5 Hz while still enabling high-fidelity audio reconstruction. In addition, to overcome the limitations of multi-preference optimization in RLHF, we propose cross-pair preference optimization. This method effectively mitigates the performance drop typically associated with model merging, allowing for more robust optimization across diverse human preferences. We further enhance musicality and structural coherence by introducing stochastic block representation alignment loss.
>
---
#### [new 027] TwinShift: Benchmarking Audio Deepfake Detection across Synthesizer and Speaker Shifts
- **分类: cs.SD**

- **简介: 该论文针对音频深度伪造检测（ADD）模型泛化能力弱的问题，提出TWINSHIFT基准。通过六种合成系统与不重叠说话人组合，严格测试模型在未知生成方法和说话人下的鲁棒性，揭示现有系统局限并提供改进方向。**

- **链接: [http://arxiv.org/pdf/2510.23096v1](http://arxiv.org/pdf/2510.23096v1)**

> **作者:** Jiyoung Hong; Yoonseo Chung; Seungyeon Oh; Juntae Kim; Jiyoung Lee; Sookyung Kim; Hyunsoo Cho
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Audio deepfakes pose a growing threat, already exploited in fraud and misinformation. A key challenge is ensuring detectors remain robust to unseen synthesis methods and diverse speakers, since generation techniques evolve quickly. Despite strong benchmark results, current systems struggle to generalize to new conditions limiting real-world reliability. To address this, we introduce TWINSHIFT, a benchmark explicitly designed to evaluate detection robustness under strictly unseen conditions. Our benchmark is constructed from six different synthesis systems, each paired with disjoint sets of speakers, allowing for a rigorous assessment of how well detectors generalize when both the generative model and the speaker identity change. Through extensive experiments, we show that TWINSHIFT reveals important robustness gaps, uncover overlooked limitations, and provide principled guidance for developing ADD systems. The TWINSHIFT benchmark can be accessed at https://github.com/intheMeantime/TWINSHIFT.
>
---
#### [new 028] Arabic Little STT: Arabic Children Speech Recognition Dataset
- **分类: cs.CL; cs.AI; cs.HC; cs.LG; cs.SD**

- **简介: 该论文聚焦于阿拉伯语儿童语音识别任务，针对低资源语言及儿童语音数据稀缺问题，构建了首个黎凡特阿拉伯语儿童语音数据集Arabic Little STT（355条来自288名6-13岁儿童的语音）。通过评估Whisper模型在该数据集上的表现，发现其词错误率高达0.66，显著高于成人数据集，凸显儿童语音识别的挑战。研究呼吁建立专用儿童语音基准与伦理合规的数据集，推动更公平的语音技术发展。**

- **链接: [http://arxiv.org/pdf/2510.23319v1](http://arxiv.org/pdf/2510.23319v1)**

> **作者:** Mouhand Alkadri; Dania Desouki; Khloud Al Jallad
>
> **摘要:** The performance of Artificial Intelligence (AI) systems fundamentally depends on high-quality training data. However, low-resource languages like Arabic suffer from severe data scarcity. Moreover, the absence of child-specific speech corpora is an essential gap that poses significant challenges. To address this gap, we present our created dataset, Arabic Little STT, a dataset of Levantine Arabic child speech recorded in classrooms, containing 355 utterances from 288 children (ages 6 - 13). We further conduct a systematic assessment of Whisper, a state-of-the-art automatic speech recognition (ASR) model, on this dataset and compare its performance with adult Arabic benchmarks. Our evaluation across eight Whisper variants reveals that even the best-performing model (Large_v3) struggles significantly, achieving a 0.66 word error rate (WER) on child speech, starkly contrasting with its sub 0.20 WER on adult datasets. These results align with other research on English speech. Results highlight the critical need for dedicated child speech benchmarks and inclusive training data in ASR development. Emphasizing that such data must be governed by strict ethical and privacy frameworks to protect sensitive child information. We hope that this study provides an initial step for future work on equitable speech technologies for Arabic-speaking children. We hope that our publicly available dataset enrich the children's demographic representation in ASR datasets.
>
---
#### [new 029] Beyond IVR Touch-Tones: Customer Intent Routing using LLMs
- **分类: cs.HC; cs.AI; cs.CL; eess.AS**

- **简介: 该论文针对传统IVR系统因固定按键导致用户体验差的问题，提出基于大语言模型（LLM）的用户意图路由方法。通过构建虚拟IVR结构与生成用户意图数据，比较不同提示设计的效果，验证了LLM在自然语言理解与路径匹配中的可行性，实现了更智能、无缝的客户语音服务引导。**

- **链接: [http://arxiv.org/pdf/2510.21715v1](http://arxiv.org/pdf/2510.21715v1)**

> **作者:** Sergio Rojas-Galeano
>
> **备注:** Accepted for publication in the Proceedings of the Workshop on Engineering Applications 2025 (WEA 2025)
>
> **摘要:** Widespread frustration with rigid touch-tone Interactive Voice Response (IVR) systems for customer service underscores the need for more direct and intuitive language interaction. While speech technologies are necessary, the key challenge lies in routing intents from user phrasings to IVR menu paths, a task where Large Language Models (LLMs) show strong potential. Progress, however, is limited by data scarcity, as real IVR structures and interactions are often proprietary. We present a novel LLM-based methodology to address this gap. Using three distinct models, we synthesized a realistic 23-node IVR structure, generated 920 user intents (230 base and 690 augmented), and performed the routing task. We evaluate two prompt designs: descriptive hierarchical menus and flattened path representations, across both base and augmented datasets. Results show that flattened paths consistently yield higher accuracy, reaching 89.13% on the base dataset compared to 81.30% with the descriptive format, while augmentation introduces linguistic noise that slightly reduces performance. Confusion matrix analysis further suggests that low-performing routes may reflect not only model limitations but also redundancies in menu design. Overall, our findings demonstrate proof-of-concept that LLMs can enable IVR routing through a smoother, more seamless user experience -- moving customer service one step ahead of touch-tone menus.
>
---
#### [new 030] Quantifying Multimodal Imbalance: A GMM-Guided Adaptive Loss for Audio-Visual Learning
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **简介: 该论文针对音视频多模态学习中的模态不平衡问题，提出基于高斯混合模型（GMM）的量化分析方法，通过“模态差距”建模识别平衡与不平衡样本，并设计三目标自适应损失函数，结合两阶段训练策略，显著提升模型性能，在CREMA-D和AVE数据集上达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2510.21797v1](http://arxiv.org/pdf/2510.21797v1)**

> **作者:** Zhaocheng Liu; Zhiwen Yu; Xiaoqing Liu
>
> **摘要:** Current mainstream approaches to addressing multimodal imbalance primarily focus on architectural modifications and optimization-based, often overlooking a quantitative analysis of the imbalance degree between modalities. To address this gap, our work introduces a novel method for the quantitative analysis of multi-modal imbalance, which in turn informs the design of a sample-level adaptive loss function.We begin by defining the "Modality Gap" as the difference between the Softmax scores of different modalities (e.g., audio and visual) for the ground-truth class prediction. Analysis of the Modality Gap distribution reveals that it can be effectively modeled by a bimodal Gaussian Mixture Model (GMM). These two components are found to correspond respectively to "modality-balanced" and "modality-imbalanced" data samples. Subsequently, we apply Bayes' theorem to compute the posterior probability of each sample belonging to these two distinct distributions.Informed by this quantitative analysis, we design a novel adaptive loss function with three objectives: (1) to minimize the overall Modality Gap; (2) to encourage the imbalanced sample distribution to shift towards the balanced one; and (3) to apply greater penalty weights to imbalanced samples. We employ a two-stage training strategy consisting of a warm-up phase followed by an adaptive training phase.Experimental results demonstrate that our approach achieves state-of-the-art (SOTA) performance on the public CREMA-D and AVE datasets, attaining accuracies of $80.65\%$ and $70.90\%$, respectively. This validates the effectiveness of our proposed methodology.
>
---
## 更新

#### [replaced 001] WhaleVAD-BPN: Improving Baleen Whale Call Detection with Boundary Proposal Networks and Post-processing Optimisation
- **分类: eess.AS; cs.AI; cs.LG; cs.SD; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2510.21280v2](http://arxiv.org/pdf/2510.21280v2)**

> **作者:** Christiaan M. Geldenhuys; Günther Tonitz; Thomas R. Niesler
>
> **摘要:** While recent sound event detection (SED) systems can identify baleen whale calls in marine audio, challenges related to false positive and minority-class detection persist. We propose the boundary proposal network (BPN), which extends an existing lightweight SED system. The BPN is inspired by work in image object detection and aims to reduce the number of false positive detections. It achieves this by using intermediate latent representations computed within the backbone classification model to gate the final output. When added to an existing SED system, the BPN achieves a 16.8 % absolute increase in precision, as well as 21.3 % and 9.4 % improvements in the F1-score for minority-class d-calls and bp-calls, respectively. We further consider two approaches to the selection of post-processing hyperparameters: a forward-search and a backward-search. By separately optimising event-level and frame-level hyperparameters, these two approaches lead to considerable performance improvements over parameters selected using empirical methods. The complete WhaleVAD-BPN system achieves a cross-validated development F1-score of 0.475, which is a 9.8 % absolute improvement over the baseline.
>
---
#### [replaced 002] ARIONet: An Advanced Self-supervised Contrastive Representation Network for Birdsong Classification and Future Frame Prediction
- **分类: cs.SD**

- **链接: [http://arxiv.org/pdf/2510.00522v3](http://arxiv.org/pdf/2510.00522v3)**

> **作者:** Md. Abdur Rahman; Selvarajah Thuseethan; Kheng Cher Yeo; Reem E. Mohamed; Sami Azam
>
> **摘要:** Automated birdsong classification is essential for advancing ecological monitoring and biodiversity studies. Despite recent progress, existing methods often depend heavily on labeled data, use limited feature representations, and overlook temporal dynamics essential for accurate species identification. In this work, we propose a self-supervised contrastive network, ARIONet (Acoustic Representation for Interframe Objective Network), that jointly optimizes contrastive classification and future frame prediction using augmented audio representations. The model simultaneously integrates multiple complementary audio features within a transformer-based encoder model. Our framework is designed with two key objectives: (1) to learn discriminative species-specific representations for contrastive learning through maximizing similarity between augmented views of the same audio segment while pushing apart different samples, and (2) to model temporal dynamics by predicting future audio frames, both without requiring large-scale annotations. We validate our framework on four diverse birdsong datasets, including the British Birdsong Dataset, Bird Song Dataset, and two extended Xeno-Canto subsets (A-M and N-Z). Our method consistently outperforms existing baselines and achieves classification accuracies of 98.41%, 93.07%, 91.89%, and 91.58%, and F1-scores of 97.84%, 94.10%, 91.29%, and 90.94%, respectively. Furthermore, it demonstrates low mean absolute errors and high cosine similarity, up to 95\%, in future frame prediction tasks. Extensive experiments further confirm the effectiveness of our self-supervised learning strategy in capturing complex acoustic patterns and temporal dependencies, as well as its potential for real-world applicability in ecological conservation and monitoring.
>
---
#### [replaced 003] FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks
- **分类: cs.LG; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.04465v2](http://arxiv.org/pdf/2502.04465v2)**

> **作者:** Luca Della Libera; Francesco Paissan; Cem Subakan; Mirco Ravanelli
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Large language models have revolutionized natural language processing through self-supervised pretraining on massive datasets. Inspired by this success, researchers have explored adapting these methods to speech by discretizing continuous audio into tokens using neural audio codecs. However, existing approaches face limitations, including high bitrates, the loss of either semantic or acoustic information, and the reliance on multi-codebook designs when trying to capture both, which increases architectural complexity for downstream tasks. To address these challenges, we introduce FocalCodec, an efficient low-bitrate codec based on focal modulation that utilizes a single binary codebook to compress speech between 0.16 and 0.65 kbps. FocalCodec delivers competitive performance in speech resynthesis and voice conversion at lower bitrates than the current state-of-the-art, while effectively handling multilingual speech and noisy environments. Evaluation on downstream tasks shows that FocalCodec successfully preserves sufficient semantic and acoustic information, while also being well-suited for generative modeling. Demo samples and code are available at https://lucadellalib.github.io/focalcodec-web/.
>
---
#### [replaced 004] Detect Any Sound: Open-Vocabulary Sound Event Detection with Multi-Modal Queries
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16343v2](http://arxiv.org/pdf/2507.16343v2)**

> **作者:** Pengfei Cai; Yan Song; Qing Gu; Nan Jiang; Haoyu Song; Ian McLoughlin
>
> **备注:** Accepted by MM 2025
>
> **摘要:** Most existing sound event detection~(SED) algorithms operate under a closed-set assumption, restricting their detection capabilities to predefined classes. While recent efforts have explored language-driven zero-shot SED by exploiting audio-language models, their performance is still far from satisfactory due to the lack of fine-grained alignment and cross-modal feature fusion. In this work, we propose the Detect Any Sound Model (DASM), a query-based framework for open-vocabulary SED guided by multi-modal queries. DASM formulates SED as a frame-level retrieval task, where audio features are matched against query vectors derived from text or audio prompts. To support this formulation, DASM introduces a dual-stream decoder that explicitly decouples event recognition and temporal localization: a cross-modality event decoder performs query-feature fusion and determines the presence of sound events at the clip-level, while a context network models temporal dependencies for frame-level localization. Additionally, an inference-time attention masking strategy is proposed to leverage semantic relations between base and novel classes, substantially enhancing generalization to novel classes. Experiments on the AudioSet Strong dataset demonstrate that DASM effectively balances localization accuracy with generalization to novel classes, outperforming CLAP-based methods in open-vocabulary setting (+ 7.8 PSDS) and the baseline in the closed-set setting (+ 6.9 PSDS). Furthermore, in cross-dataset zero-shot evaluation on DESED, DASM achieves a PSDS1 score of 42.2, even exceeding the supervised CRNN baseline. The project page is available at https://cai525.github.io/Transformer4SED/demo_page/DASM/.
>
---
#### [replaced 005] Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2504.05657v2](http://arxiv.org/pdf/2504.05657v2)**

> **作者:** Tianchi Liu; Duc-Tuan Truong; Rohan Kumar Das; Kong Aik Lee; Haizhou Li
>
> **备注:** Accepted to IEEE Transactions on Information Forensics and Security
>
> **摘要:** Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.
>
---
#### [replaced 006] ReFESS-QI: Reference-Free Evaluation For Speech Separation With Joint Quality And Intelligibility Scoring
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.21014v2](http://arxiv.org/pdf/2510.21014v2)**

> **作者:** Ari Frummer; Helin Wang; Tianyu Cao; Adi Arbel; Yuval Sieradzki; Oren Gal; Jesús Villalba; Thomas Thebaud; Najim Dehak
>
> **摘要:** Source separation is a crucial pre-processing step for various speech processing tasks, such as automatic speech recognition (ASR). Traditionally, the evaluation metrics for speech separation rely on the matched reference audios and corresponding transcriptions to assess audio quality and intelligibility. However, they cannot be used to evaluate real-world mixtures for which no reference exists. This paper introduces a text-free reference-free evaluation framework based on self-supervised learning (SSL) representations. The proposed framework utilize the mixture and separated tracks to predict jointly audio quality, through the Scale Invariant Signal to Noise Ratio (SI-SNR) metric, and speech intelligibility through the Word Error Rate (WER) metric. We conducted experiments on the WHAMR! dataset, which shows a WER estimation with a mean absolute error (MAE) of 17% and a Pearson correlation coefficient (PCC) of 0.77; and SI-SNR estimation with an MAE of 1.38 and PCC of 0.95. We further demonstrate the robustness of our estimator by using various SSL representations.
>
---
#### [replaced 007] Automatic Music Sample Identification with Multi-Track Contrastive Learning
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.11507v2](http://arxiv.org/pdf/2510.11507v2)**

> **作者:** Alain Riou; Joan Serrà; Yuki Mitsufuji
>
> **摘要:** Sampling, the technique of reusing pieces of existing audio tracks to create new music content, is a very common practice in modern music production. In this paper, we tackle the challenging task of automatic sample identification, that is, detecting such sampled content and retrieving the material from which it originates. To do so, we adopt a self-supervised learning approach that leverages a multi-track dataset to create positive pairs of artificial mixes, and design a novel contrastive learning objective. We show that such method significantly outperforms previous state-of-the-art baselines, that is robust to various genres, and that scales well when increasing the number of noise songs in the reference database. In addition, we extensively analyze the contribution of the different components of our training pipeline and highlight, in particular, the need for high-quality separated stems for this task.
>
---
#### [replaced 008] PESTO: Real-Time Pitch Estimation with Self-supervised Transposition-equivariant Objective
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.01488v2](http://arxiv.org/pdf/2508.01488v2)**

> **作者:** Alain Riou; Bernardo Torres; Ben Hayes; Stefan Lattner; Gaëtan Hadjeres; Gaël Richard; Geoffroy Peeters
>
> **摘要:** In this paper, we introduce PESTO, a self-supervised learning approach for single-pitch estimation using a Siamese architecture. Our model processes individual frames of a Variable-$Q$ Transform (VQT) and predicts pitch distributions. The neural network is designed to be equivariant to translations, notably thanks to a Toeplitz fully-connected layer. In addition, we construct pitch-shifted pairs by translating and cropping the VQT frames and train our model with a novel class-based transposition-equivariant objective, eliminating the need for annotated data. Thanks to this architecture and training objective, our model achieves remarkable performances while being very lightweight ($130$k parameters). Evaluations on music and speech datasets (MIR-1K, MDB-stem-synth, and PTDB) demonstrate that PESTO not only outperforms self-supervised baselines but also competes with supervised methods, exhibiting superior cross-dataset generalization. Finally, we enhance PESTO's practical utility by developing a streamable VQT implementation using cached convolutions. Combined with our model's low latency (less than 10 ms) and minimal parameter count, this makes PESTO particularly suitable for real-time applications.
>
---
#### [replaced 009] OpenS2S: Advancing Fully Open-Source End-to-End Empathetic Large Speech Language Model
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05177v3](http://arxiv.org/pdf/2507.05177v3)**

> **作者:** Chen Wang; Tianyu Peng; Wen Yang; Yinan Bai; Guangfu Wang; Jun Lin; Lanpeng Jia; Lingxiang Wu; Jinqiao Wang; Chengqing Zong; Jiajun Zhang
>
> **备注:** Technical Report, Update on OpenS2S_v1.5
>
> **摘要:** Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at https://casia-lm.github.io/OpenS2S
>
---
#### [replaced 010] $\texttt{AVROBUSTBENCH}$: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.00358v3](http://arxiv.org/pdf/2506.00358v3)**

> **作者:** Sarthak Kumar Maharana; Saksham Singh Kushwaha; Baoming Zhang; Adrian Rodriguez; Songtao Wei; Yapeng Tian; Yunhui Guo
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks
>
> **摘要:** While recent audio-visual models have demonstrated impressive performance, their robustness to distributional shifts at test-time remains not fully understood. Existing robustness benchmarks mainly focus on single modalities, making them insufficient for thoroughly assessing the robustness of audio-visual models. Motivated by real-world scenarios where shifts can occur $\textit{simultaneously}$ in both audio and visual modalities, we introduce $\texttt{AVROBUSTBENCH}$, a comprehensive benchmark designed to evaluate the test-time robustness of audio-visual recognition models. $\texttt{AVROBUSTBENCH}$ comprises four audio-visual benchmark datasets, $\texttt{AUDIOSET-2C}$, $\texttt{VGGSOUND-2C}$, $\texttt{KINETICS-2C}$, and $\texttt{EPICKITCHENS-2C}$, each incorporating 75 bimodal audio-visual corruptions that are $\textit{co-occurring}$ and $\textit{correlated}$. Through extensive evaluations, we observe that state-of-the-art supervised and self-supervised audio-visual models exhibit declining robustness as corruption severity increases. Furthermore, online test-time adaptation (TTA) methods, on $\texttt{VGGSOUND-2C}$ and $\texttt{KINETICS-2C}$, offer minimal improvements in performance under bimodal corruptions. We further propose $\texttt{AV2C}$, a simple TTA approach enabling on-the-fly cross-modal fusion by penalizing high-entropy samples, which achieves improvements on $\texttt{VGGSOUND-2C}$. We hope that $\texttt{AVROBUSTBENCH}$ will steer the development of more effective and robust audio-visual TTA approaches. Our code is available $\href{https://github.com/sarthaxxxxx/AV-C-Robustness-Benchmark}{here}$.
>
---
#### [replaced 011] Audio Does Matter: Importance-Aware Multi-Granularity Fusion for Video Moment Retrieval
- **分类: cs.IR; cs.CV; cs.MM; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.04273v3](http://arxiv.org/pdf/2508.04273v3)**

> **作者:** Junan Lin; Daizong Liu; Xianke Chen; Xiaoye Qu; Xun Yang; Jixiang Zhu; Sanyuan Zhang; Jianfeng Dong
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Video Moment Retrieval (VMR) aims to retrieve a specific moment semantically related to the given query. To tackle this task, most existing VMR methods solely focus on the visual and textual modalities while neglecting the complementary but important audio modality. Although a few recent works try to tackle the joint audio-vision-text reasoning, they treat all modalities equally and simply embed them without fine-grained interaction for moment retrieval. These designs are counter-practical as: Not all audios are helpful for video moment retrieval, and the audio of some videos may be complete noise or background sound that is meaningless to the moment determination. To this end, we propose a novel Importance-aware Multi-Granularity fusion model (IMG), which learns to dynamically and selectively aggregate the audio-vision-text contexts for VMR. Specifically, after integrating the textual guidance with vision and audio separately, we first design a pseudo-label-supervised audio importance predictor that predicts the importance score of the audio, and accordingly assigns weights to mitigate the interference caused by noisy audio. Then, we design a multi-granularity audio fusion module that adaptively fuses audio and visual modalities at local-, event-, and global-level, fully capturing their complementary contexts. We further propose a cross-modal knowledge distillation strategy to address the challenge of missing audio modality during inference. To evaluate our method, we further construct a new VMR dataset, i.e., Charades-AudioMatter, where audio-related samples are manually selected and re-organized from the original Charades-STA to validate the model's capability in utilizing audio modality. Extensive experiments validate the effectiveness of our method, achieving state-of-the-art with audio-video fusion in VMR methods. Our code is available at https://github.com/HuiGuanLab/IMG.
>
---
#### [replaced 012] Pindrop it! Audio and Visual Deepfake Countermeasures for Robust Detection and Fine Grained-Localization
- **分类: cs.CV; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.08141v2](http://arxiv.org/pdf/2508.08141v2)**

> **作者:** Nicholas Klein; Hemlata Tak; James Fullwood; Krishna Regmi; Leonidas Spinoulas; Ganesh Sivaraman; Tianxiang Chen; Elie Khoury
>
> **摘要:** The field of visual and audio generation is burgeoning with new state-of-the-art methods. This rapid proliferation of new techniques underscores the need for robust solutions for detecting synthetic content in videos. In particular, when fine-grained alterations via localized manipulations are performed in visual, audio, or both domains, these subtle modifications add challenges to the detection algorithms. This paper presents solutions for the problems of deepfake video classification and localization. The methods were submitted to the ACM 1M Deepfakes Detection Challenge, achieving the best performance in the temporal localization task and a top four ranking in the classification task for the TestA split of the evaluation dataset.
>
---
#### [replaced 013] PESTO: Pitch Estimation with Self-supervised Transposition-equivariant Objective
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2309.02265v3](http://arxiv.org/pdf/2309.02265v3)**

> **作者:** Alain Riou; Stefan Lattner; Gaëtan Hadjeres; Geoffroy Peeters
>
> **备注:** Best Paper Award of the 24th International Society for Music Information Retrieval Conference, ISMIR 2023
>
> **摘要:** In this paper, we address the problem of pitch estimation using Self Supervised Learning (SSL). The SSL paradigm we use is equivariance to pitch transposition, which enables our model to accurately perform pitch estimation on monophonic audio after being trained only on a small unlabeled dataset. We use a lightweight ($<$ 30k parameters) Siamese neural network that takes as inputs two different pitch-shifted versions of the same audio represented by its Constant-Q Transform. To prevent the model from collapsing in an encoder-only setting, we propose a novel class-based transposition-equivariant objective which captures pitch information. Furthermore, we design the architecture of our network to be transposition-preserving by introducing learnable Toeplitz matrices. We evaluate our model for the two tasks of singing voice and musical instrument pitch estimation and show that our model is able to generalize across tasks and datasets while being lightweight, hence remaining compatible with low-resource devices and suitable for real-time applications. In particular, our results surpass self-supervised baselines and narrow the performance gap between self-supervised and supervised methods for pitch estimation.
>
---
#### [replaced 014] Memristive Nanowire Network for Energy Efficient Audio Classification: Pre-Processing-Free Reservoir Computing with Reduced Latency
- **分类: cs.SD; cond-mat.dis-nn; eess.AS; physics.app-ph; stat.CO**

- **链接: [http://arxiv.org/pdf/2411.19611v2](http://arxiv.org/pdf/2411.19611v2)**

> **作者:** Akshaya Rajesh; Pavithra Ananthasubramanian; Nagarajan Raghavan; Ankush Kumar
>
> **备注:** 14 pages, 5 Figures
>
> **摘要:** Efficient audio feature extraction is critical for low-latency, resource-constrained speech recognition. Conventional preprocessing techniques, such as Mel Spectrogram, Perceptual Linear Prediction (PLP), and Learnable Spectrogram, achieve high classification accuracy but require large feature sets and significant computation. The low-latency and power efficiency benefits of neuromorphic computing offer a strong potential for audio classification. Here, we introduce memristive nanowire networks as a neuromorphic hardware preprocessing layer for spoken-digit classification, a capability not previously demonstrated. Nanowire networks extract compact, informative features directly from raw audio, achieving a favorable trade-off between accuracy, dimensionality reduction from the original audio size (data compression) , and training time efficiency. Compared with state-of-the-art software techniques, nanowire features reach 98.95% accuracy with 66 times data compression (XGBoost) and 97.9% accuracy with 255 times compression (Random Forest) in sub-second training latency. Across multiple classifiers nanowire features consistently achieve more than 90% accuracy with more than 62.5 times compression, outperforming features extracted by conventional state-of-the-art techniques such as MFCC in efficiency without loss of performance. Moreover, nanowire features achieve 96.5% accuracy classifying multispeaker audios, outperforming all state-of-the-art feature accuracies while achieving the highest data compression and lowest training time. Nanowire network preprocessing also enhances linear separability of audio data, improving simple classifier performance and generalizing across speakers. These results demonstrate that memristive nanowire networks provide a novel, low-latency, and data-efficient feature extraction approach, enabling high-performance neuromorphic audio classification.
>
---
#### [replaced 015] EmoSteer-TTS: Fine-Grained and Training-Free Emotion-Controllable Text-to-Speech via Activation Steering
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2508.03543v3](http://arxiv.org/pdf/2508.03543v3)**

> **作者:** Tianxin Xie; Shan Yang; Chenxing Li; Dong Yu; Li Liu
>
> **备注:** 25 pages, 9 figures, 3 tables
>
> **摘要:** Text-to-speech (TTS) has shown great progress in recent years. However, most existing TTS systems offer only coarse and rigid emotion control, typically via discrete emotion labels or a carefully crafted and detailed emotional text prompt, making fine-grained emotion manipulation either inaccessible or unstable. These models also require extensive, high-quality datasets for training. To address these limitations, we propose EmoSteer-TTS, a novel training-free approach, to achieve fine-grained speech emotion control (conversion, interpolation, erasure) by activation steering. We first empirically observe that modifying a subset of the internal activations within a flow matching-based TTS model can effectively alter the emotional tone of synthesized speech. Building on this insight, we then develop a training-free and efficient algorithm, including activation extraction, emotional token searching, and inference-time steering, which can be seamlessly integrated into a wide range of pretrained models (e.g., F5-TTS, CosyVoice2, and E2-TTS). In addition, to derive effective steering vectors, we construct a curated emotional speech dataset with diverse speakers. Extensive experiments demonstrate that EmoSteer-TTS enables fine-grained, interpretable, and continuous control over speech emotion, outperforming the state-of-the-art (SOTA). To the best of our knowledge, this is the first method that achieves training-free and continuous fine-grained emotion control in TTS. Demo samples are available at https://emosteer-tts-demo.pages.dev/.
>
---
