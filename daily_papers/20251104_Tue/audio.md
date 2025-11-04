# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Feedback-driven Retrieval-augmented Audio Generation with Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文面向文本到音频生成任务，解决特定声音事件合成缺失问题。提出一种反馈驱动的检索增强方法，利用大音频语言模型分析生成结果，检索外部知识并引导重生成，无需训练新模型即可提升生成质量。**

- **链接: [http://arxiv.org/pdf/2511.01091v1](http://arxiv.org/pdf/2511.01091v1)**

> **作者:** Junqi Zhao; Chenxing Li; Jinzheng Zhao; Rilin Chen; Dong Yu; Mark D. Plumbley; Wenwu Wang
>
> **摘要:** We propose a general feedback-driven retrieval-augmented generation (RAG) approach that leverages Large Audio Language Models (LALMs) to address the missing or imperfect synthesis of specific sound events in text-to-audio (TTA) generation. Unlike previous RAG-based TTA methods that typically train specialized models from scratch, we utilize LALMs to analyze audio generation outputs, retrieve concepts that pre-trained models struggle to generate from an external database, and incorporate the retrieved information into the generation process. Experimental results show that our method not only enhances the ability of LALMs to identify missing sound events but also delivers improvements across different models, outperforming existing RAG-specialized approaches.
>
---
#### [new 002] NaturalVoices: A Large-Scale, Spontaneous and Emotional Podcast Dataset for Voice Conversion
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出NaturalVoices，首个大规模自发播客语音数据集，用于情感感知的语音转换任务，解决现有数据缺乏真实情感与表达性的问题，提供5049小时带多维度标注的语音资源及开源工具链。**

- **链接: [http://arxiv.org/pdf/2511.00256v1](http://arxiv.org/pdf/2511.00256v1)**

> **作者:** Zongyang Du; Shreeram Suresh Chandra; Ismail Rasim Ulgen; Aurosweta Mahapatra; Ali N. Salman; Carlos Busso; Berrak Sisman
>
> **备注:** Under review for IEEE Transactions on Affective Computing
>
> **摘要:** Everyday speech conveys far more than words, it reflects who we are, how we feel, and the circumstances surrounding our interactions. Yet, most existing speech datasets are acted, limited in scale, and fail to capture the expressive richness of real-life communication. With the rise of large neural networks, several large-scale speech corpora have emerged and been widely adopted across various speech processing tasks. However, the field of voice conversion (VC) still lacks large-scale, expressive, and real-life speech resources suitable for modeling natural prosody and emotion. To fill this gap, we release NaturalVoices (NV), the first large-scale spontaneous podcast dataset specifically designed for emotion-aware voice conversion. It comprises 5,049 hours of spontaneous podcast recordings with automatic annotations for emotion (categorical and attribute-based), speech quality, transcripts, speaker identity, and sound events. The dataset captures expressive emotional variation across thousands of speakers, diverse topics, and natural speaking styles. We also provide an open-source pipeline with modular annotation tools and flexible filtering, enabling researchers to construct customized subsets for a wide range of VC tasks. Experiments demonstrate that NaturalVoices supports the development of robust and generalizable VC models capable of producing natural, expressive speech, while revealing limitations of current architectures when applied to large-scale spontaneous data. These results suggest that NaturalVoices is both a valuable resource and a challenging benchmark for advancing the field of voice conversion. Dataset is available at: https://huggingface.co/JHU-SmileLab
>
---
#### [new 003] The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity
- **分类: cs.SD; cs.AI; cs.HC**

- **简介: 该论文提出Aria-Duet系统，解决AI音乐生成与真人演奏脱节的问题，利用Disklavier钢琴实现人机实时二重奏，通过低延迟交互与风格延续，验证了具身式人机音乐协作的可行性。**

- **链接: [http://arxiv.org/pdf/2511.01663v1](http://arxiv.org/pdf/2511.01663v1)**

> **作者:** Louis Bradshaw; Alexander Spangher; Stella Biderman; Simon Colton
>
> **摘要:** While generative models for music composition are increasingly capable, their adoption by musicians is hindered by text-prompting, an asynchronous workflow disconnected from the embodied, responsive nature of instrumental performance. To address this, we introduce Aria-Duet, an interactive system facilitating a real-time musical duet between a human pianist and Aria, a state-of-the-art generative model, using a Yamaha Disklavier as a shared physical interface. The framework enables a turn-taking collaboration: the user performs, signals a handover, and the model generates a coherent continuation performed acoustically on the piano. Beyond describing the technical architecture enabling this low-latency interaction, we analyze the system's output from a musicological perspective, finding the model can maintain stylistic semantics and develop coherent phrasal ideas, demonstrating that such embodied systems can engage in musically sophisticated dialogue and open a promising new path for human-AI co-creation.
>
---
#### [new 004] ADNAC: Audio Denoiser using Neural Audio Codec
- **分类: cs.SD; cs.LG**

- **简介: 该论文提出ADNAC，利用神经音频编码器DAC实现音乐去噪，解决传统模型（如U-Net）在高保真恢复中的局限。通过自建合成数据集与多目标损失函数，实现生成式音频修复的可行性验证。**

- **链接: [http://arxiv.org/pdf/2511.01773v1](http://arxiv.org/pdf/2511.01773v1)**

> **作者:** Daniel Jimon; Mircea Vaida; Adriana Stan
>
> **备注:** Accepted and presented at the 13th International Conference on Speech Technology and Human-Computer Dialogue (SpeD), Cluj-Napoca, Romania, October 19-22, 2025. 4 pages, 1 figure. IEEE Catalog Number: CFP2555H-USB, ISBN: 979-8-3315-7485-7
>
> **摘要:** Audio denoising is critical in signal processing, enhancing intelligibility and fidelity for applications like restoring musical recordings. This paper presents a proof-of-concept for adapting a state-of-the-art neural audio codec, the Descript Audio Codec (DAC), for music denoising. This work overcomes the limitations of traditional architectures like U-Nets by training the model on a large-scale, custom-synthesized dataset built from diverse sources. Training is guided by a multi objective loss function that combines time-domain, spectral, and signal-level fidelity metrics. Ultimately, this paper aims to present a PoC for high-fidelity, generative audio restoration.
>
---
#### [new 005] WhisperVC: Target Speaker-Controllable Mandarin Whisper-to-Speech Conversion
- **分类: eess.AS**

- **简介: WhisperVC提出一种三阶段框架，实现中文耳语到语音的可控转换，解决耳语缺乏声带激励导致重建困难的问题，通过内容编码、时长对齐与声码器优化，达成高保真、高相似度的语音合成。**

- **链接: [http://arxiv.org/pdf/2511.01056v1](http://arxiv.org/pdf/2511.01056v1)**

> **作者:** Dong Liu; Ming Li
>
> **摘要:** Whispered speech lacks vocal-fold excitation and exhibits reduced energy and shifted formant frequencies, making natural and intelligible voice reconstruction highly challenging. To address this issue, we propose \emph{WhisperVC}, a three-stage framework for Mandarin whisper-to-speech (W2S) conversion. Stage~1 employs a fine-tuned Content Encoder based on the OpenAI Whisper-large~V3 model and a Conformer-based variational autoencoder with soft-DTW alignment to learn domain-invariant and temporally consistent representations. Stage~2 introduces a deterministic Length--Channel Aligner and a duration-free FastSpeech~2 model conditioned on speaker embeddings for controllable timbre and stable prosody. Stage~3 fine-tunes a HiFi-GAN vocoder on predicted mel-spectrograms to synthesize high-fidelity waveforms. Experiments on the AISHELL6-Whisper corpus demonstrate that WhisperVC achieves near ground-truth quality (\textbf{DNSMOS~3.11}, \textbf{UTMOS~2.52}, \textbf{CER~18.67\%}), while maintaining speaker similarity (\textbf{cosine~0.76}) and robust performance under whisper-only inference.
>
---
#### [new 006] Physics-Informed Neural Networks for Speech Production
- **分类: cs.SD**

- **简介: 该论文提出基于物理信息神经网络（PINN）的语音产生分析方法，解决声带振动非光滑性与周期未知性难题，通过可微近似与参数化周期建模，实现声门流、声带状态与亚声门压的联合估计，兼具正逆问题统一求解能力。**

- **链接: [http://arxiv.org/pdf/2511.00428v1](http://arxiv.org/pdf/2511.00428v1)**

> **作者:** Kazuya Yokota; Ryosuke Harakawa; Masaaki Baba; Masahiro Iwahashi
>
> **备注:** 11 pages, 10 figures
>
> **摘要:** The analysis of speech production based on physical models of the vocal folds and vocal tract is essential for studies on vocal-fold behavior and linguistic research. This paper proposes a speech production analysis method using physics-informed neural networks (PINNs). The networks are trained directly on the governing equations of vocal-fold vibration and vocal-tract acoustics. Vocal-fold collisions introduce nondifferentiability and vanishing gradients, challenging phenomena for PINNs. We demonstrate, however, that introducing a differentiable approximation function enables the analysis of vocal-fold vibrations within the PINN framework. The period of self-excited vocal-fold vibration is generally unknown. We show that by treating the period as a learnable network parameter, a periodic solution can be obtained. Furthermore, by implementing the coupling between glottal flow and vocal-tract acoustics as a hard constraint, glottis-tract interaction is achieved without additional loss terms. We confirmed the method's validity through forward and inverse analyses, demonstrating that the glottal flow rate, vocal-fold vibratory state, and subglottal pressure can be simultaneously estimated from speech signals. Notably, the same network architecture can be applied to both forward and inverse analyses, highlighting the versatility of this approach. The proposed method inherits the advantages of PINNs, including mesh-free computation and the natural incorporation of nonlinearities, and thus holds promise for a wide range of applications.
>
---
#### [new 007] MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出Multi-Bench，首个评估口语对话模型多轮交互中情感智能的基准，解决现有评测仅限单轮、忽视情感应用的问题，构建了包含5项任务、3.2K样本的分层评估框架。**

- **链接: [http://arxiv.org/pdf/2511.00850v1](http://arxiv.org/pdf/2511.00850v1)**

> **作者:** Yayue Deng; Guoqiang Hu; Haiyang Sun; Xiangyu Zhang; Haoyang Zhang; Fei Tian; Xuerui Yang; Gang Yu; Eng Siong Chng
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Spoken Dialogue Models (SDMs) have advanced rapidly, yet their ability to sustain genuinely interactive multi-turn conversations remains underexplored, as most benchmarks focus on single-turn exchanges. We introduce Multi-Bench, the first benchmark explicitly designed to evaluate SDMs in multi-turn interactive dialogue with an emphasis on emotional intelligence. Multi-Bench employs a hierarchical structure with a basic track for emotion understanding and reasoning and an advanced track for emotion support and application. It comprises five carefully designed tasks and about 3.2K samples, ranging from emotion recognition to complex reasoning and interactive dialogue, supported by a reproducible evaluation framework. We evaluate six representative SDMs on eight subsets of Multi-Bench. Results show that while current SDMs achieve good performance on basic understanding tasks, they still have room for improvement in advanced multi-turn interactive dialogue and reasoning-related tasks, particularly in emotion awareness and application.
>
---
#### [new 008] Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: Speech-DRAME提出首个语音角色扮演评估框架，解决现有ALLM评估缺失语用线索与合成语音失真问题，构建了带人工标注的评测基准、高性能评估模型与双维度评价体系，显著提升与人类评分的一致性。**

- **链接: [http://arxiv.org/pdf/2511.01261v1](http://arxiv.org/pdf/2511.01261v1)**

> **作者:** Jiatong Shi; Jionghao Han; Yichen Lu; Santiago Pascual; Pengfei Wu; Chenye Cui; Shinji Watanabe; Chao Weng; Cong Zhou
>
> **备注:** 67 pages
>
> **摘要:** Role-play has become a key testbed for generative models, expanding from text-only dialogue to multimodal interaction. Extending role-play to speech captures prosody, emotion, and delivery, but also poses new evaluation challenges. Current pipelines often use audio large language models (ALLMs) as zero-shot judges, which miss paralinguistic cues, collapse multiple aspects into coarse scores, and rely on synthetic speech references that fail to reflect real-world roles. We present Speech-DRAME, a unified framework that contributes at three levels: (i) Speech-DRAME-EvalBench, an evaluation benchmark with bilingual human-annotated data and protocols for training and testing speech evaluation models (SEMs), (ii) DRAME-Eval, a fine-tuned evaluation model, which substantially outperforms zero-shot and few-shot ALLMs, and (iii) Speech-DRAME-RoleBench, a speech role-play benchmark that leverages DRAME-Eval as an automatic judge to compare speech foundation models (SFMs). Speech-DRAME distinguishes between two complementary evaluation strategies: Archetype Evaluation, a top-down approach measuring adherence to broad role archetypes, and Realism Evaluation, a bottom-up approach grounded in real human speech that emphasizes nuanced role quality. Compared to zero-shot ALLM judges, DRAME-Eval achieves stronger agreement with human ratings (Pearson correlation from 0.480 to 0.629 in archetypes, and 0.390 to 0.625 in realism). By integrating transparent benchmark resources, modeling approaches, and system-level evaluation, Speech-DRAME provides the first comprehensive, reproducible foundation for assessing spoken role-play.
>
---
#### [new 009] Emotion Detection in Speech Using Lightweight and Transformer-Based Models: A Comparative and Ablation Study
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究语音情感识别任务，旨在提升轻量化模型在边缘设备上的实时性能。对比DistilHuBERT、PaSST与CNN-LSTM，发现DistilHuBERT以极小模型尺寸（0.02MB）实现最高准确率（70.64%），并分析PaSST不同分类头结构的影响。**

- **链接: [http://arxiv.org/pdf/2511.00402v1](http://arxiv.org/pdf/2511.00402v1)**

> **作者:** Lucky Onyekwelu-Udoka; Md Shafiqul Islam; Md Shahedul Hasan
>
> **摘要:** Emotion recognition from speech plays a vital role in the development of empathetic human-computer interaction systems. This paper presents a comparative analysis of lightweight transformer-based models, DistilHuBERT and PaSST, by classifying six core emotions from the CREMA-D dataset. We benchmark their performance against a traditional CNN-LSTM baseline model using MFCC features. DistilHuBERT demonstrates superior accuracy (70.64%) and F1 score (70.36%) while maintaining an exceptionally small model size (0.02 MB), outperforming both PaSST and the baseline. Furthermore, we conducted an ablation study on three variants of the PaSST, Linear, MLP, and Attentive Pooling heads, to understand the effect of classification head architecture on model performance. Our results indicate that PaSST with an MLP head yields the best performance among its variants but still falls short of DistilHuBERT. Among the emotion classes, angry is consistently the most accurately detected, while disgust remains the most challenging. These findings suggest that lightweight transformers like DistilHuBERT offer a compelling solution for real-time speech emotion recognition on edge devices. The code is available at: https://github.com/luckymaduabuchi/Emotion-detection-.
>
---
#### [new 010] Towards General Auditory Intelligence: Large Multimodal Models for Machine Listening and Speaking
- **分类: eess.AS**

- **简介: 该论文是一篇综述，旨在推动音频与大语言模型融合，构建具备人类级听觉智能的AGI系统。它系统梳理了音频理解、生成、语音交互及音视频融合四大方向的研究进展与挑战。**

- **链接: [http://arxiv.org/pdf/2511.01299v1](http://arxiv.org/pdf/2511.01299v1)**

> **作者:** Siyin Wang; Zengrui Jin; Changli Tang; Qiujia Li; Bo Li; Chen Chen; Yuchen Hu; Wenyi Yu; Yixuan Li; Jimin Zhuang; Yudong Yang; Mingqiu Wang; Michael Han; Yifan Ding; Junwen Bai; Tom Ouyang; Shuo-yiin Chang; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Guangzhi Sun; Zhehuai Chen; Ji Wu; Bowen Zhou; Yuxuan Wang; Tara Sainath; Yonghui Wu; Chao Zhang
>
> **备注:** 22 pages, 11 figures
>
> **摘要:** In the era of large language models (LLMs) and artificial general intelligence (AGI), computer audition must evolve beyond traditional paradigms to fully leverage the capabilities of foundation models, towards more comprehensive understanding, more natural generation and more human-like interaction. Audio, as a modality rich in semantic, emotional, and contextual cues, plays a vital role in achieving naturalistic and embodied machine intelligence. This survey provides a comprehensive review of recent progress in integrating audio into LLMs, with a focus on four key areas: audio comprehension, audio generation, speech-based interaction, and audio-visual understanding. We analyze how LLMs are reshaping audio perception and reasoning, enabling systems to understand sound at a deeper semantic level, generate expressive audio outputs, and engage in human-like spoken interaction. Furthermore, we explore how the fusion of audio and visual modalities enhances situational awareness and cross-modal reasoning, pushing the boundaries of multimodal intelligence. This survey not only synthesizes existing research but also identifies critical challenges and future directions for building audio-native AGI systems capable of perceiving, understanding, and interacting through sound as naturally as humans do.
>
---
#### [new 011] More Than A Shortcut: A Hyperbolic Approach To Early-Exit Networks
- **分类: cs.SD; cs.AI**

- **简介: 该论文提出Hyperbolic Early-Exit（HypEE）网络，用于资源受限设备上的音频事件检测。通过在双曲空间中学习层次化表示并引入蕴含损失，增强早期退出的可靠性与效率，同时提供不确定性度量，显著提升性能与计算效率。**

- **链接: [http://arxiv.org/pdf/2511.00641v1](http://arxiv.org/pdf/2511.00641v1)**

> **作者:** Swapnil Bhosale; Cosmin Frateanu; Camilla Clark; Arnoldas Jasonas; Chris Mitchell; Xiatian Zhu; Vamsi Krishna Ithapu; Giacomo Ferroni; Cagdas Bilen; Sanjeel Parekh
>
> **摘要:** Deploying accurate event detection on resource-constrained devices is challenged by the trade-off between performance and computational cost. While Early-Exit (EE) networks offer a solution through adaptive computation, they often fail to enforce a coherent hierarchical structure, limiting the reliability of their early predictions. To address this, we propose Hyperbolic Early-Exit networks (HypEE), a novel framework that learns EE representations in the hyperbolic space. Our core contribution is a hierarchical training objective with a novel entailment loss, which enforces a partial-ordering constraint to ensure that deeper network layers geometrically refine the representations of shallower ones. Experiments on multiple audio event detection tasks and backbone architectures show that HypEE significantly outperforms standard Euclidean EE baselines, especially at the earliest, most computationally-critical exits. The learned geometry also provides a principled measure of uncertainty, enabling a novel triggering mechanism that makes the overall system both more efficient and more accurate than a conventional EE and standard backbone models without early-exits.
>
---
#### [new 012] AudioNet: Supervised Deep Hashing for Retrieval of Similar Audio Events
- **分类: eess.AS**

- **简介: AudioNet提出一种监督深度哈希方法，用于高效检索相似音频事件，通过新颖的加权对比损失与离散梯度传播优化二值哈希码，在多个标准数据集上实现领先性能，解决音频检索中效率与精度平衡问题。**

- **链接: [http://arxiv.org/pdf/2511.01372v1](http://arxiv.org/pdf/2511.01372v1)**

> **作者:** Sagar Dutta; Vipul Arora
>
> **摘要:** This work presents a supervised deep hashing method for retrieving similar audio events. The proposed method, named AudioNet, is a deep-learning-based system for efficient hashing and retrieval of similar audio events using an audio example as a query. AudioNet achieves high retrieval performance on multiple standard datasets by generating binary hash codes for similar audio events, setting new benchmarks in the field, and highlighting its efficacy and effectiveness compare to other hashing methods. Through comprehensive experiments on standard datasets, our research represents a pioneering effort in evaluating the retrieval performance of similar audio events. A novel loss function is proposed which incorporates weighted contrastive and weighted pairwise loss along with hashcode balancing to improve the efficiency of audio event retrieval. The method adopts discrete gradient propagation, which allows gradients to be propagated through discrete variables during backpropagation. This enables the network to optimize the discrete hash codes using standard gradient-based optimization algorithms, which are typically used for continuous variables. The proposed method showcases promising retrieval performance, as evidenced by the experimental results, even when dealing with imbalanced datasets. The systematic analysis conducted in this study further supports the significant benefits of the proposed method in retrieval performance across multiple datasets. The findings presented in this work establish a baseline for future studies on the efficient retrieval of similar audio events using deep audio embeddings.
>
---
#### [new 013] Leveraging Language Information for Target Language Extraction
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究目标语言提取任务，旨在从多语言混音中分离指定语言语音。提出一种利用语音预训练模型语言知识的端到端框架，并构建首个公开多语言数据集，显著提升英语和德语提取性能。**

- **链接: [http://arxiv.org/pdf/2511.01652v1](http://arxiv.org/pdf/2511.01652v1)**

> **作者:** Mehmet Sinan Yıldırım; Ruijie Tao; Wupeng Wang; Junyi Ao; Haizhou Li
>
> **备注:** Accepted to APSIPA ASC 2025
>
> **摘要:** Target Language Extraction aims to extract speech in a specific language from a mixture waveform that contains multiple speakers speaking different languages. The human auditory system is adept at performing this task with the knowledge of the particular language. However, the performance of the conventional extraction systems is limited by the lack of this prior knowledge. Speech pre-trained models, which capture rich linguistic and phonetic representations from large-scale in-the-wild corpora, can provide this missing language knowledge to these systems. In this work, we propose a novel end-to-end framework to leverage language knowledge from speech pre-trained models. This knowledge is used to guide the extraction model to better capture the target language characteristics, thereby improving extraction quality. To demonstrate the effectiveness of our proposed approach, we construct the first publicly available multilingual dataset for Target Language Extraction. Experimental results show that our method achieves improvements of 1.22 dB and 1.12 dB in SI-SNR for English and German extraction, respectively, from mixtures containing both languages.
>
---
#### [new 014] Rhythm in the Air: Vision-based Real-Time Music Generation through Gestures
- **分类: cs.MM; cs.SD**

- **简介: 该论文提出基于视觉的实时手势音乐生成系统，解决手势识别精度与实时性问题。构建超1.5万样本手势数据集，设计MLA-GRU模型，提升音乐相关手势识别准确率至96.83%，实现高效人机交互式音乐创作。**

- **链接: [http://arxiv.org/pdf/2511.00793v1](http://arxiv.org/pdf/2511.00793v1)**

> **作者:** Barathi Subramanian; Rathinaraja Jeyaraj; Anand Paul; Kapilya Gangadharan
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Gesture recognition is an essential component of human-computer interaction (HCI), facilitating seamless interconnectivity between users and computer systems without physical touch. This paper introduces an innovative application of vision-based dynamic gesture recognition (VDGR) for real-time music composition through gestures. To implement this application, we generate a custom gesture dataset that encompasses over 15000 samples across 21 classes, incorporating 7 musical notes each manifesting at three distinct pitch levels. To effectively deal with the modest volume of training data and to accurately discern and prioritize complex gesture sequences for music creation, we develop a multi-layer attention-based gated recurrent unit (MLA-GRU) model, in which gated recurrent unit (GRU) is used to learn temporal patterns from the observed sequence and an attention layer is employed to focus on musically pertinent gesture segments. Our empirical studies demonstrate that MLA-GRU significantly surpasses the classical GRU model, achieving a remarkable accuracy of 96.83% compared to the baseline's 86.7%. Moreover, our approach exhibits superior efficiency and processing speed, which are crucial for interactive applications. Using our proposed system, we believe that people will interact with music in a new and exciting way. It not only advances HCI experiences but also highlights MLA-GRU's effectiveness in scenarios demanding swift and precise gesture recognition.
>
---
#### [new 015] LongCat-Flash-Omni Technical Report
- **分类: cs.MM; cs.AI; cs.CL; cs.DC; cs.LG; cs.SD**

- **简介: 论文提出560B参数的开源多模态模型LongCat-Flash-Omni，解决大规模多模态实时交互难题，通过渐进式训练与模态解耦并行架构，实现高效音视频理解与生成，兼顾单模态与多模态性能。**

- **链接: [http://arxiv.org/pdf/2511.00279v1](http://arxiv.org/pdf/2511.00279v1)**

> **作者:** Meituan LongCat Team; Bairui Wang; Bayan; Bin Xiao; Bo Zhang; Bolin Rong; Borun Chen; Chang Wan; Chao Zhang; Chen Huang; Chen Chen; Chen Chen; Chengxu Yang; Chengzuo Yang; Cong Han; Dandan Peng; Delian Ruan; Detai Xin; Disong Wang; Dongchao Yang; Fanfan Liu; Fengjiao Chen; Fengyu Yang; Gan Dong; Gang Huang; Gang Xu; Guanglu Wan; Guoqiang Tan; Guoqiao Yu; Haibo Qiu; Hao Lu; Hongbo Liu; Hongyu Xiang; Jiaheng Wu; Jian Yang; Jiaxing Liu; Jing Huang; Jingang Wang; Jinrui Ding; Juchao Jiang; Jun Kuang; Jun Wang; Junhui Mei; Ke Ding; Kefeng Zhang; Lei Chen; Liang Shi; Limeng Qiao; Liming Zheng; Lin Ma; Liuyang Guo; Liya Ma; Luying Sun; Man Gao; Mengshen Zhu; Miao Cao; Minliang Lin; Nuo Xu; Peng Shi; Qi Zhang; Qian Fang; Qian Wang; Qian Yang; Quanxiu Wang; Rongxiang Weng; Rongxin Guo; Ruoxuan Liang; Senbin Yang; Shanbo Xu; Shanglin Lei; Shengze Ye; Shimin Chen; Shuaiqi Chen; Shujie Hu; Shuo Li; Siqi Yang; Siyu Xu; Siyu Ren; Song Li; Songxiang Liu; Tianhao Bai; Tianye Dai; Wei Hong; Wei Wang; Weixiao Zhao; Wengang Cao; Wenlong Zhu; Wenlong He; Xi Su; Xi Nan; Xiaohan Zhao; Xiaohao Wang; Xiaoyu Zhao; Xiaoyu Wang; Xiaoyu Li; Xin Pan; Xin Chen; Xiusong Sun; Xu Xiang; Xudong Xing; Xuezhi Cao; Xunliang Cai; Yang Yang; Yanli Tan; Yao Yao; Yerui Sun; Yi Chen; Yifan Lu; Yin Gong; Yining Zhang; Yitian Chen; Yiyang Gan; Yuchen Tang; Yuchen Xie; Yueqian Wang; Yuewen Zheng; Yufei Zhang; Yufeng Zhong; Yulei Qian; Yuqi Peng; Yuwei Jiang; Zeyang Hu; Zheng Zhang; Zhengkun Tian; Zhiqing Hong; Zhixiong Zeng; Zhuqi Mi; Ziran Li; Ziwen Wang; Ziyi Zhao; Ziyuan Zhuang; Zizhe Zhao
>
> **摘要:** We introduce LongCat-Flash-Omni, a state-of-the-art open-source omni-modal model with 560 billion parameters, excelling at real-time audio-visual interaction. By adopting a curriculum-inspired progressive training strategy that transitions from simpler to increasingly complex modality sequence modeling tasks, LongCat-Flash-Omni attains comprehensive multimodal capabilities while maintaining strong unimodal capability. Building upon LongCat-Flash, which adopts a high-performance Shortcut-connected Mixture-of-Experts (MoE) architecture with zero-computation experts, LongCat-Flash-Omni integrates efficient multimodal perception and speech reconstruction modules. Despite its immense size of 560B parameters (with 27B activated), LongCat-Flash-Omni achieves low-latency real-time audio-visual interaction. For training infrastructure, we developed a modality-decoupled parallelism scheme specifically designed to manage the data and model heterogeneity inherent in large-scale multimodal training. This innovative approach demonstrates exceptional efficiency by sustaining over 90% of the throughput achieved by text-only training. Extensive evaluations show that LongCat-Flash-Omni achieves state-of-the-art performance on omni-modal benchmarks among open-source models. Furthermore, it delivers highly competitive results across a wide range of modality-specific tasks, including text, image, and video understanding, as well as audio understanding and generation. We provide a comprehensive overview of the model architecture design, training procedures, and data strategies, and open-source the model to foster future research and development in the community.
>
---
#### [new 016] Ultralow-power standoff acoustic leak detection
- **分类: cs.CR; eess.AS**

- **简介: 该论文提出一种超低功耗远程声学泄漏检测系统，通过检测高压管道泄漏产生的高频声波，实现无云传输、边缘处理的非接触式检测，功耗低至20–200μW，可穿透墙体并防止窃听，适用于水/气泄漏的无线监测。**

- **链接: [http://arxiv.org/pdf/2511.00348v1](http://arxiv.org/pdf/2511.00348v1)**

> **作者:** Michael P. Hasselbeck
>
> **备注:** 5 pages, 4 figures
>
> **摘要:** An automated, standoff acoustic leak detection scheme has been designed, built, and tested. It merges the principles of glass breakage and smoke detection to alert for the presence of leaks emanating from pressurized plumbing. A simulated water leak flowing at 0.15 l/min has been reliably detected at a standoff distance of more than 10 m. The device is also effective at identifying the presence of leaks located behind surfaces such as walls, doors, floors, and ceilings. The anticipated application is as an autonomous, battery-powered, remote wireless node. All signal processing and analysis takes place on the edge with no need to stream audio data to the cloud. Sensor status is conveyed on-demand with only a few bytes of information, requiring minimal bandwidth. Power consumption is the range of 20--200 micro-Watts, depending on the amount of environmental noise and desired sensor latency. To attain optimum sensitivity and reliability, the hardware operates at acoustic frequencies well above the range of human conversations, making eavesdropping impossible. Development has been done with water escaping from pressurized plumbing, but the sensor concept can be used effectively to detect gas leaks.
>
---
## 更新

#### [replaced 001] Audio Driven Real-Time Facial Animation for Social Telepresence
- **分类: cs.GR; cs.CV; cs.LG; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.01176v2](http://arxiv.org/pdf/2510.01176v2)**

> **作者:** Jiye Lee; Chenghui Li; Linh Tran; Shih-En Wei; Jason Saragih; Alexander Richard; Hanbyul Joo; Shaojie Bai
>
> **备注:** SIGGRAPH Asia 2025. Project page: https://jiyewise.github.io/projects/AudioRTA
>
> **摘要:** We present an audio-driven real-time system for animating photorealistic 3D facial avatars with minimal latency, designed for social interactions in virtual reality for anyone. Central to our approach is an encoder model that transforms audio signals into latent facial expression sequences in real time, which are then decoded as photorealistic 3D facial avatars. Leveraging the generative capabilities of diffusion models, we capture the rich spectrum of facial expressions necessary for natural communication while achieving real-time performance (<15ms GPU time). Our novel architecture minimizes latency through two key innovations: an online transformer that eliminates dependency on future inputs and a distillation pipeline that accelerates iterative denoising into a single step. We further address critical design challenges in live scenarios for processing continuous audio signals frame-by-frame while maintaining consistent animation quality. The versatility of our framework extends to multimodal applications, including semantic modalities such as emotion conditions and multimodal sensors with head-mounted eye cameras on VR headsets. Experimental results demonstrate significant improvements in facial animation accuracy over existing offline state-of-the-art baselines, achieving 100 to 1000 times faster inference speed. We validate our approach through live VR demonstrations and across various scenarios such as multilingual speeches.
>
---
#### [replaced 002] Continuous Boostlet Transform and Associated Uncertainty Principles
- **分类: eess.SP; cs.SD; eess.AS; math.FA; 42C40. 42C15. 81R30. 42A38**

- **链接: [http://arxiv.org/pdf/2504.03679v2](http://arxiv.org/pdf/2504.03679v2)**

> **作者:** Owais Ahmad; Jasifa Fayaz
>
> **备注:** 28pages,6 figures
>
> **摘要:** The Continuous Boostlet Transform (CBT) is introduced as a powerful tool for analyzing spatiotemporal signals, particularly acoustic wavefields. Overcoming the limitations of classical wavelets, the CBT leverages the Poincar\'e group and isotropic dilations to capture sparse features of natural acoustic fields. This paper presents the mathematical framework of the CBT, including its definition, fundamental properties, and associated uncertainty principles, such as Heisenberg's, logarithmic, Pitt's, and Nazarov's inequalities. These results illuminate the trade-offs between time and frequency localization in the boostlet domain. Practical examples with constant and exponential functions highlight the CBT's adaptability. With applications in radar, communications, audio processing, and seismic analysis, the CBT offers flexible time-frequency resolution, making it ideal for non-stationary and transient signals, and a valuable tool for modern signal processing.
>
---
#### [replaced 003] MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2504.03546v2](http://arxiv.org/pdf/2504.03546v2)**

> **作者:** Khai Le-Duc; Tuyen Tran; Bach Phan Tat; Nguyen Kim Hai Bui; Quan Dang; Hung-Phong Tran; Thanh-Thuy Nguyen; Ly Nguyen; Tuan-Minh Phan; Thi Thu Phuong Tran; Chris Ngo; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** EMNLP 2025
>
> **摘要:** Multilingual speech translation (ST) and machine translation (MT) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, and Simplified/Traditional Chinese, together with the models. With 290,000 samples, this is the largest medical MT dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most comprehensive ST analysis in the field's history, to our best knowledge, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: https://github.com/leduckhai/MultiMed-ST
>
---
#### [replaced 004] AnyEnhance: A Unified Generative Model with Prompt-Guidance and Self-Critic for Voice Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2501.15417v3](http://arxiv.org/pdf/2501.15417v3)**

> **作者:** Junan Zhang; Jing Yang; Zihao Fang; Yuancheng Wang; Zehua Zhang; Zhuo Wang; Fan Fan; Zhizheng Wu
>
> **备注:** Accepted by IEEE TASLP 2025. Demopage: https://amphionspace.github.io/anyenhance. Open-source implementation: https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc
>
> **摘要:** We introduce AnyEnhance, a unified generative model for voice enhancement that processes both speech and singing voices. Based on a masked generative model, AnyEnhance is capable of handling both speech and singing voices, supporting a wide range of enhancement tasks including denoising, dereverberation, declipping, super-resolution, and target speaker extraction, all simultaneously and without fine-tuning. AnyEnhance introduces a prompt-guidance mechanism for in-context learning, which allows the model to natively accept a reference speaker's timbre. In this way, it could boost enhancement performance when a reference audio is available and enable the target speaker extraction task without altering the underlying architecture. Moreover, we also introduce a self-critic mechanism into the generative process for masked generative models, yielding higher-quality outputs through iterative self-assessment and refinement. Extensive experiments on various enhancement tasks demonstrate AnyEnhance outperforms existing methods in terms of both objective metrics and subjective listening tests. Demo audios are publicly available at https://amphionspace.github.io/anyenhance. An open-source implementation is provided at https://github.com/viewfinder-annn/anyenhance-v1-ccf-aatc.
>
---
#### [replaced 005] Temporal Feature Learning in Weakly Labelled Bioacoustic Cetacean Datasets via a Variational Autoencoder and Temporal Convolutional Network: An Interdisciplinary Approach
- **分类: cs.SD; eess.AS; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2410.17006v3](http://arxiv.org/pdf/2410.17006v3)**

> **作者:** Laia Garrobé Fonollosa; Douglas Gillespie; Lina Stankovic; Vladimir Stankovic; Luke Rendell
>
> **摘要:** Bioacoustics data from Passive acoustic monitoring (PAM) poses a unique set of challenges for classification, particularly the limited availability of complete and reliable labels in datasets due to annotation uncertainty, biological complexity due the heterogeneity in duration of cetacean vocalizations, and masking of target sounds due to environmental and anthropogenic noise. This means that data is often weakly labelled, with annotations indicating presence/absence of species over several minutes. In order to effectively capture the complex temporal patterns and key features of lengthy continuous audio segments, we propose an interdisciplinary framework comprising dataset standardisation, feature extraction via Variational Autoencoders (VAE) and classification via Temporal Convolutional Networks (TCN). This approach eliminates the necessity for manual threshold setting or time-consuming strong labelling. To demonstrate the effectiveness of our approach, we use sperm whale (<i>Physeter macrocephalus</i>) click trains in 4-minute recordings as a case study, from a dataset comprising diverse sources and deployment conditions to maximise generalisability. The value of feature extraction via the VAE is demonstrated by comparing classification performance against the traditional and explainable approach of expert handpicking of features. The TCN demonstrated robust classification capabilities achieving AUC scores exceeding 0.9.
>
---
#### [replaced 006] Music Arena: Live Evaluation for Text-to-Music
- **分类: cs.SD; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2507.20900v2](http://arxiv.org/pdf/2507.20900v2)**

> **作者:** Yonghyun Kim; Wayne Chi; Anastasios N. Angelopoulos; Wei-Lin Chiang; Koichi Saito; Shinji Watanabe; Yuki Mitsufuji; Chris Donahue
>
> **备注:** NeurIPS 2025 Creative AI Track
>
> **摘要:** We present Music Arena, an open platform for scalable human preference evaluation of text-to-music (TTM) models. Soliciting human preferences via listening studies is the gold standard for evaluation in TTM, but these studies are expensive to conduct and difficult to compare, as study protocols may differ across systems. Moreover, human preferences might help researchers align their TTM systems or improve automatic evaluation metrics, but an open and renewable source of preferences does not currently exist. We aim to fill these gaps by offering *live* evaluation for TTM. In Music Arena, real-world users input text prompts of their choosing and compare outputs from two TTM systems, and their preferences are used to compile a leaderboard. While Music Arena follows recent evaluation trends in other AI domains, we also design it with key features tailored to music: an LLM-based routing system to navigate the heterogeneous type signatures of TTM systems, and the collection of *detailed* preferences including listening data and natural language feedback. We also propose a rolling data release policy with user privacy guarantees, providing a renewable source of preference data and increasing platform transparency. Through its standardized evaluation protocol, transparent data access policies, and music-specific features, Music Arena not only addresses key challenges in the TTM ecosystem but also demonstrates how live evaluation can be thoughtfully adapted to unique characteristics of specific AI domains. Music Arena is available at: https://music-arena.org . Preference data is available at: https://huggingface.co/music-arena .
>
---
#### [replaced 007] Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2507.18161v2](http://arxiv.org/pdf/2507.18161v2)**

> **作者:** Samuele Cornell; Christoph Boeddeker; Taejin Park; He Huang; Desh Raj; Matthew Wiesner; Yoshiki Masuyama; Xuankai Chang; Zhong-Qiu Wang; Stefano Squartini; Paola Garcia; Shinji Watanabe
>
> **摘要:** The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles.
>
---
#### [replaced 008] Aligning Speech to Languages to Enhance Code-switching Speech Recognition
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2403.05887v3](http://arxiv.org/pdf/2403.05887v3)**

> **作者:** Hexin Liu; Xiangyu Zhang; Haoyang Zhang; Leibny Paola Garcia; Andy W. H. Khong; Eng Siong Chng; Shinji Watanabe
>
> **备注:** Accepted to IEEE Trans. Audio Speech Lang. Process., copyright has been transferred to IEEE
>
> **摘要:** Code-switching (CS) refers to the switching of languages within a speech signal and results in language confusion for automatic speech recognition (ASR). To address language confusion, we propose a language alignment loss (LAL) that aligns acoustic features to pseudo-language labels learned from the ASR decoder during ASR training. This approach enables frame-level language identification without the need for frame-level language annotations. To further tackle the complex token alternatives for language modeling in bilingual scenarios, we propose to employ large language models via a generative error correction method. A linguistic hint, derived from LAL outputs and decoded hypotheses, is introduced to guide the prompting and enhance the LLM-based generative error correction for CS-ASR. The proposed methods are evaluated on the SEAME dataset and data from the ASRU 2019 Mandarin-English code-switching speech recognition challenge. The incorporation of the proposed language alignment loss improves CS-ASR performance for both hybrid CTC/attention and Whisper models on both datasets, with only a negligible increase in the number of parameters. This work also highlights the efficacy of language alignment loss in balancing primary-language-dominant bilingual data during training, with an 8.6% relative improvement on the ASRU dataset compared to the baseline model. Performance evaluation using large language models reveals the advantage of the linguistic hint by achieving 14.1% and 5.5% relative improvement on test sets of the ASRU and SEAME datasets, respectively.
>
---
#### [replaced 009] Instance-Specific Test-Time Training for Speech Editing in the Wild
- **分类: eess.AS; cs.SD**

- **链接: [http://arxiv.org/pdf/2506.13295v2](http://arxiv.org/pdf/2506.13295v2)**

> **作者:** Taewoo Kim; Uijong Lee; Hayoung Park; Choongsang Cho; Nam In Park; Young Han Lee
>
> **备注:** Accepted to NeurIPS 2025 Workshop on GenProCC
>
> **摘要:** Speech editing systems aim to naturally modify speech content while preserving acoustic consistency and speaker identity. However, previous studies often struggle to adapt to unseen and diverse acoustic conditions, resulting in degraded editing performance in real-world scenarios. To address this, we propose an instance-specific test-time training method for speech editing in the wild. Our approach employs direct supervision from ground-truth acoustic features in unedited regions and indirect supervision in edited regions via auxiliary losses based on duration constraints and phoneme prediction. This strategy mitigates the bandwidth discontinuity problem in speech editing, ensuring smooth acoustic transitions between unedited and edited regions. Additionally, it enables precise control over speech rate by adapting the model to target durations via mask length adjustment during test-time training. Experiments on in-the-wild benchmark datasets demonstrate that our method outperforms existing speech editing systems in both objective and subjective evaluations.
>
---
#### [replaced 010] Sound Clouds: Exploring ambient intelligence in public spaces to elicit deep human experience of awe, wonder, and beauty
- **分类: cs.HC; cs.MM; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.15865v2](http://arxiv.org/pdf/2510.15865v2)**

> **作者:** Chengzhi Zhang; Dashiel Carrera; Daksh Kapoor; Jasmine Kaur; Jisu Kim; Brian Magerko
>
> **备注:** 4 pages, Artwork accepted by NeurIPS Creative AI Track 2025
>
> **摘要:** While the ambient intelligence (AmI) systems we encounter in our daily lives, including security monitoring and energy-saving systems, typically serve pragmatic purposes, we wonder how we can design and implement ambient artificial intelligence experiences in public spaces that elicit deep human feelings of awe, wonder, and beauty. As a manifestation, we introduce Sound Clouds, an immersive art installation that generates live music based on participants' interaction with several human-height spheres. Our installation serves as a provocation into future ambient intelligence that provokes, not limits, the future possibilities of AmI.
>
---
#### [replaced 011] Prevailing Research Areas for Music AI in the Era of Foundation Models
- **分类: cs.SD; cs.AI; cs.MM; eess.AS; 68T05, 68T20; I.2; I.5.4; I.2.6; I.2.7; H.5.5**

- **链接: [http://arxiv.org/pdf/2409.09378v2](http://arxiv.org/pdf/2409.09378v2)**

> **作者:** Megan Wei; Mateusz Modrzejewski; Aswin Sivaraman; Dorien Herremans
>
> **摘要:** Parallel to rapid advancements in foundation model research, the past few years have witnessed a surge in music AI applications. As AI-generated and AI-augmented music become increasingly mainstream, many researchers in the music AI community may wonder: what research frontiers remain unexplored? This paper outlines several key areas within music AI research that present significant opportunities for further investigation. We begin by examining foundational representation models and highlight emerging efforts toward explainability and interpretability. We then discuss the evolution toward multimodal systems, provide an overview of the current landscape of music datasets and their limitations, and address the growing importance of model efficiency in both training and deployment. Next, we explore applied directions, focusing first on generative models. We review recent systems, their computational constraints, and persistent challenges related to evaluation and controllability. We then examine extensions of these generative approaches to multimodal settings and their integration into artists' workflows, including applications in music editing, captioning, production, transcription, source separation, performance, discovery, and education. Finally, we explore copyright implications of generative music and propose strategies to safeguard artist rights. While not exhaustive, this survey aims to illuminate promising research directions enabled by recent developments in music foundation models.
>
---
#### [replaced 012] As Good as It KAN Get: High-Fidelity Audio Representation
- **分类: cs.SD; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.02585v3](http://arxiv.org/pdf/2503.02585v3)**

> **作者:** Patryk Marszałek; Maciej Rut; Piotr Kawa; Przemysław Spurek; Piotr Syga
>
> **备注:** Accepted to the 34th ACM International Conference on Information and Knowledge Management (CIKM '25)
>
> **摘要:** Implicit neural representations (INR) have gained prominence for efficiently encoding multimedia data, yet their applications in audio signals remain limited. This study introduces the Kolmogorov-Arnold Network (KAN), a novel architecture using learnable activation functions, as an effective INR model for audio representation. KAN demonstrates superior perceptual performance over previous INRs, achieving the lowest Log-SpectralDistance of 1.29 and the highest Perceptual Evaluation of Speech Quality of 3.57 for 1.5 s audio. To extend KAN's utility, we propose FewSound, a hypernetwork-based architecture that enhances INR parameter updates. FewSound outperforms the state-of-the-art HyperSound, with a 33.3% improvement in MSE and 60.87% in SI-SNR. These results show KAN as a robust and adaptable audio representation with the potential for scalability and integration into various hypernetwork frameworks. The source code can be accessed at https://github.com/gmum/fewsound.git.
>
---
#### [replaced 013] DSpAST: Disentangled Representations for Spatial Audio Reasoning with Large Language Models
- **分类: eess.AS; cs.AI; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.13927v2](http://arxiv.org/pdf/2509.13927v2)**

> **作者:** Kevin Wilkinghoff; Zheng-Hua Tan
>
> **摘要:** Reasoning about spatial audio with large language models requires a spatial audio encoder as an acoustic front-end to obtain audio embeddings for further processing. Such an encoder needs to capture all information required to detect the type of sound events, as well as the direction and distance of their corresponding sources. Accomplishing this with a single audio encoder is demanding as the information required for each of these tasks is mostly independent of each other. As a result, the performance obtained with a single encoder is often worse than when using task-specific audio encoders. In this work, we present DSpAST, a novel audio encoder based on SpatialAST that learns disentangled representations of spatial audio while having only 0.2% additional parameters. Experiments on SpatialSoundQA with the spatial audio reasoning system BAT demonstrate that DSpAST significantly outperforms SpatialAST.
>
---
#### [replaced 014] Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs
- **分类: eess.AS; cs.CV; cs.SD**

- **链接: [http://arxiv.org/pdf/2510.22603v2](http://arxiv.org/pdf/2510.22603v2)**

> **作者:** Anand; Umberto Cappellazzo; Stavros Petridis; Maja Pantic
>
> **备注:** The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
>
> **摘要:** Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
>
---
