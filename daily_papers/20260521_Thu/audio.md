# 音频 cs.SD;  eess.AS

- **最新发布 19 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Causal Spatio-Temporal Sound Field Reconstruction
- **分类: eess.AS**

- **简介: 该论文属于声场重建任务，解决实时应用中因果测量下的声场重构问题。提出一种因果时空最小均方误差估计器，并优化采样策略以提高重建精度。**

- **链接: [https://arxiv.org/pdf/2605.20403](https://arxiv.org/pdf/2605.20403)**

> **作者:** David Sundström; Filip Tronarp; Johan Lindström; Andreas Jakobsson
>
> **摘要:** In sound field control applications, it is commonly assumed that one has access to an accurate representation of the sound field in the region of interest. This is a problematic assumption since the reconstruction of a sound field from available microphone measurements is especially challenging in real-time applications where only causal measurements are available. Notably, causal time-windowed observations introduce correlation between frequency components, making sound field reconstruction methods that process each frequency band independently sub-optimal. In this work, we formulate a causal finite-window spatio-temporal linear minimum mean-square error estimator for sound field reconstruction. The sound field is modeled as the solution to the wave equation driven by a stationary stochastic spatio-temporal source distribution, which induces a physically interpretable covariance function. It is shown that this covariance function is closely related to the classical diffuse-field coherence model. Since the computational complexity grows rapidly with the number of spatio-temporal observations, we formulate a budget-constrained spatio-temporal sample selection approach to minimize the posterior reconstruction variance. The proposed estimator and sampling strategy are evaluated using both simulated and measured sound fields, demonstrating improved short-window reconstruction compared to frequency domain finite-window baselines.
>
---
#### [new 002] From Numbers to Perception, Energy Decay Curves Prediction
- **分类: eess.AS; eess.SP**

- **简介: 该论文属于音频渲染任务，旨在解决RIR预测问题。通过神经网络预测多频段EDC，优化能量与衰减斜率，实现高效准确的声学模拟。**

- **链接: [https://arxiv.org/pdf/2605.20968](https://arxiv.org/pdf/2605.20968)**

> **作者:** Imran Muhammad; Gerald Schuller
>
> **摘要:** Predicting Room Impulse Responses (RIRs) remains a challenge due to the high dimensionality of audio signals and the need for perceptual accuracy. This paper introduces a neural network framework that predicts multi-band Energy Decay Curves (EDCs) directly from room geometry and material properties. Unlike standard models, our framework employs a custom composite loss function that optimizes for both energy levels and decay slopes in the log-domain. This ensures the predicted curves adhere to physical decay principles while maintaining high sensitivity to reverberation time and early reflections. Results demonstrate that the model successfully approximates ground-truth acoustics with minimal error in T30 and clarity indices. The approach offers a computationally efficient alternative to traditional simulations, facilitating realistic audio rendering for interactive virtual environments.
>
---
#### [new 003] A Survey of Large Audio Language Models: Generalization, Trustworthiness, and Outlook
- **分类: cs.SD**

- **简介: 该论文属于音频语言模型研究，旨在解决其可信性问题。分析了LALMs的架构与安全风险，提出增强可信性的方法。**

- **链接: [https://arxiv.org/pdf/2605.20266](https://arxiv.org/pdf/2605.20266)**

> **作者:** Kaiwen Luo; Zhenhong Zhou; Leo Wang; Liang Lin; Yang Xiao; Tianyu Shao; Yuanhe Zhang; Yuxuan Li; Miao Yu; Kailin Lyu; Jiaming Zhang; Dongrui Liu; Li Sun; Yueming Wu; Kai Li; Ting Dang; Xiaojun Jia; Rohan Kumar Das; Xinfeng Li; Siyuan Liang; Qiufeng Wang; Xingjun Ma; Jing Chen; Kun Wang; Junhao Dong; Deqing Zou; Yu Cheng; Xia Hu; Zhigang Zeng; Sen Su; Yang Liu; Yu-Gang Jiang; Philip S. Yu; Yew-Soon Ong
>
> **摘要:** The foundational capabilities established by Large Language Models (LLMs) have paved the way for Multimodal Large Language Models (MLLMs), within which Large Audio Language Models (LALMs) are essential for realizing universal auditory intelligence. Despite their remarkable performance, the escalation of LALMs' capabilities has significantly outpaced the development of systemic frameworks to ensure their trustworthiness. This survey provides a comprehensive investigation into the endogenous mechanisms of LALMs, detailing the architectural innovations and alignment algorithms that facilitate emergent reasoning. Specifically, we analyze how the transition to unified end-to-end frameworks and the integration of continuous acoustic signals inherently expand the attack surface. To rigorously evaluate the risks within these paradigms, we establish a comprehensive taxonomy of trustworthiness, categorizing critical vulnerabilities such as cross-modal jailbreaking, latent acoustic backdoors, and biometric privacy leakage. We review the state-of-the-art through six analytical pillars: hallucination, robustness, safety, privacy, fairness, and authentication. The profound imbalance between a mature offensive landscape and underdeveloped defenses further validates the critical trustworthiness gaps and multidimensional risks facing audio-centric intelligence. Finally, we propose a strategic roadmap advocating for "Defense-in-Depth" architectures, causal auditory world modeling, and intrinsic representation engineering to bridge the gap between empirical performance and intrinsically trustworthy audio intelligence. Our project has been uploaded to GitHub this https URL.
>
---
#### [new 004] Codec-Robust Attacks on Audio LLMs
- **分类: cs.SD; cs.AI**

- **简介: 该论文研究音频大语言模型的对抗攻击问题，提出CodecAttack方法，在编码器潜空间生成鲁棒扰动，突破压缩防御，验证了编解码器感知攻击对实际部署系统的威胁。**

- **链接: [https://arxiv.org/pdf/2605.20519](https://arxiv.org/pdf/2605.20519)**

> **作者:** Jaechul Roh; Jean-Philippe Monteuuis; Jonathan Petit; Amir Houmansdar
>
> **摘要:** Prior attacks on Audio Large Language Models (Audio LLMs) demonstrated that carefully crafted waveform-domain perturbations can force targeted adversarial outputs. As a defense mechanism against these attacks, real-world codec compression preprocessing has been studied to both detect and remove the perturbations. Yet no existing attack has demonstrated robustness against these compressions. We introduce CodecAttack, which optimizes a perturbation in a neural audio codec's continuous latent space rather than directly perturbing the audio waveform. We show that the codec's compression channel, which discards waveform perturbations, transmits perturbations crafted in its own latent space. To further harden the attack across real-world compression channels, we apply multi-bitrate straight-through Expectation-over-Transformation (EoT), all without modifying the target model. Across three realistic Audio LLM deployment scenarios and three target models, CodecAttack achieves an average 85.5% target-substring attack success rate (ASR) on Opus at moderate bitrates, while the waveform baseline trained with identical EoT hardening does not exceed 26% at any bitrate. The attack transfers to held-out codecs, reaching up to 100% ASR on MP3 and 84% on AAC-LC without retraining. A per-band energy analysis shows that the latent perturbation concentrates below 4kHz, exactly where codecs allocate the most bits, while the waveform baseline spreads into higher frequencies that codecs discard. These results demonstrate that lossy compression is not a reliable defense against adversarial audio and that codec-aware attacks pose a practical threat to deployed Audio LLM systems.
>
---
#### [new 005] Speech Quality Embeddings for Improved Detection and Classification of Degradations in Speech Signals
- **分类: eess.AS**

- **简介: 该论文属于语音质量评估任务，旨在解决现代系统中局部降级检测与分类问题。通过构建帧级嵌入，利用对比损失区分降级类型，提升降级识别效果。**

- **链接: [https://arxiv.org/pdf/2605.21332](https://arxiv.org/pdf/2605.21332)**

> **作者:** Michael Kuhlmann; Tobias Cord-Landwehr; Reinhold Haeb-Umbach
>
> **备注:** Accepted to 2026 Odyssey workshop
>
> **摘要:** Automatic subjective speech quality assessment (SSQA) traditionally estimates speech quality on an utterance or system level. While this resolution was adequate for older transmission or synthesis systems that produced speech signals of mediocre quality, modern systems generate high-quality speech with degradations that may occur only locally. With suitable model architectures and regularization losses, SSQA models trained with utterance-level targets can also yield useful local predictions of speech quality. In this work, we extend such models to produce frame-level embeddings that cluster by degradation type. Specifically, we employ a partial mix-up strategy on a parallel corpus of clean and degraded utterances and apply a contrastive loss to distinguish between degradation types. Through experiments on both in- and out-of-domain data, we demonstrate that our approach improves degradation detection and enables the identification of degradation types by analyzing embedding clusters.
>
---
#### [new 006] Instrumental Text-to-Music Generation with Auxiliary Conditioning Branches
- **分类: cs.SD**

- **简介: 该论文属于文本到音乐生成任务，研究在无歌词和音色条件下的乐器音乐生成问题。通过消融实验验证辅助分支的重要性，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.21433](https://arxiv.org/pdf/2605.21433)**

> **作者:** Junyoung Koh
>
> **备注:** ICME 2026 Grand Challenge on Academic Text-to-Music Generation
>
> **摘要:** Text-to-music generation has advanced rapidly, with modern autoregressive and diffusion-based models producing convincing music from natural-language prompts. However, much of this progress relies on large-scale training data and external pretraining, making it difficult to isolate which design choices remain effective when data and pretraining are controlled. We study this setting using a Diffusion Transformer backbone with lyric and timbre conditioning, adapted to an instrumental-only text-to-music task in which the auxiliary lyric and timbre branches receive only degenerate conditioning signals. Through controlled ablations, we find that models retrained without these branches score lower across AudioBox aesthetics, LLM-as-judge, and human MOS, and that reinvesting the saved parameters as additional DiT depth recovers only marginally. This suggests the auxiliary branches may act as training-time architectural anchors whose contribution goes beyond their explicit conditioning content. We validate the same model through comparisons with external instrumental baselines and through our submission to the ICME 2026 Academic Text-to-Music (ATTM) Grand Challenge, where our Performance submission ranked first under both the objective metrics and the subsequent organizer-administered MOS over 35 raters, attaining the highest overall MOS across all challenge submissions, while our Efficiency submission was a finalist that tied for second under the objective metrics.
>
---
#### [new 007] Advanced Scientific Methodology Plays Rossini
- **分类: cs.SD; cs.IR; cs.LG**

- **简介: 论文属于音乐文献学领域，旨在分析罗西尼对同一歌词的多版本创作。通过计算方法研究其旋律、和声与文本选择，探索创作过程，为后续研究提供基础。**

- **链接: [https://arxiv.org/pdf/2605.20220](https://arxiv.org/pdf/2605.20220)**

> **作者:** Silvia Licciardi; Daniela Macchione; Emmanuel Caronna; Elisa Francomano
>
> **摘要:** A musical score provides the essential instructions for its performance while containing indications - at times implicit - regarding the composer's intentions. The presence of authorial variants, and even more so complex series of revisions associated with a single text, presents a challenging path for analytical study. This research, situated within the application of Scientific Methodologies to Music Philology, proposes a methodological approach oriented toward the structural analysis of one of the many settings composed by Gioachino Rossini on the same Metastasio arietta ``Mi lagnerò tacendo''. Through Computational Analysis - incorporating parsing, data mining, and graph theory - the melodic, harmonic, and textual compositional choices have been rigorously explored. The results constitute a significant unicum in the field, laying the foundation for a systematic study that supports philological research and paves the way for the use of generative models to investigate the creative process.
>
---
#### [new 008] Musical Attention Transformer: Music Generation Using a Music-Specific Attention Model
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于音乐生成任务，旨在解决生成音乐重复、不自然的问题。通过引入音乐元信息改进注意力机制，提升生成音乐的多样性和质量。**

- **链接: [https://arxiv.org/pdf/2605.21081](https://arxiv.org/pdf/2605.21081)**

> **作者:** Shinnosuke Taksuka; Hideo Mukai
>
> **备注:** 32 pages, 13 figures
>
> **摘要:** This study aims to enhance the quality of music generation using Transformers by incorporating meta-information. While Transformer-based approaches are effective at capturing long-term dependencies in musical compositions, the music they generate often suffers from issues such as excessive repetition or duplication of notes, leading to unnatural melodies. To address these limitations, we propose Musical Attention, a mechanism that incorporates meta-information such as bar numbers, key, signatures, and tempos into the attention process. Musical Attention explicitly leverages both the structural properties of music and its associated metadata, enabling the Transformer's attention mechanism to operate more effectively and thereby improving the quality of the generated output. In our framework, each musical note is represented as a combination of five events-pitch, bar number, onset, duration, and velocity in addition to the three metadata elements. The attention mechanism is then modified to reflect the correlations among these eight features, allowing the model to better capture the inherent characteristics of musical composition. Experimental results demonstrate that the model incorporating Musical Attention outperforms prior methods, such as Full Attention and Strided Attention, in terms of musical coherence, variation, and overall quality. Notably, it significantly reduces repetition and enhances the model's ability to generate diverse, harmonically consistent melodies. Musical Attention thus represents a meaningful advancement in AI-driven music generation, facilitating the creation of more natural and expressive compositions.
>
---
#### [new 009] Linearly Constrained Deep Beamformer for Multi-Speaker Scenarios
- **分类: eess.AS**

- **简介: 该论文属于语音增强任务，旨在解决多说话人环境下的目标语音增强问题。通过深度学习方法设计线性约束的波束成形器，提升目标语音并抑制干扰。**

- **链接: [https://arxiv.org/pdf/2605.21141](https://arxiv.org/pdf/2605.21141)**

> **作者:** Ilai Zaidel; Ori Engel; Bar Engel; Sharon Gannot
>
> **摘要:** We propose a deep beamforming framework for enhancing target speaker(s) in multi-speaker environments. A deep neural network (DNN) is trained to estimate beamforming weights directly from noisy multichannel inputs while satisfying linear spatial constraints through an adaptive multi-term loss inspired by the augmented Lagrangian framework. The loss combines signal reconstruction with penalties that enforce a distortionless response toward the target and suppress the interference subspace. The model is further guided by the target relative transfer function (RTF) and the estimated interference subspace. The proposed model can direct a beam toward the target speaker while directing nulls toward the interfering sources, achieving superior overall enhancement performance compared with the classical LCMV beamformer constructed by the same estimated spatial signatures. Furthermore, compared with the LCMV beamformer, the proposed model produces more controlled sidelobes and improved background-noise attenuation.
>
---
#### [new 010] A Survey of Audio Reasoning in Multimodal Foundation Models
- **分类: eess.AS**

- **简介: 该论文属于音频推理任务，旨在解决音频与语言模型对齐及实时交互问题。工作包括系统梳理音频推理方法、分析挑战并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2605.21008](https://arxiv.org/pdf/2605.21008)**

> **作者:** Zhihan Guo; Wenqian Cui; Guan-Ting Lin; Daxin Tan; Jingyao Li; Qiyong Zheng; Dingdong Wang; Jing Xiong; Han Shi; Jiaya Jia; Irwin King
>
> **摘要:** Reasoning has become a defining capability of modern foundation models, yet its development in the audio modality remains limited. Audio poses challenges that are distinct from those of text and vision. It is continuous, temporally dense, and contains linguistic, paralinguistic, and environmental information at multiple time scales. As a result, audio reasoning models must align acoustic signals with the discrete semantic space of large language models, while still preserving fine-grained information needed for reliable inference. Progress is also limited by three major obstacles: the scarcity of genuinely audio-grounded reasoning data, shortcut learning and modality hallucination, and the tension between reasoning depth and real-time latency in spoken interaction. In this paper, we present the first dedicated survey of audio reasoning. We provide a unified formulation that distinguishes direct predictive modeling from reasoning-augmented generation, review the architectural and training foundations of audio reasoning models, and systematically organize recent advances in Audio-to-Text, Audio-to-Speech, Audio-Visual Reasoning and Agentic Audio Reasoning. We further examine emerging paradigms such as Chain-of-Thought prompting, supervised fine-tuning, reinforcement learning, and latency-aware spoken interaction, and discuss evaluation practices, open challenges, and future directions. Our goal is to offer a coherent roadmap for developing robust, efficient, and natively grounded audio reasoning systems.
>
---
#### [new 011] Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech
- **分类: eess.AS**

- **简介: 该论文属于文本到语音合成任务，旨在解决开放数据对TTS模型性能影响的研究空白。作者提出了Raon-OpenTTS模型及配套数据集，提升TTS的可复现性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.20830](https://arxiv.org/pdf/2605.20830)**

> **作者:** Semin Kim; Seungjun Chung; Taehong Moon; Sangheon Lee; Minyoung Ahn; Keon Lee; Nam Soo Kim; Jaewoong Cho; Ludwig Schmidt; Kangwook Lee; Dongmin Park
>
> **摘要:** Recent advances in text-to-speech (TTS) models show impressive speech naturalness and quality, yet the role of large-scale open data in driving this progress remains underexplored. In this work, we introduce Raon-OpenTTS, an open TTS model that performs competitively with state-of-the-art closed-data TTS models, and Raon-OpenTTS-Pool, a large-scale open dataset for reproducible TTS training. Raon-OpenTTS-Pool consists of 615K hours of 240M speech segments aggregated from publicly available English speech corpora and web-sourced recordings. With a model-based filtering pipeline applied to Raon-OpenTTS-Pool, we derive Raon-OpenTTS-Core, a curated, high-quality subset of 510K hours and 194M speech segments. Using Raon-OpenTTS-Core, we train Raon-OpenTTS, a series of diffusion transformer (DiT)-based TTS models from 0.3B to 1B parameters. On multiple benchmarks, Raon-OpenTTS-1B shows comparable performance to state-of-the-art models such as Qwen3-TTS and CosyVoice 3, which are trained on several million hours of proprietary speech data. Notably, on Seed-TTS-Eval, Raon-OpenTTS-1B achieves a word error rate (WER) of 1.78% and a speaker similarity (SIM) of 0.749, ranking second on WER and first on SIM among recent open-weight TTS baselines. On CV3-Hard-EN, Raon-OpenTTS-1B achieves a WER of 6.15% and a SIM of 0.775, ranking first on both metrics. Furthermore, to support robust evaluation, we introduce Raon-OpenTTS-Eval, a structured benchmark for assessing TTS robustness across diverse acoustic conditions including clean, noisy, in-the-wild, and expressive speech. On Raon-OpenTTS-Eval, Raon-OpenTTS-1B achieves the best average WER and SIM among all evaluated models, and the second-best human preference, as measured by comparative mean opinion score (CMOS). Our data pool, filtering pipeline, training code, and checkpoints are publicly available at this https URL.
>
---
#### [new 012] SEABAD: A Tropical Bird Activity Detection Dataset for Passive Acoustic Monitoring
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SEABAD数据集，用于解决热带地区鸟类声音检测问题，以提升被动声学监测的效率。**

- **链接: [https://arxiv.org/pdf/2605.20853](https://arxiv.org/pdf/2605.20853)**

> **作者:** Muhammad Mun'im Ahmad Zabidi; Mohd Yamani Idna Idris; Norisma Idris
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Passive acoustic monitoring (PAM) enables large-scale biodiversity assessment, but continuous recording generates large amounts of non-informative audio, creating challenges for storage, power consumption, and long-term edge deployment. Bird audio detection (BAD), which identifies bird vocalizations, can reduce this burden by filtering irrelevant recordings before downstream analysis. However, most BAD systems are trained on temperate datasets despite tropical soundscapes being denser, more species-rich, and acoustically unpredictable. To address this gap, we introduce SEABAD (Southeast Asian Bird Activity Detection), a dataset of 50,000 curated three-second clips from Southeast Asian soundscapes, evenly balanced between bird-present and bird-absent samples. The dataset spans 1,677 bird species and is standardized to 16 kHz mono audio for embedded and low-power inference. We developed a dual-branch curation pipeline: a six-stage positive-label workflow applied to Xeno-Canto recordings, alongside six source-specific negative-label extractions from environmental datasets. These procedures reduced class imbalance by 13.7% (Gini coefficient: 0.601 to 0.519). A manual audit of 1,000 positive clips confirmed 97.8% +/- 0.9% labeling accuracy. Baseline experiments using MobileNetV3-Small achieved 99.57% +/- 0.25% accuracy and 0.9985 +/- 0.0002 AUC across three random seeds. SEABAD and the full curation pipeline are publicly released to support tropical BAD research and energy-efficient acoustic monitoring.
>
---
#### [new 013] DuplexSLA: A Full-Duplex Spoken Language Model with Synchronized Speech, Language, and Action
- **分类: eess.AS**

- **简介: 该论文提出DuplexSLA，解决全双工对话中实时规划与工具调用问题，通过同步语音、语言和动作流实现高效交互。**

- **链接: [https://arxiv.org/pdf/2605.20755](https://arxiv.org/pdf/2605.20755)**

> **作者:** Haoyang Zhang; Jun Chen; Donghang Wu; Yuxin Li; Yuxin Zhang; Xiangyu Tony Zhang; Che Liu; Qingjian Lin; Yizhou Peng; Hexin Liu; Eng Siong Chng; Chao Yan; Boyong Wu; Yechang Huang; Xuerui Yang; Fei Tian
>
> **摘要:** Recent advances in spoken dialogue language models have shifted from turn-based to full-duplex designs, where the model continuously listens to the user while generating responses. However, existing duplex backbones still lack a native channel for in-conversation planning and tool calling, leaving real-time agentic behaviour either tied to turn boundaries or relegated to an external cascade. We propose DuplexSLA, a native full-duplex Speech-Language-Action foundation model that decodes assistant audio together with a structured action stream on a shared 160 ms chunk timeline. DuplexSLA is built on a dual-stream three-channel formulation: a continuous user audio channel, a discrete assistant audio channel, and a rate-limited textual action channel, all decoded jointly by a single backbone, so that listening, speaking, planning, and tool calling unfold on one shared clock. Two capabilities define the model: (1) semantic-driven turn-taking control, where interruption, pause, and backchannel are handled inside the same backbone instead of by an external semantic VAD; and (2) in-conversation planning and tool calling, where planning text and structured tool calls are emitted on the action channel without halting assistant audio, so that multi-action and backchannel-triggered tool use are interleaved with ongoing speech. To evaluate these capabilities together, we further construct DuplexSLA-Bench, a duplex benchmark covering pause, interrupt, and backchannel turn-taking together with three styles of in-conversation tool calling. Our project page, interactive demos, and the DuplexSLA-Bench evaluation suite are publicly available at this https URL.
>
---
#### [new 014] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding
- **分类: eess.AS**

- **简介: 该论文属于长音频理解任务，解决大模型处理长音频时的效率与准确性问题。提出PlanRAG-Audio框架，通过规划与检索增强生成，提升推理效果并减少输入长度。**

- **链接: [https://arxiv.org/pdf/2605.20414](https://arxiv.org/pdf/2605.20414)**

> **作者:** Masao; Someki; Chien-yu; Huang; Siddhant; Arora; Samuele; Cornell; Markus; Müller; Nathan; Susanj; Rupak V; Swaminathan; Grant P; Strimel; Jing; Shinji; Watanabe
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Long-form audio understanding poses significant challenges for large audio language models (LALMs) due to the extreme length of audio sequences and the need to reason over heterogeneous acoustic cues distributed over time, such as speech content, speaker identity, emotion, and sound events. To address these challenges, we propose \textbf{PlanRAG-Audio}, a planning-based retrieval-augmented generation framework for scalable long-form audio understanding. Rather than having audio LALMs process entire recordings directly, PlanRAG-Audio explicitly plans which modalities and temporal spans are required for a given query, and retrieves only query-relevant information from a structured text and audio database. This retrieval planning enables effective reasoning over complex, cross-domain audio queries while substantially reducing the input length passed to the large language models. Experiments across a wide range of speech/audio retrieval demonstrate that PlanRAG-Audio improves reasoning accuracy and stabilizes performance as audio duration increases by decoupling inference cost from raw audio length.
>
---
#### [new 015] CoarseSoundNet: Building a reliable model for ecological soundscape analysis
- **分类: cs.SD; cs.LG**

- **简介: 该论文属于 soundscape 分类任务，旨在解决生态声音成分（生物声、自然声、人为声）难以准确量化的问题。工作包括构建 CoarseSoundNet 模型，并验证其在真实 PAM 数据中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.21143](https://arxiv.org/pdf/2605.21143)**

> **作者:** Alexander Gebhard; Andreas Triantafyllopoulos; Dominik Arend; Sandra Müller; Svenja Schmidt; Michael Scherer-Lorenzen; Björn W. Schuller
>
> **备注:** Currently under review
>
> **摘要:** A soundscape is composed of three types of sound: biophony (sounds made by animals), geophony (natural abiotic sounds) and anthropophony (sounds made by humans). A key research question in the field of soundscape ecology is how these components interact with each other, specifically how biophony responds to geophony and anthropophony. Nevertheless, as of today, there are not many analytical instruments that enable the distinct quantification of these elements. Recent machine learning (ML) approaches aim to support automated analysis but often rely on task-specific or clean data, limiting generalisation to noisy passive acoustic monitoring (PAM) recordings. This study presents a clear and reproducible structure to build ML models for coarse soundscape classification and introduces CoarseSoundNet, a deep learning model trained to distinguish biophony, geophony, and anthropophony under realistic PAM conditions. We systematically investigate model architectures, the influence of an additional training class, data composition, and evaluation strategies. Our findings suggest that model performance improves with additional PAM data, especially when similar to the target domain, and by introducing an explicit silence class during training. Class-specific decision thresholds and duration-based constraints further enhance performance, particularly for anthropophony and geophony. Error analyses exhibit challenges for anthropophony due to masking effects and confusions for silence and insect sounds for geophony and biophony. Finally, we conduct an ecological case study which shows that pre-filtering recordings with CoarseSoundNet yields acoustic index trends comparable to ground-truth filtering, supporting its use as an effective preprocessing tool for ecoacoustic analyses.
>
---
#### [new 016] A strongly annotated passive acoustic dataset for tropical bird monitoring
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于鸟类监测任务，旨在解决热带地区声学数据标注不足的问题。研究构建了PteroSet数据集，包含大量鸟类声音及其详细标注，用于机器学习模型训练与评估。**

- **链接: [https://arxiv.org/pdf/2605.20578](https://arxiv.org/pdf/2605.20578)**

> **作者:** Daniela Ruiz; Juan Sebastián Ulloa; Zhongqi Miao; Nicolás Betancourt; Maria Paula Toro-Gómez; Andrés Hernández; Bruno Demuro; Eliana Barona-Cortés; Angela Mendoza-Henao; Andrés Sierra-Ricaurte; Sebastián Pérez-Peña; Rahul Dodhia; Pablo Arbeláez; Juan M. Lavista Ferres
>
> **摘要:** Passive acoustic monitoring enables continuous, non-invasive biodiversity assessment across diverse ecosystems. The scale of these datasets has driven the adoption of machine learning, with supervised approaches showing strong performance. However, supervised methods require time-resolved annotated datasets, which remain scarce, especially in complex tropical soundscapes. We present PteroSet, a curated dataset of strongly annotated Neotropical bird vocalizations recorded in Puerto Asis (Putumayo) and Pivijay (Magdalena), Colombia, between 2023 and 2025. The dataset comprises 563 recordings (73.62 h) and 15,372 time-frequency annotations, including 6,702 events identified to the species level across 168 species. We release the annotations in a COCO-inspired JSON schema that unifies audio files, taxonomic categories, and labels for machine learning workflows. Beyond providing annotated data, PteroSet serves as a realistic benchmark that highlights key characteristics of tropical soundscapes, including acoustic co-occurrence and domain shift across recording sites. We provide a deep learning baseline for binary bird detection, demonstrating PteroSet's usability and the challenges it presents.
>
---
#### [new 017] Evaluating Speech Articulation Synthesis with Articulatory Phoneme Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音合成任务，旨在解决合成语音质量评估难题。通过使用音素识别作为代理指标，评估语音发音合成效果。**

- **链接: [https://arxiv.org/pdf/2605.20920](https://arxiv.org/pdf/2605.20920)**

> **作者:** Vinicius Ribeiro; Yves Laprie
>
> **备注:** Accepted for publication at the European Signal Processing Conference (EUSIPCO), 2026
>
> **摘要:** Recent advances in machine learning and the availability of articulatory datasets allow vocal tract synthesis to be conditioned on phonetic sequences, a primary task of articulatory speech synthesis. However, quality assessment needs a better definition. Generally, ranking generative models is tricky due to subjectivity. However, articulatory synthesis has the additional difficulty of requiring specialized knowledge in vocal tract anatomy and acoustics. To address this problem, this paper proposes to evaluate speech articulation synthesis using phoneme recognition as a proxy. Our hypothesis is that phoneme recognition using articulatory features better captures nuances in phoneme production, such as correct places of articulation, which traditional metrics (e.g., point-wise distance metrics) do not. We train a neural network with acoustic and articulatory features extracted from a single-speaker RT-MRI dataset. Then, we compare the recognition performance when testing the model with different synthetic articulatory features. Our results show that our articulatory feature set is phonetically rich and helps exploring additional dimensions on speech articulation synthesis.
>
---
#### [new 018] Music of Changing Lines: Toward a Culturally Situated Approach to the I-Ching
- **分类: cs.MM; cs.CY; cs.HC; cs.SD**

- **简介: 该论文属于跨学科艺术与AI研究任务，旨在解决传统文本在现代音乐中的意义流失问题。通过构建交互系统，将《易经》作为有意义框架，结合AI进行实时音乐生成与解释。**

- **链接: [https://arxiv.org/pdf/2605.20386](https://arxiv.org/pdf/2605.20386)**

> **作者:** Ling Qi; Aleksandra Teng Ma; Alexandria Smith
>
> **备注:** Published and presented at the International Computer Music Conference (ICMC) 2026
>
> **摘要:** The I-Ching is one of the most influential texts in Chinese intellectual history, integrating divination, cosmology, and ethical reflection. While Western experimental music, most notably John Cage, has drawn on the I-Ching as a source of chance operation, such appropriations have often detached its formal mechanisms from the interpretive and philosophical processes that give the text meaning. This work, Music of Changing Lines, presents an interactive system that re-centers the I-Ching as a meaning-bearing framework rather than a neutral randomizer. Users perform Wen Wang Fa coin casting, which is accompanied in real time through probabilistic musical processes. The resulting hexagrams and changing lines are interpreted by a large language model, Gemini, in relation to the user's inquiry. This textual interpretation is then translated into a prompt for a generative music model, Lyria, producing a responsive musical realization. By situating AI as an interpretive intermediary rather than a compositional authority, the system foregrounds the I-Ching's ritual, interpretation, and participation as the primary sonic materials. Music of Changing Lines extends process-driven traditions in computer music by demonstrating how generative AI can support participatory, meaning-driven musical processes without prescribing musical structure or replacing human agency.
>
---
#### [new 019] Synchronization and Turn-Taking in Full-Duplex Speech Dialogue Models
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于对话系统任务，研究全双工语音对话模型的同步与轮流机制。通过实验分析模型在不同噪声下的表示同步和预判能力，以提升自然交互效果。**

- **链接: [https://arxiv.org/pdf/2605.20356](https://arxiv.org/pdf/2605.20356)**

> **作者:** Pablo Riera; Pablo Brusco; Cristina Kuo; Marcelo Sancinetti; S.R.K. Branavan
>
> **摘要:** Full-duplex spoken dialogue models (SDMs) can listen and speak simultaneously, enabling interaction dynamics closer to human conversation than turn-based systems. Inspired by neural coupling in human communication, we study how such models coordinate their internal representations during interaction. We simulate full-duplex dialogues between two instances of the pretrained \textit{Moshi} model under controlled conditions, manipulating channel noise and decoding bias. Synchronization is measured using Centered Kernel Alignment (CKA) across temporal lags, while anticipatory turn-taking cues are probed from delayed internal activations using causal LSTM models, from both speaker and listener perspectives. We find strong representational synchronization under no noise conditions, peaking near zero lag and degrading with noise, and we show that internal states encode anticipatory information that supports turn-taking prediction ahead of time.
>
---
## 更新

#### [replaced 001] Discriminative-Generative Target Speaker Extraction with Decoder-Only Language Models
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于目标说话人提取任务，旨在提升语音质量与自然度。提出一种判别-生成框架，结合两者优势，解决现有方法在感知质量和可控性上的不足。**

- **链接: [https://arxiv.org/pdf/2601.06006](https://arxiv.org/pdf/2601.06006)**

> **作者:** Bang Zeng; Beilong Tang; Wang Xiang; Ming Li
>
> **备注:** 13 pages,4 figures
>
> **摘要:** Target speaker extraction (TSE) aims to recover the speech of a desired speaker from a mixture given a short enrollment utterance, while speech enhancement (SE) focuses on improving speech quality under noisy conditions. Most existing TSE and SE systems are based on discriminative modeling and have shown strong interference suppression ability, but they often remain limited in perceptual quality and naturalness. To address this issue, we first introduce LauraTSE, a generative TSE model built on an autoregressive decoder-only language model. Although generative modeling is promising for quality enhancement, purely generative TSE may suffer from hallucination, content drift, and limited controllability in complex acoustic conditions. We therefore propose a discriminative-generative two-stage framework, where a discriminative front-end first produces target-related representations with strong interference suppression, and a generative back-end then reconstructs high-quality speech in the neural audio codec representation space. This design combines the controllability of discriminative extraction with the reconstruction capability of generative modeling. We further investigate several collaboration strategies for the two-stage framework, including front-end freezing, joint fine-tuning, SI-SDR regularization, and autoregressive/non-autoregressive inference. Experimental results on both TSE and SE benchmarks show that the proposed framework achieves a better balance among perceptual quality, intelligibility, and speaker consistency than purely discriminative or purely generative baselines.
>
---
#### [replaced 002] Multi-Channel Replay Speech Detection using Acoustic Maps
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文属于语音安全任务，旨在解决自动说话人验证中的重放攻击检测问题。通过多通道声学图特征和轻量CNN，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2602.16399](https://arxiv.org/pdf/2602.16399)**

> **作者:** Michael Neri; Tuomas Virtanen
>
> **备注:** Accepted in EUSIPCO 2026
>
> **摘要:** Replay attacks remain a critical vulnerability for automatic speaker verification systems, particularly in real-time voice assistant applications. In this work, we propose acoustic maps as a novel spatial feature representation for replay speech detection from multi-channel recordings. Derived from classical beamforming over discrete azimuth and elevation grids, acoustic maps encode directional energy distributions that reflect physical differences between human speech radiation and loudspeaker-based replay. A lightweight convolutional neural network is designed to operate on this representation, achieving competitive performance on the ReMASC dataset with approximately 6k trainable parameters. Experimental results show that acoustic maps provide a compact and physically interpretable feature space for replay attack detection across different devices and acoustic environments.
>
---
#### [replaced 003] AuDirector: A Self-Reflective Closed-Loop Framework for Immersive Audio Storytelling
- **分类: cs.SD**

- **简介: 该论文提出AuDirector，解决长音频叙事的一致性与表达问题。通过多智能体框架实现自反思闭环，提升语音适配、质量修正和用户交互。属于音频故事生成任务。**

- **链接: [https://arxiv.org/pdf/2605.11866](https://arxiv.org/pdf/2605.11866)**

> **作者:** Yiming Ren; Xuenan Xu; Ziyang Zhang; Wen Wu; Baoxiang Li; Chao Zhang
>
> **摘要:** Despite advances in text and visual generation, creating coherent long-form audio narratives remains challenging. Existing frameworks often exhibit limitations such as mismatched character settings with voice performance, insufficient self-correction mechanisms, and limited human interactivity. To address these challenges, we propose AuDirector, a self-reflective closed-loop multi-agent framework. Specifically, it involves an Identity-Aware Pre-production mechanism that transforms narrative texts into character profiles and utterance-level emotional instructions to retrieve suitable voice candidates and guide expressive speech synthesis, thereby promoting context-aligned voice adaptation. To enhance quality, a Collaborative Synthesis and Correction module introduces a closed-loop self-correction mechanism to systematically audit and regenerate defective audio components. Furthermore, a Human-Guided Interactive Refinement module facilitates user control by interpreting natural language feedback to interactively refine the underlying scripts. Experiments demonstrate that AuDirector achieves superior performance compared to state-of-the-art baselines in structural coherence, emotional expressiveness, and acoustic fidelity. Audio samples can be found at this https URL.
>
---
#### [replaced 004] VoxATtack: A Multimodal Attack on Voice Anonymization Systems
- **分类: eess.AS**

- **简介: 该论文提出VoxATtack，用于攻击语音匿名化系统，通过结合声学和文本信息提升识别性能，揭示现有方法的漏洞。**

- **链接: [https://arxiv.org/pdf/2507.12081](https://arxiv.org/pdf/2507.12081)**

> **作者:** Ahmad Aloradi; Ünal Ege Gaznepoglu; Emanuël A. P. Habets; Daniel Tenbrinck
>
> **备注:** 5 pages, 3 figures, 3 tables, accepted at WASPAA 2025
>
> **摘要:** Voice anonymization systems aim to protect speaker privacy by obscuring vocal traits while preserving the linguistic content relevant for downstream applications. However, because these linguistic cues remain intact, they can be exploited to identify semantic speech patterns associated with specific speakers. In this work, we present VoxATtack, a novel multimodal de-anonymization model that incorporates both acoustic and textual information to attack anonymization systems. While previous research has focused on refining speaker representations extracted from speech, we show that incorporating textual information with a standard ECAPA-TDNN improves the attacker's performance. Our proposed VoxATtack model employs a dual-branch architecture, with an ECAPA-TDNN processing anonymized speech and a pretrained BERT encoding the transcriptions. Both outputs are projected into embeddings of equal dimensionality and then fused based on confidence weights computed on a per-utterance basis. When evaluating our approach on the VoicePrivacy Attacker Challenge (VPAC) dataset, it outperforms the top-ranking attackers on five out of seven benchmarks, namely B3, B4, B5, T8-5, and T12-5. To further boost performance, we leverage anonymized speech and SpecAugment as augmentation techniques. This enhancement enables VoxATtack to achieve state-of-the-art on all VPAC benchmarks, after scoring 20.6% and 27.2% average equal error rate on T10-2 and T25-1, respectively. Our results demonstrate that incorporating textual information and selective data augmentation reveals critical vulnerabilities in current voice anonymization methods and exposes potential weaknesses in the datasets used to evaluate them.
>
---
#### [replaced 005] Voice ''Cloning'' is Style Transfer
- **分类: cs.SD; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于语音生成任务，揭示 voice cloning 实质为风格迁移，而非真实克隆。研究指出其导致语音特征同质化及人类行为影响，提出技术局限与风险。**

- **链接: [https://arxiv.org/pdf/2605.16578](https://arxiv.org/pdf/2605.16578)**

> **作者:** Kaitlyn Zhou; Federico Bianchi; Martijn Bartelds; Anna Pot; Yongchan Kwon; James Zou
>
> **摘要:** Artificially generated speech is increasingly embedded in everyday life. Voice cloning in particular enables applications where identity preservation is important, such as completing a recording, dubbing in a new language, or preserving the voices of individuals with speech loss. However, in our work, we find that despite the term, voice cloning does not faithfully ''clone'' an individual's voice. Instead, we find that widely-used voice cloning models systematically apply style transfer to source voices. As rated by human annotators, cloned voices are perceived as more authoritative, warm, customer-service-like, and human-like compared to their sources. Human annotators also report greater trust in cloned voices than source voices, and a greater willingness to disclose sensitive personal information to them. Our work furthermore shows that voice cloning leads to homogenization of speaker characteristics, as measured by reduced variance in accent, speaking rate, and the audio embedding space. Together, our results highlight a new set of limitations and risks of voice cloning technology and their potential impact on human behavior.
>
---
#### [replaced 006] You Are What You Say: Exploiting Linguistic Content for VoicePrivacy Attacks
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音隐私攻击任务，研究语言内容对说话人识别的影响。通过BERT模型评估语音匿名化系统的隐私安全性，发现文本相似性影响攻击效果，并提出数据集改进建议。**

- **链接: [https://arxiv.org/pdf/2506.09521](https://arxiv.org/pdf/2506.09521)**

> **作者:** Ünal Ege Gaznepoglu; Anna Leschanowsky; Ahmad Aloradi; Prachi Singh; Daniel Tenbrinck; Emanuël A. P. Habets; Nils Peters
>
> **备注:** 5 pages, 6 figures, 1 table, accepted at INTERSPEECH 2025 update reason: change to the acknowledgements
>
> **摘要:** Speaker anonymization systems hide the identity of speakers while preserving other information such as linguistic content and emotions. To evaluate their privacy benefits, attacks in the form of automatic speaker verification (ASV) systems are employed. In this study, we assess the impact of intra-speaker linguistic content similarity in the attacker training and evaluation datasets, by adapting BERT, a language model, as an ASV system. On the VoicePrivacy Attacker Challenge datasets, our method achieves a mean equal error rate (EER) of 35%, with certain speakers attaining EERs as low as 2%, based solely on the textual content of their utterances. Our explainability study reveals that the system decisions are linked to semantically similar keywords within utterances, stemming from how LibriSpeech is curated. Our study suggests reworking the VoicePrivacy datasets to ensure a fair and unbiased evaluation and challenge the reliance on global EER for privacy evaluations.
>
---
#### [replaced 007] Speech Enhancement Based on Drifting Models
- **分类: cs.SD; cs.AI; eess.AS; eess.SP**

- **简介: 该论文提出DriftSE，一种基于漂移模型的语音增强框架，解决单步语音去噪问题，通过分布匹配实现高效增强。**

- **链接: [https://arxiv.org/pdf/2604.24199](https://arxiv.org/pdf/2604.24199)**

> **作者:** Liang Xu; Diego Caviedes-Nozal; W. Bastiaan Kleijn; Longfei Felix Yan; Rasmus Kongsgaard Olsson
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** We propose Speech Enhancement based on Drifting Models (DriftSE), a novel generative framework that formulates denoising as an equilibrium problem. Rather than relying on iterative sampling, DriftSE natively achieves one-step inference by evolving the pushforward distribution of a mapping function to directly match the clean speech distribution. This evolution is driven by a Drifting Field, a learned correction vector that guides samples toward the high-density regions of the clean distribution, which naturally facilitates training on unpaired data by matching distributions rather than paired samples. We investigate the framework under two formulations: a direct mapping from the noisy observation, and a stochastic conditional generative model from a Gaussian prior. Experiments on the VoiceBank-DEMAND benchmark demonstrate that DriftSE achieves high-fidelity enhancement in a single step, outperforming multi-step diffusion baselines and establishing a new paradigm for speech enhancement.
>
---
#### [replaced 008] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
- **分类: eess.AS; cs.CL**

- **简介: 该论文提出FLAIR方法，用于全双工对话系统中的隐式推理，解决语音交互中同时进行思考与响应的问题。通过连续推理提升对话质量。**

- **链接: [https://arxiv.org/pdf/2603.17837](https://arxiv.org/pdf/2603.17837)**

> **作者:** Donghang Wu; Tianyu Zhang; Yuxin Li; Hexin Liu; Chen Chen; Eng Siong Chng; Yoshua Bengio
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
>
---
#### [replaced 009] Enhancing Speech Large Language Models through Reinforced Behavior Alignment
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型任务，解决SpeechLM在指令遵循上的性能不足问题。通过引入RBA框架，利用强化学习提升模型能力，取得优异效果。**

- **链接: [https://arxiv.org/pdf/2509.03526](https://arxiv.org/pdf/2509.03526)**

> **作者:** Yansong Liu; Jiateng Li; Yuan Liu
>
> **摘要:** The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data.
>
---
#### [replaced 010] Iterative LLM-based improvement for French Clinical Interview Transcription and Speaker Diarization
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于语音转录与说话人辨识任务，旨在降低法语临床对话的识别错误率。通过多轮LLM后处理提升准确性，实验验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2603.00086](https://arxiv.org/pdf/2603.00086)**

> **作者:** Ambre Marie; Thomas Bertin; Guillaume Dardenne; Gwenolé Quellec
>
> **摘要:** Automatic speech recognition for French medical conversations remains challenging, with word error rates often exceeding 30% in spontaneous clinical speech. This study proposes a multi-pass LLM post-processing architecture alternating between Speaker Recognition and Word Recognition passes to improve transcription accuracy and speaker attribution. Ablation studies on two French clinical datasets (suicide prevention telephone counseling and preoperative awake neurosurgery consultations) investigate four design choices: model selection, prompting strategy, pass ordering, and iteration depth. Using Qwen3-Next-80B, Wilcoxon signed-rank tests confirm significant WDER reductions on suicide prevention conversations (p<0.05, n=18), while maintaining stability on awake neurosurgery consultations (n=10), with zero output failures and acceptable computational cost (RTF 0.32), suggesting feasibility for offline clinical deployment, pending validation on larger corpora.
>
---
