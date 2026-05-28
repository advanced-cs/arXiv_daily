# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Do Audio LLMs Listen or Read? Analyzing and Mitigating Paralinguistic Failures with VoxParadox
- **分类: cs.SD; cs.LG**

- **简介: 该论文研究音频大语言模型在理解语气等非语言信息上的不足，提出VoxParadox基准和PCLM方法，提升模型对语音语调的理解能力。**

- **链接: [https://arxiv.org/pdf/2605.27772](https://arxiv.org/pdf/2605.27772)**

> **作者:** Jiacheng Pang; Ashutosh Chaubey; Mohammad Soleymani
>
> **备注:** Accepted as a conference paper at ICML 2026. Project page: this https URL
>
> **摘要:** Audio large language models (Audio LLMs) demonstrate strong performance on speech understanding tasks, yet their ability to understand paralinguistic information remains limited. To systematically quantify this issue, we introduce VoxParadox, an adversarial benchmark with 2,000 verified examples, spanning 10 paralinguistic tasks, created with controlled speech synthesis to intentionally mismatch transcript claims and speaking style, enabling direct measurement of speech paralinguistic understanding. Evaluation of a diverse set of Audio LLMs reveals consistently low accuracy on acoustic ground truth and a strong tendency to follow language-implied (incorrect) answers. To understand the cause of this gap, we perform layer-wise probing and find that (i) paralinguistic cues can degrade in deeper encoder layers and at the encoder--LLM interface, and (ii) even when such cues are available in audio tokens, the language model frequently ignores them. To address these problems, we propose Prompt-Conditioned Layer Mixer (PCLM), which adaptively combines information from multiple audio layers based on the input prompt, and pair it with Direct Preference Optimization (DPO) to explicitly prefer acoustically supported options over language-implied alternatives. These methods substantially improve Audio LLM paralinguistic understanding, improving Audio Flamingo 3 from 17.40% to 65.20% on VoxParadox, and from 37.74% to 54.78% on MMSU paralinguistic subset. Our project page is available at this https URL.
>
---
#### [new 002] I Hear, Therefore I Trust: A Socio-Technical Investigation of Humans as Synthetic Speech Detectors
- **分类: eess.AS; cs.AI; cs.HC**

- **简介: 该论文研究人类检测合成语音的能力，属于语音检测任务。旨在探讨合成语音的感知与信任因素影响，通过实验分析不同语音类型和信任提示对检测效果的影响。**

- **链接: [https://arxiv.org/pdf/2605.28064](https://arxiv.org/pdf/2605.28064)**

> **作者:** Lelia Erscoi; Tomi Kinnunen
>
> **备注:** To be included in Odyssey 2026: The Speaker and Language Recognition Workshop, Session 4.2, 23-26 June, Lisbon, Portugal
>
> **摘要:** Automatic deepfake detection has received considerable research attention, yet the socio-technical environment in which humans actually encounter synthetic speech remains poorly understood. We investigate voice deepfake detection as a perceptual and contextual process, presenting a localization task in which 47 participants marked suspected synthetic segments across authentic, fully synthetic, and partially synthetic utterances under three manipulated trust cues: instructional framing, affective priming, and provenance labeling. Participants provided quality ratings on mechanicalness, expressiveness, intelligibility, clarity, calmness, and confidence of evaluation. Utterance class was the primary determinant of detection accuracy and perceptual quality; trust cues produced no main effects but motivated detection behavior. Fully synthetic speech was detected at below-chance levels. Quality ratings tracked utterance type, indicating implicit discrimination where overt detection failed.
>
---
#### [new 003] DEMON: Diffusion Engine for Musical Orchestrated Noise
- **分类: cs.SD**

- **简介: 该论文提出DEMON，一个实时扩散引擎，用于音乐降噪，解决实时控制与高效生成问题，通过四项机制提升性能与交互性。**

- **链接: [https://arxiv.org/pdf/2605.28657](https://arxiv.org/pdf/2605.28657)**

> **作者:** Ryan Fosdick
>
> **备注:** 15 pages, 3 figures, 15 tables. Project page with audio samples and demo video: this https URL
>
> **摘要:** We present DEMON, a real-time diffusion engine that makes the denoising process playable as a live musical instrument: a control surface both broad (many parameters shaped per-frame across the output) and responsive (each control taking effect as fast as its place in the denoising loop allows). Built on ACE-Step 1.5 and StreamDiffusion's ring-buffer architecture with TensorRT acceleration, it sustains up to 12.3 decoder completions per second for 60-second music on a single consumer GPU (RTX 5090), or 11.3 generations per second at our production ring-depth of 4. At these rates denoising parameters become viable as live performance controls, but the ring buffer propagates per-request changes only at its drain rate, a floor of S denoising steps. We contribute four mechanisms. (1) Per-slot heterogeneous denoise scheduling: each ring-buffer slot owns its timestep schedule, so a moving denoise slider is tracked without wiping the in-flight queue, where the upstream global-schedule design must rebuild and discard it. (2) Shared mutable per-step state, giving any parameter consulted at every solver step next-tick effect, bypassing ring-buffer drain. (3) Per-frame source blending: a sampling-time control on the standard SDE re-noise step, giving a framewise transformation-strength axis that complements scalar denoise scheduling. (4) Windowed VAE decode exploiting receptive-field analysis for an 8.0x decode speedup. Together these separate streaming-diffusion parameters into four propagation classes, by onset and convergence latency.
>
---
#### [new 004] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios
- **分类: eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决长文本语音评估不足的问题。提出SwanBench-Speech基准，涵盖多种场景，提供全面评估维度，揭示现有模型在表达和连贯性上的不足。**

- **链接: [https://arxiv.org/pdf/2605.28618](https://arxiv.org/pdf/2605.28618)**

> **作者:** Changhao Pan; Rui Yang; Han Wang; Zhuan Zhou; Xuming He; Wenxiang Guo; Ziyue Jiang; Ruiqi Li; Yu Zhang; Chenyuhao Wen; Ke Lei; Xiang Yin; Jingyu Lu; Zhiyuan Zhu; Zhou Zhao
>
> **备注:** Accepted by ACL 2026(Findings). 36pages, 14figures
>
> **摘要:** Recent advances in speech generation have enabled high-fidelity synthesis, yet systematic evaluation of models under long-context conditions remains largely underexplored. A comprehensive evaluation benchmark for long-form speech is indispensable for two reasons: 1) existing test scenarios are often confined to limited domains, creating a significant gap with the diverse downstream applications; 2) existing metrics overlook critical long-text factors such as consistency and coherence, failing to generalize reliably. To this end, we propose Swanbench-Speech, a comprehensive benchmark that decomposes long-form speech quality into specific, disentangled dimensions. SwanBench-Speech has three key properties. 1) Rich speech scenarios: Focusing on long-form speech generation and dialog generation, SwanBench-Speech covers acoustics, semantics, and expressiveness challenges, and consists of 1,101 samples spanning 17 common speech scenarios; 2) Comprehensive evaluation dimensions: Along the acoustics, semantics, and expressiveness axes, SwanBench-Speech defines an automated evaluation protocol with seven metrics to provide a comprehensive, accurate, and standardized assessment; 3) Valuable Insights: Through extensive experiments, we reveal that current models still struggle in highly expressive scenarios and exhibit a notable gap in consistency and hierarchy compared to real recordings.
>
---
#### [new 005] VoiceGiraffe: A Benchmark for Extreme Long-Context Audio-Language Understanding
- **分类: cs.SD**

- **简介: 该论文属于音频-语言理解任务，旨在解决长时音频理解的瓶颈问题。提出VoiceGiraffe基准，评估模型在长时间音频中的信息 comprehension 能力。**

- **链接: [https://arxiv.org/pdf/2605.27976](https://arxiv.org/pdf/2605.27976)**

> **作者:** Jashin Ye; Dongxiao Wang; Yixuan Ye; Sashuai Zhou; Weihuang Lin; Mingyang Han; Kunpeng Wang; Zeyu Yuan; Boyu Li; Haoxiang Shi; Jingchen Shu; Jun Song; Bo Zheng
>
> **备注:** Benchmark Project: this https URL
>
> **摘要:** While large audio language models (LALMs) have achieved remarkable progress in audio processing at the second- or minute-level scale, understanding hour-level audio remains a fundamental bottleneck. Existing benchmarks predominantly rely on short clips or artificially concatenated segments, failing to faithfully assess LALM capacity for long-range information comprehension in real-world scenarios such as podcasts and lengthy speeches. To address this gap, we introduce VoiceGiraffe, a novel benchmark designed to rigorously evaluate LALMs across diverse real-world scenarios, modalities, and languages under long-context settings. It comprises 1500 curated triplets structured into a dual-level taxonomy of single-hop perception and multi-hop reasoning. We evaluate a broad suite of open-source and proprietary LALMs against human performance. Results underscore three fundamental findings. First, VoiceGiraffe remains highly challenging and far from saturation. Second, we show that no single inference paradigm universally dominates. The E2E inference benefits models with native long-context audio understanding, cascaded caption aggregation stabilizes small models overwhelmed by hour-scale audio, and reasoning-enhanced cascading with external LLM helps weaker models but can bottleneck stronger proprietary systems. Third, we reveal long-range memory persistence as a key bottleneck. LALMs are better at answering questions that require connecting salient causal cues than those requiring sustained tracking of sparse events across long audio, whereas humans show the opposite pattern. These findings position VoiceGiraffe as a challenging and diagnostic testbed for long-form audio understanding, highlighting the need for LALMs with persistent memory and robust long-range aggregation.
>
---
#### [new 006] Audio-Mind: An Auditable Agentic Framework for Audio Understanding
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出Audio-Mind框架，解决音频理解中的证据获取问题。通过动态结合前端与工具使用，提升音频问答准确性与可审计性。属于音频问答任务。**

- **链接: [https://arxiv.org/pdf/2605.28480](https://arxiv.org/pdf/2605.28480)**

> **作者:** Yucheng Wang; Jing Peng; Hanqi Li; Chenghao Wang; Wenming Tu; Yu Xi; Zhaokai Sun; Kai Yu; Shuai Wang
>
> **摘要:** Audio agents extend large audio-language models (LALMs) by decomposing audio questions into tool calls, intermediate evidence, and iterative reasoning steps. However, as LALMs become stronger, the key challenge shifts from enabling tool use to determining when agentic evidence acquisition genuinely benefits audio understanding. We propose Audio-Mind, an auditable and pluggable framework for conditional evidence acquisition in audio understanding. Audio-Mind dynamically combines a strong frontend with planner-guided tool use, preserving frontend judgment when initial evidence is sufficient while acquiring bounded external evidence for questions with unresolved evidence gaps. Experiments on MMAR and MSU-Bench show that Audio-Mind outperforms prior audio-agent baselines, reaching 80.4% accuracy on MMAR and 82.8% accuracy on MSU-Bench. A matched-backbone comparison highlights why this design matters: under strong audio frontends, agentic decomposition can become an orchestration bottleneck when the workflow does not preserve the frontend's holistic audio-grounded judgment. Beyond accuracy, Audio-Mind produces higher-quality, auditable reasoning traces that expose uncertainty, tool evidence, and answer rationales, offering a potential basis for more reliable audio-QA annotation and error analysis.
>
---
#### [new 007] Unified Synthesis of Compositional Speech and Sound from Free-Form Text Prompts
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出PlanAudio，解决从自由文本生成统一音频的问题，通过语义潜思链机制实现语音与声音的自然融合。**

- **链接: [https://arxiv.org/pdf/2605.28063](https://arxiv.org/pdf/2605.28063)**

> **作者:** Yuyue Wang; Xihua Wang; Xin Cheng; Yijing Chen; Ruihua Song
>
> **摘要:** Audio generation has made significant progress, yet synthesizing unified audio where speech and sounds are naturally composited remains a challenge. Current methods either rely on disjoint pipelines, which fail to capture fine-grained interactions, or require structured inputs and external text rewriting, which limits the flexibility of free-form text prompts. In this paper, we introduce a new task: Free-Form-Text-Prompt-to-Unified-Audio generation, which aims to directly synthesize unified audio containing speech, sound, and their composites from unconstrained natural language. To address this task, we propose PlanAudio, a unified, autoregressive LLM-based framework. First, it simplifies the model architecture by leveraging intrinsic LLM reasoning capability instead of traditional text encoders. Second, it introduces a semantic latent chain-of-thought mechanism, an implicit planning mechanism that bridges high-level semantic understanding and low-level acoustic synthesis. Furthermore, we create PlanAudio-Bench, a specialized benchmark for evaluating composite audio scenarios. We perform evaluations in the scenarios of speech, sound, and their composites. The results demonstrate that PlanAudio generally outperforms the existing pipeline and unified baselines, while staying competitive with models designed for a single scenario. Our analysis further reveals the superiority of semantic latent CoT over other CoT mechanisms and highlights the importance of continuous multi-scenario training curricula.
>
---
#### [new 008] Cross-modal characterization of infant cry: validation of a chest-surface accelerometer in extracting acoustic vocal function measures
- **分类: cs.SD; physics.med-ph**

- **简介: 该论文属于语音分析任务，旨在解决传统麦克风录音在婴儿哭声研究中的噪声和隐私问题。通过对比胸贴加速度计与麦克风的声学特征，验证加速度计的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28687](https://arxiv.org/pdf/2605.28687)**

> **作者:** Winko W. An; Saketh Sundar; Lisa Yankowitz; Daryush D. Mehta; Carol L. Wilkinson
>
> **摘要:** Background: Infant cry acoustics provide a promising window into early neurodevelopment and may serve as scalable biomarkers for neurodevelopmental disorders. However, conventional microphone-based recordings are highly susceptible to environmental noise and raise privacy concerns in real-world clinical settings. Chest-surface accelerometers may offer a robust alternative by capturing vibrations directly from the larynx. Methods: We evaluated the validity of a chest-mounted accelerometer (ACC) for infant cry analysis by comparing acoustic features derived from ACC and simultaneously recorded microphone (MIC) signals during routine vaccination visits. The final sample included 85 infants (41 at 4 months; 44 at 12 months) from a diverse pediatric population. Seven vocal measures were extracted from both modalities, including fundamental frequency (F0), jitter, shimmer, cepstral peak prominence (CPP), and harmonics-to-noise ratio (HNR). Agreement and consistency between modalities was assessed using intraclass correlation coefficients (ICCs). Results: F0 demonstrated excellent agreement between ACC and MIC recordings (ICC > 0.94). Jitter measures also showed good-to-excellent agreement, while CPP demonstrated moderate agreement. Shimmer and HNR showed lower absolute agreement and systematic bias between modalities, reflecting possible differences in signal transmission and noise sensitivity. Conclusion: In summary, chest-surface accelerometers can reliably capture several clinically relevant acoustic features of infant cry, particularly temporal measures of F0 and jitter. This approach offers a noise-robust and privacy-preserving alternative to microphone-based recordings, supporting its potential use in scalable clinical and developmental research applications.
>
---
#### [new 009] EigeNet: Geometry-Informed Multi-Modal Learning for Few-shot Novel View RIR Prediction
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文属于空间音频渲染任务，旨在解决从稀疏观测中预测变化的房间脉冲响应（RIR）问题。提出EIGENET框架，结合几何信息与多模态数据，提升少样本新视角RIR预测效果。**

- **链接: [https://arxiv.org/pdf/2605.28101](https://arxiv.org/pdf/2605.28101)**

> **作者:** Chong Jing; Zitong Lan; Junan Zhang; Zhizheng Wu
>
> **备注:** Code available on this https URL
>
> **摘要:** Predicting spatially varying Room Impulse Response (RIR) from sparse observations is a critical but highly challenging inverse problem for immersive spatial audio rendering. In this work, we present EIGENET, a geometry-informed multi-modal framework for few-shot novel view RIR prediction. At its core is a Cross-view Alternate-attention Transformer that iteratively refines local intra-view acoustic structures and global cross-view spatial relationships. We empirically demonstrate that this architecture is capable of making full use of the multi-view multi-modal context while performing spatial-temporal reasoning for RIR prediction. Inspired by acoustic ray tracing, we design a geometry-informed modulation block to formulate the connection between geometric features and RIR power spectrum. In the mean time, an auxiliary loss is introduced to transform the single-target waveform prediction into a multi-task learning framework. Through ablation studies, we demonstrate that this design yields consistent performance gains regardless of the underlying backbone, thereby confirming its foundational utility and architecture-agnostic generalizability for RIR prediction task. Evaluated on both simulated and real-world benchmarks, EIGENET achieves both state-of-the-art performance in few-shot novel view RIR prediction and sim-to-real generalization. Codes and checkpoints are available on this https URL.
>
---
#### [new 010] Dasheng AudioGen: A Unified Model for Generating Coherent Audio Scenes from Text
- **分类: cs.SD**

- **简介: 该论文属于音频生成任务，旨在解决多类型音频协同生成的问题。提出Dasheng AudioGen框架，通过结构化描述和统一表征实现高质量音频场景生成。**

- **链接: [https://arxiv.org/pdf/2605.27838](https://arxiv.org/pdf/2605.27838)**

> **作者:** Jiahao Mei; Heinrich Dinkel; Yadong Niu; Xingwei Sun; Gang Li; Yifan Liao; Jiahao Zhou; Junbo Zhang; Jian Luan; Mengyue Wu
>
> **摘要:** Audio generation has long been fragmented, with speech, music, and sound effects produced by domain-specific models that fail to jointly generate coherent audio scenes from a single description. The key obstacles are insufficient fine-grained supervision for real-world mixed audio and limited acoustic representations for modeling concurrent audio components. We present Dasheng AudioGen, a unified framework for generating general mixed-audio scenes from text. Dasheng AudioGen introduces structured multi-view captions, which explicitly decouple complex acoustic scenes into complementary description views, thereby enabling fine-grained control over audio layers. Furthermore, we employ a high-dimensional unified semantic-acoustic representation as the shared latent space. It injects semantic priors that facilitate cross-modal training convergence, while its high-dimensional feature space provides sufficient capacity to disentangle and fuse concurrent audio components effectively. With these designs, a simple flow-matching DiT achieves high-quality end-to-end audio scene generation. We also establish a comprehensive evaluation pipeline for audio scene generation. Experiments demonstrate that Dasheng AudioGen achieves performance approaching real-world recordings in mixed-audio categories, while remaining competitive with specialized models in single-type generation tasks. Demos are available at this https URL.
>
---
#### [new 011] LoSATok: Low-dimensional Semantic-Acoustic Tokenizer for Cross-Domain Audio Understanding and Generation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于音频理解与生成任务，旨在解决统一音频表示的建模负担问题。提出LoSATok，通过低维语义-声学分词器，提升DiT模型性能。**

- **链接: [https://arxiv.org/pdf/2605.27840](https://arxiv.org/pdf/2605.27840)**

> **作者:** Zhisheng Zhang; Xiang Li; Yixuan Zhou; Jing Peng; Guoyang Zeng; Zhiyong Wu
>
> **摘要:** Audio tokenizers are fundamental to unifying audio understanding and generation. Understanding requires high-level semantics, while generation demands semantic and acoustic details. Existing unified tokenizers jointly encode both in high-dimensional continuous latents, which increases the modeling burden of Diffusion Transformers (DiTs) for generation. We propose LoSATok, a low-dimensional audio tokenizer for cross-domain audio understanding and generation. Motivated by the observation that 1280-dimensional semantic encoder features are compressible, we introduce a Semantic Bottleneck that compresses them into 128 dimensions, regularized by the proposed time-relation loss for temporal feature consistency. We further design a dual-level semantic supervision method that leverages both high- and low-dimensional semantic signals, enabling the tokenizer to jointly capture semantics and acoustic details within a compact latent space. Experiments on speech, music, and general audio show that SemBo preserves strong low-dimensional semantic capacity and LoSATok retains competitive understanding performance compared with several semantic representations, while consistently improving DiT modeling performance on speech, music, and audio generation. These results demonstrate that LoSATok's low-dimensional representations can effectively support audio understanding and generation. Our code is provided at this https URL.
>
---
#### [new 012] Affective Music Recommendation: A Rollout-Based World Model for Offline Preference Optimization
- **分类: cs.LG; cs.IR; cs.SD**

- **简介: 该论文属于情感推荐任务，解决临床用户在线实验伦理问题。构建基于滚动的世界模型，通过离线优化提升音乐推荐效果。**

- **链接: [https://arxiv.org/pdf/2605.28810](https://arxiv.org/pdf/2605.28810)**

> **作者:** Audrey Chan; Aaron Labbé; Jacob Lavoie; Jordan Bannister; Arsène Fansi Tchango; Guillaume Lajoie; Laurent Charlin
>
> **摘要:** Functional music applications, from consumer focus and sleep aids to clinical interventions, share a distinctive recommendation problem: success is defined by the listener's affective state, but online experimentation on emotion is ethically constrained, particularly for clinical populations who cannot reliably skip a song or report distress. We describe AMRS, the Affective Music Recommendation System deployed on LUCID's health-and-wellness platforms, which serve clinical users (primarily older adults with neurocognitive conditions) and consumer-wellness users across energize, focus, calm, and sleep modes. AMRS is built around a rollout-based world model: a causal transformer trained on logged listening data to jointly predict engagement, binary rating, and self-reported valence and arousal. The world model serves both as an in-silico simulator for offline policy training and as a stress-testing tool before deployment. A recommender policy initialized by behaviour cloning is fine-tuned offline with Direct Preference Optimization (DPO) against a configurable multi-objective utility function. Under a strict cold-start protocol, the world model predicts both behavioural and affective signals with usable fidelity; DPO improves predicted valence and arousal over the cloned baseline while maintaining a similar diversity profile and avoiding the distributional collapse produced by greedy optimization. We position the work as an early deployed validation of a methodology for affective recommendation when online experimentation is ethically untenable.
>
---
#### [new 013] MTAVG-Bench 2.0: Diagnosing Failure Modes of Cinematic Expressiveness in Multi-Talker Audio-Video Generation
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于多说话人音视频生成任务，旨在解决场景级表达能力评估不足的问题。提出MTAVG-Bench 2.0基准，通过高阶失败分类体系系统评估模型在表演、叙事等方面的表现。**

- **链接: [https://arxiv.org/pdf/2605.28035](https://arxiv.org/pdf/2605.28035)**

> **作者:** Haitian Li; Yanghao Zhou; Heyan Huang; Liangji Chen; YiMing Cheng; Xu Liu; Dian Jin; Jiajun Xu; Jingyun Liao; Tian Lan; Ziqin Zhou; Yueying Liu; Yu Bai; Changsen Yuan; Jinxing Zhou; Xian-Ling Mao; Xuefeng Chen; Yousheng Feng
>
> **摘要:** In recent years, Multi-Talker Audio-Video Generation (MTAVG) models have shown promising performance on fundamental metrics such as lip-sync and audio-visual alignment. However, these metrics remain insufficient for assessing cinematic expressiveness in scene-level generation. In multi-character scenes, generation models must go beyond audio-visual realism to convey coherent character performance and other higher-level cinematic qualities. To fill this gap, we introduce MTAVG-Bench 2.0, a benchmark for diagnosing failure modes of cinematic expressiveness in multi-talker audio-video generation. Unlike prior settings that mainly focus on the quality of basic multi-turn dialogue, MTAVG-Bench 2.0 targets short-drama and scene-level generation, and establishes a high-level failure taxonomy spanning acting, narrative, atmosphere, and audio-visual language. Based on this taxonomy, we construct more than 10,000 question-answering evaluation instances, together with subsets for short-drama-level assessment and temporal localization of failure modes, to systematically evaluate the ability of omni large language models to diagnose high-level audio-visual failures. Experimental results show that commercial omni models such as Gemini substantially outperform other evaluators, yet even the strongest models continue to struggle with complex failures in our benchmark. These results demonstrate that MTAVG-Bench 2.0 provides a systematic benchmark for failure diagnosis in cinematic multi-talker audio-video generation.
>
---
#### [new 014] MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis
- **分类: q-bio.NC; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文提出MoDAl框架，用于语音神经假体中的多模态神经信号解码，解决传统方法忽略互补信息的问题，通过对比损失和去相关损失提升解码准确率。**

- **链接: [https://arxiv.org/pdf/2605.00025](https://arxiv.org/pdf/2605.00025)**

> **作者:** Yuanhao Chen; Peter Chin
>
> **摘要:** Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area.
>
---
#### [new 015] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection
- **分类: cs.AI; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉深度伪造检测任务，旨在解决歌唱场景下检测性能下降的问题。通过构建新数据集并提出跨模态框架，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.27944](https://arxiv.org/pdf/2605.27944)**

> **作者:** Ke Liu; Jiwei Wei; Wenyu Zhang; Shuchang Zhou; Ruikun Chai; Yutao Dai; Chaoning Zhang; Yang Yang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** With rapid advances in audio-visual generative models, reliable forgery detection becomes increasingly critical. Existing methods for audio-visual deepfake detection typically rely on cross-modal inconsistencies. In singing, rhythmic vocalization weakens this coupling and introduces a nontrivial domain shift, substantially degrading detection performance. We construct the Singing Head DeepFake (SHDF) dataset using rhythm-aware generative models to fill the gap in singing benchmarks. To cope with cross-scenario domain shifts, we propose a Text-guided Audio-Visual Forgery Detection (T-AVFD) framework that generalizes across both talking and singing scenarios. T-AVFD comprises a facial authenticity pattern learner and a multi-modal differential weight learning module. The pattern learner aligns facial features with multi-granularity textual descriptions to learn generalizable authenticity patterns. The weight learning module preserves intrinsic audio-visual consistency and adaptively integrates it with authenticity patterns via differential weighting. Extensive experiments on multiple talking head deepfake datasets and SHDF show consistent improvements over existing baselines and strong robustness under diverse perturbations.
>
---
#### [new 016] Diffusion Large Language Models for Visual Speech Recognition
- **分类: cs.AI; cs.CV; eess.AS**

- **简介: 该论文属于视觉语音识别任务，解决传统方法因上下文不足导致的误判问题。提出DLLM-VSR框架，通过扩散模型实现灵活解码，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2605.28456](https://arxiv.org/pdf/2605.28456)**

> **作者:** Jeong Hun Yeo; Chae Won Kim; Hyeongseop Rha; Yong Man Ro
>
> **备注:** Code: this https URL
>
> **摘要:** Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Model (DLLM)-based VSR framework, formulating transcription as iterative masked denoising with flexible-order decoding. With confidence-based unmasking, DLLM-VSR commits high-confidence positions early and uses the committed tokens as bidirectional context to refine ambiguous ones. To adapt DLLMs to VSR, we introduce a two-stage masked-denoising training strategy that separates visual-to-text content alignment from length modeling. We further observe a performance gap with oracle-length decoding, which assumes access to the true transcript length, indicating that reducing target-length uncertainty can improve DLLM-based VSR. To reduce this gap, we develop length-guided candidate decoding, which uses video duration to construct plausible transcript-length hypotheses, decodes under multiple hypotheses, and reranks candidates using length plausibility and decoding confidence. The proposed method achieves a state-of-the-art WER of 19.5\% on LRS3 using only its labeled training data.
>
---
## 更新

#### [replaced 001] PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于语音合成任务，旨在降低高质量TTS系统的数据与资源需求。通过轻量级架构和开源数据处理流程，实现高性能合成与多种语音风格生成。**

- **链接: [https://arxiv.org/pdf/2605.27258](https://arxiv.org/pdf/2605.27258)**

> **作者:** Bowen Li; Shaotong Guo; Zhen Wang; Yang Xiang; Mingli Jin; Yihang Lin; Jiahui Zhao; Weibo Xiong; Dongrui Zhang; Keming Chen; Yunze Gao; Zeyang Lin; Yuze Zhou; Yue Liu
>
> **摘要:** Building state-of-the-art text-to-speech (TTS) systems typically demands millions of hours of proprietary data and complex multi-stage architectures, creating substantial barriers for resource-constrained research teams. In this report, we present PilotTTS, a lightweight autoregressive TTS system that achieves competitive performance through minimalist architecture and rigorous data engineering. PilotTTS is trained on only 200K hours of data processed entirely with open-source tools. Specifically, our contributions are: (1) a reproducible multi-stage data processing pipeline covering quality assessment, label annotation, and filtering, and (2) a compact model architecture that employs Q-Former-based conditioning to decouple speaker identity from speaking style via cross-sample paired training. Within a unified framework, PilotTTS supports zero-shot voice cloning, emotion synthesis (11 categories), paralinguistic synthesis (4 categories), and Chinese dialect synthesis (14 dialects). On the Seed-TTS Eval benchmark, PilotTTS achieves the lowest WER of 1.50% on test-en, a CER of 0.87% on test-zh, and the highest speaker similarity on both test sets (0.862 and 0.815), outperforming systems trained on significantly larger datasets. We release the complete data pipeline recipe, pretrained weights, and code at this https URL.
>
---
#### [replaced 002] Assessing Factual Music Comprehension in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.LG**

- **简介: 该论文属于音乐理解任务，旨在解决LALMs在音乐事实性回答上的评估不足问题。提出新评估协议，通过结构化信息提取与指标评估，构建音乐理解基准。**

- **链接: [https://arxiv.org/pdf/2511.05550](https://arxiv.org/pdf/2511.05550)**

> **作者:** Daniel Chenyu Lin; Michael Freeman; John Thickstun
>
> **备注:** 16 pages; second submission
>
> **摘要:** Large audio language models (LALMs) leverage multimodal representations to generate open-ended answers to natural language queries about audio. In this paper, we (1) provide empirical evidence that assessment of LALMs using the popular MusicQA dataset fails to measure whether a model's responses about music are factually correct, and (2) develop a new protocol for assessing the music comprehension capabilities of LALMs. Specifically, we propose an evaluation protocol that prompts a LALM for factually verifiable information, and parses its open-ended response into a structured format that can be objectively assessed using Precision, Recall, and F1 scores. Using this protocol, we define a benchmark consisting of six factual information retrieval tasks defined on three diverse datasets: MusicNet, the Free Music Archive, and OverClocked ReMix. We benchmark nine recent LALMs, including frontier models like Gemini and the latest open models like Music Flamingo, and release the suite of evaluation scripts at this https URL to facilitate benchmarking of new LALMs.
>
---
#### [replaced 003] ChronosAudio: A Comprehensive Long-Audio Benchmark for Evaluating Audio-Large Language Models
- **分类: cs.SD**

- **简介: 该论文提出ChronosAudio基准，解决长音频理解问题，评估音频大语言模型在不同长度音频上的表现。**

- **链接: [https://arxiv.org/pdf/2601.04876](https://arxiv.org/pdf/2601.04876)**

> **作者:** Kaiwen Luo; Liang Lin; Yibo Zhang; Moayad Aloqaily; Jialiang Tao; Dexian Wang; Zhenhong Zhou; Junwei Zhang; Kun Wang; Li Sun; Qingsong Wen
>
> **摘要:** Although Audio Large Language Models (ALLMs) have witnessed substantial advancements, their long audio understanding capabilities remain unexplored. A plethora of benchmarks have been proposed for general audio tasks, they predominantly focus on short-form clips, leaving without a consensus on evaluating ALLMs over extended durations. This paper proposes ChronosAudio, the first multi-task benchmark tailored for long-audio understanding in ALLMs. It encompasses six major task categories and comprises 36,000 test instances totaling over 200 hours audio, stratified into short, middle, and long-form categories to comprehensively evaluate length generalization. Extensive experiments on 16 state-of-the-art models using ChronosAudio yield three critical findings: this http URL Long-Context Collapse: ALLMs exhibit a severe inability to sustain performance, with the transition from short to long contexts triggering a staggering performance degradation of over 90% in specific tasks. this http URL Attention Dilution: Performance degradation stems from a fundamental failure in maintaining temporal locality; attention mechanisms suffer from significant diffusion in later sequences. this http URL Ceiling of Mitigation: Current strategies only offer 50% recovery. These findings reveal significant challenges in long-audio, underscoring the urgent need for approaches to achieve robust, document-level audio reasoning.
>
---
#### [replaced 004] Voice "Cloning" is Style Transfer
- **分类: cs.SD; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于语音生成任务，探讨 voice cloning 技术的实际效果。研究发现，voice cloning 实质是风格迁移，导致语音特征变化，并影响人类对语音的信任度和行为。**

- **链接: [https://arxiv.org/pdf/2605.16578](https://arxiv.org/pdf/2605.16578)**

> **作者:** Kaitlyn Zhou; Federico Bianchi; Martijn Bartelds; Anna Pot; Yongchan Kwon; James Zou
>
> **摘要:** Artificially generated speech is increasingly embedded in everyday life. Voice cloning in particular enables applications where identity preservation is important, such as completing a recording, dubbing in a new language, or preserving the voices of individuals with speech loss. However, in our work, we find that despite the term, voice cloning does not faithfully ''clone'' an individual's voice. Instead, we find that widely-used voice cloning models systematically apply style transfer to source voices. As rated by human annotators, cloned voices are perceived as more authoritative, warm, customer-service-like, and human-like compared to their sources. Human annotators also report greater trust in cloned voices than source voices, and a greater willingness to disclose sensitive personal information to them. Our work furthermore shows that voice cloning leads to homogenization of speaker characteristics, as measured by reduced variance in accent, speaking rate, and the audio embedding space. Together, our results highlight a new set of limitations and risks of voice cloning technology and their potential impact on human behavior.
>
---
#### [replaced 005] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出DSA-Tokenizer，解决语音中语义与声学特征的解耦问题，通过分离生成离散语义和声学token，实现高效高质量语音合成与克隆。**

- **链接: [https://arxiv.org/pdf/2601.09239](https://arxiv.org/pdf/2601.09239)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Linqi Song
>
> **备注:** Submit to ACL ARR 2026 May
>
> **摘要:** Speech tokenizers are a key building block of fully discrete Speech this http URL tokenizers either prioritize semantic encoding,fuse semantic content with acoustic style inseparably,or achieve incomplete semantic-acoustic this http URL achieve better disentanglement,we propose DSA-Tokenizer,which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization this http URL,semantic tokens are supervised by ASR to capture linguistic content,while acoustic tokens focus on mel-spectrograms restoration to encode this http URL further introduce a hierarchical Flow Matching decoder and a joint reconstruction-context inpainting training strategy,allowing the model to support both high-fidelity reconstruction and cross-utterance voice this http URL speed up inference,we distill the DiT decoder to reduce sampling steps of inference to 4 and improve synthesis quality with GAN this http URL demonstrate that DSA-Tokenizer provides strong semantic-acoustic disentanglement,reliable controllable voice cloning,and efficient high-fidelity generation with low WER/CER.Moreover,our results suggest that disentangled tokenization provides a more effective interface for downstream large-model speech this http URL samples are avaialble at this https URL.
>
---
#### [replaced 006] TinyDéjàVu: Smaller RAM and Faster Inference with Neural Networks on MCUs for Sensor Data Streams
- **分类: cs.LG; cs.PF; cs.SD; eess.AS; eess.SP**

- **简介: 该论文属于嵌入式神经网络任务，解决在微控制器上高效运行神经网络的问题。通过优化数据流，减少RAM使用，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2512.09786](https://arxiv.org/pdf/2512.09786)**

> **作者:** Zhaolan Huang; Emmanuel Baccelli
>
> **摘要:** Examples of embedded intelligence include a wide variety of tiny neural networks used on-board wireless sensors and actuators, which are expected to continuously perform inference on time-series of the data they sense. In order to fit lifetime and energy consumption requirements when operating on battery, such hardware is exclusively based on microcontroller with as little memory as possible, e.g., 128 kB of RAM. In this context, optimizing data flows during inference across neural network layers becomes crucial. In this paper, we introduce a new framework, TinyDéjàVu, and novel algorithms we designed to drastically reduce the RAM budget required by inference using various neural network models for sensor data time-series on typical microcontroller hardware. We publish the implementation of TinyDéjàVu as open source, and we perform reproducible benchmarks on common microcontroller hardware (Arm Cortex-M). We show that TinyDéjàVu can save up to 90\% of RAM usage with equal compute latency compared to prior work (StreamiNNC) on overlapping sliding window inputs.
>
---
#### [replaced 007] VAANI: Capturing the language landscape for an inclusive digital India
- **分类: eess.AS**

- **简介: 该论文介绍VAANI项目，构建多模态数据集以覆盖印度多样语言，解决语音技术包容性不足的问题。任务为多语言语音与图像数据收集，工作包括数据采集、质量控制及发布。**

- **链接: [https://arxiv.org/pdf/2603.28714](https://arxiv.org/pdf/2603.28714)**

> **作者:** Sujith Pulikodan; Abhayjeet Singh; Agneedh Basu; Nihar Desai; Pavan Kumar J; Pranav D Bhat; Raghu Dharmaraju; Ritika Gupta; Sathvik Udupa; Saurabh Kumar; Sumit Sharma; Visruth Sanka; Dinesh Tewari; Harsh Dhand; Amrita Kamat; Sukhwinder Singh; Shikhar Vashishth; Partha Talukdar; Raj Acharya; Prasanta Kumar Ghosh
>
> **摘要:** Voice based technologies have the potential to bridge digital accessibility gaps; however, existing datasets fail to capture the linguistic and regional diversity of Indic languages. We present Project VAANI, a large scale multimodal dataset designed to represent India's linguistic landscape across 165 districts. Speech data is collected using image based prompts to elicit spontaneous responses, while images are curated through a separate pipeline covering diverse themes across regions. The dataset undergoes a rigorous multi stage quality control process, combining automated and manual evaluation to ensure high audio quality and transcription accuracy. We release approximately 289K images, 31,255 hours of speech, and 2,043 hours of transcribed audio spanning 105 languages from 28 states and 3 union territories. Many of these languages are represented at this scale for the first time, making VAANI a foundational resource for inclusive speech technology. The dataset enables the development of robust, multilingual, and multimodal models, and supports research in speech recognition, language understanding, and cross-modal learning for underrepresented languages.
>
---
#### [replaced 008] FSD50K-Solo: Automated Curation of Single-Source Sound Events
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于音频数据集构建任务，旨在解决多源声音样本影响数据质量的问题。通过生成模型和分类器自动筛选单源声音，构建高质量数据集FSD50K-Solo。**

- **链接: [https://arxiv.org/pdf/2605.13931](https://arxiv.org/pdf/2605.13931)**

> **作者:** Ningyuan Yang; Sile Yin; Li-Chia Yang; Bryce Irvin; Xiao Quan; Marko Stamenovic; Shuo Zhang
>
> **备注:** Accepted to EUSIPCO 2026. 5 pages, 3 figures
>
> **摘要:** High-quality training datasets are essential for the performance of neural networks. However, the audio domain still lacks a large-scale, strongly-labeled, and single-source sound event dataset. The FSD50K dataset, despite being relatively large and open, contains a considerable fraction of multi-source samples where background interference or overlapping events could limit the usefulness of the data. To address this challenge, we introduce a data curation framework designed for large-scale open audio corpora. Our approach leverages a generative diffusion model to synthesize clean single-class events to construct controlled noisy mixtures for supervision. We subsequently employ a pre-trained audio encoder coupled with a discriminative classifier to automatically identify and filter out multi-source samples. Experiments show that our framework achieves strong performance on a human expert-curated test set. Finally, we release FSD50K-Solo, a model-curated subset of FSD50K containing single-source audio samples identified by our method. Beyond FSD50K, our method establishes a scalable paradigm for curating open source audio corpora.
>
---
#### [replaced 009] Exploration of Perceptual Speech Features for Clinical Decision-Support in Mental Health Care
- **分类: cs.AI; cs.CL; cs.SD**

- **简介: 该论文属于心理健康评估任务，旨在通过语音特征分析支持临床决策。研究提取了语音的声学和语言特征，结合机器学习方法，探索其与抑郁、焦虑和ADHD症状的关系。**

- **链接: [https://arxiv.org/pdf/2605.24678](https://arxiv.org/pdf/2605.24678)**

> **作者:** Vassilis Lyberatos; Edmund G. Dervakos; Eleni Adamidi; Athanasios Voulodimos; Giorgos Stamou
>
> **备注:** Accepted to CLPsych 2026, part of ACL 2026
>
> **摘要:** Speech and language technologies offer valuable opportunities for supporting mental health assessment through objective and interpretable cues. We present a systematic feature-based analysis framework leveraging perceptually grounded acoustic and linguistic characteristics, including prosody, vocal quality, semantic coherence, syntactic structure, and sarcasm. Using statistical analysis and interpretable machine learning (XGBoost with SHAP and LIME), we examine associations between speech features and validated symptom measures of depression, anxiety, and ADHD. Evaluated on both controlled benchmark datasets (StressID, DAIC-WOZ, Androids, EATD) and a real-world clinical dataset, the framework reveals stable and consistent relationships between symptom severity and vocal irregularities (e.g., shimmer, jitter), lexical-syntactic patterns, and affective tone. An ablation study conducted across all datasets further identifies the most informative feature groups. This work explores a transparent and clinically interpretable approach to speech-based mental health analysis.
>
---
#### [replaced 010] Semantic-Aware Interpretable Multimodal Music Auto-Tagging
- **分类: cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于音乐自动标记任务，旨在提升模型的可解释性。通过融合多模态特征并进行语义聚类，提高 tagging 的透明度与用户信任。**

- **链接: [https://arxiv.org/pdf/2505.17233](https://arxiv.org/pdf/2505.17233)**

> **作者:** Andreas Patakis; Vassilis Lyberatos; Spyridon Kantarelis; Edmund Dervakos; Giorgos Stamou
>
> **备注:** Accepted at Interspeech 2025
>
> **摘要:** Music auto-tagging is essential for organizing and discovering music in extensive digital libraries. While foundation models achieve exceptional performance in this domain, their outputs often lack interpretability, limiting trust and usability for researchers and end-users alike. In this work, we present an interpretable framework for music auto-tagging that leverages groups of musically meaningful multimodal features, derived from signal processing, deep learning, ontology engineering, and natural language processing. To enhance interpretability, we cluster features semantically and employ an expectation maximization algorithm, assigning distinct weights to each group based on its contribution to the tagging process. Our method achieves competitive tagging performance while offering a deeper understanding of the decision-making process, paving the way for more transparent and user-centric music tagging systems.
>
---
#### [replaced 011] Addressing Pitfalls in Auditing Practices of Automatic Speech Recognition Technologies: A Case Study of People with Aphasia
- **分类: cs.CY; cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音识别审计任务，旨在解决ASR系统对失语症患者不公平的问题。通过识别审计中的三个误区，提出改进框架并验证了失语症患者的性能下降。**

- **链接: [https://arxiv.org/pdf/2506.08846](https://arxiv.org/pdf/2506.08846)**

> **作者:** Katelyn Xiaoying Mei; Anna Seo Gyeong Choi; Hilke Schellmann; Mona Sloane; Allison Koenecke
>
> **备注:** Published at the Proceedings of The 2026 ACM Conference on Fairness, Accountability, and Transparency (FAccT '26)
>
> **摘要:** Automatic Speech Recognition (ASR) systems' growing use warrants robust auditing approaches to ensure equitable transcription quality, especially for people with speech disorders like aphasia who disproportionately depend on ASR. While academic and industry audits have revealed performance disparities across user populations, standard auditing practices often overlook nuances that risk masking harm to marginalized groups. We identify three common pitfalls in standard ASR audits: (1) adhering to one method of text standardization, which can mask variance in ASR performance and ignore the standardization preferences of marginalized communities; (2) displaying high-level demographic findings without considering performance disparities by nuanced intersectional subgroups, or conditioning on relevant acoustic properties; and (3) reporting only one gold-standard metric (Word Error Rate), which inadequately quantifies common generative AI errors like hallucinations. We propose a holistic auditing framework addressing these pitfalls, and in a case study of six popular ASR systems, find consistently worse ASR performance for speakers with aphasia relative to a control group. We call on practitioners to implement these robust, community-driven ASR auditing practices better suited for the rapidly changing ASR landscape.
>
---
