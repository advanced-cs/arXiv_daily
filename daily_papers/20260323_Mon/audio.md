# 音频 cs.SD;  eess.AS

- **最新发布 9 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] BioDCASE 2026 Challenge Baseline for Cross-Domain Mosquito Species Classification
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于跨领域蚊种分类任务，旨在解决音频监测中物种识别困难的问题。通过构建基线系统，探索多源生物声学数据的跨域泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.20118](https://arxiv.org/pdf/2603.20118)**

> **作者:** Yuanbo Hou; Vanja Zdravkovic; Marianne Sinka; Yunpeng Li; Wenwu Wang; Mark D. Plumbley; Kathy Willis; Stephen Roberts
>
> **备注:** BioDCASE 2026 CD-MSC Baseline, source code and models: this https URL
>
> **摘要:** Mosquito-borne diseases affect more than one billion people each year and cause close to one million deaths. Traditional surveillance methods rely on traps and manual identification that are slow, labor-intensive, and difficult to scale. Audio-based mosquito monitoring offers a non-destructive, lower-cost, and more scalable complement to trap-based surveillance, but reliable species classification remains difficult under real-world recording conditions. Mosquito flight tones are narrow-band, often low in signal-to-noise ratio, and easily masked by background noise, and recordings for several epidemiologically relevant species remain limited, creating pronounced class imbalance. Variation across devices, environments, and collection protocols further increases the difficulty of robust classification. Such variation can cause models to rely on domain-specific recording artefacts rather than species-relevant acoustic cues, which makes transfer to new acquisition settings difficult. The BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC) challenge is designed around this deployment problem by evaluating performance on both seen and unseen domains. This paper presents the official baseline system and evaluation pipeline as a simple, fully reproducible reference for the CD-MSC challenge task. The baseline uses log-mel features and a multitemporal resolution convolutional neural network (MTRCNN) with species and auxiliary domain outputs, together with complete training and test scripts. The baseline system performs strongly on seen domains but degrades markedly on unseen domains, showing that cross-domain generalisation, rather than within-domain recognition, is the central challenge for practical mosquito species classification from multi-source bioacoustic recordings.
>
---
#### [new 002] Audio Avatar Fingerprinting: An Approach for Authorized Use of Voice Cloning in the Era of Synthetic Audio
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成验证任务，旨在解决合成音频的授权使用问题。通过改进语音验证模型并构建新数据集，实现对合成语音是否合法的检测。**

- **链接: [https://arxiv.org/pdf/2603.20165](https://arxiv.org/pdf/2603.20165)**

> **作者:** Candice R. Gerstner
>
> **摘要:** With the advancements in AI speech synthesis, it is easier than ever before to generate realistic audio in a target voice. One only needs a few seconds of reference audio from the target, quite literally putting words in the target person's mouth. This imposes a new set of forensics-related challenges on speech-based authentication systems, videoconferencing, and audio-visual broadcasting platforms, where we want to detect synthetic speech. At the same time, leveraging AI speech synthesis can enhance the different modes of communication through features such as low-bandwidth communication and audio enhancements - leading to ever-increasing legitimate use-cases of synthetic audio. In this case, we want to verify if the synthesized voice is actually spoken by the user. This will require a mechanism to verify whether a given synthetic audio is driven by an authorized identity, or not. We term this task audio avatar fingerprinting. As a step towards audio forensics in these new and emerging situations, we analyze and extend an off-the-shelf speaker verification model developed outside of forensics context for the task of fake speech detection and audio avatar fingerprinting, the first experimentation of its kind. Furthermore, we observe that no existing dataset allows for the novel task of verifying the authorized use of synthetic audio - a limitation which we address by introducing a new speech forensics dataset for this novel task.
>
---
#### [new 003] Listen First, Then Answer: Timestamp-Grounded Speech Reasoning
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于多模态推理任务，旨在解决音频语言模型推理缺乏音频时戳依据的问题。通过引入基于强化学习的时戳标注策略，增强模型对音频内容的依赖，提升推理准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.19468](https://arxiv.org/pdf/2603.19468)**

> **作者:** Jihoon Jeong; Pooneh Mousavi; Mirco Ravanelli; Cem Subakan
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Large audio-language models (LALMs) can generate reasoning chains for their predictions, but it remains unclear whether these reasoning chains remain grounded in the input audio. In this paper, we propose an RL-based strategy that grounds the reasoning outputs of LALMs with explicit timestamp annotations referring to relevant segments of the audio signal. Our analysis shows that timestamp grounding leads the model to attend more strongly to audio tokens during reasoning generation. Experiments on four speech-based benchmark datasets demonstrate that our approach improves performance compared to both zero-shot reasoning and fine-tuning without timestamp grounding. Additionally, grounding amplifies desirable reasoning behaviors, such as region exploration, audiology verification, and consistency, underscoring the importance of grounding mechanisms for faithful multimodal reasoning.
>
---
#### [new 004] Borderless Long Speech Synthesis
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出Borderless Long Speech Synthesis框架，解决长音频合成中多说话人、情感变化和环境多样性问题，通过统一能力集和结构化标注实现无缝语音生成。**

- **链接: [https://arxiv.org/pdf/2603.19798](https://arxiv.org/pdf/2603.19798)**

> **作者:** Xingchen Song; Di Wu; Dinghao Zhou; Pengyu Cheng; Hongwu Ding; Yunchao He; Jie Wang; Shengfan Shen; Sixiang Lv; Lichun Fan; Hang Su; Yifeng Wang; Shuai Wang; Meng Meng; Jian Luan
>
> **摘要:** Most existing text-to-speech (TTS) systems either synthesize speech sentence by sentence and stitch the results together, or drive synthesis from plain-text dialogues alone. Both approaches leave models with little understanding of global context or paralinguistic cues, making it hard to capture real-world phenomena such as multi-speaker interactions (interruptions, overlapping speech), evolving emotional arcs, and varied acoustic environments. We introduce the Borderless Long Speech Synthesis framework for agent-centric, borderless long audio synthesis. Rather than targeting a single narrow task, the system is designed as a unified capability set spanning VoiceDesigner, multi-speaker synthesis, Instruct TTS, and long-form text synthesis. On the data side, we propose a "Labeling over filtering/cleaning" strategy and design a top-down, multi-level annotation schema we call Global-Sentence-Token. On the model side, we adopt a backbone with a continuous tokenizer and add Chain-of-Thought (CoT) reasoning together with Dimension Dropout, both of which markedly improve instruction following under complex conditions. We further show that the system is Native Agentic by design: the hierarchical annotation doubles as a Structured Semantic Interface between the LLM Agent and the synthesis engine, creating a layered control protocol stack that spans from scene semantics down to phonetic detail. Text thereby becomes an information-complete, wide-band control channel, enabling a front-end LLM to convert inputs of any modality into structured generation commands, extending the paradigm from Text2Speech to borderless long speech synthesis.
>
---
#### [new 005] Gesture2Speech: How Far Can Hand Movements Shape Expressive Speech?
- **分类: eess.AS; cs.AI; cs.MM**

- **简介: 该论文属于语音合成任务，旨在解决手部动作如何影响语音韵律的问题。提出Gesture2Speech框架，通过手势信息调节合成语音的韵律，提升语音自然度与手势同步性。**

- **链接: [https://arxiv.org/pdf/2603.19831](https://arxiv.org/pdf/2603.19831)**

> **作者:** Lokesh Kumar; Nirmesh Shah; Ashishkumar P. Gudmalwar; Pankaj Wasnik
>
> **备注:** Accepted at The 2nd International Workshop on Bodily Expressed Emotion Understanding (BEEU) at AAAI 2026 [non-archival]
>
> **摘要:** Human communication seamlessly integrates speech and bodily motion, where hand gestures naturally complement vocal prosody to express intent, emotion, and emphasis. While recent text-to-speech (TTS) systems have begun incorporating multimodal cues such as facial expressions or lip movements, the role of hand gestures in shaping prosody remains largely underexplored. We propose a novel multimodal TTS framework, Gesture2Speech, that leverages visual gesture cues to modulate prosody in synthesized speech. Motivated by the observation that confident and expressive speakers coordinate gestures with vocal prosody, we introduce a multimodal Mixture-of-Experts (MoE) architecture that dynamically fuses linguistic content and gesture features within a dedicated style extraction module. The fused representation conditions an LLM-based speech decoder, enabling prosodic modulation that is temporally aligned with hand movements. We further design a gesture-speech alignment loss that explicitly models their temporal correspondence to ensure fine-grained synchrony between gestures and prosodic contours. Evaluations on the PATS dataset show that Gesture2Speech outperforms state-of-the-art baselines in both speech naturalness and gesture-speech synchrony. To the best of our knowledge, this is the first work to utilize hand gesture cues for prosody control in neural speech synthesis. Demo samples are available at this https URL
>
---
#### [new 006] CAF-Score: Calibrating CLAP with LALMs for Reference-free Audio Captioning Evaluation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于音频描述评估任务，解决无参考评价难题。提出CAF-Score，结合CLAP与LALM优势，提升语法和细节检测能力。**

- **链接: [https://arxiv.org/pdf/2603.19615](https://arxiv.org/pdf/2603.19615)**

> **作者:** Insung Lee; Taeyoung Jeong; Haejun Yoo; Du-Seong Chang; Myoung-Wan Koo
>
> **备注:** A condensed version of this work has been submitted to Interspeech 2026. Section 10 is an extended analysis added in this version
>
> **摘要:** While Large Audio-Language Models (LALMs) have advanced audio captioning, robust evaluation remains difficult. Reference-based metrics are expensive and often fail to assess acoustic fidelity, while Contrastive Language-Audio Pretraining (CLAP)-based approaches frequently overlook syntactic errors and fine-grained details. We propose CAF-Score, a reference-free metric that calibrates CLAP's coarse-grained semantic alignment with the fine-grained comprehension and syntactic awareness of LALMs. By combining contrastive audio-text embeddings with LALM reasoning, CAF-Score effectively detects syntactic inconsistencies and subtle hallucinations. Experiments on the BRACE benchmark demonstrate that our approach achieves the highest correlation with human judgments, even outperforming reference-based baselines in challenging scenarios. These results highlight the efficacy of CAF-Score for reference-free audio captioning evaluation. Code and results are available at this https URL.
>
---
#### [new 007] Plug-and-Steer: Decoupling Separation and Selection in Audio-Visual Target Speaker Extraction
- **分类: eess.AS; cs.MM; cs.SD**

- **简介: 该论文属于音频-视觉目标说话人提取任务，旨在解决传统系统融合过深导致性能受限的问题。提出Plug-and-Steer方法，分离分离与选择过程，提升效果。**

- **链接: [https://arxiv.org/pdf/2603.19697](https://arxiv.org/pdf/2603.19697)**

> **作者:** Doyeop Kwak; Suyeon Lee; Joon Son Chung
>
> **备注:** Submitted to Interspeech 2026; demo available this https URL
>
> **摘要:** The goal of this paper is to provide a new perspective on audio-visual target speaker extraction (AV-TSE) by decoupling the separation and target selection. Conventional AV-TSE systems typically integrate audio and visual features deeply to re-learn the entire separation process, which can act as a fidelity ceiling due to the noisy nature of in-the-wild audio-visual datasets. To address this, we propose Plug-and-Steer, which assigns high-fidelity separation to a frozen audio-only backbone and limits the role of visual modality strictly to target selection. We introduce the Latent Steering Matrix (LSM), a minimalist linear transformation that re-routes latent features within the backbone to anchor the target speaker to a designated channel. Experiments across four representative architectures show that our method effectively preserves the acoustic priors of diverse backbones, achieving perceptual quality comparable to the original backbones. Audio samples are available at: this https URL
>
---
#### [new 008] MOSS-TTSD: Text to Spoken Dialogue Generation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音合成任务，解决多轮对话生成中的连贯性和稳定性问题。提出MOSS-TTSD模型，支持多语言、多说话人及长时对话生成，并引入TTSD-eval评估框架。**

- **链接: [https://arxiv.org/pdf/2603.19739](https://arxiv.org/pdf/2603.19739)**

> **作者:** Yuqian Zhang; Donghua Yu; Zhengyuan Lin; Botian Jiang; Mingshu Chen; Yaozhou Jiang; Yiwei Zhao; Yiyang Zhang; Yucheng Yuan; Hanfu Chen; Kexin Huang; Jun Zhan; Cheng Chang; Zhaoye Fei; Shimin Li; Xiaogui Yang; Qinyuan Cheng; Xipeng Qiu
>
> **摘要:** Spoken dialogue generation is crucial for applications like podcasts, dynamic commentary, and entertainment content, but poses significant challenges compared to single-utterance text-to-speech (TTS). Key requirements include accurate turn-taking, cross-turn acoustic consistency, and long-form stability, which current models often fail to address due to a lack of dialogue context modeling. To bridge this gap, we present MOSS-TTSD, a spoken dialogue synthesis model designed for expressive, multi-party conversational speech across multiple languages. With enhanced long-context modeling, MOSS-TTSD generates long-form spoken conversations from dialogue scripts with explicit speaker tags, supporting up to 60 minutes of single-pass synthesis, multi-party dialogue with up to 5 speakers, and zero-shot voice cloning from a short reference audio clip. The model supports various mainstream languages, including English and Chinese, and is adapted to several long-form scenarios. Additionally, to address limitations of existing evaluation methods, we propose TTSD-eval, an objective evaluation framework based on forced alignment that measures speaker attribution accuracy and speaker similarity without relying on speaker diarization tools. Both objective and subjective evaluation results show that MOSS-TTSD surpasses strong open-source and proprietary baselines in dialogue synthesis.
>
---
#### [new 009] FoleyDirector: Fine-Grained Temporal Steering for Video-to-Audio Generation via Structured Scripts
- **分类: cs.SD; cs.CV**

- **简介: 该论文属于视频到音频生成任务，解决多事件场景下时间控制不足的问题。提出FoleyDirector框架，通过结构化脚本实现精准时间引导，提升可控性与音质。**

- **链接: [https://arxiv.org/pdf/2603.19857](https://arxiv.org/pdf/2603.19857)**

> **作者:** You Li; Dewei Zhou; Fan Ma; Fu Li; Dongliang He; Yi Yang
>
> **备注:** Accepted at IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2026, 18 pages
>
> **摘要:** Recent Video-to-Audio (V2A) methods have achieved remarkable progress, enabling the synthesis of realistic, high-quality audio. However, they struggle with fine-grained temporal control in multi-event scenarios or when visual cues are insufficient, such as small regions, off-screen sounds, or occluded or partially visible objects. In this paper, we propose FoleyDirector, a framework that, for the first time, enables precise temporal guidance in DiT-based V2A generation while preserving the base model's audio quality and allowing seamless switching between V2A generation and temporally controlled synthesis. FoleyDirector introduces Structured Temporal Scripts (STS), a set of captions corresponding to short temporal segments, to provide richer temporal information. These features are integrated via the Script-Guided Temporal Fusion Module, which employs Temporal Script Attention to fuse STS features coherently. To handle complex multi-event scenarios, we further propose Bi-Frame Sound Synthesis, enabling parallel in-frame and out-of-frame audio generation and improving controllability. To support training and evaluation, we construct the DirectorSound dataset and introduce VGGSoundDirector and DirectorBench. Experiments demonstrate that FoleyDirector substantially enhances temporal controllability while maintaining high audio fidelity, empowering users to act as Foley directors and advancing V2A toward more expressive and controllable generation.
>
---
## 更新

#### [replaced 001] VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.CV; cs.SD**

- **简介: 该论文属于视频音频生成任务，旨在统一视频到声音和视觉文本到语音生成。提出VSSFlow框架，通过联合学习解决两者差异，提升性能。**

- **链接: [https://arxiv.org/pdf/2509.24773](https://arxiv.org/pdf/2509.24773)**

> **作者:** Xin Cheng; Yuyue Wang; Xihua Wang; Yihan Wu; Kaisi Guan; Yijing Chen; Peng Zhang; Xiaojiang Liu; Meng Cao; Ruihua Song
>
> **备注:** Paper Under Review
>
> **摘要:** Video-conditioned audio generation, including Video-to-Sound (V2S) and Visual Text-to-Speech (VisualTTS), has traditionally been treated as distinct tasks, leaving the potential for a unified generative framework largely underexplored. In this paper, we bridge this gap with VSSFlow, a unified flow-matching framework that seamlessly solve both problems. To effectively handle multiple input signals within a Diffusion Transformer (DiT) architecture, we propose a disentangled condition aggregation mechanism leveraging distinct intrinsic properties of attention layers: cross-attention for semantic conditions, and self-attention for temporally-intensive conditions. Besides, contrary to the prevailing belief that joint training for the two tasks leads to performance degradation, we demonstrate that VSSFlow maintains superior performance during end-to-end joint learning process. Furthermore, we use a straightforward feature-level data synthesis method, demonstrating that our framework provides a robust foundation that easily adapts to joint sound and speech generation using synthetic data. Extensive experiments on V2S, VisualTTS and joint generation benchmarks show that VSSFlow effectively unifies these tasks and surpasses state-of-the-art domain-specific baselines, underscoring the critical potential of unified generative models. Project page: this https URL
>
---
#### [replaced 002] DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models
- **分类: cs.AI; cs.SD; eess.AS**

- **简介: 该论文属于音频语言模型评估任务，旨在解决模型是否真正理解声音信号的问题。通过构建基准DEAF，设计评估框架和诊断指标，分析模型对文本和声音的依赖程度。**

- **链接: [https://arxiv.org/pdf/2603.18048](https://arxiv.org/pdf/2603.18048)**

> **作者:** Jiaqi Xiong; Yunjia Qi; Qi Cao; Yu Zheng; Yutong Zhang; Ziteng Wang; Ruofan Liao; Weisheng Xu; Sichen Liu
>
> **备注:** 14 pages,6 figures
>
> **摘要:** Recent Audio Multimodal Large Language Models (Audio MLLMs) demonstrate impressive performance on speech benchmarks, yet it remains unclear whether these models genuinely process acoustic signals or rely on text-based semantic inference. To systematically study this question, we introduce DEAF (Diagnostic Evaluation of Acoustic Faithfulness), a benchmark of over 2,700 conflict stimuli spanning three acoustic dimensions: emotional prosody, background sounds, and speaker identity. Then, we design a controlled multi-level evaluation framework that progressively increases textual influence, ranging from semantic conflicts in the content to misleading prompts and their combination, allowing us to disentangle content-driven bias from prompt-induced sycophancy. We further introduce diagnostic metrics to quantify model reliance on textual cues over acoustic signals. Our evaluation of seven Audio MLLMs reveals a consistent pattern of text dominance: models are sensitive to acoustic variations, yet predictions are predominantly driven by textual inputs, revealing a gap between high performance on standard speech benchmarks and genuine acoustic understanding.
>
---
#### [replaced 003] Community-Informed AI Models for Police Accountability
- **分类: cs.CY; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于政府问责任务，旨在通过AI技术提升警察与公众互动的透明度。研究结合社区意见，开发多视角AI工具，分析警察执法视频，以增强民主治理的公正性。**

- **链接: [https://arxiv.org/pdf/2402.01703](https://arxiv.org/pdf/2402.01703)**

> **作者:** Benjamin A.T. Grahama; Lauren Brown; Georgios Chochlakis; Morteza Dehghani; Raquel Delerme; Brittany Friedman; Ellie Graeden; Preni Golazizian; Rajat Hebbar; Parsa Hejabi; Aditya Kommineni; Mayagüez Salinas; Michael Sierra-Arévalo; Jackson Trager; Nicholas Weller; Shrikanth Narayanan
>
> **备注:** 33 pages, 4 figures, 2 tables
>
> **摘要:** Face-to-face interactions between police officers and the public affect both individual well-being and democratic legitimacy. Many government-public interactions are captured on video, including interactions between police officers and drivers captured on bodyworn cameras (BWCs). New advances in AI technology enable these interactions to be analyzed at scale, opening promising avenues for improving government transparency and accountability. However, for AI to serve democratic governance effectively, models must be designed to include the preferences and perspectives of the governed. This article proposes a community-informed, approach to developing multi-perspective AI tools for government accountability. We illustrate our approach by describing the research project through which the approach was inductively developed: an effort to build AI tools to analyze BWC footage of traffic stops conducted by the Los Angeles Police Department. We focus on the role of social scientists as members of multidisciplinary teams responsible for integrating the perspectives of diverse stakeholders into the development of AI tools in the domain of police -- and government -- accountability.
>
---
#### [replaced 004] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer
- **分类: cs.SD; cs.CV; cs.LG; cs.MM; eess.AS**

- **简介: 该论文属于视频到音频生成任务，旨在解决现有方法依赖文本描述导致的语义模糊和声学细节不足问题。提出AC-Foley模型，通过参考音频实现精准音效合成与控制。**

- **链接: [https://arxiv.org/pdf/2603.15597](https://arxiv.org/pdf/2603.15597)**

> **作者:** Pengjun Fang; Yingqing He; Yazhou Xing; Qifeng Chen; Ser-Nam Lim; Harry Yang
>
> **备注:** Accepted at ICLR 2026. 15 pages, 5 figures, add project webpage
>
> **摘要:** Existing video-to-audio (V2A) generation methods predominantly rely on text prompts alongside visual information to synthesize audio. However, two critical bottlenecks persist: semantic granularity gaps in training data, such as conflating acoustically distinct sounds under coarse labels, and textual ambiguity in describing micro-acoustic features. These bottlenecks make it difficult to perform fine-grained sound synthesis using text-controlled modes. To address these limitations, we propose AC-Foley, an audio-conditioned V2A model that directly leverages reference audio to achieve precise and fine-grained control over generated sounds. This approach enables fine-grained sound synthesis, timbre transfer, zero-shot sound generation, and improved audio quality. By directly conditioning on audio signals, our approach bypasses the semantic ambiguities of text descriptions while enabling precise manipulation of acoustic attributes. Empirically, AC-Foley achieves state-of-the-art performance for Foley generation when conditioned on reference audio, while remaining competitive with state-of-the-art video-to-audio methods even without audio conditioning. Code and demo are available at: this https URL
>
---
#### [replaced 005] MOSS-TTS Technical Report
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文属于语音生成任务，旨在解决高质量、多语言语音合成问题。提出MOSS-TTS模型，采用离散音频标记和自回归建模，支持零样本语音克隆与精确控制。**

- **链接: [https://arxiv.org/pdf/2603.18090](https://arxiv.org/pdf/2603.18090)**

> **作者:** Yitian Gong; Botian Jiang; Yiwei Zhao; Yucheng Yuan; Kuangwei Chen; Yaozhou Jiang; Cheng Chang; Dong Hong; Mingshu Chen; Ruixiao Li; Yiyang Zhang; Yang Gao; Hanfu Chen; Ke Chen; Songlin Wang; Xiaogui Yang; Yuqian Zhang; Kexin Huang; ZhengYuan Lin; Kang Yu; Ziqi Chen; Jin Wang; Zhaoye Fei; Qinyuan Cheng; Shimin Li; Xipeng Qiu
>
> **备注:** Project page: this https URL
>
> **摘要:** This technical report presents MOSS-TTS, a speech generation foundation model built on a scalable recipe: discrete audio tokens, autoregressive modeling, and large-scale pretraining. Built on MOSS-Audio-Tokenizer, a causal Transformer tokenizer that compresses 24 kHz audio to 12.5 fps with variable-bitrate RVQ and unified semantic-acoustic representations, we release two complementary generators: MOSS-TTS, which emphasizes structural simplicity, scalability, and long-context/control-oriented deployment, and MOSS-TTS-Local-Transformer, which introduces a frame-local autoregressive module for higher modeling efficiency, stronger speaker preservation, and a shorter time to first audio. Across multilingual and open-domain settings, MOSS-TTS supports zero-shot voice cloning, token-level duration control, phoneme-/pinyin-level pronunciation control, smooth code-switching, and stable long-form generation. This report summarizes the design, training recipe, and empirical characteristics of the released models.
>
---
