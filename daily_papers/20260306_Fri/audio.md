# 音频 cs.SD;  eess.AS

- **最新发布 20 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] PolyBench: A Benchmark for Compositional Reasoning in Polyphonic Audio
- **分类: eess.AS; cs.SD**

- **简介: 该论文提出PolyBench，用于评估音频中的组合推理能力。针对多声部音频中多个事件共存的问题，设计了五个评估子集，发现现有模型在此任务上表现不佳。**

- **链接: [https://arxiv.org/pdf/2603.05128](https://arxiv.org/pdf/2603.05128)**

> **作者:** Yuanjian Chen; Yang Xiao; Han Yin; Xubo Liu; Jinjie Huang; Ting Dang
>
> **摘要:** Large Audio Language Models (LALMs) are increasingly capable of reasoning over audio. However, existing benchmarks provide limited coverage of reasoning in polyphonic audio, where multiple sound events co-occur and induce compositional structure. In this work, we introduce PolyBench, a benchmark designed to evaluate compositional reasoning in polyphonic audio. PolyBench comprises five evaluation subsets covering counting, classification, detection, concurrency, and duration estimation, requiring reasoning over multiple concurrent events and their relations. Evaluation of state-of-the-art LALMs reveals consistent performance degradation in polyphonic audio, indicating a fundamental bottleneck in current LALMs.
>
---
#### [new 002] The First Environmental Sound Deepfake Detection Challenge: Benchmarking Robustness, Evaluation, and Insights
- **分类: cs.SD**

- **简介: 该论文属于环境声音深度伪造检测任务，旨在解决虚假音频内容识别问题。工作包括构建数据集、制定评估标准、分析最佳系统，并探讨未来研究方向。**

- **链接: [https://arxiv.org/pdf/2603.04865](https://arxiv.org/pdf/2603.04865)**

> **作者:** Han Yin; Yang Xiao; Rohan Kumar Das; Jisheng Bai; Ting Dang
>
> **摘要:** Recent progress in audio generation has made it increasingly easy to create highly realistic environmental soundscapes, which can be misused to produce deceptive content, such as fake alarms, gunshots, and crowd sounds, raising concerns for public safety and trust. While deepfake detection for speech and singing voice has been extensively studied, environmental sound deepfake detection (ESDD) remains underexplored. To advance ESDD, the first edition of the ESDD challenge was launched, attracting 97 registered teams and receiving 1,748 valid submissions. This paper presents the task formulation, dataset construction, evaluation protocols, baseline systems, and key insights from the challenge results. Furthermore, we analyze common architectural choices and training strategies among top-performing systems. Finally, we discuss potential future research directions for ESDD, outlining key opportunities and open problems to guide subsequent studies in this field.
>
---
#### [new 003] Focus Then Listen: Exploring Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models
- **分类: cs.SD**

- **简介: 该论文属于音频处理任务，旨在提升大音频语言模型在噪声环境下的鲁棒性。提出FTL方法，在不微调模型的情况下增强音频信号，改善下游任务性能。**

- **链接: [https://arxiv.org/pdf/2603.04862](https://arxiv.org/pdf/2603.04862)**

> **作者:** Han Yin; Yang Xiao; Younghoo Kwon; Ting Dang; Jung-Woo Choi
>
> **摘要:** Large audio language models (LALMs) are a class of foundation models for audio understanding. Existing LALMs tend to degrade significantly in real-world noisy acoustic conditions where speech and non-speech sounds interfere. While noise-aware fine-tuning can improve robustness, it requires task-specific noisy data and expensive retraining, limiting scalability. To address this issue, we propose Focus-Then-Listen (FTL), a plug-and-play audio enhancer that improves LALMs' noise robustness. Specifically, FTL first separates the input waveform into speech and non-speech, and a modality router is applied to predict the target audio modality (e.g., speech) based on the user's instruction. Finally, a modality-aware fusion block generates a task-adaptive enhanced signal for improved downstream perception and reasoning. Experiments across multiple LALMs and tasks show that FTL improves performance across different noise levels without fine-tuning on LALMs.
>
---
#### [new 004] WhisperAlign: Word-Boundary-Aware ASR and WhisperX-Anchored Pyannote Diarization for Long-Form Bengali Speech
- **分类: cs.SD; cs.LG**

- **简介: 该论文针对孟加拉语长语音的语音识别与说话人分离任务，提出基于WhisperAlign和WhisperX的解决方案，优化音频分块与模型微调，降低错误率。**

- **链接: [https://arxiv.org/pdf/2603.04809](https://arxiv.org/pdf/2603.04809)**

> **作者:** Aurchi Chowdhury; Rubaiyat -E-Zaman; Sk. Ashrafuzzaman Nafees
>
> **摘要:** This paper presents our solution for the DL Sprint 4.0, addressing the dual challenges of Bengali Long-Form Speech Recognition (Task 1) and Speaker Diarization (Task 2). Processing long-form, multi-speaker Bengali audio introduces significant hurdles in voice activity detection, overlapping speech, and context preservation. To solve the long-form transcription challenge, we implemented a robust audio chunking strategy utilizing whisper-timestamped, allowing us to feed precise, context-aware segments into our fine-tuned acoustic model for high-accuracy transcription. For the diarization task, we developed an integrated pipeline leveraging this http URL and WhisperX. A key contribution of our approach is the domain-specific fine-tuning of the Pyannote segmentation model on the competition dataset. This adaptation allowed the model to better capture the nuances of Bengali conversational dynamics and accurately resolve complex, overlapping speaker boundaries. Our methodology demonstrates that applying intelligent timestamped chunking to ASR and targeted segmentation fine-tuning to diarization significantly drives down Word Error Rate (WER) and Diarization Error Rate (DER), in low-resource settings.
>
---
#### [new 005] Temporal Pooling Strategies for Training-Free Anomalous Sound Detection with Self-Supervised Audio Embeddings
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于训练-free 异常声音检测任务，旨在提升基于预训练音频嵌入模型的检测效果。通过提出新的时间池化策略，如相对偏差池化和混合池化，改进异常检测性能。**

- **链接: [https://arxiv.org/pdf/2603.04605](https://arxiv.org/pdf/2603.04605)**

> **作者:** Kevin Wilkinghoff; Sarthak Yadav; Zheng-Hua Tan
>
> **摘要:** Training-free anomalous sound detection (ASD) based on pre-trained audio embedding models has recently garnered significant attention, as it enables the detection of anomalous sounds using only normal reference data while offering improved robustness under domain shifts. However, existing embedding-based approaches almost exclusively rely on temporal mean pooling, while alternative pooling strategies have so far only been explored for spectrogram-based representations. Consequently, the role of temporal pooling in training-free ASD with pre-trained embeddings remains insufficiently understood. In this paper, we present a systematic evaluation of temporal pooling strategies across multiple state-of-the-art audio embedding models. We propose relative deviation pooling (RDP), an adaptive pooling method that emphasizes informative temporal deviations, and introduce a hybrid pooling strategy that combines RDP with generalized mean pooling. Experiments on five benchmark datasets demonstrate that the proposed methods consistently outperform mean pooling and achieve state-of-the-art performance for training-free ASD, including results that surpass all previously reported trained systems and ensembles on the DCASE2025 ASD dataset.
>
---
#### [new 006] Building Enterprise Realtime Voice Agents from Scratch: A Technical Tutorial
- **分类: cs.SD**

- **简介: 本文介绍如何从零构建企业级实时语音代理系统，解决传统模型实时性不足的问题。通过流式处理和组件流水线，实现低延迟的语音交互。**

- **链接: [https://arxiv.org/pdf/2603.05413](https://arxiv.org/pdf/2603.05413)**

> **作者:** Jielin Qiu; Zixiang Chen; Liangwei Yang; Ming Zhu; Zhiwei Liu; Juntao Tan; Wenting Zhao; Rithesh Murthy; Roshan Ram; Akshara Prabhakar; Shelby Heinecke; Caiming Xiong; Silvio Savarese; Huan Wang
>
> **摘要:** We present a technical tutorial for building enterprise-grade realtime voice agents from first principles. While over 25 open-source speech-to-speech models and numerous voice agent frameworks exist, no single resource explains the complete pipeline from individual components to a working streaming voice agent with function calling capabilities. Through systematic investigation, we find that (1) native speech-to-speech models like Qwen2.5-Omni, while capable of high-quality audio generation, are too slow for realtime interaction ($\sim$13s time-to-first-audio); (2) the industry-standard approach uses a cascaded streaming pipeline: STT $\rightarrow$ LLM $\rightarrow$ TTS, where each component streams its output to the next; and (3) the key to ``realtime'' is not any single fast model but rather \textit{streaming and pipelining} across components. We build a complete voice agent using Deepgram (streaming STT), vLLM-served LLMs with function calling (streaming text generation), and ElevenLabs (streaming TTS), achieving a measured P50 time-to-first-audio of 947ms (best case 729ms) with cloud LLM APIs, and comparable latency with self-hosted vLLM on NVIDIA A10G GPU. We release the full codebase as a tutorial with working, tested code for every component.
>
---
#### [new 007] Voice Timbre Attribute Detection with Compact and Interpretable Training-Free Acoustic Parameters
- **分类: eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决语音音色属性检测问题。通过研究简洁的声学参数集，提升检测效果并实现高效、可解释的模型。**

- **链接: [https://arxiv.org/pdf/2603.05091](https://arxiv.org/pdf/2603.05091)**

> **作者:** Aemon Yat Fei Chiu; Yujia Xiao; Qiuqiang Kong; Tan Lee
>
> **备注:** Under review
>
> **摘要:** Voice timbre attribute detection (vTAD) is the task of determining the relative intensity of timbre attributes between speech utterances. Voice timbre is a crucial yet inherently complex component of speech perception. While deep neural network (DNN) embeddings perform well in speaker modelling, they often act as black-box representations with limited physical interpretability and high computational cost. In this work, a compact acoustic parameter set is investigated for vTAD. The set captures important acoustic measures and their temporal dynamics which are found to be crucial in the task. Despite its simplicity, the acoustic parameter set is competitive, outperforming conventional cepstral features and supervised DNN embeddings, and approaching state-of-the-art self-supervised models. Importantly, the studied set require no trainable parameters, incur negligible computation, and offer explicit interpretability for analysing physical traits behind human timbre perception.
>
---
#### [new 008] BabAR: from phoneme recognition to developmental measures of young children's speech production
- **分类: eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决儿童语音自动识别难题。通过构建语料库并训练BabAR系统，提升儿童语音的识别准确率，用于评估语言发展水平。**

- **链接: [https://arxiv.org/pdf/2603.05213](https://arxiv.org/pdf/2603.05213)**

> **作者:** Marvin Lavechin; Elika Bergelson; Roger Levy
>
> **摘要:** Studying early speech development at scale requires automatic tools, yet automatic phoneme recognition, especially for young children, remains largely unsolved. Building on decades of data collection, we curate TinyVox, a corpus of more than half a million phonetically transcribed child vocalizations in English, French, Portuguese, German, and Spanish. We use TinyVox to train BabAR, a cross-linguistic phoneme recognition system for child speech. We find that pretraining the system on multilingual child-centered daylong recordings substantially outperforms alternatives, and that providing 20 seconds of surrounding audio context during fine-tuning further improves performance. Error analyses show that substitutions predominantly fall within the same broad phonetic categories, suggesting suitability for coarse-grained developmental analyses. We validate BabAR by showing that its automatic measures of speech maturity align with developmental estimates from the literature.
>
---
#### [new 009] Latent-Mark: An Audio Watermark Robust to Neural Resynthesis
- **分类: cs.SD; cs.AI**

- **简介: 该论文属于音频水印任务，解决神经重合成攻击下水印脆弱的问题。通过在编解码器不变潜空间嵌入水印，实现鲁棒性与不可感知性。**

- **链接: [https://arxiv.org/pdf/2603.05310](https://arxiv.org/pdf/2603.05310)**

> **作者:** Yen-Shan Chen; Shih-Yu Lai; Ying-Jung Tsou; Yi-Cheng Lin; Bing-Yu Chen; Yun-Nung Chen; Hung-Yi Lee; Shang-Tse Chen
>
> **摘要:** While existing audio watermarking techniques have achieved strong robustness against traditional digital signal processing (DSP) attacks, they remain vulnerable to neural resynthesis. This occurs because modern neural audio codecs act as semantic filters and discard the imperceptible waveform variations used in prior watermarking methods. To address this limitation, we propose Latent-Mark, the first zero-bit audio watermarking framework designed to survive semantic compression. Our key insight is that robustness to the encode-decode process requires embedding the watermark within the codec's invariant latent space. We achieve this by optimizing the audio waveform to induce a detectable directional shift in its encoded latent representation, while constraining perturbations to align with the natural audio manifold to ensure imperceptibility. To prevent overfitting to a single codec's quantization rules, we introduce Cross-Codec Optimization, jointly optimizing the waveform across multiple surrogate codecs to target shared latent invariants. Extensive evaluations demonstrate robust zero-shot transferability to unseen neural codecs, achieving state-of-the-art resilience against traditional DSP attacks while preserving perceptual imperceptibility. Our work inspires future research into universal watermarking frameworks capable of maintaining integrity across increasingly complex and diverse generative distortions.
>
---
#### [new 010] When Denoising Hinders: Revisiting Zero-Shot ASR with SAM-Audio and Whisper
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音识别任务，探讨零样本ASR中降噪处理对识别性能的影响。研究发现，尽管SAM-Audio提升音频质量，却导致WER和CER上升，揭示了人耳感知与机器识别间的不匹配。**

- **链接: [https://arxiv.org/pdf/2603.04710](https://arxiv.org/pdf/2603.04710)**

> **作者:** Akif Islam; Raufun Nahar; Md. Ekramul Hamid
>
> **备注:** 6 pages, 4 figures, 5 tables. IEEE Conference Paper
>
> **摘要:** Recent advances in automatic speech recognition (ASR) and speech enhancement have led to a widespread assumption that improving perceptual audio quality should directly benefit recognition accuracy. In this work, we rigorously examine whether this assumption holds for modern zero-shot ASR systems. We present a systematic empirical study on the impact of Segment Anything Model Audio by Meta AI, a recent foundation-scale speech enhancement model proposed by Meta, when used as a preprocessing step for zero-shot transcription with Whisper. Experiments are conducted across multiple Whisper model variants and two linguistically distinct noisy speech datasets: a real-world Bengali YouTube corpus and a publicly available English noisy dataset. Contrary to common intuition, our results show that SAM-Audio preprocessing consistently degrades ASR performance, increasing both Word Error Rate (WER) and Character Error Rate (CER) compared to raw noisy speech, despite substantial improvements in signal-level quality. Objective Peak Signal-to-Noise Ratio analysis on the English dataset confirms that SAM-Audio produces acoustically cleaner signals, yet this improvement fails to translate into recognition gains. Therefore, we conducted a detailed utterance-level analysis to understand this counterintuitive result. We found that the recognition degradation is a systematic issue affecting the majority of the audio, not just isolated outliers, and that the errors worsen as the Whisper model size increases. These findings expose a fundamental mismatch: audio that is perceptually cleaner to human listeners is not necessarily robust for machine recognition. This highlights the risk of blindly applying state-of-the-art denoising as a preprocessing step in zero-shot ASR pipelines.
>
---
#### [new 011] Hierarchical Decoding for Discrete Speech Synthesis with Multi-Resolution Spoof Detection
- **分类: cs.SD; eess.AS**

- **简介: 该论文属于语音合成任务，解决离散语音合成中的质量与真实感问题。通过多分辨率欺骗检测和分层解码，提升生成语音的鲁棒性和自然度。**

- **链接: [https://arxiv.org/pdf/2603.05373](https://arxiv.org/pdf/2603.05373)**

> **作者:** Junchuan Zhao; Minh Duc Vu; Ye Wang
>
> **备注:** 7 pages, 3 figures, 3 tables, 2 algorithms
>
> **摘要:** Neural codec language models enable high-quality discrete speech synthesis, yet their inference remains vulnerable to token-level artifacts and distributional drift that degrade perceptual realism. Rather than relying on preference optimization or retraining, we propose MSpoof-TTS, a training-free inference framework that improves zero-shot synthesis through multi-resolution spoof guidance. We introduce a Multi-Resolution Token-based Spoof Detection framework that evaluates codec sequences at different temporal granularities to detect locally inconsistent or unnatural patterns. We then integrate the spoof detectors into a hierarchical decoding strategy, progressively pruning low-quality candidates and re-ranking hypotheses. This discriminator-guided generation enhances robustness without modifying model parameters. Experiments validate the effectiveness of our framework for robust and high-quality codec-based speech generation.
>
---
#### [new 012] An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG for Articulatory, Brain, and Muscle Activity During Speech Production
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文属于多模态数据采集任务，旨在同步获取实时MRI、EEG和表面EMG信号，以研究言语生产中的脑、肌和构音活动。**

- **链接: [https://arxiv.org/pdf/2603.04840](https://arxiv.org/pdf/2603.04840)**

> **作者:** Jihwan Lee; Parsa Razmara; Kevin Huang; Sean Foley; Aditya Kommineni; Haley Hsu; Woojae Jeong; Prakash Kumar; Xuan Shi; Yoonjeong Lee; Tiantian Feng; Takfarinas Medani; Ye Tian; Sudarsana Reddy Kadiri; Krishna S. Nayak; Dani Byrd; Louis Goldstein; Richard M. Leahy; Shrikanth Narayanan
>
> **摘要:** Speech production is a complex process spanning neural planning, motor control, muscle activation, and articulatory kinematics. While the acoustic speech signal is the most accessible product of the speech production act, it does not directly reveal its causal neurophysiological substrates. We present the first simultaneous acquisition of real-time (dynamic) MRI, EEG, and surface EMG, capturing several key aspects of the speech production chain: brain signals, muscle activations, and articulatory movements. This multimodal acquisition paradigm presents substantial technical challenges, including MRI-induced electromagnetic interference and myogenic artifacts. To mitigate these, we introduce an artifact suppression pipeline tailored to this tri-modal setting. Once fully developed, this framework is poised to offer an unprecedented window into speech neuroscience and insights leading to brain-computer interface advances.
>
---
#### [new 013] TW-Sound580K: A Regional Audio-Text Dataset with Verification-Guided Curation for Localized Audio-Language Modeling
- **分类: cs.SD**

- **简介: 该论文属于音频-文本建模任务，旨在解决方言语音建模中数据稀缺的问题。通过构建TW-Sound580K数据集并优化模型推理策略，提升本地化语音识别性能。**

- **链接: [https://arxiv.org/pdf/2603.05094](https://arxiv.org/pdf/2603.05094)**

> **作者:** Hao-Hui Xie; Ho-Lam Chung; Yi-Cheng Lin; Ke-Han Lu; Wenze Ren; Xie Chen; Hung-yi Lee
>
> **摘要:** Large Audio-Language Models (LALMs) typically struggle with localized dialectal prosody due to the scarcity of specialized corpora. We present TW-Sound580K, a Taiwanese audio-text instruction dataset developed through a Verify-Generate-Critique (VGC) protocol. This pipeline leverages Dual-ASR validation to filter 522K raw clips, subsequently expanding them into 580,000 high-fidelity instruction pairs using a teacher model. The dataset's utility is demonstrated through Tai-LALM, which fine-tunes a DeSTA 2.5-Audio-initialized backbone and incorporates a dynamic Dual-ASR Arbitration strategy to optimize transcription selection during inference. On the TAU Benchmark, Tai-LALM reaches 49.1% accuracy, marking a 6.5% absolute improvement over the zero-shot baseline (42.6% with ASR text conditioning). This confirms that integrating regional corpora with rigorous curation and dynamic arbitration significantly enhances LALM performance on localized speech.
>
---
#### [new 014] Boosting ASR Robustness via Test-Time Reinforcement Learning with Audio-Text Semantic Rewards
- **分类: cs.SD; cs.AI; cs.LG**

- **简介: 该论文属于语音识别任务，旨在提升ASR系统在噪声和口音等复杂场景下的鲁棒性。通过测试时强化学习方法，结合音频与文本语义奖励，实现模型自适应优化。**

- **链接: [https://arxiv.org/pdf/2603.05231](https://arxiv.org/pdf/2603.05231)**

> **作者:** Linghan Fang; Tianxin Xie; Li Liu
>
> **摘要:** Recently, Automatic Speech Recognition (ASR) systems (e.g., Whisper) have achieved remarkable accuracy improvements but remain highly sensitive to real-world unseen data (data with large distribution shifts), including noisy environments and diverse accents. To address this issue, test-time adaptation (TTA) has shown great potential in improving the model adaptability at inference time without ground-truth labels, and existing TTA methods often rely on pseudo-labeling or entropy minimization. However, by treating model confidence as a learning signal, these methods may reinforce high-confidence errors, leading to confirmation bias that undermines adaptation. To overcome these limitations, we present ASR-TRA, a novel Test-time Reinforcement Adaptation framework inspired by causal intervention. More precisely, our method introduces a learnable decoder prompt and utilizes temperature-controlled stochastic decoding to generate diverse transcription candidates. These are scored by a reward model that measures audio-text semantic alignment, and the resulting feedback is used to update both model and prompt parameters via reinforcement learning. Comprehensive experiments on LibriSpeech with synthetic noise and L2 Arctic accented English datasets demonstrate that our method achieves higher accuracy while maintaining lower latency than existing TTA baselines. Ablation studies further confirm the effectiveness of combining audio and language-based rewards, highlighting our method's enhanced stability and interpretability. Overall, our approach provides a practical and robust solution for deploying ASR systems in challenging real-world conditions.
>
---
#### [new 015] SLICE: Speech Enhancement via Layer-wise Injection of Conditioning Embeddings
- **分类: cs.SD**

- **简介: 该论文属于语音增强任务，解决多类型噪声同时存在的问题。通过在层间注入条件嵌入，提升模型性能，优于输入层条件注入方法。**

- **链接: [https://arxiv.org/pdf/2603.05302](https://arxiv.org/pdf/2603.05302)**

> **作者:** Seokhoon Moon; Kyudan Jung; Jaegul Choo
>
> **备注:** 5 pages, 1 figure, 4 tables, submitted to INTERSPEECH 2026
>
> **摘要:** Real-world speech is often corrupted by multiple degradations simultaneously, including additive noise, reverberation, and nonlinear distortion. Diffusion-based enhancement methods perform well on single degradations but struggle with compound corruptions. Prior noise-aware approaches inject conditioning at the input layer only, which can degrade performance below that of an unconditioned model. To address this, we propose injecting degradation conditioning, derived from a pretrained encoder with multi-task heads for noise type, reverberation, and distortion, into the timestep embedding so that it propagates through all residual blocks without architectural changes. In controlled experiments where only the injection method varies, input-level conditioning performs worse than no encoder at all on compound degradations, while layer-wise injection achieves the best results. The method also generalizes to diverse real-world recordings.
>
---
#### [new 016] Visual-Informed Speech Enhancement Using Attention-Based Beamforming
- **分类: eess.AS; cs.AI**

- **简介: 该论文属于语音增强任务，旨在解决低信噪比和复杂场景下的语音增强问题。通过融合视觉信息与麦克风阵列信号，提出VI-NBFNet模型提升性能。**

- **链接: [https://arxiv.org/pdf/2603.05270](https://arxiv.org/pdf/2603.05270)**

> **作者:** Chihyun Liu; Jiaxuan Fan; Mingtung Sun; Michael Anthony; Mingsian R. Bai; Yu Tsao
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** Recent studies have demonstrated that incorporating auxiliary information, such as speaker voiceprint or visual cues, can substantially improve Speech Enhancement (SE) performance. However, single-channel methods often yield suboptimal results in low signal-to-noise ratio (SNR) conditions, when there is high reverberation, or in complex scenarios involving dynamic speakers, overlapping speech, or non-stationary noise. To address these issues, we propose a novel Visual-Informed Neural Beamforming Network (VI-NBFNet), which integrates microphone array signal processing and deep neural networks (DNNs) using multimodal input features. The proposed network leverages a pretrained visual speech recognition model to extract lip movements as input features, which serve for voice activity detection (VAD) and target speaker identification. The system is intended to handle both static and moving speakers by introducing a supervised end-to-end beamforming framework equipped with an attention mechanism. The experimental results demonstrated that the proposed audiovisual system has achieved better SE performance and robustness for both stationary and dynamic speaker scenarios, compared to several baseline methods.
>
---
#### [new 017] Training Dynamics-Aware Multi-Factor Curriculum Learning for Target Speaker Extraction
- **分类: cs.SD**

- **简介: 该论文属于目标说话人提取任务，旨在解决多说话人混合中特定说话人语音分离的问题。提出多因素课程学习策略，并引入TSE-Datamap分析训练动态，提升复杂场景下的提取效果。**

- **链接: [https://arxiv.org/pdf/2603.04943](https://arxiv.org/pdf/2603.04943)**

> **作者:** Yun Liu; Xuechen Liu; Xiaoxiao Miao; Junichi Yamagishi
>
> **摘要:** Target speaker extraction (TSE) aims to isolate a specific speaker's voice from multi-speaker mixtures. Despite strong benchmark results, real-world performance often degrades due to different interacting factors. Previous curriculum learning approaches for TSE typically address these factors separately, failing to capture their complex interactions and relying on predefined difficulty factors that may not align with actual model learning behavior. To address this challenge, we first propose a multi-factor curriculum learning strategy that jointly schedules SNR thresholds, speaker counts, overlap ratios, and synthetic/real proportions, enabling progressive learning from simple to complex scenarios. However, determining optimal scheduling without predefined assumptions remains challenging. We therefore introduce TSE-Datamap, a visualization framework that grounds curriculum design in observed training dynamics by tracking confidence and variability across training epochs. Our analysis reveals three characteristic data regions: (i) easy-to-learn examples where models consistently perform well, (ii) ambiguous examples where models oscillate between alternative predictions, and (iii) hard-to-learn examples where models persistently struggle. Guided by these data-driven insights, our methods improve extraction results over random sampling, with particularly strong gains in challenging multi-speaker scenarios.
>
---
#### [new 018] SarcasmMiner: A Dual-Track Post-Training Framework for Robust Audio-Visual Sarcasm Reasoning
- **分类: cs.MM; cs.CL; cs.SD**

- **简介: 该论文属于多模态 sarcasm 检测任务，旨在提升模型在文本、音频和视觉信息中的讽刺推理能力。提出 SarcasmMiner 框架，通过强化学习和双轨蒸馏策略优化模型表现。**

- **链接: [https://arxiv.org/pdf/2603.05275](https://arxiv.org/pdf/2603.05275)**

> **作者:** Zhu Li; Yongjian Chen; Huiyuan Lai; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Multimodal sarcasm detection requires resolving pragmatic incongruity across textual, acoustic, and visual cues through cross-modal reasoning. To enable robust sarcasm reasoning with foundation models, we propose SarcasmMiner, a reinforcement learning based post-training framework that resists hallucination in multimodal reasoning. We reformulate sarcasm detection as structured reasoning and adopt a dual-track distillation strategy: high-quality teacher trajectories initialize the student model, while the full set of trajectories trains a generative reward model (GenRM) to evaluate reasoning quality. The student is optimized with group relative policy optimization (GRPO) using decoupled rewards for accuracy and reasoning quality. On MUStARD++, SarcasmMiner increases F1 from 59.83% (zero-shot), 68.23% (supervised finetuning) to 70.22%. These findings suggest that reasoning-aware reward modeling enhances both performance and multimodal grounding.
>
---
#### [new 019] WavSLM: Single-Stream Speech Language Modeling via WavLM Distillation
- **分类: cs.LG; cs.AI; cs.CL; cs.SD**

- **简介: 该论文提出WavSLM，属于语音语言建模任务，解决语音中语义与声学信息融合难题。通过量化蒸馏WavLM表示，构建单流模型，实现无需文本监督的语音生成。**

- **链接: [https://arxiv.org/pdf/2603.05299](https://arxiv.org/pdf/2603.05299)**

> **作者:** Luca Della Libera; Cem Subakan; Mirco Ravanelli
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Large language models show that simple autoregressive training can yield scalable and coherent generation, but extending this paradigm to speech remains challenging due to the entanglement of semantic and acoustic information. Most existing speech language models rely on text supervision, hierarchical token streams, or complex hybrid architectures, departing from the single-stream generative pretraining paradigm that has proven effective in text. In this work, we introduce WavSLM, a speech language model trained by quantizing and distilling self-supervised WavLM representations into a single codebook and optimizing an autoregressive next-chunk prediction objective. WavSLM jointly models semantic and acoustic information within a single token stream without text supervision or text pretraining. Despite its simplicity, it achieves competitive performance on consistency benchmarks and speech generation while using fewer parameters, less training data, and supporting streaming inference. Demo samples are available at this https URL.
>
---
#### [new 020] Exploring the potential and limitations of Model Merging for Multi-Domain Adaptation in ASR
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究模型融合在多领域语音识别中的应用，旨在解决多任务训练的计算成本高问题。通过评估多种融合算法，提出改进方法BoostedTSV-M，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.05354](https://arxiv.org/pdf/2603.05354)**

> **作者:** Carlos Carvalho; Francisco Teixeira; Thomas Rolland; Alberto Abad
>
> **备注:** submitted for review for INTERSPEECH2026 conference
>
> **摘要:** Model merging is a scalable alternative to multi-task training that combines the capabilities of multiple specialised models into a single model. This is particularly attractive for large speech foundation models, which are typically adapted through domain-specific fine-tuning, resulting in multiple customised checkpoints, for which repeating full fine-tuning when new data becomes available is computationally prohibitive. In this work, we study model merging for multi-domain ASR and benchmark 11 merging algorithms for 10 European Portuguese domains, evaluating in-domain accuracy, robustness under distribution shift, as well as English and multilingual performance. We further propose BoostedTSV-M, a new merging algorithm based on TSV-M that mitigates rank collapse via singular-value boosting and improves numerical stability. Overall, our approach outperforms full fine-tuning on European Portuguese while preserving out-of-distribution generalisation in a single model.
>
---
## 更新

#### [replaced 001] Noise-to-Notes: Diffusion-based Generation and Refinement for Automatic Drum Transcription
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于自动鼓乐转录任务，解决传统方法在精度与速度上的局限。提出N2N框架，利用扩散模型生成鼓事件，并引入新损失函数和音乐基础模型特征以提升性能。**

- **链接: [https://arxiv.org/pdf/2509.21739](https://arxiv.org/pdf/2509.21739)**

> **作者:** Michael Yeung; Keisuke Toyama; Toya Teramoto; Shusuke Takahashi; Tamaki Kojima
>
> **备注:** Accepted to ICASSP 2026
>
> **摘要:** Automatic drum transcription (ADT) is traditionally formulated as a discriminative task to predict drum events from audio spectrograms. In this work, we redefine ADT as a conditional generative task and introduce Noise-to-Notes (N2N), a framework leveraging diffusion modeling to transform audio-conditioned Gaussian noise into drum events with associated velocities. This generative diffusion approach offers distinct advantages, including a flexible speed-accuracy trade-off and strong inpainting capabilities. However, the generation of binary onset and continuous velocity values presents a challenge for diffusion models, and to overcome this, we introduce an Annealed Pseudo-Huber loss to facilitate effective joint optimization. Finally, to augment low-level spectrogram features, we propose incorporating features extracted from music foundation models (MFMs), which capture high-level semantic information and enhance robustness to out-of-domain drum audio. Experimental results demonstrate that including MFM features significantly improves robustness and N2N establishes a new state-of-the-art performance across multiple ADT benchmarks.
>
---
#### [replaced 002] Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and Frame-Level Attention
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音情感识别任务，旨在解决情感复杂性和数据稀缺问题。提出多损失学习框架，结合能量自适应混元和帧级注意力机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.04551](https://arxiv.org/pdf/2512.04551)**

> **作者:** Cong Wang; Yizhong Geng; Yuhua Wen; Qifei Li; Yingming Gao; Ruimin Wang; Chunfeng Wang; Hao Li; Ya Li; Wei Chen
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** Speech emotion recognition (SER) is an important technology in human-computer interaction. However, achieving high performance is challenging due to emotional complexity and scarce annotated data. To tackle these challenges, we propose a multi-loss learning (MLL) framework integrating an energy-adaptive mixup (EAM) method and a frame-level attention module (FLAM). The EAM method leverages SNR-based augmentation to generate diverse speech samples capturing subtle emotional variations. FLAM enhances frame-level feature extraction for multi-frame emotional cues. Our MLL strategy combines Kullback-Leibler divergence, focal, center, and supervised contrastive loss to optimize learning, address class imbalance, and improve feature separability. We evaluate our method on four widely used SER datasets: IEMOCAP, MSP-IMPROV, RAVDESS, and SAVEE. The results demonstrate our method achieves state-of-the-art performance, suggesting its effectiveness and robustness.
>
---
#### [replaced 003] SAM: A Mamba-2 State-Space Audio-Language Model
- **分类: cs.SD; eess.AS**

- **简介: 该论文提出SAM，一个结合音频编码器和Mamba-2的音视频语言模型，解决音频理解与生成任务。通过优化音频表示和指令监督，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.15680](https://arxiv.org/pdf/2509.15680)**

> **作者:** Taehan Lee; Jaehan Jung; Hyukjun Lee
>
> **备注:** 6 pages, Submitted to Interspeech 2026
>
> **摘要:** We present SAM, a State-space Audio-language Model that integrates an audio encoder with a Mamba-2 backbone. SAM-2.7B achieves 21.1 mAP on AudioSet and 17.6 SPICE on AudioCaps, matching or surpassing larger 7B transformer-based models with fewer parameters. We further provide the first systematic, representation-level analysis of how SSMs interact with audio encoder outputs: (1) joint audio encoder finetuning is essential, supported by accuracy gains and observed adaptation of token representation rank and similarity across different SSM sizes; (2) despite linear scaling, SSMs benefit more from compact, information-rich audio token representations than from excessively long token sequences; and (3) incorporating instruction-following supervision substantially improves reasoning ability, boosting MMAU-Sound accuracy from 22.8 to 56.8. Through comprehensive experiments and analysis, we establish practical design principles for SSMs as strong, scalable backbones for audio-language models.
>
---
#### [replaced 004] A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations
- **分类: eess.AS; cs.SD**

- **简介: 该论文属于语音自监督学习领域，旨在提升模型的可解释性。通过分析11个模型，研究其如何编码说话人特定信息，揭示了模型内部表征的层次结构及动态韵律特性。**

- **链接: [https://arxiv.org/pdf/2501.05310](https://arxiv.org/pdf/2501.05310)**

> **作者:** Aemon Yat Fei Chiu; Kei Ching Fung; Roger Tsz Yeung Li; Jingyu Li; Tan Lee
>
> **备注:** Under review
>
> **摘要:** Enhancing explainability in speech self-supervised learning (SSL) is important for developing reliable SSL-based speech processing systems. This study probes how speech SSL models encode speaker-specific information via a large-scale probing analysis of 11 models, decomposing identity into acoustic, prosodic, and paralinguistic attributes. The results confirm a general hierarchy wherein initial layers encode fundamental acoustics and middle layers synthesise abstract traits. Crucially, the consensus that final layers purely abstract linguistic content is challenged. It is discovered that larger models unexpectedly recover speaker identity in their deep layers. Furthermore, the intermediate representations of speech SSL models are found to capture dynamic prosody better than specialised speaker embeddings. These insights decode the complex internal mechanics of SSL models, providing guidelines for selecting interpretable and task-optimal representations.
>
---
#### [replaced 005] TSPC: A Two-Stage Phoneme-Centric Architecture for code-switching Vietnamese-English Speech Recognition
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于语音识别任务，解决越英混语识别问题。提出TSPC模型，通过两阶段音素中心架构提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2509.05983](https://arxiv.org/pdf/2509.05983)**

> **作者:** Tran Nguyen Anh; Truong Dinh Dung; Vo Van Nam; Minh N. H. Nguyen
>
> **备注:** Update new version
>
> **摘要:** Code-switching (CS) presents a significant challenge for general Auto-Speech Recognition (ASR) systems. Existing methods often fail to capture the sub tle phonological shifts inherent in CS scenarios. The challenge is particu larly difficult for language pairs like Vietnamese and English, where both distinct phonological features and the ambiguity arising from similar sound recognition are present. In this paper, we propose a novel architecture for Vietnamese-English CS ASR, a Two-Stage Phoneme-Centric model (TSPC). TSPC adopts a phoneme-centric approach based on an extended Vietnamese phoneme set as an intermediate representation for mixed-lingual modeling, while remaining efficient under low computational-resource constraints. Ex perimental results demonstrate that TSPC consistently outperforms exist ing baselines, including PhoWhisper-base, in Vietnamese-English CS ASR, achieving a significantly lower word error rate of 19.06% with reduced train ing resources. Furthermore, the phonetic-based two-stage architecture en ables phoneme adaptation and language conversion to enhance ASR perfor mance in complex CS Vietnamese-English ASR scenarios.
>
---
#### [replaced 006] InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions
- **分类: cs.CV; cs.AI; cs.SD**

- **简介: 该论文属于多模态人类动画生成任务，旨在解决多主体互动场景下条件控制不精确的问题。提出新框架实现区域化条件绑定与布局对齐，提升多人对话视频生成质量。**

- **链接: [https://arxiv.org/pdf/2506.09984](https://arxiv.org/pdf/2506.09984)**

> **作者:** Zhenzhi Wang; Jiaqi Yang; Jianwen Jiang; Chao Liang; Gaojie Lin; Zerong Zheng; Ceyuan Yang; Yuan Zhang; Mingyuan Gao; Dahua Lin
>
> **备注:** ICLR 2026 Camera Ready Version. TL;DR: The first multi-person dialogue video generation method from pairs of reference image and audio via explicit layout-aligned condition injection. Project page this https URL
>
> **摘要:** End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios where multiple concepts could appear in the same video with rich human-human interactions and human-object interactions. Such a global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region-specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in an iterative manner. This design enables the high-quality generation of human dialogue videos between two to three people or video customization from multiple reference images. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods. Video demos are available at this https URL
>
---
#### [replaced 007] MultiAPI Spoof: A Multi-API Dataset and Local-Attention Network for Speech Anti-spoofing Detection
- **分类: cs.SD**

- **简介: 该论文属于语音防欺骗检测任务，旨在解决现有基准与真实场景的差异问题。构建了多API数据集MultiAPI Spoof，并提出改进的网络模型Nes2Net-LA以提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.07352](https://arxiv.org/pdf/2512.07352)**

> **作者:** Xueping Zhang; Zhenshan Zhang; Yechen Wang; Linxi Li; Liwei Jin; Ming Li
>
> **备注:** Submited to Interspeech 2026
>
> **摘要:** Existing speech anti-spoofing benchmarks rely on a narrow set of public models, creating a substantial gap from real-world scenarios in which commercial systems employ diverse, often proprietary APIs. To address this issue, we introduce MultiAPI Spoof, a multi-API audio anti-spoofing dataset comprising about 230 hours of synthetic speech generated by 30 distinct APIs, including commercial services, open-source models, and online platforms. Furthermore, we propose Nes2Net-LA, a local-attention enhanced variant of Nes2Net that improves local context modeling and fine-grained spoofing feature extraction. Based on this dataset, we also define the API tracing task, enabling fine-grained attribution of spoofed audio to its generation source. Experiments show that Nes2Net-LA achieves state-of-the-art performance and offers superior robustness, particularly under diverse and unseen spoofing conditions. Code \footnote{this https URL} and dataset \footnote{this https URL} have been released.
>
---
#### [replaced 008] BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings
- **分类: eess.AS; cs.LG; cs.SD**

- **简介: 该论文提出BabyHuBERT，解决儿童语音分割问题。针对儿童语音与成人差异，利用多语言数据训练模型，提升语音类型分类效果。**

- **链接: [https://arxiv.org/pdf/2509.15001](https://arxiv.org/pdf/2509.15001)**

> **作者:** Théo Charlot; Tarek Kunze; Maxime Poli; Alejandrina Cristia; Emmanuel Dupoux; Marvin Lavechin
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** Child-centered daylong recordings are essential for studying early language development, but existing speech models trained on clean adult data perform poorly due to acoustic and linguistic differences. We introduce BabyHuBERT, a self-supervised speech model trained on 13,000 hours of multilingual child-centered recordings spanning 40+ languages. Evaluated on voice type classification -- distinguishing target children from female adults, male adults, and other children, a key preprocessing step for analyzing naturalistic language experiences -- BabyHuBERT-VTC achieves F1-scores from 52.1% to 74.4% across six corpora, consistently outperforming W2V2-LL4300 (English daylongs) and HuBERT (clean adult speech). Notable gains include 13.2 and 15.9 absolute F1 points over HuBERT on Vanuatu and Solomon Islands, demonstrating effectiveness on underrepresented languages. We share code and model to support researchers working with child-centered recordings across diverse linguistic contexts.
>
---
#### [replaced 009] RA-QA: A Benchmarking System for Respiratory Audio Question Answering Under Real-World Heterogeneity
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文提出RA-QA基准，解决呼吸音频问答任务中的真实世界异质性问题，通过构建多样化数据集和评估体系，评估多种模型性能。**

- **链接: [https://arxiv.org/pdf/2602.18452](https://arxiv.org/pdf/2602.18452)**

> **作者:** Gaia A. Bertolino; Yuwei Zhang; Tong Xia; Domenico Talia; Cecilia Mascolo
>
> **摘要:** As conversational multimodal AI tools are increasingly adopted to process patient data for health assessment, robust benchmarks are needed to measure progress and expose failure modes under realistic conditions. Despite the importance of respiratory audio for mobile health screening, respiratory audio question answering remains underexplored, with existing studies evaluated narrowly and lacking real-world heterogeneity across modalities, devices, and question types. We hence introduce the Respiratory-Audio Question-Answering (RA-QA) benchmark, including a standardized data generation pipeline, a comprehensive multimodal QA collection, and a unified evaluation protocol. RA-QA harmonizes public RA datasets into a collection of 9 million format-diverse QA pairs covering diagnostic and contextual attributes. We benchmark classical ML baselines alongside multimodal audio-language models, establishing reproducible reference points and showing how current approaches fail under heterogeneity.
>
---
#### [replaced 010] VoxKnesset: A Large-Scale Longitudinal Hebrew Speech Dataset for Aging Speaker Modeling
- **分类: eess.AS; cs.CL; cs.LG; cs.SD; eess.SP**

- **简介: 该论文提出VoxKnesset数据集，用于研究语音随年龄变化的问题。任务为语音处理中的老化建模，解决现有数据不足的问题，通过构建大规模纵向 Hebrew 语音数据集并进行模型评估。**

- **链接: [https://arxiv.org/pdf/2603.01270](https://arxiv.org/pdf/2603.01270)**

> **作者:** Yanir Marmor; Arad Zulti; David Krongauz; Adam Gabet; Yoad Snapir; Yair Lifshitz; Eran Segal
>
> **备注:** 4 pages, 5 figures, 2 tables
>
> **摘要:** Speech processing systems face a fundamental challenge: the human voice changes with age, yet few datasets support rigorous longitudinal evaluation. We introduce VoxKnesset, an open-access dataset of ~2,300 hours of Hebrew parliamentary speech spanning 2009-2025, comprising 393 speakers with recording spans of up to 15 years. Each segment includes aligned transcripts and verified demographic metadata from official parliamentary records. We benchmark modern speech embeddings (WavLM-Large, ECAPA-TDNN, Wav2Vec2-XLSR-1B) on age prediction and speaker verification under longitudinal conditions. Speaker verification EER rises from 2.15\% to 4.58\% over 15 years for the strongest model, and cross-sectionally trained age regressors fail to capture within-speaker aging, while longitudinally trained models recover a meaningful temporal signal. We publicly release the dataset and pipeline to support aging-robust speech systems and Hebrew speech processing.
>
---
#### [replaced 011] Conversational Speech Reveals Structural Robustness Failures in SpeechLLM Backbones
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音LLM在对话语音中的结构鲁棒性问题，通过分析非流畅内容揭示模型的编辑策略。任务为语音理解中的结构修复，解决模型对口语输入的处理偏差。工作包括评估不同模型并分析其编辑行为。**

- **链接: [https://arxiv.org/pdf/2509.20321](https://arxiv.org/pdf/2509.20321)**

> **作者:** Maria Teleki; Sai Janjur; Haoran Liu; Oliver Grabner; Ketan Verma; Thomas Docog; Xiangjue Dong; Lingfeng Shi; Cong Wang; Stephanie Birkelbach; Jason Kim; Yin Zhang; Éva Székely; James Caverlee
>
> **摘要:** LLMs serve as the backbone in SpeechLLMs, yet their behavior on spontaneous conversational input remains poorly understood. Conversational speech contains pervasive disfluencies -- interjections, edits, and parentheticals -- that are rare in the written corpora used for pre-training. Because gold disfluency removal is a deletion-only task, it serves as a controlled probe to determine whether a model performs faithful structural repair or biased reinterpretation. Using the DRES evaluation framework, we evaluate proprietary and open-source LLMs across architectures and scales. We show that model performance clusters into stable precision-recall regimes reflecting distinct editing policies. Notably, reasoning models systematically over-delete fluent content, revealing a bias toward semantic abstraction over structural fidelity. While fine-tuning achieves SOTA results, it harms generalization. Our findings demonstrate that robustness to speech is shaped by specific training objectives.
>
---
#### [replaced 012] Fine-grained Soundscape Control for Augmented Hearing
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文属于声音场景控制任务，解决听觉设备无法精细调节多声源的问题。提出Aurchestra系统，实现多目标声音的实时分离与独立控制。**

- **链接: [https://arxiv.org/pdf/2603.00395](https://arxiv.org/pdf/2603.00395)**

> **作者:** Seunghyun Oh; Malek Itani; Aseem Gauri; Shyamnath Gollakota
>
> **备注:** 15 pages, 11 figures, 4 tables, submitted to ACM MobiSys 2026
>
> **摘要:** Hearables are becoming ubiquitous, yet their sound controls remain blunt: users can either enable global noise suppression or focus on a single target sound. Real-world acoustic scenes, however, contain many simultaneous sources that users may want to adjust independently. We introduce Aurchestra, the first system to provide fine-grained, real-time soundscape control on resource-constrained hearables. Our system has two key components: (1) a dynamic interface that surfaces only active sound classes and (2) a real-time, on-device multi-output extraction network that generates separate streams for each selected class, achieving robust performance for upto 5 overlapping target sounds, and letting users mix their environment by customizing per-class volumes, much like an audio engineer mixes tracks. We optimize the model architecture for multiple compute-limited platforms and demonstrate real-time performance on 6 ms streaming audio chunks. Across real-world environments in previously unseen indoor and outdoor scenarios, our system enables expressive per-class sound control and achieves substantial improvements in target-class enhancement and interference suppression. Our results show that the world need not be heard as a single, undifferentiated stream: with Aurchestra, the soundscape becomes truly programmable.
>
---
#### [replaced 013] Schrödinger Bridge Mamba for One-Step Speech Enhancement
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **简介: 该论文属于语音增强任务，旨在高效完成去噪与去混响。提出SBM模型，结合Schrödinger Bridge训练范式与Mamba架构，实现单步推理的高质量语音增强。**

- **链接: [https://arxiv.org/pdf/2510.16834](https://arxiv.org/pdf/2510.16834)**

> **作者:** Jing Yang; Sirui Wang; Chao Wu; Lei Guo; Fan Fan
>
> **备注:** Revised version. Submitted to Interspeech 2026
>
> **摘要:** We present Schrödinger Bridge Mamba (SBM), a novel model for efficient speech enhancement by integrating the Schrödinger Bridge (SB) training paradigm and the Mamba architecture. Experiments of joint denoising and dereverberation tasks demonstrate SBM outperforms strong generative and discriminative methods on multiple metrics with only one step of inference while achieving a competitive real-time factor for streaming feasibility. Ablation studies reveal that the SB paradigm consistently yields improved performance across diverse architectures over conventional mapping. Furthermore, Mamba exhibits a stronger performance under the SB paradigm compared to Multi-Head Self-Attention (MHSA) and Long Short-Term Memory (LSTM) backbones. These findings highlight the synergy between the Mamba architecture and the SB trajectory-based training, providing a high-quality solution for real-world speech enhancement. Demo page: this https URL
>
---
#### [replaced 014] Benchmarking Speech Systems for Frontline Health Conversations: The DISPLACE-M Challenge
- **分类: eess.AS**

- **简介: 该论文属于医疗对话理解任务，旨在解决真实医疗对话中的多说话人交互问题。构建了数据集并提供了基线系统，涵盖语音识别、话题识别等四个任务。**

- **链接: [https://arxiv.org/pdf/2603.02813](https://arxiv.org/pdf/2603.02813)**

> **作者:** Dhanya E; Ankita Meena; Manas Nanivadekar; Noumida A; Victor Azad; Ashwini Nagaraj Shenoy; Pratik Roy Chowdhuri; Shobhit Banga; Vanshika Chhabra; Chitralekha Bhat; Shareef babu Kalluri; Srikanth Raj Chetupalli; Deepu Vijayasenan; Sriram Ganapathy
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** The DIarization and Speech Processing for LAnguage understanding in Conversational Environments - Medical (DISPLACE-M) challenge introduces a conversational AI benchmark for understanding goal-oriented, real-world medical dialogues. The challenge addresses multi-speaker interactions between frontline health workers and care seekers, characterized by spontaneous, noisy and overlapping speech. As part of the challenge, medical conversational dataset comprising 40 hours of development and 15 hours of blind evaluation recordings was released. We provided baseline systems across 4 tasks - speaker diarization, automatic speech recognition, topic identification and dialogue summarization - to enable consistent benchmarking. System performance is evaluated using diarization error rate (DER), time-constrained minimum-permutation word error rate (tcpWER) and ROUGE-L. This paper describes the Phase-I evaluation - data, tasks and baseline systems - along with the summary of the evaluation results.
>
---
#### [replaced 015] The PARLO Dementia Corpus: A German Multi-Center Resource for Alzheimer's Disease
- **分类: eess.AS**

- **简介: 该论文介绍PARLO Dementia Corpus，一个用于阿尔茨海默病研究的德语多中心数据集，旨在解决非侵入性诊断方法不足的问题，通过语音分析进行认知障碍检测。**

- **链接: [https://arxiv.org/pdf/2603.03471](https://arxiv.org/pdf/2603.03471)**

> **作者:** Franziska Braun; Christopher Witzl; Florian Hönig; Elmar Nöth; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Early and accessible detection of Alzheimer's disease (AD) remains a major challenge, as current diagnostic methods often rely on costly and invasive biomarkers. Speech and language analysis has emerged as a promising non-invasive and scalable approach to detecting cognitive impairment, but research in this area is hindered by the lack of publicly available datasets, especially for languages other than English. This paper introduces the PARLO Dementia Corpus (PDC), a new multi-center, clinically validated German resource for AD collected across nine academic memory clinics in Germany. The dataset comprises speech recordings from individuals with AD-related mild cognitive impairment and mild to moderate dementia, as well as cognitively healthy controls. Speech was elicited using a standardized test battery of eight neuropsychological tasks, including confrontation naming, verbal fluency, word repetition, picture description, story reading, and recall tasks. In addition to audio recordings, the dataset includes manually verified transcriptions and detailed demographic, clinical, and biomarker metadata. Baseline experiments on ASR benchmarking, automated test evaluation, and LLM-based classification illustrate the feasibility of automatic, speech-based cognitive assessment and highlight the diagnostic value of recall-driven speech production. The PDC thus establishes the first publicly available German benchmark for multi-modal and cross-lingual research on neurodegenerative diseases.
>
---
#### [replaced 016] Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Vevo2，解决语音与歌唱生成的可控性问题。通过两个音频分词器和两种建模阶段，实现语音与歌唱的统一生成与控制。**

- **链接: [https://arxiv.org/pdf/2508.16332](https://arxiv.org/pdf/2508.16332)**

> **作者:** Xueyao Zhang; Junan Zhang; Yuancheng Wang; Chaoren Wang; Yuanzhe Chen; Dongya Jia; Zhuo Chen; Zhizheng Wu
>
> **备注:** Accepted by the IEEE Transactions on Audio, Speech and Language Processing (TASLP)
>
> **摘要:** Controllable human voice generation, particularly for expressive domains like singing, remains a significant challenge. This paper introduces Vevo2, a unified framework for controllable speech and singing voice generation. To tackle issues like the scarcity of annotated singing data and to enable flexible controllability, Vevo2 introduces two audio tokenizers: (1) a unified music-notation-free prosody tokenizer that captures prosody and melody from speech, singing, and even instrumental sounds, and (2) a unified content-style tokenizer that encodes linguistic content, prosody, and style for both speech and singing, while enabling timbre disentanglement. Vevo2 consists of an auto-regressive (AR) content-style modeling stage, which aims to enable controllability over text, prosody, and style, as well as a flow-matching acoustic modeling stage that allows for timbre control. Particularly, during the speech-singing joint training of the AR model, we propose both explicit and implicit prosody learning strategies to bridge speech and singing voice. Moreover, to further enhance the Vevo2's ability to follow text and prosody, we design a multi-objective post-training task that integrates both intelligibility and prosody similarity alignment. Experimental results show that the unified modeling in Vevo2 brings mutual benefits to both speech and singing voice generation. Additionally, Vevo2's effectiveness across a wide range of synthesis, conversion, and editing tasks for both speech and singing further demonstrates its strong generalization ability and versatility. Audio samples are are available at this https URL.
>
---
