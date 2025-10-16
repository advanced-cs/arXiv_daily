# 音频 cs.SD;  eess.AS

- **最新发布 16 篇**

- **更新 9 篇**

## 最新发布

#### [new 001] Automatic Speech Recognition in the Modern Era: Architectures, Training, and Evaluation
- **分类: eess.AS; cs.AI; cs.SD**

- **简介: 该论文属于综述任务，旨在梳理现代语音识别技术的发展。它回顾了从传统混合系统到端到端模型的演进，分析了主流架构、训练范式及评估方法，并探讨了实际部署中的关键问题与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.12827v1](http://arxiv.org/pdf/2510.12827v1)**

> **作者:** Md. Nayeem; Md Shamse Tabrej; Kabbojit Jit Deb; Shaonti Goswami; Md. Azizul Hakim
>
> **摘要:** Automatic Speech Recognition (ASR) has undergone a profound transformation over the past decade, driven by advances in deep learning. This survey provides a comprehensive overview of the modern era of ASR, charting its evolution from traditional hybrid systems, such as Gaussian Mixture Model-Hidden Markov Models (GMM-HMMs) and Deep Neural Network-HMMs (DNN-HMMs), to the now-dominant end-to-end neural architectures. We systematically review the foundational end-to-end paradigms: Connectionist Temporal Classification (CTC), attention-based encoder-decoder models, and the Recurrent Neural Network Transducer (RNN-T), which established the groundwork for fully integrated speech-to-text systems. We then detail the subsequent architectural shift towards Transformer and Conformer models, which leverage self-attention to capture long-range dependencies with high computational efficiency. A central theme of this survey is the parallel revolution in training paradigms. We examine the progression from fully supervised learning, augmented by techniques like SpecAugment, to the rise of self-supervised learning (SSL) with foundation models such as wav2vec 2.0, which drastically reduce the reliance on transcribed data. Furthermore, we analyze the impact of largescale, weakly supervised models like Whisper, which achieve unprecedented robustness through massive data diversity. The paper also covers essential ecosystem components, including key datasets and benchmarks (e.g., LibriSpeech, Switchboard, CHiME), standard evaluation metrics (e.g., Word Error Rate), and critical considerations for real-world deployment, such as streaming inference, on-device efficiency, and the ethical imperatives of fairness and robustness. We conclude by outlining open challenges and future research directions.
>
---
#### [new 002] Acoustic Teleportation via Disentangled Neural Audio Codec Representations
- **分类: eess.AS**

- **简介: 该论文研究声学遥感任务，旨在迁移语音中的房间特性同时保留内容与说话人身份。基于EnCodec架构，通过多任务训练实现语音内容与环境特征的解耦，提升客观质量，并验证了声学嵌入与RT60的相关性及聚类特性。**

- **链接: [http://arxiv.org/pdf/2510.13221v1](http://arxiv.org/pdf/2510.13221v1)**

> **作者:** Philipp Grundhuber; Mhd Modar Halimeh; Emanuël A. P. Habets
>
> **摘要:** This paper presents an approach for acoustic teleportation by disentangling speech content from acoustic environment characteristics in neural audio codec representations. Acoustic teleportation transfers room characteristics between speech recordings while preserving content and speaker identity. We build upon previous work using the EnCodec architecture, achieving substantial objective quality improvements with non-intrusive ScoreQ scores of 3.03, compared to 2.44 for prior methods. Our training strategy incorporates five tasks: clean reconstruction, reverberated reconstruction, dereverberation, and two variants of acoustic teleportation. We demonstrate that temporal downsampling of the acoustic embedding significantly degrades performance, with even 2x downsampling resulting in a statistically significant reduction in quality. The learned acoustic embeddings exhibit strong correlations with RT60. Effective disentanglement is demonstrated using t-SNE clustering analysis, where acoustic embeddings cluster by room while speech embeddings cluster by speaker.
>
---
#### [new 003] Adaptive vector steering: A training-free, layer-wise intervention for hallucination mitigation in large audio and multimodal models
- **分类: cs.SD; cs.LG; eess.AS**

- **简介: 该论文针对音频-语言模型的幻觉问题，提出一种无需训练的层间向量调控方法（AVS），通过干预模型内部表征来增强生成内容与音频的对齐，有效提升多个任务下的正确率。**

- **链接: [http://arxiv.org/pdf/2510.12851v1](http://arxiv.org/pdf/2510.12851v1)**

> **作者:** Tsung-En Lin; Kuan-Yi Lee; Hung-Yi Lee
>
> **备注:** Note: This preprint is a version of the paper submitted to ICASSP 2026. The author list here includes contributors who provided additional supervision and guidance. The official ICASSP submission may differ slightly in author composition
>
> **摘要:** Large Audio-Language Models and Multi-Modal Large Language Models have demonstrated strong capabilities in tasks such as Audio Question Answering (AQA), Audio Captioning, and Automatic Speech Recognition (ASR). However, there is growing evidence that these models can hallucinate about the content of the audio. To address this issue, we probe the models' internal states and propose Adaptive Vector Steering (AVS), a method that better grounds generation in audio content. We also identify a strong correlation between output correctness and internal representations. Experiments show consistent performance gains across two models and two benchmarks. On the Audio Hallucination QA dataset, our method boosts the F1-score of Gemma from 0.550 to 0.619 and Qwen from 0.626 to 0.632. Furthermore, our method increases the accuracy of Qwen on MMAU from 0.548 to 0.592, marking an 8% relative increase. To the best of our knowledge, this is the first work to apply vector steering to mitigate hallucination in audio.
>
---
#### [new 004] Continuous-Token Diffusion for Speaker-Referenced TTS in Multimodal LLMs
- **分类: eess.AS; cs.SD**

- **简介: 该论文研究多模态大模型中的语音合成（TTS）任务，旨在解决离散语音标记导致的音质损失问题。作者提出连续标记扩散方法，设计双头架构与两阶段训练策略，实现更高质量的说话人参考TTS。**

- **链接: [http://arxiv.org/pdf/2510.12995v1](http://arxiv.org/pdf/2510.12995v1)**

> **作者:** Xinlu He; Swayambhu Nath Ray; Harish Mallidi; Jia-Hong Huang; Ashwin Bellur; Chander Chandak; M. Maruf; Venkatesh Ravichandran
>
> **摘要:** Unified architectures in multimodal large language models (MLLM) have shown promise in handling diverse tasks within a single framework. In the text-to-speech (TTS) task, current MLLM-based approaches rely on discrete token representations, which disregard the inherently continuous nature of speech and can lead to loss of fine-grained acoustic information.In this work, we investigate the TTS within the MLLM paradigm using continuous speech representations. We design a dual-head architecture and implement two complementary training strategies for a robust model. (1) A diffusion head generating continuous speech representations is added on the MLLM, which is on frame-level and strictly autoregressive. (2) The original language model head is retained to preserve multitask capability and to control the start and end of speech synthesis. (3) Masked training is employed to address exposure bias in autoregressive decoding. (4) To stabilize optimization, we propose a two-stage scheme where the LM is frozen in the second stage, ensuring the diffusion head learns from a fixed input distribution. Evaluations on LibriSpeech(PC) test-clean show that our approach achieves state-of-the-art autoregressive performance, with a WER of 1.95%, speaker similarity of 0.54, and UTMOS of 4.00. The two-stage training yields a 46% relative WER reduction over the one-stage training baseline. These results highlight the effectiveness of combining autoregressive modeling with continuous-token diffusion, supported by a two-stage training procedure.
>
---
#### [new 005] Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses
- **分类: eess.AS; cs.CL; cs.LG**

- **简介: 该论文针对音视频语音识别中的错误修正问题，提出DualHyp框架，利用ASR和VSR双模态假设生成N-best候选，并通过噪声感知的RelPrompt机制引导大模型动态融合与纠错，显著提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.13281v1](http://arxiv.org/pdf/2510.13281v1)**

> **作者:** Sungnyun Kim; Kangwook Jang; Sungwoo Cho; Joon Son Chung; Hoirin Kim; Se-Young Yun
>
> **备注:** Preprint work
>
> **摘要:** This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, DualHyp, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce RelPrompt, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each modality stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction. Under various corruption scenarios, our framework attains up to 57.7% error rate gain on the LRS2 benchmark over standard ASR baseline, contrary to single-stream GER approaches that achieve only 10% gain. To facilitate research within our DualHyp framework, we release the code and the dataset comprising ASR and VSR hypotheses at https://github.com/sungnyun/dualhyp.
>
---
#### [new 006] Gelina: Unified Speech and Gesture Synthesis via Interleaved Token Prediction
- **分类: cs.SD; cs.AI; eess.AS; 68T07**

- **简介: 该论文研究语音与手势联合生成任务，旨在解决传统方法中语音和手势分步生成导致的同步性差问题。提出Gelina框架，通过交错离散标记预测，实现文本到语音和手势的统一生成，支持多说话人、多风格克隆及仅手势生成，显著提升协同表现。**

- **链接: [http://arxiv.org/pdf/2510.12834v1](http://arxiv.org/pdf/2510.12834v1)**

> **作者:** Téo Guichoux; Théodor Lemerle; Shivam Mehta; Jonas Beskow; Gustave Eje Henter; Laure Soulier; Catherine Pelachaud; Nicolas Obin
>
> **备注:** 5 pages
>
> **摘要:** Human communication is multimodal, with speech and gestures tightly coupled, yet most computational methods for generating speech and gestures synthesize them sequentially, weakening synchrony and prosody alignment. We introduce Gelina, a unified framework that jointly synthesizes speech and co-speech gestures from text using interleaved token sequences in a discrete autoregressive backbone, with modality-specific decoders. Gelina supports multi-speaker and multi-style cloning and enables gesture-only synthesis from speech inputs. Subjective and objective evaluations demonstrate competitive speech quality and improved gesture generation over unimodal baselines.
>
---
#### [new 007] Production and Manufacturing of 3D Printed Acoustic Guitars
- **分类: cs.SD; eess.AS**

- **简介: 该论文探索3D打印低成本、可演奏的 acoustic 吉他，解决传统木材依赖与制造成本问题。研究设计并打印PLA吉他，通过分体式结构与CAD建模实现装配，测试音准表现，验证可行性，并提出材料优化方向。**

- **链接: [http://arxiv.org/pdf/2510.12823v1](http://arxiv.org/pdf/2510.12823v1)**

> **作者:** Timothy Tran; William Schiesser
>
> **摘要:** This research investigates the feasibility of producing affordable, functional acoustic guitars using 3D printing, with a focus on producing structural designs with proper tonal performance. Conducted in collaboration with William Schiesser, the study uses a classical guitar model, chosen for its lower string tension, to evaluate the tonal characteristics of a 3D-printed prototype made from polylactic acid (PLA). Due to the build plate size constraints of the Prusa Mark 4 printer, the guitar body was divided into multiple sections joined with press-fit tolerances and minimal cyanoacrylate adhesive. CAD modeling in Fusion 360 ensured dimensional accuracy in press-fit connections and the overall assembly. Following assembly, the guitar was strung with nylon strings and tested using Audacity software to compare recorded frequencies and notes with standard reference values. Results showed large deviations in lower string frequencies, likely caused by the material choice utilized in printing. Accurate pitches were reached with all strings despite frequency differences through tuning, demonstrating that PLA and modern manufacturing methods can produce affordable, playable acoustic guitars despite inevitable challenges. Further research may investigate alternative plastics for superior frequency matching. This approach holds significant potential for expanding access to quality instruments while reducing reliance on endangered tonewoods, thereby encouraging both sustainable instrument production and increased musical participation. This also creates opportunities for disadvantaged communities where access to musical instruments remains a challenge. Keywords: Luthiery, Stereolithography, 3D-Print, Guitar Making
>
---
#### [new 008] MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding
- **分类: cs.SD; cs.AI; cs.MM**

- **简介: 该论文提出MotionBeat，旨在学习与人体运动对齐的音乐表征。针对现有音频表示忽略音乐节奏与身体动作关联的问题，引入具身对比损失和结构节奏对齐损失，结合节拍等变编码与接触感知注意力机制，在音乐到舞蹈生成等多项任务中取得优异表现。**

- **链接: [http://arxiv.org/pdf/2510.13244v1](http://arxiv.org/pdf/2510.13244v1)**

> **作者:** Xuanchen Wang; Heng Wang; Weidong Cai
>
> **备注:** 5 pages, 1 figure. demo page: https://motionbeat2025.github.io/
>
> **摘要:** Music is both an auditory and an embodied phenomenon, closely linked to human motion and naturally expressed through dance. However, most existing audio representations neglect this embodied dimension, limiting their ability to capture rhythmic and structural cues that drive movement. We propose MotionBeat, a framework for motion-aligned music representation learning. MotionBeat is trained with two newly proposed objectives: the Embodied Contrastive Loss (ECL), an enhanced InfoNCE formulation with tempo-aware and beat-jitter negatives to achieve fine-grained rhythmic discrimination, and the Structural Rhythm Alignment Loss (SRAL), which ensures rhythm consistency by aligning music accents with corresponding motion events. Architecturally, MotionBeat introduces bar-equivariant phase rotations to capture cyclic rhythmic patterns and contact-guided attention to emphasize motion events synchronized with musical accents. Experiments show that MotionBeat outperforms state-of-the-art audio encoders in music-to-dance generation and transfers effectively to beat tracking, music tagging, genre and instrument classification, emotion recognition, and audio-visual retrieval. Our project demo page: https://motionbeat2025.github.io/.
>
---
#### [new 009] Beyond Discrete Categories: Multi-Task Valence-Arousal Modeling for Pet Vocalization Analysis
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文研究宠物叫声的情感分析任务，旨在解决传统离散分类难以捕捉情绪强度和模糊性的问题。提出基于多任务学习的连续效价-唤醒度（VA）模型，利用自动生成的VA标签和4.2万条数据训练音频Transformer，提升情绪识别精度，实现更细腻的宠物情感表达建模。**

- **链接: [http://arxiv.org/pdf/2510.12819v1](http://arxiv.org/pdf/2510.12819v1)**

> **作者:** Junyao Huang; Rumin Situ
>
> **备注:** 24 pages, 6 figures, 4 tables. First continuous VA framework for pet vocalization analysis with 42,553 samples
>
> **摘要:** Traditional pet emotion recognition from vocalizations, based on discrete classification, struggles with ambiguity and capturing intensity variations. We propose a continuous Valence-Arousal (VA) model that represents emotions in a two-dimensional space. Our method uses an automatic VA label generation algorithm, enabling large-scale annotation of 42,553 pet vocalization samples. A multi-task learning framework jointly trains VA regression with auxiliary tasks (emotion, body size, gender) to enhance prediction by improving feature learning. Our Audio Transformer model achieves a validation Valence Pearson correlation of r = 0.9024 and an Arousal r = 0.7155, effectively resolving confusion between discrete categories like "territorial" and "happy." This work introduces the first continuous VA framework for pet vocalization analysis, offering a more expressive representation for human-pet interaction, veterinary diagnostics, and behavioral training. The approach shows strong potential for deployment in consumer products like AI pet emotion translators.
>
---
#### [new 010] HyWA: Hypernetwork Weight Adapting Personalized Voice Activity Detection
- **分类: eess.AS; cs.AI; cs.LG; cs.SD**

- **简介: 该论文研究个性化语音活动检测（PVAD），旨在通过说话人嵌入激活特定用户。提出HyWA方法，用超网络调整标准VAD模型部分权重，实现无需修改主干结构的说话人自适应，提升性能并简化部署。**

- **链接: [http://arxiv.org/pdf/2510.12947v1](http://arxiv.org/pdf/2510.12947v1)**

> **作者:** Mahsa Ghazvini Nejad; Hamed Jafarzadeh Asl; Amin Edraki; Mohammadreza Sadeghi; Masoud Asgharian; Yuanhao Yu; Vahid Partovi Nia
>
> **备注:** Mahsa Ghazvini Nejad and Hamed Jafarzadeh Asl contributed equally to this work
>
> **摘要:** Personalized Voice Activity Detection (PVAD) systems activate only in response to a specific target speaker by incorporating speaker embeddings from enrollment utterances. Unlike existing methods that require architectural changes, such as FiLM layers, our approach employs a hypernetwork to modify the weights of a few selected layers within a standard voice activity detection (VAD) model. This enables speaker conditioning without changing the VAD architecture, allowing the same VAD model to adapt to different speakers by updating only a small subset of the layers. We propose HyWA-PVAD, a hypernetwork weight adaptation method, and evaluate it against multiple baseline conditioning techniques. Our comparison shows consistent improvements in PVAD performance. HyWA also offers practical advantages for deployment by preserving the core VAD architecture. Our new approach improves the current conditioning techniques in two ways: i) increases the mean average precision, ii) simplifies deployment by reusing the same VAD architecture.
>
---
#### [new 011] UniMoE-Audio: Unified Speech and Music Generation with Dynamic-Capacity MoE
- **分类: cs.SD; cs.CL**

- **简介: 该论文聚焦统一语音与音乐生成任务，旨在解决领域冲突与数据不平衡问题。作者提出UniMoE-Audio模型，采用动态容量MoE架构与三阶段训练策略，实现语音和音乐的协同生成，提升跨域融合性能。**

- **链接: [http://arxiv.org/pdf/2510.13344v1](http://arxiv.org/pdf/2510.13344v1)**

> **作者:** Zhenyu Liu; Yunxin Li; Xuanyu Zhang; Qixun Teng; Shenyuan Jiang; Xinyu Chen; Haoyuan Shi; Jinchao Li; Qi Wang; Haolan Chen; Fanbo Meng; Mingjun Zhao; Yu Xu; Yancheng He; Baotian Hu; Min Zhang
>
> **摘要:** Recent advances in unified multimodal models indicate a clear trend towards comprehensive content generation. However, the auditory domain remains a significant challenge, with music and speech often developed in isolation, hindering progress towards universal audio synthesis. This separation stems from inherent task conflicts and severe data imbalances, which impede the development of a truly unified audio generation model. To address this challenge, we propose UniMoE-Audio, a unified speech and music generation model within a novel Dynamic-Capacity Mixture-of-Experts (MoE) framework. Architecturally, UniMoE-Audio introduces a Top-P routing strategy for dynamic expert number allocation, and a hybrid expert design comprising routed experts for domain-specific knowledge, shared experts for domain-agnostic features, and null experts for adaptive computation skipping. To tackle data imbalance, we introduce a three-stage training curriculum: 1) Independent Specialist Training leverages original datasets to instill domain-specific knowledge into each "proto-expert" without interference; 2) MoE Integration and Warmup incorporates these specialists into the UniMoE-Audio architecture, warming up the gate module and shared expert using a subset of balanced dataset; and 3) Synergistic Joint Training trains the entire model end-to-end on the fully balanced dataset, fostering enhanced cross-domain synergy. Extensive experiments show that UniMoE-Audio not only achieves state-of-the-art performance on major speech and music generation benchmarks, but also demonstrates superior synergistic learning, mitigating the performance degradation typically seen in naive joint training. Our findings highlight the substantial potential of specialized MoE architecture and curated training strategies in advancing the field of universal audio generation. Homepage: https://mukioxun.github.io/Uni-MoE-site/home.html
>
---
#### [new 012] Steer-MoE: Efficient Audio-Language Alignment with a Mixture-of-Experts Steering Module
- **分类: cs.SD; I.2.7**

- **简介: 该论文研究音频-语言对齐任务，旨在解决现有方法需全模型微调或静态适配器表达能力不足的问题。作者提出SteerMoE，冻结音频编码器和大语言模型，仅训练轻量化的混合专家引导模块，动态调整音频表征以匹配语言模型，实现高效、模块化的多模态对齐。**

- **链接: [http://arxiv.org/pdf/2510.13558v1](http://arxiv.org/pdf/2510.13558v1)**

> **作者:** Ruitao Feng; Bixi Zhang; Sheng Liang; Zheng Yuan
>
> **备注:** 5 pages, 1 figures. Code is available at: https://github.com/forfrt/SteerMoE. Submitted to ICASSP 2026
>
> **摘要:** Aligning pretrained audio encoders and Large Language Models (LLMs) offers a promising, parameter-efficient path to building powerful multimodal agents. However, existing methods often require costly full-model finetuning or rely on static adapters that may lack expressive power. Drawing inspiration from the Platonic Representation Hypothesis, we introduce SteerMoE, a novel and modular framework for audio-language alignment. SteerMoE freezes both the audio encoder and the LLM decoder, training only a lightweight steering module integrated within the encoder's layers. This module uses a Mixture-of-Experts (MoE) router to dynamically select and apply learned steering vectors, progressively transforming continuous audio representations into a space comprehensible to the LLM. By operating entirely in the continuous embedding space, our approach requires no modifications to the LLM's vocabulary and preserves its advanced reasoning and agentic capabilities. We demonstrate through experiments on ASR, audio understanding, and a qualitative function-calling task that SteerMoE achieves strong performance while remaining highly modular and computationally efficient, offering a robust new paradigm for developing sophisticated audio-language systems.
>
---
#### [new 013] VCTR: A Transformer-Based Model for Non-parallel Voice Conversion
- **分类: cs.SD**

- **简介: 该论文研究非平行语音转换任务，旨在解决现有模型难以训练且效果不佳的问题。作者提出VCTR模型，结合混合感知模块和双剪枝自注意力机制，利用对比学习提升全局语义建模能力。**

- **链接: [http://arxiv.org/pdf/2510.12964v1](http://arxiv.org/pdf/2510.12964v1)**

> **作者:** Maharnab Saikia
>
> **摘要:** Non-parallel voice conversion aims to convert voice from a source domain to a target domain without paired training data. Cycle-Consistent Generative Adversarial Networks (CycleGAN) and Variational Autoencoders (VAE) have been used for this task, but these models suffer from difficult training and unsatisfactory results. Later, Contrastive Voice Conversion (CVC) was introduced, utilizing a contrastive learning-based approach to address these issues. However, these methods use CNN-based generators, which can capture local semantics but lacks the ability to capture long-range dependencies necessary for global semantics. In this paper, we propose VCTR, an efficient method for non-parallel voice conversion that leverages the Hybrid Perception Block (HPB) and Dual Pruned Self-Attention (DPSA) along with a contrastive learning-based adversarial approach. The code can be found in https://github.com/Maharnab-Saikia/VCTR.
>
---
#### [new 014] Towards Multimodal Query-Based Spatial Audio Source Extraction
- **分类: eess.AS**

- **简介: 该论文研究多模态查询式空间音频源分离，旨在从多通道混合信号中提取目标声源。提出基于三轴Transformer的框架，利用音频或文本查询条件，结合CLAP嵌入实现统一多模态条件建模，并设计无标签数据生成方法，提升分离质量与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.13308v1](http://arxiv.org/pdf/2510.13308v1)**

> **作者:** Chenxin Yu; Hao Ma; Xu Li; Xiao-Lei Zhang; Mingjie Shao; Chi Zhang; Xuelong Li
>
> **备注:** Submitted to ICASSP 2026
>
> **摘要:** Query-based audio source extraction seeks to recover a target source from a mixture conditioned on a query. Existing approaches are largely confined to single-channel audio, leaving the spatial information in multi-channel recordings underexploited. We introduce a query-based spatial audio source extraction framework for recovering dry target signals from first-order ambisonics (FOA) mixtures. Our method accepts either an audio prompt or a text prompt as condition input, enabling flexible end-to-end extraction. The core of our proposed model lies in a tri-axial Transformer that jointly models temporal, frequency, and spatial channel dependencies. The model uses contrastive language-audio pretraining (CLAP) embeddings to enable unified audio-text conditioning via feature-wise linear modulation (FiLM). To eliminate costly annotations and improve generalization, we propose a label-free data pipeline that dynamically generates spatial mixtures and corresponding targets for training. The result of our experiment with high separation quality demonstrates the efficacy of multimodal conditioning and tri-axial modeling. This work establishes a new paradigm for high-fidelity spatial audio separation in immersive applications.
>
---
#### [new 015] A Critical Review of the Need for Knowledge-Centric Evaluation of Quranic Recitation
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属文献综述任务，旨在解决现有 Quranic 诵读自动评估工具效果不佳的问题。作者分析20年研究，指出当前依赖语音识别技术的局限，主张转向以知识为中心、基于 Tajweed 规则的计算框架，提出融合语言知识与音频分析的混合系统方向。**

- **链接: [http://arxiv.org/pdf/2510.12858v1](http://arxiv.org/pdf/2510.12858v1)**

> **作者:** Mohammed Hilal Al-Kharusi; Khizar Hayat; Khalil Bader Al Ruqeishi; Haroon Rashid Lone
>
> **备注:** 33 pages
>
> **摘要:** The sacred practice of Quranic recitation (Tajweed), governed by precise phonetic, prosodic, and theological rules, faces significant pedagogical challenges in the modern era. While digital technologies promise unprecedented access to education, automated tools for recitation evaluation have failed to achieve widespread adoption or pedagogical efficacy. This literature review investigates this critical gap, conducting a comprehensive analysis of academic research, web platforms, and commercial applications developed over the past two decades. Our synthesis reveals a fundamental misalignment in prevailing approaches that repurpose Automatic Speech Recognition (ASR) architectures, which prioritize lexical recognition over qualitative acoustic assessment and are plagued by data dependency, demographic biases, and an inability to provide diagnostically useful feedback. Critiquing these data--driven paradigms, we argue for a foundational paradigm shift towards a knowledge-centric computational framework. Capitalizing on the immutable nature of the Quranic text and the precisely defined rules of Tajweed, we propose that a robust evaluator must be architected around anticipatory acoustic modeling based on canonical rules and articulation points (Makhraj), rather than relying on statistical patterns learned from imperfect and biased datasets. This review concludes that the future of automated Quranic evaluation lies in hybrid systems that integrate deep linguistic knowledge with advanced audio analysis, offering a path toward robust, equitable, and pedagogically sound tools that can faithfully support learners worldwide.
>
---
#### [new 016] Closing the Gap Between Text and Speech Understanding in LLMs
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文研究语音与文本理解在大语言模型中的性能差距问题，提出SALAD方法，通过跨模态蒸馏和主动选择合成数据，在减少语音数据使用的情况下有效缩小文本-语音理解差距。**

- **链接: [http://arxiv.org/pdf/2510.13632v1](http://arxiv.org/pdf/2510.13632v1)**

> **作者:** Santiago Cuervo; Skyler Seto; Maureen de Seyssel; Richard He Bai; Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly; Zakaria Aldeneh
>
> **摘要:** Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.
>
---
## 更新

#### [replaced 001] AudioGenie-Reasoner: A Training-Free Multi-Agent Framework for Coarse-to-Fine Audio Deep Reasoning
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.16971v2](http://arxiv.org/pdf/2509.16971v2)**

> **作者:** Yan Rong; Chenxing Li; Dong Yu; Li Liu
>
> **摘要:** Audio deep reasoning is a challenging task that requires expert-level perception, multi-step logical inference, and the integration of contextual knowledge. However, existing models suffer from a gap between audio perception and reasoning abilities due to the lack of training data with explicit reasoning chains and the absence of mechanisms for active exploration and iterative refinement. To address these challenges, we propose AudioGenie-Reasoner (AGR), the first unified training-free multi-agent system that coordinates perception and reasoning over an evolving chain of textual evidence. Our key idea is a paradigm shift that transforms audio deep reasoning into complex text understanding task from a new perspective, thereby unlocking the full potential of large language models. Specifically, the design of AGR mimics the human coarse-to-fine cognitive process. It first transforms the input audio into a coarse text-based document. Then, we design a novel proactive iterative document refinement loop, featuring tool-augmented routes and specialized agents, to continuously search for missing information and augment the evidence chain in a coarse-to-fine manner until sufficient question-related information is gathered for making final predictions. Experimental results show that AGR achieves state-of-the-art (SOTA) performance over existing open-source audio deep reasoning models across various benchmarks. The code will be available at https://github.com/ryysayhi/AudioGenie-Reasoner.
>
---
#### [replaced 002] PAL: Probing Audio Encoders via LLMs - Audio Information Transfer into LLMs
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.10423v2](http://arxiv.org/pdf/2506.10423v2)**

> **作者:** Tony Alex; Wish Suharitdamrong; Sara Atito; Armin Mustafa; Philip J. B. Jackson; Imran Razzak; Muhammad Awais
>
> **备注:** 17 pages, 3 figures
>
> **摘要:** Integration of audio perception into large language models (LLMs) is an emerging research area for enabling machine listening applications, yet efficient transfer of rich audio semantics from audio encoders to LLMs remains underexplored. The most widely used integration paradigm projects the audio encoder output tokens into the LLM input space (e.g., via an MLP or a Q-Former), then prepends or inserts them to the text tokens. We refer to this generic scheme as Prepend to the LLM's input token space (PLITS) integration. We propose an efficient alternative, Lightweight Audio LLM Integration (LAL). LAL introduces audio representations solely via the attention mechanism within different layers of the LLM, bypassing its feedforward module. LAL encodes rich audio semantics at an appropriate level of abstraction for integration into different blocks of LLMs. Our design significantly reduces computational overhead compared to existing integration approaches. Observing with Whisper that the speech encoder benefits from PLITS integration, we propose an audio encoder aware approach for efficiently Probing Audio encoders via LLM (PAL), which employs PLITS integration for Whisper and LAL for general audio encoders. Under an identical training curriculum, LAL consistently maintains performance or outperforms existing integration approaches across multiple base LLMs and tasks. For general audio tasks, LAL improvement is up to 30% over a strong PLITS baseline while reducing memory usage by up to 64.1% and increasing throughput by up to 247.5%. Furthermore, for general audio-music-speech LLM, PAL performs on par with a fully PLITS integration-based system but with substantially improved computational and memory efficiency. Project page: https://ta012.github.io/PAL/
>
---
#### [replaced 003] Universal Speech Token Learning via Low-Bitrate Neural Codec and Pretrained Representations
- **分类: cs.SD; cs.AI; eess.AS**

- **链接: [http://arxiv.org/pdf/2503.12115v2](http://arxiv.org/pdf/2503.12115v2)**

> **作者:** Xue Jiang; Xiulian Peng; Yuan Zhang; Yan Lu
>
> **备注:** Accepted by IEEE Journal of Selected Topics in Signal Processing(JSTSP)
>
> **摘要:** Current large speech language models are mainly based on semantic tokens from discretization of self-supervised learned representations and acoustic tokens from a neural codec, following a semantic-modeling and acoustic-synthesis paradigm. However, semantic tokens discard paralinguistic attributes of speakers that is important for natural spoken communication, while prompt-based acoustic synthesis from semantic tokens has limits in recovering paralinguistic details and suffers from robustness issues, especially when there are domain gaps between the prompt and the target. This paper unifies two types of tokens and proposes the UniCodec, a universal speech token learning that encapsulates all semantics of speech, including linguistic and paralinguistic information, into a compact and semantically-disentangled unified token. Such a unified token can not only benefit speech language models in understanding with paralinguistic hints but also help speech generation with high-quality output. A low-bitrate neural codec is leveraged to learn such disentangled discrete representations at global and local scales, with knowledge distilled from self-supervised learned features. Extensive evaluations on multilingual datasets demonstrate its effectiveness in generating natural, expressive and long-term consistent output quality with paralinguistic attributes well preserved in several speech processing tasks.
>
---
#### [replaced 004] MelCap: A Unified Single-Codebook Neural Codec for High-Fidelity Audio Compression
- **分类: cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.01903v2](http://arxiv.org/pdf/2510.01903v2)**

> **作者:** Jingyi Li; Zhiyuan Zhao; Yunfei Liu; Lijian Lin; Ye Zhu; Jiahao Wu; Qiuqiang Kong; Yu Li
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Neural audio codecs have recently emerged as powerful tools for high-quality and low-bitrate audio compression, leveraging deep generative models to learn latent representations of audio signals. However, existing approaches either rely on a single quantizer that only processes speech domain, or on multiple quantizers that are not well suited for downstream tasks. To address this issue, we propose MelCap, a unified "one-codebook-for-all" neural codec that effectively handles speech, music, and general sound. By decomposing audio reconstruction into two stages, our method preserves more acoustic details than previous single-codebook approaches, while achieving performance comparable to mainstream multi-codebook methods. In the first stage, audio is transformed into mel-spectrograms, which are compressed and quantized into compact single tokens using a 2D tokenizer. A perceptual loss is further applied to mitigate the over-smoothing artifacts observed in spectrogram reconstruction. In the second stage, a Vocoder recovers waveforms from the mel discrete tokens in a single forward pass, enabling real-time decoding. Both objective and subjective evaluations demonstrate that MelCap achieves quality on comparable to state-of-the-art multi-codebook codecs, while retaining the computational simplicity of a single-codebook design, thereby providing an effective representation for downstream tasks.
>
---
#### [replaced 005] MSR-Codec: A Low-Bitrate Multi-Stream Residual Codec for High-Fidelity Speech Generation with Information Disentanglement
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2509.13068v2](http://arxiv.org/pdf/2509.13068v2)**

> **作者:** Jingyu Li; Guangyan Zhang; Zhen Ye; Yiwen Guo
>
> **摘要:** Audio codecs are a critical component of modern speech generation systems. This paper introduces a low-bitrate, multi-scale residual codec that encodes speech into four distinct streams: semantic, timbre, prosody, and residual. This architecture achieves high-fidelity speech reconstruction at competitive low bitrates while demonstrating an inherent ability for information disentanglement. We construct a two-stage language model for text-to-speech (TTS) synthesis using this codec, which, despite its lightweight design and minimal data requirements, achieves a state-of-the-art Word Error Rate (WER) and superior speaker similarity compared to several larger models. Furthermore, the codec's design proves highly effective for voice conversion, enabling independent manipulation of speaker timbre and prosody. Our inference code, pre-trained models, and audio samples are available at https://github.com/herbertLJY/MSRCodec.
>
---
#### [replaced 006] Latent-Domain Predictive Neural Speech Coding
- **分类: cs.SD; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2207.08363v3](http://arxiv.org/pdf/2207.08363v3)**

> **作者:** Xue Jiang; Xiulian Peng; Huaying Xue; Yuan Zhang; Yan Lu
>
> **备注:** Accepted by IEEE/ACM TRANSACTIONS ON AUDIO, SPEECH, AND LANGUAGE PROCESSING (TASLP). Code and models are available at https://github.com/microsoft/TF-Codec
>
> **摘要:** Neural audio/speech coding has recently demonstrated its capability to deliver high quality at much lower bitrates than traditional methods. However, existing neural audio/speech codecs employ either acoustic features or learned blind features with a convolutional neural network for encoding, by which there are still temporal redundancies within encoded features. This paper introduces latent-domain predictive coding into the VQ-VAE framework to fully remove such redundancies and proposes the TF-Codec for low-latency neural speech coding in an end-to-end manner. Specifically, the extracted features are encoded conditioned on a prediction from past quantized latent frames so that temporal correlations are further removed. Moreover, we introduce a learnable compression on the time-frequency input to adaptively adjust the attention paid to main frequencies and details at different bitrates. A differentiable vector quantization scheme based on distance-to-soft mapping and Gumbel-Softmax is proposed to better model the latent distributions with rate constraint. Subjective results on multilingual speech datasets show that, with low latency, the proposed TF-Codec at 1 kbps achieves significantly better quality than Opus at 9 kbps, and TF-Codec at 3 kbps outperforms both EVS at 9.6 kbps and Opus at 12 kbps. Numerous studies are conducted to demonstrate the effectiveness of these techniques. Code and models are available at https://github.com/microsoft/TF-Codec.
>
---
#### [replaced 007] SAGE-Music: Low-Latency Symbolic Music Generation via Attribute-Specialized Key-Value Head Sharing
- **分类: cs.SD; cs.AI; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2510.00395v2](http://arxiv.org/pdf/2510.00395v2)**

> **作者:** Jiaye Tan; Haonan Luo; Linfeng Song; Shuaiqi Chen; Yishan Lyu; Zian Zhong; Roujia Wang; Daniel Jiang; Haoran Zhang; Jiaming Bai; Haoran Cheng; Q. Vera Liao; Hao-Wen Dong
>
> **备注:** Withdrawn after identifying that results in Section 5 require additional re-analysis before public dissemination
>
> **摘要:** Low-latency symbolic music generation is essential for real-time improvisation and human-AI co-creation. Existing transformer-based models, however, face a trade-off between inference speed and musical quality. Traditional acceleration techniques such as embedding pooling significantly degrade quality, while recently proposed Byte Pair Encoding (BPE) methods - though effective on single-track piano data - suffer large performance drops in multi-track settings, as revealed by our analysis. We propose Attribute-Specialized Key-Value Head Sharing (AS-KVHS), adapted to music's structured symbolic representation, achieving about 30% inference speedup with only a negligible (about 0.4%) quality drop in objective evaluations and slight improvements in subjective listening tests. Our main contributions are (1) the first systematic study of BPE's generalizability in multi-track symbolic music, and (2) the introduction of AS-KVHS for low-latency symbolic music generation. Beyond these, we also release SAGE-Music, an open-source benchmark that matches or surpasses state-of-the-art models in generation quality.
>
---
#### [replaced 008] FakeMark: Deepfake Speech Attribution With Watermarked Artifacts
- **分类: eess.AS**

- **链接: [http://arxiv.org/pdf/2510.12042v2](http://arxiv.org/pdf/2510.12042v2)**

> **作者:** Wanying Ge; Xin Wang; Junichi Yamagishi
>
> **摘要:** Deepfake speech attribution remains challenging for existing solutions. Classifier-based solutions often fail to generalize to domain-shifted samples, and watermarking-based solutions are easily compromised by distortions like codec compression or malicious removal attacks. To address these issues, we propose FakeMark, a novel watermarking framework that injects artifact-correlated watermarks associated with deepfake systems rather than pre-assigned bitstring messages. This design allows a detector to attribute the source system by leveraging both injected watermark and intrinsic deepfake artifacts, remaining effective even if one of these cues is elusive or removed. Experimental results show that FakeMark improves generalization to cross-dataset samples where classifier-based solutions struggle and maintains high accuracy under various distortions where conventional watermarking-based solutions fail.
>
---
#### [replaced 009] ASE: Practical Acoustic Speed Estimation Beyond Doppler via Sound Diffusion Field
- **分类: cs.HC; cs.NI; cs.SD; eess.AS; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.20142v2](http://arxiv.org/pdf/2412.20142v2)**

> **作者:** Sheng Lyu; Chenshu Wu
>
> **备注:** Accepted at IMWUT'25
>
> **摘要:** Passive human speed estimation plays a critical role in acoustic sensing. Despite extensive study, existing systems, however, suffer from various limitations: First, the channel measurement rate proves inadequate to estimate high moving speeds. Second, previous acoustic speed estimation exploits Doppler Frequency Shifts (DFS) created by moving targets and relies on microphone arrays, making them only capable of sensing the radial speed within a constrained distance. To overcome these issues, we present ASE, an accurate and robust Acoustic Speed Estimation system on a single commodity microphone. We propose a novel Orthogonal Time-Delayed Multiplexing (OTDM) scheme for acoustic channel estimation at a high rate that was previously infeasible, making it possible to estimate high speeds. We then model the sound propagation from a unique perspective of the acoustic diffusion field, and infer the speed from the acoustic spatial distribution, a completely different way of thinking about speed estimation beyond prior DFS-based approaches. We further develop novel techniques for motion detection and signal enhancement to deliver a robust and practical system. We implement and evaluate ASE through extensive real-world experiments. Our results show that ASE reliably tracks walking speed, independently of target location and direction, with a mean error of 0.13 m/s, a reduction of 2.5x from DFS, and a detection rate of 97.4% for large coverage, e.g., free walking in a 4m x 4m room. We believe ASE pushes acoustic speed estimation beyond the conventional DFS-based paradigm and inspires exciting research in acoustic sensing. Code is available at https://github.com/aiot-lab/ASE.
>
---
