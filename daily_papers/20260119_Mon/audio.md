# 音频 cs.SD;  eess.AS

- **最新发布 6 篇**

- **更新 5 篇**

## 最新发布

#### [new 001] Unifying Speech Recognition, Synthesis and Conversion with Autoregressive Transformers
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文提出GPA，一个统一的音频基础模型，解决传统语音系统任务分离导致的效率低、扩展性差问题。通过共享音频token空间，实现语音识别、合成与转换的统一处理。**

- **链接: [https://arxiv.org/pdf/2601.10770v1](https://arxiv.org/pdf/2601.10770v1)**

> **作者:** Runyuan Cai; Yu Lin; Yiming Wang; Chunlin Fu; Xiaodong Zeng
>
> **摘要:** Traditional speech systems typically rely on separate, task-specific models for text-to-speech (TTS), automatic speech recognition (ASR), and voice conversion (VC), resulting in fragmented pipelines that limit scalability, efficiency, and cross-task generalization. In this paper, we present General-Purpose Audio (GPA), a unified audio foundation model that integrates multiple core speech tasks within a single large language model (LLM) architecture. GPA operates on a shared discrete audio token space and supports instruction-driven task induction, enabling a single autoregressive model to flexibly perform TTS, ASR, and VC without architectural modifications. This unified design combines a fully autoregressive formulation over discrete speech tokens, joint multi-task training across speech domains, and a scalable inference pipeline that achieves high concurrency and throughput. The resulting model family supports efficient multi-scale deployment, including a lightweight 0.3B-parameter variant optimized for edge and resource-constrained environments. Together, these design choices demonstrate that a unified autoregressive architecture can achieve competitive performance across diverse speech tasks while remaining viable for low-latency, practical deployment.
>
---
#### [new 002] DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
- **分类: cs.SD; cs.AI; eess.AS**

- **简介: 该论文属于语音建模任务，旨在解决语音中语义与声学特征的分离问题。提出DSA-Tokenizer，通过优化约束分离语义和声学token，并采用流匹配解码器提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.09239v2](https://arxiv.org/pdf/2601.09239v2)**

> **作者:** Hanlin Zhang; Daxin Tan; Dehua Tao; Xiao Chen; Haochen Tan; Yunhe Li; Yuchen Cao; Jianping Wang; Linqi Song
>
> **备注:** Submit to ACL ARR 2026 Jaunary
>
> **摘要:** Speech tokenizers serve as the cornerstone of discrete Speech Large Language Models (Speech LLMs). Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose DSA-Tokenizer, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. To eliminate rigid length constraints between the two sequences, we introduce a hierarchical Flow-Matching decoder that further improve speech generation quality. Furthermore, We employ a joint reconstruction-recombination training strategy to enforce this separation. DSA-Tokenizer enables high fidelity reconstruction and flexible recombination through robust disentanglement, facilitating controllable generation in speech LLMs. Our analysis highlights disentangled tokenization as a pivotal paradigm for future speech modeling. Audio samples are avaialble at https://anonymous.4open.science/w/DSA_Tokenizer_demo/. The code and model will be made publicly available after the paper has been accepted.
>
---
#### [new 003] Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings
- **分类: cs.SD; cs.IR; cs.LG**

- **简介: 该论文属于音乐版本识别任务，旨在高效准确地检索同一首歌曲的不同演绎版本。通过结合歌词信息，提出LIVI方法，在保证精度的同时提升计算效率。**

- **链接: [https://arxiv.org/pdf/2601.11262v1](https://arxiv.org/pdf/2601.11262v1)**

> **作者:** Joanne Affolter; Benjamin Martin; Elena V. Epure; Gabriel Meseguer-Brocal; Frédéric Kaplan
>
> **备注:** Published at ECIR 2026 (European Conference of Information Retrieval)
>
> **摘要:** Music Cover Retrieval, also known as Version Identification, aims to recognize distinct renditions of the same underlying musical work, a task central to catalog management, copyright enforcement, and music retrieval. State-of-the-art approaches have largely focused on harmonic and melodic features, employing increasingly complex audio pipelines designed to be invariant to musical attributes that often vary widely across covers. While effective, these methods demand substantial training time and computational resources. By contrast, lyrics constitute a strong invariant across covers, though their use has been limited by the difficulty of extracting them accurately and efficiently from polyphonic audio. Early methods relied on simple frameworks that limited downstream performance, while more recent systems deliver stronger results but require large models integrated within complex multimodal architectures. We introduce LIVI (Lyrics-Informed Version Identification), an approach that seeks to balance retrieval accuracy with computational efficiency. First, LIVI leverages supervision from state-of-the-art transcription and text embedding models during training to achieve retrieval accuracy on par with--or superior to--harmonic-based systems. Second, LIVI remains lightweight and efficient by removing the transcription step at inference, challenging the dominance of complexity-heavy pipelines.
>
---
#### [new 004] FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音对话系统任务，解决个性化语音克隆与低延迟交互问题。提出Chroma 1.0模型，实现高保真语音合成与实时对话。**

- **链接: [https://arxiv.org/pdf/2601.11141v1](https://arxiv.org/pdf/2601.11141v1)**

> **作者:** Tanyu Chen; Tairan Chen; Kai Shen; Zhenghua Bao; Zhihui Zhang; Man Yuan; Yi Shi
>
> **摘要:** Recent end-to-end spoken dialogue systems leverage speech tokenizers and neural audio codecs to enable LLMs to operate directly on discrete speech representations. However, these models often exhibit limited speaker identity preservation, hindering personalized voice interaction. In this work, we present Chroma 1.0, the first open-source, real-time, end-to-end spoken dialogue model that achieves both low-latency interaction and high-fidelity personalized voice cloning. Chroma achieves sub-second end-to-end latency through an interleaved text-audio token schedule (1:2) that supports streaming generation, while maintaining high-quality personalized voice synthesis across multi-turn conversations. Our experimental results demonstrate that Chroma achieves a 10.96% relative improvement in speaker similarity over the human baseline, with a Real-Time Factor (RTF) of 0.43, while maintaining strong reasoning and dialogue capabilities. Our code and models are publicly available at https://github.com/FlashLabs-AI-Corp/FlashLabs-Chroma and https://huggingface.co/FlashLabs/Chroma-4B .
>
---
#### [new 005] SonicBench: Dissecting the Physical Perception Bottleneck in Large Audio Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频理解任务，旨在解决大音频语言模型在物理属性感知上的不足。通过构建SonicBench基准，评估模型在音高、响度等属性上的表现，发现其基础听觉理解存在缺陷。**

- **链接: [https://arxiv.org/pdf/2601.11039v1](https://arxiv.org/pdf/2601.11039v1)**

> **作者:** Yirong Sun; Yanjun Chen; Xin Qiu; Gang Zhang; Hongyu Chen; Daokuan Wu; Chengming Li; Min Yang; Dawei Zhu; Wei Zhang; Xiaoyu Shen
>
> **摘要:** Large Audio Language Models (LALMs) excel at semantic and paralinguistic tasks, yet their ability to perceive the fundamental physical attributes of audio such as pitch, loudness, and spatial location remains under-explored. To bridge this gap, we introduce SonicBench, a psychophysically grounded benchmark that systematically evaluates 12 core physical attributes across five perceptual dimensions. Unlike previous datasets, SonicBench uses a controllable generation toolbox to construct stimuli for two complementary paradigms: recognition (absolute judgment) and comparison (relative judgment). This design allows us to probe not only sensory precision but also relational reasoning capabilities, a domain where humans typically exhibit greater proficiency. Our evaluation reveals a substantial deficiency in LALMs' foundational auditory understanding; most models perform near random guessing and, contrary to human patterns, fail to show the expected advantage on comparison tasks. Furthermore, explicit reasoning yields minimal gains. However, our linear probing analysis demonstrates crucially that frozen audio encoders do successfully capture these physical cues (accuracy at least 60%), suggesting that the primary bottleneck lies in the alignment and decoding stages, where models fail to leverage the sensory signals they have already captured.
>
---
#### [new 006] WenetSpeech-Wu: Datasets, Benchmarks, and Models for a Unified Chinese Wu Dialect Speech Processing Ecosystem
- **分类: cs.SD**

- **简介: 该论文针对吴语语音处理，解决数据与评估基准不足的问题，构建了大规模数据集和基准，发布了开源模型，推动方言语音技术发展。**

- **链接: [https://arxiv.org/pdf/2601.11027v1](https://arxiv.org/pdf/2601.11027v1)**

> **作者:** Chengyou Wang; Mingchen Shao; Jingbin Hu; Zeyu Zhu; Hongfei Xue; Bingshen Mu; Xin Xu; Xingyi Duan; Binbin Zhang; Pengcheng Zhu; Chuang Ding; Xiaojun Zhang; Hui Bu; Lei Xie
>
> **摘要:** Speech processing for low-resource dialects remains a fundamental challenge in developing inclusive and robust speech technologies. Despite its linguistic significance and large speaker population, the Wu dialect of Chinese has long been hindered by the lack of large-scale speech data, standardized evaluation benchmarks, and publicly available models. In this work, we present WenetSpeech-Wu, the first large-scale, multi-dimensionally annotated open-source speech corpus for the Wu dialect, comprising approximately 8,000 hours of diverse speech data. Building upon this dataset, we introduce WenetSpeech-Wu-Bench, the first standardized and publicly accessible benchmark for systematic evaluation of Wu dialect speech processing, covering automatic speech recognition (ASR), Wu-to-Mandarin translation, speaker attribute prediction, speech emotion recognition, text-to-speech (TTS) synthesis, and instruction-following TTS (instruct TTS). Furthermore, we release a suite of strong open-source models trained on WenetSpeech-Wu, establishing competitive performance across multiple tasks and empirically validating the effectiveness of the proposed dataset. Together, these contributions lay the foundation for a comprehensive Wu dialect speech processing ecosystem, and we open-source proposed datasets, benchmarks, and models to support future research on dialectal speech intelligence.
>
---
## 更新

#### [replaced 001] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR
- **分类: cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决对话场景中ASR因冗余上下文导致的识别错误问题。提出MARS方法，通过多模态检索与选择，提升ASR准确性。**

- **链接: [https://arxiv.org/pdf/2508.01166v3](https://arxiv.org/pdf/2508.01166v3)**

> **作者:** Bingshen Mu; Hexin Liu; Hongfei Xue; Kun Wei; Lei Xie
>
> **备注:** AAAI 2026
>
> **摘要:** Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
>
---
#### [replaced 002] Data Standards in Audiology: A Mixed-Methods Exploration of Community Perspectives and Implementation Considerations
- **分类: cs.SD; eess.AS; physics.med-ph**

- **简介: 该论文属于数据标准化研究，旨在解决听力学数据标准不统一的问题。通过调查和专家讨论，提出标准化实施路径与挑战。**

- **链接: [https://arxiv.org/pdf/2505.04728v4](https://arxiv.org/pdf/2505.04728v4)**

> **作者:** Charlotte Vercammen; Antje Heinrich; Christophe Lesimple; Alessia Paglialonga; Jan-Willem A. Wasmann; Mareike Buhl
>
> **摘要:** Objective: This study addresses conceptual issues around data standardisation in audiology, and outlines steps toward achieving it. It reports a survey of the computational audiology community on their current understanding, needs, and preferences concerning data standards. Based on survey findings and a panel discussion, recommendations are made concerning moving forward with standardisation in audiology. Design: Mixed-methods: 1) review of existing standardisation efforts; 2) a survey of the computational audiology community; 3) expert panel discussion in a dedicated session at the 2024 Virtual Conference of Computational Audiology. Sample: Survey: 82 members of the global community; Panel discussion: five experts. Results: A prerequisite for any global audiology database are agreed data standards. Although many are familiar with the general idea, few know of existing initiatives, or have actively participated in them. Ninety percent of respondents expressed willingness to follow or contribute to standardisation efforts. The panel discussed relevant initiatives (e.g. OMOP, openEHR, Noah) and explored both challenges (around harmonisation) and opportunities (alignment with other medical fields and conversion among approaches). Conclusions: Combining conceptual discussion with stakeholder views, the study offers guidance for implementing interoperable data standards in audiology. It highlights community support, key issues to address, and suggests paths for future work.
>
---
#### [replaced 003] POWSM: A Phonetic Open Whisper-Style Speech Foundation Model
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出POWSM，一个统一的语音基础模型，解决多语音任务联合处理问题。它支持ASR、G2P、P2G等任务，提升低资源场景下的语音处理效果。**

- **链接: [https://arxiv.org/pdf/2510.24992v2](https://arxiv.org/pdf/2510.24992v2)**

> **作者:** Chin-Jou Li; Kalvin Chang; Shikhar Bharadwaj; Eunjung Yeo; Kwanghee Choi; Jian Zhu; David Mortensen; Shinji Watanabe
>
> **备注:** 18 pages, under review. Model available at https://huggingface.co/espnet/powsm
>
> **摘要:** Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.
>
---
#### [replaced 004] SuperEar: Eavesdropping on Mobile Voice Calls via Stealthy Acoustic Metamaterials
- **分类: cs.SD; cs.CR; eess.AS**

- **简介: 该论文属于隐私安全任务，旨在解决户外移动通话窃听问题。通过设计声学超材料系统SuperEar，实现远距离清晰捕获语音。**

- **链接: [https://arxiv.org/pdf/2501.15032v2](https://arxiv.org/pdf/2501.15032v2)**

> **作者:** Zhiyuan Ning; Zhanyong Tang; Juan He; Weizhi Meng; Yuntian Chen; Ji Zhang; Zheng Wang
>
> **摘要:** Acoustic eavesdropping is a privacy risk, but existing attacks rarely work in real outdoor situations where people make phone calls on the move. We present SuperEar, the first portable system that uses acoustic metamaterials to reliably capture conversations in these scenarios. We show that the threat is real as a practical prototype can be implemented to enhance faint signals, cover the full range of speech with a compact design, and reduce noise and distortion to produce clear audio. We show that SuperEar can be implemented from low-cost 3D-printed parts and off-the-shelf hardware. Experimental results show that SuperEar can recover phone call audio with a success rate of over 80% at distances of up to 4.6 m - more than twice the range of previous approaches. Our findings highlight a new class of privacy threats enabled by metamaterial technology that requires attention.
>
---
#### [replaced 005] What Makes a Good Speech Tokenizer for LLM-Centric Speech Generation? A Systematic Study
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文属于语音生成任务，旨在解决语音与文本对齐及高质量合成问题。通过研究语音分词器设计，提出多标记预测和说话人感知生成方法，提升合成质量与一致性。**

- **链接: [https://arxiv.org/pdf/2506.12537v3](https://arxiv.org/pdf/2506.12537v3)**

> **作者:** Xiaoran Fan; Zhichao Sun; Yangfan Gao; Jingfei Xiong; Hang Yan; Yifei Cao; Jiajun Sun; Shuo Li; Zhihao Zhang; Zhiheng Xi; Yuhao Zhou; Senjie Jin; Changhao Jiang; Junjie Ye; Ming Zhang; Rui Zheng; Zhenhua Han; Yunke Zhang; Demei Yan; Shaokang Dong; Tao Ji; Tao Gui
>
> **摘要:** Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the role of speech tokenizer designs in LLM-centric SLMs, augmented by speech heads and speaker modeling. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
>
---
