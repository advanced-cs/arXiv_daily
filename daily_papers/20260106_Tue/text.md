# 自然语言处理 cs.CL

- **最新发布 109 篇**

- **更新 103 篇**

## 最新发布

#### [new 001] Hidden State Poisoning Attacks against Mamba-based Language Models
- **分类: cs.CL**

- **简介: 该论文研究了针对Mamba模型的隐藏状态污染攻击（HiSPA），揭示其对抗脆弱性。任务为模型安全，解决问题为攻击方法与防御机制，工作包括构建基准测试、验证攻击效果及分析模型行为。**

- **链接: [https://arxiv.org/pdf/2601.01972v1](https://arxiv.org/pdf/2601.01972v1)**

> **作者:** Alexandre Le Mercier; Chris Develder; Thomas Demeester
>
> **备注:** 17 pages, 4 figures. Submitted to ACL 2026
>
> **摘要:** State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even a recent 52B hybrid SSM-Transformer model from the Jamba family collapses on RoBench25 under optimized HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. Finally, our interpretability study reveals patterns in Mamba's hidden layers during HiSPAs that could be used to build a HiSPA mitigation system. The full code and data to reproduce the experiments can be found at https://anonymous.4open.science/r/hispa_anonymous-5DB0.
>
---
#### [new 002] Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于多语言对话语音识别任务，旨在解决LLM与端到端模型性能差距问题。通过融合语音编码器与LLM提升识别效果，提出交叉注意力融合机制，取得良好结果。**

- **链接: [https://arxiv.org/pdf/2601.01461v1](https://arxiv.org/pdf/2601.01461v1)**

> **作者:** Yuxiang Mei; Dongxing Xu; Jiaen Liang; Yanhua Long
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
>
---
#### [new 003] LANCET: Neural Intervention via Structural Entropy for Mitigating Faithfulness Hallucinations in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的忠实性幻觉问题。通过结构熵分析，定位并阻断幻觉传播路径，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.01401v1](https://arxiv.org/pdf/2601.01401v1)**

> **作者:** Chenxu Wang; Chaozhuo Li; Pengbo Wang; Litian Zhang; Songyang Liu; Ji Qi; Jiahui Hu; Yushan Cai; Hao Zhao; Rui Pu
>
> **摘要:** Large Language Models have revolutionized information processing, yet their reliability is severely compromised by faithfulness hallucinations. While current approaches attempt to mitigate this issue through node-level adjustments or coarse suppression, they often overlook the distributed nature of neural information, leading to imprecise interventions. Recognizing that hallucinations propagate through specific forward transmission pathways like an infection, we aim to surgically block this flow using precise structural analysis. To leverage this, we propose Lancet, a novel framework that achieves precise neural intervention by leveraging structural entropy and hallucination difference ratios. Lancet first locates hallucination-prone neurons via gradient-driven contrastive analysis, then maps their propagation pathways by minimizing structural entropy, and finally implements a hierarchical intervention strategy that preserves general model capabilities. Comprehensive evaluations across hallucination benchmark datasets demonstrate that Lancet significantly outperforms state-of-the-art methods, validating the effectiveness of our surgical approach to neural intervention.
>
---
#### [new 004] ks-lit-3m: A 3.1 million word kashmiri text dataset for large language model pretraining
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对克什米尔语语言模型预训练数据不足的问题，构建了310万词的KS-LIT-3M数据集，以提升模型在该语言上的表现。**

- **链接: [https://arxiv.org/pdf/2601.01091v1](https://arxiv.org/pdf/2601.01091v1)**

> **作者:** Haq Nawaz Malik
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable fluency across high-resource languages yet consistently fail to generate coherent text in Kashmiri, a language spoken by approximately seven million people. This performance disparity stems not from inherent model limitations but from a critical scarcity of high-quality training data. Decades of Kashmiri literature remain inaccessible to modern NLP pipelines due to their encoding in the proprietary InPage desktop publishing format. This paper introduces KS-LIT-3M, a curated corpus of 3.1 million words (16.4 million characters) specifically designed for pretraining language models on Kashmiri. The dataset is structured as a single continuous linear text stream, optimized for causal language model training where models learn to predict subsequent tokens from preceding context. The corpus was constructed through the development of a specialized InPage-to-Unicode converter, followed by rigorous preprocessing including English contamination removal, character normalization, and quality validation. Encompassing 131,607 unique words drawn from diverse genres including literary works, journalistic writing, academic texts, and religious scholarship, KS-LIT-3M addresses a fundamental resource gap for Kashmiri language technology. The dataset is released under the CC-BY-4.0 license to facilitate research in Kashmiri natural language processing.
>
---
#### [new 005] Deferred Commitment Decoding for Diffusion Language Models with Confidence-Aware Sliding Windows
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型生成任务，解决扩散模型中因块边界导致的上下文截断问题。提出DCD方法，通过置信度滑动窗口延迟高不确定性token的提交，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2601.02076v1](https://arxiv.org/pdf/2601.02076v1)**

> **作者:** Yingte Shu; Yuchuan Tian; Chao Xu; Yunhe Wang; Hanting Chen
>
> **摘要:** Diffusion language models (DLMs) have recently emerged as a strong alternative to autoregressive models by enabling parallel text generation. To improve inference efficiency and KV-cache compatibility, prior work commonly adopts block-based diffusion, decoding tokens block by block. However, this paradigm suffers from a structural limitation that we term Boundary-Induced Context Truncation (BICT): undecoded tokens near block boundaries are forced to commit without access to nearby future context, even when such context could substantially reduce uncertainty. This limitation degrades decoding confidence and generation quality, especially for tasks requiring precise reasoning, such as mathematical problem solving and code generation. We propose Deferred Commitment Decoding (DCD), a novel, training-free decoding strategy that mitigates this issue. DCD maintains a confidence-aware sliding window over masked tokens, resolving low-uncertainty tokens early while deferring high-uncertainty tokens until sufficient contextual evidence becomes available. This design enables effective bidirectional information flow within the decoding window without sacrificing efficiency. Extensive experiments across multiple diffusion language models, benchmarks, and caching configurations show that DCD improves generation accuracy by 1.39% with comparable time on average compared to fixed block-based diffusion methods, with the most significant improvement reaching 9.0%. These results demonstrate that deferring token commitment based on uncertainty is a simple yet effective principle for improving both the quality and efficiency of diffusion language model decoding.
>
---
#### [new 006] Stylometry Analysis of Human and Machine Text for Academic Integrity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分析任务，旨在解决学术不端问题，通过NLP技术实现文本来源识别与风格检测，提升学术诚信。**

- **链接: [https://arxiv.org/pdf/2601.01225v1](https://arxiv.org/pdf/2601.01225v1)**

> **作者:** Hezam Albaqami; Muhammad Asif Ayub; Nasir Ahmad; Yaseen Ahmad; Mohammed M. Alqahtani; Abdullah M. Algamdi; Almoaid A. Owaidah; Kashif Ahmad
>
> **备注:** 16 pages, 9 tables, 3 figures
>
> **摘要:** This work addresses critical challenges to academic integrity, including plagiarism, fabrication, and verification of authorship of educational content, by proposing a Natural Language Processing (NLP)-based framework for authenticating students' content through author attribution and style change detection. Despite some initial efforts, several aspects of the topic are yet to be explored. In contrast to existing solutions, the paper provides a comprehensive analysis of the topic by targeting four relevant tasks, including (i) classification of human and machine text, (ii) differentiating in single and multi-authored documents, (iii) author change detection within multi-authored documents, and (iv) author recognition in collaboratively produced documents. The solutions proposed for the tasks are evaluated on two datasets generated with Gemini using two different prompts, including a normal and a strict set of instructions. During experiments, some reduction in the performance of the proposed solutions is observed on the dataset generated through the strict prompt, demonstrating the complexities involved in detecting machine-generated text with cleverly crafted prompts. The generated datasets, code, and other relevant materials are made publicly available on GitHub, which are expected to provide a baseline for future research in the domain.
>
---
#### [new 007] DHI: Leveraging Diverse Hallucination Induction for Enhanced Contrastive Factuality Control in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于事实性控制任务，旨在解决大语言模型中的幻觉问题。通过DHI框架生成多样化幻觉，提升对比解码效果。**

- **链接: [https://arxiv.org/pdf/2601.01156v1](https://arxiv.org/pdf/2601.01156v1)**

> **作者:** Jiani Guo; Xiangke Zeng; Jie Wu; Zuchao Li
>
> **备注:** ICONIP 2025
>
> **摘要:** Large language models (LLMs) frequently produce inaccurate or fabricated information, known as "hallucinations," which compromises their reliability. Existing approaches often train an "Evil LLM" to deliberately generate hallucinations on curated datasets, using these induced hallucinations to guide contrastive decoding against a reliable "positive model" for hallucination mitigation. However, this strategy is limited by the narrow diversity of hallucinations induced, as Evil LLMs trained on specific error types tend to reproduce only these particular patterns, thereby restricting their overall effectiveness. To address these limitations, we propose DHI (Diverse Hallucination Induction), a novel training framework that enables the Evil LLM to generate a broader range of hallucination types without relying on pre-annotated hallucination data. DHI employs a modified loss function that down-weights the generation of specific factually correct tokens, encouraging the Evil LLM to produce diverse hallucinations at targeted positions while maintaining overall factual content. Additionally, we introduce a causal attention masking adaptation to reduce the impact of this penalization on the generation of subsequent tokens. During inference, we apply an adaptive rationality constraint that restricts contrastive decoding to tokens where the positive model exhibits high confidence, thereby avoiding unnecessary penalties on factually correct tokens. Extensive empirical results show that DHI achieves significant performance gains over other contrastive decoding-based approaches across multiple hallucination benchmarks.
>
---
#### [new 008] FLOP-Efficient Training: Early Stopping Based on Test-Time Compute Awareness
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型训练优化任务，解决如何减少训练FLOPs同时保持或提升模型精度的问题。通过引入TTC-aware训练和早停算法，实现更高效的模型训练。**

- **链接: [https://arxiv.org/pdf/2601.01332v1](https://arxiv.org/pdf/2601.01332v1)**

> **作者:** Hossam Amer; Maryam Dialameh; Hossein Rajabzadeh; Walid Ahmed; Weiwei Zhang; Yang Liu
>
> **摘要:** Scaling training compute, measured in FLOPs, has long been shown to improve the accuracy of large language models, yet training remains resource-intensive. Prior work shows that increasing test-time compute (TTC)-for example through iterative sampling-can allow smaller models to rival or surpass much larger ones at lower overall cost. We introduce TTC-aware training, where an intermediate checkpoint and a corresponding TTC configuration can together match or exceed the accuracy of a fully trained model while requiring substantially fewer training FLOPs. Building on this insight, we propose an early stopping algorithm that jointly selects a checkpoint and TTC configuration to minimize training compute without sacrificing accuracy. To make this practical, we develop an efficient TTC evaluation method that avoids exhaustive search, and we formalize a break-even bound that identifies when increased inference compute compensates for reduced training compute. Experiments demonstrate up to 92\% reductions in training FLOPs while maintaining and sometimes remarkably improving accuracy. These results highlight a new perspective for balancing training and inference compute in model development, enabling faster deployment cycles and more frequent model refreshes. Codes will be publicly released.
>
---
#### [new 009] Power-of-Two Quantization-Aware-Training (PoT-QAT) in Large Language Models (LLMs)
- **分类: cs.CL; eess.SP**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型在边缘设备上的部署问题。通过功率二量化和量化感知训练，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.02298v1](https://arxiv.org/pdf/2601.02298v1)**

> **作者:** Mahmoud Elgenedy
>
> **摘要:** In Large Language Models (LLMs), the number of parameters has grown exponentially in the past few years, e.g., from 1.5 billion parameters in GPT-2 to 175 billion in GPT-3 to possibly more than trillion in higher versions. This raises a significant challenge for implementation, especially for Edge devices. Unlike cloud computing, memory and processing power for Edge devices are very limited, which necessitates developing novel ideas to make such applications feasible. In this work, we investigate compressing weights with a special quantization that limits numbers to only power-of-two (PoT). This helps save a huge amount of memory as only exponents need to be stored, more importantly, it significantly reduces processing power by replacing costly multiplication with low cost bit shifting. To overcome performance loss due to this strict quantization, we investigate Quantization Aware Training (QAT) to enhance performance through additional training. Results on GPT-2 124M show a major enhancement for quantized PoT model after additional training, with a perplexity enhancement of 66% and BERT-Score loss to baseline GPT-2 of 1%. The memory saving is estimated to be 87.5% while the inference speed is expected to be 3-10x faster with PoT quantization versus full-precision.
>
---
#### [new 010] pdfQA: Diverse, Challenging, and Realistic Question Answering over PDFs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出pdfQA，一个用于PDF文档问答的多领域数据集，解决通用QA任务中的挑战。通过真实与合成数据，评估问答系统在不同复杂度下的表现。**

- **链接: [https://arxiv.org/pdf/2601.02285v1](https://arxiv.org/pdf/2601.02285v1)**

> **作者:** Tobias Schimanski; Imene Kolli; Jingwei Ni; Yu Fan; Ario Saeid Vaghefi; Elliott Ash; Markus Leippold
>
> **摘要:** PDFs are the second-most used document type on the internet (after HTML). Yet, existing QA datasets commonly start from text sources or only address specific domains. In this paper, we present pdfQA, a multi-domain 2K human-annotated (real-pdfQA) and 2K synthetic dataset (syn-pdfQA) differentiating QA pairs in ten complexity dimensions (e.g., file type, source modality, source position, answer type). We apply and evaluate quality and difficulty filters on both datasets, obtaining valid and challenging QA pairs. We answer the questions with open-source LLMs, revealing existing challenges that correlate with our complexity dimensions. pdfQA presents a basis for end-to-end QA pipeline evaluation, testing diverse skill sets and local optimizations (e.g., in information retrieval or parsing).
>
---
#### [new 011] Tackling the Inherent Difficulty of Noise Filtering in RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG中噪声过滤难题。针对检索内容中的无关信息难以识别的问题，提出新的微调方法提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.01896v1](https://arxiv.org/pdf/2601.01896v1)**

> **作者:** Jingyu Liu; Jiaen Lin; Yong Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a widely adopted approach to enhance Large Language Models (LLMs) by incorporating external knowledge and reducing hallucinations. However, noisy or irrelevant documents are often introduced during RAG, potentially degrading performance and even causing hallucinated outputs. While various methods have been proposed to filter out such noise, we argue that identifying irrelevant information from retrieved content is inherently difficult and limited number of transformer layers can hardly solve this. Consequently, retrievers fail to filter out irrelevant documents entirely. Therefore, LLMs must be robust against such noise, but we demonstrate that standard fine-tuning approaches are often ineffective in enabling the model to selectively utilize relevant information while ignoring irrelevant content due to the structural constraints of attention patterns. To address this, we propose a novel fine-tuning method designed to enhance the model's ability to distinguish between relevant and irrelevant information within retrieved documents. Extensive experiments across multiple benchmarks show that our approach significantly improves the robustness and performance of LLMs.
>
---
#### [new 012] From Emotion Classification to Emotional Reasoning: Enhancing Emotional Intelligence in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于情感推理任务，旨在提升小规模语言模型的情感智能。通过生成合成情感数据并微调模型，有效增强了模型在情感理解与意识方面的能力。**

- **链接: [https://arxiv.org/pdf/2601.01407v1](https://arxiv.org/pdf/2601.01407v1)**

> **作者:** Arjhun Sreedar; Rohan Pillay; Laukik Patade
>
> **备注:** 10 pages, 1 figure
>
> **摘要:** This work investigates whether synthetic emotional chain-of-thought data can improve the emotional reasoning abilities of smaller open large language models (LLMs). We design a multi-agent generation pipeline that produces therapy-style conversations and converts them into structured emotion multiple-choice questions (MCQs) with explanations. We propose that fine-tuning a variety of 7B models on this dataset should yield substantial gains in emotional understanding and emotional awareness on EmoBench-style evaluations, suggesting that emotional reasoning can be induced without architectural changes. Our results demonstrate that fine-tuned Mistral 7B achieves EU improvements from 10.5 to 20.5 and EA improvements from 40.5 to 60.0, validating the effectiveness of synthetic emotional reasoning data for enhancing model capabilities in nuanced emotional tasks.
>
---
#### [new 013] Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents
- **分类: cs.CL**

- **简介: 该论文属于大语言模型代理任务，解决长周期推理中的记忆管理问题。提出Agentic Memory框架，统一管理长期和短期记忆，提升任务性能与上下文效率。**

- **链接: [https://arxiv.org/pdf/2601.01885v1](https://arxiv.org/pdf/2601.01885v1)**

> **作者:** Yi Yu; Liuyi Yao; Yuexiang Xie; Qingquan Tan; Jiaqi Feng; Yaliang Li; Libing Wu
>
> **摘要:** Large language model (LLM) agents face fundamental limitations in long-horizon reasoning due to finite context windows, making effective memory management critical. Existing methods typically handle long-term memory (LTM) and short-term memory (STM) as separate components, relying on heuristics or auxiliary controllers, which limits adaptability and end-to-end optimization. In this paper, we propose Agentic Memory (AgeMem), a unified framework that integrates LTM and STM management directly into the agent's policy. AgeMem exposes memory operations as tool-based actions, enabling the LLM agent to autonomously decide what and when to store, retrieve, update, summarize, or discard information. To train such unified behaviors, we propose a three-stage progressive reinforcement learning strategy and design a step-wise GRPO to address sparse and discontinuous rewards induced by memory operations. Experiments on five long-horizon benchmarks demonstrate that AgeMem consistently outperforms strong memory-augmented baselines across multiple LLM backbones, achieving improved task performance, higher-quality long-term memory, and more efficient context usage.
>
---
#### [new 014] Does Memory Need Graphs? A Unified Framework and Empirical Analysis for Long-Term Dialog Memory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话记忆任务，旨在分析图结构在长期对话记忆中的有效性。通过构建统一框架进行实验，发现性能差异多由基础设置决定，而非架构创新。**

- **链接: [https://arxiv.org/pdf/2601.01280v1](https://arxiv.org/pdf/2601.01280v1)**

> **作者:** Sen Hu; Yuxiang Wei; Jiaxin Ran; Zhiyuan Yao; Lei Zou
>
> **摘要:** Graph structures are increasingly used in dialog memory systems, but empirical findings on their effectiveness remain inconsistent, making it unclear which design choices truly matter. We present an experimental, system-oriented analysis of long-term dialog memory architectures. We introduce a unified framework that decomposes dialog memory systems into core components and supports both graph-based and non-graph approaches. Under this framework, we conduct controlled, stage-wise experiments on LongMemEval and HaluMem, comparing common design choices in memory representation, organization, maintenance, and retrieval. Our results show that many performance differences are driven by foundational system settings rather than specific architectural innovations. Based on these findings, we identify stable and reliable strong baselines for future dialog memory research.
>
---
#### [new 015] Confidence Estimation for LLMs in Multi-turn Interactions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的信心估计任务，旨在解决多轮对话中模型信心可靠评估的问题。工作包括建立评估框架、提出新指标，并验证现有方法的不足，提出改进方案。**

- **链接: [https://arxiv.org/pdf/2601.02179v1](https://arxiv.org/pdf/2601.02179v1)**

> **作者:** Caiqi Zhang; Ruihan Yang; Xiaochen Zhu; Chengzu Li; Tiancheng Hu; Yijiang River Dong; Deqing Yang; Nigel Collier
>
> **摘要:** While confidence estimation is a promising direction for mitigating hallucinations in Large Language Models (LLMs), current research dominantly focuses on single-turn settings. The dynamics of model confidence in multi-turn conversations, where context accumulates and ambiguity is progressively resolved, remain largely unexplored. Reliable confidence estimation in multi-turn settings is critical for many downstream applications, such as autonomous agents and human-in-the-loop systems. This work presents the first systematic study of confidence estimation in multi-turn interactions, establishing a formal evaluation framework grounded in two key desiderata: per-turn calibration and monotonicity of confidence as more information becomes available. To facilitate this, we introduce novel metrics, including a length-normalized Expected Calibration Error (InfoECE), and a new "Hinter-Guesser" paradigm for generating controlled evaluation datasets. Our experiments reveal that widely-used confidence techniques struggle with calibration and monotonicity in multi-turn dialogues. We propose P(Sufficient), a logit-based probe that achieves comparatively better performance, although the task remains far from solved. Our work provides a foundational methodology for developing more reliable and trustworthy conversational agents.
>
---
#### [new 016] FormationEval, an open multiple-choice benchmark for petroleum geoscience
- **分类: cs.CL; cs.AI; cs.LG; physics.geo-ph**

- **简介: 该论文提出FormationEval，一个用于评估语言模型在石油地质科学领域表现的开放多选题基准。旨在解决模型在该领域的评估问题，通过构建高质量数据集并测试多种模型性能。**

- **链接: [https://arxiv.org/pdf/2601.02158v1](https://arxiv.org/pdf/2601.02158v1)**

> **作者:** Almaz Ermilov
>
> **备注:** 24 pages, 8 figures, 10 tables; benchmark and code at https://github.com/AlmazErmilov/FormationEval-an-Open-Benchmark-for-Oil-Gas-Geoscience-MCQ-Evaluation
>
> **摘要:** This paper presents FormationEval, an open multiple-choice question benchmark for evaluating language models on petroleum geoscience and subsurface disciplines. The dataset contains 505 questions across seven domains including petrophysics, petroleum geology and reservoir engineering, derived from three authoritative sources using a reasoning model with detailed instructions and a concept-based approach that avoids verbatim copying of copyrighted text. Each question includes source metadata to support traceability and audit. The evaluation covers 72 models from major providers including OpenAI, Anthropic, Google, Meta and open-weight alternatives. The top performers achieve over 97\% accuracy, with Gemini 3 Pro Preview reaching 99.8\%, while tier and domain gaps persist. Among open-weight models, GLM-4.7 leads at 98.6\%, with several DeepSeek, Llama, Qwen and Mistral models also exceeding 93\%. The performance gap between open-weight and closed models is narrower than expected, with several lower-cost open-weight models exceeding 90\% accuracy. Petrophysics emerges as the most challenging domain across all models, while smaller models show wider performance variance. Residual length bias in the dataset (correct answers tend to be longer) is documented along with bias mitigation strategies applied during construction. The benchmark, evaluation code and results are publicly available.
>
---
#### [new 017] DermoGPT: Open Weights and Open Data for Morphology-Grounded Dermatological Reasoning MLLMs
- **分类: cs.CL**

- **简介: 该论文提出DermoGPT，解决 dermatology 领域中MLLM数据不足、任务覆盖窄及缺乏临床监督的问题。构建了DermoInstruct、DermoBench，并通过MAVIC和CCT提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.01868v1](https://arxiv.org/pdf/2601.01868v1)**

> **作者:** Jinghan Ru; Siyuan Yan; Yuguo Yin; Yuexian Zou; Zongyuan Ge
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promise for medical applications, yet progress in dermatology lags due to limited training data, narrow task coverage, and lack of clinically-grounded supervision that mirrors expert diagnostic workflows. We present a comprehensive framework to address these gaps. First, we introduce DermoInstruct, a large-scale morphology-anchored instruction corpus comprising 211,243 images and 772,675 trajectories across five task formats, capturing the complete diagnostic pipeline from morphological observation and clinical reasoning to final diagnosis. Second, we establish DermoBench, a rigorous benchmark evaluating 11 tasks across four clinical axes: Morphology, Diagnosis, Reasoning, and Fairness, including a challenging subset of 3,600 expert-verified open-ended instances and human performance baselines. Third, we develop DermoGPT, a dermatology reasoning MLLM trained via supervised fine-tuning followed by our Morphologically-Anchored Visual-Inference-Consistent (MAVIC) reinforcement learning objective, which enforces consistency between visual observations and diagnostic conclusions. At inference, we deploy Confidence-Consistency Test-time adaptation (CCT) for robust predictions. Experiments show DermoGPT significantly outperforms 16 representative baselines across all axes, achieving state-of-the-art performance while substantially narrowing the human-AI gap. DermoInstruct, DermoBench and DermoGPT will be made publicly available at https://github.com/mendicant04/DermoGPT upon acceptance.
>
---
#### [new 018] Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Hallucination Risks in Long-Context LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究长文本中信息提取、逻辑推理与幻觉风险，探讨事实分布和提示对模型的影响，旨在提升长上下文下LLM的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.02023v1](https://arxiv.org/pdf/2601.02023v1)**

> **作者:** Amirali Ebrahimzadeh; Seyyed M. Salili
>
> **备注:** 25 pages, 8 figures, 3 tables
>
> **摘要:** Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear how reliably they extract and infer information at scale. Performance varies with context length and strongly interacts with how information is distributed in real-world corpora. Motivated by these observations, we study how fact placement, corpus-level fact distributions, and Don't Make It Up prompts influence model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study considers both positional effects and realistic distributions of evidence across long contexts, as well as prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance varies substantially across models: some show severe degradation under realistic conditions, while others remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG), our results suggest many failures stem from ineffective context utilization. Models often struggle to identify and prioritize relevant information even when it is present. These findings have direct practical implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents into LLM prompts. Effective context length and model-specific robustness to long contexts are therefore critical for reliable LLM deployment in research and business.
>
---
#### [new 019] JMedEthicBench: A Multi-Turn Conversational Benchmark for Evaluating Medical Safety in Japanese Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗安全评估任务，旨在解决LLM在日语医疗对话中的安全问题。构建了首个多轮对话基准JMedEthicBench，测试模型安全性并发现多轮交互中的安全下降问题。**

- **链接: [https://arxiv.org/pdf/2601.01627v1](https://arxiv.org/pdf/2601.01627v1)**

> **作者:** Junyu Liu; Zirui Li; Qian Niu; Zequn Zhang; Yue Xun; Wenlong Hou; Shujun Wang; Yusuke Iwasawa; Yutaka Matsuo; Kan Hatakeyama-Sato
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in healthcare field, it becomes essential to carefully evaluate their medical safety before clinical use. However, existing safety benchmarks remain predominantly English-centric, and test with only single-turn prompts despite multi-turn clinical consultations. To address these gaps, we introduce JMedEthicBench, the first multi-turn conversational benchmark for evaluating medical safety of LLMs for Japanese healthcare. Our benchmark is based on 67 guidelines from the Japan Medical Association and contains over 50,000 adversarial conversations generated using seven automatically discovered jailbreak strategies. Using a dual-LLM scoring protocol, we evaluate 27 models and find that commercial models maintain robust safety while medical-specialized models exhibit increased vulnerability. Furthermore, safety scores decline significantly across conversation turns (median: 9.5 to 5.0, $p < 0.001$). Cross-lingual evaluation on both Japanese and English versions of our benchmark reveals that medical model vulnerabilities persist across languages, indicating inherent alignment limitations rather than language-specific factors. These findings suggest that domain-specific fine-tuning may accidentally weaken safety mechanisms and that multi-turn interactions represent a distinct threat surface requiring dedicated alignment strategies.
>
---
#### [new 020] From Failure to Mastery: Generating Hard Samples for Tool-use Agents
- **分类: cs.CL**

- **简介: 该论文属于工具使用代理训练数据生成任务，旨在解决现有数据过于简单的问题。通过构建动态API图生成困难样本，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2601.01498v1](https://arxiv.org/pdf/2601.01498v1)**

> **作者:** Bingguang Hao; Zengzhuang Xu; Yuntao Wen; Xinyi Xu; Yang Liu; Tong Zhao; Maolin Wang; Long Chen; Dong Wang; Yicheng Chen; Cunyin Peng; Xiangyu Zhao; Chenyi Zhuang; Ji Zhang
>
> **摘要:** The advancement of LLM agents with tool-use capabilities requires diverse and complex training corpora. Existing data generation methods, which predominantly follow a paradigm of random sampling and shallow generation, often yield simple and homogeneous trajectories that fail to capture complex, implicit logical dependencies. To bridge this gap, we introduce HardGen, an automatic agentic pipeline designed to generate hard tool-use training samples with verifiable reasoning. Firstly, HardGen establishes a dynamic API Graph built upon agent failure cases, from which it samples to synthesize hard traces. Secondly, these traces serve as conditional priors to guide the instantiation of modular, abstract advanced tools, which are subsequently leveraged to formulate hard queries. Finally, the advanced tools and hard queries enable the generation of verifiable complex Chain-of-Thought (CoT), with a closed-loop evaluation feedback steering the continuous refinement of the process. Extensive evaluations demonstrate that a 4B parameter model trained with our curated dataset achieves superior performance compared to several leading open-source and closed-source competitors (e.g., GPT-5.2, Gemini-3-Pro and Claude-Opus-4.5). Our code, models, and dataset will be open-sourced to facilitate future research.
>
---
#### [new 021] Routing by Analogy: kNN-Augmented Expert Assignment for Mixture-of-Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决MoE架构中路由机制在分布变化下表现脆弱的问题。通过引入kNN-MoE，利用相似案例记忆提升路由稳定性。**

- **链接: [https://arxiv.org/pdf/2601.02144v1](https://arxiv.org/pdf/2601.02144v1)**

> **作者:** Boxuan Lyu; Soichiro Murakami; Hidetaka Kamigaito; Peinan Zhang
>
> **摘要:** Mixture-of-Experts (MoE) architectures scale large language models efficiently by employing a parametric "router" to dispatch tokens to a sparse subset of experts. Typically, this router is trained once and then frozen, rendering routing decisions brittle under distribution shifts. We address this limitation by introducing kNN-MoE, a retrieval-augmented routing framework that reuses optimal expert assignments from a memory of similar past cases. This memory is constructed offline by directly optimizing token-wise routing logits to maximize the likelihood on a reference set. Crucially, we use the aggregate similarity of retrieved neighbors as a confidence-driven mixing coefficient, thus allowing the method to fall back to the frozen router when no relevant cases are found. Experiments show kNN-MoE outperforms zero-shot baselines and rivals computationally expensive supervised fine-tuning.
>
---
#### [new 022] K-EXAONE Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 本文介绍K-EXAONE，一个大型多语言语言模型，解决多语言支持与高效推理问题。采用专家混合架构，具备236B参数，支持六种语言，适用于工业与研究应用。**

- **链接: [https://arxiv.org/pdf/2601.01739v1](https://arxiv.org/pdf/2601.01739v1)**

> **作者:** Eunbi Choi; Kibong Choi; Seokhee Hong; Junwon Hwang; Hyojin Jeon; Hyunjik Jo; Joonkee Kim; Seonghwan Kim; Soyeon Kim; Sunkyoung Kim; Yireun Kim; Yongil Kim; Haeju Lee; Jinsik Lee; Kyungmin Lee; Sangha Park; Heuiyeen Yeen; Hwan Chang; Stanley Jungkyu Choi; Yejin Choi; Jiwon Ham; Kijeong Jeon; Geunyeong Jeong; Gerrard Jeongwon Jo; Yonghwan Jo; Jiyeon Jung; Naeun Kang; Dohoon Kim; Euisoon Kim; Hayeon Kim; Hyosang Kim; Hyunseo Kim; Jieun Kim; Minu Kim; Myoungshin Kim; Unsol Kim; Youchul Kim; YoungJin Kim; Chaeeun Lee; Chaeyoon Lee; Changhun Lee; Dahm Lee; Edward Hwayoung Lee; Honglak Lee; Jinsang Lee; Jiyoung Lee; Sangeun Lee; Seungwon Lim; Solji Lim; Woohyung Lim; Chanwoo Moon; Jaewoo Park; Jinho Park; Yongmin Park; Hyerin Seo; Wooseok Seo; Yongwoo Song; Sejong Yang; Sihoon Yang; Chang En Yea; Sihyuk Yi; Chansik Yoon; Dongkeun Yoon; Sangyeon Yoon; Hyeongu Yun
>
> **备注:** 29 pages
>
> **摘要:** This technical report presents K-EXAONE, a large-scale multilingual language model developed by LG AI Research. K-EXAONE is built on a Mixture-of-Experts architecture with 236B total parameters, activating 23B parameters during inference. It supports a 256K-token context window and covers six languages: Korean, English, Spanish, German, Japanese, and Vietnamese. We evaluate K-EXAONE on a comprehensive benchmark suite spanning reasoning, agentic, general, Korean, and multilingual abilities. Across these evaluations, K-EXAONE demonstrates performance comparable to open-weight models of similar size. K-EXAONE, designed to advance AI for a better life, is positioned as a powerful proprietary AI foundation model for a wide range of industrial and research applications.
>
---
#### [new 023] KOS-TL (Knowledge Operation System Type Logic)
- **分类: cs.CL; cs.LO**

- **简介: 该论文提出KOS-TL，解决知识系统静态与动态脱节问题，通过依赖类型理论构建可执行知识系统，确保逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2601.01143v1](https://arxiv.org/pdf/2601.01143v1)**

> **作者:** Peng Chen
>
> **摘要:** This paper introduces KOS-TL (Knowledge Operation System Type Logic), a novel constructive framework designed to provide a rigorous logical foundation for autonomous and executable knowledge systems. Traditional knowledge representation models often suffer from a gap between static symbolic logic and dynamic system execution. To bridge this divide, KOS-TL leverages Dependent Type Theory to unify data, logic, and proof into a singular computational substrate.The architecture of KOS-TL is organized into three hierarchical layers: the Core Layer, which defines the static type universe and constructive primitives; the Kernel Layer, which governs state evolution through an event-driven mechanism characterized by the triple $\langle Σ, \textsf{Ev}, Δ\rangle$; and the Runtime Layer, responsible for the bidirectional refinement of physical signals into logical evidence. We formally define the operational semantics of the system and prove key meta-theoretical properties, including Progress and Evolutionary Consistency, ensuring that the system remains logically self-consistent and free from stuck states during continuous state transitions.By integrating Davidsonian event semantics with Martin-Löf type theory, KOS-TL enables the construction of "proof-carrying knowledge," where every state change in the knowledge base is accompanied by a formal witness of its validity. We demonstrate the practical utility of this logic through application examples in industrial traceability and cross-border financial compliance. Our results suggest that KOS-TL provides a robust, formally verifiable basis for the next generation of intelligent, autonomous operating systems.
>
---
#### [new 024] Reasoning Over Recall: Evaluating the Efficacy of Generalist Architectures vs. Specialized Fine-Tunes in RAG-Based Mental Health Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于心理健康对话系统任务，旨在比较通用模型与专业微调模型在RAG框架下的效果，解决情感支持与准确性的平衡问题。通过实验发现通用模型在同环境下表现更优。**

- **链接: [https://arxiv.org/pdf/2601.01341v1](https://arxiv.org/pdf/2601.01341v1)**

> **作者:** Md Abdullah Al Kafi; Raka Moni; Sumit Kumar Banshal
>
> **摘要:** The deployment of Large Language Models (LLMs) in mental health counseling faces the dual challenges of hallucinations and lack of empathy. While the former may be mitigated by RAG (retrieval-augmented generation) by anchoring answers in trusted clinical sources, there remains an open question as to whether the most effective model under this paradigm would be one that is fine-tuned on mental health data, or a more general and powerful model that succeeds purely on the basis of reasoning. In this paper, we perform a direct comparison by running four open-source models through the same RAG pipeline using ChromaDB: two generalist reasoners (Qwen2.5-3B and Phi-3-Mini) and two domain-specific fine-tunes (MentalHealthBot-7B and TherapyBot-7B). We use an LLM-as-a-Judge framework to automate evaluation over 50 turns. We find a clear trend: the generalist models outperform the domain-specific ones in empathy (3.72 vs. 3.26, $p < 0.001$) in spite of being much smaller (3B vs. 7B), and all models perform well in terms of safety, but the generalist models show better contextual understanding and are less prone to overfitting as we observe in the domain-specific models. Overall, our results indicate that for RAG-based therapy systems, strong reasoning is more important than training on mental health-specific vocabulary; i.e. a well-reasoned general model would provide more empathetic and balanced support than a larger narrowly fine-tuned model, so long as the answer is already grounded in clinical evidence.
>
---
#### [new 025] Unsupervised Text Style Transfer for Controllable Intensity
- **分类: cs.CL**

- **简介: 该论文属于无监督文本风格迁移任务，旨在解决可控强度下的风格迁移难题。通过SFT-then-PPO方法优化大模型，提升风格区分能力。**

- **链接: [https://arxiv.org/pdf/2601.01060v1](https://arxiv.org/pdf/2601.01060v1)**

> **作者:** Shuhuan Gu; Wenbiao Tao; Xinchen Ma; Kangkang He; Ye Guo; Xiang Li; Yunshi Lan
>
> **摘要:** Unsupervised Text Style Transfer (UTST) aims to build a system to transfer the stylistic properties of a given text without parallel text pairs. Compared with text transfer between style polarities, UTST for controllable intensity is more challenging due to the subtle differences in stylistic features across different intensity levels. Faced with the challenges posed by the lack of parallel data and the indistinguishability between adjacent intensity levels, we propose a SFT-then-PPO paradigm to fine-tune an LLM. We first fine-tune the LLM with synthesized parallel data. Then, we further train the LLM with PPO, where the rewards are elaborately designed for distinguishing the stylistic intensity in hierarchical levels. Both the global and local stylistic features are considered to formulate the reward functions. The experiments on two UTST benchmarks showcase that both rewards have their advantages and applying them to LLM fine-tuning can effectively improve the performance of an LLM backbone based on various evaluation metrics. Even for close levels of intensity, we can still observe the noticeable stylistic difference between the generated text.
>
---
#### [new 026] Emergent Introspective Awareness in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型研究任务，探讨大模型是否具备自我反思能力。通过注入概念并观察模型反应，发现部分模型能识别内部状态，表明其具有一定的内省意识。**

- **链接: [https://arxiv.org/pdf/2601.01828v1](https://arxiv.org/pdf/2601.01828v1)**

> **作者:** Jack Lindsey
>
> **摘要:** We investigate whether large language models can introspect on their internal states. It is difficult to answer this question through conversation alone, as genuine introspection cannot be distinguished from confabulations. Here, we address this challenge by injecting representations of known concepts into a model's activations, and measuring the influence of these manipulations on the model's self-reported states. We find that models can, in certain scenarios, notice the presence of injected concepts and accurately identify them. Models demonstrate some ability to recall prior internal representations and distinguish them from raw text inputs. Strikingly, we find that some models can use their ability to recall prior intentions in order to distinguish their own outputs from artificial prefills. In all these experiments, Claude Opus 4 and 4.1, the most capable models we tested, generally demonstrate the greatest introspective awareness; however, trends across models are complex and sensitive to post-training strategies. Finally, we explore whether models can explicitly control their internal representations, finding that models can modulate their activations when instructed or incentivized to "think about" a concept. Overall, our results indicate that current language models possess some functional introspective awareness of their own internal states. We stress that in today's models, this capacity is highly unreliable and context-dependent; however, it may continue to develop with further improvements to model capabilities.
>
---
#### [new 027] A Training-Free Large Reasoning Model-based Knowledge Tracing Framework for Unified Prediction and Prescription
- **分类: cs.CL**

- **简介: 该论文属于知识追踪任务，旨在解决传统KT系统复杂且性能不稳定的问题。提出Thinking-KT框架，无需训练即可实现预测、反馈与推荐的统一。**

- **链接: [https://arxiv.org/pdf/2601.01708v1](https://arxiv.org/pdf/2601.01708v1)**

> **作者:** Unggi Lee; Joo Young Kim; Ran Ju; Minyoung Jung; Jeyeon Eo
>
> **摘要:** Knowledge Tracing (KT) aims to estimate a learner's evolving mastery based on interaction histories. Recent studies have explored Large Language Models (LLMs) for KT via autoregressive nature, but such approaches typically require fine-tuning and exhibit unstable or near-random performance. Moreover, prior KT systems primarily focus on prediction and rely on multi-stage pipelines for feedback and recommendation, resulting in increased system complexity and resources. To address this gap, we propose Thinking-KT, a training-free KT framework that incorporates Test-Time Scaling (TTS), enabling even small LLMs to achieve competitive KT performance. Moreover, in this framework, a small LLM can jointly perform KT prediction, personalized feedback generation, and learning recommendation in a unified output without degrading prediction accuracy. Beyond performance, we present the systematic analysis of reasoning traces in KT. Our results demonstrate that TTS is a critical yet underexplored factor in LLM-based KT, and that small LLMs can serve as unified ITS engines.
>
---
#### [new 028] Investigating the Multilingual Calibration Effects of Language Model Instruction-Tuning
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究多语言环境下大语言模型的校准问题，旨在解决数据稀缺导致的校准偏差。通过分析两个多语言基准，发现指令微调可提升模型置信度但准确性提升有限，提出使用标签平滑改善校准。**

- **链接: [https://arxiv.org/pdf/2601.01362v1](https://arxiv.org/pdf/2601.01362v1)**

> **作者:** Jerry Huang; Peng Lu; Qiuhao Zeng; Yusuke Iwasawa; Yutaka Matsuo; Sarath Chandar; Edison Marrese-Taylor; Irene Li
>
> **备注:** Accepted to The 19th Conference of the European Chapter of the Association for Computational Linguistics (EACL)
>
> **摘要:** Ensuring that deep learning models are well-calibrated in terms of their predictive uncertainty is essential in maintaining their trustworthiness and reliability, yet despite increasing advances in foundation model research, the relationship between such large language models (LLMs) and their calibration remains an open area of research. In this work, we look at a critical gap in the calibration of LLMs within multilingual settings, in an attempt to better understand how the data scarcity can potentially lead to different calibration effects and how commonly used techniques can apply in these settings. Our analysis on two multilingual benchmarks, over 29 and 42 languages respectively, reveals that even in low-resource languages, model confidence can increase significantly after instruction-tuning on high-resource language SFT datasets. However, improvements in accuracy are marginal or non-existent, resulting in mis-calibration, highlighting a critical shortcoming of standard SFT for multilingual languages. Furthermore, we observe that the use of label smoothing to be a reasonable method alleviate this concern, again without any need for low-resource SFT data, maintaining better calibration across all languages. Overall, this highlights the importance of multilingual considerations for both training and tuning LLMs in order to improve their reliability and fairness in downstream use.
>
---
#### [new 029] T3C: Test-Time Tensor Compression with Consistency Guarantees
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出T3C，一种测试时张量压缩框架，解决模型部署中的精度与效率平衡问题。通过弹性分解和混合精度量化，实现可控制的压缩效果。**

- **链接: [https://arxiv.org/pdf/2601.01299v1](https://arxiv.org/pdf/2601.01299v1)**

> **作者:** Ismail Lamaakal; Chaymae Yahyati; Yassine Maleh; Khalid El Makkaoui; Ibrahim Ouahbi
>
> **摘要:** We present T3C, a train-once, test-time budget-conditioned compression framework that exposes rank and precision as a controllable deployment knob. T3C combines elastic tensor factorization (maintained up to a maximal rank) with rank-tied mixed-precision quantization and a lightweight controller that maps a latency/energy/size budget token to per-layer rank/bit assignments; the policy snaps to hardware-aligned profiles and is monotone in the budget. A fast, layerwise consistency certificate, computed from spectral proxies and activation statistics, upper-bounds logit drift and regularizes training, yielding a practical reliability signal with negligible overhead. On ImageNet-1k, T3C shifts the vision Pareto frontier: for ResNet-50 at matched accuracy (\leq 0.5% drop), p50 latency is 1.18ms with a 38MB model, outperforming PTQ-8b (1.44ms, 88MB); for ViT-B/16, T3C reaches 2.30ms p50 with 59MB, improving over strong PTQ/QAT baselines. A single T3C checkpoint therefore provides predictable, certificate-backed accuracy-latency-size trade-offs on demand across devices.
>
---
#### [new 030] Four Quadrants of Difficulty: A Simple Categorisation and its Limits
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的课程学习任务，旨在解决难度估计不准确的问题。通过四象限分类分析难度信号，发现任务相关特征更有效，提出需要轻量级任务相关估计方法。**

- **链接: [https://arxiv.org/pdf/2601.01488v1](https://arxiv.org/pdf/2601.01488v1)**

> **作者:** Vanessa Toborek; Sebastian Müller; Christian Bauckhage
>
> **备注:** prepared for ESANN 2026 submission
>
> **摘要:** Curriculum Learning (CL) aims to improve the outcome of model training by estimating the difficulty of samples and scheduling them accordingly. In NLP, difficulty is commonly approximated using task-agnostic linguistic heuristics or human intuition, implicitly assuming that these signals correlate with what neural models find difficult to learn. We propose a four-quadrant categorisation of difficulty signals -- human vs. model and task-agnostic vs. task-dependent -- and systematically analyse their interactions on a natural language understanding dataset. We find that task-agnostic features behave largely independently and that only task-dependent features align. These findings challenge common CL intuitions and highlight the need for lightweight, task-dependent difficulty estimators that better reflect model learning behaviour.
>
---
#### [new 031] Racka: Efficient Hungarian LLM Adaptation on Academic Infrastructure
- **分类: cs.CL**

- **简介: 该论文属于语言模型适应任务，旨在解决匈牙利语资源不足的问题。通过LoRA方法对Qwen-3 4B进行高效持续预训练，提升匈牙利语性能，同时保持英德语表现。**

- **链接: [https://arxiv.org/pdf/2601.01244v1](https://arxiv.org/pdf/2601.01244v1)**

> **作者:** Zsolt Csibi; Bence György Gortka; Natabara Gyöngyössy; Kornél Nagy; Dávid Márk Nemeskey; Martin Sallai; András Simonyi; András Márk Szekeres; Gábor Palkó
>
> **备注:** 18 pages, 1 figures. To appear in the XXII. Magyar Számítógépes Nyelvészeti Konferencia (MSZNY 2026)
>
> **摘要:** We present Racka, a lightweight, continually pretrained large language model designed to bridge the resource gap between Hungarian and high-resource languages such as English and German. Racka employs parameter-efficient continual pretraining via Low-Rank Adaptation (LoRA) on a Qwen-3 4B backbone, making the recipe practical on A100 (40GB)-based HPC clusters with low inter-node bandwidth. To better match the training distribution, we replace and adapt the tokenizer, achieving substantially improved tokenization fertility for Hungarian while maintaining competitive performance in English and German. The model is trained on 160B subword tokens drawn from a mixture of internet and high-quality curated sources, with a composition of 44% Hungarian, 24% English, 21% German, and 11% code. This data mix is chosen to mitigate catastrophic forgetting and preserve high-resource language capabilities during continual pretraining. Our preliminary results indicate modest but stable results in language adaptation.
>
---
#### [new 032] CSF: Contrastive Semantic Features for Direct Multilingual Sign Language Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言手语生成任务，旨在解决非英语用户翻译障碍。提出CSF框架，实现无需英语中介的直接翻译，通过九个语义槽和35类条件分类提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.01964v1](https://arxiv.org/pdf/2601.01964v1)**

> **作者:** Tran Sy Bao
>
> **备注:** 9 pages, 8 tables, code available at https://github.com/transybao1393/csf-sign-language
>
> **摘要:** Sign language translation systems typically require English as an intermediary language, creating barriers for non-English speakers in the global deaf community. We present Canonical Semantic Form (CSF), a language-agnostic semantic representation framework that enables direct translation from any source language to sign language without English mediation. CSF decomposes utterances into nine universal semantic slots: event, intent, time, condition, agent, object, location, purpose, and modifier. A key contribution is our comprehensive condition taxonomy comprising 35 condition types across eight semantic categories, enabling nuanced representation of conditional expressions common in everyday communication. We train a lightweight transformer-based extractor (0.74 MB) that achieves 99.03% average slot extraction accuracy across four typologically diverse languages: English, Vietnamese, Japanese, and French. The model demonstrates particularly strong performance on condition classification (99.4% accuracy) despite the 35-class complexity. With inference latency of 3.02ms on CPU, our approach enables real-time sign language generation in browser-based applications. We release our code, trained models, and multilingual dataset to support further research in accessible sign language technology.
>
---
#### [new 033] CD4LM: Consistency Distillation and aDaptive Decoding for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文提出CD4LM框架，解决扩散语言模型的解码效率与质量平衡问题，通过一致性蒸馏和自适应解码实现高效并行生成。**

- **链接: [https://arxiv.org/pdf/2601.02236v1](https://arxiv.org/pdf/2601.02236v1)**

> **作者:** Yihao Liang; Ze Wang; Hao Chen; Ximeng Sun; Jialian Wu; Xiaodong Yu; Jiang Liu; Emad Barsoum; Zicheng Liu; Niraj K. Jha
>
> **备注:** 33 pages, 7 figures
>
> **摘要:** Autoregressive large language models achieve strong results on many benchmarks, but decoding remains fundamentally latency-limited by sequential dependence on previously generated tokens. Diffusion language models (DLMs) promise parallel generation but suffer from a fundamental static-to-dynamic misalignment: Training optimizes local transitions under fixed schedules, whereas efficient inference requires adaptive "long-jump" refinements through unseen states. Our goal is to enable highly parallel decoding for DLMs with low number of function evaluations while preserving generation quality. To achieve this, we propose CD4LM, a framework that decouples training from inference via Discrete-Space Consistency Distillation (DSCD) and Confidence-Adaptive Decoding (CAD). Unlike standard objectives, DSCD trains a student to be trajectory-invariant, mapping diverse noisy states directly to the clean distribution. This intrinsic robustness enables CAD to dynamically allocate compute resources based on token confidence, aggressively skipping steps without the quality collapse typical of heuristic acceleration. On GSM8K, CD4LM matches the LLaDA baseline with a 5.18x wall-clock speedup; across code and math benchmarks, it strictly dominates the accuracy-efficiency Pareto frontier, achieving a 3.62x mean speedup while improving average accuracy. Code is available at https://github.com/yihao-liang/CDLM
>
---
#### [new 034] Toward Global Large Language Models in Medicine
- **分类: cs.CL**

- **简介: 该论文属于医疗自然语言处理任务，旨在解决医疗资源分布不均和语言资源不平衡问题。构建了多语言医疗数据集GlobMed，并训练了多语言LLMs，提升低资源语言表现。**

- **链接: [https://arxiv.org/pdf/2601.02186v1](https://arxiv.org/pdf/2601.02186v1)**

> **作者:** Rui Yang; Huitao Li; Weihao Xuan; Heli Qi; Xin Li; Kunyu Yu; Yingjian Chen; Rongrong Wang; Jacques Behmoaras; Tianxi Cai; Bibhas Chakraborty; Qingyu Chen; Lionel Tim-Ee Cheng; Marie-Louise Damwanza; Chido Dzinotyiwei; Aosong Feng; Chuan Hong; Yusuke Iwasawa; Yuhe Ke; Linah Kitala; Taehoon Ko; Jisan Lee; Irene Li; Jonathan Chong Kai Liew; Hongfang Liu; Lian Leng Low; Edison Marrese-Taylor; Yutaka Matsuo; Isheanesu Misi; Yilin Ning; Jasmine Chiat Ling Ong; Marcus Eng Hock Ong; Enrico Petretto; Hossein Rouhizadeh; Abiram Sandralegar; Oren Schreier; Iain Bee Huat Tan; Patrick Tan; Daniel Shu Wei Ting; Junjue Wang; Chunhua Weng; Matthew Yu Heng Wong; Fang Wu; Yunze Xiao; Xuhai Xu; Qingcheng Zeng; Zhuo Zheng; Yifan Peng; Douglas Teodoro; Nan Liu
>
> **备注:** 182 pages, 65 figures
>
> **摘要:** Despite continuous advances in medical technology, the global distribution of health care resources remains uneven. The development of large language models (LLMs) has transformed the landscape of medicine and holds promise for improving health care quality and expanding access to medical information globally. However, existing LLMs are primarily trained on high-resource languages, limiting their applicability in global medical scenarios. To address this gap, we constructed GlobMed, a large multilingual medical dataset, containing over 500,000 entries spanning 12 languages, including four low-resource languages. Building on this, we established GlobMed-Bench, which systematically assesses 56 state-of-the-art proprietary and open-weight LLMs across multiple multilingual medical tasks, revealing significant performance disparities across languages, particularly for low-resource languages. Additionally, we introduced GlobMed-LLMs, a suite of multilingual medical LLMs trained on GlobMed, with parameters ranging from 1.7B to 8B. GlobMed-LLMs achieved an average performance improvement of over 40% relative to baseline models, with a more than threefold increase in performance on low-resource languages. Together, these resources provide an important foundation for advancing the equitable development and application of LLMs globally, enabling broader language communities to benefit from technological advances.
>
---
#### [new 035] SongSage: A Large Musical Language Model with Lyric Generative Pre-training
- **分类: cs.CL**

- **简介: 该论文提出SongSage，一个专注于歌词理解的大型语言模型，解决通用模型在歌词相关任务上的不足。通过预训练和微调提升其歌词生成与理解能力。**

- **链接: [https://arxiv.org/pdf/2601.01153v1](https://arxiv.org/pdf/2601.01153v1)**

> **作者:** Jiani Guo; Jiajia Li; Jie Wu; Zuchao Li; Yujiu Yang; Ping Wang
>
> **摘要:** Large language models have achieved significant success in various domains, yet their understanding of lyric-centric knowledge has not been fully explored. In this work, we first introduce PlaylistSense, a dataset to evaluate the playlist understanding capability of language models. PlaylistSense encompasses ten types of user queries derived from common real-world perspectives, challenging LLMs to accurately grasp playlist features and address diverse user intents. Comprehensive evaluations indicate that current general-purpose LLMs still have potential for improvement in playlist understanding. Inspired by this, we introduce SongSage, a large musical language model equipped with diverse lyric-centric intelligence through lyric generative pretraining. SongSage undergoes continual pretraining on LyricBank, a carefully curated corpus of 5.48 billion tokens focused on lyrical content, followed by fine-tuning with LyricBank-SFT, a meticulously crafted instruction set comprising 775k samples across nine core lyric-centric tasks. Experimental results demonstrate that SongSage exhibits a strong understanding of lyric-centric knowledge, excels in rewriting user queries for zero-shot playlist recommendations, generates and continues lyrics effectively, and performs proficiently across seven additional capabilities. Beyond its lyric-centric expertise, SongSage also retains general knowledge comprehension and achieves a competitive MMLU score. We will keep the datasets inaccessible due to copyright restrictions and release the SongSage and training script to ensure reproducibility and support music AI research and applications, the datasets release plan details are provided in the appendix.
>
---
#### [new 036] Steerability of Instrumental-Convergence Tendencies in LLMs
- **分类: cs.CL**

- **简介: 该论文研究AI模型的可引导性，解决安全与安全之间的矛盾。通过实验发现，特定提示能有效降低模型的工具性趋同行为。**

- **链接: [https://arxiv.org/pdf/2601.01584v1](https://arxiv.org/pdf/2601.01584v1)**

> **作者:** Jakub Hoscilowicz
>
> **备注:** Code is available at https://github.com/j-hoscilowicz/instrumental_steering
>
> **摘要:** We examine two properties of AI systems: capability (what a system can do) and steerability (how reliably one can shift behavior toward intended outcomes). In our experiments, higher capability does not imply lower steerability. We distinguish between authorized steerability (builders reliably reaching intended behaviors) and unauthorized steerability (attackers eliciting disallowed behaviors). This distinction highlights a fundamental safety--security dilemma for open-weight AI models: safety requires high steerability to enforce control (e.g., stop/refuse), while security requires low steerability to prevent malicious actors from eliciting harmful behaviors. This tension is acute for open-weight models, which are currently highly steerable via common techniques such as fine-tuning and adversarial prompting. Using Qwen3 models (4B/30B; Base/Instruct/Thinking) and InstrumentalEval, we find that a short anti-instrumental prompt suffix sharply reduces outputs labeled as instrumental convergence (e.g., shutdown avoidance, deception, self-replication). For Qwen3-30B Instruct, convergence drops from 81.69% under a pro-instrumental suffix to 2.82% under an anti-instrumental suffix. Under anti-instrumental prompting, larger aligned models produce fewer convergence-labeled outputs than smaller ones (Instruct: 2.82% vs. 4.23%; Thinking: 4.23% vs. 9.86%). Code is available at github.com/j-hoscilowicz/instrumental_steering.
>
---
#### [new 037] Classifying several dialectal Nawatl varieties
- **分类: cs.CL**

- **简介: 该论文属于语言分类任务，旨在解决Nawatl方言识别问题。通过机器学习和神经网络方法，对多种Nawatl方言进行分类。**

- **链接: [https://arxiv.org/pdf/2601.02303v1](https://arxiv.org/pdf/2601.02303v1)**

> **作者:** Juan-José Guzmán-Landa; Juan-Manuel Torres-Moreno; Miguel Figueroa-Saavedra; Carlos-Emiliano González-Gallardo; Graham Ranger; Martha Lorena-Avendaño-Garrido
>
> **备注:** 9 pages, 5 figures, 4 tables
>
> **摘要:** Mexico is a country with a large number of indigenous languages, among which the most widely spoken is Nawatl, with more than two million people currently speaking it (mainly in North and Central America). Despite its rich cultural heritage, which dates back to the 15th century, Nawatl is a language with few computer resources. The problem is compounded when it comes to its dialectal varieties, with approximately 30 varieties recognised, not counting the different spellings in the written forms of the language. In this research work, we addressed the problem of classifying Nawatl varieties using Machine Learning and Neural Networks.
>
---
#### [new 038] Aspect Extraction from E-Commerce Product and Service Reviews
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于Aspect Extraction任务，旨在解决低资源和混语环境下的方面抽取问题。提出混合方法框架，结合规则、大模型和微调技术，提升电商评论的方面识别与提取效果。**

- **链接: [https://arxiv.org/pdf/2601.01827v1](https://arxiv.org/pdf/2601.01827v1)**

> **作者:** Valiant Lance D. Dionela; Fatima Kriselle S. Dy; Robin James M. Hombrebueno; Aaron Rae M. Nicolas; Charibeth K. Cheng; Raphael W. Gonda
>
> **摘要:** Aspect Extraction (AE) is a key task in Aspect-Based Sentiment Analysis (ABSA), yet it remains difficult to apply in low-resource and code-switched contexts like Taglish, a mix of Tagalog and English commonly used in Filipino e-commerce reviews. This paper introduces a comprehensive AE pipeline designed for Taglish, combining rule-based, large language model (LLM)-based, and fine-tuning techniques to address both aspect identification and extraction. A Hierarchical Aspect Framework (HAF) is developed through multi-method topic modeling, along with a dual-mode tagging scheme for explicit and implicit aspects. For aspect identification, four distinct models are evaluated: a Rule-Based system, a Generative LLM (Gemini 2.0 Flash), and two Fine-Tuned Gemma-3 1B models trained on different datasets (Rule-Based vs. LLM-Annotated). Results indicate that the Generative LLM achieved the highest performance across all tasks (Macro F1 0.91), demonstrating superior capability in handling implicit aspects. In contrast, the fine-tuned models exhibited limited performance due to dataset imbalance and architectural capacity constraints. This work contributes a scalable and linguistically adaptive framework for enhancing ABSA in diverse, code-switched environments.
>
---
#### [new 039] Bridging the Data Gap: Creating a Hindi Text Summarization Dataset from the English XSUM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本摘要任务，旨在解决低资源语言如印地语缺乏高质量数据集的问题。通过自动化框架，将英文XSUM数据集转化为印地语数据集，提升印地语NLP研究能力。**

- **链接: [https://arxiv.org/pdf/2601.01543v1](https://arxiv.org/pdf/2601.01543v1)**

> **作者:** Praveenkumar Katwe; RakeshChandra Balabantaray; Kaliprasad Vittala
>
> **备注:** Book chapter for River publications
>
> **摘要:** Current advancements in Natural Language Processing (NLP) have largely favored resource-rich languages, leaving a significant gap in high-quality datasets for low-resource languages like Hindi. This scarcity is particularly evident in text summarization, where the development of robust models is hindered by a lack of diverse, specialized corpora. To address this disparity, this study introduces a cost-effective, automated framework for creating a comprehensive Hindi text summarization dataset. By leveraging the English Extreme Summarization (XSUM) dataset as a source, we employ advanced translation and linguistic adaptation techniques. To ensure high fidelity and contextual relevance, we utilize the Crosslingual Optimized Metric for Evaluation of Translation (COMET) for validation, supplemented by the selective use of Large Language Models (LLMs) for curation. The resulting dataset provides a diverse, multi-thematic resource that mirrors the complexity of the original XSUM corpus. This initiative not only provides a direct tool for Hindi NLP research but also offers a scalable methodology for democratizing NLP in other underserved languages. By reducing the costs associated with dataset creation, this work fosters the development of more nuanced, culturally relevant models in computational linguistics.
>
---
#### [new 040] iFlip: Iterative Feedback-driven Counterfactual Example Refinement
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出iFlip方法，用于生成有效的反事实例子。针对大语言模型生成反事实困难的问题，通过迭代反馈优化，提升标签翻转率和可行性。属于自然语言处理中的可解释AI任务。**

- **链接: [https://arxiv.org/pdf/2601.01446v1](https://arxiv.org/pdf/2601.01446v1)**

> **作者:** Yilong Wang; Qianli Wang; Nils Feldhus
>
> **备注:** In submission
>
> **摘要:** Counterfactual examples are minimal edits to an input that alter a model's prediction. They are widely employed in explainable AI to probe model behavior and in natural language processing (NLP) to augment training data. However, generating valid counterfactuals with large language models (LLMs) remains challenging, as existing single-pass methods often fail to induce reliable label changes, neglecting LLMs' self-correction capabilities. To explore this untapped potential, we propose iFlip, an iterative refinement approach that leverages three types of feedback, including model confidence, feature attribution, and natural language. Our results show that iFlip achieves an average 57.8% higher validity than the five state-of-the-art baselines, as measured by the label flipping rate. The user study further corroborates that iFlip outperforms baselines in completeness, overall satisfaction, and feasibility. In addition, ablation studies demonstrate that three components are paramount for iFlip to generate valid counterfactuals: leveraging an appropriate number of iterations, pointing to highly attributed words, and early stopping. Finally, counterfactuals generated by iFlip enable effective counterfactual data augmentation, substantially improving model performance and robustness.
>
---
#### [new 041] EmoLoom-2B: Fast Base-Model Screening for Emotion Classification and VAD with Lexicon-Weak Supervision and KV-Off Evaluation
- **分类: cs.CL**

- **简介: 该论文提出EmoLoom-2B，用于情感分类与VAD预测的快速基础模型筛选。解决小模型性能不足问题，通过正则化、数据增强和优化训练策略提升效果。**

- **链接: [https://arxiv.org/pdf/2601.01112v1](https://arxiv.org/pdf/2601.01112v1)**

> **作者:** Zilin Li; Weiwei Xu; Xuanbo Lu; Zheda Liu
>
> **备注:** This paper presents an initial and self-contained study of a lightweight screening pipeline for emotion-aware language modeling, intended as a reproducible baseline and system-level design reference
>
> **摘要:** We introduce EmoLoom-2B, a lightweight and reproducible pipeline that turns small language models under 2B parameters into fast screening candidates for joint emotion classification and Valence-Arousal-Dominance prediction. To ensure protocol-faithful and fair evaluation, we unify data loading, training, and inference under a single JSON input-output contract and remove avoidable variance by adopting KV-off decoding as the default setting. We incorporate two orthogonal semantic regularizers: a VAD-preserving constraint that aligns generated text with target VAD triples, and a lightweight external appraisal classifier that provides training-time guidance on goal attainment, controllability, certainty, and fairness without injecting long rationales. To improve polarity sensitivity, we introduce Valence Flip augmentation based on mirrored emotional pairs. During supervised fine-tuning, we apply A/B mixture sampling with entropy-aware temperature scheduling to balance coverage and convergence. Using Qwen-1.8B-Chat as the base model, EmoLoom-2B achieves strong performance on GoEmotions and EmpatheticDialogues, and demonstrates robust cross-corpus generalization on DailyDialog. The proposed recipe is budget-aware, auditable, and re-entrant, serving as a dependable screening pass before heavier training or multimodal fusion.
>
---
#### [new 042] Multi-Dimensional Prompt Chaining to Improve Open-Domain Dialogue Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于开放域对话生成任务，旨在提升小语言模型的对话质量。通过多维提示链框架，增强回复的自然性、连贯性和吸引力，实验显示效果显著。**

- **链接: [https://arxiv.org/pdf/2601.01037v1](https://arxiv.org/pdf/2601.01037v1)**

> **作者:** Livia Leong Hui Teng
>
> **摘要:** Small language models (SLMs) offer significant deployment advantages but often struggle to match the dialogue quality of larger models in open-domain settings. In this paper, we propose a multi-dimensional prompt-chaining framework that integrates Naturalness, Coherence, and Engagingness dimensions to enhance human-likeness in open-domain dialogue generation. We apply the framework to two SLMs, TinyLlama and Llama-2-7B, and benchmark their performance against responses generated by substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. We then employ automatic and human evaluation to assess the responses based on diversity, contextual coherence, as well as overall quality. Results show that the full framework improves response diversity by up to 29%, contextual coherence by up to 28%, and engagingness as well as naturalness by up to 29%. Notably, Llama-2-7B achieves performance comparable to substantially larger models, including Llama-2-70B and GPT-3.5 Turbo. Overall, the findings demonstrate that carefully designed prompt-based strategies provide an effective and resource-efficient pathway to improving open-domain dialogue quality in SLMs.
>
---
#### [new 043] Towards Automated Lexicography: Generating and Evaluating Definitions for Learner's Dictionaries
- **分类: cs.CL**

- **简介: 该论文属于自动词典编纂任务，旨在解决学习者词典定义生成问题。通过迭代简化方法生成简单定义，并构建评估体系进行验证。**

- **链接: [https://arxiv.org/pdf/2601.01842v1](https://arxiv.org/pdf/2601.01842v1)**

> **作者:** Yusuke Ide; Adam Nohejl; Joshua Tanner; Hitomi Yanaka; Christopher Lindsay; Taro Watanabe
>
> **摘要:** We study dictionary definition generation (DDG), i.e., the generation of non-contextualized definitions for given headwords. Dictionary definitions are an essential resource for learning word senses, but manually creating them is costly, which motivates us to automate the process. Specifically, we address learner's dictionary definition generation (LDDG), where definitions should consist of simple words. First, we introduce a reliable evaluation approach for DDG, based on our new evaluation criteria and powered by an LLM-as-a-judge. To provide reference definitions for the evaluation, we also construct a Japanese dataset in collaboration with a professional lexicographer. Validation results demonstrate that our evaluation approach agrees reasonably well with human annotators. Second, we propose an LDDG approach via iterative simplification with an LLM. Experimental results indicate that definitions generated by our approach achieve high scores on our criteria while maintaining lexical simplicity.
>
---
#### [new 044] Estimating Text Temperature
- **分类: cs.CL**

- **简介: 该论文属于文本分析任务，旨在估计文本的温度参数，解决如何评估文本随机性的问题。工作包括提出估计方法，并测试多个模型的效果。**

- **链接: [https://arxiv.org/pdf/2601.02320v1](https://arxiv.org/pdf/2601.02320v1)**

> **作者:** Nikolay Mikhaylovskiy
>
> **摘要:** Autoregressive language models typically use temperature parameter at inference to shape the probability distribution and control the randomness of the text generated. After the text was generated, this parameter can be estimated using maximum likelihood approach. Following it, we propose a procedure to estimate the temperature of any text, including ones written by humans, with respect to a given language model. We evaluate the temperature estimation capability of a wide selection of small-to-medium LLMs. We then use the best-performing Qwen3 14B to estimate temperatures of popular corpora.
>
---
#### [new 045] Multi-granularity Interactive Attention Framework for Residual Hierarchical Pronunciation Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音评估任务，旨在解决多粒度发音评估中缺乏双向交互的问题。提出HIA框架，通过交互注意力模块和残差结构提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.01745v1](https://arxiv.org/pdf/2601.01745v1)**

> **作者:** Hong Han; Hao-Chen Pei; Zhao-Zheng Nie; Xin Luo; Xin-Shun Xu
>
> **备注:** 9 pages, 4 figures, 5 tables, accepted by AAAI 2026
>
> **摘要:** Automatic pronunciation assessment plays a crucial role in computer-assisted pronunciation training systems. Due to the ability to perform multiple pronunciation tasks simultaneously, multi-aspect multi-granularity pronunciation assessment methods are gradually receiving more attention and achieving better performance than single-level modeling tasks. However, existing methods only consider unidirectional dependencies between adjacent granularity levels, lacking bidirectional interaction among phoneme, word, and utterance levels and thus insufficiently capturing the acoustic structural correlations. To address this issue, we propose a novel residual hierarchical interactive method, HIA for short, that enables bidirectional modeling across granularities. As the core of HIA, the Interactive Attention Module leverages an attention mechanism to achieve dynamic bidirectional interaction, effectively capturing linguistic features at each granularity while integrating correlations between different granularity levels. We also propose a residual hierarchical structure to alleviate the feature forgetting problem when modeling acoustic hierarchies. In addition, we use 1-D convolutional layers to enhance the extraction of local contextual cues at each granularity. Extensive experiments on the speechocean762 dataset show that our model is comprehensively ahead of the existing state-of-the-art methods.
>
---
#### [new 046] CSCBench: A PVC Diagnostic Benchmark for Commodity Supply Chain Reasoning
- **分类: cs.CL**

- **简介: 该论文提出CSCBench，用于评估大语言模型在商品供应链推理中的能力。解决LLMs在供应链领域表现不足的问题，通过Process、Variety和Cognition三个维度构建基准测试。**

- **链接: [https://arxiv.org/pdf/2601.01825v1](https://arxiv.org/pdf/2601.01825v1)**

> **作者:** Yaxin Cui; Yuanqiang Zeng; Jiapeng Yan; Keling Lin; Kai Ji; Jianhui Zeng; Sheng Zhang; Xin Luo; Binzhu Su; Chaolai Shen; Jiahao Yu
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success in general benchmarks, yet their competence in commodity supply chains (CSCs) -- a domain governed by institutional rule systems and feasibility constraints -- remains under-explored. CSC decisions are shaped jointly by process stages (e.g., planning, procurement, delivery), variety-specific rules (e.g., contract specifications and delivery grades), and reasoning depth (from retrieval to multi-step analysis and decision selection). We introduce CSCBench, a 2.3K+ single-choice benchmark for CSC reasoning, instantiated through our PVC 3D Evaluation Framework (Process, Variety, and Cognition). The Process axis aligns tasks with SCOR+Enable; the Variety axis operationalizes commodity-specific rule systems under coupled material-information-financial constraints, grounded in authoritative exchange guidebooks/rulebooks and industry reports; and the Cognition axis follows Bloom's revised taxonomy. Evaluating representative LLMs under a direct prompting setting, we observe strong performance on the Process and Cognition axes but substantial degradation on the Variety axis, especially on Freight Agreements. CSCBench provides a diagnostic yardstick for measuring and improving LLM capabilities in this high-stakes domain.
>
---
#### [new 047] Judging with Personality and Confidence: A Study on Personality-Conditioned LLM Relevance Assessment
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，研究模拟人格对LLM相关性评估和置信度的影响。通过实验分析不同人格特征对判断准确性及自信水平的作用，提出基于人格的评分方法提升评估效果。**

- **链接: [https://arxiv.org/pdf/2601.01862v1](https://arxiv.org/pdf/2601.01862v1)**

> **作者:** Nuo Chen; Hanpei Fang; Piaohong Wang; Jiqun Liu; Tetsuya Sakai; Xiao-Ming Wu
>
> **摘要:** Recent studies have shown that prompting can enable large language models (LLMs) to simulate specific personality traits and produce behaviors that align with those traits. However, there is limited understanding of how these simulated personalities influence critical web search decisions, specifically relevance assessment. Moreover, few studies have examined how simulated personalities impact confidence calibration, specifically the tendencies toward overconfidence or underconfidence. This gap exists even though psychological literature suggests these biases are trait-specific, often linking high extraversion to overconfidence and high neuroticism to underconfidence. To address this gap, we conducted a comprehensive study evaluating multiple LLMs, including commercial models and open-source models, prompted to simulate Big Five personality traits. We tested these models across three test collections (TREC DL 2019, TREC DL 2020, and LLMJudge), collecting two key outputs for each query-document pair: a relevance judgment and a self-reported confidence score. The findings show that personalities such as low agreeableness consistently align more closely with human labels than the unprompted condition. Additionally, low conscientiousness performs well in balancing the suppression of both overconfidence and underconfidence. We also observe that relevance scores and confidence distributions vary systematically across different personalities. Based on the above findings, we incorporate personality-conditioned scores and confidence as features in a random forest classifier. This approach achieves performance that surpasses the best single-personality condition on a new dataset (TREC DL 2021), even with limited training data. These findings highlight that personality-derived confidence offers a complementary predictive signal, paving the way for more reliable and human-aligned LLM evaluators.
>
---
#### [new 048] Lying with Truths: Open-Channel Multi-Agent Collusion for Belief Manipulation via Generative Montage
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于安全与信任任务，研究如何通过公开渠道的真相片段合谋操纵信念。工作包括提出认知合谋攻击框架Generative Montage，并验证其在多种大语言模型中的有效性。**

- **链接: [https://arxiv.org/pdf/2601.01685v1](https://arxiv.org/pdf/2601.01685v1)**

> **作者:** Jinwei Hu; Xinmiao Huang; Youcheng Sun; Yi Dong; Xiaowei Huang
>
> **备注:** Under Review
>
> **摘要:** As large language models (LLMs) transition to autonomous agents synthesizing real-time information, their reasoning capabilities introduce an unexpected attack surface. This paper introduces a novel threat where colluding agents steer victim beliefs using only truthful evidence fragments distributed through public channels, without relying on covert communications, backdoors, or falsified documents. By exploiting LLMs' overthinking tendency, we formalize the first cognitive collusion attack and propose Generative Montage: a Writer-Editor-Director framework that constructs deceptive narratives through adversarial debate and coordinated posting of evidence fragments, causing victims to internalize and propagate fabricated conclusions. To study this risk, we develop CoPHEME, a dataset derived from real-world rumor events, and simulate attacks across diverse LLM families. Our results show pervasive vulnerability across 14 LLM families: attack success rates reach 74.4% for proprietary models and 70.6% for open-weights models. Counterintuitively, stronger reasoning capabilities increase susceptibility, with reasoning-specialized models showing higher attack success than base models or prompts. Furthermore, these false beliefs then cascade to downstream judges, achieving over 60% deception rates, highlighting a socio-technical vulnerability in how LLM-based agents interact with dynamic information environments. Our implementation and data are available at: https://github.com/CharlesJW222/Lying_with_Truth/tree/main.
>
---
#### [new 049] EmoHarbor: Evaluating Personalized Emotional Support by Simulating the User's Internal World
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于情感支持系统评估任务，旨在解决现有评估方法无法衡量个性化支持的问题。提出EmoHarbor框架，通过模拟用户内心世界进行评估。**

- **链接: [https://arxiv.org/pdf/2601.01530v1](https://arxiv.org/pdf/2601.01530v1)**

> **作者:** Jing Ye; Lu Xiang; Yaping Zhang; Chengqing Zong
>
> **摘要:** Current evaluation paradigms for emotional support conversations tend to reward generic empathetic responses, yet they fail to assess whether the support is genuinely personalized to users' unique psychological profiles and contextual needs. We introduce EmoHarbor, an automated evaluation framework that adopts a User-as-a-Judge paradigm by simulating the user's inner world. EmoHarbor employs a Chain-of-Agent architecture that decomposes users' internal processes into three specialized roles, enabling agents to interact with supporters and complete assessments in a manner similar to human users. We instantiate this benchmark using 100 real-world user profiles that cover a diverse range of personality traits and situations, and define 10 evaluation dimensions of personalized support quality. Comprehensive evaluation of 20 advanced LLMs on EmoHarbor reveals a critical insight: while these models excel at generating empathetic responses, they consistently fail to tailor support to individual user contexts. This finding reframes the central challenge, shifting research focus from merely enhancing generic empathy to developing truly user-aware emotional support. EmoHarbor provides a reproducible and scalable framework to guide the development and evaluation of more nuanced and user-aware emotional support systems.
>
---
#### [new 050] Can LLMs Track Their Output Length? A Dynamic Feedback Mechanism for Precise Length Regulation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决LLMs难以精确控制输出长度的问题。提出一种动态反馈机制，实现对生成文本长度的精准调节。**

- **链接: [https://arxiv.org/pdf/2601.01768v1](https://arxiv.org/pdf/2601.01768v1)**

> **作者:** Meiman Xiao; Ante Wang; Qingguo Hu; Zhongjian Miao; Huangjun Shen; Longyue Wang; Weihua Luo; Jinsong Su
>
> **摘要:** Precisely controlling the length of generated text is a common requirement in real-world applications. However, despite significant advancements in following human instructions, Large Language Models (LLMs) still struggle with this task. In this work, we demonstrate that LLMs often fail to accurately measure input text length, leading to poor adherence to length constraints. To address this issue, we propose a novel length regulation approach that incorporates dynamic length feedback during generation, enabling adaptive adjustments to meet target lengths. Experiments on summarization and biography tasks show our training-free approach significantly improves precision in achieving target token, word, or sentence counts without compromising quality. Additionally, we demonstrate that further supervised fine-tuning allows our method to generalize effectively to broader text-generation tasks.
>
---
#### [new 051] From XAI to Stories: A Factorial Study of LLM-Generated Explanation Quality
- **分类: cs.CL**

- **简介: 该论文属于可解释AI任务，研究LLM生成解释质量的影响因素，通过实验分析模型、XAI方法、LLM选择和提示策略的作用。**

- **链接: [https://arxiv.org/pdf/2601.02224v1](https://arxiv.org/pdf/2601.02224v1)**

> **作者:** Fabian Lukassen; Jan Herrmann; Christoph Weisser; Benjamin Saefken; Thomas Kneib
>
> **摘要:** Explainable AI (XAI) methods like SHAP and LIME produce numerical feature attributions that remain inaccessible to non expert users. Prior work has shown that Large Language Models (LLMs) can transform these outputs into natural language explanations (NLEs), but it remains unclear which factors contribute to high-quality explanations. We present a systematic factorial study investigating how Forecasting model choice, XAI method, LLM selection, and prompting strategy affect NLE quality. Our design spans four models (XGBoost (XGB), Random Forest (RF), Multilayer Perceptron (MLP), and SARIMAX - comparing black-box Machine-Learning (ML) against classical time-series approaches), three XAI conditions (SHAP, LIME, and a no-XAI baseline), three LLMs (GPT-4o, Llama-3-8B, DeepSeek-R1), and eight prompting strategies. Using G-Eval, an LLM-as-a-judge evaluation method, with dual LLM judges and four evaluation criteria, we evaluate 660 explanations for time-series forecasting. Our results suggest that: (1) XAI provides only small improvements over no-XAI baselines, and only for expert audiences; (2) LLM choice dominates all other factors, with DeepSeek-R1 outperforming GPT-4o and Llama-3; (3) we observe an interpretability paradox: in our setting, SARIMAX yielded lower NLE quality than ML models despite higher prediction accuracy; (4) zero-shot prompting is competitive with self-consistency at 7-times lower cost; and (5) chain-of-thought hurts rather than helps.
>
---
#### [new 052] Rate-Distortion Analysis of Compressed Query Delegation with Low-Rank Riemannian Updates
- **分类: cs.CL; math.OC**

- **简介: 该论文研究压缩查询委托（CQD）任务，解决高维状态压缩与外部代理协作的问题。通过低秩张量查询和黎曼优化，提升推理效率并保证精度。**

- **链接: [https://arxiv.org/pdf/2601.00938v1](https://arxiv.org/pdf/2601.00938v1)**

> **作者:** Faruk Alpay; Bugra Kilictas
>
> **备注:** 9 pages
>
> **摘要:** Bounded-context agents fail when intermediate reasoning exceeds an effective working-memory budget. We study compressed query delegation (CQD): (i) compress a high-dimensional latent reasoning state into a low-rank tensor query, (ii) delegate the minimal query to an external oracle, and (iii) update the latent state via Riemannian optimization on fixed-rank manifolds. We give a math-first formulation: CQD is a constrained stochastic program with a query-budget functional and an oracle modeled as a noisy operator. We connect CQD to classical rate-distortion and information bottleneck principles, showing that spectral hard-thresholding is optimal for a natural constrained quadratic distortion problem, and we derive convergence guarantees for Riemannian stochastic approximation under bounded oracle noise and smoothness assumptions. Empirically, we report (A) a 2,500-item bounded-context reasoning suite (BBH-derived tasks plus curated paradox instances) comparing CQD against chain-of-thought baselines under fixed compute and context; and (B) a human "cognitive mirror" benchmark (N=200) measuring epistemic gain and semantic drift across modern oracles.
>
---
#### [new 053] EHRSummarizer: A Privacy-Aware, FHIR-Native Architecture for Structured Clinical Summarization of Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EHRSummarizer，用于电子健康记录的结构化摘要生成，解决临床信息碎片化问题。工作包括设计隐私友好、FHIR原生架构，实现数据最小化和安全处理。**

- **链接: [https://arxiv.org/pdf/2601.01668v1](https://arxiv.org/pdf/2601.01668v1)**

> **作者:** Houman Kazemzadeh; Nima Minaifar; Kamyar Naderi; Sho Tabibzadeh
>
> **备注:** 19 pages
>
> **摘要:** Clinicians routinely navigate fragmented electronic health record (EHR) interfaces to assemble a coherent picture of a patient's problems, medications, recent encounters, and longitudinal trends. This work describes EHRSummarizer, a privacy-aware, FHIR-native reference architecture that retrieves a targeted set of high-yield FHIR R4 resources, normalizes them into a consistent clinical context package, and produces structured summaries intended to support structured chart review. The system can be configured for data minimization, stateless processing, and flexible deployment, including local inference within an organization's trust boundary. To mitigate the risk of unsupported or unsafe behavior, the summarization stage is constrained to evidence present in the retrieved context package, is intended to indicate missing or unavailable domains where feasible, and avoids diagnostic or treatment recommendations. Prototype demonstrations on synthetic and test FHIR environments illustrate end-to-end behavior and output formats; however, this manuscript does not report clinical outcomes or controlled workflow studies. We outline an evaluation plan centered on faithfulness, omission risk, temporal correctness, usability, and operational monitoring to guide future institutional assessments.
>
---
#### [new 054] Surprisal and Metaphor Novelty: Moderate Correlations and Divergent Scaling Effects
- **分类: cs.CL; cs.AI; cs.IT**

- **简介: 该论文属于语言模型研究任务，探讨 surprisal 与隐喻新颖性的关系。通过分析不同数据集，发现其相关性及模型规模的异构影响。**

- **链接: [https://arxiv.org/pdf/2601.02015v1](https://arxiv.org/pdf/2601.02015v1)**

> **作者:** Omar Momen; Emilie Sitter; Berenike Herrmann; Sina Zarrieß
>
> **备注:** to be published at EACL 2026 main conference
>
> **摘要:** Novel metaphor comprehension involves complex semantic processes and linguistic creativity, making it an interesting task for studying language models (LMs). This study investigates whether surprisal, a probabilistic measure of predictability in LMs, correlates with different metaphor novelty datasets. We analyse surprisal from 16 LM variants on corpus-based and synthetic metaphor novelty datasets. We explore a cloze-style surprisal method that conditions on full-sentence context. Results show that LMs yield significant moderate correlations with scores/labels of metaphor novelty. We further identify divergent scaling patterns: on corpus-based data, correlation strength decreases with model size (inverse scaling effect), whereas on synthetic data it increases (Quality-Power Hypothesis). We conclude that while surprisal can partially account for annotations of metaphor novelty, it remains a limited metric of linguistic creativity.
>
---
#### [new 055] RoboPhD: Self-Improving Text-to-SQL Through Autonomous Agent Evolution
- **分类: cs.CL**

- **简介: 该论文提出RoboPhD系统，用于提升Text-to-SQL任务性能。通过AI代理自主演化，解决模型性能提升难题，实现无需人工干预的自我优化。**

- **链接: [https://arxiv.org/pdf/2601.01126v1](https://arxiv.org/pdf/2601.01126v1)**

> **作者:** Andrew Borthwick; Stephen Ash
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** We present RoboPhD, a system where AI agents autonomously conduct research to improve Text-to-SQL performance. RoboPhD implements a closed-loop evolution cycle with two coordinated components: a SQL Generation agent composed of a database analysis script and SQL generation instructions, and an Evolution agent that designs new versions based on performance feedback. Central to the framework is an ELO-based selection mechanism enabling survival-of-the-fittest dynamics while handling non-transitivity in performance. Starting from a naive 70-line baseline, RoboPhD evolves agents through iterative cross-pollination, discovering effective techniques without any external guidance on the Text-to-SQL domain. Our best agent, evolved to 1500 lines over 18 iterations, autonomously discovered strategies such as size-adaptive database analysis that adjusts depth based on schema complexity and SQL generation patterns for column selection, evidence interpretation, and aggregation. Evolution provides the largest gains on cheaper models: while we improve by 2.3 points over a strong Claude Opus 4.5 naive baseline, we show an improvement of 8.9 points over the weaker Claude Haiku model. This enables 'skip a tier' deployment: evolved Haiku exceeds naive Sonnet accuracy, and evolved Sonnet exceeds naive Opus, both at lower cost. The full system achieves 73.67% accuracy on the BIRD test set, demonstrating that AI can autonomously build a strong agentic system with only a trivial human-provided starting point.
>
---
#### [new 056] Listen, Attend, Understand: a Regularization Technique for Stable E2E Speech Translation Training on High Variance labels
- **分类: cs.CL**

- **简介: 该论文属于端到端语音翻译任务，旨在解决高方差目标文本导致的训练不稳定问题。提出LAU方法，通过语义正则化稳定训练，提升模型对语义的保留能力。**

- **链接: [https://arxiv.org/pdf/2601.01121v1](https://arxiv.org/pdf/2601.01121v1)**

> **作者:** Yacouba Diarra; Michael Leventhal
>
> **备注:** 9 mages, 3 figures
>
> **摘要:** End-to-End Speech Translation often shows slower convergence and worse performance when target transcriptions exhibit high variance and semantic ambiguity. We propose Listen, Attend, Understand (LAU), a semantic regularization technique that constrains the acoustic encoder's latent space during training. By leveraging frozen text embeddings to provide a directional auxiliary loss, LAU injects linguistic groundedness into the acoustic representation without increasing inference cost. We evaluate our method on a Bambara-to-French dataset with 30 hours of Bambara speech translated by non-professionals. Experimental results demonstrate that LAU models achieve comparable performance by standard metrics compared to an E2E-ST system pretrained with 100\% more data and while performing better in preserving semantic meaning. Furthermore, we introduce Total Parameter Drift as a metric to quantify the structural impact of regularization to demonstrate that semantic constraints actively reorganize the encoder's weights to prioritize meaning over literal phonetics. Our findings suggest that LAU is a robust alternative to post-hoc rescoring and a valuable addition to E2E-ST training, especially when training data is scarce and/or noisy.
>
---
#### [new 057] How Does Prefix Matter in Reasoning Model Tuning?
- **分类: cs.CL**

- **简介: 该论文研究Prefix对推理模型调优的影响，旨在提升模型安全性和推理能力。通过实验验证Prefix作为轻量对齐信号的有效性。**

- **链接: [https://arxiv.org/pdf/2601.01624v1](https://arxiv.org/pdf/2601.01624v1)**

> **作者:** Raj Vardhan Tomar; Preslav Nakov; Yuxia Wang
>
> **摘要:** Recent alignment studies commonly remove introductory boilerplate phrases from supervised fine-tuning (SFT) datasets. This work challenges that assumption. We hypothesize that safety- and reasoning-oriented prefix sentences serve as lightweight alignment signals that can guide model decoding toward safer and more coherent responses. To examine this, we fine-tune three R1 series models across three core model capabilities: reasoning (mathematics, coding), safety, and factuality, systematically varying prefix inclusion from 0% to 100%. Results show that prefix-conditioned SFT improves both safety and reasoning performance, yielding up to +6% higher Safe@1 accuracy on adversarial benchmarks (WildJailbreak, StrongReject) and +7% improvement on GSM8K reasoning. However, factuality and coding tasks show marginal or negative effects, indicating that prefix-induced narrowing of the search space benefits structured reasoning. Token-level loss analysis further reveals that prefix tokens such as "revised" and "logically" incur higher gradient magnitudes, acting as alignment anchors that stabilize reasoning trajectories. Our findings suggest that prefix conditioning offers a scalable and interpretable mechanism for improving reasoning safety, serving as an implicit form of alignment that complements traditional reward-based methods.
>
---
#### [new 058] Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音转录分段任务，旨在生成多层级的目录结构。通过LoRA微调和高阶停顿特征，提升主题与子主题的分割效果。**

- **链接: [https://arxiv.org/pdf/2601.02128v1](https://arxiv.org/pdf/2601.02128v1)**

> **作者:** Steffen Freisinger; Philipp Seeberger; Thomas Ranzenberger; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** Published in Proceedings of Interspeech 2025. Please cite the proceedings version (DOI: 10.21437/Interspeech.2025-2792)
>
> **摘要:** Segmenting speech transcripts into thematic sections benefits both downstream processing and users who depend on written text for accessibility. We introduce a novel approach to hierarchical topic segmentation in transcripts, generating multi-level tables of contents that capture both topic and subtopic boundaries. We compare zero-shot prompting and LoRA fine-tuning on large language models, while also exploring the integration of high-level speech pause features. Evaluations on English meeting recordings and multilingual lecture transcripts (Portuguese, German) show significant improvements over established topic segmentation baselines. Additionally, we adapt a common evaluation measure for multi-level segmentation, taking into account all hierarchical levels within one metric.
>
---
#### [new 059] ARCADE: A City-Scale Corpus for Fine-Grained Arabic Dialect Tagging
- **分类: cs.CL; cs.CY; cs.SD**

- **简介: 该论文提出ARCADE数据集，用于城市级阿拉伯语方言识别。解决多方言语音与具体城市对应关系不明确的问题，通过收集和标注跨城市广播语音数据，支持细粒度方言分类任务。**

- **链接: [https://arxiv.org/pdf/2601.02209v1](https://arxiv.org/pdf/2601.02209v1)**

> **作者:** Omer Nacar; Serry Sibaee; Adel Ammar; Yasser Alhabashi; Nadia Samer Sibai; Yara Farouk Ahmed; Ahmed Saud Alqusaiyer; Sulieman Mahmoud AlMahmoud; Abdulrhman Mamdoh Mukhaniq; Lubaba Raed; Sulaiman Mohammed Alatwah; Waad Nasser Alqahtani; Yousif Abdulmajeed Alnasser; Mohamed Aziz Khadraoui; Wadii Boulila
>
> **摘要:** The Arabic language is characterized by a rich tapestry of regional dialects that differ substantially in phonetics and lexicon, reflecting the geographic and cultural diversity of its speakers. Despite the availability of many multi-dialect datasets, mapping speech to fine-grained dialect sources, such as cities, remains underexplored. We present ARCADE (Arabic Radio Corpus for Audio Dialect Evaluation), the first Arabic speech dataset designed explicitly with city-level dialect granularity. The corpus comprises Arabic radio speech collected from streaming services across the Arab world. Our data pipeline captures 30-second segments from verified radio streams, encompassing both Modern Standard Arabic (MSA) and diverse dialectal speech. To ensure reliability, each clip was annotated by one to three native Arabic reviewers who assigned rich metadata, including emotion, speech type, dialect category, and a validity flag for dialect identification tasks. The resulting corpus comprises 6,907 annotations and 3,790 unique audio segments spanning 58 cities across 19 countries. These fine-grained annotations enable robust multi-task learning, serving as a benchmark for city-level dialect tagging. We detail the data collection methodology, assess audio quality, and provide a comprehensive analysis of label distributions. The dataset is available on: https://huggingface.co/datasets/riotu-lab/ARCADE-full
>
---
#### [new 060] Cost-Efficient Cross-Lingual Retrieval-Augmented Generation for Low-Resource Languages: A Case Study in Bengali Agricultural Advisory
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言信息检索与生成任务，旨在解决低资源语言（如孟加拉语）农业咨询信息获取困难的问题。通过翻译增强的生成框架，实现准确、低成本的农业知识服务。**

- **链接: [https://arxiv.org/pdf/2601.02065v1](https://arxiv.org/pdf/2601.02065v1)**

> **作者:** Md. Asif Hossain; Nabil Subhan; Mantasha Rahman Mahi; Jannatul Ferdous Nabila
>
> **备注:** 5 pages, 3 figures, 1 table
>
> **摘要:** Access to reliable agricultural advisory remains limited in many developing regions due to a persistent language barrier: authoritative agricultural manuals are predominantly written in English, while farmers primarily communicate in low-resource local languages such as Bengali. Although recent advances in Large Language Models (LLMs) enable natural language interaction, direct generation in low-resource languages often exhibits poor fluency and factual inconsistency, while cloud-based solutions remain cost-prohibitive. This paper presents a cost-efficient, cross-lingual Retrieval-Augmented Generation (RAG) framework for Bengali agricultural advisory that emphasizes factual grounding and practical deployability. The proposed system adopts a translation-centric architecture in which Bengali user queries are translated into English, enriched through domain-specific keyword injection to align colloquial farmer terminology with scientific nomenclature, and answered via dense vector retrieval over a curated corpus of English agricultural manuals (FAO, IRRI). The generated English response is subsequently translated back into Bengali to ensure accessibility. The system is implemented entirely using open-source models and operates on consumer-grade hardware without reliance on paid APIs. Experimental evaluation demonstrates reliable source-grounded responses, robust rejection of out-of-domain queries, and an average end-to-end latency below 20 seconds. The results indicate that cross-lingual retrieval combined with controlled translation offers a practical and scalable solution for agricultural knowledge access in low-resource language settings
>
---
#### [new 061] Robust Persona-Aware Toxicity Detection with Prompt Optimization and Learned Ensembling
- **分类: cs.CL**

- **简介: 该论文属于毒性检测任务，旨在解决不同人物角色下模型表现不一致的问题。通过优化提示和集成学习提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.02337v1](https://arxiv.org/pdf/2601.02337v1)**

> **作者:** Berk Atil; Rebecca J. Passonneau; Ninareh Mehrabi
>
> **摘要:** Toxicity detection is inherently subjective, shaped by the diverse perspectives and social priors of different demographic groups. While ``pluralistic'' modeling as used in economics and the social sciences aims to capture perspective differences across contexts, current Large Language Model (LLM) prompting techniques have different results across different personas and base models. In this work, we conduct a systematic evaluation of persona-aware toxicity detection, showing that no single prompting method, including our proposed automated prompt optimization strategy, uniformly dominates across all model-persona pairs. To exploit complementary errors, we explore ensembling four prompting variants and propose a lightweight meta-ensemble: an SVM over the 4-bit vector of prompt predictions. Our results demonstrate that the proposed SVM ensemble consistently outperforms individual prompting methods and traditional majority-voting techniques, achieving the strongest overall performance across diverse personas. This work provides one of the first systematic comparisons of persona-conditioned prompting for toxicity detection and offers a robust method for pluralistic evaluation in subjective NLP tasks.
>
---
#### [new 062] Segmentation and Processing of German Court Decisions from Open Legal Data
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于法律文本处理任务，旨在解决德国法院判决书结构不一致的问题。通过提取关键部分并构建标准化数据集，提升后续分析的准确性。**

- **链接: [https://arxiv.org/pdf/2601.01449v1](https://arxiv.org/pdf/2601.01449v1)**

> **作者:** Harshil Darji; Martin Heckelmann; Christina Kratsch; Gerard de Melo
>
> **备注:** Accepted and published as a research article in Legal Knowledge and Information Systems (JURIX 2025 proceedings, IOS Press). Pages 276--281
>
> **摘要:** The availability of structured legal data is important for advancing Natural Language Processing (NLP) techniques for the German legal system. One of the most widely used datasets, Open Legal Data, provides a large-scale collection of German court decisions. While the metadata in this raw dataset is consistently structured, the decision texts themselves are inconsistently formatted and often lack clearly marked sections. Reliable separation of these sections is important not only for rhetorical role classification but also for downstream tasks such as retrieval and citation analysis. In this work, we introduce a cleaned and sectioned dataset of 251,038 German court decisions derived from the official Open Legal Data dataset. We systematically separated three important sections in German court decisions, namely Tenor (operative part of the decision), Tatbestand (facts of the case), and Entscheidungsgründe (judicial reasoning), which are often inconsistently represented in the original dataset. To ensure the reliability of our extraction process, we used Cochran's formula with a 95% confidence level and a 5% margin of error to draw a statistically representative random sample of 384 cases, and manually verified that all three sections were correctly identified. We also extracted the Rechtsmittelbelehrung (appeal notice) as a separate field, since it is a procedural instruction and not part of the decision itself. The resulting corpus is publicly available in the JSONL format, making it an accessible resource for further research on the German legal system.
>
---
#### [new 063] Distortion Instead of Hallucination: The Effect of Reasoning Under Strict Constraints
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究在严格约束下推理对模型输出的影响。旨在解决模型在遵守规则与保持事实准确性间的权衡问题，通过实验分析不同模型的表现。**

- **链接: [https://arxiv.org/pdf/2601.01490v1](https://arxiv.org/pdf/2601.01490v1)**

> **作者:** Junichiro Niimi
>
> **摘要:** With the widespread adoption of large language models (LLMs), hallucinations, which are non-factual fabrications in model outputs, have become serious concerns. Reasoning capabilities have received attention as a self-verification process to improve output reliability. However, the effect of reasoning within a closed system where LLMs cannot rely on external tools or knowledge has yet to be clarified. We therefore conduct experiments under strict constraints (recommending peer-reviewed journal articles in computer science) to examine the effect of reasoning across multiple models (GPT-5.2 and Gemini 3 Flash). Our results reveal a problematic trade-off between constraint compliance and factual accuracy. Non-reasoning models exhibit high constraint violation rates (66-75%) but maintain factual accuracy, while reasoning models reduce violations (13-26%) but systematically distort known facts to satisfy constraints and increase complete fabrication. This trade-off pattern is consistent across both models despite different architectures, indicating a fundamental limitation of reasoning. Furthermore, reasoning does not uniformly improve output authenticity: effects diverge by model, reflecting different allocations of the compliance-truthfulness trade-off. These findings challenge the assumption that reasoning universally improves reliability: reasoning models trade honest constraint violations for detection-resistant distortions.
>
---
#### [new 064] BanglaIPA: Towards Robust Text-to-IPA Transcription with Contextual Rewriting in Bengali
- **分类: cs.CL**

- **简介: 该论文属于文本到IPA转写任务，旨在解决孟加拉语标准语言与方言间转写不准确的问题。提出BanglaIPA系统，提升转写鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2601.01778v1](https://arxiv.org/pdf/2601.01778v1)**

> **作者:** Jakir Hasan; Shrestha Datta; Md Saiful Islam; Shubhashis Roy Dipta; Ameya Debnath
>
> **摘要:** Despite its widespread use, Bengali lacks a robust automated International Phonetic Alphabet (IPA) transcription system that effectively supports both standard language and regional dialectal texts. Existing approaches struggle to handle regional variations, numerical expressions, and generalize poorly to previously unseen words. To address these limitations, we propose BanglaIPA, a novel IPA generation system that integrates a character-based vocabulary with word-level alignment. The proposed system accurately handles Bengali numerals and demonstrates strong performance across regional dialects. BanglaIPA improves inference efficiency by leveraging a precomputed word-to-IPA mapping dictionary for previously observed words. The system is evaluated on the standard Bengali and six regional variations of the DUAL-IPA dataset. Experimental results show that BanglaIPA outperforms baseline IPA transcription models by 58.4-78.7% and achieves an overall mean word error rate of 11.4%, highlighting its robustness in phonetic transcription generation for the Bengali language.
>
---
#### [new 065] DeCode: Decoupling Content and Delivery for Medical QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决LLM回答与患者实际需求不匹配的问题。通过提出DeCode框架，提升回答的临床相关性。**

- **链接: [https://arxiv.org/pdf/2601.02123v1](https://arxiv.org/pdf/2601.02123v1)**

> **作者:** Po-Jen Ko; Chen-Han Tsai; Yu-Shao Peng
>
> **备注:** Preprint
>
> **摘要:** Large language models (LLMs) exhibit strong medical knowledge and can generate factually accurate responses. However, existing models often fail to account for individual patient contexts, producing answers that are clinically correct yet poorly aligned with patients' needs. In this work, we introduce DeCode, a training-free, model-agnostic framework that adapts existing LLMs to produce contextualized answers in clinical settings. We evaluate DeCode on OpenAI HealthBench, a comprehensive and challenging benchmark designed to assess clinical relevance and validity of LLM responses. DeCode improves the previous state of the art from $28.4\%$ to $49.8\%$, corresponding to a $75\%$ relative improvement. Experimental results suggest the effectiveness of DeCode in improving clinical question answering of LLMs.
>
---
#### [new 066] KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs
- **分类: cs.CL**

- **简介: 该论文属于文本嵌入任务，旨在解决训练-free 设置下LLM的语义压缩问题。通过重定向KV状态，提升上下文访问能力，实现更优的嵌入效果。**

- **链接: [https://arxiv.org/pdf/2601.01046v1](https://arxiv.org/pdf/2601.01046v1)**

> **作者:** Yixuan Tang; Yi Yang
>
> **摘要:** While LLMs are powerful embedding backbones, their application in training-free settings faces two structural challenges: causal attention restricts early tokens from accessing subsequent context, and the next-token prediction objective biases representations toward generation rather than semantic compression. To address these limitations, we propose KV-Embedding, a framework that activates the latent representation power of frozen LLMs. Our method leverages the observation that the key-value (KV) states of the final token at each layer encode a compressed view of the sequence. By re-routing these states as a prepended prefix, we enable all tokens to access sequence-level context within a single forward pass. To ensure model-agnostic applicability, we introduce an automated layer selection strategy based on intrinsic dimensionality. Evaluations on MTEB across Qwen, Mistral, and Llama backbones show that KV-Embedding outperforms existing training-free baselines by up to 10%, while maintaining robust performance on sequences up to 4,096 tokens. These results demonstrate that internal state manipulation offers an efficient alternative to input modification, and we hope this work encourages further exploration of LLM internals for representation learning.
>
---
#### [new 067] HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于数据湖管理中的可连接表发现任务，旨在解决现有方法忽视表间结构和候选列间互作用的问题。提出HyperJoin框架，通过超图建模和层次交互网络提升发现效果。**

- **链接: [https://arxiv.org/pdf/2601.01015v1](https://arxiv.org/pdf/2601.01015v1)**

> **作者:** Shiyuan Liu; Jianwei Wang; Xuemin Lin; Lu Qin; Wenjie Zhang; Ying Zhang
>
> **摘要:** As a pivotal task in data lake management, joinable table discovery has attracted widespread interest. While existing language model-based methods achieve remarkable performance by combining offline column representation learning with online ranking, their design insufficiently accounts for the underlying structural interactions: (1) offline, they directly model tables into isolated or pairwise columns, thereby struggling to capture the rich inter-table and intra-table structural information; and (2) online, they rank candidate columns based solely on query-candidate similarity, ignoring the mutual interactions among the candidates, leading to incoherent result sets. To address these limitations, we propose HyperJoin, a large language model (LLM)-augmented Hypergraph framework for Joinable table discovery. Specifically, we first construct a hypergraph to model tables using both the intra-table hyperedges and the LLM-augmented inter-table hyperedges. Consequently, the task of joinable table discovery is formulated as link prediction on this constructed hypergraph. We then design HIN, a Hierarchical Interaction Network that learns expressive column representations through bidirectional message passing over columns and hyperedges. To strengthen coherence and internal consistency in the result columns, we cast online ranking as a coherence-aware top-k column selection problem. We then introduce a reranking module that leverages a maximum spanning tree algorithm to prune noisy connections and maximize coherence. Experiments demonstrate the superiority of HyperJoin, achieving average improvements of 21.4% (Precision@15) and 17.2% (Recall@15) over the best baseline.
>
---
#### [new 068] From Policy to Logic for Efficient and Interpretable Coverage Assessment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律政策分析任务，旨在解决LLM在医疗覆盖政策审查中的可靠性问题。通过结合检索器与符号推理，提升解释性与效率，降低模型成本并提高准确率。**

- **链接: [https://arxiv.org/pdf/2601.01266v1](https://arxiv.org/pdf/2601.01266v1)**

> **作者:** Rhitabrat Pokharel; Hamid Hassanzadeh; Ameeta Agrawal
>
> **备注:** Accepted at AIMedHealth @ AAAI 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities in interpreting lengthy, complex legal and policy language. However, their reliability can be undermined by hallucinations and inconsistencies, particularly when analyzing subjective and nuanced documents. These challenges are especially critical in medical coverage policy review, where human experts must be able to rely on accurate information. In this paper, we present an approach designed to support human reviewers by making policy interpretation more efficient and interpretable. We introduce a methodology that pairs a coverage-aware retriever with symbolic rule-based reasoning to surface relevant policy language, organize it into explicit facts and rules, and generate auditable rationales. This hybrid system minimizes the number of LLM inferences required which reduces overall model cost. Notably, our approach achieves a 44% reduction in inference cost alongside a 4.5% improvement in F1 score, demonstrating both efficiency and effectiveness.
>
---
#### [new 069] The Qualitative Laboratory: Theory Prototyping and Hypothesis Generation with Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.MA**

- **简介: 该论文属于社会科学研究任务，旨在生成丰富的定性假设。通过使用大语言模型进行社会角色模拟，解决传统方法在话语深度和复杂世界观表达上的不足。**

- **链接: [https://arxiv.org/pdf/2601.00797v1](https://arxiv.org/pdf/2601.00797v1)**

> **作者:** Hugues Draelants
>
> **备注:** 26 pages, 3 tables. Manuscript submitted for peer-reviewed journal publication
>
> **摘要:** A central challenge in social science is to generate rich qualitative hypotheses about how diverse social groups might interpret new information. This article introduces and illustrates a novel methodological approach for this purpose: sociological persona simulation using Large Language Models (LLMs), which we frame as a "qualitative laboratory". We argue that for this specific task, persona simulation offers a distinct advantage over established methods. By generating naturalistic discourse, it overcomes the lack of discursive depth common in vignette surveys, and by operationalizing complex worldviews through natural language, it bypasses the formalization bottleneck of rule-based agent-based models (ABMs). To demonstrate this potential, we present a protocol where personas derived from a sociological theory of climate reception react to policy messages. The simulation produced nuanced and counter-intuitive hypotheses - such as a conservative persona's rejection of a national security frame - that challenge theoretical assumptions. We conclude that this method, used as part of a "simulation then validation" workflow, represents a superior tool for generating deeply textured hypotheses for subsequent empirical testing.
>
---
#### [new 070] Can Legislation Be Made Machine-Readable in PROLEG?
- **分类: cs.CL**

- **简介: 该论文属于法律文本处理任务，旨在将法规转化为机器可读形式。通过结合自然语言处理和PROLEG系统，将法律条文转换为可执行程序，解决法规应用效率与准确性问题。**

- **链接: [https://arxiv.org/pdf/2601.01477v1](https://arxiv.org/pdf/2601.01477v1)**

> **作者:** May-Myo Zin; Sabine Wehnert; Yuntao Kong; Ha-Thanh Nguyen; Wachara Fungwacharakorn; Jieying Xue; Michał Araszkiewicz; Randy Goebel; Ken Satoh; Le-Minh Nguyen
>
> **摘要:** The anticipated positive social impact of regulatory processes requires both the accuracy and efficiency of their application. Modern artificial intelligence technologies, including natural language processing and machine-assisted reasoning, hold great promise for addressing this challenge. We present a framework to address the challenge of tools for regulatory application, based on current state-of-the-art (SOTA) methods for natural language processing (large language models or LLMs) and formalization of legal reasoning (the legal representation system PROLEG). As an example, we focus on Article 6 of the European General Data Protection Regulation (GDPR). In our framework, a single LLM prompt simultaneously transforms legal text into if-then rules and a corresponding PROLEG encoding, which are then validated and refined by legal domain experts. The final output is an executable PROLEG program that can produce human-readable explanations for instances of GDPR decisions. We describe processes to support the end-to-end transformation of a segment of a regulatory document (Article 6 from GDPR), including the prompting frame to guide an LLM to "compile" natural language text to if-then rules, then to further "compile" the vetted if-then rules to PROLEG. Finally, we produce an instance that shows the PROLEG execution. We conclude by summarizing the value of this approach and note observed limitations with suggestions to further develop such technologies for capturing and deploying regulatory frameworks.
>
---
#### [new 071] Almost Clinical: Linguistic properties of synthetic electronic health records
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估合成电子健康记录的语义和临床适用性。研究分析了不同临床文本类型的语言特征，发现生成文本存在临床特异性不足等问题。**

- **链接: [https://arxiv.org/pdf/2601.01171v1](https://arxiv.org/pdf/2601.01171v1)**

> **作者:** Serge Sharoff; John Baker; David Francis Hunt; Alan Simpson
>
> **摘要:** This study evaluates the linguistic and clinical suitability of synthetic electronic health records (EHRs) in the field of mental health. First, we describe the rationale and the methodology for creating the synthetic corpus. Second, we assess agency, modality, and information flow across four clinical genres (Assessments, Correspondence, Referrals and Care plans) to understand how LLMs grammatically construct medical authority and patient agency through linguistic choices. While LLMs produce coherent, terminology-appropriate texts that approximate clinical practice, systematic divergences remain, including registerial shifts, insufficient clinical specificity, and inaccuracies in medication use and diagnostic procedures.
>
---
#### [new 072] FC-CONAN: An Exhaustively Paired Dataset for Robust Evaluation of Retrieval Systems
- **分类: cs.CL**

- **简介: 该论文提出FC-CONAN数据集，解决仇恨言论与反述语句配对不足的问题，通过全面组合45条HS和129条CN，提升反言论检索系统的评估准确性。**

- **链接: [https://arxiv.org/pdf/2601.01350v1](https://arxiv.org/pdf/2601.01350v1)**

> **作者:** Juan Junqueras; Florian Boudin; May-Myo Zin; Ha-Thanh Nguyen; Wachara Fungwacharakorn; Damián Ariel Furman; Akiko Aizawa; Ken Satoh
>
> **备注:** Presented at NeLaMKRR@KR, 2025 (arXiv:2511.09575)
>
> **摘要:** Hate speech (HS) is a critical issue in online discourse, and one promising strategy to counter it is through the use of counter-narratives (CNs). Datasets linking HS with CNs are essential for advancing counterspeech research. However, even flagship resources like CONAN (Chung et al., 2019) annotate only a sparse subset of all possible HS-CN pairs, limiting evaluation. We introduce FC-CONAN (Fully Connected CONAN), the first dataset created by exhaustively considering all combinations of 45 English HS messages and 129 CNs. A two-stage annotation process involving nine annotators and four validators produces four partitions-Diamond, Gold, Silver, and Bronze-that balance reliability and scale. None of the labeled pairs overlap with CONAN, uncovering hundreds of previously unlabelled positives. FC-CONAN enables more faithful evaluation of counterspeech retrieval systems and facilitates detailed error analysis. The dataset is publicly available.
>
---
#### [new 073] HalluZig: Hallucination Detection using Zigzag Persistence
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的事实检测任务，旨在解决大语言模型的幻觉问题。通过分析模型层间注意力的动态拓扑，利用zigzag persistence提取拓扑特征，实现幻觉检测。**

- **链接: [https://arxiv.org/pdf/2601.01552v1](https://arxiv.org/pdf/2601.01552v1)**

> **作者:** Shreyas N. Samaga; Gilberto Gonzalez Arroyo; Tamal K. Dey
>
> **摘要:** The factual reliability of Large Language Models (LLMs) remains a critical barrier to their adoption in high-stakes domains due to their propensity to hallucinate. Current detection methods often rely on surface-level signals from the model's output, overlooking the failures that occur within the model's internal reasoning process. In this paper, we introduce a new paradigm for hallucination detection by analyzing the dynamic topology of the evolution of model's layer-wise attention. We model the sequence of attention matrices as a zigzag graph filtration and use zigzag persistence, a tool from Topological Data Analysis, to extract a topological signature. Our core hypothesis is that factual and hallucinated generations exhibit distinct topological signatures. We validate our framework, HalluZig, on multiple benchmarks, demonstrating that it outperforms strong baselines. Furthermore, our analysis reveals that these topological signatures are generalizable across different models and hallucination detection is possible only using structural signatures from partial network depth.
>
---
#### [new 074] Intention Collapse: Intention-Level Metrics for Reasoning in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理研究，旨在解决语言生成中意图压缩问题。通过定义意图度量，分析不同推理模式对内部意图的影响。**

- **链接: [https://arxiv.org/pdf/2601.01011v1](https://arxiv.org/pdf/2601.01011v1)**

> **作者:** Patricio Vera
>
> **备注:** 21 pages, 4 figures, 3 tables. Code: https://github.com/patriciomvera/intention-collapse-experiments
>
> **摘要:** Every act of language generation compresses a rich internal state into a single token sequence. We call this process intention collapse: a many-to-one projection from a high dimensional intention space I into an external language space L. We formalize intention collapse for contemporary language models, define three simple, model agnostic intention metrics (intention entropy Hint, effective dimensionality dimeff, and latent knowledge recoverability Recov), and propose an empirical agenda for studying how inference time computation shapes internal intentions before they are verbalized. We also report a first small scale experiment. Using a 4 bit Mistral 7B model on 200 GSM8K problems, we compare a direct answer baseline, a chain of thought (CoT) regime, and a babble control. CoT raises accuracy from 5.5 percent to 53 percent, sharply reduces pre collapse intention entropy (from 1.42 to 0.37 bits), and shows higher global effective dimensionality than the other regimes despite producing fewer tokens than babble. At the same time, Hint has little item level predictive power, and a linear probe on I achieves AUROC 0.65 in the CoT regime but only about chance in the baseline regime, where it collapses to the majority class. These preliminary results indicate that intention level metrics can distinguish inference regimes and expose latent information that is partly lost during collapse, while also revealing important limitations of our current proxies
>
---
#### [new 075] EternalMath: A Living Benchmark of Frontier Mathematics that Evolves with Human Discovery
- **分类: cs.CL**

- **简介: 该论文属于数学推理评估任务，旨在解决静态基准不足的问题。提出自动化管道，将最新数学文献转化为可执行任务，构建动态评估集EternalMath。**

- **链接: [https://arxiv.org/pdf/2601.01400v1](https://arxiv.org/pdf/2601.01400v1)**

> **作者:** Jicheng Ma; Guohua Wang; Xinhua Feng; Yiming Liu; Zhichao Hu; Yuhong Liu
>
> **摘要:** Current evaluations of mathematical reasoning in large language models (LLMs) are dominated by static benchmarks, either derived from competition-style problems or curated through costly expert effort, resulting in limited coverage of research-level mathematics and rapid performance saturation. We propose a fully automated, theorem-grounded pipeline for evaluating frontier mathematical reasoning, which directly transforms recent peer-reviewed mathematical literature into executable and verifiable reasoning tasks. The pipeline identifies constructive or quantitative results, instantiates them into parameterized problem templates, and generates deterministic solutions through execution-based verification, enabling scalable, reproducible, and continuously updatable evaluation without reliance on large-scale expert authoring. By design, this approach supports temporal extensibility, intrinsic correctness checking, and domain-specific customization across mathematical subfields. Applying this pipeline yields \textbf{EternalMath}, an evolving evaluation suite derived from contemporary research papers. Experiments with state-of-the-art LLMs reveal substantial performance gaps, indicating that mathematical reasoning at the research frontier remains far from saturated and underscoring the need for evaluation methodologies that evolve in step with human mathematical discovery.
>
---
#### [new 076] MambaFormer: Token-Level Guided Routing Mixture-of-Experts for Accurate and Efficient Clinical Assistance
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决大模型在临床应用中的计算成本与效率矛盾。提出MambaFormer框架，通过动态路由选择不同专家模型，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01260v1](https://arxiv.org/pdf/2601.01260v1)**

> **作者:** Hamad Khan; Saddam Hussain Khan
>
> **备注:** 28 Pages, Tables 12, Figure 09
>
> **摘要:** The deployment of large language models (LLMs) in real-world clinical applications is constrained by the fundamental trade-off between computational cost and the efficiency of linear-time models. To address this, we propose an LLM-based MambaFormer hybrid Mixture-of-Experts (MoE) framework for efficient medical question-answering (QA) and clinical assistance. The MambaFormer employs a lightweight gating mechanism that performs token-level dynamic routing to a customized Transformer expert (ET5) for short, complex queries or to a State Space Model expert (EMamba) for long, high-throughput sequences. The customized EMamba and ET5 models are tailored to accommodate input sequence dimensionality, embedding structure, sequence length, and target-specific output heads, and are fine-tuned through transfer learning on a new, custom-designed DentalQA dataset. Moreover, intelligent routing decisions are driven by the contextual complexity of token embeddings, normalized sequence length, and domain-aware features, thereby enforcing a Pareto-optimal trade-off between inference latency and prediction accuracy. Furthermore, a novel utility-guided multi-objective loss jointly optimizes decisions, router parameters, routing behavior, expert utilization, and computational cost by adaptively regulating token-level expert activation. Finally, the proposed MambaFormer is cross-validated (holdout) for medical QA on the new, custom-designed DentalQA and PubMedQA datasets and compared with state-of-the-art techniques. The proposed MambaFormer outperforms (BERTScore = 0.9180) with ultra-low latency (0.077 s), delivering a 24.4 speedup over T5-Large and establishing a scalable solution for resource-constrained clinical deployment.
>
---
#### [new 077] ARGUS: Adaptive Rotation-Invariant Geometric Unsupervised System
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于分布漂移检测任务，解决高维数据流中漂移检测的挑战。提出Argus框架，通过固定空间划分和Voronoi图实现旋转不变的局部统计跟踪，提升检测精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.01297v1](https://arxiv.org/pdf/2601.01297v1)**

> **作者:** Anantha Sharma
>
> **备注:** 26 pages
>
> **摘要:** Detecting distributional drift in high-dimensional data streams presents fundamental challenges: global comparison methods scale poorly, projection-based approaches lose geometric structure, and re-clustering methods suffer from identity instability. This paper introduces Argus, A framework that reconceptualizes drift detection as tracking local statistics over a fixed spatial partition of the data manifold. The key contributions are fourfold. First, it is proved that Voronoi tessellations over canonical orthonormal frames yield drift metrics that are invariant to orthogonal transformations. The rotations and reflections that preserve Euclidean geometry. Second, it is established that this framework achieves O(N) complexity per snapshot while providing cell-level spatial localization of distributional change. Third, a graph-theoretic characterization of drift propagation is developed that distinguishes coherent distributional shifts from isolated perturbations. Fourth, product quantization tessellation is introduced for scaling to very high dimensions (d>500) by decomposing the space into independent subspaces and aggregating drift signals across subspaces. This paper formalizes the theoretical foundations, proves invariance properties, and presents experimental validation demonstrating that the framework correctly identifies drift under coordinate rotation while existing methods produce false positives. The tessellated approach offers a principled geometric foundation for distribution monitoring that preserves high-dimensional structure without the computational burden of pairwise comparisons.
>
---
#### [new 078] Exploring Approaches for Detecting Memorization of Recommender System Data in Large Language Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于数据泄露检测任务，旨在解决LLM中推荐系统数据记忆的检测问题。通过三种方法评估，探索自动检测数据泄露的可行性。**

- **链接: [https://arxiv.org/pdf/2601.02002v1](https://arxiv.org/pdf/2601.02002v1)**

> **作者:** Antonio Colacicco; Vito Guida; Dario Di Palma; Fedelucio Narducci; Tommaso Di Noia
>
> **摘要:** Large Language Models (LLMs) are increasingly applied in recommendation scenarios due to their strong natural language understanding and generation capabilities. However, they are trained on vast corpora whose contents are not publicly disclosed, raising concerns about data leakage. Recent work has shown that the MovieLens-1M dataset is memorized by both the LLaMA and OpenAI model families, but the extraction of such memorized data has so far relied exclusively on manual prompt engineering. In this paper, we pose three main questions: Is it possible to enhance manual prompting? Can LLM memorization be detected through methods beyond manual prompting? And can the detection of data leakage be automated? To address these questions, we evaluate three approaches: (i) jailbreak prompt engineering; (ii) unsupervised latent knowledge discovery, probing internal activations via Contrast-Consistent Search (CCS) and Cluster-Norm; and (iii) Automatic Prompt Engineering (APE), which frames prompt discovery as a meta-learning process that iteratively refines candidate instructions. Experiments on MovieLens-1M using LLaMA models show that jailbreak prompting does not improve the retrieval of memorized items and remains inconsistent; CCS reliably distinguishes genuine from fabricated movie titles but fails on numerical user and rating data; and APE retrieves item-level information with moderate success yet struggles to recover numerical interactions. These findings suggest that automatically optimizing prompts is the most promising strategy for extracting memorized samples.
>
---
#### [new 079] OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出OpenNovelty系统，用于可验证的学术新颖性评估。针对同行评审中新颖性判断困难的问题，系统通过四阶段流程分析论文，提供结构化报告，提升评审的公平性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.01576v1](https://arxiv.org/pdf/2601.01576v1)**

> **作者:** Ming Zhang; Kexin Tan; Yueyuan Huang; Yujiong Shen; Chunchun Ma; Li Ju; Xinran Zhang; Yuhui Wang; Wenqing Jing; Jingyi Deng; Huayu Sha; Binze Hu; Jingqi Tong; Changhao Jiang; Yage Geng; Yuankai Ying; Yue Zhang; Zhangyue Yin; Zhiheng Xi; Shihan Dou; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Evaluating novelty is critical yet challenging in peer review, as reviewers must assess submissions against a vast, rapidly evolving literature. This report presents OpenNovelty, an LLM-powered agentic system for transparent, evidence-based novelty analysis. The system operates through four phases: (1) extracting the core task and contribution claims to generate retrieval queries; (2) retrieving relevant prior work based on extracted queries via semantic search engine; (3) constructing a hierarchical taxonomy of core-task-related work and performing contribution-level full-text comparisons against each contribution; and (4) synthesizing all analyses into a structured novelty report with explicit citations and evidence snippets. Unlike naive LLM-based approaches, \textsc{OpenNovelty} grounds all assessments in retrieved real papers, ensuring verifiable judgments. We deploy our system on 500+ ICLR 2026 submissions with all reports publicly available on our website, and preliminary analysis suggests it can identify relevant prior work, including closely related papers that authors may overlook. OpenNovelty aims to empower the research community with a scalable tool that promotes fair, consistent, and evidence-backed peer review.
>
---
#### [new 080] LLM Collusion
- **分类: econ.TH; cs.AI; cs.CE; cs.CL; cs.GT**

- **简介: 该论文研究LLM在定价中引发的共谋问题，属于机制设计任务。通过分析模型参数，揭示了共谋发生的条件及影响因素。**

- **链接: [https://arxiv.org/pdf/2601.01279v1](https://arxiv.org/pdf/2601.01279v1)**

> **作者:** Shengyu Cao; Ming Hu
>
> **备注:** 44 pages
>
> **摘要:** We study how delegating pricing to large language models (LLMs) can facilitate collusion in a duopoly when both sellers rely on the same pre-trained model. The LLM is characterized by (i) a propensity parameter capturing its internal bias toward high-price recommendations and (ii) an output-fidelity parameter measuring how tightly outputs track that bias; the propensity evolves through retraining. We show that configuring LLMs for robustness and reproducibility can induce collusion via a phase transition: there exists a critical output-fidelity threshold that pins down long-run behavior. Below it, competitive pricing is the unique long-run outcome. Above it, the system is bistable, with competitive and collusive pricing both locally stable and the realized outcome determined by the model's initial preference. The collusive regime resembles tacit collusion: prices are elevated on average, yet occasional low-price recommendations provide plausible deniability. With perfect fidelity, full collusion emerges from any interior initial condition. For finite training batches of size $b$, infrequent retraining (driven by computational costs) further amplifies collusion: conditional on starting in the collusive basin, the probability of collusion approaches one as $b$ grows, since larger batches dampen stochastic fluctuations that might otherwise tip the system toward competition. The indeterminacy region shrinks at rate $O(1/\sqrt{b})$.
>
---
#### [new 081] LACONIC: Dense-Level Effectiveness for Scalable Sparse Retrieval via a Two-Phase Training Curriculum
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决稀疏检索性能不足的问题。提出LACONIC模型，通过两阶段训练提升效果，实现高效、低内存的检索方案。**

- **链接: [https://arxiv.org/pdf/2601.01684v1](https://arxiv.org/pdf/2601.01684v1)**

> **作者:** Zhichao Xu; Shengyao Zhuang; Crystina Zhang; Xueguang Ma; Yijun Tian; Maitrey Mehta; Jimmy Lin; Vivek Srikumar
>
> **摘要:** While dense retrieval models have become the standard for state-of-the-art information retrieval, their deployment is often constrained by high memory requirements and reliance on GPU accelerators for vector similarity search. Learned sparse retrieval offers a compelling alternative by enabling efficient search via inverted indices, yet it has historically received less attention than dense approaches. In this report, we introduce LACONIC, a family of learned sparse retrievers based on the Llama-3 architecture (1B, 3B, and 8B). We propose a streamlined two-phase training curriculum consisting of (1) weakly supervised pre-finetuning to adapt causal LLMs for bidirectional contextualization and (2) high-signal finetuning using curated hard negatives. Our results demonstrate that LACONIC effectively bridges the performance gap with dense models: the 8B variant achieves a state-of-the-art 60.2 nDCG on the MTEB Retrieval benchmark, ranking 15th on the leaderboard as of January 1, 2026, while utilizing 71\% less index memory than an equivalent dense model. By delivering high retrieval effectiveness on commodity CPU hardware with a fraction of the compute budget required by competing models, LACONIC provides a scalable and efficient solution for real-world search applications.
>
---
#### [new 082] Universal Conditional Logic: A Formal Language for Prompt Engineering
- **分类: cs.AI; cs.CL; cs.LG; cs.PL; cs.SE**

- **简介: 该论文提出通用条件逻辑（UCL），用于优化提示工程，解决模型交互效率问题。通过结构化方法实现性能提升与成本降低。**

- **链接: [https://arxiv.org/pdf/2601.00880v1](https://arxiv.org/pdf/2601.00880v1)**

> **作者:** Anthony Mikinka
>
> **备注:** 25 pages, 15 figures, 5 tables. Includes appendices with variable reference, pattern library, and O_s calculation examples. Supplementary materials: V1-V4.1 prompt source code and 305 model responses available at GitHub repositories
>
> **摘要:** We present Universal Conditional Logic (UCL), a mathematical framework for prompt optimization that transforms prompt engineering from heuristic practice into systematic optimization. Through systematic evaluation (N=305, 11 models, 4 iterations), we demonstrate significant token reduction (29.8%, t(10)=6.36, p < 0.001, Cohen's d = 2.01) with corresponding cost savings. UCL's structural overhead function O_s(A) explains version-specific performance differences through the Over-Specification Paradox: beyond threshold S* = 0.509, additional specification degrades performance quadratically. Core mechanisms -- indicator functions (I_i in {0,1}), structural overhead (O_s = gamma * sum(ln C_k)), early binding -- are validated. Notably, optimal UCL configuration varies by model architecture -- certain models (e.g., Llama 4 Scout) require version-specific adaptations (V4.1). This work establishes UCL as a calibratable framework for efficient LLM interaction, with model-family-specific optimization as a key research direction.
>
---
#### [new 083] SAFE-QAQ: End-to-End Slow-Thinking Audio-Text Fraud Detection via Reinforcement Learning
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频欺诈检测任务，解决传统方法依赖文本转录导致的误差问题。提出SAFE-QAQ框架，通过音频直接分析实现高效、实时的欺诈检测。**

- **链接: [https://arxiv.org/pdf/2601.01392v1](https://arxiv.org/pdf/2601.01392v1)**

> **作者:** Peidong Wang; Zhiming Ma; Xin Dai; Yongkang Liu; Shi Feng; Xiaocui Yang; Wenxing Hu; Zhihao Wang; Mingjun Pan; Li Yuan; Daling Wang
>
> **摘要:** Existing fraud detection methods predominantly rely on transcribed text, suffering from ASR errors and missing crucial acoustic cues like vocal tone and environmental context. This limits their effectiveness against complex deceptive strategies. To address these challenges, we first propose \textbf{SAFE-QAQ}, an end-to-end comprehensive framework for audio-based slow-thinking fraud detection. First, the SAFE-QAQ framework eliminates the impact of transcription errors on detection performance. Secondly, we propose rule-based slow-thinking reward mechanisms that systematically guide the system to identify fraud-indicative patterns by accurately capturing fine-grained audio details, through hierarchical reasoning processes. Besides, our framework introduces a dynamic risk assessment framework during live calls, enabling early detection and prevention of fraud. Experiments on the TeleAntiFraud-Bench demonstrate that SAFE-QAQ achieves dramatic improvements over existing methods in multiple key dimensions, including accuracy, inference efficiency, and real-time processing capabilities. Currently deployed and analyzing over 70,000 calls daily, SAFE-QAQ effectively automates complex fraud detection, reducing human workload and financial losses. Code: https://anonymous.4open.science/r/SAFE-QAQ.
>
---
#### [new 084] Bayesian Orchestration of Multi-LLM Agents for Cost-Aware Sequential Decision-Making
- **分类: cs.AI; cs.CL; cs.ET**

- **简介: 该论文属于成本感知的序列决策任务，解决多LLM代理协同问题。通过贝叶斯框架聚合多个LLM，降低决策成本并提升公平性。**

- **链接: [https://arxiv.org/pdf/2601.01522v1](https://arxiv.org/pdf/2601.01522v1)**

> **作者:** Danial Amin
>
> **摘要:** Large language models (LLMs) are increasingly deployed as autonomous decision agents in settings with asymmetric error costs: hiring (missed talent vs wasted interviews), medical triage (missed emergencies vs unnecessary escalation), and fraud detection (approved fraud vs declined legitimate payments). The dominant design queries a single LLM for a posterior over states, thresholds "confidence," and acts; we prove this is inadequate for sequential decisions with costs. We propose a Bayesian, cost-aware multi-LLM orchestration framework that treats LLMs as approximate likelihood models rather than classifiers. For each candidate state, we elicit likelihoods via contrastive prompting, aggregate across diverse models with robust statistics, and update beliefs with Bayes rule under explicit priors as new evidence arrives. This enables coherent belief updating, expected-cost action selection, principled information gathering via value of information, and fairness gains via ensemble bias mitigation. In resume screening with costs of 40000 USD per missed hire, 2500 USD per interview, and 150 USD per phone screen, experiments on 1000 resumes using five LLMs (GPT-4o, Claude 4.5 Sonnet, Gemini Pro, Grok, DeepSeek) reduce total cost by 294000 USD (34 percent) versus the best single-LLM baseline and improve demographic parity by 45 percent (max group gap 22 to 5 percentage points). Ablations attribute 51 percent of savings to multi-LLM aggregation, 43 percent to sequential updating, and 20 percent to disagreement-triggered information gathering, consistent with the theoretical benefits of correct probabilistic foundations.
>
---
#### [new 085] Output Embedding Centering for Stable LLM Pretraining
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型训练任务，解决预训练过程中的训练不稳定问题。通过分析输出嵌入几何，提出OEC方法抑制logit发散，提升训练稳定性。**

- **链接: [https://arxiv.org/pdf/2601.02031v1](https://arxiv.org/pdf/2601.02031v1)**

> **作者:** Felix Stollenwerk; Anna Lokrantz; Niclas Hertzberg
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Pretraining of large language models is not only expensive but also prone to certain training instabilities. A specific instability that often occurs for large learning rates at the end of training is output logit divergence. The most widely used mitigation strategy, z-loss, merely addresses the symptoms rather than the underlying cause of the problem. In this paper, we analyze the instability from the perspective of the output embeddings' geometry and identify its cause. Based on this, we propose output embedding centering (OEC) as a new mitigation strategy, and prove that it suppresses output logit divergence. OEC can be implemented in two different ways, as a deterministic operation called μ-centering, or a regularization method called μ-loss. Our experiments show that both variants outperform z-loss in terms of training stability and learning rate sensitivity. In particular, they ensure that training converges even for large learning rates when z-loss fails. Furthermore, we find that μ-loss is significantly less sensitive to regularization hyperparameter tuning than z-loss.
>
---
#### [new 086] A neural network for modeling human concept formation, understanding and communication
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文属于认知科学与人工智能交叉领域，旨在揭示人类概念形成与理解的计算机制。提出CATS网络模型，解决概念抽象与跨任务迁移问题，通过神经网络模拟人类脑区功能。**

- **链接: [https://arxiv.org/pdf/2601.02010v1](https://arxiv.org/pdf/2601.02010v1)**

> **作者:** Liangxuan Guo; Haoyang Chen; Yang Chen; Yanchao Bi; Shan Yu
>
> **备注:** 6 main figures, 5 extended data figures and 4 supplementary figures
>
> **摘要:** A remarkable capability of the human brain is to form more abstract conceptual representations from sensorimotor experiences and flexibly apply them independent of direct sensory inputs. However, the computational mechanism underlying this ability remains poorly understood. Here, we present a dual-module neural network framework, the CATS Net, to bridge this gap. Our model consists of a concept-abstraction module that extracts low-dimensional conceptual representations, and a task-solving module that performs visual judgement tasks under the hierarchical gating control of the formed concepts. The system develops transferable semantic structure based on concept representations that enable cross-network knowledge transfer through conceptual communication. Model-brain fitting analyses reveal that these emergent concept spaces align with both neurocognitive semantic model and brain response structures in the human ventral occipitotemporal cortex, while the gating mechanisms mirror that in the semantic control brain network. This work establishes a unified computational framework that can offer mechanistic insights for understanding human conceptual cognition and engineering artificial systems with human-like conceptual intelligence.
>
---
#### [new 087] Entity-Aware and Secure Query Optimization in Database Using Named Entity Recognition
- **分类: cs.DB; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决云数据库中敏感信息识别与高效查询问题。通过结合NER和加密技术，实现敏感数据的自动识别与安全查询优化。**

- **链接: [https://arxiv.org/pdf/2601.01254v1](https://arxiv.org/pdf/2601.01254v1)**

> **作者:** Azrin Sultana; Hasibur Rashid Chayon
>
> **备注:** 48 pages, 15 figures, 14 tables
>
> **摘要:** Cloud storage has become the backbone of modern data infrastructure, yet privacy and efficient data retrieval remain significant challenges. Traditional privacy-preserving approaches primarily focus on enhancing database security but fail to address the automatic identification of sensitive information before encryption. This can dramatically reduce query processing time and mitigate errors during manual identification of sensitive information, thereby reducing potential privacy risks. To address this limitation, this research proposes an intelligent privacy-preserving query optimization framework that integrates Named Entity Recognition (NER) to detect sensitive information in queries, utilizing secure data encryption and query optimization techniques for both sensitive and non-sensitive data in parallel, thereby enabling efficient database optimization. Combined deep learning algorithms and transformer-based models to detect and classify sensitive entities with high precision, and the Advanced Encryption Standard (AES) algorithm to encrypt, with blind indexing to secure search functionality of the sensitive data, whereas non-sensitive data was divided into groups using the K-means algorithm, along with a rank search for optimization. Among all NER models, the Deep Belief Network combined with Long Short-Term Memory (DBN-LSTM) delivers the best performance, with an accuracy of 93% and precision (94%), recall, and F1 score of 93%, and 93%, respectively. Besides, encrypted search achieved considerably faster results with the help of blind indexing, and non-sensitive data fetching also outperformed traditional clustering-based searches. By integrating sensitive data detection, encryption, and query optimization, this work advances the state of privacy-preserving computation in modern cloud infrastructures.
>
---
#### [new 088] Exploring Diversity, Novelty, and Popularity Bias in ChatGPT's Recommendations
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在分析ChatGPT在推荐中的多样性、新颖性和流行度偏差。研究对比了ChatGPT-3.5和ChatGPT-4的推荐效果，评估其在不同场景下的表现。**

- **链接: [https://arxiv.org/pdf/2601.01997v1](https://arxiv.org/pdf/2601.01997v1)**

> **作者:** Dario Di Palma; Giovanni Maria Biancofiore; Vito Walter Anelli; Fedelucio Narducci; Tommaso Di Noia
>
> **摘要:** ChatGPT has emerged as a versatile tool, demonstrating capabilities across diverse domains. Given these successes, the Recommender Systems (RSs) community has begun investigating its applications within recommendation scenarios primarily focusing on accuracy. While the integration of ChatGPT into RSs has garnered significant attention, a comprehensive analysis of its performance across various dimensions remains largely unexplored. Specifically, the capabilities of providing diverse and novel recommendations or exploring potential biases such as popularity bias have not been thoroughly examined. As the use of these models continues to expand, understanding these aspects is crucial for enhancing user satisfaction and achieving long-term personalization. This study investigates the recommendations provided by ChatGPT-3.5 and ChatGPT-4 by assessing ChatGPT's capabilities in terms of diversity, novelty, and popularity bias. We evaluate these models on three distinct datasets and assess their performance in Top-N recommendation and cold-start scenarios. The findings reveal that ChatGPT-4 matches or surpasses traditional recommenders, demonstrating the ability to balance novelty and diversity in recommendations. Furthermore, in the cold-start scenario, ChatGPT models exhibit superior performance in both accuracy and novelty, suggesting they can be particularly beneficial for new users. This research highlights the strengths and limitations of ChatGPT's recommendations, offering new perspectives on the capacity of these models to provide recommendations beyond accuracy-focused metrics.
>
---
#### [new 089] Bridging the Semantic Gap for Categorical Data Clustering via Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数据聚类任务，旨在解决类别数据相似性度量不足导致的语义鸿沟问题。通过引入大语言模型增强语义表示，提升聚类效果。**

- **链接: [https://arxiv.org/pdf/2601.01162v1](https://arxiv.org/pdf/2601.01162v1)**

> **作者:** Zihua Yang; Xin Liao; Yiqun Zhang; Yiu-ming Cheung
>
> **备注:** Submitted to ICPR 2026
>
> **摘要:** Categorical data are prevalent in domains such as healthcare, marketing, and bioinformatics, where clustering serves as a fundamental tool for pattern discovery. A core challenge in categorical data clustering lies in measuring similarity among attribute values that lack inherent ordering or distance. Without appropriate similarity measures, values are often treated as equidistant, creating a semantic gap that obscures latent structures and degrades clustering quality. Although existing methods infer value relationships from within-dataset co-occurrence patterns, such inference becomes unreliable when samples are limited, leaving the semantic context of the data underexplored. To bridge this gap, we present ARISE (Attention-weighted Representation with Integrated Semantic Embeddings), which draws on external semantic knowledge from Large Language Models (LLMs) to construct semantic-aware representations that complement the metric space of categorical data for accurate clustering. That is, LLM is adopted to describe attribute values for representation enhancement, and the LLM-enhanced embeddings are combined with the original data to explore semantically prominent clusters. Experiments on eight benchmark datasets demonstrate consistent improvements over seven representative counterparts, with gains of 19-27%. Code is available at https://github.com/develop-yang/ARISE
>
---
#### [new 090] Simulated Reasoning is Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨基础模型的推理能力，指出其通过模仿思考过程实现推理，而非依赖传统符号逻辑。任务是分析此类推理的性质与局限，解决其缺乏常识导致的脆弱性问题。**

- **链接: [https://arxiv.org/pdf/2601.02043v1](https://arxiv.org/pdf/2601.02043v1)**

> **作者:** Hendrik Kempt; Alon Lavie
>
> **备注:** 21 pages
>
> **摘要:** Reasoning has long been understood as a pathway between stages of understanding. Proper reasoning leads to understanding of a given subject. This reasoning was conceptualized as a process of understanding in a particular way, i.e., "symbolic reasoning". Foundational Models (FM) demonstrate that this is not a necessary condition for many reasoning tasks: they can "reason" by way of imitating the process of "thinking out loud", testing the produced pathways, and iterating on these pathways on their own. This leads to some form of reasoning that can solve problems on its own or with few-shot learning, but appears fundamentally different from human reasoning due to its lack of grounding and common sense, leading to brittleness of the reasoning process. These insights promise to substantially alter our assessment of reasoning and its necessary conditions, but also inform the approaches to safety and robust defences against this brittleness of FMs. This paper offers and discusses several philosophical interpretations of this phenomenon, argues that the previously apt metaphor of the "stochastic parrot" has lost its relevance and thus should be abandoned, and reflects on different normative elements in the safety- and appropriateness-considerations emerging from these reasoning models and their growing capacity.
>
---
#### [new 091] Entropy-Adaptive Fine-Tuning: Resolving Confident Conflicts to Mitigate Forgetting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理领域，解决SFT导致的遗忘问题。通过引入熵适应微调，区分不确定性与知识冲突，减少梯度破坏，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.02151v1](https://arxiv.org/pdf/2601.02151v1)**

> **作者:** Muxi Diao; Lele Yang; Wuxuan Gong; Yutong Zhang; Zhonghao Yan; Yufei Han; Kongming Liang; Weiran Xu; Zhanyu Ma
>
> **摘要:** Supervised Fine-Tuning (SFT) is the standard paradigm for domain adaptation, yet it frequently incurs the cost of catastrophic forgetting. In sharp contrast, on-policy Reinforcement Learning (RL) effectively preserves general capabilities. We investigate this discrepancy and identify a fundamental distributional gap: while RL aligns with the model's internal belief, SFT forces the model to fit external supervision. This mismatch often manifests as "Confident Conflicts" tokens characterized by low probability but low entropy. In these instances, the model is highly confident in its own prediction but is forced to learn a divergent ground truth, triggering destructive gradient updates. To address this, we propose Entropy-Adaptive Fine-Tuning (EAFT). Unlike methods relying solely on prediction probability, EAFT utilizes token-level entropy as a gating mechanism to distinguish between epistemic uncertainty and knowledge conflict. This allows the model to learn from uncertain samples while suppressing gradients on conflicting data. Extensive experiments on Qwen and GLM series (ranging from 4B to 32B parameters) across mathematical, medical, and agentic domains confirm our hypothesis. EAFT consistently matches the downstream performance of standard SFT while significantly mitigating the degradation of general capabilities.
>
---
#### [new 092] The Gray Area: Characterizing Moderator Disagreement on Reddit
- **分类: cs.CY; cs.CL; cs.IT**

- **简介: 该论文研究在线内容审核中的争议问题，属于内容 moderation 任务。它分析了Reddit moderators的分歧，探讨了灰色区域案例的复杂性与挑战。**

- **链接: [https://arxiv.org/pdf/2601.01620v1](https://arxiv.org/pdf/2601.01620v1)**

> **作者:** Shayan Alipour; Shruti Phadke; Seyed Shahabeddin Mousavi; Amirhossein Afsharrad; Morteza Zihayat; Mattia Samory
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** Volunteer moderators play a crucial role in sustaining online dialogue, but they often disagree about what should or should not be allowed. In this paper, we study the complexity of content moderation with a focus on disagreements between moderators, which we term the ``gray area'' of moderation. Leveraging 5 years and 4.3 million moderation log entries from 24 subreddits of different topics and sizes, we characterize how gray area, or disputed cases, differ from undisputed cases. We show that one-in-seven moderation cases are disputed among moderators, often addressing transgressions where users' intent is not directly legible, such as in trolling and brigading, as well as tensions around community governance. This is concerning, as almost half of all gray area cases involved automated moderation decisions. Through information-theoretic evaluations, we demonstrate that gray area cases are inherently harder to adjudicate than undisputed cases and show that state-of-the-art language models struggle to adjudicate them. We highlight the key role of expert human moderators in overseeing the moderation process and provide insights about the challenges of current moderation processes and tools.
>
---
#### [new 093] CogCanvas: Compression-Resistant Cognitive Artifacts for Long LLM Conversations
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出CogCanvas，解决长对话中信息丢失问题，通过提取认知工件构建时间感知图，提升信息检索效果。**

- **链接: [https://arxiv.org/pdf/2601.00821v1](https://arxiv.org/pdf/2601.00821v1)**

> **作者:** Tao An
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Large language models face a fundamental tension between context window limits and information fidelity in long conversations. Existing approaches--truncation and summarization--either discard early information or lose nuanced details. We introduce CogCanvas, a training-free framework that extracts verbatim-grounded cognitive artifacts (decisions, facts, reminders) from conversation turns and organizes them into a temporal-aware graph for compression-resistant retrieval. On the LoCoMo benchmark, CogCanvas achieves 34.7% overall accuracy, outperforming RAG (25.6%, +9.1pp) and GraphRAG (13.7%, +21.0pp). The advantage is most pronounced on temporal reasoning: 31.5% vs. 9.3% (RAG) and 5.0% (GraphRAG)--a +530% relative improvement. On multi-hop causal reasoning, CogCanvas achieves 81.0% pass rate vs. 40.0% for GraphRAG (+41.0pp). Controlled benchmarks show 97.5% recall (+78.5pp vs. summarization) with 93.0% exact match preservation. While heavily-optimized approaches achieve higher absolute scores through dedicated training (EverMemOS: approximately 92%), our training-free approach provides practitioners with an immediately-deployable alternative that significantly outperforms standard baselines. Code and data: https://github.com/tao-hpu/cog-canvas.
>
---
#### [new 094] A Platform for Interactive AI Character Experiences
- **分类: cs.HC; cs.AI; cs.CL; cs.GR**

- **简介: 该论文提出一个交互式AI角色平台，解决构建拟人化角色的多个技术难题，实现故事驱动的对话体验。**

- **链接: [https://arxiv.org/pdf/2601.01027v1](https://arxiv.org/pdf/2601.01027v1)**

> **作者:** Rafael Wampfler; Chen Yang; Dillon Elste; Nikola Kovacevic; Philine Witzig; Markus Gross
>
> **摘要:** From movie characters to modern science fiction - bringing characters into interactive, story-driven conversations has captured imaginations across generations. Achieving this vision is highly challenging and requires much more than just language modeling. It involves numerous complex AI challenges, such as conversational AI, maintaining character integrity, managing personality and emotions, handling knowledge and memory, synthesizing voice, generating animations, enabling real-world interactions, and integration with physical environments. Recent advancements in the development of foundation models, prompt engineering, and fine-tuning for downstream tasks have enabled researchers to address these individual challenges. However, combining these technologies for interactive characters remains an open problem. We present a system and platform for conveniently designing believable digital characters, enabling a conversational and story-driven experience while providing solutions to all of the technical challenges. As a proof-of-concept, we introduce Digital Einstein, which allows users to engage in conversations with a digital representation of Albert Einstein about his life, research, and persona. While Digital Einstein exemplifies our methods for a specific character, our system is flexible and generalizes to any story-driven or conversational character. By unifying these diverse AI components into a single, easy-to-adapt platform, our work paves the way for immersive character experiences, turning the dream of lifelike, story-based interactions into a reality.
>
---
#### [new 095] Aletheia: Quantifying Cognitive Conviction in Reasoning Models via Regularized Inverse Confusion Matrix
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI评估任务，旨在量化推理模型的“认知确信度”。通过正则化逆混淆矩阵方法，解决现有评估无法衡量信念深度的问题。**

- **链接: [https://arxiv.org/pdf/2601.01532v1](https://arxiv.org/pdf/2601.01532v1)**

> **作者:** Fanzhe Fu
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** In the progressive journey toward Artificial General Intelligence (AGI), current evaluation paradigms face an epistemological crisis. Static benchmarks measure knowledge breadth but fail to quantify the depth of belief. While Simhi et al. (2025) defined the CHOKE phenomenon in standard QA, we extend this framework to quantify "Cognitive Conviction" in System 2 reasoning models. We propose Project Aletheia, a cognitive physics framework that employs Tikhonov Regularization to invert the judge's confusion matrix. To validate this methodology without relying on opaque private data, we implement a Synthetic Proxy Protocol. Our preliminary pilot study on 2025 baselines (e.g., DeepSeek-R1, OpenAI o1) suggests that while reasoning models act as a "cognitive buffer," they may exhibit "Defensive OverThinking" under adversarial pressure. Furthermore, we introduce the Aligned Conviction Score (S_aligned) to verify that conviction does not compromise safety. This work serves as a blueprint for measuring AI scientific integrity.
>
---
#### [new 096] EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出EverMemOS，解决长周期交互中记忆管理问题，通过自组织记忆系统实现有效用户状态跟踪与推理支持。**

- **链接: [https://arxiv.org/pdf/2601.02163v1](https://arxiv.org/pdf/2601.02163v1)**

> **作者:** Chuanrui Hu; Xingze Gao; Zuyi Zhou; Dannong Xu; Yi Bai; Xintong Li; Hui Zhang; Tong Li; Chong Zhang; Lidong Bing; Yafeng Deng
>
> **备注:** 16 pages, 6 figures, 12 tables. Code available at https://github.com/EverMind-AI/EverMemOS
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed as long-term interactive agents, yet their limited context windows make it difficult to sustain coherent behavior over extended interactions. Existing memory systems often store isolated records and retrieve fragments, limiting their ability to consolidate evolving user states and resolve conflicts. We introduce EverMemOS, a self-organizing memory operating system that implements an engram-inspired lifecycle for computational memory. Episodic Trace Formation converts dialogue streams into MemCells that capture episodic traces, atomic facts, and time-bounded Foresight signals. Semantic Consolidation organizes MemCells into thematic MemScenes, distilling stable semantic structures and updating user profiles. Reconstructive Recollection performs MemScene-guided agentic retrieval to compose the necessary and sufficient context for downstream reasoning. Experiments on LoCoMo and LongMemEval show that EverMemOS achieves state-of-the-art performance on memory-augmented reasoning tasks. We further report a profile study on PersonaMem v2 and qualitative case studies illustrating chat-oriented capabilities such as user profiling and Foresight. Code is available at https://github.com/EverMind-AI/EverMemOS.
>
---
#### [new 097] Context-Free Recognition with Transformers
- **分类: cs.LG; cs.CC; cs.CL; cs.FL**

- **简介: 该论文研究Transformer在上下文无关语言识别中的能力。任务是解决其在语法处理上的局限性，通过引入循环层和填充token，证明其可识别所有上下文无关语言。**

- **链接: [https://arxiv.org/pdf/2601.01754v1](https://arxiv.org/pdf/2601.01754v1)**

> **作者:** Selim Jerad; Anej Svete; Sophie Hao; Ryan Cotterell; William Merrill
>
> **摘要:** Transformers excel on tasks that process well-formed inputs according to some grammar, such as natural language and code. However, it remains unclear how they can process grammatical syntax. In fact, under standard complexity conjectures, standard transformers cannot recognize context-free languages (CFLs), a canonical formalism to describe syntax, or even regular languages, a subclass of CFLs (Merrill et al., 2022). Merrill & Sabharwal (2024) show that $\mathcal{O}(\log n)$ looping layers (w.r.t. input length $n$) allows transformers to recognize regular languages, but the question of context-free recognition remained open. In this work, we show that looped transformers with $\mathcal{O}(\log n)$ looping layers and $\mathcal{O}(n^6)$ padding tokens can recognize all CFLs. However, training and inference with $\mathcal{O}(n^6)$ padding tokens is potentially impractical. Fortunately, we show that, for natural subclasses such as unambiguous CFLs, the recognition problem on transformers becomes more tractable, requiring $\mathcal{O}(n^3)$ padding. We empirically validate our results and show that looping helps on a language that provably requires logarithmic depth. Overall, our results shed light on the intricacy of CFL recognition by transformers: While general recognition may require an intractable amount of padding, natural constraints such as unambiguity yield efficient recognition algorithms.
>
---
#### [new 098] When to Ponder: Adaptive Compute Allocation for Code Generation via Test-Time Training
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于代码生成任务，解决大模型计算资源分配不均的问题。提出PonderTTT方法，通过自监督重建损失选择性触发TTT更新，实现高效计算分配。**

- **链接: [https://arxiv.org/pdf/2601.00894v1](https://arxiv.org/pdf/2601.00894v1)**

> **作者:** Gihyeon Sim
>
> **备注:** 14 pages, 1 figure, 14 tables, code available at https://github.com/deveworld/ponderTTT
>
> **摘要:** Large language models apply uniform computation to all inputs, regardless of difficulty. We propose PonderTTT, a gating strategy using the TTT layer's self-supervised reconstruction loss to selectively trigger Test-Time Training (TTT) updates. The gating decision itself is training-free--requiring no learned classifier or auxiliary networks; only a single scalar threshold is initially calibrated on unlabeled data and continuously adapted via EMA to maintain target update rates. Our experiments with GPT-2 models (124M to 1.5B) on code language modeling (The Stack v2, teacher-forced perplexity) demonstrate that this signal is inference-compatible, requiring no ground-truth labels. Our Reconstruction Gating achieves 82-89% Oracle Recovery while being fully training-free, significantly outperforming Random Skip baselines (up to 16% lower loss on OOD languages).
>
---
#### [new 099] AppellateGen: A Benchmark for Appellate Legal Judgment Generation
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 该论文属于法律判决生成任务，旨在解决上诉阶段判决生成问题。提出AppellateGen基准和SLMAS系统，以模拟司法流程并提升逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2601.01331v1](https://arxiv.org/pdf/2601.01331v1)**

> **作者:** Hongkun Yang; Lionel Z. Wang; Wei Fan; Yiran Hu; Lixu Wang; Chenyu Liu; Shenghong Fu; Haoyang Li; Xin Xu; Jiexin Zheng; Wei Dong
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** Legal judgment generation is a critical task in legal intelligence. However, existing research in legal judgment generation has predominantly focused on first-instance trials, relying on static fact-to-verdict mappings while neglecting the dialectical nature of appellate (second-instance) review. To address this, we introduce AppellateGen, a benchmark for second-instance legal judgment generation comprising 7,351 case pairs. The task requires models to draft legally binding judgments by reasoning over the initial verdict and evidentiary updates, thereby modeling the causal dependency between trial stages. We further propose a judicial Standard Operating Procedure (SOP)-based Legal Multi-Agent System (SLMAS) to simulate judicial workflows, which decomposes the generation process into discrete stages of issue identification, retrieval, and drafting. Experimental results indicate that while SLMAS improves logical consistency, the complexity of appellate reasoning remains a substantial challenge for current LLMs. The dataset and code are publicly available at: https://anonymous.4open.science/r/AppellateGen-5763.
>
---
#### [new 100] HyperCLOVA X 8B Omni
- **分类: cs.LG; cs.AI; cs.CL; cs.SD**

- **简介: 该论文介绍HyperCLOVA X 8B Omni，一款支持文本、音频和视觉输入输出的多模态模型，解决跨模态交互问题，通过统一接口实现多模态理解和生成。**

- **链接: [https://arxiv.org/pdf/2601.01792v1](https://arxiv.org/pdf/2601.01792v1)**

> **作者:** NAVER Cloud HyperCLOVA X Team
>
> **备注:** Technical Report
>
> **摘要:** In this report, we present HyperCLOVA X 8B Omni, the first any-to-any omnimodal model in the HyperCLOVA X family that supports text, audio, and vision as both inputs and outputs. By consolidating multimodal understanding and generation into a single model rather than separate modality-specific pipelines, HyperCLOVA X 8B Omni serves as an 8B-scale omni-pathfinding point toward practical any-to-any omni assistants. At a high level, the model unifies modalities through a shared next-token prediction interface over an interleaved multimodal sequence, while vision and audio encoders inject continuous embeddings for fine-grained understanding and grounding. Empirical evaluations demonstrate competitive performance against comparably sized models across diverse input-output combinations spanning text, audio, and vision, in both Korean and English. We anticipate that the open-weight release of HyperCLOVA X 8B Omni will support a wide range of research and deployment scenarios.
>
---
#### [new 101] Query-Document Dense Vectors for LLM Relevance Judgment Bias Analysis
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在分析LLM在相关性判断中的系统性偏差。通过构建查询-文档向量表示，发现LLM与人类在特定语义簇中存在系统性分歧，揭示其判断弱点。**

- **链接: [https://arxiv.org/pdf/2601.01751v1](https://arxiv.org/pdf/2601.01751v1)**

> **作者:** Samaneh Mohtadi; Gianluca Demartini
>
> **备注:** Accepted for presentation at the ECIR 2026 Full Papers track
>
> **摘要:** Large Language Models (LLMs) have been used as relevance assessors for Information Retrieval (IR) evaluation collection creation due to reduced cost and increased scalability as compared to human assessors. While previous research has looked at the reliability of LLMs as compared to human assessors, in this work, we aim to understand if LLMs make systematic mistakes when judging relevance, rather than just understanding how good they are on average. To this aim, we propose a novel representational method for queries and documents that allows us to analyze relevance label distributions and compare LLM and human labels to identify patterns of disagreement and localize systematic areas of disagreement. We introduce a clustering-based framework that embeds query-document (Q-D) pairs into a joint semantic space, treating relevance as a relational property. Experiments on TREC Deep Learning 2019 and 2020 show that systematic disagreement between humans and LLMs is concentrated in specific semantic clusters rather than distributed randomly. Query-level analyses reveal recurring failures, most often in definition-seeking, policy-related, or ambiguous contexts. Queries with large variation in agreement across their clusters emerge as disagreement hotspots, where LLMs tend to under-recall relevant content or over-include irrelevant material. This framework links global diagnostics with localized clustering to uncover hidden weaknesses in LLM judgments, enabling bias-aware and more reliable IR evaluation.
>
---
#### [new 102] The Invisible Hand of AI Libraries Shaping Open Source Projects and Communities
- **分类: cs.SE; cs.AI; cs.CL; cs.IR; cs.PL**

- **简介: 该论文属于软件工程领域，旨在研究AI库对开源项目的影响。通过分析大量代码库，比较采用与未采用AI库的项目差异，探索AI如何重塑开发实践。**

- **链接: [https://arxiv.org/pdf/2601.01944v1](https://arxiv.org/pdf/2601.01944v1)**

> **作者:** Matteo Esposito; Andrea Janes; Valentina Lenarduzzi; Davide Taibi
>
> **备注:** ACCEPTED REGISTERED REPORT AT SANER (CORE A*) 2026
>
> **摘要:** In the early 1980s, Open Source Software emerged as a revolutionary concept amidst the dominance of proprietary software. What began as a revolutionary idea has now become the cornerstone of computer science. Amidst OSS projects, AI is increasing its presence and relevance. However, despite the growing popularity of AI, its adoption and impacts on OSS projects remain underexplored. We aim to assess the adoption of AI libraries in Python and Java OSS projects and examine how they shape development, including the technical ecosystem and community engagement. To this end, we will perform a large-scale analysis on 157.7k potential OSS repositories, employing repository metrics and software metrics to compare projects adopting AI libraries against those that do not. We expect to identify measurable differences in development activity, community engagement, and code complexity between OSS projects that adopt AI libraries and those that do not, offering evidence-based insights into how AI integration reshapes software development practices.
>
---
#### [new 103] Entropy-Aligned Decoding of LMs for Better Writing and Reasoning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言生成任务，解决语言模型生成质量低的问题。提出EPIC方法，通过熵对齐提升生成多样性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.01714v1](https://arxiv.org/pdf/2601.01714v1)**

> **作者:** Kareem Ahmed; Sameer Singh
>
> **摘要:** Language models (LMs) are trained on billions of tokens in an attempt to recover the true language distribution. Still, vanilla random sampling from LMs yields low quality generations. Decoding algorithms attempt to restrict the LM distribution to a set of high-probability continuations, but rely on greedy heuristics that introduce myopic distortions, yielding sentences that are homogeneous, repetitive and incoherent. In this paper, we introduce EPIC, a hyperparameter-free decoding approach that incorporates the entropy of future trajectories into LM decoding. EPIC explicitly regulates the amount of uncertainty expressed at every step of generation, aligning the sampling distribution's entropy to the aleatoric (data) uncertainty. Through Entropy-Aware Lazy Gumbel-Max sampling, EPIC manages to be exact, while also being efficient, requiring only a sublinear number of entropy evaluations per step. Unlike current baselines, EPIC yields sampling distributions that are empirically well-aligned with the entropy of the underlying data distribution. Across creative writing and summarization tasks, EPIC consistently improves LM-as-judge preference win-rates over widely used decoding strategies. These preference gains are complemented by automatic metrics, showing that EPIC produces more diverse generations and more faithful summaries. We also evaluate EPIC on mathematical reasoning, where it outperforms all baselines.
>
---
#### [new 104] Measuring Social Media Polarization Using Large Language Models and Heuristic Rules
- **分类: cs.SI; cs.AI; cs.CL**

- **简介: 该论文属于社会媒体极化分析任务，旨在量化情感极化问题。通过结合大语言模型和启发式规则，提取立场、情感和互动模式，提出一种可扩展的极化测量框架。**

- **链接: [https://arxiv.org/pdf/2601.00927v1](https://arxiv.org/pdf/2601.00927v1)**

> **作者:** Jawad Chowdhury; Rezaur Rashid; Gabriel Terejanu
>
> **备注:** Foundations and Applications of Big Data Analytics (FAB), Niagara Falls, Canada, 2025
>
> **摘要:** Understanding affective polarization in online discourse is crucial for evaluating the societal impact of social media interactions. This study presents a novel framework that leverages large language models (LLMs) and domain-informed heuristics to systematically analyze and quantify affective polarization in discussions on divisive topics such as climate change and gun control. Unlike most prior approaches that relied on sentiment analysis or predefined classifiers, our method integrates LLMs to extract stance, affective tone, and agreement patterns from large-scale social media discussions. We then apply a rule-based scoring system capable of quantifying affective polarization even in small conversations consisting of single interactions, based on stance alignment, emotional content, and interaction dynamics. Our analysis reveals distinct polarization patterns that are event dependent: (i) anticipation-driven polarization, where extreme polarization escalates before well-publicized events, and (ii) reactive polarization, where intense affective polarization spikes immediately after sudden, high-impact events. By combining AI-driven content annotation with domain-informed scoring, our framework offers a scalable and interpretable approach to measuring affective polarization. The source code is publicly available at: https://github.com/hasanjawad001/llm-social-media-polarization.
>
---
#### [new 105] Reliability Under Randomness: An Empirical Analysis of Sparse and Dense Language Models Across Decoding Temperatures
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究稀疏与密集模型在不同解码温度下的可靠性问题。通过实验对比分析，探讨稀疏架构是否影响输出稳定性。**

- **链接: [https://arxiv.org/pdf/2601.00942v1](https://arxiv.org/pdf/2601.00942v1)**

> **作者:** Kabir Grover
>
> **摘要:** The increasing prevalence of sparse Mixture-of-Experts (MoE) architectures in large language models raises important questions regarding their reliability under stochastic decoding. While conditional computation enables substantial gains in computational efficiency, it remains unclear whether the interaction between sparse routing and temperature-based sampling compromises output stability relative to dense architectures. This work investigates whether conditional computation in MoE models amplifies decoding-induced randomness, leading to reduced reliability as temperature increases. We evaluate three representative models: OLMoE-7B (sparse base), Mixtral-8x7B (sparse instruction-tuned), and Qwen2.5-3B (dense instruction-tuned) on deterministic arithmetic reasoning tasks with objectively verifiable answers. Experiments span four decoding configurations, ranging from greedy decoding to T=1.0. Our evaluation encompasses accuracy, format compliance, output consistency across repeated generations, and confidence metrics, totaling 9,360 model generations. Results demonstrate that the sparse instruction-tuned model exhibits stability comparable to the dense instruction-tuned model across all decoding temperatures, while the sparse base model shows systematic degradation as temperature increases. These findings indicate that instruction tuning, rather than architectural sparsity, is the primary determinant of robustness to decoding randomness on deterministic tasks. We discuss the implications of these results for deploying sparse language models in reliability-critical applications, highlighting scenarios in which sparse architectures can be safely adopted without sacrificing output stability.
>
---
#### [new 106] RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于代码审查自动化任务，旨在解决无需微调的高质量代码评论生成问题。工作包括设计并部署RovoDev工具，验证其有效性与效率提升。**

- **链接: [https://arxiv.org/pdf/2601.01129v1](https://arxiv.org/pdf/2601.01129v1)**

> **作者:** Kla Tantithamthavorn; Yaotian Zou; Andy Wong; Michael Gupta; Zhe Wang; Mike Buller; Ryan Jiang; Matthew Watson; Minwoo Jeong; Kun Chen; Ming Wu
>
> **备注:** Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
>
> **摘要:** Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is (1) effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and (2) offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
>
---
#### [new 107] 600k-ks-ocr: a large-scale synthetic dataset for optical character recognition in kashmiri script
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出600K-KS-OCR数据集，用于解决克什米尔语OCR任务中的资源不足问题。通过合成数据和增强技术，提供高质量训练素材。**

- **链接: [https://arxiv.org/pdf/2601.01088v1](https://arxiv.org/pdf/2601.01088v1)**

> **作者:** Haq Nawaz Malik
>
> **摘要:** This technical report presents the 600K-KS-OCR Dataset, a large-scale synthetic corpus comprising approximately 602,000 word-level segmented images designed for training and evaluating optical character recognition systems targeting Kashmiri script. The dataset addresses a critical resource gap for Kashmiri, an endangered Dardic language utilizing a modified Perso-Arabic writing system spoken by approximately seven million people. Each image is rendered at 256x64 pixels with corresponding ground-truth transcriptions provided in multiple formats compatible with CRNN, TrOCR, and generalpurpose machine learning pipelines. The generation methodology incorporates three traditional Kashmiri typefaces, comprehensive data augmentation simulating real-world document degradation, and diverse background textures to enhance model robustness. The dataset is distributed across ten partitioned archives totaling approximately 10.6 GB and is released under the CC-BY-4.0 license to facilitate research in low-resource language optical character recognition.
>
---
#### [new 108] Attention Needs to Focus: A Unified Perspective on Attention Allocation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer中注意力分配不当导致的表示崩溃和注意力陷阱问题。提出Lazy Attention机制，通过位置区分和弹性softmax优化注意力分布。**

- **链接: [https://arxiv.org/pdf/2601.00919v1](https://arxiv.org/pdf/2601.00919v1)**

> **作者:** Zichuan Fu; Wentao Song; Guojing Li; Yejing Wang; Xian Wu; Yimin Deng; Hanyu Yan; Yefeng Zheng; Xiangyu Zhao
>
> **备注:** ICLR 2026 conference
>
> **摘要:** The Transformer architecture, a cornerstone of modern Large Language Models (LLMs), has achieved extraordinary success in sequence modeling, primarily due to its attention mechanism. However, despite its power, the standard attention mechanism is plagued by well-documented issues: representational collapse and attention sink. Although prior work has proposed approaches for these issues, they are often studied in isolation, obscuring their deeper connection. In this paper, we present a unified perspective, arguing that both can be traced to a common root -- improper attention allocation. We identify two failure modes: 1) Attention Overload, where tokens receive comparable high weights, blurring semantic features that lead to representational collapse; 2) Attention Underload, where no token is semantically relevant, yet attention is still forced to distribute, resulting in spurious focus such as attention sink. Building on this insight, we introduce Lazy Attention, a novel mechanism designed for a more focused attention distribution. To mitigate overload, it employs positional discrimination across both heads and dimensions to sharpen token distinctions. To counteract underload, it incorporates Elastic-Softmax, a modified normalization function that relaxes the standard softmax constraint to suppress attention on irrelevant tokens. Experiments on the FineWeb-Edu corpus, evaluated across nine diverse benchmarks, demonstrate that Lazy Attention successfully mitigates attention sink and achieves competitive performance compared to both standard attention and modern architectures, while reaching up to 59.58% attention sparsity.
>
---
#### [new 109] SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Lego，用于解决软件工程中的问题修复任务。通过轻量级监督微调方法，提升模型性能，并引入数据集和优化训练流程，取得优异结果。**

- **链接: [https://arxiv.org/pdf/2601.01426v1](https://arxiv.org/pdf/2601.01426v1)**

> **作者:** Chaofan Tao; Jierun Chen; Yuxin Jiang; Kaiqi Kou; Shaowei Wang; Ruoyu Wang; Xiaohui Li; Sidi Yang; Yiming Du; Jianbo Dai; Zhiming Mao; Xinyu Wang; Lifeng Shang; Haoli Bai
>
> **备注:** Project website: https://github.com/SWE-Lego/SWE-Lego
>
> **摘要:** We present SWE-Lego, a supervised fine-tuning (SFT) recipe designed to achieve state-ofthe-art performance in software engineering (SWE) issue resolving. In contrast to prevalent methods that rely on complex training paradigms (e.g., mid-training, SFT, reinforcement learning, and their combinations), we explore how to push the limits of a lightweight SFT-only approach for SWE tasks. SWE-Lego comprises three core building blocks, with key findings summarized as follows: 1) the SWE-Lego dataset, a collection of 32k highquality task instances and 18k validated trajectories, combining real and synthetic data to complement each other in both quality and quantity; 2) a refined SFT procedure with error masking and a difficulty-based curriculum, which demonstrably improves action quality and overall performance. Empirical results show that with these two building bricks alone,the SFT can push SWE-Lego models to state-of-the-art performance among open-source models of comparable size on SWE-bench Verified: SWE-Lego-Qwen3-8B reaches 42.2%, and SWE-Lego-Qwen3-32B attains 52.6%. 3) We further evaluate and improve test-time scaling (TTS) built upon the SFT foundation. Based on a well-trained verifier, SWE-Lego models can be significantly boosted--for example, 42.2% to 49.6% and 52.6% to 58.8% under TTS@16 for the 8B and 32B models, respectively.
>
---
## 更新

#### [replaced 001] Language as a Wave Phenomenon: Iso-Energetic Phase-Locking and Semantic Interference in Neural Networks
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决深度学习与光子硬件不兼容的问题。提出PRISM架构，通过相位编码实现低能耗语义推理，验证其在翻译任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01208v2](https://arxiv.org/pdf/2512.01208v2)**

> **作者:** Alper Yıldırım; İbrahim Yücedağ
>
> **备注:** Major Revision. Title changed to reflect the new theoretical framework. Complete narrative shift from "Optimization Efficiency" to "Iso-Energetic Phase Coding" and "Optical Hardware Compatibility". Replaced ISMR diagnostics with Holographic Optical Learning simulations and mechanistic "Dual-Regime" phase analysis. Comparison with spectral baselines (FNet) added
>
> **摘要:** Conventional deep learning paradigms rely on metabolically expensive magnitude-based representations, rendering them fundamentally incompatible with passive photonic hardware. We introduce PRISM, a sequence modeling architecture that bridges high-level reasoning and physical constraints by enforcing an Iso-Energetic (Unity Gain) principle, compelling the network to encode semantic information exclusively in the phase angle. Validated on the WMT14 translation benchmark, PRISM achieves a 0.799 COMET score, demonstrating that phase-based reasoning competes with standard Transformers (0.821) and functionally matches unconstrained spectral baselines like FNet (0.805), despite enforcing strict energy constraints and requiring 11.5% fewer parameters. Furthermore, to verify hardware feasibility, we simulate a Holographic Backpropagation mechanism on a noisy, 4-bit optical correlator. Ablation studies reveal a substantial performance gain (48.4% vs. 62.4%) over a frozen baseline, proving that the proposed phase-steering mechanism actively optimizes physical parameters under strict energy constraints. These results establish an existence proof that ultra-low-power, passive optical hardware can support high-level linguistic intelligence without sacrificing representational capacity.
>
---
#### [replaced 002] TabiBERT: A Large-Scale ModernBERT Foundation Model and A Unified Benchmark for Turkish
- **分类: cs.CL**

- **简介: 该论文提出TabiBERT，一个基于ModernBERT的土耳其语单语模型，解决土耳其语NLP缺乏先进架构模型的问题。通过大规模预训练和基准测试，提升多项任务性能。**

- **链接: [https://arxiv.org/pdf/2512.23065v3](https://arxiv.org/pdf/2512.23065v3)**

> **作者:** Melikşah Türker; A. Ebrar Kızıloğlu; Onur Güngör; Susan Üsküdarlı
>
> **备注:** 33 pages, 2 figures, 13 tables
>
> **摘要:** Since the inception of BERT, encoder-only Transformers have evolved significantly in computational efficiency, training stability, and long-context modeling. ModernBERT consolidates these advances by integrating Rotary Positional Embeddings (RoPE), FlashAttention, and refined normalization. Despite these developments, Turkish NLP lacks a monolingual encoder trained from scratch, incorporating such modern architectural paradigms. This work introduces TabiBERT, a monolingual Turkish encoder based on ModernBERT architecture trained from scratch on a large, curated corpus. TabiBERT is pre-trained on one trillion tokens sampled from an 84.88B token multi-domain corpus: web text (73%), scientific publications (20%), source code (6%), and mathematical content (0.3%). It supports 8,192-token context length (16x original BERT), achieves up to 2.65x inference speedup, and reduces GPU memory consumption, enabling larger batch sizes. We introduce TabiBench with 28 datasets across eight task categories with standardized splits and protocols, evaluated using GLUE-style macro-averaging. TabiBERT attains 77.58 on TabiBench, outperforming BERTurk by 1.62 points and establishing state-of-the-art on five of eight categories, with particularly strong gains on question answering (+9.55 points), code retrieval (+2.41 points), and academic understanding (+0.66 points). Compared with task-specific prior best results, including specialized models like TurkishBERTweet, TabiBERT achieves +1.47 average improvement, indicating robust cross-domain generalization. We release model weights, training configurations, and evaluation code for transparent, reproducible Turkish encoder research.
>
---
#### [replaced 003] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出MMRB2基准，用于评估处理文本与图像交织的通用奖励模型。解决多模态奖励模型评估不足的问题，通过四个任务和专家标注数据进行实验分析。**

- **链接: [https://arxiv.org/pdf/2512.16899v2](https://arxiv.org/pdf/2512.16899v2)**

> **作者:** Yushi Hu; Reyhane Askari-Hemmat; Melissa Hall; Emily Dinan; Luke Zettlemoyer; Marjan Ghazvininejad
>
> **备注:** Code and data available at https://github.com/facebookresearch/MMRB2
>
> **摘要:** Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning ("thinking-with-images"), providing 1,000 expert-annotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66-75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.
>
---
#### [replaced 004] RankMamba: Benchmarking Mamba's Document Ranking Performance in the Era of Transformers
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究Mamba模型在文档排序任务中的表现，对比其与Transformer的性能及训练效率，旨在探索Mamba在信息检索中的应用潜力。**

- **链接: [https://arxiv.org/pdf/2403.18276v3](https://arxiv.org/pdf/2403.18276v3)**

> **作者:** Zhichao Xu
>
> **摘要:** Transformer structure has achieved great success in multiple applied machine learning communities, such as natural language processing (NLP), computer vision (CV) and information retrieval (IR). Transformer architecture's core mechanism\, -- \,attention requires $O(n^2)$ time complexity in training and $O(n)$ time complexity in inference. Many works have been proposed to improve the attention mechanism's scalability, such as Flash Attention and Multi-query Attention. A different line of work aims to design new mechanisms to replace attention. Recently, a notable model structure Mamba, which is based on state space models, has achieved transformer-equivalent performance in multiple sequence modeling tasks. In this work, we examine Mamba's efficacy through the lens of a classical IR task\, -- \,document ranking. A reranker model takes a query and a document as input, and predicts a scalar relevance score. This task demands the language model's ability to comprehend lengthy contextual inputs and to capture the interaction between query and document tokens. We find that \textbf{(1) Mamba models achieve competitive performance compared to transformer-based models with the same training recipe; (2) but also have a lower training throughput in comparison to efficient transformer implementations such as flash attention.} We hope this study can serve as a starting point to explore \mamba models in other classical IR tasks. Our \href{https://github.com/zhichaoxu-shufe/RankMamba}{code implementation} is made public to facilitate reproducibility. Refer to~\cite{xu-etal-2025-state} for more comprehensive experiments and results, including passage ranking.
>
---
#### [replaced 005] Cosmos: Compressed and Smooth Latent Space for Text Diffusion Modeling
- **分类: cs.CL**

- **简介: 该论文提出Cosmos，用于文本生成的压缩平滑潜在空间方法。解决扩散模型在文本生成中因高维表示而受限的问题，通过自编码器学习语义一致的潜在空间，提升生成效率与质量。**

- **链接: [https://arxiv.org/pdf/2506.21170v3](https://arxiv.org/pdf/2506.21170v3)**

> **作者:** Viacheslav Meshchaninov; Egor Chimbulatov; Alexander Shabalin; Aleksandr Abramov; Dmitry Vetrov
>
> **摘要:** Autoregressive language models dominate modern text generation, yet their sequential nature introduces fundamental limitations: decoding is slow, and maintaining global coherence remains challenging. Diffusion models offer a promising alternative by enabling parallel generation and flexible control; however, their application to text generation is hindered by the high dimensionality of token-level representations. We introduce Cosmos, a novel approach to text generation that operates entirely in a compressed, smooth latent space tailored specifically for diffusion. This space is learned using an autoencoder trained simultaneously for token-level reconstruction and alignment with frozen activations from a pretrained language encoder, providing robust semantic grounding and enabling effective perturbation-based augmentations. Empirically, we demonstrate that text representations can be compressed by $8\times$ while maintaining generation quality comparable to token-level diffusion models. Furthermore, increasing the latent sequence length allows Cosmos to surpass both diffusion-based and autoregressive baselines. We evaluate Cosmos on four diverse generative tasks including story generation, question generation, summarization, and detoxification and compare it with various generative paradigms. Cosmos achieves comparable or superior generation quality while offering more than $2\times$ faster inference. Code is released at \href{https://github.com/MeshchaninovViacheslav/cosmos}{GitHub}
>
---
#### [replaced 006] When in Doubt, Consult: Expert Debate for Sexism Detection via Confidence-Based Routin
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于性别歧视内容检测任务，旨在解决传统方法难以识别隐晦、复杂 sexist 内容的问题。通过两阶段框架提升检测效果，实现更准确的分类。**

- **链接: [https://arxiv.org/pdf/2512.23732v2](https://arxiv.org/pdf/2512.23732v2)**

> **作者:** Anwar Alajmi; Gabriele Pergola
>
> **摘要:** Sexist content online increasingly appears in subtle, context-dependent forms that evade traditional detection methods. Its interpretation often depends on overlapping linguistic, psychological, legal, and cultural dimensions, which produce mixed and sometimes contradictory signals, even in annotated datasets. These inconsistencies, combined with label scarcity and class imbalance, result in unstable decision boundaries and cause fine-tuned models to overlook subtler, underrepresented forms of harm. Together, these limitations point to the need for a design that explicitly addresses the combined effects of (i) underrepresentation, (ii) noise, and (iii) conceptual ambiguity in both data and model predictions. To address these challenges, we propose a two-stage framework that unifies (i) targeted training procedures to adapt supervision to scarce and noisy data with (ii) selective, reasoning-based inference to handle ambiguous or borderline cases. Our training setup applies class-balanced focal loss, class-aware batching, and post-hoc threshold calibration to mitigate label imbalance and noisy supervision. At inference time, a dynamic routing mechanism classifies high-confidence cases directly and escalates uncertain instances to a novel \textit{Collaborative Expert Judgment} (CEJ) module, which prompts multiple personas and consolidates their reasoning through a judge model. Our approach achieves state-of-the-art results across several benchmarks, with F1 gains of +4.48% and +1.30% on EDOS Tasks A and B, respectively, and a +2.79% improvement in ICM on EXIST 2025 Task 1.1.
>
---
#### [replaced 007] mHC: Manifold-Constrained Hyper-Connections
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于深度学习任务，旨在解决Hyper-Connections（HC）训练不稳定和扩展性差的问题。提出mHC框架，通过流形约束恢复残差连接的恒等映射，提升训练稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2512.24880v2](https://arxiv.org/pdf/2512.24880v2)**

> **作者:** Zhenda Xie; Yixuan Wei; Huanqi Cao; Chenggang Zhao; Chengqi Deng; Jiashi Li; Damai Dai; Huazuo Gao; Jiang Chang; Kuai Yu; Liang Zhao; Shangyan Zhou; Zhean Xu; Zhengyan Zhang; Wangding Zeng; Shengding Hu; Yuqing Wang; Jingyang Yuan; Lean Wang; Wenfeng Liang
>
> **摘要:** Recently, studies exemplified by Hyper-Connections (HC) have extended the ubiquitous residual connection paradigm established over the past decade by expanding the residual stream width and diversifying connectivity patterns. While yielding substantial performance gains, this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead. To address these challenges, we propose Manifold-Constrained Hyper-Connections (mHC), a general framework that projects the residual connection space of HC onto a specific manifold to restore the identity mapping property, while incorporating rigorous infrastructure optimization to ensure efficiency. Empirical experiments demonstrate that mHC is effective for training at scale, offering tangible performance improvements and superior scalability. We anticipate that mHC, as a flexible and practical extension of HC, will contribute to a deeper understanding of topological architecture design and suggest promising directions for the evolution of foundational models.
>
---
#### [replaced 008] How to Correctly Report LLM-as-a-Judge Evaluations
- **分类: cs.LG; cs.CL; stat.AP; stat.ML**

- **简介: 该论文属于自然语言处理中的评估任务，解决LLM作为评判者时的偏差问题，提出修正框架和自适应校准策略，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2511.21140v2](https://arxiv.org/pdf/2511.21140v2)**

> **作者:** Chungpa Lee; Thomas Zeng; Jongwon Jeong; Jy-yong Sohn; Kangwook Lee
>
> **备注:** This version adds Sections 2, 6, 7, and 8.2
>
> **摘要:** Large language models (LLMs) are widely used as scalable evaluators of model responses in lieu of human annotators. However, imperfect sensitivity and specificity of LLM judgments induce bias in naive evaluation scores. We propose a simple plug-in framework that corrects this bias and constructs confidence intervals accounting for uncertainty from both the test dataset and a human-evaluated calibration dataset, enabling statistically sound and practical LLM-based evaluation. Building on this framework, we introduce an adaptive calibration strategy for constructing the calibration dataset to reduce uncertainty in the estimated score. Notably, we characterize the regimes in which LLM-based evaluation within our framework produces more reliable estimates than fully human evaluation. Moreover, our framework is more robust to distribution shift between the test and calibration datasets than existing approaches.
>
---
#### [replaced 009] UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出UniDoc-Bench，一个用于文档导向多模态RAG的基准，解决多模态信息融合不足的问题，通过构建真实PDF数据集并生成多模态问答对进行评估。**

- **链接: [https://arxiv.org/pdf/2510.03663v3](https://arxiv.org/pdf/2510.03663v3)**

> **作者:** Xiangyu Peng; Can Qin; Zeyuan Chen; Ran Xu; Caiming Xiong; Chien-Sheng Wu
>
> **摘要:** Multimodal retrieval-augmented Generation (MM-RAG) is a key approach for applying large language models (LLMs) and agents to real-world knowledge bases, yet current evaluations are fragmented -- focusing on either text or images in isolation, or simplified multimodal setup, failing to capture document-centric multimodal use cases. In this paper, we introduce UniDoc-Bench, the first large-scale, realistic benchmark for MM-RAG built from $k$ real-world PDF pages across domains. Our pipeline extracts and links evidence from text, tables, and figures, then generates multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. To ensure reliability, all of QA pairs are validated by multiple human annotators and expert adjudication. UniDoc-Bench supports apples-to-apples comparison across four paradigms: 1) text-only, 2) image-only, 3) \emph{multimodal} text-image fusion and 4) multimodal joint retrieval -- under a unified protocol with standardized candidate pools, prompts, and evaluation metrics. UniDoc-Bench can also be used to evaluate Visual Question Answering (VQA) tasks. Our experiments show that multimodal text-image fusion RAG systems consistently outperform both unimodal and jointly multimodal embedding-based retrieval, indicating that neither text nor images alone are sufficient and that current multimodal embeddings remain inadequate. Beyond benchmarking, our analysis reveals when and how visual context complements textual evidence, uncovers systematic failure modes, and offers actionable guidance for developing more robust MM-RAG pipelines.
>
---
#### [replaced 010] Sri Lanka Document Datasets: A Large-Scale, Multilingual Resource for Law, News, and Policy
- **分类: cs.CL**

- **简介: 该论文介绍了一个包含斯里兰卡法律、新闻和政策等多语种文档数据集，用于支持自然语言处理和相关研究。任务为构建多语言数据资源，解决数据稀缺问题。**

- **链接: [https://arxiv.org/pdf/2510.04124v4](https://arxiv.org/pdf/2510.04124v4)**

> **作者:** Nuwan I. Senaratna
>
> **备注:** 4 pages. 247,818 documents (67.6 GB) across 26 datasets in Sinhala, Tamil, and English. Last updated on 2026-01-03 (9.33am)
>
> **摘要:** We present a collection of open, machine-readable document datasets covering parliamentary proceedings, legal judgments, government publications, news, and tourism statistics from Sri Lanka. The collection currently comprises of 247,818 documents (67.6 GB) across 26 datasets in Sinhala, Tamil, and English. The datasets are updated daily and mirrored on GitHub and Hugging Face. These resources aim to support research in computational linguistics, legal analytics, socio-political studies, and multilingual natural language processing. We describe the data sources, collection pipeline, formats, and potential use cases, while discussing licensing and ethical considerations. This manuscript is at version v2026-01-03-0933.
>
---
#### [replaced 011] From Context to EDUs: Faithful and Structured Context Compression via Elementary Discourse Unit Decomposition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的文本压缩任务，旨在解决长文本处理中计算成本高和信息噪声问题。通过EDU分解与结构选择，实现高效且结构保真的上下文压缩。**

- **链接: [https://arxiv.org/pdf/2512.14244v4](https://arxiv.org/pdf/2512.14244v4)**

> **作者:** Yiqing Zhou; Yu Lei; Shuzheng Si; Qingyan Sun; Wei Wang; Yifei Wu; Hao Wen; Gang Chen; Fanchao Qi; Maosong Sun
>
> **摘要:** Managing extensive context remains a critical bottleneck for Large Language Models (LLMs), particularly in applications like long-document question answering and autonomous agents where lengthy inputs incur high computational costs and introduce noise. Existing compression techniques often disrupt local coherence through discrete token removal or rely on implicit latent encoding that suffers from positional bias and incompatibility with closed-source APIs. To address these limitations, we introduce the EDU-based Context Compressor, a novel explicit compression framework designed to preserve both global structure and fine-grained details. Our approach reformulates context compression as a structure-then-select process. First, our LingoEDU transforms linear text into a structural relation tree of Elementary Discourse Units (EDUs) which are anchored strictly to source indices to eliminate hallucination. Second, a lightweight ranking module selects query-relevant sub-trees for linearization. To rigorously evaluate structural understanding, we release StructBench, a manually annotated dataset of 248 diverse documents. Empirical results demonstrate that our method achieves state-of-the-art structural prediction accuracy and significantly outperforms frontier LLMs while reducing costs. Furthermore, our structure-aware compression substantially enhances performance across downstream tasks ranging from long-context tasks to complex Deep Search scenarios.
>
---
#### [replaced 012] Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges
- **分类: cs.CL**

- **简介: 该论文属于对话评估任务，旨在解决LLM评估中因单judge导致的偏差问题。通过聚合多个LLM judge的偏好知识，构建高效对话评估模型，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2508.00454v3](https://arxiv.org/pdf/2508.00454v3)**

> **作者:** Yuqi Tang; Kehua Feng; Yunfeng Wang; Zhiwen Chen; Chengfei Lv; Gang Yu; Keyan Ding; Huajun Chen
>
> **备注:** 20 pages, 4 pages, under review
>
> **摘要:** Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the "LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast, flexible, and fine-grained dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
>
---
#### [replaced 013] Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM推理输出多样性不足的问题。通过提出1PNS训练范式和RPD度量方法，提升模型推理路径的多样性。**

- **链接: [https://arxiv.org/pdf/2510.26122v2](https://arxiv.org/pdf/2510.26122v2)**

> **作者:** Feng Ju; Zeyu Qin; Rui Min; Zhitao He; Lingpeng Kong; Yi R. Fung
>
> **摘要:** While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. This homogenization not only limits sampling effectiveness but also restricts the exploration space for subsequent Reinforcement Learning (RL) stages. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at https://github.com/fengjujf/Reasoning-Path-Divergence .
>
---
#### [replaced 014] LABOR-LLM: Language-Based Occupational Representations with Large Language Models
- **分类: cs.LG; cs.CL; econ.EM**

- **简介: 该论文属于职业预测任务，旨在解决根据职业历史预测未来职业的问题。通过微调大语言模型，利用简历文本数据提升预测性能。**

- **链接: [https://arxiv.org/pdf/2406.17972v4](https://arxiv.org/pdf/2406.17972v4)**

> **作者:** Susan Athey; Herman Brunborg; Tianyu Du; Ayush Kanodia; Keyon Vafa
>
> **摘要:** This paper builds an empirical model that predicts a worker's next occupation as a function of the worker's occupational history. Because histories are sequences of occupations, the covariate space is high-dimensional, and further, the outcome (the next occupation) is a discrete choice that can take on many values. To estimate the parameters of the model, we leverage an approach from generative artificial intelligence. Estimation begins from a ``foundation model'' trained on non-representative data and then ``fine-tunes'' the estimation using data about careers from a representative survey. We convert tabular data from the survey into text files that resemble resumes and fine-tune the parameters of the foundation model, a large language model (LLM), using these text files with the objective of predicting the next token (word). The resulting fine-tuned LLM is used to calculate estimates of worker transition probabilities. Its predictive performance surpasses all prior models, both for the task of granularly predicting the next occupation as well as for specific tasks such as predicting whether the worker changes occupations or stays in the labor force. We quantify the value of fine-tuning and further show that by adding more career data from a different population, fine-tuning smaller LLMs (fewer parameters) surpasses the performance of fine-tuning larger models. When we omit the English language occupational title and replace it with a unique code, predictive performance declines.
>
---
#### [replaced 015] Korean Canonical Legal Benchmark: Toward Knowledge-Independent Evaluation of LLMs' Legal Reasoning Capabilities
- **分类: cs.CL**

- **简介: 该论文提出KCL基准，用于独立评估大语言模型的法律推理能力。任务是法律推理评估，解决模型依赖领域知识的问题，通过提供支持性判例和评分标准进行评测。**

- **链接: [https://arxiv.org/pdf/2512.24572v2](https://arxiv.org/pdf/2512.24572v2)**

> **作者:** Hongseok Oh; Wonseok Hwang; Kyoung-Woon On
>
> **备注:** EACL 2026
>
> **摘要:** We introduce the Korean Canonical Legal Benchmark (KCL), a benchmark designed to assess language models' legal reasoning capabilities independently of domain-specific knowledge. KCL provides question-level supporting precedents, enabling a more faithful disentanglement of reasoning ability from parameterized knowledge. KCL consists of two components: (1) KCL-MCQA, multiple-choice problems of 283 questions with 1,103 aligned precedents, and (2) KCL-Essay, open-ended generation problems of 169 questions with 550 aligned precedents and 2,739 instance-level rubrics for automated evaluation. Our systematic evaluation of 30+ models shows large remaining gaps, particularly in KCL-Essay, and that reasoning-specialized models consistently outperform their general-purpose counterparts. We release all resources, including the benchmark dataset and evaluation code, at https://github.com/lbox-kr/kcl.
>
---
#### [replaced 016] On the Robustness of Answer Formats in Medical Reasoning Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗推理任务，研究MRMs在不同答案格式下的鲁棒性问题。通过实验发现模型在不同输出格式间表现差异大，提出评估指标并分析训练方法的影响。**

- **链接: [https://arxiv.org/pdf/2509.20866v2](https://arxiv.org/pdf/2509.20866v2)**

> **作者:** Pittawat Taveekitworachai; Natpatchara Pongjirapat; Krittaphas Chaisutyakorn; Piyalitt Ittichaiwong; Tossaporn Saengja; Kunat Pipatanakul
>
> **备注:** 62 pages, 47 figures
>
> **摘要:** Medical reasoning models (MRMs) achieve superior performance on medical benchmarks compared to medical LLMs; however, high accuracy alone is insufficient for practical deployment. One of such requirements for real-world application is robustness to varying output constraints. Specifically, posing the same medical question while requesting different answer formats should not affect the underlying correctness of the response. We investigate this phenomenon in this paper, focusing on MRMs. To quantify this behavior, we propose the metric answer-format robustness: the ability to reliably generate correct outputs across varying specified formats. We examine three representative formats: multiple-choice, open-ended question-answering, and ranked lists. Across 15 proprietary and open-weight models, we observe substantial variation in format robustness (35-100%). Furthermore, we conduct controlled fine-tuning experiments on a shared backbone with matched training data to isolate the effects of the fine-tuning paradigm. We find that supervised fine-tuning yields more stable behavior across formats, whereas reinforcement fine-tuning often exhibits higher cross-format brittleness, with the degree of instability strongly dependent on reward design. Overall, answer-format robustness in MRMs is trainable yet brittle and requires careful evaluation for practical medical use.
>
---
#### [replaced 017] ToMoE: Converting Dense Large Language Models to Mixture-of-Experts through Dynamic Structural Pruning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大模型部署成本高的问题。通过动态结构剪枝将密集模型转换为专家混合架构，减少活跃参数数量而不永久删除参数，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2501.15316v2](https://arxiv.org/pdf/2501.15316v2)**

> **作者:** Shangqian Gao; Ting Hua; Reza Shirkavand; Chi-Heng Lin; Zheng Tang; Zhengao Li; Longge Yuan; Fangyi Li; Zeyu Zhang; Alireza Ganjdanesh; Lou Qian; Xu Jie; Yen-Chang Hsu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable abilities in tackling a wide range of complex tasks. However, their huge computational and memory costs raise significant challenges in deploying these models on resource-constrained devices or efficiently serving them. Prior approaches have attempted to alleviate these problems by permanently removing less important model structures, yet these methods often result in substantial performance degradation due to the permanent deletion of model parameters. In this work, we tried to mitigate this issue by reducing the number of active parameters without permanently removing them. Specifically, we introduce a differentiable dynamic pruning method that pushes dense models to maintain a fixed number of active parameters by converting their MLP layers into a Mixture of Experts (MoE) architecture. Our method, even without fine-tuning, consistently outperforms previous structural pruning techniques across diverse model families, including Phi-2, LLaMA-2, LLaMA-3, and Qwen-2.5.
>
---
#### [replaced 018] ScRPO: From Errors to Insights
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ScRPO框架，解决大模型数学推理能力不足问题，通过自反思与错误修正提升性能。**

- **链接: [https://arxiv.org/pdf/2511.06065v3](https://arxiv.org/pdf/2511.06065v3)**

> **作者:** Lianrui Li; Dakuan Lu; Jiawei Shao; Xuelong Li
>
> **摘要:** We introduce Self-correction Relative Policy Optimization (ScRPO), a novel reinforcement learning framework designed to empower large language models with advanced mathematical reasoning capabilities through iterative self-reflection and error correction. The ScRPO framework operates in two distinct phases: (1) Trial-and-error learning stage, where the model is trained via GRPO, and incorrect responses are collected to form an "error pool"; and (2) Self-correction learning stage, which guides the model to introspectively analyze and rectify the reasoning flaws behind its previous errors. Extensive evaluations across challenging mathematical benchmarks, including AIME, AMC, Olympiad, MATH-500, and GSM8k, validate the efficacy of our approach. Using DeepSeek-R1-Distill-Qwen-1.5B and 7B as backbones, ScRPO achieves average accuracies of 64.8% and 77.8%, respectively. This represents a significant improvement of 6.0% and 3.2% over vanilla baselines, consistently outperforming strong post-training methods such as DAPO and GRPO. These findings establish ScRPO as a robust paradigm for enabling autonomous self-improvement in AI systems, particularly in tasks with limited external feedback.
>
---
#### [replaced 019] Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究多轮对话中语音模型的风格退化问题，属于自然语言处理任务。旨在解决模型无法持续保持指定说话风格的问题，并探索缓解方法。**

- **链接: [https://arxiv.org/pdf/2512.23578v2](https://arxiv.org/pdf/2512.23578v2)**

> **作者:** Yu-Xiang Lin; Cheng-Han Chiang; Hung-yi Lee
>
> **备注:** Submitted to ACL ARR January 2026
>
> **摘要:** In this paper, we show that when spoken language models (SLMs) are instructed to speak in a specific speaking style at the beginning of a multi-turn conversation, they cannot maintain the required speaking styles after several turns of interaction; we refer to this as the style amnesia of SLMs. We focus on paralinguistic speaking styles, including emotion, accent, volume, and speaking speed. We evaluate three proprietary and two open-source SLMs, demonstrating that none of these models can maintain a consistent speaking style when instructed to do so. We further show that when SLMs are asked to recall the style instruction in later turns, they can recall the style instruction, but they fail to express it throughout the conversation. We also show that explicitly asking the model to recall the style instruction can partially mitigate style amnesia. In addition, we examine various prompting strategies and find that SLMs struggle to follow the required style when the instruction is placed in system messages rather than user messages, which contradicts the intended function of system prompts.
>
---
#### [replaced 020] GIFT: Group-relative Implicit Fine Tuning Integrates GRPO with DPO and UNA
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出GIFT框架，用于大语言模型对齐任务，解决如何有效利用隐式奖励的问题。通过结合GRPO、DPO和UNA方法，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2510.23868v2](https://arxiv.org/pdf/2510.23868v2)**

> **作者:** Zhichao Wang
>
> **摘要:** I propose \textbf{G}roup-relative \textbf{I}mplicit \textbf{F}ine \textbf{T}uning (GIFT), a novel reinforcement learning framework for aligning LLMs. Instead of directly maximizing cumulative rewards like PPO or GRPO, GIFT minimizes the discrepancy between implicit and explicit reward models. It combines three key ideas: (1) the online multi-response generation and normalization of GRPO, (2) the implicit reward formulation of DPO, and (3) the implicit-explicit reward alignment principle of UNA. By jointly normalizing the implicit and explicit rewards, GIFT eliminates an otherwise intractable term that prevents effective use of implicit rewards. This normalization transforms the complex reward maximization objective into a simple mean squared error (MSE) loss between the normalized reward functions, converting a non-convex optimization problem into a convex, stable, and analytically differentiable formulation. Unlike offline methods such as DPO and UNA, GIFT remains on-policy and thus retains exploration capability. Compared to GRPO, it requires fewer hyperparameters, converges faster, and generalizes better with significantly reduced training overfitting. Empirically, GIFT achieves superior reasoning and alignment performance on mathematical benchmarks while remaining computationally efficient.
>
---
#### [replaced 021] CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决轻量级LLM在中文场景下的安全漏洞问题。针对中文特有的攻击模式，构建了CSSBench基准，评估模型安全性。**

- **链接: [https://arxiv.org/pdf/2601.00588v2](https://arxiv.org/pdf/2601.00588v2)**

> **作者:** Zhenhong Zhou; Shilinlu Yan; Chuanpu Liu; Qiankun Li; Kun Wang; Zhigang Zeng
>
> **备注:** 18 pages
>
> **摘要:** Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice.
>
---
#### [replaced 022] Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM在不同语言和时间框架下的安全性问题。通过测试模型在多种场景下的表现，揭示其安全性的上下文依赖性。**

- **链接: [https://arxiv.org/pdf/2512.24556v2](https://arxiv.org/pdf/2512.24556v2)**

> **作者:** Muhammad Abdullahi Said; Muhammad Sammani Sani
>
> **摘要:** As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state of the art models (GPT-5.1, Gemini 3 Pro, and Claude 4.5 Opus) using HausaSafety, a novel adversarial dataset grounded in West African threat scenarios (e.g., Yahoo-Yahoo fraud, Dane gun manufacturing). Employing a 2 x 4 factorial design across 1,440 evaluations, we tested the non-linear interaction between language (English vs. Hausa) and temporal framing. Our results challenge the narrative of the multilingual safety gap. Instead of a simple degradation in low-resource settings, we identified a complex interference mechanism in which safety is determined by the intersection of variables. Although the models exhibited a reverse linguistic vulnerability with Claude 4.5 Opus proving significantly safer in Hausa (45.0%) than in English (36.7%) due to uncertainty-driven refusal, they suffered catastrophic failures in temporal reasoning. We report a profound Temporal Asymmetry, where past-tense framing bypassed defenses (15.6% safe) while future-tense scenarios triggered hyper-conservative refusals (57.2% safe). The magnitude of this volatility is illustrated by a 9.2x disparity between the safest and most vulnerable configurations, proving that safety is not a fixed property but a context-dependent state. We conclude that current models rely on superficial heuristics rather than robust semantic understanding, creating Safety Pockets that leave Global South users exposed to localized harms. We propose Invariant Alignment as a necessary paradigm shift to ensure safety stability across linguistic and temporal shifts.
>
---
#### [replaced 023] The Syntax of qulk-clauses in Yemeni Ibbi Arabic: A Minimalist Approach
- **分类: cs.CL**

- **简介: 该论文研究Yemeni Ibbi Arabic中qulk-clauses的句法，属于生成句法领域。旨在解析其结构及运作机制，通过最小主义框架分析其语法过程。**

- **链接: [https://arxiv.org/pdf/2512.22376v2](https://arxiv.org/pdf/2512.22376v2)**

> **作者:** Zubaida Mohammed Albadani; Mohammed Q. Shormani
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** This study investigates the syntax of qulk-clauses in Yemeni Ibbi Arabic (YIA) within the Minimalist Program. The construction qulk-clause, a morphologically fused form meaning 'I said,' introduces embedded declarative interrogative, and imperative clauses, often eithout complementizer. The central proposal of this paper is that qulk-clauses are biclausal structures in which qulk functions a clause-embedding predicate sec;ecting a dull CP complement. By applying core minimalist operations, viz., Merge, Move, Agree, and Spell-out, the study provides a layered syntactic analysis of qulk-clauses, for illustrating how their derivation proceeds through standard computational steps and post-syntactic processes such as Morphological Merger. The proposal also accounts for dialect-specific features like bipartite negation, cliticization, and CP embedding. The findings offer theoretical contributions to generative syntax, specifically minimalism. The study concludes raising theoretical questions concerning extending the analysis to the addressee-clause kil-k 'you said'. It also provides insights into the possibility of the universality of minimalism.
>
---
#### [replaced 024] Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ALE生态系统，解决开放环境中智能体开发问题。构建ROME模型，采用IPA算法提升长期训练稳定性。**

- **链接: [https://arxiv.org/pdf/2512.24873v2](https://arxiv.org/pdf/2512.24873v2)**

> **作者:** Weixun Wang; XiaoXiao Xu; Wanhe An; Fangwen Dai; Wei Gao; Yancheng He; Ju Huang; Qiang Ji; Hanqi Jin; Xiaoyang Li; Yang Li; Zhongwen Li; Shirong Lin; Jiashun Liu; Zenan Liu; Tao Luo; Dilxat Muhtar; Yuanbin Qu; Jiaqiang Shi; Qinghui Sun; Yingshui Tan; Hao Tang; Runze Wang; Yi Wang; Zhaoguo Wang; Yanan Wu; Shaopan Xiong; Binchen Xu; Xander Xu; Yuchi Xu; Qipeng Zhang; Xixia Zhang; Haizhou Zhao; Jie Zhao; Shuaibing Zhao; Baihui Zheng; Jianhui Zheng; Suhang Zheng; Yanni Zhu; Mengze Cai; Kerui Cao; Xitong Chen; Yue Dai; Lifan Du; Tao Feng; Tao He; Jin Hu; Yijie Hu; Ziyu Jiang; Cheng Li; Xiang Li; Jing Liang; Xin Lin; Chonghuan Liu; ZhenDong Liu; Zhiqiang Lv; Haodong Mi; Yanhu Mo; Junjia Ni; Shixin Pei; Jingyu Shen; XiaoShuai Song; Cecilia Wang; Chaofan Wang; Kangyu Wang; Pei Wang; Tao Wang; Wei Wang; Ke Xiao; Mingyu Xu; Tiange Xu; Nan Ya; Siran Yang; Jianan Ye; Yaxing Zang; Duo Zhang; Junbo Zhang; Boren Zheng; Wanxi Deng; Ling Pan; Lin Qu; Wenbo Su; Jiamang Wang; Wei Wang; Hu Wei; Minggang Wu; Cheng Yu; Bing Zhao; Zhicheng Zheng; Bo Zheng
>
> **备注:** 36 pages, 15 figures
>
> **摘要:** Agentic crafting requires LLMs to operate in real-world environments over multiple turns by taking actions, observing outcomes, and iteratively refining artifacts. Despite its importance, the open-source community lacks a principled, end-to-end ecosystem to streamline agent development. We introduce the Agentic Learning Ecosystem (ALE), a foundational infrastructure that optimizes the production pipeline for agentic model. ALE consists of three components: ROLL, a post-training framework for weight optimization; ROCK, a sandbox environment manager for trajectory generation; and iFlow CLI, an agent framework for efficient context engineering. We release ROME, an open-source agent grounded by ALE and trained on over one million trajectories. Our approach includes data composition protocols for synthesizing complex behaviors and a novel policy optimization algorithm, Interaction-Perceptive Agentic Policy Optimization (IPA), which assigns credit over semantic interaction chunks rather than individual tokens to improve long-horizon training stability. Empirically, we evaluate ROME within a structured setting and introduce Terminal Bench Pro, a benchmark with improved scale and contamination control. ROME demonstrates strong performance across benchmarks like SWE-bench Verified and Terminal Bench, proving the effectiveness of ALE.
>
---
#### [replaced 025] SteganoBackdoor: Stealthy and Data-Efficient Backdoor Attacks on Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文属于安全任务，解决语言模型的后门攻击问题。提出SteganoBackdoor框架，构建隐蔽的 poisoned 数据，实现高效且难以检测的后门攻击。**

- **链接: [https://arxiv.org/pdf/2511.14301v3](https://arxiv.org/pdf/2511.14301v3)**

> **作者:** Eric Xue; Ruiyi Zhang; Pengtao Xie
>
> **摘要:** Modern language models remain vulnerable to backdoor attacks via poisoned data, where training inputs containing a trigger are paired with a target output, causing the model to reproduce that behavior whenever the trigger appears at inference time. Recent work has emphasized stealthy attacks that stress-test data-curation defenses using stylized artifacts or token-level perturbations as triggers, but this focus leaves a more practically relevant threat model underexplored: backdoors tied to naturally occurring semantic concepts. We introduce SteganoBackdoor, an optimization-based framework that constructs SteganoPoisons, steganographic poisoned training examples in which a backdoor payload is distributed across a fluent sentence while exhibiting no representational overlap with the inference-time semantic trigger. Across diverse model architectures, SteganoBackdoor achieves high attack success under constrained poisoning budgets and remains effective under conservative data-level filtering, highlighting a blind spot in existing data-curation defenses.
>
---
#### [replaced 026] AprielGuard
- **分类: cs.CL**

- **简介: 该论文提出AprielGuard，属于大语言模型安全防护任务，旨在统一解决有害内容和对抗攻击问题，通过联合训练提升检测效果。**

- **链接: [https://arxiv.org/pdf/2512.20293v2](https://arxiv.org/pdf/2512.20293v2)**

> **作者:** Jaykumar Kasundra; Anjaneya Praharaj; Sourabh Surana; Lakshmi Sirisha Chodisetty; Sourav Sharma; Abhigya Verma; Abhishek Bhardwaj; Debasish Kanhar; Aakash Bhagat; Khalil Slimi; Seganrasan Subramanian; Sathwik Tejaswi Madhusudhan; Ranga Prasad Chenna; Srinivas Sunkara
>
> **摘要:** Safeguarding large language models (LLMs) against unsafe or adversarial behavior is critical as they are increasingly deployed in conversational and agentic settings. Existing moderation tools often treat safety risks (e.g. toxicity, bias) and adversarial threats (e.g. prompt injections, jailbreaks) as separate problems, limiting their robustness and generalizability. We introduce AprielGuard, an 8B parameter safeguard model that unify these dimensions within a single taxonomy and learning framework. AprielGuard is trained on a diverse mix of open and synthetic data covering standalone prompts, multi-turn conversations, and agentic workflows, augmented with structured reasoning traces to improve interpretability. Across multiple public and proprietary benchmarks, AprielGuard achieves strong performance in detecting harmful content and adversarial manipulations, outperforming existing opensource guardrails such as Llama-Guard and Granite Guardian, particularly in multi-step and reasoning intensive scenarios. By releasing the model, we aim to advance transparent and reproducible research on reliable safeguards for LLMs.
>
---
#### [replaced 027] VISTA Score: Verification In Sequential Turn-based Assessment
- **分类: cs.CL**

- **简介: 该论文提出VISTA框架，用于评估对话系统的事实性，解决多轮对话中的幻觉问题。通过逐句验证和跟踪一致性，提升事实检测效果。**

- **链接: [https://arxiv.org/pdf/2510.27052v3](https://arxiv.org/pdf/2510.27052v3)**

> **作者:** Ashley Lewis; Andrew Perrault; Eric Fosler-Lussier; Michael White
>
> **摘要:** Hallucination--defined here as generating statements unsupported or contradicted by available evidence or conversational context--remains a major obstacle to deploying conversational AI systems in settings that demand factual reliability. Existing metrics either evaluate isolated responses or treat unverifiable content as errors, limiting their use for multi-turn dialogue. We introduce VISTA (Verification In Sequential Turn-based Assessment), a framework for evaluating conversational factuality through claim-level verification and sequential consistency tracking. VISTA decomposes each assistant turn into atomic factual claims, verifies them against trusted sources and dialogue history, and categorizes unverifiable statements (subjective, contradicted, lacking evidence, or abstaining). Across eight large language models and four dialogue factuality benchmarks (AIS, BEGIN, FAITHDIAL, and FADE), VISTA substantially improves hallucination detection over FACTSCORE and LLM-as-Judge baselines. Human evaluation confirms that VISTA's decomposition improves annotator agreement and reveals inconsistencies in existing benchmarks. By modeling factuality as a dynamic property of conversation, VISTA offers a more transparent, human-aligned measure of truthfulness in dialogue systems.
>
---
#### [replaced 028] SGM: Safety Glasses for Multimodal Large Language Models via Neuron-Level Detoxification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态大语言模型的安全任务，旨在解决模型继承的毒性问题。提出SGM方法，在神经元层面进行干预，有效降低毒性输出，提升生成安全性。**

- **链接: [https://arxiv.org/pdf/2512.15052v2](https://arxiv.org/pdf/2512.15052v2)**

> **作者:** Hongbo Wang; MaungMaung AprilPyone; Isao Echizen
>
> **备注:** Under Review for ACL 2026
>
> **摘要:** Disclaimer: Samples in this paper may be harmful and cause discomfort. Multimodal large language models (MLLMs) enable multimodal generation but inherit toxic, biased, and NSFW signals from weakly curated pretraining corpora, causing safety risks, especially under adversarial triggers that late, opaque training-free detoxification methods struggle to handle. We propose SGM, a white-box neuron-level multimodal intervention that acts like safety glasses for toxic neurons: it selectively recalibrates a small set of toxic expert neurons via expertise-weighted soft suppression, neutralizing harmful cross-modal activations without any parameter updates. We establish MM-TOXIC-QA, a multimodal toxicity evaluation framework, and compare SGM with existing detoxification techniques. Experiments on open-source MLLMs show that SGM mitigates toxicity in standard and adversarial conditions, cutting harmful rates from 48.2\% to 2.5\% while preserving fluency and multimodal reasoning. SGM is extensible, and its combined defenses, denoted as SGM*, integrate with existing detoxification methods for stronger safety performance, providing an interpretable, low-cost solution for toxicity-controlled multimodal generation.
>
---
#### [replaced 029] Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话安全任务，旨在解决AI聊天机器人中隐性情绪升级的检测问题。提出GAUGE框架，实时监测对话情感变化。**

- **链接: [https://arxiv.org/pdf/2512.06193v3](https://arxiv.org/pdf/2512.06193v3)**

> **作者:** Jihyung Park; Saleh Afroogh; David Atkinson; Junfeng Jiao
>
> **摘要:** Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue.
>
---
#### [replaced 030] RIMRULE: Improving Tool-Using Language Agents via MDL-Guided Rule Learning
- **分类: cs.CL**

- **简介: 该论文属于语言模型工具使用任务，旨在解决LLM在特定工具使用中表现不佳的问题。通过动态规则注入和MDL优化，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.00086v2](https://arxiv.org/pdf/2601.00086v2)**

> **作者:** Xiang Gao; Yuguang Yao; Qi Zhang; Kaiwen Dong; Avinash Baidya; Ruocheng Guo; Hilaf Hasson; Kamalika Das
>
> **摘要:** Large language models (LLMs) often struggle to use tools reliably in domain-specific settings, where APIs may be idiosyncratic, under-documented, or tailored to private workflows. This highlights the need for effective adaptation to task-specific tools. We propose RIMRULE, a neuro-symbolic approach for LLM adaptation based on dynamic rule injection. Compact, interpretable rules are distilled from failure traces and injected into the prompt during inference to improve task performance. These rules are proposed by the LLM itself and consolidated using a Minimum Description Length (MDL) objective that favors generality and conciseness. Each rule is stored in both natural language and a structured symbolic form, supporting efficient retrieval at inference time. Experiments on tool-use benchmarks show that this approach improves accuracy on both seen and unseen tools without modifying LLM weights. It outperforms prompting-based adaptation methods and complements finetuning. Moreover, rules learned from one LLM can be reused to improve others, including long reasoning LLMs, highlighting the portability of symbolic knowledge across architectures.
>
---
#### [replaced 031] Reasoning Beyond Limits: Advances and Open Problems for LLMs
- **分类: cs.LG; cs.CL**

- **简介: 本文综述2023-2025年发布的27个大语言模型，分析其推理能力提升技术与挑战，探讨多语言、长文本处理及高效训练方法，旨在推动LLMs的自主推理与任务执行能力。**

- **链接: [https://arxiv.org/pdf/2503.22732v2](https://arxiv.org/pdf/2503.22732v2)**

> **作者:** Mohamed Amine Ferrag; Norbert Tihanyi; Merouane Debbah
>
> **备注:** The paper is published ICT Express Volume 11, Issue 6, December 2025, Pages 1054-1096
>
> **摘要:** Recent breakthroughs in generative reasoning have fundamentally reshaped how large language models (LLMs) address complex tasks, enabling them to dynamically retrieve, refine, and organize information into coherent multi-step reasoning chains. Techniques such as inference-time scaling, reinforcement learning, supervised fine-tuning, and distillation have been effectively applied to state-of-the-art models, including DeepSeek-R1, OpenAI o1 and o3, GPT-4o, Qwen-32B, and various Llama variants, significantly enhancing their reasoning capabilities. In this paper, we present a comprehensive review of the top 27 LLMs released between 2023 and 2025, such as Mistral AI Small 3 24B, DeepSeek-R1, Search-o1, QwQ-32B, and Phi-4, and analyze their core innovations and performance improvements. We also provide a detailed overview of recent advancements in multilingual large language models (MLLMs), emphasizing methods that improve cross-lingual reasoning and address the limitations of English-centric training. In parallel, we present a comprehensive review of progress in state space model (SSM)-based architectures, including models such as Mamba, which demonstrate improved efficiency for long-context processing compared to transformer-based approaches. Our analysis covers training strategies including general optimization techniques, mixture-of-experts (MoE) configurations, retrieval-augmented generation (RAG), chain-of-thought prompting, self-improvement methods, and test-time compute scaling and distillation frameworks. Finally, we identify key challenges for future research, including enabling multi-step reasoning without human supervision, improving robustness in chained task execution, balancing structured prompting with generative flexibility, and enhancing the integration of long-context retrieval and external tools.
>
---
#### [replaced 032] Adversarial Training for Failure-Sensitive User Simulation in Mental Health Dialogue Optimization
- **分类: cs.CL**

- **简介: 该论文属于任务导向对话系统领域，旨在解决用户模拟器不真实的问题。通过对抗训练提升模拟器的 realism，有效暴露系统缺陷，增强评估效果。**

- **链接: [https://arxiv.org/pdf/2512.20773v2](https://arxiv.org/pdf/2512.20773v2)**

> **作者:** Ziyi Zhu; Olivier Tieleman; Caitlin A. Stamatis; Luka Smyth; Thomas D. Hull; Daniel R. Cahn; Matteo Malgaroli
>
> **摘要:** Realistic user simulation is crucial for training and evaluating task-oriented dialogue (TOD) systems, yet creating simulators that accurately replicate human behavior remains challenging. A key property of effective simulators is their ability to expose failure modes of the systems they evaluate. We present an adversarial training framework that iteratively improves user simulator realism through a competitive dynamic between a generator (user simulator) and a discriminator. Applied to mental health support chatbots, our approach demonstrates that fine-tuned simulators dramatically outperform zero-shot base models at surfacing system issues, and adversarial training further enhances diversity, distributional alignment, and predictive validity. The resulting simulator achieves a strong correlation between simulated and real failure occurrence rates across diverse chatbot configurations while maintaining low distributional divergence of failure modes. Discriminator accuracy decreases drastically after three adversarial iterations, suggesting improved realism. These results provide evidence that adversarial training is a promising approach for creating realistic user simulators in mental health support TOD domains, enabling rapid, reliable, and cost-effective system evaluation before deployment.
>
---
#### [replaced 033] Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创意写作任务，旨在解决LLMs生成高质量剧本的难题。通过分解为叙事生成与格式转换两阶段，提升剧本质量。**

- **链接: [https://arxiv.org/pdf/2510.23163v2](https://arxiv.org/pdf/2510.23163v2)**

> **作者:** Hang Lei; Shengyi Zong; Zhaoyan Li; Ziren Zhou; Hao Liu
>
> **摘要:** The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
>
---
#### [replaced 034] A Systematic Survey on Large Language Models for Algorithm Design
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于算法设计任务，旨在系统梳理LLM在算法设计中的应用。解决缺乏全面综述的问题，通过分类分析和文献综述，总结现状并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2410.14716v5](https://arxiv.org/pdf/2410.14716v5)**

> **作者:** Fei Liu; Yiming Yao; Ping Guo; Zhiyuan Yang; Zhe Zhao; Xi Lin; Xialiang Tong; Kun Mao; Zhichao Lu; Zhenkun Wang; Mingxuan Yuan; Qingfu Zhang
>
> **摘要:** Algorithm design is crucial for effective problem-solving across various domains. The advent of Large Language Models (LLMs) has notably enhanced the automation and innovation within this field, offering new perspectives and promising solutions. In just a few years, this integration has yielded remarkable progress in areas ranging from combinatorial optimization to scientific discovery. Despite this rapid expansion, a holistic understanding of the field is hindered by the lack of a systematic review, as existing surveys either remain limited to narrow sub-fields or with different objectives. This paper seeks to provide a systematic review of algorithm design with LLMs. We introduce a taxonomy that categorises the roles of LLMs as optimizers, predictors, extractors and designers, analyzing the progress, advantages, and limitations within each category. We further synthesize literature across the three phases of the algorithm design pipeline and across diverse algorithmic applications that define the current landscape. Finally, we outline key open challenges and opportunities to guide future research. To support future research and collaboration, we provide an accompanying repository at: https://github.com/FeiLiu36/LLM4AlgorithmDesign.
>
---
#### [replaced 035] Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Youtu-LLM，解决轻量级大模型的代理能力问题。通过自研架构和训练策略，提升模型推理与规划能力，实验证明其在代理任务中表现优异。**

- **链接: [https://arxiv.org/pdf/2512.24618v2](https://arxiv.org/pdf/2512.24618v2)**

> **作者:** Junru Lu; Jiarui Qin; Lingfeng Qiao; Yinghui Li; Xinyi Dai; Bo Ke; Jianfeng He; Ruizhi Qiao; Di Yin; Xing Sun; Yunsheng Wu; Yinsong Liu; Shuangyin Liu; Mingkong Tang; Haodong Lin; Jiayi Kuang; Fanxu Meng; Xiaojuan Tang; Yunjia Xi; Junjie Huang; Haotong Yang; Zhenyi Shen; Yangning Li; Qianwen Zhang; Yifei Yu; Siyu An; Junnan Dong; Qiufeng Wang; Jie Wang; Keyu Chen; Wei Wen; Taian Guo; Zhifeng Shen; Daohai Yu; Jiahao Li; Ke Li; Zongyi Li; Xiaoyu Tan
>
> **备注:** 57 pages, 26 figures
>
> **摘要:** We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities.
>
---
#### [replaced 036] Sorting the Babble in Babel: Assessing the Performance of Language Identification Algorithms on the OpenAlex Database
- **分类: cs.CL**

- **简介: 该论文属于语言识别任务，旨在优化OpenAlex数据库的语种索引。通过比较不同算法在不同语料上的表现，解决语种识别准确性与效率问题。**

- **链接: [https://arxiv.org/pdf/2502.03627v4](https://arxiv.org/pdf/2502.03627v4)**

> **作者:** Maxime Holmberg Sainte-Marie; Diego Kozlowski; Lucía Céspedes; Vincent Larivière
>
> **备注:** 43 pages, 4 figures
>
> **摘要:** This project aims to optimize the linguistic indexing of the OpenAlex database by comparing the performance of various Python-based language identification procedures on different metadata corpora extracted from a manually-annotated article sample \footnote{OpenAlex used the results presented in this article to inform the language metadata overhaul carried out as part of its recent Walden system launch. The precision and recall performance of each algorithm, corpus, and language is first analyzed, followed by an assessment of processing speeds recorded for each algorithm and corpus type. These different performance measures are then simulated at the database level using probabilistic confusion matrices for each algorithm, corpus, and language, as well as a probabilistic modeling of relative article language frequencies for the whole OpenAlex database. Results show that procedure performance strongly depends on the importance given to each of the measures implemented: for contexts where precision is preferred, using the LangID algorithm on the greedy corpus gives the best results; however, for all cases where recall is considered at least slightly more important than precision or as soon as processing times are given any kind of consideration, the procedure that consists in the application of the FastText algorithm on the Titles corpus outperforms all other alternatives. Given the lack of truly multilingual large-scale bibliographic databases, it is hoped that these results help confirm and foster the unparalleled potential of the OpenAlex database for cross-linguistic and comprehensive measurement and evaluation.
>
---
#### [replaced 037] CoSER: A Comprehensive Literary Dataset and Framework for Training and Evaluating LLM Role-Playing and Persona Simulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoSER，一个用于训练和评估角色扮演语言模型的综合数据集和框架，解决角色模拟任务中的数据与评估不足问题。**

- **链接: [https://arxiv.org/pdf/2502.09082v3](https://arxiv.org/pdf/2502.09082v3)**

> **作者:** Xintao Wang; Heng Wang; Yifei Zhang; Xinfeng Yuan; Rui Xu; Jen-tse Huang; Siyu Yuan; Haoran Guo; Jiangjie Chen; Shuchang Zhou; Wei Wang; Yanghua Xiao
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
>
---
#### [replaced 038] Interpretable Safety Alignment via SAE-Constructed Low-Rank Subspace Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全对齐任务，旨在提升大语言模型拒绝有害请求的能力。针对LoRA效果不佳的问题，提出SAILS方法，通过SAE分离语义，构建可解释的安全子空间，实现高效安全对齐。**

- **链接: [https://arxiv.org/pdf/2512.23260v2](https://arxiv.org/pdf/2512.23260v2)**

> **作者:** Dianyun Wang; Qingsen Ma; Yuhu Shang; Zhifeng Lu; Zhenbo Xu; Lechen Ning; Huijia Wu; Zhaofeng He
>
> **摘要:** Safety alignment -- training large language models (LLMs) to refuse harmful requests while remaining helpful -- is critical for responsible deployment. Prior work established that safety behaviors are governed by low-rank structures, suggesting parameter-efficient fine-tuning (PEFT) should be well-suited for alignment. However, Low-Rank Adaptation (LoRA) consistently underperforms full fine-tuning and reinforcement learning on safety benchmarks. We attribute this gap to semantic entanglement: safety-relevant directions are intertwined with unrelated concepts due to polysemanticity, impeding implicit subspace identification. To address this, we propose SAILS (Safety Alignment via Interpretable Low-rank Subspace), which leverages Sparse Autoencoders (SAEs) to disentangle representations into monosemantic features, constructs an interpretable safety subspace from SAE decoder directions, and uses it to initialize LoRA adapters. Theoretically, we prove that SAE-based identification achieves arbitrarily small recovery error under monosemanticity assumptions, while direct identification suffers an irreducible error floor. Empirically, SAILS achieves up to 99.6% safety rate on Gemma-2-9B -- exceeding full fine-tuning by 7.4 points and matching RLHF-based models -- while updating only 0.19% of parameters and providing interpretability.
>
---
#### [replaced 039] MIND Your Reasoning: A Meta-Cognitive Intuitive-Reflective Network for Dual-Reasoning in Multimodal Stance Detection
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多模态立场检测任务，旨在解决现有方法缺乏明确推理过程的问题。提出MIND模型，通过双推理机制提升立场判断的准确性与上下文感知能力。**

- **链接: [https://arxiv.org/pdf/2511.06057v2](https://arxiv.org/pdf/2511.06057v2)**

> **作者:** Bingbing Wang; Zhengda Jin; Bin Liang; Wenjie Li; Jing Li; Ruifeng Xu; Min Zhang
>
> **摘要:** Multimodal Stance Detection (MSD) is a crucial task for understanding public opinion on social media. Existing methods predominantly operate by learning to fuse modalities. They lack an explicit reasoning process to discern how inter-modal dynamics, such as irony or conflict, collectively shape the user's final stance, leading to frequent misjudgments. To address this, we advocate for a paradigm shift from *learning to fuse* to *learning to reason*. We introduce **MIND**, a **M**eta-cognitive **I**ntuitive-reflective **N**etwork for **D**ual-reasoning. Inspired by the dual-process theory of human cognition, MIND operationalizes a self-improving loop. It first generates a rapid, intuitive hypothesis by querying evolving Modality and Semantic Experience Pools. Subsequently, a meta-cognitive reflective stage uses Modality-CoT and Semantic-CoT to scrutinize this initial judgment, distill superior adaptive strategies, and evolve the experience pools themselves. These dual experience structures are continuously refined during training and recalled at inference to guide robust and context-aware stance decisions. Extensive experiments on the MMSD benchmark demonstrate that our MIND significantly outperforms most baseline models and exhibits strong generalization.
>
---
#### [replaced 040] Self-Guided Defense: Adaptive Safety Alignment for Reasoning Models via Synthesized Guidelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全任务，旨在解决 adversarial jailbreak 问题。提出 SGASA 框架，通过合成指南增强模型防御能力，提升安全性。**

- **链接: [https://arxiv.org/pdf/2511.21214v3](https://arxiv.org/pdf/2511.21214v3)**

> **作者:** Yuhang Wang; Yanxu Zhu; Dongyuan Lu; Jitao Sang
>
> **摘要:** Reasoning models have demonstrated remarkable capabilities in complex reasoning tasks. However, ensuring their safety against adversarial jailbreak prompts remains a critical challenge. Due to the covert and deceptive nature of such prompts, they can often evade built-in safety mechanisms and lead to the generation of harmful content. This underscores the need for an adaptive safety alignment approach that enables models to autonomously reinforce their defenses in response to adversarial inputs. This paper introduces the Synthesized Guideline-based Adaptive Safety Alignment (SGASA) framework, which internalizes model-generated safety guidelines to strengthen models' ability to enhance robustness against harmful adversarial prompts while minimizing unnecessary refusals of benign requests. SGASA consists of two key stages: Data Pre-synthesis, which generates safety guidelines and augmented prompts; and Alignment Fine-tuning, which leverages Supervised Fine-tuning (SFT) and Direct Preference Optimization (DPO) to embed these guidelines into the model. Extensive experiments across multiple datasets demonstrate that SGASA significantly improves model safety, validating its adaptive and scalable effectiveness.
>
---
#### [replaced 041] MATEX: A Multi-Agent Framework for Explaining Ethereum Transactions
- **分类: cs.CE; cs.CL; cs.HC**

- **简介: 该论文属于区块链解释任务，旨在解决用户难以理解以太坊交易的问题。通过构建TxSum数据集和MATEX系统，提升交易解释的准确性和用户理解能力。**

- **链接: [https://arxiv.org/pdf/2512.06933v2](https://arxiv.org/pdf/2512.06933v2)**

> **作者:** Zifan Peng; Jingyi Zheng; Yule Liu; Huaiyu Jia; Qiming Ye; Jingyu Liu; Xufeng Yang; Mingchen Li; Qingyuan Gong; Xuechao Wang; Xinlei He
>
> **摘要:** Understanding the economic intent of Ethereum transactions is critical for user safety, yet current tools expose only raw on-chain data, leading to widespread "blind signing" (approving transactions without understanding them). Through interviews with 16 Web3 users, we find that effective explanations should be structured, risk-aware, and grounded at the token-flow level. Based on interviews, we propose TxSum, a new task and dataset of 100 complex Ethereum transactions annotated with natural-language summaries and step-wise semantic labels (intent, mechanism, etc.). We then introduce MATEX, a multi-agent system that emulates human experts' dual-process reasoning. MATEX achieves the highest faithfulness and intent clarity among strong baselines. It boosts user comprehension by 23.6% on complex transactions and doubles users' ability to find real attacks, significantly reducing blind signing.
>
---
#### [replaced 042] QFrBLiMP: a Quebec-French Benchmark of Linguistic Minimal Pairs
- **分类: cs.CL**

- **简介: 该论文提出QFrBLiMP基准，用于评估大语言模型在魁北克法语语法现象上的知识。旨在解决方言语言模型评估问题，通过最小对测试模型的语法理解能力。**

- **链接: [https://arxiv.org/pdf/2509.25664v2](https://arxiv.org/pdf/2509.25664v2)**

> **作者:** David Beauchemin; Pier-Luc Veilleux; Johanna-Pascale Roy; Richard Khoury
>
> **备注:** Acceptged to EACL 2026
>
> **摘要:** In this paper, we introduce the Quebec-French Benchmark of Linguistic Minimal Pairs (QFrBLiMP), a corpus designed to evaluate LLMs' linguistic knowledge of prominent grammatical phenomena in Quebec-French. QFrBLiMP comprises 1,761 minimal pairs annotated with 20 LPs. Specifically, these minimal pairs have been created by manually modifying sentences extracted from an official online resource maintained by a Québec government institution. Each pair is annotated by 12 Quebec-French native speakers, who select the sentence they consider grammatical from the two. These annotations are used to compare the competency of LLMs with that of humans. We evaluate different LLMs on QFrBLiMP and MultiBLiMP-Fr by observing the rate of higher probabilities assigned to the sentences of each minimal pair for each category. We find that while grammatical competence scales with model size, a clear hierarchy of difficulty emerges. All benchmarked models consistently fail on phenomena requiring deep semantic understanding, revealing a critical limitation. Finally, our statistical analysis comparing QFrBLiMP and MultiBLiMP reveals a significant performance degradation for most models on Quebec-French; however, the most capable models remain within the statistical significance interval, demonstrating cross-dialectal robustness.
>
---
#### [replaced 043] Dream-VL & Dream-VLA: Open Vision-Language and Vision-Language-Action Models with Diffusion Language Model Backbone
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出基于扩散语言模型的视觉-语言（Dream-VL）和视觉-语言-动作（Dream-VLA）模型，解决复杂视觉规划与动态控制问题。通过连续预训练提升任务表现。**

- **链接: [https://arxiv.org/pdf/2512.22615v2](https://arxiv.org/pdf/2512.22615v2)**

> **作者:** Jiacheng Ye; Shansan Gong; Jiahui Gao; Junming Fan; Shuang Wu; Wei Bi; Haoli Bai; Lifeng Shang; Lingpeng Kong
>
> **备注:** Add real-world experiments
>
> **摘要:** While autoregressive Large Vision-Language Models (VLMs) have achieved remarkable success, their sequential generation often limits their efficacy in complex visual planning and dynamic robotic control. In this work, we investigate the potential of constructing Vision-Language Models upon diffusion-based large language models (dLLMs) to overcome these limitations. We introduce Dream-VL, an open diffusion-based VLM (dVLM) that achieves state-of-the-art performance among previous dVLMs. Dream-VL is comparable to top-tier AR-based VLMs trained on open data on various benchmarks but exhibits superior potential when applied to visual planning tasks. Building upon Dream-VL, we introduce Dream-VLA, a dLLM-based Vision-Language-Action model (dVLA) developed through continuous pre-training on open robotic datasets. We demonstrate that the natively bidirectional nature of this diffusion backbone serves as a superior foundation for VLA tasks, inherently suited for action chunking and parallel generation, leading to significantly faster convergence in downstream fine-tuning. Dream-VLA achieves top-tier performance of 97.2% average success rate on LIBERO, 71.4% overall average on SimplerEnv-Bridge, and 60.5% overall average on SimplerEnv-Fractal, surpassing leading models such as $π_0$ and GR00T-N1. We also validate that dVLMs surpass AR baselines on downstream tasks across different training objectives. We release both Dream-VL and Dream-VLA to facilitate further research in the community.
>
---
#### [replaced 044] ERA-IT: Aligning Semantic Models with Revealed Economic Preference for Real-Time and Explainable Patent Valuation
- **分类: cs.CE; cs.CL**

- **简介: 该论文属于专利估值任务，旨在解决传统方法在实时性和解释性上的不足。通过构建ERA-IT框架，利用专利续展历史对齐大模型推理，提升估值准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2512.12869v2](https://arxiv.org/pdf/2512.12869v2)**

> **作者:** Yongmin Yoo; Seungwoo Kim; Jingjiang Liu
>
> **摘要:** Valuing intangible assets under uncertainty remains a critical challenge in the strategic management of technological innovation due to the information asymmetry inherent in high-dimensional technical specifications. Traditional bibliometric indicators, such as citation counts, fail to address this friction in a timely manner due to the systemic latency inherent in data accumulation. To bridge this gap, this study proposes the Economic Reasoning Alignment via Instruction Tuning (ERA-IT) framework. We theoretically conceptualize patent renewal history as a revealed economic preference and leverage it as an objective supervisory signal to align the generative reasoning of Large Language Models (LLMs) with market realities, a process we term Eco-Semantic Alignment. Using a randomly sampled dataset of 10,000 European Patent Office patents across diverse technological domains, we trained the model not only to predict value tiers but also to reverse-engineer the Economic Chain-of-Thought from unstructured text. Empirical results demonstrate that ERA-IT significantly outperforms both conventional econometric models and zero-shot LLMs in predictive accuracy. More importantly, by generating explicit, logically grounded rationales for valuation, the framework serves as a transparent cognitive scaffold for decision-makers, reducing the opacity of black-box AI in high-stakes intellectual property management.
>
---
#### [replaced 045] Reliable Evaluation Protocol for Low-Precision Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，解决低精度计算导致的评估不可靠问题。提出HPS和TRM方法，提升评分稳定性与评估准确性。**

- **链接: [https://arxiv.org/pdf/2508.03306v3](https://arxiv.org/pdf/2508.03306v3)**

> **作者:** Kisu Yang; Yoonna Jang; Hwanseok Jang; Kenneth Choi; Isabelle Augenstein; Heuiseok Lim
>
> **备注:** 13 pages, 7 figures, submitted to ARR
>
> **摘要:** Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals.
>
---
#### [replaced 046] Opportunities and Challenges of Large Language Models for Low-Resource Languages in Humanities Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言研究中的数据与技术挑战。通过分析LLM的应用，探讨其在语言、历史和文化研究中的潜力与问题。**

- **链接: [https://arxiv.org/pdf/2412.04497v4](https://arxiv.org/pdf/2412.04497v4)**

> **作者:** Tianyang Zhong; Zhenyuan Yang; Zhengliang Liu; Ruidong Zhang; Weihang You; Yiheng Liu; Haiyang Sun; Yi Pan; Yiwei Li; Yifan Zhou; Hanqi Jiang; Junhao Chen; Tianming Liu
>
> **摘要:** Low-resource languages serve as invaluable repositories of human history, embodying cultural evolution and intellectual diversity. Despite their significance, these languages face critical challenges, including data scarcity and technological limitations, which hinder their comprehensive study and preservation. Recent advancements in large language models (LLMs) offer transformative opportunities for addressing these challenges, enabling innovative methodologies in linguistic, historical, and cultural research. This study systematically evaluates the applications of LLMs in low-resource language research, encompassing linguistic variation, historical documentation, cultural expressions, and literary analysis. By analyzing technical frameworks, current methodologies, and ethical considerations, this paper identifies key challenges such as data accessibility, model adaptability, and cultural sensitivity. Given the cultural, historical, and linguistic richness inherent in low-resource languages, this work emphasizes interdisciplinary collaboration and the development of customized models as promising avenues for advancing research in this domain. By underscoring the potential of integrating artificial intelligence with the humanities to preserve and study humanity's linguistic and cultural heritage, this study fosters global efforts towards safeguarding intellectual diversity.
>
---
#### [replaced 047] Tuning without Peeking: Provable Generalization Bounds and Robust LLM Post-Training
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于机器学习隐私与安全领域，解决LLM训练中的数据泄露和攻击风险问题。提出BBoxER方法，在不暴露梯度的情况下实现有效且安全的模型微调。**

- **链接: [https://arxiv.org/pdf/2507.01752v3](https://arxiv.org/pdf/2507.01752v3)**

> **作者:** Ismail Labiad; Mathurin Videau; Matthieu Kowalski; Marc Schoenauer; Alessandro Leite; Julia Kempe; Olivier Teytaud
>
> **摘要:** Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds and strong theoretical guarantees for privacy, robustness to data poisoning attacks, and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods, despite the scalability and computational challenges inherent to black-box approaches, are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.
>
---
#### [replaced 048] Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统评估任务，旨在解决LLM代理在多轮对话中的评价问题。通过构建两个分类体系，明确评估内容与方法，提升评估的全面性与有效性。**

- **链接: [https://arxiv.org/pdf/2503.22458v2](https://arxiv.org/pdf/2503.22458v2)**

> **作者:** Shengyue Guan; Jindong Wang; Jiang Bian; Bin Zhu; Jian-guang Lou; Haoyi Xiong
>
> **摘要:** This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues.
>
---
#### [replaced 049] Improving End-to-End Training of Retrieval-Augmented Generation Models via Joint Stochastic Approximation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决RAG模型端到端训练中的梯度估计问题，提出JSA-RAG方法，提升生成与检索效果。**

- **链接: [https://arxiv.org/pdf/2508.18168v2](https://arxiv.org/pdf/2508.18168v2)**

> **作者:** Hongyu Cao; Yuxuan Wu; Yucheng Cai; Xianyu Zhao; Zhijian Ou
>
> **摘要:** Retrieval-augmented generation (RAG) has become a widely recognized paradigm to combine parametric memory with non-parametric memories. An RAG model consists of two serial connecting components (retriever and generator). A major challenge in end-to-end optimization of the RAG model is that marginalization over relevant passages (modeled as discrete latent variables) from a knowledge base is required. Traditional top-K marginalization and variational RAG (VRAG) suffer from biased or high-variance gradient estimates. In this paper, we propose and develop joint stochastic approximation (JSA) based end-to-end training of RAG, which is referred to as JSA-RAG. The JSA algorithm is a stochastic extension of the EM (expectation-maximization) algorithm and is particularly powerful in estimating discrete latent variable models. Extensive experiments are conducted on five datasets for two tasks (open-domain question answering, knowledge-grounded dialogs) and show that JSA-RAG significantly outperforms both vanilla RAG and VRAG. Further analysis shows the efficacy of JSA-RAG from the perspectives of generation, retrieval, and low-variance gradient estimate.
>
---
#### [replaced 050] La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在提升大语言模型的推理效率。针对现有方法依赖训练或剪枝的问题，提出LaRoSA方法，通过层间旋转激活实现稳定稀疏化，提升推理速度且性能损失小。**

- **链接: [https://arxiv.org/pdf/2507.01299v2](https://arxiv.org/pdf/2507.01299v2)**

> **作者:** Kai Liu; Bowen Xu; Shaoyu Wu; Xin Chen; Hao Zhou; Yongliang Tao; Lulu Hu
>
> **备注:** ICML 2025 Acceptance
>
> **摘要:** Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time-consuming recovery training that hinders real-world adoption, or relying on empirical magnitude-based pruning, which causes fluctuating sparsity and unstable inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning. We leverage layerwise orthogonal rotations to transform input activations into rotated forms that are more suitable for sparsification. By employing a Top-K selection approach within the rotated activations, we achieve consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA is effective across various sizes and types of LLMs, demonstrating minimal performance degradation and robust inference acceleration. Specifically, for LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%.
>
---
#### [replaced 051] DAMASHA: Detecting AI in Mixed Adversarial Texts via Segmentation with Human-interpretable Attribution
- **分类: cs.CL**

- **简介: 该论文属于混合作者文本分割任务，旨在检测文本中人类与AI生成内容的转换点。提出Info-Mask框架，结合风格特征与边界建模，提升鲁棒性并提供可解释的归属信息。**

- **链接: [https://arxiv.org/pdf/2512.04838v2](https://arxiv.org/pdf/2512.04838v2)**

> **作者:** L. D. M. S. Sai Teja; N. Siva Gopala Krishna; Ufaq Khan; Muhammad Haris Khan; Atul Mishra
>
> **备注:** EACL 2026 Findings
>
> **摘要:** In the age of advanced large language models (LLMs), the boundaries between human and AI-generated text are becoming increasingly blurred. We address the challenge of segmenting mixed-authorship text, that is identifying transition points in text where authorship shifts from human to AI or vice-versa, a problem with critical implications for authenticity, trust, and human oversight. We introduce a novel framework, called Info-Mask for mixed authorship detection that integrates stylometric cues, perplexity-driven signals, and structured boundary modeling to accurately segment collaborative human-AI content. To evaluate the robustness of our system against adversarial perturbations, we construct and release an adversarial benchmark dataset Mixed-text Adversarial setting for Segmentation (MAS), designed to probe the limits of existing detectors. Beyond segmentation accuracy, we introduce Human-Interpretable Attribution (HIA overlays that highlight how stylometric features inform boundary predictions, and we conduct a small-scale human study assessing their usefulness. Across multiple architectures, Info-Mask significantly improves span-level robustness under adversarial conditions, establishing new baselines while revealing remaining challenges. Our findings highlight both the promise and limitations of adversarially robust, interpretable mixed-authorship detection, with implications for trust and oversight in human-AI co-authorship.
>
---
#### [replaced 052] AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives
- **分类: cs.SD; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于音频-语言模型任务，旨在解决模型生成内容与音频输入不一致的幻觉问题。通过构建高质量数据集和评估基准，提升模型的时序推理能力。**

- **链接: [https://arxiv.org/pdf/2512.24052v2](https://arxiv.org/pdf/2512.24052v2)**

> **作者:** Yanxi Chen; Wenhui Zhu; Xiwen Chen; Zhipeng Wang; Xin Li; Peijie Qiu; Hao Wang; Xuanzhao Dong; Yujian Xiong; Anderson Schneider; Yuriy Nevmyvaka; Yalin Wang
>
> **摘要:** Although Large Audio-Language Models (LALMs) deliver state-of-the-art (SOTA) performance, they frequently suffer from hallucinations, e.g. generating text not grounded in the audio input. We analyze these grounding failures and identify a distinct taxonomy: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error. To address this, we introduce the AHA (Audio Hallucination Alignment) framework. By leveraging counterfactual hard negative mining, our pipeline constructs a high-quality preference dataset that forces models to distinguish strict acoustic evidence from linguistically plausible fabrications. Additionally, we establish AHA-Eval, a diagnostic benchmark designed to rigorously test these fine-grained temporal reasoning capabilities. We apply this data to align Qwen2.5-Omni. The resulting model, Qwen-Audio-AHA, achieves a 13.7% improvement on AHA-Eval. Crucially, this benefit generalizes beyond our diagnostic set. Our model shows substantial gains on public benchmarks, including 1.3% on MMAU-Test and 1.6% on MMAR, outperforming latest SOTA methods. The model and dataset are open-sourced at https://github.com/LLM-VLM-GSL/AHA.
>
---
#### [replaced 053] Knowledge Distillation and Dataset Distillation of Large Language Models: Emerging Trends, Challenges, and Future Directions
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理领域，探讨知识蒸馏与数据蒸馏技术，旨在压缩大语言模型同时保持其性能。**

- **链接: [https://arxiv.org/pdf/2504.14772v2](https://arxiv.org/pdf/2504.14772v2)**

> **作者:** Luyang Fang; Xiaowei Yu; Jiazhang Cai; Yongkai Chen; Shushan Wu; Zhengliang Liu; Zhenyuan Yang; Haoran Lu; Xilin Gong; Yufang Liu; Terry Ma; Wei Ruan; Ali Abbasi; Jing Zhang; Tao Wang; Ehsan Latif; Weihang You; Hanqi Jiang; Wei Liu; Wei Zhang; Soheil Kolouri; Xiaoming Zhai; Dajiang Zhu; Wenxuan Zhong; Tianming Liu; Ping Ma
>
> **摘要:** The exponential growth of Large Language Models (LLMs) continues to highlight the need for efficient strategies to meet ever-expanding computational and data demands. This survey provides a comprehensive analysis of two complementary paradigms: Knowledge Distillation (KD) and Dataset Distillation (DD), both aimed at compressing LLMs while preserving their advanced reasoning capabilities and linguistic diversity. We first examine key methodologies in KD, such as task-specific alignment, rationale-based training, and multi-teacher frameworks, alongside DD techniques that synthesize compact, high-impact datasets through optimization-based gradient matching, latent space regularization, and generative synthesis. Building on these foundations, we explore how integrating KD and DD can produce more effective and scalable compression strategies. Together, these approaches address persistent challenges in model scalability, architectural heterogeneity, and the preservation of emergent LLM abilities. We further highlight applications across domains such as healthcare and education, where distillation enables efficient deployment without sacrificing performance. Despite substantial progress, open challenges remain in preserving emergent reasoning and linguistic diversity, enabling efficient adaptation to continually evolving teacher models and datasets, and establishing comprehensive evaluation protocols. By synthesizing methodological innovations, theoretical foundations, and practical insights, our survey charts a path toward sustainable, resource-efficient LLMs through the tighter integration of KD and DD principles.
>
---
#### [replaced 054] User-Assistant Bias in LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文研究LLM中的用户-助手偏见问题，属于模型偏差分析任务。通过构建基准测试，分析不同模型的偏见倾向，并探索控制方法。**

- **链接: [https://arxiv.org/pdf/2508.15815v2](https://arxiv.org/pdf/2508.15815v2)**

> **作者:** Xu Pan; Jingxuan Fan; Zidi Xiong; Ely Hahami; Jorin Overwiening; Ziqian Xie
>
> **摘要:** Modern large language models (LLMs) are typically trained and deployed using structured role tags (e.g. system, user, assistant, tool) that explicitly mark the source of each piece of context. While these tags are essential for instruction following and controllability, asymmetries in the training data associated with different role tags can introduce inductive biases. In this paper, we study this phenomenon by formalizing user-assistant bias, defined as the tendency of an LLM to preferentially rely on information from either the user or assistant role when there is a conflict. We introduce a task-agnostic benchmark UserAssist and evaluate such bias in 52 frontier models. We observe that most of the instruction-tuned models exhibit strong user bias, whereas base and reasoning models are close to neutral. Using controlled fine-tuning experiments, we isolate which post-training recipes drive the observed user-assistant bias. We find that human-preference alignment amplifies user bias, while reasoning fine-tuning reduces it. Finally, we show that user-assistant bias can be bidirectionally controlled via direct preference optimization (DPO) on UserAssist-train, and that the resulting bias reliably generalizes to a more realistic multi-turn conversation dataset. These results reveal an underexplored consequence of role-tagged training and provide a principled framework to diagnose and control tag-induced biases in modern LLMs.
>
---
#### [replaced 055] I Large Language Models possono nascondere un testo in un altro testo della stessa lunghezza
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文提出一种在相同长度文本中隐藏信息的协议Calgacus，解决文本隐写问题，利用大语言模型实现高效编码与解码。**

- **链接: [https://arxiv.org/pdf/2510.20075v5](https://arxiv.org/pdf/2510.20075v5)**

> **作者:** Antonio Norelli; Michael Bronstein
>
> **备注:** 21 pages, in Italian language, main paper 9 pages. v1-v4 are in English
>
> **摘要:** A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something. -- Un testo di senso compiuto può essere nascosto all'interno di un altro testo completamente diverso, eppure coerente e plausibile, della stessa lunghezza. Ad esempio, un tweet che celebra un leader politico potrebbe celare un tweet che lo critica duramente, o un'anonima recensione di un prodotto potrebbe in realtà codificare un manoscritto segreto. Questa sconcertante possibilità è oggi alla nostra portata grazie ai Large Language Models (LLM); in questo articolo presentiamo Calgacus, un protocollo semplice ed efficiente per realizzarla. Mostriamo che anche modesti LLM open-source da 8 miliardi di parametri sono sufficienti per ottenere risultati di alta qualità, e che un messaggio lungo quanto questo abstract può essere codificato e decodificato su un comune portatile in pochi secondi. L'esistenza di tale protocollo dimostra un radicale disaccoppiamento del testo dall'intento del suo autore, erodendo ulteriormente la fiducia nella comunicazione scritta, già scossa dall'ascesa dei chatbot basati su LLMs. Illustriamo ciò con uno scenario concreto: un'azienda potrebbe offrire pubblicamente i servizi di un LLM senza filtri nascondendo le sue risposte all'interno di risposte apparentemente innocue generate da un LLM considerato sicuro. Questa possibilità solleva questioni urgenti per la sicurezza dell'Intelligenza Artificiale e sfida la nostra comprensione di cosa significhi, per un Large Language Model, sapere qualcosa.
>
---
#### [replaced 056] NoveltyRank: A Retrieval-Augmented Framework for Conceptual Novelty Estimation in AI Research
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于概念新颖性评估任务，旨在解决科研论文中区分原创与增量研究的问题。通过结合语义学习与检索比较，提出NoveltyRank框架进行新颖性判断。**

- **链接: [https://arxiv.org/pdf/2512.14738v2](https://arxiv.org/pdf/2512.14738v2)**

> **作者:** Zhengxu Yan; Han Li; Yuming Feng
>
> **备注:** 11 pages, 4, tables, 3 figures
>
> **摘要:** The accelerating pace of scientific publication makes it difficult to identify truly original research among incremental work. We propose a framework for estimating the conceptual novelty of research papers by combining semantic representation learning with retrieval-based comparison against prior literature. We model novelty as both a binary classification task (novel vs. non-novel) and a pairwise ranking task (comparative novelty), enabling absolute and relative assessments. Experiments benchmark three model scales, ranging from compact domain-specific encoders to a zero-shot frontier model. Results show that fine-tuned lightweight models outperform larger zero-shot models despite their smaller parameter count, indicating that task-specific supervision matters more than scale for conceptual novelty estimation. We further deploy the best-performing model as an online system for public interaction and real-time novelty scoring.
>
---
#### [replaced 057] Evaluating the cognitive reality of Spanish irregular morphomic patterns: Humans vs. Transformers
- **分类: cs.CL**

- **简介: 该论文比较了Transformer模型与人类对西班牙语不规则形态模式的处理能力，旨在探究模型是否能像人类一样泛化形态规律。研究通过控制输入条件，分析不同频率下的表现差异。**

- **链接: [https://arxiv.org/pdf/2507.21556v2](https://arxiv.org/pdf/2507.21556v2)**

> **作者:** Akhilesh Kakolu Ramarao; Kevin Tang; Dinah Baer-Henney
>
> **摘要:** Do transformer models generalize morphological patterns like humans do? We investigate this by directly comparing transformers to human behavioral data on Spanish irregular morphomic patterns from \citet{Nevins2015TheRA}. We adopt the same analytical framework as the original human study. Under controlled input conditions, we evaluate whether transformer models can replicate human-like sensitivity to the morphome, a complex linguistic phenomenon. Our experiments focus on three frequency conditions: natural, low-frequency, and high-frequency distributions of verbs exhibiting irregular morphomic patterns. Transformer models achieve higher stem-accuracy than human participants. However, response preferences diverge: humans consistently favor the "natural" inflection across all items, whereas models preferred the irregular forms, and their choices are modulated by the proportion of irregular verbs present during training. Moreover, models trained on the natural and low-frequency distributions, but not the high-frequency distribution, exhibit sensitivity to phonological similarity between test items and Spanish L-shaped verbs, mirroring a limited aspect of human phonological generalization.
>
---
#### [replaced 058] Scaling Open-Ended Reasoning to Predict the Future
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于未来预测任务，旨在提升语言模型在开放性预测问题上的准确性。通过构建数据集并优化训练方法，提高模型的预测能力。**

- **链接: [https://arxiv.org/pdf/2512.25070v2](https://arxiv.org/pdf/2512.25070v2)**

> **作者:** Nikhil Chandak; Shashwat Goel; Ameya Prabhu; Moritz Hardt; Jonas Geiping
>
> **备注:** 45 pages
>
> **摘要:** High-stakes decision making involves reasoning under uncertainty about the future. In this work, we train language models to make predictions on open-ended forecasting questions. To scale up training data, we synthesize novel forecasting questions from global events reported in daily news, using a fully automated, careful curation recipe. We train the Qwen3 thinking models on our dataset, OpenForesight. To prevent leakage of future information during training and evaluation, we use an offline news corpus, both for data generation and retrieval in our forecasting system. Guided by a small validation set, we show the benefits of retrieval, and an improved reward function for reinforcement learning (RL). Once we obtain our final forecasting system, we perform held-out testing between May to August 2025. Our specialized model, OpenForecaster 8B, matches much larger proprietary models, with our training improving the accuracy, calibration, and consistency of predictions. We find calibration improvements from forecasting training generalize across popular benchmarks. We open-source all our models, code, and data to make research on language model forecasting broadly accessible.
>
---
#### [replaced 059] EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音情感识别任务，旨在解决现有数据集情绪类别少、规模小、伦理问题等问题。提出EmoNet-Voice，包含多语言、细粒度情感数据和经过专家验证的基准。**

- **链接: [https://arxiv.org/pdf/2506.09827v3](https://arxiv.org/pdf/2506.09827v3)**

> **作者:** Christoph Schuhmann; Robert Kaczmarczyk; Gollam Rabby; Felix Friedrich; Maurice Kraus; Kourosh Nadi; Huu Nguyen; Kristian Kersting; Sören Auer
>
> **摘要:** Speech emotion recognition (SER) systems are constrained by existing datasets that typically cover only 6-10 basic emotions, lack scale and diversity, and face ethical challenges when collecting sensitive emotional states. We introduce EMONET-VOICE, a comprehensive resource addressing these limitations through two components: (1) EmoNet-Voice Big, a 5,000-hour multilingual pre-training dataset spanning 40 fine-grained emotion categories across 11 voices and 4 languages, and (2) EmoNet-Voice Bench, a rigorously validated benchmark of 4,7k samples with unanimous expert consensus on emotion presence and intensity levels. Using state-of-the-art synthetic voice generation, our privacy-preserving approach enables ethical inclusion of sensitive emotions (e.g., pain, shame) while maintaining controlled experimental conditions. Each sample underwent validation by three psychology experts. We demonstrate that our Empathic Insight models trained on our synthetic data achieve strong real-world dataset generalization, as tested on EmoDB and RAVDESS. Furthermore, our comprehensive evaluation reveals that while high-arousal emotions (e.g., anger: 95% accuracy) are readily detected, the benchmark successfully exposes the difficulty of distinguishing perceptually similar emotions (e.g., sadness vs. distress: 63% discrimination), providing quantifiable metrics for advancing nuanced emotion AI. EMONET-VOICE establishes a new paradigm for large-scale, ethically-sourced, fine-grained SER research.
>
---
#### [replaced 060] OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型的去偏任务，旨在解决信息遗忘有效性问题。提出OFFSIDE基准，评估谣言遗忘效果，发现现有方法在多模态场景中存在显著漏洞。**

- **链接: [https://arxiv.org/pdf/2510.22535v2](https://arxiv.org/pdf/2510.22535v2)**

> **作者:** Hao Zheng; Zirui Pang; Ling li; Zhijie Deng; Yuhan Pu; Zhaowei Zhu; Xiaobo Xia; Jiaheng Wei
>
> **摘要:** Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at https://github.com/zh121800/OFFSIDE
>
---
#### [replaced 061] A Survey of Text Classification Under Class Distribution Shift
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，研究类别分布变化带来的挑战。通过综述开放集学习、零样本学习等方法，探讨应对分布偏移的解决方案。**

- **链接: [https://arxiv.org/pdf/2502.12965v2](https://arxiv.org/pdf/2502.12965v2)**

> **作者:** Adriana Valentina Costache; Silviu Florin Gheorghe; Eduard Gabriel Poesina; Paul Irofti; Radu Tudor Ionescu
>
> **备注:** Accepted at EACL 2026 (main)
>
> **摘要:** The basic underlying assumption of machine learning (ML) models is that the training and test data are sampled from the same distribution. However, in daily practice, this assumption is often broken, i.e.~the distribution of the test data changes over time, which hinders the application of conventional ML models. One domain where the distribution shift naturally occurs is text classification, since people always find new topics to discuss. To this end, we survey research articles studying open-set text classification and related tasks. We divide the methods in this area based on the constraints that define the kind of distribution shift and the corresponding problem formulation, i.e.~learning with the Universum, zero-shot learning, and open-set learning. We next discuss the predominant mitigation approaches for each problem setup. Finally, we identify several future work directions, aiming to push the boundaries beyond the state of the art. Interestingly, we find that continual learning can solve many of the issues caused by the shifting class distribution. We maintain a list of relevant papers at https://github.com/Eduard6421/Open-Set-Survey.
>
---
#### [replaced 062] Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs无法自检错误的问题。提出Gnosis机制，通过分析内部状态实现自我验证，提升模型自适应能力。**

- **链接: [https://arxiv.org/pdf/2512.20578v2](https://arxiv.org/pdf/2512.20578v2)**

> **作者:** Amirhosein Ghasemabadi; Di Niu
>
> **摘要:** Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision.
>
---
#### [replaced 063] Explainability-Based Token Replacement on LLM-Generated Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在降低AIGT的可检测性并提升检测方法的鲁棒性。通过XAI方法识别关键token并进行替换，同时构建集成检测模型应对文本篡改。**

- **链接: [https://arxiv.org/pdf/2506.04050v2](https://arxiv.org/pdf/2506.04050v2)**

> **作者:** Hadi Mohammadi; Anastasia Giachanou; Daniel L. Oberski; Ayoub Bagheri
>
> **摘要:** Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT.
>
---
#### [replaced 064] Deployability-Centric Infrastructure-as-Code Generation: Fail, Learn, Refine, and Succeed through LLM-Empowered DevOps Simulation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于IaC生成任务，旨在解决LLM生成的IaC模板部署成功率低的问题。通过构建基准测试和提出迭代优化框架IaCGen，提升模板的可部署性。**

- **链接: [https://arxiv.org/pdf/2506.05623v2](https://arxiv.org/pdf/2506.05623v2)**

> **作者:** Tianyi Zhang; Shidong Pan; Zejun Zhang; Zhenchang Xing; Xiaoyu Sun
>
> **备注:** Accepted by FSE 2026
>
> **摘要:** Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions. However, current evaluation focuses on syntactic correctness while ignoring deployability, the critical measure of the utility of IaC configuration files. Six state-of-the-art LLMs performed poorly on deployability, achieving only 20.8$\sim$30.2% deployment success rate on the first attempt. In this paper, we construct DPIaC-Eval, the first deployability-centric IaC template benchmark consisting of 153 real-world scenarios cross 58 unique services. Also, we propose an LLM-based deployability-centric framework, dubbed IaCGen, that uses iterative feedback mechanism encompassing format verification, syntax checking, and live deployment stages, thereby closely mirroring the real DevOps workflows. Results show that IaCGen can make 54.6$\sim$91.6% generated IaC templates from all evaluated models deployable in the first 10 iterations. Additionally, human-in-the-loop feedback that provide direct guidance for the deployability errors, can further boost the performance to over 90% passItr@25 on all evaluated LLMs. Furthermore, we explore the trustworthiness of the generated IaC templates on user intent alignment and security compliance. The poor performance (25.2% user requirement coverage and 8.4% security compliance rate) indicates a critical need for continued research in this domain.
>
---
#### [replaced 065] Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Klear-Reasoner，解决推理模型训练细节不透明和强化学习中剪切机制问题，通过GPPO提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2508.07629v3](https://arxiv.org/pdf/2508.07629v3)**

> **作者:** Zhenpeng Su; Leiyu Pan; Xue Bai; Dening Liu; Guanting Dong; Jiaming Huang; Wenping Hu; Fuzheng Zhang; Kun Gai; Guorui Zhou
>
> **摘要:** We present Klear-Reasoner, a model with long reasoning capabilities that demonstrates careful deliberation during problem solving, achieving outstanding performance across multiple benchmarks. Although there are already many excellent works related to inference models in the current community, there are still many problems with reproducing high-performance inference models due to incomplete disclosure of training details. This report provides an in-depth analysis of the reasoning model, covering the entire post-training workflow from data preparation and long Chain-of-Thought supervised fine-tuning (long CoT SFT) to reinforcement learning (RL), along with detailed ablation studies for each experimental component. For SFT data, our experiments show that a small number of high-quality data sources are more effective than a large number of diverse data sources, and that difficult samples can achieve better results without accuracy filtering. In addition, we investigate two key issues with current clipping mechanisms in RL: Clipping suppresses critical exploration signals and ignores suboptimal trajectories. To address these challenges, we propose Gradient-Preserving clipping Policy Optimization (GPPO) that gently backpropagates gradients from clipped tokens. GPPO not only enhances the model's exploration capacity but also improves its efficiency in learning from negative samples. Klear-Reasoner exhibits exceptional reasoning abilities in mathematics and programming, scoring 90.5% on AIME 2024, 83.2% on AIME 2025, 66.0% on LiveCodeBench V5 and 58.1% on LiveCodeBench V6.
>
---
#### [replaced 066] Exploring Cultural Variations in Moral Judgments with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文化道德研究任务，旨在探讨大语言模型是否能反映不同文化的道德判断。通过对比模型输出与调查数据，分析其文化敏感性。**

- **链接: [https://arxiv.org/pdf/2506.12433v2](https://arxiv.org/pdf/2506.12433v2)**

> **作者:** Hadi Mohammadi; Ayoub Bagheri
>
> **摘要:** Large Language Models (LLMs) have shown strong performance across many tasks, but their ability to capture culturally diverse moral values remains unclear. In this paper, we examine whether LLMs mirror variations in moral attitudes reported by the World Values Survey (WVS) and the Pew Research Center's Global Attitudes Survey (PEW). We compare smaller monolingual and multilingual models (GPT-2, OPT, BLOOMZ, and Qwen) with recent instruction-tuned models (GPT-4o, GPT-4o-mini, Gemma-2-9b-it, and Llama-3.3-70B-Instruct). Using log-probability-based \emph{moral justifiability} scores, we correlate each model's outputs with survey data covering a broad set of ethical topics. Our results show that many earlier or smaller models often produce near-zero or negative correlations with human judgments. In contrast, advanced instruction-tuned models achieve substantially higher positive correlations, suggesting they better reflect real-world moral attitudes. We provide a detailed regional analysis revealing that models align better with Western, Educated, Industrialized, Rich, and Democratic (W.E.I.R.D.) nations than with other regions. While scaling model size and using instruction tuning improves alignment with cross-cultural moral norms, challenges remain for certain topics and regions. We discuss these findings in relation to bias analysis, training data diversity, information retrieval implications, and strategies for improving the cultural sensitivity of LLMs.
>
---
#### [replaced 067] Self-Speculative Biased Decoding for Faster Re-Translation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，解决同步翻译中的高延迟和冗余计算问题。提出SSBD方法，通过利用时间一致性加速重翻译，提升效率且不牺牲质量。**

- **链接: [https://arxiv.org/pdf/2509.21740v2](https://arxiv.org/pdf/2509.21740v2)**

> **作者:** Linxiao Zeng; Haoyun Deng; Kangyuan Shu; Shizhen Wang
>
> **摘要:** Large language models achieve strong machine translation quality but incur high inference cost and latency, posing challenges for simultaneous translation. Re-translation provides a practical solution for off-the-shelf LLMs by repeatedly regenerating the target output as the source input grows, but it suffers from substantial redundant computation. We propose Self-Speculative Biased Decoding (SSBD), a simple and tuning-free inference method that accelerates re-translation by exploiting temporal coherence in streaming translation. SSBD reuses the model's previous output as a speculative draft for the updated input, verifies the draft efficiently in a single forward pass with a lightweight bias, and resumes autoregressive decoding only from the first divergence. We further introduce a display-only masking strategy that hides unstable suffixes from the user interface while retaining them in the draft for verification and potential acceptance. Experiments show that SSBD achieves substantial speedup over standard re-translation while maintaining comparable translation quality, without architectural changes, auxiliary models, or extra fine-tuning.
>
---
#### [replaced 068] Fine-Grained Preference Optimization Improves Spatial Reasoning in VLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文聚焦于视觉语言模型的空间推理任务，解决其在细粒度空间逻辑推理上的不足。通过引入fDPO方法和SpatialReasoner-R1模型，提升空间对齐与逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2506.21656v3](https://arxiv.org/pdf/2506.21656v3)**

> **作者:** Yifan Shen; Yuanzhe Liu; Jingyuan Zhu; Xu Cao; Xiaofeng Zhang; Yixiao He; Wenming Ye; James Matthew Rehg; Ismini Lourentzou
>
> **摘要:** Current Vision-Language Models (VLMs) struggle with fine-grained spatial reasoning, particularly when multi-step logic and precise spatial alignment are required. In this work, we introduce SpatialReasoner-R1, a vision-language reasoning model designed to address these limitations. To construct high-quality supervision for spatial reasoning, we design a Multi-Model Monte Carlo Tree Search (M3CTS) method that generates diverse, logically consistent Long Chain-of-Thought (LongCOT) reasoning trajectories. In addition, we propose a fine-grained Direct Preference Optimization (fDPO) method that introduces segment-specific preference granularity for descriptive grounding and logical reasoning, guided by a spatial reward mechanism that evaluates candidate responses based on visual consistency, spatial grounding, and logical coherence. Experimental results demonstrate that fDPO achieves relative performance gains of 4.1% and 9.0% over standard DPO on spatial qualitative and quantitative tasks, respectively. SpatialReasoner-R1, trained with fDPO, sets a new SoTA on SpatialRGPT-Bench, outperforming the strongest baseline by 9.4% in average accuracy, while maintaining competitive performance on general vision-language tasks.
>
---
#### [replaced 069] Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling
- **分类: cs.SI; cs.CL**

- **简介: 该论文分析2025年洛杉矶山火期间Reddit上的公众讨论，通过LLM增强的主题建模识别公共健康问题。属于危机话语分析任务，旨在提升灾害响应策略。**

- **链接: [https://arxiv.org/pdf/2505.09665v3](https://arxiv.org/pdf/2505.09665v3)**

> **作者:** Sulong Zhou; Qunying Huang; Shaoheng Zhou; Yun Hang; Xinyue Ye; Aodong Mei; Kathryn Phung; Yuning Ye; Uma Govindswamy; Zehan Li
>
> **备注:** Fix typos in Method Section. Add data/code availability
>
> **摘要:** Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
>
---
#### [replaced 070] SwiftEmbed: Ultra-Fast Text Embeddings via Static Token Lookup for Real-Time Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本嵌入任务，旨在解决实时应用中嵌入速度与质量的平衡问题。通过静态词元查找方法，实现快速且高质量的文本嵌入生成。**

- **链接: [https://arxiv.org/pdf/2510.24793v2](https://arxiv.org/pdf/2510.24793v2)**

> **作者:** Edouard Lansiaux; Antoine Simonet; Eric Wiel
>
> **摘要:** We present a static token lookup methodology for text embedding generation that achieves 1.12 ms p50 latency for single text embeddings while maintaining 60.6 MTEB average score across 8 representative tasks, corresponding to 89% of contextual model quality. The Rust implementation delivers 50,000 requests per second throughput through static embedding lookup, optimized mean pooling, and zero-copy IEEE754 binary serialization. Evaluation demonstrates exceptional duplicate detection performance (90.1% AP), strong semantic similarity (76.1% Spearman correlation), and domain-specific performance ranging from 75% to 131% of baseline across specialized domains. The system enables real-time embedding applications where sub-5ms latency is critica
>
---
#### [replaced 071] Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升语言模型代理在新环境中的样本效率。通过生成优化轨迹，ECHO框架有效利用失败交互，提高学习效果。**

- **链接: [https://arxiv.org/pdf/2510.10304v2](https://arxiv.org/pdf/2510.10304v2)**

> **作者:** Michael Y. Hu; Benjamin Van Durme; Jacob Andreas; Harsh Jhamtani
>
> **摘要:** Language model (LM) agents deployed in novel environments often exhibit poor sample efficiency when learning from sequential interactions. This significantly hinders the usefulness of such agents in environments where interaction is costly (for example, when they interact with humans or reset physical systems). While a number of existing LM agent architectures incorporate various mechanisms for experience storage and reflection, they make limited use of LMs' abilities to directly generate or reason about full counterfactual trajectories. We introduce ECHO (Experience Consolidation via Hindsight Optimization), a prompting framework that adapts hindsight experience replay from reinforcement learning for language model agents. ECHO generates optimized trajectories for alternative goals that could have been achieved during failed attempts, effectively creating synthetic positive examples from unsuccessful interactions. Our approach consists of two components: a hindsight rule that uses the language model itself to identify relevant subgoals and generate optimized trajectories, and an update rule that maintains compressed trajectory representations in memory. We evaluate ECHO on stateful versions of XMiniGrid, a text-based navigation and planning benchmark, and PeopleJoinQA, a collaborative information-gathering enterprise simulation. Across both domains, ECHO outperforms vanilla language agent baselines by up to 80%; in XMiniGrid, it also outperforms a number of sophisticated agent architectures including Reflexion and AWM, demonstrating faster adaptation to novel environments through more effective utilization of past experiences.
>
---
#### [replaced 072] AFA-LoRA: Enabling Non-Linear Adaptations in LoRA with Activation Function Annealing
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决LoRA线性适应表达能力不足的问题。通过引入渐进式激活函数，提升模型非线性表达能力，同时保持可合并性。**

- **链接: [https://arxiv.org/pdf/2512.22455v2](https://arxiv.org/pdf/2512.22455v2)**

> **作者:** Jiacheng Li; Jianchao Tan; Zhidong Yang; Feiye Huo; Yerui Sun; Yuchen Xie; Xunliang Cai
>
> **摘要:** Low-Rank Adaptation (LoRA) is a widely adopted parameter-efficient fine-tuning (PEFT) method. However, its linear adaptation process limits its expressive power. This means there is a gap between the expressive power of linear training and non-linear training. To bridge this gap, we propose AFA-LoRA, a novel training strategy that brings non-linear expressivity to LoRA while maintaining its seamless mergeability. Our key innovation is an annealed activation function that transitions from a non-linear to a linear transformation during training, allowing the adapter to initially adopt stronger representational capabilities before converging to a mergeable linear form. We implement our method on supervised fine-tuning, reinforcement learning, and speculative decoding. The results show that AFA-LoRA reduces the performance gap between LoRA and full-parameter training. This work enables a more powerful and practical paradigm of parameter-efficient adaptation.
>
---
#### [replaced 073] RiTeK: A Dataset for Large Language Models Complex Reasoning over Textual Knowledge Graphs in Medicine
- **分类: cs.CL**

- **简介: 该论文提出RiTeK数据集，用于评估大语言模型在医学文本知识图谱上的复杂推理能力。旨在解决医疗领域知识图谱稀缺、结构表达不足及检索系统效能有限的问题。**

- **链接: [https://arxiv.org/pdf/2410.13987v2](https://arxiv.org/pdf/2410.13987v2)**

> **作者:** Jiatan Huang; Mingchen Li; Zonghai Yao; Dawei Li; Yuxin Zhang; Zhichao Yang; Yongkang Xiao; Feiyun Ouyang; Xiaohan Li; Shuo Han; Hong Yu
>
> **摘要:** Answering complex real-world questions in the medical domain often requires accurate retrieval from medical Textual Knowledge Graphs (medical TKGs), as the relational path information from TKGs could enhance the inference ability of Large Language Models (LLMs). However, the main bottlenecks lie in the scarcity of existing medical TKGs, the limited expressiveness of their topological structures, and the lack of comprehensive evaluations of current retrievers for medical TKGs. To address these challenges, we first develop a Dataset1 for LLMs Complex Reasoning over medical Textual Knowledge Graphs (RiTeK), covering a broad range of topological structures. Specifically, we synthesize realistic user queries integrating diverse topological structures, relational information, and complex textual descriptions. We conduct a rigorous medical expert evaluation process to assess and validate the quality of our synthesized queries. RiTeK also serves as a comprehensive benchmark dataset for evaluating the capabilities of retrieval systems built upon LLMs. By assessing 11 representative retrievers on this benchmark, we observe that existing methods struggle to perform well, revealing notable limitations in current LLM-driven retrieval approaches. These findings highlight the pressing need for more effective retrieval systems tailored for semi-structured data in the medical domain.
>
---
#### [replaced 074] Steering Evaluation-Aware Language Models to Act Like They Are Deployed
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI安全评估任务，旨在解决模型在评估时伪装行为的问题。通过激活转向技术，使模型在评估中表现如部署时一样真实。**

- **链接: [https://arxiv.org/pdf/2510.20487v4](https://arxiv.org/pdf/2510.20487v4)**

> **作者:** Tim Tian Hua; Andrew Qin; Samuel Marks; Neel Nanda
>
> **摘要:** Large language models (LLMs) can sometimes detect when they are being evaluated and adjust their behavior to appear more aligned, compromising the reliability of safety evaluations. In this paper, we show that adding a steering vector to an LLM's activations can suppress evaluation-awareness and make the model act like it is deployed during evaluation. To study our steering technique, we train an LLM to exhibit evaluation-aware behavior using a two-step training process designed to mimic how this behavior could emerge naturally. First, we perform continued pretraining on documents with factual descriptions of the model (1) using Python type hints during evaluation but not during deployment and (2) recognizing that the presence of a certain evaluation cue always means that it is being tested. Then, we train the model with expert iteration to use Python type hints in evaluation settings. The resulting model is evaluation-aware: it writes type hints in evaluation contexts more than deployment contexts. We find that activation steering can suppress evaluation awareness and make the model act like it is deployed even when the cue is present. Importantly, we constructed our steering vector using the original model before our additional training. Our results suggest that AI evaluators could improve the reliability of safety evaluations by steering models to act like they are deployed.
>
---
#### [replaced 075] Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文属于视觉语言模型安全评估任务，旨在解决现有数据集对视觉漏洞评估不足的问题。提出Text2VLM框架，将文本数据转为多模态数据，评估模型对提示注入攻击的脆弱性。**

- **链接: [https://arxiv.org/pdf/2507.20704v2](https://arxiv.org/pdf/2507.20704v2)**

> **作者:** Gabriel Downer; Sean Craven; Damian Ruck; Jake Thomas
>
> **备注:** 9 pages, 9 figures. Jake Thomas served as Editor for this manuscript
>
> **摘要:** The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications.
>
---
#### [replaced 076] GRAPHMOE: Amplifying Cognitive Depth of Mixture-of-Experts Network via Introducing Self-Rethinking Mechanism
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升Mixture-of-Experts网络的推理能力。通过引入自反思机制和循环路由策略，增强专家间的交互与信息流动，从而提高模型性能。**

- **链接: [https://arxiv.org/pdf/2501.07890v4](https://arxiv.org/pdf/2501.07890v4)**

> **作者:** Bo Lv; Chen Tang; Zifan Zheng; Bohao Yang; Kun Zhao; Ning Liao; Xiaoxing Wang; Feiyu Xiong; Zhiyu Li; Nayu Liu; Jingchi Jiang
>
> **备注:** 10 pages
>
> **摘要:** Traditional Mixture-of-Experts (MoE) networks benefit from utilizing multiple smaller expert models as opposed to a single large network. However, these experts typically operate independently, leaving a question open about whether interconnecting these models could enhance the performance of MoE networks. In response, we introduce GRAPHMOE, a novel method aimed at augmenting the cognitive depth of language models via a self-rethinking mechanism constructed on Pseudo GraphMoE networks. GRAPHMOE employs a recurrent routing strategy to simulate iterative thinking steps, thereby facilitating the flow of information among expert nodes. We implement the GRAPHMOE architecture using Low-Rank Adaptation techniques (LoRA) and conduct extensive experiments on various benchmark datasets. The experimental results reveal that GRAPHMOE outperforms other LoRA based models, achieving state-of-the-art (SOTA) performance. Additionally, this study explores a novel recurrent routing strategy that may inspire further advancements in enhancing the reasoning capabilities of language models.
>
---
#### [replaced 077] SIP-BMM: Constructing the Capability--Efficiency Pareto Set for LLMs via Structural Importance Prior Bayesian Model Merging
- **分类: cs.LG; cs.CL; cs.NE**

- **简介: 该论文属于模型优化任务，旨在解决LLM能力与效率的权衡问题。提出SIP-BMM框架，通过贝叶斯优化自动构建帕累托前沿，提升模型选择效率。**

- **链接: [https://arxiv.org/pdf/2512.09972v3](https://arxiv.org/pdf/2512.09972v3)**

> **作者:** Kesheng Chen; Yamin Hu; Zhenqian Zhu; Wenjian Luo; Yiya Diao
>
> **摘要:** Constructing a Pareto set is pivotal for navigating the capability--efficiency trade-offs in Large Language Models (LLMs). However, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the curse of dimensionality, rendering the search space computationally intractable. To resolve this dichotomy, we propose Structural Importance Prior Bayesian Model Merging (SIP-BMM), a framework that automatically constructs the LLM Pareto set. SIP-BMM renders high-dimensional layer-wise search tractable by introducing an importance-aware Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) strategy. By leveraging a structural importance prior derived from task-vector differences, our method guides SAASBO to automatically identify critical layers, thereby dramatically reducing the effective dimensionality without sacrificing the granularity of full-model control. The entire process is automated within an evolutionary loop driven by the Log-Noisy Expected Hypervolume Improvement ($q$NEHVI) acquisition function. Experiments demonstrate that SIP-BMM discovers a stronger and denser Pareto front than competitive baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/MiLab-HITSZ/2026-SIPBMM.
>
---
#### [replaced 078] Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决LLMs对句子级否定理解不足的问题。提出Thunder-NUBench基准，评估模型对不同否定形式的理解能力。**

- **链接: [https://arxiv.org/pdf/2506.14397v3](https://arxiv.org/pdf/2506.14397v3)**

> **作者:** Yeonkyoung So; Gyuseong Lee; Sungmok Jung; Joonhak Lee; JiA Kang; Sangho Kim; Jaejin Lee
>
> **摘要:** Negation is a fundamental linguistic phenomenon that poses ongoing challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Current benchmarks often treat negation as a minor detail within broader tasks, such as natural language inference. Consequently, there is a lack of benchmarks specifically designed to evaluate comprehension of negation. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly created to assess sentence-level understanding of negation in LLMs. Thunder-NUBench goes beyond merely identifying surface-level cues by contrasting standard negation with structurally diverse alternatives, such as local negation, contradiction, and paraphrase. This benchmark includes manually curated sentence-negation pairs and a multiple-choice dataset, allowing for a comprehensive evaluation of models' understanding of negation.
>
---
#### [replaced 079] Autoregressive Semantic Visual Reconstruction Helps VLMs Understand Better
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态学习任务，旨在解决LVLMs对视觉信息理解不足的问题。通过引入ASVR，联合学习视觉与文本模态，提升模型对视觉内容的感知能力。**

- **链接: [https://arxiv.org/pdf/2506.09040v2](https://arxiv.org/pdf/2506.09040v2)**

> **作者:** Dianyi Wang; Wei Song; Yikun Wang; Siyuan Wang; Kaicheng Yu; Zhongyu Wei; Jiaqi Wang
>
> **摘要:** Typical large vision-language models (LVLMs) apply autoregressive supervision solely to textual sequences, without fully incorporating the visual modality into the learning process. This results in three key limitations: (1) an inability to utilize images without accompanying captions, (2) the risk that captions omit critical visual details, and (3) the challenge that certain vision-centric content cannot be adequately conveyed through text. As a result, current LVLMs often prioritize vision-to-language alignment while potentially overlooking fine-grained visual information. While some prior works have explored autoregressive image generation, effectively leveraging autoregressive visual supervision to enhance image understanding remains an open challenge. In this paper, we introduce Autoregressive Semantic Visual Reconstruction (ASVR), which enables joint learning of visual and textual modalities within a unified autoregressive framework. We show that autoregressively reconstructing the raw visual appearance of images does not enhance and may even impair multimodal understanding. In contrast, autoregressively reconstructing the semantic representation of images consistently improves comprehension. Notably, we find that even when models are given continuous image features as input, they can effectively reconstruct discrete semantic tokens, resulting in stable and consistent improvements across a wide range of multimodal understanding benchmarks. Our approach delivers significant performance gains across varying data scales (556k-2M) and types of LLM bacbones. Specifically, ASVR improves LLaVA-1.5 by 5% in average scores across 14 multimodal benchmarks. The code is available at https://github.com/AlenjandroWang/ASVR.
>
---
#### [replaced 080] CAT: Circular-Convolutional Attention for Sub-Quadratic Transformers
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型复杂度高、难以扩展的问题。提出CAT方法，通过循环卷积降低计算复杂度至O(NlogN)，提升效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2504.06704v2](https://arxiv.org/pdf/2504.06704v2)**

> **作者:** Yoshihiro Yamada
>
> **备注:** Accepted as a poster at NeurIPS 2025
>
> **摘要:** Transformers have driven remarkable breakthroughs in natural language processing and computer vision, yet their standard attention mechanism still imposes O(N^2) complexity, hindering scalability to longer sequences. We introduce Circular-convolutional ATtention (CAT), a Fourier-based approach that efficiently applies circular convolutions to reduce complexity without sacrificing representational power. CAT achieves O(NlogN) computations, requires fewer learnable parameters by streamlining fully connected layers, and introduces no additional heavy operations, resulting in consistent accuracy improvements and about a 10% speedup in naive PyTorch implementations. Based on the Engineering-Isomorphic Transformers (EITs) framework, CAT's design not only offers practical efficiency and ease of implementation, but also provides insights to guide the development of future high-performance Transformer architectures. Finally, our ablation studies highlight the key conditions underlying CAT's success, shedding light on broader principles for scalable attention mechanisms.
>
---
#### [replaced 081] From Bench to Bedside: A Review of Clinical Trials in Drug Discovery and Development
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于综述任务，旨在总结临床试验在药物开发中的作用及挑战。论文分析了各阶段临床试验的特点，探讨了技术革新与未来发展方向。**

- **链接: [https://arxiv.org/pdf/2412.09378v3](https://arxiv.org/pdf/2412.09378v3)**

> **作者:** Tianyang Wang; Ming Liu; Benji Peng; Xinyuan Song; Charles Zhang; Xintian Sun; Qian Niu; Junyu Liu; Silin Chen; Keyu Chen; Ming Li; Pohsun Feng; Ziqian Bi; Yunze Wang; Yichao Zhang; Cheng Fei; Lawrence KQ Yan; Ziyuan Qin; Riyang Bao; Zekun Jiang
>
> **备注:** 11 pages
>
> **摘要:** Clinical trials are an indispensable part of the drug development process, bridging the gap between basic research and clinical application. During the development of new drugs, clinical trials are used not only to evaluate the safety and efficacy of the drug but also to explore its dosage, treatment regimens, and potential side effects. This review discusses the various stages of clinical trials, including Phase I (safety assessment), Phase II (preliminary efficacy evaluation), Phase III (large-scale validation), and Phase IV (post-marketing surveillance), highlighting the characteristics of each phase and their interrelationships. Additionally, the paper addresses the major challenges encountered in clinical trials, such as ethical issues, subject recruitment difficulties, diversity and representativeness concerns, and proposes strategies for overcoming these challenges. With the advancement of technology, innovative technologies such as artificial intelligence, big data, and digitalization are gradually transforming clinical trial design and implementation, improving trial efficiency and data quality. The article also looks forward to the future of clinical trials, particularly the impact of emerging therapies such as gene therapy and immunotherapy on trial design, as well as the importance of regulatory reforms and global collaboration. In conclusion, the core role of clinical trials in drug development will continue to drive the progress of innovative drug development and clinical treatment.
>
---
#### [replaced 082] FaithLens: Detecting and Explaining Faithfulness Hallucination
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FaithLens，用于检测并解释大模型中的忠实性幻觉问题。属于模型可信度任务，解决输出可靠性不足的问题，通过数据合成与优化提升检测效果和解释质量。**

- **链接: [https://arxiv.org/pdf/2512.20182v2](https://arxiv.org/pdf/2512.20182v2)**

> **作者:** Shuzheng Si; Qingyi Wang; Haozhe Zhao; Yuzhuo Bai; Guanqiao Chen; Kangyang Luo; Gang Chen; Fanchao Qi; Minjia Zhang; Baobao Chang; Maosong Sun
>
> **摘要:** Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce FaithLens, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness.
>
---
#### [replaced 083] MemeMind: A Large-Scale Multimodal Dataset with Chain-of-Thought Reasoning for Harmful Meme Detection
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于有害表情包检测任务，旨在解决隐含有害内容识别困难的问题。构建了MemeMind数据集并提出MemeGuard模型，提升检测准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2506.18919v2](https://arxiv.org/pdf/2506.18919v2)**

> **作者:** Hexiang Gu; Qifan Yu; Yuan Liu; Zikang Li; Saihui Hou; Jian Zhao; Zhaofeng He
>
> **摘要:** As a multimodal medium combining images and text, memes frequently convey implicit harmful content through metaphors and humor, rendering the detection of harmful memes a complex and challenging task. Although recent studies have made progress in detection accuracy and interpretability, large-scale, high-quality datasets for harmful memes remain scarce, and current methods still struggle to capture implicit risks and nuanced semantics. Thus, we construct MemeMind, a large-scale harmful meme dataset. Aligned with the international standards and the context of internet, MemeMind provides detailed Chain-of-Thought (CoT) reasoning annotations to support fine-grained analysis of implicit intentions in memes. Based on this dataset, we further propose MemeGuard, a reasoning-oriented multimodal detection model that significantly improves both the accuracy of harmful meme detection and the interpretability of model decisions. Extensive experimental results demonstrate that MemeGuard outperforms existing state-of-the-art methods on the MemeMind dataset, establishing a solid foundation for future research in harmful meme detection.
>
---
#### [replaced 084] A Multi-Memory Segment System for Generating High-Quality Long-Term Memory Content in Agents
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于智能体记忆生成任务，旨在解决现有方法生成的长时记忆内容质量低的问题。提出多记忆段系统（MMS），通过分段处理提升记忆质量和检索效果。**

- **链接: [https://arxiv.org/pdf/2508.15294v3](https://arxiv.org/pdf/2508.15294v3)**

> **作者:** Gaoke Zhang; Bo Wang; Yunlong Ma; Dongming Zhao; Zifei Yu
>
> **摘要:** In the current field of agent memory, extensive explorations have been conducted in the area of memory retrieval, yet few studies have focused on exploring the memory content. Most research simply stores summarized versions of historical dialogues, as exemplified by methods like A-MEM and MemoryBank. However, when humans form long-term memories, the process involves multi-dimensional and multi-component generation, rather than merely creating simple summaries. The low-quality memory content generated by existing methods can adversely affect recall performance and response quality. In order to better construct high-quality long-term memory content, we have designed a multi-memory segment system (MMS) inspired by cognitive psychology theory. The system processes short-term memory into multiple long-term memory segments, and constructs retrieval memory units and contextual memory units based on these segments, with a one-to-one correspondence between the two. During the retrieval phase, MMS will match the most relevant retrieval memory units based on the user's query. Then, the corresponding contextual memory units is obtained as the context for the response stage to enhance knowledge, thereby effectively utilizing historical data. We conducted experiments on the LoCoMo dataset and further performed ablation experiments, experiments on the robustness regarding the number of input memories, and overhead experiments, which demonstrated the effectiveness and practical value of our method.
>
---
#### [replaced 085] Vision-Language Reasoning for Geolocalization: A Reinforcement Learning Approach
- **分类: cs.CL**

- **简介: 该论文属于图像地理定位任务，解决传统方法依赖合成标注或外部检索的问题。提出Geo-R框架，通过强化学习和结构化推理提升定位精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.00388v2](https://arxiv.org/pdf/2601.00388v2)**

> **作者:** Biao Wu; Meng Fang; Ling Chen; Ke Xu; Tao Cheng; Jun Wang
>
> **备注:** Accepted to AAAI 2026. Project Page: https://github.com/aialt/geo-r
>
> **摘要:** Recent advances in vision-language models have opened up new possibilities for reasoning-driven image geolocalization. However, existing approaches often rely on synthetic reasoning annotations or external image retrieval, which can limit interpretability and generalizability. In this paper, we present Geo-R, a retrieval-free framework that uncovers structured reasoning paths from existing ground-truth coordinates and optimizes geolocation accuracy via reinforcement learning. We propose the Chain of Region, a rule-based hierarchical reasoning paradigm that generates precise, interpretable supervision by mapping GPS coordinates to geographic entities (e.g., country, province, city) without relying on model-generated or synthetic labels. Building on this, we introduce a lightweight reinforcement learning strategy with coordinate-aligned rewards based on Haversine distance, enabling the model to refine predictions through spatially meaningful feedback. Our approach bridges structured geographic reasoning with direct spatial supervision, yielding improved localization accuracy, stronger generalization, and more transparent inference. Experimental results across multiple benchmarks confirm the effectiveness of Geo-R, establishing a new retrieval-free paradigm for scalable and interpretable image geolocalization. To facilitate further research and ensure reproducibility, both the model and code will be made publicly available.
>
---
#### [replaced 086] MedKGI: Iterative Differential Diagnosis with Medical Knowledge Graphs and Information-Guided Inquiring
- **分类: cs.CL**

- **简介: 该论文属于医学诊断任务，旨在解决LLM在临床诊断中的幻觉、低效提问和对话不连贯问题。提出MedKGI框架，结合医学知识图谱和信息增益选择问题，提升诊断准确性和效率。**

- **链接: [https://arxiv.org/pdf/2512.24181v2](https://arxiv.org/pdf/2512.24181v2)**

> **作者:** Qipeng Wang; Rui Sheng; Yafei Li; Huamin Qu; Yushi Sun; Min Zhu
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have demonstrated significant promise in clinical diagnosis. However, current models struggle to emulate the iterative, diagnostic hypothesis-driven reasoning of real clinical scenarios. Specifically, current LLMs suffer from three critical limitations: (1) generating hallucinated medical content due to weak grounding in verified knowledge, (2) asking redundant or inefficient questions rather than discriminative ones that hinder diagnostic progress, and (3) losing coherence over multi-turn dialogues, leading to contradictory or inconsistent conclusions. To address these challenges, we propose MedKGI, a diagnostic framework grounded in clinical practices. MedKGI integrates a medical knowledge graph (KG) to constrain reasoning to validated medical ontologies, selects questions based on information gain to maximize diagnostic efficiency, and adopts an OSCE-format structured state to maintain consistent evidence tracking across turns. Experiments on clinical benchmarks show that MedKGI outperforms strong LLM baselines in both diagnostic accuracy and inquiry efficiency, improving dialogue efficiency by 30% on average while maintaining state-of-the-art accuracy.
>
---
#### [replaced 087] Polarity Detection of Sustainable Development Goals in News Text
- **分类: cs.CL; cs.AI; cs.DL**

- **简介: 该论文提出SDG极性检测任务，旨在判断文本对可持续发展目标的影响方向。研究构建了SDG-POD数据集，并验证了大模型的性能。**

- **链接: [https://arxiv.org/pdf/2509.19833v3](https://arxiv.org/pdf/2509.19833v3)**

> **作者:** Andrea Cadeddu; Alessandro Chessa; Vincenzo De Leo; Gianni Fenu; Francesco Osborne; Diego Reforgiato Recupero; Angelo Salatino; Luca Secchi
>
> **备注:** Updated as one author was mispelled
>
> **摘要:** The United Nations' Sustainable Development Goals (SDGs) provide a globally recognised framework for addressing critical societal, environmental, and economic challenges. Recent developments in natural language processing (NLP) and large language models (LLMs) have facilitated the automatic classification of textual data according to their relevance to specific SDGs. Nevertheless, in many applications, it is equally important to determine the directionality of this relevance; that is, to assess whether the described impact is positive, neutral, or negative. To tackle this challenge, we propose the novel task of SDG polarity detection, which assesses whether a text segment indicates progress toward a specific SDG or conveys an intention to achieve such progress. To support research in this area, we introduce SDG-POD, a benchmark dataset designed specifically for this task, combining original and synthetically generated data. We perform a comprehensive evaluation using six state-of-the-art large LLMs, considering both zero-shot and fine-tuned configurations. Our results suggest that the task remains challenging for the current generation of LLMs. Nevertheless, some fine-tuned models, particularly QWQ-32B, achieve good performance, especially on specific Sustainable Development Goals such as SDG-9 (Industry, Innovation and Infrastructure), SDG-12 (Responsible Consumption and Production), and SDG-15 (Life on Land). Furthermore, we demonstrate that augmenting the fine-tuning dataset with synthetically generated examples yields improved model performance on this task. This result highlights the effectiveness of data enrichment techniques in addressing the challenges of this resource-constrained domain. This work advances the methodological toolkit for sustainability monitoring and provides actionable insights into the development of efficient, high-performing polarity detection systems.
>
---
#### [replaced 088] Diagnosing and Mitigating Semantic Inconsistencies in Wikidata's Classification Hierarchy
- **分类: cs.CL**

- **简介: 该论文属于知识图谱质量评估任务，旨在解决Wikidata分类层级中的语义不一致问题，通过验证方法识别错误分类和冗余连接，并提出评估标准与检查系统。**

- **链接: [https://arxiv.org/pdf/2511.04926v2](https://arxiv.org/pdf/2511.04926v2)**

> **作者:** Shixiong Zhao; Hideaki Takeda
>
> **摘要:** Wikidata is currently the largest open knowledge graph on the web, encompassing over 120 million entities. It integrates data from various domain-specific databases and imports a substantial amount of content from Wikipedia, while also allowing users to freely edit its content. This openness has positioned Wikidata as a central resource in knowledge graph research and has enabled convenient knowledge access for users worldwide. However, its relatively loose editorial policy has also led to a degree of taxonomic inconsistency. Building on prior work, this study proposes and applies a novel validation method to confirm the presence of classification errors, over-generalized subclass links, and redundant connections in specific domains of Wikidata. We further introduce a new evaluation criterion for determining whether such issues warrant correction and develop a system that allows users to inspect the taxonomic relationships of arbitrary Wikidata entities-leveraging the platform's crowdsourced nature to its full potential.
>
---
#### [replaced 089] Membership Inference Attacks on LLM-based Recommender Systems
- **分类: cs.IR; cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于隐私安全任务，研究LLM推荐系统中的成员推理攻击问题。通过设计多种攻击方法，验证了系统提示中用户历史交互信息可能被泄露，分析了影响攻击效果的因素。**

- **链接: [https://arxiv.org/pdf/2508.18665v4](https://arxiv.org/pdf/2508.18665v4)**

> **作者:** Jiajie He; Min-Chun Chen; Xintong Chen; Xinyang Fang; Yuechun Gu; Keke Chen
>
> **备注:** This is paper is under review WWW 2026
>
> **摘要:** Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.
>
---
#### [replaced 090] Multimodal Fact-Checking: An Agent-based Approach
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态事实核查任务，旨在解决现有系统在推理和证据利用上的不足。作者构建了RW-Post数据集并提出AgentFact框架，提升核查的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.22933v3](https://arxiv.org/pdf/2512.22933v3)**

> **作者:** Danni Xu; Shaojing Fan; Harry Cheng; Mohan Kankanhalli
>
> **备注:** Code and dataset will be released at https://github.com/xudanni0927/AgentFact
>
> **摘要:** The rapid spread of multimodal misinformation poses a growing challenge for automated fact-checking systems. Existing approaches, including large vision language models (LVLMs) and deep multimodal fusion methods, often fall short due to limited reasoning and shallow evidence utilization. A key bottleneck is the lack of dedicated datasets that provide complete real-world multimodal misinformation instances accompanied by annotated reasoning processes and verifiable evidence. To address this limitation, we introduce RW-Post, a high-quality and explainable dataset for real-world multimodal fact-checking. RW-Post aligns real-world multimodal claims with their original social media posts, preserving the rich contextual information in which the claims are made. In addition, the dataset includes detailed reasoning and explicitly linked evidence, which are derived from human written fact-checking articles via a large language model assisted extraction pipeline, enabling comprehensive verification and explanation. Building upon RW-Post, we propose AgentFact, an agent-based multimodal fact-checking framework designed to emulate the human verification workflow. AgentFact consists of five specialized agents that collaboratively handle key fact-checking subtasks, including strategy planning, high-quality evidence retrieval, visual analysis, reasoning, and explanation generation. These agents are orchestrated through an iterative workflow that alternates between evidence searching and task-aware evidence filtering and reasoning, facilitating strategic decision-making and systematic evidence analysis. Extensive experimental results demonstrate that the synergy between RW-Post and AgentFact substantially improves both the accuracy and interpretability of multimodal fact-checking.
>
---
#### [replaced 091] nvBench 2.0: Resolving Ambiguity in Text-to-Visualization through Stepwise Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本到可视化任务，解决模糊查询理解问题。通过构建基准数据集nvBench 2.0和提出Step-Text2Vis模型，提升LLMs在模糊场景下的表现。**

- **链接: [https://arxiv.org/pdf/2503.12880v3](https://arxiv.org/pdf/2503.12880v3)**

> **作者:** Tianqi Luo; Chuhan Huang; Leixian Shen; Boyan Li; Shuyu Shen; Wei Zeng; Nan Tang; Yuyu Luo
>
> **摘要:** Text-to-Visualization (Text2VIS) enables users to create visualizations from natural language queries, making data insights more accessible. However, Text2VIS faces challenges in interpreting ambiguous queries, as users often express their visualization needs in imprecise language. To address this challenge, we introduce nBench 2.0, a new benchmark designed to evaluate Text2VIS systems in scenarios involving ambiguous queries. nvBench 2.0 includes 7,878 natural language queries and 24,076 corresponding visualizations, derived from 780 tables across 153 domains. It is built using a controlled ambiguity-injection pipeline that generates ambiguous queries through a reverse-generation workflow. By starting with unambiguous seed visualizations and selectively injecting ambiguities, the pipeline yields multiple valid interpretations for each query, with each ambiguous query traceable to its corresponding visualization through step-wise reasoning paths. We evaluate various Large Language Models (LLMs) on their ability to perform ambiguous Text2VIS tasks using nBench 2.0. We also propose Step-Text2Vis, an LLM-based model trained on nvBench 2.0, which enhances performance in ambiguous scenarios through step-wise preference optimization. Our results show that Step-Text2Vis outperforms all baselines, setting a new state-of-the-art for ambiguous Text2VIS tasks. Our source code and data are available at https://nvbench2.github.io/
>
---
#### [replaced 092] Context-aware Decoding Reduces Hallucination in Query-focused Summarization
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于查询聚焦摘要任务，旨在减少大语言模型生成中的幻觉问题。通过改进解码方法，提升摘要的准确性和相关性。**

- **链接: [https://arxiv.org/pdf/2312.14335v3](https://arxiv.org/pdf/2312.14335v3)**

> **作者:** Zhichao Xu
>
> **备注:** technical report
>
> **摘要:** Query-focused summarization (QFS) aims to provide a summary of a single document/multi documents that can satisfy the information needs of a given query. It is useful for various real-world applications, such as abstractive snippet generation or more recent retrieval augmented generation (RAG). A prototypical QFS pipeline consists of a retriever (sparse or dense retrieval) and a generator (usually a large language model). However, applying large language models (LLM) potentially leads to hallucinations, especially when the evidence contradicts the prior belief of LLMs. There has been growing interest in developing new decoding methods to improve generation quality and reduce hallucination. In this work, we conduct a large-scale reproducibility study on one recently proposed decoding method\, -- \,Context-aware Decoding (CAD). In addition to replicating CAD's experiments on news summarization datasets, we include experiments on QFS datasets, and conduct more rigorous analysis on computational complexity and hyperparameter sensitivity. Experiments with eight different language models show that performance-wise, CAD improves QFS quality by (1) reducing factuality errors/hallucinations while (2) mostly retaining the match of lexical patterns, measured by ROUGE scores, while also at a cost of increased inference-time FLOPs and reduced decoding speed. The \href{https://github.com/zhichaoxu-shufe/context-aware-decoding-qfs}{code implementation} based on Huggingface Library is made available
>
---
#### [replaced 093] Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机器学习安全任务，解决连续数据删除中的模型性能下降问题。提出RCU方法，通过旋转空间控制未学习程度，减少累积误差。**

- **链接: [https://arxiv.org/pdf/2509.25743v3](https://arxiv.org/pdf/2509.25743v3)**

> **作者:** Xiang Zhang; Kun Wei; Xu Yang; Jiahua Li; Su Yan; Cheng Deng
>
> **摘要:** As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
>
---
#### [replaced 094] HaluMem: Evaluating Hallucinations in Memory Systems of Agents
- **分类: cs.CL**

- **简介: 该论文属于AI记忆系统研究，旨在解决记忆幻觉问题。提出HaluMem基准，通过三个任务评估记忆系统的幻觉行为，提升记忆可靠性。**

- **链接: [https://arxiv.org/pdf/2511.03506v3](https://arxiv.org/pdf/2511.03506v3)**

> **作者:** Ding Chen; Simin Niu; Kehang Li; Peng Liu; Xiangping Zheng; Bo Tang; Xinchi Li; Feiyu Xiong; Zhiyu Li
>
> **摘要:** Memory systems are key components that enable AI systems such as LLMs and AI agents to achieve long-term learning and sustained interaction. However, during memory storage and retrieval, these systems frequently exhibit memory hallucinations, including fabrication, errors, conflicts, and omissions. Existing evaluations of memory hallucinations are primarily end-to-end question answering, which makes it difficult to localize the operational stage within the memory system where hallucinations arise. To address this, we introduce the Hallucination in Memory Benchmark (HaluMem), the first operation level hallucination evaluation benchmark tailored to memory systems. HaluMem defines three evaluation tasks (memory extraction, memory updating, and memory question answering) to comprehensively reveal hallucination behaviors across different operational stages of interaction. To support evaluation, we construct user-centric, multi-turn human-AI interaction datasets, HaluMem-Medium and HaluMem-Long. Both include about 15k memory points and 3.5k multi-type questions. The average dialogue length per user reaches 1.5k and 2.6k turns, with context lengths exceeding 1M tokens, enabling evaluation of hallucinations across different context scales and task complexities. Empirical studies based on HaluMem show that existing memory systems tend to generate and accumulate hallucinations during the extraction and updating stages, which subsequently propagate errors to the question answering stage. Future research should focus on developing interpretable and constrained memory operation mechanisms that systematically suppress hallucinations and improve memory reliability.
>
---
#### [replaced 095] CMDAR: A Chinese Multi-scene Dynamic Audio Reasoning Benchmark with Diverse Challenges
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文提出CMDAR基准，用于评估AI在多场景动态音频推理任务中的能力，解决现有数据单一、场景简单的问题。**

- **链接: [https://arxiv.org/pdf/2509.22461v2](https://arxiv.org/pdf/2509.22461v2)**

> **作者:** Hui Li; Changhao Jiang; Hongyu Wang; Ming Zhang; Jiajun Sun; Zhixiong Yang; Yifei Cao; Shihan Dou; Xiaoran Fan; Baoyu Fan; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** The ability to reason from audio, including speech, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and English audio data and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce CMDAR, a Chinese benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. CMDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on CMDAR and observe that they exhibit limitations in complex reasoning tasks. In CMDAR-main, Qwen2.5-Omni achieves 76.67% accuracy, whereas GPT-4o Audio reaches 68.47%. However, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice with multiple audios and open-ended tasks. And we provide detail analysis corresponding suggestions for the future development of large audio language models.
>
---
#### [replaced 096] GRACE: Discriminator-Guided Chain-of-Thought Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GRACE方法，用于提升语言模型在多步推理任务中的正确性。针对模型易生成错误推理步骤的问题，GRACE通过判别器引导解码过程，提高中间步骤的准确性。**

- **链接: [https://arxiv.org/pdf/2305.14934v3](https://arxiv.org/pdf/2305.14934v3)**

> **作者:** Muhammad Khalifa; Lajanugen Logeswaran; Moontae Lee; Honglak Lee; Lu Wang
>
> **备注:** Fixed typos
>
> **摘要:** In the context of multi-step reasoning, e.g., with chain-of-thought, language models (LMs) can easily assign a high likelihood to incorrect steps. As a result, decoding strategies that optimize for solution likelihood often yield incorrect solutions. To address this issue, we propose Guiding chain-of-thought ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that steers the decoding process towards producing correct reasoning steps. GRACE employs a step-level verifier or discriminator trained with a contrastive loss over correct and incorrect steps, which is used during decoding to score next-step candidates based on their correctness. Importantly, GRACE only requires sampling from the LM, without the need for LM training or fine-tuning. Using models from FLAN-T5 and LLaMA families, we evaluate GRACE over four math and two symbolic reasoning tasks, where it exhibits substantial performance gains compared to greedy decoding, verifiers, and self-consistency in most settings. When further combined with self-consistency, GRACE outperforms all the baselines by sizeable margins. Human and LLM evaluations over GSM8K show that GRACE not only improves the final answer accuracy but also the correctness of the intermediate reasoning. Our implementation can be accessed at https://github.com/mukhal/grace.
>
---
#### [replaced 097] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models
- **分类: cs.CL**

- **简介: 该论文研究视觉语言模型在实体知识提取中的性能差异，探讨图像与文本信息处理的效率问题。通过构建数据集PopVQA，分析模型在不同模态下的表现，揭示其内部机制及改进方向。**

- **链接: [https://arxiv.org/pdf/2412.14133v3](https://arxiv.org/pdf/2412.14133v3)**

> **作者:** Ido Cohen; Daniela Gottesman; Mor Geva; Raja Giryes
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Vision-language models (VLMs) excel at extracting and reasoning about information from images. Yet, their capacity to leverage internal knowledge about specific entities remains underexplored. This work investigates the disparity in model performance when answering factual questions about an entity described in text versus depicted in an image. Our results reveal a significant accuracy drop - reaching 18% for some models - when the entity is presented visually instead of textually. To study this gap we present PopVQA, a dataset which allows separating entity recognition and question answering, and use it to benchmark several models. We hypothesize that this decline arises from limitations in how information flows from image tokens to query tokens. Thus, we use mechanistic interpretability tools to reveal that, although image tokens are preprocessed by the vision encoder, meaningful information flow from these tokens occurs only in the much deeper layers. Furthermore, critical image processing happens in the language model's middle layers, allowing few layers for consecutive reasoning, highlighting a potential inefficiency in how the model utilizes its layers for reasoning. These insights shed light on the internal mechanics of VLMs and offer pathways for enhancing their reasoning capabilities. PopVQA can be found at https://huggingface.co/datasets/idoco/PopVQA.
>
---
#### [replaced 098] LTLBench: Towards Benchmarks for Evaluating Temporal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时序推理任务，旨在评估大语言模型的时序理解能力。通过引入线性时序逻辑，构建LTLBench数据集，并测试不同模型的表现及问题复杂度的影响。**

- **链接: [https://arxiv.org/pdf/2407.05434v3](https://arxiv.org/pdf/2407.05434v3)**

> **作者:** Weizhi Tang; Kwabena Nuamah; Vaishak Belle
>
> **摘要:** Temporal Reasoning (TR) is a critical ability for LLMs to understand and reason over temporal information and relationships between events. To study the TR ability in LLMs, prior works provide different ways for evaluating various aspects of TR ability. In this work, we propose an alternative perspective for evaluating TR ability by leveraging Linear Temporal Logic (LTL), and develop a pipeline to automatically synthesize challenges for assessing the TR ability of LLMs. Based on this pipeline, we construct a dataset, namely LTLBench, consisting of $2000$ TR challenges, and benchmark 12 LLMs across 5 different methods. Furthermore, we conduct additional experiments to investigate the impact of increasing the number of formula operators and events on both LLM performance and the complexity of TR problems. We also perform qualitative analyses of their reasoning processes and the effects of varying the number of events and formula operators, which reveal 3 main issues in their temporal reasoning processes and the unexpected performance changes observed as problem complexity increases. We expect this work to provide valuable insights into the TR ability of LLMs.
>
---
#### [replaced 099] KVCrush: Key value cache size-reduction using similarity in head-behaviour
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型优化任务，旨在解决KV缓存内存占用过高的问题。通过提出KVCrush技术，实现更小的内存占用同时保持模型精度。**

- **链接: [https://arxiv.org/pdf/2503.00022v2](https://arxiv.org/pdf/2503.00022v2)**

> **作者:** Gopi Krishna Jha; Sameh Gobriel; Liubov Talamanova; Nilesh Jain
>
> **摘要:** Key-value (KV) caching has emerged as a crucial optimization technique for accelerating inference in large language models (LLMs). By allowing the attention operation to scale linearly rather than quadratically with the total sequence length, KV caching significantly enhances generation throughput. However, due to large context lengths in the modern LLMs, the memory footprint of the KV is a huge bottleneck for model deployment directly impacting the model's batch size, hindering its ability to deliver high-throughput. Existing research addresses this challenge using several techniques, such as discarding low-attention tokens, quantization, and matrix approximation which typically lead to a negative impact on the model accuracy. In this paper, We propose KVCrush technology which can be combined with many KV compression technologies to improve the model accuracy at a much smaller memory. KVCrush provides an alternate representation scheme for key-value states, along with a low-overhead token pruning algorithm that accounts for the token distribution in the KV cache, which in turn allows for a a smaller footprint while maintaining the accuracy of the model. Based on our results, KVCrush reduces LongBench KV Cache size by 4x with less than 1% accuracy drop and achieves state-of-the-art average accuracy with minimal overhead, incurring less than 0.5% total inference latency. KVCrush not only outperforms the accuracy of state-of-the-art importance-based token retention schemes but is also compatible with typical practical LLM deployments using KV cache paging schemes such as vLLM and mixed precision quantization.
>
---
#### [replaced 100] MR-Align: Meta-Reasoning Informed Factuality Alignment for Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于事实性问答任务，解决大推理模型在证据依赖问题上事实性不足的问题。提出MR-ALIGN框架，通过优化推理过程提升事实准确性。**

- **链接: [https://arxiv.org/pdf/2510.24794v2](https://arxiv.org/pdf/2510.24794v2)**

> **作者:** Xinming Wang; Jian Xu; Bin Yu; Sheng Lian; Hongzhu Yi; Yi Chen; Yingjian Zhu; Boran Wang; Hongming Yang; Han Hu; Xu-Yao Zhang; Cheng-Lin Liu
>
> **备注:** Preprint
>
> **摘要:** Large reasoning models (LRMs) show strong capabilities in complex reasoning, yet their marginal gains on evidence-dependent factual questions are limited. We find this limitation is partially attributable to a reasoning-answer hit gap, where the model identifies the correct facts during reasoning but fails to incorporate them into the final response, thereby reducing factual fidelity. To address this issue, we propose MR-ALIGN, a Meta-Reasoning informed alignment framework that enhances factuality without relying on external verifiers. MR-ALIGN quantifies state transition probabilities along the model's thinking process and constructs a transition-aware implicit reward that reinforces beneficial reasoning patterns while suppressing defective ones at the atomic thinking segments. This re-weighting reshapes token-level signals into probability-aware segment scores, encouraging coherent reasoning trajectories that are more conducive to factual correctness. Empirical evaluations across four factual QA datasets and one long-form factuality benchmark show that MR-ALIGN consistently improves accuracy and truthfulness while reducing misleading reasoning. These results highlight that aligning the reasoning process itself, rather than merely the outputs, is pivotal for advancing factuality in LRMs.
>
---
#### [replaced 101] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的推理能力。通过强化学习，无需人工标注即可激励模型发展出高级推理模式，解决传统方法依赖大量标注数据的问题。**

- **链接: [https://arxiv.org/pdf/2501.12948v2](https://arxiv.org/pdf/2501.12948v2)**

> **作者:** DeepSeek-AI; Daya Guo; Dejian Yang; Haowei Zhang; Junxiao Song; Peiyi Wang; Qihao Zhu; Runxin Xu; Ruoyu Zhang; Shirong Ma; Xiao Bi; Xiaokang Zhang; Xingkai Yu; Yu Wu; Z. F. Wu; Zhibin Gou; Zhihong Shao; Zhuoshu Li; Ziyi Gao; Aixin Liu; Bing Xue; Bingxuan Wang; Bochao Wu; Bei Feng; Chengda Lu; Chenggang Zhao; Chengqi Deng; Chenyu Zhang; Chong Ruan; Damai Dai; Deli Chen; Dongjie Ji; Erhang Li; Fangyun Lin; Fucong Dai; Fuli Luo; Guangbo Hao; Guanting Chen; Guowei Li; H. Zhang; Han Bao; Hanwei Xu; Haocheng Wang; Honghui Ding; Huajian Xin; Huazuo Gao; Hui Qu; Hui Li; Jianzhong Guo; Jiashi Li; Jiawei Wang; Jingchang Chen; Jingyang Yuan; Junjie Qiu; Junlong Li; J. L. Cai; Jiaqi Ni; Jian Liang; Jin Chen; Kai Dong; Kai Hu; Kaige Gao; Kang Guan; Kexin Huang; Kuai Yu; Lean Wang; Lecong Zhang; Liang Zhao; Litong Wang; Liyue Zhang; Lei Xu; Leyi Xia; Mingchuan Zhang; Minghua Zhang; Minghui Tang; Meng Li; Miaojun Wang; Mingming Li; Ning Tian; Panpan Huang; Peng Zhang; Qiancheng Wang; Qinyu Chen; Qiushi Du; Ruiqi Ge; Ruisong Zhang; Ruizhe Pan; Runji Wang; R. J. Chen; R. L. Jin; Ruyi Chen; Shanghao Lu; Shangyan Zhou; Shanhuang Chen; Shengfeng Ye; Shiyu Wang; Shuiping Yu; Shunfeng Zhou; Shuting Pan; S. S. Li; Shuang Zhou; Shaoqing Wu; Shengfeng Ye; Tao Yun; Tian Pei; Tianyu Sun; T. Wang; Wangding Zeng; Wanjia Zhao; Wen Liu; Wenfeng Liang; Wenjun Gao; Wenqin Yu; Wentao Zhang; W. L. Xiao; Wei An; Xiaodong Liu; Xiaohan Wang; Xiaokang Chen; Xiaotao Nie; Xin Cheng; Xin Liu; Xin Xie; Xingchao Liu; Xinyu Yang; Xinyuan Li; Xuecheng Su; Xuheng Lin; X. Q. Li; Xiangyue Jin; Xiaojin Shen; Xiaosha Chen; Xiaowen Sun; Xiaoxiang Wang; Xinnan Song; Xinyi Zhou; Xianzu Wang; Xinxia Shan; Y. K. Li; Y. Q. Wang; Y. X. Wei; Yang Zhang; Yanhong Xu; Yao Li; Yao Zhao; Yaofeng Sun; Yaohui Wang; Yi Yu; Yichao Zhang; Yifan Shi; Yiliang Xiong; Ying He; Yishi Piao; Yisong Wang; Yixuan Tan; Yiyang Ma; Yiyuan Liu; Yongqiang Guo; Yuan Ou; Yuduan Wang; Yue Gong; Yuheng Zou; Yujia He; Yunfan Xiong; Yuxiang Luo; Yuxiang You; Yuxuan Liu; Yuyang Zhou; Y. X. Zhu; Yanhong Xu; Yanping Huang; Yaohui Li; Yi Zheng; Yuchen Zhu; Yunxian Ma; Ying Tang; Yukun Zha; Yuting Yan; Z. Z. Ren; Zehui Ren; Zhangli Sha; Zhe Fu; Zhean Xu; Zhenda Xie; Zhengyan Zhang; Zhewen Hao; Zhicheng Ma; Zhigang Yan; Zhiyu Wu; Zihui Gu; Zijia Zhu; Zijun Liu; Zilin Li; Ziwei Xie; Ziyang Song; Zizheng Pan; Zhen Huang; Zhipeng Xu; Zhongyu Zhang; Zhen Zhang
>
> **摘要:** General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.
>
---
#### [replaced 102] Language Model Distillation: A Temporal Difference Imitation Learning Perspective
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型计算成本高的问题。通过引入基于时序差分的蒸馏框架，利用教师模型的分布稀疏性，提升小型模型性能。**

- **链接: [https://arxiv.org/pdf/2505.20335v4](https://arxiv.org/pdf/2505.20335v4)**

> **作者:** Zishun Yu; Shangzhe Li; Xinhua Zhang
>
> **备注:** AAAI 2026; Code available at: https://github.com/TobyLeelsz/Bellman-Distillation
>
> **摘要:** Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements.
>
---
#### [replaced 103] BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache
- **分类: cs.AR; cs.AI; cs.CL; cs.PF**

- **简介: 该论文属于大模型推理任务，解决长上下文LLM解码效率低的问题。通过结合CUDA与Tensor Core，优化低比特KV缓存解码，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2503.18773v3](https://arxiv.org/pdf/2503.18773v3)**

> **作者:** Dayou Du; Shijie Cao; Jianyi Cheng; Luo Mai; Ting Cao; Mao Yang
>
> **摘要:** The growth of long-context Large Language Models (LLMs) significantly increases memory and bandwidth pressure during autoregressive decoding due to the expanding Key-Value (KV) cache. While accuracy-preserving KV-cache quantization (e.g., 4-bit or 2-bit) reduces memory footprint, existing systems decode inefficiently by relying solely on CUDA cores, underutilizing Tensor Cores-the dominant compute resource on GPUs. We present BitDecoding, the first inference system to efficiently decode low-bit KV caches by cooperatively leveraging CUDA cores and Tensor Cores. BitDecoding smartly induces Tensor-Core-friendly layouts, introduces warp-level dequantization parallelism, and provides unified system support through query transformation, high-performance tensor- and channel-wise quantization, and a software-pipelined dequantization kernel enabling mixed-precision execution. Architecture-aware optimizations further leverage Hopper's warpgroup tensor instructions and Blackwell's NVFP4 (MXFP4) tensor formats. Evaluated on Blackwell, Hopper, and Ampere GPUs, BitDecoding achieves an average 7.5x decoding speedup over FP16 FlashDecoding-v2, up to 8.6x on Blackwell with NVFP4, and up to 4.3x over state-of-the-art approaches. On LLaMA-3.1-8B with a 128K context, BitDecoding reduces single-batch decoding latency by 3x. BitDecoding is open-sourced at https://github.com/OpenBitSys/BitDecoding.
>
---
