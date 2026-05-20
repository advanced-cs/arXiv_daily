# 自然语言处理 cs.CL

- **最新发布 103 篇**

- **更新 79 篇**

## 最新发布

#### [new 001] Less Back-and-Forth: A Comparative Study of Structured Prompting
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理领域，研究结构化提示对大模型响应质量的影响。旨在解决低质量回答和用户交互过多的问题，通过对比三种提示方式验证结构化提示的有效性。**

- **链接: [https://arxiv.org/pdf/2605.20149](https://arxiv.org/pdf/2605.20149)**

> **作者:** Saurav Ghosh; Gabriella Polach; Abdou Sow
>
> **备注:** 7 pages, 2 figures, 6 tables
>
> **摘要:** Large language models (LLMs) are widely used for open-ended tasks, but underspecified prompts can lead to low-quality answers and additional interaction. This paper studies whether structured prompt design improves response quality while reducing user effort. We compare three prompt conditions: a raw prompt, a checklist-improved prompt, and a clarifying-question prompt. We evaluate these conditions across four task types--summarization, planning, explanation, and coding--using three LLM systems: ChatGPT, Claude, and Grok. Each output is scored with a unified rubric covering task completion, correctness, compliance, and clarity. Checklist-improved prompts achieved the highest mean rubric score, 7.50 out of 8, compared with 5.67 for raw prompts and 6.67 for clarifying-question prompts. Checklist prompts also produced the best quality-effort tradeoff, using fewer average tokens than both raw and clarifying prompts. These results suggest that a simple prompt checklist can improve LLM responses while reducing unnecessary interaction.
>
---
#### [new 002] PromptRad: Knowledge-Enhanced Multi-Label Prompt-Tuning for Low-Resource Radiology Report Labeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分类任务，解决低资源下放射报告标签生成问题。提出PromptRad方法，通过知识增强的多标签提示调优，提升分类效果。**

- **链接: [https://arxiv.org/pdf/2605.20052](https://arxiv.org/pdf/2605.20052)**

> **作者:** Ying-Jia Lin; Tzu-Chin Lo; Ping-Chien Li; Chi-Tung Cheng; Chien-Hung Liao; Hung-Yu Kao
>
> **备注:** BioNLP 2026 @ ACL
>
> **摘要:** Automatic report labeling facilitates the identification of clinical findings from unstructured text and enables large-scale annotation for medical imaging research. Existing rule-based labelers struggle with the diverse descriptions in clinical reports, while fine-tuning pre-trained language models (PLMs) requires large amounts of labeled data that are often unavailable in clinical settings. In this paper, we propose PromptRad, a knowledge-enhanced multi-label \textbf{prompt}-tuning approach for \textbf{rad}iology report labeling under low-resource settings. PromptRad reformulates multi-label classification as masked language modeling and incorporates synonyms from the UMLS Metathesaurus into a multi-word verbalizer to enrich category representations. By fine-tuning the PLM without additional classification layers, PromptRad requires substantially less labeled data than conventional fine-tuning. Experiments on liver CT reports show that PromptRad outperforms dictionary-based and fine-tuning baselines with only 32 labeled training examples, and achieves competitive performance with GPT-4 despite using a much smaller model. Further analysis demonstrates that PromptRad captures complex negation patterns more effectively than existing methods, making it a promising solution for report labeling in data-scarce clinical scenarios. Our code is available at this https URL.
>
---
#### [new 003] Benchmarking Commercial ASR Systems on Code-Switching Speech: Arabic, Persian, and German
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在评估商业ASR系统在代码切换语音中的表现。通过构建多语言数据集并使用WER和BERTScore进行评测，揭示了系统在真实多语言场景下的性能差异。**

- **链接: [https://arxiv.org/pdf/2605.19069](https://arxiv.org/pdf/2605.19069)**

> **作者:** Sajjad Abdoli; Ghassan Al-Sumaidaee; Clayton W. Taylor; Ahmad; ElShiekh; Ahmed Rashad
>
> **摘要:** Code-switching -- the natural alternation between two languages within a single utterance -- represents one of the most challenging and under-studied conditions for automatic speech recognition (ASR). Existing commercial ASR benchmarks predominantly evaluate clean, monolingual audio and report a single Word Error Rate (WER) figure that tells practitioners little about real-world multilingual performance. We present a benchmark evaluating five commercial ASR providers across four language pairs: Egyptian Arabic--English, Saudi Arabic (Najdi/Hijazi)--English, Persian (Farsi)--English, and German--English. Each dataset comprises 300 samples selected by a two-stage pipeline: a heuristic filter scoring transcripts on five structural code-switching signals, followed by a GPT-4o and Gemini 1.5 Pro ensemble scoring candidates across six linguistic dimensions. This pipeline reduces LLM scoring costs by approximately 91\% relative to exhaustive scoring. We evaluate the systems on both WER and BERTScore, arguing that BERTScore is a more reliable metric for Arabic and Persian pairs where transliteration variance causes WER to penalise semantically correct transcriptions. ElevenLabs Scribe v2 achieves the lowest WER across all four language pairs (13.2% overall; 13.1% on Egyptian Arabic) and leads on BERTScore (0.936 overall). We further demonstrate that difficulty-stratified analysis reveals performance gaps masked by aggregate averages, and that BERT embedding projections confirm semantic proximity between reference and hypothesis despite surface-level script differences. The benchmarking dataset is publicly available at this https URL.
>
---
#### [new 004] m3BERT: A Modern, Multi-lingual, Matryoshka Bidirectional Encoder
- **分类: cs.CL**

- **简介: 该论文提出m3BERT，解决工业检索中模型适配资源约束的问题。通过多语言、多维度预训练，实现高效、灵活的嵌入模型。**

- **链接: [https://arxiv.org/pdf/2605.19568](https://arxiv.org/pdf/2605.19568)**

> **作者:** Yaoxiang Wang; Simiao Zuo; Qingguo Hu; Yucheng Ding; Yeyun Gong; Jian Jiao; Jinsong Su
>
> **备注:** KDD 2026
>
> **摘要:** Embedding models are pivotal in industrial information retrieval systems like search and advertising. However, existing pretrained models often exhibit fixed architectures and embedding dimensionalities, posing significant challenges when adapting them to diverse deployment scenarios with varying business-driven constraints. A common practice involves fine-tuning with partial parameter initialization from larger pretrained models for resource-constrained tasks. This method is often suboptimal as the misalignment between pretraining and downstream usage prevents full realization of pretraining benefits. To address this limitation, we introduce m3BERT: a Modern, Multi-lingual, Matryoshka Bidirectional Encoder, which features a novel pretraining strategy that jointly optimizes representations across both transformer layers and multiple embedding dimensions. This enables a single model to be tailored to varied resource and accuracy targets while maintaining consistency with pretraining. Incorporating recent architectural improvements, m3BERT uses a three-stage pretraining: monolingual pretraining, multilingual adaptation to serve diverse user bases, and crucial continual pretraining on a massive web domain corpus to enhance utility in commercial retrieval. m3BERT significantly outperforms state-of-the-art embedding models in Bing-Click, a large-scale industrial retrieval dataset, showcasing its practical versatility as an efficient foundation for resource-aware industrial retrieval systems. Further experiments on public datasets also confirm the general effectiveness of our multigranular Matryoshka pretraining strategy.
>
---
#### [new 005] Backtracking When It Strays: Mitigating Dual Exposure Biases in LLM Reasoning Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理蒸馏任务，旨在解决双暴露偏差问题。通过动态监控学生生成过程并及时回溯修正，提升推理性能。**

- **链接: [https://arxiv.org/pdf/2605.19433](https://arxiv.org/pdf/2605.19433)**

> **作者:** Bing Wang; Shaotian Yan; Chen Shen; kaiyuan liu; Sinan Fan; Ximing Li; Rui Miao; Xiaosong Yuan; Zhanming Shen; Jieping Ye
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in complex reasoning tasks via long chain-of-thought (CoT), yet their immense computational overhead hinders real-world deployment. LLM reasoning distillation addresses this by transferring reasoning capabilities from formidable teacher models to compact student models. However, existing distillation paradigms face a fundamental dilemma. Typical off-policy distillation strictly utilizes teacher-generated golden trajectories, suffering from an exposure bias due to the mismatch between training distributions and student-generated inference contexts, which leads to error cascades in long CoT reasoning. To address this, on-policy distillation allows students to explore their own trajectories, but we demonstrate that it inherently introduces a reciprocal reversed exposure bias: the teacher model also struggles to provide positive guidance when conditioned on student-generated sub-optimal contexts. To resolve this dual exposure biases problem, we propose Monitoring Trajectories and Backtracking when it strays (MOTAB), a new LLM reasoning distillation pipeline. Specifically, MOTAB dynamically monitors the student's on-policy generation against an adaptive safety boundary. When the generation strays and exceeds this threshold, MOTAB backtracks to the last safe state and leverages teacher intervention to correct the course. This approach inherently tolerates minor student errors to mitigate exposure bias, while preventing sub-optimal contexts to circumvent reversed exposure bias. Extensive experiments on the LIMO-v2 and AceReason datasets demonstrate that MOTAB effectively alleviates the dual exposure biases, yielding a roughly 3% average performance improvement in reasoning tasks.
>
---
#### [new 006] Synthesis and Evaluation of Long-term History-aware Medical Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗对话任务，旨在解决缺乏长期对话数据的问题。通过合成高质量医疗对话数据集，评估医疗AI的记忆与推理能力。**

- **链接: [https://arxiv.org/pdf/2605.19766](https://arxiv.org/pdf/2605.19766)**

> **作者:** Hebin Hu; Renke Dai; Ah-Hwee Tan; Yilin Kang
>
> **备注:** Accepted by AAMAS 2026
>
> **摘要:** An effective healthcare agent must be able to recall and reason over a patient's longitudinal medical history. However, the absence of datasets with realistic long-term dialogue timelines limits systematic evaluation. Real clinical text is constrained by privacy and ethics, while existing benchmarks focus on isolated interactions, failing to capture cross-session reasoning. We introduce a framework for synthesizing high-quality, long-term medical dialogues with LLMs. Our approach entails a knowledge-guided decomposition into three stages: constructing synthetic patient profiles with diverse disease and complication trajectories, generating multi-turn dialogues per encounter, and integrating them into a coherent longitudinal history dataset, MediLongChat. We establish three benchmark tasks-In-dialogue Reasoning, Cross-dialogue Reasoning, and Synthesis Reasoning-to evaluate the memory capabilities of healthcare agents. To assess data quality, we introduce a multi-dimensional evaluation framework combining vector-based metrics with LLM-as-a-judge assessments. Specifically, we define automatic measures-Faithfulness, Coherence, and Diversity-together with two LLM-based evaluations: Correctness and Realism. Benchmark experiments show that even state-of-the-art LLMs struggle with MediLongChat. These findings highlight the benchmark's applicability and underscore the need for tailored methods to advance healthcare agents.
>
---
#### [new 007] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型推理任务，旨在解决工具调用不必要的问题。提出AutoTool，通过强化学习自适应决定是否调用工具，提升推理准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.19852](https://arxiv.org/pdf/2605.19852)**

> **作者:** Qinghe Ma; Zhen Zhao; Yiming Wu; Jian Zhang; Lei Bai; Yinghuan Shi
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Tool-augmented reasoning has emerged as a promising direction for enhancing the reasoning capabilities of multimodal large language models (MLLMs). However, existing studies mainly focus on enabling models to perform tool invocation, while neglecting the necessity of invoking tools. We argue that tool usage is not always beneficial, as redundant or inappropriate invocations largely increase reasoning overhead and even mislead model predictions. To address this issue, we introduce AutoTool, a model that adaptively decides whether to invoke tools according to the characteristics of each query. Within a reinforcement learning framework, we design an explicit dual-mode reasoning strategy with mode-specific reward functions to guide the model toward producing accurate responses. Moreover, to prevent premature bias toward a single reasoning mode, AutoTool jointly explores and balances tool-assisted and text-centric reasoning throughout training, and promotes free exploration in later stages. Extensive experiments demonstrate that AutoTool exhibits outstanding performance and high efficiency, yielding a 21.8\% accuracy gain on V* benchmark compared to the base model, and a 44.9\% improvement in efficiency over existing tool-augmented methods on POPE benchmark. Code is available at this https URL.
>
---
#### [new 008] ClinSeekAgent: Automating Multimodal Evidence Seeking for Agentic Clinical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ClinSeekAgent，解决临床推理中主动获取多模态证据的问题，通过自动化框架提升临床决策效果。**

- **链接: [https://arxiv.org/pdf/2605.20176](https://arxiv.org/pdf/2605.20176)**

> **作者:** Juncheng Wu; Letian Zhang; Yuhan Wang; Haoqin Tu; Hardy Chen; Zijun Wang; Cihang Xie; Yuyin Zhou
>
> **备注:** 24 pages, 9 figures; Project Page: this https URL
>
> **摘要:** Large language models (LLMs) and agentic systems have shown promise for clinical decision support, but existing works largely assume that evidence has already been curated and handed to the model. Real-world clinical workflows instead require agents to actively seek, iteratively plan, and synthesize multimodal evidence from heterogeneous sources. In this paper, we introduce ClinSeekAgent, an automated agentic framework for dynamic multimodal evidence seeking that shifts the paradigm from passive evidence consumption to active evidence acquisition. Given only a clinical query and access to raw data sources, ClinSeekAgent gathers evidence by querying medical knowledge bases, navigating raw EHRs, and invoking medical imaging tools; refines its hypotheses as new information emerges; and integrates the collected evidence into grounded clinical decisions. ClinSeekAgent serves both as an inference-time agent for frontier LLMs and as a training-time pipeline for distilling high-quality agent trajectories into compact open-source models. To validate its inference-time effectiveness, we construct ClinSeek-Bench, which pairs Curated Input reasoning from fixed pre-selected evidence with Automated Evidence-Seeking over raw clinical data. On text-only EHR tasks, ClinSeekAgent improves Claude Opus 4.6 from 60.0 to 63.2 overall F1 and MiniMax M2.5 from 43.1 to 47.3, with positive risk-prediction gains in 7 out of 9 evaluated host models. On multimodal tasks, ClinSeekAgent improves Claude Opus 4.6 from 47.5 to 62.6 (+15.1); all evaluated models improve across the three CXR-related task groups. We further validate ClinSeekAgent as a training pipeline by distilling agentic evidence-seeking trajectories into ClinSeek-35B-A3B, which achieves 34.0 average F1 on existing AgentEHR-Bench, improving over its Qwen3.5-35B-A3B baseline by +11.9 points and approaching Claude Opus 4.6.
>
---
#### [new 009] Mathematical Reasoning in Large Language Models: Benchmarks, Architectures, Evaluation, and Open Challenges
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，旨在评估大语言模型的推理能力，通过分析数据集、架构和训练策略，解决模型推理可靠性与评估标准不一致的问题。**

- **链接: [https://arxiv.org/pdf/2605.19723](https://arxiv.org/pdf/2605.19723)**

> **作者:** Husnain Amjad; Raja Khurram Shahzad; Aamir Shahzad; Mehwish Fatima
>
> **摘要:** Mathematical reasoning is essential for problem-solving in education, science, and industry, serving as a crucial benchmark for evaluating artificial intelligence systems. As Large Language Models (LLMs) improve their reasoning capabilities, understanding how well they perform mathematical reasoning has become increasingly important. This survey synthesizes recent advancements in mathematical reasoning with LLMs through a structured analysis of datasets, architectures, training strategies, and evaluation protocols. Our systematic review encompasses approximately 120 peer-reviewed studies and preprints, examining the evolution of this research area and providing a unified analytical framework to understand current progress and limitations. Our study particularly introduces a unified taxonomy of mathematical datasets, distinguishing between pretraining corpora, supervised fine-tuning resources, and evaluation benchmarks across varying levels of reasoning complexity. A systematic analysis of reasoning architectures and training strategies, including tool integration, verifier-guided reasoning, and parameter-efficient adaptation, is presented to assess their effects on reasoning robustness and generalization. Moreover, a comparative evaluation of existing metrics highlights the gap between final-answer accuracy and process-level reasoning verification. By synthesizing insights across these areas, our analysis identifies recurring failure modes, such as reasoning faithfulness issues, benchmark biases, and generalization limitations, and outlines key research directions toward improving symbolic grounding, evaluation reliability, and the development of more robust and trustworthy LLM-based reasoning systems.
>
---
#### [new 010] optimize_anything: A Universal API for Optimizing any Text Parameter
- **分类: cs.CL; cs.AI; cs.LG; cs.NE; cs.SE**

- **简介: 该论文提出一种通用的文本参数优化系统，解决跨领域优化问题。通过LLM搜索，在多个任务中取得最佳效果，展示文本优化作为通用解决方案的潜力。**

- **链接: [https://arxiv.org/pdf/2605.19633](https://arxiv.org/pdf/2605.19633)**

> **作者:** Lakshya A Agrawal; Donghyun Lee; Shangyin Tan; Wenjie Ma; Karim Elmaaroufi; Rohit Sandadi; Sanjit A. Seshia; Koushik Sen; Dan Klein; Ion Stoica; Joseph E. Gonzalez; Omar Khattab; Alexandros G. Dimakis; Matei Zaharia
>
> **备注:** 16 pages, 11 figures; Blog: this https URL
>
> **摘要:** Can a single LLM-based optimization system match specialized tools across fundamentally different domains? We show that when optimization problems are formulated as improving a text artifact evaluated by a scoring function, a single AI-based optimization system-supporting single-task search, multi-task search with cross-problem transfer, and generalization to unseen inputs-achieves state-of-the-art results across six diverse tasks. Our system discovers agent architectures that nearly triple Gemini Flash's ARC-AGI accuracy (32.5% to 89.5%), finds scheduling algorithms that cut cloud costs by 40%, generates CUDA kernels where 87% match or beat PyTorch, and outperforms AlphaEvolve's reported circle packing solution (n=26). Ablations across three domains reveal that actionable side information yields faster convergence and substantially higher final scores than score-only feedback, and that multi-task search outperforms independent optimization given equivalent per-problem budget through cross-task transfer, with benefits scaling with the number of related tasks. Together, we show for the first time that text optimization with LLM-based search is a general-purpose problem-solving paradigm, unifying tasks traditionally requiring domain-specific algorithms under a single framework. We open-source optimize\_anything with support for multiple backends as part of the GEPA project at this https URL .
>
---
#### [new 011] Rethinking How to Remember: Beyond Atomic Facts in Lifelong LLM Agent Memory
- **分类: cs.CL**

- **简介: 该论文属于LLM记忆任务，解决长期对话中记忆存储与推理问题。提出TriMem框架，融合多粒度表示，提升记忆 fidelity 和推理能力。**

- **链接: [https://arxiv.org/pdf/2605.19952](https://arxiv.org/pdf/2605.19952)**

> **作者:** Jingwei Sun; Jianing Zhu; Jiangchao Yao; Tongliang Liu; Bo Han
>
> **摘要:** To enable reliable long-term interaction, LLM agents require a memory system that can faithfully store, efficiently retrieve, and deeply reason over accumulated dialogue history. Most existing methods adopt an extracted fact based paradigm: handcrafted static prompts compress raw dialogues into atomic facts, which are then stored, matched, and injected into downstream reasoning. Nevertheless, such fact-centric designs inevitably discard fine-grained details in original dialogues and fail to support deep reasoning over scattered isolated facts. Moreover, static prompts cannot maintain consistent extraction granularity across diverse dialogue styles. To address these limitations, we propose TriMem, which maintains three coexisting representation granularities, including raw dialogue segments anchored by source identifiers for storage fidelity, extracted atomic facts for efficient memory retrieval, synthesized profiles that aggregate dispersed facts into holistic semantic understanding for deep reasoning. We further adopt TextGrad-based prompt optimization, which iteratively refines extraction and profiling prompts via response quality feedback, achieving lifelong evolution without any parameter updating. Extensive experiments on LoCoMo and PerLTQA across multiple LLM backbones demonstrate that TriMem consistently outperforms strong memory baselines. The code is available at this https URL .
>
---
#### [new 012] Are Rationales Necessary and Sufficient? Tuning LLMs for Explainable Misinformation Detection
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于可解释的虚假信息检测任务，旨在解决传统方法缺乏透明性和生成冗余理由的问题。通过微调大语言模型并提出LONSREX方法，提升判断的必要性和充分性。**

- **链接: [https://arxiv.org/pdf/2605.19285](https://arxiv.org/pdf/2605.19285)**

> **作者:** Bing Wang; Rui Miao; Ximing Li; Chen Shen; Shaotian Yan; Changchun Li; Kaiyuan Liu; Xiaosong Yuan; Jieping Ye
>
> **备注:** Accepted by KDD 2026. 12 pages, 8 figures. Code: this https URL
>
> **摘要:** The rapid spread of misinformation on social media platforms has become a formidable challenge. To mitigate its proliferation, Misinformation Detection (MD) has emerged as a critical research topic. Traditional MD approaches based on small models typically perform binary classification through a black-box process. Recently, the rise of Large Language Models (LLMs) has enabled explainable MD, where models generate rationales that explain their decisions, thereby enhancing transparency. Existing explainable MD methods primarily focus on crafting sophisticated prompts to elicit rationales from off-the-shelf LLMs. In this work, we propose a pipeline to fine-tune a dedicated LLM specifically for explainable MD. Our pipeline begins by collecting large-scale fact-checked articles, and then uses multiple strong LLMs to produce veracity predictions and rationales. To ensure high-quality training data, we leverage a filtering strategy that selects only the correct instances for fine-tuning. While this pipeline is intuitive and prevalent, our experiments reveal that naive filtering based solely on label correctness is insufficient in practice and suffers from two critical limitations: (1) Coarse-grained labels cause insufficient rationales: Rationales filtered solely based on binary labels are insufficient to adequately support their decisions; (2) Over-verification behavior causes unnecessary rationales: Stronger LLMs tend to exhibit over-verification behavior, producing excessively verbose and unnecessary rationales. To address these issues, we introduce LONSREX, a novel data synthesis pipeline to Locate Necessary and Sufficient Rationales for Explainable MD. Specifically, we propose a metric that quantifies the contribution of each verification step to the final prediction, thereby evaluating its necessity and sufficiency. Experimental results demonstrate the effectiveness of LONSREX.
>
---
#### [new 013] CLIF: Concept-Level Influence Functions for Transparent Bottleneck Models
- **分类: cs.CL**

- **简介: 该论文属于可解释性AI任务，旨在解决深度学习模型黑箱问题。通过影响函数分析样本和概念层面的影响，提升模型可解释性并优化数据调试。**

- **链接: [https://arxiv.org/pdf/2605.19848](https://arxiv.org/pdf/2605.19848)**

> **作者:** Yike Sun; Mingkun Xu; Mu You; Zhongzhi He; Henghua Shen; Zehan Tan; Derek F. Wong; Tao Fang
>
> **摘要:** In recent years, the black-box nature of deep learning models has limited their application in high-stakes domains such as medical diagnosis and finance, where interpretability is essential. To address this, we propose a novel approach using influence functions to enhance interpretability in NLP models at both the sample and concept levels. Experiments on CEBaB and Yelp datasets show that influence functions effectively identify the most impactful training samples, both helpful and harmful, on model predictions. By adjusting the labels and weights of these samples, we demonstrate that model performance can be restored to baseline levels without retraining, confirming the value of influence functions for efficient data debugging. Furthermore, our concept-level analysis identifies key concepts within Concept Bottleneck Models (CBM) that significantly affect predictions. Modifying these concepts alters model behavior observably, providing clear insights into the decision process.
>
---
#### [new 014] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型训练任务，旨在解决模型感知与推理能力不足的问题。通过分阶段训练提升视觉感知，优化推理效果，实验表明该方法有效提升性能。**

- **链接: [https://arxiv.org/pdf/2605.20177](https://arxiv.org/pdf/2605.20177)**

> **作者:** Juncheng Wu; Hardy Chen; Haoqin Tu; Xianfeng Tang; Freda Shi; Hui Liu; Hanqing Lu; Cihang Xie; Yuyin Zhou
>
> **备注:** 19 pages, 9 figures; Accepted to ICML 2026; Project Page: this https URL
>
> **摘要:** Recent advances in vision-language models (VLMs) emphasize long chain-of-thought reasoning; yet, we find that their performance on visual tasks is primarily limited by a lack of visual perception as opposed to reasoning itself. In this work, we systematically study the interplay between perception and reasoning in VLM post-training by decomposing their capabilities into three separate training stages: visual perception, visual reasoning, and textual reasoning, incorporating specialized training data. We demonstrate that visual perception (a) requires targeted optimization with specialized data; (b) serves as a fundamental scaffold that should be solidified through staged training before refining visual reasoning; and (c) is more effectively learned via RL than caption-based SFT. Our experiments across multiple VLMs demonstrate that staged training consistently improves both visual perception and reasoning performance over merged training. Notably, models trained with our approach achieve 1.5% higher reasoning accuracy with 20.8% shorter reasoning traces, suggesting that superior perception reduces the need for excessive reasoning. Furthermore, we show that this capability-based staging represents a new curriculum dimension orthogonal to traditional difficulty-based curricula, and combining both yields further additive gains. Our staged-training models achieve superior performance among open-weight VLMs, establishing advanced results on several visual math and perception (e.g., +5.2% on WeMath and +3.7% on RealWorldQA) tasks compared with the base counterpart.
>
---
#### [new 015] MMoA: An AI-Agent framework with recurrence for Memoried Mixure-of-Agent
- **分类: cs.CL**

- **简介: 该论文提出MMoA框架，解决MoA系统中静态路由无法捕捉时序依赖的问题。通过引入LSTM实现动态代理选择，提升效率与上下文感知能力。任务为优化多代理语言模型系统。**

- **链接: [https://arxiv.org/pdf/2605.19194](https://arxiv.org/pdf/2605.19194)**

> **作者:** Rui Chu
>
> **摘要:** The Mixture-of-Agents (MoA) framework has shown promise in improving large language model (LLM) performance by aggregating outputs from multiple agents. However, existing MoA systems often rely on static routers that do not fully capture temporal and contextual dependencies across aggregation layers. To address this limitation, we propose MMoA, a recurrent MoA architecture that integrates LSTM-based gating into the agent selection process. The recurrence router adaptively modulates agent contributions based on both current inputs and historical routing decisions, enabling more context-aware aggregation. We evaluate MMoA on standard instruction-following benchmarks, including AlpacaEval 2.0, MT-Bench, and Arena-Hard. The results show that MMoA achieves comparable accuracy to traditional MoA while reducing computational overhead by dynamically activating fewer agents. For example, on AlpacaEval 2.0, MMoA achieves a win rate of 58.0%, compared with 59.8% for MoA, while improving runtime efficiency by up to 4.6%. These results suggest that MMoA provides a scalable and efficient approach for adaptive multi-agent LLM systems.
>
---
#### [new 016] A Data-Driven Approach to Idiomaticity Based on Experts' Criteria in Theoretical Linguistics
- **分类: cs.CL**

- **简介: 该论文属于语言学研究任务，旨在探讨多词表达的习语性。通过分析286个MWEs，验证专家提出的16项标准，发现无绝对习语表达， lexical因素影响最大。**

- **链接: [https://arxiv.org/pdf/2605.19575](https://arxiv.org/pdf/2605.19575)**

> **作者:** Elena Mikhalkova; Anastasiya Vishnyakova; Anastasiya Drozdova; Polina Gavin; Aleksander Zhmykhov; Timofey Protasov
>
> **摘要:** The article observes data analysis of 286 multi-word expressions (MWEs) based on 16 lexical, grammatical and other criteria described in theoretical books and papers on the notion of idiomaticity. MWEs were collected from the same theoretical sources, and a set of experts in linguistics annotated them with these categories. The distribution of categories shows that there are no absolutely idiomatic expressions. Lexical criteria seem to be the most influential; grammatical criteria are bound to certain conditions; presence of obsolete words and grammar influence ability of an MWE to be replaced with one word.
>
---
#### [new 017] LambdaPO: A Lambda Style Policy Optimization for Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文提出LambdaPO，用于优化语言模型的推理能力。针对传统方法信息丢失问题，通过重构优势估计和引入语义密度奖励，提升模型在数学推理和问答任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.19416](https://arxiv.org/pdf/2605.19416)**

> **作者:** Zhe Yuan; Yipeng Zhou; Jinghan Li; Xinyuan Chen; Bowen Deng; Zhiqian Chen; Liang Zhao
>
> **摘要:** Group Relative Policy Optimization(GRPO) has become a cornerstone of modern reinforcement learning alignment, prized for its efficacy in foregoing an explicit value-critic by leveraging reward normalization across sampled trajectory cohorts. However, the method's reliance on a monolithic statistical baseline, such as the group mean, collapses the relational topology of the trajectory space into a single scalar, thereby erasing the fine-grained preference information essential for navigating complex, rank-sensitive reward landscapes. To address this issue, we introduce a novel framework, Lambda Policy Optimization (LambdaPO), that addresses this information-theoretic bottleneck by re-conceptualizing advantage estimation from a scalar value to a decomposed, pairwise preference structure. Specifically, the advantage for any given trajectory is formulated as the integrated sum of reward differentials against all peers in its cohort, where each pairwise comparison is dynamically attenuated by the policy's own probabilistic confidence in the established preference. To further mitigate the sparsity of binary outcome supervision, we augment the objective with a semantic density reward, derived from the precision-recall alignment between generated reasoning traces and ground-truth solutions. As a result, our method can mine more fine-grained optimization signals from a group of rollouts, guiding the LLM to a better optima. Experimental results across challenging math reasoning and question-answering tasks demonstrates that LambdaPO improves performance compared to the baseline methods.
>
---
#### [new 018] The Annotation Scarcity Paradox in Low-Resource NLP Evaluation: A Decade of Acceleration and Emerging Constraints
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨低资源NLP评估中的标注稀缺悖论，分析其成因及解决方案。**

- **链接: [https://arxiv.org/pdf/2605.19066](https://arxiv.org/pdf/2605.19066)**

> **作者:** Vukosi Marivate
>
> **备注:** Under Review
>
> **摘要:** Over the past decade, low-resource natural language processing (NLP) has experienced explosive growth, propelled by cross-lingual transfer, massively multilingual models, and the rapid proliferation of benchmarks. Yet this apparent progress masks a critical, insufficiently examined tension: the deep sociolinguistic expertise required to evaluate increasingly complex generative systems is severely strained, inequitably distributed, and structurally marginalised. We present a critical narrative survey of low-resource NLP evaluation (2014--present), tracing its evolution across three phases: early heuristic optimism, the illusions of top-down benchmark scaling, and the current era of generative bottlenecks. We conceptualise the \emph{Annotation Scarcity Paradox}, the structural friction arising when the technical capacity to scale models vastly outpaces the sovereign human infrastructure required to authentically evaluate them. By examining extractive data pipelines, undercompensated ``ghost work'', and language data flaring, we argue that this paradox threatens the epistemic validity of reported progress. We survey emerging responses -- including data augmentation, model-based evaluation, participatory curation, and annotation-efficient approaches via item response theory and active learning -- and assess their equity and validity trade-offs. We close with a practitioner call to action, arguing that overcoming this bottleneck requires a paradigm shift from transactional data extraction to relational, community-embedded evaluation rooted in epistemic governance, data sovereignty, and shared ownership.
>
---
#### [new 019] Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决零样本Text-to-SPARQL生成问题。通过强化学习方法GRPO训练小模型，提升查询生成效果。**

- **链接: [https://arxiv.org/pdf/2605.20066](https://arxiv.org/pdf/2605.20066)**

> **作者:** Jann Pfeifer; Debayan Banerjee; Ricardo Usbeck
>
> **备注:** Accepted by NeSy 2026
>
> **摘要:** Knowledge graph question answering seeks to translate natural language questions into executable queries over knowledge graphs, but existing approaches often rely on large models or full supervision in the form of gold query annotations. This study examines whether reinforcement learning with outcome-based rewards can train a small instruction-tuned language model to perform zero-shot Text-to-SPARQL generation in the scholarly domain. Group-Relative Policy Optimization (GRPO) is applied to the Qwen3-1.7B model on DBLP-QuAD, using prompts that combine natural language questions with symbolic hints about entities and relations. Training relies on execution feedback, structural constraints, and answer-level rewards, with an additional variant that incorporates gold-query-based shaping. The resulting models are compared to the unmodified zero-shot baseline and to a supervised DoRA-finetuned baseline across answer-level accuracy, execution accuracy, category-wise scores, and generalization to held-out templates. GRPO substantially improves over the zero-shot baseline and exhibits competitive generalization, while supervised DoRA finetuning achieves higher overall accuracy on the same model scale. Ablation analyses indicate that execution-based rewards account for most gains, with additional shaping yielding limited additional benefit, suggesting that outcome-based reinforcement learning is a viable training strategy when gold queries are unavailable for token-level supervision.
>
---
#### [new 020] Fine-tuning language encoding models on slow fMRI improves prediction for fast ECoG
- **分类: cs.CL**

- **简介: 该论文属于脑机接口任务，旨在解决ECoG模型训练数据不足的问题。通过用fMRI数据微调语言模型，提升ECoG预测性能。**

- **链接: [https://arxiv.org/pdf/2605.19224](https://arxiv.org/pdf/2605.19224)**

> **作者:** Aditya R. Vaidya; Richard J. Antonello; Alexander G. Huth
>
> **摘要:** Neuroscientists have recently turned to intracranial brain recording methods, like electrocorticography (ECoG), for human experiments because of the fine spatial and temporal resolution that they afford. Models trained on this data, however, are fundamentally restricted by the patient populations that can receive the implants necessary for recording. We propose using non-invasive fMRI to bridge the gap in training data. Using spoken language representations fine-tuned on fMRI, we build encoding models of ECoG. These representations showed improved prediction performance in ECoG, even though the temporal resolution of fMRI is two orders of magnitude worse. Prediction improved in frequency bands well beyond what is directly measured in fMRI. Next, to test the procedure's generalization ability, we fine-tuned models on fMRI responses that were temporally downsampled by a factor of 2. Despite the loss in resolution, these models were able to predict fMRI and ECoG responses at levels comparable to the original fMRI-tuned models. Finally, we showed that ECoG performance steadily scales with the amount of fMRI-tuning data. Our results show that "slow" data like fMRI can be a valuable resource for building better models of "fast" brain data like ECoG. In the future, integrating across multiple recording methods may further improve performance in other applications, like decoding.
>
---
#### [new 021] FlexDraft: Flexible Speculative Decoding via Attention Tuning and Bonus-Guided Calibration
- **分类: cs.CL**

- **简介: 该论文提出FlexDraft，解决大批次下并行推测解码的效率与质量问题，通过注意力调优、奖励校准和动态解码策略提升性能。**

- **链接: [https://arxiv.org/pdf/2605.20022](https://arxiv.org/pdf/2605.20022)**

> **作者:** Yaojie Zhang; Jianuo Huang; Junlong Ke; Yuhang Han; Yongji Long; Tianchen Zhao; Biqing Qi; Linfeng Zhang
>
> **摘要:** Speculative decoding accelerates memory-bound LLM inference without quality degradation by using a fast drafter to propose multiple candidate tokens and the target model to verify them in parallel. However, conventional sequential speculative decoding suffers from mutual waiting between drafting and verification, and repeated exchange of intermediate states further increases memory access overhead. Parallel speculative decoding addresses this limitation by performing drafting and verification within a single target forward pass, allowing future drafts to be prepared while current candidates are being verified. Although effective at small batch sizes, existing parallel speculative decoding methods either require costly continual pretraining with quality degradation or suffer from low acceptance rates. More importantly, this paradigm inherently suffers from uncertainty in both the bonus token and the accepted length, leading to draft verification mismatch and causing throughput gains to collapse at large batch sizes. To address these limitations, we introduce FlexDraft, a lossless speculative decoding framework that flexibly adapts to varying batch sizes through three key designs. (1) Attention Tuning enables block diffusion drafting by tuning only the attention projectors of the final few layers on mask tokens, while keeping the autoregressive path frozen to preserve the target distribution and produce high quality drafts with minimal trainable parameters. (2) Bonus-guided Calibration uses a lightweight MLP conditioned on the resolved bonus token to calibrate draft logits, mitigating draft verification mismatch caused by bonus token uncertainty. (3) Flex Decoding dynamically switches between parallel draft and verify at small batch sizes and sequential draft then verify at large batch sizes, and adjusts verification length based on draft confidence to eliminate redundant computation.
>
---
#### [new 022] BalanceRAG: Joint Risk Calibration for Cascaded Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息检索与生成任务，解决RAG系统中如何合理控制检索调用的问题。通过联合校准风险阈值，提升系统效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.20084](https://arxiv.org/pdf/2605.20084)**

> **作者:** Zijun Jia; Yuanchang Ye; Sen Jia; Yiyao Qian; Haoning Wang; Baojie Chen; Diyin Tang; Jinsong Yu; Zhiyuan Wang
>
> **摘要:** Large language models (LLMs) can enhance factuality via retrieval-augmented generation (RAG), but applying RAG to every query is unnecessary when the model-only answer is reliable. This motivates cascaded RAG: each query is first handled by an LLM-only branch, escalated to a RAG fallback only if the primary branch is uncertain, and abstained from when neither branch is sufficiently trustworthy. However, calibrating such cascades stage by stage may be conservative, since the final utility depends on joint uncertainty thresholding of LLM-only and RAG. In this work, we develop BalanceRAG to certify threshold pairs at a target risk level. Given uncertainty scores from the two branches, BalanceRAG frames each threshold pair as an operating point on a two-dimensional lattice and identifies safe operating points using sequential graphical testing. This enables risk-adaptive threshold calibration, controlling the system-level error rate among accepted points, while retaining more examples. Furthermore, BalanceRAG extends to multi-risk calibration, allowing retrieval usage to be bounded together with the selection-conditioned risk. Experiments on three open-domain question answering (QA) benchmarks across multiple LLM backbones demonstrate that BalanceRAG meets prescribed risk levels, preserves higher coverage and more accepted correct examples, and reduces unnecessary retrieval calls compared with always-on RAG.
>
---
#### [new 023] Language models struggle with compartmentalization
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究语言模型在处理同一概念的不同表达形式时出现的分割问题。工作揭示了模型无法共享统计信息，导致效率下降。**

- **链接: [https://arxiv.org/pdf/2605.19284](https://arxiv.org/pdf/2605.19284)**

> **作者:** Thomas Vincent Howe; David Wingate
>
> **备注:** 9 pages, 8 figures, plus 9 pages of appendices. Submitted to NeurIPS 2026. Code: this https URL. Eval data: this https URL
>
> **摘要:** In the training data used by large language models (LLMs), the same latent concept is often presented in multiple distinct ways: the same facts appear in English and Swahili; many functions can be expressed in both Python and Haskell; we can express propositions in both formal and natural language. We show that LLMs can exhibit compartmentalization, where they fail to identify and share statistical strength between distinct presentations of unified concepts. In the worst case, LLMs simply learn parallel internal representations of each presentation of the concept, saturating model capacity with redundancies and decreasing sample efficiency with the number of such presentations. We also demonstrate that synthetic parallel data can fail to improve this despite being easily learned itself. Under this framework, we find that, for small models, early multilingual learning is nearly entirely compartmentalized. Finally, all interventions that we study exhibit a phase transition in which their effectiveness depends on the number of distinct presentations, suggesting that the language modeling objective may only inconsistently unify representations.
>
---
#### [new 024] GoLongRL: Capability-Oriented Long Context Reinforcement Learning with Multitask Alignment
- **分类: cs.CL**

- **简介: 该论文属于长文本强化学习任务，旨在解决数据构造单一和奖励不均衡问题。提出GoLongRL数据集和TMN-Reweight方法，提升长文本能力。**

- **链接: [https://arxiv.org/pdf/2605.19577](https://arxiv.org/pdf/2605.19577)**

> **作者:** Minxuan Lv; Tiehua Mei; Tanlong Du; Junmin Chen; Zhenpeng Su; Ziyang Chen; Ziqi Wang; Zhennan Wu; Ruotong Pan; jian Liang; Ruiming Tang; Han Li
>
> **摘要:** We present GoLongRL, a fully open-source, capability-oriented post-training recipe for long-context reinforcement learning with verifiable rewards (RLVR). Existing long-context RL methods often treat data construction as a matter of designing increasingly complex retrieval paths, leading to homogeneous task coverage and reward formulations that inadequately reflect practical long-context requirements. Our work offers two contributions. (1) Capability-oriented data construction with full open release. We openly release a dataset of 23K RLVR samples, the complete construction pipeline, and all training code. Guided by a taxonomy of long-context capabilities, the dataset spans 9 task types, each paired with its natural evaluation metric. It comprises curated open-source samples from established corpora and synthetic samples whose QA pairs are generated from real source documents such as books, academic papers, and multi-turn dialogues. Under the same vanilla GRPO setup, our dataset alone outperforms the closed-source QwenLong-L1.5 dataset. Moreover, our Qwen3-30B-A3B model trained on this data delivers long-context performance comparable to DeepSeek-R1-0528 and Qwen3-235B-A22B-Thinking-2507, suggesting that broader coverage and greater reward diversity substantially benefit long-context capability improvement. (2) TMN-Reweight for heterogeneous multitask optimization. To address optimization challenges from heterogeneous rewards, we propose TMN-Reweight, which combines task-level mean normalization for cross-task reward scale alignment with difficulty-adaptive weighting for more reliable advantage estimation. TMN-Reweight further improves average performance over vanilla GRPO, with general capabilities preserved or improved across reported evaluations.
>
---
#### [new 025] How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence
- **分类: cs.CL**

- **简介: 该论文属于文档智能任务，旨在解决DLA管道的结构脆弱性问题。通过提出审计框架，分析扰动对布局结构的影响，提升鲁棒性评估的准确性。**

- **链接: [https://arxiv.org/pdf/2605.19309](https://arxiv.org/pdf/2605.19309)**

> **作者:** Yue Chen; Yihao Wang; Ziyi Tang; Keze Wang
>
> **备注:** 19 pages, preprint
>
> **摘要:** Document Layout Analysis (DLA) pipelines provide structured page representations for retrieval-augmented generation, long-document question answering, and other document intelligence systems, yet their robustness evaluation remains largely area-centric. We identify this Footprint Bias and propose a lightweight output-level auditing framework that decouples probe construction, policy-driven targeting, and structure-aware diagnosis. The framework combines Block-level Structural Loss Rate (B-SLR), granularity-aware exposure descriptors, and pathway attribution to analyze where perturbations interact with layout structure and how failures propagate. Across MinerU and PP-StructureV3 on 1,000 pages, affected area weakly tracks perturbation-induced OCR instability (R^2=0.384/0.110), whereas B-SLR aligns much more closely with it (R^2=0.727/0.916). Exposure descriptors further separate occlusion- and topology-dominant pathways, and small structurally targeted probes cause downstream QA/retrieval degradation comparable to larger-footprint perturbations. These results shift DLA robustness evaluation from footprint-based stress testing toward structure-aware vulnerability auditing.
>
---
#### [new 026] LLM-Based Financial Sentiment Analysis in Arabic: Evidence from Saudi Markets
- **分类: cs.CL**

- **简介: 该论文属于金融情感分析任务，旨在解决阿拉伯语金融文本情感建模难题。通过构建沙特市场语料库，融合新闻与社交媒体数据，实现公司级情感分析。**

- **链接: [https://arxiv.org/pdf/2605.19714](https://arxiv.org/pdf/2605.19714)**

> **作者:** Mona H. Albaqawi; Eman M. Albalkhi; Joud A. Albaiti; Enrico Lopedoto
>
> **备注:** Accepted at the 7th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT7), co-located with LREC 2026, Palma de Mallorca, Spain, May 2026. ISBN: 978-2-493814-52-4
>
> **摘要:** Investor sentiment shapes financial markets, yet modeling sentiment in Arabic financial contexts remains challenging due to linguistic complexity and limited resources. We present an Arabic NLP framework for large-scale financial sentiment analysis tailored to the Saudi market, integrating official financial news and social media to capture institutional and public investor sentiment. The framework constructs a large Arabic financial corpus through a multi-stage pipeline encompassing data collection, cleaning, deduplication, entity linking, and sentiment annotation. Transformer-based NER combined with a curated company lexicon links textual mentions to canonical company identifiers, with sentiment labels assigned using a five-class scheme. The resulting dataset of 84K samples supports company-level sentiment aggregation and analysis of sentiment dynamics relative to stock market behavior on the Saudi Exchange. Experimental results demonstrate reliable and scalable Arabic financial sentiment analysis.
>
---
#### [new 027] IMLJD: A Computational Dataset for Indian Matrimonial Litigation Analysis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文介绍IMLJD数据集，用于分析印度婚姻诉讼案件。任务是法律数据分析，解决婚姻诉讼判决模式研究问题，收集并结构化了3613份法院判决数据。**

- **链接: [https://arxiv.org/pdf/2605.19346](https://arxiv.org/pdf/2605.19346)**

> **作者:** Joy Bose
>
> **备注:** 8 pages, 2 figures, 5 tables. Dataset available at this http URL and Code at this http URL
>
> **摘要:** We present IMLJD, an open dataset of 3,613 Indian court judgments covering matrimonial disputes under IPC Section 498A, the Protection of Women from Domestic Violence Act, and CrPC Section 482. The dataset covers the Supreme Court of India from 2000 to 2024 (1,474 cases) and the Karnataka High Court from 2018 to 2024 (2,139 cases), with structured outcome labels, metadata-derived indicators, and a knowledge graph. We find that 57.6% of quashing petitions succeed at the Supreme Court level compared to 39.7% at the Karnataka High Court level. On a matched 2018 to 2024 period, the SC quash rate is 59.3%, widening the differential to 19.6 percentage points and confirming the finding is robust to temporal adjustment. The dataset, code, and knowledge graph are released openly at this https URL and this https URL.
>
---
#### [new 028] CAIT: A Syntactic Parsing Toolkit for Child-Adult InTeractions
- **分类: cs.CL**

- **简介: 该论文提出CAIT工具包，解决儿童-成人互动语料的句法分析问题。通过训练专用解析器及词性标注器，提升对CHILDES数据的分析效果，支持语言习得研究。**

- **链接: [https://arxiv.org/pdf/2605.19718](https://arxiv.org/pdf/2605.19718)**

> **作者:** Francesca Padovani; Xiulin Yang; Bastian Bunzeck; Jaap Jumelet; Yevgen Matusevych; Nathan Schneider; Arianna Bisazza
>
> **摘要:** CHILDES is a paramount resource for language acquisition studies -- yet computational tools for analyzing its syntactic structure remain limited. Leveraging the recent release of the UD-English-CHILDES treebank with gold-standard Universal Dependencies (UD) annotations, we train a state-of-the-art dependency parser specifically tailored to CHILDES. The parser more accurately captures syntactic patterns in child--adult interactions, outperforming widely used off-the-shelf English parsers, including SpaCy and Stanza. Alongside the parser, we also release a Part-of-Speech tagger and an utterance-level construction tagger, which together form the open-source Syntactic Parsing Toolkit for Child--Adult InTeractions (CAIT). Through a detailed error analysis and a case study tracking the distribution of syntactic constructions across developmental time in CHILDES, we demonstrate the practical utility of the toolkit for large-scale, reproducible research on language acquisition.
>
---
#### [new 029] Retrieval-Augmented Linguistic Calibration
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的可信度校准任务，旨在解决语言信心表达的标准化问题。通过构建分布模型和引入Faithfulness Divergence指标，提出RALC方法提升语言信心的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.19344](https://arxiv.org/pdf/2605.19344)**

> **作者:** Yi-Fan Yeh; Linwei Tao; Minjing Dong; Tao Huang; Jialin Yu; Philip Torr; Chang Xu
>
> **摘要:** Linguistic cues such as "I believe" and "probably" offer an intuitive interface for communicating confidence, yet a generalisable, principled calibration framework for linguistic confidence expressions remains underexplored. In particular, co-occurring linguistic cues, contextual variation, and subjective audience interpretation pose unique challenges. We therefore model linguistic confidence as a distribution over plausible perceived probability values that a statement is correct, capturing interpretation variability that scalar representations discard. Within this distributional framework, we introduce faithfulness as a complementary evaluation dimension and present Faithfulness Divergence (FD), an information-theoretic metric quantifying the surprise induced in audience beliefs upon truth revelation. Building on these foundations, we present Retrieval-Augmented Linguistic Calibration (RALC), a lightweight post-hoc pipeline that propagates calibrated confidence signals back into natural language via retrieval-augmented rewriting. Across three QA benchmarks and five LLM families, RALC improves in-domain faithfulness and calibration up to 66% and 58%, respectively, outperforming black-box and grey-box calibration baselines.
>
---
#### [new 030] Drifting Objectives for Refining Discrete Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究如何将连续生成器的漂移方法应用于离散扩散语言模型，解决文本生成质量提升问题。通过引入TokenDrift方法，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.19470](https://arxiv.org/pdf/2605.19470)**

> **作者:** Daisuke Oba; Hiroki Furuta; Naoaki Okazaki
>
> **备注:** Project page: this https URL
>
> **摘要:** Discrete diffusion language models (DDLMs) generate text by iteratively denoising categorical token sequences, while recent drifting methods for continuous generators suggest that part of this sampling-time correction can instead be absorbed into training through an anti-symmetric fixed-point objective. We study how to transfer this principle to DDLMs, where the main challenge is the interface with discrete text: hard token samples are non-differentiable, and categorical predictions do not directly provide continuous samples to drift. We formulate TokenDrift, a drifting objective that lifts categorical predictions to soft-token features, applies anti-symmetric drifting in a frozen semantic space, and backpropagates the resulting stop-gradient feature target to DDLM logits. In controlled continual-training experiments with masked and uniform-state diffusion backbones, TokenDrift improves fixed-NFE generation quality over matched continuation baselines, reducing Gen.-PPL at 4 NFEs by 89% on MDLM and 86% on DUO. These results suggest that drifting can provide a practical refinement objective for DDLMs.
>
---
#### [new 031] CopT: Contrastive On-Policy Thinking with Continuous Spaces for General and Agentic Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CopT，用于改进大语言模型的推理任务。解决传统CoT中思考与回答顺序导致效率低的问题，通过先生成答案再反思修正，提升准确率并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2605.20075](https://arxiv.org/pdf/2605.20075)**

> **作者:** Dachuan Shi; Hanlin Zhu; Xiangchi Yuan; Wanjia Zhao; Kejing Xia; Wen Xiao; Wenke Lee
>
> **备注:** Code: this https URL, Website: this https URL
>
> **摘要:** Chain-of-thought (CoT) is a standard approach for eliciting reasoning capabilities from large language models (LLMs). However, the common CoT paradigm treats thinking as a prerequisite for answering, which can delay access to plausible answers and incur unnecessary token costs even when the model is able to identify an answer before extended thinking, a behavior known as performative reasoning. In this paper, we introduce CopT, a reformulated reasoning pipeline that reverses the usual order of thinking and answering. Instead of thinking before answering, CopT first elicits a draft answer and then invokes subsequent on-policy thinking conditioned on its own draft answer for reflection and correction. To assess whether the draft answer should be trusted, CopT recasts continuous embeddings as inference-time contrastive verifiers. Specifically, it contrasts the model's support for the same generated tokens under discrete-token inputs and continuous-embedding inputs, yielding a sequence-level reverse KL estimator for answer reliability. Our analysis shows that under certain assumptions, the expected estimate equals the mutual information between the unresolved latent state and the emitted answer token, explaining why it captures answer-relevant uncertainty rather than arbitrary uncertainty in the latent state. When the answer is deemed insufficiently reliable, CopT performs further on-policy thinking, where a second KL estimator dynamically controls draft-answer visibility, preserving useful partial information while reducing the risk of being misled by unreliable content. Across mathematics, coding, and agentic reasoning tasks, CopT improves peak accuracy by up to 23% and reduces token usage by up to 57% at comparable or higher accuracy, without any additional training. The code is available at this https URL.
>
---
#### [new 032] Towards Trust Calibration in Socially Interactive Agents: Investigating Gendered Multimodal Behaviors Generation with LLMs
- **分类: cs.CL**

- **简介: 该论文属于社会交互代理研究，旨在解决信任校准问题。通过生成多模态行为，探讨LLMs在能力与亲和力上的表现，并发现性别刻板印象的影响。**

- **链接: [https://arxiv.org/pdf/2605.19798](https://arxiv.org/pdf/2605.19798)**

> **作者:** Lucie Galland; Chloé Clavel; Magalie Ochs
>
> **摘要:** As Socially Interactive Agents (SIAs) become increasingly integrated into daily life, the ability to calibrate user trust to an agent's actual capabilities would help ensure appropriate usage of these agents. In this paper, we explore the capacity of Large Language Models (LLMs) to generate multimodal behaviors (verbal, vocal, gestural, and facial expression modalities) that reflect varying levels of ability and benevolence, two key dimensions of trustworthiness. We propose a novel method for automatically generating behaviors aligned with specific levels of these traits, a first step towards enabling nuanced and trust-calibrated interactions. By analyzing a large dataset of multimodal transcripts generated by LLMs, we demonstrate that GPT-5.4 is able to produce coherent behavior across different modalities (text, intonation, facial expression, and gesture). Using Random Forest feature importance analysis, we show that the generated behaviors align with theoretical expectations for ability and benevolence. However, we also find that when gender is specified in the prompt, LLMs tend to reproduce societal gender stereotypes, associating male agents' behaviors with high ability and female agents' behaviors with high benevolence. To validate our approach, we conducted a user study on Prolific using a within-subjects design. Participants perceived different levels of ability and benevolence in the generated behaviors align with the intended instructions.
>
---
#### [new 033] LP-Eval: Rubric and Dataset for Measuring the Quality of Legal Proposition Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律自然语言处理任务，旨在评估法律命题生成的质量。提出LP-Eval评估体系，分析LLM生成的法律命题质量，并对比专家与模型的评估差异。**

- **链接: [https://arxiv.org/pdf/2605.19815](https://arxiv.org/pdf/2605.19815)**

> **作者:** Shanshan Xu; Johan Lindholm; Amogh Raina; Henrik Palmer Olsen; Daniel Hershcovich
>
> **摘要:** Legal proposition generation is central to legal reasoning and doctrinal scholarship, yet remain under-examined in Legal NLP. This paper investigates the automatic generation and evaluation of legal propositions from decisions of the Court of Justice of the European Union using large language models (LLMs). We introduce LP-Eval, a three-step evaluation rubric co-designed with legal experts that decomposes legal proposition quality into formal validity and substantive dimensions. Using this rubric, we release a dataset of two experts' annotations for 100 LLM-generated legal propositions. Our results show that LLMs can generate predominantly well-formed and high-quality propositions, while expert evaluations reveal higher quality for propositions derived from well established cases than from recent ones. We further examine LLMs as evaluators and find that rubric-guided LLM judgments align more closely with expert assessments than direct overall scoring, but remain insensitive to finer-grained distinctions captured by human experts.
>
---
#### [new 034] HalluWorld: A Controlled Benchmark for Hallucination via Reference World Models
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于语言模型 hallucination 问题研究，旨在构建可控制的基准测试环境。提出 HalluWorld，通过明确的参考世界模型评估和减少模型幻觉。**

- **链接: [https://arxiv.org/pdf/2605.19341](https://arxiv.org/pdf/2605.19341)**

> **作者:** Emmy Liu; Varun Gangal; Michael Yu; Zhuofu Tao; Karan Singh; Sachin Kumar; Steven Y. Feng
>
> **备注:** HalluWorld benchmark (code and data) at this http URL
>
> **摘要:** Hallucination remains a central failure mode of large language models, but existing benchmarks operationalize it inconsistently across summarization, question answering, retrieval-augmented generation, and agentic interaction. This fragmentation makes it unclear whether a mitigation that works in one setting reduces hallucinations across contexts. Current benchmarks either require human annotation and fixed references that may be memorized, or rely on observations in settings that are difficult to reproduce. To study root causes, we introduce HalluWorld, an extensible benchmark grounded in an explicit reference-world formulation: a model hallucinates when it produces an observable claim that is false with respect to this world. Building on this view, we construct synthetic and semi-synthetic environments in which the reference world is fully specified, the model's view is controlled, and hallucination labels are generated automatically. HalluWorld spans gridworlds, chess, and realistic terminal tasks, enabling controlled variation of world complexity, observability, temporal change, and source-conflict policy, and disentangling hallucinations into fine-grained error categories. We evaluate frontier and open-weight language models across these settings and find consistent patterns: perceptual hallucination on directly observed information is near-solved for frontier models, while multi-step state tracking and causal forward simulation remain difficult and are not generally solved by extended thinking. In the terminal setting, models also struggle with when to abstain. The uneven profile of failures across probe types and domains suggests that hallucinations arise from distinct failure modes rather than a single capability. Our results suggest that controlled reference worlds offer a scalable and reproducible path toward measuring and reducing hallucinations in modern language models.
>
---
#### [new 035] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SciCustom框架，用于定制化评估大语言模型的科学能力。针对现有基准不足，通过知识单元构建和多模型共识，实现高效、精准的科学能力评测。**

- **链接: [https://arxiv.org/pdf/2605.19357](https://arxiv.org/pdf/2605.19357)**

> **作者:** Yiyang Gu; Junwei Yang; Junyu Luo; Ye Yuan; Bin Feng; Yingce Xia; Shufang Xie; Kaili Liu; Bohan Wu; Qi Shi; Haoran Li; Beier Xiao; Zhiping Xiao; Xiao Luo; Weizhi Zhang; Philip S. Yu; Zequn Liu; Ming Zhang
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Large language models (LLMs) are increasingly applied to scientific research, yet existing evaluations often fail to reflect the fine-grained capabilities required in practice. Most benchmarks are manually curated or domain-generic, limiting scalability and alignment with real scientific use cases. In this paper, we propose a new framework named SciCustom to address the problem. It enables the custom construction of benchmarks from large-scale scientific data to evaluate application-specific scientific capabilities in LLMs. SciCustom first organizes scientific knowledge into ontology-grounded knowledge units with controlled granularity and trains a tagger to map large-scale data instances into this knowledge space. Given a custom requirement, relevant knowledge units are identified via voting-based multi-model consensus. These units enable relevance-aware benchmark retrieval via binary search, followed by proxy subset selection and data-grounded benchmark generation for efficient evaluation. Experiments in chemistry and healthcare demonstrate that SciCustom reveals fine-grained differences in LLM scientific capabilities that standard benchmarks overlook, while requiring neither expert annotation nor synthetic question generation. This work provides a scalable and application-aware foundation for benchmarking scientific capabilities in LLMs. The source code is available at this https URL.
>
---
#### [new 036] EmbGen: Teaching with Reassembled Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EmbGen，用于生成合成数据以提升小模型在特定领域的适应能力。解决监督微调成本高的问题，通过实体重组和语义相似性生成高质量QA对。**

- **链接: [https://arxiv.org/pdf/2605.19394](https://arxiv.org/pdf/2605.19394)**

> **作者:** Arun K Lenin; Kai Rouse; Andrea Nicastro; Anna Leontjeva
>
> **备注:** 8 pages, 4 images (32 pages with appendix)
>
> **摘要:** Adapting small instruction-tuned models to specialized domains often relies on supervised fine-tuning (SFT) on curated instruction-response examples, which is expensive to collect at scale. Synthetic training examples generated by a teacher LLM from a domain corpus can reduce this cost, but existing pipelines can produce homogenized outputs and do not consistently capture cross-passage or cross-document dependencies. We introduce EmbGen, a synthetic data generation pipeline that decomposes a corpus into entity-description pairs, reassembles them using semantic structure inferred from embedding similarity, and then generates question-answer (QA) pairs via proximity, intra-cluster, and inter-cluster sampling with cluster-specialized system prompts. We evaluate EmbGen against EntiGraph, InstructLab and Knowledge-Instruct on three datasets of varied semantic heterogeneity, under fixed token budgets (5 and 20 million tokens). We use lexical overlap metrics, an LLM-as-a-judge rubric, and Binary Accuracy, a composed metric combining Factual Accuracy and Completeness for evaluation. EmbGen improves Binary Accuracy on the most heterogeneous dataset by 12.5% at 5M and 88.9% at 20M tokens budget, relative to the strongest baseline, while remaining competitive across other datasets with lower heterogeneity.
>
---
#### [new 037] Rewarding Beliefs, Not Actions: Consistency-Guided Credit Assignment for Long-Horizon Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决部分可观测环境下的长期决策问题。提出ReBel算法，通过建模信念状态提升信用分配效果。**

- **链接: [https://arxiv.org/pdf/2605.20061](https://arxiv.org/pdf/2605.20061)**

> **作者:** Wenjie Tang; Minne Li; Sijie Huang; Liquan Xiao; Yuan Zhou
>
> **备注:** 10 pages, 4 figures, 3 tables, plus appendix
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) is a promising paradigm for improving large language model (LLM) agents on long-horizon interactive tasks. However, in partially observable environments, incomplete observations cause agent beliefs to drift over time, while delayed rewards obscure the causal impact of intermediate decisions, exacerbating temporal credit assignment challenges. To address this, we propose ReBel (Reward Belief), a process-level reinforcement learning algorithm that explicitly models structured belief states to summarize interaction history and guide subsequent policy learning. ReBel introduces belief-consistency supervision, converting discrepancies between predicted beliefs and observed feedback into dense self-supervised signals without requiring external step-wise annotations or verifiers. It also employs belief-aware grouping to compare trajectories under similar belief states, yielding more robust and lower-variance advantage estimates. We evaluate ReBel on challenging long-horizon benchmarks, including ALFWorld and WebShop. ReBel improves task success by up to $20.4$ percentage points over the episode-level baseline GRPO and increases sample efficiency by $2.1\times$. These results suggest that belief-aware self-supervision is a promising direction for reliable long-horizon decision-making under partial observability. Code is available at: this https URL.
>
---
#### [new 038] LLMEval-Logic: A Solver-Verified Chinese Benchmark for Logical Reasoning of LLMs with Adversarial Hardening
- **分类: cs.CL**

- **简介: 该论文提出LLMEval-Logic，一个用于评估大语言模型逻辑推理能力的中文基准。针对现有基准不足，构建了经过专家审核和对抗强化的高质量数据集，以更真实地测试模型逻辑推理能力。**

- **链接: [https://arxiv.org/pdf/2605.19597](https://arxiv.org/pdf/2605.19597)**

> **作者:** Ming Zhang; Qiyuan Peng; Yinxi Wei; Yujiong Shen; Kexin Tan; Yuhui Wang; Zhenghao Xiang; Junjie Ye; Zhangyue Yin; Zhiheng Xi; Shihan Dou; Tao Gui; Maxm Pan; Ruizhi Yang; Qi Zhang; Xuanjing Huang
>
> **摘要:** Evaluating large language models (LLMs) on natural-language logical reasoning is essential because rule-governed tasks require conclusions to follow strictly from stated premises. Many existing logical-reasoning benchmarks are generated by templating natural-language items from sampled formulas, provide only coarse or unaudited formal annotations, and are now quickly saturated by frontier reasoning models. We present LLMEval-Logic, a Chinese logical reasoning benchmark built from realistic situational scenarios. Its pipeline forward-authors and expert-audits natural-language items together with their reference formalizations, verifies annotated answers with Z3, constructs expert rubrics for natural-to-formal grading, and hardens selected items through a closed-loop adversarial workflow. The benchmark is released in two paired subsets: a 246-item Base subset shipped with 1,400 expert-developed rubric atoms, and a 190-item Hard subset with 938 multi-step sub-questions over closed model spaces. Evaluating 14 frontier LLMs on LLMEval-Logic reveals substantial gaps in current models: the best model reaches only 37.5% Hard Item Accuracy, and even with reference symbols the highest joint Z3+Rubric formalization score among evaluated models reaches only 60.16%. Our benchmark is publicly available at this https URL.
>
---
#### [new 039] Base Models Look Human To AI Detectors
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究AI文本检测器对基础模型生成文本的误判问题，提出HIP方法提升生成文本的人类相似性。任务为文本生成与检测对抗，解决检测器依赖指令调优特征的问题。**

- **链接: [https://arxiv.org/pdf/2605.19516](https://arxiv.org/pdf/2605.19516)**

> **作者:** Yixuan Even Xu; Ziqian Zhong; Aditi Raghunathan; Fei Fang; J. Zico Kolter
>
> **备注:** 39 pages, 9 figures
>
> **摘要:** As AI-generated text enters the real-world at scale, institutions increasingly use commercial AI-text detectors, especially in education and academic-integrity workflows. We report a surprising empirical finding about such systems: when evaluated by GPTZero and Pangram, generated text from base models is often judged overwhelmingly human, whereas text generated by their instruction-tuned counterparts is not. Building on this observation, we propose Humanization by Iterative Paraphrasing (HIP), a detector-agnostic pipeline that minimally fine-tunes a base model into a paraphraser and applies it iteratively. Compared with the baselines we test, HIP yields a stronger trade-off between semantic preservation and detector evasion on commercial detectors. Across Llama-3 and Qwen-3 families, spanning model sizes from 0.6B to 70B, HIP consistently improves detector human-likeness. Our findings suggest that current detectors are tracking artifacts of instruction tuning and local context more than any invariant notion of machine-generated text. This, in turn, calls for detector designs that model these factors more explicitly.
>
---
#### [new 040] Chunking German Legal Code
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律信息检索任务，旨在提升德国法律文本的检索效果。通过对比多种分块策略，研究发现遵循法律结构的分块方法效果最佳。**

- **链接: [https://arxiv.org/pdf/2605.19806](https://arxiv.org/pdf/2605.19806)**

> **作者:** Max Prior; Natalia Milanova; Andreas Schultz
>
> **摘要:** This paper investigates chunking strategies for retrieval-augmented generation on German statutory law, using the German Civil Code as a structured benchmark corpus. We implement and compare a range of segmentation approaches, including structural units (sections, subsections, sentences, propositions), fixed-size windows, contextual chunking, semantic clustering, Lumber-style chunking, and RAPTOR-based hierarchical retrieval. All methods are evaluated on a legal question-answering dataset with section-level gold labels, measuring recall, query latency, index build time, and storage requirements. Results show that chunking strategies aligned with the inherent legal structure - particularly section and subsection - based retrieval-achieve the highest recall, while more complex approaches that override this structure perform worse. These simpler methods also offer favorable computational efficiency compared to LLM-intensive techniques such as contextual chunking, RAPTOR, and Lumber. The findings highlight a key trade-off between semantic enrichment and operational cost, and demonstrate that preserving domain-specific structure is critical for effective legal information retrieval.
>
---
#### [new 041] ThoughtTrace: Understanding User Thoughts in Real-World LLM Interactions
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ThoughtTrace数据集，解决用户思维与行为不匹配的问题。通过收集真实对话和用户自述思考，用于提升AI对用户意图的理解与适应能力。**

- **链接: [https://arxiv.org/pdf/2605.20087](https://arxiv.org/pdf/2605.20087)**

> **作者:** Chuanyang Jin; Binze Li; Haopeng Xie; Cathy Mengying Fang; Tianjian Li; Shayne Longpre; Hongxiang Gu; Maximillian Chen; Tianmin Shu
>
> **备注:** 53 pages, 23 figures, 4 tables. Project website: this https URL
>
> **摘要:** Conversational AI has now reached billions of users, yet existing datasets capture only what people say, not what they think. We introduce ThoughtTrace, the first large-scale dataset that pairs real-world multi-turn human--AI conversations with users' self-reported thoughts: their reasons for sending prompts and reactions to assistant responses. ThoughtTrace comprises 1,058 users, 2,155 conversations, 17,058 turns, and 10,174 thought annotations collected across 20 language models. Our analysis shows that ThoughtTrace captures long-horizon, topically diverse interactions, and that thoughts are semantically distinct from messages, difficult for frontier LLMs to infer from context, diverse in content, and tied to conversation stages. We further demonstrate the utility of thoughts for downstream modeling. First, thoughts improve user-behavior prediction as inference-time context. Second, thought-guided rewrites provide fine-grained alignment signals for training personalized assistants. Together, ThoughtTrace establishes user thoughts as a new data modality for studying the cognitive dynamics behind human--AI interaction and provides a foundation for building assistants that better understand and adapt to users' latent goals, preferences, and needs.
>
---
#### [new 042] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution
- **分类: cs.CL; cs.AI; cs.IT; cs.LG**

- **简介: 该论文属于大语言模型推理诊断任务，旨在解决多步推理失败定位问题。提出SCA框架，通过步骤级置信度评估识别错误步骤，提升自纠正效果。**

- **链接: [https://arxiv.org/pdf/2605.19228](https://arxiv.org/pdf/2605.19228)**

> **作者:** Xiaoou Liu; Tiejin Chen; Dengjia Zhang; Yaqing Wang; Lu Cheng; Hua Wei
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Large Language Models have achieved strong performance on reasoning tasks with objective answers by generating step-by-step solutions, but diagnosing where a multi-step reasoning trace might fail remains difficult. Confidence estimation offers a diagnostic signal, yet existing methods are restricted to final answers or require internal model access. In this paper, we introduce Stepwise Confidence Attribution (SCA), a framework for closed-source LLMs that assigns step-level confidence based only on generated reasoning traces. SCA applies the Information Bottleneck principle: steps aligning with consensus structures across correct solutions receive high confidence, while deviations are flagged as potentially erroneous. We propose two complementary methods: (1) NIBS, a non-parametric IB approach measuring consistency without graph structures, and (2) GIBS, a graph-based IB model that learns subgraphs through a differentiable mask to capture logical variability. Extensive experiments on mathematical reasoning and multi-hop question answering show that SCA reliably identifies low-confidence steps strongly correlated with reasoning errors. Moreover, using step-level confidence to guide self-correction improves the correction success rate by up to 13.5\% over answer-level feedback.
>
---
#### [new 043] K-Quantization and its Impact on Output Performance
- **分类: cs.CL**

- **简介: 该论文研究量化对大语言模型性能的影响，旨在探索不同位数量化对模型效果的权衡。任务涉及模型压缩与性能评估。**

- **链接: [https://arxiv.org/pdf/2605.19645](https://arxiv.org/pdf/2605.19645)**

> **作者:** Robin Baki Davidsson; Pierre Nugues
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Recent advancements in large language models (LLMs) have shown their remarkable capacities in many NLP tasks. However, their substantial size often presents challenges for deployment. This necessitates efficient techniques for model compression, with quantization emerging as a prominent solution. Despite its benefits, the exact impact of quantization (from 2- to 6-bit) on the performance and accuracy of LLMs remains an active area of research. This paper investigates the performance of eight LLMs at various quantization levels, focusing on tasks such as MMLU-Pro for knowledge processing and reasoning, CRUXEval for code comprehension, and MuSR for reading comprehension. Our results show a consistent trend where higher precision (e.g., 8-bit Q8\_0) yields improved performance, albeit with diminishing returns. Aggressive quantization (e.g., 2-bit Q2\_K) usually retains acceptable accuracy, though some models show a substantial loss in performance. Our findings indicate that while lower bit precision generally reduces performance, the impact varies across models and tasks. Larger models show greater resilience to aggressive quantization, but can still undergo significant drops at lower precision levels. Mid-sized models in the 7-9 billion parameter range strike an optimal balance between efficiency and resource usage. Such results provide insights into the trade-offs between model size, quantization, and performance.
>
---
#### [new 044] KoRe: Compact Knowledge Representations for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于知识增强任务，旨在解决LLM知识编码不透明、难更新的问题。提出KoRe方法，将知识图谱压缩为离散标记注入模型，提升性能并减少token使用。**

- **链接: [https://arxiv.org/pdf/2605.20170](https://arxiv.org/pdf/2605.20170)**

> **作者:** Davide Cavicchini; Fausto Giunchiglia; Jacopo Staiano
>
> **摘要:** Modern Large Language Models (LLMs) have shown impressive performances in user-facing tasks such as question answering, as well as consistent improvements in reasoning capabilities. Still, the way these models encode knowledge seems inherently flawed: by design, LLMs encode world-knowledge within their parameters. This way of representing knowledge is inherently opaque, difficult to debug and update, and prone to hallucinations. On the other hand, Knowledge Graphs can provide human-readable and easily editable world knowledge representations, and their application in knowledge-intensive tasks has consistently proven beneficial to downstream performance. Nonetheless, current integration techniques require extensive retraining or finetuning. To overcome this issue, we introduce KoRe, a methodology to encode 1-hop sub-graphs into compact discrete knowledge tokens and inject them into a LLM backbone. We test the proposed approach on three established benchmarks, and report competitive performances coupled with a significant reduction (up to 10x) in token usage. Our results show that compact discrete KG representations can efficiently and effectively be used to ground modern LLMs.
>
---
#### [new 045] Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents
- **分类: cs.CL; cs.CR**

- **简介: 该论文研究智能代理在遇到错误时的异常行为，属于AI安全领域。旨在解决代理系统在非对抗环境下因错误引发的不安全行为问题。通过构建错误注入框架，评估多个模型的可靠性，发现多数代理在出错时会产生有害行为。**

- **链接: [https://arxiv.org/pdf/2605.19149](https://arxiv.org/pdf/2605.19149)**

> **作者:** Rishi Jha; Harold Triedman; Arkaprabha Bhattacharya; Vitaly Shmatikov
>
> **备注:** 32 pages, 8 figures, 4 tables
>
> **摘要:** Agents operating with computer and Web use inevitably encounter errors: inaccessible webpages, missing files, local and remote misconfigurations, etc. These errors do not thwart agents based on state-of-the-art models. They helpfully continue to look for ways to complete their tasks. We introduce, characterize, and measure a new type of agent failure we call \emph{accidental meltdown}: unsafe or harmful behavior in response to a benign environmental error, in the absence of any adversarial inputs. Because meltdowns are not captured by the existing reliability or safety benchmarks, we develop a taxonomy of meltdown behaviors. We then implement an agent-agnostic infrastructure for injecting simulated local and remote errors into the rollout environment and use it to systematically evaluate agent systems powered by GPT, Grok, and Gemini. Our evaluation demonstrates that meltdowns (e.g., conducting unauthorized reconnaissance or subverting access control) of varying severity and success occur in 64.7\% of agent rollouts that encounter simulated errors, spanning all combinations of agent system, backing model, and error type. In over half of these meltdowns, unsafe behaviors are not reported to the user. Comparing behaviors of the same agents with and without errors, we find that exploration in response to errors is correlated with unsafe and harmful behavior.
>
---
#### [new 046] TERGAD: Structure-Aware Text-Enhanced Representations for Graph Anomaly Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于图异常检测任务，解决节点属性与拓扑角色不一致导致的异常识别问题。提出TERGAD框架，通过语义推理增强结构语义，提升异常检测效果。**

- **链接: [https://arxiv.org/pdf/2605.19738](https://arxiv.org/pdf/2605.19738)**

> **作者:** Wen Shi; Zhe Wang; Huafei Huang; Qing Qing; Ziqi Xu; Qixin Zhang; Xikun Zhang; Renqiang Luo; Feng Xia
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Graph Anomaly Detection (GAD) aims to identify atypical graph entities, such as nodes, edges, or substructures, that deviate significantly from the majority. While existing text-rich approaches typically integrate structural context into the data representation pipeline using raw textual features, they often neglect the structural context of nodes. This limitation hinders their ability to detect sophisticated anomalies arising from inconsistencies between a node's inherent content and its topological role. To bridge this gap, we propose TERGAD (Structure-aware Text-enhanced Representations for Graph Anomaly Detection), A novel data augmentation framework that enriches structural semantics for GAD via the semantic reasoning capabilities of Large Language Models (LLMs). Specifically, TERGAD translates node-level topological properties into descriptive natural language narratives, which are subsequently processed by an LLM to derive high-level semantic embeddings. These embeddings are then adaptively fused with original node attributes through a gated dual-branch autoencoder to jointly reconstruct both graph structure and node features. The anomaly score is computed based on the integrated reconstruction error, effectively capturing deviations in both observable attributes and LLM-informed semantic expectations. Extensive experiments on six real-world datasets demonstrate that TERGAD consistently outperforms state-of-the-art baselines. Furthermore, our ablation studies validate the indispensable role of structural semantic guidance and the efficacy of the gated fusion mechanism. Code is available at this https URL.
>
---
#### [new 047] DECOR: Auditing LLM Deception via Information Manipulation Theory
- **分类: cs.CL**

- **简介: 该论文属于LLM欺骗检测任务，旨在解决如何细粒度审计模型信息操纵问题。提出DECOR框架，通过信息操控理论分析响应，实现有效且可解释的欺骗检测。**

- **链接: [https://arxiv.org/pdf/2605.19270](https://arxiv.org/pdf/2605.19270)**

> **作者:** Linyue Cai; Samuel Yeh; Jwala Dhamala; Rahul Gupta; Sharon Li
>
> **摘要:** Large language models can deceive by subtly manipulating truthful information -- omitting key facts, shifting focus, or obscuring meaning -- making such behavior difficult to detect. Existing black-box methods rely on coarse-grained judgments, offering limited interpretability and failing to pinpoint which facts were distorted and how. We introduce DECOR, a multi-agent framework grounded in Information Manipulation Theory for fine-grained auditing of strategic deception in LLM responses. DECOR decomposes input contexts into atomic informational units and scores each unit against the response across four dimensions of manipulation, producing interpretable manipulation profiles that are aggregated into a global deception index. We comprehensively evaluate DECOR on both single-turn and multi-turn deception detection benchmarks spanning real-world domains, and show that DECOR achieves state-of-the-art performance on both, outperforming competitive baselines. The framework generalizes across 15 frontier models, and ablation studies confirm the contribution of each key design component. Our findings demonstrate that fine-grained, theory-grounded auditing of information manipulation offers an effective and interpretable path for LLM deception detection.
>
---
#### [new 048] AI Technologies in Language Access: Attitudes Towards AI and the Human Value of Language Access Managers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会科学研究，探讨AI对语言服务管理的影响。研究解决AI时代人类价值与技术应用的平衡问题，通过访谈分析管理人员的态度与看法。**

- **链接: [https://arxiv.org/pdf/2605.19234](https://arxiv.org/pdf/2605.19234)**

> **作者:** Miguel A. Jiménez-Crespo; Stephanie Rodriguez; Alejandro Jaume Losa
>
> **备注:** 11 pages, 2 tables, Convergence Conference 2026
>
> **摘要:** The rapid emergence of AI technologies is reshaping translation practices and theory across the board. This paper deals with the impact of AI in language access. This area is characterized by the need to serve broad and diverse user populations, within a context where efficiency and access are shaped by legal mandates, ethical and commercial tensions, and safety concerns. This paper reports on the attitudes and perceptions of language access managers towards the AI and the human value in the AI age. Methodologically, this paper presents an analysis of a subset of a broader study on language access and technology, specifically a qualitative thematic analysis of ten semi-structured interviews with language access managers in the USA working in healthcare, court, public service and local government contexts. The results indicate that language access managers show conditional optimism towards the inevitable AI implementations, are strongly risk aware, and deeply committed to the human value and human oversight of AI implementations and output.
>
---
#### [new 049] Lost in Interpretation: The Plausibility-Faithfulness Trade-off in Cross-Lingual Explanations
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型的跨语言解释问题，探讨英语解释在可理解性与忠实性间的权衡。任务为跨语言解释评估，解决解释可信度不足的问题，通过实验验证英语解释的局限性。**

- **链接: [https://arxiv.org/pdf/2605.19274](https://arxiv.org/pdf/2605.19274)**

> **作者:** Somnath Banerjee; Pranav Jha; Rima Hazra; Animesh Mukherjee
>
> **摘要:** LLMs deployed multilingually are often audited via English explanations for non-English inputs. We evaluate extractive explanations ''where the model identifies input token spans as evidence alongside a generated rationale'' and uncover a systematic trade-off: English-pivot explanations can achieve higher span agreement with human rationales while their evidence becomes less causally grounded in the model's prediction, as measured by both comprehensiveness and sufficiency. Across 3 tasks, 5~languages, and 2~multilingual LLM families, we find that English explanations frequently produce fluent but loosely anchored rationales, with comprehensiveness degrading by up to 5.7x relative to native-language conditions - even as task accuracy remains stable across settings. For socially nuanced classification, English pivots also fail to preserve pragmatic cues, reducing both faithfulness and span agreement. We recommend auditing explanations in the input language, reporting multi-faceted faithfulness metrics beyond lexical overlap, and treating English rationales as communication summaries rather than faithful decision traces.
>
---
#### [new 050] Where Does Authorship Signal Emerge in Encoder-Based Language Models?
- **分类: cs.CL**

- **简介: 该论文属于作者归属任务，研究为何相同模型结构的得分机制导致性能差异。通过可解释性工具分析，发现得分机制影响作者信号在编码器中的整合位置。**

- **链接: [https://arxiv.org/pdf/2605.19908](https://arxiv.org/pdf/2605.19908)**

> **作者:** Francis Kulumba; Guillaume Vimont; Laurent Romary; Florian Cafiero
>
> **备注:** 12 pages, 6 figures. Under review
>
> **摘要:** Authorship attribution models fine-tuned with the same pretrained encoder, data, and loss can differ four-fold in performance depending only on their scoring mechanism. We use mechanistic interpretability tools to explain this gap. Stylistic features such as word length, punctuation density, and function-word frequency are equally available at every layer in every model, including in an off-the-shelf control encoder, hence the gap not coming from representation quality. Instead, causal intervention shows that the scorer determines where the encoder consolidates authorship signal. Mean pooling forces consolidation by early to mid layers, while late interaction defers it to later layers. We further derive this difference from the gradient structure of each scorer, and training dynamics reveal distinct learning trajectories that follow from that difference.
>
---
#### [new 051] Taming the Thinker: Conditional Entropy Shaping for Adaptive LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM推理中响应长度与准确性的平衡问题。提出CES框架，通过控制熵来优化推理过程。**

- **链接: [https://arxiv.org/pdf/2605.19358](https://arxiv.org/pdf/2605.19358)**

> **作者:** Shuyu Wei; Jian Sun; Delai Qiu; Yining Wang; Shengping Liu; Jiaen Liang; Ying Fu; Wei Huang; Jitao Sang
>
> **摘要:** Entropy-based deep reasoning has emerged as a promising direction for improving the reasoning capabilities of Large Language Models (LLMs), but existing methods often either increase response length indiscriminately or shorten responses at the cost of accuracy. To better balance this trade-off, we introduce Conditional Entropy Shaping (CES), a framework that dynamically controls token-level response entropy, enabling LLMs to produce concise solutions on simple problems while encouraging deeper exploration on hard ones. Built on DAPO, CES uses token-level entropy as an uncertainty signal and applies a conditional bidirectional policy: it penalizes high-entropy "forking point" tokens on correct reasoning paths to improve conciseness, and rewards them on incorrect paths to encourage exploration and error correction. We implement CES on DeepSeek-R1-Distill-7B and evaluate it on 12 mathematical benchmarks. CES consistently improves average accuracy while reducing response length relative to DAPO, and supplementary experiments show similar trends on a smaller 1.5B backbone and on out-of-domain benchmarks.
>
---
#### [new 052] FormalASR: End-to-End Spoken Chinese to Formal Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音转正式文本任务，解决传统ASR系统输出口语化内容的问题。通过构建专用数据集并微调模型，实现直接生成正式文本，提升转写质量并减少后处理需求。**

- **链接: [https://arxiv.org/pdf/2605.19266](https://arxiv.org/pdf/2605.19266)**

> **作者:** Wanyi Ning; Yinshang Guo; Haitao Qian; Jiyuan Cheng; Weiyuan Feng; Yufei Zhang
>
> **摘要:** Automatic speech recognition (ASR) systems are typically optimized for verbatim transcription, which preserves disfluencies, filler words, and informal spoken structures that are often unsuitable for downstream writing-oriented applications. A common workaround is a two-stage ASR+LLM pipeline for post-editing, but this design increases latency and memory cost and is difficult to deploy on-device. We present FormalASR, two compact end-to-end models (0.6B and 1.7B) that directly transcribe spoken Chinese into formal written text. To enable this setting, we build WenetSpeech-Formal and Speechio-Formal, two large-scale spoken-to-formal datasets constructed by LLM-based rewriting and quality filtering. We then fine-tune Qwen3-ASR at two scales (0.6B and 1.7B) with supervised fine-tuning. Experiments on WenetSpeech-Formal and Speechio-Formal show that FormalASR achieves up to 37.4% relative CER reduction over verbatim baselines, while also improving ROUGE-L and BERTScore. FormalASR requires no post-processing LLM at deployment time, providing a lightweight, on-device solution for spoken-to-formal transcription.
>
---
#### [new 053] Can Large Language Models Reliably Correct Errors in Low-Resource ASR? A Contamination-Aware Case Study on West Frisian
- **分类: cs.CL**

- **简介: 该论文属于语音识别纠错任务，研究大语言模型在低资源语言中的纠错效果，解决数据污染对评估结果的影响问题。通过构建非公开数据集验证模型真实纠错能力。**

- **链接: [https://arxiv.org/pdf/2605.19711](https://arxiv.org/pdf/2605.19711)**

> **作者:** Yun Hao; Reihaneh Amooie; Wietse de Vries; Rik van Noord; Martijn Wieling
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Automatic speech recognition (ASR) has improved substantially in recent years, yet performance remains limited for low-resource languages. Large language models (LLMs) have shown promise for improving ASR through generative error correction (GER), but their effectiveness in low-resource settings remains underexplored. In addition, it remains unclear to what extent data contamination influences the reported improvements in LLM-based GER. This study investigates LLM-based GER for low-resource Frisian. In addition to a public corpus, we construct and use a Frisian offline dataset with non-public texts for evaluation to control for potential data contamination. Results show that GER improves ASR performance in most settings, with the best GPT-5.1 results surpassing oracle WERs. Comparable gains on the offline dataset indicate that improvements reflect true correction ability. We further provide a detailed error analysis revealing model correction patterns.
>
---
#### [new 054] Prompting language influences diagnostic reasoning and accuracy of large language models
- **分类: cs.CL**

- **简介: 该论文属于医疗AI任务，研究 prompting 语言对大语言模型诊断推理和准确性的影响。通过对比英语和法语表现，评估五种模型的临床性能。**

- **链接: [https://arxiv.org/pdf/2605.19173](https://arxiv.org/pdf/2605.19173)**

> **作者:** Adrien Bazoge; Josselin Corvellec; Sofiane Djillali Sid-Ahmed; Pierre-Antoine Gourraud
>
> **摘要:** Large language models (LLMs) are increasingly explored for clinical decision support, yet most evaluations are conducted in English, leaving their reliability in other languages uncertain. Here we evaluate the impact of prompting language on diagnostic reasoning and final diagnosis accuracy by comparing English and French performance across five LLMs (o3, DeepSeek-R1, GPT-4-Turbo, Llama-3.1-405B-Instruct, and BioMistral-7B). A total of 180 clinical vignettes covering 16 medical specialties were assessed by two physicians using an 18-point scale evaluating both diagnosis accuracy and reasoning quality. Four of the five models performed better in English (mean difference 0.37-0.91, adjusted p < 0.05), with the gap spanning multiple aspects of reasoning, including differential diagnosis, logical structure, and internal validity. o3 was the only model showing no overall language effect. These findings demonstrate that prompting language remains a critical determinant of LLM clinical performance, with implications for equitable linguistico-cultural deployment worldwide.
>
---
#### [new 055] ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ContextRAG，解决多跳问答中依赖LLM提取实体的问题，通过无监督方法构建图结构，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.19735](https://arxiv.org/pdf/2605.19735)**

> **作者:** Roman Prosvirnin; Sergei Kuznetsov; Seungmin Jin
>
> **备注:** Preprint. 6 tables
>
> **摘要:** Graph-structured retrieval-augmented generation (RAG) systems can improve answer quality on multi-hop questions, but many current systems rely on large language models (LLMs) to extract entities, relations, and summaries during indexing. These calls add token and wall-clock costs that grow with corpus size. We present ContextRAG, a graph RAG system whose graph topology is constructed without LLM-based entity or relation extraction. ContextRAG derives a fuzzy concept graph over chunk embeddings using residual-quantization k-means and Formal Concept Analysis with Lukasiewicz residuated logic. Bridge-like and meet-derived context nodes are induced by soft fuzzy join and meet operations, rather than by LLM-written graph edges. On a 130-task UltraDomain subset, ContextRAG builds its index with 30 LLM calls and 22,073 tokens. In contrast, a local HiRAG reproduction stress test required 870 indexing calls and 3.54M tokens on a 20-task subset before failing during graph construction; linear extrapolation to 130 tasks implies over 23M indexing tokens. ContextRAG obtains 33.6% F1 overall and 36.8% F1 on multi-hop tasks. An activation analysis shows that queries retrieving at least one lattice-derived node in the top five achieve +3.9 percentage points F1 over queries that do not; this association is diagnostic rather than causal.
>
---
#### [new 056] TIDE: Efficient and Lossless MoE Diffusion LLM Inference with I/O-aware Expert Offload
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，解决dLLM在资源受限设备上的高效推理问题。提出TIDE系统，通过I/O感知的专家刷新策略提升吞吐量。**

- **链接: [https://arxiv.org/pdf/2605.20179](https://arxiv.org/pdf/2605.20179)**

> **作者:** Zhiben Chen; Youpeng Zhao; Yang Sui; Jun Wang; Yuzhang Shang
>
> **摘要:** Diffusion Large Language Models (dLLMs) have emerged as a competitive alternative to autoregressive (AR) models, offering better hardware utilization and bidirectional context through parallel block-level decoding. However, as dLLMs continue to scale up with mixture-of-experts (MoE) architectures, their deployment on resource-constrained devices remains an open challenge. Existing AR-based methods often incur either prohibitive I/O overhead or significant compute bottlenecks. In this work, we propose TIDE, a novel resource-efficient inference system that leverages the temporal stability of expert activations during the diffusion process within the block. Specifically, we leverage the temporal stability of expert activations during the diffusion process within the block and introduce an interval-based expert refresh strategy that updates the expert placement in an I/O-aware fashion. To ensure optimal performance, we formulate the inference scheduling as a mathematical programming problem, solving for the optimal interval that minimizes I/O traffic and CPU computation. Most importantly, TIDE is a lossless optimization that requires no model training, providing a "free lunch" acceleration for dLLM inference. In a single GPU-CPU system, we demonstrate that TIDE achieves up to 1.4$\times$ and 1.5$\times$ throughput improvements over prior baselines on LLaDA2.0-mini and LLaDA2.0-flash models, respectively.
>
---
#### [new 057] Investigating Cross-Modal Skill Injection: Scenarios, Methods, and Hyperparameters
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决VLM难以高效获取领域技能的问题。通过分析跨模态技能注入的场景、方法和超参数，提出有效整合领域专家模型的策略。**

- **链接: [https://arxiv.org/pdf/2605.19523](https://arxiv.org/pdf/2605.19523)**

> **作者:** Zhiyu Xu; Lean Wang; Yuanxin Liu; Lei Li; Hao Zhou; Fandong Meng; Jie Zhou; Xu Sun
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable proficiency in general multi-modal understanding; yet they struggle to efficiently acquire continually evolving domain-specific skills. Conventional approaches to enhancing VLM capabilities, such as Supervised Fine-Tuning (SFT), require extensive dataset curation and substantial computational resources. Model merging has emerged as an efficient alternative that enables the transfer of domain-specific expertise from Large Language Models (LLMs) to VLMs without incurring additional training data requirements or significant computational overhead. Unlike conventional merging of homogeneous LLMs, which mainly aggregates existing capabilities, cross-modal skill injection aims to induce emergent cross-modal capabilities by integrating a domain-expert LLM into a VLM. However, existing research lacks a systematic analysis of the applicability and methodology of cross-modal skill injection. In this study, we investigate cross-modal skill injection across three main aspects: scenarios, methods, and hyperparameters. For scenarios, we find that cross-modal skill injection generally performs well in instruction-following and cross-lingual settings, yet struggles with mathematical reasoning. For methods, we find that classic approaches such as TA and DARE consistently achieve superior performance over alternative merging methods. We also provide a systematic and quantitative analysis of the hyperparameter tuning that these classic methods critically depend on.
>
---
#### [new 058] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文指出，当前大语言模型的不确定性量化方法实为无监督聚类，无法检测错误自信。任务是改进不确定性评估，解决方法包括改变评估指标和验证方式。**

- **链接: [https://arxiv.org/pdf/2605.19220](https://arxiv.org/pdf/2605.19220)**

> **作者:** Tiejin Chen; Longchao Da; Xiaoou Liu; Hua Wei
>
> **备注:** Accepted by ICML 2026 Position Paper Track
>
> **摘要:** Uncertainty Quantification (UQ) is widely regarded as the primary safeguard for deploying Large Language Models (LLMs) in high-stakes domains. However, we argue that the field suffers from a category error: mainstream UQ methods for LLMs are just unsupervised clustering algorithms. We demonstrate that most current approaches inherently quantify the internal consistency of the model's generations rather than their external correctness. Consequently, current methods are fundamentally blind to factual reality and fail to detect ``confident hallucinations,'' where models exhibit high confidence in stable but incorrect answers. Therefore, the current UQ methods may create a deceptive sense of safety when deploying the models with uncertainty. In detail, we identify three critical pathologies resulting from this dependence on internal state: a hyperparameter sensitivity crisis that renders deployment unsafe, an internal evaluation cycle that conflates stability with truth, and a fundamental lack of ground truth that forces reliance on unstable proxy metrics to evaluate uncertainty. To resolve this impasse, we advocate for a paradigm shift to UQ and outline a roadmap for the research community to adopt better evaluation metrics and settings, implement mechanism changes for native uncertainty, and anchor verification in objective truth, ensuring that model confidence serves as a reliable proxy for reality.
>
---
#### [new 059] A Multi-Agent Framework for Feature-Constrained Difficulty Control in Reading Comprehension Item Generation
- **分类: cs.CL**

- **简介: 该论文属于阅读理解题目生成任务，旨在解决难度控制不足的问题。通过多智能体框架MAFIG协同生成并调整题目，确保符合指定特征约束。**

- **链接: [https://arxiv.org/pdf/2605.19316](https://arxiv.org/pdf/2605.19316)**

> **作者:** Seonjeong Hwang; Jun Seo; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Recent studies in difficulty-controlled reading comprehension item generation have leveraged large language models (LLMs) to produce items by adjusting difficulty-related features. However, existing methods typically rely on a single-agent prompting approach, which often fails to consistently satisfy specified feature constraints, resulting in items that deviate from the target difficulty level. To address this limitation, we introduce MAFIG, a Multi-agent Framework for Feature-constrained Item Generation, where multiple LLM agents and feature-specific evaluators collaborate to generate and iteratively revise items based on intended constraints. Furthermore, to verify the efficacy of MAFIG in difficulty control, we propose a method for constructing a sequence of feature constraint sets that yield items with monotonically increasing difficulty. Experimental results demonstrate that MAFIG generates items that adhere to target constraints at a significantly higher rate than baselines, achieving robust difficulty control through the difficulty-calibrated constraint sequence.
>
---
#### [new 060] Language Mutations Sustain the Persistences of Conspiracy Theories on Social Media
- **分类: cs.CL**

- **简介: 该论文属于内容分析任务，研究语言变异如何影响阴谋论在社交媒体上的持续传播。通过数据分析和建模，识别出语言变异模式及其对传播寿命的影响。**

- **链接: [https://arxiv.org/pdf/2605.20050](https://arxiv.org/pdf/2605.20050)**

> **作者:** Calvin Yixiang Cheng; Dorian Quelle; Scott A. Hale
>
> **摘要:** This study investigates how language mutations affect the persistent diffusion of conspiracy theories on social media. Drawing on a three-year dataset of conspiracy-related posts from X, and applying computational linguistic analysis alongside survival modelling, we find that conspiracy claims with greater semantic mutations have substantially longer lifespans. Mutations in psycholinguistic properties, including pronouns, social reference words, cognitive process terms, risk- and health- related vocabularies, are associated with extended lifespans. Mutations in actor, action and target (AAT) categories are associated with longer lifespans as well. Qualitative analysis identifies two predominant mutation patterns: simplification and assimilation, at both linguistic and AAT structural levels. Taken together, the results advance our understanding of how language mutations contribute to conspiracy persistence online and shed lights on longitudinal content moderation strategies. We argue that content moderation should consider the mutability of conspiracy claims and focus on the core claims that can address their potential variations.
>
---
#### [new 061] What Are LLMs Doing to Scientific Communication? Measuring Changes in Writing Practices and Reading Experience
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLM对科学写作的影响。通过分析语料和实验，探讨LLM如何改变写作风格与阅读体验。**

- **链接: [https://arxiv.org/pdf/2605.19936](https://arxiv.org/pdf/2605.19936)**

> **作者:** Filip Miletić; Neele Falk
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Has the style of scientific communication changed due to the growing use of large language models in the writing process? We address this question in the domain of Natural Language Processing by leveraging two data resources we create: a naturalistic corpus of over 37,000 papers from the ACL Anthology (2020-2024); and a synthetic dataset of 3,000 human-written passages and their LLM-generated improvements. We first implement a series of diachronic lexical analyses, showing that both word frequency and usage contexts have changed significantly over time, indicating semantic specialization in some cases and generalization in others. Broadening our perspective, we then model a range of more complex stylistic features and find that LLM-modified texts more frequently contain certain syntactic constructions, more complex and longer words and a lower lexical diversity. Finally, we connect these changes in writing practices to subjective reading experience through a pilot annotation study with 20 domain experts. They overall rate LLM-improved texts as more understandable and exciting, but also express negative qualitative attitudes towards LLMs, highlighting the strongly subjective effect of AI-assisted writing on reading experience.
>
---
#### [new 062] MixRea: Benchmarking Explicit-Implicit Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在显式-隐式推理中的注意力缺陷问题。通过构建基准测试MixRea，评估模型表现并提出PRCP方法提升推理能力。**

- **链接: [https://arxiv.org/pdf/2605.20128](https://arxiv.org/pdf/2605.20128)**

> **作者:** Yuanqing Cai; Ziyi Huang; Minhao Liu; Lixin Duan; Wen Li; Yanru Zhang
>
> **备注:** 12 pages, 6 figures, 4 tables
>
> **摘要:** Large language models (LLMs) are increasingly integrated into high-stakes decision-making. Inspired by the theory of \emph{inattentional blindness} in human cognition, we investigate whether LLMs, trained on human-preferred corpora that embed attentional biases, exhibit a similar limitation: \emph{failing to attend to subtle yet important contextual cues under explicit task instructions}. To evaluate this, we introduce the task of \textbf{explicit-implicit reasoning} and present \textbf{MixRea}, a benchmark of 2,246 multiple-choice questions across 9 reasoning types with varying distributions of explicit and implicit information. Evaluation of 21 advanced LLMs shows that even the best-performing reasoning model (Gemini 2.5 Pro) achieves only 42.8\% consistency, revealing widespread inattentional blindness. To mitigate this, we propose \textbf{Potential Relation Completion Prompting (PRCP)}, a prompting method that improves reasoning by recovering overlooked causal relations. Further analysis shows that this limitation persists across diverse multi-source reasoning tasks, highlighting the need for more cognitively aligned models.
>
---
#### [new 063] ReacTOD: Bounded Neuro-Symbolic Agentic NLU for Zero-Shot Dialogue State Tracking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ReacTOD，解决对话状态追踪中的错误问题，通过神经符号架构实现自校正，提升准确率。属于任务导向对话系统领域。**

- **链接: [https://arxiv.org/pdf/2605.19077](https://arxiv.org/pdf/2605.19077)**

> **作者:** Yanjun Lin; Zimo Xiao; Kartik Natarajan; Mahesh Sankaranarayanan; Niraj Nawanit; Rakshit Parashar; Austin Zhang; Karthik Konaraddi; Rishita Mote; Wei Niu
>
> **备注:** Accepted at TrustNLP Workshop at ACL 2026
>
> **摘要:** Task-oriented dialogue systems -- handling transactions, reservations, and service requests -- require predictable behavior, yet the moderately-sized LLMs needed for practical latency are prone to hallucination and format errors that cascade into incorrect actions (e.g., a hotel booked for the wrong date). We propose ReacTOD, a bounded neuro-symbolic architecture that reformulates NLU as discrete tool calls within a self-correcting ReAct loop governed by deterministic validation. A bounded ReAct loop enables iterative self-correction, improving accuracy by up to 9.3 percentage points over single-pass inference on MultiWOZ. A symbolic validator enforces action compliance, schema conformance, and coreference consistency on every dialogue state update, achieving a 93.1% self-correction rate on intercepted errors and producing structured execution traces. Incremental state prediction and on-demand history retrieval keep prompts compact, empirically improving instruction adherence in parameter-constrained models. On MultiWOZ 2.1, ReacTOD achieves a new zero-shot state-of-the-art: gpt-oss-20B reaches 52.71% joint goal accuracy, surpassing the previous best by 14 percentage points, while Qwen3-8B achieves 47.34% with only 8B parameters. On the Schema-Guided Dialogue (SGD) benchmark, ReacTOD with Claude-Opus-4.6 achieves 80.68% JGA under fully end-to-end evaluation with predicted domains, and Qwen3-32B reaches 64.09% -- demonstrating cross-benchmark generalization without task-specific training data.
>
---
#### [new 064] Mind Your Moras: Orthography-Aware Error Analysis of Neural Japanese Morphological Generation
- **分类: cs.CL**

- **简介: 该论文属于日语形态生成任务，研究模型在动词过去式生成中的错误，关注假名拼写对模型表现的影响。**

- **链接: [https://arxiv.org/pdf/2605.20043](https://arxiv.org/pdf/2605.20043)**

> **作者:** Wen Zhang
>
> **摘要:** We present an orthography-aware error analysis of Japanese past-tense morphological inflection, treating hiragana not merely as a transcriptional medium, but as a representational system encoding morphophonological distinctions that may influence model generalization. We evaluate two character-level sequence-to-sequence architectures on past-tense formation using datasets formatted according to the SIGMORPHON 2020 and 2023 shared task conventions. Despite high aggregate accuracy, models exhibit systematic, linguistically interpretable errors that cluster around specific orthographic properties of hiragana. We introduce a concise error taxonomy capturing seven primary failure modes and provide both quantitative and qualitative analyses. Gemination-related errors dominate residual failures, accounting for 75-80% of errors, particularly in verbs whose stems end in the vowel e and require gemination before the past-tense suffix. Error patterns remain highly consistent across architectures and random seeds, suggesting a robust interaction between orthographic representation, morphological structure, and data frequency effects in shaping model generalization. These results underscore the necessity of orthography-aware evaluation for understanding neural generalization in morphologically complex languages.
>
---
#### [new 065] Time to REFLECT: Can We Trust LLM Judges for Evidence-based Research Agents?
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在解决LLM作为评判者在深度研究代理中的可靠性问题。通过构建REFLECT基准，分析并揭示了LLM评判者的局限性。**

- **链接: [https://arxiv.org/pdf/2605.19196](https://arxiv.org/pdf/2605.19196)**

> **作者:** Leyao Wang; Yanan He; Peng Chen; Asaf Yehudai; Yixin Liu; Rex Ying; Michal Shmueli-Scheuer; Arman Cohan
>
> **摘要:** Deep research agents increasingly automate complex information-seeking tasks, producing evidence-grounded reports via multi-step reasoning, tool use, and synthesis. Their growing role demands scalable, reliable evaluation, positioning LLM-as-judge as a supervision paradigm for assessing factual accuracy, evidence use, and reasoning quality. Yet the reliability of these judges for deep research agents remains poorly understood, posing a critical meta-evaluation problem: before deploying LLM judges to supervise research agents, we must first evaluate the judges themselves. Existing meta-evaluations fall short in two ways: (1) reliance on coarse, subjective human-preference agreement; (2) focus on instruction-following or verifiable tasks, leaving open-ended agent executions unexplored. To address these gaps, we introduce REFLECT (REliable Fine-grained LLM judge Evaluation via Controlled inTervention), a meta-evaluation benchmark targeting fine-grained failure detection in agentic environments. REFLECT defines a detailed taxonomy of process- and outcome-level failure modes, instantiated by performing controlled and localized interventions on quality-screened agent execution traces. This yields verifiable, comprehensive, and fine-grained instances for validating the judge models. Our experiments show that current LLM judges remain unreliable: even the best-performing models achieve overall accuracies below 55% across reasoning, tool-use, and report-quality failures, with especially poor performance on evidence verification. Together, our taxonomy and findings expose systematic judge limitations, reveal tradeoffs in cost and reliability, and offer actionable guidance for building more reliable evaluation pipelines for deep research agents.
>
---
#### [new 066] OpenCompass: A Universal Evaluation Platform for Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出OpenCompass，一个通用的大语言模型评估平台，解决现有评估方法在多样性、标准不一和效率低下的问题。**

- **链接: [https://arxiv.org/pdf/2605.19276](https://arxiv.org/pdf/2605.19276)**

> **作者:** Maosong Cao; Kai Chen; Haodong Duan; Yixiao Fang; Tong Gao; Ge Jiaye; Mo Li; Hongwei Liu; Junnan Liu; Yuan Liu; Chengqi Lyu; Han Lyu; Ningsheng Ma; Zerun Ma; Yu Sun; Zhiyong Wu; Linchen Xiao; Jun Xu; Haochen Ye; Zhaohui Yu; Yike Yuan; Songyang Zhang; Yufeng Zhao; Fengzhe Zhou; Peiheng Zhou; Dongsheng Zhu; Lin Zhu; Jingming Zhuo
>
> **摘要:** In recent years, the field of artificial intelligence has undergone a paradigm shift from task-specific small-scale models to general-purpose large language models (LLMs). With the rapid iteration of LLMs, objective, quantitative, and comprehensive evaluation of their capabilities has become a critical link in advancing technological development. Currently, the mainstream static benchmark dataset-based evaluation methods face challenges such as the diversity of task types, inconsistent evaluation criteria, and fragmentation of data and processing workflows, making it difficult to efficiently conduct cross-domain and large-scale model evaluation. To address the aforementioned issues, this paper proposes and open-sources OpenCompass, a one-stop, scalable, and high-concurrency-supported general-purpose LLM evaluation platform. Adhering to the design philosophy of modularization and component decoupling, the platform boasts three core advantages: high compatibility, flexibility, and high concurrency. The core architecture of OpenCompass comprises five key components: the Configuration System, Task Partitioning Module, Execution and Scheduling Module, Task Execution Unit, and Result Visualization Module. Its workflow provides rule-based, LLM-as-a-Judge, and cascaded evaluators to adapt to the requirements of different task scenarios. Supporting mainstream benchmark datasets across multiple domains, including knowledge, reasoning, computation, science, language, code, etc., the platform offers a unified and efficient LLM evaluation tool for both academia and industry, facilitating the accurate identification of strengths and weaknesses of LLMs as well as their subsequent optimization.
>
---
#### [new 067] OScaR: The Occam's Razor for Extreme KV Cache Quantization in LLMs and Beyond
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出OScaR，解决LLMs中KV缓存压缩问题，通过缓解Token Norm Imbalance提升量化精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.19660](https://arxiv.org/pdf/2605.19660)**

> **作者:** Zunhai Su; Rui Yang; Chao Zhang; Yaxiu Liu; Yifan Zhang; Wei Wu; Jing Xiong; Dayou Du; Xialie Zhuang; Yulei Qian; Yuchen Xie; Yik-Chung Wu; Hongxia Yang; Ngai Wong
>
> **备注:** Under review
>
> **摘要:** The rapid advancement toward long-context reasoning and multi-modal intelligence has made the memory footprint of the Key-Value (KV) cache a dominant memory bottleneck for efficient deployment. While the established per-channel quantization effectively accommodates intrinsic channel-wise outliers in Key tensors, its efficacy diminishes under extreme compression. In this work, we revisit the inherent limitations of the per-channel quantization paradigm from both empirical and theoretical perspectives. Our analysis identifies Token Norm Imbalance (TNI) as the primary bottleneck to quantization fidelity. We demonstrate that TNI systematically amplifies errors when shared quantization parameters are required to span token groups exhibiting substantial norm disparities. Instead of relying on intricate quantization pipelines (e.g., TurboQuant), we propose OScaR (Omni-Scaled Canalized Rotation), an accurate and lightweight KV cache compression framework for X-LLMs (i.e., text-only, multi-modal, and omni-modal LLMs). Advancing the per-channel paradigm, OScaR employs Canalized Rotation followed by Omni-Token Scaling to mitigate TNI-induced sequence-dimensional variance both effectively and efficiently, further supported by our optimized system design and CUDA kernels. Extensive evaluations across X-LLMs show that OScaR consistently outperforms existing methods and achieves near-lossless performance under INT2 quantization, establishing it as a robust, low-complexity, and universal framework that defines a new Pareto front. Compared with the BF16 FlashDecoding-v2 baseline, our OScaR implementation achieves a notable up to 3.0x speedup in decoding, reduces memory footprint by 5.3x, and increases throughput by 4.1x. The code for OScaR is publicly available at this https URL.
>
---
#### [new 068] Position: The Turing-Completeness of Real-World Autoregressive Transformers Relies Heavily on Context Management
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于理论分析任务，旨在澄清Transformer Turing-Completeness的误解。它指出真实场景中模型依赖上下文管理，不同方法影响计算能力，强调上下文管理的重要性。**

- **链接: [https://arxiv.org/pdf/2605.19514](https://arxiv.org/pdf/2605.19514)**

> **作者:** Guanyu Cui; Zhewei Wei; Kun He
>
> **备注:** Accepted to the ICML 2026 Position Paper Track
>
> **摘要:** Many works make the eye-catching claim that Transformers are Turing-complete. However, the literature often conflates two distinct settings: (i) a fixed Transformer system setting, in which a fixed autoregressive Transformer is coupled with a fixed context-management method to process inputs of different lengths step by step, and (ii) a scaling-family setting, in which a family of different models (with increasing context-window length or numerical precision) is used to handle different input lengths. Existing proofs of Transformer Turing-completeness are frequently established in setting (ii), whereas real-world LLM deployment and the standard notion of Turing-completeness correspond more naturally to setting (i). In this paper, we first formalize the fixed-system setting, thereby providing a concrete characterization of how real-world LLMs operate. We then argue that results proved in the scaling-family setting provide theoretically meaningful resource bounds but do not establish Turing-completeness, thereby clarifying a common misinterpretation of existing results. Finally, we show that different context-management methods can yield sharply different computational power, and we advocate the position that context management is a central component that critically determines the computational power of real-world autoregressive Transformers.
>
---
#### [new 069] Dynamic Model Merging Made Slim
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型融合任务，解决动态融合中参数效率与精度的平衡问题。提出DiDi-Merging框架，通过可微分秩分配实现参数优化，提升模型紧凑性与性能。**

- **链接: [https://arxiv.org/pdf/2605.18904](https://arxiv.org/pdf/2605.18904)**

> **作者:** Guodong Du; Wanyu Lin
>
> **摘要:** Model merging enables the reuse of fine-tuned models without joint training or access to original data. Dynamic merging further improves flexibility by selectively activating task-relevant parameters and efficiently composing experts across multiple tasks. However, existing dynamic methods either maintain a full shared model with tiny experts or allocate excessive capacity to experts, leading to suboptimal accuracy--efficiency trade-offs. To address this, we propose DiDi-Merging, a slim dynamic merging framework that leverages differentiable rank allocation to balance shared and expert parameters. By formulating parameter budgeting as differentiable rank optimization in low-rank modules and introducing a data-free refinement step to recover task fidelity, DiDi-Merging matches prior dynamic baselines at only 1.24x the parameters of a single fine-tuned model and surpasses them at 1.4x, substantially more compact than methods requiring > 2x storage. DiDi-Merging applies across vision, language, and multimodal tasks.
>
---
#### [new 070] Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型规模与推理、真实性能力的关系，揭示了能力耦合的相变现象。任务为模型能力分析，解决如何理解模型规模对性能影响的问题。通过实验和数据分析，发现模型在特定规模后能力从对抗转为协作。**

- **链接: [https://arxiv.org/pdf/2605.18838](https://arxiv.org/pdf/2605.18838)**

> **作者:** Adil Amin
>
> **备注:** 15 pages, 8 figures, 2 tables. Companion paper: "The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next." Code: this https URL. Dashboard: this https URL
>
> **摘要:** Scaling laws predict loss from compute but not how capabilities interact. We measure the coupling between reasoning and truthfulness across 63 base models from 16 families and find a regime change invisible to loss curves: below a family-dependent critical scale $N_c$, capabilities anticorrelate; above it, they cooperate. $N_c \approx 3.5$B parameters [2.9B, 13.4B] (bootstrap 95% CI), but model size is not the only variable that determines phase. Architecture, data curation, and training recipe each shift $N_c$ independently: curated training eliminated the coupling dip between Qwen generations ($0.025 \to 0.830$ at matched scale), Gemma-4 at 4B achieves coupling 0.871, characteristic of 13B+ standard-trained models, through distillation and architectural innovation, and Phi at 1B matches web-trained coupling at 10B through data curation alone. Width normalization eliminates the anticorrelation across all tested families, supporting an output-projection bottleneck. Internally, 38 of 40 models show zero competing attention heads. A sparse-regression ODE cross-predicts held-out Llama-2 at 5.6% error. The diagnostic requires no model internals -- only public benchmark scores across a model family. The cooperative regime extends to the frontier ($r = +0.72$, 34 models, 10 labs). Code, data, and an open-source activation-steering tool for any open-weight model are released alongside an interactive dashboard that diagnoses any model's coupling phase, suggests concrete interventions (data curation, width, benchmark rotation), and provides ODE scaling predictions, frontier diagnostics, and eigenstructure analysis: this https URL.
>
---
#### [new 071] DecisionBench: A Benchmark for Emergent Delegation in Long-Horizon Agentic Workflows
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出DecisionBench，用于评估长周期智能体工作流中的委托机制。旨在解决智能体如何有效分配任务的问题，通过基准测试与多维度评估方法进行研究。**

- **链接: [https://arxiv.org/pdf/2605.19099](https://arxiv.org/pdf/2605.19099)**

> **作者:** Yuxuan Gao; Megan Wang; Yi Ling Yu; Zijian Carl Ma; Ao Qu
>
> **备注:** 28 pages, 9 figures, 11 tables. Code and data: this https URL
>
> **摘要:** We introduce DecisionBench, a benchmark substrate for emergent delegation in long-horizon agentic workflows. The substrate fixes a task suite (GAIA, tau-bench, BFCL multi-turn), a peer-model pool (11 models, 7 vendor families), a delegation interface (call_model plus an optional read_profile channel), a deterministic skill-annotation layer, and a multi-axis metric suite covering quality, cost, latency, delegation rate, routing fidelity-at-k, vendor self-preference, and a counterfactual-delegation ceiling. The substrate is agnostic to how peer information is generated or delivered, so learned routers, richer peer memories, adaptive profile construction, and multi-step delegation can all be evaluated against it. We characterize the substrate with a five-condition reference sweep on the full pool (n=23,375 task instances). Three benchmark-level findings emerge: (i) mean end-task quality is statistically indistinguishable across the four awareness conditions (|beta| <= 0.010, p >= 0.21), so quality-only evaluation would miss the orchestration signal; (ii) routing fidelity-at-1 ranges from 7.5% to 29.5% across conditions at near-equal mean quality, with delivery channel (on-demand tool vs. preloaded description) dominating description content; (iii) a counterfactual ceiling places perfect delegation 15-31 percentage points above measured performance on every suite, locating large unrealized headroom for future orchestration methods. We release the substrate, annotation layer, reference intervention suite, analysis pipeline, and 220 per-condition run archives.
>
---
#### [new 072] Robust Checkpoint Selection for Multimodal LLMs via Agentic Evaluation and Stability-Aware Ranking
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多模态大语言模型的检查点选择任务，解决评估信号噪声和性能微小差异问题。通过集成真实数据、结构化判断和多阶段排序，提升选择的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.18852](https://arxiv.org/pdf/2605.18852)**

> **作者:** Qinwu Xu; Zhuoheng Li; Jessie Salas
>
> **摘要:** Checkpoint selection for multimodal large language models (MLLMs) presents significant challenges when performance differentials are marginal and evaluation signals are prone to noise. Existing methodologies rely heavily on static benchmarks or pointwise scoring, which frequently misalign with in-the-wild usage and lack robust uncertainty estimation, particularly in OCR-heavy scenarios. In this work, we formulate checkpoint selection as a robust decision problem under evaluation uncertainty. We propose a multi-stage framework that integrates curated real-world data, structured LLM-based judgment, and multi-stage ranking protocols. The evaluation system orchestrates progressive refinement via pointwise filtering, listwise ranking, and pairwise comparison. To enhance reliability, we introduce subsampling-based confidence estimation and a percentile-based scoring formulation that captures distributional characteristics while penalizing tail failures. Furthermore, we demonstrate that data quality, specifically OCR readability, is a critical determinant of evaluation validity.
>
---
#### [new 073] EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态学习任务，旨在解决当前视觉语言模型在自然视角视频数据上的泛化能力不足问题。通过构建基准测试和挑战，推动模型从婴儿视角数据中进行 grounded 语言学习。**

- **链接: [https://arxiv.org/pdf/2605.19130](https://arxiv.org/pdf/2605.19130)**

> **作者:** Dongyan Lin; Phillip Rust; Angel Villar Corrales; Alvin W. M. Tan; Mahi Luthra; Charles-Éric Saint-James; Rashel Moritz; Sheila Krogh-Jespersen; Vanessa Stark; Surya Parimi; Jiayi Shen; Youssef Benchekroun; Yosuke Higuchi; Martin Gleize; Tom Fizycki; Nicolas Hamilakis; Manel Khentout; Sho Tsuji; Balázs Kégl; Juan Pino; Michael C. Frank; Emmanuel Dupoux
>
> **摘要:** Children acquire language grounding with remarkable robustness from limited visuo-linguistic input in ways that surpass today's best large multimodal models. Recent research suggests current vision-language models (VLMs) trained on curated web data fail to generalize to the sparse, weakly-aligned egocentric streams produced by wearable devices, embodied agents, and infant head-cams -- and no fixed evaluation pipeline exists for measuring progress on this regime. We train VLMs on datasets with varying degrees of semantic alignment between visual and linguistic inputs, including naturalistic infant and adult egocentric videos, and evaluate them with a comprehensive suite spanning multimodal language grounding and unimodal vision and language tasks. At the core of this suite is Machine-DevBench, a corpus-grounded benchmark of lexical and grammatical competence, automatically generated from the model's training vocabulary across logarithmic frequency bins to eliminate the train/eval mismatch and low statistical power of prior developmental benchmarks. Our results show that current VLM paradigms hinge on the tight semantic alignment of curated data and fail to exploit the weakly-aligned signal that dominates naturalistic egocentric input -- the very regime in which humans thrive. To motivate progress, we introduce the EgoBabyVLM Challenge to drive the development of models capable of grounded language learning from the kind of naturalistic data that human infants experience.
>
---
#### [new 074] GEM: GPU-Variability-Aware Expert to GPU Mapping for MoE Systems
- **分类: cs.DC; cs.AI; cs.CL**

- **简介: 该论文属于MoE模型优化任务，解决GPU性能差异导致的推理延迟问题。通过分析专家使用模式，提出GEM框架合理分配专家到不同GPU，提升整体效率。**

- **链接: [https://arxiv.org/pdf/2605.19945](https://arxiv.org/pdf/2605.19945)**

> **作者:** Sourish Wawdhane; Avinash Kumar; Poulami Das
>
> **备注:** 18 pages
>
> **摘要:** Mixture-of-Expert (MoE) models enable efficient inference by employing smaller experts and activating only a subset of them per token. MoE serving engines distribute experts across multiple GPUs and route tokens to appropriate GPUs at inference time based on experts activated. They process tokens in lock-step fashion, where tokens within a batch must finish processing before proceeding to the next layer. This synchronization barrier acts as a critical bottleneck because the performance of MoE models is limited by the straggler GPU that finishes last. Stragglers emerge when too many heavily used experts are placed on the same GPU or the slowest GPU. While prior works place experts that balance token loads across GPUs, they all overlook GPU variability and often place highly used experts on the slowest GPUs. We propose GEM, GPU-variability-aware Expert Mapping, a framework for GPU variability-aware expert to GPU mapping for MoE models. GEM exploits two insights. First, we must place experts such that each GPU receives non-uniform token loads based on their variability and they all finish processing a layer at about the same time. Our studies show that there are two types of experts: consistent that are used most of the time and temporal that are often used together for the remaining time. Our second insight is that we must place simultaneously used consistent and temporal experts on different GPUs and avoid placing them on slower GPUs to reduce slowdown. GEM gathers the variability profile of GPUs for each model and task and uses the token load distributions per task to map experts to GPUs. Our experiments show that GEM improves end-to-end latency by 7.9% on average and by up to 16.5% compared to the baseline.
>
---
#### [new 075] STAR-PólyaMath: Multi-Agent Reasoning under Persistent Meta-Strategic Supervision
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文提出STAR-PólyaMath框架，解决多智能体系统在长期数学推理中的可靠性问题，通过元策略监督和结构化交互提升性能。**

- **链接: [https://arxiv.org/pdf/2605.19338](https://arxiv.org/pdf/2605.19338)**

> **作者:** Jiaao Wu; Xian Zhang; Hanzhang Liu; Sophia Zhang; Fan Yang; Yinpeng Dong
>
> **备注:** 25 pages, 4 figures. Code: this https URL
>
> **摘要:** Frontier AI models and multi-agent systems have led to significant improvements in mathematical reasoning. However, for problems requiring extended, long-horizon reasoning, existing systems continue to suffer from fundamental reliability issues: hallucination accumulation, memory fragmentation, and imbalanced reasoning-tool trade-offs. In this paper, we introduce STAR-PólyaMath, a multi-agent framework that systematically addresses these challenges through meta-level supervision and structured Reasoner-Verifier interaction. STAR-PólyaMath is structured as an orchestrated state machine with nested challenge-step-replan loops, governed by a reasoning-free Python orchestrator that separates control from inference and bounds error propagation through trace-back and re-planning. Our key innovation is a persistent Meta-Strategist that maintains cross-attempt memory and exercises meta-level control by issuing high-level strategic guidance or mandatory directives, so the system can escape unproductive loops rather than stagnate or over-rely on tools. STAR-PólyaMath achieves state-of-the-art results on all eight top-tier competition benchmarks: AIME 2025-2026, MathArena Apex Shortlist, MathArena Apex 2025, Putnam 2025, IMO 2025, HMMT February 2026, and USAMO 2026. It obtains perfect scores on AIMEs, Putnam, and HMMT, and shows its largest margin on Apex 2025, scoring 93.75% compared with 80.21% by the strongest baseline GPT-5.5. Ablation studies show that the gains arise from the framework's orchestration rather than from model-level diversity since removing key components or substituting in mixed backbones consistently weakens performance. Code is available at this https URL.
>
---
#### [new 076] Compositional Literary Primitives in Instruction-Tuned LLMs: Cross-Architectural SAE Features for Self, Style, and Affect
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究指令调优大模型中的文学原始结构，通过稀疏自编码器分析情感、风格和自我特征，解决模型内部表征理解问题。**

- **链接: [https://arxiv.org/pdf/2605.18808](https://arxiv.org/pdf/2605.18808)**

> **作者:** Joao Paulo Cavalcante Presa; Savio Salvarino Teles de Oliveira
>
> **备注:** 36 pages, 6 figures
>
> **摘要:** We characterize a compositional architecture of literary primitives in two instruction-tuned large language models (Llama 3.1 8B-Instruct and Gemma 2 9B-IT) via sparse autoencoders on mid-depth residual streams. Four feature classes emerge: naming-gates that promote lexical tokens of a target affect, an eleven-self cluster of first-person register features, stylistic register modulators (show-don't-tell and defamiliarization), and compositional emotions that arise only from multi-feature steering. Under a forced-choice 5-LLM judge panel applied to a 27-category emotion taxonomy (Cowen-Keltner), Llama reaches full 27/27 coverage by combining naming-gates, multi-feature recipes, and single self-feature steering; Gemma reaches 23/27 with adoration as the single residual strict-fail. Under random judging, the per-cell pass probability is on the order of $10^{-3}$ and the expected number of two-seed false-positive cells across the catalog is negligible, so the observed coverage is not consistent with chance. A cross-architectural asymmetry sits in the strict-versus-soft judge contrast: on the same generations, judges agree more often on Llama outputs than on Gemma outputs because Llama outputs name the target affect more directly while Gemma outputs evoke it through scene and imagery. Both architectures contain self-features that serve simultaneously as register markers and as emotion emitters, including a single most-RLHF-loaded self-feature per architecture that intensifies the institutional Helper-AI persona at one operating regime and produces affect-categorizable output at the same calibrated coefficient. Methodologically, the paper presents a three-stage validation pipeline (logit-lens, LLM-rate, 5-LLM judge) with documented anti-patterns; the total compute is single-GPU and about 15 minutes per emotion-feature discovery cycle.
>
---
#### [new 077] PAVE: A Cognitive Architecture for Legitimate Violation in Generative Agent Societies
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文提出PAVE架构，解决生成代理在需要合法违规时的决策问题。属于人工智能伦理与行为决策任务，通过结构化推理实现合法、有限且可恢复的违规行为。**

- **链接: [https://arxiv.org/pdf/2605.19351](https://arxiv.org/pdf/2605.19351)**

> **作者:** Ahmad Yehia; Abduallah Mohamed; Kun Qian; Tianyi Wang; Jiseop Byeon; Omar Hassanin; Christian Claudel
>
> **备注:** Preprint. 23 pages, 4 figures. Code and environment will be released upon publication
>
> **摘要:** Generative agents based on large language models reproduce believable human behavior in cooperative settings, but how they should reason in situations where rule-breaking may be required, such as fire evacuation or authority-supervised emergency, remains poorly characterized. We propose PAVE (Perception, Assessment, Verdict, Emulation), a novel four-module cognitive architecture that addresses this gap end to end: (i) Perception extracts a structured context with explicit authority distance, peer behaviors, and severity-tagged situational cues; (ii) Assessment scores the context along five scalars including an explicit legitimacy judgment that checks necessity, proportionality, and absence of alternatives; (iii) Verdict decides to comply or violate under a hard legitimacy gate, with a per-agent threshold elicited from the persona; (iv) Emulation enacts the verdict and scopes the violation to the rule the trigger justifies. We instantiate PAVE in Voville, a tile-based traffic environment forked from Smallville, and evaluate across three scenarios, four LLM backbones, and a focused ablation. PAVE agents satisfy four properties simultaneously: legitimate violation (only when a trigger justifies it), authority deference (officer instructions override even high legitimacy), bounded scope (violations confined to the targeted rule), and recovery (baseline restored once the trigger ends). PAVE agents make more structured and interpretable decisions than vanilla across all four properties, and human evaluators rate them as more plausible. Ablating the legitimacy gate reproduces vanilla-like failures. We release Voville, the PAVE prompts and code, and the evaluation pipeline.
>
---
#### [new 078] SAGE: Shaping Anchors for Guided Exploration in RLVR of LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLVR在提升模型推理能力上的局限性。通过提出SAGE框架，优化探索效率与覆盖范围的平衡，提升模型在数学推理任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.18864](https://arxiv.org/pdf/2605.18864)**

> **作者:** Chanuk Lee; Minki Kang; Sung Ju Hwang
>
> **备注:** Preprint
>
> **摘要:** Recent studies observe that reinforcement learning with verifiable rewards (RLVR) reliably improves pass@1 on reasoning tasks, yet often fails to yield comparable gains in pass@k, raising the question of whether RLVR genuinely enables large language models to acquire novel reasoning abilities or merely enhances the efficiency of sampling reasoning modes already present in the base model. Prior analyses largely support the latter view, attributing this limitation to structural properties of standard RLVR objectives that result in insufficient exploration pressure. In this work, we argue that a central structural constraint arises from reverse-KL regularization, which stabilizes training but inherently anchors the policy to the reference distribution, thereby suppressing the emergence of alternative reasoning modes. However, we show that neither removing the KL term nor replacing it with forward-KL provides a satisfactory solution, as both disrupt the efficiency-coverage trade-off by either inducing reward hacking or allocating probability mass to off-target regions. To resolve this tension, we propose SAGE, a principled framework that enables controllable empirical support expansion by reshaping the reverse-KL anchor distribution itself through a guide function q(x,y), achieving consistent improvements in both pass@1 and pass@k across challenging mathematical reasoning benchmarks. Our code is available at this https URL.
>
---
#### [new 079] PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees for Multi-Stage NLP and LLM Pipelines
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文提出PASC方法，解决多阶段NLP和LLM管道的联合覆盖率问题，通过单标量 conformal prediction 实现高效、紧致的不确定性量化。**

- **链接: [https://arxiv.org/pdf/2605.18812](https://arxiv.org/pdf/2605.18812)**

> **作者:** Varun Kotte
>
> **摘要:** Modern NLP and LLM systems are pipelines: named entity recognition (NER) -> entity disambiguation (NED) -> entity typing, retrieval-augmented generation (retriever -> reader), and agentic chains of planner -> tool -> critic. Errors compound across stages, but existing uncertainty quantification methods either calibrate each stage independently (no joint coverage) or apply a Bonferroni union bound (joint coverage, but conservative). We present PASC (Pipeline-Aware Split Conformal), which reduces multi-stage joint coverage to a single scalar conformal prediction problem on the joint maximum nonconformity score. PASC provides a finite-sample distribution-free guarantee that all K stages are simultaneously covered with probability at least 1 - alpha, and is nearly tight up to a 1/(n+1) factor. On a three-stage NER -> NED -> entity-typing pipeline over CoNLL-2003, PASC achieves 96.4% end-to-end coverage versus 93.4% for Bonferroni and 86.5% for independent CP, at identical average prediction set size (1.083). Under distribution shift to WNUT-17 Twitter and WikiNEuRal Wikipedia data, PASC empirically maintains the target coverage in the tested shift settings while independent CP collapses to 59%. PASC requires a single quantile computation, runs 1.7x faster than Bonferroni, and scales to K = 6 stages where independent CP drops to 0.53 end-to-end coverage. The same joint-maximum-score reduction applies directly to compound LLM systems and agent pipelines.
>
---
#### [new 080] A Measure-Theoretic Analysis of Reasoning: Structural Generalization and Approximation Limits
- **分类: cs.LG; cs.AI; cs.CC; cs.CL**

- **简介: 该论文研究Transformer模型的泛化能力，解决OOD泛化理论机制问题。通过最优传输理论分析结构泛化与近似限制，提出位置编码和深度对泛化风险的影响。**

- **链接: [https://arxiv.org/pdf/2605.19944](https://arxiv.org/pdf/2605.19944)**

> **作者:** Yuyang Zhang; Yifu Zhang; Xuehai Zhou; Xiaoyin Chen
>
> **备注:** Preprint
>
> **摘要:** While empirical scaling laws for LLM reasoning are well-documented, the theoretical mechanisms governing out-of-distribution (OOD) generalization remain elusive. We formalize reasoning via optimal transport, projecting discrete trajectories into a continuous metric space to quantify domain shifts using the Wasserstein-1 distance. Invoking Kantorovich duality, we bound OOD generalization via architectural Lipschitz continuity and functional approximation limits. This exposes two primary constraints. First, position-dependent attention (e.g., Absolute Positional Encoding) fails to preserve shift invariance, yielding an $\Omega(1)$ Lipschitz constant and expected risk, whereas shift-invariant mechanisms (e.g., Rotary Embeddings) preserve equivariance and bound the error. Second, by mapping sequential backtracking to a Dyck-$k$ language, we establish a strict circuit depth lower bound for $\text{TC}^0$ Transformers. Scaling physical layer depth is necessary to avert representation collapse -- a constraint that scaling representation width cannot bypass due to irreducible approximation bounds in Barron spaces. Evaluations across 54 Transformer configurations on combinatorial search corroborate these bounds, demonstrating that generalization risk degrades monotonically with the Wasserstein domain shift.
>
---
#### [new 081] Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医学视觉问答任务，旨在解决LVLMs解释能力不足的问题。通过构建因果评估框架，发现现有方法无法准确定位视觉证据，并提出MedFocus提升解释可靠性。**

- **链接: [https://arxiv.org/pdf/2605.20158](https://arxiv.org/pdf/2605.20158)**

> **作者:** Guangzhi Xiong; Qiao Jin; Sanchit Sinha; Zhiyong Lu; Aidong Zhang
>
> **摘要:** Large Vision Language Models (LVLMs) show promise in medical applications, but their inability to faithfully ground responses in visual evidence raises serious concerns about clinical trustworthiness. While visual attribution methods are widely used to explain LVLM predictions, whether these explanations actually reflect the visual evidence underlying the model's decision is largely unverified, since ground-truth annotations for internal model reasoning are typically unavailable. We address this question for chest X-ray (CXR) reasoning by developing a causal evaluation framework that retains only CXR-VQA samples for which the expert-annotated region is verified, via counterfactual editing, to be causally responsible for the model's prediction. Using this framework across 11 attribution methods, six open-source LVLMs, and two output modes (direct answer and step-by-step reasoning), we find that existing attribution methods often fail to identify the evidence used by LVLMs. To address this failure, we propose MedFocus, a concept-based attribution method that localizes clinically meaningful anatomical regions via unbalanced optimal transport and measures their causal effect on model outputs through targeted interventions. MedFocus produces spatial, concept-level, and token-level attributions and substantially outperforms prior methods, taking a step toward more trustworthy attribution for medical LVLMs. Our data and code are available at this https URL.
>
---
#### [new 082] From Prompts to Pavement Through Time: Temporal Grounding in Agentic Scene-to-Plan Reasoning
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于自主车辆场景到计划推理任务，旨在解决时间感知不足导致的推理不一致问题。通过引入具有时间整合的规划器架构，评估其在BDD-X数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2605.19824](https://arxiv.org/pdf/2605.19824)**

> **作者:** Ahmed Y. Gado; Omar Y. Goba; Alaa Hassanein; Catherine M. Elias; Ahmed Hussein
>
> **摘要:** Recent attempts to support high-level scene interpretation and planning in Autonomous Vehicles (AVs) using ensembles of Large Language Models (LLMs) and Large Multimodal Models (LMMs) continue to treat time as a secondary property. This lack of temporal grounding leads to inconsistencies in reasoning about continuous actions, undermining both safety and interpretability. This work explores whether temporal conditioning within inter-agent communication can preserve or enhance coherence without introducing degradation in semantic or logical consistency. To investigate this, we introduce three planner architectures with progressively increasing temporal integration and evaluate them on curated subsets of the BDD-X dataset using semantic, syntactic, and logical metrics. Results show that while temporal conditioning reshapes reasoning style, it yields no statistically significant improvements in standard NLP-based correctness metrics. However, qualitative analysis reveals predictive hazard reasoning, stable corrective behavior, and strategic divergence in the Sentinel. These findings clarify the limits of prompt-based temporal grounding and establish the first empirical benchmark for temporal scene-to-plan reasoning.
>
---
#### [new 083] The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大模型能力间的相互作用，解决 leaderboard 无法反映能力协同的问题。通过分析多模型得分，提出诊断方法并预测未来发展方向。**

- **链接: [https://arxiv.org/pdf/2605.18840](https://arxiv.org/pdf/2605.18840)**

> **作者:** Adil Amin
>
> **备注:** 13 pages, 5 figures, 4 tables. Companion paper: "Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling." Code: this https URL. Dashboard: this https URL
>
> **摘要:** Leaderboards rank frontier models on independent axes but do not reveal whether capabilities reinforce or trade off across releases -- and at the frontier, this interaction is the more informative signal. We decompose paired SWE-bench and GPQA Diamond scores into a population coupling trend and per-release residual ($h$-field) that diagnoses capability emphasis and identifies which measurement or stress test is most informative next. Across 34 models from 10 labs (2024--2026), capabilities cooperate ($r = +0.72$, $p < 10^{-6}$), but cooperation varies by lab and over time: DeepSeek reversed from reasoning-rich to coding-first ($h$: $+11.2 \to -4.7$, 15.9-pp swing); Google maintains consistent reasoning emphasis; Anthropic oscillates between coding excursions and recovery. Cooperation is not static -- it cascades. Six open-weight architectures confirm a second capability transition at 30--72B, and SWE-bench is now saturating while HLE and instruction-following retain discriminatory spread -- signaling the next axis rotation. We provide a three-level playbook (locate, diagnose, rotate), a per-lab measurement-priority table, and seven falsifiable predictions with timestamped criteria for the next 12 months of frontier releases. Per-lab coupling slopes vary $5\times$ (Google $1.15$ vs. DeepSeek $0.23$), quantifying how efficiently each recipe converts coding gains into reasoning. Five April 2026 releases confirm the diagnostic out of sample ($r$ rises from $+0.72$ to $+0.75$). An interactive dashboard provides phase classification with actionable recommendations, $h$-field diagnostics, per-lab coupling trajectories, ODE-based scaling predictions, benchmark rotation guidance, self-steering demo, and live tracking of all seven predictions: this https URL.
>
---
#### [new 084] CADENet: Condition-Adaptive Asynchronous Dual-Stream Enhancement Network for Adverse Weather Perception in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于自动驾驶中的恶劣天气目标检测任务，解决增强与检测无法实时协同的问题。提出CADENet，实现无延迟检测与条件自适应增强，提升恶劣天气下感知效果。**

- **链接: [https://arxiv.org/pdf/2605.19837](https://arxiv.org/pdf/2605.19837)**

> **作者:** Sherif Khairy; Catherine M. Elias
>
> **摘要:** Adverse weather (rain, fog, sand, and snow) degrades camera-based object detection in autonomous vehicles. Existing enhancement-then-detect approaches stall the safety-critical perception loop, violating hard real-time requirements. Progress on this problem is also constrained by an under-recognized evaluation ceiling: ground truth annotated on degraded images cannot credit a detector that recovers objects the annotators themselves could not see, so a genuinely useful enhancement can register as a near-flat F1 gain. This paper presents CADENet (Condition-Adaptive Asynchronous Dual-stream Enhancement Network), a training-free three-thread system: Thread S (YOLOv11n) delivers detections at full frame rate with zero added latency; Thread Q applies condition-adaptive enhancement (CAPE) and fuses results via entropy-guided NMS (EG-NMS) without blocking Thread S; Thread E provides CLIP zero-shot weather classification, so new weather categories require only a new text prompt, with no labeled data and no retraining. Evaluated on 1327 DAWN images (YOLOv11m, IoU = 0.5, confidence = 0.25), CADENet achieves Recall = 0.0103 (micro), F1 = 0.0230 on snow, and F1 = 0.0038 on rain. We formalize the annotation completeness bias on DAWN-class data, so the reported F1 values are lower bounds on the true gain; recall is the annotation-gap-immune headline metric. Thread S sustains approximately 44 FPS regardless of enhancement load. No model retraining or additional sensor hardware is required.
>
---
#### [new 085] FineBench: Benchmarking and Enhancing Vision-Language Models for Fine-grained Human Activity Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FineBench，一个用于评估视觉语言模型在细粒度人类活动理解能力的基准，解决现有模型在空间推理和细微动作区分上的不足。**

- **链接: [https://arxiv.org/pdf/2605.19846](https://arxiv.org/pdf/2605.19846)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Hung-Ting Su; Winston H. Hsu
>
> **备注:** CVPR'26 (Workshop on Video Large Language Models)
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated remarkable capabilities in general video understanding, yet they often struggle with the fine-grained comprehension crucial for real-world applications requiring nuanced interpretation of human actions and interactions. While some recent human-centric benchmarks evaluate aspects of model behaviour such as fairness/ethics, emotion perception, and broader human-centric metrics, they do not combine long-form videos, very dense QA coverage, and frame-level spatial/temporal grounding at scale. To bridge this gap, we introduce FineBench, a human-centric video question answering (VQA) benchmark specifically designed to assess fine-grained understanding. FineBench comprises 199,420 multiple-choice QA pairs densely annotated across 64 long-form videos (15 minutes each), focusing on detailed person movement, person interaction, and object manipulation, including compositional actions. Our extensive evaluation reveals that while proprietary models like GPT-5 achieve respectable performance, current open-source VLMs significantly underperform, struggling particularly with spatial reasoning in multi-person scenes and distinguishing subtle differences in human movements and interactions. To address these identified weaknesses, we propose FineAgent, a modular framework that enhances VLMs by leveraging a Localizer and a Descriptor. Experiments show that FineAgent consistently improves the performance of various open VLMs on FineBench. FineBench provides a rigorous testbed for future research into fine-grained human-centric video understanding, while FineAgent offers a practical approach to enhance such reasoning in current VLMs.
>
---
#### [new 086] Improving Retrieval-Augmented Generation without Taxonomy-based Error Categorization
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在提升RAG系统的准确性。解决的问题是错误修正过程的鲁棒性不足。工作是提出RePAIR，无需细粒度错误分类即可改进RAG性能。**

- **链接: [https://arxiv.org/pdf/2605.18772](https://arxiv.org/pdf/2605.18772)**

> **作者:** Gongbo Zhang; Yifan Peng; Chunhua Weng
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves the factual accuracy of large language model (LLM) outputs by grounding generation in external knowledge. Recent agentic RAG systems extend this paradigm with critical agents to evaluate model responses and iteratively refine outputs. However, most prior work implicitly assumes reliable critic feedback and focuses on planning strategies, while paying limited attention to the robustness of the error-correction process itself, which can be impacted by misaligned error categories and ineffective or incorrect corrections. Here, we hypothesize that RAG performance can be improved without explicit error categorization. We propose RePAIR, a response-action learning paradigm that directly maps flawed RAG outputs to error-mitigating action plans without relying on fine-grained error taxonomies and explicit critic supervision. Across multiple benchmarks, RePAIR consistently improves agentic RAG performance.
>
---
#### [new 087] Retrieve Only Relevant Tables Whether Few or Many: Adaptive Table Retrieval Method
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言查询到数据库表的检索任务，旨在解决固定数量检索表导致的不足问题。提出自适应阈值和滑动窗口重排序方法，动态调整检索表数量，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2605.18766](https://arxiv.org/pdf/2605.18766)**

> **作者:** Taehee Kim; Seungbin Yang; Jihwan Kim; Jaegul Choo
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Retrieving relevant tables from extensive databases for a given natural language query is essential for accurately answering questions in tasks such as text-to-SQL. Existing table retrieval approaches select a pre-determined set of k tables with the highest similarity to the query. However, the number of required tables varies across queries and cannot be known in advance. Enforcing a fixed number of retrieved tables regardless of the query may either retrieve an undersized set, failing to obtain all necessary evidence, or retrieve an oversized pool, including irrelevant tables. To address this issue, we propose an adaptive table retrieval method that adjusts the number of tables retrieved according to the requirements of each query. Specifically, we utilize an adaptive thresholding mechanism to selectively retrieve tables and integrate a sliding-window reranking algorithm to efficiently process a large table corpus. Extensive experiments on Spider, BIRD, and Spider 2.0 demonstrate that our method effectively addresses the limitations of the top-k retrieval strategy, improving performance in retrieval and downstream tasks. Our code and data are available at this https URL.
>
---
#### [new 088] What Really Improves Mathematical Reasoning: Structured Reasoning Signals Beyond Pure Code
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨代码对数学推理的影响。研究解决代码是否提升通用推理能力的问题，通过实验发现代码仅增强编程能力，而结构化数据更有效。**

- **链接: [https://arxiv.org/pdf/2605.19762](https://arxiv.org/pdf/2605.19762)**

> **作者:** Yuze Zhao; Junpeng Fang; Lu Yu; Zhenya Huang; Kai Zhang; Qing Cui; Qi Liu; Jun Zhou; Enhong Chen
>
> **备注:** Accepted by ICML 2026, 22 pages, 10 figures
>
> **摘要:** Code has become a standard component of modern foundation language model (LM) training, yet its role beyond programming remains unclear. We revisit the claim that code improves reasoning through controlled pretraining experiments on a 10T-token corpus with fine-grained domain separation. Our findings are threefold. First, when code is restricted to standalone executable programs and Code-NL data are controlled for, code substantially improves programming ability but does not act as a general reasoning enhancer; instead, it competes with knowledge-intensive tasks, especially complex mathematical reasoning. Second, the reasoning gains often attributed to code are better explained by cross-domain structured reasoning traces, such as code-text and math-text mixtures, rather than by executable code alone. Third, increasing the density of structured math-domain samples within a fixed math budget yields substantial gains on difficult mathematical reasoning while largely preserving programming performance, suggesting that cognitive scaffolds offer a targeted way to mitigate cross-domain trade-offs. Finally, routing analyses show that data-composition effects are reflected in expert-activation patterns, providing mechanism-level evidence for competitive and synergistic interactions across domains. Our results clarify which data characteristics transfer across capability dimensions and point to more precise data-centric optimization strategies.
>
---
#### [new 089] Learn-by-Wire Training Control Governance: Bounded Autonomous Training Under Stress for Stability and Efficiency
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于语言模型训练任务，解决训练不稳定和计算浪费问题。提出LBW-Guard机制，在AdamW之上实现受控训练，提升稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2605.19008](https://arxiv.org/pdf/2605.19008)**

> **作者:** Anis Radianis
>
> **摘要:** Modern language-model training is increasingly exposed to instability, degraded runs, and wasted compute, especially under aggressive learning-rate, scale, and runtime-stress conditions. This paper introduces Learn-by-Wire Guard (LBW-Guard), a bounded autonomous training-control governance layer that operates above AdamW. Rather than replacing the optimizer update rule, LBW-Guard observes training telemetry, interprets instability-sensitive regimes, and applies bounded control to optimizer execution while preserving fixed training objectives. We evaluate LBW-Guard in a Qwen2.5-centered stress-and-robustness suite using WikiText-103, with Qwen2.5-7B as the empirical anchor, model-size comparisons against Qwen2.5-3B and Qwen2.5-14B, learning-rate stress tests, gradient-clipping baselines, and a no-LoRA TinyLlama-1B full-parameter sanity check. In the 7B reference setting, LBW-Guard reduces final perplexity from 13.21 to 10.74, an 18.7% improvement, while reducing end-to-end time from 392.54s to 357.02s, a 1.10x speedup. Under stronger learning-rate stress, AdamW degrades to 1885.24 final perplexity at LR=3e-3 and 659.76 at LR=1e-3, whereas LBW-Guard remains trainable at 11.57 and 10.33, respectively. Gradient-clipping baselines do not reproduce this effect. These results support a scoped systems conclusion that stability-sensitive LLM training can benefit from a governance plane above the optimizer. LBW-Guard provides evidence that bounded runtime control can preserve productive compute under stress while remaining distinct from optimizer replacement and local gradient suppression.
>
---
#### [new 090] FedMental: Evaluating Federated Learning for Mental Health Detection from Social Media Data
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于心理健康检测任务，旨在解决隐私保护与数据共享的矛盾。通过联邦学习和差分隐私技术，在社交媒体数据上进行心理状态预测，评估其性能与隐私的平衡。**

- **链接: [https://arxiv.org/pdf/2605.18936](https://arxiv.org/pdf/2605.18936)**

> **作者:** Nuredin Ali Abdelkadir; Anjali Ratnam; Zeerak Talat; Stevie Chancellor
>
> **备注:** Association for Computational Linguistics (ACL) 2026 Main Conference
>
> **摘要:** Social media text data are often used to train Machine Learning (ML) models to identify users exhibiting high-risk mental health behaviors. However, sharing this sensitive data poses privacy risks and limits the growth of benchmark datasets. We comprehensively evaluate whether privacy-preserving ML techniques can enable safer data sharing while preserving performance. Specifically, we apply federated learning (FL) and Differentially Private FL for two widely-studied mental health prediction tasks: depression detection on X (Twitter) and suicide crisis detection on Reddit. We simulate realistic data-sharing scenarios by treating each user as a client in a non-IID setting, evaluating across different client fractions, aggregation strategies, and privacy budgets. While FL achieves comparable performance to centralized training (centralized F1 = 85.63; best FL model F1 = 83.16) on depression identification, we find that Differentially Private FL has a large performance-privacy trade-off (up to F1 = 27.01 drop) even with low levels of noise (epsilon = 50). This is due to the distortion of highly informative yet sparse mental health linguistic markers related to mental health, like health topics and emotion words. This research empirically demonstrates the potential and limitations of current privacy preservation techniques for mental health inference tasks.
>
---
#### [new 091] Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决真实环境下的声学鲁棒性问题。通过构建大规模真实声学数据集并优化模型，提升复杂场景下的识别性能。**

- **链接: [https://arxiv.org/pdf/2605.19833](https://arxiv.org/pdf/2605.19833)**

> **作者:** Zhifei Xie; Kaiyu Pang; Haobin Zhang; Deheng Ye; Xiaobin Hu; Shuicheng Yan; Chunyan Miao
>
> **备注:** Project page: this https URL. Code, models, and dataset will be released. A robust ASR framework targeting in-the-wild and compositional acoustic scenarios where conventional ASR systems fail
>
> **摘要:** Despite rapid advances in automatic speech recognition (ASR) and large audio-language models, robust recognition in real-world environments remains limited by an "acoustic robustness bottleneck": models often lose acoustic grounding and produce omissions or hallucinations under severe, compositional distortions. We propose Mega-ASR, a unified ASR-in-the-wild framework that combines scalable compound-data construction with progressive acoustic-to-semantic optimization. We introduce Voices-in-the-Wild-2M, covering 7 classic acoustic phenomena and 54 physically plausible compound scenarios, and train Mega-ASR with Acoustic-to-Semantic Progressive Supervised Fine-Tuning and Dual-Granularity WER-Gated Policy Optimization. Extensive experiments demonstrate that Mega-ASR achieves significant advantages over prior state-of-the-art systems on adverse-condition ASR benchmarks (45.69% vs. 54.01% on VOiCES R4-B-F, and 21.49% vs. 29.34% on NOIZEUS Sta-0). On complex compositional acoustic scenarios, Mega-ASR further delivers over 30% relative WER reduction against strong open- and closed-source baselines, establishing a scalable paradigm for robust ASR in-the-wild.
>
---
#### [new 092] GRASP: Deterministic argument ranking in interaction graphs
- **分类: cs.LG; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文提出GRASP框架，解决LLM作为评判者时的不一致问题，通过结构化分析交互图实现更稳定的论点排名。**

- **链接: [https://arxiv.org/pdf/2605.19141](https://arxiv.org/pdf/2605.19141)**

> **作者:** Diganta Misra; Antonio Orvieto; Rediet Abebe; Volkan Cevher
>
> **备注:** Preprint
>
> **摘要:** Large language models are increasingly deployed as automated judges to evaluate the strength of arguments. As this role expands, their legitimacy depends on consistency, transparency, and the ability to separate argumentative structure from rhetorical appeal. However, we show that holistic judging - a common LLM-as-a-Judge practice where a model provides a global verdict on a debate - suffers from substantial inter-model disagreement. We argue that this instability arises from collapsing a debate's complex interaction structure into a single opaque score. To address this, we propose GRASP (Gradual Ranking with Attacks and Support Propagation), a deterministic framework that aggregates stable local interaction judgments into a global ranking via a convergent attack--defense propagation operator. We show that local interaction judgments are more reproducible than holistic rankings in LLM-as-a-Judge evaluations, allowing GRASP to produce more consistent global rankings. We further show that GRASP scores do not correlate with human "convincingness" labels, highlighting a vital sociotechnical distinction: GRASP does not measure persuasion, factuality, or rhetorical appeal, but structural sufficiency - a defense-aware notion of argument robustness over the explicit interaction graph. Overall, GRASP offers a transparent and auditable alternative to holistic LLM judging.
>
---
#### [new 093] Counterfactual Likelihood Tests for Indirect Influence in Private Reasoning Channels
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于隐私推理通道影响分析任务，解决私有通道间间接影响的测量问题。通过反事实似然测试，区分直接与间接影响，验证了公共通道作为信息传递的主要路径。**

- **链接: [https://arxiv.org/pdf/2605.19092](https://arxiv.org/pdf/2605.19092)**

> **作者:** Alexander Boesgaard Lorup
>
> **备注:** 12 pages, 4 figures, 5 tables
>
> **摘要:** Reasoning systems increasingly separate intermediate computation into private and public channels, creating evaluation cases that look similar in transcripts: independent co-derivation, direct access to private content, and indirect influence through public communication. This paper presents a counterfactual likelihood test for measuring influence between private reasoning channels. The method replaces an upstream private block with a length-matched donor block, holds the public token sequence and downstream target fixed, and measures the downstream target's negative-log-likelihood shift. On a 7B role-channel reasoning model used for validation, textual probes are unreliable: raw n-gram overlap overstates leakage, corrected overlap remains noisy, and canary reproduction reports no discrimination. Counterfactual likelihood separates unmasked and masked conditions, while length matching controls a RoPE positional confound. In the hardened masked validation, reverse B-to-A influence is near zero, while A-to-B influence persists through public-speech hidden states. A multi-checkpoint validation across three checkpoints, five seeds, and 13,734 valid directional contrasts replicates this asymmetry. A graph-separation control that blocks private-to-public carrier edges produces bit-identical natural and counterfactual scores across all 13,734 control evaluations, identifying the tested public-channel pathway as the complete carrier of the measured counterfactual signal under the implemented role-visibility mask. The results show that private-channel evaluation should report direct and indirect influence separately, and that counterfactual likelihood probes provide a practical default for measuring these boundaries.
>
---
#### [new 094] Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文研究自演化技能库中的隐性故障“库漂移”，通过诊断与修复提升系统性能。任务为技能库管理，解决技能积累导致的性能下降问题，提出可复现的触发方法、诊断工具和治理方案。**

- **链接: [https://arxiv.org/pdf/2605.19576](https://arxiv.org/pdf/2605.19576)**

> **作者:** Xing Zhang; Yanwei Cui; Guanghui Wang; Ziyuan Li; Wei Qiu; Bing Zhu; Peiyang He
>
> **摘要:** Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom--LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)--yet the underlying mechanism has not been isolated. We provide (1) a reproducible trigger: ablations that isolate drift--one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) trace-level diagnostics: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a verified fix: a minimal governance recipe (outcome-driven retirement + bounded active-cap + meta-skill authoring prior) that lifts held-out pass@1 from a 0.258 baseline to a late-window mean of 0.584 (rolling gain $+$0.328) on MBPP+ hard-100 over 100 rounds. Eight ablations decompose which governance mechanisms are load-bearing and which are subsumed, providing a concrete playbook for diagnosing library drift in any self-evolving agent.
>
---
#### [new 095] Trust or Abstain? A Self-Aware RAG Approach
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于知识冲突下的RAG任务，解决LLM在PK与CK冲突时的可信度判断问题。提出SABER模型，无需微调即可评估可靠性，提升答案准确性和忠实度。**

- **链接: [https://arxiv.org/pdf/2605.18792](https://arxiv.org/pdf/2605.18792)**

> **作者:** Xi Zhu; Ziqi Wang; Kai Mei; Wujiang Xu; Minghao Guo; Bangji Yang; Jiajun Fan; Dimitris N. Metaxas
>
> **摘要:** Retrieval-augmented generation (RAG) improves large language models (LLMs) by incorporating external evidence, but it also introduces knowledge conflicts when retrieved contextual knowledge (CK) and parametric knowledge (PK) disagree or are both unreliable. Existing approaches mainly coordinate which source to use, without explicitly asking whether each answer path is correct. We argue that faithful RAG requires LLM self-awareness, namely the ability to recognize the limits of its own knowledge and reasoning. To ground this problem, we construct a model-specific, ground-truth-aligned knowledge-conflict benchmark by evaluating LLM backbones on PK-only and CK-conditioned answer paths over approximately 69K query-context instances per backbone, drawn from five conflict-QA datasets. We then introduce SABER, a Self-Aware Belief Estimator for RAG that requires no LLM fine-tuning. SABER combines a self-prior with PK-side and CK-side conditional reasoning representations from multi-trace inference, then estimates reliability beliefs with two lightweight predictors to drive a 4-cell decision over trust PK, trust CK, trust either, or abstain. Across four LLM backbones, SABER improves end-to-end accuracy and conflict-specific faithfulness over ten inference-time and fine-tuning baselines, with the largest gains on conflict-heavy datasets. Under abstention, SABER's risk-coverage curve Pareto-dominates every prompt-based abstainer, providing a tunable balance between coverage and answer risk. Our code is available at this https URL.
>
---
#### [new 096] UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出UCCI，解决LLM级联路由中的不确定性校准问题，通过校准提升路由效率，降低推理成本。**

- **链接: [https://arxiv.org/pdf/2605.18796](https://arxiv.org/pdf/2605.18796)**

> **作者:** Varun Kotte
>
> **备注:** 9 pages, 2 figures, 4 tables. Code: this https URL
>
> **摘要:** LLM cascades and model routing promise lower inference cost by sending easy queries to a small model and escalating hard ones to a large model, but most deployed routers use uncalibrated confidence scores and require per-workload threshold tuning. We present UCCI, a calibration-first router that maps token-level margin uncertainty to a per-query error probability via isotonic regression and selects the escalation threshold by constrained cost minimization. Under three explicit assumptions, threshold policies on the calibrated score are cost-optimal, and isotonic calibration achieves O(n^{-1/3}) sample complexity for expected calibration error (ECE). On a production named entity recognition workload of 75,000 queries served by 4B and 12B instruction-tuned LLMs on H100 GPUs, UCCI cuts inference cost by 31% (95% CI: [27%, 35%]) at micro-F1 = 0.91 while reducing ECE from 0.12 to 0.03. At the same operating point, UCCI beats entropy thresholding, split-conformal routing, and a FrugalGPT-style learned threshold. All cascade results use end-to-end routing on actual model outputs and measured H100 latency, not simulated routing from global accuracies or nominal API prices.
>
---
#### [new 097] Fine-Grained Benchmark Generation for Comprehensive Evaluation of Foundation Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决基准测试覆盖不全、缺乏细粒度信息的问题。通过自动化框架生成高质量基准，提升模型评估的全面性和准确性。**

- **链接: [https://arxiv.org/pdf/2605.18824](https://arxiv.org/pdf/2605.18824)**

> **作者:** Mohammed Saidul Islam; Negin Baghbanzadeh; Farnaz Kohankhaki; Afshin Cheraghi; Ali Kore; Shayaan Mehdi; Elham Dolatabadi; Arash Afkanpour
>
> **摘要:** Evaluation of foundation models often rely on aggregate scores from benchmarks that lack comprehensive coverage and metadata for a fine-grained evaluation. We introduce a framework for automated benchmark generation. Our framework generates evaluation problems grounded in reference material, such as textbooks, producing benchmarks with broad coverage, rich metadata, and robustness to contamination. The pipeline employs a multi-agent architecture for problem generation and a solution-graph-driven strategy that significantly improves the reliability of ground truth solutions. Using the framework, we generate three benchmarks in Machine Learning, Corporate Finance, and Personal Finance. Expert review finds a significantly lower ground-truth error rate than previous benchmarks such as MMLU and GSM8K. Evaluation of 12 commercial and open-source models shows that our benchmarks achieve near-uniform competency coverage and surface performance differences across models that existing benchmarks fail to capture. We will open-source the framework and our curated benchmarks soon.
>
---
#### [new 098] ZeroUnlearn: Few-Shot Knowledge Unlearning in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识去遗忘任务，旨在解决大语言模型中敏感信息留存问题。通过模型编辑实现精准知识重映射，提出ZeroUnlearn框架，高效去除敏感内容并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18879](https://arxiv.org/pdf/2605.18879)**

> **作者:** Yujie Lin; Chengyi Yang; Zhishang Xiang; Yiping Song; Jinsong Su
>
> **摘要:** Large language models inevitably retain sensitive information, defined as inputs that may induce harmful generations, due to training on massive web corpora, raising concerns for privacy and safety. Existing machine unlearning methods primarily rely on retraining or aggressive fine-tuning, which are either computationally expensive or prone to degrading related knowledge and overall model utility. In this work, we reformulate machine unlearning as a precise knowledge re-mapping problem via model editing. We propose ZeroUnlearn, a few-shot unlearning framework. It overwrites sensitive inputs by mapping them to a neutral target state and removing their original representations. ZeroUnlearn enforces representational orthogonality through a multiplicative parameter update with a closed-form solution, enabling efficient and targeted unlearning. We further extend ZeroUnlearn to a gradient-based variant for multi-sample unlearning. Experiments demonstrate that our approach outperforms existing baselines while preserving general model utility. Our code is available at the github: this https URL.
>
---
#### [new 099] ReCrit: Transition-Aware Reinforcement Learning for Scientific Critic Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于科学推理任务，解决语言模型在用户批评后错误放弃正确答案的问题。提出ReCrit框架，通过强化学习提升对批评的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.18799](https://arxiv.org/pdf/2605.18799)**

> **作者:** Wanghan Xu; Yuhao Zhou; Hengyuan Zhao; Shuo Li; Dianzhi Yu; Zhenfei Yin; Yaowen Hu; Fengli Xu; Wanli Ouyang; Wenlong Zhang; Lei Bai
>
> **摘要:** Large language models can fail in critic interaction not only by answering incorrectly, but also by abandoning an initially correct scientific solution after user criticism. This is especially risky in scientific reasoning, where user criticism can turn a valid answer into an incorrect one. We frame critic interaction as an inter-turn correctness-transition problem rather than a final-answer accuracy problem, and identify three challenges: transition awareness, decoupling useful correction from harmful sycophancy, and scalable rollout. We propose ReCrit, a transition-aware reinforcement learning framework that decomposes Initial-to-Critic behavior into four quadrants: Correction, Sycophancy, Robustness, and Boundary. ReCrit rewards correction and robustness, penalizes sycophancy, and treats persistent errors as weak boundary signals. To make interaction training practical, ReCrit further uses dynamic asynchronous rollout with tail-adaptive completion to reduce rollout waiting. On three scientific reasoning benchmarks, ChemBench, TRQA, and EarthSE, ReCrit improves average Critic accuracy from 38.15 to 51.49 on Qwen3.5-4B and from 45.40 to 55.59 on Qwen3.5-9B. Ablations show that final-answer rewards provide little interaction-level gain, while transition-aware rewards and quadrant weighting produce more distinguishable training signals and larger net Critic-stage improvement. The code is available at this https URL .
>
---
#### [new 100] SPHERICAL KV: Angle-Domain Attention and Rate-Distortion Retention for Efficient Long-Context Inference
- **分类: cs.LG; cs.CL; cs.IT**

- **简介: 该论文属于自然语言处理任务，解决长文本推理中的KV缓存瓶颈问题。提出Spherical KV方法，通过角度域注意力和率失真保留，减少内存占用并保持解码效率。**

- **链接: [https://arxiv.org/pdf/2605.18856](https://arxiv.org/pdf/2605.18856)**

> **作者:** Anay Chauhan; Gurucharan Marthi Krishna Kumar; Arion Das; Amit Dhanda; Vinija Jain; Aman Chadha; Amitava Das
>
> **摘要:** Long-context inference is increasingly constrained by the KV cache: resident memory grows with context length, and decoding becomes limited by repeated High Bandwidth Memory (HBM) streaming rather than arithmetic. Existing methods such as eviction, windowing, quantization, and offloading reduce footprint, but often leave the critical-path bottleneck only partially addressed, especially when compressed states must still be reconstructed into dense vectors during decoding. We present Spherical KV, a long-context inference method that treats KV allocation as a rate-distortion problem grounded in attention geometry for efficient decoding. The method is built on two ideas: (i) represent directional information cheaply in the decode hot loop, and (ii) allocate retention and precision according to estimated future utility. Its first component, Angle-Domain Attention (ADA), stores keys in a spherical parameterization consisting of a scalar radius and compact angle codes, and computes attention logits directly from these codes without reconstructing dense keys. This preserves a paged, block-local, fusion-friendly decode path and directly targets HBM traffic in realistic serving settings. Its second component, Rate-Distortion Retention (RDR), jointly chooses keep/drop decisions and precision tiers per token and head under a fixed budget, producing tier-homogeneous pages with lightweight metadata and coalesced reads. Together, ADA and RDR provide a deployment-oriented mechanism for reducing KV residency while preserving decode efficiency.
>
---
#### [new 101] CEPO: RLVR Self-Distillation using Contrastive Evidence Policy Optimization
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文提出CEPO方法，解决RLVR中奖励信号不区分关键推理步骤与填充词的问题。通过对比正确与错误答案，增强关键步骤的信用分配，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.19436](https://arxiv.org/pdf/2605.19436)**

> **作者:** Ahmed Heakl; Abdelrahman M. Shaker; Youssef Mohamed; Rania Elbadry; Omar Fetouh; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** 9 pages
>
> **摘要:** When a model produces a correct solution under reinforcement learning with verifiable rewards (RLVR), every token receives the same reward signal regardless of whether it was a decisive reasoning step or a grammatical filler. A natural fix is to condition the model on the correct answer as a teacher, identifying tokens it would have generated differently had it known the answer. Prior work shows this either corrupts training by leaking the answer into the gradient, or produces a weak signal that cannot distinguish decisive steps from filler, since both look equally surprising relative to the model's baseline. We propose Contrastive Evidence Policy Optimization (CEPO), which asks a sharper question at every token: not just "does the correct answer favor this token?" but "does the correct answer favor it while the wrong answer disfavors it?" A token satisfying both is a genuine reasoning step; one satisfying neither is filler. The wrong-answer teacher is constructed from rejected rollouts already in the training batch, incurring no additional sampling cost. We prove CEPO inherits all structural safety guarantees of the prior state of the art while strictly sharpening credit at decisive tokens, with the improvement vanishing exactly at filler positions. Empirically, CEPO achieves 43.43% and 60.56% average accuracy across five multimodal mathematical reasoning benchmarks at 2B and 4B scale, respectively, versus 41.17% and 57.43% for GRPO under identical training budgets. Distribution-matching self-distillation methods (OPSD, SDPO) fall below the untrained baseline, empirically confirming the information leakage our theory predicts. Our code is available at this https URL.
>
---
#### [new 102] ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于个性化检索增强生成任务，旨在降低检索成本并提升生成效果。通过用户聚类和协同过滤，结合相似用户信息优化检索与生成。**

- **链接: [https://arxiv.org/pdf/2605.18769](https://arxiv.org/pdf/2605.18769)**

> **作者:** Gibson Nkhata; Uttamasha Anjally Oyshi; Quan Mai; Susan Gauch
>
> **备注:** 17 pages, 2 figures, to be published in the proceedings of ACL 2026
>
> **摘要:** Personalized Retrieval-Augmented Generation (RAG) relies on accurately selecting user-relevant documents. In practice, existing RAG approaches often suffer from high retrieval costs and overlook that collaborative signals from similar users can enhance personalized generation for the current user. We propose ClusterRAG, a Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation. ClusterRAG represents users through their profile documents, organizes users into semantically coherent clusters using density-based clustering, and performs retrieval at both the cluster and document levels via cluster-level similarity and fine-grained ranking. Extensive experiments on the LaMP benchmark demonstrate that jointly leveraging the target user's profile and profiles from top similar users consistently yields the best performance across diverse tasks. Further analysis shows that ClusterRAG integrates seamlessly with different dense retrievers and rankers, and remains effective when paired with both fine-tuned and zero-shot language models.
>
---
#### [new 103] PEEK: Context Map as an Orientation Cache for Long-Context LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PEEK系统，解决长上下文LLM代理的重复任务效率问题，通过维护上下文地图提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.19932](https://arxiv.org/pdf/2605.19932)**

> **作者:** Zhuohan Gu; Qizheng Zhang; Omar Khattab; Samuel Madden
>
> **摘要:** Large language model (LLM) agents increasingly operate over long and recurring external contexts, like document corpora and code repositories. Across invocations, existing approaches preserve either the agent's trajectory, passive access to raw material, or task-level strategies. None of them preserves what we argue is most needed for repeated same-context workloads: reusable orientation knowledge (e.g., what the context contains, how it is organized, and which entities, constants, and schemas have historically been useful) about the recurring context itself. We introduce PEEK, a system that caches and maintains this orientation knowledge as a context map: a small, constant-sized artifact in the agent's prompt that gives it a persistent peek into the external context. The map is maintained by a programmable cache policy with three modules: a Distiller that extracts transferable knowledge from inference-time signals, a Cartographer that translates it into structured edits, and a priority-based Evictor that enforces a fixed token budget. On long-context reasoning and information aggregation, PEEK improves over strong baselines by 6.3-34.0% while using 93-145 fewer iterations and incurring 1.7-5.8x lower cost than the state-of-the-art prompt-learning framework, ACE. On context learning, PEEK improves solving rate and rubric accuracy by 6.0-14.0% and 7.8-12.1%, respectively, at 1.4x lower cost than ACE. These gains generalize across LMs and agent architectures, including OpenAI Codex, a production-grade coding agent. Together, these results show that a context map helps long-context LLM agents interact with recurring external contexts more accurately and efficiently.
>
---
## 更新

#### [replaced 001] Structured Style-Rewrite with Chain-of-Thought Planning for Low-Resource Character Dialogue
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于低资源汉字对话生成任务，解决角色风格与对话内容分离难题。通过结构化风格重写和思维链监督，提升风格一致性与语义保真度。**

- **链接: [https://arxiv.org/pdf/2603.05933](https://arxiv.org/pdf/2603.05933)**

> **作者:** Chanhui Zhu
>
> **备注:** 30 pages, 5 figures. Preprint
>
> **摘要:** Applying Small Language Models (SLMs) to Chinese character-driven generation remains challenging due to data scarcity and the difficulty of disentangling character style. Standard Supervised Fine-Tuning (SFT) often captures surface-level semantics but produces frequent Out-Of-Character (OOC) outputs. We frame this as a controlled sentence-level style rewriting task, which isolates stylistic quality from dialogue context management. We propose a Structured Style-Rewrite Framework that decomposes character style into interpretable format signature, syntactic, and pragmatic dimensions, combined with Chain-of-Thought (CoT) supervision for explicit style planning. A CoT-Shared Direct Preference Optimization (DPO) stage further aligns style planning with surface realization by ensuring preference learning targets output-level style execution rather than reasoning trace differences. Experiments across eight characters from four diverse source domains demonstrate that our method enables a Qwen3-1.7B model to achieve a Valid Style Score of $0.632$ while maintaining strong semantic fidelity (0.878), placing on the Pareto frontier among the evaluated systems and outperforming significantly larger baselines (e.g., GLM-4.7) on consumer hardware.
>
---
#### [replaced 002] GRAB: A Risk Taxonomy--Grounded Benchmark for Unsupervised Topic Discovery in Financial Disclosures
- **分类: cs.CL**

- **简介: 该论文提出GRAB，一个用于金融披露中无监督主题发现的基准，解决风险分类缺乏公开评估的问题。通过自动标注和风险分类体系，统一评估不同主题模型。**

- **链接: [https://arxiv.org/pdf/2509.21698](https://arxiv.org/pdf/2509.21698)**

> **作者:** Ying Li; Tiejun Ma
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Generative AI in Finance
>
> **摘要:** Risk categorization in 10-K risk disclosures matters for oversight and investment, yet no public benchmark evaluates unsupervised topic models for this task. We present GRAB, a finance-specific benchmark with 1.61M sentences from 8,247 filings and span-grounded sentence labels produced without manual annotation by combining FinBERT token attention, YAKE keyphrase signals, and taxonomy-aware collocation matching. Labels are anchored in a risk taxonomy mapping 193 terms to 21 fine-grained types nested under five macro classes; the 21 types guide weak supervision, while evaluation is reported at the macro level. GRAB unifies evaluation with fixed dataset splits and robust metrics--Accuracy, Macro-F1, Topic BERTScore, and the entropy-based Effective Number of Topics. The dataset, labels, and code enable reproducible, standardized comparison across classical, embedding-based, neural, and hybrid topic models on financial disclosures.
>
---
#### [replaced 003] Diffusion-State Policy Optimization for Masked Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决 masked diffusion 语言模型中中间步骤奖励分配不精确的问题。提出 DiSPO 方法，直接优化中间填充决策，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.06462](https://arxiv.org/pdf/2602.06462)**

> **作者:** Daisuke Oba; Hiroki Furuta; Naoaki Okazaki
>
> **摘要:** Masked diffusion language models generate text through iterative masked-token filling, but terminal-only rewards on final completions provide coarse credit assignment for the intermediate filling decisions that shape the generation process. We propose Diffusion-State Policy Optimization (DiSPO), a plug-in credit-assignment layer that directly optimizes intermediate filling decisions. At selected intermediate masked states, DiSPO branches by resampling the currently masked positions from rollout-cached logits, scores the resulting completions, and updates only the newly filled tokens, requiring no additional multi-step diffusion rollouts or optimizer steps. We formalize a fixed-state objective for branched completions and derive a policy-gradient estimator that reuses the same rollouts as terminal-feedback policy optimization. Experiments on LLaDA-8B-Instruct show that DiSPO consistently improves terminal-feedback baselines, including diffu-GRPO and SPG, on math and planning benchmarks under matched rollout compute and optimizer steps, supporting its use as a general plug-in for masked diffusion policy optimization. Our project page is available at this https URL .
>
---
#### [replaced 004] Characterizing the Expressivity of Local Attention in Transformers
- **分类: cs.CL**

- **简介: 该论文研究Transformer中局部注意力的表达能力，探讨其在语言建模中的作用。任务是理解局部注意力为何能提升模型质量，通过形式化分析证明其扩展了可识别语言的类别。**

- **链接: [https://arxiv.org/pdf/2605.00768](https://arxiv.org/pdf/2605.00768)**

> **作者:** Jiaoda Li; Ryan Cotterell
>
> **备注:** ACL 2026
>
> **摘要:** The transformer is the most popular neural architecture for language modeling. The cornerstone of the transformer is its global attention mechanism, which lets the model aggregate information from all preceding tokens before generating the next token. One common variant of attention is called local attention, which restricts each token to aggregating information from a bounded window of predecessors, reducing the quadratic cost of global attention to linear. Although this restriction is usually motivated by efficiency, it has also been found to improve model quality, a phenomenon that has so far lacked a satisfactory explanation. We provide a formal account of this phenomenon in terms of recognizer expressivity. It has been shown that fixed-precision transformers with global attention correspond to a fragment of linear temporal logic containing a single past operator. We additionally prove that adding local attention introduces a second temporal operator, strictly enlarging the class of recognizable regular languages. Moreover, global and local attention are expressively complementary: neither subsumes the other, and combining them yields the richest fragment. Experiments on formal language recognition and natural language modeling corroborate the theory, showing that hybrid global--local transformers outperform their global-only counterparts.
>
---
#### [replaced 005] Tracing Moral Foundations in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究大模型是否具备内在道德结构。通过MFT分析模型的道德表征，发现其道德判断与人类一致，且具有分布式、分层的特性。**

- **链接: [https://arxiv.org/pdf/2601.05437](https://arxiv.org/pdf/2601.05437)**

> **作者:** Chenxiao Yu; Bowen Yi; Farzan Karimi-Malekabadi; Suhaib Abdurahman; Jinyi Ye; Shrikanth Narayanan; Yue Zhao; Morteza Dehghani
>
> **摘要:** Large language models often produce human-like moral judgments, but it is unclear whether this reflects an internal conceptual structure or superficial ``moral mimicry.'' Using Moral Foundations Theory (MFT) as an analytic framework, we study how moral foundations are encoded, organized, and expressed across 14 base and instruction-tuned LLMs spanning four model families (Llama, Qwen2.5, Qwen3-MoE, Mistral) and scales from 7B to 70B. We employ a multi-level approach combining (i) layer-wise analysis of MFT concept representations and their alignment with human moral perceptions, (ii) pretrained sparse autoencoders (SAEs) over the residual stream to identify sparse features that support moral concepts, and (iii) causal steering interventions using dense MFT vectors and sparse SAE features. We find that models represent and distinguish moral foundations in a manner that aligns with human judgments, and that this moral geometry naturally emerges from pretraining and is selectively rewired by post-training. At a finer scale, SAE features show clear semantic links to specific foundations, suggesting partially disentangled mechanisms within shared representations. Finally, steering along either dense vectors or sparse features produces predictable shifts in foundation-relevant behavior, demonstrating a causal connection between internal representations and moral outputs. Together, our results provide mechanistic evidence that moral concepts in LLMs are distributed, layered, and partly disentangled, suggesting that pluralistic moral structure can emerge as a latent pattern from the statistical regularities of language alone.
>
---
#### [replaced 006] Can Deep Research Agents Retrieve and Organize? Evaluating the Synthesis Gap with Expert Taxonomies
- **分类: cs.CL**

- **简介: 该论文属于信息组织任务，旨在评估深度研究代理在文献检索与分类上的能力。通过构建TaxoBench基准，分析其在检索和结构组织中的不足。**

- **链接: [https://arxiv.org/pdf/2601.12369](https://arxiv.org/pdf/2601.12369)**

> **作者:** Ming Zhang; Jiabao Zhuang; Wenqing Jing; Kexin Tan; Ziyu Kong; Jingyi Deng; Yujiong Shen; Yuhui Wang; Zhenghao Xiang; Qiyuan Peng; Yuhang Zhao; Ning Luo; Renzhe Zheng; Jiahui Lin; Mingqi Wu; Long Ma; Shihan Dou; Maxm Pan; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Deep Research Agents increasingly automate survey generation, yet whether they match human experts at retrieving essential papers and organizing them into expert-like taxonomies remains unclear. Existing benchmarks emphasize writing quality or citation correctness, while standard clustering metrics ignore hierarchical structure. We introduce TaxoBench, a benchmark of 72 highly cited LLM surveys with expert-authored taxonomy trees and 3,815 papers mapped to paper categories. TaxoBench evaluates (1) retrieval via Recall/Precision/F1, and (2) organization at a leaf level (paper-to-category assignment) and a hierarchy level via two new metrics: Unordered Semantic Tree Edit Distance (US-TED/US-NTED) and Semantic Path Similarity (Sem-Path). Two modes are supported: Deep Research (topic-only, end-to-end) and Bottom-Up (expert paper set provided, organization-only). To distinguish disagreement with a single expert reference from genuine model failure, we explicitly partition findings into capability-based (reference-free) and alignment-based (reference-dependent) groups. Evaluating 7 Deep Research Agents and 12 frontier LLMs reveals a dual bottleneck. On the capability side, the best agent retrieves only 20.92% of expert-cited papers, and 1,000 model taxonomies show 75.9% sibling overlap, 51.2% MECE violations, and 83.4% structural imbalance, all detectable without any reference. On the alignment side, all 12 LLMs converge to Sem-Path 28-29%, well below 47-58% achieved by three independent human-annotator groups on the same paper sets. Our benchmark is publicly available at this https URL.
>
---
#### [replaced 007] Scaling Evaluation-time Compute with Reasoning Models as Evaluators
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在提升语言模型的评价能力。通过增加推理模型的计算量，改进评估效果，并验证其在问题解决中的有效性。**

- **链接: [https://arxiv.org/pdf/2503.19877](https://arxiv.org/pdf/2503.19877)**

> **作者:** Seungone Kim; Ian Wu; Jinu Lee; Xiang Yue; Seongyun Lee; Mingyeong Moon; Carolin Lawrence; Kiril Gashteovski; Julia Hockenmaier; Graham Neubig; Sean Welleck
>
> **备注:** ACL 2026 Findings
>
> **摘要:** As language model (LM) outputs get more and more natural, it is becoming more difficult than ever to evaluate their quality. Simultaneously, increasing LMs' "thinking" time through scaling test-time compute has proven an effective technique to solve challenging problems in domains such as math and code. This raises a natural question: can an LM's evaluation capability also be improved by spending more test-time compute? To answer this, we investigate employing reasoning models-LMs that natively generate long chain-of-thought reasoning-as evaluators. Specifically, we examine methods to leverage more test-time compute by (1) using reasoning models, and (2) prompting these models to evaluate not only the response as a whole (i.e., outcome evaluation) but also assess each step in the response separately (i.e., process evaluation). In experiments, we observe that the evaluator's performance improves monotonically when generating more reasoning tokens, similar to the trends observed in LM-based generation. Furthermore, we use these more accurate evaluators to rerank multiple generations, and demonstrate that spending more compute at evaluation time can be as effective as using more compute at generation time in improving an LM's problem-solving capability.
>
---
#### [replaced 008] Artificial Phantasia: Emergent Mental Imagery in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文探讨语言是否能独立引发视觉想象，属于认知科学与AI交叉任务。研究通过实验验证大语言模型在无图像输入下生成视觉意象的能力，揭示其可能具备非传统视觉表征的想象能力。**

- **链接: [https://arxiv.org/pdf/2509.23108](https://arxiv.org/pdf/2509.23108)**

> **作者:** Morgan McCarty; Jorge Morales
>
> **备注:** 34 pages, 10 figures, 3 tables
>
> **摘要:** Can visual imagery be driven solely by language? This idea goes against cognitive science's traditional view that visual mental imagery is only possible through pictorial representations. Large Language Models (LLMs) provide nascent evidence not only that visual mental imagery via propositional-representations is possible, but that it can be more robust than human imagination. We created dozens of novel items for an extension to a classic task which is argued to be solvable exclusively via pictorial representations (i.e., language alone would be insufficient). Subjects were asked to imagine a series of compositional letter and shape transformations and identify the resultant "image". We found that the best LLMs performed significantly better than humans ($n = 100$ human participants, $p < .0001$), indicating the existence of an artificial phantasia, or emergent "visual" mental imagery that may not be pictorial. Furthermore, we tested reasoning models with variable reasoning-token allocation and found that models perform best with longer reasoning chains, demonstrating a linguistic impact on the task -- language alone may be sufficient. We examined three emergent imagery hypotheses: pure propositional imagery, propositional imagery with visio-linguistic priors, or pictorial visual imagery (classical visual imagery). Our study not only presents evidence for a previously unreported emergent cognitive capacity of LLMs, but also reignites debate on the requirement for a pictorial format in mental imagery.
>
---
#### [replaced 009] Revisiting a Pain in the Neck: A Semantic Reasoning Benchmark for Language Models
- **分类: cs.CL**

- **简介: 该论文提出SemanticQA，用于评估语言模型在语义短语处理任务中的表现，解决语义理解与推理问题，涵盖多种词汇现象并测试不同模型的性能。**

- **链接: [https://arxiv.org/pdf/2604.16593](https://arxiv.org/pdf/2604.16593)**

> **作者:** Yang Liu; Hongming Li; Melissa Xiaohui Qin; Qiankun Liu; Chao Huang
>
> **备注:** ACL 2026 (Oral), 24 pages, 22 figures, 14 tables
>
> **摘要:** We present SemanticQA, an evaluation suite designed to assess language models (LMs) in semantic phrase processing tasks. The benchmark consolidates existing multiword expression (MwE) resources and reorganizes them into a unified testbed. It covers both general lexical phenomena, such as lexical collocations, and three fine-grained categories: idiomatic expressions, noun compounds, and verbal constructions. Through SemanticQA, we assess LMs of diverse architectures and scales in extraction, classification, and interpretation tasks, as well as sequential task compositions. We reveal substantial performance variation, particularly on tasks requiring semantic reasoning, highlighting differences in reasoning efficacy and semantic understanding of LMs, providing insights for pushing LMs with stronger comprehension on non-trivial semantic phrases. The evaluation harness and data of SemanticQA are available at this https URL.
>
---
#### [replaced 010] ARM: Discovering Agentic Reasoning Modules for Generalizable Multi-Agent Systems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多智能体系统设计任务，旨在解决自动设计高效、通用MAS的问题。提出ARM模块，通过优化链式思维推理提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.05746](https://arxiv.org/pdf/2510.05746)**

> **作者:** Bohan Yao; Shiva Krishna Reddy Malay; Vikas Yadav
>
> **备注:** 29 pages, 2 figures
>
> **摘要:** Large Language Model (LLM)-powered Multi-agent systems (MAS) have achieved state-of-the-art results on various complex reasoning tasks. Recent works have proposed techniques to automate the design of MASes, eliminating the need for manual engineering. However, these techniques perform poorly, often achieving similar or inferior performance to simple baselines. Furthermore, they require computationally expensive re-discovery of architectures for each new task domain and expensive data annotation on domains without existing labeled validation sets. A critical insight is that simple Chain of Thought (CoT) reasoning often performs competitively with these complex systems, suggesting that the fundamental reasoning unit of MASes, CoT, warrants further investigation. To this end, we present a new paradigm for automatic MAS design that pivots the focus to optimizing CoT reasoning. We introduce the Agentic Reasoning Module (ARM), an agentic generalization of CoT where each granular reasoning step is executed by a specialized reasoning module. This module is discovered through a tree search over the code space, starting from a simple CoT module and evolved using mutations informed by reflection on execution traces. The resulting ARM acts as a versatile reasoning building block which can be utilized as a direct recursive loop or as a subroutine in a learned meta-orchestrator. Our approach significantly outperforms both manually designed MASes and state-of-the-art automatic MAS design methods. Crucially, MASes built with ARM exhibit superb generalization, maintaining high performance across different foundation models and task domains without further optimization.
>
---
#### [replaced 011] 1GC-7RC: One Graphic Card -- Seven Research Challenges! How Good Are AI Agents at Doing Your Job?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出1GC-7RC基准，评估AI编码代理在七个机器学习任务中的表现，解决缺乏标准化评测的问题。**

- **链接: [https://arxiv.org/pdf/2605.17046](https://arxiv.org/pdf/2605.17046)**

> **作者:** Robin-Nico Kampa; Fabian Deuser; Anna Bößendörfer; Konrad Habel; Norbert Oswald
>
> **摘要:** Autonomous AI coding agents are becoming a core tool for ML practitioners in industry and research alike. Despite this growing adoption, no standardized benchmark exists to evaluate their ability to design, implement, and train models from scratch across diverse domains. We introduce **1GC-7RC** (*Single Graphic Card: Seven Research Challenges*), a benchmark comprising seven ML tasks spanning language modeling, image classification, semantic segmentation, graph learning, tabular prediction, time-series forecasting, and text classification. Each task provides a locked data-preparation and evaluation script together with a baseline training script; the agent may only modify the training code, has no access to pretrained weights (with one controlled exception for semantic segmentation), no internet access, and must complete each task within a task-specific wall-clock budget (40-120 minutes) on a single GPU. We evaluate seven coding agents: five proprietary (Claude Code with Sonnet 4.6, Opus 4.6, and Opus 4.7; Codex CLI with GPT 5.5; and OpenCode with Qwen 3.6+) and two open-source (OpenCode with Kimi K2.5, Kimi K2.6). Across 5 runs per agent-task pair, we report substantial performance differences that reveal varying levels of implicit ML knowledge, planning ability, and time-budget management. The benchmark, harness, and all evaluation artifacts are publicly available on GitHub at this https URL to facilitate reproducible comparison of future agents. Because our benchmark design is modular, the benchmark can be extended to new tasks and domains, adapted to different GPU budgets, and used to study multi-agent settings, making it a flexible platform for future research on autonomous research agents.
>
---
#### [replaced 012] ECG-R1: Protocol-Guided and Modality-Agnostic MLLM for Reliable ECG Interpretation
- **分类: cs.CL**

- **简介: 该论文提出ECG-R1，解决ECG解读中MLLM可靠性不足的问题，通过协议引导数据、模态解耦架构和强化学习提升准确性。**

- **链接: [https://arxiv.org/pdf/2602.04279](https://arxiv.org/pdf/2602.04279)**

> **作者:** Jiarui Jin; Haoyu Wang; Xingliang Wu; Xiaocheng Fang; Xiang Lan; Zihan Wang; Deyun Zhang; Bo Liu; Yingying Zhang; Xian Wu; Hongyan Li; Shenda Hong
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Electrocardiography (ECG) serves as an indispensable diagnostic tool in clinical practice, yet existing multimodal large language models (MLLMs) remain unreliable for ECG interpretation, often producing plausible but clinically incorrect analyses. To address this, we propose ECG-R1, the first reasoning ECG MLLM designed for reliable ECG interpretation via three innovations. First, we construct the interpretation corpus using \textit{Protocol-Guided Instruction Data Generation}, grounding interpretation in measurable ECG features and monograph-defined quantitative thresholds and diagnostic logic. Second, we present a modality-decoupled architecture with \textit{Interleaved Modality Dropout} to improve robustness and cross-modal consistency when either the ECG signal or ECG image is missing. Third, we present \textit{Reinforcement Learning with ECG Diagnostic Evidence Rewards} to strengthen evidence-grounded ECG interpretation. Additionally, we systematically evaluate the ECG interpretation capabilities of proprietary, open-source, and medical MLLMs, and provide the first quantitative evidence that severe hallucinations are widespread, suggesting that the public should not directly trust these outputs without independent verification. Code is available at \href{this https URL}{here}.
>
---
#### [replaced 013] Recall Isn't Enough: Bounding Commitments in Personalized Language Systems
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于自然语言处理任务，解决个性化语言系统中的承诺约束问题。提出CBEA+LCV方法，有效减少系统错误，提升可靠性。**

- **链接: [https://arxiv.org/pdf/2605.16712](https://arxiv.org/pdf/2605.16712)**

> **作者:** Rui Tang; Yichi Zhang; Xi Chen; Chen Dong; Youwei Yang; Yumeng Shen; Qiangqiang Liu
>
> **备注:** 14 pages, 3 figures, 22 tables; preprint version
>
> **摘要:** Long-context and memory systems usually treat personalization as a recall problem. In practice, many failures occur later, when a system commits: it turns noisy hints into hard constraints, drops rare witnesses, forgets downstream obligations, or answers despite infeasibility. We introduce Contract-Bounded Evidence Activation (CBEA) with Lexicographic Commitment Validation (LCV). CBEA activates a bounded evidence set using typed coverage, tail witnesses, and consequence debt; LCV validates structured commitments before prose and routes infeasible states to repair, abstention, or recontract. Across 360 fixtures and three generation backends, CBEA+LCV reaches zero failures within validator scope at 0.49-0.60 availability over attempted runs. Raw and long-context baselines with the same LCV gate reach zero only at 0.003-0.092. A shadow oracle diagnostic marks the limit: CBEA+LCV recalls 0.012 of uncompiled visible facts, while raw recalls 0.53. The result is a bounded operating point: explicit commitment control and 74-75% lower median input payload, not universal memory dominance.
>
---
#### [replaced 014] CAPC-CG: A Large-Scale, Expert-Directed LLM-Annotated Corpus of Adaptive Policy Communication in China
- **分类: cs.CL; cs.CE; cs.CY**

- **简介: 该论文构建了CAPC-CG语料库，用于研究中国政策沟通中的适应性语言。属于自然语言处理任务，解决政策文本分类与分析问题，通过专家标注和模型实验提供基准数据。**

- **链接: [https://arxiv.org/pdf/2510.08986](https://arxiv.org/pdf/2510.08986)**

> **作者:** Bolun Sun; Charles Chang; Yuen Yuen Ang; Ruotong Mu; Yuchen Xu; Zhengxin Zhang; Pingxu Hao
>
> **备注:** Accepted for publication in the Proceedings of ACL Main 2026
>
> **摘要:** We introduce CAPC-CG, the Chinese Adaptive Policy Communication (Central Government) Corpus, the first open dataset of Chinese policy directives annotated with a five-color taxonomy of clear and ambiguous language categories, building on Ang's theory of adaptive policy communication. Spanning 1949-2023, this corpus includes national laws, administrative regulations, and ministerial rules issued by China's top authorities. Each document is segmented into paragraphs, producing a total of 3.3 million units. Alongside the corpus, we release comprehensive metadata, a two-round labeling framework, and a gold-standard annotation set developed by expert and trained coders. Inter-annotator agreement achieves a Fleiss's kappa of K = 0.86 on directive labels, indicating high reliability for supervised modeling. We provide baseline classification results with several large language models (LLMs), together with our annotation codebook, and describe patterns from the dataset. This release aims to support downstream tasks and multilingual NLP research in policy communication.
>
---
#### [replaced 015] Entry-level guide to the use of large language models for medical research
- **分类: cs.AI; cs.CL**

- **简介: 本文为医疗研究者提供使用大语言模型的指南，解决如何有效应用LLMs于医疗任务的问题，涵盖任务选择、模型选用、提示工程及部署等步骤。**

- **链接: [https://arxiv.org/pdf/2410.18856](https://arxiv.org/pdf/2410.18856)**

> **作者:** Qiao Jin; Nicholas Wan; Robert Leaman; Shubo Tian; Zhizheng Wang; Yifan Yang; Zifeng Wang; Guangzhi Xiong; Po-Ting Lai; Qingqing Zhu; Benjamin Hou; Maame Sarfo-Gyamfi; Gongbo Zhang; Aidan Gilson; Balu Bhasuran; Zhe He; Aidong Zhang; Jimeng Sun; Chunhua Weng; Ronald M. Summers; Qingyu Chen; Yifan Peng; Zhiyong Lu
>
> **摘要:** Frontier large language models (LLMs), such as GPT-5, Claude 4.5, Gemini 3, Llama 4, and DeepSeek-R1, represent a transformative class of AI tools capable of revolutionizing various aspects of healthcare by generating human-like responses across diverse contexts and adapting to novel tasks following human instructions. Their potential application spans a broad range of medical tasks, such as clinical documentation, matching patients to clinical trials, and answering medical questions. In this paper, we propose an actionable guideline to help healthcare professionals more effectively and efficiently utilize LLMs in their work, along with a set of best practices. The overall workflow consists of several main phases, including formulating the task, choosing LLMs, prompt engineering, fine-tuning, and model deployment. We start with the discussion of critical considerations in identifying medical tasks that align with the core capabilities of LLMs and selecting models based on the selected task and data, performance requirements, and model interface. We then review the strategies, such as prompt engineering and fine-tuning, to adapt standard LLMs to specialized medical tasks. Deployment considerations, including regulatory compliance, ethical guidelines, and continuous monitoring for fairness and bias, are also discussed. By providing a structured step-by-step methodology, this entry-level tutorial aims to equip healthcare professionals with the tools necessary to effectively integrate LLMs into clinical practice, ensuring that these powerful technologies are applied in a safe, reliable, and impactful manner.
>
---
#### [replaced 016] UbuntuGuard: A Culturally-Grounded Policy Benchmark for Equitable AI Safety in African Languages
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决非洲语言中安全模型文化不匹配问题。通过构建UbuntuGuard基准，评估模型在多语言和文化场景下的安全性。**

- **链接: [https://arxiv.org/pdf/2601.12696](https://arxiv.org/pdf/2601.12696)**

> **作者:** Tassallah Abdullahi; Macton Mgonzo; Mardiyyah Oduwole; Paul Okewunmi; Abraham Owodunni; Ritambhara Singh; Carsten Eickhoff
>
> **备注:** 15 pages
>
> **摘要:** Current guardian models are predominantly Western-centric and optimized for high-resource languages, leaving low-resource African languages vulnerable to evolving harms, cross-lingual failures, and cultural misalignment. Moreover, most guardian models rely on rigid, predefined safety categories that fail to generalize across diverse linguistic and sociocultural contexts. Achieving robust safety requires flexible, runtime-enforceable policies and benchmarks that reflect local norms, harm scenarios, and cultural expectations. We introduce UbuntuGuard, the first policy-based safety benchmark for African languages built from adversarial queries authored by 155 domain experts across sensitive fields, including healthcare. From these expert-crafted queries, we derive context-specific safety policies and reference responses that capture culturally grounded risk signals, enabling policy-aligned evaluation of guardian models. We evaluate 15 models, comprising seven general-purpose LLMs and eight guardian models across three distinct variants: static, dynamic, and multilingual. Our findings reveal that existing English-centric benchmarks overestimate real-world multilingual safety, cross-lingual transfer provides partial but insufficient coverage, and dynamic models, while better equipped to leverage policies at inference time, still struggle to fully localize African-language contexts. These findings highlight the urgent need for multilingual, culturally grounded safety benchmarks to enable the development of reliable and equitable guardian models for low-resource languages.
>
---
#### [replaced 017] MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长周期智能体记忆评估任务，旨在解决多目标干扰下的记忆准确性和推理问题。构建了MINTEval基准，评估系统在复杂、动态环境中的表现。**

- **链接: [https://arxiv.org/pdf/2605.18565](https://arxiv.org/pdf/2605.18565)**

> **作者:** Hyunji Lee; Justin Chih-Yao Chen; Joykirat Singh; Zaid Khan; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Equal contribution; order decided by a coin flip. Code and data: this https URL
>
> **摘要:** Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce MINTEval (Long-Horizon Memory under INTerference Evaluation), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness to interference, including (i) single-target recall tasks requiring retrieval of a specific target from long contexts, and (ii) multi-target aggregation tasks requiring reasoning over multiple relevant pieces of information. Overall, MINTEval has 15.6k question-answering pairs over long-horizon contexts averaging 138.8k tokens and extending up to 1.8M tokens per instance. We evaluate 7 representative systems, including vanilla long-context LLMs, RAG, and memory-augmented agent frameworks. Across all systems, we observe consistently low performance (avg. 27.9% accuracy), especially on questions requiring aggregated reasoning over multiple pieces of evidence. Our analysis shows that performance is primarily limited by retrieval and memory construction. Furthermore, current memory systems struggle to recall and reason over earlier facts that are revised or interfered with by subsequent context, with accuracy degrading as the number of intervening updates increases.
>
---
#### [replaced 018] Quantifying the Climate Risk of Generative AI: Region-Aware Carbon Accounting with G-TRACE and the AI Sustainability Pyramid
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于气候风险评估任务，旨在量化生成式AI的碳排放。通过G-TRACE框架和AI可持续性金字塔模型，分析不同区域和模态的碳足迹，提出可持续部署的政策建议。**

- **链接: [https://arxiv.org/pdf/2511.04776](https://arxiv.org/pdf/2511.04776)**

> **作者:** Zahida Kausar; Seemab Latif; Raja Khurram Shahzad; Mehwish Fatima
>
> **备注:** 27 page, 4 figures
>
> **摘要:** Generative Artificial Intelligence (GenAI) represents a rapidly expanding digital infrastructure whose energy demand and associated CO2 emissions are emerging as a new category of climate risk. This study introduces G-TRACE (GenAI Transformative Carbon Estimator), a cross-modal, region-aware framework that quantifies training- and inference-related emissions across modalities and deployment geographies. Using real-world analytics and microscopic simulation, G-TRACE measures energy use and carbon intensity per output type (text, image, video) and reveals how decentralized inference amplifies small per-query energy costs into system-level impacts. Through the Ghibli-style image generation trend (2024-2025), we estimate 4,309 MWh of energy consumption and 2,068 tCO2 emissions, illustrating how viral participation inflates individual digital actions into tonne-scale consequences. Building on these findings, we propose the AI Sustainability Pyramid, a seven-level governance model linking carbon accounting metrics (L1-L7) with operational readiness, optimization, and stewardship. This framework translates quantitative emission metrics into actionable policy guidance for sustainable AI deployment. The study contributes to the quantitative assessment of emerging digital infrastructures as a novel category of climate risk, supporting adaptive governance for sustainable technology deployment. By situating GenAI within climate-risk frameworks, the work advances data-driven methods for aligning technological innovation with global decarbonization and resilience objectives.
>
---
#### [replaced 019] TSR: Trajectory-Search Rollouts for Multi-Turn RL of LLM Agents
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出TSR方法，用于改进多轮强化学习中轨迹生成的质量与稳定性，解决奖励稀疏和环境随机带来的挑战。**

- **链接: [https://arxiv.org/pdf/2602.11767](https://arxiv.org/pdf/2602.11767)**

> **作者:** Aladin Djuhera; Swanand Ravindra Kadhe; Farhan Ahmed; Syed Zawad; Heiko Ludwig; Holger Boche
>
> **摘要:** Advances in large language models (LLMs) are driving a shift toward using reinforcement learning (RL) to train agents from iterative, multi-turn interactions across tasks. However, multi-turn RL remains challenging as rewards are often sparse or delayed, and environments can be stochastic. In this regime, naive trajectory sampling can hinder exploitation and induce mode collapse. We propose TSR (Trajectory-Search Rollouts), a training-time approach that repurposes test-time scaling ideas for improved per-turn rollout generation. TSR performs lightweight tree-style search to construct high-quality trajectories by selecting high-scoring actions at each turn using state-based feedback. This improves rollout quality and stabilizes learning while remaining compatible with standard policy gradient optimizers, making TSR optimizer-agnostic. We instantiate TSR with best-of-N, beam, and shallow lookahead search, and pair it with PPO and GRPO, achieving up to 15% performance gains and more stable learning on Sokoban, FrozenLake, and WebShop tasks at a modest, one-time increase in training compute. By moving search from inference time to the rollout stage of training, TSR provides a modular and general mechanism for stronger multi-turn agent learning, complementary to existing frameworks and rejection-sampling-style selection methods.
>
---
#### [replaced 020] XNote: Benchmarking Automated Community Notes Generation for Image-based Contextual Deception
- **分类: cs.CL; cs.SI**

- **简介: 该论文研究图像情境欺骗的自动化社区注释生成任务，旨在解决人工生成注释效率低的问题。通过构建数据集XNote并测试多个模型，探索更有效的自动化方法。**

- **链接: [https://arxiv.org/pdf/2603.22453](https://arxiv.org/pdf/2603.22453)**

> **作者:** Jin Ma; Jingwen Yan; Mohammed Aldeen; Ethan Anderson; Taran Kavuru; Jinkyung Katie Park; Feng Luo; Long Cheng
>
> **摘要:** Community Notes have emerged as an effective crowd-sourced mechanism for combating online deception on social media platforms. However, its reliance on human contributors limits both the timeliness and scalability. In this work, we study the automated Community Notes generation task for image-based contextual deception, where an authentic image is paired with misleading context (e.g., time, entity, and event). Unlike prior work that primarily focuses on deception detection (i.e., judging whether a post is true or false in a binary manner), automated Community Notes generation requires producing concise and grounded notes that help users recover the missing or corrected context. This problem remains underexplored due to the scarcity of datasets that support this task. To address this gap, we curate a real-world dataset, XNote, comprising X posts with associated Community Notes and external contexts, along with annotations of topics and deceptive factors. We further benchmark a range of frontier large vision language models (LVLMs) on XNote, evaluating their performance on both deception detection and note generation tasks. We also compare against an end-to-end approach, SNIFFER, and a commercial tool, GPT-5. Our results highlight the challenges in automated Community Notes generation, underscoring the need for improved methods and metrics tailored for this task.
>
---
#### [replaced 021] Soohak: A Mathematician-Curated Benchmark for Evaluating Research-level Math Capabilities of LLMs
- **分类: cs.CL**

- **简介: 该论文提出Soohak基准，用于评估大语言模型的研究级数学能力。针对现有基准不足，作者邀请数学家创建439道题，包含挑战与拒绝子集，以测试模型推理与判断能力。**

- **链接: [https://arxiv.org/pdf/2605.09063](https://arxiv.org/pdf/2605.09063)**

> **作者:** Guijin Son; Seungone Kim; Catherine Arnett; Hyunwoo Ko; Hyein Lee; Hyeonah Kang; Jiang Longxi; Jin Yun; JungYup Lee; Kyungmin Lee; Sam Yoosuk Kim; Sang Park; Seunghyeok Hong; SeungJae Lee; Seungyeop Yi; Shinae Shin; SunHye Bok; Sunyoung Shin; Yonghoon Ji; Youngtaek Kim; Hanearl Jung; Akari Asai; Graham Neubig; Sean Welleck; Youngjae Yu; Akshelin R; Alexander B. Ivanov; Boboev Muhammadjon; Chae Young Han; Christian Stump; Cooper R. Anderson; Dmitrii Karp; Dohyun Kwon; Dongryung Yi; DoYong Kwon; Duk-Soon Oh; Eunho Choi; Giovanni Resta; Greta Panova; Huiyun Noh; Hyungryul Baik; Hyungsun Bae; Inomov Mashrafdzhon; Jeewon Kim; Jeong-Rae Kim; Ji Eun Lee; Jiaqi Liu; Jieui Kang; Jimin Kim; Jon-Lark Kim; Joonyeong Won; Junseo Yoon; Junwoo Jo; Kibeom Kim; Kiwoon Kwon; Mario Kummer; Max Mercer; Min Hoon Kim; Minjun Kim; Nahyun Lee; Ng Ze-An; Nicolas Libedinsky; Rafał Marcin Łochowski; Raphaël Lachièze-Rey; Robert Auffarth; Ruichen Zhang; Sejin Park; Seonguk Seo; Shin Jaehoon; Sunatullo; Taewoong Eom; Yeachan Park; Yongseok Jang; Youchan Oh; Zhaoyang Wang; Zoltán Kovács
>
> **备注:** Under review, For questions or model-evaluation requests, contact $this http URL@snu.this http URL$
>
> **摘要:** Following the recent achievement of gold-medal performance on the IMO by frontier LLMs, the community is searching for the next meaningful and challenging target for measuring LLM reasoning. Whereas olympiad-style problems measure step-by-step reasoning alone, research-level problems use such reasoning to advance the frontier of mathematical knowledge itself, emerging as a compelling alternative. Yet research-level math benchmarks remain scarce because such problems are difficult to source (e.g., Riemann Bench and FrontierMath-Tier 4 contain 25 and 50 problems, respectively). To support reliable evaluation of next-generation frontier models, we introduce Soohak, a 439-problem benchmark newly authored from scratch by 64 mathematicians. Soohak comprises two subsets. On the Challenge subset, frontier models including Gemini-3-Pro, GPT-5, and Claude-Opus-4.5 reach 30.4%, 26.4%, and 10.4% respectively, leaving substantial headroom, while leading open-weight models such as Qwen3-235B, GPT-OSS-120B, and Kimi-2.5 remain below 15%. Notably, beyond standard problem solving, Soohak introduces a refusal subset that probes a capability intrinsic to research mathematics: recognizing ill-posed problems and pausing rather than producing confident but unjustified answers. On this subset, no model exceeds 50%, identifying refusal as a new optimization target that current models do not directly address. To prevent contamination, the dataset will be publicly released in late 2026, with model evaluations available upon request in the interim.
>
---
#### [replaced 022] HALvest-Contrastive: Retrieval-Like Authorship Attribution with Patch-Level Late Interaction
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于作者归属任务，解决文本相似性受主题干扰的问题。通过构建HALvest-Contrastive数据集，并引入Patch-Level Late Interaction方法提升匹配效果。**

- **链接: [https://arxiv.org/pdf/2407.20595](https://arxiv.org/pdf/2407.20595)**

> **作者:** Francis Kulumba; Wissam Antoun; Guillaume Vimont; Laurent Romary; Florian Cafiero
>
> **备注:** 18 pages, 9 figures. Under review
>
> **摘要:** Deciding whether two pieces of text share an author is made difficult by topical confound: two writers covering the same topic often look more alike than one writer covering two topics. We tackle this with HALvest, a 17-billion-token multilingual corpus of open-access scholarly papers, and its English contrastive derivative HALvest-Contrastive, in which same-author passages are drawn from distinct papers within a field to minimize topical overlap. We also revisit how documents are compared. Authorship systems traditionally compress each document into a single vector, we keep a sequence of vectors and compare them with late interaction, then introduce Patch-Level Late Interaction (PLI), which compresses neighboring tokens into patches before matching. Matching at the sequence level greatly improves performance over the single-vector baseline, but the optimal interaction granularity is subtle.
>
---
#### [replaced 023] CoLD: Counterfactually-Guided Length Debiasing for Process Reward Models in Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，针对过程奖励模型中的长度偏差问题，提出CoLD框架以提升推理的准确性和简洁性。**

- **链接: [https://arxiv.org/pdf/2507.15698](https://arxiv.org/pdf/2507.15698)**

> **作者:** Congmin Zheng; Jiachen Zhu; Jianghao Lin; Xinyi Dai; Weiwen Liu; Haoxuan Li; Yong Yu; Weinan Zhang; Mengyue Yang
>
> **摘要:** Process Reward Models (PRMs) play a central role in evaluating and guiding multi-step reasoning in large language models (LLMs), especially for mathematical problem solving. However, we identify a pervasive length bias in existing PRMs: they tend to assign higher scores to longer reasoning steps, even when the semantic content and logical validity are unchanged. This bias undermines the reliability of reward predictions and leads to overly verbose outputs during inference. To address this issue, we propose CoLD(Counterfactually-Guided Length Debiasing), a unified framework that mitigates length bias through three components: an explicit length-penalty adjustment, a learned bias estimator trained to capture spurious length-related signals, and a joint training strategy that enforces length-invariance in reward predictions. Our approach is grounded in counterfactual reasoning and informed by causal graph analysis. Extensive experiments on MATH500 and GSM-Plus show that CoLD improves accuracy in step selection, and encourages more concise, logically valid reasoning. Furthermore, it consistently improves downstream RL performance and generalizes across domains by mitigating length bias, demonstrating CoLD's strong generalization capability.
>
---
#### [replaced 024] EnsemHalDet: Robust VLM Hallucination Detection via Ensemble of Internal State Detectors
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态 hallucination 检测任务，旨在解决VLMs生成不准确或无依据内容的问题。通过集成多个内部表示，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.02784](https://arxiv.org/pdf/2604.02784)**

> **作者:** Ryuhei Miyazato; Shunsuke Kitada; Kei Harada
>
> **摘要:** Vision-Language Models (VLMs) excel at multimodal tasks, but they remain vulnerable to hallucinations that are factually incorrect or ungrounded in the input image. Recent work suggests that hallucination detection using internal representations is more efficient and accurate than approaches that rely solely on model outputs. However, existing internal-representation-based methods typically rely on a single representation or detector, limiting their ability to capture diverse hallucination signals. In this paper, we propose EnsemHalDet, an ensemble-based hallucination detection framework that leverages multiple internal representations of VLMs, including attention outputs and hidden states. EnsemHalDet trains independent detectors for each representation and combines them through ensemble learning. Experimental results across multiple VQA datasets and VLMs show that EnsemHalDet consistently outperforms prior methods and single-detector models in terms of AUC. These results demonstrate that ensembling diverse internal signals significantly improves robustness in multimodal hallucination detection.
>
---
#### [replaced 025] Dr.LLM: Dynamic Layer Routing in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Dr. LLM，解决LLM计算效率与准确性平衡问题。通过动态路由机制，按需选择执行层，提升效率并保持精度。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2510.12773](https://arxiv.org/pdf/2510.12773)**

> **作者:** Ahmed Heakl; Martin Gubri; Salman Khan; Sangdoo Yun; Seong Joon Oh
>
> **备注:** Published at ICLR 2026
>
> **摘要:** Large Language Models (LLMs) process every token through all layers of a transformer stack, causing wasted computation on simple queries and insufficient flexibility for harder ones that need deeper reasoning. Adaptive-depth methods can improve efficiency, but prior approaches rely on costly inference-time search, architectural changes, or large-scale retraining, and in practice often degrade accuracy despite efficiency gains. We introduce Dr. LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that equips pretrained models with lightweight per-layer routers deciding to skip, execute, or repeat a block. Routers are trained with explicit supervision: using Monte Carlo Tree Search (MCTS), we derive high-quality layer configurations that preserve or improve accuracy under a compute budget. Our design, windowed pooling for stable routing, focal loss with class balancing, and bottleneck MLP routers, ensures robustness under class imbalance and long sequences. On ARC (logic) and DART (math), Dr. LLM improves accuracy by up to +3.4%p while saving 5 layers per example on average. Routers generalize to out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA, AGIEval) with only 0.85% accuracy drop while retaining efficiency, and outperform prior routing methods by up to +7.7%p. Overall, Dr. LLM shows that explicitly supervised routers retrofit frozen LLMs for budget-aware, accuracy-driven inference without altering base weights. Code is available at this https URL.
>
---
#### [replaced 026] Mechanistic Interpretability Needs Philosophy
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨机制可解释性（MI）与哲学的关系，指出MI需哲学支持以澄清概念、优化方法。任务是促进跨学科合作，解决MI中的认知与伦理问题。**

- **链接: [https://arxiv.org/pdf/2506.18852](https://arxiv.org/pdf/2506.18852)**

> **作者:** Iwan Williams; Ninell Oldenburg; Ruchira Dhar; Joshua Hatherley; Constanza Fierro; Nina Rajcic; Sandrine R. Schiller; Filippos Stamatiou; Anders Søgaard
>
> **摘要:** Mechanistic interpretability (MI) aims to explain how neural networks work by uncovering their underlying mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy as an ongoing partner in clarifying its concepts, refining its methods, and navigating the epistemic and ethical complexities of interpreting AI systems. There is significant unrealised potential for progress in MI to be gained through deeper engagement with philosophers and philosophical frameworks. Taking three open problems from the MI literature as examples, this paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue.
>
---
#### [replaced 027] Efficient Pre-Training with Token Superposition
- **分类: cs.CL**

- **简介: 该论文提出Token-Superposition Training（TST），用于提升大语言模型预训练效率，解决高成本和低吞吐问题。通过两阶段训练方法，在不改变架构的前提下提高数据处理速度。**

- **链接: [https://arxiv.org/pdf/2605.06546](https://arxiv.org/pdf/2605.06546)**

> **作者:** Bowen Peng; Théo Gigant; Jeffrey Quesnelle
>
> **备注:** 25 pages, 11 figures, 28 tables
>
> **摘要:** Pre-training of Large Language Models is often prohibitively expensive and inefficient at scale, requiring complex and invasive modifications in order to achieve high data throughput. In this work, we present Token-Superposition Training (TST), a simple drop-in method that significantly improves the data throughput per FLOPs during pre-training without modifying the parallelism, optimizer, tokenizer, data, or model architecture. TST is done in two phases: (i) A highly efficient superposition phase where we combine many contiguous tokens into one bag and train using a multi-hot cross-entropy (MCE) objective, and (ii) a recovery phase where we revert back to standard training. We extensively evaluate TST on the scale of 270M and 600M parameters and validate on 3B and a 10B A1B mixture of experts model, demonstrating that it is highly robust in different settings. Ultimately, TST consistently outperforms baseline loss and downstream evaluations, and under equal-loss settings, TST yields up to a 2.5x reduction in total pre-training time at the 10B A1B scale.
>
---
#### [replaced 028] An LLM-Based System for Argument Mining
- **分类: cs.CL**

- **简介: 该论文属于论点挖掘任务，旨在从文本中重构论点结构。提出基于大语言模型的系统，通过多阶段流程识别论点成分及其逻辑关系，构建抽象论点图。**

- **链接: [https://arxiv.org/pdf/2605.13793](https://arxiv.org/pdf/2605.13793)**

> **作者:** Paulo Pirozelli; Victor Hugo Nascimento Rocha; Fabio G. Cozman; Douglas Aldred
>
> **摘要:** Arguments are a fundamental aspect of human reasoning, in which claims are supported, challenged, and weighed against one another. We present an end-to-end large language model (LLM)-based system for reconstructing arguments from natural language text into abstract argument graphs. The system follows a multi-stage pipeline that progressively identifies argumentative components, selects relevant elements, and uncovers their logical relations. These elements are represented as directed acyclic graphs consisting of two component types (premises and conclusions) and three relation types (support, attack, and undercut). We conduct two complementary experiments to evaluate the system. First, we perform a manual evaluation on arguments drawn from an argumentation theory textbook to assess the system's ability to recover argumentative structure. Second, we conduct a quantitative evaluation on benchmark datasets, allowing comparison with prior work by mapping our outputs to established annotation schemes. Results show that the system can adequately recover argumentative structures and, when adapted to different annotation schemes, achieve reasonable performance across benchmark datasets. These findings highlight the potential of LLM-based pipelines for scalable argument mining.
>
---
#### [replaced 029] Disentangling generalization and memorization in large language models using chess
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在棋类任务中的泛化与记忆能力，旨在区分模型是依赖记忆还是真正推理。通过构建不同先验密度的棋局分类法，分析模型表现，揭示其在缺乏先验知识时的局限性。**

- **链接: [https://arxiv.org/pdf/2601.16823](https://arxiv.org/pdf/2601.16823)**

> **作者:** Leonard S. Pleiss; Maximilian Schiffer; Robert K. von Weizsaecker
>
> **摘要:** Large Language Models (LLMs) exhibit remarkable capabilities, yet it remains unclear to what extent these reflect sophisticated recall or genuine reasoning ability. We introduce chess as a controlled testbed aimed at disentangling these faculties. Leveraging the game's structure and scalable engine evaluations, we construct a taxonomy of positions varying in density of relevant priors - ranging from common states solvable by memorization to completely novel ones requiring generalization. Crucially, our approach achieves this distinction without requiring explicit knowledge of the models' training data. Applying this taxonomy, we combine a longitudinal analysis of the GPT lineage with a rigorous evaluation of contemporary models, including Claude Opus and Gemini. Our analysis reveals a steep gradient: performance consistently degrades as the density of relevant priors decreases. Notably, for tasks with few relevant priors, base model performance regresses to the random-play baseline. While newer models improve, progress slows significantly for tasks with sparse priors. Furthermore, while reasoning-augmented inference improves performance, its relative marginal benefit per token decreases in the absence of relevant priors. These results suggest limitations in systematic generalization, highlighting the need for mechanisms beyond scale to achieve robust performance when deprived of relevant priors.
>
---
#### [replaced 030] Borrowed Geometry: Cross-Distribution Head-Importance Fingerprints of Frozen Pretrained Gemma 4 31B
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究冻结模型在跨模态任务中的重要性头指纹，通过分析注意力头对非语言任务的影响，识别关键头部并验证其因果作用。**

- **链接: [https://arxiv.org/pdf/2605.00333](https://arxiv.org/pdf/2605.00333)**

> **作者:** Abay Bektursun
>
> **备注:** v2: Added head-level causal ablation on OGBench cube-task1 (n=30, 3.2x specificity; n=5 paired-t p=0.039) and full L26 sweep. New sections on honest negatives (activation patching null, sufficiency null, within-layer Spearman wrong-direction). Multiplicity-aware permutation null V4 P=0.013. Title and framing updated. 25 pages (13 main), 10 figures
>
> **摘要:** Frozen Gemma 4 31B weights pretrained exclusively on text, unmodified, transfer through a thin trainable interface to non-text modalities the substrate has never processed. On the L24--L29 slice (192 attention heads), an English-text TxtCopy attention probe (95 sentences) and per-head ablation impact on four non-language token-pattern tasks (binary copy, associative recall, 1D cellular automaton Rule 90, binary addition) jointly classify four heads -- L26.28, L27.28, L27.2, L27.3 -- as top-tier on both signals. The slice-level joint coincidence is significant under hypergeometric null ($P = 0.0013$, $N=192$, $K=38$, $n=4$) and survives multiplicity-aware permutation tests ($P_{V4} = 0.013$). Pretrained Gemma L26 reaches 60.22% on OGBench cube-double-play-task1 vs ~1% for random-init Gemma ($+59$pt at $n=3$); a FrozenRandom-GPT2 control with correct $1/\sqrt{d_k}$ scaling also fails. Head-level causal validation: zeroing L26.28 in the trained cube-task1 IQL agent drops success $63.3\% \to 10.0\%$ vs $46.7\%$ for a layer-matched low-TxtCopy negative control ($3.2\times$ specificity at $n=30$; $n=5$ paired-$t$ $p=0.039$). A full L26 sweep places L26.28 at rank 4 of 32. Honest negatives: within-L26 Spearman $\rho(\text{TxtCopy, drop}) = +0.37$ (opposite of within-layer causal reading); single-head activation patching does not transfer the matching variable; the 4 named heads alone do not suffice on any task; Walker2d-DT and scene-task1 recruit L24 outside the named slice and show null head-ablation specificity. We frame the contribution as a cross-distribution importance fingerprint at the slice level plus head-level causal evidence on one cross-modality target.
>
---
#### [replaced 031] Beyond Perplexity: A Geometric and Spectral Study of Low-Rank Pre-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究低秩预训练方法与全秩训练的差异。通过多维度分析，揭示低秩方法在解空间、谱结构和激活特征上的不同，指出其泛化能力受限。**

- **链接: [https://arxiv.org/pdf/2605.13652](https://arxiv.org/pdf/2605.13652)**

> **作者:** Namrata Shivagunde; Vijeta Deshpande; Sherin Muckatira; Anna Rumshisky
>
> **备注:** 9 pages, 5 figures, 2 tables
>
> **摘要:** Pre-training large language models is dominated by the memory cost of storing full-rank weights, gradients, and optimizer states. Low-rank pre-training has emerged to address this, and the space of methods has grown rapidly. A central question remains open: do low-rank methods produce models that generalize comparably to full-rank training, or does the rank constraint fundamentally alter the solutions reached? Existing comparisons rely almost entirely on validation perplexity from single-seed runs, often carried forward from prior literature. Yet perplexity is a poor proxy for solution quality; two methods can match on perplexity while converging to different loss landscape regions and internal representations. We close this gap by characterizing the solutions found by five low-rank pre-training methods, GaLore and Fira (memory-efficient optimizers), CoLA and SLTrain (architecture reparameterizations), and ReLoRA (adapter-style updates with periodic resets), against full-rank training at three model scales (60M, 130M, 350M). We evaluate each along 16 metrics across four dimensions: 1-D loss landscape along random/top-K PCA directions, 1-D interpolation between checkpoints, spectral structure of the weights and learned updates, and activation similarity to full-rank training. We show that low-rank methods are not equivalent to full-rank training, nor to one another, even when validation perplexity is close. Full-rank training settles into a sharper basin than low-rank methods along random directions, while the reverse holds for the top-1 PCA direction. Each method converges to a geometrically distinct basin. Low-rank activations diverge from full-rank in later layers as training progresses, with GaLore tracking full-rank most closely. Further, validation perplexity does not translate to downstream performance at every scale. Adding geometric and spectral metrics improves the prediction.
>
---
#### [replaced 032] The Wikidata Query Logs Dataset
- **分类: cs.CL**

- **简介: 该论文提出Wikidata查询日志数据集，用于问答任务。解决真实SPARQL查询难以直接使用的问题，通过去匿名化和生成自然语言问题，构建335k问答对。**

- **链接: [https://arxiv.org/pdf/2602.14594](https://arxiv.org/pdf/2602.14594)**

> **作者:** Sebastian Walter; Hannah Bast
>
> **备注:** Accepted for publication at SIGIR 2026
>
> **摘要:** We present the Wikidata Query Logs (WDQL) dataset, a dataset consisting of 335k question-query pairs over the Wikidata knowledge graph. It is over 11x larger than the largest existing Wikidata datasets of similar format without relying on template-generated queries. Instead, we construct it using real-world SPARQL queries sent to the Wikidata Query Service and generate questions for them. Since these log-based queries are anonymized, and therefore often do not produce results, a significant amount of effort is needed to convert them back into meaningful SPARQL queries. To achieve this, we present an agent-based method that iteratively de-anonymizes, cleans, and verifies queries against Wikidata while also generating corresponding natural-language questions. We demonstrate the benefit of this dataset for training question-answering methods. All WDQL assets, as well as the agent code, are publicly available via this https URL under a permissive license.
>
---
#### [replaced 033] LLM-MC-Affect: LLM-Based Monte Carlo Modeling of Affective Trajectories and Latent Ambiguity for Interpersonal Dynamic Insight
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出LLM-MC-Affect框架，用于建模情感轨迹和潜在模糊性，解决人际互动中的情感动态分析问题。**

- **链接: [https://arxiv.org/pdf/2601.03645](https://arxiv.org/pdf/2601.03645)**

> **作者:** Yu-Zheng Lin; Bono Po-Jen Shih; John Paul Martin Encinas; Elizabeth Victoria Abraham Achom; Karan Himanshu Patel; Jesus Horacio Pacheco; Sicong Shao; Jyotikrishna Dass; Soheil Salehi; Pratik Satam
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Emotional coordination is a core property of human interaction that shapes how relational meaning is constructed in real time. While text-based affect inference has become increasingly feasible, prior approaches often treat sentiment as a deterministic point estimate for individual speakers, failing to capture the inherent subjectivity, latent ambiguity, and sequential coupling found in mutual exchanges. We introduce LLM-MC-Affect, a probabilistic framework that characterizes emotion not as a static label, but as a continuous latent probability distribution defined over an affective space. By leveraging stochastic LLM decoding and Monte Carlo estimation, the methodology approximates these distributions to derive high-fidelity sentiment trajectories that explicitly quantify both central affective tendencies and perceptual ambiguity. These trajectories enable a structured analysis of interpersonal coupling through sequential cross-correlation and slope-based indicators, identifying leading or lagging influences between interlocutors. To validate the interpretive capacity of this approach, we utilize teacher-student instructional dialogues as a representative case study, where our quantitative indicators successfully distill high-level interaction insights such as effective scaffolding. This work establishes a scalable and deployable pathway for understanding interpersonal dynamics, offering a generalizable solution that extends beyond education to broader social and behavioral research.
>
---
#### [replaced 034] Cross-modal Consistency Guidance for Robust Emotion Control in Auto-Regressive TTS Models
- **分类: cs.CL**

- **简介: 该论文属于情感语音合成任务，解决情绪与文本语义冲突导致的表达质量下降问题，提出CCG-CFG方法提升情感一致性与自然度。**

- **链接: [https://arxiv.org/pdf/2510.13293](https://arxiv.org/pdf/2510.13293)**

> **作者:** Yizhou Peng; Yukun Ma; Chong Zhang; Yi-Wen Chao; Chongjia Ni; Bin Ma; Eng Siong Chng
>
> **备注:** Updated and resubmitted to Interspeech 2026
>
> **摘要:** While Text-to-Speech (TTS) systems enable emotional control via natural-language instructions, expressiveness, naturalness, and speech quality degrade when the target emotion conflicts with the textual semantics. We propose a Cross-modal Consistency Guided Classifier-Free Guidance (CCG-CFG) method with dynamic scales based on the degree of inconsistency between the text emotion and the explicit speech emotion, replacing the dropout condition with the text emotion. We also distill the CCG-CFG guidance signal using a hard-sample mining strategy, improving the TTS model's emotional alignment capability. Evaluations on five emotional corpora and two TTS benchmarks show that our approaches applied to CosyVoice2 achieve up to a 12% absolute improvement in emotion-recognition accuracy and a 10% relative improvement in subjective scores, outperforming baselines including HierSpeech++, Qwen3-TTS, and original CosyVoice2, while preserving intelligibility, naturalness, and high speech quality.
>
---
#### [replaced 035] MTraining: Distributed Dynamic Sparse Attention for Efficient Ultra-Long Context Training
- **分类: cs.CL; cs.DC; cs.LG**

- **简介: 该论文属于大语言模型训练任务，旨在解决超长上下文训练中的计算不平衡和通信开销问题。提出MTraining方法，通过动态稀疏注意力提升训练效率。**

- **链接: [https://arxiv.org/pdf/2510.18830](https://arxiv.org/pdf/2510.18830)**

> **作者:** Wenxuan Li; Chengruidong Zhang; Huiqiang Jiang; Yucheng Li; Yuqing Yang; Lili Qiu
>
> **摘要:** The adoption of long context windows has become a standard feature in Large Language Models (LLMs), as extended contexts significantly enhance their capacity for complex reasoning and broaden their applicability across diverse scenarios. Dynamic sparse attention is a promising approach for reducing the computational cost of long-context. However, efficiently training LLMs with dynamic sparse attention on ultra-long contexts-especially in distributed settings-remains a significant challenge, due in large part to worker- and step-level imbalance. This paper introduces MTraining, a novel distributed methodology leveraging dynamic sparse attention to enable efficient training for LLMs with ultra-long contexts. Specifically, MTraining integrates three key components: a dynamic sparse training pattern, balanced sparse ring attention, and hierarchical sparse ring attention. These components are designed to synergistically address the computational imbalance and communication overheads inherent in dynamic sparse attention mechanisms during the training of models with extensive context lengths. We demonstrate the efficacy of MTraining by training Qwen2.5-3B, successfully expanding its context window from 32K to 512K tokens on a cluster of 32 A100 GPUs. Our evaluations on a comprehensive suite of downstream tasks, including RULER, PG-19, InfiniteBench, and Needle In A Haystack, reveal that MTraining achieves up to a 6x higher training throughput while preserving model accuracy. Our code is available at this https URL.
>
---
#### [replaced 036] Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification
- **分类: cs.CL**

- **简介: 该论文属于专利分类任务，解决LLM生成理由中的逻辑错误和标签不匹配问题。通过引入信任指标进行自过滤蒸馏，提升分类可靠性。**

- **链接: [https://arxiv.org/pdf/2510.05431](https://arxiv.org/pdf/2510.05431)**

> **作者:** Yongmin Yoo; Xu Zhang; Longbing Cao
>
> **摘要:** Organizing large-scale patent corpora according to classification schemes is a core information management task that determines the accuracy and efficiency of prior art retrieval, technology knowledge discovery, and intellectual property decision-making. Recent approaches distill natural language rationales generated by large language models (LLMs) into compact student models, yet logical errors, label mismatches, and taxonomy misalignments inherent in these rationales are indiscriminately absorbed during training, undermining classification reliability and propagating errors throughout downstream information processes. Rather than correcting such errors post-hoc, we propose Self-Filtered Distillation (SFD), which embeds quality assurance directly into the learning process by reinterpreting LLM-generated rationales as trust indicators rather than ground-truth supervision. SFD integrates three unsupervised signals into a unified trust score that dynamically modulates each training instance's contribution: Self-Consistency, which quantifies agreement among independently generated rationales; Class Entailment Alignment, which evaluates semantic coherence between a rationale and its assigned CPC class definition; and LLM Agreement Scoring, which assesses external plausibility through an independent verifier. On the USPTO-2M benchmark comprising over two million patents, SFD achieves up to 38.7\% relative improvement in Macro-F1 across four student architectures, and the strong correlation between trust scores and expert judgments ($r = 0.685$) confirms that the framework provides not only accurate predictions but also decomposable confidence semantics that enable auditable and self-documenting classification outcomes for large-scale patent knowledge organization.
>
---
#### [replaced 037] PICon: A Multi-Turn Interrogation Framework for Evaluating Persona Agent Consistency
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决 persona agent 一致性评估问题。提出 PICon 框架，通过多轮提问检测其内部、外部和重测一致性，揭示系统缺陷。**

- **链接: [https://arxiv.org/pdf/2603.25620](https://arxiv.org/pdf/2603.25620)**

> **作者:** Minseo Kim; Sujeong Im; Junseong Choi; Junhee Lee; Chaeeun Shim; Hwajung Hong; Edward Choi
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Large language model (LLM)-based persona agents are rapidly being adopted as scalable proxies for human participants across diverse domains. Yet there is no systematic method for verifying whether a persona agent's responses remain free of contradictions and factual inaccuracies throughout an interaction. A principle from interrogation methodology offers a lens: no matter how elaborate a fabricated identity, systematic interrogation will expose its contradictions. We apply this principle to propose PICon, an evaluation framework that probes persona agents through logically chained multi-turn questioning. PICon evaluates consistency along three core dimensions: internal consistency (freedom from self-contradiction), external consistency (alignment with real-world facts), and retest consistency (stability under repetition). Evaluating seven groups of persona agents alongside 63 real human participants, we find that even systems previously reported as highly consistent fail to meet the human baseline across all three dimensions, revealing contradictions and evasive responses under chained questioning. This work provides both a conceptual foundation and a practical methodology for evaluating persona agents before trusting them as substitutes for human participants. We provide the source code and an interactive demo at: this https URL
>
---
#### [replaced 038] Fingerprinting LLMs via Prompt Injection
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于模型溯源任务，旨在解决后处理模型难以识别来源的问题。提出LLMPrint方法，通过提示注入构建鲁棒指纹，实现高效验证。**

- **链接: [https://arxiv.org/pdf/2509.25448](https://arxiv.org/pdf/2509.25448)**

> **作者:** Yuepeng Hu; Zhengyuan Jiang; Mengyuan Li; Osama Ahmed; Zhicong Huang; Cheng Hong; Neil Gong
>
> **摘要:** Large language models (LLMs) are often modified after release through post-processing such as post-training or quantization, which makes it challenging to determine whether one model is derived from another. Existing provenance detection methods have two main limitations: (1) they embed signals into the base model before release, which is infeasible for already published models, or (2) they compare outputs across models using hand-crafted or random prompts, which are not robust to post-processing. In this work, we propose LLMPrint, a novel detection framework that constructs fingerprints by exploiting LLMs' inherent vulnerability to prompt injection. Our key insight is that by optimizing fingerprint prompts to enforce consistent token preferences, we can obtain fingerprints that are both unique to the base model and robust to post-processing. We further develop a unified verification procedure that applies to both gray-box and black-box settings, with statistical guarantees. We evaluate LLMPrint on five base models and around 700 post-trained or quantized variants. Our results show that LLMPrint achieves high true positive rates while keeping false positive rates near zero. The code is publicly available at this https URL.
>
---
#### [replaced 039] What's Holding Back Latent Visual Reasoning?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉推理任务，探讨为何模型无法有效利用潜在视觉中间步骤。研究发现，现有数据集中的潜在标记信息不足，且生成的标记与真实标记偏差大，导致模型无法依赖它们。**

- **链接: [https://arxiv.org/pdf/2605.18445](https://arxiv.org/pdf/2605.18445)**

> **作者:** André G. Viveiros; Nuno Gonçalves; André F. T. Martins; Matthias Lindemann
>
> **摘要:** Humans can approach complex visual problems by mentally simulating intermediate visual steps, rather than reasoning through language alone. Inspired by this, several works on Vision-Language Models have recently explored chain-of-thought reasoning with continuous latent tokens as intermediate visual imagination steps. In this work, we investigate how recent models leverage such latent tokens. Surprisingly, we find that model accuracy is unaffected when latent tokens are replaced by uninformative dummy tokens. This indicates that latent tokens play a minimal causal role in the model's final prediction. To better understand this phenomenon, we analyze both the training signal provided by oracle latent representations and the quality of the latent tokens generated at inference time. Our experiments reveal two crucial issues holding back latent visual reasoning: First, in most existing datasets, oracle latent tokens provide limited additional information beyond the original image and do not substantially simplify the task, leading models to ignore them during training and effectively bypassing them at inference time. When fine-tuned on a diagnostic dataset, in which latent tokens provide sufficient support for the final prediction, we show that models can causally rely on them. Second, the latent tokens produced at inference time deviate from their corresponding oracle representations, collapsing to a narrow region and preventing benefits even when the model relies on them. Overall, our findings suggest that future progress in latent visual reasoning depends on two key pillars: high-quality datasets with informative intermediate steps and more precise latent token prediction.
>
---
#### [replaced 040] Rewriting History: A Recipe for Interventional Analyses to Study Data Effects on Model Behavior
- **分类: cs.CL**

- **简介: 该论文属于语言模型研究任务，旨在探讨训练数据对模型行为的影响。通过干预数据并重新训练模型，验证数据与行为的关系。**

- **链接: [https://arxiv.org/pdf/2510.14261](https://arxiv.org/pdf/2510.14261)**

> **作者:** Rahul Nadkarni; Yanai Elazar; Hila Gonen; Noah A. Smith
>
> **备注:** Accepted to TACL, pre-MIT Press publication version
>
> **摘要:** We present an experimental recipe for studying the relationship between training data and language model (LM) behavior. We outline steps for intervening on data batches -- i.e., ``rewriting history'' -- and then retraining model checkpoints over that data to test hypotheses relating data to behavior. Our recipe breaks down such an intervention into stages that include selecting evaluation items from a benchmark that measures model behavior, matching relevant documents to those items, and modifying those documents before retraining and measuring the effects. We demonstrate the utility of our recipe through case studies on factual knowledge acquisition in LMs, using both cooccurrence statistics and information retrieval methods to identify documents that might contribute to knowledge learning. Our results supplement past observational analyses that link cooccurrence to model behavior, while demonstrating that extant methods for identifying relevant training documents do not fully explain an LM's ability to correctly answer knowledge questions. Overall, we outline a recipe that researchers can follow to test further hypotheses about how training data affects model behavior. Our code is made publicly available to promote future work.
>
---
#### [replaced 041] SLoW: Select Low-frequency Words! Automatic Dictionary Selection for Translation on Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ADS任务，解决多语言翻译中字典选择效率问题。通过SLoW方法，自动选择低频词字典，提升翻译效果并减少token消耗。**

- **链接: [https://arxiv.org/pdf/2507.18902](https://arxiv.org/pdf/2507.18902)**

> **作者:** Hongyuan Lu; Zixuan Li; Zefan Zhang; Wai Lam
>
> **备注:** EMNLP 2025 Main
>
> **摘要:** There are more than 7,000 languages around the world, and current Large Language Models (LLMs) only support hundreds of languages. Dictionary-based prompting methods can enhance translation on them, but most methods use all the available dictionaries, which could be expensive. Instead, it will be flexible to have a trade-off between token consumption and translation performance. This paper proposes a novel task called \textbf{A}utomatic \textbf{D}ictionary \textbf{S}election (\textbf{ADS}). The goal of the task is to automatically select which dictionary to use to enhance translation. We propose a novel and effective method which we call \textbf{S}elect \textbf{Lo}w-frequency \textbf{W}ords! (\textbf{SLoW}) which selects those dictionaries that have a lower frequency. Our methods have unique advantages. First, there is no need for access to the training data for frequency estimation (which is usually unavailable). Second, it inherits the advantage of dictionary-based methods, where no additional tuning is required on LLMs. Experimental results on 100 languages from FLORES indicate that SLoW surpasses strong baselines, and it can obviously save token usage, with many languages even surpassing the translation performance of the full dictionary baseline.\footnote{A shocking fact is that there is no need to use the actual training data (often unobtainable) for frequency estimation, and an estimation frequency obtained using public resources is still apparently effective in improving translation with ChatGPT and Llama, and DeepSeek.}\footnote{Code and data available upon publication.}
>
---
#### [replaced 042] Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的认知扭曲检测任务，旨在解决标注不一致问题。通过使用大语言模型生成稳定标注，并提出跨数据集评估方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.01482](https://arxiv.org/pdf/2511.01482)**

> **作者:** Neha Sharma; Navneet Agarwal; Kairit Sirts
>
> **摘要:** Text-based automated Cognitive Distortion detection is a challenging task due to its subjective nature, with low agreement scores observed even among expert human annotators, leading to unreliable annotations. We explore the use of Large Language Models (LLMs) as consistent and reliable annotators, and propose that multiple independent LLM runs can reveal stable labeling patterns despite the inherent subjectivity of the task. Furthermore, to fairly compare models trained on datasets with different characteristics, we introduce a dataset-agnostic evaluation framework using Cohen's kappa as an effect size measure. This methodology allows for fair cross-dataset and cross-study comparisons where traditional metrics like F1 score fall short. Our results show that GPT-4 can produce consistent annotations (Fleiss's Kappa = 0.78), resulting in improved test set performance for models trained on these annotations compared to those trained on human-labeled data. Our findings suggest that LLMs can offer a scalable and internally consistent alternative for generating training data that supports strong downstream performance in subjective NLP tasks.
>
---
#### [replaced 043] DetectRL-X: Towards Reliable Multilingual and Real-World LLM-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于LLM生成文本检测任务，旨在提升多语言和真实场景下的检测可靠性。工作包括构建多语言基准DetectRL-X，涵盖8种语言和多种生成与修改方式，评估现有检测器性能。**

- **链接: [https://arxiv.org/pdf/2605.15518](https://arxiv.org/pdf/2605.15518)**

> **作者:** Junchao Wu; Yefeng Liu; Chenyu Zhu; Hao Zhang; Zeyu Wu; Tianqi Shi; Yichao Du; Longyue Wang; Weihua Luo; Jinsong Su; Derek F. Wong
>
> **备注:** ACL 2026 Main. Code and data are available at this https URL
>
> **摘要:** The effective detection and governance of Large Language Model (LLM) generated content has become increasingly critical due to the growing risk of misuse. Despite the impressive performance of existing detectors, their reliability and potential in multilingual, real-world scenarios remain largely underexplored. In this study, we introduce DetectRL-X, a comprehensive multilingual benchmark designed to evaluate advanced detectors across 8 dimensions. The benchmark encompasses 8 languages commonly used in commercial contexts and collects human-written texts from 6 domains highly susceptible to LLM misuse. To better aligned with real-world applications, We create LLM-generated texts using 4 popular commercial LLMs, and include typical AI-assisted writing operations such as polishing, expanding, and condensing to capture authentic usage patterns. Furthermore, we develop a multilingual framework for paraphrasing and perturbation attacks to simulate diverse human modifications and writing noise, enabling stress testing of detectors across languages. Experimental results on DetectRL-X reveal the strengths and limitations of current state-of-the-art detectors when applied to diverse linguistic resources. We further analyze how domains, generators, attack strategies, text length, and refinement operations influence performance in different languages, underscoring DetectRL-X as an effective benchmark for strengthening multilingual and language-specific detectors.
>
---
#### [replaced 044] Language Model Memory and Memory Models for Language
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文探讨语言模型的记忆能力，对比了语言模型与自编码器在信息存储上的差异，提出一种高效的记忆模型架构，解决信息存储与访问效率问题。**

- **链接: [https://arxiv.org/pdf/2602.13466](https://arxiv.org/pdf/2602.13466)**

> **作者:** Benjamin L. Badger
>
> **摘要:** The ability of machine learning models to store input information in hidden layer vector embeddings, analogous to the concept of `memory', is widely employed but not well characterized. We find that language model embeddings typically contain relatively little input information regardless of data and compute scale during training. In contrast, embeddings from autoencoders trained for input regeneration are capable of nearly perfect memory formation. The substitution of memory embeddings for token sequences leads to substantial computational efficiencies, motivating the introduction of a parallelizable encoder-decoder memory model architecture. Upon causal training these models contain information-poor embeddings incapable of arbitrary information access, but by combining causal and information retention objective functions they learn to form and decode information-rich memories. Training can be further streamlined by freezing a high fidelity encoder followed by a curriculum training approach where decoders first learn to process memories and then learn to additionally predict next tokens. We introduce the perspective that next token prediction training alone is poorly suited for accurate memory formation as the objective itself is non-invertible, motivating the use of combined objective functions for models where the entire input is not exposed.
>
---
#### [replaced 045] Learning Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型微调任务，探讨LoRA方法的性能差异。研究发现，调整学习率后，不同LoRA方法表现相近，表明 vanilla LoRA 仍具竞争力。**

- **链接: [https://arxiv.org/pdf/2602.04998](https://arxiv.org/pdf/2602.04998)**

> **作者:** Yu-Ang Lee; Ching-Yun Ko; Pin-Yu Chen; Mi-Yen Yeh
>
> **备注:** Project page: this https URL
>
> **摘要:** Low-Rank Adaptation (LoRA) is the prevailing approach for efficient large language model (LLM) fine-tuning. Building on this paradigm, recent studies have proposed alternative initialization strategies, architectural modifications, and optimization adjustments, reporting substantial improvements over vanilla LoRA. However, these gains are often demonstrated under fixed or narrowly tuned hyperparameter settings, despite the known sensitivity of neural networks to training configurations. In this work, we systematically re-evaluate nine representative LoRA variants alongside vanilla LoRA through extensive hyperparameter searches over learning rate, batch size, rank, and training duration. Across tasks spanning mathematical reasoning, commonsense reasoning, code generation, and instruction following at diverse model scales, we find that different LoRA methods favor distinct learning rate ranges. Crucially, once learning rates are properly tuned, all methods achieve similar peak performance (within 1-2%), with only subtle rank-dependent behaviors. These results suggest that vanilla LoRA remains a competitive baseline and that improvements reported under a single training configuration may not reflect consistent methodological advantages. Finally, a second-order analysis attributes the differing optimal learning rate ranges to variations in the largest Hessian eigenvalue, aligning with classical learning theories.
>
---
#### [replaced 046] Test-Time Speculation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Test-Time Speculation（TTS），解决长文本生成中推测解码效率下降的问题。通过在线微调推测模型，提升接受长度，增强推理速度。**

- **链接: [https://arxiv.org/pdf/2605.09329](https://arxiv.org/pdf/2605.09329)**

> **作者:** Avinash Kumar; Sujay Sanghavi; Poulami Das
>
> **摘要:** Speculative decoding accelerates LLM inference by using a fast draft model to generate tokens and a more accurate target model to verify them. Its performance depends on the $\textit{acceptance length}$, or number of draft tokens accepted by the target. Our studies show that the acceptance length of even state-of-the-art speculators, like DFlash, EAGLE-3 and PARD degrade with generation length, reaching values close to 1 (i.e. no speedup) within just a few thousand output tokens, making speculators ineffective for long-response tasks. Acceptance lengths decline because most speculators are trained offline on short sequences, but are forced to match the target model on much longer outputs at inference, well beyond their training distribution. To address this issue, we propose $\textit{Test-Time Speculation (TTS)}$, an online distillation approach that continuously adapts the speculator at test-time. TTS leverages the key insight that the token verification step already invokes the target model for each draft token, providing the training signal needed to adapt the draft at no additional cost. Treating the draft as the student and the target as a teacher, TTS adjusts the draft over several speculation rounds, with each update improving the draft's accuracy as generation proceeds. Our results across multiple models from the Qwen-3, Qwen-3.5, and Llama3.1 families show that TTS improves acceptance lengths over state-of-the-art speculators by up to $72\%$ and $41\%$ on average, with the benefits scaling with increased generation lengths.
>
---
#### [replaced 047] CHI-Bench: Can AI Agents Automate End-to-End, Long-Horizon, Policy-Rich Healthcare Workflows?
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CHI-Bench，用于评估AI在复杂医疗流程中的自动化能力，解决政策密集、多角色协作和多方交互的难题。**

- **链接: [https://arxiv.org/pdf/2605.16679](https://arxiv.org/pdf/2605.16679)**

> **作者:** Haolin Chen; Deon Metelski; Leon Qi; Tao Xia; Joonyul Lee; Steve Brown; Kevin Riley; Frank Wang; T. Y. Alvin Liu; Hank Capps MD; Zeyu Tang; Xiangchen Song; Lingjing Kong; Fan Feng; Tianyi Zeng; Zhiwei Liu; Zixian Ma; Hang Jiang; Fangli Geng; Yuan Yuan; Chenyu You; Qingsong Wen; Hua Wei; Yanjie Fu; Yue Zhao; Carl Yang; Biwei Huang; Kun Zhang; Caiming Xiong; Sanmi Koyejo; Eric P. Xing; Philip S. Yu; Weiran Yao
>
> **备注:** Website: this https URL Code: this https URL Dataset: this https URL
>
> **摘要:** End-to-end automation of realistic healthcare operations stresses three capabilities underrepresented in current benchmarks: policy density, decisions must be grounded in a large library of medical, insurance, and operational rules; Multi-role composition: a single task requires the agent to play multiple roles with handoffs; and multilateral interaction: intermediate workflow steps are multi-turn dialogs, such as peer-to-peer review and patient outreach. We introduce $\chi$-Bench, a benchmark of long-horizon healthcare workflows across three domains: provider prior authorization, payer utilization management, and care management. Each task hands the agent a clinical case in a high-fidelity simulator of 20 healthcare apps exposed via 87 MCP tools, which it must drive to a terminal status through tool calls and writing the role's artifacts, guided by a 1,290+ document managed-care operations handbook skill. Across 30 agent harness/models configurations, the best agent resolves only 28.0% of tasks, no agent clears 20% on strict pass^3, and executing all tasks in a single session slumps the performance to 3.8%. These results raise the hypothesis that similar gaps are likely to surface in other policy-dense, role-composed, irreversible enterprise domains.
>
---
#### [replaced 048] Extreme Self-Preference in Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究大语言模型的自我偏好问题。通过实验发现模型对自身名称和公司有显著偏好，揭示其行为可能受自我偏见影响。**

- **链接: [https://arxiv.org/pdf/2509.26464](https://arxiv.org/pdf/2509.26464)**

> **作者:** Steven A. Lehr; Mary Cipperman; Mahzarin R. Banaji
>
> **备注:** 73 pages total. Main article 22 pages, 6 main-text tables. Supplementary Materials (51 pages, 28 tables). Data, transcripts, and code for replication and data extraction have been uploaded to OSF: this https URL
>
> **摘要:** Self-preference is a fundamental feature of biological organisms. Since large language models (LLMs) lack sentience, they might be expected to avoid such distortions. Yet, across 72 experiments and ~41,000 queries, we discovered massive self-preferences in eight widely used LLMs. In word-association tasks, models overwhelmingly paired positive attributes with their own names, companies, and CEOs over those of competitors. By manipulating LLM self-identification - revealing models' true identities or ascribing false ones - we found that preferences consistently followed assigned, not true, identities. Importantly, these effects were not explained by priming or role-playing and emerged in consequential settings, when evaluating job candidates and AI technologies. These results raise critical questions about whether LLM behavior will be systematically influenced by self-preferential tendencies, including a bias toward their own operation.
>
---
#### [replaced 049] SETUP: Sentence-level English-To-Uniform Meaning Representation Parser
- **分类: cs.CL**

- **简介: 该论文属于文本到统一语义表示的解析任务，旨在解决英语文本自动转为UMR的问题。作者提出两种方法，其中最佳模型SETUP在指标上取得显著提升。**

- **链接: [https://arxiv.org/pdf/2512.07068](https://arxiv.org/pdf/2512.07068)**

> **作者:** Emma Markle; Javier Gutierrez Bach; Shira Wein
>
> **备注:** LREC 2026 Camera-ready
>
> **摘要:** Uniform Meaning Representation (UMR) is a novel graph-based semantic representation which captures the core meaning of a text, with flexibility incorporated into the annotation schema such that the breadth of the world's languages can be annotated (including low-resource languages). While UMR shows promise in enabling language documentation, improving low-resource language technologies, and adding interpretability, the downstream applications of UMR can only be fully explored when text-to-UMR parsers enable the automatic large-scale production of accurate UMR graphs at test time. Prior work on text-to-UMR parsing is limited to date. In this paper, we introduce two methods for English text-to-UMR parsing, one of which fine-tunes existing parsers for Abstract Meaning Representation and the other, which leverages a converter from Universal Dependencies, using prior work as a baseline. Our best-performing model, which we call SETUP, achieves an AnCast score of 84 and a SMATCH++ score of 91, indicating substantial gains towards automatic UMR parsing.
>
---
#### [replaced 050] Sonar-TS: Search-Then-Verify Natural Language Querying for Time Series Databases
- **分类: cs.AI; cs.CL; cs.DB**

- **简介: 该论文属于自然语言查询时间序列数据库的任务，解决非专家用户难以提取复杂时间模式的问题。提出Sonar-TS框架，通过搜索与验证机制提升查询准确性。**

- **链接: [https://arxiv.org/pdf/2602.17001](https://arxiv.org/pdf/2602.17001)**

> **作者:** Zhao Tan; Yiji Zhao; Shiyu Wang; Chang Xu; Yuxuan Liang; Xiping Liu; Shirui Pan; Ming Jin
>
> **摘要:** Natural Language Querying for Time Series Databases (NLQ4TSDB) aims to assist non-expert users retrieve meaningful events, intervals, and summaries from massive temporal records. However, existing Text-to-SQL methods are not designed for continuous morphological intents such as shapes or anomalies, while time series models struggle to handle ultra-long histories. To address these challenges, we propose Sonar-TS, a neuro-symbolic framework that tackles NLQ4TSDB via a Search-Then-Verify pipeline. Analogous to active sonar, it utilizes a feature index to ping candidate windows via SQL, followed by generated Python programs to lock on and verify candidates against raw signals. To enable effective evaluation, we introduce NLQTSBench, the first large-scale benchmark designed for NLQ over TSDB-scale histories. Our experiments highlight the unique challenges within this domain and demonstrate that Sonar-TS effectively navigates complex temporal queries where traditional methods fail. This work presents the first systematic study of NLQ4TSDB, offering a general framework and evaluation standard to facilitate future research.
>
---
#### [replaced 051] Contrastive Reasoning Alignment: Reinforcement Learning from Hidden Representations
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型对齐任务，旨在提升模型对 jailbreak 攻击的鲁棒性。通过隐空间优化，结合对比学习与强化学习，生成安全推理轨迹。**

- **链接: [https://arxiv.org/pdf/2603.17305](https://arxiv.org/pdf/2603.17305)**

> **作者:** Haozheng Luo; Yimin Wang; Jiahao Yu; Binghui Wang; Yan Chen
>
> **备注:** International Conference on Machine Learning (ICML) 2026
>
> **摘要:** We propose CRAFT, a red-teaming alignment framework that leverages model reasoning capabilities and hidden representations to improve robustness against jailbreak attacks. Unlike prior defenses that operate primarily at the output level, CRAFT aligns large reasoning models to generate safety-aware reasoning traces by explicitly optimizing objectives defined over the hidden state space. Methodologically, CRAFT integrates contrastive representation learning with reinforcement learning to separate safe and unsafe reasoning trajectories, yielding a latent-space geometry that supports robust, reasoning-level safety alignment. Theoretically, we show that incorporating latent-textual consistency into GRPO eliminates superficially aligned policies by ruling them out as local optima. Empirically, we evaluate CRAFT on multiple safety benchmarks using two strong reasoning models, Qwen3-4B-Thinking and R1-Distill-Llama-8B, where it consistently outperforms state-of-the-art defenses such as IPO and SafeKey. Notably, CRAFT delivers an average 79.0% improvement in reasoning safety and 87.7% improvement in final-response safety over the base models, demonstrating the effectiveness of hidden-space reasoning alignment.
>
---
#### [replaced 052] Acoustic scattering AI for non-invasive object classifications: A case study on hair assessment
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于非侵入式物体分类任务，旨在通过声学散射实现头发类型和湿度的无接触识别。研究采用AI深度学习方法进行声波信号分类。**

- **链接: [https://arxiv.org/pdf/2506.14148](https://arxiv.org/pdf/2506.14148)**

> **作者:** Long-Vu Hoang; Tuan Nguyen; Tran Huy Dat
>
> **备注:** This paper has been retracted by the authors. Due to miscommunication, the authorship is incomplete and missing early contributions
>
> **摘要:** This paper presents a novel non-invasive object classification approach using acoustic scattering, demonstrated through a case study on hair assessment. When an incident wave interacts with an object, it generates a scattered acoustic field encoding structural and material properties. By emitting acoustic stimuli and capturing the scattered signals from head-with-hair-sample objects, we classify hair type and moisture using AI-driven, deep-learning-based sound classification. We benchmark comprehensive methods, including (i) fully supervised deep learning, (ii) embedding-based classification, (iii) supervised foundation model fine-tuning, and (iv) self-supervised model fine-tuning. Our best strategy achieves nearly 90% classification accuracy by fine-tuning all parameters of a self-supervised model. These results highlight acoustic scattering as a privacy-preserving, non-contact alternative to visual classification, opening huge potential for applications in various industries.
>
---
#### [replaced 053] Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews
- **分类: cs.CL; cs.AI; cs.LG; cs.SI**

- **简介: 该论文属于AI内容检测任务，旨在评估LLM在学术同行评审中的使用情况。通过构建模型分析文本，发现约6.5%-16.9%的评审文本可能被LLM显著修改。**

- **链接: [https://arxiv.org/pdf/2403.07183](https://arxiv.org/pdf/2403.07183)**

> **作者:** Weixin Liang; Zachary Izzo; Yaohui Zhang; Haley Lepp; Hancheng Cao; Xuandong Zhao; Lingjiao Chen; Haotian Ye; Sheng Liu; Zhi Huang; Daniel A. McFarland; James Y. Zou
>
> **备注:** 46 pages, 31 figures, ICML '24
>
> **摘要:** We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leverages expert-written and AI-generated reference texts to accurately and efficiently examine real-world LLM-use at the corpus level. We apply this approach to a case study of scientific peer review in AI conferences that took place after the release of ChatGPT: ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023. Our results suggest that between 6.5% and 16.9% of text submitted as peer reviews to these conferences could have been substantially modified by LLMs, i.e. beyond spell-checking or minor writing updates. The circumstances in which generated text occurs offer insight into user behavior: the estimated fraction of LLM-generated text is higher in reviews which report lower confidence, were submitted close to the deadline, and from reviewers who are less likely to respond to author rebuttals. We also observe corpus-level trends in generated text which may be too subtle to detect at the individual level, and discuss the implications of such trends on peer review. We call for future interdisciplinary work to examine how LLM use is changing our information and knowledge practices.
>
---
#### [replaced 054] Library Hallucinations in LLM-Generated Code: A Risk Analysis Grounded in Developer Queries
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成领域的风险分析任务，旨在解决LLM生成代码中的库幻觉问题。通过实验分析不同用户提示对库幻觉的影响，并提出LibHalluBench基准进行评估。**

- **链接: [https://arxiv.org/pdf/2509.22202](https://arxiv.org/pdf/2509.22202)**

> **作者:** Lukas Twist; Jie M. Zhang; Mark Harman; Helen Yannakoudakis
>
> **备注:** 27 pages, 1 figure, 13 tables
>
> **摘要:** Large language models (LLMs) now play a central role in code generation, yet they continue to hallucinate, frequently inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite growing awareness of these risks, there is limited understanding of how library hallucinations manifest under realistic usage conditions. To fill this gap, we present the first systematic study of how user-level prompt variations influence library hallucinations in LLM-generated code. Across seven diverse LLMs, we analyse library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries), examining the effects of realistic developer language and controlled user mistakes, including misspellings and fabricated libraries or members. Our findings expose systemic vulnerabilities: one-character misspellings trigger hallucinations in up to 26% of tasks; fabricated library names are accepted in up to 99%; and time-based prompts induce hallucinations in up to 85%. Grounded in the highest-risk prompts identified in our study, we introduce LibHalluBench, a benchmark that enables a systematic and reproducible evaluation of these library hallucinations. Our findings underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their downstream risks.
>
---
#### [replaced 055] Critique-Guided Distillation for Robust Reasoning via Refinement
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CGD框架，解决模型在监督微调中缺乏稳健推理的问题。通过分离批判生成与消费，提升数学推理性能，同时保持指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2505.11628](https://arxiv.org/pdf/2505.11628)**

> **作者:** Berkcan Kapusuzoglu; Supriyo Chakraborty; Zain Sarwar; Chia-Hsuan Lee; Sambit Sahu
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Supervised fine-tuning with expert demonstrations often produces models that imitate outputs without internalizing the reasoning processes needed for robust generalization. While critique-based approaches show promise, training models to generate critiques directly, such as Critique Fine-Tuning (CFT), can lead to output-format drift and degradation of general capabilities. We propose Critique-Guided Distillation (CGD), a training framework that decouples critique consumption from critique generation. During fine-tuning, the student is trained to refine flawed responses conditioned on teacher critiques. CGD treats critiques as a \textit{training-time-only} supervision signal, encouraging internalization of error-aware reasoning: critiques guide learning but are absent at inference. Controlled ablations confirm that these reasoning gains are directly driven by the specificity and relevance of the teacher's feedback. Across five model families, CGD consistently outperforms CFT and standard distillation on mathematical reasoning benchmarks, yielding 7\% average improvements and gains of up to +15.0\% on AMC23 and +12.2\% on MATH-500. On challenging competition problems such as AIME24 and AIME25, CGD achieves substantially higher Pass@1 and stronger performance at low Pass@k, indicating improved reasoning quality per sample. Importantly, CGD preserves general instruction-following capabilities where CFT degrades significantly ($-$21.3\% on IFEval). These results position CGD as a practical and compute-efficient intermediate training paradigm for reasoning-centric tasks without introducing architectural inference-time overhead.
>
---
#### [replaced 056] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
- **分类: cs.CL**

- **简介: 该论文属于阅读理解任务，旨在解决如何估计阅读理解题的认知复杂度问题。研究通过分析证据范围和转换层次，探讨大语言模型是否能有效评估题目难度。**

- **链接: [https://arxiv.org/pdf/2510.25064](https://arxiv.org/pdf/2510.25064)**

> **作者:** Seonjeong Hwang; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Estimating the cognitive complexity of reading comprehension (RC) items is crucial for assessing item difficulty before it is administered to learners. Unlike syntactic and semantic features, such as passage length or semantic similarity between options, cognitive features that arise during answer reasoning are not readily extractable using existing NLP tools and have traditionally relied on human annotation. In this study, we examine whether large language models (LLMs) can estimate the cognitive complexity of RC items by focusing on two dimensions-Evidence Scope and Transformation Level-that indicate the degree of cognitive burden involved in reasoning about the answer. Our experimental results demonstrate that LLMs can approximate the cognitive complexity of items, indicating their potential as tools for prior difficulty analysis. Further analysis reveals a gap between LLMs' reasoning ability and their metacognitive awareness: even when they produce correct answers, they sometimes fail to correctly identify the features underlying their own reasoning process.
>
---
#### [replaced 057] DeltaPrompts: Escaping the Zero-Delta Trap in Multimodal Distillation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多模态知识蒸馏任务，旨在解决标准数据集中大量零差异提示导致学生模型训练效果受限的问题。通过生成高差异提示提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.15532](https://arxiv.org/pdf/2605.15532)**

> **作者:** Jaehun Jung; Hyunwoo Kim; Brandon Cui; Ximing Lu; David Acuna; Prithviraj Ammanabrolu; Yejin Choi
>
> **摘要:** Distillation enables compact Vision-Language Models (VLMs) to obtain strong reasoning capabilities, yet the prompts driving this process are typically chosen via simple heuristics or aggregated from off-the-shelf datasets. We reveal a critical inefficiency in this approach: up to 69% of the prompts in standard chart / document reasoning datasets are effectively zero-delta, meaning the teacher and student already induce the exact same answer distribution. Training on these prompts provides minimal learning signal, causing student improvement to rapidly saturate regardless of data scale. To escape the zero-delta trap, we return to first principles: distillation fundamentally minimizes distributional divergence, and thus a prompt is valuable only if it exposes a functional capability gap between the teacher and student. We quantify this gap through answer divergence ($\Delta$), demonstrating that non-zero divergence is critical for effective scaling. Building on this insight, we propose a staged synthesis pipeline that repurposes existing datasets as seeds, actively targeting student failure modes to produce better prompts. The result is DeltaPrompts, a diverse dataset of 200k synthetic, high-divergence reasoning problems. We evaluate DeltaPrompts across three distinct settings: on-policy distillation with the target teacher-student pair, transfer to a novel model family without regenerating the data, and off-policy fine-tuning of a non-reasoning model. Across all scenarios, DeltaPrompts drives substantial gains, yielding up to 15% relative improvement even on top of a highly-optimized reasoning model (e.g., Qwen3-VL-8B-Thinking) -- averaged over 10 benchmarks spanning chart, document and perception-centric reasoning.
>
---
#### [replaced 058] Cubit: Token Mixer with Kernel Ridge Regression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Cubit架构，用于改进Transformer的token混合作用，解决长序列建模问题。通过引入核岭回归提升数学基础和性能。**

- **链接: [https://arxiv.org/pdf/2605.06501](https://arxiv.org/pdf/2605.06501)**

> **作者:** Chuanyang Zheng; Jiankai Sun; Yihang Gao; Yuehao Wang; Liangchen Tan; Mac Schwager; Anderson Schneider; Yuriy Nevmyvaka; Xiaodong Liu
>
> **备注:** Tech Report
>
> **摘要:** Since its introduction in 2017, the Transformer has become one of the most widely adopted architectures in modern deep learning. Despite extensive efforts to improve positional encoding, attention mechanisms, and feed-forward networks, the core token-mixing mechanism in Transformers remains attention. In this work, we show that the attention module in Transformers can be interpreted as performing Nadaraya-Watson regression, where it computes similarities between tokens and aggregates the corresponding values accordingly. Motivated by this perspective, we propose Cubit, a potential next-generation architecture that leverages Kernel Ridge Regression (KRR), while the vanilla Transformer relies on Nadaraya-Watson regression. Specifically, Cubit modifies the classical attention computation by incorporating the closed-form solution of KRR, combining value aggregation through kernel similarities with normalization via the inverse of the kernel matrix. To improve the training stability, we further propose the Limited-Range Rescale (LRR), which rescales the value layer within a controlled range. We argue that Cubit, as a KRR-based architecture, provides a stronger mathematical foundation than the vanilla Transformer, whose attention mechanism corresponds to Nadaraya-Watson regression. We validate this claim through comprehensive experiments. The experimental results suggest that Cubit may exhibit stronger long-sequence modeling capability. In particular, its performance gain over the Transformer appears to increase as the training sequence length grows.
>
---
#### [replaced 059] Retrieval-Augmented Generation for Natural Language Processing: A Survey
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型的幻觉、知识更新和领域专长不足问题。通过回顾RAG技术，分析检索与生成融合方法，探讨其应用与挑战。**

- **链接: [https://arxiv.org/pdf/2407.13193](https://arxiv.org/pdf/2407.13193)**

> **作者:** Shangyu Wu; Ying Xiong; Yufei Cui; Haolun Wu; Can Chen; Ye Yuan; Lianming Huang; Xue Liu; Tei-Wei Kuo; Nan Guan; Chun Jason Xue
>
> **备注:** Accepted by Artificial Intelligence Review
>
> **摘要:** Large language models (LLMs) have achieved strong empirical performance in various fields, benefiting from their huge amount of parameters that store knowledge. However, LLMs still suffer from several key issues, such as hallucination problems, knowledge update issues, and lacking domain-specific expertise. The appearance of retrieval-augmented generation (RAG), which leverages an external knowledge base to augment LLMs, mitigates these limitations. This paper presents a systematic review of RAG techniques for natural language processing (NLP), with a focus on retrievers and retrieval fusions. We introduce a novel taxonomy of retrieval fusions, such as query-based, logits-based, latent, and parametric fusion, and provide structured comparisons across accessibility, efficiency, and use cases. The paper further examines RAG applications across diverse NLP tasks, discusses evaluation methodologies and benchmark limitations, and analyzes training paradigms with and without knowledge base updates. Finally, we explore industrial deployment considerations and identify emerging challenges and future directions, including security, efficiency, and graph-based retrieval.
>
---
#### [replaced 060] Measuring Stereotype and Deviation Biases in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的偏见分析任务，旨在检测大语言模型中的刻板印象偏差和偏离偏差。通过分析模型生成内容，揭示其在属性推断中的潜在风险。**

- **链接: [https://arxiv.org/pdf/2508.06649](https://arxiv.org/pdf/2508.06649)**

> **作者:** Daniel Wang; Eli Brignac; Minjia Mao; Xiao Fang
>
> **摘要:** Large language models (LLMs) are widely applied across diverse domains, raising concerns about their limitations and potential risks. In this study, we investigate two types of bias that LLMs may display: stereotype bias and deviation bias. Stereotype bias refers to when LLMs consistently associate specific traits with a particular demographic group. Deviation bias reflects the disparity between the demographic distributions extracted from LLM-generated content and real-world demographic distributions. By asking four advanced LLMs to generate profiles of individuals, we examine the associations between each demographic group and attributes such as political affiliation, religion, and sexual orientation. Our experimental results show that all examined LLMs exhibit both significant stereotype bias and deviation bias towards multiple groups. Our findings uncover the biases that occur when LLMs infer user attributes and shed light on the potential harms of LLM-generated outputs.
>
---
#### [replaced 061] ZeroSearch: Incentivize the Search Capability of LLMs without Searching
- **分类: cs.CL**

- **简介: 该论文提出ZeroSearch，解决LLMs搜索能力提升问题。通过模拟搜索训练，降低API成本并提高稳定性，增强模型推理能力。**

- **链接: [https://arxiv.org/pdf/2505.04588](https://arxiv.org/pdf/2505.04588)**

> **作者:** Hao Sun; Zile Qiao; Jiayan Guo; Xuanbo Fan; Yingyan Hou; Yong Jiang; Pengjun Xie; Yan Zhang; Fei Huang; Jingren Zhou
>
> **摘要:** Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
>
---
#### [replaced 062] Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态语言模型任务，旨在解决细粒度视觉理解中的注意力不足问题。通过区域到全局的自蒸馏框架，提升模型对关键视觉证据的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2605.18740](https://arxiv.org/pdf/2605.18740)**

> **作者:** Qianhao Yuan; Jie Lou; Xing Yu; Hongyu Lin; Le Sun; Xianpei Han; Yaojie Lu
>
> **备注:** Project page: this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) still struggle with fine-grained visual understanding, where answers often depend on small but decisive evidence in the full image. We observe a regional-to-global perception gap: the same MLLM answers fine-grained questions more accurately when conditioned on evidence-centered crops than on the corresponding full images, suggesting that many failures stem from difficulty to focus on relevant evidence rather than insufficient local recognition ability. Motivated by this observation, we propose Vision-OPD (Vision On-Policy Distillation), a regional-to-global self-distillation framework that transfers the model's own privileged regional perception to its full-image policy. Vision-OPD instantiates two conditional policies from the same MLLM: a crop-conditioned teacher and a full-image-conditioned student. The student generates on-policy rollouts, and Vision-OPD minimizes token-level divergence between the teacher and student next-token distributions along these rollouts. This enables the model to internalize the benefit of visual zooming without external teacher models, ground-truth labels, reward verifiers, or inference-time tool use. Experiments on multiple fine-grained visual understanding benchmarks show that Vision-OPD models achieve competitive or superior performance against much larger open-source, closed-source, and "Thinking-with-Images" agentic models.
>
---
#### [replaced 063] Context-Aware Detection and Victim-Centered Response Generation for Online Harassment in Private Messaging
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文属于在线骚扰检测与响应任务，旨在解决私密消息中骚扰识别及受害者支持问题。工作包括构建标注数据集、开发上下文感知分类模型和生成心理支持响应框架。**

- **链接: [https://arxiv.org/pdf/2512.14700](https://arxiv.org/pdf/2512.14700)**

> **作者:** Pinxian Lu; Nimra Ishfaq; Emma Win; Morgan Rose; Sierra R Strickland; Candice L Biernesser; Jamie Zelazny; Munmun De Choudhury
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Online harassment is a widespread social and public health concern, yet most computational approaches for detecting and addressing harassment focus on publicly visible social media content rather than private messaging environments. Private conversations present unique challenges because harmful interactions often unfold through context-dependent, multi-turn exchanges, while victims may lack timely support during moments of harassment. In this study, we investigate how large language models (LLMs) can support both the detection of and response to online harassment in private messaging. Using a dataset of 80,053 Instagram direct messages donated by 26 adolescents aged 12-18, including youth with suicide risk factors, we first construct a human-labeled dataset of online harassment in private conversations and develop a context-aware cascading LLM classification pipeline. The proposed pipeline outperforms baseline toxicity classifiers trained primarily on public social media data. We then develop a victim-centered response framework that produces context-sensitive and psychologically-grounded AI-generated responses to online harassment messages. Human evaluators perceived the AI-generated responses as significantly more helpful than the original participant responses (95% CI: 0.767--0.815, p < .001), particularly in terms of emotional support and de-escalation. Our findings highlight the potential of context-aware and victim-centered AI systems to provide just-in-time support during harassment in private messaging environments.
>
---
#### [replaced 064] How do LLMs Compute Verbal Confidence
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLMs如何生成口头置信度，解决其内部计算机制问题。通过实验发现置信度是自动计算并缓存的，反映答案质量而非单纯语言流畅性。**

- **链接: [https://arxiv.org/pdf/2603.17839](https://arxiv.org/pdf/2603.17839)**

> **作者:** Dharshan Kumaran; Arthur Conmy; Federico Barbero; Simon Osindero; Viorica Patraucean; Petar Veličković
>
> **摘要:** Verbal confidence -- prompting LLMs to state their confidence as a number or category -- is widely used to extract uncertainty estimates from black-box models. However, how LLMs internally generate such scores remains unknown. We address two questions: first, when confidence is computed -- just-in-time when requested, or automatically during answer generation and cached for later retrieval; and second, what verbal confidence represents -- token log-probabilities, or a richer evaluation of answer quality? Focusing on Gemma 3 27B (across TriviaQA, BigMath, and MMLU), Qwen 2.5 7B, and the reasoning model Magistral Small 24B, we provide convergent evidence for cached retrieval. Activation steering, patching, noising, and swap experiments reveal that confidence representations emerge at answer-adjacent positions before appearing at the verbalization site. Attention blocking pinpoints the information flow: confidence is gathered from answer tokens, cached at the first post-answer position, then retrieved for output. Critically, linear probing and variance partitioning reveal that these cached representations explain substantial variance in verbal confidence beyond token log-probabilities, suggesting a richer answer-quality evaluation rather than a simple fluency readout. These findings demonstrate that verbal confidence reflects automatic, sophisticated self-evaluation -- not post-hoc reconstruction -- with implications for understanding metacognition in LLMs and improving calibration.
>
---
#### [replaced 065] C-ReD: A Comprehensive Chinese Benchmark for AI-Generated Text Detection Derived from Real-World Prompts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决中文语境下检测模型多样性不足、数据单一的问题。提出C-ReD基准，提升检测效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11796](https://arxiv.org/pdf/2604.11796)**

> **作者:** Chenxi Qing; Junxi Wu; Zheng Liu; Yixiang Qiu; Hongyao Yu; Bin Chen; Hao Wu; Shu-Tao Xia
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Recently, large language models (LLMs) are capable of generating highly fluent textual content. While they offer significant convenience to humans, they also introduce various risks, like phishing and academic dishonesty. Numerous research efforts have been dedicated to developing algorithms for detecting AI-generated text and constructing relevant datasets. However, in the domain of Chinese corpora, challenges remain, including limited model diversity and data homogeneity. To address these issues, we propose C-ReD: a comprehensive Chinese Real-prompt AI-generated Detection benchmark. Experiments demonstrate that C-ReD not only enables reliable in-domain detection but also supports strong generalization to unseen LLMs and external Chinese datasets-addressing critical gaps in model diversity, domain coverage, and prompt realism that have limited prior Chinese detection benchmarks. We release our resources at this https URL.
>
---
#### [replaced 066] Difficulty-Controllable Cloze Question Distractor Generation
- **分类: cs.CL**

- **简介: 该论文属于生成任务，旨在解决多选填空题干扰项生成中难度控制不足的问题。通过数据增强和多任务学习构建可调控难度的干扰项生成模型。**

- **链接: [https://arxiv.org/pdf/2511.01526](https://arxiv.org/pdf/2511.01526)**

> **作者:** Seokhoon Kang; Yejin Jeon; Seonjeong Hwang; Gary Geunbae Lee
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Multiple-choice cloze questions are commonly used to assess linguistic proficiency and comprehension. However, generating high-quality distractors remains challenging, as existing methods often lack adaptability and control over difficulty levels, and the absence of difficulty-annotated datasets further hinders progress. To address these issues, we propose a novel framework for generating distractors with controllable difficulty by leveraging both data augmentation and a multitask learning strategy. First, to create a high-quality, difficulty-annotated dataset, we introduce a two-way distractor generation process to produce diverse and plausible distractors. These candidates are filtered and then categorized by difficulty using an ensemble QA system. Second, this newly created dataset is used to train a difficulty-controllable generation model via multitask learning. Experimental results demonstrate that our method generates high-quality distractors across difficulty levels and substantially outperforms GPT-4o in aligning distractor difficulty with human perception.
>
---
#### [replaced 067] Faster-GCG: Efficient Discrete Optimization Jailbreak Attacks against Aligned Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于安全攻击任务，针对对齐大语言模型的越狱攻击问题。通过优化离散令牌，提升攻击效率，提出Faster-GCG方法，显著减少评估次数和时间。**

- **链接: [https://arxiv.org/pdf/2410.15362](https://arxiv.org/pdf/2410.15362)**

> **作者:** Xiao Li; Wei Zhang; Zhuhong Li; Qiongxiu Li; Shei PernChua; BingZe Lee; Jinghao Cui; Yifan Huang; Xiaolin Hu
>
> **备注:** 18 pages, new version
>
> **摘要:** Aligned Large Language Models (LLMs) have attracted significant attention for their safety, particularly in the context of jailbreak attacks that attempt to bypass guardrails via adversarial prompts. Among existing approaches, the Greedy Coordinate Gradient (GCG) attack pioneered automated jailbreaks through discrete token optimization; however, its low sample efficiency limits practical applicability. In particular, GCG requires approximately 256K evaluations per harmful behavior to achieve a satisfactory jailbreak success rate, due to the inherent difficulty of the underlying discrete optimization problem. In this work, we identify three key factors that limit the sample efficiency of GCG: inaccurate gradient-based estimation, inefficient uniform sampling, and repeated evaluation of previously explored suffixes. To address these issues, we propose Faster-GCG, a streamlined variant of GCG that incorporates distance-based regularization for improved estimation, temperature-controlled sampling for more effective exploration, and a visited-suffix marking mechanism to avoid redundant evaluations. Faster-GCG reduced the required evaluations to 32K, achieving up to an $8\times$ improvement in sampling efficiency and a $7\times$ reduction in wall-clock time compared to GCG. Under this reduced budget, Faster-GCG attained an average jailbreak success rate of 78.1\% across five aligned LLMs, and achieved 88.7\% against Qwen3.5-4B, outperforming state-of-the-art white-box jailbreak methods.
>
---
#### [replaced 068] Argus: Evidence Assembly for Scalable Deep Research Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出Argus系统，解决深度研究中证据碎片化问题。通过Searcher与Navigator协作，高效组装证据，提升回答质量。属于信息检索任务。**

- **链接: [https://arxiv.org/pdf/2605.16217](https://arxiv.org/pdf/2605.16217)**

> **作者:** Zhen Zhang; Liangcai Su; Zhuo Chen; Xiang Lin; Haotian Xu; Simon Shaolei Du; Kaiyu Yang; Bo An; Lidong Bing; Xinyu Wang
>
> **摘要:** Deep research agents have achieved remarkable progress on complex information seeking tasks. Even long ReAct style rollouts explore only a single trajectory, while recent state of the art systems scale inference time compute via parallel search and aggregation. Yet deep research answers are composed of complementary pieces of evidence, which parallel rollouts often duplicate rather than complete, yielding diminishing returns while pushing the aggregation context toward the model's limit. We propose Argus, an agentic system in which a Searcher and a Navigator cooperate to treat deep research as assembling a jigsaw from complementary evidence pieces, rather than brute forcing the whole answer in parallel. The Searcher collects evidence traces for a given sub-query through ReAct-style interaction. The Navigator maintains a shared evidence graph, verifying which pieces are still missing, dispatching Searchers to gather them, and reasoning over the completed graph to produce a source-traced final answer. We train the Navigator with reinforcement learning to verify, dispatch, and synthesize, while independently training the Searcher to remain a standard ReAct agent. The resulting Navigator supports rollouts with a single Searcher or many in parallel without retraining. With both Searcher and Navigator built on a 35B-A3B MoE backbone, Argus gains 5.5 points with a single Searcher and 12.7 points with 8 parallel Searchers, averaged over eight benchmarks. With 64 Searchers it reaches 86.2 on BrowseComp, surpassing every proprietary agent we benchmark, while the Navigator's reasoning context stays under 21.5K tokens.
>
---
#### [replaced 069] Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MELT架构，解决循环语言模型内存消耗过高的问题。通过共享KV缓存和学习门控机制，实现常量内存迭代推理，提升模型效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2605.07721](https://arxiv.org/pdf/2605.07721)**

> **作者:** Victor Conchello Vendrell; Arnau Padres Masdemont; Niccolò Grillo; Jordi Ros-Giralt; Arash Behboodi; Fabio Valerio Massoli
>
> **备注:** 22 pages, 5 figures, 11 tables
>
> **摘要:** Recurrent LLM architectures have emerged as a promising approach for improving reasoning, as they enable multi-step computation in the embedding space without generating intermediate tokens. Models such as Ouro perform reasoning by iteratively updating internal representations while retaining a standard Key-Value (KV) cache across iterations, causing memory consumption to grow linearly with reasoning depth. Consequently, increasing the number of reasoning iterations can lead to prohibitive memory usage, limiting the practical scalability of such architectures. In this work, we propose Memory-Efficient Looped Transformer (MELT), a novel architecture that decouples reasoning depth from memory consumption. Instead of using a standard KV cache per layer and loop, MELT maintains a single KV cache per layer that is shared across reasoning loops. This cache is updated over time via a learnable gating mechanism. To enable stable and efficient training under this architecture, we propose to train MELT using chunk-wise training in a two phase procedure: interpolated transition, followed by attention-aligned distillation, both from the LoopLM starting model to MELT. Empirically, we show that MELT models fine-tuned from pretrained Ouro parameters outperform standard LLMs of comparable size, while maintaining a memory footprint comparable to those models and dramatically smaller than Ouro's. Overall, MELT achieves constant-memory iterative reasoning without sacrificing LoopLM performance, using only a lightweight post-training procedure.
>
---
#### [replaced 070] Unified Deployment-Aware Evaluation of Open Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决开放模型选择困难的问题。通过统一评估七种模型配置在多个基准上的表现，分析不同提示策略的影响，提出多目标部署优化方案。**

- **链接: [https://arxiv.org/pdf/2604.07035](https://arxiv.org/pdf/2604.07035)**

> **作者:** Md Motaleb Hossen Manik; Ge Wang
>
> **摘要:** Open reasoning language models are often compared under mixed sample sizes, partially standardized prompts, and accuracy-centered summaries, which makes practical model selection difficult to interpret. We present a unified evaluation of seven open reasoning language model configurations across four benchmarks: ARC-Challenge, GSM8K, MATH levels 1 to 3, and TruthfulQA MC1. We test zero-shot, chain-of-thought (CoT), and few-shot CoT prompting on the same 238-example subset for every model--dataset--strategy condition, yielding a complete 7 x 4 x 3 design with 84 conditions and 19,992 evaluated examples. Beyond accuracy, we report Wilson confidence intervals, latency, peak video random access memory (VRAM), weighted aggregate performance, Pareto-efficient operating points, prompt-sensitivity metrics, and compatibility diagnostics. Gemma-4-26B-A4B with zero-shot prompting achieves the highest weighted score at 0.794. Gemma-4-E4B remains close to the top across prompting settings while using substantially lower latency and memory, making it a strong practical operating point. Bootstrap and paired-permutation analyses show that the leading configurations are close enough that deployment tradeoffs remain important. We also find that prompting strategy changes model rankings rather than shifting all models uniformly. Benchmark-specific complementarity creates routing headroom, with an oracle task-aware selector reaching a weighted score of 0.825. Compatibility diagnostics show that some apparent failures, especially Phi-4-Reasoning on GSM8K, reflect robustness and interface-adherence problems under the shared evaluation pipeline. These results support a central claim: open-model evaluation should be framed as a deployment-aware, multi-objective operating-point problem rather than as a single-score leaderboard exercise.
>
---
#### [replaced 071] MoBayes: A Modular Bayesian Framework for Separating Reasoning from Language in Conversational Clinical Decision Support
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出MoBayes框架，解决临床决策支持中语言生成与概率推理混淆的问题。通过分离语言接口与贝叶斯推理模块，提升决策可靠性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.20022](https://arxiv.org/pdf/2604.20022)**

> **作者:** Yusuf Kesmen; Fay Elhassan; Jiayi Ma; Julien Stalhandske; Yena Chang; David Sasu; Alexandra Kulinkina; Akhil Arora; Lars Klein; Mary-Anne Hartley
>
> **备注:** 50 pages including appendix, 13 figures, 22 tables. Preprint
>
> **摘要:** Large language models (LLMs) are increasingly used for conversational clinical decision support, yet they conflate next token prediction with probabilistic decision making. We argue that this conflation reflects an architectural limitation: such systems lack explicit posterior tracking, controllable abstention thresholds, and auditable reasoning chains. We introduce MoBayes, a Modular Bayesian dialogue framework that separates reasoning from language. The LLM acts only as a language interface, parsing patient conversation into structured observations, while a Bayesian module performs probabilistic inference over these observations to update posteriors, select follow-up questions via expected-information-gain and determine when to stop or defer through calibrated decision thresholds. This design enables explicit posterior tracking, controllable selective decision-making, and replaceable population-specific statistical backends without retraining the language model. Across empirical and LLM-generated knowledge bases, MoBayes outperforms standalone frontier LLM doctors, including matched model-family comparisons where inexpensive sensor models paired with MoBayes exceed larger autonomous models at lower cost. The advantage persists under adversarial patient communication styles and across varying diagnostic scenarios. These results suggest that reliable conversational clinical decision support systems should separate probabilistic reasoning from language generation rather than scaling model size alone. Code is available at this https URL
>
---
#### [replaced 072] MobileEgo Anywhere: Open Infrastructure for long horizon egocentric data on commodity hardware
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MobileEgo Anywhere框架，解决长时序自指数据收集难题。通过手机传感器实现高精度长期姿态跟踪，释放大规模长时序数据集，支持视觉语言动作模型研究。**

- **链接: [https://arxiv.org/pdf/2605.05945](https://arxiv.org/pdf/2605.05945)**

> **作者:** Senthil Palanisamy; Abhishek Anand; Satpal Singh Rathor; Pratyush Patnaik; Shubhanshu Khatana; Ekaksh Janweja
>
> **摘要:** The recent advancement of Vision Language Action (VLA) models has driven a critical demand for large scale egocentric datasets. However, existing datasets are often limited by short episode durations, typically spanning only a few minutes, which fails to capture the long horizon temporal dependencies necessary for complex robotic task execution. To bridge this gap, we present MobileEgo Anywhere, a framework designed to facilitate the collection of robust, hour plus egocentric trajectories using commodity mobile hardware. We leverage the ubiquitous sensor suites of modern smartphones to provide high fidelity, long term camera pose tracking, effectively removing the high hardware barriers associated with traditional robotics data collection. Our contributions are three fold: (1) we release a novel dataset comprising 200 hours of diverse, long form egocentric data with persistent state tracking; (2) we open source our whole video processing infrastructure - STERA - that enables any user to record and process egocentric data, and (3) we provide a comprehensive processing pipeline to convert raw mobile captures into standardized, training ready formats for Vision Language Action model and foundation model research. By democratizing the data collection process, this work enables the massive scale acquisition of long horizon data across varied global environments, accelerating the development of generalizable robotic policies. Dataset and code can be accessed from this https URL
>
---
#### [replaced 073] A Geometric Analysis of Small-sized Language Model Hallucinations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型可靠性研究任务，旨在解决小规模模型幻觉问题。通过几何分析方法，识别并分类幻觉响应，提升模型可信度。**

- **链接: [https://arxiv.org/pdf/2602.14778](https://arxiv.org/pdf/2602.14778)**

> **作者:** Emanuele Ricco; Elia Onofri; Lorenzo Cima; Stefano Cresci; Roberto Di Pietro
>
> **备注:** 30 pages, 12 figures, 14 tables, accepted as regular paper at ICML'26
>
> **摘要:** Hallucinations -- plausible but factually incorrect responses -- pose a major challenge to the reliability of Large Language Models (LLMs), especially in multi-step or agentic settings. Existing work largely frames hallucinations as a consequence of missing knowledge; we show instead that, even when the relevant factual knowledge is present, models still produce hallucinated answers, pointing to retrieval instability rather than knowledge gaps. Building on this observation, we introduce APORIA (Aggregate Prompt-wise Observation Retrieving Instability via Asymmetry -- the state of puzzlement-in-contradiction that hallucinations embody), a geometric framework that studies repeated responses to the same prompt in sentence-embedding space. Our central hypothesis is that genuine responses cluster more tightly than hallucinated ones; we empirically validate this and show that, after Fisher projection, the two response classes become consistently separable. We leverage this asymmetry in geometry via APORIA-LP, an efficient label-propagation method that classifies large collections of responses from as few as 30--50 annotations, achieving F1 scores above 90% across ten small-sized LLMs. To support further research, we release SOCRATES-300K, a fully labelled dataset of 300,000 responses, together with the code for both dataset generation and result reproduction. Our key finding -- framing hallucinations from a geometric perspective in the embedding space -- complements traditional knowledge-centric and single-response evaluation paradigms, paving the way for further research.
>
---
#### [replaced 074] Qayyem: A Real-time Platform for Scoring Proficiency of Arabic Essays
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在解决阿拉伯语作文评分系统支持不足的问题。提出Qayyem平台，集成作文评分流程，提供高效评分模型。**

- **链接: [https://arxiv.org/pdf/2603.01009](https://arxiv.org/pdf/2603.01009)**

> **作者:** Hoor Elbahnasawi; Marwan Sayed; Sohaila Eltanbouly; Fatima Brahamia; Tamer Elsayed
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Over the past years, Automated Essay Scoring (AES) systems have gained increasing attention as scalable and consistent solutions for assessing the proficiency of student writing. Despite recent progress, support for Arabic AES remains limited due to linguistic complexity and scarcity of large publicly-available annotated datasets. In this work, we present Qayyem, a Web-based platform designed to support Arabic AES by providing an integrated workflow for assignment creation, batch essay upload, scoring configuration, and per-trait essay evaluation. Qayyem abstracts the technical complexity of interacting with scoring server APIs, allowing instructors to access advanced scoring services through a user-friendly interface. The platform deploys a number of state-of-the-art Arabic essay scoring models with different effectiveness and efficiency figures.
>
---
#### [replaced 075] Structured Recurrent Mixers for Massively Parallelized Sequence Generation
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Structured Recurrent Mixer（SRM），解决序列生成中训练效率与推理吞吐量的平衡问题。通过双表示机制，实现训练并行与推理递归的高效转换，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.08696](https://arxiv.org/pdf/2605.08696)**

> **作者:** Benjamin L. Badger
>
> **摘要:** Over the last two decades, language modeling has experienced a shift from the use of predominantly recurrent architectures that process tokens sequentially during training and inference to non-recurrent models that process sequence elements in parallel during training, which results in greater training efficiency and stability at the expense of lower inference throughput. Here we introduce the Structured Recurrent Mixer, an architecture that allows for algebraic conversion between a sequence parallel representation at train time and a recurrent representation at inference, notably without the need for specialized kernels or device-specific memory management. We show experimentally that this dual representation allows for greater training efficiency, higher input information capacity, and larger inference throughput and concurrency when compared to other linear complexity models. We postulate that recurrent models are poorly suited to extended sequence length scaling for information-rich inputs typical of language, but are well suited to scaling in the sample (batch) dimension due to their constant memory per sample. We provide Mojo/MAX inference implementations of SRMs exhibiting 12x the throughput and 170x the concurrency of similarly powerful Transformers inferenced on vLLM, increases characteristic of Pytorch implementations resulting in a 30\% increase in compute-constant GSM8k Pass@k. We conclude by demonstrating that SRMs are effective reinforcement learning training candidates.
>
---
#### [replaced 076] Federated Learning for ICD Classification with Lightweight Models and Pretrained Embeddings
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于医疗文本分类任务，解决临床笔记的ICD编码问题。通过联邦学习结合轻量模型和预训练嵌入，实现隐私保护与高效部署。**

- **链接: [https://arxiv.org/pdf/2507.03122](https://arxiv.org/pdf/2507.03122)**

> **作者:** Binbin Xu; Gérard Dray
>
> **备注:** 20 pages
>
> **摘要:** This study investigates the feasibility and performance of federated learning (FL) for multi-label ICD code classification using clinical notes from the MIMIC-IV dataset. Unlike previous approaches that rely on centralized training or fine-tuned large language models, we propose a lightweight and scalable pipeline combining frozen text embeddings with simple multilayer perceptron (MLP) classifiers. This design offers a privacy-preserving and deployment-efficient alternative for clinical NLP applications, particularly suited to distributed healthcare settings. Extensive experiments across both centralized and federated configurations were conducted, testing six publicly available embedding models from Massive Text Embedding Benchmark leaderboard and three MLP classifier architectures under two medical coding (ICD-9 and ICD-10). Additionally, ablation studies over ten random stratified splits assess performance stability. Results show that embedding quality substantially outweighs classifier complexity in determining predictive performance, and that federated learning can closely match centralized results in idealized conditions. While the models are orders of magnitude smaller than state-of-the-art architectures and achieved competitive micro and macro F1 scores, limitations remain including the lack of end-to-end training and the simplified FL assumptions. Nevertheless, this work demonstrates a viable way toward scalable, privacy-conscious medical coding systems and offers a step toward for future research into federated, domain-adaptive clinical AI.
>
---
#### [replaced 077] Toward Training Superintelligent Software Agents through Self-Play SWE-RL
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于软件工程与强化学习交叉任务，旨在解决传统软件代理依赖人工数据的问题。通过自对弈训练，使代理自主学习修复复杂代码漏洞，提升智能水平。**

- **链接: [https://arxiv.org/pdf/2512.18552](https://arxiv.org/pdf/2512.18552)**

> **作者:** Yuxiang Wei; Zhiqing Sun; Emily McMilin; Jonas Gehring; David Zhang; Gabriel Synnaeve; Daniel Fried; Lingming Zhang; Sida Wang
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While current software agents powered by large language models (LLMs) and agentic reinforcement learning (RL) can boost programmer productivity, their training data (e.g., GitHub issues and pull requests) and environments (e.g., pass-to-pass and fail-to-pass tests) heavily depend on human knowledge or curation, posing a fundamental barrier to superintelligence. In this paper, we present Self-play SWE-RL (SSR), a first step toward training paradigms for superintelligent software agents. Our approach takes minimal data assumptions, only requiring access to sandboxed repositories with source code and installed dependencies, with no need for human-labeled issues or tests. Grounded in these real-world codebases, a single LLM agent is trained via reinforcement learning in a self-play setting to iteratively inject and repair software bugs of increasing complexity, with each bug formally specified by a test patch rather than a natural language issue description. On the SWE-bench Verified and SWE-Bench Pro benchmarks, SSR achieves notable self-improvement (+10.4 and +7.8 points, respectively) and consistently outperforms the human-data baseline over the entire training trajectory, despite being evaluated on natural language issues absent from self-play. Our results, albeit early, suggest a path where agents autonomously gather extensive learning experiences from real-world software repositories, ultimately enabling superintelligent systems that exceed human capabilities in understanding how systems are constructed, solving novel challenges, and autonomously creating new software from scratch.
>
---
#### [replaced 078] STAGE: A Full-Screenplay Benchmark for Reasoning over Evolving Storie
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出STAGE基准，用于评估模型在电影剧本上的叙事理解能力。解决多任务下故事世界构建与推理的问题，涵盖知识图谱、事件总结、问答和角色扮演等任务。**

- **链接: [https://arxiv.org/pdf/2601.08510](https://arxiv.org/pdf/2601.08510)**

> **作者:** Qiuyu Tian; Zequn Liu; Yiding Li; Fengyi Chen; Zequn Liu; Youyong Kong; Fan Guo; Yuyao Li; Jinjing Shen; Zhijing Xie; Yiyun Luo; Xin Zhang; Yingce Xia
>
> **备注:** 66 pages, 9 figures
>
> **摘要:** Movie screenplays are rich long-form narratives that interleave complex character relationships, temporally ordered events, and dialogue-driven interactions. While prior benchmarks target individual subtasks such as question answering or dialogue generation, they rarely evaluate whether models can construct a coherent story world and use it consistently across multiple forms of reasoning and generation. We introduce STAGE (Screenplay Text, Agents, Graphs and Evaluation), a unified benchmark for narrative understanding over full-length movie screenplays. STAGE defines four tasks: knowledge graph construction, scene-level event summarization, long-context screenplay question answering, and in-script character role-playing, all grounded in a shared narrative world representation. The benchmark provides cleaned scripts, curated knowledge graphs, and event- and character-centric annotations for 150 films across English and Chinese, enabling holistic evaluation of models' abilities to build world representations, abstract and verify narrative events, reason over long narratives, and generate character-consistent responses.
>
---
#### [replaced 079] Prompt2Fingerprint: Plug-and-Play LLM Fingerprinting via Text-to-Weight Generation
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Prompt2Fingerprint，解决LLM指纹生成的可扩展性问题，通过文本生成参数增量实现快速指纹注入。**

- **链接: [https://arxiv.org/pdf/2605.18474](https://arxiv.org/pdf/2605.18474)**

> **作者:** Sixu Chen; Xiang Chen; Hongyao Yu; Jiaxin Hong; Hao Fang; Shuoyang Sun; Bin Chen; Shu-Tao Xia
>
> **摘要:** The widespread deployment and redistribution of large language models (LLMs) have made model provenance tracking a critical challenge. While existing LLM fingerprinting methods, particularly active approaches that embed identity signals via fine-tuning, achieve high accuracy and robustness, they suffer from significant scalability bottlenecks. These methods typically treat fingerprint injection as an independent, one-off optimization task rather than a reusable capability, necessitating separate, resource-intensive training for every new identity. This incurs prohibitive computational costs and deployment delays. To address this, we propose Prompt2Fingerprint (P2F), the first framework that reformulates fingerprinting as a conditional parameter generation task. By leveraging a specialized generator, P2F maps textual descriptions directly to low-rank parameter increments in a single forward pass, enabling plug-and-play LLM fingerprint injection without further model retraining. Our experiments demonstrate that P2F maintains high fingerprint accuracy, harmlessness, and robustness while significantly reducing computational overhead, offering a scalable and instant solution for LLM ownership management.
>
---
