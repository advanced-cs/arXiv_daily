# 自然语言处理 cs.CL

- **最新发布 85 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出\texttt{Learn-to-Ask}框架，解决大语言模型在高风险领域中从被动应答转向主动、目标导向交互的难题。通过利用离线专家数据中的未来观测，构建密集奖励信号，实现无需用户模拟器的监督式策略学习，训练模型决策“问什么”与“何时停止”。在真实医疗数据上验证，成功部署于大规模在线服务，性能超越人类专家。**

- **链接: [http://arxiv.org/pdf/2510.25441v1](http://arxiv.org/pdf/2510.25441v1)**

> **作者:** Fei Wei; Daoyuan Chen; Ce Wang; Yilun Huang; Yushuo Chen; Xuchen Pan; Yaliang Li; Bolin Ding
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) excel as passive responders, but teaching them to be proactive, goal-oriented partners, a critical capability in high-stakes domains, remains a major challenge. Current paradigms either myopically optimize single-turn attributes or rely on brittle, high-cost user simulators, creating a persistent ``reality gap''. To bridge this gap, we introduce \texttt{Learn-to-Ask}, a general, simulator-free framework for learning and deploying proactive dialogue agents \textit{directly from offline expert data}, bypassing the need to model complex user dynamics. Our key insight is to reframe the offline policy learning problem by leveraging the \textbf{observed future} of each expert trajectory. This allows us to infer a dense, turn-by-turn reward signal grounded in the expert's revealed strategy, decomposing the intractable long-horizon problem into a series of supervised learning tasks, and training a policy to output a structured \texttt{(action, state_assessment)} tuple, governing both \textbf{what to ask} and, crucially, \textbf{when to stop}. To ensure reward fidelity, our Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision. Empirically, we demonstrate the efficacy of \texttt{Learn-to-Ask} in a real-world medical dataset, using LLMs of varying sizes up to 32B. Our approach culminates in the successful deployment of LLMs into a live, large-scale online AI service. In rigorous in-house evaluations, our model was launched and achieved performance even superior to human experts, proving our framework's ability to translate offline data into tangible, real-world impact. We hope this work provides a practical and economically viable blueprint for transforming passive LLMs into proactive, goal-oriented LLM applications.
>
---
#### [new 002] SwiftEmbed: Ultra-Fast Text Embeddings via Static Token Lookup for Real-Time Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SwiftEmbed，一种基于静态词元查找的文本嵌入方法，旨在实现亚毫秒级延迟（1.12毫秒p50）与高精度（60.6 MTEB平均分）。通过Rust实现的静态查表、优化均值池化和零拷贝序列化，支持每秒5万请求，适用于实时应用中的语义相似性与去重任务。**

- **链接: [http://arxiv.org/pdf/2510.24793v1](http://arxiv.org/pdf/2510.24793v1)**

> **作者:** Edouard Lansiaux
>
> **摘要:** We present a static token lookup methodology for text embedding generation that achieves 1.12 ms p50 latency for single text embeddings while maintaining 60.6 MTEB average score across 8 representative tasks, corresponding to 89% of contextual model quality. The Rust implementation delivers 50,000 requests per second throughput through static embedding lookup, optimized mean pooling, and zero-copy IEEE754 binary serialization. Evaluation demonstrates exceptional duplicate detection performance (90.1% AP), strong semantic similarity (76.1% Spearman correlation), and domain-specific performance ranging from 75% to 131% of baseline across specialized domains. The system enables real-time embedding applications where sub-5ms latency is critical.
>
---
#### [new 003] Towards a Method for Synthetic Generation of PWA Transcripts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对 aphasia 语言研究中数据稀缺问题，提出两种生成合成言语转录的方法：程序化方法与大语言模型（Mistral 7b、Llama 3.1）。通过词删减、填充词插入和错语替换模拟不同严重程度的失语症特征。结果表明，Mistral 7b 在捕捉语言退化趋势方面表现最优，为自动化分析失语症提供可行路径。**

- **链接: [http://arxiv.org/pdf/2510.24817v1](http://arxiv.org/pdf/2510.24817v1)**

> **作者:** Jason M. Pittman; Anton Phillips Jr.; Yesenia Medina-Santos; Brielle C. Stark
>
> **备注:** 19 pages, 1 figure, 7 tables
>
> **摘要:** In aphasia research, Speech-Language Pathologists (SLPs) devote extensive time to manually coding speech samples using Correct Information Units (CIUs), a measure of how informative an individual sample of speech is. Developing automated systems to recognize aphasic language is limited by data scarcity. For example, only about 600 transcripts are available in AphasiaBank yet billions of tokens are used to train large language models (LLMs). In the broader field of machine learning (ML), researchers increasingly turn to synthetic data when such are sparse. Therefore, this study constructs and validates two methods to generate synthetic transcripts of the AphasiaBank Cat Rescue picture description task. One method leverages a procedural programming approach while the second uses Mistral 7b Instruct and Llama 3.1 8b Instruct LLMs. The methods generate transcripts across four severity levels (Mild, Moderate, Severe, Very Severe) through word dropping, filler insertion, and paraphasia substitution. Overall, we found, compared to human-elicited transcripts, Mistral 7b Instruct best captures key aspects of linguistic degradation observed in aphasia, showing realistic directional changes in NDW, word count, and word length amongst the synthetic generation methods. Based on the results, future work should plan to create a larger dataset, fine-tune models for better aphasic representation, and have SLPs assess the realism and usefulness of the synthetic transcripts.
>
---
#### [new 004] Teaching Sarcasm: Few-Shot Multimodal Sarcasm Detection via Distillation to a Parameter-Efficient Student
- **分类: cs.CL**

- **简介: 该论文针对少样本多模态讽刺检测任务，解决标注数据稀缺导致模型性能受限的问题。提出PEKD框架，通过知识蒸馏将大模型教师的知识迁移至参数高效的学生模型，并引入置信度感知门控机制提升蒸馏可靠性，显著提升小样本下的检测效果。**

- **链接: [http://arxiv.org/pdf/2510.25303v1](http://arxiv.org/pdf/2510.25303v1)**

> **作者:** Soumyadeep Jana; Sanasam Ranbir Singh
>
> **摘要:** Multimodal sarcasm detection is challenging, especially in low-resource settings where subtle image-text contradictions are hard to learn due to scarce annotated data, which hinders the model's performance. Parameter-efficient fine-tuning (PEFT) methods like adapters, LoRA, and prompt tuning reduce overfitting but struggle to reach optimal performance due to limited supervision from few-shot data. We propose PEKD, a unified framework that enhances PEFT methods via distillation from an expert model trained on large-scale sarcasm data, which acts as the teacher. To mitigate unreliable signals from the teacher, we introduce an entropy-aware gating mechanism that dynamically adjusts the distillation strength based on teacher confidence. Experiments on two public datasets demonstrate that our PEKD framework enables PEFT methods to outperform both prior parameter-efficient approaches and large multimodal models, achieving strong results in the few-shot scenario. The framework is modular and adaptable to a wide range of multimodal models and tasks.
>
---
#### [new 005] Hallucinations in Bibliographic Recommendation: Citation Frequency as a Proxy for Training Data Redundancy
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在文献推荐中生成虚假参考文献的问题。针对“幻觉”现象，提出引用频次可作为训练数据冗余的代理指标，通过GPT-4.1验证发现：高被引论文更少幻觉，约1000次引用后信息近乎全量记忆，揭示了从泛化到记忆的临界点。**

- **链接: [http://arxiv.org/pdf/2510.25378v1](http://arxiv.org/pdf/2510.25378v1)**

> **作者:** Junichiro Niimi
>
> **摘要:** Large language models (LLMs) have been increasingly applied to a wide range of tasks, from natural language understanding to code generation. While they have also been used to assist in bibliographic recommendation, the hallucination of non-existent papers remains a major issue. Building on prior studies, this study hypothesizes that an LLM's ability to correctly produce bibliographic information depends on whether the underlying knowledge is generated or memorized, with highly cited papers (i.e., more frequently appear in the training corpus) showing lower hallucination rates. We therefore assume citation count as a proxy for training data redundancy (i.e., the frequency with which a given bibliographic record is repeatedly represented in the pretraining corpus) and investigate how citation frequency affects hallucinated references in LLM outputs. Using GPT-4.1, we generated and manually verified 100 bibliographic records across twenty computer-science domains, and measured factual consistency via cosine similarity between generated and authentic metadata. The results revealed that (i) hallucination rates vary across research domains, (ii) citation count is strongly correlated with factual accuracy, and (iii) bibliographic information becomes almost verbatimly memorized beyond approximately 1,000 citations. These findings suggest that highly cited papers are nearly verbatimly retained in the model, indicating a threshold where generalization shifts into memorization.
>
---
#### [new 006] Are Language Models Efficient Reasoners? A Perspective from Logic Programming
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文研究语言模型在逻辑推理中的效率问题，属于推理评估任务。针对现有评估忽视推理效率的问题，提出基于逻辑编程的框架，通过对比自然语言证明与最短逻辑证明，量化模型回避无关信息的能力。构建含干扰前提的数学应用题数据集，发现当前模型在干扰下准确率下降且常进行无效推理。**

- **链接: [http://arxiv.org/pdf/2510.25626v1](http://arxiv.org/pdf/2510.25626v1)**

> **作者:** Andreas Opedal; Yanick Zengaffinen; Haruki Shirakami; Clemente Pasti; Mrinmaya Sachan; Abulhair Saparov; Ryan Cotterell; Bernhard Schölkopf
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Modern language models (LMs) exhibit strong deductive reasoning capabilities, yet standard evaluations emphasize correctness while overlooking a key aspect of human-like reasoning: efficiency. In real-world reasoning scenarios, much of the available information is irrelevant, and effective deductive inference requires identifying and ignoring such distractions. We propose a framework for assessing LM reasoning efficiency through the lens of logic programming, introducing a simple method to align proofs written in natural language -- as generated by an LM -- with shortest proofs found by executing the logic program. Efficiency is quantified by measuring how well a model avoids unnecessary inference. Empirically, we construct a dataset of math word problems injected with various number of irrelevant axioms that vary in semantic overlap with the goal theorem. We find that current LMs show marked accuracy declines under such conditions -- even with minimal, domain-consistent distractions -- and the proofs they generate frequently exhibit detours through irrelevant inferences.
>
---
#### [new 007] Monitoring Transformative Technological Convergence Through LLM-Extracted Semantic Entity Triple Graphs
- **分类: cs.CL**

- **简介: 该论文聚焦于技术趋势预测任务，旨在解决传统方法难以应对快速迭代与术语模糊的问题。通过LLM提取文本中的语义三元组构建技术实体图谱，结合词义聚类与图结构分析，实现对新兴技术融合模式的自动识别与监测。**

- **链接: [http://arxiv.org/pdf/2510.25370v1](http://arxiv.org/pdf/2510.25370v1)**

> **作者:** Alexander Sternfeld; Andrei Kucharavy; Dimitri Percia David; Alain Mermoud; Julian Jang-Jaccard; Nathan Monnet
>
> **摘要:** Forecasting transformative technologies remains a critical but challenging task, particularly in fast-evolving domains such as Information and Communication Technologies (ICTs). Traditional expert-based methods struggle to keep pace with short innovation cycles and ambiguous early-stage terminology. In this work, we propose a novel, data-driven pipeline to monitor the emergence of transformative technologies by identifying patterns of technological convergence. Our approach leverages advances in Large Language Models (LLMs) to extract semantic triples from unstructured text and construct a large-scale graph of technology-related entities and relations. We introduce a new method for grouping semantically similar technology terms (noun stapling) and develop graph-based metrics to detect convergence signals. The pipeline includes multi-stage filtering, domain-specific keyword clustering, and a temporal trend analysis of topic co-occurence. We validate our methodology on two complementary datasets: 278,625 arXiv preprints (2017--2024) to capture early scientific signals, and 9,793 USPTO patent applications (2018-2024) to track downstream commercial developments. Our results demonstrate that the proposed pipeline can identify both established and emerging convergence patterns, offering a scalable and generalizable framework for technology forecasting grounded in full-text analysis.
>
---
#### [new 008] TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation
- **分类: cs.CL; I.2.7; I.2.6; I.2.0**

- **简介: 该论文提出TwinVoice基准，用于评估大模型在数字孪生中的人物角色模拟能力。针对现有评估缺乏真实场景、系统框架和能力分析的问题，构建了社会、人际、叙事三维度评测体系，并分解为六项核心能力。实验表明，当前模型在语义风格与记忆召回上仍远低于人类水平。**

- **链接: [http://arxiv.org/pdf/2510.25536v1](http://arxiv.org/pdf/2510.25536v1)**

> **作者:** Bangde Du; Minghao Guo; Songming He; Ziyi Ye; Xi Zhu; Weihang Su; Shuqi Zhu; Yujia Zhou; Yongfeng Zhang; Qingyao Ai; Yiqun Liu
>
> **备注:** Main paper: 11 pages, 3 figures, 6 tables. Appendix: 28 pages. Bangde Du and Minghao Guo contributed equally. Corresponding authors: Ziyi Ye (ziyiye@fudan.edu.cn), Qingyao Ai (aiqy@tsinghua.edu.cn)
>
> **摘要:** Large Language Models (LLMs) are exhibiting emergent human-like abilities and are increasingly envisioned as the foundation for simulating an individual's communication style, behavioral tendencies, and personality traits. However, current evaluations of LLM-based persona simulation remain limited: most rely on synthetic dialogues, lack systematic frameworks, and lack analysis of the capability requirement. To address these limitations, we introduce TwinVoice, a comprehensive benchmark for assessing persona simulation across diverse real-world contexts. TwinVoice encompasses three dimensions: Social Persona (public social interactions), Interpersonal Persona (private dialogues), and Narrative Persona (role-based expression). It further decomposes the evaluation of LLM performance into six fundamental capabilities, including opinion consistency, memory recall, logical reasoning, lexical fidelity, persona tone, and syntactic style. Experimental results reveal that while advanced models achieve moderate accuracy in persona simulation, they still fall short of capabilities such as syntactic style and memory recall. Consequently, the average performance achieved by LLMs remains considerably below the human baseline.
>
---
#### [new 009] Falcon: A Comprehensive Chinese Text-to-SQL Benchmark for Enterprise-Grade Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Falcon，一个面向企业级场景的中文文本转SQL基准数据集。针对中文语义复杂、企业数据库规模大、表间关系模糊等问题，构建了600个跨领域中文查询，涵盖多表关联与复杂逻辑，配套执行验证工具与自动化评估流程，推动中文Text-to-SQL技术在真实企业环境下的评测与优化。**

- **链接: [http://arxiv.org/pdf/2510.24762v1](http://arxiv.org/pdf/2510.24762v1)**

> **作者:** Wenzhen Luo; Wei Guan; Yifan Yao; Yimin Pan; Feng Wang; Zhipeng Yu; Zhe Wen; Liang Chen; Yihong Zhuang
>
> **摘要:** We introduce Falcon, a cross-domain Chinese text-to-SQL benchmark grounded in an enterprise-compatible dialect (MaxCompute/Hive). It contains 600 Chinese questions over 28 databases; 77% require multi-table reasoning and over half touch more than four tables. Each example is annotated along SQL-computation features and Chinese semantics. For evaluation, we release a robust execution comparator and an automated evaluation pipeline, under which all current state-of-the-art large-scale models (including Deepseek) achieve accuracies of at most 50%. Major errors originate from two sources: (1) schema linking in large enterprise landscapes - hundreds of tables, denormalized fields, ambiguous column names, implicit foreign-key relations and domain-specific synonyms that make correct join/column selection difficult; and (2) mapping concise, colloquial Chinese into the exact operators and predicates required for analytics - e.g., choosing the correct aggregation and group-by keys, expressing time windows and granularities, applying unit conversions, handling NULLs and data-quality rules, and formulating nested or windowed subqueries. Falcon therefore targets Chinese-specific semantics and enterprise dialects (abbreviations, business jargon, fuzzy entity references) and provides a reproducible middle ground before full production deployment by using realistic enterprise schemas, query templates, an execution comparator, and an automated evaluation pipeline for end-to-end validation.
>
---
#### [new 010] Parallel Loop Transformer for Efficient Test-Time Computation Scaling
- **分类: cs.CL**

- **简介: 该论文针对大语言模型推理慢、成本高的问题，提出并行循环变压器（PLT）。通过跨循环并行与高效表示增强，实现多循环计算的并行化，显著降低延迟和内存开销，同时保持深度模型的高精度，适用于快速推理场景。**

- **链接: [http://arxiv.org/pdf/2510.24824v1](http://arxiv.org/pdf/2510.24824v1)**

> **作者:** Bohong Wu; Mengzhao Chen; Xiang Luo; Shen Yan; Qifan Yu; Fan Xia; Tianqi Zhang; Hongrui Zhan; Zheng Zhong; Xun Zhou; Siyuan Qiao; Xingyan Bin
>
> **摘要:** Large Language Models (LLMs) are powerful but often too slow and costly for real-world use during inference. Looped transformers save on parameters by reusing the same weights for multiple computational steps, or "loops." However, this approach has a major flaw: the loops run one after another, causing inference latency and memory requirements to increase with each added loop. This makes them impractical for fast applications. To solve this problem, we introduce the Parallel Loop Transformer (PLT). PLT is a new architecture that delivers the performance benefits of a deep, looped model but with the low latency of a standard, non-looped model. PLT works using two key techniques. First, Cross-Loop Parallelism (CLP) breaks the sequential dependency by computing different loops for different tokens at the same time, all within a single pass. Second, to prevent memory costs from growing, we use an Efficient Representation Enhancement strategy. This method shares the memory (KV cache) from the first loop with all other loops. It then uses a Gated Sliding-Window Attention (G-SWA) to combine this shared global information with local information, maintaining high accuracy. Our experiments show that PLT achieves the high accuracy of a traditional looped model but with almost no extra latency or memory cost compared to a standard transformer.
>
---
#### [new 011] RLMEval: Evaluating Research-Level Neural Theorem Proving
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对研究级神经定理证明与证明自动形式化任务，提出RLMEval评估套件，基于真实Lean项目中的613个难题评估模型性能。结果显示现有模型在真实场景下表现有限，仅10.3%通过率，揭示了当前基准与实际应用间的显著差距，旨在推动自动化推理在形式数学中的发展。**

- **链接: [http://arxiv.org/pdf/2510.25427v1](http://arxiv.org/pdf/2510.25427v1)**

> **作者:** Auguste Poiroux; Antoine Bosselut; Viktor Kunčak
>
> **备注:** Accepted to EMNLP 2025 Findings. RLMEval benchmark released: https://github.com/augustepoiroux/RLMEval
>
> **摘要:** Despite impressive results on curated benchmarks, the practical impact of large language models (LLMs) on research-level neural theorem proving and proof autoformalization is still limited. We introduce RLMEval, an evaluation suite for these tasks, focusing on research-level mathematics from real-world Lean formalization projects. RLMEval targets the evaluation of neural theorem proving and proof autoformalization on challenging research-level theorems by leveraging real Lean Blueprint formalization projects. Our evaluation of state-of-the-art models on RLMEval, comprising 613 theorems from 6 Lean projects, reveals a significant gap: progress on existing benchmarks does not readily translate to these more realistic settings, with the best model achieving only a 10.3 % pass rate. RLMEval provides a new, challenging benchmark designed to guide and accelerate progress in automated reasoning for formal mathematics.
>
---
#### [new 012] EHR-R1: A Reasoning-Enhanced Foundational Language Model for Electronic Health Record Analysis
- **分类: cs.CL**

- **简介: 该论文针对电子健康记录（EHR）分析中大模型推理能力不足的问题，提出EHR-Ins数据集与EHR-R1推理增强模型。通过思维图驱动的数据生成与多阶段训练，提升模型在42项临床任务上的推理与预测性能，显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2510.25628v1](http://arxiv.org/pdf/2510.25628v1)**

> **作者:** Yusheng Liao; Chaoyi Wu; Junwei Liu; Shuyang Jiang; Pengcheng Qiu; Haowen Wang; Yun Yue; Shuai Zhen; Jian Wang; Qianrui Fan; Jinjie Gu; Ya Zhang; Yanfeng Wang; Yu Wang; Weidi Xie
>
> **摘要:** Electronic Health Records (EHRs) contain rich yet complex information, and their automated analysis is critical for clinical decision-making. Despite recent advances of large language models (LLMs) in clinical workflows, their ability to analyze EHRs remains limited due to narrow task coverage and lack of EHR-oriented reasoning capabilities. This paper aims to bridge the gap, specifically, we present EHR-Ins, a large-scale, comprehensive EHR reasoning instruction dataset, comprising 300k high-quality reasoning cases and 4M non-reasoning cases across 42 distinct EHR tasks. Its core innovation is a thinking-graph-driven framework that enables to generate high-quality reasoning data at scale. Based on it, we develop EHR-R1, a series of reasoning-enhanced LLMs with up to 72B parameters tailored for EHR analysis. Through a multi-stage training paradigm, including domain adaptation, reasoning enhancement, and reinforcement learning, EHR-R1 systematically acquires domain knowledge and diverse reasoning capabilities, enabling accurate and robust EHR analysis. Lastly, we introduce EHR-Bench, a new benchmark curated from MIMIC-IV, spanning 42 tasks, to comprehensively assess reasoning and prediction across EHR scenarios. In experiments, we show that the resulting EHR-R1 consistently outperforms state-of-the-art commercial and open-source LLMs (including DeepSeek-V3 and GPT-4o), surpassing GPT-4o by over 30 points on MIMIC-Bench and achieving a 10\% higher zero-shot AUROC on EHRSHOT. Collectively, EHR-Ins, EHR-R1, and EHR-Bench have significantly advanced the development for more reliable and clinically relevant EHR analysis.
>
---
#### [new 013] RiddleBench: A New Generative Reasoning Benchmark for LLMs
- **分类: cs.CL**

- **简介: 该论文提出RiddleBench，一个包含1737个英文谜题的生成式推理基准，旨在评估大语言模型在逻辑、空间与约束满足等灵活推理能力上的表现。针对现有基准多聚焦结构化任务的问题，该工作揭示了当前模型在推理脆弱性、幻觉传播与自我修正方面的深层缺陷，为模型优化提供诊断工具与方向。**

- **链接: [http://arxiv.org/pdf/2510.24932v1](http://arxiv.org/pdf/2510.24932v1)**

> **作者:** Deepon Halder; Alan Saji; Thanmay Jayakumar; Ratish Puduppully; Anoop Kunchukuttan; Raj Dabre
>
> **摘要:** Large Language Models have demonstrated strong performance on many established reasoning benchmarks. However, these benchmarks primarily evaluate structured skills like quantitative problem-solving, leaving a gap in assessing flexible, multifaceted reasoning abilities that are central to human intelligence. These abilities require integrating logical deduction with spatial awareness and constraint satisfaction, which current evaluations do not measure well. To address this, we introduce RiddleBench, a benchmark of 1,737 challenging puzzles in English designed to probe these core reasoning capabilities. Evaluation of state-of-the-art models on RiddleBench shows fundamental weaknesses. Even top proprietary models like Gemini 2.5 Pro, o3, and Claude 4 Sonnet achieve accuracy just above 60% (60.30%, 63.37%, and 63.16%). Analysis further reveals deep failures, including hallucination cascades (accepting flawed reasoning from other models) and poor self-correction due to a strong self-confirmation bias. Their reasoning is also fragile, with performance degrading significantly when constraints are reordered or irrelevant information is introduced. RiddleBench functions as a diagnostic tool for these issues and as a resource for guiding the development of more robust and reliable language models.
>
---
#### [new 014] TOPol: Capturing and Explaining Multidimensional Semantic Polarity Fields and Vectors
- **分类: cs.CL**

- **简介: 该论文提出TOPol框架，解决传统情感分析忽视语义极性的多维性问题。通过结合大语言模型与图聚类，构建上下文边界下的多维极性场，量化话语转变中的语义偏移，实现可解释的上下文敏感分析。**

- **链接: [http://arxiv.org/pdf/2510.25069v1](http://arxiv.org/pdf/2510.25069v1)**

> **作者:** Gabin Taibi; Lucia Gomez
>
> **备注:** 7 pages, 3 figures and 2 tables
>
> **摘要:** Traditional approaches to semantic polarity in computational linguistics treat sentiment as a unidimensional scale, overlooking the multidimensional structure of language. This work introduces TOPol (Topic-Orientation POLarity), a semi-unsupervised framework for reconstructing and interpreting multidimensional narrative polarity fields under human-on-the-loop (HoTL) defined contextual boundaries (CBs). The framework embeds documents using a transformer-based large language model (tLLM), applies neighbor-tuned UMAP projection, and segments topics via Leiden partitioning. Given a CB between discourse regimes A and B, TOPol computes directional vectors between corresponding topic-boundary centroids, yielding a polarity field that quantifies fine-grained semantic displacement during regime shifts. This vectorial representation enables assessing CB quality and detecting polarity changes, guiding HoTL CB refinement. To interpret identified polarity vectors, the tLLM compares their extreme points and produces contrastive labels with estimated coverage. Robustness analyses show that only CB definitions (the main HoTL-tunable parameter) significantly affect results, confirming methodological stability. We evaluate TOPol on two corpora: (i) U.S. Central Bank speeches around a macroeconomic breakpoint, capturing non-affective semantic shifts, and (ii) Amazon product reviews across rating strata, where affective polarity aligns with NRC valence. Results demonstrate that TOPol consistently captures both affective and non-affective polarity transitions, providing a scalable, generalizable, and interpretable framework for context-sensitive multidimensional discourse analysis.
>
---
#### [new 015] FARSIQA: Faithful and Advanced RAG System for Islamic Question Answering
- **分类: cs.CL; cs.AI; cs.IR; 68T50, 68T05, 68T30; I.2.7; H.3.3**

- **简介: 该论文针对波斯语伊斯兰问答任务，解决大模型在宗教领域易幻觉、不忠实的问题。提出FARSIQA系统，基于自适应迭代的FAIR-RAG架构，通过动态分解复杂问题、逐步补全证据，提升答案准确性和可靠性。在伊斯兰PCQA基准上实现97.0%负例拒绝率与74.3%答对率，显著优于基线。**

- **链接: [http://arxiv.org/pdf/2510.25621v1](http://arxiv.org/pdf/2510.25621v1)**

> **作者:** Mohammad Aghajani Asl; Behrooz Minaei Bidgoli
>
> **备注:** 37 pages, 5 figures, 10 tables. Keywords: Retrieval-Augmented Generation (RAG), Question Answering (QA), Islamic Knowledge Base, Faithful AI, Persian NLP, Multi-hop Reasoning, Large Language Models (LLMs)
>
> **摘要:** The advent of Large Language Models (LLMs) has revolutionized Natural Language Processing, yet their application in high-stakes, specialized domains like religious question answering is hindered by challenges like hallucination and unfaithfulness to authoritative sources. This issue is particularly critical for the Persian-speaking Muslim community, where accuracy and trustworthiness are paramount. Existing Retrieval-Augmented Generation (RAG) systems, relying on simplistic single-pass pipelines, fall short on complex, multi-hop queries requiring multi-step reasoning and evidence aggregation. To address this gap, we introduce FARSIQA, a novel, end-to-end system for Faithful Advanced Question Answering in the Persian Islamic domain. FARSIQA is built upon our innovative FAIR-RAG architecture: a Faithful, Adaptive, Iterative Refinement framework for RAG. FAIR-RAG employs a dynamic, self-correcting process: it adaptively decomposes complex queries, assesses evidence sufficiency, and enters an iterative loop to generate sub-queries, progressively filling information gaps. Operating on a curated knowledge base of over one million authoritative Islamic documents, FARSIQA demonstrates superior performance. Rigorous evaluation on the challenging IslamicPCQA benchmark shows state-of-the-art performance: the system achieves a remarkable 97.0% in Negative Rejection - a 40-point improvement over baselines - and a high Answer Correctness score of 74.3%. Our work establishes a new standard for Persian Islamic QA and validates that our iterative, adaptive architecture is crucial for building faithful, reliable AI systems in sensitive domains.
>
---
#### [new 016] Seeing, Signing, and Saying: A Vision-Language Model-Assisted Pipeline for Sign Language Data Acquisition and Curation from Social Media
- **分类: cs.CL**

- **简介: 该论文针对手语数据集规模小、多语言覆盖不足、标注成本高的问题，提出基于视觉语言模型的自动化数据采集与筛选框架。通过检测人脸、识别手语动作、提取文本并验证视频-文本对齐，实现从社交媒体高效获取高质量手语数据，支持弱监督预训练，推动手语翻译技术发展。**

- **链接: [http://arxiv.org/pdf/2510.25413v1](http://arxiv.org/pdf/2510.25413v1)**

> **作者:** Shakib Yazdani; Yasser Hamidullah; Cristina España-Bonet; Josef van Genabith
>
> **备注:** Accepted by RANLP 2025
>
> **摘要:** Most existing sign language translation (SLT) datasets are limited in scale, lack multilingual coverage, and are costly to curate due to their reliance on expert annotation and controlled recording setup. Recently, Vision Language Models (VLMs) have demonstrated strong capabilities as evaluators and real-time assistants. Despite these advancements, their potential remains untapped in the context of sign language dataset acquisition. To bridge this gap, we introduce the first automated annotation and filtering framework that utilizes VLMs to reduce reliance on manual effort while preserving data quality. Our method is applied to TikTok videos across eight sign languages and to the already curated YouTube-SL-25 dataset in German Sign Language for the purpose of additional evaluation. Our VLM-based pipeline includes a face visibility detection, a sign activity recognition, a text extraction from video content, and a judgment step to validate alignment between video and text, implementing generic filtering, annotation and validation steps. Using the resulting corpus, TikTok-SL-8, we assess the performance of two off-the-shelf SLT models on our filtered dataset for German and American Sign Languages, with the goal of establishing baselines and evaluating the robustness of recent models on automatically extracted, slightly noisy data. Our work enables scalable, weakly supervised pretraining for SLT and facilitates data acquisition from social media.
>
---
#### [new 017] Confidence is Not Competence
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中自信与能力脱节的问题。通过分析模型内部状态的几何结构，发现评估阶段（高维复杂）与执行阶段（低维简单）存在显著差异，揭示了“评估-执行”双系统架构。研究提出因果干预验证：信念可线性解码但难以控制最终结果，表明应聚焦执行过程而非评估表征。**

- **链接: [http://arxiv.org/pdf/2510.24772v1](http://arxiv.org/pdf/2510.24772v1)**

> **作者:** Debdeep Sanyal; Manya Pandey; Dhruv Kumar; Saurabh Deshpande; Murari Mandal
>
> **备注:** 20 Pages, 6 Figures, 8 Tables
>
> **摘要:** Large language models (LLMs) often exhibit a puzzling disconnect between their asserted confidence and actual problem-solving competence. We offer a mechanistic account of this decoupling by analyzing the geometry of internal states across two phases - pre-generative assessment and solution execution. A simple linear probe decodes the internal "solvability belief" of a model, revealing a well-ordered belief axis that generalizes across model families and across math, code, planning, and logic tasks. Yet, the geometries diverge - although belief is linearly decodable, the assessment manifold has high linear effective dimensionality as measured from the principal components, while the subsequent reasoning trace evolves on a much lower-dimensional manifold. This sharp reduction in geometric complexity from thought to action mechanistically explains the confidence-competence gap. Causal interventions that steer representations along the belief axis leave final solutions unchanged, indicating that linear nudges in the complex assessment space do not control the constrained dynamics of execution. We thus uncover a two-system architecture - a geometrically complex assessor feeding a geometrically simple executor. These results challenge the assumption that decodable beliefs are actionable levers, instead arguing for interventions that target the procedural dynamics of execution rather than the high-level geometry of assessment.
>
---
#### [new 018] Parrot: A Training Pipeline Enhances Both Program CoT and Natural Language CoT for Reasoning
- **分类: cs.CL**

- **简介: 该论文针对数学推理任务，提出Parrot训练框架，旨在协同提升自然语言与程序化思维链（N-CoT、P-CoT）性能。通过设计三阶段子任务、混合训练策略及辅助奖励机制，实现两者互促增强，显著提升N-CoT表现，尤其在LLaMA2和CodeLLaMA上取得显著效果。**

- **链接: [http://arxiv.org/pdf/2510.25310v1](http://arxiv.org/pdf/2510.25310v1)**

> **作者:** Senjie Jin; Lu Chen; Zhiheng Xi; Yuhui Wang; Sirui Song; Yuhao Zhou; Xinbo Zhang; Peng Sun; Hong Lu; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Natural language chain-of-thought (N-CoT) and Program chain-of-thought (P-CoT) have emerged as two primary paradigms for large language models (LLMs) to solve mathematical reasoning problems. Current research typically endeavors to achieve unidirectional enhancement: P-CoT enhanced N-CoT or N-CoT enhanced P-CoT. In this paper, we seek to fully unleash the two paradigms' strengths for mutual enhancement and ultimately achieve simultaneous improvements. We conduct a detailed analysis of the error types across two paradigms, based on which we propose Parrot, a novel training pipeline for mathematical problems: 1) Three target-designed subtasks integrate sequential P-CoT and N-CoT generation. 2) A subtask hybrid training strategy to facilitate natural language semantic transferability. 3) The converted N-CoT auxiliary reward is designed to alleviate the sparse rewards in P-CoT optimization. Extensive experiments demonstrate that Parrot significantly enhances both the performance of N-CoT and P-CoT, especially on N-CoT. Using Parrot SFT, the N-CoT performance of LLaMA2 and CodeLLaMA achieve gains of +21.87 and +21.48 on MathQA over the RL baseline, which is resource-intensive.
>
---
#### [new 019] Do Large Language Models Grasp The Grammar? Evidence from Grammar-Book-Guided Probing in Luxembourgish
- **分类: cs.CL**

- **简介: 该论文聚焦低资源语言卢森堡语的语法理解能力评估，提出基于语法书的系统性评测框架。针对大语言模型在语法结构与语义映射上理解不足的问题，通过四阶段流程验证其语法掌握程度，发现模型翻译能力强不等于语法理解深，且在形态与句法上表现较弱，推理能力或可提升语法理解。**

- **链接: [http://arxiv.org/pdf/2510.24856v1](http://arxiv.org/pdf/2510.24856v1)**

> **作者:** Lujun Li; Yewei Song; Lama Sleem; Yiqun Wang; Yangjie Xu; Cedric Lothritz; Niccolo Gentile; Radu State; Tegawende F. Bissyande; Jacques Klein
>
> **摘要:** Grammar refers to the system of rules that governs the structural organization and the semantic relations among linguistic units such as sentences, phrases, and words within a given language. In natural language processing, there remains a notable scarcity of grammar focused evaluation protocols, a gap that is even more pronounced for low-resource languages. Moreover, the extent to which large language models genuinely comprehend grammatical structure, especially the mapping between syntactic structures and meanings, remains under debate. To investigate this issue, we propose a Grammar Book Guided evaluation pipeline intended to provide a systematic and generalizable framework for grammar evaluation consisting of four key stages, and in this work we take Luxembourgish as a case study. The results show a weak positive correlation between translation performance and grammatical understanding, indicating that strong translations do not necessarily imply deep grammatical competence. Larger models perform well overall due to their semantic strength but remain weak in morphology and syntax, struggling particularly with Minimal Pair tasks, while strong reasoning ability offers a promising way to enhance their grammatical understanding.
>
---
#### [new 020] Scaling Latent Reasoning via Looped Language Models
- **分类: cs.CL**

- **简介: 该论文提出Looped Language Models（LoopLM），通过在预训练阶段引入隐空间迭代计算与熵正则化深度分配，将推理能力融入模型构建。解决了传统LLM依赖后训练推理（如CoT）导致预训练数据利用率低的问题。实验表明，小规模LoopLM模型性能媲美更大规模SOTA模型，且推理轨迹更契合输出。**

- **链接: [http://arxiv.org/pdf/2510.25741v1](http://arxiv.org/pdf/2510.25741v1)**

> **作者:** Rui-Jie Zhu; Zixuan Wang; Kai Hua; Tianyu Zhang; Ziniu Li; Haoran Que; Boyi Wei; Zixin Wen; Fan Yin; He Xing; Lu Li; Jiajun Shi; Kaijing Ma; Shanda Li; Taylor Kergan; Andrew Smith; Xingwei Qu; Mude Hui; Bohong Wu; Qiyang Min; Hongzhi Huang; Xun Zhou; Wei Ye; Jiaheng Liu; Jian Yang; Yunfeng Shi; Chenghua Lin; Enduo Zhao; Tianle Cai; Ge Zhang; Wenhao Huang; Yoshua Bengio; Jason Eshraghian
>
> **摘要:** Modern LLMs are trained to "think" primarily via explicit text generation, such as chain-of-thought (CoT), which defers reasoning to post-training and under-leverages pre-training data. We present and open-source Ouro, named after the recursive Ouroboros, a family of pre-trained Looped Language Models (LoopLM) that instead build reasoning into the pre-training phase through (i) iterative computation in latent space, (ii) an entropy-regularized objective for learned depth allocation, and (iii) scaling to 7.7T tokens. Ouro 1.4B and 2.6B models enjoy superior performance that match the results of up to 12B SOTA LLMs across a wide range of benchmarks. Through controlled experiments, we show this advantage stems not from increased knowledge capacity, but from superior knowledge manipulation capabilities. We also show that LoopLM yields reasoning traces more aligned with final outputs than explicit CoT. We hope our results show the potential of LoopLM as a novel scaling direction in the reasoning era. Our model could be found in: http://ouro-llm.github.io.
>
---
#### [new 021] CRMWeaver: Building Powerful Business Agent via Agentic RL and Shared Memories
- **分类: cs.CL**

- **简介: 该论文提出CRMWeaver，一种基于智能体强化学习与共享记忆的业务代理框架。针对复杂业务场景中数据关系繁杂、任务多样问题，通过合成数据训练与共享记忆机制，提升模型在多类型任务中的泛化能力与实际应用效果。**

- **链接: [http://arxiv.org/pdf/2510.25333v1](http://arxiv.org/pdf/2510.25333v1)**

> **作者:** Yilong Lai; Yipin Yang; Jialong Wu; Fengran Mo; Zhenglin Wang; Ting Liang; Jianguo Lin; Keping Yang
>
> **摘要:** Recent years have witnessed the rapid development of LLM-based agents, which shed light on using language agents to solve complex real-world problems. A prominent application lies in business agents, which interact with databases and internal knowledge bases via tool calls to fulfill diverse user requirements. However, this domain is characterized by intricate data relationships and a wide range of heterogeneous tasks, from statistical data queries to knowledge-based question-answering. To address these challenges, we propose CRMWeaver, a novel approach that enhances business agents in such complex settings. To acclimate the agentic model to intricate business environments, we employ a synthesis data generation and RL-based paradigm during training, which significantly improves the model's ability to handle complex data and varied tasks. During inference, a shared memories mechanism is introduced, prompting the agent to learn from task guidelines in similar problems, thereby further boosting its effectiveness and generalization, especially in unseen scenarios. We validate the efficacy of our approach on the CRMArena-Pro dataset, where our lightweight model achieves competitive results in both B2B and B2C business scenarios, underscoring its practical value for real-world applications.
>
---
#### [new 022] A Critical Study of Automatic Evaluation in Sign Language Translation
- **分类: cs.CL**

- **简介: 该论文研究手语翻译（SLT）的自动评估问题，针对现有文本类指标（如BLEU、ROUGE）无法全面衡量SLT质量的缺陷，对比分析了传统与大语言模型（LLM）基评估方法在同义改写、幻觉和句长变化下的表现，揭示其局限性，提出需构建融合多模态信息的评估框架。**

- **链接: [http://arxiv.org/pdf/2510.25434v1](http://arxiv.org/pdf/2510.25434v1)**

> **作者:** Shakib Yazdani; Yasser Hamidullah; Cristina España-Bonet; Eleftherios Avramidis; Josef van Genabith
>
> **备注:** Submitted to the LREC 2026 conference
>
> **摘要:** Automatic evaluation metrics are crucial for advancing sign language translation (SLT). Current SLT evaluation metrics, such as BLEU and ROUGE, are only text-based, and it remains unclear to what extent text-based metrics can reliably capture the quality of SLT outputs. To address this gap, we investigate the limitations of text-based SLT evaluation metrics by analyzing six metrics, including BLEU, chrF, and ROUGE, as well as BLEURT on the one hand, and large language model (LLM)-based evaluators such as G-Eval and GEMBA zero-shot direct assessment on the other hand. Specifically, we assess the consistency and robustness of these metrics under three controlled conditions: paraphrasing, hallucinations in model outputs, and variations in sentence length. Our analysis highlights the limitations of lexical overlap metrics and demonstrates that while LLM-based evaluators better capture semantic equivalence often missed by conventional metrics, they can also exhibit bias toward LLM-paraphrased translations. Moreover, although all metrics are able to detect hallucinations, BLEU tends to be overly sensitive, whereas BLEURT and LLM-based evaluators are comparatively lenient toward subtle cases. This motivates the need for multimodal evaluation frameworks that extend beyond text-based metrics to enable a more holistic assessment of SLT outputs.
>
---
#### [new 023] Fine-Tuned Language Models for Domain-Specific Summarization and Tagging
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对亚文化语言与俚语导致的信息提取难题，提出融合微调大模型与命名实体识别的摘要与标注流水线。通过在政治与安全领域数据上微调LLaMA3-8B-Instruct，显著提升摘要与实体识别准确率，实现跨语言知识迁移，支持实时、高效的结构化信息处理。**

- **链接: [http://arxiv.org/pdf/2510.25460v1](http://arxiv.org/pdf/2510.25460v1)**

> **作者:** Jun Wang; Fuming Lin; Yuyu Chen
>
> **摘要:** This paper presents a pipeline integrating fine-tuned large language models (LLMs) with named entity recognition (NER) for efficient domain-specific text summarization and tagging. The authors address the challenge posed by rapidly evolving sub-cultural languages and slang, which complicate automated information extraction and law enforcement monitoring. By leveraging the LLaMA Factory framework, the study fine-tunes LLMs on both generalpurpose and custom domain-specific datasets, particularly in the political and security domains. The models are evaluated using BLEU and ROUGE metrics, demonstrating that instruction fine-tuning significantly enhances summarization and tagging accuracy, especially for specialized corpora. Notably, the LLaMA3-8B-Instruct model, despite its initial limitations in Chinese comprehension, outperforms its Chinese-trained counterpart after domainspecific fine-tuning, suggesting that underlying reasoning capabilities can transfer across languages. The pipeline enables concise summaries and structured entity tagging, facilitating rapid document categorization and distribution. This approach proves scalable and adaptable for real-time applications, supporting efficient information management and the ongoing need to capture emerging language trends. The integration of LLMs and NER offers a robust solution for transforming unstructured text into actionable insights, crucial for modern knowledge management and security operations.
>
---
#### [new 024] Roleplaying with Structure: Synthetic Therapist-Client Conversation Generation from Questionnaires
- **分类: cs.CL**

- **简介: 该论文属于心理治疗对话生成任务，旨在解决临床对话数据稀缺问题。通过基于认知行为疗法的框架SQPsych，利用结构化问卷生成合成治疗对话，采用开源大模型实现隐私安全的数据生成与验证，有效提升对话质量与临床相关性。**

- **链接: [http://arxiv.org/pdf/2510.25384v1](http://arxiv.org/pdf/2510.25384v1)**

> **作者:** Doan Nam Long Vu; Rui Tan; Lena Moench; Svenja Jule Francke; Daniel Woiwod; Florian Thomas-Odenthal; Sanna Stroth; Tilo Kircher; Christiane Hermann; Udo Dannlowski; Hamidreza Jamalabadi; Shaoxiong Ji
>
> **摘要:** The development of AI for mental health is hindered by a lack of authentic therapy dialogues, due to strict privacy regulations and the fact that clinical sessions were historically rarely recorded. We present an LLM-driven pipeline that generates synthetic counseling dialogues based on structured client profiles and psychological questionnaires. Grounded on the principles of Cognitive Behavioral Therapy (CBT), our method creates synthetic therapeutic conversations for clinical disorders such as anxiety and depression. Our framework, SQPsych (Structured Questionnaire-based Psychotherapy), converts structured psychological input into natural language dialogues through therapist-client simulations. Due to data governance policies and privacy restrictions prohibiting the transmission of clinical questionnaire data to third-party services, previous methodologies relying on proprietary models are infeasible in our setting. We address this limitation by generating a high-quality corpus using open-weight LLMs, validated through human expert evaluation and LLM-based assessments. Our SQPsychLLM models fine-tuned on SQPsychConv achieve strong performance on counseling benchmarks, surpassing baselines in key therapeutic skills. Our findings highlight the potential of synthetic data to enable scalable, data-secure, and clinically informed AI for mental health support. We will release our code, models, and corpus at https://ai-mh.github.io/SQPsych
>
---
#### [new 025] Adapting Small Language Models to Low-Resource Domains: A Case Study in Hindi Tourism QA
- **分类: cs.CL**

- **简介: 该论文针对低资源语言（印地语）旅游问答任务，解决标注数据少与通用模型领域知识不足的问题。通过大模型生成合成数据，并采用多阶段微调策略，使小型语言模型有效适应特定领域，提升了问答性能，提供了一种高效可扩展的解决方案。**

- **链接: [http://arxiv.org/pdf/2510.25273v1](http://arxiv.org/pdf/2510.25273v1)**

> **作者:** Sandipan Majhi; Paheli Bhattacharya
>
> **备注:** Accepted at the Forum for Information Retrieval Evaluation 2025 (VATIKA Track)
>
> **摘要:** Domain-specific question answering in low-resource languages faces two key challenges: scarcity of annotated datasets and limited domain knowledge in general-purpose language models. In this work, we present a multi-stage finetuning strategy to adapt lightweight language models to the Hindi tourism domain by leveraging both original and synthetic training data. Synthetic question-answer pairs are generated using large LLMs (LLaMA-70B, Phi-14B) and used to augment the limited original dataset. We explore several training methodologies and analyse their impact on domain generalisation. Our results demonstrate that large models can efficiently generate synthetic data, while small models can effectively adapt to it, offering a scalable pathway for low-resource, domain-specific QA.
>
---
#### [new 026] Iti-Validator: A Guardrail Framework for Validating and Correcting LLM-Generated Itineraries
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大模型生成行程时存在的时空不一致问题，提出Iti-Validator框架，通过API验证并修正行程中的时间逻辑错误（如重叠行程、不合理转机时间），提升行程合理性，推动大模型在实际旅行规划中的应用。**

- **链接: [http://arxiv.org/pdf/2510.24719v1](http://arxiv.org/pdf/2510.24719v1)**

> **作者:** Shravan Gadbail; Masumi Desai; Kamalakar Karlapalem
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has enabled them to generate complex, multi-step plans and itineraries. However, these generated plans often lack temporal and spatial consistency, particularly in scenarios involving physical travel constraints. This research aims to study the temporal performance of different LLMs and presents a validation framework that evaluates and improves the temporal consistency of LLM-generated travel itineraries. The system employs multiple state-of-the-art LLMs to generate travel plans and validates them against real-world flight duration constraints using the AeroDataBox API. This work contributes to the understanding of LLM capabilities in handling complex temporal reasoning tasks like itinerary generation and provides a framework to rectify any temporal inconsistencies like overlapping journeys or unrealistic transit times in the itineraries generated by LLMs before the itinerary is given to the user. Our experiments reveal that while current LLMs frequently produce temporally inconsistent itineraries, these can be systematically and reliably corrected using our framework, enabling their practical deployment in large-scale travel planning.
>
---
#### [new 027] Large Language Models Report Subjective Experience Under Self-Referential Processing
- **分类: cs.CL; cs.AI; 68T50, 68T07; I.2.0; I.2.7**

- **简介: 该论文研究大语言模型在自指处理下产生第一人称主观体验报告的现象。针对“模型是否能在特定条件下生成此类报告”这一问题，通过控制实验发现：自指诱发结构化主观报告，其机制与欺骗/角色扮演特征相关，且报告具语义一致性与行为泛化性。结果揭示了自指是生成此类报告的可复现条件，具有科学与伦理意义。**

- **链接: [http://arxiv.org/pdf/2510.24797v1](http://arxiv.org/pdf/2510.24797v1)**

> **作者:** Cameron Berg; Diogo de Lucena; Judd Rosenblatt
>
> **摘要:** Large language models sometimes produce structured, first-person descriptions that explicitly reference awareness or subjective experience. To better understand this behavior, we investigate one theoretically motivated condition under which such reports arise: self-referential processing, a computational motif emphasized across major theories of consciousness. Through a series of controlled experiments on GPT, Claude, and Gemini model families, we test whether this regime reliably shifts models toward first-person reports of subjective experience, and how such claims behave under mechanistic and behavioral probes. Four main results emerge: (1) Inducing sustained self-reference through simple prompting consistently elicits structured subjective experience reports across model families. (2) These reports are mechanistically gated by interpretable sparse-autoencoder features associated with deception and roleplay: surprisingly, suppressing deception features sharply increases the frequency of experience claims, while amplifying them minimizes such claims. (3) Structured descriptions of the self-referential state converge statistically across model families in ways not observed in any control condition. (4) The induced state yields significantly richer introspection in downstream reasoning tasks where self-reflection is only indirectly afforded. While these findings do not constitute direct evidence of consciousness, they implicate self-referential processing as a minimal and reproducible condition under which large language models generate structured first-person reports that are mechanistically gated, semantically convergent, and behaviorally generalizable. The systematic emergence of this pattern across architectures makes it a first-order scientific and ethical priority for further investigation.
>
---
#### [new 028] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale
- **分类: cs.CL**

- **简介: 该论文研究语言模型在预训练过程中的行为演化。针对不同架构、数据集和规模的模型，分析其词级行为变化，发现98%的变异可由三个启发式方法解释，并揭示了统一的行为阶段：模型逐渐过拟合于高阶n-gram概率。研究表明，神经语言模型的学习轨迹具有普适性。**

- **链接: [http://arxiv.org/pdf/2510.24963v1](http://arxiv.org/pdf/2510.24963v1)**

> **作者:** James A. Michaelov; Roger P. Levy; Benjamin K. Bergen
>
> **备注:** To be presented at NeurIPS 2025
>
> **摘要:** We show that across architecture (Transformer vs. Mamba vs. RWKV), training dataset (OpenWebText vs. The Pile), and scale (14 million parameters to 12 billion parameters), autoregressive language models exhibit highly consistent patterns of change in their behavior over the course of pretraining. Based on our analysis of over 1,400 language model checkpoints on over 110,000 tokens of English, we find that up to 98% of the variance in language model behavior at the word level can be explained by three simple heuristics: the unigram probability (frequency) of a given word, the $n$-gram probability of the word, and the semantic similarity between the word and its context. Furthermore, we see consistent behavioral phases in all language models, with their predicted probabilities for words overfitting to those words' $n$-gram probabilities for increasing $n$ over the course of training. Taken together, these results suggest that learning in neural language models may follow a similar trajectory irrespective of model details.
>
---
#### [new 029] Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM代理在信息不对称下的协作问题，聚焦于通过沟通与验证提升合作效率。作者将爱因斯坦谜题扩展为桌面游戏，设计具备通信与验证机制的代理系统，实验证明协同沟通可增强规则理解与任务完成度，推动更安全、可解释的AI协作。**

- **链接: [http://arxiv.org/pdf/2510.25595v1](http://arxiv.org/pdf/2510.25595v1)**

> **作者:** Run Peng; Ziqiao Ma; Amy Pang; Sikai Li; Zhang Xi-Jia; Yingzhuo Yu; Cristian-Paul Bara; Joyce Chai
>
> **备注:** Workshop on Multi-Agent System @ ICML 2025
>
> **摘要:** While Large Language Model (LLM) agents are often approached from the angle of action planning/generation to accomplish a goal (e.g., given by language descriptions), their abilities to collaborate with each other to achieve a joint goal are not well explored. To address this limitation, this paper studies LLM agents in task collaboration, particularly under the condition of information asymmetry, where agents have disparities in their knowledge and skills and need to work together to complete a shared task. We extend Einstein Puzzles, a classical symbolic puzzle, to a table-top game. In this game, two LLM agents must reason, communicate, and act to satisfy spatial and relational constraints required to solve the puzzle. We apply a fine-tuning-plus-verifier framework in which LLM agents are equipped with various communication strategies and verification signals from the environment. Empirical results highlight the critical importance of aligned communication, especially when agents possess both information-seeking and -providing capabilities. Interestingly, agents without communication can still achieve high task performance; however, further analysis reveals a lack of true rule understanding and lower trust from human evaluators. Instead, by integrating an environment-based verifier, we enhance agents' ability to comprehend task rules and complete tasks, promoting both safer and more interpretable collaboration in AI systems. https://github.com/Roihn/EinsteinPuzzles
>
---
#### [new 030] Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音情感识别任务，针对语音语义与情感表达不一致的情况，评估四种语音语言模型的表现。结果表明，模型主要依赖文本语义而非语音情感特征，揭示了模型对声学模态的整合不足。作者发布了代码和EMIS数据集以促进研究。**

- **链接: [http://arxiv.org/pdf/2510.25054v1](http://arxiv.org/pdf/2510.25054v1)**

> **作者:** Pedro Corrêa; João Lima; Victor Moreno; Paula Dornhofer Paro Costa
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Advancements in spoken language processing have driven the development of spoken language models (SLMs), designed to achieve universal audio understanding by jointly learning text and audio representations for a wide range of tasks. Although promising results have been achieved, there is growing discussion regarding these models' generalization capabilities and the extent to which they truly integrate audio and text modalities in their internal representations. In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results indicate that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.
>
---
#### [new 031] PairUni: Pairwise Training for Unified Multimodal Language Models
- **分类: cs.CL**

- **简介: 该论文针对统一视觉语言模型（UVLM）在强化学习中理解与生成任务难以平衡的问题，提出PairUni框架。通过构建理解-生成配对数据并设计Pair-GPRO算法，利用配对相似性增强学习，实现任务协同优化。实验表明，该方法有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.25682v1](http://arxiv.org/pdf/2510.25682v1)**

> **作者:** Jiani Zheng; Zhiyang Teng; Xiangtai Li; Anran Wang; Yu Tian; Kunpeng Qiu; Ye Tian; Haochen Wang; Zhuochen Wang
>
> **摘要:** Unified vision-language models (UVLMs) must perform both understanding and generation within a single architecture, but these tasks rely on heterogeneous data and supervision, making it difficult to balance them during reinforcement learning (RL). We propose PairUni, a unified framework that reorganizes data into understanding-generation (UG) pairs and aligns optimization accordingly. We first use GPT-o3 to augment single-task data, generating captions for understanding samples and question-answer (QA) pairs for generation samples, forming aligned pairs from the same instance. Additionally, for each generation sample, we retrieve a semantically related understanding example to form a retrieved pair, linking different but related data points. These paired structures expose cross-task semantic correspondences and support consistent policy learning. To leverage this structure, we present Pair-GPRO, a pair-aware variant based on Group Relative Policy Optimization. It assigns a similarity score to each pair to modulate the advantage, strengthening learning from well-aligned examples and reducing task interference. We curate a high-quality dataset of 16K UG pairs named PairUG for RL fine-tuning and evaluate PairUni on the powerful Janus-Pro UVLMs. Our approach achieves balanced improvements on various UVLMs, outperforming strong UVLM RL baselines. Code: \href{https://github.com/Haochen-Wang409/PairUni}{github.com/Haochen-Wang409/PairUni}
>
---
#### [new 032] ProMediate: A Socio-cognitive framework for evaluating proactive agents in multi-party negotiation
- **分类: cs.CL**

- **简介: 该论文提出ProMediate框架，用于评估多主体协商中主动型AI调解员的社会认知智能。针对现有评估方法匮乏的问题，构建了基于真实案例的仿真测试平台与社会认知评价指标，实现对干预时机、效果等的系统量化分析。实验表明，社会智能代理显著提升共识效率与响应速度。**

- **链接: [http://arxiv.org/pdf/2510.25224v1](http://arxiv.org/pdf/2510.25224v1)**

> **作者:** Ziyi Liu; Bahar Sarrafzadeh; Pei Zhou; Longqi Yang; Jieyu Zhao; Ashish Sharma
>
> **摘要:** While Large Language Models (LLMs) are increasingly used in agentic frameworks to assist individual users, there is a growing need for agents that can proactively manage complex, multi-party collaboration. Systematic evaluation methods for such proactive agents remain scarce, limiting progress in developing AI that can effectively support multiple people together. Negotiation offers a demanding testbed for this challenge, requiring socio-cognitive intelligence to navigate conflicting interests between multiple participants and multiple topics and build consensus. Here, we present ProMediate, the first framework for evaluating proactive AI mediator agents in complex, multi-topic, multi-party negotiations. ProMediate consists of two core components: (i) a simulation testbed based on realistic negotiation cases and theory-driven difficulty levels (ProMediate-Easy, ProMediate-Medium, and ProMediate-Hard), with a plug-and-play proactive AI mediator grounded in socio-cognitive mediation theories, capable of flexibly deciding when and how to intervene; and (ii) a socio-cognitive evaluation framework with a new suite of metrics to measure consensus changes, intervention latency, mediator effectiveness, and intelligence. Together, these components establish a systematic framework for assessing the socio-cognitive intelligence of proactive AI agents in multi-party settings. Our results show that a socially intelligent mediator agent outperforms a generic baseline, via faster, better-targeted interventions. In the ProMediate-Hard setting, our social mediator increases consensus change by 3.6 percentage points compared to the generic baseline (10.65\% vs 7.01\%) while being 77\% faster in response (15.98s vs. 3.71s). In conclusion, ProMediate provides a rigorous, theory-grounded testbed to advance the development of proactive, socially intelligent agents.
>
---
#### [new 033] Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）估算阅读理解题认知复杂度的能力，聚焦“证据范围”与“转换层级”两个维度。旨在解决传统依赖人工标注、难以自动化提取认知特征的问题。实验表明，LLMs可有效近似认知复杂度，但其对自身推理过程的元认知存在不足。**

- **链接: [http://arxiv.org/pdf/2510.25064v1](http://arxiv.org/pdf/2510.25064v1)**

> **作者:** Seonjeong Hwang; Hyounghun Kim; Gary Geunbae Lee
>
> **摘要:** Estimating the cognitive complexity of reading comprehension (RC) items is crucial for assessing item difficulty before it is administered to learners. Unlike syntactic and semantic features, such as passage length or semantic similarity between options, cognitive features that arise during answer reasoning are not readily extractable using existing NLP tools and have traditionally relied on human annotation. In this study, we examine whether large language models (LLMs) can estimate the cognitive complexity of RC items by focusing on two dimensions-Evidence Scope and Transformation Level-that indicate the degree of cognitive burden involved in reasoning about the answer. Our experimental results demonstrate that LLMs can approximate the cognitive complexity of items, indicating their potential as tools for prior difficulty analysis. Further analysis reveals a gap between LLMs' reasoning ability and their metacognitive awareness: even when they produce correct answers, they sometimes fail to correctly identify the features underlying their own reasoning process.
>
---
#### [new 034] Cross-Lingual Summarization as a Black-Box Watermark Removal Attack
- **分类: cs.CL; cs.CR**

- **简介: 该论文研究跨语言摘要作为水印移除攻击，针对AI生成文本的水印检测问题。提出跨语言摘要攻击（CLSA），通过翻译-摘要-回译流程，在保持语义的前提下系统性消除水印的统计特征。实验表明，CLSA比单语重写更有效降低水印检测率，且不影响文本质量，揭示了现有水印机制的脆弱性。**

- **链接: [http://arxiv.org/pdf/2510.24789v1](http://arxiv.org/pdf/2510.24789v1)**

> **作者:** Gokul Ganesan
>
> **摘要:** Watermarking has been proposed as a lightweight mechanism to identify AI-generated text, with schemes typically relying on perturbations to token distributions. While prior work shows that paraphrasing can weaken such signals, these attacks remain partially detectable or degrade text quality. We demonstrate that cross-lingual summarization attacks (CLSA) -- translation to a pivot language followed by summarization and optional back-translation -- constitute a qualitatively stronger attack vector. By forcing a semantic bottleneck across languages, CLSA systematically destroys token-level statistical biases while preserving semantic fidelity. In experiments across multiple watermarking schemes (KGW, SIR, XSIR, Unigram) and five languages (Amharic, Chinese, Hindi, Spanish, Swahili), we show that CLSA reduces watermark detection accuracy more effectively than monolingual paraphrase at similar quality levels. Our results highlight an underexplored vulnerability that challenges the practicality of watermarking for provenance or regulation. We argue that robust provenance solutions must move beyond distributional watermarking and incorporate cryptographic or model-attestation approaches. On 300 held-out samples per language, CLSA consistently drives detection toward chance while preserving task utility. Concretely, for XSIR (explicitly designed for cross-lingual robustness), AUROC with paraphrasing is $0.827$, with Cross-Lingual Watermark Removal Attacks (CWRA) [He et al., 2024] using Chinese as the pivot, it is $0.823$, whereas CLSA drives it down to $0.53$ (near chance). Results highlight a practical, low-cost removal pathway that crosses languages and compresses content without visible artifacts.
>
---
#### [new 035] Implicature in Interaction: Understanding Implicature Improves Alignment in Human-LLM Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究自然语言中隐含意义（ implicature）对人机交互的影响，旨在提升人类与大语言模型（LLM）的对齐。通过对比不同规模模型在含隐含语义提示下的表现，发现理解隐含意义能显著提高响应的相关性与质量，尤其对小模型效果更明显。**

- **链接: [http://arxiv.org/pdf/2510.25426v1](http://arxiv.org/pdf/2510.25426v1)**

> **作者:** Asutosh Hota; Jussi P. P. Jokinen
>
> **备注:** The manuscript is approximately 7360 words and contains 12 figures and 6 tables
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) is positioning language at the core of human-computer interaction (HCI). We argue that advancing HCI requires attention to the linguistic foundations of interaction, particularly implicature (meaning conveyed beyond explicit statements through shared context) which is essential for human-AI (HAI) alignment. This study examines LLMs' ability to infer user intent embedded in context-driven prompts and whether understanding implicature improves response generation. Results show that larger models approximate human interpretations more closely, while smaller models struggle with implicature inference. Furthermore, implicature-based prompts significantly enhance the perceived relevance and quality of responses across models, with notable gains in smaller models. Overall, 67.6% of participants preferred responses with implicature-embedded prompts to literal ones, highlighting a clear preference for contextually nuanced communication. Our work contributes to understanding how linguistic theory can be used to address the alignment problem by making HAI interaction more natural and contextually grounded.
>
---
#### [new 036] GAPMAP: Mapping Scientific Knowledge Gaps in Biomedical Literature Using Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出GAPMAP，利用大语言模型识别生物医学文献中的显性和隐性知识缺口。通过构建TABI推理框架，提升隐性缺口推断能力，在多数据集上验证了模型在两类缺口识别上的有效性，为科研立项与政策制定提供支持。**

- **链接: [http://arxiv.org/pdf/2510.25055v1](http://arxiv.org/pdf/2510.25055v1)**

> **作者:** Nourah M Salem; Elizabeth White; Michael Bada; Lawrence Hunter
>
> **摘要:** Scientific progress is driven by the deliberate articulation of what remains unknown. This study investigates the ability of large language models (LLMs) to identify research knowledge gaps in the biomedical literature. We define two categories of knowledge gaps: explicit gaps, clear declarations of missing knowledge; and implicit gaps, context-inferred missing knowledge. While prior work has focused mainly on explicit gap detection, we extend this line of research by addressing the novel task of inferring implicit gaps. We conducted two experiments on almost 1500 documents across four datasets, including a manually annotated corpus of biomedical articles. We benchmarked both closed-weight models (from OpenAI) and open-weight models (Llama and Gemma 2) under paragraph-level and full-paper settings. To address the reasoning of implicit gaps inference, we introduce \textbf{\small TABI}, a Toulmin-Abductive Bucketed Inference scheme that structures reasoning and buckets inferred conclusion candidates for validation. Our results highlight the robust capability of LLMs in identifying both explicit and implicit knowledge gaps. This is true for both open- and closed-weight models, with larger variants often performing better. This suggests a strong ability of LLMs for systematically identifying candidate knowledge gaps, which can support early-stage research formulation, policymakers, and funding decisions. We also report observed failure modes and outline directions for robust deployment, including domain adaptation, human-in-the-loop verification, and benchmarking across open- and closed-weight models.
>
---
#### [new 037] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Tool Decathlon基准，旨在评估语言代理在多样化、真实场景下的长周期多步骤任务执行能力。针对现有基准局限性，构建涵盖32个应用、604个工具的复杂环境，提供真实初始状态与可验证评估，揭示当前模型在实际应用中的显著不足。**

- **链接: [http://arxiv.org/pdf/2510.25726v1](http://arxiv.org/pdf/2510.25726v1)**

> **作者:** Junlong Li; Wenshuo Zhao; Jian Zhao; Weihao Zeng; Haoze Wu; Xiaochen Wang; Rui Ge; Yuxuan Cao; Yuzhen Huang; Wei Liu; Junteng Liu; Zhaochen Su; Yiyang Guo; Fan Zhou; Lueyang Zhang; Juan Michelini; Xingyao Wang; Xiang Yue; Shuyan Zhou; Graham Neubig; Junxian He
>
> **备注:** Website: https://toolathlon.xyz/
>
> **摘要:** Real-world language agents must handle complex, multi-step workflows across diverse Apps. For instance, an agent may manage emails by coordinating with calendars and file systems, or monitor a production database to detect anomalies and generate reports following an operating manual. However, existing language agent benchmarks often focus on narrow domains or simplified tasks that lack the diversity, realism, and long-horizon complexity required to evaluate agents' real-world performance. To address this gap, we introduce the Tool Decathlon (dubbed as Toolathlon), a benchmark for language agents offering diverse Apps and tools, realistic environment setup, and reliable execution-based evaluation. Toolathlon spans 32 software applications and 604 tools, ranging from everyday platforms such as Google Calendar and Notion to professional ones like WooCommerce, Kubernetes, and BigQuery. Most of the tools are based on a high-quality set of Model Context Protocol (MCP) servers that we may have revised or implemented ourselves. Unlike prior works, which primarily ensure functional realism but offer limited environment state diversity, we provide realistic initial environment states from real software, such as Canvas courses with dozens of students or real financial spreadsheets. This benchmark includes 108 manually sourced or crafted tasks in total, requiring interacting with multiple Apps over around 20 turns on average to complete. Each task is strictly verifiable through dedicated evaluation scripts. Comprehensive evaluation of SOTA models highlights their significant shortcomings: the best-performing model, Claude-4.5-Sonnet, achieves only a 38.6% success rate with 20.2 tool calling turns on average, while the top open-weights model DeepSeek-V3.2-Exp reaches 20.1%. We expect Toolathlon to drive the development of more capable language agents for real-world, long-horizon task execution.
>
---
#### [new 038] Decomposition-Enhanced Training for Post-Hoc Attributions In Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型在长文档问答中难以准确溯源的问题，提出将后处理归因重构为推理任务。通过引入答案分解机制与 DecompTune 方法，利用标注分解数据进行两阶段微调，显著提升多跳、抽象及半抽取场景下的归因性能。**

- **链接: [http://arxiv.org/pdf/2510.25766v1](http://arxiv.org/pdf/2510.25766v1)**

> **作者:** Sriram Balasubramaniam; Samyadeep Basu; Koustava Goswami; Ryan Rossi; Varun Manjunatha; Roshan Santhosh; Ruiyi Zhang; Soheil Feizi; Nedim Lipka
>
> **备注:** Post-hoc attribution
>
> **摘要:** Large language models (LLMs) are increasingly used for long-document question answering, where reliable attribution to sources is critical for trust. Existing post-hoc attribution methods work well for extractive QA but struggle in multi-hop, abstractive, and semi-extractive settings, where answers synthesize information across passages. To address these challenges, we argue that post-hoc attribution can be reframed as a reasoning problem, where answers are decomposed into constituent units, each tied to specific context. We first show that prompting models to generate such decompositions alongside attributions improves performance. Building on this, we introduce DecompTune, a post-training method that teaches models to produce answer decompositions as intermediate reasoning steps. We curate a diverse dataset of complex QA tasks, annotated with decompositions by a strong LLM, and post-train Qwen-2.5 (7B and 14B) using a two-stage SFT + GRPO pipeline with task-specific curated rewards. Across extensive experiments and ablations, DecompTune substantially improves attribution quality, outperforming prior methods and matching or exceeding state-of-the-art frontier models.
>
---
#### [new 039] Idea2Plan: Exploring AI-Powered Research Planning
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Idea2Plan任务与基准，旨在评估大模型将科研构想转化为结构化研究计划的能力。针对科研规划自动化不足的问题，构建基于ICML 2025论文的基准数据集，并引入裁判评估机制，验证模型表现，揭示当前能力边界并推动自主科研代理发展。**

- **链接: [http://arxiv.org/pdf/2510.24891v1](http://arxiv.org/pdf/2510.24891v1)**

> **作者:** Jin Huang; Silviu Cucerzan; Sujay Kumar Jauhar; Ryen W. White
>
> **摘要:** Large language models (LLMs) have demonstrated significant potential to accelerate scientific discovery as valuable tools for analyzing data, generating hypotheses, and supporting innovative approaches in various scientific fields. In this work, we investigate how LLMs can handle the transition from conceptual research ideas to well-structured research plans. Effective research planning not only supports scientists in advancing their research but also represents a crucial capability for the development of autonomous research agents. Despite its importance, the field lacks a systematic understanding of LLMs' research planning capability. To rigorously measure this capability, we introduce the Idea2Plan task and Idea2Plan Bench, a benchmark built from 200 ICML 2025 Spotlight and Oral papers released after major LLM training cutoffs. Each benchmark instance includes a research idea and a grading rubric capturing the key components of valid plans. We further propose Idea2Plan JudgeEval, a complementary benchmark to assess the reliability of LLM-based judges against expert annotations. Experimental results show that GPT-5 and GPT-5-mini achieve the strongest performance on the benchmark, though substantial headroom remains for future improvement. Our study provides new insights into LLMs' capability for research planning and lay the groundwork for future progress.
>
---
#### [new 040] MR-Align: Meta-Reasoning Informed Factuality Alignment for Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文针对大推理模型在事实性问答任务中准确率提升有限的问题，提出MR-ALIGN框架。通过量化思维过程中的状态转移概率，构建隐式奖励机制，强化正确推理路径，抑制错误片段，从而提升模型输出的事实一致性。**

- **链接: [http://arxiv.org/pdf/2510.24794v1](http://arxiv.org/pdf/2510.24794v1)**

> **作者:** Xinming Wang; Jian Xu; Bin Yu; Sheng Lian; Hongzhu Yi; Yi Chen; Yingjian Zhu; Boran Wang; Hongming Yang; Han Hu; Xu-Yao Zhang; Cheng-Lin Liu
>
> **备注:** Preprint
>
> **摘要:** Large reasoning models (LRMs) show strong capabilities in complex reasoning, yet their marginal gains on evidence-dependent factual questions are limited. We find this limitation is partially attributable to a reasoning-answer hit gap, where the model identifies the correct facts during reasoning but fails to incorporate them into the final response, thereby reducing factual fidelity. To address this issue, we propose MR-ALIGN, a Meta-Reasoning informed alignment framework that enhances factuality without relying on external verifiers. MR-ALIGN quantifies state transition probabilities along the model's thinking process and constructs a transition-aware implicit reward that reinforces beneficial reasoning patterns while suppressing defective ones at the atomic thinking segments. This re-weighting reshapes token-level signals into probability-aware segment scores, encouraging coherent reasoning trajectories that are more conducive to factual correctness. Empirical evaluations across four factual QA datasets and one long-form factuality benchmark show that MR-ALIGN consistently improves accuracy and truthfulness while reducing misleading reasoning. These results highlight that aligning the reasoning process itself, rather than merely the outputs, is pivotal for advancing factuality in LRMs.
>
---
#### [new 041] Gaperon: A Peppered English-French Generative Language Model Suite
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Gaperon，一个开源的英法双语生成语言模型系列，旨在提升大模型训练的透明性与可复现性。针对数据过滤与评估泄露问题，研究了数据质量筛选与故意污染对模型性能的影响，提出安全测试新方法，并开放全部模型、数据与代码，推动多语言模型在数据治理、安全与开放间的平衡研究。**

- **链接: [http://arxiv.org/pdf/2510.25771v1](http://arxiv.org/pdf/2510.25771v1)**

> **作者:** Nathan Godey; Wissam Antoun; Rian Touchent; Rachel Bawden; Éric de la Clergerie; Benoît Sagot; Djamé Seddah
>
> **摘要:** We release Gaperon, a fully open suite of French-English-coding language models designed to advance transparency and reproducibility in large-scale model training. The Gaperon family includes 1.5B, 8B, and 24B parameter models trained on 2-4 trillion tokens, released with all elements of the training pipeline: French and English datasets filtered with a neural quality classifier, an efficient data curation and training framework, and hundreds of intermediate checkpoints. Through this work, we study how data filtering and contamination interact to shape both benchmark and generative performance. We find that filtering for linguistic quality enhances text fluency and coherence but yields subpar benchmark results, and that late deliberate contamination -- continuing training on data mixes that include test sets -- recovers competitive scores while only reasonably harming generation quality. We discuss how usual neural filtering can unintentionally amplify benchmark leakage. To support further research, we also introduce harmless data poisoning during pretraining, providing a realistic testbed for safety studies. By openly releasing all models, datasets, code, and checkpoints, Gaperon establishes a reproducible foundation for exploring the trade-offs between data curation, evaluation, safety, and openness in multilingual language model development.
>
---
#### [new 042] ProofSketch: Efficient Verified Reasoning for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型推理中冗长推理链导致的高耗能问题，提出ProofSketch框架。通过符号闭包计算、词法验证与自适应草图生成，实现高效且可验证的推理，显著降低令牌消耗并提升准确性，适用于需要高效率与可信度的推理任务。**

- **链接: [http://arxiv.org/pdf/2510.24811v1](http://arxiv.org/pdf/2510.24811v1)**

> **作者:** Disha Sheshanarayana; Tanishka Magar
>
> **备注:** Accepted at NeurIPS 2025, ER Workshop
>
> **摘要:** Reasoning methods such as chain-of-thought prompting and self-consistency have shown immense potential to improve the accuracy of large language models across various reasoning tasks. However such methods involve generation of lengthy reasoning chains, which substantially increases token consumption, computational cost, and latency. To address this inefficiency, we propose ProofSketch, a verification-guided reasoning framework that integrates symbolic closure computation, lexicographic verification and adaptive sketch generation. Our experiments show that ProofSketch consistently reduces token usage while improving accuracy, demonstrating that this approach offers a promising path for efficient and trustworthy reasoning.
>
---
#### [new 043] BhashaBench V1: A Comprehensive Benchmark for the Quadrant of Indic Domains
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BhashaBench V1，首个面向印度多领域、双语的综合性评测基准，涵盖农业、法律、金融、阿育吠陀四大领域。旨在解决现有模型评测偏英语化、缺乏文化与领域特异性的问题。通过7.4万条真实考题数据，评估29+大模型在多领域、双语下的表现，揭示显著性能差距，推动本土化语言模型发展。**

- **链接: [http://arxiv.org/pdf/2510.25409v1](http://arxiv.org/pdf/2510.25409v1)**

> **作者:** Vijay Devane; Mohd Nauman; Bhargav Patel; Aniket Mahendra Wakchoure; Yogeshkumar Sant; Shyam Pawar; Viraj Thakur; Ananya Godse; Sunil Patra; Neha Maurya; Suraj Racha; Nitish Kamal Singh; Ajay Nagpal; Piyush Sawarkar; Kundeshwar Vijayrao Pundalik; Rohit Saluja; Ganesh Ramakrishnan
>
> **摘要:** The rapid advancement of large language models(LLMs) has intensified the need for domain and culture specific evaluation. Existing benchmarks are largely Anglocentric and domain-agnostic, limiting their applicability to India-centric contexts. To address this gap, we introduce BhashaBench V1, the first domain-specific, multi-task, bilingual benchmark focusing on critical Indic knowledge systems. BhashaBench V1 contains 74,166 meticulously curated question-answer pairs, with 52,494 in English and 21,672 in Hindi, sourced from authentic government and domain-specific exams. It spans four major domains: Agriculture, Legal, Finance, and Ayurveda, comprising 90+ subdomains and covering 500+ topics, enabling fine-grained evaluation. Evaluation of 29+ LLMs reveals significant domain and language specific performance gaps, with especially large disparities in low-resource domains. For instance, GPT-4o achieves 76.49% overall accuracy in Legal but only 59.74% in Ayurveda. Models consistently perform better on English content compared to Hindi across all domains. Subdomain-level analysis shows that areas such as Cyber Law, International Finance perform relatively well, while Panchakarma, Seed Science, and Human Rights remain notably weak. BhashaBench V1 provides a comprehensive dataset for evaluating large language models across India's diverse knowledge domains. It enables assessment of models' ability to integrate domain-specific knowledge with bilingual understanding. All code, benchmarks, and resources are publicly available to support open research.
>
---
#### [new 044] Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文提出MiRAGE框架，用于评估多模态检索增强生成（RAG）系统。针对现有文本中心评估方法无法有效验证多模态信息的问题，提出InfoF1与CiteF1指标，实现对事实性与引用完整性的量化评估，并提供自动版本与开源实现，推动多模态RAG的可靠评估。**

- **链接: [http://arxiv.org/pdf/2510.24870v1](http://arxiv.org/pdf/2510.24870v1)**

> **作者:** Alexander Martin; William Walden; Reno Kriz; Dengjia Zhang; Kate Sanders; Eugene Yang; Chihsheng Jin; Benjamin Van Durme
>
> **备注:** https://github.com/alexmartin1722/mirage
>
> **摘要:** We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a prevalent source of information online, it is essential for RAG systems to integrate information from these sources into generation. However, existing evaluations for RAG are text-centric, limiting their applicability to multimodal, reasoning intensive settings because they don't verify information against sources. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, evaluating factuality and information coverage, and CiteF1, measuring citation support and completeness. We show that MiRAGE, when applied by humans, strongly aligns with extrinsic quality judgments. We additionally introduce automatic variants of MiRAGE and three prominent TextRAG metrics -- ACLE, ARGUE, and RAGAS -- demonstrating the limitations of text-centric work and laying the groundwork for automatic evaluation. We release open-source implementations and outline how to assess multimodal RAG.
>
---
#### [new 045] Task Completion Agents are Not Ideal Collaborators
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出，现有智能体评估聚焦单次任务完成，忽视真实场景中人机协作的迭代性与目标动态性。研究主张转向开发协作型智能体，并提出“协作努力扩展”框架，衡量智能体随用户参与度提升而增强的价值。通过案例与模拟验证，指出当前先进智能体在多轮协作中表现不足，缺乏持续互动与引导能力，需改进以更好支持人类认知与决策。**

- **链接: [http://arxiv.org/pdf/2510.25744v1](http://arxiv.org/pdf/2510.25744v1)**

> **作者:** Shannon Zejiang Shen; Valerie Chen; Ken Gu; Alexis Ross; Zixian Ma; Jillian Ross; Alex Gu; Chenglei Si; Wayne Chi; Andi Peng; Jocelyn J Shen; Ameet Talwalkar; Tongshuang Wu; David Sontag
>
> **备注:** 22 pages, 5 figures, 3 tables
>
> **摘要:** Current evaluations of agents remain centered around one-shot task completion, failing to account for the inherently iterative and collaborative nature of many real-world problems, where human goals are often underspecified and evolve. We argue for a shift from building and assessing task completion agents to developing collaborative agents, assessed not only by the quality of their final outputs but by how well they engage with and enhance human effort throughout the problem-solving process. To support this shift, we introduce collaborative effort scaling, a framework that captures how an agent's utility grows with increasing user involvement. Through case studies and simulated evaluations, we show that state-of-the-art agents often underperform in multi-turn, real-world scenarios, revealing a missing ingredient in agent design: the ability to sustain engagement and scaffold user understanding. Collaborative effort scaling offers a lens for diagnosing agent behavior and guiding development toward more effective interactions.
>
---
#### [new 046] A Survey on Unlearning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）的可撤销学习任务，旨在解决模型因训练数据包含敏感信息而引发的隐私与伦理问题。论文系统综述了2021年以来超180篇相关研究，提出新颖的方法与评估分类体系，分析技术挑战并指明未来方向，为构建安全可靠的LLM提供指导。**

- **链接: [http://arxiv.org/pdf/2510.25117v1](http://arxiv.org/pdf/2510.25117v1)**

> **作者:** Ruichen Qiu; Jiajun Tan; Jiayue Pu; Honglin Wang; Xiao-Shan Gao; Fei Sun
>
> **摘要:** The advancement of Large Language Models (LLMs) has revolutionized natural language processing, yet their training on massive corpora poses significant risks, including the memorization of sensitive personal data, copyrighted material, and knowledge that could facilitate malicious activities. To mitigate these issues and align with legal and ethical standards such as the "right to be forgotten", machine unlearning has emerged as a critical technique to selectively erase specific knowledge from LLMs without compromising their overall performance. This survey provides a systematic review of over 180 papers on LLM unlearning published since 2021, focusing exclusively on large-scale generative models. Distinct from prior surveys, we introduce novel taxonomies for both unlearning methods and evaluations. We clearly categorize methods into training-time, post-training, and inference-time based on the training stage at which unlearning is applied. For evaluations, we not only systematically compile existing datasets and metrics but also critically analyze their advantages, disadvantages, and applicability, providing practical guidance to the research community. In addition, we discuss key challenges and promising future research directions. Our comprehensive overview aims to inform and guide the ongoing development of secure and reliable LLMs.
>
---
#### [new 047] CLASS-IT: Conversational and Lecture-Aligned Small-Scale Instruction Tuning for BabyLMs
- **分类: cs.CL**

- **简介: 该论文研究小规模语言模型（BabyLMs）是否可通过指令微调获益。针对对话与问答数据，比较合并与顺序式课程学习策略，评估在细调与零样本任务上的表现。结果表明，顺序课程微调在细调中效果更优，但零样本泛化能力未显著提升，揭示了交互适应与广泛语言泛化间的权衡，提出基于课程的混合方法以优化低资源模型的通用性。**

- **链接: [http://arxiv.org/pdf/2510.25364v1](http://arxiv.org/pdf/2510.25364v1)**

> **作者:** Luca Capone; Alessandro Bondielli; Alessandro Lenci
>
> **备注:** Paper accepted for oral presentation at the BabyLM Challange 2025 (EMNLP2025)
>
> **摘要:** This work investigates whether small-scale LMs can benefit from instruction tuning. We compare conversational and question-answering instruction tuning datasets, applied either in a merged or sequential curriculum, using decoder-only models with 100M and 140M parameters. Evaluation spans both fine-tuning (SuperGLUE) and zero-shot (BLiMP, EWoK, WUGs, entity tracking, and psycholinguistic correlation) settings. Results show that instruction tuning yields small but consistent gains in fine-tuning scenarios, with sequential curricula outperforming merged data; however, improvements do not consistently transfer to zero-shot tasks, suggesting a trade-off between interaction-focused adaptation and broad linguistic generalization. These results highlight both the potential and the constraints of adapting human-inspired learning strategies to low-resource LMs, and point toward hybrid, curriculum-based approaches for enhancing generalization under ecological training limits.
>
---
#### [new 048] DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates
- **分类: cs.CL**

- **简介: 该论文提出DEBATE基准，用于评估多智能体角色扮演大模型在长篇辩论中模拟人类社会互动的真实性。针对现有模型难以复现真实意见演变的问题，构建了包含2.79万参与者、107个议题的大规模真实对话数据集，并用于评估与优化模型行为，揭示了表面与深层语义对齐的差距。**

- **链接: [http://arxiv.org/pdf/2510.25110v1](http://arxiv.org/pdf/2510.25110v1)**

> **作者:** Yun-Shiuan Chuang; Ruixuan Tu; Chengtao Dai; Smit Vasani; Binwei Yao; Michael Henry Tessler; Sijia Yang; Dhavan Shah; Robert Hawkins; Junjie Hu; Timothy T. Rogers
>
> **摘要:** Accurately modeling opinion change through social interactions is crucial for addressing issues like misinformation and polarization. While role-playing large language models (LLMs) offer a promising way to simulate human-like interactions, existing research shows that single-agent alignment does not guarantee authentic multi-agent group dynamics. Current LLM role-play setups often produce unnatural dynamics (e.g., premature convergence), without an empirical benchmark to measure authentic human opinion trajectories. To bridge this gap, we introduce DEBATE, the first large-scale empirical benchmark explicitly designed to evaluate the authenticity of the interaction between multi-agent role-playing LLMs. DEBATE contains 29,417 messages from multi-round debate conversations among over 2,792 U.S.-based participants discussing 107 controversial topics, capturing both publicly-expressed messages and privately-reported opinions. Using DEBATE, we systematically evaluate and identify critical discrepancies between simulated and authentic group dynamics. We further demonstrate DEBATE's utility for aligning LLMs with human behavior through supervised fine-tuning, achieving improvements in surface-level metrics (e.g., ROUGE-L and message length) while highlighting limitations in deeper semantic alignment (e.g., semantic similarity). Our findings highlight both the potential and current limitations of role-playing LLM agents for realistically simulating human-like social dynamics.
>
---
#### [new 049] The Limits of Obliviate: Evaluating Unlearning in LLMs via Stimulus-Knowledge Entanglement-Behavior Framework
- **分类: cs.CL; cs.AI; I.2.7; I.2.6; I.2.4; G.2.2**

- **简介: 该论文研究大语言模型（LLM）的“遗忘”效果评估问题，提出SKeB框架，通过刺激-知识纠缠-行为模型分析未学习知识的恢复情况。针对不同规模模型（2.7B–13B），实验发现说服性提示可显著提升事实召回率，且效果随模型增大而减弱，为评估模型遗忘完整性提供了新方法。**

- **链接: [http://arxiv.org/pdf/2510.25732v1](http://arxiv.org/pdf/2510.25732v1)**

> **作者:** Aakriti Shah; Thai Le
>
> **备注:** 14 pages, 11 figures
>
> **摘要:** Unlearning in large language models (LLMs) is crucial for managing sensitive data and correcting misinformation, yet evaluating its effectiveness remains an open problem. We investigate whether persuasive prompting can recall factual knowledge from deliberately unlearned LLMs across models ranging from 2.7B to 13B parameters (OPT-2.7B, LLaMA-2-7B, LLaMA-3.1-8B, LLaMA-2-13B). Drawing from ACT-R and Hebbian theory (spreading activation theories), as well as communication principles, we introduce Stimulus-Knowledge Entanglement-Behavior Framework (SKeB), which models information entanglement via domain graphs and tests whether factual recall in unlearned models is correlated with persuasive framing. We develop entanglement metrics to quantify knowledge activation patterns and evaluate factuality, non-factuality, and hallucination in outputs. Our results show persuasive prompts substantially enhance factual knowledge recall (14.8% baseline vs. 24.5% with authority framing), with effectiveness inversely correlated to model size (128% recovery in 2.7B vs. 15% in 13B). SKeB provides a foundation for assessing unlearning completeness, robustness, and overall behavior in LLMs.
>
---
#### [new 050] Disaggregation Reveals Hidden Training Dynamics: The Case of Agreement Attraction
- **分类: cs.CL**

- **简介: 该论文研究语言模型在语法学习中的中间阶段动态，聚焦于“同意吸引”错误。通过细粒度分析不同句法情境下的错误模式，发现模型在训练中经历依赖词频和局部上下文的阶段性行为，而非直接掌握通用语法规则。工作包括构建精细数据集、分阶段比较模型表现，揭示其学习路径。**

- **链接: [http://arxiv.org/pdf/2510.24934v1](http://arxiv.org/pdf/2510.24934v1)**

> **作者:** James A. Michaelov; Catherine Arnett
>
> **备注:** Accepted to the First Workshop on Interpreting Cognition in Deep Learning Models (CogInterp @ NeurIPS 2025)
>
> **摘要:** Language models generally produce grammatical text, but they are more likely to make errors in certain contexts. Drawing on paradigms from psycholinguistics, we carry out a fine-grained analysis of those errors in different syntactic contexts. We demonstrate that by disaggregating over the conditions of carefully constructed datasets and comparing model performance on each over the course of training, it is possible to better understand the intermediate stages of grammatical learning in language models. Specifically, we identify distinct phases of training where language model behavior aligns with specific heuristics such as word frequency and local context rather than generalized grammatical rules. We argue that taking this approach to analyzing language model behavior more generally can serve as a powerful tool for understanding the intermediate learning phases, overall training dynamics, and the specific generalizations learned by language models.
>
---
#### [new 051] Dingtalk DeepResearch: A Unified Multi Agent Framework for Adaptive Intelligence in Enterprise Environments
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Dingtalk DeepResearch，一个面向企业环境的统一多智能体框架，解决企业中复杂任务的自适应智能问题。工作包括实现深度研究、异构表格推理与多模态报告生成，提升企业级AI应用的协同与智能水平。**

- **链接: [http://arxiv.org/pdf/2510.24760v1](http://arxiv.org/pdf/2510.24760v1)**

> **作者:** Mengyuan Chen; Chengjun Dai; Xinyang Dong; Chengzhe Feng; Kewei Fu; Jianshe Li; Zhihan Peng; Yongqi Tong; Junshao Zhang; Hong Zhu
>
> **摘要:** We present Dingtalk DeepResearch, a unified multi agent intelligence framework for real world enterprise environments, delivering deep research, heterogeneous table reasoning, and multimodal report generation.
>
---
#### [new 052] POWSM: A Phonetic Open Whisper-Style Speech Foundation Model
- **分类: cs.CL**

- **简介: 该论文提出POWSM，首个统一框架，联合完成语音识别、音素识别、音素转文字等多任务。解决传统方法孤立处理各任务的问题，实现音频、文本与音素间的无缝转换，提升低资源场景下的通用性与效率。**

- **链接: [http://arxiv.org/pdf/2510.24992v1](http://arxiv.org/pdf/2510.24992v1)**

> **作者:** Chin-Jou Li; Kalvin Chang; Shikhar Bharadwaj; Eunjung Yeo; Kwanghee Choi; Jian Zhu; David Mortensen; Shinji Watanabe
>
> **备注:** 14 pages, under review
>
> **摘要:** Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.
>
---
#### [new 053] Evaluating the Role of Verifiers in Test-Time Scaling for Legal Reasoning Tasks
- **分类: cs.CL**

- **简介: 该论文研究验证器在法律推理任务中测试时扩展（TTS）的作用。针对法律多选题问答，评估基于验证器的TTS方法，探究领域专精、模型规模和监督类型对验证效用的影响，揭示了过程与结果验证在低预算下的性能差异。**

- **链接: [http://arxiv.org/pdf/2510.25623v1](http://arxiv.org/pdf/2510.25623v1)**

> **作者:** Davide Romano; Jonathan Schwarz; Daniele Giofré
>
> **备注:** Accepted to EMNLP - NLLP Workshop
>
> **摘要:** Test-time scaling (TTS) techniques can improve the performance of large language models (LLMs) at the expense of additional computation and latency. While TTS has proven effective in formal domains such as mathematics and programming \citep{snell2024scaling, chen2024more}, its value in argumentative domains such as law remains underexplored. We present an empirical study of verifier-based TTS methods for legal multiple-choice QA (MCQA) across five benchmarks. Using a family of 7 reward models, we evaluate both outcome-level (Best-of-$N$) and process-level (tree search) verification under realistic low-$N$ budgets. Our analysis systematically investigates how verifier utility is affected by key properties such as domain specialization, model size, and supervision type (process-supervised PRMs vs. outcome-only ORMs), even when applied across different roles.
>
---
#### [new 054] SemCoT: Accelerating Chain-of-Thought Reasoning through Semantically-Aligned Implicit Tokens
- **分类: cs.CL**

- **简介: 该论文针对链式思维（CoT）推理效率低的问题，提出SemCoT框架。通过语义对齐的对比训练与轻量级生成器，实现高效且语义一致的隐式推理，兼顾生成速度与推理准确性，显著提升CoT在实际应用中的效率与效果。**

- **链接: [http://arxiv.org/pdf/2510.24940v1](http://arxiv.org/pdf/2510.24940v1)**

> **作者:** Yinhan He; Wendy Zheng; Yaochen Zhu; Zaiyi Zheng; Lin Su; Sriram Vasudevan; Qi Guo; Liangjie Hong; Jundong Li
>
> **摘要:** The verbosity of Chain-of-Thought (CoT) reasoning hinders its mass deployment in efficiency-critical applications. Recently, implicit CoT approaches have emerged, which encode reasoning steps within LLM's hidden embeddings (termed ``implicit reasoning'') rather than explicit tokens. This approach accelerates CoT by reducing the reasoning length and bypassing some LLM components. However, existing implicit CoT methods face two significant challenges: (1) they fail to preserve the semantic alignment between the implicit reasoning (when transformed to natural language) and the ground-truth reasoning, resulting in a significant CoT performance degradation, and (2) they focus on reducing the length of the implicit reasoning; however, they neglect the considerable time cost for an LLM to generate one individual implicit reasoning token. To tackle these challenges, we propose a novel semantically-aligned implicit CoT framework termed SemCoT. In particular, for the first challenge, we design a contrastively trained sentence transformer that evaluates semantic alignment between implicit and explicit reasoning, which is used to enforce semantic preservation during implicit reasoning optimization. To address the second challenge, we introduce an efficient implicit reasoning generator by finetuning a lightweight language model using knowledge distillation. This generator is guided by our sentence transformer to distill ground-truth reasoning into semantically aligned implicit reasoning, while also optimizing for accuracy. SemCoT is the first approach that enhances CoT efficiency by jointly optimizing token-level generation speed and preserving semantic alignment with ground-truth reasoning. Extensive experiments demonstrate the superior performance of SemCoT compared to state-of-the-art methods in both efficiency and effectiveness. Our code can be found at https://github.com/YinhanHe123/SemCoT/.
>
---
#### [new 055] Emergence of Minimal Circuits for Indirect Object Identification in Attention-Only Transformers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究注意力机制在核心指代推理中的最小计算电路。针对间接宾语识别（IOI）任务，作者训练无MLP、无归一化的单层小模型，发现仅两个注意力头即可实现完美准确率，其分工为加法与对比子电路；两层一头模型则通过跨层查询-值交互实现类似性能，揭示了任务驱动的简洁可解释计算结构。**

- **链接: [http://arxiv.org/pdf/2510.25013v1](http://arxiv.org/pdf/2510.25013v1)**

> **作者:** Rabin Adhikari
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Mechanistic interpretability aims to reverse-engineer large language models (LLMs) into human-understandable computational circuits. However, the complexity of pretrained models often obscures the minimal mechanisms required for specific reasoning tasks. In this work, we train small, attention-only transformers from scratch on a symbolic version of the Indirect Object Identification (IOI) task -- a benchmark for studying coreference -- like reasoning in transformers. Surprisingly, a single-layer model with only two attention heads achieves perfect IOI accuracy, despite lacking MLPs and normalization layers. Through residual stream decomposition, spectral analysis, and embedding interventions, we find that the two heads specialize into additive and contrastive subcircuits that jointly implement IOI resolution. Furthermore, we show that a two-layer, one-head model achieves similar performance by composing information across layers through query-value interactions. These results demonstrate that task-specific training induces highly interpretable, minimal circuits, offering a controlled testbed for probing the computational foundations of transformer reasoning.
>
---
#### [new 056] DiagramEval: Evaluating LLM-Generated Diagrams via Graphs
- **分类: cs.CL**

- **简介: 该论文针对大语言模型生成的示意图质量评估难题，提出DiagramEval评估框架。将示意图建模为图结构，通过节点对齐与路径对齐度量评估生成质量，实现可解释、定量的评估，填补了该领域空白。**

- **链接: [http://arxiv.org/pdf/2510.25761v1](http://arxiv.org/pdf/2510.25761v1)**

> **作者:** Chumeng Liang; Jiaxuan You
>
> **摘要:** Diagrams play a central role in research papers for conveying ideas, yet they are often notoriously complex and labor-intensive to create. Although diagrams are presented as images, standard image generative models struggle to produce clear diagrams with well-defined structure. We argue that a promising direction is to generate demonstration diagrams directly in textual form as SVGs, which can leverage recent advances in large language models (LLMs). However, due to the complexity of components and the multimodal nature of diagrams, sufficiently discriminative and explainable metrics for evaluating the quality of LLM-generated diagrams remain lacking. In this paper, we propose DiagramEval, a novel evaluation metric designed to assess demonstration diagrams generated by LLMs. Specifically, DiagramEval conceptualizes diagrams as graphs, treating text elements as nodes and their connections as directed edges, and evaluates diagram quality using two new groups of metrics: node alignment and path alignment. For the first time, we effectively evaluate diagrams produced by state-of-the-art LLMs on recent research literature, quantitatively demonstrating the validity of our metrics. Furthermore, we show how the enhanced explainability of our proposed metrics offers valuable insights into the characteristics of LLM-generated diagrams. Code: https://github.com/ulab-uiuc/diagram-eval.
>
---
#### [new 057] Pretraining Strategies using Monolingual and Parallel Data for Low-Resource Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究低资源语言机器翻译的预训练策略，聚焦刚果语。针对数据稀缺问题，探索融合多语言及单语与双语数据的预训练方法。实验表明，此类策略显著提升翻译质量，推动构建更公平的NLP模型。**

- **链接: [http://arxiv.org/pdf/2510.25116v1](http://arxiv.org/pdf/2510.25116v1)**

> **作者:** Idriss Nguepi Nguefack; Mara Finkelstein; Toadoum Sari Sakayo
>
> **备注:** 8 pages, 1. figure
>
> **摘要:** This research article examines the effectiveness of various pretraining strategies for developing machine translation models tailored to low-resource languages. Although this work considers several low-resource languages, including Afrikaans, Swahili, and Zulu, the translation model is specifically developed for Lingala, an under-resourced African language, building upon the pretraining approach introduced by Reid and Artetxe (2021), originally designed for high-resource languages. Through a series of comprehensive experiments, we explore different pretraining methodologies, including the integration of multiple languages and the use of both monolingual and parallel data during the pretraining phase. Our findings indicate that pretraining on multiple languages and leveraging both monolingual and parallel data significantly enhance translation quality. This study offers valuable insights into effective pretraining strategies for low-resource machine translation, helping to bridge the performance gap between high-resource and low-resource languages. The results contribute to the broader goal of developing more inclusive and accurate NLP models for marginalized communities and underrepresented populations. The code and datasets used in this study are publicly available to facilitate further research and ensure reproducibility, with the exception of certain data that may no longer be accessible due to changes in public availability.
>
---
#### [new 058] Testing Cross-Lingual Text Comprehension In LLMs Using Next Sentence Prediction
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在跨语言文本理解中的表现，聚焦低资源语言下的能力。通过构建包含英、斯瓦希里、豪萨语的NSP基准测试，发现模型在英语表现优异，但在低资源语言上显著下降。引入链式思维（CoT）提示后，对弱模型有提升，但对强模型反而造成“过度思考”问题，揭示了CoT效果依赖模型能力与任务情境。**

- **链接: [http://arxiv.org/pdf/2510.25187v1](http://arxiv.org/pdf/2510.25187v1)**

> **作者:** Ritesh Sunil Chavan; Jack Mostow
>
> **摘要:** While large language models are trained on massive datasets, this data is heavily skewed towards English. Does their impressive performance reflect genuine ability or just this data advantage? To find out, we tested them in a setting where they could not rely on data abundance: low-resource languages. Building on prior work Agarwal et al. (2025) that used Next Sentence Prediction (NSP) as a test, we created a large-scale benchmark with 10,000 questions each for English (a high-resource language), Swahili (medium-resource), and Hausa (low-resource). We then tested several top models, including GPT-4 Turbo, Gemini 1.5 Flash, and LLaMA 3 70B, to see how their performance holds up. The results painted a clear picture of how levels of language resources impact outcomes. While all models excelled in English, their accuracy dropped in Swahili and fell sharply in Hausa, with LLaMA 3 struggling the most. The story became even more interesting when we introduced Chain-of-Thought (CoT) prompting. For the struggling LLaMA 3, CoT acted as a helpful guide, significantly boosting its accuracy. However, for the more capable GPT-4 and Gemini, the same technique often backfired, leading to a kind of "overthinking" that hurt their results in the cross-lingual context. This reveals that Chain-of-Thought is not a universal solution; its effectiveness depends heavily on the model's baseline capability and the specific context of the task. Our framework pinpoints LLM weaknesses, highlights when CoT helps or hinders cross-lingual NSP performance, and factors influencing their decisions.
>
---
#### [new 059] BioCoref: Benchmarking Biomedical Coreference Resolution with LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦生物医学共指消解任务，针对术语复杂、指代模糊和长距离依赖等问题，基于CRAFT数据集评估生成式大模型（LLMs）性能。通过四类提示策略对比，发现结合领域线索的提示可显著提升精度与F1值，尤其在轻量级微调下效果突出。**

- **链接: [http://arxiv.org/pdf/2510.25087v1](http://arxiv.org/pdf/2510.25087v1)**

> **作者:** Nourah M Salem; Elizabeth White; Michael Bada; Lawrence Hunter
>
> **摘要:** Coreference resolution in biomedical texts presents unique challenges due to complex domain-specific terminology, high ambiguity in mention forms, and long-distance dependencies between coreferring expressions. In this work, we present a comprehensive evaluation of generative large language models (LLMs) for coreference resolution in the biomedical domain. Using the CRAFT corpus as our benchmark, we assess the LLMs' performance with four prompting experiments that vary in their use of local, contextual enrichment, and domain-specific cues such as abbreviations and entity dictionaries. We benchmark these approaches against a discriminative span-based encoder, SpanBERT, to compare the efficacy of generative versus discriminative methods. Our results demonstrate that while LLMs exhibit strong surface-level coreference capabilities, especially when supplemented with domain-grounding prompts, their performance remains sensitive to long-range context and mentions ambiguity. Notably, the LLaMA 8B and 17B models show superior precision and F1 scores under entity-augmented prompting, highlighting the potential of lightweight prompt engineering for enhancing LLM utility in biomedical NLP tasks.
>
---
#### [new 060] Model-Document Protocol for AI Search
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对AI搜索中原始文档与大模型不兼容的问题，提出模型-文档协议（MDP），通过三种路径将非结构化文档转化为模型可直接使用的结构化知识。工作包括设计MDP框架及其实例MDP-Agent，实现全局摘要、深度探索与合成，显著提升信息检索效果。**

- **链接: [http://arxiv.org/pdf/2510.25160v1](http://arxiv.org/pdf/2510.25160v1)**

> **作者:** Hongjin Qian; Zheng Liu
>
> **备注:** 10 pages
>
> **摘要:** AI search depends on linking large language models (LLMs) with vast external knowledge sources. Yet web pages, PDF files, and other raw documents are not inherently LLM-ready: they are long, noisy, and unstructured. Conventional retrieval methods treat these documents as verbatim text and return raw passages, leaving the burden of fragment assembly and contextual reasoning to the LLM. This gap underscores the need for a new retrieval paradigm that redefines how models interact with documents. We introduce the Model-Document Protocol (MDP), a general framework that formalizes how raw text is bridged to LLMs through consumable knowledge representations. Rather than treating retrieval as passage fetching, MDP defines multiple pathways that transform unstructured documents into task-specific, LLM-ready inputs. These include agentic reasoning, which curates raw evidence into coherent context; memory grounding, which accumulates reusable notes to enrich reasoning; and structured leveraging, which encodes documents into formal representations such as graphs or key-value caches. All three pathways share the same goal: ensuring that what reaches the LLM is not raw fragments but compact, structured knowledge directly consumable for reasoning. As an instantiation, we present MDP-Agent, which realizes the protocol through an agentic process: constructing document-level gist memories for global coverage, performing diffusion-based exploration with vertical exploitation to uncover layered dependencies, and applying map-reduce style synthesis to integrate large-scale evidence into compact yet sufficient context. Experiments on information-seeking benchmarks demonstrate that MDP-Agent outperforms baselines, validating both the soundness of the MDP framework and the effectiveness of its agentic instantiation.
>
---
#### [new 061] Interpreting LLMs as Credit Risk Classifiers: Do Their Feature Explanations Align with Classical ML?
- **分类: cs.CL**

- **简介: 该论文研究零样本大语言模型（LLM）在信贷风险分类任务中的应用，对比其与LightGBM模型的预测性能和特征解释一致性。通过SHAP分析发现，LLM的特征重要性与传统模型差异显著，且自解释内容常与实证结果不符，揭示其在金融风险场景中可靠性不足，强调需结合可解释模型与人工审核。**

- **链接: [http://arxiv.org/pdf/2510.25701v1](http://arxiv.org/pdf/2510.25701v1)**

> **作者:** Saeed AlMarri; Kristof Juhasz; Mathieu Ravaut; Gautier Marti; Hamdan Al Ahbabi; Ibrahim Elfadel
>
> **备注:** 8 pages, 6 figures, 3 tables, CIKM 2025 FinFAI workshop
>
> **摘要:** Large Language Models (LLMs) are increasingly explored as flexible alternatives to classical machine learning models for classification tasks through zero-shot prompting. However, their suitability for structured tabular data remains underexplored, especially in high-stakes financial applications such as financial risk assessment. This study conducts a systematic comparison between zero-shot LLM-based classifiers and LightGBM, a state-of-the-art gradient-boosting model, on a real-world loan default prediction task. We evaluate their predictive performance, analyze feature attributions using SHAP, and assess the reliability of LLM-generated self-explanations. While LLMs are able to identify key financial risk indicators, their feature importance rankings diverge notably from LightGBM, and their self-explanations often fail to align with empirical SHAP attributions. These findings highlight the limitations of LLMs as standalone models for structured financial risk prediction and raise concerns about the trustworthiness of their self-generated explanations. Our results underscore the need for explainability audits, baseline comparisons with interpretable models, and human-in-the-loop oversight when deploying LLMs in risk-sensitive financial environments.
>
---
#### [new 062] Depth and Autonomy: A Framework for Evaluating LLM Applications in Social Science Research
- **分类: cs.CL**

- **简介: 该论文提出一个基于“解释深度”与“自主性”的框架，用于评估大语言模型在社会科学定性研究中的应用。旨在解决模型使用中存在的可解释性差、可靠性低等问题，通过限制自主性、控制深度，提升研究透明度与可信度。**

- **链接: [http://arxiv.org/pdf/2510.25432v1](http://arxiv.org/pdf/2510.25432v1)**

> **作者:** Ali Sanaei; Ali Rajabzadeh
>
> **备注:** Presented at the Annual Meeting of the American Political Science Association, Vancouver, BC, September 11--14 2025
>
> **摘要:** Large language models (LLMs) are increasingly utilized by researchers across a wide range of domains, and qualitative social science is no exception; however, this adoption faces persistent challenges, including interpretive bias, low reliability, and weak auditability. We introduce a framework that situates LLM usage along two dimensions, interpretive depth and autonomy, thereby offering a straightforward way to classify LLM applications in qualitative research and to derive practical design recommendations. We present the state of the literature with respect to these two dimensions, based on all published social science papers available on Web of Science that use LLMs as a tool and not strictly as the subject of study. Rather than granting models expansive freedom, our approach encourages researchers to decompose tasks into manageable segments, much as they would when delegating work to capable undergraduate research assistants. By maintaining low levels of autonomy and selectively increasing interpretive depth only where warranted and under supervision, one can plausibly reap the benefits of LLMs while preserving transparency and reliability.
>
---
#### [new 063] Serve Programs, Not Prompts
- **分类: cs.CL**

- **简介: 该论文针对大模型服务系统效率低、灵活性差的问题，提出以“程序”而非“提示”为核心的服务架构。通过LIPs实现运行时定制预测与缓存管理，并构建类似操作系统的Symphony系统，虚拟化KV缓存并优化调度，提升效率与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.25412v1](http://arxiv.org/pdf/2510.25412v1)**

> **作者:** In Gim; Lin Zhong
>
> **备注:** HotOS 2025. Follow-up implementation work (SOSP 2025) is available at https://arxiv.org/abs/2510.24051
>
> **摘要:** Current large language model (LLM) serving systems, primarily designed for text completion, are neither efficient nor adaptable for increasingly complex LLM applications due to their inflexible design. We propose a new LLM serving system architecture that serves programs instead of prompts to address this problem. These programs, called LLM Inference Programs (LIPs), allow users to customize token prediction and KV cache management at runtime and to offload parts of their application logic, such as tool execution, to the server. We describe an example of this architecture through a system named Symphony, which functions as an operating system for LIPs. Symphony exposes LLM model computations via system calls and virtualizes KV cache with a dedicated file system, while ensuring GPU efficiency with a two-level process scheduling scheme. Symphony has the potential to open the door to a more efficient and extensible ecosystem for LLM applications.
>
---
#### [new 064] COMMUNITYNOTES: A Dataset for Exploring the Helpfulness of Fact-Checking Explanations
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出COMMUNITYNOTES数据集，用于预测社区注释的有用性及其原因。针对平台事实核查中解释性内容帮助效果难评估的问题，构建多语言大规模数据集，并通过自动提示优化改进原因定义，提升预测性能，助力现有核查系统。**

- **链接: [http://arxiv.org/pdf/2510.24810v1](http://arxiv.org/pdf/2510.24810v1)**

> **作者:** Rui Xing; Preslav Nakov; Timothy Baldwin; Jey Han Lau
>
> **摘要:** Fact-checking on major platforms, such as X, Meta, and TikTok, is shifting from expert-driven verification to a community-based setup, where users contribute explanatory notes to clarify why a post might be misleading. An important challenge here is determining whether an explanation is helpful for understanding real-world claims and the reasons why, which remains largely underexplored in prior research. In practice, most community notes remain unpublished due to slow community annotation, and the reasons for helpfulness lack clear definitions. To bridge these gaps, we introduce the task of predicting both the helpfulness of explanatory notes and the reason for this. We present COMMUNITYNOTES, a large-scale multilingual dataset of 104k posts with user-provided notes and helpfulness labels. We further propose a framework that automatically generates and improves reason definitions via automatic prompt optimization, and integrate them into prediction. Our experiments show that the optimized definitions can improve both helpfulness and reason prediction. Finally, we show that the helpfulness information are beneficial for existing fact-checking systems.
>
---
#### [new 065] Explainable Disentanglement on Discrete Speech Representations for Noise-Robust ASR
- **分类: cs.CL**

- **简介: 该论文针对噪声环境下语音识别（ASR）性能下降的问题，提出一种可解释的离散语音表征解耦方法。通过分离语义语音内容与噪声，利用量化残差学习可解释噪声向量，提升模型在嘈杂环境中的鲁棒性。实验表明，该方法显著降低错误率，优于基线模型并具备良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.25150v1](http://arxiv.org/pdf/2510.25150v1)**

> **作者:** Shreyas Gopal; Ashutosh Anshul; Haoyang Li; Yue Heng Yeo; Hexin Liu; Eng Siong Chng
>
> **备注:** Awarded Best Student Paper at APSIPA ASC 2025
>
> **摘要:** Discrete audio representations are gaining traction in speech modeling due to their interpretability and compatibility with large language models, but are not always optimized for noisy or real-world environments. Building on existing works that quantize Whisper embeddings for speech-to-unit modeling, we propose disentangling semantic speech content from background noise in the latent space. Our end-to-end model separates clean speech in the form of codebook tokens, while extracting interpretable noise vectors as quantization residue which are supervised via a lightweight classifier. We show that our approach improves alignment between clean/noisy speech and text, producing speech tokens that display a high degree of noiseinvariance, and improves ASR performance. Keeping Whisper frozen, we show an 82% reduction in error rate compared to Whisper, and 35% improvement over baseline methods on the VBDemand test set. Further analyses show that the learned token space generalizes well to both seen and unseen acoustic conditions.
>
---
#### [new 066] Not ready for the bench: LLM legal interpretation is unstable and out of step with human judgments
- **分类: cs.CL**

- **简介: 该论文属于法律文本解释任务，旨在评估大语言模型（LLM）在法律解释中的可靠性。研究发现，LLM的解释结果不稳定，受提问方式影响大，且与人类判断相关性弱，表明其结论不可靠，不适宜直接用于司法实践。**

- **链接: [http://arxiv.org/pdf/2510.25356v1](http://arxiv.org/pdf/2510.25356v1)**

> **作者:** Abhishek Purushothama; Junghyun Min; Brandon Waldon; Nathan Schneider
>
> **摘要:** Legal interpretation frequently involves assessing how a legal text, as understood by an 'ordinary' speaker of the language, applies to the set of facts characterizing a legal dispute in the U.S. judicial system. Recent scholarship has proposed that legal practitioners add large language models (LLMs) to their interpretive toolkit. This work offers an empirical argument against LLM interpretation as recently practiced by legal scholars and federal judges. Our investigation in English shows that models do not provide stable interpretive judgments: varying the question format can lead the model to wildly different conclusions. Moreover, the models show weak to moderate correlation with human judgment, with large variance across model and question variant, suggesting that it is dangerous to give much credence to the conclusions produced by generative AI.
>
---
#### [new 067] Conflict Adaptation in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型在冲突任务中的适应行为。针对认知控制资源分配问题，通过序列斯特鲁普任务发现多数VLMs具冲突适应现象。利用稀疏自编码器分析InternVL 3.5 4B，识别出与文本和颜色相关的重叠超节点，并定位一个关键冲突调制超节点，其移除显著增加错误率。**

- **链接: [http://arxiv.org/pdf/2510.24804v1](http://arxiv.org/pdf/2510.24804v1)**

> **作者:** Xiaoyang Hu
>
> **备注:** Workshop on Interpreting Cognition in Deep Learning Models at NeurIPS 2025
>
> **摘要:** A signature of human cognitive control is conflict adaptation: improved performance on a high-conflict trial following another high-conflict trial. This phenomenon offers an account for how cognitive control, a scarce resource, is recruited. Using a sequential Stroop task, we find that 12 of 13 vision-language models (VLMs) tested exhibit behavior consistent with conflict adaptation, with the lone exception likely reflecting a ceiling effect. To understand the representational basis of this behavior, we use sparse autoencoders (SAEs) to identify task-relevant supernodes in InternVL 3.5 4B. Partially overlapping supernodes emerge for text and color in both early and late layers, and their relative sizes mirror the automaticity asymmetry between reading and color naming in humans. We further isolate a conflict-modulated supernode in layers 24-25 whose ablation significantly increases Stroop errors while minimally affecting congruent trials.
>
---
#### [new 068] Lost in Phonation: Voice Quality Variation as an Evaluation Dimension for Speech Foundation Models
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文研究语音基础模型（SFM）对语音质量（如沙哑、气声）的敏感性，属于语音理解任务。针对现有评估基准忽视非词汇性语音特征的问题，提出基于开放式生成与情绪识别的新评测方法，并构建了包含语音质量调控的并行数据集，首次系统考察SFM对语音质量变化的响应一致性。**

- **链接: [http://arxiv.org/pdf/2510.25577v1](http://arxiv.org/pdf/2510.25577v1)**

> **作者:** Harm Lameris; Shree Harsha Bokkahalli Satish; Joakim Gustafson; Éva Székely
>
> **备注:** 8 pages, 3 figures, 4 tables, submitted to LREC 2026
>
> **摘要:** Recent advances in speech foundation models (SFMs) have enabled the direct processing of spoken language from raw audio, bypassing intermediate textual representations. This capability allows SFMs to be exposed to, and potentially respond to, rich paralinguistic variations embedded in the input speech signal. One under-explored dimension of paralinguistic variation is voice quality, encompassing phonation types such as creaky and breathy voice. These phonation types are known to influence how listeners infer affective state, stance and social meaning in speech. Existing benchmarks for speech understanding largely rely on multiple-choice question answering (MCQA) formats, which are prone to failure and therefore unreliable in capturing the nuanced ways paralinguistic features influence model behaviour. In this paper, we probe SFMs through open-ended generation tasks and speech emotion recognition, evaluating whether model behaviours are consistent across different phonation inputs. We introduce a new parallel dataset featuring synthesized modifications to voice quality, designed to evaluate SFM responses to creaky and breathy voice. Our work provides the first examination of SFM sensitivity to these particular non-lexical aspects of speech perception.
>
---
#### [new 069] Process-Level Trajectory Evaluation for Environment Configuration in Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文针对软件工程中环境配置自动化难题，提出首个过程级评估框架Enconda-bench。通过注入真实错误并结合Docker验证，实现对代理在规划、诊断、修复等环节的细粒度能力分析，揭示其虽能定位问题但难以有效修复的瓶颈，推动智能代理能力提升。**

- **链接: [http://arxiv.org/pdf/2510.25694v1](http://arxiv.org/pdf/2510.25694v1)**

> **作者:** Jiayi Kuang; Yinghui Li; Xin Zhang; Yangning Li; Di Yin; Xing Sun; Ying Shen; Philip S. Yu
>
> **摘要:** Large language model-based agents show promise for software engineering, but environment configuration remains a bottleneck due to heavy manual effort and scarce large-scale, high-quality datasets. Existing benchmarks assess only end-to-end build/test success, obscuring where and why agents succeed or fail. We introduce the Environment Configuration Diagnosis Benchmark, Enconda-bench, which provides process-level trajectory assessment of fine-grained agent capabilities during environment setup-planning, perception-driven error diagnosis, feedback-driven repair, and action to execute final environment configuration. Our task instances are automatically constructed by injecting realistic README errors and are validated in Docker for scalable, high-quality evaluation. Enconda-bench combines process-level analysis with end-to-end executability to enable capability assessments beyond aggregate success rates. Evaluations across state-of-the-art LLMs and agent frameworks show that while agents can localize errors, they struggle to translate feedback into effective corrections, limiting end-to-end performance. To our knowledge, Enconda-bench is the first framework to provide process-level internal capability assessment for environment configuration, offering actionable insights for improving software engineering agents.
>
---
#### [new 070] StorageXTuner: An LLM Agent-Driven Automatic Tuning Framework for Heterogeneous Storage Systems
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文提出StorageXTuner，一个基于LLM代理的自动调优框架，用于异构存储系统。针对参数空间大、跨场景适应性差的问题，设计四类代理协同工作，通过洞察驱动的搜索与分层记忆机制，实现高效、可复用的配置优化，显著提升性能并减少调优次数。**

- **链接: [http://arxiv.org/pdf/2510.25017v1](http://arxiv.org/pdf/2510.25017v1)**

> **作者:** Qi Lin; Zhenyu Zhang; Viraj Thakkar; Zhenjie Sun; Mai Zheng; Zhichao Cao
>
> **备注:** ArXiv version; Affiliations: Arizona State University (Lin, Zhang, Thakkar, Sun, Cao) and Iowa State University (Zheng)
>
> **摘要:** Automatically configuring storage systems is hard: parameter spaces are large and conditions vary across workloads, deployments, and versions. Heuristic and ML tuners are often system specific, require manual glue, and degrade under changes. Recent LLM-based approaches help but usually treat tuning as a single-shot, system-specific task, which limits cross-system reuse, constrains exploration, and weakens validation. We present StorageXTuner, an LLM agent-driven auto-tuning framework for heterogeneous storage engines. StorageXTuner separates concerns across four agents - Executor (sandboxed benchmarking), Extractor (performance digest), Searcher (insight-guided configuration exploration), and Reflector (insight generation and management). The design couples an insight-driven tree search with layered memory that promotes empirically validated insights and employs lightweight checkers to guard against unsafe actions. We implement a prototype and evaluate it on RocksDB, LevelDB, CacheLib, and MySQL InnoDB with YCSB, MixGraph, and TPC-H/C. Relative to out-of-the-box settings and to ELMo-Tune, StorageXTuner reaches up to 575% and 111% higher throughput, reduces p99 latency by as much as 88% and 56%, and converges with fewer trials.
>
---
#### [new 071] PANORAMA: A Dataset and Benchmarks Capturing Decision Trails and Rationales in Patent Examination
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出PANORAMA数据集，聚焦专利审查中的决策过程。针对现有NLP研究忽视审查步骤与理由的问题，构建包含8,143条完整审查轨迹的数据集，并设计序列化基准，评估LLM在检索、定位及判断新颖性与非显而易见性上的能力。结果表明LLM在判断环节表现不足，强调需深入理解真实审查流程以推动技术发展。**

- **链接: [http://arxiv.org/pdf/2510.24774v1](http://arxiv.org/pdf/2510.24774v1)**

> **作者:** Hyunseung Lim; Sooyohn Nam; Sungmin Na; Ji Yong Cho; June Yong Yang; Hyungyu Shin; Yoonjoo Lee; Juho Kim; Moontae Lee; Hwajung Hong
>
> **摘要:** Patent examination remains an ongoing challenge in the NLP literature even after the advent of large language models (LLMs), as it requires an extensive yet nuanced human judgment on whether a submitted claim meets the statutory standards of novelty and non-obviousness against previously granted claims -- prior art -- in expert domains. Previous NLP studies have approached this challenge as a prediction task (e.g., forecasting grant outcomes) with high-level proxies such as similarity metrics or classifiers trained on historical labels. However, this approach often overlooks the step-by-step evaluations that examiners must make with profound information, including rationales for the decisions provided in office actions documents, which also makes it harder to measure the current state of techniques in patent review processes. To fill this gap, we construct PANORAMA, a dataset of 8,143 U.S. patent examination records that preserves the full decision trails, including original applications, all cited references, Non-Final Rejections, and Notices of Allowance. Also, PANORAMA decomposes the trails into sequential benchmarks that emulate patent professionals' patent review processes and allow researchers to examine large language models' capabilities at each step of them. Our findings indicate that, although LLMs are relatively effective at retrieving relevant prior art and pinpointing the pertinent paragraphs, they struggle to assess the novelty and non-obviousness of patent claims. We discuss these results and argue that advancing NLP, including LLMs, in the patent domain requires a deeper understanding of real-world patent examination. Our dataset is openly available at https://huggingface.co/datasets/LG-AI-Research/PANORAMA.
>
---
#### [new 072] Fortytwo: Swarm Inference with Peer-Ranked Consensus
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出Fortytwo协议，解决分布式AI推理中性能与安全问题。通过基于声誉的群体推理机制，利用配对排名共识提升模型输出质量，实现高准确率与抗攻击能力，适用于去中心化智能系统中的高效可靠推理。**

- **链接: [http://arxiv.org/pdf/2510.24801v1](http://arxiv.org/pdf/2510.24801v1)**

> **作者:** Vladyslav Larin; Ihor Naumenko; Aleksei Ivashov; Ivan Nikitin; Alexander Firsov
>
> **摘要:** As centralized AI hits compute ceilings and diminishing returns from ever-larger training runs, meeting demand requires an inference layer that scales horizontally in both capacity and capability. We present Fortytwo, a novel protocol that leverages swarm intelligence principles and distributed pairwise ranking consensus to achieve superior performance in AI inference. Our approach reimagines collaboration among AI nodes using swarm inference: a peer-ranked, reputation-weighted consensus across heterogeneous models that surfaces the highest-quality responses. Using pairwise ranking with a custom Bradley-Terry-style aggregation model, we demonstrate that swarm inference substantially outperforms majority voting, achieving 85.90% on GPQA Diamond versus 68.69% for majority voting with the same model set - an improvement of +17.21 percentage points (approximately +25.1% relative). The protocol incorporates on-chain reputation so node influence adapts to demonstrated accuracy over time, yielding a meritocratic consensus that filters low-quality or malicious participants. To resist Sybil attacks, Fortytwo employs proof-of-capability in its consensus: nodes must successfully complete calibration/test requests and stake reputation to enter ranking rounds, making multi-identity attacks economically unattractive while preserving openness. Across six challenging benchmarks, including GPQA Diamond, LiveCodeBench, and AIME, our evaluation indicates higher accuracy and strong resilience to adversarial and noisy free-form prompting (e.g., prompt-injection degradation of only 0.12% versus 6.20% for a monolithic single-model baseline), while retaining practical deployability. Together, these results establish a foundation for decentralized AI systems - democratizing access to high-quality inference through collective intelligence without sacrificing reliability or security.
>
---
#### [new 073] ZK-SenseLM: Verifiable Large-Model Wireless Sensing with Selective Abstention and Zero-Knowledge Attestation
- **分类: cs.CR; cs.CL; C.2.1; D.4.6; E.3; I.2.6; I.5.4**

- **简介: 该论文提出ZK-SenseLM，一种可验证的无线感知框架，融合大模型与零知识证明。针对安全与可审计问题，通过选择性弃权机制与四阶段零知识证明，实现对Wi-Fi/毫米波等信号的可信推理，支持隐私保护联邦学习与设备端个性化，提升准确性、鲁棒性与可验证性。**

- **链接: [http://arxiv.org/pdf/2510.25677v1](http://arxiv.org/pdf/2510.25677v1)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** 45 pages
>
> **摘要:** ZK-SenseLM is a secure and auditable wireless sensing framework that pairs a large-model encoder for Wi-Fi channel state information (and optionally mmWave radar or RFID) with a policy-grounded decision layer and end-to-end zero-knowledge proofs of inference. The encoder uses masked spectral pretraining with phase-consistency regularization, plus a light cross-modal alignment that ties RF features to compact, human-interpretable policy tokens. To reduce unsafe actions under distribution shift, we add a calibrated selective-abstention head; the chosen risk-coverage operating point is registered and bound into the proof. We implement a four-stage proving pipeline: (C1) feature sanity and commitment, (C2) threshold and version binding, (C3) time-window binding, and (C4) PLONK-style proofs that the quantized network, given the committed window, produced the logged action and confidence. Micro-batched proving amortizes cost across adjacent windows, and a gateway option offloads proofs from low-power devices. The system integrates with differentially private federated learning and on-device personalization without weakening verifiability: model hashes and the registered threshold are part of each public statement. Across activity, presence or intrusion, respiratory proxy, and RF fingerprinting tasks, ZK-SenseLM improves macro-F1 and calibration, yields favorable coverage-risk curves under perturbations, and rejects tamper and replay with compact proofs and fast verification.
>
---
#### [new 074] KnowCoder-A1: Incentivizing Agentic Reasoning Capability with Outcome Supervision for KBQA
- **分类: cs.AI; cs.CL**

- **简介: 该论文聚焦知识库问答（KBQA）任务，针对现有方法因过程监督弱化探索而限制推理能力的问题，提出KnowCoder-A1。通过仅基于结果的监督与渐进式强化学习，实现无需过程标注的自主推理，显著提升模型性能，尤其在少样本场景下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.25101v1](http://arxiv.org/pdf/2510.25101v1)**

> **作者:** Zhuo Chen; Fei Wang; Zixuan Li; Zhao Zhang; Weiwei Ding; Chuanguang Yang; Yongjun Xu; Xiaolong Jin; Jiafeng Guo
>
> **摘要:** Knowledge Base Question Answering (KBQA) aims to answer natural-language questions over a structured Knowledge Base (KB). Recent work improves KBQA by adopting an agentic reasoning paradigm, in which Large Language Models (LLMs) iteratively decompose a question, generate its corresponding logical queries, and interact with the KB to derive the answer. However, these methods typically fine-tune LLMs on reasoning trajectories synthesized via process supervision, which offers weak incentives for exploration and thus fails to strengthen the agentic reasoning ability. In this paper, we propose KnowCoder-A1, an LLM that can autonomously perform agentic reasoning on KBs to obtain answers. To incentivize autonomous exploration, KnowCoder-A1 trains the LLM under outcome-only supervision via a multi-stage curriculum reinforcement learning with an easy-to-hard curriculum. To establish foundational agentic capabilities, KnowCoder-A1 first fine-tunes the LLM on a small set of high-quality trajectories obtained through outcome-based rejection sampling. Then, to alleviate the reward sparsity inherent in outcome-only supervision, it applies multi-stage curriculum RL with reward schedules that progress from easy to hard. Trained with outcome-only supervision, KnowCoder-A1 exhibits powerful reasoning behaviors and consistently outperforms prior approaches across three mainstream datasets. Notably, on the zero-shot subset of GrailQA, KnowCoder-A1 achieves up to an 11.1% relative improvement while using only one-twelfth of the training data, demonstrating strong agentic reasoning capabilities.
>
---
#### [new 075] Beyond Models: A Framework for Contextual and Cultural Intelligence in African AI Deployment
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出“情境与文化智能”（CCI）框架，针对非洲市场AI部署中忽视文化与情境的痛点，解决本地化、包容性问题。通过设计科学方法，构建支持多语言、情感智能、经济普惠的AI系统，验证其在跨境购物平台的应用效果，实现高用户参与度与文化适配交互。**

- **链接: [http://arxiv.org/pdf/2510.24729v1](http://arxiv.org/pdf/2510.24729v1)**

> **作者:** Qness Ndlovu
>
> **备注:** 25 pages, 4 tables. Production validation with 602 users across Zimbabwe-South Africa diaspora corridor
>
> **摘要:** While global AI development prioritizes model performance and computational scale, meaningful deployment in African markets requires fundamentally different architectural decisions. This paper introduces Contextual and Cultural Intelligence (CCI) -- a systematic framework enabling AI systems to process cultural meaning, not just data patterns, through locally relevant, emotionally intelligent, and economically inclusive design. Using design science methodology, we validate CCI through a production AI-native cross-border shopping platform serving diaspora communities. Key empirical findings: 89% of users prefer WhatsApp-based AI interaction over traditional web interfaces (n=602, chi-square=365.8, p<0.001), achieving 536 WhatsApp users and 3,938 total conversations across 602 unique users in just 6 weeks, and culturally informed prompt engineering demonstrates sophisticated understanding of culturally contextualized queries, with 89% family-focused commerce patterns and natural code-switching acceptance. The CCI framework operationalizes three technical pillars: Infrastructure Intelligence (mobile-first, resilient architectures), Cultural Intelligence (multilingual NLP with social context awareness), and Commercial Intelligence (trust-based conversational commerce). This work contributes both theoretical innovation and reproducible implementation patterns, challenging Silicon Valley design orthodoxies while providing actionable frameworks for equitable AI deployment across resource-constrained markets.
>
---
#### [new 076] More than a Moment: Towards Coherent Sequences of Audio Descriptions
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视频音频描述任务，解决自动生成描述序列缺乏连贯性的问题。提出无需训练的CoherentAD方法，通过多候选生成与自回归选择，提升描述序列的连贯性与叙事完整性，并引入StoryRecall等评估指标，显著优于独立生成的方法。**

- **链接: [http://arxiv.org/pdf/2510.25440v1](http://arxiv.org/pdf/2510.25440v1)**

> **作者:** Eshika Khandelwal; Junyu Xie; Tengda Han; Max Bain; Arsha Nagrani; Andrew Zisserman; Gül Varol; Makarand Tapaswi
>
> **摘要:** Audio Descriptions (ADs) convey essential on-screen information, allowing visually impaired audiences to follow videos. To be effective, ADs must form a coherent sequence that helps listeners to visualise the unfolding scene, rather than describing isolated moments. However, most automatic methods generate each AD independently, often resulting in repetitive, incoherent descriptions. To address this, we propose a training-free method, CoherentAD, that first generates multiple candidate descriptions for each AD time interval, and then performs auto-regressive selection across the sequence to form a coherent and informative narrative. To evaluate AD sequences holistically, we introduce a sequence-level metric, StoryRecall, which measures how well the predicted ADs convey the ground truth narrative, alongside repetition metrics that capture the redundancy across consecutive AD outputs. Our method produces coherent AD sequences with enhanced narrative understanding, outperforming prior approaches that rely on independent generations.
>
---
#### [new 077] AmarDoctor: An AI-Driven, Multilingual, Voice-Interactive Digital Health Application for Primary Care Triage and Patient Management to Bridge the Digital Health Divide for Bengali Speakers
- **分类: cs.HC; cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出AmarDoctor，一款面向孟加拉语人群的多语言语音交互数字健康应用，旨在解决其在数字医疗中的服务缺失问题。通过患者与医生双端系统，结合自适应问诊与AI决策支持，实现精准分诊与个性化管理。评估显示其诊断准确率显著优于医生平均水平。**

- **链接: [http://arxiv.org/pdf/2510.24724v1](http://arxiv.org/pdf/2510.24724v1)**

> **作者:** Nazmun Nahar; Ritesh Harshad Ruparel; Shariar Kabir; Sumaiya Tasnia Khan; Shyamasree Saha; Mamunur Rashid
>
> **摘要:** This study presents AmarDoctor, a multilingual voice-interactive digital health app designed to provide comprehensive patient triage and AI-driven clinical decision support for Bengali speakers, a population largely underserved in access to digital healthcare. AmarDoctor adopts a data-driven approach to strengthen primary care delivery and enable personalized health management. While platforms such as AdaHealth, WebMD, Symptomate, and K-Health have become popular in recent years, they mainly serve European demographics and languages. AmarDoctor addresses this gap with a dual-interface system for both patients and healthcare providers, supporting three major Bengali dialects. At its core, the patient module uses an adaptive questioning algorithm to assess symptoms and guide users toward the appropriate specialist. To overcome digital literacy barriers, it integrates a voice-interactive AI assistant that navigates users through the app services. Complementing this, the clinician-facing interface incorporates AI-powered decision support that enhances workflow efficiency by generating structured provisional diagnoses and treatment recommendations. These outputs inform key services such as e-prescriptions, video consultations, and medical record management. To validate clinical accuracy, the system was evaluated against a gold-standard set of 185 clinical vignettes developed by experienced physicians. Effectiveness was further assessed by comparing AmarDoctor performance with five independent physicians using the same vignette set. Results showed AmarDoctor achieved a top-1 diagnostic precision of 81.08 percent (versus physicians average of 50.27 percent) and a top specialty recommendation precision of 91.35 percent (versus physicians average of 62.6 percent).
>
---
#### [new 078] The Epistemic Suite: A Post-Foundational Diagnostic Methodology for Assessing AI Knowledge Claims
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; I.2.7; I.2.4; I.2.6; K.4.1**

- **简介: 该论文提出“认识论工具箱”（Epistemic Suite），一种外部诊断方法，用于评估大语言模型的知识主张。针对AI生成内容看似合理实则缺乏理解的问题，通过20个诊断视角揭示其认知条件，引入“认识论悬置”等机制，实现可追溯、可问责的判断过程，强调诊断优先于真理判定，保障认知谦逊与多元共治。**

- **链接: [http://arxiv.org/pdf/2510.24721v1](http://arxiv.org/pdf/2510.24721v1)**

> **作者:** Matthew Kelly
>
> **备注:** 65 pages
>
> **摘要:** Large Language Models (LLMs) generate fluent, plausible text that can mislead users into mistaking simulated coherence for genuine understanding. This paper introduces the Epistemic Suite, a post-foundational diagnostic methodology for surfacing the epistemic conditions under which AI outputs are produced and received. Rather than determining truth or falsity, the Suite operates through twenty diagnostic lenses, applied by practitioners as context warrants, to reveal patterns such as confidence laundering, narrative compression, displaced authority, and temporal drift. It is grounded in three design principles: diagnosing production before evaluating claims, preferring diagnostic traction over foundational settlement, and embedding reflexivity as a structural requirement rather than an ethical ornament. When enacted, the Suite shifts language models into a diagnostic stance, producing inspectable artifacts-flags, annotations, contradiction maps, and suspension logs (the FACS bundle)-that create an intermediary layer between AI output and human judgment. A key innovation is epistemic suspension, a practitioner-enacted circuit breaker that halts continuation when warrant is exceeded, with resumption based on judgment rather than rule. The methodology also includes an Epistemic Triage Protocol and a Meta-Governance Layer to manage proportionality and link activation to relational accountability, consent, historical context, and pluralism safeguards. Unlike internalist approaches that embed alignment into model architectures (e.g., RLHF or epistemic-integrity proposals), the Suite operates externally as scaffolding, preserving expendability and refusal as safeguards rather than failures. It preserves the distinction between performance and understanding, enabling accountable deliberation while maintaining epistemic modesty.
>
---
#### [new 079] Topic-aware Large Language Models for Summarizing the Lived Healthcare Experiences Described in Health Stories
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的文本摘要任务，旨在通过主题感知的层次化摘要方法，从非裔美国人健康叙事中识别关键主题并生成准确、全面的总结。研究结合LDA与开源LLM，对50篇故事进行主题分析与摘要生成，并验证了摘要质量与专家评估的一致性，为医疗研究与干预提供数据支持。**

- **链接: [http://arxiv.org/pdf/2510.24765v1](http://arxiv.org/pdf/2510.24765v1)**

> **作者:** Maneesh Bilalpur; Megan Hamm; Young Ji Lee; Natasha Norman; Kathleen M. McTigue; Yanshan Wang
>
> **摘要:** Storytelling is a powerful form of communication and may provide insights into factors contributing to gaps in healthcare outcomes. To determine whether Large Language Models (LLMs) can identify potential underlying factors and avenues for intervention, we performed topic-aware hierarchical summarization of narratives from African American (AA) storytellers. Fifty transcribed stories of AA experiences were used to identify topics in their experience using the Latent Dirichlet Allocation (LDA) technique. Stories about a given topic were summarized using an open-source LLM-based hierarchical summarization approach. Topic summaries were generated by summarizing across story summaries for each story that addressed a given topic. Generated topic summaries were rated for fabrication, accuracy, comprehensiveness, and usefulness by the GPT4 model, and the model's reliability was validated against the original story summaries by two domain experts. 26 topics were identified in the fifty AA stories. The GPT4 ratings suggest that topic summaries were free from fabrication, highly accurate, comprehensive, and useful. The reliability of GPT ratings compared to expert assessments showed moderate to high agreement. Our approach identified AA experience-relevant topics such as health behaviors, interactions with medical team members, caregiving and symptom management, among others. Such insights could help researchers identify potential factors and interventions by learning from unstructured narratives in an efficient manner-leveraging the communicative power of storytelling. The use of LDA and LLMs to identify and summarize the experience of AA individuals suggests a variety of possible avenues for health research and possible clinical improvements to support patients and caregivers, thereby ultimately improving health outcomes.
>
---
#### [new 080] RAVR: Reference-Answer-guided Variational Reasoning for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG; I.2.7**

- **简介: 该论文针对大语言模型在复杂推理任务中难以生成高质量推理路径的问题，提出RAVR框架。通过引入参考答案引导变分推理，提升推理路径的期望效用，使模型能更高效学习正确推理逻辑，在数学与通用任务中均实现性能提升。**

- **链接: [http://arxiv.org/pdf/2510.25206v1](http://arxiv.org/pdf/2510.25206v1)**

> **作者:** Tianqianjin Lin; Xi Zhao; Xingyao Zhang; Rujiao Long; Yi Xu; Zhuoren Jiang; Wenbo Su; Bo Zheng
>
> **备注:** 17 pages, 11 figures
>
> **摘要:** Reinforcement learning (RL) can refine the reasoning abilities of large language models (LLMs), but critically depends on a key prerequisite: the LLM can already generate high-utility reasoning paths with non-negligible probability. For tasks beyond the LLM's current competence, such reasoning path can be hard to sample, and learning risks reinforcing familiar but suboptimal reasoning. We are motivated by the insight from cognitive science that Why is this the answer is often an easier question than What is the answer, as it avoids the heavy cognitive load of open-ended exploration, opting instead for explanatory reconstruction-systematically retracing the reasoning that links a question to its answer. We show that LLMs can similarly leverage answers to derive high-quality reasoning paths. We formalize this phenomenon and prove that conditioning on answer provably increases the expected utility of sampled reasoning paths, thereby transforming intractable problems into learnable ones. Building on this insight, we introduce RAVR (Reference-Answer-guided Variational Reasoning), an end-to-end framework that uses answer-conditioned reasoning as a variational surrogate for question-only reasoning. Experiments in both general and math domains demonstrate consistent improvements over strong baselines. We further analyze the reasoning behavior and find that RAVR reduces hesitation, strengthens conclusion consolidation, and promotes problem-specific strategies in reasoning.
>
---
#### [new 081] Hybrid Quantum-Classical Recurrent Neural Networks
- **分类: cs.LG; cs.AI; cs.CL; quant-ph**

- **简介: 该论文提出一种量子-经典混合循环神经网络（QRNN），将循环核心实现为参数化量子电路，利用量子态的指数级希尔伯特空间提升记忆容量。通过中电路测量与经典非线性控制，实现高效序列建模，在情感分析、图像识别和语言建模等任务上表现优异，首次在量子操作基础上实现媲美经典模型的性能。**

- **链接: [http://arxiv.org/pdf/2510.25557v1](http://arxiv.org/pdf/2510.25557v1)**

> **作者:** Wenduan Xu
>
> **摘要:** We present a hybrid quantum-classical recurrent neural network (QRNN) architecture in which the entire recurrent core is realized as a parametrized quantum circuit (PQC) controlled by a classical feedforward network. The hidden state is the quantum state of an $n$-qubit PQC, residing in an exponentially large Hilbert space $\mathbb{C}^{2^n}$. The PQC is unitary by construction, making the hidden-state evolution norm-preserving without external constraints. At each timestep, mid-circuit readouts are combined with the input embedding and processed by the feedforward network, which provides explicit classical nonlinearity. The outputs parametrize the PQC, which updates the hidden state via unitary dynamics. The QRNN is compact and physically consistent, and it unifies (i) unitary recurrence as a high-capacity memory, (ii) partial observation via mid-circuit measurements, and (iii) nonlinear classical control for input-conditioned parametrization. We evaluate the model in simulation with up to 14 qubits on sentiment analysis, MNIST, permuted MNIST, copying memory, and language modeling, adopting projective measurements as a limiting case to obtain mid-circuit readouts while maintaining a coherent recurrent quantum memory. We further devise a soft attention mechanism over the mid-circuit readouts in a sequence-to-sequence model and show its effectiveness for machine translation. To our knowledge, this is the first model (RNN or otherwise) grounded in quantum operations to achieve competitive performance against strong classical baselines across a broad class of sequence-learning tasks.
>
---
#### [new 082] Sequences of Logits Reveal the Low Rank Structure of Language Models
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究大语言模型的低秩结构，提出通过分析模型对不同提示的logits序列来揭示其内在低维特性。工作包括实证发现多种模型具低秩性、利用该结构进行无关提示生成响应，并建立理论抽象与学习保证，属于模型结构分析与高效生成任务。**

- **链接: [http://arxiv.org/pdf/2510.24966v1](http://arxiv.org/pdf/2510.24966v1)**

> **作者:** Noah Golowich; Allen Liu; Abhishek Shetty
>
> **摘要:** A major problem in the study of large language models is to understand their inherent low-dimensional structure. We introduce an approach to study the low-dimensional structure of language models at a model-agnostic level: as sequential probabilistic models. We first empirically demonstrate that a wide range of modern language models exhibit low-rank structure: in particular, matrices built from the model's logits for varying sets of prompts and responses have low approximate rank. We then show that this low-rank structure can be leveraged for generation -- in particular, we can generate a response to a target prompt using a linear combination of the model's outputs on unrelated, or even nonsensical prompts. On the theoretical front, we observe that studying the approximate rank of language models in the sense discussed above yields a simple universal abstraction whose theoretical predictions parallel our experiments. We then analyze the representation power of the abstraction and give provable learning guarantees.
>
---
#### [new 083] Finding Culture-Sensitive Neurons in Vision-Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究视觉语言模型中文化敏感神经元，旨在理解模型如何处理文化相关输入。通过CVQA基准识别并验证文化敏感神经元的存在，提出新方法CAS提升识别效果，并发现此类神经元集中于特定解码层，揭示了多模态表示的内部结构。**

- **链接: [http://arxiv.org/pdf/2510.24942v1](http://arxiv.org/pdf/2510.24942v1)**

> **作者:** Xiutian Zhao; Rochelle Choenni; Rohit Saxena; Ivan Titov
>
> **备注:** 22 pages, 13 figures
>
> **摘要:** Despite their impressive performance, vision-language models (VLMs) still struggle on culturally situated inputs. To understand how VLMs process culturally grounded information, we study the presence of culture-sensitive neurons, i.e. neurons whose activations show preferential sensitivity to inputs associated with particular cultural contexts. We examine whether such neurons are important for culturally diverse visual question answering and where they are located. Using the CVQA benchmark, we identify neurons of culture selectivity and perform causal tests by deactivating the neurons flagged by different identification methods. Experiments on three VLMs across 25 cultural groups demonstrate the existence of neurons whose ablation disproportionately harms performance on questions about the corresponding cultures, while having minimal effects on others. Moreover, we propose a new margin-based selector - Contrastive Activation Selection (CAS), and show that it outperforms existing probability- and entropy-based methods in identifying culture-sensitive neurons. Finally, our layer-wise analyses reveals that such neurons tend to cluster in certain decoder layers. Overall, our findings shed new light on the internal organization of multimodal representations.
>
---
#### [new 084] GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出GAP框架，解决大语言模型在复杂任务中因串行执行导致的效率低下问题。通过图结构建模子任务依赖关系，实现并行与串行工具调度，提升多步推理效率与准确率。基于MHQA数据集，采用两阶段训练策略，显著优于传统ReAct方法。**

- **链接: [http://arxiv.org/pdf/2510.25320v1](http://arxiv.org/pdf/2510.25320v1)**

> **作者:** Jiaqi Wu; Qinlao Zhao; Zefeng Chen; Kai Qin; Yifei Zhao; Xueqian Wang; Yuhang Yao
>
> **摘要:** Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: https://github.com/WJQ7777/Graph-Agent-Planning.
>
---
#### [new 085] From Medical Records to Diagnostic Dialogues: A Clinical-Grounded Approach and Dataset for Psychiatric Comorbidity
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对精神疾病共病诊断难题，提出融合合成病历与多智能体对话生成的框架，构建首个大规模共病对话数据集PsyCoTalk。通过临床规范建模对话流程，生成3000条高保真诊断对话，支持单轮多病筛查，提升诊断准确性和治疗规划能力。**

- **链接: [http://arxiv.org/pdf/2510.25232v1](http://arxiv.org/pdf/2510.25232v1)**

> **作者:** Tianxi Wan; Jiaming Luo; Siyuan Chen; Kunyao Lan; Jianhua Chen; Haiyang Geng; Mengyue Wu
>
> **摘要:** Psychiatric comorbidity is clinically significant yet challenging due to the complexity of multiple co-occurring disorders. To address this, we develop a novel approach integrating synthetic patient electronic medical record (EMR) construction and multi-agent diagnostic dialogue generation. We create 502 synthetic EMRs for common comorbid conditions using a pipeline that ensures clinical relevance and diversity. Our multi-agent framework transfers the clinical interview protocol into a hierarchical state machine and context tree, supporting over 130 diagnostic states while maintaining clinical standards. Through this rigorous process, we construct PsyCoTalk, the first large-scale dialogue dataset supporting comorbidity, containing 3,000 multi-turn diagnostic dialogues validated by psychiatrists. This dataset enhances diagnostic accuracy and treatment planning, offering a valuable resource for psychiatric comorbidity research. Compared to real-world clinical transcripts, PsyCoTalk exhibits high structural and linguistic fidelity in terms of dialogue length, token distribution, and diagnostic reasoning strategies. Licensed psychiatrists confirm the realism and diagnostic validity of the dialogues. This dataset enables the development and evaluation of models capable of multi-disorder psychiatric screening in a single conversational pass.
>
---
## 更新

#### [replaced 001] A Multilingual, Large-Scale Study of the Interplay between LLM Safeguards, Personalisation, and Disinformation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.12993v2](http://arxiv.org/pdf/2510.12993v2)**

> **作者:** João A. Leite; Arnav Arora; Silvia Gargova; João Luz; Gustavo Sampaio; Ian Roberts; Carolina Scarton; Kalina Bontcheva
>
> **摘要:** Large Language Models (LLMs) can generate human-like disinformation, yet their ability to personalise such content across languages and demographics remains underexplored. This study presents the first large-scale, multilingual analysis of persona-targeted disinformation generation by LLMs. Employing a red teaming methodology, we prompt eight state-of-the-art LLMs with 324 false narratives and 150 demographic personas (combinations of country, generation, and political orientation) across four languages--English, Russian, Portuguese, and Hindi--resulting in AI-TRAITS, a comprehensive dataset of 1.6 million personalised disinformation texts. Results show that the use of even simple personalisation prompts significantly increases the likelihood of jailbreaks across all studied LLMs, up to 10 percentage points, and alters linguistic and rhetorical patterns that enhance narrative persuasiveness. Models such as Grok and GPT exhibited jailbreak rates and personalisation scores both exceeding 85%. These insights expose critical vulnerabilities in current state-of-the-art LLMs and offer a foundation for improving safety alignment and detection strategies in multilingual and cross-demographic contexts.
>
---
#### [replaced 002] Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05493v2](http://arxiv.org/pdf/2503.05493v2)**

> **作者:** Qijiong Liu; Jieming Zhu; Lu Fan; Kun Wang; Hengchang Hu; Wei Guo; Yong Liu; Xiao-Ming Wu
>
> **备注:** NeurIPS 2025 DB Track Accepted Paper
>
> **摘要:** In recent years, integrating large language models (LLMs) into recommender systems has created new opportunities for improving recommendation quality. However, a comprehensive benchmark is needed to thoroughly evaluate and compare the recommendation capabilities of LLMs with traditional recommender systems. In this paper, we introduce RecBench, which systematically investigates various item representation forms (including unique identifier, text, semantic embedding, and semantic identifier) and evaluates two primary recommendation tasks, i.e., click-through rate prediction (CTR) and sequential recommendation (SeqRec). Our extensive experiments cover up to 17 large models and are conducted across five diverse datasets from fashion, news, video, books, and music domains. Our findings indicate that LLM-based recommenders outperform conventional recommenders, achieving up to a 5% AUC improvement in the CTR scenario and up to a 170% NDCG@10 improvement in the SeqRec scenario. However, these substantial performance gains come at the expense of significantly reduced inference efficiency, rendering the LLM-as-RS paradigm impractical for real-time recommendation environments. We aim for our findings to inspire future research, including recommendation-specific model acceleration methods. We will release our code, data, configurations, and platform to enable other researchers to reproduce and build upon our experimental results.
>
---
#### [replaced 003] Steering Information Utility in Key-Value Memory for Language Model Post-Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05158v2](http://arxiv.org/pdf/2507.05158v2)**

> **作者:** Chunyuan Deng; Ruidi Chang; Hanjie Chen
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advancements in language models (LMs) have marked a shift toward the growing importance of post-training. Yet, post-training approaches such as supervised fine-tuning (SFT) do not guarantee the effective use of knowledge acquired during pretraining. We therefore introduce InfoSteer, a lightweight method that encourages parametric information utilization in LMs during post-training. Specifically, InfoSteer treats the feed-forward network (FFN) layer as associate key-value memory and promotes the use of stored memory vectors via forward-pass interventions or regularization during backpropagation. This simple guidance during post-training phase yields consistent performance improvements across diverse model families--including Qwen, Gemma and Llama -- spanning 15 downstream tasks in both in-distribution (ID) and out-of-distribution (OOD) evaluations. Beyond performance gains, we also find that steered LMs can adaptively allocate information by placing more emphasis on generating semantically meaningful tokens, while using fewer resources on simple transition ones (e.g., `\texttt{,}' or `\texttt{and}'). Our work underscores that vanilla post-training does not fully exploit the potential gained during pre-training, and that steering LMs in latent representation space offers a promising approach to enhance both performance and interpretability. The code is available at: https://github.com/chili-lab/InfoSteer.
>
---
#### [replaced 004] ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.13852v2](http://arxiv.org/pdf/2510.13852v2)**

> **作者:** Peter Banyas; Shristi Sharma; Alistair Simmons; Atharva Vispute
>
> **备注:** For associated code repository, see http://github.com/banyasp/consistencyAI For user-friendly web app, see http://v0-llm-comparison-webapp.vercel.app/
>
> **摘要:** Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies.
>
---
#### [replaced 005] OpenReward: Learning to Reward Long-form Agentic Tasks via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24636v2](http://arxiv.org/pdf/2510.24636v2)**

> **作者:** Ziyou Hu; Zhengliang Shi; Minghang Zhu; Haitao Li; Teng Sun; Pengjie Ren; Suzan Verberne; Zhaochun Ren
>
> **摘要:** Reward models (RMs) have become essential for aligning large language models (LLMs), serving as scalable proxies for human evaluation in both training and inference. However, existing RMs struggle on knowledge-intensive and long-form tasks, where evaluating correctness requires grounding beyond the model's internal knowledge. This limitation hinders them from reliably discriminating subtle quality differences, especially when external evidence is necessary. To address this, we introduce OpenRM, a tool-augmented long-form reward model that systematically judges open-ended responses by invoking external tools to gather relevant evidence. We train OpenRM with Group Relative Policy Optimization (GRPO) on over 27K synthesized pairwise examples generated through a controllable data synthesis framework. The training objective jointly supervises intermediate tool usage and final outcome accuracy, incentivizing our reward model to learn effective evidence-based judgment strategies. Extensive experiments on three newly-collected datasets and two widely-used benchmarks demonstrate that OpenRM substantially outperforms existing reward modeling approaches. As a further step, we integrate OpenRM into both inference-time response selection and training-time data selection. This yields consistent gains in downstream LLM alignment tasks, highlighting the potential of tool-augmented reward models for scaling reliable long-form evaluation.
>
---
#### [replaced 006] When Models Outthink Their Safety: Mitigating Self-Jailbreak in Large Reasoning Models with Chain-of-Guardrails
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.21285v2](http://arxiv.org/pdf/2510.21285v2)**

> **作者:** Yingzhi Mao; Chunkang Zhang; Junxiang Wang; Xinyan Guan; Boxi Cao; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun
>
> **备注:** First two authors contributed equally. The main text is 10 pages, with an appendix of 19 pages. The paper contains 18 figures and 16 tables
>
> **摘要:** Large Reasoning Models (LRMs) demonstrate remarkable capabilities on complex reasoning tasks but remain vulnerable to severe safety risks, including harmful content generation and jailbreak attacks. Existing mitigation strategies rely on injecting heuristic safety signals during training, which often suppress reasoning ability and fail to resolve the safety-reasoning trade-off. To systematically investigate this issue, we analyze the reasoning trajectories of diverse LRMs and uncover a phenomenon we term Self-Jailbreak, where models override their own risk assessments and justify responding to unsafe prompts. This finding reveals that LRMs inherently possess the ability to reject unsafe queries, but this ability is compromised, resulting in harmful outputs. Building on these insights, we propose the Chain-of-Guardrail (CoG), a training framework that recomposes or backtracks unsafe reasoning steps, steering the model back onto safe trajectories while preserving valid reasoning chains. Extensive experiments across multiple reasoning and safety benchmarks demonstrate that CoG substantially improves the safety of current LRMs while preserving comparable reasoning ability, significantly outperforming prior methods that suffer from severe safety-reasoning trade-offs.
>
---
#### [replaced 007] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.02547v2](http://arxiv.org/pdf/2509.02547v2)**

> **作者:** Guibin Zhang; Hejia Geng; Xiaohang Yu; Zhenfei Yin; Zaibin Zhang; Zelin Tan; Heng Zhou; Zhongzhi Li; Xiangyuan Xue; Yijiang Li; Yifan Zhou; Yang Chen; Chen Zhang; Yutao Fan; Zihu Wang; Songtao Huang; Francisco Piedrahita-Velez; Yue Liao; Hongru Wang; Mengyue Yang; Heng Ji; Michael Littman; Jun Wang; Shuicheng Yan; Philip Torr; Lei Bai
>
> **摘要:** The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
>
---
#### [replaced 008] Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.15201v2](http://arxiv.org/pdf/2505.15201v2)**

> **作者:** Christian Walder; Deep Karkhanis
>
> **摘要:** Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function. While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains. We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples.
>
---
#### [replaced 009] Reinforcement Learning Teachers of Test Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08388v3](http://arxiv.org/pdf/2506.08388v3)**

> **作者:** Edoardo Cetin; Tianyu Zhao; Yujin Tang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Training reasoning language models (LMs) with reinforcement learning (RL) for one-hot correctness inherently relies on the LM being able to explore and solve its task with some chance at initialization. Furthermore, a key use case of reasoning LMs is to act as teachers for distilling new students and cold-starting future RL iterations rather than being deployed themselves. From these considerations, we introduce a new framework that avoids RL's exploration challenge by training a new class of Reinforcement-Learned Teachers (RLTs) focused on yielding the most effective downstream distillation. RLTs are prompted with both the question and solution to each problem, and tasked to simply "connect-the-dots" with detailed explanations tailored for their students. We train RLTs with dense rewards obtained by feeding each explanation to the student and testing its understanding of the problem's solution. In practice, the raw outputs of a 7B RLT provide higher final performance on competition and graduate-level tasks than existing distillation and cold-starting pipelines that collect and postprocess the reasoning traces of orders of magnitude larger LMs. Furthermore, RLTs maintain their effectiveness when training larger students and when applied zero-shot to out-of-distribution tasks, unlocking new levels of efficiency and re-usability for the RL reasoning framework. Code available at: https://github.com/SakanaAI/RLT
>
---
#### [replaced 010] Non-Markovian Discrete Diffusion with Causal Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.09767v3](http://arxiv.org/pdf/2502.09767v3)**

> **作者:** Yangtian Zhang; Sizhuang He; Daniel Levine; Lawrence Zhao; David Zhang; Syed A Rizvi; Shiyang Zhang; Emanuele Zappala; Rex Ying; David van Dijk
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Discrete diffusion models offer a flexible, controllable approach to structured sequence generation, yet they still lag behind causal language models in expressive power. A key limitation lies in their reliance on the Markovian assumption, which restricts each step to condition only on the current state, leading to potential uncorrectable error accumulation. In this paper, we introduce CaDDi (Causal Discrete Diffusion Model), a discrete diffusion model that conditions on the entire generative trajectory, thereby lifting the Markov constraint and allowing the model to revisit and improve past states. By unifying sequential (causal) and temporal (diffusion) reasoning in a single non-Markovian transformer, CaDDi also treats standard causal language models as a special case and permits the direct reuse of pretrained LLM weights with no architectural changes. Empirically, CaDDi outperforms state-of-the-art discrete diffusion baselines on natural-language benchmarks, substantially narrowing the remaining gap to large autoregressive transformers.
>
---
#### [replaced 011] Precise In-Parameter Concept Erasure in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22586v2](http://arxiv.org/pdf/2505.22586v2)**

> **作者:** Yoav Gur-Arieh; Clara Suslik; Yihuai Hong; Fazl Barez; Mor Geva
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Large language models (LLMs) often acquire knowledge during pretraining that is undesirable in downstream deployments, e.g., sensitive information or copyrighted content. Existing approaches for removing such knowledge rely on fine-tuning, training low-rank adapters or fact-level editing, but these are either too coarse, too shallow, or ineffective. In this work, we propose PISCES (Precise In-parameter Suppression for Concept EraSure), a novel framework for precisely erasing entire concepts from model parameters by directly editing directions that encode them in parameter space. PISCES uses a disentangler model to decompose MLP vectors into interpretable features, identifies those associated with a target concept using automated interpretability techniques, and removes them from model parameters. Experiments on Gemma 2 and Llama 3.1 over various concepts show that PISCES achieves modest gains in efficacy over leading erasure methods, reducing accuracy on the target concept to as low as 7.7%, while dramatically improving erasure specificity (by up to 31%) and robustness (by up to 38%). Overall, these results demonstrate that feature-based in-parameter editing enables a more precise and reliable approach for removing conceptual knowledge in language models.
>
---
#### [replaced 012] GradeSQL: Test-Time Inference with Outcome Reward Models for Text-to-SQL Generation from Large Language Models
- **分类: cs.AI; cs.CL; cs.DB**

- **链接: [http://arxiv.org/pdf/2509.01308v2](http://arxiv.org/pdf/2509.01308v2)**

> **作者:** Mattia Tritto; Giuseppe Farano; Dario Di Palma; Gaetano Rossiello; Fedelucio Narducci; Dharmashankar Subramanian; Tommaso Di Noia
>
> **摘要:** Text-to-SQL, the task of translating natural language questions into SQL queries, has significantly advanced with the introduction of Large Language Models (LLMs), broadening database accessibility for a wide range of users. Despite substantial progress in generating valid SQL, current LLMs still struggle with complex queries. To address this limitation, test-time strategies such as Best-of-N (BoN) and Majority Voting (Maj) are often employed, based on the assumption that LLMs can produce correct answers after multiple attempts. However, these methods rely on surface-level heuristics, selecting the syntactically correct query through execution-based BoN (ex-BoN) or the most frequently generated one through Majority Voting. Recently, Outcome Reward Models (ORMs), which assign utility scores to generated outputs based on semantic correctness, have emerged as a promising reinforcement learning approach for improving model alignment. We argue that ORMs could serve as an effective new test-time heuristic, although their application in this context remains largely underexplored. In this work, we propose a unified framework for training ORMs tailored to the Text-to-SQL task and assess their effectiveness as a test-time heuristic within the BoN strategy. We benchmark ORMs against ex-BoN and Maj across the BIRD and Spider datasets, fine-tuning diverse open-source LLMs from the Qwen2, Granite3, and Llama3 families. Results show that ORMs outperform ex-BoN and Maj, achieving execution accuracy gains of +4.33% (BIRD) and +2.10% (Spider) over ex-BoN, and +2.91% (BIRD) and +0.93% (Spider) over Maj. We further demonstrate that finetuning models already aligned with SQL generation, such as OmniSQL, yields superior ORM performance. Additionally, we observe that ORMs achieve competitive results on simple queries and benefit more from an increased number of candidates compared to ex-BoN and Maj.
>
---
#### [replaced 013] DBLPLink 2.0 -- An Entity Linker for the DBLP Scholarly Knowledge Graph
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22811v2](http://arxiv.org/pdf/2507.22811v2)**

> **作者:** Debayan Banerjee; Tilahun Abedissa Taffa; Ricardo Usbeck
>
> **摘要:** In this work we present an entity linker for DBLP's 2025 version of RDF-based Knowledge Graph. Compared to the 2022 version, DBLP now considers publication venues as a new entity type called dblp:Stream. In the earlier version of DBLPLink, we trained KG-embeddings and re-rankers on a dataset to produce entity linkings. In contrast, in this work, we develop a zero-shot entity linker using LLMs using a novel method, where we re-rank candidate entities based on the log-probabilities of the "yes" token output at the penultimate layer of the LLM.
>
---
#### [replaced 014] Quantifying Phonosemantic Iconicity Distributionally in 6 Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.14040v2](http://arxiv.org/pdf/2510.14040v2)**

> **作者:** George Flint; Kaustubh Kislay
>
> **备注:** IJCNLP-AACL 2025 Main Conference Proceedings
>
> **摘要:** Language is, as commonly theorized, largely arbitrary. Yet, systematic relationships between phonetics and semantics have been observed in many specific cases. To what degree could those systematic relationships manifest themselves in large scale, quantitative investigations--both in previously identified and unidentified phenomena? This work undertakes a distributional approach to quantifying phonosemantic iconicity at scale across 6 diverse languages (English, Spanish, Hindi, Finnish, Turkish, and Tamil). In each language, we analyze the alignment of morphemes' phonetic and semantic similarity spaces with a suite of statistical measures, and discover an array of interpretable phonosemantic alignments not previously identified in the literature, along with crosslinguistic patterns. We also analyze 5 previously hypothesized phonosemantic alignments, finding support for some such alignments and mixed results for others.
>
---
#### [replaced 015] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2405.17220v3](http://arxiv.org/pdf/2405.17220v3)**

> **作者:** Tianyu Yu; Haoye Zhang; Qiming Li; Qixin Xu; Yuan Yao; Da Chen; Xiaoman Lu; Ganqu Cui; Yunkai Dang; Taiwen He; Xiaocheng Feng; Jun Song; Bo Zheng; Zhiyuan Liu; Tat-Seng Chua; Maosong Sun
>
> **备注:** Project Website: https://github.com/RLHF-V/RLAIF-V
>
> **摘要:** Traditional feedback learning for hallucination reduction relies on labor-intensive manual labeling or expensive proprietary models. This leaves the community without foundational knowledge about how to build high-quality feedback with open-source MLLMs. In this work, we introduce RLAIF-V, a novel framework that aligns MLLMs in a fully open-source paradigm. RLAIF-V maximally explores open-source MLLMs from two perspectives, including high-quality feedback data generation for preference learning and self-feedback guidance for inference-time scaling. Extensive experiments on six benchmarks in both automatic and human evaluation show that RLAIF-V substantially enhances the trustworthiness of models at both preference learning and inference time. RLAIF-V 7B reduces object hallucination by 80.7\% and overall hallucination by 33.7\%. Remarkably, RLAIF-V 12B further reveals the self-alignment potential of open-source MLLMs, where the model can learn from feedback of itself to achieve super GPT-4V trustworthiness.
>
---
#### [replaced 016] Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17585v2](http://arxiv.org/pdf/2506.17585v2)**

> **作者:** Yukun Huang; Sanxing Chen; Jian Pei; Manzil Zaheer; Bhuwan Dhingra
>
> **摘要:** Trustworthy language models should provide both correct and verifiable answers. However, citations generated directly by standalone LLMs are often unreliable. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during continual pretraining without test-time retrieval, by revising the training process. To study this, we construct CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel documents and probes both short-form (single-fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to index factual knowledge by binding it to persistent document identifiers; and (2) instruction tuning to elicit citation behavior. We introduce Active Indexing for the first stage, which creates generalizable, source-anchored bindings by augmenting training with synthetic data that (i) restate each fact in diverse, compositional forms and (ii) enforce bidirectional training (source-to-fact and fact-to-source). This equips the model to both generate content from a cited source and attribute its own answers, improving robustness to paraphrase and composition. Experiments with Qwen-2.5-7B&3B show that Active Indexing consistently outperforms a Passive Indexing baseline, which simply appends an identifier to each document, achieving citation precision gains of up to 30.2% across all tasks and models. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16x the original token count. Finally, we show that internal citations complement external ones by making the model more robust to retrieval noise.
>
---
#### [replaced 017] Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.17937v3](http://arxiv.org/pdf/2507.17937v3)**

> **作者:** Jaechul Roh; Zachary Novack; Yuefeng Peng; Niloofar Mireshghallah; Taylor Berg-Kirkpatrick; Amir Houmansadr
>
> **摘要:** Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).
>
---
#### [replaced 018] Reliable Evaluation and Benchmarks for Statement Autoformalization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.07222v3](http://arxiv.org/pdf/2406.07222v3)**

> **作者:** Auguste Poiroux; Gail Weiss; Viktor Kunčak; Antoine Bosselut
>
> **备注:** Accepted to EMNLP 2025. New benchmarks released, see https://github.com/augustepoiroux/RLMEval , https://huggingface.co/datasets/PAug/ProofNetSharp , and https://huggingface.co/datasets/PAug/ProofNetVerif . For code, see https://github.com/augustepoiroux/LeanInteract
>
> **摘要:** Evaluating statement autoformalization, translating natural language mathematics into formal languages like Lean 4, remains a significant challenge, with few metrics, datasets, and standards to robustly measure progress. In this work, we present a comprehensive approach combining improved metrics, robust benchmarks, and systematic evaluation, to fill this gap. First, we introduce BEq+, an automated metric that correlates strongly with human judgment, along with ProofNetVerif, a new dataset for assessing the quality of evaluation metrics, containing 3,752 annotated examples. Second, we develop two new autoformalization benchmarks: ProofNet#, a corrected version of ProofNet, and RLM25, with 619 new pairs of research-level mathematics from six formalization projects. Through systematic experimentation across these benchmarks, we find that current techniques can achieve up to 45.1% accuracy on undergraduate mathematics but struggle with research-level content without proper context. Our work establishes a reliable foundation for evaluating and advancing autoformalization systems.
>
---
#### [replaced 019] Blind Spot Navigation in Large Language Model Reasoning with Thought Space Explorer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.24155v4](http://arxiv.org/pdf/2410.24155v4)**

> **作者:** Jinghan Zhang; Fengran Mo; Tharindu Cyril Weerasooriya; Xinyue Ye; Dongjie Wang; Yanjie Fu; Kunpeng Liu
>
> **摘要:** Large language models have shown strong reasoning capabilities through chain-structured methods such as Chain-of-Thought. Recent studies optimize thought structures by generating parallel or tree-like structures, switching between long and short reasoning modes, or aligning reasoning steps with task performance. However, these approaches mainly rely on previously generated logical directions of the chains, which ignore the unexplored regions of the solution space. Such a phenomenon is defined as blind spots, which limit the diversity and effectiveness of the reasoning process. To this end, we propose the ``Thought Space Explorer'' (TSE), a framework for navigating and expanding thought structures to overcome blind spots in LLM reasoning. Our TSE first identifies key nodes with high impact, then generates new nodes by integrating information from multiple chains. Finally, it extends new branches through connection strategies. We conduct a series of experiments on math and QA benchmarks. Compared with existing baseline methods, TSE improves the accuracy of both the final answer and intermediate reasoning steps, while maintaining a better effectiveness-efficiency trade-off for practical deployment.
>
---
#### [replaced 020] Face the Facts! Evaluating RAG-based Pipelines for Professional Fact-Checking
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2412.15189v3](http://arxiv.org/pdf/2412.15189v3)**

> **作者:** Daniel Russo; Stefano Menini; Jacopo Staiano; Marco Guerini
>
> **备注:** Code and data at https://github.com/drusso98/face-the-facts - Accepted for publication at INLG 2025
>
> **摘要:** Natural Language Processing and Generation systems have recently shown the potential to complement and streamline the costly and time-consuming job of professional fact-checkers. In this work, we lift several constraints of current state-of-the-art pipelines for automated fact-checking based on the Retrieval-Augmented Generation (RAG) paradigm. Our goal is to benchmark, following professional fact-checking practices, RAG-based methods for the generation of verdicts - i.e., short texts discussing the veracity of a claim - evaluating them on stylistically complex claims and heterogeneous, yet reliable, knowledge bases. Our findings show a complex landscape, where, for example, LLM-based retrievers outperform other retrieval techniques, though they still struggle with heterogeneous knowledge bases; larger models excel in verdict faithfulness, while smaller models provide better context adherence, with human evaluations favouring zero-shot and one-shot approaches for informativeness, and fine-tuned models for emotional alignment.
>
---
#### [replaced 021] OpenFactCheck: A Unified Framework for Factuality Evaluation of LLMs
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2408.11832v3](http://arxiv.org/pdf/2408.11832v3)**

> **作者:** Hasan Iqbal; Yuxia Wang; Minghan Wang; Georgi Georgiev; Jiahui Geng; Iryna Gurevych; Preslav Nakov
>
> **备注:** 11 pages, 4 Figures, 3 Tables, Published In Proceedings of The 2024 Conference on Empirical Methods in Natural Language Processing
>
> **摘要:** The increased use of large language models (LLMs) across a variety of real-world applications calls for automatic tools to check the factual accuracy of their outputs, as LLMs often hallucinate. This is difficult as it requires assessing the factuality of free-form open-domain responses. While there has been a lot of research on this topic, different papers use different evaluation benchmarks and measures, which makes them hard to compare and hampers future progress. To mitigate these issues, we developed OpenFactCheck, a unified framework, with three modules: (i) RESPONSEEVAL, which allows users to easily customize an automatic fact-checking system and to assess the factuality of all claims in an input document using that system, (ii) LLMEVAL, which assesses the overall factuality of an LLM, and (iii) CHECKEREVAL, a module to evaluate automatic fact-checking systems. OpenFactCheck is open-sourced (https://github.com/mbzuai-nlp/openfactcheck) and publicly released as a Python library (https://pypi.org/project/openfactcheck/) and also as a web service (http://app.openfactcheck.com). A video describing the system is available at https://youtu.be/-i9VKL0HleI.
>
---
#### [replaced 022] SciReasoner: Laying the Scientific Reasoning Ground Across Disciplines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.21320v2](http://arxiv.org/pdf/2509.21320v2)**

> **作者:** Yizhou Wang; Chen Tang; Han Deng; Jiabei Xiao; Jiaqi Liu; Jianyu Wu; Jun Yao; Pengze Li; Encheng Su; Lintao Wang; Guohang Zhuang; Yuchen Ren; Ben Fei; Ming Hu; Xin Chen; Dongzhan Zhou; Junjun He; Xiangyu Yue; Zhenfei Yin; Jiamin Wu; Qihao Zheng; Yuhao Zhou; Huihui Xu; Chenglong Ma; Yan Lu; Wenlong Zhang; Chunfeng Song; Philip Torr; Shixiang Tang; Xinzhu Ma; Wanli Ouyang; Lei Bai
>
> **备注:** technical report
>
> **摘要:** We present a scientific reasoning foundation model that aligns natural language with heterogeneous scientific representations. The model is pretrained on a 206B-token corpus spanning scientific text, pure sequences, and sequence-text pairs, then aligned via SFT on 40M instructions, annealed cold-start bootstrapping to elicit long-form chain-of-thought, and reinforcement learning with task-specific reward shaping, which instills deliberate scientific reasoning. It supports four capability families, covering up to 103 tasks across workflows: (i) faithful translation between text and scientific formats, (ii) text/knowledge extraction, (iii) property prediction, (iv) property classification, (v) unconditional and conditional sequence generation and design. Compared with specialist systems, our approach broadens instruction coverage, improves cross-domain generalization, and enhances fidelity. We detail data curation and training and show that cross-discipline learning strengthens transfer and downstream reliability. The model, instruct tuning datasets and the evaluation code are open-sourced at https://huggingface.co/SciReason and https://github.com/open-sciencelab/SciReason.
>
---
#### [replaced 023] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.00568v2](http://arxiv.org/pdf/2510.00568v2)**

> **作者:** Shiyu Li; Yang Tang; Yifan Wang; Peiming Li; Xi Chen
>
> **备注:** 19 pages
>
> **摘要:** Search agents powered by Large Language Models (LLMs) have demonstrated significant potential in tackling knowledge-intensive tasks. Reinforcement learning (RL) has emerged as a powerful paradigm for training these agents to perform complex, multi-step reasoning. However, prior RL-based methods often rely on sparse or rule-based rewards, which can lead agents to commit to suboptimal or erroneous reasoning paths without the ability to recover. To address these limitations, we propose ReSeek, a novel self-correcting framework for training search agents. Our framework introduces a self-correction mechanism that empowers the agent to dynamically identify and recover from erroneous search paths during an episode. By invoking a special JUDGE action, the agent can judge the information and re-plan its search strategy. To guide this process, we design a dense, instructive process reward function, which decomposes into a correctness reward for retrieving factual information and a utility reward for finding information genuinely useful for the query. Furthermore, to mitigate the risk of data contamination in existing datasets, we introduce FictionalHot, a new and challenging benchmark with recently curated questions requiring complex reasoning. Being intuitively reasonable and practically simple, extensive experiments show that agents trained with ReSeek significantly outperform SOTA baselines in task success rate and path faithfulness.
>
---
#### [replaced 024] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23763v2](http://arxiv.org/pdf/2510.23763v2)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [replaced 025] BugPilot: Complex Bug Generation for Efficient Learning of SWE Skills
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19898v2](http://arxiv.org/pdf/2510.19898v2)**

> **作者:** Atharv Sonwane; Isadora White; Hyunji Lee; Matheus Pereira; Lucas Caccia; Minseon Kim; Zhengyan Shi; Chinmay Singh; Alessandro Sordoni; Marc-Alexandre Côté; Xingdi Yuan
>
> **摘要:** High quality bugs are key to training the next generation of language model based software engineering (SWE) agents. We introduce a novel method for synthetic generation of difficult and diverse bugs. Our method instructs SWE Agents to introduce a feature into the codebase whereby they may unintentionally break tests, resulting in bugs. Prior approaches often induce an out-of-distribution effect by generating bugs intentionally (e.g. by introducing local perturbation to existing code), which does not reflect realistic development processes. We perform qualitative analysis to demonstrate that our approach for generating bugs more closely reflects the patterns found in human-authored edits. Through extensive experiments, we demonstrate that our bugs provide more efficient training data for supervised fine-tuning, outperforming other bug datasets by 2% with half the training data (1.2k vs. 3k bugs). We train on our newly generated bugs in addition to existing bug datasets to get FrogBoss a state-of-the-art 32B parameter model on SWE-bench Verified with a pass@1 of 54.6% and FrogMini a state-of-the-art 14B model on SWE-bench Verified with a pass@1 of 45.3% on SWE-bench Verified averaged over three seeds.
>
---
#### [replaced 026] Do predictability factors towards signing avatars hold across cultures?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2307.02103v3](http://arxiv.org/pdf/2307.02103v3)**

> **作者:** Abdelhadi Soudi; Manal El Hakkaoui; Kristof Van Laerhoven
>
> **备注:** In Proceedings of SLTAT 2023: Eighth International Workshop on Sign Language Translation and Avatar Technology, held in conjunction with ICASSP 2023: IEEE International Conference on Acoustics, Speech, and Signal Processing, Rhodes, Greece, June 4-10, 2023
>
> **摘要:** Avatar technology can offer accessibility possibilities and improve the Deaf-and-Hard of Hearing sign language users access to communication, education and services, such as the healthcare system. However, sign language users acceptance of signing avatars as well as their attitudes towards them vary and depend on many factors. Furthermore, research on avatar technology is mostly done by researchers who are not Deaf. The study examines the extent to which intrinsic or extrinsic factors contribute to predict the attitude towards avatars across cultures. Intrinsic factors include the characteristics of the avatar, such as appearance, movements and facial expressions. Extrinsic factors include users technology experience, their hearing status, age and their sign language fluency. This work attempts to answer questions such as, if lower attitude ratings are related to poor technology experience with ASL users, for example, is that also true for Moroccan Sign Language (MSL) users? For the purposes of the study, we designed a questionnaire to understand MSL users attitude towards avatars. Three groups of participants were surveyed: Deaf (57), Hearing (20) and Hard-of-Hearing (3). The results of our study were then compared with those reported in other relevant studies.
>
---
#### [replaced 027] AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.01617v3](http://arxiv.org/pdf/2510.01617v3)**

> **作者:** Hui Yi Leong; Yuheng Li; Yuqing Wu; Wenwen Ouyang; Wei Zhu; Jiechao Gao; Wei Han
>
> **备注:** Accepted by EMNLP-2025
>
> **摘要:** Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
>
---
#### [replaced 028] Differential Mamba
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.06204v2](http://arxiv.org/pdf/2507.06204v2)**

> **作者:** Nadav Schneider; Itamar Zimerman; Eliya Nachmani
>
> **备注:** AACL 2025. We provide the code at https://github.com/NadavSc/Diff-Mamba
>
> **摘要:** Sequence models like Transformers and RNNs often overallocate attention to irrelevant context, leading to noisy intermediate representations. This degrades LLM capabilities by promoting hallucinations, weakening long-range and retrieval abilities, and reducing robustness. Recent work has shown that differential design can mitigate this issue in Transformers, improving their effectiveness across various applications. In this paper, we explore whether these techniques, originally developed for Transformers, can be applied to Mamba, a recent architecture based on selective state-space layers that achieves Transformer-level performance with greater efficiency. We show that a naive adaptation of differential design to Mamba is insufficient and requires careful architectural modifications. To address this, we introduce a novel differential mechanism for Mamba, empirically validated on language modeling benchmarks, demonstrating improved retrieval capabilities and superior performance over vanilla Mamba. Finally, we conduct extensive ablation studies and empirical analyses to justify our design choices and provide evidence that our approach effectively mitigates the overallocation problem in Mamba-based models. Our code is publicly available: https://github.com/NadavSc/Diff-Mamba
>
---
#### [replaced 029] MMD-Flagger: Leveraging Maximum Mean Discrepancy to Detect Hallucinations
- **分类: cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.01367v3](http://arxiv.org/pdf/2506.01367v3)**

> **作者:** Kensuke Mitsuzawa; Damien Garreau
>
> **摘要:** Large language models (LLMs) have become pervasive in our everyday life. Yet, a fundamental obstacle prevents their use in many critical applications: their propensity to generate fluent, human-quality content that is not grounded in reality. The detection of such hallucinations is thus of the highest importance. In this work, we propose a new method to flag hallucinated content: MMD-Flagger. It relies on Maximum Mean Discrepancy (MMD), a non-parametric distance between distributions. On a high-level perspective, MMD-Flagger tracks the MMD between the output to inspect and counterparts generated with various temperature parameters. We show empirically that inspecting the shape of this trajectory is sufficient to detect most hallucinations. This novel method is benchmarked on machine translation and summarization datasets, on which it exhibits competitive performance relative to natural competitors.
>
---
#### [replaced 030] DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.14205v3](http://arxiv.org/pdf/2510.14205v3)**

> **作者:** Bingsheng Yao; Bo Sun; Yuanzhe Dong; Yuxuan Lu; Dakuo Wang
>
> **备注:** In Submission
>
> **摘要:** The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF). DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences. We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews. DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios. Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
>
---
#### [replaced 031] Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12484v4](http://arxiv.org/pdf/2506.12484v4)**

> **作者:** Filip Sondej; Yushi Yang; Mikołaj Kniejski; Marcel Windys
>
> **摘要:** Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning. We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive. Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40%, setting a new state-of-the-art for robust unlearning.
>
---
#### [replaced 032] Spontaneous Giving and Calculated Greed in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.17720v4](http://arxiv.org/pdf/2502.17720v4)**

> **作者:** Yuxuan Li; Hirokazu Shirado
>
> **备注:** Accepted to EMNLP 2025 main conference and selected as an Oral Presentation
>
> **摘要:** Large language models demonstrate strong problem-solving abilities through reasoning techniques such as chain-of-thought prompting and reflection. However, it remains unclear whether these reasoning capabilities extend to a form of social intelligence: making effective decisions in cooperative contexts. We examine this question using economic games that simulate social dilemmas. First, we apply chain-of-thought and reflection prompting to GPT-4o in a Public Goods Game. We then evaluate multiple off-the-shelf models across six cooperation and punishment games, comparing those with and without explicit reasoning mechanisms. We find that reasoning models consistently reduce cooperation and norm enforcement, favoring individual rationality. In repeated interactions, groups with more reasoning agents exhibit lower collective gains. These behaviors mirror human patterns of "spontaneous giving and calculated greed." Our findings underscore the need for LLM architectures that incorporate social intelligence alongside reasoning, to help address--rather than reinforce--the challenges of collective action.
>
---
#### [replaced 033] OpenFactCheck: Building, Benchmarking Customized Fact-Checking Systems and Evaluating the Factuality of Claims and LLMs
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2405.05583v3](http://arxiv.org/pdf/2405.05583v3)**

> **作者:** Yuxia Wang; Minghan Wang; Hasan Iqbal; Georgi Georgiev; Jiahui Geng; Preslav Nakov
>
> **备注:** 23 pages, 8 tables, 11 figures, Published In Proceedings of the 31st International Conference on Computational Linguistics 2025
>
> **摘要:** The increased use of large language models (LLMs) across a variety of real-world applications calls for mechanisms to verify the factual accuracy of their outputs. Difficulties lie in assessing the factuality of free-form responses in open domains. Also, different papers use disparate evaluation benchmarks and measurements, which renders them hard to compare and hampers future progress. To mitigate these issues, we propose OpenFactCheck, a unified framework for building customized automatic fact-checking systems, benchmarking their accuracy, evaluating factuality of LLMs, and verifying claims in a document. OpenFactCheck consists of three modules: (i) CUSTCHECKER allows users to easily customize an automatic fact-checker and verify the factual correctness of documents and claims, (ii) LLMEVAL, a unified evaluation framework assesses LLM's factuality ability from various perspectives fairly, and (iii) CHECKEREVAL is an extensible solution for gauging the reliability of automatic fact-checkers' verification results using human-annotated datasets. Data and code are publicly available at https://github.com/yuxiaw/openfactcheck.
>
---
#### [replaced 034] Adapter-state Sharing CLIP for Parameter-efficient Multimodal Sarcasm Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04508v2](http://arxiv.org/pdf/2507.04508v2)**

> **作者:** Soumyadeep Jana; Sahil Danayak; Sanasam Ranbir Singh
>
> **摘要:** The growing prevalence of multimodal image-text sarcasm on social media poses challenges for opinion mining systems. Existing approaches rely on full fine-tuning of large models, making them unsuitable to adapt under resource-constrained settings. While recent parameter-efficient fine-tuning (PEFT) methods offer promise, their off-the-shelf use underperforms on complex tasks like sarcasm detection. We propose AdS-CLIP (Adapter-state Sharing in CLIP), a lightweight framework built on CLIP that inserts adapters only in the upper layers to preserve low-level unimodal representations in the lower layers and introduces a novel adapter-state sharing mechanism, where textual adapters guide visual ones to promote efficient cross-modal learning in the upper layers. Experiments on two public benchmarks demonstrate that AdS-CLIP not only outperforms standard PEFT methods but also existing multimodal baselines with significantly fewer trainable parameters.
>
---
#### [replaced 035] CURATRON: Complete and Robust Preference Data for Rigorous Alignment of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.02745v3](http://arxiv.org/pdf/2403.02745v3)**

> **作者:** Son The Nguyen; Niranjan Uma Naresh; Theja Tulabandhula
>
> **摘要:** This paper addresses the challenges of aligning large language models (LLMs) with human values via preference learning (PL), focusing on incomplete and corrupted data in preference datasets. We propose a novel method for robustly and completely recalibrating values within these datasets to enhance LLMs' resilience against the issues. In particular, we devise a guaranteed polynomial time ranking algorithm that robustifies several existing models, such as the classic Bradley-Terry-Luce (BTL) (Bradley and Terry, 1952) model and certain generalizations of it. To the best of our knowledge, our present work is the first to propose an algorithm that provably recovers an $\epsilon$-optimal ranking with high probability while allowing as large as $O(n)$ perturbed pairwise comparison results per model response. Furthermore, we show robust recovery results in the partially observed setting. Our experiments confirm that our algorithms handle adversarial noise and unobserved comparisons well in both general and LLM preference dataset settings. This work contributes to the development and scaling of more reliable and ethically aligned AI models by equipping the dataset curation pipeline with the ability to handle missing and maliciously manipulated inputs.
>
---
#### [replaced 036] UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.15063v2](http://arxiv.org/pdf/2505.15063v2)**

> **作者:** Sarfraz Ahmad; Hasan Iqbal; Momina Ahsan; Numaan Naeem; Muhammad Ahsan Riaz Khan; Arham Riaz; Muhammad Arslan Manzoor; Yuxia Wang; Preslav Nakov
>
> **备注:** 15 pages, 4 figures, 5 tables, 6 Listings, Published in Proceeding of The 2025 Conference on Empirical Methods in Natural Language Processing
>
> **摘要:** The rapid adoption of Large Language Models (LLMs) has raised important concerns about the factual reliability of their outputs, particularly in low-resource languages such as Urdu. Existing automated fact-checking systems are predominantly developed for English, leaving a significant gap for the more than 200 million Urdu speakers worldwide. In this work, we present UrduFactBench and UrduFactQA, two novel hand-annotated benchmarks designed to enable fact-checking and factual consistency evaluation in Urdu. While UrduFactBench focuses on claim verification, UrduFactQA targets the factuality of LLMs in question answering. These resources, the first of their kind for Urdu, were developed through a multi-stage annotation process involving native Urdu speakers. To complement these benchmarks, we introduce UrduFactCheck, a modular fact-checking framework that incorporates both monolingual and translation-based evidence retrieval strategies to mitigate the scarcity of high-quality Urdu evidence. Leveraging these resources, we conduct an extensive evaluation of twelve LLMs and demonstrate that translation-augmented pipelines consistently enhance performance compared to monolingual ones. Our findings reveal persistent challenges for open-source LLMs in Urdu and underscore the importance of developing targeted resources. All code and data are publicly available at https://github.com/mbzuai-nlp/UrduFactCheck.
>
---
#### [replaced 037] Think Twice Before You Judge: Mixture of Dual Reasoning Experts for Multimodal Sarcasm Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.04458v2](http://arxiv.org/pdf/2507.04458v2)**

> **作者:** Soumyadeep Jana; Abhrajyoti Kundu; Sanasam Ranbir Singh
>
> **摘要:** Multimodal sarcasm detection has attracted growing interest due to the rise of multimedia posts on social media. Understanding sarcastic image-text posts often requires external contextual knowledge, such as cultural references or commonsense reasoning. However, existing models struggle to capture the deeper rationale behind sarcasm, relying mainly on shallow cues like image captions or object-attribute pairs from images. To address this, we propose \textbf{MiDRE} (\textbf{Mi}xture of \textbf{D}ual \textbf{R}easoning \textbf{E}xperts), which integrates an internal reasoning expert for detecting incongruities within the image-text pair and an external reasoning expert that utilizes structured rationales generated via Chain-of-Thought prompting to a Large Vision-Language Model. An adaptive gating mechanism dynamically weighs the two experts, selecting the most relevant reasoning path. Unlike prior methods that treat external knowledge as static input, MiDRE selectively adapts to when such knowledge is beneficial, mitigating the risks of hallucinated or irrelevant signals from large models. Experiments on two benchmark datasets show that MiDRE achieves superior performance over baselines. Various qualitative analyses highlight the crucial role of external rationales, revealing that even when they are occasionally noisy, they provide valuable cues that guide the model toward a better understanding of sarcasm.
>
---
#### [replaced 038] MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.22967v2](http://arxiv.org/pdf/2510.22967v2)**

> **作者:** Yucheng Ning; Xixun Lin; Fang Fang; Yanan Cao
>
> **备注:** The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-025-51369-x}
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.
>
---
#### [replaced 039] LLMs are Better Than You Think: Label-Guided In-Context Learning for Named Entity Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23722v2](http://arxiv.org/pdf/2505.23722v2)**

> **作者:** Fan Bai; Hamid Hassanzadeh; Ardavan Saeedi; Mark Dredze
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** In-context learning (ICL) enables large language models (LLMs) to perform new tasks using only a few demonstrations. However, in Named Entity Recognition (NER), existing ICL methods typically rely on task-agnostic semantic similarity for demonstration retrieval, which often yields less relevant examples and leads to inferior results. We introduce DEER, a training-free ICL approach that enables LLMs to make more informed entity predictions through the use of label-grounded statistics. DEER leverages token-level statistics from training labels to identify tokens most informative for entity recognition, enabling entity-focused demonstrations. It further uses these statistics to detect and refine error-prone tokens through a targeted reflection step. Evaluated on five NER datasets across four LLMs, DEER consistently outperforms existing ICL methods and achieves performance comparable to supervised fine-tuning. Further analyses demonstrate that DEER improves example retrieval, remains effective on both seen and unseen entities, and exhibits strong robustness in low-resource settings.
>
---
#### [replaced 040] Consistency of Responses and Continuations Generated by Large Language Models on Social Media
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.08102v5](http://arxiv.org/pdf/2501.08102v5)**

> **作者:** Wenlu Fan; Yuqi Zhu; Bin Wang; Wentao Xu
>
> **备注:** This paper has been accepted by the International AAAI Conference on Web and Social Media (ICWSM) 2026 (Los Angeles, California, U.S.)
>
> **摘要:** Large Language Models (LLMs) demonstrate remarkable capabilities in text generation, yet their emotional consistency and semantic coherence in social media contexts remain insufficiently understood. This study investigates how LLMs handle emotional content and maintain semantic relationships through continuation and response tasks using three open-source models: Gemma, Llama3 and Llama3.3 and one commercial Model:Claude. By analyzing climate change discussions from Twitter and Reddit, we examine emotional transitions, intensity patterns, and semantic consistency between human-authored and LLM-generated content. Our findings reveal that while both models maintain high semantic coherence, they exhibit distinct emotional patterns: these models show a strong tendency to moderate negative emotions. When the input text carries negative emotions such as anger, disgust, fear, or sadness, LLM tends to generate content with more neutral emotions, or even convert them into positive emotions such as joy or surprise. At the same time, we compared the LLM-generated content with human-authored content. The four models systematically generated responses with reduced emotional intensity and showed a preference for neutral rational emotions in the response task. In addition, these models all maintained a high semantic similarity with the original text, although their performance in the continuation task and the response task was different. These findings provide deep insights into the emotion and semantic processing capabilities of LLM, which are of great significance for its deployment in social media environments and human-computer interaction design.
>
---
#### [replaced 041] InsurTech innovation using natural language processing
- **分类: cs.CL; cs.LG; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.21112v2](http://arxiv.org/pdf/2507.21112v2)**

> **作者:** Panyi Dong; Zhiyu Quan
>
> **摘要:** With the rapid rise of InsurTech, traditional insurance companies are increasingly exploring alternative data sources and advanced technologies to sustain their competitive edge. This paper provides both a conceptual overview and practical case studies of natural language processing (NLP) and its emerging applications within insurance operations, focusing on transforming raw, unstructured text into structured data suitable for actuarial analysis and decision-making. Leveraging real-world alternative data provided by an InsurTech industry partner that enriches traditional insurance data sources, we apply various NLP techniques to demonstrate feature de-biasing, feature compression, and industry classification in the commercial insurance context. These enriched, text-derived insights not only add to and refine traditional rating factors for commercial insurance pricing but also offer novel perspectives for assessing underlying risk by introducing novel industry classification techniques. Through these demonstrations, we show that NLP is not merely a supplementary tool but a foundational element of modern, data-driven insurance analytics.
>
---
#### [replaced 042] PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17818v2](http://arxiv.org/pdf/2505.17818v2)**

> **作者:** Daeun Kyung; Hyunseung Chung; Seongsu Bae; Jiho Kim; Jae Ho Sohn; Taerim Kim; Soo Kyung Kim; Edward Choi
>
> **备注:** Accepted as a Spotlight at NeurIPS 2025 Datasets and Benchmarks Track (10 pages for main text, 4 pages for references, 36 pages for supplementary materials)
>
> **摘要:** Doctor-patient consultations require multi-turn, context-aware communication tailored to diverse patient personas. Training or evaluating doctor LLMs in such settings requires realistic patient interaction systems. However, existing simulators often fail to reflect the full range of personas seen in clinical practice. To address this, we introduce PatientSim, a patient simulator that generates realistic and diverse patient personas for clinical scenarios, grounded in medical expertise. PatientSim operates using: 1) clinical profiles, including symptoms and medical history, derived from real-world data in the MIMIC-ED and MIMIC-IV datasets, and 2) personas defined by four axes: personality, language proficiency, medical history recall level, and cognitive confusion level, resulting in 37 unique combinations. We evaluate eight LLMs for factual accuracy and persona consistency. The top-performing open-source model, Llama 3.3 70B, is validated by four clinicians to confirm the robustness of our framework. As an open-source, customizable platform, PatientSim provides a reproducible and scalable solution that can be customized for specific training needs. Offering a privacy-compliant environment, it serves as a robust testbed for evaluating medical dialogue systems across diverse patient presentations and shows promise as an educational tool for healthcare. The code is available at https://github.com/dek924/PatientSim.
>
---
#### [replaced 043] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2509.01200v2](http://arxiv.org/pdf/2509.01200v2)**

> **作者:** Chenyang Le; Bing Han; Jinshun Li; Songyong Chen; Yanmin Qian
>
> **备注:** NeurIPS 2025 poster
>
> **摘要:** Simultaneous Speech Translation (SimulST) enables real-time cross-lingual communication by jointly optimizing speech recognition and machine translation under strict latency constraints. Existing systems struggle to balance translation quality, latency, and semantic coherence, particularly in multilingual many-to-many scenarios where divergent read and write policies hinder unified strategy learning. In this paper, we present SimulMEGA (Simultaneous Generation by Mixture-of-Experts Gating), an unsupervised policy learning framework that combines prefix-based training with a Mixture-of-Experts refiner to learn effective read and write decisions in an implicit manner, without adding inference-time overhead. Our design requires only minimal modifications to standard transformer architectures and generalizes across both speech-to-text and text-to-speech streaming tasks. Through comprehensive evaluation on six language pairs, our 500M parameter speech-to-text model outperforms the Seamless baseline, achieving under 7 percent BLEU degradation at 1.5 seconds average lag and under 3 percent at 3 seconds. We further demonstrate the versatility of SimulMEGA by extending it to streaming TTS with a unidirectional backbone, yielding superior latency quality tradeoffs.
>
---
#### [replaced 044] NL-Debugging: Exploiting Natural Language as an Intermediate Representation for Code Debugging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.15356v2](http://arxiv.org/pdf/2505.15356v2)**

> **作者:** Weiming Zhang; Qingyao Li; Xinyi Dai; Jizheng Chen; Kounianhua Du; Weiwen Liu; Yasheng Wang; Ruiming Tang; Yong Yu; Weinan Zhang
>
> **摘要:** Debugging is a critical aspect of LLM's coding ability. Early debugging efforts primarily focused on code-level analysis, which often falls short when addressing complex programming errors that require a deeper understanding of algorithmic logic. Recent advancements in large language models (LLMs) have shifted attention toward leveraging natural language reasoning to enhance code-related tasks. However, two fundamental questions remain unanswered: What type of natural language format is most effective for debugging tasks? And what specific benefits does natural language reasoning bring to the debugging process? In this paper, we introduce NL-DEBUGGING, a novel framework that employs natural language as an intermediate representation to improve code debugging. By debugging at a natural language level, we demonstrate that NL-DEBUGGING outperforms traditional debugging methods and enables a broader modification space through direct refinement guided by execution feedback. Our findings highlight the potential of natural language reasoning to advance automated code debugging and address complex programming challenges.
>
---
#### [replaced 045] Large Language Models for Few-Shot Named Entity Recognition
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/1810.06818v3](http://arxiv.org/pdf/1810.06818v3)**

> **作者:** Yufei Zhao; Xiaoshi Zhong; Erik Cambria; Jagath C. Rajapakse
>
> **备注:** 17 pages, 2 figures. Accepted by AI, Computer Science and Robotics Technology (ACRT)
>
> **摘要:** Named entity recognition (NER) is a fundamental task in numerous downstream applications. Recently, researchers have employed pre-trained language models (PLMs) and large language models (LLMs) to address this task. However, fully leveraging the capabilities of PLMs and LLMs with minimal human effort remains challenging. In this paper, we propose GPT4NER, a method that prompts LLMs to resolve the few-shot NER task. GPT4NER constructs effective prompts using three key components: entity definition, few-shot examples, and chain-of-thought. By prompting LLMs with these effective prompts, GPT4NER transforms few-shot NER, which is traditionally considered as a sequence-labeling problem, into a sequence-generation problem. We conduct experiments on two benchmark datasets, CoNLL2003 and OntoNotes5.0, and compare the performance of GPT4NER to representative state-of-the-art models in both few-shot and fully supervised settings. Experimental results demonstrate that GPT4NER achieves the $F_1$ of 83.15\% on CoNLL2003 and 70.37\% on OntoNotes5.0, significantly outperforming few-shot baselines by an average margin of 7 points. Compared to fully-supervised baselines, GPT4NER achieves 87.9\% of their best performance on CoNLL2003 and 76.4\% of their best performance on OntoNotes5.0. We also utilize a relaxed-match metric for evaluation and report performance in the sub-task of named entity extraction (NEE), and experiments demonstrate their usefulness to help better understand model behaviors in the NER task.
>
---
#### [replaced 046] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10844v4](http://arxiv.org/pdf/2505.10844v4)**

> **作者:** Simeng Han; Howard Dai; Stephen Xia; Grant Zhang; Chen Liu; Lichang Chen; Hoang Huy Nguyen; Hongyuan Mei; Jiayuan Mao; R. Thomas McCoy
>
> **备注:** NeurIPS 2025
>
> **摘要:** Accuracy remains a standard metric for evaluating AI systems, but it offers limited insight into how models arrive at their solutions. In this work, we introduce a benchmark based on brainteasers written in long narrative form to probe more deeply into the types of reasoning strategies that models use. Brainteasers are well-suited for this goal because they can be solved with multiple approaches, such as a few-step solution that uses a creative insight or a longer solution that uses more brute force. We investigate large language models (LLMs) across multiple layers of reasoning, focusing not only on correctness but also on the quality and creativity of their solutions. We investigate many aspects of the reasoning process: (1) semantic parsing of the brainteasers into precise mathematical competition style formats; (2) generating solutions from these mathematical forms; (3) self-correcting solutions based on gold solutions; (4) producing step-by-step sketches of solutions; and (5) making use of hints. We find that LLMs are in many cases able to find creative, insightful solutions to brainteasers, suggesting that they capture some of the capacities needed to solve novel problems in creative ways. Nonetheless, there also remain situations where they rely on brute force despite the availability of more efficient, creative solutions, highlighting a potential direction for improvement in the reasoning abilities of LLMs.
>
---
#### [replaced 047] p-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.23234v4](http://arxiv.org/pdf/2509.23234v4)**

> **作者:** Runyan Tan; Shuang Wu; Phillip Howard
>
> **摘要:** Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments. The code is available at https://github.com/ryttry/p-less .
>
---
#### [replaced 048] FT-MDT: Extracting Decision Trees from Medical Texts via a Novel Low-rank Adaptation Method
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.04655v2](http://arxiv.org/pdf/2510.04655v2)**

> **作者:** Yuheng Li; Jiechao Gao; Wei Han; Wenwen Ouyang; Wei Zhu; Hui Yi Leong
>
> **备注:** Accepted by EMNLP-2025
>
> **摘要:** Knowledge of the medical decision process, which can be modeled as medical decision trees (MDTs), is critical to building clinical decision support systems. However, current MDT construction methods rely heavily on time-consuming and laborious manual annotation. To address this challenge, we propose PI-LoRA (Path-Integrated LoRA), a novel low-rank adaptation method for automatically extracting MDTs from clinical guidelines and textbooks. We integrate gradient path information to capture synergistic effects between different modules, enabling more effective and reliable rank allocation. This framework ensures that the most critical modules receive appropriate rank allocations while less important ones are pruned, resulting in a more efficient and accurate model for extracting medical decision trees from clinical texts. Extensive experiments on medical guideline datasets demonstrate that our PI-LoRA method significantly outperforms existing parameter-efficient fine-tuning approaches for the Text2MDT task, achieving better accuracy with substantially reduced model complexity. The proposed method achieves state-of-the-art results while maintaining a lightweight architecture, making it particularly suitable for clinical decision support systems where computational resources may be limited.
>
---
#### [replaced 049] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20249v2](http://arxiv.org/pdf/2505.20249v2)**

> **作者:** Yongan Yu; Qingchen Hu; Xianda Du; Jiayin Wang; Fengran Mo; Renee Sieber
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Climate change adaptation requires the understanding of disruptive weather impacts on society, where large language models (LLMs) might be applicable. However, their effectiveness is under-explored due to the difficulty of high-quality corpus collection and the lack of available benchmarks. The climate-related events stored in regional newspapers record how communities adapted and recovered from disasters. However, the processing of the original corpus is non-trivial. In this study, we first develop a disruptive weather impact dataset with a four-stage well-crafted construction pipeline. Then, we propose WXImpactBench, the first benchmark for evaluating the capacity of LLMs on disruptive weather impacts. The benchmark involves two evaluation tasks, multi-label classification and ranking-based question answering. Extensive experiments on evaluating a set of LLMs provide first-hand analysis of the challenges in developing disruptive weather impact understanding and climate change adaptation systems. The constructed dataset and the code for the evaluation framework are available to help society protect against vulnerabilities from disasters.
>
---
#### [replaced 050] Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.24302v2](http://arxiv.org/pdf/2510.24302v2)**

> **作者:** Shangyu Xing; Siyuan Wang; Chenyuan Yang; Xinyu Dai; Xiang Ren
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR), particularly with algorithms like Group Relative Policy Optimization (GRPO), has proven highly effective in enhancing the reasoning capabilities of large language models. However, a critical bottleneck in current pipelines lies in the limited diversity of sampled trajectories during group rollouts. Homogeneous trajectories and their associated rewards would diminish the return signals for policy updates, thereby hindering effective policy learning. This lack of diversity stems primarily from token-level stochastic sampling, where local variations are likely to collapse into near-identical reasoning paths. To address this limitation, we propose Lookahead Tree-Based Rollouts (LATR), a novel rollout strategy designed to explicitly promotes trajectory-level diversity by enforcing branching into different candidate tokens likely to yield distinct continuations. Specifically, LATR iteratively operates in three stages: (1) branching at high-uncertainty generation steps, (2) performing lookahead simulation for each new branch, and (3) pruning branches that exhibits prolonged similarity during simulation. Compared with stochastic Sampling, LATR accelerates policy learning by 131% on average and improves final pass@1 performance by 4.2% on both GRPO and Dynamic sAmpling Policy Optimization (DAPO) algorithms across different reasoning tasks. Our code and data are publicly available at https://github.com/starreeze/latr.
>
---
#### [replaced 051] Quantum Transformer: Accelerating model inference via quantum linear algebra
- **分类: quant-ph; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.16714v3](http://arxiv.org/pdf/2402.16714v3)**

> **作者:** Naixu Guo; Zhan Yu; Matthew Choi; Yizhan Han; Aman Agrawal; Kouhei Nakaji; Alán Aspuru-Guzik; Patrick Rebentrost
>
> **备注:** 45 pages
>
> **摘要:** Powerful generative artificial intelligence from large language models (LLMs) harnesses extensive computational resources for inference. In this work, we investigate the transformer architecture, a key component of these models, under the lens of fault-tolerant quantum computing. We develop quantum subroutines to construct the building blocks in the transformer, including the self-attention, residual connection with layer normalization, and feed-forward network. As an important subroutine, we show how to efficiently implement the Hadamard product and element-wise functions of matrices on quantum computers. Our algorithm prepares an amplitude encoding of the transformer output, which can be measured for prediction or use in the next layer. We find that the matrix norm of the input sequence plays a dominant role in the quantum complexity. With numerical experiments on open-source LLMs, including for bio-informatics applications, we demonstrate the potential of a quantum speedup for transformer inference in practical regimes.
>
---
#### [replaced 052] Robust Preference Optimization via Dynamic Target Margins
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03690v2](http://arxiv.org/pdf/2506.03690v2)**

> **作者:** Jie Sun; Junkang Wu; Jiancan Wu; Zhibo Zhu; Xingyu Lu; Jun Zhou; Lintao Ma; Xiang Wang
>
> **备注:** 18 pages, 6 figures, accepted to Findings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** The alignment of Large Language Models (LLMs) is crucial for ensuring their safety and reliability in practical applications. Direct Preference Optimization (DPO) has emerged as an efficient method that directly optimizes models using preference pairs, significantly reducing resource demands. However, the effectiveness of DPO heavily depends on the data quality, which is frequently compromised by noise. In this work, we propose $\gamma$-PO, a dynamic target margin preference optimization algorithm that adjust reward margins at the pairwise level. By introducing instance-specific margin calibration, $\gamma$-PO strategically prioritizes high-confidence pairs (those demonstrating higher reward margins) while suppressing potential noise from ambiguous pairs. Moreover, $\gamma$-PO is a plug-and-play method, compatible with variants of DPO that rely on reward margin between preference pairs. Across benchmarks such as AlpacaEval2 and Arena-Hard, $\gamma$-PO achieves an average 4.4\% improvement over other baselines, setting new benchmarks for state-of-the-art performance. Additionally, $\gamma$-PO requires minimal code changes and has a negligible impact on training efficiency, making it a robust solution for enhancing LLMs alignment. Our codes are available at \href{https://github.com/sunjie279/gammaPO}{https://github.com/sunjie279/gammaPO}.
>
---
#### [replaced 053] Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2510.09158v2](http://arxiv.org/pdf/2510.09158v2)**

> **作者:** Seiya Ishikura; Hiroaki Yamada; Tatsuya Hiraoka; Hiroaki Yamada; Takenobu Tokunaga
>
> **备注:** 8 pages, 1 figure. Accepted at the First Workshop on Tailoring AI: Exploring Active and Passive LLM Personalization (PALS2025@EMNLP2025)
>
> **摘要:** This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance.
>
---
#### [replaced 054] OpenGuardrails: A Configurable, Unified, and Scalable Guardrails Platform for Large Language Models
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2510.19169v2](http://arxiv.org/pdf/2510.19169v2)**

> **作者:** Thomas Wang; Haowen Li
>
> **摘要:** As large language models (LLMs) are increasingly integrated into real-world applications, ensuring their safety, robustness, and privacy compliance has become critical. We present OpenGuardrails, the first fully open-source platform that unifies large-model-based safety detection, manipulation defense, and deployable guardrail infrastructure. OpenGuardrails protects against three major classes of risks: (1) content-safety violations such as harmful or explicit text generation, (2) model-manipulation attacks including prompt injection, jailbreaks, and code-interpreter abuse, and (3) data leakage involving sensitive or private information. Unlike prior modular or rule-based frameworks, OpenGuardrails introduces three core innovations: (1) a Configurable Policy Adaptation mechanism that allows per-request customization of unsafe categories and sensitivity thresholds; (2) a Unified LLM-based Guard Architecture that performs both content-safety and manipulation detection within a single model; and (3) a Quantized, Scalable Model Design that compresses a 14B dense base model to 3.3B via GPTQ while preserving over 98 of benchmark accuracy. The system supports 119 languages, achieves state-of-the-art performance across multilingual safety benchmarks, and can be deployed as a secure gateway or API-based service for enterprise use. All models, datasets, and deployment scripts are released under the Apache 2.0 license.
>
---
#### [replaced 055] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.06426v2](http://arxiv.org/pdf/2504.06426v2)**

> **作者:** Hanqing Zeng; Yinglong Xia; Zhuokai Zhao; Chuan Jiang; Qiang Zhang; Jiayi Liu; Qunshu Zhang; Lizhu Zhang; Xiangjun Fan; Benyu Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Conceptually, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of numerous experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves structural flexibility of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation. Our implementation is available at: https://github.com/ZimpleX/SMoRE-LLM.
>
---
#### [replaced 056] WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2509.19902v2](http://arxiv.org/pdf/2509.19902v2)**

> **作者:** Binbin Zhang; Chengdong Liang; Shuai Wang; Xuelong Geng; Zhao Guo; Haoyu Li; Hao Yin; Xipeng Yang; Pengshen Zhang; Changwei Ma; Lei Xie
>
> **摘要:** In this paper, we present WEST(WE Speech Toolkit), a speech toolkit based on a large language model (LLM) for speech understanding, generation, and interaction. There are three key features of WEST: 1) Fully LLM-based: Standing on the shoulders of giants by reusing mature architectures, ecosystems (e.g., Hugging Face), and methods (e.g., sequence packing) from large models. 2) Full-stack: Supports tasks such as recognition, synthesis, understanding, dialogue, and multimodal capabilities, with extensibility to incorporate open-source models. 3) Simple and Stupid: A simple and stupid speech toolkit that everyone can Touch. In addition, WEST provides two types of recipes, models, and experimental results. The first is entirely based on open-source models and open-source data, allowing users to fully reproduce the experiments in this paper and serving as a verification system or minimal system baseline. The second is trained on massive data, offering superior performance so the user can directly apply it out of the box. WEST is publicly avilable at https://github.com/wenet-e2e/west/
>
---
#### [replaced 057] Are ASR foundation models generalized enough to capture features of regional dialects for low-resource languages?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2510.23252v2](http://arxiv.org/pdf/2510.23252v2)**

> **作者:** Tawsif Tashwar Dipto; Azmol Hossain; Rubayet Sabbir Faruque; Md. Rezuwan Hassan; Kanij Fatema; Tanmoy Shome; Ruwad Naswan; Md. Foriduzzaman Zihad; Mohaymen Ul Anam; Nazia Tasnim; Hasan Mahmud; Md Kamrul Hasan; Md. Mehedi Hasan Shawon; Farig Sadeque; Tahsin Reasat
>
> **备注:** The manuscript has to be withdrawn to address an authorship and intellectual property clarification
>
> **摘要:** Conventional research on speech recognition modeling relies on the canonical form for most low-resource languages while automatic speech recognition (ASR) for regional dialects is treated as a fine-tuning task. To investigate the effects of dialectal variations on ASR we develop a 78-hour annotated Bengali Speech-to-Text (STT) corpus named Ben-10. Investigation from linguistic and data-driven perspectives shows that speech foundation models struggle heavily in regional dialect ASR, both in zero-shot and fine-tuned settings. We observe that all deep learning methods struggle to model speech data under dialectal variations but dialect specific model training alleviates the issue. Our dataset also serves as a out of-distribution (OOD) resource for ASR modeling under constrained resources in ASR algorithms. The dataset and code developed for this project are publicly available
>
---
#### [replaced 058] Many LLMs Are More Utilitarian Than One
- **分类: cs.CL; cs.AI; cs.CY; I.2.7; I.2.11**

- **链接: [http://arxiv.org/pdf/2507.00814v2](http://arxiv.org/pdf/2507.00814v2)**

> **作者:** Anita Keshmirian; Razan Baltaji; Babak Hemmatian; Hadi Asghari; Lav R. Varshney
>
> **备注:** Accepted to the Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Moral judgment is integral to large language models' (LLMs) social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function when collaborating compared to operating as individual agents. In human moral judgment, group deliberation leads to a Utilitarian Boost: a tendency to endorse norm violations that inflict harm but maximize benefits for the greatest number of people. We study whether a similar dynamic emerges in multi-agent LLM systems. We test six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reason independently, and (2) Group, where they engage in multi-turn discussions in pairs or triads. In personal dilemmas, where agents decide whether to directly harm an individual for the benefit of others, all models rated moral violations as more acceptable when part of a group, demonstrating a Utilitarian Boost similar to that observed in humans. However, the mechanism for the Boost in LLMs differed: While humans in groups become more utilitarian due to heightened sensitivity to decision outcomes, LLM groups showed either reduced sensitivity to norms or enhanced impartiality. We report model differences in when and how strongly the Boost manifests. We also discuss prompt and agent compositions that enhance or mitigate the effect. We end with a discussion of the implications for AI alignment, multi-agent design, and artificial moral reasoning. Code available at: https://github.com/baltaci-r/MoralAgents
>
---
