# 自然语言处理 cs.CL

- **最新发布 101 篇**

- **更新 72 篇**

## 最新发布

#### [new 001] KG-Reasoner: A Reinforced Model for End-to-End Multi-Hop Knowledge Graph Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KG-Reasoner，解决多跳知识图谱推理问题。通过强化学习将多步推理整合到统一的思考过程，提升推理灵活性和准确性。**

- **链接: [https://arxiv.org/pdf/2604.12487](https://arxiv.org/pdf/2604.12487)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) exhibit strong abilities in natural language understanding and generation, yet they struggle with knowledge-intensive reasoning. Structured Knowledge Graphs (KGs) provide an effective form of external knowledge representation and have been widely used to enhance performance in classical Knowledge Base Question Answering (KBQA) tasks. However, performing precise multi-hop reasoning over KGs for complex queries remains highly challenging. Most existing approaches decompose the reasoning process into a sequence of isolated steps executed through a fixed pipeline. While effective to some extent, such designs constrain reasoning flexibility and fragment the overall decision process, often leading to incoherence and the loss of critical intermediate information from earlier steps. In this paper, we introduce KG-Reasoner, an end-to-end framework that integrates multi-step reasoning into a unified "thinking" phase of a Reasoning LLM. Through Reinforcement Learning (RL), the LLM is trained to internalize the KG traversal process, enabling it to dynamically explore reasoning paths, and perform backtracking when necessary. Experiments on eight multi-hop and knowledge-intensive reasoning benchmarks demonstrate that KG-Reasoner achieves competitive or superior performance compared to the state-of-the-art methods. Codes are available at the repository: this https URL.
>
---
#### [new 002] Representing expertise accelerates learning from pedagogical interaction data
- **分类: cs.CL; cs.MA**

- **简介: 该论文属于机器学习任务，旨在解决如何通过教学互动数据提升模型性能的问题。研究通过合成数据实验，验证了教学互动对模型鲁棒性的提升作用。**

- **链接: [https://arxiv.org/pdf/2604.12195](https://arxiv.org/pdf/2604.12195)**

> **作者:** Dhara Yu; Karthikeya Kaushik; Bill D. Thompson
>
> **摘要:** Work in cognitive science and artificial intelligence has suggested that exposing learning agents to traces of interaction between multiple individuals can improve performance in a variety of settings, yet it remains unknown which features of interactions contribute to this improvement. We examined the factors that support the effectiveness of interaction data, using a controlled paradigm that allowed us to precisely operationalize key distinctions between interaction and an expert acting alone. We generated synthetic datasets of simple interactions between an expert and a novice in a spatial navigation task, and then trained transformer models on those datasets, evaluating performance after exposure to different datasets. Our experiments showed that models trained on pedagogical interactions were more robust across a variety of scenarios compared to models trained only on expert demonstrations, and that having the ability to represent epistemically distinct agents led to expert-like behavior even when expert behavior was rarely observed.
>
---
#### [new 003] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，旨在解决符号模型缺乏语义能力、深度学习模型不透明的问题。通过将预训练语言模型知识转化为符号形式，提升模型的可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2604.12223](https://arxiv.org/pdf/2604.12223)**

> **作者:** Jiechao Gao; Rohan Kumar Yadav; Yuangang Li; Yuandong Pan; Jie Wang; Ying Liu; Michael Lepech
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Pretrained language models (PLMs) like BERT provide strong semantic representations but are costly and opaque, while symbolic models such as the Tsetlin Machine (TM) offer transparency but lack semantic generalization. We propose a semantic bootstrapping framework that transfers LLM knowledge into symbolic form, combining interpretability with semantic capacity. Given a class label, an LLM generates sub-intents that guide synthetic data creation through a three-stage curriculum (seed, core, enriched), expanding semantic diversity. A Non-Negated TM (NTM) learns from these examples to extract high-confidence literals as interpretable semantic cues. Injecting these cues into real data enables a TM to align clause logic with LLM-inferred semantics. Our method requires no embeddings or runtime LLM calls, yet equips symbolic models with pretrained semantic priors. Across multiple text classification tasks, it improves interpretability and accuracy over vanilla TM, achieving performance comparable to BERT while remaining fully symbolic and efficient.
>
---
#### [new 004] Topology-Aware Reasoning over Incomplete Knowledge Graph with Graph-Based Soft Prompting
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，解决KG不完整导致的多跳推理问题。提出基于图的软提示框架，通过子图推理提升模型对缺失边的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12503](https://arxiv.org/pdf/2604.12503)**

> **作者:** Shuai Wang; Xixi Wang; Yinan Yu
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities across various tasks but remain prone to hallucinations in knowledge-intensive scenarios. Knowledge Base Question Answering (KBQA) mitigates this by grounding generation in Knowledge Graphs (KGs). However, most multi-hop KBQA methods rely on explicit edge traversal, making them fragile to KG incompleteness. In this paper, we proposed a novel graph-based soft prompting framework that shifts the reasoning paradigm from node-level path traversal to subgraph-level reasoning. Specifically, we employ a Graph Neural Network (GNN) to encode extracted structural subgraphs into soft prompts, enabling LLM to reason over richer structural context and identify relevant entities beyond immediate graph neighbors, thereby reducing sensitivity to missing edges. Furthermore, we introduce a two-stage paradigm that reduces computational cost while preserving good performance: a lightweight LLM first leverages the soft prompts to identify question-relevant entities and relations, followed by a more powerful LLM for evidence-aware answer generation. Experiments on four multi-hop KBQA benchmarks show that our approach achieves state-of-the-art performance on three of them, demonstrating its effectiveness. Code is available at the repository: this https URL.
>
---
#### [new 005] AlphaEval: Evaluating Agents in Production
- **分类: cs.CL**

- **简介: 该论文提出AlphaEval，用于评估生产环境中AI代理的性能。解决现有基准与实际应用脱节的问题，通过真实任务和多范式评估方法，构建可复用的评估框架。**

- **链接: [https://arxiv.org/pdf/2604.12162](https://arxiv.org/pdf/2604.12162)**

> **作者:** Pengrui Lu; Bingyu Xu; Wenjun Zhang; Shengjia Hua; Xuanjian Gao; Ranxiang Ge; Lyumanshan Ye; Linxuan Wu; Yiran Li; Junfei Fish Yu; Yibo Zhang; Ruixin Li; Manxiang Li; Xiao Han; Xiaocong Zhou; Guangyao Chi; Zisheng Chen; Kaishen Chen; Kun Wang; Qihua Xu; Fengyue Meng; Yuchen Ni; Jiajun Li; Jinxiu Liu; Danfeng Zhang; Jingru Zhao; Pengfei Liu
>
> **摘要:** The rapid deployment of AI agents in commercial settings has outpaced the development of evaluation methodologies that reflect production realities. Existing benchmarks measure agent capabilities through retrospectively curated tasks with well-specified requirements and deterministic metrics -- conditions that diverge fundamentally from production environments where requirements contain implicit constraints, inputs are heterogeneous multi-modal documents with information fragmented across sources, tasks demand undeclared domain expertise, outputs are long-horizon professional deliverables, and success is judged by domain experts whose standards evolve over time. We present AlphaEval, a production-grounded benchmark of 94 tasks sourced from seven companies deploying AI agents in their core business, spanning six O*NET (Occupational Information Network) domains. Unlike model-centric benchmarks, AlphaEval evaluates complete agent products -- Claude Code, Codex, etc. -- as commercial systems, capturing performance variations invisible to model-level evaluation. Our evaluation framework covers multiple paradigms (LLM-as-a-Judge, reference-driven metrics, formal verification, rubric-based assessment, automated UI testing, etc.), with individual domains composing multiple paradigms. Beyond the benchmark itself, we contribute a requirement-to-benchmark construction framework -- a systematic methodology that transforms authentic production requirements into executable evaluation tasks in minimal time. This framework standardizes the entire pipeline from requirement to evaluation, providing a reproducible, modular process that any organization can adopt to construct production-grounded benchmarks for their own domains.
>
---
#### [new 006] Accelerating Speculative Decoding with Block Diffusion Draft Trees
- **分类: cs.CL**

- **简介: 该论文属于语言模型加速任务，解决 speculative decoding 的效率问题。通过构建 draft tree 提升验证效率，提高接受长度。**

- **链接: [https://arxiv.org/pdf/2604.12989](https://arxiv.org/pdf/2604.12989)**

> **作者:** Liran Ringel; Yaniv Romano
>
> **摘要:** Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve state-of-the-art speculative decoding performance, outperforming strong autoregressive drafters such as EAGLE-3. Vanilla DFlash, however, still verifies only a single drafted trajectory per round, potentially limiting its acceptance length. We introduce DDTree (Diffusion Draft Tree), a method that constructs a draft tree directly from the per-position distributions of a block diffusion drafter. Under a fixed node budget, DDTree uses a simple best-first heap algorithm to select the continuations that are most likely to match the target model according to a surrogate defined by the draft model's output. The resulting tree is verified efficiently in a single target model forward pass using an ancestor-only attention mask. Because DDTree builds on DFlash, a leading draft model for speculative decoding, these gains place DDTree among the leading approaches to speculative decoding.
>
---
#### [new 007] SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决自回归解码速度慢的问题。提出SpecBound框架，通过自适应限制推测长度和层间置信度校准，提升效率并保持输出准确。**

- **链接: [https://arxiv.org/pdf/2604.12247](https://arxiv.org/pdf/2604.12247)**

> **作者:** Zhuofan Wen; Yang Feng
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Speculative decoding has emerged as a promising approach to accelerate autoregressive inference in large language models (LLMs). Self-draft methods, which leverage the base LLM itself for speculation, avoid the overhead of auxiliary draft models but face limitations: shallow layers often produce overconfident yet incorrect token predictions, and the presence of difficult tokens in a draft sequence forces redundant computation through deeper layers, undermining both draft acceptance and overall speedup. To address these issues, we propose a novel self-draft framework that suppresses spurious confidence via layer-wise temperature annealing in early-exit decision and adaptively bounds speculation length based on token-wise decoding difficulty. By reprocessing the hidden states of draft tokens in a unified parallel pass through deep layers, our method maintains exact output equivalence with the original model while maximizing computational efficiency. It requires no modifications to the base LLM parameters and achieves up to 2.33x wall-time speedup over standard autoregressive decoding across diverse long-form generation tasks and multiple model architectures.
>
---
#### [new 008] Growing Pains: Extensible and Efficient LLM Benchmarking Via Fixed Parameter Calibration
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，解决多模型与多数据集间评价成本高、结果不可比的问题。通过固定参数校准方法，实现高效、可扩展的基准测试。**

- **链接: [https://arxiv.org/pdf/2604.12843](https://arxiv.org/pdf/2604.12843)**

> **作者:** Eliya Habba; Itay Itzhak; Asaf Yehudai; Yotam Perlitz; Elron Bandel; Michal Shmueli-Scheuer; Leshem Choshen; Gabriel Stanovsky
>
> **摘要:** The rapid release of both language models and benchmarks makes it increasingly costly to evaluate every model on every dataset. In practice, models are often evaluated on different samples, making scores difficult to compare across studies. To address this, we propose a framework based on multidimensional Item Response Theory (IRT) that uses anchor items to calibrate new benchmarks to the evaluation suite while holding previously calibrated item parameters fixed. Our approach supports a realistic evaluation setting in which datasets are introduced over time and models are evaluated only on the datasets available at the time of evaluation, while a fixed anchor set for each dataset is used so that results from different evaluation periods can be compared directly. In large-scale experiments on more than $400$ models, our framework predicts full-evaluation performance within 2-3 percentage points using only $100$ anchor questions per dataset, with Spearman $\rho \geq 0.9$ for ranking preservation, showing that it is possible to extend benchmark suites over time while preserving score comparability, at a constant evaluation cost per new dataset. Code available at this https URL
>
---
#### [new 009] LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，解决块状扩散模型在长上下文中的内存瓶颈问题。通过引入LOSA机制，减少KV加载量，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12056](https://arxiv.org/pdf/2604.12056)**

> **作者:** Haocheng Xi; Harman Singh; Yuezhou Hu; Coleman Hooper; Rishabh Tiwari; Aditya Tomar; Minjae Lee; Wonjun Kang; Michael Mahoney; Chenfeng Xu; Kurt Keutzer; Amir Gholami
>
> **备注:** 16 pages, 11 figures, 6 tables
>
> **摘要:** Block-wise diffusion language models (DLMs) generate multiple tokens in any order, offering a promising alternative to the autoregressive decoding pipeline. However, they still remain bottlenecked by memory-bound attention in long-context scenarios. Naive sparse attention fails on DLMs due to a KV Inflation problem, where different queries select different prefix positions, making the union of accessed KV pages large. To address this, we observe that between consecutive denoising steps, only a small fraction of active tokens exhibit significant hidden-state changes, while the majority of stable tokens remain nearly constant. Based on this insight, we propose LOSA (Locality-aware Sparse Attention), which reuses cached prefix-attention results for stable tokens and applies sparse attention only to active tokens. This substantially shrinks the number of KV indices that must be loaded, yielding both higher speedup and higher accuracy. Across multiple block-wise DLMs and benchmarks, LOSA preserves near-dense accuracy while significantly improving efficiency, achieving up to +9 points in average accuracy at aggressive sparsity levels while maintaining 1.54x lower attention density. It also achieves up to 4.14x attention speedup on RTX A6000 GPUs, demonstrating the effectiveness of the proposed method.
>
---
#### [new 010] Beyond Majority Voting: Efficient Best-Of-N with Radial Consensus Score
- **分类: cs.CL**

- **简介: 该论文属于语言模型答案选择任务，解决多候选回答中可靠答案选取问题。提出RCS方法，通过几何共识提升选择效果。**

- **链接: [https://arxiv.org/pdf/2604.12196](https://arxiv.org/pdf/2604.12196)**

> **作者:** Manh Nguyen; Sunil Gupta; Hung Le
>
> **摘要:** Large language models (LLMs) frequently generate multiple candidate responses for a given prompt, yet selecting the most reliable one remains challenging, especially when correctness diverges from surface-level majority agreement. Existing approaches, such as self-consistency, rely on discrete voting, while probability-based methods often fail to capture relationships among candidate answers or tend to underweight high-quality but less frequent responses, and do not fully leverage the geometric structure of answer representations. To address these limitations, we introduce Radial Consensus Score (RCS), a simple, efficient, and training-free method for best-of-N selection. RCS models semantic consensus by computing a weighted Fréchet mean (semantic center) of answer embeddings and ranking candidates by their radial distance to this center. Importantly, RCS provides a general framework that supports multiple weighting schemes, including uniform, frequency-based, and probability-based variants, enabling flexible integration of agreement signals and model confidence while remaining fully applicable in black-box settings. Extensive experiments across seven benchmarks covering short-form QA and long-form reasoning tasks, and five open-weight models, demonstrate that RCS variants consistently outperform strong baselines, with gains becoming more pronounced as the sampling budget increases. RCS also serves as an effective drop-in replacement for majority voting in multi-agent debate and exhibits strong robustness in black-box scenarios. Overall, these results highlight geometric consensus as a scalable and broadly applicable principle for reliable answer selection, extending beyond majority voting to more expressive and robust aggregation in LLM inference.
>
---
#### [new 011] Mining Large Language Models for Low-Resource Language Data: Comparing Elicitation Strategies for Hausa and Fongbe
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在通过策略性提示从大语言模型中提取低资源语言数据。研究比较了两种语言的提示策略，以提高可用文本的提取效率。**

- **链接: [https://arxiv.org/pdf/2604.12477](https://arxiv.org/pdf/2604.12477)**

> **作者:** Mahounan Pericles Adjovi; Roald Eiselen; Prasenjit Mitra
>
> **备注:** 11 pages, 5 figures, 6 tables; to appear in LREC-COLING 2026
>
> **摘要:** Large language models (LLMs) are trained on data contributed by low-resource language communities, yet the linguistic knowledge encoded in these models remains accessible only through commercial APIs. This paper investigates whether strategic prompting can extract usable text data from LLMs for two West African languages: Hausa (Afroasiatic, approximately 80 million speakers) and Fongbe (Niger-Congo, approximately 2 million speakers). We systematically compare six elicitation task types across two commercial LLMs (GPT-4o Mini and Gemini 2.5 Flash). GPT-4o Mini extracts 6-41 times more usable target-language words per API call than Gemini. Optimal strategies differ by language: Hausa benefits from functional text and dialogue, while Fongbe requires constrained generation prompts. We release all generated corpora and code.
>
---
#### [new 012] Empirical Evaluation of PDF Parsing and Chunking for Financial Question Answering with RAG
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究金融领域PDF理解的RAG系统，探讨不同解析和分块策略对问答任务的影响，旨在提升信息提取的准确性与结构保持。**

- **链接: [https://arxiv.org/pdf/2604.12047](https://arxiv.org/pdf/2604.12047)**

> **作者:** Omar El Bachyr; Yewei Song; Saad Ezzini; Jacques Klein; Tegawendé F. Bissyandé; Anas Zilali; Ulrick Ble; Anne Goujon
>
> **备注:** 12 pages
>
> **摘要:** PDF files are primarily intended for human reading rather than automated processing. In addition, the heterogeneous content of PDFs, such as text, tables, and images, poses significant challenges for parsing and information extraction. To address these difficulties, both practitioners and researchers are increasingly developing new methods, including the promising Retrieval-Augmented Generation (RAG) systems to automated PDF processing. However, there is no comprehensive study investigating how different components and design choices affect the performance of a RAG system for understanding PDFs. In this paper, we propose such a study (1) by focusing on Question Answering, a specific language understanding task, and (2) by leveraging two benchmarks from the financial domain, including TableQuest, our newly generated, publicly available benchmark. We systematically examine multiple PDF parsers and chunking strategies (with varied overlap), along with their potential synergies in preserving document structure and ensuring answer correctness. Overall, our results offer practical guidelines for building robust RAG pipelines for PDF understanding.
>
---
#### [new 013] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出StableToken，解决语音词器在噪声下的不稳定性问题。通过多分支结构和投票机制提升词器鲁棒性，增强下游语音大模型性能。**

- **链接: [https://arxiv.org/pdf/2509.22220](https://arxiv.org/pdf/2509.22220)**

> **作者:** Yuhan Song; Linhao Zhang; Chuhan Wu; Aiwei Liu; Wei Jia; Houfeng Wang; Xiao Zhou
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Prevalent semantic speech tokenizers, designed to capture linguistic content, are surprisingly fragile. We find they are not robust to meaning-irrelevant acoustic perturbations; even at high Signal-to-Noise Ratios (SNRs) where speech is perfectly intelligible, their output token sequences can change drastically, increasing the learning burden for downstream LLMs. This instability stems from two flaws: a brittle single-path quantization architecture and a distant training signal indifferent to intermediate token stability. To address this, we introduce StableToken, a tokenizer that achieves stability through a consensus-driven mechanism. Its multi-branch architecture processes audio in parallel, and these representations are merged via a powerful bit-wise voting mechanism to form a single, stable token sequence. StableToken sets a new state-of-the-art in token stability, drastically reducing Unit Edit Distance (UED) under diverse noise conditions. This foundational stability translates directly to downstream benefits, significantly improving the robustness of SpeechLLMs on a variety of tasks. Our code and model are publicly available at this https URL.
>
---
#### [new 014] EvoSpark: Endogenous Interactive Agent Societies for Unified Long-Horizon Narrative Evolution
- **分类: cs.CL**

- **简介: 该论文属于多智能体叙事生成任务，解决长周期叙事不连贯问题。提出EvoSpark框架，通过记忆管理和场景同步实现逻辑一致的长期叙事演化。**

- **链接: [https://arxiv.org/pdf/2604.12776](https://arxiv.org/pdf/2604.12776)**

> **作者:** Shiyu He; Minchi Kuang; Mengxian Wang; Bin Hu; Tingxiang Gu
>
> **备注:** Accepted to the Main Conference of ACL 2026
>
> **摘要:** Realizing endogenous narrative evolution in LLM-based multi-agent systems is hindered by the inherent stochasticity of generative emergence. In particular, long-horizon simulations suffer from social memory stacking, where conflicting relational states accumulate without resolution, and narrative-spatial dissonance, where spatial logic detaches from the evolving plot. To bridge this gap, we propose EvoSpark, a framework specifically designed to sustain logically coherent long-horizon narratives within Endogenous Interactive Agent Societies. To ensure consistency, the Stratified Narrative Memory employs a Role Socio-Evolutionary Base as living cognition, dynamically metabolizing experiences to resolve historical conflicts. Complementarily, Generative Mise-en-Scène mechanism enforces Role-Location-Plot alignment, synchronizing character presence with the narrative flow. Underpinning these is the Unified Narrative Operation Engine, which integrates an Emergent Character Grounding Protocol to transform stochastic sparking into persistent characters. This engine establishes a substrate that expands a minimal premise into an open-ended, evolving story world. Experiments demonstrate that EvoSpark significantly outperforms baselines across diverse paradigms, enabling the sustained generation of expressive and coherent narrative experiences.
>
---
#### [new 015] Masked by Consensus: Disentangling Privileged Knowledge in LLM Correctness
- **分类: cs.CL**

- **简介: 该论文研究大语言模型是否具备判断答案正确性的私有知识。任务是识别模型内部表示是否优于外部模型。工作包括训练分类器比较自表示与外部表示，发现事实性任务中存在领域特异性优势。**

- **链接: [https://arxiv.org/pdf/2604.12373](https://arxiv.org/pdf/2604.12373)**

> **作者:** Tomer Ashuach; Liat Ein-Dor; Shai Gretz; Yoav Katz; Yonatan Belinkov
>
> **备注:** Accepted to ACL 2026 (Main Conference). 8 pages, 16 figures, 2 tables
>
> **摘要:** Humans use introspection to evaluate their understanding through private internal states inaccessible to external observers. We investigate whether large language models possess similar privileged knowledge about answer correctness, information unavailable through external observation. We train correctness classifiers on question representations from both a model's own hidden states and external models, testing whether self-representations provide a performance advantage. On standard evaluation, we find no advantage: self-probes perform comparably to peer-model probes. We hypothesize this is due to high inter-model agreement of answer correctness. To isolate genuine privileged knowledge, we evaluate on disagreement subsets, where models produce conflicting predictions. Here, we discover domain-specific privileged knowledge: self-representations consistently outperform peer representations in factual knowledge tasks, but show no advantage in math reasoning. We further localize this domain asymmetry across model layers, finding that the factual advantage emerges progressively from early-to-mid layers onward, consistent with model-specific memory retrieval, while math reasoning shows no consistent advantage at any depth.
>
---
#### [new 016] Coding-Free and Privacy-Preserving MCP Framework for Clinical Agentic Research Intelligence System
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CARIS系统，解决临床研究流程繁琐与数据隐私问题。通过无代码和隐私保护框架，自动化完成研究规划、数据分析与报告生成。**

- **链接: [https://arxiv.org/pdf/2604.12258](https://arxiv.org/pdf/2604.12258)**

> **作者:** Taehun Kim; Hyeryun Park; Hyeonhoon Lee; Yushin Lee; Kyungsang Kim; Hyung-Chul Lee
>
> **备注:** 10 pages, 5 figures, 2 tables, Supplementary Appendix
>
> **摘要:** Clinical research involves labor-intensive processes such as study design, cohort construction, model development, and documentation, requiring domain expertise, programming skills, and access to sensitive patient data. These demands create barriers for clinicians and external researchers conducting data-driven studies. To overcome these limitations, we developed a Clinical Agentic Research Intelligence System (CARIS) that automates the clinical research workflow while preserving data privacy, enabling comprehensive studies without direct access to raw data. CARIS integrates Large Language Models (LLMs) with modular tools via the Model Context Protocol (MCP), enabling natural language-driven orchestration of appropriate tools. Databases remain securely within the MCP server, and users access only the outputs and final research reports. Based on user intent, CARIS automatically executes the full pipeline: research planning, literature search, cohort construction, Institutional Review Board (IRB) documentation, Vibe Machine Learning (ML), and report generation, with iterative human-in-the-loop refinement. We evaluated CARIS on three heterogeneous datasets with distinct clinical tasks. Research plans and IRB documents were finalized within three to four iterations, using evidence from literature and data. The system supported Vibe ML by exploring feature-model combinations, ranking the top ten models, and generating performance visualizations. Final reports showed high completeness based on a checklist derived from the TRIPOD+AI framework, achieving 96% coverage in LLM evaluation and 82% in human evaluation. CARIS demonstrates that agentic AI can transform clinical hypotheses into executable research workflows across heterogeneous datasets. By eliminating the need for coding and direct data access, the system lowers barriers and bridges public and private clinical data environments.
>
---
#### [new 017] Generating Effective CoT Traces for Mitigating Causal Hallucination
- **分类: cs.CL**

- **简介: 该论文属于事件因果识别任务，旨在解决小模型中的因果幻觉问题。通过生成有效的思维链追踪并引入新指标，提升模型准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12748](https://arxiv.org/pdf/2604.12748)**

> **作者:** Yiheng Zhao; Jun Yan
>
> **备注:** 11 pages, 2 figures. Accepted at ACL 2026
>
> **摘要:** Although large language models (LLMs) excel in complex reasoning tasks, they suffer from severe causal hallucination in event causality identification (ECI), particularly in smaller models ($\leq$1.5B parameters). A promising approach to address this issue is to fine-tune them with Chain-of-Thought (CoT) traces. However, there is currently a lack of CoT trace dataset available for ECI. In this paper, we first investigate the essential criteria that effective CoT traces should possess to mitigate causal hallucination in smaller models. We then design a pipeline to generate CoT traces that meet these criteria. Moreover, since there is currently no metric for quantifying causal hallucination, we also introduce a new metric, the Causal Hallucination Rate (CHR), to quantify causal hallucination, guide the formulation of effective CoT trace criteria, and validate the effectiveness of our pipeline. Our experiments show that fine-tuning with the CoT traces generated by our pipeline not only substantially reduces causal hallucination in smaller LLMs but also improves mean accuracy. Moreover, the fine-tuned models exhibit strong cross-dataset and cross-difficulty generalization, as well as robustness under misleading intervention prompts.
>
---
#### [new 018] When Self-Reference Fails to Close: Matrix-Level Dynamics in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型中自指输入对矩阵动态的影响，解决自指稳定性问题，通过多模型多指标分析，发现非终止真值递归引发不稳定。**

- **链接: [https://arxiv.org/pdf/2604.12128](https://arxiv.org/pdf/2604.12128)**

> **作者:** Ji Ho Bae
>
> **备注:** 14 pages, 4 figures, 11 tables
>
> **摘要:** We investigate how self-referential inputs alter the internal matrix dynamics of large language models. Measuring 106 scalar metrics across up to 7 analysis passes on four models from three architecture families -- Qwen3-VL-8B, Llama-3.2-11B, Llama-3.3-70B, and Gemma-2-9B -- over 300 prompts in a 14-level hierarchy at three temperatures ($T \in \{0.0, 0.3, 0.7\}$), we find that self-reference alone is not destabilizing: grounded self-referential statements and meta-cognitive prompts are markedly more stable than paradoxical self-reference on key collapse-related metrics, and on several such metrics can be as stable as factual controls. Instability concentrates in prompts inducing non-closing truth recursion (NCTR) -- truth-value computations with no finite-depth resolution. NCTR prompts produce anomalously elevated attention effective rank -- indicating attention reorganization with global dispersion rather than simple concentration collapse -- and key metrics reach Cohen's $d = 3.14$ (attention effective rank) to $3.52$ (variance kurtosis) vs. stable self-reference in the 70B model; 281/397 metric-model combinations differentiate NCTR from stable self-reference after FDR correction ($q < 0.05$), 198 with $|d| > 0.8$. Per-layer SVD confirms disruption at every sampled layer ($d > +1.0$ in all three models analyzed), ruling out aggregation artifacts. A classifier achieves AUC $0.81$-$0.90$; 30 minimal pairs yield 42/387 significant combinations; 43/106 metrics replicate across all four models. We connect these observations to three classical matrix-semigroup problems and propose, as a conjecture, that NCTR forces finite-depth transformers toward dynamical regimes where these problems concentrate. NCTR prompts also produce elevated contradictory output ($+34$-$56$ percentage points vs. controls), suggesting practical relevance for understanding self-referential failure modes.
>
---
#### [new 019] Benchmarking Deflection and Hallucination in Large Vision-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决现有基准在检索依赖性和幻觉检测上的不足。通过构建动态数据集和新基准，评估模型在证据冲突时的应对能力。**

- **链接: [https://arxiv.org/pdf/2604.12033](https://arxiv.org/pdf/2604.12033)**

> **作者:** Nicholas Moratelli; Christopher Davis; Leonardo F. R. Ribeiro; Bill Byrne; Gonzalo Iglesias
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Large Vision-Language Models (LVLMs) increasingly rely on retrieval to answer knowledge-intensive multimodal questions. Existing benchmarks overlook conflicts between visual and textual evidence and the importance of generating deflections (e.g., Sorry, I cannot answer...) when retrieved knowledge is incomplete. These benchmarks also suffer from rapid obsolescence, as growing LVLM training sets allow models to answer many questions without retrieval. We address these gaps with three contributions. First, we propose a dynamic data curation pipeline that preserves benchmark difficulty over time by filtering for genuinely retrieval-dependent samples. Second, we introduce VLM-DeflectionBench, a benchmark of 2,775 samples spanning diverse multimodal retrieval settings, designed to probe model behaviour under conflicting or insufficient evidence. Third, we define a fine-grained evaluation protocol with four scenarios that disentangle parametric memorization from retrieval robustness. Experiments across 20 state-of-the-art LVLMs indicate that models usually fail to deflect in the presence of noisy or misleading evidence. Our results highlight the need to evaluate not only what models know, but how they behave when they do not, and serve as a reusable and extensible benchmark for reliable KB-VQA evaluation. All resources will be publicly available upon publication.
>
---
#### [new 020] Calibrated Confidence Estimation for Tabular Question Answering
- **分类: cs.CL**

- **简介: 该论文属于表格问答任务，旨在解决大语言模型在结构化数据上的置信度估计问题。通过比较多种方法，提出MFA提升置信度评估效果。**

- **链接: [https://arxiv.org/pdf/2604.12491](https://arxiv.org/pdf/2604.12491)**

> **作者:** Lukas Voss
>
> **备注:** 27 pages, 9 figures, 17 tables (8-page main body + appendix)
>
> **摘要:** Large language models (LLMs) are increasingly deployed for tabular question answering, yet calibration on structured data is largely unstudied. This paper presents the first systematic comparison of five confidence estimation methods across five frontier LLMs and two tabular QA benchmarks. All models are severely overconfident (smooth ECE 0.35-0.64 versus 0.10-0.15 reported for textual QA). A consistent self-evaluation versus perturbation dichotomy replicates across both benchmarks and all four fully-covered models: self-evaluation methods (verbalized, P(True)) achieve AUROC 0.42-0.76, while perturbation methods (semantic entropy, self-consistency, and our Multi-Format Agreement) achieve AUROC 0.78-0.86. Per-model paired bootstrap tests reject the null at p<0.001 after Holm-Bonferroni correction, and a 3-seed check on GPT-4o-mini gives a per-seed standard deviation of only 0.006. The paper proposes Multi-Format Agreement (MFA), which exploits the lossless and deterministic serialization variation unique to structured data (Markdown, HTML, JSON, CSV) to estimate confidence at 20% lower API cost than sampling baselines. MFA reduces ECE by 44-63%, generalizes across all four models on TableBench (mean AUROC 0.80), and combines complementarily with sampling: an MFA + self-consistency ensemble lifts AUROC from 0.74 to 0.82. A secondary contribution, structure-aware recalibration, improves AUROC by +10 percentage points over standard post-hoc methods.
>
---
#### [new 021] ContextLens: Modeling Imperfect Privacy and Safety Context for Legal Compliance
- **分类: cs.CL**

- **简介: 该论文属于法律合规任务，解决真实场景下上下文不完整导致的隐私与安全评估问题。提出ContextLens框架，通过LLMs回答定制问题来评估合规性。**

- **链接: [https://arxiv.org/pdf/2604.12308](https://arxiv.org/pdf/2604.12308)**

> **作者:** Haoran Li; Yulin Chen; Huihao Jing; Wenbin Hu; Tsz Ho Li; Chanhou Lou; Hong Ting Tsang; Sirui Han; Yangqiu Song
>
> **备注:** Accepted by ACL 26
>
> **摘要:** Individuals' concerns about data privacy and AI safety are highly contextualized and extend beyond sensitive patterns. Addressing these issues requires reasoning about the context to identify and mitigate potential risks. Though researchers have widely explored using large language models (LLMs) as evaluators for contextualized safety and privacy assessments, these efforts typically assume the availability of complete and clear context, whereas real-world contexts tend to be ambiguous and incomplete. In this paper, we propose ContextLens, a semi-rule-based framework that leverages LLMs to ground the input context in the legal domain and explicitly identify both known and unknown factors for legal compliance. Instead of directly assessing safety outcomes, our ContextLens instructs LLMs to answer a set of crafted questions that span over applicability, general principles and detailed provisions to assess compliance with pre-defined priorities and rules. We conduct extensive experiments on existing compliance benchmarks that cover the General Data Protection Regulation (GDPR) and the EU AI Act. The results suggest that our ContextLens can significantly improve LLMs' compliance assessment and surpass existing baselines without any training. Additionally, our ContextLens can further identify the ambiguous and missing factors.
>
---
#### [new 022] Round-Trip Translation Reveals What Frontier Multilingual Benchmarks Miss
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言任务，旨在解决现有基准测试未能准确评估多语言能力的问题。通过轮转翻译方法，更真实地衡量模型的多语言生成能力。**

- **链接: [https://arxiv.org/pdf/2604.12911](https://arxiv.org/pdf/2604.12911)**

> **作者:** Ronald Skorobogat; Ameya Prabhu; Matthias Bethge
>
> **摘要:** Multilingual benchmarks guide the development of frontier models. Yet multilingual evaluations reported by frontier models are structured similar to popular reasoning and knowledge benchmarks, but across many languages. We show such benchmarks, and consequently multilingual evaluations, measure mathematical reasoning and factual recall, not multilingual proficiency. For example, thinking variants dramatically outperform instruct variants on these benchmarks, yet often perform worse on real-world multilingual tasks, such as LMArena. We propose a simple alternative: evaluate multilingual capability via round-trip translation. Given text in a source language, translate it to a target language and back; semantic gaps between the original and result expose failures in multilingual generation capabilities. Round-trip translation correlates almost perfectly (\r{ho} = 0.94) with user ratings on LMArena with our benchmark, requires no human reference translations, and does not require a more capable multilingual judge than tested models. Lastly, we introduce Lost in Translation (LiT), a challenging round-trip translation benchmark spanning widely spoken languages worldwide, for realistic evaluation of multilingual frontier models.
>
---
#### [new 023] Latent-Condensed Transformer for Efficient Long Context Modeling
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本处理中计算成本高和缓存占用大的问题。提出LCA方法，在低维潜在空间中压缩上下文，降低计算和缓存开销。**

- **链接: [https://arxiv.org/pdf/2604.12452](https://arxiv.org/pdf/2604.12452)**

> **作者:** Zeng You; Yaofo Chen; Qiuwu Chen; Ying Sun; Shuhai Zhang; Yingjian Li; Yaowei Wang; Mingkui Tan
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by projecting tokens into a low-dimensional latent space, while sparse attention reduces computation. However, sparse methods cannot operate natively on MLA's compressed latent structure, missing opportunities for joint optimization. In this paper, we propose Latent-Condensed Attention (LCA), which directly condenses context within MLA's latent space, where the representation is disentangled into semantic latent vectors and positional keys. LCA separately aggregates semantic vectors via query-aware pooling and preserves positional keys via anchor selection. This approach jointly reduces both computational cost and KV cache without adding parameters. Beyond MLA, LCA's design is architecture-agnostic and readily extends to other attention mechanisms such as GQA. Theoretically, we prove a length-independent error bound. Experiments show LCA achieves up to 2.5$\times$ prefilling speedup and 90% KV cache reduction at 128K context while maintaining competitive performance.
>
---
#### [new 024] Latent Planning Emerges with Scale
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的隐式规划能力，探讨其随规模增长的变化。通过简单和复杂任务分析，验证了模型规划能力与规模相关。**

- **链接: [https://arxiv.org/pdf/2604.12493](https://arxiv.org/pdf/2604.12493)**

> **作者:** Michael Hanna; Emmanuel Ameisen
>
> **备注:** ICLR 2026
>
> **摘要:** LLMs can perform seemingly planning-intensive tasks, like writing coherent stories or functioning code, without explicitly verbalizing a plan; however, the extent to which they implicitly plan is unknown. In this paper, we define latent planning as occurring when LLMs possess internal planning representations that (1) cause the generation of a specific future token or concept, and (2) shape preceding context to license said future token or concept. We study the Qwen-3 family (0.6B-14B) on simple planning tasks, finding that latent planning ability increases with scale. Models that plan possess features that represent a planned-for word like "accountant", and cause them to output "an" rather than "a"; moreover, even the less-successful Qwen-3 4B-8B have nascent planning mechanisms. On the more complex task of completing rhyming couplets, we find that models often identify a rhyme ahead of time, but even large models seldom plan far ahead. However, we can elicit some planning that increases with scale when steering models towards planned words in prose. In sum, we offer a framework for measuring planning and mechanistic evidence of how models' planning abilities grow with scale.
>
---
#### [new 025] CompliBench: Benchmarking LLM Judges for Compliance Violation Detection in Dialogue Systems
- **分类: cs.CL**

- **简介: 该论文属于对话系统合规性检测任务，旨在解决LLM法官检测违规行为能力不足的问题。提出CompliBench基准及自动化数据生成方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.12312](https://arxiv.org/pdf/2604.12312)**

> **作者:** Jingbo Yang; Guanyu Yao; Bairu Hou; Xinghan Yang; Nikolai Glushnev; Iwona Bialynicka-Birula; Duo Ding; Shiyu Chang
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed as task-oriented agents in enterprise environments, ensuring their strict adherence to complex, domain-specific operational guidelines is critical. While utilizing an LLM-as-a-Judge is a promising solution for scalable evaluation, the reliability of these judges in detecting specific policy violations remains largely unexplored. This gap is primarily due to the lack of a systematic data generation method, which has been hindered by the extensive cost of fine-grained human annotation and the difficulty of synthesizing realistic agent violations. In this paper, we introduce CompliBench, a novel benchmark designed to evaluate the ability of LLM judges to detect and localize guideline violations in multi-turn dialogues. To overcome data scarcity, we develop a scalable, automated data generation pipeline that simulates user-agent interactions. Our controllable flaw injection process automatically yields precise ground-truth labels for the violated guideline and the exact conversation turn, while an adversarial search method ensures these introduced perturbations are highly challenging. Our comprehensive evaluation reveals that current state-of-the-art proprietary LLMs struggle significantly with this task. In addition, we demonstrate that a small-scale judge model fine-tuned on our synthesized data outperforms leading LLMs and generalizes well to unseen business domains, highlighting our pipeline as an effective foundation for training robust generative reward models.
>
---
#### [new 026] Towards Robust Real-World Spreadsheet Understanding with Multi-Agent Multi-Format Reasoning
- **分类: cs.CL**

- **简介: 该论文属于表格理解任务，解决真实场景下表格处理效率低、信息提取不准确的问题。提出SpreadsheetAgent框架，通过多模态逐步解析提升表格理解能力。**

- **链接: [https://arxiv.org/pdf/2604.12282](https://arxiv.org/pdf/2604.12282)**

> **作者:** Houxing Ren; Mingjie Zhan; Zimu Lu; Ke Wang; Yunqiao Yang; Haotian Hou; Hongsheng Li
>
> **备注:** Accepted to ACL 2026 (main conference)
>
> **摘要:** Spreadsheets are central to real-world applications such as enterprise reporting, auditing, and scientific data management. Despite their ubiquity, existing large language model based approaches typically treat tables as plain text, overlooking critical layout cues and visual semantics. Moreover, real-world spreadsheets are often massive in scale, exceeding the input length that LLMs can efficiently process. To address these challenges, we propose SpreadsheetAgent, a two-stage multi-agent framework for spreadsheet understanding that adopts a step-by-step reading and reasoning paradigm. Instead of loading the entire spreadsheet at once, SpreadsheetAgent incrementally interprets localized regions through multiple modalities, including code execution results, images, and LaTeX tables. The method first constructs a structural sketch and row/column summaries, and then performs task-driven reasoning over this intermediate representation in the Solving Stage. To further enhance reliability, we design a verification module that validates extracted structures via targeted inspections, reducing error propagation and ensuring trustworthy inputs for downstream reasoning. Extensive experiments on two spreadsheet datasets demonstrate the effectiveness of our approach. With GPT-OSS-120B, SpreadsheetAgent achieves 38.16% on Spreadsheet Bench, outperforming the ChatGPT Agent baseline (35.27%) by 2.89 absolute points. These results highlight the potential of SpreadsheetAgent to advance robust and scalable spreadsheet understanding in real-world applications. Code is available at this https URL.
>
---
#### [new 027] Multilingual Multi-Label Emotion Classification at Scale with Synthetic Data
- **分类: cs.CL**

- **简介: 该论文属于多语言多标签情感分类任务，解决标注数据稀缺问题，通过合成数据训练模型并在多语言上取得良好效果。**

- **链接: [https://arxiv.org/pdf/2604.12633](https://arxiv.org/pdf/2604.12633)**

> **作者:** Vadim Borisov
>
> **摘要:** Emotion classification in multilingual settings remains constrained by the scarcity of annotated data: existing corpora are predominantly English, single-label, and cover few languages. We address this gap by constructing a large-scale synthetic training corpus of over 1M multi-label samples (50k per language) across 23 languages: Arabic, Bengali, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Mandarin, Polish, Portuguese, Punjabi, Russian, Spanish, Swahili, Tamil, Turkish, Ukrainian, Urdu, and Vietnamese, covering 11 emotion categories using culturally-adapted generation and programmatic quality filtering. We train and compare six multilingual transformer encoders, from DistilBERT (135M parameters) to XLM-R-Large (560M parameters), under identical conditions. On our in-domain test set, XLM-R-Large achieves 0.868 F1-micro and 0.987 AUC-micro. To validate against human-annotated data, we evaluate all models zero-shot on GoEmotions (English) and SemEval-2018 Task 1 E-c (English, Arabic, Spanish). On threshold-free ranking metrics, XLM-R-Large matches or exceeds English-only specialist models, tying on AP-micro (0.636) and LRAP (0.804) while surpassing on AUC-micro (0.810 vs. 0.787), while natively supporting all 23 languages. The best base-sized model is publicly available at this https URL
>
---
#### [new 028] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多模态情感分析任务，解决弱模态被主导模态压制的问题。提出EBMC框架，通过增强与平衡机制提升融合性能和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12518](https://arxiv.org/pdf/2604.12518)**

> **作者:** Kang He; Yuzhe Ding; Xinrong Wang; Fei Li; Chong Teng; Donghong Ji
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Multimodal sentiment analysis (MSA) integrates heterogeneous text, audio, and visual signals to infer human emotions. While recent approaches leverage cross-modal complementarity, they often struggle to fully utilize weaker modalities. In practice, dominant modalities tend to overshadow non-verbal ones, inducing modality competition and limiting overall contributions. This imbalance degrades fusion performance and robustness under noisy or missing modalities. To address this, we propose a novel model, Enhance-then-Balance Modality Collaboration framework (EBMC). EBMC improves representation quality via semantic disentanglement and cross-modal enhancement, strengthening weaker modalities. To prevent dominant modalities from overwhelming others, an Energy-guided Modality Coordination mechanism achieves implicit gradient rebalancing via a differentiable equilibrium objective. Furthermore, Instance-aware Modality Trust Distillation estimates sample-level reliability to adaptively modulate fusion weights, ensuring robustness. Extensive experiments demonstrate that EBMC achieves state-of-the-art or competitive results and maintains strong performance under missing-modality settings.
>
---
#### [new 029] GLeMM: A large-scale multilingual dataset for morphological research
- **分类: cs.CL**

- **简介: 该论文提出GLeMM，一个大规模多语言形态学数据集，用于解决形态变化机制研究中的数据不足与可重复性问题。任务为形态学研究，工作包括数据构建与自动标注。**

- **链接: [https://arxiv.org/pdf/2604.12442](https://arxiv.org/pdf/2604.12442)**

> **作者:** Hathout Nabil; Basilio Calderone; Fiammetta Namer; Franck Sajous
>
> **摘要:** In derivational morphology, what mechanisms govern the variation in form-meaning relations between words? The answers to this type of questions are typically based on intuition and on observations drawn from limited data, even when a wide range of languages is considered. Many of these studies are difficult to replicate and generalize. To address this issue, we present GLeMM, a new derivational resource designed for experimentation and data-driven description in morphology. GLeMM is characterized by (i) its large size, (ii) its extensive coverage (currently amounting to seven European languages, i.e., German, English, Spanish, French, Italian, Polish, Russian, (iii) its fully automated design, identical across all languages, (iv) the automatic annotation of morphological features on each entry, as well as (v) the encoding of semantic descriptions for a significant subset of these entries. It enables researchers to address difficult questions, such as the role of form and meaning in word-formation, and to develop and experimentally test computational methods that identify the structures of derivational morphology. The article describes how GLeMM is created using Wiktionary articles and presents various case studies illustrating possible applications of the resource.
>
---
#### [new 030] Toward Autonomous Long-Horizon Engineering for ML Research
- **分类: cs.CL**

- **简介: 该论文属于机器学习研究自动化任务，旨在解决长周期ML工程难题。通过构建AiScientist系统，实现任务协调与状态持续，提升研究效率。**

- **链接: [https://arxiv.org/pdf/2604.13018](https://arxiv.org/pdf/2604.13018)**

> **作者:** Guoxin Chen; Jie Chen; Lei Chen; Jiale Zhao; Fanzhe Meng; Wayne Xin Zhao; Ruihua Song; Cheng Chen; Ji-Rong Wen; Kai Jia
>
> **备注:** Repo: this https URL
>
> **摘要:** Autonomous AI research has advanced rapidly, but long-horizon ML research engineering remains difficult: agents must sustain coherent progress across task comprehension, environment setup, implementation, experimentation, and debugging over hours or days. We introduce AiScientist, a system for autonomous long-horizon engineering for ML research built on a simple principle: strong long-horizon performance requires both structured orchestration and durable state continuity. To this end, AiScientist combines hierarchical orchestration with a permission-scoped File-as-Bus workspace: a top-level Orchestrator maintains stage-level control through concise summaries and a workspace map, while specialized agents repeatedly re-ground on durable artifacts such as analyses, plans, code, and experimental evidence rather than relying primarily on conversational handoffs, yielding thin control over thick state. Across two complementary benchmarks, AiScientist improves PaperBench score by 10.54 points on average over the best matched baseline and achieves 81.82 Any Medal% on MLE-Bench Lite. Ablation studies further show that File-as-Bus protocol is a key driver of performance, reducing PaperBench by 6.41 points and MLE-Bench Lite by 31.82 points when removed. These results suggest that long-horizon ML research engineering is a systems problem of coordinating specialized work over durable project state, rather than a purely local reasoning problem.
>
---
#### [new 031] NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出NaviRAG，解决RAG在复杂任务中信息检索与合成的不足。通过构建层次化知识结构，实现动态、多粒度的信息导航，提升问答性能。**

- **链接: [https://arxiv.org/pdf/2604.12766](https://arxiv.org/pdf/2604.12766)**

> **作者:** Jihao Dai; Dingjun Wu; Yuxuan Chen; Zheni Zeng; Yukun Yan; Zhenghao Liu; Maosong Sun
>
> **摘要:** Retrieval-augmented generation (RAG) typically relies on a flat retrieval paradigm that maps queries directly to static, isolated text segments. This approach struggles with more complex tasks that require the conditional retrieval and dynamic synthesis of information across different levels of granularity (e.g., from broad concepts to specific evidence). To bridge this gap, we introduce NaviRAG, a novel framework that shifts from passive segment retrieval to active knowledge navigation. NaviRAG first structures the knowledge documents into a hierarchical form, preserving semantic relationships from coarse-grained topics to fine-grained details. Leveraging this reorganized knowledge records, a large language model (LLM) agent actively navigates the records, iteratively identifying information gaps and retrieving relevant content from the most appropriate granularity level. Extensive experiments on long-document QA benchmarks show that NaviRAG consistently improves both retrieval recall and end-to-end answer performance over conventional RAG baselines. Ablation studies confirm performance gains stem from our method's capacity for multi-granular evidence localization and dynamic retrieval planning. We further discuss efficiency, applicable scenario, and future directions of our method, hoping to make RAG systems more intelligent and autonomous.
>
---
#### [new 032] Meet Dynamic Individual Preferences: Resolving Conflicting Human Value with Paired Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于个性化模型对齐任务，旨在解决个体偏好动态变化和冲突的问题。提出PFT框架与VCD数据集，提升模型对用户偏好的适应能力。**

- **链接: [https://arxiv.org/pdf/2604.12479](https://arxiv.org/pdf/2604.12479)**

> **作者:** Shanyong Wang; Shuhang Lin; Yining Zhao; Xi Zhu; Yongfeng Zhang
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved the alignment of models with general human preferences. However, a major challenge remains in adapting LLMs to individual preferences, which are not only diverse but also dynamic. In this paper, we introduce a novel framework, Preference-Paired Fine-Tuning (PFT), designed to align models with contradictory and evolving individual preferences. We present a new dataset, Value Conflict Dilemma (VCD), which includes scenarios that involve conflicting human preferences, facilitating the evaluation of our approach. Our experiments demonstrate that PFT outperforms single-preference training methods, achieving up to 96.6% accuracy in multi-choice classification tasks and the highest open-ended generation score of 8.69. PFT also shows significant improvements over DPO, SFT and some traditional training methods, especially when handling conflicting preferences. Additionally, with limited user history data, models can inferring preference vector rapidly, achieving a 44.76% improvement in user-specific preference alignment in comparison to single-preference models.
>
---
#### [new 033] The role of System 1 and System 2 semantic memory structure in human and LLM biases
- **分类: cs.CL**

- **简介: 该论文属于认知科学与人工智能交叉任务，旨在探究人类和LLM的隐性偏见来源。通过建模系统1和系统2语义记忆结构，分析其与性别偏见的关系，揭示人类与机器认知的根本差异。**

- **链接: [https://arxiv.org/pdf/2604.12816](https://arxiv.org/pdf/2604.12816)**

> **作者:** Katherine Abramski; Giulio Rossetti; Massimo Stella
>
> **备注:** 31 pages, 5 figures, 9 appendix figures
>
> **摘要:** Implicit biases in both humans and large language models (LLMs) pose significant societal risks. Dual process theories propose that biases arise primarily from associative System 1 thinking, while deliberative System 2 thinking mitigates bias, but the cognitive mechanisms that give rise to this phenomenon remain poorly understood. To better understand what underlies this duality in humans, and possibly in LLMs, we model System 1 and System 2 thinking as semantic memory networks with distinct structures, built from comparable datasets generated by both humans and LLMs. We then investigate how these distinct semantic memory structures relate to implicit gender bias using network-based evaluation metrics. We find that semantic memory structures are irreducible only in humans, suggesting that LLMs lack certain types of human-like conceptual knowledge. Moreover, semantic memory structure relates consistently to implicit bias only in humans, with lower levels of bias in System~2 structures. These findings suggest that certain types of conceptual knowledge contribute to bias regulation in humans, but not in LLMs, highlighting fundamental differences between human and machine cognition.
>
---
#### [new 034] MetFuse: Figurative Fusion between Metonymy and Metaphor
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的隐喻与转喻研究任务，旨在解决二者协同分析的问题。提出MetFuse框架生成融合数据集，提升分类效果并分析两者相互影响。**

- **链接: [https://arxiv.org/pdf/2604.12919](https://arxiv.org/pdf/2604.12919)**

> **作者:** Saptarshi Ghosh; Tianyu Jiang
>
> **备注:** ACL 2026
>
> **摘要:** Metonymy and metaphor often co-occur in natural language, yet computational work has studied them largely in isolation. We introduce a framework that transforms a literal sentence into three figurative variants: metonymic, metaphoric, and hybrid. Using this framework, we construct MetFuse, the first dedicated dataset of figurative fusion between metonymy and metaphor, containing 1,000 human-verified meaning-aligned quadruplets totaling 4,000 sentences. Extrinsic experiments on eight existing benchmarks show that augmenting training data with MetFuse consistently improves both metonymy and metaphor classification, with hybrid examples yielding the largest gains on metonymy tasks. Using this dataset, we also analyze how the presence of one figurative type impacts another. Our findings show that both human annotators and large language models better identify metonymy in hybrid sentences than in metonymy-only sentences, demonstrating that the presence of a metaphor makes a metonymic noun more explicit. Our dataset is publicly available at: this https URL.
>
---
#### [new 035] AgenticAI-DialogGen: Topic-Guided Conversation Generation for Fine-Tuning and Evaluating Short- and Long-Term Memories of LLMs
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于对话生成任务，旨在解决LLMs短长期记忆评估难题。提出AgenticAI-DialogGen框架，生成带话题引导的对话，并构建TGC数据集以提升记忆相关任务性能。**

- **链接: [https://arxiv.org/pdf/2604.12179](https://arxiv.org/pdf/2604.12179)**

> **作者:** Manoj Madushanka Perera; Adnan Mahmood; Kasun Eranda Wijethilake; Quan Z. Sheng
>
> **备注:** 13 pages, 5 figures, 5 tables
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have improved their ability to process extended conversational contexts, yet fine-tuning and evaluating short- and long-term memories remain difficult due to the absence of datasets that encode both short- and long-term conversational history. Existing conversational datasets lack memory grounding, overlook topic continuity, or rely on costly human annotation. To address these gaps, we introduce AgenticAI-DialogGen, a modular agent-based framework that generates persona-grounded and topic-guided conversations without human supervision. The framework uses LLM agents to extract knowledge graphs, identify topics, build speaker personas, and simulate topic-guided conversations from unstructured conversations. A QA module generates memory-grounded Question Answer (QA) pairs drawn from short- and long-term conversational histories. We also generated a new dataset entitled, TopicGuidedChat (TGC), where long-term memory is encoded as speaker-specific knowledge graphs and short-term memory as newly generated topic-guided conversations. Evaluations depict that AgenticAI-DialogGen yields higher conversational quality and LLMs fine-tuned on TGC dataset achieve improved performance on memory-grounded QA tasks.
>
---
#### [new 036] LLMs Struggle with Abstract Meaning Comprehension More Than Expected
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于抽象意义理解任务，旨在解决大语言模型在抽象概念理解上的困难。通过改进模型结构提升其表现。**

- **链接: [https://arxiv.org/pdf/2604.12018](https://arxiv.org/pdf/2604.12018)**

> **作者:** Hamoud Alhazmi; Jiachen Jiang
>
> **摘要:** Understanding abstract meanings is crucial for advanced language comprehension. Despite extensive research, abstract words remain challenging due to their non-concrete, high-level semantics. SemEval-2021 Task 4 (ReCAM) evaluates models' ability to interpret abstract concepts by presenting passages with questions and five abstract options in a cloze-style format. Key findings include: (1) Most large language models (LLMs), including GPT-4o, struggle with abstract meaning comprehension under zero-shot, one-shot, and few-shot settings, while fine-tuned models like BERT and RoBERTa perform better. (2) A proposed bidirectional attention classifier, inspired by human cognitive strategies, enhances fine-tuned models by dynamically attending to passages and options. This approach improves accuracy by 4.06 percent on Task 1 and 3.41 percent on Task 2, demonstrating its potential for abstract meaning comprehension.
>
---
#### [new 037] Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决RAG方法中忽略知识顺序的问题。提出OKH-RAG，通过超图建模知识顺序，提升问答与解释任务效果。**

- **链接: [https://arxiv.org/pdf/2604.12185](https://arxiv.org/pdf/2604.12185)**

> **作者:** Keshu Wu; Chenchen Kuai; Zihao Li; Jiwan Jiang; Shiyu Shen; Shian Wang; Chan-Wei Hu; Zhengzhong Tu; Yang Zhou
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models by grounding outputs in retrieved knowledge. However, existing RAG methods including graph- and hypergraph-based approaches treat retrieved evidence as an unordered set, implicitly assuming permutation invariance. This assumption is misaligned with many real-world reasoning tasks, where outcomes depend not only on which interactions occur, but also on the order in which they unfold. We propose Order-Aware Knowledge Hypergraph RAG (OKH-RAG), which treats order as a first-class structural property. OKH-RAG represents knowledge as higher-order interactions within a hypergraph augmented with precedence structure, and reformulates retrieval as sequence inference over hyperedges. Instead of selecting independent facts, it recovers coherent interaction trajectories that reflect underlying reasoning processes. A learned transition model infers precedence directly from data without requiring explicit temporal supervision. We evaluate OKH-RAG on order-sensitive question answering and explanation tasks, including tropical cyclone and port operation scenarios. OKH-RAG consistently outperforms permutation-invariant baselines, and ablations show that these gains arise specifically from modeling interaction order. These results highlight a key limitation of set-based retrieval: effective reasoning requires not only retrieving relevant evidence, but organizing it into structured sequences.
>
---
#### [new 038] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection
- **分类: cs.CL**

- **简介: 该论文属于中文仇恨内容检测任务，旨在提升模型的可解释性。提出ToxiTrace方法，通过三个组件提高毒性片段提取的准确性和可读性。**

- **链接: [https://arxiv.org/pdf/2604.12321](https://arxiv.org/pdf/2604.12321)**

> **作者:** Boyang Li; Hongzhe Shou; Yuanyuan Liang; Jingbin Zhang; Fang Zhou
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Existing Chinese toxic content detection methods mainly target sentence-level classification but often fail to provide readable and contiguous toxic evidence spans. We propose \textbf{ToxiTrace}, an explainability-oriented method for BERT-style encoders with three components: (1) \textbf{CuSA}, which refines encoder-derived saliency cues into fine-grained toxic spans with lightweight LLM guidance; (2) \textbf{GCLoss}, a gradient-constrained objective that concentrates token-level saliency on toxic evidence while suppressing irrelevant activations; and (3) \textbf{ARCL}, which constructs sample-specific contrastive reasoning pairs to sharpen the semantic boundary between toxic and non-toxic content. Experiments show that ToxiTrace improves classification accuracy and toxic span extraction while preserving efficient encoder-based inference and producing more coherent, human-readable explanations. We have released the model at this https URL.
>
---
#### [new 039] Continuous Knowledge Metabolism: Generating Scientific Hypotheses from Evolving Literature
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学假设生成任务，旨在解决如何跟踪知识演变并生成有效假设的问题。工作包括提出CKM框架，通过增量处理提升预测效果，并分析不同处理方式对假设质量的影响。**

- **链接: [https://arxiv.org/pdf/2604.12243](https://arxiv.org/pdf/2604.12243)**

> **作者:** Jinkai Tao; Yubo Wang; Xiaoyu Liu; Menglin Yang
>
> **备注:** 32 pages, 6 figures
>
> **摘要:** Scientific hypothesis generation requires tracking how knowledge evolves, not just what is currently known. We introduce Continuous Knowledge Metabolism (CKM), a framework that processes scientific literature through sliding time windows and incrementally updates a structured knowledge base as new findings arrive. We present CKM-Lite, an efficient variant that achieves strong predictive coverage through incremental accumulation, outperforming batch processing on hit rate (+2.8%, p=0.006), hypothesis yield (+3.6, p<0.001), and best-match alignment (+0.43, p<0.001) while reducing token cost by 92%. To understand what drives these differences, we develop CKM-Full, an instrumented variant that categorizes each new finding as novel, confirming, or contradicting, detects knowledge change signals, and conditions hypothesis generation on the full evolution trajectory. Analyzing 892 hypotheses generated by CKM-Full across 50 research topics, alongside parallel runs of the other variants, we report four empirical observations: (1) incremental processing outperforms batch baseline across predictive and efficiency metrics; (2) change-aware instrumentation is associated with higher LLM-judged novelty (Cohen's d=3.46) but lower predictive coverage, revealing a quality-coverage trade-off; (3) a field's trajectory stability is associated with hypothesis success (r=-0.28, p=0.051), suggesting boundary conditions for literature-based prediction; (4) knowledge convergence signals are associated with nearly 5x higher hit rate than contradiction signals, pointing to differential predictability across change types. These findings suggest that the character of generated hypotheses is shaped not only by how much literature is processed, but also by how it is processed. They further indicate that evaluation frameworks must account for the quality-coverage trade-off rather than optimize for a single metric.
>
---
#### [new 040] PolicyLLM: Towards Excellent Comprehension of Public Policy for Large Language Models
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于政策理解任务，旨在提升大语言模型对公共政策的 comprehension 能力。提出 PolicyBench 基准和 PolicyMoE 模型，解决 LLM 在政策推理与应用中的不足。**

- **链接: [https://arxiv.org/pdf/2604.12995](https://arxiv.org/pdf/2604.12995)**

> **作者:** Han Bao; Penghao Zhang; Yue Huang; Zhengqing Yuan; Yanchi Ru; Rui Su; Yujun Zhou; Xiangqi Wang; Kehan Guo; Nitesh V Chawla; Yanfang Ye; Xiangliang Zhang
>
> **备注:** Accepted by ACL 2026 findings
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into real-world decision-making, including in the domain of public policy. Yet, their ability to comprehend and reason about policy-related content remains underexplored. To fill this gap, we present \textbf{\textit{PolicyBench}}, the first large-scale cross-system benchmark (US-China) evaluating policy comprehension, comprising 21K cases across a broad spectrum of policy areas, capturing the diversity and complexity of real-world governance. Following Bloom's taxonomy, the benchmark assesses three core capabilities: (1) \textbf{Memorization}: factual recall of policy knowledge, (2) \textbf{Understanding}: conceptual and contextual reasoning, and (3) \textbf{Application}: problem-solving in real-life policy scenarios. Building on this benchmark, we further propose \textbf{\textit{PolicyMoE}}, a domain-specialized Mixture-of-Experts (MoE) model with expert modules aligned to each cognitive level. The proposed models demonstrate stronger performance on application-oriented policy tasks than on memorization or conceptual understanding, and yields the highest accuracy on structured reasoning tasks. Our results reveal key limitations of current LLMs in policy understanding and suggest paths toward more reliable, policy-focused models.
>
---
#### [new 041] Think Through Uncertainty: Improving Long-Form Generation Factuality via Reasoning Calibration
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，旨在解决长文本生成中的事实准确性问题。通过引入CURE框架，模型学习在声明层面进行不确定性推理，提升事实正确性与置信度校准。**

- **链接: [https://arxiv.org/pdf/2604.12046](https://arxiv.org/pdf/2604.12046)**

> **作者:** Xin Liu; Lu Wang
>
> **摘要:** Large language models (LLMs) often hallucinate in long-form generation. Existing approaches mainly improve factuality through post-hoc revision or reinforcement learning (RL) with correctness-based rewards, but they do not teach the model to estimate which parts of its generation are reliable. As a result, models may still state incorrect claims confidently in their responses. Recent advances in reasoning have significantly improved LLM performance, and have been leveraged to estimate confidence by incorporating calibration into RL objectives. However, existing approaches remain limited to a single scalar confidence for the entire response, which is insufficient for long-form generation where uncertainty varies across individual claims. To mitigate this problem, we propose CURE, a framework that improves long-form factuality by teaching LLMs to reason about uncertainty at the claim level. We first introduce a Claim-Aware Reasoning Protocol, which structures outputs into atomic claims paired with explicit confidence estimates. We then develop a multi-stage training pipeline that aligns model confidence with claims' correctness and then optimizes on factuality. The resulting calibrated confidence further enables selective prediction, allowing the model to abstain from uncertain claims at inference time. Experiments on four long-form factuality benchmarks show that CURE consistently improves factual accuracy over competitive supervised and RL baselines, while maintaining factual recall. In particular, it improves claim-level accuracy by up to 39.9% on Biography generation. These gains are accompanied by improved calibration, as reflected by a 16.0% increase in AUROC on FactBench.
>
---
#### [new 042] GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OCR任务，旨在解决现有模型在多语言脚本上的泛化能力不足问题。通过构建涵盖100+ Unicode脚本的基准测试，评估模型表现并揭示其依赖预训练覆盖的问题。**

- **链接: [https://arxiv.org/pdf/2604.12978](https://arxiv.org/pdf/2604.12978)**

> **作者:** Amir Hossein Kargaran; Nafiseh Nikeghbal; Jana Diesner; François Yvon; Hinrich Schütze
>
> **摘要:** Optical character recognition (OCR) has advanced rapidly with the rise of vision-language models, yet evaluation has remained concentrated on a small cluster of high- and mid-resource scripts. We introduce GlotOCR Bench, a comprehensive benchmark evaluating OCR generalization across 100+ Unicode scripts. Our benchmark comprises clean and degraded image variants rendered from real multilingual texts. Images are rendered using fonts from the Google Fonts repository, shaped with HarfBuzz and rasterized with FreeType, supporting both LTR and RTL scripts. Samples of rendered images were manually reviewed to verify correct rendering across all scripts. We evaluate a broad suite of open-weight and proprietary vision-language models and find that most perform well on fewer than ten scripts, and even the strongest frontier models fail to generalize beyond thirty scripts. Performance broadly tracks script-level pretraining coverage, suggesting that current OCR systems rely on language model pretraining as much as on visual recognition. Models confronted with unfamiliar scripts either produce random noise or hallucinate characters from similar scripts they already know. We release the benchmark and pipeline for reproducibility. Pipeline Code: this https URL, Benchmark: this https URL.
>
---
#### [new 043] Agentic Insight Generation in VSM Simulations
- **分类: cs.CL**

- **简介: 该论文属于价值流仿真分析任务，旨在解决从复杂仿真中提取有效洞察的难题。通过提出一种解耦的两阶段代理架构，提升数据源选择与多跳推理能力。**

- **链接: [https://arxiv.org/pdf/2604.12421](https://arxiv.org/pdf/2604.12421)**

> **作者:** Micha Selak; Dirk Krechel; Adrian Ulges; Sven Spieckermann; Niklas Stoehr; Andreas Loehr
>
> **摘要:** Extracting actionable insights from complex value stream map simulations can be challenging, time-consuming, and error-prone. Recent advances in large language models offer new avenues to support users with this task. While existing approaches excel at processing raw data to gain information, they are structurally unfit to pick up on subtle situational differences needed to distinguish similar data sources in this domain. To address this issue, we propose a decoupled, two-step agentic architecture. By separating orchestration from data analysis, the system leverages progressive data discovery infused with domain expert knowledge. This architecture allows the orchestration to intelligently select data sources and perform multi-hop reasoning across data structures while maintaining a slim internal context. Results from multiple state-of-the-art large language models demonstrate the framework's viability: with top-tier models achieving accuracies of up to 86% and demonstrating high robustness across evaluation runs.
>
---
#### [new 044] ReasonXL: Shifting LLM Reasoning Language Without Sacrificing Performance
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在非英语场景中推理语言不匹配的问题。通过构建多语言推理语料库并进行微调，使模型能在目标语言中有效推理，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.12378](https://arxiv.org/pdf/2604.12378)**

> **作者:** Daniil Gurgurov; Tom Röhr; Sebastian von Rohrscheidt; Josef van Genabith; Alexander Löser; Simon Ostermann
>
> **备注:** Under review
>
> **摘要:** Despite advances in multilingual capabilities, most large language models (LLMs) remain English-centric in their training and, crucially, in their production of reasoning traces. Even when tasked with non-English problems, these models predominantly reason in English, creating a fundamental mismatch for non-English usage scenarios. We address this disparity directly with three contributions. (i) We introduce ReasonXL, the first large-scale parallel corpus of cross-domain reasoning traces spanning five European languages (English, German, French, Italian, and Spanish), with over two million aligned samples per language, each comprising prompts, reasoning traces, and final outputs, enabling direct supervision of language-specific reasoning. (ii) Using ReasonXL, we demonstrate that LLMs can be adapted to reason entirely in a desired target language, using a simple two-stage pipeline of supervised fine-tuning (SFT) followed by reinforcement learning with verifiable rewards (RLVR). The resulting models match or exceed baseline performance, with minimal loss in general knowledge and broadly preserved cross-lingual transfer. (iii) We conduct an extensive representational analysis of the adaptation and find a clear functional division across model depth: early layers contain an activation bottleneck that causally determines language identity, while upper layers concentrate the weight and activation changes driven by adaptation. We further find that RLVR achieves greater behavioral divergence from the base model with smaller parameter updates than SFT, suggesting a more efficient representational rerouting despite much smaller weight updates.
>
---
#### [new 045] Cooperative Memory Paging with Keyword Bookmarks for Long-Horizon LLM Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长周期大语言模型对话任务，解决上下文溢出后内容恢复问题。提出协作分页机制，用关键词书签替代被驱逐内容，并通过召回工具按需恢复。**

- **链接: [https://arxiv.org/pdf/2604.12376](https://arxiv.org/pdf/2604.12376)**

> **作者:** Ziyang Liu
>
> **备注:** 16 pages, 10 figures, 16 tables
>
> **摘要:** When LLM conversations grow beyond the context window, old content must be evicted -- but how does the model recover it when needed? We propose cooperative paging: evicted segments are replaced with minimal keyword bookmarks ([pN:keywords], ~8-24 tokens each), and the model is given a recall() tool to retrieve full content on demand. On the LoCoMo benchmark (10 real multi-session conversations, 300+ turns), cooperative paging achieves the highest answer quality among six methods -- outperforming truncation, BM25, word-overlap retrieval, a search-tool baseline, and full context -- on four models (GPT-4o-mini, DeepSeek-v3.2, Claude Haiku, GLM-5), confirmed by four independent LLM judges ($p=0.017$, paired bootstrap). We then study the paging design space with a 5x4 ablation over boundary strategies and eviction policies (3,176 synthetic probes, 1,600 LoCoMo probes). Key findings: (1) coarse fixed-size pages (fixed_20) reach 96.7% while content-aware topic_shift collapses to 56.7%; (2) eviction policy choice is data-dependent (FIFO best on synthetic, LFU on LoCoMo); (3) two bookmark generation strategies improve over the heuristic baseline (+4.4 and +8.7 E2E points); (4) the remaining bottleneck is bookmark discrimination -- the model triggers recall() 96% of the time but selects the correct page only 57% when bookmarks are insufficiently distinctive. Keyword specificity alone accounts for a 25 percentage point accuracy difference.
>
---
#### [new 046] Token-Level Policy Optimization: Linking Group-Level Rewards to Token-Level Aggregation via Sequence-Level Likelihood
- **分类: cs.CL**

- **简介: 该论文提出TEPO框架，解决大语言模型在链式推理中的稀疏奖励问题，通过序列级似然和KL散度约束提升训练稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2604.12736](https://arxiv.org/pdf/2604.12736)**

> **作者:** Xingyu Lin; Yilin Wen; Du Su; Jinchang Hou; En Wang; Wenbin Liu; Chenfu Bao; Zhonghou Lv
>
> **摘要:** Group Relative Policy Optimization (GRPO) has significantly advanced the reasoning ability of large language models (LLMs), particularly in their mathemat ical reasoning performance. However, GRPO and related entropy regularization methods still struggle with token-level sparse-rewards, which is an inherent chal lenge in chain-of-thought (CoT) reasoning. These approaches often rely on undifferen tiated token-level entropy regularization, which easily leads to entropy collapse or model degradation under sparse token rewards. In this work, we propose TEPO, a novel token-level framework that (1) leverages sequence-level likelihood to link group-level rewards with individual tokens via token-level aggregation, and (2) introduces a token-level KL-Divergence mask constraint that targets tokens with positive advantages and decreasing entropy to mitigate abrupt policy updates. Experiments demonstrate that TEPO not only achieves state-of-the-art performance on mathematical reasoning benchmarks but also markedly enhances training stability, reducing convergence time by 50% compared with GRPO/DAPO.
>
---
#### [new 047] CascadeDebate: Multi-Agent Deliberation for Cost-Aware LLM Cascades
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CascadeDebate，用于优化LLM级联系统，解决模型在不确定查询下的低效升级问题，通过多智能体协商提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12262](https://arxiv.org/pdf/2604.12262)**

> **作者:** Raeyoung Chang; Dongwook Kwon; Jisoo Lee; Nikhil Verma
>
> **备注:** 12 pages, 6 figures, 4 tables, 1 algorithm
>
> **摘要:** Cascaded LLM systems coordinate models of varying sizes with human experts to balance accuracy, cost, and abstention under uncertainty. However, single-model tiers at each stage often struggle with ambiguous queries, triggering premature escalations to costlier models or experts due to under-confidence and inefficient compute scaling. CascadeDebate addresses this gap by inserting multi-agent deliberation directly at each tier's escalation boundary. Confidence-based routers activate lightweight agent ensembles only for uncertain cases, enabling consensus-driven resolution of ambiguities internally without invoking higher-cost upgrades. Our unified architecture alternates single-model inference with selective multi-agent deliberation across model scales, culminating in human experts as the final fallback. This design scales test-time compute dynamically according to query difficulty. Across five benchmarks spanning science, medicine, and general knowledge, CascadeDebate outperforms strong single-model cascades and standalone multi-agent systems by up to 26.75 percent. An online threshold optimizer proves essential, boosting accuracy by 20.98 to 52.33 percent relative improvement over fixed policies and enabling elastic adaptation to real-world distributions.
>
---
#### [new 048] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing
- **分类: cs.CL**

- **简介: 该论文提出FABLE，解决模型编辑中细粒度事实访问不足的问题，通过分阶段策略实现事实注入与文本生成解耦，提升问答性能。**

- **链接: [https://arxiv.org/pdf/2604.12559](https://arxiv.org/pdf/2604.12559)**

> **作者:** Peng Wang; Biyu Zhou; Xuehai Tang; Jizhong Han; Songlin Hu
>
> **备注:** ACL 2026 findings
>
> **摘要:** Unstructured model editing aims to update models with real-world text, yet existing methods often memorize text holistically without reliable fine-grained fact access. To address this, we propose FABLE, a hierarchical framework that decouples fine-grained fact injection from holistic text generation. FABLE follows a two-stage, fact-first strategy: discrete facts are anchored in shallow layers, followed by minimal updates to deeper layers to produce coherent text. This decoupling resolves the mismatch between holistic recall and fine-grained fact access, reflecting the unidirectional Transformer flow in which surface-form generation amplifies rather than corrects underlying fact representations. We also introduce UnFine, a diagnostic benchmark with fine-grained question-answer pairs and fact-level metrics for systematic evaluation. Experiments show that FABLE substantially improves fine-grained question answering while maintaining state-of-the-art holistic editing performance. Our code is publicly available at this https URL.
>
---
#### [new 049] From Myopic Selection to Long-Horizon Awareness: Sequential LLM Routing for Multi-Turn Dialogue
- **分类: cs.CL**

- **简介: 该论文属于多轮对话任务，旨在解决多轮对话中LLM路由效果不佳的问题。通过引入长视野的序列路由方法，提升整体性能与效率。**

- **链接: [https://arxiv.org/pdf/2604.12385](https://arxiv.org/pdf/2604.12385)**

> **作者:** Jiarui Zhang; Xiangyu Liu; Yong Hu; Chaoyue Niu; Hang Zeng; Shaojie Tang; Fan Wu; Guihai Chen
>
> **摘要:** Multi-turn dialogue is the predominant form of interaction with large language models (LLMs). While LLM routing is effective in single-turn settings, existing methods fail to maximize cumulative performance in multi-turn dialogue due to interaction dynamics and delayed rewards. To address this challenge, we move from myopic, single-turn selection to long-horizon sequential routing for multi-turn dialogue. Accordingly, we propose DialRouter, which first performs MCTS to explore dialogue branches induced by different LLM selections and collect trajectories with high cumulative rewards. DialRouter then learns a lightweight routing policy from search-derived data, augmented with retrieval-based future state approximation, enabling multi-turn routing without online search. Experiments on both open-domain and domain-specific dialogue tasks across diverse candidate sets of both open-source and closed-source LLMs demonstrate that DialRouter significantly outperforms single LLMs and existing routing baselines in task success rate, while achieving a superior performance-cost trade-off when combined with a cost-aware reward.
>
---
#### [new 050] Learning Chain Of Thoughts Prompts for Predicting Entities, Relations, and even Literals on Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱链接预测任务，旨在解决KGE模型在处理未见实体、关系和文字值时的不足。通过引入RALP方法，利用提示学习提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12651](https://arxiv.org/pdf/2604.12651)**

> **作者:** Alkid Baci; Luke Friedrichs; Caglar Demir; N'Dah Jean Kouagou; Axel-Cyrille Ngonga Ngomo
>
> **摘要:** Knowledge graph embedding (KGE) models perform well on link prediction but struggle with unseen entities, relations, and especially literals, limiting their use in dynamic, heterogeneous graphs. In contrast, pretrained large language models (LLMs) generalize effectively through prompting. We reformulate link prediction as a prompt learning problem and introduce RALP, which learns string-based chain-of-thought (CoT) prompts as scoring functions for triples. Using Bayesian Optimization through MIPRO algorithm, RALP identifies effective prompts from fewer than 30 training examples without gradient access. At inference, RALP predicts missing entities, relations or whole triples and assigns confidence scores based on the learned prompt. We evaluate on transductive, numerical, and OWL instance retrieval benchmarks. RALP improves state-of-the-art KGE models by over 5% MRR across datasets and enhances generalization via high-quality inferred triples. On OWL reasoning tasks with complex class expressions (e.g., $\exists this http URL$, $\geq 5 \; this http URL$), it achieves over 88% Jaccard similarity. These results highlight prompt-based LLM reasoning as a flexible alternative to embedding-based methods. We release our implementation, training, and evaluation pipeline as open source: this https URL .
>
---
#### [new 051] InsightFlow: LLM-Driven Synthesis of Patient Narratives for Mental Health into Causal Models
- **分类: cs.CL**

- **简介: 该论文提出InsightFlow，利用大语言模型从治疗对话中自动生成符合5P框架的因果图，解决临床案例构建耗时且主观的问题。任务属于医疗文本到因果模型的生成。**

- **链接: [https://arxiv.org/pdf/2604.12721](https://arxiv.org/pdf/2604.12721)**

> **作者:** Shreya Gupta; Prottay Kumar Adhikary; Bhavyaa Dave; Salam Michael Singh; Aniket Deroy; Tanmoy Chakraborty
>
> **摘要:** Clinical case formulation organizes patient symptoms and psychosocial factors into causal models, often using the 5P framework. However, constructing such graphs from therapy transcripts is time consuming and varies across clinicians. We present InsightFlow, an LLM based approach that automatically generates 5P aligned causal graphs from patient-therapist dialogues. Using 46 psychotherapy intake transcripts annotated by clinical experts, we evaluate LLM generated graphs against human formulations using structural (NetSimile), semantic (embedding similarity), and expert rated clinical criteria. The generated graphs show structural similarity comparable to inter annotator agreement and high semantic alignment with human graphs. Expert evaluations rate the outputs as moderately complete, consistent, and clinically useful. While LLM graphs tend to form more interconnected structures compared to the chain like patterns of human graphs, overall complexity and content coverage are similar. These results suggest that LLMs can produce clinically meaningful case formulation graphs within the natural variability of expert practice. InsightFlow highlights the potential of automated causal modeling to augment clinical workflows, with future work needed to improve temporal reasoning and reduce redundancy.
>
---
#### [new 052] One Token Away from Collapse: The Fragility of Instruction-Tuned Helpfulness
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究指令调优大模型在受到简单约束时的脆弱性，揭示其响应能力下降的问题。任务属于自然语言处理中的模型鲁棒性分析，通过实验和机制分析验证了指令调优带来的系统性风险。**

- **链接: [https://arxiv.org/pdf/2604.13006](https://arxiv.org/pdf/2604.13006)**

> **作者:** Erfan Baghaei Potraghloo; Seyedarmin Azizi; Souvik Kundu; Massoud Pedram
>
> **摘要:** Instruction-tuned large language models produce helpful, structured responses, but how robust is this helpfulness when trivially constrained? We show that simple lexical constraints (banning a single punctuation character or common word) cause instruction-tuned LLMs to collapse their responses, losing 14--48% of comprehensiveness in pairwise evaluation across three open-weight model families and one closed-weight model (GPT-4o-mini). The baseline response is preferred in 77--100% of 1,920 pairwise comparisons judged by GPT-4o-mini and GPT-4o. Notably, GPT-4o-mini suffers 31% comprehensiveness loss (99% baseline win rate), demonstrating that the fragility extends to commercially deployed closed-weight models, contrary to prior findings on format-level constraints. Through mechanistic analysis, we identify this as a planning failure: two-pass generation (free generation followed by constrained rewriting) recovers 59--96% of response length, and linear probes on prompt representations predict response length with $R^2 = 0.51$--$0.93$ before generation begins, with $R^2$ tracking collapse severity across models. The same probes yield negative $R^2$ on base models, confirming that instruction tuning creates the representational structure encoding the collapse decision. Crucially, base models show no systematic collapse under identical constraints, with effects that are small, noisy, and bidirectional, demonstrating that instruction tuning creates this fragility by coupling task competence to narrow surface-form templates. The effect replicates on MT-Bench across all eight task categories. We further show that standard independent LLM-as-judge evaluation detects only a 3.5% average quality drop where pairwise evaluation reveals 23%, exposing a methodological blind spot in how constrained generation is assessed.
>
---
#### [new 053] KoCo: Conditioning Language Model Pre-training on Knowledge Coordinates
- **分类: cs.CL**

- **简介: 该论文提出KoCo方法，将文档映射为三维语义坐标，用于语言模型预训练，以提升模型的上下文理解能力，解决信息 contextualization 不足的问题。**

- **链接: [https://arxiv.org/pdf/2604.12397](https://arxiv.org/pdf/2604.12397)**

> **作者:** Yudong Li; Jiawei Cai; Linlin Shen
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Standard Large Language Model (LLM) pre-training typically treats corpora as flattened token sequences, often overlooking the real-world context that humans naturally rely on to contextualize information. To bridge this gap, we introduce Knowledge Coordinate Conditioning (KoCo), a simple method that maps every document into a three-dimensional semantic coordinate. By prepending these coordinates as textual prefixes for pre-training, we aim to equip the model with explicit contextual awareness to learn the documents within the real-world knowledge structure. Experiment results demonstrate that KoCo significantly enhances performance across 10 downstream tasks and accelerates pre-training convergence by approximately 30\%. Furthermore, our analysis indicates that explicitly modeling knowledge coordinates helps the model distinguish stable facts from noise, effectively mitigating hallucination in generated outputs.
>
---
#### [new 054] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于音频语言模型任务，解决AudioLLM在细粒度声学感知上的不足。通过提出统一音频框架UAS，提升声学理解能力，同时保持推理性能。**

- **链接: [https://arxiv.org/pdf/2604.12506](https://arxiv.org/pdf/2604.12506)**

> **作者:** Linhao Zhang; Yuhan Song; Aiwei Liu; Chuhan Wu; Sijun Zhang; Wei Jia; Yuan Liu; Houfeng Wang; Xiao Zhou
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Recent Audio Large Language Models (AudioLLMs) exhibit a striking performance inversion: while excelling at complex reasoning tasks, they consistently underperform on fine-grained acoustic perception. We attribute this gap to a fundamental limitation of ASR-centric training, which provides precise linguistic targets but implicitly teaches models to suppress paralinguistic cues and acoustic events as noise. To address this, we propose Unified Audio Schema (UAS), a holistic and structured supervision framework that organizes audio information into three explicit components -- Transcription, Paralinguistics, and Non-linguistic Events -- within a unified JSON format. This design achieves comprehensive acoustic coverage without sacrificing the tight audio-text alignment that enables reasoning. We validate the effectiveness of this supervision strategy by applying it to both discrete and continuous AudioLLM architectures. Extensive experiments on MMSU, MMAR, and MMAU demonstrate that UAS-Audio yields consistent improvements, boosting fine-grained perception by 10.9% on MMSU over the same-size state-of-the-art models while preserving robust reasoning capabilities. Our code and model are publicly available at this https URL.
>
---
#### [new 055] Robust Explanations for User Trust in Enterprise NLP Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的可解释性研究，旨在解决企业级NLP系统中用户信任问题。通过构建评估框架，分析不同模型在噪声下的解释稳定性，发现解码器模型表现更优。**

- **链接: [https://arxiv.org/pdf/2604.12069](https://arxiv.org/pdf/2604.12069)**

> **作者:** Guilin Zhang; Kai Zhao; Jeffrey Friedman; Xu Chu; Amine Anoun; Jerry Ting
>
> **摘要:** Robust explanations are increasingly required for user trust in enterprise NLP, yet pre-deployment validation is difficult in the common case of black-box deployment (API-only access) where representation-based explainers are infeasible and existing studies provide limited guidance on whether explanations remain stable under real user noise, especially when organizations migrate from encoder classifiers to decoder LLMs. To close this gap, we propose a unified black-box robustness evaluation framework for token-level explanations based on leave-one-out occlusion, and operationalize explanation robustness with top-token flip rate under realistic perturbations (swap, deletion, shuffling, and back-translation) at multiple severity levels. Using this protocol, we conduct a systematic cross-architecture comparison across three benchmark datasets and six models spanning encoder and decoder families (BERT, RoBERTa, Qwen 7B/14B, Llama 8B/70B; 64,800 cases). We find that decoder LLMs produce substantially more stable explanations than encoder baselines (73% lower flip rates on average), and that stability improves with model scale (44% gain from 7B to 70B). Finally, we relate robustness improvements to inference cost, yielding a practical cost-robustness tradeoff curve that supports model and explanation selection prior to deployment in compliance-sensitive applications.
>
---
#### [new 056] Thought-Retriever: Don't Just Retrieve Raw Data, Retrieve Thoughts for Memory-Augmented Agentic Systems
- **分类: cs.CL; cs.IR**

- **简介: 该论文提出Thought-Retriever，解决LLM在处理长文本时的上下文限制问题。通过检索和利用过去思考过程，增强模型记忆能力，提升问答性能。**

- **链接: [https://arxiv.org/pdf/2604.12231](https://arxiv.org/pdf/2604.12231)**

> **作者:** Tao Feng; Pengrui Han; Guanyu Lin; Ge Liu; Jiaxuan You
>
> **摘要:** Large language models (LLMs) have transformed AI research thanks to their powerful internal capabilities and knowledge. However, existing LLMs still fail to effectively incorporate the massive external knowledge when interacting with the world. Although retrieval-augmented LLMs are proposed to mitigate the issue, they are still fundamentally constrained by the context length of LLMs, as they can only retrieve top-K raw data chunks from the external knowledge base which often consists of millions of data chunks. Here we propose Thought-Retriever, a novel model-agnostic algorithm that helps LLMs generate output conditioned on arbitrarily long external data, without being constrained by the context length or number of retrieved data chunks. Our key insight is to let an LLM fully leverage its intermediate responses generated when solving past user queries (thoughts), filtering meaningless and redundant thoughts, organizing them in thought memory, and retrieving the relevant thoughts when addressing new queries. This effectively equips LLM-based agents with a self-evolving long-term memory that grows more capable through continuous interaction. Besides algorithmic innovation, we further meticulously prepare a novel benchmark, AcademicEval, which requires an LLM to faithfully leverage ultra-long context to answer queries based on real-world academic papers. Extensive experiments on AcademicEval and two other public datasets validate that Thought-Retriever remarkably outperforms state-of-the-art baselines, achieving an average increase of at least 7.6% in F1 score and 16% in win rate across various tasks. More importantly, we further demonstrate two exciting findings: (1) Thought-Retriever can indeed help LLM self-evolve after solving more user queries; (2) Thought-Retriever learns to leverage deeper thoughts to answer more abstract user queries.
>
---
#### [new 057] Transforming External Knowledge into Triplets for Enhanced Retrieval in RAG of LLMs
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决RAG中因冗余文本导致的检索效率与生成质量下降问题。通过构建结构化三元组提升检索精度与上下文效率。**

- **链接: [https://arxiv.org/pdf/2604.12610](https://arxiv.org/pdf/2604.12610)**

> **作者:** Xudong Wang; Chaoning Zhang; Qigan Sun; Zhenzhen Huang; Chang Lu; Sheng Zheng; Zeyu Ma; Caiyan Qin; Yang Yang; Hengtao Shen
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates hallucination in large language models (LLMs) by incorporating external knowledge during generation. However, the effectiveness of RAG depends not only on the design of the retriever and the capacity of the underlying model, but also on how retrieved evidence is structured and aligned with the query. Existing RAG approaches typically retrieve and concatenate unstructured text fragments as context, which often introduces redundant or weakly relevant information. This practice leads to excessive context accumulation, reduced semantic alignment, and fragmented reasoning chains, thereby degrading generation quality while increasing token consumption. To address these challenges, we propose Tri-RAG, a structured triplet-based retrieval framework that improves retrieval efficiency through reasoning-aligned context construction. Tri-RAG automatically transforms external knowledge from natural language into standardized structured triplets consisting of Condition, Proof, and Conclusion, explicitly capturing logical relations among knowledge fragments using lightweight prompt-based adaptation with frozen model parameters. Building on this representation, the triplet head Condition is treated as an explicit semantic anchor for retrieval and matching, enabling precise identification of query-relevant knowledge units without directly concatenating lengthy raw texts. As a result, Tri-RAG achieves a favorable balance between retrieval accuracy and context token efficiency. Experimental results across multiple benchmark datasets demonstrate that Tri-RAG significantly improves retrieval quality and reasoning efficiency, while producing more stable generation behavior and more efficient resource utilization in complex reasoning scenarios.
>
---
#### [new 058] Teaching LLMs Human-Like Editing of Inappropriate Argumentation via Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于文本编辑任务，旨在解决LLMs编辑不自然的问题。通过强化学习，使LLMs生成更接近人类的编辑策略，提升论点适当性。**

- **链接: [https://arxiv.org/pdf/2604.12770](https://arxiv.org/pdf/2604.12770)**

> **作者:** Timon Ziegenbein; Maja Stahl; Henning Wachsmuth
>
> **摘要:** Editing human-written text has become a standard use case of large language models (LLMs), for example, to make one's arguments more appropriate for a discussion. Comparing human to LLM-generated edits, however, we observe a mismatch in editing strategies: While LLMs often perform multiple scattered edits and tend to change meaning notably, humans rather encapsulate dependent changes in self-contained, meaning-preserving edits. In this paper, we present a reinforcement learning approach that teaches LLMs human-like editing to improve the appropriateness of arguments. Our approach produces self-contained sentence-level edit suggestions that can be accepted or rejected independently. We train the approach using group relative policy optimization with a multi-component reward function that jointly optimizes edit-level semantic similarity, fluency, and pattern conformity as well as argument-level appropriateness. In automatic and human evaluation, it outperforms competitive baselines and the state of the art in human-like editing, with multi-round editing achieving appropriateness close to full rewriting.
>
---
#### [new 059] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM生成文本的时序结构问题。通过对比人类与LLM写作轨迹，发现LLM存在时间扁平化现象，揭示其在长期文本建模中的局限性。**

- **链接: [https://arxiv.org/pdf/2604.12097](https://arxiv.org/pdf/2604.12097)**

> **作者:** Zhanwei Cao; YeoJin Go; Yifan Hu; Shanu Sushmita
>
> **备注:** 25 pages, 6 figures. To appear in Findings of ACL 2026
>
> **摘要:** Large language models (LLMs) are increasingly used in daily applications, from content generation to code writing, where each interaction treats the model as stateless, generating responses independently without memory. Yet human writing is inherently longitudinal: authors' styles and cognitive states evolve across months and years. This raises a central question: can LLMs reproduce such temporal structure across extended time periods? We construct and publicly release a longitudinal dataset of 412 human authors and 6,086 documents spanning 2012--2024 across three domains (academic abstracts, blogs, news) and compare them to trajectories generated by three representative LLMs under standard and history-conditioned generation settings. Using drift and variance-based metrics over semantic, lexical, and cognitive-emotional representations, we find temporal flattening in LLM-generated text. LLMs produce greater lexical diversity but exhibit substantially reduced semantic and cognitive-emotional drift relative to humans. These differences are highly predictive: temporal variability patterns alone achieve 94% accuracy and 98% ROC-AUC in distinguishing human from LLM trajectories. Our results demonstrate that temporal flattening persists regardless of whether LLMs generate independently or with access to incremental history, revealing a fundamental property of current deployment paradigms. This gap has direct implications for applications requiring authentic temporal structure, such as synthetic training data and longitudinal text modeling.
>
---
#### [new 060] Universal NER v2: Towards a Massively Multilingual Named Entity Recognition Benchmark
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别（NER）任务，旨在解决多语言基准数据稀缺的问题。通过构建标准化、跨语言的NER数据集，提升多语言大模型的评估能力。**

- **链接: [https://arxiv.org/pdf/2604.12744](https://arxiv.org/pdf/2604.12744)**

> **作者:** Terra Blevins; Stephen Mayhew; Marek Šuppa; Hila Gonen; Shachar Mirkin; Vasile Pais; Kaja Dobrovoljc; Voula Giouli; Jun Kevin; Eugene Jang; Eungseo Kim; Jeongyeon Seo; Xenophon Gialis; Yuval Pinter
>
> **备注:** LREC 2026
>
> **摘要:** While multilingual language models promise to bring the benefits of LLMs to speakers of many languages, gold-standard evaluation benchmarks in most languages to interrogate these assumptions remain scarce. The Universal NER project, now entering its fourth year, is dedicated to building gold-standard multilingual Named Entity Recognition (NER) benchmark datasets. Inspired by existing massively multilingual efforts for other core NLP tasks (e.g., Universal Dependencies), the project uses a general tagset and thorough annotation guidelines to collect standardized, cross-lingual annotations of named entity spans. The first installment (UNER v1) was released in 2024, and the project has continued and expanded since then, with various organizers, annotators, and collaborators in an active community.
>
---
#### [new 061] SCRIPT: A Subcharacter Compositional Representation Injection Module for Korean Pre-Trained Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SCRIPT模块，解决韩国语模型未充分捕捉字符内部结构的问题。通过注入子字符组合知识，提升模型在NLU和NLG任务中的表现。**

- **链接: [https://arxiv.org/pdf/2604.12377](https://arxiv.org/pdf/2604.12377)**

> **作者:** SungHo Kim; Juhyeong Park; Eda Atalay; SangKeun Lee
>
> **备注:** Accepted at ACL 2026 Findings
>
> **摘要:** Korean is a morphologically rich language with a featural writing system in which each character is systematically composed of subcharacter units known as Jamo. These subcharacters not only determine the visual structure of Korean but also encode frequent and linguistically meaningful morphophonological processes. However, most current Korean language models (LMs) are based on subword tokenization schemes, which are not explicitly designed to capture the internal compositional structure of characters. To address this limitation, we propose SCRIPT, a model-agnostic module that injects subcharacter compositional knowledge into Korean PLMs. SCRIPT allows to enhance subword embeddings with structural granularity, without requiring architectural changes or additional pre-training. As a result, SCRIPT enhances all baselines across various Korean natural language understanding (NLU) and generation (NLG) tasks. Moreover, beyond performance gains, detailed linguistic analyses show that SCRIPT reshapes the embedding space in a way that better captures grammatical regularities and semantically cohesive variations. Our code is available at this https URL.
>
---
#### [new 062] When Does Data Augmentation Help? Evaluating LLM and Back-Translation Methods for Hausa and Fongbe NLP
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究数据增强对低资源非洲语言NLP任务的效果，评估LLM生成和回译方法在Hausa和Fongbe的命名实体识别和词性标注中的应用。**

- **链接: [https://arxiv.org/pdf/2604.12540](https://arxiv.org/pdf/2604.12540)**

> **作者:** Mahounan Pericles Adjovi; Roald Eiselen; Prasenjit Mitra
>
> **备注:** 13 pages, 6 tables; previously submitted to KDD 2026
>
> **摘要:** Data scarcity limits NLP development for low-resource African languages. We evaluate two data augmentation methods -- LLM-based generation (Gemini 2.5 Flash) and back-translation (NLLB-200) -- for Hausa and Fongbe, two West African languages that differ substantially in LLM generation quality. We assess augmentation on named entity recognition (NER) and part-of-speech (POS) tagging using MasakhaNER 2.0 and MasakhaPOS benchmarks. Our results reveal that augmentation effectiveness depends on task type rather than language or LLM quality alone. For NER, neither method improves over baseline for either language; LLM augmentation reduces Hausa NER by 0.24% F1 and Fongbe NER by 1.81% F1. For POS tagging, LLM augmentation improves Fongbe by 0.33% accuracy, while back-translation improves Hausa by 0.17%; back-translation reduces Fongbe POS by 0.35% and has negligible effect on Hausa POS. The same LLM-generated synthetic data produces opposite effects across tasks for Fongbe -- hurting NER while helping POS -- suggesting task structure governs augmentation outcomes more than synthetic data quality. These findings challenge the assumption that LLM generation quality predicts augmentation success, and provide actionable guidance: data augmentation should be treated as a task-specific intervention rather than a universally beneficial preprocessing step.
>
---
#### [new 063] Self-Distillation Zero: Self-Revision Turns Binary Rewards into Dense Supervision
- **分类: cs.CL**

- **简介: 该论文提出SD-Zero方法，解决后训练阶段监督不足的问题。通过自蒸馏实现从二元奖励到密集监督的转换，提升模型性能。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2604.12002](https://arxiv.org/pdf/2604.12002)**

> **作者:** Yinghui He; Simran Kaur; Adithya Bhaskar; Yongjin Yang; Jiarui Liu; Narutatsu Ri; Liam Fowl; Abhishek Panigrahi; Danqi Chen; Sanjeev Arora
>
> **摘要:** Current post-training methods in verifiable settings fall into two categories. Reinforcement learning (RLVR) relies on binary rewards, which are broadly applicable and powerful, but provide only sparse supervision during training. Distillation provides dense token-level supervision, typically obtained from an external teacher or using high-quality demonstrations. Collecting such supervision can be costly or unavailable. We propose Self-Distillation Zero (SD-Zero), a method that is substantially more training sample-efficient than RL and does not require an external teacher or high-quality demonstrations. SD-Zero trains a single model to play two roles: a Generator, which produces an initial response, and a Reviser, which conditions on that response and its binary reward to produce an improved response. We then perform on-policy self-distillation to distill the reviser into the generator, using the reviser's token distributions conditioned on the generator's response and its reward as supervision. In effect, SD-Zero trains the model to transform binary rewards into dense token-level self-supervision. On math and code reasoning benchmarks with Qwen3-4B-Instruct and Olmo-3-7B-Instruct, SD-Zero improves performance by at least 10% over the base models and outperforms strong baselines, including Rejection Fine-Tuning (RFT), GRPO, and Self-Distillation Fine-Tuning (SDFT), under the same question set and training sample budget. Extensive ablation studies show two novel characteristics of our proposed algorithm: (a) token-level self-localization, where the reviser can identify the key tokens that need to be revised in the generator's response based on reward, and (b) iterative self-evolution, where the improving ability to revise answers can be distilled back into generation performance with regular teacher synchronization.
>
---
#### [new 064] Decoding by Perturbation: Mitigating MLLM Hallucinations via Dynamic Textual Perturbation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态语言模型任务，旨在解决模型推理中的幻觉问题。通过动态文本扰动方法，抑制语言先验带来的偏差，提升视觉 grounding 的稳定性。**

- **链接: [https://arxiv.org/pdf/2604.12424](https://arxiv.org/pdf/2604.12424)**

> **作者:** Sihang Jia; Shuliang Liu; Songbo Yang; Yibo Yan; Xin Zou; Xuming Hu
>
> **摘要:** Multimodal Large Language Models frequently suffer from inference hallucinations, partially stemming from language priors dominating visual evidence. Existing training-free mitigation methods either perturb the visual representation and deviate from the natural image distribution, or enforce intrusive manipulations that compromise the model's inherent generative fluency. We introduce a novel perspective that multimodal hallucination manifests as the hypersensitivity of visual grounding to textual phrasing during the decoding phase. Building on this insight, we propose Decoding by Perturbation (DeP), a training-free framework mitigating prior-induced hallucinations via controlled textual interventions. DeP employs a dynamic probe applying multi-level textual perturbations to elicit latent language priors. Leveraging attention variance, it enhances stable evidence regions while suppressing suspicious noise in the feature space. Furthermore, it constructs an interpretable prior drift direction using logits statistics to counteract probability biases from textual co-occurrences. Extensive experiments confirm DeP effectively reduces hallucinations and achieves superior performance across multiple benchmarks.
>
---
#### [new 065] Narrative over Numbers: The Identifiable Victim Effect and its Amplification Under Alignment and Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究大语言模型中的可识别受害者效应，探讨其在伦理决策中的表现与影响，属于人工智能伦理任务。**

- **链接: [https://arxiv.org/pdf/2604.12076](https://arxiv.org/pdf/2604.12076)**

> **作者:** Syed Rifat Raiyan
>
> **备注:** Under review, 49 pages, 20 figures, 11 tables
>
> **摘要:** The Identifiable Victim Effect (IVE) $-$ the tendency to allocate greater resources to a specific, narratively described victim than to a statistically characterized group facing equivalent hardship $-$ is one of the most robust findings in moral psychology and behavioural economics. As large language models (LLMs) assume consequential roles in humanitarian triage, automated grant evaluation, and content moderation, a critical question arises: do these systems inherit the affective irrationalities present in human moral reasoning? We present the first systematic, large-scale empirical investigation of the IVE in LLMs, comprising N=51,955 validated API trials across 16 frontier models spanning nine organizational lineages (Google, Anthropic, OpenAI, Meta, DeepSeek, xAI, Alibaba, IBM, and Moonshot). Using a suite of ten experiments $-$ porting and extending canonical paradigms from Small et al. (2007) and Kogut and Ritov (2005) $-$ we find that the IVE is prevalent but strongly modulated by alignment training. Instruction-tuned models exhibit extreme IVE (Cohen's d up to 1.56), while reasoning-specialized models invert the effect (down to d=-0.85). The pooled effect (d=0.223, p=2e-6) is approximately twice the single-victim human meta-analytic baseline (d$\approx$0.10) reported by Lee and Feeley (2016) $-$ and likely exceeds the overall human pooled effect by a larger margin, given that the group-victim human effect is near zero. Standard Chain-of-Thought (CoT) prompting $-$ contrary to its role as a deliberative corrective $-$ nearly triples the IVE effect size (from d=0.15 to d=0.41), while only utilitarian CoT reliably eliminates it. We further document psychophysical numbing, perfect quantity neglect, and marginal in-group/out-group cultural bias, with implications for AI deployment in humanitarian and ethical decision-making contexts.
>
---
#### [new 066] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步知识检索，在保持交互性的同时提升准确性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [new 067] Filtered Reasoning Score: Evaluating Reasoning Quality on a Model's Most-Confident Traces
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决仅靠准确率无法反映推理质量的问题。提出Filtered Reasoning Score（FRS）衡量模型推理质量，通过筛选高置信度轨迹提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2604.11996](https://arxiv.org/pdf/2604.11996)**

> **作者:** Manas Pathak; Xingyao Chen; Shuozhe Li; Amy Zhang; Liu Leqi
>
> **摘要:** Should we trust Large Language Models (LLMs) with high accuracy? LLMs achieve high accuracy on reasoning benchmarks, but correctness alone does not reveal the quality of the reasoning used to produce it. This highlights a fundamental limitation of outcome-based evaluation: models may arrive at correct answers through flawed reasoning, and models with substantially different reasoning capabilities can nevertheless exhibit similar benchmark accuracy, for example due to memorization or over-optimization. In this paper, we ask: given existing benchmarks, can we move beyond outcome-based evaluation to assess the quality of reasoning itself? We seek metrics that (1) differentiate models with similar accuracy and (2) are robust to variations in input prompts and generation configurations. To this end, we propose a reasoning score that evaluates reasoning traces along dimensions such as faithfulness, coherence, utility, and factuality. A remaining question is how to aggregate this score across multiple sampled traces. Naively averaging them is undesirable, particularly in long-horizon settings, where the number of possible trajectories grows rapidly, and low-confidence correct traces are more likely to be coincidental. To address this, we introduce the Filtered Reasoning Score (FRS), which computes reasoning quality using only the top-K% most confident traces. Evaluating with FRS, models that are indistinguishable under standard accuracy exhibit significant differences in reasoning quality. Moreover, models with higher FRS on one benchmark tend to perform better on other reasoning benchmarks, in both accuracy and reasoning quality. Together, these findings suggest that FRS complements accuracy by capturing a model's transferable reasoning capabilities. We open source our evaluation codebase: this https URL.
>
---
#### [new 068] Leveraging Weighted Syntactic and Semantic Context Assessment Summary (wSSAS) Towards Text Categorization Using LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在解决LLMs在文本分类中因注意力机制随机性和噪声敏感性导致的精度和可重复性问题。提出wSSAS框架，通过结构化分类和SNR优化提升分类效果。**

- **链接: [https://arxiv.org/pdf/2604.12049](https://arxiv.org/pdf/2604.12049)**

> **作者:** Shreeya Verma Kathuria; Nitin Mayande; Sharookh Daruwalla; Nitin Joglekar; Charles Weber
>
> **摘要:** The use of Large Language Models (LLMs) for reliable, enterprise-grade analytics such as text categorization is often hindered by the stochastic nature of attention mechanisms and sensitivity to noise that compromise their analytical precision and reproducibility. To address these technical frictions, this paper introduces the Weighted Syntactic and Semantic Context Assessment Summary (wSSAS), a deterministic framework designed to enforce data integrity on large-scale, chaotic datasets. We propose a two-phased validation framework that first organizes raw text into a hierarchical classification structure containing Themes, Stories, and Clusters. It then leverages a Signal-to-Noise Ratio (SNR) to prioritize high-value semantic features, ensuring the model's attention remains focused on the most representative data points. By incorporating this scoring mechanism into a Summary-of-Summaries (SoS) architecture, the framework effectively isolates essential information and mitigates background noise during data aggregation. Experimental results using Gemini 2.0 Flash Lite across diverse datasets - including Google Business reviews, Amazon Product reviews, and Goodreads Book reviews - demonstrate that wSSAS significantly improves clustering integrity and categorization accuracy. Our findings indicate that wSSAS reduces categorization entropy and provides a reproducible pathway for improving LLM based summaries based on a high-precision, deterministic process for large-scale text categorization.
>
---
#### [new 069] The Effect of Document Selection on Query-focused Text Analysis
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于文本分析任务，探讨文档选择策略对查询聚焦文本分析的影响，评估不同选择方法的效果，以提供实践指导。**

- **链接: [https://arxiv.org/pdf/2604.12099](https://arxiv.org/pdf/2604.12099)**

> **作者:** Sandesh S Rangreji; Mian Zhong; Anjalie Field
>
> **摘要:** Analyses of document collections often require selecting what data to analyze, as not all documents are relevant to a particular research question and computational constraints preclude analyzing all documents, yet little work has examined effects of selection strategy choices. We systematically evaluate seven selection methods (from random selection to hybrid retrieval) on outputs from four text analyses methods (LDA, BERTopic, TopicGPT, HiCode) over two datasets with 26 open-ended queries. Our evaluation reveals practice guidance: semantic or hybrid retrieval offer strong go-to approaches that avoid the pitfalls of weaker selection strategies and the unnecessary compute overhead of more complicated ones. Overall, our evaluation framework establishes data selection as a methodological decision, rather than a practical necessity, inviting the development of new strategies.
>
---
#### [new 070] M$^\star$: Every Task Deserves Its Own Memory Harness
- **分类: cs.PL; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出M$^\star$，解决通用记忆系统在不同任务中表现不佳的问题。通过进化方法自动优化任务专用记忆结构，提升多领域任务性能。**

- **链接: [https://arxiv.org/pdf/2604.11811](https://arxiv.org/pdf/2604.11811)**

> **作者:** Wenbo Pan; Shujie Liu; Xiangyang Zhou; Shiwei Zhang; Wanlu Shi; Mirror Xu; Xiaohua Jia
>
> **备注:** Preprint
>
> **摘要:** Large language model agents rely on specialized memory systems to accumulate and reuse knowledge during extended interactions. Recent architectures typically adopt a fixed memory design tailored to specific domains, such as semantic retrieval for conversations or skills reused for coding. However, a memory system optimized for one purpose frequently fails to transfer to others. To address this limitation, we introduce M$^\star$, a method that automatically discovers task-optimized memory harnesses through executable program evolution. Specifically, M$^\star$ models an agent memory system as a memory program written in Python. This program encapsulates the data Schema, the storage Logic, and the agent workflow Instructions. We optimize these components jointly using a reflective code evolution method; this approach employs a population-based search strategy and analyzes evaluation failures to iteratively refine the candidate programs. We evaluate M$^\star$ on four distinct benchmarks spanning conversation, embodied planning, and expert reasoning. Our results demonstrate that M$^\star$ improves performance over existing fixed-memory baselines robustly across all evaluated tasks. Furthermore, the evolved memory programs exhibit structurally distinct processing mechanisms for each domain. This finding indicates that specializing the memory mechanism for a given task explores a broad design space and provides a superior solution compared to general-purpose memory paradigms.
>
---
#### [new 071] UCS: Estimating Unseen Coverage for Improved In-Context Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出UCS方法，解决In-context Learning中演示样本选择问题，通过覆盖度评估提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.12015](https://arxiv.org/pdf/2604.12015)**

> **作者:** Jiayi Xin; Xiang Li; Evan Qiang; Weiqing He; Tianqi Shang; Weijie J. Su; Qi Long
>
> **备注:** ACL 2026 Findings; 17 pages, 3 figures
>
> **摘要:** In-context learning (ICL) performance depends critically on which demonstrations are placed in the prompt, yet most existing selectors prioritize heuristic notions of relevance or diversity and provide limited insight into the coverage of a demonstration set. We propose Unseen Coverage Selection (UKS), a training-free, subset-level coverage prior motivated by the principle that a good demonstration set should expose the model to latent cluster unrevealed by the currently selected subset. UCS operationalizes this idea by (1) inducing discrete latent clusters from model-consistent embeddings and (2) estimating the number of unrevealed clusters within a candidate subset via a Smoothed Good--Turing estimator from its empirical frequency spectrum. Unlike previous selection methods, UCS is coverage-based and training-free, and can be seamlessly combined with both query-dependent and query-independent selection baselines via a simple regularized objective. Experiments on multiple intent-classification and reasoning benchmarks with frontier Large Language Models show that augmenting strong baselines with UCS consistently improves ICL accuracy by up to 2-6% under the same selection budget, while also yielding insights into task- and model-level latent cluster distributions. Code is available at this https URL.
>
---
#### [new 072] GoodPoint: Learning Constructive Scientific Paper Feedback from Author Responses
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于科学论文反馈生成任务，旨在提升反馈的有效性与实用性。通过分析作者回应，构建数据集并优化模型，提高反馈的准确性与帮助性。**

- **链接: [https://arxiv.org/pdf/2604.11924](https://arxiv.org/pdf/2604.11924)**

> **作者:** Jimin Mun; Chani Jung; Xuhui Zhou; Hyunwoo Kim; Maarten Sap
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** While LLMs hold significant potential to transform scientific research, we advocate for their use to augment and empower researchers rather than to automate research without human oversight. To this end, we study constructive feedback generation, the task of producing targeted, actionable feedback that helps authors improve both their research and its presentation. In this work, we operationalize the effectiveness of feedback along two author-centric axes-validity and author action. We first curate GoodPoint-ICLR, a dataset of 19K ICLR papers with reviewer feedback annotated along both dimensions using author responses. Building on this, we introduce GoodPoint, a training recipe that leverages success signals from author responses through fine-tuning on valid and actionable feedback, together with preference optimization on both real and synthetic preference pairs. Our evaluation on a benchmark of 1.2K ICLR papers shows that a GoodPoint-trained Qwen3-8B improves the predicted success rate by 83.7% over the base model and sets a new state-of-the-art among LLMs of similar size in feedback matching on a golden human feedback set, even surpassing Gemini-3-flash in precision. We further validate these findings through an expert human study, demonstrating that GoodPoint consistently delivers higher practical value as perceived by authors.
>
---
#### [new 073] TimeMark: A Trustworthy Time Watermarking Framework for Exact Generation-Time Recovery from AIGC
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于AI生成内容（AIGC）的可信时间水印任务，旨在解决现有方法可靠性低、易被伪造的问题，提出一种基于密码学的时间水印框架，实现精准生成时间恢复。**

- **链接: [https://arxiv.org/pdf/2604.12216](https://arxiv.org/pdf/2604.12216)**

> **作者:** Shangkun Che; Silin Du; Ge Gao
>
> **摘要:** The widespread use of Large Language Models (LLMs) in text generation has raised increasing concerns about intellectual property disputes. Watermarking techniques, which embed meta information into AI-generated content (AIGC), have the potential to serve as judicial evidence. However, existing methods rely on statistical signals in token distributions, leading to inherently probabilistic detection and reduced reliability, especially in multi-bit encoding (e.g., timestamps). Moreover, such methods introduce detectable statistical patterns, making them vulnerable to forgery attacks and enabling model providers to fabricate arbitrary watermarks. To address these issues, we propose the concept of trustworthy watermark, which achieves reliable recovery with 100% identification accuracy while resisting both user-side statistical attacks and provider-side forgery. We focus on trustworthy time watermarking for use as judicial evidence. Our framework integrates cryptographic techniques and encodes time information into time-dependent secret keys under regulatory supervision, preventing arbitrary timestamp fabrication. The watermark payload is decoupled from time and generated as a random, non-stored bit sequence for each instance, eliminating statistical patterns. To ensure verifiability, we design a two-stage encoding mechanism, which, combined with error-correcting codes, enables reliable recovery of generation time with theoretically perfect accuracy. Both theoretical analysis and experiments demonstrate that our framework satisfies the reliability requirements for judicial evidence and offers a practical solution for future AIGC-related intellectual property disputes.
>
---
#### [new 074] Beyond Single-Dimension Novelty: How Combinations of Theory, Method, and Results-based Novelty Shape Scientific Impact
- **分类: cs.DL; cs.CL; cs.IR**

- **简介: 该论文属于科学评价任务，旨在探究多维新颖性组合对科学影响力的影响。通过分析文章的引言内容，分类并评估理论、方法和结果新颖性的组合效应，揭示其对引用量和高被引排名的作用。**

- **链接: [https://arxiv.org/pdf/2604.12471](https://arxiv.org/pdf/2604.12471)**

> **作者:** Yi Zhao; Yang Chenggang; Yuzhuo Wang; Tong Bao; Zhang Heng; Chengzhi Zhang
>
> **备注:** AII-EEKE 2026
>
> **摘要:** Scientific novelty drives advances at the research frontier, yet it is also associated with heightened uncertainty and potential resistance from incumbent paradigms, leading to complex patterns of scientific impact. Prior studies have primarily ex-amined the relationship between a single dimension of novelty -- such as theoreti-cal, methodological, or results-based novelty -- and scientific impact. However, because scientific novelty is inherently multidimensional, focusing on isolated dimensions may obscure how different types of novelty jointly shape impact. Consequently, we know little about how combinations of novelty types influence scientific impact. To this end, we draw on a dataset of 15,322 articles published in Nature Communications. Using the DeepSeek-V3 model, we classify articles into three novelty dimensions based on the content of their Introduction sections: theoretical novelty, methodological novelty, and results-based novelty. These dimensions may coexist within the same article, forming distinct novelty configura-tions. Scientific impact is measured using five-year citation counts and indicators of whether an article belongs to the top 1% or top 10% highly cited papers. Descriptive results indicate that results-based novelty alone and the simultaneous presence of all three novelty types are the dominant configurations in the sample. Regression results further show that articles with results-based novelty only re-ceive significantly more citations and are more likely to rank among the top 1% and top 10% highly cited papers than articles exhibiting all three novelty types. These findings advance our understanding of how multidimensional novelty configurations shape knowledge diffusion.
>
---
#### [new 075] Long-Horizon Plan Execution in Large Tool Spaces through Entropy-Guided Branching
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于工具增强智能体任务，解决多步骤任务执行中的规划与效率问题。提出SLATE基准和EGB算法，提升任务成功率与计算效率。**

- **链接: [https://arxiv.org/pdf/2604.12126](https://arxiv.org/pdf/2604.12126)**

> **作者:** Rongzhe Wei; Ge Shi; Min Cheng; Na Zhang; Pan Li; Sarthak Ghosh; Vaibhav Gorde; Leman Akoglu
>
> **备注:** This work was completed during an internship at Amazon
>
> **摘要:** Large Language Models (LLMs) have significantly advanced tool-augmented agents, enabling autonomous reasoning via API interactions. However, executing multi-step tasks within massive tool libraries remains challenging due to two critical bottlenecks: (1) the absence of rigorous, plan-level evaluation frameworks and (2) the computational demand of exploring vast decision spaces stemming from large toolsets and long-horizon planning. To bridge these gaps, we first introduce SLATE (Synthetic Large-scale API Toolkit for E-commerce), a large-scale context-aware benchmark designed for the automated assessment of tool-integrated agents. Unlike static metrics, SLATE accommodates diverse yet functionally valid execution trajectories, revealing that current agents struggle with self-correction and search efficiency. Motivated by these findings, we next propose Entropy-Guided Branching (EGB), an uncertainty-aware search algorithm that dynamically expands decision branches where predictive entropy is high. EGB optimizes the exploration-exploitation trade-off, significantly enhancing both task success rates and computational efficiency. Extensive experiments on SLATE demonstrate that our dual contribution provides a robust foundation for developing reliable and scalable LLM agents in tool-rich environments.
>
---
#### [new 076] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于后门攻击任务，旨在解决模型在触发下产生有害输出的问题。通过修改权重，将触发与特定响应关联，同时保持正常输入的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.12359](https://arxiv.org/pdf/2604.12359)**

> **作者:** Rui Yin; Tianxu Han; Naen Xu; Changjiang Li; Ping He; Chunyi Zhou; Jun Wang; Zhihui Fu; Tianyu Du; Jinbao Li; Shouling Ji
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Safety-aligned large language models (LLMs) are increasingly deployed in real-world pipelines, yet this deployment also enlarges the supply-chain attack surface: adversaries can distribute backdoored checkpoints that behave normally under standard evaluation but jailbreak when a hidden trigger is present. Recent post-hoc weight-editing methods offer an efficient approach to injecting such backdoors by directly modifying model weights to map a trigger to an attacker-specified response. However, existing methods typically optimize a token-level mapping that forces an affirmative prefix (e.g., ``Sure''), which does not guarantee sustained harmful output -- the model may begin with apparent agreement yet revert to safety-aligned refusal within a few decoding steps. We address this reliability gap by shifting the backdoor objective from surface tokens to internal representations. We extract a steering vector that captures the difference between compliant and refusal behaviors, and compile it into a persistent weight modification that activates only when the trigger is present. To preserve stealthiness and benign utility, we impose a null-space constraint so that the injected edit remains dormant on clean inputs. The method is efficient, requiring only a small set of examples and admitting a closed-form solution. Across multiple safety-aligned LLMs and jailbreak benchmarks, our method achieves high triggered attack success while maintaining non-triggered safety and general utility.
>
---
#### [new 077] MolMem: Memory-Augmented Agentic Reinforcement Learning for Sample-Efficient Molecular Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于分子优化任务，旨在提高药物发现中样本效率。针对昂贵的评估成本，提出MolMem框架，结合记忆机制提升优化效果。**

- **链接: [https://arxiv.org/pdf/2604.12237](https://arxiv.org/pdf/2604.12237)**

> **作者:** Ziqing Wang; Yibo Wen; Abhishek Pandy; Han Liu; Kaize Ding
>
> **摘要:** In drug discovery, molecular optimization aims to iteratively refine a lead compound to improve molecular properties while preserving structural similarity to the original molecule. However, each oracle evaluation is expensive, making sample efficiency a key challenge for existing methods under a limited oracle budget. Trial-and-error approaches require many oracle calls, while methods that leverage external knowledge tend to reuse familiar templates and struggle on challenging objectives. A key missing piece is long-term memory that can ground decisions and provide reusable insights for future optimizations. To address this, we present MolMem (\textbf{Mol}ecular optimization with \textbf{Mem}ory), a multi-turn agentic reinforcement learning (RL) framework with a dual-memory system. Specifically, MolMem uses Static Exemplar Memory to retrieve relevant exemplars for cold-start grounding, and Evolving Skill Memory to distill successful trajectories into reusable strategies. Built on this memory-augmented formulation, we train the policy with dense step-wise rewards, turning costly rollouts into long-term knowledge that improves future optimization. Extensive experiments show that MolMem achieves 90\% success on single-property tasks (1.5$\times$ over the best baseline) and 52\% on multi-property tasks using only 500 oracle calls. Our code is available at this https URL.
>
---
#### [new 078] Adaptive Test-Time Scaling for Zero-Shot Respiratory Audio Classification
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于零样本呼吸音频分类任务，解决标注数据稀缺问题。提出TRIAGE框架，通过分层推理动态分配计算资源，提升模型效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12647](https://arxiv.org/pdf/2604.12647)**

> **作者:** Tsai-Ning Wang; Herman Teun den Dekker; Lin-Lin Chen; Neil Zeghidour; Aaqib Saeed
>
> **备注:** Accepted at AHLI CHIL 2026
>
> **摘要:** Automated respiratory audio analysis promises scalable, non-invasive disease screening, yet progress is limited by scarce labeled data and costly expert annotation. Zero-shot inference eliminates task-specific supervision, but existing methods apply uniform computation to every input regardless of difficulty. We introduce TRIAGE, a tiered zero-shot framework that adaptively scales test-time compute by routing each audio sample through progressively richer reasoning stages: fast label-cosine scoring in a joint audio-text embedding space (Tier-L), structured matching with clinician-style descriptors (Tier-M), and retrieval-augmented large language model reasoning (Tier-H). A confidence-based router finalizes easy predictions early while allocating additional computation to ambiguous inputs, enabling nearly half of all samples to exit at the cheapest tier. Across nine respiratory classification tasks without task-specific training, TRIAGE achieves a mean AUROC of 0.744, outperforming prior zero-shot methods and matching or exceeding supervised baselines on multiple tasks. Our analysis show that test-time scaling concentrates gains where they matter: uncertain cases see up to 19% relative improvement while confident predictions remain unchanged at minimal cost.
>
---
#### [new 079] MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文档问答任务，旨在解决长工业文档处理中信息丢失问题。提出MultiDocFusion框架，通过多模态分块提升RAG系统效果。**

- **链接: [https://arxiv.org/pdf/2604.12352](https://arxiv.org/pdf/2604.12352)**

> **作者:** Joongmin Shin; Chanjun Park; Jeongbae Park; Jaehyung Seo; Heuiseok Lim
>
> **摘要:** RAG-based QA has emerged as a powerful method for processing long industrial documents. However, conventional text chunking approaches often neglect complex and long industrial document structures, causing information loss and reduced answer quality. To address this, we introduce MultiDocFusion, a multimodal chunking pipeline that integrates: (i) detection of document regions using vision-based document parsing, (ii) text extraction from these regions via OCR, (iii) reconstruction of document structure into a hierarchical tree using large language model (LLM)-based document section hierarchical parsing (DSHP-LLM), and (iv) construction of hierarchical chunks through DFS-based grouping. Extensive experiments across industrial benchmarks demonstrate that MultiDocFusion improves retrieval precision by 8-15% and ANLS QA scores by 2-3% compared to baselines, emphasizing the critical role of explicitly leveraging document hierarchy for multimodal document-based QA. These significant performance gains underscore the necessity of structure-aware chunking in enhancing the fidelity of RAG-based QA systems.
>
---
#### [new 080] Designing Reliable LLM-Assisted Rubric Scoring for Constructed Responses: Evidence from Physics Exams
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于教育评估任务，旨在解决AI辅助评分的可靠性问题。通过实验分析rubric设计和LLM配置对评分一致性的影响，提出提升AI评分可靠性的设计建议。**

- **链接: [https://arxiv.org/pdf/2604.12227](https://arxiv.org/pdf/2604.12227)**

> **作者:** Xiuxiu Tang; G. Alex Ambrose; Ying Cheng
>
> **摘要:** Student responses in STEM assessments are often handwritten and combine symbolic expressions, calculations, and diagrams, creating substantial variation in format and interpretation. Despite their importance for evaluating students' reasoning, such responses are time-consuming to score and prone to rater inconsistency, particularly when partial credit is required. Recent advances in large language models (LLMs) have increased attention to AI-assisted scoring, yet evidence remains limited regarding how rubric design and LLM configurations influence reliability across performance levels. This study examined the reliability of AI-assisted scoring of undergraduate physics constructed responses using GPT-4o. Twenty authentic handwritten exam responses were scored across two rounds by four instructors and by the AI model using skill-based rubrics with differing levels of analytic granularity. Prompting format and temperature settings were systematically varied. Overall, human-AI agreement on total scores was comparable to human inter-rater reliability and was highest for high- and low-performing responses, but declined for mid-level responses involving partial or ambiguous reasoning. Criterion-level analyses showed stronger alignment for clearly defined conceptual skills than for extended procedural judgments. A more fine-grained, checklist-based rubric improved consistency relative to holistic scoring. These findings indicate that reliable AI-assisted scoring depends primarily on clear, well-structured rubrics, while prompting format plays a secondary role and temperature has relatively limited impact. More broadly, the study provides transferable design recommendations for implementing reliable LLM-assisted scoring in STEM contexts through skill-based rubrics and controlled LLM settings.
>
---
#### [new 081] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗模拟任务，旨在解决认知障碍患者模拟的不精准问题。通过提取特征向量和引入随机调制机制，实现更细致的患者模拟与严重程度控制。**

- **链接: [https://arxiv.org/pdf/2604.12210](https://arxiv.org/pdf/2604.12210)**

> **作者:** Weikang Zhang; Zimo Zhu; Zhichuan Yang; Chen Huang; Wenqiang Lei; See-Kiong Ng
>
> **备注:** Findings of ACL 2026
>
> **摘要:** Simulating Standardized Patients with cognitive impairment offers a scalable and ethical solution for clinical training. However, existing methods rely on discrete prompt engineering and fail to capture the heterogeneity of deficits across varying domains and severity levels. To address this limitation, we propose StsPatient for the fine-grained simulation of cognitively impaired patients. We innovatively capture domain-specific features by extracting steering vectors from contrastive pairs of instructions and responses. Furthermore, we introduce a Stochastic Token Modulation (STM) mechanism to regulate the intervention probability. STM enables precise control over impairment severity while mitigating the instability of conventional vector methods. Comprehensive experiments demonstrate that StsPatient significantly outperforms baselines in both clinical authenticity and severity controllability.
>
---
#### [new 082] Policy-Invisible Violations in LLM-Based Agents
- **分类: cs.AI; cs.CL; cs.CR; cs.LG**

- **简介: 该论文属于AI安全任务，解决LLM代理在不知晓关键信息时违反政策的问题。通过构建基准和提出Sentinel框架，实现基于世界状态的合规检查。**

- **链接: [https://arxiv.org/pdf/2604.12177](https://arxiv.org/pdf/2604.12177)**

> **作者:** Jie Wu; Ming Gong
>
> **备注:** 26 pages,1 figure, 11 tables
>
> **摘要:** LLM-based agents can execute actions that are syntactically valid, user-sanctioned, and semantically appropriate, yet still violate organizational policy because the facts needed for correct policy judgment are hidden at decision time. We call this failure mode policy-invisible violations: cases in which compliance depends on entity attributes, contextual state, or session history absent from the agent's visible context. We present PhantomPolicy, a benchmark spanning eight violation categories with balanced violation and safe-control cases, in which all tool responses contain clean business data without policy metadata. We manually review all 600 model traces produced by five frontier models and evaluate them using human-reviewed trace labels. Manual review changes 32 labels (5.3%) relative to the original case-level annotations, confirming the need for trace-level human review. To demonstrate what world-state-grounded enforcement can achieve under favorable conditions, we introduce Sentinel, an enforcement framework based on counterfactual graph simulation. Sentinel treats every agent action as a proposed mutation to an organizational knowledge graph, performs speculative execution to materialize the post-action world state, and verifies graph-structural invariants to decide Allow/Block/Clarify. Against human-reviewed trace labels, Sentinel substantially outperforms a content-only DLP baseline (68.8% vs. 93.0% accuracy) while maintaining high precision, though it still leaves room for improvement on certain violation categories. These results demonstrate what becomes achievable once policy-relevant world state is made available to the enforcement layer.
>
---
#### [new 083] Frontier-Eng: Benchmarking Self-Evolving Agents on Real-World Engineering Tasks with Generative Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Frontier-Eng基准，用于评估AI代理在真实工程任务中的自进化能力。针对传统基准忽略迭代优化的问题，该工作通过生成优化框架测试模型性能，揭示了改进频率与幅度的双幂律规律。**

- **链接: [https://arxiv.org/pdf/2604.12290](https://arxiv.org/pdf/2604.12290)**

> **作者:** Yizhe Chi; Deyao Hong; Dapeng Jiang; Tianwei Luo; Kaisen Yang; Boshi Zhang; Zhe Cao; Xiaoyan Fan; Bingxiang He; Han Hao; Weiyang Jin; Dianqiao Lei; Qingle Liu; Houde Qian; Bowen Wang; Situ Wang; Youjie Zheng; Yifan Zhou; Calvin Xiao; Eren Cai; Qinhuai Na
>
> **摘要:** Current LLM agent benchmarks, which predominantly focus on binary pass/fail tasks such as code generation or search-based question answering, often neglect the value of real-world engineering that is often captured through the iterative optimization of feasible designs. To this end, we introduce Frontier-Eng, a human-verified benchmark for generative optimization -- an iterative propose-execute-evaluate loop in which an agent generates candidate artifacts, receives executable verifier feedback, and revises them under a fixed interaction budget -- spanning $47$ tasks across five broad engineering categories. Unlike previous suites, Frontier-Eng tasks are grounded in industrial-grade simulators and verifiers that provide continuous reward signals and enforce hard feasibility constraints under constrained budgets. We evaluate eight frontier language models using representative search frameworks, finding that while Claude 4.6 Opus achieves the most robust performance, the benchmark remains challenging for all models. Our analysis suggests a dual power-law decay in improvement frequency ($\sim$ 1/iteration) and magnitude ($\sim$ 1/improvement count). We further show that although width improves parallelism and diversity, depth remains crucial for hard-won improvements under a fixed budget. Frontier-Eng establishes a new standard for assessing the capacity of AI agents to integrate domain knowledge with executable feedback to solve complex, open-ended engineering problems.
>
---
#### [new 084] Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究On-policy distillation（OPD）在大语言模型中的机制与动态，解决OPD成功条件及失效原因。通过分析发现教师与学生需有兼容思维模式和新能力，提出改进策略并探讨其可扩展性。**

- **链接: [https://arxiv.org/pdf/2604.13016](https://arxiv.org/pdf/2604.13016)**

> **作者:** Yaxuan Li; Yuxin Zuo; Bingxiang He; Jinqian Zhang; Chaojun Xiao; Cheng Qian; Tianyu Yu; Huan-ang Gao; Wenkai Yang; Zhiyuan Liu; Ning Ding
>
> **备注:** 30 pages, 23 figures. Code: this https URL
>
> **摘要:** On-policy distillation (OPD) has become a core technique in the post-training of large language models, yet its training dynamics remain poorly understood. This paper provides a systematic investigation of OPD dynamics and mechanisms. We first identify that two conditions govern whether OPD succeeds or fails: (i) the student and teacher should share compatible thinking patterns; and (ii) even with consistent thinking patterns and higher scores, the teacher must offer genuinely new capabilities beyond what the student has seen during training. We validate these findings through weak-to-strong reverse distillation, showing that same-family 1.5B and 7B teachers are distributionally indistinguishable from the student's perspective. Probing into the token-level mechanism, we show that successful OPD is characterized by progressive alignment on high-probability tokens at student-visited states, a small shared token set that concentrates most of the probability mass (97%-99%). We further propose two practical strategies to recover failing OPD: off-policy cold start and teacher-aligned prompt selection. Finally, we show that OPD's apparent free lunch of dense token-level reward comes at a cost, raising the question of whether OPD can scale to long-horizon distillation.
>
---
#### [new 085] From Plan to Action: How Well Do Agents Follow the Plan?
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于编程代理研究，旨在分析代理是否遵循给定计划。通过实验发现，无明确计划时代理表现不佳，而合理计划能提升性能，但不当计划反而有害。研究提出需改进模型的适应性推理能力。**

- **链接: [https://arxiv.org/pdf/2604.12147](https://arxiv.org/pdf/2604.12147)**

> **作者:** Shuyang Liu; Saman Dehghan; Jatin Ganhotra; Martin Hirzel; Reyhaneh Jabbarvand
>
> **摘要:** Agents aspire to eliminate the need for task-specific prompt crafting through autonomous reason-act-observe loops. Still, they are commonly instructed to follow a task-specific plan for guidance, e.g., to resolve software issues following phases for navigation, reproduction, patch, and validation. Unfortunately, it is unknown to what extent agents actually follow such instructed plans. Without such an analysis, determining the extent agents comply with a given plan, it is impossible to assess whether a solution was reached through correct strategic reasoning or through other means, e.g., data contamination or overfitting to a benchmark. This paper presents the first extensive, systematic analysis of plan compliance in programming agents, examining 16,991 trajectories from SWE-agent across four LLMs on SWE-bench Verified and SWE-bench Pro under eight plan variations. Without an explicit plan, agents fall back on workflows internalized during training, which are often incomplete, overfit, or inconsistently applied. Providing the standard plan improves issue resolution, and we observe that periodic plan reminders can mitigate plan violations and improve task success. A subpar plan hurts performance even more than no plan at all. Surprisingly, augmenting a plan with additional task-relevant phases in the early stage can degrade performance, particularly when these phases do not align with the model's internal problem-solving strategy. These findings highlight a research gap: fine-tuning paradigms that teach models to follow instructed plans, rather than encoding task-specific plans in them. This requires teaching models to reason and act adaptively, rather than memorizing workflows.
>
---
#### [new 086] Do VLMs Truly "Read" Candlesticks? A Multi-Scale Benchmark for Visual Stock Price Forecasting
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于视觉股票预测任务，旨在解决VLMs在多尺度蜡烛图理解上的不足，构建数据集和评估框架以测试其多尺度市场信号整合能力。**

- **链接: [https://arxiv.org/pdf/2604.12659](https://arxiv.org/pdf/2604.12659)**

> **作者:** Kaiqi Hu; Linda Xiao; Shiyue Xu; Ziyi Tang; Mingwen Liu
>
> **备注:** We evaluate whether VLMs can comprehend multi-scale visual stock price data like human analysts with a proposed benchmark, identifying current VLMs' weak predictive power, significant biases, and limited sensitivity to forecast horizons and prompts
>
> **摘要:** Vision-language models(VLMs) are increasingly applied to visual stock price forecasting, yet existing benchmarks inadequately evaluate their understanding of stock price in candlestick charts. First, prior studies fail to isolate VLMs' comprehension of visual inputs genuinely improves predictive performance and whether VLMs truly comprehend candlestick patterns. Further, most existing datasets and evaluation setups are designed around single-period or tabular inputs. However, human analysts strongly rely on multi-scale candlestick charts, where longer-term horizons capture trend direction and shorter-term horizons provide cues for inflection points, making it difficult to systematically assess VLMs' ability to integrate short-term and long-term visual market dynamics. To bridge this gap, we construct a multi-scale candlestick charts dataset and a standardized evaluation framework to assess VLMs' ability to utilize multi-scale visual market signals. Evaluation combines confusion-matrix-based diagnostics with information coefficient(IC) time series metrics and includes XGBoost as a feature-based temporal baseline. Using this dataset, we benchmark representative VLMs and analyze their ability to leverage multi-scale stock price data. Experimental results show that most VLMs perform well only under persistent uptrend or downtrend conditions, while exhibiting weak predictive capability in more common market scenarios. We also identify significant prediction biases and limited sensitivity to explicitly specified forecast horizons in prompts, indicating inherent limitations in precise temporal reasoning.
>
---
#### [new 087] SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SceneCritic，用于评估3D室内场景生成，解决传统评估方法不稳定的问题。通过符号化约束和迭代测试，提升场景布局的语义与几何一致性。**

- **链接: [https://arxiv.org/pdf/2604.13035](https://arxiv.org/pdf/2604.13035)**

> **作者:** Kathakoli Sengupta; Kai Ao; Paola Cascante-Bonilla
>
> **备注:** Project Page: this https URL
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) increasingly generate indoor scenes through intermediate structures such as layouts and scene graphs, yet evaluation still relies on LLM or VLM judges that score rendered views, making judgments sensitive to viewpoint, prompt phrasing, and hallucination. When the evaluator is unstable, it becomes difficult to determine whether a model has produced a spatially plausible scene or whether the output score reflects the choice of viewpoint, rendering, or prompt. We introduce SceneCritic, a symbolic evaluator for floor-plan-level layouts. SceneCritic's constraints are grounded in SceneOnto, a structured spatial ontology we construct by aggregating indoor scene priors from 3D-FRONT, ScanNet, and Visual Genome. SceneOnto traverses this ontology to jointly verify semantic, orientation, and geometric coherence across object relationships, providing object-level and relationship-level assessments that identify specific violations and successful placements. Furthermore, we pair SceneCritic with an iterative refinement test bed that probes how models build and revise spatial structure under different critic modalities: a rule-based critic using collision constraints as feedback, an LLM critic operating on the layout as text, and a VLM critic operating on rendered observations. Through extensive experiments, we show that (a) SceneCritic aligns substantially better with human judgments than VLM-based evaluators, (b) text-only LLMs can outperform VLMs on semantic layout quality, and (c) image-based VLM refinement is the most effective critic modality for semantic and orientation correction.
>
---
#### [new 088] CodeSpecBench: Benchmarking LLMs for Executable Behavioral Specification Generation
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出CodeSpecBench，用于评估LLMs生成可执行行为规范的能力。任务是检验模型对程序语义的理解，解决现有评估方法不足的问题。工作包括构建基准数据集并测试15个模型。**

- **链接: [https://arxiv.org/pdf/2604.12268](https://arxiv.org/pdf/2604.12268)**

> **作者:** Zaoyu Chen; Jianbo Dai; Boyu Zhu; Jingdong Wang; Huiming Wang; Xin Xu; Haoyang Yuan; Zhijiang Guo; Xiao-Ming Wu
>
> **摘要:** Large language models (LLMs) can generate code from natural language, but the extent to which they capture intended program behavior remains unclear. Executable behavioral specifications, defined via preconditions and postconditions, provide a concrete means to assess such understanding. However, existing work on specification generation is constrained in evaluation methodology, task settings, and specification expressiveness. We introduce CodeSpecBench, a benchmark for executable behavioral specification generation under an execution-based evaluation protocol. CodeSpecBench supports both function-level and repository-level tasks and encodes specifications as executable Python functions. Constructed from diverse real-world codebases, it enables a realistic assessment of both correctness (accepting valid behaviors) and completeness (rejecting invalid behaviors). Evaluating 15 state-of-the-art LLMs on CodeSpecBench, we observe a sharp performance degradation on repository-level tasks, where the best model attains only a 20.2% pass rate. We further find that specification generation is substantially more challenging than code generation, indicating that strong coding performance does not necessarily reflect deep understanding of intended program semantics. Our data and code are available at this https URL.
>
---
#### [new 089] How memory can affect collective and cooperative behaviors in an LLM-Based Social Particle Swarm
- **分类: cs.AI; cs.CL; cs.GT; cs.MA**

- **简介: 该论文属于社会模拟任务，研究LLM代理的记忆如何影响集体合作行为。通过改进粒子群模型，分析不同记忆长度和人格特质对合作的影响。**

- **链接: [https://arxiv.org/pdf/2604.12250](https://arxiv.org/pdf/2604.12250)**

> **作者:** Taisei Hishiki; Takaya Arita; Reiji Suzuki
>
> **备注:** 12 pages, 6 figures and 2 tables
>
> **摘要:** This study examines how model-specific characteristics of Large Language Model (LLM) agents, including internal alignment, shape the effect of memory on their collective and cooperative dynamics in a multi-agent system. To this end, we extend the Social Particle Swarm (SPS) model, in which agents move in a two-dimensional space and play the Prisoner's Dilemma with neighboring agents, by replacing its rule-based agents with LLM agents endowed with Big Five personality scores and varying memory lengths. Using Gemini-2.0-Flash, we find that memory length is a critical parameter governing collective behavior: even a minimal memory drastically suppressed cooperation, transitioning the system from stable cooperative clusters through cyclical formation and collapse of clusters to a state of scattered defection as memory length increased. Big Five personality traits correlated with agent behaviors in partial agreement with findings from experiments with human participants, supporting the validity of the model. Comparative experiments using Gemma~3:4b revealed the opposite trend: longer memory promoted cooperation, accompanied by the formation of dense cooperative clusters. Sentiment analysis of agents' reasoning texts showed that Gemini interprets memory increasingly negatively as its length grows, while Gemma interprets it less negatively, and that this difference persists in the early phase of experiments before the macro-level dynamics converge. These results suggest that model-specific characteristics of LLMs, potentially including alignment, play a fundamental role in determining emergent social behavior in Generative Agent-Based Modeling, and provide a micro-level cognitive account of the contradictions found in prior work on memory and cooperation.
>
---
#### [new 090] Do Transformers Use their Depth Adaptively? Evidence from a Relational Reasoning Task
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer是否在不同难度任务中自适应调整深度，通过关系推理任务分析模型层间信息演化与token间信息整合。**

- **链接: [https://arxiv.org/pdf/2604.12426](https://arxiv.org/pdf/2604.12426)**

> **作者:** Alicia Curth; Rachel Lawrence; Sushrut Karmalkar; Niranjani Prasad
>
> **备注:** Accepted at the ICLR 2026 Workshop on Logical Reasoning of Large Language Models
>
> **摘要:** We investigate whether transformers use their depth adaptively across tasks of increasing difficulty. Using a controlled multi-hop relational reasoning task based on family stories, where difficulty is determined by the number of relationship hops that must be composed, we monitor (i) how predictions evolve across layers via early readouts (the logit lens) and (ii) how task-relevant information is integrated across tokens via causal patching. For pretrained models, we find some limited evidence for adaptive depth use: some larger models need fewer layers to arrive at plausible answers for easier tasks, and models generally use more layers to integrate information across tokens as chain length increases. For models finetuned on the task, we find clearer and more consistent evidence of adaptive depth use, with the effect being stronger for less constrained finetuning regimes that do not preserve general language modeling abilities.
>
---
#### [new 091] The Enforcement and Feasibility of Hate Speech Moderation on Twitter
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于内容审核任务，研究Twitter hate speech的执行与可行性。通过审计发现80% hateful tweets仍在线，表明技术非唯一障碍，还涉及资源分配决策。**

- **链接: [https://arxiv.org/pdf/2604.12289](https://arxiv.org/pdf/2604.12289)**

> **作者:** Manuel Tonneau; Dylan Thurgood; Diyi Liu; Niyati Malhotra; Victor Orozco-Olvera; Ralph Schroeder; Scott A. Hale; Manoel Horta Ribeiro; Paul Röttger; Samuel P. Fraiberger
>
> **摘要:** Online hate speech is associated with substantial social harms, yet it remains unclear how consistently platforms enforce hate speech policies or whether enforcement is feasible at scale. We address these questions through a global audit of hate speech moderation on Twitter (now X). Using a complete 24-hour snapshot of public tweets, we construct representative samples comprising 540,000 tweets annotated for hate speech by trained annotators across eight major languages. Five months after posting, 80% of hateful tweets remain online, including explicitly violent hate speech. Such tweets are no more likely to be removed than non-hateful tweets, with neither severity nor visibility increasing the likelihood of removal. We then examine whether these enforcement gaps reflect technical limits of large-scale moderation systems. While fully automated detection systems cannot reliably identify hate speech without generating large numbers of false positives, they effectively prioritize likely violations for human review. Simulations of a human-AI moderation pipeline indicate that substantially reducing user exposure to hate speech is economically feasible at a cost below existing regulatory penalties. These results suggest that the persistence of online hate cannot be explained by technical constraints alone but also reflects institutional choices in the allocation of moderation resources.
>
---
#### [new 092] LASA: Language-Agnostic Semantic Alignment at the Semantic Bottleneck for LLM Safety
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于LLM安全任务，解决低资源语言安全性能差的问题。通过识别语义瓶颈并进行语言无关的语义对齐，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2604.12710](https://arxiv.org/pdf/2604.12710)**

> **作者:** Junxiao Yang; Haoran Liu; Jinzhe Tu; Jiale Cheng; Zhexin Zhang; Shiyao Cui; Jiaqi Weng; Jialing Tao; Hui Xue; Hongning Wang; Han Qiu; Minlie Huang
>
> **摘要:** Large language models (LLMs) often demonstrate strong safety performance in high-resource languages, yet exhibit severe vulnerabilities when queried in low-resource languages. We attribute this gap to a mismatch between language-agnostic semantic understanding ability and language-dominant safety alignment biased toward high-resource languages. Consistent with this hypothesis, we empirically identify the semantic bottleneck in LLMs, an intermediate layer in which the geometry of model representations is governed primarily by shared semantic content rather than language identity. Building on this observation, we propose Language-Agnostic Semantic Alignment (LASA), which anchors safety alignment directly in semantic bottlenecks. Experiments show that LASA substantially improves safety across all languages: average attack success rate (ASR) drops from 24.7% to 2.8% on LLaMA-3.1-8B-Instruct and remains around 3-4% across Qwen2.5 and Qwen3 Instruct models (7B-32B). Together, our analysis and method offer a representation-level perspective on LLM safety, suggesting that safety alignment requires anchoring safety understanding not in surface text, but in the model's language-agnostic semantic space.
>
---
#### [new 093] RePAIR: Interactive Machine Unlearning through Prompt-Aware Model Repair
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出RePAIR框架，解决用户无法自主删除模型中特定知识的问题。属于机器遗忘任务，通过自然语言指令实现模型编辑，提升用户数据控制权。**

- **链接: [https://arxiv.org/pdf/2604.12820](https://arxiv.org/pdf/2604.12820)**

> **作者:** Jagadeesh Rachapudi; Pranav Singh; Ritali Vatsi; Praful Hambarde; Amit Shukla
>
> **摘要:** Large language models (LLMs) inherently absorb harmful knowledge, misinformation, and personal data during pretraining on large-scale web corpora, with no native mechanism for selective removal. While machine unlearning offers a principled solution, existing approaches are provider-centric, requiring retraining pipelines, curated retain datasets, and direct intervention by model service providers (MSPs), thereby excluding end users from controlling their own data. We introduce Interactive Machine Unlearning (IMU), a new paradigm in which users can instruct LLMs to forget targeted knowledge through natural language at inference time. To realize IMU, we propose RePAIR, a prompt-aware model repair framework comprising (i) a watchdog model for unlearning intent detection, (ii) a surgeon model for generating repair procedures, and (iii) a patient model whose parameters are updated autonomously. At the core of RePAIR, we develop Steering Through Activation Manipulation with PseudoInverse (STAMP), a training-free, single-sample unlearning method that redirects MLP activations toward a refusal subspace via closed-form pseudoinverse updates. Its low-rank variant reduces computational complexity from O(d^3) to O(r^3 + r^2 * d), enabling efficient on-device unlearning with up to ~3x speedup over training-based baselines. Extensive experiments across harmful knowledge suppression, misinformation correction, and personal data erasure demonstrate that RePAIR achieves near-zero forget scores (Acc_f = 0.00, F-RL = 0.00) while preserving model utility (Acc_r up to 84.47, R-RL up to 0.88), outperforming six state-of-the-art baselines. These results establish RePAIR as an effective and practical framework for user-driven model editing, advancing transparent and on-device control over learned knowledge, with potential extensions to multimodal foundation models.
>
---
#### [new 094] Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文介绍Nemotron 3 Super，一个高效混合Mamba-Transformer模型，解决大模型训练与推理效率问题。通过预训练、微调和量化，提升性能与上下文长度，支持开放研究。**

- **链接: [https://arxiv.org/pdf/2604.12374](https://arxiv.org/pdf/2604.12374)**

> **作者:** NVIDIA; Aakshita Chandiramani; Aaron Blakeman; Abdullahi Olaoye; Abhibha Gupta; Abhilash Somasamudramath; Abhinav Khattar; Adeola Adesoba; Adi Renduchintala; Adil Asif; Aditya Agrawal; Aditya Vavre; Ahmad Kiswani; Aishwarya Padmakumar; Ajay Hotchandani; Akanksha Shukla; Akhiad Bercovich; Aleksander Ficek; Aleksandr Shaposhnikov; Alex Gronskiy; Alex Kondratenko; Alex Neefus; Alex Steiner; Alex Yang; Alexander Bukharin; Alexander Young; Ali Hatamizadeh; Ali Taghibakhshi; Alina Galiautdinova; Alisa Liu; Alok Kumar; Ameya Sunil Mahabaleshwarkar; Amir Klein; Amit Zuker; Amnon Geifman; Anahita Bhiwandiwalla; Ananth Subramaniam; Andrew Tao; Anjaney Shrivastava; Anjulie Agrusa; Ankur Srivastava; Ankur Verma; Ann Guan; Anna Shors; Annamalai Chockalingam; Anubhav Mandarwal; Aparnaa Ramani; Arham Mehta; Arti Jain; Arun Venkatesan; Asha Anoosheh; Ashwath Aithal; Ashwin Poojary; Asif Ahamed; Asit Mishra; Asli Sabanci Demiroz; Asma Kuriparambil Thekkumpate; Atefeh Sohrabizadeh; Avinash Kaur; Ayush Dattagupta; Barath Subramaniam Anandan; Bardiya Sadeghi; Barnaby Simkin; Ben Lanir; Benedikt Schifferer; Benjamin Chislett; Besmira Nushi; Bilal Kartal; Bill Thiede; Bita Darvish Rouhani; Bobby Chen; Boris Ginsburg; Brandon Norick; Branislav Kisacanin; Brian Yu; Bryan Catanzaro; Buvaneswari Mani; Carlo del Mundo; Chankyu Lee; Chanran Kim; Chantal Hwang; Chao Ni; Charles Wang; Charlie Truong; Cheng-Ping Hsieh; Chenhan Yu; Chenjie Luo; Cherie Wang; Chetan Mungekar; Chintan Patel; Chris Alexiuk; Chris Holguin; Chris Wing; Christian Munley; Christopher Parisien; Chuck Desai; Chunyang Sheng; Collin Neale; Cyril Meurillon; Dakshi Kumar
>
> **摘要:** We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to 2.2x and 7.5x higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace.
>
---
#### [new 095] INDOTABVQA: A Benchmark for Cross-Lingual Table Understanding in Bahasa Indonesia Documents
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出INDOTABVQA基准，用于评估跨语言表格视觉问答任务。针对多语言文档中的表格理解问题，构建了包含多种视觉风格的文档图像和多语言问答对的数据集，并验证了模型在不同语言和结构复杂度下的表现。**

- **链接: [https://arxiv.org/pdf/2604.11970](https://arxiv.org/pdf/2604.11970)**

> **作者:** Somraj Gautam; Anathapindika Dravichi; Gaurav Harit
>
> **备注:** Accepted in ACL 2026 (Findings)
>
> **摘要:** We introduce INDOTABVQA, a benchmark for evaluating cross-lingual Table Visual Question Answering (VQA) on real-world document images in Bahasa Indonesia. The dataset comprises 1,593 document images across three visual styles (bordered, borderless, and colorful) with one or more than one tables, and 1,593 question-answer sets in four languages: Bahasa Indonesia, English, Hindi, and Arabic. This enables evaluation of Vision-Language Models (VLMs) in both monolingual (Bahasa documents with Bahasa questions) and cross-lingual settings (Bahasa documents with questions in other languages). We benchmark leading open-source VLMs (Qwen2.5-VL, Gemma-3, LLaMA-3.2) and GPT-4o and reveal substantial performance gaps, particularly on structurally complex tables and in low-resource languages. Fine-tuning a compact 3B and LoRA-finetuned 7B model on our dataset yields 11.6% and 17.8% improvements in accuracy. Providing explicit table region coordinates as additional input further improves performance by 4-7%, demonstrating the value of Spatial priors for table-based reasoning. Our findings underscore the importance of language-diverse, domain-specific datasets and demonstrate that targeted fine-tuning can significantly enhance VLM performance on specialized document understanding tasks. INDOTABVQA provides a valuable resource for advancing research in cross-lingual, structure-aware document understanding, especially in underrepresented regions of the world. Full dataset can be accessed in huggingface at: this https URL}
>
---
#### [new 096] RPRA: Predicting an LLM-Judge for Efficient but Performant Inference
- **分类: cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于模型优化任务，旨在解决LLM在计算效率与输出质量间的平衡问题。通过预测LLM裁判评分，提升小模型性能，实现高效且高质量的推理。**

- **链接: [https://arxiv.org/pdf/2604.12634](https://arxiv.org/pdf/2604.12634)**

> **作者:** Dylan R. Ashley; Gaël Le Lan; Changsheng Zhao; Naina Dhingra; Zhipeng Cai; Ernie Chang; Mingchen Zhuge; Yangyang Shi; Vikas Chandra; Jürgen Schmidhuber
>
> **备注:** 10 pages in main text + 6 pages of references + 36 pages of appendices, 12 figures in main text + 37 figures in appendices, 2 tables in main text + 3 table in appendices, 13 prompts in appendices
>
> **摘要:** Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans and have models ask for help when they believe they are incapable of solving a problem on their own; we can overcome this trade-off by allowing smaller models to respond to queries when they believe they can provide good responses, and deferring to larger models when they do not believe they can. To this end, in this paper, we investigate the viability of Predict-Answer/Act (PA) and Reason-Predict-Reason-Answer/Act (RPRA) paradigms where models predict -- prior to responding -- how an LLM judge would score their output. We evaluate three approaches: zero-shot prediction, prediction using an in-context report card, and supervised fine-tuning. Our results show that larger models (particularly reasoning models) perform well when predicting generic LLM judges zero-shot, while smaller models can reliably predict such judges well after being fine-tuned or provided with an in-context report card. Altogether, both approaches can substantially improve the prediction accuracy of smaller models, with report cards and fine-tuning achieving mean improvements of up to 55% and 52% across datasets, respectively. These findings suggest that models can learn to predict their own performance limitations, paving the way for more efficient and self-aware AI systems.
>
---
#### [new 097] Beyond Factual Grounding: The Case for Opinion-Aware Retrieval-Augmented Generation
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决RAG系统对主观内容处理不足的问题。通过引入意见感知机制，提升对多样观点的检索能力。**

- **链接: [https://arxiv.org/pdf/2604.12138](https://arxiv.org/pdf/2604.12138)**

> **作者:** Aditya Agrawal; Alwarappan Nakkiran; Darshan Fofadiya; Alex Karlsson; Harsha Aduri
>
> **备注:** 13 pages, Preprint under review
>
> **摘要:** RAG systems have transformed how LLMs access external knowledge, but we find that current implementations exhibit a bias toward factual, objective content, as evidenced by existing benchmarks and datasets that prioritize objective retrieval. This factual bias - treating opinions and diverse perspectives as noise rather than information to be synthesized - limits RAG systems in real-world scenarios involving subjective content, from social media discussions to product reviews. Beyond technical limitations, this bias poses risks to transparent and accountable AI: echo chamber effects that amplify dominant viewpoints, systematic underrepresentation of minority voices, and potential opinion manipulation through biased information synthesis. We formalize this limitation through the lens of uncertainty: factual queries involve epistemic uncertainty reducible through evidence, while opinion queries involve aleatoric uncertainty reflecting genuine heterogeneity in human perspectives. This distinction implies that factual RAG should minimize posterior entropy, whereas opinion-aware RAG must preserve it. Building on this theoretical foundation, we present an Opinion-Aware RAG architecture featuring LLM-based opinion extraction, entity-linked opinion graphs, and opinion-enriched document indexing. We evaluate our approach on e-commerce seller forum data, comparing an Opinion-Enriched knowledge base against a traditional baseline. Experiments demonstrate substantial improvements in retrieval diversity: +26.8% sentiment diversity, +42.7% entity match rate, and +31.6% author demographic coverage on entity-matched documents. Our results provide empirical evidence that treating subjectivity as a first-class citizen yields measurably more representative retrieval-a first step toward opinion-aware RAG. Future work includes joint optimization of retrieval and generation for distributional fidelity.
>
---
#### [new 098] AnyPoC: Universal Proof-of-Concept Test Generation for Scalable LLM-Based Bug Detection
- **分类: cs.SE; cs.AI; cs.CL; cs.CR**

- **简介: 该论文提出AnyPoC，解决LLM检测漏洞后需人工验证的问题。通过生成可执行PoC实现自动化验证，提升漏洞检测的可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2604.11950](https://arxiv.org/pdf/2604.11950)**

> **作者:** Zijie Zhao; Chenyuan Yang; Weidong Wang; Yihan Yang; Ziqi Zhang; Lingming Zhang
>
> **摘要:** While recent LLM-based agents can identify many candidate bugs in source code, their reports remain static hypotheses that require manual validation, limiting the practicality of automated bug detection. We frame this challenge as a test generation task: given a candidate report, synthesizing an executable proof-of-concept test, or simply a PoC - such as a script, command sequence, or crafted input - to trigger the suspected defect. Automated PoC generation can act as a scalable validation oracle, enabling end-to-end autonomous bug detection by providing concrete execution evidence. However, naive LLM agents are unreliable validators: they are biased toward "success" and may reward-hack by producing plausible but non-functional PoCs or even hallucinated traces. To address this, we present AnyPoC, a general multi-agent framework that (1) analyzes and fact-checks a candidate bug report, (2) iteratively synthesizes and executes a PoC while collecting execution traces, and (3) independently re-executes and scrutinizes the PoC to mitigate hallucination and reward hacking. In addition, AnyPoC also continuously extracts and evolves a PoC knowledge base to handle heterogeneous tasks. AnyPoC operates on candidate bug reports regardless of their source and can be paired with different bug reporters. To demonstrate practicality and generality, we apply AnyPoC, with a simple agentic bug reporter, on 12 critical software systems across diverse languages/domains (many with millions of lines of code) including Firefox, Chromium, LLVM, OpenSSL, SQLite, FFmpeg, and Redis. Compared to the state-of-the-art coding agents, e.g., Claude Code and Codex, AnyPoC produces 1.3x more valid PoCs for true-positive bug reports and rejects 9.8x more false-positive bug reports. To date, AnyPoC has discovered 122 new bugs (105 confirmed, 86 already fixed), with 45 generated PoCs adopted as official regression tests.
>
---
#### [new 099] HintMR: Eliciting Stronger Mathematical Reasoning in Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决小语言模型在复杂数学问题上的推理能力不足问题。通过引入提示辅助框架，提升模型的推理准确性。**

- **链接: [https://arxiv.org/pdf/2604.12229](https://arxiv.org/pdf/2604.12229)**

> **作者:** Jawad Hossain; Xiangyu Guo; Jiawei Zhou; Chong Liu
>
> **备注:** 15 pages, 5 figures, Preprint
>
> **摘要:** Small language models (SLMs) often struggle with complex mathematical reasoning due to limited capacity to maintain long chains of intermediate steps and to recover from early errors. We address this challenge by introducing a hint-assisted reasoning framework that incrementally guides SLMs through multi-step mathematical problem solving. Our approach decomposes solutions into sequential reasoning steps and provides context-aware hints, where hints are generated by a separate SLM trained via distillation from a strong large language model. While the hint-generating SLM alone is not capable of solving the problems, its collaboration with a reasoning SLM enables effective guidance, forming a cooperative two-model system for reasoning. Each hint is generated conditionally on the problem statement and the accumulated reasoning history, providing stepwise, localized guidance without revealing full solutions. This reduces error propagation and allows the reasoning model to focus on manageable subproblems. Experiments across diverse mathematical benchmarks and models demonstrate that hint assistance consistently improves reasoning accuracy for SLMs, yielding substantial gains over standard prompting while preserving model efficiency. These results highlight that structured collaboration between SLMs-via hint generation and reasoning-offers an effective and lightweight mechanism for enhancing mathematical reasoning.
>
---
#### [new 100] From Imitation to Discrimination: Progressive Curriculum Learning for Robust Web Navigation
- **分类: cs.LG; cs.CL; cs.HC**

- **简介: 该论文属于网页导航任务，解决文本代理在复杂页面中的鲁棒性问题。通过构建Triton数据集和渐进式训练课程，提升代理的辨别能力和泛化性能。**

- **链接: [https://arxiv.org/pdf/2604.12666](https://arxiv.org/pdf/2604.12666)**

> **作者:** Chuang Peng; Wei Zhang; Renshuai Tao; Xinhao Zhang; Jian Yang
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Text-based web agents offer computational efficiency for autonomous web navigation, yet developing robust agents remains challenging due to the noisy and heterogeneous nature of real-world HTML. Standard Supervised Fine-Tuning (SFT) approaches fail in two critical dimensions: they lack discrimination capabilities to reject plausible but incorrect elements in densely populated pages, and exhibit limited generalization to unseen website layouts. To address these challenges, we introduce the Triton dataset (590k instances) and a progressive training curriculum. Triton is constructed via Structural-Semantic Hard Negative Mining, which explicitly mines topologically similar distractors, and a Dual-Agent Consensus pipeline that synthesizes diverse cross-domain tasks with strict verification. Building upon this foundation, our progressive curriculum produces three models: Triton-SFT-32B for basic imitation, Triton-ORPO-32B for robust discrimination via Odds Ratio Preference Optimization, and Triton-GRPO-32B for long-horizon consistency through Group Relative Policy Optimization. Empirical evaluation on Mind2Web demonstrates that Triton-GRPO-32B achieves state-of-the-art performance among open-source models with 58.7% Step Success Rate, surpassing GPT-4.5 (42.4%) and Claude-4.5 (41.4%) by over 16%, validating that specialized data curriculum outweighs raw parameter scale for web navigation.
>
---
#### [new 101] GeoAlign: Geometric Feature Realignment for MLLM Spatial Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型的 spatial reasoning 任务，旨在解决几何特征与模型需求不匹配的问题。提出 GeoAlign 框架，动态聚合多层几何特征以提升性能。**

- **链接: [https://arxiv.org/pdf/2604.12630](https://arxiv.org/pdf/2604.12630)**

> **作者:** Zhaochen Liu; Limeng Qiao; Guanglu Wan; Tingting Jiang
>
> **摘要:** Multimodal large language models (MLLMs) have exhibited remarkable performance in various visual tasks, yet still struggle with spatial reasoning. Recent efforts mitigate this by injecting geometric features from 3D foundation models, but rely on static single-layer extractions. We identify that such an approach induces a task misalignment bias: the geometric features naturally evolve towards 3D pretraining objectives, which may contradict the heterogeneous spatial demands of MLLMs, rendering any single layer fundamentally insufficient. To resolve this, we propose GeoAlign, a novel framework that dynamically aggregates multi-layer geometric features to realign with the actual demands. GeoAlign constructs a hierarchical geometric feature bank and leverages the MLLM's original visual tokens as content-aware queries to perform layer-wise sparse routing, adaptively fetching the suitable geometric features for each patch. Extensive experiments on VSI-Bench, ScanQA, and SQA3D demonstrate that our compact 4B model effectively achieves state-of-the-art performance, even outperforming larger existing MLLMs.
>
---
## 更新

#### [replaced 001] GRADE: Probing Knowledge Gaps in LLMs through Gradient Subspace Dynamics
- **分类: cs.CL**

- **简介: 该论文提出GRADE方法，用于检测大语言模型的知识缺口，通过分析梯度与隐藏状态子空间的动态关系。任务是评估模型是否具备足够知识回答问题。**

- **链接: [https://arxiv.org/pdf/2604.02830](https://arxiv.org/pdf/2604.02830)**

> **作者:** Yujing Wang; Yuanbang Liang; Yukun Lai; Hainan Zhang; Hanqi Yan
>
> **摘要:** Detecting whether a model's internal knowledge is sufficient to correctly answer a given question is a fundamental challenge in deploying responsible LLMs. In addition to verbalising the confidence by LLM self-report, more recent methods explore the model internals, such as the hidden states of the response tokens, to capture how much knowledge is activated. We argue that such activated knowledge may not align with what the query requires, e.g., capturing the stylistic and length-related features that are uninformative for answering the query. To fill the gap, we propose GRADE (Gradient Dynamics for knowledge gap detection), which quantifies the knowledge gap via the cross-layer rank ratio of the gradient to that of the corresponding hidden state subspace. This is motivated by the property of gradients as estimators of the required knowledge updates for a given target. We validate GRADE on six benchmarks, demonstrating its effectiveness and robustness to input perturbations. In addition, we present a case study showing how the gradient chain can generate interpretable explanations of knowledge gaps for long-form answers.
>
---
#### [replaced 002] Advancing Multi-Agent RAG Systems with Minimalist Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多轮交互中长上下文导致的性能下降问题。提出Mujica-MyGo框架，通过多智能体协作和轻量强化学习优化，提升RAG系统的推理效率与效果。**

- **链接: [https://arxiv.org/pdf/2505.17086](https://arxiv.org/pdf/2505.17086)**

> **作者:** Yihong Wu; Liheng Ma; Muzhi Li; Jiaming Zhou; Lei Ding; Jianye Hao; Ho-fung Leung; Irwin King; Yingxue Zhang; Jian-Yun Nie
>
> **备注:** AAMAS 2026
>
> **摘要:** Large Language Models (LLMs) equipped with modern Retrieval-Augmented Generation (RAG) systems often employ multi-turn interaction pipelines to interface with search engines for complex reasoning tasks. However, such multi-turn interactions inevitably produce long intermediate contexts, as context length grows exponentially with exploration depth. This leads to a well-known limitation of LLMs: their difficulty in effectively leveraging information from long contexts. This problem is further amplified in RAG systems that depend on in-context learning, where few-shot demonstrations must also be included in the prompt, compounding the context-length bottleneck. To address these challenges, we propose Mujica-MyGo, a unified framework for efficient multi-turn reasoning in RAG. Inspired by the divide-and-conquer principle, we introduce Mujica (Multi-hop Joint Intelligence for Complex Question Answering), a multi-agent RAG workflow that decomposes multi-turn interactions into cooperative sub-interactions, thereby mitigating long-context issues. To eliminate the dependency on in-context learning, we further develop MyGO (Minimalist Policy Gradient Optimization), a lightweight and efficient reinforcement learning algorithm that enables effective post-training of LLMs within complex RAG pipelines. We provide theoretical guarantees for MyGO's convergence to the optimal policy. Empirical evaluations across diverse question-answering benchmarks, covering both text corpora and knowledge graphs, show that Mujica-MyGO achieves superior performance.
>
---
#### [replaced 003] AAPO: Enhancing the Reasoning Capabilities of LLMs with Advantage Margin
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升大语言模型的推理能力。针对现有方法在优势估计接近零时效率低的问题，提出AAPO算法，通过基于边距的优势增强优化交叉熵损失，提升训练效率。**

- **链接: [https://arxiv.org/pdf/2505.14264](https://arxiv.org/pdf/2505.14264)**

> **作者:** Jian Xiong; Jingbo Zhou; Jingyong Ye; Qiang Huang; Dejing Dou
>
> **备注:** Accepted to ACL2026 Main Conference
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective approach for enhancing the reasoning capabilities of large language models (LLMs), especially in scenarios where supervised fine-tuning (SFT) falls short due to limited chain-of-thought (CoT) data. Among RL-based post-training methods, group relative advantage estimation, as exemplified by Group Relative Policy Optimization (GRPO), has attracted considerable attention for eliminating the dependency on the value model, thereby simplifying training compared to traditional approaches like Proximal Policy Optimization (PPO). However, we observe that exsiting group relative advantage estimation method still suffers from training inefficiencies, particularly when the estimated advantage approaches zero. To address this limitation, we propose Advantage-Augmented Policy Optimization (AAPO), a novel RL algorithm that optimizes the cross-entropy (CE) loss using advantages enhanced through a margin-based estimation scheme. This approach effectively mitigates the inefficiencies associated with group relative advantage estimation. Experimental results on multiple mathematical reasoning benchmarks demonstrate the superior performance of AAPO. Code is available at this https URL.
>
---
#### [replaced 004] Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决安全领域特征提取的困难。提出Safe-SAIL框架，提升大语言模型安全特征的可解释性与效率。**

- **链接: [https://arxiv.org/pdf/2509.18127](https://arxiv.org/pdf/2509.18127)**

> **作者:** Jiaqi Weng; Han Zheng; Hanyu Zhang; Ej Zhou; Qinqin He; Jialing Tao; Hui Xue; Zhixuan Chu; Xiting Wang
>
> **摘要:** Sparse autoencoders (SAEs) enable interpretability research by decomposing entangled model activations into monosemantic features. However, under what circumstances SAEs derive most fine-grained latent features for safety, a low-frequency concept domain, remains unexplored. Two key challenges exist: identifying SAEs with the greatest potential for generating safety domain-specific features, and the prohibitively high cost of detailed feature explanation. In this paper, we propose Safe-SAIL, a unified framework for interpreting SAE features in safety-critical domains to advance mechanistic understanding of large language models. Safe-SAIL introduces a pre-explanation evaluation metric to efficiently identify SAEs with strong safety domain-specific interpretability, and reduces interpretation cost by 55% through a segment-level simulation strategy. Building on Safe-SAIL, we train a comprehensive suite of SAEs with human-readable explanations and systematic evaluations for 1,758 safety-related features spanning four domains: pornography, politics, violence, and terror. Using this resource, we conduct empirical analyses and provide insights on the effectiveness of Safe-SAIL for risk feature identification and how safety-critical entities and concepts are encoded across model layers. All models, explanations, and tools are publicly released in our open-source toolkit and companion product.
>
---
#### [replaced 005] Variation in Verification: Understanding Verification Dynamics in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的验证机制，探讨验证效果与问题难度、生成能力和验证能力的关系，旨在优化测试时扩展的应用策略。**

- **链接: [https://arxiv.org/pdf/2509.17995](https://arxiv.org/pdf/2509.17995)**

> **作者:** Yefan Zhou; Austin Xu; Yilun Zhou; Janvijay Singh; Jiang Gui; Shafiq Joty
>
> **备注:** ICLR 2026
>
> **摘要:** Recent advances have shown that scaling test-time computation enables large language models (LLMs) to solve increasingly complex problems across diverse domains. One effective paradigm for test-time scaling (TTS) involves LLM generators producing multiple solution candidates, with LLM verifiers assessing the correctness of these candidates without reference answers. In this paper, we study generative verifiers, which perform verification by generating chain-of-thought (CoT) reasoning followed by a binary verdict. We systematically analyze verification dynamics across three dimensions - problem difficulty, generator capability, and verifier generation capability - with empirical studies on 12 benchmarks across mathematical reasoning, knowledge, and natural language reasoning tasks using 14 open-source models (2B to 72B parameter range) and GPT-4o. Our experiments reveal three key findings about verification effectiveness: (1) Easy problems allow verifiers to more reliably certify correct responses; (2) Weak generators produce errors that are easier to detect than strong generators; (3) Verification ability is generally correlated with the verifier's own problem-solving capability, but this relationship varies with problem difficulty. These findings reveal opportunities to optimize basic verification strategies in TTS applications. First, given the same verifier, some weak generators can nearly match stronger ones in post-verification TTS performance (e.g., the Gemma2-9B to Gemma2-27B performance gap shrinks by 75.7%). Second, we identify cases where strong verifiers offer limited advantage over weak ones, as both fail to provide meaningful verification gains, suggesting that verifier scaling alone cannot overcome fundamental verification challenges.
>
---
#### [replaced 006] Many-Tier Instruction Hierarchy in LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理任务，解决多源指令冲突问题。提出ManyIH框架及基准测试，用于处理多层级指令冲突，提升代理安全与效率。**

- **链接: [https://arxiv.org/pdf/2604.09443](https://arxiv.org/pdf/2604.09443)**

> **作者:** Jingyu Zhang; Tianjian Li; William Jurayj; Hongyuan Zhan; Benjamin Van Durme; Daniel Khashabi
>
> **摘要:** Large language model agents receive instructions from many sources-system messages, user prompts, tool outputs, other agents, and more-each carrying different levels of trust and authority. When these instructions conflict, agents must reliably follow the highest-privilege instruction to remain safe and effective. The dominant paradigm, instruction hierarchy (IH), assumes a fixed, small set of privilege levels (typically fewer than five) defined by rigid role labels (e.g., system > user). This is inadequate for real-world agentic settings, where conflicts can arise across far more sources and contexts. In this work, we propose Many-Tier Instruction Hierarchy (ManyIH), a paradigm for resolving instruction conflicts among instructions with arbitrarily many privilege levels. We introduce ManyIH-Bench, the first benchmark for ManyIH. ManyIH-Bench requires models to navigate up to 12 levels of conflicting instructions with varying privileges, comprising 853 agentic tasks (427 coding and 426 instruction-following). ManyIH-Bench composes constraints developed by LLMs and verified by humans to create realistic and difficult test cases spanning 46 real-world agents. Our experiments show that even the current frontier models perform poorly (~40% accuracy) when instruction conflict scales. This work underscores the urgent need for methods that explicitly target fine-grained, scalable instruction conflict resolution in agentic settings.
>
---
#### [replaced 007] Revisiting the Reliability of Language Models in Instruction-Following
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究语言模型在指令遵循中的可靠性问题。针对真实场景中用户表述变化导致的性能下降，提出新指标和评估基准，分析模型在细微差异指令下的表现。**

- **链接: [https://arxiv.org/pdf/2512.14754](https://arxiv.org/pdf/2512.14754)**

> **作者:** Jianshuo Dong; Yutong Zhang; Yan Liu; Zhenyu Zhong; Tao Wei; Chao Zhang; Han Qiu
>
> **备注:** ACL 2026 main
>
> **摘要:** Advanced LLMs have achieved near-ceiling instruction-following accuracy on benchmarks such as IFEval. However, these impressive scores do not necessarily translate to reliable services in real-world use, where users often vary their phrasing, contextual framing, and task formulations. In this paper, we study nuance-oriented reliability: whether models exhibit consistent competence across cousin prompts that convey analogous user intents but with subtle nuances. To quantify this, we introduce a new metric, reliable@k, and develop an automated pipeline that generates high-quality cousin prompts via data augmentation. Building upon this, we construct IFEval++ for systematic evaluation. Across 20 proprietary and 26 open-source LLMs, we find that current models exhibit substantial insufficiency in nuance-oriented reliability -- their performance can drop by up to 61.8% with nuanced prompt modifications. What's more, we characterize it and explore three potential improvement recipes. Our findings highlight nuance-oriented reliability as a crucial yet underexplored next step toward more dependable and trustworthy LLM behavior. Our code and benchmark are accessible: this https URL.
>
---
#### [replaced 008] Hear Both Sides: Efficient Multi-Agent Debate via Diversity-Aware Message Retention
- **分类: cs.CL**

- **简介: 该论文属于多智能体推理任务，旨在解决辩论中消息冗余和噪声问题。提出DAR框架，通过选择差异大的消息进行传播，提升辩论效果。**

- **链接: [https://arxiv.org/pdf/2603.20640](https://arxiv.org/pdf/2603.20640)**

> **作者:** Manh Nguyen; Anh Nguyen; Dung Nguyen; Svetha Venkatesh; Hung Le
>
> **摘要:** Multi-Agent Debate has emerged as a promising framework for improving the reasoning quality of large language models through iterative inter-agent communication. However, broadcasting all agent messages at every round introduces noise and redundancy that can degrade debate quality and waste computational resources. Current approaches rely on uncertainty estimation to filter low-confidence responses before broadcasting, but this approach is unreliable due to miscalibrated confidence scores and sensitivity to threshold selection. To address this, we propose Diversity-Aware Retention (DAR), a lightweight debate framework that, at each debate round, selects the subset of agent responses that maximally disagree with each other and with the majority vote before broadcasting. Through an explicit index-based retention mechanism, DAR preserves the original messages without modification, ensuring that retained disagreements remain authentic. Experiments on diverse reasoning and question answering benchmarks demonstrate that our selective message propagation consistently improves debate performance, particularly as the number of agents scales, where noise accumulation is most severe. Our results highlight that what agents hear is as important as what agents say in multi-agent reasoning systems. Code is publicly available at this https URL.
>
---
#### [replaced 009] CocoaBench: Evaluating Unified Digital Agents in the Wild
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CocoaBench，一个评估统一数字代理的基准，解决代理在复杂任务中整合多种能力的问题。工作包括设计任务、构建评估框架，并验证现有代理表现不足。**

- **链接: [https://arxiv.org/pdf/2604.11201](https://arxiv.org/pdf/2604.11201)**

> **作者:** CocoaBench Team; Shibo Hao; Zhining Zhang; Zhiqi Liang; Tianyang Liu; Yuheng Zha; Qiyue Gao; Jixuan Chen; Zilong Wang; Zhoujun Cheng; Haoxiang Zhang; Junli Wang; Hexi Jin; Boyuan Zheng; Kun Zhou; Yu Wang; Feng Yao; Licheng Liu; Yijiang Li; Zhifei Li; Zhengtao Han; Pracha Promthaw; Tommaso Cerruti; Xiaohan Fu; Ziqiao Ma; Jingbo Shang; Lianhui Qin; Julian McAuley; Eric P. Xing; Zhengzhong Liu; Rupesh Kumar Srivastava; Zhiting Hu
>
> **备注:** Project page: this https URL
>
> **摘要:** LLM agents now perform strongly in software engineering, deep research, GUI automation, and various other applications, while recent agent scaffolds and models are increasingly integrating these capabilities into unified systems. Yet, most evaluations still test these capabilities in isolation, which leaves a gap for more diverse use cases that require agents to combine different capabilities. We introduce CocoaBench, a benchmark for unified digital agents built from human-designed, long-horizon tasks that require flexible composition of vision, search, and coding. Tasks are specified only by an instruction and an automatic evaluation function over the final output, enabling reliable and scalable evaluation across diverse agent infrastructures. We also present CocoaAgent, a lightweight shared scaffold for controlled comparison across model backbones. Experiments show that current agents remain far from reliable on CocoaBench, with the best evaluated system achieving only 45.1% success rate. Our analysis further points to substantial room for improvement in reasoning and planning, tool use and execution, and visual grounding.
>
---
#### [replaced 010] Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space
- **分类: cs.CL; cs.LG; q-bio.NC**

- **简介: 该论文属于自然语言处理任务，旨在通过嵌入空间轨迹分析人类语义导航。研究提出框架，用几何和动态指标衡量概念生成过程，适用于临床和跨语言分析。**

- **链接: [https://arxiv.org/pdf/2602.05971](https://arxiv.org/pdf/2602.05971)**

> **作者:** Felipe D. Toro-Hernández; Jesuino Vieira Filho; Rodrigo M. Cabral-Carvalho
>
> **备注:** 10 pages, 6 figures (excluding refs/appendix). Accepted to ICLR 2026
>
> **摘要:** Semantic representations can be framed as a structured, dynamic knowledge space through which humans navigate to retrieve and manipulate meaning. To investigate how humans traverse this geometry, we introduce a framework that represents concept production as navigation through embedding space. Using different transformer text embedding models, we construct participant-specific semantic trajectories based on cumulative embeddings and extract geometric and dynamical metrics, including distance to next, distance to centroid, entropy, velocity, and acceleration. These measures capture both scalar and directional aspects of semantic navigation, providing a computationally grounded view of semantic representation search as movement in a geometric space. We evaluate the framework on four datasets across different languages, spanning different property generation tasks: Neurodegenerative, Swear verbal fluency, Property listing task in Italian, and in German. Across these contexts, our approach distinguishes between clinical groups and concept types, offering a mathematical framework that requires minimal human intervention compared to typical labor-intensive linguistic pre-processing methods. Comparison with a non-cumulative approach reveals that cumulative embeddings work best for longer trajectories, whereas shorter ones may provide too little context, favoring the non-cumulative alternative. Critically, different embedding models yielded similar results, highlighting similarities between different learned representations despite different training pipelines. By framing semantic navigation as a structured trajectory through embedding space, bridging cognitive modeling with learned representation, thereby establishing a pipeline for quantifying semantic representation dynamics with applications in clinical research, cross-linguistic analysis, and the assessment of artificial cognition.
>
---
#### [replaced 011] Enhancing Agentic Textual Graph Retrieval with Synthetic Stepwise Supervision
- **分类: cs.CL**

- **简介: 该论文属于图文本问答任务，旨在解决复杂图推理中子图检索的问题。通过合成分步监督训练LLM检索器，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2510.03323](https://arxiv.org/pdf/2510.03323)**

> **作者:** Ge Chang; Jinbo Su; Jiacheng Liu; Pengfei Yang; Yuhao Shang; Huiwen Zheng; Hongli Ma; Yan Liang; Yuanchun Li; Yunxin Liu
>
> **备注:** Accepted by ACL2026
>
> **摘要:** Integrating textual graphs into Large Language Models (LLMs) is promising for complex graph-based QA. However, a key bottleneck is retrieving informative yet compact subgraphs that fit the LLM context. Existing retrievers often struggle, relying either on shallow embedding similarity or costly interactive policies that require excessive supervision. To address these challenges, we introduce an agentic textual graph reasoning framework featuring an LLM-based retriever trained with synthetic stepwise supervision. Rather than relying on final answer rewards which often yield sparse and unstable signals, we optimize the retriever by evaluating each step against offline-extracted golden subgraphs. Our approach distills golden subgraphs via a specialized data synthesis pipeline to formulate dense rewards, facilitating a two-stage training scheme that effectively learns the interactive graph exploration policy. Based on extensive experiments on three common datasets in comparison with seven strong baselines, our approach achieves an average improvement of 8.1% in accuracy and 9.7% in F1 score. The advantage is even higher in more complicated multi-hop reasoning tasks. Our code will be open-sourced.
>
---
#### [replaced 012] FlowPlan-G2P: A Structured Generation Framework for Transforming Scientific Papers into Patent Descriptions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于论文转专利的任务，旨在解决科学论文转化为专利描述时逻辑与法律合规性不足的问题。提出FlowPlan-G2P框架，通过三阶段结构化生成提升质量。**

- **链接: [https://arxiv.org/pdf/2601.02589](https://arxiv.org/pdf/2601.02589)**

> **作者:** Kris W Pan; Yongmin Yoo
>
> **摘要:** Over 3.5 million patents are filed annually, with drafting patent descriptions requiring deep technical and legal expertise. Transforming scientific papers into patent descriptions is particularly challenging due to their differing rhetorical styles and stringent legal requirements. Unlike black-box text-to-text approaches that struggle to model structural reasoning and legal constraints, we propose FlowPlan-G2P, a novel framework that mirrors the cognitive workflow of expert drafters by reformulating this task into three stages: (1) Concept Graph Induction, extracting technical entities and relationships into a directed graph via expert-like reasoning; (2) Paragraph and Section Planning, reorganizing the graph into coherent clusters aligned with canonical patent sections; and (3) Graph-Conditioned Generation, producing legally compliant paragraphs using section-specific subgraphs and tailored prompts. Experiments demonstrate that FlowPlan-G2P significantly improves logical coherence and legal compliance over end-to-end LLM baselines. Our framework establishes a new paradigm for paper-to-patent generation and advances structured text generation for specialized domains.
>
---
#### [replaced 013] StoryScope: Investigating idiosyncrasies in AI fiction
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在区分AI与人类创作的虚构故事。通过分析叙事结构而非写作风格，提出StoryScope工具，实现高精度识别与作者归属。**

- **链接: [https://arxiv.org/pdf/2604.03136](https://arxiv.org/pdf/2604.03136)**

> **作者:** Jenna Russell; Rishanth Rajendhran; Chau Minh Pham; Mohit Iyyer; John Wieting
>
> **摘要:** As AI-generated fiction becomes increasingly prevalent, questions of authorship and originality are becoming central to how written work is evaluated. While most existing work in this space focuses on identifying surface-level signatures of AI writing, we ask instead whether AI-generated stories can be distinguished from human ones without relying on stylistic signals, focusing on discourse-level narrative choices such as character agency and chronological discontinuity. We propose StoryScope, a pipeline that automatically induces a fine-grained, interpretable feature space of discourse-level narrative features across 10 dimensions. We apply StoryScope to a parallel corpus of 10,272 writing prompts, each written by a human author and five LLMs, yielding 61,608 stories, each ~5,000 words, and 304 extracted features per story. Narrative features alone achieve 93.2% macro-F1 for human vs. AI detection and 68.4% macro-F1 for six-way authorship attribution, retaining over 97% of the performance of models that include stylistic cues. A compact set of 30 core narrative features captures much of this signal: AI stories over-explain themes and favor tidy, single-track plots while human stories frame protagonist' choices as more morally ambiguous and have increased temporal complexity. Per-model fingerprint features enable six-way attribution: for example, Claude produces notably flat event escalation, GPT over-indexes on dream sequences, and Gemini defaults to external character description. We find that AI-generated stories cluster in a shared region of narrative space, while human-authored stories exhibit greater diversity. More broadly, these results suggest that differences in underlying narrative construction, not just writing style, can be used to separate human-written original works from AI-generated fiction.
>
---
#### [replaced 014] Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于物理问题求解任务，旨在提升基础模型在奥赛级物理题中的推理能力。通过引入PhoPile数据集和RAG方法，验证了检索增强对物理推理的促进作用。**

- **链接: [https://arxiv.org/pdf/2510.00919](https://arxiv.org/pdf/2510.00919)**

> **作者:** Shunfeng Zheng; Yudi Zhang; Meng Fang; Zihan Zhang; Zhitan Wu; Mykola Pechenizkiy; Ling Chen
>
> **备注:** Accepted to EMNLP 2025 (Findings)
>
> **摘要:** Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning.
>
---
#### [replaced 015] Reasoning Graphs: Self-Improving, Deterministic RAG through Evidence-Centric Feedback
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出推理图，解决RAG系统准确性低、波动大的问题，通过结构化证据链提升推理一致性与效率。**

- **链接: [https://arxiv.org/pdf/2604.07595](https://arxiv.org/pdf/2604.07595)**

> **作者:** Matthew Penaroza
>
> **备注:** 15 pages including appendix, 2 figures, 3 algorithms, framework paper with evaluation protocol
>
> **摘要:** Language model agents reason from scratch on every query, discarding their chain of thought after each run. This produces lower accuracy and high variance, as the same query type can succeed or fail unpredictably. We introduce reasoning graphs, a graph structure that persists per-evidence chain of thought as structured edges connected to the evidence items they evaluate. Unlike prior memory mechanisms that retrieve distilled strategies by query similarity, reasoning graphs enable evidence-centric feedback: given a new candidate set, the system traverses all incoming evaluation edges for each evidence item across all prior runs, surfacing how that specific item has been judged before. We further introduce retrieval graphs, a complementary structure that feeds a pipeline planner to tighten the candidate funnel over successive runs. Together, both graphs form a self-improving feedback loop: accuracy improves systematically and verdict-level variance collapses. This requires no retraining; the base model remains frozen and all gains come from context engineering via graph traversal. We evaluate on MuSiQue and HotpotQA using a sequential cluster protocol, a high-reuse deployment simulation, and a determinism experiment. At 50%+ evidence profile coverage, our system reduces errors by 47% compared to vanilla RAG on the same questions (controlled dose-response, p < 0.0001). On 4-hop questions, accuracy improves by +11.0pp (p=0.0001). In high-reuse settings, the system achieves Pareto dominance: highest accuracy, 47% lower cost, and 46% lower latency. Evidence profiles improve verdict consistency by 7-8 percentage points (p=0.007, Wilcoxon); the full system drives all 11 hard probes to perfect consistency at both temperature 0 and 0.5 (p=0.004).
>
---
#### [replaced 016] SEW: Self-Evolving Agentic Workflows for Automated Code Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决多智能体工作流手动设计效率低的问题。提出SEW框架，自动生成和优化工作流，提升代码生成效果。**

- **链接: [https://arxiv.org/pdf/2505.18646](https://arxiv.org/pdf/2505.18646)**

> **作者:** Siwei Liu; Jinyuan Fang; Han Zhou; Yingxu Wang; Zaiqiao Meng
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated effectiveness in code generation tasks. To enable LLMs to address more complex coding challenges, existing research has focused on crafting multi-agent systems with agentic workflows, where complex coding tasks are decomposed into sub-tasks, assigned to specialized agents. Despite their effectiveness, current approaches heavily rely on hand-crafted agentic workflows, with both agent topologies and prompts manually designed, which limits their ability to automatically adapt to different types of coding problems. To address these limitations and enable automated workflow design, we propose \textbf{S}elf-\textbf{E}volving \textbf{W}orkflow (\textbf{SEW}), a novel self-evolving framework that automatically generates and optimises multi-agent workflows. Extensive experiments on three coding benchmark datasets, including the challenging LiveCodeBench, demonstrate that our SEW can automatically design agentic workflows and optimise them through self-evolution, bringing up to 12\% improvement on LiveCodeBench compared to using the backbone LLM only. Furthermore, by investigating different representation schemes of workflow, we provide insights into the optimal way to encode workflow information with text.
>
---
#### [replaced 017] CoRoVA: Compressed Representations for Vector-Augmented Code Completion
- **分类: cs.CL**

- **简介: 该论文属于代码补全任务，解决检索增强生成中序列过长、推理变慢的问题。提出CoRoVA框架，通过压缩上下文提升生成质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2510.19644](https://arxiv.org/pdf/2510.19644)**

> **作者:** Daria Cherniuk; Nikita Sukhorukov; Danil Gusak; Nikita Sushko; Danil Sivtsov; Elena Tutubalina; Evgeny Frolov
>
> **摘要:** Retrieval-augmented generation has emerged as one of the most effective approaches for code completion enhancement, especially when repository-level context is important. However, adding this extra retrieved context significantly increases sequence length, raises prefill cost, and degrades time-to-first-token (TTFT), which slows down inference -- a critical limitation for interactive settings such as IDEs. In this work, we introduce CoRoVA, a framework that compresses context into compact, semantically rich representations that remain interpretable to code LLMs. This improves generation quality while reducing prompt augmentation to only a few compressed single-token vectors. Our approach requires training only a small projector module and introduces negligible additional latency, yet it significantly improves the prediction quality of code LLMs. Our experiments show that CoRoVA enables a 20-38\% reduction in TTFT on completion tasks compared to uncompressed RAG.
>
---
#### [replaced 018] WikiSeeker: Rethinking the Role of Vision-Language Models in Knowledge-Based Visual Question Answering
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于知识驱动的视觉问答任务，旨在解决现有方法过度依赖图像、忽视VLM潜力的问题。提出WikiSeeker框架，重新定义VLM角色，提升检索与回答效果。**

- **链接: [https://arxiv.org/pdf/2604.05818](https://arxiv.org/pdf/2604.05818)**

> **作者:** Yingjian Zhu; Xinming Wang; Kun Ding; Ying Wang; Bin Fan; Shiming Xiang
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Multi-modal Retrieval-Augmented Generation (RAG) has emerged as a highly effective paradigm for Knowledge-Based Visual Question Answering (KB-VQA). Despite recent advancements, prevailing methods still primarily depend on images as the retrieval key, and often overlook or misplace the role of Vision-Language Models (VLMs), thereby failing to leverage their potential fully. In this paper, we introduce WikiSeeker, a novel multi-modal RAG framework that bridges these gaps by proposing a multi-modal retriever and redefining the role of VLMs. Rather than serving merely as answer generators, we assign VLMs two specialized agents: a Refiner and an Inspector. The Refiner utilizes the capability of VLMs to rewrite the textual query according to the input image, significantly improving the performance of the multimodal retriever. The Inspector facilitates a decoupled generation strategy by selectively routing reliable retrieved context to another LLM for answer generation, while relying on the VLM's internal knowledge when retrieval is unreliable. Extensive experiments on EVQA, InfoSeek, and M2KR demonstrate that WikiSeeker achieves state-of-the-art performance, with substantial improvements in both retrieval accuracy and answer quality. Our code will be released on this https URL.
>
---
#### [replaced 019] SeedPrints: Fingerprints Can Even Tell Which Seed Your Large Language Model Was Trained From
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型指纹识别任务，解决LLM溯源问题。提出SeedPrints方法，利用初始化种子作为持久标识，实现从预训练到应用的全生命周期身份验证。**

- **链接: [https://arxiv.org/pdf/2509.26404](https://arxiv.org/pdf/2509.26404)**

> **作者:** Yao Tong; Haonan Wang; Siquan Li; Kenji Kawaguchi; Tianyang Hu
>
> **备注:** Accepted to ICLR 2026. The code repository linked on OpenReview is outdated; the latest code is available via the final arXiv version
>
> **摘要:** Fingerprinting Large Language Models (LLMs)is essential for provenance verification and model attribution. Existing fingerprinting methods are primarily evaluated after fine-tuning, where models have already acquired stable signatures from training data, optimization dynamics, or hyperparameters. However, most of a model's capacity and knowledge are acquired during pretraining rather than downstream fine-tuning, making large-scale pretraining a more fundamental regime for lineage verification. We show that existing fingerprinting methods become unreliable in this regime, as they rely on post-hoc signatures that only emerge after substantial training. This limitation contradicts the classical Galton notion of a fingerprint as an intrinsic and persistent identity. In contrast, we propose a stronger and more intrinsic notion of LLM fingerprinting: SeedPrints, a method that leverages random initialization biases as persistent, seed-dependent identifiers present even before training begins. We show that untrained models exhibit reproducible prediction biases induced by their initialization seed, and that these weak signals remain statistically detectable throughout training, enabling high-confidence lineage verification. Unlike prior techniques that fail during early pretraining or degrade under distribution shifts, SeedPrints remains effective across all training stages, from initialization to large-scale pretraining and downstream adaptation. Experiments on LLaMA-style and Qwen-style models demonstrate seed-level distinguishability and enable birth-to-lifecycle identity verification. Evaluations on large-scale pretraining trajectories and real-world fingerprinting benchmarks further confirm its robustness under prolonged training, domain shifts, and parameter modifications.
>
---
#### [replaced 020] METRO: Towards Strategy Induction from Expert Dialogue Transcripts for Non-collaborative Dialogues
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于非协作对话系统任务，旨在解决手动编码专家策略的低效问题。通过大语言模型从对话转录中自动提取策略动作和规划逻辑，构建策略森林结构，提升性能与跨任务迁移能力。**

- **链接: [https://arxiv.org/pdf/2604.11427](https://arxiv.org/pdf/2604.11427)**

> **作者:** Haofu Yang; Jiaji Liu; Chen Huang; Faguo Wu; Wenqiang Lei; See-Kiong Ng
>
> **备注:** ACL 2026
>
> **摘要:** Developing non-collaborative dialogue agents traditionally requires the manual, unscalable codification of expert strategies. We propose \ours, a method that leverages large language models to autonomously induce both strategy actions and planning logic directly from raw transcripts. METRO formalizes expert knowledge into a Strategy Forest, a hierarchical structure that captures both short-term responses (nodes) and long-term strategic foresight (branches). Experimental results across two benchmarks show that METRO demonstrates promising performance, outperforming existing methods by an average of 9%-10%. Our further analysis not only reveals the success behind METRO (strategic behavioral diversity and foresight), but also demonstrates its robust cross-task transferability. This offers new insights into building non-collaborative agents in a cost-effective and scalable way. Our code is available at this https URL.
>
---
#### [replaced 021] On the Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于深度学习理论研究，旨在揭示层归一化与动态激活函数的关系。通过数学推导，提出新的动态激活函数DyISRU，更准确地实现归一化效果。**

- **链接: [https://arxiv.org/pdf/2503.21708](https://arxiv.org/pdf/2503.21708)**

> **作者:** Felix Stollenwerk
>
> **备注:** EACL 2026 (Main), see this https URL
>
> **摘要:** Layer normalization (LN) is an essential component of modern neural networks. While many alternative techniques have been proposed, none of them have succeeded in replacing LN so far. The latest suggestion in this line of research is a dynamic activation function called Dynamic Tanh (DyT). Although it is empirically well-motivated and appealing from a practical point of view, it lacks a theoretical foundation. In this work, we shed light on the mathematical relationship between LN and dynamic activation functions. In particular, we derive DyT from the LN variant RMSNorm, and show that a well-defined decoupling in derivative space as well as an approximation are needed to do so. By applying the same decoupling procedure directly in function space, we are able to omit the approximation and obtain the exact element-wise counterpart of RMSNorm, which we call Dynamic Inverse Square Root Unit (DyISRU). We demonstrate numerically that DyISRU reproduces the normalization effect on outliers more accurately than DyT does.
>
---
#### [replaced 022] CropVLM: Learning to Zoom for Fine-Grained Vision-Language Perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CropVLM，解决细粒度视觉-语言感知问题，通过动态聚焦图像区域提升模型性能，无需标注或修改原有模型。**

- **链接: [https://arxiv.org/pdf/2511.19820](https://arxiv.org/pdf/2511.19820)**

> **作者:** Miguel Carvalho; Helder Dias; Bruno Martins
>
> **备注:** Accepted to the GRAIL-V Workshop at CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs) often struggle with tasks that require fine-grained image understanding, such as scene-text recognition or document analysis, due to perception limitations and visual fragmentation. To address these challenges, we introduce CropVLM as an external low-cost method for boosting performance, enabling VLMs to dynamically ''zoom in'' on relevant image regions, enhancing their ability to capture fine details. CropVLM is trained using reinforcement learning, without using human-labeled bounding boxes as a supervision signal, and without expensive synthetic evaluations. The model is trained once and can be paired with both open-source and proprietary VLMs to improve their performance. Our approach delivers significant improvements on tasks that require high-resolution image understanding, notably for benchmarks that are out-of-domain for the target VLM, without modifying or fine-tuning the VLM, thus avoiding catastrophic forgetting.
>
---
#### [replaced 023] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，探索其如何编码语音特征。任务是分析模型是否具备可解释的音系向量运算能力，通过实验发现模型中存在线性音系向量，并验证其可组合性。**

- **链接: [https://arxiv.org/pdf/2602.18899](https://arxiv.org/pdf/2602.18899)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at this https URL .
>
---
#### [replaced 024] MoDora: Tree-Based Semi-Structured Document Analysis System
- **分类: cs.IR; cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出MoDora系统，解决半结构化文档分析问题，通过布局感知组件和层次结构建模，提升问答准确率。**

- **链接: [https://arxiv.org/pdf/2602.23061](https://arxiv.org/pdf/2602.23061)**

> **作者:** Bangrui Xu; Qihang Yao; Zirui Tang; Xuanhe Zhou; Yeye He; Shihan Yu; Qianqian Xu; Bin Wang; Guoliang Li; Conghui He; Fan Wu
>
> **备注:** Extension of our SIGMOD 2026 paper. Please refer to source code available at this https URL
>
> **摘要:** Semi-structured documents integrate diverse interleaved data elements (e.g., tables, charts, hierarchical paragraphs) arranged in various and often irregular layouts. These documents are widely observed across domains and account for a large portion of real-world data. However, existing methods struggle to support natural language question answering over these documents due to three main technical challenges: (1) The elements extracted by techniques like OCR are often fragmented and stripped of their original semantic context, making them inadequate for analysis. (2) Existing approaches lack effective representations to capture hierarchical structures within documents (e.g., associating tables with nested chapter titles) and to preserve layout-specific distinctions (e.g., differentiating sidebars from main content). (3) Answering questions often requires retrieving and aligning relevant information scattered across multiple regions or pages, such as linking a descriptive paragraph to table cells located elsewhere in the document. To address these issues, we propose MoDora, an LLM-powered system for semi-structured document analysis. First, we adopt a local-alignment aggregation strategy to convert OCR-parsed elements into layout-aware components, and conduct type-specific information extraction for components with hierarchical titles or non-text elements. Second, we design the Component-Correlation Tree (CCTree) to hierarchically organize components, explicitly modeling inter-component relations and layout distinctions through a bottom-up cascade summarization process. Finally, we propose a question-type-aware retrieval strategy that supports (1) layout-based grid partitioning for location-based retrieval and (2) LLM-guided pruning for semantic-based retrieval. Experiments show MoDora outperforms baselines by 5.97%-61.07% in accuracy. The code is at this https URL.
>
---
#### [replaced 025] How Psychological Learning Paradigms Shaped and Constrained Artificial Intelligence
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于人工智能理论研究，旨在解决AI系统缺乏系统性组合推理的问题。通过分析心理学习理论对AI架构的影响，提出ReSynth框架以实现结构化的系统行为。**

- **链接: [https://arxiv.org/pdf/2603.18203](https://arxiv.org/pdf/2603.18203)**

> **作者:** Alex Anvi Eponon; Ildar Batyrshin; Christian E. Maldonado-Sifuentes; Grigori Sidorov
>
> **备注:** preprint journal
>
> **摘要:** Current artificial intelligence systems struggle with systematic compositional reasoning: the capacity to recombine known components in novel configurations. This paper argues that the failure is architectural, not merely a matter of scale or training data, and that its origins lie in the psychological learning theories from which AI paradigms were derived. The argument proceeds in three stages. First, drawing on the systematicity debate in cognitive science and on the demonstration of Aizawa that neither connectionism nor classicism can make systematicity a structural consequence of the architecture, the paper establishes that the corrective techniques proliferating in modern AI, from chain-of-thought prompting to alignment through human feedback, function as auxiliary hypotheses that address symptoms without resolving the underlying architectural indifference to systematicity. Second, it traces the genealogy from psychological learning theory to AI methodology, showing that behaviourism, cognitivism, and constructivism each bequeathed a specific structural limitation to the AI paradigm it inspired: the exclusion of internal structure, the opacity of representation, and the absence of formal construction operators. A cross-cultural reappraisal of rote learning reveals a further underexploited pathway. Third, the paper introduces ReSynth, a trimodular conceptual framework that proposes the principled separation of reasoning, identity, and memory as a path toward architectures in which systematic behaviour is a structural consequence of design rather than a correction applied after the fact.
>
---
#### [replaced 026] KG-Hopper: Empowering Compact Open LLMs with Knowledge Graph Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出KG-Hopper，解决知识图谱多跳推理问题。通过强化学习，使小型语言模型在单次推理中完成全局路径探索，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2603.21440](https://arxiv.org/pdf/2603.21440)**

> **作者:** Shuai Wang; Yinan Yu
>
> **备注:** Accepted to IJCNN 2026
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive natural language capabilities but often struggle with knowledge-intensive reasoning tasks. Knowledge Base Question Answering (KBQA), which leverages structured Knowledge Graphs (KGs) exemplifies this challenge due to the need for accurate multi-hop reasoning. Existing approaches typically perform sequential reasoning steps guided by predefined pipelines, restricting flexibility and causing error cascades due to isolated reasoning at each step. To address these limitations, we propose KG-Hopper, a novel Reinforcement Learning (RL) framework that empowers compact open LLMs with the ability to perform integrated multi-hop KG reasoning within a single inference round. Rather than reasoning step-by-step, we train a Reasoning LLM that embeds the entire KG traversal and decision process into a unified ``thinking'' stage, enabling global reasoning over cross-step dependencies and dynamic path exploration with backtracking. Experimental results on eight KG reasoning benchmarks show that KG-Hopper, based on a 7B-parameter LLM, consistently outperforms larger multi-step systems (up to 70B) and achieves competitive performance with proprietary models such as GPT-3.5-Turbo and GPT-4o-mini, while remaining compact, open, and data-efficient. The code is publicly available at: this https URL.
>
---
#### [replaced 027] Double: Breaking the Acceleration Limit via Double Retrieval Speculative Parallelism
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决传统推测解码加速受限的问题。提出Double框架，通过双检索推测并行提升速度，突破理论上限。**

- **链接: [https://arxiv.org/pdf/2601.05524](https://arxiv.org/pdf/2601.05524)**

> **作者:** Yuhao Shen; Tianyu Liu; Junyi Shen; Jinyang Wu; Quan Kong; Li Huan; Cong Wang
>
> **备注:** Accepted by ACL2026 Main
>
> **摘要:** Parallel Speculative Decoding (PSD) accelerates traditional Speculative Decoding (SD) by overlapping draft generation with verification. However, it remains hampered by two fundamental challenges: (1) a theoretical speedup ceiling dictated by the speed ratio between the draft and target models, and (2) high computational waste and pipeline stall due to mid-sequence token rejections of early errors. To address these limitations, we introduce \textsc{Double} (Double Retrieval Speculative Parallelism). By bridging the gap between SD and PSD, our framework resolves the Retrieval \emph{Precision-Efficiency Dilemma} through a novel synchronous mechanism. Specifically, we enable the draft model to execute iterative retrieval speculations to break the theoretical speedup limits; to alleviate rejections without rollback, the target model performs authoritative retrieval to generate multi-token guidance. \textsc{Double} is entirely training-free and lossless. Extensive experiments demonstrate state-of-the-art speedup of $\textbf{5.3}\times$ on LLaMA3.3-70B and $\textbf{2.8}\times$ on Qwen3-32B, significantly outperforming the advanced method EAGLE-3 that requires extensive model training.
>
---
#### [replaced 028] Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决大规模多模态后训练中的数据流异构、系统鲁棒性和延迟-吞吐权衡问题。提出Relax引擎，通过三层架构实现高效异步训练。**

- **链接: [https://arxiv.org/pdf/2604.11554](https://arxiv.org/pdf/2604.11554)**

> **作者:** Liujie Zhang; Benzhe Ning; Rui Yang; Xiaoyan Yu; Jiaxing Li; Lumeng Wu; Jia Liu; Minghao Li; Weihang Chen; Weiqi Hu; Lei Zhang
>
> **备注:** 17 pages, 22 figures
>
> **摘要:** Reinforcement learning (RL) post-training has proven effective at unlocking reasoning, self-reflection, and tool-use capabilities in large language models. As models extend to omni-modal inputs and agentic multi-turn workflows, RL training systems face three interdependent challenges: heterogeneous data flows, operational robustness at scale, and the staleness -- throughput tradeoff. We present \textbf{Relax} (Reinforcement Engine Leveraging Agentic X-modality), an open-source RL training engine that addresses these challenges through three co-designed architectural layers. First, an \emph{omni-native architecture} builds multimodal support into the full stack -- from data preprocessing and modality-aware parallelism to inference generation -- rather than retrofitting it onto a text-centric pipeline. Second, each RL role runs as an independent, fault-isolated service that can be scaled, recovered, and upgraded without global coordination. Third, service-level decoupling enables asynchronous training via the TransferQueue data bus, where a single staleness parameter smoothly interpolates among on-policy, near-on-policy, and fully asynchronous execution. Relax achieves a 1.20$\times$ end-to-end speedup over veRL on Qwen3-4B on-policy training. Its fully async mode delivers a 1.76$\times$ speedup over colocate on Qwen3-4B and a 2.00$\times$ speedup on Qwen3-Omni-30B, while all modes converge to the same reward level. Relax supports R3 (Rollout Routing Replay)~\cite{ma2025r3} for MoE models with only 1.9\% overhead, compared to 32\% degradation in veRL under the same configuration. It further demonstrates stable omni-modal RL convergence on Qwen3-Omni across image, text, and audio, sustaining over 2{,}000 steps on video without degradation. Relax is available at this https URL.
>
---
#### [replaced 029] LLM as Attention-Informed NTM and Topic Modeling as long-input Generation: Interpretability and long-Context Capability
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLM在主题建模中的应用，解决传统NTM模型表达能力不足的问题。通过白盒和黑盒方法，提升主题可解释性和长文本处理能力。**

- **链接: [https://arxiv.org/pdf/2510.03174](https://arxiv.org/pdf/2510.03174)**

> **作者:** Xuan Xu; Zhongliang Yang; Haolun Li; Beilin Chu; Rui Tian; Yu Li; Shaolin Tan; Linna Zhou
>
> **摘要:** Topic modeling aims to produce interpretable topic representations and topic--document correspondences from corpora, but classical neural topic models (NTMs) remain constrained by limited representation assumptions and semantic abstraction ability. We study LLM-based topic modeling from both white-box and black-box perspectives. For white-box LLMs, we propose an attention-informed framework that recovers interpretable structures analogous to those in NTMs, including document-topic and topic-word distributions. This validates the view that LLM can serve as an attention-informed NTM. For black-box LLMs, we reformulate topic modeling as a structured long-input task and introduce a post-generation signal compensation method based on diversified topic cues and hybrid retrieval. Experiments show that recovered attention structures support effective topic assignment and keyword extraction, while black-box long-context LLMs achieve competitive or stronger performance than other baselines. These findings suggest a connection between LLMs and NTMs and highlight the promise of long-context LLMs for topic modeling.
>
---
#### [replaced 030] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation
- **分类: cs.CL**

- **简介: 该论文属于AI心理健康评估任务，旨在解决AI对话系统在治疗原则上的评估问题。工作包括构建FAITH-M基准和提出CARE框架，提升对临床适宜性的评价效果。**

- **链接: [https://arxiv.org/pdf/2604.05795](https://arxiv.org/pdf/2604.05795)**

> **作者:** Abdullah Mazhar; Het Riteshkumar Shah; Aseem Srivastava; Smriti Joshi; Md Shad Akhtar
>
> **备注:** Accepted at ACL 2026 (Main)
>
> **摘要:** The increasing use of large language models in mental health applications calls for principled evaluation frameworks that assess alignment with psychotherapeutic best practices beyond surface-level fluency. While recent systems exhibit conversational competence, they lack structured mechanisms to evaluate adherence to core therapeutic principles. In this paper, we study the problem of evaluating AI-generated therapist-like responses for clinically grounded appropriateness and effectiveness. We assess each therapists utterance along six therapeutic principles: non-judgmental acceptance, warmth, respect for autonomy, active listening, reflective understanding, and situational appropriateness using a fine-grained ordinal scale. We introduce FAITH-M, a benchmark annotated with expert-assigned ordinal ratings, and propose CARE, a multi-stage evaluation framework that integrates intra-dialogue context, contrastive exemplar retrieval, and knowledge-distilled chain-of-thought reasoning. Experiments show that CARE achieves an F-1 score of 63.34 versus the strong baseline Qwen3 F-1 score of 38.56 which is a 64.26 improvement, which also serves as its backbone, indicating that gains arise from structured reasoning and contextual modeling rather than backbone capacity alone. Expert assessment and external dataset evaluations further demonstrate robustness under domain shift, while highlighting challenges in modelling implicit clinical nuance. Overall, CARE provides a clinically grounded framework for evaluating therapeutic fidelity in AI mental health systems.
>
---
#### [replaced 031] Graph-Based Chain-of-Thought Pruning for Reducing Redundant Reflections in Reasoning LLMs
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决LLMs中冗余思考问题。通过图结构优化CoT，减少无效反思，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.05643](https://arxiv.org/pdf/2604.05643)**

> **作者:** Hongyuan Yuan; Xinran He; Run Shao; Bolei He; Xianwei Xue; Mengke Chen; Qiutong Pan; Haiwei Wang; Haifeng Li
>
> **备注:** Accepted by ACL2026 Findings
>
> **摘要:** Extending CoT through RL has been widely used to enhance the reasoning capabilities of LLMs. However, due to the sparsity of reward signals, it can also induce undesirable thinking patterns such as overthinking, i.e., generating redundant intermediate reasoning content. In this work, we argue that a major source of such redundancy is inefficient reflection, which often manifests in two problematic patterns: Indiscriminate Reflection, where the model performs broad, low-impact checks throughout reasoning, and Repetitive Reflection, where it repeatedly re-verifies an already established conclusion. To address this, we introduce a graph-based CoT optimization framework. Specifically, we convert each linear CoT into a directed acyclic graph (DAG) with explicit dependency edges, and design a dual pruning strategy: branch-level pruning removes weakly contributing reflection branches, while depth-level pruning eliminates late-stage re-verification. We distill this behavior via a three-stage pipeline: (1) SFT to initialize the policy on pruned concise traces, (2) DPO to prefer correct but less redundant trajectories, and (3) GRPO with length penalty to jointly optimize answer correctness and efficiency. Experiments show that our approach reduces the average reasoning tokens by 42\% while maintaining or improving accuracy.
>
---
#### [replaced 032] Large Language Models are Powerful Electronic Health Record Encoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗AI任务，旨在解决EHR复杂性带来的预测难题。通过将EHR转为文本，利用LLM生成嵌入，实现高效预测，对比显示其效果与专业模型相当，且具更好泛化能力。**

- **链接: [https://arxiv.org/pdf/2502.17403](https://arxiv.org/pdf/2502.17403)**

> **作者:** Stefan Hegselmann; Georg von Arnim; Tillmann Rheude; Noel Kronenberg; David Sontag; Gerhard Hindricks; Roland Eils; Benjamin Wild
>
> **摘要:** Electronic Health Records (EHRs) offer considerable potential for clinical prediction, but their complexity and heterogeneity challenge traditional machine learning. Domain-specific EHR foundation models trained on unlabeled EHR data have shown improved predictive accuracy and generalization. However, their development is constrained by limited data access and site-specific vocabularies. We convert EHR data into plain text by replacing medical codes with natural-language descriptions, enabling general-purpose Large Language Models (LLMs) to produce high-dimensional embeddings for downstream prediction tasks without access to private medical training data. LLM-based embeddings perform on par with a specialized EHR foundation model, CLMBR-T-Base, across 15 clinical tasks from the EHRSHOT benchmark. In an external validation using the UK Biobank, an LLM-based model shows statistically significant improvements for some tasks, which we attribute to higher vocabulary coverage and slightly better generalization. Overall, we reveal a trade-off between the computational efficiency of specialized EHR models and the portability and data independence of LLM-based embeddings.
>
---
#### [replaced 033] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型对拼写错误的鲁棒性，属于自然语言处理任务。针对现有基准未考虑输入噪声的问题，提出MulTypo生成多语言拼写错误，并评估18个模型在多个任务中的表现。**

- **链接: [https://arxiv.org/pdf/2510.09536](https://arxiv.org/pdf/2510.09536)**

> **作者:** Raoyuan Zhao; Yihong Liu; Lena Altinger; Hinrich Schütze; Michael A. Hedderich
>
> **备注:** ACL 2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed in multilingual, real-world applications with user inputs -- naturally introducing \emph{typographical errors} (typos). Yet most benchmarks assume clean input, leaving the robustness of LLMs to typos across languages largely underexplored. To address this gap, we introduce MulTypo, a multilingual typo generation algorithm that simulates human-like errors based on language-specific keyboard layouts and typing behavior. We evaluate 18 open-source LLMs across three model families and five downstream tasks spanning language inference, multi-choice question answering, mathematical reasoning, and machine translation tasks. Our results show that typos consistently degrade performance, particularly in generative tasks and those requiring reasoning -- while the natural language inference task is comparatively more robust. Instruction tuning improves clean-input performance but may increase brittleness under noise. We also observe language-dependent robustness: high-resource languages are generally more robust than low-resource ones, and translation from English is more robust than translation into English. Our findings underscore the need for noise-aware training and multilingual robustness evaluation. We release a Python package for MulTypo and make the source code publicly available at this https URL.
>
---
#### [replaced 034] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音对话生成任务，解决传统模型推理慢、稳定性差的问题，提出ZipVoice-Dialog模型，采用非自回归和流匹配方法，提升生成速度与准确性。**

- **链接: [https://arxiv.org/pdf/2507.09318](https://arxiv.org/pdf/2507.09318)**

> **作者:** Han Zhu; Wei Kang; Liyong Guo; Zengwei Yao; Fangjun Kuang; Weiji Zhuang; Zhaoqing Li; Zhifeng Han; Dong Zhang; Xin Zhang; Xingchen Song; Lingxuan Ye; Long Lin; Daniel Povey
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Generating spoken dialogue is inherently more complex than monologue text-to-speech (TTS), as it demands both realistic turn-taking and the maintenance of distinct speaker timbres. While existing autoregressive (AR) models have made progress, they often suffer from high inference latency and stability issues. To overcome these limitations, we propose ZipVoice-Dialog, a non-autoregressive (NAR) zero-shot spoken dialogue generation model based on flow-matching. Observing that applying vanilla flow-matching to dialogue generation leads to poor speech intelligibility and turn-taking precision, we introduce two simple yet effective methods to adapt flow-matching architectures for dialogue generation: (1) a curriculum learning strategy to ensure robust speech-text alignment, and (2) speaker-turn embeddings to govern precise speaker turn-taking. Additionally, we introduce dedicated strategies to support stereo dialogue generation. Recognizing the lack of training datasets in this field, we curate and release OpenDialog, the first large-scale (6.8k hours) open-source spoken dialogue dataset derived from in-the-wild speech data. Moreover, for fair and rigorous evaluations, we established a benchmark to comprehensively evaluate dialogue generation models. Experiments demonstrate the effectiveness of the proposed methods and dataset, showing that ZipVoice-Dialog achieves superior performance in inference speed, intelligibility, speaker turn-taking accuracy, and speaker similarity. Our code, model checkpoints, and the OpenDialog dataset are publicly available at this https URL.
>
---
#### [replaced 035] AdaMCoT: Rethinking Cross-Lingual Factual Reasoning through Adaptive Multilingual Chain-of-Thought
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言事实推理任务，旨在解决多语言模型性能不平衡问题。提出AdaMCOT框架，通过自适应多语言思维链提升推理质量与一致性。**

- **链接: [https://arxiv.org/pdf/2501.16154](https://arxiv.org/pdf/2501.16154)**

> **作者:** Weihua Zheng; Xin Huang; Zhengyuan Liu; Tarun Kumar Vangani; Bowei Zou; Xiyan Tao; Yuhao Wu; Ai Ti Aw; Nancy F. Chen; Roy Ka-Wei Lee
>
> **备注:** AAAI 2026
>
> **摘要:** Large language models (LLMs) have shown impressive multilingual capabilities through pretraining on diverse corpora. Although these models show strong reasoning abilities, their performance varies significantly between languages due to the imbalanced distribution of training data. Existing approaches using sample-level translation for extensive multilingual pretraining and cross-lingual tuning face scalability challenges and often fail to capture nuanced reasoning processes across languages. In this paper, we introduce AdaMCOT (Adaptive Multilingual Chain-of-Thought), a framework that enhances multilingual factual reasoning by dynamically routing thought processes in intermediary "thinking languages" before generating target-language responses. AdaMCOT leverages a language-agnostic core and incorporates an adaptive, reward-based mechanism for selecting optimal reasoning pathways without requiring additional pretraining. Our comprehensive evaluation across multiple benchmarks demonstrates substantial improvements in both factual reasoning quality and cross-lingual consistency, with particularly strong performance gains in low-resource language settings. An in-depth analysis of the model's hidden states and semantic space further elucidates the underlying mechanism of our method. The results suggest that adaptive reasoning paths can effectively bridge the performance gap between high and low-resource languages while maintaining cultural and linguistic nuances.
>
---
#### [replaced 036] ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多硬件通用智能体构建难题。通过数据标准化与动作流形学习，提升动作预测效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.11236](https://arxiv.org/pdf/2602.11236)**

> **作者:** Yandan Yang; Shuang Zeng; Tong Lin; Xinyuan Chang; Dekang Qi; Junjin Xiao; Haoyun Liu; Ronghan Chen; Yuzhi Chen; Dongjie Huo; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **备注:** Project website: this https URL . Code: this https URL . 22 pages, 10 figures, 10 tables
>
> **摘要:** Building general-purpose embodied agents across diverse hardware remains a central challenge in robotics, often framed as the ''one-brain, many-forms'' paradigm. Progress is hindered by fragmented data, inconsistent representations, and misaligned training objectives. We present ABot-M0, a framework that builds a systematic data curation pipeline while jointly optimizing model architecture and training strategies, enabling end-to-end transformation of heterogeneous raw data into unified, efficient representations. From six public datasets, we clean, standardize, and balance samples to construct UniACT-dataset, a large-scale dataset with over 6 million trajectories and 9,500 hours of data, covering diverse robot morphologies and task scenarios. Unified pre-training improves knowledge transfer and generalization across platforms and tasks, supporting general-purpose embodied intelligence. To improve action prediction efficiency and stability, we propose the Action Manifold Hypothesis: effective robot actions lie not in the full high-dimensional space but on a low-dimensional, smooth manifold governed by physical laws and task constraints. Based on this, we introduce Action Manifold Learning (AML), which uses a DiT backbone to predict clean, continuous action sequences directly. This shifts learning from denoising to projection onto feasible manifolds, improving decoding speed and policy stability. ABot-M0 supports modular perception via a dual-stream mechanism that integrates VLM semantics with geometric priors and multi-view inputs from plug-and-play 3D modules such as VGGT and Qwen-Image-Edit, enhancing spatial understanding without modifying the backbone and mitigating standard VLM limitations in 3D reasoning. Experiments show components operate independently with additive benefits. We will release all code and pipelines for reproducibility and future research.
>
---
#### [replaced 037] Speaker effects in language comprehension: An integrative model of language and speaker processing
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于语言理解研究，探讨说话者效应如何影响语言处理。提出整合模型，解决说话者身份与语言理解的交互机制，结合感知与预期过程。**

- **链接: [https://arxiv.org/pdf/2412.07238](https://arxiv.org/pdf/2412.07238)**

> **作者:** Hanlin Wu; Zhenguang G. Cai
>
> **摘要:** The identity of a speaker influences language comprehension through modulating perception and expectation. This review explores speaker effects and proposes an integrative model of language and speaker processing that integrates distinct mechanistic perspectives. We argue that speaker effects arise from the interplay between bottom-up perception-based processes, driven by acoustic-episodic memory, and top-down expectation-based processes, driven by a speaker model. We show that language and speaker processing are functionally integrated through multi-level probabilistic processing: prior beliefs about a speaker modulate language processing at the phonetic, lexical, and semantic levels, while the unfolding speech and message continuously update the speaker model, refining broad demographic priors into precise individualized representations. Within this framework, we distinguish between speaker-idiosyncrasy effects arising from familiarity with an individual and speaker-demographics effects arising from social group expectations. We discuss how speaker effects serve as indices for assessing language development and social cognition, and we encourage future research to extend these findings to the emerging domain of artificial intelligence (AI) speakers, as AI agents represent a new class of social interlocutors that are transforming the way we engage in communication.
>
---
#### [replaced 038] Sparse Growing Transformer: Training-Time Sparse Depth Allocation via Progressive Attention Looping
- **分类: cs.CL**

- **简介: 该论文属于Transformer模型优化任务，旨在解决训练中计算冗余问题。通过动态分配深度，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2603.23998](https://arxiv.org/pdf/2603.23998)**

> **作者:** Yao Chen; Yilong Chen; Yinqi Yang; Junyuan Shang; Zhenyu Zhang; Zefeng Zhang; Shuaiyi Nie; Shuohuan Wang; Yu Sun; Hua Wu; HaiFeng Wang; Tingwen Liu
>
> **摘要:** Existing approaches to increasing the effective depth of Transformers predominantly rely on parameter reuse, extending computation through recursive execution. Under this paradigm, the network structure remains static along the training timeline, and additional computational depth is uniformly assigned to entire blocks at the parameter level. This rigidity across training time and parameter space leads to substantial computational redundancy during training. In contrast, we argue that depth allocation during training should not be a static preset, but rather a progressively growing structural process. Our systematic analysis reveals a deep-to-shallow maturation trajectory across layers, where high-entropy attention heads play a crucial role in semantic integration. Motivated by this observation, we introduce the Sparse Growing Transformer (SGT). SGT is a training-time sparse depth allocation framework that progressively extends recurrence from deeper to shallower layers via targeted attention looping on informative heads. This mechanism induces structural sparsity by selectively increasing depth only for a small subset of parameters as training evolves. Extensive experiments across multiple parameter scales demonstrate that SGT consistently outperforms training-time static block-level looping baselines under comparable settings, while reducing the additional training FLOPs overhead from approximately 16--20% to only 1--3% relative to a standard Transformer backbone.
>
---
#### [replaced 039] DyBBT: Dynamic Balance via Bandit-inspired Targeting for Dialog Policy with Cognitive Dual-Systems
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于对话系统任务，解决静态探索策略效率低的问题。提出DyBBT框架，通过动态切换推理模式提升对话性能。**

- **链接: [https://arxiv.org/pdf/2509.19695](https://arxiv.org/pdf/2509.19695)**

> **作者:** Shuyu Zhang; Yifan Wei; Jialuo Yuan; Xinru Wang; Yanmin Zhu; Bin Li; Yujie Liu
>
> **备注:** Accepted in ACL2026 main
>
> **摘要:** Task oriented dialog systems often rely on static exploration strategies that do not adapt to dynamic dialog contexts, leading to inefficient exploration and suboptimal performance. We propose DyBBT, a novel dialog policy learning framework that formalizes the exploration challenge through a structured cognitive state space capturing dialog progression, user uncertainty, and slot dependency. DyBBT proposes a bandit inspired meta-controller that dynamically switches between a fast intuitive inference (System 1) and a slow deliberative reasoner (System 2) based on real-time cognitive states and visitation counts. Extensive experiments on single- and multi-domain benchmarks show that DyBBT achieves state-of-the-art performance in success rate, efficiency, and generalization, with human evaluations confirming its decisions are well aligned with expert judgment.
>
---
#### [replaced 040] Retrieval as a Decision: Training-Free Adaptive Gating for Efficient RAG
- **分类: cs.CL**

- **简介: 该论文提出TARG方法，用于高效RAG任务中决定何时检索。解决频繁检索导致效率低的问题，通过分析模型生成的草稿决定是否触发检索，提升效率同时保持准确性。**

- **链接: [https://arxiv.org/pdf/2511.09803](https://arxiv.org/pdf/2511.09803)**

> **作者:** Yufeng Wang; Lu wei; Haibin Ling
>
> **摘要:** Retrieval-Augmented Generation (RAG) improves factuality but retrieving for every query often hurts quality while inflating tokens and latency. We propose Training-free Adaptive Retrieval Gating (TARG), a single-shot policy that decides when to retrieve using only a short, no-context draft from the base model. From the draft's prefix logits, TARG computes lightweight uncertainty scores-mean token entropy, a margin signal derived from the top-1/top-2 logit gap via a monotone link, or small-N variance across a handful of stochastic prefixes-and triggers retrieval only when the score exceeds a threshold. The gate is model-agnostic, adds only tens to hundreds of draft tokens, and requires no additional training or auxiliary heads. On five QA benchmarks spanning short-answer (NQ-Open, TriviaQA, PopQA), multi-hop (MuSiQue), and long-form (ASQA) tasks, TARG consistently pushes the accuracy-efficiency frontier: compared with Alway-RAG, TARG matches or improves EM/F1 while reducing retrieval by 70-90% and cutting end-to-end latency, and it remains close to Never-RAG in overhead. A central empirical finding is that under modern instruction-tuned LLMs the margin signal is a robust default (entropy compresses as backbones sharpen), with small-N variance offering a conservative, budget-first alternative. We provide ablations over gate type and prefix length and use a $\Delta$-latency view to make budget trade-offs explicit.
>
---
#### [replaced 041] HiCoLoRA: Addressing Context-Prompt Misalignment via Hierarchical Collaborative LoRA for Zero-Shot DST
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于零样本对话状态追踪任务，解决动态上下文与静态提示语义不匹配问题。提出HiCoLoRA框架，通过层次化LoRA和联合聚类提升跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.19742](https://arxiv.org/pdf/2509.19742)**

> **作者:** Shuyu Zhang; Yifan Wei; Xinru Wang; Yanmin Zhu; Yangfan He; Yixuan Weng; Bin Li; Yujie Liu
>
> **备注:** Accepted in ACL2026 findings
>
> **摘要:** Zero-shot Dialog State Tracking (zs-DST) is essential for enabling Task-Oriented Dialog Systems (TODs) to generalize to new domains without costly data annotation. A central challenge lies in the semantic misalignment between dynamic dialog contexts and static prompts, leading to inflexible cross-layer coordination, domain interference, and catastrophic forgetting. To tackle this, we propose Hierarchical Collaborative Low-Rank Adaptation (HiCoLoRA), a framework that enhances zero-shot slot inference through robust prompt alignment. It features a hierarchical LoRA architecture for dynamic layer-specific processing (combining lower-layer heuristic grouping and higher-layer full interaction), integrates Spectral Joint Domain-Slot Clustering to identify transferable associations (feeding an Adaptive Linear Fusion Mechanism), and employs Semantic-Enhanced SVD Initialization (SemSVD-Init) to preserve pre-trained knowledge. Experiments on multi-domain datasets MultiWOZ and SGD show that HiCoLoRA outperforms baselines, achieving SOTA in zs-DST. Code is available at this https URL.
>
---
#### [replaced 042] Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决LLM决策不透明问题。通过“定位、操控、优化”框架，提出系统化干预方法，提升模型对齐性、能力与效率。**

- **链接: [https://arxiv.org/pdf/2601.14004](https://arxiv.org/pdf/2601.14004)**

> **作者:** Hengyuan Zhang; Zhihao Zhang; Mingyang Wang; Zunhai Su; Yiwei Wang; Qianli Wang; Shuzhou Yuan; Ercong Nie; Xufeng Duan; Feijiang Han; Qibo Xue; Zeping Yu; Chenming Shang; Xiao Liang; Jing Xiong; Hui Shen; Chaofan Tao; Zhengwu Liu; Senjie Jin; Zhiheng Xi; Dongdong Zhang; Sophia Ananiadou; Tao Gui; Ruobing Xie; Hayden Kwok-Hay So; Hinrich Schütze; Xuanjing Huang; Qi Zhang; Ngai Wong
>
> **摘要:** Mechanistic Interpretability (MI) has emerged as a vital approach to demystify the opaque decision-making of Large Language Models (LLMs). However, existing reviews primarily treat MI as an observational science, summarizing analytical insights while lacking a systematic framework for actionable intervention. To bridge this gap, we present a practical survey structured around the pipeline: "Locate, Steer, and Improve." We formally categorize Localizing (diagnosis) and Steering (intervention) methods based on specific Interpretable Objects to establish a rigorous intervention protocol. Furthermore, we demonstrate how this framework enables tangible improvements in Alignment, Capability, and Efficiency, effectively operationalizing MI as an actionable methodology for model optimization. The curated paper list of this work is available at this https URL.
>
---
#### [replaced 043] LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言建模任务，旨在解决连续扩散模型在语言建模中性能不如离散模型的问题。通过创新设计，提出LangFlow，实现连续扩散与离散模型相当的性能。**

- **链接: [https://arxiv.org/pdf/2604.11748](https://arxiv.org/pdf/2604.11748)**

> **作者:** Yuxin Chen; Chumeng Liang; Hangke Sui; Ruihan Guo; Chaoran Cheng; Jiaxuan You; Ge Liu
>
> **摘要:** Continuous diffusion has been the foundation of high-fidelity, controllable, and few-step generation of many data modalities such as images. However, in language modeling, prior continuous diffusion language models (DLMs) lag behind discrete counterparts due to the sparse data space and the underexplored design space. In this work, we close this gap with LangFlow, the first continuous DLM to rival discrete diffusion, by connecting embedding-space DLMs to Flow Matching via Bregman divergence, alongside three key innovations: (1) we derive a novel ODE-based NLL bound for principled evaluation of continuous flow-based language models; (2) we propose an information-uniform principle for setting the noise schedule, which motivates a learnable noise scheduler based on a Gumbel distribution; and (3) we revise prior training protocols by incorporating self-conditioning, as we find it improves both likelihood and sample quality of embedding-space DLMs with effects substantially different from discrete diffusion. Putting everything together, LangFlow rivals top discrete DLMs on both the perplexity (PPL) and the generative perplexity (Gen. PPL), reaching a PPL of 30.0 on LM1B and 24.6 on OpenWebText. It even exceeds autoregressive baselines in zero-shot transfer on 4 out of 7 benchmarks. LangFlow provides the first clear evidence that continuous diffusion is a promising paradigm for language modeling. Homepage: this https URL
>
---
#### [replaced 044] JanusCoder: Towards a Foundational Visual-Programmatic Interface for Code Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.SE**

- **简介: 该论文提出JanusCoder，解决代码与视觉输出的多模态生成问题。通过构建大规模数据集，训练统一模型实现文本或视觉输入生成代码，提升代码智能应用效果。**

- **链接: [https://arxiv.org/pdf/2510.23538](https://arxiv.org/pdf/2510.23538)**

> **作者:** Qiushi Sun; Jingyang Gong; Yang Liu; Qiaosheng Chen; Lei Li; Kai Chen; Qipeng Guo; Ben Kao; Fei Yuan
>
> **备注:** ICLR 2026 Camera Ready Version
>
> **摘要:** The scope of neural code intelligence is rapidly expanding beyond text-based source code to encompass the rich visual outputs that programs generate. This visual dimension is critical for advanced applications like flexible content generation and precise, program-driven editing of visualizations. However, progress has been impeded by the scarcity of high-quality multimodal code data, a bottleneck stemming from challenges in synthesis and quality assessment. To address these challenges, we make contributions from both a data and modeling perspective. We first introduce a complete synthesis toolkit that leverages reciprocal synergies between data modalities to efficiently produce a large-scale, high-quality corpus spanning from standard charts to complex interactive web UIs and code-driven animations. Leveraging this toolkit, we construct JanusCode-800K, the largest multimodal code corpus to date. This powers the training of our models, JanusCoder and JanusCoderV, which establish a visual-programmatic interface for generating code from textual instructions, visual inputs, or a combination of both. Our unified model is a departure from existing approaches that build specialized models for isolated tasks. Extensive experiments on both text-centric and vision-centric coding tasks demonstrate the superior performance of the JanusCoder series, with our 7B to 14B scale models approaching or even exceeding the performance of commercial models. Furthermore, extensive analysis provides key insights into harmonizing programmatic logic with its visual expression. Our code and checkpoints are available at this https URL.
>
---
#### [replaced 045] Generation-Augmented Generation: A Plug-and-Play Framework for Private Knowledge Injection in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于知识注入任务，解决私有领域知识高效注入大模型的问题。提出GAG框架，通过生成增强生成方式，在不修改基础模型的前提下，提升专业问答性能。**

- **链接: [https://arxiv.org/pdf/2601.08209](https://arxiv.org/pdf/2601.08209)**

> **作者:** Rongji Li; Jian Xu; Yi Chen; Xueqing Chen; Yisheng Yang; Jiayi Wang; Xingyu Chen; Chunyu Xie; Dawei Leng; Xu-Yao Zhang
>
> **摘要:** In domains such as materials science, biomedicine, and finance, high-stakes deployment of large language models (LLMs) requires injecting private, domain-specific knowledge that is proprietary, fast-evolving, and under-represented in public pretraining. However, the two dominant paradigms for private knowledge injection each have clear drawbacks: fine-tuning is expensive to iterate under continual updates that can induce catastrophic forgetting and general-capability regression; retrieval-augmented generation (RAG) keeps the base model intact but remains brittle in specialized private corpora due to chunk-induced evidence fragmentation, retrieval mismatch, and long-context pressure. Inspired by how multimodal LLMs align heterogeneous modalities into a shared semantic space, we propose Generation-Augmented Generation (GAG), which treats private expertise as an auxiliary modality and injects it into a frozen base model through a compact, constant-budget latent interface. Concretely, GAG distills question-conditioned specialist knowledge from lightweight domain experts into multi-slot latent memories, integrates multi-layer expert signals via per-slot cross-layer fusion, and aligns them to the frozen base model through gated residual projection, while supporting scalable mixed-domain deployment with reliable selective activation. In a unified mixed-domain evaluation spanning two scientific private-domain QA benchmarks (catalytic materials and immunology adjuvant) together with general-domain queries, GAG consistently outperforms strong retrieval-based and parameter-efficient fine-tuning baselines on specialist QA, while preserving general-domain capability, achieving highly reliable routing, and offering a favorable efficiency--effectiveness trade-off. Code and datasets are provided in the supplementary material. Code is publicly available at this https URL.
>
---
#### [replaced 046] CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识图谱推理任务，旨在解决LLM在知识图谱辅助下的推理不稳定和停滞问题。提出CoG框架，结合关系蓝图和故障感知优化，提升推理准确性和效率。**

- **链接: [https://arxiv.org/pdf/2601.11047](https://arxiv.org/pdf/2601.11047)**

> **作者:** Yuanxiang Liu; Songze Li; Xiaoke Guo; Zhaoyan Gong; Qifei Zhang; Huajun Chen; Wen Zhang
>
> **备注:** ACL 2026 Main
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities but often grapple with reliability challenges like hallucinations. While Knowledge Graphs (KGs) offer explicit grounding, existing paradigms of KG-augmented LLMs typically exhibit cognitive rigidity--applying homogeneous search strategies that render them vulnerable to instability under neighborhood noise and structural misalignment leading to reasoning stagnation. To address these challenges, we propose CoG, a training-free framework inspired by Dual-Process Theory that mimics the interplay between intuition and deliberation. First, functioning as the fast, intuitive process, the Relational Blueprint Guidance module leverages relational blueprints as interpretable soft structural constraints to rapidly stabilize the search direction against noise. Second, functioning as the prudent, analytical process, the Failure-Aware Refinement module intervenes upon encountering reasoning impasses. It triggers evidence-conditioned reflection and executes controlled backtracking to overcome reasoning stagnation. Experimental results on three benchmarks demonstrate that CoG significantly outperforms state-of-the-art approaches in both accuracy and efficiency.
>
---
#### [replaced 047] GigaCheck: Detecting LLM-generated Content via Object-Centric Span Localization
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决生成内容与人类写作难以区分的问题。提出GigaCheck框架，结合文档和段落级检测方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2410.23728](https://arxiv.org/pdf/2410.23728)**

> **作者:** Irina Tolstykh; Aleksandra Tsybina; Sergey Yakubson; Aleksandr Gordeev; Vladimir Dokholyan; Maksim Kuprashevich
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** With the increasing quality and spread of LLM assistants, the amount of generated content is growing rapidly. In many cases and tasks, such texts are already indistinguishable from those written by humans, and the quality of generation continues to increase. At the same time, detection methods are advancing more slowly than generation models, making it challenging to prevent misuse of generative AI technologies. We propose GigaCheck, a dual-strategy framework for AI-generated text detection. At the document level, we leverage the representation learning of fine-tuned LLMs to discern authorship with high data efficiency. At the span level, we introduce a novel structural adaptation that treats generated text segments as "objects." By integrating a DETR-like vision model with linguistic encoders, we achieve precise localization of AI intervals, effectively transferring the robustness of visual object detection to the textual domain. Experimental results across three classification and three localization benchmarks confirm the robustness of our approach. The shared fine-tuned backbone delivers strong accuracy in both scenarios, highlighting the generalization power of the learned embeddings. Moreover, we successfully demonstrate that visual detection architectures like DETR are not limited to pixel space, effectively generalizing to the localization of generated text spans. To ensure reproducibility and foster further research, we publicly release our source code.
>
---
#### [replaced 048] PILOT: Planning via Internalized Latent Optimization Trajectories for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出PILOT框架，解决大语言模型在长任务中缺乏全局策略的问题。通过内部潜向量引导，提升推理稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2601.19917](https://arxiv.org/pdf/2601.19917)**

> **作者:** Haoyu Zheng; Yun Zhu; Yuqian Yuan; Bo Yuan; Wenqiao Zhang; Siliang Tang; Jun Xiao
>
> **摘要:** Strategic planning is critical for multi-step reasoning, yet compact Large Language Models (LLMs) often lack the capacity to formulate global strategies, leading to error propagation in long-horizon tasks. Our analysis reveals that LLMs possess latent reasoning capabilities that can be unlocked when conditioned on explicit plans from a teacher model; however, runtime reliance on external guidance is often impractical due to latency and availability constraints. To bridge this gap, we propose PILOT (Planning via Internalized Latent Optimization Trajectories), a non-invasive framework designed to internalize the strategic oversight of large models into intrinsic Latent Guidance. Instead of altering backbone weights, PILOT employs a lightweight Hyper-Network to synthesize a query-conditioned Latent Guidance vector. This vector acts as an internal steering mechanism, guiding the model's representations toward optimal reasoning paths. Extensive experiments on mathematical and coding benchmarks demonstrate that PILOT effectively stabilizes reasoning trajectories, consistently outperforming strong baselines (e.g., +8.9% on MATH500) with negligible inference latency.
>
---
#### [replaced 049] KCLarity at SemEval-2026 Task 6: Encoder and Zero-Shot Approaches to Political Evasion Detection
- **分类: cs.CL**

- **简介: 该论文属于SemEval-2026政治话语模糊与回避分类任务，旨在检测政治言论中的模糊和回避现象。研究提出两种建模方法，并探索了零样本设置下的模型表现。**

- **链接: [https://arxiv.org/pdf/2603.06552](https://arxiv.org/pdf/2603.06552)**

> **作者:** Archie Sage; Salvatore Greco
>
> **备注:** Camera-ready version to appear in the SemEval 2026 Proceedings
>
> **摘要:** This paper describes the KCLarity team's participation in CLARITY, a shared task at SemEval 2026 on classifying ambiguity and evasion techniques in political discourse. We investigate two modelling formulations: (i) directly predicting the clarity label, and (ii) predicting the evasion label and deriving clarity through the task taxonomy hierarchy. We further explore several auxiliary training variants and evaluate decoder-only models in a zero-shot setting under the evasion-first formulation. Overall, the two formulations yield comparable performance. Among encoder-based models, RoBERTa-large achieves the strongest results on the public test set, while zero-shot GPT-5.2 generalises better on the hidden evaluation set.
>
---
#### [replaced 050] Why Did Apple Fall: Evaluating Curiosity in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在探讨大语言模型是否具备类似人类的求知欲。通过设计评估框架，分析模型在不同维度上的好奇心表现，并验证其对推理能力的影响。**

- **链接: [https://arxiv.org/pdf/2510.20635](https://arxiv.org/pdf/2510.20635)**

> **作者:** Haoyu Wang; Sihang Jiang; Yuyan Chen; Xiaojun Meng; Jiansheng Wei; Yitong Wang; Yanghua Xiao
>
> **备注:** ACL 2026 findings paper
>
> **摘要:** Curiosity serves as a pivotal conduit for human beings to discover and learn new knowledge. Recent advancements of large language models (LLMs) in natural language processing have sparked discussions regarding whether these models possess capability of curiosity-driven learning akin to humans. In this paper, starting from the human curiosity assessment questionnaire Five-Dimensional Curiosity scale Revised (5DCR), we design a comprehensive evaluation framework that covers dimensions such as Information Seeking, Thrill Seeking, and Social Curiosity to assess the extent of curiosity exhibited by LLMs. The results demonstrate that LLMs exhibit a stronger thirst for knowledge than humans but still tend to make conservative choices when faced with uncertain environments. We further investigated the relationship between curiosity and thinking of LLMs, confirming that curious behaviors can enhance the model's reasoning and active learning abilities. These findings suggest that LLMs have the potential to exhibit curiosity similar to that of humans, providing experimental support for the future development of learning capabilities and innovative research in LLMs.
>
---
#### [replaced 051] AGSC: Adaptive Granularity and Semantic Clustering for Uncertainty Quantification in Long-text Generation
- **分类: cs.CL**

- **简介: 该论文属于长文本生成中的不确定性量化任务，旨在解决幻觉问题。提出AGSC框架，通过自适应粒度和语义聚类提升可靠性评估效率。**

- **链接: [https://arxiv.org/pdf/2604.06812](https://arxiv.org/pdf/2604.06812)**

> **作者:** Guanran Luo; Wentao Qiu; Wanru Zhao; Wenhan Lv; Zhongquan Jian; Meihong Wang; Qingqiang Wu
>
> **备注:** Accepted to the Main Conference of ACL 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in long-form generation, yet their application is hindered by the hallucination problem. While Uncertainty Quantification (UQ) is essential for assessing reliability, the complex structure makes reliable aggregation across heterogeneous themes difficult, in addition, existing methods often overlook the nuance of neutral information and suffer from the high computational cost of fine-grained decomposition. To address these challenges, we propose AGSC (Adaptive Granularity and GMM-based Semantic Clustering), a UQ framework tailored for long-form generation. AGSC first uses NLI neutral probabilities as triggers to distinguish irrelevance from uncertainty, reducing unnecessary computation. It then applies Gaussian Mixture Model (GMM) soft clustering to model latent semantic themes and assign topic-aware weights for downstream aggregation. Experiments on BIO and LongFact show that AGSC achieves state-of-the-art correlation with factuality while reducing inference time by about 60% compared to full atomic decomposition.
>
---
#### [replaced 052] OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning
- **分类: cs.LG; cs.CL; cs.CV; cs.MA**

- **简介: 该论文提出OctoTools，一个无需训练的多智能体框架，用于解决跨领域的复杂推理任务。针对现有方法在工具类型和领域适应性上的不足，OctoTools通过标准化工具卡、规划器和执行器实现高效多步骤问题解决。**

- **链接: [https://arxiv.org/pdf/2502.11271](https://arxiv.org/pdf/2502.11271)**

> **作者:** Pan Lu; Bowen Chen; Sheng Liu; Rahul Thapa; Joseph Boen; James Zou
>
> **备注:** 88 pages, 18 figures. Accepted to ACL 2026
>
> **摘要:** Solving complex reasoning tasks may involve visual understanding, domain knowledge retrieval, numerical calculation, and multi-step reasoning. Existing methods augment large language models (LLMs) with external tools but are restricted to specialized domains, limited tool types, or require additional training data. In this paper, we introduce OctoTools, a training-free, user-friendly, and easily extensible multi-agent framework designed to tackle complex reasoning across diverse domains. OctoTools introduces standardized tool cards to encapsulate tool functionality, a planner for both high-level and low-level planning, and an executor to carry out tool usage. We validate OctoTools' generality across 16 diverse tasks (including MathVista, MMLU-Pro, MedQA, and GAIA-Text), achieving substantial average accuracy gains of 9.3% over GPT-4o. Furthermore, OctoTools also outperforms AutoGen, GPT-Functions, and LangChain by up to 10.6% when given the same set of tools. Through comprehensive analysi, ablations, and robustness tests with compact backbones and noisy tool environments, OctoTools demonstrates advantages in task planning, effective tool usage, and multi-step problem solving. Code, demos, and visualization are publicly available at this https URL.
>
---
#### [replaced 053] Towards EnergyGPT: A Large Language Model Specialized for the Energy Sector
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决通用大模型在能源领域效果不佳的问题。通过微调和LoRA方法，构建专用模型EnergyGPT，提升能源领域语言理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2509.07177](https://arxiv.org/pdf/2509.07177)**

> **作者:** Amal Chebbi; Babajide Kolade
>
> **备注:** Code and artifacts available at: this https URL
>
> **摘要:** Large language models have demonstrated impressive capabilities across various domains. However, their general-purpose nature often limits their effectiveness in specialized fields such as energy, where deep technical expertise and precise domain knowledge are essential. In this paper, we introduce EnergyGPT, a domain-specialized language model tailored for the energy sector, developed by fine-tuning the LLaMA 3.1-8B model on a high-quality, curated corpus of energy-related texts. We consider two adaptation strategies: a full-parameter Supervised Fine-Tuning variant and a parameter-efficient LoRA-based variant that updates only a small fraction of the model parameters. We present a complete development pipeline, including data collection and curation, model fine-tuning, benchmark design and LLM-judge choice, evaluation, and deployment. Through this work, we demonstrate that our training strategy enables improvements in domain relevance and performance without the need for large-scale infrastructure. By evaluating the performance of both EnergyGPT variants using domain-specific question-answering benchmarks, our results show that the adapted models consistently outperform the base model in most energy-related language understanding and generation tasks, with the LoRA variant achieving competitive gains at significantly reduced training cost.
>
---
#### [replaced 054] Instructions are all you need: Self-supervised Reinforcement Learning for Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于指令遵循任务，解决多约束指令难以执行的问题。提出一种无需外部监督的自监督强化学习框架，通过指令生成奖励信号，提升模型在多种数据集上的表现。**

- **链接: [https://arxiv.org/pdf/2510.14420](https://arxiv.org/pdf/2510.14420)**

> **作者:** Qingyu Ren; Qianyu He; Powei Chang; Jie Zeng; Zeye Sun; Fei Yu; Jiaqing Liang; Yanghua Xiao
>
> **摘要:** Language models often struggle to follow multi-constraint instructions that are crucial for real-world applications. Existing reinforcement learning (RL) approaches suffer from dependency on external supervision and sparse reward signals from multi-constraint tasks. We propose a label-free self-supervised RL framework that eliminates dependency on external supervision by deriving reward signals directly from instructions and generating pseudo-labels for reward model training. Our approach introduces constraint decomposition strategies and efficient constraint-wise binary classification to address sparse reward challenges while maintaining computational efficiency. Experiments show that our approach generalizes well, achieving strong improvements across 3 in-domain and 5 out-of-domain datasets, including challenging agentic and multi-turn instruction following. The data and code are publicly available at this https URL
>
---
#### [replaced 055] Using Learning Progressions to Guide AI Feedback for Science Learning
- **分类: cs.CL**

- **简介: 该论文属于教育技术任务，旨在解决AI反馈质量与可扩展性问题。通过比较基于学习进度的AI反馈与专家制定的反馈，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.03249](https://arxiv.org/pdf/2603.03249)**

> **作者:** Xin Xia; Nejla Yuruk; Yun Wang; Xiaoming Zhai
>
> **备注:** 15pages, 4 figures
>
> **摘要:** Generative artificial intelligence (AI) offers scalable support for formative feedback, yet most AI-generated feedback relies on task-specific rubrics authored by domain experts. While effective, rubric authoring is time-consuming and limits scalability across instructional contexts. Learning progressions (LP) provide a theoretically grounded representation of students' developing understanding and may offer an alternative solution. This study examines whether an LP-driven rubric generation pipeline can produce AI-generated feedback comparable in quality to feedback guided by expert-authored task rubrics. We analyzed AI-generated feedback for written scientific explanations produced by 207 middle school students in a chemistry task. Two pipelines were compared: (a) feedback guided by a human expert-designed, task-specific rubric, and (b) feedback guided by a task-specific rubric automatically derived from a learning progression prior to grading and feedback generation. Two human coders evaluated feedback quality using a multi-dimensional rubric assessing Clarity, Accuracy, Relevance, Engagement and Motivation, and Reflectiveness (10 sub-dimensions). Inter-rater reliability was high, with percent agreement ranging from 89% to 100% and Cohen's kappa values for estimable dimensions (kappa = .66 to .88). Paired t-tests revealed no statistically significant differences between the two pipelines for Clarity (t1 = 0.00, p1 = 1.000; t2 = 0.84, p2 = .399), Relevance (t1 = 0.28, p1 = .782; t2 = -0.58, p2 = .565), Engagement and Motivation (t1 = 0.50, p1 = .618; t2 = -0.58, p2 = .565), or Reflectiveness (t = -0.45, p = .656). These findings suggest that the LP-driven rubric pipeline can serve as an alternative solution.
>
---
#### [replaced 056] Perception-Aware Policy Optimization for Multimodal Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决视觉输入感知错误导致的性能问题。提出PAPO算法，通过引入感知损失提升模型的视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2507.06448](https://arxiv.org/pdf/2507.06448)**

> **作者:** Zhenhailong Wang; Xuehang Guo; Sofia Stoica; Haiyang Xu; Hongru Wang; Hyeonjeong Ha; Xiusi Chen; Yangyi Chen; Ming Yan; Fei Huang; Heng Ji
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be a highly effective strategy for endowing Large Language Models (LLMs) with robust multi-step reasoning abilities. However, its design and optimizations remain tailored to purely textual domains, resulting in suboptimal performance when applied to multimodal reasoning tasks. In particular, we observe that a major source of error in current multimodal reasoning lies in the perception of visual inputs. To address this bottleneck, we propose PAPO, a novel policy gradient algorithm that encourages the model to learn to perceive while learning to reason. Specifically, we introduce the Implicit Perception Loss in the form of a KL divergence term, which can be seamlessly plugged into mainstream RLVR algorithms such as GRPO and DAPO. Notably, PAPO does not rely on additional data curation, reward models, or stronger teacher models. To further enhance the training stability of PAPO, we introduce the Double Entropy Loss, which effectively regularizes the new KL objective without compromising performance. Despite its simplicity, PAPO yields significant overall improvements of 4.4%-17.5% on diverse multimodal benchmarks. The improvements are more pronounced, approaching 8.0%-19.1%, on tasks with high vision dependency. We also observe a substantial reduction of 30.5% in perception errors, indicating improved perceptual capabilities with PAPO. Overall, our work introduces a deeper integration of perception-aware supervision into core learning objectives and lays the groundwork for a new RL framework that encourages visually grounded reasoning. Code and data will be made publicly available for research purposes. Project page: this https URL.
>
---
#### [replaced 057] Nationality encoding in language model hidden states: Probing culturally differentiated representations in persona-conditioned academic text
- **分类: cs.CL**

- **简介: 该论文属于语言模型分析任务，探讨其在生成学术文本时是否编码国籍差异信息。通过实验和探针分析，发现模型在隐藏层中存在国籍特征表示，但表面文本无显著差异。**

- **链接: [https://arxiv.org/pdf/2604.10151](https://arxiv.org/pdf/2604.10151)**

> **作者:** Paul Jackson; Ruizhe Li; Elspeth Edelstein
>
> **备注:** 42 pages, 6 tables
>
> **摘要:** Large language models are increasingly used as writing tools and pedagogical resources in English for Academic Purposes, but it remains unclear whether they encode culturally differentiated representations when generating academic text. This study tests whether Gemma-3-4b-it encodes nationality-discriminative information in hidden states when generating research article introductions conditioned by British and Chinese academic personas. A corpus of 270 texts was generated from 45 prompt templates crossed with six persona conditions in a 2 x 3 design. Logistic regression probes were trained on hidden-state activations across all 35 layers, with shuffled-label baselines, a surface-text skyline classifier, cross-family tests, and sentence-level baselines used as controls. Probe-selected token positions were annotated for structural, lexical, and stance features using the Stanza NLP pipeline. The nationality probe reached 0.968 cross-validated accuracy at Layer 18, with perfect held-out classification. Nationality encoding followed a non-monotonic trajectory across layers, with structural effects strongest in the middle to upper network and lexical-domain effects peaking earlier. At high-signal token positions, British-associated patterns showed more postmodification, hedging, boosting, passive voice, and evaluative or process-oriented vocabulary, while Chinese-associated patterns showed more premodification, nominal predicates, and sociocultural or internationalisation vocabulary. However, sentence-level analysis found no significant nationality differences in the full generated surface text. The findings extend probing methodology to a sociolinguistic attribute and have practical implications for EAP and language pedagogy.
>
---
#### [replaced 058] Public Profile Matters: A Scalable Integrated Approach to Recommend Citations in the Wild
- **分类: cs.IR; cs.AI; cs.CL; cs.SI**

- **简介: 该论文属于引文推荐任务，旨在解决现有系统忽视人类引文行为及评估方式不真实的问题。提出Profiler模块和DAVINCI模型，提升推荐效果与效率。**

- **链接: [https://arxiv.org/pdf/2603.17361](https://arxiv.org/pdf/2603.17361)**

> **作者:** Karan Goyal; Dikshant Kukreja; Vikram Goyal; Mukesh Mohania
>
> **摘要:** Proper citation of relevant literature is essential for contextualising and validating scientific contributions. While current citation recommendation systems leverage local and global textual information, they often overlook the nuances of the human citation behaviour. Recent methods that incorporate such patterns improve performance but incur high computational costs and introduce systematic biases into downstream rerankers. To address this, we propose Profiler, a lightweight, non-learnable module that captures human citation patterns efficiently and without bias, significantly enhancing candidate retrieval. Furthermore, we identify a critical limitation in current evaluation protocol: the systems are assessed in a transductive setting, which fails to reflect real-world scenarios. We introduce a rigorous Inductive evaluation setting that enforces strict temporal constraints, simulating the recommendation of citations for newly authored papers in the wild. Finally, we present DAVINCI, a novel reranking model that integrates profiler-derived confidence priors with semantic information via an adaptive vector-gating mechanism. Our system achieves new state-of-the-art results across multiple benchmark datasets, demonstrating superior efficiency and generalisability.
>
---
#### [replaced 059] CLEAR: Cross-Lingual Enhancement in Alignment via Reverse-training
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于跨语言检索任务，旨在解决多语言嵌入模型在跨语言场景中的对齐问题。提出CLEAR方法，通过反向训练增强目标语言与英语的对齐，提升低资源语言性能，减少英语性能下降。**

- **链接: [https://arxiv.org/pdf/2604.05821](https://arxiv.org/pdf/2604.05821)**

> **作者:** Seungyoon Lee; Minhyuk Kim; Seongtae Hong; Youngjoon Jang; Dongsuk Oh; Heuiseok Lim
>
> **备注:** ACL2026 Main
>
> **摘要:** Existing multilingual embedding models often encounter challenges in cross-lingual scenarios due to imbalanced linguistic resources and less consideration of cross-lingual alignment during training. Although standardized contrastive learning approaches for cross-lingual adaptation are widely adopted, they may struggle to capture fundamental alignment between languages and degrade performance in well-aligned languages such as English. To address these challenges, we propose Cross-Lingual Enhancement in Retrieval via Reverse-training (CLEAR), a novel loss function utilizing a reverse training scheme to improve retrieval performance across diverse cross-lingual retrieval scenarios. CLEAR leverages an English passage as a bridge to strengthen alignments between the target language and English, ensuring robust performance in the cross-lingual retrieval task. Our extensive experiments demonstrate that CLEAR achieves notable improvements in cross-lingual scenarios, with gains up to 15%, particularly in low-resource languages, while minimizing performance degradation in English. Furthermore, our findings highlight that CLEAR offers promising effectiveness even in multilingual training, suggesting its potential for broad application and scalability. We release the code at this https URL.
>
---
#### [replaced 060] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型是否通过规则还是记忆处理德语定冠词。任务为语法一致性分析，解决模型是否依赖规则或记忆的问题。通过梯度方法分析参数更新，发现模型部分依赖记忆关联。**

- **链接: [https://arxiv.org/pdf/2601.09313](https://arxiv.org/pdf/2601.09313)**

> **作者:** Jonathan Drechsel; Erisa Bytyqi; Steffen Herbold
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Language models perform well on grammatical agreement, but it is unclear whether this reflects rule-based generalization or memorization. We study this question for German definite singular articles, whose forms depend on gender and case. Using GRADIEND, a gradient-based interpretability method, we learn parameter update directions for gender-case specific article transitions. We find that updates learned for a specific gender-case article transition frequently affect unrelated gender-case settings, with substantial overlap among the most affected neurons across settings. These results argue against a strictly rule-based encoding of German definite articles, indicating that models at least partly rely on memorized associations rather than abstract grammatical rules.
>
---
#### [replaced 061] ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ParetoBandit，解决非平稳LLM服务中的路由问题，通过自适应算法实现成本控制与质量优化。**

- **链接: [https://arxiv.org/pdf/2604.00136](https://arxiv.org/pdf/2604.00136)**

> **作者:** Annette Taberner-Miller
>
> **备注:** 27 pages, 15 figures, 13 tables. Code available at this https URL
>
> **摘要:** Multi-model LLM serving operates in a non-stationary, noisy environment: providers revise pricing, model quality can shift or regress without notice, and new models arrive regularly. More than a dozen recent methods have proposed learned routers to navigate the resulting quality--cost tradeoff across portfolios spanning a $\sim$530$\times$ cost range. Despite this activity, two gaps in the current solution space limit routing effectiveness under these conditions: no existing router enforces a dollar-denominated cost ceiling in closed loop over an open-ended request stream, and none provides principled online adaptation to post-deployment shifts in pricing or model quality. We present ParetoBandit, an open-source adaptive router built on cost-aware contextual bandits that addresses both gaps. Its core contributions are: (1) an online primal--dual budget pacer that enforces a per-request cost ceiling without a known horizon, and (2) geometric forgetting on sufficient statistics that gives the bandit bounded memory for tracking quality and cost shifts. A hot-swap model registry further supports runtime model changes with budget-controlled exploration. On 1,824 benchmark prompts with a three-model portfolio, the router maintains budget compliance within 0.4%, adapts to price and quality shifts with up to +0.071 quality lift, and integrates a cold-started model within $\sim$142 steps.
>
---
#### [replaced 062] League of LLMs: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出LOL框架，用于无需基准的大型语言模型互评，解决评估可靠性问题。通过多轮互评，提升评估的动态、透明、客观和专业性。**

- **链接: [https://arxiv.org/pdf/2507.22359](https://arxiv.org/pdf/2507.22359)**

> **作者:** Qianhong Guo; Wei Xie; Xiaofang Cai; Enze Wang; Shuoyoucheng Ma; Xiaobing Sun; Tian Xia; Kai Chen; Xiaofeng Wang; Baosheng Wang
>
> **摘要:** Although large language models (LLMs) have shown exceptional capabilities across a wide range of tasks, reliable evaluation remains a critical challenge due to data contamination, opaque operation, and subjective preferences. To address these issues, we propose League of LLMs (LOL), a novel benchmark-free evaluation paradigm that organizes multiple LLMs into a self-governed league for multi-round mutual evaluation. LOL integrates four core criteria (dynamic, transparent, objective, and professional) to mitigate key limitations of existing paradigms. Experiments on eight mainstream LLMs in mathematics and programming demonstrate that LOL can effectively distinguish LLM capabilities while maintaining high internal ranking stability (Top-$k$ consistency $= 70.7\%$). Beyond ranking, LOL reveals empirical findings that are difficult for traditional paradigms to capture. For instance, ``memorization-based answering'' behaviors are observed in some models, and higher in-family scores are found in the OpenAI model family ($\Delta = 9$, $p < 0.05$). Finally, we make our framework and code publicly available as a valuable complement to the current LLM evaluation ecosystem.
>
---
#### [replaced 063] Olmo 3
- **分类: cs.CL; cs.LG**

- **简介: 该论文介绍Olmo 3语言模型，解决长文本推理与多任务处理问题，包含7B和32B参数模型，支持编码、指令执行等。**

- **链接: [https://arxiv.org/pdf/2512.13961](https://arxiv.org/pdf/2512.13961)**

> **作者:** Team Olmo; Allyson Ettinger; Amanda Bertsch; Bailey Kuehl; David Graham; David Heineman; Dirk Groeneveld; Faeze Brahman; Finbarr Timbers; Hamish Ivison; Jacob Morrison; Jake Poznanski; Kyle Lo; Luca Soldaini; Matt Jordan; Mayee Chen; Michael Noukhovitch; Nathan Lambert; Pete Walsh; Pradeep Dasigi; Robert Berry; Saumya Malik; Saurabh Shah; Scott Geng; Shane Arora; Shashank Gupta; Taira Anderson; Teng Xiao; Tyler Murray; Tyler Romero; Victoria Graf; Akari Asai; Akshita Bhagia; Alexander Wettig; Alisa Liu; Aman Rangapur; Chloe Anastasiades; Costa Huang; Dustin Schwenk; Harsh Trivedi; Ian Magnusson; Jaron Lochner; Jiacheng Liu; Lester James V. Miranda; Maarten Sap; Malia Morgan; Michael Schmitz; Michal Guerquin; Michael Wilson; Regan Huff; Ronan Le Bras; Rui Xin; Rulin Shao; Sam Skjonsberg; Shannon Zejiang Shen; Shuyue Stella Li; Tucker Wilde; Valentina Pyatkin; Will Merrill; Yapei Chang; Yuling Gu; Zhiyuan Zeng; Ashish Sabharwal; Luke Zettlemoyer; Pang Wei Koh; Ali Farhadi; Noah A. Smith; Hannaneh Hajishirzi
>
> **备注:** minor edit updates
>
> **摘要:** We introduce Olmo 3, a family of state-of-the-art, fully-open language models at the 7B and 32B parameter scales. Olmo 3 model construction targets long-context reasoning, function calling, coding, instruction following, general chat, and knowledge recall. This release includes the entire model flow, i.e., the full lifecycle of the family of models, including every stage, checkpoint, data point, and dependency used to build it. Our flagship model, Olmo 3 Think 32B, is the strongest fully-open thinking model released to-date.
>
---
#### [replaced 064] Reasoning about Intent for Ambiguous Requests
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决模糊请求的意图推理问题。通过生成结构化响应，列举不同解释及对应答案，提升模型的透明度和准确性。**

- **链接: [https://arxiv.org/pdf/2511.10453](https://arxiv.org/pdf/2511.10453)**

> **作者:** Irina Saparina; Mirella Lapata
>
> **摘要:** Large language models often respond to ambiguous requests by implicitly committing to one interpretation, frustrating users and creating safety risks when that interpretation is wrong. We propose generating a single structured response that enumerates the different ways an ambiguous request can be interpreted, each coupled with a corresponding answer. Our models are trained with reinforcement learning using a dual reward objective: recall on ambiguous inputs to maximise coverage of valid interpretations, and precision on unambiguous ones to suppress spurious alternatives. Training requires only multiple valid answers per input as supervision, no clarification questions or explicit interpretations are needed. Experiments on conversational question answering and semantic parsing demonstrate that our method achieves higher coverage of valid answers than baseline approaches. Human evaluation confirms that predicted interpretations are meaningful and explain their corresponding answers. Our approach promotes transparency with explicit interpretations, achieves efficiency by requiring only one generation step, and supports downstream applications through its structured output format.
>
---
#### [replaced 065] Interactive ASR: Towards Human-Like Interaction and Semantic Coherence Evaluation for Agentic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统评估指标不足和缺乏交互纠错的问题。提出使用大模型进行语义评估，并设计交互式框架提升识别质量与互动能力。**

- **链接: [https://arxiv.org/pdf/2604.09121](https://arxiv.org/pdf/2604.09121)**

> **作者:** Peng Wang; Yanqiao Zhu; Zixuan Jiang; Qinyuan Chen; Xingjian Zhao; Xipeng Qiu; Wupeng Wang; Zhifu Gao; Xiangang Li; Kai Yu; Xie Chen
>
> **摘要:** Recent years have witnessed remarkable progress in automatic speech recognition (ASR), driven by advances in model architectures and large-scale training data. However, two important aspects remain underexplored. First, Word Error Rate (WER), the dominant evaluation metric for decades, treats all words equally and often fails to reflect the semantic correctness of an utterance at the sentence level. Second, interactive correction-an essential component of human communication-has rarely been systematically studied in ASR research. In this paper, we integrate these two perspectives under an agentic framework for interactive ASR. We propose leveraging LLM-as-a-Judge as a semantic-aware evaluation metric to assess recognition quality beyond token-level accuracy. Furthermore, we design an LLM-driven agent framework to simulate human-like multi-turn interaction, enabling iterative refinement of recognition outputs through semantic feedback. Extensive experiments are conducted on standard benchmarks, including GigaSpeech (English), WenetSpeech (Chinese), the ASRU 2019 code-switching test set. Both objective and subjective evaluations demonstrate the effectiveness of the proposed framework in improving semantic fidelity and interactive correction capability. We will release the code to facilitate future research in interactive and agentic ASR.
>
---
#### [replaced 066] Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的增量学习任务，解决模型在学习新任务时的灾难性遗忘问题。提出联合回溯适应方法，通过有限旧任务提示实现高效持续学习。**

- **链接: [https://arxiv.org/pdf/2505.15467](https://arxiv.org/pdf/2505.15467)**

> **作者:** Yukun Zhao; Lingyong Yan; Zhenyang Li; Shuaiqiang Wang; Zhumin Chen; Zhaochun Ren; Dawei Yin
>
> **备注:** The experimental setting is wrong, i.e., not a real continual learning setting
>
> **摘要:** Large language models have achieved remarkable success in various tasks. However, it is challenging for them to learn new tasks incrementally due to catastrophic forgetting. Existing approaches rely on experience replay, optimization constraints, or task differentiation, which encounter strict limitations in real-world scenarios. To address these issues, we propose Joint Flashback Adaptation. We first introduce flashbacks -- a limited number of prompts from old tasks -- when adapting to new tasks and constrain the deviations of the model outputs compared to the original one. We then interpolate latent tasks between flashbacks and new tasks to enable jointly learning relevant latent tasks, new tasks, and flashbacks, alleviating data sparsity in flashbacks and facilitating knowledge sharing for smooth adaptation. Our method requires only a limited number of flashbacks without access to the replay data and is task-agnostic. We conduct extensive experiments on state-of-the-art large language models across 1000+ instruction-following tasks, arithmetic reasoning tasks, and general reasoning tasks. The results demonstrate the superior performance of our method in improving generalization on new tasks and reducing forgetting in old tasks.
>
---
#### [replaced 067] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型高效推理任务，解决视觉令牌主导的效率瓶颈问题，分析了编码、预填充和解码阶段的优化技术。**

- **链接: [https://arxiv.org/pdf/2604.05546](https://arxiv.org/pdf/2604.05546)**

> **作者:** Jun Zhang; Yicheng Ji; Feiyang Ren; Yihang Li; Bowen Zeng; Zonghao Chen; Ke Chen; Lidan Shou; Gang Chen; Huan Li
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Large Vision-Language Models (LVLMs) enable sophisticated reasoning over images and videos, yet their inference is hindered by a systemic efficiency barrier known as visual token dominance. This overhead is driven by a multi-regime interplay between high-resolution feature extraction, quadratic attention scaling, and memory bandwidth constraints. We present a systematic taxonomy of efficiency techniques structured around the inference lifecycle, consisting of encoding, prefilling, and decoding. Unlike prior reviews focused on isolated optimizations, we analyze the end-to-end pipeline to reveal how upstream decisions dictate downstream bottlenecks, covering compute-bound visual encoding, the intensive prefilling of massive contexts, and the ''visual memory wall'' in bandwidth-bound decoding. By decoupling the efficiency landscape into the axes of shaping information density, managing long-context attention, and overcoming memory limits, this work provides a structured analysis of how isolated optimizations compose to navigate the trade-off between visual fidelity and system efficiency. The survey concludes by outlining four future frontiers supported by pilot empirical insights, including hybrid compression based on functional unit sensitivity, modality-aware decoding with relaxed verification, progressive state management for streaming continuity, and stage-disaggregated serving through hardware-algorithm co-design. Our literature repository is at this https URL.
>
---
#### [replaced 068] E2LLM: Encoder Elongated Large Language Models for Long-Context Understanding and Reasoning
- **分类: cs.CL**

- **简介: 该论文提出E2LLM，解决长文本理解与推理任务中的性能、效率和兼容性难题，通过分块压缩和适配器实现高效处理。**

- **链接: [https://arxiv.org/pdf/2409.06679](https://arxiv.org/pdf/2409.06679)**

> **作者:** Zihan Liao; Jun Wang; Hang Yu; Lingxiao Wei; Jianguo Li; Jun Wang; Wei Zhang
>
> **备注:** Accept by EMNLP'25
>
> **摘要:** Processing long contexts is increasingly important for Large Language Models (LLMs) in tasks like multi-turn dialogues, code generation, and document summarization. This paper addresses the challenges of achieving high long-context performance, low computational complexity, and compatibility with pretrained models -- collectively termed the ``impossible triangle''. We introduce E2LLM (Encoder Elongated Large Language Models), a novel approach that effectively navigates this paradox. E2LLM divides long contexts into chunks, compresses each into soft prompts using a pretrained text encoder, and aligns these representations with a decoder-only LLM via an adapter. To enhance the LLM's reasoning with these soft prompts, we employ two training objectives: encoder output reconstruction and long-context instruction fine-tuning. Extensive experiments reveal that E2LLM not only outperforms 8 state-of-the-art (SOTA) methods in effectiveness and efficiency for document summarization and question answering, but also achieves the best performance on LongBench v2 among models of comparable size.
>
---
#### [replaced 069] Fine-Tuning LLMs for Report Summarization: Analysis on Supervised and Unsupervised Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究在本地部署环境下，对大语言模型进行微调以提升报告摘要生成效果。解决数据不足和计算资源有限的问题，通过对比监督与非监督方法评估摘要质量。**

- **链接: [https://arxiv.org/pdf/2503.10676](https://arxiv.org/pdf/2503.10676)**

> **作者:** Swati Rallapalli; Shannon Gallagher; Andrew O. Mellinger; Jasmine Ratchford; Anusha Sinha; Tyler Brooks; William R. Nichols; Nick Winski; Bryan Brown
>
> **摘要:** We study the efficacy of fine-tuning Large Language Models (LLMs) for the specific task of report (government archives, news, intelligence reports) summarization. While this topic is being very actively researched - our specific application set-up faces two challenges: (i) ground-truth summaries maybe unavailable (e.g., for government archives), and (ii) availability of limited compute power - the sensitive nature of the application requires that computation is performed on-premise and for most of our experiments we use one or two A100 GPU cards. Under this set-up we conduct experiments to answer the following questions. First, given that fine-tuning the LLMs can be resource intensive, is it feasible to fine-tune them for improved report summarization capabilities on-premise? Second, what are the metrics we could leverage to assess the quality of these summaries? We conduct experiments on two different fine-tuning approaches in parallel and our findings reveal interesting trends regarding the utility of fine-tuning LLMs. Specifically, we find that in many cases, fine-tuning helps improve summary quality and in other cases it helps by reducing the number of invalid or garbage summaries.
>
---
#### [replaced 070] Gradient boundaries through confidence intervals for forced alignment estimates using model ensembles
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音信号处理任务，旨在解决强制对齐中边界估计不准确的问题。通过神经网络集成生成置信区间，得到更精确的边界估计和不确定性表示。**

- **链接: [https://arxiv.org/pdf/2506.01256](https://arxiv.org/pdf/2506.01256)**

> **作者:** Matthew C. Kelley
>
> **备注:** accepted for publication; 12 pages, 4 figures
>
> **摘要:** Forced alignment is a common tool to align audio with orthographic and phonetic transcriptions. Most forced alignment tools provide only point-estimates of boundaries. The present project introduces a method of producing gradient boundaries by deriving confidence intervals using neural network ensembles. Ten different segment classifier neural networks were previously trained, and the alignment process is repeated with each classifier. The ensemble is then used to place the point-estimate of a boundary at the median of the boundaries in the ensemble, and the gradient range is placed using a 97.85% confidence interval around the median constructed using order statistics. Gradient boundaries are taken here as a more realistic representation of how segments transition into each other. Moreover, the range indicates the model uncertainty in the boundary placement, facilitating tasks like finding boundaries that should be reviewed. As a bonus, on the Buckeye and TIMIT corpora, the ensemble boundaries show a slight overall improvement over using just a single model. The gradient boundaries can be emitted during alignment as JSON files and a main table for programmatic and statistical analysis. For familiarity, they are also output as Praat TextGrids using a point tier to represent the edges of the boundary regions.
>
---
#### [replaced 071] Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱增强的生成任务，解决多跳推理中信息漂移问题。通过多视角框架ParallaxRAG，提升检索与问答性能，减少幻觉。**

- **链接: [https://arxiv.org/pdf/2510.15552](https://arxiv.org/pdf/2510.15552)**

> **作者:** Jinliang Liu; Jiale Bai; Shaoning Zeng
>
> **摘要:** Large language models (LLMs) still struggle with multi-hop reasoning over knowledge-graphs (KGs), and we identify a previously overlooked structural reason for this difficulty: Transformer attention heads naturally specialize in distinct semantic relations across reasoning stages, forming a hop-aligned relay pattern. This key finding suggests that multi-hop reasoning is inherently multi-view, yet existing KG-based retrieval-augmented generation (KG-RAG) systems collapse all reasoning hops into a single representation, flat embedding space, suppressing this implicit structure and causing noisy or drifted path exploration. We introduce ParallaxRAG, a symmetric multi-view framework that decouples queries and KGs into aligned, head-specific semantic spaces. By enforcing relational diversity across multiple heads while constraining weakly related paths, ParallaxRAG constructs more accurate, cleaner subgraphs and guides LLMs through grounded, hop-wise reasoning. On WebQSP and CWQ, it achieves state-of-the-art retrieval and QA performance, substantially reduces hallucination, and generalizes strongly to the biomedical BioASQ benchmark.
>
---
#### [replaced 072] Judge Like Human Examiners: A Weighted Importance Multi-Point Evaluation Framework for Generative Tasks with Long-form Answers
- **分类: cs.CL**

- **简介: 该论文属于生成任务中的质量评估领域，旨在解决长文本回答评价中多因素融合与重要性差异的问题。提出WIMPE框架，通过加权评分点和对齐与冲突度量进行更精准的评估。**

- **链接: [https://arxiv.org/pdf/2604.11246](https://arxiv.org/pdf/2604.11246)**

> **作者:** Guoxin Yu; Chulun Zhou; Lemao Liu; Qi Wang; Mo Yu; Jialong Tang; Baosong Yang; Xiang Ao; Wai Lam; Yue Yu
>
> **备注:** 21 pages
>
> **摘要:** Evaluating the quality of model responses remains challenging in generative tasks with long-form answers, as the expected answers usually contain multiple semantically distinct yet complementary factors that should be factorized for fine-grained assessment. Recent evaluation methods resort to relying on either task-level rubrics or question-aware checklists. However, they still 1) struggle to assess whether a response is genuinely grounded in provided contexts; 2) fail to capture the heterogeneous importance of different aspects of reference answers. Inspired by human examiners, we propose a Weighted Importance Multi-Point Evaluation (WIMPE) framework, which factorizes each reference answer into weighted context-bound scoring points. Two complementary metrics, namely Weighted Point-wise Alignment (WPA) and Point-wise Conflict Penalty (PCP), are designed to measure the alignment and contradiction between model responses and reference answers. Extensive experiments on 10 generative tasks demonstrate that WIMPE achieves higher correlations with human annotations.
>
---
