# 自然语言处理 cs.CL

- **最新发布 77 篇**

- **更新 62 篇**

## 最新发布

#### [new 001] Field-Theoretic Memory for AI Agents: Continuous Dynamics for Context Preservation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于场论的AI记忆系统，将信息视为连续场而非离散条目，解决长上下文任务中的记忆保持问题。通过偏微分方程建模，提升多轮对话和多智能体协作性能。**

- **链接: [https://arxiv.org/pdf/2602.21220v1](https://arxiv.org/pdf/2602.21220v1)**

> **作者:** Subhadip Mitra
>
> **备注:** 15 pages, 6 figures. Code: https://github.com/rotalabs/rotalabs-fieldmem
>
> **摘要:** We present a memory system for AI agents that treats stored information as continuous fields governed by partial differential equations rather than discrete entries in a database. The approach draws from classical field theory: memories diffuse through semantic space, decay thermodynamically based on importance, and interact through field coupling in multi-agent scenarios. We evaluate the system on two established long-context benchmarks: LoCoMo (ACL 2024) with 300-turn conversations across 35 sessions, and LongMemEval (ICLR 2025) testing multi-session reasoning over 500+ turns. On LongMemEval, the field-theoretic approach achieves significant improvements: +116% F1 on multi-session reasoning (p<0.01, d= 3.06), +43.8% on temporal reasoning (p<0.001, d= 9.21), and +27.8% retrieval recall on knowledge updates (p<0.001, d= 5.00). Multi-agent experiments show near-perfect collective intelligence (>99.8%) through field coupling. Code is available at github.com/rotalabs/rotalabs-fieldmem.
>
---
#### [new 002] Reasoning-Based Personalized Generation for Users with Sparse Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化文本生成任务，解决用户数据稀疏导致的个性化不足问题。提出GraSPer框架，通过预测用户未来交互并生成文本，增强用户上下文以提升生成效果。**

- **链接: [https://arxiv.org/pdf/2602.21219v1](https://arxiv.org/pdf/2602.21219v1)**

> **作者:** Bo Ni; Branislav Kveton; Samyadeep Basu; Subhojyoti Mukherjee; Leyao Wang; Franck Dernoncourt; Sungchul Kim; Seunghyun Yoon; Zichao Wang; Ruiyi Zhang; Puneet Mathur; Jihyung Kil; Jiuxiang Gu; Nedim Lipka; Yu Wang; Ryan A. Rossi; Tyler Derr
>
> **摘要:** Large Language Model (LLM) personalization holds great promise for tailoring responses by leveraging personal context and history. However, real-world users usually possess sparse interaction histories with limited personal context, such as cold-start users in social platforms and newly registered customers in online E-commerce platforms, compromising the LLM-based personalized generation. To address this challenge, we introduce GraSPer (Graph-based Sparse Personalized Reasoning), a novel framework for enhancing personalized text generation under sparse context. GraSPer first augments user context by predicting items that the user would likely interact with in the future. With reasoning alignment, it then generates texts for these interactions to enrich the augmented context. In the end, it generates personalized outputs conditioned on both the real and synthetic histories, ensuring alignment with user style and preferences. Extensive experiments on three benchmark personalized generation datasets show that GraSPer achieves significant performance gain, substantially improving personalization in sparse user context settings.
>
---
#### [new 003] Improving Parametric Knowledge Access in Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型访问参数化知识的推理能力，旨在提升其知识回忆效果。通过引入推理训练，改善模型在多项问答任务中的表现。**

- **链接: [https://arxiv.org/pdf/2602.22193v1](https://arxiv.org/pdf/2602.22193v1)**

> **作者:** Melody Ma; John Hewitt
>
> **摘要:** We study reasoning for accessing world knowledge stored in a language model's parameters. For example, recalling that Canberra is Australia's capital may benefit from thinking through major cities and the concept of purpose-built capitals. While reasoning language models are trained via reinforcement learning to produce reasoning traces on tasks such as mathematics, they may not reason well for accessing their own world knowledge. We first find that models do not generate their best world knowledge reasoning by default: adding a simple "think step-by-step" cue demonstrates statistically significant improvement in knowledge recall but not math. Motivated by this, we propose training models to reason over their parametric knowledge using world-knowledge question answering as a verifiable reward. After reinforcement learning on TriviaQA (+9.9%), performance also improves on Natural Questions, HotpotQA, SimpleQA, and StrategyQA by 4.2%, 2.1%, 0.6%, and 3.0%, respectively. Reasoning models are under-optimized for parametric knowledge access, but can be easily trained to reason better.
>
---
#### [new 004] Task-Aware LoRA Adapter Composition via Similarity Retrieval in Vector Databases
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种基于向量数据库相似性检索的LoRA适配器动态组合方法，解决多任务下参数高效微调的问题。通过检索训练样本并融合适配器，实现零样本泛化。**

- **链接: [https://arxiv.org/pdf/2602.21222v1](https://arxiv.org/pdf/2602.21222v1)**

> **作者:** Riya Adsul; Balachandra Devarangadi Sunil; Isha Nalawade; Sudharshan Govindan
>
> **摘要:** Parameter efficient fine tuning methods like LoRA have enabled task specific adaptation of large language models, but efficiently composing multiple specialized adapters for unseen tasks remains challenging. We present a novel framework for dynamic LoRA adapter composition that leverages similarity retrieval in vector databases to enable zero-shot generalization across diverse NLP tasks. Our approach constructs a task-aware vector database by embedding training examples from 22 datasets spanning commonsense reasoning, question answering, natural language inference, and sentiment analysis. At inference time, we retrieve the most similar training examples, compute task similarity distributions via nucleus sampling, and dynamically merge relevant LoRA adapters using retrieval weighted fusion strategies. We evaluated four merging methods Linear, Concatenation, TIES, and Magnitude Prune demonstrating that our dataset centric retrieval approach often matches or exceeds the performance of individually fine-tuned task-specific adapters. Notably, Linear merging achieves 70.95% on PIQA and 77.62% on RTE, substantially outperforming single-task baselines (46% and 52%, respectively). Our framework requires no additional retriever training, operates with frozen embeddings, and enables efficient, interpretable adapter composition. These results suggest that retrieval based dynamic merging offers a promising direction for scalable, parameter-efficient multitask learning without requiring full model retraining for each new task.
>
---
#### [new 005] Small Wins Big: Comparing Large Language Models and Domain Fine-Tuned Models for Sarcasm Detection in Code-Mixed Hinglish Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的讽刺检测任务，旨在解决多语言和混码环境下讽刺识别困难的问题。通过比较大模型与微调的DistilBERT模型，发现微调模型在低资源场景下表现更优。**

- **链接: [https://arxiv.org/pdf/2602.21933v1](https://arxiv.org/pdf/2602.21933v1)**

> **作者:** Bitan Majumder; Anirban Sen
>
> **摘要:** Sarcasm detection in multilingual and code-mixed environments remains a challenging task for natural language processing models due to structural variations, informal expressions, and low-resource linguistic availability. This study compares four large language models, Llama 3.1, Mistral, Gemma 3, and Phi-4, with a fine-tuned DistilBERT model for sarcasm detection in code-mixed Hinglish text. The results indicate that the smaller, sequentially fine-tuned DistilBERT model achieved the highest overall accuracy of 84%, outperforming all of the LLMs in zero and few-shot set ups, using minimal LLM generated code-mixed data used for fine-tuning. These findings indicate that domain-adaptive fine-tuning of smaller transformer based models may significantly improve sarcasm detection over general LLM inference, in low-resource and data scarce settings.
>
---
#### [new 006] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion
- **分类: cs.CL**

- **简介: 该论文属于多语言多模态机器翻译任务，旨在解决数据稀缺问题。提出语音-文本融合框架，利用合成语音提升翻译质量，并通过自进化机制优化模型。**

- **链接: [https://arxiv.org/pdf/2602.21646v1](https://arxiv.org/pdf/2602.21646v1)**

> **作者:** Yexing Du; Youcheng Pan; Zekun Wang; Zheng Chu; Yichong Huang; Kaiyuan Liu; Bo Yang; Yang Xiang; Ming Liu; Bing Qin
>
> **备注:** Accepted in ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) have achieved notable success in enhancing translation performance by integrating multimodal information. However, existing research primarily focuses on image-guided methods, whose applicability is constrained by the scarcity of multilingual image-text pairs. The speech modality overcomes this limitation due to its natural alignment with text and the abundance of existing speech datasets, which enable scalable language coverage. In this paper, we propose a Speech-guided Machine Translation (SMT) framework that integrates speech and text as fused inputs into an MLLM to improve translation quality. To mitigate reliance on low-resource data, we introduce a Self-Evolution Mechanism. The core components of this framework include a text-to-speech model, responsible for generating synthetic speech, and an MLLM capable of classifying synthetic speech samples and iteratively optimizing itself using positive samples. Experimental results demonstrate that our framework surpasses all existing methods on the Multi30K multimodal machine translation benchmark, achieving new state-of-the-art results. Furthermore, on general machine translation datasets, particularly the FLORES-200, it achieves average state-of-the-art performance in 108 translation directions. Ablation studies on CoVoST-2 confirms that differences between synthetic and authentic speech have negligible impact on translation quality. The code and models are released at https://github.com/yxduir/LLM-SRT.
>
---
#### [new 007] RuCL: Stratified Rubric-Based Curriculum Learning for Multimodal Large Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型推理任务，解决奖励黑客问题。提出RuCL框架，通过分层评分体系优化训练过程，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2602.21628v1](https://arxiv.org/pdf/2602.21628v1)**

> **作者:** Yukun Chen; Jiaming Li; Longze Chen; Ze Gong; Jingpeng Li; Zhen Qin; Hengyu Chang; Ancheng Xu; Zhihao Yang; Hamid Alinejad-Rokny; Qiang Qu; Bo Zheng; Min Yang
>
> **备注:** 8 pages
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a prevailing paradigm for enhancing reasoning in Multimodal Large Language Models (MLLMs). However, relying solely on outcome supervision risks reward hacking, where models learn spurious reasoning patterns to satisfy final answer checks. While recent rubric-based approaches offer fine-grained supervision signals, they suffer from high computational costs of instance-level generation and inefficient training dynamics caused by treating all rubrics as equally learnable. In this paper, we propose Stratified Rubric-based Curriculum Learning (RuCL), a novel framework that reformulates curriculum learning by shifting the focus from data selection to reward design. RuCL generates generalized rubrics for broad applicability and stratifies them based on the model's competence. By dynamically adjusting rubric weights during training, RuCL guides the model from mastering foundational perception to tackling advanced logical reasoning. Extensive experiments on various visual reasoning benchmarks show that RuCL yields a remarkable +7.83% average improvement over the Qwen2.5-VL-7B model, achieving a state-of-the-art accuracy of 60.06%.
>
---
#### [new 008] Structured Prompt Language: Declarative Context Management for LLMs
- **分类: cs.CL; cs.DB; cs.PL**

- **简介: 该论文提出SPL，一种用于大语言模型的声明式上下文管理语言，解决提示工程复杂与资源优化问题，集成RAG、持久化记忆及自动优化功能。**

- **链接: [https://arxiv.org/pdf/2602.21257v1](https://arxiv.org/pdf/2602.21257v1)**

> **作者:** Wen G. Gong
>
> **备注:** 44 pages, 6 figures, 14 tables, 15 code-listings
>
> **摘要:** We present SPL (Structured Prompt Language), a declarative SQL-inspired language that treats large language models as generative knowledge bases and their context windows as constrained resources. SPL provides explicit WITH BUDGET/LIMIT token management, an automatic query optimizer, EXPLAIN transparency analogous to SQL's EXPLAIN ANALYZE, and native integration of retrieval-augmented generation (RAG) and persistent memory in a single declarative framework. SPL-flow extends SPL into resilient agentic pipelines with a three-tier provider fallback strategy (Ollama -> OpenRouter -> self-healing retry) fully transparent to the .spl script. Five extensions demonstrate the paradigm's breadth: (1) Text2SPL (multilingual NL->SPL translation); (2) Mixture-of-Models (MoM) routing that dispatches each PROMPT to a domain-specialist model at runtime; (3) Logical Chunking, an intelligent strategy for documents exceeding a single context window--expressed naturally through SPL's existing CTE syntax with no new constructs, decomposing a large query into a Map-Reduce pipeline that reduces attention cost from O(N^2) to O(N^2/k) and runs identically on cloud (parallel) or local hardware (sequential); (4) SPL-flow, a declarative agentic orchestration layer with resilient three-tier provider fallback; and (5) BENCHMARK for parallel multi-model comparison with automatic winner persistence. We provide a formal EBNF grammar, two pip-installable Python packages (spl-llm, spl-flow), and comparison against Prompty, DSPy, and LMQL. SPL reduces prompt boilerplate by 65% on average, surfaces a 68x cost spread across model tiers as a pre-execution signal, and runs the identical .spl script at $0.002 on OpenRouter or at zero marginal cost on a local Ollama instance--without modification.
>
---
#### [new 009] Applied Sociolinguistic AI for Community Development (ASA-CD): A New Scientific Paradigm for Linguistically-Grounded Social Intervention
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出ASA-CD，通过AI技术解决社区发展中的语言问题，聚焦于语言碎片化与负面情绪的关联，设计干预方案以促进社区积极变化。**

- **链接: [https://arxiv.org/pdf/2602.21217v1](https://arxiv.org/pdf/2602.21217v1)**

> **作者:** S M Ruhul Alam; Rifa Ferzana
>
> **备注:** 13 pages, 2 figures, 3 tables; simulation-based study introducing the ASA-CD framework
>
> **摘要:** This paper establishes Applied Sociolinguistic AI for Community Development (ASA-CD) as a novel scientific paradigm for addressing community challenges through linguistically grounded, AI-enabled intervention. ASA-CD introduces three key contributions: (1) linguistic biomarkers as computational indicators of discursive fragmentation; (2) development-aligned natural language processing (NLP), an AI optimisation paradigm prioritising collective outcomes; and (3) a standardised five-phase protocol for discursive intervention. A proof-of-concept study, incorporating real-world and synthetic corpora, demonstrates systematic associations between exclusionary language and negative sentiment and simulates intervention-based improvements. ASA-CD provides a unified methodological, ethical and empirical framework for scalable, value-aligned AI in the service of community empowerment.
>
---
#### [new 010] MEDSYN: Benchmarking Multi-EviDence SYNthesis in Complex Clinical Cases for Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医学多模态语言模型任务，旨在解决复杂临床案例中多证据合成的问题。通过构建基准MEDSYN，评估模型在诊断生成与选择中的表现，发现模型在整合多模态证据上存在不足。**

- **链接: [https://arxiv.org/pdf/2602.21950v1](https://arxiv.org/pdf/2602.21950v1)**

> **作者:** Boqi Chen; Xudong Liu; Jiachuan Peng; Marianne Frey-Marti; Bang Zheng; Kyle Lam; Lin Li; Jianing Qiu
>
> **摘要:** Multimodal large language models (MLLMs) have shown great potential in medical applications, yet existing benchmarks inadequately capture real-world clinical complexity. We introduce MEDSYN, a multilingual, multimodal benchmark of highly complex clinical cases with up to 7 distinct visual clinical evidence (CE) types per case. Mirroring clinical workflow, we evaluate 18 MLLMs on differential diagnosis (DDx) generation and final diagnosis (FDx) selection. While top models often match or even outperform human experts on DDx generation, all MLLMs exhibit a much larger DDx--FDx performance gap compared to expert clinicians, indicating a failure mode in synthesis of heterogeneous CE types. Ablations attribute this failure to (i) overreliance on less discriminative textual CE ($\it{e.g.}$, medical history) and (ii) a cross-modal CE utilization gap. We introduce Evidence Sensitivity to quantify the latter and show that a smaller gap correlates with higher diagnostic accuracy. Finally, we demonstrate how it can be used to guide interventions to improve model performance. We will open-source our benchmark and code.
>
---
#### [new 011] Understanding Artificial Theory of Mind: Perturbed Tasks and Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的理论心智能力，通过扰动任务评估其鲁棒性，并探索链式思维提示的改进效果。**

- **链接: [https://arxiv.org/pdf/2602.22072v1](https://arxiv.org/pdf/2602.22072v1)**

> **作者:** Christian Nickel; Laura Schrewe; Florian Mai; Lucie Flek
>
> **摘要:** Theory of Mind (ToM) refers to an agent's ability to model the internal states of others. Contributing to the debate whether large language models (LLMs) exhibit genuine ToM capabilities, our study investigates their ToM robustness using perturbations on false-belief tasks and examines the potential of Chain-of-Thought prompting (CoT) to enhance performance and explain the LLM's decision. We introduce a handcrafted, richly annotated ToM dataset, including classic and perturbed false belief tasks, the corresponding spaces of valid reasoning chains for correct task completion, subsequent reasoning faithfulness, task solutions, and propose metrics to evaluate reasoning chain correctness and to what extent final answers are faithful to reasoning traces of the generated CoT. We show a steep drop in ToM capabilities under task perturbation for all evaluated LLMs, questioning the notion of any robust form of ToM being present. While CoT prompting improves the ToM performance overall in a faithful manner, it surprisingly degrades accuracy for some perturbation classes, indicating that selective application is necessary.
>
---
#### [new 012] Architecture-Agnostic Curriculum Learning for Document Understanding: Empirical Evidence from Text-Only and Multimodal
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究文档理解任务，探讨渐进式数据调度是否提升模型训练效率。通过对比不同模型和数据集，验证了该策略在参数受限模型中的有效性。**

- **链接: [https://arxiv.org/pdf/2602.21225v1](https://arxiv.org/pdf/2602.21225v1)**

> **作者:** Mohammed Hamdan; Vincenzo Dentamaro; Giuseppe Pirlo; Mohamed Cheriet
>
> **摘要:** We investigate whether progressive data scheduling -- a curriculum learning strategy that incrementally increases training data exposure (33\%$\rightarrow$67\%$\rightarrow$100\%) -- yields consistent efficiency gains across architecturally distinct document understanding models. By evaluating BERT (text-only, 110M parameters) and LayoutLMv3 (multimodal, 126M parameters) on the FUNSD and CORD benchmarks, we establish that this schedule reduces wall-clock training time by approximately 33\%, commensurate with the reduction from 6.67 to 10.0 effective epoch-equivalents of data. To isolate curriculum effects from compute reduction, we introduce matched-compute baselines (Standard-7) that control for total gradient updates. On the FUNSD dataset, the curriculum significantly outperforms the matched-compute baseline for BERT ($Δ$F1 = +0.023, $p=0.022$, $d_z=3.83$), constituting evidence for a genuine scheduling benefit in capacity-constrained models. In contrast, no analogous benefit is observed for LayoutLMv3 ($p=0.621$), whose multimodal representations provide sufficient inductive bias. On the CORD dataset, all conditions converge to equivalent F1 scores ($\geq$0.947) irrespective of scheduling, indicating a performance ceiling. Schedule ablations comparing progressive, two-phase, reverse, and random pacing confirm that the efficiency gain derives from reduced data volume rather than ordering. Taken together, these findings demonstrate that progressive scheduling is a reliable compute-reduction strategy across model families, with curriculum-specific benefits contingent on the interaction between model capacity and task complexity.
>
---
#### [new 013] Recovered in Translation: Efficient Pipeline for Automated Translation of Benchmarks and Datasets
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决翻译基准质量不一致的问题。通过自动化框架提升翻译质量，确保任务结构和语言细节保留，提高模型评估准确性。**

- **链接: [https://arxiv.org/pdf/2602.22207v1](https://arxiv.org/pdf/2602.22207v1)**

> **作者:** Hanna Yukhymenko; Anton Alexandrov; Martin Vechev
>
> **摘要:** The reliability of multilingual Large Language Model (LLM) evaluation is currently compromised by the inconsistent quality of translated benchmarks. Existing resources often suffer from semantic drift and context loss, which can lead to misleading performance metrics. In this work, we present a fully automated framework designed to address these challenges by enabling scalable, high-quality translation of datasets and benchmarks. We demonstrate that adapting test-time compute scaling strategies, specifically Universal Self-Improvement (USI) and our proposed multi-round ranking method, T-RANK, allows for significantly higher quality outputs compared to traditional pipelines. Our framework ensures that benchmarks preserve their original task structure and linguistic nuances during localization. We apply this approach to translate popular benchmarks and datasets into eight Eastern and Southern European languages (Ukrainian, Bulgarian, Slovak, Romanian, Lithuanian, Estonian, Turkish, Greek). Evaluations using both reference-based metrics and LLM-as-a-judge show that our translations surpass existing resources, resulting in more accurate downstream model assessment. We release both the framework and the improved benchmarks to facilitate robust and reproducible multilingual AI development.
>
---
#### [new 014] ImpRIF: Stronger Implicit Reasoning Leads to Better Complex Instruction Following
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的指令跟随任务，旨在解决复杂指令理解问题。通过增强隐式推理能力，提升模型对复杂指令的遵循效果。**

- **链接: [https://arxiv.org/pdf/2602.21228v1](https://arxiv.org/pdf/2602.21228v1)**

> **作者:** Yuancheng Yang; Lin Yang; Xu Wang; Chao Tong; Haihua Yang
>
> **摘要:** As applications of large language models (LLMs) become increasingly complex, the demand for robust complex instruction following capabilities is growing accordingly. We argue that a thorough understanding of the instruction itself, especially the latent reasoning structure embedded between the lines, is crucial for improving instruction following. Therefore we target complex instructions that involve implicit reasoning, intricate logical relations, and multi-constraint dependencies. We propose ImpRIF, a method to enhance LLMs' understanding of implicit reasoning instructions, thereby improving its ability to follow complex instructions. We formalize such instructions as verifiable reasoning graphs, enabling programmatic verification and graph-driven chain-of-thought reasoning. Based on this formulation, we synthesize large-scale single- and multi-turn data, propose fine-tuning with graph reasoning, and apply reinforcement learning to explicitly train models to reason along the graph. On five complex instruction following benchmarks, our models substantially outperform their base models. These results demonstrate that enhancing implicit reasoning capabilities can significantly improve complex instruction following. This project will be open-sourced in the near future.
>
---
#### [new 015] DySCO: Dynamic Attention-Scaling Decoding for Long-Context LMs
- **分类: cs.CL**

- **简介: 该论文提出DySCO，解决长文本推理中注意力失效的问题。通过动态调整注意力权重，提升模型对关键信息的捕捉能力，适用于多种语言模型。**

- **链接: [https://arxiv.org/pdf/2602.22175v1](https://arxiv.org/pdf/2602.22175v1)**

> **作者:** Xi Ye; Wuwei Zhang; Fangcong Yin; Howard Yen; Danqi Chen
>
> **摘要:** Understanding and reasoning over long contexts is a crucial capability for language models (LMs). Although recent models support increasingly long context windows, their accuracy often deteriorates as input length grows. In practice, models often struggle to keep attention aligned with the most relevant context throughout decoding. In this work, we propose DySCO, a novel decoding algorithm for improving long-context reasoning. DySCO leverages retrieval heads--a subset of attention heads specialized for long-context retrieval--to identify task-relevant tokens at each decoding step and explicitly up-weight them. By doing so, DySCO dynamically adjusts attention during generation to better utilize relevant context. The method is training-free and can be applied directly to any off-the-shelf LMs. Across multiple instruction-tuned and reasoning models, DySCO consistently improves performance on challenging long-context reasoning benchmarks, yielding relative gains of up to 25% on MRCR and LongBenchV2 at 128K context length with modest additional compute. Further analysis highlights the importance of both dynamic attention rescaling and retrieval-head-guided selection for the effectiveness of the method, while providing interpretability insights into decoding-time attention behavior. Our code is available at https://github.com/princeton-pli/DySCO.
>
---
#### [new 016] Enhancing Multilingual Embeddings via Multi-Way Parallel Text Alignment
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于跨语言自然语言处理任务，旨在解决多语言嵌入对齐不足的问题。通过多语种平行文本训练，提升模型的多语言和跨语言表示能力。**

- **链接: [https://arxiv.org/pdf/2602.21543v1](https://arxiv.org/pdf/2602.21543v1)**

> **作者:** Barah Fazili; Koustava Goswami
>
> **摘要:** Multilingual pretraining typically lacks explicit alignment signals, leading to suboptimal cross-lingual alignment in the representation space. In this work, we show that training standard pretrained models for cross-lingual alignment with a multi-way parallel corpus in a diverse pool of languages can substantially improve multilingual and cross-lingual representations for NLU tasks. We construct a multi-way parallel dataset using translations of English text from an off-the-shelf NMT model for a pool of six target languages and achieve strong cross-lingual alignment through contrastive learning. This leads to substantial performance gains across both seen and unseen languages for multiple tasks from the MTEB benchmark evaluated for XLM-Roberta and multilingual BERT base models. Using a multi-way parallel corpus for contrastive training yields substantial gains on bitext mining (21.3%), semantic similarity (5.3%), and classification (28.4%) compared to English-centric (En-X) bilingually parallel data, where X is sampled from a pool of multiple target languages. Furthermore, finetuning mE5 model on a small dataset with multi-way parallelism significantly improves bitext mining compared to one without, underscoring the importance of multi-way cross-lingual supervision even for models already pretrained for high-quality sentence embeddings.
>
---
#### [new 017] A Diversity Diet for a Healthier Model: A Case Study of French ModernBERT
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究如何通过提升数据多样性来优化模型预训练。旨在减少数据量同时保持性能，通过比较不同采样算法实现这一目标。**

- **链接: [https://arxiv.org/pdf/2602.22014v1](https://arxiv.org/pdf/2602.22014v1)**

> **作者:** Louis Estève; Christophe Servan; Thomas Lavergne; Agata Savary
>
> **摘要:** Diversity has been gaining interest in the NLP community in recent years. At the same time, state-of-the-art transformer models such as ModernBERT use very large pre-training datasets, which are driven by size rather than by diversity. This summons for an investigation of the impact of diversity on the ModernBERT pre-training. We do so in this study, with the express intent of reducing pre-training dataset size, while retaining at least comparable performance. We compare diversity-driven sampling algorithms, so as to pick the best one. We find that diversity-driven sampling allows in some tasks to gain 10 points relative to randomly-sampled pre-training data of commensurate size. We also see that a model pre-trained for 483h on a diversity-driven dataset of 150M tokens can yield a commensurate performance to a model pre-trained for 1,775h on a randomly-driven dataset of 2.4B tokens.
>
---
#### [new 018] MixSarc: A Bangla-English Code-Mixed Corpus for Implicit Meaning Identification
- **分类: cs.CL**

- **简介: 该论文提出MixSarc，首个用于识别隐含意义的孟加拉-英语代码混合语料库，解决多语言社交媒体中讽刺、幽默等隐含意义识别问题。**

- **链接: [https://arxiv.org/pdf/2602.21608v1](https://arxiv.org/pdf/2602.21608v1)**

> **作者:** Kazi Samin Yasar Alam; Md Tanbir Chowdhury; Tamim Ahmed; Ajwad Abrar; Md Rafid Haque
>
> **备注:** Under Review
>
> **摘要:** Bangla-English code-mixing is widespread across South Asian social media, yet resources for implicit meaning identification in this setting remain scarce. Existing sentiment and sarcasm models largely focus on monolingual English or high-resource languages and struggle with transliteration variation, cultural references, and intra-sentential language switching. To address this gap, we introduce MixSarc, the first publicly available Bangla-English code-mixed corpus for implicit meaning identification. The dataset contains 9,087 manually annotated sentences labeled for humor, sarcasm, offensiveness, and vulgarity. We construct the corpus through targeted social media collection, systematic filtering, and multi-annotator validation. We benchmark transformer-based models and evaluate zero-shot large language models under structured prompting. Results show strong performance on humor detection but substantial degradation on sarcasm, offense, and vulgarity due to class imbalance and pragmatic complexity. Zero-shot models achieve competitive micro-F1 scores but low exact match accuracy. Further analysis reveals that over 42\% of negative sentiment instances in an external dataset exhibit sarcastic characteristics. MixSarc provides a foundational resource for culturally aware NLP and supports more reliable multi-label modeling in code-mixed environments.
>
---
#### [new 019] Mitigating Structural Noise in Low-Resource S2TT: An Optimized Cascaded Nepali-English Pipeline with Punctuation Restoration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于低资源语种语音到文本翻译任务，旨在解决ASR引入的结构噪声问题。通过优化级联系统和引入标点恢复模块，提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2602.21647v1](https://arxiv.org/pdf/2602.21647v1)**

> **作者:** Tangsang Chongbang; Pranesh Pyara Shrestha; Amrit Sarki; Anku Jaiswal
>
> **备注:** 13 pages, 4 figures, 12 tables
>
> **摘要:** This paper presents and evaluates an optimized cascaded Nepali speech-to-English text translation (S2TT) system, focusing on mitigating structural noise introduced by Automatic Speech Recognition (ASR). We first establish highly proficient ASR and NMT components: a Wav2Vec2-XLS-R-300m model achieved a state-of-the-art 2.72% CER on OpenSLR-54, and a multi-stage fine-tuned MarianMT model reached a 28.32 BLEU score on the FLORES-200 benchmark. We empirically investigate the influence of punctuation loss, demonstrating that unpunctuated ASR output significantly degrades translation quality, causing a massive 20.7% relative BLEU drop on the FLORES benchmark. To overcome this, we propose and evaluate an intermediate Punctuation Restoration Module (PRM). The final S2TT pipeline was tested across three configurations on a custom dataset. The optimal configuration, which applied the PRM directly to ASR output, achieved a 4.90 BLEU point gain over the direct ASR-to-NMT baseline (BLEU 36.38 vs. 31.48). This improvement was validated by human assessment, which confirmed the optimized pipeline's superior Adequacy (3.673) and Fluency (3.804). This work validates that targeted punctuation restoration is the most effective intervention for mitigating structural noise in the Nepali S2TT pipeline. It establishes an optimized baseline and demonstrates a critical architectural insight for developing cascaded speech translation systems for similar low-resource languages.
>
---
#### [new 020] DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决跨分词器对齐问题。提出DWA-KD框架，通过双空间加权和时间扭曲对齐提升模型压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.21669v1](https://arxiv.org/pdf/2602.21669v1)**

> **作者:** Duc Trung Vu; Pham Khanh Chi; Dat Phi Van; Linh Ngo Van; Sang Dinh; Trung Le
>
> **备注:** EACL Findings
>
> **摘要:** Knowledge Distillation (KD) has emerged as a crucial technique for compressing Large Language Models (LLMs). Although existing cross-tokenizer KD methods have made notable progress, their effectiveness remains constrained by suboptimal alignment across sequence and vocabulary levels. To address these limitations, we introduce Dual-Space Weighting and Time-Warped Alignment (DWA-KD), a novel cross-tokenizer distillation framework that enhances token-wise distillation through dual-space entropy-based weighting and achieves precise sequence-level alignment by leveraging both lexical and semantic information. At the token level, DWA-KD maps teacher representations into the student space and vice versa, performing dual-space KD via Kullback-Leibler divergence (KL). The process is modulated by dual-space weights that up-weight tokens where the student is uncertain and the teacher is confident, thereby focusing learning on informative tokens rather than treating all positions equally. At the sequence level, DWA-KD applies Soft Dynamic Time Warping (Soft-DTW) to both the embedding and final hidden-state layers, enabling robust alignment of lexical and contextual semantics between teacher and student sequences. Extensive experiments across diverse NLP benchmarks demonstrate that DWA-KD outperforms state-of-the-art KD baselines, while ablation studies confirm the complementary contributions of entropy-based token weighting and embedding and final hidden state layer Soft-DTW alignment.
>
---
#### [new 021] Large Language Models are Algorithmically Blind
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，探讨大语言模型在算法推理上的不足。研究发现模型在算法选择上表现不佳，存在“算法盲视”现象，揭示了其在程序化预测上的局限性。**

- **链接: [https://arxiv.org/pdf/2602.21947v1](https://arxiv.org/pdf/2602.21947v1)**

> **作者:** Sohan Venkatesh; Ashish Mahendran Kurapath; Tejas Melkote
>
> **备注:** 20 pages, 11 figures, 14 tables
>
> **摘要:** Large language models (LLMs) demonstrate remarkable breadth of knowledge, yet their ability to reason about computational processes remains poorly understood. Closing this gap matters for practitioners who rely on LLMs to guide algorithm selection and deployment. We address this limitation using causal discovery as a testbed and evaluate eight frontier LLMs against ground truth derived from large-scale algorithm executions and find systematic, near-total failure. Models produce ranges far wider than true confidence intervals yet still fail to contain the true algorithmic mean in the majority of instances; most perform worse than random guessing and the marginal above-random performance of the best model is most consistent with benchmark memorization rather than principled reasoning. We term this failure algorithmic blindness and argue it reflects a fundamental gap between declarative knowledge about algorithms and calibrated procedural prediction.
>
---
#### [new 022] DLT-Corpus: A Large-Scale Text Collection for the Distributed Ledger Technology Domain
- **分类: cs.CL**

- **简介: 该论文提出DLT-Corpus，解决DLT领域文本资源不足问题，涵盖科研、专利和社交媒体数据，用于分析技术发展与市场关系。**

- **链接: [https://arxiv.org/pdf/2602.22045v1](https://arxiv.org/pdf/2602.22045v1)**

> **作者:** Walter Hernandez Cruz; Peter Devine; Nikhil Vadgama; Paolo Tasca; Jiahua Xu
>
> **摘要:** We introduce DLT-Corpus, the largest domain-specific text collection for Distributed Ledger Technology (DLT) research to date: 2.98 billion tokens from 22.12 million documents spanning scientific literature (37,440 publications), United States Patent and Trademark Office (USPTO) patents (49,023 filings), and social media (22 million posts). Existing Natural Language Processing (NLP) resources for DLT focus narrowly on cryptocurrencies price prediction and smart contracts, leaving domain-specific language under explored despite the sector's ~$3 trillion market capitalization and rapid technological evolution. We demonstrate DLT-Corpus' utility by analyzing technology emergence patterns and market-innovation correlations. Findings reveal that technologies originate in scientific literature before reaching patents and social media, following traditional technology transfer patterns. While social media sentiment remains overwhelmingly bullish even during crypto winters, scientific and patent activity grow independently of market fluctuations, tracking overall market expansion in a virtuous cycle where research precedes and enables economic growth that funds further innovation. We publicly release the full DLT-Corpus; LedgerBERT, a domain-adapted model achieving 23% improvement over BERT-base on a DLT-specific Named Entity Recognition (NER) task; and all associated tools and code.
>
---
#### [new 023] Confidence-Driven Multi-Scale Model Selection for Cost-Efficient Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理成本高的问题。通过信心驱动的多尺度模型选择，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.22090v1](https://arxiv.org/pdf/2602.22090v1)**

> **作者:** Bo-Wei Chen; Chung-Chi Chen; An-Zi Yen
>
> **备注:** Accepted by EACL 2026 Findings
>
> **摘要:** Large Language Models (LLMs) have revolutionized inference across diverse natural language tasks, with larger models performing better but at higher computational costs. We propose a confidence-driven strategy that dynamically selects the most suitable model based on confidence estimates. By assessing a model's confidence in handling the task and response accuracy, tasks that are likely to be solved correctly are retained, while more uncertain or complex cases are delegated to a larger model, ensuring reliability while minimizing computation. Specifically, we evaluate a model's likelihood of knowing the correct answer and the probability that its response is accurate. Experiments on the Massive Multitask Language Understanding (MMLU) benchmark show that our approach achieves accuracy comparable to the largest model while reducing computational costs by 20\% to 40\%. When applied to GPT-4o API calls, it reduces token usage by approximately 60\%, further improving cost efficiency. These findings indicate the potential of confidence-based model selection to enhance real-world LLM deployment, particularly in resource-constrained settings such as edge devices and commercial API applications.
>
---
#### [new 024] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出ToolMATH，一个用于评估语言模型在多工具环境下数学推理能力的基准。解决模型在复杂工具环境中出现的错误累积与决策偏差问题，通过系统测试提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2602.21265v1](https://arxiv.org/pdf/2602.21265v1)**

> **作者:** Hyeonje Choi; Jeongsoo Lee; Hyojun Lee; Jay-Yoon Lee
>
> **备注:** Conference : Submitted to ICML 2026. 8 pages (+ abstract 16 pages), 5 figures
>
> **摘要:** We introduce \ToolMATH, a math-grounded benchmark that evaluates tool-augmented language models in realistic multi-tool environments where the output depends on calling schema-specified tools and sustaining multi-step execution. It turns math problems into a controlled, correctness-checkable benchmark with tool sets, enabling systematic evaluation of model reliability under (1) large, overlapping tool catalogs and (2) the absence of the intended capability. \ToolMATH provides actionable diagnostic evidence of failure modes in tool-augmented agents, helping identify the control mechanisms required for robustness. \ToolMATH roughly contains 8k questions and 12k tools; we provide an additional hard-set \ToolMATHHard with questions and tools. Our evaluation reveals that the key failure factor is due to the inability to reason, leading to the accumulation of intermediate results' errors and constrain later decisions. Tool-list redundancy do not simply add noise, but amplify small early deviations into irreversible execution drift. The benchmark highlights that when the intended capability is missing, distractor tools can sometimes serve as partial substitutes in solution paths, yet they can also mislead models into ungrounded tool trajectories. Finally, comparisons between tool-use protocols emphasize that improvements come less from local action selection and more from long-range plan coherence and disciplined use of observations.
>
---
#### [new 025] Explore-on-Graph: Incentivizing Autonomous Exploration of Large Language Models on Knowledge Graphs with Path-refined Reward Modeling
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决LLM在推理中出现的幻觉和事实缺失问题。通过引入强化学习和路径奖励机制，提升LLM在KG上的自主探索能力。**

- **链接: [https://arxiv.org/pdf/2602.21728v1](https://arxiv.org/pdf/2602.21728v1)**

> **作者:** Shiqi Yan; Yubo Chen; Ruiqi Zhou; Zhengxi Yao; Shuai Chen; Tianyi Zhang; Shijie Zhang; Wei Qiang Zhang; Yongfeng Huang; Haixin Duan; Yunqi Zhang
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** The reasoning process of Large Language Models (LLMs) is often plagued by hallucinations and missing facts in question-answering tasks. A promising solution is to ground LLMs' answers in verifiable knowledge sources, such as Knowledge Graphs (KGs). Prevailing KG-enhanced methods typically constrained LLM reasoning either by enforcing rules during generation or by imitating paths from a fixed set of demonstrations. However, they naturally confined the reasoning patterns of LLMs within the scope of prior experience or fine-tuning data, limiting their generalizability to out-of-distribution graph reasoning problems. To tackle this problem, in this paper, we propose Explore-on-Graph (EoG), a novel framework that encourages LLMs to autonomously explore a more diverse reasoning space on KGs. To incentivize exploration and discovery of novel reasoning paths, we propose to introduce reinforcement learning during training, whose reward is the correctness of the reasoning paths' final answers. To enhance the efficiency and meaningfulness of the exploration, we propose to incorporate path information as additional reward signals to refine the exploration process and reduce futile efforts. Extensive experiments on five KGQA benchmark datasets demonstrate that, to the best of our knowledge, our method achieves state-of-the-art performance, outperforming not only open-source but also even closed-source LLMs.
>
---
#### [new 026] FewMMBench: A Benchmark for Multimodal Few-Shot Learning
- **分类: cs.CL**

- **简介: 该论文提出FewMMBench，用于评估多模态大模型的少样本学习能力，解决模型在少量示例下的性能评估问题。通过多种任务和策略分析模型表现。**

- **链接: [https://arxiv.org/pdf/2602.21854v1](https://arxiv.org/pdf/2602.21854v1)**

> **作者:** Mustafa Dogan; Ilker Kesen; Iacer Calixto; Aykut Erdem; Erkut Erdem
>
> **备注:** Preprint. 49 pages, 38 Figures, 5 Tables
>
> **摘要:** As multimodal large language models (MLLMs) advance in handling interleaved image-text data, assessing their few-shot learning capabilities remains an open challenge. In this paper, we introduce FewMMBench, a comprehensive benchmark designed to evaluate MLLMs under few-shot conditions, with a focus on In-Context Learning (ICL) and Chain-of-Thought (CoT) prompting. Covering a diverse suite of multimodal understanding tasks, from attribute recognition to temporal reasoning, FewMMBench enables systematic analysis across task types, model families, and prompting strategies. We evaluate 26 open-weight MLLMs from six model families across zero-shot, few-shot, and CoT-augmented few-shot settings. Our findings reveal that instruction-tuned models exhibit strong zero-shot performance but benefit minimally, or even regress, with additional demonstrations or CoT reasoning. Retrieval-based demonstrations and increased context size also yield limited gains. These results highlight FewMMBench as a rigorous testbed for diagnosing and advancing few-shot capabilities in multimodal LLMs. The data is available at: https://huggingface.co/datasets/mustafaa/FewMMBench
>
---
#### [new 027] Alignment-Weighted DPO: A principled reasoning approach to improve safety alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全对齐任务，旨在解决LLM对隐蔽有害指令的脆弱性问题。通过引入基于推理的微调和权重调整方法，提升模型的安全性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.21346v1](https://arxiv.org/pdf/2602.21346v1)**

> **作者:** Mengxuan Hu; Vivek V. Datla; Anoop Kumar; Zihan Guan; Sheng Li; Alfy Samuel; Daben Liu
>
> **摘要:** Recent advances in alignment techniques such as Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF), and Direct Preference Optimization (DPO) have improved the safety of large language models (LLMs). However, these LLMs remain vulnerable to jailbreak attacks that disguise harmful intent through indirect or deceptive phrasing. Using causal intervention, we empirically demonstrate that this vulnerability stems from shallow alignment mechanisms that lack deep reasoning, often rejecting harmful prompts without truly understanding why they are harmful. To mitigate this vulnerability, we propose enhancing alignment through reasoning-aware post-training. We construct and release a novel Chain-of-Thought (CoT) fine-tuning dataset that includes both utility-oriented and safety-critical prompts with step-by-step rationales. Fine-tuning on this dataset encourages models to produce principled refusals grounded in reasoning, outperforming standard SFT baselines. Furthermore, inspired by failure patterns in CoT fine-tuning, we introduce Alignment-Weighted DPO, which targets the most problematic parts of an output by assigning different preference weights to the reasoning and final-answer segments. This produces finer-grained, targeted updates than vanilla DPO and improves robustness to diverse jailbreak strategies. Extensive experiments across multiple safety and utility benchmarks show that our method consistently improves alignment robustness while maintaining overall model utility.
>
---
#### [new 028] Personalized Graph-Empowered Large Language Model for Proactive Information Access
- **分类: cs.CL**

- **简介: 该论文属于个性化信息检索任务，旨在解决用户遗忘事件的回忆问题。通过结合大语言模型与知识图谱，提升主动信息获取的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.21862v1](https://arxiv.org/pdf/2602.21862v1)**

> **作者:** Chia Cheng Chang; An-Zi Yen; Hen-Hsen Huang; Hsin-Hsi Chen
>
> **摘要:** Since individuals may struggle to recall all life details and often confuse events, establishing a system to assist users in recalling forgotten experiences is essential. While numerous studies have proposed memory recall systems, these primarily rely on deep learning techniques that require extensive training and often face data scarcity due to the limited availability of personal lifelogs. As lifelogs grow over time, systems must also adapt quickly to newly accumulated data. Recently, large language models (LLMs) have demonstrated remarkable capabilities across various tasks, making them promising for personalized applications. In this work, we present a framework that leverages LLMs for proactive information access, integrating personal knowledge graphs to enhance the detection of access needs through a refined decision-making process. Our framework offers high flexibility, enabling the replacement of base models and the modification of fact retrieval methods for continuous improvement. Experimental results demonstrate that our approach effectively identifies forgotten events, supporting users in recalling past experiences more efficiently.
>
---
#### [new 029] MERRY: Semantically Decoupled Evaluation of Multimodal Emotional and Role Consistencies of Role-Playing Agents
- **分类: cs.CL**

- **简介: 该论文提出MERRY框架，用于评估角色扮演代理的多模态情感和角色一致性。解决现有评估方法依赖文本和人工判断的问题，通过语义解耦提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2602.21941v1](https://arxiv.org/pdf/2602.21941v1)**

> **作者:** Zhenyu Wang; Xiaofen Xing; Yirong Chen; Xiangmin Xu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Multimodal Role-Playing Agents (MRPAs) are attracting increasing attention due to their ability to deliver more immersive multimodal emotional interactions. However, existing studies still rely on pure textual benchmarks to evaluate the text responses of MRPAs, while delegating the assessment of their multimodal expressions solely to modality-synthesis metrics. This evaluation paradigm, on the one hand, entangles semantic assessment with modality generation, leading to ambiguous error attribution, and on the other hand remains constrained by the heavy reliance on human judgment. To this end, we propose MERRY, a semantically decoupled evaluation framework for assessing Multimodal Emotional and Role consistencies of Role-playing agents. This framework introduce five refined metrics for EC and three for RC. Notably, we transform the traditional subjective scoring approach into a novel bidirectional-evidence-finding task, significantly improving the human agreement of LLM-as-Judge evaluations. Based on MERRY, we conduct extensive evaluations. Our empirical results primarily reveal that: (1) Training on synthetic datasets tends to reduce emotional consistency, whereas training on real-world datasets improves it; (2) Existing models suffer from emotional templatization and simplification, exhibiting positive-bias and performance bottleneck in fine-grained negative emotions; (3) Simple prompting method strengthens the weak models but constrains the strong ones, while simple fine-tuning method suffers from poor role generalization. Codes and dataset are available.
>
---
#### [new 030] IslamicLegalBench: Evaluating LLMs Knowledge and Reasoning of Islamic Law Across 1,200 Years of Islamic Pluralist Legal Traditions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI评估任务，旨在检验大语言模型对伊斯兰法的理解与推理能力。通过构建基准测试，发现模型存在知识缺失和错误推理问题。**

- **链接: [https://arxiv.org/pdf/2602.21226v1](https://arxiv.org/pdf/2602.21226v1)**

> **作者:** Ezieddin Elmahjub; Junaid Qadir; Abdullah Mushtaq; Rafay Naeem; Ibrahim Ghaznavi; Waleed Iqbal
>
> **备注:** This manuscript has been submitted for review to Artificial Intelligence \& Law
>
> **摘要:** As millions of Muslims turn to LLMs like GPT, Claude, and DeepSeek for religious guidance, a critical question arises: Can these AI systems reliably reason about Islamic law? We introduce IslamicLegalBench, the first benchmark evaluating LLMs across seven schools of Islamic jurisprudence, with 718 instances covering 13 tasks of varying complexity. Evaluation of nine state-of-the-art models reveals major limitations: the best model achieves only 68% correctness with 21% hallucination, while several models fall below 35% correctness and exceed 55% hallucination. Few-shot prompting provides minimal gains, improving only 2 of 9 models by >1%. Moderate-complexity tasks requiring exact knowledge show the highest errors, whereas high-complexity tasks display apparent competence through semantic reasoning. False premise detection indicates risky sycophancy, with 6 of 9 models accepting misleading assumptions at rates above 40%. These results highlight that prompt-based methods cannot compensate for missing foundational knowledge. IslamicLegalBench offers the first systematic framework to evaluate Islamic legal reasoning in AI, revealing critical gaps in tools increasingly relied on for spiritual guidance.
>
---
#### [new 031] Improving Implicit Discourse Relation Recognition with Natural Language Explanations from LLMs
- **分类: cs.CL**

- **简介: 该论文属于隐含话语关系识别任务，旨在提升模型性能与可解释性。通过引入大语言模型生成的解释，提出一种联合预测与生成的框架，增强模型理解与解释能力。**

- **链接: [https://arxiv.org/pdf/2602.21763v1](https://arxiv.org/pdf/2602.21763v1)**

> **作者:** Heng Wang; Changxing Wu
>
> **备注:** AAAI26'0ral
>
> **摘要:** Implicit Discourse Relation Recognition (IDRR) remains a challenging task due to the requirement for deep semantic understanding in the absence of explicit discourse markers. A further limitation is that existing methods only predict relations without providing any supporting explanations. Recent advances in large language models (LLMs) have shown strong reasoning capabilities in both deep language understanding and natural language explanation generation. In this work, we propose a simple yet effective approach to distill the reasoning capabilities of LLMs into lightweight IDRR models to improve both performance and interpretability. Specifically, we first prompt an LLM to generate explanations for each training instance conditioned on its gold label. Then, we introduce a novel classification-generation framework that jointly performs relation prediction and explanation generation, and train it with the additional supervision of LLM-generated explanations. Our framework is plug-and-play, enabling easy integration with most existing IDRR models. Experimental results on PDTB demonstrate that our approach significantly improves IDRR performance, while human evaluation further confirms that the generated explanations enhance model interpretability. Furthermore, we validate the generality of our approach on sentiment classification and natural language inference
>
---
#### [new 032] EPSVec: Efficient and Private Synthetic Data Generation via Dataset Vectors
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文提出EPSVec，用于高效私有合成数据生成。解决传统方法效率低、计算成本高的问题，通过数据集向量引导大模型生成，提升合成数据质量与隐私保护。**

- **链接: [https://arxiv.org/pdf/2602.21218v1](https://arxiv.org/pdf/2602.21218v1)**

> **作者:** Amin Banayeeanzade; Qingchuan Yang; Deqing Fu; Spencer Hong; Erin Babinsky; Alfy Samuel; Anoop Kumar; Robin Jia; Sai Praneeth Karimireddy
>
> **摘要:** High-quality data is essential for modern machine learning, yet many valuable corpora are sensitive and cannot be freely shared. Synthetic data offers a practical substitute for downstream development, and large language models (LLMs) have emerged as powerful engines for generating it. However, existing private text generation methods are severely inefficient: they are data-intensive, computationally slow, and often require large private corpora or batch sizes to achieve usable quality. We introduce EPSVec, a differentially-private lightweight alternative that steers LLM generation using *dataset vectors*--directions in activation space that capture the distributional gap between private data and public priors. EPSVec extracts and sanitizes steering vectors just once and then performs standard decoding. This decouples the privacy budget from generation, enabling arbitrarily many synthetic samples without additional privacy cost and yielding strong fidelity even in low-data regimes. Furthermore, we enhance our method by utilizing pretrained (base) models and introducing fixed-shot prompting to boost generation diversity and fidelity. Our experiments demonstrate that EPSVec outperforms existing baselines in distributional alignment and downstream utility, particularly in low-data regimes, while significantly reducing computational overhead.
>
---
#### [new 033] Measuring Pragmatic Influence in Large Language Model Instructions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型指令中的语用影响，解决如何测量指令框架对模型行为的影响问题。通过分解指令、分类框架策略和优先级评估，建立可量化的分析框架。**

- **链接: [https://arxiv.org/pdf/2602.21223v1](https://arxiv.org/pdf/2602.21223v1)**

> **作者:** Yilin Geng; Omri Abend; Eduard Hovy; Lea Frermann
>
> **摘要:** It is not only what we ask large language models (LLMs) to do that matters, but also how we prompt. Phrases like "This is urgent" or "As your supervisor" can shift model behavior without altering task content. We study this effect as pragmatic framing, contextual cues that shape directive interpretation rather than task specification. While prior work exploits such cues for prompt optimization or probes them as security vulnerabilities, pragmatic framing itself has not been treated as a measurable property of instruction following. Measuring this influence systematically remains challenging, requiring controlled isolation of framing cues. We introduce a framework with three novel components: directive-framing decomposition separating framing context from task specification; a taxonomy organizing 400 instantiations of framing into 13 strategies across 4 mechanism clusters; and priority-based measurement that quantifies influence through observable shifts in directive prioritization. Across five LLMs of different families and sizes, influence mechanisms cause consistent and structured shifts in directive prioritization, moving models from baseline impartiality toward favoring the framed directive. This work establishes pragmatic framing as a measurable and predictable factor in instruction-following systems.
>
---
#### [new 034] TRACE: Trajectory-Aware Comprehensive Evaluation for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于深度研究代理评估任务，旨在解决传统评估方法无法全面反映代理推理过程的问题。提出TRACE框架，通过轨迹分析和能力评估，更准确地衡量代理的效率、鲁棒性和潜在能力。**

- **链接: [https://arxiv.org/pdf/2602.21230v1](https://arxiv.org/pdf/2602.21230v1)**

> **作者:** Yanyu Chen; Jiyue Jiang; Jiahong Liu; Yifei Zhang; Xiao Guo; Irwin King
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** The evaluation of Deep Research Agents is a critical challenge, as conventional outcome-based metrics fail to capture the nuances of their complex reasoning. Current evaluation faces two primary challenges: 1) a reliance on singular metrics like Pass@1, creating a "high-score illusion" that ignores the quality, efficiency, and soundness of the reasoning process; and 2) the failure of static benchmarks to quantify crucial attributes like robustness and latent capability. To address these gaps, we introduce TRACE (Trajectory-Aware Comprehensive Evaluation), a framework that holistically assesses the entire problem-solving trajectory. To counter the "high-score illusion", we propose a Hierarchical Trajectory Utility Function that quantifies process efficiency and cognitive quality, including evidence grounding, alongside accuracy. To measure deeper attributes, TRACE introduces a Scaffolded Capability Assessment protocol, quantifying an agent's latent ability by determining the minimum guidance needed for success. Our contributions include the TRACE framework, its novel metrics, and the accompanying DeepResearch-Bench with controllable complexity. Experiments show TRACE delivers a granular ranking that uncovers critical trade-offs between agent accuracy, efficiency, and robustness entirely missed by singular metrics.
>
---
#### [new 035] Budget-Aware Agentic Routing via Boundary-Guided Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能体路由任务，解决长期任务中模型调用成本过高的问题。通过预算感知的动态决策方法，在保证效果的同时降低开支。**

- **链接: [https://arxiv.org/pdf/2602.21227v1](https://arxiv.org/pdf/2602.21227v1)**

> **作者:** Caiqi Zhang; Menglin Xia; Xuchao Zhang; Daniel Madrigal; Ankur Mallick; Samuel Kessler; Victor Ruehle; Saravan Rajmohan
>
> **摘要:** As large language models (LLMs) evolve into autonomous agents that execute long-horizon workflows, invoking a high-capability model at every step becomes economically unsustainable. While model routing is effective for single-turn queries, agentic routing is a sequential, path-dependent problem: early mistakes compound, feedback is often at the end of the episode, and deployments often demand strict per-task spending limits. We propose Budget-Aware Agentic Routing, which selects between a cheap and an expensive model at each step to optimize the cost--success frontier and to operate under strict per-task budgets. We propose Boundary-Guided Training, which leverages two boundary policies (always-small vs.\ always-large) to build a difficulty taxonomy and to anchor learning under sparse rewards. Our approach warms start with boundary-guided SFT data synthesis via stratified sampling of cost-efficient trajectories, then applies Boundary-Guided Policy Optimization (BoPO), combining boundary-relative rewards with a reference-guided advantage to avoid degenerate cheap-failure solutions. Experiment results show that our method improves the efficiency frontier, matching strong routing baselines at substantially lower cost while demonstrating generalization to strict inference-time budget constraints. Overall, our work establishes a foundational framework for agentic routing, shifting the paradigm from static model selection to dynamic, budget-aware sequential decision-making.
>
---
#### [new 036] SumTablets: A Transliteration Dataset of Sumerian Tablets
- **分类: cs.CL**

- **简介: 该论文提出SumTablets数据集，解决Sumerian transliteration中缺乏配对数据的问题，通过整合Unicode字符与译文，支持NLP应用。**

- **链接: [https://arxiv.org/pdf/2602.22200v1](https://arxiv.org/pdf/2602.22200v1)**

> **作者:** Cole Simmons; Richard Diehl Martinez; Dan Jurafsky
>
> **备注:** 11 pages with 3 figures
>
> **摘要:** Sumerian transliteration is a conventional system for representing a scholar's interpretation of a tablet in the Latin script. Thanks to visionary digital Assyriology projects such as ETCSL, CDLI, and Oracc, a large number of Sumerian transliterations have been published online, and these data are well-structured for a variety of search and analysis tasks. However, the absence of a comprehensive, accessible dataset pairing transliterations with a digital representation of the tablet's cuneiform glyphs has prevented the application of modern Natural Language Processing (NLP) methods to the task of Sumerian transliteration. To address this gap, we present SumTablets, a dataset pairing Unicode representations of 91,606 Sumerian cuneiform tablets (totaling 6,970,407 glyphs) with the associated transliterations published by Oracc. We construct SumTablets by first preprocessing and standardizing the Oracc transliterations before mapping each reading back to the Unicode representation of the source glyph. Further, we retain parallel structural information (e.g., surfaces, newlines, broken segments) through the use of special tokens. We release SumTablets as a Hugging Face Dataset (CC BY 4.0) and open source data preparation code via GitHub. Additionally, we leverage SumTablets to implement and evaluate two transliteration baselines: (1) weighted sampling from a glyph's possible readings, and (2) fine-tuning an autoregressive language model. Our fine-tuned language model achieves an average transliteration character-level F-score (chrF) of 97.55, demonstrating the immediate potential of transformer-based transliteration models in allowing experts to rapidly verify generated transliterations rather than manually transliterating tablets one-by-one.
>
---
#### [new 037] ExpLang: Improved Exploration and Exploitation in LLM Reasoning with On-Policy Thinking Language Selection
- **分类: cs.CL**

- **简介: 该论文属于大模型推理任务，旨在解决单一语言训练限制问题。通过引入多语言思维选择机制，提升强化学习中的探索与利用效果。**

- **链接: [https://arxiv.org/pdf/2602.21887v1](https://arxiv.org/pdf/2602.21887v1)**

> **作者:** Changjiang Gao; Zixian Huang; Kaichen Yang; Jiajun Chen; Jixing Li; Shujian Huang
>
> **摘要:** Current large reasoning models (LRMs) have shown strong ability on challenging tasks after reinforcement learning (RL) based post-training. However, previous work mainly focuses on English reasoning in expectation of the strongest performance, despite the demonstrated potential advantage of multilingual thinking, as well as the requirement for native thinking traces by global users. In this paper, we propose ExpLang, a novel LLM post-training pipeline that enables on-policy thinking language selection to improve exploration and exploitation during RL with the use of multiple languages. The results show that our method steadily outperforms English-only training with the same training budget, while showing high thinking language compliance for both seen and unseen languages. Analysis shows that, by enabling on-policy thinking language selection as an action during RL, ExpLang effectively extends the RL exploration space with diversified language preference and improves the RL exploitation outcome with leveraged non-English advantage. The method is orthogonal to most RL algorithms and opens up a new perspective on using multilinguality to improve LRMs.
>
---
#### [new 038] CxMP: A Linguistic Minimal-Pair Benchmark for Evaluating Constructional Understanding in Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在评估语言模型对构式意义的掌握。提出CxMP基准，通过最小对设计测试模型对九种构式的理解能力，揭示模型在形式与意义整合上的不足。**

- **链接: [https://arxiv.org/pdf/2602.21978v1](https://arxiv.org/pdf/2602.21978v1)**

> **作者:** Miyu Oba; Saku Sugawara
>
> **摘要:** Recent work has examined language models from a linguistic perspective to better understand how they acquire language. Most existing benchmarks focus on judging grammatical acceptability, whereas the ability to interpret meanings conveyed by grammatical forms has received much less attention. We introduce the Linguistic Minimal-Pair Benchmark for Evaluating Constructional Understanding in Language Models (CxMP), a benchmark grounded in Construction Grammar that treats form-meaning pairings, or constructions, as fundamental linguistic units. CxMP evaluates whether models can interpret the semantic relations implied by constructions, using a controlled minimal-pair design across nine construction types, including the let-alone, caused motion, and ditransitive constructions. Our results show that while syntactic competence emerges early, constructional understanding develops more gradually and remains limited even in large language models (LLMs). CxMP thus reveals persistent gaps in how language models integrate form and meaning, providing a framework for studying constructional understanding and learning trajectories in language models.
>
---
#### [new 039] Robust Long-Form Bangla Speech Processing: Automatic Speech Recognition and Speaker Diarization
- **分类: cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于 Bengali 语音处理任务，解决长文本语音识别和说话人分割问题。通过模型微调、语音分离和分段优化，提升了识别与分割效果。**

- **链接: [https://arxiv.org/pdf/2602.21741v1](https://arxiv.org/pdf/2602.21741v1)**

> **作者:** MD. Sagor Chowdhury; Adiba Fairooz Chowdhury
>
> **备注:** 6 pages, 5 figures, 3 tables; system paper submitted to DL Sprint 4.0 (Kaggle)
>
> **摘要:** We describe our end-to-end system for Bengali long-form speech recognition (ASR) and speaker diarization submitted to the DL Sprint 4.0 competition on Kaggle. Bengali presents substantial challenges for both tasks: a large phoneme inventory, significant dialectal variation, frequent code-mixing with English, and a relative scarcity of large-scale labelled corpora. For ASR we achieve a best private Word Error Rate (WER) of 0.37738 and public WER of 0.36137, combining a BengaliAI fine-tuned Whisper medium model with Demucs source separation for vocal isolation, silence-boundary chunking, and carefully tuned generation hyperparameters. For speaker diarization we reach a best private Diarization Error Rate (DER) of 0.27671 and public DER of 0.20936 by replacing the default segmentation model inside the pyannote.audio pipeline with a Bengali-fine-tuned variant, pairing it with wespeaker-voxceleb-resnet34-LM embeddings and centroid-based agglomerative clustering. Our experiments demonstrate that domain-specific fine-tuning of the segmentation component, vocal source separation, and natural silence-aware chunking are the three most impactful design choices for low-resource Bengali speech processing.
>
---
#### [new 040] Evaluating the Usage of African-American Vernacular English in Large Language Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自然语言理解任务，研究大语言模型对非洲裔美国黑人英语（AAVE）的使用情况。旨在解决模型对AAVE表达不准确及可能强化刻板印象的问题。通过对比分析模型与人类使用模式，发现模型存在用法偏差和 stereotypes 复制问题。**

- **链接: [https://arxiv.org/pdf/2602.21485v1](https://arxiv.org/pdf/2602.21485v1)**

> **作者:** Deja Dunlap; R. Thomas McCoy
>
> **摘要:** In AI, most evaluations of natural language understanding tasks are conducted in standardized dialects such as Standard American English (SAE). In this work, we investigate how accurately large language models (LLMs) represent African American Vernacular English (AAVE). We analyze three LLMs to compare their usage of AAVE to the usage of humans who natively speak AAVE. We first analyzed interviews from the Corpus of Regional African American Language and TwitterAAE to identify the typical contexts where people use AAVE grammatical features such as ain't. We then prompted the LLMs to produce text in AAVE and compared the model-generated text to human usage patterns. We find that, in many cases, there are substantial differences between AAVE usage in LLMs and humans: LLMs usually underuse and misuse grammatical features characteristic of AAVE. Furthermore, through sentiment analysis and manual inspection, we found that the models replicated stereotypes about African Americans. These results highlight the need for more diversity in training data and the incorporation of fairness methods to mitigate the perpetuation of stereotypes.
>
---
#### [new 041] Disaster Question Answering with LoRA Efficiency and Accurate End Position
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于灾害问答任务，旨在解决灾害情境下信息不准确和响应困难的问题。通过优化模型结构，提升问答准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.21212v1](https://arxiv.org/pdf/2602.21212v1)**

> **作者:** Takato Yasuno
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Natural disasters such as earthquakes, torrential rainfall, floods, and volcanic eruptions occur with extremely low frequency and affect limited geographic areas. When individuals face disaster situations, they often experience confusion and lack the domain-specific knowledge and experience necessary to determine appropriate responses and actions. While disaster information is continuously updated, even when utilizing RAG search and large language models for inquiries, obtaining relevant domain knowledge about natural disasters and experiences similar to one's specific situation is not guaranteed. When hallucinations are included in disaster question answering, artificial misinformation may spread and exacerbate confusion. This work introduces a disaster-focused question answering system based on Japanese disaster situations and response experiences. Utilizing the cl-tohoku/bert-base-japanese-v3 + Bi-LSTM + Enhanced Position Heads architecture with LoRA efficiency optimization, we achieved 70.4\% End Position accuracy with only 5.7\% of the total parameters (6.7M/117M). Experimental results demonstrate that the combination of Japanese BERT-base optimization and Bi-LSTM contextual understanding achieves accuracy levels suitable for real disaster response scenarios, attaining a 0.885 Span F1 score. Future challenges include: establishing natural disaster Q\&A benchmark datasets, fine-tuning foundation models with disaster knowledge, developing lightweight and power-efficient edge AI Disaster Q\&A applications for situations with insufficient power and communication during disasters, and addressing disaster knowledge base updates and continual learning capabilities.
>
---
#### [new 042] Sparsity Induction for Accurate Post-Training Pruning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决后训练剪枝中因稀疏性不足导致性能下降的问题。通过提升分布和特征层面的稀疏性，实现更高效的模型剪枝。**

- **链接: [https://arxiv.org/pdf/2602.21652v1](https://arxiv.org/pdf/2602.21652v1)**

> **作者:** Minhao Jiang; Zhikai Li; Xuewen Liu; Jing Zhang; Mengjuan Chen; Qingyi Gu
>
> **备注:** 5 pages, 1 figure, 4 tables
>
> **摘要:** Large language models have demonstrated capabilities in text generation, while their increasing parameter scales present challenges in computational and memory efficiency. Post-training sparsity (PTS), which reduces model cost by removing weights from dense networks, is an effective approach. However, native dense matrices lack high sparsity, making existing approaches that directly remove weights disrupt model states, resulting in unsatisfactory performance recovery even with post-tuning. We propose Sparsity Induction, which promotes models toward higher sparsity at both distribution and feature levels before pruning, to push the limits of PTS. At the distribution level, we enhance distributional sparsity through mathematically equivalent scaling transformations, which are fully absorbable and incur no extra parameters or inference-time overhead. At the feature level, we introduce Spectral Norm Loss to promote feature sparsity from a low-rank perspective. Experiments across diverse model architectures and tasks demonstrate that our method further enhances sparsity-friendliness, achieving superior pruning performance over existing approaches.
>
---
#### [new 043] IndicIFEval: A Benchmark for Verifiable Instruction-Following Evaluation in 14 Indic Languages
- **分类: cs.CL**

- **简介: 该论文提出IndicIFEval，一个用于评估14种印地语系语言中指令遵循能力的基准。旨在解决多语言指令生成评估不足的问题，通过自动验证的规则指令进行测试。**

- **链接: [https://arxiv.org/pdf/2602.22125v1](https://arxiv.org/pdf/2602.22125v1)**

> **作者:** Thanmay Jayakumar; Mohammed Safi Ur Rahman Khan; Raj Dabre; Ratish Puduppully; Anoop Kunchukuttan
>
> **备注:** 8 pages + Appendix
>
> **摘要:** Instruction-following benchmarks remain predominantly English-centric, leaving a critical evaluation gap for the hundreds of millions of Indic language speakers. We introduce IndicIFEval, a benchmark evaluating constrained generation of LLMs across 14 Indic languages using automatically verifiable, rule-based instructions. It comprises around 800 human-verified examples per language spread across two complementary subsets: IndicIFEval-Ground, translated prompts from IFEval (Zhou et al., 2023) carefully localized for Indic contexts, and IndicIFEval-Ground, synthetically generated instructions grounded in native Indic content. We conduct a comprehensive evaluation of major open-weight and proprietary models spanning both reasoning and non-reasoning models. While models maintain strong adherence to formatting constraints, they struggle significantly with lexical and cross-lingual tasks -- and despite progress in high-resource languages, instruction-following across the broader Indic family lags significantly behind English. We release IndicIFEval and its evaluation scripts to support progress on multilingual constrained generation (http://github.com/ai4bharat/IndicIFEval).
>
---
#### [new 044] EQ-5D Classification Using Biomedical Entity-Enriched Pre-trained Language Models and Multiple Instance Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本分类任务，旨在解决EQ-5D文献自动识别问题。通过融合生物医学实体信息和预训练语言模型，提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2602.21216v1](https://arxiv.org/pdf/2602.21216v1)**

> **作者:** Zhyar Rzgar K Rostam; Gábor Kertész
>
> **备注:** 12 tables
>
> **摘要:** The EQ-5D (EuroQol 5-Dimensions) is a standardized instrument for the evaluation of health-related quality of life. In health economics, systematic literature reviews (SLRs) depend on the correct identification of publications that use the EQ-5D, but manual screening of large volumes of scientific literature is time-consuming, error-prone, and inconsistent. In this study, we investigate fine-tuning of general-purpose (BERT) and domain-specific (SciBERT, BioBERT) pre-trained language models (PLMs), enriched with biomedical entity information extracted through scispaCy models for each statement, to improve EQ-5D detection from abstracts. We conduct nine experimental setups, including combining three scispaCy models with three PLMs, and evaluate their performance at both the sentence and study levels. Furthermore, we explore a Multiple Instance Learning (MIL) approach with attention pooling to aggregate sentence-level information into study-level predictions, where each abstract is represented as a bag of enriched sentences (by scispaCy). The findings indicate consistent improvements in F1-scores (reaching 0.82) and nearly perfect recall at the study-level, significantly exceeding classical bag-of-words baselines and recently reported PLM baselines. These results show that entity enrichment significantly improves domain adaptation and model generalization, enabling more accurate automated screening in systematic reviews.
>
---
#### [new 045] Make Every Draft Count: Hidden State based Speculative Decoding
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **简介: 该论文属于语言模型推理优化任务，旨在解决推测解码中计算浪费的问题。通过隐藏状态级的自回归预测，提升废弃草稿的复用性，实现更高效的推理速度。**

- **链接: [https://arxiv.org/pdf/2602.21224v1](https://arxiv.org/pdf/2602.21224v1)**

> **作者:** Yuetao Chen; Xuliang Wang; Xinzhou Zheng; Ming Li; Peng Wang; Hong Xu
>
> **摘要:** Speculative decoding has emerged as a pivotal technique to accelerate LLM inference by employing a lightweight draft model to generate candidate tokens that are subsequently verified by the target model in parallel. However, while this paradigm successfully increases the arithmetic intensity of memory-bound inference, it causes significant compute inefficiency: the majority of draft tokens fail verification and are discarded, resulting in waste of computation. Motivated by the goal of recollecting this wasted computation, we propose a novel system that transforms discarded drafts into reusable tokens. Our key insight is to perform auto-regressive prediction at the hidden states level and postpone the integrating token information after the hidden states generation, so the draft hidden states are not contaminated by incorrect tokens, enabling hidden state reuse. To implement such a system, first we introduce a draft model architecture based on auto-regressive hidden states, which preserves richer semantics than token-based drafters to facilitate draft repurposing. Second, we design an efficient token information injection mechanism that leverages our specialized draft model to construct high-quality draft token trees and enables resampling tokens from verification failures. Third, we eliminate the overhead hidden in our design to further maximize hardware utilization. We conducted extensive evaluations against various baselines, demonstrating up to a 3.3x speedup against standard speculative decoding.
>
---
#### [new 046] Under the Influence: Quantifying Persuasion and Vigilance in Large Language Models
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于AI安全领域，研究LLMs在说服与警惕性方面的表现。通过Sokoban游戏，探讨模型在任务执行、被说服及识别误导间的关系，揭示三者可分离的特性。**

- **链接: [https://arxiv.org/pdf/2602.21262v1](https://arxiv.org/pdf/2602.21262v1)**

> **作者:** Sasha Robinson; Kerem Oktar; Katherine M. Collins; Ilia Sucholutsky; Kelsey R. Allen
>
> **摘要:** With increasing integration of Large Language Models (LLMs) into areas of high-stakes human decision-making, it is important to understand the risks they introduce as advisors. To be useful advisors, LLMs must sift through large amounts of content, written with both benevolent and malicious intent, and then use this information to convince a user to take a specific action. This involves two social capacities: vigilance (the ability to determine which information to use, and which to discard) and persuasion (synthesizing the available evidence to make a convincing argument). While existing work has investigated these capacities in isolation, there has been little prior investigation of how these capacities may be linked. Here, we use a simple multi-turn puzzle-solving game, Sokoban, to study LLMs' abilities to persuade and be rationally vigilant towards other LLM agents. We find that puzzle-solving performance, persuasive capability, and vigilance are dissociable capacities in LLMs. Performing well on the game does not automatically mean a model can detect when it is being misled, even if the possibility of deception is explicitly mentioned. % as part of the prompt. However, LLMs do consistently modulate their token use, using fewer tokens to reason when advice is benevolent and more when it is malicious, even if they are still persuaded to take actions leading them to failure. To our knowledge, our work presents the first investigation of the relationship between persuasion, vigilance, and task performance in LLMs, and suggests that monitoring all three independently will be critical for future work in AI safety.
>
---
#### [new 047] Small Language Models for Privacy-Preserving Clinical Information Extraction in Low-Resource Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于临床信息提取任务，旨在解决低资源语言中隐私保护的临床数据抽取问题。通过翻译与小模型结合的方法，提升提取效果。**

- **链接: [https://arxiv.org/pdf/2602.21374v1](https://arxiv.org/pdf/2602.21374v1)**

> **作者:** Mohammadreza Ghaffarzadeh-Esfahani; Nahid Yousefian; Ebrahim Heidari-Farsani; Ali Akbar Omidvarian; Sepehr Ghahraei; Atena Farangi; AmirBahador Boroumand
>
> **备注:** 16 pages, 3 figures, 2 supplementary files
>
> **摘要:** Extracting clinical information from medical transcripts in low-resource languages remains a significant challenge in healthcare natural language processing (NLP). This study evaluates a two-step pipeline combining Aya-expanse-8B as a Persian-to-English translation model with five open-source small language models (SLMs) -- Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-1.5B-Instruct, and Gemma-3-1B-it -- for binary extraction of 13 clinical features from 1,221 anonymized Persian transcripts collected at a cancer palliative care call center. Using a few-shot prompting strategy without fine-tuning, models were assessed on macro-averaged F1-score, Matthews Correlation Coefficient (MCC), sensitivity, and specificity to account for class imbalance. Qwen2.5-7B-Instruct achieved the highest overall performance (median macro-F1: 0.899; MCC: 0.797), while Gemma-3-1B-it showed the weakest results. Larger models (7B--8B parameters) consistently outperformed smaller counterparts in sensitivity and MCC. A bilingual analysis of Aya-expanse-8B revealed that translating Persian transcripts to English improved sensitivity, reduced missing outputs, and boosted metrics robust to class imbalance, though at the cost of slightly lower specificity and precision. Feature-level results showed reliable extraction of physiological symptoms across most models, whereas psychological complaints, administrative requests, and complex somatic features remained challenging. These findings establish a practical, privacy-preserving blueprint for deploying open-source SLMs in multilingual clinical NLP settings with limited infrastructure and annotation resources, and highlight the importance of jointly optimizing model scale and input language strategy for sensitive healthcare applications.
>
---
#### [new 048] Evaluating the relationship between regularity and learnability in recursive numeral systems using Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言学习与结构研究任务，探讨规则性与可学性之间的关系。通过强化学习方法，验证规则系统更易学习，揭示语言结构的可学性对语言普遍性的影响。**

- **链接: [https://arxiv.org/pdf/2602.21720v1](https://arxiv.org/pdf/2602.21720v1)**

> **作者:** Andrea Silvi; Ponrawee Prasertsom; Jennifer Culbertson; Devdatt Dubhashi; Moa Johansson; Kenny Smith
>
> **摘要:** Human recursive numeral systems (i.e., counting systems such as English base-10 numerals), like many other grammatical systems, are highly regular. Following prior work that relates cross-linguistic tendencies to biases in learning, we ask whether regular systems are common because regularity facilitates learning. Adopting methods from the Reinforcement Learning literature, we confirm that highly regular human(-like) systems are easier to learn than unattested but possible irregular systems. This asymmetry emerges under the natural assumption that recursive numeral systems are designed for generalisation from limited data to represent all integers exactly. We also find that the influence of regularity on learnability is absent for unnatural, highly irregular systems, whose learnability is influenced instead by signal length, suggesting that different pressures may influence learnability differently in different parts of the space of possible numeral systems. Our results contribute to the body of work linking learnability to cross-linguistic prevalence.
>
---
#### [new 049] MrBERT: Modern Multilingual Encoders via Vocabulary, Domain, and Dimensional Adaptation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MrBERT，一种多语言编码器，解决跨语言和领域适应问题。通过优化词汇、领域和维度，提升性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.21379v1](https://arxiv.org/pdf/2602.21379v1)**

> **作者:** Daniel Tamayo; Iñaki Lacunza; Paula Rivera-Hidalgo; Severino Da Dalt; Javier Aula-Blasco; Aitor Gonzalez-Agirre; Marta Villegas
>
> **备注:** 24 pages, 14 tables and 4 figures
>
> **摘要:** We introduce MrBERT, a family of 150M-300M parameter encoders built on the ModernBERT architecture and pre-trained on 35 languages and code. Through targeted adaptation, this model family achieves state-of-the-art results on Catalan- and Spanish-specific tasks, while establishing robust performance across specialized biomedical and legal domains. To bridge the gap between research and production, we incorporate Matryoshka Representation Learning (MRL), enabling flexible vector sizing that significantly reduces inference and storage costs. Ultimately, the MrBERT family demonstrates that modern encoder architectures can be optimized for both localized linguistic excellence and efficient, high-stakes domain specialization. We open source the complete model family on Huggingface.
>
---
#### [new 050] LiCQA : A Lightweight Complex Question Answering System
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于问答系统任务，旨在解决多文档复杂问题的回答难题。提出LiCQA模型，基于语料库证据，无需训练数据，效率更高。**

- **链接: [https://arxiv.org/pdf/2602.22182v1](https://arxiv.org/pdf/2602.22182v1)**

> **作者:** Sourav Saha; Dwaipayan Roy; Mandar Mitra
>
> **摘要:** Over the last twenty years, significant progress has been made in designing and implementing Question Answering (QA) systems. However, addressing complex questions, the answers to which are spread across multiple documents, remains a challenging problem. Recent QA systems that are designed to handle complex questions work either on the basis of knowledge graphs, or utilise contem- porary neural models that are expensive to train, in terms of both computational resources and the volume of training data required. In this paper, we present LiCQA, an unsupervised question answer- ing model that works primarily on the basis of corpus evidence. We empirically compare the effectiveness and efficiency of LiCQA with two recently presented QA systems, which are based on different underlying principles. The results of our experiments show that LiCQA significantly outperforms these two state-of-the-art systems on benchmark data with noteworthy reduction in latency.
>
---
#### [new 051] Beyond Subtokens: A Rich Character Embedding for Low-resource and Morphologically Complex Languages
- **分类: cs.CL**

- **简介: 该论文针对低资源和形态复杂语言的词向量表示问题，提出基于字符的丰富嵌入方法RCE，结合Transformer与卷积机制，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.21377v1](https://arxiv.org/pdf/2602.21377v1)**

> **作者:** Felix Schneider; Maria Gogolev; Sven Sickert; Joachim Denzler
>
> **备注:** 12 content pages, 2 figures, 8 tables, one example textbox
>
> **摘要:** Tokenization and sub-tokenization based models like word2vec, BERT and the GPTs are the state-of-the-art in natural language processing. Typically, these approaches have limitations with respect to their input representation. They fail to fully capture orthographic similarities and morphological variations, especially in highly inflected and under-resource languages. To mitigate this problem, we propose to computes word vectors directly from character strings, integrating both semantic and syntactic information. We denote this transformer-based approach Rich Character Embeddings (RCE). Furthermore, we propose a hybrid model that combines transformer and convolutional mechanisms. Both vector representations can be used as a drop-in replacement for dictionary- and subtoken-based word embeddings in existing model architectures. It has the potential to improve performance for both large context-based language models like BERT and small models like word2vec for under-resourced and morphologically rich languages. We evaluate our approach on various tasks like the SWAG, declension prediction for inflected languages, metaphor and chiasmus detection for various languages. Our experiments show that it outperforms traditional token-based approaches on limited data using OddOneOut and TopK metrics.
>
---
#### [new 052] Multi-dimensional Assessment and Explainable Feedback for Counselor Responses to Client Resistance in Text-based Counseling with LLMs
- **分类: cs.CL**

- **简介: 该论文属于心理辅导质量评估任务，旨在解决如何精准评价咨询师应对来访者抗拒的回应问题。研究构建了多维评估框架，开发模型进行细粒度评价并生成解释性反馈。**

- **链接: [https://arxiv.org/pdf/2602.21638v1](https://arxiv.org/pdf/2602.21638v1)**

> **作者:** Anqi Li; Ruihan Wang; Zhaoming Chen; Yuqian Chen; Yu Lu; Yi Zhu; Yuan Xie; Zhenzhong Lan
>
> **备注:** 8 pages
>
> **摘要:** Effectively addressing client resistance is a sophisticated clinical skill in psychological counseling, yet practitioners often lack timely and scalable supervisory feedback to refine their approaches. Although current NLP research has examined overall counseling quality and general therapeutic skills, it fails to provide granular evaluations of high-stakes moments where clients exhibit resistance. In this work, we present a comprehensive pipeline for the multi-dimensional evaluation of human counselors' interventions specifically targeting client resistance in text-based therapy. We introduce a theory-driven framework that decomposes counselor responses into four distinct communication mechanisms. Leveraging this framework, we curate and share an expert-annotated dataset of real-world counseling excerpts, pairing counselor-client interactions with professional ratings and explanatory rationales. Using this data, we perform full-parameter instruction tuning on a Llama-3.1-8B-Instruct backbone to model fine-grained evaluative judgments of response quality and generate explanations underlying. Experimental results show that our approach can effectively distinguish the quality of different communication mechanisms (77-81% F1), substantially outperforming GPT-4o and Claude-3.5-Sonnet (45-59% F1). Moreover, the model produces high-quality explanations that closely align with expert references and receive near-ceiling ratings from human experts (2.8-2.9/3.0). A controlled experiment with 43 counselors further confirms that receiving these AI-generated feedback significantly improves counselors' ability to respond effectively to client resistance.
>
---
#### [new 053] RADAR: Reasoning as Discrimination with Aligned Representations for LLM-based Knowledge Graph Reasoning
- **分类: cs.CL**

- **简介: 该论文属于知识图谱推理任务，旨在解决LLM在推理中过度依赖表面共现而非真实关系语义的问题。提出RADAR方法，通过判别式实体选择提升推理效果和泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21951v1](https://arxiv.org/pdf/2602.21951v1)**

> **作者:** Bo Xue; Yuan Jin; Luoyi Fu; Jiaxin Ding; Xinbing Wang
>
> **摘要:** Knowledge graph reasoning (KGR) infers missing facts, with recent advances increasingly harnessing the semantic priors and reasoning abilities of Large Language Models (LLMs). However, prevailing generative paradigms are prone to memorizing surface-level co-occurrences rather than learning genuine relational semantics, limiting out-of-distribution generalization. To address this, we propose RADAR, which reformulates KGR from generative pattern matching to discriminative relational reasoning. We recast KGR as discriminative entity selection, where reinforcement learning enforces relative entity separability beyond token-likelihood imitation. Leveraging this separability, inference operates directly in representation space, ensuring consistency with the discriminative optimization and bypassing generation-induced hallucinations. Across four benchmarks, RADAR achieves 5-6% relative gains on link prediction and triple classification over strong LLM baselines, while increasing task-relevant mutual information in intermediate representations by 62.9%, indicating more robust and transferable relational reasoning.
>
---
#### [new 054] Dynamic Personality Adaptation in Large Language Models via State Machines
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决LLM无法动态调整个性表达的问题。通过状态机和评分管道实现个性适应，提升交互效果。**

- **链接: [https://arxiv.org/pdf/2602.22157v1](https://arxiv.org/pdf/2602.22157v1)**

> **作者:** Leon Pielage; Ole Hätscher; Mitja Back; Bernhard Marschall; Benjamin Risse
>
> **备注:** 22 pages, 5 figures, submitted to ICPR 2026
>
> **摘要:** The inability of Large Language Models (LLMs) to modulate their personality expression in response to evolving dialogue dynamics hinders their performance in complex, interactive contexts. We propose a model-agnostic framework for dynamic personality simulation that employs state machines to represent latent personality states, where transition probabilities are dynamically adapted to the conversational context. Part of our architecture is a modular pipeline for continuous personality scoring that evaluates dialogues along latent axes while remaining agnostic to the specific personality models, their dimensions, transition mechanisms, or LLMs used. These scores function as dynamic state variables that systematically reconfigure the system prompt, steering behavioral alignment throughout the interaction.We evaluate this framework by operationalizing the Interpersonal Circumplex (IPC) in a medical education setting. Results demonstrate that the system successfully adapts its personality state to user inputs, but also influences user behavior, thereby facilitating de-escalation training. Notably, the scoring pipeline maintains comparable precision even when utilizing lightweight, fine-tuned classifiers instead of large-scale LLMs. This work demonstrates the feasibility of modular, personality-adaptive architectures for education, customer support, and broader human-computer interaction.
>
---
#### [new 055] Inference-time Alignment via Sparse Junction Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，解决推理时对齐效率与质量的平衡问题。提出SIA方法，在关键决策点进行稀疏干预，提升对齐效果并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.21215v1](https://arxiv.org/pdf/2602.21215v1)**

> **作者:** Runyi Hu; Jie Zhang; Shiqian Zhao; Jiale Meng; Jiwei Li; Jason Zeng; Ming Wu; Michael Heinrich; Yonggang Wen; Tianwei Zhang
>
> **备注:** 28 pages, 17 figures
>
> **摘要:** Token-level steering has emerged as a pivotal approach for inference-time alignment, enabling fine grained control over large language models by modulating their output distributions without parameter updates. While effective, existing methods rely on dense intervention at every decoding step. This persistent manipulation not only incurs substantial computational overhead but also risks compromising generation quality by excessively drifting from the model's intrinsic distribution. In this work, we show that dense intervention is unnecessary and propose Sparse Inference time Alignment (SIA), which performs sparse junction steering by intervening only at critical decision points along the generation trajectory. Our key insight is that high entropy junctions mark pivotal decision points in the generation trajectory and are particularly susceptible to misalignment, indicating the need to introduce alignment related reward signals at these points. Extensive experiments across different model families and alignment objectives show that steering only 20% to 80% of tokens achieves superior alignment-efficiency trade offs. For strong base models such as Qwen3, intervening on as few as 20% of tokens matches or even surpasses heavily post-trained instruct models. This sparsity enables stronger guidance while better preserving the model's native distribution, integrates seamlessly with search based methods such as Best-of-N, and reduces computational cost by up to 6x.
>
---
#### [new 056] VecGlypher: Unified Vector Glyph Generation with Language Models
- **分类: cs.CL**

- **简介: 该论文提出VecGlypher，解决矢量字形生成问题，通过语言模型直接生成高质量矢量字形，提升可编辑性和可访问性。**

- **链接: [https://arxiv.org/pdf/2602.21461v1](https://arxiv.org/pdf/2602.21461v1)**

> **作者:** Xiaoke Huang; Bhavul Gauri; Kam Woh Ng; Tony Ng; Mengmeng Xu; Zhiheng Liu; Weiming Ren; Zhaochong An; Zijian Zhou; Haonan Qiu; Yuyin Zhou; Sen He; Ziheng Wang; Tao Xiang; Xiao Han
>
> **备注:** Accepted to CVPR'26. Project page: https://xk-huang.github.io/VecGlypher/
>
> **摘要:** Vector glyphs are the atomic units of digital typography, yet most learning-based pipelines still depend on carefully curated exemplar sheets and raster-to-vector postprocessing, which limits accessibility and editability. We introduce VecGlypher, a single multimodal language model that generates high-fidelity vector glyphs directly from text descriptions or image exemplars. Given a style prompt, optional reference glyph images, and a target character, VecGlypher autoregressively emits SVG path tokens, avoiding raster intermediates and producing editable, watertight outlines in one pass. A typography-aware data and training recipe makes this possible: (i) a large-scale continuation stage on 39K noisy Envato fonts to master SVG syntax and long-horizon geometry, followed by (ii) post-training on 2.5K expert-annotated Google Fonts with descriptive tags and exemplars to align language and imagery with geometry; preprocessing normalizes coordinate frames, canonicalizes paths, de-duplicates families, and quantizes coordinates for stable long-sequence decoding. On cross-family OOD evaluation, VecGlypher substantially outperforms both general-purpose LLMs and specialized vector-font baselines for text-only generation, while image-referenced generation reaches a state-of-the-art performance, with marked gains over DeepVecFont-v2 and DualVector. Ablations show that model scale and the two-stage recipe are critical and that absolute-coordinate serialization yields the best geometry. VecGlypher lowers the barrier to font creation by letting users design with words or exemplars, and provides a scalable foundation for future multimodal design tools.
>
---
#### [new 057] When More Is Less: A Systematic Analysis of Spatial and Commonsense Information for Visual Spatial Reasoning
- **分类: cs.CL**

- **简介: 该论文属于视觉空间推理任务，探讨信息注入对模型性能的影响。研究发现过多或不相关的信息可能降低效果，强调精准、有针对性的信息使用。**

- **链接: [https://arxiv.org/pdf/2602.21619v1](https://arxiv.org/pdf/2602.21619v1)**

> **作者:** Muku Akasaka; Soyeon Caren Han
>
> **备注:** 5 pages, 6 figures, Under review
>
> **摘要:** Visual spatial reasoning (VSR) remains challenging for modern vision-language models (VLMs), despite advances in multimodal architectures. A common strategy is to inject additional information at inference time, such as explicit spatial cues, external commonsense knowledge, or chain-of-thought (CoT) reasoning instructions. However, it remains unclear when such information genuinely improves reasoning and when it introduces noise. In this paper, we conduct a hypothesis-driven analysis of information injection for VSR across three representative VLMs and two public benchmarks. We examine (i) the type and number of spatial contexts, (ii) the amount and relevance of injected commonsense knowledge, and (iii) the interaction between spatial grounding and CoT prompting. Our results reveal a consistent pattern: more information does not necessarily yield better reasoning. Targeted single spatial cues outperform multi-context aggregation, excessive or weakly relevant commonsense knowledge degrades performance, and CoT prompting improves accuracy only when spatial grounding is sufficiently precise. These findings highlight the importance of selective, task-aligned information injection and provide practical guidance for designing reliable multimodal reasoning pipelines.
>
---
#### [new 058] D-COT: Disciplined Chain-of-Thought Learning for Efficient Reasoning in Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于小语言模型的推理优化任务，解决CoT蒸馏导致的过度思考问题。通过引入控制标签构建结构化推理流程，提升准确率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2602.21786v1](https://arxiv.org/pdf/2602.21786v1)**

> **作者:** Shunsuke Ubukata
>
> **备注:** 9 pages, 3 figures. Code: https://github.com/gitpullpull/DisciplinedChainOfThought | Benchmarks: https://huggingface.co/datasets/gitpullpull/D-CoT-Benchmarks | Dataset: https://huggingface.co/datasets/gitpullpull/D-CoT-datasets
>
> **摘要:** Chain-of-Thought (CoT) distillation from Large Language Models (LLMs) often induces "overthinking" in Small Language Models (SLMs), leading to performance degradation and excessive token consumption. In this study, we propose Disciplined Chain-of-Thought (D-CoT), a novel framework that enforces a structured reasoning process using control tags -- such as <TEMP_LOW> for fact-checking and <TEMP_HIGH> for multi-perspective exploration -- as auxiliary scaffolding during training. By optimizing the CoT trajectory, D-CoT suppresses reasoning drift and simultaneously achieves token reduction and performance improvement. We demonstrate the efficacy of our approach on Qwen3-8B: with only 5,000 training samples, D-CoT significantly boosts accuracy on GPQA-diamond by 9.9% and MMLU-Pro (0-shot) by 9.1%, while drastically reducing computational costs. Furthermore, we confirm that the model internalizes this disciplined thought structure, maintaining high performance even without explicit control tags during inference.
>
---
#### [new 059] Forecasting Future Language: Context Design for Mention Markets
- **分类: q-fin.GN; cs.CL; cs.LG**

- **简介: 该论文研究预测市场中关键词提及的准确预测任务，解决如何设计输入上下文以提升预测性能的问题。通过实验和引入Market-Conditioned Prompting方法，提升预测准确性与校准度。**

- **链接: [https://arxiv.org/pdf/2602.21229v1](https://arxiv.org/pdf/2602.21229v1)**

> **作者:** Sumin Kim; Jihoon Kwon; Yoon Kim; Nicole Kagan; Raffi Khatchadourian; Wonbin Ahn; Alejandro Lopez-Lira; Jaewon Lee; Yoontae Hwang; Oscar Levy; Yongjae Lee; Chanyeol Choi
>
> **备注:** 10 pages
>
> **摘要:** Mention markets, a type of prediction market in which contracts resolve based on whether a specified keyword is mentioned during a future public event, require accurate probabilistic forecasts of keyword-mention outcomes. While recent work shows that large language models (LLMs) can generate forecasts competitive with human forecasters, it remains unclear how input context should be designed to support accurate prediction. In this paper, we study this question through experiments on earnings-call mention markets, which require forecasting whether a company will mention a specified keyword during its upcoming call. We run controlled comparisons varying (i) which contextual information is provided (news and/or prior earnings-call transcripts) and (ii) how \textit{market probability}, (i.e., prediction market contract price) is used. We introduce Market-Conditioned Prompting (MCP), which explicitly treats the market-implied probability as a prior and instructs the LLM to update this prior using textual evidence, rather than re-predicting the base rate from scratch. In our experiments, we find three insights: (1) richer context consistently improves forecasting performance; (2) market-conditioned prompting (MCP), which treats the market probability as a prior and updates it using textual evidence, yields better-calibrated forecasts; and (3) a mixture of the market probability and MCP (MixMCP) outperforms the market baseline. By dampening the LLM's posterior update with the market prior, MixMCP yields more robust predictions than either the market or the LLM alone.
>
---
#### [new 060] Black-Box Reliability Certification for AI Agents via Self-Consistency Sampling and Conformal Calibration
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文提出一种黑盒AI系统可靠性认证方法，通过自一致性采样和 conformal 校准，为每个系统-任务对提供可靠度指标，解决AI输出可信度评估问题。**

- **链接: [https://arxiv.org/pdf/2602.21368v1](https://arxiv.org/pdf/2602.21368v1)**

> **作者:** Charafeddine Mouzouni
>
> **备注:** 41 pages, 11 figures, 10 tables, including appendices
>
> **摘要:** Given a black-box AI system and a task, at what confidence level can a practitioner trust the system's output? We answer with a reliability level -- a single number per system-task pair, derived from self-consistency sampling and conformal calibration, that serves as a black-box deployment gate with exact, finite-sample, distribution-free guarantees. Self-consistency sampling reduces uncertainty exponentially; conformal calibration guarantees correctness within 1/(n+1) of the target level, regardless of the system's errors -- made transparently visible through larger answer sets for harder questions. Weaker models earn lower reliability levels (not accuracy -- see Definition 2.4): GPT-4.1 earns 94.6% on GSM8K and 96.8% on TruthfulQA, while GPT-4.1-nano earns 89.8% on GSM8K and 66.5% on MMLU. We validate across five benchmarks, five models from three families, and both synthetic and real data. Conditional coverage on solvable items exceeds 0.93 across all configurations; sequential stopping reduces API costs by around 50%.
>
---
#### [new 061] Revisiting Text Ranking in Deep Research
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于深度研究任务，旨在解决文本排序方法在深度研究中的效果问题。通过实验分析不同检索单元、管道配置和查询特征的影响，提升搜索效果。**

- **链接: [https://arxiv.org/pdf/2602.21456v1](https://arxiv.org/pdf/2602.21456v1)**

> **作者:** Chuan Meng; Litu Ou; Sean MacAvaney; Jeff Dalton
>
> **摘要:** Deep research has emerged as an important task that aims to address hard queries through extensive open-web exploration. To tackle it, most prior work equips large language model (LLM)-based agents with opaque web search APIs, enabling agents to iteratively issue search queries, retrieve external evidence, and reason over it. Despite search's essential role in deep research, black-box web search APIs hinder systematic analysis of search components, leaving the behaviour of established text ranking methods in deep research largely unclear. To fill this gap, we reproduce a selection of key findings and best practices for IR text ranking methods in the deep research setting. In particular, we examine their effectiveness from three perspectives: (i) retrieval units (documents vs. passages), (ii) pipeline configurations (different retrievers, re-rankers, and re-ranking depths), and (iii) query characteristics (the mismatch between agent-issued queries and the training queries of text rankers). We perform experiments on BrowseComp-Plus, a deep research dataset with a fixed corpus, evaluating 2 open-source agents, 5 retrievers, and 3 re-rankers across diverse setups. We find that agent-issued queries typically follow web-search-style syntax (e.g., quoted exact matches), favouring lexical, learned sparse, and multi-vector retrievers; passage-level units are more efficient under limited context windows, and avoid the difficulties of document length normalisation in lexical retrieval; re-ranking is highly effective; translating agent-issued queries into natural-language questions significantly bridges the query mismatch.
>
---
#### [new 062] Latent Context Compilation: Distilling Long Context into Compact Portable Memory
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Latent Context Compilation，解决长文本处理中压缩与泛化难题。通过编译机制将长上下文压缩为便携记忆，无需修改模型参数，提升部署效率。**

- **链接: [https://arxiv.org/pdf/2602.21221v1](https://arxiv.org/pdf/2602.21221v1)**

> **作者:** Zeju Li; Yizhou Zhou; Qiang Xu
>
> **摘要:** Efficient long-context LLM deployment is stalled by a dichotomy between amortized compression, which struggles with out-of-distribution generalization, and Test-Time Training, which incurs prohibitive synthetic data costs and requires modifying model weights, creating stateful parameters that complicate concurrent serving. We propose Latent Context Compilation, a framework that fundamentally shifts context processing from adaptation to compilation. By utilizing a disposable LoRA module as a compiler, we distill long contexts into compact buffer tokens -- stateless, portable memory artifacts that are plug-and-play compatible with frozen base models. Crucially, we introduce a self-aligned optimization strategy that eliminates the need for synthetic context-relevant QA pairs. By regularizing context reconstruction task with context-agnostic random queries, we force compressed tokens to reside within the model's existing instruction-following manifold. Experiments with Llama-3.1-8B demonstrate that Latent Context Compilation preserves fine-grained details and reasoning capabilities where prior methods falter, effectively decoupling memory density from model parameters even at a 16x compression ratio.
>
---
#### [new 063] Toward Effective Multi-Domain Rumor Detection in Social Networks Using Domain-Gated Mixture-of-Experts
- **分类: cs.SI; cs.CL; cs.IR**

- **简介: 该论文属于多领域谣言检测任务，旨在解决跨领域谣言识别准确率下降的问题。提出PerFact数据集和基于门控专家混合的检测模型，提升多领域场景下的检测效果。**

- **链接: [https://arxiv.org/pdf/2602.21214v1](https://arxiv.org/pdf/2602.21214v1)**

> **作者:** Mohadeseh Sheikhqoraei; Zainabolhoda Heshmati; Zeinab Rajabi; Leila Rabiei
>
> **摘要:** Social media platforms have become key channels for spreading and tracking rumors due to their widespread accessibility and ease of information sharing. Rumors can continuously emerge across diverse domains and topics, often with the intent to mislead society for personal or commercial gain. Therefore, developing methods that can accurately detect rumors at early stages is crucial to mitigating their negative impact. While existing approaches often specialize in single-domain detection, their performance degrades when applied to new domains due to shifts in data distribution, such as lexical patterns and propagation dynamics. To bridge this gap, this study introduces PerFact, a large-scale multi-domain rumor dataset comprising 8,034 annotated posts from the X platform, annotated into two primary categories: rumor (including true, false, and unverified rumors) and non-rumor. Annotator agreement, measured via Fleiss' Kappa ($κ= 0.74$), ensures high-quality labels. This research further proposes an effective multi-domain rumor detection model that employs a domain gate to dynamically aggregate multiple feature representations extracted through a Mixture-of-Experts method. Each expert combines CNN and BiLSTM networks to capture local syntactic features and long-range contextual dependencies. By leveraging both textual content and publisher information, the proposed model classifies posts into rumor and non-rumor categories with high accuracy. Evaluations demonstrate state-of-the-art performance, achieving an F1-score of 79.86\% and an accuracy of 79.98\% in multi-domain settings. Keywords: Rumor Detection, Multi-Domain, Natural Language Processing, Social Networks, Mixture-of-Experts Model
>
---
#### [new 064] TG-ASR: Translation-Guided Learning with Parallel Gated Cross Attention for Low-Resource Automatic Speech Recognition
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 论文提出TG-ASR框架，解决低资源语言（如台湾闽南语）语音识别难题。通过翻译引导和跨语言注意力机制，提升识别效果。**

- **链接: [https://arxiv.org/pdf/2602.22039v1](https://arxiv.org/pdf/2602.22039v1)**

> **作者:** Cheng-Yeh Yang; Chien-Chun Wang; Li-Wei Chen; Hung-Shin Lee; Hsin-Min Wang; Berlin Chen
>
> **备注:** Accepted to LREC 2026
>
> **摘要:** Low-resource automatic speech recognition (ASR) continues to pose significant challenges, primarily due to the limited availability of transcribed data for numerous languages. While a wealth of spoken content is accessible in television dramas and online videos, Taiwanese Hokkien exemplifies this issue, with transcriptions often being scarce and the majority of available subtitles provided only in Mandarin. To address this deficiency, we introduce TG-ASR for Taiwanese Hokkien drama speech recognition, a translation-guided ASR framework that utilizes multilingual translation embeddings to enhance recognition performance in low-resource environments. The framework is centered around the parallel gated cross-attention (PGCA) mechanism, which adaptively integrates embeddings from various auxiliary languages into the ASR decoder. This mechanism facilitates robust cross-linguistic semantic guidance while ensuring stable optimization and minimizing interference between languages. To support ongoing research initiatives, we present YT-THDC, a 30-hour corpus of Taiwanese Hokkien drama speech with aligned Mandarin subtitles and manually verified Taiwanese Hokkien transcriptions. Comprehensive experiments and analyses identify the auxiliary languages that most effectively enhance ASR performance, achieving a 14.77% relative reduction in character error rate and demonstrating the efficacy of translation-guided learning for underrepresented languages in practical applications.
>
---
#### [new 065] iMiGUE-Speech: A Spontaneous Speech Dataset for Affective Analysis
- **分类: eess.AS; cs.CL**

- **简介: 论文介绍iMiGUE-Speech数据集，用于情感分析任务，解决自发情绪研究问题。该数据集包含语音、转录文本及对齐信息，支持语音情感识别和基于文本的 sentiment 分析。**

- **链接: [https://arxiv.org/pdf/2602.21464v1](https://arxiv.org/pdf/2602.21464v1)**

> **作者:** Sofoklis Kakouros; Fang Kang; Haoyu Chen
>
> **备注:** Accepted to Speech Prosody 2026
>
> **摘要:** This work presents iMiGUE-Speech, an extension of the iMiGUE dataset that provides a spontaneous affective corpus for studying emotional and affective states. The new release focuses on speech and enriches the original dataset with additional metadata, including speech transcripts, speaker-role separation between interviewer and interviewee, and word-level forced alignments. Unlike existing emotional speech datasets that rely on acted or laboratory-elicited emotions, iMiGUE-Speech captures spontaneous affect arising naturally from real match outcomes. To demonstrate the utility of the dataset and establish initial benchmarks, we introduce two evaluation tasks for comparative assessment: speech emotion recognition and transcript-based sentiment analysis. These tasks leverage state-of-the-art pre-trained representations to assess the dataset's ability to capture spontaneous affective states from both acoustic and linguistic modalities. iMiGUE-Speech can also be synchronously paired with micro-gesture annotations from the original iMiGUE dataset, forming a uniquely multimodal resource for studying speech-gesture affective dynamics. The extended dataset is available at https://github.com/CV-AC/imigue-speech.
>
---
#### [new 066] DynamicGTR: Leveraging Graph Topology Representation Preferences to Boost VLM Capabilities on Graph QAs
- **分类: cs.CV; cs.AI; cs.CL; cs.GR**

- **简介: 该论文属于视觉-语言模型在图问答任务中的研究，旨在解决VLM对结构化图理解不足的问题。提出DynamicGTR框架，动态选择最优图拓扑表示，提升问答准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.21864v1](https://arxiv.org/pdf/2602.21864v1)**

> **作者:** Yanbin Wei; Jiangyue Yan; Chun Kang; Yang Chen; Hua Liu; James Kwok; Yu Zhang
>
> **备注:** CVPR 2026
>
> **摘要:** Vision-Language Models (VLMs) have emerged as versatile solutions for zero-shot question answering (QA) across various domains. However, enabling VLMs to effectively comprehend structured graphs and perform accurate, efficient QA remains challenging. Existing approaches typically rely on one single graph topology representation (GTR), such as fixed-style visual images or unified text descriptions. This ``one-size-fits-all'' strategy often neglects model-specific and task-specific preferences, resulting in inaccurate or over-lengthy responses to graph-related queries. To address this, we propose the $\mbox{DynamicGTR}$ framework, which dynamically selects the optimal GTR for each query during inference, thereby enhancing the zero-shot graph QA capabilities of VLMs with a customizable accuracy and brevity trade-off. Extensive experiments show that DynamicGTR not only improves VLM-based graph algorithm QA performance but also successfully transfers the experience trained from synthetic graph algorithm tasks to real-world applications like link prediction and node classification, without any additional training. Additionally, DynamicGTR demonstrates strong transferability across tasks, domains, and models, suggesting its potential as a flexible solution for broad graph scenarios.
>
---
#### [new 067] GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出GradAlign，用于大语言模型强化学习中的数据选择，解决训练数据质量不稳定问题，通过梯度对齐选择有效样本，提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2602.21492v1](https://arxiv.org/pdf/2602.21492v1)**

> **作者:** Ningyuan Yang; Weihua Du; Weiwei Sun; Sean Welleck; Yiming Yang
>
> **备注:** 14 pages. Preliminary work
>
> **摘要:** Reinforcement learning (RL) has become a central post-training paradigm for large language models (LLMs), but its performance is highly sensitive to the quality of training problems. This sensitivity stems from the non-stationarity of RL: rollouts are generated by an evolving policy, and learning is shaped by exploration and reward feedback, unlike supervised fine-tuning (SFT) with fixed trajectories. As a result, prior work often relies on manual curation or simple heuristic filters (e.g., accuracy), which can admit incorrect or low-utility problems. We propose GradAlign, a gradient-aligned data selection method for LLM reinforcement learning that uses a small, trusted validation set to prioritize training problems whose policy gradients align with validation gradients, yielding an adaptive curriculum. We evaluate GradAlign across three challenging data regimes: unreliable reward signals, distribution imbalance, and low-utility training corpus, showing that GradAlign consistently outperforms existing baselines, underscoring the importance of directional gradient signals in navigating non-stationary policy optimization and yielding more stable training and improved final performance. We release our implementation at https://github.com/StigLidu/GradAlign
>
---
#### [new 068] NoLan: Mitigating Object Hallucinations in Large Vision-Language Models via Dynamic Suppression of Language Priors
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决对象幻觉问题。通过实验发现语言解码器的先验知识是主要原因，并提出NoLan框架动态抑制语言先验以减少幻觉。**

- **链接: [https://arxiv.org/pdf/2602.22144v1](https://arxiv.org/pdf/2602.22144v1)**

> **作者:** Lingfeng Ren; Weihao Yu; Runpeng Yu; Xinchao Wang
>
> **备注:** Code: https://github.com/lingfengren/NoLan
>
> **摘要:** Object hallucination is a critical issue in Large Vision-Language Models (LVLMs), where outputs include objects that do not appear in the input image. A natural question arises from this phenomenon: Which component of the LVLM pipeline primarily contributes to object hallucinations? The vision encoder to perceive visual information, or the language decoder to generate text responses? In this work, we strive to answer this question through designing a systematic experiment to analyze the roles of the vision encoder and the language decoder in hallucination generation. Our observations reveal that object hallucinations are predominantly associated with the strong priors from the language decoder. Based on this finding, we propose a simple and training-free framework, No-Language-Hallucination Decoding, NoLan, which refines the output distribution by dynamically suppressing language priors, modulated based on the output distribution difference between multimodal and text-only inputs. Experimental results demonstrate that NoLan effectively reduces object hallucinations across various LVLMs on different tasks. For instance, NoLan achieves substantial improvements on POPE, enhancing the accuracy of LLaVA-1.5 7B and Qwen-VL 7B by up to 6.45 and 7.21, respectively. The code is publicly available at: https://github.com/lingfengren/NoLan.
>
---
#### [new 069] ACAR: Adaptive Complexity Routing for Multi-Model Ensembles with Auditable Decision Traces
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ACAR，一种用于多模型集成的自适应路由框架，解决任务分配与模型协作问题。通过分析自一致性方差，实现高效模型选择，提升准确性并减少计算开销。**

- **链接: [https://arxiv.org/pdf/2602.21231v1](https://arxiv.org/pdf/2602.21231v1)**

> **作者:** Ramchand Kumaresan
>
> **备注:** 12 pages, 9 figures. Measurement framework for adaptive multi-model routing with auditable execution traces
>
> **摘要:** We present ACAR (Adaptive Complexity and Attribution Routing), a measurement framework for studying multi-model orchestration under auditable conditions. ACAR uses self-consistency variance (sigma) computed from N=3 probe samples to route tasks across single-model, two-model, and three-model execution modes. The system is implemented on top of TEAMLLM, a deterministic execution substrate with immutable artifacts and complete decision traces. We evaluate ACAR on 1,510 tasks spanning four benchmarks: MathArena, Reasoning Gym, LiveCodeBench, and SuperGPQA, using Claude Sonnet 4, GPT-4o, and Gemini 2.0 Flash, producing more than 7,550 auditable runs. Results show that sigma-based routing achieves 55.6 percent accuracy, exceeding the two-model baseline of 54.4 percent while avoiding full ensembling on 54.2 percent of tasks. The routing mechanism is model-agnostic and requires no learned components. We also document negative results. First, retrieval augmentation reduced accuracy by 3.4 percentage points, as median retrieval similarity was only 0.167, demonstrating that experience injection without semantic alignment introduces noise rather than grounding. Second, when models agree on incorrect answers (sigma equals zero), no downstream ensemble can recover; this agreement-but-wrong failure mode is intrinsic to self-consistency and bounds achievable accuracy at approximately eight percentage points below full ensembling. Third, attribution estimates based on proxy signals such as response similarity and entropy showed weak correlation with ground-truth leave-one-out values, indicating that practical attribution requires explicit counterfactual computation. This work documents which assumptions fail in practice and provides falsifiable baselines for future research on routing, retrieval, and multi-model attribution.
>
---
#### [new 070] SWE-Protégé: Learning to Selectively Collaborate With an Expert Unlocks Small Language Models as Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于软件工程任务，旨在解决小语言模型在长周期任务中的行动循环和低修复率问题。通过引入SWE-Protégé框架，使模型学会选择性地与专家协作，提升性能。**

- **链接: [https://arxiv.org/pdf/2602.22124v1](https://arxiv.org/pdf/2602.22124v1)**

> **作者:** Patrick Tser Jern Kon; Archana Pradeep; Ang Chen; Alexander P. Ellis; Warren Hunt; Zijian Wang; John Yang; Samuel Thompson
>
> **摘要:** Small language models (SLMs) offer compelling advantages in cost, latency, and adaptability, but have so far lagged behind larger models on long-horizon software engineering tasks such as SWE-bench, where they suffer from pervasive action looping and low resolution rates. We introduce SWE-Protégé, a post-training framework that reframes software repair as an expert-protégé collaboration problem. In SWE-Protégé, an SLM remains the sole decision-maker while learning to selectively seek guidance from a strong expert model, recognize stalled states, and follow through on expert feedback. Our approach combines supervised fine-tuning on expert-augmented trajectories with agentic reinforcement learning that explicitly discourages degenerative looping and unproductive expert collaboration. We lightly post-train Qwen2.5-Coder-7B-Instruct to achieve 42.4% Pass@1 on SWE-bench Verified, a +25.4% improvement over the prior SLM state of the art, while using expert assistance sparsely (~4 calls per task and 11% of total tokens).
>
---
#### [new 071] When AI Writes, Whose Voice Remains? Quantifying Cultural Marker Erasure Across World English Varieties in Large Language Models
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文研究AI语言模型在处理不同英语变体时的文化标记消失问题，属于自然语言处理任务。旨在解决文化身份被系统性抹除的问题，通过实验分析并提出量化指标。**

- **链接: [https://arxiv.org/pdf/2602.22145v1](https://arxiv.org/pdf/2602.22145v1)**

> **作者:** Satyam Kumar Navneet; Joydeep Chandra; Yong Zhang
>
> **摘要:** Large Language Models (LLMs) are increasingly used to ``professionalize'' workplace communication, often at the cost of linguistic identity. We introduce "Cultural Ghosting", the systematic erasure of linguistic markers unique to non-native English varieties during text processing. Through analysis of 22,350 LLM outputs generated from 1,490 culturally marked texts (Indian, Singaporean,& Nigerian English) processed by five models under three prompt conditions, we quantify this phenomenon using two novel metrics: Identity Erasure Rate (IER) & Semantic Preservation Score (SPS). Across all prompts, we find an overall IER of 10.26%, with model-level variation from 3.5% to 20.5% (5.9x range). Crucially, we identify a Semantic Preservation Paradox: models maintain high semantic similarity (mean SPS = 0.748) while systematically erasing cultural markers. Pragmatic markers (politeness conventions) are 1.9x more vulnerable than lexical markers (71.5% vs. 37.1% erasure). Our experiments demonstrate that explicit cultural-preservation prompts reduce erasure by 29% without sacrificing semantic quality.
>
---
#### [new 072] GUI-Libra: Training Native GUI Agents to Reason and Act with Action-aware Supervision and Partially Verifiable RL
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于GUI代理训练任务，解决数据不足与部分可验证强化学习的问题。提出GUI-Libra，通过数据构建、动作感知微调和KL正则化提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2602.22190v1](https://arxiv.org/pdf/2602.22190v1)**

> **作者:** Rui Yang; Qianhui Wu; Zhaoyang Wang; Hanyang Chen; Ke Yang; Hao Cheng; Huaxiu Yao; Baoling Peng; Huan Zhang; Jianfeng Gao; Tong Zhang
>
> **备注:** 57 pages, 17 figures
>
> **摘要:** Open-source native GUI agents still lag behind closed-source systems on long-horizon navigation tasks. This gap stems from two limitations: a shortage of high-quality, action-aligned reasoning data, and the direct adoption of generic post-training pipelines that overlook the unique challenges of GUI agents. We identify two fundamental issues in these pipelines: (i) standard SFT with CoT reasoning often hurts grounding, and (ii) step-wise RLVR-tyle training faces partial verifiability, where multiple actions can be correct but only a single demonstrated action is used for verification. This makes offline step-wise metrics weak predictors of online task success. In this work, we present GUI-Libra, a tailored training recipe that addresses these challenges. First, to mitigate the scarcity of action-aligned reasoning data, we introduce a data construction and filtering pipeline and release a curated 81K GUI reasoning dataset. Second, to reconcile reasoning with grounding, we propose action-aware SFT that mixes reasoning-then-action and direct-action data and reweights tokens to emphasize action and grounding. Third, to stabilize RL under partial verifiability, we identify the overlooked importance of KL regularization in RLVR and show that a KL trust region is critical for improving offline-to-online predictability; we further introduce success-adaptive scaling to downweight unreliable negative gradients. Across diverse web and mobile benchmarks, GUI-Libra consistently improves both step-wise accuracy and end-to-end task completion. Our results suggest that carefully designed post-training and data curation can unlock significantly stronger task-solving capabilities without costly online data collection. We release our dataset, code, and models to facilitate further research on data-efficient post-training for reasoning-capable GUI agents.
>
---
#### [new 073] Distill and Align Decomposition for Enhanced Claim Verification
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于事实核查任务，旨在解决句子分解与验证对齐的问题。通过强化学习优化分解质量和验证一致性，提升验证效果。**

- **链接: [https://arxiv.org/pdf/2602.21857v1](https://arxiv.org/pdf/2602.21857v1)**

> **作者:** Jabez Magomere; Elena Kochkina; Samuel Mensah; Simerjot Kaur; Fernando Acero; Arturo Oncevay; Charese H. Smiley; Xiaomo Liu; Manuela Veloso
>
> **备注:** EACL Findings 2026
>
> **摘要:** Complex claim verification requires decomposing sentences into verifiable subclaims, yet existing methods struggle to align decomposition quality with verification performance. We propose a reinforcement learning (RL) approach that jointly optimizes decomposition quality and verifier alignment using Group Relative Policy Optimization (GRPO). Our method integrates: (i) structured sequential reasoning; (ii) supervised finetuning on teacher-distilled exemplars; and (iii) a multi-objective reward balancing format compliance, verifier alignment, and decomposition quality. Across six evaluation settings, our trained 8B decomposer improves downstream verification performance to (71.75%) macro-F1, outperforming prompt-based approaches ((+1.99), (+6.24)) and existing RL methods ((+5.84)). Human evaluation confirms the high quality of the generated subclaims. Our framework enables smaller language models to achieve state-of-the-art claim verification by jointly optimising for verification accuracy and decomposition quality.
>
---
#### [new 074] One Brain, Omni Modalities: Towards Unified Non-Invasive Brain Decoding with Large Language Models
- **分类: q-bio.NC; cs.AI; cs.CL**

- **简介: 该论文属于脑机接口任务，旨在统一非侵入式脑信号解码。解决多模态信号分析孤立的问题，通过NOBEL模型融合EEG/MEG与fMRI数据，提升解码准确性。**

- **链接: [https://arxiv.org/pdf/2602.21522v1](https://arxiv.org/pdf/2602.21522v1)**

> **作者:** Changli Tang; Shurui Li; Junliang Wang; Qinfan Xiao; Zhonghao Zhai; Lei Bai; Yu Qiao; Bowen Zhou; Wen Wu; Yuanning Li; Chao Zhang
>
> **摘要:** Deciphering brain function through non-invasive recordings requires synthesizing complementary high-frequency electromagnetic (EEG/MEG) and low-frequency metabolic (fMRI) signals. However, despite their shared neural origins, extreme discrepancies have traditionally confined these modalities to isolated analysis pipelines, hindering a holistic interpretation of brain activity. To bridge this fragmentation, we introduce \textbf{NOBEL}, a \textbf{n}euro-\textbf{o}mni-modal \textbf{b}rain-\textbf{e}ncoding \textbf{l}arge language model (LLM) that unifies these heterogeneous signals within the LLM's semantic embedding space. Our architecture integrates a unified encoder for EEG and MEG with a novel dual-path strategy for fMRI, aligning non-invasive brain signals and external sensory stimuli into a shared token space, then leverages an LLM as a universal backbone. Extensive evaluations demonstrate that NOBEL serves as a robust generalist across standard single-modal tasks. We also show that the synergistic fusion of electromagnetic and metabolic signals yields higher decoding accuracy than unimodal baselines, validating the complementary nature of multiple neural modalities. Furthermore, NOBEL exhibits strong capabilities in stimulus-aware decoding, effectively interpreting visual semantics from multi-subject fMRI data on the NSD and HAD datasets while uniquely leveraging direct stimulus inputs to verify causal links between sensory signals and neural responses. NOBEL thus takes a step towards unifying non-invasive brain decoding, demonstrating the promising potential of omni-modal brain understanding.
>
---
#### [new 075] Both Ends Count! Just How Good are LLM Agents at "Text-to-Big SQL"?
- **分类: cs.DB; cs.CL; cs.IR**

- **简介: 该论文研究"Text-to-Big SQL"任务，解决传统文本到SQL评估在大数据场景下的不足。通过引入新指标，评估LLM代理在大数据中的效率与成本表现。**

- **链接: [https://arxiv.org/pdf/2602.21480v1](https://arxiv.org/pdf/2602.21480v1)**

> **作者:** Germán T. Eizaguirre; Lars Tissen; Marc Sánchez-Artigas
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Text-to-SQL and Big Data are both extensively benchmarked fields, yet there is limited research that evaluates them jointly. In the real world, Text-to-SQL systems are often embedded with Big Data workflows, such as large-scale data processing or interactive data analytics. We refer to this as "Text-to-Big SQL". However, existing text-to-SQL benchmarks remain narrowly scoped and overlook the cost and performance implications that arise at scale. For instance, translation errors that are minor on small datasets lead to substantial cost and latency overheads as data scales, a relevant issue completely ignored by text-to-SQL metrics. In this paper, we overcome this overlooked challenge by introducing novel and representative metrics for evaluating Text-to-Big SQL. Our study focuses on production-level LLM agents, a database-agnostic system adaptable to diverse user needs. Via an extensive evaluation of frontier models, we show that text-to-SQL metrics are insufficient for Big Data. In contrast, our proposed text-to-Big SQL metrics accurately reflect execution efficiency, cost, and the impact of data scale. Furthermore, we provide LLM-specific insights, including fine-grained, cross-model comparisons of latency and cost.
>
---
#### [new 076] Adversarial Intent is a Latent Variable: Stateful Trust Inference for Securing Multimodal Agentic RAG
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全任务，解决多模态代理RAG中的对抗策略检测问题。通过构建状态感知的防御框架，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2602.21447v1](https://arxiv.org/pdf/2602.21447v1)**

> **作者:** Inderjeet Singh; Vikas Pahuja; Aishvariya Priya Rathina Sabapathy; Chiara Picardi; Amit Giloni; Roman Vainshtein; Andrés Murillo; Hisashi Kojima; Motoyoshi Sekiya; Yuki Unno; Junichi Suga
>
> **备注:** 13 pages, 2 figures, 5 tables
>
> **摘要:** Current stateless defences for multimodal agentic RAG fail to detect adversarial strategies that distribute malicious semantics across retrieval, planning, and generation components. We formulate this security challenge as a Partially Observable Markov Decision Process (POMDP), where adversarial intent is a latent variable inferred from noisy multi-stage observations. We introduce MMA-RAG^T, an inference-time control framework governed by a Modular Trust Agent (MTA) that maintains an approximate belief state via structured LLM reasoning. Operating as a model-agnostic overlay, MMA-RAGT mediates a configurable set of internal checkpoints to enforce stateful defence-in-depth. Extensive evaluation on 43,774 instances demonstrates a 6.50x average reduction factor in Attack Success Rate relative to undefended baselines, with negligible utility cost. Crucially, a factorial ablation validates our theoretical bounds: while statefulness and spatial coverage are individually necessary (26.4 pp and 13.6 pp gains respectively), stateless multi-point intervention can yield zero marginal benefit under homogeneous stateless filtering when checkpoint detections are perfectly correlated.
>
---
#### [new 077] Prompt Architecture Determines Reasoning Quality: A Variable Isolation Study on the Car Wash Problem
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究如何通过提示架构提升模型的推理质量，解决“汽车清洗问题”中的隐式约束推理难题。通过实验验证不同提示结构的效果。**

- **链接: [https://arxiv.org/pdf/2602.21814v1](https://arxiv.org/pdf/2602.21814v1)**

> **作者:** Heejin Jo
>
> **备注:** 9 pages, 4 tables
>
> **摘要:** Large language models consistently fail the "car wash problem," a viral reasoning benchmark requiring implicit physical constraint inference. We present a variable isolation study (n=20 per condition, 6 conditions, 120 total trials) examining which prompt architecture layers in a production system enable correct reasoning. Using Claude 3.5 Sonnet with controlled hyperparameters (temperature 0.7, top_p 1.0), we find that the STAR (Situation-Task-Action-Result) reasoning framework alone raises accuracy from 0% to 85% (p=0.001, Fisher's exact test, odds ratio 13.22). Adding user profile context via vector database retrieval provides a further 10 percentage point gain, while RAG context contributes an additional 5 percentage points, achieving 100% accuracy in the full-stack condition. These results suggest that structured reasoning scaffolds -- specifically, forced goal articulation before inference -- matter substantially more than context injection for implicit constraint reasoning tasks.
>
---
## 更新

#### [replaced 001] QueryPlot: Generating Geological Evidence Layers using Natural Language Queries for Mineral Exploration
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出QueryPlot，用于矿产勘探的地质证据层生成。任务是整合地质文本与空间数据，解决传统人工处理效率低的问题。通过自然语言查询和语义相似度计算，生成矿化潜力图层。**

- **链接: [https://arxiv.org/pdf/2602.17784v2](https://arxiv.org/pdf/2602.17784v2)**

> **作者:** Meng Ye; Xiao Lin; Georgina Lukoczki; Graham W. Lederer; Yi Yao
>
> **摘要:** Mineral prospectivity mapping requires synthesizing heterogeneous geological knowledge, including textual deposit models and geospatial datasets, to identify regions likely to host specific mineral deposit types. This process is traditionally manual and knowledge-intensive. We present QueryPlot, a semantic retrieval and mapping framework that integrates large-scale geological text corpora with geologic map data using modern Natural Language Processing techniques. We curate descriptive deposit models for over 120 deposit types and transform the State Geologic Map Compilation (SGMC) polygons into structured textual representations. Given a user-defined natural language query, the system encodes both queries and region descriptions using a pretrained embedding model and computes semantic similarity scores to rank and spatially visualize regions as continuous evidence layers. QueryPlot supports compositional querying over deposit characteristics, enabling aggregation of multiple similarity-derived layers for multi-criteria prospectivity analysis. In a case study on tungsten skarn deposits, we demonstrate that embedding-based retrieval achieves high recall of known occurrences and produces prospective regions that closely align with expert-defined permissive tracts. Furthermore, similarity scores can be incorporated as additional features in supervised learning pipelines, yielding measurable improvements in classification performance. QueryPlot is implemented as a web-based system supporting interactive querying, visualization, and export of GIS-compatible prospectivity layers.To support future research, we have made the source code and datasets used in this study publicly available.
>
---
#### [replaced 002] RPTS: Tree-Structured Reasoning Process Scoring for Faithful Multimodal Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决现有评估方法忽视推理过程的问题。提出RPTS评分机制，构建RPTS-Eval基准，评估模型推理准确性与缺陷。**

- **链接: [https://arxiv.org/pdf/2511.06899v3](https://arxiv.org/pdf/2511.06899v3)**

> **作者:** Haofeng Wang; Yu Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) excel in multimodal reasoning and have shown impressive performance on various multimodal benchmarks. However, most of these benchmarks evaluate models primarily through multiple-choice or short-answer formats, which do not take the reasoning process into account. Although some benchmarks assess the reasoning process, their methods are often overly simplistic and only examine reasoning when answers are incorrect. This approach overlooks scenarios where flawed reasoning leads to correct answers. In addition, these benchmarks do not consider the impact of intermodal relationships on reasoning. To address this issue, we propose the Reasoning Process Tree Score (RPTS), a tree structure-based metric to assess reasoning processes. Specifically, we organize the reasoning steps into a reasoning tree and leverage its hierarchical information to assign weighted faithfulness scores to each reasoning step. By dynamically adjusting these weights, RPTS not only evaluates the overall correctness of the reasoning, but also pinpoints where the model fails in the reasoning. To validate RPTS in real-world multimodal scenarios, we construct a new benchmark, RPTS-Eval, comprising 374 images and 390 reasoning instances. Each instance includes reliable visual-textual clues that serve as leaf nodes of the reasoning tree. Furthermore, we define three types of intermodal relationships to investigate how intermodal interactions influence the reasoning process. We evaluated representative LVLMs (e.g., GPT4o, Llava-Next), uncovering their limitations in multimodal reasoning and highlighting the differences between open-source and closed-source commercial LVLMs. We believe that this benchmark will contribute to the advancement of research in the field of multimodal reasoning.
>
---
#### [replaced 003] Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，研究LLMs是否遵循外部标签定义。通过实验分析不同定义条件下的表现，探讨其对任务解决的影响。**

- **链接: [https://arxiv.org/pdf/2509.02452v3](https://arxiv.org/pdf/2509.02452v3)**

> **作者:** Seyedali Mohammadi; Bhaskara Hanuma Vedula; Hemank Lamba; Edward Raff; Ponnurangam Kumaraguru; Francis Ferraro; Manas Gaur
>
> **备注:** EMNLP 2025 (Main Conference)
>
> **摘要:** Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities.
>
---
#### [replaced 004] FML-bench: Benchmarking Machine Learning Agents for Scientific Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出FML-bench，用于评估机器学习代理在科学研究中的能力。针对现有基准不足，设计8个任务并引入探索多样性指标，以更全面衡量代理的科研表现。**

- **链接: [https://arxiv.org/pdf/2510.10472v2](https://arxiv.org/pdf/2510.10472v2)**

> **作者:** Qiran Zou; Hou Hei Lam; Wenhao Zhao; Yiming Tang; Tingting Chen; Samson Yu; Tianyi Zhang; Chang Liu; Xiangyang Ji; Dianbo Liu
>
> **备注:** Our benchmark is available at: https://github.com/qrzou/FML-bench
>
> **摘要:** Large language models (LLMs) have sparked growing interest in machine learning research agents that can autonomously propose ideas and conduct experiments. However, existing benchmarks predominantly adopt an engineering-oriented perspective: they emphasize application-oriented tasks and evaluate primarily on final performance and computational cost, overlooking agents' research processes and limiting assessment of their capabilities in scientific research settings. To more comprehensively evaluate agents in scientific research settings, we introduce FML-bench, a benchmark comprising 8 diverse and fundamental ML research tasks, and further propose complementary metrics, notably Exploration Diversity, which quantifies the variance of proposals across iterations and reveals how exploration patterns influence research outcomes. We evaluate state-of-the-art research agents on FML-bench, showing that agents employing broad exploration strategies exhibit higher exploration diversity and achieve superior performance, and that exploration diversity positively correlates with performance improvements across multiple tasks. We hope these findings and our benchmark inform future agent design and support the community in further investigating agent behavior. Our benchmark is available at https://github.com/qrzou/FML-bench.
>
---
#### [replaced 005] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长序列推理任务，旨在解决大模型在长文本推理中的计算复杂度高、受限于上下文长度等问题。提出InftyThink方法，通过迭代推理与摘要机制实现高效长推理。**

- **链接: [https://arxiv.org/pdf/2503.06692v5](https://arxiv.org/pdf/2503.06692v5)**

> **作者:** Yuchen Yan; Yongliang Shen; Yang Liu; Jin Jiang; Mengdi Zhang; Jian Shao; Yueting Zhuang
>
> **备注:** ICLR 2026: https://openreview.net/forum?id=T1h5em349L Project Page: https://zju-real.github.io/InftyThink Code: https://github.com/ZJU-REAL/InftyThink
>
> **摘要:** Advanced reasoning in large language models has achieved remarkable performance on challenging tasks, but the prevailing long-context reasoning paradigm faces critical limitations: quadratic computational scaling with sequence length, reasoning constrained by maximum context boundaries, and performance degradation beyond pre-training context windows. Existing approaches primarily compress reasoning chains without addressing the fundamental scaling problem. To overcome these challenges, we introduce InftyThink, a paradigm that transforms monolithic reasoning into an iterative process with intermediate summarization. By interleaving short reasoning segments with concise progress summaries, our approach enables unbounded reasoning depth while maintaining bounded computational costs. This creates a characteristic sawtooth memory pattern that significantly reduces computational complexity compared to traditional approaches. Furthermore, we develop a methodology for reconstructing long-context reasoning datasets into our iterative format, transforming OpenR1-Math into 333K training instances. Experiments across multiple model architectures demonstrate that our approach reduces computational costs while improving performance, with Qwen2.5-Math-7B showing 3-11% improvements across MATH500, AIME24, and GPQA_diamond benchmarks. Our work challenges the assumed trade-off between reasoning depth and computational efficiency, providing a more scalable approach to complex reasoning without architectural modifications.
>
---
#### [replaced 006] RebuttalAgent: Strategic Persuasion in Academic Rebuttal via Theory of Mind
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于学术反驳任务，旨在解决AI在复杂战略沟通中的不足。提出RebuttalAgent框架，结合心智理论进行有效说服，提升反驳效果。**

- **链接: [https://arxiv.org/pdf/2601.15715v3](https://arxiv.org/pdf/2601.15715v3)**

> **作者:** Zhitao He; Zongwei Lyu; Yi R Fung
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) framework that models reviewer mental state, formulates persuasion strategy, and generates evidence-based response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations.
>
---
#### [replaced 007] SpecMind: Cognitively Inspired, Interactive Multi-Turn Framework for Postcondition Inference
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SpecMind，解决后置条件生成任务中的准确性不足问题。通过多轮交互式 prompting 提升生成结果的准确性和完整性。**

- **链接: [https://arxiv.org/pdf/2602.20610v2](https://arxiv.org/pdf/2602.20610v2)**

> **作者:** Cuong Chi Le; Minh V. T Pham; Tung Vu Duy; Cuong Duc Van; Huy N. Phan; Hoang N. Phan; Tien N. Nguyen
>
> **摘要:** Specifications are vital for ensuring program correctness, yet writing them manually remains challenging and time-intensive. Recent large language model (LLM)-based methods have shown successes in generating specifications such as postconditions, but existing single-pass prompting often yields inaccurate results. In this paper, we present SpecMind, a novel framework for postcondition generation that treats LLMs as interactive and exploratory reasoners rather than one-shot generators. SpecMind employs feedback-driven multi-turn prompting approaches, enabling the model to iteratively refine candidate postconditions by incorporating implicit and explicit correctness feedback, while autonomously deciding when to stop. This process fosters deeper code comprehension and improves alignment with true program behavior via exploratory attempts. Our empirical evaluation shows that SpecMind significantly outperforms state-of-the-art approaches in both accuracy and completeness of generated postconditions.
>
---
#### [replaced 008] Small Reward Models via Backward Inference
- **分类: cs.CL**

- **简介: 该论文提出FLIP方法，解决奖励模型构建中的参考与评分依赖问题。通过反向推理生成指令，提升小模型奖励建模效果，适用于非验证领域。**

- **链接: [https://arxiv.org/pdf/2602.13551v2](https://arxiv.org/pdf/2602.13551v2)**

> **作者:** Yike Wang; Faeze Brahman; Shangbin Feng; Teng Xiao; Hannaneh Hajishirzi; Yulia Tsvetkov
>
> **摘要:** Reward models (RMs) play a central role throughout the language model (LM) pipeline, particularly in non-verifiable domains. However, the dominant LLM-as-a-Judge paradigm relies on the strong reasoning capabilities of large models, while alternative approaches require reference responses or explicit rubrics, limiting flexibility and broader accessibility. In this work, we propose FLIP (FLipped Inference for Prompt reconstruction), a reference-free and rubric-free reward modeling approach that reformulates reward modeling through backward inference: inferring the instruction that would most plausibly produce a given response. The similarity between the inferred and the original instructions is then used as the reward signal. Evaluations across four domains using 13 small language models show that FLIP outperforms LLM-as-a-Judge baselines by an average of 79.6%. Moreover, FLIP substantially improves downstream performance in extrinsic evaluations under test-time scaling via parallel sampling and GRPO training. We further find that FLIP is particularly effective for longer outputs and robust to common forms of reward hacking. By explicitly exploiting the validation-generation gap, FLIP enables reliable reward modeling in downscaled regimes where judgment methods fail. Code available at https://github.com/yikee/FLIP.
>
---
#### [replaced 009] BARREL: Boundary-Aware Reasoning for Factual and Reliable LRMs
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于事实性推理任务，旨在解决LRMs过度自信导致的不可靠问题。提出BARREL框架，提升模型的准确性和可靠性。**

- **链接: [https://arxiv.org/pdf/2505.13529v2](https://arxiv.org/pdf/2505.13529v2)**

> **作者:** Junxiao Yang; Jinzhe Tu; Haoran Liu; Xiaoce Wang; Chujie Zheng; Zhexin Zhang; Shiyao Cui; Caishun Chen; Tiantian He; Hongning Wang; Yew-Soon Ong; Minlie Huang
>
> **摘要:** Recent advances in Large Reasoning Models (LRMs) have shown impressive capabilities in mathematical and logical reasoning. However, current LRMs rarely admit ignorance or respond with "I don't know". Instead, they often produce incorrect answers while showing undue confidence, raising concerns about their factual reliability. In this work, we identify two pathological reasoning patterns characterized by overthinking that contribute to the overconfident and incorrect answers: last-minute guessing and second-thought spiraling. To address these issues, we propose BARREL-a novel framework that promotes concise and boundary-aware factual reasoning. Our experiments show that BARREL-training increases the reliability of DeepSeek-R1-Distill-Llama-8B from 39.33% to 61.48%, while still achieving accuracy comparable to models finetuned on reasoning data generated by R1. These results demonstrate that our pilot study is inspiring to build more reliable and factual System 2 LRMs.
>
---
#### [replaced 010] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型的推理能力。通过填充中间步骤的方法扩展推理过程，解决训练数据质量不足的问题。**

- **链接: [https://arxiv.org/pdf/2502.11684v3](https://arxiv.org/pdf/2502.11684v3)**

> **作者:** Yuchen Yan; Yongliang Shen; Yang Liu; Jin Jiang; Xin Xu; Mengdi Zhang; Jian Shao; Yueting Zhuang
>
> **备注:** ICLR 2026: https://openreview.net/forum?id=14i2wzPPfn
>
> **摘要:** Mathematical reasoning represents a critical frontier in advancing large language models (LLMs). While step-by-step approaches have emerged as the dominant paradigm for mathematical problem-solving in LLMs, the quality of reasoning steps in training data fundamentally constrains the performance of the models. Recent studies have demonstrated that more detailed intermediate steps can enhance model performance, yet existing methods for step expansion either require more powerful external models or incur substantial computational costs. In this paper, we introduce MathFimer, a novel framework for mathematical reasoning step expansion inspired by the ''Fill-in-the-middle'' task from code reasoning. By decomposing solution chains into prefix-suffix pairs and training models to reconstruct missing intermediate steps, we develop a specialized model, MathFimer-7B, on our carefully curated NuminaMath-FIM dataset. We then apply these models to enhance existing mathematical reasoning datasets by inserting detailed intermediate steps into their solution chains, creating MathFimer-expanded versions. Through comprehensive experiments on multiple mathematical reasoning datasets, including MathInstruct, MetaMathQA and etc., we demonstrate that models trained on MathFimer-expanded data consistently outperform their counterparts trained on original data across various benchmarks such as GSM8K and MATH. Our approach offers a practical, scalable solution for enhancing mathematical reasoning capabilities in LLMs without relying on powerful external models or expensive inference procedures.
>
---
#### [replaced 011] DOTResize: Reducing LLM Width via Discrete Optimal Transport-based Neuron Merging
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决LLM宽度缩减问题。通过引入最优传输理论，提出DOTResize方法，实现神经元宽度的重新投影与信息重组。**

- **链接: [https://arxiv.org/pdf/2507.04517v2](https://arxiv.org/pdf/2507.04517v2)**

> **作者:** Neha Verma; Kenton Murray; Kevin Duh
>
> **摘要:** Structured pruning methods designed for Large Language Models (LLMs) generally focus on identifying and removing the least important components to optimize model size. However, in this work, we question this prevalent approach by instead exploring how to recombine information from structures designated for pruning back into the reduced model. We specifically focus on neuron width reduction, and frame this problem as a Discrete Optimal Transport problem, and propose DOTResize, a novel Transformer compression method that uses optimal transport theory to transform and compress model width. To ensure applicability within the Transformer architecture, we motivate and incorporate necessary entropic regularization and matrix factorization techniques into the transportation maps produced by our method. Unlike pruning-based approaches which discard neurons based on importance measures, DOTResize re-projects the entire neuron width, allowing the retention and redistribution of useful signal across the reduced layer. Empirical results show that compared to simple or state-of-the-art neuron width-pruning techniques, DOTResize serves as a useful add-on to pruning, while achieving measurable reductions in real-world computational cost.
>
---
#### [replaced 012] From Raw Corpora to Domain Benchmarks: Automated Evaluation of LLM Domain Expertise
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM领域知识评估问题。提出一种自动化流程，将原始文本转化为无偏基准，评估模型的领域知识。**

- **链接: [https://arxiv.org/pdf/2506.07658v2](https://arxiv.org/pdf/2506.07658v2)**

> **作者:** Nitin Sharma; Thomas Wolfers; Çağatay Yıldız
>
> **备注:** 35 pages, 24 figures. Second version
>
> **摘要:** Accurate domain-specific benchmarking of LLMs is essential, specifically in domains with direct implications for humans, such as law, healthcare, and education. However, existing benchmarks are documented to be contaminated and are based on multiple choice questions, which suffer from inherent biases. To measure domain-specific knowledge in LLMs, we present a deterministic pipeline that transforms raw domain corpora into completion-style benchmarks without relying on other LLMs or costly human annotation. Our approach first extracts domain-specific keywords and related target vocabulary from an input corpus. It then constructs prompt-target pairs where domain-specific words serve as prediction targets. By measuring LLMs' ability to complete these prompts, we provide a direct assessment of domain knowledge at low computational cost. Our pipeline avoids benchmark contamination, enables automated updates with new domain data, and facilitates fair comparisons between base and instruction-tuned (chat) models. We validate our approach by showing that model performances on our benchmark significantly correlate with those on an expert-curated benchmark. We then demonstrate how our benchmark provides insights into knowledge acquisition in domain-adaptive, continual, and general pretraining. Finally, we examine the effects of instruction fine-tuning by comparing base and chat models within our unified evaluation framework. In conclusion, our pipeline enables scalable, domain-specific, LLM-independent, and unbiased evaluation of both base and chat models.
>
---
#### [replaced 013] Beyond RAG for Agent Memory: Retrieval by Decoupling and Aggregation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统中的记忆管理任务，旨在解决RAG在代理记忆中因冗余和重复导致的检索效率问题。提出xMemory方法，通过解耦与聚合实现更高效的检索。**

- **链接: [https://arxiv.org/pdf/2602.02007v2](https://arxiv.org/pdf/2602.02007v2)**

> **作者:** Zhanghao Hu; Qinglin Zhu; Hanqi Yan; Yulan He; Lin Gui
>
> **备注:** Project Address: https://zhanghao-xmemory.github.io/Academic-project-page-template/
>
> **摘要:** Agent memory systems often adopt the standard Retrieval-Augmented Generation (RAG) pipeline, yet its underlying assumptions differ in this setting. RAG targets large, heterogeneous corpora where retrieved passages are diverse, whereas agent memory is a bounded, coherent dialogue stream with highly correlated spans that are often duplicates. Under this shift, fixed top-$k$ similarity retrieval tends to return redundant context, and post-hoc pruning can delete temporally linked prerequisites needed for correct reasoning. We argue retrieval should move beyond similarity matching and instead operate over latent components, following decoupling to aggregation: disentangle memories into semantic components, organise them into a hierarchy, and use this structure to drive retrieval. We propose xMemory, which builds a hierarchy of intact units and maintains a searchable yet faithful high-level node organisation via a sparsity--semantics objective that guides memory split and merge. At inference, xMemory retrieves top-down, selecting a compact, diverse set of themes and semantics for multi-fact queries, and expanding to episodes and raw messages only when it reduces the reader's uncertainty. Experiments on LoCoMo and PerLTQA across the three latest LLMs show consistent gains in answer quality and token efficiency.
>
---
#### [replaced 014] Meenz bleibt Meenz, but Large Language Models Do Not Speak Its Dialect
- **分类: cs.CL**

- **简介: 该论文属于语言保护任务，旨在研究大语言模型对美因茨方言的处理能力。通过构建词典数据集，实验发现模型在生成定义和词汇方面效果不佳，表明需更多资源与研究支持方言保护。**

- **链接: [https://arxiv.org/pdf/2602.16852v2](https://arxiv.org/pdf/2602.16852v2)**

> **作者:** Minh Duc Bui; Manuel Mager; Peter Herbert Kann; Katharina von der Wense
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Meenzerisch, the dialect spoken in the German city of Mainz, is also the traditional language of the Mainz carnival, a yearly celebration well known throughout Germany. However, Meenzerisch is on the verge of dying out-a fate it shares with many other German dialects. Natural language processing (NLP) has the potential to help with the preservation and revival efforts of languages and dialects. However, so far no NLP research has looked at Meenzerisch. This work presents the first research in the field of NLP that is explicitly focused on the dialect of Mainz. We introduce a digital dictionary-an NLP-ready dataset derived from an existing resource (Schramm, 1966)-to support researchers in modeling and benchmarking the language. It contains 2,351 words in the dialect paired with their meanings described in Standard German. We then use this dataset to answer the following research questions: (1) Can state-of-the-art large language models (LLMs) generate definitions for dialect words? (2) Can LLMs generate words in Meenzerisch, given their definitions? Our experiments show that LLMs can do neither: the best model for definitions reaches only 6.27% accuracy and the best word generation model's accuracy is 1.51%. We then conduct two additional experiments in order to see if accuracy is improved by few-shot learning and by extracting rules from the training set, which are then passed to the LLM. While those approaches are able to improve the results, accuracy remains below 10%. This highlights that additional resources and an intensification of research efforts focused on German dialects are desperately needed.
>
---
#### [replaced 015] Document Reconstruction Unlocks Scalable Long-Context RLVR
- **分类: cs.CL**

- **简介: 该论文属于长文本理解任务，旨在提升大语言模型的长上下文能力。通过无监督的文档重建方法，无需人工标注或教师模型，显著提升了模型的长文本处理性能。**

- **链接: [https://arxiv.org/pdf/2602.08237v2](https://arxiv.org/pdf/2602.08237v2)**

> **作者:** Yao Xiao; Lei Wang; Yue Deng; Guanzheng Chen; Ziqi Jin; Jung-jae Kim; Xiaoli Li; Roy Ka-wei Lee; Lidong Bing
>
> **摘要:** Reinforcement Learning with Verifiable Rewards~(RLVR) has become a prominent paradigm to enhance the capabilities (i.e.\ long-context) of Large Language Models~(LLMs). However, it often relies on gold-standard answers or explicit evaluation rubrics provided by powerful teacher models or human experts, which are costly and time-consuming. In this work, we investigate unsupervised approaches to enhance the long-context capabilities of LLMs, eliminating the need for heavy human annotations or teacher models' supervision. Specifically, we first replace a few paragraphs with special placeholders in a long document. LLMs are trained through reinforcement learning to reconstruct the document by correctly identifying and sequencing missing paragraphs from a set of candidate options. This training paradigm enables the model to capture global narrative coherence, significantly boosting long-context performance. We validate the effectiveness of our method on two widely used benchmarks, RULER and LongBench~v2. While acquiring noticeable gains on RULER, it can also achieve a reasonable improvement on LongBench~v2 without any manually curated long-context QA data. Furthermore, we conduct extensive ablation studies to analyze the impact of reward design, data curation strategies, training schemes, and data scaling effects on model performance. We publicly release our code, data, and models.
>
---
#### [replaced 016] Diffusion Language Models Know the Answer Before Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决DLM推理速度慢的问题。通过发现早期答案收敛现象，提出Prophet方法实现快速解码，减少步骤数并保持质量。**

- **链接: [https://arxiv.org/pdf/2508.19982v4](https://arxiv.org/pdf/2508.19982v4)**

> **作者:** Pengxiang Li; Yefan Zhou; Dilxat Muhtar; Lu Yin; Shilin Yan; Li Shen; Soroush Vosoughi; Shiwei Liu
>
> **摘要:** Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high quality outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer convergence: in many cases, the correct answer can be internally identified by half steps before the final decoding step, both under semi-autoregressive and random remasking schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can be decoded correctly using only half of the refinement steps. Building on this observation, we introduce Prophet, a training-free fast decoding paradigm that enables early commit decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training. Empirical evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4x while preserving high generation quality. These results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early decode convergence provides a simple yet powerful mechanism for accelerating DLM inference, complementary to existing speedup techniques. Our code is publicly available at https://github.com/pixeli99/Prophet.
>
---
#### [replaced 017] Annotation-Efficient Universal Honesty Alignment
- **分类: cs.CL**

- **简介: 该论文属于大语言模型的可信部署任务，旨在解决模型自信与知识边界不匹配的问题。提出EliCal框架，通过少量标注实现高效校准，提升模型诚实对齐效果。**

- **链接: [https://arxiv.org/pdf/2510.17509v2](https://arxiv.org/pdf/2510.17509v2)**

> **作者:** Shiyu Ni; Keping Bi; Jiafeng Guo; Minghao Tang; Jingtong Wu; Zengxin Han; Xueqi Cheng
>
> **备注:** ICLR 2026
>
> **摘要:** Honesty alignment-the ability of large language models (LLMs) to recognize their knowledge boundaries and express calibrated confidence-is essential for trustworthy deployment. Existing methods either rely on training-free confidence estimation (e.g., token probabilities, self-consistency) or training-based calibration with correctness annotations. While effective, achieving universal honesty alignment with training-based calibration requires costly, large-scale labeling. To support annotation-efficient training, we introduce Elicitation-Then-Calibration (EliCal), a two-stage framework that first elicits internal confidence using inexpensive self-consistency supervision, then calibrates this confidence with a small set of correctness annotations. To support a large-scale study, we release HonestyBench, a benchmark covering ten free-form QA datasets with 560k training and 70k evaluation instances annotated with correctness and self-consistency signals. Experiments show that EliCal achieves near-optimal alignment with only 1k correctness annotations (0.18% of full supervision) and better alignment performance on unseen MMLU tasks than the calibration-only baseline, offering a scalable solution toward universal honesty alignment in LLMs.
>
---
#### [replaced 018] When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于安全防护任务，解决LLMs因风格对齐导致的安全漏洞问题。通过分析风格模式影响，提出SafeStyle防御策略提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2506.07452v3](https://arxiv.org/pdf/2506.07452v3)**

> **作者:** Yuxin Xiao; Sana Tonekaboni; Walter Gerych; Vinith Suriyakumar; Marzyeh Ghassemi
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 36 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.
>
---
#### [replaced 019] Diversity Boosts AI-Generated Text Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI生成文本检测任务，旨在解决高质文本难以检测的问题。提出DivEye框架，通过 surprisal 特征捕捉文本不可预测性差异，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2509.18880v3](https://arxiv.org/pdf/2509.18880v3)**

> **作者:** Advik Raj Basani; Pin-Yu Chen
>
> **备注:** Accepted to Transactions on Machine Learning Research (TMLR '26). Project page and demos: https://diveye.vercel.app/
>
> **摘要:** Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection.
>
---
#### [replaced 020] A Proof of Learning Rate Transfer under $μ$P
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于深度学习理论任务，解决学习率迁移问题。研究证明在μP参数化下，宽度增加时最优学习率收敛至非零常数，解释了学习率迁移现象。**

- **链接: [https://arxiv.org/pdf/2511.01734v3](https://arxiv.org/pdf/2511.01734v3)**

> **作者:** Soufiane Hayou
>
> **备注:** 21 pages
>
> **摘要:** We provide the first proof of learning rate transfer with width in a linear multi-layer perceptron (MLP) parametrized with $μ$P, a neural network parameterization designed to ``maximize'' feature learning in the infinite-width limit. We show that under $μP$, the optimal learning rate converges to a \emph{non-zero constant} as width goes to infinity, providing a theoretical explanation to learning rate transfer. In contrast, we show that this property fails to hold under alternative parametrizations such as Standard Parametrization (SP) and Neural Tangent Parametrization (NTP). We provide intuitive proofs and support the theoretical findings with extensive empirical results.
>
---
#### [replaced 021] ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Finetuning, and Decoding the Curse of Multilinguality
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言模型的扩展规律，解决多语言训练与迁移中的性能问题。通过大量实验，提出ATLAS方法，优化模型规模与数据配置，提升跨语言迁移效果。**

- **链接: [https://arxiv.org/pdf/2510.22037v2](https://arxiv.org/pdf/2510.22037v2)**

> **作者:** Shayne Longpre; Sneha Kudugunta; Niklas Muennighoff; I-Hung Hsu; Isaac Caswell; Alex Pentland; Sercan Arik; Chen-Yu Lee; Sayna Ebrahimi
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Scaling laws research has focused overwhelmingly on English -- yet the most prominent AI models explicitly serve billions of international users. In this work, we undertake the largest multilingual scaling laws study to date, totaling 774 multilingual training experiments, spanning 10M-8B model parameters, 400+ training languages and 48 evaluation languages. We introduce the Adaptive Transfer Scaling Law (ATLAS) for both monolingual and multilingual pretraining, which outperforms existing scaling laws' out-of-sample generalization often by more than 0.3 R^2. Our analyses of the experiments shed light on multilingual learning dynamics, transfer properties between languages, and the curse of multilinguality. First, we derive a cross-lingual transfer matrix, empirically measuring mutual benefit scores between 38 x 38=1444 language pairs. Second, we derive a language-agnostic scaling law that reveals how to optimally scale model size and data when adding languages without sacrificing performance. Third, we identify the computational crossover points for when to pretrain from scratch versus finetune from multilingual checkpoints. We hope these findings provide the scientific foundation for democratizing scaling laws across languages, and enable practitioners to efficiently scale models -- beyond English-first AI.
>
---
#### [replaced 022] Compose and Fuse: Revisiting the Foundational Bottlenecks in Multimodal Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态推理任务，旨在解决模态融合效果不稳定的问题。通过分析模态交互模式，发现集成不足是主要障碍，并提出改进方法。**

- **链接: [https://arxiv.org/pdf/2509.23744v2](https://arxiv.org/pdf/2509.23744v2)**

> **作者:** Yucheng Wang; Yifan Hou; Aydin Javadov; Mubashara Akhtar; Mrinmaya Sachan
>
> **备注:** Our code (https://github.com/DELTA-DoubleWise/OmniReason) and data (https://huggingface.co/datasets/ycwang11/OmniReason) are publicly available
>
> **摘要:** Multimodal large language models (MLLMs) promise enhanced reasoning by integrating diverse inputs such as text, vision, and audio. Yet cross-modal reasoning remains underexplored, with conflicting reports on whether added modalities help or harm performance. These inconsistencies stem from a lack of controlled evaluation frameworks and analysis of models' internals to isolate when and why modality interactions support or undermine reasoning. We address this gap through a logic-grounded evaluation framework that categorizes multimodal reasoning into six interaction patterns, varying how facts are distributed across modalities and logically combined. Empirically, additional modalities enhance reasoning only when they provide independent and sufficient reasoning paths, while redundant or chained entailment support often hurts performance. Moreover, reasoning degrades in three systematic ways: weaker modalities drag down overall performance, conflicts bias preference toward certain modalities, and joint signals from different modalities fail to be integrated effectively. Therefore, we identify two core failures: task-composition bottleneck, where recognition and reasoning cannot be jointly executed in one pass, and fusion bottleneck, where early integration introduces bias. For further investigation, we find that attention patterns fail to encode fact usefulness, but a simple two-step prompting (recognize then reason) restores performance, confirming the task-composition bottleneck. Moreover, modality identity remains recoverable in early layers, and softening attention in early fusion improves reasoning, highlighting biased fusion as another failure mode. Overall, our findings show that integration, not perception, is the main barrier to multimodal reasoning, suggesting composition-aware training and early fusion control as promising directions.
>
---
#### [replaced 023] Slm-mux: Orchestrating small language models for reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模型协同任务，旨在提升小语言模型（SLMs）的推理能力。通过提出SLM-MUX架构及优化策略，有效整合多个SLMs，显著提高准确率。**

- **链接: [https://arxiv.org/pdf/2510.05077v2](https://arxiv.org/pdf/2510.05077v2)**

> **作者:** Chenyu Wang; Zishen Wan; Hao Kang; Emma Chen; Zhiqiang Xie; Tushar Krishna; Vijay Janapa Reddi; Yilun Du
>
> **摘要:** With the rapid development of language models, the number of small language models (SLMs) has grown significantly. Although they do not achieve state-of-the-art accuracy, they are more efficient and often excel at specific tasks. This raises a natural question: can multiple SLMs be orchestrated into a system where each contributes effectively, achieving higher accuracy than any individual model? Existing orchestration methods have primarily targeted frontier models (e.g., GPT-4) and perform suboptimally when applied to SLMs. To address this gap, we propose a three-stage approach for orchestrating SLMs. First, we introduce SLM-MUX, a multi-model architecture that effectively coordinates multiple SLMs. Building on this, we develop two optimization strategies: (i) a model selection search that identifies the most complementary SLMs from a given pool, and (ii) test-time scaling tailored to SLM-MUX. Our approach delivers strong results: Compared to existing orchestration methods, our approach achieves up to 13.4% improvement on MATH, 8.8% on GPQA, and 7.0% on GSM8K. With just two SLMs, SLM-MUX outperforms Qwen 2.5 72B on GPQA and GSM8K, and matches its performance on MATH. We further provide theoretical analyses to substantiate the advantages of our method. Additional experiments show that the core principle of SLM-MUX extends to open-ended generation tasks (e.g., HumanEval) and benefits other model classes, including frontier LLMs and domain-specific fine-tuned SLMs. In summary, we demonstrate that SLMs can be effectively orchestrated into more accurate and efficient systems through the proposed approach. The project page is available at https://slm-mux.github.io/.
>
---
#### [replaced 024] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出VULCA-Bench，属于视觉-语言模型的文化理解评估任务，旨在解决现有基准对文化深层解读不足的问题。工作包括构建多文化图像-评论数据集及评估框架。**

- **链接: [https://arxiv.org/pdf/2601.07986v3](https://arxiv.org/pdf/2601.07986v3)**

> **作者:** Haorui Yu; Diji Yang; Hang He; Fengrui Zhang; Qiufeng Yi
>
> **备注:** 8 pages, 4 figures, submitted to ACL 2026 Dataset Track
>
> **摘要:** We introduce VULCA-Bench, a multicultural art-critique benchmark for evaluating Vision-Language Models' (VLMs) cultural understanding beyond surface-level visual perception. Existing VLM benchmarks predominantly measure L1-L2 capabilities (object recognition, scene description, and factual question answering) while under-evaluate higher-order cultural interpretation. VULCA-Bench contains 7,410 matched image-critique pairs spanning eight cultural traditions, with Chinese-English bilingual coverage. We operationalise cultural understanding using a five-layer framework (L1-L5, from Visual Perception to Philosophical Aesthetics), instantiated as 225 culture-specific dimensions and supported by expert-written bilingual critiques. Our pilot results indicate that higher-layer reasoning (L3-L5) is consistently more challenging than visual and technical analysis (L1-L2). The dataset, evaluation scripts, and annotation tools are available under CC BY 4.0 at https://github.com/yha9806/VULCA-Bench.
>
---
#### [replaced 025] Mechanistic Indicators of Understanding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI理解机制研究，旨在探讨大语言模型是否具备理解能力。通过构建分层框架，整合机制可解释性成果，分析模型不同层次的理解表现。**

- **链接: [https://arxiv.org/pdf/2507.08017v5](https://arxiv.org/pdf/2507.08017v5)**

> **作者:** Pierre Beckmann; Matthieu Queloz
>
> **备注:** 38 pages
>
> **摘要:** Large language models (LLMs) are often portrayed as merely imitating linguistic patterns without genuine understanding. We argue that recent findings in mechanistic interpretability (MI), the emerging field probing the inner workings of LLMs, render this picture increasingly untenable--but only once those findings are integrated within a theoretical account of understanding. We propose a tiered framework for thinking about understanding in LLMs and use it to synthesize the most relevant findings to date. The framework distinguishes three hierarchical varieties of understanding, each tied to a corresponding level of computational organization: conceptual understanding emerges when a model forms "features" as directions in latent space, learning connections between diverse manifestations of a single entity or property; state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world; principled understanding emerges when a model ceases to rely on memorized facts and discovers a compact "circuit" connecting these facts. Across these tiers, MI uncovers internal organizations that can underwrite understanding-like unification. However, these also diverge from human cognition in their parallel exploitation of heterogeneous mechanisms. Fusing philosophical theory with mechanistic evidence thus allows us to transcend binary debates over whether AI understands, paving the way for a comparative, mechanistically grounded epistemology that explores how AI understanding aligns with--and diverges from--our own.
>
---
#### [replaced 026] Spilled Energy in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文将大语言模型的softmax分类器重新解释为能量模型，通过分析能量溢出检测事实错误和幻觉，无需额外训练即可实现有效检测。**

- **链接: [https://arxiv.org/pdf/2602.18671v2](https://arxiv.org/pdf/2602.18671v2)**

> **作者:** Adrian Robert Minut; Hazem Dewidar; Iacopo Masi
>
> **摘要:** We reinterpret the final Large Language Model (LLM) softmax classifier as an Energy-Based Model (EBM), decomposing the sequence-to-sequence probability chain into multiple interacting EBMs at inference. This principled approach allows us to track "energy spills" during decoding, which we empirically show correlate with factual errors, biases, and failures. Similar to Orgad et al. (2025), our method localizes the exact answer token and subsequently tests for hallucinations. Crucially, however, we achieve this without requiring trained probe classifiers or activation ablations. Instead, we introduce two completely training-free metrics derived directly from output logits: spilled energy, which captures the discrepancy between energy values across consecutive generation steps that should theoretically match, and marginalized energy, which is measurable at a single step. Evaluated on nine benchmarks across state-of-the-art LLMs (including LLaMA, Mistral, and Gemma) and on synthetic algebraic operations (Qwen3), our approach demonstrates robust, competitive hallucination detection and cross-task generalization. Notably, these results hold for both pretrained and instruction-tuned variants without introducing any training overhead.
>
---
#### [replaced 027] PACE: Procedural Abstractions for Communicating Efficiently
- **分类: cs.CL**

- **简介: 该论文提出PACE方法，解决AI中抽象能力不足的问题。通过神经符号方法实现高效沟通，用于协作任务，促进语言效率提升。**

- **链接: [https://arxiv.org/pdf/2409.20120v4](https://arxiv.org/pdf/2409.20120v4)**

> **作者:** Jonathan D. Thomas; Andrea Silvi; Devdatt Dubhashi; Moa Johansson
>
> **备注:** Accepted to CogSci 2025 for presentation
>
> **摘要:** A central but unresolved aspect of problem-solving in AI is the capability to introduce and use abstractions, something humans excel at. Work in cognitive science has demonstrated that humans tend towards higher levels of abstraction when engaged in collaborative task-oriented communication, enabling gradually shorter and more information-efficient utterances. Several computational methods have attempted to replicate this phenomenon, but all make unrealistic simplifying assumptions about how abstractions are introduced and learned. Our method, Procedural Abstractions for Communicating Efficiently (PACE), overcomes these limitations through a neuro-symbolic approach. On the symbolic side, we draw on work from library learning for proposing abstractions. We combine this with neural methods for communication and reinforcement learning, via a novel use of bandit algorithms for controlling the exploration and exploitation trade-off in introducing new abstractions. PACE exhibits similar tendencies to humans on a collaborative construction task from the cognitive science literature, where one agent (the architect) instructs the other (the builder) to reconstruct a scene of block-buildings. PACE results in the emergence of an efficient language as a by-product of collaborative communication. Beyond providing mechanistic insights into human communication, our work serves as a first step to providing conversational agents with the ability for human-like communicative abstractions.
>
---
#### [replaced 028] Complexity counts: global and local perspectives on Indo-Aryan numeral systems
- **分类: physics.soc-ph; cs.CL**

- **简介: 该论文属于语言类型学研究，探讨印欧语系数词系统的复杂性。旨在分析印度-雅利安语言数词系统为何复杂，通过量化指标比较其与全球语言的差异，并探究影响因素。**

- **链接: [https://arxiv.org/pdf/2505.21510v2](https://arxiv.org/pdf/2505.21510v2)**

> **作者:** Chundra Cathcart
>
> **摘要:** The numeral systems of Indo-Aryan languages such as Hindi, Gujarati, and Bengali are highly unusual in that unlike most numeral systems (e.g., those of English, Chinese, etc.), forms referring to 1--99 are highly non-transparent and are cannot be constructed using straightforward rules for forming combinations of tens and digits. As an example, Hindi/Urdu {\it ikyānve} `91' is not decomposable into the composite elements {\it ek} `one' and {\it nave} `ninety' in the way that its English counterpart is. This paper further clarifies the position of Indo-Aryan languages within the typology of numeral systems, and explores the linguistic and non-linguistic factors that may be responsible for the persistence of complex systems in these languages. Using data from multiple databases, we develop and employ a number of cross-linguistically applicable metrics to quantifies the complexity of languages' numeral systems, and demonstrate that Indo-Aryan languages have decisively more complex numeral systems than the world's languages as a whole, though individual Indo-Aryan languages differ from each other in terms of the complexity of the patterns they display. We investigate the factors (e.g., religion, geographic isolation, etc.) that underlie complexity in numeral systems, with a focus on South Asia, in an attempt to develop an account of why complex numeral systems developed and persisted in certain Indo-Aryan languages but not elsewhere. Finally, we demonstrate that Indo-Aryan numeral systems adhere to certain general pressures toward efficient communication found cross-linguistically, despite their high complexity. We call for this somewhat overlooked dimension of complexity to be taken seriously when discussing general variation in numeral systems.
>
---
#### [replaced 029] Compressing Language Models for Specialized Domains
- **分类: cs.CL**

- **简介: 该论文属于语言模型压缩任务，旨在解决压缩后模型在专业领域性能下降的问题。提出MixCal方法，在不增加计算成本的情况下提升压缩模型的领域适应性。**

- **链接: [https://arxiv.org/pdf/2502.18424v2](https://arxiv.org/pdf/2502.18424v2)**

> **作者:** Miles Williams; George Chrysostomou; Vitor Jeronymo; Nikolaos Aletras
>
> **备注:** EACL 2026
>
> **摘要:** Language models (LMs) excel at tasks across diverse domains, yet require substantial computational resources during inference. Compression techniques such as pruning and quantization offer a practical path towards efficient LM deployment, exemplified by their ability to preserve performance on general-purpose benchmarks. However, general-purpose LM compression methods can negatively affect performance in specialized domains (e.g. biomedical or legal). Recent work has sought to address this issue, but requires a computationally expensive full-parameter fine-tuning pipeline. To this end, we propose MixCal, a novel calibration method designed to improve the in-domain performance of compressed LMs in a post-training setting. Through extensive experimentation, we demonstrate that MixCal substantially outperforms existing approaches on domain-specific tasks and preserves general performance. Notably, these performance gains are achieved while also reducing the computational cost of LM compression.
>
---
#### [replaced 030] PeruMedQA: Benchmarking Large Language Models (LLMs) on Peruvian Medical Exams -- Dataset Construction and Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗问答任务，旨在评估大语言模型在秘鲁医学考试中的表现。研究构建了PeruMedQA数据集，并对比了不同模型的准确率。**

- **链接: [https://arxiv.org/pdf/2509.11517v2](https://arxiv.org/pdf/2509.11517v2)**

> **作者:** Rodrigo M. Carrillo-Larco; Jesus Lovón Melgarejo; Manuel Castillo-Cara; Gusseppe Bravo-Rocca
>
> **备注:** https://github.com/rodrigo-carrillo/PeruMedQA
>
> **摘要:** BACKGROUND: Medical large language models (LLMs) have demonstrated remarkable performance in answering medical examinations. However, the extent to which this high performance is transferable to medical questions in Spanish and from a Latin American country remains unexplored. This knowledge is crucial as LLM-based medical applications gain traction in Latin America. AIMS: To build a dataset of questions medical examinations taken by Peruvian physicians pursuing specialty training; to fine-tune a LLM on this dataset; to evaluate and compare the performance in terms of accuracy between vanilla LLMs and the fine-tuned LLM. METHODS: We curated PeruMedQA, a multiple-choice question-answering (MCQA) dataset containing 8,380 questions spanning 12 specialties (2018-2025). We selected ten medical LLMs, including medgemma-4b-it and medgemma-27b-text-it, and developed zero-shot task specific prompts to answer the questions. We employed parameter-efficient fine tuning (PEFT) and low-rand adaptation (LoRA) to fine-tune medgemma-4b-it utilizing all questions except those from 2025 (test set). RESULTS: Medgemma-27b showed the highest accuracy across all specialities, achieving the highest score of 89.29% in Psychiatry; yet, in two specialties, OctoMed-7B exhibited slight superiority: Neurosurgery with 77.27% and 77.38, respectively; and Radiology with 76.13% and 77.39%, respectively. Across specialties, most LLMs with <10 billion parameters exhibited <50% of correct answers. The fine-tuned version of medgemma-4b-it emerged victorious against all LLMs with <10 billion parameters and rivaled a LLM with 70 billion parameters across various examinations. CONCLUSIONS: For medical AI applications and research that require knowledge bases from Spanish-speaking countries and those exhibiting similar epidemiological profile to Peru's, interested parties should utilize medgemma-27b-text-it.
>
---
#### [replaced 031] Refusal Direction is Universal Across Safety-Aligned Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的安全任务，研究多语言大模型的拒绝行为。通过跨语言实验，发现拒绝方向具有普遍性，为构建更 robust 的多语言安全防御提供依据。**

- **链接: [https://arxiv.org/pdf/2505.17306v2](https://arxiv.org/pdf/2505.17306v2)**

> **作者:** Xinpeng Wang; Mingyang Wang; Yihong Liu; Hinrich Schütze; Barbara Plank
>
> **摘要:** Refusal mechanisms in large language models (LLMs) are essential for ensuring safety. Recent research has revealed that refusal behavior can be mediated by a single direction in activation space, enabling targeted interventions to bypass refusals. While this is primarily demonstrated in an English-centric context, appropriate refusal behavior is important for any language, but poorly understood. In this paper, we investigate the refusal behavior in LLMs across 14 languages using PolyRefuse, a multilingual safety dataset created by translating malicious and benign English prompts into these languages. We uncover the surprising cross-lingual universality of the refusal direction: a vector extracted from English can bypass refusals in other languages with near-perfect effectiveness, without any additional fine-tuning. Even more remarkably, refusal directions derived from any safety-aligned language transfer seamlessly to others. We attribute this transferability to the parallelism of refusal vectors across languages in the embedding space and identify the underlying mechanism behind cross-lingual jailbreaks. These findings provide actionable insights for building more robust multilingual safety defenses and pave the way for a deeper mechanistic understanding of cross-lingual vulnerabilities in LLMs.
>
---
#### [replaced 032] Emergence of a phonological bias in ChatGPT
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨ChatGPT是否存在语音偏见。研究发现，ChatGPT表现出类似人类的辅音偏好，揭示其语言处理机制与人类相似。**

- **链接: [https://arxiv.org/pdf/2305.15929v3](https://arxiv.org/pdf/2305.15929v3)**

> **作者:** Juan Manuel Toro
>
> **备注:** 15 pages, 1 figure, corrected typo
>
> **摘要:** Current large language models, such as OpenAI's ChatGPT, have captured the public's attention because how remarkable they are in the use of language. Here, I demonstrate that ChatGPT displays phonological biases that are a hallmark of human language processing. More concretely, just like humans, ChatGPT has a consonant bias. That is, the chatbot has a tendency to use consonants over vowels to identify words. This is observed across languages that differ in their relative distribution of consonants and vowels such as English and Spanish. Despite the differences in how current artificial intelligence language models are trained to process linguistic stimuli and how human infants acquire language, such training seems to be enough for the emergence of a phonological bias in ChatGPT
>
---
#### [replaced 033] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于知识增强的生成任务，旨在解决RAG中因错误检索内容导致的干扰问题。通过引入参数化知识强化框架，提升模型在冲突场景下的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2506.05154v4](https://arxiv.org/pdf/2506.05154v4)**

> **作者:** Chenyu Lin; Yilin Wen; Du Su; Hexiang Tan; Fei Sun; Muhan Chen; Chenfu Bao; Zhonghou Lyu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Retrieval-augmented generation (RAG) improves performance on knowledge-intensive tasks but can be derailed by wrong, irrelevant, or conflicting retrieved text, causing models to rely on inaccurate evidence and cascade errors. We propose Knowledgeable-R1, a reinforcement-learning framework that explicitly trains large language models to use parametric knowledge (PK) to resist contextual interference while still exploiting external context when it is reliably helpful. Knowledgeable-R1 introduces a joint sampling scheme that generates paired responses with and without retrieval, and learns both local advantages (within each decoding regime) and global advantages under the same input to quantify when to ignore misleading context versus adopt it. We employ an asymmetric advantage transformation that amplifies exploratory behaviors toward parametric knowledge. Experiments show that Knowledgeable-R1 significantly improves robustness and reasoning accuracy in knowledge conflict scenarios and general RAG scenarios, outperforming SOTA baselines by +22.89% in counterfactual scenarios, and without degradation when the retrieved context is fully accurate.Our code are available at https://github.com/lcy80366872/knowledgeable-R1.
>
---
#### [replaced 034] Embodied Task Planning via Graph-Informed Action Generation with Large Language Model
- **分类: cs.CL**

- **简介: 该论文属于机器人任务规划领域，解决长周期规划中策略不连贯和幻觉问题。提出GiG框架，利用图结构增强记忆与决策，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2601.21841v2](https://arxiv.org/pdf/2601.21841v2)**

> **作者:** Xiang Li; Ning Yan; Masood Mortazavi
>
> **摘要:** While Large Language Models (LLMs) have demonstrated strong zero-shot reasoning capabilities, their deployment as embodied agents still faces fundamental challenges in long-horizon planning. Unlike open-ended text generation, embodied agents must decompose high-level intent into actionable sub-goals while strictly adhering to the logic of a dynamic, observed environment. Standard LLM planners frequently fail to maintain strategy coherence over extended horizons due to context window limitation or hallucinate transitions that violate constraints. We propose GiG, a novel planning framework that structures embodied agents' memory using a Graph-in-Graph architecture. Our approach employs a Graph Neural Network (GNN) to encode environmental states into embeddings, organizing these embeddings into action-connected execution trace graphs within an experience memory bank. By clustering these graph embeddings, the framework enables retrieval of structure-aware priors, allowing agents to ground current decisions in relevant past structural patterns. Furthermore, we introduce a novel bounded lookahead module that leverages symbolic transition logic to enhance the agents' planning capabilities through the grounded action projection. We evaluate our framework on three embodied planning benchmarks-Robotouille Synchronous, Robotouille Asynchronous, and ALFWorld. Our method outperforms state-of-the-art baselines, achieving Pass@1 performance gains of up to 22% on Robotouille Synchronous, 37% on Asynchronous, and 15% on ALFWorld with comparable or lower computational cost.
>
---
#### [replaced 035] SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM在决策过程中因忽略不确定性导致的探索效率低问题。提出SELAUR框架，通过引入不确定性信号提升学习效果。**

- **链接: [https://arxiv.org/pdf/2602.21158v2](https://arxiv.org/pdf/2602.21158v2)**

> **作者:** Dengjia Zhang; Xiaoou Liu; Lu Cheng; Yaqing Wang; Kenton Murray; Hua Wei
>
> **备注:** Accepted by PAKDD'26
>
> **摘要:** Large language models (LLMs) are increasingly deployed as multi-step decision-making agents, where effective reward design is essential for guiding learning. Although recent work explores various forms of reward shaping and step-level credit assignment, a key signal remains largely overlooked: the intrinsic uncertainty of LLMs. Uncertainty reflects model confidence, reveals where exploration is needed, and offers valuable learning cues even in failed trajectories. We introduce SELAUR: Self Evolving LLM Agent via Uncertainty-aware Rewards, a reinforcement learning framework that incorporates uncertainty directly into the reward design. SELAUR integrates entropy-, least-confidence-, and margin-based metrics into a combined token-level uncertainty estimate, providing dense confidence-aligned supervision, and employs a failure-aware reward reshaping mechanism that injects these uncertainty signals into step- and trajectory-level rewards to improve exploration efficiency and learning stability. Experiments on two benchmarks, ALFWorld and WebShop, show that our method consistently improves success rates over strong baselines. Ablation studies further demonstrate how uncertainty signals enhance exploration and robustness.
>
---
#### [replaced 036] When Can Transformers Count to n?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究Transformer在计数任务中的表现，探讨嵌入维度、上下文长度和词表大小的关系。揭示了当词表大于嵌入维度时，计数能力显著下降的理论瓶颈。**

- **链接: [https://arxiv.org/pdf/2407.15160v3](https://arxiv.org/pdf/2407.15160v3)**

> **作者:** Gilad Yehudai; Haim Kaplan; Guy Dar; Royi Rassin; Asma Ghandeharioun; Mor Geva; Amir Globerson
>
> **摘要:** Large language models based on the transformer architecture can solve highly complex tasks, yet their fundamental limitations on simple algorithmic problems remain poorly understood. In this work, we focus on basic counting tasks and investigate how the difficulty of these tasks scales with the transformer embedding dimension, the context length, and the vocabulary size. We reveal a sharp theoretical phase transition governed by the relationship between the embedding dimension and the vocabulary size. When the dimension is at least as large as the vocabulary, transformers can perfectly maintain token counts. However, when the vocabulary exceeds the embedding dimension, the interference between non-orthogonal token representations forces the network weights to scale polynomially. This renders the exact counting algorithm numerically unstable and practically unlearnable. We empirically validate this bottleneck by training transformers from scratch, demonstrating a strict performance drop at the theoretical threshold and catastrophic out of distribution failure when scaling the vocabulary or context length. Furthermore, we show that state-of-the-art pretrained models suffer from similar failure cases. Our work reveals a critical blind spot absent from the current literature regarding the connection among these three parameters, proving that vocabulary size fundamentally dictates the difficulty of counting tasks.
>
---
#### [replaced 037] A Multi-faceted Analysis of Cognitive Abilities: Evaluating Prompt Methods with Large Language Models on the CONSORT Checklist
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLM在临床试验报告评估中的校准与可靠性问题。通过对比两种模型在不同提示策略下的表现，分析其认知适应与校准误差。**

- **链接: [https://arxiv.org/pdf/2510.19139v3](https://arxiv.org/pdf/2510.19139v3)**

> **作者:** Sohyeon Jeon; Hyung-Chul Lee
>
> **备注:** We have decided to withdraw this manuscript because we believe it requires further revision and substantial improvement before it is suitable for dissemination to the academic community
>
> **摘要:** Despite the rapid expansion of Large Language Models (LLMs) in healthcare, robust and explainable evaluation of their ability to assess clinical trial reporting according to CONSORT standards remains an open challenge. In particular, uncertainty calibration and metacognitive reliability of LLM reasoning are poorly understood and underexplored in medical automation. This study applies a behavioral and metacognitive analytic approach using an expert-validated dataset, systematically comparing two representative LLMs - one general and one domain-specialized - across three prompt strategies. We analyze both cognitive adaptation and calibration error using metrics: Expected Calibration Error (ECE) and a baseline-normalized Relative Calibration Error (RCE) that enables reliable cross-model comparison. Our results reveal pronounced miscalibration and overconfidence in both models, especially under clinical role-playing conditions, with calibration error persisting above clinically relevant thresholds. These findings underscore the need for improved calibration, transparent code, and strategic prompt engineering to develop reliable and explainable medical AI.
>
---
#### [replaced 038] Probabilistic distances-based hallucination detection in LLMs with RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决LLMs中幻觉检测问题。通过分析提示与响应的分布距离，提出一种新的检测方法，提升RAG系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2506.09886v2](https://arxiv.org/pdf/2506.09886v2)**

> **作者:** Rodion Oblovatny; Alexandra Kuleshova; Konstantin Polev; Alexey Zaytsev
>
> **备注:** Updated approach to constructing a hallucination detection score. Added results from experiments with the NLI task. The approach with trainable deep kernels has been removed, with a focus on the unsupervised approach
>
> **摘要:** Detecting hallucinations in large language models (LLMs) is critical for their safety in many applications. Without proper detection, these systems often provide harmful, unreliable answers. In recent years, LLMs have been actively used in retrieval-augmented generation (RAG) settings. However, hallucinations remain even in this setting, and while numerous hallucination detection methods have been proposed, most approaches are not specifically designed for RAG systems. To overcome this limitation, we introduce a hallucination detection method based on estimating the distances between the distributions of prompt token embeddings and language model response token embeddings. The method examines the geometric structure of token hidden states to reliably extract a signal of factuality in text, while remaining friendly to long sequences. Extensive experiments demonstrate that our method achieves state-of-the-art or competitive performance. It also has transferability from solving the NLI task to the hallucination detection task, making it a fully unsupervised and efficient method with a competitive performance on the final task.
>
---
#### [replaced 039] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于艺术评价任务，旨在解决VLM在文化解读上的评估不足。通过构建三级评估框架，验证了模型在不同文化背景下的表现差异。**

- **链接: [https://arxiv.org/pdf/2601.07984v3](https://arxiv.org/pdf/2601.07984v3)**

> **作者:** Haorui Yu; Xuehang Wen; Fengrui Zhang; Qiufeng Yi
>
> **备注:** 16 pages, 7 figures, submitted to ACL 2026
>
> **摘要:** Vision-Language Models (VLMs) excel at visual description yet remain under-validated for cultural interpretation. Existing benchmarks assess perception without interpretation, and common evaluation proxies, such as automated metrics and LLM-judge averaging, are unreliable for culturally sensitive generative tasks. We address this measurement gap with a tri-tier evaluation framework grounded in art-theoretical constructs (Section 2). The framework operationalises cultural understanding through five levels (L1--L5) and 165 culture-specific dimensions across six traditions: Tier I computes automated quality indicators, Tier II applies rubric-based single-judge scoring, and Tier III calibrates the aggregate score to human expert ratings via sigmoid calibration. Applied to 15 VLMs across 294 evaluation pairs, the validated instrument reveals that (i) automated metrics and judge scoring measure different constructs, establishing single-judge calibration as the more reliable alternative; (ii) cultural understanding degrades from visual description (L1--L2) to cultural interpretation (L3--L5); and (iii) Western art samples consistently receive higher scores than non-Western ones. To our knowledge, this is the first cross-cultural evaluation instrument for generative art critique, providing a reproducible methodology for auditing VLM cultural competence. Framework code is available at https://github.com/yha9806/VULCA-Framework.
>
---
#### [replaced 040] Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE
- **分类: cs.CL; econ.EM**

- **简介: 该论文提出OccCANINE，解决职业描述自动编码问题，通过微调模型实现高效准确的HISCO编码，提升研究效率。**

- **链接: [https://arxiv.org/pdf/2402.13604v3](https://arxiv.org/pdf/2402.13604v3)**

> **作者:** Christian Møller Dahl; Torben Johansen; Christian Vedel
>
> **备注:** All code and guides on how to use OccCANINE is available on GitHub https://github.com/christianvedels/OccCANINE
>
> **摘要:** This paper introduces OccCANINE, an open-source tool that maps occupational descriptions to HISCO codes. Manual coding is slow and error-prone; OccCANINE replaces weeks of work with results in minutes. We fine-tune CANINE on 15.8 million description-code pairs from 29 sources in 13 languages. The model achieves 96 percent accuracy, precision, and recall. We also show that the approach generalizes to three systems - OCC1950, OCCICEM, and ISCO-68 - and release them open source. By breaking the "HISCO barrier," OccCANINE democratizes access to high-quality occupational coding, enabling broader research in economics, economic history, and related disciplines.
>
---
#### [replaced 041] Unleashing Low-Bit Inference on Ascend NPUs: A Comprehensive Evaluation of HiFloat Formats
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究低比特浮点格式在Ascend NPU上的推理效率问题。针对大模型推理中的精度与效率矛盾，提出HiFloat格式，通过实验验证其在不同任务中的性能优势。**

- **链接: [https://arxiv.org/pdf/2602.12635v2](https://arxiv.org/pdf/2602.12635v2)**

> **作者:** Pengxiang Zhao; Hui-Ling Zhen; Xing Li; Han Bao; Weizhe Lin; Zhiyuan Yang; Ziwei Yu; Xin Wang; Mingxuan Yuan; Xianzhi Yu; Zhenhua Dong
>
> **摘要:** As LLMs scale, low-bit floating-point formats like MXFP and NVFP4 offer new opportunities for precision and efficiency. In this work, we evaluate HiFloat (HiF8 and HiF4), a family of formats tailored for Ascend NPUs. Through rigorous comparison across weight-activation and KV-cache tasks, we provide three key insights: (1) INT8 suits narrow-range data, while floating-point formats excel with high-variance data; (2) in 4-bit regimes, HiF4's hierarchical scaling prevents the accuracy collapse seen in integer formats; and (3) HiFloat is fully compatible with state-of-the-art post-training quantization frameworks. Overall, HiFloat provides a solution for high-efficiency LLM inference on NPUs.
>
---
#### [replaced 042] Stabilizing Off-Policy Training for Long-Horizon LLM Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于多轮LLM代理训练任务，旨在解决离策略训练中的不稳定问题。提出SORL框架，通过重要性采样和裁剪规范化提升训练稳定性。**

- **链接: [https://arxiv.org/pdf/2511.20718v2](https://arxiv.org/pdf/2511.20718v2)**

> **作者:** Chenliang Li; Adel Elmahdy; Alex Boyd; Zhongruo Wang; Siliang Zeng; Alfredo Garcia; Parminder Bhatia; Taha Kass-Hout; Cao Xiao; Mingyi Hong
>
> **摘要:** Reinforcement learning (RL) algorithms such as PPO and GRPO are widely used to train large language models (LLMs) for multi-turn agentic tasks. However, in off-policy training pipelines, these methods often exhibit unstable optimization dynamics and are prone to performance collapse. Through empirical analysis, we identify two fundamental sources of instability in this setting: (1)~a granularity mismatch between token-level policy optimization and turn-structured interactions, and (2) high-variance and unreliable gradient updates induced by off-policy importance sampling and inaccurate advantage estimation. To address these challenges, we propose SORL, \underline{S}tabilizing \underline{O}ff-Policy \underline{R}einforcement \underline{L}earning for Long-Horizon Agent Training. SORL introduces principled mechanisms that align policy optimization with the structure of multi-turn interactions and adaptively suppress unreliable off-policy updates, yielding more conservative and robust learning dynamics. Within this framework, we instantiate two stabilized algorithms: SO-PPO and SO-GRPO. Both algorithms are designed to mitigate gradient variance and prevent optimization collapse without requiring careful early stopping or heuristic tuning. We evaluate SO-PPO and SO-GRPO on a range of multi-turn search benchmarks, including general question answering, multi-hop question answering, and medical multiple-choice QA tasks. Experimental results show that both methods consistently prevent training instabilities and performance collapses observed in standard PPO and GRPO, maintain lower clipping ratios and more stable optimization trajectories, and achieve superior or comparable task performance. These results demonstrate that the proposed algorithm provides a practical, scalable, and general framework for stabilizing reinforcement learning in multi-turn LLM agent training.
>
---
#### [replaced 043] FigEx2: Visual-Conditioned Panel Detection and Captioning for Scientific Compound Figures
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出FigEx2，解决科学复合图中面板检测与描述生成问题，通过视觉条件框架实现精准定位和高质量 captions 生成。**

- **链接: [https://arxiv.org/pdf/2601.08026v3](https://arxiv.org/pdf/2601.08026v3)**

> **作者:** Jifeng Song; Arun Das; Pan Wang; Hui Ji; Kun Zhao; Yufei Huang
>
> **摘要:** Scientific compound figures combine multiple labeled panels into a single image, but captions in real pipelines are often missing or only provide figure-level summaries, making panel-level understanding difficult. In this paper, we propose FigEx2, visual-conditioned framework that localizes panels and generates panel-wise captions directly from the compound figure. To mitigate the impact of diverse phrasing in open-ended captioning, we introduce a noise-aware gated fusion module that adaptively filters token-level features to stabilize the detection query space. Furthermore, we employ a staged optimization strategy combining supervised learning with reinforcement learning (RL), utilizing CLIP-based alignment and BERTScore-based semantic rewards to enforce strict multimodal consistency. To support high-quality supervision, we curate BioSci-Fig-Cap, a refined benchmark for panel-level grounding, alongside cross-disciplinary test suites in physics and chemistry. Experimental results demonstrate that FigEx2 achieves a superior 0.726 mAP@0.5:0.95 for detection and significantly outperforms Qwen3-VL-8B by 0.51 in METEOR and 0.24 in BERTScore. Notably, FigEx2 exhibits remarkable zero-shot transferability to out-of-distribution scientific domains without any fine-tuning.
>
---
#### [replaced 044] Embedding-Based Context-Aware Reranker
- **分类: cs.CL**

- **简介: 该论文提出EBCAR，解决RAG系统中跨段落推理的重排序问题，通过嵌入和注意力机制提升跨段理解，提升检索效果与效率。**

- **链接: [https://arxiv.org/pdf/2510.13329v2](https://arxiv.org/pdf/2510.13329v2)**

> **作者:** Ye Yuan; Mohammad Amin Shabani; Siqi Liu
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems rely on retrieving relevant evidence from a corpus to support downstream generation. The common practice of splitting a long document into multiple shorter passages enables finer-grained and targeted information retrieval. However, it also introduces challenges when a correct retrieval would require inference across passages, such as resolving coreference, disambiguating entities, and aggregating evidence scattered across multiple sources. Many state-of-the-art (SOTA) reranking methods, despite utilizing powerful large pretrained language models with potentially high inference costs, still neglect the aforementioned challenges. Therefore, we propose Embedding-Based Context-Aware Reranker (EBCAR), a lightweight reranking framework operating directly on embeddings of retrieved passages with enhanced cross-passage understandings through the structural information of the passages and a hybrid attention mechanism, which captures both high-level interactions across documents and low-level relationships within each document. We evaluate EBCAR against SOTA rerankers on the ConTEB benchmark, demonstrating its effectiveness for information retrieval requiring cross-passage inference and its advantages in both accuracy and efficiency.
>
---
#### [replaced 045] Renaissance: Investigating the Pretraining of Vision-Language Encoders
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言任务，旨在研究VL编码器的预训练方法。通过引入Renaissance框架，探索模型冻结和结构设计对性能的影响。**

- **链接: [https://arxiv.org/pdf/2411.06657v2](https://arxiv.org/pdf/2411.06657v2)**

> **作者:** Clayton Fields; Casey Kennington
>
> **备注:** 9 pages
>
> **摘要:** In the past several years there has been an explosion of available models for vision-language (VL) tasks. Unfortunately, the literature still leaves open a number of questions related to best practices in designing and training such models. Additionally, the limited programming tools available for modeling make conducting VL research more difficult than necessary. In this paper, we seek to answer several questions related to the pretraining of VL encoders through meta-analysis. To conduct these experiments, we introduce a VL evaluation framework called Renaissance. In our first set of experiments, we show that we can save significant compute at little to no cost to downstream performance, by freezing large parts of VL models during pretraining. In our second set of experiments, we examine the effect of basing a VL transformer on a vision model versus a text model. Renaissance offers a great deal of flexibility in creating, training and evaluating transformer encoders for VL modeling. Its source code will be made publicly available upon publication. The source code for Renaissance can be found at https://github.com/bsu-slim/renaissance.
>
---
#### [replaced 046] The Dark Side of ChatGPT: Legal and Ethical Challenges from Stochastic Parrots and Hallucination
- **分类: cs.CY; cs.CL; cs.LG**

- **简介: 论文探讨ChatGPT等LLM带来的法律与伦理问题，属于AI监管研究任务，旨在分析其风险并呼吁完善欧盟监管体系。**

- **链接: [https://arxiv.org/pdf/2304.14347v2](https://arxiv.org/pdf/2304.14347v2)**

> **作者:** Zihao Li
>
> **备注:** This is the preprint version of the paper 'Why the European AI Act transparency obligation is insufficient' (2023) Nature Machine Intelligence
>
> **摘要:** With the launch of ChatGPT, Large Language Models (LLMs) are shaking up our whole society, rapidly altering the way we think, create and live. For instance, the GPT integration in Bing has altered our approach to online searching. While nascent LLMs have many advantages, new legal and ethical risks are also emerging, stemming in particular from stochastic parrots and hallucination. The EU is the first and foremost jurisdiction that has focused on the regulation of AI models. However, the risks posed by the new LLMs are likely to be underestimated by the emerging EU regulatory paradigm. Therefore, this correspondence warns that the European AI regulatory paradigm must evolve further to mitigate such risks.
>
---
#### [replaced 047] LUMI: Unsupervised Intent Clustering with Multiple Pseudo-Labels
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于短文本聚类任务，旨在解决传统方法依赖单一标签和二元相似性判断的问题。提出LUMI方法，通过共享伪标签增强文本相似性，实现更稳定的无监督聚类。**

- **链接: [https://arxiv.org/pdf/2510.14640v4](https://arxiv.org/pdf/2510.14640v4)**

> **作者:** I-Fan Lin; Faegheh Hasibi; Suzan Verberne
>
> **摘要:** In this paper, we propose an intuitive, training-free and label-free method for intent clustering in conversational search. Current approaches to short text clustering use LLM-generated pseudo-labels to enrich text representations or to identify similar text pairs for pooling. The limitations are: (1) each text is assigned only a single label, and refining representations toward a single label can be unstable; (2) text-level similarity is treated as a binary selection, which fails to account for continuous degrees of similarity. Our method LUMI is designed to amplify similarities between texts by using shared pseudo-labels. We first generate pseudo-labels for each text and collect them into a pseudo-label set. Next, we compute the mean of the pseudo-label embeddings and pool it with the text embedding. Finally, we perform text-level pooling: Each text representation is pooled with its similar pairs, where similarity is determined by the degree of shared labels. Our evaluation on four benchmark sets shows that our approach achieves competitive results, better than recent state-of-the-art baselines, while avoiding the need to estimate the number of clusters during embedding refinement, as is required by most methods. Our findings indicate that LUMI can effectively be applied in unsupervised short-text clustering scenarios.
>
---
#### [replaced 048] The Art of Efficient Reasoning: Data, Reward, and Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究高效推理任务，旨在减少大语言模型的计算开销。通过奖励机制优化推理路径，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.20945v2](https://arxiv.org/pdf/2602.20945v2)**

> **作者:** Taiqiang Wu; Zenan Xu; Bo Zhou; Ngai Wong
>
> **备注:** Tech Report, Insights on Efficient Reasoning via Reward Shaping
>
> **摘要:** Large Language Models (LLMs) consistently benefit from scaled Chain-of-Thought (CoT) reasoning, but also suffer from heavy computational overhead. To address this issue, efficient reasoning aims to incentivize short yet accurate thinking trajectories, typically through reward shaping with Reinforcement Learning (RL). In this paper, we systematically investigate the mechanics of efficient reasoning for LLMs. For comprehensive evaluation, we advocate for more fine-grained metrics, including length distribution conditioned on correctness and performance across a wide spectrum of token budgets ranging from 2k to 32k. First, we reveal that the training process follows a two-stage paradigm: length adaptation and reasoning refinement. After that, we conduct extensive experiments (about 0.2 million GPU hours) in a unified protocol, deconstructing training prompts and rollouts, reward shaping, and optimization strategies. In particular, a key finding is to train on relatively easier prompts, ensuring the density of positive reward signals and thus avoiding the length collapse. Meanwhile, the learned length bias can be generalized across domains. We distill all findings into valuable insights and practical guidelines, and further validate them across the Qwen3 series, ranging from 0.6B to 30B, demonstrating the robustness and generalization.
>
---
#### [replaced 049] Chain-of-Thought Compression Should Not Be Blind: V-Skip for Efficient Multimodal Reasoning via Dual-Path Anchoring
- **分类: cs.MM; cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，解决CoT推理延迟高问题。通过V-Skip方法，结合语言和视觉信息优化token压缩，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2601.13879v3](https://arxiv.org/pdf/2601.13879v3)**

> **作者:** Dongxu Zhang; Yiding Sun; Cheng Tan; Wenbiao Yan; Ning Yang; Jihua Zhu; Haijun Zhang
>
> **摘要:** While Chain-of-Thought (CoT) reasoning significantly enhances the performance of Multimodal Large Language Models (MLLMs), its autoregressive nature incurs prohibitive latency constraints. Current efforts to mitigate this via token compression often fail by blindly applying text-centric metrics to multimodal contexts. We identify a critical failure mode termed Visual Amnesia, where linguistically redundant tokens are erroneously pruned, leading to hallucinations. To address this, we introduce V-Skip that reformulates token pruning as a Visual-Anchored Information Bottleneck (VA-IB) optimization problem. V-Skip employs a dual-path gating mechanism that weighs token importance through both linguistic surprisal and cross-modal attention flow, effectively rescuing visually salient anchors. Extensive experiments on Qwen2-VL and Llama-3.2 families demonstrate that V-Skip achieves a $2.9\times$ speedup with negligible accuracy loss. Specifically, it preserves fine-grained visual details, outperforming other baselines over 30\% on the DocVQA.
>
---
#### [replaced 050] Not All Errors Are Created Equal: ASCoT Addresses Late-Stage Fragility in Efficient LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理任务，解决推理可靠性问题。针对后期错误影响更大的现象，提出ASCoT方法，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2508.05282v4](https://arxiv.org/pdf/2508.05282v4)**

> **作者:** Dongxu Zhang; Ning Yang; Yiding Sun; Jihua Zhu; Jinnan Yang; Miao Xin; Baoliang Tian
>
> **摘要:** While Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs), ensuring reasoning reliability remains an open challenge. Contrary to the prevailing cascading failure hypothesis which posits that early errors are most detrimental, we identify a counter-intuitive phenomenon termed \textbf{Late-Stage Fragility}: errors introduced in later reasoning stages are significantly more prone to corrupting final answers. To address this, we introduce ASCoT (Adaptive Self-Correction Chain-of-Thought), a method harmonizing efficiency with robust verification. ASCoT first employs semantic pruning to compress redundant steps, then utilizes an Adaptive Verification Manager (AVM) to prioritize high risk, late-stage steps via a positional impact score, triggering a Multi-Perspective Self-Correction Engine (MSCE) only when necessary. Experiments on GSM8K and MATH-500 demonstrate that ASCoT effectively reallocates computational resources: it reduces token usage by 21\%--30\% for LLaMA-3.1-8B with negligible accuracy drops ($<1.8\%$), achieving a superior trade-off between inference efficiency and reasoning fidelity.
>
---
#### [replaced 051] Search or Accelerate: Confidence-Switched Position Beam Search for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SOAR算法，用于扩散语言模型的解码任务，解决生成质量与效率的平衡问题。通过动态调整搜索策略，提升生成效果并保持速度。**

- **链接: [https://arxiv.org/pdf/2602.10953v2](https://arxiv.org/pdf/2602.10953v2)**

> **作者:** Mingyu Cao; Alvaro H. C. Correia; Christos Louizos; Shiwei Liu; Lu Yin
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Diffusion Language Models (DLMs) generate text by iteratively denoising a masked sequence, repeatedly deciding which positions to commit at each step. Standard decoding follows a greedy rule: unmask the most confident positions, yet this local choice can lock the model into a suboptimal unmasking order, especially on reasoning-heavy prompts. We present SOAR, a training-free decoding algorithm that adapts its behavior to the model's uncertainty. When confidence is low, SOAR briefly widens the search over alternative unmasking decisions to avoid premature commitments; when confidence is high, it collapses the search and decodes many positions in parallel to reduce the number of denoising iterations. Across mathematical reasoning and code generation benchmarks (GSM8K, MBPP, HumanEval) on Dream-7B and LLaDA-8B, SOAR improves generation quality while maintaining competitive inference speed, offering a practical way to balance quality and efficiency in DLM decoding. Our Code is available at https://github.com/duterscmy/SOAR
>
---
#### [replaced 052] BURMESE-SAN: Burmese NLP Benchmark for Evaluating Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出BURMESE-SAN，首个针对缅甸语的NLP基准，评估语言模型的理解、推理和生成能力，解决低资源语言模型评估问题。**

- **链接: [https://arxiv.org/pdf/2602.18788v2](https://arxiv.org/pdf/2602.18788v2)**

> **作者:** Thura Aung; Jann Railey Montalan; Jian Gang Ngui; Peerat Limkonchotiwat
>
> **摘要:** We introduce BURMESE-SAN, the first holistic benchmark that systematically evaluates large language models (LLMs) for Burmese across three core NLP competencies: understanding (NLU), reasoning (NLR), and generation (NLG). BURMESE-SAN consolidates seven subtasks spanning these competencies, including Question Answering, Sentiment Analysis, Toxicity Detection, Causal Reasoning, Natural Language Inference, Abstractive Summarization, and Machine Translation, several of which were previously unavailable for Burmese. The benchmark is constructed through a rigorous native-speaker-driven process to ensure linguistic naturalness, fluency, and cultural authenticity while minimizing translation-induced artifacts. We conduct a large-scale evaluation of both open-weight and commercial LLMs to examine challenges in Burmese modeling arising from limited pretraining coverage, rich morphology, and syntactic variation. Our results show that Burmese performance depends more on architectural design, language representation, and instruction tuning than on model scale alone. In particular, Southeast Asia regional fine-tuning and newer model generations yield substantial gains. Finally, we release BURMESE-SAN as a public leaderboard to support systematic evaluation and sustained progress in Burmese and other low-resource languages. https://leaderboard.sea-lion.ai/detailed/MY
>
---
#### [replaced 053] Toward Safe and Human-Aligned Game Conversational Recommendation via Multi-Agent Decomposition
- **分类: cs.IR; cs.CL; cs.HC**

- **简介: 该论文属于对话推荐任务，旨在解决游戏推荐中的个性化、长尾覆盖和安全性问题。提出MATCHA框架，通过多智能体协作提升推荐效果与安全。**

- **链接: [https://arxiv.org/pdf/2504.20094v3](https://arxiv.org/pdf/2504.20094v3)**

> **作者:** Zheng Hui; Xiaokai Wei; Yexi Jiang; Kevin Gao; Chen Wang; Frank Ong; Se-eun Yoon; Rachit Pareek; Michelle Gong
>
> **备注:** ICML 2025 MAS, EACL 2026
>
> **摘要:** Conversational recommender systems (CRS) have advanced with large language models, showing strong results in domains like movies. These domains typically involve fixed content and passive consumption, where user preferences can be matched by genre or theme. In contrast, games present distinct challenges: fast-evolving catalogs, interaction-driven preferences (e.g., skill level, mechanics, hardware), and increased risk of unsafe responses in open-ended conversation. We propose MATCHA, a multi-agent framework for CRS that assigns specialized agents for intent parsing, tool-augmented retrieval, multi-LLM ranking with reflection, explanation, and risk control which enabling finer personalization, long-tail coverage, and stronger safety. Evaluated on real user request dataset, MATCHA outperforms six baselines across eight metrics, improving Hit@5 by 20%, reducing popularity bias by 24%, and achieving 97.9% adversarial defense. Human and virtual-judge evaluations confirm improved explanation quality and user alignment.
>
---
#### [replaced 054] HEART: A Unified Benchmark for Assessing Humans and LLMs in Emotional Support Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HEART框架，用于评估人类与大语言模型在情感支持对话中的表现。任务是对比两者在情感互动能力上的差异，解决如何衡量模型情感支持水平的问题。**

- **链接: [https://arxiv.org/pdf/2601.19922v2](https://arxiv.org/pdf/2601.19922v2)**

> **作者:** Laya Iyer; Kriti Aggarwal; Sanmi Koyejo; Gail Heyman; Desmond C. Ong; Subhabrata Mukherjee
>
> **摘要:** Supportive conversation depends on skills that go beyond language fluency, including reading emotions, adjusting tone, and navigating moments of resistance, frustration, or distress. Despite rapid progress in language models, we still lack a clear way to understand how their abilities in these interpersonal domains compare to those of humans. We introduce HEART, the first-ever framework that directly compares humans and LLMs on the same multi-turn emotional-support conversations. For each dialogue history, we pair human and model responses and evaluate them through blinded human raters and an ensemble of LLM-as-judge evaluators. All assessments follow a rubric grounded in interpersonal communication science across five dimensions: Human Alignment, Empathic Responsiveness, Attunement, Resonance, and Task-Following. HEART uncovers striking behavioral patterns. Several frontier models approach or surpass the average human responses in perceived empathy and consistency. At the same time, humans maintain advantages in adaptive reframing, tension-naming, and nuanced tone shifts, particularly in adversarial turns. Human and LLM-as-judge preferences align on about 80 percent of pairwise comparisons, matching inter-human agreement, and their written rationales emphasize similar HEART dimensions. This pattern suggests an emerging convergence in the criteria used to assess supportive quality. By placing humans and models on equal footing, HEART reframes supportive dialogue as a distinct capability axis, separable from general reasoning or linguistic fluency. It provides a unified empirical foundation for understanding where model-generated support aligns with human social judgment, where it diverges, and how affective conversational competence scales with model size.
>
---
#### [replaced 055] Knowledge Fusion of Large Language Models Via Modular SkillPacks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大语言模型知识融合任务，旨在解决跨能力迁移中的参数冲突与遗忘问题。提出GraftLLM方法，通过SkillPack实现高效知识传递与模型融合。**

- **链接: [https://arxiv.org/pdf/2505.18502v2](https://arxiv.org/pdf/2505.18502v2)**

> **作者:** Guodong Du; Zhuo Li; Xuanning Zhou; Junlin Li; Zesheng Shi; Wanyu Lin; Ho-Kin Tang; Xiucheng Li; Fangming Liu; Wenya Wang; Min Zhang; Jing Li
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Cross-capability transfer is a key challenge in large language model (LLM) research, with applications in multi-task integration, model compression, and continual learning. Recent works like FuseLLM and FuseChat have demonstrated the potential of transferring multiple model capabilities to lightweight models, enhancing adaptability and efficiency, which motivates our investigation into more efficient cross-capability transfer methods. However, existing approaches primarily focus on small, homogeneous models, limiting their applicability. For large, heterogeneous models, knowledge distillation with full-parameter fine-tuning often overlooks the student model's intrinsic capacity and risks catastrophic forgetting, while PEFT methods struggle to effectively absorb knowledge from source LLMs. To address these issues, we introduce GraftLLM, a novel method that stores source model capabilities in a target model with SkillPack format. This approach preserves general capabilities, reduces parameter conflicts, and supports forget-free continual learning and model fusion. We employ a module-aware adaptive compression strategy to compress parameter updates, ensuring efficient storage while maintaining task-specific knowledge. The resulting SkillPack serves as a compact and transferable knowledge carrier, ideal for heterogeneous model fusion and continual learning. Experiments across various scenarios demonstrate that GraftLLM outperforms existing techniques in knowledge transfer, knowledge fusion, and forget-free learning, providing a scalable and efficient solution for cross-capability transfer. The code is publicly available at: https://github.com/duguodong7/GraftLLM.
>
---
#### [replaced 056] Axis Decomposition for ODRL: Resolving Dimensional Ambiguity in Policy Constraints through Interval Semantics
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于形式化验证任务，解决ODRL策略约束中的维度歧义问题。通过轴分解框架，将多维操作数拆分为轴特定的标量，确保策略评估的确定性与一致性。**

- **链接: [https://arxiv.org/pdf/2602.19878v2](https://arxiv.org/pdf/2602.19878v2)**

> **作者:** Daham Mustafa; Diego Collarana; Yixin Peng; Rafiqul Haque; Christoph Lange-Bever; Christoph Quix; Stephan Decker
>
> **备注:** 16 pages, 5 tables. Preprint. v2: corrected projection soundness property; clarified verdict mapping table
>
> **摘要:** Every ODRL 2.2 constraint compares a single scalar value: (leftOperand, operator, rightOperand). Five of ODRL's left operands, however, denote multi-dimensional quantities--image dimensions, canvas positions, geographic coordinates--whose specification text explicitly references multiple axes. For these operands, a single scalar constraint admits one interpretation per axis, making policy evaluation non-deterministic. We classify ODRL's left operands by value-domain structure (scalar, dimensional, concept-valued), grounded in the ODRL 2.2 specification text, and show that dimensional ambiguity is intrinsic to the constraint syntax. We present an axis-decomposition framework that refines each dimensional operand into axis-specific scalar operands and prove four properties: deterministic interpretation, AABB completeness, projection soundness, and conservative extension. Conflict detection operates in two layers: per-axis verdicts are always decidable; box-level verdicts compose through Strong Kleene conjunction into a three-valued logic (Conflict, Compatible, Unknown). For ODRL's disjunctive (odrl:or) and exclusive-or (odrl:xone) logical constraints, where per-axis decomposition does not apply, the framework encodes coupled multi-axis conjectures directly. We instantiate the framework as the ODRL Spatial Axis Profile--15 axis-specific left operands for the five affected base terms--and evaluate it on 117 benchmark problems spanning nine categories across both TPTP FOF (Vampire) and SMT-LIB (Z3) encodings, achieving full concordance between provers. Benchmark scenarios are inspired by constraints arising in cultural heritage dataspaces such as Datenraum Kultur. All meta-theorems are mechanically verified in Isabelle/HOL.
>
---
#### [replaced 057] EmoGRACE: Aspect-based emotion analysis for social media data
- **分类: cs.CL**

- **简介: 该论文提出EmoGRACE，解决社交媒体中基于方面的情感分析问题，构建首个ABEA数据集并微调BERT模型，提升情感分类准确性。**

- **链接: [https://arxiv.org/pdf/2503.15133v2](https://arxiv.org/pdf/2503.15133v2)**

> **作者:** Christina Zorenböhmer; Sebastian Schmidt; Bernd Resch
>
> **摘要:** While sentiment analysis has advanced from sentence to aspect-level, i.e., the identification of concrete terms related to a sentiment, the equivalent field of Aspect-based Emotion Analysis (ABEA) is faced with dataset bottlenecks and the increased complexity of emotion classes in contrast to binary sentiments. This paper addresses these gaps, by generating a first ABEA training dataset, consisting of 2,621 English Tweets, and fine-tuning a BERT-based model for the ABEA sub-tasks of Aspect Term Extraction (ATE) and Aspect Emotion Classification (AEC). The dataset annotation process was based on the hierarchical emotion theory by Shaver et al. [1] and made use of group annotation and majority voting strategies to facilitate label consistency. The resulting dataset contained aspect-level emotion labels for Anger, Sadness, Happiness, Fear, and a None class. Using the new ABEA training dataset, the state-of-the-art ABSA model GRACE by Luo et al. [2] was fine-tuned for ABEA. The results reflected a performance plateau at an F1-score of 70.1% for ATE and 46.9% for joint ATE and AEC extraction. The limiting factors for model performance were broadly identified as the small training dataset size coupled with the increased task complexity, causing model overfitting and limited abilities to generalize well on new data.
>
---
#### [replaced 058] Robust Preference Alignment via Directional Neighborhood Consensus
- **分类: cs.CL**

- **简介: 该论文属于模型对齐任务，解决LLM在特定偏好上表现不佳的问题。提出RPS方法，通过局部邻域共识提升模型鲁棒性，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2510.20498v3](https://arxiv.org/pdf/2510.20498v3)**

> **作者:** Ruochen Mao; Yuling Shi; Xiaodong Gu; Jiaheng Wei
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Aligning large language models with human preferences is critical for creating reliable and controllable AI systems. A human preference can be visualized as a high-dimensional vector where different directions represent trade-offs between desired attributes (e.g., helpfulness vs. verbosity). Yet, because the training data often reflects dominant, average preferences, LLMs tend to perform well on common requests but fall short in specific, individual needs. This mismatch creates a preference coverage gap. Existing methods often address this through costly retraining, which may not be generalized to the full spectrum of diverse preferences. This brittleness means that when a user's request reflects a nuanced preference deviating from the training data's central tendency, model performance can degrade unpredictably. To address this challenge, we introduce Robust Preference Selection (RPS), a post-hoc, training-free method by leveraging directional neighborhood consensus. Instead of forcing a model to generate a response from a single, highly specific preference, RPS samples multiple responses from a local neighborhood of related preferences to create a superior candidate pool. It then selects the response that best aligns with the user's original intent. We provide a theoretical framework showing our neighborhood generation strategy is provably superior to a strong baseline that also samples multiple candidates. Comprehensive experiments across three distinct alignment paradigms (DPA, DPO, and SFT) demonstrate that RPS consistently improves robustness against this baseline, achieving win rates of up to 69% on challenging preferences from under-represented regions of the space without any model retraining. Our work presents a practical, theoretically-grounded solution for enhancing the reliability of preference-aligned models.
>
---
#### [replaced 059] Incentive-Aligned Multi-Source LLM Summaries
- **分类: cs.CL; cs.AI; cs.GT**

- **简介: 该论文属于文本摘要任务，旨在解决多源信息合成中的事实准确性与抗攻击问题。提出TTS框架，通过激励机制提升摘要的可信度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.25184v2](https://arxiv.org/pdf/2509.25184v2)**

> **作者:** Yanchen Jiang; Zhe Feng; Aranyak Mehta
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source's stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source's incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation.
>
---
#### [replaced 060] Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，研究联邦大语言模型中的数据泄露风险，提出有效提取私有信息的方法，评估其在法律领域的隐私威胁。**

- **链接: [https://arxiv.org/pdf/2506.06060v2](https://arxiv.org/pdf/2506.06060v2)**

> **作者:** Yingqi Hu; Zhuo Zhang; Jingyuan Zhang; Jinghua Wang; Qifan Wang; Lizhen Qu; Zenglin Xu
>
> **备注:** IJCNLP 2025 Findings
>
> **摘要:** Federated large language models (FedLLMs) enable cross-silo collaborative training among institutions while preserving data locality, making them appealing for privacy-sensitive domains such as law, finance, and healthcare. However, the memorization behavior of LLMs can lead to privacy risks that may cause cross-client data leakage. In this work, we study the threat of cross-client data extraction, where a semi-honest participant attempts to recover personally identifiable information (PII) memorized from other clients' data. We propose three simple yet effective extraction strategies that leverage contextual prefixes from the attacker's local data, including frequency-based prefix sampling and local fine-tuning to amplify memorization. To evaluate these attacks, we construct a Chinese legal-domain dataset with fine-grained PII annotations consistent with CPIS, GDPR, and CCPA standards, and assess extraction performance using two metrics: coverage and efficiency. Experimental results show that our methods can recover up to 56.6% of victim-exclusive PII, where names, addresses, and birthdays are particularly vulnerable. These findings highlight concrete privacy risks in FedLLMs and establish a benchmark and evaluation framework for future research on privacy-preserving federated learning. Code and data are available at https://github.com/SMILELab-FL/FedPII.
>
---
#### [replaced 061] In-Context Algebra
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究Transformer在上下文代数任务中的推理机制，解决变量意义不固定的符号推理问题。通过设计数据分布，发现模型学习到三种机制：交换复制、身份识别和闭包消去。**

- **链接: [https://arxiv.org/pdf/2512.16902v2](https://arxiv.org/pdf/2512.16902v2)**

> **作者:** Eric Todd; Jannik Brinkmann; Rohit Gandikota; David Bau
>
> **备注:** ICLR 2026. 35 pages, 22 figures. Code and data at https://algebra.baulab.info
>
> **摘要:** We investigate the mechanisms that arise when transformers are trained to solve arithmetic on sequences where tokens are variables whose meaning is determined only through their interactions in-context. While prior work has studied transformers in settings where the answer relies on fixed parametric or geometric information encoded in token embeddings, we devise a new in-context reasoning task where the assignment of tokens to specific algebraic elements varies from one sequence to another. Despite this challenging setup, transformers achieve near-perfect accuracy on the task and even generalize to unseen groups. We develop targeted data distributions to create causal tests of a set of hypothesized mechanisms, and we isolate three mechanisms models consistently learn: commutative copying where a dedicated head copies answers, identity element recognition that distinguishes identity-containing facts, and closure-based cancellation that tracks group membership to constrain valid answers. Our findings show that the kinds of reasoning strategies learned by transformers are dependent on the task structure and that models can develop symbolic reasoning mechanisms when trained to reason in-context about variables whose meanings are not fixed.
>
---
#### [replaced 062] Scaling Model and Data for Multilingual Machine Translation with Open Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言机器翻译任务，研究如何通过模型和数据扩展提升开放大语言模型的多语种翻译能力，提出MiLMMT-46模型，在46种语言上取得优异性能。**

- **链接: [https://arxiv.org/pdf/2602.11961v2](https://arxiv.org/pdf/2602.11961v2)**

> **作者:** Yuzhe Shang; Pengzhi Gao; Wei Liu; Jian Luan; Jinsong Su
>
> **摘要:** Open large language models (LLMs) have demonstrated improving multilingual capabilities in recent years. In this paper, we present a study of open LLMs for multilingual machine translation (MT) across a range of languages, and investigate the effects of model scaling and data scaling when adapting open LLMs to multilingual MT through continual pretraining and instruction finetuning. Based on the Gemma3 model family, we develop MiLMMT-46, which achieves top-tier multilingual translation performance across 46 languages. Extensive experiments show that MiLMMT-46 consistently outperforms recent state-of-the-art (SOTA) models, including Seed-X, HY-MT-1.5, and TranslateGemma, and achieves competitive performance with strong proprietary systems such as Google Translate and Gemini 3 Pro. Models are released at https://huggingface.co/collections/xiaomi-research/milmmt-46. Codes are released at https://github.com/xiaomi-research/gemmax.
>
---
