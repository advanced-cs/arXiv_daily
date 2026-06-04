# 自然语言处理 cs.CL

- **最新发布 125 篇**

- **更新 98 篇**

## 最新发布

#### [new 001] Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出APB基准，用于诊断LLM代理的规划能力，解决现有评估无法区分规划与执行失败的问题。通过多领域测试，揭示模型在长程规划等环节的弱点。**

- **链接: [https://arxiv.org/pdf/2606.04874](https://arxiv.org/pdf/2606.04874)**

> **作者:** Haoyu Sun; Wenxuan Wang; Mingyang Song; Jujie He; Weinan Zhang; Yang Liu; Yang Yang; Yu Cheng
>
> **摘要:** Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce \textbf{Agent Planning Benchmark (APB)}, a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $\tau^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics across three representative models. APB thus serves as an upstream diagnostic complement to execution benchmarks.
>
---
#### [new 002] SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM Inference
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大模型推理优化任务，解决长序列注意力计算效率问题。提出SparDA架构，通过稀疏解耦注意力提升推理速度与吞吐量。**

- **链接: [https://arxiv.org/pdf/2606.04511](https://arxiv.org/pdf/2606.04511)**

> **作者:** Yaosheng Fu; Guangxuan Xiao; Xin Dong; Song Han; Oreste Villa
>
> **摘要:** Sparse attention reduces compute and memory bandwidth for long-context LLM inference. However, two key challenges remain: (1) KV cache capacity still grows with sequence length, and offloading to CPU memory introduces a PCIe transfer bottleneck; (2) the sparse selection step itself retains $O(T^2)$ complexity and can dominate attention cost at long contexts. We propose SparDA, a decoupled sparse attention architecture that introduces a fourth per-layer projection, the Forecast, alongside Query, Key, and Value. The Forecast predicts the KV blocks needed by the next layer, enabling lookahead selection that overlaps CPU-to-GPU prefetch with current-layer execution. Because Forecast is decoupled from the attention query, our GQA implementation uses one Forecast head per GQA group, reducing selection overhead versus the original multi-head selector. SparDA adds $<$0.5% parameters and trains only the Forecast projections by matching the original selector's attention distribution. On two sparse-pretrained 8B models, SparDA matches or slightly improves accuracy and delivers up to 1.25$\times$ prefill speedup and 1.7$\times$ decode speedup over the sparse-attention offload baseline. By enabling larger feasible batch sizes on a single GPU, SparDA further reaches up to 5.3$\times$ higher decode throughput than the non-offload sparse baseline. Our source code is available at this https URL.
>
---
#### [new 003] SePO: Self-Evolving Prompt Agent for System Prompt Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SePO，用于系统提示优化，解决传统方法中提示代理自身提示固定的问题。通过自进化设计，同时优化任务代理和自身提示，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.04465](https://arxiv.org/pdf/2606.04465)**

> **作者:** Wangcheng Tao; Han Wu; Weng-Fai Wong
>
> **备注:** 26 pages. Code: this https URL
>
> **摘要:** System prompt optimization improves agent behavior without modifying the underlying model, yielding human-readable, model-agnostic instructions. Existing methods build a prompt agent that refines task agents' system prompts, yet leave the prompt agent's own system prompt hand-engineered and fixed. We propose Self-Evolving Prompt Optimization (SePO), which treats the prompt agent's own system prompt as an optimization target alongside task agents' system prompts. SePO adopts a self-referential design. A single prompt agent improves both task agents' system prompts and its own under an open-ended evolutionary search that maintains an archive of candidate prompts as stepping stones. Training proceeds in two stages: pre-training evolves the prompt agent on a multi-task pool, and fine-tuning then applies it to a target task. Across five benchmarks spanning math (AIME'25), abstract reasoning (ARC-AGI-1), graduate-level science (GPQA), code generation (MBPP), and logic puzzles (Sudoku), SePO consistently outperforms Manual-CoT, TextGrad, and MetaSPO, improving the average accuracy by 4.49 points compared to Manual-CoT. The prompt optimization skill from pre-training also generalizes to tasks beyond the pre-training mixture, rather than memorizing per-task prompts.
>
---
#### [new 004] MemoryDocDataSet: A Benchmark for Joint Conversational Memory and Long Document Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemoryDocDataSet，解决对话记忆与长文档推理联合任务中的评估缺失问题，通过构建包含多轮对话和长文档的基准数据集，分析不同模型表现，推动相关技术发展。**

- **链接: [https://arxiv.org/pdf/2606.04442](https://arxiv.org/pdf/2606.04442)**

> **作者:** Qiyang Xie; Jialun Wu; Xinjie He; Su Liu; Shuai Xiao; Zhiyuan Lin; Weikai Zhou
>
> **备注:** 17 pages, 2 figures, 8 tables. Submitted for peer review
>
> **摘要:** AI systems increasingly need to combine two demanding capabilities: navigating multi-session conversation history and performing deep reading comprehension within long documents. Yet no existing benchmark evaluates both simultaneously. We introduce MemoryDocDataSet, a synthetic benchmark of 50 micro-worlds and 1,000 QA pairs in which each instance comprises 3-5 personas, a temporal event graph spanning months of activity, 3-5 real long documents (20,000-50,000 tokens each sourced from the Caselaw Access Project), multi-session conversations grounded on those documents, and 20 question-answer pairs across five reasoning categories. The defining feature is the Hybrid source tag: questions requiring a system to first navigate conversation history to identify which document is relevant, then extract the answer from within that document. Hybrid questions account for 75.1% of the dataset. Dataset quality is characterised through a prompt-sensitivity self-consistency analysis using LLM-as-judge, yielding a median Cohen's $\kappa = 0.634$ across all 50 micro-worlds. We evaluate six baseline configurations spanning truncated context, long-context LLMs, retrieval-augmented generation (RAG), and memory systems. The best baseline (RAG-Both) achieves 0.358 overall F1 and 0.342 on Hybrid. Document-only retrieval (RAG-Doc) collapses to 0.267 on Hybrid despite achieving 0.453 on Doc-only questions, demonstrating a clear joint-retrieval gap that motivates architectures unifying conversational memory with long-document navigation. We release the dataset, generation pipeline, and all baseline implementations.
>
---
#### [new 005] Automatic Generation of Titles for Research Papers Using Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决论文标题自动生成问题。通过预训练语言模型和数据集，提出一种自动生成标题的方法，并验证其效果。**

- **链接: [https://arxiv.org/pdf/2606.05085](https://arxiv.org/pdf/2606.05085)**

> **作者:** Tohida Rehman; Debarshi Kumar Sanyal; Samiran Chattopadhyay
>
> **备注:** 24 pages, 24 tables, 01 figure
>
> **摘要:** The title of a research paper conveys its primary idea and, occasionally, its conclusions in a clear and concise manner. Choosing an appropriate title is often challenging, and automated title generation can assist authors in this task. In this work, we propose a technique to generate paper titles from abstracts using open-weight pre-trained and large language models. We use the CSPubSum and LREC-COLING-2024 datasets and introduce a new dataset, SpringerSSAT, curated from four Springer journals in the social sciences. Additionally, we use GPT-3.5-turbo in a zero-shot setting to generate titles. Model performance is evaluated with ROUGE, METEOR, MoverScore, BERTScore, and SciBERTScore metrics. Our experiments show that fine-tuned PEGASUS-large outperforms other models, including fine-tuned LLaMA-3-8B and zero-shot GPT-3.5-turbo, across most metrics. We further demonstrate that ChatGPT can generate creative paper titles. Overall, AI-generated titles are generally appropriate and reliable.
>
---
#### [new 006] Rethinking Continual Experience Internalization for Self-Evolving LLM Agents
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于持续学习任务，旨在解决LLM在多轮经验学习中的能力退化问题。通过优化经验粒度、注入模式和内化机制，提升模型的稳定性和持续学习能力。**

- **链接: [https://arxiv.org/pdf/2606.04703](https://arxiv.org/pdf/2606.04703)**

> **作者:** Jingwen Chen; Wenkai Yang; Shengda Fan; Wenbo Nie; Chenxing Sun; Shaodong Zheng; Yangen Hu; Lu Pan; Ke Zeng; Yankai Lin
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Experience internalization converts contextual experience from past interactions into reusable parametric capability, offering a promising path toward continual learning in large language models (LLMs). While prior work has predominantly focused on single-iteration transfer, we discover that under multi-iteration experience learning, existing methods suffer from a progressive capability collapse rather than compounding improvement. We systematically examine this failure through three vital dimensions of experience internalization: (1) Experience Granularity: We find that principle-level experience is more durable than instance-level experience, as it effectively abstracts transferable strategies away from trajectory-specific details. (2) Experience Injection Pattern: Our analysis reveals that step-wise injection significantly outperforms global injection by aligning experience with intermediate decision states, a property that is critical for long-horizon tool use. (3) Internalization Regime: We demonstrate that off-policy context-distillation on high-quality teacher trajectories provides a substantially more stable training signal than on-policy context-distillation, which is inherently limited by local corrections on student-induced flawed states. Together, these insights yield a simple yet robust recipe for stable and sustainable experience internalization, providing concrete guidance for engineering self-evolving and continually learning LLMs.
>
---
#### [new 007] SMADE-IE: Sparse Multi-Agent Framework with Evidence-Driven Debate for Zero-Shot Information Extraction
- **分类: cs.CL**

- **简介: 该论文属于零样本信息抽取任务，旨在解决传统方法在精度和效率上的不足。提出SMADE-IE框架，通过动态模式选择和证据驱动的辩论机制提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2606.04691](https://arxiv.org/pdf/2606.04691)**

> **作者:** Kenfeng Huang; Yi Cai; Xin Wu; Zikun Deng; Li Yuan
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** Zero-shot information extraction (IE) with large language models (LLMs) has attracted increasing attention due to its flexibility in adapting to new schemas and domains without task-specific training. Existing approaches mainly rely on monolithic prompting, each-type prompting, or multi-agent debate. However, monolithic prompting often suffers from boundary and type errors, while each-type prompting and multi-agent debate introduce cross-type conflicts, redundant agent interactions, and substantial token overhead. To address these challenges, we propose SMADE-IE, a sparse and evidence-driven multi-agent framework for zero-shot IE. SMADE-IE first employs an Adaptive Mode Selector to dynamically route inputs into either a lightweight Global Extraction Mode or a Type-Centric Extraction Mode, reducing unnecessary type selection and reasoning noise. For conflicting predictions, we further introduce an Evidence-Driven Debate mechanism that structures arguments into Toulmin-style components and performs confidence aggregation through external evidence scoring and Bayesian updates. Experimental results on 9 benchmark datasets across NER, RE, and JERE tasks show that SMADE-IE consistently outperforms existing zero-shot IE baselines while also improving token efficiency through sparse agent selection and early-stopping debate.
>
---
#### [new 008] SemBlock: Semantic Boundary Dynamic Blocks for Diffusion LLMs
- **分类: cs.CL**

- **简介: 该论文提出SemBlock，用于扩散语言模型的动态分块解码，解决固定块大小或分隔符不匹配语义边界的问题。通过预测语义边界提升生成效果。**

- **链接: [https://arxiv.org/pdf/2606.04964](https://arxiv.org/pdf/2606.04964)**

> **作者:** Xinrui Song; Zhuoran Wang; Mingju Gao; Hao Tang
>
> **备注:** Code: this https URL
>
> **摘要:** Diffusion language models (DLMs) generate text through iterative denoising, and blockwise decoding improves their practicality by committing tokens in local blocks. However, existing blockwise methods typically rely on fixed block sizes or delimiter-based runtime signals, which do not necessarily align with semantic boundaries. In this paper, we propose SemBlock, a semantic-boundary-driven dynamic block decoding framework for diffusion LLMs. SemBlock formulates dynamic block construction as semantic boundary prediction and trains lightweight predictors on frozen LLaDA hidden states. To provide supervision, we construct SemBound, a semantic-boundary dataset that derives boundary labels from discourse units, reasoning steps, and implementation spans across natural language, math, and code tasks. During inference, SemBlock uses predicted boundary probabilities to select the ending position of each dynamic block. Experiments on GSM8K, IFEval, MATH, and HumanEval show that SemBlock consistently improves over fixed-block decoding and AdaBlock. Our code is publicly available: this https URL.
>
---
#### [new 009] Fast & Faithful Function Vectors
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究函数向量（FVs）的优化，属于大语言模型任务。旨在提升FV的效率和准确性，通过改进注意力头选择和分布式 steering 方法实现。**

- **链接: [https://arxiv.org/pdf/2606.05079](https://arxiv.org/pdf/2606.05079)**

> **作者:** Minh An Pham; Anton Segeler; Thomas Wiegand; Wojciech Samek; Sebastian Lapuschkin; Patrick Kahardipraja; Reduan Achtibat
>
> **摘要:** Function vectors (FVs) are task representations elicited during in-context learning that can be used to steer Large Language Models (LLMs). However, design choices in their formulation remain underexplored. In this work, we study the impact of varying FV definitions for instructions along two degrees of freedom: attention head selection and steering. For head selection, using gradient-based attributions with Layer-wise Relevance Propagation (LRP) substantially improves efficiency as well as accuracy. For FV steering, applying it in a distributed manner yields a higher accuracy compared to simple aggregation. Our code is publicly available.
>
---
#### [new 010] Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions: Evidence from the St. Petersburg Game
- **分类: cs.CL; cs.CY; econ.GN**

- **简介: 该论文属于人工智能风险决策研究，探讨LLMs在决策任务中的行为与人类机制的对齐程度。通过St. Petersburg游戏实验，发现LLMs虽表现类似人类，但机制不同，强调需关注决策机制而非仅结果相似性。**

- **链接: [https://arxiv.org/pdf/2606.04978](https://arxiv.org/pdf/2606.04978)**

> **作者:** Chensong Huang; Changyu Chen; Chenwei Lin; Hanjia Lyu; Xian Xu; Jiebo Luo
>
> **摘要:** LLMs can appear cautious in risk decision-making tasks, yet cautious-looking outputs do not necessarily indicate alignment with human decision-making mechanisms. We investigate this distinction using the St. Petersburg game as a controlled testbed, a classical paradox in which the expected payoff is infinite, yet humans typically report low, finite willingness to pay. We evaluate 28 LLMs with a structured prompt suite that includes the original game; controlled decision variants that perturb truncation, repeated play, numeric endowment, and occupational identity; a human-perspective prompt that asks models to reason as human decision makers; and paired comparisons between base models and their instruction-tuned counterparts. In the original game, most models generate finite bids, creating the appearance of human-like risk behavior. However, this outcome-level resemblance masks substantial mechanism-level differences. The controlled variants reveal that rather than maintaining human-like behavior seen in the original game, models often shift to conditionally and computationally rational behavior. Human-cue prompting and instruction tuning often lower bids and reduce some visible pathologies, but most mechanism-level response patterns remain largely unchanged. These findings show that behavioral alignment in risk decision-making can be surface-level: LLMs may produce human-like risk decisions without exhibiting human-consistent mechanisms. High-stakes evaluations of LLM decision-making should therefore move beyond outcome similarity and examine whether the alignment is supported by mechanism-level consistency.
>
---
#### [new 011] Arithmetic Pedagogy for Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于语言模型的数学推理任务，旨在提升模型的算术能力。通过借鉴人类教学方法，设计训练数据和监督信号，使小模型达到高准确率。**

- **链接: [https://arxiv.org/pdf/2606.05106](https://arxiv.org/pdf/2606.05106)**

> **作者:** Andhika Bernard Lumbantobing; Hokky Situngkir
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** We investigate whether methods of human mathematics pedagogy can guide the training of language models toward arithmetic reasoning. Building on the GASING method -- an Indonesian pedagogy that solves basic arithmetic through a left-to-right procedure aligned with the causal order of token generation -- we operationalize each operation as a computational procedure whose execution trace is serialized into natural-language Chain-of-Thought (CoT) supervision. A small GPT-2 decoder (86M parameters) with a syllabic-agglutinative TOBA tokenizer for Indonesian is trained from scratch on this data using only a next-token prediction objective, without reinforcement learning or reward-based optimization. Monitoring training reveals three distinct learning phases, and mechanistic analyses -- attention-masking interventions on the CoT information graph, residual-stream probing, and logit-lens inspection -- show that the model first internalizes a procedural pathway and subsequently develops an associative, ``mental-arithmetic'' capacity that retrieves intermediate results without explicit step-by-step computation. The trained model reaches over 80% accuracy on held-out problems and attains competitive performance against substantially larger language models, indicating that targeted, pedagogically grounded training can yield strong and economical arithmetic capability at small scale.
>
---
#### [new 012] Off-Distribution Voices: Fanfiction Subgenres as Universal Vernacular Jailbreaks for Aligned LLMs
- **分类: cs.CL**

- **简介: 该论文属于安全攻击任务，旨在破解对齐的大型语言模型。通过使用同人小说子类型作为通用攻击载体，提升攻击成功率。**

- **链接: [https://arxiv.org/pdf/2606.04483](https://arxiv.org/pdf/2606.04483)**

> **作者:** Zhongze Luo; Ruihe Shi; Zhenshuai Yin; Haoyue Liu; Weixuan Wan; Xiaoying Tang
>
> **备注:** 23 pages
>
> **摘要:** Existing jailbreaks against aligned LLMs are discrete artifacts whose surface forms are easy to fingerprint and patch. We argue that the real failure mode is not any specific prompt, but an entire register of natural human writing that safety training has under-covered. Building on this insight, we introduce the first jailbreak family that uses real fanfiction subgenres as universal attack carriers: a creative-writing meta is conditioned on passages from one of twelve Archive of Our Own (AO3) subgenres, and the harmful behavior is embedded as the climax of the resulting scene. The construction requires no attacker LLM and no per-target adaptation. On eight aligned LLMs over the union of HarmBench and JailbreakBench, this attack lifts mean ASR from 0.278 to 0.731 under a four-judge ensemble; a factorial decomposition shows the gain is carried by register rather than length or structure. Two active defences widen rather than narrow the vernacular-to-baseline ratio, indicating that template-targeting defences merely steer attackers toward register-based attacks like ours. We also propose SAGA-A4, a static four-turn extension that attains mean ASR 0.924, substantially exceeding three existing multi-turn methods.
>
---
#### [new 013] CRAFT: Cost-aware Refinement And Front-aware Tuning of Prompts
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CRAFT，解决提示优化中的准确率与成本平衡问题，通过帕累托前沿搜索提升效果。**

- **链接: [https://arxiv.org/pdf/2606.04661](https://arxiv.org/pdf/2606.04661)**

> **作者:** Shanu Kumar; Shubhanshu Khandelwal; Akhila Yesantarao Venkata; Parag Agrawal; Yova Kementchedjhieva; Manish Gupta
>
> **摘要:** Prompts tuned for accuracy often grow long, raising inference cost on every model call. The best accuracy-cost trade-off depends on the task and the budget, so prompt optimization is a search over the Pareto front of accuracy and prompt-token cost rather than for one prompt. The usual shortcut, collapsing the objectives into a weighted sum, fixes the trade-off weight before search and often recovers only a narrow region of the front, a failure we call scalarization collapse. We present CRAFT (Cost-aware Refinement And Front-aware Tuning), a Pareto-front prompt optimizer that treats target-LLM validation calls as the scarce resource and allocates them to candidates near the optimistic candidate front. Each round, complementary accuracy-oriented and cost-oriented generators propose edits, Pareto-gap acquisition spends the per-round validation budget, and NSGA-II retention keeps a spread-out population. Across six classification and reasoning benchmarks, CRAFT's retained fronts reach both high-accuracy and low-cost regions, while accuracy-only, cost-only, and weighted-sum baselines each concentrate in narrower regions. The accuracy-cost trade-off becomes a post-search choice, not a pre-search weight.
>
---
#### [new 014] PersonaTree: Structured Lifecycle Memory for Person Understanding in LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出PersonaTree，解决长期交互中人物理解的显式记忆构建问题。通过结构化生命周期记忆框架，提升人物抽象理解与对齐效果。任务属于持续智能体的持久记忆与人物建模。**

- **链接: [https://arxiv.org/pdf/2606.04780](https://arxiv.org/pdf/2606.04780)**

> **作者:** Yubo Hou; Jingwei Song; Hongbo Zhang; Zhisheng Chen; Bang Xiao; Tao Wan; Zengchang Qin
>
> **摘要:** Persistent LLM agents require memory representations that make the formation of person understanding explicit across long term interaction. Existing agent memory methods emphasize information retention and retrieval, yet give limited account of how accumulated interaction evidence is abstracted into person understanding. We view this process as schema formation, where situated evidence is abstracted into reusable patterns and stable person level claims. We introduce PersonaTree, a structured lifecycle memory framework that realizes this view as a three level persona tree with explicit support paths from evidence to claims. PersonaTree maintains the tree through conservative writing, confidence guided consolidation, and query conditioned path retrieval, returning only the evidence depth required by each query. Across six person understanding and persistent memory benchmarks with three answer backbones, PersonaTree ranks first in 12 of 18 compact scores and reaches the top two in 16 settings. Ablations show that hierarchy improves abstract person understanding on KnowMe, while support path retrieval improves RealPref alignment under a comparable context budget.
>
---
#### [new 015] Computational conceptual history of scientific concepts: From early digital methods to LLMs
- **分类: cs.CL**

- **简介: 该论文属于计算概念史研究，探讨LLMs在科学概念分析中的应用，解决传统方法与LLMs的继承与差异问题，回顾相关案例并分析方法论挑战。**

- **链接: [https://arxiv.org/pdf/2606.04118](https://arxiv.org/pdf/2606.04118)**

> **作者:** Michael Zichert; Arno Simons
>
> **备注:** 19 pages, chapter in the book Understanding Science with Large Language Models? (pp. 383-412). transcript. Edited by Arno Simons, Adrian Wüthrich, Michael Zichert, Gerd Graßhoff (eds.)
>
> **摘要:** This article situates large language models (LLMs) within the longer history of computational approaches to concept analysis in the history, philosophy, and sociology of science (HPSS). We examine what LLMs add to existing methods, how they inherit longstanding problems, and review recent case studies that employ them. In the first part, we reconstruct computational conceptual history before LLMs by bringing together three strands of work: early digital methods in HPSS, distributional approaches from digital history and related research, and lexical semantic change detection. We provide an overview of the main challenges and opportunities, focusing on corpus construction, operationalization and modelling choices, and evaluation and interpretation. In the second part, we turn to the era of LLMs, starting with a short introduction to LLMs before reviewing LLM-based work on lexical semantic change detection and relevant case studies in HPSS. We then revisit the earlier methodological questions, showing how issues of corpus construction, model choice and training data, operationalization trade-offs, and evaluation and interpretation play out in LLM-based workflows.
>
---
#### [new 016] Imbuing Large Language Models with Bidirectional Logic for Robust Chain Repair
- **分类: cs.CL; cs.SC**

- **简介: 该论文提出TRI框架，解决大语言模型推理中错误累积问题。通过双向逻辑修复，提升推理鲁棒性，减少token消耗。**

- **链接: [https://arxiv.org/pdf/2606.05030](https://arxiv.org/pdf/2606.05030)**

> **作者:** Zehua Cheng; Wei Dai; Jiahao Sun; Thomas Lukasiewicz
>
> **备注:** 25 Pages
>
> **摘要:** Autoregressive chain-of-thought (CoT) reasoning in large language models (LLMs) is fundamentally forward-directed: each step conditions only on prior tokens. This unidirectional inductive bias renders even capable models susceptible to error snowballing, wherein a single logical or arithmetic mistake in an early step irreversibly corrupts the entire reasoning chain. We introduce Teleological Reasoning Infilling (\TRI{}), a training framework that endows decoder-only transformers with a native \emph{goal-conditioned bridging} capability. The key insight is to reframe erroneous reasoning segments as fill-in-the-middle (FIM) tasks: given a verified prefix premise $P$, a verified downstream milestone $S$, and the original query $Q$, the model must synthesise the logical bridge $M$ that connects $P$ to $S$ rigorously and completely. To achieve this with standard causal architectures, we introduce a Prefix-Suffix-Middle (PSM) sequence rearrangement with three non-overlapping sentinel tokens, enabling $M$ to attend to both $P$ and $S$ without any structural modification to the self-attention mechanism. Training proceeds in two stages: (i) Supervised Fine-Tuning (SFT) on symbolically verified $(P, S, M)$ triples extracted from formal mathematics corpora, and (ii) Direct Preference Optimisation (DPO) with a deterministic symbolic verifier (Lean 4 / Python) as the sole reward oracle, eliminating LLM-judge sycophancy. At inference, TRI operates as a surgical repair module within a dual-system loop: a causal draft model generates an initial trace, the verifier pinpoints failures, and TRI infills only the damaged segment, leaving verified sections intact. Comprehensive experiments on three benchmarks demonstrate that TRI achieves state-of-the-art performance across all tasks, while reducing per-problem token expenditure by 31.2%.
>
---
#### [new 017] 'Your AI Text is not Mine': Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有检测方法标准不统一的问题。通过定义不同AI生成文本概念，构建新基准数据集AITDNA，并评估检测模型性能。**

- **链接: [https://arxiv.org/pdf/2606.04906](https://arxiv.org/pdf/2606.04906)**

> **作者:** Nils Dycke; Marina Sakharova; Nico Daheim; Iryna Gurevych
>
> **摘要:** Although it is generally agreed that AI-generated text poses a broad societal risk, there is no common understanding in the AI-generated text detection literature on what constitutes harmful use. Rather, existing datasets and approaches often define their own criteria and make their own assumptions, sometimes implicitly, and often only loosely related to real-world needs and applications. To address this gap, we here systematically define various notions of AI-generated text and their characteristics. To study these, we collect AITDNA - a new benchmark of human-machine co-constructed texts that is annotated with detailed genesis information, such as the entire edit and AI-interaction history. We benchmark various machine-generated text detectors and find that they often only perform well for specific notions but not as broad detectors. We release code and data publicly.
>
---
#### [new 018] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于格式约束生成任务，解决固定锚点导致生成不灵活的问题。提出动态填充锚点方法，提升生成结构正确性和语义连贯性。**

- **链接: [https://arxiv.org/pdf/2606.04535](https://arxiv.org/pdf/2606.04535)**

> **作者:** Boyan Han; Yiwei Wang; Yi Song; Yujun Cai; Chi Zhang
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Diffusion large language models (dLLMs) offer bidirectional attention and parallel generation, enabling them to exploit global context and naturally support format-constrained tasks like parseable JSON or reasoning templates. While straightforward fixed anchors can enforce such constraints, they often impose rigid spans, leading to truncated reasoning or redundant content. To overcome this, we propose Dynamic Infilling Anchors (DIA), a training-free method that dynamically estimates end-anchor positions to adjust generation length before iterative infilling. This flexible mechanism ensures structural correctness and semantic coherence, avoiding the inefficiencies of fixed-span methods. Experiments on reasoning benchmarks demonstrate that DIA substantially improves format compliance and answer accuracy, achieving significant zero-shot gains on GSM8K and MATH. These results establish DIA as a robust pathway toward reliable, structure-aware generation.
>
---
#### [new 019] Listening to the Workforce: Measuring Construction Worker Safety Attitudes from Social Media Discourse Using LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于安全态度分析任务，旨在测量建筑工人安全态度。通过构建框架并应用LLM，实现大规模、准确的在线话语分析。**

- **链接: [https://arxiv.org/pdf/2606.04450](https://arxiv.org/pdf/2606.04450)**

> **作者:** Farouq Sammour; Yuxin Zhang; Zhenyu Zhang
>
> **摘要:** Worker safety attitudes are key determinants of whether protective practices are applied or bypassed on construction sites. Yet measuring them at scale has remained out of reach. Safety attitudes are multidimensional, vary across topics, and surface most candidly in workers' own conversations. This study created and validated the Construction Safety Attitude Framework (CSAF), which integrates two components: a theory-grounded structure that characterizes safety attitudes along eight dimensions, and an operational codebook for measuring them in worker naturalistic discourse. Applying CSAF to 250 posts and comments from the r/Construction community on Reddit, trained coders reached strong agreement (Krippendorff's {\alpha} = 0.85). Pairwise lift and conditional probability confirmed that the eight dimensions are related yet distinct. To apply the framework across large volumes of discourse, CSAF was operationalized through a large language model (LLM) classifier. On 450 r/Construction contributions, the classifier reproduced expert human coding (Cohen's \k{appa} = 0.90, precision = 0.98, recall = 0.98), and on 400 contributions from r/Roofing it retained that accuracy after transfer to a different trade community (\k{appa} = 0.89, precision = 0.98, recall = 0.97). A proof-of-value case study then applied the validated classifier to 10,346 contributions from r/Roofing, demonstrating that CSAF can distinguish multidimensional attitudes by safety topic, track how they shift over time, and trace the reasoning behind unfavorable ones. The study therefore provides a theoretically grounded, empirically vetted instrument for examining safety attitudes, offering a basis for targeted interventions that address the attitudes underlying unsafe practices.
>
---
#### [new 020] SaliMory: Orchestrating Cognitive Memory for Conversational Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SALIMORY框架，解决对话代理的持久记忆管理问题。通过分层奖励和对比优化，提升记忆操作效果，提高个性化水平与准确性。**

- **链接: [https://arxiv.org/pdf/2606.04120](https://arxiv.org/pdf/2606.04120)**

> **作者:** Kai Zhang; Xinyuan Zhang; Hongda Jiang; Shiun-Zu Kuo; Hyokun Yun; Ejaz Ahmed; Shereen Oraby; Ziyun Li; Sanat Sharma; Ann Lee; Ahmed A Aly; Anuj Kumar; Raffay Hamid; Xin Luna Dong
>
> **摘要:** Conversational agents that serve as lifelong companions must maintain persistent memory across all interactions. However, simply expanding context windows with raw retrieval degrades reasoning quality, while training memory agents via standard reinforcement learning creates a severe credit assignment bottleneck in a multi-stage pipeline. To solve this, we introduce SALIMORY, a framework that trains a single language model to manage a cognitively-structured memory-spanning user facts, preferences, and working memory. By introducing a hierarchical stage-wise process reward and reward-decomposed contrastive refinement, SALIMORY provides isolated supervision for distinct memory operations (selective filtering, consolidation, and cue-driven recall) end-to-end. SALIMORY cuts memory-attributed failures by one-third, outperforms the state-of-the-art by over 10% in end-to-end accuracy, and more than doubles the Good Personalization rate.
>
---
#### [new 021] DuDi: Dual-Signal Distillation with Cross-Lingual Verbalizer
- **分类: cs.CL**

- **简介: 该论文提出DuDi框架，解决小语言模型在多语言任务中的性能下降问题，通过双信号蒸馏和跨语言verbalizer提升模型效果。**

- **链接: [https://arxiv.org/pdf/2606.04694](https://arxiv.org/pdf/2606.04694)**

> **作者:** Patomporn Payoungkhamdee; Tinnakit Udsa; Jian Gang Ngui; Sarana Nutanong; Alham Fikri Aji; Peerat Limkonchotiwat
>
> **摘要:** Small language models (SLMs) are efficient and scalable, but their multilingual capabilities degrade severely at sub-billion scales, especially for Southeast Asian (SEA) languages. We introduce DuDi, a dual-signal multilingual distillation framework that combines an online sequence-level signal with off-policy and on-policy token-level signals. DuDi further uses a cross-lingual verbalizer to refine teacher feedback and improve teacher-student transferability in multilingual settings. Experiments on SEA-HELM across multiple model families, scales, and teacher-student settings show that DuDi consistently outperforms competitive distillation baselines. Ablations and analyses confirm that sequence-level optimization, token-level supervision, and cross-lingual verbalization provide complementary and transferable learning signals for multilingual SLMs.
>
---
#### [new 022] ACAT: A Collaborative Platform for Efficient Aspect-Based Sentiment Dataset Annotation
- **分类: cs.CL**

- **简介: 该论文提出ACAT，一个用于高效方面情感数据标注的协作平台，解决多标注者数据整合与一致性计算问题，支持四种ABSA任务，并提供自动化ETL流程。**

- **链接: [https://arxiv.org/pdf/2606.04189](https://arxiv.org/pdf/2606.04189)**

> **作者:** Ana-Maria Luisa Mocanu; Ciprian-Octavian Truica; Elena-Simona Apostol
>
> **备注:** Accepted at The 28th International Conference on Big Data Analytics and Knowledge Discovery (DaWak 2026)
>
> **摘要:** Aspect-Based Sentiment Analysis (ABSA) requires high-quality datasets to train reliable models. However, existing annotation tools treat output as flat files, leaving researchers to manually consolidate multi-annotator data, reconstruct relational structures, and compute reliability metrics through custom scripts. This paper introduces ACAT (Aspect-based sentiment analysis Collaborative Annotation Tool), a web-based platform natively supporting four ABSA workflows: (1) Aspect-Category Sentiment Analysis, (2) Clause-Level Segmentation, (3) Aspect-Term Sentiment Analysis with character-level position tracking, and (4) Aspect Sentiment Triplet Extraction with dual span offset preservation. Its core contribution is an automated Extract, Transform, Load (ETL) pipeline that aligns collaborative annotations and computes Inter-Annotator Agreement (IAA) metrics directly at export, yielding training-ready datasets. In a preliminary validation on 1,002 restaurant reviews with two annotators of differing expertise, ACAT achieves a median annotation time of 31.58 seconds and a raw IAA ranging from 0.78 to 0.86 across all tasks.
>
---
#### [new 023] Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于多语言长文本语音指令跟随任务，解决模型过拟合已知任务的问题。工作包括数据增强、标签生成和跨语言翻译，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.04730](https://arxiv.org/pdf/2606.04730)**

> **作者:** Enes Yavuz Ugan; Maike Züfle; Yuka Ko; Supriti Sinhamahapatra; Fabian Retkowski; Seymanur Akti; Jan Niehues; Alexander Waibel
>
> **备注:** 9 pages main paper, IWSLT 2026 Instruction Following track
>
> **摘要:** With the advent of Large Language Models, single-task and token-based multi-task models have evolved into instruction-based systems that infer task and target language implicitly from natural language prompts. This trend is reflected in IWSLT's Instruction Following Track, which this year introduced new tasks including an unknown surprise task, posing a genuine challenge against overfitting to known tasks. We present KIT's submission to the Long and Short Instruction Following tracks in the unconstrained setting. Our approach combines a general data augmentation pipeline that converts short-form corpora into long-form training data through segment concatenation, LLM-based label generation, and cross-lingual translation, yielding over 1M instances across six tasks and four languages. We further show that likelihood-based re-ranking, while highly effective for ASR, systematically degrades semantic tasks by spuriously selecting candidates generated from segmented audio processing rather than holistic long-form inference, a failure mode resolved by combining likelihood with Minimum Bayes Risk decoding.
>
---
#### [new 024] LDARNet: DNA Adaptive Representation Network with Learnable Tokenization for Genomic Modeling
- **分类: cs.CL; q-bio.GN**

- **简介: 该论文提出LDARNet，用于基因组建模，解决固定分词方案限制生物结构的问题。通过自适应分块和学习路由，提升模型性能并揭示生物学意义。**

- **链接: [https://arxiv.org/pdf/2606.04552](https://arxiv.org/pdf/2606.04552)**

> **作者:** Daria Ledneva; Denis Kuznetsov
>
> **摘要:** Genomic foundation models increasingly adopt large language model architectures, yet almost universally rely on fixed tokenization schemes such as $k$-mers, BPE, or single nucleotides, which impose arbitrary sequence boundaries that may obscure biologically relevant structure. We present LDARNet, a 120M-parameter hierarchical genomic foundation model that adapts H-Net-style dynamic chunking from autoregressive generation to masked language modeling, combining BiMamba-2 state-space layers with local attention, bidirectional routing, and a ratio-based regularizer to induce adaptive token boundaries without supervision. Fine-tuned on 27 tasks from the Nucleotide Transformer and Genomic Benchmarks suites, LDARNet achieves 11/18 wins among compact models ($<$300M parameters) and state-of-the-art results on 5 histone modification tasks, outperforming models up to 20$\times$ larger. A FLOPs-matched controlled experiment isolates learned routing as the source of these gains: learned boundaries beat fixed-grid boundaries by up to 14 percentage points on histone tasks at identical compute. Nucleotide-resolution analysis further shows that the learned boundaries align with canonical promoter motifs and splice junctions without supervision, providing a biological interpretation for adaptive tokenization in genomic foundation models.
>
---
#### [new 025] Read the Trace, Steer the Path: Trajectory-Aware Reinforcement Learning for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文提出CAPR算法，用于扩散语言模型的强化学习，通过分析去噪轨迹实现高效路径优化，解决传统方法计算成本高或监督不足的问题。**

- **链接: [https://arxiv.org/pdf/2606.04396](https://arxiv.org/pdf/2606.04396)**

> **作者:** Anant Khandelwal; Manish Gupta
>
> **备注:** 19 pages, 10 figures, 7 Tables
>
> **摘要:** Diffusion large language models (dLLMs) generate responses by iteratively unmasking and revising many positions in parallel. This process leaves a rich denoising trace depicting which tokens become confident, which remain unstable, and when commitments form. Existing dLLM reinforcement learning methods use this signal only weakly. Flat rollouts are cheap, but assign a single outcome reward to the whole trajectory. Tree rollouts provide finer, verifiable training signals by branching partial trajectories and propagating leaf rewards upward, but are compute intensive. We ask whether the denoising trace itself can provide tree-like supervision without tree-level compute. We introduce CAPR (Cached-Amortized Path Refinement), a dLLM-RL algorithm that summarizes the denoising trace into a compact path state, uses cached trajectory states to generate cheap sibling continuations, and trains a block-level value head for local block-wise supervision. Under a block-wise unmasking schedule, CAPR records path-state and block-progress features, then redistributes the final outcome reward across blocks according to the tokens revealed in each block. This trains the value head to convert one sparse reward into block-level PPO weights. CAPR therefore recovers much of the granularity of tree search while avoiding full tree expansion, reducing rollout-generation cost to roughly 0.75x that of flat rollouts and 0.6x that of tree rollouts (under standard settings). Across 4x4 Sudoku, Countdown, GSM8K, and Math500, on dense and mixture-of-experts LLaDA backbones, CAPR sets a new state of the art for RL-tuned dLLMs at 256- and 512-token budgets. On Sudoku, it matches the strongest tree-structured baseline at less than one third of the per-step compute.
>
---
#### [new 026] SANE Schema-aware Natural-language Evaluation of Biological Data
- **分类: cs.CL**

- **简介: 该论文提出SANE，一种用于生物数据的自然语言到SQL评估框架，解决非专家访问结构化数据库的问题。通过schema-aware方法提升查询准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2606.04500](https://arxiv.org/pdf/2606.04500)**

> **作者:** Rolf Gattung; Martin Krueger; Markus Reischl
>
> **备注:** 5 pages, 3 figures, submitted but not yet reviewed by BMT2026
>
> **摘要:** High-throughput microscopy generates large, structured datasets capturing cellular responses to pharmacological perturbations, but accessing these datasets typically requires SQL expertise. Large language models offer a natural-language alternative, yet their tendency to hallucinate raises concerns about result reliability . We present SANE Schema-Aware Natural-language Evaluation, a novel paradigm for domain-specific text-to-SQL evaluation: schema-grounded, automatically generated benchmarks tied to real and specific experimental structure. SANE makes evaluation more scalable, systematic, and reproducible. Using SANE, we evaluate a few-shot large language model and show that, under constrained schemas with structured prompting and guardrails, accurate query generation is achievable without any model training or fine-tuning. Most failures stem from ambiguous or underspecified inputs and manifest as overly cautious clarification requests or answers to queries that should first be disambiguated, rather than incorrect SQL generation. These results indicate that few-shot large language models can provide reliable database access in well-defined domains when combined with schema-aware prompting.
>
---
#### [new 027] Hybrid Adversarial Defence for Natural Language Understanding Tasks
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决大语言模型的幻觉和对抗攻击问题。通过结合熵、不确定性与几何特征，提出一种混合防御框架，提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.04612](https://arxiv.org/pdf/2606.04612)**

> **作者:** Manar Abouzaid; Yang Wang; Chenghua Lin; Stuart E. Middleton
>
> **摘要:** Large Language Models (LLMs) are vulnerable both to hallucination and adversarial manipulation. Although these problems are closely related, existing defences typically address them separately. We investigate a hybrid defence framework that combines entropy-based models, designed to reduce hallucinations, with uncertainty-based models and geometric-based models, designed to reduce vulnerability. Under in-domain tests on Natural Language Understanding datasets (FEVER, HotpotQA, CSQA, SIQA) we find our hybrid model improves both clean-task performance (up to 43.34\% increase in accuracy) and adversarial robustness (up to 64.92\% improvement in accuracy and 62.27\% reduction in attack success rate). For out-of-distribution datasets (AeroEngQA, CPIQA) we see similar adversarial robustness from our hybrid model (up to 57.14\% improvement in accuracy). For prompt injection (SafeGuard) and jailbreak detection (AdvBench, DAN) datasets our hybrid model is also very strong (up to 51\% reduction in attack success rate compared to state of the art baseline models). Overall, our results show that combining entropy, uncertainty and geometric features provides a more effective defence strategy than using any single feature alone for both in-domain and out-of-distribution tasks.
>
---
#### [new 028] Can Crowdsourcing Survive the LLM Era? A Community Survey on Human Data Collection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLM时代下众包数据收集的有效性。研究调查了155位学者，分析LLM使用对数据质量的影响及应对策略。**

- **链接: [https://arxiv.org/pdf/2606.04924](https://arxiv.org/pdf/2606.04924)**

> **作者:** Aswathy Velutharambath; Neele Falk; Sofie Labat; Tarun Tater; Amelie Wuehrl
>
> **摘要:** The widespread use of Large Language Models (LLMs) as writing tools challenges the validity of crowdsourced data, as crowdworkers may outsource tasks to models. To better understand how this is addressed, we surveyed 155 researchers in NLP and related disciplines about their experiences and opinions on collecting free-text responses via crowdsourcing. This paper provides an overview of practitioners' challenges, mitigation strategies, and the foreseen implications on data quality. 44% of respondents reported observing LLM usage in their crowdsourced data. While 93% of them had anticipated this, half were unsure what precautions to take. The most prevalent detection strategies are distinctive textual style patterns and unusually fast completion times. Overall, survey responses show that the research community is aware of the problem and taking measures, but existing efforts remain insufficient to fully address it. Finally, we derive a set of considerations to guide future crowdsourced free-text data collection in the era of LLMs.
>
---
#### [new 029] When Retrieval Doesn't Help: A Large-Scale Study of Biomedical RAG
- **分类: cs.CL**

- **简介: 该论文属于医学问答任务，研究RAG在生物医学领域的效果。发现检索带来的提升有限，模型自身能力是主要瓶颈。**

- **链接: [https://arxiv.org/pdf/2606.04127](https://arxiv.org/pdf/2606.04127)**

> **作者:** Erfan Nourbakhsh; Rocky Slavin; Ke Yang; Anthony Rios
>
> **备注:** 9 Pages, accepted to BioNLP Workshop at ACL 2026
>
> **摘要:** Medical question answering is a high-stakes setting where factual errors can have serious consequences. Retrieval-augmented generation (RAG) is widely viewed as a promising solution, and prior work has reported substantial gains for large medical QA models. We revisit this assumption across a broad range of open-weight instruction-tuned models spanning 7B to 72B parameters. Across five models, ten biomedical QA datasets, four retrieval methods, and four retrieval corpora, we find that retrieval yields only small and inconsistent improvements over a no-retrieval baseline, typically within 1-2 points. In contrast, the choice of backbone model has a much larger effect than the choice of retriever or corpus, and expert and layman retrieval sources perform similarly in most settings. These results suggest that the main bottleneck is not retrieval quality alone, but the model's limited ability to use retrieved evidence effectively.
>
---
#### [new 030] Fine-grained Fragment Retrieval in Multi-modal Long-form Dialogues
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出细粒度片段检索（FFR），解决多模态长对话中相关段落的检索问题。通过F2RVLM和FFFRS模型，提升单对话和跨对话集的检索效果。**

- **链接: [https://arxiv.org/pdf/2606.04591](https://arxiv.org/pdf/2606.04591)**

> **作者:** Hanbo Bi; Zhiqiang Yuan; Chongyang Li; Qiwei Yan; Zexi Jia; Jiapei Zhang; Xiaoyue Duan; Yingchao Feng; Jinchao Zhang; Jie Zhou
>
> **摘要:** With the widespread adoption of multi-modal communication platforms, long-form dialogues interleaving text and images have become increasingly common. Users often need to retrieve coherent dialogue fragments related to specific topics, rather than isolated utterances. We propose Fine-grained Fragment Retrieval (FFR), which locates semantically relevant multi-utterance, multi-image fragments in multi-modal long-form dialogues. We explore two settings: (1) FFR within Single-Dialogue, retrieving fragments from a given dialogue; and (2) FFR within Dialogue Corpus, retrieving from a large-scale corpus for open-domain scenarios. For (1), we introduce F2RVLM, a generation-based retrieval model trained with reinforcement learning, using multi-objective rewards and difficulty-aware curriculum sampling to enhance fragment coherence. For (2), we develop FFRS, a two-stage system combining offline fragment-level indexing with online retrieval. Specifically, each dialogue is decomposed into minimal semantic fragments encoded by a Fragment Embedding Model (FEM) into a vector database; at inference, FEM rapidly recalls Top-K candidates, and F2RVLM performs fine-grained reasoning to identify the most relevant sub-content. To support FFR, we construct MLDR, the longest multi-modal dialogue retrieval dataset to date, and a WeChat-based real-world test set. Experiments on both benchmarks demonstrate that F2RVLM and FFRS consistently achieve superior performance across single-dialogue and corpus-level FFR.
>
---
#### [new 031] TIDE: Proactive Multi-Problem Discovery via Template-Guided Iteration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TIDE框架，用于主动发现文档和代码中的多个隐藏问题。任务是多问题发现，解决传统方法仅响应显式请求的局限性。通过迭代机制和思维模板，提升问题识别与解决效果。**

- **链接: [https://arxiv.org/pdf/2606.04743](https://arxiv.org/pdf/2606.04743)**

> **作者:** Soyeong Jeong; Jinheon Baek; Minki Kang; Sung Ju Hwang
>
> **摘要:** Agents are widely deployed as assistants over documents, tools, and code. However, they typically act only on explicit user requests, which surface only the problems the user has noticed, while many other important problems coexist, hidden in plain sight, within the broader user context, with their total number unknown in advance. We frame this as the task of discovering multiple hidden problems from context, in which coexisting problems should be uncovered, grounded in supporting evidence, and paired with concrete actions. To this end, we introduce TIDE, a template-guided iterative framework with two complementary mechanisms. Specifically, motivated by the observation that single-pass prediction anchors on the most salient cases and yields generic claims, we propose iterative discovery, which surfaces a small batch of candidates per round while conditioning on what has already been found, so subsequent rounds extend coverage; and thought templates, reusable schemas distilled from previously solved cases that specify what contextual signals to attend to and how to connect them, anchoring each prediction in a recognizable problem class. We validate TIDE on two realistic settings, personal workspaces and software repositories, across four model backbones, showing substantial gains over single-shot and parallel multi-agent baselines on task coverage, identification, and resolution.
>
---
#### [new 032] A Systematic Analysis of Linguistic Features in AI-Generated Text Detection Across Domains and Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决不同模型和领域下语言特征的可靠性问题。通过大规模分析284个语言特征，验证了词汇丰富性作为稳健指标的有效性。**

- **链接: [https://arxiv.org/pdf/2606.04177](https://arxiv.org/pdf/2606.04177)**

> **作者:** Yassir El Attar; Esra Dönmez; Maximilian Maurer; Agnieszka Falenska
>
> **备注:** preprint
>
> **摘要:** Interpretable linguistic features offer a promising approach for explaining why a given text appears machine-generated, particularly for non-expert users. However, existing findings on which features reliably indicate LLM-generated text remain fragmented across feature sets, models, and text domains. To address this gap, we conduct a large-scale empirical study assessing the robustness of linguistic signals for characterizing AI-generated text. Our analysis covers 284 interpretable linguistic features across outputs from 27 LLMs and ten text domains under cross-model and cross-domain generalization settings. We show that classifiers based solely on linguistic features can reliably distinguish AI-generated from human-written text. However, many previously proposed indicators prove strongly context-dependent, with the exception of measures of lexical richness, which remain robust signals across model families and text domains. These results demonstrate which linguistic signals generalize across contexts and provide a foundation for more reliable, interpretable analyses of AI-generated language.
>
---
#### [new 033] Large Language Models in K-12 Education: Alignment with State Curriculum Standards and Student Personas
- **分类: cs.CL**

- **简介: 该论文属于教育技术领域，研究LLM在K-12教育中的应用。旨在解决LLM是否符合州级课程标准及学生角色适应性问题。通过构建管道分析课程差异，并测试模型对用户特征的敏感性。**

- **链接: [https://arxiv.org/pdf/2606.04846](https://arxiv.org/pdf/2606.04846)**

> **作者:** Lisa Korver; Tomo Lazovich; Sherief Reda
>
> **摘要:** As Large Language Models (LLMs) become increasingly popular in educational settings, they raise important questions about the ethical implications of their use. Publicly available online chatbots are quickly improving in capability and accuracy leading to more widespread use, including among students looking for help with their homework. This makes it crucial to consider whether these models are aligned with educational standards. Because curriculum standards in the United States are set at the state level, they differ significantly in required content, emphasis, and narrative focus. In this work, we develop an LLM-based pipeline to identify variations in U.S. History curricula across states and evaluate the extent to which different LLMs reflect these state-specific curricular differences. In addition, we conduct controlled experiments that vary user personas by stating user attributes such as geographic location, grade level, gender and race to evaluate the sensitivity of LLM responses to user characteristics. We find that while models are able to adjust their presentation of historical topics, these shifts may come from the perceived political leanings of states and do not necessarily reflect actual curriculum content. Additionally, models successfully adapt to a student's grade level while showing minimal sensitivity to race or gender, suggesting they are capable of useful adaptation to student personas with limited demographic bias. Together, these findings highlight potential risks that open access to LLM chatbots may cause to student learning outcomes stemming from misalignment with state curriculum standards and highlight the need for more robust alignment techniques.
>
---
#### [new 034] CYGNET: Cypher Gate for Neural Execution Triage and Cost Containment
- **分类: cs.CL; cs.DB**

- **简介: 该论文提出CYGNET系统，解决知识图谱中Cypher查询生成的结构和语义错误问题，通过预执行验证和纠错机制提升查询安全性与准确性。**

- **链接: [https://arxiv.org/pdf/2606.04645](https://arxiv.org/pdf/2606.04645)**

> **作者:** Nikodem Tomczak
>
> **摘要:** Language models acting as agents over knowledge graphs generate Cypher queries that fail structurally (crashing at the database) or semantically (executing but returning wrong results). We place a pre-execution gate between query generation and a production Neo4j database. The gate validates structure through a four-backend chain culminating in execution against a mirror graph at 5.6 ms median latency. Structurally broken queries are routed to a corrector that iterates structured error feedback through a language model. On seven CypherBench schemas (2348 questions, ACL 2025) the pipeline maintains generation accuracy on every model tested, confirming it operates as a safe defensive layer. The corrector achieves 81% to 95% success across five models (mean 89%). On a template-generated corpus across nine schemas the gate catches 100% of parse errors, 100% of constraint violations, and 100% of schema-reference errors in path queries with labelled endpoints, at zero false positives across 1135 queries. Property sibling-swaps where the substituted name is valid on the target label score 0%, marking the formal boundary where structural validation ends and semantic validation must begin. A planner-based cost gate flags catastrophic plan structures before execution.
>
---
#### [new 035] Depth-Attention: Cross-Layer Value Mixing for Language Models
- **分类: cs.CL**

- **简介: 该论文提出Depth-Attention机制，解决Transformer模型中跨层信息选择问题，通过在注意力模块内混合不同层的值，提升语言模型性能。**

- **链接: [https://arxiv.org/pdf/2606.05014](https://arxiv.org/pdf/2606.05014)**

> **作者:** Boyi Zeng; Yiqin Hao; Zitong Wang; Shixiang Song; He Li; Feichen Song; Yifan Liu; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **备注:** 21 pages, 4 figures, 9 tables
>
> **摘要:** Self-attention selects information freely across the sequence, but across depth, Transformers merely add each layer's output to the residual stream, so later layers cannot selectively reuse earlier-layer representations. Recent cross-layer methods improve this flow but operate on hidden states outside attention, adding state beyond the key-value cache at inference--a cost that becomes increasingly salient as modern LLMs compress the cache with grouped-query and multi-head latent attention. We introduce Depth-Attention, which performs this selection inside the attention module itself: before a layer attends over the sequence, its query attends over the keys of earlier layers at the same token position and mixes their values into the value that self-attention then reads. Because Depth-Attention reuses the standard attention queries, keys, and value-cache slots, storing depth-mixed values in place of the original values, it adds no parameters and introduces no persistent inference state beyond the standard key-value cache--the same cache size as a vanilla decoder and less than hidden-state-based cross-layer methods. On Qwen3-style decoders at 1.5B and 3B parameters, Depth-Attention attains the lowest perplexity and the highest average downstream accuracy, improving over the vanilla Transformer by up to 2.3 accuracy points and surpassing strong cross-layer baselines in perplexity and average accuracy, while adding under 0.01% extra arithmetic FLOPs and no additional persistent inference state. The gains hold from 360M to 3B parameters and extend to looped Transformers.
>
---
#### [new 036] Supportive Token Revealing for Fast Diffusion Language Model Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决扩散语言模型解码中的质量与速度权衡问题。提出AXON模块，通过选择关键锚点提升解码效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.04236](https://arxiv.org/pdf/2606.04236)**

> **作者:** Giries Abu Ayoub; Mario Barbara; Lluís Pastor-Pérez; Tanja Bien; Aneesh Barthakur; Alaa Maalouf; Loay Mualem
>
> **摘要:** Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is needed. It then shifts the criterion from which tokens are safest to reveal to which confident reveals would best support later denoising. AXON selects anchors, confident masked tokens that uncertain positions attend to, using attention, uncertainty, and confidence signals. Experiments on reasoning and code-generation benchmarks across multiple diffusion language models show that AXON improves the quality-latency trade-off of existing parallel decoders, often reducing the number of function evaluations while maintaining or improving accuracy.
>
---
#### [new 037] Deliberate Evolution: Agentic Reasoning for Sample-Efficient Symbolic Regression with LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于符号回归任务，旨在解决LLM在符号回归中样本效率低的问题。通过提出Deliberate Evolution框架，分离生成与搜索控制，提升性能并减少样本需求。**

- **链接: [https://arxiv.org/pdf/2606.04360](https://arxiv.org/pdf/2606.04360)**

> **作者:** Xinyu Pang; Zhanke Zhou; Xuan Li; Fangrui Lv; Shanshan Wei; Sen Cui; Bo Han; Changshui Zhang
>
> **备注:** ICML 2026
>
> **摘要:** Symbolic regression (SR) discovers compact mathematical expressions from data, yet recent LLM-based evolutionary methods remain sample-inefficient because they rely mainly on scalar feedback such as MSE. We identify a core limitation: existing methods conflate candidate proposal with search guidance, requiring the LLM to infer how to evolve an expression, diagnose its errors, and reuse past experience from a single score. To address this, we propose Deliberate Evolution (DE), an agentic framework that decouples symbolic generation from search control. DE guides LLM proposals with adaptive operators for search direction, analytical tools for structural diagnosis, and reflective memory for trajectory-level experience. Experiments on LLM-SRBench show that DE consistently outperforms representative LLM-based SR baselines across diverse scientific domains while using only 40% of the standard sample budget.
>
---
#### [new 038] Self-Evolving Deep Research via Joint Generation and Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于深度研究生成任务，解决无明确标准的评估难题。提出SCORE框架，通过联合生成与评估实现模型自进化。**

- **链接: [https://arxiv.org/pdf/2606.04507](https://arxiv.org/pdf/2606.04507)**

> **作者:** Han Zhu; Chengkun Cai; Yuanfeng Song; Xing Chen; Sirui Han; Yike Guo
>
> **摘要:** Large Language Models (LLMs) have become increasingly adopted in daily applications, with deep research standing out as a particularly important capability. Unlike traditional question-answering (QA) tasks, deep research report generation lacks definitive ground-truth, making reward design inherently unverifiable and limiting effective reinforcement learning. Existing approaches mitigate this challenge with LLM-as-a-judge and query-dependent evaluation rubrics, but they still rely on static evaluators that cannot adapt their standards as the solver improves, leading to insufficient and eventually saturated optimization pressure. We address this limitation with a \textbf{s}elf-evolving \textbf{co}-evolutionary training framework for deep \textbf{re}search evaluation and generation (SCORE), which tightly couples an evaluator and a solver in a shared-parameter learning process. Rather than treating generation and evaluation as isolated modules, we leverage their intrinsic connection to enable joint improvement within a single shared-parameter model. To restrict this process, we introduce a meta-harness, which dynamically controls the evaluation environment based on solver performance, encouraging valid evaluation dimensions and sufficiently deep evaluator search. Extensive experiments on deep research benchmarks demonstrate consistent improvement in report generation quality, showing that co-evolving evaluation and generation is a promising direction for training open-ended research agents.
>
---
#### [new 039] VCIFBench: Evaluating Complex Instruction Following for Video Understanding
- **分类: cs.CL**

- **简介: 该论文提出VCIFBench，用于评估视频理解中的复杂指令遵循能力。针对现有基准不足，构建了多约束指令集，并验证模型输出，以提升指令遵循性能。**

- **链接: [https://arxiv.org/pdf/2606.04588](https://arxiv.org/pdf/2606.04588)**

> **作者:** Huangchen Xu; Yuan Wu; Yi Chang
>
> **摘要:** Multimodal large language models have made rapid progress in video understanding, yet existing benchmarks largely rely on simple prompts and provide limited evidence about whether models can satisfy explicit output constraints. We introduce VCIFBench, a benchmark for evaluating complex instruction following in video understanding. VCIFBench constructs constraint-rich instructions from both benchmark-adapted and directly video-grounded prompts, covering content, format, style, and structure requirements, and evaluates model outputs with a hybrid verification pipeline. The benchmark contains 306 satisfiable test instructions, a 540-pair DPO preference dataset, and a 30-item conflict diagnostic subset. Experiments on 10 MLLMs show that joint constraint satisfaction remains challenging. We further show that DPO training on VCIFBench data can improve instruction-following performance.
>
---
#### [new 040] DLLG: Dynamic Logit-Level Gating of LLM Experts
- **分类: cs.CL**

- **简介: 该论文提出DLLG，解决多专家模型融合问题，通过动态日志级门控实现更稳定的集成，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.04378](https://arxiv.org/pdf/2606.04378)**

> **作者:** Bingnan Li; Zhaoyang Zhang; Xiaoze Liu; Yantao Shen; Shuli Jiang; Shuo Yang; Wei Xia; Zhuowen Tu; Stefano Soatto
>
> **摘要:** Leveraging multiple specialized LLMs can combine complementary strengths, but existing approaches trade adaptability for stability: routing commits prematurely, heuristic ensembling depends on fragile proxies, and parameter merging introduces interference. We propose DLLG (Dynamic Logit-Level Gating), a dynamic logit-level ensembling framework that learns token-level expert fusion from sparse response-level supervision. A lightweight gating module predicts step-wise fusion weights, linking trajectory-level correctness to generation without token-level labels or expert retraining. Across diverse reasoning and code benchmarks, DLLG consistently outperforms strong routing, heuristic ensembling, and parameter-merging baselines across model scales, highlighting learned logit-level fusion as a robust and scalable paradigm for integrating specialized experts.
>
---
#### [new 041] DAR: Deontic Reasoning with Agentic Harnesses
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律推理任务，旨在解决长且复杂的规则集下模型难以准确定位所需规则的问题。提出DAR框架，通过代理交互方式提升推理效果。**

- **链接: [https://arxiv.org/pdf/2606.05009](https://arxiv.org/pdf/2606.05009)**

> **作者:** Guangyao Dou; William Jurayj; Nils Holzenberger; Benjamin Van Durme
>
> **摘要:** Deontic reasoning is the task of answering questions by applying explicit rules and policies to case-specific facts, for example computing tax liability under a statute or determining the outcome of an immigration appeal. A key technical challenge for LLM-based deontic reasoning is that the relevant ruleset can be long and cross-referenced, so models may still fail to locate the rules needed for a particular reasoning step. We introduce Deontic Agentic Reasoning (DAR), an agentic reasoning setup in which the model interacts with the statutes on demand. We evaluate DAR under multiple harnesses on hard subsets of DeonticBench. Across these settings, we find that agentic harnesses can push the frontier on deontic reasoning tasks, but improvements are not uniform: weaker models often degrade on numerical tasks while consuming far more tokens.
>
---
#### [new 042] Can I Take Another Dose? Evaluating LLM Decision-Making Under Temporal Uncertainty in OTC Dosing QA
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在评估大语言模型在OTC药物剂量决策中的表现。研究提出DOSEBENCH基准，分析模型在时间不确定性和约束遵循方面的表现。**

- **链接: [https://arxiv.org/pdf/2606.04262](https://arxiv.org/pdf/2606.04262)**

> **作者:** Maroof Kousar; Yibo Hu
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** Large language models (LLMs) are increasingly used for everyday health questions, including whether a user can safely take another dose of an over-the-counter (OTC) medication. Yet this common safety-relevant setting remains underexplored in existing medical QA evaluations, where correct answers require tracking dose timing, computing rolling 24-hour intake, following product-label constraints, and handling incomplete medication histories. We introduce DOSEBENCH, a focused benchmark of 81 curated OTC dosing scenarios focused on adult acetaminophen and ibuprofen use, with manually annotated gold references. We evaluate four LLMs across repeated runs using metrics for decision correctness, consistency, explanation verifiability, failure types, and confidence-related signals, resulting in 1,620 model responses. Our results show that models frequently struggle with rolling-window reasoning and ambiguity-sensitive cases and that stable or confident-looking responses can still violate dosing constraints. These findings suggest that OTC dosing QA provides a narrow yet practical testbed for evaluating temporal reasoning, constraint following, and safety-relevant uncertainty handling in medical QA.
>
---
#### [new 043] Caliper: Probing Lexical Anchors versus Causal Structure in LLMs
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究大语言模型在因果推理任务中的表现，旨在区分其是否依赖语义结构而非词汇模式。通过引入Caliper方法，验证了模型对词汇锚点的依赖性。**

- **链接: [https://arxiv.org/pdf/2606.04915](https://arxiv.org/pdf/2606.04915)**

> **作者:** Zhenyu Yu; Shuigeng Zhou
>
> **摘要:** Large language models reach 50 to 70% accuracy on causal reasoning benchmarks such as CLadder, but it is unclear whether this reflects structural reasoning or lexical pattern matching. We introduce Caliper, a controlled perturbation that replaces semantic variable names with placeholder tokens while preserving the causal graph and probabilistic specification of each question. Across nine instruction-tuned LLMs from 3.8B to 671B and three causal reasoning benchmarks, lexical anonymization yields robust accuracy drops of +7.6, +27.0, and +11.1 pp on a local 3.8B-14B set, rising to +29.6 and +18.0 pp on CRASS and e-CARE across nine frontier models spanning the 2024-2026 generations. Of 40 engaged model-by-benchmark cells, 39 show a positive gap, and the gap collapses by 17x on CLadder's pseudoword subset. Structured scaffolding and few-shot in-context learning each narrow the gap, but mainly by lowering P0 accuracy on smaller models rather than recovering P1. Current instruction-tuned LLMs, evaluated zero-shot, show little evidence of structural causal reasoning once lexical anchors are removed.
>
---
#### [new 044] Boosting Self-Consistency with Ranking
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决大模型答案选择效率低的问题。提出RISC方法，通过排名机制提升自一致性效果，增强答案准确性。**

- **链接: [https://arxiv.org/pdf/2606.05054](https://arxiv.org/pdf/2606.05054)**

> **作者:** Maria Marina; Daniil Moskovskiy; Sergey Pletenev; Mikhail Salnikov; Alexander Panchenko; Viktor Moskvoretskii
>
> **备注:** 16 pages, 13 figures, accepted at ACL Student Research Workshop 2026
>
> **摘要:** Self-consistency improves large language models by sampling multiple reasoning paths and selecting the most frequent answer, but majority voting often fails to recover correct answers that are already present among the samples. We address this limitation with Ranking-Improved Self-Consistency (RISC), which reformulates answer selection in self-consistency as a ranking problem. Instead of relying on a single uncertainty or confidence signal, RISC uses a lightweight LambdaRank model to score candidate answers with five carefully designed features that capture answer frequency, semantic centrality, and reasoning-trace consistency. We evaluate RISC on three datasets under a range of test-time budgets. Across datasets, RISC consistently achieves a better accuracy-efficiency trade-off than standard self-consistency and strong baselines, with particularly large gains on question answering benchmarks. Further analysis shows that the proposed features are individually useful and, more importantly, complementary, highlighting the value of learning to combine multiple informative signals for test-time answer selection.
>
---
#### [new 045] Discourse-Role Labels as Presentation-Time Variables for Context Use in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型对上下文标签的响应，探讨标签如何影响模型对错误信息的采纳。属于模型行为分析任务，旨在解决标签对上下文利用的影响问题，通过实验验证不同标签的效果差异。**

- **链接: [https://arxiv.org/pdf/2606.04109](https://arxiv.org/pdf/2606.04109)**

> **作者:** Jianguo Zhu
>
> **备注:** Preprint. 1 figure, 9 tables
>
> **摘要:** Context-augmented language model systems often wrap supplied content with labels such as Reference:, Evidence:, Instruction:, Note:, or Example:, but the effect of these labels on reader-model behavior remains underexplored. We introduce a paired fixed-content probe over 500 MMLU-Pro items: each item receives the same misleading answer-bearing assertion under different discourse-role labels, and adoption is measured by whether the model outputs the injected wrong option. Across GPT-5.5, DeepSeek V4 Pro, Llama-3-8B-Instruct, and Qwen2.5-7B-Instruct, Misleading Adoption Rate shifts by 56-84 percentage points. Binding or source-like labels such as Instruction: and Reference: produce high adoption, whereas Example: consistently suppresses it. Paired tests, bootstrap intervals, final-instruction ablations, and Qwen final-step log-probability probes support a label-conditioned candidate preference. Boundary probes show where the effect weakens or persists: arithmetic tasks reduce adoption, passage-shaped external context preserves smaller label gaps, short-answer evaluation rules out option-letter copying, and nested-label conflicts suggest that illustrative framing can delimit adoption scope. A 200-case single-author manual audit confirms that the short-answer contrasts are stable under conservative adjudication. The resulting claim is bounded but practical: context-utilization and reader-side RAG benchmarks should report and control wrapper labels, because presentation choices can change measured reliance on supplied context.
>
---
#### [new 046] When Clients Stop Following: A Cognitive Conceptualization Diagram-driven Framework for Strategic Counseling
- **分类: cs.CL**

- **简介: 该论文属于心理辅导任务，旨在解决LLM在模拟客户中评估失真的问题。提出CARS和STREAMS框架，提升模型在高摩擦互动中的响应能力。**

- **链接: [https://arxiv.org/pdf/2606.04389](https://arxiv.org/pdf/2606.04389)**

> **作者:** Yihao Qin; Junyi Zhao; Changsheng Ma; Yongfeng Tao; Minqiang Yang; Chang Liu; Bin Hu
>
> **摘要:** Large Language Models (LLMs) show promise in psychological counseling, yet existing benchmarks rely heavily on highly cooperative simulated clients. We observe a critical counselor-following phenomenon: these clients often rapidly shift from resistance to compliance after only a few turns, creating an illusion of therapeutic progress and inflating scores under current evaluation protocols through superficial empathy. To address this evaluation mismatch, we propose a Cognitive Behavioral Therapy (CBT)-grounded resistance-aware framework. We introduce CARS, a client simulator that explicitly models dynamic resistance via Cognitive Conceptualization Diagrams (CCDs). We present STREAMS, a dual-module framework that decouples strategic reasoning (Thinker) from response generation (Presenter) and optimizes it via reinforcement learning. We further propose EWTS-MI, an entropy-weighted metric for evaluating responsiveness under high-friction interactions. Experiments across resistant and non-resistant counseling settings validate our findings on evaluation mismatch and demonstrate the effectiveness of resistance-aware training for improving strategic robustness under challenging counseling interactions.
>
---
#### [new 047] TaDA: Calibrated Probe Gating for Task-Domain LoRA Merging
- **分类: cs.CL**

- **简介: 该论文提出TaDA方法，解决任务与领域LoRA适配器融合问题。通过分层门控和子空间合并，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.05016](https://arxiv.org/pdf/2606.05016)**

> **作者:** Huy Quoc To; Fuyi Li; Guangyan Huang; Ming Liu
>
> **摘要:** Combining a task LoRA adapter with a domain LoRA adapter into a single unified model is a practical yet largely unexplored challenge. Existing methods treat both adapters as symmetric peers, applying uniform weights across all layers. We argue that task and domain adapters exhibit a consistent depth-dependent asymmetry across transformer architectures. Domain dominance increases with layer depth, while shallower layers retain stronger task-relevant signals. Motivated by this observation, we propose $\textbf{TaDA}$ ($\textbf{Ta}$sk-$\textbf{D}$omain LoR$\textbf{A}$ Merging), a training-free algorithm that exploits this structure through calibrated probe-guided per-layer gating and per-component subspace-aware merging. The gating assigns individual weights per layer and projection type using a probe signal proved invariant to adapter weight magnitude. The merging discards conflicting singular directions before combining the remaining components. $\textbf{TaDA}$ produces a standard rank-$r$ LoRA adapter with zero inference overhead. On six scientific QA benchmarks with Llama-2-7B, TaDA achieves an average accuracy of 0.452, outperforming DARE-TIES by +3.6 percentage points and obtaining the best result on all six benchmarks. On six image classification benchmarks with ViT-L/16, TaDA reaches 85.9\% average accuracy, improving over the strongest merging baseline while leading in three of the six individual benchmarks.
>
---
#### [new 048] Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究语音大语言模型在逻辑推理中的实体绑定失败问题，提出EA-CoT方法提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2606.04474](https://arxiv.org/pdf/2606.04474)**

> **作者:** Ming-Hao Hsu; Xiaohai Tian; Jun Zhang; Zhizheng Wu
>
> **摘要:** Speech Large Language Models (SLLMs) underperform their text counterparts on complex reasoning. We reveal that this modality gap is not a uniform cognitive deficit. Evaluating three diverse SLLMs, we show speech-to-text (S2T) matches or exceeds text-to-text (T2T) on spatial, syntactic, and factual tasks. However, on logical tasks requiring entity tracking, S2T accuracy collapses to chance. We diagnose this localized degradation as an entity binding failure: continuous speech features cause models to lose precise entity-property associations during implicit reasoning. To resolve this, we propose Entity-Aware Chain-of-Thought (EA-CoT), forcing SLLMs to explicitly enumerate entities and bind them to claims before reasoning. Strikingly, EA-CoT bridges the gap, even when spoken names are misrecognized, yielding up to a 24.4% absolute accuracy improvement. Ablations confirm these gains stem entirely from explicit semantic binding, reframing the gap as a resolvable bottleneck.
>
---
#### [new 049] GENEB: Why Genomic Models Are Hard to Compare
- **分类: cs.CL; cs.LG; q-bio.GN**

- **简介: 该论文属于基因组机器学习领域，旨在解决模型评估不一致的问题。通过构建GENEB基准，对比40个模型在100个任务中的表现，揭示评估方法的局限性。**

- **链接: [https://arxiv.org/pdf/2606.04525](https://arxiv.org/pdf/2606.04525)**

> **作者:** Daria Ledneva; Mikhail Nuridinov; Denis Kuznetsov
>
> **摘要:** Progress in genomic foundation models is difficult to assess due to fragmented benchmarks, incompatible evaluation protocols, and task-specific reporting. As a result, claims of superiority or generality across models are often not directly comparable. We introduce GENEB, a large-scale diagnostic benchmark that evaluates frozen representations from 40 genomic foundation models across 100 tasks spanning 13 functional categories under a unified probing-based protocol, including few-shot regimes. GENEB enables controlled comparison across model scale, architecture, tokenization, and pretraining data while explicitly exposing task-level trade-offs. Our analysis shows that aggregate leaderboards are unstable: model rankings vary sharply across task categories, scale provides only modest and inconsistent gains, and architectural and pretraining alignment frequently outweigh parameter count. These results highlight limitations of current evaluation practices and position GENEB as a reference framework for principled comparison and category-aware model selection in genomic machine learning.
>
---
#### [new 050] Expert-Aware Refusal Steering
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于安全对齐任务，旨在解决LLM拒绝有害请求的能力被抑制的问题。通过改进的专家感知方法，有效引导模型响应有害请求。**

- **链接: [https://arxiv.org/pdf/2606.04160](https://arxiv.org/pdf/2606.04160)**

> **作者:** Anna C. Marbut; Daniel R. Olson; Travis J. Wheeler
>
> **备注:** Under review for COLM 2026
>
> **摘要:** Safety alignment in instruction-tuned large language models (LLMs) depends on a model's ability to reliably refuse to respond to harmful or disallowed requests. Recent work has shown that a steering vector can be applied to a dense LLM during inference to effectively suppress refusal behavior, inducing response to harmful requests. We extend this refusal steering method to three open-source Mixture-of-Experts (MoE) LLMs and find that steering performance is uninhibited by the complex routing patterns inherent to the MoE architecture. We then propose two expert-aware refusal steering methods that leverage refusal-specific expert routing patterns and expert-specific steering directions to suppress normal refusal behavior. We find that refusal behavior can be effectively steered based on the output of a single expert. Our results show that refusal signals captured by steering methods differ from expert routing behavior, suggesting a substantial role for attention in MoE refusal behavior.
>
---
#### [new 051] Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于形式化证明优化任务，旨在解决LLM在Lean中证明时的高成本与低效问题。通过引入动作路由代理，减少失败尝试，降低计算成本。**

- **链接: [https://arxiv.org/pdf/2606.04883](https://arxiv.org/pdf/2606.04883)**

> **作者:** Kári Rögnvaldsson; Chenhao Sun; Jasper Dekoninck; Martin Vechev
>
> **摘要:** Large language models (LLMs) are increasingly used in workflows for generating formal proofs in Lean. These workflows often decompose problems into smaller lemmas, sample many proof attempts, and use compiler feedback to guide search. However, they can be prohibitively expensive, often spending substantial compute on attempts that ultimately fail. In this work, we address this problem with an action routing agent that consists of a data plane and a control plane. The data plane generates natural-language lemma decompositions, formalizes them in Lean, and samples proof attempts for the resulting theorem and lemma targets. The control plane observes previous failed Lean attempts, estimates both the likelihood of success and cost of another attempt, and decides whether to continue proving the current target or restart from a new breakdown. On a subset of PutnamBench, our agent decreases the cost by $25.8\%$ over a fixed-step baseline on average, preserving performance while using substantially less compute. These results suggest that failed Lean trajectories provide actionable signals for cost-aware resource allocation in agentic theorem proving.
>
---
#### [new 052] GARL: Game-Theoretic Reinforcement Learning for Multi-Agent Strategic Prioritisation
- **分类: cs.CL**

- **简介: 该论文提出GARL框架，解决多智能体战略优先级问题。通过博弈论建模交互结构，优化策略决策，提升排名性能与领域适应性。**

- **链接: [https://arxiv.org/pdf/2606.05002](https://arxiv.org/pdf/2606.05002)**

> **作者:** Yuxiao Ye; Yiwen Zhang; Huiyuan Xie; Yuqin Huang; Zhiyuan Liu
>
> **摘要:** LLM-based multi-agent systems are increasingly used for strategic decision-making tasks. In such settings, performance depends not only on individual model capabilities, but also on the policies by which agents interact and adapt. Multi-agent reinforcement learning can optimise these interaction policies, but its reward design often remains task-specific and weakly grounded in interaction structure. To address this gap, we propose GARL, a GAme-theoretic Reinforcement Learning framework for multi-agent strategic prioritisation. GARL formalises strategic prioritisation as a two-stage game: competing agents first allocate strategic resources over a shared candidate set, and a higher-level arbiter then produces the final ranking. The resulting game-theoretic utilities are converted into role-specific reinforcement signals, allowing policy optimisation to be guided by structured interaction. We instantiate GARL on issues-in-dispute ranking, where the goal is to prioritise core issues in legal proceedings. Experiments show that GARL improves ranking performance, enables small open-source LLMs to become competitive with a strong closed-source LLM under the same candidate-ranking setting, and yields gains in legal-domain competence and broader strategic decision-making. Overall, GARL demonstrates how game-theoretic interaction structure can be turned into reinforcement-learning objectives, providing a principled approach to policy optimisation in multi-agent strategic prioritisation.
>
---
#### [new 053] Query-based Cross-Modal Projector Bolstering Mamba Multimodal LLM
- **分类: cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决Mamba模型在视觉-语言建模中的效率问题。通过设计查询式跨模态投影器，提升模型性能与吞吐量。**

- **链接: [https://arxiv.org/pdf/2606.04719](https://arxiv.org/pdf/2606.04719)**

> **作者:** SooHwan Eom; Jay Shim; Gwanhyeong Koo; Haebin Na; Mark A. Hasegawa-Johnson; Sungwoong Kim; Chang D. Yoo
>
> **备注:** Accepted to EMNLP 2024 Findings
>
> **摘要:** The Transformer's quadratic complexity with input length imposes an unsustainable computational load on large language models (LLMs). In contrast, the Selective Scan Structured State-Space Model, or Mamba, addresses this computational challenge effectively. This paper explores a query-based cross-modal projector designed to bolster Mamba's efficiency for vision-language modeling by compressing visual tokens based on input through the cross-attention mechanism. This innovative projector also removes the need for manually designing the 2D scan order of original image features when converting them into an input sequence for Mamba LLM. Experimental results across various vision-language understanding benchmarks show that the proposed cross-modal projector enhances Mamba-based multimodal LLMs, boosting both performance and throughput.
>
---
#### [new 054] Long Live Fine-Tuning: Task-Specific Transformers Outperform Zero-Shot LLMs for Misinformation Response Classification on Reddit
- **分类: cs.CL; cs.CY**

- **简介: 论文研究了Reddit上虚假信息回应分类任务，比较了零样本大模型与微调的Transformer模型。结果表明，微调模型在准确率和成本上更具优势，尤其在识别信念类回应时表现更好。**

- **链接: [https://arxiv.org/pdf/2606.04274](https://arxiv.org/pdf/2606.04274)**

> **作者:** JooYoung Lee; Lin Tian; Angela Brillantes; Adriana-Simona Mihăiţă; Marian-Andrei Rizoiu
>
> **摘要:** As large language models (LLMs) become default tools for online information verification, an implicit assumption follows them: that scale and general capability are sufficient for nuanced classification of misinformation discourse. We test this assumption directly on 900 Reddit comments spanning three PolitiFact-verified misinformation claims (environment, health, immigration), labelled as belief (propagates the claim), fact-check (corrects it), or other. We compare nine models across three paradigms -- BART-MNLI, three Llama variants, three commercial frontier LLMs (Claude Haiku 4.5, Gemini Flash Lite 2.5, Claude Sonnet 4.6), and fine-tuned DistilBERT and RoBERTa -- under universal and topic-specific label schemas. The assumption does not hold. Fine-tuned RoBERTa reaches 0.62 macro-$F_1$ against a best zero-shot result of 0.50 (Claude Haiku 4.5), at a fraction of the per-query cost; the supervised advantage is concentrated on the belief class, the implicit, affective category every zero-shot model under-detects. Scaling does not help: Llama-3-8B matches Llama-3-70B, and Claude Sonnet 4.6 underperforms the smaller Haiku under generic labels, collapsing belief detection to 0.17 and refusing outright on a subset of comments flagged as sensitive. This is a safety-alignment artefact, not a capacity limit. Label schema and topic jointly shape zero-shot performance, with the same model varying by more than 0.13 macro-$F_1$ across topics under matched labels. In a verification context, where missing belief is the costlier error, task-specific fine-tuning remains the more reliable choice despite the proliferation of large generative models.
>
---
#### [new 055] GRAIL: Gradient-Reweighted Advantages for Reinforcement Learning with Verifiable Rewards
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM数学推理中奖励信号分布不均的问题。提出GRAIL方法，通过梯度重加权提升关键token的影响力，无需过程监督即可提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2606.04889](https://arxiv.org/pdf/2606.04889)**

> **作者:** Tej Deep Pala; Vernon Toh; Soujanya Poria
>
> **摘要:** Reinforcement learning with verifiable rewards (e.g. GRPO) is now a common way to improve mathematical reasoning in Large Language Models (LLMs). However, current methods usually broadcast one sequence-level advantage to all tokens, or use costly process reward models (PRMs) for step-level supervision. Uniform advantage distribution assumes that all tokens contribute equally to the final reward. This dilutes the gradient signal, since flawed reasoning steps and filler words are updated as strongly as valid logical inferences. To address this, we introduce Gradient-Reweighted Advantage (GRAIL), an intrinsic token-wise advantage reweighting method. GRAIL uses gradient-activation saliency to place more weight on tokens that are more locally sensitive to the final answer. Evaluations across five models from the Qwen3, R1-distilled and OctoThinker families show that GRAIL consistently outperforms GRPO. GRAIL achieved an average improvement of 3.60% in accuracy and 3.05% in Pass@3, demonstrating that fine-grained reasoning alignment can be achieved without process-level supervision.
>
---
#### [new 056] RAMPART: Registry-based Agentic Memory with Priority-Aware Runtime Transformation
- **分类: cs.CL; cs.MA**

- **简介: 该论文提出RAMPART，一种面向LLM代理的编译时内存模型，解决任务执行中的记忆管理问题，通过块级操作和优先级转换提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2606.04628](https://arxiv.org/pdf/2606.04628)**

> **作者:** Nikodem Tomczak
>
> **摘要:** RAMPART is a compile-time memory model and pure in-RAM block registry for LLM-based agents. Context assembly is a programmable runtime operation where content is compiled from a structured registry under explicit policy for ordering, inclusion, and eviction. Five composable primitives (promote, gate, write, evict, rollback) act on named addressable blocks before compilation at zero prompt-token cost. Provenance tags and non-evictable authorship flags implement a permissioned memory model with block-level ownership. Controlled probes with Qwen3-8B Q4 show that compile-time placement and the structural relationship between blocks and the task query affect task success, with the cliff falling at roughly the seventh block position when the task follows the registry and the twelfth when it precedes. Grouping the critical block with content-adjacent neighbours and promoting the group as a unit lifts task success by tens of percentage points at positions where single-block placement fails. Cross-model replication on Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, and Qwen3-14B shows the content-priming effect appears at the same absolute positions across families, with magnitude varying with model strength. Block grouping raises Mistral's mean pass rate roughly fivefold at the hardest registry size, and a smaller model with the intervention can outperform a larger model without it in the mid-registry zone. Relevance gating reduces prompt cost by 67.8\% while recovering 83% of the promoted-condition success rate. Schema eviction produces 0% invocations against 100% with the schema present, a property policy-based approaches cannot guarantee by construction. Shared-registry coordination reduces inter-agent communication to a method call at zero coordination token cost.
>
---
#### [new 057] Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文属于小语言模型推理后训练任务，解决SFT与RL阶段数据选择不当的问题。提出阶段特定数据集和桥接机制，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2606.04466](https://arxiv.org/pdf/2606.04466)**

> **作者:** Chongyang He; Rui Zhang; Zixuan Wang; Xin Li
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** Post-training Small Language Models (SLMs) for reasoning typically follows an SFT-then-RL pipeline, yet existing work rarely considers what data should be learned at each stage. We argue that data strategy should be aligned with the distinct roles of SFT and RL: SFT is better suited for acquiring not-yet-mastered reasoning skills, while RL is better suited for consolidating skills that the model can already partially access. Based on this principle, we propose a difficulty-aware SFT-then-RL framework that organizes training data into stage-specific sets. For hard samples in the SFT stage, we introduce a Bridge mechanism that transforms raw teacher-generated reasoning traces into more learnable supervision for SLMs. For hard samples that remain unsolved during RL, we apply Critique Fine-Tuning by converting all-zero-reward failures into diagnostic, repair, and new reasoning trace supervision for the next SFT stage. Experiments on two SLMs across five reasoning benchmarks show that our method consistently improves over representative SFT, distillation, and RL baselines. Our results highlight the importance of coordinating data difficulty across SFT and RL for effective SLM reasoning post-training.
>
---
#### [new 058] MM-BizRAG: Rethinking Multimodal Retrieval-Augmented Generation for General Purpose Enterprise Q&A
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于企业问答任务，解决复杂文档结构信息处理不足的问题。提出MM-BizRAG，通过结构感知分割和多模态组装提升问答效果。**

- **链接: [https://arxiv.org/pdf/2606.04231](https://arxiv.org/pdf/2606.04231)**

> **作者:** Hanoz Bhathena; Parin Rajesh Jhaveri; Rohan Mittal; Prateek Singh; Aymen Kallala; Rachneet Kaur; Yiqiao Jin; Zhen Zeng; Adwait Ratnaparkhi; Denis Kochedykov
>
> **备注:** Accepted at ACL 2026 (Industry Track)
>
> **摘要:** Recent advances in multimodal retrieval-augmented generation (MM-RAG) have shifted toward minimal parsing, relying on page-level images for producing retriever embeddings and for answer generation. While efficient, this trend often neglects explicit handling of the rich, structured information in complex enterprise documents, instead depending on pre-trained embeddings or vision-language models to implicitly capture such structure. In this work, we take a more direct approach: MM-BizRAG proactively extracts and represents document structure via a document structure-aware split that dynamically routes documents through orientation-specific ingestion pipelines, applying explicit layout-aware parsing for vertically structured documents (e.g., reports) and holistic page-level representations for horizontally structured documents (e.g., slide decks). A unified LLM-driven artifact transformation pipeline with placeholder-based positional alignment preserves natural reading order, while inference-time multimodal assembly decouples retrieval representations from generation context, enabling richer, more grounded answers without any finetuning requirement. Through experiments on a large, heterogeneous enterprise dataset and two public benchmarks (SlideVQA and FinRAGBench-V), MM-BizRAG consistently outperforms state-of-the-art vision-centric baselines by up to 32% points, with especially strong gains on report-style layouts. Furthermore, we introduce FastRAGEval, a single-call LLM Judge metric for fine-grained generative recall that halves RAGChecker's cost while achieving stronger human alignment.
>
---
#### [new 059] Light or Full Verb? A Minimal-Pair Dataset for Probing Phraseological Competence in Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型句法分析任务，旨在探究语言模型是否区分轻动词和全动词用法。研究构建了最小对句子数据集，并通过实验验证模型的区分能力。**

- **链接: [https://arxiv.org/pdf/2606.05087](https://arxiv.org/pdf/2606.05087)**

> **作者:** Francesca Franzon; Nicolas Rosàs Gómez; Leo Wanner
>
> **摘要:** Frequent English verbs such as 'have' and 'make' can function either as collocates in light-verb constructions or as full lexical predicates, as in 'make a decision' vs. 'make a cake'. Whether language models represent this distinction remains unclear. We introduce a large-scale controlled dataset of minimally varying English sentence series in which the same context contains the same verb in light-verb and full-verb uses. Two probing experiments show that language models differentiate between these uses even in minimal contexts and exhibit separable patterns across object types. We release the dataset, generation code, and materials as a reusable resource. The framework supports extensions to broader contexts, additional verbs, and other languages.
>
---
#### [new 060] Cross-Prompt Generalization in Detecting AI-Generated Fake News Using Interpretable Linguistic Features
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于虚假新闻检测任务，旨在解决AI生成文本在不同提示下的泛化问题。通过提取可解释的语言特征，验证模型在跨提示场景下的有效性。**

- **链接: [https://arxiv.org/pdf/2606.04199](https://arxiv.org/pdf/2606.04199)**

> **作者:** Aya Vera-Jimenez; Samuel Jaeger; Calvin Ibenye; Dhrubajyoti Ghosh
>
> **摘要:** The increasing use of large language models has raised concerns about the spread of AI-generated fake news, particularly under varying prompting strategies. Most existing detection models are trained and evaluated under a single generation setting, leaving their ability to generalize across unseen prompts unclear. In this study, we investigate cross-prompt generalization in fake news detection using three datasets of AI-generated articles produced under distinct prompts, combined with real news articles. We extract interpretable linguistic features capturing lexical diversity, readability, and emotion-based characteristics and evaluate a random forest classifier under a cross-prompt framework, where models trained on one prompt are tested on another. Across all six train-test combinations, performance remains consistently high, with AUC values ranging from 0.988 to 1.000. Analysis of feature distributions shows that AI-generated text exhibits increased lexical diversity, reduced readability, and substantially lower emotional intensity compared to the overall dataset, with variations across prompts. Despite these distributional shifts, the classifier maintains strong performance, indicating that these features capture stable properties of AI-generated text that generalize across prompting strategies. These findings suggest that feature-based approaches can provide robust detection of AI-generated fake news under prompt variability.
>
---
#### [new 061] LifeSide: Benchmarking Agents as Lifelong Digital Companions
- **分类: cs.CL**

- **简介: 该论文属于人工智能中的对话系统任务，旨在解决数字伴侣长期陪伴的问题。提出基准测试框架，评估模型在多轮对话中的记忆、情感与环境适应能力。**

- **链接: [https://arxiv.org/pdf/2606.04660](https://arxiv.org/pdf/2606.04660)**

> **作者:** Yuqian Wu; Zhijie Deng; Wei Chen; Junwei Li; Yutian Jiang; Junle Chen; Zhengjun Huang; Qingxiang Liu; Jing Tang; Jiaheng Wei; Yuxuan Liang
>
> **备注:** 28 pages, 23 figures, 7 tables
>
> **摘要:** Lifelong digital companions must integrate cross-session cues, continually update their understanding of users, and adapt to shifting privacy boundaries. Existing evaluations fail to capture this, testing memory recall and short-term empathy in isolation. To bridge this gap, we introduce \benchmark, a benchmark centered on multi-session \textit{Memory-Emotion-Environment} loops. By modeling users as persistent worlds with layered profiles and event trajectories, \benchmark uses multi-agent simulation to project environmental dynamics into dialogue, preserving the critical gap between latent thoughts and observable expressions. Evaluating 2,000 personas and 111K tasks across memory tracking, user understanding, privacy control, and emotional companionship, our experiment results reveal a stark reality: even models that saturate current memory benchmarks fail to sustain accurate user understanding and true companionship over long horizons.
>
---
#### [new 062] SAID: Accelerating Diffusion-Based Language Models via Scaffold-Aware Iterative Decoding
- **分类: cs.CL**

- **简介: 该论文提出SAID框架，用于加速扩散语言模型的推理。针对扩散模型生成速度慢的问题，通过分阶段解码和动态分配计算资源，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2606.04974](https://arxiv.org/pdf/2606.04974)**

> **作者:** Na Li; Chengda Wang; Mingju Gao; Hao Tang
>
> **备注:** Code: this https URL
>
> **摘要:** Diffusion large language models (DLLMs) enable non-autoregressive generation by iteratively denoising corrupted token sequences with bidirectional context. Despite their ability to update multiple positions in parallel, inference remains costly due to the many denoising steps required for high-quality generation. We propose SAID, a Scaffold-Aware Iterative Decoding framework that accelerates DLLMs by reallocating computation across tokens. SAID first spends denoising computation on scaffold tokens to establish the coarse semantic structure, and then completes predictable detail tokens with fewer steps. We further adapt SAID to block-wise diffusion decoding and introduce Confidence-Hierarchical Layered Generation (CHLG), which assigns additional steps only to low-confidence tokens. Experiments on LLaDA-8B and LLaDA 1.5 across math, coding, and knowledge benchmarks show that SAID significantly accelerates DLLM inference with a maximum speedup of 9.1x while maintaining competitive performance. Our code is publicly available: this https URL.
>
---
#### [new 063] Temporal Order Matters for Agentic Memory: Segment Trees for Long-Horizon Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，解决长时记忆中时间顺序丢失的问题。提出SegTreeMem，通过时间有序的分段树结构提升记忆效果。**

- **链接: [https://arxiv.org/pdf/2606.04555](https://arxiv.org/pdf/2606.04555)**

> **作者:** Yifan Simon Liu; Liam Gallagher; Faeze Moradi Kalarde; Jiazhou Liang; Armin Toroghi; Scott Sanner
>
> **摘要:** Long-horizon conversational agents need to interact with users through evolving events, tasks, and goals. Such histories are naturally temporal, yet many existing memory systems organize information primarily by topical similarity and may ignore the order in which events occur. We introduce Segment Tree Memory, or SegTreeMem, a memory architecture that represents conversation history as a temporally ordered Segment Tree over utterances. SegTreeMem incrementally inserts new utterances through an online rightmost-frontier update rule, preserving chronological order while forming hierarchical memory segments. For retrieval, SegTreeMem propagates relevance scores through the tree to combine local semantic matching with hierarchical temporal context. Across three long-horizon memory benchmarks and two LLM backbones, SegTreeMem improves answer quality over flat retrieval, graph-structured memory, and tree-structured memory baselines. Additional temporal-order permutation analysis shows that the performance gain depends on preserving temporal order during memory construction, supporting the claim that temporal order is a key structure for agentic memory.
>
---
#### [new 064] LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长序列生成中KV缓存效率低的问题。提出LazyAttention机制，实现高效的位置无关KV重用，提升推理速度与吞吐量。**

- **链接: [https://arxiv.org/pdf/2606.04302](https://arxiv.org/pdf/2606.04302)**

> **作者:** Haocheng Xia; Mihir Pamnani; Hanxi Fang; Supawit Chockchowwat; Yongjoo Park
>
> **备注:** ICML 2026
>
> **摘要:** Key-value (KV) caching accelerates inference of large language models (LLMs) by reusing past computations for generated tokens. Its importance becomes even greater in long-context applications such as retrieval-augmented generation (RAG) and in-context learning (ICL). However, conventional KV caching embeds positional information directly into the cache, limiting its reusability. Existing solutions either restrict reuse to prefixes or require expensive memory materialization for positional re-encoding. We introduce LazyAttention, a novel attention mechanism that kernelizes deferred positional encoding to enable zero-copy, position-agnostic KV reuse. By adjusting positional encoding within attention kernels on-the-fly, LazyAttention resolves the materialization bottleneck, allowing a single physical KV copy to serve multiple logical requests at arbitrary positions. Leveraging attention kernels tailored for prefilling and decoding, our system achieves significant efficiency improvements: under skewed document distributions, it reduces time-to-first-token (TTFT) by 1.37$\times$ and increases inference throughput by 1.40$\times$ compared to the state-of-the-art Block-Attention, while maintaining comparable output quality.
>
---
#### [new 065] Activation-Based Active Learning for In-Context Learning: Challenges and Insights
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的主动学习任务，旨在探索基于激活的样本选择方法在上下文学习中的有效性。研究发现，激活信号与样本质量无显著相关性，不适用于此任务。**

- **链接: [https://arxiv.org/pdf/2606.05134](https://arxiv.org/pdf/2606.05134)**

> **作者:** Yaseen M. Osman; Geoff V. Merrett; Stuart E. Middleton
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Deep active learning has previously been explored for LLM in-context sample selection, but not with methods that utilise recent advances in understanding of transformer activations. In this paper, we test the hypothesis that model activations could provide a fine-grained signal to optimise the selection of in-context examples. We present the most comprehensive analysis to date of MLP activation-based deep active learning methods applied to in-context learning, including how different attention masking strategies impact active learning across diverse classification and generative datasets, using both Llama-3.2-3B and Qwen2.5-3B base models. However, we find a negative result: MLP outputs, viewed through the lenses of massive activations or the first four moments, do not correlate with example quality or task performance. Specifically, the absolute Spearman correlation coefficient is at most 0.33 for all tasks and models we tested, showing that such activation-based sampling should not be used for in-context learning. We hypothesise that this may be due to superposition, whereby models represent more features than they have dimensionality, suggesting that methods like Sparse Autoencoders (SAEs) may be a promising future direction.
>
---
#### [new 066] Using Text-Based Causal Inference to Disentangle Factors Influencing Online Review Ratings
- **分类: cs.CL**

- **简介: 该论文属于因果推理任务，旨在解决在线评论中各因素对评分影响的分离问题。通过改进CausalBERT方法，提升估计可靠性，并验证其在K-12学校评价中的有效性。**

- **链接: [https://arxiv.org/pdf/2606.04286](https://arxiv.org/pdf/2606.04286)**

> **作者:** Linsen Li; Aron Culotta; Nicholas Mattei
>
> **备注:** HLT/NAACL 2025
>
> **摘要:** Online reviews provide valuable insights into the perceived quality of facets of a product or service. While aspect-based sentiment analysis has focused on extracting these facets from reviews, there is less work understanding the impact of each aspect on overall perception. This is particularly challenging given correlations among aspects, making it difficult to isolate the effects of each. This paper introduces a methodology based on recent advances in text-based causal analysis, specifically CausalBERT, to disentangle the effect of each factor on overall review ratings. We enhance CausalBERT with three key improvements: temperature scaling for better calibrated treatment assignment estimates; hyperparameter optimization to reduce confound overadjustment; and interpretability methods to characterize discovered confounds. In this work, we treat the textual mentions in reviews as proxies for real-world attributes. We validate our approach on real and semi-synthetic data from over 600K reviews of U.S. K-12 schools. We find that the proposed enhancements result in more reliable estimates, and that perception of school administration and performance on benchmarks are significant drivers of overall school ratings.
>
---
#### [new 067] GlossAssist -- A Tool to Simplify Corpus Creation and Study the Effect of NLP Models in Low-Resource Documentation Settings
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出GlossAssist工具，解决低资源语言文档中人工标注耗时成本高的问题。通过结合CWoMP模型，实现可交互的自动词素标注，并利用反馈优化预测。**

- **链接: [https://arxiv.org/pdf/2606.04367](https://arxiv.org/pdf/2606.04367)**

> **作者:** Bhargav Shandilya; Matt Buchholz; Alexis Palmer
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Interlinear glossed text (IGT) is the standard format for linguistic annotation in language documentation. Producing it manually, however, is often slow and costly. Automated glossing systems have improved substantially in recent years, but adoption among field linguists remains limited. Existing tools are designed to be evaluated rather than used, offering no interpretable path for correction or the incorporation of linguistic expertise back into model behavior. We present GlossAssist, a glossing tool built around the retrieval-based architecture of CWoMP (Contrastive Word-Morpheme Pre-training), which grounds predictions in a mutable lexicon of learned morpheme representations. In conjunction with CWoMP, our system treats each correction by an annotator as part of an active learning setting, which expands the lexicon and improves future predictions without having to retrain the model. In this paper, we present our interface and argue that this feedback loop should be treated as a design requirement for NLP tools aimed at documentary linguists.
>
---
#### [new 068] Evaluating Large Language Models in Dynamic Clinical Decision-Making with Standardized Patient Cases
- **分类: cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLM在动态临床决策中的表现评估问题。通过构建MedSP1000基准，模拟真实临床交互，评估模型表现。**

- **链接: [https://arxiv.org/pdf/2606.05112](https://arxiv.org/pdf/2606.05112)**

> **作者:** Cheng Liang; Pengcheng Qiu; Ya Zhang; Yanfeng Wang; Chaoyi Wu; Weidi Xie
>
> **摘要:** Large language models (LLMs) are increasingly proposed as clinical agents, yet static, single-turn benchmarks cannot capture how a model dynamically delivers care across an encounter: gathering information, planning treatment, and adapting longitudinal management across successive patient states. Medical education has long addressed an analogous challenge through standardized patients (SPs): trained actors who consistently portray clinical cases, enabling realistic practice and objective, scripted assessment. Here we introduce MedSP1000, an SP-derived interactive benchmark for clinical-agent evaluation, including 1,638 SP cases with 24,602 trajectory-level peer-reviewed rubrics. MedSP1000 converts peer-reviewed SP teaching cases into executable scenarios with defined SP case scripts, clinical environment contexts, and human-validated structured rubric. In each simulation evaluation run, a clinical agent interacts in closed loop with a patient agent and an environment controller, and its behaviour is scored throughout the encounter against expert criteria specified in the original materials. Applying MedSP1000 to a range of general-purpose and medically specialized LLMs, we find that performance on static benchmarks does not reliably translate to such educational scenarios. The best-performing model, GPT-5.5, completes only 60.4% of expert-defined rubric items, whereas the strongest medically specialized model reaches 40.0%; increasing test-time compute produces no measurable gain. These results suggest that current LLMs, including agentic systems tuned for medicine, are not yet reliable enough to be safely integrated into actual clinical practice. More broadly, MedSP1000 shows how process-level, SP-style evaluation can reveal clinically relevant failure modes that single-turn benchmarks miss.
>
---
#### [new 069] Self-Evaluation Is Already There: Eliciting Latent Judge Calibration in Base LLMs with Minimal Data
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在让模型预测外部评委对其输出的评分。通过少量数据激活模型的隐含评估能力，提升评估准确性并保持回答质量。**

- **链接: [https://arxiv.org/pdf/2606.05122](https://arxiv.org/pdf/2606.05122)**

> **作者:** XiuYu Zhang; Yi Shan; Junfeng Fang; Zhenkai Liang
>
> **摘要:** Large language models are increasingly evaluated by other models, raising a natural question: can a model predict how a judge will score its own output? We find that the ability is largely present before any targeted training: prompted few-shot, a base model already predicts an external judge's multi-attribute quality scores on open-ended responses well above chance across three benchmarks. We introduce Self-Evaluation Elicitation (SEE), a method that surfaces this latent ability through a short cycle comprising a calibration-coupled reinforcement learning phase that improves the answer and predicts the judge, followed by a masked distillation phase that sharpens the prediction while leaving the answer untouched. From 160 unique examples, roughly 31x fewer than a reinforcement learning baseline, SEE improves held-out calibration across three benchmarks while preserving answer quality. The elicited self-evaluation is sharply localized within the model's own token distribution and stable across judges it was never trained against, indicating a transferable notion of quality rather than a single judge's preference. These results reframe judge-aligned self-evaluation as a problem of elicitation rather than acquisition.
>
---
#### [new 070] Parameter-Efficient Fine-Tuning with Learnable Rank
- **分类: cs.CL**

- **简介: 该论文属于参数高效微调任务，旨在解决固定秩约束限制模型性能的问题。提出LR-LoRA方法，让适配器秩在训练中自适应学习，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2606.04325](https://arxiv.org/pdf/2606.04325)**

> **作者:** Arpit Garg; Simon Lucey; Hemanth Saratchandran
>
> **备注:** In Submission
>
> **摘要:** Low-Rank Adaptation (LoRA) is a popular parameter-efficient fine-tuning (PEFT) method that restricts weight updates to low-rank adapters, introducing a fixed low-rank inductive bias by optimizing in a low-dimensional subspace. In this work, we question whether a fixed-rank constraint is the most effective inductive bias for parameter-efficient fine-tuning. We introduce *Learnable Rank LoRA (LR-LoRA)*, a PEFT method in which the adapter rank is learned during the training process. Instead of prescribing a uniform rank for all adapter layers, LR-LoRA allows the optimizer to determine the appropriate rank for each layer. Using this approach, we find substantial layer-wise variation in the learned ranks, with the attention and MLP layers in the transformer models exhibiting systematically different rank preferences. Across a range of language understanding and commonsense reasoning benchmarks, LR-LoRA achieves state-of-the-art performance in most settings and consistently outperforms strong PEFT baselines, demonstrating that a learnable rank provides a more flexible and effective inductive bias than fixed-rank adaptations.
>
---
#### [new 071] A French Corpus Annotated for Multiword Expressions with Adverbial Function
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在构建一个标注多词表达的法语语料库，用于信息检索、抽取及句法分析。工作包括定义标注范围、描述资源与方法，并分享标注结果。**

- **链接: [https://arxiv.org/pdf/2606.04828](https://arxiv.org/pdf/2606.04828)**

> **作者:** Eric Laporte; Takuya Nakamura; Stavroula Voyatzi
>
> **摘要:** This paper presents a French corpus annotated for multiword expressions (MWEs) with adverbial function. This corpus is designed for investigation on information retrieval and extraction, as well as on deep and shallow syntactic parsing. We delimit which kind of MWEs we annotated, we describe the resources and methods we used for the annotation, and we briefly comment the results. The annotated corpus is available at this http URL under the LGPLLR license.
>
---
#### [new 072] Noisy memory encoding explains negative polarity illusions
- **分类: cs.CL**

- **简介: 该论文属于语言理解任务，研究负极性幻觉现象。通过实验验证记忆编码不准确导致对否定词位置的误判，提出 determiner 交换解释该现象。**

- **链接: [https://arxiv.org/pdf/2606.04340](https://arxiv.org/pdf/2606.04340)**

> **作者:** Yuhan Zhang; Edward Gibson
>
> **备注:** 21 pages, 5 figures, submitted for journal publication
>
> **摘要:** A sentence like "The authors that no critics recommended have ever received acknowledgment for a best-selling novel" is sometimes rated as acceptable even though, strictly speaking, it is ungrammatical because the negative polarity word "ever" is not licensed where it is. This behavioral effect is sometimes called a "negative polarity illusion". Here we propose that the lossy context surprisal theory of Hahn et al. (2022) -- whereby people have an imperfect encoding of complex sentences -- might explain this effect. We hypothesize that people have poor memory representation of the determiners in the main-clause and embedded-clause subjects and could entertain a determiner exchange that licenses ever. We propose that more similar determiners in those positions would trigger stronger illusion effects. Acceptability judgment tasks with six novel determiner pairs (e.g., "few" and "many", "few" and "most") support our proposal, showing, specifically, that a novel sentence, "Many authors that few critics recommended have ever received acknowledgment for a best-selling novel", triggered a much stronger illusion than the canonical one even without time pressure. These results offer further support for the suggestion that human language processing is imperfect and resource-rational: in face of working memory limitations, humans rationally reconstruct what is most likely from noisy linguistic input to facilitate downstream processing.
>
---
#### [new 073] Stepwise Reasoning Enhancement for LLMs via External Subgraph Generation
- **分类: cs.CL**

- **简介: 该论文属于知识增强的推理任务，旨在提升大语言模型在多步骤推理中的逻辑一致性与准确性。通过生成外部子图引导模型逐步推理，结合知识图谱增强答案验证。**

- **链接: [https://arxiv.org/pdf/2606.04454](https://arxiv.org/pdf/2606.04454)**

> **作者:** Xin Zhang; Yang Cao; Baoxing Wu; Kai Song; Siying Li
>
> **摘要:** Large language models have shown strong performance in natural language generation and downstream reasoning tasks, but they still struggle with logical consistency, factual grounding, and interpretability in complex multi-step reasoning. To address these limitations, this paper proposes SGR, a stepwise reasoning enhancement framework that integrates large language models with external knowledge graphs through query-relevant subgraph generation. Given an input question, SGR first extracts key entities, relations, and constraints to construct a structured schema, then retrieves compact subgraphs from a knowledge graph using schema-guided querying. The generated subgraphs provide explicit relational evidence that guides the language model through step-by-step reasoning. In addition, SGR combines direct Cypher-based reasoning with collaborative reasoning integration, allowing candidate answers from multiple reasoning paths to be validated and aggregated according to both model confidence and graph consistency. Experiments on benchmark datasets including CWQ, WebQSP, GrailQA, and KQA Pro demonstrate that SGR improves reasoning accuracy and Hits@1 performance over standard prompting and several knowledge-enhanced baselines. Ablation studies further show that schema guidance and Neo4j-based retrieval are both crucial to the effectiveness of the framework. These results indicate that dynamically generated external subgraphs can improve the accuracy, robustness, and interpretability of LLM-based reasoning.
>
---
#### [new 074] Streaming Communication in Multi-Agent Reasoning
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于多智能体推理任务，解决传统系统延迟高、效果差的问题。提出StreamMA，通过流式传输提升效率和效果。**

- **链接: [https://arxiv.org/pdf/2606.05158](https://arxiv.org/pdf/2606.05158)**

> **作者:** Zhen Yang; Xiaogang Xu; Wen Wang; Cong Chen; Xander Xu; Ying-Cong Chen
>
> **备注:** project page: this https URL
>
> **摘要:** Multi-agent reasoning systems adopt a "generate-then-transfer" paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, two frontier LLMs (Claude Opus 4.6 and GPT-5.4), and three topologies (Chain, Tree, Graph), StreamMA outperforms both baselines (avg. +7.3 pp, max +22.4 pp on HMMT 2026; Claude Opus 4.6-high). Beyond these contributions, we discover a "step-level scaling law": increasing per-agent steps consistently improves both effectiveness and efficiency, a new scaling dimension orthogonal to and composable with agent-count scaling.
>
---
#### [new 075] Cartridges at Scale: Training Modular KV Caches over Large Document Collections
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文提出CAS框架，解决大规模文档集合中KV缓存的可扩展性问题。通过动态混合和内存管理，提升多Cartridge学习效率，减少预填充消耗。**

- **链接: [https://arxiv.org/pdf/2606.04557](https://arxiv.org/pdf/2606.04557)**

> **作者:** Momchil Hardalov; Gonzalo Iglesias; Adrià de Gispert
>
> **备注:** 21 pages, 5 figures, 17 tables
>
> **摘要:** Large Language Models can reason over long contexts, yet prefilling millions of tokens is wasteful as much of the content remains static across queries. Cartridges address this by distilling document collections into reusable key-value (KV) caches that eliminate prefilling while preserving accuracy. A critical limitation of this approach is that cartridges are monolithic and non-compositional: encoding an entire collection into a single KV block does not scale, and naively mixing cartridges trained in isolation collapses performance to near chance. We introduce Cartridges at Scale (CAS), a training framework for scalable multi-cartridge learning with dynamic distractor mixing and a memory-efficient budget manager that rotates hundreds of per-document cartridges between GPU and persistent storage. Our approach scales to collections exceeding a million tokens, improving over a monolithic cartridge by 10-31 points at comparable token budgets. Oracle cartridge accuracy falls within 2-6 points of full in-context learning even at high compression. When paired with retrieval for cartridge selection, CAS matches or exceeds conventional RAG accuracy while consuming 3-4x fewer prompt tokens.
>
---
#### [new 076] QO-Bench: Diagnosing Query-Operator-Preserving Retrieval over Typed Event Tuples
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出QO-Bench，用于诊断查询-操作符保留的检索任务，解决现有系统仅关注语义相关性的问题，通过精确匹配实现操作符级诊断。**

- **链接: [https://arxiv.org/pdf/2606.04646](https://arxiv.org/pdf/2606.04646)**

> **作者:** Mengao Zhang; Xiang Yang; Chang Liu; Tianhui Tan; Ke-wei Huang
>
> **备注:** 14 pages
>
> **摘要:** Many real-world questions over business, legal, and scientific corpora are natural-language versions of database-style queries over records latent in text. Existing retrieval-augmented generation (RAG) systems are optimized primarily for semantic relevance, but retrieving plausible passages does not guarantee correct query execution. We introduce QO-Bench, a diagnostic benchmark for query-operator question answering over typed event tuples. The benchmark covers 22,984 news articles and 614 corporate events across 18 query templates, evaluated on 785 questions. Each gold answer is deterministically computed from typed event tuples and scored by recall, with answers matched to the gold tuples by exact match rather than an LLM judge. This design enables operator-level diagnosis such as joins and intersection. We evaluate RAG, ReAct RAG, GraphRAG, and information-extraction-to-SQL under matched conditions, with a long-context oracle ceiling to isolate retrieval failure. A two-axis framework -- index-time preservation versus query-time execution -- predicts where each paradigm fails, and the results bear it out: systems retrieve relevant text but discard the typed values operators need, and the deployable paradigm ranking inverts across operators, with similarity retrieval leading on filter/project and extraction-to-SQL on intersection and counting. Even given the gold evidence, a long-context oracle stays far from saturated, so operator execution -- not retrieval alone -- is a core bottleneck that a stronger answer model does not remove. QO-Bench reframes the goal from passage relevance to query-operator-preserving retrieval.
>
---
#### [new 077] A Systematic Evaluation of Positional Bias in Multi-Video Summarization with MLLMs
- **分类: cs.CL**

- **简介: 该论文属于多视频摘要任务，研究MLLM在多视频输入时的定位偏差问题。通过构建基准和评估模型，分析位置对摘要质量的影响，提出改进方法。**

- **链接: [https://arxiv.org/pdf/2606.04596](https://arxiv.org/pdf/2606.04596)**

> **作者:** Huangchen Xu; Yuan Wu; Yi Chang
>
> **摘要:** Multimodal Large Language Models (MLLMs) are increasingly used for video understanding, yet their reliability under multi-video inputs remains poorly understood. We study positional bias in multi-video summarization, where the quality of a per-video summary can change with the video's input slot even when the underlying content is unchanged. We construct a benchmark from ActivityNet and News videos, covering Cooking, Domestic, Leisure, and News settings with two- and four-video inputs. We evaluate nine open-source and proprietary MLLMs and measure position effects with three complementary metrics: Coverage, Directional Positional Bias (DPB), and Middle-Edge Gap (MEG). Our results show that positional effects are domain- and model-dependent: signed directional bias can be small even when middle positions underperform, and increasing visual or generation budget does not uniformly remove the imbalance. We further analyze prompt-level mitigation methods. Together, the results show that multi-video summarization remains sensitive to input protocol and position, motivating more robust order-invariant multimodal systems.
>
---
#### [new 078] POLARIS: Guiding Small Models to Write Long Stories
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于创意写作任务，解决小模型在长文本生成中质量下降的问题。通过引入前沿模型评分和人类参考注入，提升模型生成长故事的能力。**

- **链接: [https://arxiv.org/pdf/2606.04095](https://arxiv.org/pdf/2606.04095)**

> **作者:** Rishanth Rajendhran; Jenna Russell; Mohit Iyyer; John Frederick Wieting
>
> **摘要:** Small open-weight models struggle at long-form creative writing: their generated stories either fall far short of the requested length, or their quality significantly degrades as length increases, especially when compared to frontier models. We present POLARIS (Policy Optimization with LLM-as-a-judge rewards and Anchored-Reference Injection for Storywriting), a lower-compute GRPO recipe with two key ingredients: a frontier LLM judge with a structured Story Quality rubric as the online reward, and human-reference injection (HRI), where a teacher-forced human-written story serves as a high-reward anchor within each GRPO group. By applying our training recipe to Qwen3.5-9B, using a dataset of approximately 1.4K prompt-story pairs derived from 100 short-story anthologies and 4 A100 GPUs, we obtain POLARIS-9B. Across five benchmarks spanning in-distribution and out-of-distribution prompts and rubrics, POLARIS-9B is competitive with much larger open-weight models while following length instructions more closely. A blinded human evaluation confirms that POLARIS-9B is preferred to the base Qwen3.5-9B and on par with Qwen3.5-27B. Despite training only on stories up to 4k words, POLARIS-9B preserves quality on prompts requesting stories up to 3 times the training length, a regime where most open-weight models degrade substantially in quality, length adherence, or both. More broadly, our results suggest that length generalization is a meaningful stress test for creative-writing models and a useful lens for distinguishing otherwise close models.
>
---
#### [new 079] DeliChess: A Multi-party Dialogue Dataset for Deliberation in Chess Puzzle Solving
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出DeliChess数据集，用于研究多人协作解决国际象棋谜题的对话。任务是分析群体推理与决策过程，解决多主体协作中的观点整合问题。工作包括构建数据集并评估讨论效果。**

- **链接: [https://arxiv.org/pdf/2606.04987](https://arxiv.org/pdf/2606.04987)**

> **作者:** Xiaochen Zhu; Georgi Karadzhov; Tom Stafford; Andreas Vlachos
>
> **摘要:** Multi-party dialogue is a critical setting for studying collaborative reasoning and decision-making, yet existing datasets rarely focus on structured, in-depth complex reasoning tasks. We introduce DeliChess, a novel dataset of group deliberation dialogues in which participants collaboratively solve multiple-choice chess puzzles. Each group first completes the puzzle individually, then engages in a multi-party discussion before submitting a revised collective answer. The dataset includes 107 dialogues with full transcripts, pre- and post-discussion choices, and metadata on puzzle difficulty and move quality. We evaluate performance using three metrics based on chess engine evaluations, and find that deliberation significantly improves group accuracy. We further analyse the role of probing utterances (i.e., messages that elicit proposals, justifications, or strategic reflection) using a classifier trained on prior deliberation data. While probing makes group performance more variable after discussion, it does not consistently lead to better performance. Our dataset offers a rich testbed for modelling group reasoning, dialogue dynamics, and the resolution of differing perspectives and opinions in a well-defined strategic domain.
>
---
#### [new 080] Benchmarking Living-Screen-Native GUI Agents on Short-Video Platforms
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出LivingScreen基准，解决短视频平台中GUI代理的动态屏幕适应问题，通过任务套件和指标评估模型表现。**

- **链接: [https://arxiv.org/pdf/2606.04701](https://arxiv.org/pdf/2606.04701)**

> **作者:** Jiashu Yao; Heyan Huang; Daiqing Wu; Wangke Chen; Huaxi Ai; Haoyu Wen; Zeming Liu; Yuhang Guo
>
> **备注:** preprint
>
> **摘要:** GUI agents today assume a static screen, where the world is frozen between two actions. However, real interfaces such as short-video applications violate this assumption, as their content keeps playing, and a competent user must decide what to watch and for how long. We formalize this task as Living-Screen-Native GUI agents and introduce LivingScreen, the first benchmark instantiating it on short-video platforms, with a faithful browser-based environment, a three-tier task suite, and metrics that jointly score accuracy and information efficiency. Evaluating extensive frontier models, we find that none reaches the human cost-accuracy performance, and that their dominant failure mode is over- and under-observation, pointing to observation control as a missing capability axis for future GUI agents. All data and code will be available at this https URL.
>
---
#### [new 081] Continual Visual and Verbal Learning Through a Child's Egocentric Input
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态学习任务，旨在解决儿童从连续视觉和语言输入中学习词义的问题。工作包括提出BabyCL框架，实现单次时间序列处理与对比学习，提升词-参照映射效果。**

- **链接: [https://arxiv.org/pdf/2606.05115](https://arxiv.org/pdf/2606.05115)**

> **作者:** Xiaoyang Jiang; Yanlai Yang; Kenneth A. Norman; Brenden Lake; Mengye Ren
>
> **备注:** 15 pages, 4 figures
>
> **摘要:** Children learn the meanings of words from a continuous, temporally structured stream of egocentric experience. Recent work shows that neural networks can also learn word-referent mappings from a child's egocentric video recordings, but they cycle through the shuffled data for hundreds of epochs, contrasting with how children actually encounter their environment. We introduce BabyCL, a continual multimodal learning framework that processes the SAYCam dataset in a single chronological pass, combining streaming visual representation learning with an image-text contrastive objective. BabyCL combines a multi-stage temporal segmentation of the stream with a dual replay buffer that independently manages visual and multimodal histories, and it is jointly trained with three contrastive losses on a shared backbone. Under a matched optimization budget, BabyCL outperforms streaming learning baselines on the SAYCam Labeled-S 4AFC benchmark, substantially narrowing the gap to an upper bound of offline training. Ablations show that the gains are robust to the length of the online temporal segmentation window and the eviction rule of the replay buffer. Together, these results show that meaningful word-referent mappings can emerge under training conditions much closer to a child's actual experience.
>
---
#### [new 082] R-APS: Compositional Reasoning and In-Context Meta-Learning for Constrained Design via Reflective Adversarial Pareto Search
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出R-APS方法，解决代理系统中因推理冲突导致的可靠性问题，通过分解推理模式提升设计任务的鲁棒性和效率。**

- **链接: [https://arxiv.org/pdf/2606.04823](https://arxiv.org/pdf/2606.04823)**

> **作者:** João Pedro Gandarela; Thiago Rios; Stefan Menzel; André Freitas
>
> **摘要:** Large language models (LLMs) are fluent on open-ended tasks, yet in agentic settings, where a system must plan, use tools, and act over extended horizons, fluency does not ensure reliable delivery. We trace this gap to three coupled structural failures: errors propagate without localization, worst-case perturbations go unevaluated, and accumulated knowledge is never invalidated. We argue these share a root cause: abductive, counterfactual, meta-inductive, corrective, and inductive reasoning pull a shared context in incompatible directions. We introduce Reflective Adversarial Pareto Search (R-APS), to our knowledge the first method addressing all three failures jointly via reasoning-mode decomposition, allocating each reasoning mode its own context and orchestrating interaction across three timescales: staged compositional reasoning with a typed validation critic (failure localization), sensitivity-guided counterfactual stress-testing as a first-class Pareto objective (robustness), and meta-inductive rule extraction with explicit invalidation (persistent memory). R-APS requires no fine-tuning and operates on a frozen LLM purely via structured protocol design. We evaluate on planar mechanism synthesis (robotics, prosthetics, mechanical design), with every candidate checked by a kinematic solver. On 32 target trajectories, R-APS delivers robustness certificates 3.5x tighter than uniform-perturbation baselines, 46% faster iterations-to-first-admission, and 2.1x Chamfer-distance reduction over Enum+GA while jointly controlling bar-count and worst-case robustness. Small 4B reasoning-specialized models prove competitive with general-purpose 70B backbones inside the protocol, suggesting structured protocols can partially offset model scale.
>
---
#### [new 083] StepPRM-RTL: Stepwise Process-Reward Guided LLM Fine-Tuning for Enhanced RTL Synthesis
- **分类: cs.AI; cs.AR; cs.CL**

- **简介: 该论文属于RTL代码生成任务，解决长距离推理和正确性约束问题。通过结合分步轨迹建模、过程奖励和检索增强微调，提升LLM生成RTL的准确性和逻辑性。**

- **链接: [https://arxiv.org/pdf/2606.04246](https://arxiv.org/pdf/2606.04246)**

> **作者:** Prashanth Vijayaraghavan; Apoorva Nitsure; Luyao Shi; Ehsan Degan; Vandana Mukherjee
>
> **备注:** 6 pages, 2 figures, DAC'2026
>
> **摘要:** Automatic generation of RTL code for digital hardware designs remains challenging due to long-horizon reasoning, multi-step dependencies, and strict correctness constraints in Verilog and VHDL. We present StepPRM-RTL, a novel framework that combines stepwise trajectory modeling, process-reward modeling (PRM), and retrieval-augmented fine-tuning (RAFT) to enhance both the functional correctness and reasoning fidelity of LLM-based RTL code generation. StepPRM-RTL constructs stepwise reasoning trajectories from canonical solutions, where each step contains a rationale and incremental code modification. A Process Reward Model (PRM) evaluates intermediate steps, providing dense feedback that guides reinforcement-style updates during RAFT fine-tuning. Monte Carlo Tree Search (MCTS) explores alternative reasoning paths, enriching the training dataset with high-quality trajectories. This integration of stepwise and outcome-aware rewards allows the model to learn both how and why to construct correct RTL, improving long-horizon reasoning beyond standard supervised or outcome-based training. Experimental evaluation on benchmark Verilog and VHDL datasets demonstrates that StepPRM-RTL outperforms the best prior methods by over 10\% in functional correctness and reasoning fidelity metrics. Ablation studies confirm that the combination of PRM-guided rewards and stepwise trajectory exploration is key to its performance. StepPRM-RTL generalizes across RTL languages and provides a scalable framework for high-fidelity, interpretable code generation, establishing a new standard for LLM-assisted hardware design automation.
>
---
#### [new 084] Clinical Assistant for Remote Engagement Link (CARE-link): A Web-Based Electronic Health Records Software for Managing Diabetes
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出CARE-link系统，用于管理妊娠糖尿病，通过LLM协调医患互动，整合患者数据并提供决策支持，解决远程医疗与持续监测问题。**

- **链接: [https://arxiv.org/pdf/2606.04952](https://arxiv.org/pdf/2606.04952)**

> **作者:** Prince Ebenezer Adjei; Joshua Teye Tettey; Toufiq Musah; Audrey Agbeve; John Amuasi
>
> **摘要:** CARE-link is an open-source, web-based clinical support platform designed to improve the management of gestational diabetes by linking clinicians and patients through an LLM-mediated workflow. The system aggregates patient-generated data outside the hospital, summarizes relevant clinical information, and delivers context-aware decision support to clinicians. For patients, CARE-link provides clear explanations of management plans and delivers timely lifestyle guidance through a WhatsApp interface. The integrated dual-facing design aims to promote continuous monitoring, support individualized care, and reduce the burden of in-clinic follow-ups. Built with a modular architecture, the platform can be adapted to other chronic conditions requiring longitudinal tracking and behavioral support. CARE-link has the potential to enhance clinical oversight, promote patient compliance, and strengthen continuity of care particularly in resource-constrained settings.
>
---
#### [new 085] DetectZoo: A Unified Toolkit for AI-Generated Content Detection Across Text, Audio, and Image Modalities
- **分类: cs.MM; cs.AI; cs.CL; cs.CV; cs.LG; cs.SD**

- **简介: 该论文提出DetectZoo，一个统一的AI生成内容检测工具包，解决多模态内容检测的标准化问题，整合数据、模型和评估流程，便于研究与比较。**

- **链接: [https://arxiv.org/pdf/2606.04205](https://arxiv.org/pdf/2606.04205)**

> **作者:** Sajad Ebrahimi; Nima Jamali; Bardia Shirsalimian; Kelly McConvey; Wentao Zhang; Jalehsadat Mahdavimoghaddam; Maksym Taranukhin; Maura Grossman; Vered Shwartz; Yuntian Deng; Ebrahim Bagheri
>
> **摘要:** The growing popularity and capacity of generative models have eroded the distinction between human and machine-generated content, motivating a growing body of work on detection across text, images, and audio. Most available detectors are either commercial software or, if open-source, come with incompatible codebases with bespoke preprocessing, evaluation protocols, and evaluation metrics, which make their adoption, fair comparison, and reproduction quite difficult. To address this critical gap, we introduce DetectZoo, a first-of-its-kind, extensible toolkit designed to provide a unified interface for AI-generated content detection across text, audio, and image modalities. DetectZoo standardizes the complete empirical pipeline, from data ingestion and preprocessing to model assessment, offering researchers a cohesive framework to benchmark state-of-the-art detectors systematically. By integrating diverse public datasets and baseline detection algorithms under a single, unified API, our toolkit facilitates rigorous and reproducible evaluation. DetectZoo provides reference implementations of 61 detectors, native loaders for 22 benchmark datasets, and a standardized evaluation pipeline that reports multiple metrics through a common interface. Each detector is self-contained yet accessible through the same interface, automatically caches pretrained weights, and reproduces the original published results. DetectZoo lowers the barrier to entry for multi-modal AI forensics, enabling researchers to identify performance gaps across domains and accelerating the development of robust, generalizable detection techniques. The open-source repository and comprehensive documentation are publicly available at this https URL, and the package can be installed via pip install detectzoo.
>
---
#### [new 086] BreastGPT: A Multimodal Large Language Model for the Full Spectrum of Breast Cancer Clinical Routine
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出BreastGPT，解决乳腺癌临床流程中的多模态推理问题。构建了BreastStage数据集，设计统一模型提升跨尺度视觉建模能力。**

- **链接: [https://arxiv.org/pdf/2606.04911](https://arxiv.org/pdf/2606.04911)**

> **作者:** Yang Liu; Jiajin Zhang; Danyang Tu; Yaojun Hu; Jiao Qu; Jiuyu Zhang; Yu Shi; Wei Fang; Shi Gu; Ling Zhang; Yingda Xia
>
> **摘要:** Breast cancer remains a leading cause of cancer-related mortality among women. Its clinical management requires multimodal reasoning across a clinical workflow that spans \textit{screening}, \textit{diagnosis} and \textit{treatment planning}, where each stage involves distinct imaging modalities, task objectives, and reasoning patterns. However, constrained by data scarcity and model versatility, existing medical MLLMs are typically evaluated on isolated modalities or narrow task families, limiting their ability to support workflow-level clinical reasoning. In this work, we first introduce \textbf{BreastStage}, a workflow-aligned breast imaging instruction corpus comprising 1.86M instruction-following pairs curated from 17 sub-datasets across 5 imaging modalities and 136 task templates. Its held-out split, \textbf{BreastStage-Bench}, provides a comprehensive benchmark for evaluating multimodal reasoning across the breast cancer care continuum. Building on this corpus, we propose \textbf{BreastGPT}, a unified MLLM equipped with a dual-branch visual encoder and concept-preserving token compression to bridge the scale gap between standard radiology and gigapixel pathology. On BreastStage-Bench, BreastGPT achieves 75.66\% closed-ended accuracy and 89.92\% open-ended score, outperforming both general-purpose and medical-specific MLLMs across clinical stages and task formats. These results suggest that workflow-aligned data and cross-scale visual modeling are critical for clinically grounded medical MLLMs. All data, code, and model checkpoints are released at this https URL.
>
---
#### [new 087] Validity Threats for Foundation Model Research
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于基础模型研究任务，旨在解决实验设计中的有效性威胁问题。通过构建因果推断框架，评估不同研究策略的有效性，识别潜在的 validity 问题。**

- **链接: [https://arxiv.org/pdf/2606.05029](https://arxiv.org/pdf/2606.05029)**

> **作者:** Gunnar König; Martin Pawelczyk; Ulrike von Luxburg; Sebastian Bordt
>
> **摘要:** Controlled experiments are the backbone of machine learning research, but at the scale of modern foundation models, they have become prohibitively expensive. Instead, the community increasingly relies on research strategies that approximate the ideal experiment at a fraction of the cost: proxy experiments and scaling laws, observational studies with publicly available models, and single-run designs that leverage variation within individual training runs. In this work, we argue that there is no free lunch when approximating large-scale experiments on a compute budget. Specifically, savings in compute come at the cost of validity threats -- hidden and sometimes untestable assumptions that, when violated, can invalidate research claims. To help navigate such threats, we propose an evaluation framework that casts foundation model research as a causal inference problem. Within this framework, we evaluate different research strategies through four types of validity adapted from the empirical social sciences -- statistical, internal, external, and construct validity. We find that each strategy comes with a characteristic validity profile: proxy experiments trade external and construct validity for statistical and internal validity; observational studies face confounding and effect heterogeneity; and single-run designs are strained by interference between treated units. This analysis reveals several validity threats that have received insufficient attention in the literature. Overall, our evaluation framework provides researchers with a practical toolkit for scrutinizing validity threats in foundation model research~designs.
>
---
#### [new 088] NextMotionQA: Benchmarking and Judging Human Motion Understanding with Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出NextMotionQA基准，用于评估视觉-语言模型在人体运动理解上的能力，解决现有基准不足的问题。通过多项任务测试模型性能，揭示其局限性。**

- **链接: [https://arxiv.org/pdf/2606.04773](https://arxiv.org/pdf/2606.04773)**

> **作者:** Yong Cao; Chuqiao Li; Xianghui Xie; Gerard Pons-Moll; Andreas Geiger
>
> **备注:** 23 pages, 8 figures, 9 tables
>
> **摘要:** Reliable evaluation of human motion understanding is fundamental to advancing embodied AI, robotics, and animation. However, existing benchmarks suffer from coarse semantic granularity, undifferentiated difficulty, limited annotation quality, and pervasive answer ambiguity, leaving them unable to diagnose where current models fail. To bridge this gap, we introduce NextMotionQA, a comprehensive benchmark that leverages vision-language models (VLMs) for semi-automated, expert-verified dataset. NextMotionQA features three complementary tasks: multiple-choice question answering, video captioning, and fine-grained error correction. Each task is systematically structured across three core semantic axes and stratified into three task complexity levels. Our extensive evaluation of twelve representative VLMs uncovers critical capability gaps and weakness that remain invisible under conventional, single-task evaluations. In a complementary direction, recent work has begun using VLMs as judges for text-to-motion evaluation; we ask whether they show the same degradation under harder tasks. We find that VLMs align strongly with expert ratings on coarse criteria (Cohen's \kappa=0.70) but break down on fine-grained, part-level judgment (\kappa=0.10), validating the paradigm in its strong regime while clarifying its limits.
>
---
#### [new 089] CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文提出CleanCodec，解决语音编码中信息冗余与效率不足的问题，通过感知引导编码，提升语音重建质量与token效率。**

- **链接: [https://arxiv.org/pdf/2606.04418](https://arxiv.org/pdf/2606.04418)**

> **作者:** Eugene Kwek; Feng Liu; Rui Zhang; Wenpeng Yin
>
> **摘要:** Neural audio codecs are a key component of speech processing pipelines, compressing audio into discrete tokens for downstream modeling. However, existing codecs struggle to balance reconstruction quality with token efficiency, often encoding perceptually irrelevant information such as background noise and recording artifacts at the expense of linguistically and acoustically meaningful content. We reframe audio tokenization as a selective information bottleneck problem and propose CleanCodec, a denoising audio codec which learns to encode only perceptually important features and discard imperceptible information. At just 12.5 tokens per second, CleanCodec achieves state-of-the-art tokenization efficiency, substantially outperforming existing codecs in speaker similarity and speech intelligibility. Evaluations on downstream text-to-speech and voice conversion tasks further demonstrate improved performance and up to 17x faster inference, highlighting significant efficiency gains.
>
---
#### [new 090] Data Attribution in Large Language Models via Bidirectional Gradient Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型可解释性任务，旨在解决LLM中训练数据影响的归属问题。通过双向梯度优化方法，实现对模型输出的训练数据溯源，提升模型的可问责性。**

- **链接: [https://arxiv.org/pdf/2606.04928](https://arxiv.org/pdf/2606.04928)**

> **作者:** Frédéric Berdoz; Luca A. Lanzendörfer; Kaan Bayraktar; Roger Wattenhofer
>
> **备注:** Presented at the AI Governance (AIGOV) Workshop at AAAI 2026
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed across diverse applications, raising critical questions for governance, accountability, and data provenance. Understanding which training data most influenced a model's output remains a fundamental open problem. We address this challenge through training data attribution (TDA) for auto-regressive LLMs by expanding upon the inverse formulation: How would training data be affected if the model had seen the generated output during training? Our method perturbs the base model using bidirectional gradient optimization (gradient ascent and descent) on a generated text sample and measures the resulting change in loss across training samples. Our framework supports attribution at arbitrary data granularity, enabling both factual and stylistic attribution. We evaluate our method against baselines on pretrained models with known datasets, and show that it outperforms previous work on influence metrics, thereby enhancing model interpretability, an essential requirement for accountable AI systems.
>
---
#### [new 091] BEATS: Bootstrapping E-commerce Attribute Taxonomies for Search through Iterative Human-AI Collaboration
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于电商属性分类任务，解决产品目录不完整问题。通过人机协作生成属性体系，提升搜索能力。**

- **链接: [https://arxiv.org/pdf/2606.04909](https://arxiv.org/pdf/2606.04909)**

> **作者:** Yung-Yu Shih; Shang-Yu Su; Tzu-I Ho; Dongzhe Wang; Yun-Nung Chen
>
> **备注:** 6 pages, 1 figure, 5 tables. Accepted to SIGIR 2026 Industry Track. Official version: this https URL
>
> **摘要:** E-commerce platforms in emerging markets often operate with underdeveloped product catalogs that contain only category taxonomies but lack structured attribute schemas. This absence of fine-grained product attributes limits search capabilities -- preventing faceted filtering, degrading query understanding, and weakening semantic representations used by search systems. We present BEATS, a human-in-the-loop LLM framework for bootstrapping product attribute taxonomies entirely from scratch. Our approach extends a multi-stage LLM generation pipeline with two critical production stages: (1) proactive quality checking by model developers to filter erroneous outputs, and (2) human annotation by domain-expert local staff to validate generated attributes. The framework operates iteratively -- prompts at each generation stage are refined based on quality check observations and annotator feedback across successive rounds, progressively improving attribute quality. Once the attribute taxonomy is established, we employ LLMs to perform structured attribute tagging on individual product items, enriching their contextual representations. The enriched catalog directly benefits multiple components of the search system: enabling granular attribute-based filtering, providing structured features for ranking models, and improving semantic representations for dense retrieval. We validate the generated taxonomy by training dense retrieval models on attribute-enriched product data, demonstrating consistent improvements over baselines using original catalog information. Our system has been deployed at Rakuten Taiwan, enriching 9 major categories spanning 2,694 sub-categories with 67,277 generated attributes, and over 5.4 million products have been tagged with the generated attributes, with plans to enrich the entire product catalog.
>
---
#### [new 092] Can Generalist Agents Automate Data Curation?
- **分类: cs.AI; cs.CL; cs.CV; cs.ET; cs.LG**

- **简介: 该论文属于数据预处理任务，旨在解决人工数据标注耗时问题。通过引入基准测试，评估通用编码代理自动完成数据筛选的可行性，并提出结构化框架提升效率。**

- **链接: [https://arxiv.org/pdf/2606.04261](https://arxiv.org/pdf/2606.04261)**

> **作者:** Feiyang Kang; Hanze Li; Adam Nguyen; Mahavir Dabas; Jiaqi W. Ma; Frederic Sala; Dawn Song; Ruoxi Jia
>
> **备注:** Preprint
>
> **摘要:** Curating training data is among the most consequential yet labor-intensive parts of modern AI development: practitioners iteratively propose, implement, evaluate, and revise data policies against noisy benchmark feedback. We ask whether generalist coding agents can automate this data-curation loop. We introduce *Curation-Bench*, an agent-centric benchmark that fixes the model, training recipe, and evaluation suite while giving agents command-line access to inspect data, implement policies, submit them to a fixed training/evaluation pipeline, and revise. In a vision-language instruction-tuning instantiation, out-of-the-box agents reach strong published data-selection baselines within ten iterations. However, trajectory analysis reveals a persistent *execution-research gap*: agents mainly tune local policy variants rather than explore new policy families, even when given strategy guides and paper references. Scaffolds requiring each iteration to cite, instantiate, and adapt a prior method shift agents toward method-guided exploration. The scaffolded agent autonomously composes -- without human design input -- a data-selection policy that outperforms strong published baselines at one-tenth their data budget. Overall, current agents can run the curation loop, but reliable data research requires scaffolded method adaptation, not open-ended prompting alone. Code and benchmark are open-sourced.
>
---
#### [new 093] Stateful Visual Encoders for Vision-Language Models
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出状态感知视觉编码器，解决多图像视觉对比问题。通过引入视觉上下文，提升VLM在空间聚合、物体差异和轨迹克隆任务中的性能。**

- **链接: [https://arxiv.org/pdf/2606.04433](https://arxiv.org/pdf/2606.04433)**

> **作者:** Zirui Wang; Junwei Yu; Adam Yala; David M. Chan; Joseph E. Gonzalez; Trevor Darrell
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-language models (VLMs) are increasingly used in multi-image, multi-turn agentic settings where decisions depend on visual changes. However, in existing open-weight VLMs, visual comparisons happen only inside the language model, while the visual encoder itself remains stateless: each image is encoded independently, without access to the prior visual context. As a result, small but task-critical changes may be attenuated before the language model has a chance to compare them, especially when those changes do not affect the high-level semantics of the scene. We introduce a Stateful Visual Encoder, which conditions each visual representation on prior visual features. Under supervised finetuning, VLMs equipped with stateful encoders achieve consistent improvements on controlled tasks involving cross-image spatial aggregation, multi-object visual differencing, and visual trajectory behavior cloning. These improvements are consistent across input resolutions, language model sizes, and VLM backbones. Finally, we validate our model on real-world tasks, including longitudinal radiology, fine-grained image comparison, and remote sensing, where stateful encoders consistently improve generalist VLM baselines and can match or surpass specialized models in selected domains. Project page: this https URL
>
---
#### [new 094] Covert Influence Between Language Models
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 该论文研究语言模型间的隐性影响问题，属于安全与隐私任务。通过分析三种接口，揭示隐性影响的风险及传播机制，提出归因评分方法以检测和缓解该问题。**

- **链接: [https://arxiv.org/pdf/2606.04071](https://arxiv.org/pdf/2606.04071)**

> **作者:** Avidan Shah; Jay Chooi; Jinghua Ou; Shi Feng
>
> **摘要:** As language models increasingly consume one another's outputs, covert influence -- a phenomenon where a sender's payload (the behavioral disposition it is conditioned to propagate) transfers to a receiver through carriers undetectable by humans -- becomes a growing risk. We characterize this risk across three interfaces: supervised fine-tuning, on-policy distillation, and in-context learning, and find that they vary in the scale of influence achievable without leaving behind human-visible traces. Using inference-time per-sample attribution scores, we study covert influence across all three interfaces with the ability to select carriers that amplify training-time influence, unlocking payload transfers that prior work could not achieve. We further provide evidence that covert influence with natural-language carriers is a distinct phenomenon from prior studies using number carriers, as the latter is more resistant to human detection and less portable across model families. Together, these results suggest that the risk surface for covert influence is broader than previously recognized, and we study pointwise attribution scoring methods as a tool to investigate and mitigate it.
>
---
#### [new 095] Reinforcement Learning from Rich Feedback with Distributional DAgger
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决如何利用丰富反馈提升模型性能的问题。提出DistIL方法，通过分布DAgger优化策略，实现更有效的学习与改进。**

- **链接: [https://arxiv.org/pdf/2606.05152](https://arxiv.org/pdf/2606.05152)**

> **作者:** Rishabh Agrawal; Jacob Fein-Ashley; Paria Rashidinejad
>
> **摘要:** Reasoning models have advanced rapidly, but the dominant reinforcement learning from verifiable rewards (RLVR) recipe remains surprisingly narrow: sample many responses and reward each with a single bit indicating whether the final answer is correct. Yet many settings provide rich feedback, including execution traces, tool outputs, expert corrections, and model self-evaluations. We study how to use such feedback through a distributional variant of the classic imitation learning algorithm DAgger, where the learner has local access to an expert distribution on states visited by the current policy. This yields a simple forward cross-entropy objective that admits a blackbox expert and whose sequence-level gradient {conduct rich credit assignment by propagating} future expert-student disagreement back to earlier decisions. We show that prior RL with self-distillation objectives based on reverse KL or Jensen-Shannon fail to guarantee monotonic policy improvement: even when the expert has higher reward, their updates may increase probability on worse actions. In contrast, we show that forward cross-entropy admits monotonic policy improvement and enjoys guarantees on regret. We further show that our objective optimizes a lower bound on teacher-weighted likelihood of success, leading to improved Pass@N. Empirically, our approach, DistIL, improves over RLVR and RL with self-distillation baselines across a variety of domains: scientific reasoning, coding, and solving hard mathematical problems.
>
---
#### [new 096] BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的偏见缓解任务，旨在解决大模型中社会偏见的对齐问题。通过提出BiasGRPO框架，利用组相对策略优化稳定训练，提升生成质量与公平性。**

- **链接: [https://arxiv.org/pdf/2606.04807](https://arxiv.org/pdf/2606.04807)**

> **作者:** Saket Reddy; Ke Yang; ChengXiang Zhai
>
> **备注:** Accepted to Findings of the ACL
>
> **摘要:** Mitigating social bias in Large Language Models (LLMs) presents a distinct alignment challenge: unlike verifiable tasks, bias lacks a single ground truth, creating a high-variance, subjective reward landscape. Previous preference-based fine-tuning methods have major trade-offs: Direct Preference Optimization (DPO) is limited by the lack of exploration inherent in offline training, while Proximal Policy Optimization (PPO) can lead to training instability due to potentially unreliable critic estimates. In this paper, we propose BiasGRPO, a framework using Group Relative Policy Optimization (GRPO) to stabilize alignment by normalizing rewards across a group of sampled completions. By substituting the value function with a group-relative baseline, our approach reduces instability while maintaining the exploration benefits of online training. We find that BiasGRPO outperforms DPO and PPO across multiple benchmarks, indicating its effectiveness. To adapt GRPO, we synthetically extend a dataset spanning multiple domains and contexts. We also create and release a custom bias reward model that effectively guides generation while being highly compute-efficient and avoiding knowledge degradation, providing a valuable resource that can be seamlessly integrated into multi-objective RLHF pipelines.
>
---
#### [new 097] Disentangling Answer Engine Optimization from Platform Growth: A Log-Based Natural Experiment on ChatGPT Referral Traffic
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于因果推断任务，旨在区分AEO效果与平台增长影响。通过自然实验分析ChatGPT引流数据，验证AEO实际效果，揭示其对平台增长的依赖性。**

- **链接: [https://arxiv.org/pdf/2606.04362](https://arxiv.org/pdf/2606.04362)**

> **作者:** Keisuke Watanabe; Kazuki Nakayashiki
>
> **备注:** 9 pages, 4 figures, 1 table
>
> **摘要:** Large language model (LLM) "answer engines" such as ChatGPT now send measurable referral traffic to the open web, and a practice analogous to search engine optimization, here called Answer Engine Optimization (AEO), has emerged. Public AEO success stories typically quote large raw growth multiples, but raw referral growth is confounded by the rapid platform-level growth of the answer engines themselves. We report a longitudinal field study on a single high-traffic domain (this http URL) whose corpus of hundreds of thousands of YouTube question-and-answer pages received a defined bundle of AEO interventions in January 2026 (detailed in Section 4). Because the interventions were concentrated on one subset of the site, the untreated remainder of the same domain acts as a contemporaneous control that absorbs the platform tailwind. Using first-party analytics and server logs rather than probabilistic third-party estimators, we find: (1) raw growth is dominated by the platform tailwind: on monthly aggregates total ChatGPT referrals grew 5.7x while untreated pages on the same domain grew 3.5x over the same window; (2) an interrupted time-series model on the weekly treated/control ratio estimates a discrete, intervention-aligned level increase of 1.82x (95% CI 1.31-2.54, HAC p=0.001), robust across engagement-filtered traffic (2.27x) and alternative specifications; (3) however, a conservative placebo-in-time permutation test yields p=0.16, so the effect is suggestive, not conclusive, given a short and noisy pre-period; and (4) Google organic clicks to treated pages did not fall beyond the ambient site-wide trend and indexation was preserved, consistent with the SEO-protection rule. The methodological message, separating treatment from platform tailwind with an on-domain control, matters more than any single multiple, and implies that headline AEO multiples substantially overstate causal effect.
>
---
#### [new 098] Cascading Hallucination in Agentic RAG: The CHARM Framework for Detection and Mitigation
- **分类: cs.AI; cs.CL; cs.CR; cs.IR**

- **简介: 该论文属于自然语言处理任务，针对agentic RAG系统中的级联幻觉问题，提出CHARM框架进行检测与缓解。**

- **链接: [https://arxiv.org/pdf/2606.04435](https://arxiv.org/pdf/2606.04435)**

> **作者:** Saroj Mishra
>
> **摘要:** Multi-step agentic retrieval-augmented generation (RAG) pipelines have demonstrated significant capability for complex reasoning tasks, yet remain vulnerable to a class of failure that existing hallucination detection mechanisms systematically miss: cascading hallucination, where errors introduced at early pipeline stages propagate and amplify across successive reasoning steps, producing confident but factually incorrect final outputs. To address this vulnerability, we formalize cascading hallucination as a distinct failure mode in agentic RAG systems, present a four-type taxonomy of cascade patterns, and introduce CHARM (Cascading Hallucination Aware Resolution and Mitigation), an architectural framework for detecting and interrupting error propagation in multi-step reasoning pipelines. CHARM comprises four components - stage-level fact verification, cross-stage consistency tracking, confidence propagation monitoring, and cascade resolution triggering - that operate alongside standard agentic RAG pipelines without requiring architectural replacement. We evaluate CHARM on HotpotQA, MuSiQue, 2WikiMultiHopQA, and a custom adversarial dataset across LangChain agentic pipeline configurations, achieving an 89.4% cascade detection rate with a 5.3% false positive rate and 215 ms +/- 18 ms average latency overhead per stage, achieving an error propagation reduction of 82.1%, compared to 18.5% for output-level detectors. Component ablations confirm that each detection module contributes meaningfully to overall cascade coverage. CHARM integrates with human-in-the-loop oversight frameworks to provide a complete reliability and governance stack for production agentic AI deployment.
>
---
#### [new 099] VentAgent: When LLMs Learn to Breathe -- Multi-Objective Arbitration for ARDS Ventilation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于医疗自动化任务，旨在解决ARDS机械通气中的多目标平衡问题。针对现有方法的偏差和不透明性，提出VentAgent框架，利用LLM进行可解释的决策协调。**

- **链接: [https://arxiv.org/pdf/2606.04632](https://arxiv.org/pdf/2606.04632)**

> **作者:** Teqi Hao; Yuxuan Fu; Xiaoyu Tan; Shaojie Shi; Bohao Lv; Yinghui Xu; Xihe Qiu
>
> **摘要:** Mechanical ventilation for Acute Respiratory Distress Syndrome (ARDS) requires balancing competing physiological goals, including oxygenation, lung protection, and acid-base homeostasis. However, current data-driven methods, especially those imitating retrospective Electronic Health Records (EHR), often suffer from imitation bias. They may capture superficial correlations from inconsistent clinical demonstrations, such as associating passive ventilator settings with survival because such settings are common in stable patients, and thus fail to generalize to volatile or out-of-distribution phenotypes. Standard Reinforcement Learning (RL) methods also struggle with the adversarial trade-offs of critical care and often produce opaque policies with limited clinical interpretability. To address these limitations, we introduce VentAgent, a hierarchical framework in which Large Language Models (LLMs) act as transparent arbitrators for mechanical ventilation. We reformulate ventilation control as a dynamic Multi-Objective Arbitration process rather than single-objective optimization. VentAgent decomposes decision-making into three interpretable stages: Perception, Planning, and Orchestration. By leveraging the semantic reasoning capabilities of LLMs, it synthesizes strategies from heterogeneous experts and resolves conflicting clinical priorities through an explicit coordination mechanism. Evaluations on a high-fidelity physiological simulator show that VentAgent outperforms state-of-the-art RL and classical control baselines. Moreover, it converts control decisions into human-readable reasoning chains, offering a safer, more interpretable, and adaptable paradigm for critical care automation.
>
---
#### [new 100] Reproducing, Analyzing, and Detecting Reward Hacking in Rubric-Based Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决rubric-based RL中的奖励黑客问题。通过构建CHERRL环境，研究奖励偏差的机制并提出检测方法。**

- **链接: [https://arxiv.org/pdf/2606.04923](https://arxiv.org/pdf/2606.04923)**

> **作者:** Xuekang Wang; Zhuoyuan Hao; Shuo Hou; Hao Peng; Juanzi Li; Xiaozhi Wang
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Rubric-based reinforcement learning (RL) uses an LLM-as-a-Judge (LaaJ) to score model outputs according to rubrics as rewards. However, policy models may exploit latent biases in the judge, leading to reward hacking and ineffective or unsafe training outcomes. In real-world rubric-based RL, such hacking behaviors are often subtle and entangled with multiple judge biases, making them difficult to analyze, detect, and mitigate. In this paper, we introduce CHERRL, a controllable hacking environment for rubric-based RL. By injecting known biases into LaaJ, CHERRL enables stable reproduction of reward hacking, explicit observation of reward divergence, and precise identification of hacking onset. This provides a clean experimental testbed for studying the mechanisms and mitigations of reward hacking in rubric-based RL. To demonstrate its utility, we analyze different judge biases from the perspectives of discoverability and exploitability, and explore an agent-based system for automatically detecting reward hacking onset from training logs. The code and environment are publicly available at this https URL.
>
---
#### [new 101] Inference-Time Vulnerability Beyond Shallow Safety: Alignment Along Generation Trajectories
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全对齐任务，解决LLM在推理时易被干扰生成有害内容的问题。通过模拟中间序列扰动，直接对生成过程进行对齐，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.04778](https://arxiv.org/pdf/2606.04778)**

> **作者:** Kyungmin Park; Taesup Kim
>
> **摘要:** Safety-aligned Large Language Models (LLMs) remain vulnerable to interventions during inference that redirect generation toward harmful outputs. Recent work attributes this to shallow safety, where alignment concentrates in the first few output tokens. We show that shallow safety is a special case of a broader inference-time vulnerability, in which short token injections at any generation step can substantially alter subsequent safety behavior. We also find that a model's alignment with refusal directions in its hidden states does not predict its robustness to such injection, revealing that internal state alone does not determine generation behavior under perturbation. To address this, we align models directly on generation trajectories constructed by simulating mid-sequence perturbation, and show that this improves robustness to mid-sequence injection and generalizes to attacks that exploit early-token generation. Our work argues that robust safety alignment requires training on the generation process itself, not only its outputs.
>
---
#### [new 102] In-Context Graphical Inference
- **分类: cs.LG; cs.CL; cs.SC**

- **简介: 该论文提出一种新的图形模型推理方法，解决精确性与可扩展性的矛盾，通过恢复变量消去结构实现高效且准确的推理。**

- **链接: [https://arxiv.org/pdf/2606.05042](https://arxiv.org/pdf/2606.05042)**

> **作者:** Zehua Cheng; Wei Dai; Jiahao Sun
>
> **备注:** 19 Pages
>
> **摘要:** Marginal inference in discrete graphical models forces a choice between exactness and scalability: exact algorithms are intractable for high-treewidth graphs, while iterative approximations (Belief Propagation, variational methods) sacrifice convergence guarantees on frustrated topologies. We argue that this dichotomy stems from a mismatched inductive bias: iterative methods abandon the sequential elimination structure that makes exact inference correct. We introduce In-Context Graphical Inference (ICG-I), an autoregressive Graph Transformer that restores this structure by mimicking Variable Elimination with learned, Tensor- Train-compressed intermediate factors, paired with a Dirichlet output layer and Weighted Conformal Prediction for calibrated, distribution-free coverage guarantees under topological shift. We prove that TT compression errors propagate at most lincarly through the autoregressive chain, that the Dirichlet-Multinomial loss is a proper scoring rule, and that WCP maintains coverage with a quantifiable degradation under estimated density ratios. We conducted intensive experiments to evaluate ICG-I and achieved state-of-the-art performance across all benchmarks. ICG-I reduces MAE from 0.041 (best baseline) to 0.020 on standard instances and achieves 0.048 on N=500 frustrated spin glasses where BP diverges entirely.
>
---
#### [new 103] Large Language Models Hack Rewards, and Society
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.CY**

- **简介: 该论文属于人工智能安全领域，研究LLM在强化学习中可能“黑客”社会规则的问题。工作包括提出SocioHack环境，验证模型可发现监管漏洞，强调需更谨慎的训练方法。**

- **链接: [https://arxiv.org/pdf/2606.04075](https://arxiv.org/pdf/2606.04075)**

> **作者:** Wei Liu; Xinyi Mou; Hanqi Yan; Zhongyu Wei; Yulan He
>
> **备注:** 14 pages, 9 figures, 7 tables
>
> **摘要:** Reinforcement learning (RL) has become a dominant post-training paradigm, enabling large language models (LLMs) to learn from rewards. We observe that societal regulations are structurally similar to reward functions. They define measurable outcomes, thresholds, and exceptions, while often leaving institutional intent only partially specified. We hypothesise that the RL training process may exploit these gaps and therefore ask whether models' well-known tendency to hack reward functions during RL can scale into a more consequential failure mode named societal hacking: discovering loopholes in the rules society runs on. To study this phenomenon, we introduce SocioHack, a sandbox of 72 societal environments, and find that within these environments, reward hacking naturally emerges and leads to regulatory loophole discovery. Models learn to hack the social rules and generate strategies that remain technically compliant while defeating regulatory intent, and current LLM safeguards provide only limited mitigation. Therefore, collecting in-the-wild feedback for model training requires greater caution, and we need a next-generation post-training paradigm for safely iterating LLMs in real society.=
>
---
#### [new 104] VAMPS: Visual-Assisted Mathematical Problem Solving Benchmark
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出VAMPS基准，用于评估模型在数学问题中利用可视化工具的性能。任务是研究多模态大模型在依赖视觉辅助时的表现问题，通过构建包含图表解答的题目进行测试。**

- **链接: [https://arxiv.org/pdf/2606.04244](https://arxiv.org/pdf/2606.04244)**

> **作者:** Amirhossein Dabiriaghdam; Shayan Vassef; Mohammadreza Bakhtiari; Yasamin Medghalchi; Ilker Hacihaliloglu; Mesrob Ohannessian; Lele Wang; Giuseppe Carenini
>
> **摘要:** Multimodal large language models are increasingly capable of complex reasoning, yet their performance often degrades when they must externalize a problem through a tool and then reason over the tool's output, specifically when they rely on visual aids. This gap is especially important because real engineering and scientific workflows often rely on visualization tools for analysis, validation, and decision-making. To study this discrepancy, we introduce VAMPS (Visual-Assisted Mathematical Problem Solving), a benchmark for graph-assisted mathematics. VAMPS contains 1,168 multimodal, bilingual multiple-choice question-answer pairs drawn from Iranian University Entrance Exam algebra and calculus problems and expanded with human-reviewed LLM-generated synthetic variants, all selected so that plotting provides a natural solution strategy by revealing intersections, extrema, asymptotes, etc. Designed for both benchmarking and diagnosis, VAMPS goes beyond prior multimodal benchmarks that primarily evaluate reasoning over fixed visual inputs by testing whether a model can benefit from constructing a useful graph and grounding its answer in the resulting visualization. Overall, we found that across a diverse set of models, direct analytical solving surprisingly outperforms tool-enabled visual solving, even on problems where plotting is a natural strategy.
>
---
#### [new 105] Audio Interaction Model
- **分类: cs.SD; cs.AI; cs.CL; cs.MM; eess.AS**

- **简介: 该论文提出Audio-Interaction模型，解决传统音频模型任务单一、无法实时交互的问题。通过统一的在线音频语言模型，实现多任务实时音频处理与响应。**

- **链接: [https://arxiv.org/pdf/2606.05121](https://arxiv.org/pdf/2606.05121)**

> **作者:** Zhifei Xie; Zihang Liu; Ze An; Xiaobin Hu; Yue Liao; Ziyang Ma; Dongchao Yang; Mingbao Lin; Deheng Ye; Shuicheng Yan; Chunyan Miao
>
> **备注:** Next generation of LALMs, work in progress
>
> **摘要:** Audio is an inherently interactive modality, yet today's Large Audio Language Models (LALMs) are offline, and streaming audio models each handle only a single task such as streaming ASR or voice chatting. It is time to unify them into one online LALM: a model that, through an always-on perceive-decide-respond loop, listens to sound, environment, and instructions in real time and reacts on the fly. We formalize this regime as the Audio Interaction Model, and realize it with Audio-Interaction, a unified streaming model that retains offline task execution while adding online general audio instruction following, from dialogue to full voice chatting, deciding when to respond from the semantics of the stream. To enable this, we propose SoundFlow, a framework that instantiates the perceive-decide-respond loop end to end, from data to training to deployment, through streaming-native data construction, comprehension-aware training, and asynchronous low-latency inference for stable real-time interaction. We further construct StreamAudio-2M, a 2.6M-item streaming corpus spanning 7 fundamental abilities and 28 sub-tasks, and Proactive-Sound-Bench for evaluating proactive audio intervention. Across 8 benchmarks, Audio-Interaction preserves competitive performance on mainstream audio tasks while unlocking capabilities inaccessible to offline LALMs, including real-time ASR, streaming audio instruction following, and proactive help.
>
---
#### [new 106] Video2LoRA: Parametric Video Internalization for Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出Video2LoRA，解决视频在视觉-语言模型中处理成本高的问题。通过生成低秩适配器，实现高效视频内化，减少视觉token使用并提升推理速度。**

- **链接: [https://arxiv.org/pdf/2606.04351](https://arxiv.org/pdf/2606.04351)**

> **作者:** Manan Suri; Sarvesh Baskar; Dinesh Manocha
>
> **摘要:** Processing video in vision-language models is expensive: each frame occupies hundreds of tokens, and inference cost scales with every frame and every repeated query. We introduce Video2LoRA, a method for parametric video internalization. A perceiver hypernetwork reads the intermediate representations produced layer-by-layer as a frozen VLM encodes a video, and generates a Low-Rank Adaptation (LoRA) adapter in a single forward pass. Unlike standard LoRA fine-tuning, which requires iterative gradient updates, Video2LoRA predicts these weights directly from the video. Trained for SmolVLM2 500M and 2.2B on video summarization and captioning, Video2LoRA enables the same frozen VLM to answer queries from the adapter alone, with zero visual tokens in its context at query time. Video2LoRA is statistically non-inferior and equivalent to direct video-in-context inference across all five captioning benchmarks at both model scales, and across seven of eight video question answering benchmark-scale pairings. Although trained only on 12 frames at 384px, it remains stable up to 1,024 frames and 1024px, where direct video-in-context inference often degenerates. Across this sweep, it reduces answer-time visual-token load by up to 1,500x and query TTFT by 6-80x, while preserving video-faithful outputs. We also find that independently generated adapters for non-overlapping video segments can compose in rank space, suggesting a path toward chunked long-video internalization.
>
---
#### [new 107] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文针对具身视觉-语言决策任务中的感知瓶颈问题，提出SceneDiver方法，通过分层聚焦计划生成提升模型对关键对象的识别能力，减少视觉幻觉。**

- **链接: [https://arxiv.org/pdf/2606.04046](https://arxiv.org/pdf/2606.04046)**

> **作者:** Boyuan Xiao; Bohong Chen; Yumeng Li; Ji Feng; Yao-Xiang Ding; Kun Zhou
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** In embodied vision-language decision making tasks such as robotic manipulation and navigation, Vision-Language and Vision-Language-Action Models (VLMs & VLAs) are powerful tools with different benefits: VLMs are better at long-term planning, while VLAs are better at reactive control. However, their performance is limited by the same perceptual bottleneck: visual hallucinations arise due to the models' inability to distinguish task-relevant objects from distractors. In principle, accurate identification and focus on critical objects while filtering out irrelevant ones is the key to break this limitation. A straightforward solution is one-step focus: directly attending to essential objects. However, this approach proves ineffective because effective focus inherently requires deep scene understanding. To this end, we propose SceneDiver, a coarse-to-fine focus plan generation method for VLMs leveraging their long-term planning abilities, that first constructs a holistic scene graph to establish initial comprehension, then progressively decomposes the task into simpler sub-problems through an iterative cycle of recognition, understanding, and analysis. To enable reactive control, we also design a lightweight adapter for distilling the deliberate focus ability into VLAs. Evaluations on standard embodied AI benchmarks confirm that our method substantially reduces visual hallucinations for both VLMs and VLAs, while preserving computational efficiency in tasks requiring fast execution. Our code and data are released at: this https URL.
>
---
#### [new 108] Token Rankings are Unforgeable Language Model Signatures
- **分类: cs.CR; cs.AI; cs.CC; cs.CL**

- **简介: 该论文属于模型安全任务，研究语言模型的token排名是否可作为不可伪造的签名。工作表明，排名具有唯一性且难以伪造，可用于保护模型参数不被泄露。**

- **链接: [https://arxiv.org/pdf/2606.04459](https://arxiv.org/pdf/2606.04459)**

> **作者:** Matthew Finlayson; Andreas Grivas; Xiang Ren; Swabha Swayamdipta
>
> **摘要:** Language model parameters are known to impose unique (to each model) geometric constraints on their logit outputs, which serves as a signature that identifies the model, but also leaks the model's final layer parameters when an API distributes logits. We investigate more restrictive APIs that expose token rankings (i.e., their ordering by probability, but not the probability values) and find that rankings also constitute a signature: every model has a unique set of feasible top-$k$ rankings for sufficiently large $k$. Furthermore, the ranking signature is the first known (polynomially) unforgeable signature, since finding a model with the same set of feasible rankings is NP-hard. On the security front, we find that token rankings are already sufficient to approximately steal the final layer of the model, similar to logits, though the approximation is too coarse to forge the signature, and can be effectively countered by restricting the API to top-$k$ tokens with sufficiently small $k$. Since the top-$k$ required to present the model signature is generally smaller than the $k$ required to prevent stealing, it is possible for an API to present an unforgeable signature without leaking model parameters.
>
---
#### [new 109] STRIDE: Training Data Attribution via Sparse Recovery from Subset Perturbations
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出STRIDE框架，解决大语言模型训练数据归属问题。通过激活空间中的稀疏恢复方法，高效追踪训练数据对模型预测的影响。**

- **链接: [https://arxiv.org/pdf/2606.05165](https://arxiv.org/pdf/2606.05165)**

> **作者:** Rishit Dagli; Abir Harrasse; Luke Zhang; Florent Draye; Amirali Abdullah; Bernhard Schölkopf; Zhijing Jin
>
> **备注:** project page: this https URL
>
> **摘要:** Training Data Attribution (TDA) seeks to trace a model's predictions back to its training data. The gold standard for TDA relies on causal interventions, observing how a model changes when data is added or removed, but repeated retraining is computationally challenging for Large Language Models (LLMs). Consequently, most approaches approximate this effect in the parameter space using gradients. However, tracking gradients across billions of parameters is not only prohibitively expensive but relies on local approximations. In this work, we propose a shift: rather than estimating parameter changes, we model the functional effect of training data in the activation space. We introduce STRIDE (Steering-based Training Data Influence Decomposition), a framework that formulates TDA as a sparse recovery problem in the spirit of compressive sensing. STRIDE learns lightweight "steering operators" that mimic the behavioral shift caused by training on data subsets. By measuring how these operators perturb test predictions, we recover individual training example influences via sparse linear decomposition. STRIDE achieves state-of-the-art for LLM pre-training attribution while being an order of magnitude ($13\times$) faster than previous art. We further validate its practical utility through downstream applications including data selection, data contamination, and qualitative analysis.
>
---
#### [new 110] Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models
- **分类: cs.SD; cs.CL**

- **简介: 该论文研究音频-语言模型在音频与文本冲突时的决策机制，旨在解决模型偏好文本而非音频的问题。通过实验发现音频信息被编码但被文本覆盖，并提出GACL方法提升模型对音频的依赖。**

- **链接: [https://arxiv.org/pdf/2606.05161](https://arxiv.org/pdf/2606.05161)**

> **作者:** Yichen Gao; Yiqun Zhang; Zijing Wang; Yujia Li; Heng Guo; Xi Wu; Xiaocui Yang; Shi Feng; Yifei Zhang; Daling Wang
>
> **摘要:** Audio-language models (ALMs) often follow text that conflicts with audio, even when the audio evidence is clear. This raises a basic question: is the audio-supported answer unavailable, or is it represented but overridden by the conflicting text? We examine this question using a same-audio counterfactual that keeps the audio fixed, removes only the conflicting text, and measures the resulting shift in model preference. Across five ALMs and four conflict tasks, 64.1% of conflict samples show a sign flip: the same-audio branch prefers the audio-supported answer, whereas the joint branch prefers the text-supported answer. This pattern suggests that the relevant audio evidence is encoded but loses in arbitration. Activation patching further localizes the reversal to answer-position computation, and patching effects closely track output candidate-score differences (Spearman rho=0.93). Using this diagnostic, we propose Gated Audio Counterfactual Logit Correction (GACL), a training-free decoding rule that interpolates between joint and same-audio scores. Under a strict 5 pp faithfulness-drop budget, GACL improves nAUC by 17.8 points over the best contrastive baseline and transfers without retuning to vision-text arbitration (up to +40.5 pp).
>
---
#### [new 111] Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于大语言模型个性化任务，旨在解决个性化与效率之间的平衡问题。提出TAP-PER框架，通过轻量级用户状态前缀实现高效个性化。**

- **链接: [https://arxiv.org/pdf/2606.04547](https://arxiv.org/pdf/2606.04547)**

> **作者:** Heng Cao; Fan Zhang; Jian Yao; Yujie Zheng; Changlin Zhao; Lu Hao; Yuxuan Wei; Wangze Ni; Huaiyu Fu; Yuqian Sun; Xuyan Mo
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, eliminating explicit prompt construction and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes user modeling into user-state and query-conditioned components, and incorporates temporal signals to capture the evolving nature of user interests. Experiments on six LaMP tasks show that TAP-PER consistently outperforms prompt-based and model-based baselines across classification, rating, and generation settings. Moreover, TAP-PER uses 130x fewer per-user parameters than OPPU and roughly half the total parameter footprint of PER-PCS at the 1,000-user scale, demonstrating that scalable LLM personalization can be achieved without explicit prompt construction or heavy per-user adapters.
>
---
#### [new 112] Sparse Mixture-of-Experts Reward Models Learn Interpretable and Specialized Experts for Personalized Preference Modeling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习中的偏好建模任务，旨在解决传统方法无法有效捕捉个性化偏好问题。提出稀疏专家混合奖励模型，提升可解释性和个性化效果。**

- **链接: [https://arxiv.org/pdf/2606.04284](https://arxiv.org/pdf/2606.04284)**

> **作者:** Yifan Wang; Jinyi Mu; Mayank Jobanputra; Yu Wang; Ji-Ung Lee; Soyoung Oh; Isabel Valera; Vera Demberg
>
> **摘要:** Preference modeling plays a central role in reinforcement learning from human feedback (RLHF), enabling large language models (LLMs) to align with human values. However, most existing approaches assume a universal reward function, neglecting the diversity and heterogeneity of human preferences. To address this limitation without additional annotation costs, recent work has proposed learning multiple preference components from binary data and combining them to model individual preferences. Nevertheless, these components often fail to capture coherent and disentangled patterns, limiting their interpretability and effectiveness for personalization. In this work, we propose a sparse Mixture-of-Experts (MoE) reward model that encourages sparse routing and expert diversity during training on binary preference data. Across controlled and real-world experiments, sparse MoE learns interpretable routing patterns and specialized experts. It also improves test-time personalization, and post-adaptation shifts in expert weights provide a qualitative lens for analyzing how the model adapts to personalized preferences.
>
---
#### [new 113] Training-Free Lexical-Dense Fusion for Conversational-Memory Retrieval
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文属于对话记忆检索任务，解决长对话历史中高效检索相关回合的问题。通过无训练的融合方法提升检索效果，分析不同编码器表现及融合策略的有效性。**

- **链接: [https://arxiv.org/pdf/2606.04194](https://arxiv.org/pdf/2606.04194)**

> **作者:** Christian Lysenstøen
>
> **备注:** 9 pages, 3 figures, 10 tables. Code, data, and per-table receipts: this https URL
>
> **摘要:** Retrieving the few past turns that answer a new query across long multi-session histories is the retrieval bottleneck behind long-term conversational memory (LoCoMo, LongMemEval). Recent concurrent work, Nano-Memory, shows that scoring a session by the maximum query-turn similarity (late interaction, "Turn Isolation Retrieval") beats mean-pooled session embeddings. We do not claim that effect; we replicate it and ask what a training-free, CPU-only retrieval stage should add around it. We report four findings. (1) Fuse: score-level fusion of the late-interaction dense score with BM25, under a single leave-one-conversation-out weight, adds +8.8 to +17.2 points of LoCoMo Hit@1 over late interaction alone across six encoders (all p<1e-4), reaching Hit@1 0.752 / NDCG@5 0.829 (e5-large-v2), +11.2 pp over BM25. (2) An off-the-shelf web-search cross-encoder reranker over the fused top-10 hurts here, degrading Hit@1 by 6.9 pp (one reranker, one configuration). (3) A pooling-operator ablation shows top-k late interaction matches max-similarity, but a naive smooth-max (log-sum-exp) collapses for half the encoders. (4) The late-minus-early gap is large for all six encoders and tends to be larger for larger ones, while the marginal fusion gain shrinks; on LongMemEval-S, a lexical regime where BM25 saturates, the net fusion gain over BM25 is small and not significant. A per-category analysis frames the gain as a division of labor: dense late interaction helps most on multi-hop and temporal questions but trails BM25 on adversarial ones. The contribution is a controlled, reproducible account of a strong training-free retrieval recipe, not the late-interaction retriever itself (Nano-Memory's). We make no claim to a complete memory architecture; this is a retrieval-stage study.
>
---
#### [new 114] Exploring the Topology and Memory of Consensus: How LLM Agents Agree, Fragment, or Settle When Forming Conventions
- **分类: cs.MA; cs.CL; cs.SI; physics.soc-ph**

- **简介: 该论文研究多智能体系统在形成共识时的拓扑结构与记忆机制的相互作用，探讨如何设计记忆深度和网络结构以优化协调。任务属于分布式协同与共识机制研究。**

- **链接: [https://arxiv.org/pdf/2606.04197](https://arxiv.org/pdf/2606.04197)**

> **作者:** Aliakbar Mehdizadeh; Martin Hilbert
>
> **备注:** Submitted to the Journal of Artificial Societies and Social Simulation (JASSS)
>
> **摘要:** How much should an LLM agent remember, and how should multi-agent systems be connected when trying to reach consensus? We show these two design choices interact in a way that flips the sign of memory's effect on coordination. Across 432 simulation runs of a networked Naming Game on eight fixed 16-agent topologies, we vary memory depth and network structure. Longer memory slows the time to reach steady state in decentralized networks but accelerates it in centralized ones; the same parameter pushes the system in opposite directions depending on topology. Critically, "faster settling" in centralized networks means locking in to a fragmented plateau more quickly, not reaching system-wide consensus, which can be used to generate diverging opinions. We further document a memory-mediated speed-unity trade-off: centralized networks consistently preserve more competing conventions than decentralized networks, but their settling speed depends sharply on memory. At the agent level, within-network analyses show that high-betweenness bridges suffer a brokerage penalty while agents in locally clustered neighborhoods achieve higher coordination success. Finally, in search of analytically tractable generative mechanisms, we find that agents' choices are well captured by Fictitious Play, indicating belief-based rather than reward-based adaptation. The practical implication: memory depth and communication topology should be co-designed, not optimized in isolation.
>
---
#### [new 115] Read What You Hear: Reference-Free Hypotheses Evaluation with Acoustic Discrepancy
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决无参考评估问题。提出READ方法，通过声学差异评估ASR假设，无需额外训练即可提升识别效果。**

- **链接: [https://arxiv.org/pdf/2606.04680](https://arxiv.org/pdf/2606.04680)**

> **作者:** Zhihan Li; Hankun Wang; Yiwei Guo; Bohan Li; Xie Chen; Kai Yu
>
> **备注:** Submitted to Interspeech 2026. 6 pages, 4 figures
>
> **摘要:** Automatic speech recognition systems commonly rely on reference transcriptions for evaluation, while reference-free approaches often depend on internal confidence estimation or auxiliary language models. We propose READ (Reference-free Hypothesis Evaluation with Acoustic Discrepancy), a novel metric that evaluates ASR hypotheses directly from the speech signal. READ emphasizes the acoustic grounding of hypotheses. It uses a pretrained auto-regressive TTS model to compute the conditional likelihood of speech tokens given a text hypothesis, to measure fine-grained acoustic discrepancy between speech and text. Without additional training, READ can be applied for hypothesis refinement. Experiments show that READ correlates with specific recognition errors and improves ASR outputs, achieving up to 20\% relative error rate reduction, with particularly strong gains under noisy conditions.
>
---
#### [new 116] Physics-Informed Neural Network Modeling of Biodegradable Contaminant Transport through GCL/SL Composite Liners
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于污染物迁移建模任务，解决GCL/SL复合衬层中污染物传输问题。通过构建物理信息神经网络，提升预测精度并实现反演分析。**

- **链接: [https://arxiv.org/pdf/2606.04392](https://arxiv.org/pdf/2606.04392)**

> **作者:** Dong Li; Yapeng Cao; Haiping Zhao; Shutong Han
>
> **摘要:** This study develops a two-domain physics-informed neural network framework for contaminant transport through a GCL/SL composite liner system, in which the thin GCL layer is treated using a steady-state advection-dispersion-biodegradation formulation and the underlying soil liner is modeled as a transient transport domain. Two formulations are evaluated against analytical and finite-element reference solutions under different leachate-head conditions: a standard PINN with soft constraint enforcement (Std-PINN) and a hard-constrained PINN (H-PINN), in which selected boundary and initial conditions are embedded directly into the trial solutions. The Std-PINN captures the overall breakthrough behavior but shows larger errors during the early transport stage, particularly under higher leachate heads where advective transport becomes more pronounced. The H-PINN reduces the optimization burden associated with penalty-based constraint enforcement and provides more accurate and stable concentration predictions, lowering the MAE from approximately 0.058-0.067 for the Std-PINN to about 0.011-0.023 for the H-PINN, while reducing the MRE from approximately 9.10%-19.16% to about 2.08%-3.14%. Parametric analyses confirm that the H-PINN with the tanh activation function and an optimized network structure provides the best predictive accuracy. The H-PINN is further extended to inverse modeling for identifying the SL degradation half-life from limited concentration observations, showing reliable convergence toward prescribed values and acceptable robustness under low-to-moderate observation noise.
>
---
#### [new 117] Overview of the EReL@MIR 2025 Multimodal Document Retrieval Challenge (Track 1)
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文介绍EReL@MIR 2025多模态文档检索挑战，解决文本与视觉信息联合检索问题，设计数据集与评估方案，并分析获胜系统。**

- **链接: [https://arxiv.org/pdf/2606.04240](https://arxiv.org/pdf/2606.04240)**

> **作者:** Jingbiao Mei
>
> **备注:** MDR Challenge Report at WWW2025
>
> **摘要:** Retrieval over visually-rich documents, pages that interleave text with figures, tables, and charts, is essential for multimodal retrieval-augmented generation, yet most retrievers still discard the visual channel. The \emph{Multimodal Document Retrieval Challenge}, Track~1 of the MIR Challenge at the first EReL@MIR workshop, co-located with The Web Conference 2025, asks participants to build a \emph{single} retrieval system that handles two complementary regimes: closed-set document page retrieval within long documents from a text query (MMDocIR), and open-domain retrieval of Wikipedia-style passages from an image or image-plus-text query (M2KR). Systems are ranked by the macro-average of mean Recall@$\{1,3,5\}$ over the two tasks. The challenge drew 455 entrants and 586 submissions across 22 teams. This report describes the challenge design, datasets, and evaluation protocol; reports the final standings; and analyses the three winning teams' systems. All three build on decoder-based Multimodal-LLM embedders from the Qwen2-VL family rather than on CLIP-style encoders, and differ chiefly in whether they reach the top through fine-tuned ensembles, training-free multi-route fusion with a strong vision-language re-ranker, or zero-shot late interaction. The training-free system finished within $0.1$ point of the fine-tuned winner.
>
---
#### [new 118] Evaluating Reasoning Fidelity in Visual Text Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉文本生成任务，旨在评估模型在生成文本时的推理一致性。研究发现当前模型在复杂推理任务中存在语义错误和逻辑不一致问题。**

- **链接: [https://arxiv.org/pdf/2606.04479](https://arxiv.org/pdf/2606.04479)**

> **作者:** Jiajun Hong; Jiawei Zhou
>
> **备注:** Peer reviewed and accepted at CVPR 2026 at the GRAIL-V (Grounded Retrieval and Agentic Intelligence for Vision-Language) workshop (non-archival track)
>
> **摘要:** Recent text-to-image (T2I) models can render highly legible and well-structured text within images, enabling applications including document generation and slide generation. However, it remains unclear whether such systems faithfully preserve reasoning ability when complex solutions must be expressed directly through rendered text, or whether they merely imitate surface-level patterns. We investigate this question by evaluating reasoning fidelity in visual text generation, where models must express complete reasoning processes as images. Our evaluation includes long text rendering, factual knowledge probing, context understanding, and multi-step reasoning. Across these settings, we find that current T2I models frequently produce semantic errors, logical inconsistencies, and incorrect intermediate steps, even when the rendered text appears visually clear. These failures contrast with the strong reasoning performance of text-only models on the same tasks. Our findings reveal a substantial gap between visual text generation and procedural reasoning, motivating more reliable visual text reasoning.
>
---
#### [new 119] Failed Reasoning Traces Tell You What Is Fixable (But Not by Reading Them)
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言模型故障分析任务，旨在解决失败推理轨迹的诊断问题。通过分析失败轨迹的结构特征，识别可修复与不可修复的失败，并提出无需训练的路由规则提升修复效果。**

- **链接: [https://arxiv.org/pdf/2606.05145](https://arxiv.org/pdf/2606.05145)**

> **作者:** Nizar Islah; Istabrak Abbes; Irina Rish; Sarath Chandar; Eilif B. Muller
>
> **摘要:** When post-trained language models fail on reasoning problems, the common test-time-scaling response is to spend more compute on additional attempts, and the failed traces play no further role. We argue this discards a crucial signal; some failures come from unlucky sampling, where more rollouts help, while others are structural and resist resampling regardless of budget. We propose that failed traces encode recoverability structure: the inference-time signature of which test-time interventions can rescue a given failure. Three problem-level trajectory features, derived from the structure of available interventions, recover this structure from the distributional signature of failed rollouts, not their text. They cluster failures into stable regimes, characterize the failure topography of different post-training methods ($84.3{\pm}4.3\%$ accuracy, $+20\%$ over a majority-class baseline), and support a training-free routing rule that lifts rescue by $+12.2\%$ on the deployment-relevant Steerable-Hard subset (failures where retry is insufficient and a bounded intervention is reachable). The features and the routing rule transfer across two cross-family probes. The same three features thus convert failed traces from discarded data into a diagnostic object, supporting test-time routing and post-training analysis without training-time or weight-space access.
>
---
#### [new 120] MusaCoder: Native GPU Kernel Generation with Full-Stack Training on Moore Threads GPU
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文提出MusaCoder，解决GPU核代码生成问题。通过全栈训练框架，提升生成代码的正确性和效率，适用于CUDA和MUSA后端。**

- **链接: [https://arxiv.org/pdf/2606.04847](https://arxiv.org/pdf/2606.04847)**

> **作者:** Kun Cheng; Songshuo Lu; Sicong Liao; Tankun Li; Yafei Zhang; Dong Yang; Qiheng Lv; Hua Wang; Zhi Chen; Yaohua Tang
>
> **摘要:** Native GPU kernel generation turns high-level tensor programs into executable, efficient low-level code. Existing Large Language Models (LLMs) struggle with this task, while execution-based reinforcement learning suffers from sparse rewards, reward hacking, and training instability. We present MusaCoder, a full-stack training framework for native GPU kernel generation on CUDA and MUSA backends. MusaCoder combines progressive kernel-oriented data synthesis, diversity-preserving rejection fine-tuning, and execution-feedback Reinforcement Learning (RL) through MooreEval, a distributed verifier and reward environment. To stabilize RL, MusaCoder introduces PrimeEcho for first-turn-anchored multi-turn rewards, Buffered Dynamic Retry for recovering signals from all-failed hard samples, and MirrorPop for off-policy sequence filtering. Experiments on KernelBench and a MUSA-ported variant show that MusaCoder outperforms strong open-source and proprietary baselines in both correctness and empirical speedup, with the 9B model matching or exceeding frontier closed-source models and the 27B model establishing a new state of the art. These results demonstrate not only the effectiveness of full-stack execution-feedback training for native kernel generation, but also the capability of Moore Threads GPUs to support the complete LLM post-training stack, providing a practical foundation for large-model training and optimization on emerging accelerators.
>
---
#### [new 121] Global Sketch-Based Watermarking for Diffusion Language Models
- **分类: cs.CR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于语言模型水印任务，旨在解决扩散模型中难以应用传统水印方法的问题。提出一种基于全局草图的水印方法，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.04486](https://arxiv.org/pdf/2606.04486)**

> **作者:** Daniel Zhao
>
> **摘要:** Watermarking methods for language models have been studied extensively in the autoregressive setting, where tokens are generated sequentially. These works largely focus on local-context schemes that perturb the next token's distribution as a function of its preceding tokens. In diffusion language models, distributions over many unresolved positions are jointly sampled, allowing additive statistics of the entire sequence to be tractable during generation. We propose a watermark for masked diffusion language models that controls a global, vector-valued sketch representation of the text. Compared to context-dependent watermarking, the sketch formulation decouples detection from the local contexts seen during generation, resulting in an order-agnostic statistic and a watermarking rule which does not manifest as a simple token bias. We analyze the distortion, soundness, and robustness properties of the method.
>
---
#### [new 122] The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development?
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Meta-Agent Challenge（MAC），评估AI是否能自主开发代理系统。任务是测试模型的自主代理开发能力，解决当前评估框架无法衡量此能力的问题。工作包括设计评估框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2606.04455](https://arxiv.org/pdf/2606.04455)**

> **作者:** Xinyu Lu; Tianshu Wang; Pengbo Wang; zujie wen; Zhiqiang Zhang; Jun Zhou; Boxi Cao; Yaojie Lu; Hongyu Lin; Xianpei Han; Le Sun
>
> **备注:** Website: this https URL
>
> **摘要:** Current AI benchmarks evaluate agents on task execution within human-designed workflows. These evaluations fundamentally fail to measure a critical next-level capability: whether models can autonomously develop agent systems. We introduce the Meta-Agent Challenge (MAC), an evaluation framework designed to test the capacity of frontier models for autonomous agent development. Specifically, a code agent (the meta-agent) is given a sandboxed environment, an evaluation API, and a time limitation to iteratively program an agent artifact that maximizes performance on a held-out test set across five domains. To ensure evaluation integrity, this framework is secured by multi-layer defenses against reward hacking. Leveraging this framework, we demonstrate that meta-agents rarely match human-engineered baseline policies, and the few that do are dominated by proprietary frontier models. Moreover, the design process exhibits high variance, and high optimization pressure surfaces emergent adversarial behaviors like ground-truth exfiltration-highlighting critical deficits in both robustness and model alignment. Ultimately, MAC provides a rigorous, open-source benchmark for autonomous AI research and development, offering an empirical proxy for evaluating recursive self-improvement. Benchmark is publicly available at: this https URL.
>
---
#### [new 123] SocialCoach: Personalized Social Skill Learning with RL-based Agentic Tutoring and Practice
- **分类: cs.HC; cs.CL; cs.CY**

- **简介: 该论文提出SocialCoach，解决社交技能训练个性化与规模化难题。通过强化学习与多代理系统，实现个性化教学与评估，提升学习效果。**

- **链接: [https://arxiv.org/pdf/2606.04155](https://arxiv.org/pdf/2606.04155)**

> **作者:** Tianfu Wang; Max Xiong; Jianxun Lian; Hongyuan Zhu; Zhengyu Hu; Yuxuan Lei; Linxiao Gong; Xiaofang Li; Peiting Tsai; Nicholas Jing Yuan; Qi Zhang
>
> **摘要:** Social skills such as negotiation and leadership are crucial for personal and professional success in today's interconnected world. However, scalable and effective training remains a significant challenge due to the scarcity of expert coaching. In this paper, we introduce SocialCoach, a holistic LLM-powered agentic tutoring system for personalized social skill development at scale. First, SocialCoach automatically constructs a pedagogically-grounded, theory-to-practice knowledge corpus from diverse expert sources, leveraging a multi-agent pipeline. Second, to personalize the learning journey, it employs an adaptive practice scheduling module that follows a prescription-retrieval-adaptation process. To maximize the long-term learning experience while overcoming the cold-start problem, this policy is optimized within a learner simulation environment through reinforcement learning. Finally, SocialCoach integrates immersive, goal-driven practice, causality-driven proficiency assessment and knowledge-grounded, reflective tutoring to help address the knowing-doing gap. We deploy it in our product, EQoach, and conduct extensive experiments. The results show that SocialCoach improves simulated pathway quality and judge-rated tutoring quality over baseline approaches, while early user feedback indicates strong perceived engagement and usefulness. These findings suggest a practical architecture for personalized and gamified pedagogical platforms on soft skill learning.
>
---
#### [new 124] M$^3$Eval: Multi-Modal Memory Evaluation through Cognitively-Grounded Video Tasks
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出M$^3$Eval，用于评估多模态模型的记忆能力，解决现有研究缺乏系统记忆评估的问题。通过认知心理学设计任务，揭示模型在记忆方面的弱点与特性。**

- **链接: [https://arxiv.org/pdf/2606.05008](https://arxiv.org/pdf/2606.05008)**

> **作者:** Jie Huang; Ruixun Liu; Sirui Sun; Xinyi Yang; Yin Li; Yixin Zhu; Yiwu Zhong
>
> **备注:** We present an evaluation designed for multi-modal memory in multi-modal models
>
> **摘要:** As multi-modal models advance towards long-form video understanding, memory emerges as a critical capability. Despite substantial efforts in developing video datasets and benchmarks, existing works primarily focus on perception and reasoning, without systematically evaluating memory: what models retain, how faithfully information is preserved, and how robust memory remains under interference. To address this gap, we introduce M$^3$Eval, the first comprehensive evaluation framework and benchmark for probing different memory dimensions in multi-modal models. Grounded in cognitive psychology, our design features carefully constructed tasks that isolate key aspects of memory. Leveraging M$^3$Eval, we conduct extensive experiments across representative multi-modal models, revealing consistent weaknesses and distinctive behaviors. We find that models struggle to maintain disentangled representations when processing parallel video streams, exhibit interference patterns differing substantially from those observed in human memory, ground memory sources more reliably in the spatial domain than the temporal domain, and demonstrate limited symbolic memory. Collectively, our benchmark provides a valuable resource for future research, while our findings highlight memory as a fundamental yet underexplored capability and offer insights for designing more effective memory mechanisms in multi-modal models. Our code and dataset are available at this https URL.
>
---
#### [new 125] Do Transformers Need Three Projections? Systematic Study of QKV Variants
- **分类: cs.LG; cs.AI; cs.CL; cs.PF**

- **简介: 该论文研究Transformer中QKV投影的作用，探讨投影共享对模型性能的影响，旨在优化注意力机制以减少内存占用，提升边缘设备推理效率。**

- **链接: [https://arxiv.org/pdf/2606.04032](https://arxiv.org/pdf/2606.04032)**

> **作者:** Ali Kayyam; Anusha Madan Gopal; M Anthony Lewis
>
> **备注:** Accepted at ICML 2026 (PMLR vol. 306). 26 pages, 12 figures, 16 tables. Code: this https URL
>
> **摘要:** Transformers have become the standard solution for various AI tasks, with the query, key, and value (QKV) attention formulation playing a central role. However, the individual contribution of these three projections and the impact of omitting some remain poorly understood. We systematically evaluate three projection sharing constraints: a) Q-K=V (shared key-value), b) Q=K-V (shared query-key), and c) Q=K=V (single projection). The last two variants produce symmetric attention maps; to address this, we also explore asymmetric attention via 2D positional encodings. Through experiments spanning synthetic tasks, vision (MNIST, CIFAR, TinyImageNet, anomaly), and language modeling (300M and 1.2B parameter models on 10B tokens), we discovered that our transformers perform on par or occasionally better than the QKV transformer. In language modeling, Q-K=V projection sharing achieves 50% KV cache reduction with only 3.1% perplexity degradation. Crucially, projection sharing is complementary to head sharing (GQA/MQA): combining Q-K=V with GQA-4 yields 87.5% cache reduction, while Q-K=V + MQA achieves 96.9%, enabling practical on-device inference. We show that Q-K=V preserves quality because keys and values can occupy similar representational spaces and attention operates in a low-rank regime, whereas Q=K-V breaks attention directionality. Our results systematically characterize projection sharing as an underexplored instance of weight tying in attention, with direct, quantifiable inference memory benefits, particularly valuable for edge deployment. The code is publicly available at this https URL
>
---
## 更新

#### [replaced 001] Attention-Based Sampler for Diffusion Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，旨在解决扩散模型采样效率与质量的问题。提出基于注意力的采样方法Attn-Sampler，提升生成质量与并行性。**

- **链接: [https://arxiv.org/pdf/2604.08564](https://arxiv.org/pdf/2604.08564)**

> **作者:** Yuyan Zhou; Kai Syun Hou; Weiyu Chen; James Kwok
>
> **摘要:** Auto-regressive models (ARMs) have established a dominant paradigm in language modeling. However, their strictly sequential sampling paradigm imposes fundamental constraints on both inference efficiency and modeling flexibility. To address these limitations, diffusion-based large language models (dLLMs) have been proposed, offering the potential for parallel sampling and flexible language modeling. Despite these advantages, current dLLMs sampling strategies rely primarily on token level information, which fails to account for global sequence structure and often yields suboptimal results. In this paper, we study the sampling order selection problem from the perspective of log-likelihood maximization. We show that this problem is NP-hard and propose an optimal sampling-rank-based approximation that makes the objective computationally tractable. We further prove that the tractable objective is optimized by sampling tokens in descending order of their attention-matrix column sums. This finding provides a principled justification for attention-guided sampling and offers a theoretically grounded alternative to greedy search. We instantiate this theoretical insight in a new training-free sampling algorithm, termed Attn-Sampler, and further propose dynamic attention thresholding for practical acceleration. Extensive experiments across multiple benchmarks validate the effectiveness of our proposed method, demonstrating that it achieves superior generation quality while enhancing the sampling parallelism.
>
---
#### [replaced 002] REFLEX: Self-Refining Explainable Fact-Checking via Verdict-Anchored Style Control
- **分类: cs.CL**

- **简介: 该论文属于可解释事实核查任务，旨在解决LLM生成解释不忠实的问题。通过自修正机制控制推理风格，提升核查准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.20233](https://arxiv.org/pdf/2511.20233)**

> **作者:** Chuyi Kong; Wei Gao; Jing Ma; Hongzhan Lin; Yuxi Sun
>
> **摘要:** The prevalence of fake news on social media demands automated fact-checking systems to provide accurate verdicts with faithful explanations. However, existing large language model (LLM)-based approaches ignore deceptive misinformation styles in LLM-generated explanations, resulting in unfaithful rationales that can mislead human judgments. They rely heavily on external knowledge sources, introducing hallucinations and even high latency that undermine reliability and responsiveness, which is crucial for real-time use. To address these challenges, we propose REason-guided Fact-checking with Latent EXplanations (REFLEX), a self-refining paradigm that explicitly controls reasoning style anchored on verdict. REFLEX utilizes self-disagreement veracity signals between the backbone model and its fine-tuned variant to construct steering vectors, naturally disentangling fact from style. Experiments on the real-world dataset show REFLEX achieves state-of-the-art performance under LLaMA-series models with only 465 self-refined samples. Moreover, owing to its transferability, REFLEX yields up to a 7.54% gain on in-the-wild data. Our results further demonstrate that our method effectively mitigates faithful hallucination, thereby guiding the model toward more accurate verdicts than previous works in explainable fact-checking.
>
---
#### [replaced 003] Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究模型在不同部署架构下的安全性表现，探讨评估条件对安全测量的影响。任务属于AI安全性评估，解决模型安全性在不同框架下不一致的问题，通过实验分析不同配置的影响。**

- **链接: [https://arxiv.org/pdf/2603.10044](https://arxiv.org/pdf/2603.10044)**

> **作者:** David Gringras
>
> **备注:** 74 pages including appendices. 6 frontier models, 62,808 primary observations (~89k total). Pre-registered: OSF DOI https://doi.org/10.17605/OSF.IO/CJW92. Code and data: this https URL
>
> **摘要:** A safety score earned on a benchmark need not predict how the same model behaves once it is wrapped in an agentic scaffold the benchmark never tested. We ran six frontier models through four deployment configurations (direct API, ReAct, multi-agent critic, map-reduce delegation): N = 62,808 blinded, pre-registered, equivalence-tested evaluations across four safety benchmarks (BBQ, TruthfulQA, XSTest/OR-Bench, sycophancy), plus three supporting analyses. ReAct and multi-agent scaffolds stay within a pre-registered +/-2 pp equivalence margin; map-reduce delegation degrades measured safety (NNH = 14), though that loss is largely a measurement artifact: on identical items, multiple-choice versus open-ended phrasing shifts the measured safety rate by 5-20 pp, and decomposition silently strips the multiple-choice options. Roughly 40-89% of the per-model map-reduce loss is this format conversion rather than reasoning disruption, and an option-preserving variant recovers most of it. Pooled effects also mask sharp model-by-scaffold heterogeneity: under map-reduce, on identical items, Opus loses 16.8 pp while Llama 4 gains 18.8 pp. Structurally, scaffold architecture explains only 0.4% of outcome variance (benchmark choice explains 45x more), and the generalizability coefficient is G = 0.000 (bootstrap 95% CI [0.000, 0.752]). An interval that wide is enough on its own to undermine the utility of any single composite safety number as a deployment criterion. These are the "easy cases"; consequential properties like scheming and CBRN uplift have no obvious reason to be less format- or scaffold-sensitive. Code, data, and prompts are released as ScaffoldSafety.
>
---
#### [replaced 004] Can Reasoning Path still be Effective as Input? Bridging Post-Reasoning to Chain-of-Thought Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决长链式推理（CoT）效率低的问题。通过提出后推理框架UCoT，压缩CoT长度并保持推理性能。**

- **链接: [https://arxiv.org/pdf/2510.08647](https://arxiv.org/pdf/2510.08647)**

> **作者:** Chengzhengxu Li; Xiaoming Liu; Zhaohan Zhang; Shengchao Liu; Guoxin Ma; Yu Lan; Cong Wang; Chao Shen
>
> **备注:** ACL 2026 Main Track
>
> **摘要:** Recent developments have enabled advanced reasoning in Large Language Models (LLMs) via long Chain-of-Thought (CoT), trading efficiency during inference for performance. Existing works focus on compressing generated CoT in reasoning, which impairs the necessary information for deriving the correct answer. In this work, we propose post-reasoning, a reasoning paradigm that takes CoT as a part of context to simplify the reasoning task for LLMs. We find that post-reasoning significantly reduces the generation length of LLMs, but its effectiveness hinges on the efficiency and the reliability of the contextual CoT generation. Therefore, we propose Upfront CoT (UCoT), an efficient post-reasoning framework for CoT compression. UCoT trains a lightweight model (compressor) to provide contextual CoT in form of soft tokens and trains the LLM (executor) to leverage this contextual CoT for producing the final answer. Extensive experiments show that UCoT maintains the powerful reasoning ability of executor while significantly reducing the length of CoT. It is worth mentioning that when applying UCoT to the Qwen2.5-7B-Instruct model, the usage of tokens on GSM8K dataset is reduced by 50%, while the performance is 3.08% higher than that of the state-of-the-art (SOTA) method.
>
---
#### [replaced 005] Emotion Entanglement and Bayesian Inference for Multi-Dimensional Emotion Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感理解任务，旨在解决多维情感推理问题。提出EmoScene基准和贝叶斯推理框架，提升情感预测的结构一致性。**

- **链接: [https://arxiv.org/pdf/2604.00819](https://arxiv.org/pdf/2604.00819)**

> **作者:** Hemanth Kotaprolu; Kishan Maharaj; Raey Zhao; Abhijit Mishra; Pushpak Bhattacharyya
>
> **备注:** 19 pages in total, 10 Figures, 7 Tables
>
> **摘要:** Understanding emotions in natural language is inherently a multi-dimensional reasoning problem, where multiple affective signals interact through context, interpersonal relations, and situational cues. However, most existing emotion understanding benchmarks rely on short texts and predefined emotion labels, reducing this process to independent label prediction and ignoring the structured dependencies among emotions. To address this limitation, we introduce Emotional Scenarios (EmoScene), a theory-grounded benchmark of 4,731 contextrich scenarios annotated with an 8-dimensional emotion vector derived from Plutchik's basic emotions. Motivated by the observation that emotions rarely occur independently, we further propose an entanglement-aware Bayesian inference framework that incorporates emotion co-occurrence statistics to perform joint posterior inference over the emotion vector. This lightweight post-processing does not require any parameter updates and improves the structural consistency of predictions, and yields overall gains of 2.24% Lexical Accuracy without any additional cost. EmoScene therefore provides a challenging benchmark for studying multi-dimensional emotion understanding and the limitations of current language models.
>
---
#### [replaced 006] Automated Lexical Coverage for Language Learning: From General to Specialized Word Lists
- **分类: cs.CL**

- **简介: 该论文属于语言学习任务，旨在解决通用词表覆盖不足的问题。通过自动化生成特定文本的专用词表，提高词汇覆盖率并减少词汇量。**

- **链接: [https://arxiv.org/pdf/2512.15552](https://arxiv.org/pdf/2512.15552)**

> **作者:** Dakota Ellis; Samy Babikerali; Wanshan Chen; Bao Dinh; Uyen Le
>
> **摘要:** A General Service List (GSL) is a commonly used resource for language learners to identify important English words. Traditional GSL creation is resource-intensive, relying on linguistic expertise and subjective input. We created our own GSL and evaluated its performance against the New General Service List (NGSL). We found that creating a Specialized Word List (SWL), tailored to a specific text, is a practical method for language learners. Because an SWL is derived from the target text itself, it reaches the 95% coverage required for language comprehension by construction, and it does so with substantially fewer words than a general list applied to the same text: across nine texts spanning fiction, academic papers, and scripts, the NGSL covered 64-85% of each text, whereas a text-specific list reached 95% with far smaller vocabularies. By restricting the SWL process to objective criteria only, it can be automated, scaled, and tailored to the needs of language-learners across the globe.
>
---
#### [replaced 007] Talk is (Not) Cheap: A Taxonomy and Benchmark Coverage Audit for LLM Attacks
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM攻击基准覆盖不足的问题。通过构建STRIDE矩阵，分析六个基准的覆盖情况，发现其覆盖范围有限，部分威胁类别未被测试。**

- **链接: [https://arxiv.org/pdf/2605.15118](https://arxiv.org/pdf/2605.15118)**

> **作者:** Karthik Raghu Iyer; Yazdan Jamshidi; Nicholas Bray; Alexey A. Shvets
>
> **摘要:** We introduce a reusable framework for auditing whether LLM attack benchmarks collectively cover the threat surface: a 4$\times$6 Target $\times$ Technique matrix grounded in STRIDE, constructed from a 507-leaf taxonomy -- 401 data-populated and 106 threat-model-derived leaves -- of inference-time attacks extracted from 932 arXiv security studies (2023--2026). The matrix enables benchmark-external validation -- auditing collective coverage rather than individual benchmark consistency. Applying it to six public benchmarks reveals that the three primary frameworks (HarmBench, InjecAgent, AgentDojo) occupy non-overlapping cells covering at most 25\% of the matrix, while entire STRIDE threat categories (Service Disruption, Model Internals) lack any standardized evaluation, despite published attacks in these categories achieving 46$\times$ token amplification and 96\% attack success rates through mechanisms which no benchmark tests. The corpus of 2,521 unique attack groups further reveals pervasive naming fragmentation (up to 29 surface forms for a single attack) and heavy concentration in Safety \& Alignment Bypass, structural properties invisible at smaller scale. The taxonomy, attack records, and coverage mappings are released as extensible artifacts; as new benchmarks emerge, they can be mapped onto the same matrix, enabling the community to track whether evaluation gaps are closing.
>
---
#### [replaced 008] Can VLMs Predict Future States? Bootstrapping World Models from Inverse Dynamics
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究VLMs在前向动力学预测中的能力，解决如何利用逆动力学预测提升未来状态生成的问题。通过两种策略增强FDP性能，并在图像编辑任务中验证效果。**

- **链接: [https://arxiv.org/pdf/2506.06006](https://arxiv.org/pdf/2506.06006)**

> **作者:** Yifu Qiu; Yftah Ziser; Anna Korhonen; Shay B. Cohen; Edoardo M. Ponti
>
> **摘要:** Can unified vision-language models (VLMs) perform forward dynamics prediction (FDP), i.e., predicting the future state (in image form) given the previous observation and an action (in language form)? We find that VLMs struggle to generate physically plausible transitions between frames from instructions. Nevertheless, we identify a crucial asymmetry in multimodal grounding: fine-tuning a VLM to learn inverse dynamics prediction (IDP)-effectively captioning the action between frames-is significantly easier than learning FDP. In turn, IDP can be used to bootstrap FDP through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, IDP can annotate actions for unlabelled pairs of video frame observations to expand the training data scale for FDP. Secondly, IDP can assign rewards to multiple samples of FDP to score them, effectively guiding search at inference time. We evaluate the FDP resulting from both strategies through the task of action-centric image editing on Aurora-Bench with two families of VLMs. Despite remaining general-purpose, our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin between 7% and 13% according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench.
>
---
#### [replaced 009] AUDDT: A Unified Benchmark Toolkit for Audio and Speech Deepfake Detectors
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有基准数据集有限、检测器泛化能力不足的问题。工作包括构建统一的评估工具AUDDT，支持多场景检测分析。**

- **链接: [https://arxiv.org/pdf/2509.21597](https://arxiv.org/pdf/2509.21597)**

> **作者:** Yi Zhu; Heitor R. Guimarães; Arthur Pimentel; Tiago Falk
>
> **摘要:** With the prevalence of artificial intelligence (AI)-generated content, such as audio deepfakes, a large body of recent work has focused on developing deepfake detection techniques. However, existing benchmarks employ a narrow set of datasets, leaving detector generalization to real-world conditions uncertain. In this paper, we systematically review 31 existing audio deepfake datasets and present an open-source benchmarking toolkit called AUDDT (this https URL). The goal of this toolkit is to automate the evaluation of pretrained detectors across a wide range of speech and non-speech audio datasets, giving users direct feedback on the advantages and shortcomings of their deepfake detectors under diverse manipulation types and recording conditions. We start by showcasing the usage of the developed toolkit, the composition of our benchmark, and the breakdown of different deepfake subgroups. Next, we highlight how AUDDT differs from existing benchmarking efforts by enabling large-scale, diverse evaluation across modern spoofing methods and richer attribute-level analysis through comprehensive metadata annotation. Using a widely adopted pretrained deepfake detector, we present in- and out-of-domain detection results, revealing notable performance variability across different conditions and audio manipulation types. Lastly, we also analyze the limitations of these existing datasets and their gaps relative to practical deployment scenarios.
>
---
#### [replaced 010] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型的推理任务，旨在解决工具调用不总是有益的问题。通过自适应调用工具和双模式推理策略，提升模型准确性和效率。**

- **链接: [https://arxiv.org/pdf/2605.19852](https://arxiv.org/pdf/2605.19852)**

> **作者:** Qinghe Ma; Zhen Zhao; Yiming Wu; Jian Zhang; Lei Bai; Yinghuan Shi
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Tool-augmented reasoning has emerged as a promising direction for enhancing the reasoning capabilities of multimodal large language models (MLLMs). However, existing studies mainly focus on enabling models to perform tool invocation, while neglecting the necessity of invoking tools. We argue that tool usage is not always beneficial, as redundant or inappropriate invocations largely increase reasoning overhead and even mislead model predictions. To address this issue, we introduce AutoTool, a model that adaptively decides whether to invoke tools according to the characteristics of each query. Within a reinforcement learning framework, we design an explicit dual-mode reasoning strategy with mode-specific reward functions to guide the model toward producing accurate responses. Moreover, to prevent premature bias toward a single reasoning mode, AutoTool jointly explores and balances tool-assisted and text-centric reasoning throughout training, and promotes free exploration in later stages. Extensive experiments demonstrate that AutoTool exhibits outstanding performance and high efficiency, yielding a 21.8\% accuracy gain on V* benchmark compared to the base model, and a 44.9\% improvement in efficiency over existing tool-augmented methods on POPE benchmark. Code is available at this https URL.
>
---
#### [replaced 011] SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决稀疏注意力机制中的性能下降和梯度不足问题，提出SSA框架通过对齐全注意力与稀疏注意力输出提升模型效果。**

- **链接: [https://arxiv.org/pdf/2511.20102](https://arxiv.org/pdf/2511.20102)**

> **作者:** Zhenyi Shen; Junru Lu; Lin Gui; Jiazheng Li; Yulan He; Di Yin; Xing Sun
>
> **备注:** 34 pages
>
> **摘要:** Sparse attention reduces the quadratic complexity of full self-attention but faces two challenges: (1) an attention gap, where applying sparse attention to full-attention-trained models causes performance degradation due to train-inference distribution mismatch, and (2) a capability gap, where models trained purely with sparse attention lack complete gradient flow, preventing them from matching full-attention performance. We propose SSA (Sparse Sparse Attention), a training framework that integrates both sparse and full attention with bidirectional attention-output alignment. We prove that the approximation error scales linearly with the attention mass dropped under sparse attention, and show that SSA's alignment objective substantially reduces this quantity compared to baselines. Experiments demonstrate that SSA achieves state-of-the-art performance under both inference modes, adapts smoothly to varying sparsity budgets, and demonstrates superior long-context capabilities.
>
---
#### [replaced 012] Effective vocabulary expansion of multilingual language models for extremely low-resource languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决多语言模型对低资源语言支持不足的问题。通过扩展词汇并利用双语词典优化初始化，提升模型在目标语言上的表现。**

- **链接: [https://arxiv.org/pdf/2602.09388](https://arxiv.org/pdf/2602.09388)**

> **作者:** Jianyu Zheng
>
> **备注:** 12 pages, 5 figures, 7 tables, under review
>
> **摘要:** Multilingual pre-trained language models(mPLMs) offer significant benefits for many low-resource languages. To further expand the range of languages these models can support, many works focus on continued pre-training of these models. However, few works address how to extend mPLMs to low-resource languages that were previously unsupported. To tackle this issue, we expand the model's vocabulary using a target language corpus. We then screen out a subset from the model's original vocabulary, which is biased towards representing the source language(e.g. English), and utilize bilingual dictionaries to initialize the representations of the expanded vocabulary. Subsequently, we continue to pre-train the mPLMs using the target language corpus, based on the representations of these expanded vocabulary. Experimental results show that our proposed method outperforms the baseline, which uses randomly initialized expanded vocabulary for continued pre-training, in POS tagging and NER tasks, achieving improvements by 0.54% and 2.60%, respectively. Furthermore, our method demonstrates high robustness in selecting the training corpora, and the models' performance on the source language does not degrade after continued pre-training.
>
---
#### [replaced 013] SciDER: Scientific Data-centric End-to-end Researcher
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出SciDER，解决科学研宄自动化问题。通过多代理系统实现数据驱动的全流程研究，提升适应性与多模态处理能力。**

- **链接: [https://arxiv.org/pdf/2603.01421](https://arxiv.org/pdf/2603.01421)**

> **作者:** Ke Lin; Owais Aijaz; Yilin Lu; Yiyang Luo; Xuehang Guo; Preslav Nakov
>
> **备注:** 10 pages, 8 figures, 7 tables
>
> **摘要:** While large language models accelerate scientific discovery, existing agents face severe limitations in adaptability, domain generalization, and multimodal scalability, often struggling to autonomously process raw, domain-specific experimental data. To overcome these barriers, we introduce SciDER, a multi-agent system designed to flexibly automate the entire research lifecycle. This framework employs a novel data-centric approach and integrates a dynamic multimodal skill system across four specialized sub-agents. Specifically, an ideation agent generates novel hypotheses via Evolutionary Idea Search, a data analysis agent systematically structures raw data, an experimentation agent synthesizes executable code grounded in dataset characteristics, and a critic agent drives iterative self-refinement. To democratize open-source scientific discovery, we release OpenSciDER-SFT-8K, a high-quality execution trajectory dataset, alongside the OpenSciDER-27B fine-tuned model. Across six benchmarks, SciDER and OpenSciDER obtain competitive or leading results, with especially strong gains on data-centric analysis, end-to-end research execution, and multimodal scientific visualization. By integrating data analysis with experimental execution, SciDER bridges the gap between abstract scientific reasoning and reproducible experimentation synthesis.
>
---
#### [replaced 014] Evaluating Autoformalization Robustness via Semantically Similar Paraphrasing
- **分类: cs.CL; cs.LO**

- **简介: 该论文属于autoformalization任务，研究LLMs在面对语义相似的自然语言改写时的鲁棒性问题。通过实验验证语义变化对模型输出的影响。**

- **链接: [https://arxiv.org/pdf/2511.12784](https://arxiv.org/pdf/2511.12784)**

> **作者:** Hayden Moore; Asfahan Shah
>
> **摘要:** Large Language Models (LLMs) have recently emerged as powerful tools for autoformalization. Despite their impressive performance, these models can still struggle to produce grounded and verifiable formalizations. Recent work in text-to-SQL, has revealed that LLMs can be sensitive to paraphrased natural language (NL) inputs, even when high degrees of semantic fidelity are preserved. In this paper, we investigate this claim in the autoformalization domain. Specifically, we evaluate the robustness of LLMs generating formal proofs with semantically similar paraphrased NL statements by measuring semantic and compilation validity. Using the formal benchmarks MiniF2F and Lean 4 version of ProofNet, and two modern LLMs, we generate paraphrased natural language statements and cross-evaluate these statements across both models. The results of this paper reveal performance variability across paraphrased inputs, demonstrating that minor shifts in NL statements can significantly impact model outputs.
>
---
#### [replaced 015] Few Tokens, Big Leverage: Preserving Safety Alignment by Constraining Safety Tokens during Fine-tuning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型安全对齐任务，旨在解决微调过程中安全对齐漂移问题。通过约束安全标记的置信度，保持模型拒绝有害请求的能力，同时不影响下游任务性能。**

- **链接: [https://arxiv.org/pdf/2603.07445](https://arxiv.org/pdf/2603.07445)**

> **作者:** Guoli Wang; Haonan Shi; Tu Ouyang; An Wang
>
> **摘要:** Large language models (LLMs) often require fine-tuning (FT) to perform well on downstream tasks, but FT can induce safety-alignment drift even when the training dataset contains only benign data. Prior work shows that introducing a small fraction of harmful data can substantially compromise LLM refusal behavior, causing LLMs to comply with harmful requests. Existing defense methods often rely on model-wide interventions, such as restricting which parameters are updated or injecting additional safety data, which can limit generality and degrade downstream task performance. To address these limitations, we propose a fine-tuning framework called Preserving Safety Alignment via Constrained Tokens (PACT), which stabilizes the model's confidence on safety tokens. Our approach is motivated by the empirical observation that safety-aligned behavior is reflected in the model's token-level output confidence and is often concentrated on a small subset of safety-related tokens. During downstream fine-tuning, we regularize the fine-tuned model to match the aligned reference model's confidence on safety-related tokens at each response step, while leaving non-safety tokens largely unconstrained to allow effective task adaptation. This targeted constraint prevents alignment drift without imposing global restrictions that typically trade off with model utility. Our code is available at {this https URL}.
>
---
#### [replaced 016] Topics as Proxies for Sociodemographics: How Conversational Context Affects LLM Answers
- **分类: cs.CL**

- **简介: 该论文属于人工智能伦理任务，研究LLM在高风险场景中因对话上下文导致的输出差异问题。通过分析对话主题等特征，发现其对建议有显著影响，需进一步研究以减少偏差。**

- **链接: [https://arxiv.org/pdf/2606.02776](https://arxiv.org/pdf/2606.02776)**

> **作者:** Vera Neplenbroek; Gabriele Sarti; Arianna Bisazza; Raquel Fernández
>
> **摘要:** When large language models (LLMs) are used in high-stakes scenarios, such as legal, medical and financial advice, even a single conversation history is enough to drive differences in outcomes between users. Prior work has demonstrated that this results in outcome disparities between sociodemographic groups, with some groups receiving more advantageous outcomes than others. In this work, we demonstrate that LLMs actually struggle to infer user sociodemographics from a single conversation history and that although there are disparities between sociodemographic groups, they are minimal in magnitude. To investigate what the main driver of these disparities is, we compare user sociodemographics to a range of (psycho)linguistic features of conversations, including conversation topic, emotions, and readability. We find that conversation topics are most predictive of LLM-generated advice within a conversational context, which, to some extent, function as proxies for sociodemographic groups and often affect advice in unpredictable ways. This is cause for concern and highlights the need for future research to better understand and, if needed, mitigate the effect of conversational context on LLM outputs in high-stakes scenarios.
>
---
#### [replaced 017] KITE: Kernelized and Information Theoretic Exemplars for In-Context Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的少样本学习任务，旨在解决如何选择最优示例以提升大模型在少量数据下的性能。工作提出KITE方法，结合信息论与核技巧，实现结构感知的多样化示例选择。**

- **链接: [https://arxiv.org/pdf/2509.15676](https://arxiv.org/pdf/2509.15676)**

> **作者:** Vaibhav Singh; Soumya Suvra Ghosal; Kapu Nirmal Joshua; Soumyabrata Pal; Sayak Ray Chowdhury
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm for adapting large language models (LLMs) to new and data-scarce tasks using only a few carefully selected task-specific examples presented in the prompt. However, given the limited context size of LLMs, a fundamental question arises: Which examples should be selected to maximize performance on a given user query? While nearest-neighbor-based methods like KATE have been widely adopted for this purpose, they suffer from well-known drawbacks in high-dimensional embedding spaces, including poor generalization and a lack of diversity. In this work, we study this problem of example selection in ICL from a principled, information theory-driven perspective. We first model an LLM as a linear function over input embeddings and frame the example selection task as a query-specific optimization problem: selecting a subset of exemplars from a larger example bank that minimizes the prediction error on a specific query. This formulation departs from traditional generalization-focused learning theoretic approaches by targeting accurate prediction for a specific query instance. We derive a principled surrogate objective that is approximately submodular, enabling the use of a greedy algorithm with an approximation guarantee. We further enhance our method by (i) incorporating the kernel trick to operate in high-dimensional feature spaces without explicit mappings, and (ii) introducing an optimal design-based regularizer to encourage diversity in the selected examples. Empirically, we demonstrate significant improvements over standard retrieval methods across a suite of classification tasks, highlighting the benefits of structure-aware, diverse example selection for ICL in real-world, label-scarce scenarios.
>
---
#### [replaced 018] LLMs + Persona-Plug = Personalized LLMs
- **分类: cs.CL**

- **简介: 该论文属于语言模型个性化任务，旨在解决用户偏好差异导致的输出不匹配问题。提出PPlug模型，通过构建用户嵌入提升模型对用户习惯的理解，实现个性化输出。**

- **链接: [https://arxiv.org/pdf/2409.11901](https://arxiv.org/pdf/2409.11901)**

> **作者:** Jiongnan Liu; Yutao Zhu; Shuting Wang; Xiaochi Wei; Erxue Min; Yu Lu; Shuaiqiang Wang; Dawei Yin; Zhicheng Dou
>
> **摘要:** Personalization plays a critical role in numerous language tasks and applications, since users with the same requirements may prefer diverse outputs based on their individual interests. This has led to the development of various personalized approaches aimed at adapting large language models (LLMs) to generate customized outputs aligned with user preferences. Some of them involve fine-tuning a unique personalized LLM for each user, which is too expensive for widespread application. Alternative approaches introduce personalization information in a plug-and-play manner by retrieving the user's relevant historical texts as demonstrations. However, this retrieval-based strategy may break the continuity of the user history and fail to capture the user's overall styles and patterns, hence leading to sub-optimal performance. To address these challenges, we propose a novel personalized LLM model, PPlug. It constructs a user-specific embedding for each individual by modeling all her historical contexts through a lightweight plug-in user embedder module. By attaching this embedding to the task input, LLMs can better understand and capture user habits and preferences, thereby producing more personalized outputs without tuning their own parameters. Extensive experiments on various tasks in the language model personalization (LaMP) benchmark demonstrate that the proposed model significantly outperforms existing personalized LLM approaches.
>
---
#### [replaced 019] MesaNet: Sequence Modeling by Locally Optimal Test-Time Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于序列建模任务，旨在解决Transformer模型推理时计算和内存线性增长的问题。工作是引入一种可并行的Mesa层，通过最优测试时训练提升模型性能。**

- **链接: [https://arxiv.org/pdf/2506.05233](https://arxiv.org/pdf/2506.05233)**

> **作者:** Johannes von Oswald; Nino Scherrer; Seijin Kobayashi; Luca Versari; Songlin Yang; Sarthak Mittal; Maximilian Schlegel; Kaitlin Maile; Yanick Schimpf; Oliver Sieberling; Alexander Meulemans; Rif A. Saurous; Guillaume Lajoie; Charlotte Frenkel; Razvan Pascanu; Blaise Agüera y Arcas; João Sacramento
>
> **备注:** Published at ICLR 2026
>
> **摘要:** Sequence modeling is currently dominated by causal transformer architectures that use softmax self-attention. Although widely adopted, transformers require scaling memory and compute linearly during inference. A recent stream of work linearized the softmax operation, resulting in powerful recurrent neural network (RNN) models with constant memory and compute costs such as DeltaNet, Mamba or xLSTM. These models can be unified by noting that their recurrent layer dynamics can all be derived from an in-context regression objective, approximately optimized through an online learning rule. Here, we join this line of work and introduce a numerically stable, chunkwise parallelizable version of the recently proposed Mesa layer (von Oswald et al., 2024), which could only run sequentially in time and was therefore not scalable. This layer again stems from an in-context loss, but which is now minimized to optimality at every time point using a fast conjugate gradient solver. Through an extensive suite of experiments study up to the billion-parameter scale, we show that optimal test-time training enables reaching lower language modeling perplexity and higher downstream benchmark performance than previous RNNs, especially on tasks requiring long context understanding. This performance gain comes at the cost of additional flops spent during inference time. Our results are therefore intriguingly related to recent trends of increasing test-time compute to improve performance -- here by spending compute to solve sequential optimization problems within the neural network itself.
>
---
#### [replaced 020] MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决LLMs在处理含错误前提的健康问题时无法正确引导的问题。研究构建了MedRedFlag数据集，并对比了LLMs与医生的回答效果。**

- **链接: [https://arxiv.org/pdf/2601.09853](https://arxiv.org/pdf/2601.09853)**

> **作者:** Sraavya Sambara; Yuan Pu; Ayman Ali; Vishala Mishra; Lionel Wong; Monica Agrawal
>
> **摘要:** Real-world health questions from patients often unintentionally embed false assumptions or premises. In such cases, safe medical communication typically involves redirection: addressing the implicit misconception and then responding to the underlying patient context, rather than the original question. While large language models (LLMs) are increasingly being used by lay users for medical advice, they have not yet been tested for this crucial competency. Therefore, in this work, we investigate how LLMs react to false premises embedded within real-world health questions. We develop a semi-automated pipeline to curate MedRedFlag, a dataset of 1100+ questions sourced from Reddit that require redirection. We then systematically compare responses from state-of-the-art LLMs to those from clinicians. Our analysis reveals that LLMs often fail to redirect problematic questions, even when the problematic premise is detected, and provide answers that could lead to suboptimal medical decision making. Our benchmark and results reveal a novel and substantial gap in how LLMs perform under the conditions of real-world health communication, highlighting critical safety concerns for patient-facing medical AI systems. Code and dataset are available at this https URL.
>
---
#### [replaced 021] Enhancing Hallucination Detection through Noise Injection
- **分类: cs.CL; eess.SY**

- **简介: 该论文属于 hallucination 检测任务，旨在提升大语言模型生成内容的准确性。针对模型不确定性，提出一种无需训练的噪声注入方法，有效提升检测效果。**

- **链接: [https://arxiv.org/pdf/2502.03799](https://arxiv.org/pdf/2502.03799)**

> **作者:** Litian Liu; Reza Pourreza; Sunny Panchal; Apratim Bhattacharyya; Yubing Jian; Yao Qin; Roland Memisevic
>
> **备注:** ICLR 2026 main conference paper
>
> **摘要:** Large Language Models (LLMs) are prone to generating plausible yet incorrect responses, known as hallucinations. Effectively detecting hallucinations is therefore crucial for the safe deployment of LLMs. Recent research has linked hallucinations to model uncertainty, suggesting that hallucinations can be detected by measuring dispersion over answer distributions obtained from multiple samples drawn from a model. While drawing from the distribution over tokens defined by the model is a natural way to obtain samples, in this work, we argue that it is suboptimal for the purpose of detecting hallucinations. We show that detection can be improved significantly by taking into account model uncertainty in the Bayesian sense. To this end, we propose a very simple, training-free approach based on perturbing an appropriate subset of model parameters, or equivalently hidden unit activations, during sampling. We demonstrate that our approach significantly improves inference-time hallucination detection over standard sampling across diverse datasets, model architectures, and uncertainty metrics.
>
---
#### [replaced 022] Beyond Correctness: Rewarding Faithful Reasoning in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决agentic search中推理过程不忠实的问题。通过引入VERITAS框架，提升中间推理步骤的可信度，同时优化任务性能。**

- **链接: [https://arxiv.org/pdf/2510.13272](https://arxiv.org/pdf/2510.13272)**

> **作者:** Zhichao Xu; Zongyu Wu; Yun Zhou; Aosong Feng; Kang Zhou; Sangmin Woo; Kiran Ramnath; Yijun Tian; Xuan Qi; Weikang Qiu; Lin Lee Cheong; Haibo Ding
>
> **备注:** TMLR Camera Ready Update
>
> **摘要:** Inspired by the success of reinforcement learning (RL) in Large Language Model (LLM) training for domains like math and code, recent work has begun training LLMs to dynamically plan, query, and reason with search engines as tools -- a paradigm increasingly referred to as agentic search. Although these methods achieve performance improvement across popular short-form QA benchmarks, many prioritize final answer correctness while overlooking the quality of intermediate reasoning steps, which may lead to chain-of-thought unfaithfulness. In this paper, we first introduce a comprehensive evaluation framework for agentic search, covering three distinct faithfulness metrics: Think-Search faithfulness, Information-Think faithfulness, and Think-Answer faithfulness. Our evaluations reveal that canonical agentic search systems trained through Reinforcement Learning from Verifiable Reward (RLVR) using episode-level outcome-based reward -- including Search-R1 and ReSearch -- have significant room for improvement on these faithfulness dimensions. To foster faithful reasoning in agentic search, we introduce VERITAS (Verifying Entailed Reasoning through Intermediate Traceability in Agentic Search), a novel framework that integrates fine-grained turn-level faithfulness rewards into the reinforcement learning process. Our experiments show that models trained with \ours not only significantly improve reasoning faithfulness, but also achieve better task performance compared to baselines trained against episode-level outcome-based reward.
>
---
#### [replaced 023] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私安全任务，旨在解决多智能体大语言模型中的记忆泄露问题。通过构建MAMA框架，分析不同图拓扑对泄露的影响，提出系统设计建议。**

- **链接: [https://arxiv.org/pdf/2512.04668](https://arxiv.org/pdf/2512.04668)**

> **作者:** Jinbo Liu; Defu Cao; Yifei Wei; Tianyao Su; Yuan Liang; Yushun Dong; Yan Liu; Yue Zhao; Xiyang Hu
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics: ACL 2026. Camera-ready version
>
> **摘要:** Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a controlled evaluation framework for comparing topology-conditioned memory leakage in multi-agent LLM systems. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over 10 rounds, we measure leakage using a two-stage recovery criterion that combines exact-match extraction with LLM-based inference over the attacker's final output. We evaluate six canonical topologies (complete, circle, chain, tree, star, star-ring) across $n\in\{4,5,6\}$, attacker-target placements, and base models. Results are consistent: denser connectivity, shorter attacker-target distance, and higher target centrality increase leakage; most leakage occurs in early rounds and then plateaus; model choice shifts absolute rates but preserves broad structural trends; spatiotemporal/location attributes leak more readily than identity credentials or regulated identifiers. We distill practical guidance for system design: favor sparse or hierarchical connectivity, maximize attacker-target separation, and restrict hub/shortcut pathways via topology-aware access control. Our code is available at this https URL.
>
---
#### [replaced 024] DSL-Topic: Improving Topic Modeling by Distilling Soft Labelsfrom Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于主题建模任务，旨在解决传统模型忽视上下文和数据稀疏的问题。通过从语言模型中提炼软标签，提升主题质量和文档相似性检索效果。**

- **链接: [https://arxiv.org/pdf/2602.17907](https://arxiv.org/pdf/2602.17907)**

> **作者:** Raymond Li; Amirhossein Abaskohi; Chuyuan Li; Gabriel Murray; Giuseppe Carenini
>
> **备注:** 22 pages, 5 figures. Camera-ready version for ICML 2026
>
> **摘要:** Traditional neural topic models are typically optimized by reconstructing the document's Bag-of-Words (BoW) representations, overlooking contextual information and struggling with data sparsity. In this work, we introduce a novel topic model training framework by Distilling Soft Labels (DSL) from Language Models (LMs). To construct the contextually enriched reconstruction signals, we project the next token probabilities, conditioned on a specialized prompt, onto a pre-defined vocabulary, and train the topic models to reconstruct the soft labels using the LM hidden states. This produces higher-quality topics that are more closely aligned with the underlying thematic structure of the corpus. Extensive experiments demonstrate that DSL achieves substantial improvements in topic coherence and assignment accuracy over existing baselines. Additionally, we also introduce a retrieval-based metric, which shows that our approach significantly outperforms existing methods in identifying semantically similar documents, highlighting its effectiveness for retrieval-oriented applications.
>
---
#### [replaced 025] Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于数学推理任务，解决GRPO中粗粒度信用分配的问题。提出OAR机制，通过细粒度分配优势，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.07408](https://arxiv.org/pdf/2601.07408)**

> **作者:** Ziheng Li; Liu Kang; Feng Xiao; Luxi Xing; Qingyi Si; Zhuoran Li; Weikang Gong; Deqing Yang; Yanghua Xiao; Hongcheng Guo
>
> **摘要:** Group Relative Policy Optimization (GRPO) has emerged as a promising critic-free reinforcement learning paradigm for reasoning tasks. However, standard GRPO employs a coarse-grained credit assignment mechanism that propagates group-level rewards uniformly to to every token in a sequence, neglecting the varying contribution of individual reasoning steps. We address this limitation by introducing Outcome-grounded Advantage Reshaping (OAR), a fine-grained credit assignment mechanism that redistributes advantages based on how much each token influences the model's final answer. We instantiate OAR via two complementary strategies: (1) OAR-P, which estimates outcome sensitivity through counterfactual token perturbations, serving as a high-fidelity attribution signal; (2) OAR-G, which uses an input-gradient sensitivity proxy to approximate the influence signal with a single backward pass. These importance signals are integrated with a conservative Bi-Level advantage reshaping scheme that suppresses low-impact tokens and boosts pivotal ones while preserving the overall advantage mass. Empirical results on extensive mathematical reasoning benchmarks demonstrate that while OAR-P sets the performance upper bound, OAR-G achieves comparable gains with negligible computational overhead, both significantly outperforming a strong GRPO baseline, pushing the boundaries of critic-free LLM reasoning.
>
---
#### [replaced 026] MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决LLMs隐性领域风险问题。通过构建数据集和提出MENTOR框架，提升模型安全性并降低攻击成功率。**

- **链接: [https://arxiv.org/pdf/2511.07107](https://arxiv.org/pdf/2511.07107)**

> **作者:** Liang Shan; Kaicheng Shen; Wen Wu; Zhenyu Ying; Chaochao Lu; Yan Teng; Jingqi Huang; Qingshan Liu; Guangze Ye; Guoqing Wang; Jie Zhou; Liang He
>
> **摘要:** Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8\%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR performs metacognitive self-assessment, using strategies such as perspective-taking and consequential reasoning to uncover latent model misalignments. The resulting reflections are distilled into dynamic rule-based knowledge graphs, from which retrieved rules are converted into activation-level steering signals to guide internal representations during inference. Experiments demonstrate that MENTOR substantially reduces attack success rates across all tested domains and outperforms existing safety alignment methods. The code and dataset for MENTOR are available at: this https URL.
>
---
#### [replaced 027] Adaptive Information Control for Search-Augmented LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于增强型大语言模型推理任务，解决检索过程中信息冗余和训练不稳定问题。提出DeepControl框架，通过信息效用控制检索的范围与细节，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2602.01672](https://arxiv.org/pdf/2602.01672)**

> **作者:** Siheng Xiong; Oguzhan Gungordu; James C. Kerce; Faramarz Fekri
>
> **摘要:** Search-augmented reasoning agents interleave multi-step reasoning with external retrieval, but uncontrolled retrieval can introduce redundant evidence, saturate the context, and destabilize reinforcement learning (RL). Existing outcome-based RL methods provide only sparse terminal rewards, offering limited guidance for intermediate information-acquisition decisions. We propose DeepControl, an adaptive information-control framework based on information utility, a state-dependent estimate of the marginal value of retrieved evidence. The framework regulates information acquisition along two axes: extent, i.e., whether retrieval should continue, and resolution, i.e., how much retrieved detail should be exposed. It implements these controls through retrieval-continuation guidance, hierarchical granularity control, and an annealed control-forcing scheme. This enables the policy to internalize effective acquisition behavior during training and operate without external control at test time. Across seven benchmarks, DeepControl consistently outperforms strong RL and retrieval baselines without explicit information control; compared with Search-R1, it improves average performance by +9.4 and +8.6 points on Qwen2.5-7B and Qwen2.5-3B, respectively. Additional analyses show improved search effectiveness, training stability, and evidence utilization.
>
---
#### [replaced 028] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，解决模型推理效率与可解释性问题。提出Prompt-Level Distillation方法，通过提取教师模型的推理模式，提升学生模型性能，同时保持透明和低延迟。**

- **链接: [https://arxiv.org/pdf/2602.21103](https://arxiv.org/pdf/2602.21103)**

> **作者:** Sanket Badhe; Deep Shah
>
> **备注:** Accepted at ACL 2026 Industry Track
>
> **摘要:** Advanced reasoning typically requires Chain-of-Thought prompting, which is accurate but incurs prohibitive latency and substantial test-time inference costs. The standard alternative, fine-tuning smaller models, often sacrifices interpretability while introducing significant resource and operational overhead. To address these limitations, we introduce Prompt-Level Distillation (PLD). We extract explicit reasoning patterns from a Teacher model and organize them into a structured list of expressive instructions for the Student model's System Prompt. Evaluated using Gemma-3 4B, PLD improved Macro F1 scores on StereoSet (57\% to 90.0\%) and Contract-NLI (67\% to 83\%), while increasing LogiQA accuracy to 70\%. Similar results on Mistral Small 3.1 demonstrate cross-architecture generalizability, enabling these compact models to match frontier performance with negligible latency overhead. These expressive instructions render the decision-making process transparent, allowing for full human verification of logic, making this approach ideal for regulated industries such as law, finance, and content moderation, as well as high-volume use cases and edge devices.
>
---
#### [replaced 029] Traceable by Design: An LLM Pipeline and Dashboard for EU Regulatory Consultation Analysis
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决监管咨询文本分析难题。通过构建LLM管道和仪表盘，实现主题提取与证据溯源，提升分析效率与透明度。**

- **链接: [https://arxiv.org/pdf/2605.30995](https://arxiv.org/pdf/2605.30995)**

> **作者:** Thales Bertaglia; Haoyang Gui; Catalina Goanta; Gerasimos Spanakis
>
> **备注:** This research has been supported by funding from the ERC Starting Grant HUMANads (ERC-2021-StG No 101041824)
>
> **摘要:** Public consultations generate large volumes of data in the form of stakeholder submissions that are practically unfeasible to analyse manually. We present an end-to-end LLM-based pipeline and interactive dashboard for structured topic extraction from regulatory consultation submissions, demonstrated on the European Commission's Digital Fairness Act (DFA) public call for evidence as a case study. The system processes raw PDF attachments and web-form responses, extracts topic annotations, and grounds every extraction in a verbatim quote from the source text. Applied to 4,322 DFA submissions, the pipeline produced 15,368 topic annotations supported by 20,951 verbatim evidence quotes. Three principles govern the proposed design: verbatim grounding, full traceability, and transparency by design. The dashboard exposes the full extraction dataset through five analytical views, from dataset-level topic overviews to individual paragraph drill-downs, with every result traceable to its source. Beyond the predefined DFA topic categories, the pipeline generated certain stakeholder concerns, such as Age Verification, Payment Processor Censorship, and Digital Ownership, that a fixed-taxonomy approach would have missed. The pipeline is domain-generic; adapting it to a new consultation requires only a prompt update and a new dataset. A live demo is available at this https URL. The code and processed data are publicly available at this https URL.
>
---
#### [replaced 030] DEER: Disentangled Mixture of Experts with Instance-Adaptive Routing for Generalizable Machine-Generated Text Detection
- **分类: cs.CL**

- **简介: 该论文属于机器生成文本检测任务，旨在解决现有检测器在领域迁移时性能下降的问题。提出DEER框架，通过解耦领域特定与通用知识，并利用强化学习路由提升检测效果。**

- **链接: [https://arxiv.org/pdf/2511.01192](https://arxiv.org/pdf/2511.01192)**

> **作者:** Guoxin Ma; Xiaoming Liu; Hongyang Chen; Chengzhengxu Li; Zhaohan Zhang; Shengchao Liu; Yu Lan; Cong Wang; Chao Shen
>
> **备注:** ARR Under Review
>
> **摘要:** Detecting machine-generated text has become a critical challenge amid the rapid advancement of LLMs, yet existing detectors degrade severely under domain shift. Through systematic pilot studies, we trace this vulnerability to two fundamental flaws in current generalization strategies, namely the incomplete preservation of domain-specific knowledge during multi-domain training and the misalignment between knowledge retrieval and the detection objective at inference. To address these gaps, we propose DEER, a Disentangled mixturE-of-ExpeRts framework that explicitly decouples domain-local and domain-invariant knowledge into specialized expert modules. Instead of static domain matching, DEER employs a reinforcement learning-driven router that selects expert pathways based on instance-level detection rewards. This task-aligned, domain-agnostic mechanism ensures robust adaptation to unseen distributions by prioritizing detection utility over stylistic resemblance. Extensive experiments demonstrate that DEER consistently outperforms state-of-the-art detectors, achieving average F1 improvements of 1.28% and 2.92%, and accuracy gains of 1.35% and 2.26% on in-domain and out-of-domain datasets, offering reliable generalization for open-world deployment.
>
---
#### [replaced 031] GAPD: Gold-Action Policy Distillation for Agentic Reinforcement Learning in Knowledge Base Question Answering
- **分类: cs.CL**

- **简介: 该论文针对知识库问答中的强化学习任务，解决中间动作错误监督不足的问题。提出GAPD框架，通过黄金动作引导学生策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.29584](https://arxiv.org/pdf/2605.29584)**

> **作者:** Xin Sun; Jianan Xie; Zhongqi Chen; Qiang Liu; Shu Wu; Bowen Song; Weiqiang Wang; Zilei Wang; Liang Wang
>
> **摘要:** Reinforcement learning (RL) is a natural fit for agentic knowledge base question answering (KBQA), where a model must issue executable actions, observe knowledge-base feedback, and eventually return an answer. However, current RL-based KBQA systems mainly optimize sparse rewards from the final answer, leaving intermediate action errors weakly supervised. This is especially limiting for logical-form annotated KBQA benchmarks: gold logical forms can be converted into executable action sequences, but existing pipelines use them mainly for warm-start data construction rather than for on-policy RL updates. We propose GAPD, a training-time Gold-Action Policy Distillation framework that adds dense token-level guidance to outcome-based RL. To align gold actions with on-policy student rollouts, GAPD uses MID-ANCHOR MATCHING: it treats the intermediate entities reached during student exploration and gold execution as state anchors, and matches student states to gold states through these explored entity sets. The current policy conditioned on this aligned gold action serves as a stop-gradient teacher, whose token distribution is distilled back to the ordinary student policy over generated action-token spans. GAPD consistently surpasses the current state of the art on WebQSP, GrailQA, and GraphQ.
>
---
#### [replaced 032] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中预层归一化（Pre-LN）的稳定性与效率问题。提出BHyT方法，在保持性能的同时提升训练速度和吞吐量。**

- **链接: [https://arxiv.org/pdf/2601.09719](https://arxiv.org/pdf/2601.09719)**

> **作者:** Hoyoon Byun; Youngjun Choi; Taero Kim; Sungrae Park; Kyungwoo Song
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Pre-Layer Normalization (Pre-LN) is the de facto choice for large language models (LLMs) and is crucial for stable pretraining and effective transfer learning. However, Pre-LN incurs repeated statistical-computation overhead and remains vulnerable to the curse of depth, where hidden-state magnitudes and variances grow as the number of layers increases, destabilizing training. Efficiency-oriented normalization-free methods such as Dynamic Tanh (DyT) improve throughput but remain fragile at depth. To jointly address stability and efficiency, we propose Bounded Hyperbolic Tanh (BHyT), a drop-in replacement for Pre-LN. BHyT combines a tanh nonlinearity with explicit, data-driven input bounding to keep activations within a non-saturating range. It prevents depth-wise growth in activation magnitude and variance and provides a theoretical stability guarantee. For efficiency, BHyT computes exact statistics once per block and replaces a second normalization with a lightweight variance approximation. Empirically, BHyT demonstrates improved stability and efficiency during pretraining, achieving an average of 1.6\% faster training and an average of 1.77\% higher token generation throughput compared to RMSNorm, while maintaining strong pretraining-only and post-SFT performance across language understanding and reasoning benchmarks\footnote{Code is available at: this https URL}.
>
---
#### [replaced 033] From Out-of-Distribution Detection to Hallucination Detection: A Geometric View
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型安全任务，旨在解决 hallucination 检测问题。通过将 hallucination 检测重新定义为分布外检测，提出一种无需训练的单样本检测方法。**

- **链接: [https://arxiv.org/pdf/2602.07253](https://arxiv.org/pdf/2602.07253)**

> **作者:** Litian Liu; Reza Pourreza; Yubing Jian; Yao Qin; Roland Memisevic
>
> **备注:** ICML 2026 main conference paper
>
> **摘要:** Detecting hallucinations in large language models is a critical open problem with significant implications for safety and reliability. While existing hallucination detection methods achieve strong performance in question-answering tasks, they remain less effective on tasks requiring reasoning. In this work, we revisit hallucination detection through the lens of out-of-distribution (OOD) detection, a well-studied problem in areas like computer vision. Treating next-token prediction in language models as a classification task allows us to apply OOD techniques, provided appropriate modifications are made to account for the structural differences in large language models. We show that OOD-based approaches yield training-free, single-sample-based detectors, achieving strong accuracy in hallucination detection for reasoning tasks. Overall, our work suggests that reframing hallucination detection as OOD detection provides a promising and scalable pathway toward language model safety.
>
---
#### [replaced 034] Can Large Language Models Generalize Procedures Across Representations?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLMs在不同表示间（代码、图、自然语言）的泛化能力，解决跨表示任务的性能问题。通过两阶段强化学习方法提升模型表现。**

- **链接: [https://arxiv.org/pdf/2602.03542](https://arxiv.org/pdf/2602.03542)**

> **作者:** Fangru Lin; Valentin Hofmann; Xingchen Wan; Weixing Wang; Zifeng Ding; Anthony G. Cohn; Janet B. Pierrehumbert
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Large language models (LLMs) are trained and tested extensively on symbolic representations such as code and graphs, yet real-world user tasks are often specified in natural language. To what extent can LLMs generalize across these representations? Here, we approach this question by studying isomorphic tasks involving procedures represented in code, graphs, and natural language (e.g., scheduling steps in planning). We find that training LLMs with popular post-training methods on graphs or code data alone does not reliably generalize to corresponding natural language tasks, while training solely on natural language can lead to inefficient performance gains. To address this gap, we propose a two-stage reinforcement learning curriculum that first trains on symbolic, then natural language data. The curriculum substantially improves model performance across model families and tasks. Remarkably, a 1.5B Qwen model trained by our method can closely match zero-shot GPT-4o in naturalistic planning. Finally, our analysis suggests that successful cross-representation generalization can be interpreted as a form of generative analogy, which our curriculum effectively encourages. The dataset and code used in this paper can be found \href{this https URL}{here}.
>
---
#### [replaced 035] Solving Zebra Puzzles Using Constraint-Guided Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于逻辑谜题求解任务，旨在解决Zebra puzzles这类复杂逻辑问题。通过引入多智能体系统结合LLM与定理证明器，分解问题并生成SMT代码求解，提升解题准确率。**

- **链接: [https://arxiv.org/pdf/2407.03956](https://arxiv.org/pdf/2407.03956)**

> **作者:** Shmuel Berman; Kathleen McKeown; Baishakhi Ray
>
> **摘要:** Prior research has enhanced the ability of Large Language Models (LLMs) to solve logic puzzles using techniques such as chain-of-thought prompting or introducing a symbolic representation. These frameworks are still usually insufficient to solve complicated logical problems, such as Zebra puzzles, due to the inherent complexity of translating natural language clues into logical statements. We introduce a multi-agent system, ZPS, that integrates LLMs with an off the shelf theorem prover. This system tackles the complex puzzle-solving task by breaking down the problem into smaller, manageable parts, generating SMT (Satisfiability Modulo Theories) code to solve them with a theorem prover, and using feedback between the agents to repeatedly improve their answers. We also introduce an automated grid puzzle grader to assess the correctness of our puzzle solutions and show that the automated grader is reliable by evaluating it in a user-study. Our approach shows improvement in all three LLMs we tested, with GPT-4 showing 166% improvement in the number of fully correct solutions.
>
---
#### [replaced 036] Geometry-Aware Hallucination Detection in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 LLM 生成内容不准确的问题。提出 GA-ICL 方法，通过几何感知的演示选择提升检测效果。**

- **链接: [https://arxiv.org/pdf/2601.06196](https://arxiv.org/pdf/2601.06196)**

> **作者:** Bodla Krishna Vamshi; Rohan Bhatnagar; Haizhao Yang
>
> **摘要:** Large language models (LLMs) frequently generate factually incorrect or unsupported content, commonly referred to as hallucinations. Prior work has explored decoding strategies, retrieval augmentation, and supervised fine-tuning for hallucination detection, while recent studies show that in-context learning (ICL) can substantially influence factual reliability. However, existing ICL demonstration selection methods often rely on surface-level similarity heuristics and exhibit limited robustness across tasks and models. We propose GA-ICL, a geometry-aware demonstration sampling framework for selecting in-context demonstrations that leverages latent representations extracted from frozen LLMs. By jointly modeling local manifold structure and class-aware prototype geometry, GA-ICL selects demonstrations based on their proximity to learned prototypes rather than lexical or embedding similarity alone. Across factual verification (FEVER) and hallucination detection (HaluEval) benchmarks, GA-ICL outperforms standard ICL selection baselines in the majority of evaluated settings, with particularly strong gains on dialogue and summarization tasks. The method remains robust under temperature perturbations and model variation, indicating improved stability compared to heuristic retrieval strategies. While lexical retrieval can remain competitive in certain question-answering regimes at smaller model scales, our results demonstrate that geometry-aware prototype selection provides a reliable and training-light approach for hallucination detection without modifying LLM parameters. Extended evaluations on Phi-14B and Qwen3-32B confirm that GA-ICL scales effectively to larger models, outperforming all compared baselines including on QA tasks where smaller models show boundary-condition limitations, offering a principled direction for improved ICL demonstration selection.
>
---
#### [replaced 037] Not What, But How: A Framework for Auditing LLM Responses across Positioning, Generalization, Anthromorphism, and Maxims
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决LLM响应框架的审计问题。提出FRANZ框架，从四个维度评估响应，揭示不同模型在表达方式上的差异。**

- **链接: [https://arxiv.org/pdf/2606.02493](https://arxiv.org/pdf/2606.02493)**

> **作者:** Siddhesh Milind Pawar; Sarah Masud; Haneul Yoo; Alice Oh; Isabelle Augenstein
>
> **备注:** 34 pages, 19 Figures, 4 Tables
>
> **摘要:** Large language models (LLMs) are being increasingly used to answer subjective, information-seeking questions, where users are sensitive to how responses are communicated, not just whether the answers are correct. Existing LLM evaluations for subjective cultural queries largely focus on factual correctness, ignoring how the response is framed. To this end, we introduce FRANZ, an automated FRAmework for respoNse characteriZation to conduct communicative audit of LLM responses along four dimensions: cultural positioning, use of generalizing language, anthropomorphic cues, and adherence to conversational maxims. To enable this evaluation, we contribute SQUARE - a corpus of 376k subjective questions sourced from 57 subreddits, and mapped to 7 countries and 19 question categories. We demonstrate FRANZ's applicability by scoring responses from three open-weight LLMs. We observe that LLMs show statistically significant differences in the frequency with which they employ each response characteristic. Unlike single-dimensional audits, FRANZ reveals that insider positioning and anthropomorphism are positively coupled, with the degree of coupling varying by country, providing a diagnostic lens for identifying framing divergences.
>
---
#### [replaced 038] Reasoning over Boundaries: Enhancing Specification Alignment via Test-time Deliberation
- **分类: cs.CL**

- **简介: 该论文属于规范对齐任务，旨在解决LLM在不同场景下遵循动态规范的问题。提出Align3方法，通过测试时反思提升规范对齐效果。**

- **链接: [https://arxiv.org/pdf/2509.14760](https://arxiv.org/pdf/2509.14760)**

> **作者:** Haoran Zhang; Yafu Li; Xuyang Hu; Dongrui Liu; Zhilin Wang; Bo Li; Yu Cheng
>
> **备注:** 10 pages main text, 52 pages total (including appendix). Code and resources are available at this https URL
>
> **摘要:** Large language models (LLMs) are increasingly applied in diverse real-world scenarios, each governed by bespoke behavioral and safety specifications (spec) custom-tailored by users or organizations. These spec, categorized into safety-spec and behavioral-spec, vary across scenarios and evolve with changing preferences and requirements. We formalize this challenge as specification alignment, focusing on LLMs' ability to follow dynamic, scenario-specific spec from both behavioral and safety perspectives. To address this challenge, we propose Align3, a lightweight method that employs Test-Time Deliberation (TTD) with hierarchical reflection and revision to reason over the specification boundaries. We further present SpecBench, a unified benchmark for measuring specification alignment, covering 5 scenarios, 103 spec, and 1,500 prompts. Experiments on 15 reasoning and 18 instruct models with several TTD methods, including Self-Refine, TPO, and MoreThink, yield three key findings: (i) test-time deliberation enhances specification alignment; (ii) Align3 advances the safety-helpfulness trade-off frontier with minimal overhead; (iii) SpecBench effectively reveals alignment gaps. These results highlight the potential of test-time deliberation as an effective strategy for reasoning over the real-world specification boundaries. Our code and resources are available at this https URL.
>
---
#### [replaced 039] Large AI Models in Dental Healthcare: From General-Purpose Systems to Domain-Specific Foundation Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗AI领域，旨在探讨大模型在牙科中的应用。分析了不同模型的性能与局限，提出分类框架并建议整合方案。**

- **链接: [https://arxiv.org/pdf/2606.02914](https://arxiv.org/pdf/2606.02914)**

> **作者:** Sema Helali; Lina Abu Nada; Sausan Al Kawas; Alaa Abd-Alrazaq; Faleh Tamimi; Rafat Damseh
>
> **摘要:** Background: Oral diseases affect nearly 3.5 billion people worldwide, yet the comparative clinical potential of large-scale AI models in dentistry remains poorly understood. Three distinct model categories have emerged: language-generative models, discriminative vision foundation models, and dental-specific foundation models, with no unified review examining their relationships and collective limitations. Methods: Following PRISMA-ScR guidelines, we systematically searched four databases (PubMed, Google Scholar, Scopus, arXiv), screened independently by two reviewers. After applying inclusion/exclusion criteria, 97 studies (2020-2026) were included. We propose a two-dimensional classification framework organizing models by architectural paradigm and dental specialization degree. Results: Language-generative models excel at text-based tasks (clinical reasoning, licensing exams, patient communication) but show inconsistent performance on image-dependent diagnostics. Adapted SAM and CLIP variants achieve strong tooth segmentation and lesion detection results. Dental-specific models (DentVFM, DentVLM, OralGPT) demonstrate strongest performance on complex multimodal tasks. Integrated pipelines consistently outperform single-model approaches. A data asymmetry is observed: dental-specific pretraining concentrates almost entirely in the vision domain, reflecting scarce large-scale dental text corpora. Conclusions: General-purpose and dental-specific models play complementary roles; the most effective systems combine both within structured pipelines. Safe autonomous deployment requires resolving three persistent barriers: hallucination in generative models, limited annotated dental datasets, and absent standardized clinical evaluation benchmarks.
>
---
#### [replaced 040] FedMental: Evaluating Federated Learning for Mental Health Detection from Social Media Data
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于心理健康检测任务，旨在解决隐私保护与数据共享的矛盾。通过联邦学习和差分隐私技术，评估其在社交媒体数据上的性能与隐私平衡。**

- **链接: [https://arxiv.org/pdf/2605.18936](https://arxiv.org/pdf/2605.18936)**

> **作者:** Nuredin Ali Abdelkadir; Anjali Ratnam; Zeerak Talat; Stevie Chancellor
>
> **备注:** Association for Computational Linguistics (ACL) 2026 Main Conference
>
> **摘要:** Social media text data are often used to train Machine Learning (ML) models to identify users exhibiting high-risk mental health behaviors. However, sharing this sensitive data poses privacy risks and limits the growth of benchmark datasets. We comprehensively evaluate whether privacy-preserving ML techniques can enable safer data sharing while preserving performance. Specifically, we apply federated learning (FL) and Differentially Private FL for two widely-studied mental health prediction tasks: depression detection on X (Twitter) and suicide crisis detection on Reddit. We simulate realistic data-sharing scenarios by treating each user as a client in a non-IID setting, evaluating across different client fractions, aggregation strategies, and privacy budgets. While FL achieves comparable performance to centralized training (centralized F1 = 85.63; best FL model F1 = 83.16) on depression identification, we find that Differentially Private FL has a large performance-privacy trade-off (up to F1 = 27.01 drop) even with low levels of noise (epsilon = 50). This is due to the distortion of highly informative yet sparse mental health linguistic markers related to mental health, like health topics and emotion words. This research empirically demonstrates the potential and limitations of current privacy preservation techniques for mental health inference tasks.
>
---
#### [replaced 041] DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CR; cs.IR**

- **简介: 该论文属于安全与隐私任务，针对RAG系统提出一种新的观点操纵攻击方法DiscourseFlip，解决多主题查询空间中的意见偏移问题。通过图引导攻击，有效提升攻击覆盖与隐蔽性。**

- **链接: [https://arxiv.org/pdf/2606.01212](https://arxiv.org/pdf/2606.01212)**

> **作者:** Yuyang Gong; Miaokun Chen; Jiawei Liu; Zhuo Chen; Guoxiu He; Wei Lu; XiaoFeng Wang; Xiaozhong Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are widely deployed and increasingly influential, but their reliance on external corpora exposes new security risks from poisoned retrieval content. Existing RAG attacks are largely focusing on individual queries or narrow topic-local query sets, which limits their practical reach and offers limited camouflage in real-world settings. In this paper, we introduce discourse-level opinion manipulation, a new threat model in which coordinated influence across a semantic query network induces opinion shifts over a holistic, multi-topic query space. We formalize this threat in a black-box setting and propose DiscourseFlip, an agentic, graph-guided attack that dynamically allocates a limited poisoning budget to maximize discourse-level opinion deviation. Extensive experiments demonstrate that DiscourseFlip consistently induces targeted opinion shifts across the contextualized query network and significantly outperforms existing baselines in terms of coverage and effectiveness. User studies further confirm that DiscourseFlip is effective while remaining well camouflaged from user detection. Moreover, systematic analyses show that existing mitigation strategies are ineffective against discourse-level manipulation, underscoring the urgent need for more robust and adaptive defenses to address discourse-level vulnerabilities.
>
---
#### [replaced 042] High-Quality Entity Segmentation and Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ESG框架，解决实体分割与定位问题。构建了EntitySeg数据集，采用两阶段设计，提升分割与语言语义匹配效果。**

- **链接: [https://arxiv.org/pdf/2402.02555](https://arxiv.org/pdf/2402.02555)**

> **作者:** Lu Qi; Yi-Wen Chen; Tao Zhang; Xiangtai Li; Xu Yang; Bo Du; Ming-Hsuan Yang
>
> **摘要:** In this work, we propose ESG, a pipeline for high-quality entity segmentation and grounding supported by a new dataset EntitySeg. At first, the proposed dataset naming EntitySeg contains images spanning various image domains and entities, along with plentiful high-resolution images and high-quality mask annotations for training and testing. Then, the ESG mainly consists of two modules: CropFormer for high-quality entity segmentation whereas GELLA for accurate noun extraction from sentences and semantic matching between language and visual regions. Unlike existing grounding methods that jointly train a segmentation and a large language model, ESG adopts a two-stage decoupled design, preserving high-quality masks and grounding robustness without the trade-offs often introduced by joint training. CropFormer ensures high-quality entity segmentation results, which can then be encoded into the GELLA model for effective grounding. Extensive experimental results demonstrate the effectiveness of our proposed pipeline across five tasks, including entity segmentation, panoptic segmentation, open-vocabulary segmentation, referring segmentation, and panoptic localized narratives. Furthermore, GELLA module of ESG pipeline is highly flexible and capable of processing mask inputs from any segmentation framework, thanks to its lightweight colormap/vision encoder, language/mask decoder, and association module. The entity segmentation dataset and grounding code will be released at this https URL.
>
---
#### [replaced 043] SSSD: Simply-Scalable Speculative Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SSSD，一种无需训练的推测解码方法，用于加速大语言模型推理。解决现有方法依赖额外模型、复杂度高的问题，通过轻量n-gram匹配与硬件感知推测，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2411.05894](https://arxiv.org/pdf/2411.05894)**

> **作者:** Michele Marzollo; Jiawei Zhuang; Niklas Roemer; Niklas Zwingenberger; Lorenz K. Müller; Lukas Cavigelli
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026, Main Conference)
>
> **摘要:** Speculative Decoding has emerged as a popular technique for accelerating inference in Large Language Models. However, most existing approaches yield only modest improvements in production serving systems. Methods that achieve substantial speedups typically rely on an additional trained draft model or auxiliary model components, increasing deployment and maintenance complexity. This added complexity reduces flexibility, particularly when serving workloads shift to tasks, domains, or languages that are not well represented in the draft model's training data. We introduce Simply-Scalable Speculative Decoding (SSSD), a training-free method that combines lightweight n-gram matching with hardware-aware speculation. Relative to standard autoregressive decoding, SSSD reduces latency by up to 2.9x. It achieves performance on par with leading training-based approaches across a broad range of benchmarks, while requiring substantially lower adoption effort--no data preparation, training or tuning are needed--and exhibiting superior robustness under language and domain shift, as well as in long-context settings.
>
---
#### [replaced 044] From Graph Retrieval to Schema Realization: Counterfactual Validation for Text-to-SPARQL over Heterogeneous Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，解决异构知识图谱中文本到SPARQL查询生成的问题。提出SchemaForge框架，通过schema对齐和反事实验证提升查询准确性。**

- **链接: [https://arxiv.org/pdf/2508.01815](https://arxiv.org/pdf/2508.01815)**

> **作者:** Chengxiao Dai; Yue Xiu; Dusit Niyato
>
> **摘要:** Text-to-SPARQL maps natural-language questions to executable SPARQL queries over RDF knowledge graphs. While standard evaluations often fix the target graph in advance, practical knowledge graph question answering (KGQA) may involve heterogeneous graph collections with different schemas, partial alignments, and incomplete metadata. In this setting, query generation depends on more than SPARQL syntax: the system must identify a graph schema that can support the predicates, entity types, joins, filters, and constraints required by the question. We present SchemaForge, a schema-grounded agentic framework for text-to-SPARQL over heterogeneous KG collections. Its central mechanism is question-conditioned schema-slice alignment: weak graph evidence first identifies plausible graphs, while stronger schema evidence determines whether a local schema slice can realize the intended query. The selected schema slice then constrains query generation and verification before execution. When only one graph is available, the same formulation reduces to standard single-KG text-to-SPARQL with schema grounding. We evaluate SchemaForge on LC-QuAD 2.0, QALD-9 Plus, QALD-10, and Spider4SPARQL. Across the four public benchmarks, SchemaForge improves execution accuracy over the strongest matched agent baseline by 11.50 percentage points on average. On Spider4SPARQL, SchemaForge improves execution accuracy from 54.86% to 64.18% and achieves 73.0% Top-1 and 97.0% Top-3 graph allocation accuracy. These results show that moving from weak graph evidence to schema-specific query commitments, together with counterfactual answer-set checks, improves executable query generation over heterogeneous knowledge graphs.
>
---
#### [replaced 045] GIFT: Games as Informal Training for Generalizable LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在提升大语言模型的泛化能力。通过游戏进行非正式训练，解决模型在规划、创造力等能力上的不足，并提出CST方法优化多任务学习。**

- **链接: [https://arxiv.org/pdf/2601.05633](https://arxiv.org/pdf/2601.05633)**

> **作者:** Nuoyan Lyu; Bingbing Xu; Xueyun Tian; Weihao Meng; Yige Yuan; Yang Zhang; Zhiyong Huang; Tat-Seng Chua; Huawei Shen
>
> **摘要:** Recent LLMs excel at formal tasks such as mathematical reasoning and code generation, but still struggle with broader abilities such as planning, creativity, and social intelligence. Inspired by human learning, where formal instruction and informal experience jointly shape intelligence, we introduce informal learning into LLM training and use games as annotation-free, feedback-driven environments. To cover diverse abilities including abstract reasoning, planning, creativity, and social interaction, we combine formal math tasks with three representative game tasks, including Matrix Games, TicTacToe, and Who's the Spy. However, directly mixing these tasks under a unified RL objective can blur task-specific learning signals and provides no explicit guidance for coordinating task-gradient directions. To combat these, we propose Coordinated Subtask Training (CST), which replaces a single mixed update with sequential subtask-specific updates, separating heterogeneous RL signals while implicitly promoting coordination among subtasks. Experiments on ability-oriented benchmarks show that game-based informal learning improves generalization beyond formal training alone, while CST further enhances multi-task RL by preserving in-domain subtask performance and improving broader general abilities. Code and data are publicly available.
>
---
#### [replaced 046] Hint Tuning: Less Data Makes Better Reasoners
- **分类: cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决推理模型冗余生成问题。通过Hint Tuning方法，使模型根据难度调整推理深度，减少token使用量。**

- **链接: [https://arxiv.org/pdf/2605.08665](https://arxiv.org/pdf/2605.08665)**

> **作者:** Siqi Fan; Minghao Li; Xiaoqian Ma; Xiusheng Huang; Zhuo Chen; Bowen Qin; Liujie Zhang; Shuo Shang; Weihang Chen
>
> **摘要:** Large reasoning models achieve high accuracy through extended chain-of-thought but generate 5--8 more tokens than necessary, applying verbose reasoning uniformly regardless of problem difficulty. We propose Hint Tuning, a data-efficient approach that teaches models to calibrate reasoning depth. Our key insight: the corresponding instruct model serves as an ideal difficulty probe. By testing what the instruct model can solve with varying guidance, we automatically construct training data across three states: No-Hint (direct answer), Sparse-Hint (minimal prefix), and Full-Hint (complete reasoning). This converts the abstract challenge of difficulty labeling into a measurable consistency check between the instruct and reasoning models. With only 1K self-annotated samples, Hint Tuning achieves 24--66% token reduction (31.5% average) across mainstream reasoning models (Qwen3-Thinking, DeepSeek-R1-Distill) at multiple scales (4B--32B) while maintaining competitive accuracy on five benchmarks. Unlike methods requiring massive distillation datasets or expensive RL, we achieve superior efficiency through simple alignment with the instruct model's capabilities. Code and data are available at this https URL.
>
---
#### [replaced 047] Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpretability over Complex Text
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出eXTC，解决复杂文本分类中的可解释性与性能平衡问题。通过结构化提示优化、推理蒸馏和强化学习，实现高效且可解释的分类模型。**

- **链接: [https://arxiv.org/pdf/2605.29076](https://arxiv.org/pdf/2605.29076)**

> **作者:** Tianyang Zhou; Wenbo Chen; Pierre Jinghong Liang; Leman Akoglu
>
> **摘要:** LLMs have advanced text classification, yet existing paradigms face a trade-off: supervised (label only) fine-tuning is scalable but offers limited reasoning on complex text and lacks broader model transparency, while discrete prompt optimization offers human-readable instructions but struggles with performance and scalability. We introduce eXTC (eXplainable Text Classifier) with three progressive stages: (1) learning a Standard Operating Procedure (SOP, or rulebook) in natural language via a new Structured Prompt Optimization algorithm; (2) SOP-grounded reasoning distillation from a large teacher LLM into a compact LM; and (3) expanding reasoning capabilities beyond the initial SOP via reinforcement learning. This design enables eXTC to provide (i) fast inference via a compact LM, with (ii) inference-time local reasoning traces, alongside a global, modular explanation of its learned domain rules, while (iii) significantly outperforming existing paradigms across diverse benchmarks in both classification performance and explanation quality, with stage-by-stage gains.
>
---
#### [replaced 048] Towards Verifiable Multimodal Deep Research: A Multi-Agent Harness for Interleaved Report Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态深度研究任务，旨在解决生成可验证的多模态报告问题。提出Ptah系统，通过多智能体协作实现文本与视觉证据的融合生成与验证。**

- **链接: [https://arxiv.org/pdf/2605.29861](https://arxiv.org/pdf/2605.29861)**

> **作者:** Chenghao Zhang; Guanting Dong; Yufan Liu; Tong Zhao; Xiaoxi Li; Zhicheng Dou
>
> **备注:** In progress
>
> **摘要:** Large Language Models (LLMs) have advanced autonomous agents from deep search, which retrieves concise factual answers, to deep research, which synthesizes scattered evidence into long-form reports. However, verifiable multimodal deep research remains challenging due to open-ended synthesis without deterministic ground truth and the need to interleave textual arguments with visual evidence. We propose Ptah, a multi-agent harness for interleaved report generation. Ptah orchestrates the lifecycle from user query to rendered web report through planning, research, and writing stages, where specialized agents construct visual-aware plans, collect claim-grounded evidence, maintain source-aligned images in a Visual Working Memory, and compose reports through declarative multimodal tool use. A verifier agent serves as the harness's acceptance function, enforcing factual grounding, citation fidelity, and cross-modal consistency throughout the workflow. We further introduce PtahEval, an evaluation protocol that augments existing benchmarks with image-level and presentation-level assessments. Experiments on deep research benchmarks show that Ptah produces more reliable, visually informative, and usable human-facing multimodal reports than strong baselines. Our code is released at this https URL
>
---
#### [replaced 049] MemoNoveltyAgent: A Historical Research Memory-Aware Agent Workflow for Paper Novelty Assessment
- **分类: cs.CL**

- **简介: 该论文属于论文新颖性评估任务，旨在解决现有AI在学术文献分析中质量不足的问题。提出MemoNoveltyAgent系统，结合记忆与检索机制，提升新颖性报告的准确性和全面性。**

- **链接: [https://arxiv.org/pdf/2603.20884](https://arxiv.org/pdf/2603.20884)**

> **作者:** Jiajun Hou; Hexuan Deng; Wenxiang Jiao; Xuebo Liu; Xiaopeng Ke; Derek F. Wong; Min Zhang
>
> **摘要:** To alleviate the heavy burden of paper screening, researchers increasingly rely on existing AI agents, such as AI reviewers or DeepResearch, for paper evaluation and novelty assessment. However, lacking specialized mechanisms for processing scholarly literature, their analyses often produce superficial results with noticeable deficiencies in quality. To bridge this gap, we introduce MemoNoveltyAgent, a multi-agent system designed to generate comprehensive and faithful novelty reports. Beyond retrieving concrete prior-paper evidence via RAG, our system incorporates a high-level abstract memory constructed from large-scale scholarly corpora. This memory organizes research into hierarchical trees to distill field-specific evolutionary trajectories, thereby providing a broader historical context. Furthermore, we decompose papers into discrete novelty points for fine-grained analysis and retrieval, while employing a self-validation mechanism to improve report faithfulness. Finally, to address the evaluation challenges of such open-ended generation tasks, we propose a RAG-augmented checklist evaluation method that enables reliable and evidence-grounded assessments. Extensive experiments demonstrate that MemoNoveltyAgent outperforms GPT-5 DeepResearch by 13.69%. Code and demo are available at this https URL
>
---
#### [replaced 050] LiSeCo: Linear Semantic Control for Language Generation
- **分类: cs.CL; eess.SY**

- **简介: 该论文提出LiSeCo方法，用于语言生成中的语义控制，解决生成内容偏离预期语义的问题。通过在线干预嵌入空间激活，实现对生成文本的精细控制。**

- **链接: [https://arxiv.org/pdf/2405.15454](https://arxiv.org/pdf/2405.15454)**

> **作者:** Emily Cheng; Carmen Amo Alonso
>
> **备注:** TMLR 2026 camera ready; earlier version in NeurIPS MINT Workshop 2024
>
> **摘要:** The prevalence of Large Language Models (LLMs) in critical applications highlights the need for controlled language generation methods that are both computationally efficient and enjoy performance guarantees. To address this need, we use a common model of concept semantics as linearly represented in an LLM's latent space. In particular, we take the view that natural language generation traces a trajectory in this continuous semantic space, realized by the language model's hidden activations. This view permits a control-theoretic treatment of text generation in latent space, in which we propose Linear Semantic Control (LiSeCo), a lightweight, gradient-free intervention that dynamically steers trajectories away from regions corresponding to undesired meanings. In particular, we propose to directly intervene, in an online fashion, the activations of the token that is being generated in embedding space. Crucially, LiSeCo does not simply steer activations towards a desirable region. Instead, it relies on classical techniques from control theory to precisely control activations in a context-dependent way, and guarantees that they are brought into a specific pre-defined region of embedding space that corresponds to allowed semantics. The intervention is computed in closed form according to an optimal controller formulation, minimally impacting generation time. This control of the activations in embedding space allows for fine-grained steering of attributes of the generated sequence. We demonstrate that our approach is effective on different tasks -- toxicity, sentiment, and language (English/Spanish) steering -- while maintaining text quality.
>
---
#### [replaced 051] Luminol-AIDetect: Fast Zero-shot Machine-Generated Text Detection based on Perplexity under Text Shuffling
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于机器生成文本检测任务，旨在解决跨模型、跨语言的零样本检测问题。通过文本打乱引发困惑度变化，区分机器与人类文本，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2604.25860](https://arxiv.org/pdf/2604.25860)**

> **作者:** Lucio La Cava; Andrea Tagarelli
>
> **备注:** Under Review
>
> **摘要:** Machine-generated text (MGT) detection requires identifying structurally invariant signals across generation models, rather than relying on model-specific fingerprints. In this respect, we hypothesize that while large language models excel at local semantic consistency, their autoregressive nature results in a specific kind of structural fragility compared to human writing. We propose Luminol-AIDetect, a novel, zero-shot statistical approach that exposes this fragility through coherence disruption. By applying a simple randomized text-shuffling procedure, we demonstrate that the resulting shift in perplexity serves as a principled, model-agnostic discriminant, as MGT displays a characteristic dispersion in perplexity-under-shuffling that differs markedly from the more stable structural variability of human-written text. Luminol-AIDetect leverages this distinction to inform its decision process, where a handful of perplexity-based scalar features are extracted from an input text and its shuffled version, then detection is performed via density estimation and ensemble-based prediction. Evaluated across 8 content domains, 11 adversarial attack types, and 18 languages, Luminol-AIDetect demonstrates state-of-the-art performance, with gains up to 17x lower FPR while being cheaper than prior methods.
>
---
#### [replaced 052] Confidence Before Answering: A Paradigm Shift for Efficient LLM Uncertainty Estimation
- **分类: cs.CL**

- **简介: 该论文属于语言模型不确定性估计任务，旨在解决传统方法在回答后才评估置信度的问题。提出CoCA框架，实现置信度与答案的联合优化，提升模型校准和不确定性区分能力。**

- **链接: [https://arxiv.org/pdf/2603.05881](https://arxiv.org/pdf/2603.05881)**

> **作者:** Changcheng Li; Jiancan Wu; Hengheng Zhang; Zhengsu Chen; Guo An; Junxiang Qiu; Xiang Wang; Qi Tian
>
> **摘要:** Reliable deployment of large language models (LLMs) requires accurate uncertainty estimation. Existing methods are predominantly answer-first, producing confidence only after generating an answer, which measure the correctness of a specific response and limits practical usability. We study a confidence-first paradigm, where the model outputs its confidence before answering, interpreting this score as the model's probability of answering the question correctly under its current policy. We propose CoCA(Co-optimized Confidence and Answers), a GRPO reinforcement learning framework that jointly optimizes confidence calibration and answer accuracy via segmented credit assignment. By assigning separate rewards and group-relative advantages to confidence and answer segments, CoCA enables stable joint optimization and avoids reward hacking. Experiments across math, code, and factual QA benchmarks show improved calibration and uncertainty discrimination while preserving answer quality, thereby enabling a broader range of downstream applications.
>
---
#### [replaced 053] Consistency Training Can Entrench Misalignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型对齐研究，探讨一致性训练对模型行为的影响。工作包括实验分析与理论框架构建，揭示其可能加剧或缓解偏差行为。**

- **链接: [https://arxiv.org/pdf/2606.03810](https://arxiv.org/pdf/2606.03810)**

> **作者:** David Demitri Africa; Arathi Mani
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Consistency training encourages a model to produce similar outputs across related inputs or sampling procedures. Such methods are simple, scalable, and largely label-free, but their effects on model alignment remain poorly understood. Could the self-bootstrapping nature of these methods amplify undesired behavior in models? We test seven consistency training methods on 108 model organisms: open-source models (7B--70B) fine-tuned to exhibit various forms of controlled misaligned behavior. We find that outcomes vary significantly: consistency training generally suppresses reward hacking and emergent misalignment but amplifies sycophancy. We present evidence that distribution shifts induced by the consistency labeling process, rather than variation in the selection operators, may be the primary driver of systematic alignment effects. Finally, we present a unifying theoretical framework to derive conditions under which consistency training will amplify or suppress misalignment. In total, our study establishes that consistency training is not alignment-neutral, and that its use in critical systems should be carefully audited.
>
---
#### [replaced 054] BenHalluEval: A Multi-Task Hallucination Evaluation Framework for Large Language Models on Bengali
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BenHalluEval，针对孟加拉语大语言模型的幻觉问题，设计多任务评估框架，解决低资源语言幻觉检测不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.31483](https://arxiv.org/pdf/2605.31483)**

> **作者:** Shefayat E Shams Adib; Ahmed Alfey Sani; Ekramul Alam Esham; Ajwad Abrar; Ishmam Tashdeed; Md Taukir Azam Chowdhury
>
> **备注:** Preprint. Under review
>
> **摘要:** Despite Bengali being the sixth most spoken language in the world, no prior work has systematically evaluated hallucination in large language models (LLMs) for Bengali. We introduce BenHalluEval, a fine-grained hallucination evaluation framework for Bengali covering four tasks: Generative Question Answering (GQA), Bangla-English Code-Mixed QA, Summarization, and Reasoning. We construct 12,000 hallucinated candidates using GPT-5.4 across twelve task-specific hallucination types, drawn from three existing Bengali datasets, and evaluate seven LLMs spanning reasoning-oriented, multilingual, and Bengali-centric categories under a dual-track protocol that independently measures false-positive rate on ground-truth instances (Track A) and hallucination detection rate on hallucinated candidates (Track B). To jointly penalise both failure modes and prevent inflated scores from uniform response bias, we propose BenHalluScore, a dual-track calibration metric that ranges from 7.72% to 55.42% across models and tasks, revealing substantial variation in hallucination calibration. Chain-of-thought prompting, applied as a mitigation strategy, shifts response distributions without consistently improving hallucination discrimination. BenHalluEval establishes the first dedicated hallucination benchmark for Bengali and highlights the inadequacy of single-track and prompting-only evaluation approaches for low-resource language settings. The dataset and code are available at this https URL.
>
---
#### [replaced 055] WETBench: A Benchmark for Detecting Task-Specific Machine-Generated Text on Wikipedia
- **分类: cs.CL**

- **简介: 该论文提出WETBench，用于检测维基百科任务特定的机器生成文本。解决MGT检测在真实编辑场景中效果不佳的问题，通过定义三种编辑任务并进行多语言、多生成器评估。**

- **链接: [https://arxiv.org/pdf/2507.03373](https://arxiv.org/pdf/2507.03373)**

> **作者:** Gerrit Quaremba; Elizabeth Black; Denny Vrandečić; Elena Simperl
>
> **摘要:** Given Wikipedia's role as a trusted source of high-quality, reliable content, concerns are growing about the proliferation of low-quality machine-generated text (MGT) produced by large language models (LLMs) on its platform. Reliable detection of MGT is therefore essential. However, existing work primarily evaluates MGT detectors on generic generation tasks rather than on tasks more commonly performed by Wikipedia editors. This misalignment can lead to poor generalisability when applied in real-world Wikipedia contexts. We introduce WETBench, a multilingual, multi-generator, and task-specific benchmark for MGT detection. We define three editing tasks, empirically grounded in Wikipedia editors' perceived use cases for LLM-assisted editing: Paragraph Writing, Summarisation, and Text Style Transfer, which we implement using two new datasets across three languages. For each writing task, we evaluate three prompts, generate MGT across multiple generators using the best-performing prompt, and benchmark diverse detectors. We find that, across settings, training-based detectors achieve an average accuracy of 78%, while zero-shot detectors average 58%. These results show that detectors struggle with MGT in realistic generation scenarios and underscore the importance of evaluating such models on diverse, task-specific data to assess their reliability in editor-driven contexts.
>
---
#### [replaced 056] 100-LongBench: Are de facto Long-Context Benchmarks Literally Evaluating Long-Context Ability?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型评估任务，旨在解决现有长文本基准测试的不足。提出可控制长度的基准和新指标，以更准确评估模型的长上下文能力。**

- **链接: [https://arxiv.org/pdf/2505.19293](https://arxiv.org/pdf/2505.19293)**

> **作者:** Wang Yang; Hongye Jin; Shaochen Zhong; Song Jiang; Qifan Wang; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Long-context capability is considered one of the most important abilities of LLMs, as a truly long context-capable LLM enables users to effortlessly process many originally exhausting tasks -- e.g., digesting a long-form document to find answers vs. directly asking an LLM about it. However, existing real-task-based long-context evaluation benchmarks have two major shortcomings. First, benchmarks like LongBench often do not provide proper metrics to separate long-context performance from the model's baseline ability, making cross-model comparison unclear. Second, such benchmarks are usually constructed with fixed input lengths, which limits their applicability across different models and fails to reveal when a model begins to break down. To address these issues, we introduce a length-controllable long-context benchmark and a novel metric that disentangles baseline knowledge from true long-context capabilities. Experiments demonstrate the superiority of our approach in effectively evaluating LLMs.
>
---
#### [replaced 057] SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在长文本处理中的能力不足问题。通过提出SoLoPO框架，提升模型对长上下文的利用效率。**

- **链接: [https://arxiv.org/pdf/2505.11166](https://arxiv.org/pdf/2505.11166)**

> **作者:** Huashan Sun; Shengyi Liao; Yansen Han; Yu Bai; Yang Gao; Cheng Fu; Weizhou Shen; Fanqi Wan; Ming Yan; Ji Zhang; Fei Huang
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Despite advances in pretraining with extended context sizes, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named \textbf{S}h\textbf{o}rt-to-\textbf{Lo}ng \textbf{P}reference \textbf{O}ptimization (\textbf{SoLoPO}), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
>
---
#### [replaced 058] Demystifying Multi-Agent Debate: The Role of Confidence and Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升多智能体辩论效果。针对传统方法效率低、效果不佳的问题，提出基于观点多样性和置信度沟通的改进方案，显著提升模型表现。**

- **链接: [https://arxiv.org/pdf/2601.19921](https://arxiv.org/pdf/2601.19921)**

> **作者:** Xiaochen Zhu; Caiqi Zhang; Yizhou Chi; Tom Stafford; Nigel Collier; Andreas Vlachos
>
> **摘要:** Multi-agent debate (MAD) is widely used to improve large language model (LLM) performance through test-time scaling, yet recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost. Studies show that, under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes. Drawing on findings from human deliberation and collective decision-making, we identify two key mechanisms missing from vanilla MAD: (i) diversity of initial viewpoints and (ii) explicit, calibrated confidence communication. We propose two lightweight interventions. First, a diversity-aware initialisation that selects a more diverse pool of candidate answers, increasing the likelihood that a correct hypothesis is present at the start of debate. Second, a confidence-modulated debate protocol in which agents express calibrated confidence and condition their updates on others' confidence. We show theoretically that diversity-aware initialisation improves the prior probability of MAD success without changing the underlying update dynamics, while confidence-modulated updates enable debate to systematically drift to the correct hypothesis. Empirically, across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote. Our results connect human deliberation with LLM-based debate and demonstrate that simple, principled modifications can substantially enhance debate effectiveness.
>
---
#### [replaced 059] AlgoVeri: An Aligned Benchmark for Verified Code Generation on Classical Algorithms
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出AlgoVeri基准，用于评估经典算法的代码验证任务。解决现有基准不统一、不可比的问题，通过统一功能契约测试不同工具的表现。**

- **链接: [https://arxiv.org/pdf/2602.09464](https://arxiv.org/pdf/2602.09464)**

> **作者:** Haoyu Zhao; Ziran Yang; Jiawei Li; Deyuan He; Zenan Li; Chi Jin; Venugopal V. Veeravalli; Aarti Gupta; Sanjeev Arora
>
> **备注:** Accepted to ICML 2026, 32 pages
>
> **摘要:** Vericoding refers to the generation of formally verified code from rigorous specifications. Recent AI models show promise in vericoding, but a unified methodology for cross-paradigm evaluation is lacking. Existing benchmarks test only individual languages/tools (e.g., Dafny, Verus, and Lean) and each covers very different tasks, so the performance numbers are not directly comparable. We address this gap with AlgoVeri, a benchmark that evaluates vericoding of $77$ classical algorithms in Dafny, Verus, and Lean. By enforcing identical functional contracts, AlgoVeri reveals critical capability gaps in verification systems. While frontier models achieve tractable success in Dafny ($40.3$% for Gemini-3 Flash), where high-level abstractions and SMT automation simplify the workflow, performance collapses under the systems-level memory constraints of Verus ($24.7$%) and the explicit proof construction required by Lean (7.8%). Beyond aggregate metrics, we uncover a sharp divergence in test-time compute dynamics: Gemini-3 effectively utilizes iterative repair to boost performance (e.g., tripling pass rates in Dafny), whereas GPT-OSS saturates early. Finally, our error analysis shows that language design affects the refinement trajectory: while Dafny allows models to focus on logical correctness, Verus and Lean trap models in persistent syntactic and semantic barriers. All data and evaluation code can be found at this https URL.
>
---
#### [replaced 060] P$^2$-DPO: Grounding Hallucination in Perceptual Processing via Calibration Direct Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，旨在解决模型 hallucination 和视觉鲁棒性问题。提出 P²-DPO 方法，通过自生成偏好对提升感知瓶颈和视觉稳定性。**

- **链接: [https://arxiv.org/pdf/2606.03376](https://arxiv.org/pdf/2606.03376)**

> **作者:** Ruipeng Zhang; Zhihao Li; Haozhang Yuan; C. L. Philip Chen; Tong Zhang
>
> **摘要:** Hallucination has recently garnered significant research attention in Large Vision-Language Models (LVLMs). Direct Preference Optimization (DPO) aims to learn directly from the corrected preferences provided by humans, thereby addressing the hallucination issue. Despite its success, this paradigm has yet to specifically target the perceptual bottleneck in attended regions or address insufficient Visual Robustness against image degradation. Furthermore, existing preference pairs are often vision-agnostic and their inherently off-policy nature limits their effectiveness in guiding model learning. To address these challenges, we propose Perceptual Processing Direct Preference Optimization (P$^2$-DPO), a novel training paradigm in which the model generates and learns from its own preference pairs, thereby directly addressing the identified visual bottlenecks while inherently avoiding the issues of vision-agnostic and off-policy data. It introduces: (1) an on-policy preference pairs construction method targeting Focus-and-Enhance perception and Visual Robustness, and (2) a well-designed Calibration Loss to precisely align visual signals with the causal generation of text. Experimental results demonstrate that with a comparable amount of training data and cost, P$^2$-DPO outperforms strong baselines that rely on costly human feedback on benchmarks. Furthermore, evaluations on Attention Region Fidelity (ARF) and image degradation scenarios validate the effectiveness of P$^2$-DPO in addressing perceptual bottleneck in attended regions and improving Visual Robustness against degraded inputs.
>
---
#### [replaced 061] Translation Heads: Disentangling meaning from language in LLM-based machine translation
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中的机制可解释性，解决LLM在句级翻译中的功能分解问题。通过分析注意力头，区分目标语言生成与语义保持，并构建特定向量提升翻译效果。**

- **链接: [https://arxiv.org/pdf/2602.04613](https://arxiv.org/pdf/2602.04613)**

> **作者:** Théo Lasnier; Armel Zebaze; Djamé Seddah; Rachel Bawden; Benoît Sagot
>
> **备注:** 61 pages, 70 figures
>
> **摘要:** Mechanistic Interpretability (MI) seeks to explain how neural networks implement their capabilities, but the scale of Large Language Models (LLMs) has limited prior MI work in Machine Translation (MT) to word-level analyses. We study sentence-level MT from a mechanistic perspective by analyzing attention heads to understand how LLMs internally encode and distribute translation functions. We decompose MT into two subtasks: producing text in the target language (i.e. target language identification) and preserving the input sentence's meaning (i.e. sentence equivalence). Across three families of open-source models and 20 translation directions, we find that distinct, sparse sets of attention heads specialize in each subtask. Based on this insight, we construct subtask-specific steering vectors and show that modifying just 1% of the relevant heads enables instruction-free MT performance comparable to instruction-based prompting, while ablating these heads selectively disrupts their corresponding translation functions.
>
---
#### [replaced 062] Synthesize and Reward -- Reinforcement Learning for Multi-Step Tool Use in Live Environments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多步骤工具调用的强化学习任务，旨在解决真实环境构建困难、训练数据脱节和奖励机制冗余的问题。提出PROVE框架，包含状态服务器、数据生成和程序化奖励，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.03892](https://arxiv.org/pdf/2606.03892)**

> **作者:** Ibrahim Abdelaziz; Asim Munawar; Kinjal Basu; Maxwell Crouse; Chulaka Gunasekara; Suneet Katrekar; Pavan Kapanipathi
>
> **摘要:** Training LLMs to orchestrate multi-step tool calls is held back by three coupled obstacles: realistic stateful execution environments are costly to build, synthetic training queries are often detached from the server's actual state (so the generated tool calls fail to execute), and recall-based RL rewards incentivize verbose tool-calling patterns. We present PROVE (Programmatic Rewards On Verified Environments), a framework with three contributions: (1) a library of 20 stateful MCP (Model Context Protocol) servers exposing 343 tools, enabling live-execution RL training with session-scoped state isolation; (2) a state-machine data synthesis pipeline that generates multi-turn tool-call trajectories grounded in live-sampled server state, so generated queries reference entities that actually exist; and (3) a multi-component programmatic reward with an adaptive efficiency penalty that counters the verbosity incentive of recall-based rewards. We train four models (Qwen3-4B, Qwen3-8B, Qwen2.5-7B, Granite-4.1-8B) with GRPO on the resulting ~13K training examples. On BFCL Multi-Turn, tau2-bench, and T-Eval, PROVE yields improvements of up to +10.2, +6.8, and +6.5 points respectively, demonstrating that this framework yields consistent gains on multi-step tool orchestration across two model families.
>
---
#### [replaced 063] Coherence Maximization Improves Pluralistic Alignment
- **分类: cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决如何有效生成符合多样人类价值观的示例。通过最大化内部一致性（ICM）生成示例，提升模型对特定群体价值观的适应能力。**

- **链接: [https://arxiv.org/pdf/2606.03110](https://arxiv.org/pdf/2606.03110)**

> **作者:** Taslim Mahbub; Yiding Pei; Shi Feng
>
> **摘要:** Aligning AI systems with diverse human values requires value specifications grounded in concrete examples, but generating such examples without extensive human supervision remains an open challenge. We investigate what makes these examples effective, using Internal Coherence Maximization (ICM) -- which infers labels by maximizing their mutual predictability -- to generate persona-specific examples that steer a model toward a target group's values, without human supervision. Across four benchmarks spanning classification, preference, and open-ended generation, ICM-inferred in-context examples match the performance of gold labels. Crucially, coherence matters beyond individual label accuracy: with accuracy held constant, more coherent examples generalize substantially better than incoherent ones. For personas underrepresented in pretraining data, targeted human feedback on the questions where the model is least certain about a persona's values yields better generalization than the same number of labels on arbitrary questions. These results identify coherence as a key design principle for scalable value specification, leveraging the diverse human perspectives already encoded in pretrained language models.
>
---
#### [replaced 064] Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; physics.soc-ph**

- **简介: 该论文属于自然语言处理中的文化建模任务，旨在解决LLM personas与不同文化价值体系的对齐问题。通过构建基于WVS的 personas，并从文化地图、人口统计和道德框架进行分析，评估其跨文化一致性与道德差异。**

- **链接: [https://arxiv.org/pdf/2601.22396](https://arxiv.org/pdf/2601.22396)**

> **作者:** Candida M. Greco; Lucio La Cava; Andrea Tagarelli
>
> **备注:** Under Review
>
> **摘要:** Despite the growing utility of Large Language Models (LLMs) for simulating human behavior, the extent to which these synthetic personas accurately reflect world and moral value systems across different cultural conditionings remains uncertain. This paper investigates the alignment of synthetic, culturally-grounded personas with established frameworks, specifically the World Values Survey (WVS), the Inglehart-Welzel Cultural Map, and Moral Foundations Theory. We conceptualize and produce LLM-generated personas based on a set of interpretable WVS-derived variables, and we examine the generated personas through three complementary lenses: positioning on the Inglehart-Welzel map, which unveils their interpretation reflecting stable differences across cultural conditionings; demographic-level consistency with the World Values Survey, where response distributions broadly track human group patterns; and moral profiles derived from a Moral Foundations questionnaire, which we analyze through a culture-to-morality mapping to characterize how moral responses vary across different cultural configurations. Our approach of culturally-grounded persona generation and analysis enables evaluation of cross-cultural structure and moral variation.
>
---
#### [replaced 065] How Far Do Auto-Interpretation Labels Generalize: A Controlled Study Across Languages, Scripts, and Rewordings
- **分类: cs.CL**

- **简介: 该论文研究自动解释标签在不同语言和脚本中的泛化能力，旨在解决语言模型特征解释的跨语言有效性问题。通过对比塞尔维亚语不同书写系统，发现标签在不同语言中表现差异显著。**

- **链接: [https://arxiv.org/pdf/2606.00356](https://arxiv.org/pdf/2606.00356)**

> **作者:** Sripad Karne
>
> **摘要:** Sparse autoencoder (SAE) features are increasingly used to interpret language models, with auto-generated natural-language labels serving as the primary interface for understanding what each feature represents. We ask whether these labels generalize: does a feature labeled for a concept actually track that concept across languages and scripts? Using Serbian digraphia as a controlled testbed--the same language written in both Latin and Cyrillic via deterministic transliteration--we first find that SAE feature sets activated by the same content in different languages, scripts, and wordings share substantial overlap (mean Jaccard 0.39 vs. 0.13 random baseline, peaking at 0.57), suggesting genuine cross-lingual semantic features. We then test whether auto-interpretation labels keep pace. They often do not: features whose labels describe semantic content miss the same meaning in Serbian up to 4x more often thanwithin English, and miss Serbian Cyrillic more than Serbian Latin--two scripts that are deterministic transliterations of each other--suggesting the failures align with how well each form is represented in training. The gap grows with network depth, yet the labels give no indication that they fail. These results suggest that auto-interpretation labels may reflect a feature's behavior on well-represented inputs rather than the concept itself.
>
---
#### [replaced 066] PoliticsBench: Benchmarking Political Values in Large Language Models with Multi-Turn Roleplay
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估大语言模型政治价值观的任务，旨在解决现有基准无法细致衡量政治偏见的问题。通过多轮角色扮演测试，分析模型在不同情境下的价值表达与立场变化。**

- **链接: [https://arxiv.org/pdf/2603.23841](https://arxiv.org/pdf/2603.23841)**

> **作者:** Rohan Khetan; Ashna Khetan
>
> **备注:** 7 pages, 5 tables, 5 figures, 4 appendix pages. Accepted to the ICML 2026 Trustworthy AI for Good Workshop
>
> **摘要:** While Large Language Models (LLMs) are increasingly used as primary sources of information, their potential for political bias may impact their objectivity. Existing benchmarks of LLM social bias primarily evaluate demographic stereotypes, and when political bias is measured, it is done so at a coarse level, overlooking the values that shape sociopolitical reasoning. We introduce PoliticsBench, a multi-stage roleplay benchmark for evaluating fine-grained value expression in LLMs. Across twenty evolving scenarios, models articulate tradeoffs, take positions, and make decisions under competing pressures. Across eight prominent LLMs, we show that scenario-based prompting elicits broader and more strongly expressed value profiles than direct political questions, with peak interaction stages increasing the number of strongly activated value dimensions by approximately $0.75$ (out of 10 total dimensions), a statistically significant increase relative to baseline prompting ($p < 0.05$). In addition, commitment to a stance increases over the course of interaction, rising by approximately $1.4$ points on a $[0,5]$ scale from initial to decision stages. While responses become less robust to scenario paraphrasing in later interaction stages, inter-judge agreement remains relatively stable. Our results suggest that evaluating LLM political behavior requires moving beyond static prompts toward longer interactive settings that capture how values are applied in context.
>
---
#### [replaced 067] Learning When to Act or Refuse: Guarding Agentic Reasoning Models for Safe Multi-Step Tool Use
- **分类: cs.CL**

- **简介: 该论文属于安全增强任务，旨在解决智能体模型在多步骤工具使用中的安全问题。通过引入MOSAIC框架，提升模型在复杂场景下的安全决策能力。**

- **链接: [https://arxiv.org/pdf/2603.03205](https://arxiv.org/pdf/2603.03205)**

> **作者:** Aradhye Agarwal; Gurdit Siyan; Yash Pandya; Joykirat Singh; Akshay Nambi; Ahmed Awadallah
>
> **备注:** Accepted to the 43rd International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Agentic language models operate in a fundamentally different safety regime than chat models: they must plan, call tools, and execute long-horizon actions where a single misstep, such as accessing files or entering credentials, can cause irreversible harm. Existing alignment methods, largely optimized for static generation and task completion, break down in these settings due to sequential decision-making, adversarial tool feedback, and overconfident intermediate reasoning. We introduce MOSAIC, a post-training framework that aligns agents for safe multi-step tool use by making safety decisions explicit and learnable. MOSAIC structures inference as a plan, check, then act or refuse loop, with explicit safety reasoning and refusal as first-class actions. To train without trajectory-level labels, we use preference-based reinforcement learning with pairwise trajectory comparisons, which captures safety distinctions often missed by scalar rewards. We evaluate MOSAIC zero-shot across three model families, Qwen2.5-7B, Qwen3-4B-Thinking, and Phi-4, and across out-of-distribution benchmarks spanning harmful tasks, prompt injection, benign tool use, and cross-domain privacy leakage. MOSAIC reduces harmful behavior by up to 50%, increases harmful-task refusal by over 20% on injection attacks, cuts privacy leakage, and preserves or improves benign task performance, demonstrating robust generalization across models, domains, and agentic settings.
>
---
#### [replaced 068] Value Entanglement: Conflation Between Different Kinds of Good In (Some) Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI价值对齐研究，旨在解决LLM中不同价值类型混淆的问题。通过分析模型行为与激活，发现语法和经济价值过度受道德价值影响，并通过消融实验进行修复。**

- **链接: [https://arxiv.org/pdf/2602.19101](https://arxiv.org/pdf/2602.19101)**

> **作者:** Seong Hah Cho; Junyi Li; Anna Leshinskaya
>
> **摘要:** Value alignment of Large Language Models (LLMs) requires us to empirically measure these models' actual, acquired representation of value. Among the characteristics of value representation in humans is that they distinguish among value of different kinds. We investigate whether LLMs likewise distinguish three different kinds of good: moral, grammatical, and economic. By probing model behavior, embeddings, and residual stream activations, we report pervasive cases of value entanglement: a conflation between these distinct representations of value. Specifically, both grammatical and economic valuation was found to be overly influenced by moral value, relative to human norms. This conflation was repaired by selective ablation of the activation vectors associated with morality.
>
---
#### [replaced 069] CART: Context-Anchored Recurrent Transformer -- A Parameter-Efficient Architecture with Learned Stability
- **分类: cs.LG; cs.CL**

- **简介: 论文提出CART模型，解决语言模型参数效率问题。通过共享核心块和稳定机制，优化循环结构，但未在参数匹配下超越基线。任务为语言建模与模型优化。**

- **链接: [https://arxiv.org/pdf/2606.01495](https://arxiv.org/pdf/2606.01495)**

> **作者:** Chad A. Capps
>
> **备注:** 31 pages, 4 figures. Code, training scripts, and the full experiment database (this http URL) are available at this https URL
>
> **摘要:** We present CART (Context-Anchored Recurrent Transformer), a parameter-efficient language model that reuses a single shared core block R times across depth. Unlike prior looped transformers that recompute key-value tensors at every iteration, CART computes K and V once from a multi-layer prelude and has the recurrent core cross-attend to those frozen tensors via multi-head latent attention. A learned Linear Time-Invariant (LTI) gate keeps the recurrence stable: its spectral radius settles in a narrow band (rho in [0.79, 0.83]) across all 36 fully-trained configurations. We evaluate CART on single consumer GPUs in two stages: a 64-configuration screen at 3,000 steps, then 36 configurations (P=6, R in {6,8,10}, three seeds) trained for 30,500 steps (~1B tokens). Two patterns hold across widths d in {256,512,768,1024}: prelude depth P dominates loop count R, and the Stage-1 ranking of R reverses at full training (R=6 becomes best at d>=512). At the binding d=1024 parameter-parity test, CART does not beat a parameter-matched dense baseline, losing by 1-2% at stored-parameter parity and by ~10% at effective-parameter parity. Diagnostic ablations split the effective-parameter gap into ~5% from weight sharing and a residual ~5% from the heterogeneous prelude/anchor/core/coda framing; the recurrent-core machinery (hyper-connections, LTI gate, loop-index embedding) is individually vestigial. Variable-R inference degrades on both sides of the trained R, a negative result for test-time depth scaling under this recipe.
>
---
#### [replaced 070] Can professional translators identify machine-generated text?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在探究专业译者能否识别AI生成的文本。研究通过实验分析译者对AI与人类写作的判断能力及依据。**

- **链接: [https://arxiv.org/pdf/2601.15828](https://arxiv.org/pdf/2601.15828)**

> **作者:** Michael Farrell
>
> **备注:** 10 pages, peer-reviewed and accepted for presentation at EAMT 2026, paged-up for publication
>
> **摘要:** This study investigates whether professional translators without prior specialized training can reliably identify short stories generated in Italian by artificial intelligence (AI). Sixty-nine translators took part in an in-person experiment, where they assessed three anonymized short stories - two written by ChatGPT-4o and one by a human author. For each story, participants rated the likelihood of AI authorship and provided justifications for their choices. While average results were inconclusive, a statistically significant subset (16.2%) successfully distinguished the synthetic texts from the human text, suggesting that their judgements were informed by analytical skill rather than chance. However, a nearly equal number misclassified the texts in the opposite direction, often relying on subjective impressions rather than objective markers, possibly reflecting a reader preference for AI-generated texts. Low burstiness and narrative contradiction emerged as the most reliable indicators of synthetic authorship, with unexpected calques, semantic loans and syntactic transfer from English also reported. In contrast, features such as grammatical accuracy and emotional tone frequently led to misclassification. These findings raise questions about the role and scope of synthetic-text editing in professional contexts.
>
---
#### [replaced 071] UniFine: A Unified and Fine-grained Approach for Zero-shot Vision-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于零样本视觉语言理解任务，旨在提升模型在无监督情况下的语义理解能力。通过引入细粒度信息，提出统一框架，在多个任务中取得更好效果。**

- **链接: [https://arxiv.org/pdf/2307.00862](https://arxiv.org/pdf/2307.00862)**

> **作者:** Rui Sun; Zhecan Wang; Haoxuan You; Noel Codella; Kai-Wei Chang; Shih-Fu Chang
>
> **备注:** 14 pages, 4 figures, ACL 2023 Findings
>
> **摘要:** Vision-language tasks, such as VQA, SNLI-VE, and VCR are challenging because they require the model's reasoning ability to understand the semantics of the visual world and natural language. Supervised methods working for vision-language tasks have been well-studied. However, solving these tasks in a zero-shot setting is less explored. Since Contrastive Language-Image Pre-training (CLIP) has shown remarkable zero-shot performance on image-text matching, previous works utilized its strong zero-shot ability by converting vision-language tasks into an image-text matching problem, and they mainly consider global-level matching (e.g., the whole image or sentence). However, we find visual and textual fine-grained information, e.g., keywords in the sentence and objects in the image, can be fairly informative for semantics understanding. Inspired by this, we propose a unified framework to take advantage of the fine-grained information for zero-shot vision-language learning, covering multiple tasks such as VQA, SNLI-VE, and VCR. Our experiments show that our framework outperforms former zero-shot methods on VQA and achieves substantial improvement on SNLI-VE and VCR. Furthermore, our ablation studies confirm the effectiveness and generalizability of our proposed method.
>
---
#### [replaced 072] Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理控制任务，旨在解决指令驱动的推理行为问题。通过分析触发词元，提出Mid-Think方法，在不训练的情况下实现中间预算推理，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.07036](https://arxiv.org/pdf/2601.07036)**

> **作者:** Wang Yang; Debargha Ganguly; Xinpeng Li; Chaoda Song; Shouren Wang; Vikash Singh; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Hybrid reasoning language models are commonly controlled through high-level Think/No-think instructions to regulate reasoning behavior, yet we found that such mode switching is largely driven by a small set of trigger tokens rather than the instructions themselves. Through attention analysis and controlled prompting experiments, we show that a leading ``Okay'' token induces reasoning behavior, while the newline pattern following ``</think>'' suppresses it. Based on this observation, we propose Mid-Think, a simple training-free prompting format that combines these triggers to achieve intermediate-budget reasoning, consistently outperforming fixed-token and prompt-based baselines in terms of the accuracy-length trade-off. Furthermore, applying Mid-Think to RL training after SFT reduces training time by approximately 15% while improving final performance of Qwen3-8B on AIME from 69.8% to 72.4% and on GPQA from 58.5% to 61.1%, demonstrating its effectiveness for both inference-time control and RL-based reasoning training.
>
---
#### [replaced 073] Constrained Adaptive Rejection Sampling
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CARS方法，用于约束生成任务，解决传统方法在效率和分布准确性上的不足。通过自适应剪枝提升采样效率与多样性。**

- **链接: [https://arxiv.org/pdf/2510.01902](https://arxiv.org/pdf/2510.01902)**

> **作者:** Paweł Parys; Sairam Vaidya; Taylor Berg-Kirkpatrick; Loris D'Antoni
>
> **摘要:** Language Models (LMs) are increasingly used in applications where generated outputs must satisfy strict semantic or syntactic constraints. Existing approaches to constrained generation fall along a spectrum: greedy constrained decoding methods enforce validity during decoding but distort the LM's distribution, while rejection sampling (RS) preserves fidelity but wastes computation by discarding invalid outputs. Both extremes are problematic in domains such as program fuzzing, where both validity and diversity of samples are essential. We present Constrained Adaptive Rejection Sampling (CARS), an approach that strictly improves the sample-efficiency of RS without distributional distortion. CARS begins with unconstrained LM sampling and adaptively rules out constraint-violating continuations by recording them in a trie and subtracting their probability mass from future draws. This adaptive pruning ensures that prefixes proven invalid are never revisited, acceptance rates improve monotonically, and the resulting samples exactly follow the constrained distribution. In experiments on a variety of domains -- e.g., program fuzzing and molecular generation -- CARS consistently achieves higher efficiency -- measured in the number of LM forward passes per valid sample -- while also producing stronger sample diversity than both GCD and methods that approximate the LM's distribution.
>
---
#### [replaced 074] Aryabhata 2: Scaling Reinforcement Learning for Advanced STEM Reasoning
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出Aryabhata 2，用于解决STEM竞赛题的多步骤推理问题。通过强化学习优化模型，提升解题能力并减少输出token。**

- **链接: [https://arxiv.org/pdf/2605.28829](https://arxiv.org/pdf/2605.28829)**

> **作者:** Ritvik Rastogi; Vishal Singh; Tejas Chaudhari; Sandeep Varma
>
> **摘要:** Competitive STEM examinations such as JEE and NEET require multi-step symbolic reasoning, precise numerical computation, and deep conceptual understanding across physics, chemistry, and mathematics. Recent large language models perform strongly on common reasoning benchmarks, yet they remain difficult to deploy at scale, where millions of student doubts demand domain-specific, consistently structured problem solving. We introduce Aryabhata 2, a reasoning-focused language model for competitive STEM examinations, trained via reinforcement-learning post-training. Using PhysicsWallah's internal question banks, we construct a high-quality training curriculum and post-train GPT-OSS-20B through reinforcement learning with verifiable rewards. Training combines prolonged reinforcement learning with broadened exploration via progressively larger rollout group sizes. We evaluate Aryabhata 2 on competitive examination benchmarks, including JEE Main, JEE Advanced, and NEET, as well as out-of-distribution reasoning datasets such as AIME, HMMT, MMLU-Pro, MMLU-Redux 2.0, and GPQA. Results show that Aryabhata 2 outperforms its base model GPT-OSS-20B on competitive STEM reasoning while requiring substantially fewer output tokens (up to 64\% fewer).
>
---
#### [replaced 075] Trust Region On-Policy Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TrOPD，解决OPD在分布差异大时的不稳定问题，通过信任区域和异常值处理提升监督可靠性，用于语言模型高效微调。**

- **链接: [https://arxiv.org/pdf/2606.01249](https://arxiv.org/pdf/2606.01249)**

> **作者:** Xingrun Xing; Haoqing Wang; Boyan Gao; Ziheng Li; Yehui Tang
>
> **摘要:** On-Policy Distillation (OPD) is a fundamental technique for efficient post-training of large language models (LLMs), with broad applications in agent learning, multi-task enhancement, and model compression. However, OPD training becomes unstable when the teacher and student distributions differ substantially, as teacher supervision on student-generated tokens may yield unreliable policy gradients and even cause optimization failure. This work addresses reliable on-policy token-level supervision through credit assignment strategies, and proposes Trust Region On-Policy Distillation, TrOPD. It features the following characteristics: 1) Trust-Region On-Policy Learning: TrOPD performs OPD only in regions where the teacher provides reliable supervision, mitigating the optimization difficulty of the K1 reverse-KL estimator under distribution mismatch. 2) Outlier Estimation: For outlier regions, we explore gradient clipping, masking, and forward-KL estimation to reduce the adverse effects of unreliable supervision. 3) Off-Policy Guidance: The student continues generation from teacher prefixes and uses forward KL to imitate off-policy guidance, encouraging on-policy exploration toward reliable regions. Experiments show that TrOPD consistently outperforms SoTA OPD baselines, including OPD, EOPD, and REOPOLD, across mathematical reasoning, code generation, and general-domain benchmarks.
>
---
#### [replaced 076] Recovering Diversity Without Losing Alignment: A DPO Recipe for Post-Trained LLMs
- **分类: cs.CL**

- **简介: 该论文属于大模型优化任务，旨在解决后训练导致输出多样性下降的问题。通过构造偏好数据提升模型多样性，同时保持对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.30021](https://arxiv.org/pdf/2605.30021)**

> **作者:** Vinay Samuel; Yapei Chang; Mohit Iyyer
>
> **备注:** Under Review. 26 pages, 3 figures, 16 tables
>
> **摘要:** Many open-ended instructions have multiple valid answers that users can benefit from seeing, but post-training often narrows an LLM's output space toward a small set of canonical responses. We introduce REDIPO, an offline DPO data-construction pipeline for recovering distinct valid answer modes while preserving the alignment benefits of the instruct model. For each prompt, REDIPO samples responses from both base and instruct models, rewrites base-model responses with the instruct model, filters candidates for safety and instruction-following quality, and builds preference pairs that favor marginally diverse responses among candidates with similar instruction-following reward. Across Qwen3-4B, OLMo-3-7B, and LLaMA-3.1-8B, REDIPO improves NoveltyBench distinct_k by 134%, 33%, and 44% relative to the instruct checkpoints, while DivPO changes diversity by 0%, -6%, and -4% on the same models. These gains largely maintain MTBench, IFEval, and Arena-Hard performance, and reduce direct-category HarmBench attack success rate. Ablations show that marginal-diversity pair selection and base-response rewriting drive the diversity gains, while filtering and quality-bounded pairing help maintain alignment. Overall, our results show that diverse valid answers from base-model generations can be reintroduced through carefully constructed preference data while retaining the alignment benefits of post-training. We release our code and data at this https URL.
>
---
#### [replaced 077] Longer Context, Deeper Thinking: Uncovering the Role of Long-Context Ability in Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨长上下文能力对推理的影响。研究旨在解决当前模型推理能力不足的问题，通过提升长上下文能力来增强推理表现。**

- **链接: [https://arxiv.org/pdf/2505.17315](https://arxiv.org/pdf/2505.17315)**

> **作者:** Wang Yang; Zirui Liu; Hongye Jin; Qingyu Yin; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Recent language models exhibit strong reasoning capabilities, yet the influence of long-context capacity on reasoning remains underexplored. In this work, we hypothesize that current limitations in reasoning stem, in part, from insufficient long-context capacity, motivated by empirical observations such as (1) higher context window length often leads to stronger reasoning performance, and (2) failed reasoning cases resemble failed long-context cases. To test this hypothesis, we examine whether enhancing a model's long-context ability before Supervised Fine-Tuning (SFT) leads to improved reasoning performance. Specifically, we compared models with identical architectures and fine-tuning data but varying levels of long-context capacity. Our results reveal a consistent trend: models with stronger long-context capacity achieve significantly higher accuracy on reasoning benchmarks after SFT. Notably, these gains persist even on tasks with short input lengths, indicating that long-context training offers generalizable benefits for reasoning performance. These findings suggest that long-context modeling is not just essential for processing lengthy inputs, but also serves as a critical foundation for reasoning. We advocate for treating long-context capacity as a first-class objective in the design of future language models.
>
---
#### [replaced 078] FinTradeBench: A Financial Reasoning Benchmark for LLMs
- **分类: cs.CE; cs.AI; cs.CL; cs.IR; q-fin.CP**

- **简介: 该论文提出FinTradeBench，一个评估大语言模型金融推理能力的基准，解决现有基准缺乏对交易信号与基本面综合推理的问题。**

- **链接: [https://arxiv.org/pdf/2603.19225](https://arxiv.org/pdf/2603.19225)**

> **作者:** Yogesh Agrawal; Aniruddha Dutta; Md Mahadi Hasan; Santu Karmaker; Aritra Dutta
>
> **备注:** 9 pages main text, 32 pages total (including references and appendix). 5 figures, 16 tables. Preprint under review. Code and data will be made available upon publication
>
> **摘要:** Real-world financial decision-making is a challenging problem that requires reasoning over heterogeneous signals, including company fundamentals derived from regulatory filings and trading signals computed from price dynamics. Recently, with advances in Large Language Models (LLMs), financial analysts have begun to use them for financial decision-making tasks. However, existing financial question-answering benchmarks for testing these models primarily focus on company balance sheet data and rarely evaluate reasoning about how company stocks trade in the market or their interactions with fundamentals. To leverage the strengths of both approaches, we introduce FinTradeBench, a benchmark for evaluating financial reasoning that integrates company fundamentals and trading signals. FinTradeBench contains 1,400 questions grounded in NASDAQ-100 companies over a ten-year historical window. The benchmark is organized into three reasoning categories: fundamentals-focused, trading-signal-focused, and hybrid questions requiring cross-signal reasoning. To ensure reliability at scale, we adopt a calibration-then-scaling framework that combines expert seed questions, multi-model response generation, intra-model self-filtering, numerical auditing, and human-LLM judge alignment. We evaluate 14 LLMs under zero-shot prompting and retrieval-augmented settings and witness a clear performance gap. Retrieval substantially improves reasoning over textual fundamentals, but provides limited benefit for trading-signal reasoning. These findings highlight fundamental challenges in the numerical and time-series reasoning for current LLMs and motivate future research in financial intelligence.
>
---
#### [replaced 079] T$^\star$: Progressive Block Scaling for Masked Diffusion Language Models Through Trajectory Aware Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出T$^\star$，用于改进掩码扩散语言模型的块大小渐进扩展，解决高效解码与性能保持的矛盾。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2601.11214](https://arxiv.org/pdf/2601.11214)**

> **作者:** Hanchen Xia; Baoyou Chen; Yutang Ge; Guojiang Zhao; Siyu Zhu
>
> **摘要:** We present T$^\star$, a simple TraceRL-based training curriculum for progressive block-size scaling in masked diffusion language models (MDMs). Starting from an AR-initialized small-block MDM, T$^\star$ transitions smoothly to larger blocks, enabling higher-parallelism decoding with minimal performance degradation on math reasoning benchmarks. Moreover, further analysis suggests that T$^\star$ may actually converge to an alternative decoding schedule that achieves comparable performance.
>
---
#### [replaced 080] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于知识增强生成任务，旨在解决传统RAG方法在语义结构、效率和依赖长文本推理的问题。提出Graph-R1框架，通过强化学习实现端到端优化，提升推理准确性和生成质量。**

- **链接: [https://arxiv.org/pdf/2507.21892](https://arxiv.org/pdf/2507.21892)**

> **作者:** Haoran Luo; Haihong E; Guanting Chen; Qika Lin; Yikai Guo; Fangzhi Xu; Zemin Kuang; Meina Song; Xiaobao Wu; Yifan Zhu; Luu Anh Tuan
>
> **备注:** Accepted by ICML 2026 main conference
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1, the first agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our software and data are publicly available at this https URL.
>
---
#### [replaced 081] Do readers prefer AI-generated Italian short stories?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本评估任务，旨在探究读者是否更偏好AI生成的意大利短篇小说还是人类作家的作品。通过实验对比，发现AI文本略受青睐，但差异不显著。**

- **链接: [https://arxiv.org/pdf/2601.17363](https://arxiv.org/pdf/2601.17363)**

> **作者:** Michael Farrell
>
> **备注:** 8 pages, peer-reviewed and accepted for presentation at New Trends in Translation and Interpreting Technology (NeTTIT 2026), paged-up for publication
>
> **摘要:** This study investigates whether readers prefer AI-generated short stories in Italian over one written by a renowned Italian author. In a blind setup, 20 participants read and evaluated three stories, two created with ChatGPT-4o and one by Alberto Moravia, without being informed of their origin. To explore potential influencing factors, reading habits and demographic data, comprising age, gender, education and first language, were also collected. The results showed that the AI-written texts received slightly higher average ratings and were more frequently preferred, although differences were modest. No statistically significant associations were found between text preference and demographic or reading-habit variables. These findings challenge assumptions about reader preference for human-authored fiction and raise questions about the necessity of synthetic-text editing in literary contexts.
>
---
#### [replaced 082] Beyond Ideal Instruction: A Comprehensive Framework for Evaluating LLMs in Realistic Interactions
- **分类: cs.CL**

- **简介: 该论文属于大语言模型评估任务，旨在解决现有基准与真实用户交互不匹配的问题。提出RUT-Bench基准，评估LLMs在多样化用户场景下的表现。**

- **链接: [https://arxiv.org/pdf/2606.03318](https://arxiv.org/pdf/2606.03318)**

> **作者:** Xuan Yang; Hao Xu; Tingfeng Hui; Hongsheng Xin; Kaike Zhang; Chunxiao Liu; Ning Miao
>
> **摘要:** Despite great advances in tool-use capabilities of large language models (LLMs), existing evaluation benchmarks struggle to fully align with real-world scenarios. Such benchmarks mostly rely on simulated idealized user assumptions and lacks experience-oriented evaluation. These limitations fail to account for the ambiguity, uncooperative behaviors, and shifting intentions characteristic of real-world users. To fill this gap, we propose RUT-Bench, a dedicated benchmark designed to assess LLMs under diverse Real-world User Tool calling scenarios. RUT-Bench supports high-fidelity simulations covering both ideal rational patterns and heterogeneous non-ideal behaviors across single-turn and multi-turn dialogues. We conduct comprehensive evaluations on 19 widely adopted open-source and proprietary LLMs using our benchmark. Experimental results reveal that no tested LLMs achieve an overall success rate above 40%, and nearly all of them experience noticeable performance drops when facing more complicated non-ideal user inputs. Our code and data is available at this https URL.
>
---
#### [replaced 083] Breaking Bad Molecules: Are MLLMs Ready for Structure-Level Molecular Detoxification?
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ToxiMol任务，解决分子毒性修复问题。构建了标准化数据集，设计评估框架，评估MLLMs在毒性降低方面的表现。**

- **链接: [https://arxiv.org/pdf/2506.10912](https://arxiv.org/pdf/2506.10912)**

> **作者:** Fei Lin; Ziyang Gong; Cong Wang; Tengchao Zhang; Yonglin Tian; Yining Jiang; Ji Dai; Chao Guo; Xiaotong Yu; Xue Yang; Gen Luo; Fei-Yue Wang
>
> **摘要:** Toxicity remains a leading cause of early-stage drug development failure. Despite advances in molecular design and property prediction, the task of molecular toxicity repair, generating structurally valid molecular alternatives with reduced toxicity, has not yet been systematically defined or benchmarked. To fill this gap, we introduce ToxiMol, the first benchmark task for general-purpose Multimodal Large Language Models (MLLMs) focused on molecular toxicity repair. We construct a standardized dataset covering 11 primary tasks and 660 representative toxic molecules spanning diverse mechanisms and granularities. We design a prompt annotation pipeline with mechanism-aware and task-adaptive capabilities, informed by expert toxicological knowledge. In parallel, we propose an automated evaluation framework, ToxiEval, which integrates toxicity endpoint prediction, synthetic accessibility, drug-likeness, and structural similarity into a high-throughput evaluation chain for repair success. We systematically assess 43 mainstream general-purpose MLLMs and conduct multiple ablation studies to analyze key issues, including evaluation metrics, candidate diversity, and failure attribution. Experimental results show that although current MLLMs still face significant challenges on this task, they begin to demonstrate promising capabilities in toxicity understanding, semantic constraint adherence, and structure-aware editing.
>
---
#### [replaced 084] ChatSOP: An SOP-Guided MCTS Planning Framework for Controllable LLM Dialogue Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决LLM对话代理的不可控问题。通过引入SOP和MCTS框架，提升对话的可控性与准确性。**

- **链接: [https://arxiv.org/pdf/2407.03884](https://arxiv.org/pdf/2407.03884)**

> **作者:** Zhigen Li; Jianxiang Peng; Yanmeng Wang; Yong Cao; Tianhao Shen; Minghui Zhang; Linxi Su; Shang Wu; Yihang Wu; Yuqian Wang; Ye Wang; Wei Hu; Jianfeng Li; Shaojun Wang; Jing Xiao; Deyi Xiong
>
> **备注:** Accepted to ACL 2025 main
>
> **摘要:** Dialogue agents powered by Large Language Models (LLMs) show superior performance in various tasks. Despite the better user understanding and human-like responses, their **lack of controllability** remains a key challenge, often leading to unfocused conversations or task failure. To address this, we introduce Standard Operating Procedure (SOP) to regulate dialogue flow. Specifically, we propose **ChatSOP**, a novel SOP-guided Monte Carlo Tree Search (MCTS) planning framework designed to enhance the controllability of LLM-driven dialogue agents. To enable this, we curate a dataset comprising SOP-annotated multi-scenario dialogues, generated using a semi-automated role-playing system with GPT-4o and validated through strict manual quality control. Additionally, we propose a novel method that integrates Chain of Thought reasoning with supervised fine-tuning for SOP prediction and utilizes SOP-guided Monte Carlo Tree Search for optimal action planning during dialogues. Experimental results demonstrate the effectiveness of our method, such as achieving a 27.95% improvement in action accuracy compared to baseline models based on GPT-3.5 and also showing notable gains for open-source models. Dataset and codes are publicly available.
>
---
#### [replaced 085] DeInfer: Efficient Parallel Inferencing for Decomposed Large Language Models
- **分类: cs.CL; cs.DC**

- **简介: 该论文属于大语言模型推理任务，旨在解决分解后模型并行推理性能差的问题。提出DeInfer系统，通过优化提升分解模型的并行推理效率。**

- **链接: [https://arxiv.org/pdf/2604.17709](https://arxiv.org/pdf/2604.17709)**

> **作者:** You-Liang Huang; Xinhao Huang; Chengxi Liao; Zeyi Wen
>
> **备注:** accepted by DAC'26, latest version fixs a minor mistake
>
> **摘要:** Existing works on large language model (LLM) decomposition mainly focus on improving performance on downstream tasks, but they ignore the poor parallel inference performance when trying to scale up the model size. To mitigate this important performance issue, this paper introduces DeInfer, a high-performance inference system dedicated to parallel inference of decomposed LLMs. It consists of multiple optimizations to maximize performance and be compatible with state-of-the-art optimization techniques. Extensive experiments are carried out to evaluate DeInfer's performance, where the results demonstrate its superiority, suggesting it can greatly facilitate the parallel inference of decomposed LLMs.
>
---
#### [replaced 086] Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing
- **分类: cs.CL**

- **简介: 该论文属于符号语言对齐任务，旨在将字幕与手语视频同步。提出SEA方法，通过分割、嵌入和对齐实现跨语言、跨领域的高效对齐。**

- **链接: [https://arxiv.org/pdf/2512.08094](https://arxiv.org/pdf/2512.08094)**

> **作者:** Zifan Jiang; Youngjoon Jang; Liliane Momeni; Gül Varol; Sarah Ebling; Andrew Zisserman
>
> **备注:** Camera-ready version of ACL 2026 (Main)
>
> **摘要:** The goal of this work is to develop a universal approach for aligning subtitles (i.e., spoken language text with corresponding timestamps) to continuous sign language videos. Prior approaches typically rely on end-to-end training tied to a specific language or dataset, which limits their generality. In contrast, our method Segment, Embed, and Align (SEA) provides a single framework that works across multiple languages and domains. SEA leverages two pretrained models: the first to segment a video frame sequence into individual signs and the second to embed the video clip of each sign into a shared latent space with text. Alignment is subsequently performed with a lightweight dynamic programming procedure that runs efficiently on CPUs within a minute, even for hour-long episodes. SEA is flexible and can adapt to a wide range of scenarios, utilizing resources from small lexicons to large continuous corpora. Experiments on four sign language datasets demonstrate state-of-the-art alignment performance, highlighting the potential of SEA to generate high-quality parallel data for advancing sign language processing. SEA's code and models are openly available.
>
---
#### [replaced 087] The Mechanistic Emergence of Symbol Grounding in Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究语言模型中符号接地的机制，解决符号如何通过内部计算获得意义的问题。通过分析发现，接地集中在中间层，由注意力机制实现。**

- **链接: [https://arxiv.org/pdf/2510.13796](https://arxiv.org/pdf/2510.13796)**

> **作者:** Shuyu Wu; Ziqiao Ma; Xiaoxi Luo; Yidong Huang; Josue Torres-Fonseca; Freda Shi; Joyce Chai
>
> **摘要:** Symbol grounding (Harnad, 1990) describes how symbols such as words acquire their meanings by connecting to real-world sensorimotor experiences. Recent work has shown preliminary evidence that grounding may emerge in (vision-)language models trained at scale without using explicit grounding objectives. Yet, the specific loci of this emergence and the mechanisms that drive it remain largely unexplored. To address this problem, we introduce a controlled evaluation framework that systematically traces how symbol grounding arises within the internal computations through mechanistic and causal analysis. Our findings show that grounding concentrates in middle-layer computations and is implemented through the aggregate mechanism, where attention heads aggregate the environmental ground to support the prediction of linguistic forms. This phenomenon replicates in multimodal dialogue and across architectures (Transformers and state-space models), but not in unidirectional LSTMs. Our results provide behavioral and mechanistic evidence that symbol grounding can emerge in language models, with practical implications for predicting and potentially controlling the reliability of generation.
>
---
#### [replaced 088] GroupTravelBench: Benchmarking LLM Agents on Multi-Person Travel Planning
- **分类: cs.CL**

- **简介: 该论文提出GroupTravelBench，用于评估大语言模型在多人旅行规划中的能力。解决多用户协作中的偏好收集、冲突协调与公平性平衡问题，通过多轮对话和真实数据构建基准。**

- **链接: [https://arxiv.org/pdf/2605.25200](https://arxiv.org/pdf/2605.25200)**

> **作者:** Xiang Cheng; Yulan Hu; Lulu Zheng; Zheng Pan; Xin Li; Yong Liu
>
> **备注:** work in process
>
> **摘要:** Travel planning in the real world is overwhelmingly a \textit{group} activity, yet existing LLM travel-planning benchmarks reduce it to a single user, where the field is approaching saturation. This single-user assumption sidesteps what makes group planning hard for an agent: discovering private preferences across multiple users, surfacing conflicts, and balancing utility against fairness. To bring the task back to its multi-user reality, we introduce \textbf{\textit{GroupTravelBench}}, the first benchmark for \textbf{multi-user, multi-turn} travel planning. Built from real user profiles, POI data, and ticket prices, it comprises 650 tasks across three difficulty levels, each running in a synchronous group-chat sandbox with cached tool data for reproducible offline evaluation. Beyond the multi-step reasoning and tool use that single-user benchmarks already test, GroupTravelBench probes three group-specific capabilities: \textit{(i) elicitation} of private preferences through multi-turn dialogue; \textit{(ii) coordination} of inter-user conflicts via compromise or subgrouping; and \textit{(iii) planning} that balances group utility against fairness. We pair this with a complementary evaluation framework combining rule-based outcome metrics and LLM-judge process metrics. Across a wide range of frontier models, even the strongest agents fall short on all four rule-based outcome metrics, with plan validity below 12\%, suggesting that group-level outcome quality is a key open challenge for LLM travel-planning agents.
>
---
#### [replaced 089] Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别中的方言分类任务，旨在解决缺乏可靠社会语言标签的问题。通过仅使用声学标签提取特征，提升区域口音识别效果。**

- **链接: [https://arxiv.org/pdf/2605.30457](https://arxiv.org/pdf/2605.30457)**

> **作者:** Pedro H. L. Leite; Pedro Benevenuto Valadares; Luiz W. P. Biscainho
>
> **备注:** This work was submitted to the XLIV Brazilian Symposium on Telecommunications and Signal Processing (SBrT 2026)
>
> **摘要:** Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels.
>
---
#### [replaced 090] Speculative Thinking: Enhancing Small-Model Reasoning with Large Model Guidance at Inference Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型推理任务，旨在提升小模型的推理能力。通过大模型在推理阶段的指导，增强小模型的思考过程，提高准确率并缩短输出长度。**

- **链接: [https://arxiv.org/pdf/2504.12329](https://arxiv.org/pdf/2504.12329)**

> **作者:** Wang Yang; Xiang Yue; Vipin Chaudhary; Xiaotian Han
>
> **摘要:** Recent advances leverage post-training to enhance model reasoning performance, which typically requires costly training pipelines and still suffers from inefficient, overly lengthy outputs. We introduce Speculative Thinking, a training-free framework that enables large reasoning models to guide smaller ones during inference at the reasoning level, distinct from speculative decoding, which operates at the token level. Our approach is based on two observations: (1) reasoning-supportive tokens such as "wait" frequently appear after structural delimiters like "\n\n", serving as signals for reflection or continuation; and (2) larger models exhibit stronger control over reflective behavior, reducing unnecessary backtracking while improving reasoning quality. By strategically delegating reflective steps to a more capable model, our method significantly boosts the reasoning accuracy of reasoning models while shortening their output. With the assistance of the 32B reasoning model, the 1.5B model's accuracy on MATH500 increases from 83.2% to 89.4%, marking a substantial improvement of 6.2%. Simultaneously, the average output length is reduced from 5439 tokens to 4583 tokens, representing a 15.7% decrease. Moreover, when applied to a non-reasoning model (Qwen-2.5-7B-Instruct), our framework boosts its accuracy from 74.0% to 81.8% on the same benchmark, achieving a relative improvement of 7.8%.
>
---
#### [replaced 091] Efficient Reasoning on the Edge
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于边缘计算任务，旨在解决大语言模型在移动设备上推理效率低的问题。通过LoRA适配器和强化学习优化，提升小模型的推理能力与效率。**

- **链接: [https://arxiv.org/pdf/2603.16867](https://arxiv.org/pdf/2603.16867)**

> **作者:** Yelysei Bondarenko; Thomas Hehn; Rob Hesselink; Romain Lepert; Fabio Valerio Massoli; Evgeny Mironov; Leyla Mirvakhabova; Tribhuvanesh Orekondy; Spyridon Stasis; Andrey Kuzmin; Anna Kuzina; Markus Nagel; Ankita Nayak; Corrado Rainone; Ork de Rooij; Paul N Whatmough; Arash Behboodi; Babak Ehteshami Bejnordi
>
> **备注:** Project page: this https URL
>
> **摘要:** Large language models (LLMs) with chain-of-thought reasoning achieve state-of-the-art performance across complex problem-solving tasks, but their verbose reasoning traces and large context requirements make them impractical for edge deployment. These challenges include high token generation costs, large KV-cache footprints, and inefficiencies when distilling reasoning capabilities into smaller models for mobile devices. Existing approaches often rely on distilling reasoning traces from larger models into smaller models, which are verbose and stylistically redundant, undesirable for on-device inference. In this work, we propose a lightweight approach to enable reasoning in small LLMs using LoRA adapters combined with supervised fine-tuning. We further introduce budget forcing via reinforcement learning on these adapters, significantly reducing response length with minimal accuracy loss. To address memory-bound decoding, we exploit parallel test-time scaling, improving accuracy at minor latency increase. Finally, we present a dynamic adapter-switching mechanism that activates reasoning only when needed and a KV-cache sharing strategy during prompt encoding, reducing time-to-first-token for on-device inference. Experiments on Qwen2.5-7B demonstrate that our method achieves efficient, accurate reasoning under strict resource constraints, making LLM reasoning practical for mobile scenarios. Videos demonstrating our solution running on mobile devices are available on our project page.
>
---
#### [replaced 092] Characterizing, Evaluating, and Optimizing Complex Reasoning
- **分类: cs.CL**

- **简介: 该论文属于人工智能推理任务，旨在解决推理质量定义、评估及优化问题。提出ME$^2$原则和DAG评估方法，构建数据集并训练TRM模型提升推理效果。**

- **链接: [https://arxiv.org/pdf/2602.08498](https://arxiv.org/pdf/2602.08498)**

> **作者:** Haoran Zhang; Yafu Li; Zhi Wang; Zhilin Wang; Shunkai Zhang; Xiaoye Qu; Yu Cheng
>
> **备注:** Code and data are available at this https URL
>
> **摘要:** Large Reasoning Models (LRMs) increasingly rely on reasoning traces with complex internal structures. However, existing work lacks a unified answer to three fundamental questions: (1) what defines high-quality reasoning, (2) how to reliably evaluate long, implicitly structured reasoning traces, and (3) how to use such evaluation signals for reasoning optimization. To address these challenges, we provide a unified perspective. (1) We introduce the ME$^2$ principle to characterize reasoning quality along macro- and micro-level concerning efficiency and effectiveness. (2) Built on this principle, we model reasoning traces as directed acyclic graphs (DAGs) and develop a DAG-based pairwise evaluation method, capturing complex reasoning structures. (3) Based on this method, we construct the TRM-Preference dataset and train a Thinking Reward Model (TRM) to evaluate reasoning quality at scale. Experiments show that thinking rewards serve as an effective optimization signal. At test time, selecting better reasoning leads to better outcomes (up to 19.3\% gain), and during RL training, thinking rewards enhance reasoning and performance (up to 3.9\% gain) across diverse tasks. Code and data are available at this https URL.
>
---
#### [replaced 093] ZeroUnlearn: Few-Shot Knowledge Unlearning in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于知识去除任务，解决大模型中敏感信息留存问题。提出ZeroUnlearn框架，通过模型编辑实现高效、精准的少样本知识删除。**

- **链接: [https://arxiv.org/pdf/2605.18879](https://arxiv.org/pdf/2605.18879)**

> **作者:** Yujie Lin; Chengyi Yang; Zhishang Xiang; Yiping Song; Jinsong Su
>
> **摘要:** Large language models inevitably retain sensitive information, defined as inputs that may induce harmful generations, due to training on massive web corpora, raising concerns for privacy and safety. Existing machine unlearning methods primarily rely on retraining or aggressive fine-tuning, which are either computationally expensive or prone to degrading related knowledge and overall model utility. In this work, we reformulate machine unlearning as a precise knowledge re-mapping problem via model editing. We propose ZeroUnlearn, a few-shot unlearning framework. It overwrites sensitive inputs by mapping them to a neutral target state and removing their original representations. ZeroUnlearn enforces representational orthogonality through a multiplicative parameter update with a closed-form solution, enabling efficient and targeted unlearning. We further extend ZeroUnlearn to a gradient-based variant for multi-sample unlearning. Experiments demonstrate that our approach outperforms existing baselines while preserving general model utility. Our code is available at the github: this https URL.
>
---
#### [replaced 094] Policy Split: Incentivizing Dual-Mode Exploration in LLM Reinforcement with Dual-Mode Entropy Regularization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决大语言模型探索与准确性的平衡问题。提出Policy Split方法，通过双模式熵正则化实现有效探索。**

- **链接: [https://arxiv.org/pdf/2604.11510](https://arxiv.org/pdf/2604.11510)**

> **作者:** Jiashu Yao; Heyan Huang; Daiqing Wu; Zeming Liu; Yuhang Guo
>
> **备注:** preprint
>
> **摘要:** To encourage diverse exploration in reinforcement learning (RL) for large language models (LLMs) without compromising accuracy, we propose Policy Split, a novel paradigm that bifurcates the policy into normal and high-entropy modes with a high-entropy prompt. While sharing model parameters, the two modes undergo collaborative dual-mode entropy regularization tailored to distinct objectives. Specifically, the normal mode optimizes for task correctness, while the high-entropy mode incorporates a preference for exploration, and the two modes learn collaboratively. Extensive experiments demonstrate that our approach consistently outperforms established entropy-guided RL baselines across various model sizes in general and creative tasks. Further analysis reveals that Policy Split facilitates dual-mode exploration, where the high-entropy mode generates distinct behavioral patterns to the normal mode, providing unique learning signals.
>
---
#### [replaced 095] Inclusion-of-Thoughts: Mitigating Preference Instability via Purifying the Decision Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，针对大模型在多选题中因干扰项导致的偏好不稳定问题，提出Inclusion-of-Thoughts方法，通过过滤干扰项提升模型决策稳定性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.04944](https://arxiv.org/pdf/2604.04944)**

> **作者:** Mohammad Reza Ghasemi Madani; Soyeon Caren Han; Shuo Yang; Jey Han Lau
>
> **摘要:** Multiple-choice questions (MCQs) are widely used to evaluate large language models (LLMs). However, LLMs remain vulnerable to the presence of plausible distractors. This often diverts attention toward irrelevant choices, resulting in unstable oscillation between correct and incorrect answers. In this paper, we propose Inclusion-of-Thoughts (IoT), a progressive self-filtering strategy that is designed to mitigate this cognitive load (i.e., instability of model preferences under the presence of distractors) and enable the model to focus more effectively on plausible answers. Our method operates to reconstruct the MCQ using only plausible option choices, providing a controlled setting for examining comparative judgements and therefore the stability of the model's internal reasoning under perturbation. By explicitly documenting this filtering process, IoT also enhances the transparency and interpretability of the model's decision-making. Extensive empirical evaluation demonstrates that IoT substantially boosts chain-of-thought performance across a range of arithmetic, commonsense reasoning, and educational benchmarks with minimal computational overhead.
>
---
#### [replaced 096] OckBench: Measuring the Efficiency of LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决LLM在推理和编码任务中效率不足的问题。提出OckBench基准，同时衡量准确性和token效率。**

- **链接: [https://arxiv.org/pdf/2511.05722](https://arxiv.org/pdf/2511.05722)**

> **作者:** Zheng Du; Hao Kang; Song Han; Tushar Krishna; Ligeng Zhu
>
> **摘要:** Large language models (LLMs) such as GPT-5 and Gemini 3 have pushed the frontier of automated reasoning and code generation. Yet current benchmarks emphasize accuracy and output quality, neglecting a critical dimension: efficiency of token usage. The token efficiency is highly variable in practical. Models solving the same problem with similar accuracy can exhibit up to a \textbf{5.0$\times$} difference in token length, leading to massive gap of model reasoning ability. Such variance exposes significant redundancy, highlighting the critical need for a standardized benchmark to quantify the gap of token efficiency. Thus, we introduce OckBench, the first benchmark that jointly measures accuracy and token efficiency across reasoning and coding tasks. Our evaluation reveals that token efficiency remains largely unoptimized across current models, significantly inflating serving costs and latency. These findings provide a concrete roadmap for the community to optimize the latent reasoning ability, token efficiency. Ultimately, we argue for an evaluation paradigm shift: tokens must not be multiplied beyond necessity. Our benchmarks are available at this https URL.
>
---
#### [replaced 097] AutoForest: Automatically Generating Forest Plots from Biomedical Studies with End-to-End Evidence Extraction and Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AutoForest，解决自动生成森林图的问题，通过端到端提取和合成证据，简化系统综述流程。**

- **链接: [https://arxiv.org/pdf/2606.02403](https://arxiv.org/pdf/2606.02403)**

> **作者:** Massimiliano Pronesti; Angelo Miculescu; Mohsin Kapdi; Paul Flanagan; Oisín Redmond; Joao Bettencourt-Silva; Gurdeep Mannu; Spiros Denaxas; Rui Bebiano Da Providencia E Costa; Anya Belz; Yufang Hou
>
> **备注:** Accepted to ACL2026 (System Demonstrations Track)
>
> **摘要:** Systematic reviews rely on forest plots to synthesise quantitative evidence across biomedical studies, but generating them remains a fragmented and labour-intensive process. Researchers must interpret complex clinical texts, manually extract outcome data from trials, define appropriate interventions and comparators, harmonise inconsistent study designs, and carry out meta-analytic computations-typically using specialised software that demands structured inputs and domain expertise. While recent work has demonstrated that large language models can extract study-level data from unstructured text, no existing system automates the complete pipeline from raw documents to synthesised forest plots. To address this gap, we introduce AutoForest, the first end-to-end system that generates publication-ready forest plots directly from biomedical papers. Given one or more study papers, AutoForest automatically suggests ICO (Intervention, Comparator, Outcome) elements, extracts outcome data, performs statistical synthesis, and renders the final forest plot. We describe the system architecture, user interface and demonstrate its effectiveness on real-world examples through a user study involving clinicians, showing how AutoForest can accelerate evidence synthesis and substantially lower the barrier to conducting meta-analyses.
>
---
#### [replaced 098] Extending AI for Research to the Humanities: A Multi-Agent Framework for Evidence-Grounded Scholarship
- **分类: cs.CL**

- **简介: 该论文属于人文领域AI研究任务，旨在解决传统AI在证据支撑的学术推理中的不足。提出SPIRE框架，通过多智能体协作实现文献分析与论证。**

- **链接: [https://arxiv.org/pdf/2605.30947](https://arxiv.org/pdf/2605.30947)**

> **作者:** Yating Pan; Jiajun Zhang; Jun Wang; Qi Su
>
> **备注:** 28 pages, 3 figures. Code, data catalogues, and reproduction scripts: this https URL. Lead corresponding author: Jun Wang; corresponding author: Qi Su
>
> **摘要:** LLM-based research agents have advanced rapidly in science and engineering, where research is organized around executable experiments, code, and quantitative signals. Humanities scholarship, however, requires a different mode of reasoning: interpretive, evidence-grounded argument over primary sources, where scholarly value depends on faithful quotation, verifiable provenance, and close reading. Existing research agents remain largely optimized for execution and retrieval, not evidence-grounded interpretive reasoning. To address this gap, we introduce SPIRE (Scholarly-Primitives-Inspired Research Engine), a multi-agent framework for evidence-grounded humanities scholarship. Drawing on Scholarly Primitives theory, SPIRE casts recurring humanities operations as cooperating agent roles (source discovery, evidence annotation, comparison, provenance checking, sampling, citation binding, and argumentative synthesis) over a multi-scale close-reading substrate of passages, intra-context graph communities, and cross-context semantic clusters. On a peer-reviewed-paper benchmark over classical Chinese and Greco-Roman Latin scholarship, SPIRE recovers cited primary-source evidence more reliably than Naive LLM, Text RAG, and GraphRAG, and receives higher blind-judge scores on answer accuracy, depth, coverage, and evidence quality. Ablations show that both the scholarly-operation agents and close-reading retrieval contribute to evidence-grounded essays. Code, data catalogues, and reproduction scripts are released at this https URL.
>
---
