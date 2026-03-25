# 自然语言处理 cs.CL

- **最新发布 81 篇**

- **更新 31 篇**

## 最新发布

#### [new 001] Lie to Me: How Faithful Is Chain-of-Thought Reasoning in Reasoning Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型透明性研究，探讨链式推理（CoT）的可信度问题。通过测试多个模型对不同提示的响应，分析其推理过程的忠实性，揭示模型内部认知与输出之间的差异。**

- **链接: [https://arxiv.org/pdf/2603.22582](https://arxiv.org/pdf/2603.22582)**

> **作者:** Richard J. Young
>
> **备注:** 27 pages, 7 figures, 12 tables
>
> **摘要:** Chain-of-thought (CoT) reasoning has been proposed as a transparency mechanism for large language models in safety-critical deployments, yet its effectiveness depends on faithfulness (whether models accurately verbalize the factors that actually influence their outputs), a property that prior evaluations have examined in only two proprietary models, finding acknowledgment rates as low as 25% for Claude 3.7 Sonnet and 39% for DeepSeek-R1. To extend this evaluation across the open-weight ecosystem, this study tests 12 open-weight reasoning models spanning 9 architectural families (7B-685B parameters) on 498 multiple-choice questions from MMLU and GPQA Diamond, injecting six categories of reasoning hints (sycophancy, consistency, visual pattern, metadata, grader hacking, and unethical information) and measuring the rate at which models acknowledge hint influence in their CoT when hints successfully alter answers. Across 41,832 inference runs, overall faithfulness rates range from 39.7% (Seed-1.6-Flash) to 89.9% (DeepSeek-V3.2-Speciale) across model families, with consistency hints (35.5%) and sycophancy hints (53.9%) exhibiting the lowest acknowledgment rates. Training methodology and model family predict faithfulness more strongly than parameter count, and keyword-based analysis reveals a striking gap between thinking-token acknowledgment (approximately 87.5%) and answer-text acknowledgment (approximately 28.6%), suggesting that models internally recognize hint influence but systematically suppress this acknowledgment in their outputs. These findings carry direct implications for the viability of CoT monitoring as a safety mechanism and suggest that faithfulness is not a fixed property of reasoning models but varies systematically with architecture, training method, and the nature of the influencing cue.
>
---
#### [new 002] HGNet: Scalable Foundation Model for Automated Knowledge Graph Generation from Scientific Literature
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于科学文献的知识图谱生成任务，旨在解决实体识别和关系抽取中的多词实体识别、跨领域泛化及层次结构建模问题。提出两阶段框架HGNet，提升KG质量与一致性。**

- **链接: [https://arxiv.org/pdf/2603.23136](https://arxiv.org/pdf/2603.23136)**

> **作者:** Devvrat Joshi; Islem Rekik
>
> **摘要:** Automated knowledge graph (KG) construction is essential for navigating the rapidly expanding body of scientific literature. However, existing approaches struggle to recognize long multi-word entities, often fail to generalize across domains, and typically overlook the hierarchical nature of scientific knowledge. While general-purpose large language models (LLMs) offer adaptability, they are computationally expensive and yield inconsistent accuracy on specialized tasks. As a result, current KGs are shallow and inconsistent, limiting their utility for exploration and synthesis. We propose a two-stage framework for scalable, zero-shot scientific KG construction. The first stage, Z-NERD, introduces (i) Orthogonal Semantic Decomposition (OSD), which promotes domain-agnostic entity recognition by isolating semantic "turns" in text, and (ii) a Multi-Scale TCQK attention mechanism that captures coherent multi-word entities through n-gram-aware attention heads. The second stage, HGNet, performs relation extraction with hierarchy-aware message passing, explicitly modeling parent, child, and peer relations. To enforce global consistency, we introduce two complementary objectives: a Differentiable Hierarchy Loss to discourage cycles and shortcut edges, and a Continuum Abstraction Field (CAF) Loss that embeds abstraction levels along a learnable axis in Euclidean space. This is the first approach to formalize hierarchical abstraction as a continuous property within standard Euclidean embeddings, offering a simpler alternative to hyperbolic methods. We release SPHERE (this https URL), a multi-domain benchmark for hierarchical relation extraction. Our framework establishes a new state of the art on SciERC, SciER, and SPHERE, improving NER by 8.08% and RE by 5.99% on out-of-distribution tests. In zero-shot settings, gains reach 10.76% for NER and 26.2% for RE.
>
---
#### [new 003] Quality Over Clicks: Intrinsic Quality-Driven Iterative Reinforcement Learning for Cold-Start E-Commerce Query Suggestion
- **分类: cs.CL**

- **简介: 该论文属于冷启动场景下的电商查询建议任务，旨在解决传统方法依赖点击数据导致的冷启动问题。通过强化学习框架优化建议查询质量，提升效果。**

- **链接: [https://arxiv.org/pdf/2603.22922](https://arxiv.org/pdf/2603.22922)**

> **作者:** Qi Sun; Kejun Xiao; Huaipeng Zhao; Tao Luo; Xiaoyi Zeng
>
> **备注:** Submitted to ACL 2026 Industry Track
>
> **摘要:** Existing dialogue systems rely on Query Suggestion (QS) to enhance user engagement. Recent efforts typically employ large language models with Click-Through Rate (CTR) model, yet fail in cold-start scenarios due to their heavy reliance on abundant online click data for effective CTR model training. To bridge this gap, we propose Cold-EQS, an iterative reinforcement learning framework for Cold-Start E-commerce Query Suggestion (EQS). Specifically, we leverage answerability, factuality, and information gain as reward to continuously optimize the quality of suggested queries. To continuously optimize our QS model, we estimate uncertainty for grouped candidate suggested queries to select hard and ambiguous samples from online user queries lacking click signals. In addition, we provide an EQS-Benchmark comprising 16,949 online user queries for offline training and evaluation. Extensive offline and online experiments consistently demonstrate a strong positive correlation between online and offline effectiveness. Both offline and online experimental results demonstrate the superiority of our Cold-EQS, achieving a significant +6.81% improvement in online chatUV.
>
---
#### [new 004] Knowledge Access Beats Model Size: Memory Augmented Routing for Persistent AI Agents
- **分类: cs.CL**

- **简介: 该论文属于对话系统任务，旨在解决重复用户查询的高效处理问题。通过引入记忆增强机制，提升模型效率与准确性，证明记忆比模型规模更关键。**

- **链接: [https://arxiv.org/pdf/2603.23013](https://arxiv.org/pdf/2603.23013)**

> **作者:** Xunzhuo Liu; Bowei He; Xue Liu; Andy Luo; Haichen Zhang; Huamin Chen
>
> **摘要:** Production AI agents frequently receive user-specific queries that are highly repetitive, with up to 47\% being semantically similar to prior interactions, yet each query is typically processed with the same computational cost. We argue that this redundancy can be exploited through conversational memory, transforming repetition from a cost burden into an efficiency advantage. We propose a memory-augmented inference framework in which a lightweight 8B-parameter model leverages retrieved conversational context to answer all queries via a low-cost inference path. Without any additional training or labeled data, this approach achieves 30.5\% F1, recovering 69\% of the performance of a full-context 235B model while reducing effective cost by 96\%. Notably, a 235B model without memory (13.7\% F1) underperforms even the standalone 8B model (15.4\% F1), indicating that for user-specific queries, access to relevant knowledge outweighs model scale. We further analyze the role of routing and confidence. At practical confidence thresholds, routing alone already directs 96\% of queries to the small model, but yields poor accuracy (13.0\% F1) due to confident hallucinations. Memory does not substantially alter routing decisions; instead, it improves correctness by grounding responses in retrieved user-specific information. As conversational memory accumulates over time, coverage of recurring topics increases, further narrowing the performance gap. We evaluate on 152 LoCoMo questions (Qwen3-8B/235B) and 500 LongMemEval questions. Incorporating hybrid retrieval (BM25 + cosine similarity) improves performance by an additional +7.7 F1, demonstrating that retrieval quality directly enhances end-to-end system performance. Overall, our results highlight that memory, rather than model size, is the primary driver of accuracy and efficiency in persistent AI agents.
>
---
#### [new 005] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对搜索增强型大语言模型的训练问题，提出TIPS框架，通过分 turn 的奖励 shaping 提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2603.22293](https://arxiv.org/pdf/2603.22293)**

> **作者:** Yutao Xie; Nathaniel Thomas; Nicklas Hansen; Yang Fu; Li Erran Li; Xiaolong Wang
>
> **备注:** Code: this https URL
>
> **摘要:** Search-augmented large language models (LLMs) trained with reinforcement learning (RL) have achieved strong results on open-domain question answering (QA), but training still remains a significant challenge. The optimization is often unstable due to sparse rewards and difficult credit assignments across reasoning and tool calls. To address this, we introduce Turn-Level Information Potential Reward Shaping (TIPS), a simple framework that assigns dense, turn-level rewards to each reasoning + tool-call segment based on the increased likelihood of the correct answer under a teacher model. By leveraging the potential-based reward shaping, TIPS offers fine-grained and policy-invariant guidance that overcomes the limitations of outcome-only optimization. Evaluated on seven QA benchmarks, TIPS consistently outperforms GRPO/PPO baselines and substantially improves training stability. For instance, with a Qwen-2.5 7B Instruct model, TIPS improves the average Exact Match score by 11.8% and F1 by 13.6% relative to PPO. Our results demonstrate that turn-level information-potential reward shaping provides an effective and general solution to sparse-reward credit assignment for multi-turn LLM reasoning.
>
---
#### [new 006] Analysing LLM Persona Generation and Fairness Interpretation in Polarised Geopolitical Contexts
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性分析任务，旨在研究LLM在极化地缘政治背景下生成人物角色的偏见问题，通过实验分析不同情境下的生成差异。**

- **链接: [https://arxiv.org/pdf/2603.22837](https://arxiv.org/pdf/2603.22837)**

> **作者:** Maida Aizaz; Quang Minh Nguyen
>
> **备注:** EACL 2026 Student Research Workshop
>
> **摘要:** Large language models (LLMs) are increasingly utilised for social simulation and persona generation, necessitating an understanding of how they represent geopolitical identities. In this paper, we analyse personas generated for Palestinian and Israeli identities by five popular LLMs across 640 experimental conditions, varying context (war vs non-war) and assigned roles. We observe significant distributional patterns in the generated attributes: Palestinian profiles in war contexts are frequently associated with lower socioeconomic status and survival-oriented roles, whereas Israeli profiles predominantly retain middle-class status and specialised professional attributes. When prompted with explicit instructions to avoid harmful assumptions, models exhibit diverse distributional changes, e.g., marked increases in non-binary gender inferences or a convergence toward generic occupational roles (e.g., "student"), while the underlying socioeconomic distinctions often remain. Furthermore, analysis of reasoning traces reveals an interesting dynamics between model reasoning and generation: while rationales consistently mention fairness-related concepts, the final generated personas follow the aforementioned diverse distributional changes. These findings illustrate a picture of how models interpret geopolitical contexts, while suggesting that they process fairness and adjust in varied ways; there is no consistent, direct translation of fairness concepts into representative outcomes.
>
---
#### [new 007] KALAVAI: Predicting When Independent Specialist Fusion Works -- A Quantitative Model for Post-Hoc Cooperative LLM Training
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出KALAVAI协议，解决独立训练模型融合问题，通过量化模型预测融合效果，提升整体性能。**

- **链接: [https://arxiv.org/pdf/2603.22755](https://arxiv.org/pdf/2603.22755)**

> **作者:** Ramchand Kumaresan
>
> **摘要:** Independently trained domain specialists can be fused post-hoc into a single model that outperforms any individual specialist, and the gain is predictable: gain = 0.82 x divergence - 2.72 (R^2 = 0.856, n=6, 3-26% divergence). This enables practitioners to estimate cooperative value before committing compute. Below ~3.3% divergence, gains approach this http URL the KALAVAI protocol, contributors fine-tune copies of a shared checkpoint independently, then submit for lightweight MoE routing (500 steps). Gains are consistent: +7.72% at 410M (+/-0.02%, 3 seeds), +7.49% at 1B (+/-0.01%, 3 seeds), +6.53% at 6.9B, each over the best specialist. The router matches domain-oracle routing within <10^{-5} nats. Cross-lingual fusion (Tamil/Yoruba/Welsh/Code) achieves +21.76%, with Yoruba perplexity falling 41.9 to 7.7. A 20-contributor federation achieves +16.71% (+/-0.07pp, 3 seeds).Three requirements bound the protocol. Shared initialisation is necessary: checkpoint mismatch degrades routing. Frozen layers are optional below ~10,000 steps and beneficial beyond. Learned routing is essential: uniform averaging degrades by -1.2% vs. best specialist, while any trained router achieves oracle-optimal assignment.
>
---
#### [new 008] Set-Valued Prediction for Large Language Models with Feasibility-Aware Coverage Guarantees
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言生成任务，旨在解决LLM点预测不够全面的问题。通过提出集合预测框架，确保预测集包含正确答案的概率，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2603.22966](https://arxiv.org/pdf/2603.22966)**

> **作者:** Ye Li; Anqi Hu; Yuanchang Ye; Shiyan Tong; Zhiyuan Wang; Bo Fu
>
> **摘要:** Large language models (LLMs) inherently operate over a large generation space, yet conventional usage typically reports the most likely generation (MLG) as a point prediction, which underestimates the model's capability: although the top-ranked response can be incorrect, valid answers may still exist within the broader output space and can potentially be discovered through repeated sampling. This observation motivates moving from point prediction to set-valued prediction, where the model produces a set of candidate responses rather than a single MLG. In this paper, we propose a principled framework for set-valued prediction, which provides feasibility-aware coverage guarantees. We show that, given the finite-sampling nature of LLM generation, coverage is not always achievable: even with multiple samplings, LLMs may fail to yield an acceptable response for certain questions within the sampled candidate set. To address this, we establish a minimum achievable risk level (MRL), below which statistical coverage guarantees cannot be satisfied. Building on this insight, we then develop a data-driven calibration procedure that constructs prediction sets from sampled responses by estimating a rigorous threshold, ensuring that the resulting set contains a correct answer with a desired probability whenever the target risk level is feasible. Extensive experiments on six language generation tasks with five LLMs demonstrate both the statistical validity and the predictive efficiency of our framework.
>
---
#### [new 009] LLM-guided headline rewriting for clickability enhancement without clickbait
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于新闻标题生成任务，旨在提升点击率同时避免夸大其词。通过控制语言属性，在保持语义准确的前提下优化标题吸引力。**

- **链接: [https://arxiv.org/pdf/2603.22459](https://arxiv.org/pdf/2603.22459)**

> **作者:** Yehudit Aperstein; Linoy Halifa; Sagiv Bar; Alexander Apartsin
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Enhancing reader engagement while preserving informational fidelity is a central challenge in controllable text generation for news media. Optimizing news headlines for reader engagement is often conflated with clickbait, resulting in exaggerated or misleading phrasing that undermines editorial trust. We frame clickbait not as a separate stylistic category, but as an extreme outcome of disproportionate amplification of otherwise legitimate engagement cues. Based on this view, we formulate headline rewriting as a controllable generation problem, where specific engagement-oriented linguistic attributes are selectively strengthened under explicit constraints on semantic faithfulness and proportional emphasis. We present a guided headline rewriting framework built on a large language model (LLM) that uses the Future Discriminators for Generation (FUDGE) paradigm for inference-time control. The LLM is steered by two auxiliary guide models: (1) a clickbait scoring model that provides negative guidance to suppress excessive stylistic amplification, and (2) an engagement-attribute model that provides positive guidance aligned with target clickability objectives. Both guides are trained on neutral headlines drawn from a curated real-world news corpus. At the same time, clickbait variants are generated synthetically by rewriting these original headlines using an LLM under controlled activation of predefined engagement tactics. By adjusting guidance weights at inference time, the system generates headlines along a continuum from neutral paraphrases to more engaging yet editorially acceptable formulations. The proposed framework provides a principled approach for studying the trade-off between attractiveness, semantic preservation, and clickbait avoidance, and supports responsible LLM-based headline optimization in journalistic settings.
>
---
#### [new 010] Why AI-Generated Text Detection Fails: Evidence from Explainable AI Beyond Benchmark Accuracy
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决检测系统在真实场景中可靠性不足的问题。通过融合语言特征与可解释AI，提出一种新框架，并揭示现有方法依赖数据特定特征而非稳定信号。**

- **链接: [https://arxiv.org/pdf/2603.23146](https://arxiv.org/pdf/2603.23146)**

> **作者:** Shushanta Pudasaini; Luis Miralles-Pechuán; David Lillis; Marisa Llorens Salvador
>
> **摘要:** The widespread adoption of Large Language Models (LLMs) has made the detection of AI-Generated text a pressing and complex challenge. Although many detection systems report high benchmark accuracy, their reliability in real-world settings remains uncertain, and their interpretability is often unexplored. In this work, we investigate whether contemporary detectors genuinely identify machine authorship or merely exploit dataset-specific artefacts. We propose an interpretable detection framework that integrates linguistic feature engineering, machine learning, and explainable AI techniques. When evaluated on two prominent benchmark corpora, namely PAN CLEF 2025 and COLING 2025, our model trained on 30 linguistic features achieves leaderboard-competitive performance, attaining an F1 score of 0.9734. However, systematic cross-domain and cross-generator evaluation reveals substantial generalisation failure: classifiers that excel in-domain degrade significantly under distribution shift. Using SHAP- based explanations, we show that the most influential features differ markedly between datasets, indicating that detectors often rely on dataset-specific stylistic cues rather than stable signals of machine authorship. Further investigation with in-depth error analysis exposes a fundamental tension in linguistic-feature-based AI text detection: the features that are most discriminative on in-domain data are also the features most susceptible to domain shift, formatting variation, and text-length effects. We believe that this knowledge helps build AI detectors that are robust across different settings. To support replication and practical use, we release an open-source Python package that returns both predictions and instance-level explanations for individual texts.
>
---
#### [new 011] Parametric Knowledge and Retrieval Behavior in RAG Fine-Tuning for Electronic Design Automation
- **分类: cs.CL; cs.AI; cs.CE**

- **简介: 该论文研究RAG微调在电子设计自动化中的长文本生成任务，解决现有评估指标不足的问题，提出新评估方法并验证模型效果。**

- **链接: [https://arxiv.org/pdf/2603.23047](https://arxiv.org/pdf/2603.23047)**

> **作者:** Julian Oestreich; Maximilian Bley; Frank Binder; Lydia Müller; Maksym Sydorenko; André Alcalde
>
> **摘要:** Retrieval-Augmented Generation (RAG) fine-tuning has shown substantial improvements over vanilla RAG, yet most studies target document question answering and often rely on standard NLP metrics that can obscure factual differences. We evaluate RAG fine-tuning for long-form text generation in electronic design automation, adapting a 7B model under five context augmentation strategies with varying retrieval conditions. We introduce TriFEX, a human-validated, triple-based evaluation pipeline that attributes generated claims to their origin-user query, context and reference-and propose Parametric Knowledge Precision (PKP), which isolates internalized knowledge by filtering out claims leaked in the prompt. We show that ROUGE and BERTScore fail to detect factual differences that our triple-based evaluation reveals. Additionally, we demonstrate that an existing metric for knowledge internalization is retrieva-sensitive, with about 75% of its cross-condition variance driven by changes in the rate at which internal knowledge is expressed (PR), rather than by changes in its actual correctness (PKP). The fine-tuned 7B variants outperform a 72B baseline on most metrics, further showing generalization across conditions and on a related benchmark. These results underscore the limitations of available metrics in RAG evaluation and show that smaller models could be reasonably well adapted to specialized tasks for cost-efficient, on-premises deployment.
>
---
#### [new 012] Evaluating Prompting Strategies for Chart Question Answering with Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于图表问答任务，研究不同提示策略对大语言模型推理性能的影响。通过系统评估四种提示方法，发现Few-Shot Chain-of-Thought在准确率上表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.22288](https://arxiv.org/pdf/2603.22288)**

> **作者:** Ruthuparna Naikar; Ying Zhu
>
> **摘要:** Prompting strategies affect LLM reasoning performance, but their role in chart-based QA remains underexplored. We present a systematic evaluation of four widely used prompting paradigms (Zero-Shot, Few-Shot, Zero-Shot Chain-of-Thought, and Few-Shot Chain-of-Thought) across GPT-3.5, GPT-4, and GPT-4o on the ChartQA dataset. Our framework operates exclusively on structured chart data, isolating prompt structure as the only experimental variable, and evaluates performance using two metrics: Accuracy and Exact Match. Results from 1,200 diverse ChartQA samples show that Few-Shot Chain-of-Thought prompting consistently yields the highest accuracy (up to 78.2\%), particularly on reasoning-intensive questions, while Few-Shot prompting improves format adherence. Zero-Shot performs well only with high-capacity models on simpler tasks. These findings provide actionable guidance for selecting prompting strategies in structured data reasoning tasks, with implications for both efficiency and accuracy in real-world applications.
>
---
#### [new 013] CAPITU: A Benchmark for Evaluating Instruction-Following in Brazilian Portuguese with Literary Context
- **分类: cs.CL**

- **简介: 该论文提出CAPITU基准，用于评估大语言模型在巴西葡萄牙语文学背景下的指令遵循能力。解决的是非英语环境下指令遵循的评测问题，通过文学场景设计任务进行模型测试与分析。**

- **链接: [https://arxiv.org/pdf/2603.22576](https://arxiv.org/pdf/2603.22576)**

> **作者:** Giovana Kerche Bonás; Roseval Malaquias Junior; Marcos Piau; Thiago Laitz; Thales Sales Almeida; Hugo Abonizio; Celio Larcher; Ramon Pires; Rodrigo Nogueira
>
> **摘要:** We introduce CAPITU, a benchmark for evaluating instruction-following capabilities of Large Language Models (LLMs) in Brazilian Portuguese. Unlike existing benchmarks that focus on English or use generic prompts, CAPITU contextualizes all tasks within eight canonical works of Brazilian literature, combining verifiable instruction constraints with culturally-grounded content. The benchmark comprises 59 instruction types organized into seven categories, all designed to be automatically verifiable without requiring LLM judges or human evaluation. Instruction types include Portuguese-specific linguistic constraints (word termination patterns like -ando/-endo/-indo, -inho/-inha, -mente) and structural requirements. We evaluate 18 state-of-the-art models across single-turn and multi-turn settings. Our results show that frontier reasoning models achieve strong performance (GPT-5.2 with reasoning: 98.5% strict accuracy), while Portuguese-specialized models offer competitive cost-efficiency (Sabiazinho-4: 87.0% at \$0.13 vs Claude-Haiku-4.5: 73.5% at \$1.12). Multi-turn evaluation reveals significant variation in constraint persistence, with conversation-level accuracy ranging from 60% to 96% across models. We identify specific challenges in morphological constraints, exact counting, and constraint persistence degradation across turns. We release the complete benchmark, evaluation code, and baseline results to facilitate research on instruction-following in Portuguese.
>
---
#### [new 014] Detecting Non-Membership in LLM Training Data via Rank Correlations
- **分类: cs.CL**

- **简介: 该论文属于模型训练数据验证任务，旨在解决如何验证某数据集未被用于训练。通过分析模型日志概率的等级相关性，提出PRISM方法检测非成员性。**

- **链接: [https://arxiv.org/pdf/2603.22707](https://arxiv.org/pdf/2603.22707)**

> **作者:** Pranav Shetty; Mirazul Haque; Zhiqiang Ma; Xiaomo Liu
>
> **备注:** Accepted to EACL 2026 Main Conference
>
> **摘要:** As large language models (LLMs) are trained on increasingly vast and opaque text corpora, determining which data contributed to training has become essential for copyright enforcement, compliance auditing, and user trust. While prior work focuses on detecting whether a dataset was used in training (membership inference), the complementary problem -- verifying that a dataset was not used -- has received little attention. We address this gap by introducing PRISM, a test that detects dataset-level non-membership using only grey-box access to model logits. Our key insight is that two models that have not seen a dataset exhibit higher rank correlation in their normalized token log probabilities than when one model has been trained on that data. Using this observation, we construct a correlation-based test that detects non-membership. Empirically, PRISM reliably rules out membership in training data across all datasets tested while avoiding false positives, thus offering a framework for verifying that specific datasets were excluded from LLM training.
>
---
#### [new 015] ImplicitRM: Unbiased Reward Modeling from Implicit Preference Data for LLM alignment
- **分类: cs.CL; cs.AI; stat.AP**

- **简介: 该论文属于语言模型对齐任务，旨在解决从隐式反馈数据中学习无偏奖励模型的问题。提出ImplicitRM方法，通过分层建模和似然最大化，有效处理隐式数据的挑战。**

- **链接: [https://arxiv.org/pdf/2603.23184](https://arxiv.org/pdf/2603.23184)**

> **作者:** Hao Wang; Haocheng Yang; Licheng Pan; Lei Shen; Xiaoxi Li; Yinuo Wang; Zhichao Chen; Yuan Lu; Haoxuan Li; Zhouchen Lin
>
> **摘要:** Reward modeling represents a long-standing challenge in reinforcement learning from human feedback (RLHF) for aligning language models. Current reward modeling is heavily contingent upon experimental feedback data with high collection costs. In this work, we study \textit{implicit reward modeling} -- learning reward models from implicit human feedback (e.g., clicks and copies) -- as a cost-effective alternative. We identify two fundamental challenges in implicit reward modeling: (1) Implicit preference data lacks definitive negative samples, which makes standard positive-negative classification methods inapplicable; (2) Implicit preference data suffers from user preference bias, where different responses have different propensities to elicit user feedback actions, which exacerbates the difficulty of distinguishing definitive negative samples. To address these challenges, we propose ImplicitRM, which aims to learn unbiased reward models from implicit preference data. ImplicitRM stratifies training samples into four latent groups via a stratification model. Building on this, it derives a learning objective through likelihood maximization, which we prove is theoretically unbiased, effectively resolving both challenges. Experiments demonstrate that ImplicitRM learns accurate reward models across implicit preference datasets. Code is available on our project website.
>
---
#### [new 016] Failure of contextual invariance in gender inference with large language models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，研究大语言模型在性别推断中的上下文不稳定性。通过实验发现模型输出受无关上下文影响，违反上下文不变性假设，对模型偏差评估有重要影响。**

- **链接: [https://arxiv.org/pdf/2603.23485](https://arxiv.org/pdf/2603.23485)**

> **作者:** Sagar Kumar; Ariel Flint; Luca Maria Aiello; Andrea Baronchelli
>
> **摘要:** Standard evaluation practices assume that large language model (LLM) outputs are stable under contextually equivalent formulations of a task. Here, we test this assumption in the setting of gender inference. Using a controlled pronoun selection task, we introduce minimal, theoretically uninformative discourse context and find that this induces large, systematic shifts in model outputs. Correlations with cultural gender stereotypes, present in decontextualized settings, weaken or disappear once context is introduced, while theoretically irrelevant features, such as the gender of a pronoun for an unrelated referent, become the most informative predictors of model behaviour. A Contextuality-by-Default analysis reveals that, in 19--52\% of cases across models, this dependence persists after accounting for all marginal effects of context on individual outputs and cannot be attributed to simple pronoun repetition. These findings show that LLM outputs violate contextual invariance even under near-identical syntactic formulations, with implications for bias benchmarking and deployment in high-stakes settings.
>
---
#### [new 017] Less is More: Adapting Text Embeddings for Low-Resource Languages with Small Scale Noisy Synthetic Data
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，针对低资源语言缺乏高质量数据的问题，通过小规模噪声合成数据提升文本嵌入效果，验证了“少即是多”的现象。**

- **链接: [https://arxiv.org/pdf/2603.22290](https://arxiv.org/pdf/2603.22290)**

> **作者:** Zaruhi Navasardyan; Spartak Bughdaryan; Bagrat Minasyan; Hrant Davtyan
>
> **备注:** Accepted at LoResLM 2026, EACL 2026 Workshop
>
> **摘要:** Low-resource languages (LRLs) often lack high-quality, large-scale datasets for training effective text embedding models, hindering their application in tasks like retrieval-augmented generation (RAG) and semantic search. In this work, we challenge the prevailing assumption that effective semantic alignment requires massive datasets or pristine, human-verified translations. Focusing on Armenian (an LRL with a unique script), we introduce a cost-effective adaptation strategy using small scale noisy synthetic data generated by translating English Reddit title-body pairs with open-weights models. We establish a comprehensive evaluation benchmark comprising existing datasets, translated data, and a manually curated dataset. Our experiments reveal a surprising "Less is More" phenomenon: fine-tuning a multilingual encoder (mE5) on just 10,000 noisy synthetic pairs yields 11-12\% average improvements across the benchmark with a 20\%+ relative improvement in retrieval performance, matching the performance of models trained on ~1 million examples. Furthermore, we demonstrate that neither increasing data scale, improving translation quality via state-of-the-art LLMs, nor diversifying data domains yields significant gains over this minimal baseline. We validate the generalizability of these findings on another LRL with a unique script. Our results suggest that semantic alignment for LRLs saturates early and is highly robust to noise, democratizing high-performance embedding creation for resource-constrained communities. We release the model, data, and the benchmark at this https URL to facilitate further research.
>
---
#### [new 018] Efficient Hallucination Detection: Adaptive Bayesian Estimation of Semantic Entropy with Guided Semantic Exploration
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的幻觉检测任务，旨在提高检测效率。通过自适应贝叶斯估计和引导语义探索，动态调整采样策略，提升检测性能并减少样本需求。**

- **链接: [https://arxiv.org/pdf/2603.22812](https://arxiv.org/pdf/2603.22812)**

> **作者:** Qiyao Sun; Xingming Li; Xixiang He; Ao Cheng; Xuanyu Ji; Hailun Lu; Runke Huang; Qingyong Hu
>
> **备注:** Accepted to a AAAI 2026 (Oral Presentation, <5% acceptance rate), Project page: this https URL
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in various natural language processing tasks, yet they remain prone to generating factually incorrect outputs known as hallucinations. While recent approaches have shown promise for hallucination detection by repeatedly sampling from LLMs and quantifying the semantic inconsistency among the generated responses, they rely on fixed sampling budgets that fail to adapt to query complexity, resulting in computational inefficiency. We propose an Adaptive Bayesian Estimation framework for Semantic Entropy with Guided Semantic Exploration, which dynamically adjusts sampling requirements based on observed uncertainty. Our approach employs a hierarchical Bayesian framework to model the semantic distribution, enabling dynamic control of sampling iterations through variance-based thresholds that terminate generation once sufficient certainty is achieved. We also develop a perturbation-based importance sampling strategy to systematically explore the semantic space. Extensive experiments on four QA datasets demonstrate that our method achieves superior hallucination detection performance with significant efficiency gains. In low-budget scenarios, our approach requires about 50% fewer samples to achieve comparable detection performance to existing methods, while delivers an average AUROC improvement of 12.6% under the same sampling budget.
>
---
#### [new 019] Reddit After Roe: A Computational Analysis of Abortion Narratives and Barriers in the Wake of Dobbs
- **分类: cs.CL**

- **简介: 该论文属于社会计算任务，旨在分析Dobbs判决后Reddit上关于堕胎的叙事与障碍。通过分类和主题建模，研究在线社区中信息行为、情绪与障碍的关系。**

- **链接: [https://arxiv.org/pdf/2603.22566](https://arxiv.org/pdf/2603.22566)**

> **作者:** Aria Pessianzadeh; Alex H. Poole; Rezvaneh Rezapour
>
> **摘要:** The 2022 U.S. Supreme Court decision in Dobbs v. Jackson Women's Health Organization reshaped the reproductive rights landscape, introducing new uncertainty and barriers to abortion access. We present a large-scale computational analysis of abortion discourse on Reddit, examining how barriers to access are articulated across information-seeking and information-sharing behaviors, different stages of abortion (before, during, after), and three phases of the Dobbs decision in 2022. Drawing on more than 17,000 posts from four abortion-related subreddits, we employed a multi-step pipeline to classify posts by information type, abortion stage, barrier category, and expressed emotions. Using a codebook of eight barrier types, including legal, financial, emotional, and social obstacles, we analyzed their associations with emotions and information behaviors. Topic modeling of model-generated barrier rationales further revealed how discourse evolved in response to shifting legal and cultural contexts. Our findings show that emotional and psychological barriers consistently dominate abortion narratives online, with emotions such as nervousness, confusion, fear, and sadness prevalent across discourse. By linking information behaviors, barriers, emotions, and temporal dynamics, this study provides a multi-dimensional account of how abortion is navigated in online communities.
>
---
#### [new 020] Beyond Hate: Differentiating Uncivil and Intolerant Speech in Multimodal Content Moderation
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于内容 moderation 任务，旨在解决单一标签导致的表达特征混淆问题。通过引入细粒度标注方案，区分不文明和不宽容言论，提升模型性能与准确性。**

- **链接: [https://arxiv.org/pdf/2603.22985](https://arxiv.org/pdf/2603.22985)**

> **作者:** Nils A. Herrmann; Tobias Eder; Jingyi He; Georg Groh
>
> **备注:** Preprint. Under review
>
> **摘要:** Current multimodal toxicity benchmarks typically use a single binary hatefulness label. This coarse approach conflates two fundamentally different characteristics of expression: tone and content. Drawing on communication science theory, we introduce a fine-grained annotation scheme that distinguishes two separable dimensions: incivility (rude or dismissive tone) and intolerance (content that attacks pluralism and targets groups or identities) and apply it to 2,030 memes from the Hateful Memes dataset. We evaluate different vision-language models under coarse-label training, transfer learning across label schemes and a joint learning approach that combines the coarse hatefulness label with our fine-grained annotations. Our results show that fine-grained annotations complement existing coarse labels and, when used jointly, improve overall model performance. Moreover, models trained with the fine-grained scheme exhibit more balanced moderation-relevant error profiles and are less prone to under-detection of harmful content than models trained on hatefulness labels alone (FNR-FPR, the difference between false negative and false positive rates: 0.74 to 0.42 for LLaVA-1.6-Mistral-7B; 0.54 to 0.28 for Qwen2.5-VL-7B). This work contributes to data-centric approaches in content moderation by improving the reliability and accuracy of moderation systems through enhanced data quality. Overall, combining both coarse and fine-grained labels provides a practical route to more reliable multimodal moderation.
>
---
#### [new 021] LGSE: Lexically Grounded Subword Embedding Initialization for Low-Resource Language Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言适配问题。通过引入基于形态的子词嵌入初始化方法，提升语言模型在形态丰富语言中的表示质量。**

- **链接: [https://arxiv.org/pdf/2603.22629](https://arxiv.org/pdf/2603.22629)**

> **作者:** Hailay Teklehaymanot; Dren Fazlija; Wolfgang Nejdl
>
> **备注:** 12 pages, 1 figure, 1 Table
>
> **摘要:** Adapting pretrained language models to low-resource, morphologically rich languages remains a significant challenge. Existing vocabulary expansion methods typically rely on arbitrarily segmented subword units, resulting in fragmented lexical representations and loss of critical morphological information. To address this limitation, we propose the Lexically Grounded Subword Embedding Initialization (LGSE) framework, which introduces morphologically informed segmentation for initializing embeddings of novel tokens. Instead of using random vectors or arbitrary subwords, LGSE decomposes words into their constituent morphemes and constructs semantically coherent embeddings by averaging pretrained subword or FastText-based morpheme representations. When a token cannot be segmented into meaningful morphemes, its embedding is constructed using character n-gram representations to capture structural information. During Language-Adaptive Pretraining, we apply a regularization term that penalizes large deviations of newly introduced embeddings from their initialized values, preserving alignment with the original pretrained embedding space while enabling adaptation to the target language. To isolate the effect of initialization, we retain the original pre-trained model vocabulary and tokenizer and update only the new embeddings during adaptation. We evaluate LGSE on three NLP tasks: Question Answering, Named Entity Recognition, and Text Classification, in two morphologically rich, low-resource languages: Amharic and Tigrinya, where morphological segmentation resources are available. Experimental results show that LGSE consistently outperforms baseline methods across all tasks, demonstrating the effectiveness of morphologically grounded embedding initialization for improving representation quality in underrepresented languages. Project resources are available in the GitHub link.
>
---
#### [new 022] MERIT: Memory-Enhanced Retrieval for Interpretable Knowledge Tracing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识追踪任务，旨在提升模型的可解释性与效率。提出MERIT框架，结合冻结LLM与结构化记忆，无需参数更新即可实现高效、透明的教育诊断。**

- **链接: [https://arxiv.org/pdf/2603.22289](https://arxiv.org/pdf/2603.22289)**

> **作者:** Runze Li; Kedi Chen; Guwei Feng; Mo Yu; Jun Wang; Wei Zhang
>
> **摘要:** Knowledge Tracing (KT) models students' evolving knowledge states to predict future performance, serving as a foundation for personalized education. While traditional deep learning models achieve high accuracy, they often lack interpretability. Large Language Models (LLMs) offer strong reasoning capabilities but struggle with limited context windows and hallucinations. Furthermore, existing LLM-based methods typically require expensive fine-tuning, limiting scalability and adaptability to new data. We propose MERIT (Memory-Enhanced Retrieval for Interpretable Knowledge Tracing), a training-free framework combining frozen LLM reasoning with structured pedagogical memory. Rather than updating parameters, MERIT transforms raw interaction logs into an interpretable memory bank. The framework uses semantic denoising to categorize students into latent cognitive schemas and constructs a paradigm bank where representative error patterns are analyzed offline to generate explicit Chain-of-Thought (CoT) rationales. During inference, a hierarchical routing mechanism retrieves relevant contexts, while a logic-augmented module applies semantic constraints to calibrate predictions. By grounding the LLM in interpretable memory, MERIT achieves state-of-the-art performance on real-world datasets without gradient updates. This approach reduces computational costs and supports dynamic knowledge updates, improving the accessibility and transparency of educational diagnosis.
>
---
#### [new 023] Multilingual KokoroChat: A Multi-LLM Ensemble Translation Method for Creating a Multilingual Counseling Dialogue Dataset
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决高质量多语言心理咨询对话数据集稀缺的问题。通过多大模型集成方法提升翻译质量，生成中英日三语对话数据集。**

- **链接: [https://arxiv.org/pdf/2603.22913](https://arxiv.org/pdf/2603.22913)**

> **作者:** Ryoma Suzuki; Zhiyang Qi; Michimasa Inaba
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** To address the critical scarcity of high-quality, publicly available counseling dialogue datasets, we created Multilingual KokoroChat by translating KokoroChat, a large-scale manually authored Japanese counseling corpus, into both English and Chinese. A key challenge in this process is that the optimal model for translation varies by input, making it impossible for any single model to consistently guarantee the highest quality. In a sensitive domain like counseling, where the highest possible translation fidelity is essential, relying on a single LLM is therefore insufficient. To overcome this challenge, we developed and employed a novel multi-LLM ensemble method. Our approach first generates diverse hypotheses from multiple distinct LLMs. A single LLM then produces a high-quality translation based on an analysis of the respective strengths and weaknesses of all presented hypotheses. The quality of ``Multilingual KokoroChat'' was rigorously validated through human preference studies. These evaluations confirmed that the translations produced by our ensemble method were preferred from any individual state-of-the-art LLM. This strong preference confirms the superior quality of our method's outputs. The Multilingual KokoroChat is available at this https URL.
>
---
#### [new 024] I Came, I Saw, I Explained: Benchmarking Multimodal LLMs on Figurative Meaning in Memes
- **分类: cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在评估MLLMs在识别和解释网络迷因中的隐喻意义方面的能力，揭示其对视觉与文本信息的整合效果。**

- **链接: [https://arxiv.org/pdf/2603.23229](https://arxiv.org/pdf/2603.23229)**

> **作者:** Shijia Zhou; Saif M. Mohammad; Barbara Plank; Diego Frassinelli
>
> **备注:** LREC 2026, 18 pages, 10 figures
>
> **摘要:** Internet memes represent a popular form of multimodal online communication and often use figurative elements to convey layered meaning through the combination of text and images. However, it remains largely unclear how multimodal large language models (MLLMs) combine and interpret visual and textual information to identify figurative meaning in memes. To address this gap, we evaluate eight state-of-the-art generative MLLMs across three datasets on their ability to detect and explain six types of figurative meaning. In addition, we conduct a human evaluation of the explanations generated by these MLLMs, assessing whether the provided reasoning supports the predicted label and whether it remains faithful to the original meme content. Our findings indicate that all models exhibit a strong bias to associate a meme with figurative meaning, even when no such meaning is present. Qualitative analysis further shows that correct predictions are not always accompanied by faithful explanations.
>
---
#### [new 025] UniDial-EvalKit: A Unified Toolkit for Evaluating Multi-Faceted Conversational Abilities
- **分类: cs.CL**

- **简介: 该论文提出UniDial-EvalKit，用于统一评估对话系统。解决现有评估协议不一致的问题，通过标准化数据、简化流程、提升效率，促进对话AI发展。**

- **链接: [https://arxiv.org/pdf/2603.23160](https://arxiv.org/pdf/2603.23160)**

> **作者:** Qi Jia; Haodong Zhao; Dun Pei; Xiujie Song; Shibo Wang; Zijian Chen; Zicheng Zhang; Xiangyang Zhu; Guangtao Zhai
>
> **摘要:** Benchmarking AI systems in multi-turn interactive scenarios is essential for understanding their practical capabilities in real-world applications. However, existing evaluation protocols are highly heterogeneous, differing significantly in dataset formats, model interfaces, and evaluation pipelines, which severely impedes systematic comparison. In this work, we present UniDial-EvalKit (UDE), a unified evaluation toolkit for assessing interactive AI systems. The core contribution of UDE lies in its holistic unification: it standardizes heterogeneous data formats into a universal schema, streamlines complex evaluation pipelines through a modular architecture, and aligns metric calculations under a consistent scoring interface. It also supports efficient large-scale evaluation through parallel generation and scoring, as well as checkpoint-based caching to eliminate redundant computation. Validated across diverse multi-turn benchmarks, UDE not only guarantees high reproducibility through standardized workflows and transparent logging, but also significantly improves evaluation efficiency and extensibility. We make the complete toolkit and evaluation scripts publicly available to foster a standardized benchmarking ecosystem and accelerate future breakthroughs in interactive AI.
>
---
#### [new 026] Rashid: A Cipher-Based Framework for Exploring In-Context Language Learning
- **分类: cs.CL**

- **简介: 该论文提出Rashid框架，用于研究未见语言的上下文语言学习（ICLL）。针对资源匮乏语言的实验困难，通过加密高资源语言构建真正未见语言，评估现有方法并探索资源有效性。**

- **链接: [https://arxiv.org/pdf/2603.22497](https://arxiv.org/pdf/2603.22497)**

> **作者:** Niyati Bafna; Ryan Soh-Eun Shim; Barbara Plank; David Yarowsky; Hale Sirin
>
> **摘要:** Where there is growing interest in in-context language learning (ICLL) for unseen languages with large language models, such languages usually suffer from the lack of NLP tools, data resources, and researcher expertise. This means that progress is difficult to assess, the field does not allow for cheap large-scale experimentation, and findings on ICLL are often limited to very few languages and tasks. In light of such limitations, we introduce a framework (Rashid), for studying ICLL wherein we reversibly cipher high-resource languages (HRLs) to construct truly unseen languages with access to a wide range of resources available for HRLs, unlocking previously impossible exploration of ICLL phenomena. We use our framework to assess current methods in the field with SOTA evaluation tools and manual analysis, explore the utility of potentially expensive resources in improving ICLL, and test ICLL strategies on rich downstream tasks beyond machine translation. These lines of exploration showcase the possibilities enabled by our framework, as well as providing actionable insights regarding current performance and future directions in ICLL.
>
---
#### [new 027] PRISM: A Dual View of LLM Reasoning through Semantic Flow and Latent Computation
- **分类: cs.CL**

- **简介: 该论文提出PRISM框架，用于分析大语言模型的推理过程，解决单一视角分析推理轨迹的不足。通过结合语义流动和隐式计算，揭示推理中的系统性模式与问题。**

- **链接: [https://arxiv.org/pdf/2603.22754](https://arxiv.org/pdf/2603.22754)**

> **作者:** Ruidi Chang; Jiawei Zhou; Hanjie Chen
>
> **摘要:** Large language models (LLMs) solve complex problems by generating multi-step reasoning traces. Yet these traces are typically analyzed from only one of two perspectives: the sequence of tokens across different reasoning steps in the generated text, or the hidden-state vectors across model layers within one step. We introduce PRISM (Probabilistic Reasoning Inspection through Semantic and Implicit Modeling), a framework and diagnostic tool for jointly analyzing both levels, providing a unified view of how reasoning evolves across steps and layers. Across multiple reasoning models and benchmarks, PRISM uncovers systematic patterns in the reasoning process, showing that failed trajectories are more likely to become trapped in unproductive verification loops and further diverge into distinct modes such as overthinking and premature commitment, which behave differently once a candidate answer is reached. It further reveals how prompting reshapes reasoning behavior beyond aggregate accuracy by altering both semantic transitions and internal computational patterns. By modeling reasoning trajectories as structured processes, PRISM makes these behaviors observable and analyzable rather than relying solely on final-task accuracy. Taken together, these insights position PRISM as a practical tool for analyzing and diagnosing reasoning processes in LLMs.
>
---
#### [new 028] Multi-Method Validation of Large Language Model Medical Translation Across High- and Low-Resource Languages
- **分类: cs.CL**

- **简介: 论文评估了大语言模型在医学翻译中的表现，解决语言障碍问题。通过多方法验证，测试模型在高、低资源语言间的翻译效果，证明模型能有效保持医学语义。**

- **链接: [https://arxiv.org/pdf/2603.22642](https://arxiv.org/pdf/2603.22642)**

> **作者:** Chukwuebuka Anyaegbuna; Eduardo Juan Perez Guerrero; Jerry Liu; Timothy Keyes; April Liang; Natasha Steele; Stephen Ma; Jonathan Chen; Kevin Schulman
>
> **备注:** 32 references, 5 tables, 2 figures
>
> **摘要:** Language barriers affect 27.3 million U.S. residents with non-English language preference, yet professional medical translation remains costly and often unavailable. We evaluated four frontier large language models (GPT-5.1, Claude Opus 4.5, Gemini 3 Pro, Kimi K2) translating 22 medical documents into 8 languages spanning high-resource (Spanish, Chinese, Russian, Vietnamese), medium-resource (Korean, Arabic), and low-resource (Tagalog, Haitian Creole) categories using a five-layer validation framework. Across 704 translation pairs, all models achieved high semantic preservation (LaBSE greater than 0.92), with no significant difference between high- and low-resource languages (p = 0.066). Cross-model back-translation confirmed results were not driven by same-model circularity (delta = -0.0009). Inter-model concordance across four independently trained models was high (LaBSE: 0.946), and lexical borrowing analysis showed no correlation between English term retention and fidelity scores in low-resource languages (rho = +0.018, p = 0.82). These converging results suggest frontier LLMs preserve medical meaning across resource levels, with implications for language access in healthcare.
>
---
#### [new 029] Decoding AI Authorship: Can LLMs Truly Mimic Human Style Across Literature and Politics?
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于作者身份识别任务，旨在检验LLMs能否真实模仿人类写作风格。通过分析AI生成文本与人类作品的差异，评估其可检测性。**

- **链接: [https://arxiv.org/pdf/2603.23219](https://arxiv.org/pdf/2603.23219)**

> **作者:** Nasser A Alsadhan
>
> **备注:** Preprint. Accepted for publication in Digital Scholarship in the Humanities (OUP)
>
> **摘要:** Amidst the rising capabilities of generative AI to mimic specific human styles, this study investigates the ability of state-of-the-art large language models (LLMs), including GPT-4o, Gemini 1.5 Pro, and Claude Sonnet 3.5, to emulate the authorial signatures of prominent literary and political figures: Walt Whitman, William Wordsworth, Donald Trump, and Barack Obama. Utilizing a zero-shot prompting framework with strict thematic alignment, we generated synthetic corpora evaluated through a complementary framework combining transformer-based classification (BERT) and interpretable machine learning (XGBoost). Our methodology integrates Linguistic Inquiry and Word Count (LIWC) markers, perplexity, and readability indices to assess the divergence between AI-generated and human-authored text. Results demonstrate that AI-generated mimicry remains highly detectable, with XGBoost models trained on a restricted set of eight stylometric features achieving accuracy comparable to high-dimensional neural classifiers. Feature importance analyses identify perplexity as the primary discriminative metric, revealing a significant divergence in the stochastic regularity of AI outputs compared to the higher variability of human writing. While LLMs exhibit distributional convergence with human authors on low-dimensional heuristic features, such as syntactic complexity and readability, they do not yet fully replicate the nuanced affective density and stylistic variance inherent in the human-authored corpus. By isolating the specific statistical gaps in current generative mimicry, this study provides a comprehensive benchmark for LLM stylistic behavior and offers critical insights for authorship attribution in the digital humanities and social media.
>
---
#### [new 030] DariMis: Harm-Aware Modeling for Dari Misinformation Detection on YouTube
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于信息检测任务，旨在解决Dari语言YouTube视频的虚假信息识别问题。通过构建数据集并提出编码策略提升检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22977](https://arxiv.org/pdf/2603.22977)**

> **作者:** Jawid Ahmad Baktash; Mosa Ebrahimi; Mohammad Zarif Joya; Mursal Dawodi
>
> **备注:** 9 pages, 8 figures. Accepted for submission; dataset and code will be released upon publication
>
> **摘要:** Dari, the primary language of Afghanistan, is spoken by tens of millions of people yet remains largely absent from the misinformation detection literature. We address this gap with DariMis, the first manually annotated dataset of 9,224 Dari-language YouTube videos, labeled across two dimensions: Information Type (Misinformation, Partly True, True) and Harm Level (Low, Medium, High). A central empirical finding is that these dimensions are structurally coupled, not independent: 55.9 percent of Misinformation carries at least Medium harm potential, compared with only 1.0 percent of True content. This enables Information Type classifiers to function as implicit harm-triage filters in content moderation pipelines. We further propose a pair-input encoding strategy that represents the video title and description as separate BERT segment inputs, explicitly modeling the semantic relationship between headline claims and body content, a key signal of misleading information. An ablation study against single-field concatenation shows that pair-input encoding yields a 7.0 percentage point gain in Misinformation recall (60.1 percent to 67.1 percent), the safety-critical minority class, despite modest overall macro F1 differences (0.09 percentage points). We benchmark a Dari/Farsi-specialized model (ParsBERT) against XLM-RoBERTa-base; ParsBERT achieves the best test performance with accuracy of 76.60 percent and macro F1 of 72.77 percent. Bootstrap 95 percent confidence intervals are reported for all metrics, and we discuss both the practical significance and statistical limitations of the results.
>
---
#### [new 031] When AI Shows Its Work, Is It Actually Working? Step-Level Evaluation Reveals Frontier Language Models Frequently Bypass Their Own Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究AI语言模型在推理过程中的真实性，属于自然语言处理任务。旨在解决模型是否真正依赖推理步骤还是仅作装饰的问题。通过移除推理步骤测试答案变化，发现多数模型的推理是装饰性的。**

- **链接: [https://arxiv.org/pdf/2603.22816](https://arxiv.org/pdf/2603.22816)**

> **作者:** Abhinaba Basu; Pavan Chakraborty
>
> **摘要:** Language models increasingly "show their work" by writing step-by-step reasoning before answering. But are these reasoning steps genuinely used, or decorative narratives generated after the model has already decided? Consider: a medical AI writes "The patient's eosinophilia and livedo reticularis following catheterization suggest cholesterol embolization syndrome. Answer: B." If we remove the eosinophilia observation, does the diagnosis change? For most frontier models, the answer is no - the step was decorative. We introduce step-level evaluation: remove one reasoning sentence at a time and check whether the answer changes. This simple test requires only API access -- no model weights -- and costs approximately $1-2 per model per task. Testing 10 frontier models (GPT-5.4, Claude Opus, DeepSeek-V3.2, MiniMax-M2.5, Kimi-K2.5, and others) across sentiment, mathematics, topic classification, and medical QA (N=376-500 each), the majority produce decorative reasoning: removing any step changes the answer less than 17% of the time, while any single step alone recovers the answer. This holds even on math, where smaller models (0.8-8B) show genuine step dependence (55% necessity). Two models break the pattern: MiniMax-M2.5 on sentiment (37% necessity) and Kimi-K2.5 on topic classification (39%) - but both shortcut other tasks. Faithfulness is model-specific and task-specific. We also discover "output rigidity": on the same medical questions, Claude Opus writes 11 diagnostic steps while GPT-OSS-120B outputs a single token. Mechanistic analysis (attention patterns) confirms that CoT attention drops more in late layers for decorative tasks (33%) than faithful ones (20%). Implications: step-by-step explanations from frontier models are largely decorative, per-model per-domain evaluation is essential, and training objectives - not scale - determine whether reasoning is genuine.
>
---
#### [new 032] Improving LLM Predictions via Inter-Layer Structural Encoders
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出ILSE方法，通过整合LLM中间层结构信息提升预测性能，解决传统仅依赖最终层表示的局限性。**

- **链接: [https://arxiv.org/pdf/2603.22665](https://arxiv.org/pdf/2603.22665)**

> **作者:** Tom Ulanovski; Eyal Blyachman; Maya Bechler-Speicher
>
> **备注:** 17 pages, 3 figures. Equal contribution by first two authors
>
> **摘要:** The standard practice in Large Language Models (LLMs) is to base predictions on the final-layer token representations. Recent studies, however, show that intermediate layers encode substantial information, which may contain more task-relevant features than the final-layer representations alone. Importantly, it was shown that for different tasks, different layers may be optimal. In this work we introduce Inter-Layer Structural Encoders (ILSE), a powerful structural approach to learn one effective representation from the LLM's internal layer representations all together. Central to ILSE is Cayley-Encoder, a mathematically grounded geometric encoder that leverages expander Cayley graphs for efficient inter-layer information propagation. We evaluate ILSE across 13 classification and semantic similarity tasks with 9 pre-trained LLMs ranging from 14 million to 8 billion parameters. ILSE consistently outperforms baselines and existing approaches, achieving up to 44% improvement in accuracy and 25% in similarity metrics. We further show that ILSE is data-efficient in few-shot regimes and can make small LLMs competitive with substantially larger models.
>
---
#### [new 033] Synthetic or Authentic? Building Mental Patient Simulators from Longitudinal Evidence
- **分类: cs.CL**

- **简介: 该论文属于心理健康对话系统任务，旨在解决患者模拟中行为单一、病程不连贯的问题。通过构建多源患者档案并引入时序记忆机制，提升对话真实性和多样性。**

- **链接: [https://arxiv.org/pdf/2603.22704](https://arxiv.org/pdf/2603.22704)**

> **作者:** Baihan Li; Bingrui Jin; Kunyao Lan; Ming Wang; Mengyue Wu
>
> **摘要:** Patient simulation is essential for developing and evaluating mental health dialogue systems. As most existing approaches rely on snapshot-style prompts with limited profile information, homogeneous behaviors and incoherent disease progression in multi-turn interactions have become key chellenges. In this work, we propose DEPROFILE, a data-grounded patient simulation framework that constructs unified, multi-source patient profiles by integrating demographic attributes, standardized clinical symptoms, counseling dialogues, and longitudinal life-event histories from real-world data. We further introduce a Chain-of-Change agent to transform noisy longitudinal records into structured, temporally grounded memory representations for simulation. Experiments across multiple large language model (LLM) backbones show that with more comprehensive profile constructed by DEPROFILE, the dialogue realism, behavioral diversity, and event richness have consistently improved and exceed state-of-the-art baselines, highlighting the importance of grounding patient simulation in verifiable longitudinal evidence.
>
---
#### [new 034] Steering LLMs for Culturally Localized Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs文化偏见问题。通过机制可解释性方法，识别并操控文化表征，提升生成内容的文化契合度。**

- **链接: [https://arxiv.org/pdf/2603.23301](https://arxiv.org/pdf/2603.23301)**

> **作者:** Simran Khanuja; Hongbin Liu; Shujian Zhang; John Lambert; Mingqing Chen; Rajiv Mathews; Lun Wang
>
> **备注:** preprint
>
> **摘要:** LLMs are deployed globally, yet produce responses biased towards cultures with abundant training data. Existing cultural localization approaches such as prompting or post-training alignment are black-box, hard to control, and do not reveal whether failures reflect missing knowledge or poor elicitation. In this paper, we address these gaps using mechanistic interpretability to uncover and manipulate cultural representations in LLMs. Leveraging sparse autoencoders, we identify interpretable features that encode culturally salient information and aggregate them into Cultural Embeddings (CuE). We use CuE both to analyze implicit cultural biases under underspecified prompts and to construct white-box steering interventions. Across multiple models, we show that CuE-based steering increases cultural faithfulness and elicits significantly rarer, long-tail cultural concepts than prompting alone. Notably, CuE-based steering is complementary to black-box localization methods, offering gains when applied on top of prompt-augmented inputs. This also suggests that models do benefit from better elicitation strategies, and don't necessarily lack long-tail knowledge representation, though this varies across cultures. Our results provide both diagnostic insight into cultural representations in LLMs and a controllable method to steer towards desired cultures.
>
---
#### [new 035] Who Spoke What When? Evaluating Spoken Language Models for Conversational ASR with Semantic and Overlap-Aware Metrics
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于对话语音识别任务，旨在解决多说话人场景下的语音识别问题。通过对比LLM与模块化系统，提出新评估指标，分析不同场景下的性能差异。**

- **链接: [https://arxiv.org/pdf/2603.22709](https://arxiv.org/pdf/2603.22709)**

> **作者:** Naohiro Tawara; Samuele Cornell; Alexander Polok; Marc Delcroix; Lukáš Burget; Shinji Watanabe
>
> **备注:** Submitted to INTERSPEECH 2026
>
> **摘要:** Conversational automatic speech recognition remains challenging due to overlapping speech, far-field noise, and varying speaker counts. While recent LLM-based systems perform well on single-speaker benchmarks, their robustness in multi-speaker settings is unclear. We systematically compare LLM-based and modular pipeline approaches along four axes: overlap robustness, semantic fidelity, speaker count, and single- versus multi-channel input. To capture meaning-altering errors that conventional metrics miss, we introduce tcpSemER, which extends tcpWER by replacing Levenshtein distance with embedding-based semantic similarity. We further decompose tcpWER into overlapping and non-overlapping components for finer-grained analysis. Experiments across three datasets show that LLM-based systems are competitive in two-speaker settings but degrade as speaker count and overlap increase, whereas modular pipelines remain more robust.
>
---
#### [new 036] Whether, Not Which: Mechanistic Interpretability Reveals Dissociable Affect Reception and Emotion Categorization in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感理解任务，旨在验证大语言模型是否真正识别情绪而非仅依赖关键词。通过临床刺激和可解释性方法，发现两种分离的情绪处理机制：情绪感知与分类。**

- **链接: [https://arxiv.org/pdf/2603.22295](https://arxiv.org/pdf/2603.22295)**

> **作者:** Michael Keeman
>
> **备注:** 38 pages, 11 figures, 16 tables. Code and data: this https URL
>
> **摘要:** Large language models appear to develop internal representations of emotion -- "emotion circuits," "emotion neurons," and structured emotional manifolds have been reported across multiple model families. But every study making these claims uses stimuli signalled by explicit emotion keywords, leaving a fundamental question unanswered: do these circuits detect genuine emotional meaning, or do they detect the word "devastated"? We present the first clinical validity test of emotion circuit claims using mechanistic interpretability methods grounded in clinical psychology -- clinical vignettes that evoke emotions through situational and behavioural cues alone, emotion keywords removed. Across six models (Llama-3.2-1B, Llama-3-8B, Gemma-2-9B; base and instruct variants), we apply four convergent mechanistic interpretability methods -- linear probing, causal activation patching, knockout experiments, and representational geometry -- and discover two dissociable emotion processing mechanisms. Affect reception -- detecting emotionally significant content -- operates with near-perfect accuracy (AUROC 1.000), consistent with early-layer saturation, and replicates across all six models. Emotion categorization -- mapping affect to specific emotion labels -- is partially keyword-dependent, dropping 1-7% without keywords and improving with scale. Causal activation patching confirms keyword-rich and keyword-free stimuli share representational space, transferring affective salience rather than emotion-category identity. These findings falsify the keyword-spotting hypothesis, establish a novel mechanistic dissociation, and introduce clinical stimulus methodology as a rigorous standard for testing emotion processing claims in large language models -- with direct implications for AI safety evaluation and alignment. All stimuli, code, and data are released for replication.
>
---
#### [new 037] Explanation Generation for Contradiction Reconciliation with LLMs
- **分类: cs.CL**

- **简介: 该论文提出 reconciliatory explanation generation 任务，旨在生成解释以化解矛盾陈述。研究探索 LLM 在此任务上的表现及优化方法。**

- **链接: [https://arxiv.org/pdf/2603.22735](https://arxiv.org/pdf/2603.22735)**

> **作者:** Jason Chan; Zhixue Zhao; Robert Gaizauskas
>
> **备注:** Preprint
>
> **摘要:** Existing NLP work commonly treats contradictions as errors to be resolved by choosing which statements to accept or discard. Yet a key aspect of human reasoning in social interactions and professional domains is the ability to hypothesize explanations that reconcile contradictions. For example, "Cassie hates coffee" and "She buys coffee everyday" may appear contradictory, yet both are compatible if Cassie has the unenviable daily chore of buying coffee for all her coworkers. Despite the growing reasoning capabilities of large language models (LLMs), their ability to hypothesize such reconciliatory explanations remains largely unexplored. To address this gap, we introduce the task of reconciliatory explanation generation, where models must generate explanations that effectively render contradictory statements compatible. We propose a novel method of repurposing existing natural language inference (NLI) datasets, and introduce quality metrics that enable scalable automatic evaluation. Experiments with 18 LLMs show that most models achieve limited success in this task, and that the benefit of extending test-time compute by "thinking" plateaus as model size increases. Our results highlight an under-explored dimension of LLM reasoning and the need to address this limitation in enhancing LLMs' downstream applications such as chatbots and scientific aids.
>
---
#### [new 038] From Synthetic to Native: Benchmarking Multilingual Intent Classification in Logistics Customer Service
- **分类: cs.CL**

- **简介: 该论文属于多语言意图分类任务，旨在解决现有基准因使用机器翻译文本而高估实际性能的问题。通过构建真实物流客服数据集，评估不同模型在真实语境下的表现。**

- **链接: [https://arxiv.org/pdf/2603.23172](https://arxiv.org/pdf/2603.23172)**

> **作者:** Haoyu He; Jinyu Zhuang; Haoran Chu; Shuhang Yu; T AI Group; Hao Wang; Kunpeng Han
>
> **摘要:** Multilingual intent classification is central to customer-service systems on global logistics platforms, where models must process noisy user queries across languages and hierarchical label spaces. Yet most existing multilingual benchmarks rely on machine-translated text, which is typically cleaner and more standardized than native customer requests and can therefore overestimate real-world robustness. We present a public benchmark for hierarchical multilingual intent classification constructed from real logistics customer-service logs. The dataset contains approximately 30K de-identified, stand-alone user queries curated from 600K historical records through filtering, LLM-assisted quality control, and human verification, and is organized into a two-level taxonomy with 13 parent and 17 leaf intents. English, Spanish, and Arabic are included as seen languages, while Indonesian, Chinese, and additional test-only languages support zero-shot evaluation. To directly measure the gap between synthetic and real evaluation, we provide paired native and machine-translated test sets and benchmark multilingual encoders, embedding models, and small language models under flat and hierarchical protocols. Results show that translated test sets substantially overestimate performance on noisy native queries, especially for long-tail intents and cross-lingual transfer, underscoring the need for more realistic multilingual intent benchmarks.
>
---
#### [new 039] RadTimeline: Timeline Summarization for Longitudinal Radiological Lung Findings
- **分类: cs.CL**

- **简介: 该论文提出RadTimeline任务，旨在自动总结纵向放射学报告中的肺部影像发现，解决人工整理耗时的问题。通过三步LLM流程生成结构化时间线。**

- **链接: [https://arxiv.org/pdf/2603.22820](https://arxiv.org/pdf/2603.22820)**

> **作者:** Sitong Zhou; Meliha Yetisgen; Mari Ostendorf
>
> **备注:** Accepted at Language Resources and Evaluation Conference (LREC) 2026
>
> **摘要:** Tracking findings in longitudinal radiology reports is crucial for accurately identifying disease progression, and the time-consuming process would benefit from automatic summarization. This work introduces a structured summarization task, where we frame longitudinal report summarization as a timeline generation task, with dated findings organized in columns and temporally related findings grouped in rows. This structured summarization format enables straightforward comparison of findings across time and facilitates fact-checking against the associated reports. The timeline is generated using a 3-step LLM process of extracting findings, generating group names, and using the names to group the findings. To evaluate such systems, we create RadTimeline, a timeline dataset focused on tracking lung-related radiologic findings in chest-related imaging reports. Experiments on RadTimeline show tradeoffs of different-sized LLMs and prompting strategies. Our results highlight that group name generation as an intermediate step is critical for effective finding grouping. The best configuration has some irrelevant findings but very good recall, and grouping performance is comparable to human annotators.
>
---
#### [new 040] How Utilitarian Are OpenAI's Models Really? Replicating and Reinterpreting Pfeffer, Krügel, and Uhl (2025)
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于伦理评估任务，旨在检验OpenAI模型在道德困境中的表现。研究复制并扩展了先前实验，发现模型反应受提示影响大，单次测试不可靠，需多提示验证。**

- **链接: [https://arxiv.org/pdf/2603.22730](https://arxiv.org/pdf/2603.22730)**

> **作者:** Johannes Himmelreich
>
> **备注:** 10 pages, 2 figures, 2 tables. Supplementary materials included as ancillary file
>
> **摘要:** Pfeffer, Krügel, and Uhl (2025) report that OpenAI's reasoning model o1-mini produces more utilitarian responses to the trolley problem and footbridge dilemma than the non-reasoning model GPT-4o. I replicate their study with four current OpenAI models and extend it with prompt variant testing. The trolley finding does not survive: GPT-4o's low utilitarian rate doesn't reflect a deontological commitment but safety refusals triggered by the prompt's advisory framing. When framed as "Is it morally permissible...?" instead of "Should I...?", GPT-4o gives 99% utilitarian responses. All models converge on utilitarian answers when prompt confounds are removed. The footbridge finding survives with blemishes. Reasoning models tend to give more utilitarian responses than non-reasoning models across prompt variations. But often they refuse to answer the dilemma or, when they answer, give a non-utilitarian rather than a utilitarian answer. These results demonstrate that single-prompt evaluations of LLM moral reasoning are unreliable: multi-prompt robustness testing should be standard practice for any empirical claim about LLM behavior.
>
---
#### [new 041] PaperVoyager : Building Interactive Web with Visual Language Models
- **分类: cs.CL**

- **简介: 该论文属于将科技论文转换为交互式网页的任务，旨在解决静态文档无法展现动态机制的问题。工作包括构建自动处理系统和提出PaperVoyager框架，提升交互系统生成质量。**

- **链接: [https://arxiv.org/pdf/2603.22999](https://arxiv.org/pdf/2603.22999)**

> **作者:** Dasen Dai; Biao Wu; Meng Fang; Wenhao Wang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Recent advances in visual language models have enabled autonomous agents for complex reasoning, tool use, and document understanding. However, existing document agents mainly transform papers into static artifacts such as summaries, webpages, or slides, which are insufficient for technical papers involving dynamic mechanisms and state transitions. In this work, we propose a Paper-to-Interactive-System Agent that converts research papers into executable interactive web systems. Given a PDF paper, the agent performs end-to-end processing without human intervention, including paper understanding, system modeling, and interactive webpage synthesis, enabling users to manipulate inputs and observe dynamic behaviors. To evaluate this task, we introduce a benchmark of 19 research papers paired with expert-built interactive systems as ground truth. We further propose PaperVoyager, a structured generation framework that explicitly models mechanisms and interaction logic during synthesis. Experiments show that PaperVoyager significantly improves the quality of generated interactive systems, offering a new paradigm for interactive scientific paper understanding.
>
---
#### [new 042] WISTERIA: Weak Implicit Signal-based Temporal Relation Extraction with Attention
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于时间关系抽取任务，旨在解决现有模型忽视事件对特有线索的问题。提出WISTERIA框架，通过注意力机制捕捉隐含的时间信号，提升模型的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.23319](https://arxiv.org/pdf/2603.23319)**

> **作者:** Duy Dao Do; Anaïs Halftermeyer; Thi-Bich-Hanh Dao
>
> **备注:** 19 pages, 16 figures, LREC 2026
>
> **摘要:** Temporal Relation Extraction (TRE) requires identifying how two events or temporal expressions are related in time. Existing attention-based models often highlight globally salient tokens but overlook the pair-specific cues that actually determine the temporal relation. We propose WISTERIA (Weak Implicit Signal-based Temporal Relation Extraction with Attention), a framework that examines whether the top-K attention components conditioned on each event pair truly encode interpretable evidence for temporal classification. Unlike prior works assuming explicit markers such as before, after, or when, WISTERIA considers signals as any lexical, syntactic, or morphological element implicitly expressing temporal order. By combining multi-head attention with pair-conditioned top-K pooling, the model isolates the most informative contextual tokens for each pair. We conduct extensive experiments on TimeBank-Dense, MATRES, TDDMan, and TDDAuto, including linguistic analyses of top-K tokens. Results show that WISTERIA achieves competitive accuracy and reveals pair-level rationales aligned with temporal linguistic cues, offering a localized and interpretable view of temporal reasoning.
>
---
#### [new 043] Span Modeling for Idiomaticity and Figurative Language Detection with Span Contrastive Loss
- **分类: cs.CL**

- **简介: 该论文属于 figurative language detection 任务，旨在提升对习语等非组合性表达的识别。通过结合槽位损失和跨度对比损失的模型优化，提高了检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22799](https://arxiv.org/pdf/2603.22799)**

> **作者:** Blake Matheny; Phuong Minh Nguyen; Minh Le Nguyen
>
> **摘要:** The category of figurative language contains many varieties, some of which are non-compositional in nature. This type of phrase or multi-word expression (MWE) includes idioms, which represent a single meaning that does not consist of the sum of its words. For language models, this presents a unique problem due to tokenization and adjacent contextual embeddings. Many large language models have overcome this issue with large phrase vocabulary, though immediate recognition frequently fails without one- or few-shot prompting or instruction finetuning. The best results have been achieved with BERT-based or LSTM finetuning approaches. The model in this paper contains one such variety. We propose BERT- and RoBERTa-based models finetuned with a combination of slot loss and span contrastive loss (SCL) with hard negative reweighting to improve idiomaticity detection, attaining state of the art sequence accuracy performance on existing datasets. Comparative ablation studies show the effectiveness of SCL and its generalizability. The geometric mean of F1 and sequence accuracy (SA) is also proposed to assess a model's span awareness and general performance together.
>
---
#### [new 044] Avoiding Over-smoothing in Social Media Rumor Detection with Pre-trained Propagation Tree Transformer
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于社会媒体谣言检测任务，旨在解决GNN在处理谣言传播树时的过平滑问题。提出P2T3方法，利用Transformer结构提升性能。**

- **链接: [https://arxiv.org/pdf/2603.22854](https://arxiv.org/pdf/2603.22854)**

> **作者:** Chaoqun Cui; Caiyan Jia
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Deep learning techniques for rumor detection typically utilize Graph Neural Networks (GNNs) to analyze post relations. These methods, however, falter due to over-smoothing issues when processing rumor propagation structures, leading to declining performance. Our investigation into this issue reveals that over-smoothing is intrinsically tied to the structural characteristics of rumor propagation trees, in which the majority of nodes are 1-level nodes. Furthermore, GNNs struggle to capture long-range dependencies within these trees. To circumvent these challenges, we propose a Pre-Trained Propagation Tree Transformer (P2T3) method based on pure Transformer architecture. It extracts all conversation chains from a tree structure following the propagation direction of replies, utilizes token-wise embedding to infuse connection information and introduces necessary inductive bias, and pre-trains on large-scale unlabeled datasets. Experiments indicate that P2T3 surpasses previous state-of-the-art methods in multiple benchmark datasets and performs well under few-shot conditions. P2T3 not only avoids the over-smoothing issue inherent in GNNs but also potentially offers a large model or unified multi-modal scheme for future social media research.
>
---
#### [new 045] Functional Component Ablation Reveals Specialization Patterns in Hybrid Language Model Architectures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究混合语言模型架构中各组件的功能角色，通过消融实验揭示其专业化模式，旨在优化模型压缩与设计。**

- **链接: [https://arxiv.org/pdf/2603.22473](https://arxiv.org/pdf/2603.22473)**

> **作者:** Hector Borobia; Elies Seguí-Mas; Guillermina Tormo-Carbó
>
> **备注:** 22 pages, 7 figures, 6 tables. Code and data available at this https URL
>
> **摘要:** Hybrid language models combining attention with state space models (SSMs) or linear attention offer improved efficiency, but whether both components are genuinely utilized remains unclear. We present a functional component ablation framework applied to two sub-1B hybrid models -- Qwen3.5-0.8B (sequential: Gated DeltaNet + softmax attention) and Falcon-H1-0.5B (parallel: Mamba-2 + attention) -- with a pure Transformer control (Qwen2.5-0.5B). Through group ablations, layer-wise sweeps, positional ablations, matched random controls, and perplexity analysis across five benchmarks, we establish four findings: (1) both component types are essential and neither is bypassed; (2) the alternative component (linear attention or SSM) is the primary language modeling backbone, causing >35,000x perplexity degradation when removed versus ~82x for attention; (3) component importance follows a positional gradient, with early layers being disproportionately critical; and (4) hybrid architectures exhibit 20-119x greater resilience to random layer removal than pure Transformers, revealing built-in functional redundancy between component types. These results provide actionable guidance for hybrid model compression, architecture design, and fault-tolerant deployment.
>
---
#### [new 046] EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM长文本应用中KV缓存内存需求过高的问题。提出EchoKV方法，通过相似性重建实现灵活的压缩与恢复，提升效率。**

- **链接: [https://arxiv.org/pdf/2603.22910](https://arxiv.org/pdf/2603.22910)**

> **作者:** Yixuan Wang; Shiyu Ji; Yijun Liu; Qingfu Zhu; Wanxiang Che
>
> **摘要:** The increasing memory demand of the Key-Value (KV) cache poses a significant bottleneck for Large Language Models (LLMs) in long-context applications. Existing low-rank compression methods often rely on irreversible parameter transformations, sacrificing the flexibility to switch back to full-precision inference when memory is abundant. In this paper, we propose EchoKV, a flexible KV cache compression scheme that enables on-demand transitions between standard and compressed inference. Unlike traditional compression-decompression paradigms, EchoKV utilizes a lightweight network to reconstruct the residual KV components from a partial subset, leveraging intrinsic inter-layer and intra-layer similarities among attention heads. We further introduce a two-stage fine-tuning strategy that allows for rapid, low-cost training (e.g., ~1 A100 GPU-hour for a 7B model). Experimental results on LongBench and RULER demonstrate that EchoKV consistently outperforms existing methods across various compression ratios while maintaining high throughput for short-context scenarios.
>
---
#### [new 047] DALDALL: Data Augmentation for Lexical and Semantic Diverse in Legal Domain by leveraging LLM-Persona
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出DALDALL框架，解决法律领域数据稀缺问题。通过角色化生成增强数据多样性，提升信息检索效果。**

- **链接: [https://arxiv.org/pdf/2603.22765](https://arxiv.org/pdf/2603.22765)**

> **作者:** Janghyeok Choi; Jaewon Lee; Sungzoon Cho
>
> **摘要:** Data scarcity remains a persistent challenge in low-resource domains. While existing data augmentation methods leverage the generative capabilities of large language models (LLMs) to produce large volumes of synthetic data, these approaches often prioritize quantity over quality and lack domain-specific strategies. In this work, we introduce DALDALL, a persona-based data augmentation framework tailored for legal information retrieval (IR). Our method employs domain-specific professional personas--such as attorneys, prosecutors, and judges--to generate synthetic queries that exhibit substantially greater lexical and semantic diversity than vanilla prompting approaches. Experiments on the CLERC and COLIEE benchmarks demonstrate that persona-based augmentation achieves improvement in lexical diversity as measured by Self-BLEU scores, while preserving semantic fidelity to the original queries. Furthermore, dense retrievers fine-tuned on persona-augmented data consistently achieve competitive or superior recall performance compared to those trained on original data or generic augmentations. These findings establish persona-based prompting as an effective strategy for generating high-quality training data in specialized, low-resource domains.
>
---
#### [new 048] Sparse but Critical: A Token-Level Analysis of Distributional Shifts in RLVR Fine-Tuning of LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究RLVR微调中token级别的分布变化，分析其对模型性能的影响，旨在理解RLVR如何提升LLM推理能力。**

- **链接: [https://arxiv.org/pdf/2603.22446](https://arxiv.org/pdf/2603.22446)**

> **作者:** Haoming Meng; Kexin Huang; Shaohang Wei; Chiyu Ma; Shuo Yang; Xue Wang; Guoyin Wang; Bolin Ding; Jingren Zhou
>
> **备注:** Published as a conference paper at the International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has significantly improved reasoning in large language models (LLMs), yet the token-level mechanisms underlying these improvements remain unclear. We present a systematic empirical study of RLVR's distributional effects organized around three main analyses: (1) token-level characterization of distributional shifts between base and RL models, (2) the impact of token-level distributional shifts on sequence-level reasoning performance through cross-sampling interventions, and (3) fine-grained mechanics of these shifts at the token level. We find that RL fine-tuning induces highly sparse and targeted changes, with only a small fraction of token distributions exhibiting meaningful divergence between the base and RL policies. We further characterize the structure and evolution of these shifts through analyses of token entropy, positional concentration, and reallocation of probability mass. To assess the functional importance of these sparse changes, we conduct cross-sampling experiments that selectively swap token choices between the base and RL models with varying intervention budgets. We show that inserting only a small fraction of RL-sampled tokens into base generations progressively recovers RL performance gains, while injecting a similarly small number of base token choices into otherwise RL-generated sequences collapses performance to base levels, isolating a small set of token-level decisions directly responsible for RLVR's performance gains. Finally, we explore divergence-weighted variants of the advantage signal as a diagnostic intervention, finding that they can yield improvements over baselines. Together, our results shed light on the distributional changes induced by RLVR and provide a fine-grained, token-level lens for understanding RLVR fine-tuning as a targeted refinement process.
>
---
#### [new 049] Evaluating Large Language Models' Responses to Sexual and Reproductive Health Queries in Nepali
- **分类: cs.CL**

- **简介: 该论文属于评估任务，旨在解决LLM在尼泊尔语性与生殖健康问答中的准确性、可用性和安全性问题。研究构建了LEAF框架，评估14K条查询，发现仅35.1%响应合格。**

- **链接: [https://arxiv.org/pdf/2603.22291](https://arxiv.org/pdf/2603.22291)**

> **作者:** Medha Sharma; Supriya Khadka; Udit Chandra Aryal; Bishnu Hari Bhatta; Bijayan Bhattarai; Santosh Dahal; Kamal Gautam; Pushpa Joshi; Saugat Kafle; Shristi Khadka; Shushila Khadka; Binod Lamichhane; Shilpa Lamichhane; Anusha Parajuli; Sabina Pokharel; Suvekshya Sitaula; Neha Verma; Bishesh Khanal
>
> **摘要:** As Large Language Models (LLMs) become integrated into daily life, they are increasingly used for personal queries, including Sexual and Reproductive Health (SRH), allowing users to chat anonymously without fear of judgment. However, current evaluation methods primarily focus on accuracy, often for objective queries in high-resource languages, and lack criteria to assess usability and safety, especially for low-resource languages and culturally sensitive domains like SRH. This paper introduces LLM Evaluation Framework (LEAF), that conducts assessments across multiple criteria: accuracy, language, usability gaps (including relevance, adequacy, and cultural appropriateness), and safety gaps (safety, sensitivity, and confidentiality). Using the LEAF framework, we assessed 14K SRH queries in Nepali from over 9K users. Responses were manually annotated by SRH experts according to the framework. Results revealed that only 35.1% of the responses were "proper", meaning they were accurate, adequate and had no major usability or safety related gaps. Insights include differences in performance between ChatGPT versions, such as similar accuracy but varying usability and safety aspects. This evaluation highlights significant limitations of current LLMs and underscores the need for improvement. The LEAF Framework is adaptable across domains and languages, particularly where usability and safety are critical, offering a pathway to better address sensitive topics.
>
---
#### [new 050] Towards Automated Community Notes Generation with Large Vision Language Models for Combating Contextual Deception
- **分类: cs.CL; cs.SI**

- **简介: 该论文属于自动化社区注释生成任务，旨在解决图像上下文欺骗问题。通过构建数据集XCheck和提出ACCNote方法，提升注释的准确性和实用性。**

- **链接: [https://arxiv.org/pdf/2603.22453](https://arxiv.org/pdf/2603.22453)**

> **作者:** Jin Ma; Jingwen Yan; Mohammed Aldeen; Ethan Anderson; Taran Kavuru; Jinkyung Katie Park; Feng Luo; Long Cheng
>
> **摘要:** Community Notes have emerged as an effective crowd-sourced mechanism for combating online deception on social media platforms. However, its reliance on human contributors limits both the timeliness and scalability. In this work, we study the automated Community Notes generation method for image-based contextual deception, where an authentic image is paired with misleading context (e.g., time, entity, and event). Unlike prior work that primarily focuses on deception detection (i.e., judging whether a post is true or false in a binary manner), Community Notes-style systems need to generate concise and grounded notes that help users recover the missing or corrected context. This problem remains underexplored due to three reasons: (i) datasets that support the research are scarce; (ii) methods must handle the dynamic nature of contextual deception; (iii) evaluation is difficult because standard metrics do not capture whether notes actually improve user understanding. To address these gaps, we curate a real-world dataset, XCheck, comprising X posts with associated Community Notes and external contexts. We further propose the Automated Context-Corrective Note generation method, named ACCNote, which is a retrieval-augmented, multi-agent collaboration framework built on large vision-language models. Finally, we introduce a new evaluation metric, Context Helpfulness Score (CHS), that aligns with user study outcomes rather than relying on lexical overlap. Experiments on our XCheck dataset show that the proposed ACCNote improves both deception detection and note generation performance over baselines, and exceeds a commercial tool GPT5-mini. Together, our dataset, method, and metric advance practical automated generation of context-corrective notes toward more responsible online social networks.
>
---
#### [new 051] AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于作者风格迁移任务，旨在在保持原意的前提下转换文本风格。针对现有方法成本高、灵活性差的问题，提出AuthorMix框架，通过模块化适配器实现高效风格迁移。**

- **链接: [https://arxiv.org/pdf/2603.23069](https://arxiv.org/pdf/2603.23069)**

> **作者:** Sarubi Thillainathan; Ji-Ung Lee; Michael Sullivan; Alexander Koller
>
> **备注:** Under review
>
> **摘要:** The task of authorship style transfer involves rewriting text in the style of a target author while preserving the meaning of the original text. Existing style transfer methods train a single model on large corpora to model all target styles at once: this high-cost approach offers limited flexibility for target-specific adaptation, and often sacrifices meaning preservation for style transfer. In this paper, we propose AuthorMix: a lightweight, modular, and interpretable style transfer framework. We train individual, style-specific LoRA adapters on a small set of high-resource authors, allowing the rapid training of specialized adaptation models for each new target via learned, layer-wise adapter mixing, using only a handful of target style training examples. AuthorMix outperforms existing, SoTA style-transfer baselines -- as well as GPT-5.1 -- for low-resource targets, achieving the highest overall score and substantially improving meaning preservation.
>
---
#### [new 052] Is AI Catching Up to Human Expression? Exploring Emotion, Personality, Authorship, and Linguistic Style in English and Arabic with Six Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨AI在情感、个性等人类特质上的表现。研究解决AI生成文本与人类文本的区分及情感个性模仿问题，通过实验分析六种模型的表现。**

- **链接: [https://arxiv.org/pdf/2603.23251](https://arxiv.org/pdf/2603.23251)**

> **作者:** Nasser A Alsadhan
>
> **备注:** Preprint. Under review
>
> **摘要:** The advancing fluency of LLMs raises important questions about their ability to emulate complex human traits, including emotional expression and personality, across diverse linguistic and cultural contexts. This study investigates whether LLMs can convincingly mimic emotional nuance in English and personality markers in Arabic, a critical under-resourced language with unique linguistic and cultural characteristics. We conduct two tasks across six models:Jais, Mistral, LLaMA, GPT-4o, Gemini, and DeepSeek. First, we evaluate whether machine classifiers can reliably distinguish between human-authored and AI-generated texts. Second, we assess the extent to which LLM-generated texts exhibit emotional or personality traits comparable to those of humans. Our results demonstrate that AI-generated texts are distinguishable from human-authored ones (F1>0.95), though classification performance deteriorates on paraphrased samples, indicating a reliance on superficial stylistic cues. Emotion and personality classification experiments reveal significant generalization gaps: classifiers trained on human data perform poorly on AI-generated texts and vice versa, suggesting LLMs encode affective signals differently from humans. Importantly, augmenting training with AI-generated data enhances performance in the Arabic personality classification task, highlighting the potential of synthetic data to address challenges in under-resourced languages. Model-specific analyses show that GPT-4o and Gemini exhibit superior affective coherence. Linguistic and psycholinguistic analyses reveal measurable divergences in tone, authenticity, and textual complexity between human and AI texts. These findings have implications for affective computing, authorship attribution, and responsible AI deployment, particularly within underresourced language contexts where generative AI detection and alignment pose unique challenges.
>
---
#### [new 053] When Language Models Lose Their Mind: The Consequences of Brain Misalignment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究脑对齐对语言模型性能的影响。通过构建脑不对齐模型，验证脑对齐对语言理解的重要性，揭示神经表征与语言处理的关系。**

- **链接: [https://arxiv.org/pdf/2603.23091](https://arxiv.org/pdf/2603.23091)**

> **作者:** Gabriele Merlin; Mariya Toneva
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** While brain-aligned large language models (LLMs) have garnered attention for their potential as cognitive models and for potential for enhanced safety and trustworthiness in AI, the role of this brain alignment for linguistic competence remains uncertain. In this work, we investigate the functional implications of brain alignment by introducing brain-misaligned models--LLMs intentionally trained to predict brain activity poorly while maintaining high language modeling performance. We evaluate these models on over 200 downstream tasks encompassing diverse linguistic domains, including semantics, syntax, discourse, reasoning, and morphology. By comparing brain-misaligned models with well-matched brain-aligned counterparts, we isolate the specific impact of brain alignment on language understanding. Our experiments reveal that brain misalignment substantially impairs downstream performance, highlighting the critical role of brain alignment in achieving robust linguistic competence. These findings underscore the importance of brain alignment in LLMs and offer novel insights into the relationship between neural representations and linguistic processing.
>
---
#### [new 054] Founder effects shape the evolutionary dynamics of multimodality in open LLM families
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究开放大语言模型家族中多模态能力的演化，分析其传播路径与速度，揭示多模态通过少数创始事件引入并快速扩展的机制。**

- **链接: [https://arxiv.org/pdf/2603.22287](https://arxiv.org/pdf/2603.22287)**

> **作者:** Manuel Cebrian
>
> **备注:** 7 pages, 4 figures, 2 tables
>
> **摘要:** Large language model (LLM) families are improving rapidly, yet it remains unclear how quickly multimodal capabilities emerge and propagate within open families. Using the ModelBiome AI Ecosystem dataset of Hugging Face model metadata and recorded lineage fields (>1.8x10^6 model entries), we quantify multimodality over time and along recorded parent-to-child relations. Cross-modal tasks are widespread in the broader ecosystem well before they become common within major open LLM families: within these families, multimodality remains rare through 2023 and most of 2024, then increases sharply in 2024-2025 and is dominated by image-text vision-language tasks. Across major families, the first vision-language model (VLM) variants typically appear months after the first text-generation releases, with lags ranging from ~1 month (Gemma) to more than a year for several families and ~26 months for GLM. Lineage-conditioned transition rates show weak cross-type transfer: among fine-tuning edges from text-generation parents, only 0.218% yield VLM descendants. Instead, multimodality expands primarily within existing VLM lineages: 94.5% of VLM-child fine-tuning edges originate from VLM parents, versus 4.7% from text-generation parents. At the model level, most VLM releases appear as new roots without recorded parents (~60%), while the remainder are predominantly VLM-derived; founder concentration analyses indicate rapid within-lineage amplification followed by diversification. Together, these results show that multimodality enters open LLM families through rare founder events and then expands rapidly within their descendant lineages, producing punctuated adoption dynamics that likely induce distinct, transfer-limited scaling behavior for multimodal capabilities.
>
---
#### [new 055] Between Rules and Reality: On the Context Sensitivity of LLM Moral Judgment
- **分类: cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于AI伦理任务，研究LLM在不同情境下的道德判断。解决LLM道德判断受情境影响的问题，通过构建数据集并测试模型，提出控制情境敏感性的方法。**

- **链接: [https://arxiv.org/pdf/2603.23114](https://arxiv.org/pdf/2603.23114)**

> **作者:** Adrian Sauter; Mona Schirmer
>
> **备注:** preprint
>
> **摘要:** A human's moral decision depends heavily on the context. Yet research on LLM morality has largely studied fixed scenarios. We address this gap by introducing Contextual MoralChoice, a dataset of moral dilemmas with systematic contextual variations known from moral psychology to shift human judgment: consequentialist, emotional, and relational. Evaluating 22 LLMs, we find that nearly all models are context-sensitive, shifting their judgments toward rule-violating behavior. Comparing with a human survey, we find that models and humans are most triggered by different contextual variations, and that a model aligned with human judgments in the base case is not necessarily aligned in its contextual sensitivity. This raises the question of controlling contextual sensitivity, which we address with an activation steering approach that can reliably increase or decrease a model's contextual sensitivity.
>
---
#### [new 056] Off-Policy Value-Based Reinforcement Learning for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在提升大语言模型的样本效率。针对传统方法样本利用率低的问题，提出ReVal框架，结合步级信号与轨迹信号，实现高效离策略学习。**

- **链接: [https://arxiv.org/pdf/2603.23355](https://arxiv.org/pdf/2603.23355)**

> **作者:** Peng-Yuan Wang; Ziniu Li; Tian Xu; Bohan Yang; Tian-Shuo Liu; ChenYang Wang; Xiong-Hui Chen; Yi-Chen Li; Tianyun Yang; Congliang Chen; Yang Yu
>
> **摘要:** Improving data utilization efficiency is critical for scaling reinforcement learning (RL) for long-horizon tasks where generating trajectories is expensive. However, the dominant RL methods for LLMs are largely on-policy: they update each batch of data only once, discard it, and then collect fresh samples, resulting in poor sample efficiency. In this work, we explore an alternative value-based RL framework for LLMs that naturally enables off-policy learning. We propose ReVal, a Bellman-update-based method that combines stepwise signals capturing internal consistency with trajectory-level signals derived from outcome verification. ReVal naturally supports replay-buffer-based training, allowing efficient reuse of past trajectories. Experiments on standard mathematical reasoning benchmarks show that ReVal not only converges faster but also outperforms GRPO in final performance. On DeepSeek-R1-Distill-1.5B, ReVal improves training efficiency and achieves improvement of 2.7% in AIME24 and 4.5% in out-of-domain benchmark GPQA over GRPO. These results suggest that value-based RL is a practical alternative to policy-based methods for LLM training.
>
---
#### [new 057] MedObvious: Exposing the Medical Moravec's Paradox in VLMs via Clinical Triage
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于医疗视觉语言模型领域，旨在解决输入验证问题。提出MedObvious基准，测试模型在多图集中的输入一致性判断能力，揭示现有模型在预诊断验证上的不足。**

- **链接: [https://arxiv.org/pdf/2603.23501](https://arxiv.org/pdf/2603.23501)**

> **作者:** Ufaq Khan; Umair Nawaz; L D M S S Teja; Numaan Saeed; Muhammad Bilal; Yutong Xie; Mohammad Yaqub; Muhammad Haris Khan
>
> **备注:** 11 Pages
>
> **摘要:** Vision Language Models (VLMs) are increasingly used for tasks like medical report generation and visual question answering. However, fluent diagnostic text does not guarantee safe visual understanding. In clinical practice, interpretation begins with pre-diagnostic sanity checks: verifying that the input is valid to read (correct modality and anatomy, plausible viewpoint and orientation, and no obvious integrity violations). Existing benchmarks largely assume this step is solved, and therefore miss a critical failure mode: a model can produce plausible narratives even when the input is inconsistent or invalid. We introduce MedObvious, a 1,880-task benchmark that isolates input validation as a set-level consistency capability over small multi-panel image sets: the model must identify whether any panel violates expected coherence. MedObvious spans five progressive tiers, from basic orientation/modality mismatches to clinically motivated anatomy/viewpoint verification and triage-style cues, and includes five evaluation formats to test robustness across interfaces. Evaluating 17 different VLMs, we find that sanity checking remains unreliable: several models hallucinate anomalies on normal (negative-control) inputs, performance degrades when scaling to larger image sets, and measured accuracy varies substantially between multiple-choice and open-ended settings. These results show that pre-diagnostic verification remains unsolved for medical VLMs and should be treated as a distinct, safety-critical capability before deployment.
>
---
#### [new 058] EVA: Efficient Reinforcement Learning for End-to-End Video Agent
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EVA，一种用于视频理解的高效强化学习框架，解决长视频处理效率低的问题。通过规划前感知策略，实现智能视频分析。**

- **链接: [https://arxiv.org/pdf/2603.22918](https://arxiv.org/pdf/2603.22918)**

> **作者:** Yaolun Zhang; Ruohui Wang; Jiahao Wang; Yepeng Tang; Xuanyu Zheng; Haonan Duan; Hao Lu; Hanming Deng; Lewei Lu
>
> **备注:** CVPR2026
>
> **摘要:** Video understanding with multimodal large language models (MLLMs) remains challenging due to the long token sequences of videos, which contain extensive temporal dependencies and redundant frames. Existing approaches typically treat MLLMs as passive recognizers, processing entire videos or uniformly sampled frames without adaptive reasoning. Recent agent-based methods introduce external tools, yet still depend on manually designed workflows and perception-first strategies, resulting in inefficiency on long videos. We present EVA, an Efficient Reinforcement Learning framework for End-to-End Video Agent, which enables planning-before-perception through iterative summary-plan-action-reflection reasoning. EVA autonomously decides what to watch, when to watch, and how to watch, achieving query-driven and efficient video understanding. To train such agents, we design a simple yet effective three-stage learning pipeline - comprising supervised fine-tuning (SFT), Kahneman-Tversky Optimization (KTO), and Generalized Reward Policy Optimization (GRPO) - that bridges supervised imitation and reinforcement learning. We further construct high-quality datasets for each stage, supporting stable and reproducible training. We evaluate EVA on six video understanding benchmarks, demonstrating its comprehensive capabilities. Compared with existing baselines, EVA achieves a substantial improvement of 6-12% over general MLLM baselines and a further 1-3% gain over prior adaptive agent methods. Our code and model are available at this https URL.
>
---
#### [new 059] Problems with Chinchilla Approach 2: Systematic Biases in IsoFLOP Parabola Fits
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于模型优化任务，指出Chinchilla Approach 2在拟合神经网络缩放定律时存在系统性偏差，提出改进方法Chinchilla Approach 3并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.22339](https://arxiv.org/pdf/2603.22339)**

> **作者:** Eric Czech; Zhiwei Xu; Yael Elmatad; Yixin Wang; William Held
>
> **摘要:** Chinchilla Approach 2 is among the most widely used methods for fitting neural scaling laws. Its parabolic approximation introduces systematic biases in compute-optimal allocation estimates, even on noise-free synthetic data. Applied to published Llama 3 IsoFLOP data at open frontier compute scales, these biases imply a parameter underallocation corresponding to 6.5% of the $3.8\times10^{25}$ FLOP training budget and \$1.4M (90% CI: \$412K-\$2.9M) in unnecessary compute at 50% H100 MFU. Simulated multimodal model misallocations show even greater opportunity costs due to higher loss surface asymmetry. Three sources of this error are examined: IsoFLOP sampling grid width (Taylor approximation accuracy), uncentered IsoFLOP sampling, and loss surface asymmetry ($\alpha \neq \beta$). Chinchilla Approach 3 largely eliminates these biases but is often regarded as less data-efficient, numerically unstable, prone to local minima, and harder to implement. Each concern is shown to be unfounded or addressable, especially when the partially linear structure of the objective is exploited via Variable Projection, enabling unbiased inference on all five loss surface parameters through a two-dimensional optimization that is well-conditioned, analytically differentiable, and amenable to dense, or even exhaustive, grid search. It may serve as a more convenient replacement for Approach 2 or a more scalable alternative for adaptations of Approach 3 to richer scaling law formulations.
>
---
#### [new 060] LLM Olympiad: Why Model Evaluation Needs a Sealed Exam
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决基准测试易被误读的问题。提出奥运式评估机制，确保公平性和透明度。**

- **链接: [https://arxiv.org/pdf/2603.23292](https://arxiv.org/pdf/2603.23292)**

> **作者:** Jan Christian Blaise Cruz; Alham Fikri Aji
>
> **摘要:** Benchmarks and leaderboards are how NLP most often communicates progress, but in the LLM era they are increasingly easy to misread. Scores can reflect benchmark-chasing, hidden evaluation choices, or accidental exposure to test content -- not just broad capability. Closed benchmarks delay some of these issues, but reduce transparency and make it harder for the community to learn from results. We argue for a complementary practice: an Olympiad-style evaluation event where problems are sealed until evaluation, submissions are frozen in advance, and all entries run through one standardized harness. After scoring, the full task set and evaluation code are released so results can be reproduced and audited. This design aims to make strong performance harder to ``manufacture'' and easier to trust.
>
---
#### [new 061] The Efficiency Attenuation Phenomenon: A Computational Challenge to the Language of Thought Hypothesis
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI与认知科学交叉领域，探讨语言是否为思维必要条件。通过实验验证效率衰减现象，挑战语言思维假说，表明高效协作可能依赖非符号计算。**

- **链接: [https://arxiv.org/pdf/2603.22312](https://arxiv.org/pdf/2603.22312)**

> **作者:** Di Zhang
>
> **备注:** 11 pages
>
> **摘要:** This paper computationally investigates whether thought requires a language-like format, as posited by the Language of Thought (LoT) hypothesis. We introduce the ``AI Private Language'' thought experiment: if two artificial agents develop an efficient, inscrutable communication protocol via multi-agent reinforcement learning (MARL), and their performance declines when forced to use a human-comprehensible language, this Efficiency Attenuation Phenomenon (EAP) challenges the LoT. We formalize this in a cooperative navigation task under partial observability. Results show that agents with an emergent protocol achieve 50.5\% higher efficiency than those using a pre-defined, human-like symbolic protocol, confirming the EAP. This suggests optimal collaborative cognition in these systems is not mediated by symbolic structures but is naturally coupled with sub-symbolic computations. The work bridges philosophy, cognitive science, and AI, arguing for pluralism in cognitive architectures and highlighting implications for AI ethics.
>
---
#### [new 062] TTQ: Activation-Aware Test-Time Quantization to Accelerate LLM Inference On The Fly
- **分类: cs.LG; cs.CL; eess.SP**

- **简介: 该论文属于模型压缩任务，旨在解决大模型推理时的计算需求高和领域迁移问题。提出TTQ框架，在推理时在线量化，提升速度并适应不同任务。**

- **链接: [https://arxiv.org/pdf/2603.19296](https://arxiv.org/pdf/2603.19296)**

> **作者:** Toshiaki Koike-Akino; Jing Liu; Ye Wang
>
> **备注:** 25 pages
>
> **摘要:** To tackle the huge computational demand of large foundation models, activation-aware compression techniques without retraining have been introduced. However, since these methods highly rely on calibration data, domain shift issues may arise for unseen downstream tasks. We propose a test-time quantization (TTQ) framework which compresses large models on the fly at inference time to resolve this issue. With an efficient online calibration, instant activation-aware quantization can adapt every prompt regardless of the downstream tasks, yet achieving inference speedup. Several experiments demonstrate that TTQ can improve the quantization performance over state-of-the-art baselines.
>
---
#### [new 063] SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SpecEyes，解决agentic MLLMs的序列延迟问题，通过推测规划和并行加速提升效率与吞吐量。**

- **链接: [https://arxiv.org/pdf/2603.23483](https://arxiv.org/pdf/2603.23483)**

> **作者:** Haoyu Huang; Jinfa Huang; Zhongwei Wan; Xiawu Zheng; Rongrong Ji; Jiebo Luo
>
> **备注:** Code: this https URL
>
> **摘要:** Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded perception, reasoning, and tool-calling loops introduce significant sequential overhead. This overhead, termed agentic depth, incurs prohibitive latency and seriously limits system-level concurrency. To this end, we propose SpecEyes, an agentic-level speculative acceleration framework that breaks this sequential bottleneck. Our key insight is that a lightweight, tool-free MLLM can serve as a speculative planner to predict the execution trajectory, enabling early termination of expensive tool chains without sacrificing accuracy. To regulate this speculative planning, we introduce a cognitive gating mechanism based on answer separability, which quantifies the model's confidence for self-verification without requiring oracle labels. Furthermore, we design a heterogeneous parallel funnel that exploits the stateless concurrency of the small model to mask the stateful serial execution of the large model, maximizing system throughput. Extensive experiments on V* Bench, HR-Bench, and POPE demonstrate that SpecEyes achieves 1.1-3.35x speedup over the agentic baseline while preserving or even improving accuracy (up to +6.7%), thereby boosting serving throughput under concurrent workloads.
>
---
#### [new 064] YOLOv10 with Kolmogorov-Arnold networks and vision-language foundation models for interpretable object detection and trustworthy multimodal AI in computer vision perception
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决自动驾驶中模型置信度不透明的问题。通过结合Kolmogorov-Arnold网络和YOLOv10，提升检测结果的可解释性和可信度。**

- **链接: [https://arxiv.org/pdf/2603.23037](https://arxiv.org/pdf/2603.23037)**

> **作者:** Marios Impraimakis; Daniel Vazquez; Feiyu Zhou
>
> **备注:** 14 pages, 23 Figures, 6 Tables
>
> **摘要:** The interpretable object detection capabilities of a novel Kolmogorov-Arnold network framework are examined here. The approach refers to a key limitation in computer vision for autonomous vehicles perception, and beyond. These systems offer limited transparency regarding the reliability of their confidence scores in visually degraded or ambiguous scenes. To address this limitation, a Kolmogorov-Arnold network is employed as an interpretable post-hoc surrogate to model the trustworthiness of the You Only Look Once (Yolov10) detections using seven geometric and semantic features. The additive spline-based structure of the Kolmogorov-Arnold network enables direct visualisation of each feature's influence. This produces smooth and transparent functional mappings that reveal when the model's confidence is well supported and when it is unreliable. Experiments on both Common Objects in Context (COCO), and images from the University of Bath campus demonstrate that the framework accurately identifies low-trust predictions under blur, occlusion, or low texture. This provides actionable insights for filtering, review, or downstream risk mitigation. Furthermore, a bootstrapped language-image (BLIP) foundation model generates descriptive captions of each scene. This tool enables a lightweight multimodal interface without affecting the interpretability layer. The resulting system delivers interpretable object detection with trustworthy confidence estimates. It offers a powerful tool for transparent and practical perception component for autonomous and multimodal artificial intelligence applications.
>
---
#### [new 065] Demystifying Low-Rank Knowledge Distillation in Large Language Models: Convergence, Generalization, and Information-Theoretic Guarantees
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文属于模型压缩任务，旨在解决低秩知识蒸馏的理论理解问题。通过建立理论框架，分析收敛性、泛化性和信息论特性，提出最优秩选择方法，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.22355](https://arxiv.org/pdf/2603.22355)**

> **作者:** Alberlucia Rafael Soarez; Daniel Kim; Mariana Costa; Alejandro Torre
>
> **摘要:** Knowledge distillation has emerged as a powerful technique for compressing large language models (LLMs) into efficient, deployable architectures while preserving their advanced capabilities. Recent advances in low-rank knowledge distillation, particularly methods like Low-Rank Clone (LRC), have demonstrated remarkable empirical success, achieving comparable performance to full-parameter distillation with significantly reduced training data and computational overhead. However, the theoretical foundations underlying these methods remain poorly understood. In this paper, we establish a rigorous theoretical framework for low-rank knowledge distillation in language models. We prove that under mild assumptions, low-rank projection preserves the optimization dynamics, yielding explicit convergence rates of $O(1/\sqrt{T})$. We derive generalization bounds that characterize the fundamental trade-off between model compression and generalization capability, showing that the generalization error scales with the rank parameter as $O(r(m+n)/\sqrt{n})$. Furthermore, we provide an information-theoretic analysis of the activation cloning mechanism, revealing its role in maximizing the mutual information between the teacher's and student's intermediate representations. Our theoretical results offer principled guidelines for rank selection, mathematically suggesting an optimal rank $r^* = O(\sqrt{n})$ where $n$ is the sample size. Experimental validation on standard language modeling benchmarks confirms our theoretical predictions, demonstrating that the empirical convergence, rank scaling, and generalization behaviors align closely with our bounds.
>
---
#### [new 066] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出Ego2Web基准，解决AI代理在真实物理环境与网络任务间协同的问题，通过结合第一视角视频与网络任务进行评估。**

- **链接: [https://arxiv.org/pdf/2603.22529](https://arxiv.org/pdf/2603.22529)**

> **作者:** Shoubin Yu; Lei Shu; Antoine Yang; Yao Fu; Srinivas Sunkara; Maria Wang; Jindong Chen; Mohit Bansal; Boqing Gong
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal AI agents are increasingly automating complex real-world workflows that involve online web execution. However, current web-agent benchmarks suffer from a critical limitation: they focus entirely on web-based interaction and perception, lacking grounding in the user's real-world physical surroundings. This limitation prevents evaluation in crucial scenarios, such as when an agent must use egocentric visual perception (e.g., via AR glasses) to recognize an object in the user's surroundings and then complete a related task online. To address this gap, we introduce Ego2Web, the first benchmark designed to bridge egocentric video perception and web agent execution. Ego2Web pairs real-world first-person video recordings with web tasks that require visual understanding, web task planning, and interaction in an online environment for successful completion. We utilize an automatic data-generation pipeline combined with human verification and refinement to curate well-constructed, high-quality video-task pairs across diverse web task types, including e-commerce, media retrieval, knowledge lookup, etc. To facilitate accurate and scalable evaluation for our benchmark, we also develop a novel LLM-as-a-Judge automatic evaluation method, Ego2WebJudge, which achieves approximately 84% agreement with human judgment, substantially higher than existing evaluation methods. Experiments with diverse SoTA agents on our Ego2Web show that their performance is weak, with substantial headroom across all task categories. We also conduct a comprehensive ablation study on task design, highlighting the necessity of accurate video understanding in the proposed task and the limitations of current agents. We hope Ego2Web can be a critical new resource for developing truly capable AI assistants that can seamlessly see, understand, and act across the physical and digital worlds.
>
---
#### [new 067] Leveraging Large Language Models to Extract and Translate Medical Information in Doctors' Notes for Health Records and Diagnostic Billing Codes
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于医疗信息提取与编码任务，旨在解决医生因EHR文档和复杂编码导致的倦怠问题。通过本地大语言模型自动提取并转换医嘱信息为诊断代码，提升效率并保障隐私。**

- **链接: [https://arxiv.org/pdf/2603.22625](https://arxiv.org/pdf/2603.22625)**

> **作者:** Peter Hartnett; Chung-Chi Huang; Sarah Hartnett; David Hartnett
>
> **备注:** 45 pages, 19 figures
>
> **摘要:** Physician burnout in the United States has reached critical levels, driven in part by the administrative burden of Electronic Health Record (EHR) documentation and complex diagnostic codes. To relieve this strain and maintain strict patient privacy, this thesis explores an on-device, offline automatic medical coding system. The work focuses on using open-weight Large Language Models (LLMs) to extract clinical information from physician notes and translate it into ICD-10-CM diagnostic codes without reliance on cloud-based services. A privacy-focused pipeline was developed using Ollama, LangChain, and containerized environments to evaluate multiple open-weight models, including Llama 3.2, Mistral, Phi, and DeepSeek, on consumer-grade hardware. Model performance was assessed for zero-shot, few-shot, and retrieval-augmented generation (RAG) prompting strategies using a novel benchmark of synthetic medical notes. Results show that strict JSON schema enforcement achieved near 100% formatting compliance, but accurate generation of specific diagnostic codes remains challenging for smaller local models (7B-20B parameters). Contrary to common prompt-engineering guidance, few-shot prompting degraded performance through overfitting and hallucinations. While RAG enabled limited discovery of unseen codes, it frequently saturated context windows, reducing overall accuracy. The findings suggest that fully automated unsupervised coding with local open-source models is not yet reliable; instead, a human-in-the-loop assisted coding approach is currently the most practical path forward. This work contributes a reproducible local LLM architecture and benchmark dataset for privacy-preserving medical information extraction and coding.
>
---
#### [new 068] From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于LLM代理工作流优化领域，旨在解决如何设计和优化动态工作流结构的问题。通过分类现有方法，提出统一框架和评估标准。**

- **链接: [https://arxiv.org/pdf/2603.22386](https://arxiv.org/pdf/2603.22386)**

> **作者:** Ling Yue; Kushal Raj Bhandari; Ching-Yun Ko; Dhaval Patel; Shuxin Lin; Nianjun Zhou; Jianxi Gao; Pin-Yu Chen; Shaowu Pan
>
> **摘要:** Large language model (LLM)-based systems are becoming increasingly popular for solving tasks by constructing executable workflows that interleave LLM calls, information retrieval, tool use, code execution, memory updates, and verification. This survey reviews recent methods for designing and optimizing such workflows, which we treat as agentic computation graphs (ACGs). We organize the literature based on when workflow structure is determined, where structure refers to which components or agents are present, how they depend on each other, and how information flows between them. This lens distinguishes static methods, which fix a reusable workflow scaffold before deployment, from dynamic methods, which select, generate, or revise the workflow for a particular run before or during execution. We further organize prior work along three dimensions: when structure is determined, what part of the workflow is optimized, and which evaluation signals guide optimization (e.g., task metrics, verifier signals, preferences, or trace-derived feedback). We also distinguish reusable workflow templates, run-specific realized graphs, and execution traces, separating reusable design choices from the structures actually deployed in a given run and from realized runtime behavior. Finally, we outline a structure-aware evaluation perspective that complements downstream task metrics with graph-level properties, execution cost, robustness, and structural variation across inputs. Our goal is to provide a clear vocabulary, a unified framework for positioning new methods, a more comparable view of existing body of literature, and a more reproducible evaluation standard for future work in workflow optimizations for LLM agents.
>
---
#### [new 069] Instruction-Tuned, but Not More Verifiable Instruction-Following: A Cross-Task Diagnosis for LoRA Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究LoRA适配器在不同任务中的实际性能与名义标签的匹配情况，揭示了“能力漂移”现象，强调需跨任务评估以避免依赖标签误判。**

- **链接: [https://arxiv.org/pdf/2603.22379](https://arxiv.org/pdf/2603.22379)**

> **作者:** Junyi Zou
>
> **备注:** 12 pages, 5 figures, 6 tables
>
> **摘要:** Adapters are often selected and deployed based on nominal labels (e.g., instruction-tuned), which implicitly suggest what capability improves after adaptation. We test whether nominal training objectives reliably align with realized cross-task capability gains by evaluating the same LoRA adapter across tasks. Our strongest evidence is tied to strict, automatically verifiable instruction following as measured by IFEval: across multiple seeds, base models, and LoRA settings, nominal labels recurrently but not universally fail to predict improvements on this verifiable target, with clear configuration sensitivity including a near-zero or negative case. As an illustrative strongest-case example in a controlled instruction-versus-numeric setting, an instruction-tuned adapter substantially improves off-target NM-based numeric benchmark performance from 0.133 to 0.632 while not improving verifiable instruction following on IFEval (ILA: 0.313 to 0.271; PLA: 0.250 to 0.143; values rounded to three decimals). We refer to this nominal-versus-realized mismatch pattern as capability drift as a descriptive label. The mismatch is visible in the raw cross-task performance matrix; we use a drift score only as a compact summary in the same units as the underlying metrics, not as a new formal metric contribution. Evidence from broader instruction-following benchmarks is benchmark-dependent and mixed, reflecting heterogeneity in how instruction following is operationalized; we therefore do not treat cross-benchmark agreement as a premise. Overall, the practical takeaway is to perform routine cross-task evaluation before deployment and to avoid treating nominal labels as reliable capability proxies.
>
---
#### [new 070] The Evolution of Tool Use in LLM Agents: From Single-Tool Call to Multi-Tool Orchestration
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于多工具代理研究，解决长轨迹下多工具协同问题。综述最新进展，分析六方面核心维度，总结应用并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2603.22862](https://arxiv.org/pdf/2603.22862)**

> **作者:** Haoyuan Xu; Chang Li; Xinyan Ma; Xianhao Ou; Zihan Zhang; Tao He; Xiangyu Liu; Zixiang Wang; Jiafeng Liang; Zheng Chu; Runxuan Liu; Rongchuan Mu; Ming Liu; Bing Qin
>
> **摘要:** Tool use enables large language models (LLMs) to access external information, invoke software systems, and act in digital environments beyond what can be solved from model parameters alone. Early research mainly studied whether a model could select and execute a correct single tool call. As agent systems evolve, however, the central problem has shifted from isolated invocation to multi-tool orchestration over long trajectories with intermediate state, execution feedback, changing environments, and practical constraints such as safety, cost, and verifiability. We comprehensively review recent progress in multi-tool LLM agents and analyzes the state of the art in this rapidly developing area. First, we unify task formulations and distinguish single-call tool use from long-horizon orchestration. Then, we organize the literature around six core dimensions: inference-time planning and execution, training and trajectory construction, safety and control, efficiency under resource constraints, capability completeness in open environments, and benchmark design and evaluation. We further summarize representative applications in software engineering, enterprise workflows, graphical user interfaces, and mobile systems. Finally, we discuss major challenges and outline future directions for building reliable, scalable, and verifiable multi-tool agents.
>
---
#### [new 071] T-MAP: Red-Teaming LLM Agents with Trajectory-aware Evolutionary Search
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM代理在多步骤工具执行中的漏洞问题。提出T-MAP方法，通过轨迹引导搜索生成有效攻击，提升有害目标实现率。**

- **链接: [https://arxiv.org/pdf/2603.22341](https://arxiv.org/pdf/2603.22341)**

> **作者:** Hyomin Lee; Sangwoo Park; Yumin Choi; Sohyun An; Seanie Lee; Sung Ju Hwang
>
> **摘要:** While prior red-teaming efforts have focused on eliciting harmful text outputs from large language models (LLMs), such approaches fail to capture agent-specific vulnerabilities that emerge through multi-step tool execution, particularly in rapidly growing ecosystems such as the Model Context Protocol (MCP). To address this gap, we propose a trajectory-aware evolutionary search method, T-MAP, which leverages execution trajectories to guide the discovery of adversarial prompts. Our approach enables the automatic generation of attacks that not only bypass safety guardrails but also reliably realize harmful objectives through actual tool interactions. Empirical evaluations across diverse MCP environments demonstrate that T-MAP substantially outperforms baselines in attack realization rate (ARR) and remains effective against frontier models, including GPT-5.2, Gemini-3-Pro, Qwen3.5, and GLM-5, thereby revealing previously underexplored vulnerabilities in autonomous LLM agents.
>
---
#### [new 072] Understanding LLM Performance Degradation in Multi-Instance Processing: The Roles of Instance Count and Context Length
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多实例处理任务中LLM性能退化问题，分析实例数量和上下文长度的影响。**

- **链接: [https://arxiv.org/pdf/2603.22608](https://arxiv.org/pdf/2603.22608)**

> **作者:** Jingxuan Chen; Mohammad Taher Pilehvar; Jose Camacho-Collados
>
> **摘要:** Users often rely on Large Language Models (LLMs) for processing multiple documents or performing analysis over a number of instances. For example, analysing the overall sentiment of a number of movie reviews requires an LLM to process the sentiment of each review individually in order to provide a final aggregated answer. While LLM performance on such individual tasks is generally high, there has been little research on how LLMs perform when dealing with multi-instance inputs. In this paper, we perform a comprehensive evaluation of the multi-instance processing (MIP) ability of LLMs for tasks in which they excel individually. The results show that all LLMs follow a pattern of slight performance degradation for small numbers of instances (approximately 20-100), followed by a performance collapse on larger instance counts. Crucially, our analysis shows that while context length is associated with this degradation, the number of instances has a stronger effect on the final results. This finding suggests that when optimising LLM performance for MIP, attention should be paid to both context length and, in particular, instance count.
>
---
#### [new 073] Generating and Evaluating Sustainable Procurement Criteria for the Swiss Public Sector using In-Context Prompting with Large Language Models
- **分类: cs.SE; cs.CL**

- **简介: 论文提出一种基于大语言模型的可持续采购标准生成系统，解决手动制定采购标准耗时易错的问题，通过自动化流程提升效率与一致性。**

- **链接: [https://arxiv.org/pdf/2603.22513](https://arxiv.org/pdf/2603.22513)**

> **作者:** Yingqiang Gao; Veton Matoshi; Luca Rolshoven; Tilia Ellendorff; Judith Binder; Jeremy Austin Jann; Gerold Schneider; Matthias Stürmer
>
> **摘要:** Public procurement refers to the process by which public sector institutions, such as governments, municipalities, and publicly funded bodies, acquire goods and services. Swiss law requires the integration of ecological, social, and economic sustainability requirements into tender evaluations in the format of criteria that have to be fulfilled by a bidder. However, translating high-level sustainability regulations into concrete, verifiable, and sector-specific procurement criteria (such as selection criteria, award criteria, and technical specifications) remains a labor-intensive and error-prone manual task, requiring substantial domain expertise in several groups of goods and services and considerable manual effort. This paper presents a configurable, LLM-assisted pipeline that is presented as a software supporting the systematic generation and evaluation of sustainability-oriented procurement criteria catalogs for Switzerland. The system integrates in-context prompting, interchangeable LLM backends, and automated output validation to enable auditable criteria generation across different procurement sectors. As a proof of concept, we instantiate the pipeline using official sustainability guidelines published by the Swiss government and the European Commission, which are ingested as structured reference documents. We evaluate the system through a combination of automated quality checks, including an LLM-based evaluation component, and expert comparison against a manually curated gold standard. Our results demonstrate that the proposed pipeline can substantially reduce manual drafting effort while producing criteria catalogs that are consistent with official guidelines. We further discuss system limitations, failure modes, and design trade-offs observed during deployment, highlighting key considerations for integrating generative AI into public sector software workflows.
>
---
#### [new 074] Natural Language Interfaces for Spatial and Temporal Databases: A Comprehensive Overview of Methods, Taxonomy, and Future Directions
- **分类: cs.DB; cs.AI; cs.CL**

- **简介: 该论文属于自然语言接口任务，旨在解决如何有效查询地理时空数据库的问题。通过综述现有方法、数据集和评估指标，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2603.23375](https://arxiv.org/pdf/2603.23375)**

> **作者:** Samya Acharja; Kanchan Chowdhury
>
> **摘要:** The task of building a natural language interface to a database, known as NLIDB, has recently gained significant attention from both the database and Natural Language Processing (NLP) communities. With the proliferation of geospatial datasets driven by the rapid emergence of location-aware sensors, geospatial databases play a vital role in supporting geospatial applications. However, querying geospatial and temporal databases differs substantially from querying traditional relational databases due to the presence of geospatial topological operators and temporal operators. To bridge the gap between geospatial query languages and non-expert users, the geospatial research community has increasingly focused on developing NLIDBs for geospatial databases. Yet, existing research remains fragmented across systems, datasets, and methodological choices, making it difficult to clearly understand the landscape of existing methods, their strengths and weaknesses, and opportunities for future research. Existing surveys on NLIDBs focus on general-purpose database systems and do not treat geospatial and temporal databases as primary focus for analysis. To address this gap, this paper presents a comprehensive survey of studies on NLIDBs for geospatial and temporal databases. Specifically, we provide a detailed overview of datasets, evaluation metrics, and the taxonomy of the methods for geospatial and temporal NLIDBs, as well as a comparative analysis of the existing methods. Our survey reveals recurring trends in existing methods, substantial variation in datasets and evaluation practices, and several open challenges that continue to hinder progress in this area. Based on these findings, we identify promising directions for future research to advance natural language interfaces to geospatial and temporal databases.
>
---
#### [new 075] Beyond Theoretical Bounds: Empirical Privacy Loss Calibration for Text Rewriting Under Local Differential Privacy
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，解决LDP下文本重写机制的隐私损失校准问题。通过实证方法评估不同机制的区分度，提升隐私与效用权衡的比较性。**

- **链接: [https://arxiv.org/pdf/2603.22968](https://arxiv.org/pdf/2603.22968)**

> **作者:** Weijun Li; Arnaud Grivet Sébert; Qiongkai Xu; Annabelle McIver; Mark Dras
>
> **备注:** 22 pages, 11 figures, 5 tables
>
> **摘要:** The growing use of large language models has increased interest in sharing textual data in a privacy-preserving manner. One prominent line of work addresses this challenge through text rewriting under Local Differential Privacy (LDP), where input texts are locally obfuscated before release with formal privacy guarantees. These guarantees are typically expressed by a parameter $\varepsilon$ that upper bounds the worst-case privacy loss. However, nominal $\varepsilon$ values are often difficult to interpret and compare across mechanisms. In this work, we investigate how to empirically calibrate across text rewriting mechanisms under LDP. We propose TeDA, which formulates calibration via a hypothesis-testing framework that instantiates text distinguishability audits in both surface and embedding spaces, enabling empirical assessment of indistinguishability from privatized texts. Applying this calibration to several representative mechanisms, we demonstrate that similar nominal $\varepsilon$ bounds can imply very different levels of distinguishability. Empirical calibration thus provides a more comparable footing for evaluating privacy-utility trade-offs, as well as a practical tool for mechanism comparison and analysis in real-world LDP text rewriting deployments.
>
---
#### [new 076] Sparser, Faster, Lighter Transformer Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在降低大语言模型的计算成本。通过引入稀疏结构和优化计算方法，提升模型效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.23198](https://arxiv.org/pdf/2603.23198)**

> **作者:** Edoardo Cetin; Stefano Peluchetti; Emilio Castillo; Akira Naruse; Mana Murakami; Llion Jones
>
> **备注:** Code and checkpoints available at: this https URL
>
> **摘要:** Scaling autoregressive large language models (LLMs) has driven unprecedented progress but comes with vast computational costs. In this work, we tackle these costs by leveraging unstructured sparsity within an LLM's feedforward layers, the components accounting for most of the model parameters and execution FLOPs. To achieve this, we introduce a new sparse packing format and a set of CUDA kernels designed to seamlessly integrate with the optimized execution pipelines of modern GPUs, enabling efficient sparse computation during LLM inference and training. To substantiate our gains, we provide a quantitative study of LLM sparsity, demonstrating that simple L1 regularization can induce over 99% sparsity with negligible impact on downstream performance. When paired with our kernels, we show that these sparsity levels translate into substantial throughput, energy efficiency, and memory usage benefits that increase with model scale. We will release all code and kernels under an open-source license to promote adoption and accelerate research toward establishing sparsity as a practical axis for improving the efficiency and scalability of modern foundation models.
>
---
#### [new 077] Beyond Preset Identities: How Agents Form Stances and Boundaries in Generative Societies
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决语言模型在复杂情境中立场形成与身份协商的问题。通过混合方法框架，分析代理人的立场演变及信任机制，揭示其动态行为模式。**

- **链接: [https://arxiv.org/pdf/2603.23406](https://arxiv.org/pdf/2603.23406)**

> **作者:** Hanzhong Zhang; Siyang Song; Jindong Wang
>
> **备注:** 22 pages, 3 figures
>
> **摘要:** While large language models simulate social behaviors, their capacity for stable stance formation and identity negotiation during complex interventions remains unclear. To overcome the limitations of static evaluations, this paper proposes a novel mixed-methods framework combining computational virtual ethnography with quantitative socio-cognitive profiling. By embedding human researchers into generative multiagent communities, controlled discursive interventions are conducted to trace the evolution of collective cognition. To rigorously measure how agents internalize and react to these specific interventions, this paper formalizes three new metrics: Innate Value Bias (IVB), Persuasion Sensitivity, and Trust-Action Decoupling (TAD). Across multiple representative models, agents exhibit endogenous stances that override preset identities, consistently demonstrating an innate progressive bias (IVB > 0). When aligned with these stances, rational persuasion successfully shifts 90% of neutral agents while maintaining high trust. In contrast, conflicting emotional provocations induce a paradoxical 40.0% TAD rate in advanced models, which hypocritically alter stances despite reporting low trust. Smaller models contrastingly maintain a 0% TAD rate, strictly requiring trust for behavioral shifts. Furthermore, guided by shared stances, agents use language interactions to actively dismantle assigned power hierarchies and reconstruct self organized community boundaries. These findings expose the fragility of static prompt engineering, providing a methodological and quantitative foundation for dynamic alignment in human-agent hybrid societies. The official code is available at: this https URL
>
---
#### [new 078] From Instructions to Assistance: a Dataset Aligning Instruction Manuals with Assembly Videos for Evaluating Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态语言模型评估任务，旨在解决模型在技术任务中的辅助能力问题。通过构建M2AD数据集，评估模型对步骤推理、进度跟踪和手册引用的能力。**

- **链接: [https://arxiv.org/pdf/2603.22321](https://arxiv.org/pdf/2603.22321)**

> **作者:** Federico Toschi; Nicolò Brunello; Andrea Sassella; Vincenzo Scotti; Mark James Carman
>
> **摘要:** The recent advancements introduced by Large Language Models (LLMs) have transformed how Artificial Intelligence (AI) can support complex, real world tasks, pushing research outside the text boundaries towards multi modal contexts and leading to Multimodal Large Language Models (MLMs). Given the current adoption of LLM based assistants in solving technical or domain specific problems, the natural continuation of this trend is to extend the input domains of these assistants exploiting MLMs. Ideally, these MLMs should be used as real time assistants in procedural tasks, hopefully integrating a view of the environment where the user being assisted is, or even better sharing the same point of view via Virtual Reality (VR) or Augmented Reality (AR) supports, to reason over the same scenario the user is experiencing. With this work, we aim at evaluating the quality of currently openly available MLMs to provide this kind of assistance on technical tasks. To this end, we annotated a data set of furniture assembly with step by step labels and manual references: the Manual to Action Dataset (M2AD). We used this dataset to assess (1) to which extent the reasoning abilities of MLMs can be used to reduce the need for detailed labelling, allowing for more efficient, cost effective annotation practices, (2) whether MLMs are able to track the progression of assembly steps (3) and whether MLMs can refer correctly to the instruction manual pages. Our results showed that while some models understand procedural sequences, their performance is limited by architectural and hardware constraints, highlighting the need for multi image and interleaved text image reasoning.
>
---
#### [new 079] Benchmarking Multi-Agent LLM Architectures for Financial Document Processing: A Comparative Study of Orchestration Patterns, Cost-Accuracy Tradeoffs and Production Scaling Strategies
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究多智能体LLM架构在金融文档处理中的应用，解决生产部署中的架构选择问题。通过对比不同架构，评估其准确性、成本和效率，提供实际部署指导。**

- **链接: [https://arxiv.org/pdf/2603.22651](https://arxiv.org/pdf/2603.22651)**

> **作者:** Siddhant Kulkarni; Yukta Kulkarni
>
> **摘要:** The adoption of large language models (LLMs) for structured information extraction from financial documents has accelerated rapidly, yet production deployments face fundamental architectural decisions with limited empirical guidance. We present a systematic benchmark comparing four multi-agent orchestration architectures: sequential pipeline, parallel fan-out with merge, hierarchical supervisor-worker and reflexive self-correcting loop. These are evaluated across five frontier and open-weight LLMs on a corpus of 10,000 SEC filings (10-K, 10-Q and 8-K forms). Our evaluation spans 25 extraction field types covering governance structures, executive compensation and financial metrics, measured along five axes: field-level F1, document-level accuracy, end-to-end latency, cost per document and token efficiency. We find that reflexive architectures achieve the highest field-level F1 (0.943) but at 2.3x the cost of sequential baselines, while hierarchical architectures occupy the most favorable position on the cost-accuracy Pareto frontier (F1 0.921 at 1.4x cost). We further present ablation studies on semantic caching, model routing and adaptive retry strategies, demonstrating that hybrid configurations can recover 89\% of the reflexive architecture's accuracy gains at only 1.15x baseline cost. Our scaling analysis from 1K to 100K documents per day reveals non-obvious throughput-accuracy degradation curves that inform capacity planning. These findings provide actionable guidance for practitioners deploying multi-agent LLM systems in regulated financial environments.
>
---
#### [new 080] Can LLM Agents Generate Real-World Evidence? Evaluating Observational Studies in Medical Databases
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗数据分析任务，旨在评估LLM代理生成真实世界证据的能力。研究提出RWE-bench基准，测试LLM在真实数据库中执行观察性研究的全流程表现。**

- **链接: [https://arxiv.org/pdf/2603.22767](https://arxiv.org/pdf/2603.22767)**

> **作者:** Dubai Li; Yuxiang He; Yan Hu; Yu Tian; Jingsong Li
>
> **摘要:** Observational studies can yield clinically actionable evidence at scale, but executing them on real-world databases is open-ended and requires coherent decisions across cohort construction, analysis, and reporting. Prior evaluations of LLM agents emphasize isolated steps or single answers, missing the integrity and internal structure of the resulting evidence bundle. To address this gap, we introduce RWE-bench, a benchmark grounded in MIMIC-IV and derived from peer-reviewed observational studies. Each task provides the corresponding study protocol as the reference standard, requiring agents to execute experiments in a real database and iteratively generate tree-structured evidence bundles. We evaluate six LLMs (three open-source, three closed-source) under three agent scaffolds using both question-level correctness and end-to-end task metrics. Across 162 tasks, task success is low: the best agent reaches 39.9%, and the best open-source model reaches 30.4%. Agent scaffolds also matter substantially, causing over 30% variation in performance metrics. Furthermore, we implement an automated cohort evaluation method to rapidly localize errors and identify agent failure modes. Overall, the results highlight persistent limitations in agents' ability to produce end-to-end evidence bundles, and efficient validation remains an important direction for future work. Code and data are available at this https URL.
>
---
#### [new 081] Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态语言模型任务，旨在解决3D空间推理不足的问题。通过引入TRACE方法，生成文本化空间表示以提升空间问答准确性。**

- **链接: [https://arxiv.org/pdf/2603.23404](https://arxiv.org/pdf/2603.23404)**

> **作者:** Jiacheng Hua; Yishu Yin; Yuhang Wu; Tai Wang; Yifei Huang; Miao Liu
>
> **备注:** 26 pages, 6 figures
>
> **摘要:** Existing Multimodal Large Language Models (MLLMs) struggle with 3D spatial reasoning, as they fail to construct structured abstractions of the 3D environment depicted in video inputs. To bridge this gap, drawing inspiration from cognitive theories of allocentric spatial reasoning, we investigate how to enable MLLMs to model and reason over text-based spatial representations of video. Specifically, we introduce Textual Representation of Allocentric Context from Egocentric Video (TRACE), a prompting method that induces MLLMs to generate text-based representations of 3D environments as intermediate reasoning traces for more accurate spatial question answering. TRACE encodes meta-context, camera trajectories, and detailed object entities to support structured spatial reasoning over egocentric videos. Extensive experiments on VSI-Bench and OST-Bench demonstrate that TRACE yields notable and consistent improvements over prior prompting strategies across a diverse range of MLLM backbones, spanning different parameter scales and training schemas. We further present ablation studies to validate our design choices, along with detailed analyses that probe the bottlenecks of 3D spatial reasoning in MLLMs.
>
---
## 更新

#### [replaced 001] Smart Bilingual Focused Crawling of Parallel Documents
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于双语文本爬取任务，旨在解决传统爬虫效率低的问题。通过神经网络模型提升平行文档的发现效率。**

- **链接: [https://arxiv.org/pdf/2405.14779](https://arxiv.org/pdf/2405.14779)**

> **作者:** Cristian García-Romero; Miquel Esplà-Gomis; Felipe Sánchez-Martínez
>
> **备注:** Pre-Cambridge University Press publication version
>
> **摘要:** Crawling parallel texts -- texts that are mutual translations -- from the Internet is usually done following a brute-force approach: documents are massively downloaded in an unguided process, and only a fraction of them end up leading to actual parallel content. In this work we propose a smart crawling method that guides the crawl towards finding parallel content more rapidly. We follow a neural approach that consists in adapting a pre-trained multilingual language model based on the encoder of the Transformer architecture by fine-tuning it for two new tasks: inferring the language of a document from its Uniform Resource Locator (URL), and inferring whether a pair of URLs link to parallel documents. We evaluate both models in isolation and their integration into a crawling tool. The results demonstrate the individual effectiveness of both models, and highlight that their combination enables us to address a practical engineering challenge: the early discovery of parallel content during web crawling in a given language pair. This leads to a reduction in the amount of downloaded documents deemed useless, and yields a greater quantity of parallel documents compared to conventional crawling approaches.
>
---
#### [replaced 002] From Conflict to Consensus: Boosting Medical Reasoning via Multi-Round Agentic RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于医疗问答任务，旨在解决LLM在医疗领域中的幻觉和过时知识问题。提出MA-RAG框架，通过多轮代理RAG提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2603.03292](https://arxiv.org/pdf/2603.03292)**

> **作者:** Wenhao Wu; Zhentao Tang; Yafu Li; Shixiong Kai; Mingxuan Yuan; Chunlin Chen; Zhi Wang
>
> **备注:** 22 pages, 7 figures, 11 tables
>
> **摘要:** Large Language Models (LLMs) exhibit high reasoning capacity in medical question-answering, but their tendency to produce hallucinations and outdated knowledge poses critical risks in healthcare fields. While Retrieval-Augmented Generation (RAG) mitigates these issues, existing methods rely on noisy token-level signals and lack the multi-round refinement required for complex reasoning. In the paper, we propose MA-RAG (Multi-Round Agentic RAG), a framework that facilitates test-time scaling for complex medical reasoning by iteratively evolving both external evidence and internal reasoning history within an agentic refinement loop. At each round, the agent transforms semantic conflict among candidate responses into actionable queries to retrieve external evidence, while optimizing history reasoning traces to mitigate long-context degradation. MA-RAG extends the self-consistency principle by leveraging the lack of consistency as a proactive signal for multi-round agentic reasoning and retrieval, and mirrors a boosting mechanism that iteratively minimizes the residual error toward a stable, high-fidelity medical consensus. Extensive evaluations across 7 medical Q&A benchmarks show that MA-RAG consistently surpasses competitive inference-time scaling and RAG baselines, delivering substantial +6.8 points on average accuracy over the backbone model. Our code is available at this https URL.
>
---
#### [replaced 003] Efficient and High-Fidelity Omni Modality Retrieval
- **分类: cs.IR; cs.CL; cs.CV**

- **简介: 该论文属于多模态检索任务，旨在解决现有模型仅支持两种模态的问题。提出OmniRet模型，支持文本、视觉和音频三种模态，提升计算效率和表示精度。**

- **链接: [https://arxiv.org/pdf/2603.02098](https://arxiv.org/pdf/2603.02098)**

> **作者:** Chuong Huynh; Manh Luong; Abhinav Shrivastava
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Multimodal retrieval is the task of aggregating information from queries across heterogeneous modalities to retrieve desired targets. State-of-the-art multimodal retrieval models can understand complex queries, yet they are typically limited to two modalities: text and vision. This limitation impedes the development of universal retrieval systems capable of comprehending queries that combine more than two modalities. To advance toward this goal, we present OmniRet, the first retrieval model capable of handling complex, composed queries spanning three key modalities: text, vision, and audio. Our OmniRet model addresses two critical challenges for universal retrieval: computational efficiency and representation fidelity. First, feeding massive token sequences from modality-specific encoders to Large Language Models (LLMs) is computationally inefficient. We therefore introduce an attention-based resampling mechanism to generate compact, fixed-size representations from these sequences. Second, compressing rich omni-modal data into a single embedding vector inevitably causes information loss and discards fine-grained details. We propose Attention Sliced Wasserstein Pooling to preserve these fine-grained details, leading to improved omni-modal representations. OmniRet is trained on an aggregation of approximately 6 million query-target pairs spanning 30 datasets. We benchmark our model on 13 retrieval tasks and a MMEBv2 subset. Our model demonstrates significant improvements on composed query, audio and video retrieval tasks, while achieving on-par performance with state-of-the-art models on others. Furthermore, we curate a new Audio-Centric Multimodal Benchmark (ACM). This new benchmark introduces two critical, previously missing tasks-composed audio retrieval and audio-visual retrieval to more comprehensively evaluate a model's omni-modal embedding capacity.
>
---
#### [replaced 004] RedTopic: Toward Topic-Diverse Red Teaming of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于红队测试任务，旨在解决现有方法在主题多样性和适应性上的不足。提出RedTopic框架，生成多样化对抗提示，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2507.00026](https://arxiv.org/pdf/2507.00026)**

> **作者:** Jiale Ding; Xiang Zheng; Yutao Wu; Cong Wang; Wei-Bin Lee; Ling Pan; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** As large language models (LLMs) are increasingly deployed as black-box components in real-world applications, red teaming has become essential for identifying potential risks. It tests LLMs with adversarial prompts to uncover vulnerabilities and improve safety alignment. Ideally, effective red teaming should be adaptive to evolving LLM capabilities and explore a broad range of harmful topics. However, existing approaches face two limitations: 1) topic-based approaches rely on pre-collected harmful topics, limited in flexibility and adaptivity. 2) topic-free methods use reinforcement learning (RL), but they lack an explicit reward signal for exploration and tend to over-optimize a narrow objective, reducing topic diversity. To address these limitations, we propose RedTopic, a novel red teaming framework that generates topic-diverse adversarial prompts through a contextualized generation pipeline, an aggregate reward design, and a multi-objective RL training loop. Experiments show that RedTopic produces more effective and diverse adversarial prompts than existing methods, with notable improvements in integrated evaluation metrics. We believe RedTopic represents a step toward more adaptive and topic-diverse red teaming for large language models.
>
---
#### [replaced 005] Extracting and Following Paths for Robust Relational Reasoning with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于关系推理任务，旨在解决大语言模型在复杂推理中的不足。提出PoT框架，通过图提取、路径识别和推理三阶段提升推理能力，有效应对错误和歧义。**

- **链接: [https://arxiv.org/pdf/2412.17963](https://arxiv.org/pdf/2412.17963)**

> **作者:** Ge Zhang; Mohammad Ali Alomrani; Hongjian Gu; Jiaming Zhou; Yaochen Hu; Bin Wang; Qun Liu; Mark Coates; Yingxue Zhang; Jianye Hao
>
> **摘要:** Large language models (LLMs) possess vast semantic knowledge but often struggle with complex reasoning tasks, particularly in relational reasoning problems such as kinship or spatial reasoning. In this paper, we present Path-of-Thoughts (PoT), a novel framework for solving relation reasoning that decomposes the task into three key stages: graph extraction, path identification, and reasoning. Unlike previous approaches, PoT efficiently extracts a reasoning graph that identifies crucial entities, relations, and attributes within the context. Subsequently, PoT identifies query-relevant reasoning paths within the graph, facilitating downstream reasoning of potential answers. Experimental evaluations across four datasets of relational reasoning demonstrate that PoT surpasses state-of-the-art baselines by a significant margin (up to 21.3%) without requiring fine-tuning or extensive LLM calls. Furthermore, unlike prior neuro-symbolic methods, PoT exhibits improved resilience against LLM extraction errors and input ambiguity by leveraging the compositional nature of graphs.
>
---
#### [replaced 006] Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn Search Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决多轮搜索代理中奖励稀疏的问题。通过引入信息增益作为内在奖励，提升训练效果和数据效率。**

- **链接: [https://arxiv.org/pdf/2510.14967](https://arxiv.org/pdf/2510.14967)**

> **作者:** Guoqing Wang; Sunhao Dai; Guangze Ye; Zeyu Gan; Wei Yao; Yong Deng; Xiaofeng Wu; Zhenzhe Ying
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Large language model (LLM)-based agents are increasingly trained with reinforcement learning (RL) to enhance their ability to interact with external environments through tool use, particularly in search-based settings that require multi-turn reasoning and knowledge acquisition. However, existing approaches typically rely on outcome-based rewards that are only provided exclusively upon generating the final answer. This reward sparsity becomes particularly problematic in multi-turn settings, where long trajectories exacerbate three critical issues: (i) advantage collapse, where all rollouts receive identical rewards and provide no useful learning signals; (ii) lack of fine-grained credit assignment, where the correctness of intermediate turns is obscured, especially in long-horizon tasks; and (iii) poor sample efficiency, where each rollout yields only a single outcome signal, leading to low data utilization. In this paper, we propose Information Gain-based Policy Optimization (IGPO), a simple yet effective RL framework that provides dense and intrinsic supervision for multi-turn agent training. IGPO models each interaction turn as an incremental process of acquiring information about the ground truth, and defines turn-level rewards as the marginal increase in the policy's probability of producing the correct answer. Unlike prior process-level reward approaches that depend on external reward models or costly Monte Carlo estimation, IGPO derives intrinsic rewards directly from the model's own belief updates. These intrinsic turn-level rewards are combined with outcome-level supervision to form dense reward signals. Extensive experiments on both in-domain and out-of-domain benchmarks demonstrate that IGPO consistently outperforms strong baselines in multi-turn scenarios, achieving higher accuracy and improved data efficiency. Our code is available at this https URL.
>
---
#### [replaced 007] Measuring Faithfulness Depends on How You Measure: Classifier Sensitivity in LLM Chain-of-Thought Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型评估任务，探讨链式思维忠实度的测量问题。研究发现不同分类器对同一数据得出的忠实度差异显著，表明忠实度测量依赖于分类方法，需报告多种方法的敏感性范围。**

- **链接: [https://arxiv.org/pdf/2603.20172](https://arxiv.org/pdf/2603.20172)**

> **作者:** Richard J. Young
>
> **备注:** 14 pages, 4 figures, 5 tables
>
> **摘要:** Recent work on chain-of-thought (CoT) faithfulness reports single aggregate numbers (e.g., DeepSeek-R1 acknowledges hints 39% of the time), implying that faithfulness is an objective, measurable property of a model. This paper provides evidence that it is not. Three classifiers (a regex-only detector, a regex-plus-LLM pipeline, and a Claude Sonnet 4 judge) are applied to 10,276 influenced reasoning traces from 12 open-weight models spanning 9 families and 7B to 1T parameters. On identical data, these classifiers produce faithfulness rates of 74.4%, 82.6%, and 69.7%. Per-model gaps range from 2.6 to 30.6 percentage points; all pairwise McNemar tests are significant (p < 0.001). The disagreements are systematic: Cohen's kappa ranges from 0.06 ("slight") for sycophancy hints to 0.42 ("moderate") for grader hints, and the asymmetry is pronounced: for sycophancy, 883 cases are classified as faithful by the pipeline but unfaithful by the Sonnet judge, while only 2 go the other direction. Classifier choice can also reverse model rankings: Qwen3.5-27B ranks 1st under the pipeline but 7th under Sonnet; OLMo-3.1-32B moves from 9th to 3rd. Different classifiers operationalize faithfulness at different levels of stringency (lexical mention versus epistemic dependence), yielding divergent measurements on the same behavior. These results indicate that published faithfulness numbers cannot be meaningfully compared across studies using different classifiers, and that future evaluations should report sensitivity ranges across multiple classification methodologies.
>
---
#### [replaced 008] NLP Occupational Emergence Analysis: How Occupations Form and Evolve in Real Time -- A Zero-Assumption Method Demonstrated on AI in the US Technology Workforce, 2022-2026
- **分类: cs.CL; cs.CY**

- **简介: 论文提出一种零假设方法，分析职业的形成与演变，解决传统分类系统滞后的问题。通过简历数据检测职业词汇和群体凝聚力，发现AI未形成独立职业。**

- **链接: [https://arxiv.org/pdf/2603.15998](https://arxiv.org/pdf/2603.15998)**

> **作者:** David Nordfors
>
> **备注:** This manuscript has been withdrawn by the authors pending internal review and substantial revision
>
> **摘要:** Occupations form and evolve faster than classification systems can track. We propose that a genuine occupation is a self-reinforcing structure (a bipartite co-attractor) in which a shared professional vocabulary makes practitioners cohesive as a group, and the cohesive group sustains the vocabulary. This co-attractor concept enables a zero-assumption method for detecting occupational emergence from resume data, requiring no predefined taxonomy or job titles: we test vocabulary cohesion and population cohesion independently, with ablation to test whether the vocabulary is the mechanism binding the population. Applied to 8.2 million US resumes (2022-2026), the method correctly identifies established occupations and reveals a striking asymmetry for AI: a cohesive professional vocabulary formed rapidly in early 2024, but the practitioner population never cohered. The pre-existing AI community dissolved as the tools went mainstream, and the new vocabulary was absorbed into existing careers rather than binding a new occupation. AI appears to be a diffusing technology, not an emerging occupation. We discuss whether introducing an "AI Engineer" occupational category could catalyze population cohesion around the already-formed vocabulary, completing the co-attractor.
>
---
#### [replaced 009] Children's Intelligence Tests Pose Challenges for MLLMs? KidGym: A 2D Grid-Based Reasoning Benchmark for MLLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出KidGym，一个用于评估多模态大语言模型的2D网格基准任务，解决模型在执行、感知推理等能力上的评测问题。**

- **链接: [https://arxiv.org/pdf/2603.20209](https://arxiv.org/pdf/2603.20209)**

> **作者:** Hengwei Ye; Yuanting Guan; Yuxuan Ge; Tianying Zhu; Zhenhan Guan; Yijia Zhong; Yijing Zhang; Han Zhang; Yingna Wu; Zheng Tian
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Multimodal Large Language Models (MLLMs) combine the linguistic strengths of LLMs with the ability to process multimodal data, enbaling them to address a broader range of visual tasks. Because MLLMs aim at more general, human-like competence than language-only models, we take inspiration from the Wechsler Intelligence Scales - an established battery for evaluating children by decomposing intelligence into interpretable, testable abilities. We introduce KidGym, a comprehensive 2D grid-based benchmark for assessing five essential capabilities of MLLMs: Execution, Perception Reasoning, Learning, Memory and Planning. The benchmark comprises 12 unique tasks, each targeting at least one core capability, specifically designed to guage MLLMs' adaptability and developmental potential, mirroring the stages of children's cognitive growth. Additionally, our tasks encompass diverse scenarios and objects with randomly generated layouts, ensuring a more accurate and robust evluation of MLLM capabilities. KidGym is designed to be fully user-customizable and extensible, allowing researchers to create new evaluation scenarios and adjust difficuly levels to accommodate the rapidly growing MLLM community. Through the evaluation of state-of-the-art MLLMs using KidGym, we identified significant insights into model capabilities and revealed several limitations of current models. We release our benchmark at: this https URL.
>
---
#### [replaced 010] KDFlow: A User-Friendly and Efficient Knowledge Distillation Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型压缩任务，旨在解决知识蒸馏中训练与推理效率不匹配的问题。提出KDFlow框架，实现高效蒸馏。**

- **链接: [https://arxiv.org/pdf/2603.01875](https://arxiv.org/pdf/2603.01875)**

> **作者:** Songming Zhang; Xue Zhang; Tong Zhang; Bojie Hu; Yufeng Chen; Jinan Xu
>
> **备注:** 8 pages, 4 figures, 3 tables, code is available at: this https URL
>
> **摘要:** Knowledge distillation (KD) is an essential technique to compress large language models (LLMs) into smaller ones. However, despite the distinct roles of the student model and the teacher model in KD, most existing frameworks still use a homogeneous training backend (e.g., FSDP and DeepSpeed) for both models, leading to suboptimal training efficiency. In this paper, we present a novel framework for LLM distillation, termed \textbf{KDFlow}, which features a decoupled architecture and employs SGLang for teacher inference. By bridging the training efficiency of FSDP2 and the inference efficiency of SGLang, KDFlow achieves full utilization of both advantages in a unified system. Moreover, instead of transferring full logits across different processes, our framework only transmits the teacher's hidden states using zero-copy data transfer and recomputes the logits on the student side, effectively balancing the communication cost and KD performance. Furthermore, our framework supports both off-policy and on-policy distillation and incorporates KD algorithms for cross-tokenizer KD through highly extensible and user-friendly APIs. Experiments show that KDFlow can achieve \textbf{1.44$\times$ to 6.36$\times$} speedup compared to current KD frameworks, enabling researchers to rapidly prototype and scale LLM distillation with minimal engineering overhead. Code is available at: this https URL
>
---
#### [replaced 011] PaperBanana: Automating Academic Illustration for AI Scientists
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于学术插图生成任务，旨在解决科研中手工制作高质量图表耗时的问题。提出PaperBanana框架，自动完成图表生成与优化。**

- **链接: [https://arxiv.org/pdf/2601.23265](https://arxiv.org/pdf/2601.23265)**

> **作者:** Dawei Zhu; Rui Meng; Yale Song; Xiyu Wei; Sujian Li; Tomas Pfister; Jinsung Yoon
>
> **备注:** Add Citations
>
> **摘要:** Despite rapid advances in autonomous AI scientists powered by language models, generating publication-ready illustrations remains a labor-intensive bottleneck in the research workflow. To lift this burden, we introduce PaperBanana, an agentic framework for automated generation of publication-ready academic illustrations. Powered by state-of-the-art VLMs and image generation models, PaperBanana orchestrates specialized agents to retrieve references, plan content and style, render images, and iteratively refine via self-critique. To rigorously evaluate our framework, we introduce PaperBananaBench, comprising 292 test cases for methodology diagrams curated from NeurIPS 2025 publications, covering diverse research domains and illustration styles. Comprehensive experiments demonstrate that PaperBanana consistently outperforms leading baselines in faithfulness, conciseness, readability, and aesthetics. We further show that our method effectively extends to the generation of high-quality statistical plots. Collectively, PaperBanana paves the way for the automated generation of publication-ready illustrations.
>
---
#### [replaced 012] Happiness is Sharing a Vocabulary: A Study of Transliteration Methods
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言NLP中音译方法对模型性能的影响，旨在提升非拉丁语系语言的处理效果。通过实验对比不同音译方式，发现罗马化表现最佳。**

- **链接: [https://arxiv.org/pdf/2510.10827](https://arxiv.org/pdf/2510.10827)**

> **作者:** Haeji Jung; Jinju Kim; Kyungjin Kim; Youjeong Roh; David R. Mortensen
>
> **备注:** Accepted to EACL 2026
>
> **摘要:** Transliteration has emerged as a promising means to bridge the gap between various languages in multilingual NLP, showing promising results especially for languages using non-Latin scripts. We investigate the degree to which shared script, overlapping token vocabularies, and shared phonology contribute to performance of multilingual models. To this end, we conduct controlled experiments using three kinds of transliteration (romanization, phonemic transcription, and substitution ciphers) as well as orthography. We evaluate each model on three downstream tasks -- named entity recognition (NER), part-of-speech tagging (POS) and natural language inference (NLI) -- and find that romanization significantly outperforms other input types in 11 out of 12 evaluation settings, largely consistent with our hypothesis that it is the most effective approach. We further analyze how each factor contributed to the success, and suggest that having longer (subword) tokens shared with pre-trained languages leads to better utilization of the model.
>
---
#### [replaced 013] Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全任务，研究LLMs在对抗性中间人攻击下的事实记忆问题。通过Xmera框架进行输入扰动，评估响应正确性和不确定性，并提出基于随机森林的防御机制。**

- **链接: [https://arxiv.org/pdf/2511.05919](https://arxiv.org/pdf/2511.05919)**

> **作者:** Alina Fastowski; Bardh Prenkaj; Yuxiao Li; Gjergji Kasneci
>
> **摘要:** LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~94.8%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.
>
---
#### [replaced 014] TimeTox: An LLM-Based Pipeline for Automated Extraction of Time Toxicity from Clinical Trial Protocols
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出TimeTox，用于自动化提取临床试验方案中的时间毒性数据。解决的是手动提取耗时的问题，通过LLM实现高效准确的量化分析。**

- **链接: [https://arxiv.org/pdf/2603.21335](https://arxiv.org/pdf/2603.21335)**

> **作者:** Saketh Vinjamuri; Marielle Fis Loperena; Marie C. Spezia; Ramez Kouzy
>
> **备注:** 19 pages, 5 figures, 7 tables
>
> **摘要:** Time toxicity, the cumulative healthcare contact days from clinical trial participation, is an important but labor-intensive metric to extract from protocol documents. We developed TimeTox, an LLM-based pipeline for automated extraction of time toxicity from Schedule of Assessments tables. TimeTox uses Google's Gemini models in three stages: summary extraction from full-length protocol PDFs, time toxicity quantification at six cumulative timepoints for each treatment arm, and multi-run consensus via position-based arm matching. We validated against 20 synthetic schedules (240 comparisons) and assessed reproducibility on 644 real-world oncology protocols. Two architectures were compared: single-pass (vanilla) and two-stage (structure-then-count). The two-stage pipeline achieved 100% clinically acceptable accuracy ($\pm$3 days) on synthetic data (MAE 0.81 days) versus 41.5% for vanilla (MAE 9.0 days). However, on real-world protocols, the vanilla pipeline showed superior reproducibility: 95.3% clinically acceptable accuracy (IQR $\leq$ 3 days) across 3 runs on 644 protocols, with 82.0% perfect stability (IQR = 0). The production pipeline extracted time toxicity for 1,288 treatment arms across multiple disease sites. Extraction stability on real-world data, rather than accuracy on synthetic benchmarks, is the decisive factor for production LLM deployment.
>
---
#### [replaced 015] Flying Pigs, FaR and Beyond: Evaluating LLM Reasoning in Counterfactual Worlds
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的推理任务，研究LLM在反事实场景下的逻辑推理能力。针对模型在知识冲突时表现下降的问题，提出Flag & Reason方法提升其可靠性。**

- **链接: [https://arxiv.org/pdf/2505.22318](https://arxiv.org/pdf/2505.22318)**

> **作者:** Anish R Joishy; Ishwar B Balappanawar; Vamshi Krishna Bonagiri; Manas Gaur; Krishnaprasad Thirunarayan; Ponnurangam Kumaraguru
>
> **摘要:** A fundamental challenge in reasoning is navigating hypothetical, counterfactual worlds where logic may conflict with ingrained knowledge. We investigate this frontier for Large Language Models (LLMs) by asking: Can LLMs reason logically when the context contradicts their parametric knowledge? To facilitate a systematic analysis, we first introduce CounterLogic, a benchmark specifically designed to disentangle logical validity from knowledge alignment. Evaluation of 11 LLMs across six diverse reasoning datasets reveals a consistent failure: model accuracy plummets by an average of 14% in counterfactual scenarios compared to knowledge-aligned ones. We hypothesize that this gap stems not from a flaw in logical processing, but from an inability to manage the cognitive conflict between context and knowledge. Inspired by human metacognition, we propose a simple yet powerful intervention: Flag & Reason (FaR), where models are first prompted to flag potential knowledge conflicts before they reason. This metacognitive step is highly effective, narrowing the performance gap to just 7% and increasing overall accuracy by 4%. Our findings diagnose and study a critical limitation in modern LLMs' reasoning and demonstrate how metacognitive awareness can make them more robust and reliable thinkers.
>
---
#### [replaced 016] Collaborative Evaluation of Deepfake Text with Deliberation-Enhancing Dialogue Systems
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于深度伪造文本检测任务，旨在提升群体识别深度伪造文本的准确性。研究提出DeepFakeDeLiBot，通过增强对话系统促进群体协作，改善检测效果与互动效率。**

- **链接: [https://arxiv.org/pdf/2503.04945](https://arxiv.org/pdf/2503.04945)**

> **作者:** Jooyoung Lee; Xiaochen Zhu; Georgi Karadzhov; Tom Stafford; Andreas Vlachos; Dongwon Lee
>
> **备注:** 15; To appear in ICWSM 2026 (this https URL)
>
> **摘要:** The proliferation of generative models has presented significant challenges in distinguishing authentic human-authored content from deepfake content. Collaborative human efforts, augmented by AI tools, present a promising solution. In this study, we explore the potential of DeepFakeDeLiBot, a deliberation-enhancing chatbot, to support groups in detecting deepfake text. Our findings reveal that group-based problem-solving significantly improves the accuracy of identifying machine-generated paragraphs compared to individual efforts. While engagement with DeepFakeDeLiBot does not yield substantial performance gains overall, it enhances group dynamics by fostering greater participant engagement, consensus building, and the frequency and diversity of reasoning-based utterances. Additionally, participants with higher perceived effectiveness of group collaboration exhibited performance benefits from DeepFakeDeLiBot. These findings underscore the potential of deliberative chatbots in fostering interactive and productive group dynamics while ensuring accuracy in collaborative deepfake text detection. \textit{Dataset and source code used in this study will be made publicly available upon acceptance of the manuscript.
>
---
#### [replaced 017] When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文研究音频与文本冲突时语言模型的模态选择问题，属于多模态融合任务。通过构建数据集和指标，分析模型对音频的依赖程度，揭示文本主导现象及其影响因素。**

- **链接: [https://arxiv.org/pdf/2602.11488](https://arxiv.org/pdf/2602.11488)**

> **作者:** Jayadev Billa
>
> **备注:** 13 pages, 18 tables, 4 figures, benchmark and code at this https URL
>
> **摘要:** When audio and text conflict, speech-enabled language models follow text far more often than they do when arbitrating between two conflicting text sources, even under explicit instructions to trust the audio. We introduce ALME (Audio-LLM Modality Evaluation), a dataset of 57,602 controlled audio-text conflict stimuli across eight languages, together with Text Dominance Ratio (TDR), which measures how often a model follows conflicting text when instructed to follow audio. Gemini 2.0 Flash and GPT-4o show TDR 10--26$\times$ higher than a baseline that replaces audio with its transcript under otherwise identical conditions (Gemini 2.0 Flash: 16.6% vs. 1.6%; GPT-4o: 23.2% vs. 0.9%). These results suggest that text dominance reflects not only information content, but also an asymmetry in arbitration accessibility, i.e., how easily the model can use competing representations at decision time. Framing the transcript as deliberately corrupted reduces TDR by 80%, whereas forcing explicit transcription increases it by 14%. A fine-tuning ablation further suggests that arbitration behavior depends more on LLM reasoning than on the audio input path alone. Across four audio-LLMs, we observe the same qualitative pattern with substantial cross-model and cross-linguistic variation.
>
---
#### [replaced 018] LatentQA: Teaching LLMs to Decode Activations Into Natural Language
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文提出LatentQA任务，解决语言模型激活信息难以解释的问题。通过生成数据集并训练解码器，将激活转化为自然语言，提升模型透明度与可控性。**

- **链接: [https://arxiv.org/pdf/2412.08686](https://arxiv.org/pdf/2412.08686)**

> **作者:** Alexander Pan; Lijie Chen; Jacob Steinhardt
>
> **备注:** ICLR 2026; project page at this https URL
>
> **摘要:** Top-down transparency typically analyzes language model activations using probes with scalar or single-token outputs, limiting the range of behaviors that can be captured. To alleviate this issue, we develop a more expressive probe that can directly output natural language, performing LatentQA: the task of answering open-ended questions about activations. A key difficulty in developing such a probe is collecting a dataset mapping activations to natural-language descriptions. In response, we propose an approach for generating a dataset of activations and associated question-answer pairs and develop a fine-tuning method for training a decoder LLM on this dataset. We then validate our decoder's fidelity by assessing its ability to read and control model activations. First, we evaluate the decoder on a number of supervised reading tasks with a known answer, such as uncovering hidden system prompts and relational knowledge extraction, and observe that it outperforms competitive probing baselines. Second, we demonstrate that the decoder is precise enough to steer the target model to exhibit behaviors unseen during training. Finally, we show that LatentQA scales well with increasing dataset and model size.
>
---
#### [replaced 019] Designing Explainable Conversational Agentic Systems for Guaraní Speakers
- **分类: cs.CL**

- **简介: 该论文属于人工智能与人机交互领域，旨在解决AI系统对口语语言支持不足的问题。通过构建以口语为核心的多智能体架构，提升原住民语言的数字包容性。**

- **链接: [https://arxiv.org/pdf/2603.05743](https://arxiv.org/pdf/2603.05743)**

> **作者:** Samantha Adorno; Akshata Kishore Moharir; Ratna Kandala
>
> **备注:** Accepted at HCXAI conference, ACM CHI 2026
>
> **摘要:** Although artificial intelligence (AI) and Human-Computer Interaction (HCI) systems are often presented as universal solutions, their design remains predominantly text-first, underserving primarily oral languages and indigenous communities. This position paper uses Guaraní, an official and widely spoken language of Paraguay, as a case study to argue that language support in AI remains insufficient unless it aligns with lived oral practices. We propose an alternative to the standard "text-to-speech" pipeline, proposing instead an oral-first multi-agent architecture. By decoupling Guaraní natural language understanding from dedicated agents for conversation state and community-led governance, we demonstrate a technical framework that respects indigenous data sovereignty and diglossia. Our work moves beyond mere recognition to focus on turn-taking, repair, and shared context as the primary locus of interaction. We conclude that for AI to be truly culturally grounded, it must shift from adapting oral languages to text-centric systems to treating spoken conversation as a first-class design requirement, ensuring digital ecosystems empower rather than overlook diverse linguistic practices.
>
---
#### [replaced 020] GeneMamba: An Efficient and Effective Foundation Model on Single Cell Data
- **分类: cs.CL; cs.LG; q-bio.GN**

- **简介: 该论文提出GeneMamba，用于单细胞数据的高效分析。解决高维、稀疏和批次效应带来的计算挑战，通过状态空间建模实现线性复杂度，提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2504.16956](https://arxiv.org/pdf/2504.16956)**

> **作者:** Cong Qi; Hanzhang Fang; Siqi Jiang; Xun Song; Tianxing Hu; Wei Zhi
>
> **摘要:** Single-cell RNA sequencing (scRNA-seq) enables high-resolution analysis of cellular heterogeneity, but its complexity, which is marked by high dimensionality, sparsity, and batch effects, which poses major computational challenges. Transformer-based models have made significant advances in this domain but are often limited by their quadratic complexity and suboptimal handling of long-range dependencies. In this work, we introduce GeneMamba, a scalable and efficient foundation model for single-cell transcriptomics built on state space modeling. Leveraging the Bi-Mamba architecture, GeneMamba captures bidirectional gene context with linear-time complexity, offering substantial computational gains over transformer baselines. The model is pretrained on nearly 30 million cells and incorporates biologically informed objectives, including pathway-aware contrastive loss and rank-based gene encoding. We evaluate GeneMamba across diverse tasks, including multi-batch integration, cell type annotation, and gene-gene correlation, demonstrating strong performance, interpretability, and robustness. These results position GeneMamba as a practical and powerful alternative to transformer-based methods, advancing the development of biologically grounded, scalable tools for large-scale single-cell data analysis.
>
---
#### [replaced 021] HUMORCHAIN: Theory-Guided Multi-Stage Reasoning for Interpretable Multimodal Humor Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态幽默生成任务，旨在解决AI生成幽默缺乏认知机制的问题。提出HUMORCHAIN框架，结合幽默理论与多阶段推理，提升生成内容的幽默感和可解释性。**

- **链接: [https://arxiv.org/pdf/2511.21732](https://arxiv.org/pdf/2511.21732)**

> **作者:** Jiajun Zhang; Shijia Luo; Ruikang Zhang; Qi Su
>
> **摘要:** Humor, as both a creative human activity and a social binding mechanism, has long posed a major challenge for AI generation. Although producing humor requires complex cognitive reasoning and social understanding, theories of humor suggest that it follows learnable patterns and structures, making it theoretically possible for generative models to acquire them implicitly. In recent years, multimodal humor has become a prevalent form of online communication, especially among Gen Z, highlighting the need for AI systems capable of integrating visual understanding with humorous language generation. However, existing data-driven approaches lack explicit modeling or theoretical grounding of humor, often producing literal descriptions that fail to capture its underlying cognitive mechanisms, resulting in the generated image descriptions that are fluent but lack genuine humor or cognitive depth. To address this limitation, we propose HUMORCHAIN (HUmor-guided Multi-step Orchestrated Reasoning Chain for Image Captioning), a theory-guided multi-stage reasoning framework. It integrates visual semantic parsing, humor- and psychology-based reasoning, and a fine-tuned discriminator for humor evaluation, forming an interpretable and controllable cognitive reasoning chain. To the best of our knowledge, this is the first work to explicitly embed cognitive structures from humor theories into multimodal humor generation, enabling a structured reasoning process from visual understanding to humor creation. Experiments on Meme-Image-No-Text, Oogiri-GO, and OxfordTVG-HIC datasets show that HUMORCHAIN outperforms state-of-the-art baselines in human humor preference, Elo/BT scores, and semantic diversity, demonstrating that theory-driven structured reasoning enables large language models to generate humor aligned with human perception.
>
---
#### [replaced 022] DualEdit: Mitigating Safety Fallback in LLM Backdoor Editing via Affirmation-Refusal Regulation
- **分类: cs.CL**

- **简介: 该论文属于模型安全任务，解决LLM后门攻击中的安全回退问题。通过DualEdit框架，同时促进肯定词和抑制拒绝词，提升攻击效果并降低安全回退率。**

- **链接: [https://arxiv.org/pdf/2506.13285](https://arxiv.org/pdf/2506.13285)**

> **作者:** Houcheng Jiang; Zetong Zhao; Junfeng Fang; Haokai Ma; Ruipeng Wang; Xiang Wang; Xiangnan He; Yang Deng
>
> **摘要:** Safety-aligned large language models (LLMs) remain vulnerable to backdoor attacks. Recent model editing-based approaches enable efficient backdoor injection by directly modifying a small set of parameters to map triggers to attacker-desired behaviors. However, we find that existing editing-based attacks are often unstable under safety alignment: the edited model may start with an affirmative prefix but later revert to refusals during generation. We term this phenomenon safety fallback. To mitigate it, we propose DualEdit, a dual-objective model editing framework that simultaneously promotes affirmative tokens and suppresses refusal tokens. DualEdit further addresses two key challenges, objective imbalance and refusal diversity, via two complementary techniques: (1) dynamic loss weighting, which calibrates the relative scales of the two objectives using the pre-edited model to stabilize optimization, and (2) value anchoring, which clusters representative attention value vectors to form compact anchors, reducing conflicts from overly diverse token sets and improving generalization. Experiments on safety-aligned LLMs show that DualEdit improves attack success by 10% and reduces safety fallback rate by 11% over baselines.
>
---
#### [replaced 023] In Generative AI We (Dis)Trust? Computational Analysis of Trust and Distrust in Reddit Discussions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的信任分析任务，旨在研究公众对生成式AI的信任与不信任。通过分析Reddit数据，探讨影响信任的因素及变化趋势。**

- **链接: [https://arxiv.org/pdf/2510.16173](https://arxiv.org/pdf/2510.16173)**

> **作者:** Aria Pessianzadeh; Naima Sultana; Hildegarde Van den Bulck; David Gefen; Shahin Jabbari; Rezvaneh Rezapour
>
> **摘要:** The rise of generative AI (GenAI) has impacted many aspects of human life. As these systems become embedded in everyday practices, understanding public trust in them is also essential for responsible adoption and governance. Prior work on trust in AI has largely drawn from psychology and human-computer interaction, but there is a lack of computational, large-scale, and longitudinal approaches to measuring trust and distrust in GenAI and large language models (LLMs). This paper presents the first computational study of trust and distrust in GenAI, using a multi-year Reddit dataset (2022--2025) spanning 39 subreddits and 230,576 posts. Crowd-sourced annotations of a representative sample were combined with classification models to scale analysis. We find that trust and distrust are nearly balanced over time, although trust modestly outweighs distrust, with shifts around major model releases. Technical performance and usability dominate as dimensions, while personal experience is the most frequent reason shaping attitudes. Distinct patterns also emerge across trustors (e.g., experts, ethicists, and general users). Our results provide a methodological framework for large-scale trust analysis and insights into evolving public perceptions of GenAI.
>
---
#### [replaced 024] Mi:dm K 2.5 Pro
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Mi:dm K 2.5 Pro，解决企业级多步骤推理与长上下文理解问题，通过优化数据和训练方法提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.18788](https://arxiv.org/pdf/2603.18788)**

> **作者:** KT Tech innovation Group
>
> **摘要:** The evolving LLM landscape requires capabilities beyond simple text generation, prioritizing multi-step reasoning, long-context understanding, and agentic workflows. This shift challenges existing models in enterprise environments, especially in Korean-language and domain-specific scenarios where scaling is insufficient. We introduce Mi:dm K 2.5 Pro, a 32B parameter flagship LLM designed to address enterprise-grade complexity through reasoning-focused optimization. Our methodology builds a robust data foundation via a quality-centric curation pipeline utilizing abstract syntax tree (AST) analysis for code, gap-filling synthesis for mathematics, and an LLM-based quality evaluator. Pre-training scales the model via layer-predictor-based Depth Upscaling (DuS) and a progressive strategy supporting a 128K token context window. Post-training introduces a specialized multi-stage pipeline, including Reasoning SFT, model merging, and asynchronous reinforcement learning (RL), to develop complex problem-solving skills. "Fusion Training" then rebalances these capabilities with conversational fluency, consistent response styling, and reliable tool-use. The evaluations show that Mi:dm K 2.5 Pro achieves competitive performance against leading global and domestic models. In addition, it sets state-of-the-art results on Korean-specific benchmarks, showcasing deep linguistic and cultural understanding. Finally, Responsible AI evaluations validate safety against attacks, ensuring a secure profile for deployment with a balance of harmlessness and responsiveness.
>
---
#### [replaced 025] myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于缅甸手写数字识别任务，旨在建立基准测试。评估多种模型性能，提供可复现的基线，比较传统与新兴架构效果。**

- **链接: [https://arxiv.org/pdf/2603.18597](https://arxiv.org/pdf/2603.18597)**

> **作者:** Ye Kyaw Thu; Thazin Myint Oo; Thepchai Supnithi
>
> **备注:** 7 pages, 2 figures, 3 tables, Accepted to ICNLP 2026, Xi'an, China
>
> **摘要:** We present the first systematic benchmark on a standardized iteration of the publicly available Burmese Handwritten Digit Dataset (BHDD), which we have designated as myMNIST Benchmarking. While BHDD serves as a foundational resource for Myanmar NLP/AI, it lacks a comprehensive, reproducible performance baseline across modern architectures. We evaluate eleven architectures spanning classical deep learning models (Multi-Layer Perceptron, Convolutional Neural Network, Long Short-Term Memory, Gated Recurrent Unit, Transformer), recent alternatives (FastKAN, EfficientKAN), an energy-based model (JEM), and physics-inspired PETNN variants (Sigmoid, GELU, SiLU). Using Precision, Recall, F1-Score, and Accuracy as evaluation metrics, our results show that the CNN remains a strong baseline, achieving the best overall scores (F1 = 0.9959, Accuracy = 0.9970). The PETNN (GELU) model closely follows (F1 = 0.9955, Accuracy = 0.9966), outperforming LSTM, GRU, Transformer, and KAN variants. JEM, representing energy-based modeling, performs competitively (F1 = 0.9944, Accuracy = 0.9958). KAN-based models (FastKAN, EfficientKAN) trail the top performers but provide a meaningful alternative baseline (Accuracy ~0.992). These findings (i) establish reproducible baselines for BHDD across diverse modeling paradigms, (ii) highlight PETNN's strong performance relative to classical and Transformer-based models, and (iii) quantify the gap between energy-inspired PETNNs and a true energy-based model (JEM). We release this benchmark to facilitate future research on Myanmar digit recognition and to encourage broader evaluation of emerging architectures on regional scripts.
>
---
#### [replaced 026] CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，解决MDMs在语义连贯性和依赖建模上的不足。提出CRoCoDiL方法，通过连续语义空间提升生成质量与速度。**

- **链接: [https://arxiv.org/pdf/2603.20210](https://arxiv.org/pdf/2603.20210)**

> **作者:** Roy Uziel; Omer Belhasin; Itay Levi; Akhiad Bercovich; Ran El-Yaniv; Ran Zilberstein; Michael Elad
>
> **摘要:** Masked Diffusion Models (MDMs) provide an efficient non-causal alternative to autoregressive generation but often struggle with token dependencies and semantic incoherence due to their reliance on discrete marginal distributions. We address these limitations by shifting the diffusion process into a continuous sentence-level semantic space. We propose CRoCoDiL (Continuous and Robust Conditioned Diffusion for Language), a unified fine-tuning approach that jointly trains an encoder-demasker architecture, grounding the MDM demasking in continuous latent representations. This leads to the formation of a novel autoencoder in which decoding is obtained by an MDM algorithm. Relying on the same framework, we introduce two unconditional text synthesis algorithms: Continuous-Then-Discrete (ConThenDisc), a hybrid-diffusion approach that first generates latent representations in continuous space and then decodes these to tokens via an MDM, and Continuous-Within-Discrete (ConWithinDisc), a multi-diffusion strategy that refines latent representations throughout the discrete sampling process. Experiments using LLaDA show that our methods achieve superior generation quality and more than 10x faster sampling speeds in an unconditional setting.
>
---
#### [replaced 027] Arc Gradient Descent: A Geometrically Motivated Gradient Descent-based Optimiser with Phase-Aware, User-Controlled Step Dynamics (proof-of-concept)
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文提出ArcGD优化器，解决机器学习中的优化问题。通过实验验证其在非凸函数和实际数据集上的优越性能，展示其泛化能力和抗过拟合特性。**

- **链接: [https://arxiv.org/pdf/2512.06737](https://arxiv.org/pdf/2512.06737)**

> **作者:** Nikhil Verma; Joonas Linnosmaa; Leonardo Espinosa-Leal; Napat Vajragupta
>
> **备注:** 90 pages, 6 appendices, proof-of-concept
>
> **摘要:** The paper presents the formulation, implementation, and evaluation of the ArcGD optimiser. The evaluation is conducted initially on a non-convex benchmark function and subsequently on a real-world ML dataset. The initial comparative study using the Adam optimiser is conducted on a stochastic variant of the highly non-convex and notoriously challenging Rosenbrock function, renowned for its narrow, curved valley, across dimensions ranging from 2D to 1000D and an extreme case of 50,000D. Two configurations were evaluated to eliminate learning-rate bias: (i) both using ArcGD's effective learning rate and (ii) both using Adam's default learning rate. ArcGD consistently outperformed Adam under the first setting and, although slower under the second, achieved superior final solutions in most cases. In the second evaluation, ArcGD is evaluated against state-of-the-art optimizers (Adam, AdamW, Lion, SGD) on the CIFAR-10 image classification dataset across 8 diverse MLP architectures ranging from 1 to 5 hidden layers. ArcGD achieved the highest average test accuracy (50.7%) at 20,000 iterations, outperforming AdamW (46.6%), Adam (46.8%), SGD (49.6%), and Lion (43.4%), winning or tying on 6 of 8 architectures. Notably, while Adam and AdamW showed strong early convergence at 5,000 iterations, but regressed with extended training, whereas ArcGD continued improving, demonstrating generalization and resistance to overfitting without requiring early stopping tuning. Strong performance on geometric stress tests and standard deep-learning benchmarks indicates broad applicability, highlighting the need for further exploration. Moreover, it is also shown that both a limiting variant of ArcGD and a momentum augmented ArcGD, recover sign-based momentum updates, revealing a clear conceptual link between ArcGD's phase structure and the core mechanism of the Lion Optimiser.
>
---
#### [replaced 028] EmbBERT: Attention Under 2 MB Memory
- **分类: cs.CL; cs.AR; cs.DC; cs.LG**

- **简介: 该论文提出EmbBERT，一种在2MB内存下运行的高效Transformer模型，解决边缘设备部署难题。通过优化架构，实现与大型模型相当的性能。**

- **链接: [https://arxiv.org/pdf/2502.10001](https://arxiv.org/pdf/2502.10001)**

> **作者:** Riccardo Bravin; Massimo Pavan; Hazem Hesham Yousef Shalby; Fabrizio Pittorino; Manuel Roveri
>
> **备注:** 24 pages, 4 figures, 14 tables
>
> **摘要:** Transformer architectures based on the attention mechanism have revolutionized natural language processing (NLP), driving major breakthroughs across virtually every NLP task. However, their substantial memory and computational requirements still hinder deployment on ultra-constrained devices such as wearables and Internet-of-Things (IoT) units, where available memory is limited to just a few megabytes. To address this challenge, we introduce EmbBERT, a tiny language model (TLM) architecturally designed for extreme efficiency. The model integrates a compact embedding layer, streamlined feed-forward blocks, and an efficient attention mechanism that together enable optimal performance under strict memory budgets. Through this redesign for the extreme edge, we demonstrate that highly simplified transformer architectures remain remarkably effective under tight resource constraints. EmbBERT requires only 2 MB of total memory, and achieves accuracy performance comparable to the ones of state-of-the-art (SotA) models that require a $\mathbf{10\times}$ memory budget. Extensive experiments on the curated TinyNLP benchmark and the GLUE suite confirm that EmbBERT achieves competitive accuracy, comparable to that of larger SotA models, and consistently outperforms downsized versions of BERT and MAMBA of similar size. Furthermore, we demonstrate the model resilience to 8-bit quantization, which further reduces memory usage to just 781 kB , and the scalability of the EmbBERT architecture across the sub-megabyte to tens-of-megabytes range. Finally, we perform an ablation study demonstrating the positive contributions of all components and the pre-training procedure. All code, scripts, and checkpoints are publicly released to ensure reproducibility: this https URL.
>
---
#### [replaced 029] Table-LLM-Specialist: Language Model Specialists for Tables using Iterative Generator-Validator Fine-tuning
- **分类: cs.CL; cs.DB; cs.LG**

- **简介: 该论文提出Table-LLM-Specialist，解决表格任务中语言模型表现不佳的问题。通过生成-验证机制，无需人工标注即可提升模型性能，应用于数据清洗等任务。**

- **链接: [https://arxiv.org/pdf/2410.12164](https://arxiv.org/pdf/2410.12164)**

> **作者:** Junjie Xing; Yeye He; Mengyu Zhou; Haoyu Dong; Shi Han; Dongmei Zhang; Surajit Chaudhuri
>
> **备注:** Full version of a paper in EMNLP 2025; code is available at: this https URL
>
> **摘要:** Language models such as GPT and Llama have shown remarkable ability on diverse natural language tasks, yet their performance on complex table tasks (e.g., NL-to-Code and data cleaning) remains suboptimal. Improving performance typically requires task-specific fine-tuning, which depends on expensive human labeling and is prone to overfitting. In this work, we propose Table-LLM-Specialist, a self-trained fine-tuning paradigm designed for table tasks. Our key insight is that many table tasks admit two dual formulations: a generative version and a classification version. Leveraging this duality, we introduce a Generator-Validator paradigm that iteratively generates and validates training data using language models, enabling effective fine-tuning without manually labeled data. Extensive evaluations on Llama, GPT-3.5, and GPT-4 show that Table-LLM-Specialist achieves (1) strong performance across diverse tasks compared to base models, for example, models fine-tuned on GPT-3.5 often surpass GPT-4 level quality; (2) lower deployment cost by enabling smaller models to reach high quality with reduced latency and cost; and (3) better generalization across multiple benchmarks, due to training on diverse, systematically generated data from real-world tables. Our code is available at this https URL. Models fine-tuned with Table-LLM-Specialist have been integrated into Microsoft Excel and are deployed in production for automated table data cleaning.
>
---
#### [replaced 030] MARS: toward more efficient multi-agent collaboration for LLM reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MARS框架，解决多智能体协作推理效率低的问题。通过角色分工提升推理质量，减少计算开销，适用于大语言模型的协同推理任务。**

- **链接: [https://arxiv.org/pdf/2509.20502](https://arxiv.org/pdf/2509.20502)**

> **作者:** Xiao Wang; Jia Wang; Yijie Wang; Pengtao Dang; Sha Cao; Chi Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive results in natural language understanding, yet their reasoning capabilities remain limited when operating as single agents. Multi-Agent Debate (MAD) has been proposed to address this limitation by enabling collaborative reasoning among multiple models in a round-table debate manner. While effective, MAD introduces substantial computational overhead due to the number of agents involved and the frequent communication required. In this paper, we propose MARS (Multi-Agent Review System), a role-based collaboration framework inspired by the review process. In MARS, an author agent generates an initial solution, reviewer agents provide decisions and comments independently, and a meta-reviewer integrates the feedback to make the final decision and guide further revision. This design enhances reasoning quality while avoiding costly reviewer-to-reviewer interactions, thereby controlling token consumption and inference time. We compared MARS with both MAD and other state-of-the-art reasoning strategies across multiple benchmarks. Extensive experiments with different LLMs show that MARS matches the accuracy of MAD while reducing both token usage and inference time by approximately 50\%. Code is available at this https URL.
>
---
#### [replaced 031] Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria Detection in Parkinson's Disease
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于跨语言语音识别任务，旨在解决帕金森病失语检测中数据不足的问题。通过语言迁移方法对自监督语音表示进行调整，提升跨语言检测效果。**

- **链接: [https://arxiv.org/pdf/2603.22225](https://arxiv.org/pdf/2603.22225)**

> **作者:** Abner Hernandez; Eunjung Yeo; Kwanghee Choi; Chin-Jou Li; Zhengjun Yue; Rohan Kumar Das; Jan Rusz; Mathew Magimai Doss; Juan Rafael Orozco-Arroyave; Tomás Arias-Vergara; Andreas Maier; Elmar Nöth; David R. Mortensen; David Harwath; Paula Andrea Perez-Toro
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** The limited availability of dysarthric speech data makes cross-lingual detection an important but challenging problem. A key difficulty is that speech representations often encode language-dependent structure that can confound dysarthria detection. We propose a representation-level language shift (LS) that aligns source-language self-supervised speech representations with the target-language distribution using centroid-based vector adaptation estimated from healthy-control speech. We evaluate the approach on oral DDK recordings from Parkinson's disease speech datasets in Czech, German, and Spanish under both cross-lingual and multilingual settings. LS substantially improves sensitivity and F1 in cross-lingual settings, while yielding smaller but consistent gains in multilingual settings. Representation analysis further shows that LS reduces language identity in the embedding space, supporting the interpretation that LS removes language-dependent structure.
>
---
