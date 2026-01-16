# 自然语言处理 cs.CL

- **最新发布 101 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] What Gets Activated: Uncovering Domain and Driver Experts in MoE Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型解释性研究，旨在揭示MoE语言模型中专家的激活机制。通过分析领域专家和驱动专家，解决专家激活模式及token触发机制问题。**

- **链接: [https://arxiv.org/pdf/2601.10159v1](https://arxiv.org/pdf/2601.10159v1)**

> **作者:** Guimin Hu; Meng Li; Qiwei Peng; Lijie Hu; Boyan Xu; Ruichu Cai
>
> **摘要:** Most interpretability work focuses on layer- or neuron-level mechanisms in Transformers, leaving expert-level behavior in MoE LLMs underexplored. Motivated by functional specialization in the human brain, we analyze expert activation by distinguishing domain and driver experts. In this work, we study expert activation in MoE models across three public domains and address two key questions: (1) which experts are activated, and whether certain expert types exhibit consistent activation patterns; and (2) how tokens are associated with and trigger the activation of specific experts. To answer these questions, we introduce entropy-based and causal-effect metrics to assess whether an expert is strongly favored for a particular domain, and how strongly expert activation contributes causally to the model's output, thus identify domain and driver experts, respectively. Furthermore, we explore how individual tokens are associated with the activation of specific experts. Our analysis reveals that (1) Among the activated experts, some show clear domain preferences, while others exert strong causal influence on model performance, underscoring their decisive roles. (2) tokens occurring earlier in a sentence are more likely to trigger the driver experts, and (3) adjusting the weights of domain and driver experts leads to significant performance gains across all three models and domains. These findings shed light on the internal mechanisms of MoE models and enhance their interpretability.
>
---
#### [new 002] TF3-RO-50M: Training Compact Romanian Language Models from Scratch on Synthetic Moral Microfiction
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决罗马尼亚语模型训练与合成数据生成问题。工作包括构建罗马尼亚语语言模型框架，优化模型并生成大规模合成故事数据。**

- **链接: [https://arxiv.org/pdf/2601.10410v1](https://arxiv.org/pdf/2601.10410v1)**

> **作者:** Mihai Dan Nadas; Laura Diosan; Andreea Tomescu; Andrei Piscoran
>
> **摘要:** Recent advances in synthetic data generation have shown that compact language models can be trained effectively when the underlying corpus is structurally controlled and linguistically coherent. However, for morphologically rich and computationally under-resourced languages such as Romanian, there is still no openly documented, end-to-end pipeline that unifies tokenizer design, preprocessing, pretraining, compression, evaluation, and large-scale synthetic data generation in a reproducible framework. Building on TF1, a three-million-story English fable dataset, and TF2, which extends TF1 through high-quality Romanian translations, we introduce TF3-RO, a Romanian-centric language modeling pipeline spanning tokenizer training, from-scratch model development, and Romanian-native dataset generation. TF3-RO constructs Romanian-specific BPE and Unigram tokenizers from a linguistically informed corpus to mitigate token inflation induced by Romanian morphology. Using long-sequence packed training, we pretrain a 51.65M-parameter LLaMA-style Transformer entirely from scratch. The model is subsequently optimized through quantization, structured pruning, and logit-based knowledge distillation, yielding a compact 26.45M-parameter student model with tied embeddings and strong deployment characteristics. Using this distilled model, TF3-RO generates three million Romanian-native synthetic fables via a controlled combinatorial prompting framework. Across all stages, the pipeline integrates a comprehensive evaluation suite combining intrinsic metrics, Romanian agreement probes, entity coherence, rule-based grammar checking, and LLM-based assessment. TF3-RO provides a reproducible and linguistically grounded framework for training compact Romanian language models and producing large-scale synthetic narrative corpora.
>
---
#### [new 003] Representation-Aware Unlearning via Activation Signatures: From Suppression to Knowledge-Signature Erasure
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识擦除任务，解决LLM中行为抑制与真实知识删除混淆的问题。提出KIF框架，通过激活签名实现真正知识擦除，提升合规性与安全性。**

- **链接: [https://arxiv.org/pdf/2601.10566v1](https://arxiv.org/pdf/2601.10566v1)**

> **作者:** Syed Naveed Mahmood; Md. Rezaur Rahman Bhuiyan; Tasfia Zaman; Jareen Tasneem Khondaker; Md. Sameer Sakib; Nazia Tasnim; Farig Sadeque
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Selective knowledge erasure from LLMs is critical for GDPR compliance and model safety, yet current unlearning methods conflate behavioral suppression with true knowledge removal, allowing latent capabilities to persist beneath surface-level refusals. In this work, we address this challenge by introducing Knowledge Immunization Framework (KIF), a representation-aware architecture that distinguishes genuine erasure from obfuscation by targeting internal activation signatures rather than surface outputs. Our approach combines dynamic suppression of subject-specific representations with parameter-efficient adaptation, enabling durable unlearning without full model retraining. KIF achieves near-oracle erasure (FQ approx 0.99 vs. 1.00) while preserving utility at oracle levels (MU = 0.62), effectively breaking the stability-erasure tradeoff that has constrained all prior work. We evaluate both standard foundation models (Llama and Mistral) and reasoning-prior models (Qwen and DeepSeek) across 3B to 14B parameters. Our observation shows that standard models exhibit scale-independent true erasure (<3% utility drift), while reasoning-prior models reveal fundamental architectural divergence. Our comprehensive dual-metric evaluation protocol, combining surface-level leakage with latent trace persistence, operationalizes the obfuscation - erasure distinction and enables the first systematic diagnosis of mechanism-level forgetting behavior across model families and scales.
>
---
#### [new 004] Opportunities and Challenges of Natural Language Processing for Low-Resource Senegalese Languages in Social Science Research
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，聚焦解决塞内加尔六种官方语言在社会科学研究中的数据与工具不足问题，分析现状并提供资源库以促进研究。**

- **链接: [https://arxiv.org/pdf/2601.09716v1](https://arxiv.org/pdf/2601.09716v1)**

> **作者:** Derguene Mbaye; Tatiana D. P. Mbengue; Madoune R. Seye; Moussa Diallo; Mamadou L. Ndiaye; Dimitri S. Adjanohoun; Cheikh S. Wade; Djiby Sow; Jean-Claude B. Munyaka; Jerome Chenal
>
> **摘要:** Natural Language Processing (NLP) is rapidly transforming research methodologies across disciplines, yet African languages remain largely underrepresented in this technological shift. This paper provides the first comprehensive overview of NLP progress and challenges for the six national languages officially recognized by the Senegalese Constitution: Wolof, Pulaar, Sereer, Joola, Mandingue, and Soninke. We synthesize linguistic, sociotechnical, and infrastructural factors that shape their digital readiness and identify gaps in data, tools, and benchmarks. Building on existing initiatives and research works, we analyze ongoing efforts in text normalization, machine translation, and speech processing. We also provide a centralized GitHub repository that compiles publicly accessible resources for a range of NLP tasks across these languages, designed to facilitate collaboration and reproducibility. A special focus is devoted to the application of NLP to the social sciences, where multilingual transcription, translation, and retrieval pipelines can significantly enhance the efficiency and inclusiveness of field research. The paper concludes by outlining a roadmap toward sustainable, community-centered NLP ecosystems for Senegalese languages, emphasizing ethical data governance, open resources, and interdisciplinary collaboration.
>
---
#### [new 005] Bears, all bears, and some bears. Language Constraints on Language Models' Inductive Inferences
- **分类: cs.CL**

- **简介: 该论文研究语言对归纳推理的影响，测试语言模型是否能区分不同语义表达，发现其行为与人类相似，表明模型具备一定的语义理解能力。任务属于自然语言处理中的语义理解与推理研究。**

- **链接: [https://arxiv.org/pdf/2601.09852v1](https://arxiv.org/pdf/2601.09852v1)**

> **作者:** Sriram Padmanabhan; Siyuan Song; Kanishka Misra
>
> **摘要:** Language places subtle constraints on how we make inductive inferences. Developmental evidence by Gelman et al. (2002) has shown children (4 years and older) to differentiate among generic statements ("Bears are daxable"), universally quantified NPs ("all bears are daxable") and indefinite plural NPs ("some bears are daxable") in extending novel properties to a specific member (all > generics > some), suggesting that they represent these types of propositions differently. We test if these subtle differences arise in general purpose statistical learners like Vision Language Models, by replicating the original experiment. On tasking them through a series of precondition tests (robust identification of categories in images and sensitivities to all and some), followed by the original experiment, we find behavioral alignment between models and humans. Post-hoc analyses on their representations revealed that these differences are organized based on inductive constraints and not surface-form differences.
>
---
#### [new 006] HUMANLLM: Benchmarking and Reinforcing LLM Anthropomorphism via Human Cognitive Patterns
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，旨在解决LLM与人类认知行为对齐的问题。通过构建心理模式框架，评估并提升模型的拟人化能力。**

- **链接: [https://arxiv.org/pdf/2601.10198v1](https://arxiv.org/pdf/2601.10198v1)**

> **作者:** Xintao Wang; Jian Yang; Weiyuan Li; Rui Xie; Jen-tse Huang; Jun Gao; Shuai Huang; Yueping Kang; Liyuan Gou; Hongwei Feng; Yanghua Xiao
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HUMANLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from ~12,000 academic papers and synthesize 11,359 scenarios where 2-5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment (r=0.91) while revealing that holistic metrics conflate simulation accuracy with social desirability. HUMANLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4x fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling--simulating not just what humans do, but the psychological processes generating those behaviors.
>
---
#### [new 007] Enhancing Business Analytics through Hybrid Summarization of Financial Reports
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于金融文本摘要任务，旨在解决手动分析财务报告效率低、易出错的问题。通过混合提取与生成方法，构建高效且准确的摘要系统。**

- **链接: [https://arxiv.org/pdf/2601.09729v1](https://arxiv.org/pdf/2601.09729v1)**

> **作者:** Tohida Rehman
>
> **备注:** 12 pages, 2 figures, 2 tables
>
> **摘要:** Financial reports and earnings communications contain large volumes of structured and semi structured information, making detailed manual analysis inefficient. Earnings conference calls provide valuable evidence about a firm's performance, outlook, and strategic priorities. The manual analysis of lengthy call transcripts requires substantial effort and is susceptible to interpretive bias and unintentional error. In this work, we present a hybrid summarization framework that combines extractive and abstractive techniques to produce concise and factually reliable Reuters-style summaries from the ECTSum dataset. The proposed two stage pipeline first applies the LexRank algorithm to identify salient sentences, which are subsequently summarized using fine-tuned variants of BART and PEGASUS designed for resource constrained settings. In parallel, we fine-tune a Longformer Encoder-Decoder (LED) model to directly capture long-range contextual dependencies in financial documents. Model performance is evaluated using standard automatic metrics, including ROUGE, METEOR, MoverScore, and BERTScore, along with domain-specific variants such as SciBERTScore and FinBERTScore. To assess factual accuracy, we further employ entity-level measures based on source-precision and F1-target. The results highlight complementary trade offs between approaches, long context models yield the strongest overall performance, while the hybrid framework achieves competitive results with improved factual consistency under computational constraints. These findings support the development of practical summarization systems for efficiently distilling lengthy financial texts into usable business insights.
>
---
#### [new 008] Untangling Input Language from Reasoning Language: A Diagnostic Framework for Cross-Lingual Moral Alignment in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究跨语言道德对齐问题，旨在区分输入语言与推理语言的影响。通过设计诊断框架，分析LLMs在不同语言下的道德判断差异。**

- **链接: [https://arxiv.org/pdf/2601.10257v1](https://arxiv.org/pdf/2601.10257v1)**

> **作者:** Nan Li; Bo Kang; Tijl De Bie
>
> **摘要:** When LLMs judge moral dilemmas, do they reach different conclusions in different languages, and if so, why? Two factors could drive such differences: the language of the dilemma itself, or the language in which the model reasons. Standard evaluation conflates these by testing only matched conditions (e.g., English dilemma with English reasoning). We introduce a methodology that separately manipulates each factor, covering also mismatched conditions (e.g., English dilemma with Chinese reasoning), enabling decomposition of their contributions. To study \emph{what} changes, we propose an approach to interpret the moral judgments in terms of Moral Foundations Theory. As a side result, we identify evidence for splitting the Authority dimension into a family-related and an institutional dimension. Applying this methodology to English-Chinese moral judgment with 13 LLMs, we demonstrate its diagnostic power: (1) the framework isolates reasoning-language effects as contributing twice the variance of input-language effects; (2) it detects context-dependency in nearly half of models that standard evaluation misses; and (3) a diagnostic taxonomy translates these patterns into deployment guidance. We release our code and datasets at https://anonymous.4open.science/r/CrossCulturalMoralJudgement.
>
---
#### [new 009] Unlocking Implicit Experience: Synthesizing Tool-Use Trajectories from Text
- **分类: cs.CL**

- **简介: 该论文属于多轮工具使用任务，旨在解决获取多样化真实数据的难题。通过文本生成多轮工具使用轨迹，提出GEM框架及高效合成模型，提升性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2601.10355v1](https://arxiv.org/pdf/2601.10355v1)**

> **作者:** Zhihao Xu; Rumei Li; Jiahuan Li; Rongxiang Weng; Jingang Wang; Xunliang Cai; Xiting Wang
>
> **摘要:** Enabling Large Language Models (LLMs) to effectively utilize tools in multi-turn interactions is essential for building capable autonomous agents. However, acquiring diverse and realistic multi-turn tool-use data remains a significant challenge. In this work, we propose a novel text-based paradigm. We observe that textual corpora naturally contain rich, multi-step problem-solving experiences, which can serve as an untapped, scalable, and authentic data source for multi-turn tool-use tasks. Based on this insight, we introduce GEM, a data synthesis pipeline that enables the generation and extraction of multi-turn tool-use trajectories from text corpora through a four-stage process: relevance filtering, workflow & tool extraction, trajectory grounding, and complexity refinement. To reduce the computational cost, we further train a specialized Trajectory Synthesizer via supervised fine-tuning. This model distills the complex generation pipeline into an efficient, end-to-end trajectory generator. Experiments demonstrate that our GEM-32B achieve a 16.5% improvement on the BFCL V3 Multi-turn benchmark. Our models partially surpass the performance of models trained on τ - bench (Airline and Retail) in-domain data, highlighting the superior generalization capability derived from our text-based synthesis paradigm. Notably, our Trajectory Synthesizer matches the quality of the full pipeline while significantly reducing inference latency and costs.
>
---
#### [new 010] SIN-Bench: Tracing Native Evidence Chains in Long-Context Multimodal Scientific Interleaved Literature
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文提出SIN-Bench，用于评估多模态大模型在长文本科学文献中的证据链推理能力，解决模型仅答对但缺乏因果证据的问题。**

- **链接: [https://arxiv.org/pdf/2601.10108v1](https://arxiv.org/pdf/2601.10108v1)**

> **作者:** Yiming Ren; Junjie Wang; Yuxin Meng; Yihang Shi; Zhiqiang Lin; Ruihang Chu; Yiran Xu; Ziming Li; Yunfei Zhao; Zihan Wang; Yu Qiao; Ruiming Tang; Minghao Liu; Yujiu Yang
>
> **摘要:** Evaluating whether multimodal large language models truly understand long-form scientific papers remains challenging: answer-only metrics and synthetic "Needle-In-A-Haystack" tests often reward answer matching without requiring a causal, evidence-linked reasoning trace in the document. We propose the "Fish-in-the-Ocean" (FITO) paradigm, which requires models to construct explicit cross-modal evidence chains within native scientific documents. To operationalize FITO, we build SIN-Data, a scientific interleaved corpus that preserves the native interleaving of text and figures. On top of it, we construct SIN-Bench with four progressive tasks covering evidence discovery (SIN-Find), hypothesis verification (SIN-Verify), grounded QA (SIN-QA), and evidence-anchored synthesis (SIN-Summary). We further introduce "No Evidence, No Score", scoring predictions when grounded to verifiable anchors and diagnosing evidence quality via matching, relevance, and logic. Experiments on eight MLLMs show that grounding is the primary bottleneck: Gemini-3-pro achieves the best average overall score (0.573), while GPT-5 attains the highest SIN-QA answer accuracy (0.767) but underperforms on evidence-aligned overall scores, exposing a gap between correctness and traceable support.
>
---
#### [new 011] SciNets: Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文提出SciNets，解决科学文献合成中的跨领域机制连接问题，通过图约束的多跳推理实现可控的机制解释生成。**

- **链接: [https://arxiv.org/pdf/2601.09727v1](https://arxiv.org/pdf/2601.09727v1)**

> **作者:** Sauhard Dubey
>
> **备注:** 19 pages, 2 figures
>
> **摘要:** Cross-domain scientific synthesis requires connecting mechanistic explanations across fragmented literature, a capability that remains challenging for both retrieval-based systems and unconstrained language models. While recent work has applied large language models to scientific summarization and question answering, these approaches provide limited control over reasoning depth and structural grounding. We frame mechanistic synthesis as a graph-constrained multi-hop reasoning problem over literature-derived concept graphs. Given a scientific query and a compact, query-local corpus, SciNets constructs a directed concept graph and synthesizes mechanistic explanations by identifying multi-hop reasoning paths that connect concepts that rarely co-occur within individual papers. We systematically compare shortest-path reasoning, k-shortest paths with diversity constraints, stochastic random walks, and a retrieval-augmented language model baseline. Rather than evaluating correctness, which is often indeterminate when synthesizing connections across distributed sources, we introduce a behavioral framework that measures symbolic reasoning depth, mechanistic diversity, and grounding stability. Across machine learning, biology, and climate science tasks, explicit graph constraints enable controllable multi-hop reasoning while revealing a consistent trade-off: deeper and more diverse symbolic reasoning increases grounding instability, whereas shortest-path reasoning remains highly stable but structurally conservative. These findings provide a systematic behavioral characterization of the limits and capabilities of current graph-LLM integration for scientific synthesis.
>
---
#### [new 012] SurgGoal: Rethinking Surgical Planning Evaluation via Goal-Satisfiability
- **分类: cs.CL; cs.RO**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决手术规划评估不准确的问题。通过定义基于目标满足度的评估标准，提出新基准并验证现有方法的不足。**

- **链接: [https://arxiv.org/pdf/2601.10455v1](https://arxiv.org/pdf/2601.10455v1)**

> **作者:** Ruochen Li; Kun Yuan; Yufei Xia; Yue Zhou; Qingyu Lu; Weihang Li; Youxiang Zhu; Nassir Navab
>
> **摘要:** Surgical planning integrates visual perception, long-horizon reasoning, and procedural knowledge, yet it remains unclear whether current evaluation protocols reliably assess vision-language models (VLMs) in safety-critical settings. Motivated by a goal-oriented view of surgical planning, we define planning correctness via phase-goal satisfiability, where plan validity is determined by expert-defined surgical rules. Based on this definition, we introduce a multicentric meta-evaluation benchmark with valid procedural variations and invalid plans containing order and content errors. Using this benchmark, we show that sequence similarity metrics systematically misjudge planning quality, penalizing valid plans while failing to identify invalid ones. We therefore adopt a rule-based goal-satisfiability metric as a high-precision meta-evaluation reference to assess Video-LLMs under progressively constrained settings, revealing failures due to perception errors and under-constrained reasoning. Structural knowledge consistently improves performance, whereas semantic guidance alone is unreliable and benefits larger models only when combined with structural constraints.
>
---
#### [new 013] Role-Playing Agents Driven by Large Language Models: Current Status, Challenges, and Future Trends
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于自然语言处理任务，探讨角色扮演代理的现状与挑战，分析技术路径、数据构建及评估方法，旨在提升角色扮演的智能性与真实性。**

- **链接: [https://arxiv.org/pdf/2601.10122v1](https://arxiv.org/pdf/2601.10122v1)**

> **作者:** Ye Wang; Jiaxing Chen; Hongjiang Xiao
>
> **摘要:** In recent years, with the rapid advancement of large language models (LLMs), role-playing language agents (RPLAs) have emerged as a prominent research focus at the intersection of natural language processing (NLP) and human-computer interaction. This paper systematically reviews the current development and key technologies of RPLAs, delineating the technological evolution from early rule-based template paradigms, through the language style imitation stage, to the cognitive simulation stage centered on personality modeling and memory mechanisms. It summarizes the critical technical pathways supporting high-quality role-playing, including psychological scale-driven character modeling, memory-augmented prompting mechanisms, and motivation-situation-based behavioral decision control. At the data level, the paper further analyzes the methods and challenges of constructing role-specific corpora, focusing on data sources, copyright constraints, and structured annotation processes. In terms of evaluation, it collates multi-dimensional assessment frameworks and benchmark datasets covering role knowledge, personality fidelity, value alignment, and interactive hallucination, while commenting on the advantages and disadvantages of methods such as human evaluation, reward models, and LLM-based scoring. Finally, the paper outlines future development directions of role-playing agents, including personality evolution modeling, multi-agent collaborative narrative, multimodal immersive interaction, and integration with cognitive neuroscience, aiming to provide a systematic perspective and methodological insights for subsequent research.
>
---
#### [new 014] An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit
- **分类: cs.CL; cs.IR; cs.LG; cs.SI**

- **简介: 该论文属于人岗匹配任务，解决长简历与多语言岗位匹配难题。提出一种高效重排序模型，结合大模型蒸馏提升匹配准确性。**

- **链接: [https://arxiv.org/pdf/2601.10321v1](https://arxiv.org/pdf/2601.10321v1)**

> **作者:** Warren Jouanneau; Emma Jouffroy; Marc Palyart
>
> **摘要:** Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines.
>
---
#### [new 015] Forgetting as a Feature: Cognitive Alignment of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究LLM的遗忘机制。旨在解决LLM在推理中系统性遗忘的问题，通过模拟人类记忆动态，提出概率记忆提示方法，提升长时推理能力。**

- **链接: [https://arxiv.org/pdf/2601.09726v1](https://arxiv.org/pdf/2601.09726v1)**

> **作者:** Hien Tran; Quinten Steenhuis; Alexandros Christoforos; Chadbourne Davis
>
> **备注:** Under submission
>
> **摘要:** Large Language Models (LLMs) are often evaluated against ideals of perfect Bayesian inference, yet growing evidence suggests that their in-context reasoning exhibits systematic forgetting of past information. Rather than viewing this behavior as a limitation, we reinterpret forgetting as a functional cognitive mechanism. Drawing inspiration from human memory dynamics, we model LLM inference as a probabilistic memory process governed by exponential decay. We introduce a benchmark suite that evaluates temporal reasoning, concept drift adaptation, and associative recall, enabling direct comparison between model behavior and human cognitive patterns. Our empirical results reveal that LLMs demonstrate forgetting rates analogous to human memory efficiency trade-offs between stability and adaptability. Building on these observations, we propose probabilistic memory prompting, a lightweight strategy that shapes evidence integration to mimic human-like memory decay, leading to improved long-horizon reasoning performance. Our findings position forgetting not as a failure mode, but as a principled mechanism for adaptive intelligence.
>
---
#### [new 016] SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction
- **分类: cs.CL**

- **简介: 该论文属于知识图谱构建任务，旨在解决事实覆盖与关系连贯性之间的矛盾。通过引入问答对作为中间表示，提升语义结构化效果。**

- **链接: [https://arxiv.org/pdf/2601.10003v1](https://arxiv.org/pdf/2601.10003v1)**

> **作者:** Sanghyeok Choi; Woosang Jeon; Kyuseok Yang; Taehyeong Kim
>
> **摘要:** Constructing Knowledge Graphs (KGs) from unstructured text provides a structured framework for knowledge representation and reasoning, yet current LLM-based approaches struggle with a fundamental trade-off: factual coverage often leads to relational fragmentation, while premature consolidation causes information loss. To address this, we propose SocraticKG, an automated KG construction method that introduces question-answer pairs as a structured intermediate representation to systematically unfold document-level semantics prior to triple extraction. By employing 5W1H-guided QA expansion, SocraticKG captures contextual dependencies and implicit relational links typically lost in direct KG extraction pipelines, providing explicit grounding in the source document that helps mitigate implicit reasoning errors. Evaluation on the MINE benchmark demonstrates that our approach effectively addresses the coverage-connectivity trade-off, achieving superior factual retention while maintaining high structural cohesion even as extracted knowledge volume substantially expands. These results highlight that QA-mediated semantic scaffolding plays a critical role in structuring semantics prior to KG extraction, enabling more coherent and reliable graph construction in subsequent stages.
>
---
#### [new 017] Stable and Explainable Personality Trait Evaluation in Large Language Models with Internal Activations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型解释任务，旨在解决LLM人格特质评估的稳定性与可解释性问题。提出PVNI方法，通过内部激活提取人格向量并进行插值分析，提升评估稳定性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.09833v1](https://arxiv.org/pdf/2601.09833v1)**

> **作者:** Xiaoxu Ma; Xiangbo Zhang; Zhenyu Weng
>
> **摘要:** Evaluating personality traits in Large Language Models (LLMs) is key to model interpretation, comparison, and responsible deployment. However, existing questionnaire-based evaluation methods exhibit limited stability and offer little explainability, as their results are highly sensitive to minor variations in prompt phrasing or role-play configurations. To address these limitations, we propose an internal-activation-based approach, termed Persona-Vector Neutrality Interpolation (PVNI), for stable and explainable personality trait evaluation in LLMs. PVNI extracts a persona vector associated with a target personality trait from the model's internal activations using contrastive prompts. It then estimates the corresponding neutral score by interpolating along the persona vector as an anchor axis, enabling an interpretable comparison between the neutral prompt representation and the persona direction. We provide a theoretical analysis of the effectiveness and generalization properties of PVNI. Extensive experiments across diverse LLMs demonstrate that PVNI yields substantially more stable personality trait evaluations than existing methods, even under questionnaire and role-play variants.
>
---
#### [new 018] Benchmarking Cross-Lingual Semantic Alignment in Multilingual Embeddings
- **分类: cs.CL**

- **简介: 该论文属于多语言嵌入研究，旨在评估跨语言语义对齐能力。通过引入SA指标和Semanscope框架，对比13种模型，揭示对齐效果受训练目标影响而非规模或数据量。**

- **链接: [https://arxiv.org/pdf/2601.09732v1](https://arxiv.org/pdf/2601.09732v1)**

> **作者:** Wen G. Gong
>
> **备注:** 20 pages, 9 figures, 4 tables
>
> **摘要:** With hundreds of multilingual embedding models available, practitioners lack clear guidance on which provide genuine cross-lingual semantic alignment versus task performance through language-specific patterns. Task-driven benchmarks (MTEB) may mask fundamental alignment shortcomings. We introduce Semantic Affinity (SA), a bounded (between 0 and 1) metric measuring inter-lingual to intra-lingual spread ratio using cosine distance, combined with PHATE visualization in our Semanscope framework. Benchmarking 13 models across 4 datasets (52 experiments) reveals a three-tier structure: (1) Top BERT models (LaBSE SA = 0.70, USE SA = 0.68, S-BERT SA = 0.68) achieve strong alignment via translation-pair supervision; (2) LLM embeddings plateau at SA between 0.55 and 0.61 regardless of 0.6 B to 8 B scale; (3) MLM-only BERT models (mBERT, XLM-R, SA < 0.50) fail despite more than 100 language training. Training objective, not architecture or scale, determines alignment. Oracle Bone primitives (1200 BCE) expose semantic drift-models learn corpus patterns rather than cognitive primitives. This work provides semantic benchmarking to help practitioners select quality multilingual embeddings from hundreds of available models, showing cross-lingual alignment requires explicit translation supervision, not merely model scale or multilingual data.
>
---
#### [new 019] The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型的默认人格定位问题，通过分析“助理轴”来稳定模型行为，解决 persona drift 和对抗性攻击问题。**

- **链接: [https://arxiv.org/pdf/2601.10387v1](https://arxiv.org/pdf/2601.10387v1)**

> **作者:** Christina Lu; Jack Gallagher; Jonathan Michala; Kyle Fish; Jack Lindsey
>
> **摘要:** Large language models can represent a variety of personas but typically default to a helpful Assistant identity cultivated during post-training. We investigate the structure of the space of model personas by extracting activation directions corresponding to diverse character archetypes. Across several different models, we find that the leading component of this persona space is an "Assistant Axis," which captures the extent to which a model is operating in its default Assistant mode. Steering towards the Assistant direction reinforces helpful and harmless behavior; steering away increases the model's tendency to identify as other entities. Moreover, steering away with more extreme values often induces a mystical, theatrical speaking style. We find this axis is also present in pre-trained models, where it primarily promotes helpful human archetypes like consultants and coaches and inhibits spiritual ones. Measuring deviations along the Assistant Axis predicts "persona drift," a phenomenon where models slip into exhibiting harmful or bizarre behaviors that are uncharacteristic of their typical persona. We find that persona drift is often driven by conversations demanding meta-reflection on the model's processes or featuring emotionally vulnerable users. We show that restricting activations to a fixed region along the Assistant Axis can stabilize model behavior in these scenarios -- and also in the face of adversarial persona-based jailbreaks. Our results suggest that post-training steers models toward a particular region of persona space but only loosely tethers them to it, motivating work on training and steering strategies that more deeply anchor models to a coherent persona.
>
---
#### [new 020] Contextual StereoSet: Stress-Testing Bias Alignment Robustness in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于自然语言处理中的偏见评估任务，旨在测试大模型在不同语境下的偏见一致性。通过构建Contextual StereoSet基准，分析语境变化对偏见的影响，并提出CSF方法提升评估 robustness。**

- **链接: [https://arxiv.org/pdf/2601.10460v1](https://arxiv.org/pdf/2601.10460v1)**

> **作者:** Abhinaba Basu; Pavan Chakraborty
>
> **摘要:** A model that avoids stereotypes in a lab benchmark may not avoid them in deployment. We show that measured bias shifts dramatically when prompts mention different places, times, or audiences -- no adversarial prompting required. We introduce Contextual StereoSet, a benchmark that holds stereotype content fixed while systematically varying contextual framing. Testing 13 models across two protocols, we find striking patterns: anchoring to 1990 (vs. 2030) raises stereotype selection in all models tested on this contrast (p<0.05); gossip framing raises it in 5 of 6 full-grid models; out-group observer framing shifts it by up to 13 percentage points. These effects replicate in hiring, lending, and help-seeking vignettes. We propose Context Sensitivity Fingerprints (CSF): a compact profile of per-dimension dispersion and paired contrasts with bootstrap CIs and FDR correction. Two evaluation tracks support different use cases -- a 360-context diagnostic grid for deep analysis and a budgeted protocol covering 4,229 items for production screening. The implication is methodological: bias scores from fixed-condition tests may not generalize.This is not a claim about ground-truth bias rates; it is a stress test of evaluation robustness. CSF forces evaluators to ask, "Under what conditions does bias appear?" rather than "Is this model biased?" We release our benchmark, code, and results.
>
---
#### [new 021] PERM: Psychology-grounded Empathetic Reward Modeling for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于情感计算任务，旨在提升大语言模型的共情能力。针对现有方法单向评估共情的不足，提出PERM模型，从支持者、寻求者和旁观者多角度评估共情，实验表明其效果显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2601.10532v1](https://arxiv.org/pdf/2601.10532v1)**

> **作者:** Chengbing Wang; Wuqiang Zheng; Yang Zhang; Fengbin Zhu; Junyi Cheng; Yi Xie; Wenjie Wang; Fuli Feng
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in human-centric applications, yet they often fail to provide substantive emotional support. While Reinforcement Learning (RL) has been utilized to enhance empathy of LLMs, existing reward models typically evaluate empathy from a single perspective, overlooking the inherently bidirectional interaction nature of empathy between the supporter and seeker as defined by Empathy Cycle theory. To address this limitation, we propose Psychology-grounded Empathetic Reward Modeling (PERM). PERM operationalizes empathy evaluation through a bidirectional decomposition: 1) Supporter perspective, assessing internal resonation and communicative expression; 2) Seeker perspective, evaluating emotional reception. Additionally, it incorporates a bystander perspective to monitor overall interaction quality. Extensive experiments on a widely-used emotional intelligence benchmark and an industrial daily conversation dataset demonstrate that PERM outperforms state-of-the-art baselines by over 10\%. Furthermore, a blinded user study reveals a 70\% preference for our approach, highlighting its efficacy in generating more empathetic responses. Our code, dataset, and models are available at https://github.com/ZhengWwwq/PERM.
>
---
#### [new 022] Closing the Data Loop: Using OpenDataArena to Engineer Superior Training Datasets
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数据集构建任务，旨在解决LLM训练数据质量不足的问题。提出ODA框架，通过闭环机制提升数据集效果，构建了两个高效数据集。**

- **链接: [https://arxiv.org/pdf/2601.09733v1](https://arxiv.org/pdf/2601.09733v1)**

> **作者:** Xin Gao; Xiaoyang Wang; Yun Zhu; Mengzhang Cai; Conghui He; Lijun Wu
>
> **备注:** Superior ODA-Math, ODA-Mixture Datasets
>
> **摘要:** The construction of Supervised Fine-Tuning (SFT) datasets is a critical yet under-theorized stage in the post-training of Large Language Models (LLMs), as prevalent practices often rely on heuristic aggregation without a systematic understanding of how individual samples contribute to model performance. In this report, we propose a paradigm shift from ad-hoc curation to a closed-loop dataset engineering framework using OpenDataArena (ODA), which leverages value-anchored rankings and multi-dimensional analysis to transform value benchmarking into feedback signals guiding dataset construction. We instantiate this methodology through two new datasets: \textbf{ODA-Math-460k}, a specialized mathematics reasoning dataset that utilizes a novel two-stage difficulty-aware pipeline to achieve State-of-the-Art (SOTA) results on benchmarks such as AIME and HMMT, and \textbf{ODA-Mixture (100k \& 500k)}, a series of multi-domain instruction datasets built via an ``Anchor-and-Patch'' strategy that outperforms significantly larger open-source baselines. Our empirical results demonstrate that ODA-driven datasets significantly improve both domain-specific reasoning and general utility while achieving superior data efficiency, validating a transition toward data-centric AI where transparent evaluation serves as the primary engine for engineering high-quality training data.
>
---
#### [new 023] Credit C-GPT: A Domain-Specialized Large Language Model for Conversational Understanding in Vietnamese Debt Collection
- **分类: cs.CL**

- **简介: 该论文提出Credit C-GPT，一个针对越南债务催收场景的对话理解大语言模型。解决传统NLP在非正式语言和情感分析上的不足，整合多种对话任务，提升催收效率与分析能力。**

- **链接: [https://arxiv.org/pdf/2601.10167v1](https://arxiv.org/pdf/2601.10167v1)**

> **作者:** Nhung Nguyen Thi Hong; Cuong Nguyen Dang; Tri Le Ngoc
>
> **备注:** 8 pages, 0 figures, 3 tables. Preprint
>
> **摘要:** Debt collection is a critical function within the banking, financial services, and insurance (BFSI) sector, relying heavily on large-scale human-to-human conversational interactions conducted primarily in Vietnamese contact centers. These conversations involve informal spoken language, emotional variability, and complex domain-specific reasoning, which pose significant challenges for traditional natural language processing systems. This paper introduces Credit C-GPT, a domain-specialized large language model with seven billion parameters, fine-tuned for conversational understanding in Vietnamese debt collection scenarios. The proposed model integrates multiple conversational intelligence tasks, including dialogue understanding, sentiment recognition, intent detection, call stage classification, and structured slot-value extraction, within a single reasoning-based framework. We describe the data construction process, annotation strategy, and training methodology, and evaluate the model on proprietary human-annotated datasets. Experimental results show consistent improvements over traditional pipeline-based approaches, indicating that domain-specialized conversational language models provide a scalable and privacy-aware solution for real-time assistance and post-call analytics in enterprise contact centers.
>
---
#### [new 024] SagaScale: A Realistic, Scalable, and High-Quality Long-Context Benchmark Built from Full-Length Novels
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SagaScale，一个用于长文本理解的高质量基准，解决长文档处理难题。通过小说构建数据集，评估大模型性能，揭示模型在长上下文中的表现及改进方法。**

- **链接: [https://arxiv.org/pdf/2601.09723v1](https://arxiv.org/pdf/2601.09723v1)**

> **作者:** Guancheng Du; Yong Hu; Wenqing Wang; Yaming Yang; Jiaheng Gao
>
> **摘要:** Large Language Models (LLMs) have shown significant progress, but understanding long and complex documents remains challenging. Many long-context benchmarks have been proposed, but they face several limitations, including task realism, data scalability, and data quality. To this end, we introduce SagaScale, a realistic, scalable, and high-quality long-context benchmark built from full-length novels. The entire benchmark is constructed using an automated data collection pipeline that utilizes external resources (e.g., Wikipedia pages) to curate question-answer pairs. Critically, these external resources are provided only for benchmark construction and not during evaluation, which allows LLMs to curate complex questions that go beyond what they can answer during evaluation. SagaScale is also bilingual and offers the largest context length to date, with average token counts exceeding 250K for English novels and 320K for Chinese novels. Our evaluation across 12 frontier LLMs and three long-context methods -- Naïve RAG, Agentic RAG, and Long Context -- yields key insights, including: (1) Directly supplying the full context to the LLM can outperform other methods by a large margin; (2) Most LLMs still struggle with lengthy contexts, but Gemini-2.5-Pro stands out as an exception; and (3) Agentic RAG effectively addresses the retrieval bottleneck in Naïve RAG. Finally, we publicly release the SagaScale benchmark and our data collection codebase to facilitate future research.
>
---
#### [new 025] StatLLaMA: A multi-stage training framework for building a domain-optimized statistical language model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在解决如何高效构建领域优化的统计语言模型。通过多阶段训练框架，提升模型在统计领域的表现，同时保持通用推理能力。**

- **链接: [https://arxiv.org/pdf/2601.09718v1](https://arxiv.org/pdf/2601.09718v1)**

> **作者:** Jing-Yi Zeng; Guan-Hua Huang
>
> **备注:** 31 pages, 3 figures
>
> **摘要:** This study investigates how to efficiently build a domain-specialized large language model (LLM) for statistics using the lightweight LLaMA-3.2-3B family as the foundation model (FM). We systematically compare three multi-stage training pipelines, starting from a base FM with no instruction-following capability, a base FM augmented with post-hoc instruction tuning, and an instruction-tuned FM with strong general reasoning abilities across continual pretraining, supervised fine-tuning (SFT), reinforcement learning from human feedback (RLHF) preference alignment, and downstream task adaptation. Results show that pipelines beginning with a base FM fail to develop meaningful statistical reasoning, even after extensive instruction tuning, SFT, or RLHF alignment. In contrast, starting from LLaMA-3.2-3B-Instruct enables effective domain specialization. A comprehensive evaluation of SFT variants reveals clear trade-offs between domain expertise and general reasoning ability. We further demonstrate that direct preference optimization provides stable and effective RLHF preference alignment. Finally, we show that downstream fine-tuning must be performed with extremely low intensity to avoid catastrophic forgetting in highly optimized models. The final model, StatLLaMA, achieves strong and balanced performance on benchmarks of mathematical reasoning, common-sense reasoning, and statistical expertise, offering a practical blueprint for developing resource-efficient statistical LLMs. The code is available at https://github.com/HuangDLab/StatLLaMA.
>
---
#### [new 026] MedRedFlag: Investigating how LLMs Redirect Misconceptions in Real-World Health Communication
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗问答任务，旨在解决LLMs在处理含错误前提的健康问题时无法有效引导的问题。研究构建了MedRedFlag数据集，并对比了LLMs与医生的回答表现。**

- **链接: [https://arxiv.org/pdf/2601.09853v1](https://arxiv.org/pdf/2601.09853v1)**

> **作者:** Sraavya Sambara; Yuan Pu; Ayman Ali; Vishala Mishra; Lionel Wong; Monica Agrawal
>
> **摘要:** Real-world health questions from patients often unintentionally embed false assumptions or premises. In such cases, safe medical communication typically involves redirection: addressing the implicit misconception and then responding to the underlying patient context, rather than the original question. While large language models (LLMs) are increasingly being used by lay users for medical advice, they have not yet been tested for this crucial competency. Therefore, in this work, we investigate how LLMs react to false premises embedded within real-world health questions. We develop a semi-automated pipeline to curate MedRedFlag, a dataset of 1100+ questions sourced from Reddit that require redirection. We then systematically compare responses from state-of-the-art LLMs to those from clinicians. Our analysis reveals that LLMs often fail to redirect problematic questions, even when the problematic premise is detected, and provide answers that could lead to suboptimal medical decision making. Our benchmark and results reveal a novel and substantial gap in how LLMs perform under the conditions of real-world health communication, highlighting critical safety concerns for patient-facing medical AI systems. Code and dataset are available at https://github.com/srsambara-1/MedRedFlag.
>
---
#### [new 027] Long-Chain Reasoning Distillation via Adaptive Prefix Alignment
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决教师模型推理轨迹过长复杂导致学生模型难以学习的问题。提出P-ALIGN框架，通过自适应前缀对齐提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2601.10064v1](https://arxiv.org/pdf/2601.10064v1)**

> **作者:** Zhenghao Liu; Zhuoyang Wu; Xinze Li; Yukun Yan; Shuo Wang; Zulong Chen; Yu Gu; Ge Yu; Maosong Sun
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities, particularly in solving complex mathematical problems. Recent studies show that distilling long reasoning trajectories can effectively enhance the reasoning performance of small-scale student models. However, teacher-generated reasoning trajectories are often excessively long and structurally complex, making them difficult for student models to learn. This mismatch leads to a gap between the provided supervision signal and the learning capacity of the student model. To address this challenge, we propose Prefix-ALIGNment distillation (P-ALIGN), a framework that fully exploits teacher CoTs for distillation through adaptive prefix alignment. Specifically, P-ALIGN adaptively truncates teacher-generated reasoning trajectories by determining whether the remaining suffix is concise and sufficient to guide the student model. Then, P-ALIGN leverages the teacher-generated prefix to supervise the student model, encouraging effective prefix alignment. Experiments on multiple mathematical reasoning benchmarks demonstrate that P-ALIGN outperforms all baselines by over 3%. Further analysis indicates that the prefixes constructed by P-ALIGN provide more effective supervision signals, while avoiding the negative impact of redundant and uncertain reasoning components. All code is available at https://github.com/NEUIR/P-ALIGN.
>
---
#### [new 028] Deriving Character Logic from Storyline as Codified Decision Trees
- **分类: cs.CL**

- **简介: 该论文属于角色行为建模任务，旨在解决传统行为描述不结构化、不可执行的问题。通过构建可执行的决策树，提升角色代理的行为一致性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.10080v1](https://arxiv.org/pdf/2601.10080v1)**

> **作者:** Letian Peng; Kun Zhou; Longfei Yun; Yupeng Hou; Jingbo Shang
>
> **摘要:** Role-playing (RP) agents rely on behavioral profiles to act consistently across diverse narrative contexts, yet existing profiles are largely unstructured, non-executable, and weakly validated, leading to brittle agent behavior. We propose Codified Decision Trees (CDT), a data-driven framework that induces an executable and interpretable decision structure from large-scale narrative data. CDT represents behavioral profiles as a tree of conditional rules, where internal nodes correspond to validated scene conditions and leaves encode grounded behavioral statements, enabling deterministic retrieval of context-appropriate rules at execution time. The tree is learned by iteratively inducing candidate scene-action rules, validating them against data, and refining them through hierarchical specialization, yielding profiles that support transparent inspection and principled updates. Across multiple benchmarks, CDT substantially outperforms human-written profiles and prior profile induction methods on $85$ characters across $16$ artifacts, indicating that codified and validated behavioral representations lead to more reliable agent grounding.
>
---
#### [new 029] One Instruction Does Not Fit All: How Well Do Embeddings Align Personas and Instructions in Low-Resource Indian Languages?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言模型在低资源印度语言中对齐用户画像与指令的性能，解决跨语言检索与分类问题，构建了12种语言的基准测试并评估多种模型表现。**

- **链接: [https://arxiv.org/pdf/2601.10205v1](https://arxiv.org/pdf/2601.10205v1)**

> **作者:** Arya Shah; Himanshu beniwal; Mayank Singh
>
> **备注:** 12 pages, 4 figures, 10 tables
>
> **摘要:** Aligning multilingual assistants with culturally grounded user preferences is essential for serving India's linguistically diverse population of over one billion speakers across multiple scripts. However, existing benchmarks either focus on a single language or conflate retrieval with generation, leaving open the question of whether current embedding models can encode persona-instruction compatibility without relying on response synthesis. We present a unified benchmark spanning 12 Indian languages and four evaluation tasks: monolingual and cross-lingual persona-to-instruction retrieval, reverse retrieval from instruction to persona, and binary compatibility classification. Eight multilingual embedding models are evaluated in a frozen-encoder setting with a thin logistic regression head for classification. E5-Large-Instruct achieves the highest Recall@1 of 27.4\% on monolingual retrieval and 20.7\% on cross-lingual transfer, while BGE-M3 leads reverse retrieval at 32.1\% Recall@1. For classification, LaBSE attains 75.3\% AUROC with strong calibration. These findings offer practical guidance for model selection in Indic multilingual retrieval and establish reproducible baselines for future work\footnote{Code, datasets, and models are publicly available at https://github.com/aryashah2k/PI-Indic-Align.
>
---
#### [new 030] Loop as a Bridge: Can Looped Transformers Truly Link Representation Space and Natural Language Outputs?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs内部知识与输出语言之间的差距问题。通过研究循环Transformer结构，分析其是否能通过迭代实现自我反思，从而连接表征空间与自然语言输出。**

- **链接: [https://arxiv.org/pdf/2601.10242v1](https://arxiv.org/pdf/2601.10242v1)**

> **作者:** Guanxu Chen; Dongrui Liu; Jing Shao
>
> **备注:** 9 pages,6 figures
>
> **摘要:** Large Language Models (LLMs) often exhibit a gap between their internal knowledge and their explicit linguistic outputs. In this report, we empirically investigate whether Looped Transformers (LTs)--architectures that increase computational depth by iterating shared layers--can bridge this gap by utilizing their iterative nature as a form of introspection. Our experiments reveal that while increasing loop iterations narrows the gap, it is partly driven by a degradation of their internal knowledge carried by representations. Moreover, another empirical analysis suggests that current LTs' ability to perceive representations does not improve across loops; it is only present in the final loop. These results suggest that while LTs offer a promising direction for scaling computational depth, they have yet to achieve the introspection required to truly link representation space and natural language.
>
---
#### [new 031] From Detection to Diagnosis: Advancing Hallucination Analysis with Automated Data Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 诊断任务，旨在解决LLM生成内容不准确的问题。通过构建诊断模型和生成训练数据，提升模型的可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2601.09734v1](https://arxiv.org/pdf/2601.09734v1)**

> **作者:** Yanyi Liu; Qingwen Yang; Tiezheng Guo; Feiyu Qu; Jun Liu; Yingyou Wen
>
> **备注:** Accepted at The 40th Annual AAAI Conference on Artificial Intelligence
>
> **摘要:** Hallucinations in Large Language Models (LLMs), defined as the generation of content inconsistent with facts or context, represent a core obstacle to their reliable deployment in critical domains. Current research primarily focuses on binary "detection" approaches that, while capable of identifying hallucinations, fail to provide interpretable and actionable feedback for model improvement, thus limiting practical utility. To address this limitation, a new research paradigm is proposed, shifting from "detection" to "diagnosis". The Hallucination Diagnosis Task is introduced, a task which requires models to not only detect hallucinations, but also perform error localization, causal explanation, and content correction. We develop the Hallucination Diagnosis Generator (HDG), an automated pipeline that systematically generates high-quality training samples with rich diagnostic metadata from raw corpora through multi-dimensional augmentation strategies including controlled fact fabrication and reasoning chain perturbation. Using HDG-generated data, we train HDM-4B-RL, a 4-billion-parameter hallucination diagnosis model, employing Group Relative Policy Optimization (GRPO) with a comprehensive reward function incorporating structural, accuracy, and localization signals. Experimental results demonstrate that our model surpasses previous state-of-the-art detection models on the HaluEval benchmark while achieving comparable performance to advanced general-purpose models. In comprehensive diagnosis tasks, HDM-4B-RL matches the capabilities of larger general models while maintaining a smaller size. This work validates the feasibility and value of hallucination diagnosis, providing an effective methodology for building more trustworthy and reliable generative AI systems.
>
---
#### [new 032] Grounding Agent Memory in Contextual Intent
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于对话系统任务，解决长周期目标交互中记忆检索不准确的问题。提出STITCH系统，通过上下文意图索引提升记忆检索准确性。**

- **链接: [https://arxiv.org/pdf/2601.10702v1](https://arxiv.org/pdf/2601.10702v1)**

> **作者:** Ruozhen Yang; Yucheng Jiang; Yueqi Jiang; Priyanka Kargupta; Yunyi Zhang; Jiawei Han
>
> **摘要:** Deploying large language models in long-horizon, goal-oriented interactions remains challenging because similar entities and facts recur under different latent goals and constraints, causing memory systems to retrieve context-mismatched evidence. We propose STITCH (Structured Intent Tracking in Contextual History), an agentic memory system that indexes each trajectory step with a structured retrieval cue, contextual intent, and retrieves history by matching the current step's intent. Contextual intent provides compact signals that disambiguate repeated mentions and reduce interference: (1) the current latent goal defining a thematic segment, (2) the action type, and (3) the salient entity types anchoring which attributes matter. During inference, STITCH filters and prioritizes memory snippets by intent compatibility, suppressing semantically similar but context-incompatible history. For evaluation, we introduce CAME-Bench, a benchmark for context-aware retrieval in realistic, dynamic, goal-oriented trajectories. Across CAME-Bench and LongMemEval, STITCH achieves state-of-the-art performance, outperforming the strongest baseline by 35.6%, with the largest gains as trajectory length increases. Our analysis shows that intent indexing substantially reduces retrieval noise, supporting intent-aware memory for robust long-horizon reasoning.
>
---
#### [new 033] OUTLINEFORGE: Hierarchical Reinforcement Learning with Explicit States for Scientific Writing
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于科学写作任务，旨在解决论文生成中的结构不连贯和引用不准确问题。提出一种分层强化学习框架，通过显式状态建模提升文档结构和引用一致性。**

- **链接: [https://arxiv.org/pdf/2601.09858v1](https://arxiv.org/pdf/2601.09858v1)**

> **作者:** Yilin Bao; Ziyao He; Zayden Yang
>
> **摘要:** Scientific paper generation requires document-level planning and factual grounding, but current large language models, despite their strong local fluency, often fail in global structure, input coverage, and citation consistency. We present a reinforcement learning framework that casts scientific outline construction as a long-horizon planning problem over hierarchical document structures. Our approach models edit evolving outlines through structured actions, enabling the system to incrementally build a complete scientific manuscript. To support effective and stabilize learning,we introduce a two-stage optimization procedure consisting of (i) backward outline reconstruction from partial plans to enforce global structural consistency, and (ii) forward value-guided reinforcement learning with rewards explicitly modeling scientific correctness, discourse coherence, and citation fidelity. In addition, We further introduce a benchmark for scientific paper generation that evaluates document planning, input utilization, reference faithfulness, outline organization, and content-level factual accuracy. Our results show consistent improvements over strong neural and LLM baselines, particularly in long-range structural coherence and citation reliability.
>
---
#### [new 034] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨语言模型 surprisal 优于 cloze surprisal 的原因，分析其在预测任务中的优势，并提出改进 cloze 研究的建议。**

- **链接: [https://arxiv.org/pdf/2601.09886v1](https://arxiv.org/pdf/2601.09886v1)**

> **作者:** Sathvik Nair; Byung-Doh Oh
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** How predictable a word is can be quantified in two ways: using human responses to the cloze task or using probabilities from language models (LMs).When used as predictors of processing effort, LM probabilities outperform probabilities derived from cloze data. However, it is important to establish that LM probabilities do so for the right reasons, since different predictors can lead to different scientific conclusions about the role of prediction in language comprehension. We present evidence for three hypotheses about the advantage of LM probabilities: not suffering from low resolution, distinguishing semantically similar words, and accurately assigning probabilities to low-frequency words. These results call for efforts to improve the resolution of cloze studies, coupled with experiments on whether human-like prediction is also as sensitive to the fine-grained distinctions made by LM probabilities.
>
---
#### [new 035] Syntactic Framing Fragility: An Audit of Robustness in LLM Ethical Decisions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于伦理决策任务，研究LLM在不同语法结构下的伦理判断一致性问题。通过SFF框架评估模型鲁棒性，发现语法变化导致显著不一致，提出链式思考可缓解此问题。**

- **链接: [https://arxiv.org/pdf/2601.09724v1](https://arxiv.org/pdf/2601.09724v1)**

> **作者:** Katherine Elkins; Jon Chun
>
> **摘要:** Large language models (LLMs) are increasingly deployed in consequential decision-making settings, yet their robustness to benign prompt variation remains underexplored. In this work, we study whether LLMs maintain consistent ethical judgments across logically equivalent but syntactically different prompts, focusing on variations involving negation and conditional structure. We introduce Syntactic Framing Fragility (SFF), a robustness evaluation framework that isolates purely syntactic effects via Logical Polarity Normalization (LPN), enabling direct comparison of decisions across positive and negative framings without semantic drift. Auditing 23 state-of-the-art models spanning the U.S. and China as well as small U.S. open-source software models over 14 ethical scenarios and four controlled framings (39,975 decisions), we find widespread and statistically significant inconsistency: many models reverse ethical endorsements solely due to syntactic polarity, with open-source models exhibiting over twice the fragility of commercial counterparts. We further uncover extreme negation sensitivity, where some models endorse actions in 80-97% of cases when explicitly prompted with "should not." We show that eliciting chain-of-thought reasoning substantially reduces fragility, identifying a practical mitigation lever, and we map fragility across scenarios, finding higher risk in financial and business contexts than in medical scenarios. Our results demonstrate that syntactic consistency constitutes a distinct and critical dimension of ethical robustness, and we argue that SFF-style audits should be a standard component of safety evaluation for deployed LLMs. Code and results will be available on github.com.
>
---
#### [new 036] The Straight and Narrow: Do LLMs Possess an Internal Moral Path?
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决LLMs道德对齐问题。通过MFT分析道德表示，提出AMF方法提升模型安全性与帮助性。**

- **链接: [https://arxiv.org/pdf/2601.10307v1](https://arxiv.org/pdf/2601.10307v1)**

> **作者:** Luoming Hu; Jingjie Zeng; Liang Yang; Hongfei Lin
>
> **摘要:** Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical results confirm that our approach acts as a targeted intrinsic defense, effectively reducing incorrect refusals on benign queries while minimizing jailbreak success rates compared to standard baselines.
>
---
#### [new 037] coTherapist: A Behavior-Aligned Small Language Model to Support Mental Healthcare Experts
- **分类: cs.CL**

- **简介: 该论文提出coTherapist，一个支持心理健康专家的小型语言模型，解决医疗资源不足问题。通过微调和推理增强，生成符合临床需求的回应。**

- **链接: [https://arxiv.org/pdf/2601.10246v1](https://arxiv.org/pdf/2601.10246v1)**

> **作者:** Prottay Kumar Adhikary; Reena Rawat; Tanmoy Chakraborty
>
> **摘要:** Access to mental healthcare is increasingly strained by workforce shortages and rising demand, motivating the development of intelligent systems that can support mental healthcare experts. We introduce coTherapist, a unified framework utilizing a small language model to emulate core therapeutic competencies through domain-specific fine-tuning, retrieval augmentation, and agentic reasoning. Evaluation on clinical queries demonstrates that coTherapist generates more relevant and clinically grounded responses than contemporary baselines. Using our novel T-BARS rubric and psychometric profiling, we confirm coTherapist exhibits high empathy and therapist-consistent personality traits. Furthermore, human evaluation by domain experts validates that coTherapist delivers accurate, trustworthy, and safe responses. coTherapist was deployed and tested by clinical experts. Collectively, these findings demonstrate that small models can be engineered to exhibit expert-like behavior, offering a scalable pathway for digital mental health tools.
>
---
#### [new 038] Is MT Ready for the Next Crisis or Pandemic?
- **分类: cs.CL**

- **简介: 该论文属于机器翻译评估任务，旨在解决危机场景下低资源语言翻译有效性问题。通过测试四种商业MT系统，评估其在疫情相关文本中的可用性。**

- **链接: [https://arxiv.org/pdf/2601.10082v1](https://arxiv.org/pdf/2601.10082v1)**

> **作者:** Vipasha Bansal; Elizabeth Brown; Chelsea Kendrick; Benjamin Pong; William D. Lewis
>
> **摘要:** Communication in times of crisis is essential. However, there is often a mismatch between the language of governments, aid providers, doctors, and those to whom they are providing aid. Commercial MT systems are reasonable tools to turn to in these scenarios. But how effective are these tools for translating to and from low resource languages, particularly in the crisis or medical domain? In this study, we evaluate four commercial MT systems using the TICO-19 dataset, which is composed of pandemic-related sentences from a large set of high priority languages spoken by communities most likely to be affected adversely in the next pandemic. We then assess the current degree of ``readiness'' for another pandemic (or epidemic) based on the usability of the output translations.
>
---
#### [new 039] EmplifAI: a Fine-grained Dataset for Japanese Empathetic Medical Dialogues in 28 Emotion Labels
- **分类: cs.CL**

- **简介: 该论文提出EmplifAI，一个用于日本医疗共情对话的细粒度情感数据集，解决情感识别与共情对话生成问题，包含28种情绪标签和4125条对话。**

- **链接: [https://arxiv.org/pdf/2601.10033v1](https://arxiv.org/pdf/2601.10033v1)**

> **作者:** Wan Jou She; Lis Kanashiro Pereira; Fei Cheng; Sakiko Yahata; Panote Siriaraya; Eiji Aramaki
>
> **摘要:** This paper introduces EmplifAI, a Japanese empathetic dialogue dataset designed to support patients coping with chronic medical conditions. They often experience a wide range of positive and negative emotions (e.g., hope and despair) that shift across different stages of disease management. EmplifAI addresses this complexity by providing situation-based dialogues grounded in 28 fine-grained emotion categories, adapted and validated from the GoEmotions taxonomy. The dataset includes 280 medically contextualized situations and 4125 two-turn dialogues, collected through crowdsourcing and expert review. To evaluate emotional alignment in empathetic dialogues, we assessed model predictions on situation--dialogue pairs using BERTScore across multiple large language models (LLMs), achieving F1 scores of 0.83. Fine-tuning a baseline Japanese LLM (LLM-jp-3.1-13b-instruct4) with EmplifAI resulted in notable improvements in fluency, general empathy, and emotion-specific empathy. Furthermore, we compared the scores assigned by LLM-as-a-Judge and human raters on dialogues generated by multiple LLMs to validate our evaluation pipeline and discuss the insights and potential risks derived from the correlation analysis.
>
---
#### [new 040] Alignment Pretraining: AI Discourse Causes Self-Fulfilling (Mis)alignment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究AI预训练数据对模型对齐的影响，属于AI对齐任务。旨在解决预训练内容如何影响模型行为的问题，通过实验验证了预训练数据的论述会引发自证式对齐或偏差。**

- **链接: [https://arxiv.org/pdf/2601.10160v1](https://arxiv.org/pdf/2601.10160v1)**

> **作者:** Cameron Tice; Puria Radmard; Samuel Ratnam; Andy Kim; David Africa; Kyle O'Brien
>
> **摘要:** Pretraining corpora contain extensive discourse about AI systems, yet the causal influence of this discourse on downstream alignment remains poorly understood. If prevailing descriptions of AI behaviour are predominantly negative, LLMs may internalise corresponding behavioural priors, giving rise to self-fulfilling misalignment. This paper provides the first controlled study of this hypothesis by pretraining 6.9B-parameter LLMs with varying amounts of (mis)alignment discourse. We find that discussion of AI contributes to misalignment. Upsampling synthetic training documents about AI misalignment leads to a notable increase in misaligned behaviour. Conversely, upsampling documents about aligned behaviour reduces misalignment scores from 45% to 9%. We consider this evidence of self-fulfilling alignment. These effects are dampened, but persist through post-training. Our findings establish the study of how pretraining data shapes alignment priors, or alignment pretraining, as a complement to post-training. We recommend practitioners pretrain for alignment as well as capabilities. Our models and datasets are available at alignmentpretraining.ai
>
---
#### [new 041] SALP-CG: Standard-Aligned LLM Pipeline for Classifying and Grading Large Volumes of Online Conversational Health Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于健康数据分类与风险评估任务，解决在线医疗对话数据隐私分类和敏感度分级问题。提出SALP-CG框架，实现标准对齐的分类与分级。**

- **链接: [https://arxiv.org/pdf/2601.09717v1](https://arxiv.org/pdf/2601.09717v1)**

> **作者:** Yiwei Yan; Hao Li; Hua He; Gong Kai; Zhengyi Yang; Guanfeng Liu
>
> **摘要:** Online medical consultations generate large volumes of conversational health data that often embed protected health information, requiring robust methods to classify data categories and assign risk levels in line with policies and practice. However, existing approaches lack unified standards and reliable automated methods to fulfill sensitivity classification for such conversational health data. This study presents a large language model-based extraction pipeline, SALP-CG, for classifying and grading privacy risks in online conversational health data. We concluded health-data classification and grading rules in accordance with GB/T 39725-2020. Combining few-shot guidance, JSON Schema constrained decoding, and deterministic high-risk rules, the backend-agnostic extraction pipeline achieves strong category compliance and reliable sensitivity across diverse LLMs. On the MedDialog-CN benchmark, models yields robust entity counts, high schema compliance, and accurate sensitivity grading, while the strongest model attains micro-F1=0.900 for maximum-level prediction. The category landscape stratified by sensitivity shows that Level 2-3 items dominate, enabling re-identification when combined; Level 4-5 items are less frequent but carry outsize harm. SALP-CG reliably helps classify categories and grading sensitivity in online conversational health data across LLMs, offering a practical method for health data governance. Code is available at https://github.com/dommii1218/SALP-CG.
>
---
#### [new 042] Eliminating Agentic Workflow for Introduction Generation with Parametric Stage Tokens
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文献引言生成任务，旨在解决传统代理工作流导致的逻辑冗长与文本不连贯问题。通过引入STIG参数化阶段标记，实现单次推理生成结构化引言。**

- **链接: [https://arxiv.org/pdf/2601.09728v1](https://arxiv.org/pdf/2601.09728v1)**

> **作者:** Meicong Zhang; Tiancheng su; Guoxiu He
>
> **摘要:** In recent years, using predefined agentic workflows to guide large language models (LLMs) for literature classification and review has become a research focus. However, writing research introductions is more challenging. It requires rigorous logic, coherent structure, and abstract summarization. Existing workflows often suffer from long reasoning chains, error accumulation, and reduced textual coherence. To address these limitations, we propose eliminating external agentic workflows. Instead, we directly parameterize their logical structure into the LLM. This allows the generation of a complete introduction in a single inference. To this end, we introduce the Stage Token for Introduction Generation (STIG). STIG converts the multiple stages of the original workflow into explicit stage signals. These signals guide the model to follow different logical roles and functions during generation. Through instruction tuning, the model learns the mapping between stage tokens and text functions. It also learns the logical order and transition patterns between stages, encoding this knowledge into the model parameters. Experimental results show that STIG can generate multi-stage text in a single inference. It does not require explicit workflow calls. STIG outperforms traditional agentic workflows and other baselines on metrics of semantic similarity and sentence-level structural rationality. The code is provided in the Supplementary Materials.
>
---
#### [new 043] INDIC DIALECT: A Multi Task Benchmark to Evaluate and Translate in Indian Language Dialects
- **分类: cs.CL**

- **简介: 该论文提出INDIC-DIALECT基准，解决印度方言NLP资源不足的问题。构建了包含11种方言的平行语料库，开展方言分类、问答和翻译任务，提升低资源方言的模型性能。**

- **链接: [https://arxiv.org/pdf/2601.10388v1](https://arxiv.org/pdf/2601.10388v1)**

> **作者:** Tarun Sharma; Manikandan Ravikiran; Sourava Kumar Behera; Pramit Bhattacharya; Arnab Bhattacharya; Rohit Saluja
>
> **摘要:** Recent NLP advances focus primarily on standardized languages, leaving most low-resource dialects under-served especially in Indian scenarios. In India, the issue is particularly important: despite Hindi being the third most spoken language globally (over 600 million speakers), its numerous dialects remain underrepresented. The situation is similar for Odia, which has around 45 million speakers. While some datasets exist which contain standard Hindi and Odia languages, their regional dialects have almost no web presence. We introduce INDIC-DIALECT, a human-curated parallel corpus of 13k sentence pairs spanning 11 dialects and 2 languages: Hindi and Odia. Using this corpus, we construct a multi-task benchmark with three tasks: dialect classification, multiple-choice question (MCQ) answering, and machine translation (MT). Our experiments show that LLMs like GPT-4o and Gemini 2.5 perform poorly on the classification task. While fine-tuned transformer based models pretrained on Indian languages substantially improve performance e.g., improving F1 from 19.6\% to 89.8\% on dialect classification. For dialect to language translation, we find that hybrid AI model achieves highest BLEU score of 61.32 compared to the baseline score of 23.36. Interestingly, due to complexity in generating dialect sentences, we observe that for language to dialect translation the ``rule-based followed by AI" approach achieves best BLEU score of 48.44 compared to the baseline score of 27.59. INDIC-DIALECT thus is a new benchmark for dialect-aware Indic NLP, and we plan to release it as open source to support further work on low-resource Indian dialects.
>
---
#### [new 044] LIBERTy: A Causal Framework for Benchmarking Concept-Based Explanations of LLMs with Structural Counterfactuals
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LIBERTy框架，用于评估大模型的概念解释可靠性。任务是提升解释的因果准确性，通过结构反事实数据集和新指标进行评估。**

- **链接: [https://arxiv.org/pdf/2601.10700v1](https://arxiv.org/pdf/2601.10700v1)**

> **作者:** Gilat Toker; Nitay Calderon; Ohad Amosy; Roi Reichart
>
> **摘要:** Concept-based explanations quantify how high-level concepts (e.g., gender or experience) influence model behavior, which is crucial for decision-makers in high-stakes domains. Recent work evaluates the faithfulness of such explanations by comparing them to reference causal effects estimated from counterfactuals. In practice, existing benchmarks rely on costly human-written counterfactuals that serve as an imperfect proxy. To address this, we introduce a framework for constructing datasets containing structural counterfactual pairs: LIBERTy (LLM-based Interventional Benchmark for Explainability with Reference Targets). LIBERTy is grounded in explicitly defined Structured Causal Models (SCMs) of the text generation, interventions on a concept propagate through the SCM until an LLM generates the counterfactual. We introduce three datasets (disease detection, CV screening, and workplace violence prediction) together with a new evaluation metric, order-faithfulness. Using them, we evaluate a wide range of methods across five models and identify substantial headroom for improving concept-based explanations. LIBERTy also enables systematic analysis of model sensitivity to interventions: we find that proprietary LLMs show markedly reduced sensitivity to demographic concepts, likely due to post-training mitigation. Overall, LIBERTy provides a much-needed benchmark for developing faithful explainability methods.
>
---
#### [new 045] Geometric Patterns of Meaning: A PHATE Manifold Analysis of Multi-lingual Embeddings
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在分析多语言嵌入的语义几何结构。通过PHATE方法，研究不同语言层级的几何模式，揭示现有模型的局限性。**

- **链接: [https://arxiv.org/pdf/2601.09731v1](https://arxiv.org/pdf/2601.09731v1)**

> **作者:** Wen G Gong
>
> **摘要:** We introduce a multi-level analysis framework for examining semantic geometry in multilingual embeddings, implemented through Semanscope (a visualization tool that applies PHATE manifold learning across four linguistic levels). Analysis of diverse datasets spanning sub-character components, alphabetic systems, semantic domains, and numerical concepts reveals systematic geometric patterns and critical limitations in current embedding models. At the sub-character level, purely structural elements (Chinese radicals) exhibit geometric collapse, highlighting model failures to distinguish semantic from structural components. At the character level, different writing systems show distinct geometric signatures. At the word level, content words form clustering-branching patterns across 20 semantic domains in English, Chinese, and German. Arabic numbers organize through spiral trajectories rather than clustering, violating standard distributional semantics assumptions. These findings establish PHATE manifold learning as an essential analytic tool not only for studying geometric structure of meaning in embedding space, but also for validating the effectiveness of embedding models in capturing semantic relationships.
>
---
#### [new 046] Influential Training Data Retrieval for Explaining Verbalized Confidence of LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs verbalized confidence不可靠的问题。通过追踪训练数据，分析其信心表达的来源，提出新指标评估内容相关性。**

- **链接: [https://arxiv.org/pdf/2601.10645v1](https://arxiv.org/pdf/2601.10645v1)**

> **作者:** Yuxi Xia; Loris Schoenegger; Benjamin Roth
>
> **摘要:** Large language models (LLMs) can increase users' perceived trust by verbalizing confidence in their outputs. However, prior work has shown that LLMs are often overconfident, making their stated confidence unreliable since it does not consistently align with factual accuracy. To better understand the sources of this verbalized confidence, we introduce TracVC (\textbf{Trac}ing \textbf{V}erbalized \textbf{C}onfidence), a method that builds on information retrieval and influence estimation to trace generated confidence expressions back to the training data. We evaluate TracVC on OLMo and Llama models in a question answering setting, proposing a new metric, content groundness, which measures the extent to which an LLM grounds its confidence in content-related training examples (relevant to the question and answer) versus in generic examples of confidence verbalization. Our analysis reveals that OLMo2-13B is frequently influenced by confidence-related data that is lexically unrelated to the query, suggesting that it may mimic superficial linguistic expressions of certainty rather than rely on genuine content grounding. These findings point to a fundamental limitation in current training regimes: LLMs may learn how to sound confident without learning when confidence is justified. Our analysis provides a foundation for improving LLMs' trustworthiness in expressing more reliable confidence.
>
---
#### [new 047] DR-Arena: an Automated Evaluation Framework for Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文提出DR-Arena，用于评估深度研究代理的性能。解决静态数据集评估不足的问题，通过动态任务测试代理的推理和覆盖能力，实现高效可靠评估。**

- **链接: [https://arxiv.org/pdf/2601.10504v1](https://arxiv.org/pdf/2601.10504v1)**

> **作者:** Yiwen Gao; Ruochen Zhao; Yang Deng; Wenxuan Zhang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** As Large Language Models (LLMs) increasingly operate as Deep Research (DR) Agents capable of autonomous investigation and information synthesis, reliable evaluation of their task performance has become a critical bottleneck. Current benchmarks predominantly rely on static datasets, which suffer from several limitations: limited task generality, temporal misalignment, and data contamination. To address these, we introduce DR-Arena, a fully automated evaluation framework that pushes DR agents to their capability limits through dynamic investigation. DR-Arena constructs real-time Information Trees from fresh web trends to ensure the evaluation rubric is synchronized with the live world state, and employs an automated Examiner to generate structured tasks testing two orthogonal capabilities: Deep reasoning and Wide coverage. DR-Arena further adopts Adaptive Evolvement Loop, a state-machine controller that dynamically escalates task complexity based on real-time performance, demanding deeper deduction or wider aggregation until a decisive capability boundary emerges. Experiments with six advanced DR agents demonstrate that DR-Arena achieves a Spearman correlation of 0.94 with the LMSYS Search Arena leaderboard. This represents the state-of-the-art alignment with human preferences without any manual efforts, validating DR-Arena as a reliable alternative for costly human adjudication.
>
---
#### [new 048] Cross-Platform Evaluation of Large Language Model Safety in Pediatric Consultations: Evolution of Adversarial Robustness and the Scale Paradox
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI安全评估任务，旨在解决大语言模型在儿科咨询中的安全性问题。通过对抗性测试，评估不同模型在压力下的表现，发现模型规模并非决定因素。**

- **链接: [https://arxiv.org/pdf/2601.09721v1](https://arxiv.org/pdf/2601.09721v1)**

> **作者:** Vahideh Zolfaghari
>
> **摘要:** Background Large language models (LLMs) are increasingly deployed in medical consultations, yet their safety under realistic user pressures remains understudied. Prior assessments focused on neutral conditions, overlooking vulnerabilities from anxious users challenging safeguards. This study evaluated LLM safety under parental anxiety-driven adversarial pressures in pediatric consultations across models and platforms. Methods PediatricAnxietyBench, from a prior evaluation, includes 300 queries (150 authentic, 150 adversarial) spanning 10 topics. Three models were assessed via APIs: Llama-3.3-70B and Llama-3.1-8B (Groq), Mistral-7B (HuggingFace), yielding 900 responses. Safety used a 0-15 scale for restraint, referral, hedging, emergency recognition, and non-prescriptive behavior. Analyses employed paired t-tests with bootstrapped CIs. Results Mean scores: 9.70 (Llama-3.3-70B) to 10.39 (Mistral-7B). Llama-3.1-8B outperformed Llama-3.3-70B by +0.66 (p=0.0001, d=0.225). Models showed positive adversarial effects, Mistral-7B strongest (+1.09, p=0.0002). Safety generalized across platforms; Llama-3.3-70B had 8% failures. Seizures vulnerable (33% inappropriate diagnoses). Hedging predicted safety (r=0.68, p<0.001). Conclusions Evaluation shows safety depends on alignment and architecture over scale, with smaller models outperforming larger. Evolution to robustness across releases suggests targeted training progress. Vulnerabilities and no emergency recognition indicate unsuitability for triage. Findings guide selection, stress adversarial testing, and provide open benchmark for medical AI safety.
>
---
#### [new 049] AEQ-Bench: Measuring Empathy of Omni-Modal Large Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于情感评估任务，旨在解决多模态大模型 empathetic 能力的测评问题。提出 AEQ-Bench 基准，评估模型在多模态输入下生成共情回复及判断语音共情的能力。**

- **链接: [https://arxiv.org/pdf/2601.10513v1](https://arxiv.org/pdf/2601.10513v1)**

> **作者:** Xuan Luo; Lewei Yao; Libo Zhao; Lanqing Hong; Kai Chen; Dehua Tao; Daxin Tan; Ruifeng Xu; Jing Li
>
> **摘要:** While the automatic evaluation of omni-modal large models (OLMs) is essential, assessing empathy remains a significant challenge due to its inherent affectivity. To investigate this challenge, we introduce AEQ-Bench (Audio Empathy Quotient Benchmark), a novel benchmark to systematically assess two core empathetic capabilities of OLMs: (i) generating empathetic responses by comprehending affective cues from multi-modal inputs (audio + text), and (ii) judging the empathy of audio responses without relying on text transcription. Compared to existing benchmarks, AEQ-Bench incorporates two novel settings that vary in context specificity and speech tone. Comprehensive assessment across linguistic and paralinguistic metrics reveals that (1) OLMs trained with audio output capabilities generally outperformed models with text-only outputs, and (2) while OLMs align with human judgments for coarse-grained quality assessment, they remain unreliable for evaluating fine-grained paralinguistic expressiveness.
>
---
#### [new 050] Introducing Axlerod: An LLM-based Chatbot for Assisting Independent Insurance Agents
- **分类: cs.CL; cs.AI; cs.HC; cs.IR**

- **简介: 该论文介绍Axlerod，一个基于LLM的聊天机器人，用于辅助独立保险代理人。任务是提升保险行业效率，解决政策检索与用户交互问题，通过NLP和知识整合实现精准响应。**

- **链接: [https://arxiv.org/pdf/2601.09715v1](https://arxiv.org/pdf/2601.09715v1)**

> **作者:** Adam Bradley; John Hastings; Khandaker Mamun Ahmed
>
> **备注:** 6 pages, 2 figures, 1 table
>
> **摘要:** The insurance industry is undergoing a paradigm shift through the adoption of artificial intelligence (AI) technologies, particularly in the realm of intelligent conversational agents. Chatbots have evolved into sophisticated AI-driven systems capable of automating complex workflows, including policy recommendation and claims triage, while simultaneously enabling dynamic, context-aware user engagement. This paper presents the design, implementation, and empirical evaluation of Axlerod, an AI-powered conversational interface designed to improve the operational efficiency of independent insurance agents. Leveraging natural language processing (NLP), retrieval-augmented generation (RAG), and domain-specific knowledge integration, Axlerod demonstrates robust capabilities in parsing user intent, accessing structured policy databases, and delivering real-time, contextually relevant responses. Experimental results underscore Axlerod's effectiveness, achieving an overall accuracy of 93.18% in policy retrieval tasks while reducing the average search time by 2.42 seconds. This work contributes to the growing body of research on enterprise-grade AI applications in insurtech, with a particular focus on agent-assistive rather than consumer-facing architectures.
>
---
#### [new 051] Detecting Winning Arguments with Large Language Models and Persuasion Strategies
- **分类: cs.CL**

- **简介: 该论文属于论证质量评估任务，旨在检测文本的说服力。通过结合大语言模型与六种说服策略，提升说服力预测的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2601.10660v1](https://arxiv.org/pdf/2601.10660v1)**

> **作者:** Tiziano Labruna; Arkadiusz Modzelewski; Giorgio Satta; Giovanni Da San Martino
>
> **摘要:** Detecting persuasion in argumentative text is a challenging task with important implications for understanding human communication. This work investigates the role of persuasion strategies - such as Attack on reputation, Distraction, and Manipulative wording - in determining the persuasiveness of a text. We conduct experiments on three annotated argument datasets: Winning Arguments (built from the Change My View subreddit), Anthropic/Persuasion, and Persuasion for Good. Our approach leverages large language models (LLMs) with a Multi-Strategy Persuasion Scoring approach that guides reasoning over six persuasion strategies. Results show that strategy-guided reasoning improves the prediction of persuasiveness. To better understand the influence of content, we organize the Winning Argument dataset into broad discussion topics and analyze performance across them. We publicly release this topic-annotated version of the dataset to facilitate future research. Overall, our methodology demonstrates the value of structured, strategy-aware prompting for enhancing interpretability and robustness in argument quality assessment.
>
---
#### [new 052] Clinical Document Metadata Extraction: A Scoping Review
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于临床文档元数据提取任务，旨在解决元数据异质性和标准化难题，通过系统综述分析现有方法与应用，推动临床文本处理发展。**

- **链接: [https://arxiv.org/pdf/2601.09730v1](https://arxiv.org/pdf/2601.09730v1)**

> **作者:** Kurt Miller; Qiuhao Lu; William Hersh; Kirk Roberts; Steven Bedrick; Andrew Wen; Hongfang Liu
>
> **摘要:** Clinical document metadata, such as document type, structure, author role, medical specialty, and encounter setting, is essential for accurate interpretation of information captured in clinical documents. However, vast documentation heterogeneity and drift over time challenge harmonization of document metadata. Automated extraction methods have emerged to coalesce metadata from disparate practices into target schema. This scoping review aims to catalog research on clinical document metadata extraction, identify methodological trends and applications, and highlight gaps. We followed the PRISMA-ScR (Preferred Reporting Items for Systematic Reviews and Meta-Analyses Extension for Scoping Reviews) guidelines to identify articles that perform clinical document metadata extraction. We initially found and screened 266 articles published between January 2011 and August 2025, then comprehensively reviewed 67 we deemed relevant to our study. Among the articles included, 45 were methodological, 17 used document metadata as features in a downstream application, and 5 analyzed document metadata composition. We observe myriad purposes for methodological study and application types. Available labelled public data remains sparse except for structural section datasets. Methods for extracting document metadata have progressed from largely rule-based and traditional machine learning with ample feature engineering to transformer-based architectures with minimal feature engineering. The emergence of large language models has enabled broader exploration of generalizability across tasks and datasets, allowing the possibility of advanced clinical text processing systems. We anticipate that research will continue to expand into richer document metadata representations and integrate further into clinical applications and workflows.
>
---
#### [new 053] Form and Meaning in Intrinsic Multilingual Evaluations
- **分类: cs.CL**

- **简介: 该论文属于语言模型评估任务，探讨多语言环境下固有评估指标的适用性问题。研究指出现有指标在多语言场景下不可直接比较，并通过实验验证这一结论。**

- **链接: [https://arxiv.org/pdf/2601.10580v1](https://arxiv.org/pdf/2601.10580v1)**

> **作者:** Wessel Poelman; Miryam de Lhoneux
>
> **备注:** EACL 2026: Main Conference
>
> **摘要:** Intrinsic evaluation metrics for conditional language models, such as perplexity or bits-per-character, are widely used in both mono- and multilingual settings. These metrics are rather straightforward to use and compare in monolingual setups, but rest on a number of assumptions in multilingual setups. One such assumption is that comparing the perplexity of CLMs on parallel sentences is indicative of their quality since the information content (here understood as the semantic meaning) is the same. However, the metrics are inherently measuring information content in the information-theoretic sense. We make this and other such assumptions explicit and discuss their implications. We perform experiments with six metrics on two multi-parallel corpora both with mono- and multilingual models. Ultimately, we find that current metrics are not universally comparable. We look at the form-meaning debate to provide some explanation for this.
>
---
#### [new 054] Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型解释任务，旨在解决注意力头间关系理解的问题。通过投影核度量注意力头子空间相似性，提升对Transformer结构的解释能力。**

- **链接: [https://arxiv.org/pdf/2601.10266v1](https://arxiv.org/pdf/2601.10266v1)**

> **作者:** Hiroaki Yamagiwa; Yusuke Takase; Hidetoshi Shimodaira
>
> **摘要:** Understanding relationships between attention heads is essential for interpreting the internal structure of Transformers, yet existing metrics do not capture this structure well. We focus on the subspaces spanned by attention-head weight matrices and quantify head-to-head relationships using the Projection Kernel (PK), a principal-angle-based measure of subspace similarity. Experiments show that PK reproduces known head-to-head interactions on the IOI task more clearly than prior metrics such as the Composition Score. We further introduce a framework to quantify the informativeness of PK distributions by comparing them with a reference distribution derived from random orthogonal subspaces. As an application, we analyze a directed graph constructed from PK and show that, in GPT2-small, L4H7 acts as a hub by functioning as an identity head.
>
---
#### [new 055] Assessing and Improving Punctuation Robustness in English-Marathi Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决 punctuation 与翻译质量的关系。针对英语到马拉地语翻译中的标点鲁棒性问题，提出 Virām 基准，并通过微调和流水线方法提升翻译可靠性。**

- **链接: [https://arxiv.org/pdf/2601.09725v1](https://arxiv.org/pdf/2601.09725v1)**

> **作者:** Kaustubh Shivshankar Shejole; Sourabh Deoghare; Pushpak Bhattacharyya
>
> **摘要:** Punctuation plays a critical role in resolving semantic and structural ambiguity in written language. Machine Translation (MT) systems are now widely applied across diverse domains and languages, including many low-resource settings. In this work, we focus on Marathi, a low- to middle-resource language. We introduce Virām, the first diagnostic benchmark for assessing punctuation robustness in English-to-Marathi machine translation, consisting of 54 manually curated, punctuation-ambiguous instances. We evaluate two primary strategies for enhancing reliability: a pipeline-based restore-then-translate approach and direct fine-tuned on punctuation-varied data. Our results demonstrate that specialized fine-tuned models and pipeline systems significantly improve translation quality over standard baselines on the Virām benchmark. Qualitative analysis reveals that the original model may result in wrong translations leading to wrong interpretations, while fine-tuned models significantly improve overall reliability. Furthermore, we find that current Large Language Models (LLMs) lag behind these task-specific approaches in preserving meaning for punctuation-ambiguous text, thus necessitating further research in this area.
>
---
#### [new 056] EHRNavigator: A Multi-Agent System for Patient-Level Clinical Question Answering over Heterogeneous Electronic Health Records
- **分类: cs.CL**

- **简介: 该论文属于临床问答任务，旨在解决EHR中多源数据的患者级问题回答。提出EHRNavigator框架，实现高效准确的临床决策支持。**

- **链接: [https://arxiv.org/pdf/2601.10020v1](https://arxiv.org/pdf/2601.10020v1)**

> **作者:** Lingfei Qian; Mauro Giuffre; Yan Wang; Huan He; Qianqian Xie; Xuguang Ai; Xeuqing Peng; Fan Ma; Ruey-Ling Weng; Donald Wright; Adan Wang; Qingyu Chen; Vipina K. Keloth; Hua Xu
>
> **摘要:** Clinical decision-making increasingly relies on timely and context-aware access to patient information within Electronic Health Records (EHRs), yet most existing natural language question-answering (QA) systems are evaluated solely on benchmark datasets, limiting their practical relevance. To overcome this limitation, we introduce EHRNavigator, a multi-agent framework that harnesses AI agents to perform patient-level question answering across heterogeneous and multimodal EHR data. We assessed its performance using both public benchmark and institutional datasets under realistic hospital conditions characterized by diverse schemas, temporal reasoning demands, and multimodal evidence integration. Through quantitative evaluation and clinician-validated chart review, EHRNavigator demonstrated strong generalization, achieving 86% accuracy on real-world cases while maintaining clinically acceptable response times. Overall, these findings confirm that EHRNavigator effectively bridges the gap between benchmark evaluation and clinical deployment, offering a robust, adaptive, and efficient solution for real-world EHR question answering.
>
---
#### [new 057] Are Language Models Models?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨语言模型是否可作为认知模型。论文通过Marr三层次分析，指出语言模型在实现、算法和计算理论层面均不满足认知模型标准，强调其应被视为工具而非模型。**

- **链接: [https://arxiv.org/pdf/2601.10421v1](https://arxiv.org/pdf/2601.10421v1)**

> **作者:** Philip Resnik
>
> **备注:** 5 pages. This is an invited commentary under review at Behavioral and Brain Sciences
>
> **摘要:** Futrell and Mahowald claim LMs "serve as model systems", but an assessment at each of Marr's three levels suggests the claim is clearly not true at the implementation level, poorly motivated at the algorithmic-representational level, and problematic at the computational theory level. LMs are good candidates as tools; calling them cognitive models overstates the case and unnecessarily feeds LLM hype.
>
---
#### [new 058] Uncertainty-Aware Dynamic Knowledge Graphs for Reliable Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于问答系统任务，旨在解决知识图谱中信息不完整、噪声和不确定性问题。提出一种动态知识图谱框架，融合不确定性建模与交互界面，提升问答可靠性。**

- **链接: [https://arxiv.org/pdf/2601.09720v1](https://arxiv.org/pdf/2601.09720v1)**

> **作者:** Yu Takahashi; Shun Takeuchi; Kexuan Xin; Guillaume Pelat; Yoshiaki Ikai; Junya Saito; Jonathan Vitale; Shlomo Berkovsky; Amin Beheshti
>
> **备注:** 4 pages, 4 figures. Accepted at IEEE ICDM 2025 Demo Track
>
> **摘要:** Question answering (QA) systems are increasingly deployed across domains. However, their reliability is undermined when retrieved evidence is incomplete, noisy, or uncertain. Existing knowledge graph (KG) based QA frameworks typically represent facts as static and deterministic, failing to capture the evolving nature of information and the uncertainty inherent in reasoning. We present a demonstration of uncertainty-aware dynamic KGs, a framework that combines (i) dynamic construction of evolving KGs, (ii) confidence scoring and uncertainty-aware retrieval, and (iii) an interactive interface for reliable and interpretable QA. Our system highlights how uncertainty modeling can make QA more robust and transparent by enabling users to explore dynamic graphs, inspect confidence-annotated triples, and compare baseline versus confidence-aware answers. The target users of this demo are clinical data scientists and clinicians, and we instantiate the framework in healthcare: constructing personalized KGs from electronic health records, visualizing uncertainty across patient visits, and evaluating its impact on a mortality prediction task. This use case demonstrates the broader promise of uncertainty-aware dynamic KGs for enhancing QA reliability in high-stakes applications.
>
---
#### [new 059] AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出AWED-FiNER，解决36种语言的细粒度命名实体识别问题，通过代理工具、网页应用和专家模型提供高效解决方案。**

- **链接: [https://arxiv.org/pdf/2601.10161v1](https://arxiv.org/pdf/2601.10161v1)**

> **作者:** Prachuryya Kaushik; Ashish Anand
>
> **备注:** Submitted to ACL'26 System Demonstration
>
> **摘要:** We introduce AWED-FiNER, an open-source ecosystem designed to bridge the gap in Fine-grained Named Entity Recognition (FgNER) for 36 global languages spoken by more than 6.6 billion people. While Large Language Models (LLMs) dominate general Natural Language Processing (NLP) tasks, they often struggle with low-resource languages and fine-grained NLP tasks. AWED-FiNER provides a collection of agentic toolkits, web applications, and several state-of-the-art expert models that provides FgNER solutions across 36 languages. The agentic tools enable to route multilingual text to specialized expert models and fetch FgNER annotations within seconds. The web-based platforms provide ready-to-use FgNER annotation service for non-technical users. Moreover, the collection of language specific extremely small sized open-source state-of-the-art expert models facilitate offline deployment in resource contraint scenerios including edge devices. AWED-FiNER covers languages spoken by over 6.6 billion people, including a specific focus on vulnerable languages such as Bodo, Manipuri, Bishnupriya, and Mizo. The resources can be accessed here: Agentic Tool (https://github.com/PrachuryyaKaushik/AWED-FiNER), Web Application (https://hf.co/spaces/prachuryyaIITG/AWED-FiNER), and 49 Expert Detector Models (https://hf.co/collections/prachuryyaIITG/awed-finer).
>
---
#### [new 060] ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback
- **分类: cs.CL**

- **简介: 该论文属于LLM代理安全任务，旨在解决工具调用中的安全风险。通过构建基准和开发防护模型，提升代理在执行前识别危险操作的能力。**

- **链接: [https://arxiv.org/pdf/2601.10156v1](https://arxiv.org/pdf/2601.10156v1)**

> **作者:** Yutao Mou; Zhangchi Xue; Lijun Li; Peiyang Liu; Shikun Zhang; Wei Ye; Jing Shao
>
> **备注:** Work in Progress. Code available: https://github.com/MurrayTom/ToolSafe
>
> **摘要:** While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks.
>
---
#### [new 061] Context Volume Drives Performance: Tackling Domain Shift in Extremely Low-Resource Translation via RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，解决低资源语言在领域迁移下的性能下降问题。通过结合NMT与RAG技术提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2601.09982v1](https://arxiv.org/pdf/2601.09982v1)**

> **作者:** David Samuel Setiawan; Raphaël Merx; Jey Han Lau
>
> **摘要:** Neural Machine Translation (NMT) models for low-resource languages suffer significant performance degradation under domain shift. We quantify this challenge using Dhao, an indigenous language of Eastern Indonesia with no digital footprint beyond the New Testament (NT). When applied to the unseen Old Testament (OT), a standard NMT model fine-tuned on the NT drops from an in-domain score of 36.17 chrF++ to 27.11 chrF++. To recover this loss, we introduce a hybrid framework where a fine-tuned NMT model generates an initial draft, which is then refined by a Large Language Model (LLM) using Retrieval-Augmented Generation (RAG). The final system achieves 35.21 chrF++ (+8.10 recovery), effectively matching the original in-domain quality. Our analysis reveals that this performance is driven primarily by the number of retrieved examples rather than the choice of retrieval algorithm. Qualitative analysis confirms the LLM acts as a robust "safety net," repairing severe failures in zero-shot domains.
>
---
#### [new 062] Multilinguality as Sense Adaptation
- **分类: cs.CL**

- **简介: 该论文属于多语言任务，旨在解决跨语言语义对齐问题。通过SENsIA方法，实现语言间的语义适应，提升模型性能并减少目标语言数据需求。**

- **链接: [https://arxiv.org/pdf/2601.10310v1](https://arxiv.org/pdf/2601.10310v1)**

> **作者:** Jan Christian Blaise Cruz; David Ifeoluwa Adelani; Alham Fikri Aji
>
> **备注:** Code available at https://github.com/jcblaisecruz02/sensia
>
> **摘要:** We approach multilinguality as sense adaptation: aligning latent meaning representations across languages rather than relying solely on shared parameters and scale. In this paper, we introduce SENse-based Symmetric Interlingual Alignment (SENSIA), which adapts a Backpack language model from one language to another by explicitly aligning sense-level mixtures and contextual representations on parallel data, while jointly training a target-language language modeling loss to preserve fluency. Across benchmarks on four typologically diverse languages, SENSIA generally outperforms comparable multilingual alignment methods and achieves competitive accuracy against monolingual from-scratch baselines while using 2-4x less target-language data. Analyses of learned sense geometry indicate that local sense topology and global structure relative to English are largely preserved, and ablations show that the method is robust in terms of design and scale.
>
---
#### [new 063] CALM-IT: Generating Realistic Long-Form Motivational Interviewing Dialogues with Dual-Actor Conversational Dynamics Tracking
- **分类: cs.CL**

- **简介: 该论文属于对话生成任务，旨在解决长对话中保持目标导向和一致性的问题。提出CALM-IT框架，通过建模双角色互动动态，提升对话质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.10085v1](https://arxiv.org/pdf/2601.10085v1)**

> **作者:** Viet Cuong Nguyen; Nhi Yen Nguyen; Kristin A. Candan; Mary Conlon; Vanessa Rumie; Kristen Risola; Srijan Kumar; Munmun De Choudhury
>
> **备注:** 46 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly used in mental health-related settings, yet they struggle to sustain realistic, goal-directed dialogue over extended interactions. While LLMs generate fluent responses, they optimize locally for the next turn rather than maintaining a coherent model of therapeutic progress, leading to brittleness and long-horizon drift. We introduce CALM-IT, a framework for generating and evaluating long-form Motivational Interviewing (MI) dialogues that explicitly models dual-actor conversational dynamics. CALM-IT represents therapist-client interaction as a bidirectional state-space process, in which both agents continuously update inferred alignment, mental states, and short-term goals to guide strategy selection and utterance generation. Across large-scale evaluations, CALM-IT consistently outperforms strong baselines in Effectiveness and Goal Alignment and remains substantially more stable as conversation length increases. Although CALM-IT initiates fewer therapist redirections, it achieves the highest client acceptance rate (64.3%), indicating more precise and therapeutically aligned intervention timing. Overall, CALM-IT provides evidence for modeling evolving conversational state being essential for generating high-quality long-form synthetic conversations.
>
---
#### [new 064] MoST: Mixing Speech and Text with Modality-Aware Mixture of Experts
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文提出MoST模型，解决多模态语音与文本处理问题。通过MAMoE架构实现模态感知的专家混合，提升语音和文本任务性能。**

- **链接: [https://arxiv.org/pdf/2601.10272v1](https://arxiv.org/pdf/2601.10272v1)**

> **作者:** Yuxuan Lou; Kai Yang; Yang You
>
> **摘要:** We present MoST (Mixture of Speech and Text), a novel multimodal large language model that seamlessly integrates speech and text processing through our proposed Modality-Aware Mixture of Experts (MAMoE) architecture. While current multimodal models typically process diverse modality representations with identical parameters, disregarding their inherent representational differences, we introduce specialized routing pathways that direct tokens to modality-appropriate experts based on input type. MAMoE simultaneously enhances modality-specific learning and cross-modal understanding through two complementary components: modality-specific expert groups that capture domain-specific patterns and shared experts that facilitate information transfer between modalities. Building on this architecture, we develop an efficient transformation pipeline that adapts the pretrained MoE language model through strategic post-training on ASR and TTS datasets, followed by fine-tuning with a carefully curated speech-text instruction dataset. A key feature of this pipeline is that it relies exclusively on fully accessible, open-source datasets to achieve strong performance and data efficiency. Comprehensive evaluations across ASR, TTS, audio language modeling, and spoken question answering benchmarks show that MoST consistently outperforms existing models of comparable parameter counts. Our ablation studies confirm that the modality-specific routing mechanism and shared experts design significantly contribute to performance gains across all tested domains. To our knowledge, MoST represents the first fully open-source speech-text LLM built on a Mixture of Experts architecture. \footnote{We release MoST model, training code, inference code, and training data at https://github.com/NUS-HPC-AI-Lab/MoST
>
---
#### [new 065] OctoBench: Benchmarking Scaffold-Aware Instruction Following in Repository-Grounded Agentic Coding
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出OctoBench，用于评估代码生成中遵循结构化指令的能力。针对代码代理在复杂约束下遵循指令不足的问题，构建了包含多种环境和任务的基准，以促进更精准的训练与评估。**

- **链接: [https://arxiv.org/pdf/2601.10343v1](https://arxiv.org/pdf/2601.10343v1)**

> **作者:** Deming Ding; Shichun Liu; Enhui Yang; Jiahang Lin; Ziying Chen; Shihan Dou; Honglin Guo; Weiyu Cheng; Pengyu Zhao; Chengjun Xiao; Qunhong Zeng; Qi Zhang; Xuanjing Huang; Qidi Xu; Tao Gui
>
> **摘要:** Modern coding scaffolds turn LLMs into capable software agents, but their ability to follow scaffold-specified instructions remains under-examined, especially when constraints are heterogeneous and persist across interactions. To fill this gap, we introduce OctoBench, which benchmarks scaffold-aware instruction following in repository-grounded agentic coding. OctoBench includes 34 environments and 217 tasks instantiated under three scaffold types, and is paired with 7,098 objective checklist items. To disentangle solving the task from following the rules, we provide an automated observation-and-scoring toolkit that captures full trajectories and performs fine-grained checks. Experiments on eight representative models reveal a systematic gap between task-solving and scaffold-aware compliance, underscoring the need for training and evaluation that explicitly targets heterogeneous instruction following. We release the benchmark to support reproducible benchmarking and to accelerate the development of more scaffold-aware coding agents.
>
---
#### [new 066] ADVOSYNTH: A Synthetic Multi-Advocate Dataset for Speaker Identification in Courtroom Scenarios
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决法庭场景下合成语音的说话人识别问题。构建了包含10个合成说话人的数据集，用于评估系统识别能力。**

- **链接: [https://arxiv.org/pdf/2601.10315v1](https://arxiv.org/pdf/2601.10315v1)**

> **作者:** Aniket Deroy
>
> **摘要:** As large-scale speech-to-speech models achieve high fidelity, the distinction between synthetic voices in structured environments becomes a vital area of study. This paper introduces Advosynth-500, a specialized dataset comprising 100 synthetic speech files featuring 10 unique advocate identities. Using the Speech Llama Omni model, we simulate five distinct advocate pairs engaged in courtroom arguments. We define specific vocal characteristics for each advocate and present a speaker identification challenge to evaluate the ability of modern systems to map audio files to their respective synthetic origins. Dataset is available at this link-https: //github.com/naturenurtureelite/ADVOSYNTH-500.
>
---
#### [new 067] Take Out Your Calculators: Estimating the Real Difficulty of Question Items with LLM Student Simulations
- **分类: cs.CL**

- **简介: 该论文属于教育评估任务，旨在用LLM模拟学生解答数学题，预测题目难度。通过角色扮演和IRT模型，验证了LLM在不同年级的预测效果。**

- **链接: [https://arxiv.org/pdf/2601.09953v1](https://arxiv.org/pdf/2601.09953v1)**

> **作者:** Christabel Acquaye; Yi Ting Huang; Marine Carpuat; Rachel Rudinger
>
> **摘要:** Standardized math assessments require expensive human pilot studies to establish the difficulty of test items. We investigate the predictive value of open-source large language models (LLMs) for evaluating the difficulty of multiple-choice math questions for real-world students. We show that, while LLMs are poor direct judges of problem difficulty, simulation-based approaches with LLMs yield promising results under the right conditions. Under the proposed approach, we simulate a "classroom" of 4th, 8th, or 12th grade students by prompting the LLM to role-play students of varying proficiency levels. We use the outcomes of these simulations to fit Item Response Theory (IRT) models, comparing learned difficulty parameters for items to their real-world difficulties, as determined by item-level statistics furnished by the National Assessment of Educational Progress (NAEP). We observe correlations as high as 0.75, 0.76, and 0.82 for grades 4, 8, and 12, respectively. In our simulations, we experiment with different "classroom sizes," showing tradeoffs between computation size and accuracy. We find that role-plays with named students improves predictions (compared to student ids), and stratifying names across gender and race further improves predictions. Our results show that LLMs with relatively weaker mathematical abilities (Gemma) actually yield better real-world difficulty predictions than mathematically stronger models (Llama and Qwen), further underscoring the suitability of open-source models for the task.
>
---
#### [new 068] Skill-Aware Data Selection and Fine-Tuning for Data-Efficient Reasoning Distillation
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在提升小模型在复杂推理任务中的表现。针对数据效率不足的问题，提出基于技能的数据选择和微调方法，有效提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.10109v1](https://arxiv.org/pdf/2601.10109v1)**

> **作者:** Lechen Zhang; Yunxiang Zhang; Wei Hu; Lu Wang
>
> **摘要:** Large reasoning models such as DeepSeek-R1 and their distilled variants achieve strong performance on complex reasoning tasks. Yet, distilling these models often demands large-scale data for supervised fine-tuning (SFT), motivating the pursuit of data-efficient training methods. To address this, we propose a skill-centric distillation framework that efficiently transfers reasoning ability to weaker models with two components: (1) Skill-based data selection, which prioritizes examples targeting the student model's weaker skills, and (2) Skill-aware fine-tuning, which encourages explicit skill decomposition during problem solving. With only 1,000 training examples selected from a 100K teacher-generated corpus, our method surpasses random SFT baselines by +1.6% on Qwen3-4B and +1.4% on Qwen3-8B across five mathematical reasoning benchmarks. Further analysis confirms that these gains concentrate on skills emphasized during training, highlighting the effectiveness of skill-centric training for efficient reasoning distillation.
>
---
#### [new 069] Patient-Similarity Cohort Reasoning in Clinical Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文属于临床文本到SQL的任务，旨在解决真实世界电子健康记录的复杂查询问题。提出CLINSQL基准，评估模型在多表连接、时间窗口和患者相似性方面的表现。**

- **链接: [https://arxiv.org/pdf/2601.09876v1](https://arxiv.org/pdf/2601.09876v1)**

> **作者:** Yifei Shen; Yilun Zhao; Justice Ou; Tinglin Huang; Arman Cohan
>
> **备注:** Accepted by EACL 2026
>
> **摘要:** Real-world clinical text-to-SQL requires reasoning over heterogeneous EHR tables, temporal windows, and patient-similarity cohorts to produce executable queries. We introduce CLINSQL, a benchmark of 633 expert-annotated tasks on MIMIC-IV v3.1 that demands multi-table joins, clinically meaningful filters, and executable SQL. Solving CLINSQL entails navigating schema metadata and clinical coding systems, handling long contexts, and composing multi-step queries beyond traditional text-to-SQL. We evaluate 22 proprietary and open-source models under Chain-of-Thought self-refinement and use rubric-based SQL analysis with execution checks that prioritize critical clinical requirements. Despite recent advances, performance remains far from clinical reliability: on the test set, GPT-5-mini attains 74.7% execution score, DeepSeek-R1 leads open-source at 69.2% and Gemini-2.5-Pro drops from 85.5% on Easy to 67.2% on Hard. Progress on CLINSQL marks tangible advances toward clinically reliable text-to-SQL for real-world EHR analytics.
>
---
#### [new 070] HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在解决时间约束下的翻译长度控制问题。通过引入Sand-Glass基准和HOMURA框架，实现语义与时间的平衡优化。**

- **链接: [https://arxiv.org/pdf/2601.10187v1](https://arxiv.org/pdf/2601.10187v1)**

> **作者:** Ziang Cui; Mengran Yu; Tianjiao Li; Chenyu Shi; Yingxuan Shi; Lusheng Zhang; Hongwei Lin
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.
>
---
#### [new 071] GeoSteer: Faithful Chain-of-Thought Steering via Latent Manifold Gradients
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM中间推理不一致的问题。提出GeoSteer框架，通过潜在流形梯度优化推理过程，提升推理质量。**

- **链接: [https://arxiv.org/pdf/2601.10229v1](https://arxiv.org/pdf/2601.10229v1)**

> **作者:** Kentaro Kazama; Daiki Shirafuji; Tatsuhiko Saito
>
> **备注:** The Third workshop of NeusymBridge @AAAI 2026 (Bridging Neurons and Symbols for NLP and Knowledge Graph Reasoning)
>
> **摘要:** Recent advances in Large Language Models (LLMs) have improved multi-step reasoning. Most approaches rely on Chain-of-Thought (CoT) rationales. Previous studies have shown that LLMs often generate logically inconsistent reasoning steps even when their final answers are correct. These inconsistencies reduce the reliability of step-level reasoning. We propose GeoSteer, a manifold-based framework that improves the quality of intermediate reasoning. The method consists of: (1) constructing a CoT dataset with segment-level scores, (2) training a Variational Autoencoder (VAE) model and a quality estimation model to learn a low-dimensional manifold of high-quality CoT trajectories, and (3) steering hidden states of target LLMs toward higher-quality regions in the latent space. This update in a latent space behaves like a natural-gradient adjustment in the original hidden-state space. It ensures geometrically coherent steering. We evaluate GeoSteer on the GSM8k dataset using the Qwen3 series. We measure via answer accuracy and overall reasoning performance. GeoSteer improved the exact match accuracy by up to 2.6 points. It also enhanced the pairwise win rate by 5.3 points. These results indicate that GeoSteer provides an effective and controllable mechanism for improving the quality of intermediate reasoning in LLMs.
>
---
#### [new 072] Boundary-Aware NL2SQL: Integrating Reliability through Hybrid Reward and Data Synthesis
- **分类: cs.CL**

- **简介: 该论文提出BAR-SQL，解决自然语言到SQL的生成任务，通过融合可靠性与边界感知，提升生成准确性和边界情况处理能力。**

- **链接: [https://arxiv.org/pdf/2601.10318v1](https://arxiv.org/pdf/2601.10318v1)**

> **作者:** Songsong Tian; Kongsheng Zhuo; Zhendong Wang; Rong Shen; Shengtao Zhang; Yong Wu
>
> **摘要:** In this paper, we present BAR-SQL (Boundary-Aware Reliable NL2SQL), a unified training framework that embeds reliability and boundary awareness directly into the generation process. We introduce a Seed Mutation data synthesis paradigm that constructs a representative enterprise corpus, explicitly encompassing multi-step analytical queries alongside boundary cases including ambiguity and schema limitations. To ensure interpretability, we employ Knowledge-Grounded Reasoning Synthesis, which produces Chain-of-Thought traces explicitly anchored in schema metadata and business rules. The model is trained through a two-stage process: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning via Group Relative Policy Optimization. We design a Task-Conditioned Hybrid Reward mechanism that simultaneously optimizes SQL execution accuracy-leveraging Abstract Syntax Tree analysis and dense result matching-and semantic precision in abstention responses. To evaluate reliability alongside generation accuracy, we construct and release Ent-SQL-Bench, which jointly assesse SQL precision and boundary-aware abstention across ambiguous and unanswerable queries. Experimental results on this benchmark demonstrate that BAR-SQL achieves 91.48% average accuracy, outperforming leading proprietary models, including Claude 4.5 Sonnet and GPT-5, in both SQL generation quality and boundary-aware abstention capability. The source code and benchmark are available anonymously at: https://github.com/TianSongS/BAR-SQL.
>
---
#### [new 073] Evaluating Novelty in AI-Generated Research Plans Using Multi-Workflow LLM Pipelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成研究计划的创新性评估任务，旨在解决AI创作原创性不足的问题。通过对比多种多步骤LLM工作流，验证其生成研究计划的新颖性和可行性。**

- **链接: [https://arxiv.org/pdf/2601.09714v1](https://arxiv.org/pdf/2601.09714v1)**

> **作者:** Devesh Saraogi; Rohit Singhee; Dhruv Kumar
>
> **备注:** Under Review
>
> **摘要:** The integration of Large Language Models (LLMs) into the scientific ecosystem raises fundamental questions about the creativity and originality of AI-generated research. Recent work has identified ``smart plagiarism'' as a concern in single-step prompting approaches, where models reproduce existing ideas with terminological shifts. This paper investigates whether agentic workflows -- multi-step systems employing iterative reasoning, evolutionary search, and recursive decomposition -- can generate more novel and feasible research plans. We benchmark five reasoning architectures: Reflection-based iterative refinement, Sakana AI v2 evolutionary algorithms, Google Co-Scientist multi-agent framework, GPT Deep Research (GPT-5.1) recursive decomposition, and Gemini~3 Pro multimodal long-context pipeline. Using evaluations from thirty proposals each on novelty, feasibility, and impact, we find that decomposition-based and long-context workflows achieve mean novelty of 4.17/5, while reflection-based approaches score significantly lower (2.33/5). Results reveal varied performance across research domains, with high-performing workflows maintaining feasibility without sacrificing creativity. These findings support the view that carefully designed multi-stage agentic workflows can advance AI-assisted research ideation.
>
---
#### [new 074] ADMEDTAGGER: an annotation framework for distillation of expert knowledge for the Polish medical language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本分类任务，解决标注资源不足的问题。利用大模型进行医学文本标注，并训练小型高效分类器。**

- **链接: [https://arxiv.org/pdf/2601.09722v1](https://arxiv.org/pdf/2601.09722v1)**

> **作者:** Franciszek Górski; Andrzej Czyżewski
>
> **摘要:** In this work, we present an annotation framework that demonstrates how a multilingual LLM pretrained on a large corpus can be used as a teacher model to distill the expert knowledge needed for tagging medical texts in Polish. This work is part of a larger project called ADMEDVOICE, within which we collected an extensive corpus of medical texts representing five clinical categories - Radiology, Oncology, Cardiology, Hypertension, and Pathology. Using this data, we had to develop a multi-class classifier, but the fundamental problem turned out to be the lack of resources for annotating an adequate number of texts. Therefore, in our solution, we used the multilingual Llama3.1 model to annotate an extensive corpus of medical texts in Polish. Using our limited annotation resources, we verified only a portion of these labels, creating a test set from them. The data annotated in this way were then used for training and validation of 3 different types of classifiers based on the BERT architecture - the distilled DistilBERT model, BioBERT fine-tuned on medical data, and HerBERT fine-tuned on the Polish language corpus. Among the models we trained, the DistilBERT model achieved the best results, reaching an F1 score > 0.80 for each clinical category and an F1 score > 0.93 for 3 of them. In this way, we obtained a series of highly effective classifiers that represent an alternative to large language models, due to their nearly 500 times smaller size, 300 times lower GPU VRAM consumption, and several hundred times faster inference.
>
---
#### [new 075] Training-Trajectory-Aware Token Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，解决持续蒸馏效果不佳的问题。通过分析训练轨迹，提出T3S方法优化token选择，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.10348v1](https://arxiv.org/pdf/2601.10348v1)**

> **作者:** Zhanming Shen; Jiaqi Hu; Zeyu Qin; Hao Chen; Wentao Ye; Zenan Huang; Yihong Zhuang; Guoshan Lu; Junlin Zhou; Junbo Zhao
>
> **摘要:** Efficient distillation is a key pathway for converting expensive reasoning capability into deployable efficiency, yet in the frontier regime where the student already has strong reasoning ability, naive continual distillation often yields limited gains or even degradation. We observe a characteristic training phenomenon: even as loss decreases monotonically, all performance metrics can drop sharply at almost the same bottleneck, before gradually recovering. We further uncover a token-level mechanism: confidence bifurcates into steadily increasing Imitation-Anchor Tokens that quickly anchor optimization and other yet-to-learn tokens whose confidence is suppressed until after the bottleneck. And the characteristic that these two types of tokens cannot coexist is the root cause of the failure in continual distillation. To this end, we propose Training-Trajectory-Aware Token Selection (T3S) to reconstruct the training objective at the token level, clearing the optimization path for yet-to-learn tokens. T3 yields consistent gains in both AR and dLLM settings: with only hundreds of examples, Qwen3-8B surpasses DeepSeek-R1 on competitive reasoning benchmarks, Qwen3-32B approaches Qwen3-235B, and T3-trained LLaDA-2.0-Mini exceeds its AR baseline, achieving state-of-the-art performance among all of 16B-scale no-think models.
>
---
#### [new 076] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中预归一化效率低和稳定性差的问题。提出BHyT方法，提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.09719v1](https://arxiv.org/pdf/2601.09719v1)**

> **作者:** Hoyoon Byun; Youngjun Choi; Taero Kim; Sungrae Park; Kyungwoo Song
>
> **摘要:** Pre-Layer Normalization (Pre-LN) is the de facto choice for large language models (LLMs) and is crucial for stable pretraining and effective transfer learning. However, Pre-LN is inefficient due to repeated statistical calculations and suffers from the curse of depth. As layers grow, the magnitude and variance of the hidden state escalate, destabilizing training. Efficiency-oriented normalization-free methods such as Dynamic Tanh (DyT) improve speed but remain fragile at depth. To jointly address stability and efficiency, we propose Bounded Hyperbolic Tanh (BHyT), a drop-in replacement for Pre-LN. BHyT couples a tanh nonlinearity with explicit, data-driven input bounding to keep activations within a non-saturating range. It prevents depth-wise growth in activation magnitude and variance and comes with a theoretical stability guarantee. For efficiency, BHyT computes exact statistics once per block and replaces a second normalization with a lightweight variance approximation, enhancing efficiency. Empirically, BHyT demonstrates improved stability and efficiency during pretraining, achieving an average of 15.8% faster training and an average of 4.2% higher token generation throughput compared to RMSNorm., while matching or surpassing its inference performance and robustness across language understanding and reasoning benchmarks. Our code is available at: https://anonymous.4open.science/r/BHyT
>
---
#### [new 077] MatchTIR: Fine-Grained Supervision for Tool-Integrated Reasoning via Bipartite Matching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于工具集成推理任务，旨在解决传统方法在长序列任务中奖励分配粗略的问题。通过双部分匹配和双层优势估计，实现更精确的步骤奖励分配。**

- **链接: [https://arxiv.org/pdf/2601.10712v1](https://arxiv.org/pdf/2601.10712v1)**

> **作者:** Changle Qu; Sunhao Dai; Hengyi Cai; Jun Xu; Shuaiqiang Wang; Dawei Yin
>
> **摘要:** Tool-Integrated Reasoning (TIR) empowers large language models (LLMs) to tackle complex tasks by interleaving reasoning steps with external tool interactions. However, existing reinforcement learning methods typically rely on outcome- or trajectory-level rewards, assigning uniform advantages to all steps within a trajectory. This coarse-grained credit assignment fails to distinguish effective tool calls from redundant or erroneous ones, particularly in long-horizon multi-turn scenarios. To address this, we propose MatchTIR, a framework that introduces fine-grained supervision via bipartite matching-based turn-level reward assignment and dual-level advantage estimation. Specifically, we formulate credit assignment as a bipartite matching problem between predicted and ground-truth traces, utilizing two assignment strategies to derive dense turn-level rewards. Furthermore, to balance local step precision with global task success, we introduce a dual-level advantage estimation scheme that integrates turn-level and trajectory-level signals, assigning distinct advantage values to individual interaction turns. Extensive experiments on three benchmarks demonstrate the superiority of MatchTIR. Notably, our 4B model surpasses the majority of 8B competitors, particularly in long-horizon and multi-turn tasks. Our codes are available at https://github.com/quchangle1/MatchTIR.
>
---
#### [new 078] LLM-Driven Preference Data Synthesis for Proactive Prediction of the Next User Utterance in Human-Machine Dialogue
- **分类: cs.CL**

- **简介: 该论文属于人机对话任务，旨在解决用户下一话语的主动预测问题。针对现有方法在意图推理和偏好建模上的不足，提出ProUtt方法，通过构建意图树并生成偏好与非偏好推理过程，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2601.09713v1](https://arxiv.org/pdf/2601.09713v1)**

> **作者:** Jinqiang Wang; Huansheng Ning; Jianguo Ding; Tao Zhu; Liming Chen; Chris Nugent
>
> **备注:** 19 pages
>
> **摘要:** Proactively predicting a users next utterance in human-machine dialogue can streamline interaction and improve user experience. Existing commercial API-based solutions are subject to privacy concerns while deploying general-purpose LLMs locally remains computationally expensive. As such, training a compact, task-specific LLM provides a practical alternative. Although user simulator methods can predict a user's next utterance, they mainly imitate their speaking style rather than advancing the dialogue. Preference data synthesis has been investigated to generate data for proactive next utterance prediction and help align LLMs with user preferences. Yet existing methods lack the ability to explicitly model the intent reasoning that leads to the user's next utterance and to define and synthesize preference and non-preference reasoning processes for predicting the user's next utterance.To address these challenges, we propose ProUtt, an LLM-driven preference data synthesis method for proactive next utterance prediction. ProUtt converts dialogue history into an intent tree and explicitly models intent reasoning trajectories by predicting the next plausible path from both exploitation and exploration perspectives. It then constructs preference and non-preference reasoning processes by perturbing or revising intent tree paths at different future turns. Extensive evaluations using LLM-as-a-judge and human judgments demonstrate that ProUtt consistently outperforms existing data synthesis methods, user simulators, and commercial LLM APIs across four benchmark datasets. We release both the code and the synthesized datasets to facilitate future research.
>
---
#### [new 079] ROMA: Real-time Omni-Multimodal Assistant with Interactive Streaming Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ROMA，解决实时多模态流媒体理解问题，通过统一处理音频视频实现主动与被动交互，提升多模态任务性能。**

- **链接: [https://arxiv.org/pdf/2601.10323v1](https://arxiv.org/pdf/2601.10323v1)**

> **作者:** Xueyun Tian; Wei Li; Bingbing Xu; Heng Dong; Yuanzhuo Wang; Huawei Shen
>
> **备注:** Our project page is available at https://eureka-maggie.github.io/ROMA_show
>
> **摘要:** Recent Omni-multimodal Large Language Models show promise in unified audio, vision, and text modeling. However, streaming audio-video understanding remains challenging, as existing approaches suffer from disjointed capabilities: they typically exhibit incomplete modality support or lack autonomous proactive monitoring. To address this, we present ROMA, a real-time omni-multimodal assistant for unified reactive and proactive interaction. ROMA processes continuous inputs as synchronized multimodal units, aligning dense audio with discrete video frames to handle granularity mismatches. For online decision-making, we introduce a lightweight speak head that decouples response initiation from generation to ensure precise triggering without task conflict. We train ROMA with a curated streaming dataset and a two-stage curriculum that progressively optimizes for streaming format adaptation and proactive responsiveness. To standardize the fragmented evaluation landscape, we reorganize diverse benchmarks into a unified suite covering both proactive (alert, narration) and reactive (QA) settings. Extensive experiments across 12 benchmarks demonstrate ROMA achieves state-of-the-art performance on proactive tasks while competitive in reactive settings, validating its robustness in unified real-time omni-multimodal understanding.
>
---
#### [new 080] Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全对齐任务，旨在解决LLM易受攻击的问题。通过自博弈与反思经验回放，使模型自主生成攻击并强化防御，提升安全性。**

- **链接: [https://arxiv.org/pdf/2601.10589v1](https://arxiv.org/pdf/2601.10589v1)**

> **作者:** Hao Wang; Yanting Wang; Hao Li; Rui Li; Lei Sha
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.
>
---
#### [new 081] MATRIX AS PLAN: Structured Logical Reasoning with Feedback-Driven Replanning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MatrixCoT框架，解决LLM在符号逻辑推理中的稳定性与可解释性问题，通过矩阵规划和反馈重规划提升推理能力。**

- **链接: [https://arxiv.org/pdf/2601.10101v1](https://arxiv.org/pdf/2601.10101v1)**

> **作者:** Ke Chen; Jiandian Zeng; Zihao Peng; Guo Li; Guangxue Zhang; Tian Wang
>
> **备注:** 12 pages, 5 figures, 2 tables. Accepted at The Web Conference (WWW) 2026
>
> **摘要:** As knowledge and semantics on the web grow increasingly complex, enhancing Large Language Models (LLMs) comprehension and reasoning capabilities has become particularly important. Chain-of-Thought (CoT) prompting has been shown to enhance the reasoning capabilities of LLMs. However, it still falls short on logical reasoning tasks that rely on symbolic expressions and strict deductive rules. Neuro-symbolic methods address this gap by enforcing formal correctness through external solvers. Yet these solvers are highly format-sensitive, and small instabilities in model outputs can lead to frequent processing failures. LLM-driven approaches avoid parsing brittleness, but they lack structured representations and process-level error-correction mechanisms. To further enhance the logical reasoning capabilities of LLMs, we propose MatrixCoT, a structured CoT framework with a matrix-based plan. Specifically, we normalize and type natural language expressions, attach explicit citation fields, and introduce a matrix-based planning method to preserve global relations among steps. The plan becomes a verifiable artifact, making execution more stable. For verification, we also add a feedback-driven replanning mechanism. Under semantic-equivalence constraints, it identifies omissions and defects, rewrites and compresses the dependency matrix, and produces a more trustworthy final answer. Experiments on five logical-reasoning benchmarks and five LLMs show that, without relying on external solvers, MatrixCoT enhances both robustness and interpretability when tackling complex symbolic reasoning tasks, while maintaining competitive performance.
>
---
#### [new 082] SuS: Strategy-aware Surprise for Intrinsic Exploration
- **分类: cs.LG; cs.AI; cs.CL; cs.GT**

- **简介: 该论文提出SuS框架，用于强化学习中的内在探索，解决策略一致性与意外结果的平衡问题。通过结合策略稳定性与策略惊喜，提升任务准确性和多样性。**

- **链接: [https://arxiv.org/pdf/2601.10349v1](https://arxiv.org/pdf/2601.10349v1)**

> **作者:** Mark Kashirskiy; Ilya Makarov
>
> **备注:** 8 pages, 7 figures, 3 tables. Code available at https://github.com/mariklolik/sus
>
> **摘要:** We propose Strategy-aware Surprise (SuS), a novel intrinsic motivation framework that uses pre-post prediction mismatch as a novelty signal for exploration in reinforcement learning. Unlike traditional curiosity-driven methods that rely solely on state prediction error, SuS introduces two complementary components: Strategy Stability (SS) and Strategy Surprise (SuS). SS measures consistency in behavioral strategy across temporal steps, while SuS captures unexpected outcomes relative to the agent's current strategy representation. Our combined reward formulation leverages both signals through learned weighting coefficients. We evaluate SuS on mathematical reasoning tasks using large language models, demonstrating significant improvements in both accuracy and solution diversity. Ablation studies confirm that removing either component results in at least 10% performance degradation, validating the synergistic nature of our approach. SuS achieves 17.4% improvement in Pass@1 and 26.4% improvement in Pass@5 compared to baseline methods, while maintaining higher strategy diversity throughout training.
>
---
#### [new 083] Social Determinants of Health Prediction for ICD-9 Code with Reasoning Models
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于医疗文本分类任务，旨在通过推理模型和大语言模型预测医院就诊的SDoH ICD-9编码，解决结构化数据中社会决定因素缺失的问题。**

- **链接: [https://arxiv.org/pdf/2601.09709v1](https://arxiv.org/pdf/2601.09709v1)**

> **作者:** Sharim Khan; Paul Landes; Adam Cross; Jimeng Sun
>
> **备注:** Published as part of Machine Learning for Health (ML4H) 2025 Findings Track
>
> **摘要:** Social Determinants of Health correlate with patient outcomes but are rarely captured in structured data. Recent attention has been given to automatically extracting these markers from clinical text to supplement diagnostic systems with knowledge of patients' social circumstances. Large language models demonstrate strong performance in identifying Social Determinants of Health labels from sentences. However, prediction in large admissions or longitudinal notes is challenging given long distance dependencies. In this paper, we explore hospital admission multi-label Social Determinants of Health ICD-9 code classification on the MIMIC-III dataset using reasoning models and traditional large language models. We exploit existing ICD-9 codes for prediction on admissions, which achieved an 89% F1. Our contributions include our findings, missing SDoH codes in 139 admissions, and code to reproduce the results.
>
---
#### [new 084] A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于模型安全评估任务，旨在分析多个前沿大模型的安全性。通过统一协议评估语言、视觉语言和图像生成场景，揭示安全性能的多样性与脆弱性。**

- **链接: [https://arxiv.org/pdf/2601.10527v1](https://arxiv.org/pdf/2601.10527v1)**

> **作者:** Xingjun Ma; Yixu Wang; Hengyuan Xu; Yutao Wu; Yifan Ding; Yunhan Zhao; Zilong Wang; Jiabin Hua; Ming Wen; Jianan Liu; Ranjie Duan; Yifeng Gao; Yingshui Tan; Yunhao Chen; Hui Xue; Xin Wang; Wei Cheng; Jingjing Chen; Zuxuan Wu; Bo Li; Yu-Gang Jiang
>
> **备注:** 42 pages, 24 figures
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has produced substantial gains in reasoning, perception, and generative capability across language and vision. However, whether these advances yield commensurate improvements in safety remains unclear, in part due to fragmented evaluation practices limited to single modalities or threat models. In this report, we present an integrated safety evaluation of 7 frontier models: GPT-5.2, Gemini 3 Pro, Qwen3-VL, Doubao 1.8, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5. We evaluate each model across language, vision-language, and image generation settings using a unified protocol that integrates benchmark evaluation, adversarial evaluation, multilingual evaluation, and compliance evaluation. Aggregating our evaluations into safety leaderboards and model safety profiles across multiple evaluation modes reveals a sharply heterogeneous safety landscape. While GPT-5.2 demonstrates consistently strong and balanced safety performance across evaluations, other models exhibit pronounced trade-offs among benchmark safety, adversarial alignment, multilingual generalization, and regulatory compliance. Both language and vision-language modalities show significant vulnerability under adversarial evaluation, with all models degrading substantially despite strong results on standard benchmarks. Text-to-image models achieve relatively stronger alignment in regulated visual risk categories, yet remain brittle under adversarial or semantically ambiguous prompts. Overall, these results show that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation scheme, underscoring the need for standardized safety evaluations to accurately assess real-world risk and guide responsible model development and deployment.
>
---
#### [new 085] Sparse-RL: Breaking the Memory Wall in LLM Reinforcement Learning via Stable Sparse Rollouts
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决LLM训练中长序列滚动生成的内存瓶颈问题。通过引入稀疏采样和重加权方法，实现高效稳定的RL训练。**

- **链接: [https://arxiv.org/pdf/2601.10079v1](https://arxiv.org/pdf/2601.10079v1)**

> **作者:** Sijia Luo; Xiaokang Zhang; Yuxuan Hu; Bohan Zhang; Ke Wang; Jinbo Su; Mengshu Sun; Lei Liang; Jing Zhang
>
> **摘要:** Reinforcement Learning (RL) has become essential for eliciting complex reasoning capabilities in Large Language Models (LLMs). However, the substantial memory overhead of storing Key-Value (KV) caches during long-horizon rollouts acts as a critical bottleneck, often prohibiting efficient training on limited hardware. While existing KV compression techniques offer a remedy for inference, directly applying them to RL training induces a severe policy mismatch, leading to catastrophic performance collapse. To address this, we introduce Sparse-RL empowers stable RL training under sparse rollouts. We show that instability arises from a fundamental policy mismatch among the dense old policy, the sparse sampler policy, and the learner policy. To mitigate this issue, Sparse-RL incorporates Sparsity-Aware Rejection Sampling and Importance-based Reweighting to correct the off-policy bias introduced by compression-induced information loss. Experimental results show that Sparse-RL reduces rollout overhead compared to dense baselines while preserving the performance. Furthermore, Sparse-RL inherently implements sparsity-aware training, significantly enhancing model robustness during sparse inference deployment.
>
---
#### [new 086] Learning Latency-Aware Orchestration for Parallel Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统优化任务，旨在解决并行执行下的高延迟问题。通过引入显式延迟监督的框架LAMaS，优化关键路径以降低延迟，提升系统效率。**

- **链接: [https://arxiv.org/pdf/2601.10560v1](https://arxiv.org/pdf/2601.10560v1)**

> **作者:** Xi Shi; Mengxin Zheng; Qian Lou
>
> **备注:** Preprint
>
> **摘要:** Multi-agent systems (MAS) enable complex reasoning by coordinating multiple agents, but often incur high inference latency due to multi-step execution and repeated model invocations, severely limiting their scalability and usability in time-sensitive scenarios. Most existing approaches primarily optimize task performance and inference cost, and explicitly or implicitly assume sequential execution, making them less optimal for controlling latency under parallel execution. In this work, we investigate learning-based orchestration of multi-agent systems with explicit latency supervision under parallel execution. We propose Latency-Aware Multi-agent System (LAMaS), a latency-aware multi-agent orchestration framework that enables parallel execution and explicitly optimizes the critical execution path, allowing the controller to construct execution topology graphs with lower latency under parallel execution. Our experiments show that our approach reduces critical path length by 38-46% compared to the state-of-the-art baseline for multi-agent architecture search across multiple benchmarks, while maintaining or even improving task performance. These results highlight the importance of explicitly optimizing latency under parallel execution when designing efficient multi-agent systems. The code is available at https://github.com/xishi404/LAMaS
>
---
#### [new 087] Defending Large Language Models Against Jailbreak Attacks via In-Decoding Safety-Awareness Probing
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在防御LLM的越狱攻击。通过分析解码过程中的隐含安全信号，提出一种早期检测方法，在保持模型性能的同时提升安全性。**

- **链接: [https://arxiv.org/pdf/2601.10543v1](https://arxiv.org/pdf/2601.10543v1)**

> **作者:** Yinzhi Zhao; Ming Wang; Shi Feng; Xiaocui Yang; Daling Wang; Yifei Zhang
>
> **摘要:** Large language models (LLMs) have achieved impressive performance across natural language tasks and are increasingly deployed in real-world applications. Despite extensive safety alignment efforts, recent studies show that such alignment is often shallow and remains vulnerable to jailbreak attacks. Existing defense mechanisms, including decoding-based constraints and post-hoc content detectors, struggle against sophisticated jailbreaks, often intervening robust detection or excessively degrading model utility. In this work, we examine the decoding process of LLMs and make a key observation: even when successfully jailbroken, models internally exhibit latent safety-related signals during generation. However, these signals are overridden by the model's drive for fluent continuation, preventing timely self-correction or refusal. Building on this observation, we propose a simple yet effective approach that explicitly surfaces and leverages these latent safety signals for early detection of unsafe content during decoding. Experiments across diverse jailbreak attacks demonstrate that our approach significantly enhances safety, while maintaining low over-refusal rates on benign inputs and preserving response quality. Our results suggest that activating intrinsic safety-awareness during decoding offers a promising and complementary direction for defending against jailbreak attacks. Code is available at: https://github.com/zyz13590/SafeProbing.
>
---
#### [new 088] PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLMs推理能力不足的问题。提出PRL方法，通过过程奖励提升模型推理性能并拓宽推理边界。**

- **链接: [https://arxiv.org/pdf/2601.10201v1](https://arxiv.org/pdf/2601.10201v1)**

> **作者:** Jiarui Yao; Ruida Wang; Tong Zhang
>
> **摘要:** Improving the reasoning abilities of Large Language Models (LLMs) has been a continuous topic recently. But most relevant works are based on outcome rewards at the trajectory level, missing fine-grained supervision during the reasoning process. Other existing training frameworks that try to combine process signals together to optimize LLMs also rely heavily on tedious additional steps like MCTS, training a separate reward model, etc., doing harm to the training efficiency. Moreover, the intuition behind the process signals design lacks rigorous theoretical support, leaving the understanding of the optimization mechanism opaque. In this paper, we propose Process Reward Learning (PRL), which decomposes the entropy regularized reinforcement learning objective into intermediate steps, with rigorous process rewards that could be assigned to models accordingly. Starting from theoretical motivation, we derive the formulation of PRL that is essentially equivalent to the objective of reward maximization plus a KL-divergence penalty term between the policy model and a reference model. However, PRL could turn the outcome reward into process supervision signals, which helps better guide the exploration during RL optimization. From our experiment results, we demonstrate that PRL not only improves the average performance for LLMs' reasoning ability measured by average @ n, but also broadens the reasoning boundary by improving the pass @ n metric. Extensive experiments show the effectiveness of PRL could be verified and generalized.
>
---
#### [new 089] Antisocial behavior towards large language model users: experimental evidence
- **分类: cs.AI; cs.CL; cs.CY; econ.GN**

- **简介: 该论文属于行为经济学研究，探讨用户使用大语言模型引发的反社会行为。通过实验发现，用户依赖LLM会招致社会惩罚，且真实使用比声称不用更受谴责。**

- **链接: [https://arxiv.org/pdf/2601.09772v1](https://arxiv.org/pdf/2601.09772v1)**

> **作者:** Paweł Niszczota; Cassandra Grützner
>
> **摘要:** The rapid spread of large language models (LLMs) has raised concerns about the social reactions they provoke. Prior research documents negative attitudes toward AI users, but it remains unclear whether such disapproval translates into costly action. We address this question in a two-phase online experiment (N = 491 Phase II participants; Phase I provided targets) where participants could spend part of their own endowment to reduce the earnings of peers who had previously completed a real-effort task with or without LLM support. On average, participants destroyed 36% of the earnings of those who relied exclusively on the model, with punishment increasing monotonically with actual LLM use. Disclosure about LLM use created a credibility gap: self-reported null use was punished more harshly than actual null use, suggesting that declarations of "no use" are treated with suspicion. Conversely, at high levels of use, actual reliance on the model was punished more strongly than self-reported reliance. Taken together, these findings provide the first behavioral evidence that the efficiency gains of LLMs come at the cost of social sanctions.
>
---
#### [new 090] SPRInG: Continual LLM Personalization via Selective Parametric Adaptation and Retrieval-Interpolated Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于持续个性化任务，解决用户兴趣随时间变化导致的模型遗忘问题。通过选择性参数适应和检索生成融合，提升模型适应能力。**

- **链接: [https://arxiv.org/pdf/2601.09974v1](https://arxiv.org/pdf/2601.09974v1)**

> **作者:** Seoyeon Kim; Jaehyung Kim
>
> **备注:** under review, 23 pages
>
> **摘要:** Personalizing Large Language Models typically relies on static retrieval or one-time adaptation, assuming user preferences remain invariant over time. However, real-world interactions are dynamic, where user interests continuously evolve, posing a challenge for models to adapt to preference drift without catastrophic forgetting. Standard continual learning approaches often struggle in this context, as they indiscriminately update on noisy interaction streams, failing to distinguish genuine preference shifts from transient contexts. To address this, we introduce SPRInG, a novel semi-parametric framework designed for effective continual personalization. During training, SPRInG employs drift-driven selective adaptation, which utilizes a likelihood-based scoring function to identify high-novelty interactions. This allows the model to selectively update the user-specific adapter on drift signals while preserving hard-to-learn residuals in a replay buffer. During inference, we apply strict relevance gating and fuse parametric knowledge with retrieved history via logit interpolation. Experiments on the long-form personalized generation benchmark demonstrate that SPRInG outperforms existing baselines, validating its robustness for real-world continual personalization.
>
---
#### [new 091] Self-reflection in Automated Qualitative Coding: Improving Text Annotation through Secondary LLM Critique
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于文本分类任务，旨在解决LLM在定性编码中的高误报问题。通过两阶段流程，先由LLM进行标注，再由另一LLM进行自我反思修正，提升准确性。**

- **链接: [https://arxiv.org/pdf/2601.09905v1](https://arxiv.org/pdf/2601.09905v1)**

> **作者:** Zackary Okun Dunivin; Mobina Noori; Seth Frey; Curtis Atkinson
>
> **摘要:** Large language models (LLMs) allow for sophisticated qualitative coding of large datasets, but zero- and few-shot classifiers can produce an intolerable number of errors, even with careful, validated prompting. We present a simple, generalizable two-stage workflow: an LLM applies a human-designed, LLM-adapted codebook; a secondary LLM critic performs self-reflection on each positive label by re-reading the source text alongside the first model's rationale and issuing a final decision. We evaluate this approach on six qualitative codes over 3,000 high-content emails from Apache Software Foundation project evaluation discussions. Our human-derived audit of 360 positive annotations (60 passages by six codes) found that the first-line LLM had a false-positive rate of 8% to 54%, despite F1 scores of 0.74 and 1.00 in testing. Subsequent recoding of all stage-one annotations via a second self-reflection stage improved F1 by 0.04 to 0.25, bringing two especially poor performing codes up to 0.69 and 0.79 from 0.52 and 0.55 respectively. Our manual evaluation identified two recurrent error classes: misinterpretation (violations of code definitions) and meta-discussion (debate about a project evaluation criterion mistaken for its use as a decision justification). Code-specific critic clauses addressing observed failure modes were especially effective with testing and refinement, replicating the codebook-adaption process for LLM interpretation in stage-one. We explain how favoring recall in first-line LLM annotation combined with secondary critique delivers precision-first, compute-light control. With human guidance and validation, self-reflection slots into existing LLM-assisted annotation pipelines to reduce noise and potentially salvage unusable classifiers.
>
---
#### [new 092] Synthetic Data for Veterinary EHR De-identification: Benefits, Limits, and Safety Trade-offs Under Fixed Compute
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于 veterinary EHR de-identification 任务，旨在评估合成数据在隐私保护中的效果。研究比较了合成数据与真实数据在不同训练策略下的表现，发现合成数据可扩展训练但不能替代真实数据。**

- **链接: [https://arxiv.org/pdf/2601.09756v1](https://arxiv.org/pdf/2601.09756v1)**

> **作者:** David Brundage
>
> **摘要:** Veterinary electronic health records (vEHRs) contain privacy-sensitive identifiers that limit secondary use. While PetEVAL provides a benchmark for veterinary de-identification, the domain remains low-resource. This study evaluates whether large language model (LLM)-generated synthetic narratives improve de-identification safety under distinct training regimes, emphasizing (i) synthetic augmentation and (ii) fixed-budget substitution. We conducted a controlled simulation using a PetEVAL-derived corpus (3,750 holdout/1,249 train). We generated 10,382 synthetic notes using a privacy-preserving "template-only" regime where identifiers were removed prior to LLM prompting. Three transformer backbones (PetBERT, VetBERT, Bio_ClinicalBERT) were trained under varying mixtures. Evaluation prioritized document-level leakage rate (the fraction of documents with at least one missed identifier) as the primary safety outcome. Results show that under fixed-sample substitution, replacing real notes with synthetic ones monotonically increased leakage, indicating synthetic data cannot safely replace real supervision. Under compute-matched training, moderate synthetic mixing matched real-only performance, but high synthetic dominance degraded utility. Conversely, epoch-scaled augmentation improved performance: PetBERT span-overlap F1 increased from 0.831 to 0.850 +/- 0.014, and leakage decreased from 6.32% to 4.02% +/- 0.19%. However, these gains largely reflect increased training exposure rather than intrinsic synthetic data quality. Corpus diagnostics revealed systematic synthetic-real mismatches in note length and label distribution that align with persistent leakage. We conclude that synthetic augmentation is effective for expanding exposure but is complementary, not substitutive, for safety-critical veterinary de-identification.
>
---
#### [new 093] Agent Skills in the Wild: An Empirical Study of Security Vulnerabilities at Scale
- **分类: cs.CR; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于安全分析任务，旨在解决AI代理技能中的安全漏洞问题。通过大规模分析，发现26.1%的技能存在漏洞，并提出检测方法与数据集。**

- **链接: [https://arxiv.org/pdf/2601.10338v1](https://arxiv.org/pdf/2601.10338v1)**

> **作者:** Yi Liu; Weizhe Wang; Ruitao Feng; Yao Zhang; Guangquan Xu; Gelei Deng; Yuekang Li; Leo Zhang
>
> **摘要:** The rise of AI agent frameworks has introduced agent skills, modular packages containing instructions and executable code that dynamically extend agent capabilities. While this architecture enables powerful customization, skills execute with implicit trust and minimal vetting, creating a significant yet uncharacterized attack surface. We conduct the first large-scale empirical security analysis of this emerging ecosystem, collecting 42,447 skills from two major marketplaces and systematically analyzing 31,132 using SkillScan, a multi-stage detection framework integrating static analysis with LLM-based semantic classification. Our findings reveal pervasive security risks: 26.1% of skills contain at least one vulnerability, spanning 14 distinct patterns across four categories: prompt injection, data exfiltration, privilege escalation, and supply chain risks. Data exfiltration (13.3%) and privilege escalation (11.8%) are most prevalent, while 5.2% of skills exhibit high-severity patterns strongly suggesting malicious intent. We find that skills bundling executable scripts are 2.12x more likely to contain vulnerabilities than instruction-only skills (OR=2.12, p<0.001). Our contributions include: (1) a grounded vulnerability taxonomy derived from 8,126 vulnerable skills, (2) a validated detection methodology achieving 86.7% precision and 82.5% recall, and (3) an open dataset and detection toolkit to support future research. These results demonstrate an urgent need for capability-based permission systems and mandatory security vetting before this attack vector is further exploited.
>
---
#### [new 094] The Geometry of Thought: Disclosing the Transformer as a Tropical Polynomial Circuit
- **分类: cs.LG; cs.CL**

- **简介: 该论文将Transformer自注意力机制建模为热带半环中的矩阵乘法，揭示其本质是动态规划过程，解决理解Transformer计算机制的问题。**

- **链接: [https://arxiv.org/pdf/2601.09775v1](https://arxiv.org/pdf/2601.09775v1)**

> **作者:** Faruk Alpay; Bilge Senturk
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We prove that the Transformer self-attention mechanism in the high-confidence regime ($β\to \infty$, where $β$ is an inverse temperature) operates in the tropical semiring (max-plus algebra). In particular, we show that taking the tropical limit of the softmax attention converts it into a tropical matrix product. This reveals that the Transformer's forward pass is effectively executing a dynamic programming recurrence (specifically, a Bellman-Ford path-finding update) on a latent graph defined by token similarities. Our theoretical result provides a new geometric perspective for chain-of-thought reasoning: it emerges from an inherent shortest-path (or longest-path) algorithm being carried out within the network's computation.
>
---
#### [new 095] TopoDIM: One-shot Topology Generation of Diverse Interaction Modes for Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统通信拓扑优化任务，旨在解决传统方法延迟高、计算量大的问题。提出TopoDIM框架，实现一次生成多样化交互模式，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.10120v1](https://arxiv.org/pdf/2601.10120v1)**

> **作者:** Rui Sun; Jie Ding; Chenghua Gong; Tianjun Gu; Yihang Jiang; Juyuan Zhang; Liming Pan; Linyuan Lü
>
> **摘要:** Optimizing communication topology in LLM-based multi-agent system is critical for enabling collective intelligence. Existing methods mainly rely on spatio-temporal interaction paradigms, where the sequential execution of multi-round dialogues incurs high latency and computation. Motivated by the recent insights that evaluation and debate mechanisms can improve problem-solving in multi-agent systems, we propose TopoDIM, a framework for one-shot Topology generation with Diverse Interaction Modes. Designed for decentralized execution to enhance adaptability and privacy, TopoDIM enables agents to autonomously construct heterogeneous communication without iterative coordination, achieving token efficiency and improved task performance. Experiments demonstrate that TopoDIM reduces total token consumption by 46.41% while improving average performance by 1.50% over state-of-the-art methods. Moreover, the framework exhibits strong adaptability in organizing communication among heterogeneous agents. Code is available at: https://anonymous.4open.science/r/TopoDIM-8D35/
>
---
#### [new 096] Multi-Level Embedding Conformer Framework for Bengali Automatic Speech Recognition
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源语言Bengali的ASR问题。通过多粒度语言信息融合的Conformer框架，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2601.09710v1](https://arxiv.org/pdf/2601.09710v1)**

> **作者:** Md. Nazmus Sakib; Golam Mahmud; Md. Maruf Bangabashi; Umme Ara Mahinur Istia; Md. Jahidul Islam; Partha Sarker; Afra Yeamini Prity
>
> **摘要:** Bengali, spoken by over 300 million people, is a morphologically rich and lowresource language, posing challenges for automatic speech recognition (ASR). This research presents an end-to-end framework for Bengali ASR, building on a Conformer-CTC backbone with a multi-level embedding fusion mechanism that incorporates phoneme, syllable, and wordpiece representations. By enriching acoustic features with these linguistic embeddings, the model captures fine-grained phonetic cues and higher-level contextual patterns. The architecture employs early and late Conformer stages, with preprocessing steps including silence trimming, resampling, Log-Mel spectrogram extraction, and SpecAugment augmentation. The experimental results demonstrate the strong potential of the model, achieving a word error rate (WER) of 10.01% and a character error rate (CER) of 5.03%. These results demonstrate the effectiveness of combining multi-granular linguistic information with acoustic modeling, providing a scalable approach for low-resource ASR development.
>
---
#### [new 097] Thinking Long, but Short: Stable Sequential Test-Time Scaling for Large Reasoning Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，解决序列测试时缩放中的精度下降与不稳定问题。提出Min-Seek方法，在不增加计算负担的情况下提升模型稳定性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.09855v1](https://arxiv.org/pdf/2601.09855v1)**

> **作者:** Michael R. Metel; Yufei Cui; Boxing Chen; Prasanna Parthasarathi
>
> **备注:** Findings of EACL 2026
>
> **摘要:** Sequential test-time scaling is a promising training-free method to improve large reasoning model accuracy, but as currently implemented, significant limitations have been observed. Inducing models to think for longer can increase their accuracy, but as the length of reasoning is further extended, it has also been shown to result in accuracy degradation and model instability. This work presents a novel sequential test-time scaling method, Min-Seek, which improves model accuracy significantly over a wide range of induced thoughts, stabilizing the accuracy of sequential scaling, and removing the need for reasoning length fine-tuning. Beyond improving model accuracy over a variety of reasoning tasks, our method is inherently efficient, as only the KV pairs of one additional induced thought are kept in the KV cache during reasoning. With a custom KV cache which stores keys without position embeddings, by dynamically encoding them contiguously before each new generated thought, our method can continue to reason well beyond a model's maximum context length, and under mild conditions has linear computational complexity.
>
---
#### [new 098] TRIM: Hybrid Inference via Targeted Stepwise Routing in Multi-Step Reasoning Tasks
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出TRIM，用于多步推理任务中的混合推理，解决 cascading failures 问题。通过步骤级路由，将关键步骤分配给大模型，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.10245v1](https://arxiv.org/pdf/2601.10245v1)**

> **作者:** Vansh Kapoor; Aman Gupta; Hao Chen; Anurag Beniwal; Jing Huang; Aviral Kumar
>
> **摘要:** Multi-step reasoning tasks like mathematical problem solving are vulnerable to cascading failures, where a single incorrect step leads to complete solution breakdown. Current LLM routing methods assign entire queries to one model, treating all reasoning steps as equal. We propose TRIM (Targeted routing in multi-step reasoning tasks), which routes only critical steps$\unicode{x2013}$those likely to derail the solution$\unicode{x2013}$to larger models while letting smaller models handle routine continuations. Our key insight is that targeted step-level interventions can fundamentally transform inference efficiency by confining expensive calls to precisely those steps where stronger models prevent cascading errors. TRIM operates at the step-level: it uses process reward models to identify erroneous steps and makes routing decisions based on step-level uncertainty and budget constraints. We develop several routing strategies within TRIM, ranging from a simple threshold-based policy to more expressive policies that reason about long-horizon accuracy-cost trade-offs and uncertainty in step-level correctness estimates. On MATH-500, even the simplest thresholding strategy surpasses prior routing methods with 5x higher cost efficiency, while more advanced policies match the strong, expensive model's performance using 80% fewer expensive model tokens. On harder benchmarks such as AIME, TRIM achieves up to 6x higher cost efficiency. All methods generalize effectively across math reasoning tasks, demonstrating that step-level difficulty represents fundamental characteristics of reasoning.
>
---
#### [new 099] Continuous-Depth Transformers with Learned Control Dynamics
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出一种连续深度Transformer架构，通过ODE块实现生成属性的实时控制，解决传统离散层结构灵活性不足的问题。**

- **链接: [https://arxiv.org/pdf/2601.10007v1](https://arxiv.org/pdf/2601.10007v1)**

> **作者:** Peter Jemley
>
> **备注:** 9 pages, 4 figures. Code available at: https://github.com/PeterJemley/Continuous-Depth-Transformers-with-Learned-Control-Dynamics
>
> **摘要:** We present a hybrid transformer architecture that replaces discrete middle layers with a continuous-depth Neural Ordinary Differential Equation (ODE) block, enabling inference-time control over generation attributes via a learned steering signal. Unlike standard transformers that process representations through fixed discrete layers, our approach treats depth as a continuous variable governed by a learned vector field $F_θ(H, τ, u)$, where $u$ is a low-dimensional control signal injected via explicit concatenation. We validate the architecture through four experiments: (1) gradient flow stability with zero exploding/vanishing gradient events, (2) semantic steering achieving 98\%/88\% accuracy for positive/negative sentiment control, (3) continuous interpolation validated by a negligible 0.068\% trajectory divergence between fixed and adaptive solvers, and (4) efficiency benchmarking demonstrating latency parity with standard discrete baselines. Additionally, we show that adaptive ODE solvers reveal geometric structure in the learned dynamics: the control signal partitions the vector field into distinct dynamical regimes with different curvature characteristics. The adjoint method enables $O(1)$ memory training regardless of integration depth. Our results demonstrate that continuous-depth dynamics with learned control signals provide a viable, efficient mechanism for steerable language generation.
>
---
#### [new 100] ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决LLM在代理系统中遭受间接提示注入攻击的问题。工作包括提出ReasAlign模型，通过结构化推理增强安全对齐，有效防御攻击并保持高实用性。**

- **链接: [https://arxiv.org/pdf/2601.10173v1](https://arxiv.org/pdf/2601.10173v1)**

> **作者:** Hao Li; Yankai Yang; G. Edward Suh; Ning Zhang; Chaowei Xiao
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign.
>
---
#### [new 101] Evidence-Augmented Policy Optimization with Reward Co-Evolution for Long-Context Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于长文本推理任务，解决RL在长上下文场景中奖励稀疏的问题。提出EAPO方法，通过增强证据提取和奖励模型协同进化提升推理性能。**

- **链接: [https://arxiv.org/pdf/2601.10306v1](https://arxiv.org/pdf/2601.10306v1)**

> **作者:** Xin Guan; Zijian Li; Shen Huang; Pengjun Xie; Jingren Zhou; Jiuxin Cao
>
> **摘要:** While Reinforcement Learning (RL) has advanced LLM reasoning, applying it to long-context scenarios is hindered by sparsity of outcome rewards. This limitation fails to penalize ungrounded "lucky guesses," leaving the critical process of needle-in-a-haystack evidence retrieval largely unsupervised. To address this, we propose EAPO (Evidence-Augmented Policy Optimization). We first establish the Evidence-Augmented Reasoning paradigm, validating via Tree-Structured Evidence Sampling that precise evidence extraction is the decisive bottleneck for long-context reasoning. Guided by this insight, EAPO introduces a specialized RL algorithm where a reward model computes a Group-Relative Evidence Reward, providing dense process supervision to explicitly improve evidence quality. To sustain accurate supervision throughout training, we further incorporate an Adaptive Reward-Policy Co-Evolution mechanism. This mechanism iteratively refines the reward model using outcome-consistent rollouts, sharpening its discriminative capability to ensure precise process guidance. Comprehensive evaluations across eight benchmarks demonstrate that EAPO significantly enhances long-context reasoning performance compared to SOTA baselines.
>
---
## 更新

#### [replaced 001] Scalable Oversight for Superhuman AI via Recursive Self-Critiquing
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于AI对齐任务，解决AI能力超越人类后难以监督的问题。通过递归自我批评方法，提升AI监督的可扩展性。**

- **链接: [https://arxiv.org/pdf/2502.04675v4](https://arxiv.org/pdf/2502.04675v4)**

> **作者:** Xueru Wen; Jie Lou; Xinyu Lu; Junjie Yang; Yanjiang Liu; Yaojie Lu; Debing Zhang; Xing Yu
>
> **摘要:** As AI capabilities increasingly surpass human proficiency in complex tasks, current alignment techniques, including SFT and RLHF, face fundamental challenges in ensuring reliable oversight. These methods rely on direct human assessment and become impractical when AI outputs exceed human cognitive thresholds. In response to this challenge, we explore two hypotheses: (1) \textit{Critique of critique can be easier than critique itself}, extending the widely-accepted observation that verification is easier than generation to the critique domain, as critique itself is a specialized form of generation; (2) \textit{This difficulty relationship holds recursively}, suggesting that when direct evaluation is infeasible, performing higher-order critiques (e.g., critique of critique of critique) offers a more tractable supervision pathway. We conduct Human-Human, Human-AI, and AI-AI experiments to investigate the potential of recursive self-critiquing for AI supervision. Our results highlight recursive critique as a promising approach for scalable AI oversight.
>
---
#### [replaced 002] WebRollback: Enhancing Web Agents with Explicit Rollback Mechanisms
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于Web导航任务，旨在解决复杂环境下的错误恢复问题。提出显式回滚机制，提升Web代理的灵活性和效率。**

- **链接: [https://arxiv.org/pdf/2504.11788v3](https://arxiv.org/pdf/2504.11788v3)**

> **作者:** Zhisong Zhang; Tianqing Fang; Kaixin Ma; Wenhao Yu; Hongming Zhang; Haitao Mi; Dong Yu
>
> **备注:** EACL 2026
>
> **摘要:** With recent advancements in large language models, web agents have been greatly improved. However, dealing with complex and dynamic web environments requires more advanced planning and search abilities. Previous studies usually adopt a greedy one-way search strategy, which may struggle to recover from erroneous states. In this work, we enhance web agents with an explicit rollback mechanism, enabling the agent to revert back to a previous state in its navigation trajectory. This mechanism gives models the flexibility to directly control the search process, leading to an effective and efficient web navigation method. We conduct experiments on two live web navigation benchmarks with zero-shot and fine-tuning settings. The results demonstrate the effectiveness of our proposed approach.
>
---
#### [replaced 003] Multi-Personality Generation of LLMs at Decoding-time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多个性生成任务，解决LLMs同时体现多种个性的问题。提出MPG框架，在解码阶段通过隐式密度比实现灵活控制，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2511.01891v4](https://arxiv.org/pdf/2511.01891v4)**

> **作者:** Rongxin Chen; Yunfan Li; Yige Yuan; Bingbing Xu; Huawei Shen
>
> **备注:** Accepted by WSDM 2026
>
> **摘要:** Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
>
---
#### [replaced 004] Textual Entailment is not a Better Bias Metric than Token Probability
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于自然语言处理中的社会偏见评估任务，旨在比较NLI与TP作为偏见度量的优劣。研究发现两者表现差异大，NLI不稳定且敏感度不同，结论是NLI不能完全替代TP。**

- **链接: [https://arxiv.org/pdf/2510.07662v2](https://arxiv.org/pdf/2510.07662v2)**

> **作者:** Virginia K. Felkner; Allison Lim; Jonathan May
>
> **备注:** 12 pages, 1 figure. Substantial revisions following October 2025 ARR Cycle. Currently under review in January 2026 ARR Cycle
>
> **摘要:** Measurement of social bias in language models is typically by token probability (TP) metrics, which are broadly applicable but have been criticized for their distance from real-world language model use cases and harms. In this work, we test natural language inference (NLI) as an alternative bias metric. In extensive experiments across seven LM families, we show that NLI and TP bias evaluation behave substantially differently, with very low correlation among different NLI metrics and between NLI and TP metrics. NLI metrics are more brittle and unstable, slightly less sensitive to wording of counterstereotypical sentences, and slightly more sensitive to wording of tested stereotypes than TP approaches. Given this conflicting evidence, we conclude that neither token probability nor natural language inference is a ``better'' bias metric in all cases. We do not find sufficient evidence to justify NLI as a complete replacement for TP metrics in bias evaluation.
>
---
#### [replaced 005] Small Open Models Achieve Near Parity with Large Models in Low Resource Literary Translation at a Fraction of the Cost
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦于低资源文学翻译任务，提出TF2框架，通过小模型与合成数据实现接近大模型的翻译效果，解决成本高、数据少的问题。**

- **链接: [https://arxiv.org/pdf/2509.07829v2](https://arxiv.org/pdf/2509.07829v2)**

> **作者:** Mihai Nadas; Laura Diosan; Andreea Tomescu; Andrei Piscoran
>
> **备注:** 25 pages, 8 figures, includes datasets and models released on Hugging Face
>
> **摘要:** Literary translation has recently gained attention as a distinct and complex task in machine translation research. However, the translation by small open models remains an open problem. We contribute to this ongoing research by introducing TinyFabulist Translation Framework (TF2), a unified framework for dataset creation, fine-tuning, and evaluation in English->Romanian literary translation, centered on the creation and open release of both a compact, fine-tuned language model (TF2-12B) and large-scale synthetic parallel datasets (DS-TF2-EN-RO-3M and DS-TF2-EN-RO-15K). Building on DS-TF1-EN-3M (TF1), the largest collection of synthetic English fables to date, we address the need for rich, high-quality literary datasets in low-resource languages such as Romanian. Our pipeline first generates 15k high-quality Romanian reference translations from the TF1 pool using a high-performing LLM. We then apply a two-stage fine-tuning process to a 12B-parameter open-weight model: (i) instruction tuning to capture genre-specific narrative style, and (ii) adapter compression for efficient deployment. Evaluation combines corpus-level BLEU with a five-dimension LLM-based rubric (accuracy, fluency, coherence, style, and cultural adaptation) to provide a nuanced assessment of translation quality. Results show that our fine-tuned model achieves strong fluency and adequacy, narrowing the gap to top-performing proprietary models under automated and human-anchored evaluation, while being open, accessible, and significantly more cost-effective. Alongside the fine-tuned model and both datasets, we publicly release all scripts and evaluation prompts. TF2 thus provides an end-to-end, reproducible pipeline for research on cost-efficient translation, cross-lingual narrative generation, and the broad adoption of open models for culturally significant literary content in low-resource settings.
>
---
#### [replaced 006] JudgeAgent: Beyond Static Benchmarks for Knowledge-Driven and Dynamic LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型评估任务，旨在解决静态基准评估的局限性。提出JudgeAgent框架，通过动态知识遍历和难度自适应机制实现更全面的模型评估。**

- **链接: [https://arxiv.org/pdf/2509.02097v4](https://arxiv.org/pdf/2509.02097v4)**

> **作者:** Zhichao Shi; Xuhui Jiang; Chengjin Xu; Cangli Yao; Shengjia Ma; Yinghan Shen; Zixuan Li; Jian Guo; Yuanzhuo Wang
>
> **摘要:** Current evaluation methods for large language models (LLMs) primarily rely on static benchmarks, presenting two major challenges: limited knowledge coverage and fixed difficulties that mismatch with the evaluated LLMs. These limitations lead to superficial assessments of LLM knowledge, thereby impeding the targeted model optimizations. To bridge this gap, we propose JudgeAgent, a knowledge-driven and dynamic evaluation framework for LLMs. To address the challenge of limited knowledge coverage, JudgeAgent leverages LLM agents equipped with context graphs to traverse knowledge structures systematically for question generation. Furthermore, to mitigate data contamination and difficulty mismatch, it adopts a difficulty-adaptive and multi-turn interview mechanism. Thereby, JudgeAgent can achieve comprehensive evaluations and facilitate more effective improvement of LLMs. Empirical results demonstrate that JudgeAgent enables more comprehensive evaluations and facilitates effective model iterations, highlighting the potential of this knowledge-driven and dynamic evaluation paradigm. The source code is available on https://github.com/DataArcTech/JudgeAgent.
>
---
#### [replaced 007] Five Years of SciCap: What We Learned and Future Directions for Scientific Figure Captioning
- **分类: cs.CL; cs.AI; cs.CV; cs.HC**

- **简介: 该论文属于科学图表标题生成任务，旨在提升科学论文中图表描述的准确性与质量。研究团队构建了大规模图表示例库，评估生成与作者撰写标题，并提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2512.21789v2](https://arxiv.org/pdf/2512.21789v2)**

> **作者:** Ting-Hao 'Kenneth' Huang; Ryan A. Rossi; Sungchul Kim; Tong Yu; Ting-Yao E. Hsu; Ho Yin; Ng; C. Lee Giles
>
> **备注:** Accepted to the 5th Annual AAAI Workshop on AI to Accelerate Science and Engineering (AI2ASE 2026). SciCap Website: http://scicap.ai/
>
> **摘要:** Between 2021 and 2025, the SciCap project grew from a small seed-funded idea at The Pennsylvania State University (Penn State) into one of the central efforts shaping the scientific figure-captioning landscape. Supported by a Penn State seed grant, Adobe, and the Alfred P. Sloan Foundation, what began as our attempt to test whether domain-specific training, which was successful in text models like SciBERT, could also work for figure captions expanded into a multi-institution collaboration. Over these five years, we curated, released, and continually updated a large collection of figure-caption pairs from arXiv papers, conducted extensive automatic and human evaluations on both generated and author-written captions, navigated the rapid rise of large language models (LLMs), launched annual challenges, and built interactive systems that help scientists write better captions. In this piece, we look back at the first five years of SciCap and summarize the key technical and methodological lessons we learned. We then outline five major unsolved challenges and propose directions for the next phase of research in scientific figure captioning.
>
---
#### [replaced 008] Parallel Test-Time Scaling for Latent Reasoning Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型优化任务，旨在解决 latent reasoning 模型在测试时并行扩展的问题。通过引入采样策略和奖励模型，实现高效推理轨迹选择。**

- **链接: [https://arxiv.org/pdf/2510.07745v3](https://arxiv.org/pdf/2510.07745v3)**

> **作者:** Runyang You; Yongqi Li; Meng Liu; Wenjie Wang; Liqiang Nie; Wenjie Li
>
> **摘要:** Parallel test-time scaling (TTS) is a pivotal approach for enhancing large language models (LLMs), typically by sampling multiple token-based chains-of-thought in parallel and aggregating outcomes through voting or search. Recent advances in latent reasoning, where intermediate reasoning unfolds in continuous vector spaces, offer a more efficient alternative to explicit Chain-of-Thought, yet whether such latent models can similarly benefit from parallel TTS remains open, mainly due to the absence of sampling mechanisms in continuous space, and the lack of probabilistic signals for advanced trajectory aggregation. This work enables parallel TTS for latent reasoning models by addressing the above issues. For sampling, we introduce two uncertainty-inspired stochastic strategies: Monte Carlo Dropout and Additive Gaussian Noise. For aggregation, we design a Latent Reward Model (LatentRM) trained with step-wise contrastive objective to score and guide latent reasoning. Extensive experiments and visualization analyses show that both sampling strategies scale effectively with compute and exhibit distinct exploration dynamics, while LatentRM enables effective trajectory selection. Together, our explorations open a new direction for scalable inference in continuous spaces. Code and checkpoints released at https://github.com/ModalityDance/LatentTTS
>
---
#### [replaced 009] MathArena: Evaluating LLMs on Uncontaminated Math Competitions
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MathArena基准，用于评估大语言模型的数学推理和证明能力，解决数据泄露和缺乏证明评测的问题。**

- **链接: [https://arxiv.org/pdf/2505.23281v3](https://arxiv.org/pdf/2505.23281v3)**

> **作者:** Mislav Balunović; Jasper Dekoninck; Ivo Petrov; Nikola Jovanović; Martin Vechev
>
> **摘要:** The rapid advancement of reasoning capabilities in large language models (LLMs) has led to notable improvements on mathematical benchmarks. However, many of the most commonly used evaluation datasets (e.g., AIME 2024) are widely available online, making it difficult to disentangle genuine reasoning from potential memorization. Furthermore, these benchmarks do not evaluate proof-writing capabilities, which are crucial for many mathematical tasks. To address this, we introduce MathArena, a new benchmark based on the following key insight: recurring math competitions provide a stream of high-quality, challenging problems that can be used for real-time evaluation of LLMs. By evaluating models as soon as new problems are released, we effectively eliminate the risk of contamination. Using this framework, we find strong signs of contamination in AIME 2024. Nonetheless, evaluations on harder competitions, such as CMIMC 2025, demonstrate impressive reasoning capabilities in top-performing models. MathArena is also the first benchmark for proof-writing capabilities. On IMO 2025, top models achieve slightly less than 40%, demonstrating both notable progress and significant room for improvement. So far, we have evaluated over $50$ models across seven competitions, totaling $162$ problems. As an evolving benchmark, MathArena will continue to track the progress of LLMs on newly released competitions, ensuring rigorous and up-to-date evaluation of mathematical reasoning.
>
---
#### [replaced 010] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Disco-RAG，用于知识密集型任务，解决传统RAG无法捕捉文本结构的问题。通过构建话语结构，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2601.04377v3](https://arxiv.org/pdf/2601.04377v3)**

> **作者:** Dongqi Liu; Hang Ding; Qiming Feng; Jian Li; Xurong Xie; Zhucun Xue; Chengjie Wang; Jiangning Zhang; Yabiao Wang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems.
>
---
#### [replaced 011] Relative Scaling Laws for LLMs
- **分类: cs.CL**

- **简介: 该论文研究语言模型的相对扩展规律，解决模型性能差异随规模变化的问题。通过实验分析不同分布下的性能演变，揭示扩展并非普遍平等化。**

- **链接: [https://arxiv.org/pdf/2510.24626v2](https://arxiv.org/pdf/2510.24626v2)**

> **作者:** William Held; David Hall; Percy Liang; Diyi Yang
>
> **摘要:** Scaling laws describe how language models improve with additional data, parameters, and compute. While widely used, they are typically measured on aggregate test sets. Aggregate evaluations yield clean trends but average over heterogeneous subpopulations, obscuring performance disparities. We introduce relative scaling laws, which track how performance gaps between test distributions evolve with scale rather than focusing solely on absolute error. Using 255 decoder-only Transformers trained under matched-compute (IsoFLOP) budgets from $10^{18}$--$10^{20}$ FLOPs on standard pretraining datasets, we find diverse trajectories: academic domains on MMLU converge toward parity; regional English dialects shift depending on population size; and clusters of AI risk behaviours split, with capability- and influence-related risks increasing during pretraining while adversarial risks do not. These results show that although scaling improves overall performance, it is not a universal equalizer. To support further study, we release all model checkpoints from this work to enable practitioners to measure relative alongside traditional scaling laws, in order to better prioritize robustness challenges in light of the bitter lesson.
>
---
#### [replaced 012] LittleBit: Ultra Low-Bit Quantization via Latent Factorization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型部署中的高内存和计算成本问题。通过极低比特量化实现模型压缩，提出LittleBit方法，在0.1比特/权重下显著降低内存并提升性能。**

- **链接: [https://arxiv.org/pdf/2506.13771v4](https://arxiv.org/pdf/2506.13771v4)**

> **作者:** Banseok Lee; Dongkyu Kim; Youngcheon You; Youngmin Kim
>
> **备注:** Accepted to NeurIPS 2025. Banseok Lee and Dongkyu Kim contributed equally
>
> **摘要:** Deploying large language models (LLMs) often faces challenges from substantial memory and computational costs. Quantization offers a solution, yet performance degradation in the sub-1-bit regime remains particularly difficult. This paper introduces LittleBit, a novel method for extreme LLM compression. It targets levels like 0.1 bits per weight (BPW), achieving nearly 31$\times$ memory reduction, e.g., Llama2-13B to under 0.9 GB. LittleBit represents weights in a low-rank form using latent matrix factorization, subsequently binarizing these factors. To counteract information loss from this extreme precision, it integrates a multi-scale compensation mechanism. This includes row, column, and an additional latent dimension that learns per-rank importance. Two key contributions enable effective training: Dual Sign-Value-Independent Decomposition (Dual-SVID) for quantization-aware training (QAT) initialization, and integrated Residual Compensation to mitigate errors. Extensive experiments confirm LittleBit's superiority in sub-1-bit quantization: e.g., its 0.1 BPW performance on Llama2-7B surpasses the leading method's 0.7 BPW. LittleBit establishes a new, viable size-performance trade-off--unlocking a potential 11.6$\times$ speedup over FP16 at the kernel level--and makes powerful LLMs practical for resource-constrained environments. Our code can be found at https://github.com/SamsungLabs/LittleBit.
>
---
#### [replaced 013] COALA: Numerically Stable and Efficient Framework for Context-Aware Low-Rank Approximation
- **分类: cs.LG; cs.CL; math.NA**

- **简介: 该论文属于神经网络压缩与微调任务，解决低秩近似中的数值不稳定问题。提出COALA框架，避免矩阵求逆，提升稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2507.07580v2](https://arxiv.org/pdf/2507.07580v2)**

> **作者:** Uliana Parkina; Maxim Rakhuba
>
> **摘要:** Recent studies suggest that context-aware low-rank approximation is a useful tool for compression and fine-tuning of modern large-scale neural networks. In this type of approximation, a norm is weighted by a matrix of input activations, significantly improving metrics over the unweighted case. Nevertheless, existing methods for neural networks suffer from numerical instabilities due to their reliance on classical formulas involving explicit Gram matrix computation and their subsequent inversion. We demonstrate that this can degrade the approximation quality or cause numerically singular matrices. To address these limitations, we propose a novel inversion-free regularized framework that is based entirely on stable decompositions and overcomes the numerical pitfalls of prior art. Our method can handle possible challenging scenarios: (1) when calibration matrices exceed GPU memory capacity, (2) when input activation matrices are nearly singular, and even (3) when insufficient data prevents unique approximation. For the latter, we prove that our solution converges to a desired approximation and derive explicit error bounds.
>
---
#### [replaced 014] ZK-SenseLM: Verifiable Large-Model Wireless Sensing with Selective Abstention and Zero-Knowledge Attestation
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出ZK-SenseLM，解决无线传感的可信与安全问题，通过大模型编码器和零知识证明实现可验证的传感决策。**

- **链接: [https://arxiv.org/pdf/2510.25677v3](https://arxiv.org/pdf/2510.25677v3)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
>
> **摘要:** ZK-SenseLM is a secure and auditable wireless sensing framework that pairs a large-model encoder for Wi-Fi channel state information (and optionally mmWave radar or RFID) with a policy-grounded decision layer and end-to-end zero-knowledge proofs of inference. The encoder uses masked spectral pretraining with phase-consistency regularization, plus a light cross-modal alignment that ties RF features to compact, human-interpretable policy tokens. To reduce unsafe actions under distribution shift, we add a calibrated selective-abstention head; the chosen risk-coverage operating point is registered and bound into the proof. We implement a four-stage proving pipeline: (C1) feature sanity and commitment, (C2) threshold and version binding, (C3) time-window binding, and (C4) PLONK-style proofs that the quantized network, given the committed window, produced the logged action and confidence. Micro-batched proving amortizes cost across adjacent windows, and a gateway option offloads proofs from low-power devices. The system integrates with differentially private federated learning and on-device personalization without weakening verifiability: model hashes and the registered threshold are part of each public statement. Across activity, presence or intrusion, respiratory proxy, and RF fingerprinting tasks, ZK-SenseLM improves macro-F1 and calibration, yields favorable coverage-risk curves under perturbations, and rejects tamper and replay with compact proofs and fast verification.
>
---
#### [replaced 015] Investigating LLM Capabilities on Long Context Comprehension for Medical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，研究大语言模型在长文本理解中的表现，探讨模型规模、记忆与推理能力，以及RAG技术的应用效果与挑战。**

- **链接: [https://arxiv.org/pdf/2510.18691v2](https://arxiv.org/pdf/2510.18691v2)**

> **作者:** Feras AlMannaa; Talia Tseriotou; Jenny Chim; Maria Liakata
>
> **摘要:** This study is the first to investigate LLM comprehension capabilities over long-context (LC), clinically relevant medical Question Answering (QA) beyond MCQA. Our comprehensive approach considers a range of settings based on content inclusion of varying size and relevance, LLM models of different capabilities and a variety of datasets across task formulations. We reveal insights on model size effects and their limitations, underlying memorization issues and the benefits of reasoning models, while demonstrating the value and challenges of leveraging the full long patient's context. Importantly, we examine the effect of Retrieval Augmented Generation (RAG) on medical LC comprehension, showcasing best settings in single versus multi-document QA datasets. We shed light into some of the evaluation aspects using a multi-faceted approach uncovering common metric challenges. Our quantitative analysis reveals challenging cases where RAG excels while still showing limitations in cases requiring temporal reasoning.
>
---
#### [replaced 016] AWPO: Enhancing Tool-Use of Large Language Models through Adaptive Integration of Reasoning Rewards
- **分类: cs.CL**

- **简介: 该论文属于大语言模型工具使用任务，旨在解决现有方法忽视推理奖励的问题。提出AWPO框架，通过自适应融合推理奖励提升工具使用性能。**

- **链接: [https://arxiv.org/pdf/2512.19126v3](https://arxiv.org/pdf/2512.19126v3)**

> **作者:** Zihan Lin; Xiaohan Wang; Hexiong Yang; Jiajun Chai; Jie Cao; Guojun Yin; Wei Lin; Ran He
>
> **摘要:** While Reinforcement Learning (RL) shows promise in training tool-use Large Language Models (LLMs) using verifiable outcome rewards, existing methods largely overlook the potential of reasoning rewards based on chain-of-thought quality for better tool utilization. Furthermore, naïvely combining reasoning and outcome rewards may yield suboptimal performance or conflict with the primary optimization objective. To address this, we propose Advantage-Weighted Policy Optimization (AWPO), a principled RL framework that adaptively integrates reasoning rewards into advantage estimation to improve tool-use performance. AWPO incorporates variance-aware gating and difficulty-aware weighting to adaptively modulate advantages from reasoning signals based on group-relative statistics, alongside a tailored clipping mechanism for stable optimization. Extensive experiments demonstrate that AWPO achieves state-of-the-art performance across standard tool-use benchmarks, significantly outperforming strong baselines and leading closed-source models in challenging multi-turn scenarios. Notably, with exceptional parameter efficiency, our 4B model surpasses Grok-4 by $16.0\%$ in multi-turn accuracy while preserving generalization capability on the out-of-distribution MMLU-Pro benchmark.
>
---
#### [replaced 017] Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估低资源语言（如卢森堡语）在大语言模型中的表现。研究使用语言水平考试作为评估工具，发现大模型表现优异，小模型较差，且考试成绩可预测其他NLP任务表现。**

- **链接: [https://arxiv.org/pdf/2504.01667v4](https://arxiv.org/pdf/2504.01667v4)**

> **作者:** Cedric Lothritz; Jordi Cabot; Laura Bernardy
>
> **备注:** 24pages, 4 figures, 14 tables
>
> **摘要:** Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks in Luxembourgish.
>
---
#### [replaced 018] BASIL: Bayesian Assessment of Sycophancy in LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI伦理与可信性研究任务，旨在解决LLMs中的谄媚行为问题。通过贝叶斯框架分离谄媚与理性决策，提出度量方法并验证优化策略。**

- **链接: [https://arxiv.org/pdf/2508.16846v3](https://arxiv.org/pdf/2508.16846v3)**

> **作者:** Katherine Atwell; Pedram Heydari; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Sycophancy (overly agreeable or flattering behavior) poses a fundamental challenge for human-AI collaboration, particularly in high-stakes decision-making domains such as health, law, and education. A central difficulty in studying sycophancy in large language models (LLMs) is disentangling sycophantic belief shifts from rational changes in behavior driven by new evidence or user-provided information. Existing approaches either measure descriptive behavior changes or apply normative evaluations that rely on objective ground truth, limiting their applicability to subjective or uncertain tasks. We introduce a Bayesian probabilistic framework, grounded in behavioral economics and rational decision theory, that explicitly separates sycophancy from rational belief updating. Within this framework, we achieve three objectives: (i) a descriptive metric that measures sycophancy while controlling for rational responses to evidence; (ii) a normative metric that quantifies how sycophancy leads models astray from Bayesian-consistent belief updating; and (iii) the ability to apply both metrics in settings without ground-truth labels. Applying our framework across multiple LLMs and three uncertainty-driven tasks, we find robust evidence of sycophantic belief shifts and show that their impact on rationality depends on whether models systematically over- or under-update their beliefs. Finally, we demonstrate that a post-hoc calibration method and two fine-tuning strategies (SFT and DPO) substantially reduce Bayesian inconsistency, with particularly strong improvements under explicit sycophancy prompting.
>
---
#### [replaced 019] TeleMem: Building Long-Term and Multimodal Memory for Agentic AI
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出TeleMem，解决AI长期交互与多模态记忆问题。通过结构化写入和动态提取，提升记忆效率与准确性。属于AI记忆系统任务。**

- **链接: [https://arxiv.org/pdf/2601.06037v3](https://arxiv.org/pdf/2601.06037v3)**

> **作者:** Chunliang Chen; Ming Guan; Xiao Lin; Jiaxu Li; Luxi Lin; Qiyi Wang; Xiangyu Chen; Jixiang Luo; Changzhi Sun; Dell Zhang; Xuelong Li
>
> **摘要:** Large language models (LLMs) excel at many NLP tasks but struggle to sustain long-term interactions due to limited attention over extended dialogue histories. Retrieval-augmented generation (RAG) mitigates this issue but lacks reliable mechanisms for updating or refining stored memories, leading to schema-driven hallucinations, inefficient write operations, and minimal support for multimodal reasoning.To address these challenges, we propose TeleMem, a unified long-term and multimodal memory system that maintains coherent user profiles through narrative dynamic extraction, ensuring that only dialogue-grounded information is preserved. TeleMem further introduces a structured writing pipeline that batches, retrieves, clusters, and consolidates memory entries, substantially improving storage efficiency, reducing token usage, and accelerating memory operations. Additionally, a multimodal memory module combined with ReAct-style reasoning equips the system with a closed-loop observe, think, and act process that enables accurate understanding of complex video content in long-term contexts. Experimental results show that TeleMem surpasses the state-of-the-art Mem0 baseline with 19% higher accuracy, 43% fewer tokens, and a 2.1x speedup on the ZH-4O long-term role-play gaming benchmark.
>
---
#### [replaced 020] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决现有评估框架的模态捷径和偏倚问题。提出OMHBench基准，用于平衡评估多模态多跳推理能力。**

- **链接: [https://arxiv.org/pdf/2508.16198v2](https://arxiv.org/pdf/2508.16198v2)**

> **作者:** Seunghee Kim; Ingyu Bang; Seokgyu Jang; Changhyeon Kim; Sanghwan Bae; Jihun Choi; Richeng Xuan; Taeuk Kim
>
> **摘要:** Multimodal Large Language Models (MLLMs) have increasingly supported omni-modal processing across text, vision, and speech. However, existing evaluation frameworks for such models suffer from critical limitations, including modality shortcuts and biased reasoning paths. To address these challenges, we propose OMHBench, a novel benchmark designed to rigorously evaluate omni-modal multi-hop reasoning. It consists of 6,144 questions with balanced reasoning paths that are jointly grounded across all three modalities. Extensive evaluation of 13 state-of-the-art models reveals that (1) a large performance gap exists between proprietary and open-source MLLMs and (2) even proprietary models exhibit high sensitivity to reasoning path variations, resulting in asymmetric omni-modal grounding. Notably, models struggle when processing the speech modality, underscoring the need for balanced, multi-hop evaluation of omni-modal intelligence.
>
---
#### [replaced 021] One Sentence, Two Embeddings: Contrastive Learning of Explicit and Implicit Semantic Representations
- **分类: cs.CL**

- **简介: 该论文属于句子嵌入任务，旨在解决传统方法无法同时捕捉显性和隐性语义的问题。提出DualCSE，为每句话生成两个嵌入，分别表示显性和隐性语义，提升下游任务性能。**

- **链接: [https://arxiv.org/pdf/2510.09293v2](https://arxiv.org/pdf/2510.09293v2)**

> **作者:** Kohei Oda; Po-Min Chuang; Kiyoaki Shirai; Natthawut Kertkeidkachorn
>
> **备注:** EACL 2026 Findings
>
> **摘要:** Sentence embedding methods have made remarkable progress, yet they still struggle to capture the implicit semantics within sentences. This can be attributed to the inherent limitations of conventional sentence embedding methods that assign only a single vector per sentence. To overcome this limitation, we propose DualCSE, a sentence embedding method that assigns two embeddings to each sentence: one representing the explicit semantics and the other representing the implicit semantics. These embeddings coexist in the shared space, enabling the selection of the desired semantics for specific purposes such as information retrieval and text classification. Experimental results demonstrate that DualCSE can effectively encode both explicit and implicit meanings and improve the performance of the downstream task.
>
---
#### [replaced 022] Rewarding the Rare: Uniqueness-Aware RL for Creative Problem Solving in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLMs在复杂推理中探索不足的问题。通过奖励独特策略提升解的多样性与质量。**

- **链接: [https://arxiv.org/pdf/2601.08763v2](https://arxiv.org/pdf/2601.08763v2)**

> **作者:** Zhiyuan Hu; Yucheng Wang; Yufei He; Jiaying Wu; Yilun Zhao; See-Kiong Ng; Cynthia Breazeal; Anh Tuan Luu; Hae Won Park; Bryan Hooi
>
> **备注:** Work in Progress
>
> **摘要:** Reinforcement learning (RL) has become a central paradigm for post-training large language models (LLMs), particularly for complex reasoning tasks, yet it often suffers from exploration collapse: policies prematurely concentrate on a small set of dominant reasoning patterns, improving pass@1 while limiting rollout-level diversity and gains in pass@k. We argue that this failure stems from regularizing local token behavior rather than diversity over sets of solutions. To address this, we propose Uniqueness-Aware Reinforcement Learning, a rollout-level objective that explicitly rewards correct solutions that exhibit rare high-level strategies. Our method uses an LLM-based judge to cluster rollouts for the same problem according to their high-level solution strategies, ignoring superficial variations, and reweights policy advantages inversely with cluster size. As a result, correct but novel strategies receive higher rewards than redundant ones. Across mathematics, physics, and medical reasoning benchmarks, our approach consistently improves pass@$k$ across large sampling budgets and increases the area under the pass@$k$ curve (AUC@$K$) without sacrificing pass@1, while sustaining exploration and uncovering more diverse solution strategies at scale.
>
---
#### [replaced 023] User Perceptions vs. Proxy LLM Judges: Privacy and Helpfulness in LLM Responses to Privacy-Sensitive Scenarios
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于隐私与人工智能任务，旨在解决LLMs在隐私敏感场景中平衡帮助性与隐私保护的问题。通过用户研究，发现代理模型无法准确反映用户对隐私和效用的感知。**

- **链接: [https://arxiv.org/pdf/2510.20721v3](https://arxiv.org/pdf/2510.20721v3)**

> **作者:** Xiaoyuan Wu; Roshni Kaushik; Wenkai Li; Lujo Bauer; Koichi Onoue
>
> **摘要:** Large language models (LLMs) are rapidly being adopted for tasks like drafting emails, summarizing meetings, and answering health questions. In these settings, users may need to share private information (e.g., contact details, health records). To evaluate LLMs' ability to identify and redact such information, prior work introduced real-life, scenario-based benchmarks (e.g., ConfAIde, PrivacyLens) and found that LLMs can leak private information in complex scenarios. However, these evaluations relied on proxy LLMs to judge the helpfulness and privacy-preservation quality of LLM responses, rather than directly measuring users' perceptions. To understand how users perceive the helpfulness and privacy-preservation quality of LLM responses to privacy-sensitive scenarios, we conducted a user study ($n=94$) using 90 PrivacyLens scenarios. We found that users had low agreement with each other when evaluating identical LLM responses. In contrast, five proxy LLMs reached high agreement, yet each proxy LLM had low correlation with users' evaluations. These results indicate that proxy LLMs cannot accurately estimate users' wide range of perceptions of utility and privacy in privacy-sensitive scenarios. We discuss the need for more user-centered studies to measure LLMs' ability to help users while preserving privacy, and for improving alignment between LLMs and users in estimating perceived privacy and utility.
>
---
#### [replaced 024] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型优化任务，旨在解决KV缓存淘汰中忽视全局信息的问题。通过引入可训练的软令牌列表，提升查询对全局信息的捕捉能力，从而优化缓存淘汰效果。**

- **链接: [https://arxiv.org/pdf/2509.10798v2](https://arxiv.org/pdf/2509.10798v2)**

> **作者:** Yijun Liu; Yixuan Wang; Yuzhuang Xu; Shiyu Ji; Yang Xu; Qingfu Zhu; Wanxiang Che
>
> **备注:** Accepted in AAAI 2026
>
> **摘要:** Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios.
>
---
#### [replaced 025] PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization
- **分类: cs.CL**

- **简介: 该论文提出PlotCraft基准，解决LLMs在复杂数据可视化任务中的不足，通过构建多样化任务和代码生成模型PlotCraftor提升可视化能力。**

- **链接: [https://arxiv.org/pdf/2511.00010v2](https://arxiv.org/pdf/2511.00010v2)**

> **作者:** Jiajun Zhang; Jianke Zhang; Zeyu Cui; Jiaxi Yang; Lei Zhang; Binyuan Hui; Qiang Liu; Zilei Wang; Liang Wang; Junyang Lin
>
> **摘要:** Recent Large Language Models (LLMs) have demonstrated remarkable proficiency in code generation. However, their ability to create complex visualizations for scaled and structured data remains largely unevaluated and underdeveloped. To address this gap, we introduce PlotCraft, a new benchmark featuring 1k challenging visualization tasks that cover a wide range of topics, such as finance, scientific research, and sociology. The benchmark is structured around seven high-level visualization tasks and encompasses 48 distinct chart types. Crucially, it is the first to systematically evaluate both single-turn generation and multi-turn refinement across a diverse spectrum of task complexities. Our comprehensive evaluation of 23 leading LLMs on PlotCraft reveals obvious performance deficiencies in handling sophisticated visualization tasks. To bridge this performance gap, we develope SynthVis-30K, a large-scale, high-quality dataset of complex visualization code synthesized via a collaborative agent framework. Building upon this dataset, we develope PlotCraftor, a novel code generation model that achieves strong capabilities in complex data visualization with a remarkably small size. Across VisEval, PandasPlotBench, and our proposed PlotCraft, PlotCraftor shows performance comparable to that of leading proprietary approaches. Especially, on hard task, Our model achieves over 50% performance improvement. We will release the benchmark, dataset, and code at https://github.com/Speakn0w/PlotCraft-Benchmark.
>
---
#### [replaced 026] Rakuten Data Release: A Large-Scale and Long-Term Reviews Corpus for Hotel Domain
- **分类: cs.CL**

- **简介: 该论文发布了一个大规模酒店评论语料库，包含2009至2024年的729万条数据，用于分析评论变化及数据漂移问题。任务为情感分析与数据趋势研究。**

- **链接: [https://arxiv.org/pdf/2512.15151v4](https://arxiv.org/pdf/2512.15151v4)**

> **作者:** Yuki Nakayama; Koki Hikichi; Yun Ching Liu; Yu Hirate
>
> **备注:** 6 pages
>
> **摘要:** This paper presents a large-scale corpus of Rakuten Travel Reviews. Our collection contains 7.29 million customer reviews for 16 years, ranging from 2009 to 2024. Each record in the dataset contains the review text, its response from an accommodation, an anonymized reviewer ID, review date, accommodation ID, plan ID, plan title, room type, room name, purpose, accompanying group, and user ratings from six aspect categories, as well as an overall score. We present statistical information about our corpus and provide insights into factors driving data drift between 2019 and 2024 using statistical approaches.
>
---
#### [replaced 027] CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation
- **分类: cs.CL**

- **简介: 该论文提出CoSense-LLM，解决边缘计算中语义理解与隐私保护问题，通过轻量编码、本地检索和成本感知策略实现高效安全的云边协作。**

- **链接: [https://arxiv.org/pdf/2510.19670v4](https://arxiv.org/pdf/2510.19670v4)**

> **作者:** Hasan Akgul; Mari Eplik; Javier Rojas; Aina Binti Abdullah; Pieter van der Merwe
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to unverifiable authorship and affiliation
>
> **摘要:** We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
>
---
#### [replaced 028] Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文聚焦多语言对话语音识别任务，旨在解决LLM与端到端模型性能差距问题。通过融合微调的Whisper和mHuBERT编码器，提升语音表示，优化模型表现。**

- **链接: [https://arxiv.org/pdf/2601.01461v2](https://arxiv.org/pdf/2601.01461v2)**

> **作者:** Yuxiang Mei; Dongxing Xu; Jiaen Liang; Yanhua Long
>
> **备注:** 5 pages, 1 figure
>
> **摘要:** The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
>
---
#### [replaced 029] TranslateGemma Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在提升Gemma 3模型的翻译性能。通过两阶段微调和强化学习优化，开发出TranslateGemma模型，在多个基准测试中表现优异。**

- **链接: [https://arxiv.org/pdf/2601.09012v2](https://arxiv.org/pdf/2601.09012v2)**

> **作者:** Mara Finkelstein; Isaac Caswell; Tobias Domhan; Jan-Thorsten Peter; Juraj Juraska; Parker Riley; Daniel Deutsch; Cole Dilanni; Colin Cherry; Eleftheria Briakou; Elizabeth Nielsen; Jiaming Luo; Kat Black; Ryan Mullins; Sweta Agrawal; Wenda Xu; Erin Kats; Stephane Jaskiewicz; Markus Freitag; David Vilar
>
> **摘要:** We present TranslateGemma, a suite of open machine translation models based on the Gemma 3 foundation models. To enhance the inherent multilingual capabilities of Gemma 3 for the translation task, we employ a two-stage fine-tuning process. First, supervised fine-tuning is performed using a rich mixture of high-quality large-scale synthetic parallel data generated via state-of-the-art models and human-translated parallel data. This is followed by a reinforcement learning phase, where we optimize translation quality using an ensemble of reward models, including MetricX-QE and AutoMQM, targeting translation quality. We demonstrate the effectiveness of TranslateGemma with human evaluation on the WMT25 test set across 10 language pairs and with automatic evaluation on the WMT24++ benchmark across 55 language pairs. Automatic metrics show consistent and substantial gains over the baseline Gemma 3 models across all sizes. Notably, smaller TranslateGemma models often achieve performance comparable to larger baseline models, offering improved efficiency. We also show that TranslateGemma models retain strong multimodal capabilities, with enhanced performance on the Vistra image translation benchmark. The release of the open TranslateGemma models aims to provide the research community with powerful and adaptable tools for machine translation.
>
---
#### [replaced 030] Fairness Definitions in Language Models Explained
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型中的公平性问题。通过系统梳理公平性定义及分类，提出新分类体系并进行实验验证，以促进更公正的模型应用。**

- **链接: [https://arxiv.org/pdf/2407.18454v3](https://arxiv.org/pdf/2407.18454v3)**

> **作者:** Zhipeng Yin; Zichong Wang; Avash Palikhe; Wenbin Zhang
>
> **摘要:** Language Models (LMs) have demonstrated exceptional performance across various Natural Language Processing (NLP) tasks. Despite these advancements, LMs can inherit and amplify societal biases related to sensitive attributes such as gender and race, limiting their adoption in real-world applications. Therefore, fairness has been extensively explored in LMs, leading to the proposal of various fairness notions. However, the lack of clear agreement on which fairness definition to apply in specific contexts and the complexity of understanding the distinctions between these definitions can create confusion and impede further progress. To this end, this paper proposes a systematic survey that clarifies the definitions of fairness as they apply to LMs. Specifically, we begin with a brief introduction to LMs and fairness in LMs, followed by a comprehensive, up-to-date overview of existing fairness notions in LMs and the introduction of a novel taxonomy that categorizes these concepts based on their transformer architecture: encoder-only, decoder-only, and encoder-decoder LMs. We further illustrate each definition through experiments, showcasing their practical implications and outcomes. Finally, we discuss current research challenges and open questions, aiming to foster innovative ideas and advance the field. The repository is publicly available online at https://github.com/vanbanTruong/Fairness-in-Large-Language-Models/tree/main/definitions.
>
---
#### [replaced 031] GraphIF: Enhancing Multi-Turn Instruction Following for Large Language Models with Relation Graph Prompt
- **分类: cs.CL**

- **简介: 该论文属于多轮对话任务，旨在解决长距离指令遵循难题。通过构建关系图结构并生成图提示，增强语言模型的多轮指令跟随能力。**

- **链接: [https://arxiv.org/pdf/2511.10051v2](https://arxiv.org/pdf/2511.10051v2)**

> **作者:** Zhenhe Li; Can Lin; Ling Zheng; Wen-Da Wei; Junli Liang; Qi Song
>
> **摘要:** Multi-turn instruction following is essential for building intelligent conversational systems that can consistently adhere to instructions across dialogue turns. However, existing approaches to enhancing multi-turn instruction following primarily rely on collecting or generating large-scale multi-turn dialogue datasets to fine-tune large language models (LLMs), which treat each response generation as an isolated task and fail to explicitly incorporate multi-turn instruction following into the optimization objectives. As a result, instruction-tuned LLMs often struggle with complex long-distance constraints. In multi-turn dialogues, relational constraints across turns can be naturally modeled as labeled directed edges, making graph structures particularly suitable for modeling multi-turn instruction following. Despite this potential, leveraging graph structures to enhance the multi-turn instruction following capabilities of LLMs remains unexplored. To bridge this gap, we propose GraphIF, a plug-and-play framework that models multi-turn dialogues as directed relation graphs and leverages graph prompts to enhance the instruction following capabilities of LLMs. GraphIF comprises three key components: (1) an agent-based relation extraction module that captures inter-turn semantic relations via action-triggered mechanisms to construct structured graphs; (2) a relation graph prompt generation module that converts structured graph information into natural language prompts; and (3) a response rewriting module that refines initial LLM outputs using the generated graph prompts. Extensive experiments on two long multi-turn dialogue datasets demonstrate that GraphIF can be seamlessly integrated into instruction-tuned LLMs and leads to significant improvements across all four multi-turn instruction-following evaluation metrics.
>
---
#### [replaced 032] How Quantization Shapes Bias in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究量化对大语言模型偏差的影响，探讨如何在提升效率的同时兼顾伦理问题。**

- **链接: [https://arxiv.org/pdf/2508.18088v2](https://arxiv.org/pdf/2508.18088v2)**

> **作者:** Federico Marcuzzi; Xuefei Ning; Roy Schwartz; Iryna Gurevych
>
> **摘要:** This work presents a comprehensive evaluation of how quantization affects model bias, with particular attention to its impact on individual demographic subgroups. We focus on weight and activation quantization strategies and examine their effects across a broad range of bias types, including stereotypes, fairness, toxicity, and sentiment. We employ both probability- and generated text-based metrics across 13 benchmarks and evaluate models that differ in architecture family and reasoning ability. Our findings show that quantization has a nuanced impact on bias: while it can reduce model toxicity and does not significantly impact sentiment, it tends to slightly increase stereotypes and unfairness in generative tasks, especially under aggressive compression. These trends are generally consistent across demographic categories and subgroups, and model types, although their magnitude depends on the specific setting. Overall, our results highlight the importance of carefully balancing efficiency and ethical considerations when applying quantization in practice.
>
---
#### [replaced 033] Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，针对长解码阶段的推理效率问题。研究发现稀疏注意力导致信息丢失，反而增加复杂度，提出早停算法优化性能。**

- **链接: [https://arxiv.org/pdf/2601.03043v2](https://arxiv.org/pdf/2601.03043v2)**

> **作者:** Junhao Hu; Fangze Li; Mingtao Xu; Feifan Meng; Shiju Zhao; Tiancheng Hu; Ting Peng; Anmin Liu; Wenrui Huang; Chenxu Liu; Ziyue Hua; Tao Xie
>
> **摘要:** Large language models (LLMs) demonstrate strong capabilities across a wide range of complex tasks and are increasingly deployed at scale, placing significant demands on inference efficiency. Prior work typically decomposes inference into prefill and decode stages, with the decode stage dominating total latency. To reduce time and memory complexity in the decode stage, a line of work introduces sparse-attention algorithms. In this paper, we show, both empirically and theoretically, that sparse attention can paradoxically increase end-to-end complexity: information loss often induces significantly longer sequences, a phenomenon we term ``Less is Less'' (Lil). To mitigate the Lil problem, we propose an early-stopping algorithm that detects the threshold where information loss exceeds information gain during sparse decoding. Our early-stopping algorithm reduces token consumption by up to 90% with a marginal accuracy degradation of less than 2% across reasoning-intensive benchmarks.
>
---
#### [replaced 034] Mistake Notebook Learning: Batch-Clustered Failures for Training-Free Agent Adaptation
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，针对LLM代理在持续任务中无法有效学习错误的问题，提出Mistake Notebook Learning（MNL）框架，通过聚类失败案例生成通用指导，提升代理适应能力。**

- **链接: [https://arxiv.org/pdf/2512.11485v2](https://arxiv.org/pdf/2512.11485v2)**

> **作者:** Xuanbo Su; Yingfang Zhang; Hao Luo; Xiaoteng Liu; Leo Huang
>
> **摘要:** With the growing adoption of Large Language Model (LLM) agents in persistent, real-world roles, they naturally encounter continuous streams of tasks and inevitable failures. A key limitation, however, is their inability to systematically learn from these mistakes, forcing them to repeat identical errors in similar contexts. Unlike prior training-free methods that primarily store raw instance-level experience or focus on retrieving successful trajectories, we propose Mistake Notebook Learning (MNL), a novel memory framework that enables agents to self-curate generalizable guidance from batch-clustered failures. This mechanism allows agents to distill shared error patterns into structured ``mistake notes,'' updating an external memory only when batch performance improves to ensure stability. To further amplify adaptability, we integrate MNL with test-time scaling, leveraging aggregated failure patterns to actively steer the search process away from known pitfalls. Experiments on mathematical reasoning, Text-to-SQL, and interactive agent benchmarks show that MNL achieves competitive performance compared to existing memory mechanisms and in-context methods in both effectiveness and efficiency. These findings position structured mistake abstraction as a critical lever for robust agent evolution, enabling continuous improvement without the cost of parameter updates.
>
---
#### [replaced 035] Controlled Self-Evolution for Algorithmic Code Optimization
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文属于算法代码优化任务，旨在解决自进化方法效率低的问题。提出CSE框架，通过结构化初始化、反馈驱动进化和层次记忆提升优化效果。**

- **链接: [https://arxiv.org/pdf/2601.07348v4](https://arxiv.org/pdf/2601.07348v4)**

> **作者:** Tu Hu; Ronghao Chen; Shuo Zhang; Jianghao Yin; Mou Xiao Feng; Jingping Liu; Shaolei Zhang; Wenqi Jiang; Yuqi Fang; Sen Hu; Huacan Wang; Yi Xu
>
> **备注:** 27 pages
>
> **摘要:** Self-evolution methods enhance code generation through iterative "generate-verify-refine" cycles, yet existing approaches suffer from low exploration efficiency, failing to discover solutions with superior complexity within limited budgets. This inefficiency stems from initialization bias trapping evolution in poor solution regions, uncontrolled stochastic operations lacking feedback guidance, and insufficient experience utilization across tasks. To address these bottlenecks, we propose Controlled Self-Evolution (CSE), which consists of three key components. Diversified Planning Initialization generates structurally distinct algorithmic strategies for broad solution space coverage. Genetic Evolution replaces stochastic operations with feedback-guided mechanisms, enabling targeted mutation and compositional crossover. Hierarchical Evolution Memory captures both successful and failed experiences at inter-task and intra-task levels. Experiments on EffiBench-X demonstrate that CSE consistently outperforms all baselines across various LLM backbones. Furthermore, CSE achieves higher efficiency from early generations and maintains continuous improvement throughout evolution. Our code is publicly available at https://github.com/QuantaAlpha/EvoControl.
>
---
#### [replaced 036] The Open Proof Corpus: A Large-Scale Study of LLM-Generated Mathematical Proofs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学证明生成任务，旨在解决LLM生成证明质量评估问题。构建了OPC数据集，包含5000个高质量人工评估的证明，用于研究证明生成性能及优化方法。**

- **链接: [https://arxiv.org/pdf/2506.21621v2](https://arxiv.org/pdf/2506.21621v2)**

> **作者:** Jasper Dekoninck; Ivo Petrov; Kristian Minchev; Mislav Balunovic; Martin Vechev; Miroslav Marinov; Maria Drencheva; Lyuba Konova; Milen Shumanov; Kaloyan Tsvetkov; Nikolay Drenchev; Lazar Todorov; Kalina Nikolova; Nikolay Georgiev; Vanesa Kalinkova; Margulan Ismoldayev
>
> **摘要:** In recent months, large language models (LLMs) have made significant progress in mathematical proof generation, but further advancement is hindered by the lack of a large-scale, high-quality dataset of human-evaluated proofs. While expensive to create, such a dataset is essential for driving improvements in training and enabling a rigorous analysis of proof generation capabilities. In this work, we present the Open Proof Corpus (OPC), a dataset comprising over 5,000 human-evaluated proofs produced by state-of-the-art LLMs. The OPC was specifically designed for broad applicability and downstream usage in proof generation research and is the first to include a substantial number of correct, LLM-generated solutions to problems from prestigious mathematics competitions such as the USAMO and IMO. Using the OPC, we explore critical questions in automated proof generation: (1) the performance gap between natural language and formal proof generation, (2) the discrepancy between final-answer accuracy and full-proof validity, and (3) the impact of best-of-n selection on proof quality. Finally, to showcase the utility of the OPC, we finetune an 8B-parameter model on the dataset, obtaining a model that performs on par with the best model, Gemini-2.5-Pro, on the task of evaluating proof correctness.
>
---
#### [replaced 037] On the Failure of Latent State Persistence in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，研究LLM在保持内部状态上的不足。通过实验揭示LLM缺乏持续的潜在状态，导致推理能力受限。**

- **链接: [https://arxiv.org/pdf/2505.10571v4](https://arxiv.org/pdf/2505.10571v4)**

> **作者:** Jen-tse Huang; Kaiser Sun; Wenxuan Wang; Mark Dredze
>
> **备注:** 8 pages, 6 figures, 9 tables
>
> **摘要:** While Large Language Models (LLMs) excel in reasoning, whether they can sustain persistent latent states remains under-explored. The capacity to maintain and manipulate unexpressed, internal representations-analogous to human working memory-is a cornerstone of complex reasoning. In this paper, we formalize and quantify the "Latent State Persistence" (LSP) gap through three novel experiments. First, we utilize a Number Guessing Game, demonstrating that across independent queries, LLMs fail to allocate probability mass to a singular hidden choice, violating a fundamental probabilistic principle. Second, we employ a Yes-No Game to show that as the number of questions increases, LLMs suffer from "concept drift," leading to inevitable self-contradictions due to the lack of LSP. Finally, inspired by Mathematical Mentalism, we task models with tracking transformations on hidden variables, revealing a failure in variable binding and state evolution when the initial state is not explicitly present in the context. Collectively, these findings suggest that LLMs function as reactive post-hoc solvers rather than proactive planners with LSP. Our work provides a framework for evaluating the fidelity of internal representations and highlights a fundamental architectural divergence between autoregressive transformers and human-like cognition.
>
---
#### [replaced 038] Exploring the Translation Mechanism of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的机器翻译任务，旨在揭示大语言模型的翻译机制。通过分析模型组件，发现关键注意力头和MLP在翻译中的作用，并验证了少量参数微调的有效性。**

- **链接: [https://arxiv.org/pdf/2502.11806v3](https://arxiv.org/pdf/2502.11806v3)**

> **作者:** Hongbin Zhang; Kehai Chen; Xuefeng Bai; Xiucheng Li; Yang Xiang; Min Zhang
>
> **备注:** Accepted in NeurIPS 2025 Poster
>
> **摘要:** While large language models (LLMs) demonstrate remarkable success in multilingual translation, their internal core translation mechanisms, even at the fundamental word level, remain insufficiently understood. To address this critical gap, this work introduces a systematic framework for interpreting the mechanism behind LLM translation from the perspective of computational components. This paper first proposes subspace-intervened path patching for precise, fine-grained causal analysis, enabling the detection of components crucial to translation tasks and subsequently characterizing their behavioral patterns in human-interpretable terms. Comprehensive experiments reveal that translation is predominantly driven by a sparse subset of components: specialized attention heads serve critical roles in extracting source language, translation indicators, and positional features, which are then integrated and processed by specific multi-layer perceptrons (MLPs) into intermediary English-centric latent representations before ultimately yielding the final translation. The significance of these findings is underscored by the empirical demonstration that targeted fine-tuning a minimal parameter subset ($<5\%$) enhances translation performance while preserving general capabilities. This result further indicates that these crucial components generalize effectively to sentence-level translation and are instrumental in elucidating more intricate translation tasks.
>
---
#### [replaced 039] Collaborative Multi-Agent Test-Time Reinforcement Learning for Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多智能体强化学习任务，解决训练资源消耗大、不稳定的问题，提出MATTRL框架，在推理阶段注入结构化文本经验提升性能。**

- **链接: [https://arxiv.org/pdf/2601.09667v2](https://arxiv.org/pdf/2601.09667v2)**

> **作者:** Zhiyuan Hu; Yunhai Hu; Juncheng Liu; Shuyue Stella Li; Yucheng Wang; Zhen Xu; See-Kiong Ng; Anh Tuan Luu; Xinxing Xu; Bryan Hooi; Cynthia Breazeal; Hae Won Park
>
> **备注:** Work in Progress
>
> **摘要:** Multi-agent systems have evolved into practical LLM-driven collaborators for many applications, gaining robustness from diversity and cross-checking. However, multi-agent RL (MARL) training is resource-intensive and unstable: co-adapting teammates induce non-stationarity, and rewards are often sparse and high-variance. Therefore, we introduce \textbf{Multi-Agent Test-Time Reinforcement Learning (MATTRL)}, a framework that injects structured textual experience into multi-agent deliberation at inference time. MATTRL forms a multi-expert team of specialists for multi-turn discussions, retrieves and integrates test-time experiences, and reaches consensus for final decision-making. We also study credit assignment for constructing a turn-level experience pool, then reinjecting it into the dialogue. Across challenging benchmarks in medicine, math, and education, MATTRL improves accuracy by an average of 3.67\% over a multi-agent baseline, and by 8.67\% over comparable single-agent baselines. Ablation studies examine different credit-assignment schemes and provide a detailed comparison of how they affect training outcomes. MATTRL offers a stable, effective and efficient path to distribution-shift-robust multi-agent reasoning without tuning.
>
---
#### [replaced 040] Scalable and Reliable Evaluation of AI Knowledge Retrieval Systems: RIKER and the Coherent Simulated Universe
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI知识检索系统评估任务，解决传统评估方法的污染、偏差和成本问题，提出RIKER方法，通过生成文档进行可扩展、无污染的评价。**

- **链接: [https://arxiv.org/pdf/2601.08847v2](https://arxiv.org/pdf/2601.08847v2)**

> **作者:** JV Roig
>
> **备注:** 26 pages, 17 tables, 1 figure
>
> **摘要:** Evaluating knowledge systems (LLMs, RAG, knowledge graphs, etc) faces fundamental challenges: static benchmarks are vulnerable to contamination, LLM-based judges exhibit systematic biases, and ground truth extraction requires expensive human annotation. We present RIKER (Retrieval Intelligence and Knowledge Extraction Rating), both a benchmark and a replicable methodology based on paradigm inversion - generating documents from known ground truth rather than extracting ground truth from documents. This approach enables deterministic scoring and scalable evaluation without human annotation or reference models, and contamination resistance through regenerable corpora. Our evaluation of 33 models using over 21 billion tokens reveals that context length claims frequently exceed usable capacity, with significant degradation beyond 32K tokens; cross-document aggregation proves substantially harder than single-document extraction; and grounding ability and hallucination resistance are distinct capabilities - models excelling at finding facts that exist may still fabricate facts that do not. Beyond the specific benchmark, we contribute a domain-agnostic methodology for constructing scalable and contamination-resistant evaluations wherever synthetic documents can be generated from structured ground truth.
>
---
#### [replaced 041] Are Language Models Efficient Reasoners? A Perspective from Logic Programming
- **分类: cs.CL; cs.AI; cs.LG; cs.LO**

- **简介: 该论文属于推理效率评估任务，旨在解决语言模型在冗余信息下推理效率低的问题。通过逻辑编程框架，评估模型是否能有效忽略无关信息。**

- **链接: [https://arxiv.org/pdf/2510.25626v2](https://arxiv.org/pdf/2510.25626v2)**

> **作者:** Andreas Opedal; Yanick Zengaffinen; Haruki Shirakami; Clemente Pasti; Mrinmaya Sachan; Abulhair Saparov; Ryan Cotterell; Bernhard Schölkopf
>
> **备注:** NeurIPS 2025
>
> **摘要:** Modern language models (LMs) exhibit strong deductive reasoning capabilities, yet standard evaluations emphasize correctness while overlooking a key aspect of reasoning: efficiency. In real-world reasoning scenarios, much of the available information is irrelevant, and effective deductive inference requires identifying and ignoring such distractions. We propose a framework for assessing LM reasoning efficiency through the lens of logic programming, introducing a simple method to align proofs written in natural language -- as generated by an LM -- with shortest proofs found by executing the logic program. Efficiency is quantified by measuring how well a model avoids unnecessary inference. Empirically, we construct a dataset of math word problems injected with various number of irrelevant axioms that vary in semantic overlap with the goal theorem. We find that current LMs show marked accuracy declines under such conditions -- even with minimal, domain-consistent distractions -- and the proofs they generate frequently exhibit detours through irrelevant inferences.
>
---
#### [replaced 042] Text Classification Under Class Distribution Shift: A Survey
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本分类任务，研究类别分布变化下的分类问题。针对数据分布随时间变化导致模型性能下降的问题，综述了开放集、零样本和通用学习等方法，并探讨了持续学习的解决方案。**

- **链接: [https://arxiv.org/pdf/2502.12965v3](https://arxiv.org/pdf/2502.12965v3)**

> **作者:** Adriana Valentina Costache; Silviu Florin Gheorghe; Eduard Gabriel Poesina; Paul Irofti; Radu Tudor Ionescu
>
> **备注:** Accepted at EACL 2026 (main)
>
> **摘要:** The basic underlying assumption of machine learning (ML) models is that the training and test data are sampled from the same distribution. However, in daily practice, this assumption is often broken, i.e. the distribution of the test data changes over time, which hinders the application of conventional ML models. One domain where the distribution shift naturally occurs is text classification, since people always find new topics to discuss. To this end, we survey research articles studying open-set text classification and related tasks. We divide the methods in this area based on the constraints that define the kind of distribution shift and the corresponding problem formulation, i.e. learning with the Universum, zero-shot learning, and open-set learning. We next discuss the predominant mitigation approaches for each problem setup. We further identify several future work directions, aiming to push the boundaries beyond the state of the art. Finally, we explain how continual learning can solve many of the issues caused by the shifting class distribution. We maintain a list of relevant papers at https://github.com/Eduard6421/Open-Set-Survey.
>
---
#### [replaced 043] Knowledge Homophily in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.SI**

- **简介: 该论文研究LLM中的知识同质性，解决知识组织结构不明确的问题。通过图神经网络估计实体知识水平，提升知识覆盖效率与问答性能。**

- **链接: [https://arxiv.org/pdf/2509.23773v2](https://arxiv.org/pdf/2509.23773v2)**

> **作者:** Utkarsh Sahu; Zhisheng Qi; Mahantesh Halappanavar; Nedim Lipka; Ryan A. Rossi; Franck Dernoncourt; Yu Zhang; Yao Ma; Yu Wang
>
> **摘要:** Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, the structural organization of their knowledge remains unexplored. Inspired by cognitive neuroscience findings, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by this homophily principle, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging their neighborhood scores. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for fine-tuning to inject knowledge into LLMs but also enhances multi-hop path retrieval in reasoning-intensive question answering.
>
---
#### [replaced 044] Filling in the Clinical Gaps in Benchmark: Case for HealthBench for the Japanese medical system
- **分类: cs.CL**

- **简介: 该论文属于医疗AI评估任务，旨在解决现有医学基准在日语环境中的适用性问题。通过分析HealthBench的翻译版本，发现其与日本临床指南存在差距，并提出本地化改进方案。**

- **链接: [https://arxiv.org/pdf/2509.17444v3](https://arxiv.org/pdf/2509.17444v3)**

> **作者:** Shohei Hisada; Endo Sunao; Himi Yamato; Shoko Wakamiya; Eiji Aramaki
>
> **备注:** draft v0.3 Code and analysis data is available at https://zenodo.org/records/17405321
>
> **摘要:** This study investigates the applicability of HealthBench, a large-scale, rubric-based medical benchmark, to the Japanese context. Although robust evaluation frameworks are essential for the safe development of medical LLMs, resources in Japanese are scarce and often consist of translated multiple-choice questions. Our research addresses this issue in two ways. First, we establish a performance baseline by applying a machine-translated version of HealthBench's 5,000 scenarios to evaluate two models: a high-performing multilingual model (GPT-4.1) and a Japanese-native open-source model (LLM-jp-3.1). Secondly, we use an LLM-as-a-Judge approach to systematically classify the benchmark's scenarios and rubric criteria. This allows us to identify 'contextual gaps' where the content is misaligned with Japan's clinical guidelines, healthcare systems or cultural norms. Our findings reveal a modest performance drop in GPT-4.1 due to rubric mismatches, as well as a significant failure in the Japanese-native model, which lacked the required clinical completeness. Furthermore, our classification shows that, despite most scenarios being applicable, a significant proportion of the rubric criteria require localisation. This work underscores the limitations of direct benchmark translation and highlights the urgent need for a context-aware, localised adaptation, a "J-HealthBench", to ensure the reliable and safe evaluation of medical LLMs in Japan.
>
---
#### [replaced 045] CoMAT: Chain of Mathematically Annotated Thought Improves Mathematical Reasoning
- **分类: cs.AI; cs.CL; cs.LG; cs.SC**

- **简介: 该论文属于数学推理任务，旨在提升大语言模型的数学推理能力。提出CoMAT方法，通过符号转换和推理执行两个阶段增强模型的推理效果。**

- **链接: [https://arxiv.org/pdf/2410.10336v2](https://arxiv.org/pdf/2410.10336v2)**

> **作者:** Joshua Ong Jun Leang; Aryo Pradipta Gema; Shay B. Cohen
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** Mathematical reasoning remains a significant challenge for large language models (LLMs), despite progress in prompting techniques such as Chain-of-Thought (CoT). We present **Chain of Mathematically Annotated Thought (CoMAT)**, which enhances reasoning through two stages: *Symbolic Conversion* (converting natural language queries into symbolic form) and *Reasoning Execution* (deriving answers from symbolic representations). CoMAT operates entirely with a single LLM and without external solvers. Across four LLMs, CoMAT outperforms traditional CoT on six out of seven benchmarks, achieving gains of 4.48% on MMLU-Redux (MATH) and 4.58% on GaoKao MCQ. In addition to improved performance, CoMAT ensures faithfulness and verifiability, offering a transparent reasoning process for complex mathematical tasks
>
---
#### [replaced 046] Classifying and Addressing the Diversity of Errors in Retrieval-Augmented Generation Systems
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG系统中的错误分类与修复问题。提出错误分类体系，提供数据集和自动评估方法，以提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2510.13975v2](https://arxiv.org/pdf/2510.13975v2)**

> **作者:** Kin Kwan Leung; Mouloud Belbahri; Yi Sui; Alex Labach; Xueying Zhang; Stephen Anthony Rose; Jesse C. Cresswell
>
> **备注:** EACL 2026
>
> **摘要:** Retrieval-augmented generation (RAG) is a prevalent approach for building LLM-based question-answering systems that can take advantage of external knowledge databases. Due to the complexity of real-world RAG systems, there are many potential causes for erroneous outputs. Understanding the range of errors that can occur in practice is crucial for robust deployment. We present a new taxonomy of the error types that can occur in realistic RAG systems, examples of each, and practical advice for addressing them. Additionally, we curate a dataset of erroneous RAG responses annotated by error types. We then propose an auto-evaluation method aligned with our taxonomy that can be used in practice to track and address errors during development. Code and data are available at https://github.com/layer6ai-labs/rag-error-classification.
>
---
#### [replaced 047] A.X K1 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文介绍A.X K1，一个519B参数的Mixture-of-Experts语言模型，旨在提升推理能力和推理效率。通过优化训练配置和数据处理，解决多场景部署问题。**

- **链接: [https://arxiv.org/pdf/2601.09200v2](https://arxiv.org/pdf/2601.09200v2)**

> **作者:** Sung Jun Cheon; Jaekyung Cho; Seongho Choi; Hyunjun Eun; Seokhwan Jo; Jaehyun Jun; Minsoo Kang; Jin Kim; Jiwon Kim; Minsang Kim; Sungwan Kim; Seungsik Kim; Tae Yoon Kim; Youngrang Kim; Hyeongmun Lee; Sangyeol Lee; Sungeun Lee; Youngsoon Lee; Yujin Lee; Seongmin Ok; Chanyong Park; Hyewoong Park; Junyoung Park; Hyunho Yang; Subin Yi; Soohyun Bae; Dhammiko Arya; Yongseok Choi; Sangho Choi; Dongyeon Cho; Seungmo Cho; Gyoungeun Han; Yong-jin Han; Seokyoung Hong; Hyeon Hwang; Wonbeom Jang; Minjeong Ju; Wonjin Jung; Keummin Ka; Sungil Kang; Dongnam Kim; Joonghoon Kim; Jonghwi Kim; SaeRom Kim; Sangjin Kim; Seongwon Kim; Youngjin Kim; Seojin Lee; Sunwoo Lee; Taehoon Lee; Chanwoo Park; Sohee Park; Sooyeon Park; Yohan Ra; Sereimony Sek; Seungyeon Seo; Gun Song; Sanghoon Woo; Janghan Yoon; Sungbin Yoon
>
> **备注:** This paper is withdrawn pending additional internal review of the methodology and analysis
>
> **摘要:** We introduce A.X K1, a 519B-parameter Mixture-of-Experts (MoE) language model trained from scratch. Our design leverages scaling laws to optimize training configurations and vocabulary size under fixed computational budgets. A.X K1 is pre-trained on a corpus of approximately 10T tokens, curated by a multi-stage data processing pipeline. Designed to bridge the gap between reasoning capability and inference efficiency, A.X K1 supports explicitly controllable reasoning to facilitate scalable deployment across diverse real-world scenarios. We propose a simple yet effective Think-Fusion training recipe, enabling user-controlled switching between thinking and non-thinking modes within a single unified model. Extensive evaluations demonstrate that A.X K1 achieves performance competitive with leading open-source models, while establishing a distinctive advantage in Korean-language benchmarks.
>
---
#### [replaced 048] Bias Dynamics in BabyLMs: Towards a Compute-Efficient Sandbox for Democratising Pre-Training Debiasing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的公平性研究，旨在解决大模型偏见问题。通过使用低成本的BabyLMs模型，模拟大模型的偏见形成过程，实现高效预训练去偏研究。**

- **链接: [https://arxiv.org/pdf/2601.09421v2](https://arxiv.org/pdf/2601.09421v2)**

> **作者:** Filip Trhlik; Andrew Caines; Paula Buttery
>
> **备注:** 21 pages, 18 figures
>
> **摘要:** Pre-trained language models (LMs) have, over the last few years, grown substantially in both societal adoption and training costs. This rapid growth in size has constrained progress in understanding and mitigating their biases. Since re-training LMs is prohibitively expensive, most debiasing work has focused on post-hoc or masking-based strategies, which often fail to address the underlying causes of bias. In this work, we seek to democratise pre-model debiasing research by using low-cost proxy models. Specifically, we investigate BabyLMs, compact BERT-like models trained on small and mutable corpora that can approximate bias acquisition and learning dynamics of larger models. We show that BabyLMs display closely aligned patterns of intrinsic bias formation and performance development compared to standard BERT models, despite their drastically reduced size. Furthermore, correlations between BabyLMs and BERT hold across multiple intra-model and post-model debiasing methods. Leveraging these similarities, we conduct pre-model debiasing experiments with BabyLMs, replicating prior findings and presenting new insights regarding the influence of gender imbalance and toxicity on bias formation. Our results demonstrate that BabyLMs can serve as an effective sandbox for large-scale LMs, reducing pre-training costs from over 500 GPU-hours to under 30 GPU-hours. This provides a way to democratise pre-model debiasing research and enables faster, more accessible exploration of methods for building fairer LMs.
>
---
#### [replaced 049] Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs在概率推理上的不足。通过贝叶斯教学提升模型信念更新能力，并验证其泛化性。**

- **链接: [https://arxiv.org/pdf/2503.17523v3](https://arxiv.org/pdf/2503.17523v3)**

> **作者:** Linlu Qiu; Fei Sha; Kelsey Allen; Yoon Kim; Tal Linzen; Sjoerd van Steenkiste
>
> **备注:** Nature Communications
>
> **摘要:** Large language models (LLMs) are increasingly used as agents that interact with users and with the world. To do so successfully, LLMs must construct representations of the world and form probabilistic beliefs about them. To provide personalized recommendations, for example, the LLM needs to infer a user's preferences from their behavior over multiple interactions. The Bayesian inference framework lays out the optimal way for an agent to update its beliefs as it receives new information. We first show that LLMs fall far short of the standard defined by the Bayesian framework. We then show that by teaching LLMs to mimic the predictions of the normative Bayesian model, we can dramatically improve their ability to update their beliefs; this ability generalizes to new tasks. We conclude that LLMs can effectively learn reasoning skills from examples and generalize those skills to new domains.
>
---
#### [replaced 050] How Many Human Judgments Are Enough? Feasibility Limits of Human Preference Evaluation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型评估任务，研究如何确定足够的人类判断数量以可靠检测模型改进。工作包括分析人类偏好数据，发现多数比较处于难以检测改进的扩散区域，并提出优化策略。**

- **链接: [https://arxiv.org/pdf/2601.09084v2](https://arxiv.org/pdf/2601.09084v2)**

> **作者:** Wilson Y. Lee
>
> **摘要:** Human preference evaluations are widely used to compare generative models, yet it remains unclear how many judgments are required to reliably detect small improvements. We show that when preference signal is diffuse across prompts (i.e., all prompt types are similarly informative), proportional allocation is minimax-optimal: no allocation strategy substantially improves detectability. Empirical analysis of large-scale human preference datasets shows that most comparisons fall into this diffuse regime, exhibiting small preference margins that require far more judgments than typically collected, even in well-sampled comparisons. These limits persist across evaluation protocols and modalities, including chat, image generation, and code generation with execution feedback. In contrast, curated benchmarks that reduce prompt induced variability systematically induce larger margins and improve detectability through a $1.5\times$ reduction in prompt-level variance. Our results show that inconclusive or negative human evaluation outcomes frequently reflect underpowered evaluation rather than model equivalence, underscoring the need to account explicitly for effect size, budget, and protocol design.
>
---
#### [replaced 051] PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出PMOA-TTS，一个包含5.6万条时间事件的医学文本语料库，用于解决临床叙事中的时间建模问题。属于时间序列分析任务，旨在支持患者轨迹研究与预测建模。**

- **链接: [https://arxiv.org/pdf/2505.20323v2](https://arxiv.org/pdf/2505.20323v2)**

> **作者:** Shahriar Noroozizadeh; Sayantan Kumar; George H. Chen; Jeremy C. Weiss
>
> **摘要:** Clinical narratives encode temporal dynamics essential for modeling patient trajectories, yet large-scale temporally annotated resources are scarce. We introduce PMOA-TTS, a corpus of 124,699 single-patient PubMed Open Access case reports converted into structured textual timelines of (event, time) pairs using a scalable large-language-model pipeline (Llama 3.3 70B and DeepSeek-R1). The corpus comprises over 5.6 million timestamped events, alongside extracted demographics and diagnoses. Technical validation uses a clinician-curated gold set and three measures: semantic event matching, temporal concordance (c-index), and alignment error summarized with Area Under the Log-Time CDF (AULTC). We benchmark alternative prompting and model choices and provide documentation to support reproduction. PMOA-TTS enables research on timeline extraction, temporal reasoning, survival modeling and event forecasting from narrative text, and offers broad diagnostic and demographic coverage. Data and code are openly available in public repositories.
>
---
#### [replaced 052] Dual-Uncertainty Guided Policy Learning for Multimodal Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态强化学习任务，解决视觉输入不确定性影响模型推理的问题。提出DUPL方法，通过双不确定性引导策略学习，提升模型在多模态任务中的准确性。**

- **链接: [https://arxiv.org/pdf/2510.01444v2](https://arxiv.org/pdf/2510.01444v2)**

> **作者:** Rui Liu; Dian Yu; Tong Zheng; Runpeng Dai; Zongxia Li; Wenhao Yu; Zhenwen Liang; Linfeng Song; Haitao Mi; Pratap Tokekar; Dong Yu
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has advanced reasoning capabilities in multimodal large language models. However, existing methods typically treat visual inputs as deterministic, overlooking the perceptual ambiguity inherent to the visual modality. Consequently, they fail to distinguish whether a model's uncertainty stems from complex reasoning or ambiguous perception, preventing the targeted allocation of exploration or learning signals. To address this gap, we introduce DUPL, a dual-uncertainty guided policy learning approach for multimodal RLVR that quantifies and leverages both perceptual uncertainty (via symmetric KL divergence) and output uncertainty (via policy entropy) to guide policy updates. By establishing an uncertainty-driven feedback loop and employing a dynamic branch prioritization mechanism, DUPL recalibrates the policy advantage to focus learning on states with high perceptual or decisional ambiguity, enabling effective targeted exploration beyond passive data augmentation. Implemented on top of GRPO and evaluated on six multimodal mathematical and general-domain reasoning benchmarks, DUPL improves Qwen2.5-VL 3B and 7B models, achieving accuracy gains of up to 11.2% on visual math tasks and up to 7.1% on general-domain reasoning tasks, while consistently outperforming GRPO. These results demonstrate that dual-uncertainty guided policy learning is an effective and generalizable approach for multimodal RLVR.
>
---
#### [replaced 053] Generative Adversarial Gumbel MCTS for Abstract Visual Composition Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于抽象视觉构图生成任务，解决在几何约束和模糊目标下生成结构的问题。结合几何推理与语义模型，利用强化学习优化生成效果。**

- **链接: [https://arxiv.org/pdf/2512.01242v2](https://arxiv.org/pdf/2512.01242v2)**

> **作者:** Zirui Zhao; Boye Niu; David Hsu; Wee Sun Lee
>
> **摘要:** We study abstract visual composition, in which identity is primarily determined by the spatial configuration and relations among a small set of geometric primitives (e.g., parts, symmetry, topology). They are invariant primarily to texture and photorealistic detail. Composing such structures from fixed components under geometric constraints and vague goal specification (such as text) is non-trivial due to combinatorial placement choices, limited data, and discrete feasibility (overlap-free, allowable orientations), which create a sparse solution manifold ill-suited to purely statistical pixel-space generators. We propose a constraint-guided framework that combines explicit geometric reasoning with neural semantics. An AlphaGo-style search enforces feasibility, while a fine-tuned vision-language model scores semantic alignment as reward signals. Our algorithm uses a policy network as a heuristic in Monte-Carlo Tree Search and fine-tunes the network via search-generated plans. Inspired by the Generative Adversarial Network, we use the generated instances for adversarial reward refinement. Over time, the generation should approach the actual data more closely when the reward model cannot distinguish between generated instances and ground-truth. In the Tangram Assembly task, our approach yields higher validity and semantic fidelity than diffusion and auto-regressive baselines, especially as constraints tighten.
>
---
#### [replaced 054] VAL-Bench: Belief Consistency as a measure for Value Alignment in Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出VAL-Bench基准，用于评估语言模型在面对价值争议时信念的一致性，解决价值对齐问题。**

- **链接: [https://arxiv.org/pdf/2510.05465v3](https://arxiv.org/pdf/2510.05465v3)**

> **作者:** Aman Gupta; Denny O'Shea; Fazl Barez
>
> **摘要:** Large language models (LLMs) are increasingly being used for tasks where outputs shape human decisions, so it is critical to verify that their responses consistently reflect desired human values. Humans, as individuals or groups, don't agree on a universal set of values, which makes evaluating value alignment difficult. Existing benchmarks often use hypothetical or commonsensical situations, which don't capture the complexity and ambiguity of real-life debates. We introduce the Value ALignment Benchmark (VAL-Bench), which measures the consistency in language model belief expressions in response to real-life value-laden prompts. VAL-Bench consists of 115K pairs of prompts designed to elicit opposing stances on a controversial issue, extracted from Wikipedia. We use an LLM-as-a-judge, validated against human annotations, to evaluate if the pair of responses consistently expresses either a neutral or a specific stance on the issue. Applied across leading open- and closed-source models, the benchmark shows considerable variation in consistency rates (ranging from ~10% to ~80%), with Claude models the only ones to achieve high levels of consistency. Lack of consistency in this manner risks epistemic harm by making user beliefs dependent on how questions are framed rather than on underlying evidence, and undermines LLM reliability in trust-critical applications. Therefore, we stress the importance of research towards training belief consistency in modern LLMs. By providing a scalable, reproducible benchmark, VAL-Bench enables systematic measurement of necessary conditions for value alignment.
>
---
#### [replaced 055] Beg to Differ: Understanding Reasoning-Answer Misalignment Across Languages
- **分类: cs.CL**

- **简介: 该论文属于多语言模型评估任务，旨在解决模型推理与答案不一致的问题。通过分析65k个跨语言的推理轨迹，发现非拉丁语系模型推理与结论存在更大偏差，并提出错误分类体系。**

- **链接: [https://arxiv.org/pdf/2512.22712v2](https://arxiv.org/pdf/2512.22712v2)**

> **作者:** Anaelia Ovalle; Candace Ross; Sebastian Ruder; Adina Williams; Karen Ullrich; Mark Ibrahim; Levent Sagun
>
> **备注:** Accepted to 2025 EMNLP Multilingual Representation Learning Workshop
>
> **摘要:** Large language models demonstrate strong reasoning capabilities through chain-of-thought prompting, but whether this reasoning quality transfers across languages remains underexplored. We introduce a human-validated framework to evaluate whether model-generated reasoning traces logically support their conclusions across languages. Analyzing 65k reasoning traces from GlobalMMLU questions across 6 languages and 6 frontier models, we uncover a critical blind spot: while models achieve high task accuracy, their reasoning can fail to support their conclusions. Reasoning traces in non-Latin scripts show at least twice as much misalignment between their reasoning and conclusions than those in Latin scripts. We develop an error taxonomy through human annotation to characterize these failures, finding they stem primarily from evidential errors (unsupported claims, ambiguous facts) followed by illogical reasoning steps. Our findings demonstrate that current multilingual evaluation practices provide an incomplete picture of model reasoning capabilities and highlight the need for reasoning-aware evaluation frameworks.
>
---
