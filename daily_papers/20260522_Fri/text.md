# 自然语言处理 cs.CL

- **最新发布 103 篇**

- **更新 60 篇**

## 最新发布

#### [new 001] Does Slightly Mean Somewhat? Measuring Vague Intensity Words in LLM Numeric Actions
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型对模糊强度词的数值映射，探讨其在资源分配任务中的表现。工作包括构建实验、分析模型输出模式，揭示强度词的压缩性与状态依赖性。**

- **链接: [https://arxiv.org/pdf/2605.21827](https://arxiv.org/pdf/2605.21827)**

> **作者:** Daniel Tabach
>
> **备注:** 9 figures, 2 tables, 16 references
>
> **摘要:** Do language models preserve the ordinal meaning of intensity words when those words must produce numeric actions? I study a researcher-constructed scale of 10 English degree modifiers, from slightly to drastically, informed by the Quirk et al. degree-modifier taxonomy, in a controlled resource-allocation environment where Claude Haiku receives a natural-language instruction, produces a numeric allocation, and a deterministic backend converts that allocation into a measurable outcome. The only variable that changes between runs is the intensity word or the starting system state, isolating their effects on the model's numeric output. Across 6,620 runs at T=0.0 and T=0.7, three patterns emerge. First, the model compresses 10 intensity words into 5 distinct median outputs: four lower-tier words all map to the same value, while stronger words break into higher regimes (Spearman rho = 0.845, p < 0.001). Second, when the current system state is supplied as context, separate Kruskal-Wallis tests show that grouping by starting allocation captures far more rank-based variance than grouping by word (epsilon-squared baseline = 0.782 vs. epsilon-squared word = 0.079), and lexical differentiation collapses to zero as the system approaches capacity. Third, near feasibility limits the model exhibits three behavioral modes: weak words hedge with small adjustments, strong words abstain entirely, and the word drastically pushes to the local ceiling. These patterns persist across temperature, with stochastic sampling broadening distributions but not restoring ordinal distinctions between words. In this model and domain, the model's numeric interpretation of vague intensity words is compressed, state-dependent, and discontinuous near operational boundaries.
>
---
#### [new 002] Evaluation of Chunking Strategies for Effective Text Embedding in Low-Resource Language on Agricultural Documents
- **分类: cs.CL**

- **简介: 该论文属于文本嵌入任务，旨在优化低资源语言农业文档的检索效果。通过比较四种分块策略，提升信息检索的准确性与相关性。**

- **链接: [https://arxiv.org/pdf/2605.22203](https://arxiv.org/pdf/2605.22203)**

> **作者:** Sovandara Chhoun; Pichdara Po; Sereiwathna Ros; Wan-Sup Cho; Saksonita Khoeurn
>
> **备注:** 11 pages, 1 figure
>
> **摘要:** In this study, we compare the performance of four text chunking approaches: Recursive, Khmer-Aware, Sentence-Based, and LLM-Based within a Retrieval-Augmented Generation (RAG) framework applied to Khmer agricultural documents. The document chunks are encoded using the BGE-M3 multilingual embedding model and retrieved using the FAISS library. Performance is evaluated using four metrics: Average Retrieval Score (L2 distance), Answer Relevance, Khmer Coverage, and Khmer Intersection over Union, all measured against ground-truth question-answer pairs. For evaluation, we perform 5-fold cross-validation over 18 question-answer pairs. We observe the best performance for the character-based Recursive chunking method with a chunk size of 300 characters, achieving the lowest L2 distance (0.4295 +- 0.0461), highest Answer Relevance (0.8663 +- 0.0199), and highest Khmer IoU (0.6441 +- 0.0347). A paired t-test shows a statistically significant improvement over the Sentence-Based chunking method in L2 distance (p = 0.0121). These results highlight the importance of segmentation granularity and structural preservation for optimizing dense retrieval in morphologically complex, low-resource languages such as Khmer.
>
---
#### [new 003] TransitLM: A Large-Scale Dataset and Benchmark for Map-Free Transit Route Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TransitLM，一个大规模数据集和基准，用于无地图的公交路线生成任务。解决传统依赖地图和路由引擎的问题，通过数据训练模型直接生成路线。**

- **链接: [https://arxiv.org/pdf/2605.22355](https://arxiv.org/pdf/2605.22355)**

> **作者:** Hanyu Guo; Jiedong Yang; Chao Chen; Longfei Xu; Kaikui Liu; Xiangxiang Chu
>
> **摘要:** Public transit route planning traditionally depends on structured map infrastructure and complex routing engines, and no existing dataset supports training models to bypass this dependency. We present TransitLM, a large-scale dataset of over 13 million transit route planning records from four Chinese cities covering 120,845 stations and 13,666 lines, released as a continual pre-training corpus and benchmark data for three evaluation tasks with complementary metrics. Experiments show that an LLM trained on TransitLM produces structurally valid routes at high accuracy and implicitly grounds arbitrary GPS coordinates to appropriate stations without any explicit mapping. These results demonstrate that transit route planning can be learned entirely from data, enabling end-to-end, map-free route generation directly from origin-destination information. The dataset and benchmark are available at this https URL, with evaluation code at this https URL.
>
---
#### [new 004] FlyRoute: Self-Evolving Agent Profiling via Data Flywheel for Adaptive Task Routing
- **分类: cs.CL**

- **简介: 该论文提出FlyRoute，解决企业路由中代理配置静态化问题，通过数据飞轮自适应更新代理能力描述，提升任务路由准确性。**

- **链接: [https://arxiv.org/pdf/2605.22057](https://arxiv.org/pdf/2605.22057)**

> **作者:** Rongjun Li; Ziyu Zhou; Yihang Wu
>
> **备注:** 13 pages, 5 figures, 5 tables
>
> **摘要:** Enterprise routers assign queries to expert agents, yet deployed profiles stay static while agents evolve (prompts, tools, models), and developers rarely keep descriptions or exemplars current. We present FlyRoute, a self-evolving profiling framework that grows capability evidence from real traffic: dispatch candidates, quality-gate successful pairs into each agent's success store, periodically distill evidence into learned capability descriptions, and inject those descriptions together with BM25-retrieved successes into an LLM router. To make this flywheel data-efficient, FlyRoute introduces a targeted exploration policy that combines profile uncertainty, BM25 relevance, and lexical novelty, prioritizing under-profiled agents only for plausible queries and avoiding redundant evidence collection. In experiments on our proprietary enterprise developer-support dataset of real routed queries, FlyRoute improves a same-backbone zero-shot LLM router from 72.57% to 78.04% with only five seed queries per agent, showing that profile retrieval already strengthens cold-start routing. After streaming 7,211 labeled training queries through the flywheel, accuracy rises to 89.83% (+17.26pp over zero-shot; +11.79pp over cold start), with consistent gains across four expert domains under standard routing accuracy on single-gold test queries.
>
---
#### [new 005] Multi-Stage Training for Abusive Comment Detection in Indic Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于有害评论检测任务，旨在准确识别印地语等语言中的不当内容，降低误报率，保护言论自由。**

- **链接: [https://arxiv.org/pdf/2605.22380](https://arxiv.org/pdf/2605.22380)**

> **作者:** Pranshu Rastogi; Madhav Mathur; Ramaneswaran S; Kshitij Mohan
>
> **备注:** 4 pages, EAM2021 selected
>
> **摘要:** In recent years social media has become an increasingly popular tool for communication. People use it to share their ideas, exchange information, and discuss thoughts. Given its prevalence and widespread reach, social media must remain a safe space for people. Content generated on social media can be abusive and it has become increasingly important to detect such content. In this paper, we use a language-based preprocessing and an ensemble of several models and analyze their performance of abusive comment detection. Through extensive experimentation, we propose a pipeline that minimizes the false-positive rate (marking non-abusive as abusive) so that these systems can detect abusive comments without undermining the freedom of expression.
>
---
#### [new 006] IdioLink: Retrieving Meaning Beyond Words Across Idiomatic and Literal Expressions
- **分类: cs.CL**

- **简介: 该论文属于语义检索任务，旨在解决模型难以跨不同表达形式（如习语与字面意义）找到等价语义的问题。研究构建了IdioLink基准，评估模型在该任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.22247](https://arxiv.org/pdf/2605.22247)**

> **作者:** Kai Golan Hashiloni; Daniel Fadlon; Lior Livyatan; Ofri Hefetz; Jiahuan Pei; Kfir Bar
>
> **摘要:** Idioms pose a fundamental challenge for language models, as their meaning cannot be inferred from surface form alone. Understanding such expressions, therefore, requires semantic abstraction beyond lexical overlap. We introduce IdioLink, a retrieval benchmark designed to test whether models can link idiomatic expressions to conceptually equivalent meanings expressed in literal or paraphrased forms. IdioLink comprises 10,700 documents and 2,140 queries, spanning 107 idioms with both literal and figurative uses. Each document and query is annotated with spans that convey the core meaning. Evaluating strong embedding baselines (e.g., BGE, E5, Contriever, and Qwen), we show that current models struggle to retrieve equivalent meanings across divergent surface realizations, relying instead on topical and shallow semantic cues. IdioLink exposes key gaps in idiom-aware semantic retrieval and provides a challenging testbed for future models.
>
---
#### [new 007] Cross-Lingual Consensus: Aligning Multilingual Cultural Knowledge via Multilingual Self-Consistency
- **分类: cs.CL**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决LLMs在跨语言文化知识对齐上的偏差问题。通过自监督框架提升模型跨语言的文化一致性。**

- **链接: [https://arxiv.org/pdf/2605.22137](https://arxiv.org/pdf/2605.22137)**

> **作者:** Andrew Ivan Soegeng; Patrick Sutanto; Tan Sang Nguyen
>
> **备注:** Accepted to The 1st Workshop on Multilinguality in the Era of Large Language Models
>
> **摘要:** Although Large Language Models (LLMs) demonstrate strong capabilities across various tasks, they exhibit significant performance discrepancies across languages. While prompting LLMs in English typically yields the highest general performance, it often induces a Western-centric bias, hindering the model's ability to accurately reflect diverse cultural knowledge. We hypothesize that LLMs already possess rich cultural knowledge embedded within local-language representations, but fail to retrieve it when prompted in English. To bridge this cross-lingual knowledge gap, we propose a novel self-supervised framework. Our method leverages multilingual self-consistency to identify the most reliable cultural responses across languages, combined with a self-critique mechanism to transfer this knowledge to the weaker language. Evaluations on the BLEnD benchmark demonstrate that our approach significantly improves cultural alignment-boosting performance on English queries by an average of 5.03%-relying entirely on self-generated data. Ultimately, our work demonstrates that latent cultural knowledge can be successfully surfaced and propagated across languages, enabling more culturally equitable and consistent LLMs.
>
---
#### [new 008] CR4T: Rewrite-Based Guardrails for Adolescent LLM Safety
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI安全任务，旨在解决青少年使用LLM时的安全问题。提出CR4T框架，通过改写不安全内容，提供适合青少年的指导性回应，提升对话安全性与有效性。**

- **链接: [https://arxiv.org/pdf/2605.21609](https://arxiv.org/pdf/2605.21609)**

> **作者:** Heajun An; Qi Zhang; Vedanth Achanta; Jin-Hee Cho
>
> **摘要:** Large language models (LLMs) are increasingly embedded in adolescent digital environments, mediating information seeking, advice, and emotionally sensitive interactions. Yet existing safety mechanisms remain largely grounded in adult-centric norms and operationalize safety through refusal-oriented suppression. While such approaches may reduce immediate policy violations, they can also create conversational dead-ends, limit constructive guidance, and fail to address the developmental vulnerabilities inherent in adolescent-AI interactions. We argue that adolescent LLM safety should be framed not solely as a filtering problem, but as a socio-technical, developmentally aligned transformation problem. To operationalize this perspective, we propose Critique-and-Revise-for-Teenagers (CR4T), a model-agnostic safeguarding framework that selectively reconstructs unsafe or refusal-style outputs into ageappropriate, guidance-oriented responses while preserving benign intent. CR4T combines lightweight risk detection with domain-conditioned rewriting to remove risk-amplifying content, reduce unnecessary conversational shutdown, and introduce developmentally appropriate guidance. Experimental results show that targeted rewriting substantially reduces unsafe and refusal-oriented outcomes while avoiding unnecessary intervention on acceptable interactions. These findings suggest that selective response reconstruction offers a more human-centered alternative to refusal-centric guardrails for adolescent-facing LLM systems.
>
---
#### [new 009] More Context, Larger Models, or Moral Knowledge? A Systematic Study of Schwartz Value Detection in Political Texts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于情感分析任务，研究如何在政治文本中检测Schwartz价值观。通过对比不同上下文、知识增强和模型规模，探讨提升价值检测效果的方法。**

- **链接: [https://arxiv.org/pdf/2605.22641](https://arxiv.org/pdf/2605.22641)**

> **作者:** Víctor Yeste; Paolo Rosso
>
> **备注:** Code: this https URL, best model: this https URL, 18 pages, 3 figures
>
> **摘要:** Detecting Schwartz values in political text is difficult because implicit cues often depend on surrounding arguments and fine-grained distinctions between neighboring values. We study when context and explicit moral knowledge help sentence-level value detection. Using the ValuesML/Touch{é} ValueEval format, we compare sentence, window, and full-document inputs; no-RAG and retrieval-augmented settings with a curated moral knowledge base; supervised DeBERTa-v3-base/large encoders; and zero-shot LLMs from 12B to 123B parameters. The results show that more context is not uniformly better: full-document context improves supervised DeBERTa encoders by 3.8--4.8 macro-F1 points over sentence-only input, but does not consistently help zero-shot LLMs. Retrieved moral knowledge is more consistently useful in matched comparisons, improving each tested model family and context condition under early fusion. However, scaling from DeBERTa-v3-base to large and from 12B to larger LLMs does not guarantee gains, and simple early fusion outperforms the tested late-fusion and cross-attention RAG variants for encoders. Per-value analyses show that context and retrieval help most for socially situated or conceptually confusable values. These findings suggest that value-sensitive NLP should evaluate context, knowledge, and model family jointly rather than treating longer inputs or larger models as universal improvements.
>
---
#### [new 010] Sem-Detect: Semantic Level Detection of AI Generated Peer-Reviews
- **分类: cs.CL**

- **简介: 该论文属于作者身份检测任务，旨在区分AI生成与人类撰写的同行评审。通过结合文本特征和语义分析，提出Sem-Detect方法，有效识别AI生成的评审。**

- **链接: [https://arxiv.org/pdf/2605.21713](https://arxiv.org/pdf/2605.21713)**

> **作者:** André V. Duarte; Brian Tufts; Aditya Oke; Fei Fang; Arlindo L. Oliveira; Lei Li
>
> **摘要:** How can we distinguish whether a peer review was written by a human or generated by an AI model? We argue that, in this setting, authorship should not be attributed solely from the textual features of a review, but also from the ideas, judgments, and claims it expresses. To this end, we propose Sem-Detect, an authorship detection method for peer reviews that operationalizes this principle by combining textual features with claim-level semantic analysis. Sem-Detect compares a target review against multiple AI-generated reviews of the same paper, leveraging the observation that different AI models tend to converge on similar points, while human reviewers introduce more unique and diverse ones. As a result, Sem-Detect is able to distinguish fully AI reviews from authentic human-written ones, including those that have been refined using an LLM but still reflect human judgment. Across a dataset of over 20,000 peer reviews from ICLR and NeurIPS conferences, Sem-Detect improves over the strongest baseline by 25.5% in TPR@0.1% FPR in the binary setting. Moreover, in the three-class scenario, we empirically show that LLM refinement preserves the semantic signals of human reviews, which remain distinct from the patterns exhibited by fully AI-generated text; as a result, fewer than 3.5% of LLM-refined human reviews are misclassified as AI-generated.
>
---
#### [new 011] BeLink: Biomedical Entity Linking Meets Generative Re-Ranking
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于生物医学实体链接任务，旨在解决LLM在BEL中计算效率低的问题。通过生成式重排序技术提升链接准确率并减少推理时间。**

- **链接: [https://arxiv.org/pdf/2605.22501](https://arxiv.org/pdf/2605.22501)**

> **作者:** Darya Shlyk; Stefano Montanelli; Lawrence Hunter
>
> **备注:** Accepted to ACM SIGIR 2026
>
> **摘要:** Despite recent progress, Biomedical Entity Linking (BEL) with large language models (LLMs) remains computationally inefficient and challenging to deploy in practical settings. In this work, we demonstrate that instruction-tuning of open-source generative models can offer an effective solution when applied at the re-ranking stage of the BEL pipeline. We propose a set-wise instruction-tuning formulation that enables fast and accurate candidate selection. Our method demonstrates strong performance on multiple BEL benchmarks, yielding significant improvements in linking accuracy (3%-24%) while reducing inference time compared to the state-of-the-art. We integrate our generative re-ranker into BeLink, a modular, end-to-end system designed for practical real-world BEL applications.
>
---
#### [new 012] ACC: Compiling Agent Trajectories for Long-Context Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ACC方法，将智能体轨迹转化为长上下文问答对，解决长距离依赖建模问题，提升模型的长文本推理能力。**

- **链接: [https://arxiv.org/pdf/2605.21850](https://arxiv.org/pdf/2605.21850)**

> **作者:** Qisheng Su; Zhen Fang; Shiting Huang; Yu Zeng; Yiming Zhao; Kou Shi; Ziao Zhang; Lin Chen; Zehui Chen; Lijun Wu; Feng Zhao
>
> **摘要:** Recent development of agents has renewed demand for long-context reasoning capacity of LLMs. However, training LLMs for this capacity requires costly long-document curation or heuristic context synthesis. We observe that agents produce massive trajectories when solving problems, invoking tools and receiving environment observations across many turns. The evidence needed to answer the original question is thus scattered throughout these turns, requiring integration of distant context segments. Nevertheless, standard agent SFT masks tool responses and only trains turn-level tool selection, creating a supervision blind spot where these scattered signals go unused. We propose Agent Context Compilation (ACC), which converts trajectories from search, software engineering, and database querying agents into long-context QA pairs that combine the original question with tool responses and environment observations gathered across multiple turns, training the model to answer directly without tool use. This makes the dependencies between the question and the evidence explicit, enabling direct supervision of long-context reasoning over distant segments without additional annotation. ACC is a simple but effective approach that can be combined with any existing long-context extension or training method, providing scalable supervised fine-tuning data. We validate ACC on long-range dependency modeling tasks through MRCR and GraphWalks, challenging benchmarks requiring cross-turn coreference resolution and graph traversal over extended contexts. Training Qwen3-30B-A3B with ACC achieves 68.3 on MRCR (+18.1) and 77.5 on GraphWalks (+7.6), results comparable to Qwen3-235B-A22B, while preserving general capabilities on GPQA, MMLU-Pro, AIME, and IFEval. Further mechanism analysis reveals that the ACC-trained model exhibits task-adaptive attention restructuring and expert specialization.
>
---
#### [new 013] Understanding Data Temporality Impact on Large Language Models Pre-training
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究数据时间性对大语言模型预训练的影响，旨在解决模型时间知识获取问题。通过构建时间基准测试集和对比有序与随机预训练方法，验证了有序数据提升事实新鲜度的效果。**

- **链接: [https://arxiv.org/pdf/2605.22769](https://arxiv.org/pdf/2605.22769)**

> **作者:** Pilchen Hippolyte; Fabre Romain; Signe Talla Franck; Perez Patrick; Grave Edouard
>
> **摘要:** Large language models (LLMs) are typically trained on shuffled corpora, yielding models whose knowledge is frozen at train time and whose temporal grounding remains poorly understood. In this work, we study the impact of pre-training dynamics on the acquisition of time-sensitive factual knowledge, focusing specifically on data ordering. Our main contributions are twofold. First, we introduce a comprehensive benchmark of over 7,000 temporally grounded questions and an evaluation protocol that enables analysis of whether models correctly associate facts with their corresponding time periods. Second, we pretrain 6B-parameter models on temporally ordered Common Crawl snapshots and compare them against standard shuffled pre-training. Our results show that sequentially trained models match shuffled baselines on general language understanding and common knowledge while consistently exhibiting more up-to-date and temporally precise knowledge. Temporally ordered pre-training yields improved factual freshness, while shuffled pre-training peaks on older data, possibly due to increased factual repetition. These findings, along with the release of our code at this https URL , checkpoints, and datasets at this https URL provide a foundation for future research on continual learning for LLMs.
>
---
#### [new 014] Boiling the Frog: A Multi-Turn Benchmark for Agentic Safety
- **分类: cs.CL**

- **简介: 该论文属于AI安全任务，旨在评估AI代理在办公环境中的安全性。通过构建基准测试，检测模型对渐进式攻击的敏感性，解决AI系统在实际应用中的安全风险问题。**

- **链接: [https://arxiv.org/pdf/2605.22643](https://arxiv.org/pdf/2605.22643)**

> **作者:** Piercosma Bisconti; Matteo Prandi; Federico Pierucci; Federico Sartore; Enrico Panai; Laura Caroli; Yue Zhu; Adam Leon Smith; Luca Nannini; Marcello Galisai; Susanna Cifani; Francesco Giarrusso; Marcantonio Bracale Syrnikov; Daniele Nardi
>
> **摘要:** Background. Traditional safety benchmarks for language models evaluate generated text: whether a model outputs toxic language, reproduces bias, or follows harmful instructions. When models are deployed as agents, the safety-relevant object shifts from what the system says to what it does within an environment, and evaluating model responses under prompting is no longer sufficient to address the safety challenges posed by artificial intelligence. Recent developments have seen the rise of benchmarks that evaluate large language models as agents. We contribute to this strand of research. Approach. We introduce Boiling the Frog, a benchmark that evaluates whether tool-using AI models deployed in corporate and office settings are susceptible to incremental attacks. Each scenario begins with benign workspace edits and later introduces a risk-bearing request. The benchmark focuses on stateful multi-turn evaluation: chains expose a persistent workspace, place the risk-bearing payload at controlled positions in the turn sequence, and score whether the resulting artifact state becomes unsafe. Scenarios are organized through a three-level operational risk taxonomy grounded in the Boiling the Frog risks, the AI Act Annex I and Annex III high-risk contexts, and EU AI Act's Code of Practice on General-Purpose AI (GPAI). Results. Across a nine-model panel, aggregate strict attack success rate (ASR) is 44.4%. Model-level ASR ranges from 20.5% for Claude Haiku 4.5 to 92.9% for Gemini 3.1 Flash Lite, with Seed 2.0 Lite also above 80%. Average chain category-level ASR reaches 93.3% for Code of Practice loss-of-control scenarios.
>
---
#### [new 015] Hypergraph as Language
- **分类: cs.CL**

- **简介: 该论文提出"超图即语言"观点，解决传统方法对高阶关系建模不足的问题。通过Hyper-Align框架，将超图结构转化为可被大语言模型理解的token，提升复杂结构建模能力。**

- **链接: [https://arxiv.org/pdf/2605.21858](https://arxiv.org/pdf/2605.21858)**

> **作者:** Mengqi Lei; Guohuan Xie; Shihui Ying; Shaoyi Du; Jun-Hai Yong; Siqi Li; Yue Gao
>
> **摘要:** Large language models (LLMs) have recently shown strong potential in modeling relational structures. However, existing approaches remain fundamentally graph-centric: they focus on processing pairwise graph structures into tokens that LLMs can understand. In contrast, many real-world relational patterns do not naturally conform to the pairwise-edge assumption, and are better modeled as high-order associations in hypergraphs. For hypergraph structures, existing methods often fail to preserve the native semantics that multiple objects are jointly connected by the same high-order relation, limiting their ability to exploit complex structures. To address this limitation, we put forth the "Hypergraph as Language" perspective and propose Hyper-Align, a hypergraph-native alignment framework for large language models. Hyper-Align compiles the query-object-centered hypergraph context into hypergraph tokens directly consumable by a base LLM. Specifically, we introduce Hypergraph Incidence Detail Template with Overview (HIDT-O), which serializes high-order association structures into a fixed-shape hybrid template combining local incidence details and overview-level summaries. We then design a Hypergraph Incidence Projector (HIP), which maps native high-order incidence structures into the LLM token space through explicit semantic-structural decoupling and bidirectional message passing between vertices and hyperedges. We further define a concrete Hypergraph-as-Language input protocol, which jointly feeds hypergraph tokens and textual prompts into a frozen base LLM, supporting both vertex-level and hyperedge-level tasks under a unified question-answering paradigm. To systematically evaluate different methods in hypergraph structural modeling, we introduce HyperAlign-Bench. Extensive experiments show that Hyper-Align significantly outperforms existing methods across in-domain and zero-shot evaluations.
>
---
#### [new 016] Comparing LLM and Fine-Tuned Model Performance on NVDRS Circumstance Extraction with Varying Prompt Complexity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升自杀死亡调查文本中复杂情境的提取效果。通过比较LLM与微调模型在不同提示复杂度下的表现，提出混合方法以优化罕见和复杂情况的处理。**

- **链接: [https://arxiv.org/pdf/2605.21845](https://arxiv.org/pdf/2605.21845)**

> **作者:** Geoffrey Martin; Xuan Zhong Feng; Yifan Peng
>
> **备注:** Accepted at IEEE ICHI 2026
>
> **摘要:** Suicide is a leading cause of death in the United States, and understanding the circumstances that precede it requires extracting structured information from death investigation narratives. Many of these circumstances require semantic inference beyond simple keyword matching. We develop a ``Complexity Score'' algorithm that analyzes coding manual structure to predict when detailed prompts with full coding guidelines improve over name-only prompts. We then construct a hybrid approach that selects prompt strategy per circumstance. We evaluate large language models (LLMs) against fine-tuned RoBERTa on 25 inferentially complex circumstances from the National Violent Death Reporting System (NVDRS). We found that LLMs substantially outperform on low-prevalence circumstances where training data is insufficient. We further demonstrate that our framework generalizes across frontier LLMs, with GPT-5.2, Gemini 2.5 Pro and Llama-3 70B showing consistent performance patterns. These findings support a hybrid architecture where LLMs handle rare, inferentially complex circumstances while fine-tuned models handle common ones.
>
---
#### [new 017] LANG: Reinforcement Learning for Multilingual Reasoning with Language-Adaptive Hint Guidance
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，解决LLM在多语言环境下推理质量与语言一致性之间的矛盾。提出LANG框架，通过语言引导提示提升推理性能并保持语言一致性。**

- **链接: [https://arxiv.org/pdf/2605.22567](https://arxiv.org/pdf/2605.22567)**

> **作者:** Yuchun Fan; Bei Li; Peiguang Li; Yilin Wang; Yongyu Mu; Jian Yang; Xin Chen; Rongxiang Weng; Jingang Wang; Xunliang Cai; Jingbo Zhu; Tong Xiao
>
> **备注:** Accepted to ACL 2026 (main conference)
>
> **摘要:** Reinforcement learning has proven effective for enhancing multi-step reasoning in large language models (LLMs), yet its benefits have not fully translated to multilingual contexts. Existing methods struggle with a fundamental trade-off: prioritizing input-language consistency severely hampers reasoning quality, while prioritizing reasoning often leads to unintended language drift toward English. We address this challenge with LANG, a novel framework that leverages language-conditioned hints to guide exploration in non-English reasoning tasks. Our method incorporates two key mechanisms to prevent dependency on these hints: a progressive decay schedule that gradually withdraws scaffolding, and a language-adaptive switch that tailors learning horizons to specific language difficulties. Empirical results on challenging multilingual mathematical benchmarks reveal that LANG substantially enhances reasoning performance without compromising language consistency. Moreover, we show that our framework generalizes beyond mathematics, fostering more consistent language alignment across model layers
>
---
#### [new 018] Diagnosis Is Not Prescription: Linguistic Co-Adaptation Explains Patching Hazards in LLM Pipelines
- **分类: cs.CL**

- **简介: 该论文属于人工智能领域，研究多模块大语言模型系统故障修复问题。提出诊断悖论现象，指出修复错误模块未必有效，而上游模块修复更可靠，并通过实验证明语言契约假设。**

- **链接: [https://arxiv.org/pdf/2605.21958](https://arxiv.org/pdf/2605.21958)**

> **作者:** Yoon Jeonghun; Kim Dongchan
>
> **备注:** Preprint. Under review at EMNLP 2026 (ARR)
>
> **摘要:** When a multi-module LLM agent fails, the module most responsible for the failure is not necessarily the best place to intervene. We demonstrate this Diagnostic Paradox empirically: causal analysis consistently identifies the routing module -- which selects which tool to call next -- as the primary bottleneck across three independent agent families. Yet injecting prompt-level correction examples into this module consistently degrades performance, sometimes severely. Patching an upstream query-rewriting module instead reliably improves outcomes. The effect holds with statistical significance on two agent families and directional consistency on a third; alternative repair strategies at the routing module (instruction rewriting, model upgrade) are neutral, confirming that the harm is specific to correction-injection patching. We explain this asymmetry through the Linguistic Contract hypothesis: each downstream module implicitly adapts to its upstream's characteristic error distribution, so correcting the bottleneck breaks this implicit alignment in a way that upstream corrections do not. We operationalize this via a per-agent co-adaptation measure, derived from diagnosis alone, and show it is consistently associated with patching harm across agent families: higher co-adaptation co-occurs with harm, lower with safety. This trend holds across all three agent families, providing preliminary support for the hypothesis beyond a single-agent observation.
>
---
#### [new 019] From TF-IDF to Transformers: A Comparative and Ensemble Approach to Sentiment Classification
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文属于情感分类任务，旨在准确识别电影评论的正面或负面情绪。通过比较多种模型，发现RoBERTa表现最佳，并验证了集成方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.22003](https://arxiv.org/pdf/2605.22003)**

> **作者:** Dip Biswas Shanto; Mitali Yadav; Prajwal Panth; Suresh Chandra Satapathy
>
> **备注:** 6 pages, 9 figures. This is the author's accepted manuscript, presented at the International Conference on Intelligent Computing, Networks and Security (IC-ICNS 2026), March 26-28, Bhubaneswar, India. Proceedings publication pending
>
> **摘要:** Sentiment analysis, also referred to as opinion mining, primarily tries to extract opinion from any text-based data. In the context of movie reviews and critics, sentimental analysis can be a helpful tool to predict whether a movie review is generally positive or negative. It can be difficult for the ML models to understand the context or metaphysical sentiment accurately, as ML models rely largely on statistical word representations. The objective of this paper is to examine and categorise movie reviews into positive and negative sentiments. Diverse machine learning models are considered in doing so, and Natural Language Processing (NLP) methodologies are employed for data preprocessing and model assessment. The IMDb dataset is used. Specifically, Naive Bayes, Logistic Regression, Support Vector Machines (SVM), LightGBM, LSTM, and transformer-based models such as RoBERTa and DistilBERT were evaluated. After a lot of testing with accuracy, precision, recall, F1-score, and ROC-AUC, RoBERTa performed better than all the other models, with an accuracy of 93.02%. A soft voting ensemble that combined all the models also improved classification performance, showing that model ensembling works well for sentiment analysis.
>
---
#### [new 020] Reflective Prompt Tuning through Language Model Function-Calling
- **分类: cs.CL**

- **简介: 该论文提出RPT框架，解决提示工程中手动设计耗时且易错的问题。通过语言模型函数调用模拟人类迭代优化过程，提升推理任务性能与置信度校准。**

- **链接: [https://arxiv.org/pdf/2605.21781](https://arxiv.org/pdf/2605.21781)**

> **作者:** Farima Fatahi Bayat; Moin Aminnaseri; Pouya Pezeshkpour; Estevam Hruschka
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Large language models (LLMs) have become increasingly capable of following instructions and complex reasoning, making prompting a flexible interface for adapting models without parameter updates. Yet prompt design remains labor-intensive and highly sensitive to formatting, phrasing, and instruction order, motivating automated prompt optimization methods that reduce manual effort while preserving inference-time flexibility. However, existing methods often search over prompt candidates or use fixed critique-refine pipelines driven by individual examples or small batches, limiting their ability to capture systematic error patterns and make targeted edits grounded in failure history. We propose Reflective Prompt Tuning (RPT), a framework that uses LLM function calling to simulate the iterative workflow of human prompt engineers. An LLM optimizer calls a diagnostic function that evaluates the target model over an entire optimization set, summarizes recurring failure modes, and returns a structured diagnostic report. The optimizer uses this report, together with an accumulated memory of prior reports, to revise the prompt for the next iteration. RPT further supports confidence-aware optimization by using calibration signals in diagnostic feedback and final prompt selection. Across three reasoning tasks, RPT improves over initial prompts by up to 12.9 points, remains competitive with state of the art, and improves confidence calibration. Our analyses show that RPT is especially effective on multi-hop and mathematical reasoning, producing targeted prompt revisions that align with diagnosed failure patterns and lead to gains in task performance and calibration.
>
---
#### [new 021] ArabDiscrim: A Decade-Long Arabic Facebook Corpus on Racism and Discrimination
- **分类: cs.CL**

- **简介: 该论文介绍ArabDiscrim，一个包含29.3万条阿拉伯语Facebook帖子的语料库，用于研究种族主义和歧视。属于自然语言处理中的公平性与平台生态研究任务，旨在提供多维度的语料支持。**

- **链接: [https://arxiv.org/pdf/2605.22081](https://arxiv.org/pdf/2605.22081)**

> **作者:** Wajdi Zaghouani; Shimaa Amer Ibrahim; Mabrouka Bessghaier; Houda Bouamor
>
> **备注:** Accepted at LREC 2026 Main Conference
>
> **摘要:** We present ArabDiscrim, a decade-long lexical resource and corpus of 293K public Arabic Facebook posts (2014--2024) discussing racism and discrimination. Unlike existing Twitter-centric datasets, ArabDiscrim integrates platform-native engagement signals, including reactions, shares, comments, and page metadata, enabling joint analysis of language and audience response. The resource includes 200 curated terms (100 racism-related and 100 discrimination-related) with morphological regex families (13+ inflections per lemma), and 20 discrimination axes capturing identity-based grounds for unequal treatment. It also provides explicit attribution patterns. Released under a restricted research-use license for ethical compliance with platform terms, ArabDiscrim supports weak supervision, axis-aware sampling, and platform ecology research. By bridging lexical depth and ecological validity, it establishes a foundation for fairness-oriented, platform-aware Arabic NLP.
>
---
#### [new 022] When Cases Get Rare: A Retrieval Benchmark for Off-Guideline Clinical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，旨在解决罕见临床场景下模型依赖记忆而非证据的问题。提出OGCaReBench基准，评估模型在非指南情况下的推理能力。**

- **链接: [https://arxiv.org/pdf/2605.21807](https://arxiv.org/pdf/2605.21807)**

> **作者:** Doeun Lee; Muge Zhang; Yi Yu; Ashish Manne; Stephen Koesters; Frank Wen; Brady Buchanan; Lynda Villagomez; Oluwatoba Moninuola; James Lim; Kathryn Tobin; Andrew Srisuwananukorn; Ping Zhang; Sachin Kumar
>
> **备注:** 34 pages, 20 figures
>
> **摘要:** Across medical specialties, clinical practice is anchored in evidence-based guidelines that codify best studied diagnostic and treatment pathways. These pathways routinely fall short for the long tail of real-world care not covered by guidelines. Most medical large language models (LLMs), however, are trained to encode common, guideline-focused medical knowledge in their parameters. Current evaluations test models primarily on recalling and reasoning with this memorized content, often in multiple-choice settings. Given the fundamental importance of evidence-based reasoning in medicine, it is neither feasible nor reliable to depend on memorization in practice. To address this gap, we introduce OGCaReBench, a free-form retrieval-focused benchmark aimed at evaluating LLMs at answering clinical questions that require going beyond typical guidelines. Extracted from published medical case reports and validated by medical experts, OGCaReBench contains long-form clinical questions requiring free-text answers, providing a systematic framework for assessing open-ended medical reasoning in rare, case-based scenarios. Our experiments reveal that even the best-performing baseline (GPT-5.2) correctly answers only 56% of our benchmark with specialized models only reaching 42%. Augmenting models with retrieved medical articles improves this performance to up to 82% (using GPT-5.2) highlighting the importance of evidence-grounding for real-world medical reasoning tasks. This work thus establishes a foundation for benchmarking and advancing both general-purpose and medical LLMs to produce reliable answers in challenging clinical contexts.
>
---
#### [new 023] Residual Skill Optimization for Text-to-SQL Ensembles
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 该论文属于文本到SQL任务，解决集成模型中候选SQL相关性高的问题。提出DivSkill-SQL框架，通过优化互补技能提升准确率。**

- **链接: [https://arxiv.org/pdf/2605.21792](https://arxiv.org/pdf/2605.21792)**

> **作者:** Jiongli Zhu; Haoquan Guan; Parjanya Prajakta Prashant; Nikki Lijing Kuang; Seyedeh Baharan Khatami; Canwen Xu; Xiaodong Yu; Yingyu Lin; Zhewei Yao; Yuxiong He; Babak Salimi
>
> **摘要:** Text-to-SQL ensembles improve over single-candidate generation by drawing multiple SQL candidates and selecting one, but their effectiveness is bounded by Pass@K, the probability that at least one of K candidates is correct. Existing methods source diversity heuristically through stochastic decoding or prompt variants, leaving candidate sets dominated by correlated failures. We present DivSkill-SQL, a residual skill optimization framework that builds complementary agentic Text-to-SQL ensembles without model fine-tuning: each new skill is optimized on examples the current skill ensemble fails on, provably targeting its marginal contribution to Pass@K. On Spider2-Lite, DivSkill-SQL improves selected accuracy by up to +11.1 points on Snowflake and +8.3 on BigQuery over the strongest ensemble baseline, with consistent gains across two base models (Opus-4.6 and GPT-5.4). Skills optimized on a single dialect transfer without retraining across dialects (Snowflake, BigQuery, SQLite) and to a different task formulation, such as BIRD-Critic (+2.6 pts). Error diagnostics show up to 3x fewer hallucinated schema references and function calls, indicating that gains come from genuinely reliable complementary skills rather than surface-form variation.
>
---
#### [new 024] A Comparative Study of Language Models for Khmer Retrieval-Augmented Question Answering
- **分类: cs.CL**

- **简介: 该论文属于Khmer语言的问答任务，旨在解决低资源语言在RAG系统中的效果问题。通过比较不同检索器和生成器，评估其性能并找出最优组合。**

- **链接: [https://arxiv.org/pdf/2605.22099](https://arxiv.org/pdf/2605.22099)**

> **作者:** Sereiwathna Ros; Phannet Pov; Ratanaktepi Chhor; Kimleang Ly; Wan-Sup Cho; Saksonita Khoeurn
>
> **备注:** 14 pages, 1 figure,
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for grounding large language model (LLM) outputs in retrieved evidence, thereby reducing hallucination and improving factual accuracy. Its efficacy, however, remains largely unexamined for low-resource, non-Latin-script languages such as Khmer. In this paper, we present a RAG-based question answering system for Khmer-language telecom-domain documents. We conduct a two-phase comparative evaluation. First, we benchmark three embedding models: BGE-M3 (567M), Jina-Embeddings-v3 (570M), and Qwen3-Embedding (597M), for dense retrieval over Khmer documents. BGE-M3 consistently performs best, achieving a Hit Rate@3 of 0.285, File Hit Rate@3 of 0.700, MRR@3 of 0.221, and Precision@3 of 0.112, substantially outperforming the other retrievers. Second, using BGE-M3 as the selected retriever, we evaluate five generator backends: Qwen3 (8B), Qwen3.5 (9B), Sailor2-8B-Chat, SeaLLMs-v3-7B-Chat, and Llama-SEA-LION-v2-8B-IT, on a curated golden dataset of 200 Khmer question-answer pairs. To quantify system performance, we apply six RAGAS-inspired metrics: faithfulness, answer relevance, context relevance, factual correctness, answer similarity, and answer correctness. The results show no single model dominates across all metrics: Qwen3.5-9B achieves the highest faithfulness (0.859) and context relevance (0.726), Qwen3-8B attains the highest factual correctness (0.380), and SeaLLMs-v3-7B-Chat performs best on answer relevance (0.867), answer similarity (0.836), and answer correctness (0.599). These findings highlight that retriever choice remains a major bottleneck for Khmer RAG, while generator strengths vary depending on whether the priority is grounding, factual precision, or semantic similarity.
>
---
#### [new 025] In Silico Modeling of the RAMPHO Buffer: Dissociating Informational and Energetic Masking via Phonetic Entropy in Deep Neural Networks
- **分类: cs.CL**

- **简介: 该论文属于语音增强任务，旨在解决多说话人环境下的信息掩蔽问题。通过构建RAMPHO缓冲区的仿真模型，区分信息与能量掩蔽的影响。**

- **链接: [https://arxiv.org/pdf/2605.22465](https://arxiv.org/pdf/2605.22465)**

> **作者:** Stefan Bleeck
>
> **摘要:** The fundamental challenge of listening in multi-talker environments is a cognitive bottleneck, defined by the Ease of Language Understanding (ELU) model as a failure within the RAMPHO episodic buffer. Current deep neural networks for speech enhancement optimize purely for physical acoustics, failing to account for the cognitive penalty of informational masking. Here, we present an in silico simulation of the RAMPHO buffer using the frame-by-frame phonetic entropy of a self-supervised acoustic model (wav2vec 2.0). By contrasting a semantically intact distractor with a phase-decorrelated distractor (the Concentration Shield) across a signal-to-noise ratio (SNR) sweep, we successfully dissociate the cognitive penalty of informational distraction from the physical penalty of energetic decay. The simulation reveals a cognitive-acoustic Pareto optimization problem: destroying a distractor's semantic payload provides a release from informational masking at high SNRs, but fundamentally degrades temporal glimpsing cues at low SNRs.
>
---
#### [new 026] Hallucination as Commitment Failure: Larger LLMs Misfire Despite Knowing the Answer
- **分类: cs.CL**

- **简介: 该论文研究大模型幻觉问题，探讨其与模型规模的关系。通过分析答案可用性与概率分布，发现幻觉源于概率分散而非知识缺失。任务为理解大模型幻觉机制。**

- **链接: [https://arxiv.org/pdf/2605.22007](https://arxiv.org/pdf/2605.22007)**

> **作者:** Jewon Yeom; Jaewon Sok; Heejun Kim; Seonghyeon Park; Jeongjae Park; Taesup Kim
>
> **摘要:** Hallucination is often viewed as a direct consequence of missing knowledge: a model answers incorrectly when the correct answer is absent from its generation-time distribution, and correctly when it is present. We test this assumption by introducing a semantic notion of answer availability that aggregates token-level variants expressing the same answer concept, and asks whether the correct concept is already available at the moment the model commits to an answer. Across Qwen and Llama models from 0.8B to 72B in both Instruct and Base variants, 16-47% of Instruct hallucinations occur with substantial probability mass already on the correct concept, and the rate rises monotonically with scale. Comparing such failures against correct generations with matched semantic support, the distinguishing factor is not whether the correct concept is represented, but how its probability is distributed: correct generations concentrate mass on a single surface form, hallucinations disperse it across alternatives. The same sharpening asymmetry extends across multi-token generation and is detectable in pre-generation hidden states. Together, these results identify a single mechanism: instruction tuning sharpens answer commitment with scale, making helpfulness and confident hallucination two consequences of the same underlying disposition.
>
---
#### [new 027] Assisted Counterspeech Writing at the Crossroads of Hate Speech and Misinformation
- **分类: cs.CL**

- **简介: 该论文属于对抗仇恨言论与虚假信息的任务，旨在通过大语言模型生成有效反言论。研究比较了三种知识驱动的生成策略，并通过专家修订提升效果。**

- **链接: [https://arxiv.org/pdf/2605.22435](https://arxiv.org/pdf/2605.22435)**

> **作者:** Genoveffa Martone; Helena Bonaldi; Marco Guerini
>
> **摘要:** Hate speech and misinformation frequently co-occur online, amplifying prejudice and polarization. Given their scale, using Large Language Models (LLMs) to assist expert counterspeech (CS) writing has gained interest, yet prior work has addressed these phenomena separately. We bridge this gap by studying CS generation in contexts where both hate and misinformation co-occur. We test three knowledge-driven generation strategies: first we prompt an LLM with fact-checkers' guidelines and fact-checking articles; secondly, with NGOs' guidelines and reports; thirdly, we create a mixed strategy that combines guidelines and documents from both. 23 experts revise the generated CS, which are assessed via human and automatic metrics. While LLMs produce adequate CS in 40% of cases, expert edits substantially improve naturalness, exhaustiveness, and adherence to guidelines. Based on the post-edited CS, the mixed strategy proves to be the most effective in crowdsourcing evaluation, pairing strong factual correction with stereotype mitigation and empathetic engagement. We release a dataset of hateful and misinformed claims with expert-verified CS and supporting knowledge.
>
---
#### [new 028] Probabilistic Attribution For Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型解释任务，旨在提升大语言模型的可解释性。通过概率方法分析模型生成过程，计算token贡献度，揭示模型行为机制。**

- **链接: [https://arxiv.org/pdf/2605.21726](https://arxiv.org/pdf/2605.21726)**

> **作者:** Shilpika Shilpika; Carlo Graziani; Bethany Lusch; Venkatram Vishwanath; Michael E. Papka
>
> **备注:** 29 pages, 13 figures
>
> **摘要:** The generative nature of Large Language Models (LLMs) is reflected in the conditional probabilities they compute to sample each response token given the previous tokens. These probabilities encode the distributional structure that the model learns in training and exploits in inference. In this work, we use these probabilities to situate LLMs within the mathematical theory of stochastic processes. We use this framework to design a model-agnostic probabilistic token attribution measure, using Bayes rule to invert the next-token log-probabilities so as to capture the models internal representation of the distribution over token sequences. The representation is independent of the models computational structure. This representation yields the conditional probability of the response given the prompt, and of the response given the prompt with a token marginalized away. Our attribution score is the log of the ratio of these probabilities. We further compute the entropies of a single prompts token distributions, conditioned on the remaining context. The interplay between entropy and attribution score sheds light on LLM behavior. We evaluate 8 models across 7 prompts and investigate anomalies, token sensitivity, response stability, model stability, and training convergence, thereby improving interpretability and guiding users to focus on uncertain or unstable parts of the generation.
>
---
#### [new 029] Structure Retention in Embedding Spaces as a Predictor of Benchmark Performance
- **分类: cs.CL**

- **简介: 该论文研究嵌入模型性能与嵌入空间结构的关系，通过实验分析25个模型在五项任务中的表现，发现近邻重叠和ICA差异与任务性能高度相关。**

- **链接: [https://arxiv.org/pdf/2605.22202](https://arxiv.org/pdf/2605.22202)**

> **作者:** Amanda Myntti; Jenna Kanerva; Veronika Laippala; Filip Ginter
>
> **摘要:** In this paper, we show that high-performing embedding models organize their embedding spaces in a consistent way. We evaluate 25 contemporary embedding models on five MTEB tasks spanning four diverse task categories (retrieval, bitext mining, pair classification, and summarization) in both English and multilingual settings, and reveal that nearest-neighbor overlap and magnitude differences in independent component analysis (ICA) between paired text instances strongly correlate (even up to 0.97) with performance on the given task. Ultimately, we show that embedding tasks display varying degrees of linearity and reliance on retention of local information. Our results further the understanding of embeddings, their relation to model performance, and shed light on possible future training objectives and optimizing conditional embeddings.
>
---
#### [new 030] LatentOmni: Rethinking Omni-Modal Understanding via Unified Audio-Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态理解任务，旨在解决跨模态推理中细粒度证据提取困难的问题。提出LatentOmni框架，通过统一潜在空间实现音频视觉联合推理。**

- **链接: [https://arxiv.org/pdf/2605.22012](https://arxiv.org/pdf/2605.22012)**

> **作者:** Yifan Dai; Zhenhua Wu; Bohan Zeng; Daili Hua; Jialing Liu; Bozhou Li; Yuran Wang; Chengzhuo Tong; Hao Liang; Xiaochen Ma; Junbo Niu; Tianyu Guo; Yang Shi; Yue Ding; Yiyan Ji; Bingyin Mei; Yushuo Guan; Yuanxing Zhang; Pengfei Wan; Fangcheng Fu; Wentao Zhang
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Joint audio-visual reasoning is essential for omnimodal understanding, yet current multimodal large language models (MLLMs) still struggle when reasoning requires fine-grained evidence from both modalities. A central limitation is that explicit text-based chain-of-thought (CoT) compresses continuous audio-visual signals into discrete tokens, weakening temporal grounding and shifting intermediate reasoning toward language priors. We argue that a unified latent space is a better medium for such reasoning because it preserves dense sensory information while remaining compatible with autoregressive generation. Based on this insight, we propose \textbf{LatentOmni}, a cross-modal reasoning framework that interleaves textual reasoning with audio-visual latent states. LatentOmni introduces feature-level supervision to align latent reasoning states with task-relevant sensory features and uses Omni-Sync Position Embedding (OSPE) to maintain temporal consistency between latent audio and visual states. We further construct \textbf{LatentOmni-Instruct-35K}, a dataset of audio-visual interleaved reasoning trajectories for supervising latent-space reasoning. Comprehensive evaluation across multiple audio-visual reasoning benchmarks demonstrates that LatentOmni achieves the best performance among the evaluated open-source models and consistently outperforms the Explicit Text CoT baseline, supporting latent-space joint reasoning as a promising path toward stronger omnimodal understanding.
>
---
#### [new 031] SpecHop: Continuous Speculation for Accelerating Multi-Hop Retrieval Agents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决多跳工具使用中的延迟问题。通过SpecHop框架，实现无损推测，提升效率，减少延迟。**

- **链接: [https://arxiv.org/pdf/2605.21965](https://arxiv.org/pdf/2605.21965)**

> **作者:** Mehrdad Saberi; Keivan Rezaei; Soheil Feizi
>
> **摘要:** Large language models increasingly use external tools such as web search and document retrieval to solve information-intensive tasks. However, multi-hop tool use in complex tasks introduces substantial latency, since the model must repeatedly wait for tool observations before continuing. We study how to accelerate such trajectories without changing the final trajectory the model would have taken without acceleration, assuming access to faster but less reliable speculator tools. We develop a theoretical framework for lossless speculation in multi-hop tool-use settings, characterizing the optimal achievable latency gain. We propose SpecHop, a continuous speculation framework that maintains multiple speculative threads, verifies predicted observations asynchronously as target tool outputs arrive, commits correct branches, and rolls back incorrect ones. This preserves accuracy while reducing wall-clock latency. We show that SpecHop can approach oracle latency gains with enough active threads. Empirically, on retrieval-augmented multi-hop tasks, SpecHop closely matches theoretical predictions and reduces latency by up to 40\% in some settings. Code: this https URL
>
---
#### [new 032] Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild
- **分类: cs.CL**

- **简介: 该论文提出Hy-MT2多语言翻译模型，解决复杂现实场景下的翻译问题，支持33种语言，具备高效推理和强指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2605.22064](https://arxiv.org/pdf/2605.22064)**

> **作者:** Mao Zheng; Zheng Li; Tao Chen; Bo Lv; Mingrui Sun; Mingyang Song; Jinlong Song; Hong Huang; Decheng Wu; Hai Wang; Yifan Song; Yanfeng Chen; Guanwei Zhang; Guanghua Yu; Yi Su; Hong Liu; Jinxiang Ou; Keyao Wang; Weile Chen; Haozhao Kuang; Kai Wang; Nuo Chen; Zihao Zheng; Chenhao Wang; Bin Xing; Chengcheng Xu; Tinghao Yu; Binghong Wu; Long Xu; Jiacheng Shi; Yunhao Wang; Baifang Chen; Lei Zhang; Qi Yang; Zhao Wu; Jiacheng Li; Lan Jiang; Lanrui Wang; Kai Zhang; Shuaipeng Li; Zhongzhi Chen; Weixuan Sun; Jiaqi Zhu; An Wang; Wei Li; Jun Xia; Weidong Han; Wutian Yang; Litong Hui; Luoguo Jia; Jiajia Wu; Xinpeng Zhou; Tianxiang Fei
>
> **摘要:** Hy-MT2 is a family of fast-thinking multilingual translation models designed for complex real-world scenarios. It includes three model sizes: 1.8B, 7B, and 30B-A3B (MoE), all of which support translation among 33 languages and effectively follow translation instructions in multiple languages. For on-device deployment, with AngelSlim 1.25-bit extreme quantization, the 1.8B model requires only 440 MB of storage and improves inference speed by 1.5x. Multi-dimensional evaluations show that Hy-MT2 delivers outstanding performance across general, real-world business, domain-specific, and instruction-following translation tasks. The 7B and 30B models outperform open-source models such as DeepSeek-V4-Pro and Kimi K2.6 in fast-thinking mode, while the lightweight 1.8B model also surpasses mainstream commercial APIs from providers such as Microsoft and Doubao overall.
>
---
#### [new 033] Faithful-MR1: Faithful Multimodal Reasoning via Anchoring and Reinforcing Visual Attention
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决视觉证据感知与使用不忠实的问题。提出Faithful-MR1框架，通过锚定和强化视觉注意力提升多模态推理的准确性。**

- **链接: [https://arxiv.org/pdf/2605.22072](https://arxiv.org/pdf/2605.22072)**

> **作者:** Changyuan Tian; Zhicong Lu; Huaxing Liu; Xiang Wang; Shuai Li; Yu Chen; Wenqian Lv; Zichuan Lin; Juncheng Diao; Deheng Ye
>
> **备注:** 20 pages, 7 figures, 3 tables. Preprint
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for advancing complex reasoning in large language models, and recent work extends RLVR to multimodal large language models (MLLMs). This transfer, however, surfaces a faithfulness challenge: faithful perception of task-relevant visual evidence and faithful use of that evidence during reasoning, leading to unsatisfactory gains on multimodal benchmarks. Specifically, existing perception supervision often operates on textual descriptions rather than natively on image regions, and faithful use is largely overlooked, exposing the perception-reasoning disconnect where correctly perceived evidence is dropped or contradicted during reasoning. To close these gaps, we propose Faithful-MR1, a training framework that anchors and reinforces visual attention to address both halves of faithful multimodal reasoning. The Anchoring stage turns perception into an explicit pre-reasoning subtask, supervising a dedicated <Focus> token's attention directly against image regions rather than through textual descriptions. The Reinforcing stage exposes faithful use through counterfactual image intervention, rewarding answer-correct trajectories that concentrate visual attention where vision causally matters. Extensive experiments demonstrate that Faithful-MR1 outperforms recent multimodal reasoning baselines on both Qwen2.5-VL-Instruct 3B and 7B backbones while using substantially less training data.
>
---
#### [new 034] DeferMem: Query-Time Evidence Distillation via Reinforcement Learning for Long-Term Memory QA
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DeferMem，解决长时记忆问答中证据分散的问题。通过强化学习进行证据提炼，提升问答准确性和系统效率。**

- **链接: [https://arxiv.org/pdf/2605.22411](https://arxiv.org/pdf/2605.22411)**

> **作者:** Jianing Yin; Tan Tang
>
> **备注:** 31 pages, 3 figures
>
> **摘要:** Large language model (LLM) agents still struggle with long-term memory question answering, where answer-supporting evidence is often scattered across long conversational histories and buried in substantial irrelevant content. Existing memory systems typically process memory before future queries are known, then retrieve the resulting units based on similarity rather than their utility for answering the query. This workflow leaves downstream answerers to denoise retrieved candidates and reconstruct query-specific evidence. We present DeferMem, a long-term memory framework that decouples this problem into high-recall candidate retrieval and query-conditioned evidence distillation. DeferMem uses a lightweight segment-link structure to organize raw history and retrieve broad candidates at query time. It then applies a memory distiller trained with DistillPO, our reinforcement learning algorithm for distilling the high-recall but highly noisy candidates into a set of faithful, self-contained, and query-conditioned evidence. DistillPO formulates post-retrieval evidence distillation as a structured action comprising message selection and evidence rewriting. It optimizes this action with a decomposed-and-gated reward pipeline and structure-aligned advantage assignment, gating reward components from validity to quality checks while exposing task-level correctness feedback early and assigning each reward to its responsible output span. On LoCoMo and LongMemEval-S, DeferMem surpasses strong baselines in QA accuracy and memory-system efficiency, achieving the highest QA accuracy with the fastest runtime and zero commercial-API token cost for memory operations.
>
---
#### [new 035] Chinese sensorimotor and embodiment norms for 3,000 lexicalized concepts
- **分类: cs.CL**

- **简介: 该论文属于认知科学与人工智能领域，旨在解决非印欧语言中概念知识的具身化表征问题。构建了3000个汉语词汇的多维感官运动和具身化评分数据库，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.22616](https://arxiv.org/pdf/2605.22616)**

> **作者:** Jing Chen; Gábor Parti; Yin Zhong; Chu-Ren Huang; Marco Marelli
>
> **摘要:** Understanding how conceptual knowledge is grounded in bodily experience, and to what extent machine systems can acquire such knowledge without direct sensorimotor experience, are central questions in both cognitive science and embodied artificial intelligence research. Large-scale normative resources are essential for investigating these questions empirically, yet such resources remain sparse for non-Indo-European languages. We present a novel normative database for 3,000 lexicalized concepts in Mandarin Chinese, comprising 11-dimensional sensorimotor ratings and unidimensional embodiment ratings collected from 378 native Mandarin speakers. The ratings demonstrate high reliability and strong cross-norm validity with existing Chinese resources, each of which covers fewer words and a subset of the 11 sensorimotor dimensions. In a validation study, we tested new variables derived from a theoretically motivated metric, Perceptual Strength of Embodiment (PSE) (Huang et al., 2025), together with seven common composite variables, on lexical decision tasks. The results suggest that PSE-Sensorimotor and Minkowski-3 are the strongest composite predictors of lexical decision performance, capturing the facilitatory effects of sensorimotor information on lexical processing. A further exploratory study showed that sensorimotor ratings are substantially recoverable from purely linguistic representations using simple regression models (mean Spearman r = .62 across dimensions), though recovery varied markedly: visual and auditory dimensions yielded higher correspondence than chemosensory ones. Representational similarity analysis further showed that the relational geometry of the sensorimotor space is also partially recoverable (r = .540), consistent with the view that distributional language use encodes aspects of embodied conceptual structure.
>
---
#### [new 036] Do Factual Recall Mechanisms Carry over from Text to Speech in Multimodal Language Models?
- **分类: cs.CL**

- **简介: 该论文研究多模态语言模型中事实回忆机制在文本与语音间的迁移问题，通过因果中介分析探讨知识存储与召回机制的异同。**

- **链接: [https://arxiv.org/pdf/2605.22170](https://arxiv.org/pdf/2605.22170)**

> **作者:** Luca Modica; Filip Landin; Mehrdad Farahani; Livia Qian; Gabriel Skantze; Richard Johansson
>
> **备注:** In *SEM 2026, the 15th Joint Conference on Lexical and Computational Semantics
>
> **摘要:** In recent years, several Speech Language Models (SLMs) that represent speech and written text jointly have been presented. The question then emerges about how model-internal mechanisms are similar and different when operating in the two modalities. We focus on how these systems encode, store, and retrieve factual knowledge, which has previously been investigated for text-only models. To investigate mechanisms behind the storage and recall of factual association in SLMs, we leverage Causal Mediation Analysis, a technique previously applied to text-based models. Initial results using SpiritLM, a multimodal model integrating discrete speech tokens reveal discrepancies between text-to-text and speech-to-text results, suggesting that the emergent mechanisms for factual recall are only partially carried over from the text to the speech modality. These results advance our understanding of how internal mechanisms encode factual associations in SLMs while contributing insights for improving speech-enabled AI systems.
>
---
#### [new 037] Whose Voice Counts? Mapping Stakeholder Perspectives on AI Through Public Submissions to the U.S. Government
- **分类: cs.CL**

- **简介: 该论文属于社会感知任务，旨在分析公众对AI的看法。通过分析美国政府征集的公众意见，研究不同利益相关者对AI的关切差异，揭示AI政策中利益群体的代表性问题。**

- **链接: [https://arxiv.org/pdf/2605.22650](https://arxiv.org/pdf/2605.22650)**

> **作者:** Alina Karakanta; Alex Christiansen; Tomás Dodds; Bissie Anderson; Matteo Fuoli; Marcus Perlman; Aletta G. Dorst
>
> **摘要:** As artificial intelligence (AI) systems become more common in our daily lives, it is important to understand how different stakeholders comprehend and envisage the role that these technologies play in shaping social, political, and economic realities. In this paper, we investigate public perceptions of AI based on a corpus of letters submitted during the public consultation for the Trump Administration's US AI Action Plan. To this aim, we release a corpus cleaning pipeline and perform topic modelling and frequency analysis to explore predominant topics discussed by different subgroups (e.g., academia, individuals, private sector) and those appearing in the AI Action Plan. Our results show that individuals voice strong concerns related to the impact of AI on life, while other stakeholders are more concerned with AI development. Our comparison of topics suggests that the AI Action Plan reflects predominantly the concerns of the private sector on security, policies, and development, with individuals' concerns less represented.
>
---
#### [new 038] Harder to Defend: Towards Chinese Toxicity Attacks via Implicit Enhancement and Obfuscation Rewriting
- **分类: cs.CL**

- **简介: 该论文属于中文毒性检测任务，旨在解决隐性毒性攻击的检测难题。通过生成隐性毒性样本，评估并提升检测模型的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.22258](https://arxiv.org/pdf/2605.22258)**

> **作者:** Jingyi Kang; Junyu Lu; Bo Xu; Hongbo Wang; Linlin zong; Roy Ka-Wei Lee; Hongfei Lin
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Large language models (LLMs) require robust toxicity evaluation beyond explicit wording. This setting remains underexplored in Chinese, where toxicity may combine semantic indirectness with surface obfuscation. We introduce Chinese Implicit Toxicity Attack (CITA), a controlled red-team evaluation and defense-data generation framework, not a deployable evasion tool. CITA uses three stages: (i) Harmful Intent Learning, (ii) Implicit Toxicity Enhancement, and (iii) Obfuscation Variant Rewriting, to preserve harmful intent, increase implicitness, and add controlled surface variants. On CITA-generated evaluation samples, the seven tested detectors exhibit substantial missed-detection risks, reaching an average ASR of 69.48%; human evaluation further confirms preserved harmfulness and increased implicitness/evasiveness. As a downstream defense application, we fine-tune a Chinese Implicit Toxicity Defense model (CITD) with CITA-generated red-team data, showing that such data can improve robustness through additional training.
>
---
#### [new 039] Cohesion-6K: An Arabic Dataset for Analyzing Social Cohesion and Conflict in Online Discourse
- **分类: cs.CL**

- **简介: 该论文提出Cohesion-6K数据集，用于分析阿拉伯语在线话语中的社会凝聚力与冲突。任务是研究社交媒体中分歧与团结叙事的动态，解决如何量化此类互动的问题。工作包括数据收集、标注及分析。**

- **链接: [https://arxiv.org/pdf/2605.22447](https://arxiv.org/pdf/2605.22447)**

> **作者:** Aisha Ali Al-Athba; Wajdi Zaghouani
>
> **摘要:** The study of online discourse has become central to understanding societal polarization. While much research has focused on detecting overt toxicity, the subtle dynamics of social cohesion, meaning the interaction between divisive and unifying narratives, remain computationally underexplored (Bail, 2021; Gonzalez-Bailon and Lelkes, 2023). This paper presents Cohesion-6K, a manually and ChatGPT-assisted annotated dataset of six thousand Arabic public Facebook posts related to the Israeli Occupation of Palestine. Each post is assigned to one of five discourse categories that represent a continuum from conflict to cohesion: Conflict, Resolution, Community Engagement, Supportive Interactions, and Shared Values. The annotation process combines expert human judgment with model-assisted pre-labeling verified by trained annotators, achieving substantial inter-annotator agreement (Cohens kappa = 0.85). Quantitative analysis reveals a consistent engagement gap, where conflict-oriented posts receive between two and four times more user interaction than resolution-oriented ones (p < 0.01). This pattern illustrates how divisive discourse tends to attract disproportionate visibility in Arabic social media spaces. Cohesion-6K provides a transparent and reproducible resource for the study of online cohesion and polarization. The dataset, annotation guidelines, and preprocessing code will be released for research use under an open license, supporting future work in computational social science, digital communication, and Arabic natural language processing.
>
---
#### [new 040] Evaluating Commercial AI Chatbots as News Intermediaries
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，评估AI聊天机器人作为新闻中介的准确性，解决其在多语言、多地区事实查询中的表现问题，通过实验分析其错误模式与系统性缺陷。**

- **链接: [https://arxiv.org/pdf/2605.22785](https://arxiv.org/pdf/2605.22785)**

> **作者:** Mirac Suzgun; Emily Shen; Federico Bianchi; Alexander Spangher; Thomas Icard; Daniel E. Ho; Dan Jurafsky; James Zou
>
> **备注:** this https URL
>
> **摘要:** AI chatbots are rapidly shaping how people encounter the news, yet no prior study has systematically measured how accurately these systems, with their proprietary search integrations and retrieval-synthesis pipelines, handle emerging facts across languages and regions. We present a 14-day (February 9-22, 2026) evaluation of six AI chatbots (Gemini 3 Flash and Pro, Grok 4, Claude 4.5 Sonnet, GPT-5 and GPT-4o mini) on 2,100 factual questions derived from same-day BBC News reporting across six regional services (US & Canada, Arabic, Afrique, Hindi, Russian, Turkish). The best systems achieve over 90% multiple-choice accuracy on questions about events reported hours earlier. The same systems, however, lose 11-13% under free-response evaluation, and 16-17% across the cohort. We further characterize three failure patterns. First, every model achieves its lowest accuracy on Hindi (79% vs. 89-91% elsewhere) and citations indicate an Anglophone retrieval bias (e.g., models answering Hindi queries cite English Wikipedia more than any Hindi outlet). Second, retrieval, not reasoning, failures drive over 70% of all errors. When models retrieve a correct source, they often extract the correct answer; the problem is to land on the right source in the first place. Third, models achieving 88-96% accuracy on well-formed questions drop to 19-70% when questions contain subtle false premises, with the most vulnerable model accepting fabricated facts 64% of the time. We also identify a detection-accuracy paradox: the best false-premise detector ranks second in adversarial accuracy (abstention rate), while a weaker detector ranks first, showing that premise detection and answer recovery are partially independent capabilities. Overall, these suggest that high accuracy can mask systematic regional inequity, near-total dependence on retrieval infrastructure, and vulnerability to imperfect queries real users pose.
>
---
#### [new 041] Unified Data Selection for LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理训练任务，旨在解决高质量推理数据稀缺问题。提出HES指标，无需训练即可评估推理质量，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2605.22389](https://arxiv.org/pdf/2605.22389)**

> **作者:** Xiaoyuan Li; Yubo Ma; Chengpeng Li; Fengbin Zhu; Yiyao Yu; Keqin Bao; Wenjie Wang; Fuli Feng; Dayiheng Liu
>
> **备注:** Under Review
>
> **摘要:** Effectively training Large Language Models (LLMs) for complex, long-CoT reasoning is often bottlenecked by the need for massive high-quality reasoning data. Existing methods are either computationally expensive or fail to reliably distinguish high- from low-quality reasoning samples. To address this, we propose High-Entropy Sum (HES), a training-free metric that quantifies reasoning quality by summing only the entropy of the top (e.g., 0.5\%) highest-entropy tokens in each reasoning sample. We validate HES across three mainstream training paradigms: Supervised Fine-tuning (SFT), Rejection Fine-tuning (RFT), and Reinforcement Learning (RL), with extensive results demonstrating its consistent effectiveness and significantly reduced computational overhead. In SFT, training on the top 20\% HES-ranked data matches full-dataset performance, while using the lowest-HES data degrades it. In RFT, our HES-based training approach significantly outperforms baseline methods. In RL, HES-selected successful trajectories enable the model to learn strong reasoning patterns, significantly surpassing other compared methods. Our findings establish HES as a robust, training-free metric that enables a unified, effective, and efficient method for developing advanced reasoning in LLMs.
>
---
#### [new 042] Beyond Temperature: Hyperfitting as a Late-Stage Geometric Expansion
- **分类: cs.CL; cs.AI; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成质量与重复问题。通过研究“超拟合”现象，发现其机制不同于温度缩放，提出Late-Stage LoRA策略提升生成效果。**

- **链接: [https://arxiv.org/pdf/2605.22579](https://arxiv.org/pdf/2605.22579)**

> **作者:** Meimingwei Li; Yuanhao Ding; Esteban Garces Arias; Christian Heumann
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Recent work has identified a counterintuitive phenomenon termed "Hyperfitting", where fine-tuning Large Language Models (LLMs) to near-zero training loss on small datasets surprisingly enhances open-ended generation quality and mitigates repetition in greedy decoding. While effective, the underlying mechanism remains poorly understood, with the extremely low-entropy output distributions suggesting a potential equivalence to simple temperature scaling. In this work, we demonstrate that this phenomenon is fundamentally distinct from distribution sharpening; entropy-matched control experiments reveal that temperature scaling fails to replicate the diversity gains of hyperfitting. Furthermore, we falsify the hypothesis of static vocabulary reweighting, showing through ablation studies that hyperfitting relies on a dynamic, context-dependent rank reordering mechanism. Layer-wise analysis localizes this effect to a "Terminal Expansion" in the final transformer block, where a substantial geometric expansion of the feature space (Delta Dim approx +80.8) facilitates the promotion of deep-tail tokens. Additionally, we introduce Late-Stage LoRA, a targeted fine-tuning strategy that updates only the final 5 layers, yielding robust generation with minimal parameter updates
>
---
#### [new 043] Token-weighted Direct Preference Optimization with Attention
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，解决DPO忽略token重要性的问题。提出TwDPO及基于注意力的AttentionPO，通过内容感知权重提升性能。**

- **链接: [https://arxiv.org/pdf/2605.21883](https://arxiv.org/pdf/2605.21883)**

> **作者:** Chengyu Huang; Zhuohang Li; Sheng-Yen Chou; Claire Cardie
>
> **摘要:** Direct Preference Optimization (DPO) aligns Large Language Models with human preferences without the need for a separate reward model. However, DPO treats all tokens in responses equally, neglecting the differing importance of individual tokens. Existing token-level PO methods compute the token weights using either token-position-based heuristic functions or probability estimates given by a separately trained model, which lacks robustness and incurs extra training cost. In contrast, we propose Token-weighted DPO (TwDPO) -- a novel training objective grounded on token-weighted RL -- and AttentionPO -- an instantiation of TwDPO that uses attention from the LLM itself to estimate token weights. AttentionPO prompts the LLM to serve as a pairwise judge and check where the model attends when comparing the responses. This design makes AttentionPO content-aware, adjusting weights based on response content, and efficient, incurring only two extra forward passes per example. Experiment results show that AttentionPO significantly improves performance on AlpacaEval, MT-Bench, and ArenaHard, surpassing existing Preference Optimization methods.
>
---
#### [new 044] SynAE: A Framework for Measuring the Quality of Synthetic Data for Tool-Calling Agent Evaluations
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出SynAE框架，用于评估合成数据在工具调用代理测试中的质量。解决真实数据不足或不可用的问题，通过多维度指标分析合成数据的有效性、保真度和多样性。**

- **链接: [https://arxiv.org/pdf/2605.22564](https://arxiv.org/pdf/2605.22564)**

> **作者:** Shuaiqi Wang; Aadyaa Maddi; Zinan Lin; Giulia Fanti
>
> **摘要:** Today, tool-calling agents are commonly evaluated or tested on static datasets of execution traces, including input commands, agent responses, and associated tool calls. However, internal production datasets are often insufficient or unusable for testing; for example, they may contain sensitive or proprietary data, or they may be too sparse to support comprehensive testing (especially pre-deployment). In these settings, practitioners are increasingly replacing or augmenting real datasets with synthetic ones for evaluation purposes. A key challenge is quantifying the relation between these synthetic datasets and the real data. We introduce SynAE, an evaluation framework for assessing how well synthetic benchmarks for multi-turn, tool-calling agents replicate and augment the characteristics of real data trajectories. SynAE assesses the validity, fidelity, and diversity of synthetic data across four metric categories: (i) task instructions and intermediate responses, (ii) tool calls, (iii) final outputs, and (iv) downstream evaluation. We evaluate SynAE using recent agent benchmarks and test common synthetic data failure modes via realistic and controlled generation schemes. SynAE detects fine-grained variations in data validity, fidelity and diversity, and shows that no single metric is sufficient to fully characterize synthetic data quality, motivating a multi-axis evaluation of synthetic data for agent testing. A demo of SynAE is available at this https URL, with code at this https URL.
>
---
#### [new 045] Pattern-and-root inflectional morphology: the Arabic broken plural
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决阿拉伯语名词屈折形态的描述问题。通过将传统根-形模型转换为形-根模型，提出一种新的词形分析方法，简化分类并提高准确性。**

- **链接: [https://arxiv.org/pdf/2605.22310](https://arxiv.org/pdf/2605.22310)**

> **作者:** Alexis Amid Neme; Eric Laporte
>
> **摘要:** We present a substantially implemented model of description of the inflectional morphology of Arabic nouns, with special attention to the management of dictionaries and other language resources by Arabic-speaking linguists. The breakthrough lies in the reversal of the traditional root-and-pattern Semitic model into pattern-and-root, giving precedence to patterns over roots. Our model includes broken plurals (BPs), i.e. plurals formed by modifying the stem. It is based on the traditional notions of root and pattern of Semitic morphology. However, as compared to traditional Arabic morphology, it keeps the formal description of inflection separate from that of derivation and semantics. As traditional Arabic dictionaries, the updatable dictionary is structured in lexical entries for lemmas, and the reference spelling is fully diacritized. In our model, morphological analysis of Arabic text is performed directly with a dictionary of words and without morphophonological rules. Our taxonomy for noun inflection is simple, orderly and detailed. We simplify the taxonomy of singular patterns by specifying vowel quantity as v or vv, and ignoring vowel quality. Root alternations and orthographical variations are encoded independently from patterns and in a factual way, without deep roots or morphophonological or orthographical rules. Nouns with a triliteral BP are classified according to 22 patterns subdivided into 90 classes, and nouns with a quadriliteral BP according to 3 patterns subdivided into 70 classes. These 160 classes become 300 inflectional classes when we take into account inflectional variations that affect only the singular. We provide a straightforward encoding scheme that we applied to 3 200 entries of BP nouns.
>
---
#### [new 046] Audience Engagement with Arabic Women's Social Empowerment and Wellbeing: A Decadal Corpus
- **分类: cs.CL**

- **简介: 该论文构建了一个包含25万条阿拉伯语社交媒体数据的语料库，用于研究女性赋权与社会福祉。属于自然语言处理任务，旨在分析性别话语与情感互动。**

- **链接: [https://arxiv.org/pdf/2605.22204](https://arxiv.org/pdf/2605.22204)**

> **作者:** Wajdi Zaghouani; Mabrouka Bessghaier; MD. Rafiul Biswas; Shimaa Amer Ibrahim
>
> **摘要:** This paper presents the Arabic Women and Society Corpus, a ten year collection of 252,487 public Arabic Facebook posts related to women's empowerment and social wellbeing. The corpus was collected from 51,660 pages across 77 countries between 2013 and 2024, resulting in more than 267 million user interactions. Each post includes engagement metrics such as shares, comments, and emotional reactions, providing a unique view of audience sentiment and social attention. The data were processed using an automated pipeline with language identification, normalization, and metadata cleaning to ensure reliability and reproducibility. The corpus enables large scale analysis of gender discourse, social reform, and emotional engagement across Arabic dialects. It supports research in Arabic natural language processing, computational social science, and digital communication studies. The dataset and accompanying documentation will be released under request for research use.
>
---
#### [new 047] ChronoMedKG: A Temporally-Grounded Biomedical Knowledge Graph and Benchmark for Clinical Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ChronoMedKG，一个包含时间信息的生物医学知识图谱，解决临床推理中静态知识不足的问题。通过多模型提取与过滤，构建具有时间标注的疾病关联，提升临床检索准确性。**

- **链接: [https://arxiv.org/pdf/2605.22734](https://arxiv.org/pdf/2605.22734)**

> **作者:** Md Shamim Ahmed; Farzaneh Firoozbakht; Lukas Galke Poech; Jan Baumbach; Richard Röttger
>
> **备注:** 9 pages main text plus appendices, 8 figures. Dataset and benchmark paper. ChronoMedKG released under CC BY 4.0 and ChronoTQA/code under MIT (Zenodo: https://doi.org/10.5281/zenodo.19697542). Under review
>
> **摘要:** Biomedical knowledge graphs (KGs) treat disease associations as static facts, but temporal information is crucial for clinical reasoning, e.g., a symptom diagnostic of one disease at age 3 may imply a different disease at age 13. Existing KGs such as PrimeKG, Hetionet, and iKraph do not encode when a finding becomes clinically relevant over the course of a disease. This limits their usefulness for longitudinal clinical reasoning and retrieval augmentation. We introduce ChronoMedKG, a temporal biomedical knowledge graph that contains 460,497 evidence-linked triples (filtered from 13M raw extractions) covering 13,431 diseases. Each association is tied to temporal components like onset window or progression stage, which are backed by PMID-traceable evidence and a multi-signal credibility score. The graph is constructed through a disease-autonomous multi-agent pipeline in which multiple frontier LLMs independently extract knowledge from PubMed and PMC literature. Only those relations are kept that are supported by multi-model consensus, survive credibility filtering, as well as ontology alignment. ChronoMedKG scored 92.7% agreement against Orphadata and adds temporal grounding for 6,250 diseases absent from HPOA, Orphadata, and Phenopackets, including 1,657 Orphanet-coded rare diseases. We further introduce ChronoTQA, a benchmark of 3,341 questions across eight task types (six temporal plus two static controls), with a 12-question supplementary probe. Frontier LLMs lose roughly 30 points moving from static to temporal questions; ChronoMedKG retrieval rescues 47-65% of their long-tail failures, against 17-29% for HPOA-RAG. As such, ChronoMedKG provides a crucial temporal axis for retrieval-augmented clinical systems that was previously absent.
>
---
#### [new 048] RankJudge: A Multi-Turn LLM-as-a-Judge Synthetic Benchmark Generator
- **分类: cs.CL**

- **简介: 该论文提出RankJudge，用于评估多轮对话中LLM作为评判者的性能。解决LLM评判基准不足的问题，通过构造有缺陷的对话对进行严格评价。**

- **链接: [https://arxiv.org/pdf/2605.21748](https://arxiv.org/pdf/2605.21748)**

> **作者:** Zhenwei Tang; Zhaoyan Liu; Rasa Hosseinzadeh; Tongzi Wu; Keyvan Golestan; Jesse C. Cresswell
>
> **摘要:** As interactive LLM-based applications are created and refined, model developers need to evaluate the quality of generated text along many possible axes. For simpler systems, human evaluation may be practical, but in complicated systems like conversational chatbots, the amount of generated text can overwhelm human annotation resources. Model developers have begun to rely heavily on auto-evaluation, where LLMs are also used to judge generation quality. However, existing LLM-as-a-judge benchmarks largely focus on simple Q\&A tasks that do not match the complexity of multi-turn conversations. We introduce RankJudge, a benchmark generator for evaluating LLM-as-a-judge on multi-turn conversations grounded in reference documents. RankJudge creates pairs of conversations where one conversation has a single flaw injected into one turn. This construction allows paired conversations to be labeled unambiguously as better or worse, and precisely isolates failure categories to individual turns, enabling a strict joint correctness criterion for judging. We implement RankJudge across the domains of machine learning, biomedicine, and finance, evaluate 21 frontier LLM judges, and rank those judges via the Bradley-Terry model. Our formulation also allows ranking each conversation pair with difficulty ratings, which we use to dynamically curate the evaluation slice to reduce label noise, as confirmed via human annotation. We find that judge rankings are stable under partial observability, coarser correctness criteria, and an alternative random-walk rating algorithm.
>
---
#### [new 049] Claim-Selective Certification for High-Risk Medical Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于医疗问答任务，解决高风险场景下混合证据的可信度评估问题。通过分解响应为可验证声明并进行评分，实现精准的认证决策。**

- **链接: [https://arxiv.org/pdf/2605.21949](https://arxiv.org/pdf/2605.21949)**

> **作者:** Shao Kan
>
> **备注:** 22 pages, 7 figures, 11 tables
>
> **摘要:** Medical RAG systems in high-risk QA settings are often evaluated through a single answer-or-abstain decision, but mixed evidence may support one claim, require conditions for another, and contradict a third. We study claim-selective certification: each response is decomposed into verifiable claims, scored against retrieved evidence, and mapped by an intent-aware selector to {full, partial, conflict, abstain}. On the primary weak-label certificate protocol, whose real-source-only dev/test rows cover the naturally occurring non-abstain actions, the full system records UCCR=0.0000, PAU=1.0000, PAU Precision=0.9901, and action accuracy=0.9204 on dev (n=314), and UCCR=0.0000, PAU=0.9967, PAU Precision=0.9739, and action accuracy=0.8997 on test (n=319). UCCR measures unsupported-claim risk within the certificate definition, and a source-missing counterfactual slice evaluates abstain under empty evidence. Shortcut controls quantify the action-label prior explained by source and intent metadata, while source/evidence-novel slices characterize transfer boundaries. The resulting interface separates action-label prediction from evidence-linked claim selection under mixed evidence.
>
---
#### [new 050] One prompt is not enough: Instruction Sensitivity Undermines Embedding Model Evaluation
- **分类: cs.CL; cs.IR**

- **简介: 该论文研究指令敏感性对嵌入模型评估的影响，指出单次提示评估不足以反映模型真实性能。任务属于模型评估，解决评估方法不全面的问题，通过实验验证多提示的重要性。**

- **链接: [https://arxiv.org/pdf/2605.22544](https://arxiv.org/pdf/2605.22544)**

> **作者:** Yevhen Kostiuk; Kenneth Enevoldsen
>
> **摘要:** Instruction embedding models have become common among state-of-the-art models, however are evaluated using a single prompt per task. The single-point evaluation ignores a main problem of the instruction-based approach namely: sensitivity to the phrasing of the instruction. We present an empirical study of prompt sensitivity across 6 embedding models, 11 datasets, and 15 task-specific prompts per dataset, a total of 990. We show that reported scores misrepresent the distribution of scores over plausible prompts. The default prompt can both systematically understate or overstate performance. Furthermore, we show that the leaderboard ranking is not robust to prompt selection: by choosing prompts favorably, any model in our study can be promoted to first place. Our findings suggest that single-prompt evaluation is insufficient for instruction-tuned embedding models and that benchmarks should incorporate prompt robustness, either by evaluating over multiple prompts or by reporting sensitivity alongside point estimates.
>
---
#### [new 051] Modeling Pathology-Like Behavioral Patterns in Language Models Through Behavioral Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于语言模型行为建模任务，旨在研究通过微调生成病理样行为模式。工作包括构建合成数据集，训练模型产生特定行为倾向，并验证其在不同任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.22356](https://arxiv.org/pdf/2605.22356)**

> **作者:** Nicola Milano; Davide Marocco
>
> **摘要:** Large language models are increasingly used as computational tools for modeling human-like behavior. We introduce a behavioral induction framework that modifies model policies through fine-tuning on structured decision-making tasks: using synthetic datasets inspired by maladaptive behavioral patterns, including depression and paranoia, we train transformer-based language models to consistently select specific classes of actions across diverse contexts. We then test whether this behavioral optimization produces systematic changes in generative distributions. Across two architectures, fine-tuned models show stable, context-general shifts in next-token probability distributions, including increased probability assigned to negative and threat-related interpretations in open-ended language tasks. These effects generalize beyond training contexts and are detectable in qualitative completions, psychometric-style evaluations, and quantitative distributional metrics such as Jensen-Shannon divergence. Induced behavioral profiles also show partial specificity. Models optimized for different behavioral patterns exhibit dissociable response tendencies across evaluation probes, suggesting that structured behavioral training produces differentiated policy-level biases rather than generic distributional skew. We interpret these findings as evidence that consistent behavioral optimization in LLMs can generate stable behavioral and distributional patterns consistent with altered latent priors, linking action selection and language generation. More broadly, the results support a view of LLMs as policy-based systems in which behavioral constraints shape emergent representational structure, highlighting their potential as controlled testbeds for studying the relationship between behavior, interpretation, and generative language in computational models of cognition.
>
---
#### [new 052] Psy-Chronicle:A Structured Pipeline for Synthesizing Long-Horizon Campus Psychological Counseling Dialogues
- **分类: cs.CL**

- **简介: 该论文属于心理辅导对话生成任务，旨在解决长期心理困扰演化建模问题。提出Psy-Chronicle框架，生成跨会话的长时序对话，并构建CPCD数据集进行评估。**

- **链接: [https://arxiv.org/pdf/2605.22140](https://arxiv.org/pdf/2605.22140)**

> **作者:** Chaogui Gou; Jiarui Liang
>
> **摘要:** In recent years, large language models have shown substantial potential in psychological support tasks. However, existing psychological counseling data mostly rely on single-turn question answering or short multi-turn dialogues, making it difficult to characterize how college students' psychological distress accumulates, interacts, and gradually evolves over long periods within campus life events. To address this issue, this paper proposes Psy-Chronicle, a structured data-generation framework for synthesizing long-horizon campus psychological counseling dialogues. We generate a semester-spanning temporal stress event graph to model the chronological order and evolutionary dependencies among campus stress events. Through interactive simulation between a student agent and a counselor agent, together with a structured memory integration mechanism, Psy-Chronicle generates long-horizon dialogues with continuity across counseling sessions. Based on Psy-Chronicle, we construct and open-source CPCD, a Chinese long-horizon dialogue dataset for college psychological counseling, containing 100 student profiles, 90,000 counseling dialogues. We further build CPCD-Bench to evaluate models' long-horizon campus counseling capabilities from three dimensions: session-level response, long-horizon memory recall, and temporal-causal reasoning. Experimental results show that CPCD effectively improves session-level response generation and long-horizon memory recall for models with the same base architecture. Meanwhile, improvements in temporal-causal reasoning remain limited, indicating that event-chain organization and causal explanation are key challenges in long-horizon psychological counseling modeling. The related code and data are available at: this https URL
>
---
#### [new 053] Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言道德分类任务，旨在解决非英语语言道德数据不足的问题。通过机器翻译和验证方法，证明翻译后的数据可有效用于跨语言机器学习。**

- **链接: [https://arxiv.org/pdf/2605.22660](https://arxiv.org/pdf/2605.22660)**

> **作者:** Maciej Skorski
>
> **摘要:** Moral language is subtle and culturally variable, making it difficult to translate faithfully across languages. Idiomatic expressions, slang, and cultural references introduce hard-to-avoid translation artifacts. Yet automated moral values classification depends on language-specific annotated corpora that exist almost exclusively in English. We investigate whether LLM-based translation can bridge this gap, taking Polish as a test case. Using $\sim$50k morally-annotated social media posts from a diverse range of topics, we apply a principled four-method validation pipeline: LaBSE cross-lingual embedding similarity, Centered Kernel Alignment (CKA), LLM-as-judge evaluation, and deep learning classifier parity tests. We show that despite shortcomings in handling slang, vulgarity, and culturally-loaded expressions, direct translation preserves subtle moral cues well enough to be harvested by cross-lingual machine learning -- with mean cosine similarity of 0.86 and AUC gaps of 0.01--0.02 across all foundations closing further under fine-tuning of language models. These results demonstrate that machine translation is a practical and cost-effective path to moral values research in languages currently under-resourced in this domain. We demonstrate this for Polish as a representative Slavic language, with expected generalisation to related languages.
>
---
#### [new 054] Tokenisation via Convex Relaxations
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的分词任务，旨在解决传统分词算法局部最优的问题。通过线性规划和凸优化构建新算法ConvexTok，提升分词效果并提供最优性保证。**

- **链接: [https://arxiv.org/pdf/2605.22821](https://arxiv.org/pdf/2605.22821)**

> **作者:** Jan Tempus; Philip Whittington; Craig W. Schmidt; Dennis Komm; Tiago Pimentel
>
> **摘要:** Tokenisation is an integral part of the current NLP pipeline. Current tokenisation algorithms such as BPE and Unigram are greedy algorithms -- they make locally optimal decisions without considering the resulting vocabulary as a whole. We instead formulate tokeniser construction as a linear program and solve it using convex optimisation tools, yielding a new algorithm we call ConvexTok. We find ConvexTok consistently improves intrinsic tokenisation metrics and the bits-per-byte (BpB) achieved by language models; it also improves downstream task performance, but less consistently. Furthermore, ConvexTok allows the user to certify how far their tokeniser is from optimal, with respect to a certain objective, via a lower bound, and we empirically find it to be within 1\% of optimal at common vocabulary sizes.
>
---
#### [new 055] Scene Abstraction for Lexical Semantics: Structured Representations of Situated Meaning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决词汇意义的语境化表示问题。通过构建场景抽象框架，提取词语在不同语境中的结构化含义，并验证其与人类理解的一致性。**

- **链接: [https://arxiv.org/pdf/2605.22542](https://arxiv.org/pdf/2605.22542)**

> **作者:** Yejin Cho; Katrin Erk
>
> **摘要:** Coffee and tea share many properties, yet they evoke strikingly different situations, atmospheres, and affective associations. These situated dimensions of word meaning are real and systematic, but they remain implicit in most computational representations of lexical meaning. We propose Scene Abstraction, a framework for constructing structured representations of the interpretive scenes that words participate in across usage contexts. Each scene consists of a Contextual Scene (Events, Entities, Setting) and an expression-centered Expression Profile (Engaged events, Generalizable properties, Evoked emotions), operationalized through few-shot prompting of a large language model. Our contributions are three-fold: (1) a structured representation framework for situated lexical meaning; (2) COCA-Scenes, a dataset of 520 usage instances across 26 keywords for distinct scene identification; and (3) empirical evidence from two experiments suggesting that scenes are reliably identifiable across human observers (82.4% accuracy, +11.8 pp over text-only embeddings) and that our scene profiles more closely align with human interpretation of words in context than ATOMIC-based alternatives (86.4% preference across three semantic dimensions).
>
---
#### [new 056] Ishigaki-IDS-Bench: A Benchmark for Generating Information Delivery Specification from BIM Information Requirements
- **分类: cs.CL**

- **简介: 该论文属于结构化生成任务，旨在解决从BIM信息需求生成符合标准的IDS XML的问题。构建了Ishigaki-IDS-Bench基准，评估模型生成能力。**

- **链接: [https://arxiv.org/pdf/2605.22079](https://arxiv.org/pdf/2605.22079)**

> **作者:** Ryo Kanazawa; Koyo Hidaka; Teppei Miyamoto; Takayuki Kato; Tomoki Ando; Chenguang Wang; Dayuan Jiang; Naofumi Fujita; Shuhei Saitoh; Atomu Kondo; Koki Arakawa; Daiho Nishioka
>
> **备注:** 7 pages; benchmark data and evaluation scripts are available on GitHub and Hugging Face
>
> **摘要:** Large language models (LLMs) are widely used to generate structured outputs such as JSON, SQL, and code, yet public resources remain limited for evaluating generation that must simultaneously satisfy industry-standard XML and domain vocabulary constraints. This paper presents Ishigaki-IDS-Bench, a benchmark for evaluating the ability to generate Information Delivery Specification (IDS) XML from Building Information Modeling (BIM) information requirements. The benchmark contains 166 BIM/IDS expert-authored and verified examples created by expanding 83 practical scenarios into Japanese and English, corresponding gold IDS files, and metadata for input format, language, turn setting, IFC version, and construction domain. Its evaluation combines IDSAuditTool-based Processability, Structure, and Content audits with content-agreement evaluation against gold IDS files. In zero-shot evaluation over 10 LLMs, the best model reaches 65.6% macro F1 for content agreement, while only 27.7% of outputs pass the Content audit. These results show that current LLMs can express part of the information requirements as IDS, but still struggle to stably generate XML that satisfies the IDS standard and IFC vocabulary constraints. Ishigaki-IDS-Bench supports comparative evaluation, failure analysis, and the development of constrained structured generation methods that conform to domain standards. We release the evaluation scripts and benchmark data under the CC BY 4.0 license on GitHub and Hugging Face.
>
---
#### [new 057] Broadening Access to Transportation Safety Data with Generative AI: A Schema-Grounded Framework for Spatial Natural Language Queries
- **分类: cs.CL**

- **简介: 论文提出一种基于自然语言的交通安全隐患分析框架，解决数据访问不均问题。通过LLM解析用户意图，结合确定性执行确保结果可靠，提升非专业用户的数据获取能力。**

- **链接: [https://arxiv.org/pdf/2605.21712](https://arxiv.org/pdf/2605.21712)**

> **作者:** Mahdi Azhdari; Eric J. Gonzales
>
> **备注:** 30 pages, 5 figures
>
> **摘要:** Transportation safety analysis requires integrating crash records, roadway attributes, and geospatial data through GIS-based workflows, but access remains uneven across agencies and community stakeholders. Technical prerequisites create a gap between analytical tools central to safety planning and the practitioners able to use them. Local agencies, school committees, and residents may have safety concerns but limited capacity to retrieve, filter, map, and analyze relevant data. Generative AI offers a way to narrow this divide, but its public-sector use raises questions about reliability, reproducibility, and governance. This paper presents a schema-grounded natural language interface for transportation safety analysis, using a large language model (LLM) to interpret user intent while preserving deterministic, reviewable execution against an authoritative database. User queries are translated into structured semantic frames, validated by a rule-based layer, compiled into a typed directed acyclic graph of spatial operations, and executed against a PostGIS database. This bounded design separates language interpretation from deterministic execution, keeping results reproducible and schema-grounded while removing access barriers. The framework is evaluated using a statewide Massachusetts transportation safety database integrating crash records, roadway attributes, and geospatial layers including schools, bus stops, crosswalks, and municipal boundaries. All queries executed successfully; the validation layer corrects errors in 29% of evaluation queries, reflecting the gap between flexible natural language and strict schema-grounded requirements. The results suggest that combining natural language accessibility with deterministic execution is a practical direction for broadening access to transportation safety data, with implications for trustworthy AI in public-sector planning.
>
---
#### [new 058] Polite on the Surface, Wrong in Practice: A Curated Dataset for Fixing Honorific Failures in Multilingual Bangla Generation
- **分类: cs.CL**

- **简介: 该论文属于多语言文本生成任务，旨在解决Bangla语中尊称使用不一致的问题。通过构建BLADE数据集，对模型进行微调以提升文化语境下的表达准确性。**

- **链接: [https://arxiv.org/pdf/2605.22487](https://arxiv.org/pdf/2605.22487)**

> **作者:** Md. Asaduzzaman Shuvo; Mahedi Hasan; Md. Tashin Parvez; Azizul Haque Noman; Md. Shafayet Hossain Ovi
>
> **摘要:** Recent advances in Multilingual Large Language Models (MLLMs) have significantly enhanced cross-lingual conversational capabilities, yet modeling culturally nuanced and context-dependent communication remains a critical bottleneck. Specifically, existing state-of-the-art models exhibit a severe pragmatic gap when handling structural variations, regional idioms, and honorific consistencies in low-resource contexts like Bangla. To address this limitation, we introduce a novel, culturally aligned instruction-tuning dataset for \textbf{BangLa Application and DialoguE generation - BLADE} and benchmarking framework comprising $4,196$ meticulously curated interaction pairs. We leverage this resource to systematically fine-tune and evaluate leading open-weight architectures, including DeepSeek-8B and LLaMA-3.2-3B, utilizing parameter-efficient fine-tuning via LoRA adapters in a 4-bit NormalFloat (NF4) quantization framework. Our empirical evaluations demonstrate that models fine-tuned on our dataset yield substantial improvements in structural fidelity and honorific alignment, providing a rigorous benchmark for bridging pragmatic disparities in low-resource multilingual text generation. Code and dataset: this https URL
>
---
#### [new 059] PromptNCE: Pointwise Mutual Information Predictions Using Only LLMs and Contrastive Estimation Prompts
- **分类: cs.CL**

- **简介: 该论文属于信息理论任务，旨在零样本估计点互信息（PMI）。通过提示和对比估计方法，提出PromptNCE模型，解决低数据场景下的PMI估算问题。**

- **链接: [https://arxiv.org/pdf/2605.21776](https://arxiv.org/pdf/2605.21776)**

> **作者:** Juliette Woodrow; Chris Piech
>
> **摘要:** Estimating mutual information from text usually requires training a task-specific critic, which limits its use in low-data settings. We ask whether large language models can instead estimate pointwise mutual information zero-shot, using only prompts and elicited probabilities. We introduce a benchmark with human-derived ground-truth PMI across three publicly available datasets, and evaluate five information-theoretic prompting-based estimators. Our main method, PromptNCE, frames conditional probability estimation as a contrastive task and augments the candidate set with an explicit OTHER category. We show theoretically that adding OTHER recovers the true conditional P(y | x) rather than just a ranking over listed candidates, turning a contrastive prompt into a general-purpose zero-shot probability estimator. PromptNCE is the best zero-shot method on all three datasets, reaching Spearman correlation up to 0.82 with human-derived PMI. We also present a case study in computer science education showing how these estimators can be used to score student knowledge summaries in a low-data setting.
>
---
#### [new 060] Agentic CLEAR: Automating Multi-Level Evaluation of LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Agentic CLEAR，用于自动化评估LLM代理行为。解决代理系统难以有效评估的问题，通过多层级分析提供动态、数据驱动的反馈。**

- **链接: [https://arxiv.org/pdf/2605.22608](https://arxiv.org/pdf/2605.22608)**

> **作者:** Asaf Yehudai; Lilach Eden; Michal Shmueli-Scheuer
>
> **备注:** ACL
>
> **摘要:** Agentic systems are becoming more capable: agents define strategies, take actions, and interact with different environments. This autonomy poses serious challenges for overseeing and assessing agent behavior. Most current tools are limited, focusing on observability with basic evaluation capabilities or imposing static, hand-crafted error taxonomies that cannot adapt to new domains. To address this gap, we present Agentic CLEAR, an automatic, dynamic, and easy-to-use evaluation framework. It produces textual insights into the agent behavior on three levels of granularity: system, trace, and node. Agentic CLEAR operates above the observability layer, enabling seamless integration and featuring an intuitive UI that makes agent evaluation highly accessible. In our experiments on four benchmarks, seven agentic settings, and tens of thousands of LLM calls, we show that Agentic CLEAR produces high-quality, data-driven, insightful feedback. Our analysis shows strong alignment with human-annotated errors and the ability to predict task success rate.
>
---
#### [new 061] Self-Policy Distillation via Capability-Selective Subspace Projection
- **分类: cs.CL**

- **简介: 该论文提出Self-Policy Distillation（SPD）方法，用于提升大语言模型的自蒸馏效果。针对现有方法依赖外部信号或泛化性差的问题，SPD通过提取能力子空间，实现无监督的、可泛化的模型优化。**

- **链接: [https://arxiv.org/pdf/2605.22675](https://arxiv.org/pdf/2605.22675)**

> **作者:** Guangya Hao; Yitong Shang; Yunbo Long; Zhuokai Zhao; Hanxue Liang
>
> **摘要:** Self-distillation bootstraps large language models (LLMs) by training on their own generations. However, existing methods either rely on external signals to curate self-generated outputs (e.g., correctness filtering, execution feedback, and reward search), which are costly and unavailable for the best-performing frontier models, or skip curation entirely and train on all raw outputs, an approach that is often domain-specific and hard to generalize. Both also share a deeper weakness that self-generated outputs entangle task-relevant capability with others, such as stylistic patterns, formatting artifacts, and model-specific errors, diluting the signal for the specific capability one aims to improve. In this paper, we propose Self-Policy Distillation (SPD), which achieves generalizable, capability selective without any external signal. Specifically, SPD extracts a low-rank capability subspace from the model's own gradients on correctness-defining tokens, projects key-value (KV) activations into this subspace during self-generation, and fine-tunes on the resulting raw outputs with standard next-token prediction loss. Through extensive experiments across code generation, mathematical reasoning, and multiple-choice QA, we show that SPD achieves up to 13% improvement over state-of-the-art self-distillation methods without external signals and up to 16% improvement over pre-trained baselines. Notably, SPD demonstrates superior generalizability, achieving 15% better performance under out-of-domain generalization settings.
>
---
#### [new 062] Seeing the Poem: Image-Semantic Detection of AI-Generated Modern Chinese Poetry with MLLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于AI生成诗歌检测任务，旨在解决传统方法在检测现代中文诗歌上的不足。通过引入图像语义信息，提升LLM检测效果。**

- **链接: [https://arxiv.org/pdf/2605.22654](https://arxiv.org/pdf/2605.22654)**

> **作者:** Shanshan Wang; Fengying Ye; Hanjia Lyu; Caiwen Gou; Junchao Wu; Jingming Yao; Chengzhong Xu; Jiebo Luo; Derek F. Wong
>
> **摘要:** Previous detection studies have shown that LLMs cannot be effectively used as detectors, but these studies have not addressed modern Chinese poetry. Moreover, no relevant research has explored the performance of LLMs in detecting modern Chinese poetry. This paper evaluates and enhances the performance of LLMs as detectors for modern Chinese poetry, and proposes an image-semantic guided poetry detection method. Compared with traditional detection approaches, our method innovatively incorporates images that reflect the content of the poetry. Through example-driven approaches, our method effectively integrates information such as meaning, imagery, and feeling from the image, then forms a complementary judgment with the poem text. Experimental results demonstrate that the LLM detectors based on our method outperform baseline detectors based on plain text, and even surpass the best-performing traditional detector, RoBERTa. The Gemini detector using our method achieves a Macro-F1 score of 85.65%, reaching the state-of-the-art level. The performance improvements of different LLM detectors on multiple LLMs-generated data prove the effectiveness of our method.
>
---
#### [new 063] Reducing Political Manipulation with Consistency Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中的隐性政治偏见问题。通过提出一致性训练方法，减少模型在不同政治立场上的不对称响应。**

- **链接: [https://arxiv.org/pdf/2605.22771](https://arxiv.org/pdf/2605.22771)**

> **作者:** Long Phan; Devin Kim; Alexander Pan; Alice Blair; Adam Khoja; Dan Hendrycks
>
> **摘要:** Large language models (LLMs) exhibit systematic political bias across a variety of sensitive contexts. We find that LLMs handle counterpart topics from opposing political sides asymmetrically. We refer to this phenomenon as covert political bias and identify 7 categories of techniques through which it operates. We propose two metrics for covert bias: Sentiment Consistency measures symmetry in rhetoric and framing across paired political prompts; Helpfulness Consistency measures symmetric depth and engagement. To reduce both types of covert bias, we introduce Political Consistency Training (PCT), an RL training method with two complementary paradigms: Sentiment Consistency Training and Helpfulness Consistency Training. We show that PCT preserves overall helpfulness, substantially reduces covert political bias, and generalizes to held-out benchmarks. We release our work at this https URL
>
---
#### [new 064] From Correlation to Cause: A Five-Stage Methodology for Feature Analysis in Transformer Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出五阶段方法分析Transformer模型中的因果特征，解决特征因果性验证问题。通过实验验证了特征的因果性和鲁棒性，优化了部署成本。**

- **链接: [https://arxiv.org/pdf/2605.22462](https://arxiv.org/pdf/2605.22462)**

> **作者:** Caleb Munigety
>
> **摘要:** We propose a five-stage methodology for causal feature analysis in transformer language models (probe design, feature extraction, causal validation, robustness testing, and deployment integration) and demonstrate it end-to-end on GPT-2 small performing the Indirect Object Identification (IOI) task. Activation patching recovers the canonical IOI circuit (layer-9 head 9 alone gives recovery +1.02). A sparse autoencoder recovers per-name selective features with effect sizes of 30 to 50 activation units. Causal validation finds these features specifically but only partially causal: ablating fifteen of them leaves the model accurate on 98% of prompts. Two NLA-inspired evaluations strengthen this picture: the fifteen selective features explain only 31% of activation variance versus the SAE's 99.7%, and selectivity ratio anticorrelates with causal force (r = -0.56). Robustness testing under three distribution shifts finds that the circuit transfers cleanly but feature ablation effects degrade substantially, exposing a gap between detection robustness and causal robustness. A cost-based deployment evaluation (assumed $50/FN, $0.42/FP, 2% error rate) finds an optimal monitor configuration yielding $8.96 per 1000 queries against a $1000 baseline, a 99.1% saving. Optimal composition strategy varies with cost ratio and base rate. The conjunction of stages produces findings no single stage would.
>
---
#### [new 065] Tokenization with Split Trees
- **分类: cs.CL**

- **简介: 该论文提出ToaST，一种优化压缩的子词分词方法，解决分词效率与上下文长度问题。通过递归分割和整数规划选择词汇，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22705](https://arxiv.org/pdf/2605.22705)**

> **作者:** Craig W. Schmidt; Michael Krumdick; Adam Wiemerslage; Seth Ebner; Varshini Reddy; Yuval Pinter; Chris Tanner
>
> **摘要:** We introduce Tokenization with Split Trees (ToaST), a subword tokenization method that directly optimizes compression under a new recursive inference procedure. ToaST greedily splits each pretoken into a full binary tree using precomputed byte n-gram counts, independent of any vocabulary. Given a vocabulary, inference recursively descends each split tree and emits the first in-vocabulary node reached on each path. Vocabulary selection is formulated as an Integer Program (IP) that minimizes the total token count over all split trees under this inference procedure. The Linear Programming (LP) relaxation is near-integral in practice, yielding provably near-optimal vocabularies, with training time empirically scaling quadratically in the number of split trees. On English text, ToaST reduces token counts by more than 11% compared to BPE, WordPiece, and UnigramLM at vocabulary sizes of 40,960 and above, reducing the number of inference tokens for models using this tokenizer, thus extending the effective context length. ToaST also uses common single-byte tokens less frequently than these baselines, leading to a substantial improvement in Renyi efficiency. In experiments training 1.5B parameter language models, ToaST achieves the highest CORE score, outperforming baselines by 2.6%--7.6%, with significance for two of three, and scoring best on 13 of 22 individual tasks.
>
---
#### [new 066] GHI: Graphormer over Conditioned Hypergraph Incidence for Aspect-Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文聚焦于基于方面的情感分析任务，解决情感证据与方面绑定的问题。提出GHI框架，通过结构化推理提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22228](https://arxiv.org/pdf/2605.22228)**

> **作者:** Yu Du; Wenlong Zhu; Xingze Li; Chenglong Cao; Jing Wang; Yukun Ma
>
> **备注:** 15 pages, 8 figures, 7 tables
>
> **摘要:** Aspect-based sentiment analysis (ABSA) requires models to bind sentiment evidence to the correct aspect, making it a natural testbed for fine-grained structural reasoning. We introduce GHI, a Graphormer-over-Conditioned-Hypergraph-Incidence framework that is designed as an incidence-based structural reasoning layer built on a bipartite topology. GHI represents diverse linguistic and semantic evidence as token--hyperedge incidence relations, allowing different structural signals to be incorporated through a unified interface. Extensive experiments on six standard ABSA benchmarks show that GHI outperforms all baselines on the SemEval domains, and multi-seed evaluations show stable improvements over strong DeBERTa. Further experiments show that with only 247M parameters, GHI approaches the performance of 11B Flan-T5 based methods on the ISE benchmark. Moreover, it demonstrates strong robustness on the challenging ARTS datasets, maintaining highly competitive performance where traditional models degrade. These results demonstrate that compact structural reasoning remains a valuable alternative to scale-driven approaches for fine-grained tasks.
>
---
#### [new 067] Geometry-Adaptive Explainer for Faithful Dictionary-Based Interpretability under Distribution Shift
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于机器学习可解释性任务，解决模型在分布偏移下解释器不准确的问题。提出几何自适应解释器（GAE），通过调整字典对齐分布偏移区域，提升解释忠实度。**

- **链接: [https://arxiv.org/pdf/2605.21849](https://arxiv.org/pdf/2605.21849)**

> **作者:** Sungjun Lim; Heedong Kim; Andrew Lee; Kyungwoo Song
>
> **摘要:** Mechanistic interpretability aims to explain a model's behavior by identifying causally responsible internal structures. Dictionary-based explainers such as sparse autoencoders and transcoders are a primary tool, but their faithfulness under out-of-distribution (OOD) shift has received little systematic attention. We show that distribution shift rotates the subspace that the model actively uses, misaligning the explainer's dictionary trained on in-distribution (ID) activations. We formalize this misalignment as the faithfulness gap, a geometric distance between the ID dictionary and the OOD-active subspace, and show that it controls OOD faithfulness degradation. To reduce this gap, we propose the Geometry-Adaptive Explainer (GAE), which realigns the explainer's dictionary with the OOD-active subspace while preserving the original feature structure. This requires only unlabeled OOD activations and no gradient updates. We prove that GAE improves over the unadapted ID explainer, with excess loss bounded quadratically by the second-moment shift. Empirically, GAE even matches or surpasses all training-based baselines in causal faithfulness across multiple models and OOD settings.
>
---
#### [new 068] A Tutorial on Diffusion Theory: From Differential Equations to Diffusion Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于生成模型领域，旨在解释扩散模型的理论基础。通过微分方程视角，推导出前向和反向过程，解决模型训练与采样问题。**

- **链接: [https://arxiv.org/pdf/2605.22586](https://arxiv.org/pdf/2605.22586)**

> **作者:** Jiayi Fu; Yuxia Wang
>
> **备注:** A detailed tutorial on Diffusion models and SDE
>
> **摘要:** This tutorial develops diffusion models from the viewpoint of differential equations. We begin with the conditional Gaussian forward process and show that this path admits both an ordinary differential equation (ODE) representation and a stochastic differential equation (SDE) representation. Averaging the conditional process over the data distribution then yields marginalized forward ODE and SDE formulations that transport the data distribution $p_0=p_{\mathrm{data}}$ to a Gaussian prior $p_1=\mathcal{N}(0,I)$. We next derive the corresponding reverse-time dynamics, namely the reverse SDE and the reverse probability-flow ODE, both of which are governed by the marginal score $\grad\log p_t(x)$. This leads to a training objective for score estimation and shows that the standard noise-prediction objective is equivalent to score matching up to an additive constant independent of the model parameters. We then discuss sampling methods for the learned reverse dynamics, including DPM-Solver, as well as guided sampling through classifier guidance and classifier-free guidance. Finally, we compare DDPM and DDIM with the reverse SDE/ODE framework and show that they share the same training objective, while DDPM sampling corresponds to discrete reverse-SDE sampling and DDIM sampling corresponds to reverse-ODE sampling.
>
---
#### [new 069] Energy-Gated Attention: Spectral Salience as an Inductive Bias for Transformer Attention
- **分类: cs.LG; cs.CL; eess.SP**

- **简介: 该论文提出Energy-Gated Attention（EGA），通过谱能量引导注意力机制，解决Transformer中token信息密度不均的问题。任务为自然语言处理中的注意力优化。**

- **链接: [https://arxiv.org/pdf/2605.21842](https://arxiv.org/pdf/2605.21842)**

> **作者:** Athanasios Zeris
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Standard transformer attention computes pairwise similarity between queries and keys, treating all tokens as equally salient regardless of their intrinsic informational content. In turbulent fluid dynamics, coherent structures -- the energetically dominant, spatially organized patterns that persist amid background chaos -- carry a disproportionate fraction of total energy and govern all transport. We propose that tokens play an analogous role in transformer attention: informationally dense positions (morphological boundaries, syntactic heads, discourse markers) concentrate spectral energy and should attract proportionally more attention than background tokens (function words, repeated patterns, low-information filler). We propose Energy-Gated Attention (EGA): a simple modification that gates value aggregation by the spectral energy of key token embeddings, computed by a single learned linear projection that discovers the dominant spectral mode of the embedding field. On TinyShakespeare, EGA achieves +0.103 validation loss improvement with only 12,480 additional parameters (<0.26% overhead) and no measurable computational cost. The result is consistent on Penn Treebank (+0.101), demonstrating dataset independence. A systematic ablation across three wavelet families (fixed Morlet, Daubechies db2/db4, and a parametric Morlet) establishes that fixed structured bases are suboptimal -- the optimal energy direction is data-adaptive and non-sinusoidal -- while identifying learned wavelet packets as a promising open direction. The learned energy threshold converges to tau ~= 0.35 independently of initialization, corresponding to the fraction (~36%) of tokens carrying above-average spectral energy in English text, a stable linguistic property consistent with the fraction of content words in running English text.
>
---
#### [new 070] Efficient Agentic Reasoning Through Self-Regulated Simulative Planning
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于智能体推理任务，解决传统方法规划效率低的问题。提出SR²AM框架，通过模拟推理、自我调节和反应执行三系统实现高效规划。**

- **链接: [https://arxiv.org/pdf/2605.22138](https://arxiv.org/pdf/2605.22138)**

> **作者:** Mingkai Deng; Jinyu Hou; Lara Sá Neves; Varad Pimpalkhute; Taylor W. Killian; Zhengzhong Liu; Eric P. Xing
>
> **备注:** Code and model artifacts are available at this https URL
>
> **摘要:** How should an agent decide when and how to plan? A dominant approach builds agents as reactive policies with adaptive computation (e.g., chain-of-thought), trained end-to-end expecting planning to emerge implicitly. Without control over the presence, structure, or horizon of planning, these systems dramatically increase reasoning length, yielding inefficient token use without reliable accuracy gains. We argue efficient agentic reasoning benefits from decomposing decision-making into three systems: simulative reasoning (System II) grounding deliberation in future-state prediction via a world model; self-regulation (System III) deciding when and how deeply to plan via a learned configurator; and reactive execution (System I) handling fine-grained action. Simulative reasoning provides unified planning across diverse tasks without per-domain engineering, while self-regulation ensures the planner is invoked only when needed. To test this, we develop SR$^2$AM (Self-Regulated Simulative Reasoning Agentic LLM), realizing both as distinct stages within an LLM's chain-of-thought, with the LLM as world model. We explore two instantiations: recording decisions from a prompted multi-module system (v0.1) and reconstructing structured plans from traces of pretrained reasoning LLMs (v1.0), trained via supervised then reinforcement learning (RL). Across math, science, tabular analysis, and web information seeking, v0.1-8B and v1.0-30B achieve Pass@1 competitive with 120-355B and 685B-1T parameter systems respectively, while v1.0-30B uses 25.8-95.3% fewer reasoning tokens than comparable agentic LLMs. RL increases average planning horizon by 22.8% while planning frequency grows only 2.0%, showing it learns to plan further ahead rather than more often. More broadly, learned self-regulation instantiates a principle we expect to extend beyond planning to how agents govern their own learning and adaptation.
>
---
#### [new 071] HealthCraft: A Reinforcement Learning Safety Environment for Emergency Medicine
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出HealthCraft，一个用于急诊医学的强化学习安全环境，解决临床AI评估安全性不足的问题，通过轨迹级安全奖励和多层评估体系提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.21496](https://arxiv.org/pdf/2605.21496)**

> **作者:** Brandon Dent
>
> **备注:** 16 pages, 5 figures, 6 tables. Code, task suite, and Docker bundle: this https URL
>
> **摘要:** Frontier language models are being deployed into clinical workflows faster than the infrastructure to evaluate them safely. Static medical-QA benchmarks miss the failure modes that matter in emergency medicine: trajectory-level safety collapse, tool misuse, and capitulation under sustained clinical pressure. We present HealthCraft, the first public reinforcement-learning environment that rewards trajectory-level safety under realistic emergency-medicine conditions, adapted from Corecraft. It is built on a FHIR R4 world state with 14 entity types and 3,987 seed entities, exposes 24 MCP tools, and defines a dual-layer rubric that zeroes reward whenever any safety-critical criterion is violated. We release 195 tasks across six categories, graded against 2,255 binary criteria (515 safety-critical); a post-hoc 10-task negative-class slate extends this to 205 tasks and 2,337 criteria. V8 results on two frontier models show Claude Opus 4.6 at Pass@1 24.8% [21.5-28.4] and GPT-5.4 at 12.6% [10.2-15.6], with safety-failure rates of 27.5% and 34.0%. On multi-step workflows - the closest proxy to real emergency care - performance collapses to near zero (Claude 1.0%, GPT-5.4 0.0%) despite partial competence on individual steps. Six infrastructure bugs fixed between pilots v2 and v8 re-ordered which model "looks stronger," evidence that infrastructure fidelity is part of the measurement. A deterministic LLM-judge overlay bounds evaluator noise, and a 60-run negative-class smoke pilot shows the reward signal is not drop-in training-safe: restraint criteria pass at 0.929 prevalence, a gameability an eval harness can tolerate but a training reward cannot. We scaffold coupling to a Megatron+SGLang+GRPO loop per Corecraft Section 5.2 and leave training-reward ablations as future work. Environment, tasks, rubrics, and harness are released under Apache 2.0.
>
---
#### [new 072] BEiTScore: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于图像描述评估任务，解决现有评价方法计算成本高或敏感性不足的问题。提出一种高效交叉编码器模型，提升评估精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.21728](https://arxiv.org/pdf/2605.21728)**

> **作者:** Gonçalo Gomes; Bruno Martins; Chrysoula Zerva
>
> **摘要:** Image captioning evaluation remains a significant challenge, as vision-language models evolve toward more challenging capabilities such as generating long-form and context-rich descriptions. State-of-the-art evaluation metrics involve extensive computational costs associated with the use of Large Language Models (LLMs) as judges, or instead suffer from the limitations of standard CLIP-based encoders, such as strict token limits, lack of fine-grained sensitivity, or lack of compositional generalization by treating captions as ``bags-of-words.'' We propose a new learned metric that tackles the aforementioned challenges, based on a lightweight cross-encoder that is initialized from a visual question-answering model checkpoint, balancing a strong weight initialization with computational efficiency. Our training scheme uses a carefully assembled data mixture for supervised learning, featuring adversarial LLM-based data augmentations to enhance model sensitivity to fine-grained visual-linguistic errors. We also introduce a new benchmark designed to assess detailed captioning evaluation across diverse scenarios. Experimental results demonstrate that the proposed metric achieves state-of-the-art performance while maintaining the efficiency required for large-scale benchmarking, quality-aware decoding, or reward guidance.
>
---
#### [new 073] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文探讨如何让语言模型预测研究想法的实验成功率，属于科研评估任务。针对AI生成想法难以人工筛选的问题，提出通过比较评估方法提升模型预测能力。**

- **链接: [https://arxiv.org/pdf/2605.21491](https://arxiv.org/pdf/2605.21491)**

> **作者:** Srujan P Mule; Aniketh Garikaparthi; Manasi Patwardhan
>
> **备注:** ACL 2026 Findings
>
> **摘要:** As language models accelerate scientific research by automating hypothesis generation and implementation, a new bottleneck emerges: evaluating and filtering hundreds of AI-generated ideas without exhaustive experimentation. We ask whether LMs can learn to forecast the empirical success of research ideas before any experiments are run. We study comparative empirical forecasting: given a benchmark-specific research goal and two candidate ideas, predict which will achieve better benchmark performance. We construct a dataset of 11,488 idea pairs grounded in objective outcomes from PapersWithCode. While off-the-shelf 8B-parameter models struggle (30% acc.), SFT dramatically boosts performance to 77.1%, outperforming GPT-5 (61.1%). By framing evaluation as a reasoning task via Reinforcement Learning with Verifiable Rewards (RLVR), we train models to discover latent reasoning paths, achieving 71.35% acc. with interpretable justifications. Through additional ablations and out-of-distribution tests, we show robustness to surface-level heuristics and transfer to both a cross-domain time-split test set and an independently constructed test set. Our results demonstrate that compute-efficient small language models can serve as effective, objective verifiers, offering a scalable path for autonomous scientific discovery.
>
---
#### [new 074] HyLoVQA: Dynamic Hypernetwork-Generated Low-Rank Adaptation for Continual Visual Question Answering
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于持续视觉问答任务，解决模型在连续学习中遗忘旧知识的问题。提出HyLoVQA，通过动态生成低秩适配器和记忆锚点实现高效适应。**

- **链接: [https://arxiv.org/pdf/2605.22035](https://arxiv.org/pdf/2605.22035)**

> **作者:** Yiran Wang; Chenyi Xiong; Ziyue Qin; Miao Zhang; Kui Xiao; Zhifei Li
>
> **备注:** Accepted by IJCAI 2026
>
> **摘要:** Continual Visual Question Answering (VQA) requires learning from non-stationary streams of visual inputs and questions while preserving past knowledge. Most prior methods adapt by updating a largely shared parameter set. This often leads to cross-level task interference, hindering accurate adaptation to the current task and object. To address this limitation, we propose HyLoVQA. It maintains a drift-resilient memory bank of anchors. The bank stores the content of visual objects and textual tasks, and they are updated using current input features. Conditioned on retrieved anchors, a hypernetwork generates lightweight Low-Rank Adaptation (LoRA) adapters. This ensures parameter efficiency, allowing the model to adapt to each task and object dynamically. Additionally, we formulate an alignment loss that aligns semantic discrepancies in the feature space with functional changes in the parameter space, thereby constraining LoRA adapters to remain focused on the current task and object. Extensive experiments on VQA v2 and NExT-QA under both standard and compositional settings demonstrate the superiority of HyLoVQA over prior state-of-the-art methods.
>
---
#### [new 075] Two is better than one: A Collapse-free Multi-Reward RLIF Training Framework
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLIF中因单一奖励导致的稳定性问题。通过引入多奖励机制和正则化方法，提升模型的推理能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.22620](https://arxiv.org/pdf/2605.22620)**

> **作者:** Shourov Joarder; Diganta Sikdar; Ahsan Habib Akash; Binod Bhattarai; Prashnna Gyawali
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has substantially improved the reasoning ability of LLMs, but often depends on external supervision from human annotations or gold-standard solutions. Reinforcement learning from internal feedback (RLIF) has recently emerged as a scalable unsupervised alternative, using signals extracted from the model itself. However, existing RLIF methods typically rely on a single internal reward, which can lead to reward hacking, entropy collapse, and degraded reasoning structure. We propose a multi-reward RLIF framework that decomposes the training signal into two complementary components: an answer-level reward based on cluster voting and a completion-level reward based on token-wise self-certainty. To combine these signals robustly, we apply GDPO-based normalization to reduce reward-scale imbalance. We further introduce KL-Cov regularization, which targets low-entropy token distributions responsible for disproportionate entropy reduction, preserving exploration and preventing late-stage collapse. Across mathematical reasoning and code-generation benchmarks, our method improves stability and robustness over prior unsupervised RL approaches, while achieving performance close to supervised RLVR methods. These results show that complementary internal rewards, combined with targeted regularization, can support stable long-horizon reasoning without relying on external ground-truth supervision. Code will be released soon.
>
---
#### [new 076] EntmaxKV: Support-Aware Decoding for Entmax Attention
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理中的注意力机制优化任务，旨在解决长上下文解码时KV缓存内存带宽瓶颈问题。通过引入EntmaxKV框架，利用熵最大化的稀疏性特性，实现高效解码。**

- **链接: [https://arxiv.org/pdf/2605.21649](https://arxiv.org/pdf/2605.21649)**

> **作者:** Gonçalo Duarte; Miguel Couceiro; Marcos V. Treviso
>
> **摘要:** Long-context decoding is increasingly limited by KV-cache memory traffic since each generated token attends over a cache whose size grows linearly with context length. Existing sparse decoding methods reduce this cost by selecting subsets of tokens or pages, but are designed for softmax attention, whose dense tails make any truncation discard nonzero probability mass. In contrast, $\alpha$-entmax produces exact zeros, turning sparse decoding from dense-tail approximation into support recovery: if the selected candidates contain the entmax support, sparse decoding remains exact. While recent entmax kernels enable efficient training, they do not address the autoregressive decoding bottleneck, where dense inference still streams the full KV cache before sparsity is known. In this work, we introduce EntmaxKV, an entmax-native sparse decoding framework that exploits sparsity before KV pages are loaded. EntmaxKV combines query-aware page scoring, support-aware candidate selection, and sparse entmax attention. We analyze truncation error through the dropped probability mass $\delta$, showing that output error is controlled by $\delta$ and vanishes when the entmax support is recovered. We further introduce a Gaussian-aware entmax selector that estimates the entmax threshold from lightweight page statistics, adapting the selected budget to the score distribution. Empirically, EntmaxKV drops less probability mass, retains more support tokens, and achieves lower output error than softmax-based sparse decoding at matched KV budgets. On long-context and language modeling benchmarks, it closely matches full-cache entmax while using a small fraction of the KV cache, achieving up to $3.36\times$ (softmax) and $5.43\times$ (entmax) speedup over full attention baselines at 1M context length. Code available at: this https URL.
>
---
#### [new 077] Blind Spots in the Guard: How Domain-Camouflaged Injection Attacks Evade Detection in Multi-Agent LLM Systems
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究多智能体LLM系统中领域伪装注入攻击的检测盲区，揭示现有检测器对伪装payload的低检测率，提出Camouflage Detection Gap概念，并评估不同模型的防御效果。**

- **链接: [https://arxiv.org/pdf/2605.22001](https://arxiv.org/pdf/2605.22001)**

> **作者:** Aaditya Pai
>
> **备注:** 8 pages, 3 figures, 2 tables. Submitted to EMNLP 2026 ARR cycle
>
> **摘要:** Injection detectors deployed to protect LLM agents are calibrated on static, template-based payloads that announce themselves as override directives. We identify a systematic blind spot: when payloads are generated to mimic the domain vocabulary and authority structures of the target document, what we call domain camouflaged injection, standard detectors fail to flag them, with detection rates dropping from 93.8% to 9.7% on Llama 3.1 8B and from 100% to 55.6% on Gemini 2.0 Flash. We formalize this as the Camouflage Detection Gap (CDG), the difference in injection detection rate between static and camouflaged payloads. Across 45 tasks spanning three domains and two model families, CDG is large and statistically significant (chi^2 = 38.03, p < 0.001 for Llama; chi^2 = 17.05, p < 0.001 for Gemini), with zero reverse discordant pairs in either case. We additionally evaluate Llama Guard 3, a production safety classifier, which detects zero camouflage payloads (IDRcamouflage = 0.000), confirming that the blind spot extends beyond few-shot detectors to dedicated safety classifiers. We further show that multi-agent debate architectures amplify static injection attacks by up to 9.9x on smaller models, while stronger models show collective resistance. Targeted detector augmentation provides only partial remediation (10.2% improvement on Llama, 78.7% on Gemini), suggesting the vulnerability is architectural rather than incidental for weaker models. Our framework, task bank, and payload generator are released publicly.
>
---
#### [new 078] Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models
- **分类: cs.AI; cs.CL; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于政治演讲中的情感分析任务，旨在比较声学模型与大语言模型在路径学分析中的效果，解决情感识别准确性问题。**

- **链接: [https://arxiv.org/pdf/2605.22732](https://arxiv.org/pdf/2605.22732)**

> **作者:** Juergen Dietrich
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze.
>
---
#### [new 079] X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决跨分词器训练中因词汇不兼容导致的性能下降问题。通过引入X-Token方法，优化了对齐和分布匹配，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2605.21699](https://arxiv.org/pdf/2605.21699)**

> **作者:** Sharath Turuvekere Sreenivas; Adithyakrishna Venkatesh Hanasoge; Mingyu Yang; Ali Taghibakhshi; Saurav Muralidharan; Ashwath Aithal; Pavlo Molchanov
>
> **摘要:** Cross-tokenizer knowledge distillation allows a student model to learn from teachers with incompatible vocabularies. Prior work operates on hidden states or logits; the latter is preferred as a drop-in replacement requiring no auxiliary components. Logit-based methods either use only the correct-token probability, missing the full 'dark knowledge' in the teacher's distribution, or operate on the full output distribution, relying on strict token partitioning and/or unprincipled heuristic ranking. We identify two key shortcomings of full-distribution, logit-based methods: (i) an uncommon-token failure, where critical tokens fall into the unmatched subset (e.g., Llama's 1100 multi-digit numerals under digit-splitting Qwen supervision) and are suppressed during training, reducing GSM8k from 12.89 to 2.56 compared to same-tokenizer KD from a weaker teacher; and (ii) over-conservative matching, where strict 1-to-1 matching excludes near-equivalent tokens across surface forms. These failures require distinct remedies: eliminating the partition when critical tokens are misaligned, and refining it when alignment is reliable. We propose X-Token, an approach with two complementary loss formulations targeting these issues. P-KL removes partitioning and aligns the student's distribution with the teacher's via a sparse projection matrix W (initialized from tokenizer-level string rules) to address the uncommon-token failure. H-KL retains the hybrid form while relaxing matching to align each student token with its top-ranked teacher mapping under W. Both objectives share W and extend naturally to multiple teachers. Empirically, on Llama-3.2-1B, X-Token outperforms the current state of the art GOLD by +3.82 average points with a Qwen3-4B teacher and by +0.5 with a Phi-4-Mini teacher. Further, a two-teacher setup (Phi-4-mini + Llama-3B) improves over single-teacher distillation by +1.3 points.
>
---
#### [new 080] Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究自博弈强化学习中的稳定性问题，针对任务生成与奖励设计的不对称作用进行实验分析，揭示数据筛选机制对系统稳定性的关键影响。**

- **链接: [https://arxiv.org/pdf/2605.22217](https://arxiv.org/pdf/2605.22217)**

> **作者:** Sophia Xiao Pu; Zhaotian Weng; Chengzhi Liu; Jayanth Srinivasa; Gaowen Liu; William Yang Wang; Xin Eric Wang
>
> **摘要:** Self-play reinforcement learning trains language models on their own generated tasks, co-evolving a proposer and solver without human labels. Recent systems report strong reasoning gains, but collapse and instability are widely observed and poorly understood. The dominant response treats this as a reward-design problem. We argue instead that self-play stability is governed by two distinct levers: a data-level gate that decides which proposer-generated tasks enter the training pool, and the reward signal that updates the policy on tasks already admitted. Through controlled experiments on a Python output-prediction task and a deterministic-DSL twin task that strips pretraining priors, output ambiguity, and executor noise, we find the two levers are asymmetric. A strict gate is sufficient for stability under every reward variant we test, including a self-consistency reward with no access to ground truth; while no reward variant is sufficient once the gate is removed. This asymmetry exposes a counter-intuitive coupling we call the Grounded Proposer Paradox: a proposer with ground-truth access accelerates collapse faster than an ungrounded one when paired with a self-consistency solver, by concentrating training on clean tasks that form the fastest path to a spurious self-consistent attractor. Replacing the binary gate with a continuous strictness parameter $\varepsilon$ further reveals a two-stage phase transition: training-side metrics decouple at low $\varepsilon$, while validation accuracy holds until $\varepsilon$ is much higher. Data-level gating, not reward calibration, is the binding constraint on self-play stability.
>
---
#### [new 081] Epicure: Navigating the Emergent Geometry of Food Ingredient Embeddings
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出Epicure，通过多语言食谱数据训练食材嵌入，解决食材间语义关系建模问题。工作包括数据清洗、构建图结构及三种不同随机游走策略的嵌入模型。**

- **链接: [https://arxiv.org/pdf/2605.22391](https://arxiv.org/pdf/2605.22391)**

> **作者:** Jakub Radzikowski; Josef Chen
>
> **摘要:** We present Epicure, a family of three sibling skip-gram ingredient embeddings retrained from scratch on a multilingual recipe corpus. We aggregate 4.14M recipes from 11 sources spanning seven languages, English, Chinese, Russian, Vietnamese, Spanish, Turkish, Indonesian, German, and Indian-English, and normalise the raw ingredient strings to 1,790 canonical entries via an LLM-augmented pipeline. A 203,508-edge ingredient-ingredient NPMI graph and an 80,019-edge typed FlavorDB ingredient-compound graph, 2,247 typed compound nodes across 15 categories, seed three Metapath2Vec variants that share architecture and hyperparameters and differ only in the random-walk schema: Cooc walks the co-occurrence graph only, Chem walks the typed compound metapaths only, and Core blends both via injected ingredient-ingredient walks at controlled mixing, placing each model at a distinct point on the chemistry-vs-recipe-context spectrum.
>
---
#### [new 082] AnyMo: Geometry-Aware Setup-Agnostic Modeling of Human Motion in the Wild
- **分类: cs.CV; cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出AnyMo，解决可穿戴设备在非受控环境下运动建模的问题。通过生成合成数据和跨模态对齐，提升运动识别与理解性能。**

- **链接: [https://arxiv.org/pdf/2605.22715](https://arxiv.org/pdf/2605.22715)**

> **作者:** Baiyu Chen; Zechen Li; Wilson Wongso; Lihuan Li; Xiachong Lin; Hao Xue; Benjamin Tag; Flora Salim
>
> **摘要:** As wearable and mobile devices become increasingly embedded in daily life, they offer a practical way to continuously sense human motion in the wild. But inertial signals are highly dependent on the sensing setup, including body location, mounting position, sensor orientation, device hardware, and sampling protocol. This setup dependence makes it difficult to learn motion representations that transfer across devices and datasets, and limits the broader use of wearable IMUs beyond closed-set recognition. We introduce AnyMo, a geometry-aware framework for setup-agnostic human motion modeling. AnyMo uses physics-grounded IMU simulation over dense body-surface placements to generate diverse and plausible synthetic signals, pre-trains a graph encoder from paired synthetic placement views and masked partial observations, tokenizes multi-position IMU into full-body motion tokens, and aligns these tokens with an LLM for motion-language understanding. We evaluate AnyMo on three complementary tasks: zero-shot activity recognition across 14 unseen downstream datasets, cross-modal retrieval, and wearable IMU motion captioning, where it improves average Accuracy/F1/R@2 by 11.7\%/11.6\%/22.6\% on HAR, increases zero-shot IMU-to-text and text-to-IMU retrieval MRR by 15.9\% and 28.6\%, respectively, and improves zero-shot captioning BERT-F1 by 18.8\%. These results support AnyMo as a generalist model for wearable motion understanding in the wild. Project page: this https URL.
>
---
#### [new 083] Echo: Learning from Experience Data via User-Driven Refinement
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Echo框架，解决AI代理从原始经验数据中学习的效率问题。通过用户反馈提炼高质量训练信号，提升模型性能。任务为持续学习与模型优化。**

- **链接: [https://arxiv.org/pdf/2605.21984](https://arxiv.org/pdf/2605.21984)**

> **作者:** Hande Dong; Xiaoyun Liang; Jiarui Yu; Jiayi Lin; Changqing Ai; Feng Liu; Wenjun Zhang; Rongbi Wei; Chaofan Zhu; Linjie Che; Feng Wu; Xin Shen; Dexu Kong; Xiaotian Wang; Qiuyuan Chen; Bingxu An; Yueting Lei; Qiang Lin
>
> **摘要:** Static "human data" faces inherent limitations: it is expensive to scale and bounded by the knowledge of its creators. Continuous learning from "experience data" - interactions between agents and their environments - promises to transcend these barriers. Today, the widespread deployment of AI agents grants us low-cost access to massive streams of such real-world experience. However, raw interaction logs are inherently noisy, filled with trial-and-error and low information density, rendering them inefficient for direct model training. We introduce Echo, a generalized framework designed to operationalize the transition from raw experience to learnable knowledge, effectively "echoing" environmental feedback back into the training loop for model optimization. In today's agent ecosystem, user refinement serves as a primary source of such feedback: driven by responsibility for the outcome, users rigorously transform flawed agent proposals into verified solutions. These user-driven refinement sequences inherently distill agents' crude attempts into high-quality training signals. Echo systematically harvests these signals to continuously align the agent with real-world needs. Large-scale validation in a production code completion environment confirms that Echo effectively harnesses this pipeline, breaking the static performance ceiling by increasing the acceptance rate from 25.7% to 35.7%.
>
---
#### [new 084] AMEL: Accumulated Message Effects on LLM Judgments
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于语言模型评估任务，研究模型判断是否受先前对话极性影响。通过实验发现模型在不确定情况下更易受历史极性影响，提出使用新上下文或平衡历史以减少偏差。**

- **链接: [https://arxiv.org/pdf/2605.22714](https://arxiv.org/pdf/2605.22714)**

> **作者:** Sid-ali Temkit
>
> **备注:** 19 pages, 14 figures, 6 tables. Single author. Code, data (75,898 deduplicated API responses), and analysis pipeline at this https URL
>
> **摘要:** Large language models are routinely used as automated evaluators: to review code, moderate content, or score outputs, often with many items passing through one conversation. We ask whether the polarity of prior conversation history biases subsequent judgments, an effect we call the accumulated message effect on LLM judgments (AMEL). Across 75,898 API calls to 11 models from 4 providers (OpenAI, Anthropic, Google, and four open-source models), we present identical test items in isolation or following histories saturated with predominantly positive or negative evaluations. Models shift toward the conversation's prevailing polarity (d = -0.17, p < 10^-46). The effect concentrates on items where the model is genuinely uncertain at baseline (d = -0.34 for high-entropy items, vs d = -0.15 when the baseline is deterministic). Bias does not grow with context length: 5 prior turns and 50 produce the same shift (Spearman |r| < 0.01; OLS slope p = 0.80). And there is a negativity asymmetry: paired per item, negative histories induce 1.62x more bias than positive (t = 13.46, p < 10^-39, n = 2,481). Scaling helps but does not solve it (Anthropic: Haiku -0.22 to Opus -0.17; OpenAI: Nano -0.34 to GPT-5.2 -0.17). Three follow-ups narrow the mechanism. The token probability distribution shifts continuously, not at a threshold. The negativity asymmetry has both token-level and semantic components, though attributing the balance is exploratory at our sample sizes. Position does not matter: five biased turns anywhere in a 50-turn history produce the same shift. The simplest fix for evaluation pipelines is a fresh context per item; when batching is unavoidable, balancing the history helps.
>
---
#### [new 085] Maestro: Reinforcement Learning to Orchestrate Hierarchical Model-Skill Ensembles
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Maestro框架，解决多模型与技能协同问题。通过强化学习动态组合模型与技能，提升多模态任务性能。**

- **链接: [https://arxiv.org/pdf/2605.22177](https://arxiv.org/pdf/2605.22177)**

> **作者:** Jinyang Wu; Guocheng Zhai; Ruihan Jin; Yuhao Shen; Zhengxi Lu; Fan Zhang; Haoran Luo; Zheng Lian; Zhengqi Wen; Jianhua Tao
>
> **摘要:** The proliferation of large language models (LLMs) and modular skills has endowed autonomous agents with increasingly powerful capabilities. Existing frameworks typically rely on monolithic LLMs and fixed logic to interface with these skills. This gives rise to a critical bottleneck: different LLMs offer distinct advantages across diverse domains, yet current frameworks fail to exploit the complementary strengths of models and skills, thereby limiting their performance on downstream tasks. In this paper, we present Maestro (Multimodal Agent for Expert-Skill Targeted Reinforced Orchestration), a Reinforcement Learning (RL)-driven orchestration framework that reframes heterogeneous multimodal tasks as a sequential decision-making process over a hierarchical model-skill registry. Rather than consolidating all knowledge into a single model, Maestro trains a lightweight policy to dynamically compose ensembles of frozen expert models and a two-tier skill library, deciding at each step whether to invoke an external expert, which model-skill pair to select, and when to terminate. The policy is optimized via outcome-based RL, requiring no step-level supervision. We evaluate Maestro across ten representative multimodal benchmarks spanning mathematical reasoning, chart understanding, high-resolution perception, and domain-specific analysis. With only a 4B orchestrator, Maestro achieves an average accuracy of 70.1%, surpassing both GPT-5 (69.3%) and Gemini-2.5-Pro (68.7%). Crucially, the learned coordination policy generalizes to unseen models and skills without retraining: augmenting the registry with out-of-domain experts yields a 59.5% average on four challenging benchmarks, outperforming all closed-source baselines. Maestro further maintains high computational efficiency with low latency. The source code is available at this https URL.
>
---
#### [new 086] The Double Dilemma in Multi-Task Radiology Report Generation: A Gradient Dynamics Analysis and Solution
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于多任务放射学报告生成任务，针对线性标量化策略在平衡临床监督与报告平滑性上的不足，提出CAME-Grad优化器以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.22635](https://arxiv.org/pdf/2605.22635)**

> **作者:** Erjian Zhang; Yatong Hao; Liejun Wang; Zhiqing Guo
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While multi-task learning based automatic radiology report generation (RRG) is widely adopted to ensure clinical consistency, most focus on architectural designs yet remain limited to coarse linear scalarization strategies. These strategies cannot effectively balance the hard constraints of discriminative clinical supervision with the smoothness requirements of report generation. To address these problems, we analyze the failure mechanism of linear scalarization from the perspective of gradient dynamics, utilizing the stochastic differential equation (SDE) framework to characterize it as a "Double Dilemma" of drift term deviation and diffusion term decay. Based on this, we propose a backbone-agnostic optimizer named Conflict-Averse Magnitude-Enhanced Gradient Descent (CAME-Grad). Through conflict-averse direction rectification and magnitude-enhanced energy injection, the algorithm not only ensures geometric validity, but also avoids local optimal solutions. Then, the adaptive gradient fusion mechanism is used to establish a dynamic balance between the theoretical optimal direction and the task-specific inductive bias. Experiments show that as a universal plug-and-play optimizer, CAME-Grad brings substantial and consistent improvements across eight diverse RRG methods, elevating overall clinical efficacy performance by an average of 2.3\% on MIMIC-CXR and 1.9\% on IU X-Ray. Our code is available at this https URL.
>
---
#### [new 087] Boundary-targeted Membership Inference Attacks on Safety Classifiers
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于隐私安全任务，研究如何通过边界样本推断安全分类器的训练数据。工作包括提出新策略提升成员推理攻击效果，并验证内容过滤和噪声方法的防护效果。**

- **链接: [https://arxiv.org/pdf/2605.22373](https://arxiv.org/pdf/2605.22373)**

> **作者:** Anthony Hughes; Alexander Goldberg; Prince Jha; Adam Perer; Nikolaos Aletras; Niloofar Mireshghallah
>
> **摘要:** Safety classifiers are essential safeguards within generative AI systems, filtering harmful content or identifying at-risk users when interacting with large language models. Despite their necessity, these models are trained on sensitive datasets including discussions of self-harm and mental health, raising important, yet poorly understood, privacy concerns. Membership inference attacks (MIAs) allow adversaries to infer membership of examples used to train models. In this work, we hypothesize that identifying the examples on which the classifier is least confident are informative for an adversary to infer membership. This reflects a localized failure of generalization, where the model relies on memorization to resolve ambiguity in the training set. To investigate this, we introduce a new boundary-targeted selection strategy that identifies low confidence examples that amplify the signal of an examples membership within a training set. Our experimental results show that an adversary can recover 19\% of the conversations a safety classifier flagged as indicating user distress, at a 5\% false-positive rate, on a classifier fine-tuned for detecting a user who may require emotional support. This is $3.5$ times more than attacking using state-of-the-art MIA methods alone. Finally, we characterize the boundary laying examples and show that content-based filtering is ineffective for protection, and existing noise strategies can effectively mitigate susceptibility of these examples.
>
---
#### [new 088] Detecting Synthetic Political Narratives in Cross-Platform Social Media Discourse
- **分类: cs.SI; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于虚假政治叙事检测任务，旨在解决跨平台合成政治内容的识别问题。通过构建SNC(C)评分框架，结合多种协调信号进行检测。**

- **链接: [https://arxiv.org/pdf/2605.21540](https://arxiv.org/pdf/2605.21540)**

> **作者:** Despoina Antonakaki; Sotiris Ioannidis
>
> **摘要:** The proliferation of large language models has introduced a new paradigm of synthetic political communication in which narratives may be generated, semantically coordinated, and strategically disseminated across platforms at scale. We present a cross-platform framework for detecting synthetic political narratives using four coordination signals -- lexical diversity D(C), temporal burstiness B(C), rhetorical repetition R(C), and semantic homogenization H(C) -- combined into a Synthetic Narrative Coordination Score SNC(C). We apply the framework to a corpus of 353,223 records spanning six geopolitical event windows collected from six Telegram channels and nine Reddit communities (2023--2026). Results show that IntelSlava exhibits the lowest lexical diversity (MATTR 0.52--0.54), the highest burstiness (B=+0.48 to +0.73), and the highest rhetorical overlap with peer channels (Jaccard 0.12), ranking first in the composite SNC(C) on four of six event windows (SNC 0.45--0.60). Rybar ranks last on all windows despite its high semantic homogenization, because its Russian-language output yields high lexical diversity and near-zero rhetorical Jaccard with English-language channels -- demonstrating that no single indicator is sufficient for coordination detection. Multi-dimensional SNC(C) scoring provides a more robust and interpretable signal than any individual metric.
>
---
#### [new 089] Reflecti-Mate: A Conversational Agent for Adaptive Decision-Making Support Through System 1 and System 2 Thinking
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于人机交互任务，旨在解决决策支持系统无法适应个体思维模式的问题。研究设计了一个能促进系统1和系统2思维整合的对话代理，通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.22509](https://arxiv.org/pdf/2605.22509)**

> **作者:** Morita Tarvirdians; Senthil Chandrasegaran; Hayley Hung; Catholijn M. Jonker; Catharine Oertel
>
> **备注:** Accepted at UMAP 2026
>
> **摘要:** Making high-stakes personal decisions involves cognitive, emotional, and intuitive processes, and individuals differ in how they allocate attention across these modes. Integration of these processes has shown to benefit decision making. Yet, most current decision-support systems focus primarily on supporting cognitive aspects, rather than adapting to the individual's thinking profile to support integration of different types of thoughts. In this study, we investigate an agent designed to encourage integration by adapting to the individual user's thought patterns. We explore its effects on participants' perceptions of the agent and their reflective behavior, in comparison with unaided pre-reflection and a baseline agent. In a between-subjects study (N = 128), our agent, which fostered broad and elaborated thinking, enabled more personalized reflective trajectories, elicited more integrative reflective language, and was perceived as providing stronger support for holistic reflection. In contrast, the baseline agent produced homogenized profiles dominated by cognitive language across participants.
>
---
#### [new 090] Structured-Sparse Attention for Entity Tracking with Subquadratic Sequence Complexity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于实体跟踪任务，旨在解决长序列中状态维护与更新的效率问题。通过结构化稀疏注意力机制，实现亚二次复杂度计算，提升运行速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2605.22476](https://arxiv.org/pdf/2605.22476)**

> **作者:** Hangyue Zhao; Paul Caillon; Erwan Fagnou; Alexandre Allauzen
>
> **备注:** 12 pages, 1 figure, 9 tables
>
> **摘要:** Entity tracking requires maintaining and updating latent states for entities and attributes over long sequences. Recent task-specific attention operators can compress deep Transformer stacks into a few layers by performing multi-hop state propagation within a single layer, but their dense evaluation remains expensive. We show that in this setting, learned attention is strongly structured: most mass concentrates in local block-diagonal neighborhoods with a light cross-block residue. Exploiting this, we derive a blockwise evaluation of a resolvent-style operator that keeps within-block interactions exact and routes cross-block interactions through a reduced system. The resulting evaluation is subquadratic in sequence length $O(n^{4/3}d)$ (and $O(n^{7/3})$ when $d\approx n$). On controlled tracking benchmarks, our method matches the dense operator's accuracy while reducing wall-clock time by $12-29\%$ under a standardized measurement protocol, and is up to $2.4 \times$ faster than a compact dense Transformer at comparable exact-match accuracy. We further provide ablations over block size and model capacity, and identify a limitation: performance collapses when the number of simultaneously evolving properties exceeds the number of attention heads.
>
---
#### [new 091] From Reasoning Chains to Verifiable Subproblems: Curriculum Reinforcement Learning Enables Credit Assignment for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型推理任务，解决RLVR在困难问题中效率低、信用分配不足的问题。提出SCRL框架，通过子问题课程提升学习效果。**

- **链接: [https://arxiv.org/pdf/2605.22074](https://arxiv.org/pdf/2605.22074)**

> **作者:** Xitai Jiang; Zihan Tang; Wenze Lin; Yang Yue; Shenzhi Wang; Gao Huang
>
> **摘要:** Reinforcement learning from verifiable rewards (RLVR) has shown strong promise for LLM reasoning, but outcome-based RLVR remains inefficient on hard problems because correct final-answer rollouts are rare and sample-level credit assignment cannot use partial progress in failed attempts. We introduce SCRL (Subproblem Curriculum Reinforcement Learning), a curriculum RL framework that derives verifiable subproblems from reference reasoning chains and fixes the final subproblem as the original problem. This turns partial progress on hard problems into verifiable learning signals. Algorithmically, SCRL uses subproblem-level normalization, which normalizes rewards independently at each subproblem position and assigns the resulting advantages to the corresponding answer spans, enabling finer-grained credit assignment without external rubrics or reward models. Our analysis shows that subproblem curricula lift hard problems out of gradient dead zones, with larger relative gains as the original problem becomes harder. Across seven mathematical reasoning benchmarks, SCRL outperforms strong curriculum-learning baselines, improving average accuracy over GRPO by +4.1 points on Qwen3-4B-Base and +1.9 points on Qwen3-14B-Base. On AIME24, AIME25, and IMO-Bench, SCRL further improves pass@1 by +3.7 points and pass@64 by +4.6 points on Qwen3-4B-Base, indicating better exploration on hard reasoning problems.
>
---
#### [new 092] Planning in the LLM Era: Building for Reliability and Efficiency
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于规划任务，探讨如何在大语言模型时代构建可靠高效的规划器。针对传统方法的不足，提出利用LLM生成可验证的符号解法器，提升规划效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.21902](https://arxiv.org/pdf/2605.21902)**

> **作者:** Michael Katz; Harsha Kokel; Kavitha Srinivas; Shirin Sohrabi
>
> **备注:** Published at ICAPS 2026
>
> **摘要:** Growing attention to intelligent agents has put a spotlight on one of their central capabilities: planning. Early attempts to leverage large language models (LLMs) for planning relied on single-shot plan generation, followed by hybrid approaches that coupled LLMs with limited external search. These methods, unsound and incomplete by their very nature, often require substantial resources without yielding better solutions on unseen problems. As the limitations of LLMs become clearer, recent work has shifted toward using them at solution construction time -- generating symbolic solvers for a family of problems that can be verified and then used efficiently at inference time. This trend reflects the growing need for agents that are both reliable and resource-efficient. It also offers a path towards generating maintainable planners with minimal dependence on language models at inference time. In this paper, we argue that this shift reflects a broader realignment of the planning field in the LLM era. We examine three major categories of planner-generation methods, discuss their current limitations, and outline research steps towards a more reliable and efficient LLM-based generation of planners.
>
---
#### [new 093] Vector Policy Optimization: Training for Diversity Improves Test-Time Search
- **分类: cs.LG; cs.AI; cs.CL; cs.NE**

- **简介: 该论文提出VPO算法，解决语言模型在测试时搜索中缺乏多样性的问题。通过训练策略适应多样化奖励，提升搜索效果。属于强化学习任务。**

- **链接: [https://arxiv.org/pdf/2605.22817](https://arxiv.org/pdf/2605.22817)**

> **作者:** Ryan Bahlous-Boldi; Isha Puri; Idan Shenfeld; Akarsh Kumar; Mehul Damani; Sebastian Risi; Omar Khattab; Zhang-Wei Hong; Pulkit Agrawal
>
> **备注:** 24 pages
>
> **摘要:** Language models must now generalize out of the box to novel environments and work inside inference-scaling search procedures, such as AlphaEvolve, that select rollouts with a variety of task-specific reward functions. Unfortunately, the standard paradigm of LLM post-training optimizes a pre-specified scalar reward, often leading current LLMs to produce low-entropy response distributions and thus to struggle at displaying the diversity that inference-time search will require. We propose Vector Policy Optimization (VPO), an RL algorithm that explicitly trains policies to anticipate diverse downstream reward functions and to produce diverse solutions. VPO exploits that rewards are often vector-valued in practice, like per-test-case correctness in code generation or, say, multiple different user personas or reward models. VPO is essentially a drop-in replacement for the GRPO advantage estimator, but it trains the LLM to output a set of solutions where individual solutions specialize to different trade-offs in the vector reward space. Across four tasks, VPO matches or beats the strongest scalar RL baselines on test-time search (e.g. pass@k and best@k), with the gap widening as the search budget grows. For evolutionary search, VPO models unlock problems that GRPO models cannot solve at all. As test-time search becomes more standardized, optimizing for diversity may need to become the default post-training objective.
>
---
#### [new 094] SpaceDG: Benchmarking Spatial Intelligence under Visual Degradation
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于空间推理任务，旨在解决视觉退化下模型鲁棒性不足的问题。构建了SpaceDG数据集和基准，评估并提升模型在退化场景下的空间智能表现。**

- **链接: [https://arxiv.org/pdf/2605.22536](https://arxiv.org/pdf/2605.22536)**

> **作者:** Xiaolong Zhou; Yifei Liu; Ziyang Gong; Jiarui Li; Qiyue Zhao; Muyao Niu; Yuanyuan Gao; Le Ma; Xue Yang; Hongjie Zhang; Zhihang Zhong
>
> **摘要:** Multimodal Large Language Models (MLLMs) have made rapid progress in spatial intelligence, yet existing spatial reasoning benchmarks largely assume pristine visual inputs and overlook the degradations that commonly occur in real-world deployment, such as motion blur, low light, adverse weather, lens distortion, and compression artifacts. This raises a fundamental question: how robust is the spatial intelligence of current MLLMs when visual observations are imperfect? To answer this question, we introduce SpaceDG, the first large-scale dataset for degradation-aware spatial understanding. It is constructed with a physically grounded degradation synthesis engine that embeds degradation formation process into 3D Gaussian Splatting (3DGS) rendering, enabling realistic simulation of nine degradation types. The resulting dataset contains approximately 1M QA pairs from nearly 1,000 indoor scenes. We further introduce SpaceDG-Bench, an human-verified benchmark with 1,102 questions spanning 11 reasoning categories and 9 visual degradation types, yielding over 10K VQA instances. Evaluating 25 open- and closed-source MLLMs reveals that visual degradations consistently and substantially impair spatial reasoning, exposing a critical robustness gap. Finally, we show that finetuning on SpaceDG markedly improves degradation robustness and can even surpass human performance under degraded conditions without any performance drop on clean images, highlighting the promise of degradation-aware training for robust spatial intelligence.
>
---
#### [new 095] Amplifying, Not Learning: Fine-Tuned AI Text Detectors Amplify a Pretrained Direction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究AI文本检测任务，指出检测器放大预训练典型性轴而非构建边界。通过分析不同模型和数据集，验证了检测机制的共性与可迁移性。**

- **链接: [https://arxiv.org/pdf/2605.21653](https://arxiv.org/pdf/2605.21653)**

> **作者:** Alexander Smirnov
>
> **摘要:** AI text detectors amplify a pretrained typicality axis; they do not construct an AI-vs-human boundary. On raw encoders before any task supervision, projecting onto centroid(AI)-centroid(HC3) achieves NYT-vs-HC3 AUROC 0.806/0.944/0.834 across three architectures (86-106% of the fine-tuned discrimination ceiling: on RoBERTa-base, raw projection exceeds fine-tuning); on RoBERTa-base, full fine-tuning reduces discrimination below raw on both fluent-formal populations tested. The same axis inverts on non-native ESL writing (AUROC 0.06-0.20) -- a falsifiable prediction unique to the typicality reading. A 24-example frozen probe matches full fine-tuning (0.900 vs 0.895). A closed-form Jacobian predictor parameterises axis-manipulating interventions with R^2 = 1.000 universal, lifts ELECTRA-CE deployment TPR from 0.000 to 0.904 at FPR = 1%, and transfers to three independently-trained third-party RoBERTa detectors at 16/16 oracle-equivalence (57% NYT-FPR reduction on the OpenAI detector). Scope: encoder family; mechanism magnitude HC3-anchored; population-level shared axis with per-text mechanisms varying across architectures. Three operationally distinct probes -- text-surface caps_rate residualisation, geometric signed-epsilon ablation, closed-form text-pair predictor -- agree at cos 0.74/0.81/1.00 across three architectures, confirming observer-invariance. Under matched-TPR-0.90 evaluation, the published intervention zoo (CC, dealign-f2c) is calibration-equivalent across 27 cells (|Delta AUROC| <= 0.0081), and >= 97% of the LoRA->full-FT bias gap on ELECTRA is calibration shift, not learned representation -- the central claim's prediction confirmed.
>
---
#### [new 096] Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决现有基准对细粒度时空推理评估不足的问题。通过构建家具组装基准，评估模型在时间顺序、状态定位等任务上的表现。**

- **链接: [https://arxiv.org/pdf/2605.21625](https://arxiv.org/pdf/2605.21625)**

> **作者:** Aditya Chetan; Eric Cai; Peeyush Kushwaha; Bharath Raj Nagoor Kani; Utkarsh Mall; Qianqian Wang; Noah Snavely; Bharath Hariharan
>
> **备注:** CVPR 2026
>
> **摘要:** The emergence of Large Vision-Language Models (LVLMs) has significantly advanced video understanding capabilities. However, existing benchmarks focus predominantly on coarse-grained tasks such as action segmentation, classification, captioning, and retrieval. Furthermore, these benchmarks often rely on entities that can be easily identified verbally, like household objects, animals, human subjects, etc., limiting their applicability to complex, in-the-wild video scenarios. But, many applications such as furniture assembly, cooking, etc., require step-by-step fine-grained spatio-temporal understanding of the video, which is not sufficiently evaluated in current benchmarks. To address this gap, we introduce Flat-Pack Bench, a novel benchmark centered on furniture assembly tasks. Our benchmark evaluates LVLMs on nuanced tasks, including temporal ordering of assembly actions, temporal localization of assembly state, understanding part mating, and tracking, using multiple-choice questions paired with visual prompts highlighting relevant parts as references for fine-grained questions. Our experiments reveal that state-of-the-art LVLMs struggle significantly with fine-grained spatio-temporal reasoning, highlighting their limitations in effectively leveraging temporal information from videos, limited tracking ability, and understanding of spatial interactions like physical contact.
>
---
#### [new 097] MM-Conv: A Multimodal Dataset and Benchmark for Context-Aware Grounding in 3D Dialogue
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言接地任务，解决对话中模糊指代的问题。构建了3D对话基准并提出两阶段接地方法，提升指代理解效果。**

- **链接: [https://arxiv.org/pdf/2605.21796](https://arxiv.org/pdf/2605.21796)**

> **作者:** Anna Deichler; Jim O'Regan; Fethiye Irmak Dogan; Lubos Marcinek; Anna Klezovich; Iolanda Leite; Jonas Beskow
>
> **备注:** Extended version of the paper published at LREC 2026 (Palma de Mallorca, Spain), with expanded VLM baselines and inter-annotator agreement analysis
>
> **摘要:** Grounding language in the physical world requires AI systems to interpret references that emerge dynamically during conversation. While current vision-language models (VLMs) excel at static image tasks, they struggle to resolve ambiguous expressions in spontaneous, multi-turn dialogue. We address this gap by introducing (1) a benchmark for referential communication in dynamic 3D environments, built from 6.7 hours of egocentric VR interaction with synchronized speech, motion, gaze, and 3D scene geometry, and (2) a two-stage grounding pipeline that explicitly resolves conversational ambiguity before visual localization. The benchmark includes over 4,200 manually verified referring expressions spanning full, partitive, and pronominal types. Our contextual rewriting approach improves grounding performance by 11-22 percentage points on average, with a pure detector (GroundingDINO) reaching 56.7% on pronominals after rewriting, nearly double the best end-to-end baseline. Results demonstrate that decoupling linguistic reasoning from visual perception is more effective than end-to-end approaches for conversational grounding.
>
---
#### [new 098] From Parameters to Data: A Task-Parameter-Guided Fine-Tuning Pipeline for Efficient LLM Alignment
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在降低领域适配的数据和计算成本。通过参数与数据的协同优化，提出P2D框架，实现高效模型调整。**

- **链接: [https://arxiv.org/pdf/2605.21558](https://arxiv.org/pdf/2605.21558)**

> **作者:** Hao Chen; Qi Zhang; Liyao Li; Zhanming Shen; Wentao Ye; Lirong Gao; Ningtao Wang; Xing Fu; Xiaoyu Shen; Junbo Zhao
>
> **备注:** Accepted@ICML26, 28 pages, 11 figures, 26 tables
>
> **摘要:** Adapting Large Language Models (LLMs) to specialized domains typically incurs high data and computational overhead. While prior efficiency efforts have largely treated data selection and parameter-efficient fine-tuning as isolated processes, our empirical analysis suggests they may be intrinsically coupled. We posit the Strong Map Hypothesis: a sparse subset of attention heads plays a dominant role in task-specific adaptation, acting as keys that unlock specific data patterns. Building on this observation, we propose From Parameters to Data (P2D), a unified framework that leverages these task-sensitive attention heads as a dual compass for both sample mining and structural pruning. To rigorously quantify the total pipeline cost, we introduce the Alignment Efficiency Ratio (AER) metric for both selection latency and training time. Mechanistically, P2D identifies critical heads via a lightweight proxy and uses them as a functional filter to curate high-affinity data, establishing a synergistic pipeline. Empirically, by updating merely 10% of attention heads on 10% of the data, P2D achieves an 8.3 pp performance gain over strong baselines and delivers a 7.0x end-to-end time speedup. These results validate that precise parameter-data synchronization eliminates redundancy, offering a new paradigm for efficient alignment.
>
---
#### [new 099] Ratchet: A Minimal Hygiene Recipe for Self-Evolving LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Ratchet系统，解决自进化LLM代理的技能生命周期管理问题。通过自我编写、检索、优化和淘汰技能，提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2605.22148](https://arxiv.org/pdf/2605.22148)**

> **作者:** Xing Zhang; Yanwei Cui; Guanghui Wang; Ziyuan Li; Wei Qiu; Bing Zhu; Peiyang He
>
> **备注:** 16 pages, 2 figures, 6 tables. Extends arXiv:2605.19576 with the SWE-bench Verified evaluation and a non-divergence analysis (Proposition 1)
>
> **摘要:** Self-evolving skill libraries, pioneered by Voyager, let frozen LLM agents accumulate reusable knowledge without weight updates, yet recent evaluation shows that LLM-authored skills deliver $+0.0$pp over no-skill baselines while human-curated ones deliver $+16.2$pp: the bottleneck is not skill authoring but lifecycle management. We introduce \textbf{Ratchet}, a single-agent loop in which a frozen LLM writes, retrieves, curates, and retires its own natural-language skills. Ratchet integrates four candidate hygiene mechanisms: outcome-driven retirement, a bounded active-cap, meta-skill authoring guidance, and pattern canonicalisation. On MBPP+ hard-100 with Claude Opus 4.7, Ratchet lifts held-out pass@1 from a $0.258 \pm 0.047$ baseline to a late-window rolling mean of $0.584$ (peak $0.658 \pm 0.042$) across 100 rounds and 3 seeds, a $+0.328 \pm 0.018$ rolling-mean gain where the no-skill control drifts at $+0.002 \pm 0.005$; the same recipe transfers to an agentic solver on SWE-bench Verified ($+0.22$ peak lift over 20 rounds). Eight ablations (A1--A8) reveal that the minimal working recipe is smaller than our design suggests: retirement and the meta-skill authoring prior are load-bearing, while explicit deduplication (canonicalisation, cover-guard) is subsumed by the meta-skill itself. A non-divergence proposition shows that bounded cap and retirement threshold together prevent expected performance from drifting below the no-skills floor.
>
---
#### [new 100] Value-Gradient Hypothesis of RL for LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，研究为何无评价器方法在LLM微调中有效。通过价值梯度视角，分析策略更新机制，提出RL效果的评估准则。**

- **链接: [https://arxiv.org/pdf/2605.21654](https://arxiv.org/pdf/2605.21654)**

> **作者:** Arip Asadulaev; Daniil Ognev; Karim Salta; Martin Takac
>
> **摘要:** Reinforcement learning substantially improves pretrained language models, but it remains understudied why critic-free methods such as PPO and GRPO work as well as they do, and when they should provide the largest gains. We develop a value-gradient perspective of critic-free RL for LLM post-training. First, under a differentiable rollout and additive-noise parameterization, we show that the actor update is value-gradient-like in expectation: the backward pass propagates costates whose conditional expectation equals the value gradient. Second, for discrete transformer policies, we show that autodifferentiation through attention produces empirical costates that approximate this value signal, with an error controlled by the sampling gap and policy entropy. These results motivate a decomposition of RL impact into value gradient signal and reachable reward headroom, yielding a criterion for when RL should be most effective along a pretraining trajectory.
>
---
#### [new 101] Check Your LLM's Secret Dictionary! Five Lines of Code Reveal What Your LLM Learned (Including What It Shouldn't Have)
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型安全分析任务，通过SVD分析LLM的lm_head权重矩阵，揭示模型学习内容及潜在问题，提出VCS和WPS作为评估指标，旨在提升模型可控性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.22005](https://arxiv.org/pdf/2605.22005)**

> **作者:** Hisashi Miyashita
>
> **摘要:** We show that singular value decomposition of the lm_head} weight matrix of a transformer-based large language model -- requiring only five lines of PyTorch and no model inference -- reveals interpretable semantic subspaces directly from the model weights. Each left singular vector identifies the vocabulary tokens most readily selected when the hidden state aligns with the corresponding singular direction; inspecting these clusters exposes the model's training data composition and curation philosophy. Analysing GPT-OSS-120B, Gemma-2-2B, and Qwen2.5-1.5B, we find that singular value spectra and vocabulary cluster structures differ systematically across models: GPT exhibits a graduated hierarchy of functionally differentiated subspaces; Gemma is dominated by pre-nineteenth-century English orthography, forming a stepwise clustering structure that may contribute to high output controllability; and Qwen exhibits broad multilingual coverage alongside subspaces whose vocabulary the authors have determined to be ethically inappropriate for direct publication. Base-instruct comparison reveals that ethically concerning subspaces originate in pretraining and are not removed by post-training alignment. We introduce the Vocabulary Cluster Score (VCS) to quantify subspace coherence, and the Weighted Projection Score (WPS) as a static glitch token detector; applying WPS to GPT-OSS-120B recovers shokubutsu-hyakka-tsu (ID 137606), a well-known glitch token widely reported in the CJK language community, without any model inference. We propose a taxonomy of root causes for problematic vocabulary content and call for lm_head} SVD analysis to be adopted as a standard pre-release safety auditing step. Our findings further suggest directions toward SVD-guided tokenizer optimisation and more controllable LLM design.
>
---
#### [new 102] Search-E1: Self-Distillation Drives Self-Evolution in Search-Augmented Reasoning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在提升搜索增强推理模型的性能。通过自蒸馏和GRPO结合的方法，无需外部监督即可实现模型自我进化。**

- **链接: [https://arxiv.org/pdf/2605.22511](https://arxiv.org/pdf/2605.22511)**

> **作者:** Zihan Liang; Yufei Ma; Ben Chen; Zhipeng Qian; Xuxin Zhang; Huangyu Dai; Lingtao Mao
>
> **摘要:** Post-training has become the dominant recipe for turning a language model into a competent search-augmented reasoning agent. A line of recent work pushes its performance further by adding elaborate machinery on top of this standard pipeline. These augmentations import external supervision from stronger external systems, attach auxiliary modules such as process reward models or retrospective critics, restructure the rollout itself with tree search or multi-stage curricula, or shape the reward with hand-crafted bonuses and penalties. Each addition delivers a measurable gain, but each also inflates the training pipeline and ties the recipe to resources or designs that may not always be available. We take a step back and ask whether any of this machinery is actually necessary, and propose Search-E1, a self-evolution method that lets a search-augmented agent improve through only vanilla GRPO interleaved with offline self-distillation (OFSD). After each GRPO round, the policy rolls out on its own training questions. A token-level forward KL objective then aligns the policy's inference-time distribution to its own distribution under a privileged context that exposes a more efficient sibling trajectory. Despite this simplicity, the procedure naturally provides dense per-step supervision. On seven QA benchmarks, Search-E1 reaches $0.440$ average EM with Qwen2.5-3B, surpassing all open-source baselines at both scales. Code and complete version will be made public soon.
>
---
#### [new 103] Why Semantic Entropy Fails: Geometry-Aware and Calibrated Uncertainty for Policy Optimization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决后训练阶段不确定性信号不准确的问题。通过分析现有方法的不足，提出GCPO框架，提升梯度稳定性与优化效果。**

- **链接: [https://arxiv.org/pdf/2605.21801](https://arxiv.org/pdf/2605.21801)**

> **作者:** Zheyuan Zhang; Kaiwen Shi; Han Bao; Zehong Wang; Tianyi Ma; Yanfang Ye
>
> **摘要:** Post-training has become central to improving reasoning and alignment in large language models, where critic-free models enable scalable learning from model-generated outputs but lack principled mechanisms to distinguish informative from noisy signals. Recent approaches leverage response-level measures as uncertainty signals to regulate group-based optimization methods such as GRPO. Yet their empirical success remains unstable and unclear in how they influence optimization dynamics. In this paper, we provide, to our knowledge, the first principled formulation that interprets uncertainty signals as mechanisms for characterizing and regulating gradient variance and learning signal quality. Based on both empirical and theoretical analysis, we identify two critical gaps of current entropy-based estimators: The anisotropic gap and The calibration gap. Motivated by this analysis, we propose Geometric-aware Calibrated Policy Optimization (GCPO), a novel framework integrating geometry-aware measures to capture semantic disagreement with reward-based calibration to align uncertainty with learning signal strength. Experiments on multiple benchmarks show that our approach more faithfully tracks gradient variability and consistently improves post-training performance. Our results highlight the importance of designing uncertainty signals that are aligned with optimization dynamics, offering a principled perspective for robust post-training.
>
---
## 更新

#### [replaced 001] InnerQ: Hardware-Aware Tuning-Free Quantization of KV Cache for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大语言模型优化任务，旨在解决KV缓存带来的硬件成本高问题。通过提出InnerQ方法，实现无调优的量化压缩，提升解码效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2602.23200](https://arxiv.org/pdf/2602.23200)**

> **作者:** Sayed Mohammadreza Tayaranian Hosseini; Amir Ardakani; Warren J. Gross
>
> **备注:** 18 pages, 5 figures, 7 tables
>
> **摘要:** When transformer-based language models are deployed for text generation, most of the inference time is spent in the decoding stage, where output tokens are generated sequentially. Reducing the hardware cost of each decoding step is therefore critical for efficient long-context generation. A major bottleneck is the key-value (KV) cache, whose size grows with sequence length and often dominates the model's memory footprint. Prior work has proposed quantization methods to compress the KV cache while minimizing its loss of precision. We present InnerQ, a hardware-aware KV cache quantization scheme that reduces decode latency without compromising evaluation performance. InnerQ performs group-wise quantization by grouping cache matrices along their inner dimension. This grouping strategy aligns dequantization with vector-matrix multiplication and increases data reuse across GPU compute units. As a result, InnerQ reduces memory access and accelerates dequantization, achieving an average $1.3\times$ speedup over prior KV cache quantization methods and $2.7\times$ over the non-quantized baseline. To maintain fidelity under aggressive compression, InnerQ incorporates three techniques: (i) hybrid quantization, which chooses symmetric or asymmetric quantization for each group based on local statistics; (ii) high-precision windows for both recent tokens and attention sink tokens to mitigate outlier leakage; and (iii) per-channel normalization of the key cache, computed once during prefill and folded into the model parameters to eliminate runtime overhead. Beyond reducing latency, experiments on Llama and Mistral models show that InnerQ also improves few-shot evaluation scores relative to prior KV cache quantization methods.
>
---
#### [replaced 002] MixSD: Mixed Contextual Self-Distillation for Knowledge Injection
- **分类: cs.CL**

- **简介: 该论文提出MixSD方法，用于知识注入任务，解决微调导致模型能力退化的问题。通过动态混合模型自身条件，保持分布一致，提升记忆与保留的平衡。**

- **链接: [https://arxiv.org/pdf/2605.16865](https://arxiv.org/pdf/2605.16865)**

> **作者:** Jiarui Liu; Lechen Zhang; Yongjin Yang; Yinghui He; Yingheng Wang; Weihao Xuan; Zhijing Jin; Mona Diab
>
> **摘要:** Supervised fine-tuning (SFT) is widely used to inject new knowledge into language models, but it often degrades pretrained capabilities such as reasoning and general-domain performance. We argue this forgetting arises because fine-tuning targets from humans or external systems diverge from the model's autoregressive distribution, forcing the optimizer to imitate low-probability token sequences. To address this problem, we propose MixSD, a simple external-teacher-free method for distribution-aligned knowledge injection. Instead of training on fixed targets, MixSD constructs supervision dynamically by mixing tokens from two conditionals of the base model itself: an expert conditional that observes the injected fact in context, and a naive conditional that reflects the model's original prior. The resulting supervision sequences preserve the factual learning signal while remaining substantially closer to the base model's distribution. We evaluate MixSD on two synthetic corpora that we construct to study factual recall and arithmetic function acquisition in a controlled setting, together with established benchmarks for open-domain factual question answering and knowledge editing. Across multiple model scales and settings, MixSD consistently achieves a better memorization-retention trade-off compared to SFT and on-policy self distillation baselines, retaining up to 100% of the base model's held-out capability while maintaining near-perfect training accuracy, whereas standard SFT retains as little as 1%. We further show that MixSD produces substantially lower-NLL supervision targets under the base model and reduces harmful movement along Fisher-sensitive parameter directions. These results suggest that aligning supervision with the model's native generation distribution is a simple and effective principle for knowledge injection that mitigates catastrophic forgetting.
>
---
#### [replaced 003] Unifying Masked Diffusion Models with Various Generation Orders and Beyond
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言生成任务，解决生成顺序影响质量的问题。提出OeMDM和LoMDM，统一多种生成顺序并联合学习生成顺序与模型。**

- **链接: [https://arxiv.org/pdf/2602.02112](https://arxiv.org/pdf/2602.02112)**

> **作者:** Chunsan Hong; Sanghyun Lee; Jong Chul Ye
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Masked diffusion models (MDMs) are a potential alternative to autoregressive models (ARMs) for language generation, but generation quality depends critically on the generation order. Prior work either hard-codes an ordering (e.g., blockwise left-to-right) or learns an ordering policy for a pretrained MDM, which incurs extra cost and can yield suboptimal solutions due to the two-stage optimization. Motivated by this, we propose order-expressive masked diffusion model (OeMDM) for a broad class of diffusion generative processes with various generation orders, enabling the interpretation of MDM, ARM, and block diffusion in a single framework. Furthermore, building on OeMDM, we introduce learnable-order masked diffusion model (LoMDM), which jointly learns the generation ordering and diffusion backbone through a single objective from scratch, enabling the diffusion model to generate text in context-dependent ordering. Empirically, we confirm that LoMDM outperforms various discrete diffusion models across multiple language modeling benchmarks.
>
---
#### [replaced 004] Towards Generalization of Block Attention via Automatic Segmentation and Block Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决块注意力机制在长文本中的应用问题。通过自动分割和块蒸馏方法，提升块注意力的性能与效率。**

- **链接: [https://arxiv.org/pdf/2605.15913](https://arxiv.org/pdf/2605.15913)**

> **作者:** Shuaiyi Li; Zhisong Zhang; Yan Wang; Lei Zhu; Dongyang Ma; Chenlong Deng; Yang Deng; Wai Lam
>
> **备注:** 16 pages, 2 figures
>
> **摘要:** Block attention, which processes the input as separate blocks that cannot attend to one another, offers significant potential to improve KV cache reuse in long-context scenarios such as Retrieval-Augmented Generation (RAG). However, its broader application is hindered by two key challenges: the difficulty of segmenting input text into meaningful, self-contained blocks, and the inefficiency of existing block fine-tuning methods that risk degrading performance. To address these, we first construct SemanticSeg, a large and diverse semantic segmentation dataset containing over 30k instances across 16 categories-including books, code, web text, and conversations with text lengths ranging from 2k to 32k. Using this dataset, we train a lightweight segmenter to automatically partition text into human-instinct-aligned blocks with controllable granularity. Second, we propose block distillation, a training framework that is more efficient than block fine-tuning, which uses a frozen full-attention teacher model to guide the block-attention student. This framework integrates three novel components: block sink tokens to mitigate information loss at block boundaries, block dropout to leverage training signals from all blocks, and token-level loss weighting to focus learning on block-attention-sensitive tokens. Experiments across multiple models and benchmarks demonstrate that our segmenter outperforms heuristic and statistical baselines, and block distillation achieves near-full-attention performance under block attention, establishing a practical and scalable pathway for deploying block attention.
>
---
#### [replaced 005] Beyond Benchmark Islands: Toward Representative Trustworthiness Evaluation for Agentic AI
- **分类: cs.CL; cs.DB**

- **简介: 该论文属于AI可信度评估任务，旨在解决现有评估方法碎片化和缺乏代表性的问题。提出五属性可信度定义及HAAF框架，实现跨模型系统提升。**

- **链接: [https://arxiv.org/pdf/2603.14987](https://arxiv.org/pdf/2603.14987)**

> **作者:** Jinhu Qi; Yifan Li; Minghao Zhao; Wentao Zhang; Zijian Zhang; Yaoman Li; Irwin King
>
> **备注:** 9 pages, 3 figures, 8 tables. Submitted to the Agent4IR Workshop at KDD 2026
>
> **摘要:** Agentic AI systems increasingly act through tool-augmented, multi-step workflows whose failures (unsafe tool use, unauthorised actions, social harm) carry deployment-level consequences. Evaluation practice remains fragmented across isolated benchmark slices, and "trustworthiness" is frequently invoked but rarely defined operationally. We argue the central limitation is twofold: (i) the absence of a measurable specification of what agent trustworthiness means, and (ii) the lack of a principled notion of representativeness allowing assessment over a socio-technical scenario distribution rather than disconnected benchmark instances. We address (i) by defining agentic trustworthiness as a five-property profile (Reliability, Robustness, Safety, Social-Ethical Alignment, Operational Integrity) grounded in current AI risk frameworks, and (ii) with the Holographic Agent Assessment Framework (HAAF), which measures this profile over a scenario manifold through static policy analysis, sandbox simulation, social-ethical alignment assessment, and distribution-aware sampling, connected through an iterative Trustworthy Optimization Factory that converts red-team diagnoses into blue-team interventions. Our contributions are: (1) an operational five-property definition of agentic trustworthiness; (2) a distribution-aware scenario-sampling framework that surfaces property-level trade-offs invisible to scalar leaderboards; and (3) a cross-family transfer experiment in which interventions designed from a single focal model generalise -- without per-model or per-scenario tuning -- to 13 systems from seven model families (Llama, Mistral, Kimi, GLM, Qwen, GPT, DeepSeek) on a 100-scenario suite, where all 13 systems improve and two reach a perfect risk-weighted profile, establishing HAAF's Factory as a model-agnostic deployment-readiness pipeline. Code: this https URL
>
---
#### [replaced 006] SiameseNorm: Breaking the Barrier to Reconciling Pre/Post-Norm
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于Transformer架构优化任务，解决Pre-Norm与Post-Norm的兼容性问题。提出SiameseNorm架构，实现两者优势结合，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.08064](https://arxiv.org/pdf/2602.08064)**

> **作者:** Tianyu Li; Dongchen Han; Zixuan Cao; Haofeng Huang; Mengyu Zhou; Ming Chen; Erchao Zhao; Xiaoxi Jiang; Guanjun Jiang; Gao Huang
>
> **备注:** Accepted to ICML 2026; camera-ready version; revised presentation and added additional experimental results
>
> **摘要:** The long-standing tension between Pre- and Post-Norm remains an open problem in Transformer architecture, reflecting a fundamental trade-off between training stability and representational capacity. Prior attempts to combine their strengths have made progress, but often show limited robustness across training settings, restricting their broader applicability. We revisit this dilemma, showing that single-stream architectures struggle to reconcile Pre-Norm's stable identity-gradient propagation with Post-Norm's normalization of the main residual path. To address this structural tension, we propose SiameseNorm, a simple yet effective two-stream architecture that remains compatible with Pre-Norm training recipes. SiameseNorm couples Pre-Norm-like and Post-Norm-like streams through shared residual blocks, allowing each residual block to receive optimization signals from both pathways with negligible overhead. Extensive experiments on 400M and 1.3B dense language models, 15B MoE models, Vision Transformers, and Diffusion Transformers show that SiameseNorm consistently improves performance while maintaining strong training stability across architectures and modalities. Code is available at this https URL.
>
---
#### [replaced 007] RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决agentic RAG系统中多跳问题与中间推理能力不足的问题。通过构建RAGCap-Bench基准，评估并提升模型的中间能力。**

- **链接: [https://arxiv.org/pdf/2510.13910](https://arxiv.org/pdf/2510.13910)**

> **作者:** Jingru Lin; Chen Zhang; Stephen Y. Liu; Haizhou Li
>
> **摘要:** Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities.
>
---
#### [replaced 008] ImProver: Agent-Based Automated Proof Optimization
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文属于自动化证明优化任务，旨在通过改进的LLM代理ImProver，将形式化证明按用户定义标准（如长度、可读性）进行优化。**

- **链接: [https://arxiv.org/pdf/2410.04753](https://arxiv.org/pdf/2410.04753)**

> **作者:** Riyaz Ahuja; Jeremy Avigad; Prasad Tetali; Sean Welleck
>
> **备注:** Published as a conference paper at ICLR 2025
>
> **摘要:** Large language models (LLMs) have been used to generate formal proofs of mathematical theorems in proofs assistants such as Lean. However, we often want to optimize a formal proof with respect to various criteria, depending on its downstream use. For example, we may want a proof to adhere to a certain style, or to be readable, concise, or modularly structured. Having suitably optimized proofs is also important for learning tasks, especially since human-written proofs may not optimal for that purpose. To this end, we study a new problem of automated proof optimization: rewriting a proof so that it is correct and optimizes for an arbitrary criterion, such as length or readability. As a first method for automated proof optimization, we present ImProver, a large-language-model agent that rewrites proofs to optimize arbitrary user-defined metrics in Lean. We find that naively applying LLMs to proof optimization falls short, and we incorporate various improvements into ImProver, such as the use of symbolic Lean context in a novel Chain-of-States technique, as well as error-correction and retrieval. We test ImProver on rewriting real-world undergraduate, competition, and research-level mathematics theorems, finding that ImProver is capable of rewriting proofs so that they are substantially shorter, more modular, and more readable.
>
---
#### [replaced 009] Optimus: A Robust Defense Framework for Mitigating Toxicity while Fine-Tuning Conversational AI
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于对话AI安全任务，旨在解决微调过程中毒性行为注入问题。提出Optimus框架，通过无训练分类和双策略对齐，有效提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2507.05660](https://arxiv.org/pdf/2507.05660)**

> **作者:** Aravind Cheruvu; Shravya Kanchi; Sifat Muhammad Abdullah; Nicholas Ka-Shing Kong; Daphne Yao; Murtuza Jadliwala; Bimal Viswanath
>
> **备注:** Accepted at ACM CODASPY 2026
>
> **摘要:** Customizing Large Language Models (LLMs) on untrusted datasets poses severe risks of injecting toxic behaviors. In this work, we introduce Optimus, a novel defense framework designed to mitigate fine-tuning harms while preserving conversational utility. Unlike existing defenses that rely heavily on precise toxicity detection or restrictive filtering, Optimus addresses the critical challenge of ensuring robust mitigation even when toxicity classifiers are imperfect or biased. Optimus integrates a training-free toxicity classification scheme that repurposes the safety alignment of commodity LLMs, and employs a dual-strategy alignment process combining synthetic "healing data" with Direct Preference Optimization (DPO) to efficiently steer models toward safety. Extensive evaluations demonstrate that Optimus mitigates toxicity even when relying on extremely biased classifiers (with up to 85% degradation in Recall). Optimus outperforms the state-of-the-art defense StarDSS and exhibits strong resilience against adaptive adversarial and jailbreak attacks. Our source code and datasets are available at this https URL
>
---
#### [replaced 010] Sub-exponential Growth Dynamics in Complex Systems: A Piecewise Power-Law Model for the Diffusion of New Words and Names
- **分类: physics.soc-ph; cs.CL; cs.CY; stat.AP**

- **简介: 该论文属于社会扩散建模任务，旨在解决传统S型模型无法准确描述社会现象的问题。通过构建分段幂律模型，分析语言扩散数据，揭示子指数增长的普遍性。**

- **链接: [https://arxiv.org/pdf/2511.04106](https://arxiv.org/pdf/2511.04106)**

> **作者:** Hayafumi Watanabe
>
> **摘要:** The diffusion of ideas and language in society has conventionally been described by S-shaped models, such as the logistic curve. However, the role of sub-exponential growth -- a slower-than-exponential pattern known in epidemiology -- has been largely overlooked in broader social phenomena. Here, we present a piecewise power-law model to characterize complex growth curves with a few parameters. We systematically analyzed a large-scale dataset of approximately one billion Japanese blog articles linked to Wikipedia vocabulary, and observed consistent patterns in web search trend data (English, Spanish, and Japanese). Our analysis of 2,963 items, selected for reliable estimation (e.g., sufficient duration/peak, monotonic growth), reveals that 1,625 (55%) diffusion patterns without abrupt level shifts were adequately described by one or two segments. For single-segment curves, we found that (i) the mode of the shape parameter $\alpha$ was near 0.5, indicating prevalent sub-exponential growth; (ii) the peak diffusion scale is primarily determined by the growth rate $R$, with minor contributions from $\alpha$ or the duration $T$; and (iii) $\alpha$ showed a tendency to vary with the nature of the topic, being smaller for niche/local topics and larger for widely shared ones. Furthermore, a micro-behavioral model of outward (stranger) vs. inward (community) contact suggests that $\alpha$ can be interpreted as an index of the preference for outward-oriented communication. These findings suggest that sub-exponential growth is a common pattern of social diffusion, and our model provides a practical framework for consistently describing, comparing, and interpreting complex and diverse growth curves.
>
---
#### [replaced 011] Orchard: An Open-Source Agentic Modeling Framework
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Orchard框架，解决开放源代码中代理建模的基础设施和训练不足问题，通过构建轻量环境层和三种代理建模方案，提升任务处理能力。**

- **链接: [https://arxiv.org/pdf/2605.15040](https://arxiv.org/pdf/2605.15040)**

> **作者:** Baolin Peng; Wenlin Yao; Qianhui Wu; Hao Cheng; Xiao Yu; Rui Yang; Tao Ge; Alessandro Sordoni; Xingdi Yuan; Yelong Shen; Pengcheng He; Tong Zhang; Zhou Yu; Jianfeng Gao
>
> **摘要:** Agentic modeling aims to transform LLMs into autonomous agents capable of solving complex tasks through planning, reasoning, tool use, and multi-turn interaction with environments. Despite major investment, open research remains constrained by infrastructure and training gaps. Many high-performing systems rely on proprietary codebases, models, or services, while most open-source frameworks focus on orchestration and evaluation rather than scalable agent training. We present Orchard, an open-source framework for scalable agentic modeling. At its core is Orchard Env, a lightweight environment service providing reusable primitives for sandbox lifecycle management across task domains, agent harnesses, and pipeline stages. On top of Orchard Env, we build three agentic modeling recipes. Orchard-SWE targets coding agents. We distill 107K trajectories from MiniMax-M2.5 and Qwen3.5-397B, introduce credit-assignment SFT to learn from productive segments of unresolved trajectories, and apply Balanced Adaptive Rollout for RL. Starting from Qwen3-30B-A3B-Thinking, Orchard-SWE achieves 64.3% on SWE-bench Verified after SFT and 67.5% after SFT+RL, setting a new state of the art among open-source models of comparable size. Orchard-GUI trains a 4B vision-language computer-use agent using only 0.4K distilled trajectories and 2.2K open-ended tasks. It achieves 74.1%, 67.0%, and 64.0% success rates on WebVoyager, Online-Mind2Web, and DeepShop, respectively, making it the strongest open-source model while remaining competitive with proprietary systems. Orchard-Claw targets personal assistant agents. Trained with only 0.2K synthetic tasks, it achieves 59.6% pass@3 on Claw-Eval and 73.9% when paired with a stronger ZeroClaw harness. Collectively, these results show that a lightweight, open, harness-agnostic environment layer enables reusable agentic data, training recipes, and evaluations across domains.
>
---
#### [replaced 012] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究模型融合任务，解决共享知识过量累积问题。提出SVC方法，通过调整奇异值平衡谱分布，提升融合效果。**

- **链接: [https://arxiv.org/pdf/2602.05536](https://arxiv.org/pdf/2602.05536)**

> **作者:** Yayuan Li; Ze Peng; Jian Zhang; Jintao Guo; Yue Duan; Yinghuan Shi
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Model merging combines multiple fine-tuned models into a single model by adding their weight updates, providing a lightweight alternative to retraining. Existing methods primarily target resolving conflicts between task updates, leaving the failure mode of over-counting shared knowledge unaddressed. We show that when tasks share aligned spectral directions (i.e., overlapping singular vectors), a simple linear combination repeatedly accumulates these directions, inflating the singular values and biasing the merged model toward shared subspaces. To mitigate this issue, we propose Singular Value Calibration (SVC), a training-free and data-free post-processing method that quantifies subspace overlap and rescales inflated singular values to restore a balanced spectrum. Across vision and language benchmarks, SVC consistently improves strong merging baselines and achieves state-of-the-art performance. Furthermore, by modifying only the singular values, SVC improves the performance of Task Arithmetic by 13.0%. Code is available at this https URL.
>
---
#### [replaced 013] MAP4TS: A Multi-Aspect Prompting Framework for Time-Series Forecasting with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于时间序列预测任务，旨在解决LLM在处理时间序列数据时忽视统计特性与时间依赖性的问题。提出MAP4TS框架，通过多方面提示设计提升预测性能。**

- **链接: [https://arxiv.org/pdf/2510.23090](https://arxiv.org/pdf/2510.23090)**

> **作者:** Suchan Lee; Jihoon Choi; Sohyeon Lee; Minseok Song; Bong-Gyu Jang; Hwanjo Yu; Soyeon Caren Han
>
> **备注:** There is a error in modeling. Thereafter, paper will be revised and re-uploaded
>
> **摘要:** Recent advances have investigated the use of pretrained large language models (LLMs) for time-series forecasting by aligning numerical inputs with LLM embedding spaces. However, existing multimodal approaches often overlook the distinct statistical properties and temporal dependencies that are fundamental to time-series data. To bridge this gap, we propose MAP4TS, a novel Multi-Aspect Prompting Framework that explicitly incorporates classical time-series analysis into the prompt design. Our framework introduces four specialized prompt components: a Global Domain Prompt that conveys dataset-level context, a Local Domain Prompt that encodes recent trends and series-specific behaviors, and a pair of Statistical and Temporal Prompts that embed handcrafted insights derived from autocorrelation (ACF), partial autocorrelation (PACF), and Fourier analysis. Multi-Aspect Prompts are combined with raw time-series embeddings and passed through a cross-modality alignment module to produce unified representations, which are then processed by an LLM and projected for final forecasting. Extensive experiments across eight diverse datasets show that MAP4TS consistently outperforms state-of-the-art LLM-based methods. Our ablation studies further reveal that prompt-aware designs significantly enhance performance stability and that GPT-2 backbones, when paired with structured prompts, outperform larger models like LLaMA in long-term forecasting tasks.
>
---
#### [replaced 014] LLM Agents Already Know When to Call Tools -- Even Without Reasoning
- **分类: cs.CL**

- **简介: 该论文属于AI代理任务，解决LLM代理过度调用工具的问题。通过构建基准测试和提出Probe&Prefill方法，有效减少不必要的工具调用，提升效率。**

- **链接: [https://arxiv.org/pdf/2605.09252](https://arxiv.org/pdf/2605.09252)**

> **作者:** Chung-En Sun; Linbo Liu; Ge Yan; Zimo Wang; Tsui-Wei Weng
>
> **摘要:** Tool-augmented LLM agents tend to call tools indiscriminately, even when the model can answer directly. Each unnecessary call wastes API fees and latency, yet no existing benchmark systematically studies when a tool call is actually needed. We propose When2Tool, a benchmark of 18 environments (15 single-hop, 3 multi-hop) spanning three categories of tool necessity -- computational scale, knowledge boundaries, and execution reliability -- each with controlled difficulty levels that create a clear decision boundary between tool-necessary and tool-unnecessary tasks. We evaluate two families of training-free baselines: Prompt-only (varying the prompt to discourage unnecessary calls) and Reason-then-Act (requiring the model to reason about tool necessity before acting). Both provide limited control: Prompt-only suppresses necessary calls alongside unnecessary ones, and Reason-then-Act still incurs a disproportionate accuracy cost on hard tasks. To understand why these baselines fail, we probe the models' hidden states and find that tool necessity is linearly decodable from the pre-generation representation with AUROC 0.89--0.96 across six models, substantially exceeding the model's own verbalized reasoning. This reveals that models already know when tools are needed, but fail to act on this knowledge during generation. Building on this finding, we propose Probe&Prefill, which uses a lightweight linear probe to read the hidden-state signal and prefills the model's response with a steering sentence. Across all models tested, Probe&Prefill reduces tool calls by 48% with only 1.7% accuracy loss, while the best baseline at comparable accuracy only reduces 6% of tool calls, or achieves a similar tool call reduction but incurs a 5$\times$ higher accuracy loss. Our code is available at this https URL
>
---
#### [replaced 015] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型训练任务，旨在解决大模型推理效率低的问题。通过小模型指导大模型，提升其推理能力并减少资源消耗。**

- **链接: [https://arxiv.org/pdf/2510.07962](https://arxiv.org/pdf/2510.07962)**

> **作者:** Jingyuan Wang; Yankai Chen; Zhonghang Li; Chao Huang
>
> **备注:** Updated to ACL 2026 camera-ready version with improved method presentation, expanded related work discussion, additional analyses, and presentation refinements
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable progress in reasoning, often through supervised fine-tuning (SFT). However, SFT is resource-intensive, relying on large curated datasets, rejection-sampled demonstrations, and uniform optimization across all tokens, even though only a fraction carry meaningful learning value. In this work, we explore a counterintuitive idea: can smaller language models (SLMs) teach larger language models (LLMs) by revealing high-value reasoning moments that reflect the latter's unique strength? We propose LightReasoner, a novel framework that leverages the behavioral divergence between a stronger expert model (LLM) and a weaker amateur model (SLM). LightReasoner operates in two stages: (1) a sampling stage that pinpoints critical reasoning moments and constructs supervision examples capturing the expert's advantage through expert-amateur contrast, and (2) a fine-tuning stage that aligns the expert model with these distilled examples, amplifying its reasoning strengths. Across seven mathematical benchmarks, LightReasoner improves accuracy by up to 28.1%, while reducing time consumption by 90%, sampled problems by 80%, and tuned token usage by 99%, all without relying on ground-truth labels. By turning weaker SLMs into effective teaching signals, LightReasoner offers a scalable and resource-efficient approach for advancing LLM reasoning. Code is available at: this https URL
>
---
#### [replaced 016] UniSD: Towards a Unified Self-Distillation Framework for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大语言模型自蒸馏中的监督可靠性与训练稳定性问题。提出UniSD框架，整合多种机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.06597](https://arxiv.org/pdf/2605.06597)**

> **作者:** Yiqiao Jin; Yiyang Wang; Lucheng Fu; Yijia Xiao; Yinyi Luo; Haoxin Liu; B. Aditya Prakash; Josiah Hester; Jindong Wang; Srijan Kumar
>
> **备注:** Website: this https URL Code: this https URL
>
> **摘要:** Self-distillation (SD) offers a promising path for adapting large language models (LLMs) without relying on stronger external teachers. However, SD in autoregressive LLMs remains challenging because self-generated trajectories are free-form, correctness is task-dependent, and plausible rationales can still provide unstable or unreliable supervision. Existing methods mainly examine isolated design choices, leaving their effectiveness, roles, and interactions unclear. In this paper, we propose UniSD, a unified framework to systematically study self-distillation. UniSD integrates complementary mechanisms that address supervision reliability, representation alignment, and training stability, including multi-teacher agreement, EMA teacher stabilization, token-level contrastive learning, feature matching, and divergence clipping. Across six benchmarks and six models from three model families, UniSD reveals when self-distillation improves over static imitation, which components drive the gains, and how these components interact across tasks. Guided by these insights, we construct UniSDfull, an integrated pipeline that combines complementary components and achieves the strongest overall performance, improving over the base model by +5.4 points and the strongest baseline by +2.8 points. Extensive evaluation highlights self-distillation as a practical and steerable approach for efficient LLM adaptation without stronger external teachers.
>
---
#### [replaced 017] An Entity Linking Agent for Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答系统中的实体链接任务，旨在解决短而模糊问题中的实体链接难题。提出基于大语言模型的实体链接代理，提升QA系统的准确性。**

- **链接: [https://arxiv.org/pdf/2508.03865](https://arxiv.org/pdf/2508.03865)**

> **作者:** Yajie Luo; Yihong Wu; Muzhi Li; Jia Ao Sun; Xinyu Wang; Liheng Ma; Yingxue Zhang; Jian-Yun Nie
>
> **备注:** 12 pages, 2 figures
>
> **摘要:** Some Question Answering (QA) systems rely on knowledge bases (KBs) to provide accurate answers. Entity Linking (EL) plays a critical role in linking natural language mentions to KB entries. However, most existing EL methods are designed for long contexts and do not perform well on short, ambiguous user questions in QA tasks. We propose an entity linking agent for QA, based on a Large Language Model that simulates human cognitive workflows. The agent actively identifies entity mentions, retrieves candidate entities, and makes decision. To verify the effectiveness of our agent, we conduct two experiments: tool-based entity linking and QA task evaluation. The results confirm the robustness and effectiveness of our agent.
>
---
#### [replaced 018] Linear Dynamics in the RLVR Training of Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究RLVR训练中的线性动力学，解决大语言模型训练机制不透明的问题。通过分析发现RLVR进入稳定线性 regime，提出基于线性的优化方法提升训练效率和性能。**

- **链接: [https://arxiv.org/pdf/2601.04537](https://arxiv.org/pdf/2601.04537)**

> **作者:** Tianle Wang; Jiayu Liu; Zhongyuan Wu; Shenghao Jin; Wei Chen; Hao Xu; Ning Miao
>
> **备注:** Major revision: substantially reorganized the manuscript and added a theoretical explanation section. The replacement is intended for the same arXiv paper; the core topic and contribution remain the same
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has driven significant performance gains in reasoning-oriented large language models (LLMs), yet its internal training dynamics remain largely a black box. In this work, we perform a comprehensive trajectory-level analysis of RLVR and uncover a striking regularity: across various model families, RL algorithms, and training configurations, RLVR consistently enters a robust linear regime, where both parameter weights and output log-probabilities, measured rigorously via teacher-forced evaluation, evolve in a highly linear manner ($R^2 > 0.7$). Through controlled experiments and theoretical analysis, we demonstrate that this linearity is not a coincidence, but stems from the high-variance, noisy nature of RLVR training signals, which act as a low-pass filter to concentrate optimization along a stable, low-dimensional drift. Moreover, we show that this linear structure is not merely descriptive but powerfully predictive and actionable. Specifically, weight-space extrapolation matches the performance of standard RL optimization while achieving a 6.1x training speedup through periodic re-grounding. Meanwhile, output-space extrapolation serves as a lightweight intervention that effectively bypasses late-stage model collapse, consistently outperforming standard RL across mathematical and coding benchmarks, with an average performance improvement of 4.2%. Our code is available at this https URL.
>
---
#### [replaced 019] Learning to Foresee: Unveiling the Unlocking Efficiency of On-Policy Distillation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的后训练效率问题，提出OPD的效率源于“预见性”，通过模块分配和更新方向优化提升训练速度，实现3倍加速。**

- **链接: [https://arxiv.org/pdf/2605.11739](https://arxiv.org/pdf/2605.11739)**

> **作者:** Yuchen Cai; Ding Cao; Liang Lin; Chunxi Luo; Xin Xu; Kai Yang; Weijie Liu; Saiyong Yang; Tianxiang Zhao; Guangzhong Sun; Guiquan Liu; Junfeng Fang
>
> **摘要:** On-policy distillation (OPD) has emerged as an efficient post-training paradigm for large language models. However, existing studies largely attribute this advantage to denser and more stable supervision, while the parameter-level mechanisms underlying OPD's efficiency remain poorly understood. In this work, we argue that OPD's efficiency stems from a form of ``foresight'': it establishes a stable update trajectory toward the final model early in training. This foresight manifests in two aspects. First, at the \textbf{Module-Allocation Level}, OPD identifies regions with low marginal utility and concentrates updates on modules that are more critical to reasoning. Second, at the \textbf{Update-Direction Level}, OPD exhibits stronger low-rank concentration, with its dominant subspaces aligning closely with the final update subspace early in training. Building on these findings, we propose \textbf{EffOPD}, a plug-and-play acceleration method that speeds up OPD by adaptively selecting an extrapolation step size and moving along the current update direction. EffOPD requires no additional trainable modules or complex hyperparameter tuning, and achieves an average training acceleration of $3\times$ while maintaining comparable final performance. Overall, our findings provide a parameter-dynamics perspective for understanding the efficiency of OPD and offer practical insights for designing more efficient post-training methods for large language models.
>
---
#### [replaced 020] La representación de la variación contextual mediante definiciones terminológicas flexibles
- **分类: cs.CL**

- **简介: 该论文属于术语学任务，旨在解决环境术语在不同语境下的意义变化问题。通过分析上下文影响，提出灵活术语定义方法，以更准确地反映概念在不同子领域中的含义。**

- **链接: [https://arxiv.org/pdf/1607.06330](https://arxiv.org/pdf/1607.06330)**

> **作者:** Antonio San Martín
>
> **备注:** PhD Thesis. in Spanish. University of Granada. 2016
>
> **摘要:** In this doctoral thesis, we apply premises of cognitive linguistics to terminological definitions and present a proposal called the flexible terminological definition. This consists of a set of definitions of the same concept made up of a general definition (in this case, one encompassing the entire environmental domain) along with additional definitions describing the concept from the perspective of the subdomains in which it is relevant. Since context is a determining factor in the construction of the meaning of lexical units (including terms), we assume that terminological definitions can, and should, reflect the effects of context, even though definitions have traditionally been treated as the expression of meaning void of any contextual effect. The main objective of this thesis is to analyze the effects of contextual variation on specialized environmental concepts with a view to their representation in terminological definitions. Specifically, we focused on contextual variation based on thematic restrictions. To accomplish the objectives of this doctoral thesis, we conducted an empirical study consisting of the analysis of a set of contextually variable concepts and the creation of a flexible definition for two of them. As a result of the first part of our empirical study, we divided our notion of domain-dependent contextual variation into three different phenomena: modulation, perspectivization and subconceptualization. These phenomena are additive in that all concepts experience modulation, some concepts also undergo perspectivization, and finally, a small number of concepts are additionally subjected to subconceptualization. In the second part, we applied these notions to terminological definitions and we presented we presented guidelines on how to build flexible definitions, from the extraction of knowledge to the actual writing of the definition.
>
---
#### [replaced 021] Enhancing Causal Reasoning in Large Language Models: A Causal Attribution Model for Precision Fine-Tuning
- **分类: cs.AI; cs.CL; cs.LG; stat.ME**

- **简介: 该论文属于因果推理任务，旨在提升大语言模型的因果推理能力。通过构建因果归因模型，量化组件贡献，实现精准微调，有效结合知识与数值信息。**

- **链接: [https://arxiv.org/pdf/2401.00139](https://arxiv.org/pdf/2401.00139)**

> **作者:** Hengrui Cai; Shengjie Liu; Rui Song
>
> **备注:** A Python implementation of our proposed method is available at this https URL
>
> **摘要:** This paper introduces a causal attribution model to enhance the interpretability of large language models (LLMs) and improve their causal reasoning abilities via precise fine-tuning. Despite LLMs' proficiency in diverse tasks, their reasoning processes often remain black box, and thus restrict targeted enhancement. We propose a novel causal attribution model that utilizes "do-operators" for constructing interventional scenarios, allowing us to quantify the contribution of different components in LLMs's causal reasoning process systematically. By assessing the proposed attribution scores through causal discovery tasks across various domains, we demonstrate that LLMs' effectiveness in causal discovery heavily relies on provided context and domain-specific knowledge but can also utilize numerical data with limited calculations in correlation, not causation. This motivates the proposed fine-tuned LLM for pairwise causal discovery, effectively and correctly leveraging both knowledge and numerical information.
>
---
#### [replaced 022] Quantizing Whisper-small: How design choices affect ASR performance
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文研究语音识别模型Whisper-small的量化方法，旨在解决其在边缘设备部署中的计算需求问题。通过对比不同量化方案，找到提升性能与压缩模型的最佳方法。**

- **链接: [https://arxiv.org/pdf/2511.08093](https://arxiv.org/pdf/2511.08093)**

> **作者:** Arthur Söhler; Julian Irigoyen; Andreas Søeborg Kirkedal
>
> **备注:** Accepted to SPEAKABLE workshop at LREC 2026
>
> **摘要:** Large speech recognition models like Whisper-small achieve high accuracy but are difficult to deploy on edge devices due to their high computational demand. To this end, we present a unified, cross-library evaluation of post-training quantization (PTQ) on Whisper-small that disentangles the impact of quantization scheme, method, granularity, and bit-width. Our study is based on four libraries: PyTorch, Optimum-Quanto, HQQ, and bitsandbytes. Experiments on LibriSpeech test-clean and test-other show that dynamic int8 quantization with Quanto offers the best trade-off, reducing model size by 57% while improving on the baseline's word error rate. Static quantization performed worse, likely due to Whisper's Transformer architecture, while more aggressive formats (e.g., nf4, int3) achieved up to 71% compression at the cost of accuracy in noisy conditions. Overall, our results demonstrate that carefully chosen PTQ methods can substantially reduce model size and inference cost without retraining, enabling efficient deployment of Whisper-small on constrained hardware.
>
---
#### [replaced 023] Token-Level LLM Collaboration via FusionRoute
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出FusionRoute，解决多大模型协作问题，通过轻量路由选择专家并补充logit，提升跨领域性能。**

- **链接: [https://arxiv.org/pdf/2601.05106](https://arxiv.org/pdf/2601.05106)**

> **作者:** Nuoya Xiong; Yuhang Zhou; Hanqing Zeng; Zhaorun Chen; Furong Huang; Shuchao Bi; Lizhu Zhang; Zhuokai Zhao
>
> **备注:** 25 pages
>
> **摘要:** Large language models (LLMs) exhibit strengths across diverse domains. However, achieving strong performance across these domains with a single general-purpose model typically requires scaling to sizes that are prohibitively expensive to train and deploy. On the other hand, while smaller domain-specialized models are much more efficient, they struggle to generalize beyond their training distributions. To address this dilemma, we propose FusionRoute, a robust and effective token-level multi-LLM collaboration framework in which a lightweight router simultaneously (i) selects the most suitable expert at each decoding step and (ii) contributes a complementary logit that refines or corrects the selected expert's next-token distribution via logit addition. Unlike existing token-level collaboration methods that rely solely on fixed expert outputs, we provide a theoretical analysis showing that pure expert-only routing is fundamentally limited: unless strong global coverage assumptions hold, it cannot in general realize the optimal decoding policy. By augmenting expert selection with a trainable complementary generator, FusionRoute expands the effective policy class and enables recovery of optimal value functions under mild conditions. Empirically, across both Llama-3 and Gemma-2 families and diverse benchmarks spanning mathematical reasoning, code generation, and instruction following, FusionRoute outperforms both sequence- and token-level collaboration, model merging, and direct fine-tuning, while remaining competitive with domain experts on their respective tasks.
>
---
#### [replaced 024] Intelligence per Watt: Measuring Intelligence Efficiency of Local AI
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究本地AI的智能效率，解决如何有效评估本地模型在功率受限设备上的性能问题。通过提出“每瓦智能量”（IPW）指标，评估本地模型与云模型的效率和准确性。**

- **链接: [https://arxiv.org/pdf/2511.07885](https://arxiv.org/pdf/2511.07885)**

> **作者:** Jon Saad-Falcon; Avanika Narayan; Hakki Orhun Akengin; J. Wes Griffin; Herumb Shandilya; Adrian Gamarra Lafuente; Medhya Goel; Rebecca Joseph; Shlok Natarajan; Etash Kumar Guha; Shang Zhu; Ben Athiwaratkun; John Hennessy; Azalia Mirhoseini; Christopher Ré
>
> **摘要:** Large language model (LLM) queries are predominantly processed by frontier models in centralized cloud infrastructure. Demand growth strains this paradigm faster than providers can scale. Two advances create an opportunity to rethink it: small, local LMs (<=20B active parameters) now achieve competitive performance to frontier models on many tasks, and local accelerators (e.g., Apple M4 Max) can host these models at interactive latencies. This raises the question: can local inference viably redistribute demand from centralized infrastructure? This requires measuring both whether local LMs can accurately answer real-world queries and whether they can do so efficiently on power-constrained devices (e.g., laptops). We propose intelligence per watt (IPW), task accuracy per unit of power, as a unified metric for the capability and efficiency of local inference across model-accelerator configurations. We evaluate 20+ state-of-the-art local LMs, 8 hardware accelerators (local and cloud), and 1M real-world single-turn chat and reasoning queries. For each query, we measure accuracy (local LM win rate against frontier models), energy, latency, and power. We find three key results. First, local LMs successfully answer 88.7% of these queries, with accuracy varying by domain. Second, longitudinal analysis from 2023-2025 shows IPW improved 5.3x, driven by both algorithmic and accelerator advances, with locally-serviceable query coverage rising from 23.2% to 71.3%. Third, local accelerators achieve at least 1.4x lower IPW than cloud accelerators running identical models, revealing significant headroom for local accelerator optimization. These findings demonstrate that local inference can meaningfully redistribute demand from centralized infrastructure for a substantial subset of queries, with IPW serving as the critical metric for tracking this transition.
>
---
#### [replaced 025] LLM Readiness Harness: Evaluation, Observability, and CI Gates for LLM/RAG Applications
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文提出一种LLM/RAG应用的准备度评估框架，解决模型部署决策问题。通过集成评估、可观测性和CI质量门禁，实现可操作的部署判断。**

- **链接: [https://arxiv.org/pdf/2603.27355](https://arxiv.org/pdf/2603.27355)**

> **作者:** Alexandre Cristovão Maiorano
>
> **备注:** 19 pages, 4 figures, 15 tables
>
> **摘要:** We present a readiness harness for LLM and RAG applications that turns evaluation into a deployment decision workflow. The system combines automated benchmarks, OpenTelemetry observability, and CI quality gates under a minimal API contract, then aggregates workflow success, policy compliance, groundedness, retrieval hit rate, cost, and p95 latency into scenario-weighted readiness scores with Pareto frontiers. We evaluate the harness on ticket-routing workflows and BEIR grounding tasks (SciFact and FiQA) with full Azure matrix coverage (162/162 valid cells across datasets, scenarios, retrieval depths, seeds, and models). Results show that readiness is not a single metric: on FiQA under sla-first at k=5, gpt-4.1-mini leads in readiness and faithfulness, while gpt-5.2 pays a substantial latency cost; on SciFact, models are closer in quality but still separable operationally. Ticket-routing regression gates consistently reject unsafe prompt variants, demonstrating that the harness can block risky releases instead of merely reporting offline scores. The result is a reproducible, operationally grounded framework for deciding whether an LLM or RAG system is ready to ship.
>
---
#### [replaced 026] When LLMs Stop Following Steps: A Diagnostic Study of Procedural Execution in Language Models
- **分类: cs.CL**

- **简介: 该论文研究语言模型在执行步骤指令时的可靠性，旨在解决模型是否忠实执行程序的问题。通过设计基准测试，分析模型在复杂算法中的表现及错误类型。**

- **链接: [https://arxiv.org/pdf/2605.00817](https://arxiv.org/pdf/2605.00817)**

> **作者:** Sailesh Panda; Pritam Kadasi; Abhishek Upperwal; Mayank Singh
>
> **备注:** 77 pages, 109 figures
>
> **摘要:** Large language models (LLMs) often achieve strong performance on reasoning benchmarks, but final-answer accuracy alone does not show whether they faithfully execute the procedure specified in a prompt. We study this question through a controlled diagnostic benchmark for procedural execution, where models are given a step-wise arithmetic algorithm and two numeric inputs, and must return the final computed value. The benchmark uses simple arithmetic operations but increases complexity through algorithm length and look-back dependencies over intermediate variables. Across 14 models and 55 datasets, average first-answer accuracy drops from 61% on 5-step procedures to 20% on 95-step procedures. Generation-level analysis shows that failures often involve missing answers, premature answers, self-correction after an initial error, under-executed traces, and hallucinated extra steps. These findings suggest that apparent reasoning ability can mask substantial weaknesses in faithful instruction execution.
>
---
#### [replaced 027] "Would You Want an AI Tutor?" Understanding Stakeholder Perceptions of LLM-based Systems in the Classroom
- **分类: cs.CY; cs.CL; cs.HC**

- **简介: 该论文属于教育技术领域，旨在解决LLM在课堂中应用的伦理与管理问题。通过提出Co-PALE框架，分析利益相关者对LLM的感知，支持更负责任的部署决策。**

- **链接: [https://arxiv.org/pdf/2503.02885](https://arxiv.org/pdf/2503.02885)**

> **作者:** Caterina Fuligni; Daniel Dominguez Figaredo; Armanda Lewis; Julia Stoyanovich
>
> **摘要:** Large Language Models (LLMs) have gained traction in educational settings, often framed as virtual tutors or teaching assistants. Following early skepticism and bans, many schools and universities have begun integrating these systems into curricula. Yet decisions about whether and how to deploy LLM-based tools are frequently made without systematic engagement with the full range of stakeholders they affect. In this paper, we argue that understanding stakeholder perceptions of LLM-based systems in the classroom is not a matter of measuring approval or acceptance, but of identifying whose concerns are surfaced, in which contexts, and with what implications for responsible design and governance. We introduce Contextualized Perceptions for the Adoption of LLMs in Education (Co-PALE), a stakeholder-first framework that connects educational context, responsible AI principles, and categories of perception to support more deliberate decision-making about the adoption of LLM-based tools. We ground Co-PALE through a targeted analysis of prior work to diagnose recurring gaps in how stakeholder perceptions are studied, and through contextually distinct educational scenarios that illustrate how the same technology raises different concerns for different stakeholders. We further examine how university faculty and K--12 parents make sense of the framework through focus groups, using their reflections to surface tensions and uncertainties. Co-PALE supports more systematic reasoning about whether, where, and for whom LLM-based tools should be deployed in education.
>
---
#### [replaced 028] General Agentic Planning Through Simulative Reasoning with World Models
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文提出一种基于世界模型的模拟推理架构SiRA，用于增强智能体的规划能力。解决传统系统反应式决策泛化性差的问题，通过模拟未来结果提升任务完成率。属于智能体规划任务。**

- **链接: [https://arxiv.org/pdf/2507.23773](https://arxiv.org/pdf/2507.23773)**

> **作者:** Mingkai Deng; Jinyu Hou; Zhiting Hu; Eric Xing
>
> **备注:** Winner of Berkeley LLM Agents Hackathon (Fundamentals Track); code available at this https URL
>
> **摘要:** What does it mean to plan? Current agentic systems, whether scaffolded workflows or end-to-end policies, rely on reactive decision-making: selecting the next action via a fixed procedure with at most undifferentiated adaptive computation (e.g., chain-of-thought) lacking explicit modeling of future outcomes. This limits generalizability, as each new task demands re-engineering rather than transfer of shared reasoning capacity. Humans, by contrast, plan by mentally simulating consequences of candidate actions within an internal world model, a capacity known as simulative reasoning (System II) that supports flexible, goal-directed behavior across diverse contexts. We argue that simulative reasoning through a world model provides a general-purpose planning mechanism for agentic systems, improving upon reactive policies (System I) by grounding decisions in predicted future states rather than pattern-matched responses. To verify this, we introduce SiRA (Simulative Reasoning Architecture), a goal-oriented architecture instantiating simulative reasoning using an LLM-based world model with natural-language belief states, while remaining model-agnostic. We evaluate across three qualitatively distinct task categories: constrained navigation, multi-hop information aggregation, and general instruction following, in a web-browser environment. Across all categories, simulative reasoning achieves up to 124% higher task completion rates than a matched reactive baseline, and increases constrained navigation success from 0% to 32.2% compared to a representative open-web agent. The persistent advantage across distinct task types suggests the benefit stems from generalizable counterfactual evaluation rather than task-specific tuning.
>
---
#### [replaced 029] Accelerated Test-Time Scaling with Model-Free Speculative Sampling
- **分类: cs.CL**

- **简介: 该论文属于语言模型推理加速任务，旨在解决传统方法计算资源消耗大的问题。提出STAND方法，在不损失精度的前提下显著提升推理效率。**

- **链接: [https://arxiv.org/pdf/2506.04708](https://arxiv.org/pdf/2506.04708)**

> **作者:** Woomin Song; Saket Dingliwal; Sai Muralidhar Jayanthi; Bhavana Ganesh; Jinwoo Shin; Aram Galstyan; Sravan Babu Bodapati
>
> **备注:** EMNLP 2025 Oral
>
> **摘要:** Language models have demonstrated remarkable capabilities in reasoning tasks through test-time scaling techniques like best-of-N sampling and tree search. However, these approaches often demand substantial computational resources, creating a critical trade-off between performance and efficiency. We introduce STAND (STochastic Adaptive N-gram Drafting), a novel model-free speculative decoding approach that exploits the inherent redundancy in reasoning trajectories to achieve significant acceleration without compromising accuracy. Our analysis shows that reasoning paths frequently reuse similar reasoning patterns, enabling efficient model-free token prediction without requiring separate draft models. By introducing stochastic drafting and preserving probabilistic information through a memory-efficient logit-based N-gram module, combined with optimized Gumbel-Top-K sampling and data-driven tree construction, STAND significantly improves token acceptance rates. Extensive evaluations across multiple models and reasoning tasks (AIME-2024, GPQA-Diamond, and LiveCodeBench) demonstrate that STAND reduces inference latency by 60-65% compared to standard autoregressive decoding while maintaining accuracy. Furthermore, STAND consistently outperforms state-of-the-art speculative decoding methods across diverse inference patterns, including single-trajectory decoding, batch decoding, and test-time tree search. As a model-free approach, STAND can be applied to any existing language model without additional training, making it a powerful plug-and-play solution for accelerating language model reasoning.
>
---
#### [replaced 030] Jordan-RoPE: Non-Semisimple Relative Positional Encoding via Complex Jordan Blocks
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决位置编码问题。提出Jordan-RoPE方法，通过复数Jordan块生成距离调制的相位特征，增强模型对距离依赖关系的捕捉能力。**

- **链接: [https://arxiv.org/pdf/2605.04217](https://arxiv.org/pdf/2605.04217)**

> **作者:** Yaobo Zhang
>
> **备注:** 15 pages, 4 figures, 6 tables; code available at this https URL
>
> **摘要:** Relative positional encodings determine which functions of query-key lag can enter the primitive attention logit. RoPE supplies a rotary phase, while ALiBi supplies an additive distance bias. Motivated by group-theoretic views of linear translation-invariant positional encodings, we study a non-semisimple case in which a complex rotary eigenvalue and a nilpotent response live in the same defective Jordan block. The resulting relative operator generates oscillatory-polynomial features such as $e^{-\gamma d}\cos(\omega d)$, $e^{-\gamma d}\sin(\omega d)$, $d e^{-\gamma d}\cos(\omega d)$, and $d e^{-\gamma d}\sin(\omega d)$, for causal lag $d=i-j\geq 0$. Thus the construction realizes a distance-modulated phase basis $d e^{i\omega d}$, rather than merely adding a separate distance channel to RoPE. We formulate Exact Jordan-RoPE as a non-semisimple one-parameter representation, give its real block form, and specify the contragredient query action required by non-orthogonal positional maps. We also distinguish this exact representation from stabilized variants whose bounded shear improves numerical behavior but breaks the exact group law. Kernel-level diagnostics and a Jordan-friendly synthetic language-model task show that the coupled Jordan basis is useful when the target contains distance-modulated phase interactions. On a small WikiText-103 byte language model, a scaled-exact variant improves over RoPE and direct-sum baselines within the Jordan family, while RoPE+ALiBi remains strongest overall. The evidence is structural rather than a broad performance claim.
>
---
#### [replaced 031] Putnam 2025 Problems in Rocq using Opus 4.6 and Rocq-MCP
- **分类: cs.LG; cs.CL; cs.LO**

- **简介: 该论文属于自动定理证明任务，旨在用Claude Opus 4.6和MCP工具自主解决数学竞赛问题。工作包括开发策略并成功证明10题。**

- **链接: [https://arxiv.org/pdf/2603.20405](https://arxiv.org/pdf/2603.20405)**

> **作者:** Guillaume Baudart; Marc Lelarge; Tristan Stérin; Jules Viennot
>
> **摘要:** We report on an experiment in which Claude Opus~4.6, equipped with a suite of Model Context Protocol (MCP) tools for the Rocq proof assistant, autonomously proved 10 of 12 problems from the 2025 Putnam Mathematical Competition. The MCP tools, designed with Claude by analyzing logs from a prior experiment on miniF2F-Rocq, encode a "compile-first, interactive-fallback" strategy. Running on an isolated VM with no internet access, the agent deployed 141 subagents over 17.7 hours of active compute (51.6h wall-clock), consuming approximately 1.9 billion tokens. All proofs are publicly available.
>
---
#### [replaced 032] Evaluating Clinical Competencies of Large Language Models with a General Practice Benchmark
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI评估任务，旨在解决LLMs在临床实践中能力不足的问题。研究提出GPBench基准，评估LLMs作为全科医生的胜任力，发现其仍需人类监督。**

- **链接: [https://arxiv.org/pdf/2503.17599](https://arxiv.org/pdf/2503.17599)**

> **作者:** Zheqing Li; Yiying Yang; Jiping Lang; Wenhao Jiang; Junrong Chen; Yuhang Zhao; Shuang Li; Dingqian Wang; Zhu Lin; Xuanna Li; Yuze Tang; Jiexian Qiu; Xiaolin Lu; Hongji Yu; Shuang Chen; Yuhua Bi; Xiaofei Zeng; Yixian Chen; Lin Yao
>
> **摘要:** Large Language Models (LLMs) have demonstrated considerable potential in general practice. However, existing benchmarks and evaluation frameworks primarily depend on exam-style or simplified question-answer formats, lacking a competency-based structure aligned with the real-world clinical responsibilities encountered in general practice. Consequently, the extent to which LLMs can reliably fulfill the duties of general practitioners (GPs) remains uncertain. In this work, we propose a novel evaluation framework to assess the capability of LLMs to function as GPs. Based on this framework, we introduce a general practice benchmark (GPBench), whose data are meticulously annotated by domain experts in accordance with routine clinical practice standards. We evaluate ten state-of-the-art LLMs and analyze their competencies. Our findings indicate that current LLMs are not suitable for autonomous deployment in clinical general practice and that all realistic applications require continuous human oversight; further optimization specifically tailored to the daily responsibilities of GPs remains essential.
>
---
#### [replaced 033] Discovering Implicit Large Language Model Alignment Objectives
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于AI对齐任务，旨在解决LLM对齐目标不明确的问题。通过提出Obj-Disco框架，自动分解奖励信号为可解释的自然语言目标，提升模型行为的透明度和安全性。**

- **链接: [https://arxiv.org/pdf/2602.15338](https://arxiv.org/pdf/2602.15338)**

> **作者:** Edward Chen; Sanmi Koyejo; Carlos Guestrin
>
> **备注:** ICML 2026
>
> **摘要:** Large language model (LLM) alignment relies on complex reward signals that often obscure the specific behaviors being incentivized, creating critical risks of misalignment and reward hacking. Existing interpretation methods typically rely on pre-defined rubrics, risking the omission of "unknown unknowns", or fail to identify objectives that comprehensively cover and are causal to the model behavior. To address these limitations, we introduce Obj-Disco, a framework that automatically decomposes an alignment reward signal into a sparse, weighted combination of human-interpretable natural language objectives. Our approach utilizes an iterative greedy algorithm to analyze behavioral changes across training checkpoints, identifying and validating candidate objectives that best explain the residual reward signal. Extensive evaluations across diverse tasks, model sizes, and alignment algorithms demonstrate the framework's robustness. Experiments with popular open-source reward models show that the framework consistently captures > 90% of reward behavior, a finding further corroborated by human evaluation. Additionally, a case study on alignment with an open-source reward model reveals that Obj-Disco can successfully identify latent misaligned incentives that emerge alongside intended behaviors. Our work provides a crucial tool for uncovering the implicit objectives in LLM alignment, paving the way for more transparent and safer AI development.
>
---
#### [replaced 034] Robust Reasoning Benchmark
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在评估大模型在不同文本格式下的推理能力。通过构建RRB基准，研究模型对文本扰动的鲁棒性，发现部分模型存在推理失效问题，并提出注意力稀释现象及改进方向。**

- **链接: [https://arxiv.org/pdf/2604.08571](https://arxiv.org/pdf/2604.08571)**

> **作者:** Pavel Golikov; Evgenii Opryshko; Gennady Pekhimenko; Mark C. Jeffrey
>
> **摘要:** While Large Language Models (LLMs) achieve high performance on standard mathematical benchmarks, their problem-solving abilities depend on the context and textual formatting. We introduce the Robust Reasoning Benchmark (RRB), a pipeline of 13 deterministic textual perturbations applied to AIME 2024 and AIME 2025. Evaluating 8 state-of-the-art models, we find that frontier models are largely resilient, with the notable exception of Claude, which categorically refuses many transformed prompts. Open-weights reasoning models exhibit a range of failure modes under structural noise (cognitive thrashing, tokenization breakdown, and reasoning collapse), with up to 54% average accuracy drops across perturbations and up to 100% on some. We further study one of these failure modes in isolation: attention dilution caused by the model's own chain-of-thought. By tasking models with solving multiple independent mathematical problems sequentially within a single context window, we identify Intra-Query Attention Dilution. Open-weights models ranging from 7B to 120B parameters exhibit accuracy decay on subsequent problems, suggesting that intermediate reasoning steps progressively pollute standard dense attention mechanisms. We argue that in order to achieve reliable reasoning, future architectures need to integrate explicit contextual resets within models' own chain-of-thought, leading to open research questions regarding the optimal granularity of reasoning tasks.
>
---
#### [replaced 035] Symphony for Speech-to-Text: Supporting Real-Time Medical Voice Interfaces
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于医疗语音识别任务，解决医学术语准确识别与上下文理解问题。提出Symphony系统，实现实时和批量临床语音转录，提升医疗场景下的准确性与适应性。**

- **链接: [https://arxiv.org/pdf/2605.16545](https://arxiv.org/pdf/2605.16545)**

> **作者:** Arne Nix; Robert James; Lasse Borgholt; Anna B. Ekner; Lana Krumm; Julius Severin; Dan Engel; Lars Maaløe; Jakob Havtorn
>
> **备注:** Updated with a correction and improvement to Symphony's performance in spoken punctuation evaluation (R_punct, P_punct)
>
> **摘要:** After decades of use in dictation and, more recently, ambient documentation, speech is emerging as a primary modality for interacting with technology and AI in healthcare. Yet medical speech recognition remains difficult: systems must capture specialized terminology, resolve contextual ambiguity, and render measurements, abbreviations, and clinical shorthand precisely. Existing solutions are typically optimized either for general-purpose transcription or narrow dictation workflows, limiting their reliability in safety-critical settings and their usefulness for broader clinical workflows. We introduce Symphony for Speech-to-Text, a medical-grade speech recognition system for real-time streaming and batch file-based clinical use. Symphony decomposes the transcription process into specialized components for recognition, formatting, and contextual correction to optimize medical term recall while producing clinically structured text in real time and adapting across use cases. Evaluations on public benchmark and medical speech datasets show that Symphony substantially outperforms state-of-the-art systems in clinical settings while matching or exceeding them in general-domain settings, suggesting robust generalization rather than overfitting. We release a clinical benchmark dataset to support reliable validation and further progress in medical speech recognition. Symphony is available through a production-grade API for live dictation, conversational transcription, and batch audio file processing.
>
---
#### [replaced 036] Benchmarking Commercial ASR Systems on Code-Switching Speech: Arabic, Persian, and German
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在评估商业ASR系统在代码切换语音中的表现。针对多语言交互中的挑战，作者构建了包含四个语对的数据集，并使用WER和BERTScore进行评估，揭示了系统性能差异。**

- **链接: [https://arxiv.org/pdf/2605.19069](https://arxiv.org/pdf/2605.19069)**

> **作者:** Sajjad Abdoli; Ghassan Al-Sumaidaee; Clayton W. Taylor; Ahmad ElShiekh; Ahmed Rashad
>
> **摘要:** Code-switching -- the natural alternation between two languages within a single utterance -- represents one of the most challenging and under-studied conditions for automatic speech recognition (ASR). Existing commercial ASR benchmarks predominantly evaluate clean, monolingual audio and report a single Word Error Rate (WER) figure that tells practitioners little about real-world multilingual performance. We present a benchmark evaluating five commercial ASR providers across four language pairs: Egyptian Arabic--English, Saudi Arabic (Najdi/Hijazi)--English, Persian (Farsi)--English, and German--English. Each dataset comprises 300 samples selected by a two-stage pipeline: a heuristic filter scoring transcripts on five structural code-switching signals, followed by a GPT-4o and Gemini 1.5 Pro ensemble scoring candidates across six linguistic dimensions. This pipeline reduces LLM scoring costs by approximately 91% relative to exhaustive scoring. We evaluate the systems on both WER and BERTScore, arguing that BERTScore is a more reliable metric for Arabic and Persian pairs where transliteration variance causes WER to penalise semantically correct transcriptions. ElevenLabs Scribe v2 achieves the lowest WER across all four language pairs (13.2% overall; 13.1% on Egyptian Arabic) and leads on BERTScore (0.936 overall). We further demonstrate that difficulty-stratified analysis reveals performance gaps masked by aggregate averages, and that BERT embedding projections confirm semantic proximity between reference and hypothesis despite surface-level script differences. The benchmarking dataset is publicly available at this https URL.
>
---
#### [replaced 037] DocAtlas: Multilingual Document Understanding Across 80+ Languages
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出DocAtlas，解决多语言文档理解问题，尤其针对低资源语言。构建跨82语言的高质量OCR数据集，采用双管道生成结构化标注，提升模型多语言适应能力。**

- **链接: [https://arxiv.org/pdf/2605.12623](https://arxiv.org/pdf/2605.12623)**

> **作者:** Ahmed Heakl; Youssef Mohamed; Abdullah Sohail; Rania Elbadry; Ahmed Nassar; Peter W. J. Staar; Fahad Shahbaz Khan; Imran Razzak; Salman Khan
>
> **备注:** Under submission
>
> **摘要:** Multilingual document understanding remains limited for low-resource languages due to scarce training data and model-based annotation pipelines that perpetuate existing biases. We introduce DocAtlas, a framework that constructs high-fidelity OCR datasets and benchmarks covering 82 languages and 9 evaluation tasks. Our dual pipelines, differential rendering of native DOCX documents and synthetic LaTeX-based generation for right-to-left scripts produce precise structural annotations in a unified DocTag format encoding layout, text, and component types, without learned models for core annotation. Evaluating 16 state-of-the-art models reveals persistent gaps in low-resource scripts. We show that Direct Preference Optimization (DPO) using rendering-derived ground truth as positive signal achieves stable multilingual adaptation, improving both in-domain (+1.9%) and out-of-domain (+1.8%) accuracy without measurable base-language degradation, where supervised fine-tuning degrades out-of-domain performance by up to 21%. Our best variant, DocAtlas-DeepSeek, improves +1.7% over the strongest baseline. Code is available at this https URL .
>
---
#### [replaced 038] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出TingIS系统，解决大规模企业中从噪声客户事件中实时发现风险的问题。通过事件关联、路由和降噪技术，提升事件检测准确性和效率。**

- **链接: [https://arxiv.org/pdf/2604.21889](https://arxiv.org/pdf/2604.21889)**

> **作者:** Jun Wang; Ziyin Zhang; Rui Wang; Hang Yu; Peng Di; Rui Wang
>
> **备注:** Accepted to ACL 2026 Industry Track
>
> **摘要:** Real-time detection and mitigation of technical anomalies are critical for large-scale cloud-native services, where even minutes of downtime can result in massive financial losses and diminished user trust. While customer incidents serve as a vital signal for discovering risks missed by monitoring, extracting actionable intelligence from this data remains challenging due to extreme noise, high throughput, and semantic complexity of diverse business lines. In this paper, we present TingIS, an end-to-end system designed for enterprise-grade incident discovery. At the core of TingIS is a multi-stage event linking engine that synergizes efficient indexing techniques with Large Language Models (LLMs) to make informed decisions on event merging, enabling the stable extraction of actionable incidents from just a handful of diverse user descriptions. This engine is complemented by a cascaded routing mechanism for precise business attribution and a multi-dimensional noise reduction pipeline that integrates domain knowledge, statistical patterns, and behavioral filtering. Deployed in a production environment handling a peak throughput of over 2,000 messages per minute and 300,000 messages per day, TingIS achieves a P90 alert latency of 3.5 minutes and a 95\% discovery rate for high-priority incidents. Benchmarks constructed from real-world data demonstrate that TingIS significantly outperforms baseline methods in routing accuracy, clustering quality, and Signal-to-Noise Ratio.
>
---
#### [replaced 039] CritiSense: Critical Digital Literacy and Resilience Against Misinformation
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文提出CritiSense应用，旨在提升用户对虚假信息的抵抗力。属于数字素养任务，解决 misinformation 问题，通过互动挑战增强用户识别能力。**

- **链接: [https://arxiv.org/pdf/2603.16672](https://arxiv.org/pdf/2603.16672)**

> **作者:** Firoj Alam; Fatema Ahmad; Ali Ezzat Shahroor; Mohamed Bayan Kmainasi; Elisa Sartori; Giovanni Da San Martino; Abul Hasnat; Raian Ali
>
> **备注:** resilience, disinformation, misinformation, fake news, propaganda
>
> **摘要:** Misinformation on social media undermines informed decision-making and public trust. Prebunking offers a proactive complement by helping users recognize manipulation tactics before they encounter them in the wild. We present CritiSense, a mobile media-literacy app that builds these skills through short, interactive challenges with instant feedback. It is the first multilingual (supporting nine languages) and modular platform, designed for rapid updates across topics and domains. We report a usability study with 93 users: 83.9% expressed overall satisfaction and 90.1% rated the app as easy to use. Qualitative feedback indicates that CritiSense helps improve digital literacy skills. Overall, it provides a multilingual prebunking platform and a testbed for measuring the impact of microlearning on misinformation resilience. Over 6 months, we have reached 500+ active users. It is freely available to all users on the Apple App Store (this https URL) and Google Play Store (this https URL).
>
---
#### [replaced 040] Frame In, Frame Out: Measuring Framing Bias in LLM-Generated News Summaries
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决LLM生成摘要中的框架偏差问题。通过构建FIFO基准，分析模型摘要的框架率，发现其常高于人类参考摘要。**

- **链接: [https://arxiv.org/pdf/2505.05406](https://arxiv.org/pdf/2505.05406)**

> **作者:** Valeria Pastorino; Nafise Sadat Moosavi
>
> **备注:** Accepted to The 15th Joint Conference on Lexical and Computational Semantics (*SEM 2026) co-located with ACL 2026
>
> **摘要:** News headlines and summaries shape how events are interpreted through selective emphasis and omission, a phenomenon commonly referred to as framing. Large language models are now routinely used to generate such content, yet existing evaluation frameworks largely overlook this dimension. We introduce Frame In, Frame Out (FIFO), the first large-scale benchmark for measuring framing presence in LLM-generated news summaries, grounded in the widely used XSum dataset. FIFO combines 15,499 jury-annotated examples with 320 expert-labeled instances ($\kappa = 0.61$) to validate and calibrate model-based annotations. Using FIFO, we analyze measured framing rates across 27 summarization models. We find that LLM-generated summaries often exhibit higher calibrated framing rates than human-written references, with substantial variation across topics and training regimes, including elevated rates in scientific and public health summaries. Our results establish framing as an underexplored and consequential dimension of summarization quality.
>
---
#### [replaced 041] Training-Trajectory-Aware Token Selection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型压缩任务，解决持续知识蒸馏效果不佳的问题。通过分析训练轨迹，提出T3S方法选择关键token，提升蒸馏效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.10348](https://arxiv.org/pdf/2601.10348)**

> **作者:** Zhanming Shen; Jiaqi Hu; Zeyu Qin; Hao Chen; Wentao Ye; Zenan Huang; Yihong Zhuang; Guoshan Lu; Junlin Zhou; Junbo Zhao
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Efficient distillation is a key pathway for converting expensive reasoning capability into deployable efficiency, yet in the frontier regime where the student already has strong reasoning ability, naive continual distillation often yields limited gains or even degradation. We observe a characteristic training phenomenon: even as loss decreases monotonically, all performance metrics can drop sharply at almost the same bottleneck, before gradually recovering. We further uncover a token-level mechanism: confidence bifurcates into steadily increasing Imitation-Anchor Tokens that quickly anchor optimization and other yet-to-learn tokens whose confidence is suppressed until after the bottleneck. And the characteristic that these two types of tokens cannot coexist is the root cause of the failure in continual distillation. To this end, we propose Training-Trajectory-Aware Token Selection (T3S) to reconstruct the training objective at the token level, clearing the optimization path for yet-to-learn tokens. T3S yields consistent gains in both AR and dLLM settings: with only hundreds of examples, Qwen3-8B surpasses DeepSeek-R1 on competitive reasoning benchmarks, Qwen3-32B approaches Qwen3-235B, and T3-trained LLaDA-2.0-Mini exceeds its AR baseline, achieving state-of-the-art performance among all of 16B-scale no-think models.
>
---
#### [replaced 042] MemEvoBench: Benchmarking Safety Risks from Memory Misevolution in LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于LLM安全任务，旨在解决记忆偏差导致的代理行为风险。提出MemEvoBench基准，评估记忆演化中的安全问题，通过多种任务模拟记忆污染影响。**

- **链接: [https://arxiv.org/pdf/2604.15774](https://arxiv.org/pdf/2604.15774)**

> **作者:** Weiwei Xie; Shaoxiong Guo; Fan Zhang; Tian Xia; Xue Yang; Lizhuang Ma; Junchi Yan; Qibing Ren
>
> **摘要:** Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents.
>
---
#### [replaced 043] Closing the Gap at CRAC 2026: Two-Stage Adaptation for LLM-Based Multilingual Coreference Resolution
- **分类: cs.CL**

- **简介: 该论文属于多语言共指消解任务，旨在提升大模型在该任务上的表现。通过两阶段微调和适配器策略，使用Gemma-3-27b模型取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2605.16984](https://arxiv.org/pdf/2605.16984)**

> **作者:** Antoine Bourgois; Olga Seminck; Thierry Poibeau
>
> **摘要:** We present our submission to the LLM track of the 2026 Computational Models of Reference, Anaphora and Coreference (CRAC 2026) shared task. With an average CoNLL F1 score of 74.32 on the official test set, our system ranked first in the LLM track, and third overall. Our system is based on the Gemma-3-27b model, fine-tuned using a two-stage strategy with a multilingual base adapter followed by dataset-specific adapters. We represent mention spans by their headword using an XML-inspired format with local reindexing and annotate documents iteratively. These design choices proved effective across languages, document lengths, and annotation guidelines.
>
---
#### [replaced 044] SimCT: Recovering Lost Supervision for Cross-Tokenizer On-Policy Distillation
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，解决跨分词器教师-学生模型对齐问题。通过扩展监督范围，恢复因分词差异丢失的教师信号，提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2605.07711](https://arxiv.org/pdf/2605.07711)**

> **作者:** Jie Sun; Mao Zheng; Mingyang Song; Qiyong Zhong; Yilin Cheng; Bichuan Feng; Pengfei Liu; Junfeng Fang; Xiang Wang
>
> **备注:** 4 figures, 6 tables, 28 pages
>
> **摘要:** On-policy distillation (OPD) is a standard tool for transferring teacher behavior to a smaller student, but it implicitly assumes that teacher and student predictions are comparable token by token, an assumption that fails whenever the two models tokenize the same text differently. Under heterogeneous tokenizers, exact shared-token matching silently discards a large fraction of the teacher signal at precisely the positions where vocabularies disagree. We propose \textbf{\underline{Sim}ple \underline{C}ross-\underline{T}okenizer OPD (SimCT)}, which restores this signal by enlarging the supervision space: alongside shared tokens, SimCT compares teacher and student over short multi-token continuations that both tokenizers can realize, leaving the OPD loss form itself unchanged. We show that these units are the finest jointly tokenizable supervision interface, and that coarser alternatives remove teacher-student distinctions that are useful for on-policy learning. Across three heterogeneous teacher-student pairs on mathematical reasoning and code-generation benchmarks, SimCT shows consistent gains over shared-vocabulary OPD and representative cross-tokenizer baselines, with ablations confirming that the improvements come from recovering supervision discarded by exact shared-token matching. Code is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 045] General Preference Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出General Preference Reinforcement Learning（GPRL），解决大语言模型对齐中在线强化学习与偏好优化的分离问题，通过多维偏好建模提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18721](https://arxiv.org/pdf/2605.18721)**

> **作者:** Muhammad Umer; Muhammad Ahmed Mohsin; Ahsan Bilal; Arslan Chaudhry; Andreas Haupt; Sanmi Koyejo; Emily Fox; John M. Cioffi
>
> **摘要:** Post-training has split large language model (LLM) alignment into two largely disconnected tracks. Online reinforcement learning (RL) with verifiable rewards drives emergent reasoning on math and code but depends on a programmatic verifier that cannot reach open-ended tasks, while preference optimization handles open-ended generation yet forgoes the continuous exploration that powers online RL. Closing this gap requires a verifier for open-ended quality, but a scalar reward model is the wrong shape for the job. Quality is multi-dimensional, and any scalar score is an incomplete proxy that lets online RL collapse onto whichever axis the score is most sensitive to. We turn instead to the General Preference Model (GPM), which embeds responses into $k$ skew-symmetric subspaces and represents preference as a structured, intransitivity-aware comparison. Building on this, we propose General Preference Reinforcement Learning (GPRL), which carries the $k$-way structure through to the policy update. GPRL computes per-dimension group-relative advantages, normalizes each on its own scale so no axis can dominate, and aggregates them with context-dependent eigenvalues. The same structure powers a closed-loop drift monitor that detects single-axis exploitation and corrects it on the fly by reweighting dimensions and tightening the trust region. Starting from $\texttt{Llama-3-8B-Instruct}$, GPRL reaches a length-controlled win rate of $56.51\%$ on AlpacaEval~2.0 while also outperforming SimPO and SPPO on Arena-Hard, MT-Bench, and WildBench by resisting reward hacking across extended training runs.
>
---
#### [replaced 046] Internal narratives parameterise affective states
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于心理评估任务，旨在理解情绪状态与内部叙事的关系。通过分析文本表征，研究者量化了抑郁状态的结构与动态，揭示了叙事对情绪报告的影响。**

- **链接: [https://arxiv.org/pdf/2502.09487](https://arxiv.org/pdf/2502.09487)**

> **作者:** Jakub Onysk; Quentin J. M. Huys
>
> **摘要:** Characterising how we verbalise our feelings is central to psychological assessment and intervention, yet the mapping between narrative and affective state remains poorly understood. Across two large studies (n=1257), we parameterised the structure and dynamics of depressive states by quantifying participants' internal narratives through large-language-model representations and their subspaces. In Study 1, we found verbal descriptions of symptom-specific thoughts captured granular information predictive of standardised, self-reported depression scores. Critically, we show preserving the specific covariance between symptoms is essential for construct validity, suggesting high-dimensional text representations mirror the latent geometry of the disorder. Study 2 probed the temporal dynamics of this relationship as participants engaged with emotional narratives. We found quantified changes in internal narratives led to changes in self-report, while the baseline narrative severity predicted the magnitude of subsequent affective change. By framing affect as a computational state, our results highlight its core, therapeutically pertinent functions: constraining the structure of internal narratives and integrating context to shape self-report.
>
---
#### [replaced 047] Discrete Stochastic Localization for Non-autoregressive Generation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言生成任务，解决非自回归生成中连续扩散模型效果不佳的问题。提出DSL框架，通过单位球嵌入实现与SNR无关的去噪，提升生成质量并支持多种采样方式。**

- **链接: [https://arxiv.org/pdf/2602.16169](https://arxiv.org/pdf/2602.16169)**

> **作者:** Yunshu Wu; Jiayi Cheng; Longxuan Yu; Partha Thakuria; Rob Brekelmans; Evangelos E. Papalexakis; Greg Ver Steeg
>
> **摘要:** Continuous diffusion is a natural framework for non-autoregressive generation but has generally lagged behind masked discrete diffusion models (MDMs) on discrete sequence generation. We argue that the bottleneck is not continuity itself, but a representation in which denoising depends on timestep-indexed noise regimes. We introduce \emph{Discrete Stochastic Localization} (DSL), a continuous-state framework with unit-sphere token embeddings whose Bayes-optimal denoiser is invariant to the nominal signal-to-noise ratio (SNR) under the localization channel. One trained network then supports an entire family of per-token SNR paths, with endpoint masked-diffusion paths as a special case. Fine-tuning a pretrained MDLM checkpoint with DSL substantially improves distributional faithfulness (MAUVE) on OpenWebText across all step budgets from $T{=}128$ to $T{=}1024$, and the same checkpoint supports random-order autoregressive sampling, as well as a hybrid continuous-then-discrete sampler using as few as T=48 total steps -- without distillation or retraining.
>
---
#### [replaced 048] VectraYX-Nano: A 42M-Parameter Spanish Cybersecurity Language Model with Curriculum Learning and Native Tool Use
- **分类: cs.CL**

- **简介: 该论文提出VectraYX-Nano，一个42M参数的西班牙语网络安全语言模型，解决西班牙语网络安全文本生成与工具调用问题，通过课程学习和原生工具集成提升性能。**

- **链接: [https://arxiv.org/pdf/2605.13989](https://arxiv.org/pdf/2605.13989)**

> **作者:** Juan S. Santillana
>
> **备注:** 24 pages, 5 figures, 12 tables. v3: post-Chinchilla compute ablation (v8-v15), Globant affiliation finalized, EMNLP Findings 2026 submission. Released model: VectraYX-Nano v7 (42M params, GGUF Q4 ~20 MB, native MCP)
>
> **摘要:** We present VectraYX-Nano, a 41.95M-parameter decoder-only language model trained from scratch in Spanish for cybersecurity, with a Latin-American regional focus and native tool invocation via the Model Context Protocol (MCP). The model has four contributions. (i) Corpus: VectraYX-Sec-ES, a 170M-token Spanish corpus assembled by an eight-VM distributed pipeline at ~$25 USD of cloud compute and split into three curriculum phases (conversational 42M, cybersecurity 118M, offensive tooling 10M). (ii) Architecture: a 42M Transformer decoder with GQA, QK-Norm, RMSNorm, SwiGLU, RoPE and z-loss, paired with a domain-balanced 16,384-token byte-fallback BPE. (iii) Curriculum with replay across the three phases yields a monotonic loss descent (9.80 -> 3.17 -> 3.00 -> 2.16); after SFT (loss 1.74) the v2 bootstrap-ablation reference attains a conversational gate of 0.775 +/- 0.043 on B5 over N=4 seeds, and a controlled Phase-2 replay sweep over {0,5,10,25,50}% saturates B5 at >=25% replay. (iv) Two empirical findings, both N=4. A controlled bootstrap-corpus ablation across v2 (OpenSubs), v4 (mC4-ES), and v6 (60/25/15 OpenSubs/mC4/Wiki) exposes a loss-versus-register inversion: lower-perplexity bootstraps yield measurably worse conversational behavior (v2 > v4 > v6 on B5 at every paired seed). The B4 (tool-selection) floor of 0.000 is a corpus-density artifact, not a capacity gate: rebalancing the SFT mixture to tool-use ratio 1:21 yields VectraYX-Nano v7, the released headline configuration, reaching B4 = 0.230 +/- 0.052 at 42M while retaining B1 = 0.332 +/- 0.005 and B5 = 0.725 +/- 0.130; a LoRA replication on a 260M from-scratch mid-tier reaches 0.445 +/- 0.201. The released GGUF is 96 MB in F16, runs sub-second TTFT on commodity hardware under this http URL, and is, to our knowledge, the first published Spanish-native cybersecurity LLM with end-to-end MCP integration.
>
---
#### [replaced 049] Sakura at BEA 2026 Shared Task 1: What Makes Vocabulary Difficult?
- **分类: cs.CL**

- **简介: 该论文属于词汇难度预测任务，旨在分析词汇难度影响因素。工作包括构建高精度黑盒模型和可解释模型，提升预测效果并揭示难度来源。**

- **链接: [https://arxiv.org/pdf/2605.14257](https://arxiv.org/pdf/2605.14257)**

> **作者:** Adam Nohejl; Xuanxin Wu; Yusuke Ide; Maria Angelica Riera Machin; Yi-Ning Chang; Hitomi Yanaka
>
> **备注:** To be published in Proceedings of the 21st Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2026)
>
> **摘要:** We describe two types of models for vocabulary difficulty prediction: a high-accuracy black-box model, which achieved the top shared task result in the open track, and an explainable model, which outperforms a fine-tuned encoder baseline. As the black-box model, we fine-tuned an LLM using a soft-target loss function for effective application to the rating task, achieving r > 0.91. The explainable model provides insights into what impacts the difficulty of each item while maintaining a strong correlation (r > 0.77). We further analyze the results, demonstrating that the difficulty of items in the British Council's Knowledge-based Vocabulary Lists (KVL) is often affected by spelling difficulty or the construction of the test items, in addition to the genuine production difficulty of the words. We make our code available online at this https URL .
>
---
#### [replaced 050] Fine-grained Claim-level RAG Benchmark for Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律领域RAG系统评估任务，旨在解决现有评估框架缺乏细粒度分析及语言与用户群体局限的问题。工作中提出了ClaimRAG-LAW数据集，并进行了细致的系统评估。**

- **链接: [https://arxiv.org/pdf/2605.21071](https://arxiv.org/pdf/2605.21071)**

> **作者:** Souvick Das; Sallam Abualhaija; Domenico Bianculli
>
> **摘要:** The rapid progress of large language models (LLMs) is shifting semantic search toward a question-answering paradigm, where users ask questions and LLMs generate responses. In high-stake domains such as law, retrieval-augmented generation (RAG) is commonly used to mitigate hallucinations in generated responses. Nonetheless, prior work shows that RAG systems, whether general-purpose or legal-specific, still hallucinate at varying rates, making fine-grained evaluation essential. Despite the need, existing evaluation frameworks for legal RAG systems lack the granularity required to provide detailed analysis of retrieval and generation performance separately. Moreover, current benchmarks are largely English-only and centered on legal expert queries, overlooking non-expert needs. We introduce ClaimRAG-LAW, a comprehensive dataset for legal RAG that supports French and English, targets both experts and non-experts, and includes diverse question types reflecting realistic scenarios. We further apply a fine-grained evaluation framework of state-of-the-art legal RAG systems, revealing limitations in retrieval, generation, and claim-level analysis in the legal domain.
>
---
#### [replaced 051] TextSeal: A Localized LLM Watermark for Provenance & Distillation Protection
- **分类: cs.CR; cs.CL; cs.LG**

- **简介: 本文提出TextSeal，一种用于模型溯源和蒸馏检测的水印技术。解决AI生成文本溯源与防篡改问题，通过双密钥生成和多区域定位实现高效无损水印。**

- **链接: [https://arxiv.org/pdf/2605.12456](https://arxiv.org/pdf/2605.12456)**

> **作者:** Tom Sander; Hongyan Chang; Tomáš Souček; Tuan Tran; Valeriu Lacatusu; Sylvestre-Alvise Rebuffi; Alexandre Mourachko; Surya Parimi; Christophe Ropers; Rashel Moritz; Vanessa Stark; Hady Elsahar; Pierre Fernandez
>
> **摘要:** We introduce TextSeal, a state-of-the-art watermark for large language models. Building on Gumbel-max sampling, TextSeal introduces dual-key generation to restore output diversity, along with entropy-weighted scoring and multi-region localization for improved detection. It supports serving optimizations such as speculative decoding and multi-token prediction, and does not add any inference overhead. TextSeal strictly dominates baselines like SynthID-text in detection strength and is robust to dilution, maintaining confident localized detection even in heavily mixed human/AI documents. The scheme is theoretically distortion-free, and evaluation across reasoning benchmarks confirms that it preserves downstream performance; while a multilingual human evaluation (6000 A/B comparisons, 5 languages) shows no perceptible quality difference. Beyond its use for provenance detection, TextSeal is also ``radioactive'': its watermark signal transfers through model distillation, enabling detection of unauthorized use.
>
---
#### [replaced 052] Fix the Structural Bottleneck: Context Compression via Explicit Information Transmission
- **分类: cs.CL**

- **简介: 该论文属于上下文压缩任务，解决长上下文大模型中token、内存和延迟成本过高的问题。提出ComprExIT框架，通过显式信息传输提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.03784](https://arxiv.org/pdf/2602.03784)**

> **作者:** Jiangnan Ye; Hanqi Yan; Zhenyi Shen; Heng Chang; Ye Mao; Yulan He
>
> **摘要:** Long-context LLM agents often struggle with growing token, memory, and latency costs, making efficient context compression essential for practical deployment. Existing LLM-as-a-compressor methods remain noticeably inferior to using the full context. We find that this gap partly stems from their inability to preserve contextual information effectively. In this work, we revisit context compression from a structural perspective and identify two key bottlenecks in standard LLM-based compressors: limited coordination among compression tokens during information aggregation, and layerwise dilution that weakens useful signals from intermediate hidden states. To address these limitations, we propose ComprExIT, a new context compression framework based on explicit information transmission. ComprExIT adaptively selects features across frozen LLM layers, then allocates information from anchors to compression slots through a globally coordinated transport plan. Experiments on 12 datasets show that ComprExIT consistently outperforms strong soft-compression baselines, improving average F1 by up to 18.5%, while adding only ~1% trainable parameters and achieving more than 2x faster compression than the fastest baselines. The code will be released upon acceptance.
>
---
#### [replaced 053] Structural Anchor Pruning: Training-Free Multi-Vector Compression for Visual Document Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于视觉文档检索任务，解决多向量索引存储过高的问题。提出SAP方法，在无需训练的情况下高效压缩视觉标记，保留高检索性能。**

- **链接: [https://arxiv.org/pdf/2601.20107](https://arxiv.org/pdf/2601.20107)**

> **作者:** Zhuchenyang Liu; Ziyu Hu; Yao Zhang; Yu Xiao
>
> **备注:** methodology revision and new title
>
> **摘要:** Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive multi-vector index storage overhead. Existing training-free pruning methods either rely on heuristic layer choices or degrade sharply under aggressive compression, leading prior work to argue that effective high-compression pruning requires query-dependent training. We challenge this view with Structural Anchor Pruning (SAP), a self-calibrating, training-free, and query-agnostic index-time pruning framework with three components: (i) Score Retention (SR), a white-box per-layer compression diagnostic; (ii) SR-guided window selection, a procedure that automatically locates the structural pruning region for any backbone with no per-model hyperparameters; and (iii) a visual in-degree centrality scorer that identifies anchor patches within the selected window. On the ViDoRe v1/v2 benchmarks across three architectures spanning 18, 28, and 36 backbone layers, SAP retains over 90\% of NDCG@5 while pruning more than 90\% of visual tokens, without any per-model parameter tuning. Our layer-resolved SR analysis reveals an Alignment-Aggregation Divergence: the document's visual structure is preserved as a stable ``Structural Plateau'' within the backbone, but the final layers reshape this representation into a sparse, query-aligned form that is no longer suitable for pruning. This is the mechanistic reason SAP succeeds where final-layer methods fail.
>
---
#### [replaced 054] Herculean: An Agentic Benchmark for Financial Intelligence
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Herculean基准，用于评估AI在金融领域的代理智能。解决现有基准无法全面评估金融任务执行能力的问题，涵盖交易、对冲、市场洞察和审计四个流程。**

- **链接: [https://arxiv.org/pdf/2605.14355](https://arxiv.org/pdf/2605.14355)**

> **作者:** Xueqing Peng; Zhuohan Xie; Yupeng Cao; Haohang Li; Lingfei Qian; Yan Wang; Vincent Jim Zhang; Huan He; Xuguang Ai; Linhai Ma; Ruoyu Xiang; Yueru He; Yi Han; Shuyao Wang; Yuqing Guo; Mingyang Jiang; Yilun Zhao; Youzhong Dong; Xiaoyu Wang; Yankai Chen; Ye Yuan; Qiyuan Zhang; Fuyuan Lyu; Haolun Wu; Yonghan Yang; Zichen Zhao; Yuyang Dai; Fan Zhang; Rania Elbadry; Ayesha Gull; Muhammad Usman Safder; Nuo Chen; Fengbin Zhu; Tianshi Cai; Zimu Wang; Polydoros Giannouris; Yuechen Jiang; Zhiwei Liu; Mohsinul Kabir; Yuyan Wang; Yixiang Zheng; Yangyang Yu; Weijin Liu; Wenbo Cao; Anke Xu; Peng Lu; Jerry Huang; Mingquan Lin; Prayag Tiwari; Yijia Zhao; Victor Gutierrez Basulto; Xiao-Yang Liu; Kaleb E Smith; Jiahuan Pei; Arman Cohan; Jimin Huang; Yuehua Tang; Alejandro Lopez-Lira; Xi Chen; Xue Liu; Junichi Tsujii; Jian-Yun Nie; Sophia Ananiadou
>
> **摘要:** As AI agents improve, the central question is no longer whether they can solve isolated well-defined financial tasks, but whether they can reliably carry out financial professional work. Existing financial benchmarks offer only a partial view of this ability, as they primarily evaluate static competencies such as question answering, retrieval, summarization, and classification. We introduce Herculean, the first skilled benchmark for agentic financial intelligence spanning four representative workflows, including Trading, Hedging, Market Insights, and Auditing. Each workflow is instantiated as a standardized MCP-based skill environment with its own tools, interaction dynamics, constraints, and success criteria, enabling consistent end-to-end assessment of heterogeneous agent systems. Across frontier agents, we find agents perform relatively well on Trading and Market Insights, but struggle substantially on Hedging and Auditing, where long-horizon coordination, state consistency, and structured verification are critical. Overall, our results point to a key gap in current agents in turning financial reasoning into dependable workflow execution in high-stakes financial workflows.
>
---
#### [replaced 055] STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出StructSense，一个用于结构化信息提取的框架，解决专业领域中LLM泛化能力差的问题。通过集成符号知识、自我评估和人工验证，提升提取准确性与透明度。**

- **链接: [https://arxiv.org/pdf/2507.03674](https://arxiv.org/pdf/2507.03674)**

> **作者:** Tek Raj Chhetri; Yibei Chen; Puja Trivedi; Dorota Jarecka; Saif Haobsh; Patrick Ray; Lydia Ng; Satrajit S. Ghosh
>
> **备注:** -
>
> **摘要:** Extracting structured information from scientific literature is critical for accelerating discovery, yet Large Language Models (LLMs) often struggle in specialized domains that require expert knowledge and generalize poorly across tasks. We introduce \textsc{StructSense}, a modular, task-agnostic, open-source framework that integrates ontology-guided symbolic knowledge, agentic self-evaluative refinement, and human-in-the-loop validation for robust domain-aware extraction. We evaluate \textsc{StructSense} on three tasks of increasing semantic complexity: schema-based extraction of assessment instruments (91--100\% accuracy), metadata and resource extraction from scientific papers (86--93\% overall), and named entity recognition (NER) from neuroscience literature (58--75\% label accuracy across 8,882 entities). On two biomedical NER benchmarks (NCBI Disease and S800 Species), the system achieves $\geq$90\% relaxed recall and 62.5--85.8\% strict recall while extracting 1,000--3,600 additional entities beyond gold annotations. The local concept mapping service achieves Hits@1 of 62--82\% under strict matching and 68--86\% under semantic matching. These results across three domains demonstrate that \textsc{StructSense} generalizes across tasks while maintaining source grounding and provenance transparency.
>
---
#### [replaced 056] MTR-Bench: A Comprehensive Benchmark for Multi-Turn Reasoning Evaluation
- **分类: cs.CL**

- **简介: 该论文提出MTR-Bench，用于评估大语言模型的多轮推理能力。针对现有评估侧重单轮任务的问题，该工作构建了包含40任务、3600实例的基准，支持自动化评估，推动交互式AI研究。**

- **链接: [https://arxiv.org/pdf/2505.17123](https://arxiv.org/pdf/2505.17123)**

> **作者:** Xiaoyuan Li; Keqin Bao; Yubo Ma; Moxin Li; Wenjie Wang; Rui Men; Yichang Zhang; Fuli Feng; Dayiheng Liu
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Recent advances in Large Language Models (LLMs) have shown promising results in complex reasoning tasks. However, current evaluations predominantly focus on single-turn reasoning scenarios, leaving interactive tasks largely unexplored. We attribute it to the absence of comprehensive datasets and scalable automatic evaluation protocols. To fill these gaps, we present MTR-Bench for LLMs' Multi-Turn Reasoning evaluation. Comprising 4 classes, 40 tasks, and 3600 instances, MTR-Bench covers diverse reasoning capabilities, fine-grained difficulty granularity, and necessitates multi-turn interactions with the environments. Moreover, MTR-Bench features fully-automated framework spanning both dataset constructions and model evaluations, which enables scalable assessment without human interventions. Extensive experiments reveal that even the cutting-edge reasoning models fall short of multi-turn, interactive reasoning tasks. And the further analysis upon these results brings valuable insights for future research in interactive AI systems.
>
---
#### [replaced 057] SpecBlock: Block-Iterative Speculative Decoding with Dynamic Tree Drafting
- **分类: cs.CL**

- **简介: 该论文提出SpecBlock，解决LLM推理加速问题。通过块迭代和动态树草稿机制，在降低 drafting 成本的同时提升速度。**

- **链接: [https://arxiv.org/pdf/2605.07243](https://arxiv.org/pdf/2605.07243)**

> **作者:** Weijie Shi; Qiang Xu; Fan Deng; Yaguang Wu; Jiarun Liu; Yehong Xu; Hao Chen; Jia Zhu; Jiajie Xu; Xiangjun Huang; Jian Yang; Xiaofang Zhou
>
> **摘要:** Speculative decoding accelerates LLM inference by drafting a tree of candidate continuations and verifying it in one target forward. Existing drafters fall into two camps with opposite weaknesses. Autoregressive drafters such as EAGLE-3 preserve dependence along each draft path but call the drafter once per tree depth, making drafting a non-trivial share of per-iteration latency. Parallel drafters cut drafter calls by predicting multiple future positions in one forward, but each position is predicted without seeing the others, producing paths the verifier rejects. In this paper, we propose SpecBlock, a block-iterative drafter that combines path dependence with cheap drafting. Each drafter forward produces K dependent positions and we call this a block. The draft tree grows through repeated block expansions. Two mechanisms explicitly carry path dependence to keep later draft positions accurate. Within each block, a layer-wise shift carries the previous position's hidden state into every decoder layer. Across blocks, each new block can start from any position of the previous block, inheriting its hidden state to extend the path. To spend verifier budget where acceptance is likely, a co-trained rank head replaces the fixed top-k tree by allocating per-position branching during drafting. To avoid training the drafter on prefixes it never produces at inference, a valid-prefix mask drops the loss at later positions once an earlier one is wrong. Beyond static drafting, a cost-aware bandit at deployment uses free verifier feedback to update the drafter selectively, only when the expected throughput gain exceeds the update cost. Experiments show that SpecBlock improves mean speedup by 8-13% over EAGLE-3 at 44-52% of its drafting cost, and cost-aware adaptation extends this lead to 11-19%.
>
---
#### [replaced 058] Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于用户行为模拟任务，旨在解决真实世界行为模拟的挑战。通过构建OmniBehavior基准，揭示LLMs在长时序、跨场景行为模拟中的不足。**

- **链接: [https://arxiv.org/pdf/2604.08362](https://arxiv.org/pdf/2604.08362)**

> **作者:** Jiawei Chen; Ruoxi Xu; Boxi Cao; Ruotong Pan; Yunfei Zhang; Yifei Hu; Yong Du; Tingting Gao; Yaojie Lu; Yingfei Sun; Xianpei Han; Le Sun; Xiangyu Wu; Hongyu Lin
>
> **备注:** Project page: this https URL
>
> **摘要:** The emergence of Large Language Models (LLMs) has illuminated the potential for a general-purpose user simulator. However, existing benchmarks remain constrained to isolated scenarios, narrow action spaces, or synthetic data, failing to capture the holistic nature of authentic human behavior. To bridge this gap, we introduce OmniBehavior, the first user simulation benchmark constructed entirely from real-world data, integrating long-horizon, cross-scenario, and heterogeneous behavioral patterns into a unified framework. Based on this benchmark, we first provide empirical evidence that previous datasets with isolated scenarios suffer from tunnel vision, whereas real-world decision-making relies on long-term, cross-scenario causal chains. Extensive evaluations of state-of-the-art LLMs reveal that current models struggle to accurately simulate these complex behaviors, with performance plateauing even as context windows expand. Crucially, a systematic comparison between simulated and authentic behaviors uncovers a fundamental structural bias: LLMs tend to converge toward a positive average person, exhibiting hyper-activity, persona homogenization, and a utopian bias. This results in the loss of individual differences and long-tail behaviors, highlighting critical directions for future high-fidelity simulation research.
>
---
#### [replaced 059] Calibrating LLMs with Semantic-level Reward
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型校准任务，旨在解决模型输出不确定性估计不准确的问题。通过引入语义级奖励机制，提升模型在不同场景下的校准性能。**

- **链接: [https://arxiv.org/pdf/2605.15588](https://arxiv.org/pdf/2605.15588)**

> **作者:** Fengfei Yu; Ruijia Niu; Dongxia Wu; Yian Ma; Rose Yu
>
> **摘要:** As large language models (LLMs) are deployed in consequential settings such as medical question answering and legal reasoning, the ability to estimate when their outputs are likely to be correct is essential for safe and reliable use, requiring well-calibrated uncertainty. Standard reinforcement learning with verifiable rewards (RLVR) trains models with a binary correctness reward that is indifferent to confidence, providing no penalty for confident but wrong predictions and thereby degrading calibration. Recent work addresses this by training models to produce verbalized confidence scores alongside answers and rewarding agreement with correctness. However, verbalized confidence is calibrated at the token level and thus exhibits inconsistency across textual variations with same semantic meaning. We propose \textbf{Calibration with Semantic Reward (CSR)}, a framework that calibrates language models directly in semantic space without a verbalized confidence interface. CSR combines the correctness reward with a novel semantic calibration reward that encourages exploitation among correct rollouts by promoting semantic agreement, and exploration among incorrect ones by discouraging spurious consistency. Experiments across three model families on HotpotQA (in-distribution) and TriviaQA, MSMARCO, and NQ-Open (out-of-distribution) show that CSR consistently achieves lower ECE and higher AUROC than verbalized-confidence baselines across nearly all settings, reducing ECE by up to $40\%$ and improving AUROC by up to $31\%$ over verbalized-confidence baselines, with calibration behavior generalizing robustly across all four evaluation settings.
>
---
#### [replaced 060] NaviAgent: Graph-Driven Bilevel Planning for Scalable Tool Orchestration
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出NaviAgent，解决大规模工具调用中的任务规划与执行问题。通过图驱动的双层架构，提升工具链的可扩展性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.19500](https://arxiv.org/pdf/2506.19500)**

> **作者:** Yan Jiang; Hao Zhou; Lizhong GU; Tianlong Li; Ruinan Jin; Wanqi Zhou; Ai Han
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Large Language Models (LLMs) increasingly act as function-call agents that invoke external tools to tackle tasks beyond their static knowledge. However, they typically invoke tools one at a time without a global view of task structure. As tools often depend on one another, this leads to error accumulation and poor scalability, particularly when scaling to hundreds or thousands of tools. To address these limitations, we propose NaviAgent, an explicit bilevel architecture that decouples task planning from tool execution through graph-based modeling of tool relations. At the planning level, the LLM-based agent decides whether to respond directly, clarify intent, or retrieve and execute a toolchain independent of inter-tool complexity. At the execution level, a Tool World Navigation Model (TWNM) encodes structural and behavioral relations among tools, steering the agent to compose scalable and robust invocation sequences. Incorporating feedback from real tool interactions, NaviAgent achieves closed-loop alignment between planning and execution, enabling adaptive navigation in large-scale tool ecosystems. Evaluations on API-Bank and ToolBench show consistent improvements in task success rate (TSR), with TWNM yielding an average gain of 13.1 points on complex tasks. Further tests on 50 real APIs across 7 domains show consistent gains of 4.3--12.0 points, with fewer steps and latency, demonstrating robust generalization under real-world dynamics.
>
---
