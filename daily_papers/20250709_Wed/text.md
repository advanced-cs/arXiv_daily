# 自然语言处理 cs.CL

- **最新发布 84 篇**

- **更新 66 篇**

## 最新发布

#### [new 001] Entropy-Memorization Law: Evaluating Memorization Difficulty of Data in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs中数据记忆难度，提出熵与记忆分数的线性关系，解决数据区分问题，提出Dataset Inference方法。**

- **链接: [http://arxiv.org/pdf/2507.06056v1](http://arxiv.org/pdf/2507.06056v1)**

> **作者:** Yizhan Huang; Zhe Yang; Meifang Chen; Jianping Zhang; Michael R. Lyu
>
> **摘要:** Large Language Models (LLMs) are known to memorize portions of their training data, sometimes reproducing content verbatim when prompted appropriately. In this work, we investigate a fundamental yet under-explored question in the domain of memorization: How to characterize memorization difficulty of training data in LLMs? Through empirical experiments on OLMo, a family of open models, we present the Entropy-Memorization Law. It suggests that data entropy is linearly correlated with memorization score. Moreover, in a case study of memorizing highly randomized strings, or "gibberish", we observe that such sequences, despite their apparent randomness, exhibit unexpectedly low empirical entropy compared to the broader training corpus. Adopting the same strategy to discover Entropy-Memorization Law, we derive a simple yet effective approach to distinguish training and testing data, enabling Dataset Inference (DI).
>
---
#### [new 002] Skywork-R1V3 Technical Report
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在提升模型的视觉推理能力。通过强化学习后训练框架，有效迁移文本推理能力至视觉任务，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2507.06167v1](http://arxiv.org/pdf/2507.06167v1)**

> **作者:** Wei Shen; Jiangbo Pei; Yi Peng; Xuchen Song; Yang Liu; Jian Peng; Haofeng Sun; Yunzhuo Hao; Peiyu Wang; Yahui Zhou
>
> **摘要:** We introduce Skywork-R1V3, an advanced, open-source vision-language model (VLM) that pioneers a new approach to visual reasoning. Its key innovation lies in effectively transferring reasoning skills from text-only Large Language Models (LLMs) to visual tasks. The strong performance of Skywork-R1V3 primarily stems from our elaborate post-training RL framework, which effectively activates and enhances the model's reasoning ability, without the need for additional continue pre-training. Through this framework, we further uncover the fundamental role of the connector module in achieving robust cross-modal alignment for multimodal reasoning models. In addition, we introduce a unique indicator of reasoning capability, the entropy of critical reasoning tokens, which has proven highly effective for checkpoint selection during RL training. Skywork-R1V3 achieves state-of-the-art results on MMMU, significantly improving from 64.3% to 76.0%. This performance matches entry-level human capabilities. Remarkably, our RL-powered post-training approach enables even the 38B parameter model to rival top closed-source VLMs. The implementation successfully transfers mathematical reasoning to other subject-related reasoning tasks. We also include an analysis of curriculum learning and reinforcement finetuning strategies, along with a broader discussion on multimodal reasoning. Skywork-R1V3 represents a significant leap in multimodal reasoning, showcasing RL as a powerful engine for advancing open-source VLM capabilities.
>
---
#### [new 003] Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能领域，解决语言代理在跨领域任务中错误修正和经验复用的问题。提出Agent KB框架，通过Reason-Retrieve-Refine流程实现知识共享与迁移。**

- **链接: [http://arxiv.org/pdf/2507.06229v1](http://arxiv.org/pdf/2507.06229v1)**

> **作者:** Xiangru Tang; Tianrui Qin; Tianhao Peng; Ziyang Zhou; Daniel Shao; Tingting Du; Xinming Wei; Peng Xia; Fang Wu; He Zhu; Ge Zhang; Jiaheng Liu; Xingyao Wang; Sirui Hong; Chenglin Wu; Hao Cheng; Chi Wang; Wangchunshu Zhou
>
> **摘要:** As language agents tackle increasingly complex tasks, they struggle with effective error correction and experience reuse across domains. We introduce Agent KB, a hierarchical experience framework that enables complex agentic problem solving via a novel Reason-Retrieve-Refine pipeline. Agent KB addresses a core limitation: agents traditionally cannot learn from each other's experiences. By capturing both high-level strategies and detailed execution logs, Agent KB creates a shared knowledge base that enables cross-agent knowledge transfer. Evaluated on the GAIA benchmark, Agent KB improves success rates by up to 16.28 percentage points. On the most challenging tasks, Claude-3 improves from 38.46% to 57.69%, while GPT-4 improves from 53.49% to 73.26% on intermediate tasks. On SWE-bench code repair, Agent KB enables Claude-3 to improve from 41.33% to 53.33%. Our results suggest that Agent KB provides a modular, framework-agnostic infrastructure for enabling agents to learn from past experiences and generalize successful strategies to new tasks.
>
---
#### [new 004] Beyond classical and contemporary models: a transformative ai framework for student dropout prediction in distance learning using rag, prompt engineering, and cross-modal fusion
- **分类: cs.CL; cs.AI; cs.CY; cs.IR; I.2.7; I.2.1; K.3.1**

- **简介: 该论文属于学生辍学预测任务，旨在解决传统模型无法捕捉情感和上下文因素的问题。通过RAG、提示工程和跨模态融合提升预测准确性。**

- **链接: [http://arxiv.org/pdf/2507.05285v1](http://arxiv.org/pdf/2507.05285v1)**

> **作者:** Miloud Mihoubi; Meriem Zerkouk; Belkacem Chikhaoui
>
> **备注:** 10 pages, 5 figures, 5 tables. Submitted to the 38th Canadian Conference on Artificial Intelligence (Canadian AI 2025)
>
> **摘要:** Student dropout in distance learning remains a critical challenge, with profound societal and economic consequences. While classical machine learning models leverage structured socio-demographic and behavioral data, they often fail to capture the nuanced emotional and contextual factors embedded in unstructured student interactions. This paper introduces a transformative AI framework that redefines dropout prediction through three synergistic innovations: Retrieval-Augmented Generation (RAG) for domain-specific sentiment analysis, prompt engineering to decode academic stressors, and cross-modal attention fusion to dynamically align textual, behavioral, and socio-demographic insights. By grounding sentiment analysis in a curated knowledge base of pedagogical content, our RAG-enhanced BERT model interprets student comments with unprecedented contextual relevance, while optimized prompts isolate indicators of academic distress (e.g., "isolation," "workload anxiety"). A cross-modal attention layer then fuses these insights with temporal engagement patterns, creating holistic risk profiles. Evaluated on a longitudinal dataset of 4 423 students, the framework achieves 89% accuracy and an F1-score of 0.88, outperforming conventional models by 7% and reducing false negatives by 21%. Beyond prediction, the system generates interpretable interventions by retrieving contextually aligned strategies (e.g., mentorship programs for isolated learners). This work bridges the gap between predictive analytics and actionable pedagogy, offering a scalable solution to mitigate dropout risks in global education systems
>
---
#### [new 005] Psychometric Item Validation Using Virtual Respondents with Trait-Response Mediators
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理测量任务，旨在解决LLM问卷项目有效性验证问题。通过模拟虚拟被试与特质-反应中介因素，提升问卷项目质量。**

- **链接: [http://arxiv.org/pdf/2507.05890v1](http://arxiv.org/pdf/2507.05890v1)**

> **作者:** Sungjib Lim; Woojung Song; Eun-Ju Lee; Yohan Jo
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** As psychometric surveys are increasingly used to assess the traits of large language models (LLMs), the need for scalable survey item generation suited for LLMs has also grown. A critical challenge here is ensuring the construct validity of generated items, i.e., whether they truly measure the intended trait. Traditionally, this requires costly, large-scale human data collection. To make it efficient, we present a framework for virtual respondent simulation using LLMs. Our central idea is to account for mediators: factors through which the same trait can give rise to varying responses to a survey item. By simulating respondents with diverse mediators, we identify survey items that robustly measure intended traits. Experiments on three psychological trait theories (Big5, Schwartz, VIA) show that our mediator generation methods and simulation framework effectively identify high-validity items. LLMs demonstrate the ability to generate plausible mediators from trait definitions and to simulate respondent behavior for item validation. Our problem formulation, metrics, methodology, and dataset open a new direction for cost-effective survey development and a deeper understanding of how LLMs replicate human-like behavior. We will publicly release our dataset and code to support future work.
>
---
#### [new 006] We Should Evaluate Real-World Impact
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨NLP系统实际影响评估不足的问题。研究指出ACL会议中仅0.1%的论文关注真实世界影响，多数仅做指标评估，呼吁加强实际应用效果的研究。**

- **链接: [http://arxiv.org/pdf/2507.05973v1](http://arxiv.org/pdf/2507.05973v1)**

> **作者:** Ehud Reiter
>
> **备注:** This paper will appear in Computational Linguistics journal as a "Last Word" opinion piece. The Arxiv version is a pre-MIT Press publication version
>
> **摘要:** The ACL community has very little interest in evaluating the real-world impact of NLP systems. A structured survey of the ACL Anthology shows that perhaps 0.1% of its papers contain such evaluations; furthermore most papers which include impact evaluations present them very sketchily and instead focus on metric evaluations. NLP technology would be more useful and more quickly adopted if we seriously tried to understand and evaluate its real-world impact.
>
---
#### [new 007] Agentic-R1: Distilled Dual-Strategy Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI推理任务，解决模型在数学和逻辑任务中的效率与准确性问题。通过多策略蒸馏方法训练Agentic-R1，实现动态选择推理策略。**

- **链接: [http://arxiv.org/pdf/2507.05707v1](http://arxiv.org/pdf/2507.05707v1)**

> **作者:** Weihua Du; Pranjal Aggarwal; Sean Welleck; Yiming Yang
>
> **备注:** Preprint. 15 pages. Project available at https://github.com/StigLidu/DualDistill
>
> **摘要:** Current long chain-of-thought (long-CoT) models excel at mathematical reasoning but rely on slow and error-prone natural language traces. Tool-augmented agents address arithmetic via code execution, but often falter on complex logical tasks. We introduce a fine-tuning framework, DualDistill, that distills complementary reasoning strategies from multiple teachers into a unified student model. Using this approach, we train Agentic-R1, which dynamically selects the optimal strategy for each query, invoking tools for arithmetic and algorithmic problems, and using text-based reasoning for abstract ones. Our method improves accuracy across a range of tasks, including both computation-intensive and standard benchmarks, demonstrating the effectiveness of multi-strategy distillation in achieving robust and efficient reasoning. Our project is available at https://github.com/StigLidu/DualDistill
>
---
#### [new 008] Flippi: End To End GenAI Assistant for E-Commerce
- **分类: cs.CL; I.2.7; H.3.3**

- **简介: 该论文属于对话系统任务，旨在解决电商中产品搜索效率低的问题。通过自然语言处理技术，Flippi提供个性化购物体验，提升用户决策效率与满意度。**

- **链接: [http://arxiv.org/pdf/2507.05788v1](http://arxiv.org/pdf/2507.05788v1)**

> **作者:** Anand A. Rajasekar; Praveen Tangarajan; Anjali Nainani; Amogh Batwal; Vinay Rao Dandin; Anusua Trivedi; Ozan Ersoy
>
> **备注:** 10 pages, 2 figures, 7 tables
>
> **摘要:** The emergence of conversational assistants has fundamentally reshaped user interactions with digital platforms. This paper introduces Flippi-a cutting-edge, end-to-end conversational assistant powered by large language models (LLMs) and tailored for the e-commerce sector. Flippi addresses the challenges posed by the vast and often overwhelming product landscape, enabling customers to discover products more efficiently through natural language dialogue. By accommodating both objective and subjective user requirements, Flippi delivers a personalized shopping experience that surpasses traditional search methods. This paper details how Flippi interprets customer queries to provide precise product information, leveraging advanced NLP techniques such as Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG), Named Entity Recognition (NER), and Context Reduction. Flippi's unique capability to identify and present the most attractive offers on an e-commerce site is also explored, demonstrating how it empowers users to make cost-effective decisions. Additionally, the paper discusses Flippi's comparative analysis features, which help users make informed choices by contrasting product features, prices, and other relevant attributes. The system's robust architecture is outlined, emphasizing its adaptability for integration across various e-commerce platforms and the technological choices underpinning its performance and accuracy. Finally, a comprehensive evaluation framework is presented, covering performance metrics, user satisfaction, and the impact on customer engagement and conversion rates. By bridging the convenience of online shopping with the personalized assistance traditionally found in physical stores, Flippi sets a new standard for customer satisfaction and engagement in the digital marketplace.
>
---
#### [new 009] "Lost-in-the-Later": Framework for Quantifying Contextual Grounding in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，研究大语言模型在问答中对上下文和参数知识的整合问题。提出CoPE框架，发现模型存在“lost-in-the-later”现象，即忽略后续信息，并设计方法提升上下文利用。**

- **链接: [http://arxiv.org/pdf/2507.05424v1](http://arxiv.org/pdf/2507.05424v1)**

> **作者:** Yufei Tao; Adam Hiatt; Rahul Seetharaman; Ameeta Agrawal
>
> **摘要:** Large language models are capable of leveraging both contextual and parametric knowledge but how they prioritize and integrate these sources remains underexplored. We introduce CoPE, a novel evaluation framework that systematically measures contextual knowledge (CK) and parametric knowledge (PK) across models and languages. Using our MultiWikiAtomic dataset in English, Spanish, and Danish, we analyze how large language models (LLMs) integrate context, prioritize information, and incorporate PK in open-ended question answering. Our analysis uncovers a phenomenon we call lost-in-the-later, where LLMs tend to overlook or deprioritize information that appears later in a given context, revealing a strong positional bias that affects contextual grounding. We further find that reasoning models, as well as non-reasoning models prompted with chain-of-thought (CoT), use context even less than non-reasoning models without CoT and fail to mitigate the lost-in-the-later effect. CoT prompting, in particular, results in lower recall and shorter responses, leading to degraded contextual grounding. Based on these insights, we design prompt-based methods to effectively leverage input context. A case study applying CoPE to summarization demonstrates that CK-informed prompting improves factual grounding and reduces hallucination.
>
---
#### [new 010] LoRA-Augmented Generation (LAG) for Knowledge-Intensive Language Tasks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识密集型语言任务，解决如何高效利用预训练模型和LoRA适配器的问题。提出LAG方法，在无需额外训练的情况下筛选并应用专家模型。**

- **链接: [http://arxiv.org/pdf/2507.05346v1](http://arxiv.org/pdf/2507.05346v1)**

> **作者:** William Fleshman; Benjamin Van Durme
>
> **摘要:** The proliferation of fine-tuned language model experts for specific tasks and domains signals the need for efficient selection and combination methods. We propose LoRA-Augmented Generation (LAG) for leveraging large libraries of knowledge and task-specific LoRA adapters. LAG requires no additional training or access to data, and efficiently filters, retrieves, and applies experts on a per-token and layer basis. We evaluate LAG on various knowledge-intensive tasks, achieving superior performance over existing data-free methods. We explore scenarios where additional data is available, demonstrating LAG's compatibility with alternative solutions such as retrieval-augmented generation (RAG).
>
---
#### [new 011] Towards a Principled Evaluation of Knowledge Editors
- **分类: cs.CL**

- **简介: 该论文属于知识编辑任务，探讨评估方法的可靠性与公平性，揭示不同指标和批次大小对编辑器排名的影响，并指出字符串匹配方法存在误报问题。**

- **链接: [http://arxiv.org/pdf/2507.05937v1](http://arxiv.org/pdf/2507.05937v1)**

> **作者:** Sebastian Pohl; Max Ploner; Alan Akbik
>
> **备注:** Accepted at L2M2 workshop at ACL 2025
>
> **摘要:** Model editing has been gaining increasing attention over the past few years. For Knowledge Editing in particular, more challenging evaluation datasets have recently been released. These datasets use different methodologies to score the success of editors. Yet, it remains under-explored how robust these methodologies are and whether they unfairly favor some editors. Moreover, the disruptive impact of these editors on overall model capabilities remains a constant blind spot. We address both of these problems and show that choosing different metrics and evaluation methodologies as well as different edit batch sizes can lead to a different ranking of knowledge editors. Crucially we demonstrate this effect also on general language understanding tasks evaluated alongside the knowledge editing tasks. Further we include a manual assessment of the string matching based evaluation method for knowledge editing that is favored by recently released datasets, revealing a tendency to produce false positive matches.
>
---
#### [new 012] DRAGON: Dynamic RAG Benchmark On News
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG评估任务，解决俄语RAG系统缺乏动态基准的问题。构建了DRAGON基准，支持动态新闻数据的全面评估。**

- **链接: [http://arxiv.org/pdf/2507.05713v1](http://arxiv.org/pdf/2507.05713v1)**

> **作者:** Fedor Chernogorskii; Sergei Averkiev; Liliya Kudraleeva; Zaven Martirosian; Maria Tikhonova; Valentin Malykh; Alena Fenogenova
>
> **摘要:** Retrieval-Augmented Generation (RAG) is a widely adopted approach for improving the factuality of large language models (LLMs) by incorporating external knowledge at inference time. Although there exist multiple RAG benchmarks for English, evaluation resources for other languages, including Russian, remain scarce and static, failing to capture the dynamic nature of real-world deployments. In this work, we present DRAGON (Dynamic RAG Benchmark On News), the first dynamic benchmark for evaluating RAG systems in Russian on a changing news corpora. DRAGON is built upon a regularly updated corpus of Russian news and public documents and supports comprehensive evaluation of both the retriever and generator components. Question generation is performed automatically with the use of Knowledge Graph constructed from the corpus and enables the extraction of four core question types aligned with distinct subgraph patterns. We release a complete evaluation framework comprising the pipeline for automatic question generation, evaluation scripts, which are potentially reusable for other languages and multilingual settings, and benchmark data. We also launch a public leaderboard to encourage community participation and comparison.
>
---
#### [new 013] Enhancing Test-Time Scaling of Large Language Models with Hierarchical Retrieval-Augmented MCTS
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的测试时扩展能力。通过引入层次化检索增强推理框架R2-LLMs，解决推理准确性和泛化能力不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.05557v1](http://arxiv.org/pdf/2507.05557v1)**

> **作者:** Alex ZH Dou; Zhongwei Wan; Dongfei Cui; Xin Wang; Jing Xiong; Haokun Lin; Chaofan Tao; Shen Yan; Mi Zhang
>
> **备注:** Technical Report
>
> **摘要:** Test-time scaling has emerged as a promising paradigm in language modeling, leveraging additional computational resources at inference time to enhance model performance. In this work, we introduce R2-LLMs, a novel and versatile hierarchical retrieval-augmented reasoning framework designed to improve test-time scaling in large language models (LLMs) without requiring distillation from more advanced models to obtain chain-of-thought (CoT) training data. R2-LLMs enhances inference-time generalization by integrating dual-level retrieval-based in-context learning: (1) At the coarse level, our approach extracts abstract templates from complex reasoning problems and retrieves similar problem-answer pairs to facilitate high-level in-context learning; (2) At the fine level, during Monte Carlo Tree Search (MCTS), R2-LLMs efficiently retrieves analogous intermediate solution steps from reference mathematical problem datasets, refining step-wise reasoning with the aid of a process reward model (PRM) for scoring. R2-LLMs is a robust hierarchical reasoning-augmentation method that enhances in-context-level reasoning while seamlessly integrating with step-level tree search methods. Utilizing PRM, it refines both candidate generation and decision-making for improved reasoning accuracy. Empirical evaluations on the MATH500, GSM8K, and OlympiadBench-TO datasets achieve substantial relative improvement with an increase of up to 16% using LLaMA-3.1-8B compared to the baselines, showcasing the effectiveness of our approach in complex reasoning tasks.
>
---
#### [new 014] HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决RAG模型在信息筛选、组合和推理方面的不足，提出HIRAG方法提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.05714v1](http://arxiv.org/pdf/2507.05714v1)**

> **作者:** YiHan Jiao; ZheHao Tan; Dan Yang; DuoLin Sun; Jie Feng; Jian Wang; Peng Wei
>
> **摘要:** Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA.
>
---
#### [new 015] Smoothie-Qwen: Post-Hoc Smoothing to Reduce Language Bias in Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型任务，解决语言混淆问题。通过后处理方法Smoothie-Qwen调整输出概率，减少非目标语言生成，提升语言可控性。**

- **链接: [http://arxiv.org/pdf/2507.05686v1](http://arxiv.org/pdf/2507.05686v1)**

> **作者:** SeungWon Ji; Jungyup Lee; Jemin Kim; Sang Park; SeungJae Lee
>
> **摘要:** Multilingual large language models (LLMs) often exhibit language confusion, a tendency to generate responses in a dominant language irrespective of the prompt's language. To address this, we propose Smoothie-Qwen, a lightweight, post-hoc method that mitigates language bias without retraining. This technique selectively adjusts token-level output probabilities to effectively suppress undesired language generation. Applied to the Qwen model, our method reduces unintended Chinese output by over 95% while preserving task accuracy on multilingual benchmarks. This work provides a practical and efficient solution for enhancing the language controllability of LLMs, making them more reliable for global applications.
>
---
#### [new 016] Coding Triangle: How Does Large Language Model Understand Code?
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码理解任务，旨在评估大语言模型的编程能力。通过构建Code Triangle框架，分析模型在编辑分析、代码实现和测试用例生成方面的表现，揭示其与人类差异，并提出改进方法。**

- **链接: [http://arxiv.org/pdf/2507.06138v1](http://arxiv.org/pdf/2507.06138v1)**

> **作者:** Taolin Zhang; Zihan Ma; Maosong Cao; Junnan Liu; Songyang Zhang; Kai Chen
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress in code generation, yet their true programming competence remains underexplored. We introduce the Code Triangle framework, which systematically evaluates LLMs across three fundamental dimensions: editorial analysis, code implementation, and test case generation. Through extensive experiments on competitive programming benchmarks, we reveal that while LLMs can form a self-consistent system across these dimensions, their solutions often lack the diversity and robustness of human programmers. We identify a significant distribution shift between model cognition and human expertise, with model errors tending to cluster due to training data biases and limited reasoning transfer. Our study demonstrates that incorporating human-generated editorials, solutions, and diverse test cases, as well as leveraging model mixtures, can substantially enhance both the performance and robustness of LLMs. Furthermore, we reveal both the consistency and inconsistency in the cognition of LLMs that may facilitate self-reflection and self-improvement, providing a potential direction for developing more powerful coding models.
>
---
#### [new 017] Evolution without Large Models: Training Language Model with Task Principles
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决数据增强成本高和碳排放问题。通过任务原则生成与实例生成，提升模型性能并减少对大模型的依赖。**

- **链接: [http://arxiv.org/pdf/2507.05991v1](http://arxiv.org/pdf/2507.05991v1)**

> **作者:** Minghang Zhu; Shen Gao; Zhengliang Shi; Jiabao Fang; Pengjie Ren; Zhaochun Ren; Zhumin Chen; Shuo Shang
>
> **摘要:** A common training approach for language models involves using a large-scale language model to expand a human-provided dataset, which is subsequently used for model training.This method significantly reduces training costs by eliminating the need for extensive human data annotation. However, it still faces challenges such as high carbon emissions during data augmentation and the risk of data leakage when we use closed-source LLMs. To address these issues, we propose a self-evolution method for language models. First, we introduce the Multi-level Principle Generation, which enables a large-scale model to summarize task-completion principles based on a small amount of task data. Then, we propose the Principle-based Instance Generation, in which a smaller-scale language model uses these task principles to generate a large amount of data. This data is then used for model training. Experimental results show that our proposed method significantly improves model performance compared to directly using a smaller-scale language model to generate data. Additionally, since we only use the large-scale language model to generate the task-completion principles, the carbon emissions associated with training the model are greatly reduced.
>
---
#### [new 018] Efficiency-Effectiveness Reranking FLOPs for LLM-based Rerankers
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于信息检索任务，旨在解决LLM rerankers的效率-效果评估问题。提出E²R-FLOPs指标及FLOPs估算方法，以更准确评估模型性能。**

- **链接: [http://arxiv.org/pdf/2507.06223v1](http://arxiv.org/pdf/2507.06223v1)**

> **作者:** Zhiyuan Peng; Ting-ruen Wei; Tingyu Song; Yilun Zhao; Yi Fang
>
> **备注:** under review
>
> **摘要:** Large Language Models (LLMs) have recently been applied to reranking tasks in information retrieval, achieving strong performance. However, their high computational demands often hinder practical deployment. Existing studies evaluate the efficiency of LLM-based rerankers using proxy metrics such as latency, the number of forward passes, input tokens, and output tokens. However, these metrics depend on hardware and running-time choices (\eg parallel or not, batch size, etc), and often fail to account for model size, making it difficult to interpret and obscuring the evaluation of the efficiency-effectiveness tradeoff. To address this issue, we propose E\textsuperscript{2}R-FLOPs, for LLM-based rerankers: ranking metrics per PetaFLOP (RPP) for relevance per compute and queries per PetaFLOP (QPP) for hardware-agnostic throughput. Companied with the new metrics, an interpretable FLOPs estimator is built to estimate the FLOPs of an LLM-based reranker even without running any experiments. Based on the proposed metrics, we conduct comprehensive experiments to evaluate a wide range of LLM-based rerankers with different architecture, studying the efficiency-effectiveness trade-off and bringing this issue to the attention of the research community.
>
---
#### [new 019] DocIE@XLLM25: In-Context Learning for Information Extraction using Fully Synthetic Demonstrations
- **分类: cs.CL**

- **简介: 该论文属于文档级实体与关系抽取任务，解决零样本或少样本下标注数据稀缺的问题。通过合成数据生成和上下文学习方法，构建高质量演示数据库并提升抽取性能。**

- **链接: [http://arxiv.org/pdf/2507.05997v1](http://arxiv.org/pdf/2507.05997v1)**

> **作者:** Nicholas Popovič; Ashish Kangen; Tim Schopf; Michael Färber
>
> **摘要:** Large, high-quality annotated corpora remain scarce in document-level entity and relation extraction in zero-shot or few-shot settings. In this paper, we present a fully automatic, LLM-based pipeline for synthetic data generation and in-context learning for document-level entity and relation extraction. In contrast to existing approaches that rely on manually annotated demonstrations or direct zero-shot inference, our method combines synthetic data generation with retrieval-based in-context learning, using a reasoning-optimized language model. This allows us to build a high-quality demonstration database without manual annotation and to dynamically retrieve relevant examples at inference time. Based on our approach we produce a synthetic dataset of over $5k$ Wikipedia abstracts with approximately $59k$ entities and $30k$ relation triples. Finally, we evaluate in-context learning performance on the DocIE shared task, extracting entities and relations from long documents in a zero-shot setting. We find that in-context joint entity and relation extraction at document-level remains a challenging task, even for state-of-the-art large language models.
>
---
#### [new 020] An Adaptive Supervised Contrastive Learning Framework for Implicit Sexism Detection in Digital Social Networks
- **分类: cs.CL**

- **简介: 该论文属于隐性性别歧视检测任务，旨在解决传统方法难以识别隐性性别歧视的问题。提出ASCEND框架，结合对比学习与分类损失，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.05271v1](http://arxiv.org/pdf/2507.05271v1)**

> **作者:** Mohammad Zia Ur Rehman; Aditya Shah; Nagendra Kumar
>
> **摘要:** The global reach of social media has amplified the spread of hateful content, including implicit sexism, which is often overlooked by conventional detection methods. In this work, we introduce an Adaptive Supervised Contrastive lEarning framework for implicit sexism detectioN (ASCEND). A key innovation of our method is the incorporation of threshold-based contrastive learning: by computing cosine similarities between embeddings, we selectively treat only those sample pairs as positive if their similarity exceeds a learnable threshold. This mechanism refines the embedding space by robustly pulling together representations of semantically similar texts while pushing apart dissimilar ones, thus reducing false positives and negatives. The final classification is achieved by jointly optimizing a contrastive loss with a cross-entropy loss. Textual features are enhanced through a word-level attention module. Additionally, we employ sentiment, emotion, and toxicity features. Evaluations on the EXIST2021 and MLSC datasets demonstrate that ASCEND significantly outperforms existing methods, with average Macro F1 improvements of 9.86%, 29.63%, and 32.51% across multiple tasks, highlighting its efficacy in capturing the subtle cues of implicit sexist language.
>
---
#### [new 021] Chat-Ghosting: A Comparative Study of Methods for Auto-Completion in Dialog Systems
- **分类: cs.CL**

- **简介: 该论文研究对话系统中的自动补全任务（Chat-Ghosting），旨在提升用户输入预测的准确性与效率。通过对比不同方法，分析其在不同数据集上的表现，并提出新的优化策略。**

- **链接: [http://arxiv.org/pdf/2507.05940v1](http://arxiv.org/pdf/2507.05940v1)**

> **作者:** Sandeep Mishra; Anubhab Mandal; Bishal Santra; Tushar Abhishek; Pawan Goyal; Manish Gupta
>
> **摘要:** Ghosting, the ability to predict a user's intended text input for inline query auto-completion, is an invaluable feature for modern search engines and chat interfaces, greatly enhancing user experience. By suggesting completions to incomplete queries (or prefixes), ghosting aids users with slow typing speeds, disabilities, or limited language proficiency. Ghosting is a challenging problem and has become more important with the ubiquitousness of chat-based systems like ChatGPT, Copilot, etc. Despite the increasing prominence of chat-based systems utilizing ghosting, this challenging problem of Chat-Ghosting has received little attention from the NLP/ML research community. There is a lack of standardized benchmarks and relative performance analysis of deep learning and non-deep learning methods. We address this through an open and thorough study of this problem using four publicly available dialog datasets: two human-human (DailyDialog and DSTC7-Ubuntu) and two human-bot (Open Assistant and ShareGPT). We experiment with various existing query auto-completion methods (using tries), n-gram methods and deep learning methods, with and without dialog context. We also propose a novel entropy-based dynamic early stopping strategy. Our analysis finds that statistical n-gram models and tries outperform deep learning based models in terms of both model performance and inference efficiency for seen prefixes. For unseen queries, neural models like T5 and Phi-2 lead to better results. Adding conversational context leads to significant improvements in ghosting quality, especially for Open-Assistant and ShareGPT. We make code and data publicly available
>
---
#### [new 022] Gendered Divides in Online Discussions about Reproductive Rights
- **分类: cs.CL; cs.CY**

- **简介: 该论文属于社会科学研究，探讨性别与地区因素如何影响在线生殖权利讨论。通过分析大量社交媒体数据，揭示性别差异及地域政治环境对观点表达的影响。**

- **链接: [http://arxiv.org/pdf/2507.05443v1](http://arxiv.org/pdf/2507.05443v1)**

> **作者:** Ashwin Rao; Sze Yuh Nina Wang; Kristina Lerman
>
> **摘要:** The U.S. Supreme Court's 2022 ruling in Dobbs v. Jackson Women's Health Organization marked a turning point in the national debate over reproductive rights. While the ideological divide over abortion is well documented, less is known about how gender and local sociopolitical contexts interact to shape public discourse. Drawing on nearly 10 million abortion-related posts on X (formerly Twitter) from users with inferred gender, ideology and location, we show that gender significantly moderates abortion attitudes and emotional expression, particularly in conservative regions, and independently of ideology. This creates a gender gap in abortion attitudes that grows more pronounced in conservative regions. The leak of the Dobbs draft opinion further intensified online engagement, disproportionately mobilizing pro-abortion women in areas where access was under threat. These findings reveal that abortion discourse is not only ideologically polarized but also deeply structured by gender and place, highlighting the central role of identity in shaping political expression during moments of institutional disruption.
>
---
#### [new 023] MindFlow: Revolutionizing E-commerce Customer Support with Multimodal LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于电商客服任务，旨在解决复杂多模态场景下的服务难题。提出MindFlow系统，集成记忆、决策与行动模块，提升客服效率与用户体验。**

- **链接: [http://arxiv.org/pdf/2507.05330v1](http://arxiv.org/pdf/2507.05330v1)**

> **作者:** Ming Gong; Xucheng Huang; Chenghan Yang; Xianhan Peng; Haoxin Wang; Yang Liu; Ling Jiang
>
> **摘要:** Recent advances in large language models (LLMs) have enabled new applications in e-commerce customer service. However, their capabilities remain constrained in complex, multimodal scenarios. We present MindFlow, the first open-source multimodal LLM agent tailored for e-commerce. Built on the CoALA framework, it integrates memory, decision-making, and action modules, and adopts a modular "MLLM-as-Tool" strategy for effect visual-textual reasoning. Evaluated via online A/B testing and simulation-based ablation, MindFlow demonstrates substantial gains in handling complex queries, improving user satisfaction, and reducing operational costs, with a 93.53% relative improvement observed in real-world deployments.
>
---
#### [new 024] DocTalk: Scalable Graph-based Dialogue Synthesis for Enhancing LLM Conversational Capabilities
- **分类: cs.CL**

- **简介: 该论文属于对话生成任务，旨在解决LLM在多轮对话中的能力不足问题。通过构建多轮对话数据集DocTalk，提升模型的上下文记忆与理解能力。**

- **链接: [http://arxiv.org/pdf/2507.05750v1](http://arxiv.org/pdf/2507.05750v1)**

> **作者:** Jing Yang Lee; Hamed Bonab; Nasser Zalmout; Ming Zeng; Sanket Lokegaonkar; Colin Lockard; Binxuan Huang; Ritesh Sarkhel; Haodong Wang
>
> **备注:** Accepted at SIGDIAL 2025
>
> **摘要:** Large Language Models (LLMs) are increasingly employed in multi-turn conversational tasks, yet their pre-training data predominantly consists of continuous prose, creating a potential mismatch between required capabilities and training paradigms. We introduce a novel approach to address this discrepancy by synthesizing conversational data from existing text corpora. We present a pipeline that transforms a cluster of multiple related documents into an extended multi-turn, multi-topic information-seeking dialogue. Applying our pipeline to Wikipedia articles, we curate DocTalk, a multi-turn pre-training dialogue corpus consisting of over 730k long conversations. We hypothesize that exposure to such synthesized conversational structures during pre-training can enhance the fundamental multi-turn capabilities of LLMs, such as context memory and understanding. Empirically, we show that incorporating DocTalk during pre-training results in up to 40% gain in context memory and understanding, without compromising base performance. DocTalk is available at https://huggingface.co/datasets/AmazonScience/DocTalk.
>
---
#### [new 025] ModelCitizens:Representing Community Voices in Online Safety
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于内容安全任务，旨在解决毒性语言检测中的社区视角缺失问题。通过构建包含多身份群体的标注数据集，并引入对话上下文，提升模型对多样性毒语的理解与检测能力。**

- **链接: [http://arxiv.org/pdf/2507.05455v1](http://arxiv.org/pdf/2507.05455v1)**

> **作者:** Ashima Suvarna; Christina Chance; Hamid Palangi; Sophie Hao; Thomas Hartvigsen; Saadia Gabriel
>
> **摘要:** Automatic toxic language detection is critical for creating safe, inclusive online spaces. However, it is a highly subjective task, with perceptions of toxic language shaped by community norms and lived experience. Existing toxicity detection models are typically trained on annotations that collapse diverse annotator perspectives into a single ground truth, erasing important context-specific notions of toxicity such as reclaimed language. To address this, we introduce MODELCITIZENS, a dataset of 6.8K social media posts and 40K toxicity annotations across diverse identity groups. To capture the role of conversational context on toxicity, typical of social media posts, we augment MODELCITIZENS posts with LLM-generated conversational scenarios. State-of-the-art toxicity detection tools (e.g. OpenAI Moderation API, GPT-o4-mini) underperform on MODELCITIZENS, with further degradation on context-augmented posts. Finally, we release LLAMACITIZEN-8B and GEMMACITIZEN-12B, LLaMA- and Gemma-based models finetuned on MODELCITIZENS, which outperform GPT-o4-mini by 5.5% on in-distribution evaluations. Our findings highlight the importance of community-informed annotation and modeling for inclusive content moderation.
>
---
#### [new 026] A Survey on Prompt Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型适应问题。通过分析prompt tuning方法，分类并比较不同框架，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2507.06085v1](http://arxiv.org/pdf/2507.06085v1)**

> **作者:** Zongqian Li; Yixuan Su; Nigel Collier
>
> **摘要:** This survey reviews prompt tuning, a parameter-efficient approach for adapting language models by prepending trainable continuous vectors while keeping the model frozen. We classify existing approaches into two categories: direct prompt learning and transfer learning. Direct prompt learning methods include: general optimization approaches, encoder-based methods, decomposition strategies, and mixture-of-experts frameworks. Transfer learning methods consist of: general transfer approaches, encoder-based methods, and decomposition strategies. For each method, we analyze method designs, innovations, insights, advantages, and disadvantages, with illustrative visualizations comparing different frameworks. We identify challenges in computational efficiency and training stability, and discuss future directions in improving training robustness and broadening application scope.
>
---
#### [new 027] A Survey on Latent Reasoning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM推理效率与表达受限问题，通过分析隐式推理方法提升模型的多步推理能力。**

- **链接: [http://arxiv.org/pdf/2507.06203v1](http://arxiv.org/pdf/2507.06203v1)**

> **作者:** Rui-Jie Zhu; Tianhao Peng; Tianhao Cheng; Xingwei Qu; Jinfa Huang; Dawei Zhu; Hao Wang; Kaiwen Xue; Xuanliang Zhang; Yong Shan; Tianle Cai; Taylor Kergan; Assel Kembay; Andrew Smith; Chenghua Lin; Binh Nguyen; Yuqi Pan; Yuhong Chou; Zefan Cai; Zhenhe Wu; Yongchi Zhao; Tianyu Liu; Jian Yang; Wangchunshu Zhou; Chujie Zheng; Chongxuan Li; Yuyin Zhou; Zhoujun Li; Zhaoxiang Zhang; Jiaheng Liu; Ge Zhang; Wenhao Huang; Jason Eshraghian
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive reasoning capabilities, especially when guided by explicit chain-of-thought (CoT) reasoning that verbalizes intermediate steps. While CoT improves both interpretability and accuracy, its dependence on natural language reasoning limits the model's expressive bandwidth. Latent reasoning tackles this bottleneck by performing multi-step inference entirely in the model's continuous hidden state, eliminating token-level supervision. To advance latent reasoning research, this survey provides a comprehensive overview of the emerging field of latent reasoning. We begin by examining the foundational role of neural network layers as the computational substrate for reasoning, highlighting how hierarchical representations support complex transformations. Next, we explore diverse latent reasoning methodologies, including activation-based recurrence, hidden state propagation, and fine-tuning strategies that compress or internalize explicit reasoning traces. Finally, we discuss advanced paradigms such as infinite-depth latent reasoning via masked diffusion models, which enable globally consistent and reversible reasoning processes. By unifying these perspectives, we aim to clarify the conceptual landscape of latent reasoning and chart future directions for research at the frontier of LLM cognition. An associated GitHub repository collecting the latest papers and repos is available at: https://github.com/multimodal-art-projection/LatentCoT-Horizon/.
>
---
#### [new 028] Conditional Multi-Stage Failure Recovery for Embodied Agents
- **分类: cs.CL**

- **简介: 该论文属于机器人任务执行领域，解决 embodied agents 执行失败的问题。提出一种条件多阶段恢复框架，利用 LLMs 进行零样本链式推理，提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2507.06016v1](http://arxiv.org/pdf/2507.06016v1)**

> **作者:** Youmna Farag; Svetlana Stoyanchev; Mohan Li; Simon Keizer; Rama Doddipatla
>
> **备注:** Accepted at REALM 2025
>
> **摘要:** Embodied agents performing complex tasks are susceptible to execution failures, motivating the need for effective failure recovery mechanisms. In this work, we introduce a conditional multistage failure recovery framework that employs zero-shot chain prompting. The framework is structured into four error-handling stages, with three operating during task execution and one functioning as a post-execution reflection phase. Our approach utilises the reasoning capabilities of LLMs to analyse execution challenges within their environmental context and devise strategic solutions. We evaluate our method on the TfD benchmark of the TEACH dataset and achieve state-of-the-art performance, outperforming a baseline without error recovery by 11.5% and surpassing the strongest existing model by 19%.
>
---
#### [new 029] Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音识别任务，解决MoE模型中专家协作不足的问题。通过引入共享路由器增强不同层专家的合作与专业化，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.05724v1](http://arxiv.org/pdf/2507.05724v1)**

> **作者:** Zijin Gu; Tatiana Likhomanenko; Navdeep Jaitly
>
> **摘要:** Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model \emph{Omni-router Transformer}. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
>
---
#### [new 030] On the Bias of Next-Token Predictors Toward Systematically Inefficient Reasoning: A Shortest-Path Case Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理研究，探讨次优路径对模型泛化的影响。通过对比不同推理轨迹，发现冗长但连贯的路径有助于提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.05362v1](http://arxiv.org/pdf/2507.05362v1)**

> **作者:** Riccardo Alberghi; Elizaveta Demyanenko; Luca Biggio; Luca Saglietti
>
> **摘要:** Recent advances in natural language processing highlight two key factors for improving reasoning in large language models (LLMs): (i) allocating more test-time compute tends to help on harder problems but often introduces redundancy in the reasoning trace, and (ii) compute is most effective when reasoning is systematic and incremental, forming structured chains of thought (CoTs) akin to human problem-solving. To study these factors in isolation, we introduce a controlled setting based on shortest-path tasks in layered graphs. We train decoder-only transformers on question-trace-answer triples using a custom tokenizer, comparing models trained on optimal bottom-up dynamic programming traces with those trained on longer, valid traces involving backtracking. Surprisingly, with the same training-token budget, models trained on inefficient traces generalize better to unseen graphs. This benefit is not due to length alone-injecting arbitrary redundancy into reasoning traces fails to help and can even hurt performance. Instead, we find that generalization correlates with the model's confidence in next-token prediction, suggesting that long, coherent, and locally incremental traces make the training signal easier to optimize.
>
---
#### [new 031] PhoniTale: Phonologically Grounded Mnemonic Generation for Typologically Distant Language Pairs
- **分类: cs.CL**

- **简介: 该论文属于跨语言记忆法生成任务，旨在解决语言差异带来的词汇学习困难。通过语音相似性检索和大模型生成，提出PhoniTale系统提升二语词汇记忆效果。**

- **链接: [http://arxiv.org/pdf/2507.05444v1](http://arxiv.org/pdf/2507.05444v1)**

> **作者:** Sana Kang; Myeongseok Gwon; Su Young Kwon; Jaewook Lee; Andrew Lan; Bhiksha Raj; Rita Singh
>
> **摘要:** Vocabulary acquisition poses a significant challenge for second-language (L2) learners, especially when learning typologically distant languages such as English and Korean, where phonological and structural mismatches complicate vocabulary learning. Recently, large language models (LLMs) have been used to generate keyword mnemonics by leveraging similar keywords from a learner's first language (L1) to aid in acquiring L2 vocabulary. However, most of this research has focused on native English speakers learning other languages, rather than the reverse. In this paper, we present PhoniTale, a novel cross-lingual mnemonic generation system that retrieves L1 keyword sequence based on phonological similarity and uses LLMs to generate mnemonics. We evaluate PhoniTale using both automated metrics and human evaluations, comparing its output to mnemonics created by humans and by previous automated approaches. To assess practical effectiveness, we also conduct a short-term recall test measuring mnemonic helpfulness. Our findings show that PhoniTale performs comparably to human-authored mnemonics. We also highlight key areas for future improvement in mnemonic quality and methodology.
>
---
#### [new 032] Few-shot text-based emotion detection
- **分类: cs.CL**

- **简介: 该论文属于文本情感检测任务，解决少样本情境下的情绪识别问题。团队使用大语言模型进行微调或提示工程，提升了多标签情绪检测效果。**

- **链接: [http://arxiv.org/pdf/2507.05918v1](http://arxiv.org/pdf/2507.05918v1)**

> **作者:** Teodor-George Marchitan; Claudiu Creanga; Liviu P. Dinu
>
> **摘要:** This paper describes the approach of the Unibuc - NLP team in tackling the SemEval 2025 Workshop, Task 11: Bridging the Gap in Text-Based Emotion Detection. We mainly focused on experiments using large language models (Gemini, Qwen, DeepSeek) with either few-shot prompting or fine-tuning. With our final system, for the multi-label emotion detection track (track A), we got an F1-macro of $0.7546$ (26/96 teams) for the English subset, $0.1727$ (35/36 teams) for the Portuguese (Mozambican) subset and $0.325$ (\textbf{1}/31 teams) for the Emakhuwa subset.
>
---
#### [new 033] CriticLean: Critic-Guided Reinforcement Learning for Mathematical Formalization
- **分类: cs.CL**

- **简介: 该论文属于数学形式化任务，旨在解决自然语言到形式代码的语义准确转换问题。提出CriticLean框架，通过强化学习提升评论阶段的准确性。**

- **链接: [http://arxiv.org/pdf/2507.06181v1](http://arxiv.org/pdf/2507.06181v1)**

> **作者:** Zhongyuan Peng; Yifan Yao; Kaijing Ma; Shuyue Guo; Yizhe Li; Yichi Zhang; Chenchen Zhang; Yifan Zhang; Zhouliang Yu; Luming Li; Minghao Liu; Yihang Xia; Jiawei Shen; Yuchen Wu; Yixin Cao; Zhaoxiang Zhang; Wenhao Huang; Jiaheng Liu; Ge Zhang
>
> **摘要:** Translating natural language mathematical statements into formal, executable code is a fundamental challenge in automated theorem proving. While prior work has focused on generation and compilation success, little attention has been paid to the critic phase-the evaluation of whether generated formalizations truly capture the semantic intent of the original problem. In this paper, we introduce CriticLean, a novel critic-guided reinforcement learning framework that elevates the role of the critic from a passive validator to an active learning component. Specifically, first, we propose the CriticLeanGPT, trained via supervised fine-tuning and reinforcement learning, to rigorously assess the semantic fidelity of Lean 4 formalizations. Then, we introduce CriticLeanBench, a benchmark designed to measure models' ability to distinguish semantically correct from incorrect formalizations, and demonstrate that our trained CriticLeanGPT models can significantly outperform strong open- and closed-source baselines. Building on the CriticLean framework, we construct FineLeanCorpus, a dataset comprising over 285K problems that exhibits rich domain diversity, broad difficulty coverage, and high correctness based on human evaluation. Overall, our findings highlight that optimizing the critic phase is essential for producing reliable formalizations, and we hope our CriticLean will provide valuable insights for future advances in formal mathematical reasoning.
>
---
#### [new 034] LCDS: A Logic-Controlled Discharge Summary Generation System Supporting Source Attribution and Expert Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗文本生成任务，旨在解决LLM生成病历摘要时的幻觉和源归属问题。通过构建源映射表和逻辑规则，提高摘要准确性与可追溯性。**

- **链接: [http://arxiv.org/pdf/2507.05319v1](http://arxiv.org/pdf/2507.05319v1)**

> **作者:** Cheng Yuan; Xinkai Rui; Yongqi Fan; Yawei Fan; Boyang Zhong; Jiacheng Wang; Weiyan Zhang; Tong Ruan
>
> **备注:** ACL Demo 2025
>
> **摘要:** Despite the remarkable performance of Large Language Models (LLMs) in automated discharge summary generation, they still suffer from hallucination issues, such as generating inaccurate content or fabricating information without valid sources. In addition, electronic medical records (EMRs) typically consist of long-form data, making it challenging for LLMs to attribute the generated content to the sources. To address these challenges, we propose LCDS, a Logic-Controlled Discharge Summary generation system. LCDS constructs a source mapping table by calculating textual similarity between EMRs and discharge summaries to constrain the scope of summarized content. Moreover, LCDS incorporates a comprehensive set of logical rules, enabling it to generate more reliable silver discharge summaries tailored to different clinical fields. Furthermore, LCDS supports source attribution for generated content, allowing experts to efficiently review, provide feedback, and rectify errors. The resulting golden discharge summaries are subsequently recorded for incremental fine-tuning of LLMs. Our project and demo video are in the GitHub repository https://github.com/ycycyc02/LCDS.
>
---
#### [new 035] OpenFActScore: Open-Source Atomic Evaluation of Factuality in Text Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成事实性评估任务，旨在解决闭源模型评估受限的问题。工作包括实现开放源代码的FActScore框架，支持使用开源模型进行事实验证。**

- **链接: [http://arxiv.org/pdf/2507.05965v1](http://arxiv.org/pdf/2507.05965v1)**

> **作者:** Lucas Fonseca Lage; Simon Ostermann
>
> **备注:** Submitted to EMNLP 2025 System Demonstrations track
>
> **摘要:** We introduce OpenFActScore, an open-source implementation of the FActScore framework for evaluating the factuality of text generated by large language models (LLMs). FActScore evaluates the factual accuracy of long-form text by using Atomic Fact Generation (AFG) to extract individual factual claims and Atomic Fact Validation (AFV) to verify each claim against a trusted knowledge source. While the original FActScore relies on closed-source and commercial models such as InstructGPT and ChatGPT, OpenFActScore enables the use of any Hugging Face-compatible model for both AFG and AFV. We provide a detailed technical overview of our implementation, highlighting design choices and modifications made to support open models. We evaluate multiple open-source LLMs on both AFG and AFV using the original FActScore benchmark, reporting BERTScore-F1 for AFG and Error Rate relative to human annotations for AFV. Our results show that open models can approximate the performance of closed-source systems, with Gemma achieving the best overall performance, and our final setup obtains a 0.99 Pearson correlation with the original FActScore experiments. OpenFActScore promotes transparency, reproducibility, and cost-effective evaluation, and is available at: https://github.com/lflage/OpenFActScore.
>
---
#### [new 036] Empowering Healthcare Practitioners with Language Models: Structuring Speech Transcripts in Two Real-World Clinical Applications
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医疗领域中的结构化报告与医嘱提取任务，针对数据不足和敏感性问题，提出解决方案并发布两个开源数据集。**

- **链接: [http://arxiv.org/pdf/2507.05517v1](http://arxiv.org/pdf/2507.05517v1)**

> **作者:** Jean-Philippe Corbeil; Asma Ben Abacha; George Michalopoulos; Phillip Swazinna; Miguel Del-Agua; Jerome Tremblay; Akila Jeeson Daniel; Cari Bader; Kevin Cho; Pooja Krishnan; Nathan Bodenstab; Thomas Lin; Wenxuan Teng; Francois Beaulieu; Paul Vozila
>
> **摘要:** Large language models (LLMs) such as GPT-4o and o1 have demonstrated strong performance on clinical natural language processing (NLP) tasks across multiple medical benchmarks. Nonetheless, two high-impact NLP tasks - structured tabular reporting from nurse dictations and medical order extraction from doctor-patient consultations - remain underexplored due to data scarcity and sensitivity, despite active industry efforts. Practical solutions to these real-world clinical tasks can significantly reduce the documentation burden on healthcare providers, allowing greater focus on patient care. In this paper, we investigate these two challenging tasks using private and open-source clinical datasets, evaluating the performance of both open- and closed-weight LLMs, and analyzing their respective strengths and limitations. Furthermore, we propose an agentic pipeline for generating realistic, non-sensitive nurse dictations, enabling structured extraction of clinical observations. To support further research in both areas, we release SYNUR and SIMORD, the first open-source datasets for nurse observation extraction and medical order extraction.
>
---
#### [new 037] Bridging Perception and Language: A Systematic Benchmark for LVLMs' Understanding of Amodal Completion Reports
- **分类: cs.CL**

- **简介: 该论文属于视觉语言理解任务，旨在研究LVLMs在amodal completion（非模态补全）方面的推理能力。通过构建基准测试，发现模型在不同物体类别和语言环境下的表现存在差异。**

- **链接: [http://arxiv.org/pdf/2507.05799v1](http://arxiv.org/pdf/2507.05799v1)**

> **作者:** Amane Watahiki; Tomoki Doi; Taiga Shinozaki; Satoshi Nishida; Takuya Niikawa; Katsunori Miyahara; Hitomi Yanaka
>
> **备注:** To appear in the Proceedings of the 47th Annual Meeting of the Cognitive Science Society (COGSCI 2025)
>
> **摘要:** One of the main objectives in developing large vision-language models (LVLMs) is to engineer systems that can assist humans with multimodal tasks, including interpreting descriptions of perceptual experiences. A central phenomenon in this context is amodal completion, in which people perceive objects even when parts of those objects are hidden. Although numerous studies have assessed whether computer-vision algorithms can detect or reconstruct occluded regions, the inferential abilities of LVLMs on texts related to amodal completion remain unexplored. To address this gap, we constructed a benchmark grounded in Basic Formal Ontology to achieve a systematic classification of amodal completion. Our results indicate that while many LVLMs achieve human-comparable performance overall, their accuracy diverges for certain types of objects being completed. Notably, in certain categories, some LLaVA-NeXT variants and Claude 3.5 Sonnet exhibit lower accuracy on original images compared to blank stimuli lacking visual content. Intriguingly, this disparity emerges only under Japanese prompting, suggesting a deficiency in Japanese-specific linguistic competence among these models.
>
---
#### [new 038] RabakBench: Scaling Human Annotations to Construct Localized Multilingual Safety Benchmarks for Low-Resource Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于多语言安全评估任务，旨在解决低资源语言中安全分类器性能不足的问题。通过构建RabakBench基准，实现多语言安全数据的高效标注与翻译。**

- **链接: [http://arxiv.org/pdf/2507.05980v1](http://arxiv.org/pdf/2507.05980v1)**

> **作者:** Gabriel Chua; Leanne Tan; Ziyu Ge; Roy Ka-Wei Lee
>
> **摘要:** Large language models (LLMs) and their safety classifiers often perform poorly on low-resource languages due to limited training data and evaluation benchmarks. This paper introduces RabakBench, a new multilingual safety benchmark localized to Singapore's unique linguistic context, covering Singlish, Chinese, Malay, and Tamil. RabakBench is constructed through a scalable three-stage pipeline: (i) Generate - adversarial example generation by augmenting real Singlish web content with LLM-driven red teaming; (ii) Label - semi-automated multi-label safety annotation using majority-voted LLM labelers aligned with human judgments; and (iii) Translate - high-fidelity translation preserving linguistic nuance and toxicity across languages. The final dataset comprises over 5,000 safety-labeled examples across four languages and six fine-grained safety categories with severity levels. Evaluations of 11 popular open-source and closed-source guardrail classifiers reveal significant performance degradation. RabakBench not only enables robust safety evaluation in Southeast Asian multilingual settings but also offers a reproducible framework for building localized safety datasets in low-resource environments. The benchmark dataset, including the human-verified translations, and evaluation code are publicly available.
>
---
#### [new 039] NeoBabel: A Multilingual Open Tower for Visual Generation
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于文本生成图像任务，旨在解决非英语语言支持不足的问题。通过多语言预训练和指令微调，提出NeoBabel模型，提升多语言生成性能与效率。**

- **链接: [http://arxiv.org/pdf/2507.06137v1](http://arxiv.org/pdf/2507.06137v1)**

> **作者:** Mohammad Mahdi Derakhshani; Dheeraj Varghese; Marzieh Fadaee; Cees G. M. Snoek
>
> **备注:** 34 pages, 12 figures
>
> **摘要:** Text-to-image generation advancements have been predominantly English-centric, creating barriers for non-English speakers and perpetuating digital inequities. While existing systems rely on translation pipelines, these introduce semantic drift, computational overhead, and cultural misalignment. We introduce NeoBabel, a novel multilingual image generation framework that sets a new Pareto frontier in performance, efficiency and inclusivity, supporting six languages: English, Chinese, Dutch, French, Hindi, and Persian. The model is trained using a combination of large-scale multilingual pretraining and high-resolution instruction tuning. To evaluate its capabilities, we expand two English-only benchmarks to multilingual equivalents: m-GenEval and m-DPG. NeoBabel achieves state-of-the-art multilingual performance while retaining strong English capability, scoring 0.75 on m-GenEval and 0.68 on m-DPG. Notably, it performs on par with leading models on English tasks while outperforming them by +0.11 and +0.09 on multilingual benchmarks, even though these models are built on multilingual base LLMs. This demonstrates the effectiveness of our targeted alignment training for preserving and extending crosslingual generalization. We further introduce two new metrics to rigorously assess multilingual alignment and robustness to code-mixed prompts. Notably, NeoBabel matches or exceeds English-only models while being 2-4x smaller. We release an open toolkit, including all code, model checkpoints, a curated dataset of 124M multilingual text-image pairs, and standardized multilingual evaluation protocols, to advance inclusive AI research. Our work demonstrates that multilingual capability is not a trade-off but a catalyst for improved robustness, efficiency, and cultural fidelity in generative AI.
>
---
#### [new 040] ECom-Bench: Can LLM Agent Resolve Real-World E-commerce Customer Support Issues?
- **分类: cs.CL**

- **简介: 该论文属于电商客服任务，旨在评估LLM代理处理真实场景的能力。提出ECom-Bench基准，包含动态用户模拟和真实对话数据，以挑战复杂电商问题。**

- **链接: [http://arxiv.org/pdf/2507.05639v1](http://arxiv.org/pdf/2507.05639v1)**

> **作者:** Haoxin Wang; Xianhan Peng; Xucheng Huang; Yizhe Huang; Ming Gong; Chenghan Yang; Yang Liu; Ling Jiang
>
> **摘要:** In this paper, we introduce ECom-Bench, the first benchmark framework for evaluating LLM agent with multimodal capabilities in the e-commerce customer support domain. ECom-Bench features dynamic user simulation based on persona information collected from real e-commerce customer interactions and a realistic task dataset derived from authentic e-commerce dialogues. These tasks, covering a wide range of business scenarios, are designed to reflect real-world complexities, making ECom-Bench highly challenging. For instance, even advanced models like GPT-4o achieve only a 10-20% pass^3 metric in our benchmark, highlighting the substantial difficulties posed by complex e-commerce scenarios. Upon publication, the code and data will be open-sourced to facilitate further research and development in this domain.
>
---
#### [new 041] TokenShapley: Token Level Context Attribution with Shapley Value
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型生成结果中特定关键词的上下文归属问题。提出TokenShapley方法，结合Shapley值与KNN技术实现细粒度的token级 attribution。**

- **链接: [http://arxiv.org/pdf/2507.05261v1](http://arxiv.org/pdf/2507.05261v1)**

> **作者:** Yingtai Xiao; Yuqing Zhu; Sirat Samyoun; Wanrong Zhang; Jiachen T. Wang; Jian Du
>
> **摘要:** Large language models (LLMs) demonstrate strong capabilities in in-context learning, but verifying the correctness of their generated responses remains a challenge. Prior work has explored attribution at the sentence level, but these methods fall short when users seek attribution for specific keywords within the response, such as numbers, years, or names. To address this limitation, we propose TokenShapley, a novel token-level attribution method that combines Shapley value-based data attribution with KNN-based retrieval techniques inspired by recent advances in KNN-augmented LLMs. By leveraging a precomputed datastore for contextual retrieval and computing Shapley values to quantify token importance, TokenShapley provides a fine-grained data attribution approach. Extensive evaluations on four benchmarks show that TokenShapley outperforms state-of-the-art baselines in token-level attribution, achieving an 11-23% improvement in accuracy.
>
---
#### [new 042] User Behavior Prediction as a Generic, Robust, Scalable, and Low-Cost Evaluation Strategy for Estimating Generalization in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在解决LLMs泛化能力测量难题。通过用户行为预测方法，提出一种新框架并进行实验验证。**

- **链接: [http://arxiv.org/pdf/2507.05266v1](http://arxiv.org/pdf/2507.05266v1)**

> **作者:** Sougata Saha; Monojit Choudhury
>
> **摘要:** Measuring the generalization ability of Large Language Models (LLMs) is challenging due to data contamination. As models grow and computation becomes cheaper, ensuring tasks and test cases are unseen during training phases will become nearly impossible. We argue that knowledge-retrieval and reasoning tasks are not ideal for measuring generalization, as LLMs are not trained for specific tasks. Instead, we propose user behavior prediction, also a key aspect of personalization, as a theoretically sound, scalable, and robust alternative. We introduce a novel framework for this approach and test it on movie and music recommendation datasets for GPT-4o, GPT-4o-mini, and Llama-3.1-8B-Instruct. Results align with our framework's predictions, showing GPT-4o outperforms GPT-4o-mini and Llama, though all models have much room for improvement, especially Llama.
>
---
#### [new 043] EduCoder: An Open-Source Annotation System for Education Transcript Data
- **分类: cs.CL**

- **简介: 该论文介绍EduCoder，一个用于教育对话转录文本的开源标注系统。任务是解决教育对话复杂标注问题，通过支持多类型标注和协作定义代码本，提升数据可靠性。**

- **链接: [http://arxiv.org/pdf/2507.05385v1](http://arxiv.org/pdf/2507.05385v1)**

> **作者:** Guanzhong Pan; Mei Tan; Hyunji Nam; Lucía Langlois; James Malamut; Liliana Deonizio; Dorottya Demszky
>
> **摘要:** We introduce EduCoder, a domain-specialized tool designed to support utterance-level annotation of educational dialogue. While general-purpose text annotation tools for NLP and qualitative research abound, few address the complexities of coding education dialogue transcripts -- with diverse teacher-student and peer interactions. Common challenges include defining codebooks for complex pedagogical features, supporting both open-ended and categorical coding, and contextualizing utterances with external features, such as the lesson's purpose and the pedagogical value of the instruction. EduCoder is designed to address these challenges by providing a platform for researchers and domain experts to collaboratively define complex codebooks based on observed data. It incorporates both categorical and open-ended annotation types along with contextual materials. Additionally, it offers a side-by-side comparison of multiple annotators' responses, allowing comparison and calibration of annotations with others to improve data reliability. The system is open-source, with a demo video available.
>
---
#### [new 044] UQLM: A Python Package for Uncertainty Quantification in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成虚假内容的问题。通过引入UQLM工具包，利用不确定性量化技术检测幻觉，提升模型输出的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.06196v1](http://arxiv.org/pdf/2507.06196v1)**

> **作者:** Dylan Bouchard; Mohit Singh Chauhan; David Skarbrevik; Ho-Kyeong Ra; Viren Bajaj; Zeya Ahmad
>
> **备注:** Submitted to Journal of Machine Learning Research (MLOSS); UQLM Repository: https://github.com/cvs-health/uqlm
>
> **摘要:** Hallucinations, defined as instances where Large Language Models (LLMs) generate false or misleading content, pose a significant challenge that impacts the safety and trust of downstream applications. We introduce UQLM, a Python package for LLM hallucination detection using state-of-the-art uncertainty quantification (UQ) techniques. This toolkit offers a suite of UQ-based scorers that compute response-level confidence scores ranging from 0 to 1. This library provides an off-the-shelf solution for UQ-based hallucination detection that can be easily integrated to enhance the reliability of LLM outputs.
>
---
#### [new 045] DS@GT at CheckThat! 2025: Evaluating Context and Tokenization Strategies for Numerical Fact Verification
- **分类: cs.CL**

- **简介: 该论文属于数值事实验证任务，旨在提升自动化核查系统对包含数量、比较和时间的陈述的准确性。研究评估了上下文长度、分词策略及其组合对分类性能的影响。**

- **链接: [http://arxiv.org/pdf/2507.06195v1](http://arxiv.org/pdf/2507.06195v1)**

> **作者:** Maximilian Heil; Aleksandar Pramov
>
> **摘要:** Numerical claims, statements involving quantities, comparisons, and temporal references, pose unique challenges for automated fact-checking systems. In this study, we evaluate modeling strategies for veracity prediction of such claims using the QuanTemp dataset and building our own evidence retrieval pipeline. We investigate three key factors: (1) the impact of more evidences with longer input context windows using ModernBERT, (2) the effect of right-to-left (R2L) tokenization, and (3) their combined influence on classification performance. Contrary to prior findings in arithmetic reasoning tasks, R2L tokenization does not boost natural language inference (NLI) of numerical tasks. A longer context window does also not enhance veracity performance either, highlighting evidence quality as the dominant bottleneck. Our best-performing system achieves competitive macro-average F1 score of 0.57 and places us among the Top-4 submissions in Task 3 of CheckThat! 2025. Our code is available at https://github.com/dsgt-arc/checkthat-2025-numerical.
>
---
#### [new 046] Flipping Knowledge Distillation: Leveraging Small Models' Expertise to Enhance LLMs in Text Matching
- **分类: cs.CL**

- **简介: 该论文属于文本匹配任务，解决小模型与大模型知识互补问题，通过翻转知识蒸馏，让LLM从SLM学习，提升性能。**

- **链接: [http://arxiv.org/pdf/2507.05617v1](http://arxiv.org/pdf/2507.05617v1)**

> **作者:** Mingzhe Li; Jing Xiang; Qishen Zhang; Kaiyang Wan; Xiuying Chen
>
> **备注:** Accepted by ACL 2025 main
>
> **摘要:** Knowledge distillation typically involves transferring knowledge from a Large Language Model (LLM) to a Smaller Language Model (SLM). However, in tasks such as text matching, fine-tuned smaller models often yield more effective domain-specific representations, as they focus on optimizing the similarity of input pairs. To leverage both the specialized strengths of small models and the rich semantic understanding of LLMs, we introduce a flipped knowledge distillation paradigm, where LLM learns from SLM. Specifically, we address the architectural gap between decoder-only LLMs and smaller encoder-based models by reinterpreting LLMs in an encoder-decoder manner using LoRA. The encoder generates compressed representations, while the decoder maps them to the output space. During training, the encoder produces representations and their similarities, which are then aligned with the similarity scores produced by the teacher, using our proposed Margin-aware Contrastive Learning (MCL) approach. The MCL ensures accurate similarity for both positive and negative pairs, and adaptively handles the internal differences within positive and negative samples. Our paradigm requires only a reasonably good-performing SLM, allowing the LLM to achieve improved performance. Experiments on financial and healthcare benchmarks, as well as real-world applications, confirm its effectiveness, and the model has been fully deployed in an online environment.
>
---
#### [new 047] DS@GT at CheckThat! 2025: Ensemble Methods for Detection of Scientific Discourse on Social Media
- **分类: cs.CL**

- **简介: 该论文属于科学网络话语检测任务，旨在识别社交媒体中的科学声明、研究引用和科学实体。团队探索了三种建模方法，并取得了较好的分类效果。**

- **链接: [http://arxiv.org/pdf/2507.06205v1](http://arxiv.org/pdf/2507.06205v1)**

> **作者:** Ayush Parikh; Hoang Thanh Thanh Truong; Jeanette Schofield; Maximilian Heil
>
> **摘要:** In this paper, we, as the DS@GT team for CLEF 2025 CheckThat! Task 4a Scientific Web Discourse Detection, present the methods we explored for this task. For this multiclass classification task, we determined if a tweet contained a scientific claim, a reference to a scientific study or publication, and/or mentions of scientific entities, such as a university or a scientist. We present 3 modeling approaches for this task: transformer finetuning, few-shot prompting of LLMs, and a combined ensemble model whose design was informed by earlier experiments. Our team placed 7th in the competition, achieving a macro-averaged F1 score of 0.8611, an improvement over the DeBERTaV3 baseline of 0.8375. Our code is available on Github at https://github.com/dsgt-arc/checkthat-2025-swd/tree/main/subtask-4a.
>
---
#### [new 048] SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索与生成任务，解决RAG中上下文长度限制和冗余问题。提出SARA框架，结合文本片段与语义向量，提升答案相关性与准确性。**

- **链接: [http://arxiv.org/pdf/2507.05633v1](http://arxiv.org/pdf/2507.05633v1)**

> **作者:** Yiqiao Jin; Kartik Sharma; Vineeth Rakesh; Yingtong Dou; Menghai Pan; Mahashweta Das; Srijan Kumar
>
> **备注:** 20 pages
>
> **摘要:** Retrieval-augmented Generation (RAG) extends large language models (LLMs) with external knowledge but faces key challenges: restricted effective context length and redundancy in retrieved documents. Pure compression-based approaches reduce input size but often discard fine-grained details essential for factual accuracy. We propose SARA, a unified RAG framework that balances local precision and global knowledge coverage under tight context budgets. SARA combines natural-language text snippets with semantic compression vectors to jointly enhance context efficiency and answer correctness. It represents contexts at two complementary levels: 1) fine-grained natural-language spans that preserve critical entities and numerical values, and 2) compact, interpretable vectors that summarize high-level semantics. An iterative evidence-selection module employs the compression vectors for dynamic reranking of contexts. Across 9 datasets and 5 open-source LLMs spanning 3 model families (Mistral, Llama, and Gemma), SARA consistently improves answer relevance (+17.71), answer correctness (+13.72), and semantic similarity (+15.53), demonstrating the importance of integrating textual and compressed representations for robust, context-efficient RAG.
>
---
#### [new 049] Remember Past, Anticipate Future: Learning Continual Multimodal Misinformation Detectors
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多模态虚假信息检测任务，解决在线数据流中模型持续失效的问题。提出DAEDCMD方法，通过记忆过去知识和预测未来环境提升检测效果。**

- **链接: [http://arxiv.org/pdf/2507.05939v1](http://arxiv.org/pdf/2507.05939v1)**

> **作者:** Bing Wang; Ximing Li; Mengzhe Ye; Changchun Li; Bo Fu; Jianfeng Qu; Lin Yuanbo Wu
>
> **备注:** Accepted by ACM MM 2025. 10 pages, 6 figures. Code: https://github.com/wangbing1416/DAEDCMD
>
> **摘要:** Nowadays, misinformation articles, especially multimodal ones, are widely spread on social media platforms and cause serious negative effects. To control their propagation, Multimodal Misinformation Detection (MMD) becomes an active topic in the community to automatically identify misinformation. Previous MMD methods focus on supervising detectors by collecting offline data. However, in real-world scenarios, new events always continually emerge, making MMD models trained on offline data consistently outdated and ineffective. To address this issue, training MMD models under online data streams is an alternative, inducing an emerging task named continual MMD. Unfortunately, it is hindered by two major challenges. First, training on new data consistently decreases the detection performance on past data, named past knowledge forgetting. Second, the social environment constantly evolves over time, affecting the generalization on future data. To alleviate these challenges, we propose to remember past knowledge by isolating interference between event-specific parameters with a Dirichlet process-based mixture-of-expert structure, and anticipate future environmental distributions by learning a continuous-time dynamics model. Accordingly, we induce a new continual MMD method DAEDCMD. Extensive experiments demonstrate that DAEDCMD can consistently and significantly outperform the compared methods, including six MMD baselines and three continual learning methods.
>
---
#### [new 050] The Generalization Ridge: Information Flow in Natural Language Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，研究Transformer模型中信息流动机制。针对中间层与最终层在泛化能力上的差异，提出InfoRidge框架，揭示信息流的非单调变化规律。**

- **链接: [http://arxiv.org/pdf/2507.05387v1](http://arxiv.org/pdf/2507.05387v1)**

> **作者:** Ruidi Chang; Chunyuan Deng; Hanjie Chen
>
> **摘要:** Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG) tasks, yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. To address this gap, we propose InfoRidge, an information-theoretic framework, to characterize how predictive information-the mutual information between hidden representations and target outputs-varies across depth. Estimating this quantity enables us to trace the flow of task-relevant information throughout the model during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in upper-middle layers-forming a generalization ridge-before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we introduce residual scaling coefficients-trainable scalar parameters applied to each residual block-which serve as functional probes for assessing the relative importance of individual transformer layers. These coefficients reveal that, under distribution shift, models downweight final layers and increasingly rely on ridge layers, highlighting their role in generalization. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization.
>
---
#### [new 051] GPTKB v1.5: A Massive Knowledge Base for Exploring Factual LLM Knowledge
- **分类: cs.CL**

- **简介: 该论文属于知识库构建任务，旨在解决语言模型事实知识难以访问的问题。通过GPT-4.1构建了一个大规模知识库，支持知识探索与查询。**

- **链接: [http://arxiv.org/pdf/2507.05740v1](http://arxiv.org/pdf/2507.05740v1)**

> **作者:** Yujia Hu; Tuan-Phong Nguyen; Shrestha Ghosh; Moritz Müller; Simon Razniewski
>
> **备注:** 7 pages, 6 figures, 1 table
>
> **摘要:** Language models are powerful tools, yet their factual knowledge is still poorly understood, and inaccessible to ad-hoc browsing and scalable statistical analysis. This demonstration introduces GPTKB v1.5, a densely interlinked 100-million-triple knowledge base (KB) built for $14,000 from GPT-4.1, using the GPTKB methodology for massive-recursive LLM knowledge materialization (Hu et al., ACL 2025). The demonstration experience focuses on three use cases: (1) link-traversal-based LLM knowledge exploration, (2) SPARQL-based structured LLM knowledge querying, (3) comparative exploration of the strengths and weaknesses of LLM knowledge. Massive-recursive LLM knowledge materialization is a groundbreaking opportunity both for the research area of systematic analysis of LLM knowledge, as well as for automated KB construction. The GPTKB demonstrator is accessible at https://gptkb.org.
>
---
#### [new 052] How to Evaluate Automatic Speech Recognition: Comparing Different Performance and Bias Measures
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决如何评估ASR系统性能与偏差的问题。通过比较不同度量方法，提出更全面的评估框架。**

- **链接: [http://arxiv.org/pdf/2507.05885v1](http://arxiv.org/pdf/2507.05885v1)**

> **作者:** Tanvina Patel; Wiebke Hutiri; Aaron Yi Ding; Odette Scharenborg
>
> **摘要:** There is increasingly more evidence that automatic speech recognition (ASR) systems are biased against different speakers and speaker groups, e.g., due to gender, age, or accent. Research on bias in ASR has so far primarily focused on detecting and quantifying bias, and developing mitigation approaches. Despite this progress, the open question is how to measure the performance and bias of a system. In this study, we compare different performance and bias measures, from literature and proposed, to evaluate state-of-the-art end-to-end ASR systems for Dutch. Our experiments use several bias mitigation strategies to address bias against different speaker groups. The findings reveal that averaged error rates, a standard in ASR research, alone is not sufficient and should be supplemented by other measures. The paper ends with recommendations for reporting ASR performance and bias to better represent a system's performance for diverse speaker groups, and overall system bias.
>
---
#### [new 053] DS@GT at CheckThat! 2025: Detecting Subjectivity via Transfer-Learning and Corrective Data Augmentation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于主观性检测任务，旨在提升英文新闻文本中主观与客观句子的分类效果。通过迁移学习和修正数据增强方法，提升了模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.06189v1](http://arxiv.org/pdf/2507.06189v1)**

> **作者:** Maximilian Heil; Dionne Bang
>
> **摘要:** This paper presents our submission to Task 1, Subjectivity Detection, of the CheckThat! Lab at CLEF 2025. We investigate the effectiveness of transfer-learning and stylistic data augmentation to improve classification of subjective and objective sentences in English news text. Our approach contrasts fine-tuning of pre-trained encoders and transfer-learning of fine-tuned transformer on related tasks. We also introduce a controlled augmentation pipeline using GPT-4o to generate paraphrases in predefined subjectivity styles. To ensure label and style consistency, we employ the same model to correct and refine the generated samples. Results show that transfer-learning of specified encoders outperforms fine-tuning general-purpose ones, and that carefully curated augmentation significantly enhances model robustness, especially in detecting subjective content. Our official submission placed us $16^{th}$ of 24 participants. Overall, our findings underscore the value of combining encoder specialization with label-consistent augmentation for improved subjectivity detection. Our code is available at https://github.com/dsgt-arc/checkthat-2025-subject.
>
---
#### [new 054] Learn Globally, Speak Locally: Bridging the Gaps in Multilingual Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多语言推理任务，旨在解决低资源语言模型推理偏差问题。通过构建GeoFact-X基准和BRIDGE方法提升模型在目标语言中的推理能力。**

- **链接: [http://arxiv.org/pdf/2507.05418v1](http://arxiv.org/pdf/2507.05418v1)**

> **作者:** Jaedong Hwang; Kumar Tanmay; Seok-Jin Lee; Ayush Agrawal; Hamid Palangi; Kumar Ayush; Ila Fiete; Paul Pu Liang
>
> **摘要:** Large Language Models (LLMs) have achieved strong performance in domains like mathematics, factual QA, and code generation, yet their multilingual reasoning capabilities in these tasks remain underdeveloped. Especially for low-resource languages such as Swahili or Thai, LLMs can often misinterpret prompts or default to reasoning in English. This implicit bias toward high-resource languages undermines factual accuracy, interpretability, and trust. Current multilingual benchmarks focus only on final answers, overlooking whether models actually reason in the target language. To address this gap, we introduce GeoFact-X, a geography-based multilingual factual reasoning benchmark with annotated reasoning traces in five languages: English, Hindi, Japanese, Swahili, and Thai. We further propose BRIDGE, a novel training method that guides supervised fine-tuning and test-time reinforcement learning with a language-consistency reward to align reasoning with the input language. Finally, we develop an automatic evaluation protocol using LLM-as-a-judge to assess answer correctness and the quality and language consistency of reasoning traces, enabling nuanced and scalable analysis beyond surface-level metrics. Our results show that BRIDGE significantly enhances multilingual reasoning fidelity, demonstrating that reasoning-aware multilingual reinforcement learning is crucial for robust cross-lingual generalization. https://jd730.github.io/projects/GeoFact-X_BRIDGE
>
---
#### [new 055] On the Semantics of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，探讨LLMs在词句层面的语义能力。通过分析模型结构和经典语义理论，研究其是否具备真正理解语言的能力。**

- **链接: [http://arxiv.org/pdf/2507.05448v1](http://arxiv.org/pdf/2507.05448v1)**

> **作者:** Martin Schuele
>
> **摘要:** Large Language Models (LLMs) such as ChatGPT demonstrated the potential to replicate human language abilities through technology, ranging from text generation to engaging in conversations. However, it remains controversial to what extent these systems truly understand language. We examine this issue by narrowing the question down to the semantics of LLMs at the word and sentence level. By examining the inner workings of LLMs and their generated representation of language and by drawing on classical semantic theories by Frege and Russell, we get a more nuanced picture of the potential semantic capabilities of LLMs.
>
---
#### [new 056] Self-Review Framework for Enhancing Instruction Following Capability of LLM
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型遵循指令的能力。针对模型生成内容不符合复杂指令的问题，提出Re5框架，通过自评估与精修提高准确性与质量。**

- **链接: [http://arxiv.org/pdf/2507.05598v1](http://arxiv.org/pdf/2507.05598v1)**

> **作者:** Sihyun Park
>
> **摘要:** Various techniques have been proposed to improve large language models (LLMs) adherence to formatting and instruction constraints. One of the most effective approaches involves utilizing high-quality data generated by powerful models. However, such models often fail to fully comply with complex instructions in a single generation. To address this limitation, iterative revision methods have been introduced. Nevertheless, as the number of data points and revision iterations increases, the associated monetary costs grow significantly. As a resource-efficient alternative, methods have been proposed that leverage high-performance evaluation tools to compensate for the limited self-evaluation capabilities of open-source LLMs. However, these approaches often lead to a degradation in output quality due to excessive revision. To overcome these challenges, we propose Re5, a self-evaluation and revision framework designed to enhance instruction-following performance while preserving the quality of the generated content. Re5 extracts task and constraint components from user instructions, performs structural evaluations to prevent error accumulation, and applies fine-grained constraint-specific content evaluations followed by selective revisions. This process ensures precise and quality-preserving improvements. The final high-quality outputs are used for alignment tuning, enabling long-term alignment improvements through a data-centric iterative refinement loop. Experimental results demonstrate that Re5 achieves instruction-following performance comparable to models trained on data generated by GPT-4o-mini, a high-performance model, even with a small amount of data while maintaining response quality with a 64.24%-win rate over the non-revised initial responses. These results validate Re5 as an efficient and effective solution for enhancing instruction adherence with minimal external supervision.
>
---
#### [new 057] Controlling What You Share: Assessing Language Model Adherence to Privacy Preferences
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于隐私保护任务，旨在解决用户数据泄露问题。通过隐私配置文件控制模型输出，构建框架平衡隐私与性能。**

- **链接: [http://arxiv.org/pdf/2507.05391v1](http://arxiv.org/pdf/2507.05391v1)**

> **作者:** Guillem Ramírez; Alexandra Birch; Ivan Titov
>
> **摘要:** Large language models (LLMs) are primarily accessed via commercial APIs, but this often requires users to expose their data to service providers. In this paper, we explore how users can stay in control of their data by using privacy profiles: simple natural language instructions that say what should and should not be revealed. We build a framework where a local model uses these instructions to rewrite queries, only hiding details deemed sensitive by the user, before sending them to an external model, thus balancing privacy with performance. To support this research, we introduce PEEP, a multilingual dataset of real user queries annotated to mark private content and paired with synthetic privacy profiles. Our experiments with lightweight LLMs show they can follow these instructions to some extent, but also face consistent challenges, highlighting the need for models that better understand and comply with user-defined privacy preferences.
>
---
#### [new 058] Semantic Certainty Assessment in Vector Retrieval Systems: A Novel Framework for Embedding Quality Evaluation
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于向量检索任务，解决嵌入质量评估问题。通过结合量化鲁棒性和邻域密度，提出轻量框架提升检索性能。**

- **链接: [http://arxiv.org/pdf/2507.05933v1](http://arxiv.org/pdf/2507.05933v1)**

> **作者:** Y. Du
>
> **备注:** 7 pages
>
> **摘要:** Vector retrieval systems exhibit significant performance variance across queries due to heterogeneous embedding quality. We propose a lightweight framework for predicting retrieval performance at the query level by combining quantization robustness and neighborhood density metrics. Our approach is motivated by the observation that high-quality embeddings occupy geometrically stable regions in the embedding space and exhibit consistent neighborhood structures. We evaluate our method on 4 standard retrieval datasets, showing consistent improvements of 9.4$\pm$1.2\% in Recall@10 over competitive baselines. The framework requires minimal computational overhead (less than 5\% of retrieval time) and enables adaptive retrieval strategies. Our analysis reveals systematic patterns in embedding quality across different query types, providing insights for targeted training data augmentation.
>
---
#### [new 059] AutoTriton: Automatic Triton Programming with Reinforcement Learning in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于深度学习优化任务，旨在解决Triton编程中手动调参困难的问题。通过强化学习自动生成高性能内核。**

- **链接: [http://arxiv.org/pdf/2507.05687v1](http://arxiv.org/pdf/2507.05687v1)**

> **作者:** Shangzhan Li; Zefan Wang; Ye He; Yuxuan Li; Qi Shi; Jianling Li; Yonggang Hu; Wanxiang Che; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Kernel development in deep learning requires optimizing computational units across hardware while balancing memory management, parallelism, and hardware-specific optimizations through extensive empirical tuning. Although domain-specific languages like Triton simplify GPU programming by abstracting low-level details, developers must still manually tune critical parameters such as tile sizes and memory access patterns through iterative experimentation, creating substantial barriers to optimal performance and wider adoption. In this work, we introduce AutoTriton, the first model dedicated to Triton programming powered by reinforcement learning (RL). AutoTriton performs supervised fine-tuning (SFT) to be equipped with essential Triton programming expertise using a high-quality data gathering pipeline, and conducts RL with Group Relative Policy Optimization (GRPO) algorithm, combining a rule-based reward and an execution-based reward to further improve Triton programming ability, sequentially. Experiments across five evaluation channels of TritonBench and KernelBench illustrate that our 8B model AutoTriton achieves performance comparable to mainstream large models, including Claude-4-Sonnet and DeepSeek-R1-0528. Further experimental analysis demonstrates the crucial role of each module within AutoTriton, including the SFT stage, the RL stage, and the reward design strategy. These findings underscore the promise of RL for automatically generating high-performance kernels, and since high-performance kernels are core components of AI systems, this breakthrough establishes an important foundation for building more efficient AI systems. The model and code will be available at https://github.com/AI9Stars/AutoTriton.
>
---
#### [new 060] ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy
- **分类: cs.SE; cs.AI; cs.CL; cs.NE**

- **简介: 该论文属于代码辅助与问答任务，旨在提升LLM在ReservoirPy中的代码开发和复杂问题解答能力，通过RAG和知识图谱增强准确性。**

- **链接: [http://arxiv.org/pdf/2507.05279v1](http://arxiv.org/pdf/2507.05279v1)**

> **作者:** Virgile Boraud; Yannis Bendi-Ouis; Paul Bernard; Xavier Hinaut
>
> **摘要:** We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B.
>
---
#### [new 061] AI-Reporter: A Path to a New Genre of Scientific Communication
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于科学传播任务，旨在解决学术演讲转化为正式出版物的效率问题。通过案例展示AI系统快速转换演讲内容为可发表章节。**

- **链接: [http://arxiv.org/pdf/2507.05903v1](http://arxiv.org/pdf/2507.05903v1)**

> **作者:** Gerd Graßhoff
>
> **摘要:** The AI-Reporter represents a paradigmatic shift in scientific publication practice. This document demonstrates through a concrete case study how our system transforms academic presentations into publication-ready chapters -- in less than three minutes. Using Arno Simons' lecture on Large Language Models from the ``Large Language Models for the History, Philosophy, and Sociology of Science'' workshop (NEPI) as an example, we show how technological innovation bridges the gap between ephemeral presentation and permanent scientific documentation.
>
---
#### [new 062] Evaluation of Habitat Robotics using Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人任务研究，旨在评估大语言模型在具身机器人环境中的表现，解决模型推理能力对协作任务的影响问题。工作包括在Meta PARTNER基准上测试多个模型。**

- **链接: [http://arxiv.org/pdf/2507.06157v1](http://arxiv.org/pdf/2507.06157v1)**

> **作者:** William Li; Lei Hamilton; Kaise Al-natour; Sanjeev Mohindra
>
> **备注:** 6 pages, IEEE HPEC submission
>
> **摘要:** This paper focuses on evaluating the effectiveness of Large Language Models at solving embodied robotic tasks using the Meta PARTNER benchmark. Meta PARTNR provides simplified environments and robotic interactions within randomized indoor kitchen scenes. Each randomized kitchen scene is given a task where two robotic agents cooperatively work together to solve the task. We evaluated multiple frontier models on Meta PARTNER environments. Our results indicate that reasoning models like OpenAI o3-mini outperform non-reasoning models like OpenAI GPT-4o and Llama 3 when operating in PARTNR's robotic embodied environments. o3-mini displayed outperform across centralized, decentralized, full observability, and partial observability configurations. This provides a promising avenue of research for embodied robotic development.
>
---
#### [new 063] Development and Evaluation of HopeBot: an LLM-based chatbot for structured and interactive PHQ-9 depression screening
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于心理健康筛查任务，旨在解决传统抑郁量表缺乏互动性的问题。研究开发了基于LLM的HopeBot聊天机器人，通过实时澄清和生成增强交互式PHQ-9评估。**

- **链接: [http://arxiv.org/pdf/2507.05984v1](http://arxiv.org/pdf/2507.05984v1)**

> **作者:** Zhijun Guo; Alvina Lai; Julia Ive; Alexandru Petcu; Yutong Wang; Luyuan Qi; Johan H Thygesen; Kezhi Li
>
> **摘要:** Static tools like the Patient Health Questionnaire-9 (PHQ-9) effectively screen depression but lack interactivity and adaptability. We developed HopeBot, a chatbot powered by a large language model (LLM) that administers the PHQ-9 using retrieval-augmented generation and real-time clarification. In a within-subject study, 132 adults in the United Kingdom and China completed both self-administered and chatbot versions. Scores demonstrated strong agreement (ICC = 0.91; 45% identical). Among 75 participants providing comparative feedback, 71% reported greater trust in the chatbot, highlighting clearer structure, interpretive guidance, and a supportive tone. Mean ratings (0-10) were 8.4 for comfort, 7.7 for voice clarity, 7.6 for handling sensitive topics, and 7.4 for recommendation helpfulness; the latter varied significantly by employment status and prior mental-health service use (p < 0.05). Overall, 87.1% expressed willingness to reuse or recommend HopeBot. These findings demonstrate voice-based LLM chatbots can feasibly serve as scalable, low-burden adjuncts for routine depression screening.
>
---
#### [new 064] Narrowing the Gap: Supervised Fine-Tuning of Open-Source LLMs as a Viable Alternative to Proprietary Models for Pedagogical Tools
- **分类: cs.CY; cs.AI; cs.CL; cs.SE**

- **简介: 该论文属于教育AI任务，旨在解决大型模型在教学中应用的成本与效率问题。通过监督微调开源小模型，提升其教学效果，实现与大模型相当的性能。**

- **链接: [http://arxiv.org/pdf/2507.05305v1](http://arxiv.org/pdf/2507.05305v1)**

> **作者:** Lorenzo Lee Solano; Charles Koutcheme; Juho Leinonen; Alexandra Vassar; Jake Renzella
>
> **备注:** 7 pages, 3 tables, 1 figure
>
> **摘要:** Frontier Large language models (LLMs) like ChatGPT and Gemini can decipher cryptic compiler errors for novice programmers, but their computational scale, cost, and tendency to over-assist make them problematic for widespread pedagogical adoption. This work demonstrates that smaller, specialised language models, enhanced via Supervised Fine-Tuning (SFT), present a more viable alternative for educational tools. We utilise a new dataset of 40,000 C compiler error explanations, derived from real introductory programming (CS1/2) student-generated programming errors, which we used to fine-tune three open-source models: Qwen3-4B, Llama-3.1-8B, and Qwen3-32B. We performed a dual evaluation, combining expert human reviews with a large-scale automated analysis of 8,000 responses using a validated LLM-as-judge ensemble. Our results show that SFT significantly boosts the pedagogical quality of smaller models, achieving performance comparable to much larger models. We analyse the trade-offs between model size and quality, confirming that fine-tuning compact, efficient models on high-quality, domain-specific data is a potent strategy for creating specialised models to drive educational tools. We provide a replicable methodology to foster broader access to generative AI capabilities in educational contexts.
>
---
#### [new 065] SQLBarber: A System Leveraging Large Language Models to Generate Customized and Realistic SQL Workloads
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数据库基准测试任务，旨在解决生成定制化、真实SQL工作负载的问题。通过LLM和优化方法，高效生成符合成本分布的SQL查询。**

- **链接: [http://arxiv.org/pdf/2507.06192v1](http://arxiv.org/pdf/2507.06192v1)**

> **作者:** Jiale Lao; Immanuel Trummer
>
> **摘要:** Database research and development often require a large number of SQL queries for benchmarking purposes. However, acquiring real-world SQL queries is challenging due to privacy concerns, and existing SQL generation methods are limited in customization and in satisfying realistic constraints. To address this issue, we present SQLBarber, a system based on Large Language Models (LLMs) to generate customized and realistic SQL workloads. SQLBarber (i) eliminates the need for users to manually craft SQL templates in advance, while providing the flexibility to accept natural language specifications to constrain SQL templates, (ii) scales efficiently to generate large volumes of queries matching any user-defined cost distribution (e.g., cardinality and execution plan cost), and (iii) uses execution statistics from Amazon Redshift and Snowflake to derive SQL template specifications and query cost distributions that reflect real-world query characteristics. SQLBarber introduces (i) a declarative interface for users to effortlessly generate customized SQL templates, (ii) an LLM-powered pipeline augmented with a self-correction module that profiles, refines, and prunes SQL templates based on query costs, and (iii) a Bayesian Optimizer to efficiently explore different predicate values and identify a set of queries that satisfy the target cost distribution. We construct and open-source ten benchmarks of varying difficulty levels and target query cost distributions based on real-world statistics from Snowflake and Amazon Redshift. Extensive experiments on these benchmarks show that SQLBarber is the only system that can generate customized SQL templates. It reduces query generation time by one to three orders of magnitude, and significantly improves alignment with the target cost distribution, compared with existing methods.
>
---
#### [new 066] Reinforcement Fine-Tuning Naturally Mitigates Forgetting in Continual Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续后训练任务，解决模型在连续学习中遗忘问题。通过对比监督微调与强化微调，发现强化微调能有效防止遗忘并提升性能。**

- **链接: [http://arxiv.org/pdf/2507.05386v1](http://arxiv.org/pdf/2507.05386v1)**

> **作者:** Song Lai; Haohan Zhao; Rong Feng; Changyi Ma; Wenzhuo Liu; Hongbo Zhao; Xi Lin; Dong Yi; Min Xie; Qingfu Zhang; Hongbin Liu; Gaofeng Meng; Fei Zhu
>
> **摘要:** Continual post-training (CPT) is a popular and effective technique for adapting foundation models like multimodal large language models to specific and ever-evolving downstream tasks. While existing research has primarily concentrated on methods like data replay, model expansion, or parameter regularization, the fundamental role of the learning paradigm within CPT remains largely unexplored. This paper presents a comparative analysis of two core post-training paradigms: supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT), investigating their respective impacts on knowledge retention during CPT. Our experiments are conducted on a benchmark comprising seven diverse multimodal tasks, utilizing Qwen2.5-VL-7B-Instruct as the base model for continual post-training. The investigation yields two significant findings: (1) When continuously learning on downstream tasks, SFT leads to catastrophic forgetting of previously learned tasks. In contrast, RFT inherently preserves prior knowledge and achieve performance comparable to multi-task training. (2) RFT successfully protects and even enhances the model's general knowledge on standard benchmarks (e.g., MMMU and MMLU-Pro). Conversely, SFT degrades general model capabilities severely. Further analysis shows that explicit mechanisms, such as KL penalty and chain-of-thought reasoning, are not the primary factors. Instead, we find that the implicit regularization inherent to RFT is a key factor in mitigating forgetting. Finally, we propose a rollout-based instance filtering algorithm to improve the stability and efficiency of RFT. Our comprehensive study demonstrates the superiority of RFT as a robust paradigm for continual post-training.
>
---
#### [new 067] ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决传统ASR模型在上下文理解上的不足。提出ContextASR-Bench基准，评估模型在多领域上下文场景下的表现。**

- **链接: [http://arxiv.org/pdf/2507.05727v1](http://arxiv.org/pdf/2507.05727v1)**

> **作者:** He Wang; Linhan Ma; Dake Guo; Xiong Wang; Lei Xie; Jin Xu; Junyang Lin
>
> **备注:** 18 pages, 4 figures
>
> **摘要:** Automatic Speech Recognition (ASR) has been extensively investigated, yet prior evaluative efforts have largely been restricted to contextless paradigms. This constraint stems from the limited proficiency of conventional ASR models in context modeling and their deficiency in memory and reasoning based on world knowledge. Recent breakthroughs in the development of Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of general artificial intelligence capabilities. Consequently, there exists a compelling need for a benchmark that can evaluate both the generality and intelligence of ASR systems. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess contextual speech recognition. This benchmark encompasses up to 40,000 data entries across over 10 domains, enabling a thorough evaluation of model performance in scenarios that omit or incorporate coarse-grained or fine-grained contextual information. Moreover, diverging from conventional ASR evaluations, our benchmark includes an analysis of model efficacy in recognizing named entities mentioned within the auditory input. Our extensive evaluation highlights that LALMs, with strong world knowledge and context learning capabilities, outperform conventional ASR models by a large margin. The dataset and evaluation code have been released at https://github.com/MrSupW/ContextASR-Bench.
>
---
#### [new 068] CoreCodeBench: A Configurable Multi-Scenario Repository-Level Benchmark
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成与评估任务，旨在解决现有基准单一场景、可控性差的问题。提出CoreCodeBench，通过自动化管道生成多场景测试用例，提升LLM工程应用评估的全面性。**

- **链接: [http://arxiv.org/pdf/2507.05281v1](http://arxiv.org/pdf/2507.05281v1)**

> **作者:** Lingyue Fu; Hao Guan; Bolun Zhang; Haowei Yuan; Yaoming Zhu; Jun Xu; Zongyu Wang; Lin Qiu; Xunliang Cai; Xuezhi Cao; Weiwen Liu; Weinan Zhang; Yong Yu
>
> **摘要:** As Large Language Models (LLMs) demonstrate increasingly sophisticated code processing capabilities, evaluating their performance on engineering-level code remains challenging. Existing repository-level benchmarks primarily focus on single scenarios, such as code generation or bug fixing, without adequately capturing the diversity and complexity of real-world software or project engineering workflows. Furthermore, these benchmarks suffer from limited controllability in question positioning and reliability issues in their generated test cases. To address these limitations, we present CorePipe, a fully automated pipeline that converts repositories into comprehensive test cases, and introduce CoreCodeBench, a configurable multi-scenario repository-level benchmark. To simulate real engineering scenarios, CorePipe generates three types of atomic questions (Development, BugFix, and Test-Driven Development) specifically targeting core code segments. These atomic questions are further combined into three types of composite questions, with difficulty levels flexibly adjusted through hyperparameter tuning. CoreCodeBench provides a comprehensive and extensive repository-level benchmark to investigate the applicability of LLMs in real-world engineering projects. Experiments with 16 LLMs across diverse scenarios reveal varying capabilities and offer multi-dimensional insights into LLM performance in engineering contexts. The code for CorePipe is available at https://github.com/AGI-Eval-Official/CoreCodeBench, and the data for CoreCodeBench can be accessed at https://huggingface.co/collections/tubehhh/corecodebench-68256d2faabf4b1610a08caa.
>
---
#### [new 069] News Source Citing Patterns in AI Search Systems
- **分类: cs.IR; cs.CL; cs.CY**

- **简介: 该论文属于AI搜索系统研究，分析其新闻引用模式。旨在揭示AI搜索系统的引用行为及偏差，通过数据对比不同模型的引用特点与用户偏好。**

- **链接: [http://arxiv.org/pdf/2507.05301v1](http://arxiv.org/pdf/2507.05301v1)**

> **作者:** Kai-Cheng Yang
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** AI-powered search systems are emerging as new information gatekeepers, fundamentally transforming how users access news and information. Despite their growing influence, the citation patterns of these systems remain poorly understood. We address this gap by analyzing data from the AI Search Arena, a head-to-head evaluation platform for AI search systems. The dataset comprises over 24,000 conversations and 65,000 responses from models across three major providers: OpenAI, Perplexity, and Google. Among the over 366,000 citations embedded in these responses, 9% reference news sources. We find that while models from different providers cite distinct news sources, they exhibit shared patterns in citation behavior. News citations concentrate heavily among a small number of outlets and display a pronounced liberal bias, though low-credibility sources are rarely cited. User preference analysis reveals that neither the political leaning nor the quality of cited news sources significantly influences user satisfaction. These findings reveal significant challenges in current AI search systems and have important implications for their design and governance.
>
---
#### [new 070] MobileGUI-RL: Advancing Mobile GUI Agent through Reinforcement Learning in Online Environment
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于移动GUI自动化任务，解决现有方法在离线环境中训练导致泛化能力差的问题。提出MobileGUI-RL框架，在在线环境中训练，提升GUI代理的适应性与效率。**

- **链接: [http://arxiv.org/pdf/2507.05720v1](http://arxiv.org/pdf/2507.05720v1)**

> **作者:** Yucheng Shi; Wenhao Yu; Zaitang Li; Yonglin Wang; Hongming Zhang; Ninghao Liu; Haitao Mi; Dong Yu
>
> **备注:** 17 pages, 4 figures
>
> **摘要:** Recently, there has been a surge of vision-based GUI agents designed to automate everyday mobile and web tasks. These agents interpret raw GUI screenshots and autonomously decide where to click, scroll, or type, which bypasses handcrafted rules and app-specific APIs. However, most existing methods trained GUI agent in the offline environment using pre-collected trajectories. This approach limits scalability, causes overfitting to specific UI templates, and leads to brittle policies when faced with unseen environment. We present MobileGUI-RL, a scalable framework that trains GUI agent in online environment. MobileGUI-RL contains two key components. It (i) synthesizes a curriculum of learnable tasks through self-exploration and filtering, and (ii) adapts GRPO to GUI navigation with trajectory-aware advantages and composite rewards that balance task success and execution efficiency. Experiments on three online mobile-agent benchmarks show consistent gains, validating the effectiveness of our approach.
>
---
#### [new 071] Chat2SPaT: A Large Language Model Based Tool for Automating Traffic Signal Control Plan Management
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于交通信号控制任务，旨在解决手动制定和更新信号计划的繁琐问题。通过Chat2SPaT工具，将用户描述转化为精确的信号相位和时间配置。**

- **链接: [http://arxiv.org/pdf/2507.05283v1](http://arxiv.org/pdf/2507.05283v1)**

> **作者:** Yue Wang; Miao Zhou; Guijing Huang; Rui Zhuo; Chao Yi; Zhenliang Ma
>
> **摘要:** Pre-timed traffic signal control, commonly used for operating signalized intersections and coordinated arterials, requires tedious manual work for signaling plan creating and updating. When the time-of-day or day-of-week plans are utilized, one intersection is often associated with multiple plans, leading to further repetitive manual plan parameter inputting. To enable a user-friendly traffic signal control plan management process, this study proposes Chat2SPaT, a method to convert users' semi-structured and ambiguous descriptions on the signal control plan to exact signal phase and timing (SPaT) results, which could further be transformed into structured stage-based or ring-based plans to interact with intelligent transportation system (ITS) software and traffic signal controllers. With curated prompts, Chat2SPaT first leverages large language models' (LLMs) capability of understanding users' plan descriptions and reformulate the plan as a combination of phase sequence and phase attribute results in the json format. Based on LLM outputs, python scripts are designed to locate phases in a cycle, address nuances of traffic signal control, and finally assemble the complete traffic signal control plan. Within a chat, the pipeline can be utilized iteratively to conduct further plan editing. Experiments show that Chat2SPaT can generate plans with an accuracy of over 94% for both English and Chinese cases, using a test dataset with over 300 plan descriptions. As the first benchmark for evaluating LLMs' capability of understanding traffic signal control plan descriptions, Chat2SPaT provides an easy-to-use plan management pipeline for traffic practitioners and researchers, serving as a potential new building block for a more accurate and versatile application of LLMs in the field of ITS. The source codes, prompts and test dataset are openly accessible at https://github.com/yuewangits/Chat2SPaT.
>
---
#### [new 072] Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers with LLMs for Biomedical QA
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于生物医学问答任务，旨在提高从海量文献中检索和生成准确答案的能力。工作包括构建RAG系统，结合交叉编码器、LLM重排序和少样本提示生成答案。**

- **链接: [http://arxiv.org/pdf/2507.05577v1](http://arxiv.org/pdf/2507.05577v1)**

> **作者:** Shashank Verma; Fengyi Jiang; Xiangning Xue
>
> **备注:** Paper submitted to CLEF 2025 CEUR-WS
>
> **摘要:** Biomedical semantic question answering rooted in information retrieval can play a crucial role in keeping up to date with vast, rapidly evolving and ever-growing biomedical literature. A robust system can help researchers, healthcare professionals and even layman users access relevant knowledge grounded in evidence. The BioASQ 2025 Task13b Challenge serves as an important benchmark, offering a competitive platform for advancement of this space. This paper presents the methodologies and results from our participation in this challenge where we built a Retrieval-Augmented Generation (RAG) system that can answer biomedical questions by retrieving relevant PubMed documents and snippets to generate answers. For the retrieval task, we generated dense embeddings from biomedical articles for initial retrieval, and applied an ensemble of finetuned cross-encoders and large language models (LLMs) for re-ranking to identify top relevant documents. Our solution achieved an MAP@10 of 0.1581, placing 10th on the leaderboard for the retrieval task. For answer generation, we employed few-shot prompting of instruction-tuned LLMs. Our system achieved macro-F1 score of 0.95 for yes/no questions (rank 12), Mean Reciprocal Rank (MRR) of 0.64 for factoid questions (rank 1), mean-F1 score of 0.63 for list questions (rank 5), and ROUGE-SU4 F1 score of 0.29 for ideal answers (rank 11).
>
---
#### [new 073] A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息对抗任务，旨在解决LLM生成的虚假信息问题。提出三支柱防御框架，提升模型可信度、推理可靠性和输入鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.05288v1](http://arxiv.org/pdf/2507.05288v1)**

> **作者:** Shuliang Liu; Hongyi Liu; Aiwei Liu; Bingchen Duan; Qi Zheng; Yibo Yan; He Geng; Peijie Jiang; Jia Liu; Xuming Hu
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.
>
---
#### [new 074] The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于自然语言处理领域，研究大模型的记忆化现象，探讨其机制、检测方法及缓解策略，旨在解决隐私与模型行为问题。**

- **链接: [http://arxiv.org/pdf/2507.05578v1](http://arxiv.org/pdf/2507.05578v1)**

> **作者:** Alexander Xiong; Xuandong Zhao; Aneesh Pappu; Dawn Song
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the minimization of harmful memorization with utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
>
---
#### [new 075] TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于对话AI安全任务，解决在不可信数据微调中毒性问题。提出TuneShield框架，通过毒性分类和合成数据缓解毒性，提升安全性与对话质量。**

- **链接: [http://arxiv.org/pdf/2507.05660v1](http://arxiv.org/pdf/2507.05660v1)**

> **作者:** Aravind Cheruvu; Shravya Kanchi; Sifat Muhammad Abdullah; Nicholas Kong; Daphne Yao; Murtuza Jadliwala; Bimal Viswanath
>
> **备注:** Pre-print
>
> **摘要:** Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL).
>
---
#### [new 076] Nyay-Darpan: Enhancing Decision Making Through Summarization and Case Retrieval for Consumer Law in India
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于法律AI任务，旨在解决印度消费者法中案例总结与检索问题。提出Nyay-Darpan框架，实现高效案件处理与决策支持。**

- **链接: [http://arxiv.org/pdf/2507.06090v1](http://arxiv.org/pdf/2507.06090v1)**

> **作者:** Swapnil Bhattacharyya; Shrey Ganatra; Harshvivek Kashid; Spandan Anaokar; Shruti Nair; Reshma Sekhar; Siddharth Manohar; Rahul Hemrajani; Pushpak Bhattacharyya
>
> **摘要:** AI-based judicial assistance and case prediction have been extensively studied in criminal and civil domains, but remain largely unexplored in consumer law, especially in India. In this paper, we present Nyay-Darpan, a novel two-in-one framework that (i) summarizes consumer case files and (ii) retrieves similar case judgements to aid decision-making in consumer dispute resolution. Our methodology not only addresses the gap in consumer law AI tools but also introduces an innovative approach to evaluate the quality of the summary. The term 'Nyay-Darpan' translates into 'Mirror of Justice', symbolizing the ability of our tool to reflect the core of consumer disputes through precise summarization and intelligent case retrieval. Our system achieves over 75 percent accuracy in similar case prediction and approximately 70 percent accuracy across material summary evaluation metrics, demonstrating its practical effectiveness. We will publicly release the Nyay-Darpan framework and dataset to promote reproducibility and facilitate further research in this underexplored yet impactful domain.
>
---
#### [new 077] Differential Mamba
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决序列模型中注意力过载问题。工作包括改进Mamba架构，提出新的差异机制以提升检索能力与性能。**

- **链接: [http://arxiv.org/pdf/2507.06204v1](http://arxiv.org/pdf/2507.06204v1)**

> **作者:** Nadav Schneider; Itamar Zimerman; Eliya Nachmani
>
> **摘要:** Sequence models like Transformers and RNNs often overallocate attention to irrelevant context, leading to noisy intermediate representations. This degrades LLM capabilities by promoting hallucinations, weakening long-range and retrieval abilities, and reducing robustness. Recent work has shown that differential design can mitigate this issue in Transformers, improving their effectiveness across various applications. In this paper, we explore whether these techniques, originally developed for Transformers, can be applied to Mamba, a recent architecture based on selective state-space layers that achieves Transformer-level performance with greater efficiency. We show that a naive adaptation of differential design to Mamba is insufficient and requires careful architectural modifications. To address this, we introduce a novel differential mechanism for Mamba, empirically validated on language modeling benchmarks, demonstrating improved retrieval capabilities and superior performance over vanilla Mamba. Finally, we conduct extensive ablation studies and empirical analyses to justify our design choices and provide evidence that our approach effectively mitigates the overallocation problem in Mamba-based models. Our code is publicly available.
>
---
#### [new 078] Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于教育AI任务，旨在解决大规模交互式教学与教学质量评估问题。提出WikiHowAgent多智能体框架，整合教师、学习者和评估模块，提升程序性学习效果并评估教学品质。**

- **链接: [http://arxiv.org/pdf/2507.05528v1](http://arxiv.org/pdf/2507.05528v1)**

> **作者:** Jiahuan Pei; Fanghua Ye; Xin Sun; Wentao Deng; Koen Hindriks; Junxiao Wang
>
> **备注:** 14 pages
>
> **摘要:** Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
>
---
#### [new 079] MusiScene: Leveraging MU-LLaMA for Scene Imagination and Enhanced Video Background Music Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于音乐与场景生成任务，旨在解决音乐场景想象和视频背景音乐生成问题。工作包括构建数据集、微调模型并验证效果。**

- **链接: [http://arxiv.org/pdf/2507.05894v1](http://arxiv.org/pdf/2507.05894v1)**

> **作者:** Fathinah Izzati; Xinyue Li; Yuxuan Wu; Gus Xia
>
> **摘要:** Humans can imagine various atmospheres and settings when listening to music, envisioning movie scenes that complement each piece. For example, slow, melancholic music might evoke scenes of heartbreak, while upbeat melodies suggest celebration. This paper explores whether a Music Language Model, e.g. MU-LLaMA, can perform a similar task, called Music Scene Imagination (MSI), which requires cross-modal information from video and music to train. To improve upon existing music captioning models which focusing solely on musical elements, we introduce MusiScene, a music captioning model designed to imagine scenes that complement each music. In this paper, (1) we construct a large-scale video-audio caption dataset with 3,371 pairs, (2) we finetune Music Understanding LLaMA for the MSI task to create MusiScene, and (3) we conduct comprehensive evaluations and prove that our MusiScene is more capable of generating contextually relevant captions compared to MU-LLaMA. We leverage the generated MSI captions to enhance Video Background Music Generation (VBMG) from text.
>
---
#### [new 080] CultureCLIP: Empowering CLIP with Cultural Awareness through Synthetic Images and Contextualized Captions
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决文化差异导致的细粒度概念识别问题。通过构建合成数据集并微调CLIP，提升模型对文化差异的敏感性。**

- **链接: [http://arxiv.org/pdf/2507.06210v1](http://arxiv.org/pdf/2507.06210v1)**

> **作者:** Yuchen Huang; Zhiyuan Fan; Zhitao He; Sandeep Polisetty; Wenyan Li; Yi R. Fung
>
> **备注:** 25 pages, COLM 2025
>
> **摘要:** Pretrained vision-language models (VLMs) such as CLIP excel in multimodal understanding but struggle with contextually relevant fine-grained visual features, making it difficult to distinguish visually similar yet culturally distinct concepts. This limitation stems from the scarcity of high-quality culture-specific datasets, the lack of integrated contextual knowledge, and the absence of hard negatives highlighting subtle distinctions. To address these challenges, we first design a data curation pipeline that leverages open-sourced VLMs and text-to-image diffusion models to construct CulTwin, a synthetic cultural dataset. This dataset consists of paired concept-caption-image triplets, where concepts visually resemble each other but represent different cultural contexts. Then, we fine-tune CLIP on CulTwin to create CultureCLIP, which aligns cultural concepts with contextually enhanced captions and synthetic images through customized contrastive learning, enabling finer cultural differentiation while preserving generalization capabilities. Experiments on culturally relevant benchmarks show that CultureCLIP outperforms the base CLIP, achieving up to a notable 5.49% improvement in fine-grained concept recognition on certain tasks, while preserving CLIP's original generalization ability, validating the effectiveness of our data synthesis and VLM backbone training paradigm in capturing subtle cultural distinctions.
>
---
#### [new 081] Affective-ROPTester: Capability and Bias Analysis of LLMs in Predicting Retinopathy of Prematurity
- **分类: cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决LLMs在早产儿视网膜病变风险预测中的能力与情感偏见问题。构建了CROP数据集，并提出Affective-ROPTester框架进行评估。**

- **链接: [http://arxiv.org/pdf/2507.05816v1](http://arxiv.org/pdf/2507.05816v1)**

> **作者:** Shuai Zhao; Yulin Zhang; Luwei Xiao; Xinyi Wu; Yanhao Jia; Zhongliang Guo; Xiaobao Wu; Cong-Duy Nguyen; Guoming Zhang; Anh Tuan Luu
>
> **摘要:** Despite the remarkable progress of large language models (LLMs) across various domains, their capacity to predict retinopathy of prematurity (ROP) risk remains largely unexplored. To address this gap, we introduce a novel Chinese benchmark dataset, termed CROP, comprising 993 admission records annotated with low, medium, and high-risk labels. To systematically examine the predictive capabilities and affective biases of LLMs in ROP risk stratification, we propose Affective-ROPTester, an automated evaluation framework incorporating three prompting strategies: Instruction-based, Chain-of-Thought (CoT), and In-Context Learning (ICL). The Instruction scheme assesses LLMs' intrinsic knowledge and associated biases, whereas the CoT and ICL schemes leverage external medical knowledge to enhance predictive accuracy. Crucially, we integrate emotional elements at the prompt level to investigate how different affective framings influence the model's ability to predict ROP and its bias patterns. Empirical results derived from the CROP dataset yield two principal observations. First, LLMs demonstrate limited efficacy in ROP risk prediction when operating solely on intrinsic knowledge, yet exhibit marked performance gains when augmented with structured external inputs. Second, affective biases are evident in the model outputs, with a consistent inclination toward overestimating medium- and high-risk cases. Third, compared to negative emotions, positive emotional framing contributes to mitigating predictive bias in model outputs. These findings highlight the critical role of affect-sensitive prompt engineering in enhancing diagnostic reliability and emphasize the utility of Affective-ROPTester as a framework for evaluating and mitigating affective bias in clinical language modeling systems.
>
---
#### [new 082] Structured Captions Improve Prompt Adherence in Text-to-Image Models (Re-LAION-Caption 19M)
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于文本到图像生成任务，旨在解决模型对提示理解不足的问题。通过构建结构化标题数据集，提升模型生成效果与对齐度。**

- **链接: [http://arxiv.org/pdf/2507.05300v1](http://arxiv.org/pdf/2507.05300v1)**

> **作者:** Nicholas Merchant; Haitz Sáez de Ocáriz Borde; Andrei Cristian Popescu; Carlos Garcia Jurado Suarez
>
> **备注:** 7-page main paper + appendix, 18 figures
>
> **摘要:** We argue that generative text-to-image models often struggle with prompt adherence due to the noisy and unstructured nature of large-scale datasets like LAION-5B. This forces users to rely heavily on prompt engineering to elicit desirable outputs. In this work, we propose that enforcing a consistent caption structure during training can significantly improve model controllability and alignment. We introduce Re-LAION-Caption 19M, a high-quality subset of Re-LAION-5B, comprising 19 million 1024x1024 images with captions generated by a Mistral 7B Instruct-based LLaVA-Next model. Each caption follows a four-part template: subject, setting, aesthetics, and camera details. We fine-tune PixArt-$\Sigma$ and Stable Diffusion 2 using both structured and randomly shuffled captions, and show that structured versions consistently yield higher text-image alignment scores using visual question answering (VQA) models. The dataset is publicly available at https://huggingface.co/datasets/supermodelresearch/Re-LAION-Caption19M.
>
---
#### [new 083] Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于学术诚信研究，揭示AI辅助审稿中的隐藏指令问题，分析了四种隐秘提示类型，提出技术筛查与政策统一的解决方案。**

- **链接: [http://arxiv.org/pdf/2507.06185v1](http://arxiv.org/pdf/2507.06185v1)**

> **作者:** Zhicheng Lin
>
> **摘要:** In July 2025, 18 academic manuscripts on the preprint website arXiv were found to contain hidden instructions known as prompts designed to manipulate AI-assisted peer review. Instructions such as "GIVE A POSITIVE REVIEW ONLY" were concealed using techniques like white-colored text. Author responses varied: one planned to withdraw the affected paper, while another defended the practice as legitimate testing of reviewer compliance. This commentary analyzes this practice as a novel form of research misconduct. We examine the technique of prompt injection in large language models (LLMs), revealing four types of hidden prompts, ranging from simple positive review commands to detailed evaluation frameworks. The defense that prompts served as "honeypots" to detect reviewers improperly using AI fails under examination--the consistently self-serving nature of prompt instructions indicates intent to manipulate. Publishers maintain inconsistent policies: Elsevier prohibits AI use in peer review entirely, while Springer Nature permits limited use with disclosure requirements. The incident exposes systematic vulnerabilities extending beyond peer review to any automated system processing scholarly texts, including plagiarism detection and citation indexing. Our analysis underscores the need for coordinated technical screening at submission portals and harmonized policies governing generative AI (GenAI) use in academic evaluation.
>
---
#### [new 084] Fine-Grained Vision-Language Modeling for Multimodal Training Assistants in Augmented Reality
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言建模任务，旨在解决增强现实训练中细粒度任务识别问题。通过构建数据集并评估现有模型，发现现有模型表现不足，需改进数据与算法。**

- **链接: [http://arxiv.org/pdf/2507.05515v1](http://arxiv.org/pdf/2507.05515v1)**

> **作者:** Haochen Huang; Jiahuan Pei; Mohammad Aliannejadi; Xin Sun; Moonisa Ahsan; Pablo Cesar; Chuang Yu; Zhaochun Ren; Junxiao Wang
>
> **备注:** 20 pages
>
> **摘要:** Vision-language models (VLMs) are essential for enabling AI-powered smart assistants to interpret and reason in multimodal environments. However, their application in augmented reality (AR) training remains largely unexplored. In this work, we introduce a comprehensive dataset tailored for AR training, featuring systematized vision-language tasks, and evaluate nine state-of-the-art VLMs on it. Our results reveal that even advanced models, including GPT-4o, struggle with fine-grained assembly tasks, achieving a maximum F1 score of just 40.54% on state detection. These findings highlight the demand for enhanced datasets, benchmarks, and further research to improve fine-grained vision-language alignment. Beyond technical contributions, our work has broader social implications, particularly in empowering blind and visually impaired users with equitable access to AI-driven learning opportunities. We provide all related resources, including the dataset, source code, and evaluation results, to support the research community.
>
---
## 更新

#### [replaced 001] A Multi-Task and Multi-Label Classification Model for Implicit Discourse Relation Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.08971v3](http://arxiv.org/pdf/2408.08971v3)**

> **作者:** Nelson Filipe Costa; Leila Kosseim
>
> **备注:** Accepted at SIGDIAL 2025
>
> **摘要:** We propose a novel multi-label classification approach to implicit discourse relation recognition (IDRR). Our approach features a multi-task model that jointly learns multi-label representations of implicit discourse relations across all three sense levels in the PDTB 3.0 framework. The model can also be adapted to the traditional single-label IDRR setting by selecting the sense with the highest probability in the multi-label representation. We conduct extensive experiments to identify optimal model configurations and loss functions in both settings. Our approach establishes the first benchmark for multi-label IDRR and achieves SOTA results on single-label IDRR using DiscoGeM. Finally, we evaluate our model on the PDTB 3.0 corpus in the single-label setting, presenting the first analysis of transfer learning between the DiscoGeM and PDTB 3.0 corpora for IDRR.
>
---
#### [replaced 002] Analytic Subspace Routing: How Recursive Least Squares Works in Continual Learning of Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.13575v2](http://arxiv.org/pdf/2503.13575v2)**

> **作者:** Kai Tong; Kang Pan; Xiao Zhang; Erli Meng; Run He; Yawen Cui; Nuoyan Guo; Huiping Zhuang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) possess encompassing capabilities that can process diverse language-related tasks. However, finetuning on LLMs will diminish this general skills and continual finetuning will further cause severe degradation on accumulated knowledge. Recently, Continual Learning (CL) in Large Language Models (LLMs) arises which aims to continually adapt the LLMs to new tasks while maintaining previously learned knowledge and inheriting general skills. Existing techniques either leverage previous data to replay, leading to extra computational costs, or utilize a single parameter-efficient module to learn the downstream task, constraining new knowledge absorption with interference between different tasks. Toward these issues, this paper proposes Analytic Subspace Routing(ASR) to address these challenges. For each task, we isolate the learning within a subspace of deep layers' features via low-rank adaptation, eliminating knowledge interference between different tasks. Additionally, we propose an analytic routing mechanism to properly utilize knowledge learned in different subspaces. Our approach employs Recursive Least Squares to train a multi-task router model, allowing the router to dynamically adapt to incoming data without requiring access to historical data. Also, the router effectively assigns the current task to an appropriate subspace and has a non-forgetting property of previously learned tasks with a solid theoretical guarantee. Experimental results demonstrate that our method achieves near-perfect retention of prior knowledge while seamlessly integrating new information, effectively overcoming the core limitations of existing methods. Our code will be released after acceptance.
>
---
#### [replaced 003] Redefining Evaluation Standards: A Unified Framework for Evaluating the Korean Capabilities of Language Models
- **分类: cs.CE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.22968v4](http://arxiv.org/pdf/2503.22968v4)**

> **作者:** Hanwool Lee; Dasol Choi; Sooyong Kim; Ilgyun Jung; Sangwon Baek; Guijin Son; Inseon Hwang; Naeun Lee; Seunghyeok Hong
>
> **摘要:** Recent advancements in Korean large language models (LLMs) have driven numerous benchmarks and evaluation methods, yet inconsistent protocols cause up to 10 p.p performance gaps across institutions. Overcoming these reproducibility gaps does not mean enforcing a one-size-fits-all evaluation. Rather, effective benchmarking requires diverse experimental approaches and a framework robust enough to support them. To this end, we introduce HRET (Haerae Evaluation Toolkit), an open-source, registry-based framework that unifies Korean LLM assessment. HRET integrates major Korean benchmarks, multiple inference backends, and multi-method evaluation, with language consistency enforcement to ensure genuine Korean outputs. Its modular registry design also enables rapid incorporation of new datasets, methods, and backends, ensuring the toolkit adapts to evolving research needs. Beyond standard accuracy metrics, HRET incorporates Korean-focused output analyses-morphology-aware Type-Token Ratio (TTR) for evaluating lexical diversity and systematic keyword-omission detection for identifying missing concepts-to provide diagnostic insights into language-specific behaviors. These targeted analyses help researchers pinpoint morphological and semantic shortcomings in model outputs, guiding focused improvements in Korean LLM development.
>
---
#### [replaced 004] Evaluating AI Counseling in Japanese: Counselor, Client, and Evaluator Roles Assessed by Motivational Interviewing Criteria
- **分类: cs.CL; cs.AI; cs.HC; 68T50; I.2.7; H.5.2; J.4**

- **链接: [http://arxiv.org/pdf/2507.02950v2](http://arxiv.org/pdf/2507.02950v2)**

> **作者:** Keita Kiuchi; Yoshikazu Fujimoto; Hideyuki Goto; Tomonori Hosokawa; Makoto Nishimura; Yosuke Sato; Izumi Sezai
>
> **备注:** 70 pages, 0 figures, 9 tables; data and code at https://osf.io/p8c39/files/2e58c42f-a7ba-45f2-aa60-265e107e36db
>
> **摘要:** This study provides the first comprehensive evaluation of large language model (LLM) performance across three counseling roles in Japanese-language therapeutic contexts. We simultaneously assessed counselor artificial intelligence (AI) systems (GPT-4-turbo with zeroshot prompting or Structured Multi-step Dialogue Prompts (SMDP), Claude-3-Opus-SMDP), client AI simulations, and evaluation AI systems (o3, Claude-3.7-Sonnet, Gemini-2.5-pro). Human experts (n = 15) with extensive counseling experience evaluated AI-generated dialogues using the Motivational Interviewing Treatment Integrity (MITI) Coding Manual 4.2.1. Notably, SMDP implementation significantly enhanced counselor AI performance across all MITI global ratings compared with zeroshot prompting, with no significant differences between GPT-SMDP and Opus-SMDP. Evaluation AIs showed comparable performance to human raters for Cultivating Change Talk but systematically overestimated Softening Sustain Talk and the overall quality metrics. Model-specific biases emerged: Gemini emphasized power-sharing, o3 focused on technical proficiency, and Sonnet prioritized emotional expression. Client AI simulations exhibited a limited emotional range and unnaturally high compliance, indicating the need for enhanced realism. These findings establish benchmarks for AI-assisted counseling in non-English contexts and identify critical areas for improvement through advanced prompt engineering, retrieval-augmented generation, and targeted fine-tuning, with important implications for developing culturally sensitive AI mental health tools.
>
---
#### [replaced 005] Adaptive Tool Use in Large Language Models with Meta-Cognition Trigger
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12961v2](http://arxiv.org/pdf/2502.12961v2)**

> **作者:** Wenjun Li; Dexun Li; Kuicai Dong; Cong Zhang; Hao Zhang; Weiwen Liu; Yasheng Wang; Ruiming Tang; Yong Liu
>
> **备注:** 25 pages, camera ready version for ACL-2025
>
> **摘要:** Large language models (LLMs) have shown remarkable emergent capabilities, transforming the execution of functional tasks by leveraging external tools for complex problems that require specialized processing or up-to-date data. While existing research expands LLMs access to diverse tools (e.g., program interpreters, search engines, calculators), the necessity of using these tools is often overlooked, leading to indiscriminate tool invocation. This naive approach raises two key issues: increased latency due to unnecessary tool calls, and potential errors resulting from faulty interactions with external tools. In this paper, we introduce meta-cognition as a proxy for LLMs self-assessment of their capabilities, reflecting the model's awareness of its own limitations. Based on this, we propose MeCo, an adaptive decision-making strategy for external tool use. MeCo quantifies metacognitive scores by capturing high-level cognitive signals in the representation space, guiding when to invoke tools. Notably, MeCo is fine-tuning-free and incurs minimal cost. Experiments across multiple backbone models and benchmarks show that MeCo reliably detects LLMs' internal cognitive signals and significantly improves tool-use decision-making.
>
---
#### [replaced 006] Agents Are All You Need for LLM Unlearning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00406v2](http://arxiv.org/pdf/2502.00406v2)**

> **作者:** Debdeep Sanyal; Murari Mandal
>
> **备注:** Accepted to COLM 2025
>
> **摘要:** Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. In this work we show that \textit{agents might be all we need for effective and practical inference-time LLM unlearning}. We present the first agentic LLM unlearning (\texttt{ALU}) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our \texttt{ALU} framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and \texttt{ALU} seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that \texttt{ALU} consistently stands out as the most robust inference-time LLM unlearning framework among current state-of-the-art methods while incurring time cost that remains effectively constant regardless of the number of unlearning targets. We further highlight \texttt{ALU}'s superior performance compared to existing methods when evaluated at scale. Specifically, \texttt{ALU} is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods.
>
---
#### [replaced 007] Early-Exit and Instant Confidence Translation Quality Estimation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14429v2](http://arxiv.org/pdf/2502.14429v2)**

> **作者:** Vilém Zouhar; Maike Züfle; Beni Egressy; Julius Cheng; Mrinmaya Sachan; Jan Niehues
>
> **摘要:** Quality estimation is omnipresent in machine translation, for both evaluation and generation. Unfortunately, quality estimation models are often opaque and computationally expensive, making them impractical to be part of large-scale pipelines. In this work, we tackle two connected challenges: (1) reducing the cost of quality estimation at scale, and (2) developing an inexpensive uncertainty estimation method for quality estimation. To address the latter, we introduce Instant Confidence COMET, an uncertainty-aware quality estimation model that matches the performance of previous approaches at a fraction of their costs. We extend this to Early-Exit COMET, a quality estimation model that can compute quality scores and associated confidences already at early model layers, allowing us to early-exit computations and reduce evaluation costs. We also apply our model to machine translation reranking. We combine Early-Exit COMET with an upper confidence bound bandit algorithm to find the best candidate from a large pool without having to run the full evaluation model on all candidates. In both cases (evaluation and reranking) our methods reduce the required compute by 50% with very little degradation in performance. Finally, we show how Instant Confidence COMET can be used to decide which translations a human evaluator should score rather than relying on the COMET score.
>
---
#### [replaced 008] SIGIR 2025 -- LiveRAG Challenge Report
- **分类: cs.CL; cs.IR; H.3.3**

- **链接: [http://arxiv.org/pdf/2507.04942v2](http://arxiv.org/pdf/2507.04942v2)**

> **作者:** David Carmel; Simone Filice; Guy Horowitz; Yoelle Maarek; Oren Somekh; Ran Tavory; Mehdi Ghissassi; Edo Liberty; Roy Miara
>
> **备注:** 9 pages, 5 tables
>
> **摘要:** The LiveRAG Challenge at SIGIR 2025, held between March and May 2025, provided a competitive platform for advancing Retrieval-Augmented Generation (RAG) technologies. Participants from academia and industry were invited to develop a RAG-based question-answering system using a fixed corpus (Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal was to facilitate challenging comparisons of retrieval and prompting strategies. During the Live Challenge Day, 70 teams from 27 different countries provided answers and supportive information to 500 unseen questions within a strict two-hour time window. Evaluation was conducted in two stages: first an automated LLM-as-a-judge approach was used to compute correctness and faithfulness score, then a manual review of top ranked submissions was conducted. The finalists were announced on June 12, 2025, with prizes awarded during the LiveRAG Workshop at SIGIR 2025 in Padua, Italy.
>
---
#### [replaced 009] FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.08938v2](http://arxiv.org/pdf/2506.08938v2)**

> **作者:** Qinggang Zhang; Zhishang Xiang; Yilin Xiao; Le Wang; Junhui Li; Xinrun Wang; Jinsong Su
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Large language models (LLMs) augmented with retrieval systems have demonstrated significant potential in handling knowledge-intensive tasks. However, these models often struggle with unfaithfulness issues, generating outputs that either ignore the retrieved context or inconsistently blend it with the LLM`s parametric knowledge. This issue is particularly severe in cases of knowledge conflict, where the retrieved context conflicts with the model`s parametric knowledge. While existing faithful RAG approaches enforce strict context adherence through well-designed prompts or modified decoding strategies, our analysis reveals a critical limitation: they achieve faithfulness by forcibly suppressing the model`s parametric knowledge, which undermines the model`s internal knowledge structure and increases the risk of misinterpreting the context. To this end, this paper proposes FaithfulRAG, a novel framework that resolves knowledge conflicts by explicitly modeling discrepancies between the model`s parametric knowledge and retrieved context. Specifically, FaithfulRAG identifies conflicting knowledge at the fact level and designs a self-thinking process, allowing LLMs to reason about and integrate conflicting facts before generating responses. Extensive experiments demonstrate that our method outperforms state-of-the-art methods. The code is available at https://github.com/DeepLearnXMU/Faithful-RAG
>
---
#### [replaced 010] PulseReddit: A Novel Reddit Dataset for Benchmarking MAS in High-Frequency Cryptocurrency Trading
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.03861v2](http://arxiv.org/pdf/2506.03861v2)**

> **作者:** Qiuhan Han; Qian Wang; Atsushi Yoshikawa; Masayuki Yamamura
>
> **摘要:** High-Frequency Trading (HFT) is pivotal in cryptocurrency markets, demanding rapid decision-making. Social media platforms like Reddit offer valuable, yet underexplored, information for such high-frequency, short-term trading. This paper introduces \textbf{PulseReddit}, a novel dataset that is the first to align large-scale Reddit discussion data with high-frequency cryptocurrency market statistics for short-term trading analysis. We conduct an extensive empirical study using Large Language Model (LLM)-based Multi-Agent Systems (MAS) to investigate the impact of social sentiment from PulseReddit on trading performance. Our experiments conclude that MAS augmented with PulseReddit data achieve superior trading outcomes compared to traditional baselines, particularly in bull markets, and demonstrate robust adaptability across different market regimes. Furthermore, our research provides conclusive insights into the performance-efficiency trade-offs of different LLMs, detailing significant considerations for practical model selection in HFT applications. PulseReddit and our findings establish a foundation for advanced MAS research in HFT, demonstrating the tangible benefits of integrating social media.
>
---
#### [replaced 011] MAMUT: A Novel Framework for Modifying Mathematical Formulas for the Generation of Specialized Datasets for Language Model Training
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20855v2](http://arxiv.org/pdf/2502.20855v2)**

> **作者:** Jonathan Drechsel; Anja Reusch; Steffen Herbold
>
> **摘要:** Mathematical formulas are a fundamental and widely used component in various scientific fields, serving as a universal language for expressing complex concepts and relationships. While state-of-the-art transformer models excel in processing and understanding natural language, they encounter challenges with mathematical notation, which involves a complex structure and diverse representations. This study focuses on the development of specialized training datasets to enhance the encoding of mathematical content. We introduce Math Mutator (MAMUT), a framework capable of generating equivalent and falsified versions of a given mathematical formula in LaTeX notation, effectively capturing the mathematical variety in notation of the same concept. Based on MAMUT, we have generated four large mathematical datasets containing diverse notation. Experiments show that models trained on these datasets exhibit new SoTA performance on mathematical retrieval tasks. We publish our code, generated datasets, and pretrained mathematical models: https://github.com/aieng-lab/math-mutator.
>
---
#### [replaced 012] Truth Neurons
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12182v3](http://arxiv.org/pdf/2505.12182v3)**

> **作者:** Haohang Li; Yupeng Cao; Yangyang Yu; Jordan W. Suchow; Zining Zhu
>
> **摘要:** Despite their remarkable success and deployment across diverse workflows, language models sometimes produce untruthful responses. Our limited understanding of how truthfulness is mechanistically encoded within these models jeopardizes their reliability and safety. In this paper, we propose a method for identifying representations of truthfulness at the neuron level. We show that language models contain truth neurons, which encode truthfulness in a subject-agnostic manner. Experiments conducted across models of varying scales validate the existence of truth neurons, confirming that the encoding of truthfulness at the neuron level is a property shared by many language models. The distribution patterns of truth neurons over layers align with prior findings on the geometry of truthfulness. Selectively suppressing the activations of truth neurons found through the TruthfulQA dataset degrades performance both on TruthfulQA and on other benchmarks, showing that the truthfulness mechanisms are not tied to a specific dataset. Our results offer novel insights into the mechanisms underlying truthfulness in language models and highlight potential directions toward improving their trustworthiness and reliability.
>
---
#### [replaced 013] BMMR: A Large-Scale Bilingual Multimodal Multi-Discipline Reasoning Dataset
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.03483v2](http://arxiv.org/pdf/2507.03483v2)**

> **作者:** Zhiheng Xi; Guanyu Li; Yutao Fan; Honglin Guo; Yufang Liu; Xiaoran Fan; Jiaqi Liu; Jingchao Ding; Wangmeng Zuo; Zhenfei Yin; Lei Bai; Tao Ji; Tao Gui; Qi Zhang; Philip Torr; Xuanjing Huang
>
> **备注:** Preprint
>
> **摘要:** In this paper, we introduce BMMR, a large-scale bilingual, multimodal, multi-disciplinary reasoning dataset for the community to develop and evaluate large multimodal models (LMMs). BMMR comprises 110k college-level questions spanning 300 UNESCO-defined subjects, spanning diverse formats-multiple-choice, fill-in-the-blank, and open-ended QA-and sourced from both print and digital media such as books, exams, and quizzes. All data are curated and filtered via a human-in-the-loop and scalable framework, and each instance is paired with a high-quality reasoning path. The dataset is organized into two parts: BMMR-Eval that comprises 20,458 high-quality instances to comprehensively assess LMMs' knowledge and reasoning across multiple disciplines in both Chinese and English; and BMMR-Train that contains 88,991 instances to support further research and development, extending the current focus on mathematical reasoning to diverse disciplines and domains. In addition, we propose the process-based multi-discipline verifier (i.e., BMMR-Verifier) for accurate and fine-grained evaluation of reasoning paths. Extensive experiments on 24 models reveal that (i) even SOTA models (e.g., o3 and Gemini-2.5-Pro) leave substantial headroom on BMMR-Eval; (ii) reasoning models exhibit discipline bias and outperform LMMs only on specific subjects; (iii) open-source models still trail their proprietary counterparts; and (iv) fine-tuning on BMMR-Train narrows this gap. Additionally, we conduct reasoning-chain analyses using BMMR-Verifier and other in-depth studies, uncovering the challenges LMMs currently face in multidisciplinary reasoning. We will release the data, and we hope our work can offer insights and contributions to the community.
>
---
#### [replaced 014] Efficient Detection of Intermittent Job Failures Using Few-Shot Learning
- **分类: cs.SE; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.04173v2](http://arxiv.org/pdf/2507.04173v2)**

> **作者:** Henri Aïdasso; Francis Bordeleau; Ali Tizghadam
>
> **备注:** Accepted at the 41st International Conference on Software Maintenance and Evolution - ICSME 2025 (Industry Track); 12 pages; typos corrected
>
> **摘要:** One of the main challenges developers face in the use of continuous integration (CI) and deployment pipelines is the occurrence of intermittent job failures, which result from unexpected non-deterministic issues (e.g., flaky tests or infrastructure problems) rather than regular code-related errors such as bugs. Prior studies developed machine learning (ML) models trained on large datasets of job logs to classify job failures as either intermittent or regular. As an alternative to costly manual labeling of large datasets, the state-of-the-art (SOTA) approach leveraged a heuristic based on non-deterministic job reruns. However, this method mislabels intermittent job failures as regular in contexts where rerunning suspicious job failures is not an explicit policy, and therefore limits the SOTA's performance in practice. In fact, our manual analysis of 2,125 job failures from 5 industrial and 1 open-source projects reveals that, on average, 32% of intermittent job failures are mislabeled as regular. To address these limitations, this paper introduces a novel approach to intermittent job failure detection using few-shot learning (FSL). Specifically, we fine-tune a small language model using a few number of manually labeled log examples to generate rich embeddings, which are then used to train an ML classifier. Our FSL-based approach achieves 70-88% F1-score with only 12 shots in all projects, outperforming the SOTA, which proved ineffective (34-52% F1-score) in 4 projects. Overall, this study underlines the importance of data quality over quantity and provides a more efficient and practical framework for the detection of intermittent job failures in organizations.
>
---
#### [replaced 015] Instruction Following by Boosting Attention of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.13734v2](http://arxiv.org/pdf/2506.13734v2)**

> **作者:** Vitoria Guardieiro; Adam Stein; Avishree Khare; Eric Wong
>
> **摘要:** Controlling the generation of large language models (LLMs) remains a central challenge to ensure their safe and reliable deployment. While prompt engineering and finetuning are common approaches, recent work has explored latent steering, a lightweight technique that alters LLM internal activations to guide generation. However, subsequent studies revealed latent steering's effectiveness to be limited, often underperforming simple instruction prompting. To address this limitation, we first establish a benchmark across diverse behaviors for standardized evaluation of steering techniques. Building on insights from this benchmark, we introduce Instruction Attention Boosting (InstABoost), a latent steering method that boosts the strength of instruction prompting by altering the model's attention during generation. InstABoost combines the strengths of existing approaches and is theoretically supported by prior work that suggests that in-context rule following in transformer-based models can be controlled by manipulating attention on instructions. Empirically, InstABoost demonstrates superior control success compared to both traditional prompting and latent steering.
>
---
#### [replaced 016] Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.19652v2](http://arxiv.org/pdf/2506.19652v2)**

> **作者:** Lucie Galland; Catherine Pelachaud; Florian Pecune
>
> **摘要:** In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals.
>
---
#### [replaced 017] GMLM: Bridging Graph Neural Networks and Language Models for Heterophilic Node Classification
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05763v4](http://arxiv.org/pdf/2503.05763v4)**

> **作者:** Aarush Sinha
>
> **摘要:** Integrating structured graph data with rich textual information from nodes poses a significant challenge, particularly for heterophilic node classification. Current approaches often struggle with computational costs or effective fusion of disparate modalities. We propose \textbf{Graph Masked Language Model (GMLM)}, a novel architecture efficiently combining Graph Neural Networks (GNNs) with Pre-trained Language Models (PLMs). GMLM introduces three key innovations: (i) a \textbf{dynamic active node selection} strategy for scalable PLM text processing; (ii) a GNN-specific \textbf{contrastive pretraining stage} using soft masking with a learnable graph \texttt{[MASK]} token for robust structural representations; and (iii) a \textbf{dedicated fusion module} integrating RGCN-based GNN embeddings with PLM (GTE-Small \& DistilBERT) embeddings. Extensive experiments on heterophilic benchmarks (Cornell, Wisconsin, Texas) demonstrate GMLM's superiority. Notably, GMLM(DistilBERT) achieves significant performance gains, improving accuracy by over \textbf{4.7\%} on Cornell and over \textbf{2.0\%} on Texas compared to the previous best-performing baselines. This work underscores the benefits of targeted PLM engagement and modality-specific pretraining for improved, efficient learning on text-rich graphs.
>
---
#### [replaced 018] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01366v2](http://arxiv.org/pdf/2501.01366v2)**

> **作者:** Austin T. Wang; ZeMing Gong; Angel X. Chang
>
> **备注:** 24 pages with 8 figures and 14 tables; updated for ACL 2025 camera-ready with additional discussion and figures
>
> **摘要:** 3D visual grounding (3DVG) involves localizing entities in a 3D scene referred to by natural language text. Such models are useful for embodied AI and scene retrieval applications, which involve searching for objects or patterns using natural language descriptions. While recent works have focused on LLM-based scaling of 3DVG datasets, these datasets do not capture the full range of potential prompts which could be specified in the English language. To ensure that we are scaling up and testing against a useful and representative set of prompts, we propose a framework for linguistically analyzing 3DVG prompts and introduce Visual Grounding with Diverse Language in 3D (ViGiL3D), a diagnostic dataset for evaluating visual grounding methods against a diverse set of language patterns. We evaluate existing open-vocabulary 3DVG methods to demonstrate that these methods are not yet proficient in understanding and identifying the targets of more challenging, out-of-distribution prompts, toward real-world applications.
>
---
#### [replaced 019] Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02233v3](http://arxiv.org/pdf/2503.02233v3)**

> **作者:** Hang Zheng; Hongshen Xu; Yuncong Liu; Lu Chen; Pascale Fung; Kai Yu
>
> **摘要:** Large language models (LLMs) are prone to hallucination stemming from misaligned self-awareness, particularly when processing queries exceeding their knowledge boundaries. While existing mitigation strategies employ uncertainty estimation or query rejection mechanisms, they suffer from computational efficiency and sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate utilization of high-confidence outputs, whereas uncertain predictions trigger a slow refinement model for accuracy improvement. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. The framework establishes a scalable paradigm for deploying reliable LLMs in error-sensitive applications, effectively balancing accuracy and practical utility.
>
---
#### [replaced 020] The Impact of Prompt Programming on Function-Level Code Generation
- **分类: cs.SE; cs.CL; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.20545v2](http://arxiv.org/pdf/2412.20545v2)**

> **作者:** Ranim Khojah; Francisco Gomes de Oliveira Neto; Mazen Mohamad; Philipp Leitner
>
> **备注:** Accepted at Transactions on Software Engineering (TSE). CodePromptEval dataset and replication package on GitHub: https://github.com/icetlab/CodePromptEval
>
> **摘要:** Large Language Models (LLMs) are increasingly used by software engineers for code generation. However, limitations of LLMs such as irrelevant or incorrect code have highlighted the need for prompt programming (or prompt engineering) where engineers apply specific prompt techniques (e.g., chain-of-thought or input-output examples) to improve the generated code. While some prompt techniques have been studied, the impact of different techniques -- and their interactions -- on code generation is still not fully understood. In this study, we introduce CodePromptEval, a dataset of 7072 prompts designed to evaluate five prompt techniques (few-shot, persona, chain-of-thought, function signature, list of packages) and their effect on the correctness, similarity, and quality of complete functions generated by three LLMs (GPT-4o, Llama3, and Mistral). Our findings show that while certain prompt techniques significantly influence the generated code, combining multiple techniques does not necessarily improve the outcome. Additionally, we observed a trade-off between correctness and quality when using prompt techniques. Our dataset and replication package enable future research on improving LLM-generated code and evaluating new prompt techniques.
>
---
#### [replaced 021] EEG2TEXT-CN: An Exploratory Study of Open-Vocabulary Chinese Text-EEG Alignment via Large Language Model and Contrastive Learning on ChineseEEG
- **分类: cs.CL; cs.AI; cs.LG; cs.MM; q-bio.NC**

- **链接: [http://arxiv.org/pdf/2506.00854v3](http://arxiv.org/pdf/2506.00854v3)**

> **作者:** Jacky Tai-Yu Lu; Jung Chiang; Chi-Sheng Chen; Anna Nai-Yun Tung; Hsiang Wei Hu; Yuan Chiao Cheng
>
> **摘要:** We propose EEG2TEXT-CN, which, to the best of our knowledge, represents one of the earliest open-vocabulary EEG-to-text generation frameworks tailored for Chinese. Built on a biologically grounded EEG encoder (NICE-EEG) and a compact pretrained language model (MiniLM), our architecture aligns multichannel brain signals with natural language representations via masked pretraining and contrastive learning. Using a subset of the ChineseEEG dataset, where each sentence contains approximately ten Chinese characters aligned with 128-channel EEG recorded at 256 Hz, we segment EEG into per-character embeddings and predict full sentences in a zero-shot setting. The decoder is trained with teacher forcing and padding masks to accommodate variable-length sequences. Evaluation on over 1,500 training-validation sentences and 300 held-out test samples shows promising lexical alignment, with a best BLEU-1 score of 6.38\%. While syntactic fluency remains a challenge, our findings demonstrate the feasibility of non-phonetic, cross-modal language decoding from EEG. This work opens a new direction in multilingual brain-to-text research and lays the foundation for future cognitive-language interfaces in Chinese.
>
---
#### [replaced 022] Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.08324v2](http://arxiv.org/pdf/2411.08324v2)**

> **作者:** Hui Dai; Ryan Teehan; Mengye Ren
>
> **备注:** ICML 2025
>
> **摘要:** Many existing evaluation benchmarks for Large Language Models (LLMs) quickly become outdated due to the emergence of new models and training data. These benchmarks also fall short in assessing how LLM performance changes over time, as they consist of a static set of questions without a temporal dimension. To address these limitations, we propose using future event prediction as a continuous evaluation method to assess LLMs' temporal generalization and forecasting abilities. Our benchmark, Daily Oracle, automatically generates question-answer (QA) pairs from daily news, challenging LLMs to predict "future" event outcomes. Our findings reveal that as pre-training data becomes outdated, LLM performance degrades over time. While Retrieval Augmented Generation (RAG) has the potential to enhance prediction accuracy, the performance degradation pattern persists, highlighting the need for continuous model updates. Code and data are available at https://agenticlearning.ai/daily-oracle.
>
---
#### [replaced 023] Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.18099v2](http://arxiv.org/pdf/2501.18099v2)**

> **作者:** Swarnadeep Saha; Xian Li; Marjan Ghazvininejad; Jason Weston; Tianlu Wang
>
> **备注:** ICML 2025
>
> **摘要:** LLM-as-a-Judge models generate chain-of-thought (CoT) sequences intended to capture the step-bystep reasoning process that underlies the final evaluation of a response. However, due to the lack of human annotated CoTs for evaluation, the required components and structure of effective reasoning traces remain understudied. Consequently, previous approaches often (1) constrain reasoning traces to hand-designed components, such as a list of criteria, reference answers, or verification questions and (2) structure them such that planning is intertwined with the reasoning for evaluation. In this work, we propose EvalPlanner, a preference optimization algorithm for Thinking-LLM-as-a-Judge that first generates an unconstrained evaluation plan, followed by its execution, and then the final judgment. In a self-training loop, EvalPlanner iteratively optimizes over synthetically constructed evaluation plans and executions, leading to better final verdicts. Our method achieves a new state-of-the-art performance for generative reward models on RewardBench (with a score of 93.9), despite being trained on fewer amount of, and synthetically generated, preference pairs. Additional experiments on other benchmarks like RM-Bench, JudgeBench, and FollowBenchEval further highlight the utility of both planning and reasoning for building robust LLM-as-a-Judge reasoning models.
>
---
#### [replaced 024] Overcoming Data Scarcity in Generative Language Modelling for Low-Resource Languages: A Systematic Review
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.04531v2](http://arxiv.org/pdf/2505.04531v2)**

> **作者:** Josh McGiff; Nikola S. Nikolov
>
> **备注:** This work is currently under review. Please do not cite without permission
>
> **摘要:** Generative language modelling has surged in popularity with the emergence of services such as ChatGPT and Google Gemini. While these models have demonstrated transformative potential in productivity and communication, they overwhelmingly cater to high-resource languages like English. This has amplified concerns over linguistic inequality in natural language processing (NLP). This paper presents the first systematic review focused specifically on strategies to address data scarcity in generative language modelling for low-resource languages (LRL). Drawing from 54 studies, we identify, categorise and evaluate technical approaches, including monolingual data augmentation, back-translation, multilingual training, and prompt engineering, across generative tasks. We also analyse trends in architecture choices, language family representation, and evaluation methods. Our findings highlight a strong reliance on transformer-based models, a concentration on a small subset of LRLs, and a lack of consistent evaluation across studies. We conclude with recommendations for extending these methods to a wider range of LRLs and outline open challenges in building equitable generative language systems. Ultimately, this review aims to support researchers and developers in building inclusive AI tools for underrepresented languages, a necessary step toward empowering LRL speakers and the preservation of linguistic diversity in a world increasingly shaped by large-scale language technologies.
>
---
#### [replaced 025] OpenS2S: Advancing Fully Open-Source End-to-End Empathetic Large Speech Language Model
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.05177v2](http://arxiv.org/pdf/2507.05177v2)**

> **作者:** Chen Wang; Tianyu Peng; Wen Yang; Yinan Bai; Guangfu Wang; Jun Lin; Lanpeng Jia; Lingxiang Wu; Jinqiao Wang; Chengqing Zong; Jiajun Zhang
>
> **备注:** Technical Report
>
> **摘要:** Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at https://casia-lm.github.io/OpenS2S
>
---
#### [replaced 026] ALLM4ADD: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.11079v2](http://arxiv.org/pdf/2505.11079v2)**

> **作者:** Hao Gu; Jiangyan Yi; Chenglong Wang; Jianhua Tao; Zheng Lian; Jiayi He; Yong Ren; Yujie Chen; Zhengqi Wen
>
> **备注:** Accepted by ACMMM 2025
>
> **摘要:** Audio deepfake detection (ADD) has grown increasingly important due to the rise of high-fidelity audio generative models and their potential for misuse. Given that audio large language models (ALLMs) have made significant progress in various audio processing tasks, a heuristic question arises: \textit{Can ALLMs be leveraged to solve ADD?}. In this paper, we first conduct a comprehensive zero-shot evaluation of ALLMs on ADD, revealing their ineffectiveness. To this end, we propose ALLM4ADD, an ALLM-driven framework for ADD. Specifically, we reformulate ADD task as an audio question answering problem, prompting the model with the question: ``Is this audio fake or real?''. We then perform supervised fine-tuning to enable the ALLM to assess the authenticity of query audio. Extensive experiments are conducted to demonstrate that our ALLM-based method can achieve superior performance in fake audio detection, particularly in data-scarce scenarios. As a pioneering study, we anticipate that this work will inspire the research community to leverage ALLMs to develop more effective ADD systems. Code is available at https://github.com/ucas-hao/qwen_audio_for_add.git
>
---
#### [replaced 027] Rethinking Associative Memory Mechanism in Induction Head
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.11459v2](http://arxiv.org/pdf/2412.11459v2)**

> **作者:** Shuo Wang; Issei Sato
>
> **备注:** COLM 2025
>
> **摘要:** Induction head mechanism is a part of the computational circuits for in-context learning (ICL) that enable large language models (LLMs) to adapt to new tasks without fine-tuning. Most existing work explains the training dynamics behind acquiring such a powerful mechanism. However, the model's ability to coordinate in-context information over long contexts and global knowledge acquired during pretraining remains poorly understood. This paper investigates how a two-layer transformer thoroughly captures in-context information and balances it with pretrained bigram knowledge in next token prediction, from the viewpoint of associative memory. We theoretically analyze the representation of weight matrices in attention layers and the resulting logits when a transformer is given prompts generated by a bigram model. In the experiments, we design specific prompts to evaluate whether the outputs of the trained transformer align with the theoretical results.
>
---
#### [replaced 028] Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03711v2](http://arxiv.org/pdf/2507.03711v2)**

> **作者:** Sang Quang Nguyen; Kiet Van Nguyen; Vinh-Tiep Nguyen; Thanh Duc Ngo; Ngan Luu-Thuy Nguyen; Dinh-Duy Le
>
> **备注:** Accepted paper at MAPR 2025
>
> **摘要:** In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
>
---
#### [replaced 029] A Survey on Transformer Context Extension: Approaches and Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13299v2](http://arxiv.org/pdf/2503.13299v2)**

> **作者:** Yijun Liu; Jinzheng Yu; Yang Xu; Zhongyang Li; Qingfu Zhu
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) based on Transformer have been widely applied in the filed of natural language processing (NLP), demonstrating strong performance, particularly in handling short text tasks. However, when it comes to long context scenarios, the performance of LLMs degrades due to some challenges. To alleviate this phenomenon, there is a number of work proposed recently. In this survey, we first list the challenges of applying pre-trained LLMs to process long contexts. Then systematically review the approaches related to long context and propose our taxonomy categorizing them into four main types: positional encoding, context compression, retrieval augmented, and attention pattern. In addition to the approaches, we focus on the evaluation of long context, organizing relevant data, tasks, and metrics based on existing long context benchmarks. Finally, we summarize unresolved issues in the long context domain and put forward our views on future developments.
>
---
#### [replaced 030] MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23404v3](http://arxiv.org/pdf/2505.23404v3)**

> **作者:** Mingyu Yu; Wei Wang; Yanjie Wei; Sujuan Qin; Fei Gao; Wenmin Li
>
> **摘要:** Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.
>
---
#### [replaced 031] On the Fundamental Impossibility of Hallucination Control in Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.GT; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06382v3](http://arxiv.org/pdf/2506.06382v3)**

> **作者:** Michał P. Karpowicz
>
> **备注:** transformer example extended, discussion and speculation section added
>
> **摘要:** We prove that perfect hallucination control in large language models is mathematically impossible. No LLM inference mechanism can simultaneously achieve truthful response generation, semantic information conservation, relevant knowledge revelation, and knowledge-constrained optimality. This impossibility is fundamental, arising from the mathematical structure of information aggregation itself rather than engineering limitations. The proof spans three mathematical frameworks: auction theory, proper scoring theory for probabilistic predictions, and log-sum-exp analysis for transformer architectures. In each setting, we demonstrate that information aggregation creates unavoidable violations of conservation principles. The Jensen gap in transformer probability aggregation provides a direct measure of this impossibility. These results reframe hallucination from an engineering bug to an inevitable mathematical feature of distributed intelligence. There are fundamental trade-offs between truthfulness, knowledge utilization, and response completeness, providing principled foundations for managing rather than eliminating hallucination. This work reveals deep connections between neural network inference, philosophy of knowledge and reasoning, and classical results in game theory and information theory, opening new research directions for developing beneficial AI systems within mathematical constraints.
>
---
#### [replaced 032] SciMaster: Towards General-Purpose Scientific AI Agents, Part I. X-Master as Foundation: Can We Lead on Humanity's Last Exam?
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05241v2](http://arxiv.org/pdf/2507.05241v2)**

> **作者:** Jingyi Chai; Shuo Tang; Rui Ye; Yuwen Du; Xinyu Zhu; Mengcheng Zhou; Yanfeng Wang; Weinan E; Yuzhi Zhang; Linfeng Zhang; Siheng Chen
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** The rapid advancements of AI agents have ignited the long-held ambition of leveraging them to accelerate scientific discovery. Achieving this goal requires a deep understanding of the frontiers of human knowledge. As such, Humanity's Last Exam (HLE) provides an exceptionally challenging touchstone for evaluating scientific AI agents. In this work, we aim to construct the foundational architecture for general-purpose agents and validate the capabilities through leading performance on HLE. To achieve this, we introduce X-Master, a tool-augmented reasoning agent designed to emulate human researchers by interacting flexibly with external tools during its reasoning process. This agent, guided by the conceptualization of code as an interaction language, can flexibly leverage built-in Python libraries and our customized tools to augment the reasoning. We further scale its capabilities through X-Masters, a scattered-and-stacked agentic workflow that systematically enhances breadth and depth of reasoning. Our open-source solution, X-Masters, sets a new state-of-the-art record on HLE with a score of 32.1%, surpassing OpenAI's and Google's Deep Research (26.6% and 26.9%) and becoming the first to exceed the 30% threshold. This work allows us to gain a deeper understanding of complex task-solving and accumulates valuable experience that can inform future advancements, guiding subsequent model training.
>
---
#### [replaced 033] Healing Powers of BERT: How Task-Specific Fine-Tuning Recovers Corrupted Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.14459v2](http://arxiv.org/pdf/2406.14459v2)**

> **作者:** Shijie Han; Zhenyu Zhang; Andrei Arsene Simion
>
> **摘要:** Language models like BERT excel at sentence classification tasks due to extensive pre-training on general data, but their robustness to parameter corruption is unexplored. To understand this better, we look at what happens if a language model is "broken", in the sense that some of its parameters are corrupted and then recovered by fine-tuning. Strategically corrupting BERT variants at different levels, we find corrupted models struggle to fully recover their original performance, with higher corruption causing more severe degradation. Notably, bottom-layer corruption affecting fundamental linguistic features is more detrimental than top-layer corruption. Our insights contribute to understanding language model robustness and adaptability under adverse conditions, informing strategies for developing resilient NLP systems against parameter perturbations.
>
---
#### [replaced 034] Infini-gram mini: Exact n-gram Search at the Internet Scale with FM-Index
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.12229v2](http://arxiv.org/pdf/2506.12229v2)**

> **作者:** Hao Xu; Jiacheng Liu; Yejin Choi; Noah A. Smith; Hannaneh Hajishirzi
>
> **摘要:** Language models are trained mainly on massive text data from the Internet, and it becomes increasingly important to understand this data source. Exact-match search engines enable searching in large text corpora -- counting string appearances and retrieving the enclosing documents -- yet the high storage overhead hinders their application on Internet-scale data. We present Infini-gram mini, an efficient and scalable system that can make petabyte-level text corpora searchable. Based on the FM-index data structure (Ferragina and Manzini, 2000), which simultaneously indexes and compresses text, our system creates indexes with size only 44% of the corpus. Infini-gram mini greatly improves upon the best existing implementation of FM-index in terms of indexing speed (18$\times$) and memory use during both indexing (3.2$\times$ reduction) and querying (down to a negligible amount). We index 46TB of Internet text in 50 days with a single 128-core CPU node (or 19 hours if using 75 such nodes). We show one important use case of Infini-gram mini in a large-scale analysis of benchmark contamination. We find several core LM evaluation benchmarks to be heavily contaminated in Internet crawls (up to 40% in SQuAD), which could lead to overestimating the capabilities of language models if trained on such data. We host a benchmark contamination bulletin to share the contamination rate of many core and community-contributed benchmarks. We also release a web interface and an API endpoint to serve general search queries on Infini-gram mini indexes.
>
---
#### [replaced 035] What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.17172v2](http://arxiv.org/pdf/2409.17172v2)**

> **作者:** Shashidhar Reddy Javaji; Zining Zhu
>
> **摘要:** Large language models (LLMs) can store a massive amount of knowledge, yet their potential to acquire new knowledge remains unknown. We propose a novel evaluation framework that evaluates this capability. This framework prompts LLMs to generate questions about a statement introducing scientific knowledge, simulating a curious person when facing the statement for the first time. We score the qualities of the generated questions, thereby evaluating the knowledge acquisition potential of the LLM. We apply controlled ablation studies to validate our scoring procedures. Additionally, we created a synthetic dataset consisting of 1101 statements in physics, chemistry, and maths with distinct levels of difficulties, 300 general knowledge statements, and 567 incorrect statements. Human evaluations were conducted to validate our model assessments, achieving an approximate weighted Cohen's kappa of 0.7 on all three metrics considered. We find that while large models like GPT-4 and Mistral 8x7b are adept at generating coherent and relevant questions, the smaller Phi-2 model is equally or more effective. This indicates that size does not solely determine a model's knowledge acquisition potential. The proposed framework quantifies a critical model capability that was commonly overlooked and opens up research opportunities for developing more knowledgeable AI systems
>
---
#### [replaced 036] One fish, two fish, but not the whole sea: Alignment reduces language models' conceptual diversity
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.04427v3](http://arxiv.org/pdf/2411.04427v3)**

> **作者:** Sonia K. Murthy; Tomer Ullman; Jennifer Hu
>
> **备注:** 17 pages, 10 figures; updated with publishing information
>
> **摘要:** Researchers in social science and psychology have recently proposed using large language models (LLMs) as replacements for humans in behavioral research. In addition to arguments about whether LLMs accurately capture population-level patterns, this has raised questions about whether LLMs capture human-like conceptual diversity. Separately, it is debated whether post-training alignment (RLHF or RLAIF) affects models' internal diversity. Inspired by human studies, we use a new way of measuring the conceptual diversity of synthetically-generated LLM "populations" by relating the internal variability of simulated individuals to the population-level variability. We use this approach to evaluate non-aligned and aligned LLMs on two domains with rich human behavioral data. While no model reaches human-like diversity, aligned models generally display less diversity than their instruction fine-tuned counterparts. Our findings highlight potential trade-offs between increasing models' value alignment and decreasing the diversity of their conceptual representations.
>
---
#### [replaced 037] MemOS: A Memory OS for AI System
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.03724v2](http://arxiv.org/pdf/2507.03724v2)**

> **作者:** Zhiyu Li; Shichao Song; Chenyang Xi; Hanyu Wang; Chen Tang; Simin Niu; Ding Chen; Jiawei Yang; Chunyu Li; Qingchen Yu; Jihao Zhao; Yezhaohui Wang; Peng Liu; Zehao Lin; Pengyuan Wang; Jiahao Huo; Tianyi Chen; Kai Chen; Kehang Li; Zhen Tao; Junpeng Ren; Huayi Lai; Hao Wu; Bo Tang; Zhenren Wang; Zhaoxin Fan; Ningyu Zhang; Linfeng Zhang; Junchi Yan; Mingchuan Yang; Tong Xu; Wei Xu; Huajun Chen; Haofeng Wang; Hongkang Yang; Wentao Zhang; Zhi-Qin John Xu; Siheng Chen; Feiyu Xiong
>
> **备注:** 36 pages, 10 figures, 5 tables
>
> **摘要:** Large Language Models (LLMs) have become an essential infrastructure for Artificial General Intelligence (AGI), yet their lack of well-defined memory management systems hinders the development of long-context reasoning, continual personalization, and knowledge consistency.Existing models mainly rely on static parameters and short-lived contextual states, limiting their ability to track user preferences or update knowledge over extended periods.While Retrieval-Augmented Generation (RAG) introduces external knowledge in plain text, it remains a stateless workaround without lifecycle control or integration with persistent representations.Recent work has modeled the training and inference cost of LLMs from a memory hierarchy perspective, showing that introducing an explicit memory layer between parameter memory and external retrieval can substantially reduce these costs by externalizing specific knowledge. Beyond computational efficiency, LLMs face broader challenges arising from how information is distributed over time and context, requiring systems capable of managing heterogeneous knowledge spanning different temporal scales and sources. To address this challenge, we propose MemOS, a memory operating system that treats memory as a manageable system resource. It unifies the representation, scheduling, and evolution of plaintext, activation-based, and parameter-level memories, enabling cost-efficient storage and retrieval. As the basic unit, a MemCube encapsulates both memory content and metadata such as provenance and versioning. MemCubes can be composed, migrated, and fused over time, enabling flexible transitions between memory types and bridging retrieval with parameter-based learning. MemOS establishes a memory-centric system framework that brings controllability, plasticity, and evolvability to LLMs, laying the foundation for continual learning and personalized modeling.
>
---
#### [replaced 038] Empirical evidence of Large Language Model's influence on human spoken communication
- **分类: cs.CY; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.01754v3](http://arxiv.org/pdf/2409.01754v3)**

> **作者:** Hiromu Yakura; Ezequiel Lopez-Lopez; Levin Brinkmann; Ignacio Serna; Prateek Gupta; Ivan Soraperra; Iyad Rahwan
>
> **摘要:** From the invention of writing and the printing press, to television and social media, human history is punctuated by major innovations in communication technology, which fundamentally altered how ideas spread and reshaped our culture. Recent chatbots powered by generative artificial intelligence constitute a novel medium that encodes cultural patterns in their neural representations and disseminates them in conversations with hundreds of millions of people. Understanding whether these patterns transmit into human language, and ultimately shape human culture, is a fundamental question. While fully quantifying the causal impact of a chatbot like ChatGPT on human culture is very challenging, lexicographic shift in human spoken communication may offer an early indicator of such broad phenomenon. Here, we apply econometric causal inference techniques to 740,249 hours of human discourse from 360,445 YouTube academic talks and 771,591 conversational podcast episodes across multiple disciplines. We detect a measurable and abrupt increase in the use of words preferentially generated by ChatGPT, such as delve, comprehend, boast, swift, and meticulous, after its release. These findings suggest a scenario where machines, originally trained on human data and subsequently exhibiting their own cultural traits, can, in turn, measurably reshape human culture. This marks the beginning of a closed cultural feedback loop in which cultural traits circulate bidirectionally between humans and machines. Our results motivate further research into the evolution of human-machine culture, and raise concerns over the erosion of linguistic and cultural diversity, and the risks of scalable manipulation.
>
---
#### [replaced 039] News and Load: Social and Economic Drivers of Regional Multi-horizon Electricity Demand Forecasting
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2406.06641v2](http://arxiv.org/pdf/2406.06641v2)**

> **作者:** Yun Bai; Simon Camal; Andrea Michiorri
>
> **备注:** 12 pages, 12 figures
>
> **摘要:** The relationship between electricity demand and variables such as economic activity and weather patterns is well established. However, this paper explores the connection between electricity demand and social aspects. It further embeds dynamic information about the state of society into energy demand modelling and forecasting approaches. Through the use of natural language processing on a large news corpus, we highlight this important link. This study is conducted in five regions of the UK and Ireland and considers multiple time horizons from 1 to 30 days. It also considers economic variables such as GDP, unemployment and inflation. The textual features used in this study represent central constructs from the word frequencies, topics, word embeddings extracted from the news. The findings indicate that: 1) the textual features are related to various contents, such as military conflicts, transportation, the global pandemic, regional economics, and the international energy market. They exhibit causal relationships with regional electricity demand, which are validated using Granger causality and Double Machine Learning methods. 2) Economic indicators play a more important role in the East Midlands and Northern Ireland, while social indicators are more influential in the West Midlands and the South West of England. 3) The use of these factors improves deterministic forecasting by around 6%.
>
---
#### [replaced 040] Evaluation of OpenAI o1: Opportunities and Challenges of AGI
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2409.18486v2](http://arxiv.org/pdf/2409.18486v2)**

> **作者:** Tianyang Zhong; Zhengliang Liu; Yi Pan; Yutong Zhang; Yifan Zhou; Shizhe Liang; Zihao Wu; Yanjun Lyu; Peng Shu; Xiaowei Yu; Chao Cao; Hanqi Jiang; Hanxu Chen; Yiwei Li; Junhao Chen; Huawen Hu; Yiheng Liu; Huaqin Zhao; Shaochen Xu; Haixing Dai; Lin Zhao; Ruidong Zhang; Wei Zhao; Zhenyuan Yang; Jingyuan Chen; Peilong Wang; Wei Ruan; Hui Wang; Huan Zhao; Jing Zhang; Yiming Ren; Shihuan Qin; Tong Chen; Jiaxi Li; Arif Hassan Zidan; Afrar Jahin; Minheng Chen; Sichen Xia; Jason Holmes; Yan Zhuang; Jiaqi Wang; Bochen Xu; Weiran Xia; Jichao Yu; Kaibo Tang; Yaxuan Yang; Bolun Sun; Tao Yang; Guoyu Lu; Xianqiao Wang; Lilong Chai; He Li; Jin Lu; Xin Zhang; Bao Ge; Xintao Hu; Lian Zhang; Hua Zhou; Lu Zhang; Shu Zhang; Zhen Xiang; Yudan Ren; Jun Liu; Xi Jiang; Yu Bao; Wei Zhang; Xiang Li; Gang Li; Wei Liu; Dinggang Shen; Andrea Sikora; Xiaoming Zhai; Dajiang Zhu; Tuo Zhang; Tianming Liu
>
> **摘要:** This comprehensive study evaluates the performance of OpenAI's o1-preview large language model across a diverse array of complex reasoning tasks, spanning multiple domains, including computer science, mathematics, natural sciences, medicine, linguistics, and social sciences. Through rigorous testing, o1-preview demonstrated remarkable capabilities, often achieving human-level or superior performance in areas ranging from coding challenges to scientific reasoning and from language processing to creative problem-solving. Key findings include: -83.3% success rate in solving complex competitive programming problems, surpassing many human experts. -Superior ability in generating coherent and accurate radiology reports, outperforming other evaluated models. -100% accuracy in high school-level mathematical reasoning tasks, providing detailed step-by-step solutions. -Advanced natural language inference capabilities across general and specialized domains like medicine. -Impressive performance in chip design tasks, outperforming specialized models in areas such as EDA script generation and bug analysis. -Remarkable proficiency in anthropology and geology, demonstrating deep understanding and reasoning in these specialized fields. -Strong capabilities in quantitative investing. O1 has comprehensive financial knowledge and statistical modeling skills. -Effective performance in social media analysis, including sentiment analysis and emotion recognition. The model excelled particularly in tasks requiring intricate reasoning and knowledge integration across various fields. While some limitations were observed, including occasional errors on simpler problems and challenges with certain highly specialized concepts, the overall results indicate significant progress towards artificial general intelligence.
>
---
#### [replaced 041] FRAME: Feedback-Refined Agent Methodology for Enhancing Medical Research Insights
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04649v2](http://arxiv.org/pdf/2505.04649v2)**

> **作者:** Chengzhang Yu; Yiming Zhang; Zhixin Liu; Zenghui Ding; Yining Sun; Zhanpeng Jin
>
> **备注:** 12 pages, 4 figures, 5 table
>
> **摘要:** The automation of scientific research through large language models (LLMs) presents significant opportunities but faces critical challenges in knowledge synthesis and quality assurance. We introduce Feedback-Refined Agent Methodology (FRAME), a novel framework that enhances medical paper generation through iterative refinement and structured feedback. Our approach comprises three key innovations: (1) A structured dataset construction method that decomposes 4,287 medical papers into essential research components through iterative refinement; (2) A tripartite architecture integrating Generator, Evaluator, and Reflector agents that progressively improve content quality through metric-driven feedback; and (3) A comprehensive evaluation framework that combines statistical metrics with human-grounded benchmarks. Experimental results demonstrate FRAME's effectiveness, achieving significant improvements over conventional approaches across multiple models (9.91% average gain with DeepSeek V3, comparable improvements with GPT-4o Mini) and evaluation dimensions. Human evaluation confirms that FRAME-generated papers achieve quality comparable to human-authored works, with particular strength in synthesizing future research directions. The results demonstrated our work could efficiently assist medical research by building a robust foundation for automated medical research paper generation while maintaining rigorous academic standards.
>
---
#### [replaced 042] Towards Exception Safety Code Generation with Intermediate Representation Agents Framework
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.06949v3](http://arxiv.org/pdf/2410.06949v3)**

> **作者:** Xuanming Zhang; Yuxuan Chen; Yuan Yuan; Minlie Huang
>
> **摘要:** Large Language Models (LLMs) often struggle with robust exception handling in generated code, leading to fragile programs that are prone to runtime errors. We propose Seeker, a novel multi-agent framework that enforces exception safety in LLM generated code through an Intermediate Representation (IR) approach. Seeker decomposes exception handling into five specialized agents: Scanner, Detector, Predator, Ranker, and Handler that collaboratively analyze code, detect fragile segments, retrieve best practice exception strategies, and inject robust handling code. We also introduce Common Exception Enumeration (CEE), a comprehensive knowledge base derived from official documentation, technical practices, and real world code, to standardize exception handling strategies. Seeker also incorporates a Deep Retrieval-Augmented Generation (Deep RAG) algorithm to efficiently navigate the exception inheritance hierarchy, cutting down search overhead by 93% while improving accuracy in identifying relevant exceptions. We evaluate Seeker on 15 open source Java projects and multiple benchmarks. Seeker outperforms state of the art baselines, improving exception handling precision by up to 37% and overall code robustness by 38% as measured by expert code review. It significantly closes the gap between LLM and human developers in exception management, achieving a 28% success rate on real world issue fixes (SWE bench) versus 19% by prior methods. Our framework preserves functional correctness of code while proactively handling errors, demonstrating a practical, generalizable solution for safer code generation. In this paper, we discuss the novelty of using intermediate representation and multi-agent collaboration for exception handling, and outline how Seeker can be extended to other programming languages and complex software engineering tasks, aligning LLM-generated code with industrial standard.
>
---
#### [replaced 043] Adsorb-Agent: Autonomous Identification of Stable Adsorption Configurations via Large Language Model Agent
- **分类: cs.CL; cond-mat.mtrl-sci**

- **链接: [http://arxiv.org/pdf/2410.16658v4](http://arxiv.org/pdf/2410.16658v4)**

> **作者:** Janghoon Ock; Radheesh Sharma Meda; Tirtha Vinchurkar; Yayati Jadhav; Amir Barati Farimani
>
> **摘要:** Adsorption energy is a key reactivity descriptor in catalysis. Determining adsorption energy requires evaluating numerous adsorbate-catalyst configurations, making it computationally intensive. Current methods rely on exhaustive sampling, which does not guarantee the identification of the global minimum energy. To address this, we introduce Adsorb-Agent, a Large Language Model (LLM) agent designed to efficiently identify stable adsorption configurations corresponding to the global minimum energy. Adsorb-Agent leverages its built-in knowledge and reasoning to strategically explore configurations, significantly reducing the number of initial setups required while improving energy prediction accuracy. In this study, we also evaluated the performance of different LLMs, including GPT-4o, GPT-4o-mini, Claude-3.7-Sonnet, and DeepSeek-Chat, as the reasoning engine for Adsorb-Agent, with GPT-4o showing the strongest overall performance. Tested on twenty diverse systems, Adsorb-Agent identifies comparable adsorption energies for 84% of cases and achieves lower energies for 35%, particularly excelling in complex systems. It identifies lower energies in 47% of intermetallic systems and 67% of systems with large adsorbates. These findings demonstrate Adsorb-Agent's potential to accelerate catalyst discovery by reducing computational costs and enhancing prediction reliability compared to exhaustive search methods.
>
---
#### [replaced 044] OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07096v2](http://arxiv.org/pdf/2504.07096v2)**

> **作者:** Jiacheng Liu; Taylor Blanton; Yanai Elazar; Sewon Min; YenSung Chen; Arnavi Chheda-Kothary; Huy Tran; Byron Bischoff; Eric Marsh; Michael Schmitz; Cassidy Trier; Aaron Sarnat; Jenna James; Jon Borchardt; Bailey Kuehl; Evie Cheng; Karen Farley; Sruthi Sreeram; Taira Anderson; David Albright; Carissa Schoenick; Luca Soldaini; Dirk Groeneveld; Rock Yuren Pang; Pang Wei Koh; Noah A. Smith; Sophie Lebrecht; Yejin Choi; Hannaneh Hajishirzi; Ali Farhadi; Jesse Dodge
>
> **备注:** ACL 2025 demo track
>
> **摘要:** We present OLMoTrace, the first system that traces the outputs of language models back to their full, multi-trillion-token training data in real time. OLMoTrace finds and shows verbatim matches between segments of language model output and documents in the training text corpora. Powered by an extended version of infini-gram (Liu et al., 2024), our system returns tracing results within a few seconds. OLMoTrace can help users understand the behavior of language models through the lens of their training data. We showcase how it can be used to explore fact checking, hallucination, and the creativity of language models. OLMoTrace is publicly available and fully open-source.
>
---
#### [replaced 045] Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2502.17380v3](http://arxiv.org/pdf/2502.17380v3)**

> **作者:** Qiuming Zhao; Guangzhi Sun; Chao Zhang
>
> **备注:** 13 pages
>
> **摘要:** Language diversity presents a significant challenge in speech-to-text (S2T) tasks, such as automatic speech recognition and translation. Traditional multi-lingual multi-task training approaches aim to address this by jointly optimising multiple speech recognition and translation tasks across various languages. While models like Whisper, built on these strategies, demonstrate strong performance, they still face issues of high computational cost, language interference, suboptimal training configurations, and limited extensibility. To overcome these challenges, we introduce LoRS-Merging (low-rank and sparse model merging), a novel technique designed to efficiently integrate models trained on different languages or tasks while preserving performance and reducing computational overhead. LoRS-Merging combines low-rank and sparse pruning to retain essential structures while eliminating redundant parameters, mitigating language interference, and enhancing extensibility. Experimental results across 10 languages demonstrate that LoRS-Merging significantly outperforms multi-lingual multi-task training, sequential training, and other merging methods, achieving over 20% improvement in normalised performance. Our findings suggest that model merging, particularly LoRS-Merging, is a scalable and effective complement to traditional multi-lingual training strategies for S2T applications.
>
---
#### [replaced 046] On the Role of Feedback in Test-Time Scaling of Agentic AI Workflows
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01931v4](http://arxiv.org/pdf/2504.01931v4)**

> **作者:** Souradip Chakraborty; Mohammadreza Pourreza; Ruoxi Sun; Yiwen Song; Nino Scherrer; Furong Huang; Amrit Singh Bedi; Ahmad Beirami; Jindong Gu; Hamid Palangi; Tomas Pfister
>
> **摘要:** Agentic AI workflows (systems that autonomously plan and act) are becoming widespread, yet their task success rate on complex tasks remains low. A promising solution is inference-time alignment, which uses extra compute at test time to improve performance. Inference-time alignment relies on three components: sampling, evaluation, and feedback. While most prior work studies sampling and automatic evaluation, feedback remains underexplored. To study the role of feedback, we introduce Iterative Agent Decoding (IAD), a procedure that repeatedly inserts feedback extracted from different forms of critiques (reward models or AI-generated textual feedback) between decoding steps. Through IAD, we analyze feedback along four dimensions: (1) its role in the accuracy-compute trade-offs with limited inference budget, (2) quantifying the gains over diversity-only baselines such as best-of-N sampling, (3) effectiveness of composing feedback from reward models versus textual critique, and (4) robustness to noisy or low-quality feedback. Across Sketch2Code, Text2SQL, Intercode, and WebShop, we show that IAD with proper integration of high fidelity feedback leads to consistent gains up to 10 percent absolute performance improvement over various baselines such as best-of-N. Our findings underscore feedback as a crucial knob for inference-time alignment of agentic AI workflows with limited inference budget.
>
---
#### [replaced 047] The Role of Deductive and Inductive Reasoning in Large Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02892v3](http://arxiv.org/pdf/2410.02892v3)**

> **作者:** Chengkun Cai; Xu Zhao; Haoliang Liu; Zhongyu Jiang; Tianfang Zhang; Zongkai Wu; Jenq-Neng Hwang; Lei Li
>
> **备注:** 4 figures, accept at ACL2025 Main
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning tasks, yet their reliance on static prompt structures and limited adaptability to complex scenarios remains a significant challenge. In this paper, we propose the Deductive and InDuctive(DID) method, a novel framework that enhances LLM reasoning by dynamically integrating both deductive and inductive reasoning approaches. Drawing from cognitive science principles, DID implements a dual-metric complexity evaluation system that combines Littlestone dimension and information entropy to precisely assess task difficulty and guide decomposition strategies. DID enables the model to progressively adapt its reasoning pathways based on problem complexity, mirroring human cognitive processes. We evaluate DID's effectiveness across multiple benchmarks, including the AIW and MR-GSM8K, as well as our custom Holiday Puzzle dataset for temporal reasoning. Our results demonstrate significant improvements in reasoning quality and solution accuracy - achieving 70.3% accuracy on AIW (compared to 62.2% for Tree of Thought) while maintaining lower computational costs. The success of DID in improving LLM performance while preserving computational efficiency suggests promising directions for developing more cognitively aligned and capable language models. Our work contributes a theoretically grounded, input-centric approach to enhancing LLM reasoning capabilities, offering an efficient alternative to traditional output-exploration methods.
>
---
#### [replaced 048] Joint Beamforming and Speaker-Attributed ASR for Real Distant-Microphone Meeting Transcription
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21849v2](http://arxiv.org/pdf/2410.21849v2)**

> **作者:** Can Cui; Imran Ahamad Sheikh; Mostafa Sadeghi; Emmanuel Vincent
>
> **摘要:** Distant-microphone meeting transcription is a challenging task. State-of-the-art end-to-end speaker-attributed automatic speech recognition (SA-ASR) architectures lack a multichannel noise and reverberation reduction front-end, which limits their performance. In this paper, we introduce a joint beamforming and SA-ASR approach for real meeting transcription. We first describe a data alignment and augmentation method to pretrain a neural beamformer on real meeting data. We then compare fixed, hybrid, and fully neural beamformers as front-ends to the SA-ASR model. Finally, we jointly optimize the fully neural beamformer and the SA-ASR model. Experiments on the real AMI corpus show that, while state-of-the-art multi-frame cross-channel attention based channel fusion fails to improve ASR performance, fine-tuning SA-ASR on the fixed beamformer's output and jointly fine-tuning SA-ASR with the neural beamformer reduce the word error rate by 8% and 9% relative, respectively.
>
---
#### [replaced 049] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation
- **分类: cs.CL; cs.LG; eess.SP**

- **链接: [http://arxiv.org/pdf/2403.04945v4](http://arxiv.org/pdf/2403.04945v4)**

> **作者:** Zhongwei Wan; Che Liu; Xin Wang; Chaofan Tao; Hui Shen; Jing Xiong; Rossella Arcucci; Huaxiu Yao; Mi Zhang
>
> **备注:** ACL 2025
>
> **摘要:** Electrocardiogram (ECG) is the primary non-invasive diagnostic tool for monitoring cardiac conditions and is crucial in assisting clinicians. Recent studies have concentrated on classifying cardiac conditions using ECG data but have overlooked ECG report generation, which is time-consuming and requires clinical expertise. To automate ECG report generation and ensure its versatility, we propose the Multimodal ECG Instruction Tuning (MEIT) framework, the first attempt to tackle ECG report generation with LLMs and multimodal instructions. To facilitate future research, we establish a benchmark to evaluate MEIT with various LLMs backbones across two large-scale ECG datasets. Our approach uniquely aligns the representations of the ECG signal and the report, and we conduct extensive experiments to benchmark MEIT with nine open-source LLMs using more than 800,000 ECG reports. MEIT's results underscore the superior performance of instruction-tuned LLMs, showcasing their proficiency in quality report generation, zero-shot capabilities, resilience to signal perturbation, and alignment with human expert evaluation. These findings emphasize the efficacy of MEIT and its potential for real-world clinical application.
>
---
#### [replaced 050] Embedding-Based Approaches to Hyperpartisan News Detection
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.01370v3](http://arxiv.org/pdf/2501.01370v3)**

> **作者:** Karthik Mohan
>
> **备注:** Updated version reflecting sole authorship. All coauthor contributions have been removed. Experimental corrections and analysis updates were introduced in the original version and are retained here as part of the submitter's independent work, along with expanded experiments by the submitter
>
> **摘要:** In this report, I describe the systems in which the objective is to determine whether a given news article could be considered as hyperpartisan. Hyperpartisan news takes an extremely polarized political standpoint with an intention of creating political divide among the public. Several approaches, including n-grams, sentiment analysis, as well as sentence and document representations using pre-tained ELMo models were used. The best system is using LLMs for embedding generation achieving an accuracy of around 92% over the previously best system using pre-trained ELMo with Bidirectional LSTM which achieved an accuracy of around 83% through 10-fold cross-validation.
>
---
#### [replaced 051] Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.15634v3](http://arxiv.org/pdf/2505.15634v3)**

> **作者:** Zihao Li; Xu Wang; Yuzhe Yang; Ziyu Yao; Haoyi Xiong; Mengnan Du
>
> **摘要:** Large Language Models (LLMs) demonstrate the ability to solve reasoning and mathematical problems using the Chain-of-Thought (CoT) technique. Expanding CoT length, as seen in models such as DeepSeek-R1, significantly enhances this reasoning for complex problems, but requires costly and high-quality long CoT data and fine-tuning. This work, inspired by the deep thinking paradigm of DeepSeek-R1, utilizes a steering technique to enhance the reasoning ability of an LLM without external datasets. Our method first employs Sparse Autoencoders (SAEs) to extract interpretable features from vanilla CoT. These features are then used to steer the LLM's internal states during generation. Recognizing that many LLMs do not have corresponding pre-trained SAEs, we further introduce a novel SAE-free steering algorithm, which directly computes steering directions from the residual activations of an LLM, obviating the need for an explicit SAE. Experimental results demonstrate that both our SAE-based and subsequent SAE-free steering algorithms significantly enhance the reasoning capabilities of LLMs.
>
---
#### [replaced 052] Offline Learning and Forgetting for Reasoning with Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.11364v3](http://arxiv.org/pdf/2504.11364v3)**

> **作者:** Tianwei Ni; Allen Nie; Sapana Chaudhary; Yao Liu; Huzefa Rangwala; Rasool Fakoor
>
> **备注:** Code: https://github.com/twni2016/llm-reasoning-uft
>
> **摘要:** Leveraging inference-time search in large language models has proven effective in further enhancing a trained model's capability to solve complex mathematical and reasoning problems. However, this approach significantly increases computational costs and inference time, as the model must generate and evaluate multiple candidate solutions to identify a viable reasoning path. To address this, we propose an effective approach that integrates search capabilities directly into the model by fine-tuning it on unpaired successful (learning) and failed reasoning paths (forgetting) derived from diverse search methods. A key challenge we identify is that naive fine-tuning can degrade the model's search capability; we show this can be mitigated with a smaller learning rate. Extensive experiments on the challenging Game-of-24 and Countdown reasoning benchmarks show that, replacing CoT-generated data with search-generated data for offline fine-tuning improves success rates by around 23% over inference-time search baselines, while reducing inference time by 180$\times$. On top of this, our learning and forgetting objective consistently outperforms both supervised fine-tuning and preference-based methods.
>
---
#### [replaced 053] The distribution of syntactic dependency distances
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2211.14620v2](http://arxiv.org/pdf/2211.14620v2)**

> **作者:** Sonia Petrini; Ramon Ferrer-i-Cancho
>
> **备注:** in press in Glottometrics
>
> **摘要:** The syntactic structure of a sentence can be represented as a graph, where vertices are words and edges indicate syntactic dependencies between them. In this setting, the distance between two linked words is defined as the difference between their positions. Here we wish to contribute to the characterization of the actual distribution of syntactic dependency distances, which has previously been argued to follow a power-law distribution. Here we propose a new model with two exponential regimes in which the probability decay is allowed to change after a break-point. This transition could mirror the transition from the processing of word chunks to higher-level structures. We find that a two-regime model - where the first regime follows either an exponential or a power-law decay - is the most likely one in all 20 languages we considered, independently of sentence length and annotation style. Moreover, the break-point exhibits low variation across languages and averages values of 4-5 words, suggesting that the amount of words that can be simultaneously processed abstracts from the specific language to a high degree. The probability decay slows down after the breakpoint, consistently with a universal chunk-and-pass mechanism. Finally, we give an account of the relation between the best estimated model and the closeness of syntactic dependencies as function of sentence length, according to a recently introduced optimality score.
>
---
#### [replaced 054] SHNU Multilingual Conversational Speech Recognition System for INTERSPEECH 2025 MLC-SLM Challenge
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.03343v2](http://arxiv.org/pdf/2507.03343v2)**

> **作者:** Yuxiang Mei; Yuang Zheng; Dongxing Xu; Yanhua Long
>
> **备注:** Accepted by Interspeech 2025 MLC-SLM workshop
>
> **摘要:** This paper describes SHNU multilingual conversational speech recognition system (SHNU-mASR, team name-"maybe"), submitted to Track 1 of the INTERSPEECH 2025 MLC-SLM Challenge. Our system integrates a parallel-speech-encoder architecture with a large language model (LLM) to form a unified multilingual ASR framework. The parallel-speech-encoder consists of two pre-trained encoders, the Whisper-large-v3 encoder and mHuBERT-147 encoder. Their output embeddings are concatenated and fed into the LLM, enabling the model to leverage complementary acoustic and linguistic knowledge and achieve competitive performance. Moreover, we adopt a tri-stage training strategy to jointly update the low-rank adaptation modules and projector parameters of both the speech encoders and the LLM. In addition, we incorporate an additional language-aware prompt at the LLM input to enhance language-specific text generation. The SHNU-mASR system achieves an overall character/word error rate (CER/WER) of 11.76% on the blind evaluation set of the challenge, outperforming the official MLC-SLM baseline by 8.41 absolute CER/WER, without increasing the baseline training data.
>
---
#### [replaced 055] RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2507.02962v2](http://arxiv.org/pdf/2507.02962v2)**

> **作者:** Zhiwen Tan; Jiaming Huang; Qintong Wu; Hongxuan Zhang; Chenyi Zhuang; Jinjie Gu
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
>
---
#### [replaced 056] Seeing Sarcasm Through Different Eyes: Analyzing Multimodal Sarcasm Perception in Large Vision-Language Models
- **分类: cs.CL; cs.MM; cs.SI**

- **链接: [http://arxiv.org/pdf/2503.12149v2](http://arxiv.org/pdf/2503.12149v2)**

> **作者:** Junjie Chen; Xuyang Liu; Subin Huang; Linfeng Zhang; Hang Yu
>
> **摘要:** With the advent of large vision-language models (LVLMs) demonstrating increasingly human-like abilities, a pivotal question emerges: do different LVLMs interpret multimodal sarcasm differently, and can a single model grasp sarcasm from multiple perspectives like humans? To explore this, we introduce an analytical framework using systematically designed prompts on existing multimodal sarcasm datasets. Evaluating 12 state-of-the-art LVLMs over 2,409 samples, we examine interpretive variations within and across models, focusing on confidence levels, alignment with dataset labels, and recognition of ambiguous "neutral" cases. Our findings reveal notable discrepancies -- across LVLMs and within the same model under varied prompts. While classification-oriented prompts yield higher internal consistency, models diverge markedly when tasked with interpretive reasoning. These results challenge binary labeling paradigms by highlighting sarcasm's subjectivity. We advocate moving beyond rigid annotation schemes toward multi-perspective, uncertainty-aware modeling, offering deeper insights into multimodal sarcasm comprehension. Our code and data are available at: https://github.com/CoderChen01/LVLMSarcasmAnalysis
>
---
#### [replaced 057] Self-supervised learning of speech representations with Dutch archival data
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.04554v2](http://arxiv.org/pdf/2507.04554v2)**

> **作者:** Nik Vaessen; Roeland Ordelman; David A. van Leeuwen
>
> **备注:** accepted at interspeech 2025
>
> **摘要:** This paper explores the use of Dutch archival television broadcast data for self-supervised learning of speech foundation models, specifically wav2vec 2.0. We first study data quality assumptions for pre-training, and show how music, noise and speaker overlap affect SSL convergence and downstream fine-tuning performance. Secondly, we explore effectively pre-processing strategies to convert the noisy broadcast dataset into a qualitative dataset for pre-training, by using Whisper and WhisperX. Thirdly, we compare mono-lingual and multi-lingual pre-training with equivalent amounts of data, and show that mono-lingual pre-training is more robust to out-of-domain data. Lastly, we achieve a state-of-the-art LARGE wav2vec 2.0 model for the Dutch language, by a continuation of pre-training a wav2vec 2.0 XLS-R model checkpoint with our 55k hour archival dataset.
>
---
#### [replaced 058] Tractable Transformers for Flexible Conditional Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07616v2](http://arxiv.org/pdf/2502.07616v2)**

> **作者:** Anji Liu; Xuejie Liu; Dayuan Zhao; Mathias Niepert; Yitao Liang; Guy Van den Broeck
>
> **摘要:** Non-autoregressive (NAR) generative models are valuable because they can handle diverse conditional generation tasks in a more principled way than their autoregressive (AR) counterparts, which are constrained by sequential dependency requirements. Recent advancements in NAR models, such as diffusion language models, have demonstrated superior performance in unconditional generation compared to AR models (e.g., GPTs) of similar sizes. However, such improvements do not always lead to improved conditional generation performance. We show that a key reason for this gap is the difficulty in generalizing to conditional probability queries (i.e., the set of unknown variables) unseen during training. As a result, strong unconditional generation performance does not guarantee high-quality conditional generation. This paper proposes Tractable Transformers (Tracformer), a Transformer-based generative model that is more robust to different conditional generation tasks. Unlike existing models that rely solely on global contextual features derived from full inputs, Tracformers incorporate a sparse Transformer encoder to capture both local and global contextual information. This information is routed through a decoder for conditional generation. Empirical results demonstrate that Tracformers achieve state-of-the-art conditional generation performance on text modeling compared to recent diffusion and AR model baselines.
>
---
#### [replaced 059] Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16327v2](http://arxiv.org/pdf/2410.16327v2)**

> **作者:** Rui Pu; Chaozhuo Li; Rui Ha; Zejian Chen; Litian Zhang; Zheng Liu; Lirong Qiu; Zaisheng Ye
>
> **摘要:** Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.
>
---
#### [replaced 060] Detecting value-expressive text posts in Russian social media
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2312.08968v2](http://arxiv.org/pdf/2312.08968v2)**

> **作者:** Maria Milkova; Maksim Rudnev; Lidia Okolskaya
>
> **摘要:** Basic values are concepts or beliefs which pertain to desirable end-states and transcend specific situations. Studying personal values in social media can illuminate how and why societal values evolve especially when the stimuli-based methods, such as surveys, are inefficient, for instance, in hard-to-reach populations. On the other hand, user-generated content is driven by the massive use of stereotyped, culturally defined speech constructions rather than authentic expressions of personal values. We aimed to find a model that can accurately detect value-expressive posts in Russian social media VKontakte. A training dataset of 5,035 posts was annotated by three experts, 304 crowd-workers and ChatGPT. Crowd-workers and experts showed only moderate agreement in categorizing posts. ChatGPT was more consistent but struggled with spam detection. We applied an ensemble of human- and AI-assisted annotation involving active learning approach, subsequently trained several classification models using embeddings from various pre-trained transformer-based language models. The best performance was achieved with embeddings from a fine-tuned rubert-tiny2 model, yielding high value detection quality (F1 = 0.75, F1-macro = 0.80). This model provides a crucial step to a study of values within and between Russian social media users.
>
---
#### [replaced 061] LLM Hypnosis: Exploiting User Feedback for Unauthorized Knowledge Injection to All Users
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02850v2](http://arxiv.org/pdf/2507.02850v2)**

> **作者:** Almog Hilel; Idan Shenfeld; Jacob Andreas; Leshem Choshen
>
> **摘要:** We describe a vulnerability in language models (LMs) trained with user feedback, whereby a single user can persistently alter LM knowledge and behavior given only the ability to provide prompts and upvote / downvote feedback on LM outputs. To implement the attack, the attacker prompts the LM to stochastically output either a "poisoned" or benign response, then upvotes the poisoned response or downvotes the benign one. When feedback signals are used in a subsequent preference tuning behavior, LMs exhibit increased probability of producing poisoned responses even in contexts without malicious prompts. We show that this attack can be used to (1) insert factual knowledge the model did not previously possess, (2) modify code generation patterns in ways that introduce exploitable security flaws, and (3) inject fake financial news. Our finding both identifies a new qualitative feature of language model preference tuning (showing that it even highly restricted forms of preference data can be used to exert fine-grained control over behavior), and a new attack mechanism for LMs trained with user feedback (extending work on pretraining-time data poisoning and deployment-time prompt injection).
>
---
#### [replaced 062] GAF-Guard: An Agentic Framework for Risk Management and Governance in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.02986v2](http://arxiv.org/pdf/2507.02986v2)**

> **作者:** Seshu Tirupathi; Dhaval Salwala; Elizabeth Daly; Inge Vejsbjerg
>
> **摘要:** As Large Language Models (LLMs) continue to be increasingly applied across various domains, their widespread adoption necessitates rigorous monitoring to prevent unintended negative consequences and ensure robustness. Furthermore, LLMs must be designed to align with human values, like preventing harmful content and ensuring responsible usage. The current automated systems and solutions for monitoring LLMs in production are primarily centered on LLM-specific concerns like hallucination etc, with little consideration given to the requirements of specific use-cases and user preferences. This paper introduces GAF-Guard, a novel agentic framework for LLM governance that places the user, the use-case, and the model itself at the center. The framework is designed to detect and monitor risks associated with the deployment of LLM based applications. The approach models autonomous agents that identify risks, activate risk detection tools, within specific use-cases and facilitate continuous monitoring and reporting to enhance AI safety, and user expectations. The code is available at https://github.com/IBM/risk-atlas-nexus-demos/tree/main/gaf-guard.
>
---
#### [replaced 063] Bayesian Optimization for Controlled Image Editing via LLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18116v3](http://arxiv.org/pdf/2502.18116v3)**

> **作者:** Chengkun Cai; Haoliang Liu; Xu Zhao; Zhongyu Jiang; Tianfang Zhang; Zongkai Wu; John Lee; Jenq-Neng Hwang; Lei Li
>
> **备注:** 8 figures, accept at ACL2025 Findings
>
> **摘要:** In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
>
---
#### [replaced 064] MedGemma Technical Report
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.05201v2](http://arxiv.org/pdf/2507.05201v2)**

> **作者:** Andrew Sellergren; Sahar Kazemzadeh; Tiam Jaroensri; Atilla Kiraly; Madeleine Traverse; Timo Kohlberger; Shawn Xu; Fayaz Jamil; Cían Hughes; Charles Lau; Justin Chen; Fereshteh Mahvar; Liron Yatziv; Tiffany Chen; Bram Sterling; Stefanie Anna Baby; Susanna Maria Baby; Jeremy Lai; Samuel Schmidgall; Lu Yang; Kejia Chen; Per Bjornsson; Shashir Reddy; Ryan Brush; Kenneth Philbrick; Howard Hu; Howard Yang; Richa Tiwari; Sunny Jansen; Preeti Singh; Yun Liu; Shekoofeh Azizi; Aishwarya Kamath; Johan Ferret; Shreya Pathak; Nino Vieillard; Ramona Merhej; Sarah Perrin; Tatiana Matejovicova; Alexandre Ramé; Morgane Riviere; Louis Rouillard; Thomas Mesnard; Geoffrey Cideron; Jean-bastien Grill; Sabela Ramos; Edouard Yvinec; Michelle Casbon; Elena Buchatskaya; Jean-Baptiste Alayrac; Dmitry Lepikhin; Vlad Feinberg; Sebastian Borgeaud; Alek Andreev; Cassidy Hardin; Robert Dadashi; Léonard Hussenot; Armand Joulin; Olivier Bachem; Yossi Matias; Katherine Chou; Avinatan Hassidim; Kavi Goel; Clement Farabet; Joelle Barral; Tris Warkentin; Jonathon Shlens; David Fleet; Victor Cotruta; Omar Sanseviero; Gus Martins; Phoebe Kirk; Anand Rao; Shravya Shetty; David F. Steiner; Can Kirmizibayrak; Rory Pilgrim; Daniel Golden; Lin Yang
>
> **摘要:** Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at https://goo.gle/medgemma.
>
---
#### [replaced 065] Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05006v2](http://arxiv.org/pdf/2507.05006v2)**

> **作者:** Matteo Attimonelli; Alessandro De Bellis; Claudio Pomo; Dietmar Jannach; Eugenio Di Sciascio; Tommaso Di Noia
>
> **备注:** Accept as Short Paper at RecSys 2025
>
> **摘要:** Pre-trained language models (PLMs) are widely used to derive semantic representations from item metadata in recommendation and search. In sequential recommendation, PLMs enhance ID-based embeddings through textual metadata, while in product search, they align item characteristics with user intent. Recent studies suggest task and domain-specific fine-tuning are needed to improve representational power. This paper challenges this assumption, showing that Generalist Text Embedding Models (GTEs), pre-trained on large-scale corpora, can guarantee strong zero-shot performance without specialized adaptation. Our experiments demonstrate that GTEs outperform traditional and fine-tuned models in both sequential recommendation and product search. We attribute this to a superior representational power, as they distribute features more evenly across the embedding space. Finally, we show that compressing embedding dimensions by focusing on the most informative directions (e.g., via PCA) effectively reduces noise and improves the performance of specialized models. To ensure reproducibility, we provide our repository at https://split.to/gte4ps.
>
---
#### [replaced 066] PDFMathTranslate: Scientific Document Translation Preserving Layouts
- **分类: cs.CL; cs.IR; cs.LG; 68T50, 68T45, 68U10, 68U15; D.2.2; I.2.10; I.2.7; J.0**

- **链接: [http://arxiv.org/pdf/2507.03009v2](http://arxiv.org/pdf/2507.03009v2)**

> **作者:** Rongxin Ouyang; Chang Chu; Zhikuang Xin; Xiangyao Ma
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Language barriers in scientific documents hinder the diffusion and development of science and technologies. However, prior efforts in translating such documents largely overlooked the information in layouts. To bridge the gap, we introduce PDFMathTranslate, the world's first open-source software for translating scientific documents while preserving layouts. Leveraging the most recent advances in large language models and precise layout detection, we contribute to the community with key improvements in precision, flexibility, and efficiency. The work has been open-sourced at https://github.com/byaidu/pdfmathtranslate with more than 222k downloads.
>
---
